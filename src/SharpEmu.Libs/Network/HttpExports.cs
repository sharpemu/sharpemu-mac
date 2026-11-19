// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Network;

/// <summary>
/// libSceHttp in "offline" mode. The handle hierarchy (context, template, connection, request)
/// is tracked faithfully so that init-before-use, bad ids and double deletes behave like the
/// SDK, but no request ever reaches a network: sending fails as on a console whose host name never
/// resolves (SCE_HTTP_ERROR_RESOLVER_ENODNS), and every response query reports that no response
/// exists (SCE_HTTP_ERROR_BEFORE_SEND) with zeroed outputs.
/// </summary>
/// <remarks>
/// Error codes come from the two references that agree (BEFORE_INIT, OUT_OF_MEMORY, ABORTED,
/// INVALID_ID, INVALID_VALUE, INVALID_URL); the offline send and response codes (RESOLVER_ENODNS,
/// BEFORE_SEND) are upstream SharpEmu's. Arguments the SDK documents as invalid
/// but whose dedicated code no reference pins (an unknown scheme, a bad method) report
/// SCE_HTTP_ERROR_INVALID_VALUE, the generic documented code.
/// </remarks>
public static partial class HttpExports
{
    private const int HttpErrorBeforeInit = unchecked((int)0x80431001);
    private const int HttpErrorBeforeSend = unchecked((int)0x80431065);
    private const int HttpErrorResolverNoDns = unchecked((int)0x80436002);
    private const int HttpErrorAborted = unchecked((int)0x80431080);
    private const int HttpErrorInvalidId = unchecked((int)0x80431100);
    private const int HttpErrorInvalidValue = unchecked((int)0x804311FE);
    private const int MaxMethod = 7;
    private const int MaxStringLength = 16 * 1024;

    private static readonly object Gate = new();
    private static readonly Dictionary<int, HttpContext> Contexts = new();
    private static readonly Dictionary<int, HttpNode> Nodes = new();
    private static readonly Dictionary<ulong, HttpEpoll> Epolls = new();
    private static int _nextContextId;
    private static int _nextNodeId = 0x1000;
    private static ulong _nextEpollHandle = 0x7E00_0000;

    private enum NodeKind
    {
        Template,
        Connection,
        Request,
    }

    private sealed class HttpContext(int netMemoryId, int sslContextId, ulong poolSize)
    {
        public int NetMemoryId { get; } = netMemoryId;

        public int SslContextId { get; } = sslContextId;

        public ulong PoolSize { get; } = poolSize;

        public int LoadedCaCount { get; set; }
    }

    private sealed class HttpNode(NodeKind kind, int id, int contextId, int parentId)
    {
        public NodeKind Kind { get; } = kind;

        public int Id { get; } = id;

        public int ContextId { get; } = contextId;

        public int ParentId { get; } = parentId;

        public Dictionary<string, ulong> Settings { get; } = new(StringComparer.Ordinal);

        public List<(string Name, string Value)> Headers { get; } = [];

        public ulong EpollHandle { get; set; }

        public bool Sent { get; set; }

        public bool Aborted { get; set; }
    }

    private sealed class HttpEpoll(int contextId)
    {
        public int ContextId { get; } = contextId;

        public bool Aborted { get; set; }

        public bool Destroyed { get; set; }
    }

    internal static void ResetRuntimeState()
    {
        lock (Gate)
        {
            Contexts.Clear();
            Nodes.Clear();
            Epolls.Clear();
            _nextContextId = 0;
            _nextNodeId = 0x1000;
            _nextEpollHandle = 0x7E00_0000;
        }
    }

    internal static int LiveContextCount
    {
        get
        {
            lock (Gate)
            {
                return Contexts.Count;
            }
        }
    }

    internal static int LiveHandleCount
    {
        get
        {
            lock (Gate)
            {
                return Nodes.Count;
            }
        }
    }

    [SysAbiExport(
        Nid = "A9cVMUtEp4Y",
        ExportName = "sceHttpInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpInit(CpuContext ctx)
    {
        var netMemoryId = unchecked((int)ctx[CpuRegister.Rdi]);
        var sslContextId = unchecked((int)ctx[CpuRegister.Rsi]);
        var poolSize = ctx[CpuRegister.Rdx];
        if (poolSize == 0)
        {
            return ctx.SetReturn(HttpErrorInvalidValue);
        }

        int id;
        lock (Gate)
        {
            id = ++_nextContextId;
            Contexts[id] = new HttpContext(netMemoryId, sslContextId, poolSize);
        }

        TraceHttp("init", id, unchecked((ulong)netMemoryId), unchecked((ulong)sslContextId), poolSize, 0);
        return ReturnId(ctx, id);
    }

    [SysAbiExport(
        Nid = "Ik-KpLTlf7Q",
        ExportName = "sceHttpTerm",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpTerm(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        List<HttpEpoll> released = [];
        lock (Gate)
        {
            var error = CheckContext(contextId, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            Contexts.Remove(contextId);
            foreach (var pair in Nodes.Where(pair => pair.Value.ContextId == contextId).ToArray())
            {
                Nodes.Remove(pair.Key);
            }

            foreach (var pair in Epolls.Where(pair => pair.Value.ContextId == contextId).ToArray())
            {
                Epolls.Remove(pair.Key);
                released.Add(pair.Value);
            }
        }

        foreach (var epoll in released)
        {
            MarkEpollDestroyed(epoll);
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "0gYjPTR-6cY",
        ExportName = "sceHttpCreateTemplate",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateTemplate(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        var userAgentAddress = ctx[CpuRegister.Rsi];
        var httpVersion = unchecked((int)ctx[CpuRegister.Rdx]);
        var autoProxyConfig = unchecked((int)ctx[CpuRegister.Rcx]);
        lock (Gate)
        {
            var error = CheckContext(contextId, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            // A null user agent is accepted (as upstream does); nothing is ever sent, so it is never read.
            if (httpVersion is < 0 or > 2 || autoProxyConfig is < 0 or > 1)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            var node = AddNode(NodeKind.Template, contextId, 0);
            node.Settings["httpversion"] = unchecked((ulong)httpVersion);
            node.Settings["autoproxy"] = unchecked((ulong)autoProxyConfig);
            TraceHttp("create_template", node.Id, unchecked((ulong)contextId), userAgentAddress, unchecked((ulong)httpVersion), unchecked((ulong)autoProxyConfig));
            return ReturnId(ctx, node.Id);
        }
    }

    [SysAbiExport(
        Nid = "4I8vEpuEhZ8",
        ExportName = "sceHttpDeleteTemplate",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpDeleteTemplate(CpuContext ctx) => DeleteNode(ctx, NodeKind.Template);

    [SysAbiExport(
        Nid = "Kiwv9r4IZCc",
        ExportName = "sceHttpCreateConnection",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateConnection(CpuContext ctx)
    {
        var templateId = unchecked((int)ctx[CpuRegister.Rdi]);
        var keepAlive = unchecked((int)ctx[CpuRegister.R8]);
        lock (Gate)
        {
            var error = CheckNode(templateId, NodeKind.Template, out var template);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (!TryReadCString(ctx, ctx[CpuRegister.Rsi], out var serverName) || serverName.Length == 0 ||
                !TryReadCString(ctx, ctx[CpuRegister.Rdx], out var scheme) || !IsHttpScheme(scheme) ||
                keepAlive is < 0 or > 1)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            var node = AddNode(NodeKind.Connection, template.ContextId, templateId);
            node.Settings["port"] = ctx[CpuRegister.Rcx] & 0xFFFF;
            return ReturnId(ctx, node.Id);
        }
    }

    [SysAbiExport(
        Nid = "qgxDBjorUxs",
        ExportName = "sceHttpCreateConnectionWithURL",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateConnectionWithUrl(CpuContext ctx)
    {
        var templateId = unchecked((int)ctx[CpuRegister.Rdi]);
        var keepAlive = unchecked((int)ctx[CpuRegister.Rdx]);
        lock (Gate)
        {
            var error = CheckNode(templateId, NodeKind.Template, out var template);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (!TryReadCString(ctx, ctx[CpuRegister.Rsi], out var url) || url.Length == 0 || keepAlive is < 0 or > 1)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            if (!IsHttpUrl(url))
            {
                return ctx.SetReturn(HttpErrorInvalidUrl);
            }

            var node = AddNode(NodeKind.Connection, template.ContextId, templateId);
            return ReturnId(ctx, node.Id);
        }
    }

    [SysAbiExport(
        Nid = "P6A3ytpsiYc",
        ExportName = "sceHttpDeleteConnection",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpDeleteConnection(CpuContext ctx) => DeleteNode(ctx, NodeKind.Connection);

    // (connectionId, method, path, contentLength)
    [SysAbiExport(
        Nid = "tsGVru3hCe8",
        ExportName = "sceHttpCreateRequest",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateRequest(CpuContext ctx) => CreateRequest(ctx, methodIsString: false, targetIsUrl: false);

    // (connectionId, method, url, contentLength)
    [SysAbiExport(
        Nid = "Aeu5wVKkF9w",
        ExportName = "sceHttpCreateRequestWithURL",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateRequestWithUrl(CpuContext ctx) => CreateRequest(ctx, methodIsString: false, targetIsUrl: true);

    // (connectionId, "METHOD", path, contentLength)
    [SysAbiExport(
        Nid = "rGNm+FjIXKk",
        ExportName = "sceHttpCreateRequest2",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateRequest2(CpuContext ctx) => CreateRequest(ctx, methodIsString: true, targetIsUrl: false);

    // (connectionId, "METHOD", url, contentLength)
    [SysAbiExport(
        Nid = "Cnp77podkCU",
        ExportName = "sceHttpCreateRequestWithURL2",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateRequestWithUrl2(CpuContext ctx) => CreateRequest(ctx, methodIsString: true, targetIsUrl: true);

    [SysAbiExport(
        Nid = "qe7oZ+v4PWA",
        ExportName = "sceHttpDeleteRequest",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpDeleteRequest(CpuContext ctx) => DeleteNode(ctx, NodeKind.Request);

    // (requestId, postData, size). Offline: the request is marked sent and fails because the host
    // name never resolves.
    [SysAbiExport(
        Nid = "1e2BNwI-XzE",
        ExportName = "sceHttpSendRequest",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpSendRequest(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(requestId, NodeKind.Request, out var request);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (ctx[CpuRegister.Rsi] == 0 && ctx[CpuRegister.Rdx] != 0)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            if (request.Aborted)
            {
                return ctx.SetReturn(HttpErrorAborted);
            }

            request.Sent = true;
            TraceHttp("send_request_no_network", requestId, ctx[CpuRegister.Rsi], ctx[CpuRegister.Rdx], 0, 0);
            return ctx.SetReturn(HttpErrorResolverNoDns);
        }
    }

    [SysAbiExport(
        Nid = "hvG6GfBMXg8",
        ExportName = "sceHttpAbortRequest",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpAbortRequest(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(requestId, NodeKind.Request, out var request);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            request.Aborted = true;
            return ctx.SetReturn(0);
        }
    }

    // (requestId, int* statusCode). No response exists offline.
    [SysAbiExport(
        Nid = "0a2TBNfE3BU",
        ExportName = "sceHttpGetStatusCode",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpGetStatusCode(CpuContext ctx) => ResponseQuery(ctx, [(CpuRegister.Rsi, sizeof(int))]);

    // (requestId, int* result, uint64* contentLength)
    [SysAbiExport(
        Nid = "yuO2H2Uvnos",
        ExportName = "sceHttpGetResponseContentLength",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpGetResponseContentLength(CpuContext ctx) =>
        ResponseQuery(ctx, [(CpuRegister.Rsi, sizeof(int)), (CpuRegister.Rdx, sizeof(ulong))]);

    // (requestId, char** headers, size_t* headerSize)
    [SysAbiExport(
        Nid = "aCYPMSUIaP8",
        ExportName = "sceHttpGetAllResponseHeaders",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpGetAllResponseHeaders(CpuContext ctx) =>
        ResponseQuery(ctx, [(CpuRegister.Rsi, sizeof(ulong)), (CpuRegister.Rdx, sizeof(ulong))]);

    // (requestId, void* buffer, size_t size)
    [SysAbiExport(
        Nid = "P5pdoykPYTk",
        ExportName = "sceHttpReadData",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpReadData(CpuContext ctx) => ResponseQuery(ctx, []);

    // (requestId, int* errorNumber)
    [SysAbiExport(
        Nid = "0onIrKx9NIE",
        ExportName = "sceHttpGetLastErrno",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpGetLastErrno(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(requestId, NodeKind.Request, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }
        }

        // No reference documents the errno of an offline failure; 0 means "no OS error".
        return WriteZeros(ctx, ctx[CpuRegister.Rsi], sizeof(int))
            ? ctx.SetReturn(0)
            : ctx.SetReturn(HttpErrorInvalidValue);
    }

    // (id, int* errorNumber, uint32* detail)
    [SysAbiExport(
        Nid = "JBN6N-EY+3M",
        ExportName = "sceHttpsGetSslError",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpsGetSslError(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(id, null, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }
        }

        if (ctx[CpuRegister.Rsi] == 0 || ctx[CpuRegister.Rdx] == 0)
        {
            return ctx.SetReturn(HttpErrorInvalidValue);
        }

        // No TLS session ever starts offline, hence no SSL error.
        return WriteZeros(ctx, ctx[CpuRegister.Rsi], sizeof(int)) && WriteZeros(ctx, ctx[CpuRegister.Rdx], sizeof(uint))
            ? ctx.SetReturn(0)
            : ctx.SetReturn(HttpErrorInvalidValue);
    }

    // (requestId, name, value, mode): mode 0 overwrites, 1 appends. Works on templates,
    // connections and requests.
    [SysAbiExport(
        Nid = "EY28T2bkN7k",
        ExportName = "sceHttpAddRequestHeader",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpAddRequestHeader(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        var mode = unchecked((uint)ctx[CpuRegister.Rcx]);
        lock (Gate)
        {
            var error = CheckNode(id, null, out var node);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (ctx[CpuRegister.Rsi] == 0 || ctx[CpuRegister.Rdx] == 0 || mode > 1 ||
                !TryReadCString(ctx, ctx[CpuRegister.Rsi], out var name) || name.Length == 0 ||
                !TryReadCString(ctx, ctx[CpuRegister.Rdx], out var value))
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            if (mode == 0)
            {
                node.Headers.RemoveAll(header => string.Equals(header.Name, name, StringComparison.OrdinalIgnoreCase));
            }

            node.Headers.Add((name, value));
            return ctx.SetReturn(0);
        }
    }

    [SysAbiExport(
        Nid = "zNGh-zoQTD0",
        ExportName = "sceHttpRemoveRequestHeader",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpRemoveRequestHeader(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(id, null, out var node);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (ctx[CpuRegister.Rsi] == 0 || !TryReadCString(ctx, ctx[CpuRegister.Rsi], out var name) || name.Length == 0)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            node.Headers.RemoveAll(header => string.Equals(header.Name, name, StringComparison.OrdinalIgnoreCase));
            return ctx.SetReturn(0);
        }
    }

    // (requestId, uint64 contentLength)
    [SysAbiExport(
        Nid = "PTiFIUxCpJc",
        ExportName = "sceHttpSetRequestContentLength",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpSetRequestContentLength(CpuContext ctx) => SetSetting(ctx, NodeKind.Request, "contentlength", ctx[CpuRegister.Rsi]);

    // Registers CA certificates on an HTTP context; no host TLS session uses them. The
    // certificates themselves are not retained, only how many were loaded.
    // (contextId, caCertNum, caList**, clientCert*, privateKey*)
    [SysAbiExport(
        Nid = "DK+GoXCNT04",
        ExportName = "sceHttpsLoadCert",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpsLoadCert(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        var caCount = unchecked((int)ctx[CpuRegister.Rsi]);
        lock (Gate)
        {
            var error = CheckContext(contextId, out var context);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (caCount < 0 || (caCount > 0 && ctx[CpuRegister.Rdx] == 0) ||
                ((ctx[CpuRegister.Rcx] == 0) != (ctx[CpuRegister.R8] == 0)))
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            context.LoadedCaCount = checked(context.LoadedCaCount + caCount);
            TraceHttp("https_load_cert", contextId, ctx[CpuRegister.Rsi], ctx[CpuRegister.Rdx], ctx[CpuRegister.Rcx], ctx[CpuRegister.R8]);
            return ctx.SetReturn(0);
        }
    }

    [SysAbiExport(
        Nid = "zXqcE0fizz0",
        ExportName = "sceHttpsUnloadCert",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpsUnloadCert(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckContext(contextId, out var context);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            context.LoadedCaCount = 0;
            return ctx.SetReturn(0);
        }
    }

    // (contextId, SceHttpsCaList* { void* caList; int caNum }): certificates are not retained,
    // so the list is always empty.
    [SysAbiExport(
        Nid = "gcUjwU3fa0M",
        ExportName = "sceHttpsGetCaList",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpsGetCaList(CpuContext ctx) => ClearCaList(ctx);

    [SysAbiExport(
        Nid = "7WcNoAI9Zcw",
        ExportName = "sceHttpsFreeCaList",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpsFreeCaList(CpuContext ctx) => ClearCaList(ctx);

    [SysAbiExport(
        Nid = "u05NnI+P+KY",
        ExportName = "sceHttpRedirectCacheFlush",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpRedirectCacheFlush(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            return ctx.SetReturn(CheckContext(contextId, out _));
        }
    }

    // (contextId, HttpEpollHandle* handle)
    [SysAbiExport(
        Nid = "6381dWF+xsQ",
        ExportName = "sceHttpCreateEpoll",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpCreateEpoll(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        var handleAddress = ctx[CpuRegister.Rsi];
        lock (Gate)
        {
            var error = CheckContext(contextId, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (handleAddress == 0)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            var handle = ++_nextEpollHandle;
            Span<byte> bytes = stackalloc byte[sizeof(ulong)];
            BinaryPrimitives.WriteUInt64LittleEndian(bytes, handle);
            if (!ctx.Memory.TryWrite(handleAddress, bytes))
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            Epolls[handle] = new HttpEpoll(contextId);
            return ctx.SetReturn(0);
        }
    }

    // (contextId, HttpEpollHandle handle)
    [SysAbiExport(
        Nid = "wYhXVfS2Et4",
        ExportName = "sceHttpDestroyEpoll",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpDestroyEpoll(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        var handle = ctx[CpuRegister.Rsi];
        HttpEpoll? epoll;
        lock (Gate)
        {
            var error = CheckContext(contextId, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (handle == 0)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            if (!Epolls.Remove(handle, out epoll))
            {
                return ctx.SetReturn(HttpErrorInvalidId);
            }

            foreach (var node in Nodes.Values.Where(node => node.EpollHandle == handle))
            {
                node.EpollHandle = 0;
            }
        }

        MarkEpollDestroyed(epoll);
        return ctx.SetReturn(0);
    }

    // (id, HttpEpollHandle handle, void* userArg)
    [SysAbiExport(
        Nid = "-xm7kZQNpHI",
        ExportName = "sceHttpSetEpoll",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpSetEpoll(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        var handle = ctx[CpuRegister.Rsi];
        lock (Gate)
        {
            var error = CheckNode(id, null, out var node);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            if (handle == 0)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            if (!Epolls.ContainsKey(handle))
            {
                return ctx.SetReturn(HttpErrorInvalidId);
            }

            node.EpollHandle = handle;
            node.Settings["epoll.arg"] = ctx[CpuRegister.Rdx];
            return ctx.SetReturn(0);
        }
    }

    [SysAbiExport(
        Nid = "59tL1AQBb8U",
        ExportName = "sceHttpUnsetEpoll",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpUnsetEpoll(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(id, null, out var node);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            node.EpollHandle = 0;
            node.Settings.Remove("epoll.arg");
            return ctx.SetReturn(0);
        }
    }

    // (HttpEpollHandle handle, HttpNBEvent* events, int maxEvents, int timeoutMicroseconds).
    // No request can ever be pending offline, so the wait only ends by timeout (0 events),
    // abort or destruction. A negative timeout waits without limit.
    [SysAbiExport(
        Nid = "qISjDHrxONc",
        ExportName = "sceHttpWaitRequest",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpWaitRequest(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var maxEvents = unchecked((int)ctx[CpuRegister.Rdx]);
        var timeoutMicroseconds = unchecked((int)ctx[CpuRegister.Rcx]);
        HttpEpoll? epoll;
        lock (Gate)
        {
            if (Contexts.Count == 0)
            {
                return ctx.SetReturn(HttpErrorBeforeInit);
            }

            if (handle == 0 || ctx[CpuRegister.Rsi] == 0 || maxEvents <= 0)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            if (!Epolls.TryGetValue(handle, out epoll))
            {
                return ctx.SetReturn(HttpErrorInvalidId);
            }
        }

        lock (epoll)
        {
            var deadline = timeoutMicroseconds < 0
                ? (DateTime?)null
                : DateTime.UtcNow.AddTicks((long)timeoutMicroseconds * TimeSpan.TicksPerMicrosecond);
            while (!epoll.Aborted && !epoll.Destroyed)
            {
                if (deadline is null)
                {
                    Monitor.Wait(epoll);
                    continue;
                }

                var remaining = deadline.Value - DateTime.UtcNow;
                if (remaining <= TimeSpan.Zero || !Monitor.Wait(epoll, remaining))
                {
                    break;
                }
            }

            return ctx.SetReturn(epoll.Aborted ? HttpErrorAborted : 0);
        }
    }

    // An abort sticks: every later wait on the handle reports ABORTED.
    [SysAbiExport(
        Nid = "sWQiqKvYTVA",
        ExportName = "sceHttpAbortWaitRequest",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceHttp")]
    public static int HttpAbortWaitRequest(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        HttpEpoll? epoll;
        lock (Gate)
        {
            if (Contexts.Count == 0)
            {
                return ctx.SetReturn(HttpErrorBeforeInit);
            }

            if (handle == 0)
            {
                return ctx.SetReturn(HttpErrorInvalidValue);
            }

            if (!Epolls.TryGetValue(handle, out epoll))
            {
                return ctx.SetReturn(HttpErrorInvalidId);
            }
        }

        lock (epoll)
        {
            epoll.Aborted = true;
            Monitor.PulseAll(epoll);
        }

        return ctx.SetReturn(0);
    }

    // The next four share one id space with templates and requests. Callers hold Gate.
    private static int CheckContext(int contextId, out HttpContext context)
    {
        context = null!;
        if (Contexts.Count == 0)
        {
            return HttpErrorBeforeInit;
        }

        return Contexts.TryGetValue(contextId, out context!) ? 0 : HttpErrorInvalidId;
    }

    private static int CheckNode(int id, NodeKind? kind, out HttpNode node)
    {
        node = null!;
        if (Contexts.Count == 0)
        {
            return HttpErrorBeforeInit;
        }

        if (!Nodes.TryGetValue(id, out node!) || (kind is not null && node.Kind != kind))
        {
            return HttpErrorInvalidId;
        }

        return 0;
    }

    private static HttpNode AddNode(NodeKind kind, int contextId, int parentId)
    {
        var id = ++_nextNodeId;
        var node = new HttpNode(kind, id, contextId, parentId);
        Nodes[id] = node;
        return node;
    }

    private static void RemoveNodeTree(int id)
    {
        Nodes.Remove(id);
        foreach (var child in Nodes.Values.Where(node => node.ParentId == id).ToArray())
        {
            RemoveNodeTree(child.Id);
        }
    }

    private static int DeleteNode(CpuContext ctx, NodeKind kind)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(id, kind, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            // Deleting a parent takes its connections and requests with it.
            RemoveNodeTree(id);
            return ctx.SetReturn(0);
        }
    }

    private static int CreateRequest(CpuContext ctx, bool methodIsString, bool targetIsUrl)
    {
        var connectionId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(connectionId, NodeKind.Connection, out var connection);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }

            string method;
            if (methodIsString)
            {
                if (!TryReadCString(ctx, ctx[CpuRegister.Rsi], out method) || method.Length == 0)
                {
                    return ctx.SetReturn(HttpErrorInvalidValue);
                }
            }
            else
            {
                var methodId = unchecked((int)ctx[CpuRegister.Rsi]);
                if (methodId is < 0 or > MaxMethod)
                {
                    return ctx.SetReturn(HttpErrorInvalidValue);
                }

                method = methodId.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }

            if (ctx[CpuRegister.Rdx] == 0 || !TryReadCString(ctx, ctx[CpuRegister.Rdx], out var target) || target.Length == 0)
            {
                return ctx.SetReturn(targetIsUrl ? HttpErrorInvalidUrl : HttpErrorInvalidValue);
            }

            if (targetIsUrl ? !IsHttpUrl(target) : !target.StartsWith('/') && target != "*")
            {
                return ctx.SetReturn(targetIsUrl ? HttpErrorInvalidUrl : HttpErrorInvalidValue);
            }

            var request = AddNode(NodeKind.Request, connection.ContextId, connectionId);
            request.Settings["contentlength"] = ctx[CpuRegister.Rcx];
            return ReturnId(ctx, request.Id);
        }
    }

    // Reports the offline network error for any response query, zeroing the outputs first.
    private static int ResponseQuery(CpuContext ctx, (CpuRegister Register, int Size)[] outputs)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckNode(requestId, NodeKind.Request, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }
        }

        foreach (var (register, size) in outputs)
        {
            WriteZeros(ctx, ctx[register], size);
        }

        // No send ever completes, so no response exists to read.
        return ctx.SetReturn(HttpErrorBeforeSend);
    }

    private static int ClearCaList(CpuContext ctx)
    {
        var contextId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (Gate)
        {
            var error = CheckContext(contextId, out _);
            if (error != 0)
            {
                return ctx.SetReturn(error);
            }
        }

        return ctx[CpuRegister.Rsi] != 0 && WriteZeros(ctx, ctx[CpuRegister.Rsi], 16)
            ? ctx.SetReturn(0)
            : ctx.SetReturn(HttpErrorInvalidValue);
    }

    private static void MarkEpollDestroyed(HttpEpoll epoll)
    {
        lock (epoll)
        {
            epoll.Destroyed = true;
            Monitor.PulseAll(epoll);
        }
    }

    private static int ReturnId(CpuContext ctx, int id)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)id);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static bool WriteZeros(CpuContext ctx, ulong address, int size)
    {
        if (address == 0)
        {
            return false;
        }

        Span<byte> zeros = stackalloc byte[size];
        return ctx.Memory.TryWrite(address, zeros);
    }

    private static bool IsHttpScheme(string scheme) =>
        string.Equals(scheme, "http", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(scheme, "https", StringComparison.OrdinalIgnoreCase);

    private static bool IsHttpUrl(string url)
    {
        var separator = url.IndexOf("://", StringComparison.Ordinal);
        return separator > 0 && IsHttpScheme(url[..separator]) && url.Length > separator + 3 && url[separator + 3] != '/';
    }

    private static bool TryReadCString(CpuContext ctx, ulong address, out string value)
    {
        value = string.Empty;
        if (address == 0)
        {
            return false;
        }

        var bytes = new List<byte>();
        Span<byte> one = stackalloc byte[1];
        while (bytes.Count <= MaxStringLength)
        {
            if (!ctx.Memory.TryRead(address + (ulong)bytes.Count, one))
            {
                return false;
            }

            if (one[0] == 0)
            {
                value = System.Text.Encoding.Latin1.GetString(bytes.ToArray());
                return true;
            }

            bytes.Add(one[0]);
        }

        return false;
    }

    private static void TraceHttp(string operation, int id, ulong arg0, ulong arg1, ulong arg2, ulong arg3)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_HTTP"), "1", StringComparison.Ordinal))
        {
            return;
        }

        Console.Error.WriteLine(
            $"[LOADER][TRACE] http.{operation} id={id} arg0=0x{arg0:X16} arg1=0x{arg1:X16} arg2=0x{arg2:X16} arg3=0x{arg3:X16}");
    }
}
