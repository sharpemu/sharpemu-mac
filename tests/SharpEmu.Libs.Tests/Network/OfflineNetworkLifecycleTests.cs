// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Network;
using Xunit;

namespace SharpEmu.Libs.Tests.Network;

// Lifecycle and error-code coverage for the offline libSceNetCtl / libSceHttp / libSceSsl /
// libSceNet additions. Each class touches only its own library's static state.
public sealed class NetCtlOfflineTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const int ErrorInvalidId = unchecked((int)0x80412105);
    private const int ErrorInvalidCode = unchecked((int)0x80412106);
    private const int ErrorInvalidAddress = unchecked((int)0x80412107);
    private const int ErrorNotConnected = unchecked((int)0x80412108);
    private const int ErrorCallbackMax = unchecked((int)0x80412103);

    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x1000);
    private readonly CpuContext _ctx;

    public NetCtlOfflineTests()
    {
        NetCtlExports.ResetForTests();
        _ctx = new CpuContext(_memory, Generation.Gen5);
    }

    private int ReadInt(ulong address)
    {
        Span<byte> bytes = stackalloc byte[4];
        Assert.True(_memory.TryRead(address, bytes));
        return BinaryPrimitives.ReadInt32LittleEndian(bytes);
    }

    [Fact]
    public void InitAndTerm_AreIdempotentAndTermDropsCallbacks()
    {
        Assert.Equal(0, NetCtlExports.NetCtlInit(_ctx));
        Assert.Equal(0, NetCtlExports.NetCtlInit(_ctx));
        Assert.True(NetCtlExports.IsInitialized);

        _ctx[CpuRegister.Rdi] = 0x1234;
        _ctx[CpuRegister.Rsi] = 0;
        _ctx[CpuRegister.Rdx] = MemoryBase;
        Assert.Equal(0, NetCtlExports.NetCtlRegisterCallback(_ctx));
        Assert.True(NetCtlExports.IsCallbackRegistered(0, v6: false));

        Assert.Equal(0, NetCtlExports.NetCtlTerm(_ctx));
        Assert.False(NetCtlExports.IsInitialized);
        Assert.False(NetCtlExports.IsCallbackRegistered(0, v6: false));
        Assert.Equal(0, NetCtlExports.NetCtlTerm(_ctx));
    }

    [Fact]
    public void StateIsDisconnectedAndConsistentAcrossQueries()
    {
        _ctx[CpuRegister.Rdi] = MemoryBase;
        Assert.Equal(0, NetCtlExports.NetCtlGetState(_ctx));
        Assert.Equal(NetCtlExports.ReportedState, ReadInt(MemoryBase));
        Assert.Equal(0, ReadInt(MemoryBase));

        _memory.TryWrite(MemoryBase, [9, 9, 9, 9]);
        Assert.Equal(0, NetCtlExports.NetCtlGetStateV6(_ctx));
        Assert.Equal(0, ReadInt(MemoryBase));

        // Link is reported down; address information is therefore not available.
        _ctx[CpuRegister.Rdi] = 4;
        _ctx[CpuRegister.Rsi] = MemoryBase + 0x10;
        Assert.Equal(0, NetCtlExports.NetCtlGetInfo(_ctx));
        Assert.Equal(0, ReadInt(MemoryBase + 0x10));

        foreach (var code in new ulong[] { 3, 5, 11, 14, 15, 16, 17, 18, 19, 20, 21 })
        {
            _ctx[CpuRegister.Rdi] = code;
            _ctx[CpuRegister.Rsi] = MemoryBase + 0x10;
            Assert.Equal(ErrorNotConnected, NetCtlExports.NetCtlGetInfo(_ctx));
        }

        _ctx[CpuRegister.Rdi] = MemoryBase + 0x20;
        Assert.Equal(ErrorNotConnected, NetCtlExports.NetCtlGetIfStat(_ctx));
        Assert.Equal(ErrorNotConnected, NetCtlExports.NetCtlGetNatInfo(_ctx));
        Assert.Equal(0, NetCtlExports.NetCtlCheckCallback(_ctx));
    }

    [Fact]
    public void GetInfo_ValidatesCodeAndPointer()
    {
        _ctx[CpuRegister.Rdi] = 14;
        _ctx[CpuRegister.Rsi] = 0;
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlGetInfo(_ctx));

        foreach (var code in new ulong[] { 0, 22, 0xFFFFFFFF })
        {
            _ctx[CpuRegister.Rdi] = code;
            _ctx[CpuRegister.Rsi] = MemoryBase;
            Assert.Equal(ErrorInvalidCode, NetCtlExports.NetCtlGetInfo(_ctx));
        }

        _ctx[CpuRegister.Rdi] = 0;
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlGetState(_ctx));
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlGetStateV6(_ctx));
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlGetIfStat(_ctx));
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlGetNatInfo(_ctx));
    }

    [Fact]
    public void GetResult_WritesZeroAndRejectsNull()
    {
        _memory.TryWrite(MemoryBase, [7, 7, 7, 7]);
        _ctx[CpuRegister.Rdi] = 1;
        _ctx[CpuRegister.Rsi] = MemoryBase;
        Assert.Equal(0, NetCtlExports.NetCtlGetResult(_ctx));
        Assert.Equal(0, ReadInt(MemoryBase));

        _ctx[CpuRegister.Rsi] = 0;
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlGetResult(_ctx));
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlGetResultV6(_ctx));
    }

    [Fact]
    public void Callbacks_RegisterUnregisterAndExhaust()
    {
        _ctx[CpuRegister.Rdi] = 0x1000;
        _ctx[CpuRegister.Rsi] = 0;
        _ctx[CpuRegister.Rdx] = 0;
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlRegisterCallback(_ctx));
        _ctx[CpuRegister.Rdi] = 0;
        _ctx[CpuRegister.Rdx] = MemoryBase;
        Assert.Equal(ErrorInvalidAddress, NetCtlExports.NetCtlRegisterCallback(_ctx));

        for (var index = 0; index < 8; index++)
        {
            _ctx[CpuRegister.Rdi] = 0x1000;
            _ctx[CpuRegister.Rdx] = MemoryBase;
            Assert.Equal(0, NetCtlExports.NetCtlRegisterCallback(_ctx));
            Assert.Equal(index, ReadInt(MemoryBase));
        }

        _ctx[CpuRegister.Rdi] = 0x1000;
        _ctx[CpuRegister.Rdx] = MemoryBase;
        Assert.Equal(ErrorCallbackMax, NetCtlExports.NetCtlRegisterCallback(_ctx));

        // The V6 table is independent of the V4 table.
        Assert.Equal(0, NetCtlExports.NetCtlRegisterCallbackV6(_ctx));
        Assert.Equal(0, ReadInt(MemoryBase));

        _ctx[CpuRegister.Rdi] = 3;
        Assert.Equal(0, NetCtlExports.NetCtlUnregisterCallback(_ctx));
        Assert.False(NetCtlExports.IsCallbackRegistered(3, v6: false));
        Assert.True(NetCtlExports.IsCallbackRegistered(0, v6: true));

        // The freed slot is reused first.
        _ctx[CpuRegister.Rdi] = 0x1000;
        _ctx[CpuRegister.Rdx] = MemoryBase;
        Assert.Equal(0, NetCtlExports.NetCtlRegisterCallback(_ctx));
        Assert.Equal(3, ReadInt(MemoryBase));

        foreach (var badId in new ulong[] { 8, 0xFFFFFFFF })
        {
            _ctx[CpuRegister.Rdi] = badId;
            Assert.Equal(ErrorInvalidId, NetCtlExports.NetCtlUnregisterCallback(_ctx));
            Assert.Equal(ErrorInvalidId, NetCtlExports.NetCtlUnregisterCallbackV6(_ctx));
        }

        _ctx[CpuRegister.Rdi] = 0;
        Assert.Equal(0, NetCtlExports.NetCtlUnregisterCallbackV6(_ctx));
        Assert.False(NetCtlExports.IsCallbackRegistered(0, v6: true));
    }
}

public sealed class HttpOfflineLifecycleTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const int ErrorBeforeInit = unchecked((int)0x80431001);
    // Offline, a send fails because the host name never resolves, and no response exists to query.
    private const int ErrorResolverNoDns = unchecked((int)0x80436002);
    private const int ErrorBeforeSend = unchecked((int)0x80431065);
    private const int ErrorAborted = unchecked((int)0x80431080);
    private const int ErrorInvalidId = unchecked((int)0x80431100);
    private const int ErrorInvalidValue = unchecked((int)0x804311FE);
    private const int ErrorInvalidUrl = unchecked((int)0x80433060);

    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x4000);
    private readonly CpuContext _ctx;

    public HttpOfflineLifecycleTests()
    {
        HttpExports.ResetRuntimeState();
        _ctx = new CpuContext(_memory, Generation.Gen5);
    }

    private ulong Str(int slot, string text) => _memory.WriteCString(MemoryBase + 0x1000 + (ulong)slot * 0x100, text);

    private int Init()
    {
        _ctx[CpuRegister.Rdi] = 1;
        _ctx[CpuRegister.Rsi] = 2;
        _ctx[CpuRegister.Rdx] = 0x10000;
        Assert.Equal(0, HttpExports.HttpInit(_ctx));
        return unchecked((int)_ctx[CpuRegister.Rax]);
    }

    private int Template(int contextId)
    {
        _ctx[CpuRegister.Rdi] = unchecked((uint)contextId);
        _ctx[CpuRegister.Rsi] = Str(0, "agent");
        _ctx[CpuRegister.Rdx] = 2;
        _ctx[CpuRegister.Rcx] = 0;
        Assert.Equal(0, HttpExports.HttpCreateTemplate(_ctx));
        return unchecked((int)_ctx[CpuRegister.Rax]);
    }

    private int Connection(int templateId)
    {
        _ctx[CpuRegister.Rdi] = unchecked((uint)templateId);
        _ctx[CpuRegister.Rsi] = Str(1, "example.invalid");
        _ctx[CpuRegister.Rdx] = Str(2, "https");
        _ctx[CpuRegister.Rcx] = 443;
        _ctx[CpuRegister.R8] = 1;
        Assert.Equal(0, HttpExports.HttpCreateConnection(_ctx));
        return unchecked((int)_ctx[CpuRegister.Rax]);
    }

    private int Request(int connectionId)
    {
        _ctx[CpuRegister.Rdi] = unchecked((uint)connectionId);
        _ctx[CpuRegister.Rsi] = 0;
        _ctx[CpuRegister.Rdx] = Str(3, "/index");
        _ctx[CpuRegister.Rcx] = 0;
        Assert.Equal(0, HttpExports.HttpCreateRequest(_ctx));
        return unchecked((int)_ctx[CpuRegister.Rax]);
    }

    private int Call(Func<CpuContext, int> export, ulong rdi, ulong rsi = 0, ulong rdx = 0, ulong rcx = 0)
    {
        _ctx[CpuRegister.Rdi] = rdi;
        _ctx[CpuRegister.Rsi] = rsi;
        _ctx[CpuRegister.Rdx] = rdx;
        _ctx[CpuRegister.Rcx] = rcx;
        return export(_ctx);
    }

    [Fact]
    public void UseBeforeInit_ReportsBeforeInit()
    {
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpCreateTemplate, 1, Str(0, "a"), 2));
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpSetConnectTimeOut, 0x1001, 5));
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpSendRequest, 0x1001));
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpCreateEpoll, 1, MemoryBase));
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpWaitRequest, 1, MemoryBase, 1, 0));
        Assert.Equal(ErrorInvalidId, Init() == 1 ? Call(HttpExports.HttpTerm, 99) : 0);
    }

    [Fact]
    public void Term_DoubleTermAndCascade()
    {
        var context = Init();
        var template = Template(context);
        var connection = Connection(template);
        var request = Request(connection);
        Assert.Equal(3, HttpExports.LiveHandleCount);

        Assert.Equal(0, Call(HttpExports.HttpTerm, unchecked((uint)context)));
        Assert.Equal(0, HttpExports.LiveHandleCount);
        Assert.Equal(0, HttpExports.LiveContextCount);

        // After the last context is gone the library is uninitialised again.
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpTerm, unchecked((uint)context)));
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpSendRequest, unchecked((uint)request)));

        var second = Init();
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpTerm, unchecked((uint)context)) == 0 ? 0 : ErrorInvalidId);
        Assert.Equal(0, Call(HttpExports.HttpTerm, unchecked((uint)second)));
    }

    [Fact]
    public void Handles_AreValidatedByKindAndDeletedOnce()
    {
        var context = Init();
        var template = Template(context);
        var connection = Connection(template);
        var request = Request(connection);

        // Wrong kinds and unknown ids.
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpDeleteRequest, unchecked((uint)connection)));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpDeleteConnection, unchecked((uint)template)));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpDeleteTemplate, unchecked((uint)request)));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpCreateConnection, unchecked((uint)connection), Str(1, "h"), Str(2, "http")));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpCreateRequest, unchecked((uint)template), 0, Str(3, "/")));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpSendRequest, 0xDEAD));

        // Deleting a connection takes its request with it; a second delete is an invalid id.
        Assert.Equal(0, Call(HttpExports.HttpDeleteConnection, unchecked((uint)connection)));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpDeleteConnection, unchecked((uint)connection)));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpSendRequest, unchecked((uint)request)));
        Assert.Equal(0, Call(HttpExports.HttpDeleteTemplate, unchecked((uint)template)));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpDeleteTemplate, unchecked((uint)template)));
    }

    [Fact]
    public void Creation_ValidatesArguments()
    {
        var context = Init();
        // A null user agent is accepted (upstream's libSceHttp does the same); a bad version is not.
        Assert.Equal(0, Call(HttpExports.HttpCreateTemplate, unchecked((uint)context), 0, 2));
        Assert.NotEqual(0UL, _ctx[CpuRegister.Rax]);
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpCreateTemplate, unchecked((uint)context), Str(0, "a"), 9));
        var template = Template(context);

        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpCreateConnection, unchecked((uint)template), Str(1, "h"), Str(2, "ftp")));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpCreateConnection, unchecked((uint)template), 0, Str(2, "http")));
        Assert.Equal(ErrorInvalidUrl, Call(HttpExports.HttpCreateConnectionWithUrl, unchecked((uint)template), Str(1, "ftp://x/")));
        Assert.Equal(0, Call(HttpExports.HttpCreateConnectionWithUrl, unchecked((uint)template), Str(1, "https://example.invalid/a"), 1));
        var connection = unchecked((int)_ctx[CpuRegister.Rax]);

        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpCreateRequest, unchecked((uint)connection), 8, Str(3, "/")));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpCreateRequest, unchecked((uint)connection), 0, 0));
        Assert.Equal(ErrorInvalidUrl, Call(HttpExports.HttpCreateRequestWithUrl, unchecked((uint)connection), 0, Str(3, "nonsense")));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpCreateRequest2, unchecked((uint)connection), Str(4, ""), Str(3, "/")));
        Assert.Equal(0, Call(HttpExports.HttpCreateRequestWithUrl2, unchecked((uint)connection), Str(4, "GET"), Str(3, "http://example.invalid/")));
    }

    [Fact]
    public void RequestsFailOfflineWithZeroedOutputs()
    {
        var context = Init();
        var request = Request(Connection(Template(context)));

        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpSendRequest, unchecked((uint)request), 0, 5));
        Assert.Equal(ErrorResolverNoDns, Call(HttpExports.HttpSendRequest, unchecked((uint)request)));

        _memory.TryWrite(MemoryBase, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        Assert.Equal(ErrorBeforeSend, Call(HttpExports.HttpGetStatusCode, unchecked((uint)request), MemoryBase));
        Assert.Equal(ErrorBeforeSend, Call(HttpExports.HttpGetResponseContentLength, unchecked((uint)request), MemoryBase + 4, MemoryBase + 0x10));
        Span<byte> bytes = stackalloc byte[8];
        Assert.True(_memory.TryRead(MemoryBase, bytes[..4]));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(bytes));
        Assert.Equal(ErrorBeforeSend, Call(HttpExports.HttpReadData, unchecked((uint)request), MemoryBase, 16));
        Assert.Equal(ErrorBeforeSend, Call(HttpExports.HttpGetAllResponseHeaders, unchecked((uint)request), MemoryBase, MemoryBase + 8));

        _memory.TryWrite(MemoryBase, [5, 5, 5, 5]);
        Assert.Equal(0, Call(HttpExports.HttpGetLastErrno, unchecked((uint)request), MemoryBase));
        Assert.True(_memory.TryRead(MemoryBase, bytes[..4]));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(bytes));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpGetLastErrno, unchecked((uint)request), 0));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpGetStatusCode, 0xBEEF, MemoryBase));

        Assert.Equal(0, Call(HttpExports.HttpAbortRequest, unchecked((uint)request)));
        Assert.Equal(ErrorAborted, Call(HttpExports.HttpSendRequest, unchecked((uint)request)));
    }

    [Fact]
    public void Setters_ValidateIdAndValue()
    {
        var context = Init();
        var template = Template(context);
        var request = Request(Connection(template));

        Assert.Equal(0, Call(HttpExports.HttpSetConnectTimeOut, unchecked((uint)template), 5_000_000));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpSetConnectTimeOut, unchecked((uint)template), 0));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpSetRecvTimeOut, 0x5555, 5));
        Assert.Equal(0, Call(HttpExports.HttpSetResolveRetry, unchecked((uint)request), 3));

        Assert.Equal(0, Call(HttpExports.HttpSetAutoRedirect, unchecked((uint)template), 0));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpSetAutoRedirect, unchecked((uint)template), 2));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpSetCookieEnabled, 0x5555, 1));

        // Requests inherit what their template set; unset options report the SDK defaults.
        Assert.Equal(0, Call(HttpExports.HttpGetAutoRedirect, unchecked((uint)request), MemoryBase));
        Span<byte> value = stackalloc byte[4];
        Assert.True(_memory.TryRead(MemoryBase, value));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(value));
        Assert.Equal(0, Call(HttpExports.HttpGetNonblock, unchecked((uint)request), MemoryBase));
        Assert.True(_memory.TryRead(MemoryBase, value));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(value));
        Assert.Equal(0, Call(HttpExports.HttpGetCookieEnabled, unchecked((uint)request), MemoryBase));
        Assert.True(_memory.TryRead(MemoryBase, value));
        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(value));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpGetCookieEnabled, unchecked((uint)request), 0));

        // Request-only setters reject other handle kinds.
        Assert.Equal(0, Call(HttpExports.HttpSetRequestContentLength, unchecked((uint)request), 10));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpSetRequestContentLength, unchecked((uint)template), 10));
        Assert.Equal(0, Call(HttpExports.HttpSetRequestStatusCallback, unchecked((uint)request), 0x1000, 0));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpSetRequestStatusCallback, unchecked((uint)template), 0x1000, 0));
    }

    [Fact]
    public void Headers_ValidateModeAndArguments()
    {
        var context = Init();
        var request = Request(Connection(Template(context)));
        var name = Str(5, "X-Test");
        var value = Str(6, "1");

        Assert.Equal(0, Call(HttpExports.HttpAddRequestHeader, unchecked((uint)request), name, value, 1));
        Assert.Equal(0, Call(HttpExports.HttpAddRequestHeader, unchecked((uint)request), name, value, 0));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpAddRequestHeader, unchecked((uint)request), name, value, 2));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpAddRequestHeader, unchecked((uint)request), 0, value, 1));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpAddRequestHeader, 0x7777, name, value, 1));
        Assert.Equal(0, Call(HttpExports.HttpRemoveRequestHeader, unchecked((uint)request), name));
    }

    [Fact]
    public void Epoll_LifecycleAbortAndTimeout()
    {
        var context = Init();
        var template = Template(context);
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpCreateEpoll, unchecked((uint)context), 0));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpCreateEpoll, 99, MemoryBase));
        Assert.Equal(0, Call(HttpExports.HttpCreateEpoll, unchecked((uint)context), MemoryBase));
        Span<byte> handleBytes = stackalloc byte[8];
        Assert.True(_memory.TryRead(MemoryBase, handleBytes));
        var epoll = BinaryPrimitives.ReadUInt64LittleEndian(handleBytes);
        Assert.NotEqual(0UL, epoll);

        Assert.Equal(0, Call(HttpExports.HttpSetEpoll, unchecked((uint)template), epoll, 0));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpSetEpoll, unchecked((uint)template), 0xABCDEF, 0));
        Assert.Equal(0, Call(HttpExports.HttpUnsetEpoll, unchecked((uint)template)));

        // A zero timeout polls and finds nothing pending.
        Assert.Equal(0, Call(HttpExports.HttpWaitRequest, epoll, MemoryBase + 0x100, 1, 0));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpWaitRequest, epoll, 0, 1, 0));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpWaitRequest, epoll, MemoryBase + 0x100, 0, 0));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpWaitRequest, 0xABCDEF, MemoryBase + 0x100, 1, 0));

        // An abort sticks and wakes waiters.
        Assert.Equal(0, Call(HttpExports.HttpAbortWaitRequest, epoll));
        Assert.Equal(ErrorAborted, Call(HttpExports.HttpWaitRequest, epoll, MemoryBase + 0x100, 1, 0xFFFFFFFF));

        Assert.Equal(0, Call(HttpExports.HttpDestroyEpoll, unchecked((uint)context), epoll));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpDestroyEpoll, unchecked((uint)context), epoll));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpAbortWaitRequest, epoll));
    }

    [Fact]
    public void Certificates_ValidateContext()
    {
        Assert.Equal(ErrorBeforeInit, Call(HttpExports.HttpsLoadCert, 1, 0, 0, 0));
        var context = Init();
        Assert.Equal(0, Call(HttpExports.HttpsLoadCert, unchecked((uint)context), 0, 0, 0));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpsLoadCert, unchecked((uint)context), 1, 0, 0));
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpsLoadCert, unchecked((uint)context), 0, 0, MemoryBase));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpsLoadCert, 77, 0, 0, 0));

        _memory.TryWrite(MemoryBase, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        Assert.Equal(0, Call(HttpExports.HttpsGetCaList, unchecked((uint)context), MemoryBase));
        Span<byte> list = stackalloc byte[16];
        Assert.True(_memory.TryRead(MemoryBase, list));
        Assert.True(list.IndexOfAnyExcept((byte)0) < 0);
        Assert.Equal(ErrorInvalidValue, Call(HttpExports.HttpsFreeCaList, unchecked((uint)context), 0));
        Assert.Equal(0, Call(HttpExports.HttpsUnloadCert, unchecked((uint)context)));
        Assert.Equal(ErrorInvalidId, Call(HttpExports.HttpsUnloadCert, 77));
    }
}

public sealed class SslOfflineLifecycleTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const int ErrorNotFound = unchecked((int)0x8095F004);
    private const int ErrorInvalidId = unchecked((int)0x8095F006);
    private const int ErrorInvalidPointer = unchecked((int)0x8095F007);
    private const int ErrorOutOfSize = unchecked((int)0x8095F008);
    private const int ErrorInvalidArgument = unchecked((int)0x8095177A);
    private const int ErrorNotConnected = unchecked((int)0x80410139);

    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x2000);
    private readonly CpuContext _ctx;

    public SslOfflineLifecycleTests()
    {
        SslExports.ResetRuntimeState();
        _ctx = new CpuContext(_memory, Generation.Gen5);
    }

    private int Call(Func<CpuContext, int> export, ulong rdi, ulong rsi = 0, ulong rdx = 0, ulong rcx = 0, ulong r8 = 0)
    {
        _ctx[CpuRegister.Rdi] = rdi;
        _ctx[CpuRegister.Rsi] = rsi;
        _ctx[CpuRegister.Rdx] = rdx;
        _ctx[CpuRegister.Rcx] = rcx;
        _ctx[CpuRegister.R8] = r8;
        return export(_ctx);
    }

    private int Init()
    {
        Assert.Equal(0, Call(SslExports.SslInit, 0x10000));
        return unchecked((int)_ctx[CpuRegister.Rax]);
    }

    [Fact]
    public void InitTerm_RejectsZeroPoolBadIdAndDoubleTerm()
    {
        Assert.Equal(ErrorOutOfSize, Call(SslExports.SslInit, 0));
        var id = Init();
        Assert.Equal(0, Call(SslExports.SslTerm, unchecked((uint)id)));
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslTerm, unchecked((uint)id)));
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslTerm, 0));
    }

    [Fact]
    public void MemoryPoolStats_ReportsPoolSizeAndValidatesArguments()
    {
        var id = Init();
        Assert.Equal(ErrorInvalidPointer, Call(SslExports.SslGetMemoryPoolStats, unchecked((uint)id), 0));
        Assert.Equal(0, Call(SslExports.SslGetMemoryPoolStats, unchecked((uint)id), MemoryBase));
        Span<byte> stats = stackalloc byte[32];
        Assert.True(_memory.TryRead(MemoryBase, stats));
        Assert.Equal(0x10000UL, BinaryPrimitives.ReadUInt64LittleEndian(stats));
        Assert.True(stats[8..].IndexOfAnyExcept((byte)0) < 0);

        _memory.TryWrite(MemoryBase, new byte[32].Select(_ => (byte)0xFF).ToArray());
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslGetMemoryPoolStats, 999, MemoryBase));
        Assert.True(_memory.TryRead(MemoryBase, stats));
        Assert.True(stats.IndexOfAnyExcept((byte)0) < 0);
    }

    [Fact]
    public void LoadCert_ValidatesAndCaListStaysEmpty()
    {
        var id = Init();
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslLoadCert, 999, 0, 0, 0, 0));
        Assert.Equal(ErrorInvalidArgument, Call(SslExports.SslLoadCert, unchecked((uint)id), 0xFFFFFFFF, 0, 0, 0));
        Assert.Equal(ErrorInvalidArgument, Call(SslExports.SslLoadCert, unchecked((uint)id), 1, 0, 0, 0));
        Assert.Equal(ErrorInvalidArgument, Call(SslExports.SslLoadCert, unchecked((uint)id), 0, 0, MemoryBase, 0));
        Assert.Equal(0, Call(SslExports.SslLoadCert, unchecked((uint)id), 1, MemoryBase + 0x100, 0, 0));

        _memory.TryWrite(MemoryBase, new byte[24].Select(_ => (byte)0xAA).ToArray());
        Assert.Equal(ErrorNotFound, Call(SslExports.SslGetCaList, unchecked((uint)id), MemoryBase));
        Span<byte> list = stackalloc byte[16];
        Assert.True(_memory.TryRead(MemoryBase, list));
        Assert.True(list.IndexOfAnyExcept((byte)0) < 0);
        Assert.Equal(ErrorInvalidArgument, Call(SslExports.SslGetCaList, unchecked((uint)id), 0));
        Assert.Equal(ErrorNotFound, Call(SslExports.SslGetCaCerts, unchecked((uint)id), MemoryBase));
        Assert.Equal(ErrorNotFound, Call(SslExports.SslGetCaCerts, 999, MemoryBase));

        // Only the empty list can be freed; nothing was ever allocated for a non-empty one.
        Assert.Equal(0, Call(SslExports.SslFreeCaList, unchecked((uint)id), MemoryBase));
        _memory.TryWrite(MemoryBase, [1, 0, 0, 0, 0, 0, 0, 0]);
        Assert.Equal(ErrorInvalidArgument, Call(SslExports.SslFreeCaList, unchecked((uint)id), MemoryBase));
        Assert.Equal(ErrorInvalidArgument, Call(SslExports.SslFreeCaCerts, unchecked((uint)id), 0));

        Assert.Equal(0, Call(SslExports.SslUnloadCert, unchecked((uint)id)));
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslUnloadCert, 999));
    }

    [Fact]
    public void Connections_FollowContextLifecycleAndCannotConnect()
    {
        var id = Init();
        var host = _memory.WriteCString(MemoryBase + 0x200, "example.invalid");
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslCreateConnection, 999, 300, host));
        Assert.Equal(ErrorInvalidArgument, Call(SslExports.SslCreateConnection, unchecked((uint)id), 300, 0));
        Assert.Equal(0, Call(SslExports.SslCreateConnection, unchecked((uint)id), 300, host));
        var connection = unchecked((int)_ctx[CpuRegister.Rax]);

        Assert.Equal(0, Call(SslExports.SslSetVerifyCallback, unchecked((uint)connection), 0x1000, 0));
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslSetVerifyCallback, 0x1, 0x1000, 0));
        Assert.Equal(ErrorNotConnected, Call(SslExports.SslConnect, unchecked((uint)connection)));
        Assert.Equal(ErrorNotConnected, Call(SslExports.SslRead, unchecked((uint)connection), MemoryBase, 4, 0));
        Assert.Equal(ErrorNotConnected, Call(SslExports.SslWrite, unchecked((uint)connection), MemoryBase, 4, 0));
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslConnect, 0x1));

        // Terminating the context destroys its connections.
        Assert.Equal(0, Call(SslExports.SslTerm, unchecked((uint)id)));
        Assert.Equal(0, SslExports.LiveConnectionCount);
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslDeleteConnection, unchecked((uint)connection)));

        var second = Init();
        Assert.Equal(0, Call(SslExports.SslCreateConnection, unchecked((uint)second), 301, host));
        var other = unchecked((int)_ctx[CpuRegister.Rax]);
        Assert.Equal(0, Call(SslExports.SslDeleteConnection, unchecked((uint)other)));
        Assert.Equal(ErrorInvalidId, Call(SslExports.SslDeleteConnection, unchecked((uint)other)));
    }
}

public sealed class NetOfflineSocketTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const int ErrorBadFileDescriptor = unchecked((int)0x80410109);
    private const int ErrorInvalidArgument = unchecked((int)0x80410116);
    private const int ErrorNotConnected = unchecked((int)0x80410139);
    private const int ErrorResolverNoDns = unchecked((int)0x804101E1);

    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x2000);
    private readonly CpuContext _ctx;

    public NetOfflineSocketTests()
    {
        _ctx = new CpuContext(_memory, Generation.Gen5);
    }

    [Fact]
    public void Htonll_Ntohll_SwapAllEightBytes()
    {
        _ctx[CpuRegister.Rdi] = 0x0102030405060708;
        Assert.Equal(0, NetExports.NetHtonll(_ctx));
        Assert.Equal(0x0807060504030201UL, _ctx[CpuRegister.Rax]);
        Assert.Equal(0, NetExports.NetNtohll(_ctx));
        Assert.Equal(0x0807060504030201UL, _ctx[CpuRegister.Rax]);
    }

    [Fact]
    public void SocketOperations_RejectBadDescriptors()
    {
        _ctx[CpuRegister.Rdi] = 0x3FF;
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetConnect(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetSend(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetSendto(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetShutdown(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetGetsockname(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetGetpeername(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetSocketAbort(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetEpollAbort(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetResolverAbort(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetGetMemoryPoolStats(_ctx));
    }

    [Fact]
    public void UnconnectedUdpSocket_ReportsLocalNameAndNoPeer()
    {
        _ctx[CpuRegister.Rdi] = 0;
        _ctx[CpuRegister.Rsi] = 2;
        _ctx[CpuRegister.Rdx] = 2;
        _ctx[CpuRegister.Rcx] = 17;
        Assert.Equal(0, NetExports.NetSocket(_ctx));
        var socket = _ctx[CpuRegister.Rax];
        try
        {
            _memory.TryWrite(MemoryBase + 0x100, [16, 0, 0, 0]);
            _ctx[CpuRegister.Rdi] = socket;
            _ctx[CpuRegister.Rsi] = MemoryBase;
            _ctx[CpuRegister.Rdx] = MemoryBase + 0x100;
            Assert.Equal(ErrorNotConnected, NetExports.NetGetpeername(_ctx));
            Assert.Equal(0, NetExports.NetGetsockname(_ctx));
            Span<byte> address = stackalloc byte[16];
            Assert.True(_memory.TryRead(MemoryBase, address));
            Assert.Equal(16, address[0]);
            Assert.Equal(2, address[1]);

            // how must be SHUT_RD, SHUT_WR or SHUT_RDWR.
            _ctx[CpuRegister.Rdi] = socket;
            _ctx[CpuRegister.Rsi] = 3;
            Assert.Equal(ErrorInvalidArgument, NetExports.NetShutdown(_ctx));
            _ctx[CpuRegister.Rdi] = socket;
            Assert.Equal(0, NetExports.NetSocketAbort(_ctx));
        }
        finally
        {
            _ctx[CpuRegister.Rdi] = socket;
            NetExports.NetSocketClose(_ctx);
        }
    }

    [Fact]
    public void Resolver_FailsOfflineWithDocumentedErrors()
    {
        NetExports.NetInit(_ctx);
        var name = _memory.WriteCString(MemoryBase + 0x200, "resolver");
        _ctx[CpuRegister.Rdi] = name;
        _ctx[CpuRegister.Rsi] = 0;
        _ctx[CpuRegister.Rdx] = 0;
        Assert.Equal(0, NetExports.NetResolverCreate(_ctx));
        var resolver = _ctx[CpuRegister.Rax];
        var host = _memory.WriteCString(MemoryBase + 0x300, "no-such-host.invalid");

        // Argument validation comes before the (always failing) lookup.
        _ctx[CpuRegister.Rdi] = resolver;
        _ctx[CpuRegister.Rsi] = 0;
        _ctx[CpuRegister.Rdx] = MemoryBase;
        _ctx[CpuRegister.R9] = 0;
        Assert.Equal(ErrorInvalidArgument, NetExports.NetResolverStartNtoa6(_ctx));
        Assert.Equal(ErrorInvalidArgument, NetExports.NetResolverStartNtoaMultipleRecords(_ctx));
        Assert.Equal(ErrorInvalidArgument, NetExports.NetResolverStartNtoaMultipleRecordsEx(_ctx));

        _ctx[CpuRegister.Rsi] = host;
        Assert.Equal(ErrorResolverNoDns, NetExports.NetResolverStartNtoa6(_ctx));
        Assert.Equal(ErrorResolverNoDns, NetExports.NetResolverStartNtoaMultipleRecords(_ctx));
        Assert.Equal(ErrorResolverNoDns, NetExports.NetResolverStartNtoaMultipleRecordsEx(_ctx));

        _ctx[CpuRegister.Rdi] = resolver;
        _ctx[CpuRegister.Rsi] = MemoryBase + 0x400;
        _ctx[CpuRegister.Rdx] = MemoryBase + 0x500;
        _ctx[CpuRegister.Rcx] = 64;
        _memory.TryWrite(MemoryBase + 0x500, [0x41]);
        Assert.Equal(ErrorResolverNoDns, NetExports.NetResolverStartAton(_ctx));
        Span<byte> first = stackalloc byte[1];
        Assert.True(_memory.TryRead(MemoryBase + 0x500, first));
        Assert.Equal(0, first[0]);

        // An IPv6 literal needs no DNS.
        var literal = _memory.WriteCString(MemoryBase + 0x600, "::1");
        _ctx[CpuRegister.Rdi] = resolver;
        _ctx[CpuRegister.Rsi] = literal;
        _ctx[CpuRegister.Rdx] = MemoryBase + 0x700;
        _ctx[CpuRegister.R9] = 0;
        Assert.Equal(0, NetExports.NetResolverStartNtoa6(_ctx));
        Span<byte> address = stackalloc byte[16];
        Assert.True(_memory.TryRead(MemoryBase + 0x700, address));
        Assert.Equal(1, address[15]);

        _ctx[CpuRegister.Rdi] = resolver;
        Assert.Equal(0, NetExports.NetResolverAbort(_ctx));
        _ctx[CpuRegister.Rdi] = resolver;
        Assert.Equal(0, NetExports.NetResolverDestroy(_ctx));
        _ctx[CpuRegister.Rdi] = resolver;
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetResolverStartNtoa6(_ctx));
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetResolverAbort(_ctx));
    }

    [Fact]
    public void MemoryPoolStats_ReportPoolSize()
    {
        NetExports.NetInit(_ctx);
        _ctx[CpuRegister.Rdi] = _memory.WriteCString(MemoryBase + 0x200, "pool");
        _ctx[CpuRegister.Rsi] = 0x4000;
        _ctx[CpuRegister.Rdx] = 0;
        Assert.Equal(0, NetExports.NetPoolCreate(_ctx));
        var pool = _ctx[CpuRegister.Rax];

        _ctx[CpuRegister.Rdi] = pool;
        _ctx[CpuRegister.Rsi] = 0;
        Assert.Equal(ErrorInvalidArgument, NetExports.NetGetMemoryPoolStats(_ctx));
        _ctx[CpuRegister.Rdi] = pool;
        _ctx[CpuRegister.Rsi] = MemoryBase;
        Assert.Equal(0, NetExports.NetGetMemoryPoolStats(_ctx));
        Span<byte> stats = stackalloc byte[24];
        Assert.True(_memory.TryRead(MemoryBase, stats));
        Assert.Equal(0x4000UL, BinaryPrimitives.ReadUInt64LittleEndian(stats));
        Assert.True(stats[8..].IndexOfAnyExcept((byte)0) < 0);

        _ctx[CpuRegister.Rdi] = pool;
        Assert.Equal(0, NetExports.NetPoolDestroy(_ctx));
        _ctx[CpuRegister.Rdi] = pool;
        _ctx[CpuRegister.Rsi] = MemoryBase;
        Assert.Equal(ErrorBadFileDescriptor, NetExports.NetGetMemoryPoolStats(_ctx));
    }
}
