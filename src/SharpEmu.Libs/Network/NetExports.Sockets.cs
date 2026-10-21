// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Network;

// Remaining libSceNet socket, resolver and pool entry points. SCE_NET_ERROR_* is 0x80410100 plus
// the BSD errno, the same scheme the constants in NetExports.cs follow; the errno is also stored
// behind sceNetErrnoLoc.
public static partial class NetExports
{
    private const int NetErrorNotSocketOperation = unchecked((int)0x80410138);
    private const int NetErrorNotConnected = unchecked((int)0x80410139);
    private const int NetErrnoNotConnected = 57;
    private const int NetErrorInProgress = unchecked((int)0x80410124);
    private const int NetErrnoInProgress = 36;
    private const int NetErrorConnectionReset = unchecked((int)0x80410136);
    private const int NetErrnoConnectionReset = 54;
    private const int NetErrorTimedOut = unchecked((int)0x8041013C);
    private const int NetErrnoTimedOut = 60;
    private const int NetErrorConnectionRefused = unchecked((int)0x8041013D);
    private const int NetErrnoConnectionRefused = 61;
    private const int NetErrorNetworkUnreachable = unchecked((int)0x80410133);
    private const int NetErrnoNetworkUnreachable = 51;
    private const int NetErrorHostUnreachable = unchecked((int)0x80410141);
    private const int NetErrnoHostUnreachable = 65;
    private const uint NetMsgDontWait = 0x80;
    private const uint NetMsgNoSignal = 0x20000;
    private const int NetResolverFlagsMask = 0x10000;
    private const int MaxSendLength = 1 << 24;

    [SysAbiExport(
        Nid = "OXXX4mUk3uk",
        ExportName = "sceNetConnect",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetConnect(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        if (!_sockets.TryGetValue(id, out var socket))
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        if (!TryReadSocketAddress(ctx, ctx[CpuRegister.Rsi], unchecked((int)ctx[CpuRegister.Rdx]), out var endpoint))
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        try
        {
            socket.Connect(endpoint);
            TraceNet("socket.connect", id, unchecked((ulong)endpoint.Port), 0, 0);
            return ctx.SetReturn(0);
        }
        catch (SocketException exception)
        {
            return SetSocketException(ctx, exception);
        }
        catch (ObjectDisposedException)
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }
    }

    [SysAbiExport(
        Nid = "beRjXBn-z+o",
        ExportName = "sceNetSend",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetSend(CpuContext ctx) => SendSocket(ctx, withAddress: false);

    [SysAbiExport(
        Nid = "gvD1greCu0A",
        ExportName = "sceNetSendto",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetSendto(CpuContext ctx) => SendSocket(ctx, withAddress: true);

    [SysAbiExport(
        Nid = "TSM6whtekok",
        ExportName = "sceNetShutdown",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetShutdown(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        var how = unchecked((int)ctx[CpuRegister.Rsi]);
        if (!_sockets.TryGetValue(id, out var socket))
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        if (how is < 0 or > 2)
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        try
        {
            socket.Shutdown((SocketShutdown)how);
            return ctx.SetReturn(0);
        }
        catch (SocketException exception)
        {
            return SetSocketException(ctx, exception);
        }
        catch (ObjectDisposedException)
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }
    }

    // (s, level, optname, optval, optlen*) exactly like the POSIX getsockopt alias.
    [SysAbiExport(
        Nid = "xphrZusl78E",
        ExportName = "sceNetGetsockopt",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetGetsockopt(CpuContext ctx) => PosixGetsockopt(ctx);

    [SysAbiExport(
        Nid = "hoOAofhhRvE",
        ExportName = "sceNetGetsockname",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetGetsockname(CpuContext ctx) => WriteSocketName(ctx, peer: false);

    [SysAbiExport(
        Nid = "TCkRD0DWNLg",
        ExportName = "sceNetGetpeername",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetGetpeername(CpuContext ctx) => WriteSocketName(ctx, peer: true);

    [SysAbiExport(
        Nid = "3CHi1K1wsCQ",
        ExportName = "sceNetHtonll",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetHtonll(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = BinaryPrimitives.ReverseEndianness(ctx[CpuRegister.Rdi]);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "tOrRi-v3AOM",
        ExportName = "sceNetNtohll",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetNtohll(CpuContext ctx) => NetHtonll(ctx);

    // Blocking host calls cannot be interrupted from here, so an abort only validates the
    // descriptor; nothing is left waiting that an abort could wake.
    [SysAbiExport(
        Nid = "zJGf8xjFnQE",
        ExportName = "sceNetSocketAbort",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetSocketAbort(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        return _sockets.ContainsKey(id)
            ? ctx.SetReturn(0)
            : SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
    }

    [SysAbiExport(
        Nid = "w21YgGGNtBk",
        ExportName = "sceNetEpollAbort",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetEpollAbort(CpuContext ctx)
    {
        var epollId = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (EpollGate)
        {
            if (!_epolls.ContainsKey(epollId))
            {
                return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
            }

            Monitor.PulseAll(EpollGate);
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "AzqoBha7js4",
        ExportName = "sceNetResolverAbort",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetResolverAbort(CpuContext ctx)
    {
        // Lookups complete synchronously (and fail offline), so none is ever running to abort.
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        return _resolvers.ContainsKey(id)
            ? ctx.SetReturn(0)
            : SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
    }

    // (rid, hostname, SceNetIn6Addr*, timeout, retry, flags). Only an IPv6 literal resolves.
    [SysAbiExport(
        Nid = "zl35YNs9jnI",
        ExportName = "sceNetResolverStartNtoa6",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetResolverStartNtoa6(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        if (!_resolvers.TryGetValue(id, out var resolver))
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        var destination = ctx[CpuRegister.Rdx];
        var flags = unchecked((int)ctx[CpuRegister.R9]);
        if (destination == 0 || ctx[CpuRegister.Rsi] == 0 || flags != 0 ||
            !TryReadUtf8Z(ctx, ctx[CpuRegister.Rsi], MaxNameLength, out var hostname))
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        if (IPAddress.TryParse(hostname, out var parsed) && parsed.AddressFamily == AddressFamily.InterNetworkV6)
        {
            return ctx.Memory.TryWrite(destination, parsed.GetAddressBytes())
                ? ctx.SetReturn(0)
                : SetNetError(ctx, NetErrorFault, NetErrnoFault);
        }

        return FailResolver(ctx, id, resolver);
    }

    [SysAbiExport(
        Nid = "RCCY01Xd+58",
        ExportName = "sceNetResolverStartNtoaMultipleRecords",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetResolverStartNtoaMultipleRecords(CpuContext ctx) => ResolveMultipleRecords(ctx);

    [SysAbiExport(
        Nid = "sT4nBQKUPqM",
        ExportName = "sceNetResolverStartNtoaMultipleRecordsEx",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetResolverStartNtoaMultipleRecordsEx(CpuContext ctx) => ResolveMultipleRecords(ctx);

    // (rid, SceNetInAddr*, hostname buffer, length, timeout, retry, flags). There is no reverse
    // DNS offline: the output buffer is emptied and the lookup fails with ENODNS.
    [SysAbiExport(
        Nid = "Apb4YDxKsRI",
        ExportName = "sceNetResolverStartAton",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNet")]
    public static int NetResolverStartAton(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        if (!_resolvers.TryGetValue(id, out var resolver))
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        var hostnameAddress = ctx[CpuRegister.Rdx];
        var length = unchecked((int)ctx[CpuRegister.Rcx]);
        if (ctx[CpuRegister.Rsi] == 0 || hostnameAddress == 0 || length <= 0)
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        Span<byte> terminator = stackalloc byte[1];
        ctx.Memory.TryWrite(hostnameAddress, terminator);
        return FailResolver(ctx, id, resolver);
    }

    private static int ResolveMultipleRecords(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        if (!_resolvers.TryGetValue(id, out var resolver))
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        var flags = unchecked((int)ctx[CpuRegister.R9]);
        if (ctx[CpuRegister.Rsi] == 0 || ctx[CpuRegister.Rdx] == 0 || (flags & ~NetResolverFlagsMask) != 0 ||
            !TryReadUtf8Z(ctx, ctx[CpuRegister.Rsi], MaxNameLength, out _))
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        return FailResolver(ctx, id, resolver);
    }

    private static int FailResolver(CpuContext ctx, int id, ResolverContext resolver)
    {
        _resolvers.TryUpdate(id, resolver with { LastError = NetErrorResolverNoDns }, resolver);
        return SetNetError(ctx, NetErrorResolverNoDns, NetErrnoResolverNoDns);
    }

    private static int SendSocket(CpuContext ctx, bool withAddress)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        var bufferAddress = ctx[CpuRegister.Rsi];
        var requested = ctx[CpuRegister.Rdx];
        var flags = unchecked((uint)ctx[CpuRegister.Rcx]);
        var addressAddress = withAddress ? ctx[CpuRegister.R8] : 0;
        var addressLength = withAddress ? unchecked((int)ctx[CpuRegister.R9]) : 0;
        if (!_sockets.TryGetValue(id, out var socket))
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        if ((flags & ~(NetMsgDontWait | NetMsgNoSignal)) != 0 || requested > MaxSendLength ||
            (requested != 0 && bufferAddress == 0))
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        IPEndPoint? destination = null;
        if (addressAddress != 0 &&
            !TryReadSocketAddress(ctx, addressAddress, addressLength, out destination))
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        if (requested == 0)
        {
            return ctx.SetReturn(0);
        }

        var payload = new byte[requested];
        if (!ctx.Memory.TryRead(bufferAddress, payload))
        {
            return SetNetError(ctx, NetErrorFault, NetErrnoFault);
        }

        try
        {
            if ((flags & NetMsgDontWait) != 0 && !socket.Poll(0, SelectMode.SelectWrite))
            {
                return SetNetError(ctx, NetErrorWouldBlock, NetErrnoWouldBlock);
            }

            var sent = destination is null
                ? socket.Send(payload, SocketFlags.None)
                : socket.SendTo(payload, SocketFlags.None, destination);
            TraceNet("socket.send", id, requested, unchecked((ulong)sent), 0);
            return ctx.SetReturn(sent);
        }
        catch (SocketException exception)
        {
            return SetSocketException(ctx, exception);
        }
        catch (ObjectDisposedException)
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }
    }

    private static int WriteSocketName(CpuContext ctx, bool peer)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        var addressAddress = ctx[CpuRegister.Rsi];
        var lengthAddress = ctx[CpuRegister.Rdx];
        if (!_sockets.TryGetValue(id, out var socket))
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        if (addressAddress == 0 || lengthAddress == 0)
        {
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        }

        IPEndPoint endpoint;
        try
        {
            var raw = peer ? socket.RemoteEndPoint : socket.LocalEndPoint;
            if (raw is null && peer)
            {
                return SetNetError(ctx, NetErrorNotConnected, NetErrnoNotConnected);
            }

            endpoint = raw as IPEndPoint ?? new IPEndPoint(IPAddress.Any, 0);
        }
        catch (ObjectDisposedException)
        {
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        }

        Span<byte> lengthBytes = stackalloc byte[sizeof(uint)];
        if (!ctx.Memory.TryRead(lengthAddress, lengthBytes))
        {
            return SetNetError(ctx, NetErrorFault, NetErrnoFault);
        }

        // SceNetSockaddrIn: { len, family, port (big endian), addr, 8 bytes padding }.
        Span<byte> guestAddress = stackalloc byte[16];
        guestAddress[0] = 16;
        guestAddress[1] = 2;
        BinaryPrimitives.WriteUInt16BigEndian(guestAddress[2..], unchecked((ushort)endpoint.Port));
        endpoint.Address.MapToIPv4().GetAddressBytes().CopyTo(guestAddress[4..]);
        var copyLength = (int)Math.Min(BinaryPrimitives.ReadUInt32LittleEndian(lengthBytes), 16u);
        BinaryPrimitives.WriteUInt32LittleEndian(lengthBytes, 16);
        return ctx.Memory.TryWrite(addressAddress, guestAddress[..copyLength]) &&
            ctx.Memory.TryWrite(lengthAddress, lengthBytes)
                ? ctx.SetReturn(0)
                : SetNetError(ctx, NetErrorFault, NetErrnoFault);
    }

    private static int SetSocketException(CpuContext ctx, SocketException exception)
    {
        return exception.SocketErrorCode switch
        {
            SocketError.WouldBlock => SetNetError(ctx, NetErrorWouldBlock, NetErrnoWouldBlock),
            SocketError.InProgress or SocketError.IOPending =>
                SetNetError(ctx, NetErrorInProgress, NetErrnoInProgress),
            SocketError.ConnectionRefused =>
                SetNetError(ctx, NetErrorConnectionRefused, NetErrnoConnectionRefused),
            SocketError.NetworkUnreachable =>
                SetNetError(ctx, NetErrorNetworkUnreachable, NetErrnoNetworkUnreachable),
            SocketError.HostUnreachable =>
                SetNetError(ctx, NetErrorHostUnreachable, NetErrnoHostUnreachable),
            SocketError.TimedOut => SetNetError(ctx, NetErrorTimedOut, NetErrnoTimedOut),
            SocketError.NotConnected => SetNetError(ctx, NetErrorNotConnected, NetErrnoNotConnected),
            SocketError.ConnectionReset or SocketError.ConnectionAborted =>
                SetNetError(ctx, NetErrorConnectionReset, NetErrnoConnectionReset),
            SocketError.AddressAlreadyInUse => SetNetError(ctx, NetErrorAddressInUse, NetErrnoAddressInUse),
            _ => SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument),
        };
    }
}
