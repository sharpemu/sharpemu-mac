// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using System.Buffers.Binary;
using System.Collections.Concurrent;

namespace SharpEmu.Libs.Np;

public static class NpManagerExports
{
    private const int NpTitleIdSize = 16;
    private const int NpTitleSecretSize = 128;
    private const int NpErrorInvalidArgument = unchecked((int)0x80550003);
    private const int NpErrorSignedOut = unchecked((int)0x80550006);

    // SceNpState. The ordering matters and is easy to get backwards: SIGNED_OUT is 1 and
    // SIGNED_IN is 2, so the value this emulator reports for a console with no PSN
    // session is 1. fpPS4 and shadPS4 report the same, which is the combination titles
    // are known to tolerate.
    internal const uint NpStateUnknown = 0;
    internal const uint NpStateSignedOut = 1;
    internal const uint NpStateSignedIn = 2;

    // SceNpReachabilityState: UNAVAILABLE means PSN cannot be used or cannot be
    // confirmed, which is what an emulator with no PSN session reports.
    internal const uint NpReachabilityUnavailable = 0;
    internal const uint NpReachabilityAvailable = 1;
    internal const uint NpReachabilityReachable = 2;
    private static int _nextNpRequestId;
    private static readonly ConcurrentDictionary<int, int> _requestResults = new();
    private static readonly ConcurrentDictionary<int, byte> _liveRequests = new();
    private static int _reachabilityCallbackRegistered;

    private const int NpErrorUserNotFound = unchecked((int)0x80550007);
    private const int NpErrorCallbackNotRegistered = unchecked((int)0x80550009);
    private const int NpErrorRequestNotFound = unchecked((int)0x80550014);
    private const int NpIdSize = 36; // SceNpId: online id (20), opt[8], reserved[8]
    private const int NpOnlineIdSize = 20; // data[16], term, dummy[3]

    [SysAbiExport(
        Nid = "3Zl8BePTh9Y",
        ExportName = "sceNpCheckCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpCheckCallback(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(Nid = "eiqMCt9UshI", ExportName = "sceNpCreateAsyncRequest", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNpManager")]
    public static int NpCreateAsyncRequest(CpuContext ctx) => SetReturn(ctx, NextLiveRequestId());

    [SysAbiExport(Nid = "KfGZg2y73oM", ExportName = "sceNpCheckNpReachability", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNpManager")]
    public static int NpCheckNpReachability(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        if (requestId <= 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        _requestResults[requestId] = NpErrorSignedOut;
        return SetReturn(ctx, NpErrorSignedOut);
    }

    // The account language and age are server requests like the reachability
    // check above, and the offline profile is signed out: each request fails
    // with SIGNED_OUT directly and through sceNpPollAsync.
    [SysAbiExport(Nid = "3Tcz5bNCfZQ", ExportName = "sceNpGetAccountLanguage2", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNpManager")]
    public static int NpGetAccountLanguage2(CpuContext ctx) => SignedOutAccountRequest(ctx);

    [SysAbiExport(Nid = "+4DegjBqV1g", ExportName = "sceNpGetAccountAge", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNpManager")]
    public static int NpGetAccountAge(CpuContext ctx) => SignedOutAccountRequest(ctx);

    private static int SignedOutAccountRequest(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        var userId = unchecked((int)ctx[CpuRegister.Rsi]);
        if (requestId <= 0 || userId == -1 || ctx[CpuRegister.Rdx] == 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        _requestResults[requestId] = NpErrorSignedOut;
        return SetReturn(ctx, NpErrorSignedOut);
    }

    [SysAbiExport(Nid = "GpLQDNKICac", ExportName = "sceNpCreateRequest", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNpManager")]
    public static int NpCreateRequest(CpuContext ctx) => SetReturn(ctx, NextLiveRequestId());

    [SysAbiExport(Nid = "OzKvTvg3ZYU", ExportName = "sceNpAbortRequest", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNpManager")]
    public static int NpAbortRequest(CpuContext ctx) => SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_OK);

    [SysAbiExport(Nid = "uqcPJLWL08M", ExportName = "sceNpPollAsync", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNpManager")]
    public static int NpPollAsync(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        var resultAddress = ctx[CpuRegister.Rsi];
        if (requestId <= 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        if (resultAddress != 0)
        {
            Span<byte> result = stackalloc byte[sizeof(int)];
            BinaryPrimitives.WriteInt32LittleEndian(result, _requestResults.GetValueOrDefault(requestId));
            if (!ctx.Memory.TryWrite(resultAddress, result))
            {
                return SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }
        }

        return SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_OK);
    }

    private static int NextLiveRequestId()
    {
        var id = NextRequestId();
        _liveRequests[id] = 0;
        return id;
    }

    private static int NextRequestId()
    {
        var id = Interlocked.Increment(ref _nextNpRequestId) & 0x7FFF_FFFF;
        return id == 0 ? 1 : id;
    }

    [SysAbiExport(
        Nid = "S7QTn72PrDw",
        ExportName = "sceNpDeleteRequest",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpDeleteRequest(CpuContext ctx)
    {
        _requestResults.TryRemove(unchecked((int)ctx[CpuRegister.Rdi]), out _);
        _liveRequests.TryRemove(unchecked((int)ctx[CpuRegister.Rdi]), out _);
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "JELHf4xPufo",
        ExportName = "sceNpCheckCallbackForLib",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpCheckCallbackForLib(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // Offline profile: the online id payload is left untouched and the call
    // reports success, matching the other offline NpManager stubs here.
    [SysAbiExport(
        Nid = "XDncXQIJUSk",
        ExportName = "sceNpGetOnlineId",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpGetOnlineId(CpuContext ctx)
    {
        // Gen5 ABI: user ID, then output structure.
        return WriteOfflineOnlineId(ctx, ctx[CpuRegister.Rsi]);
    }

    [SysAbiExport(
        Nid = "VfRSmPmj8Q8",
        ExportName = "sceNpRegisterStateCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpRegisterStateCallback(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    /// <summary>
    /// Accepts the reachability callback and never invokes it. Reachability
    /// transitions only ever fire on a real PSN connection, which an offline
    /// session does not have, so registering successfully and staying silent is
    /// the accurate emulation of a signed-out console rather than a stub.
    /// </summary>
    [SysAbiExport(
        Nid = "hw5KNqAAels",
        ExportName = "sceNpRegisterNpReachabilityStateCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpRegisterNpReachabilityStateCallback(CpuContext ctx)
    {
        Volatile.Write(ref _reachabilityCallbackRegistered, 1);
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    /// <summary>
    /// Accepts the premium-event callback and never invokes it. Offline sessions
    /// have no PS Plus / premium transitions to deliver, and leaving this NID
    /// unresolved returns NOT_FOUND which soft-locks titles that register it
    /// during settings / store probes (GTA V Enhanced).
    /// </summary>
    [SysAbiExport(
        Nid = "+yqjab2fUJA",
        ExportName = "sceNpRegisterPremiumEventCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpRegisterPremiumEventCallback(CpuContext ctx)
    {
        TraceNp(
            $"register_premium_event_callback cb=0x{ctx[CpuRegister.Rdi]:X16} " +
            $"userdata=0x{ctx[CpuRegister.Rsi]:X16}");
        return ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_OK);
    }

    [SysAbiExport(
        Nid = "qQJfO8HAiaY",
        ExportName = "sceNpRegisterStateCallbackA",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpRegisterStateCallbackA(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "0c7HbXRKUt4",
        ExportName = "sceNpRegisterStateCallbackForToolkit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManagerForToolkit")]
    public static int NpRegisterStateCallbackForToolkit(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "eQH7nWPcAgc",
        ExportName = "sceNpGetState",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpGetState(CpuContext ctx)
    {
        var stateAddress = ctx[CpuRegister.Rsi];
        if (stateAddress == 0)
        {
            return ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        // No PSN session: signed out. Reported alongside sceNpGetNpReachabilityState
        // UNAVAILABLE and sceNpCheckNpReachability SIGNED_OUT so the three agree.
        Span<byte> stateBytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(stateBytes, NpStateSignedOut);
        return ctx.Memory.TryWrite(stateAddress, stateBytes)
            ? ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_OK)
            : ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    [SysAbiExport(
        Nid = "rbknaUjpqWo",
        ExportName = "sceNpGetAccountIdA",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpGetAccountIdA(CpuContext ctx)
    {
        var userId = unchecked((int)ctx[CpuRegister.Rdi]);
        var accountIdAddress = ctx[CpuRegister.Rsi];
        if (userId == -1 || accountIdAddress == 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        // A stable local-only id, which is what titles that use the account id purely
        // as a save or profile key need. It is deliberately not gated on the signed-out
        // state sceNpGetState reports: titles built on Unity's PSN integration poll the
        // account id every frame and treat a failure as an exceptional state, so failing
        // here costs far more than the inconsistency.
        Span<byte> accountId = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(accountId, 1);
        return ctx.Memory.TryWrite(accountIdAddress, accountId)
            ? SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_OK)
            : SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    [SysAbiExport(
        Nid = "JT+t00a3TxA",
        ExportName = "sceNpGetAccountCountryA",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpGetAccountCountryA(CpuContext ctx)
    {
        var userId = unchecked((int)ctx[CpuRegister.Rdi]);
        var countryAddress = ctx[CpuRegister.Rsi];
        if (userId == -1 || countryAddress == 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        Span<byte> country = stackalloc byte[4];
        country[0] = (byte)'U';
        country[1] = (byte)'S';
        country[2] = 0;
        country[3] = 0;
        return ctx.Memory.TryWrite(countryAddress, country)
            ? SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_OK)
            : SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    [SysAbiExport(
        Nid = "e-ZuhGEoeC4",
        ExportName = "sceNpGetNpReachabilityState",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpGetNpReachabilityState(CpuContext ctx)
    {
        var userId = unchecked((int)ctx[CpuRegister.Rdi]);
        var stateAddress = ctx[CpuRegister.Rsi];
        if (userId == -1 || stateAddress == 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        Span<byte> state = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(state, NpReachabilityUnavailable);
        return ctx.Memory.TryWrite(stateAddress, state)
            ? SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_OK)
            : SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    [SysAbiExport(
        Nid = "Oad3rvY-NJQ",
        ExportName = "sceNpHasSignedUp",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpHasSignedUp(CpuContext ctx)
    {
        var resultAddress = ctx[CpuRegister.Rsi];
        if (resultAddress == 0 || unchecked((int)ctx[CpuRegister.Rdi]) == -1)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        Span<byte> signedUp = stackalloc byte[1];
        signedUp[0] = 0; // No online account is configured in the offline profile.
        return ctx.Memory.TryWrite(resultAddress, signedUp)
            ? SetReturn(ctx, 0)
            : SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    [SysAbiExport(
        Nid = "Ec63y59l9tw",
        ExportName = "sceNpSetNpTitleId",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpSetNpTitleId(CpuContext ctx)
    {
        var titleIdAddress = ctx[CpuRegister.Rdi];
        var titleSecretAddress = ctx[CpuRegister.Rsi];
        if (titleIdAddress == 0 || titleSecretAddress == 0)
        {
            return ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        Span<byte> titleId = stackalloc byte[NpTitleIdSize];
        Span<byte> titleSecret = stackalloc byte[NpTitleSecretSize];
        if (!ctx.Memory.TryRead(titleIdAddress, titleId) ||
            !ctx.Memory.TryRead(titleSecretAddress, titleSecret))
        {
            return ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        TraceNp($"set_np_title_id title='{ReadTitleId(titleId)}'");
        return ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_OK);
    }

    internal static void ResetForTests()
    {
        _requestResults.Clear();
        _liveRequests.Clear();
        Volatile.Write(ref _reachabilityCallbackRegistered, 0);
    }

    /// <summary>
    /// Gen5 ABI: request id, then the online id to look up. The only online id this
    /// emulator knows is the local profile's, which has no PSN session behind it, so it
    /// reports SIGNED_OUT (the answer sceNpCheckNpReachability gives); any other id matches
    /// no user. Both references report USER_NOT_FOUND for every id, so the local-profile
    /// case is the only deviation. As with sceNpCheckNpReachability the failure is returned
    /// directly and also kept for sceNpPollAsync.
    /// </summary>
    [SysAbiExport(
        Nid = "2rsFmlGWleQ",
        ExportName = "sceNpCheckNpAvailability",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpCheckNpAvailability(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        var onlineIdAddress = ctx[CpuRegister.Rsi];
        if (requestId <= 0 || onlineIdAddress == 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        Span<byte> onlineId = stackalloc byte[NpOnlineIdSize];
        if (!ctx.Memory.TryRead(onlineIdAddress, onlineId))
        {
            return SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var result = onlineId[..OfflineOnlineId.Length].SequenceEqual(OfflineOnlineId) &&
                     onlineId[OfflineOnlineId.Length] == 0
            ? NpErrorSignedOut
            : NpErrorUserNotFound;
        _requestResults[requestId] = result;
        return SetReturn(ctx, result);
    }

    /// <summary>
    /// Gen5 ABI: user id, then a SceNpId output (36 bytes). Reports the local profile, the
    /// same online id sceNpGetOnlineId writes, with the opt and reserved bytes zeroed.
    /// </summary>
    [SysAbiExport(
        Nid = "p-o74CnoNzY",
        ExportName = "sceNpGetNpId",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpGetNpId(CpuContext ctx)
    {
        var userId = unchecked((int)ctx[CpuRegister.Rdi]);
        var npIdAddress = ctx[CpuRegister.Rsi];
        if (userId == -1 || npIdAddress == 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        Span<byte> npId = stackalloc byte[NpIdSize];
        npId.Clear();
        OfflineOnlineId.CopyTo(npId);
        return ctx.Memory.TryWrite(npIdAddress, npId)
            ? SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_OK)
            : SetReturn(ctx, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    /// <summary>
    /// Gen5 ABI: request id, parameter, result output. A premium (PS Plus) check needs a PSN
    /// session, so it fails SIGNED_OUT without writing the result, like the other online
    /// queries. Null arguments are INVALID_ARGUMENT.
    /// </summary>
    [SysAbiExport(
        Nid = "O80NrhUOPGY",
        ExportName = "sceNpCheckPremium",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpCheckPremium(CpuContext ctx)
    {
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        if (requestId <= 0 || ctx[CpuRegister.Rsi] == 0 || ctx[CpuRegister.Rdx] == 0)
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        _requestResults[requestId] = NpErrorSignedOut;
        return SetReturn(ctx, NpErrorSignedOut);
    }

    /// <summary>
    /// Gen5 ABI: request id, user id, language output. The account language comes from the
    /// PSN account, which a signed-out profile does not have, so this fails SIGNED_OUT without
    /// writing the output. (The output layout is not confirmed by either reference.)
    /// </summary>
    /// <summary>
    /// Gen5 ABI: request id, resolve retry count, then resolve, connect, send and receive
    /// timeouts in microseconds. The validation follows shadPS4 via the anyps5 notes and is
    /// not checked on a console: a non-positive request id, a negative retry count, all five
    /// values zero, a non-zero resolve timeout under 1 s or any other non-zero timeout under
    /// 10 s is INVALID_ARGUMENT; a request that was never created or was deleted is
    /// REQUEST_NOT_FOUND. There is no network, so the values are not stored.
    /// </summary>
    [SysAbiExport(
        Nid = "-QglDeRr8D8",
        ExportName = "sceNpSetTimeout",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpSetTimeout(CpuContext ctx)
    {
        const uint minResolveTimeout = 1_000_000;
        const uint minTransferTimeout = 10_000_000;
        var requestId = unchecked((int)ctx[CpuRegister.Rdi]);
        var resolveRetry = unchecked((int)ctx[CpuRegister.Rsi]);
        var resolveTimeout = unchecked((uint)ctx[CpuRegister.Rdx]);
        var connectTimeout = unchecked((uint)ctx[CpuRegister.Rcx]);
        var sendTimeout = unchecked((uint)ctx[CpuRegister.R8]);
        var receiveTimeout = unchecked((uint)ctx[CpuRegister.R9]);
        static bool Below(uint timeout, uint minimum) => timeout != 0 && timeout < minimum;
        if (requestId <= 0 ||
            resolveRetry < 0 ||
            (resolveRetry == 0 && resolveTimeout == 0 && connectTimeout == 0 && sendTimeout == 0 && receiveTimeout == 0) ||
            Below(resolveTimeout, minResolveTimeout) ||
            Below(connectTimeout, minTransferTimeout) ||
            Below(sendTimeout, minTransferTimeout) ||
            Below(receiveTimeout, minTransferTimeout))
        {
            return SetReturn(ctx, NpErrorInvalidArgument);
        }

        return SetReturn(ctx, _liveRequests.ContainsKey(requestId) ? 0 : NpErrorRequestNotFound);
    }

    /// <summary>Accepts the PS Plus event callback and never invokes it (no PS Plus offline).</summary>
    [SysAbiExport(
        Nid = "GImICnh+boA",
        ExportName = "sceNpRegisterPlusEventCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpRegisterPlusEventCallback(CpuContext ctx) =>
        SetReturn(ctx, ctx[CpuRegister.Rdi] == 0 ? NpErrorInvalidArgument : 0);

    /// <summary>
    /// Gen5 ABI: account id, user id output. The local account id sceNpGetAccountIdA hands
    /// out has no user mapping behind it here, so a valid id reports SIGNED_OUT without writing
    /// the output (as shadPS4 and anyps5 do); account id 0 or a null output is INVALID_ARGUMENT.
    /// </summary>
    [SysAbiExport(
        Nid = "VgYczPGB5ss",
        ExportName = "sceNpGetUserIdByAccountId",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpGetUserIdByAccountId(CpuContext ctx) =>
        SetReturn(
            ctx,
            ctx[CpuRegister.Rdi] == 0 || ctx[CpuRegister.Rsi] == 0 ? NpErrorInvalidArgument : NpErrorSignedOut);

    // The unregister calls take no arguments. Only the reachability callback is tracked;
    // unregistering it when none is registered is CALLBACK_NOT_REGISTERED (both references).
    // The other three return success either way (anyps5), since a title tearing down after a
    // register that failed or never ran must not be failed for it.
    [SysAbiExport(
        Nid = "cRILAEvn+9M",
        ExportName = "sceNpUnregisterNpReachabilityStateCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpUnregisterNpReachabilityStateCallback(CpuContext ctx) =>
        SetReturn(
            ctx,
            Interlocked.Exchange(ref _reachabilityCallbackRegistered, 0) == 0 ? NpErrorCallbackNotRegistered : 0);

    [SysAbiExport(
        Nid = "mjjTXh+NHWY",
        ExportName = "sceNpUnregisterStateCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpUnregisterStateCallback(CpuContext ctx) => SetReturn(ctx, 0);

    [SysAbiExport(
        Nid = "M3wFXbYQtAA",
        ExportName = "sceNpUnregisterStateCallbackA",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpUnregisterStateCallbackA(CpuContext ctx) => SetReturn(ctx, 0);

    [SysAbiExport(
        Nid = "-Rjp3-YViXc",
        ExportName = "sceNpUnregisterPremiumEventCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpUnregisterPremiumEventCallback(CpuContext ctx) => SetReturn(ctx, 0);

    [SysAbiExport(
        Nid = "xViqJdDgKl0",
        ExportName = "sceNpUnregisterPlusEventCallback",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNpManager")]
    public static int NpUnregisterPlusEventCallback(CpuContext ctx) => SetReturn(ctx, 0);

    private static ReadOnlySpan<byte> OfflineOnlineId => "Player"u8;

    private static int SetReturn(CpuContext ctx, int result)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)result);
        return result;
    }

    private static string ReadTitleId(ReadOnlySpan<byte> bytes)
    {
        var length = 0;
        while (length < 12 && length < bytes.Length && bytes[length] != 0)
        {
            length++;
        }

        return length == 0
            ? string.Empty
            : System.Text.Encoding.ASCII.GetString(bytes[..length]);
    }

    private static void TraceNp(string message)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_NP"), "1", StringComparison.Ordinal))
        {
            return;
        }

        Console.Error.WriteLine($"[LOADER][TRACE] np.{message}");
    }

    private static int WriteOfflineOnlineId(CpuContext ctx, ulong address)
    {
        if (address == 0)
        {
            return ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        // SceNpOnlineId is a 16-byte handle plus four trailing bytes.
        Span<byte> onlineId = stackalloc byte[20];
        "Player"u8.CopyTo(onlineId);
        return ctx.Memory.TryWrite(address, onlineId)
            ? ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_OK)
            : ctx.SetReturn(OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }
}
