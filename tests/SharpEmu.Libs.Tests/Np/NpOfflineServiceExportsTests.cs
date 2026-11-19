// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Np;
using Xunit;

namespace SharpEmu.Libs.Tests.Np;

// NpManager, NpEntitlementAccess and NpCommerce answers for an offline profile with a local
// account and no PSN session. They share static state, so they run serially.
[Collection("NpCommerce")]
public sealed class NpOfflineServiceExportsTests : IDisposable
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const int InvalidArgument = unchecked((int)0x80550003);
    private const int SignedOut = unchecked((int)0x80550006);
    private const int UserNotFound = unchecked((int)0x80550007);
    private const int CallbackNotRegistered = unchecked((int)0x80550009);
    private const int RequestNotFound = unchecked((int)0x80550014);
    private const int EntitlementParameter = unchecked((int)0x817D0002);
    // libSceNpEntitlementAccess reports a signed-out profile with its own code (upstream's error model).
    private const int EntitlementSignedOut = unchecked((int)0x817D0014);
    private const int EntitlementRequestNotFound = unchecked((int)0x817D0015);
    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x1_0000);
    private readonly CpuContext _ctx;

    public NpOfflineServiceExportsTests()
    {
        NpManagerExports.ResetForTests();
        NpCommerceDialogState.ResetForTests();
        NpCommerceExports.ResetIconForTests();
        _ctx = new CpuContext(_memory, Generation.Gen5);
    }

    public void Dispose()
    {
        NpManagerExports.ResetForTests();
        NpCommerceDialogState.ResetForTests();
        NpCommerceExports.ResetIconForTests();
    }

    private void Args(ulong rdi = 0, ulong rsi = 0, ulong rdx = 0, ulong rcx = 0, ulong r8 = 0, ulong r9 = 0)
    {
        _ctx[CpuRegister.Rdi] = rdi;
        _ctx[CpuRegister.Rsi] = rsi;
        _ctx[CpuRegister.Rdx] = rdx;
        _ctx[CpuRegister.Rcx] = rcx;
        _ctx[CpuRegister.R8] = r8;
        _ctx[CpuRegister.R9] = r9;
    }

    private byte[] Read(ulong address, int length)
    {
        var bytes = new byte[length];
        Assert.True(_memory.TryRead(address, bytes));
        return bytes;
    }

    private ulong ReadUInt64(ulong address)
    {
        Assert.True(_ctx.TryReadUInt64(address, out var value));
        return value;
    }

    // ---- NpManager ----

    [Fact]
    public void GetNpId_WritesTheLocalProfileThatGetOnlineIdReports()
    {
        var npId = MemoryBase + 0x100;
        var onlineId = MemoryBase + 0x200;
        _memory.TryWrite(npId, Enumerable.Repeat((byte)0xAB, 36).ToArray());
        Args(1, npId);
        Assert.Equal(0, NpManagerExports.NpGetNpId(_ctx));
        var written = Read(npId, 36);
        Args(1, onlineId);
        Assert.Equal(0, NpManagerExports.NpGetOnlineId(_ctx));
        Assert.Equal(Read(onlineId, 20), written[..20]);
        Assert.All(written[20..], b => Assert.Equal(0, b));

        Args(1, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpGetNpId(_ctx));
        Args(unchecked((ulong)(uint)-1), npId);
        Assert.Equal(InvalidArgument, NpManagerExports.NpGetNpId(_ctx));
    }

    [Fact]
    public void OnlineQueries_FailSignedOutAndLeaveOutputsUntouched()
    {
        var requestId = NpManagerExports.NpCreateAsyncRequest(_ctx);
        var output = MemoryBase + 0x300;
        _memory.TryWrite(output, [0x5A, 0x5A, 0x5A, 0x5A]);

        Args((ulong)requestId, MemoryBase + 0x400, output);
        Assert.Equal(SignedOut, NpManagerExports.NpCheckPremium(_ctx));
        Args((ulong)requestId, 1, output);
        Assert.Equal(SignedOut, NpManagerExports.NpGetAccountLanguage2(_ctx));
        Args(1, output);
        Assert.Equal(SignedOut, NpManagerExports.NpGetUserIdByAccountId(_ctx));
        Assert.All(Read(output, 4), b => Assert.Equal(0x5A, b));

        // The failure is also what sceNpPollAsync reports for the request.
        var result = MemoryBase + 0x500;
        Args((ulong)requestId, result);
        Assert.Equal(0, NpManagerExports.NpPollAsync(_ctx));
        Assert.Equal(SignedOut, BinaryPrimitives.ReadInt32LittleEndian(Read(result, 4)));

        Args((ulong)requestId, 0, output);
        Assert.Equal(InvalidArgument, NpManagerExports.NpCheckPremium(_ctx));
        Args(0, MemoryBase + 0x400, output);
        Assert.Equal(InvalidArgument, NpManagerExports.NpCheckPremium(_ctx));
        Args((ulong)requestId, 1, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpGetAccountLanguage2(_ctx));
        Args(0, output);
        Assert.Equal(InvalidArgument, NpManagerExports.NpGetUserIdByAccountId(_ctx));
        Args(1, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpGetUserIdByAccountId(_ctx));
    }

    [Fact]
    public void CheckNpAvailability_SeparatesTheLocalProfileFromUnknownOnlineIds()
    {
        var requestId = NpManagerExports.NpCreateAsyncRequest(_ctx);
        var local = MemoryBase + 0x100;
        var other = MemoryBase + 0x200;
        _memory.TryWrite(local, "Player"u8);
        _memory.TryWrite(other, "SomeoneElse"u8);

        Args((ulong)requestId, local);
        Assert.Equal(SignedOut, NpManagerExports.NpCheckNpAvailability(_ctx));
        Args((ulong)requestId, other);
        Assert.Equal(UserNotFound, NpManagerExports.NpCheckNpAvailability(_ctx));
        Args((ulong)requestId, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpCheckNpAvailability(_ctx));
        Args(0, local);
        Assert.Equal(InvalidArgument, NpManagerExports.NpCheckNpAvailability(_ctx));
    }

    [Fact]
    public void SetTimeout_ValidatesValuesAndTracksRequestLifetime()
    {
        var requestId = (ulong)NpManagerExports.NpCreateRequest(_ctx);

        Args(requestId, 3, 1_000_000, 10_000_000, 10_000_000, 10_000_000);
        Assert.Equal(0, NpManagerExports.NpSetTimeout(_ctx));
        Args(requestId, 0, 0, 0, 0, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpSetTimeout(_ctx));
        Args(requestId, unchecked((ulong)(uint)-1), 1_000_000, 0, 0, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpSetTimeout(_ctx));
        Args(requestId, 1, 999_999, 0, 0, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpSetTimeout(_ctx));
        Args(requestId, 1, 0, 9_999_999, 0, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpSetTimeout(_ctx));
        Args(0, 1, 1_000_000, 0, 0, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpSetTimeout(_ctx));

        Args(requestId);
        Assert.Equal(0, NpManagerExports.NpDeleteRequest(_ctx));
        Args(requestId, 3, 1_000_000, 10_000_000, 10_000_000, 10_000_000);
        Assert.Equal(RequestNotFound, NpManagerExports.NpSetTimeout(_ctx));
        Args(12345, 3, 1_000_000, 10_000_000, 10_000_000, 10_000_000);
        Assert.Equal(RequestNotFound, NpManagerExports.NpSetTimeout(_ctx));
    }

    [Fact]
    public void CallbackFamily_RegistersPlusAndUnregistersEachKind()
    {
        Args(0, 0);
        Assert.Equal(InvalidArgument, NpManagerExports.NpRegisterPlusEventCallback(_ctx));
        Args(0x1234, 0x5678);
        Assert.Equal(0, NpManagerExports.NpRegisterPlusEventCallback(_ctx));

        Assert.Equal(CallbackNotRegistered, NpManagerExports.NpUnregisterNpReachabilityStateCallback(_ctx));
        Args(0x1234, 0);
        Assert.Equal(0, NpManagerExports.NpRegisterNpReachabilityStateCallback(_ctx));
        Assert.Equal(0, NpManagerExports.NpUnregisterNpReachabilityStateCallback(_ctx));
        Assert.Equal(CallbackNotRegistered, NpManagerExports.NpUnregisterNpReachabilityStateCallback(_ctx));

        Assert.Equal(0, NpManagerExports.NpUnregisterStateCallback(_ctx));
        Assert.Equal(0, NpManagerExports.NpUnregisterStateCallbackA(_ctx));
        Assert.Equal(0, NpManagerExports.NpUnregisterPremiumEventCallback(_ctx));
        Assert.Equal(0, NpManagerExports.NpUnregisterPlusEventCallback(_ctx));
    }

    // ---- NpEntitlementAccess ----

    [Fact]
    public void EntitlementRequests_FailSignedOutAtTheRequestStageAndPollReportsTheSame()
    {
        // The requests whose request-id argument no reference or title pins create no request.
        var calls = new Func<CpuContext, int>[]
        {
            NpEntitlementAccessExports.NpEntitlementAccessRequestServiceEntitlementInfo,
            NpEntitlementAccessExports.NpEntitlementAccessRequestUnifiedEntitlementInfo,
            NpEntitlementAccessExports.NpEntitlementAccessRequestConsumeServiceEntitlement,
            NpEntitlementAccessExports.NpEntitlementAccessRequestConsumeUnifiedEntitlement,
            NpEntitlementAccessExports.NpEntitlementAccessPollServiceEntitlementInfo,
            NpEntitlementAccessExports.NpEntitlementAccessPollUnifiedEntitlementInfo,
            NpEntitlementAccessExports.NpEntitlementAccessPollConsumeEntitlement,
        };
        var output = MemoryBase + 0x100;
        _memory.TryWrite(output, [0x7E, 0x7E, 0x7E, 0x7E]);
        foreach (var call in calls)
        {
            Args(1, 2, output, 4);
            Assert.Equal(EntitlementSignedOut, call(_ctx));
            Assert.Equal(unchecked((ulong)EntitlementSignedOut), _ctx[CpuRegister.Rax]);
        }

        Assert.All(Read(output, 4), b => Assert.Equal(0x7E, b));
    }

    // The unified entitlement list request and its poll keep upstream's request lifecycle
    // (NpEntitlementAccessExportsTests): abort and delete name a request, and one never made is not found.
    [Fact]
    public void EntitlementRequestLifecycle_AbortAndDeleteReportARequestNeverMade()
    {
        Args(0x7FFF_FFFF_0000);
        Assert.Equal(EntitlementRequestNotFound, NpEntitlementAccessExports.NpEntitlementAccessAbortRequest(_ctx));
        Assert.Equal(EntitlementRequestNotFound, NpEntitlementAccessExports.NpEntitlementAccessDeleteRequest(_ctx));
    }

    // The service entitlement list request follows the unified list's lifecycle (Silent Hill: The Short Message
    // calls both pairs with the same arguments): it finishes at once, signed out, with no entitlements.
    [Fact]
    public void ServiceEntitlementListRequest_FinishesSignedOutUntilDeleted()
    {
        var requestIdAddress = MemoryBase + 0x300;
        Args(0x1000_0000, 0, 0, 0, MemoryBase + 0x380, requestIdAddress);
        Assert.Equal(0, NpEntitlementAccessExports.NpEntitlementAccessRequestServiceEntitlementInfoList(_ctx));
        var requestId = BinaryPrimitives.ReadUInt64LittleEndian(Read(requestIdAddress, 8));
        Assert.NotEqual(0UL, requestId);

        var result = MemoryBase + 0x400;
        _memory.TryWrite(result, Enumerable.Repeat((byte)0xA5, 12).ToArray());
        Args(requestId, result, MemoryBase + 0x500, 100, result + 4, result + 8);
        Assert.Equal(0, NpEntitlementAccessExports.NpEntitlementAccessPollServiceEntitlementInfoList(_ctx));
        var written = Read(result, 12);
        Assert.Equal(EntitlementSignedOut, BinaryPrimitives.ReadInt32LittleEndian(written));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(written.AsSpan(4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(written.AsSpan(8)));

        Args(requestId);
        Assert.Equal(0, NpEntitlementAccessExports.NpEntitlementAccessDeleteRequest(_ctx));
        Args(requestId, result);
        Assert.Equal(EntitlementRequestNotFound, NpEntitlementAccessExports.NpEntitlementAccessPollServiceEntitlementInfoList(_ctx));
    }

    [Fact]
    public void GenerateTransactionIdAndPftFlag_ValidateTheirOutputPointers()
    {
        Args(0);
        Assert.Equal(EntitlementParameter, NpEntitlementAccessExports.NpEntitlementAccessGenerateTransactionId(_ctx));
        Args(MemoryBase + 0x100);
        Assert.Equal(EntitlementSignedOut, NpEntitlementAccessExports.NpEntitlementAccessGenerateTransactionId(_ctx));

        var flag = MemoryBase + 0x200;
        _memory.TryWrite(flag, [0xFF, 0xFF, 0xFF, 0xFF]);
        Args(flag);
        Assert.Equal(0, NpEntitlementAccessExports.NpEntitlementAccessGetPftFlag(_ctx));
        Assert.Equal(0U, BinaryPrimitives.ReadUInt32LittleEndian(Read(flag, 4)));
        Args(0);
        Assert.Equal(EntitlementParameter, NpEntitlementAccessExports.NpEntitlementAccessGetPftFlag(_ctx));
    }

    // ---- NpCommerce ----

    [Fact]
    public void DialogClose_AndGetStatus_FollowTheStatusMachine()
    {
        Assert.Equal(NpCommerceDialogState.ErrorNotInitialized, NpCommerceExports.NpCommerceDialogClose(_ctx));
        Assert.Equal(0, NpCommerceExports.NpCommerceDialogGetStatus(_ctx));

        Assert.Equal(0, NpCommerceExports.NpCommerceDialogInitialize(_ctx));
        Assert.Equal(1, NpCommerceExports.NpCommerceDialogGetStatus(_ctx));
        Assert.Equal(NpCommerceDialogState.ErrorNotRunning, NpCommerceExports.NpCommerceDialogClose(_ctx));

        Assert.Equal(0, NpCommerceDialogState.Open(Request()));
        Assert.Equal(NpCommerceDialogStatus.Running, NpCommerceDialogState.Status);
        Assert.Equal(0, NpCommerceExports.NpCommerceDialogClose(_ctx));
        Assert.Equal(NpCommerceDialogStatus.Finished, NpCommerceDialogState.Status);
        Assert.Equal(3, NpCommerceExports.NpCommerceDialogGetStatus(_ctx));
        Assert.Equal(NpCommerceDialogState.ErrorNotRunning, NpCommerceExports.NpCommerceDialogClose(_ctx));
        Assert.Equal(0, NpCommerceDialogState.TryGetResult(out _));

        // GetStatus alone also carries a running dialog to FINISHED (it polls like UpdateStatus).
        Assert.Equal(0, NpCommerceExports.NpCommerceDialogTerminate(_ctx));
        Assert.Equal(0, NpCommerceExports.NpCommerceDialogInitialize(_ctx));
        Assert.Equal(0, NpCommerceDialogState.Open(Request()));
        Assert.Equal(3, NpCommerceExports.NpCommerceDialogGetStatus(_ctx));
    }

    [Fact]
    public void PsStoreIcon_RecordsVisibilityPositionAndLayout()
    {
        Assert.False(NpCommerceExports.PsStoreIconVisible);
        Args(2);
        Assert.Equal(0, NpCommerceExports.NpCommerceShowPsStoreIcon(_ctx));
        Assert.True(NpCommerceExports.PsStoreIconVisible);
        Assert.Equal(2, NpCommerceExports.PsStoreIconPosition);
        Args(5);
        Assert.Equal(0, NpCommerceExports.NpCommerceSetPsStoreIconLayout(_ctx));
        Assert.Equal(5, NpCommerceExports.PsStoreIconLayout);
        Assert.Equal(0, NpCommerceExports.NpCommerceHidePsStoreIcon(_ctx));
        Assert.False(NpCommerceExports.PsStoreIconVisible);
    }

    private static NpCommerceDialogRequest Request() =>
        new(IsOpen2: false, UserId: 1, Mode: 1, ServiceLabel: 0, Features: 0, UserData: 0, Targets: []);

    // ---- NpUniversalDataSystem additions ----

    [Fact]
    public void Registration_CoversTheNewExports()
    {
        var manager = new ModuleManager();
        manager.RegisterExports(SharpEmu.Generated.SysAbiExportRegistry.CreateExports(Generation.Gen5));
        foreach (var (nid, name, library) in new[]
                 {
                     ("2rsFmlGWleQ", "sceNpCheckNpAvailability", "libSceNpManager"),
                     ("p-o74CnoNzY", "sceNpGetNpId", "libSceNpManager"),
                     ("O80NrhUOPGY", "sceNpCheckPremium", "libSceNpManager"),
                     ("3Tcz5bNCfZQ", "sceNpGetAccountLanguage2", "libSceNpManager"),
                     ("-QglDeRr8D8", "sceNpSetTimeout", "libSceNpManager"),
                     ("GImICnh+boA", "sceNpRegisterPlusEventCallback", "libSceNpManager"),
                     ("VgYczPGB5ss", "sceNpGetUserIdByAccountId", "libSceNpManager"),
                     ("cRILAEvn+9M", "sceNpUnregisterNpReachabilityStateCallback", "libSceNpManager"),
                     ("mjjTXh+NHWY", "sceNpUnregisterStateCallback", "libSceNpManager"),
                     ("M3wFXbYQtAA", "sceNpUnregisterStateCallbackA", "libSceNpManager"),
                     ("-Rjp3-YViXc", "sceNpUnregisterPremiumEventCallback", "libSceNpManager"),
                     ("xViqJdDgKl0", "sceNpUnregisterPlusEventCallback", "libSceNpManager"),
                     ("HFcQl9TMcFQ", "sceNpEntitlementAccessAbortRequest", "libSceNpEntitlementAccess"),
                     ("Z0eQj8m7XA8", "sceNpEntitlementAccessDeleteRequest", "libSceNpEntitlementAccess"),
                     ("IQtb-TaIjSM", "sceNpEntitlementAccessGenerateTransactionId", "libSceNpEntitlementAccess"),
                     ("eDXKe9FndlE", "sceNpEntitlementAccessGetPftFlag", "libSceNpEntitlementAccess"),
                     ("M8icY9OwkKs", "sceNpEntitlementAccessRequestServiceEntitlementInfo", "libSceNpEntitlementAccess"),
                     ("nAEqawEZG5s", "sceNpEntitlementAccessPollUnifiedEntitlementInfoList", "libSceNpEntitlementAccess"),
                     ("NU3ckGHMFXo", "sceNpCommerceDialogClose", "libSceNpCommerce"),
                     ("CCbC+lqqvF0", "sceNpCommerceDialogGetStatus", "libSceNpCommerce"),
                     ("DHmwsa6S8Tc", "sceNpCommerceShowPsStoreIcon", "libSceNpCommerce"),
                     ("dsqCVsNM0Zg", "sceNpCommerceHidePsStoreIcon", "libSceNpCommerce"),
                     ("uKTDW8hk-ts", "sceNpCommerceSetPsStoreIconLayout", "libSceNpCommerce"),
                     ("+s14jq-KGYw", "sceNpUniversalDataSystemEventEstimateSize", "libSceNpUniversalDataSystem"),
                     ("BypQuF113-k", "sceNpUniversalDataSystemEventPropertyArraySetInt32", "libSceNpUniversalDataSystem"),
                     ("sbSYZLR5AiE", "sceNpUniversalDataSystemEventPropertyArraySetFloat64", "libSceNpUniversalDataSystem"),
                 })
        {
            Assert.True(manager.TryGetExport(nid, out var export), $"NID {nid} did not register.");
            Assert.Equal(name, export.Name);
            Assert.Equal(library, export.LibraryName);
        }
    }
}
