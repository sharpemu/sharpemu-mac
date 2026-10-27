// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;

namespace SharpEmu.GpuSplit;

/// <summary>
/// The guest process's image store: the GPU process's caches, reached over the transport. The
/// guest's GPU memory manager calls it exactly as it calls the Metal image cache in one process.
/// </summary>
/// <remarks>
/// <para><b>Write faults</b> (<see cref="MarkCpuWrite"/>) come from the signal handler: one
/// <see cref="GpuRemoteSyncOpcode.Fault"/> call, allocation-free and lock-free, answered once the
/// GPU process's stores handled the store and the protections it dropped are applied. Every fault
/// goes there, interest summary or not: the page faulted because the GPU process protected it.</para>
///
/// <para><b>Managed writes</b> are filtered by the GPU-interest summary first; GTA V makes over a
/// million a second and the caches watch a few pages. One about to land
/// (<see cref="MarkPreciseCpuWrite"/>) is a call when the pending-writeback summary has a page
/// under it, because a GPU-written image there must reach guest memory, or a writeback recorded
/// there must land, before the write lands over it. Under no such page the call would only
/// invalidate, which the write's ordered record does as well, so it is sent as a record
/// (<see cref="SplitSharedState"/>, "Pending-writeback summary"). From a thread that is answering the GPU process
/// or that runs inside a GPU worker bracket it is sent as a landed write instead: there the GPU
/// worker cannot take the image's writeback, which is the in-process cache's own rule for a store
/// from the worker. One that has landed (<see cref="MarkCpuWriteLanded"/>) is a record, handled
/// in ring order before any later submission.</para>
///
/// <para>Span changes (<see cref="IGuestSpanObserver"/>) and unregistrations are calls, so the
/// GPU process's spans and ledger change in step with this process's.</para>
/// </remarks>
[SupportedOSPlatform("macos")]
internal sealed unsafe class RemoteImageStore : IGuestImageStore, IGuestSpanObserver
{
    private const long FaultTimeoutNanoseconds = 60_000_000_000;

    private readonly SplitGpuSession _session;
    private readonly SyncCallTable _calls;
    private readonly SplitSharedState _state;

    public RemoteImageStore(SplitGpuSession session, SyncCallTable calls, SplitSharedState state)
    {
        _session = session;
        _calls = calls;
        _state = state;
    }

    // The last released-write retry of this thread, so an immediate repeat needs a new restoration
    // (the guard of GuestGpuMemory.TryRetryReleasedWrite, which keeps it per thread as well).
    [ThreadStatic]
    private static ulong t_retriedPage;

    [ThreadStatic]
    private static long t_retriedVersion;

    // From the guest's write-fault signal handler: nothing here may allocate, lock or throw.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool MarkCpuWrite(ulong address, ulong size)
    {
        FaultRequest request;
        request.Kind = (uint)FaultKind.Write;
        request.Reserved = 0;
        request.Address = address;
        request.Size = size;
        HandledResponse response;
        response.Handled = 0;
        response.Reserved = 0;
        response.RestorationVersion = 0;
        var result = _calls.Call(
            (uint)GpuRemoteSyncOpcode.Fault,
            (byte*)&request,
            sizeof(FaultRequest),
            (byte*)&response,
            sizeof(HandledResponse),
            out var length,
            out var status,
            FaultTimeoutNanoseconds);
        if (result != SyncCallStatus.Completed || status != GpuSplitStatus.Ok || length < sizeof(HandledResponse))
        {
            return false;
        }

        if (response.Handled != 0)
        {
            return true;
        }

        // The GPU process removed the page's watch before the fault reached it, and its protection
        // is applied by now: the store may be retried, once per restoration, when the page is
        // writable here.
        var page = address & ~(TrackerLayout.PageBytes - 1);
        if (response.RestorationVersion == 0 ||
            (t_retriedPage == page && t_retriedVersion == response.RestorationVersion) ||
            (_session.GuestSpace is IGuestBackedSpace backed && !backed.AllowsMappedAccess(address, GuestPageProtection.Write)))
        {
            return false;
        }

        t_retriedPage = page;
        t_retriedVersion = response.RestorationVersion;
        return true;
    }

    public bool MarkPreciseCpuWrite(ulong address, ulong size)
    {
        if (!_state.MayInterest(address, size))
        {
            return false;
        }

        SplitCrossingStats.Count(SplitCrossing.ManagedWriteInterest);
        if (RemoteGpuQueueRelay.CurrentBracket != 0 || SplitGpuSession.OnReverseServerThread)
        {
            return SendLanded(address, size);
        }

        // Nothing GPU-written to put under the write: the record (ahead of every later submission)
        // invalidates as the call would. Sent now, as well as after the write lands, for a caller
        // that reports only the write about to land.
        if (!_state.MayPendWriteback(address, size))
        {
            SplitCrossingStats.Count(SplitCrossing.ManagedWriteRecord);
            return SendLanded(address, size);
        }

        // No barrier: in one process a submission this thread enqueued before may still wait for
        // the worker as the write is published, so the GPU caches need nothing more from the ring
        // here, and a managed write does not wait for a ring thread held by the shader prewarm.
        var request = new CpuWriteRequest { Barrier = 0, Address = address, Size = size };
        var start = SplitCrossingStats.Enabled ? Stopwatch.GetTimestamp() : 0;
        var handled = _session.Call(GpuRemoteSyncOpcode.CpuWrite, request, out HandledResponse response, SplitGpuSession.CallNanoseconds) &&
            response.Handled != 0;
        if (SplitCrossingStats.Enabled)
        {
            SplitCrossingStats.Count(SplitCrossing.ManagedWriteCall);
            SplitCrossingStats.Time(SplitCrossing.ManagedWriteCall, Stopwatch.GetTimestamp() - start);
        }

        return handled;
    }

    public bool MarkCpuWriteLanded(ulong address, ulong size) =>
        _state.MayInterest(address, size) && SendLanded(address, size);

    public void Unregister(ulong address, ulong size) =>
        _ = _session.Call(GpuRemoteSyncOpcode.SpanUnregister, new SpanRequest
        {
            Address = address,
            Size = size,
            BracketId = RemoteGpuQueueRelay.CurrentBracket,
        }, SplitGpuSession.LongCallNanoseconds);

    // The GPU process's page guard owns the host protections: it derives them from the ledger these
    // calls keep equal to this process's, with its watches, and they are applied here before the
    // call answers. When the call fails this process's own guard applies the plain guest ones.
    public bool OnSpanRegistered(ulong address, ulong size, GuestPageProtection protection) =>
        _session.Call(GpuRemoteSyncOpcode.SpanRegister, new SpanRequest
        {
            Address = address,
            Size = size,
            Protection = (uint)protection,
            BracketId = RemoteGpuQueueRelay.CurrentBracket,
        }, SplitGpuSession.LongCallNanoseconds);

    public bool OnSpanProtected(ulong address, ulong size, GuestPageProtection protection) =>
        _session.Call(GpuRemoteSyncOpcode.SpanProtect, new SpanRequest
        {
            Address = address,
            Size = size,
            Protection = (uint)protection,
            BracketId = RemoteGpuQueueRelay.CurrentBracket,
        }, out HandledResponse response, SplitGpuSession.LongCallNanoseconds) && response.Handled != 0;

    /// <summary>
    /// Compiles the fault path and makes one call through it from an ordinary thread, so the
    /// first fault in a signal handler finds everything resolved.
    /// </summary>
    public void PrepareForSignalContext()
    {
        SyncCallTable.PrepareForSignalContext();
        RuntimeHelpers.PrepareMethod(typeof(RemoteImageStore).GetMethod(nameof(MarkCpuWrite))!.MethodHandle);
        // An address no store holds: the GPU process answers "not handled".
        _ = ((IGuestImageStore)this).MarkCpuWrite(0, 8);
    }

    private bool SendLanded(ulong address, ulong size) =>
        _session.Writer.Write(GpuRemoteMessage.CpuWrite, new CpuWriteMessage { Address = address, Size = size });
}
