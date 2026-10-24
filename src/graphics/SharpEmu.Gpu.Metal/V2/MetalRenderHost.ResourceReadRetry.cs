// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    // Only a stalled compute materialization pays for the exception/callback. It has
    // not bound resources or encoded its dispatch, so another queue can run while its
    // already-committed producer finishes. The retry re-reads every required byte.
    internal bool AsyncResourceReads { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ASYNC_RESOURCE_READS") != "0";

    private bool _resourceReadRetryAllowed;
    private ulong _pendingResourceReadSerial;
    private ulong _resourceReadReadySerial;
    private ulong _resourceReadRetiredSerial;
    private long _resourceReadSuspends;
    private long _resourceReadDisabled;
    private long _resourceReadScopeRefused;
    private long _resourceReadUncommittedRefused;
    private long _resourceReadImageRefused;
    private long _resourceReadWakeups;
    private long _resourceReadDeferredTicks;

    private sealed class ResourceReadPending(ulong serial) : Exception
    {
        public ulong Serial { get; } = serial;
    }

    // A buffer may stay open across the suspension. Advance its cached completion
    // floor too, as the old blocking wait did, so rings/copies are not kept lent just
    // because no new command buffer was opened. No native event read is needed here.
    private void RetireCompletedResourceReads()
    {
        var completed = Volatile.Read(ref _resourceReadReadySerial);
        if (completed <= _resourceReadRetiredSerial) return;
        _resourceReadRetiredSerial = completed;
        _guestMemory.NoteCompleted(completed);
        _gpuWrites.Retire(completed);
        RetirePendingCopies(completed);
        _cpOverlay.Retire(completed);
        _labelOverlay.Retire(completed);
        InvalidateWordLines();
    }

    ulong SharpEmu.Libs.Gpu.GpuCommands.ICommandStreamHost.ConsumePendingResourceRead()
    {
        var serial = _pendingResourceReadSerial;
        _pendingResourceReadSerial = 0;
        return serial;
    }

    private void SuspendResourceReadIfSafe(ulong serial, ulong address, ulong size)
    {
        if (!AsyncResourceReads)
        {
            _resourceReadDisabled++;
            return;
        }

        if (!_resourceReadRetryAllowed || _commandStream is null || size != sizeof(uint) ||
            _wordLineThread != Environment.CurrentManagedThreadId)
        {
            _resourceReadScopeRefused++;
            return;
        }

        if (serial > _previousSerial)
        {
            // Do not split an open pass or defer work the worker must still submit.
            _resourceReadUncommittedRefused++;
            return;
        }

        if (ProfileGpuWaits)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][RESOURCE-READ-SUSPEND] serial={serial} read=0x{address:X}+0x{size:X} " +
                $"reader=0x{SharpEmu.ShaderCompiler.Resources.MaterializingShader.Hash:X16}");
        }

        throw new ResourceReadPending(serial);
    }

    internal (long Suspended, long Disabled, long Scope, long Uncommitted, long Image) ResourceReadRetryCounters =>
        (_resourceReadSuspends, _resourceReadDisabled, _resourceReadScopeRefused,
            _resourceReadUncommittedRefused, _resourceReadImageRefused);

    private string TakeResourceReadRetryCounters() =>
        $" resource_read_suspend={Interlocked.Exchange(ref _resourceReadSuspends, 0)}" +
        $" resource_read_wakes={Interlocked.Exchange(ref _resourceReadWakeups, 0)}" +
        FormattableString.Invariant($" resource_read_deferred_ms={Interlocked.Exchange(ref _resourceReadDeferredTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F3}") +
        $" resource_read_refused_disabled={Interlocked.Exchange(ref _resourceReadDisabled, 0)}" +
        $" resource_read_refused_scope={Interlocked.Exchange(ref _resourceReadScopeRefused, 0)}" +
        $" resource_read_refused_uncommitted={Interlocked.Exchange(ref _resourceReadUncommittedRefused, 0)}" +
        $" resource_read_refused_image={Interlocked.Exchange(ref _resourceReadImageRefused, 0)}";
}
