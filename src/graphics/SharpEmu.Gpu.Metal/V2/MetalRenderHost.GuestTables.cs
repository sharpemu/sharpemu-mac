// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Pipelines;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    internal bool GpuWordsEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GPU_WORDS") != "0";
    private readonly MetalHostPublications _hostPublications = new();
    private long _publicationDisjoint, _gpuByteWaits, _preciseImageWaits;

    private bool HostPublicationPending(ulong address, ulong size) =>
        Volatile.Read(ref _owedHostLabels) != 0 &&
        (!GpuWordsEnabled || _hostPublications.Pending(address, size, _backing) != 0);

    private void WaitForHostPublication(ulong address, ulong size)
    {
        if (Volatile.Read(ref _owedHostLabels) == 0) return;
        if (!GpuWordsEnabled) { SynchronizeGuestTables(); return; }
        var serial = _hostPublications.Pending(address, size, _backing);
        if (serial != 0) WaitForCommittedSerial(serial, MetalWaitReason.HostPublication);
        else _publicationDisjoint++;
    }

    private string TakeGpuWordCounters() =>
        $" gpu_words={(GpuWordsEnabled ? 1 : 0)} publication_disjoint={Interlocked.Exchange(ref _publicationDisjoint, 0)}" +
        $" publication_alias_fallbacks={_hostPublications.AliasFallbacks}" +
        $" gpu_bytes_event_waits={Interlocked.Exchange(ref _gpuByteWaits, 0)}" +
        $" gds_gpu_fills={Interlocked.Exchange(ref _gdsGpuFills, 0)} gds_gpu_copies={Interlocked.Exchange(ref _gdsGpuCopies, 0)}" +
        $" gds_host_fallbacks={Interlocked.Exchange(ref _gdsHostFallbacks, 0)}" +
        $" gds_fallback_align={Interlocked.Exchange(ref _gdsAlignmentFallbacks, 0)} gds_fallback_backing={Interlocked.Exchange(ref _gdsBackingFallbacks, 0)}" +
        $" gds_fallback_disabled={Interlocked.Exchange(ref _gdsDisabledFallbacks, 0)} gds_snapshot_waits={Interlocked.Exchange(ref _gdsSnapshotWaits, 0)}" +
        $" image_serial_waits={Interlocked.Exchange(ref _preciseImageWaits, 0)}";

    // No live-source copy substitution, speculative labels or cached CPU word lines.
    // =0 restores all previous shortcuts and their individual kill switches.
    internal bool FinalGuestTablesEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FINAL_GUEST_TABLES") != "0";

    public GuestTableReadState LastGuestTableRead { get; private set; } = new("none", 0, 0, "none");

    private void CaptureGuestTableRead(ulong address, ulong size, string path)
    {
        var completed = CompletedSerialHint;
        var pending = _gpuWrites.MayCover(address, size) ? PendingSerialHinted(address, size) : 0;
        var writer = pending == 0 ? "none" : MetalGpuWriteOrigins.Names[(int)_gpuWrites.LastMatch.Origin];
        LastGuestTableRead = new(path, pending, completed, writer,
            _imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size),
            Volatile.Read(ref _owedHostLabels),
            _cpOverlay.Count != 0 && OverlayCoverageHinted(address, size) != MetalOverlayCoverage.None,
            _labelOverlay.Count != 0 && _labelOverlay.Coverage(address, size,
                _completedSerialReader ??= () => CompletedSerial) != MetalOverlayCoverage.None);
    }

    private bool GuestTableNotReady(ulong address, ulong size) =>
        HostPublicationPending(address, size) ||
        _imageCache.HasGpuModifiedImageBytes(address, size) ||
        (_gpuWrites.MayCover(address, size) && PendingSerialHinted(address, size) != 0) ||
        (_cpOverlay.Count != 0 && OverlayCoverageHinted(address, size) != MetalOverlayCoverage.None) ||
        (_labelOverlay.Count != 0 && _labelOverlay.Coverage(address, size,
            _completedSerialReader ??= () => CompletedSerial) != MetalOverlayCoverage.None);

    public void SynchronizeGuestTables()
    {
        // This executes directly on B. Never enqueue a relay request and wait on B itself.
        FlushAndWaitForGpu(MetalWaitReason.ReadRange);
        InvalidateWordLines();
    }

    public string DescribeGuestTable(ulong address)
    {
        var size = 64UL;
        var pending = _gpuWrites.PendingSerial(address, size, CompletedSerial);
        var writer = pending == 0 ? "none" : _gpuWrites.LastMatch.ToString();
        // Diagnostic only: do not wait or re-enter the evaluator to inspect bad pointers.
        Span<byte> bytes = stackalloc byte[(int)size];
        var readable = _memory.TryRead(address, bytes);
        return $"table=0x{address:X16} bytes={(readable ? Convert.ToHexString(bytes) : "unreadable")} " +
            $"pending={pending} writer={writer} completed={CompletedSerial} recording={_submissionSerial} " +
            $"host_labels={Volatile.Read(ref _owedHostLabels)} cp={OverlayCoverageHinted(address, size)} " +
            $"labels={_labelOverlay.Coverage(address, size, _completedSerialReader ??= () => CompletedSerial)} " +
            $"image={_imageCache.HasGpuModifiedImageBytes(address, size)}";
    }
}
