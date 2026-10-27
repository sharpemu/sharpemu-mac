// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    // No live-source copy substitution, speculative labels or cached CPU word lines.
    // =0 restores all previous shortcuts and their individual kill switches.
    internal bool FinalGuestTablesEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FINAL_GUEST_TABLES") != "0";

    public GuestTableReadState LastGuestTableRead { get; private set; } = new("none", 0, 0, "none");

    // Only a negative ownership proof is reused, never the bytes. CPU writes
    // remain visible on every scalar read, even when all ownership versions match.
    internal bool FinalReadHintsEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FINAL_READ_HINTS") != "0";
    private readonly ulong[] _finalReadLineAddress = new ulong[WordLineSlots];
    private readonly ulong[] _finalReadLineGeneration = new ulong[WordLineSlots];
    private readonly bool[] _finalReadLineUnowned = new bool[WordLineSlots];
    private CommandReadToken _finalReadToken;
    private long _finalReadLabelVersion;
    private long _finalReadHintHits, _finalReadHintFills, _finalReadHintRefused;

    private bool FinalReadRangeIsUnowned(ulong address, ulong size = sizeof(uint))
    {
        if (!FinalReadHintsEnabled || !FinalGuestTablesEnabled ||
            _wordLineThread != Environment.CurrentManagedThreadId ||
            Volatile.Read(ref _owedHostLabels) != 0 || size == 0 || size > (ulong)WordLineBytes ||
            (address & (ulong)(WordLineBytes - 1)) > (ulong)WordLineBytes - size)
            return false;

        var token = CurrentCommandReadToken();
        var labels = _labelOverlay.Version;
        if (token != _finalReadToken || labels != _finalReadLabelVersion)
        {
            _finalReadToken = token;
            _finalReadLabelVersion = labels;
            InvalidateWordLines();
        }
        var line = address & ~(ulong)(WordLineBytes - 1);
        var slot = (int)((line / (ulong)WordLineBytes) & (ulong)(WordLineSlots - 1));
        if (_finalReadLineAddress[slot] != line || _finalReadLineGeneration[slot] != _wordLineGenerationCurrent)
        {
            _finalReadLineAddress[slot] = line;
            _finalReadLineGeneration[slot] = _wordLineGenerationCurrent;
            _finalReadLineUnowned[slot] = !GuestTableNotReady(line, (ulong)WordLineBytes);
            _finalReadHintFills++;
        }
        if (_finalReadLineUnowned[slot]) { _finalReadHintHits++; return true; }
        _finalReadHintRefused++;
        return false;
    }

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
        Volatile.Read(ref _owedHostLabels) != 0 ||
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
