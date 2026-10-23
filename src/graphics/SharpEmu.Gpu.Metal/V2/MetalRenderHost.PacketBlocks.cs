// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Metal.V2;

// The command stream reads a run of command memory at once (GpuCommandInterpreter.PacketBlocks)
// instead of asking TryReadGuest for every packet. A run is handed over only when a read of any
// part of it would be a plain read of memory: no pending command-processor bytes over it, no
// GPU-modified image bytes and no recorded write that has yet to complete - the tests a word line
// passes. Everything else is refused and read packet by packet, so a block never waits for the GPU
// and never reads around an owed write.
//
// The token is the four counters whose movement could change either the bytes of such a run or the
// answer to those tests:
//   - the pending-write overlay's version (a command-processor write added, forgotten or cleared,
//     including every host store that forgets owed bytes: TryWriteGuest, host fills and copies);
//   - the recorded GPU writes' version (a store, a copy, a writeback or an unbounded write noted);
//   - the image set's version and the GPU-modified transition count (an image that takes or gives
//     up the guest bytes). Retiring landed entries moves none of them: a landed write changed
//     nothing the run's tests had not already refused.
// They are read before the tests and the copy, so a change racing the fill is seen as stale.
// Not covered, as for every read of guest memory here: a store by the guest CPU or through the raw
// backing (TryWriteBacking), which reach no counter. The stream bounds that by never keeping a
// block past its slice.
internal sealed partial class MetalRenderHost
{
    // SHARPEMU_GPU_PACKET_BLOCK=0 reads every packet through TryReadGuest.
    public bool SupportsCommandBlocks => PacketBlockReads.Enabled;

    public bool TryReadCommandBlock(ulong address, Span<byte> block, out CommandReadToken token)
    {
        token = CurrentCommandReadToken();
        return PacketBlockReads.Enabled && !block.IsEmpty && TryReadUnownedBytes(address, (ulong)block.Length, block);
    }

    public bool IsCommandBlockCurrent(in CommandReadToken token) => token == CurrentCommandReadToken();

    private CommandReadToken CurrentCommandReadToken() =>
        new(_cpOverlay.Version, (long)_gpuWrites.Version, _imageCache.ImageSetVersion, _imageCache.Epochs.GpuModified + (long)_hostPublications.Version);

    // True when every byte of the range would be read straight from memory, without overlay bytes
    // and without a wait, and the range is read into destination. Every test is an overlap test,
    // so passing for the range implies passing for each read inside it.
    private bool TryReadUnownedBytes(ulong address, ulong size, Span<byte> destination)
    {
        if (HostPublicationPending(address, size)) return false;
        if (_cpOverlay.Count != 0 && OverlayCoverageHinted(address, size) != MetalOverlayCoverage.None)
        {
            return false;
        }

        var mayCover = _gpuWrites.MayCover(address, size);
        if ((mayCover || _imageCache.MayOwn(address, size)) &&
            (_imageCache.HasGpuModifiedImageBytes(address, size) ||
             (mayCover && PendingSerialHinted(address, size) != 0)))
        {
            return false;
        }

        return _memory.TryRead(address, destination);
    }

    // SHARPEMU_METAL_V2_SERIAL_HINT=0 asks the GPU for the completed serial at every query, as before.
    private static readonly bool SerialHintEnabled =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_SERIAL_HINT");

    /// <summary>
    /// The newest completed serial this host has seen: a floor under <see cref="CompletedSerial"/>
    /// that costs no message send. A lower serial can only make a query report more pending work,
    /// never less, so a query that wants an exact answer asks again with the real one when the hint's
    /// answer is not "nothing pending" (<see cref="PendingSerialHinted"/>).
    /// </summary>
    public ulong CompletedSerialHint => SerialHintEnabled ? _completions.CompletedHint : CompletedSerial;

    private Func<ulong>? _completedSerialHintReader;

    // PendingSerial as it would answer for the real completed serial: the hint can only say
    // zero when the real serial does too, so only a non-zero answer is confirmed with it.
    private ulong PendingSerialHinted(ulong address, ulong size)
    {
        if (!SerialHintEnabled)
        {
            return _gpuWrites.PendingSerial(address, size, CompletedSerial);
        }

        var pending = _gpuWrites.PendingSerial(address, size, CompletedSerialHint);
        return pending == 0 ? 0 : _gpuWrites.PendingSerial(address, size, CompletedSerial);
    }

    // The same for the overlay's coverage: entries the hint has not retired can only add coverage.
    private MetalOverlayCoverage OverlayCoverageHinted(ulong address, ulong size)
    {
        if (!SerialHintEnabled)
        {
            return _cpOverlay.Coverage(address, size, _completedSerialReader ??= () => CompletedSerial);
        }

        var coverage = _cpOverlay.Coverage(address, size, _completedSerialHintReader ??= () => CompletedSerialHint);
        return coverage == MetalOverlayCoverage.None
            ? coverage
            : _cpOverlay.Coverage(address, size, _completedSerialReader ??= () => CompletedSerial);
    }
}
