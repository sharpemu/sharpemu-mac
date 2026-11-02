// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Each guard of the host kernel copy check refuses a kernel whose bytes memory does not yet say:
/// surface metadata on either side, pending command-processor or label bytes on either side, and a
/// pending write that names no range. Each case starts from a destination that already holds the
/// copy's result, so only the guard stands between the kernel and its elision.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    // Indices into HostKernelCopyOutcomes.
    private const int OutcomeIdentical = 0;
    private const int OutcomeSourcePending = 2;
    private const int OutcomeDestinationWrittenPending = 3;
    private const int OutcomeDestinationImage = 4;
    private const int OutcomeSourceMetadata = 7;

    [Theory]
    [InlineData("none", OutcomeIdentical)]
    [InlineData("dst-metadata", OutcomeDestinationImage)]
    [InlineData("src-metadata", OutcomeSourceMetadata)]
    [InlineData("dst-cp", OutcomeDestinationWrittenPending)]
    [InlineData("src-cp", OutcomeSourcePending)]
    [InlineData("dst-label", OutcomeDestinationWrittenPending)]
    [InlineData("src-label", OutcomeSourcePending)]
    [InlineData("unbounded", OutcomeDestinationWrittenPending)]
    public void AKernelWhoseBytesMemoryDoesNotYetSayIsRefusedForThatReason(string guard, int expected)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Identical;
        host.LabelOverlayEnabled = true;
        WriteKernelSource(rig);
        WriteKernelResult(rig);
        var word = BitConverter.GetBytes(0xB001u);
        switch (guard)
        {
            case "dst-metadata":
                // A DCC fill seen before its target: a pending metadata entry at the address.
                host.TryAbsorbDccFill(KernelDestination, KernelBytes, 0);
                break;
            case "src-metadata":
                host.TryAbsorbDccFill(KernelSource, KernelPeriod * sizeof(uint), 0);
                break;
            case "dst-cp":
                host.RecordCommandProcessorWrite(KernelDestination + 4, word, "WRITE_DATA");
                break;
            case "src-cp":
                host.RecordCommandProcessorWrite(KernelSource + 4, word, "WRITE_DATA");
                break;
            case "dst-label":
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, KernelDestination + 4, 0xB001));
                break;
            case "src-label":
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, KernelSource + 4, 0xB001));
                break;
            case "unbounded":
                host.NoteUnboundedGuestMemoryWrite();
                break;
        }

        // The pending bytes are the ones already there, so only the guard can refuse it.
        Assert.Equal(expected == OutcomeIdentical, host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        var outcomes = (ITuple)host.HostKernelCopyOutcomes;
        for (var index = 0; index < outcomes.Length; index++)
        {
            Assert.Equal(index == expected ? 1L : 0L, (long)outcomes[index]!);
        }
    }
}
