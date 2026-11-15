// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The words of an owned line answered from the line's mask, and the words a pending copy writes read from the copy
/// line (MetalRenderHost.ConcurrentLineMasks), against the same resolution with every word tested on its own: one
/// owned word among free ones in the line (a pending GPU write, a pending copy, a pending label), the same bytes or
/// the same abandon for every read, the same recorded answers, and the worker's re-check deciding the same way
/// before and after the owner's source changes.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong MaskLine = OwnedPage + 0x400;
    private const ulong MaskCopySource = GuestBase + 0x68000;

    private sealed record MaskOutcome(
        string Reads, string Answers, bool CurrentAtTake, bool CurrentAfterSourceWrite, string Abandons, string Counters);

    private static MaskOutcome ResolveAroundAnOwnedWord(string owner, MetalRenderHost.LineMaskMode mode)
    {
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentOwnedLines = true;
        host.ConcurrentScopedSerial = true;
        host.ConcurrentLineMasks = mode;
        host.ConcurrentCopyReads = true;
        for (var word = 0UL; word < ConcurrentPageWords; word++)
        {
            rig.WriteBacking32(OwnedPage + word * sizeof(uint), 0xD000u + (uint)word);
            rig.WriteBacking32(MaskCopySource + word * sizeof(uint), 0xC000u + (uint)word);
        }

        // The owned bytes sit at MaskLine + 0x40 .. 0x80 (copy) or at the word MaskLine + 0x80.
        ulong[] reads;
        switch (owner)
        {
            case "gpu_write":
                host.NoteGuestMemoryWrite(MaskLine + 0x80, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
                reads = [MaskLine, MaskLine + 0x7C, MaskLine + 0x84, MaskLine + 0xFC, MaskLine + 0x80];
                break;
            case "copy":
                Assert.False(host.TryCopyWordsOnHost(MaskLine + 0x40, MaskCopySource, 4, 16));
                host.NoteGuestMemoryWrite(MaskLine + 0x40, 16 * sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
                host.ResetBindings();
                reads = [MaskLine, MaskLine + 0x44, MaskLine + 0x48, MaskLine + 0x7C, MaskLine + 0x40, MaskLine + 0x80, MaskLine + 0x3C];
                break;
            case "label":
                host.LabelOverlayEnabled = true;
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, MaskLine + 0x80, 0x1234));
                reads = [MaskLine, MaskLine + 0x84, MaskLine + 0x80];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(owner));
        }

        _ = host.TakeOwnerTestCounters();
        _ = host.TakeAbandonCauses();
        var concurrent = (IConcurrentResolveHost)host;
        var log = new System.Text.StringBuilder();
        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            foreach (var address in reads)
            {
                log.Append($"0x{address - MaskLine:X}=");
                try
                {
                    log.Append(host.TryReadGuestWord(address, out var word) ? $"0x{word:X}" : "refused");
                }
                catch (GpuFatal.AbandonedWorkException)
                {
                    log.Append("abandoned");
                    break;
                }
                finally
                {
                    log.Append(' ');
                }

                // A run over the owned bytes, through the resident reader.
                if (address == MaskLine + 0x7C)
                {
                    log.Append($"run={host.TryReadResidentGuestBytes(MaskLine + 0x78, new byte[16], clean: false)} ");
                }
            }
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        var answers = string.Join(' ', ((MetalRenderHost.ConcurrentAssumptions)assumptions).Answers
            .Select(answer => $"{answer.Query}:0x{answer.Address:X}+{answer.Size}:{answer.Owned}"));
        var counters = host.TakeOwnerTestCounters();
        var abandons = host.TakeAbandonCauses();
        var current = concurrent.IsStillCurrent(assumptions);
        // A write on the copy's source (and on a clean line of the owned page) moves a version: every answer is checked again.
        host.NoteGuestMemoryWrite(MaskCopySource + 4, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        host.NoteGuestMemoryWrite(MaskLine + 0x100, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        var currentAfter = concurrent.IsStillCurrent(assumptions);
        concurrent.ReleaseAssumptions(assumptions);
        return new MaskOutcome(log.ToString(), answers, current, currentAfter, abandons, counters);
    }

    [Theory]
    [InlineData("gpu_write")]
    [InlineData("copy")]
    [InlineData("label")]
    public void AnOwnedWordAmongFreeOnesIsAnsweredAndCheckedAgainAsWithoutTheLineMask(string owner)
    {
        if (!RequireDevice()) return;
        var tested = ResolveAroundAnOwnedWord(owner, MetalRenderHost.LineMaskMode.Off);
        var masked = ResolveAroundAnOwnedWord(owner, MetalRenderHost.LineMaskMode.On);
        var verified = ResolveAroundAnOwnedWord(owner, MetalRenderHost.LineMaskMode.Verify);

        Assert.Equal(tested.Reads, masked.Reads);
        Assert.Equal(tested.Answers, masked.Answers);
        Assert.Equal(tested.CurrentAtTake, masked.CurrentAtTake);
        Assert.Equal(tested.CurrentAfterSourceWrite, masked.CurrentAfterSourceWrite);
        Assert.Equal(tested.Abandons, masked.Abandons);
        Assert.Equal((tested.Reads, tested.Answers, tested.CurrentAtTake, tested.CurrentAfterSourceWrite, tested.Abandons),
            (verified.Reads, verified.Answers, verified.CurrentAtTake, verified.CurrentAfterSourceWrite, verified.Abandons));
        Assert.Equal(0, OwnerTestCount(verified.Counters, "mask_verify_failed"));
        Assert.Equal(0, OwnerTestCount(masked.Counters, "word") - (owner == "label" ? OwnerTestCount(masked.Counters, "mask_refused") : 0));

        switch (owner)
        {
            case "gpu_write":
                // Free words read as the backing holds them, the run over the write refused, the written word abandons.
                Assert.Equal("0x0=0xD100 0x7C=0xD11F run=False 0x84=0xD121 0xFC=0xD13F 0x80=abandoned ", tested.Reads);
                Assert.True(OwnerTestCount(masked.Counters, "mask_words") >= 5, masked.Counters);
                Assert.True(OwnerTestCount(verified.Counters, "mask_verified") >= 5, verified.Counters);
                break;
            case "copy":
                // Copied words read as the copy's source (period 4), the words around the copy as the backing.
                Assert.Equal("0x0=0xD100 0x44=0xC001 0x48=0xC002 0x7C=0xC003 run=False 0x40=0xC000 0x80=0xD120 0x3C=0xD10F ", tested.Reads);
                Assert.True(tested.CurrentAtTake);
                Assert.False(tested.CurrentAfterSourceWrite);
                // 0x44 read through the copy fills the copy line; 0x48, 0x7C and 0x40 come from it.
                Assert.Equal(3, OwnerTestCount(masked.Counters, "copy_line_words"));
                Assert.True(OwnerTestCount(verified.Counters, "mask_verified") >= 3, verified.Counters);
                break;
            case "label":
                // A line a label owns has no usable mask: its words take their own tests.
                Assert.EndsWith("0x80=abandoned ", tested.Reads);
                Assert.Equal(0, OwnerTestCount(masked.Counters, "mask_words"));
                Assert.True(OwnerTestCount(masked.Counters, "mask_refused") >= 2, masked.Counters);
                break;
        }
    }
}
