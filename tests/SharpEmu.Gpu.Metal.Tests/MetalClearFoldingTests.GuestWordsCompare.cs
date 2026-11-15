// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A program evaluation's recorded reads compared where they lie (MetalRenderHost.TryCompareGuestWords, the read
/// proof's comparer): off the worker, each range takes the ownership test a word read of it takes and records the
/// answer as that read does, so the worker's re-check at the take sees every compared word; a range a pending GPU write
/// or label owns refuses (the caller then reads the words), with no owned answer left behind; changed or unmapped bytes
/// say false. On the worker, the resident compare's refusals.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    // Two ranges in the owned page's clean lines and one in a clean page.
    private static readonly (ulong Address, int Length)[] ProofRanges =
        [(OwnedPage + 0x100, 16), (OwnedPage + 0x8C0, 64), (GuestBase + 0x70000, 8)];

    // The ranges' backing bytes, range after range, as an evaluation would have recorded them.
    private static byte[] ProofBytes(Rig rig, (ulong Address, int Length)[] ranges)
    {
        var bytes = new byte[ranges.Sum(range => range.Length)];
        var offset = 0;
        foreach (var (address, length) in ranges)
        {
            for (var word = 0; word < length / sizeof(uint); word++)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(offset + word * sizeof(uint)), rig.ReadBacking32(address + (ulong)(word * sizeof(uint))));
            }

            offset += length;
        }

        return bytes;
    }

    // The comparison inside one concurrent resolution, with the answers it left recorded.
    private static (bool Equal, string Answers, object Assumptions) CompareConcurrent(Rig rig, (ulong Address, int Length)[] ranges, byte[] recorded)
    {
        var host = rig.Host;
        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        bool equal;
        object assumptions;
        try
        {
            equal = host.TryCompareGuestWords([.. ranges.Select(range => range.Address)], [.. ranges.Select(range => range.Length)], recorded);
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        var answers = string.Join(' ', ((MetalRenderHost.ConcurrentAssumptions)assumptions).Answers
            .Select(answer => $"{answer.Query}:0x{answer.Address - GuestBase:X}+{answer.Size}:{answer.Owned}"));
        return (equal, answers, assumptions);
    }

    private static Rig ProofRig()
    {
        var rig = new Rig(Device.Value);
        rig.Host.ConcurrentOwnedLines = true;
        rig.Host.ConcurrentScopedSerial = true;
        rig.Host.ConcurrentLineMasks = MetalRenderHost.LineMaskMode.On;
        WriteOwnedPage(rig);
        for (var word = 0UL; word < 4; word++)
        {
            rig.WriteBacking32(GuestBase + 0x70000 + word * sizeof(uint), 0xE000u + (uint)word);
        }

        return rig;
    }

    [Fact]
    public void EqualWordsCompareTrueAndRecordAnswersTheTakeChecksAgain()
    {
        if (!RequireDevice()) return;
        using var rig = ProofRig();
        var recorded = ProofBytes(rig, ProofRanges);
        var (equal, answers, assumptions) = CompareConcurrent(rig, ProofRanges, recorded);
        Assert.True(equal);
        // The owned page's clean line, the range itself in the line a pending write shares (its mask), and the clean
        // page, as word reads in them record.
        Assert.Equal("Word:0x60100+256:False Word:0x608C0+64:False Word:0x70000+4096:False", answers);
        var concurrent = (IConcurrentResolveHost)rig.Host;
        Assert.True(concurrent.IsStillCurrent(assumptions));

        // A GPU write recorded over a compared word after the compare: the take refuses the resolution.
        rig.Host.NoteGuestMemoryWrite(OwnedPage + 0x8F0, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.False(concurrent.IsStillCurrent(assumptions));
        concurrent.ReleaseAssumptions(assumptions);
    }

    [Fact]
    public void AChangedWordInsideARangeComparesFalseAndOneOutsideDoesNot()
    {
        if (!RequireDevice()) return;
        using var rig = ProofRig();
        var recorded = ProofBytes(rig, ProofRanges);
        rig.WriteBacking32(OwnedPage + 0x8C0 + 60, 0xBAD);
        var inside = CompareConcurrent(rig, ProofRanges, recorded);
        Assert.False(inside.Equal);
        ((IConcurrentResolveHost)rig.Host).ReleaseAssumptions(inside.Assumptions);

        rig.WriteBacking32(OwnedPage + 0x8C0 + 60, 0xD000u + (0x8C0 + 60) / 4);
        rig.WriteBacking32(OwnedPage + 0x8C0 + 64, 0xBAD);
        rig.WriteBacking32(OwnedPage + 0x110, 0xBAD);
        var outside = CompareConcurrent(rig, ProofRanges, recorded);
        Assert.True(outside.Equal);
        ((IConcurrentResolveHost)rig.Host).ReleaseAssumptions(outside.Assumptions);
    }

    [Theory]
    [InlineData("gpu_write")]
    [InlineData("label")]
    public void ARangeAPendingWriteOwnsRefusesWithNoOwnedAnswerLeft(string owner)
    {
        if (!RequireDevice()) return;
        using var rig = ProofRig();
        // The owned page's pending write sits at 0x800 .. 0x810, under the second range; or a label pending over a word
        // of the first range, in a line of the page no write owns.
        (ulong Address, int Length)[] ranges = owner == "gpu_write"
            ? [(OwnedPage + 0x100, 16), (OwnedPage + 0x808, 32), (GuestBase + 0x70000, 8)]
            : [(OwnedPage + 0x100, 16), (OwnedPage + 0x880, 32), (GuestBase + 0x70000, 8)];
        var recorded = ProofBytes(rig, ranges);
        if (owner == "label")
        {
            rig.Host.LabelOverlayEnabled = true;
            rig.Host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, OwnedPage + 0x104, 0x1234));
        }

        var (equal, answers, assumptions) = CompareConcurrent(rig, ranges, recorded);
        Assert.False(equal);
        Assert.DoesNotContain(":True", answers);
        ((IConcurrentResolveHost)rig.Host).ReleaseAssumptions(assumptions);
    }

    [Fact]
    public void ARangeAPendingCopyWritesComparesThroughTheCopyAndTheTakeChecksItsSource()
    {
        if (!RequireDevice()) return;
        using var rig = ProofRig();
        RecordPendingCopy(rig);
        // Words 4 .. 11 of the destination: the source's period twice.
        (ulong Address, int Length)[] ranges = [(CopyDestination + 0x10, 32), (GuestBase + 0x70000, 8)];
        var recorded = ProofBytes(rig, [(GuestBase + 0x70000, 8)]);
        var copied = new byte[32];
        for (var word = 0; word < 8; word++)
        {
            BitConverter.TryWriteBytes(copied.AsSpan(word * sizeof(uint)), 0xA000u + (uint)(word % (int)CopyPeriod));
        }

        recorded = [.. copied, .. recorded];
        var (equal, answers, assumptions) = CompareConcurrent(rig, ranges, recorded);
        Assert.True(equal);
        // The destination's owned answer is dropped; its source's answer stands in for it.
        Assert.DoesNotContain(":True", answers);
        Assert.Contains("0x30000", answers);
        var concurrent = (IConcurrentResolveHost)rig.Host;
        Assert.True(concurrent.IsStillCurrent(assumptions));
        concurrent.ReleaseAssumptions(assumptions);

        // Bytes the copy would write other than those recorded compare false.
        copied[4] ^= 1;
        recorded = [.. copied, .. recorded.AsSpan(32)];
        var changed = CompareConcurrent(rig, ranges, recorded);
        Assert.False(changed.Equal);
        concurrent.ReleaseAssumptions(changed.Assumptions);

        // A write recorded after the copy over a compared word comes from stream items after the draw, so the take
        // accepts a compare made before it exactly as it accepts a word read through the copy made then
        // (TryReadThroughPendingCopyConcurrent); a compare made after it refuses, the copy no longer the range's latest
        // write, with no owned answer left behind.
        copied[4] ^= 1;
        recorded = [.. copied, .. recorded.AsSpan(32)];
        var before = CompareConcurrent(rig, ranges, recorded);
        Assert.True(before.Equal);
        concurrent.BeginConcurrentResolve();
        object read;
        try
        {
            Assert.True(rig.Host.TryReadGuestWord(CopyDestination + 0x18, out var word));
            Assert.Equal(0xA002u, word);
        }
        finally
        {
            read = concurrent.EndConcurrentResolve();
        }

        rig.Host.NoteGuestMemoryWrite(CopyDestination + 0x18, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.True(concurrent.IsStillCurrent(read));
        Assert.True(concurrent.IsStillCurrent(before.Assumptions));
        concurrent.ReleaseAssumptions(read);
        concurrent.ReleaseAssumptions(before.Assumptions);
        var after = CompareConcurrent(rig, ranges, recorded);
        Assert.False(after.Equal);
        Assert.DoesNotContain(":True", after.Answers);
        concurrent.ReleaseAssumptions(after.Assumptions);
    }

    [Fact]
    public void AWriteOverTheCopySourceAfterAThroughCopyCompareFailsTheTake()
    {
        if (!RequireDevice()) return;
        using var rig = ProofRig();
        RecordPendingCopy(rig);
        (ulong Address, int Length)[] ranges = [(CopyDestination + 0x10, 16)];
        var recorded = new byte[16];
        for (var word = 0; word < 4; word++)
        {
            BitConverter.TryWriteBytes(recorded.AsSpan(word * sizeof(uint)), 0xA000u + (uint)word);
        }

        var (equal, _, assumptions) = CompareConcurrent(rig, ranges, recorded);
        Assert.True(equal);
        var concurrent = (IConcurrentResolveHost)rig.Host;
        rig.Host.NoteGuestMemoryWrite(CopySource + 4, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.False(concurrent.IsStillCurrent(assumptions));
        concurrent.ReleaseAssumptions(assumptions);
    }

    [Fact]
    public void UnmappedOrMalformedRangesCompareFalse()
    {
        if (!RequireDevice()) return;
        using var rig = ProofRig();
        var far = (Address: GuestBase + 0x4000_0000UL, Length: 8);
        Assert.False(CompareConcurrent(rig, [far], new byte[8]).Equal);
        // Unaligned, a length past the recorded bytes, and recorded bytes left over.
        Assert.False(CompareConcurrent(rig, [(GuestBase + 0x70002, 4)], new byte[4]).Equal);
        Assert.False(CompareConcurrent(rig, [(GuestBase + 0x70000, 8)], new byte[4]).Equal);
        Assert.False(CompareConcurrent(rig, [(GuestBase + 0x70000, 4)], new byte[8]).Equal);
    }

    [Fact]
    public void OnTheWorkerAPendingGpuWriteRefusesAndCleanWordsCompare()
    {
        if (!RequireDevice()) return;
        using var rig = ProofRig();
        rig.AliasAccess = true;
        (ulong Address, int Length)[] clean = [(GuestBase + 0x70000, 8)];
        Assert.True(rig.Host.TryCompareGuestWords([clean[0].Address], [clean[0].Length], ProofBytes(rig, clean)));
        (ulong Address, int Length)[] owned = [(OwnedPage + 0x800, 16)];
        Assert.False(rig.Host.TryCompareGuestWords([owned[0].Address], [owned[0].Length], ProofBytes(rig, owned)));
    }
}
