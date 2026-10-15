// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Loader;
using SharpEmu.Libs.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class DemonsSoulsIdleNapPatchTests
{
    private const ulong Stub = 0x6FFF_FFFF_0000UL;
    private const ulong RegionStart = 0x860000;

    [Fact]
    public void TrampolinesFitTheDeadReportAndEverySiteKeepsItsLength()
    {
        var patches = DemonsSoulsIdleNapPatch.BuildPatches(500_000, 250, Stub);

        Assert.Equal(6, patches.Count);
        Assert.All(patches, patch => Assert.Equal(patch.Old!.Length, patch.New.Length));
        var block = Find(patches, DemonsSoulsIdleNapPatch.ReportBlock).New;
        Assert.True(block.Length <= 102);
        Assert.Equal(9, Find(patches, DemonsSoulsIdleNapPatch.ReportTail).New.Length);
    }

    [Fact]
    public void CallSitesReachTheirTrampolinesAndT3CarriesTheConfiguration()
    {
        var patches = DemonsSoulsIdleNapPatch.BuildPatches(500_000, 250, Stub);
        var block = Find(patches, DemonsSoulsIdleNapPatch.ReportBlock).New;
        var t3 = DemonsSoulsIdleNapPatch.ReportBlock;

        Assert.Equal(t3, CallTarget(patches, DemonsSoulsIdleNapPatch.WaitTscCall));
        Assert.Equal(t3 + 36, CallTarget(patches, DemonsSoulsIdleNapPatch.ChooseJobCall));
        Assert.Equal(t3 + 77, CallTarget(patches, DemonsSoulsIdleNapPatch.RunJobCall));
        Assert.Equal(500_000, BinaryPrimitives.ReadInt32LittleEndian(block.AsSpan(11)));
        Assert.Equal(250, BinaryPrimitives.ReadInt32LittleEndian(block.AsSpan(18)));
        Assert.Equal(Stub, BinaryPrimitives.ReadUInt64LittleEndian(block.AsSpan(24)));

        // The deadlock branch always skips the report now, and T2 ends in the report's tail.
        var branch = Find(patches, DemonsSoulsIdleNapPatch.DeadlockBranch).New;
        Assert.Equal(0xE9, branch[0]);
        Assert.Equal(0x86EC80UL, DemonsSoulsIdleNapPatch.DeadlockBranch + 5 + (ulong)BinaryPrimitives.ReadInt32LittleEndian(branch.AsSpan(1)));
        Assert.Equal(0xEB, block[^2]);
        Assert.Equal(DemonsSoulsIdleNapPatch.ReportTail, DemonsSoulsIdleNapPatch.ReportBlock + (ulong)block.Length + block[^1]);
    }

    [Fact]
    public void ApplyInstallsOnlyOnTheKnownBuildOfTheTitle()
    {
        using var memory = new SharpEmu.Core.Memory.PhysicalVirtualMemory();
        Assert.True(memory.TryAllocateAtOrAbove(0x200000000UL, 0x10000, executable: false, 0x4000, out var region));
        var imageBase = region - RegionStart;
        ProgramHeader[] headers = [GuestBytePatcherTests.CreateLoad(RegionStart, 0x10000)];
        var requested = new List<string>();
        ulong AddStub(string nid)
        {
            requested.Add(nid);
            return Stub;
        }

        // Another title, and this title without the expected bytes, are left alone.
        Assert.False(DemonsSoulsIdleNapPatch.Apply(memory, headers, imageBase, "PPSA00000", AddStub));
        Assert.False(DemonsSoulsIdleNapPatch.Apply(memory, headers, imageBase, "PPSA01342", AddStub));
        Assert.Empty(requested);

        foreach (var patch in DemonsSoulsIdleNapPatch.BuildPatches(1, 1, 0))
        {
            Assert.True(memory.TryWrite(imageBase + patch.Offset, patch.Old!));
        }

        Assert.True(DemonsSoulsIdleNapPatch.Apply(memory, headers, imageBase, "PPSA01342", AddStub));
        Assert.Equal([IdleNapExports.Nid], requested);
        var call = new byte[5];
        Assert.True(memory.TryRead(imageBase + DemonsSoulsIdleNapPatch.WaitTscCall, call));
        Assert.Equal(0xE8, call[0]);
        Assert.Equal(DemonsSoulsIdleNapPatch.ReportBlock, DemonsSoulsIdleNapPatch.WaitTscCall + 5 + (ulong)BinaryPrimitives.ReadInt32LittleEndian(call.AsSpan(1)));

        // Patched code no longer matches, so a second application changes nothing.
        Assert.False(DemonsSoulsIdleNapPatch.Apply(memory, headers, imageBase, "PPSA01342", AddStub));
        Assert.Single(requested);
    }

    [Fact]
    public void ThresholdFollowsTheTscFrequencyAndFitsAnImmediate()
    {
        Assert.Equal(500_000L, DemonsSoulsIdleNapPatch.ThresholdTicks(500, 1_000_000_000));
        Assert.Equal(1_500_000L, DemonsSoulsIdleNapPatch.ThresholdTicks(500, 3_000_000_000));
        Assert.Equal((long)int.MaxValue, DemonsSoulsIdleNapPatch.ThresholdTicks(100_000, 100_000_000_000));
    }

    [Fact]
    public void AbPhasesAlternateFromTheEpoch()
    {
        Assert.True(IdleNapExports.IsNapPhase(now: 5, epoch: 0, periodTicks: 10));
        Assert.False(IdleNapExports.IsNapPhase(now: 15, epoch: 0, periodTicks: 10));
        Assert.True(IdleNapExports.IsNapPhase(now: 125, epoch: 100, periodTicks: 10));
    }

    private static GuestBytePatcher.Patch Find(List<GuestBytePatcher.Patch> patches, ulong offset) =>
        patches.Single(patch => patch.Offset == offset);

    private static ulong CallTarget(List<GuestBytePatcher.Patch> patches, ulong offset)
    {
        var bytes = Find(patches, offset).New;
        Assert.Equal(0xE8, bytes[0]);
        return offset + 5 + (ulong)(long)BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(1));
    }
}
