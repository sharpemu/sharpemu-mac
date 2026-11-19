// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A page one resolution found clean is taken as clean by the next only while the ownership
/// versions stand (MetalRenderHost.IsCrossDrawClean): a GPU write recorded on it in between
/// moves a version, and the next resolution reading it abandons as it did before.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    [Fact]
    public void APageAResolutionFoundCleanIsOwnedAgainOnceAGpuWriteIsRecordedOnIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        WriteKernelSource(rig);
        var concurrent = (IConcurrentResolveHost)host;

        for (var pass = 0; pass < 2; pass++)
        {
            concurrent.BeginConcurrentResolve();
            try
            {
                Assert.True(host.TryReadGuestWord(KernelSource + 4, out var word));
                Assert.Equal(rig.ReadBacking32(KernelSource + 4), word);
            }
            finally
            {
                concurrent.ReleaseAssumptions(concurrent.EndConcurrentResolve());
            }
        }

        host.NoteGuestMemoryWrite(KernelSource + 0x10, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        concurrent.BeginConcurrentResolve();
        try
        {
            Assert.Throws<GpuFatal.AbandonedWorkException>(() => host.TryReadGuestWord(KernelSource + 0x10, out _));
        }
        finally
        {
            concurrent.ReleaseAssumptions(concurrent.EndConcurrentResolve());
        }
    }

    // A page holding a pending GPU write, as Demon's Souls keeps one in its descriptor page: the 256-byte line
    // at 0x800 is owned, every other line of the page is clean.
    private const ulong OwnedPage = GuestBase + 0x60000;
    private const ulong OwnedPageWrite = OwnedPage + 0x800;

    private static void WriteOwnedPage(Rig rig)
    {
        for (var word = 0UL; word < ConcurrentPageWords; word++)
        {
            rig.WriteBacking32(OwnedPage + word * sizeof(uint), 0xD000u + (uint)word);
        }

        rig.Host.NoteGuestMemoryWrite(OwnedPageWrite, 16, MetalGpuWriteOrigin.StorageBuffer);
    }

    private const ulong ConcurrentPageWords = 4096 / sizeof(uint);

    private static long OwnerTestCount(string counters, string name)
    {
        var start = counters.IndexOf(name + "=", StringComparison.Ordinal) + name.Length + 1;
        var end = start;
        while (end < counters.Length && char.IsAsciiDigit(counters[end]))
        {
            end++;
        }

        return long.Parse(counters.AsSpan(start, end - start), System.Globalization.CultureInfo.InvariantCulture);
    }

    private static uint OwnedPageWord(ulong address) => 0xD000u + (uint)((address - OwnedPage) / sizeof(uint));

    // Reads the words in one resolution and returns its assumptions; each word must read as the backing holds it.
    private static object ResolveWords(MetalRenderHost host, params ulong[] addresses)
    {
        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        try
        {
            foreach (var address in addresses)
            {
                Assert.True(host.TryReadGuestWord(address, out var word));
                Assert.Equal(OwnedPageWord(address), word);
            }
        }
        catch
        {
            concurrent.ReleaseAssumptions(concurrent.EndConcurrentResolve());
            throw;
        }

        return concurrent.EndConcurrentResolve();
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheCleanLinesOfAnOwnedPageAreTestedOnceAndTheirWordsReadFromThem(bool ownedLines)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentOwnedLines = ownedLines;
        WriteOwnedPage(rig);
        _ = host.TakeOwnerTestCounters();
        var concurrent = (IConcurrentResolveHost)host;

        // Two lines: three words in the first, one in the second, and the first again.
        var assumptions = ResolveWords(host, OwnedPage + 0x10, OwnedPage + 0x14, OwnedPage + 0x40, OwnedPage + 0x110, OwnedPage + 0x18);
        var counters = host.TakeOwnerTestCounters();
        Assert.Contains(ownedLines ? "page=1,line=2,word=0," : "page=5,line=0,word=5,", counters);
        Assert.True(concurrent.IsStillCurrent(assumptions));

        // A write elsewhere moves a version: the worker checks every answer again, with one serial.
        host.NoteGuestMemoryWrite(GuestBase + 0x70000, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.True(concurrent.IsStillCurrent(assumptions));
        if (ownedLines)
        {
            // Two line answers, and the serial asked for once at most for both of them.
            var recheck = host.TakeOwnerTestCounters();
            Assert.Contains("recheck=2,", recheck);
            Assert.True(OwnerTestCount(recheck, "serial_reads") <= (host.ConcurrentScopedSerial ? 1 : 2), recheck);
        }

        concurrent.ReleaseAssumptions(assumptions);
    }

    // A range that starts in a clean line and runs on into the written one is in no single line: the clean
    // line it starts in says nothing for it.
    [Theory]
    [InlineData(0x300, true)]
    [InlineData(0x300, false)]
    [InlineData(0x108, true)]
    [InlineData(0x108, false)]
    public void ARangeLeavingItsCleanLineForAWrittenOneIsRefused(int length, bool ownedLines)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentOwnedLines = ownedLines;
        WriteOwnedPage(rig);
        var concurrent = (IConcurrentResolveHost)host;

        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            // The page is found owned and the line before the write clean, first.
            Assert.True(host.TryReadGuestWord(OwnedPageWrite - 0x100, out var word));
            Assert.Equal(OwnedPageWord(OwnedPageWrite - 0x100), word);
            Assert.False(host.TryReadResidentGuestBytes(OwnedPageWrite - 0x100, new byte[length], clean: false));
            // Inside the clean line the same reader answers.
            Assert.True(host.TryReadResidentGuestBytes(OwnedPageWrite - 0x100, new byte[0x100], clean: false));
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        concurrent.ReleaseAssumptions(assumptions);
    }

    [Fact]
    public void TheWordsOfAnOwnedLineAreTestedOneByOneAndAnOwnedWordAbandons()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentOwnedLines = true;
        // Answered from the line's mask otherwise (MetalClearFoldingTests.LineMasks).
        host.ConcurrentLineMasks = MetalRenderHost.LineMaskMode.Off;
        WriteOwnedPage(rig);
        _ = host.TakeOwnerTestCounters();
        var concurrent = (IConcurrentResolveHost)host;

        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            // The line the write is in, past the write: read, after its own test.
            Assert.True(host.TryReadGuestWord(OwnedPageWrite + 0x20, out var word));
            Assert.Equal(OwnedPageWord(OwnedPageWrite + 0x20), word);
            Assert.Throws<GpuFatal.AbandonedWorkException>(() => host.TryReadGuestWord(OwnedPageWrite + 4, out _));
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        Assert.Contains("page=1,line=1,word=2,", host.TakeOwnerTestCounters());

        Assert.Equal("gpu_write=1,copy_dst=0,label=0,cp=0,image=0,code=0,queued=0,other=0", host.TakeAbandonCauses());
        concurrent.ReleaseAssumptions(assumptions);
    }

    [Theory]
    [InlineData(OwnedPage + 0x14, false)]
    [InlineData(OwnedPage + 0x10, false)]
    // The same page, another line: the line answer the resolution relied on still holds.
    [InlineData(OwnedPage + 0x900, true)]
    public void AWriteLandingBetweenTheResolutionAndTheTakeRefusesItWhereItCoversALineItRead(ulong written, bool stands)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentOwnedLines = true;
        WriteOwnedPage(rig);
        var concurrent = (IConcurrentResolveHost)host;
        var assumptions = ResolveWords(host, OwnedPage + 0x10, OwnedPage + 0x40);
        Assert.True(concurrent.IsStillCurrent(assumptions));

        host.NoteGuestMemoryWrite(written, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.Equal(stands, concurrent.IsStillCurrent(assumptions));
        concurrent.ReleaseAssumptions(assumptions);
    }
}
