// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A dispatch's program may resolve before the dispatches ahead of it ran (ParallelProgramPrefetch.SubmitCompute).
/// Their stores are recorded when they run, and may land and retire before the dispatch takes the resolution: then
/// the ownership answers alone hold again, but the bytes read may be the ones from before the store. The take checks
/// the GPU write log too (IsStillCurrentAcrossDispatches), and a draw's take does not.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private static readonly FieldInfo GpuWritesOfHost =
        typeof(MetalRenderHost).GetField("_gpuWrites", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static object ResolveWordConcurrently(MetalRenderHost host, ulong address)
    {
        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        bool read;
        try
        {
            read = host.TryReadGuestWord(address, out _);
        }
        catch
        {
            concurrent.ReleaseAssumptions(concurrent.EndConcurrentResolve());
            throw;
        }

        var assumptions = concurrent.EndConcurrentResolve();
        Assert.True(read);
        return assumptions;
    }

    [Fact]
    public void ADispatchResolutionIsRefusedWhenAWriteLoggedAfterItCoversWhatItReadEvenOnceTheWriteRetired()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        WriteKernelSource(rig);
        var concurrent = (IConcurrentResolveHost)host;
        var assumptions = ResolveWordConcurrently(host, KernelSource + 4);

        host.NoteGuestMemoryWrite(KernelSource + 0x10, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        // The store landed and its record retired: the page is unowned again.
        ((MetalGpuWriteRanges)GpuWritesOfHost.GetValue(host)!).Clear(ulong.MaxValue);
        // Every take tests the log: a write the worker recorded after the resolution began is ahead of its draw too.
        Assert.False(concurrent.IsStillCurrent(assumptions));
        Assert.False(concurrent.IsStillCurrentAcrossDispatches(assumptions));
        concurrent.ReleaseAssumptions(assumptions);
    }

    [Fact]
    public void ADispatchResolutionStandsAcrossAWriteToBytesItDidNotRead()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        WriteKernelSource(rig);
        var concurrent = (IConcurrentResolveHost)host;
        var assumptions = ResolveWordConcurrently(host, KernelSource + 4);

        // Another page entirely: neither the answers nor the log cover what the resolution read.
        host.NoteGuestMemoryWrite(KernelSource + 0x4000, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.True(concurrent.IsStillCurrentAcrossDispatches(assumptions));
        concurrent.ReleaseAssumptions(assumptions);
    }

    // A resolution (a draw's resolved ahead of dispatches, ParallelProgramPrefetch.SubmitDrawAhead, or a dispatch's) read
    // a destination word as the source of the pending copy that writes it, dropping the copy's owned answer. A dispatch
    // it was resolved ahead of then stores: over the word read (that store, not the copy, is now the word's latest
    // write), over the copy's source (the word would be copied from other bytes), or elsewhere. The first two refuse
    // the resolution, pending or retired; the last leaves it standing.
    [Theory]
    [InlineData(0x0UL, true, false, false)]
    [InlineData(0x0UL, true, true, false)]
    [InlineData(0x4UL, false, false, false)]
    [InlineData(0x4UL, false, true, false)]
    [InlineData(0x4000UL, false, false, true)]
    public void AWordReadThroughAPendingCopyIsRefusedWhenADispatchAheadStoresOverItOrItsSource(
        ulong offset, bool overDestination, bool retired, bool stands)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentCopyReads = true;
        WriteKernelSource(rig);
        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        host.NoteGuestMemoryWrite(KernelDestination, KernelWords * sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        host.ResetBindings();
        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            Assert.True(host.TryReadGuestWord(KernelDestination + 4, out var word));
            Assert.Equal(rig.ReadBacking32(KernelSource + 4), word);
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        // The dispatch ahead runs: a 4-byte store over the destination word read, the copy's source word, or another page.
        var stored = overDestination ? KernelDestination + 4 + offset : KernelSource + offset;
        host.NoteGuestMemoryWrite(stored, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        if (retired)
        {
            ((MetalGpuWriteRanges)GpuWritesOfHost.GetValue(host)!).Clear(ulong.MaxValue);
        }

        Assert.Equal(stands, concurrent.IsStillCurrentAcrossDispatches(assumptions));
        concurrent.ReleaseAssumptions(assumptions);
    }
}
