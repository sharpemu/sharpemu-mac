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
        Assert.True(concurrent.IsStillCurrent(assumptions));
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
}
