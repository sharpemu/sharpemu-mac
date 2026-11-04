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
}
