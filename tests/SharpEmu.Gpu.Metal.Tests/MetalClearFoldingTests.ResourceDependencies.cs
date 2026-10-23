// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalClearFoldingTests
{
    [Fact]
    public void DependencyEpochTracksGpuImageOwnershipAndPendingLabelOverlays()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var source = (IResourceDependencySource)host;
        using (host.BeginPreparation())
        {
            var request = Rig.Request(TargetAddress, ImageRole.Texture);
            var image = host.ImageCache.FindImage(ref request);
            var before = source.DependencyEpoch;
            host.ImageCache.MarkGpuWritten(image);
            Assert.NotEqual(before.Images, source.DependencyEpoch.Images);

            before = source.DependencyEpoch;
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, GuestBase + 0x50000, 123));
            Assert.NotEqual(before.Labels, source.DependencyEpoch.Labels);
        }
        host.FlushAndWaitForGpu();
    }

    [Fact]
    public void AnUnobservedNativeRangeCannotAcquireADependencyCertificate()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.WriteBacking32(GuestBase + 0x50000, 123);
        Assert.False(((IResourceDependencySource)rig.Host).TryCaptureDependency(GuestBase + 0x50000,
            BitConverter.GetBytes(123u), clean: false, out var dependency));
        Assert.Null(dependency);
    }
}
