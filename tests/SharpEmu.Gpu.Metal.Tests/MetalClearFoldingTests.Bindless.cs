// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalClearFoldingTests
{
    [Fact]
    public void AReadOnlyDeviceBufferPublishesAnImageAliasBeforeItReadsGuestMemory()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var info = new ShaderResourceInfo
        {
            UsesDeviceAddresses = true, ReadOnlyDeviceBuffers = true,
            Buffers = [new BufferResource { Read = true }],
        };
        var layout = BindingLayout.Allocate(info, [], false, false, false, 0);
        Assert.Null(layout.Find(DescriptorBindingKind.Buffers));
        var program = new ShaderProgramInfo
        {
            Stage = ShaderStageKind.Compute, Hash = 0xB1AD1E55,
            Resources = new SpecializedResourceInfo { Info = info }, Bindings = layout,
            UsesDeviceAddresses = true,
        };
        var snapshot = new ResourceSnapshot
        {
            Buffers = [[unchecked((uint)TargetAddress), (uint)(TargetAddress >> 32), (uint)ImageBytes,
                DescriptorConstants.IdentityDestinationSelect | (20u << 12) | (3u << 28)]],
        };
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            rig.Clear((nint)attachment.View.Handle);
            Assert.True(host.ImageCache.HasGpuModifiedImageBytes(TargetAddress, ImageBytes));
            host.BindResources(host.PrepareBindings(new ShaderStageResources(program, snapshot)));
            Assert.False(host.ImageCache.HasGpuModifiedImageBytes(TargetAddress, ImageBytes));
        }
        AssertCleared(rig.Read(TargetAddress));
    }
}
