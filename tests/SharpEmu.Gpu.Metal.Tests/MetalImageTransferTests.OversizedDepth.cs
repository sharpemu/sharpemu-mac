// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed partial class MetalImageTransferTests
{
    // Dynamic resolution keeps a depth surface's full-size allocation and describes a smaller extent over it: Ghost of
    // Tsushima's 1600x1800 checkerboard depth in the bytes of a 1920x2160 one. The one slice is larger than the extent's
    // tile layout, which holds the image's tiles first; the transfer used to stop the GPU process on the mismatch.
    [Fact]
    public void ADepthSurfaceLargerThanItsTileLayoutRoundTripsItsLayoutsTiles()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var fitted = Depth(GuestDepthFormat.Z32Float, stencil: false, 96, 48, 1);
        var layout = BuildGuest(fitted, 81);
        var oversized = fitted;
        oversized.Data = new GuestSpan(fitted.Data.Address, fitted.Data.Size + 0x10000);
        var guest = new byte[oversized.Data.Size];
        layout.CopyTo(guest, 0);

        var source = harness.Buffer(guest);
        var target = harness.Buffer(oversized.Data.Size);
        var texture = harness.Texture(harness.Transfer.ShapeFor(oversized));
        var range = new SubresourceRange(0, 1, 0, 1);
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, oversized, range, new MetalGuestRange(source, 0, oversized.Data.Size, 0), default,
            texture, 0, oversized.Data.Size);
        harness.Submit();
        harness.Begin();
        harness.Transfer.EncodeTextureToGuest(harness, oversized, range, texture, new MetalGuestRange(target, 0, oversized.Data.Size, 0), default,
            0, oversized.Data.Size);
        harness.Submit();

        // The layout's tiles come back; the bytes past it are not part of the image and stay as they were.
        Assert.Equal(guest, harness.Read(target, 0, oversized.Data.Size));
    }
}
