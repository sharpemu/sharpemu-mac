// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalDrawPathResidencyTests
{
    [Fact]
    public void DrawPathResidencyRequiresEncoderSetSubmissionAndStageCoverage()
    {
        var stamp = new MetalResidencyStamp();
        Assert.False(stamp.Covers(0, 0, 0, 0));
        stamp.Record(1, 2, 3, Mtl.RenderStageVertex);
        Assert.True(stamp.Covers(1, 2, 3, Mtl.RenderStageVertex));
        Assert.False(stamp.Covers(2, 2, 3, Mtl.RenderStageVertex));
        Assert.False(stamp.Covers(1, 4, 3, Mtl.RenderStageVertex));
        Assert.False(stamp.Covers(1, 2, 4, Mtl.RenderStageVertex));
        Assert.False(stamp.Covers(1, 2, 3, Mtl.RenderStageFragment));
        stamp.Record(1, 2, 3, Mtl.RenderStageFragment);
        Assert.True(stamp.Covers(1, 2, 3, Mtl.RenderStageVertex | Mtl.RenderStageFragment));
        Assert.False(stamp.Covers(1, 2, 3, Mtl.RenderStageMesh | Mtl.RenderStageFragment));
        stamp.Record(1, 2, 3, Mtl.RenderStageMesh | Mtl.RenderStageFragment);
        Assert.True(stamp.Covers(1, 2, 3, Mtl.RenderStageMesh | Mtl.RenderStageFragment));
        // Compute's stage mask is zero, and its new encoder must still be named once.
        Assert.False(stamp.Covers(2, 2, 3, 0));
        stamp.Record(2, 2, 3, 0);
        Assert.True(stamp.Covers(2, 2, 3, 0));
        Assert.False(stamp.Covers(2, 2, 3, Mtl.RenderStageFragment));
        stamp.Record(2, 4, 3, Mtl.RenderStageFragment);
        Assert.False(stamp.Covers(2, 4, 3, Mtl.RenderStageVertex));
    }
}
