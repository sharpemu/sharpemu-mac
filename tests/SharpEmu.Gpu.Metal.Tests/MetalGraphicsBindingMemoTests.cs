// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalGraphicsBindingMemoTests
{
    [Fact]
    public void OnlyAnExactlyRecordedBindingMatches()
    {
        var memo = new MetalGraphicsBindingMemo();
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 256));
        // An attempted native call is not a recorded success.
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 256));
        memo.Record(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 256);
        Assert.True(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 256));
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 512));
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 8, 256));
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Vertex, 12, 0, 7, 256));
        memo.Record(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 512);
        Assert.True(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 512));
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 256));
    }

    [Fact]
    public void PhysicalStagesAndBothSlotsKeepIndependentState()
    {
        var memo = new MetalGraphicsBindingMemo();
        memo.Record(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 256);
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 1, 7, 256));
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Mesh, 11, 0, 7, 256));
        Assert.False(memo.Matches(MetalGraphicsBindingStage.Fragment, 11, 0, 7, 256));
        foreach (var stage in Enum.GetValues<MetalGraphicsBindingStage>())
            for (var slot = 0; slot < 2; slot++)
                memo.Record(stage, 11 + (int)stage, slot, 7 + slot, (ulong)(256 + slot * 256));
        foreach (var stage in Enum.GetValues<MetalGraphicsBindingStage>())
            for (var slot = 0; slot < 2; slot++)
                Assert.True(memo.Matches(stage, 11 + (int)stage, slot, 7 + slot, (ulong)(256 + slot * 256)));
        // Returning from mesh to vertex still uses vertex's distinct physical binding.
        Assert.True(memo.Matches(MetalGraphicsBindingStage.Vertex, 11, 0, 7, 256));
    }

    [Fact]
    public void ExplicitEncoderResetForgetsBindingsEvenWhenNativePointersRepeat()
    {
        var memo = new MetalGraphicsBindingMemo();
        foreach (var stage in Enum.GetValues<MetalGraphicsBindingStage>())
            for (var slot = 0; slot < 2; slot++) memo.Record(stage, 11, slot, 7, 256);
        memo.Reset();
        foreach (var stage in Enum.GetValues<MetalGraphicsBindingStage>())
            for (var slot = 0; slot < 2; slot++) Assert.False(memo.Matches(stage, 11, slot, 7, 256));
        memo.Record(MetalGraphicsBindingStage.Fragment, 11, 0, 7, 256);
        Assert.True(memo.Matches(MetalGraphicsBindingStage.Fragment, 11, 0, 7, 256));
    }
}
