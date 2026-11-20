// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// A colour target rendered at a mip level past its extent's 1x1 level: a 4x4 surface has three levels (4, 2, 1), and
/// CB_COLOR_VIEW names level 3. The request keeps MipLevel + 1 levels (ImageRequestBuilders.ColorTarget), which a Metal
/// texture of that extent cannot have. Upstream's host keeps the texture to its chain (CachedImage: MipLevels clamped to
/// MipChainLevels) and renders such a level into the last level the texture has (ClampToHostLevels), which is 1x1 too.
/// </summary>
public sealed class ColorTargetLevelSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 4;
    private const uint Fmt8888 = 10;
    private const uint NtUnorm = 0;
    private const uint SurfaceBytes = 0x20000;

    // Texture-tiled colour targets (64KB_S, 4KB_S): their layout covers the level past the chain, so the request reaches
    // the host. A render-target-swizzled one (64KB_R_X) is refused by the shared request builder on every host
    // (TryGetRenderTargetMipLayout has no layout for it: "The render-target layout is not supported").
    [Theory]
    [InlineData(9u)]
    [InlineData(5u)]
    public void ADrawToALevelPastTheChainRendersIntoTheLastLevel(uint tileMode)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var target = rig.Memory.Allocate(SurfaceBytes, 0x10000);
        rig.Memory.Write(target, new byte[SurfaceBytes]);
        var state = new SceneState(1, 1).ColorTarget(0, target, Size, Size, Fmt8888, NtUnorm, tileMode: tileMode,
            mipLevel: 3, maxMip: 3, blendBypass: true);
        rig.Submit(rig.DrawMesh(new ScenePm4(), state, SceneDraws.Quad(0.5f, 1, 0, 0, 1)));

        // The 1x1 texel the draw wrote lands in the surface's guest bytes once; nothing else does.
        var bytes = rig.Read(target, (int)SurfaceBytes);
        var red = 0;
        var other = 0;
        for (var offset = 0; offset < bytes.Length; offset += 4)
        {
            var word = BitConverter.ToUInt32(bytes, offset);
            red += word == 0xFF0000FFu ? 1 : 0;
            other += word != 0 && word != 0xFF0000FFu ? 1 : 0;
        }

        Assert.Equal(0, other);
        Assert.Equal(1, red);
    }
}
