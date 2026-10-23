// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.Tests.Scenes;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalShaderIdentityWriteTests(ITestOutputHelper output) : SceneTestBase(output)
{
    [Fact]
    public void GpuCopyOverCodeInvalidatesIdentityBeforeItsBytesArePublished()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var code = rig.Memory.Upload<uint>([0xBF810000u], 4096);
        var source = rig.Memory.Upload<uint>([0xBF800000u], 4096);
        var registered = new RegisteredShader(code, 0, 4, 0, 0, 0, 0, 0, 0);
        var cache = new ShaderIdentityCache(rig.Memory, rig.Host.SynchronizeShaderCode, GuestGpuMemoryHook.Current!.WriteGenerations);
        var first = cache.Get(registered, "compute");
        rig.Host.CopyGuestMemory(code, source, 4, allowHost: false);
        var second = cache.Get(registered, "compute");
        Assert.NotEqual(first.ContentHash, second.ContentHash);
        Assert.Same(second, cache.Get(registered, "compute"));
    }
}
