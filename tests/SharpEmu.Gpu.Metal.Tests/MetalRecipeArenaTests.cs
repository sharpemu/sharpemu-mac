// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalRecipeArenaTests
{
    [Fact]
    public unsafe void IdenticalRecordsShareStorageAndChangedRecordsNeverPatchBusyBytes()
    {
        if (!OperatingSystem.IsMacOS()) return;
        var device = MetalNative.MTLCreateSystemDefaultDevice(); Assert.NotEqual(0, device);
        using var arena = new MetalRecipeArena(device);
        Assert.True(arena.Intern([1, 2, 3, 4], 1, 0, out var first));
        Assert.True(arena.Intern([1, 2, 3, 4], 10, 0, out var repeat));
        Assert.Equal(first.Range, repeat.Range);
        Assert.Equal(10UL, first.Owner.LastSerial);
        Assert.True(arena.Intern([5, 6, 7, 8], 11, 0, out var changed));
        Assert.NotEqual(first.Range, changed.Range);
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, new ReadOnlySpan<byte>((byte*)first.Owner.Contents + first.Offset, 4).ToArray());
        Assert.Equal(new byte[] { 5, 6, 7, 8 }, new ReadOnlySpan<byte>((byte*)changed.Owner.Contents + changed.Offset, 4).ToArray());
    }

    [Fact]
    public unsafe void GpuLagLongerThanThreeFramesCannotRecycleAnOldGeneration()
    {
        if (!OperatingSystem.IsMacOS()) return;
        var device = MetalNative.MTLCreateSystemDefaultDevice(); Assert.NotEqual(0, device);
        using var arena = new MetalRecipeArena(device);
        var bytes = new byte[1024 * 1024];
        MetalRecipeArena.Record first = default;
        for (var i = 0; i < 32; i++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(), i);
            Assert.True(arena.Intern(bytes, (ulong)i + 1, 0, out var current));
            if (i == 0) first = current;
        }
        Assert.True(arena.Generation > 0);
        BitConverter.TryWriteBytes(bytes.AsSpan(), 32);
        Assert.False(arena.Intern(bytes, 33, 0, out _));
        Assert.Equal(0, *(int*)((byte*)first.Owner.Contents + first.Offset));
        Assert.True(arena.Intern(bytes, 33, 32, out _));
    }
}
