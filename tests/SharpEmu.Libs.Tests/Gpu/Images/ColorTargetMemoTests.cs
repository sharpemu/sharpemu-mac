// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Images;

[Collection(SchedulingStateCollection.Name)]
public sealed class ColorTargetMemoTests
{
    [Fact]
    public void EquivalentMasksAndCollidingTargetsReuseBuiltLayouts()
    {
        // These addresses collide in the original eight-slot address hash.
        var targets = Enumerable.Range(0, 4).Select(i =>
            RegisterWords.Color(0x31000000UL + (ulong)i * 0x800000, 64, 64)).ToArray();
        foreach (var words in targets)
        {
            Assert.NotNull(ImageRequestBuilders.ColorTarget(words, 1, 0, false));
        }

        ulong sum = 0;
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var repeat = 0; repeat < 128; repeat++)
        {
            foreach (var words in targets)
            {
                var result = ImageRequestBuilders.ColorTarget(words, 0xFFFF, 0, true)!.Value;
                sum += result.Request.Description.MipLayout[0].Size;
            }
        }
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(sum > 0);
        Assert.True(allocated <= 512, $"Warm equivalent targets allocated {allocated} bytes.");
    }

    [Fact]
    public void NativeMetadataModeIsPartOfTheMemoKey()
    {
        var previous = ImageRequestBuilders.NativeColorClearMetadata;
        var words = RegisterWords.Color(0x32000000, 64, 64, GuestTileMode.RenderTarget);
        words = words with { Info = words.Info | (1u << 13), CmaskAddress = 0x34000000, ClearWord0 = 0xff000000 };
        try
        {
            ImageRequestBuilders.NativeColorClearMetadata = true;
            var enabled = ImageRequestBuilders.ColorTarget(words, 15, 0, false)!.Value;
            Assert.True(enabled.Request.Description.Metadata.NativeColorClear);
            ImageRequestBuilders.NativeColorClearMetadata = false;
            var disabled = ImageRequestBuilders.ColorTarget(words, 15, 0, false)!.Value;
            Assert.False(disabled.Request.Description.Metadata.NativeColorClear);
            Assert.Equal(MetadataKind.None, disabled.Request.Description.Metadata.Kind);
            ImageRequestBuilders.NativeColorClearMetadata = true;
            Assert.True(ImageRequestBuilders.ColorTarget(words, 15, 0, false)!.Value.Request.Description.Metadata.NativeColorClear);
        }
        finally
        {
            ImageRequestBuilders.NativeColorClearMetadata = previous;
        }
    }

    [Fact]
    public void MaskedTargetAndChangedRegistersDoNotReuseAnActiveRequest()
    {
        var words = RegisterWords.Color(0x35000000, 64, 64);
        var active = ImageRequestBuilders.ColorTarget(words, 1, 0, false)!.Value;
        Assert.Null(ImageRequestBuilders.ColorTarget(words, 0xF0, 0, false));
        Assert.NotNull(ImageRequestBuilders.ColorTarget(words, 0, 0, true));
        Assert.Null(ImageRequestBuilders.ColorTarget(words with { BaseAddress = 0 }, 0, 0, true));
        var changed = RegisterWords.Color(words.BaseAddress, 128, 32);
        var resolved = ImageRequestBuilders.ColorTarget(changed, 2, 0, false)!.Value;
        Assert.Equal(128u, resolved.Extent.Width);
        Assert.Equal(32u, resolved.Extent.Height);
        Assert.Equal(64u, active.Extent.Width);
        var copied = active.Request.Description;
        copied.MipLayout[0] = default;
        Assert.NotEqual(0UL, ImageRequestBuilders.ColorTarget(words, 15, 0, false)!.Value.Request.Description.MipLayout[0].Size);
    }
}
