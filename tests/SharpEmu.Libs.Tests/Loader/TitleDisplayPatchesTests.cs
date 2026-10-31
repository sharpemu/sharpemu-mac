// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class TitleDisplayPatchesTests
{
    private static readonly TitleDisplayPatches.Site[] Sites =
    [
        new(0x100, TitleDisplayPatches.Slot.Width, 3840, 2160, TitleDisplayPatches.Role.Output),
        new(0x104, TitleDisplayPatches.Slot.Height, 3840, 2160, TitleDisplayPatches.Role.Output),
        new(0x108, TitleDisplayPatches.Slot.WidthHeight, 3840, 2160, TitleDisplayPatches.Role.Output),
        new(0x110, TitleDisplayPatches.Slot.WidthF32, 3840, 2160, TitleDisplayPatches.Role.Output),
        new(0x114, TitleDisplayPatches.Slot.HeightF32, 3840, 2160, TitleDisplayPatches.Role.Output),
        new(0x118, TitleDisplayPatches.Slot.Width, 2560, 1440, TitleDisplayPatches.Role.Internal),
        new(0x11C, TitleDisplayPatches.Slot.Height, 2560, 1440, TitleDisplayPatches.Role.Internal),
    ];

    [Fact]
    public void BuildsEveryEncodingForA1080pDisplayAndShrinksTheInternalSize()
    {
        var patches = TitleDisplayPatches.BuildPatches(Sites, 1920, 1080);

        Assert.Equal(7, patches.Count);
        Assert.Equal(Le32(3840), patches[0].Old);
        Assert.Equal(Le32(1920), patches[0].New);
        Assert.Equal(Le32(2160), patches[1].Old);
        Assert.Equal(Le32(1080), patches[1].New);
        Assert.Equal(Le64(3840 | (2160UL << 32)), patches[2].Old);
        Assert.Equal(Le64(1920 | (1080UL << 32)), patches[2].New);
        Assert.Equal(F32(3840f), patches[3].Old);
        Assert.Equal(F32(1920f), patches[3].New);
        Assert.Equal(F32(2160f), patches[4].Old);
        Assert.Equal(F32(1080f), patches[4].New);
        Assert.Equal(Le32(2560), patches[5].Old);
        Assert.Equal(Le32(1920), patches[5].New);
        Assert.Equal(Le32(1440), patches[6].Old);
        Assert.Equal(Le32(1080), patches[6].New);
    }

    [Fact]
    public void A1440pDisplayLeavesA1440pInternalSizeAlone()
    {
        var patches = TitleDisplayPatches.BuildPatches(Sites, 2560, 1440);

        Assert.Equal(5, patches.Count);
        Assert.DoesNotContain(patches, patch => patch.Offset >= 0x118);
    }

    [Fact]
    public void ADisplayAtTheCompiledSizeNeedsNoPatch()
    {
        Assert.Empty(TitleDisplayPatches.BuildPatches(Sites, 3840, 2160));
    }

    [Fact]
    public void KnownTitlesHaveSitesAndOthersDoNot()
    {
        Assert.NotNull(TitleDisplayPatches.SitesFor("PPSA01342"));
        Assert.Null(TitleDisplayPatches.SitesFor("PPSA00000"));
        Assert.Null(TitleDisplayPatches.SitesFor(null));
    }

    [Fact]
    public void ApplyRewritesVerifiedSitesAndSkipsEverythingOnAMismatch()
    {
        using var memory = new SharpEmu.Core.Memory.PhysicalVirtualMemory();
        Assert.True(memory.TryAllocateAtOrAbove(0x200000000UL, 0x10000, executable: false, 0x4000, out var imageBase));
        ProgramHeader[] headers = [CreateLoad(0, 0x4000)];
        var sites = TitleDisplayPatches.SitesFor("PPSA01342")!;
        foreach (var site in sites)
        {
            Assert.True(memory.TryWrite(imageBase + (site.Offset & 0x3FFF), Encode(site, site.OldWidth, site.OldHeight)));
        }

        // The real sites lie beyond this small image, so apply through relocated copies.
        var relocated = sites.Select(site => site with { Offset = site.Offset & 0x3FFF }).ToArray();
        var patches = TitleDisplayPatches.BuildPatches(relocated, 1920, 1080);
        GuestBytePatcher.Apply(memory, headers, imageBase, patches);
        var first = new byte[4];
        Assert.True(memory.TryRead(imageBase + (relocated[0].Offset), first));
        Assert.Equal(Le32(1920), first);

        // A second application sees 1920 where 3840 was expected and touches nothing.
        Assert.False(TitleDisplayPatches.Apply(memory, headers, imageBase, "PPSA00000", 1920, 1080));
        Assert.False(TitleDisplayPatches.Apply(memory, headers, imageBase, "PPSA01342", 1920, 1080));
    }

    [Theory]
    [InlineData(null, "AllSites")]
    [InlineData("1", "AllSites")]
    [InlineData("0", "Off")]
    [InlineData("mode", "DisplayMode")]
    public void TheVariableSelectsTheStrategyAndUnsetKeepsTheFastRewrite(
        string? value, string expected)
    {
        var previous = Environment.GetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, value);
            Assert.Equal(expected, TitleDisplayPatches.SelectedStrategy.ToString());
        }
        finally
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, previous);
        }
    }

    // =0 keeps the title's own sizes, as a PS5 runs it; unset rewrites every site.
    [Theory]
    [InlineData(1280u, 720u)]
    [InlineData(1920u, 1080u)]
    [InlineData(2560u, 1440u)]
    public void TheTitlesOwnDisplaySizesAreKeptOnlyWhenThePatchIsTurnedOff(uint width, uint height)
    {
        var sites = TitleDisplayPatches.SitesFor("PPSA01342")!;
        var imageSize = (sites.Max(site => site.Offset) + 0x10000) & ~0xFFFFUL;
        using var memory = new SharpEmu.Core.Memory.PhysicalVirtualMemory();
        Assert.True(memory.TryAllocateAtOrAbove(0x200000000UL, imageSize, executable: false, 0x4000, out var imageBase));
        ProgramHeader[] headers = [CreateLoad(0, imageSize)];
        foreach (var site in sites)
        {
            Assert.True(memory.TryWrite(imageBase + site.Offset, Encode(site, site.OldWidth, site.OldHeight)));
        }

        WriteImageScaleSites(memory, imageBase);
        var previous = Environment.GetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, "0");
            Assert.False(TitleDisplayPatches.Apply(memory, headers, imageBase, "PPSA01342", width, height));
            foreach (var site in sites)
            {
                var expected = Encode(site, site.OldWidth, site.OldHeight);
                var actual = new byte[expected.Length];
                Assert.True(memory.TryRead(imageBase + site.Offset, actual));
                Assert.Equal(expected, actual);
            }

            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, null);
            Assert.True(TitleDisplayPatches.Apply(memory, headers, imageBase, "PPSA01342", width, height));
            var canvas = sites.First(site => site.Slot == TitleDisplayPatches.Slot.WidthF32);
            var rewritten = new byte[4];
            Assert.True(memory.TryRead(imageBase + canvas.Offset, rewritten));
            Assert.Equal(F32(width), rewritten);
        }
        finally
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, previous);
        }
    }

    [Theory]
    [InlineData(1920u, 1080u)]
    [InlineData(2560u, 1440u)]
    [InlineData(1280u, 720u)]
    public void TheDisplayModeStrategyHoldsTheHdModeAndShrinksOnlyTheSceneSize(uint width, uint height)
    {
        var profile = TitleDisplayPatches.DisplayModeFor("PPSA01342")!;
        Assert.Null(TitleDisplayPatches.DisplayModeFor("PPSA00000"));
        Assert.Equal(profile.HdMode, profile.ModeFor(width, height));

        var patches = TitleDisplayPatches.BuildDisplayModePatches(profile, width, height);
        foreach (var offset in profile.DefaultModeSites)
        {
            var patch = Assert.Single(patches, patch => patch.Offset == offset);
            Assert.Equal(Le32(3), patch.Old);
            Assert.Equal(Le32(2), patch.New);
        }

        // The saved visual mode cannot take the title back to 4K: every override is dropped.
        foreach (var site in profile.OverrideSites)
        {
            var patch = Assert.Single(patches, patch => patch.Offset == site.Offset);
            Assert.Equal(0x0F, patch.Old[0]);
            Assert.Equal(0x45, patch.Old[1]);
            Assert.Equal(new byte[] { 0x0F, 0x1F, 0x00 }, patch.New);
        }

        // The display buffers and the render-config ceiling the HD mode still compiles at 4K
        // become the HD size whatever the display.
        var bufferOffsets = profile.HdBufferSites.Select(site => site.Offset).ToHashSet();
        Assert.Equal(7, bufferOffsets.Count);
        foreach (var site in profile.HdBufferSites)
        {
            var patch = Assert.Single(patches, patch => patch.Offset == site.Offset);
            Assert.Equal(Encode(site, 3840, 2160), patch.Old);
            Assert.Equal(Encode(site, 1920, 1080), patch.New);
        }

        var sizePatches = patches
            .Where(patch => !profile.DefaultModeSites.Contains(patch.Offset)
                && profile.OverrideSites.All(site => site.Offset != patch.Offset)
                && !bufferOffsets.Contains(patch.Offset))
            .ToList();
        if (height >= 1080)
        {
            Assert.Empty(sizePatches);
        }
        else
        {
            Assert.Equal(
                profile.HdInternalSites.Select(site => site.Offset).Order(),
                sizePatches.Select(patch => patch.Offset).Order());
            Assert.All(sizePatches.Where(patch => patch.Old.SequenceEqual(Le32(1920))),
                patch => Assert.Equal(Le32(width), patch.New));
            Assert.All(sizePatches.Where(patch => patch.Old.SequenceEqual(Le32(1080))),
                patch => Assert.Equal(Le32(height), patch.New));
        }

        // The title composites its UI at the HD output size: the output, the display-buffer
        // views it registers with video out and the UI's own sizes all stay 1920x1080.
        Assert.DoesNotContain(patches, patch => HdOutputOffsets.Contains(patch.Offset));

        // Nothing the UI lays out in is touched: the canvas tables, the 4K literals and the
        // display-size getters keep the title's values. Of the all-sites set, only the display
        // buffers and the ceiling change.
        var allSiteOffsets = TitleDisplayPatches.SitesFor("PPSA01342")!.Select(site => site.Offset).ToHashSet();
        Assert.Subset(allSiteOffsets, bufferOffsets);
        Assert.DoesNotContain(patches, patch => allSiteOffsets.Contains(patch.Offset) && !bufferOffsets.Contains(patch.Offset));
        Assert.DoesNotContain(patches, patch => patch.Offset is 0x2172C00 or 0x2172A08);
    }

    [Fact]
    public void A4KDisplayKeepsTheTitlesOwnMode()
    {
        var profile = TitleDisplayPatches.DisplayModeFor("PPSA01342")!;
        Assert.Null(profile.ModeFor(3840, 2160));
        Assert.Empty(TitleDisplayPatches.BuildDisplayModePatches(profile, 3840, 2160));
    }

    [Fact]
    public void ApplyInDisplayModeAt720pShrinksOnlyTheSceneAndKeepsTheHdOutput()
    {
        var profile = TitleDisplayPatches.DisplayModeFor("PPSA01342")!;
        var sites = TitleDisplayPatches.SitesFor("PPSA01342")!;
        var imageSize = (sites.Max(site => site.Offset) + 0x10000) & ~0xFFFFUL;
        using var memory = new SharpEmu.Core.Memory.PhysicalVirtualMemory();
        Assert.True(memory.TryAllocateAtOrAbove(0x200000000UL, imageSize, executable: false, 0x4000, out var imageBase));
        ProgramHeader[] headers = [CreateLoad(0, imageSize)];
        foreach (var site in sites.Concat(profile.HdInternalSites))
        {
            Assert.True(memory.TryWrite(imageBase + site.Offset, Encode(site, site.OldWidth, site.OldHeight)));
        }

        foreach (var (offset, hdValue) in HdOutputValues)
        {
            Assert.True(memory.TryWrite(imageBase + offset, Le32(hdValue)));
        }

        foreach (var offset in profile.DefaultModeSites)
        {
            Assert.True(memory.TryWrite(imageBase + offset, Le32(3)));
        }

        foreach (var site in profile.OverrideSites)
        {
            Assert.True(memory.TryWrite(imageBase + site.Offset, site.Old));
        }

        var previous = Environment.GetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, "mode");
            Assert.True(TitleDisplayPatches.Apply(memory, headers, imageBase, "PPSA01342", 1280, 720));
        }
        finally
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, previous);
        }

        var value = new byte[4];
        foreach (var offset in profile.DefaultModeSites)
        {
            Assert.True(memory.TryRead(imageBase + offset, value));
            Assert.Equal(Le32(2), value);
        }

        var code = new byte[3];
        foreach (var site in profile.OverrideSites)
        {
            Assert.True(memory.TryRead(imageBase + site.Offset, code));
            Assert.Equal(site.New, code);
        }

        foreach (var site in profile.HdInternalSites)
        {
            Assert.True(memory.TryRead(imageBase + site.Offset, value));
            Assert.Equal(Le32(site.Slot == TitleDisplayPatches.Slot.Width ? 1280u : 720u), value);
        }

        foreach (var (offset, expected) in HdOutputValues)
        {
            Assert.True(memory.TryRead(imageBase + offset, value));
            Assert.Equal(Le32(expected), value);
        }

        foreach (var site in profile.HdBufferSites)
        {
            var expected = Encode(site, 1920, 1080);
            var actual = new byte[expected.Length];
            Assert.True(memory.TryRead(imageBase + site.Offset, actual));
            Assert.Equal(expected, actual);
        }

        var bufferOffsets = profile.HdBufferSites.Select(site => site.Offset).ToHashSet();
        foreach (var site in sites.Where(site => !bufferOffsets.Contains(site.Offset)))
        {
            var expected = Encode(site, site.OldWidth, site.OldHeight);
            var actual = new byte[expected.Length];
            Assert.True(memory.TryRead(imageBase + site.Offset, actual));
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TheImageScaleStandInDividesTheTexturesCanvasAndEveryCallerReachesIt()
    {
        var sites = TitleDisplayPatches.ImageScaleSitesFor("PPSA01342")!;
        Assert.Null(TitleDisplayPatches.ImageScaleSitesFor("PPSA00000"));

        // mov eax, 2160.0f; vmovd xmm0, eax; jmp back into the ratio after its canvas load.
        var standIn = sites[0];
        Assert.All(standIn.Old, value => Assert.Equal(0xCC, value));
        Assert.Equal(0xB8, standIn.New[0]);
        Assert.Equal(2160f, BinaryPrimitives.ReadSingleLittleEndian(standIn.New.AsSpan(1)));
        Assert.Equal(0xEB, standIn.New[^2]);
        Assert.Equal(0xEA5FD5UL, standIn.Offset + (ulong)standIn.New.Length + unchecked((ulong)(sbyte)standIn.New[^1]));

        foreach (var call in sites.Skip(1))
        {
            Assert.Equal(0xEA5FD0UL, CallTarget(call.Offset, call.Old));
            Assert.Equal(standIn.Offset, CallTarget(call.Offset, call.New));
        }
    }

    [Theory]
    [InlineData(1920u, 1080u, true)]
    [InlineData(3840u, 2160u, false)]
    public void TheAllSitesRewriteRedirectsTheImageScaleOnlyWhenTheCanvasShrinks(uint width, uint height, bool redirected)
    {
        var sites = TitleDisplayPatches.SitesFor("PPSA01342")!;
        var imageSize = (sites.Max(site => site.Offset) + 0x10000) & ~0xFFFFUL;
        using var memory = new SharpEmu.Core.Memory.PhysicalVirtualMemory();
        Assert.True(memory.TryAllocateAtOrAbove(0x200000000UL, imageSize, executable: false, 0x4000, out var imageBase));
        ProgramHeader[] headers = [CreateLoad(0, imageSize)];
        foreach (var site in sites)
        {
            Assert.True(memory.TryWrite(imageBase + site.Offset, Encode(site, site.OldWidth, site.OldHeight)));
        }

        WriteImageScaleSites(memory, imageBase);
        var previous = Environment.GetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, null);
            Assert.Equal(redirected, TitleDisplayPatches.Apply(memory, headers, imageBase, "PPSA01342", width, height));
            foreach (var site in TitleDisplayPatches.ImageScaleSitesFor("PPSA01342")!)
            {
                var actual = new byte[site.New.Length];
                Assert.True(memory.TryRead(imageBase + site.Offset, actual));
                Assert.Equal(redirected ? site.New : site.Old, actual);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(TitleDisplayPatches.EnvironmentVariable, previous);
        }
    }

    private static void WriteImageScaleSites(SharpEmu.Core.Memory.PhysicalVirtualMemory memory, ulong imageBase)
    {
        foreach (var site in TitleDisplayPatches.ImageScaleSitesFor("PPSA01342")!)
        {
            Assert.True(memory.TryWrite(imageBase + site.Offset, site.Old));
        }
    }

    private static ulong CallTarget(ulong offset, byte[] call)
    {
        Assert.Equal(0xE8, call[0]);
        return (ulong)((long)offset + 5 + BinaryPrimitives.ReadInt32LittleEndian(call.AsSpan(1)));
    }

    // Demon's Souls' HD output size (output table index 1) and the 1920x1080 views of its
    // display buffers that it registers with sceVideoOutSetBufferAttribute2 in the HD mode.
    private static readonly (ulong Offset, uint Value)[] HdOutputValues =
    [
        (0x1FFA5D4, 1920), (0x1FFBE34, 1080),
        (0x9453E2, 1920), (0x9453E7, 1080),
        (0x94547B, 1920), (0x945480, 1080),
    ];

    private static readonly HashSet<ulong> HdOutputOffsets = HdOutputValues.Select(site => site.Offset).ToHashSet();

    private static byte[] Encode(TitleDisplayPatches.Site site, uint width, uint height) =>
        site.Slot switch
        {
            TitleDisplayPatches.Slot.Width => Le32(width),
            TitleDisplayPatches.Slot.Height => Le32(height),
            TitleDisplayPatches.Slot.WidthHeight => Le64(width | ((ulong)height << 32)),
            TitleDisplayPatches.Slot.WidthF32 => F32(width),
            TitleDisplayPatches.Slot.HeightF32 => F32(height),
            _ => throw new ArgumentOutOfRangeException(nameof(site)),
        };

    private static byte[] Le32(uint value)
    {
        var bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] Le64(ulong value)
    {
        var bytes = new byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] F32(float value)
    {
        var bytes = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        return bytes;
    }

    private static ProgramHeader CreateLoad(ulong virtualAddress, ulong size) =>
        GuestBytePatcherTests.CreateLoad(virtualAddress, size);
}
