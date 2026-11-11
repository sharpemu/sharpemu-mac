// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Buffers;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The image owner index (MetalImagePageOwners): an image smaller than an owner page is listed per 64 KiB fine page,
/// a larger one per 1 MiB owner page, and either way the owner page's version moves and its live count shows it.
/// </summary>
public sealed class MetalImagePageOwnersTests
{
    private const ulong OwnerPage = 1UL << MetalImagePageOwners.PageBits;
    private const ulong FinePage = 1UL << MetalImagePageOwners.FinePageBits;
    private const ulong Base = 0x2_0000_0000;

    [Fact]
    public void SmallImageIsListedOnTheFinePagesItCovers()
    {
        if (!MetalImagePageOwners.FinePagesEnabled)
        {
            return;
        }

        var owners = new MetalImagePageOwners();
        var image = new ResourceSlotIdentifier(1, 1);
        var address = Base + 3 * FinePage + 0x100;
        var page = address >> MetalImagePageOwners.PageBits;
        var versionBefore = owners.VersionSum(address, 1);

        owners.Add(address, FinePage, image);

        Assert.True(owners.Find(page) is null or { Count: 0 });
        Assert.True(owners.HasFineOwners(page));
        Assert.Contains(image, owners.FindFine(address >> MetalImagePageOwners.FinePageBits)!);
        Assert.Contains(image, owners.FindFine((address + FinePage - 1) >> MetalImagePageOwners.FinePageBits)!);
        Assert.True(owners.FindFine((address >> MetalImagePageOwners.FinePageBits) + 2) is null or { Count: 0 });
        Assert.True(owners.MayHaveOwners(address, 1));
        Assert.NotEqual(versionBefore, owners.VersionSum(address, 1));
    }

    [Fact]
    public void SmallImageAcrossAnOwnerPageBoundaryMarksBothOwnerPages()
    {
        if (!MetalImagePageOwners.FinePagesEnabled)
        {
            return;
        }

        var owners = new MetalImagePageOwners();
        var image = new ResourceSlotIdentifier(2, 1);
        var address = Base + 5 * OwnerPage - FinePage / 2;

        owners.Add(address, FinePage, image);

        Assert.True(owners.HasFineOwners(address >> MetalImagePageOwners.PageBits));
        Assert.True(owners.HasFineOwners((address + FinePage - 1) >> MetalImagePageOwners.PageBits));
        Assert.False(owners.HasFineOwners((address >> MetalImagePageOwners.PageBits) - 1));
    }

    [Fact]
    public void LargeImageStaysOnTheOwnerPages()
    {
        var owners = new MetalImagePageOwners();
        var image = new ResourceSlotIdentifier(3, 1);
        var address = Base + 9 * OwnerPage;

        owners.Add(address, 2 * OwnerPage, image);

        for (var page = address >> MetalImagePageOwners.PageBits; page < (address + 2 * OwnerPage) >> MetalImagePageOwners.PageBits; page++)
        {
            Assert.Contains(image, owners.Find(page)!);
            Assert.False(owners.HasFineOwners(page));
        }

        Assert.True(owners.FindFine(address >> MetalImagePageOwners.FinePageBits) is null or { Count: 0 });
    }

    // The fine pages a query walks inside one owner page: from the one holding its first byte to the one holding its
    // last, clipped to the owner page.
    [Theory]
    [InlineData(0x30100ul, 0x40ul, 0b0000_0000_0000_1000u)]
    [InlineData(0x2F000ul, 0x32000ul, 0b0000_0000_0111_1100u)]
    [InlineData(0x0ul, 0x100000ul, 0xFFFFu)]
    [InlineData(0x40000ul, 0x10000ul, 0b0000_0000_0001_0000u)]
    [InlineData(0xFFFF0ul, 0x20ul, 0b1000_0000_0000_0000u)]
    public void FineBitsCoveredNameTheFinePagesARangeTouches(ulong offset, ulong size, uint expected)
    {
        const ulong page = 7;
        var pageStart = page * OwnerPage;
        Assert.Equal(expected, MetalImagePageOwners.FineBitsCovered(page, pageStart + offset, size));
    }

    [Fact]
    public void FineBitsCoveredClipsToTheOwnerPage()
    {
        const ulong page = 7;
        var pageStart = page * OwnerPage;
        // Starting in the page before: the fine pages from the first one.
        Assert.Equal(0b0000_0000_0000_0011u, MetalImagePageOwners.FineBitsCovered(page, pageStart - 0x1000, 0x1000 + 2 * FinePage));
        // Running into the next page: the fine pages up to the last one.
        Assert.Equal(0b1100_0000_0000_0000u, MetalImagePageOwners.FineBitsCovered(page, pageStart + 14 * FinePage, 3 * FinePage));
        // Wholly outside: none.
        Assert.Equal(0u, MetalImagePageOwners.FineBitsCovered(page, pageStart + OwnerPage, FinePage));
        Assert.Equal(0u, MetalImagePageOwners.FineBitsCovered(page, pageStart - FinePage, FinePage));
    }

    [Fact]
    public void TheFineMaskMarksTheFinePagesThatListAnOwner()
    {
        if (!MetalImagePageOwners.FinePagesEnabled)
        {
            return;
        }

        var owners = new MetalImagePageOwners();
        var address = Base + 20 * OwnerPage;
        var page = address >> MetalImagePageOwners.PageBits;
        var first = new ResourceSlotIdentifier(6, 1);
        var second = new ResourceSlotIdentifier(7, 1);
        var third = new ResourceSlotIdentifier(8, 1);

        // Fine pages 2-3 (one image across both), 3 again, and 15.
        owners.Add(address + 2 * FinePage + 0x8000, FinePage, first);
        owners.Add(address + 3 * FinePage, 0x2000, second);
        owners.Add(address + 15 * FinePage, 0x2000, third);
        Assert.Equal((ushort)0b1000_0000_0000_1100, owners.FineMask(page));

        // A fine page keeps its bit while any owner is left on it.
        owners.Remove(address + 3 * FinePage, 0x2000, second);
        Assert.Equal((ushort)0b1000_0000_0000_1100, owners.FineMask(page));
        owners.Remove(address + 2 * FinePage + 0x8000, FinePage, first);
        Assert.Equal((ushort)0b1000_0000_0000_0000, owners.FineMask(page));
        owners.Remove(address + 15 * FinePage, 0x2000, third);
        Assert.Equal((ushort)0, owners.FineMask(page));
        Assert.False(owners.HasFineOwners(page));
    }

    [Fact]
    public void RemovingASmallImageClearsItsFinePagesAndMovesTheVersion()
    {
        var owners = new MetalImagePageOwners();
        var image = new ResourceSlotIdentifier(4, 1);
        var other = new ResourceSlotIdentifier(5, 1);
        var address = Base + 12 * OwnerPage + FinePage;
        var page = address >> MetalImagePageOwners.PageBits;

        owners.Add(address, 0x4000, image);
        owners.Add(address + 4 * FinePage, 0x4000, other);
        var versionAdded = owners.VersionSum(address, 1);

        owners.Remove(address, 0x4000, image);

        Assert.NotEqual(versionAdded, owners.VersionSum(address, 1));
        Assert.False(owners.MayHaveOwners(address, 0x4000));
        Assert.True(owners.FindFine(address >> MetalImagePageOwners.FinePageBits) is null or { Count: 0 });
        if (MetalImagePageOwners.FinePagesEnabled)
        {
            Assert.True(owners.HasFineOwners(page));
            owners.Remove(address + 4 * FinePage, 0x4000, other);
            Assert.False(owners.HasFineOwners(page));
        }
    }
}
