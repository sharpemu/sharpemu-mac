// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Loader;
using SharpEmu.Core.Memory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

/// <summary>
/// A module loaded after startup goes to the first free 2 MB step of the PS5 module range
/// from its preferred base. The host can hold more than half a gigabyte right after the
/// guest's last module; the search must reach past it (or into the gaps below) rather than
/// leave the module to the host, which maps it below the range the return-address checks
/// treat as code (Black Myth: Wukong's Wwise plugins at 0x230C43000).
/// </summary>
public sealed class ModulePlacementTests
{
    private const ulong ModuleRangeStart = 0x0000_0008_0400_0000UL;
    private const ulong ModuleRangeEnd = 0x0000_0009_0000_0000UL;
    private const ulong ModuleSize = 0xC950;

    [Fact]
    public void AModuleIsPlacedPastAHostMappingLongerThanHalfAGigabyte()
    {
        const ulong preferred = 0x0000_0008_412B_2000UL;
        var occupiedEnd = preferred + 0x2600_0000UL;
        using var memory = new PhysicalVirtualMemory(new BandedHostMemory((preferred, occupiedEnd)));

        Assert.True(SelfLoader.TryAllocateAdditionalImageAtExact(memory, preferred, ModuleSize, isNextGen: true, out var placed));

        Assert.True(placed >= occupiedEnd, $"placed at 0x{placed:X}");
        Assert.True(placed + ModuleSize <= ModuleRangeEnd, $"placed at 0x{placed:X}");
    }

    [Fact]
    public void AModuleTakesAGapBelowItsPreferredBaseWhenNothingAboveIsFree()
    {
        const ulong preferred = 0x0000_0008_8000_0000UL;
        const ulong gap = 0x0000_0008_1000_0000UL;
        using var memory = new PhysicalVirtualMemory(new BandedHostMemory(
            (ModuleRangeStart, gap),
            (gap + 0x0020_0000UL, ModuleRangeEnd)));

        Assert.True(SelfLoader.TryAllocateAdditionalImageAtExact(memory, preferred, ModuleSize, isNextGen: true, out var placed));

        Assert.Equal(gap, placed);
    }

    [Fact]
    public void AFullModuleRangeIsReportedAsAFailure()
    {
        using var memory = new PhysicalVirtualMemory(new BandedHostMemory((ModuleRangeStart, ModuleRangeEnd)));

        Assert.False(SelfLoader.TryAllocateAdditionalImageAtExact(
            memory, 0x0000_0008_412B_2000UL, ModuleSize, isNextGen: true, out _));
    }

    /// <summary>A host that maps fixed requests anywhere but the given bands; a request in a
    /// band comes back somewhere else, as mmap without MAP_FIXED does.</summary>
    private sealed class BandedHostMemory(params (ulong Start, ulong End)[] occupied) : IHostMemory
    {
        private const ulong Elsewhere = 0x0000_0002_30C4_3000UL;

        public ulong Allocate(ulong desiredAddress, ulong size, HostPageProtection protection)
        {
            if (desiredAddress == 0)
            {
                return Elsewhere;
            }

            foreach (var (start, end) in occupied)
            {
                if (desiredAddress < end && desiredAddress + size > start)
                {
                    return Elsewhere;
                }
            }

            return desiredAddress;
        }

        public ulong Reserve(ulong desiredAddress, ulong size, HostPageProtection protection) =>
            Allocate(desiredAddress, size, protection);

        public bool Commit(ulong address, ulong size, HostPageProtection protection) => true;

        public bool Free(ulong address) => true;

        public bool Protect(ulong address, ulong size, HostPageProtection protection, out uint rawOldProtection)
        {
            rawOldProtection = 0;
            return true;
        }

        public bool ProtectRaw(ulong address, ulong size, uint rawProtection, out uint rawOldProtection)
        {
            rawOldProtection = 0;
            return true;
        }

        public bool Query(ulong address, out HostRegionInfo info)
        {
            info = default;
            return false;
        }

        public void FlushInstructionCache(ulong address, ulong size)
        {
        }
    }
}
