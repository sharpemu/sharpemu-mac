// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.Tests.Memory.GuestMemory;
using Xunit;

namespace SharpEmu.Libs.Tests.Kernel;

[Collection(KernelMemoryCompatStateCollection.Name)]
public sealed class KernelMemoryValidationTests
{
    private const int Enomem = unchecked((int)0x8002000C);
    private const int Efault = unchecked((int)0x8002000E);
    private const int Einval = unchecked((int)0x80020016);
    private const ulong GuestMemoryBase = 0x1_0000_0000;

    private static int MapDirect(BackedKernelMemory test, ulong offset, ulong size, ulong address, ulong flags,
        ulong alignment = 0)
    {
        Assert.True(test.Context.TryWriteUInt64(test.Output, address));
        test.Context[CpuRegister.Rdi] = test.Output;
        test.Context[CpuRegister.Rsi] = size;
        test.Context[CpuRegister.Rdx] = 0x33;
        test.Context[CpuRegister.Rcx] = flags;
        test.Context[CpuRegister.R8] = offset;
        test.Context[CpuRegister.R9] = alignment;
        return KernelMemoryCompatExports.KernelMapDirectMemory(test.Context);
    }

    private static int MapFlexible(BackedKernelMemory test, ulong size, ulong address, ulong flags)
    {
        Assert.True(test.Context.TryWriteUInt64(test.Output, address));
        test.Context[CpuRegister.Rdi] = test.Output;
        test.Context[CpuRegister.Rsi] = size;
        test.Context[CpuRegister.Rdx] = 0x3;
        test.Context[CpuRegister.Rcx] = flags;
        return KernelMemoryCompatExports.KernelMapFlexibleMemory(test.Context);
    }

    [Fact]
    public void NoOverwriteMapsIntoAReservationButNotOverAMapping()
    {
        using var test = new BackedKernelMemory();
        test.Allocate(0, 0x4000);
        var address = test.Reserve(0x10000);
        Assert.Equal(0, MapDirect(test, 0, 0x4000, address, 0x90));
        Assert.Equal(Enomem, MapDirect(test, 0, 0x4000, address, 0x90));
        Assert.Equal(0, MapDirect(test, 0, 0x4000, address + 0x4000, 0x90));
    }

    [Theory]
    [InlineData(-1L, 0x10000L, 0x4000UL, 0UL, 0)]
    [InlineData(0x8000L, 0x8000L, 0x4000UL, 0UL, 0)]
    [InlineData(0x8000L, 0x4000L, 0x4000UL, 0UL, 0)]
    [InlineData(0L, 0x10000L, 0UL, 0UL, 0)]
    [InlineData(0L, 0x10000L, 0x1000UL, 0UL, 0)]
    [InlineData(0L, 0x10000L, 0x4000UL, 0x1000UL, 0)]
    [InlineData(0L, 0x10000L, 0x4000UL, 0xC000UL, 0)]
    [InlineData(0L, 0x10000L, 0x4000UL, 0UL, 21)]
    [InlineData(0L, 0x10000L, 0x4000UL, 0UL, -1)]
    public void AllocateDirectMemoryRejectsInvalidArguments(long start, long end, ulong length, ulong alignment, int type)
    {
        var context = new CpuContext(new FakeCpuMemory(GuestMemoryBase, 0x1000), Generation.Gen5);
        context[CpuRegister.Rdi] = unchecked((ulong)start);
        context[CpuRegister.Rsi] = unchecked((ulong)end);
        context[CpuRegister.Rdx] = length;
        context[CpuRegister.Rcx] = alignment;
        context[CpuRegister.R8] = unchecked((ulong)type);
        context[CpuRegister.R9] = GuestMemoryBase + 0x100;
        Assert.Equal(Einval, KernelMemoryCompatExports.KernelAllocateDirectMemory(context));
    }

    [Fact]
    public void AllocateDirectMemoryWithAnUnwritableResultAllocatesNothing()
    {
        var context = new CpuContext(new FakeCpuMemory(GuestMemoryBase, 0x1000), Generation.Gen5);
        context[CpuRegister.Rdi] = 0;
        context[CpuRegister.Rsi] = 0x4000;
        context[CpuRegister.Rdx] = 0x4000;
        context[CpuRegister.Rcx] = 0;
        context[CpuRegister.R8] = 0;
        context[CpuRegister.R9] = GuestMemoryBase + 0x10000;
        Assert.Equal(Efault, KernelMemoryCompatExports.KernelAllocateDirectMemory(context));

        context[CpuRegister.R9] = GuestMemoryBase + 0x100;
        Assert.Equal(0, KernelMemoryCompatExports.KernelAllocateDirectMemory(context));
        Assert.True(context.TryReadUInt64(GuestMemoryBase + 0x100, out var address));
        Assert.Equal(0UL, address);
        context[CpuRegister.Rdi] = 0;
        context[CpuRegister.Rsi] = 0x4000;
        Assert.Equal(0, KernelMemoryCompatExports.KernelCheckedReleaseDirectMemory(context));
    }

    [Fact]
    public void FailedFixedDirectReplacementRestoresTheOldMapping()
    {
        using var test = new BackedKernelMemory();
        test.Allocate(0, 0x8000);
        var address = test.Map(0, 0x4000);
        Assert.True(test.Context.TryWriteUInt64(address, 0x1122334455667788));
        test.Host.FailNext(FailingHostViews.Op.MapView);
        Assert.Equal(Enomem, MapDirect(test, 0x4000, 0x4000, address, 0x10));

        Assert.True(test.Memory.IsBackedView(address));
        Assert.True(test.Context.TryReadUInt64(address, out var value));
        Assert.Equal(0x1122334455667788UL, value);
        var info = KernelGuestVaMapTests.Query(test, address);
        Assert.Equal((address, address + 0x4000, 0UL, (byte)0x12), (info.Start, info.End, info.Offset, info.State));
    }

    [Fact]
    public void FailedFixedFlexibleReplacementRestoresTheOldMapping()
    {
        using var test = new BackedKernelMemory();
        var previous = KernelMemoryCompatExports.SetFlexibleBackingForTests(new FlexibleBackingPool(0x4000000, 0x4000000));
        try
        {
            var initial = AvailableFlexible(test);
            Assert.Equal(0, MapFlexible(test, 0x4000, 0, 0));
            Assert.True(test.Context.TryReadUInt64(test.Output, out var address));
            Assert.True(test.Context.TryWriteUInt64(address, 0xCAFE));
            test.Host.FailNext(FailingHostViews.Op.MapView);
            Assert.Equal(Enomem, MapFlexible(test, 0x4000, address, 0x10));

            Assert.True(test.Context.TryReadUInt64(address, out var value));
            Assert.Equal(0xCAFEUL, value);
            Assert.Equal(0x11, KernelGuestVaMapTests.Query(test, address).State);
            // The restored mapping still owns its backing: unmapping it returns the bytes.
            Assert.Equal(0, test.Unmap(address, 0x4000));
            Assert.Equal(initial, AvailableFlexible(test));
        }
        finally
        {
            KernelMemoryCompatExports.ResetBackingMappings(test.Memory);
            KernelMemoryCompatExports.SetFlexibleBackingForTests(previous);
        }
    }

    private static ulong AvailableFlexible(BackedKernelMemory test)
    {
        test.Context[CpuRegister.Rdi] = test.Output + 0x200;
        Assert.Equal(0, KernelMemoryCompatExports.KernelAvailableFlexibleMemorySize(test.Context));
        Assert.True(test.Context.TryReadUInt64(test.Output + 0x200, out var available));
        return available;
    }

    [Fact]
    public void MapDirectHonoursAlignedFlagsAndRejectsBadAlignment()
    {
        using var test = new BackedKernelMemory();
        test.Allocate(0, 0x4000);
        Assert.Equal(0, MapDirect(test, 0, 0x4000, 0, 20UL << 24));
        Assert.True(test.Context.TryReadUInt64(test.Output, out var address));
        Assert.Equal(0UL, address % 0x100000);

        Assert.Equal(Einval, MapDirect(test, 0, 0x4000, 0, 0, alignment: 0xC000));
        Assert.Equal(Einval, MapDirect(test, 0, 0x4000, 0, 0x8));
        Assert.Equal(Einval, MapDirect(test, 0, 0x4000, 0, 13UL << 24));
    }

    [Fact]
    public void ProtectionBitsOutsideTheSdkSetAreRejected()
    {
        using var test = new BackedKernelMemory();
        test.Allocate(0, 0x4000);
        var address = test.Map(0, 0x4000);
        test.Context[CpuRegister.Rdi] = address;
        test.Context[CpuRegister.Rsi] = 0x4000;
        test.Context[CpuRegister.Rdx] = 0x433;
        Assert.Equal(Einval, KernelMemoryCompatExports.KernelMprotect(test.Context));
        test.Context[CpuRegister.Rdx] = 0x1B3;
        Assert.Equal(0, KernelMemoryCompatExports.KernelMprotect(test.Context));
    }

    [Fact]
    public void BatchMapRejectsANegativeCount()
    {
        using var test = new BackedKernelMemory();
        test.Context[CpuRegister.Rdi] = test.Output + 0x200;
        test.Context[CpuRegister.Rsi] = unchecked((ulong)-1);
        test.Context[CpuRegister.Rdx] = test.Output + 0x300;
        Assert.Equal(Einval, KernelMemoryCompatExports.KernelBatchMap(test.Context));
        Assert.True(test.Context.TryReadUInt32(test.Output + 0x300, out var processed));
        Assert.Equal(0u, processed);
    }

    [Fact]
    public void UnmappedReservationCanBeMappedAgainAtTheSameAddress()
    {
        using var test = new BackedKernelMemory();
        test.Allocate(0, 0x4000);
        var address = test.Reserve(0x10000);
        Assert.Equal(0, test.Unmap(address, 0x10000));
        Assert.Equal(address, test.Map(0, 0x4000, address));
        Assert.Equal(0, test.Unmap(address, 0x4000));
        Assert.Equal(address, test.Reserve(0x10000, address));
    }

    [Fact]
    public void FixedRequestsInTheBlockedHostWindowFailBeforeTouchingStateWithRelocationOff()
    {
        using var test = new BackedKernelMemory();
        GuestVaHole.BlocksFixedOverride = true;
        GuestVaHole.RelocatesOverride = false;
        try
        {
            test.Allocate(0, 0x4000);
            var before = test.Host.Log.Count;
            Assert.Equal(Enomem, MapDirect(test, 0, 0x4000, 0x20_0000_0000, 0x10));
            Assert.True(test.Context.TryWriteUInt64(test.Output, 0x6F_FFFF_C000));
            test.Context[CpuRegister.Rdi] = test.Output;
            test.Context[CpuRegister.Rsi] = 0x8000;
            test.Context[CpuRegister.Rdx] = 0x10;
            test.Context[CpuRegister.Rcx] = 0;
            Assert.Equal(Enomem, KernelRuntimeCompatExports.KernelReserveVirtualRange(test.Context));
            Assert.Equal(before, test.Host.Log.Count);
        }
        finally
        {
            GuestVaHole.BlocksFixedOverride = null;
            GuestVaHole.RelocatesOverride = null;
        }
    }

    private const ulong WindowAddress = 0x30_0000_0000;
    private const ulong RelocatedAddress = WindowAddress + 0x70_0000_0000;

    [Fact]
    public void AFixedMapInTheBlockedWindowIsRelocatedAndReportsTheMappedAddress()
    {
        using var test = new BackedKernelMemory();
        GuestVaHole.BlocksFixedOverride = true;
        GuestVaHole.RelocatesOverride = true;
        try
        {
            test.Allocate(0, 0x4000);
            Assert.Equal(0, MapDirect(test, 0, 0x4000, WindowAddress, 0x10));
            Assert.True(test.Context.TryReadUInt64(test.Output, out var address));
            Assert.Equal(RelocatedAddress, address);
            Assert.True(test.Memory.IsBackedRange(RelocatedAddress, 0x4000));
        }
        finally
        {
            GuestVaHole.BlocksFixedOverride = null;
            GuestVaHole.RelocatesOverride = null;
        }
    }

    [Fact]
    public void UnmapProtectAndQueryAtTheWindowAddressActOnTheRelocatedMapping()
    {
        using var test = new BackedKernelMemory();
        GuestVaHole.BlocksFixedOverride = true;
        GuestVaHole.RelocatesOverride = true;
        try
        {
            test.Allocate(0, 0x8000);
            Assert.Equal(0, MapDirect(test, 0, 0x8000, WindowAddress, 0x10));

            Assert.Equal((RelocatedAddress, RelocatedAddress + 0x8000), test.Query(WindowAddress));
            Assert.Equal((RelocatedAddress, RelocatedAddress + 0x8000), test.Query(WindowAddress + 0x4000));

            test.Context[CpuRegister.Rdi] = WindowAddress;
            test.Context[CpuRegister.Rsi] = 0x4000;
            test.Context[CpuRegister.Rdx] = 0x1;
            Assert.Equal(0, KernelMemoryCompatExports.KernelMprotect(test.Context));
            Assert.Equal((RelocatedAddress, RelocatedAddress + 0x4000), test.Query(RelocatedAddress));

            Assert.Equal(0, test.Unmap(WindowAddress, 0x8000));
            Assert.False(test.Memory.IsBackedRange(RelocatedAddress, 0x4000));
        }
        finally
        {
            GuestVaHole.BlocksFixedOverride = null;
            GuestVaHole.RelocatesOverride = null;
        }
    }

    [Fact]
    public void ARangeStraddlingTheWindowEdgeIsStillRefusedWhenRelocating()
    {
        using var test = new BackedKernelMemory();
        GuestVaHole.BlocksFixedOverride = true;
        GuestVaHole.RelocatesOverride = true;
        try
        {
            test.Allocate(0, 0x8000);
            Assert.Equal(Enomem, MapDirect(test, 0, 0x8000, 0x6F_FFFF_C000, 0x10));
            Assert.False(GuestVaHole.TryTranslateRange(0x6F_FFFF_C000, 0x8000, out _));
            Assert.True(GuestVaHole.TryTranslateRange(0x6F_FFFF_C000, 0x4000, out var translated));
            Assert.Equal(0xDF_FFFF_C000UL, translated);
        }
        finally
        {
            GuestVaHole.BlocksFixedOverride = null;
            GuestVaHole.RelocatesOverride = null;
        }
    }

    [Fact]
    public void KernelChosenSearchesStayOutOfTheShadowOfTheWindowWhileThereIsRoomElsewhere()
    {
        using var test = new BackedKernelMemory();
        GuestVaHole.SkipsSearchOverride = true;
        GuestVaHole.BlocksFixedOverride = true;
        GuestVaHole.RelocatesOverride = true;
        try
        {
            // A hint ending inside the shadow [0x80_0000_0000, 0xE0_0000_0000) moves above it.
            Assert.True(test.Context.TryWriteUInt64(test.Output, 0x7F_FFFF_C000));
            test.Context[CpuRegister.Rdi] = test.Output;
            test.Context[CpuRegister.Rsi] = 0x8000;
            test.Context[CpuRegister.Rdx] = 0;
            test.Context[CpuRegister.Rcx] = 0;
            Assert.Equal(0, KernelRuntimeCompatExports.KernelReserveVirtualRange(test.Context));
            Assert.True(test.Context.TryReadUInt64(test.Output, out var address));
            Assert.False(GuestVaHole.OverlapsShadow(address, 0x8000));
            Assert.Equal(0xE0_0000_0000UL, address);
            Assert.Equal(0xE0_0000_0000UL, GuestVaHole.SkipForSearch(0x7F_FFFF_C000, 0x8000));
            // With no other room the search may use the shadow after all.
            Assert.Equal(0x7F_FFFF_C000UL, GuestVaHole.SkipForSearch(0x7F_FFFF_C000, 0x8000, avoidShadow: false));
        }
        finally
        {
            GuestVaHole.SkipsSearchOverride = null;
            GuestVaHole.BlocksFixedOverride = null;
            GuestVaHole.RelocatesOverride = null;
        }
    }

    [Fact]
    public void KernelChosenPlacementsSkipTheBlockedHostWindow()
    {
        using var test = new BackedKernelMemory();
        GuestVaHole.SkipsSearchOverride = true;
        try
        {
            test.Allocate(0, 0x4000);
            // A hint inside the window is only a hint: the placement moves above it.
            Assert.Equal(0, MapDirect(test, 0, 0x4000, 0x20_0000_0000, 0));
            Assert.True(test.Context.TryReadUInt64(test.Output, out var address));
            Assert.False(GuestVaHole.Overlaps(address, 0x4000));
        }
        finally
        {
            GuestVaHole.SkipsSearchOverride = null;
        }
    }

    [Theory]
    [InlineData(0x0F_FFFF_C000UL, 0x4000UL, false)]
    [InlineData(0x0F_FFFF_C000UL, 0x8000UL, true)]
    [InlineData(0x10_0000_0000UL, 0x4000UL, true)]
    [InlineData(0x6F_FFFF_C000UL, 0x4000UL, true)]
    [InlineData(0x70_0000_0000UL, 0x4000UL, false)]
    public void TheBlockedWindowCoversTheFirstPartOfTheAppArea(ulong address, ulong length, bool overlaps)
    {
        Assert.Equal(overlaps, GuestVaHole.Overlaps(address, length));
        Assert.Equal(overlaps ? GuestVaHole.End : address,
            GuestVaHole.SkipsSearch ? GuestVaHole.SkipForSearch(address, length) : (overlaps ? GuestVaHole.End : address));
    }
}
