// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using Xunit;

namespace SharpEmu.Libs.Tests.Memory.GpuMemory;

public sealed class GuestWriteGenerationsTests
{
    [Fact]
    public void VersionsSurviveDirtyConsumptionAndRepeatedWrites()
    {
        using var generations = new GuestWriteGenerations();
        var old = generations.Register(0x1FFF, 2);
        generations.NoteWrite(0x2000, 1);
        Assert.False(old.IsCurrent);
        var next = generations.Register(0x1FFF, 2);
        Assert.True(next.IsCurrent);
        generations.NoteWrite(0x1FFF, 1);
        Assert.False(next.IsCurrent);
        Assert.False(old.IsCurrent);
    }

    [Fact]
    public void DisjointWritesDoNotInvalidateAndUnboundedWritesDo()
    {
        using var generations = new GuestWriteGenerations();
        var token = generations.Register(0x1000, 16);
        generations.NoteWrite(0x2000, 16);
        Assert.True(token.IsCurrent);
        generations.NoteUnboundedWrite();
        Assert.False(token.IsCurrent);
    }

    [Fact]
    public void EnrolledWatchIsSharedAndFaultRecoveryRearmsExactlyOnce()
    {
        using var guard = new PageGuard(new RecordingAddressSpace());
        guard.Permissions.Set(0x1000, 4096, GuestPageProtection.Read | GuestPageProtection.Write);
        using var generations = new GuestWriteGenerations(guard);
        var first = generations.Register(0x1000, 16);
        var second = generations.Register(0x1080, 4);
        Assert.Equal(1, guard.LiveWriteWatches);
        Assert.True(generations.ReleaseFaultWatch(0x1004));
        Assert.False(first.IsCurrent);
        Assert.False(second.IsCurrent);
        Assert.Equal(0, guard.LiveWriteWatches);
        var third = generations.Register(0x1000, 16);
        Assert.True(third.IsCurrent);
        Assert.Equal(1, guard.LiveWriteWatches);
    }

    [Fact]
    public void LandedManagedWriteInvalidatesACaptureBetweenBeforeAndLandedHooks()
    {
        using var manager = new GuestGpuMemory(new RecordingAddressSpace());
        manager.Register(0x1000, 4096, GuestPageProtection.Read | GuestPageProtection.Write);
        manager.MarkCpuWrite(0x1000, 4);
        var token = manager.WriteGenerations.Register(0x1000, 4);
        manager.MarkCpuWriteLanded(0x1000, 4);
        Assert.False(token.IsCurrent);
    }

    [Fact]
    public void NotificationHasNoManagedAllocation()
    {
        using var generations = new GuestWriteGenerations();
        generations.NoteWrite(0x1000, 4);
        var before = GC.GetAllocatedBytesForCurrentThread();
        generations.NoteWrite(0x1000, 4);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}
