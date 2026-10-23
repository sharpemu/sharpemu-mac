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
    [Fact]
    public void BulkWriteInvalidatesOnlyWatchedPagesInsideItsExactPageFootprint()
    {
        using var generations = new GuestWriteGenerations();
        var first = generations.Register(0x1FF000, 1);
        var middle = generations.Register(0x400000, 1);
        var last = generations.Register(0x600000, 1);
        var outside = generations.Register(0x601000, 1);
        generations.NoteWrite(0x1FFFFF, 0x400002);
        Assert.False(first.IsCurrent);
        Assert.False(middle.IsCurrent);
        Assert.False(last.IsCurrent);
        Assert.True(outside.IsCurrent);
        var recaptured = generations.Register(0x400000, 1);
        generations.NoteWrite(0x400000, 8 * 1024 * 1024);
        Assert.False(recaptured.IsCurrent);
    }

    [Fact]
    public void UnwatchedBulkWritesDoNotAllocateOrInvalidateDisjointTokens()
    {
        using var generations = new GuestWriteGenerations();
        var token = generations.Register(0x1000, 4);
        generations.NoteWrite(0x2000, 16 * 1024 * 1024);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 1000; i++) generations.NoteWrite(0x2000, 16 * 1024 * 1024);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.True(token.IsCurrent);
        var later = generations.Register(0x400000, 4);
        Assert.True(later.IsCurrent);
        generations.NoteWrite(0x400000, 4);
        Assert.False(later.IsCurrent);
    }

    [Theory]
    [InlineData(0UL, 268435456UL)]
    [InlineData(ulong.MaxValue, 2UL)]
    public void HugeOrInvalidRangesConservativelyVetoTokens(ulong address, ulong size)
    {
        using var generations = new GuestWriteGenerations();
        var token = generations.Register(0x1000, 4);
        generations.NoteWrite(address, size);
        Assert.False(token.IsCurrent);
    }

    [Fact]
    public void RemapInvalidatesBothPreviouslyWatchedAndNewlyEnrolledPages()
    {
        using var generations = new GuestWriteGenerations();
        var token = generations.Register(0x1000, 4);
        generations.NoteRemap(0x1000, 4096);
        Assert.False(token.IsCurrent);
        var next = generations.Register(0x1000, 4);
        Assert.True(next.IsCurrent);
        generations.NoteWrite(0x1000, 4096);
        Assert.False(next.IsCurrent);
    }
    [Fact]
    public void FarAwayUnwatchedPagesDoNotCollideWithRegisteredPages()
    {
        using var generations = new GuestWriteGenerations();
        var token = generations.Register(0x1000, 4);
        generations.NoteWrite(0x10001000, 4);
        Assert.True(token.IsCurrent);
    }

    [Fact]
    public void StoreNotificationDoesNotWaitForNativeEnrollmentGate()
    {
        using var guard = new PageGuard(new RecordingAddressSpace());
        guard.Permissions.Set(0x1000, 4096, GuestPageProtection.Read | GuestPageProtection.Write);
        using var generations = new GuestWriteGenerations(guard);
        var token = generations.Register(0x1000, 4);
        var pages = (Array)typeof(GuestWriteGenerations.Token).GetField("_pages",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(token)!;
        var watch = pages.GetValue(0)!;
        var gate = watch.GetType().GetField("Gate")!;
        gate.SetValue(watch, 1);
        try
        {
            var store = Task.Run(() => generations.NoteWrite(0x1000, 16 * 1024 * 1024));
            Assert.True(store.Wait(TimeSpan.FromSeconds(10)));
            Assert.False(token.IsCurrent);
        }
        finally { gate.SetValue(watch, 0); }
    }
}
