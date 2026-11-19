// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.CxxAbi;
using Xunit;

namespace SharpEmu.Libs.Tests;

// __cxa_guard_acquire hands the construction of a function-local static to one caller only.
public sealed class CxaGuardExportsTests
{
    private const ulong Base = 0x1000;

    // Runs an action once, right after a chosen read of a chosen address has returned its bytes.
    private sealed class ReadHookMemory(ICpuMemory inner, ulong address, int onRead, Action hook) : ICpuMemory
    {
        private int _reads;

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            var read = inner.TryRead(virtualAddress, destination);
            if (virtualAddress == address && ++_reads == onRead)
            {
                hook();
            }

            return read;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => inner.TryWrite(virtualAddress, source);
    }

    private static int Acquire(CpuContext context, ulong guard)
    {
        context[CpuRegister.Rdi] = guard;
        Assert.Equal(0, CxaGuardExports.CxaGuardAcquire(context));
        return (int)context[CpuRegister.Rax];
    }

    private static void Release(CpuContext context, ulong guard)
    {
        context[CpuRegister.Rdi] = guard;
        Assert.Equal(0, CxaGuardExports.CxaGuardRelease(context));
    }

    [Fact]
    public void TheFirstCallerConstructsAndLaterCallersDoNot()
    {
        var guard = Base + 0x100;
        var context = new CpuContext(new FakeCpuMemory(Base, 0x1000), Generation.Gen5);
        Assert.Equal(1, Acquire(context, guard));
        Release(context, guard);
        Assert.True(context.TryReadUInt64(guard, out var word));
        Assert.Equal(1ul, word & 0xFF);
        Assert.Equal(0, Acquire(context, guard));
    }

    // The owner releases after a second thread has read the guard as pending and before that thread claims it: the
    // second thread must find the guard complete, not construct the object again.
    [Fact]
    public void AReleaseBetweenAWaitersReadAndItsClaimDoesNotHandItTheConstruction()
    {
        var guard = Base + 0x200;
        var memory = new FakeCpuMemory(Base, 0x1000);
        var owner = new CpuContext(memory, Generation.Gen5);
        var ownerDone = new ManualResetEventSlim();
        var ownerAcquired = new ManualResetEventSlim();
        var release = new ManualResetEventSlim();
        var ownerThread = new Thread(() =>
        {
            Assert.Equal(1, Acquire(owner, guard));
            ownerAcquired.Set();
            release.Wait();
            Release(owner, guard);
            ownerDone.Set();
        });
        ownerThread.Start();
        Assert.True(ownerAcquired.Wait(TimeSpan.FromSeconds(10)));

        // The waiter's first read of the guard sees "pending"; the owner then releases before the waiter goes on.
        var hooked = new ReadHookMemory(memory, guard, onRead: 1, () =>
        {
            release.Set();
            Assert.True(ownerDone.Wait(TimeSpan.FromSeconds(10)));
        });
        var waiter = new CpuContext(hooked, Generation.Gen5);
        Assert.Equal(0, Acquire(waiter, guard));
        ownerThread.Join();
        var first = new byte[1];
        Assert.True(memory.TryRead(guard, first));
        Assert.Equal(1, first[0]);
    }

    // After an abort the guard is free again and the next caller constructs.
    [Fact]
    public void AnAbortedConstructionIsHandedToTheNextCaller()
    {
        var guard = Base + 0x300;
        var context = new CpuContext(new FakeCpuMemory(Base, 0x1000), Generation.Gen5);
        Assert.Equal(1, Acquire(context, guard));
        context[CpuRegister.Rdi] = guard;
        Assert.Equal(0, CxaGuardExports.CxaGuardAbort(context));
        Assert.Equal(1, Acquire(context, guard));
        Release(context, guard);
        Assert.Equal(0, Acquire(context, guard));
    }
}
