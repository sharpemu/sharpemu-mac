// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Kernel;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class KernelEventQueueSignalSkipCollection
{
    public const string Name = "KernelEventQueueSignalSkip";
}

// Signal skips the scheduler scan while no guest waiter is registered on the queue. These tests pin the
// waiter count and that skipping cannot lose a wake against a waiter that blocks concurrently.
[Collection(KernelEventQueueSignalSkipCollection.Name)]
public sealed class KernelEventQueueSignalSkipTests
{
    private const ulong BaseAddress = 0x1_0000_0000;
    private const ulong HandleAddress = BaseAddress + 0x100;
    private const ulong EventsAddress = BaseAddress + 0x200;
    private const ulong OutCountAddress = BaseAddress + 0x300;

    private static KernelEventQueueCompatExports.KernelQueuedEvent UserEvent(ulong ident) =>
        new(
            Ident: ident,
            Filter: KernelEventQueueCompatExports.KernelEventFilterUser,
            Flags: KernelEventQueueCompatExports.KernelEventFlagClear,
            Fflags: 1,
            Data: 0x1234,
            UserData: 0x5678);

    [Fact]
    public void SignalWithoutWaiter_SkipsTheSchedulerScan()
    {
        var (scheduler, parking) = CreateScheduler();
        var previous = GuestThreadExecution.Scheduler;
        GuestThreadExecution.Scheduler = scheduler;
        try
        {
            var (_, ctx, handle) = CreateEqueue();
            Assert.Equal(0, KernelEventQueueCompatExports.BlockedWaiterCountForTest(handle));

            Assert.True(KernelEventQueueCompatExports.EnqueueEvent(handle, UserEvent(1)));
            Assert.Equal(0, parking.WakeCalls);

            DeleteEqueue(ctx, handle);
        }
        finally
        {
            GuestThreadExecution.Scheduler = previous;
        }
    }

    [Fact]
    public void SignalWithWaiter_ScansAndTheCountReturnsToZeroOnResume()
    {
        var (scheduler, parking) = CreateScheduler();
        var previous = GuestThreadExecution.Scheduler;
        GuestThreadExecution.Scheduler = scheduler;
        try
        {
            var (_, ctx, handle) = CreateEqueue();
            var waiter = StageGuestWait(handle, threadHandle: 0x801, out _);
            Assert.NotNull(waiter);
            Assert.Equal(1, KernelEventQueueCompatExports.BlockedWaiterCountForTest(handle));

            Assert.True(KernelEventQueueCompatExports.EnqueueEvent(handle, UserEvent(2)));
            Assert.True(parking.WakeCalls >= 1);

            Assert.True(waiter!.TryWake());
            Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_OK, waiter.Resume());
            Assert.Equal(0, KernelEventQueueCompatExports.BlockedWaiterCountForTest(handle));

            DeleteEqueue(ctx, handle);
        }
        finally
        {
            GuestThreadExecution.Scheduler = previous;
        }
    }

    [Fact]
    public void DeleteWithWaiter_StillWakesIt()
    {
        var (scheduler, parking) = CreateScheduler();
        var previous = GuestThreadExecution.Scheduler;
        GuestThreadExecution.Scheduler = scheduler;
        try
        {
            var (_, ctx, handle) = CreateEqueue();
            var waiter = StageGuestWait(handle, threadHandle: 0x802, out _);
            Assert.NotNull(waiter);

            DeleteEqueue(ctx, handle);
            Assert.True(parking.WakeCalls >= 1);
            Assert.True(waiter!.TryWake());
            Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_DELETED, waiter.Resume());
        }
        finally
        {
            GuestThreadExecution.Scheduler = previous;
        }
    }

    // Models the scheduler's contract: a blocking waiter is parked under one gate and re-checked with
    // TryWake after parking; a wake scans the parked waiters under the same gate. The signaller enqueues
    // while the waiter is blocking, with varying offsets. A lost wake leaves the waiter parked forever.
    [Fact]
    public void ConcurrentSignalAndWait_NeverLosesAWake()
    {
        const int Iterations = 3000;
        var (scheduler, parking) = CreateScheduler();
        var previous = GuestThreadExecution.Scheduler;
        GuestThreadExecution.Scheduler = scheduler;
        try
        {
            var (_, ctx, firstHandle) = CreateEqueue();
            var handles = new ulong[Iterations];
            handles[0] = firstHandle;
            for (var i = 1; i < Iterations; i++)
            {
                handles[i] = CreateEqueue().Handle;
            }

            using var barrier = new Barrier(2);
            using var stop = new CancellationTokenSource();
            Exception? signalerFailure = null;
            var signaler = new Thread(() =>
            {
                try
                {
                    for (var i = 0; i < Iterations; i++)
                    {
                        barrier.SignalAndWait(stop.Token);
                        Spin(i * 5 % 97);
                        Assert.True(KernelEventQueueCompatExports.EnqueueEvent(handles[i], UserEvent((ulong)i)));
                    }
                }
                catch (OperationCanceledException) when (stop.IsCancellationRequested)
                {
                }
                catch (Exception ex)
                {
                    signalerFailure = ex;
                    stop.Cancel();
                }
            })
            { IsBackground = true };
            signaler.Start();

            var lost = 0;
            var delivered = 0;
            try
            {
                for (var i = 0; i < Iterations; i++)
                {
                    barrier.SignalAndWait(stop.Token);
                    Spin(i * 3 % 89);
                    var waiter = StageGuestWait(handles[i], threadHandle: 0x900, out var immediate);
                    if (waiter is null)
                    {
                        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_OK, immediate);
                        delivered++;
                        continue;
                    }

                    if (!parking.Park(waiter, handles[i], TimeSpan.FromSeconds(3)))
                    {
                        lost++;
                        break;
                    }

                    delivered++;
                    Assert.Equal(0, KernelEventQueueCompatExports.BlockedWaiterCountForTest(handles[i]));
                }
            }
            finally
            {
                // A lost wake must fail the test, not leave the signaller at the next barrier forever.
                stop.Cancel();
                Assert.True(signaler.Join(TimeSpan.FromSeconds(5)), "The signal worker did not stop.");
                foreach (var handle in handles)
                    DeleteEqueue(ctx, handle);
            }

            Assert.Null(signalerFailure);
            Assert.Equal(0, lost);
            Assert.Equal(Iterations, delivered);
        }
        finally
        {
            GuestThreadExecution.Scheduler = previous;
        }
    }

    private static void Spin(int iterations)
    {
        for (var i = 0; i < iterations; i++)
        {
            Thread.SpinWait(20);
        }
    }

    private static (IGuestThreadScheduler Scheduler, ParkingScheduler Parking) CreateScheduler()
    {
        var proxy = DispatchProxy.Create<IGuestThreadScheduler, ParkingScheduler>();
        return (proxy, (ParkingScheduler)(object)proxy);
    }

    private static (FakeCpuMemory Memory, CpuContext Context, ulong Handle) CreateEqueue()
    {
        var memory = new FakeCpuMemory(BaseAddress, 0x1000);
        var ctx = new CpuContext(memory, Generation.Gen5);
        ctx[CpuRegister.Rdi] = HandleAddress;
        Assert.Equal(
            (int)OrbisGen2Result.ORBIS_GEN2_OK,
            KernelEventQueueCompatExports.KernelCreateEqueue(ctx));
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        Assert.True(memory.TryRead(HandleAddress, bytes));
        return (memory, ctx, System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(bytes));
    }

    private static void DeleteEqueue(CpuContext ctx, ulong handle)
    {
        ctx[CpuRegister.Rdi] = handle;
        Assert.Equal(
            (int)OrbisGen2Result.ORBIS_GEN2_OK,
            KernelEventQueueCompatExports.KernelDeleteEqueue(ctx));
    }

    // Returns the blocking waiter, or null when the wait completed without blocking (result in immediate).
    private static IGuestThreadBlockWaiter? StageGuestWait(ulong handle, ulong threadHandle, out int immediate)
    {
        var memory = new FakeCpuMemory(BaseAddress, 0x1000);
        var ctx = new CpuContext(memory, Generation.Gen5);
        var previousThread = GuestThreadExecution.EnterGuestThread(threadHandle);
        var previousFrame = GuestThreadExecution.EnterImportCallFrame(
            returnRip: 0x1_0000 + threadHandle,
            resumeRsp: 0x2_0000 + threadHandle,
            returnSlotAddress: 0x3_0000 + threadHandle);
        try
        {
            ctx[CpuRegister.Rdi] = handle;
            ctx[CpuRegister.Rsi] = EventsAddress;
            ctx[CpuRegister.Rdx] = 1;
            ctx[CpuRegister.Rcx] = OutCountAddress;
            ctx[CpuRegister.R8] = 0;
            immediate = KernelEventQueueCompatExports.KernelWaitEqueue(ctx);
            if (!GuestThreadExecution.TryConsumeCurrentThreadBlock(
                    out _,
                    out _,
                    out var hasContinuation,
                    out _,
                    out var waiter,
                    out _))
            {
                return null;
            }

            Assert.True(hasContinuation);
            return Assert.IsAssignableFrom<IGuestThreadBlockWaiter>(waiter);
        }
        finally
        {
            GuestThreadExecution.RestoreImportCallFrame(previousFrame);
            GuestThreadExecution.RestoreGuestThread(previousThread);
        }
    }

    public class ParkingScheduler : DispatchProxy
    {
        private readonly object _gate = new();
        private readonly Dictionary<ulong, (IGuestThreadBlockWaiter Waiter, ManualResetEventSlim Woken)> _parked = new();
        private int _wakeCalls;

        public int WakeCalls => Volatile.Read(ref _wakeCalls);

        // Parks the waiter the way the scheduler does: register, then re-check TryWake under the gate.
        public bool Park(IGuestThreadBlockWaiter waiter, ulong handle, TimeSpan timeout)
        {
            using var woken = new ManualResetEventSlim(false);
            lock (_gate)
            {
                if (waiter.TryWake())
                {
                    woken.Set();
                }
                else
                {
                    _parked[handle] = (waiter, woken);
                }
            }

            if (!woken.Wait(timeout))
            {
                // The signaller must not retain an event that this method is about to dispose.
                lock (_gate)
                    _parked.Remove(handle);
                return false;
            }

            Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_OK, waiter.Resume());
            return true;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod?.Name == nameof(IGuestThreadScheduler.WakeBlockedThreads))
            {
                Interlocked.Increment(ref _wakeCalls);
                var woke = 0;
                lock (_gate)
                {
                    foreach (var (handle, entry) in _parked.ToArray())
                    {
                        if (entry.Waiter.TryWake())
                        {
                            _parked.Remove(handle);
                            entry.Woken.Set();
                            woke++;
                        }
                    }
                }

                return woke;
            }

            var returnType = targetMethod?.ReturnType;
            return returnType is null || returnType == typeof(void) || !returnType.IsValueType
                ? null
                : Activator.CreateInstance(returnType);
        }
    }
}
