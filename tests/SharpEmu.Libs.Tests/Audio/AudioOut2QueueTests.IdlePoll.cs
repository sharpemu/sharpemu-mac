// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Text.Json;
using SharpEmu.HLE;
using SharpEmu.Libs.Audio;
using SharpEmu.Libs.Kernel;
using SharpEmu.PerfAudit;
using Xunit;

namespace SharpEmu.Libs.Tests.Audio;

public sealed partial class AudioOut2QueueTests
{
    private const ulong PollFsBase = MemoryBase + 0x9000;
    private const ulong PollEventHandleSlot = MemoryBase + 0x700;
    private const ulong PollEventName = MemoryBase + 0x720;
    private const ulong PollTimeoutSlot = MemoryBase + 0x740;
    private const ulong PollResultSlot = MemoryBase + 0x750;
    private const ulong PollLevelAddress = MemoryBase + 0x760;
    private const ulong PollAvailableAddress = MemoryBase + 0x764;

    private static readonly bool IdleWaitDisabled =
        Environment.GetEnvironmentVariable("SHARPEMU_AUDIO_OUT2_IDLE_WAIT") == "0";

    private readonly record struct PollRun(
        long Iterations, long Pushes, long EmptyPolls, uint MaxLevel, long WallMilliseconds, long ThreadCpuNanoseconds);

    // Wwise's audio thread: read the queue level, render and push as many grains as are available
    // (Advance, then a non-blocking Push), then wait on an event flag with a zero timeout.
    private PollRun RunWwiseAudioThread(uint queueDepth, uint grainSamples, int milliseconds)
    {
        var context = CreateContext(queueDepth, grainSamples);

        var setup = new CpuContext(_memory, Generation.Gen5) { FsBase = PollFsBase };
        Assert.True(_memory.TryWrite(PollEventName, "idle-poll\0"u8));
        setup[CpuRegister.Rdi] = PollEventHandleSlot;
        setup[CpuRegister.Rsi] = PollEventName;
        setup[CpuRegister.Rdx] = 0x11;
        setup[CpuRegister.Rcx] = 0;
        setup[CpuRegister.R8] = 0;
        Assert.Equal(0, KernelEventFlagCompatExports.KernelCreateEventFlag(setup));
        Assert.True(setup.TryReadUInt64(PollEventHandleSlot, out var flag));

        PollRun run = default;
        var thread = new Thread(() =>
        {
            var ctx = new CpuContext(_memory, Generation.Gen5) { FsBase = PollFsBase };
            long iterations = 0, pushes = 0, emptyPolls = 0;
            uint maxLevel = 0;
            var cpuStart = OperatingSystem.IsMacOS() ? DarwinThreadCpuClock.Nanoseconds() : 0;
            var stopwatch = Stopwatch.StartNew();
            while (stopwatch.ElapsedMilliseconds < milliseconds)
            {
                iterations++;
                ctx[CpuRegister.Rdi] = context;
                ctx[CpuRegister.Rsi] = PollLevelAddress;
                ctx[CpuRegister.Rdx] = PollAvailableAddress;
                Assert.Equal(0, AudioOut2Exports.AudioOut2ContextGetQueueLevel(ctx));
                var level = ReadUInt32(PollLevelAddress);
                var available = ReadUInt32(PollAvailableAddress);
                maxLevel = Math.Max(maxLevel, level);
                if (level == 0 && pushes >= queueDepth)
                {
                    emptyPolls++;
                }

                for (var i = 0u; i < available; i++)
                {
                    ctx[CpuRegister.Rdi] = context;
                    Assert.Equal(0, AudioOut2Exports.AudioOut2ContextAdvance(ctx));
                    ctx[CpuRegister.Rdi] = context;
                    ctx[CpuRegister.Rsi] = NonBlocking;
                    Assert.Equal(0, AudioOut2Exports.AudioOut2ContextPush(ctx));
                    pushes++;
                }

                Assert.True(ctx.TryWriteUInt32(PollTimeoutSlot, 0));
                ctx[CpuRegister.Rdi] = flag;
                ctx[CpuRegister.Rsi] = 1;
                ctx[CpuRegister.Rdx] = 0x12;
                ctx[CpuRegister.Rcx] = PollResultSlot;
                ctx[CpuRegister.R8] = PollTimeoutSlot;
                // Nothing sets the flag: every wait ends by timing out.
                Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT, KernelEventFlagCompatExports.KernelWaitEventFlag(ctx));
            }

            var cpuEnd = OperatingSystem.IsMacOS() ? DarwinThreadCpuClock.Nanoseconds() : 0;
            run = new PollRun(iterations, pushes, emptyPolls, maxLevel, stopwatch.ElapsedMilliseconds, (long)(cpuEnd - cpuStart));
        }) { IsBackground = true };
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(30)));
        return run;
    }

    [Fact]
    public void AWwiseAudioThreadThatFindsTheQueueFullBlocksUntilASlotFrees()
    {
        if (IdleWaitDisabled)
        {
            return;
        }

        // 0x200 samples: 10.7 ms grains, four slots.
        var run = RunWwiseAudioThread(queueDepth: 4, grainSamples: 0x200, milliseconds: 200);

        var grainPeriods = run.WallMilliseconds / (0x200 * 1000.0 / 48000);
        // About one poll per grain period; a zero-timeout poll loop makes hundreds of thousands.
        Assert.True(run.Iterations <= 2 * grainPeriods + 8, $"{run.Iterations} polls in {run.WallMilliseconds} ms");
        // The queue is kept fed: the four slots, then a grain per grain period.
        Assert.True(run.Pushes >= 4 + grainPeriods - 3, $"{run.Pushes} pushes");
        Assert.True(run.Pushes <= 4 + grainPeriods + 2, $"{run.Pushes} pushes");
        Assert.Equal(4u, run.MaxLevel);
        Assert.True(run.EmptyPolls <= 1, $"{run.EmptyPolls} polls found the queue empty");
    }

    [Fact]
    public void AnEventFlagWaitAfterAFullQueueReadStillEndsWhenTheFlagIsSet()
    {
        if (IdleWaitDisabled)
        {
            return;
        }

        var context = CreateContext(queueDepth: 2, grainSamples: LongGrain);
        Assert.Equal(0, Push(context, NonBlocking));
        Assert.Equal(0, Push(context, NonBlocking));
        Assert.Equal((2u, 0u), GetQueueLevel(context));

        var ctx = new CpuContext(_memory, Generation.Gen5) { FsBase = PollFsBase };
        Assert.True(_memory.TryWrite(PollEventName, "idle-set\0"u8));
        ctx[CpuRegister.Rdi] = PollEventHandleSlot;
        ctx[CpuRegister.Rsi] = PollEventName;
        ctx[CpuRegister.Rdx] = 0x11;
        ctx[CpuRegister.Rcx] = 0;
        ctx[CpuRegister.R8] = 0;
        Assert.Equal(0, KernelEventFlagCompatExports.KernelCreateEventFlag(ctx));
        Assert.True(ctx.TryReadUInt64(PollEventHandleSlot, out var flag));

        // Another thread sets the flag 5 ms in: the wait ends then, not at the 50 ms cap. A dedicated
        // thread, not the pool: a busy test host can start a pool task only after the cap.
        var clock = Stopwatch.StartNew();
        var setAtMs = -1L;
        var setResult = -1;
        var setter = new Thread(() =>
        {
            Thread.Sleep(5);
            var setCtx = new CpuContext(_memory, Generation.Gen5) { FsBase = PollFsBase + 0x400 };
            setCtx[CpuRegister.Rdi] = flag;
            setCtx[CpuRegister.Rsi] = 1;
            Volatile.Write(ref setAtMs, clock.ElapsedMilliseconds);
            setResult = KernelEventFlagCompatExports.KernelSetEventFlag(setCtx);
        }) { IsBackground = true };
        setter.Start();
        _ctx[CpuRegister.Rdi] = context;
        _ctx[CpuRegister.Rsi] = PollLevelAddress;
        _ctx[CpuRegister.Rdx] = PollAvailableAddress;
        Assert.Equal(0, AudioOut2Exports.AudioOut2ContextGetQueueLevel(_ctx));
        _ctx.FsBase = PollFsBase;
        Assert.True(_ctx.TryWriteUInt32(PollTimeoutSlot, 0));
        _ctx[CpuRegister.Rdi] = flag;
        _ctx[CpuRegister.Rsi] = 1;
        _ctx[CpuRegister.Rdx] = 0x12;
        _ctx[CpuRegister.Rcx] = PollResultSlot;
        _ctx[CpuRegister.R8] = PollTimeoutSlot;
        Assert.Equal(0, KernelEventFlagCompatExports.KernelWaitEventFlag(_ctx));
        var endedAtMs = clock.ElapsedMilliseconds;
        setter.Join();
        Assert.Equal(0, setResult);

        // Measured from the set: a setter the host scheduled only after the cap says nothing.
        var setAt = Volatile.Read(ref setAtMs);
        if (setAt >= 0 && setAt < endedAtMs)
        {
            Assert.True(endedAtMs - setAt < 35, $"wait ended {endedAtMs - setAt} ms after the flag was set");
        }
    }

    // Optional benchmark: SHARPEMU_AUDIO_IDLE_WAIT_BENCH_OUTPUT=<json path>. Run it once as is and once
    // with SHARPEMU_AUDIO_OUT2_IDLE_WAIT=0 to compare the audio thread's CPU time per second.
    [Fact]
    public void BenchmarkWwiseAudioThreadPollLoop()
    {
        var output = Environment.GetEnvironmentVariable("SHARPEMU_AUDIO_IDLE_WAIT_BENCH_OUTPUT");
        if (string.IsNullOrEmpty(output))
        {
            return;
        }

        var run = RunWwiseAudioThread(queueDepth: 4, grainSamples: 0x400, milliseconds: 1000);
        File.WriteAllText(output, JsonSerializer.Serialize(new
        {
            idleWait = !IdleWaitDisabled,
            run.WallMilliseconds,
            run.Iterations,
            run.Pushes,
            run.EmptyPolls,
            threadCpuMilliseconds = run.ThreadCpuNanoseconds / 1e6,
        }));
    }
}
