// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Tests.Gpu.Images;
using SharpEmu.Libs.Tests.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu;

// The reuse sites of the GPU worker path (SHARPEMU_WORKER_SCRATCH_REUSE): each must answer as the
// allocating form does, across reuse of the buffer and with the switch off.
[Collection(SchedulingStateCollection.Name)]
public sealed class WorkerScratchReuseTests
{
    // A guest the log's reader reads from.
    private sealed class Memory(byte[] bytes, ulong @base)
    {
        public readonly byte[] Bytes = bytes;

        public bool Read(ulong address, Span<byte> destination, bool clean)
        {
            _ = clean;
            if (address < @base || address - @base + (ulong)destination.Length > (ulong)Bytes.Length)
            {
                return false;
            }

            Bytes.AsSpan((int)(address - @base), destination.Length).CopyTo(destination);
            return true;
        }
    }

    private const ulong Base = 0x4000;

    private static byte[] Pattern(int length, byte seed)
    {
        var bytes = new byte[length];
        for (var index = 0; index < length; index++)
        {
            bytes[index] = (byte)(seed + index * 7);
        }

        return bytes;
    }

    private static void Log(GuestReadLog log, Memory memory, ulong address, int length)
    {
        var buffer = new byte[length];
        Assert.True(memory.Read(address, buffer, false));
        log.Bytes(address, true, buffer);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ReadLogHoldsWhileMemoryIsUnchangedAndFailsOnAnyChangedByte(bool reuse)
    {
        using var _ = WorkerScratch.Override(reuse);
        var memory = new Memory(Pattern(8192, 3), Base);
        var log = new GuestReadLog();
        // Runs of mixed lengths, long enough to grow the buffer more than once and to pass the stack limit.
        var lengths = new[] { 4, 32, 160, 7, 1500, 64, 12 };
        var address = Base;
        foreach (var length in lengths)
        {
            Log(log, memory, address, length);
            address += (ulong)length + 3;
        }

        log.Word(Base + 4000, true, BitConverter.ToUInt32(memory.Bytes, 4000));
        Assert.True(log.Complete);
        Assert.True(log.Holds(memory.Read));

        for (var probe = 0; probe < 4; probe++)
        {
            var offset = probe switch { 0 => 0, 1 => 100, 2 => 700, _ => 1700 };
            memory.Bytes[offset] ^= 0x5A;
            Assert.False(log.Holds(memory.Read), $"a change at +{offset} must fail the log");
            memory.Bytes[offset] ^= 0x5A;
            Assert.True(log.Holds(memory.Read));
        }
    }

    [Fact]
    public void ReadLogAfterClearHoldsOnlyTheNewRuns()
    {
        using var _ = WorkerScratch.Override(true);
        var memory = new Memory(Pattern(4096, 9), Base);
        var log = new GuestReadLog();
        Log(log, memory, Base, 256);
        Log(log, memory, Base + 512, 256);
        Assert.Equal(2, log.Count);
        log.Clear();
        Assert.Equal(0, log.Count);
        Assert.True(log.Complete);

        // The old runs' bytes are overwritten in the shared buffer, and a change under them is not seen.
        Log(log, memory, Base + 1024, 8);
        memory.Bytes[0] ^= 0xFF;
        memory.Bytes[600] ^= 0xFF;
        Assert.True(log.Holds(memory.Read));
        memory.Bytes[1024] ^= 0xFF;
        Assert.False(log.Holds(memory.Read));
    }

    [Fact]
    public void ReadLogPastItsBoundIsNotComplete()
    {
        using var _ = WorkerScratch.Override(true);
        var memory = new Memory(Pattern(70 * 1024, 1), Base);
        var log = new GuestReadLog();
        Log(log, memory, Base, 40 * 1024);
        Assert.True(log.Complete);
        Log(log, memory, Base + 40 * 1024, 40);
        Assert.True(log.Complete);
        // 64 KiB bound: this read passes it.
        var big = new byte[30 * 1024];
        log.Bytes(Base, true, big);
        Assert.False(log.Complete);
        Assert.False(log.Holds(memory.Read));
    }

    [Fact]
    public void ReadLogReusesItsBufferAcrossClears()
    {
        using var _ = WorkerScratch.Override(true);
        var memory = new Memory(Pattern(4096, 5), Base);
        var log = new GuestReadLog();
        for (var round = 0; round < 3; round++)
        {
            log.Clear();
            for (var run = 0; run < 8; run++)
            {
                Log(log, memory, Base + (ulong)run * 128, 100);
            }
        }

        var before = WorkerScratch.Peek(WorkerScratch.Site.ReadLogRuns);
        log.Clear();
        for (var run = 0; run < 8; run++)
        {
            Log(log, memory, Base + (ulong)run * 128, 100);
        }

        var after = WorkerScratch.Peek(WorkerScratch.Site.ReadLogRuns);
        Assert.Equal(8, after.Reused - before.Reused);
        Assert.Equal(before.Allocated, after.Allocated);
        Assert.True(log.Holds(memory.Read));
    }

    [Fact]
    public void ReadLogWithTheSwitchOffCopiesEveryRun()
    {
        using var _ = WorkerScratch.Override(false);
        var memory = new Memory(Pattern(1024, 5), Base);
        var log = new GuestReadLog();
        var before = WorkerScratch.Peek(WorkerScratch.Site.ReadLogRuns);
        Log(log, memory, Base, 64);
        Log(log, memory, Base + 100, 64);
        var after = WorkerScratch.Peek(WorkerScratch.Site.ReadLogRuns);
        Assert.Equal(2, after.Allocated - before.Allocated);
        Assert.Equal(before.Reused, after.Reused);
        Assert.Equal(2, log.Count);
    }

    // A frame alternates between depth states; the memo answers each from its own slot, as a rebuilt one would.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DepthTargetAlternatingStatesMatchAFreshBuild(bool reuse)
    {
        using var fatal = new FatalScope();
        using var _ = WorkerScratch.Override(reuse);
        var device = new AcceptingFormatSupport();
        var first = RegisterWords.Depth(0x1_0000_0000, 64, 64);
        var second = RegisterWords.Depth(0x1_0004_0000, 128, 32);
        var third = RegisterWords.Depth(0x1_0008_0000, 96, 96);
        for (var round = 0; round < 3; round++)
        {
            foreach (var (words, address, width, height) in new[]
            {
                (first, 0x1_0000_0000ul, 64u, 64u),
                (second, 0x1_0004_0000ul, 128u, 32u),
                (third, 0x1_0008_0000ul, 96u, 96u),
                (first, 0x1_0000_0000ul, 64u, 64u),
            })
            {
                var resolved = ImageRequestBuilders.DepthTarget(words, device);
                Assert.NotNull(resolved);
                Assert.Equal(address, resolved.Value.DepthAddress);
                Assert.Equal(width, resolved.Value.Width);
                Assert.Equal(height, resolved.Value.Height);
            }
        }

        Assert.Empty(fatal.Messages);
    }
}
