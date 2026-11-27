// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Audio;
using Xunit;

namespace SharpEmu.Libs.Tests.Audio;

// The context queue the way Wwise drives it: Advance then a non-blocking Push for each grain the
// queue level says is available.
[Collection(AudioOutStateCollection.Name)]
public sealed partial class AudioOut2QueueTests : IDisposable
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const ulong ContextParamAddress = MemoryBase + 0x100;
    private const ulong ContextOutAddress = MemoryBase + 0x200;
    private const ulong PortParamAddress = MemoryBase + 0x300;
    private const ulong PortOutAddress = MemoryBase + 0x400;
    private const ulong LevelAddress = MemoryBase + 0x480;
    private const ulong AvailableAddress = MemoryBase + 0x484;
    private const ulong AttributeAddress = MemoryBase + 0x500;
    private const ulong PcmAttributeValueAddress = MemoryBase + 0x600;
    private const ulong ContextMemoryAddress = MemoryBase + 0x1000;
    private const ulong PcmAddress = MemoryBase + 0x4000;
    private const uint Blocking = 1;
    private const uint NonBlocking = 0;

    // 0x4000 samples at 48 kHz: a grain lasts 341 ms, so a push that waits for one is unmistakable.
    private const uint LongGrain = 0x4000;

    private const int MemorySize = 0x20000;

    private readonly FakeCpuMemory _memory = new(MemoryBase, MemorySize);
    private readonly CpuContext _ctx;

    public AudioOut2QueueTests()
    {
        AudioOut2Exports.ResetForTests();
        _ctx = new CpuContext(_memory, Generation.Gen5);
    }

    public void Dispose() => AudioOut2Exports.ResetForTests();

    [Fact]
    public void ANonBlockingPushIntoAFullQueueReturnsAtOnce()
    {
        var context = CreateContext(queueDepth: 2, grainSamples: LongGrain);

        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < 6; i++)
        {
            Assert.Equal(0, Push(context, NonBlocking));
        }

        Assert.True(stopwatch.ElapsedMilliseconds < 150, $"pushes took {stopwatch.ElapsedMilliseconds} ms");
        Assert.Equal((2u, 0u), GetQueueLevel(context));
    }

    [Fact]
    public void AdvanceThenANonBlockingPushFillsTheQueueWithoutWaiting()
    {
        var context = CreateContext(queueDepth: 4, grainSamples: LongGrain);
        Assert.Equal((0u, 4u), GetQueueLevel(context));

        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < 3; i++)
        {
            Advance(context);
            Assert.Equal(0, Push(context, NonBlocking));
        }

        Assert.Equal((3u, 1u), GetQueueLevel(context));

        // More grains than the queue holds: the ones without room are refused, not waited for.
        for (var i = 0; i < 4; i++)
        {
            Advance(context);
            Assert.Equal(0, Push(context, NonBlocking));
        }

        Assert.True(stopwatch.ElapsedMilliseconds < 150, $"pushes took {stopwatch.ElapsedMilliseconds} ms");
        Assert.Equal((4u, 0u), GetQueueLevel(context));
    }

    [Fact]
    public void TheQueueDrainsOneGrainPerGrainPeriod()
    {
        // 0x800 samples: 42.7 ms grains.
        var context = CreateContext(queueDepth: 2, grainSamples: 0x800);
        Assert.Equal(0, Push(context, NonBlocking));
        Assert.Equal(0, Push(context, NonBlocking));
        Assert.Equal((2u, 0u), GetQueueLevel(context));

        Thread.Sleep(150);

        Assert.Equal((0u, 2u), GetQueueLevel(context));
    }

    [Fact]
    public void ABlockingPushStillWaitsForAFreeSlot()
    {
        // 0x800 samples: 42.7 ms grains; with two slots the third and fourth push each wait a grain.
        var context = CreateContext(queueDepth: 2, grainSamples: 0x800);

        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(0, Push(context, Blocking));
        }

        Assert.True(stopwatch.ElapsedMilliseconds >= 70, $"pushes took {stopwatch.ElapsedMilliseconds} ms");
    }

    [Fact]
    public void SilentGrainsAfterDeviceAudioStillHoldTheQueue()
    {
        var stream = new DrainedAudioStream();
        AudioOut2Exports.SetStreamFactoryForTests(_ => stream);
        var context = CreateContext(queueDepth: 4, grainSamples: LongGrain);
        var port = CreateStereoPort(context);

        // One grain with audio: the context is now paced by the device, which reports it ran dry.
        SetPcmAttribute(port, LongGrain);
        Advance(context);
        Assert.Equal(0, Push(context, NonBlocking));
        // The first output primes the dropout reserve with silence before the grain (upstream's AudioOut2 reserve).
        Assert.Equal(2, stream.Submissions.Count);

        // Grains with no audio that follow occupy the queue on the context's clock; reporting the
        // empty device instead made Wwise render a full queue every update.
        for (var i = 0; i < 4; i++)
        {
            Advance(context);
            Assert.Equal(0, Push(context, NonBlocking));
        }

        Assert.Equal((4u, 0u), GetQueueLevel(context));
    }

    [Fact]
    public void DeviceAudioBeyondTheCushionCountsAsQueuedGrains()
    {
        var stream = new DrainedAudioStream { QueuedMilliseconds = 40 + 64 };
        AudioOut2Exports.SetStreamFactoryForTests(_ => stream);
        // 0x600 samples: 32 ms grains, so 64 ms beyond the cushion is two grains.
        var context = CreateContext(queueDepth: 4, grainSamples: 0x600);
        var port = CreateStereoPort(context);

        SetPcmAttribute(port, 0x600);
        Advance(context);
        Assert.Equal(0, Push(context, NonBlocking));

        Assert.Equal((2u, 2u), GetQueueLevel(context));
    }

    private int Push(ulong context, uint blocking)
    {
        _ctx[CpuRegister.Rdi] = context;
        _ctx[CpuRegister.Rsi] = blocking;
        return AudioOut2Exports.AudioOut2ContextPush(_ctx);
    }

    private void Advance(ulong context)
    {
        _ctx[CpuRegister.Rdi] = context;
        Assert.Equal(0, AudioOut2Exports.AudioOut2ContextAdvance(_ctx));
    }

    private (uint Level, uint Available) GetQueueLevel(ulong context)
    {
        _ctx[CpuRegister.Rdi] = context;
        _ctx[CpuRegister.Rsi] = LevelAddress;
        _ctx[CpuRegister.Rdx] = AvailableAddress;
        Assert.Equal(0, AudioOut2Exports.AudioOut2ContextGetQueueLevel(_ctx));
        return (ReadUInt32(LevelAddress), ReadUInt32(AvailableAddress));
    }

    private ulong CreateContext(uint queueDepth, uint grainSamples)
    {
        Span<byte> param = stackalloc byte[0x40];
        param.Clear();
        BinaryPrimitives.WriteUInt32LittleEndian(param[0x0C..], queueDepth);
        BinaryPrimitives.WriteUInt32LittleEndian(param[0x10..], grainSamples);
        Assert.True(_memory.TryWrite(ContextParamAddress, param));

        _ctx[CpuRegister.Rdi] = ContextParamAddress;
        _ctx[CpuRegister.Rsi] = ContextMemoryAddress;
        _ctx[CpuRegister.Rdx] = 0x2000;
        _ctx[CpuRegister.Rcx] = ContextOutAddress;
        Assert.Equal(0, AudioOut2Exports.AudioOut2ContextCreate(_ctx));
        return ReadUInt64(ContextOutAddress);
    }

    private ulong CreateStereoPort(ulong context)
    {
        Span<byte> param = stackalloc byte[0x40];
        param.Clear();
        // Main port, signed 16-bit stereo, 48 kHz.
        BinaryPrimitives.WriteUInt32LittleEndian(param[0x04..], 0x201);
        BinaryPrimitives.WriteUInt32LittleEndian(param[0x08..], 48000);
        Assert.True(_memory.TryWrite(PortParamAddress, param));

        _ctx[CpuRegister.Rdi] = context;
        _ctx[CpuRegister.Rsi] = PortParamAddress;
        _ctx[CpuRegister.Rdx] = PortOutAddress;
        _ctx[CpuRegister.Rcx] = 0;
        Assert.Equal(0, AudioOut2Exports.AudioOut2PortCreate(_ctx));
        return ReadUInt64(PortOutAddress);
    }

    private void SetPcmAttribute(ulong port, uint grainSamples)
    {
        // The PCM is read straight from guest memory, so the grain must fit the fake memory.
        Assert.True(grainSamples * 2 * sizeof(short) <= (ulong)MemorySize - (PcmAddress - MemoryBase));

        Span<byte> pcm = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(pcm, PcmAddress);
        Assert.True(_memory.TryWrite(PcmAttributeValueAddress, pcm));

        Span<byte> attribute = stackalloc byte[0x18];
        attribute.Clear();
        BinaryPrimitives.WriteUInt64LittleEndian(attribute[0x08..], PcmAttributeValueAddress);
        BinaryPrimitives.WriteUInt64LittleEndian(attribute[0x10..], 8);
        Assert.True(_memory.TryWrite(AttributeAddress, attribute));

        _ctx[CpuRegister.Rdi] = port;
        _ctx[CpuRegister.Rsi] = AttributeAddress;
        _ctx[CpuRegister.Rdx] = 1;
        Assert.Equal(0, AudioOut2Exports.AudioOut2PortSetAttributes(_ctx));
    }

    private uint ReadUInt32(ulong address)
    {
        Span<byte> value = stackalloc byte[4];
        Assert.True(_memory.TryRead(address, value));
        return BinaryPrimitives.ReadUInt32LittleEndian(value);
    }

    private ulong ReadUInt64(ulong address)
    {
        Span<byte> value = stackalloc byte[8];
        Assert.True(_memory.TryRead(address, value));
        return BinaryPrimitives.ReadUInt64LittleEndian(value);
    }

    private sealed class DrainedAudioStream : IHostAudioStream
    {
        public List<byte[]> Submissions { get; } = [];

        public int QueuedMilliseconds { get; init; }

        public bool Submit(ReadOnlySpan<byte> stereoPcm16)
        {
            Submissions.Add(stereoPcm16.ToArray());
            return true;
        }

        public void Dispose()
        {
        }
    }
}
