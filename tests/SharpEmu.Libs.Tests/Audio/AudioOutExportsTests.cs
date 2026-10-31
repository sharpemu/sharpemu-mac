// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Audio;
using Xunit;

namespace SharpEmu.Libs.Tests.Audio;

[CollectionDefinition("AudioOutState", DisableParallelization = true)]
public sealed class AudioOutStateCollection
{
    public const string Name = "AudioOutState";
}

[Collection(AudioOutStateCollection.Name)]
public sealed class AudioOutExportsTests : IDisposable
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const int MemorySize = 0x4000;
    private const ulong ParameterAddress = MemoryBase + 0x100;
    private const ulong FirstSourceAddress = MemoryBase + 0x1000;
    private const ulong SecondSourceAddress = MemoryBase + 0x2000;

    private readonly FakeCpuMemory _memory = new(MemoryBase, MemorySize);
    private readonly CpuContext _ctx;
    private readonly List<RecordingAudioStream> _streams = [];

    public AudioOutExportsTests()
    {
        AudioOutExports.ResetForTests();
        AudioOutExports.SetStreamFactoryForTests(_ =>
        {
            var stream = new RecordingAudioStream();
            _streams.Add(stream);
            return stream;
        });
        _ctx = new CpuContext(_memory, Generation.Gen5);
        Assert.Equal(0, AudioOutExports.AudioOutInit(_ctx));
    }

    [Fact]
    public void Outputs_StagesAndSubmitsSingleStereoPort()
    {
        var handle = OpenPort(bufferLength: 2);
        Span<byte> source = stackalloc byte[8];
        BinaryPrimitives.WriteInt16LittleEndian(source, -32768);
        BinaryPrimitives.WriteInt16LittleEndian(source[2..], 32767);
        BinaryPrimitives.WriteInt16LittleEndian(source[4..], -1234);
        BinaryPrimitives.WriteInt16LittleEndian(source[6..], 5678);
        Assert.True(_memory.TryWrite(FirstSourceAddress, source));
        WriteDescriptor(0, handle, FirstSourceAddress);

        var result = Submit(outputCount: 1);

        Assert.Equal(2, result);
        Assert.Equal(2UL, _ctx[CpuRegister.Rax]);
        Assert.Equal(source.ToArray(), Assert.Single(_streams[0].Submissions));
    }

    [Fact]
    public void Output_PacesAfterShutdownWithoutUsingDisposedStream()
    {
        var handle = OpenPort(bufferLength: 4800);
        var stream = Assert.Single(_streams);
        AudioOutExports.ShutdownAllPorts();

        // Started before the first submit: pacing runs from that submit, so a late start would
        // leave scheduling delay out of the elapsed time on a loaded runner.
        var timer = Stopwatch.StartNew();
        Assert.Equal(0, SubmitSingle(handle, sourceAddress: 0));
        var result = SubmitSingle(handle, sourceAddress: 0);

        Assert.Equal(0, result);
        Assert.True(timer.Elapsed >= TimeSpan.FromMilliseconds(50));
        Assert.True(stream.IsDisposed);
        Assert.Empty(stream.Submissions);
    }

    [Fact]
    public void Outputs_SubmitsEveryPortInTheBatch()
    {
        var firstHandle = OpenPort(bufferLength: 2);
        var secondHandle = OpenPort(bufferLength: 2);
        byte[] firstSource = [1, 0, 2, 0, 3, 0, 4, 0];
        byte[] secondSource = [5, 0, 6, 0, 7, 0, 8, 0];
        Assert.True(_memory.TryWrite(FirstSourceAddress, firstSource));
        Assert.True(_memory.TryWrite(SecondSourceAddress, secondSource));

        // Reverse handle order to exercise canonical lock ordering without
        // changing which guest buffer belongs to each port.
        WriteDescriptor(0, secondHandle, FirstSourceAddress);
        WriteDescriptor(1, firstHandle, SecondSourceAddress);

        var result = Submit(outputCount: 2);

        Assert.Equal(2, result);
        Assert.Equal(secondSource, Assert.Single(_streams[0].Submissions));
        Assert.Equal(firstSource, Assert.Single(_streams[1].Submissions));
    }

    [Fact]
    public void Outputs_AcceptsNullBufferAsSynchronizationOnly()
    {
        var handle = OpenPort(bufferLength: 2);
        WriteDescriptor(0, handle, sourceAddress: 0);

        var result = Submit(outputCount: 1);

        Assert.Equal(2, result);
        Assert.Empty(_streams[0].Submissions);
    }

    [Fact]
    public void Outputs_FaultInLaterBufferDoesNotPartiallySubmit()
    {
        var firstHandle = OpenPort(bufferLength: 2);
        var secondHandle = OpenPort(bufferLength: 2);
        Assert.True(_memory.TryWrite(FirstSourceAddress, new byte[8]));
        WriteDescriptor(0, firstHandle, FirstSourceAddress);
        WriteDescriptor(1, secondHandle, MemoryBase + MemorySize);

        var result = Submit(outputCount: 2);

        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPointer, result);
        Assert.All(_streams, stream => Assert.Empty(stream.Submissions));
    }

    [Fact]
    public void Outputs_RejectsDuplicateHandlesBeforeSubmission()
    {
        var handle = OpenPort(bufferLength: 2);
        Assert.True(_memory.TryWrite(FirstSourceAddress, new byte[8]));
        Assert.True(_memory.TryWrite(SecondSourceAddress, new byte[8]));
        WriteDescriptor(0, handle, FirstSourceAddress);
        WriteDescriptor(1, handle, SecondSourceAddress);

        var result = Submit(outputCount: 2);

        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, result);
        Assert.Empty(_streams[0].Submissions);
    }

    [Fact]
    public void Outputs_RejectsPortsWithDifferentBufferLengths()
    {
        var firstHandle = OpenPort(bufferLength: 2);
        var secondHandle = OpenPort(bufferLength: 4);
        WriteDescriptor(0, firstHandle, FirstSourceAddress);
        WriteDescriptor(1, secondHandle, SecondSourceAddress);

        var result = Submit(outputCount: 2);

        Assert.Equal(AudioOutExports.AudioOutErrorInvalidSize, result);
        Assert.All(_streams, stream => Assert.Empty(stream.Submissions));
    }

    [Fact]
    public void Outputs_RejectsUnknownHandle()
    {
        WriteDescriptor(0, 99, FirstSourceAddress);

        Assert.Equal(
            AudioOutExports.AudioOutErrorInvalidPort,
            Submit(outputCount: 1));
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(26u)]
    public void Outputs_RejectsInvalidOutputCount(uint outputCount)
    {
        Assert.Equal(
            AudioOutExports.AudioOutErrorPortFull,
            Submit(outputCount));
    }

    [Fact]
    public void Outputs_RejectsNullOrUnreadableParameterArray()
    {
        Assert.Equal(
            AudioOutExports.AudioOutErrorInvalidPointer,
            Submit(outputCount: 1, parameterAddress: 0));
        Assert.Equal(
            AudioOutExports.AudioOutErrorInvalidPointer,
            Submit(outputCount: 1, parameterAddress: MemoryBase + MemorySize));
    }

    [Fact]
    public void OutputsExportRegistersForBothGenerations()
    {
        foreach (var generation in new[] { Generation.Gen4, Generation.Gen5 })
        {
            var manager = new ModuleManager();
            manager.RegisterExports(
                SharpEmu.Generated.SysAbiExportRegistry.CreateExports(generation));

            Assert.True(manager.TryGetExport("w3PdaSTSwGE", out var export));
            Assert.Equal("sceAudioOutOutputs", export.Name);
            Assert.Equal("libSceAudioOut", export.LibraryName);
            Assert.True(manager.TryGetExport("QOQtbeDqsT4", out var singleExport));
            Assert.Equal("sceAudioOutOutput", singleExport.Name);
            Assert.Equal("libSceAudioOut", singleExport.LibraryName);
        }
    }

    [Fact]
    public void Init_RejectsSecondCallAndOpenRequiresInit()
    {
        Assert.Equal(AudioOutExports.AudioOutErrorAlreadyInit, AudioOutExports.AudioOutInit(_ctx));

        AudioOutExports.ResetForTests();
        Assert.Equal(AudioOutExports.AudioOutErrorNotInit, Open(type: 0));
        Assert.Equal(0, AudioOutExports.AudioOutInit(_ctx));
        Assert.Equal(1, Open(type: 0));
    }

    [Theory]
    [InlineData(5, 2u, 48000u, 0, 1u, unchecked((int)0x8026000A))]
    [InlineData(6, 2u, 48000u, 0, 1u, unchecked((int)0x8026000A))]
    [InlineData(128, 2u, 48000u, 0, 1u, unchecked((int)0x8026000A))]
    [InlineData(0, 0u, 48000u, 0, 1u, unchecked((int)0x80260006))]
    [InlineData(0, 2u, 0u, 0, 1u, unchecked((int)0x80260008))]
    [InlineData(0, 2u, 48000u, 1, 1u, unchecked((int)0x80260016))]
    [InlineData(0, 2u, 48000u, 0, 8u, unchecked((int)0x80260007))]
    public void Open_ValidatesWithAudioOutErrorCodes(
        int type, uint bufferLength, uint frequency, int index, uint format, int expected)
    {
        Assert.Equal(expected, Open(type, bufferLength, frequency, index, format));
        Assert.Empty(_streams);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(126)]
    [InlineData(127)]
    public void Open_AcceptsEveryApplicationPortType(int type)
    {
        Assert.Equal(1, Open(type));
        Assert.Single(_streams);
    }

    [Fact]
    public void Open_LimitsOpenPortsAndReusesClosedSlots()
    {
        for (var expected = 1; expected <= AudioOutExports.MaxOpenPorts; expected++)
        {
            Assert.Equal(expected, Open(type: expected % 2));
        }

        Assert.Equal(AudioOutExports.AudioOutErrorPortFull, Open(type: 0));

        _ctx[CpuRegister.Rdi] = 7;
        Assert.Equal(0, AudioOutExports.AudioOutClose(_ctx));
        Assert.Equal(7, Open(type: 0));
    }

    [Fact]
    public void Close_RejectsUnknownAndClosedHandles()
    {
        var handle = Open(type: 0);
        _ctx[CpuRegister.Rdi] = 99;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutClose(_ctx));
        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        Assert.Equal(0, AudioOutExports.AudioOutClose(_ctx));
        Assert.True(_streams[0].IsDisposed);
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutClose(_ctx));
    }

    [Theory]
    [InlineData(0, 1u, 0x01, 2, -1)]
    [InlineData(0, 2u, 0x01, 2, -1)]
    [InlineData(0, 0u, 0x01, 1, -1)]
    [InlineData(2, 1u, 0x40, 1, -1)]
    [InlineData(4, 0u, 0x04, 1, 127)]
    [InlineData(127, 1u, 0x80, 0, -1)]
    public void GetPortState_WritesTheOutputOfThePortType(
        int type, uint format, int output, int channel, int volume)
    {
        const ulong stateAddress = MemoryBase + 0x200;
        Assert.True(_memory.TryWrite(stateAddress, Enumerable.Repeat((byte)0xA5, 0x28).ToArray()));
        var handle = Open(type, format: format);

        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        _ctx[CpuRegister.Rsi] = stateAddress;
        Assert.Equal(0, AudioOutExports.AudioOutGetPortState(_ctx));

        var state = new byte[0x28];
        Assert.True(_memory.TryRead(stateAddress, state));
        Assert.Equal(output, BinaryPrimitives.ReadUInt16LittleEndian(state));
        Assert.Equal(channel, state[2]);
        Assert.Equal(volume, BinaryPrimitives.ReadInt16LittleEndian(state.AsSpan(4)));
        Assert.All(state[6..0x20], value => Assert.Equal(0, value));
        Assert.All(state[0x20..], value => Assert.Equal(0xA5, value));
    }

    [Fact]
    public void GetPortState_RejectsNullStateAndUnknownPort()
    {
        var handle = Open(type: 0);
        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        _ctx[CpuRegister.Rsi] = 0;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPointer, AudioOutExports.AudioOutGetPortState(_ctx));
        _ctx[CpuRegister.Rdi] = 99;
        _ctx[CpuRegister.Rsi] = MemoryBase + 0x200;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutGetPortState(_ctx));
    }

    [Fact]
    public void Output_ReturnsFrameCountAndRejectsBadHandleOrBuffer()
    {
        var handle = OpenPort(bufferLength: 2);
        Assert.True(_memory.TryWrite(FirstSourceAddress, new byte[8]));

        Assert.Equal(2, SubmitSingle(handle, FirstSourceAddress));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, SubmitSingle(99, FirstSourceAddress));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPointer, SubmitSingle(handle, MemoryBase + MemorySize));
        Assert.Single(_streams[0].Submissions);
    }

    [Fact]
    public void GetLastOutputTime_TracksBlocksCarryingData()
    {
        const ulong timeAddress = MemoryBase + 0x300;
        var handle = OpenPort(bufferLength: 2);
        Assert.True(_memory.TryWrite(FirstSourceAddress, new byte[8]));

        Assert.Equal(0UL, LastOutputTime(handle, timeAddress));
        var before = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.ReadProcessTimeMicroseconds();
        Assert.Equal(2, SubmitSingle(handle, FirstSourceAddress));
        var after = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.ReadProcessTimeMicroseconds();
        var stamped = LastOutputTime(handle, timeAddress);
        Assert.InRange(stamped, before, after);

        Assert.Equal(2, SubmitSingle(handle, sourceAddress: 0));
        Assert.Equal(stamped, LastOutputTime(handle, timeAddress));

        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        _ctx[CpuRegister.Rsi] = 0;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPointer, AudioOutExports.AudioOutGetLastOutputTime(_ctx));
        _ctx[CpuRegister.Rdi] = 99;
        _ctx[CpuRegister.Rsi] = timeAddress;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutGetLastOutputTime(_ctx));
    }

    [Fact]
    public void SetVolume_ValidatesPointerAndPort()
    {
        var handle = OpenPort(bufferLength: 2);
        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        _ctx[CpuRegister.Rsi] = 3;
        _ctx[CpuRegister.Rdx] = 0;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPointer, AudioOutExports.AudioOutSetVolume(_ctx));
        _ctx[CpuRegister.Rdi] = 99;
        _ctx[CpuRegister.Rdx] = MemoryBase + 0x200;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutSetVolume(_ctx));
    }

    [Fact]
    public void SetMixLevelPadSpk_ValidatesAndScalesPadSpeakerOutput()
    {
        var main = Open(type: 0);
        var padSpeaker = Open(type: 4);
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPortType, SetMixLevel(main, 16384));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, SetMixLevel(99, 16384));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidMixLevel, SetMixLevel(padSpeaker, 32769));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidMixLevel, SetMixLevel(padSpeaker, -1));
        Assert.Equal(0, SetMixLevel(padSpeaker, 16384));

        byte[] source = [0xE8, 0x03, 0x18, 0xFC, 0xD0, 0x07, 0x30, 0xF8];
        Assert.True(_memory.TryWrite(FirstSourceAddress, source));
        Assert.Equal(2, SubmitSingle(padSpeaker, FirstSourceAddress));
        var played = Assert.Single(_streams[1].Submissions);
        Assert.Equal(500, BinaryPrimitives.ReadInt16LittleEndian(played));
        Assert.Equal(-500, BinaryPrimitives.ReadInt16LittleEndian(played.AsSpan(2)));
    }

    [Fact]
    public void ExOpen_OpensBitstreamPortsWithoutAHostStream()
    {
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidArg, ExOpen(5));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidArg, ExOpen(7));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidArg, ExOpen(11));

        var ac3 = ExOpen(0);
        var eac3 = ExOpen(4);
        Assert.Equal(1, ac3);
        Assert.Equal(2, eac3);
        Assert.Empty(_streams);
        Assert.True(_memory.TryWrite(FirstSourceAddress, new byte[256 * 4]));
        Assert.Equal(256, SubmitSingle(ac3, FirstSourceAddress));

        _ctx[CpuRegister.Rdi] = unchecked((ulong)ac3);
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutClose(_ctx));
        Assert.Equal(0, AudioOutExports.AudioOutExClose(_ctx));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutExClose(_ctx));

        var main = Open(type: 0);
        _ctx[CpuRegister.Rdi] = unchecked((ulong)main);
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutExClose(_ctx));
    }

    [Theory]
    [InlineData(1, 0UL, 0u, 1u, 0UL, unchecked((int)0x80268009))]
    [InlineData(0, 0UL, 0u, 1u, 0UL, 0)]
    [InlineData(0, 0UL, 0u, 0xFFu, 0UL, 0)]
    [InlineData(0, 0UL, 0xFFu, 0xFFu, 0UL, 0)]
    [InlineData(0, 0UL, 11u, 1u, 0UL, unchecked((int)0x80260016))]
    [InlineData(0, 0UL, 0u, 4u, 0UL, unchecked((int)0x80260016))]
    [InlineData(0, 1UL, 0u, 1u, 0UL, unchecked((int)0x80260016))]
    public void ExConfigureOutput_ValidatesArguments(
        int control, ulong flags, uint mode, uint target, ulong option, int expected)
    {
        _ctx[CpuRegister.Rdi] = unchecked((ulong)control);
        _ctx[CpuRegister.Rsi] = flags;
        _ctx[CpuRegister.Rdx] = mode;
        _ctx[CpuRegister.Rcx] = target;
        _ctx[CpuRegister.R8] = option;
        Assert.Equal(expected, AudioOutExports.AudioOutExConfigureOutput(_ctx));
    }

    [Fact]
    public void MonitorInfo_ZeroesTheDeclaredSize()
    {
        const ulong infoAddress = MemoryBase + 0x2000;
        Assert.True(_memory.TryWrite(infoAddress, Enumerable.Repeat((byte)0xA5, 0x400).ToArray()));

        _ctx[CpuRegister.Rdi] = 0;
        _ctx[CpuRegister.Rsi] = 0;
        _ctx[CpuRegister.Rdx] = infoAddress;
        _ctx[CpuRegister.Rcx] = 0x9F;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidSize, AudioOutExports.AudioOutExGetMonitorInfo(_ctx));
        _ctx[CpuRegister.Rcx] = 0x100;
        _ctx[CpuRegister.Rdx] = 0;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPointer, AudioOutExports.AudioOutExGetMonitorInfo(_ctx));
        _ctx[CpuRegister.Rdx] = infoAddress;
        Assert.Equal(0, AudioOutExports.AudioOutExGetMonitorInfo(_ctx));
        var info = new byte[0x400];
        Assert.True(_memory.TryRead(infoAddress, info));
        Assert.All(info[..0x100], value => Assert.Equal(0, value));
        Assert.All(info[0x100..], value => Assert.Equal(0xA5, value));

        _ctx[CpuRegister.Rdi] = 1;
        _ctx[CpuRegister.Rsi] = infoAddress;
        _ctx[CpuRegister.Rdx] = 0x100;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidSize, AudioOutExports.AudioOutSysGetHdmiMonitorInfo(_ctx));
        _ctx[CpuRegister.Rdx] = 0x180;
        Assert.Equal(0, AudioOutExports.AudioOutSysGetHdmiMonitorInfo(_ctx));
        Assert.True(_memory.TryRead(infoAddress, info));
        Assert.All(info[..0x180], value => Assert.Equal(0, value));
        _ctx[CpuRegister.Rdi] = 2;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidParam, AudioOutExports.AudioOutSysGetHdmiMonitorInfo(_ctx));
    }

    [Fact]
    public void SysAndDeviceIdOpen_ReportNoUsablePort()
    {
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPortType, AudioOutExports.AudioOutSysOpen(_ctx));
        _ctx[CpuRegister.Rdi] = 1;
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidPort, AudioOutExports.AudioOutSysClose(_ctx));
        Assert.Equal(AudioOutExports.AudioOutErrorInvalidParam, AudioOutExports.AudioOutDeviceIdOpen(_ctx));
    }

    private int ExOpen(uint mode)
    {
        _ctx[CpuRegister.Rdi] = 0xFF;
        _ctx[CpuRegister.Rsi] = mode;
        return AudioOutExports.AudioOutExOpen(_ctx);
    }

    private int SetMixLevel(int handle, int level)
    {
        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        _ctx[CpuRegister.Rsi] = unchecked((ulong)level);
        return AudioOutExports.AudioOutSetMixLevelPadSpk(_ctx);
    }

    private ulong LastOutputTime(int handle, ulong address)
    {
        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        _ctx[CpuRegister.Rsi] = address;
        Assert.Equal(0, AudioOutExports.AudioOutGetLastOutputTime(_ctx));
        var value = new byte[8];
        Assert.True(_memory.TryRead(address, value));
        return BinaryPrimitives.ReadUInt64LittleEndian(value);
    }

    [Fact]
    public void VibrationPort_DrivesPadHapticsAndNeverReachesSpeakers()
    {
        var input = new SharpEmu.Libs.Tests.Pad.RecordingHostInput();
        SharpEmu.Libs.Pad.PadHaptics.ResetForTests();
        SharpEmu.Libs.Pad.PadHaptics.SetHostInputForTests(input);
        try
        {
            // Float stereo (format 4): left actuator RMS 0.25, right RMS 0.5.
            var handle = Open(type: 10, bufferLength: 2, format: 4);
            Assert.Equal(1, handle);
            Assert.Empty(_streams);
            Assert.True(_memory.TryWrite(
                FirstSourceAddress,
                SharpEmu.Libs.Tests.Pad.PadHapticsTests.Float32(0.25f, -0.5f, -0.25f, 0.5f)));

            Assert.Equal(2, SubmitSingle(handle, FirstSourceAddress));
            Assert.Equal([((byte)128, (byte)255)], input.Rumble);

            _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
            Assert.Equal(0, AudioOutExports.AudioOutClose(_ctx));
            Assert.Equal(((byte)0, (byte)0), input.Rumble[^1]);
            Assert.Empty(_streams);
        }
        finally
        {
            SharpEmu.Libs.Pad.PadHaptics.ResetForTests();
        }
    }

    [Theory]
    [InlineData(Generation.Gen4)]
    [InlineData(Generation.Gen5)]
    public void OutputExportSubmitsThroughTheSharedHostAudioPath(Generation generation)
    {
        var handle = OpenPort(bufferLength: 2);
        byte[] samples = [1, 0, 2, 0, 3, 0, 4, 0];
        Assert.True(_memory.TryWrite(FirstSourceAddress, samples));
        var manager = new ModuleManager();
        manager.RegisterExports(SharpEmu.Generated.SysAbiExportRegistry.CreateExports(generation));
        var context = new CpuContext(_memory, generation);
        context[CpuRegister.Rdi] = (ulong)handle;
        context[CpuRegister.Rsi] = FirstSourceAddress;

        Assert.True(manager.TryDispatch("QOQtbeDqsT4", context, out var result));
        // The fork's sceAudioOutOutput returns the port's frame count, like sceAudioOutOutputs.
        Assert.Equal(2, (int)result);
        Assert.Equal(samples, Assert.Single(Assert.Single(_streams).Submissions));
    }

    public void Dispose() => AudioOutExports.ResetForTests();

    private int OpenPort(uint bufferLength) => Open(type: 0, bufferLength: bufferLength);

    private int Open(int type, uint bufferLength = 2, uint frequency = 48000, int index = 0, uint format = 1)
    {
        _ctx[CpuRegister.Rdi] = 1;
        _ctx[CpuRegister.Rsi] = unchecked((ulong)type);
        _ctx[CpuRegister.Rdx] = unchecked((ulong)index);
        _ctx[CpuRegister.Rcx] = bufferLength;
        _ctx[CpuRegister.R8] = frequency;
        _ctx[CpuRegister.R9] = format;
        return AudioOutExports.AudioOutOpen(_ctx);
    }

    private int Submit(uint outputCount, ulong parameterAddress = ParameterAddress)
    {
        _ctx[CpuRegister.Rdi] = parameterAddress;
        _ctx[CpuRegister.Rsi] = outputCount;
        return AudioOutExports.AudioOutOutputs(_ctx);
    }

    private int SubmitSingle(int handle, ulong sourceAddress)
    {
        _ctx[CpuRegister.Rdi] = unchecked((ulong)handle);
        _ctx[CpuRegister.Rsi] = sourceAddress;
        return AudioOutExports.AudioOutOutput(_ctx);
    }

    private void WriteDescriptor(int index, int handle, ulong sourceAddress)
    {
        Span<byte> descriptor = stackalloc byte[16];
        descriptor.Clear();
        BinaryPrimitives.WriteInt32LittleEndian(descriptor, handle);
        BinaryPrimitives.WriteUInt64LittleEndian(descriptor[8..], sourceAddress);
        Assert.True(_memory.TryWrite(
            ParameterAddress + unchecked((ulong)(index * descriptor.Length)),
            descriptor));
    }

    private sealed class RecordingAudioStream : IHostAudioStream
    {
        public List<byte[]> Submissions { get; } = [];
        public bool IsDisposed { get; private set; }

        public bool Submit(ReadOnlySpan<byte> stereoPcm16)
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            Submissions.Add(stereoPcm16.ToArray());
            return true;
        }

        public void Dispose()
        {
            IsDisposed = true;
        }
    }
}
