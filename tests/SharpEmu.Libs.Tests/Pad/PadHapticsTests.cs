// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.Tests.Audio;
using Xunit;

namespace SharpEmu.Libs.Tests.Pad;

// Shares the AudioOut collection: PadHaptics and the AudioOut port table are
// process-wide state that the AudioOut tests also drive.
[Collection(AudioOutStateCollection.Name)]
public sealed class PadHapticsTests : IDisposable
{
    private const ulong Base = 0x1_0000_0000;

    private readonly FakeCpuMemory _memory = new(Base, 0x1000);
    private readonly CpuContext _ctx;
    private readonly RecordingHostInput _input = new();

    public PadHapticsTests()
    {
        PadHaptics.ResetForTests();
        PadHaptics.SetHostInputForTests(_input);
        _ctx = new CpuContext(_memory, Generation.Gen5);
    }

    public void Dispose()
    {
        PadHaptics.ResetForTests();
    }

    [Fact]
    public void IntensityFromPcm_MapsEachActuatorChannelRms()
    {
        // Left actuator RMS 0.25 -> 0.5 of full scale; right RMS 0.5 -> clamps at 255.
        var pcm = Float32(0.25f, -0.5f, -0.25f, 0.5f);
        Assert.Equal(((byte)128, (byte)255), PadHaptics.IntensityFromPcm(pcm, 2, 2, 4, isFloat: true, gain: 1f));
    }

    [Fact]
    public void IntensityFromPcm_TreatsQuietChunksAsIdleAndAppliesGain()
    {
        var quiet = Float32(0.005f, 0.005f, -0.005f, -0.005f);
        Assert.Equal(((byte)0, (byte)0), PadHaptics.IntensityFromPcm(quiet, 2, 2, 4, isFloat: true, gain: 1f));

        var pcm = Float32(0.25f, 0.25f, -0.25f, -0.25f);
        Assert.Equal(((byte)64, (byte)64), PadHaptics.IntensityFromPcm(pcm, 2, 2, 4, isFloat: true, gain: 0.5f));
        Assert.Equal(((byte)0, (byte)0), PadHaptics.IntensityFromPcm(pcm, 2, 2, 4, isFloat: true, gain: 0f));
    }

    [Fact]
    public void IntensityFromPcm_MonoDrivesBothActuatorsAndReadsS16()
    {
        var pcm = new byte[4];
        BinaryPrimitives.WriteInt16LittleEndian(pcm, 16384);
        BinaryPrimitives.WriteInt16LittleEndian(pcm.AsSpan(2), -16384);
        Assert.Equal(((byte)255, (byte)255), PadHaptics.IntensityFromPcm(pcm, 2, 1, 2, isFloat: false, gain: 1f));
        Assert.Equal(((byte)0, (byte)0), PadHaptics.IntensityFromPcm(pcm, 4, 1, 2, isFloat: false, gain: 1f));
    }

    [Fact]
    public void AudioHapticsAreTimedRumbleBesideCompatVibration()
    {
        var now = 1_000L;
        PadHaptics.SetClockForTests(() => now);

        PadHaptics.SubmitAudioHaptics(100, 20);
        PadHaptics.SubmitAudioHaptics(100, 20);
        PadHaptics.SetVibration(50, 60);
        PadHaptics.ReleaseAudioHaptics();

        // The compat vibration holds the steady motors; the haptic envelope is a timed layer the host combines with
        // them, so neither overwrites the other.
        Assert.Equal([((byte)50, (byte)60)], _input.SteadyRumble);
        Assert.Equal([((byte)100, (byte)20), ((byte)0, (byte)0)], _input.TimedRumble);
        Assert.All(_input.TimedDurations, duration => Assert.Equal(PadHaptics.HapticDurationMs, duration));
    }

    [Fact]
    public void AudioHapticsAreRateLimitedAndRefreshedWhileRunning()
    {
        var now = 1_000L;
        PadHaptics.SetClockForTests(() => now);

        PadHaptics.SubmitAudioHaptics(100, 100);
        now += 5;
        PadHaptics.SubmitAudioHaptics(200, 200); // too soon after the start
        now += PadHaptics.HapticMinimumIntervalMs;
        PadHaptics.SubmitAudioHaptics(102, 100); // below the change threshold
        PadHaptics.SubmitAudioHaptics(200, 100); // a real change after the minimum interval
        now += PadHaptics.HapticRefreshIntervalMs;
        PadHaptics.SubmitAudioHaptics(200, 100); // same level: refreshed before it lapses
        PadHaptics.SubmitAudioHaptics(0, 0); // a stop goes at once

        Assert.Equal(
            [((byte)100, (byte)100), ((byte)200, (byte)100), ((byte)200, (byte)100), ((byte)0, (byte)0)],
            _input.TimedRumble);
        Assert.True(PadHaptics.HapticRefreshIntervalMs < PadHaptics.HapticDurationMs);
    }

    [NativeX64Fact]
    public void SetVibrationExport_RoutesThroughHaptics()
    {
        PadHaptics.SubmitAudioHaptics(200, 0);
        Assert.Equal(0, PadExports.PadInit(_ctx));
        _ctx[CpuRegister.Rdi] = 0x10000000;
        _ctx[CpuRegister.Rsi] = 0;
        _ctx[CpuRegister.Rdx] = 0;
        _ctx[CpuRegister.Rcx] = 0;
        var handle = PadExports.PadOpen(_ctx);
        Assert.True(handle > 0);
        _ctx[CpuRegister.Rdi] = (ulong)handle;
        _ctx[CpuRegister.Rsi] = Base + 0x100;
        Assert.True(_memory.TryWrite(Base + 0x100, new byte[] { 10, 90 }));

        Assert.Equal(0, PadExports.PadSetVibration(_ctx));
        Assert.Equal(((byte)10, (byte)90), _input.SteadyRumble[^1]);
        Assert.Equal(((byte)200, (byte)0), _input.TimedRumble[^1]);
    }

    internal static byte[] Float32(params float[] samples)
    {
        var bytes = new byte[samples.Length * sizeof(float)];
        for (var index = 0; index < samples.Length; index++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index * sizeof(float)), samples[index]);
        }

        return bytes;
    }
}

internal sealed class RecordingHostInput : IHostInput
{
    /// <summary>Every rumble change, steady and timed, in order.</summary>
    public List<(byte Large, byte Small)> Rumble { get; } = [];

    public List<(byte Large, byte Small)> SteadyRumble { get; } = [];

    public List<(byte Large, byte Small)> TimedRumble { get; } = [];

    public List<uint> TimedDurations { get; } = [];

    public void EnsureStarted()
    {
    }

    public int GetGamepadStates(Span<HostGamepadState> destination) => 0;

    public string? DescribeConnectedGamepad() => null;

    public void SetRumble(byte largeMotor, byte smallMotor)
    {
        Rumble.Add((largeMotor, smallMotor));
        SteadyRumble.Add((largeMotor, smallMotor));
    }

    public void SetHapticRumble(byte largeMotor, byte smallMotor, uint durationMs)
    {
        Rumble.Add((largeMotor, smallMotor));
        TimedRumble.Add((largeMotor, smallMotor));
        TimedDurations.Add(durationMs);
    }

    public void SetTriggerRumble(byte? leftTrigger, byte? rightTrigger)
    {
    }

    public void SetAdaptiveTriggerEffect(HostAdaptiveTriggerEffect? leftTrigger, HostAdaptiveTriggerEffect? rightTrigger)
    {
    }

    public void SetLightbar(byte red, byte green, byte blue)
    {
    }

    public void ResetLightbar()
    {
    }

    public bool IsHostWindowFocused() => false;

    public bool IsKeyDown(int virtualKey) => false;
}
