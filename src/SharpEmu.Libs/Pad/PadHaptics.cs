// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;
using System.Buffers.Binary;

namespace SharpEmu.Libs.Pad;

/// <summary>
/// DualSense voice-coil haptics. On PS5 a title drives the two actuators by
/// writing a haptic waveform to a vibration port (sceAudioOut port type
/// VIBRATION, sceAudioOut2 port type 0x6): channel 0 is the left actuator,
/// channel 1 the right one, and a mono port drives both. The host pad only
/// exposes two rumble motors through <see cref="IHostInput.SetRumble"/>, so
/// each output chunk is reduced to one intensity per actuator and the left
/// actuator drives the large motor, the right one the small motor. The audio
/// itself never reaches the speakers.
/// </summary>
/// <remarks>
/// scePadSetVibration keeps working alongside: it sets the host pad's steady
/// motors (<see cref="IHostInput.SetRumble"/>), while the haptic envelope goes
/// out as timed rumble (<see cref="IHostInput.SetHapticRumble"/>) that the host
/// layers over them at the stronger of the two, so a title that clears the
/// compat vibration every frame does not cut its haptics. The timed level
/// lapses <see cref="HapticDurationMs"/> after the last refresh, so a port the
/// guest stops feeding (a stall, a hang) does not leave the motors running.
/// Updates are rate-limited as upstream's HapticRumble was: a stop and a start
/// go at once, a change of at least <see cref="HapticChangeThreshold"/> after
/// <see cref="HapticMinimumIntervalMs"/>, and a running level is refreshed every
/// <see cref="HapticRefreshIntervalMs"/>. <c>SHARPEMU_HAPTIC_RUMBLE_GAIN</c>
/// scales the envelope (default 1; 0 turns audio-driven rumble off).
/// </remarks>
public static class PadHaptics
{
    // Below this RMS a chunk is treated as silence (the actuators are idle),
    // so dither and DC-free noise floors do not keep the motors spinning.
    internal const double SilenceRms = 0.01;

    internal const uint HapticDurationMs = 250;
    internal const long HapticRefreshIntervalMs = 100;
    internal const long HapticMinimumIntervalMs = 16;
    internal const int HapticChangeThreshold = 6;

    private static readonly float Gain = ReadGain();

    private static readonly object Gate = new();
    private static byte _sentHapticLeft;
    private static byte _sentHapticRight;
    private static long _sentHapticAt;
    private static IHostInput? _hostInputForTests;
    private static Func<long>? _clockForTests;

    /// <summary>
    /// Reduces one chunk of interleaved haptic PCM to an intensity per
    /// actuator: the RMS of the actuator's channel, doubled and clamped to
    /// full scale, then scaled to 0..255. A chunk whose RMS is below
    /// <see cref="SilenceRms"/> yields 0 for that actuator.
    /// </summary>
    /// <param name="pcm">Interleaved samples, <paramref name="frames"/> × <paramref name="channels"/>.</param>
    /// <param name="frames">Frame count of the chunk.</param>
    /// <param name="channels">Interleaved channel count (1 drives both actuators).</param>
    /// <param name="bytesPerSample">2 (signed 16-bit) or 4 (float32).</param>
    /// <param name="isFloat">Whether samples are float32.</param>
    /// <param name="gain">Linear gain applied before the envelope (port volume).</param>
    public static (byte Left, byte Right) IntensityFromPcm(
        ReadOnlySpan<byte> pcm,
        int frames,
        int channels,
        int bytesPerSample,
        bool isFloat,
        float gain)
    {
        if (frames <= 0 || channels <= 0 || bytesPerSample is not (2 or 4) ||
            (isFloat && bytesPerSample != 4) ||
            pcm.Length < (long)frames * channels * bytesPerSample ||
            !float.IsFinite(gain) || gain <= 0f)
        {
            return (0, 0);
        }

        var left = ChannelIntensity(pcm, frames, channels, bytesPerSample, isFloat, 0, gain);
        var right = channels == 1
            ? left
            : ChannelIntensity(pcm, frames, channels, bytesPerSample, isFloat, 1, gain);
        return (left, right);
    }

    private static byte ChannelIntensity(
        ReadOnlySpan<byte> pcm,
        int frames,
        int channels,
        int bytesPerSample,
        bool isFloat,
        int channel,
        float gain)
    {
        var energy = 0.0;
        var stride = channels * bytesPerSample;
        for (var frame = 0; frame < frames; frame++)
        {
            var sample = pcm.Slice(frame * stride + channel * bytesPerSample, bytesPerSample);
            double value = isFloat
                ? BinaryPrimitives.ReadSingleLittleEndian(sample)
                : BinaryPrimitives.ReadInt16LittleEndian(sample) / 32768.0;
            if (double.IsFinite(value))
            {
                value *= gain;
                energy += value * value;
            }
        }

        var rms = Math.Sqrt(energy / frames);
        return rms < SilenceRms
            ? (byte)0
            : (byte)Math.Round(Math.Min(1.0, rms * 2.0) * 255.0, MidpointRounding.AwayFromZero);
    }

    /// <summary>
    /// Sets the actuator intensities derived from the latest haptic chunk.
    /// The host pad gets them as timed rumble, rate-limited (see the remarks).
    /// </summary>
    public static void SubmitAudioHaptics(byte left, byte right)
    {
        if (Gain <= 0f)
        {
            return;
        }

        left = Scale(left);
        right = Scale(right);
        lock (Gate)
        {
            var now = (Volatile.Read(ref _clockForTests) ?? ClockMilliseconds)();
            if (!ShouldSendHaptics(left, right, now - _sentHapticAt))
            {
                return;
            }

            _sentHapticLeft = left;
            _sentHapticRight = right;
            _sentHapticAt = now;
            Input.SetHapticRumble(left, right, HapticDurationMs);
        }
    }

    /// <summary>Stops the haptic contribution (vibration port closed).</summary>
    public static void ReleaseAudioHaptics() => SubmitAudioHaptics(0, 0);

    /// <summary>scePadSetVibration: always forwarded to the host.</summary>
    internal static void SetVibration(byte largeMotor, byte smallMotor)
    {
        lock (Gate)
        {
            Input.SetRumble(largeMotor, smallMotor);
        }
    }

    private static IHostInput Input => Volatile.Read(ref _hostInputForTests) ?? HostPlatform.Current.Input;

    private static long ClockMilliseconds() => Environment.TickCount64;

    // Caller holds Gate.
    private static bool ShouldSendHaptics(byte left, byte right, long elapsedMs)
    {
        var wasRunning = (_sentHapticLeft | _sentHapticRight) != 0;
        if ((left | right) == 0)
        {
            return wasRunning;
        }

        if (!wasRunning || elapsedMs >= HapticRefreshIntervalMs)
        {
            return true;
        }

        return elapsedMs >= HapticMinimumIntervalMs &&
               (Math.Abs(left - _sentHapticLeft) >= HapticChangeThreshold ||
                Math.Abs(right - _sentHapticRight) >= HapticChangeThreshold);
    }

    private static byte Scale(byte level) =>
        Gain == 1f ? level : (byte)Math.Clamp(MathF.Round(level * Gain), 0f, 255f);

    private static float ReadGain()
    {
        var value = Environment.GetEnvironmentVariable("SHARPEMU_HAPTIC_RUMBLE_GAIN");
        return float.TryParse(value, System.Globalization.NumberStyles.Float,
                   System.Globalization.CultureInfo.InvariantCulture, out var gain) && float.IsFinite(gain)
            ? Math.Max(0f, gain)
            : 1f;
    }

    internal static void SetClockForTests(Func<long>? clock) =>
        Volatile.Write(ref _clockForTests, clock);

    internal static void SetHostInputForTests(IHostInput? input) =>
        Volatile.Write(ref _hostInputForTests, input);

    internal static void ResetForTests()
    {
        lock (Gate)
        {
            _sentHapticLeft = _sentHapticRight = 0;
            _sentHapticAt = 0;
        }

        Volatile.Write(ref _hostInputForTests, null);
        Volatile.Write(ref _clockForTests, null);
    }
}
