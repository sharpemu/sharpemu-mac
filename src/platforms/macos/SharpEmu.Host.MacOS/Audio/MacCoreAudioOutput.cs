// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Versioning;

using SharpEmu.HLE.Host;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// Native CoreAudio output for macOS — the SDL audio backend's replacement, so
/// a macOS process never initialises SDL audio.
///
/// Each stream drives one default-output AudioUnit with a render callback, which
/// is the lower-latency of the two AudioToolbox options: an AudioQueue adds its
/// own buffer ring and its own thread on top of the device's, whereas the render
/// callback is pulled directly by the device's I/O cycle (512 frames, ~10.7 ms at
/// 48 kHz by default). The unit's subtype is kAudioUnitSubType_DefaultOutput,
/// which AUComponent.h describes as tracking the user's default-device
/// selection, so switching output device while a title runs keeps playing
/// without this backend having to re-point anything.
/// </summary>
[SupportedOSPlatform("macos")]
internal sealed class MacCoreAudioOutput : IHostPcmAudioOutput
{
    /// <summary>
    /// Cap for streams this class paces itself (AudioOut). Blocking the guest in
    /// <see cref="IHostAudioStream.Submit"/> is that path's only pacing, so the
    /// device settles at this depth — it is the playback latency, and the floor
    /// under it is how much jitter the stream can absorb before it runs dry.
    /// Same default and same environment override as the SDL backend so the two
    /// behave identically.
    /// </summary>
    internal static readonly int TargetQueuedMilliseconds =
        int.TryParse(
            Environment.GetEnvironmentVariable("SHARPEMU_AUDIO_LATENCY_MS"),
            out var latencyMs) && latencyMs > 0
            ? latencyMs
            : 60;

    public string BackendName => "coreaudio";

    /// <summary>
    /// Stereo PCM16 stream with a caller-chosen backpressure cap. Callers that
    /// pace the guest themselves (AudioOut2, the FFmpeg decoder) pass a deeper
    /// cap so this class's backpressure does not fight their pacing.
    /// </summary>
    public IHostAudioStream OpenStereoPcm16Stream(uint sampleRate, int maxQueuedPcmBytes = 32 * 1024)
        => OpenStream(
            sampleRate,
            channels: 2,
            HostPcmFormat.Signed16,
            maxQueuedPcmBytes > 0 ? maxQueuedPcmBytes : 32 * 1024);

    /// <summary>
    /// Guest-format stream for AudioOut, which has no queue model of its own:
    /// blocking here is that path's only pacing, so the device settles at
    /// <see cref="TargetQueuedMilliseconds"/> and that depth is the playback
    /// latency.
    /// </summary>
    public IHostAudioStream OpenPcmStream(uint sampleRate, int channels, HostPcmFormat format)
        => OpenStream(sampleRate, channels, format, LatencyCapBytes(sampleRate, channels, format));

    /// <summary>
    /// Guest-format stream with a caller-chosen backpressure cap in guest
    /// bytes, for callers that pace the guest themselves (AudioOut2's context
    /// mix): its device cushion, dropout reserve and producer naps read
    /// <see cref="IHostAudioStream.QueuedMilliseconds"/> against a queue this
    /// deep, so the cap must not fall back to the
    /// <see cref="TargetQueuedMilliseconds"/> one. Zero or less keeps that
    /// default, as the SDL backend does.
    /// </summary>
    public IHostAudioStream OpenPcmStream(uint sampleRate, int channels, HostPcmFormat format, int maxQueuedBytes)
        => maxQueuedBytes > 0
            ? OpenStream(sampleRate, channels, format, maxQueuedBytes)
            : OpenPcmStream(sampleRate, channels, format);

    /// <summary>
    /// Queue cap in guest bytes for <see cref="TargetQueuedMilliseconds"/> of
    /// audio in the guest's own format.
    /// </summary>
    internal static int LatencyCapBytes(uint sampleRate, int channels, HostPcmFormat format) =>
        checked((int)((long)sampleRate * MacAudioSampleConverter.SourceFrameSize(channels, format) *
                      TargetQueuedMilliseconds / 1_000));

    private static IHostAudioStream OpenStream(
        uint sampleRate,
        int channels,
        HostPcmFormat format,
        int maximumQueuedBytes)
    {
        if (sampleRate is < 8_000 or > 384_000 ||
            channels is < 1 or > MacAudioChannelMap.MaxChannels)
        {
            throw new ArgumentOutOfRangeException(
                sampleRate is < 8_000 or > 384_000 ? nameof(sampleRate) : nameof(channels));
        }

        return new MacCoreAudioStream(sampleRate, channels, format, maximumQueuedBytes);
    }
}
