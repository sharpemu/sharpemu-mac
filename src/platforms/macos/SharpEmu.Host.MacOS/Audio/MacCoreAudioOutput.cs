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
    /// SHARPEMU_DUALSENSE_AUDIO=0 keeps the controller-less paths (the pad
    /// speaker on the speakers, vibration as rumble) even with a DualSense.
    /// </summary>
    internal static readonly bool ControllerAudioEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_DUALSENSE_AUDIO") != "0";

    /// <summary>
    /// The DualSense's own output, as upstream's SDL host opens it: on USB its
    /// four-channel CoreAudio device (speaker pair, then the two actuators),
    /// on Bluetooth the audio output report with the actuator samples and the
    /// speaker's Opus packets. Exactly one pad on the transport and, on USB,
    /// exactly one matching device, or null: nothing ties a GameController pad
    /// to its audio device, so anything else gives no stream rather than a
    /// guess, and the callers keep their controller-less paths. While the
    /// stream is open the pad's speaker plays the front pair (restored on
    /// close). The stream drops its oldest audio past the cap instead of
    /// blocking and does not report the guest audio clock.
    /// </summary>
    public IHostAudioStream? TryOpenControllerStream(uint sampleRate, int maxQueuedBytes, out int channels, bool routeSpeaker = true)
    {
        channels = 0;
        if (!ControllerAudioEnabled)
        {
            return null;
        }

        IHostAudioStream stream;
        string outputName;
        MacDualSenseHid? usbPad = null;
        MacAudioDeviceInfo device;
        if (MacCoreAudioDevices.TrySelectControllerDevice(MacCoreAudioDevices.ListOutputDevices(), out device))
        {
            // The device exists only while the pad is wired. Its HID interface carries the speaker routing; without
            // it (another process holds the pad exclusively) the audio still plays, unrouted.
            usbPad = MacDualSenseHid.TryOpenSingle(MacDualSenseTransport.Usb);
            try
            {
                stream = new MacCoreAudioStream(
                    sampleRate,
                    MacCoreAudioDevices.ControllerChannels,
                    HostPcmFormat.Float32,
                    maxQueuedBytes > 0
                        ? maxQueuedBytes
                        : LatencyCapBytes(sampleRate, MacCoreAudioDevices.ControllerChannels, HostPcmFormat.Float32),
                    new MacAudioStreamTarget(device.Id, ReportsClock: false, DropOldest: true, RawChannels: true));
            }
            catch
            {
                usbPad?.Dispose();
                throw;
            }

            outputName = $"\"{device.Name}\"";
        }
        else
        {
            device = default;
            if (MacDualSenseBluetooth.TryOpen(sampleRate, maxQueuedBytes, routeSpeaker) is not { } bluetooth)
            {
                return null;
            }

            stream = bluetooth;
            outputName = "DualSense over Bluetooth";
        }

        var route = new ControllerRoute(device.Id != 0, usbPad);
        var routed = routeSpeaker && route.Send(MacDualSenseReports.AudioRoute(speaker: true));
        channels = MacCoreAudioDevices.ControllerChannels;
        Console.Error.WriteLine(
            $"[LOADER][INFO] Controller audio: {outputName} (speaker {(routed ? "routed" : "not routed")})");
        return new ControllerAudioStream(stream, route, routed);
    }

    /// <summary>
    /// Where the routing effects go: the wired pad's HID interface (none when it would not open), or the Bluetooth
    /// link's.
    /// </summary>
    private sealed class ControllerRoute(bool wired, MacDualSenseHid? usbPad) : IDisposable
    {
        public bool Send(ReadOnlySpan<byte> effects) =>
            wired ? usbPad?.TrySendEffects(effects) == true : MacDualSenseBluetooth.TrySendEffects(effects);

        public void Dispose() => usbPad?.Dispose();
    }

    private sealed class ControllerAudioStream(IHostAudioStream inner, ControllerRoute route, bool speakerRouted)
        : IHostAudioStream
    {
        private int _disposed;

        public int QueuedMilliseconds => inner.QueuedMilliseconds;

        public bool Submit(ReadOnlySpan<byte> pcm) => inner.Submit(pcm);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            // Restore the route while the Bluetooth link is still open.
            if (speakerRouted)
            {
                route.Send(MacDualSenseReports.AudioRoute(speaker: false));
            }

            inner.Dispose();
            route.Dispose();
        }
    }

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
