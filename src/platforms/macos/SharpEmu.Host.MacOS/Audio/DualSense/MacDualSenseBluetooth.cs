// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Concentus;
using Concentus.Enums;
using SharpEmu.HLE.Host;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// The controller stream of a DualSense on Bluetooth, which lists no audio device: the actuators' samples (3 kHz, a
/// signed byte each) and the speaker's audio (two 200-byte Opus packets) go in output report 0x39, one report every
/// 64 actuator frames (21.3 ms), as upstream's SdlDualSenseBluetooth sends them. One sender thread serves every open
/// stream: the AudioOut and AudioOut2 controller streams mix into the same report.
/// </summary>
[SupportedOSPlatform("macos")]
internal sealed class MacDualSenseBluetooth
{
    internal const int InputChannels = 4;
    internal const int HapticRate = 3000;
    internal const int SpeakerRate = 45000;
    internal const int OpusSampleRate = 48000;
    internal const int OpusFrames = 480;
    internal const int SpeakerFrames = MacDualSenseReports.OpusPackets * OpusFrames;
    private const int ReopenDelayMilliseconds = 1000;
    private static readonly long PeriodTicks = Stopwatch.Frequency * MacDualSenseReports.HapticFrames / HapticRate;

    private static readonly object RegistryGate = new();
    private static MacDualSenseBluetooth? _shared;

    private readonly object _gate = new();
    private readonly object _deviceGate = new();
    private readonly List<Client> _clients = [];
    private readonly AutoResetEvent _signal = new(false);
    private readonly Thread _sender;
    private readonly MacDualSenseOpus _opus = new();
    private MacDualSenseHid? _device;
    private bool _stopping;
    private byte _sequence;
    private byte _counter;
    private long _reopenAt;
    private bool _writeFailureLogged;

    private MacDualSenseBluetooth(MacDualSenseHid device)
    {
        _device = device;
        _sender = new Thread(Run)
        {
            IsBackground = true,
            Name = "DualSense Bluetooth audio",
            Priority = ThreadPriority.AboveNormal,
        };
        _sender.Start();
    }

    /// <summary>A stream on the one DualSense connected over Bluetooth, or null when there is none (or several).</summary>
    public static IHostAudioStream? TryOpen(uint sampleRate, int maxQueuedBytes, bool speaker)
    {
        lock (RegistryGate)
        {
            var link = _shared;
            if (link is null)
            {
                var device = MacDualSenseHid.TryOpenSingle(MacDualSenseTransport.Bluetooth);
                if (device is null)
                {
                    return null;
                }

                link = new MacDualSenseBluetooth(device);
            }

            _shared = link;
            try
            {
                return link.Attach(sampleRate, maxQueuedBytes, speaker);
            }
            catch
            {
                if (link.ClientCount == 0)
                {
                    link.Shutdown();
                }

                throw;
            }
        }
    }

    /// <summary>The effects report (the speaker routing) on the link's device, serialized with the audio reports.</summary>
    public static bool TrySendEffects(ReadOnlySpan<byte> effects)
    {
        MacDualSenseBluetooth? link;
        lock (RegistryGate)
        {
            link = _shared;
        }

        if (link is null)
        {
            return false;
        }

        lock (link._deviceGate)
        {
            return link._device?.TrySendEffects(effects) == true;
        }
    }

    private int ClientCount
    {
        get
        {
            lock (_gate)
            {
                return _clients.Count;
            }
        }
    }

    private Client Attach(uint sampleRate, int maxQueuedBytes, bool speaker)
    {
        var queueMilliseconds = maxQueuedBytes * 1000.0 / ((double)sampleRate * InputChannels * sizeof(float));
        var client = new Client(this, sampleRate, queueMilliseconds, speaker);
        lock (_gate)
        {
            _clients.Add(client);
        }

        return client;
    }

    private void Detach(Client client)
    {
        lock (RegistryGate)
        {
            lock (_gate)
            {
                _clients.Remove(client);
                if (_clients.Count > 0)
                {
                    return;
                }
            }

            Shutdown();
        }
    }

    // Caller holds RegistryGate.
    private void Shutdown()
    {
        lock (_gate)
        {
            _stopping = true;
        }

        if (ReferenceEquals(_shared, this))
        {
            _shared = null;
        }

        _signal.Set();
        _sender.Join();
        lock (_deviceGate)
        {
            _device?.Dispose();
            _device = null;
        }
    }

    private void Run()
    {
        var report = new byte[MacDualSenseReports.BluetoothAudioReportLength];
        var haptics = new float[MacDualSenseReports.HapticFrames * 2];
        var speaker = new float[SpeakerFrames * 2];
        var packets = new byte[MacDualSenseReports.OpusPackets * MacDualSenseReports.OpusPacketBytes];
        var next = 0L;
        while (true)
        {
            bool active;
            bool speakerOpen;
            lock (_gate)
            {
                if (_stopping)
                {
                    return;
                }

                Array.Clear(haptics);
                Array.Clear(speaker);
                active = false;
                speakerOpen = false;
                foreach (var client in _clients)
                {
                    active |= client.Haptics.MixInto(haptics);
                    if (client.Speaker is { } lane)
                    {
                        speakerOpen = true;
                        active |= lane.MixInto(speaker);
                    }
                }
            }

            if (!active)
            {
                next = 0;
                _signal.WaitOne(250);
                continue;
            }

            var encoded = speakerOpen && _opus.TryEncode(speaker, packets);
            _counter += 2;
            MacDualSenseReports.BuildBluetoothAudioReport(
                report,
                _sequence,
                _counter,
                haptics,
                encoded ? packets : ReadOnlySpan<byte>.Empty);
            _sequence = (byte)((_sequence + 1) & 0x0F);
            Send(report);

            var now = Stopwatch.GetTimestamp();
            next = (next == 0 ? now : next) + PeriodTicks;
            if (now - next > PeriodTicks)
            {
                next = now;
            }

            var wait = next - Stopwatch.GetTimestamp();
            if (wait > 0)
            {
                Thread.Sleep(TimeSpan.FromTicks(wait * TimeSpan.TicksPerSecond / Stopwatch.Frequency));
            }
        }
    }

    private void Send(byte[] report)
    {
        lock (_deviceGate)
        {
            if (_device is null)
            {
                if (Stopwatch.GetTimestamp() < _reopenAt)
                {
                    return;
                }

                _device = MacDualSenseHid.TryOpenSingle(MacDualSenseTransport.Bluetooth);
                if (_device is null)
                {
                    _reopenAt = Stopwatch.GetTimestamp() + Stopwatch.Frequency * ReopenDelayMilliseconds / 1000;
                    return;
                }
            }

            if (_device.TrySendOutputReport(report, out var status))
            {
                return;
            }

            if (!_writeFailureLogged)
            {
                _writeFailureLogged = true;
                Console.Error.WriteLine(
                    $"[LOADER][WARN] DualSense Bluetooth audio report refused: IOReturn 0x{status:X8}");
            }

            _device.Dispose();
            _device = null;
            _reopenAt = Stopwatch.GetTimestamp() + Stopwatch.Frequency * ReopenDelayMilliseconds / 1000;
        }
    }

    /// <summary>
    /// One controller stream: four float channels per frame, the speaker pair first and the actuator pair last, as on
    /// a wired DualSense's audio device.
    /// </summary>
    internal sealed class Client : IHostAudioStream
    {
        private readonly MacDualSenseBluetooth? _link;
        private int _disposed;

        internal Client(MacDualSenseBluetooth? link, uint sampleRate, double queueMilliseconds, bool speaker)
        {
            _link = link;
            Haptics = new MacDualSenseLane(sampleRate, HapticRate, MacDualSenseReports.HapticFrames, queueMilliseconds);
            Speaker = speaker ? new MacDualSenseLane(sampleRate, SpeakerRate, SpeakerFrames, queueMilliseconds) : null;
        }

        public MacDualSenseLane Haptics { get; }

        public MacDualSenseLane? Speaker { get; }

        public int QueuedMilliseconds => Volatile.Read(ref _disposed) != 0 ? -1 : Haptics.QueuedMilliseconds;

        public bool Submit(ReadOnlySpan<byte> pcm)
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                return false;
            }

            var samples = MemoryMarshal.Cast<byte, float>(pcm);
            var frames = samples.Length / InputChannels;
            if (frames == 0)
            {
                return true;
            }

            var pair = ArrayPool<float>.Shared.Rent(frames * 2);
            try
            {
                var stereo = pair.AsSpan(0, frames * 2);
                if (Speaker is not null)
                {
                    CopyPair(samples, stereo, 0);
                    Speaker.Put(stereo);
                }

                CopyPair(samples, stereo, 2);
                Haptics.Put(stereo);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(pair);
            }

            _link?._signal.Set();
            return true;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                _link?.Detach(this);
            }
        }

        /// <summary>One pair of the four channels, starting at <paramref name="firstChannel"/>, as stereo.</summary>
        internal static void CopyPair(ReadOnlySpan<float> samples, Span<float> stereo, int firstChannel)
        {
            for (var frame = 0; frame < stereo.Length / 2; frame++)
            {
                stereo[2 * frame] = samples[(frame * InputChannels) + firstChannel];
                stereo[(2 * frame) + 1] = samples[(frame * InputChannels) + firstChannel + 1];
            }
        }
    }
}

/// <summary>
/// The DualSense speaker's Opus encoder: 48 kHz stereo, restricted low delay, constant 200-byte packets of 480
/// frames, complexity 0, as upstream configures it. Concentus is fully managed.
/// </summary>
internal sealed class MacDualSenseOpus
{
    private IOpusEncoder? _encoder;
    private bool _failureLogged;

    /// <summary>
    /// Encodes <see cref="MacDualSenseBluetooth.SpeakerFrames"/> stereo frames into two packets of exactly
    /// <see cref="MacDualSenseReports.OpusPacketBytes"/>; false (and a cleared destination) when either is not.
    /// </summary>
    public bool TryEncode(float[] speaker, Span<byte> destination)
    {
        for (var index = 0; index < speaker.Length; index++)
        {
            speaker[index] = float.IsFinite(speaker[index]) ? Math.Clamp(speaker[index], -1f, 1f) : 0f;
        }

        try
        {
            if (_encoder is null)
            {
                OpusCodecFactory.AttemptToUseNativeLibrary = false;
                var encoder = OpusCodecFactory.CreateEncoder(
                    MacDualSenseBluetooth.OpusSampleRate,
                    2,
                    OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);
                encoder.Bitrate = MacDualSenseReports.OpusPacketBytes * 8 * MacDualSenseBluetooth.OpusSampleRate /
                                  MacDualSenseBluetooth.OpusFrames;
                encoder.UseVBR = false;
                encoder.Complexity = 0;
                _encoder = encoder;
            }

            const int frames = MacDualSenseBluetooth.OpusFrames;
            const int bytes = MacDualSenseReports.OpusPacketBytes;
            for (var packet = 0; packet < MacDualSenseReports.OpusPackets; packet++)
            {
                var encoded = _encoder.Encode(
                    speaker.AsSpan(packet * frames * 2, frames * 2),
                    frames,
                    destination.Slice(packet * bytes, bytes),
                    bytes);
                if (encoded != bytes)
                {
                    destination.Clear();
                    return false;
                }
            }

            return true;
        }
        catch (Exception exception)
        {
            destination.Clear();
            if (!_failureLogged)
            {
                _failureLogged = true;
                Console.Error.WriteLine($"[LOADER][WARN] DualSense Bluetooth speaker encoding failed: {exception.Message}");
            }

            return false;
        }
    }
}

/// <summary>
/// One pair of the controller stream at the rate its report wants: resampled from the guest's rate, held up to the
/// stream's queue limit (the oldest audio goes past it, as the stream never blocks the guest), and handed out one
/// report at a time once 1.5 reports are queued, so a report is not sent half empty at the start.
/// </summary>
internal sealed class MacDualSenseLane
{
    private readonly object _gate = new();
    private readonly int _outputRate;
    private readonly int _primeFrames;
    private readonly int _capacityFrames;
    private readonly double _step;
    private readonly float[] _queue;
    private int _head;
    private int _count;
    private bool _primed;

    // Area-averaging resampler state: the input still owed to the output frame being built.
    private double _need;
    private double _left;
    private double _right;

    public MacDualSenseLane(uint inputRate, int outputRate, int reportFrames, double queueMilliseconds)
    {
        if (inputRate == 0 || outputRate <= 0 || reportFrames <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputRate));
        }

        _outputRate = outputRate;
        _primeFrames = reportFrames * 3 / 2;
        _capacityFrames = Math.Max(2 * reportFrames, (int)(outputRate * queueMilliseconds / 1000));
        _queue = new float[_capacityFrames * 2];
        _step = (double)inputRate / outputRate;
        _need = _step;
    }

    public int QueuedFrames
    {
        get
        {
            lock (_gate)
            {
                return _count;
            }
        }
    }

    public int QueuedMilliseconds => (int)(QueuedFrames * 1000L / _outputRate);

    /// <summary>
    /// Appends guest-rate stereo audio. Each output frame is the mean of the input over its own interval, so the
    /// 16:1 step down to the actuators' 3 kHz filters what they cannot play instead of folding it back.
    /// </summary>
    public void Put(ReadOnlySpan<float> stereo)
    {
        lock (_gate)
        {
            for (var frame = 0; frame < stereo.Length / 2; frame++)
            {
                double left = stereo[2 * frame];
                double right = stereo[(2 * frame) + 1];
                if (!double.IsFinite(left))
                {
                    left = 0;
                }

                if (!double.IsFinite(right))
                {
                    right = 0;
                }

                var available = 1.0;
                while (available > 1e-12)
                {
                    var take = Math.Min(available, _need);
                    _left += left * take;
                    _right += right * take;
                    available -= take;
                    _need -= take;
                    if (_need <= 1e-12)
                    {
                        Enqueue((float)(_left / _step), (float)(_right / _step));
                        _left = 0;
                        _right = 0;
                        _need = _step;
                    }
                }
            }
        }
    }

    /// <summary>Adds the next report's frames into <paramref name="destination"/>; false when it had none to give.</summary>
    public bool MixInto(Span<float> destination)
    {
        lock (_gate)
        {
            if (!_primed && _count < _primeFrames)
            {
                return false;
            }

            var wanted = destination.Length / 2;
            var frames = Math.Min(wanted, _count);
            _primed = frames == wanted;
            if (frames == 0)
            {
                return false;
            }

            for (var frame = 0; frame < frames; frame++)
            {
                destination[2 * frame] += _queue[2 * _head];
                destination[(2 * frame) + 1] += _queue[(2 * _head) + 1];
                _head = (_head + 1) % _capacityFrames;
            }

            _count -= frames;
            return true;
        }
    }

    // Caller holds _gate. A full queue drops its oldest frame.
    private void Enqueue(float left, float right)
    {
        if (_count == _capacityFrames)
        {
            _head = (_head + 1) % _capacityFrames;
            _count--;
        }

        var tail = (_head + _count) % _capacityFrames;
        _queue[2 * tail] = left;
        _queue[(2 * tail) + 1] = right;
        _count++;
    }
}
