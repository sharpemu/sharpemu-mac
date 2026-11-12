// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using SharpEmu.HLE.Host;
using static SharpEmu.Host.MacOS.MacCoreAudioInterop;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// One open CoreAudio output stream: a default-output AudioUnit pulling
/// interleaved float frames out of a lock-free ring through its render callback.
///
/// The guest's submitting thread converts and folds channels into the ring; the
/// render callback only copies and, when the ring runs dry, writes silence and
/// marks the buffer silent. Submit paces the guest by waiting for the ring to
/// drain under its cap, exactly as the SDL backend waits on
/// SDL_GetAudioStreamQueued: the same 250 ms ceiling, the same drain-estimate
/// sleep, and the same decision to enqueue an over-deep buffer rather than drop
/// it, because a gap in the stream is an audible click and a few extra
/// milliseconds of latency is not.
/// </summary>
[SupportedOSPlatform("macos")]
internal sealed unsafe class MacCoreAudioStream : IHostAudioStream
{
    private const int MaximumWaitMilliseconds = 250;

    /// <summary>Frames the unit may ask for in one render callback.</summary>
    private const uint MaximumFramesPerSlice = 4096;

    private static readonly bool TraceQueue = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_LOG_AUDIO_QUEUE"),
        "1",
        StringComparison.Ordinal);

    private static int _nextStreamId;

    private readonly object _gate = new();
    private readonly int _streamId = Interlocked.Increment(ref _nextStreamId);
    private readonly uint _sampleRate;
    private readonly int _guestChannels;
    private readonly int _guestBytesPerFrame;
    private readonly HostPcmFormat _format;
    private readonly int _renderChannels;
    private readonly float[]? _matrix;
    private readonly long _maximumQueuedFrames;
    private readonly MacAudioRingBuffer? _ring;

    private GCHandle _selfHandle;
    private nint _unit;
    private bool _initialized;
    private bool _running;
    private volatile bool _disposed;
    private float[] _scratch = [];
    private long _totalSubmittedFrames;

    // Written by the render thread, read for diagnostics only.
    private long _underrunFrames;

    // Queue diagnostics for the current report window.
    private long _windowStart = Stopwatch.GetTimestamp();
    private long _submissions;
    private long _submittedFrames;
    private long _blockedTicks;
    private long _overruns;
    private long _emptyObservations;
    private long _minQueuedFrames = long.MaxValue;
    private long _maxQueuedFrames;
    private long _queuedFrameSum;

    internal MacCoreAudioStream(
        uint sampleRate,
        int channels,
        HostPcmFormat format,
        int maximumQueuedBytes)
    {
        _sampleRate = sampleRate;
        _guestChannels = channels;
        _format = format;
        _guestBytesPerFrame = MacAudioSampleConverter.SourceFrameSize(channels, format);
        _maximumQueuedFrames = Math.Max(1, maximumQueuedBytes / _guestBytesPerFrame);

        var description = new AudioComponentDescription
        {
            ComponentType = AudioUnitTypeOutput,
            ComponentSubType = AudioUnitSubTypeDefaultOutput,
            ComponentManufacturer = AudioUnitManufacturerApple,
        };

        var component = AudioComponentFindNext(0, &description);
        if (component == 0)
        {
            throw new InvalidOperationException(
                "CoreAudio has no default output audio unit.");
        }

        Check(AudioComponentInstanceNew(component, out _unit), "AudioComponentInstanceNew");
        try
        {
            // The device's own format tells us how many outputs it has. CoreAudio
            // documents no downmix for a client format wider than the device, so
            // the stream renders at the device's channel count and the fold from
            // the guest's layout happens in MacAudioChannelMap.
            _renderChannels = QueryDeviceChannels();
            _matrix = MacAudioChannelMap.Build(_guestChannels, _renderChannels);

            // The client format keeps the guest's sample rate: the output unit
            // resamples to the device rate internally, so a 44.1 kHz title on a
            // 48 kHz Mac needs no resampler here.
            var clientFormat = new AudioStreamBasicDescription
            {
                SampleRate = sampleRate,
                FormatId = AudioFormatLinearPcm,
                FormatFlags = FormatFlagIsFloat | FormatFlagIsPacked,
                FramesPerPacket = 1,
                ChannelsPerFrame = (uint)_renderChannels,
                BitsPerChannel = 32,
                BytesPerFrame = (uint)(_renderChannels * sizeof(float)),
                BytesPerPacket = (uint)(_renderChannels * sizeof(float)),
            };
            Check(
                AudioUnitSetProperty(
                    _unit,
                    PropertyStreamFormat,
                    ScopeInput,
                    0,
                    &clientFormat,
                    (uint)sizeof(AudioStreamBasicDescription)),
                "AudioUnitSetProperty(StreamFormat)");

            TrySetChannelLayout();

            var maximumFrames = MaximumFramesPerSlice;
            // Advisory (TN2321): only strictly needed when the device buffer is
            // enlarged, and a device that refuses it still renders fine.
            _ = AudioUnitSetProperty(
                _unit,
                PropertyMaximumFramesPerSlice,
                ScopeGlobal,
                0,
                &maximumFrames,
                sizeof(uint));

            _ring = new MacAudioRingBuffer(_renderChannels, CapacityFrames());
            _selfHandle = GCHandle.Alloc(this);
            var callback = new AURenderCallbackStruct
            {
                InputProc = &Render,
                InputProcRefCon = GCHandle.ToIntPtr(_selfHandle),
            };
            Check(
                AudioUnitSetProperty(
                    _unit,
                    PropertySetRenderCallback,
                    ScopeInput,
                    0,
                    &callback,
                    (uint)sizeof(AURenderCallbackStruct)),
                "AudioUnitSetProperty(SetRenderCallback)");

            Check(AudioUnitInitialize(_unit), "AudioUnitInitialize");
            _initialized = true;

            // Start immediately rather than on the first submission: the callback
            // emits silence until the guest feeds it, and a unit that is already
            // running makes the first submission's latency the same as every
            // later one's.
            Check(AudioOutputUnitStart(_unit), "AudioOutputUnitStart");
            _running = true;
        }
        catch
        {
            Release();
            throw;
        }
    }

    public int QueuedMilliseconds
    {
        get
        {
            // Deliberately lock-free: the pacing loops in AudioOut2 poll this
            // while another thread may be blocked inside Submit, and the ring's
            // counters are safe to read from anywhere.
            if (_disposed || _unit == 0 || _ring is null)
            {
                return -1;
            }

            return (int)(_ring.QueuedFrames * 1_000 / _sampleRate);
        }
    }

    public bool Submit(ReadOnlySpan<byte> pcm)
    {
        if (pcm.IsEmpty)
        {
            return true;
        }

        lock (_gate)
        {
            if (_disposed || _unit == 0 || _ring is null)
            {
                return false;
            }

            var frames = pcm.Length / _guestBytesPerFrame;
            if (frames <= 0)
            {
                return true;
            }

            var blockStart = Stopwatch.GetTimestamp();
            var deadline = blockStart + (Stopwatch.Frequency * MaximumWaitMilliseconds / 1_000);
            var framesPerMillisecond = _sampleRate / 1_000.0;
            long queued;
            var overrun = false;
            while ((queued = _ring.QueuedFrames) > _maximumQueuedFrames)
            {
                if (Stopwatch.GetTimestamp() >= deadline)
                {
                    // Enqueue anyway rather than discarding the buffer; the queue
                    // recovers as soon as the device drains back under the cap.
                    overrun = true;
                    break;
                }

                // Sleep until the device should have drained to the cap rather
                // than polling every millisecond: this wait paces every audio
                // grain, and 1 ms polls cost several wakeups per grain.
                var drainMilliseconds = (queued - _maximumQueuedFrames) / framesPerMillisecond;
                var remainingMilliseconds =
                    (deadline - Stopwatch.GetTimestamp()) * 1_000.0 / Stopwatch.Frequency;
                HostBlockedTime.Sleep((int)Math.Clamp(
                    Math.Ceiling(Math.Min(drainMilliseconds, remainingMilliseconds)),
                    1,
                    MaximumWaitMilliseconds));
            }

            RecordSubmission(queued, blockStart, overrun, frames);

            var samples = MacAudioSampleConverter.RequiredSamples(frames, _renderChannels);
            if (_scratch.Length < samples)
            {
                // Grown once per stream in practice: AudioOut submits a fixed
                // grain, so this does not allocate on the steady-state path.
                _scratch = new float[samples];
            }

            var converted = MacAudioSampleConverter.Convert(
                pcm,
                _scratch.AsSpan(0, samples),
                _guestChannels,
                _renderChannels,
                _format,
                _matrix);
            if (converted <= 0)
            {
                return true;
            }

            WriteToRing(_scratch.AsSpan(0, converted * _renderChannels));

            // Everything handed over minus what the device still holds is what
            // the player has actually heard; video decoded on the host follows
            // this clock rather than wall clock.
            GuestAudioClock.Report(
                Math.Max(0, _totalSubmittedFrames - queued) / (double)_sampleRate);
            _totalSubmittedFrames += converted;
            return true;
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            Release();
        }
    }

    /// <summary>
    /// Pulls one device buffer's worth of frames out of the ring. Called from
    /// CoreAudio's real-time thread, so it takes no lock and allocates nothing;
    /// the caller fills whatever is missing with silence.
    /// </summary>
    private int ReadFrames(float* destination, int frames)
    {
        var filled = _ring is null ? 0 : _ring.Read(destination, frames);
        if (filled < frames)
        {
            _underrunFrames += frames - filled;
        }

        return filled;
    }

    /// <summary>
    /// The render callback. A native thread enters managed code here, so it must
    /// not allocate, must not take a lock the guest thread can hold, and must
    /// not throw.
    /// </summary>
    [UnmanagedCallersOnly]
    private static int Render(
        nint refCon,
        uint* actionFlags,
        nint timeStamp,
        uint busNumber,
        uint frames,
        AudioBufferList* data)
    {
        if (data is null)
        {
            return 0;
        }

        // The handle is freed only after the unit has been stopped and
        // uninitialised, which is CoreAudio's guarantee that this callback is no
        // longer in flight, so the target is always alive here.
        var stream = refCon == 0
            ? null
            : GCHandle.FromIntPtr(refCon).Target as MacCoreAudioStream;

        var silent = true;
        var buffers = &data->Buffers;
        for (var index = 0u; index < data->NumberBuffers; index++)
        {
            var buffer = buffers + index;
            var samples = (float*)buffer->Data;
            if (samples is null)
            {
                continue;
            }

            var totalSamples = (int)(buffer->DataByteSize / sizeof(float));
            var channels = stream?._renderChannels ?? 0;
            var filled = channels > 0
                ? stream!.ReadFrames(samples, totalSamples / channels) * channels
                : 0;
            if (filled > 0)
            {
                silent = false;
            }

            for (var sample = filled; sample < totalSamples; sample++)
            {
                samples[sample] = 0.0f;
            }
        }

        if (silent && actionFlags is not null)
        {
            *actionFlags |= RenderActionOutputIsSilence;
        }

        return 0;
    }

    /// <summary>
    /// The device's output channel count, or stereo when the device cannot be
    /// asked — a device that disappears between opening the unit and this query
    /// must not take the port down with it.
    /// </summary>
    private int QueryDeviceChannels()
    {
        var deviceFormat = default(AudioStreamBasicDescription);
        var size = (uint)sizeof(AudioStreamBasicDescription);
        if (AudioUnitGetProperty(
                _unit,
                PropertyStreamFormat,
                ScopeOutput,
                0,
                &deviceFormat,
                &size) != 0 ||
            deviceFormat.ChannelsPerFrame == 0)
        {
            return 2;
        }

        // Cap the width: a 64-output interface would otherwise size the ring and
        // every conversion by channels the guest never fills.
        return (int)Math.Min(deviceFormat.ChannelsPerFrame, 16);
    }

    /// <summary>
    /// Tells the unit the render format is WAV order (FL FR FC LFE BL BR SL SR)
    /// using explicit channel labels, so a device whose own order differs can
    /// place the channels correctly. Advisory: a device that rejects the layout
    /// still gets correctly folded audio from the gain matrix.
    /// </summary>
    private void TrySetChannelLayout()
    {
        var labels = MacAudioChannelMap.ChannelLabels(_renderChannels);
        var size = sizeof(AudioChannelLayoutHeader) +
                   (labels.Length * sizeof(AudioChannelDescription));
        var blob = stackalloc byte[size];
        new Span<byte>(blob, size).Clear();
        var header = (AudioChannelLayoutHeader*)blob;
        header->ChannelLayoutTag = ChannelLayoutTagUseChannelDescriptions;
        header->NumberChannelDescriptions = (uint)labels.Length;
        var descriptions = (AudioChannelDescription*)(blob + sizeof(AudioChannelLayoutHeader));
        for (var index = 0; index < labels.Length; index++)
        {
            descriptions[index].ChannelLabel = labels[index];
        }

        _ = AudioUnitSetProperty(
            _unit,
            PropertyAudioChannelLayout,
            ScopeInput,
            0,
            blob,
            (uint)size);
    }

    /// <summary>
    /// Ring size: twice the backpressure cap plus a quarter second, so the
    /// over-deep submission Submit allows after a 250 ms wait always fits and no
    /// audio is ever dropped for want of space.
    /// </summary>
    private int CapacityFrames()
    {
        var frames = (_maximumQueuedFrames * 2) + (_sampleRate / 4);
        var ceiling = int.MaxValue / (_renderChannels * sizeof(float));
        return (int)Math.Clamp(frames, 4096, ceiling);
    }

    private void WriteToRing(ReadOnlySpan<float> samples)
    {
        if (_ring is null)
        {
            return;
        }

        var written = _ring.Write(samples);
        if (written * _renderChannels >= samples.Length)
        {
            return;
        }

        // Only reachable if a single submission is deeper than the whole ring.
        // Wait for the device to make room rather than tearing a hole in the
        // stream, then give up and drop the tail instead of blocking forever.
        var deadline = Stopwatch.GetTimestamp() +
                       (Stopwatch.Frequency * MaximumWaitMilliseconds / 1_000);
        var offset = written * _renderChannels;
        while (offset < samples.Length && Stopwatch.GetTimestamp() < deadline)
        {
            HostBlockedTime.Sleep(1);
            offset += _ring.Write(samples[offset..]) * _renderChannels;
        }
    }

    /// <summary>
    /// Stops and tears down the unit, then frees what the render callback could
    /// have been touching. Safe to call twice and safe during process shutdown.
    /// </summary>
    private void Release()
    {
        var quiesced = true;
        if (_unit != 0)
        {
            if (_running)
            {
                quiesced &= AudioOutputUnitStop(_unit) == 0;
                _running = false;
            }

            if (_initialized)
            {
                // Uninitialize is the call that guarantees the render callback
                // has returned and will not be entered again.
                quiesced &= AudioUnitUninitialize(_unit) == 0;
                _initialized = false;
            }

            _ = AudioComponentInstanceDispose(_unit);
            _unit = 0;
        }

        if (!quiesced)
        {
            // The unit would not stop cleanly: leak the handle and the ring
            // rather than free memory a live render callback may still read.
            Console.Error.WriteLine(
                $"[LOADER][WARN] CoreAudio stream#{_streamId} did not stop cleanly; " +
                "its buffers are left allocated.");
            return;
        }

        if (_selfHandle.IsAllocated)
        {
            _selfHandle.Free();
        }

        _ring?.Dispose();
    }

    private static void Check(int status, string what)
    {
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"{what} failed with OSStatus {status}.");
        }
    }

    /// <summary>
    /// Samples the queue depth at the moment the guest was allowed to write.
    /// That depth is the playback latency the guest's audio is subject to, so it
    /// is the number to look at when the sound is late; an observed depth of zero
    /// is a genuine underrun, which is what a crackle sounds like. Caller holds
    /// <see cref="_gate"/>.
    /// </summary>
    private void RecordSubmission(long queuedFrames, long blockStart, bool overrun, int frames)
    {
        if (!TraceQueue)
        {
            return;
        }

        var now = Stopwatch.GetTimestamp();
        _submissions++;
        _submittedFrames += frames;
        _blockedTicks += now - blockStart;
        _queuedFrameSum += queuedFrames;
        _minQueuedFrames = Math.Min(_minQueuedFrames, queuedFrames);
        _maxQueuedFrames = Math.Max(_maxQueuedFrames, queuedFrames);
        if (overrun)
        {
            _overruns++;
        }

        if (queuedFrames == 0)
        {
            _emptyObservations++;
        }

        var elapsedTicks = now - _windowStart;
        if (elapsedTicks < Stopwatch.Frequency)
        {
            return;
        }

        _windowStart = now;
        var seconds = elapsedTicks / (double)Stopwatch.Frequency;
        Console.Error.WriteLine(
            $"[PERF][AUDIO] stream#{_streamId} {seconds:F1}s " +
            $"{_guestChannels}ch->{_renderChannels}ch " +
            $"queued_ms min={ToMilliseconds(_minQueuedFrames):F0} " +
            $"avg={ToMilliseconds(_queuedFrameSum / Math.Max(1, _submissions)):F0} " +
            $"max={ToMilliseconds(_maxQueuedFrames):F0} " +
            $"cap={ToMilliseconds(_maximumQueuedFrames):F0} " +
            $"submits/s={_submissions / seconds:F0} " +
            $"fill={_submittedFrames / seconds / _sampleRate * 100.0:F0}% " +
            $"blocked={_blockedTicks * 100.0 / elapsedTicks:F0}% " +
            $"empty={_emptyObservations} overruns={_overruns} " +
            $"underrun_ms={ToMilliseconds(Interlocked.Read(ref _underrunFrames)):F0}");

        _submissions = 0;
        _submittedFrames = 0;
        _blockedTicks = 0;
        _overruns = 0;
        _emptyObservations = 0;
        _minQueuedFrames = long.MaxValue;
        _maxQueuedFrames = 0;
        _queuedFrameSum = 0;
        Interlocked.Exchange(ref _underrunFrames, 0);
    }

    private double ToMilliseconds(long frames) => frames / (double)_sampleRate * 1_000.0;
}
