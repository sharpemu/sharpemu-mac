// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

using SharpEmu.HLE.Host;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// The hand-off between the guest's submitting thread and CoreAudio's render
/// thread: a single-producer/single-consumer ring of interleaved float frames.
///
/// Lock-free by construction, because the render thread is a real-time thread —
/// taking a lock there would let a blocked guest submission stall playback, and
/// a stalled render callback is a dropout. Producer and consumer each own one
/// monotonically increasing frame counter, and the difference between them is
/// the queue depth the stream reports as its latency cushion.
///
/// The samples live in unmanaged memory so the render callback never touches a
/// GC-tracked object while copying.
/// </summary>
internal sealed unsafe class MacAudioRingBuffer : IDisposable
{
    private readonly int _channels;
    private readonly long _capacityFrames;
    private float* _samples;

    // Frames ever written / ever read. Monotonic, so the difference is the
    // depth without any empty-versus-full ambiguity.
    private long _written;
    private long _read;

    // Producer-set read position the consumer skips to before its next read (DiscardOldestBeyond): the oldest frames
    // are dropped on the consumer's side, so the ring stays single-producer, single-consumer.
    private long _discardTo;

    public MacAudioRingBuffer(int channels, int capacityFrames)
    {
        if (channels < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(channels));
        }

        if (capacityFrames < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityFrames));
        }

        _channels = channels;
        _capacityFrames = capacityFrames;
        _samples = (float*)NativeMemory.AllocZeroed(
            (nuint)capacityFrames * (nuint)channels,
            sizeof(float));
    }

    public int Channels => _channels;

    public long CapacityFrames => _capacityFrames;

    /// <summary>Frames handed over and not yet pulled by the device.</summary>
    public long QueuedFrames
    {
        get
        {
            // Read the consumer's counter first: a concurrent advance of either
            // counter can then only make the result look deeper than it is,
            // which errs towards the guest waiting rather than overrunning.
            var read = Math.Max(Volatile.Read(ref _read), Volatile.Read(ref _discardTo));
            return Math.Max(0, Volatile.Read(ref _written) - read);
        }
    }

    // Space the producer can write: frames still awaiting a discard occupy the ring until the consumer skips them.
    public long FreeFrames => _capacityFrames - Math.Max(0, Volatile.Read(ref _written) - Volatile.Read(ref _read));

    /// <summary>
    /// Producer side: drops all but the newest <paramref name="keepFrames"/> queued frames. The consumer skips them on
    /// its next read; <see cref="QueuedFrames"/> no longer counts them from now on. For a stream the guest is not
    /// paced by, so a device stall leaves a gap near one block rather than a growing delay.
    /// </summary>
    public void DiscardOldestBeyond(long keepFrames)
    {
        var target = Volatile.Read(ref _written) - Math.Max(0, keepFrames);
        if (target > Volatile.Read(ref _discardTo))
        {
            Volatile.Write(ref _discardTo, target);
        }
    }

    /// <summary>
    /// Producer side. Copies whole frames and returns how many fit; a short
    /// return means the ring is full and the caller has to wait or drop.
    /// </summary>
    public int Write(ReadOnlySpan<float> interleaved)
    {
        var frames = interleaved.Length / _channels;
        if (frames <= 0 || _samples is null)
        {
            return 0;
        }

        var written = Volatile.Read(ref _written);
        var free = _capacityFrames - Math.Max(0, written - Volatile.Read(ref _read));
        var count = (int)Math.Min(frames, free);
        if (count <= 0)
        {
            return 0;
        }

        var offset = (int)(written % _capacityFrames);
        var firstChunk = (int)Math.Min(count, _capacityFrames - offset);
        interleaved[..(firstChunk * _channels)]
            .CopyTo(new Span<float>(_samples + ((long)offset * _channels), firstChunk * _channels));
        if (count > firstChunk)
        {
            interleaved.Slice(firstChunk * _channels, (count - firstChunk) * _channels)
                .CopyTo(new Span<float>(_samples, (count - firstChunk) * _channels));
        }

        // Publish only after the samples are in place.
        Volatile.Write(ref _written, written + count);
        return count;
    }

    /// <summary>
    /// Consumer side, called from the render callback. Copies up to
    /// <paramref name="frames"/> frames out and returns how many were available;
    /// the caller fills the remainder with silence.
    /// </summary>
    public int Read(float* destination, int frames)
    {
        if (destination is null || frames <= 0 || _samples is null)
        {
            return 0;
        }

        var read = Volatile.Read(ref _read);
        var written = Volatile.Read(ref _written);
        var discardTo = Volatile.Read(ref _discardTo);
        var skipped = false;
        if (discardTo > read)
        {
            read = Math.Min(discardTo, written);
            skipped = true;
        }

        var available = Math.Max(0, written - read);
        var count = (int)Math.Min(frames, available);
        if (count <= 0)
        {
            if (skipped)
            {
                Volatile.Write(ref _read, read);
            }

            return 0;
        }

        var offset = (int)(read % _capacityFrames);
        var firstChunk = (int)Math.Min(count, _capacityFrames - offset);
        var sampleBytes = (nuint)sizeof(float);
        NativeMemory.Copy(
            _samples + ((long)offset * _channels),
            destination,
            (nuint)(firstChunk * _channels) * sampleBytes);
        if (count > firstChunk)
        {
            NativeMemory.Copy(
                _samples,
                destination + ((long)firstChunk * _channels),
                (nuint)((count - firstChunk) * _channels) * sampleBytes);
        }

        Volatile.Write(ref _read, read + count);
        return count;
    }

    /// <summary>Managed-span overload for tests; the device path uses the pointer one.</summary>
    public int Read(Span<float> destination)
    {
        fixed (float* target = destination)
        {
            return Read(target, destination.Length / _channels);
        }
    }

    /// <summary>
    /// Frees the sample storage. Only safe once the owning audio unit has been
    /// stopped and uninitialised, which is the only guarantee CoreAudio gives
    /// that no render callback is still in flight.
    /// </summary>
    public void Dispose()
    {
        var samples = _samples;
        _samples = null;
        if (samples is not null)
        {
            NativeMemory.Free(samples);
        }
    }
}
