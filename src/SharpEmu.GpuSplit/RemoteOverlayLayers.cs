// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using SharpEmu.GpuRemote;
using SharpEmu.Libs.Overlay;

namespace SharpEmu.GpuSplit;

/// <summary>
/// Process B's overlay layers: the frames the guest process painted (<see cref="SplitOverlayLayerSender"/>),
/// put together from their <see cref="GpuRemoteMessage.OverlayLayer"/> pieces, for the presenter
/// (<see cref="OverlayLayerOutput.Source"/>).
/// </summary>
internal sealed class RemoteOverlayLayers : IOverlayLayerSource
{
    private readonly object _gate = new();
    private readonly SortedDictionary<int, OverlayLayerFrame> _frames = new();
    private readonly Action<string> _report;
    private long _version;

    // The layer being received: its pieces arrive back to back on the ring thread.
    private OverlayLayerMessage _pending;
    private byte[]? _pixels;
    private uint _received;

    public RemoteOverlayLayers(Action<string> report)
    {
        _report = report;
    }

    public long Version => Interlocked.Read(ref _version);

    public void Collect(int surfaceWidth, int surfaceHeight, List<OverlayLayerFrame> frames)
    {
        lock (_gate)
        {
            foreach (var frame in _frames.Values)
            {
                // A layer painted for the previous size waits for its repaint at this one.
                if (frame.SurfaceWidth == surfaceWidth && frame.SurfaceHeight == surfaceHeight)
                {
                    frames.Add(frame);
                }
            }
        }
    }

    /// <summary>One <see cref="GpuRemoteMessage.OverlayLayer"/> record, on the ring thread.</summary>
    public void Take(ReadOnlySpan<byte> payload)
    {
        var header = GpuSplitWire.Read<OverlayLayerMessage>(payload);
        var piece = payload[Unsafe.SizeOf<OverlayLayerMessage>()..];
        if (header.Visible == 0)
        {
            _pixels = null;
            lock (_gate)
            {
                if (_frames.Remove((int)header.Id))
                {
                    Interlocked.Increment(ref _version);
                }
            }

            return;
        }

        if ((ulong)header.Width * header.Height * 4 != header.TotalBytes || header.TotalBytes == 0)
        {
            _report($"An overlay layer record's size {header.Width}x{header.Height} does not match its {header.TotalBytes} bytes; the layer is dropped.");
            _pixels = null;
            return;
        }

        if (header.Offset == 0)
        {
            _pending = header;
            _pixels = new byte[header.TotalBytes];
            _received = 0;
        }
        else if (_pixels is null || _pending.Id != header.Id || _pending.Revision != header.Revision ||
                 _pending.TotalBytes != header.TotalBytes || _received != header.Offset)
        {
            _report("An overlay layer piece arrived out of order; the layer is dropped.");
            _pixels = null;
            return;
        }

        if ((ulong)header.Offset + (ulong)piece.Length > header.TotalBytes)
        {
            throw new InvalidDataException("An overlay layer piece reaches past its layer.");
        }

        piece.CopyTo(_pixels!.AsSpan((int)header.Offset));
        _received = header.Offset + (uint)piece.Length;
        if (_received != header.TotalBytes)
        {
            return;
        }

        var frame = new OverlayLayerFrame(
            (int)header.Id, header.Revision, (int)header.SurfaceWidth, (int)header.SurfaceHeight, header.Left,
            header.Top, (int)header.Width, (int)header.Height, _pixels);
        _pixels = null;
        lock (_gate)
        {
            _frames[frame.Id] = frame;
            Interlocked.Increment(ref _version);
        }
    }
}
