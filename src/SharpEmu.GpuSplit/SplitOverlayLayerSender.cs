// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.GpuRemote;
using SharpEmu.Libs.Overlay;

namespace SharpEmu.GpuSplit;

/// <summary>
/// Process A's half of the overlay layers. The layers (the IME keyboard, the save data list) live
/// with the guest's system dialogs in this process, and the presenter in the GPU process: this paints
/// them for the drawable size the GPU process reports (<see cref="GpuRemoteMessage.OverlaySurface"/>)
/// and sends each repaint, and each hide, as <see cref="GpuRemoteMessage.OverlayLayer"/> records.
/// </summary>
/// <remarks>
/// The thread sleeps until a layer changes (<see cref="OverlayLayers.Changed"/>) or the surface
/// does; a layer that did not change is not sent again.
/// </remarks>
internal sealed class SplitOverlayLayerSender : IDisposable
{
    // Under a ring record's payload cap with room for the header, as a host frame's pieces.
    internal const int PieceBytes = 256 * 1024;

    // Wakes now and then even without a change signal, so a missed one costs a delay, not a stuck layer.
    private static readonly TimeSpan IdleWake = TimeSpan.FromMilliseconds(250);

    private readonly SplitWriter _writer;
    private readonly Func<bool> _alive;
    private readonly AutoResetEvent _wake = new(false);
    private readonly Dictionary<int, OverlayLayerFrame> _sent = new();
    private readonly List<OverlayLayerFrame> _frames = [];
    private OverlayLayerPainter _painter = new();
    private long _surface;
    private volatile bool _stopping;
    private SharpEmu.HLE.Host.HostHelperThread? _thread;

    public SplitOverlayLayerSender(SplitWriter writer, Func<bool> alive)
    {
        _writer = writer;
        _alive = alive;
        OverlayLayers.Changed += Wake;
    }

    /// <summary>Records sent so far, pieces included (tests).</summary>
    internal long RecordsSent { get; private set; }

    /// <summary>Replaces where the layers come from (tests); the next pass sends what it paints.</summary>
    internal Func<OverlayLayer[]> Layers
    {
        set
        {
            lock (_sent)
            {
                _painter = new OverlayLayerPainter(value);
            }

            Wake();
        }
    }

    public void Start() =>
        _thread = SharpEmu.HLE.Host.HostHelperThread.Start("SharpEmu GPU split overlay layers", 1024 * 1024, Run);

    /// <summary>The GPU process's drawable took this size.</summary>
    public void SetSurface(uint width, uint height)
    {
        Interlocked.Exchange(ref _surface, (long)width << 32 | height);
        Wake();
    }

    private void Wake()
    {
        if (!_stopping)
        {
            _wake.Set();
        }
    }

    private void Run()
    {
        while (!_stopping && _alive())
        {
            try
            {
                SendChanges();
            }
            catch (Exception exception)
            {
                // A layer that cannot paint must not take the split down; it is reported, and the
                // presenter keeps the last frame it received.
                Console.Error.WriteLine($"[GPU-SPLIT][ERROR] An overlay layer could not be painted: {exception}");
            }

            _ = _wake.WaitOne(IdleWake);
        }
    }

    /// <summary>One pass: paints what changed and sends it; internal so a test can drive it.</summary>
    internal void SendChanges()
    {
        var packed = Interlocked.Read(ref _surface);
        var width = (int)(packed >> 32);
        var height = (int)(uint)packed;
        if (width <= 0 || height <= 0)
        {
            return;
        }

        lock (_sent)
        {
            _frames.Clear();
            _painter.Collect(width, height, _frames);
            foreach (var frame in _frames)
            {
                if (_sent.TryGetValue(frame.Id, out var sent) && ReferenceEquals(sent, frame))
                {
                    continue;
                }

                if (!Send(frame))
                {
                    return;
                }

                _sent[frame.Id] = frame;
            }

            foreach (var id in _sent.Keys.Where(id => !_frames.Exists(frame => frame.Id == id)).ToList())
            {
                var hide = new OverlayLayerMessage { Id = (uint)id, Visible = 0 };
                if (!_writer.Write(GpuRemoteMessage.OverlayLayer, in hide))
                {
                    return;
                }

                RecordsSent++;
                _sent.Remove(id);
            }
        }
    }

    // The pieces go back to back: another writer cannot put a record between them.
    private bool Send(OverlayLayerFrame frame)
    {
        lock (_writer.Gate)
        {
            for (var offset = 0; offset < frame.Pixels.Length; offset += PieceBytes)
            {
                var length = Math.Min(PieceBytes, frame.Pixels.Length - offset);
                var header = new OverlayLayerMessage
                {
                    Id = (uint)frame.Id,
                    Visible = 1,
                    Revision = frame.Revision,
                    SurfaceWidth = (uint)frame.SurfaceWidth,
                    SurfaceHeight = (uint)frame.SurfaceHeight,
                    Left = frame.Left,
                    Top = frame.Top,
                    Width = (uint)frame.Width,
                    Height = (uint)frame.Height,
                    TotalBytes = (uint)frame.Pixels.Length,
                    Offset = (uint)offset,
                };
                if (!_writer.Write(GpuRemoteMessage.OverlayLayer, in header, frame.Pixels.AsSpan(offset, length)))
                {
                    return false;
                }

                RecordsSent++;
            }
        }

        return true;
    }

    public void Dispose()
    {
        _stopping = true;
        OverlayLayers.Changed -= Wake;
        _wake.Set();
        if (_thread is { IsCurrentThread: false } thread)
        {
            _ = thread.Join(TimeSpan.FromSeconds(2));
        }
    }
}
