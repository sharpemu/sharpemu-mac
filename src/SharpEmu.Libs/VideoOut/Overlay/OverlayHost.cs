// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Fonts.Abstractions;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>
/// Owns the overlay's background render thread and its triple-buffered output. A graphics backend
/// calls <see cref="AcquireFrame"/> on its present thread and <see cref="ReleaseFrame"/> after the
/// upload; the window forwards pointer and key events to <see cref="HandlePointer"/> /
/// <see cref="HandleKey"/>. Everything else (snapshot capture, layout, rasterization) happens on
/// the low-priority thread, at most <see cref="TargetHz"/> times a second and immediately after an
/// interaction.
/// </summary>
public sealed class OverlayHost : IDisposable
{
    private const int BufferCount = 3;

    private readonly object _poolLock = new();
    private readonly OverlayFrame?[] _pool = new OverlayFrame?[BufferCount];
    private readonly bool[] _leased = new bool[BufferCount];
    private int _latest = -1;

    private readonly object _surfaceLock = new();
    private int _surfaceW, _surfaceH;
    private double _surfaceScale = 1.0;

    private readonly ManualResetEventSlim _wake = new(false);
    private readonly OverlayLayoutStore? _store;
    private readonly Func<bool> _isVisible;
    private readonly OverlayAnalyzer _analyzer = new();
    private readonly OverlayView _view = new();
    private IOverlayDataSource? _source;
    private IFontFactory? _fonts;
    private OverlayRenderer? _renderer;
    private OverlaySnapshot? _frozen;
    private Thread? _thread;
    private volatile bool _disposed;
    private volatile bool _wantsPointer;
    private OverlayCaptureDemand _demand = (OverlayCaptureDemand)(-1);
    private long _nextFontRetry;
    private long _savedRevision;
    private long _lastRevisionSeen;
    private long _lastChangeTicks;
    private readonly long _memoryIntervalTicks = Stopwatch.Frequency / 2;

    public OverlayHost(IOverlayDataSource? source = null, IFontFactory? fonts = null, Func<bool>? isVisible = null,
        OverlayLayoutStore? store = null, string? initialPreset = null, string? launchPreset = null)
    {
        _source = source;
        _fonts = fonts;
        _isVisible = isVisible ?? (static () => true);
        _store = store;
        Model = new OverlayLayoutModel();
        Controller = new OverlayController(Model);
        var loaded = store?.LoadInto(Model) ?? false;
        // A preset named at launch wins over the saved layout, except that a saved layout of that same
        // preset is kept, so edits made to it survive a relaunch.
        var launched = launchPreset is not null &&
            ((loaded && string.Equals(Model.PresetName, launchPreset, StringComparison.Ordinal)) || Model.ApplyPreset(launchPreset));
        if (!launched && !loaded && initialPreset is not null)
        {
            Model.ApplyPreset(initialPreset);
        }

        _savedRevision = _lastRevisionSeen = Model.Revision;
        _storeStamp = store?.Stamp() ?? 0;
    }

    private long _storeStamp;
    private long _nextStampCheck;

    // Picks up layout and style changes written by another process (the launcher's settings page)
    // while the game runs. Edits in progress here win: nothing reloads mid-drag or with a popover open.
    private void ReloadIfChangedOnDisk(long now)
    {
        if (_store is null || now < _nextStampCheck)
        {
            return;
        }

        _nextStampCheck = now + Stopwatch.Frequency;
        var stamp = _store.Stamp();
        if (stamp == _storeStamp)
        {
            return;
        }

        lock (Controller.Gate)
        {
            if (Controller.Dragging || Controller.Popover != OverlayPopoverKind.None || Model.Revision != _savedRevision)
            {
                return;
            }

            _storeStamp = stamp;
            if (_store.LoadInto(Model))
            {
                _savedRevision = _lastRevisionSeen = Model.Revision;
                Controller.Dirty = true;
            }
        }
    }

    public OverlayLayoutModel Model { get; }
    public OverlayController Controller { get; }

    /// <summary>Redraw rate cap in Hz (default 30).</summary>
    public int TargetHz { get; set; } = 30;

    /// <summary>True when the pointer is over a widget or the overlay is in edit mode (show a cursor, do not forward clicks).</summary>
    public bool WantsPointer => _wantsPointer;

    public void AttachDataSource(IOverlayDataSource? source)
    {
        _source = source;
        _demand = (OverlayCaptureDemand)(-1);
        Controller.Dirty = true;
        _wake.Set();
    }

    public void SetFontFactory(IFontFactory? factory)
    {
        lock (Controller.Gate)
        {
            _fonts = factory;
            _renderer?.Dispose();
            _renderer = null;
            Controller.Dirty = true;
        }

        _wake.Set();
    }

    // ------------------------------------------------------------------ backend side

    /// <summary>
    /// The newest completed frame for this surface size, or null when the overlay is hidden or not
    /// ready yet. The caller must pass it to <see cref="ReleaseFrame"/> after uploading it.
    /// </summary>
    public OverlayFrame? AcquireFrame(int pixelWidth, int pixelHeight, double pixelScale)
    {
        if (pixelWidth <= 0 || pixelHeight <= 0 || !_isVisible())
        {
            return null;
        }

        lock (_surfaceLock)
        {
            if (_surfaceW != pixelWidth || _surfaceH != pixelHeight || Math.Abs(_surfaceScale - pixelScale) > 1e-6)
            {
                _surfaceW = pixelWidth;
                _surfaceH = pixelHeight;
                _surfaceScale = pixelScale;
                _wake.Set();
            }
        }

        EnsureThread();
        lock (_poolLock)
        {
            if (_latest < 0 || _pool[_latest] is not { } frame || frame.Width != pixelWidth || frame.Height != pixelHeight)
            {
                return null;
            }

            _leased[_latest] = true;
            return frame;
        }
    }

    public void ReleaseFrame(OverlayFrame frame)
    {
        lock (_poolLock)
        {
            for (var i = 0; i < BufferCount; i++)
            {
                if (ReferenceEquals(_pool[i], frame))
                {
                    _leased[i] = false;
                    return;
                }
            }
        }
    }

    // ------------------------------------------------------------------ window side

    public bool HandlePointer(in OverlayPointerEvent e)
    {
        if (!_isVisible())
        {
            return false;
        }

        bool consumed, dirty;
        lock (Controller.Gate)
        {
            consumed = Controller.HandlePointer(e);
            _wantsPointer = Controller.WantsPointer;
            dirty = Controller.Dirty;
        }

        if (dirty)
        {
            _wake.Set();
        }

        return consumed;
    }

    /// <summary>Handles an overlay key; <see cref="OverlayKey.ToggleVisible"/> belongs to the caller's visibility switch.</summary>
    public bool HandleKey(OverlayKey key)
    {
        if (key == OverlayKey.ToggleVisible || !_isVisible())
        {
            return false;
        }

        bool consumed;
        lock (Controller.Gate)
        {
            consumed = Controller.HandleKey(key);
            _wantsPointer = Controller.WantsPointer;
        }

        _wake.Set();
        return consumed;
    }

    // ------------------------------------------------------------------ render thread

    private void EnsureThread()
    {
        if (_thread is not null || _disposed)
        {
            return;
        }

        lock (_poolLock)
        {
            if (_thread is not null)
            {
                return;
            }

            _thread = new Thread(RenderLoop)
            {
                Name = "SharpEmu.Overlay",
                IsBackground = true,
                Priority = ThreadPriority.BelowNormal,
            };
            _thread.Start();
        }
    }

    private bool _reportedFailure;

    private void RenderLoop()
    {
        var lastRender = 0L;
        while (!_disposed)
        {
            var interval = Stopwatch.Frequency / Math.Max(1, TargetHz);
            var sinceLast = Stopwatch.GetTimestamp() - lastRender;
            var waitMs = (int)Math.Max(1, (interval - sinceLast) * 1000 / Stopwatch.Frequency);
            // Interactions wake the thread early; otherwise it ticks at the target rate.
            _wake.Wait(Controller.Dirty ? 1 : waitMs);
            _wake.Reset();
            if (_disposed)
            {
                break;
            }

            try
            {
                if (!_isVisible())
                {
                    PublishDemand(OverlayCaptureDemand.None);
                    _wake.Wait(100);
                    _wake.Reset();
                    continue;
                }

                int w, h;
                double scale;
                lock (_surfaceLock)
                {
                    (w, h, scale) = (_surfaceW, _surfaceH, _surfaceScale);
                }

                if (w <= 0 || h <= 0)
                {
                    continue;
                }

                var now = Stopwatch.GetTimestamp();
                var due = now - lastRender >= interval - interval / 8;
                bool live, toast;
                lock (Controller.Gate)
                {
                    live = !Controller.Paused;
                    Controller.ExpireToast(now);
                    toast = Controller.ToastActive(now);
                }

                var dirty = Controller.Dirty;
                if (live ? !due && !dirty : !dirty && !(toast && due) && !SurfaceStale(w, h))
                {
                    continue;
                }

                RenderOnce(w, h, scale);
                lastRender = Stopwatch.GetTimestamp();
                SaveIfChanged();
                ReloadIfChangedOnDisk(lastRender);
            }
            catch (Exception ex)
            {
                // The overlay must never take the emulator down; try again on the next tick.
                // The first failure is reported so a dead overlay is not silent.
                if (!_reportedFailure)
                {
                    _reportedFailure = true;
                    Console.Error.WriteLine($"[OVERLAY][ERROR] The overlay render loop failed: {ex}");
                }

                Thread.Sleep(50);
            }
        }
    }

    private bool SurfaceStale(int w, int h)
    {
        lock (_poolLock)
        {
            return _latest < 0 || _pool[_latest] is not { } f || f.Width != w || f.Height != h;
        }
    }

    private void PublishDemand(OverlayCaptureDemand demand)
    {
        if (_demand == demand || _source is not { } source)
        {
            return;
        }

        _demand = demand;
        source.SetCaptureDemand(demand);
    }

    /// <summary>The captures the widgets in a layout need.</summary>
    public static OverlayCaptureDemand DemandFor(IEnumerable<WidgetInstance> widgets)
    {
        var demand = OverlayCaptureDemand.None;
        foreach (var w in widgets)
        {
            demand |= w.Type switch
            {
                OverlayWidgetType.Budget or OverlayWidgetType.Strip => OverlayCaptureDemand.RenderPhases | OverlayCaptureDemand.ThreadCpu,
                OverlayWidgetType.Split => OverlayCaptureDemand.RenderPhases,
                OverlayWidgetType.Timeline => OverlayCaptureDemand.Timeline | OverlayCaptureDemand.ThreadCpu,
                OverlayWidgetType.TopCosts => OverlayCaptureDemand.RenderPhases | OverlayCaptureDemand.HleCalls,
                OverlayWidgetType.Cpu => OverlayCaptureDemand.ThreadCpu,
                _ => OverlayCaptureDemand.None,
            };
        }

        return demand;
    }

    /// <summary>
    /// Captures a snapshot (or reuses the frozen one while paused), builds the view and rasterizes
    /// one frame into a free buffer, then publishes it. Public so tests and tools can render
    /// synchronously without the thread.
    /// </summary>
    public OverlayFrame? RenderOnce(int pixelWidth, int pixelHeight, double pixelScale)
    {
        PublishDemand(DemandFor(Model.Widgets));
        bool paused, freeze;
        lock (Controller.Gate)
        {
            paused = Controller.Paused;
            freeze = paused && (_frozen is null || Controller.FreezeRequested);
            if (!paused)
            {
                _frozen = null;
            }
        }

        var snapshot = paused && !freeze ? _frozen! : (_source?.Capture() ?? OverlaySnapshot.Empty);
        var nowTicks = Stopwatch.GetTimestamp();
        lock (Controller.Gate)
        {
            if (freeze && Controller.Paused)
            {
                _frozen = snapshot;
                var frames = snapshot.Frames;
                Controller.OnFrozen(frames.Count > 0 ? frames[^1].Sequence : 0);
            }

            if (Controller.Paused && snapshot.Frames.Count > 0)
            {
                Controller.ClampSelection(snapshot.Frames[0].Sequence, snapshot.Frames[^1].Sequence);
            }

            if (!Controller.Paused && snapshot.System.ProcessMemoryBytes is { } bytes)
            {
                _analyzer.AddMemorySample(bytes / (1024.0 * 1024.0 * 1024.0), nowTicks, _memoryIntervalTicks);
            }

            _analyzer.Build(snapshot, Controller.Paused, Controller.EndSequence, Controller.SelectedSequence, _view);
            var frame = AcquireBack(pixelWidth, pixelHeight);
            if (frame is null)
            {
                return null;
            }

            if (_renderer is { UsesBuiltInFont: true } && _fonts is null && nowTicks >= _nextFontRetry)
            {
                // The host composes its font services after the overlay may already have drawn once.
                _nextFontRetry = nowTicks + 2 * Stopwatch.Frequency;
                if (TryDefaultFonts() is { } late)
                {
                    _renderer.Dispose();
                    _renderer = new OverlayRenderer(Controller, _analyzer, late);
                }
            }

            _renderer ??= new OverlayRenderer(Controller, _analyzer, _fonts ?? TryDefaultFonts());
            _renderer.Render(frame, _view, pixelScale, nowTicks);
            Controller.Dirty = false;
            _wantsPointer = Controller.WantsPointer;
            Publish(frame);
            return frame;
        }
    }

    /// <summary>Optional hook: where the host looks for fonts when none were injected.</summary>
    public static Func<IFontFactory?>? DefaultFontProvider { get; set; }

    private static IFontFactory? TryDefaultFonts()
    {
        try
        {
            return DefaultFontProvider?.Invoke();
        }
        catch (Exception)
        {
            return null;
        }
    }

    private OverlayFrame? AcquireBack(int w, int h)
    {
        lock (_poolLock)
        {
            for (var i = 0; i < BufferCount; i++)
            {
                if (i == _latest || _leased[i])
                {
                    continue;
                }

                if (_pool[i] is not { } f || f.Width != w || f.Height != h)
                {
                    _pool[i] = new OverlayFrame(w, h);
                }

                return _pool[i];
            }

            return null;
        }
    }

    private void Publish(OverlayFrame frame)
    {
        lock (_poolLock)
        {
            for (var i = 0; i < BufferCount; i++)
            {
                if (ReferenceEquals(_pool[i], frame))
                {
                    _latest = i;
                    return;
                }
            }
        }
    }

    private void SaveIfChanged()
    {
        if (_store is null)
        {
            return;
        }

        var now = Stopwatch.GetTimestamp();
        long revision;
        lock (Controller.Gate)
        {
            revision = Model.Revision;
        }

        if (revision != _lastRevisionSeen)
        {
            _lastRevisionSeen = revision;
            _lastChangeTicks = now;
            return;
        }

        if (revision != _savedRevision && now - _lastChangeTicks > Stopwatch.Frequency / 4 && !Controller.Dragging)
        {
            lock (Controller.Gate)
            {
                _store.Save(Model);
                _savedRevision = Model.Revision;
                _storeStamp = _store.Stamp();
            }
        }
    }

    /// <summary>Writes the layout now (used on shutdown).</summary>
    public void Flush()
    {
        if (_store is null)
        {
            return;
        }

        lock (Controller.Gate)
        {
            if (Model.Revision != _savedRevision)
            {
                _store.Save(Model);
                _savedRevision = Model.Revision;
                _storeStamp = _store.Stamp();
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _wake.Set();
        _thread?.Join(500);
        Flush();
        _renderer?.Dispose();
        _wake.Dispose();
    }
}
