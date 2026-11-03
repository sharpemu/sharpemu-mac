// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.VideoOut.Overlay;

public enum OverlayPopoverKind : byte
{
    None,
    WidgetSettings,
    AddWidget,
    Style,
}

/// <summary>
/// The overlay's interaction state machine: edit mode, dragging, snapping, popovers, pause and
/// frame stepping. It is pure logic over the layout model and the hit list the renderer records;
/// callers serialise access with <see cref="Gate"/>.
/// </summary>
public sealed class OverlayController
{
    public const float GridSize = 10f;
    public const float SnapDistance = 6f;
    public const float EdgeMargin = 10f;
    public const float EdgeGap = 6f;
    public const int HistoryWindow = 240;

    private enum DragKind : byte { None, Move, Resize, Slider }

    private readonly OverlayLayoutModel _model;
    private DragKind _drag;
    private WidgetInstance? _dragWidget;
    private RectF _dragStartBox;
    private float _dragPointerX, _dragPointerY;
    private bool _dragMoved;
    private int _sliderHit = -1;
    private OverlayHit _sliderRegion;
    private long _toastUntil;
    private readonly float[] _edgesLowX = new float[64];
    private readonly float[] _edgesHighX = new float[64];
    private readonly float[] _edgesLowY = new float[64];
    private readonly float[] _edgesHighY = new float[64];
    private int _edgeCount;

    public OverlayController(OverlayLayoutModel model)
    {
        _model = model;
    }

    /// <summary>Held by the host while it draws and by the input entry points.</summary>
    public object Gate { get; } = new();

    public OverlayLayoutModel Model => _model;
    public OverlayHitMap Hits { get; } = new();
    public OverlayMetrics Metrics { get; private set; } = new(960, 540, 1, 1);

    public bool Edit { get; private set; }
    public bool Paused { get; private set; }
    /// <summary>The frame inspected while paused (a sequence number), or -1.</summary>
    public long SelectedSequence { get; private set; } = -1;
    /// <summary>The last frame of the displayed history window while paused, or -1 until a snapshot is frozen.</summary>
    public long EndSequence { get; private set; } = -1;
    /// <summary>The frames of the snapshot being displayed (set by the renderer); used to map history clicks.</summary>
    public IReadOnlyList<FrameRecord> Frames { get; set; } = [];

    public string? SelectedWidgetId { get; private set; }
    public OverlayPopoverKind Popover { get; private set; }
    public string? PopoverWidgetId { get; private set; }
    public RectF PopoverAnchor { get; private set; }
    public float PopoverScroll { get; set; }

    public float PointerX { get; private set; } = -1f;
    public float PointerY { get; private set; } = -1f;
    public bool PointerInside { get; private set; }
    public int HoverHit { get; private set; } = -1;
    public string? ToastText { get; private set; }

    /// <summary>Guide lines (points) shown while snapping, or NaN.</summary>
    public float GuideX { get; private set; } = float.NaN;
    public float GuideY { get; private set; } = float.NaN;

    /// <summary>True while the host should redraw (any state or pointer change since the last redraw).</summary>
    public bool Dirty { get; set; } = true;

    /// <summary>True when a pause or frame selection needs the host to freeze a snapshot.</summary>
    public bool FreezeRequested { get; private set; }

    public bool Dragging => _drag != DragKind.None;

    /// <summary>The overlay wants the pointer (cursor shown, clicks not forwarded): edit mode or a pointer over a widget.</summary>
    public bool WantsPointer => Edit || (PointerInside && HoverHit >= 0);

    // ---------------------------------------------------------------- surface / bookkeeping

    public void UpdateSurface(OverlayMetrics metrics) => Metrics = metrics;

    /// <summary>Re-resolves the hovered region after a redraw rebuilt the hit list.</summary>
    public void RefreshHover() => HoverHit = PointerInside ? Hits.Find(PointerX, PointerY) : -1;

    /// <summary>Called by the renderer before drawing; the new hit list is rebuilt during the draw.</summary>
    public void BeginHits() => Hits.Clear();

    public void ShowToast(string text, TimeSpan? duration = null)
    {
        ToastText = text;
        _toastUntil = Stopwatch.GetTimestamp() + (long)((duration ?? TimeSpan.FromMilliseconds(1700)).TotalSeconds * Stopwatch.Frequency);
        Dirty = true;
    }

    public bool ToastActive(long nowTicks) => ToastText is not null && nowTicks < _toastUntil;

    /// <summary>Clears an expired toast; returns true when one was cleared (a redraw is needed).</summary>
    public bool ExpireToast(long nowTicks)
    {
        if (ToastText is not null && nowTicks >= _toastUntil)
        {
            ToastText = null;
            Dirty = true;
            return true;
        }

        return false;
    }

    /// <summary>The host froze a snapshot: fill in the window end and selection if they were left open.</summary>
    public void OnFrozen(long lastSequence)
    {
        FreezeRequested = false;
        if (EndSequence < 0)
        {
            EndSequence = lastSequence;
        }

        if (SelectedSequence < 0)
        {
            SelectedSequence = lastSequence;
        }
    }

    /// <summary>Keeps the selection inside the frames of the frozen snapshot.</summary>
    public void ClampSelection(long lowestSequence, long highestSequence)
    {
        if (!Paused)
        {
            return;
        }

        if (SelectedSequence >= 0)
        {
            SelectedSequence = Math.Clamp(SelectedSequence, lowestSequence, highestSequence);
        }

        if (EndSequence >= 0)
        {
            EndSequence = Math.Clamp(EndSequence, lowestSequence, highestSequence);
        }
    }

    // ---------------------------------------------------------------- modes

    public void SetEdit(bool on)
    {
        if (Edit == on)
        {
            return;
        }

        Edit = on;
        ClosePopover();
        SelectedWidgetId = null;
        _drag = DragKind.None;
        GuideX = GuideY = float.NaN;
        ShowToast(on ? "Edit mode · drag, resize, gear for settings" : "Edit mode off");
        Dirty = true;
    }

    public void SetPaused(bool paused)
    {
        if (paused == Paused)
        {
            return;
        }

        Paused = paused;
        if (paused)
        {
            EndSequence = -1;
            SelectedSequence = -1;
            FreezeRequested = true;
        }
        else
        {
            EndSequence = -1;
            SelectedSequence = -1;
            FreezeRequested = false;
            ShowToast("Resumed live");
        }

        Dirty = true;
    }

    /// <summary>Selects a frame by sequence number and pauses if live.</summary>
    public void SelectFrame(long sequence)
    {
        if (!Paused)
        {
            Paused = true;
            EndSequence = -1;
            FreezeRequested = true;
        }

        SelectedSequence = sequence;
        Dirty = true;
    }

    /// <summary>Moves the selected frame by <paramref name="delta"/> (pausing first when live).</summary>
    public void Step(int delta, long lowestSequence = long.MinValue, long highestSequence = long.MaxValue)
    {
        if (!Paused)
        {
            SetPaused(true);
            return;
        }

        if (SelectedSequence < 0)
        {
            return;
        }

        SelectedSequence = Math.Clamp(SelectedSequence + delta, lowestSequence, highestSequence);
        if (EndSequence >= 0)
        {
            if (SelectedSequence < EndSequence - (HistoryWindow - 1))
            {
                EndSequence = Math.Min(highestSequence, SelectedSequence + (HistoryWindow - 1));
            }

            if (SelectedSequence > EndSequence)
            {
                EndSequence = SelectedSequence;
            }
        }

        Dirty = true;
    }

    public void ClosePopover()
    {
        if (Popover != OverlayPopoverKind.None)
        {
            Popover = OverlayPopoverKind.None;
            PopoverWidgetId = null;
            PopoverScroll = 0;
            Dirty = true;
        }
    }

    private void OpenPopover(OverlayPopoverKind kind, string? widgetId, RectF anchor)
    {
        Popover = kind;
        PopoverWidgetId = widgetId;
        PopoverAnchor = anchor;
        PopoverScroll = 0;
        Dirty = true;
    }

    // ---------------------------------------------------------------- keys

    /// <summary>Handles a key; returns true when the overlay consumed it. <see cref="OverlayKey.ToggleVisible"/> is not handled here.</summary>
    public bool HandleKey(OverlayKey key)
    {
        switch (key)
        {
            case OverlayKey.CyclePreset:
                _model.CyclePreset(+1);
                ShowToast("Preset: " + _model.PresetName);
                ClosePopover();
                Dirty = true;
                return true;
            case OverlayKey.ToggleEdit:
                SetEdit(!Edit);
                return true;
            case OverlayKey.TogglePause:
                SetPaused(!Paused);
                return true;
            case OverlayKey.StepBack:
                if (!Paused)
                {
                    return false;
                }

                Step(-1, 0, long.MaxValue);
                return true;
            case OverlayKey.StepForward:
                if (!Paused)
                {
                    return false;
                }

                Step(+1, 0, long.MaxValue);
                return true;
            case OverlayKey.Escape:
                if (Popover != OverlayPopoverKind.None)
                {
                    ClosePopover();
                    return true;
                }

                if (Paused)
                {
                    SetPaused(false);
                    return true;
                }

                if (Edit)
                {
                    SetEdit(false);
                    return true;
                }

                return false;
            case OverlayKey.Preset1:
            case OverlayKey.Preset2:
            case OverlayKey.Preset3:
            case OverlayKey.Preset4:
                if (!Edit)
                {
                    return false;
                }

                var name = OverlayLayoutModel.BuiltInPresetNames[key - OverlayKey.Preset1];
                _model.ApplyPreset(name);
                ClosePopover();
                SelectedWidgetId = null;
                ShowToast("Preset: " + name);
                Dirty = true;
                return true;
            default:
                return false;
        }
    }

    // ---------------------------------------------------------------- pointer

    /// <summary>Handles a pointer event; returns true when the overlay consumed it (the game must not see it).</summary>
    public bool HandlePointer(in OverlayPointerEvent e)
    {
        switch (e.Kind)
        {
            case OverlayPointerKind.Leave:
                PointerInside = false;
                HoverHit = -1;
                if (_drag == DragKind.None)
                {
                    Dirty = true;
                }

                return false;
            case OverlayPointerKind.Move:
                return OnMove(e);
            case OverlayPointerKind.Down:
                return OnDown(e);
            case OverlayPointerKind.Up:
                return OnUp(e);
            case OverlayPointerKind.Scroll:
                return OnScroll(e);
            default:
                return false;
        }
    }

    private void Track(float x, float y)
    {
        PointerX = x;
        PointerY = y;
        PointerInside = true;
        HoverHit = Hits.Find(x, y);
    }

    private bool OnMove(in OverlayPointerEvent e)
    {
        var previous = HoverHit;
        var px = PointerX;
        var py = PointerY;
        Track(e.X, e.Y);
        if (_drag != DragKind.None)
        {
            ContinueDrag(e);
            Dirty = true;
            return true;
        }

        // Tooltips follow the pointer; other hover changes only matter in edit mode.
        var kind = HoverHit >= 0 ? Hits[HoverHit].Kind : OverlayHitKind.None;
        if (previous != HoverHit || kind is OverlayHitKind.HistoryBar or OverlayHitKind.TimelineLane ||
            (Edit && (px != e.X || py != e.Y)))
        {
            Dirty = true;
        }

        return Edit || HoverHit >= 0;
    }

    private bool OnUp(in OverlayPointerEvent e)
    {
        Track(e.X, e.Y);
        if (_drag == DragKind.None)
        {
            return Edit || HoverHit >= 0;
        }

        if (_dragMoved && _drag is DragKind.Move or DragKind.Resize)
        {
            _model.MarkModified();
        }

        _drag = DragKind.None;
        _dragWidget = null;
        _sliderHit = -1;
        GuideX = GuideY = float.NaN;
        Dirty = true;
        return true;
    }

    private bool OnScroll(in OverlayPointerEvent e)
    {
        Track(e.X, e.Y);
        if (Popover == OverlayPopoverKind.AddWidget)
        {
            PopoverScroll = MathF.Max(0f, PopoverScroll - e.ScrollDelta * 24f);
            Dirty = true;
            return true;
        }

        return Edit || HoverHit >= 0;
    }

    private bool OnDown(in OverlayPointerEvent e)
    {
        Track(e.X, e.Y);
        var hit = HoverHit;
        var kind = hit >= 0 ? Hits[hit].Kind : OverlayHitKind.None;
        if (e.Button != OverlayMouseButton.Left)
        {
            return Edit || hit >= 0;
        }

        var onPopover = kind is >= OverlayHitKind.PopoverBody and <= OverlayHitKind.PopoverDeletePreset;
        if (Popover != OverlayPopoverKind.None && !onPopover)
        {
            // A click anywhere else dismisses the popover; a click on its own gear toggles it.
            var wasGear = kind == OverlayHitKind.WidgetGear && Popover == OverlayPopoverKind.WidgetSettings &&
                          WidgetOf(Hits[hit].A)?.Id == PopoverWidgetId;
            ClosePopover();
            if (wasGear)
            {
                return true;
            }
        }

        if (hit < 0)
        {
            if (Edit)
            {
                SelectedWidgetId = null;
                Dirty = true;
                return true;
            }

            return false;
        }

        ref var region = ref Hits[hit];
        switch (region.Kind)
        {
            case OverlayHitKind.WidgetBody:
                if (Edit)
                {
                    BeginMove(WidgetOf(region.A), e);
                }

                return true;
            case OverlayHitKind.WidgetResize:
                BeginResize(WidgetOf(region.A), e);
                return true;
            case OverlayHitKind.WidgetGear:
                if (WidgetOf(region.A) is { } gearWidget)
                {
                    SelectedWidgetId = gearWidget.Id;
                    OpenPopover(OverlayPopoverKind.WidgetSettings, gearWidget.Id, region.Rect);
                }

                return true;
            case OverlayHitKind.WidgetRemove:
                if (WidgetOf(region.A) is { } victim)
                {
                    _model.Remove(victim);
                    SelectedWidgetId = null;
                    ClosePopover();
                    Dirty = true;
                }

                return true;
            case OverlayHitKind.HistoryBar:
                SelectHistorySlot(in region, e.X);
                return true;
            case OverlayHitKind.TopCostsSort:
                CycleTopCosts(WidgetOf(region.A), WidgetKeys.TopCosts.Sort, 3);
                return true;
            case OverlayHitKind.TopCostsSource:
                CycleTopCosts(WidgetOf(region.A), WidgetKeys.TopCosts.Source, 2);
                return true;
            case OverlayHitKind.TimelineLane:
                return true;
            case OverlayHitKind.EditAdd:
                OpenPopover(OverlayPopoverKind.AddWidget, null, region.Rect);
                return true;
            case OverlayHitKind.EditPresetPrev:
            case OverlayHitKind.EditPresetNext:
                _model.CyclePreset(region.Kind == OverlayHitKind.EditPresetNext ? +1 : -1);
                ClosePopover();
                SelectedWidgetId = null;
                ShowToast("Preset: " + _model.PresetName);
                return true;
            case OverlayHitKind.EditSaveAs:
                var saved = _model.SaveAs(_model.NextCustomName());
                ShowToast("Saved preset “" + saved + "”");
                Dirty = true;
                return true;
            case OverlayHitKind.EditReset:
                _model.ResetLayout();
                ClosePopover();
                SelectedWidgetId = null;
                ShowToast("Layout reset");
                return true;
            case OverlayHitKind.EditStyle:
                OpenPopover(OverlayPopoverKind.Style, null, region.Rect);
                return true;
            case OverlayHitKind.EditDone:
                SetEdit(false);
                return true;
            case OverlayHitKind.EditBar:
            case OverlayHitKind.PopoverBody:
                return true;
            case OverlayHitKind.PopoverAddItem:
                if (region.A >= 0 && region.A < WidgetCatalog.All.Count)
                {
                    var added = _model.AddWidget(WidgetCatalog.All[region.A].Type);
                    SelectedWidgetId = added.Id;
                    ShowToast("Added " + added.Spec.Title);
                }

                ClosePopover();
                return true;
            case OverlayHitKind.PopoverDone:
                ClosePopover();
                return true;
            case OverlayHitKind.PopoverRemove:
                if (Popover == OverlayPopoverKind.WidgetSettings && _model.Find(PopoverWidgetId ?? string.Empty) is { } target)
                {
                    _model.Remove(target);
                    SelectedWidgetId = null;
                }

                ClosePopover();
                return true;
            case OverlayHitKind.PopoverDeletePreset:
                if (!_model.Modified && _model.CustomPresets.ContainsKey(_model.PresetName))
                {
                    var name = _model.PresetName;
                    _model.DeleteCustom(name);
                    ShowToast("Deleted preset “" + name + "”");
                }

                Dirty = true;
                return true;
            case OverlayHitKind.PopoverToggle:
                ApplySetting(region.A, region.Kind, 0);
                return true;
            case OverlayHitKind.PopoverCyclePrev:
                ApplySetting(region.A, region.Kind, -1);
                return true;
            case OverlayHitKind.PopoverCycleNext:
                ApplySetting(region.A, region.Kind, +1);
                return true;
            case OverlayHitKind.PopoverStepDown:
                ApplySetting(region.A, region.Kind, -1);
                return true;
            case OverlayHitKind.PopoverStepUp:
                ApplySetting(region.A, region.Kind, +1);
                return true;
            case OverlayHitKind.PopoverSlider:
                _drag = DragKind.Slider;
                _sliderHit = hit;
                _sliderRegion = region;
                ApplySlider(region, e.X);
                return true;
            default:
                return true;
        }
    }

    private WidgetInstance? WidgetOf(int index) =>
        index >= 0 && index < _model.Widgets.Count ? _model.Widgets[index] : null;

    private void CycleTopCosts(WidgetInstance? w, int setting, int count)
    {
        if (w is null)
        {
            return;
        }

        w.SetValue(setting, (w.Choice(setting) + 1) % count);
        _model.MarkModified();
        Dirty = true;
    }

    private void SelectHistorySlot(in OverlayHit region, float x)
    {
        var slots = region.A;
        var endIndex = region.B;
        if (slots <= 0 || region.F2 <= 0)
        {
            return;
        }

        var slot = (int)MathF.Floor((x - region.F1) / region.F2);
        if (slot < 0 || slot >= slots)
        {
            return;
        }

        var index = endIndex - (slots - 1 - slot);
        if (index >= 0 && index < Frames.Count)
        {
            SelectFrame(Frames[index].Sequence);
        }
    }

    // ---------------------------------------------------------------- settings popovers

    private void ApplySetting(int index, OverlayHitKind kind, int direction)
    {
        if (Popover == OverlayPopoverKind.WidgetSettings)
        {
            var w = _model.Find(PopoverWidgetId ?? string.Empty);
            if (w is null || index < 0 || index >= w.Spec.Settings.Length)
            {
                return;
            }

            var def = w.Spec.Settings[index];
            switch (kind)
            {
                case OverlayHitKind.PopoverToggle:
                    w.SetValue(index, w.Bool(index) ? 0f : 1f);
                    break;
                case OverlayHitKind.PopoverCyclePrev:
                case OverlayHitKind.PopoverCycleNext:
                    var n = Math.Max(1, def.Options.Length);
                    w.SetValue(index, ((w.Choice(index) + direction) % n + n) % n);
                    break;
                default:
                    w.SetValue(index, w.Values[index] + direction * def.Step);
                    break;
            }

            _model.MarkModified();
        }
        else if (Popover == OverlayPopoverKind.Style && index == 3)
        {
            var themes = Enum.GetValues<OverlayTheme>().Length;
            _model.Globals.Theme = (OverlayTheme)(((int)_model.Globals.Theme + (direction >= 0 ? 1 : -1) + themes) % themes);
            _model.Touch();
        }

        Dirty = true;
    }

    private void ApplySlider(in OverlayHit region, float x)
    {
        var t = region.F2 > 0 ? Math.Clamp((x - region.Rect.X) / region.F2, 0f, 1f) : 0f;
        if (Popover == OverlayPopoverKind.WidgetSettings)
        {
            var w = _model.Find(PopoverWidgetId ?? string.Empty);
            if (w is not null && region.A >= 0 && region.A < w.Spec.Settings.Length)
            {
                var def = w.Spec.Settings[region.A];
                w.SetValue(region.A, def.Min + (def.Max - def.Min) * t);
                _model.MarkModified();
            }
        }
        else if (Popover == OverlayPopoverKind.Style)
        {
            var g = _model.Globals;
            switch (region.A)
            {
                case 0:
                    g.Opacity = OverlayGlobals.MinOpacity + (1f - OverlayGlobals.MinOpacity) * t;
                    g.Opacity = MathF.Round(g.Opacity * 100f) / 100f;
                    break;
                case 1:
                    g.Scale = OverlayGlobals.MinScale + (OverlayGlobals.MaxScale - OverlayGlobals.MinScale) * t;
                    g.Scale = MathF.Round(g.Scale * 20f) / 20f;
                    break;
                case 2:
                    g.FontSize = OverlayGlobals.MinFont + (OverlayGlobals.MaxFont - OverlayGlobals.MinFont) * t;
                    g.FontSize = MathF.Round(g.FontSize * 2f) / 2f;
                    break;
            }

            g.Clamp();
            _model.Touch();
        }

        Dirty = true;
    }

    // ---------------------------------------------------------------- drag / resize

    private void CollectEdges(WidgetInstance except)
    {
        var k = Metrics.K;
        var surfaceW = Metrics.WidthPoints / k;
        var surfaceH = Metrics.HeightPoints / k;
        var n = 0;
        _edgesLowX[n] = 0; _edgesHighX[n] = surfaceW; _edgesLowY[n] = 0; _edgesHighY[n] = surfaceH; n++;
        _edgesLowX[n] = EdgeMargin; _edgesHighX[n] = surfaceW - EdgeMargin;
        _edgesLowY[n] = EdgeMargin; _edgesHighY[n] = surfaceH - EdgeMargin; n++;
        foreach (var other in _model.Widgets)
        {
            if (ReferenceEquals(other, except) || n >= _edgesLowX.Length)
            {
                continue;
            }

            var b = Metrics.BoxOf(other);
            float x = b.X / k, y = b.Y / k, w = b.W / k, h = b.H / k;
            _edgesLowX[n] = x + w + EdgeGap;
            _edgesHighX[n] = x - EdgeGap;
            _edgesLowY[n] = y + h + EdgeGap;
            _edgesHighY[n] = y - EdgeGap;
            n++;
            if (n < _edgesLowX.Length)
            {
                // Edge-to-edge alignment with the other widget (same side).
                _edgesLowX[n] = x; _edgesHighX[n] = x + w; _edgesLowY[n] = y; _edgesHighY[n] = y + h;
                n++;
            }
        }

        _edgeCount = n;
    }

    private void BeginMove(WidgetInstance? widget, in OverlayPointerEvent e)
    {
        if (widget is null)
        {
            return;
        }

        _model.BringToFront(widget);
        SelectedWidgetId = widget.Id;
        _drag = DragKind.Move;
        _dragWidget = widget;
        _dragStartBox = Metrics.BoxOf(widget);
        _dragPointerX = e.X;
        _dragPointerY = e.Y;
        _dragMoved = false;
        CollectEdges(widget);
        Dirty = true;
    }

    private void BeginResize(WidgetInstance? widget, in OverlayPointerEvent e)
    {
        if (widget is null)
        {
            return;
        }

        _model.BringToFront(widget);
        SelectedWidgetId = widget.Id;
        _drag = DragKind.Resize;
        _dragWidget = widget;
        _dragStartBox = Metrics.BoxOf(widget);
        _dragPointerX = e.X;
        _dragPointerY = e.Y;
        _dragMoved = false;
        CollectEdges(widget);
        Dirty = true;
    }

    private void ContinueDrag(in OverlayPointerEvent e)
    {
        if (_drag == DragKind.Slider)
        {
            ApplySlider(_sliderRegion, e.X);
            return;
        }

        if (_dragWidget is null)
        {
            return;
        }

        _dragMoved = true;
        var m = Metrics;
        var k = m.K;
        var snap = (e.Modifiers & OverlayModifiers.Alt) == 0;
        var surfaceW = m.WidthPoints / k;
        var surfaceH = m.HeightPoints / k;
        var box = _dragStartBox;
        float x0 = box.X / k, y0 = box.Y / k, w0 = box.W / k, h0 = box.H / k;
        var dx = (e.X - _dragPointerX) / k;
        var dy = (e.Y - _dragPointerY) / k;
        GuideX = GuideY = float.NaN;
        if (_drag == DragKind.Move)
        {
            float nx = x0 + dx, ny = y0 + dy;
            if (snap)
            {
                var rx = OverlaySnap.SnapMove(nx, w0, surfaceW, _edgesLowX.AsSpan(0, _edgeCount), _edgesHighX.AsSpan(0, _edgeCount), SnapDistance, GridSize);
                var ry = OverlaySnap.SnapMove(ny, h0, surfaceH, _edgesLowY.AsSpan(0, _edgeCount), _edgesHighY.AsSpan(0, _edgeCount), SnapDistance, GridSize);
                nx = rx.Position;
                ny = ry.Position;
                GuideX = rx.HasGuide ? rx.Guide * k : float.NaN;
                GuideY = ry.HasGuide ? ry.Guide * k : float.NaN;
            }

            nx = Math.Clamp(nx, 0f, MathF.Max(0f, surfaceW - w0));
            ny = Math.Clamp(ny, 0f, MathF.Max(0f, surfaceH - h0));
            _dragWidget.X = nx * k / m.WidthPoints;
            _dragWidget.Y = ny * k / m.HeightPoints;
            return;
        }

        var spec = _dragWidget.Spec;
        var us = m.UserScale;
        var nw = w0 + dx;
        var nh = h0 + dy;
        if (snap)
        {
            var rx = OverlaySnap.SnapResize(x0, nw, _edgesHighX.AsSpan(0, _edgeCount), SnapDistance, GridSize);
            var ry = OverlaySnap.SnapResize(y0, nh, _edgesHighY.AsSpan(0, _edgeCount), SnapDistance, GridSize);
            nw = rx.Position;
            nh = ry.Position;
            GuideX = rx.HasGuide ? rx.Guide * k : float.NaN;
            GuideY = ry.HasGuide ? ry.Guide * k : float.NaN;
        }

        nw = Math.Clamp(nw, spec.MinW * us, MathF.Max(spec.MinW * us, surfaceW - x0));
        nh = Math.Clamp(nh, spec.MinH * us, MathF.Max(spec.MinH * us, surfaceH - y0));
        _dragWidget.W = nw / us / OverlayMetrics.RefWidth;
        _dragWidget.H = nh / us / OverlayMetrics.RefHeight;
    }
}
