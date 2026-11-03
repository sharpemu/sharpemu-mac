// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut.Overlay;

public enum OverlayPointerKind : byte
{
    Move,
    Down,
    Up,
    Scroll,
    /// <summary>The pointer left the window.</summary>
    Leave,
}

public enum OverlayMouseButton : byte
{
    None,
    Left,
    Right,
    Middle,
}

[Flags]
public enum OverlayModifiers : byte
{
    None = 0,
    Alt = 1,
    Shift = 2,
    Control = 4,
}

/// <summary>A pointer event in surface points (top-left origin, not pixels).</summary>
public readonly record struct OverlayPointerEvent(
    OverlayPointerKind Kind,
    float X,
    float Y,
    OverlayMouseButton Button = OverlayMouseButton.Left,
    OverlayModifiers Modifiers = OverlayModifiers.None,
    float ScrollDelta = 0f);

/// <summary>The keys the overlay reacts to; the host window maps its own key codes onto these.</summary>
public enum OverlayKey : byte
{
    /// <summary>F1: show or hide the overlay (handled by the existing toggle).</summary>
    ToggleVisible,
    /// <summary>F2: next layout preset (replaces corner cycling).</summary>
    CyclePreset,
    /// <summary>F3: enter or leave edit mode.</summary>
    ToggleEdit,
    /// <summary>F4: pause or resume the history view.</summary>
    TogglePause,
    /// <summary>Left arrow while paused: older frame.</summary>
    StepBack,
    /// <summary>Right arrow while paused: newer frame.</summary>
    StepForward,
    Escape,
    /// <summary>1-4, only while editing.</summary>
    Preset1,
    Preset2,
    Preset3,
    Preset4,
}

public enum OverlayHitKind : byte
{
    None,
    WidgetBody,
    WidgetGear,
    WidgetRemove,
    WidgetResize,
    HistoryBar,
    TimelineLane,
    TopCostsSort,
    TopCostsSource,
    EditBar,
    EditAdd,
    EditPresetPrev,
    EditPresetNext,
    EditSaveAs,
    EditReset,
    EditStyle,
    EditDone,
    PopoverBody,
    PopoverToggle,
    PopoverCyclePrev,
    PopoverCycleNext,
    PopoverStepDown,
    PopoverStepUp,
    PopoverSlider,
    PopoverRemove,
    PopoverDone,
    PopoverAddItem,
    PopoverDeletePreset,
}

/// <summary>A clickable rectangle (surface points) recorded while drawing, consumed by the controller.</summary>
public struct OverlayHit
{
    public RectF Rect;
    public OverlayHitKind Kind;
    /// <summary>Widget index, setting index, catalog index or slot count, depending on the kind.</summary>
    public int A;
    public int B;
    public float F1;
    public float F2;
}

/// <summary>The hit list of the last drawn frame, in draw order (later entries are on top).</summary>
public sealed class OverlayHitMap
{
    private OverlayHit[] _items = new OverlayHit[256];
    private int _count;

    public int Count => _count;
    public ref OverlayHit this[int index] => ref _items[index];

    public void Clear() => _count = 0;

    public void Add(RectF rect, OverlayHitKind kind, int a = 0, int b = 0, float f1 = 0f, float f2 = 0f)
    {
        if (_count == _items.Length)
        {
            Array.Resize(ref _items, _items.Length * 2);
        }

        _items[_count++] = new OverlayHit { Rect = rect, Kind = kind, A = a, B = b, F1 = f1, F2 = f2 };
    }

    /// <summary>The topmost entry under the point, or -1.</summary>
    public int Find(float x, float y)
    {
        for (var i = _count - 1; i >= 0; i--)
        {
            if (_items[i].Rect.Contains(x, y))
            {
                return i;
            }
        }

        return -1;
    }
}

/// <summary>Pure snapping / grid math for dragging and resizing widgets (units are the caller's).</summary>
public static class OverlaySnap
{
    public readonly record struct Result(float Position, float Guide, bool Snapped)
    {
        public bool HasGuide => !float.IsNaN(Guide);
    }

    /// <summary>
    /// Snaps a span [position, position+size] to the nearest candidate edge or the centre of
    /// [0, max] within <paramref name="distance"/>; failing that to the grid.
    /// <paramref name="lowEdges"/> align the span's start, <paramref name="highEdges"/> its end.
    /// </summary>
    public static Result SnapMove(float position, float size, float max, ReadOnlySpan<float> lowEdges,
        ReadOnlySpan<float> highEdges, float distance, float grid)
    {
        var bestDelta = 0f;
        var bestLine = float.NaN;
        var found = false;

        void Try(float candidate, float offset)
        {
            var d = candidate - (position + offset);
            if (MathF.Abs(d) < distance && (!found || MathF.Abs(d) < MathF.Abs(bestDelta)))
            {
                found = true;
                bestDelta = d;
                bestLine = candidate;
            }
        }

        foreach (var c in lowEdges)
        {
            Try(c, 0f);
        }

        foreach (var c in highEdges)
        {
            Try(c, size);
        }

        Try(max / 2f, size / 2f);
        if (found)
        {
            return new(position + bestDelta, bestLine, true);
        }

        return grid > 0 ? new(MathF.Round(position / grid) * grid, float.NaN, false) : new(position, float.NaN, false);
    }

    /// <summary>Snaps the far edge of a resized span to a candidate edge, else to the grid; returns the new size.</summary>
    public static Result SnapResize(float start, float size, ReadOnlySpan<float> highEdges, float distance, float grid)
    {
        var edge = start + size;
        var bestDelta = 0f;
        var bestLine = float.NaN;
        var found = false;
        foreach (var c in highEdges)
        {
            var d = c - edge;
            if (MathF.Abs(d) < distance && (!found || MathF.Abs(d) < MathF.Abs(bestDelta)))
            {
                found = true;
                bestDelta = d;
                bestLine = c;
            }
        }

        if (found)
        {
            return new(size + bestDelta, bestLine, true);
        }

        return grid > 0 ? new(MathF.Round(edge / grid) * grid - start, float.NaN, false) : new(size, float.NaN, false);
    }
}
