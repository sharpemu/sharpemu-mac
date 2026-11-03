// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut.Overlay;

public sealed partial class OverlayRenderer
{
    private const float PadX = 10f, PadY = 8f, HeaderH = 14f, Gap = 5f, PanelRadius = 10f;
    private const int TipLines = 7;

    private readonly TextBuf _title = new();
    private readonly TextBuf _chip = new();
    private readonly TextBuf _tb3 = new();

    // Tooltip content collected while widgets draw (hovered timeline segment / history bar).
    private readonly TextBuf[] _tipText = CreateTipBuffers();
    private readonly OverlayColor[] _tipColor = new OverlayColor[TipLines];
    private readonly bool[] _tipBold = new bool[TipLines];
    private int _tipCount;

    private static TextBuf[] CreateTipBuffers()
    {
        var buffers = new TextBuf[TipLines];
        for (var i = 0; i < buffers.Length; i++)
        {
            buffers[i] = new TextBuf();
        }

        return buffers;
    }

    private TextBuf TipLine(OverlayColor color, bool bold = false)
    {
        var index = Math.Min(_tipCount, TipLines - 1);
        if (_tipCount < TipLines)
        {
            _tipCount++;
        }

        _tipColor[index] = color;
        _tipBold[index] = bold;
        return _tipText[index].Clear();
    }

    // ------------------------------------------------------------------ widget shell

    private void DrawWidget(WidgetInstance w, int index)
    {
        var box = _m.BoxOf(w);
        var dw = w.W * OverlayMetrics.RefWidth;
        var dh = w.H * OverlayMetrics.RefHeight;
        _c.ResetState();
        _c.SetTransform(box.X * _ps, box.Y * _ps, _m.ContentScale);
        _c.ReplaceSolid = !_ctl.Edit;
        _c.FillRoundRect(0, 0, dw, dh, PanelRadius, _pal.Panel);
        _c.ReplaceSolid = false;
        _c.StrokeRoundRect(0, 0, dw, dh, PanelRadius, HairlineWidth(), _pal.Border);
        _ctl.Hits.Add(box, OverlayHitKind.WidgetBody, index);

        var edit = _ctl.Edit;
        var isStrip = w.Type == OverlayWidgetType.Strip;
        _c.PushClip(0, 0, dw, dh);
        var content = isStrip
            ? new RectF(PadX, PadY, dw - 2 * PadX, dh - 2 * PadY)
            : new RectF(PadX, PadY + HeaderH + Gap, dw - 2 * PadX, dh - (PadY + HeaderH + Gap) - PadY);

        _chip.Clear();
        _title.Clear().Append(w.Spec.Title);
        var chipColor = _pal.Dim;
        switch (w.Type)
        {
            case OverlayWidgetType.Fps: DrawFps(w, content, ref chipColor); break;
            case OverlayWidgetType.Strip: DrawStrip(w, content, ref chipColor); break;
            case OverlayWidgetType.Budget: DrawBudget(w, content, ref chipColor); break;
            case OverlayWidgetType.Timeline: DrawTimeline(w, content, ref chipColor); break;
            case OverlayWidgetType.History: DrawHistory(w, content, ref chipColor); break;
            case OverlayWidgetType.TopCosts: DrawTopCosts(w, index, content, ref chipColor); break;
            case OverlayWidgetType.Split: DrawSplit(w, content, ref chipColor); break;
            case OverlayWidgetType.Cpu: DrawCpu(w, content, ref chipColor); break;
            case OverlayWidgetType.Gpu: DrawGpu(w, content, ref chipColor); break;
            case OverlayWidgetType.Memory: DrawMemory(w, content, ref chipColor); break;
            case OverlayWidgetType.Session: DrawSession(w, content, ref chipColor); break;
        }

        if (!isStrip)
        {
            var chipW = 0f;
            if (!edit && _chip.Length > 0)
            {
                chipW = Txt(_chip.Span, dw - PadX, PadY + 1, _fs * 0.82f, chipColor, true, OverlayTextAlign.Right) + 8f;
            }

            Label(_title.Span, PadX, PadY + 1, dw - 2 * PadX - chipW - (edit ? 44f : 0f));
        }

        _c.PopClip();
        if (edit)
        {
            DrawWidgetEditAdornments(w, index, dw, dh, isStrip);
        }
    }

    /// <summary>A 1-pixel line expressed in the current transform's units.</summary>
    private float HairlineWidth() => 1f / MathF.Max(0.01f, _c.Scale) * MathF.Max(1f, MathF.Round(_ps));

    private void DrawWidgetEditAdornments(WidgetInstance w, int index, float dw, float dh, bool isStrip)
    {
        var selected = string.Equals(_ctl.SelectedWidgetId, w.Id, StringComparison.Ordinal);
        DrawDashedRect(-3, -3, dw + 6, dh + 6, selected ? 0f : 4f,
            selected ? OverlayColor.Hex(0xFFFFFF) : OverlayColor.Hex(0xFFFFFF, 0.5f));

        var top = isStrip ? 5f : 6.5f;
        var right = dw - (isStrip ? 6f : PadX);
        var removeRect = new RectF(right - 17, top, 17, 17);
        var gearRect = new RectF(right - 37, top, 17, 17);
        var hoverGear = IsHovering(OverlayHitKind.WidgetGear, index);
        var hoverRemove = IsHovering(OverlayHitKind.WidgetRemove, index);
        _c.FillRoundRect(gearRect.X, gearRect.Y, gearRect.W, gearRect.H, 5, hoverGear ? OverlayColor.Hex(0xFFFFFF, 0.28f) : _pal.Chip);
        DrawGearIcon(gearRect.X + 8.5f, gearRect.Y + 8.5f, 4.2f, _pal.Text);
        _c.FillRoundRect(removeRect.X, removeRect.Y, removeRect.W, removeRect.H, 5, hoverRemove ? OverlayColor.Hex(0xFF6B8A) : _pal.Chip);
        DrawCross(removeRect.X + 8.5f, removeRect.Y + 8.5f, 3.3f, 1.4f, hoverRemove ? OverlayColor.Hex(0x111111) : _pal.Text);
        _ctl.Hits.Add(ToPoints(gearRect.X, gearRect.Y, gearRect.W, gearRect.H), OverlayHitKind.WidgetGear, index);
        _ctl.Hits.Add(ToPoints(removeRect.X, removeRect.Y, removeRect.W, removeRect.H), OverlayHitKind.WidgetRemove, index);

        _c.FillRect(dw - 6, dh - 12, 1.6f, 7, OverlayColor.Hex(0xFFFFFF, 0.8f));
        _c.FillRect(dw - 12, dh - 6, 7, 1.6f, OverlayColor.Hex(0xFFFFFF, 0.8f));
        _ctl.Hits.Add(ToPoints(dw - 16, dh - 16, 16, 16), OverlayHitKind.WidgetResize, index);
    }

    private void DrawDashedRect(float x, float y, float w, float h, float dash, OverlayColor color)
    {
        if (dash <= 0)
        {
            _c.StrokeRoundRect(x, y, w, h, 12, 1.5f, color);
            return;
        }

        const float lw = 1.4f;
        _c.DashedHLine(x, x + w, y + lw / 2, lw, dash, dash * 0.8f, color);
        _c.DashedHLine(x, x + w, y + h - lw / 2, lw, dash, dash * 0.8f, color);
        for (var yy = y; yy < y + h; yy += dash * 1.8f)
        {
            var len = MathF.Min(dash, y + h - yy);
            _c.FillRect(x, yy, lw, len, color);
            _c.FillRect(x + w - lw, yy, lw, len, color);
        }
    }

    private void DrawGearIcon(float cx, float cy, float r, OverlayColor color)
    {
        _c.StrokeRoundRect(cx - r, cy - r, r * 2, r * 2, r, 1.7f, color);
        for (var i = 0; i < 8; i++)
        {
            var a = i * MathF.PI / 4f;
            var dx = MathF.Cos(a);
            var dy = MathF.Sin(a);
            _c.Line(cx + dx * (r + 0.4f), cy + dy * (r + 0.4f), cx + dx * (r + 2.1f), cy + dy * (r + 2.1f), i % 2 == 0 ? 1.9f : 1.5f, color);
        }
    }

    private void DrawCross(float cx, float cy, float r, float width, OverlayColor color)
    {
        _c.Line(cx - r, cy - r, cx + r, cy + r, width, color);
        _c.Line(cx - r, cy + r, cx + r, cy - r, width, color);
    }

    /// <summary>Direction: 0 down, 1 up, 2 left, 3 right.</summary>
    private void DrawTriangle(float cx, float cy, float size, int direction, OverlayColor color)
    {
        var p = Points(3);
        switch (direction)
        {
            case 0:
                p[0] = new(cx - size, cy - size * 0.55f); p[1] = new(cx + size, cy - size * 0.55f); p[2] = new(cx, cy + size * 0.65f);
                break;
            case 1:
                p[0] = new(cx - size, cy + size * 0.55f); p[1] = new(cx + size, cy + size * 0.55f); p[2] = new(cx, cy - size * 0.65f);
                break;
            case 2:
                p[0] = new(cx + size * 0.55f, cy - size); p[1] = new(cx + size * 0.55f, cy + size); p[2] = new(cx - size * 0.65f, cy);
                break;
            default:
                p[0] = new(cx - size * 0.55f, cy - size); p[1] = new(cx - size * 0.55f, cy + size); p[2] = new(cx + size * 0.65f, cy);
                break;
        }

        _c.FillPolygon(p.AsSpan(0, 3), color);
    }

    // ------------------------------------------------------------------ shared pieces

    /// <summary>Widget content is clickable only outside edit mode, where the whole widget is a drag handle.</summary>
    private void AddContentHit(RectF rect, OverlayHitKind kind, int a = 0, int b = 0, float f1 = 0f, float f2 = 0f)
    {
        if (!_ctl.Edit)
        {
            _ctl.Hits.Add(rect, kind, a, b, f1, f2);
        }
    }

    private OverlayColor FpsColor() => OverlayColors.Fps(_v.Fps, _v.TargetFps);

    private void SetPausedChip(ref OverlayColor chipColor, bool stallCheck)
    {
        var f = _v.Frame;
        if (f is null)
        {
            return;
        }

        if (_v.Paused)
        {
            _chip.Clear().Append("PAUSED #").AppendInt(f.Sequence);
            chipColor = OverlayColors.Mid;
        }
        else if (stallCheck && f.FrameMs > 400)
        {
            _chip.Clear().Append("STALL");
            chipColor = OverlayColors.Bad;
        }
    }

    /// <summary>Text positioned by its baseline instead of its line box.</summary>
    private float TxtBase(ReadOnlySpan<char> s, float x, float baseline, float size, OverlayColor color, bool bold = false,
        OverlayTextAlign align = OverlayTextAlign.Left, float max = 0f) =>
        Txt(s, x, baseline - _t.BaselineOffset(size, bold), size, color, bold, align, max);

    /// <summary>Frame-time sparkline with the budget line and the selected-frame marker.</summary>
    private void DrawSparkline(float x, float y, float w, float h, int n)
    {
        var frames = _v.Frames;
        if (_v.EndIndex < 0 || w < 8 || h < 6)
        {
            return;
        }

        var end = _v.WindowEnd(n);
        var start = Math.Max(0, end - n + 1);
        var count = end - start + 1;
        var mx = 0f;
        for (var i = start; i <= end; i++)
        {
            mx = MathF.Max(mx, frames[i].FrameMs);
        }

        var b = _v.BudgetMs;
        var yMax = Math.Clamp(mx * 1.08f, b * 1.6f, b * 4f);
        var first = end - n + 1;
        float X(int i) => x + 1 + (i - first) / (float)Math.Max(1, n - 1) * (w - 3);
        float Y(float ms) => y + h - 2 - MathF.Min(ms, yMax) / yMax * (h - 4);

        _c.DashedHLine(x, x + w, Y(b), HairlineWidth(), 3f, 3f, _pal.ChartLine);
        var color = FpsColor();
        var p = Points(count + 2);
        for (var i = start; i <= end; i++)
        {
            p[i - start] = new OverlayPoint(X(i), Y(frames[i].FrameMs));
        }

        _c.Polyline(p.AsSpan(0, count), 1.4f, color);
        if (_v.SelIndex >= start && _v.SelIndex <= end)
        {
            _c.FillRect(X(_v.SelIndex) - 0.5f, y, 1f, h, OverlayColor.Hex(0xFFFFFF));
        }

        _c.FillCircle(X(end), Y(frames[end].FrameMs), 2f, color);
    }

    private void AreaChart(ReadOnlySpan<float> values, float x, float y, float w, float h, float min, float max, OverlayColor color)
    {
        var n = values.Length;
        if (n < 2 || w < 4 || h < 4)
        {
            return;
        }

        var p = Points(n + 3);
        var range = max - min;
        if (range <= 0)
        {
            range = 1;
        }

        for (var i = 0; i < n; i++)
        {
            p[i] = new OverlayPoint(x + i / (float)(n - 1) * w, y + h - 1.5f - Math.Clamp((values[i] - min) / range, 0f, 1f) * (h - 3f));
        }

        _c.Polyline(p.AsSpan(0, n), 1.3f, color);
        p[n] = new OverlayPoint(x + w, y + h);
        p[n + 1] = new OverlayPoint(x, y + h);
        _c.FillPolygon(p.AsSpan(0, n + 2), color.WithAlpha(0.16f));
    }

    /// <summary>One label / value row (value right-aligned); returns the next row's y.</summary>
    private float DrawRow(float x, float y, float w, ReadOnlySpan<char> label, ReadOnlySpan<char> value, OverlayColor? valueColor = null)
    {
        Txt(label, x, y, _fs, _pal.Dim, false, OverlayTextAlign.Left, w * 0.55f);
        Txt(value, x + w, y, _fs, valueColor ?? _pal.Text, false, OverlayTextAlign.Right, w * 0.62f);
        return y + LineH(_fs) + 2f;
    }

    private static readonly string[] LaneLabels = ["Guest main", "Busiest worker", "Render", "GPU"];
    private static readonly string[] LaneLabelsShort = ["Main", "Worker", "Render", "GPU"];

    /// <summary>
    /// Four parallel bars (guest main, busiest worker, render, GPU) against the frame budget line.
    /// </summary>
    private void DrawLaneBars(FrameAggregate a, float x, float y, float w, float rowH, float barH, float size,
        bool labels, bool values, bool shortLabels)
    {
        var names = shortLabels ? LaneLabelsShort : LaneLabels;
        size = MathF.Min(size, rowH * 0.9f);
        var labelW = 0f;
        if (labels)
        {
            foreach (var name in names)
            {
                labelW = MathF.Max(labelW, Measure(name, size));
            }

            labelW = MathF.Min(labelW + 6f, w * 0.4f);
        }

        var valueW = values ? Measure("000.0 ms", size) + 4f : 0f;
        var barX = x + labelW;
        var barW = MathF.Max(8f, w - labelW - valueW);
        var budget = _v.BudgetMs;
        var axis = Math.Clamp(MathF.Max(budget * 1.25f, a.FrameMs * 1.04f), budget * 1.25f, budget * 4f);
        var verdict = _v.Verdict;
        for (var i = 0; i < 4; i++)
        {
            var lane = (OverlayBusyLane)i;
            var rowY = y + i * rowH;
            var barY = rowY + (rowH - barH) / 2f;
            if (labels)
            {
                TxtBase(names[i], x, rowY + rowH / 2f + size * 0.34f, size, _pal.Dim, false, OverlayTextAlign.Left, labelW);
            }

            _c.FillRoundRect(barX, barY, barW, barH, barH / 2f, _pal.Track);
            var fill = Math.Clamp(a.Busy(lane) / axis, 0f, 1f) * barW;
            if (fill > 0.4f)
            {
                _c.FillRoundRect(barX, barY, MathF.Max(fill, barH), barH, barH / 2f, OverlayColors.OfLane(lane));
            }

            if (values)
            {
                _tb.Clear().AppendMs(a.Busy(lane), true);
                var bound = verdict.Kind == OverlayVerdictKind.Bound && verdict.Lane == lane;
                TxtBase(_tb.Span, x + w, rowY + rowH / 2f + size * 0.34f, size, bound ? _pal.Text : _pal.Dim, bound, OverlayTextAlign.Right);
            }
        }

        var mark = barX + budget / axis * barW;
        _c.FillRect(mark - 0.75f, y - 2f, 1.5f, rowH * 4 + 4f, OverlayColor.Hex(0xFFFFFF, 0.85f));
    }

    private void AppendOverText(TextBuf buf, out OverlayColor color)
    {
        var a = _v.Aggregate;
        var b = _v.BudgetMs;
        if (a.FrameMs > b * 1.015f)
        {
            var over = a.FrameMs - b;
            buf.Append('+').AppendMs(over, over < 1000).Append(" over");
            color = over > b * 0.2f ? OverlayColors.Bad : OverlayColors.Mid;
        }
        else
        {
            var busiest = MathF.Max(MathF.Max(a.Busy(OverlayBusyLane.GuestMain), a.Busy(OverlayBusyLane.Workers)),
                MathF.Max(a.Busy(OverlayBusyLane.Render), a.Busy(OverlayBusyLane.Gpu)));
            buf.AppendFixed(Math.Max(0, b - busiest), 1).Append(" ms headroom");
            color = OverlayColors.Good;
        }
    }

    // ------------------------------------------------------------------ FPS

    private void DrawFps(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        SetPausedChip(ref chip, true);
        var ms = w.Choice(WidgetKeys.Fps.Units) == 1;
        var big = _fs * 2.35f;
        var col = FpsColor();
        var top = r.Y - big * 0.12f;
        var number = _tb.Clear();
        if (v.Frame is null)
        {
            number.Append("--");
        }
        else if (ms)
        {
            number.AppendMs(v.AvgFrameMs);
        }
        else
        {
            number.AppendFps(v.Fps);
        }

        var baseline = top + _t.BaselineOffset(big, true);
        var numW = Txt(number.Span, r.X, top, big, col, true);
        var unitW = TxtBase(ms ? "ms" : "fps", r.X + numW + 4f, baseline, _fs, _pal.Dim);
        if (v.Frame is not null)
        {
            var sec = _tb.Clear();
            if (ms)
            {
                sec.AppendFps(v.Fps).Append(" fps");
            }
            else
            {
                sec.AppendMs(v.AvgFrameMs, true);
            }

            // A secondary figure cut to "76.3..." reads worse than none: draw it whole or not at all.
            var sx = r.X + numW + 4f + unitW + 7f;
            if (sx + Measure(sec.Span, _fs * 1.15f) <= r.Right)
            {
                TxtBase(sec.Span, sx, baseline, _fs * 1.15f, _pal.Dim);
            }
        }

        var y = r.Y + big * 1.0f + 4f;
        var lowsSize = _fs * 0.95f;
        if (w.Bool(WidgetKeys.Fps.Lows) && v.Frame is not null)
        {
            if (r.W >= 290f)
            {
                DrawLows(ms, r.Right, baseline - _t.BaselineOffset(lowsSize), true);
            }
            else
            {
                DrawLows(ms, r.X, y, false);
                y += LineH(lowsSize) + 2f;
            }
        }

        if (w.Bool(WidgetKeys.Fps.Spark) && r.Bottom - y >= 12f)
        {
            DrawSparkline(r.X, y, r.W, r.Bottom - y, (int)w.Number(WidgetKeys.Fps.Len));
        }
    }

    /// <summary>"1% low 24.1 · 0.1% 22.0" with the figures bold.</summary>
    private void DrawLows(bool ms, float x, float y, bool rightAligned)
    {
        var size = _fs * 0.95f;
        var low1 = _tb2.Clear();
        var low01 = _tb3.Clear();
        if (ms)
        {
            low1.AppendMs(1000.0 / Math.Max(0.001, _v.Low1Fps));
            low01.AppendMs(1000.0 / Math.Max(0.001, _v.Low01Fps));
        }
        else
        {
            low1.AppendFps(_v.Low1Fps);
            low01.AppendFps(_v.Low01Fps);
        }

        const string t1 = "1% low ";
        const string t2 = "  ·  0.1% ";
        var total = Measure(t1, size) + Measure(low1.Span, size, true) + Measure(t2, size) + Measure(low01.Span, size, true);
        var cx = rightAligned ? x - total : x;
        cx += Txt(t1, cx, y, size, _pal.Dim);
        cx += Txt(low1.Span, cx, y, size, _pal.Text, true);
        cx += Txt(t2, cx, y, size, _pal.Dim);
        Txt(low01.Span, cx, y, size, _pal.Text, true);
    }

    // ------------------------------------------------------------------ strip

    private void DrawStrip(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        var ms = w.Choice(WidgetKeys.Strip.Units) == 1;
        var big = _fs * 2.35f;
        var col = FpsColor();
        var labelSize = _fs * 0.86f;
        var numTop = r.Y + LineH(labelSize) - big * 0.04f;
        var colW = Measure(v.AvgFrameMs >= 1000 ? "8.88 s" : "888.8", big, true) + 10f;
        var x = r.X;

        Label(ms ? "MS" : "FPS", x, r.Y);
        var n1 = _tb.Clear();
        if (v.Frame is null) { n1.Append("--"); }
        else if (ms) { n1.AppendMs(v.AvgFrameMs); }
        else { n1.AppendFps(v.Fps); }

        Txt(n1.Span, x, numTop, big, col, true);
        x += colW;
        var showFrame = r.W - 2 * colW >= 170f;
        if (showFrame)
        {
            Label(ms ? "FPS" : "FRAME MS", x, r.Y);
            var n2 = _tb.Clear();
            if (v.Frame is null) { n2.Append("--"); }
            else if (ms) { n2.AppendFps(v.Fps); }
            else { n2.AppendMs(v.AvgFrameMs); }

            Txt(n2.Span, x, numTop, big, _pal.Text, true);
            x += colW;
        }

        // Right block: budget header, over text and four lane bars.
        var bw = r.Right - x;
        var topH = Math.Max(Math.Max(LineH(labelSize) + big, 46f), LineH(labelSize) + 2f + 4f * 10.5f);
        if (v.Frame is not null && bw > 60f)
        {
            _tb.Clear().Append("Frame budget · ").AppendFixed(v.BudgetMs, 1).Append(" ms");
            if (v.Paused)
            {
                _tb.Append(" · PAUSED #").AppendInt(v.Frame.Sequence);
            }

            var over = _tb2.Clear();
            AppendOverText(over, out var overColor);
            var overW = Txt(over.Span, r.Right, r.Y, _fs * 0.95f, overColor, false, OverlayTextAlign.Right);
            Label(_tb.Span, x, r.Y, MathF.Max(10f, bw - overW - 8f));
            var barsY = r.Y + LineH(labelSize) + 2f;
            var rowH = Math.Clamp((topH - LineH(labelSize) - 2f) / 4f, 8f, 12f);
            DrawLaneBars(v.Aggregate, x, barsY, bw, rowH, 4f, _fs * 0.76f, w.Bool(WidgetKeys.Strip.Legend), w.Bool(WidgetKeys.Strip.Legend), true);
        }

        var y = r.Y + topH + 4f;
        var statH = w.Bool(WidgetKeys.Strip.Stats) ? LineH(_fs * 0.95f) : 0f;
        var sparkH = r.Bottom - y - (statH > 0 ? statH + 3f : 0f);
        if (w.Bool(WidgetKeys.Strip.Spark) && sparkH >= 12f)
        {
            DrawSparkline(r.X, y, r.W, sparkH, (int)w.Number(WidgetKeys.Strip.Len));
            y += sparkH + 3f;
        }

        if (statH > 0 && r.Bottom - y >= statH - 1f && v.Frame is not null)
        {
            DrawStripStats(r.X, r.Bottom - statH, r.W);
        }
    }

    private void DrawStripStats(float x, float y, float w)
    {
        var size = _fs * 0.95f;
        var s = _v.System;
        var a = _v.Aggregate;
        var cx = x;
        void Item(ReadOnlySpan<char> key, ReadOnlySpan<char> value)
        {
            if (cx - x > w - 30f)
            {
                return;
            }

            cx += Txt(key, cx, y, size, _pal.Dim) + 4f;
            cx += Txt(value, cx, y, size, _pal.Text, true) + 12f;
        }

        var tb = _tb.Clear();
        Item("CPU", double.IsFinite(s.CpuPercent) ? tb.AppendPercent(s.CpuPercent).Span : "-");
        tb = _tb.Clear();
        var gpu = a.FrameMs > 0 ? a.Busy(OverlayBusyLane.Gpu) / a.FrameMs * 100.0 : s.GpuPercent;
        Item("GPU", tb.AppendPercent(gpu).Span);
        tb = _tb.Clear();
        Item("MEM", s.ProcessMemoryBytes is { } mem ? tb.AppendBytes(mem).Span : "-");
        tb = _tb.Clear();
        Item("1% LOW", tb.AppendFps(_v.Low1Fps).Span);
        tb = _tb.Clear();
        Item("DRAWS", tb.AppendCount(a.Draws).Append("/f").Span);
    }

    // ------------------------------------------------------------------ budget

    private void DrawBudget(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        _title.Clear().Append("Frame budget · ").AppendFixed(v.BudgetMs, 1).Append(" ms");
        if (v.Frame is null)
        {
            Txt("Waiting for frames", r.X, r.Y, _fs, _pal.Dim);
            return;
        }

        var a = v.Aggregate;
        var line1 = _tb.Clear().Append("frame ").AppendMs(a.FrameMs, a.FrameMs < 1000);
        if (v.SelIndex >= 0)
        {
            line1.Append(" · #").AppendInt(v.Frame.Sequence);
        }

        var over = _tb2.Clear();
        AppendOverText(over, out var overColor);
        Txt(line1.Span, r.X, r.Y, _fs, _pal.Text);
        Txt(over.Span, r.Right, r.Y, _fs, overColor, false, OverlayTextAlign.Right);

        var verdictH = w.Bool(WidgetKeys.Budget.Verdict) ? LineH(_fs * 0.95f) : 0f;
        var y = r.Y + LineH(_fs) + 3f;
        var availableForBars = r.Bottom - y - (verdictH > 0 ? verdictH + 3f : 0f);
        if (availableForBars < 28f && verdictH > 0)
        {
            verdictH = 0f;
            availableForBars = r.Bottom - y;
        }

        var rowH = Math.Clamp(availableForBars / 4f, 8f, 16f);
        var barH = MathF.Min(rowH - 3f, 8f);
        DrawLaneBars(a, r.X, y, r.W, rowH, MathF.Max(3f, barH), _fs * 0.92f, w.Bool(WidgetKeys.Budget.Legend), w.Bool(WidgetKeys.Budget.Legend), false);
        if (verdictH > 0)
        {
            DrawVerdict(r.X, r.Bottom - verdictH, r.W);
        }
    }

    private void DrawVerdict(float x, float y, float width)
    {
        var size = _fs * 0.95f;
        var vd = _v.Verdict;
        var cx = x + Txt("bottleneck: ", x, y, size, _pal.Dim);
        OverlayColor color;
        _tb2.Clear();
        switch (vd.Kind)
        {
            case OverlayVerdictKind.Bound:
                _tb2.Append("bound by ").Append(vd.Lane switch
                {
                    OverlayBusyLane.GuestMain => "guest main",
                    OverlayBusyLane.Workers => "workers",
                    OverlayBusyLane.Render => "render",
                    _ => "GPU",
                });
                color = OverlayColors.OfLane(vd.Lane).WithAlpha(1f);
                break;
            case OverlayVerdictKind.Stall:
                _tb2.Append("loading stall (I/O)");
                color = OverlayColors.Other;
                break;
            case OverlayVerdictKind.Limiter:
                _tb2.Append("none (frame limiter)");
                color = OverlayColors.Good;
                break;
            case OverlayVerdictKind.Sync:
                _tb2.Append("sync / waits");
                color = OverlayColors.Wait;
                break;
            default:
                _tb2.Append("-");
                color = _pal.Dim;
                break;
        }

        cx += Txt(_tb2.Span, cx, y, size, color, true) + 7f;
        var ratios = _tb3.Clear().Append("G ").AppendFixed(vd.GuestRatio * 100, 0).Append("%  W ").AppendFixed(vd.WorkerRatio * 100, 0)
            .Append("%  R ").AppendFixed(vd.RenderRatio * 100, 0).Append("%  GPU ").AppendFixed(vd.GpuRatio * 100, 0).Append('%');
        Txt(ratios.Span, cx, y, size * 0.95f, _pal.Dim, false, OverlayTextAlign.Left, MathF.Max(0f, x + width - cx));
    }
}
