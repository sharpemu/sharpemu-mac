// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut.Overlay;

public sealed partial class OverlayRenderer
{
    private static readonly CostKind[] StackOrder =
        [CostKind.Guest, CostKind.Hle, CostKind.Render, CostKind.Shader, CostKind.Gpu, CostKind.Other, CostKind.Wait];

    private static readonly OverlayLane[] TimelineLanes =
        [OverlayLane.GuestMain, OverlayLane.GuestWorkers, OverlayLane.Render, OverlayLane.Gpu];

    /// <summary>One row of kind swatches; entries that no longer fit are dropped. Returns the used height.</summary>
    private float DrawKindLegend(float x, float y, float width, ReadOnlySpan<CostKind> kinds)
    {
        var size = _fs * 0.9f;
        var cx = x;
        foreach (var kind in kinds)
        {
            var label = OverlayColors.KindLabel(kind);
            var w = Measure(label, size) + 7f + 4f + 9f;
            if (cx + w > x + width + 0.5f)
            {
                break;
            }

            _c.FillRoundRect(cx, y + LineH(size) / 2f - 3.5f, 7, 7, 2, OverlayColors.OfKind(kind));
            Txt(label, cx + 11f, y, size, _pal.Dim);
            cx += w;
        }

        return LineH(size);
    }

    private static readonly string[] NamePrefixes = ["Command", "sceKernel", "scePthread"];

    private static ReadOnlySpan<char> ShortSegmentName(string name)
    {
        // Prefixes carry no information in a narrow segment.
        foreach (var prefix in NamePrefixes)
        {
            if (name.Length > prefix.Length + 2 && name.StartsWith(prefix, StringComparison.Ordinal))
            {
                return name.AsSpan(prefix.Length);
            }
        }

        return name;
    }

    // ------------------------------------------------------------------ timeline

    private void DrawTimeline(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        var f = v.Frame;
        if (f is null)
        {
            _title.Clear().Append("Last frame · threads");
            Txt("Waiting for frames", r.X, r.Y, _fs, _pal.Dim);
            return;
        }

        if (v.Paused || v.SelIndex >= 0)
        {
            _title.Clear().Append("Frame #").AppendInt(f.Sequence).Append(" · threads");
        }
        else
        {
            _title.Clear().Append("Last frame · threads");
        }

        _chip.Clear().AppendMs(f.FrameMs, f.FrameMs < 1000);
        if (v.FrameVsAverage >= 1.5f || v.FrameVsAverage <= 0.67f)
        {
            _chip.Append(" · ").AppendFixed(v.FrameVsAverage, 1).Append("× avg");
            chip = v.FrameVsAverage >= 1.5f ? OverlayColors.Mid : _pal.Dim;
        }

        var legend = w.Bool(WidgetKeys.Timeline.Legend);
        var legendH = legend ? LineH(_fs * 0.9f) : 0f;
        var bodyH = r.H - (legend ? legendH + Gap : 0f);
        if (f.Segments.Length == 0)
        {
            Txt("No timeline data", r.X, r.Y, _fs, _pal.Dim);
        }
        else if (bodyH >= 30f)
        {
            DrawTimelineBody(w, f, new RectF(r.X, r.Y, r.W, bodyH));
        }

        if (legend && bodyH >= 30f)
        {
            DrawKindLegend(r.X, r.Bottom - legendH, r.W, StackOrder);
        }
    }

    private void DrawTimelineBody(WidgetInstance w, FrameRecord f, RectF r)
    {
        var laneCount = 0;
        Span<OverlayLane> lanes = stackalloc OverlayLane[4];
        foreach (var lane in TimelineLanes)
        {
            if (lane == OverlayLane.GuestWorkers && !w.Bool(WidgetKeys.Timeline.Workers))
            {
                continue;
            }

            lanes[laneCount++] = lane;
        }

        var labelSize = Math.Clamp(_fs * 0.78f, 7f, 10f);
        var gx = Math.Clamp(Measure("Workers × 13", labelSize) + 8f, 50f, 100f);
        const float pr = 6f, top = 3f, axisH = 13f;
        var pw = r.W - gx - pr;
        var laneH = MathF.Max(8f, (r.H - top - axisH) / laneCount - 3f);
        var total = MathF.Max(0.001f, f.FrameMs);
        float X(float ms) => r.X + gx + Math.Clamp(ms / total, 0f, 1f) * pw;
        var hasPointer = LocalPointer(out var px, out var py);
        var hoverSeg = -1;
        var hoverRect = default(RectF);
        var showLabels = w.Bool(WidgetKeys.Timeline.Labels);

        for (var li = 0; li < laneCount; li++)
        {
            var lane = lanes[li];
            var y = r.Y + top + li * (laneH + 3f);
            var label = _tb.Clear();
            switch (lane)
            {
                case OverlayLane.GuestMain: label.Append("Guest main"); break;
                case OverlayLane.GuestWorkers:
                    label.Append("Workers");
                    if (f.ActiveWorkers > 0)
                    {
                        label.Append(" ×").AppendInt(f.ActiveWorkers);
                    }

                    break;
                case OverlayLane.Render: label.Append("Render"); break;
                default: label.Append("GPU"); break;
            }

            TxtBase(label.Span, r.X, y + laneH / 2f + labelSize * 0.34f, labelSize, _pal.Dim, true, OverlayTextAlign.Left, gx - 4f);
            _c.FillRect(r.X + gx, y, pw, laneH, OverlayColor.Hex(0x7F7F7F, 0.12f));
            var segs = f.Segments;
            for (var i = 0; i < segs.Length; i++)
            {
                ref readonly var s = ref segs[i];
                if (s.Lane != lane || s.EndMs <= s.StartMs)
                {
                    continue;
                }

                var sx = X(s.StartMs);
                var sw = MathF.Max(0.7f, X(s.EndMs) - sx);
                var color = OverlayColors.OfKind(s.Kind);
                var alpha = s.Kind == CostKind.Wait ? 0.8f : lane == OverlayLane.GuestWorkers ? 0.85f : 1f;
                _c.FillRect(sx, y, sw, laneH, color.WithAlpha(alpha));
                if (hasPointer && px >= sx && px <= sx + MathF.Max(sw, 3f) && py >= y && py <= y + laneH)
                {
                    hoverSeg = i;
                    hoverRect = new RectF(sx, y, MathF.Max(sw, 1f), laneH);
                }

                if (showLabels && sw > 30f && laneH >= 11f)
                {
                    var name = ShortSegmentName(_v.Snapshot.NameOf(s.NameId));
                    if (Measure(name, labelSize, true) + 8f <= sw)
                    {
                        TxtBase(name, sx + 4f, y + laneH / 2f + labelSize * 0.34f, labelSize, OverlayColor.Hex(0x0A0C10, 0.82f), true);
                    }
                }
            }
        }

        var bottom = r.Y + top + laneCount * (laneH + 3f);
        _c.FillRect(r.X + gx, r.Y, HairlineWidth(), bottom - r.Y, _pal.ChartLine);
        _c.FillRect(r.X + gx + pw, r.Y, HairlineWidth(), bottom - r.Y, _pal.ChartLine);
        if (hoverSeg >= 0)
        {
            _c.StrokeRoundRect(hoverRect.X - 1, hoverRect.Y - 1, hoverRect.W + 2, hoverRect.H + 2, 1, 1.2f, OverlayColor.Hex(0xFFFFFF));
            BuildSegmentTooltip(f, hoverSeg);
        }

        // Time axis with ticks at a readable spacing.
        ReadOnlySpan<float> steps = [0.5f, 1f, 2f, 5f, 10f, 20f, 50f, 100f, 200f, 500f, 1000f, 2000f, 5000f];
        var step = steps[^1];
        foreach (var s in steps)
        {
            if (pw / (total / s) >= 34f)
            {
                step = s;
                break;
            }
        }

        var axisSize = Math.Clamp(_fs * 0.74f, 7f, 9f);
        for (var t = 0f; t <= total + 1e-4f; t += step)
        {
            var x = X(t);
            _c.FillRect(x - 0.4f, bottom, HairlineWidth(), 3f, _pal.ChartLine);
            if (t + step * 0.4f < total && x < r.Right - 90f)
            {
                var text = _tb.Clear();
                if (t >= 1000f) { text.AppendFixed(t / 1000f, 0).Append('s'); } else { text.AppendFixed(t, step < 1 ? 1 : 0); }

                Txt(text.Span, x, bottom + 2f, axisSize, _pal.Dim, false, t == 0 ? OverlayTextAlign.Left : OverlayTextAlign.Center);
            }
        }

        var present = _tb.Clear().Append("present ").AppendMs(f.FrameMs, f.FrameMs < 1000);
        Txt(present.Span, r.Right, bottom + 2f, axisSize, _pal.Dim, false, OverlayTextAlign.Right);
        AddContentHit(ToPoints(r.X + gx, r.Y, pw, bottom - r.Y), OverlayHitKind.TimelineLane);
    }

    private void BuildSegmentTooltip(FrameRecord f, int segmentIndex)
    {
        var s = f.Segments[segmentIndex];
        var color = OverlayColors.OfKind(s.Kind);
        var duration = s.EndMs - s.StartMs;
        TipLine(_pal.Text, true).Append(_v.Snapshot.NameOf(s.NameId)).Append("   ").AppendMs(duration, duration < 1000);
        var sub = TipLine(OverlayColors.OfKind(s.Kind));
        sub.Append(s.Lane switch
        {
            OverlayLane.GuestMain => "Guest main",
            OverlayLane.GuestWorkers => "Workers",
            OverlayLane.Render => "Render thread",
            _ => "GPU",
        }).Append(" · ").Append(OverlayColors.KindLabel(s.Kind));
        _ = color;
        if (f.FrameMs > 0)
        {
            TipLine(_pal.Dim).AppendFixed(duration / f.FrameMs * 100f, 1).Append("% of frame");
        }
    }

    // ------------------------------------------------------------------ history

    private void DrawHistory(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        var n = (int)w.Number(WidgetKeys.History.Len);
        _title.Clear().Append("Frame time · ").AppendInt(n).Append(" frames");
        if (v.SelIndex >= 0 && v.Frame is not null)
        {
            _chip.Clear().Append('#').AppendInt(v.Frame.Sequence).Append(" selected");
            chip = OverlayColors.Mid;
        }

        if (v.EndIndex < 0)
        {
            Txt("Waiting for frames", r.X, r.Y, _fs, _pal.Dim);
            return;
        }

        var legend = w.Bool(WidgetKeys.History.Legend);
        var legendH = legend ? LineH(_fs * 0.9f) : 0f;
        var chartH = r.H - (legend ? legendH + Gap : 0f);
        if (chartH < 24f)
        {
            return;
        }

        var frames = v.Frames;
        var end = v.WindowEnd(n);
        var s0 = end - n + 1;
        var b = v.BudgetMs;
        var axisSize = Math.Clamp(_fs * 0.74f, 7f, 9f);
        var padL = Measure("000", axisSize) + 6f;
        const float padT = 3f, padB = 2f;
        var pw = r.W - padL - 2f;
        var ph = chartH - padT - padB;
        var bw = pw / n;
        var mx = 0f;
        for (var k = Math.Max(0, s0); k <= end; k++)
        {
            mx = MathF.Max(mx, frames[k].FrameMs);
        }

        var viewMode = w.Choice(WidgetKeys.History.View);
        var yMax = w.Choice(WidgetKeys.History.YMax) == 1 ? b * 2f : Math.Clamp(mx * 1.08f, b * 1.5f, b * 5f);
        var fMax = 1f;
        if (viewMode == 2)
        {
            var m = 0f;
            for (var k = Math.Max(0, s0); k <= end; k++)
            {
                m = MathF.Max(m, 1000f / MathF.Max(0.01f, frames[k].FrameMs));
            }

            fMax = MathF.Max(m * 1.08f, (float)v.TargetFps * 1.2f);
        }

        var x0 = r.X + padL;
        var y0 = r.Y + padT;
        float YMs(float ms) => y0 + ph - MathF.Min(ms, yMax) / yMax * ph;
        float YFps(float f) => y0 + ph - MathF.Min(f, fMax) / fMax * ph;

        var hasPointer = LocalPointer(out var px, out var py);
        var hoverIdx = -1;
        if (hasPointer && px >= x0 && px < x0 + pw && py >= r.Y && py <= r.Y + chartH)
        {
            var slot = (int)MathF.Floor((px - x0) / bw);
            var idx = end - (n - 1 - slot);
            if (slot >= 0 && slot < n && idx >= 0 && idx <= v.Last)
            {
                hoverIdx = idx;
            }
        }

        if (viewMode is 0 or 3)
        {
            for (var k = Math.Max(0, s0); k <= end; k++)
            {
                var fr = frames[k];
                var x = x0 + (k - s0) * bw;
                var barW = MathF.Max(0.8f, bw - (bw > 3f ? 0.6f : 0.2f));
                if (viewMode == 0)
                {
                    DrawBoundBar(fr, x, barW, y0, ph, yMax);
                }
                else
                {
                    DrawStackedBar(fr, x, barW, y0, ph, yMax);
                }

                if (fr.FrameMs > yMax)
                {
                    _c.FillRect(x, y0, barW, 1.5f, OverlayColor.Hex(0xFFFFFF));
                }

                if (fr.Hitch)
                {
                    _c.FillRect(x, r.Y - 3f, MathF.Max(1.2f, barW), 2.2f, OverlayColors.Bad);
                }
            }
        }
        else
        {
            var count = end - Math.Max(0, s0) + 1;
            var p = Points(count + 1);
            for (var k = Math.Max(0, s0); k <= end; k++)
            {
                var fr = frames[k];
                p[k - Math.Max(0, s0)] = new OverlayPoint(x0 + (k - s0 + 0.5f) * bw,
                    viewMode == 2 ? YFps(1000f / MathF.Max(0.01f, fr.FrameMs)) : YMs(fr.FrameMs));
            }

            _c.Polyline(p.AsSpan(0, count), 1.4f, FpsColor());
            for (var k = Math.Max(0, s0); k <= end; k++)
            {
                if (frames[k].Hitch)
                {
                    _c.FillRect(x0 + (k - s0) * bw, r.Y - 3f, MathF.Max(1.2f, bw - 0.4f), 2.2f, OverlayColors.Bad);
                }
            }
        }

        var budgetY = viewMode == 2 ? YFps((float)v.TargetFps) : YMs(b);
        _c.DashedHLine(x0, r.Right, budgetY, HairlineWidth(), 3f, 3f, _pal.ChartLine);
        var tb = _tb.Clear();
        if (viewMode == 2) { tb.AppendFixed(v.TargetFps, 0); } else { tb.AppendFixed(b, 1); }

        var lineH = LineH(axisSize);
        var budgetLabelY = Math.Clamp(budgetY - lineH / 2f, y0, y0 + ph - lineH);
        Txt(tb.Span, r.X, budgetLabelY, axisSize, _pal.Dim);
        // The scale's ends are labelled only where the budget label leaves room for them.
        if (budgetLabelY - y0 >= lineH)
        {
            tb = _tb.Clear();
            tb.AppendFixed(viewMode == 2 ? fMax : yMax, 0);
            Txt(tb.Span, r.X, y0 - 1f, axisSize, _pal.Dim);
        }

        var zeroY = y0 + ph - lineH + 1f;
        if (zeroY - budgetLabelY >= lineH)
        {
            Txt("0", r.X, zeroY, axisSize, _pal.Dim);
        }

        if (hoverIdx >= 0)
        {
            DrawHistoryHighlight(hoverIdx, s0, x0, bw, y0, ph, 0.10f, false);
            BuildFrameTooltip(frames[hoverIdx]);
        }

        if (v.SelIndex >= 0)
        {
            DrawHistoryHighlight(v.SelIndex, s0, x0, bw, y0, ph, 0.20f, true);
        }

        AddContentHit(ToPoints(x0, r.Y, pw, chartH), OverlayHitKind.HistoryBar, n, end, ToPoints(x0, 0, 0, 0).X, bw * _c.Scale / _ps);
        if (legend)
        {
            DrawKindLegend(r.X, r.Bottom - legendH, r.W, StackOrder);
        }
    }

    private void DrawHistoryHighlight(int idx, int s0, float x0, float bw, float y0, float ph, float alpha, bool outline)
    {
        if (idx < s0)
        {
            return;
        }

        var x = x0 + (idx - s0) * bw;
        _c.FillRect(x - 0.5f, y0, bw + 1f, ph, OverlayColor.Hex(0xFFFFFF, alpha));
        if (outline)
        {
            _c.StrokeRoundRect(x - 0.5f, y0 + 0.5f, MathF.Max(bw, 2f) + 1f, ph - 1f, 0f, 1.2f, OverlayColor.Hex(0xFFFFFF));
        }
    }

    private readonly float[] _boundKinds = new float[(int)CostKind.Count];

    /// <summary>
    /// A bar of the frame's height: the slowest lane's busy time split by kind, then the rest of the
    /// frame (time that lane spent waiting on the others or the pacer) as sync / wait.
    /// </summary>
    private void DrawBoundBar(FrameRecord fr, float x, float barW, float y0, float ph, float yMax)
    {
        OverlayAnalyzer.BoundBreakdown(fr, _boundKinds);
        var bottom = y0 + ph;
        var pxPerMs = ph / yMax;
        var y = bottom;
        foreach (var kind in StackOrder)
        {
            var h = _boundKinds[(int)kind] * pxPerMs;
            if (h <= 0.05f)
            {
                continue;
            }

            var ny = MathF.Max(y0, y - h);
            var color = OverlayColors.OfKind(kind);
            _c.FillRect(x, ny, barW, y - ny, kind == CostKind.Wait ? color.WithAlpha(0.45f) : color);
            y = ny;
            if (y <= y0)
            {
                break;
            }
        }
    }

    /// <summary>A bar of the frame's height split by each kind's share of the frame's work.</summary>
    private void DrawStackedBar(FrameRecord fr, float x, float barW, float y0, float ph, float yMax)
    {
        var kinds = fr.KindMs;
        var sum = 0f;
        for (var k = 0; k < kinds.Length; k++)
        {
            sum += MathF.Max(0f, kinds[k]);
        }

        var bottom = y0 + ph;
        var heightTotal = MathF.Min(fr.FrameMs, yMax) / yMax * ph;
        if (sum <= 0f)
        {
            _c.FillRect(x, bottom - heightTotal, barW, heightTotal, OverlayColors.Other.WithAlpha(0.7f));
            return;
        }

        var y = bottom;
        foreach (var kind in StackOrder)
        {
            var share = MathF.Max(0f, kinds[(int)kind]) / sum;
            var h = share * heightTotal;
            if (h <= 0.05f)
            {
                continue;
            }

            var ny = MathF.Max(y0, y - h);
            _c.FillRect(x, ny, barW, y - ny, OverlayColors.OfKind(kind));
            y = ny;
            if (y <= y0)
            {
                break;
            }
        }
    }

    private void BuildFrameTooltip(FrameRecord fr)
    {
        var head = TipLine(_pal.Text, true);
        head.Append('#').AppendInt(fr.Sequence).Append("   ").AppendMs(fr.FrameMs, fr.FrameMs < 1000).Append(" · ")
            .AppendFps(fr.FrameMs > 0 ? 1000.0 / fr.FrameMs : 0).Append(" fps");
        if (fr.Hitch)
        {
            TipLine(OverlayColors.Bad).Append("hitch: ").Append(string.IsNullOrEmpty(fr.HitchReason) ? "slow frame" : fr.HitchReason);
        }

        var lane = OverlayAnalyzer.BoundBreakdown(fr, _boundKinds);
        TipLine(_pal.Text).Append("bound by ").Append(OverlayAnalyzer.BusyLaneLabel(lane)).Append("  ")
            .AppendMs(fr.FrameMs - _boundKinds[(int)CostKind.Wait]);
        var shown = 0;
        foreach (var kind in StackOrder)
        {
            var ms = _boundKinds[(int)kind];
            if (ms <= 0.04f || shown >= 3)
            {
                continue;
            }

            TipLine(OverlayColors.OfKind(kind)).Append(OverlayColors.KindLabel(kind)).Append("  ").AppendMs(ms);
            shown++;
        }

        TipLine(_pal.Dim).Append("click to inspect");
    }
}
