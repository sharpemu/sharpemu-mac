// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut.Overlay;

public sealed partial class OverlayRenderer
{
    private readonly List<ThreadSample> _threads = [];

    /// <summary>A small pill button with a dropdown arrow; returns its width.</summary>
    private float DrawChip(float x, float y, ReadOnlySpan<char> text, OverlayHitKind kind, int widgetIndex)
    {
        var size = _fs * 0.85f;
        var w = Measure(text, size, true) + 22f;
        var h = LineH(size) + 1f;
        _c.FillRoundRect(x, y, w, h, h / 2f, _pal.Chip);
        Txt(text, x + 7f, y + 0.5f, size, _pal.Dim, true);
        DrawTriangle(x + w - 8.5f, y + h / 2f, 2.6f, 0, _pal.Dim);
        AddContentHit(ToPoints(x, y, w, h), kind, widgetIndex);
        return w;
    }

    // ------------------------------------------------------------------ top costs

    private void DrawTopCosts(WidgetInstance w, int widgetIndex, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        var sort = (TopCostSort)w.Choice(WidgetKeys.TopCosts.Sort);
        var source = (TopCostSource)w.Choice(WidgetKeys.TopCosts.Source);
        var options = new TopCostOptions(w.Int(WidgetKeys.TopCosts.Rows), sort, source,
            (int)w.Number(WidgetKeys.TopCosts.AvgN), (TopCostThread)w.Choice(WidgetKeys.TopCosts.Thread),
            w.Bool(WidgetKeys.TopCosts.Waits));
        _analyzer.ComputeTopCosts(v, options, _top);

        var chipsY = r.Y;
        _tb.Clear().Append("by ").Append(sort switch { TopCostSort.Total => "total time", TopCostSort.Count => "count", _ => "self-time" });
        var cw = DrawChip(r.X, chipsY, _tb.Span, OverlayHitKind.TopCostsSort, widgetIndex);
        DrawChip(r.X + cw + 5f, chipsY, source == TopCostSource.Average ? "avg" : "frame", OverlayHitKind.TopCostsSource, widgetIndex);
        if (_top.Frames > 0)
        {
            if (source == TopCostSource.Average)
            {
                _chip.Clear().Append("avg of ").AppendInt(_top.Frames).Append(" frames");
            }
            else
            {
                _chip.Clear().Append("frame #").AppendInt(_top.FrameSequence);
            }
        }

        var y = chipsY + LineH(_fs * 0.85f) + 5f;
        var rowH = LineH(_fs) + 3f;
        var fit = Math.Max(0, (int)MathF.Floor((r.Bottom - y) / rowH));
        var rows = Math.Min(Math.Min(options.MaxRows, fit), _top.Rows.Count);
        if (rows == 0)
        {
            if (v.Frame is not null && _top.Rows.Count == 0)
            {
                Txt("No cost data", r.X, y, _fs, _pal.Dim);
            }

            return;
        }

        var showPct = w.Bool(WidgetKeys.TopCosts.Pct) && sort != TopCostSort.Count;
        var showBars = w.Bool(WidgetKeys.TopCosts.Bars);
        var showCount = w.Bool(WidgetKeys.TopCosts.Count);
        const float colGap = 6f;
        var msW = _fs * 3.4f;
        var pctW = showPct ? _fs * 2.9f : 0f;
        var barW = showBars ? _fs * 3.6f : 0f;
        var cntW = showCount ? _fs * 3.4f : 0f;
        var rightX = r.Right;
        var cntX = rightX - cntW;
        var barX = cntX - (showCount ? colGap : 0f) - barW;
        var pctX = barX - (showBars ? colGap : 0f) - pctW;
        var msX = pctX - (showPct ? colGap : 0f) - msW;
        var nameX = r.X + 13f;
        var nameW = MathF.Max(10f, msX - colGap - nameX);
        var max = MathF.Max(1e-6f, _top.Rows[0].Value);
        for (var i = 0; i < rows; i++)
        {
            var row = _top.Rows[i];
            var ry = y + i * rowH;
            var color = OverlayColors.OfKind(row.Kind);
            _c.FillRoundRect(r.X, ry + rowH / 2f - 3.5f, 6, 6, 2, color);
            Txt(_v.Snapshot.NameOf(row.NameId), nameX, ry + 1f, _fs, _pal.Text, false, OverlayTextAlign.Left, nameW);
            var value = _tb.Clear();
            if (sort == TopCostSort.Count) { value.AppendCount(row.Value); } else { value.AppendMs(row.Value); }

            Txt(value.Span, msX + msW, ry + 1f, _fs, _pal.Text, false, OverlayTextAlign.Right);
            if (showPct && _top.FrameMs > 0)
            {
                var pct = _tb.Clear().AppendFixed(row.Value / _top.FrameMs * 100f, 0).Append('%');
                Txt(pct.Span, pctX + pctW, ry + 1f, _fs, _pal.Dim, false, OverlayTextAlign.Right);
            }

            if (showBars)
            {
                var by = ry + rowH / 2f - 2f;
                _c.FillRoundRect(barX, by, barW, 4f, 2f, _pal.Track);
                _c.FillRoundRect(barX, by, MathF.Max(2f, Math.Clamp(row.Value / max, 0f, 1f) * barW), 4f, 2f, color);
            }

            if (showCount && row.Count >= 0.5f)
            {
                Txt(_tb.Clear().AppendCount(row.Count).Span, cntX + cntW, ry + 1f, _fs, _pal.Dim, false, OverlayTextAlign.Right);
            }

            _c.FillRect(r.X, ry + rowH - 0.5f, r.W, HairlineWidth(), _pal.Border);
        }
    }

    // ------------------------------------------------------------------ split

    private void DrawSplit(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        if (v.Frame is null)
        {
            Txt("Waiting for frames", r.X, r.Y, _fs, _pal.Dim);
            return;
        }

        if (v.SelIndex >= 0)
        {
            _chip.Clear().Append('#').AppendInt(v.Frame.Sequence);
        }

        var a = v.Aggregate;
        var includeWaits = w.Bool(WidgetKeys.Split.Waits);
        var compact = w.Bool(WidgetKeys.Split.Compact);
        var pct = w.Choice(WidgetKeys.Split.Labels) == 1;
        Span<float> values = stackalloc float[StackOrder.Length];
        Span<OverlayColor> colors = stackalloc OverlayColor[StackOrder.Length];
        var total = 0f;
        for (var i = 0; i < StackOrder.Length; i++)
        {
            var kind = StackOrder[i];
            var ms = kind == CostKind.Wait && !includeWaits ? 0f : MathF.Max(0f, a.KindMs[(int)kind]);
            values[i] = ms;
            colors[i] = OverlayColors.OfKind(kind);
            total += ms;
        }

        var donutW = compact ? r.W : MathF.Max(60f, r.W * 0.46f);
        var radius = MathF.Min(donutW, r.H) / 2f - 2f;
        if (radius < 8f)
        {
            return;
        }

        var cx = r.X + donutW / 2f;
        var cy = r.Y + r.H / 2f;
        var thickness = MathF.Max(7f, radius * 0.34f);
        _c.Donut(cx, cy, radius, thickness, values, colors, total, _pal.Track);
        var center = _tb.Clear().AppendMs(total);
        Txt(center.Span, cx, cy - radius * 0.4f * 0.62f - 1f, MathF.Max(9f, radius * 0.4f), _pal.Text, true, OverlayTextAlign.Center);
        Txt(total >= 1000f ? "" : "ms/frame", cx, cy + radius * 0.2f, MathF.Max(7f, radius * 0.2f), _pal.Dim, false, OverlayTextAlign.Center);

        if (compact)
        {
            return;
        }

        var lx = r.X + donutW + 8f;
        var lw = r.Right - lx;
        var size = _fs * 0.95f;
        var rowH = LineH(size) + 1f;
        var rowCount = 0;
        for (var i = 0; i < StackOrder.Length; i++)
        {
            var kind = StackOrder[i];
            var primary = kind is CostKind.Guest or CostKind.Render or CostKind.Gpu or CostKind.Wait;
            if (kind == CostKind.Wait && !includeWaits)
            {
                continue;
            }

            if (!primary && values[i] <= 0.05f)
            {
                continue;
            }

            rowCount++;
        }

        var y = r.Y + MathF.Max(0f, (r.H - rowCount * rowH) / 2f);
        for (var i = 0; i < StackOrder.Length; i++)
        {
            var kind = StackOrder[i];
            var primary = kind is CostKind.Guest or CostKind.Render or CostKind.Gpu or CostKind.Wait;
            if ((kind == CostKind.Wait && !includeWaits) || (!primary && values[i] <= 0.05f))
            {
                continue;
            }

            if (y + rowH > r.Bottom + 1f)
            {
                break;
            }

            _c.FillRoundRect(lx, y + rowH / 2f - 3.5f, 7, 7, 2, colors[i]);
            Txt(OverlayColors.KindLabel(kind), lx + 11f, y, size, _pal.Dim, false, OverlayTextAlign.Left, lw - 11f - 30f);
            var value = _tb.Clear();
            if (pct) { value.AppendFixed(total > 0 ? values[i] / total * 100f : 0f, 0).Append('%'); } else { value.AppendMs(values[i]); }

            Txt(value.Span, lx + lw, y, size, _pal.Text, false, OverlayTextAlign.Right);
            y += rowH;
        }
    }

    // ------------------------------------------------------------------ cpu

    private void DrawCpu(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        if (v.Frame is null)
        {
            Txt("Waiting for frames", r.X, r.Y, _fs, _pal.Dim);
            return;
        }

        var frameMs = MathF.Max(0.01f, v.Aggregate.FrameMs);
        var maxRows = w.Int(WidgetKeys.Cpu.Rows);
        var includeWorkers = w.Bool(WidgetKeys.Cpu.Workers);
        var showTotal = w.Bool(WidgetKeys.Cpu.Total);
        var totalH = showTotal ? LineH(_fs) + 4f : 0f;
        var rowH = LineH(_fs) + 2f;
        var fit = Math.Max(0, (int)MathF.Floor((r.H - totalH) / rowH));
        var count = Math.Min(maxRows, fit);
        _analyzer.ComputeThreads(v, _threads);
        var y = r.Y;
        var nameW = r.W * 0.40f;
        var pctW = _fs * 3.2f;
        var barX = r.X + nameW + 6f;
        var barW = MathF.Max(8f, r.W - nameW - 6f - pctW - 6f);
        var drawn = 0;
        if (_threads.Count > 0)
        {
            foreach (var t in _threads)
            {
                if (drawn >= count)
                {
                    break;
                }

                if (!includeWorkers && t.Lane == OverlayLane.GuestWorkers)
                {
                    continue;
                }

                var color = t.Lane switch
                {
                    OverlayLane.Render => OverlayColors.Render,
                    OverlayLane.Gpu => OverlayColors.Gpu,
                    OverlayLane.GuestWorkers => OverlayColors.Guest.WithAlpha(0.6f),
                    _ => OverlayColors.Guest,
                };
                DrawCpuRow(y, _v.Snapshot.NameOf(t.NameId), t.BusyMs / frameMs * 100f, color, r.X, nameW, barX, barW, pctW, r.Right);
                y += rowH;
                drawn++;
            }
        }
        else
        {
            // No per-thread sampling: fall back to the lane totals of the frame.
            var a = v.Aggregate;
            ReadOnlySpan<OverlayBusyLane> lanes = [OverlayBusyLane.GuestMain, OverlayBusyLane.Workers, OverlayBusyLane.Render];
            foreach (var lane in lanes)
            {
                if (drawn >= count || (!includeWorkers && lane == OverlayBusyLane.Workers))
                {
                    continue;
                }

                DrawCpuRow(y, LaneLabels[(int)lane], a.Busy(lane) / frameMs * 100f, OverlayColors.OfLane(lane), r.X, nameW, barX, barW, pctW, r.Right);
                y += rowH;
                drawn++;
            }
        }

        if (showTotal)
        {
            var s = v.System;
            var total = _tb.Clear();
            if (double.IsFinite(s.CpuPercent))
            {
                total.AppendPercent(s.CpuPercent);
                if (double.IsFinite(s.CpuCores))
                {
                    total.Append(" · ").AppendFixed(s.CpuCores, 1).Append(" of ").AppendInt(s.LogicalCores).Append(" cores");
                }
            }
            else
            {
                total.Append('-');
            }

            var ty = r.Bottom - LineH(_fs);
            Txt("Total", r.X, ty, _fs, _pal.Dim);
            Txt(total.Span, r.Right, ty, _fs, _pal.Text, false, OverlayTextAlign.Right);
        }
    }

    private void DrawCpuRow(float y, string name, float percent, OverlayColor color, float x, float nameW, float barX, float barW,
        float pctW, float right)
    {
        Txt(name, x, y, _fs, _pal.Dim, false, OverlayTextAlign.Left, nameW);
        var cy = y + LineH(_fs) / 2f - 2f;
        _c.FillRoundRect(barX, cy, barW, 4f, 2f, _pal.Track);
        _c.FillRoundRect(barX, cy, MathF.Max(2f, Math.Clamp(percent, 0f, 100f) / 100f * barW), 4f, 2f, color);
        Txt(_tb.Clear().AppendFixed(percent, 0).Append('%').Span, right, y, _fs, _pal.Text, false, OverlayTextAlign.Right);
    }

    // ------------------------------------------------------------------ gpu

    /// <summary>Short name for where the GPU figure comes from.</summary>
    internal static string GpuSourceLabel(bool fromFrames, string systemSource)
    {
        if (fromFrames)
        {
            return OperatingSystem.IsMacOS() ? "Metal cmd buffers" : "GPU timestamps";
        }

        var comma = systemSource.IndexOf(',');
        return comma > 0 ? systemSource[..comma] : systemSource;
    }

    private void DrawGpu(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var v = _v;
        var a = v.Aggregate;
        var big = _fs * 2.35f;
        // Frame intervals when the backend reports GPU time; the OS counter otherwise.
        var fromFrames = a.FrameMs > 0 && a.Busy(OverlayBusyLane.Gpu) > 0;
        var util = fromFrames ? Math.Clamp(a.Busy(OverlayBusyLane.Gpu) / a.FrameMs * 100.0, 0, 100) : v.System.GpuPercent;
        var top = r.Y - big * 0.12f;
        var number = _tb.Clear();
        if (v.Frame is null || !double.IsFinite(util)) { number.Append("--"); } else { number.AppendFixed(util, 0); }

        var numW = Txt(number.Span, r.X, top, big, util > 92 ? OverlayColors.Mid : _pal.Text, true);
        var baseline = top + _t.BaselineOffset(big, true);
        TxtBase("%", r.X + numW + 3f, baseline, _fs, _pal.Dim);
        if (v.Frame is not null)
        {
            var ms = _tb.Clear().AppendMs(a.Busy(OverlayBusyLane.Gpu), a.Busy(OverlayBusyLane.Gpu) < 1000);
            TxtBase(ms.Span, r.Right, baseline, _fs * 1.15f, _pal.Dim, false, OverlayTextAlign.Right);
        }

        var y = r.Y + big + 4f;
        if (v.Frame is null)
        {
            return;
        }

        if (w.Bool(WidgetKeys.Gpu.Draws))
        {
            y = DrawRow(r.X, y, r.W, "Draws", _tb.Clear().AppendCount(a.Draws).Span);
            y = DrawRow(r.X, y, r.W, "Dispatches", _tb2.Clear().AppendFixed(a.Dispatches, 0).Span);
        }

        if (w.Bool(WidgetKeys.Gpu.Res))
        {
            var s = v.System;
            var res = _tb.Clear();
            if (s.RenderWidth > 0 && s.RenderHeight > 0) { res.AppendInt(s.RenderWidth).Append('×').AppendInt(s.RenderHeight); } else { res.Append('-'); }

            y = DrawRow(r.X, y, r.W, "Resolution", res.Span);
        }

        y = DrawRow(r.X, y, r.W, "Source", GpuSourceLabel(fromFrames, v.System.GpuSource));

        if (w.Bool(WidgetKeys.Gpu.Graph) && r.Bottom - y >= 14f)
        {
            var end = v.WindowEnd(90);
            var start = Math.Max(0, end - 89);
            var n = end - start + 1;
            var values = FloatScratch(n);
            for (var i = 0; i < n; i++)
            {
                var f = v.Frames[start + i];
                values[i] = f.FrameMs > 0 ? Math.Clamp(f.GpuMs / f.FrameMs * 100f, 0f, 100f) : 0f;
            }

            AreaChart(values.AsSpan(0, n), r.X, y + 1f, r.W, r.Bottom - y - 1f, 0f, 100f, OverlayColors.Gpu);
        }
    }

    // ------------------------------------------------------------------ memory

    private void DrawMemory(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var s = _v.System;
        const double gib = 1024.0 * 1024.0 * 1024.0;
        var y = r.Y;
        if (w.Bool(WidgetKeys.Memory.Proc))
        {
            var t = _tb.Clear();
            if (s.ProcessMemoryBytes is { } p) { t.AppendBytes(p); } else { t.Append("n/a"); }

            y = DrawRow(r.X, y, r.W, "Process", t.Span);
        }

        if (w.Bool(WidgetKeys.Memory.Gpu))
        {
            var t = _tb.Clear();
            if (s.GpuAllocatedBytes is { } g) { t.AppendBytes(g); } else { t.Append("n/a"); }

            y = DrawRow(r.X, y, r.W, "GPU alloc", t.Span);
        }

        if (w.Bool(WidgetKeys.Memory.Cache))
        {
            var t = _tb.Clear().AppendFixed(s.GuestImageCacheBytes / gib, 1).Append(" / ").AppendFixed(s.GuestBufferCacheBytes / gib, 1).Append(" GB");
            y = DrawRow(r.X, y, r.W, "Img / buf cache", t.Span);
        }

        if (w.Bool(WidgetKeys.Memory.Alloc))
        {
            var t = _tb.Clear();
            if (s.AllocatedMbPerSecond >= 1000) { t.AppendFixed(s.AllocatedMbPerSecond / 1000.0, 2).Append(" GB/s"); } else { t.AppendFixed(s.AllocatedMbPerSecond, 0).Append(" MB/s"); }

            y = DrawRow(r.X, y, r.W, "Alloc rate", t.Span, s.AllocatedMbPerSecond > 400 ? OverlayColors.Mid : null);
        }

        if (w.Bool(WidgetKeys.Memory.Gc))
        {
            var t = _tb.Clear().AppendInt(s.Gen0).Append(" / ").AppendInt(s.Gen1).Append(" / ").AppendInt(s.Gen2);
            y = DrawRow(r.X, y, r.W, "GC 0 / 1 / 2", t.Span);
        }

        var trend = _analyzer.MemoryTrend;
        if (w.Bool(WidgetKeys.Memory.Trend) && trend.Length >= 2 && r.Bottom - y >= 14f)
        {
            float lo = float.MaxValue, hi = float.MinValue;
            foreach (var value in trend)
            {
                lo = MathF.Min(lo, value);
                hi = MathF.Max(hi, value);
            }

            AreaChart(trend, r.X, y + 1f, r.W, r.Bottom - y - 1f, lo - 0.1f, MathF.Max(hi, lo + 0.4f), OverlayColor.Hex(0x9AA4B5));
        }
    }

    // ------------------------------------------------------------------ session

    private void DrawSession(WidgetInstance w, RectF r, ref OverlayColor chip)
    {
        var s = _v.System;
        var y = r.Y;
        if (w.Bool(WidgetKeys.Session.Elapsed))
        {
            y = DrawRow(r.X, y, r.W, "Elapsed", _tb.Clear().AppendElapsed(s.ElapsedSeconds).Span);
        }

        if (w.Bool(WidgetKeys.Session.Frame))
        {
            var t = _tb.Clear();
            if (_v.Frame is { } f) { t.Append('#').AppendInt(f.Sequence); } else { t.Append('-'); }

            y = DrawRow(r.X, y, r.W, "Frame", t.Span);
        }

        if (w.Bool(WidgetKeys.Session.Title) && s.Title.Length > 0)
        {
            y = DrawRow(r.X, y, r.W, "Title", s.Title);
        }

        if (w.Bool(WidgetKeys.Session.Backend))
        {
            y = DrawRow(r.X, y, r.W, "Backend", s.Backend.Length > 0 ? s.Backend : "-");
        }

        if (w.Bool(WidgetKeys.Session.Os))
        {
            DrawRow(r.X, y, r.W, "OS", s.Os.Length > 0 ? s.Os : "-");
        }
    }
}
