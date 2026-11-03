// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.VideoOut.Overlay;

public sealed partial class OverlayRenderer
{
    private static readonly OverlayColor ChromeBg = OverlayColor.Hex(0x0E1014, 0.88f);
    private static readonly OverlayColor PopoverBg = OverlayColor.Hex(0x121419, 1f);
    private static readonly OverlayColor ChromeLine = OverlayColor.Hex(0xFFFFFF, 0.14f);
    private static readonly OverlayColor ChromeText = OverlayColor.Hex(0xE8EAEE);
    private static readonly OverlayColor ChromeDim = OverlayColor.Hex(0x8B93A1);
    private static readonly OverlayColor ChromeLabel = OverlayColor.Hex(0xC9CED6);
    private static readonly OverlayColor ButtonBg = OverlayColor.Hex(0xFFFFFF, 0.10f);
    private static readonly OverlayColor ButtonHover = OverlayColor.Hex(0xFFFFFF, 0.20f);

    private float _cs = 1f;
    private OverlayHit _hover;
    private bool _hasHover;

    private float ChromeW => _frame!.Width / _cs;
    private float ChromeH => _frame!.Height / _cs;

    private RectF ChromePt(float x, float y, float w, float h) => new(x * _cs / _ps, y * _cs / _ps, w * _cs / _ps, h * _cs / _ps);

    private void PrepareChrome()
    {
        _hasHover = _ctl.HoverHit >= 0 && _ctl.HoverHit < _ctl.Hits.Count;
        _hover = _hasHover ? _ctl.Hits[_ctl.HoverHit] : default;
        _tipCount = 0;
        _cs = _ps * Math.Clamp(_m.WidthPoints / 1280f, 0.85f, 1.5f);
        if (_ctl.Edit)
        {
            _cs = MathF.Min(_cs, _frame!.Width / 580f);
        }

        _cs = MathF.Max(_cs, 0.5f);
    }

    private bool IsHovering(OverlayHitKind kind, int a = -1) =>
        _hasHover && _hover.Kind == kind && (a < 0 || _hover.A == a);

    // ------------------------------------------------------------------ regions

    private void AddChromeRegions()
    {
        if (_ctl.Paused)
        {
            AddRegionPx(_frame!.Width / 2f - 240f * _cs, 6f * _cs, 480f * _cs, 34f * _cs);
        }

        if (_ctl.ToastActive(_now))
        {
            AddRegionPx(_frame!.Width / 2f - 240f * _cs, 40f * _cs, 480f * _cs, 34f * _cs);
        }

        if (_hasHover && _hover.Kind is OverlayHitKind.HistoryBar or OverlayHitKind.TimelineLane)
        {
            AddRegionPx(_ctl.PointerX * _ps - 360f * _cs, _ctl.PointerY * _ps - 180f * _cs, 720f * _cs, 360f * _cs);
        }
    }

    // ------------------------------------------------------------------ backdrop (edit mode)

    private void DrawEditBackdrop()
    {
        _c.ResetState();
        var step = OverlayController.GridSize * _m.K * _ps;
        var dot = MathF.Max(1.6f, _ps * 1.3f);
        var color = OverlayColor.Hex(0xFFFFFF, 0.26f);
        if (step >= 5f)
        {
            for (var y = step; y < _frame!.Height; y += step)
            {
                for (var x = step; x < _frame.Width; x += step)
                {
                    _c.FillRect(x - dot / 2f, y - dot / 2f, dot, dot, color);
                }
            }
        }

        _c.StrokeRoundRect(0, 0, _frame!.Width, _frame.Height, 12f * _cs, MathF.Max(1.5f, 1.5f * _ps), OverlayColor.Hex(0xFFFFFF, 0.28f));
    }

    // ------------------------------------------------------------------ chrome

    private void DrawChrome()
    {
        _c.ResetState();
        if (_ctl.Edit)
        {
            if (!float.IsNaN(_ctl.GuideX))
            {
                _c.FillRect(_ctl.GuideX * _ps - 0.5f, 0, MathF.Max(1f, _ps), _frame!.Height, OverlayColor.Hex(0xFFFFFF, 0.75f));
            }

            if (!float.IsNaN(_ctl.GuideY))
            {
                _c.FillRect(0, _ctl.GuideY * _ps - 0.5f, _frame!.Width, MathF.Max(1f, _ps), OverlayColor.Hex(0xFFFFFF, 0.75f));
            }
        }

        _c.SetTransform(0, 0, _cs);
        if (_ctl.Paused)
        {
            DrawPausePill();
        }

        if (_ctl.ToastText is { } toast && _ctl.ToastActive(_now))
        {
            var w = Measure(toast, 12f) + 26f;
            var x = (ChromeW - w) / 2f;
            // In edit mode the top of the surface holds widgets; park the toast above the edit bar instead.
            var ty = _ctl.Edit ? ChromeH - 34f - 10f - 8f - 28f : 42f;
            _c.FillRoundRect(x, ty, w, 28f, 8f, OverlayColor.Hex(0x0E1014, 0.96f));
            _c.StrokeRoundRect(x, ty, w, 28f, 8f, 1f, ChromeLine);
            Txt(toast, ChromeW / 2f, ty + 5f, 12f, ChromeText, false, OverlayTextAlign.Center);
        }

        if (_ctl.Edit)
        {
            DrawEditBar();
            if (_ctl.Popover != OverlayPopoverKind.None)
            {
                DrawPopover();
            }
        }

        if (_tipCount > 0 && !_ctl.Edit && _ctl.PointerInside)
        {
            DrawTooltip();
        }

        _c.ResetState();
    }

    private void DrawPausePill()
    {
        var text = _tb.Clear().Append("PAUSED");
        if (_v.Frame is { } f)
        {
            text.Append("   #").AppendInt(f.Sequence);
        }

        text.Append("   ·   Left / Right step   ·   F4 resume");
        var w = Measure(text.Span, 11f, true) + 44f;
        var x = (ChromeW - w) / 2f;
        _c.FillRoundRect(x, 8f, w, 26f, 13f, OverlayColor.Hex(0x0E1014, 0.84f));
        _c.StrokeRoundRect(x, 8f, w, 26f, 13f, 1f, ChromeLine);
        _c.FillRect(x + 13f, 15f, 3f, 12f, OverlayColors.Mid);
        _c.FillRect(x + 19f, 15f, 3f, 12f, OverlayColors.Mid);
        Txt(text.Span, x + 30f, 13f, 11f, ChromeText, true);
    }

    // ------------------------------------------------------------------ buttons

    private void Button(float x, float y, float w, float h, ReadOnlySpan<char> text, OverlayHitKind kind, int a = 0, bool primary = false,
        bool danger = false, bool round = true)
    {
        var hover = IsHovering(kind, a) && _hover.Rect.Contains(_ctl.PointerX, _ctl.PointerY);
        OverlayColor bg, fg;
        if (primary)
        {
            bg = hover ? OverlayColor.Hex(0xFFFFFF) : ChromeText;
            fg = OverlayColor.Hex(0x111111);
        }
        else if (danger && hover)
        {
            bg = OverlayColors.Bad;
            fg = OverlayColor.Hex(0x111111);
        }
        else
        {
            bg = hover ? ButtonHover : ButtonBg;
            fg = ChromeText;
        }

        _c.FillRoundRect(x, y, w, h, round ? h / 2f : 7f, bg);
        Txt(text, x + w / 2f, y + (h - LineH(11f)) / 2f, 11f, fg, primary, OverlayTextAlign.Center, w - 8f);
        _ctl.Hits.Add(ChromePt(x, y, w, h), kind, a);
    }

    private void ArrowButton(float x, float y, float w, float h, int direction, OverlayHitKind kind, int a)
    {
        var hover = IsHovering(kind, a) && _hover.Rect.Contains(_ctl.PointerX, _ctl.PointerY);
        _c.FillRoundRect(x, y, w, h, MathF.Min(w, h) / 2f, hover ? ButtonHover : ButtonBg);
        DrawTriangle(x + w / 2f, y + h / 2f, 3.4f, direction, ChromeText);
        _ctl.Hits.Add(ChromePt(x, y, w, h), kind, a);
    }

    // ------------------------------------------------------------------ edit bar

    private void DrawEditBar()
    {
        const float h = 34f, bh = 22f, gap = 8f, pad = 8f;
        const float addW = 96f, presetW = 176f, saveW = 74f, resetW = 54f, styleW = 54f, doneW = 56f;
        var total = addW + presetW + saveW + resetW + styleW + doneW + gap * 5 + pad * 2;
        var x = (ChromeW - total) / 2f;
        var y = ChromeH - h - 10f;
        _c.FillRoundRect(x, y, total, h, 12f, ChromeBg);
        _c.StrokeRoundRect(x, y, total, h, 12f, 1f, ChromeLine);
        _ctl.Hits.Add(ChromePt(x, y, total, h), OverlayHitKind.EditBar);
        var by = y + (h - bh) / 2f;
        var cx = x + pad;
        Button(cx, by, addW, bh, "+ Add widget", OverlayHitKind.EditAdd, 0, true);
        cx += addW + gap;
        ArrowButton(cx, by, 22f, bh, 2, OverlayHitKind.EditPresetPrev, 0);
        var name = _tb.Clear().Append(_ctl.Model.PresetName);
        if (_ctl.Model.Modified)
        {
            name.Append(" · modified");
        }

        Txt(name.Span, cx + 22f + (presetW - 44f) / 2f, by + (bh - LineH(11f)) / 2f, 11f, ChromeText, false, OverlayTextAlign.Center, presetW - 52f);
        ArrowButton(cx + presetW - 22f, by, 22f, bh, 3, OverlayHitKind.EditPresetNext, 0);
        cx += presetW + gap;
        Button(cx, by, saveW, bh, "Save as…", OverlayHitKind.EditSaveAs);
        cx += saveW + gap;
        Button(cx, by, resetW, bh, "Reset", OverlayHitKind.EditReset);
        cx += resetW + gap;
        Button(cx, by, styleW, bh, "Style", OverlayHitKind.EditStyle);
        cx += styleW + gap;
        Button(cx, by, doneW, bh, "Done", OverlayHitKind.EditDone, 0, true);
    }

    // ------------------------------------------------------------------ popovers

    private const float PopW = 276f, PopPad = 12f, RowH = 26f;

    private void DrawPopover()
    {
        var kind = _ctl.Popover;
        WidgetInstance? widget = kind == OverlayPopoverKind.WidgetSettings ? _ctl.Model.Find(_ctl.PopoverWidgetId ?? string.Empty) : null;
        if (kind == OverlayPopoverKind.WidgetSettings && widget is null)
        {
            _ctl.ClosePopover();
            return;
        }

        int rows;
        float contentH;
        switch (kind)
        {
            case OverlayPopoverKind.WidgetSettings:
                rows = widget!.Spec.Settings.Length;
                contentH = rows * RowH;
                break;
            case OverlayPopoverKind.Style:
                rows = 5;
                contentH = rows * RowH + 4f;
                break;
            default:
                rows = WidgetCatalog.All.Count;
                contentH = rows * 46f;
                break;
        }

        var footerH = kind == OverlayPopoverKind.AddWidget ? 0f : 40f;
        var headerH = 26f;
        var fullH = PopPad * 2 + headerH + contentH + footerH;
        var maxH = ChromeH - 16f;
        var h = MathF.Min(fullH, maxH);
        var anchor = _ctl.PopoverAnchor;
        var ax = anchor.X * _ps / _cs;
        var ay = anchor.Y * _ps / _cs;
        var ah = anchor.H * _ps / _cs;
        var x = Math.Clamp(ax, 8f, MathF.Max(8f, ChromeW - PopW - 8f));
        var y = ay + ah + 6f;
        if (y + h > ChromeH - 8f)
        {
            y = MathF.Max(8f, ay - h - 6f);
        }

        _c.FillRoundRect(x, y, PopW, h, 10f, PopoverBg);
        _c.StrokeRoundRect(x, y, PopW, h, 10f, 1f, ChromeLine);
        _ctl.Hits.Add(ChromePt(x, y, PopW, h), OverlayHitKind.PopoverBody);

        var title = _tb.Clear();
        switch (kind)
        {
            case OverlayPopoverKind.WidgetSettings: title.Append(widget!.Spec.Title).Append(" settings"); break;
            case OverlayPopoverKind.Style: title.Append("Overlay style"); break;
            default: title.Append("Add widget"); break;
        }

        Txt(title.Span, x + PopPad, y + PopPad - 1f, 10f, ChromeDim, true, OverlayTextAlign.Left, PopW - 2 * PopPad, true, 0.09f);

        var contentTop = y + PopPad + headerH;
        var viewH = h - PopPad * 2 - headerH - footerH;
        var scroll = 0f;
        if (kind == OverlayPopoverKind.AddWidget)
        {
            scroll = Math.Clamp(_ctl.PopoverScroll, 0f, MathF.Max(0f, contentH - viewH));
            _ctl.PopoverScroll = scroll;
        }

        _c.PushClip(x, contentTop, PopW, viewH);
        switch (kind)
        {
            case OverlayPopoverKind.WidgetSettings:
                DrawSettingRows(widget!, x, contentTop);
                break;
            case OverlayPopoverKind.Style:
                DrawStyleRows(x, contentTop);
                break;
            default:
                DrawAddRows(x, contentTop - scroll, contentTop, viewH);
                break;
        }

        _c.PopClip();

        var footY = y + h - PopPad - 28f + 4f;
        if (kind == OverlayPopoverKind.WidgetSettings)
        {
            Button(x + PopPad, footY, 110f, 26f, "Remove widget", OverlayHitKind.PopoverRemove, 0, false, true, false);
            Button(x + PopW - PopPad - 60f, footY, 60f, 26f, "Done", OverlayHitKind.PopoverDone, 0, false, false, false);
        }
        else if (kind == OverlayPopoverKind.Style)
        {
            var model = _ctl.Model;
            if (!model.Modified && model.CustomPresets.ContainsKey(model.PresetName))
            {
                Button(x + PopPad, footY, 110f, 26f, "Delete preset", OverlayHitKind.PopoverDeletePreset, 0, false, true, false);
            }

            Button(x + PopW - PopPad - 60f, footY, 60f, 26f, "Done", OverlayHitKind.PopoverDone, 0, false, false, false);
        }
    }

    private void DrawSettingRows(WidgetInstance w, float x, float top)
    {
        var settings = w.Spec.Settings;
        var right = x + PopW - PopPad;
        for (var i = 0; i < settings.Length; i++)
        {
            var def = settings[i];
            var y = top + i * RowH;
            Txt(def.Label, x + PopPad, y + (RowH - LineH(12f)) / 2f, 12f, ChromeLabel, false, OverlayTextAlign.Left, PopW - 2 * PopPad - 118f);
            switch (def.Kind)
            {
                case WidgetSettingKind.Toggle:
                    DrawCheckbox(right - 15f, y + (RowH - 15f) / 2f, w.Bool(i));
                    _ctl.Hits.Add(ChromePt(x + PopPad, y, PopW - 2 * PopPad, RowH), OverlayHitKind.PopoverToggle, i);
                    break;
                case WidgetSettingKind.Choice:
                {
                    var valueW = 92f;
                    ArrowButton(right - 20f, y + 3f, 20f, RowH - 6f, 3, OverlayHitKind.PopoverCycleNext, i);
                    ArrowButton(right - 20f - valueW - 4f - 20f, y + 3f, 20f, RowH - 6f, 2, OverlayHitKind.PopoverCyclePrev, i);
                    var label = def.Options[Math.Clamp(w.Choice(i), 0, def.Options.Length - 1)].Label;
                    Txt(label, right - 24f - valueW / 2f, y + (RowH - LineH(11.5f)) / 2f, 11.5f, ChromeText, false, OverlayTextAlign.Center, valueW);
                    break;
                }

                default:
                {
                    ArrowButton(right - 20f, y + 3f, 20f, RowH - 6f, 1, OverlayHitKind.PopoverStepUp, i);
                    ArrowButton(right - 20f - 34f - 4f - 20f, y + 3f, 20f, RowH - 6f, 0, OverlayHitKind.PopoverStepDown, i);
                    var value = _tb.Clear().AppendFixed(w.Values[i], def.Step < 1f ? 1 : 0);
                    Txt(value.Span, right - 24f - 17f, y + (RowH - LineH(11.5f)) / 2f, 11.5f, ChromeText, true, OverlayTextAlign.Center);
                    break;
                }
            }
        }
    }

    private void DrawCheckbox(float x, float y, bool on)
    {
        if (on)
        {
            _c.FillRoundRect(x, y, 15f, 15f, 4f, ChromeText);
            _c.Line(x + 3.6f, y + 7.8f, x + 6.4f, y + 10.6f, 1.8f, OverlayColor.Hex(0x111111));
            _c.Line(x + 6.4f, y + 10.6f, x + 11.6f, y + 4.6f, 1.8f, OverlayColor.Hex(0x111111));
        }
        else
        {
            _c.StrokeRoundRect(x, y, 15f, 15f, 4f, 1.4f, OverlayColor.Hex(0xFFFFFF, 0.4f));
        }
    }

    private void Slider(float x, float y, float width, float t, int a)
    {
        var knob = Math.Clamp(t, 0f, 1f) * width;
        _c.FillRoundRect(x, y - 2f, width, 4f, 2f, OverlayColor.Hex(0xFFFFFF, 0.18f));
        _c.FillRoundRect(x, y - 2f, MathF.Max(2f, knob), 4f, 2f, ChromeText);
        _c.FillCircle(x + knob, y, 6f, OverlayColor.Hex(0xFFFFFF));
        var pad = 8f;
        var pt = ChromePt(x, y - 11f, width, 22f);
        _ctl.Hits.Add(pt, OverlayHitKind.PopoverSlider, a, 0, 0f, width * _cs / _ps);
        _ = pad;
    }

    private void DrawStyleRows(float x, float top)
    {
        var g = _ctl.Model.Globals;
        var right = x + PopW - PopPad;
        const float trackW = 108f;
        var trackX = right - 42f - trackW;
        string[] labels = ["Opacity", "Scale", "Font size", "Theme", "Font"];
        for (var i = 0; i < labels.Length; i++)
        {
            var y = top + i * RowH;
            Txt(labels[i], x + PopPad, y + (RowH - LineH(12f)) / 2f, 12f, ChromeLabel);
            switch (i)
            {
                case 0:
                    Slider(trackX, y + RowH / 2f, trackW, (g.Opacity - OverlayGlobals.MinOpacity) / (1f - OverlayGlobals.MinOpacity), 0);
                    Txt(_tb.Clear().AppendFixed(g.Opacity * 100f, 0).Append('%').Span, right, y + (RowH - LineH(11.5f)) / 2f, 11.5f, ChromeText, true, OverlayTextAlign.Right);
                    break;
                case 1:
                    Slider(trackX, y + RowH / 2f, trackW, (g.Scale - OverlayGlobals.MinScale) / (OverlayGlobals.MaxScale - OverlayGlobals.MinScale), 1);
                    Txt(_tb.Clear().AppendFixed(g.Scale, 2).Append('×').Span, right, y + (RowH - LineH(11.5f)) / 2f, 11.5f, ChromeText, true, OverlayTextAlign.Right);
                    break;
                case 2:
                    Slider(trackX, y + RowH / 2f, trackW, (g.FontSize - OverlayGlobals.MinFont) / (OverlayGlobals.MaxFont - OverlayGlobals.MinFont), 2);
                    Txt(_tb.Clear().AppendFixed(g.FontSize, g.FontSize % 1f == 0 ? 0 : 1).Append(" px").Span, right, y + (RowH - LineH(11.5f)) / 2f, 11.5f, ChromeText, true, OverlayTextAlign.Right);
                    break;
                default:
                {
                    const float valueW = 80f;
                    ArrowButton(right - 20f, y + 3f, 20f, RowH - 6f, 3, OverlayHitKind.PopoverCycleNext, i);
                    ArrowButton(right - 20f - valueW - 4f - 20f, y + 3f, 20f, RowH - 6f, 2, OverlayHitKind.PopoverCyclePrev, i);
                    var label = i == 3 ? ThemeLabel(g.Theme) : FontLabel(g.Font);
                    Txt(label, right - 24f - valueW / 2f, y + (RowH - LineH(11.5f)) / 2f, 11.5f, ChromeText, false, OverlayTextAlign.Center, valueW);
                    break;
                }
            }
        }
    }

    internal static string ThemeLabel(OverlayTheme theme) =>
        theme switch { OverlayTheme.Oled => "OLED black", OverlayTheme.Glass => "Glass", _ => "Graphite" };

    internal static string FontLabel(OverlayFont font) =>
        font switch { OverlayFont.System => "System", OverlayFont.JetBrainsMono => "JetBrains Mono", _ => "Inter" };

    private void DrawAddRows(float x, float top, float viewTop, float viewH)
    {
        var specs = WidgetCatalog.All;
        var widgets = _ctl.Model.Widgets;
        for (var i = 0; i < specs.Count; i++)
        {
            var y = top + i * 46f;
            if (y + 46f < viewTop || y > viewTop + viewH)
            {
                continue;
            }

            var spec = specs[i];
            var count = 0;
            foreach (var w in widgets)
            {
                if (w.Type == spec.Type)
                {
                    count++;
                }
            }

            var textW = PopW - 2 * PopPad - 56f;
            var titleW = Txt(spec.Title, x + PopPad, y + 5f, 12f, ChromeText, true, OverlayTextAlign.Left, textW);
            if (count > 0)
            {
                Txt(_tb.Clear().Append("· ").AppendInt(count).Append(" on screen").Span, x + PopPad + titleW + 6f, y + 5f, 11f, ChromeDim,
                    false, OverlayTextAlign.Left, MathF.Max(0f, textW - titleW - 6f));
            }

            Txt(spec.Description, x + PopPad, y + 22f, 10.5f, ChromeDim, false, OverlayTextAlign.Left, textW);
            Button(x + PopW - PopPad - 44f, y + 10f, 44f, 22f, "Add", OverlayHitKind.PopoverAddItem, i);
            _c.FillRect(x + PopPad, y + 45f, PopW - 2 * PopPad, 1f, OverlayColor.Hex(0xFFFFFF, 0.07f));
        }
    }

    // ------------------------------------------------------------------ tooltip

    private void DrawTooltip()
    {
        const float size = 11f, padX = 9f, padY = 6f;
        var lineH = LineH(size) + 1f;
        var width = 0f;
        for (var i = 0; i < _tipCount; i++)
        {
            width = MathF.Max(width, Measure(_tipText[i].Span, size, _tipBold[i]));
        }

        width = MathF.Min(width + padX * 2f, 300f);
        var height = _tipCount * lineH + padY * 2f - 1f;
        var px = _ctl.PointerX * _ps / _cs;
        var py = _ctl.PointerY * _ps / _cs;
        var x = px + 14f;
        var y = py + 14f;
        if (x + width > ChromeW - 4f)
        {
            x = px - width - 12f;
        }

        if (y + height > ChromeH - 4f)
        {
            y = py - height - 12f;
        }

        x = MathF.Max(4f, x);
        y = MathF.Max(4f, y);
        _c.FillRoundRect(x, y, width, height, 8f, OverlayColor.Hex(0x0E1014, 0.95f));
        _c.StrokeRoundRect(x, y, width, height, 8f, 1f, ChromeLine);
        for (var i = 0; i < _tipCount; i++)
        {
            Txt(_tipText[i].Span, x + padX, y + padY + i * lineH, size, _tipColor[i], _tipBold[i], OverlayTextAlign.Left, width - padX * 2f);
        }
    }
}
