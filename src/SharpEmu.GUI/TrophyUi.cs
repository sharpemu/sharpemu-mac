// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia;
using Avalonia.Animation;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Media.Transformation;
using Avalonia.Threading;
using SharpEmu.GUI.Controls;

namespace SharpEmu.GUI;

internal static class TrophyUi
{
    private static readonly IBrush ChipBrush = new SolidColorBrush(Color.FromArgb(24, 255, 255, 255));
    private static readonly IBrush ChipBorderBrush = new SolidColorBrush(Color.FromArgb(34, 255, 255, 255));

    // A .zar game keeps its metadata, trophy package included, in its cache folder.
    public static string? GameDirectory(string gamePath) =>
        SharpEmu.HLE.FileSystem.ZArchive.IsArchivePath(gamePath)
            ? SharpEmu.HLE.FileSystem.ZArchiveCache.GetCacheRoot(gamePath)
            : Path.GetDirectoryName(gamePath);

    public static string GradeName(int grade) =>
        Localization.Instance.Get($"Trophies.Grade.{Math.Clamp(grade, 1, 4)}");

    public static Bitmap? LoadIcon(byte[]? png)
    {
        if (png is null || png.Length == 0)
        {
            return null;
        }

        try
        {
            using var stream = new MemoryStream(png);
            return Bitmap.DecodeToWidth(stream, 192);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            return null;
        }
    }

    public static TextBlock Glyph(string name, double size, Color color) => new()
    {
        Classes = { "materialSymbol" },
        Text = name,
        FontSize = size,
        LineHeight = size + 2,
        Foreground = new SolidColorBrush(color),
        HorizontalAlignment = HorizontalAlignment.Center,
        VerticalAlignment = VerticalAlignment.Center,
    };

    public static Border Chip(string text)
    {
        var chip = new Border
        {
            CornerRadius = new CornerRadius(16),
            Padding = new Thickness(16, 7),
            BorderThickness = new Thickness(1),
            Cursor = new Cursor(StandardCursorType.Hand),
            Child = new TextBlock { Text = text, FontSize = 13, FontWeight = FontWeight.SemiBold },
        };
        var duration = TimeSpan.FromMilliseconds(160);
        chip.Transitions =
        [
            new BrushTransition { Property = Border.BackgroundProperty, Duration = duration },
            new BrushTransition { Property = Border.BorderBrushProperty, Duration = duration },
        ];
        return chip;
    }

    public static void SetChipSelected(Border chip, bool selected)
    {
        chip.Background = selected ? Brushes.White : ChipBrush;
        chip.BorderBrush = selected ? Brushes.White : ChipBorderBrush;
        if (chip.Child is TextBlock text)
        {
            text.Foreground = selected ? new SolidColorBrush(Color.Parse("#140F26")) : Brushes.White;
        }
    }

    public static Border GradeChip(int grade, int earned, int total) => new()
    {
        CornerRadius = new CornerRadius(18),
        Padding = new Thickness(6, 5, 14, 5),
        Margin = new Thickness(0, 0, 10, 8),
        Background = ChipBrush,
        BorderBrush = ChipBorderBrush,
        BorderThickness = new Thickness(1),
        Child = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            Spacing = 8,
            Children =
            {
                new TrophyMedal { Width = 26, Height = 26, Grade = grade, Opacity = earned == 0 ? 0.6 : 1 },
                Count(earned, total, 14),
            },
        },
    };

    public static TextBlock Count(int earned, int total, double size)
    {
        var text = new TextBlock { FontSize = size, VerticalAlignment = VerticalAlignment.Center };
        text.Inlines!.Add(new Avalonia.Controls.Documents.Run(earned.ToString()) { FontWeight = FontWeight.Bold });
        text.Inlines.Add(new Avalonia.Controls.Documents.Run($" / {total}") { Foreground = new SolidColorBrush(Color.Parse("#8E89A6")) });
        return text;
    }

    public static Control StatusChip(string glyph, string text, Color color) => new StackPanel
    {
        Orientation = Orientation.Horizontal,
        Spacing = 5,
        HorizontalAlignment = HorizontalAlignment.Right,
        Children =
        {
            Glyph(glyph, 14, color),
            new TextBlock { Text = text, FontSize = 12, Foreground = new SolidColorBrush(color), VerticalAlignment = VerticalAlignment.Center },
        },
    };

    public static void Lift(Border card)
    {
        var duration = TimeSpan.FromMilliseconds(170);
        card.RenderTransform = TransformOperations.Parse("translateY(0px)");
        card.Transitions ??= [];
        card.Transitions.Add(new TransformOperationsTransition { Property = Visual.RenderTransformProperty, Duration = duration, Easing = new Avalonia.Animation.Easings.CubicEaseOut() });
        card.Transitions.Add(new DoubleTransition { Property = Visual.OpacityProperty, Duration = TimeSpan.FromMilliseconds(320) });
        IBrush? resting = null;
        card.PointerEntered += (_, _) =>
        {
            resting = card.BorderBrush;
            card.RenderTransform = TransformOperations.Parse("translateY(-3px)");
            card.BorderBrush = new SolidColorBrush(Color.FromArgb(150, 255, 255, 255));
        };
        card.PointerExited += (_, _) =>
        {
            card.RenderTransform = TransformOperations.Parse("translateY(0px)");
            card.BorderBrush = resting;
        };
    }

    public static void FlipIn(Control card)
    {
        card.RenderTransformOrigin = new RelativePoint(0.5, 0.5, RelativeUnit.Relative);
        card.RenderTransform = TransformOperations.Parse("scaleY(0.04)");
        Dispatcher.UIThread.Post(() => card.RenderTransform = TransformOperations.Parse("scaleY(1)"), DispatcherPriority.Background);
    }

    public static void StaggerIn(IReadOnlyList<Control> rows)
    {
        for (var i = 0; i < rows.Count; i++)
        {
            var row = rows[i];
            row.Opacity = 0;
            row.Transitions ??= [];
            if (!row.Transitions.Any(t => t is DoubleTransition d && d.Property == Visual.OpacityProperty))
            {
                row.Transitions.Add(new DoubleTransition { Property = Visual.OpacityProperty, Duration = TimeSpan.FromMilliseconds(320) });
            }

            if (row.RenderTransform is ITransform && row.Transitions.Any(t => t is TransformOperationsTransition))
            {
                row.RenderTransform = TransformOperations.Parse("translateY(18px)");
            }
        }

        var next = 0;
        var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(35) };
        timer.Tick += (_, _) =>
        {
            var batch = next < 9 ? 1 : rows.Count;
            for (var i = 0; i < batch && next < rows.Count; i++)
            {
                var row = rows[next++];
                row.Opacity = 1;
                if (row.Transitions?.Any(t => t is TransformOperationsTransition) == true)
                {
                    row.RenderTransform = TransformOperations.Parse("translateY(0px)");
                }
            }

            if (next >= rows.Count)
            {
                timer.Stop();
            }
        };
        timer.Start();
    }

    public static void Animate(double from, double to, TimeSpan duration, Action<double> apply)
    {
        apply(from);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) };
        timer.Tick += (_, _) =>
        {
            var t = Math.Min(1.0, clock.Elapsed.TotalMilliseconds / duration.TotalMilliseconds);
            apply(from + (to - from) * (1 - Math.Pow(1 - t, 3)));
            if (t >= 1)
            {
                timer.Stop();
            }
        };
        timer.Start();
    }
}
