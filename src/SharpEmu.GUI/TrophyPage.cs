// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia;
using Avalonia.Animation;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Transformation;
using Avalonia.Threading;

namespace SharpEmu.GUI;

internal sealed class TrophyPage : UserControl
{
    private static readonly TimeSpan SlideTime = TimeSpan.FromMilliseconds(260);

    private readonly TrophyHubView _hub = new();
    private readonly TrophyGameView _game = new() { IsVisible = false };
    private IReadOnlyList<GameEntry> _games = [];
    private bool _inGame;
    private int _navigation;

    public TrophyPage()
    {
        Focusable = true;
        var stage = new Grid();
        stage.Children.Add(_hub);
        stage.Children.Add(_game);
        Content = new Border
        {
            CornerRadius = new CornerRadius(28),
            ClipToBounds = true,
            Background = new LinearGradientBrush
            {
                StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
                EndPoint = new RelativePoint(1, 1, RelativeUnit.Relative),
                GradientStops =
                {
                    new GradientStop(Color.Parse("#16112B"), 0),
                    new GradientStop(Color.Parse("#0C0A17"), 1),
                },
            },
            BorderBrush = new SolidColorBrush(Color.FromArgb(28, 255, 255, 255)),
            BorderThickness = new Thickness(1),
            Child = stage,
        };

        foreach (var view in new Control[] { _hub, _game })
        {
            view.RenderTransform = TransformOperations.Parse("translateX(0px)");
            view.Transitions =
            [
                new DoubleTransition { Property = OpacityProperty, Duration = SlideTime, Easing = new Avalonia.Animation.Easings.CubicEaseOut() },
                new TransformOperationsTransition { Property = RenderTransformProperty, Duration = SlideTime, Easing = new Avalonia.Animation.Easings.CubicEaseOut() },
            ];
        }

        _hub.GameChosen += (game, directory) => OpenGame(game, directory);
        _game.BackRequested += ShowCollection;
        KeyDown += OnKeyDown;
        PointerPressed += (_, e) =>
        {
            if (e.GetCurrentPoint(this).Properties.PointerUpdateKind == PointerUpdateKind.XButton1Pressed && _inGame)
            {
                ShowCollection();
                e.Handled = true;
            }
        };
    }

    public async void ShowCollection(IReadOnlyList<GameEntry> games)
    {
        _games = games;
        Focus();
        if (_inGame)
        {
            Slide(_game, _hub, forward: false);
            _inGame = false;
        }

        await _hub.ShowAsync(games);
    }

    public async void ShowGame(IReadOnlyList<GameEntry> games, GameEntry game)
    {
        _games = games;
        Focus();
        if (TrophyUi.GameDirectory(game.Path) is not { } directory)
        {
            return;
        }

        _hub.Opacity = 0;
        _game.Show(game, directory);
        _game.IsVisible = true;
        _game.Opacity = 1;
        _game.RenderTransform = TransformOperations.Parse("translateX(0px)");
        _hub.IsVisible = false;
        _inGame = true;
        await _hub.ShowAsync(games);
    }

    private void ShowCollection()
    {
        Slide(_game, _hub, forward: false);
        _inGame = false;
        Focus();
        _ = _hub.ShowAsync(_games);
    }

    private void OpenGame(GameEntry game, string directory)
    {
        _game.Show(game, directory);
        Slide(_hub, _game, forward: true);
        _inGame = true;
        Focus();
    }

    private void Slide(Control leaving, Control arriving, bool forward)
    {
        var token = ++_navigation;
        var distance = forward ? -48 : 48;
        arriving.IsVisible = true;
        arriving.Opacity = 0;
        arriving.RenderTransform = TransformOperations.Parse($"translateX({-distance}px)");
        Dispatcher.UIThread.Post(
            () =>
            {
                leaving.Opacity = 0;
                leaving.RenderTransform = TransformOperations.Parse($"translateX({distance}px)");
                arriving.Opacity = 1;
                arriving.RenderTransform = TransformOperations.Parse("translateX(0px)");
            },
            DispatcherPriority.Background);
        DispatcherTimer.RunOnce(
            () =>
            {
                if (token == _navigation)
                {
                    leaving.IsVisible = false;
                }
            },
            SlideTime + TimeSpan.FromMilliseconds(40));
    }

    private void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key is Key.Escape or Key.Back && _inGame)
        {
            ShowCollection();
            e.Handled = true;
        }
        else if (!_inGame && _hub.HandleKey(e.Key))
        {
            e.Handled = true;
        }
    }
}
