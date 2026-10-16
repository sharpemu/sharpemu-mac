// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using SharpEmu.GUI.Controls;
using SharpEmu.Libs.Np;

namespace SharpEmu.GUI;

public partial class TrophyHubView : UserControl
{
    private enum Sort
    {
        Recent,
        Progress,
        Name,
    }

    private sealed record GameSummary(GameEntry Game, string Directory, IReadOnlyList<TrophyEntry> Trophies)
    {
        public int Earned => Trophies.Count(t => t.Unlocked);

        public double Percent => Trophies.Count == 0 ? 0 : Earned * 100.0 / Trophies.Count;

        public bool Complete => Trophies.Count > 0 && Earned == Trophies.Count;

        public TrophyEntry? Latest => Trophies.Where(t => t.UnlockedAtUtc is not null).MaxBy(t => t.UnlockedAtUtc);

        public int EarnedOf(int grade) => Trophies.Count(t => t.Grade == grade && t.Unlocked);

        public int TotalOf(int grade) => Trophies.Count(t => t.Grade == grade);
    }

    private static readonly Color Gold = Color.Parse("#F2B93B");

    private IReadOnlyList<GameEntry> _games = [];
    private readonly List<(Border Card, GameSummary Summary)> _cards = [];
    private readonly List<(Border Chip, Sort Sort)> _sortChips = [];
    private readonly Dictionary<GameEntry, Bitmap?> _art = [];
    private readonly DispatcherTimer _idle = new() { Interval = TimeSpan.FromMilliseconds(16) };
    private readonly System.Diagnostics.Stopwatch _clock = System.Diagnostics.Stopwatch.StartNew();
    private IReadOnlyList<GameSummary> _summaries = [];
    private Sort _sort = Sort.Recent;
    private int _selected = -1;
    private int _backdropToken;
    private bool _backdropOnA;
    private double _shownAt;
    private bool _wasShown;
    private bool _animated;

    public event Action<GameEntry, string>? GameChosen;

    public TrophyHubView()
    {
        InitializeComponent();
        SearchBox.TextChanged += (_, _) =>
        {
            SearchClear.IsVisible = !string.IsNullOrEmpty(SearchBox.Text);
            RenderCards(stagger: false);
        };
        SearchClear.Click += (_, _) =>
        {
            SearchBox.Text = string.Empty;
            SearchBox.Focus();
        };
        SearchBox.KeyDown += (_, e) =>
        {
            if (e.Key == Key.Escape && !string.IsNullOrEmpty(SearchBox.Text))
            {
                SearchBox.Text = string.Empty;
                e.Handled = true;
            }
        };

        _idle.Tick += (_, _) =>
        {
            if (!IsEffectivelyVisible)
            {
                _wasShown = false;
                return;
            }

            if (!_wasShown)
            {
                _wasShown = true;
                _shownAt = _clock.Elapsed.TotalSeconds;
            }

            var t = _clock.Elapsed.TotalSeconds - _shownAt;
            HeroMedal.Spin = t < 1.4 ? (1 - (1 - Math.Pow(1 - t / 1.4, 4))) * Math.PI * 4 : 0;
            var cycle = (t - 1.2) % 4.5;
            HeroMedal.Glint = t > 1.2 && cycle < 0.9 ? cycle / 0.9 : -1;
        };
        _idle.Start();
    }

    public async Task ShowAsync(IReadOnlyList<GameEntry> games)
    {
        _games = games;
        SearchBox.PlaceholderText = Localization.Instance.Get("Trophies.SearchWatermark");
        BuildSortChips();
        await LoadAsync();
    }

    public bool HandleKey(Key key)
    {
        switch (key)
        {
            case Key.Down:
                Select(Math.Min(_cards.Count - 1, _selected + 1));
                return true;
            case Key.Up:
                Select(Math.Max(0, _selected - 1));
                return true;
            case Key.Enter when _selected >= 0 && _selected < _cards.Count:
                Open(_cards[_selected].Summary);
                return true;
            default:
                return false;
        }
    }

    private void BuildSortChips()
    {
        SortBar.Children.Clear();
        _sortChips.Clear();
        foreach (var (sort, key) in new[] { (Sort.Recent, "Trophies.Sort.Recent"), (Sort.Progress, "Trophies.Sort.Progress"), (Sort.Name, "Trophies.Sort.Name") })
        {
            var chip = TrophyUi.Chip(Localization.Instance.Get(key));
            chip.PointerPressed += (_, _) =>
            {
                _sort = sort;
                UpdateSortChips();
                RenderCards(stagger: true);
            };
            _sortChips.Add((chip, sort));
            SortBar.Children.Add(chip);
        }

        UpdateSortChips();
    }

    private void UpdateSortChips()
    {
        foreach (var (chip, sort) in _sortChips)
        {
            TrophyUi.SetChipSelected(chip, sort == _sort);
        }
    }

    private async Task LoadAsync()
    {
        _summaries = await Task.Run(() => _games
            .Select(game => (Game: game, Directory: TrophyUi.GameDirectory(game.Path)))
            .Where(entry => entry.Directory is not null)
            .AsParallel()
            .Select(entry => new GameSummary(
                entry.Game,
                entry.Directory!,
                TrophyCatalog.Load(entry.Directory!, entry.Game.TitleId!, includeIcons: false)))
            .Where(summary => summary.Trophies.Count > 0)
            .ToList());

        RenderTotals(_summaries);
        RenderCards(stagger: !_animated);
        _animated = true;
    }

    private IEnumerable<GameSummary> Sorted()
    {
        var query = SearchBox.Text?.Trim();
        var matching = string.IsNullOrEmpty(query)
            ? _summaries
            : _summaries.Where(s => s.Game.Name.Contains(query, StringComparison.CurrentCultureIgnoreCase));
        return _sort switch
        {
            Sort.Progress => matching
                .OrderByDescending(s => s.Percent)
                .ThenBy(s => s.Game.Name, StringComparer.CurrentCultureIgnoreCase),
            Sort.Name => matching.OrderBy(s => s.Game.Name, StringComparer.CurrentCultureIgnoreCase),
            _ => matching
                .OrderByDescending(s => s.Latest?.UnlockedAtUtc ?? DateTime.MinValue)
                .ThenBy(s => s.Game.Name, StringComparer.CurrentCultureIgnoreCase),
        };
    }

    private void RenderCards(bool stagger)
    {
        GameList.Children.Clear();
        _cards.Clear();
        foreach (var summary in Sorted())
        {
            var card = BuildCard(summary);
            _cards.Add((card, summary));
            GameList.Children.Add(card);
        }

        if (_cards.Count == 0)
        {
            GameList.Children.Add(new TextBlock
            {
                Text = Localization.Instance.Get(_summaries.Count == 0 ? "Trophies.None" : "Trophies.NoMatch"),
                Foreground = new SolidColorBrush(Color.Parse("#8E89A6")),
                FontSize = 15,
            });
            return;
        }

        if (stagger)
        {
            TrophyUi.StaggerIn(_cards.Select(c => (Control)c.Card).ToList());
        }

        Select(Math.Clamp(_selected, 0, _cards.Count - 1));
    }

    private void RenderTotals(IReadOnlyList<GameSummary> summaries)
    {
        var earned = summaries.Sum(s => s.Earned);
        var total = summaries.Sum(s => s.Trophies.Count);
        var percent = total == 0 ? 0 : earned * 100.0 / total;
        TotalOfText.Text = Localization.Instance.Format("Trophies.HubSummary", earned, total, summaries.Count);
        var duration = TimeSpan.FromMilliseconds(_animated ? 400 : 1300);
        TrophyUi.Animate(_animated ? earned : 0, earned, duration, value => TotalEarnedText.Text = ((int)Math.Round(value)).ToString());
        TrophyUi.Animate(_animated ? percent : 0, percent, duration, value => TotalRing.Value = value);
        TotalGrades.Children.Clear();
        for (var grade = 1; grade <= 4; grade++)
        {
            TotalGrades.Children.Add(TrophyUi.GradeChip(grade, summaries.Sum(s => s.EarnedOf(grade)), summaries.Sum(s => s.TotalOf(grade))));
        }
    }

    private Border BuildCard(GameSummary summary)
    {
        var cover = new Border
        {
            Width = 104,
            Height = 104,
            CornerRadius = new CornerRadius(18),
            ClipToBounds = true,
            Background = summary.Game.PlaceholderBrush,
            Child = summary.Game.Cover is { } bitmap
                ? new Image { Source = bitmap, Stretch = Stretch.UniformToFill }
                : new TextBlock
                {
                    Text = summary.Game.Initials,
                    FontSize = 28,
                    FontWeight = FontWeight.Bold,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                },
        };
        var coverFrame = new Grid { Width = 104, Height = 104, VerticalAlignment = VerticalAlignment.Center };
        coverFrame.Children.Add(new Border
        {
            CornerRadius = new CornerRadius(18),
            BoxShadow = new BoxShadows(new BoxShadow { OffsetY = 10, Blur = 24, Color = Color.FromArgb(150, 0, 0, 0) }),
            Child = cover,
        });
        if (summary.Complete)
        {
            coverFrame.Children.Add(new TrophyMedal
            {
                Width = 40,
                Height = 40,
                Grade = 1,
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Bottom,
                Margin = new Thickness(0, 0, -12, -12),
                Effect = new DropShadowEffect { BlurRadius = 10, OffsetX = 0, OffsetY = 3, Color = Colors.Black, Opacity = 0.6 },
            });
        }

        var name = new TextBlock
        {
            Text = summary.Game.Name,
            FontSize = 20,
            FontWeight = FontWeight.SemiBold,
            TextTrimming = TextTrimming.CharacterEllipsis,
        };
        var counts = new TextBlock
        {
            Text = Localization.Instance.Format("Trophies.Progress", summary.Earned, summary.Trophies.Count, (int)Math.Round(summary.Percent)),
            Foreground = new SolidColorBrush(Color.Parse("#B8B4CC")),
            FontSize = 13,
        };

        var latest = summary.Latest;
        var latestRow = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            Spacing = 6,
            Children =
            {
                TrophyUi.Glyph(latest is null ? "hourglass_empty" : "workspace_premium", 15, latest is null ? Color.Parse("#8E89A6") : TrophyMedal.Palette(latest.Grade).Mid),
                new TextBlock
                {
                    Text = latest is null
                        ? Localization.Instance.Get("Trophies.NoneYet")
                        : Localization.Instance.Format("Trophies.Latest", latest.Name, latest.UnlockedAtUtc!.Value.ToLocalTime().ToString("d")),
                    Foreground = new SolidColorBrush(Color.Parse(latest is null ? "#8E89A6" : "#D8D4EA")),
                    FontSize = 12.5,
                    TextTrimming = TextTrimming.CharacterEllipsis,
                    VerticalAlignment = VerticalAlignment.Center,
                },
            },
        };

        var medals = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 16, Margin = new Thickness(0, 2, 0, 0) };
        for (var grade = 1; grade <= 4; grade++)
        {
            var total = summary.TotalOf(grade);
            if (total == 0)
            {
                continue;
            }

            var earned = summary.EarnedOf(grade);
            medals.Children.Add(new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Spacing = 6,
                Children =
                {
                    new TrophyMedal { Width = 24, Height = 24, Grade = grade, Opacity = earned == 0 ? 0.5 : 1 },
                    TrophyUi.Count(earned, total, 13.5),
                },
            });
        }

        var middle = new StackPanel
        {
            Spacing = 5,
            VerticalAlignment = VerticalAlignment.Center,
            Children = { name, counts, latestRow, medals },
        };

        var ring = new Grid { Width = 84, Height = 84, VerticalAlignment = VerticalAlignment.Center };
        var progressRing = new TrophyProgressRing { Thickness = 8 };
        TrophyUi.Animate(_animated ? summary.Percent : 0, summary.Percent, TimeSpan.FromMilliseconds(_animated ? 1 : 1100), value => progressRing.Value = value);
        ring.Children.Add(progressRing);
        ring.Children.Add(new TextBlock
        {
            Text = $"{Math.Round(summary.Percent)}%",
            FontSize = 17,
            FontWeight = FontWeight.Bold,
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
        });

        var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("Auto,*,Auto"), ColumnSpacing = 24, Margin = new Thickness(18) };
        grid.Children.Add(coverFrame);
        Grid.SetColumn(middle, 1);
        grid.Children.Add(middle);
        Grid.SetColumn(ring, 2);
        grid.Children.Add(ring);

        var spotlight = new Border
        {
            IsHitTestVisible = false,
            Opacity = 0,
            Transitions = [new Avalonia.Animation.DoubleTransition { Property = Visual.OpacityProperty, Duration = TimeSpan.FromMilliseconds(220) }],
        };
        var layers = new Grid();
        layers.Children.Add(spotlight);
        layers.Children.Add(grid);

        var card = new Border
        {
            CornerRadius = new CornerRadius(24),
            BorderThickness = new Thickness(1),
            ClipToBounds = true,
            Background = summary.Complete
                ? new LinearGradientBrush
                {
                    StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
                    EndPoint = new RelativePoint(1, 1, RelativeUnit.Relative),
                    GradientStops =
                    {
                        new GradientStop(Color.FromArgb(70, Gold.R, Gold.G, Gold.B), 0),
                        new GradientStop(Color.FromArgb(14, 255, 255, 255), 0.7),
                    },
                }
                : new LinearGradientBrush
                {
                    StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
                    EndPoint = new RelativePoint(1, 1, RelativeUnit.Relative),
                    GradientStops =
                    {
                        new GradientStop(Color.FromArgb(32, 255, 255, 255), 0),
                        new GradientStop(Color.FromArgb(12, 255, 255, 255), 1),
                    },
                },
            BorderBrush = RestingBorder(summary, selected: false),
            Cursor = new Cursor(StandardCursorType.Hand),
            Child = layers,
        };
        TrophyUi.Lift(card);
        card.PointerEntered += (_, _) =>
        {
            _selected = _cards.FindIndex(c => c.Card == card);
            spotlight.Opacity = 1;
            SetBackdrop(summary.Game);
        };
        card.PointerExited += (_, _) => spotlight.Opacity = 0;
        card.PointerMoved += (_, e) =>
        {
            var position = e.GetPosition(card);
            var width = Math.Max(1, card.Bounds.Width);
            var height = Math.Max(1, card.Bounds.Height);
            var center = new RelativePoint(position.X / width, position.Y / height, RelativeUnit.Relative);
            spotlight.Background = new RadialGradientBrush
            {
                Center = center,
                GradientOrigin = center,
                RadiusX = new RelativeScalar(0.4, RelativeUnit.Relative),
                RadiusY = new RelativeScalar(0.4 * width / height, RelativeUnit.Relative),
                GradientStops =
                {
                    new GradientStop(Color.FromArgb(46, 255, 255, 255), 0),
                    new GradientStop(Color.FromArgb(0, 255, 255, 255), 1),
                },
            };
        };
        card.PointerPressed += (_, _) => Open(summary);
        return card;
    }

    private static IBrush RestingBorder(GameSummary summary, bool selected) =>
        new SolidColorBrush(selected
            ? Color.FromArgb(150, 255, 255, 255)
            : summary.Complete ? Color.FromArgb(120, Gold.R, Gold.G, Gold.B) : Color.FromArgb(30, 255, 255, 255));

    private async void SetBackdrop(GameEntry game)
    {
        var token = ++_backdropToken;
        if (!_art.TryGetValue(game, out var art))
        {
            art = game.Background;
            if (art is null && game.BackgroundPath is { } path && File.Exists(path))
            {
                art = await Task.Run(() =>
                {
                    try
                    {
                        using var stream = File.OpenRead(path);
                        return Bitmap.DecodeToWidth(stream, 1280);
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException or NotSupportedException)
                    {
                        return null;
                    }
                });
            }

            _art[game] = art;
        }

        if (token != _backdropToken || art is null)
        {
            return;
        }

        var next = _backdropOnA ? BackdropB : BackdropA;
        var previous = _backdropOnA ? BackdropA : BackdropB;
        next.Source = art;
        next.Opacity = 0.55;
        previous.Opacity = 0;
        _backdropOnA = !_backdropOnA;
    }

    private void Select(int index)
    {
        if (index < 0 || index >= _cards.Count)
        {
            return;
        }

        _selected = index;
        for (var i = 0; i < _cards.Count; i++)
        {
            _cards[i].Card.BorderBrush = RestingBorder(_cards[i].Summary, i == index);
        }

        _cards[index].Card.BringIntoView();
        SetBackdrop(_cards[index].Summary.Game);
    }

    private void Open(GameSummary summary) => GameChosen?.Invoke(summary.Game, summary.Directory);
}
