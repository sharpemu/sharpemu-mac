// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.ComponentModel;
namespace SharpEmu.GUI;

/// <summary>XAML facade; translation ownership belongs to the application package.</summary>
public sealed class Localization : INotifyPropertyChanged
{
    public static Localization Instance { get; } = new(LauncherLocalization.Instance);
    private readonly LauncherLocalization strings;
    private Localization(LauncherLocalization strings) => this.strings = strings;
    internal Localization(string directory) : this(new LauncherLocalization(directory)) { }
    public static string LanguagesDirectory => LauncherLocalization.LanguagesDirectory;
    public string CurrentCode => strings.CurrentCode;
    public string this[string key] => Get(key);
    public event PropertyChangedEventHandler? PropertyChanged
    { add => strings.PropertyChanged += value; remove => strings.PropertyChanged -= value; }
    public string Get(string key) => strings.Get(key);
    public string Format(string key, params object?[] args) => strings.Format(key, args);
    public List<LanguageInfo> DiscoverLanguages() => strings.DiscoverLanguages();
    public void Load(string code) => strings.Load(code);
    public IReadOnlyDictionary<string, string> WithPrefix(string prefix) => strings.WithPrefix(prefix);
}
