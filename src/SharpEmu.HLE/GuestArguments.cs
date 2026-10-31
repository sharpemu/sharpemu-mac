// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE;

/// <summary>
/// The command line the title's process starts with, after argv[0]: the user's arguments
/// (<c>--guest-args</c>, else <c>SHARPEMU_GUEST_ARGS</c>) and defaults an engine adapter adds.
/// A user argument with the same switch as a default (the text before '=') replaces it. The
/// entry-point parameter block and getargc/getargv both report this list.
/// </summary>
public static class GuestArguments
{
    public const string EnvironmentVariable = "SHARPEMU_GUEST_ARGS";

    private static readonly object Gate = new();
    private static string? _user;
    private static readonly List<string> Defaults = [];

    /// <summary>Sets the user's arguments (whitespace-separated); null falls back to the environment.</summary>
    public static void SetUser(string? text)
    {
        lock (Gate)
        {
            _user = text;
        }
    }

    /// <summary>Adds an adapter's default argument, e.g. <c>-width=1920</c>.</summary>
    public static void AddDefault(string argument)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(argument);
        lock (Gate)
        {
            Defaults.RemoveAll(existing => SameSwitch(existing, argument));
            Defaults.Add(argument);
        }
    }

    public static void Reset()
    {
        lock (Gate)
        {
            _user = null;
            Defaults.Clear();
        }
    }

    /// <summary>The arguments after argv[0], defaults first, then the user's.</summary>
    public static IReadOnlyList<string> Effective
    {
        get
        {
            lock (Gate)
            {
                return Merge(Defaults, Split(_user ?? Environment.GetEnvironmentVariable(EnvironmentVariable)));
            }
        }
    }

    internal static List<string> Merge(IReadOnlyList<string> defaults, IReadOnlyList<string> user)
    {
        var merged = defaults.Where(argument => !user.Any(given => SameSwitch(given, argument))).ToList();
        merged.AddRange(user);
        return merged;
    }

    internal static string[] Split(string? text) =>
        string.IsNullOrWhiteSpace(text)
            ? []
            : text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static bool SameSwitch(string left, string right) =>
        string.Equals(SwitchOf(left), SwitchOf(right), StringComparison.OrdinalIgnoreCase);

    private static string SwitchOf(string argument)
    {
        var equals = argument.IndexOf('=');
        return equals < 0 ? argument : argument[..equals];
    }
}
