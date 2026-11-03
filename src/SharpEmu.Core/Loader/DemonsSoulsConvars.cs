// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using System.Text.RegularExpressions;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Turns off Demon's Souls' (PPSA01342) XPR index-buffer culling through the engine's own convar.
/// With <c>doIndexBufferCulling</c> on, the title runs a compute pass that culls every object's
/// triangles before drawing it, ~2800 small dispatches a frame in Boletaria 1-1; with it off the
/// GPU draws the unculled triangles and the pass is gone.
/// <para>The engine builds its command line from <c>$/CoreData/EngineSupport/DefaultCmdLineArgs.txt</c>
/// at start-up (<c>$/</c> is /app0; v01.004 reads it at 0x853792 through the loader at 0x84A7D0,
/// which splits the file on CR and LF and parses <c>+name=value</c> commands; the title's
/// <c>CmdLine:</c> log line lists them; they run when the main loop starts, after the renderer
/// has registered the convar, whose byte is read every frame). The served copy of that file gets one more line,
/// <see cref="ConvarCommand"/>, appended in the file's own line ending; nothing is written to disk.
/// A file that already carries a <c>+doIndexBufferCulling</c> command is left as it is, so a value
/// put there by hand wins. The only other command-line source, <c>#/savedCmdArgs.txt</c>, is the
/// debug menu's save file on /hostapp and not read by a retail start-up.</para>
/// <para>The served file grows, which <see cref="SharpEmu.HLE.FileSystem.App0FileOverlay"/> allows
/// for a file served from an archive-mounted /app0 (a .zar) but not for one on disk: a loose
/// /app0 install logs a warning and keeps the culling.</para>
/// <see cref="EnvironmentVariable"/>=1 keeps the game's culling.
/// </summary>
internal static class DemonsSoulsConvars
{
    public const string EnvironmentVariable = "SHARPEMU_DS_INDEX_BUFFER_CULLING";
    public const string TitleId = DemonsSoulsIdleNapPatch.TitleId;
    public const string ConvarName = "doIndexBufferCulling";
    public const string ConvarCommand = "+" + ConvarName + "=false";
    public const string ArgsGuestPath = "/app0/CoreData/EngineSupport/DefaultCmdLineArgs.txt";

    internal enum Outcome
    {
        /// <summary>Not Demon's Souls, or no /app0.</summary>
        NotApplicable,

        /// <summary><see cref="EnvironmentVariable"/>=1: the game's culling stays.</summary>
        KeptBySwitch,

        /// <summary>The file already sets the convar; its value stands.</summary>
        AlreadySet,

        /// <summary>The file is not in /app0, so there is nothing to serve it from.</summary>
        Absent,

        /// <summary>The file is on disk, where a served copy must keep the file's length.</summary>
        CannotGrow,

        Injected,
    }

    // "+doIndexBufferCulling" as a command: '+' and the name, then '=', blank or the end.
    private static readonly Regex ExistingCommand = new(
        @"\+" + ConvarName + @"(?=[\s=]|$)",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    internal static bool KeepsCulling(string? switchValue) =>
        string.Equals(switchValue?.Trim(), "1", StringComparison.Ordinal);

    internal static string HostPath(string app0Root) =>
        Path.Combine(app0Root, "CoreData", "EngineSupport", "DefaultCmdLineArgs.txt");

    /// <summary>Serves the title's command-line file with the convar turned off; called once at load.</summary>
    public static Outcome Apply(string? titleId)
    {
        var app0Root = EngineResolution.App0Root;
        var outcome = Apply(
            titleId,
            app0Root,
            KeepsCulling(Environment.GetEnvironmentVariable(EnvironmentVariable)),
            path => EngineResolution.TryReadApp0File(path),
            (path, content) => EngineResolution.TryReplaceApp0File(path, content, "Demon's Souls " + ConvarCommand));
        Report(outcome, outcome == Outcome.Absent && ExistsInApp0(HostPath(app0Root!)));
        return outcome;
    }

    /// <summary>The decision, with the file access passed in: <paramref name="read"/> returns the
    /// file's bytes or null when it is absent, <paramref name="replace"/> serves new content and
    /// returns false when it cannot.</summary>
    internal static Outcome Apply(
        string? titleId,
        string? app0Root,
        bool keepCulling,
        Func<string, byte[]?> read,
        Func<string, byte[], bool> replace)
    {
        if (!string.Equals(titleId, TitleId, StringComparison.Ordinal) || string.IsNullOrEmpty(app0Root))
        {
            return Outcome.NotApplicable;
        }

        if (keepCulling)
        {
            return Outcome.KeptBySwitch;
        }

        var path = HostPath(app0Root);
        var original = read(path);
        if (original is null)
        {
            return Outcome.Absent;
        }

        var merged = Merge(original);
        if (merged is null)
        {
            return Outcome.AlreadySet;
        }

        return replace(path, merged) ? Outcome.Injected : Outcome.CannotGrow;
    }

    /// <summary>The file with <see cref="ConvarCommand"/> appended as a line of its own, in the
    /// file's line ending (CRLF when it has any, else LF); a leading byte-order mark and the rest
    /// stay as they are. Null when the file already has a <c>+doIndexBufferCulling</c> command.</summary>
    internal static byte[]? Merge(byte[] original)
    {
        ArgumentNullException.ThrowIfNull(original);
        // Latin-1 maps every byte to one char, so offsets and the ASCII command survive any encoding.
        var text = Encoding.Latin1.GetString(original);
        if (ExistingCommand.IsMatch(text))
        {
            return null;
        }

        var newline = text.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
        var unterminated = text.Length != 0 && text[^1] is not ('\n' or '\r');
        var tail = Encoding.ASCII.GetBytes((unterminated ? newline : string.Empty) + ConvarCommand + newline);
        return [.. original, .. tail];
    }

    // A file the read returned nothing for is either missing or over the read's 1 MiB limit.
    private static bool ExistsInApp0(string hostPath) =>
        File.Exists(hostPath) || SharpEmu.HLE.FileSystem.App0Archive.TryGetFile(hostPath, out _);

    private static void Report(Outcome outcome, bool presentButUnread = false)
    {
        switch (outcome)
        {
            case Outcome.Injected:
                Console.Error.WriteLine(
                    $"[LOADER][INFO] [DS] {ConvarName}=false injected via {ArgsGuestPath} " +
                    $"(no XPR index-buffer culling pass; {EnvironmentVariable}=1 keeps it)");
                break;
            case Outcome.KeptBySwitch:
                Console.Error.WriteLine($"[LOADER][INFO] [DS] {ConvarName} left to the game ({EnvironmentVariable}=1)");
                break;
            case Outcome.AlreadySet:
                Console.Error.WriteLine($"[LOADER][INFO] [DS] {ConvarName} left as {ArgsGuestPath} sets it");
                break;
            case Outcome.Absent:
                Console.Error.WriteLine(
                    $"[LOADER][WARN] [DS] {ConvarName} not injected: {ArgsGuestPath} " +
                    (presentButUnread ? "is larger than 1 MiB" : "is missing"));
                break;
            case Outcome.CannotGrow:
                Console.Error.WriteLine(
                    $"[LOADER][WARN] [DS] {ConvarName} not injected: {ArgsGuestPath} is on disk, where a served " +
                    "copy must keep its length (run the game from a .zar to turn the culling pass off)");
                break;
        }
    }
}
