// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Reflection;
using SharpEmu.Application;
using SharpEmu.Logging;

if (args.Length != 2 || !Path.IsPathFullyQualified(args[1]) ||
    !string.Equals(Path.GetFullPath(args[1]), SharpEmuPaths.DataRoot, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
    throw new InvalidOperationException("Persistence worker requires its exact isolated data root.");
Directory.CreateDirectory(SharpEmuPaths.DataRoot);
switch (args[0])
{
    case "global":
    {
        var settings = new ApplicationSettings { Language = "tr", LibraryLayout = "Grid", PlayTitleMusic = false };
        Directory.CreateDirectory(ApplicationSettings.SettingsPath);
        MustFail(() => settings.SaveOrThrow()); settings.Save(); // legacy best-effort stays nonthrowing
        if (!Directory.Exists(ApplicationSettings.SettingsPath)) throw new Exception("Failure changed obstacle.");
        Directory.Delete(ApplicationSettings.SettingsPath);
        settings.SaveOrThrow(); settings.Language = "de";
        PreservePriorFileOnFailure(ApplicationSettings.SettingsPath, settings.SaveOrThrow);
        settings.SaveOrThrow();
        var restored = ApplicationSettings.Load();
        if (restored.Language != "de" || restored.LibraryLayout != "Grid" || restored.PlayTitleMusic) throw new Exception("Global retry did not retain fields.");
        break;
    }
    case "game":
    {
        const string title = "PERSISTENCE_FIXTURE";
        var settings = new PerGameSettings { LogLevel = "Debug", VSync = false, EnvironmentToggles = [], CustomEnvironmentVariables = ["VALUE=one"] };
        Directory.CreateDirectory(PerGameSettings.PathFor(title));
        MustFail(() => settings.SaveOrThrow(title)); settings.Save(title);
        Directory.Delete(PerGameSettings.PathFor(title));
        settings.SaveOrThrow(title); settings.LogLevel = "Critical";
        PreservePriorFileOnFailure(PerGameSettings.PathFor(title), () => settings.SaveOrThrow(title));
        settings.SaveOrThrow(title);
        var restored = PerGameSettings.Load(title);
        if (restored?.LogLevel != "Critical" || restored.VSync != false || restored.EnvironmentToggles?.Count != 0 || restored.CustomEnvironmentVariables?.Single() != "VALUE=one")
            throw new Exception("Per-game retry changed override/inheritance schema.");
        PreserveEmptyOverrideRemovalFailure(title);
        new PerGameSettings().SaveOrThrow(title);
        if (File.Exists(PerGameSettings.PathFor(title))) throw new Exception("Empty override was not removed.");
        try { settings.SaveOrThrow(" "); throw new Exception("Invalid title accepted."); } catch (ArgumentException) { }
        break;
    }
    case "environment":
    {
        Directory.CreateDirectory(GlobalCustomEnvironmentSettings.FilePath);
        MustFail(() => GlobalCustomEnvironmentSettings.SaveOrThrow(["VALUE=first"]));
        GlobalCustomEnvironmentSettings.Save(["VALUE=first"]);
        Directory.Delete(GlobalCustomEnvironmentSettings.FilePath);
        GlobalCustomEnvironmentSettings.SaveOrThrow(["VALUE=first"]);
        PreservePriorFileOnFailure(GlobalCustomEnvironmentSettings.FilePath, () => GlobalCustomEnvironmentSettings.SaveOrThrow(["VALUE=later"]));
        GlobalCustomEnvironmentSettings.SaveOrThrow(["VALUE=later", "=ignored", "OTHER=two words"]);
        var restored = GlobalCustomEnvironmentSettings.Load();
        if (!restored.SequenceEqual(new[] { "VALUE=later", "OTHER=two words" })) throw new Exception("Environment retry/normalization failed.");
        break;
    }
    default: throw new ArgumentException("Unknown persistence case.");
}
Console.WriteLine($"ACTUAL_PERSISTENCE_FAILURE_AND_RETRY:{args[0]}");
return 0;

static Exception MustFail(Action action)
{
    try { action(); }
    catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
    { Console.WriteLine($"EXPECTED_FILESYSTEM_FAILURE:{exception.GetType().Name}"); return exception; }
    throw new Exception("Explicit persistence falsely acknowledged a blocked file path.");
}

static void PreservePriorFileOnFailure(string path, Action save)
{
    var before = File.ReadAllBytes(path);
    if (OperatingSystem.IsWindows())
    {
        // A genuine native replacement denial while retaining the readable old file.
        using var held = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        MustFail(save);
    }
    else
    {
        var directory = Path.GetDirectoryName(path)!; var mode = File.GetUnixFileMode(directory);
        try { File.SetUnixFileMode(directory, UnixFileMode.UserRead | UnixFileMode.UserExecute); MustFail(save); }
        finally { File.SetUnixFileMode(directory, mode); }
    }
    if (!before.SequenceEqual(File.ReadAllBytes(path))) throw new Exception("Failure damaged prior valid settings.");
    if (Directory.GetFiles(Path.GetDirectoryName(path)!, $".{Path.GetFileName(path)}.*.tmp").Length != 0)
        throw new Exception("Temporary staging file retained after ordinary failure.");
    Console.WriteLine("PRIOR_VALID_FILE_PRESERVED");
    PreservePriorFileOnReplacementAndCleanupFailure(path, save);
}

static void PreservePriorFileOnReplacementAndCleanupFailure(string path, Action save)
{
    if (OperatingSystem.IsWindows()) return; // permission-based fixture requires an ordinary POSIX uid
    var before = File.ReadAllBytes(path);
    var directory = Path.GetDirectoryName(path)!;
    var mode = File.GetUnixFileMode(directory);
    var seam = typeof(ApplicationSettings).Assembly.GetType("SharpEmu.Application.SettingsFilePersistence")!
        .GetField("beforeReplacementForTesting", BindingFlags.NonPublic | BindingFlags.Static)!;
    string? staged = null;
    Exception? failure = null;
    try
    {
        seam.SetValue(null, (Action<string>)(temporary =>
        {
            staged = temporary;
            if (!File.Exists(temporary)) throw new Exception("Replacement seam ran before staging completed.");
            if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(directory, UnixFileMode.UserRead | UnixFileMode.UserExecute);
        }));
        failure = MustFail(save); // real rename denial, followed by a real temp-unlink denial
    }
    finally
    {
        seam.SetValue(null, null);
        File.SetUnixFileMode(directory, mode);
    }
    if (staged is null || !File.Exists(staged)) throw new Exception("Cleanup denial did not retain the exact staged file.");
    if (!before.SequenceEqual(File.ReadAllBytes(path))) throw new Exception("Failed replacement damaged prior valid settings.");
    if (failure!.Data["SharpEmu.SettingsTemporaryCleanupFailure"] is not (IOException or UnauthorizedAccessException))
        throw new Exception("Primary replacement error lost its actual secondary cleanup failure.");
    File.Delete(staged); // only the exact fixture-owned temporary file after permissions are restored
    Console.WriteLine($"REPLACEMENT_AND_CLEANUP_FAILURE_PRESERVED:{failure.GetType().Name}");
}

static void PreserveEmptyOverrideRemovalFailure(string title)
{
    if (OperatingSystem.IsWindows()) return; // real directory-search permission fixture is POSIX-only
    var path = PerGameSettings.PathFor(title); var before = File.ReadAllBytes(path);
    var directory = Path.GetDirectoryName(path)!; var mode = File.GetUnixFileMode(directory);
    try
    {
        File.SetUnixFileMode(directory, UnixFileMode.UserRead | UnixFileMode.UserWrite); // no search permission
        MustFail(() => new PerGameSettings().SaveOrThrow(title));
        new PerGameSettings().Save(title); // best-effort remains nonthrowing
    }
    finally { File.SetUnixFileMode(directory, mode); }
    if (!before.SequenceEqual(File.ReadAllBytes(path))) throw new Exception("Failed empty removal damaged prior override.");
    new PerGameSettings().SaveOrThrow(title);
    if (File.Exists(path)) throw new Exception("Empty override retry failed.");
    Console.WriteLine("EMPTY_OVERRIDE_REMOVAL_DENIAL_PRESERVED");
}
