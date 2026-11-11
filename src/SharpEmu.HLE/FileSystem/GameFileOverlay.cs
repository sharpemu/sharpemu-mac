// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Enumeration;

namespace SharpEmu.HLE.FileSystem;

// A session-local lookup: mod files stay in their own directories, including for archived games.
public sealed class GameFileOverlay : IDisposable
{
    private static GameFileOverlay? _current;
    private static readonly StringComparer DefaultComparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
    private static readonly StringComparison Comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
    private readonly string _root;
    // Names under the application root compare the way the host compares the application's own
    // names, so a guest path that reaches an app0 file in another case reaches the mod's file too.
    private readonly StringComparer _comparer;
    private readonly StringComparison _comparison;
    private readonly Dictionary<string, (string Path, string Root)> _files;
    private readonly Dictionary<string, (string Path, string Root)> _directories;
    private readonly Dictionary<string, HashSet<string>> _children;
    private readonly Dictionary<string, (string Path, string Root)>.AlternateLookup<ReadOnlySpan<char>> _fileLookup;
    private readonly Dictionary<string, (string Path, string Root)>.AlternateLookup<ReadOnlySpan<char>> _directoryLookup;

    public GameFileOverlay(string applicationRoot, IEnumerable<string> modDirectories)
        : this(applicationRoot, modDirectories, OperatingSystem.IsWindows())
    {
    }

    /// <param name="foldsCase">Whether the host resolves names under <paramref name="applicationRoot"/>
    /// case-insensitively (a case-insensitive volume).</param>
    public GameFileOverlay(string applicationRoot, IEnumerable<string> modDirectories, bool foldsCase)
    {
        _comparer = foldsCase ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        _comparison = foldsCase ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        _files = new(_comparer);
        _directories = new(_comparer);
        _children = new(_comparer);
        _fileLookup = _files.GetAlternateLookup<ReadOnlySpan<char>>();
        _directoryLookup = _directories.GetAlternateLookup<ReadOnlySpan<char>>();
        _root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(applicationRoot));
        foreach (var mod in modDirectories)
        {
            var directory = Path.GetFullPath(mod);
            if (!Directory.Exists(directory) || !IsSafePath(directory, directory)) continue;
            foreach (var file in Directory.EnumerateFiles(directory, "*", new EnumerationOptions
                     { RecurseSubdirectories = true, AttributesToSkip = FileAttributes.ReparsePoint, IgnoreInaccessible = false }))
            {
                if (!IsSafePath(directory, file)) continue;
                var relative = Path.GetRelativePath(directory, file);
                // A later mod wins, under its own spelling.
                _files.Remove(relative);
                _files[relative] = (file, directory);
                var child = relative;
                while (Path.GetDirectoryName(child) is { } parent)
                {
                    if (!_children.TryGetValue(parent, out var names)) _children[parent] = names = new(_comparer);
                    names.Add(Path.GetFileName(child));
                    _directories[parent] = (Path.Combine(directory, parent), directory);
                    if (parent.Length == 0) break;
                    child = parent;
                }
            }
        }
        Volatile.Write(ref _current, this);
    }

    /// <summary>Whether a mod overlay is mounted. Every lookup below is a null check without one.</summary>
    public static bool IsMounted => Volatile.Read(ref _current) is not null;

    /// <summary>How the mounted overlay compares host paths under the application root; the host
    /// default when none is mounted. Listings that merge mod names with host names dedupe by it.</summary>
    public static StringComparer PathComparer => Volatile.Read(ref _current)?._comparer ?? DefaultComparer;

    public static string ResolveFile(string hostPath)
    {
        var current = Volatile.Read(ref _current);
        if (current is null || !current.TryRelative(hostPath, out var relative) ||
            !current._fileLookup.TryGetValue(relative, out var file)) return hostPath;
        // Do not follow a mod directory replaced with a symlink after the session started.
        if (File.Exists(file.Path) && IsSafePath(file.Root, file.Path)) return file.Path;
        lock (current._withdrawn)
        {
            if (current._withdrawn.Add(file.Path))
                Console.Error.WriteLine($"[MODS][WARN] {file.Path} is gone or now behind a link; serving the game's own file for {hostPath}");
        }
        return hostPath;
    }

    // Mod files the session indexed that a later lookup refused, each reported once.
    private readonly HashSet<string> _withdrawn = new(StringComparer.Ordinal);

    public static bool IsDirectory(string hostPath) => DirectorySource(hostPath) is not null;

    public static string? DirectorySource(string hostPath)
    {
        var current = Volatile.Read(ref _current);
        return current is not null && current.TryRelative(hostPath, out var relative) &&
               current._directoryLookup.TryGetValue(relative, out var directory) && Directory.Exists(directory.Path) && IsSafePath(directory.Root, directory.Path)
            ? directory.Path : null;
    }

    public static IEnumerable<string> List(string hostDirectory)
    {
        var current = Volatile.Read(ref _current);
        return current is not null && current.TryRelative(hostDirectory, out var relative) &&
               current._children.GetAlternateLookup<ReadOnlySpan<char>>().TryGetValue(relative, out var children) ? children : [];
    }

    public static IEnumerable<string> EnumerateFiles(string hostDirectory, string pattern, SearchOption option)
    {
        var current = Volatile.Read(ref _current);
        if (current is null || !current.TryRelativeString(hostDirectory, out var relative)) yield break;
        var prefix = relative.Length == 0 ? string.Empty : relative + Path.DirectorySeparatorChar;
        foreach (var file in current._files.Keys)
        {
            if (!file.StartsWith(prefix, current._comparison)) continue;
            var tail = file[prefix.Length..];
            if (option == SearchOption.TopDirectoryOnly && tail.Contains(Path.DirectorySeparatorChar)) continue;
            if (FileSystemName.MatchesSimpleExpression(pattern, Path.GetFileName(file), ignoreCase: current._comparison == StringComparison.OrdinalIgnoreCase))
                yield return Path.Combine(current._root, file);
        }
    }

    private bool TryRelativeString(string hostPath, out string relative)
    {
        var found = TryRelative(hostPath, out ReadOnlySpan<char> span);
        relative = found ? span.ToString() : string.Empty;
        return found;
    }

    // The path resolver hands over normalized absolute paths, so the common case is sliced in
    // place: no Path.GetFullPath, no substring. Anything that could hold a "." or ".." segment or a
    // doubled separator is normalized first, as before.
    private bool TryRelative(string hostPath, out ReadOnlySpan<char> relative)
    {
        relative = default;
        if (string.IsNullOrEmpty(hostPath)) return false;
        ReadOnlySpan<char> full = hostPath;
        if (OperatingSystem.IsWindows() || !Path.IsPathRooted(full) || full.Contains("/.", StringComparison.Ordinal) ||
            full.Contains("//", StringComparison.Ordinal))
        {
            full = Path.GetFullPath(hostPath);
        }
        full = Path.TrimEndingDirectorySeparator(full);
        if (full.Equals(_root, _comparison)) return true;
        if (full.Length <= _root.Length + 1 || full[_root.Length] != Path.DirectorySeparatorChar ||
            !full.StartsWith(_root, _comparison)) return false;
        relative = full[(_root.Length + 1)..];
        return true;
    }

    internal static bool IsSafePath(string root, string path)
    {
        root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
        path = Path.GetFullPath(path);
        if (!string.Equals(root, path, Comparison) && !path.StartsWith(root + Path.DirectorySeparatorChar, Comparison)) return false;
        try
        {
            for (var current = path; current is not null; current = Path.GetDirectoryName(current))
            {
                if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) return false;
                if (string.Equals(current, root, Comparison)) return true;
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException) { return false; }
        return false;
    }

    public void Dispose() => Interlocked.CompareExchange(ref _current, null, this);
}
