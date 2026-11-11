// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Core.Loader;

// A plan's key holds the build identity, so each new build leaves the previous build's plans behind
// and nothing reads them again. A hit stamps its plan as used; a miss, which is when a plan is added,
// drops the plans no boot has used for a while and then the least recently used ones over a size cap.
internal static partial class GuestRedZonePatcher
{
    internal static readonly TimeSpan PatchCacheMaximumAge = TimeSpan.FromDays(30);
    internal const long PatchCacheMaximumTotalBytes = 256L << 20;
    // A writer that died between creating its temporary file and renaming it.
    private static readonly TimeSpan AbandonedTemporaryAge = TimeSpan.FromHours(1);
    private const int PatchCacheKeyLength = 64;

    // The access time, not the write time: a hit must not look like a rewrite of the plan.
    private static void MarkPatchCacheUsed(string path)
    {
        try
        {
            File.SetLastAccessTimeUtc(path, DateTime.UtcNow);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // The plan was used either way; unstamped, it only ages out sooner.
        }
    }

    private static void PrunePatchCache(string keptPath)
    {
        if (Path.GetDirectoryName(keptPath) is { Length: > 0 } directory)
            PrunePatchCache(directory, keptPath, DateTime.UtcNow, PatchCacheMaximumAge, PatchCacheMaximumTotalBytes);
    }

    /// <summary>Removes this cache's own files from <paramref name="directory"/>: plans last used more than
    /// <paramref name="maximumAge"/> ago, then the least recently used plans until the rest fit in
    /// <paramref name="maximumTotalBytes"/>, and abandoned temporaries. <paramref name="keptPath"/>, the plan
    /// just written, stays. Returns the number of plans removed.</summary>
    internal static int PrunePatchCache(string directory, string? keptPath, DateTime nowUtc, TimeSpan maximumAge, long maximumTotalBytes)
    {
        var removed = 0;
        var removedBytes = 0L;
        var keptBytes = 0L;
        var failures = 0;
        try
        {
            var plans = new List<(FileInfo File, DateTime Used)>();
            foreach (var file in new DirectoryInfo(directory).EnumerateFiles())
            {
                if (IsAbandonedTemporary(file, nowUtc))
                {
                    if (!TryDelete(file)) failures++;
                }
                else if (IsPlanName(file.Name))
                {
                    plans.Add((file, LastUsedUtc(file)));
                    keptBytes += file.Length;
                }
            }

            plans.Sort(static (left, right) => left.Used.CompareTo(right.Used));
            var kept = keptPath is null ? null : Path.GetFullPath(keptPath);
            foreach (var (file, used) in plans)
            {
                var expired = nowUtc - used > maximumAge;
                if (!expired && keptBytes <= maximumTotalBytes)
                    break;
                if (string.Equals(file.FullName, kept, StringComparison.Ordinal))
                    continue;
                var length = file.Length;
                if (!TryDelete(file))
                {
                    failures++;
                    continue;
                }
                removed++;
                removedBytes += length;
                keptBytes -= length;
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            failures++;
        }

        if (removed > 0)
        {
            Console.Error.WriteLine(
                $"[LOADER][INFO] red-zone plan cache: removed {removed} unused plan(s) ({removedBytes / 1048576.0:F1} MB), " +
                $"{keptBytes / 1048576.0:F1} MB kept in {directory}.");
        }
        if (failures > 0)
            Console.Error.WriteLine($"[LOADER][WARN] red-zone plan cache: {failures} file(s) in {directory} could not be removed.");
        return removed;
    }

    // Whichever is later: a file system that does not keep access times still has the write time.
    private static DateTime LastUsedUtc(FileInfo file) =>
        file.LastAccessTimeUtc > file.LastWriteTimeUtc ? file.LastAccessTimeUtc : file.LastWriteTimeUtc;

    // "<key>.bin", the key being a SHA-256 in lower-case hex. Nothing else in the directory is touched.
    private static bool IsPlanName(ReadOnlySpan<char> name) =>
        name.Length == PatchCacheKeyLength + 4 && name.EndsWith(".bin", StringComparison.Ordinal) && IsLowerHex(name[..PatchCacheKeyLength]);

    // "<key>.bin.<guid>.tmp", as WritePatchCache names it.
    private static bool IsAbandonedTemporary(FileInfo file, DateTime nowUtc)
    {
        ReadOnlySpan<char> name = file.Name;
        const int GuidLength = 32;
        return name.Length == PatchCacheKeyLength + 4 + 1 + GuidLength + 4 && name.EndsWith(".tmp", StringComparison.Ordinal) &&
               IsPlanName(name[..(PatchCacheKeyLength + 4)]) && name[PatchCacheKeyLength + 4] == '.' &&
               IsLowerHex(name.Slice(PatchCacheKeyLength + 5, GuidLength)) && nowUtc - file.LastWriteTimeUtc > AbandonedTemporaryAge;
    }

    private static bool IsLowerHex(ReadOnlySpan<char> text) => text.IndexOfAnyExcept("0123456789abcdef") < 0;

    private static bool TryDelete(FileInfo file)
    {
        try
        {
            file.Delete();
            return true;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }
}
