// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class GuestRedZoneCachePruneTests : IDisposable
{
    private static readonly DateTime Now = new(2026, 11, 11, 12, 0, 0, DateTimeKind.Utc);
    private static readonly TimeSpan MaximumAge = TimeSpan.FromDays(30);
    private readonly DirectoryInfo _directory = Directory.CreateTempSubdirectory("sharpemu-redzone-prune-");

    public void Dispose() => _directory.Delete(recursive: true);

    [Fact]
    public void PlansUnusedForLongerThanTheMaximumAgeAreRemoved()
    {
        var fresh = Plan('a', bytes: 10, usedDaysAgo: 1);
        var edge = Plan('b', bytes: 10, usedDaysAgo: 29);
        var stale = Plan('c', bytes: 10, usedDaysAgo: 31);

        Assert.Equal(1, Prune(maximumBytes: 1000));

        Assert.True(File.Exists(fresh));
        Assert.True(File.Exists(edge));
        Assert.False(File.Exists(stale));
    }

    [Fact]
    public void APlanReadRecentlyStaysHoweverLongAgoItWasWritten()
    {
        var plan = Plan('a', bytes: 10, usedDaysAgo: 400);
        File.SetLastAccessTimeUtc(plan, Now.AddDays(-2));

        Assert.Equal(0, Prune(maximumBytes: 1000));

        Assert.True(File.Exists(plan));
    }

    [Fact]
    public void OverTheSizeCapTheLeastRecentlyUsedPlansGoFirst()
    {
        var oldest = Plan('a', bytes: 400, usedDaysAgo: 9);
        var middle = Plan('b', bytes: 400, usedDaysAgo: 5);
        var newest = Plan('c', bytes: 400, usedDaysAgo: 1);

        Assert.Equal(1, Prune(maximumBytes: 900));

        Assert.False(File.Exists(oldest));
        Assert.True(File.Exists(middle));
        Assert.True(File.Exists(newest));
    }

    [Fact]
    public void ThePlanJustWrittenStaysEvenWhenItAloneExceedsTheCapOrLooksStale()
    {
        var written = Plan('a', bytes: 2000, usedDaysAgo: 90);
        var other = Plan('b', bytes: 100, usedDaysAgo: 1);

        Assert.Equal(1, Prune(maximumBytes: 900, kept: written));

        Assert.True(File.Exists(written));
        Assert.False(File.Exists(other));
    }

    [Fact]
    public void OnlyTheCachesOwnFilesAreTouched()
    {
        var notes = Path.Combine(_directory.FullName, "notes.txt");
        var shortName = Path.Combine(_directory.FullName, "abc.bin");
        var upperCase = Path.Combine(_directory.FullName, new string('A', 64) + ".bin");
        var key = new string('d', 64);
        var abandoned = Path.Combine(_directory.FullName, $"{key}.bin.{Guid.NewGuid():N}.tmp");
        var inProgress = Path.Combine(_directory.FullName, $"{key}.bin.{Guid.NewGuid():N}.tmp");
        foreach (var path in new[] { notes, shortName, upperCase, abandoned, inProgress })
        {
            File.WriteAllBytes(path, new byte[10]);
            File.SetLastWriteTimeUtc(path, Now.AddDays(-365));
            File.SetLastAccessTimeUtc(path, Now.AddDays(-365));
        }
        File.SetLastWriteTimeUtc(inProgress, Now.AddMinutes(-5));

        Assert.Equal(0, Prune(maximumBytes: 0));

        Assert.True(File.Exists(notes));
        Assert.True(File.Exists(shortName));
        Assert.True(File.Exists(upperCase));
        Assert.True(File.Exists(inProgress));
        Assert.False(File.Exists(abandoned));
    }

    [Fact]
    public void AMissingDirectoryIsNotAnError() =>
        Assert.Equal(0, GuestRedZonePatcher.PrunePatchCache(
            Path.Combine(_directory.FullName, "absent"), keptPath: null, Now, MaximumAge, maximumTotalBytes: 0));

    private int Prune(long maximumBytes, string? kept = null) =>
        GuestRedZonePatcher.PrunePatchCache(_directory.FullName, kept, Now, MaximumAge, maximumBytes);

    private string Plan(char key, int bytes, int usedDaysAgo)
    {
        var path = Path.Combine(_directory.FullName, new string(key, 64) + ".bin");
        File.WriteAllBytes(path, new byte[bytes]);
        File.SetLastWriteTimeUtc(path, Now.AddDays(-usedDaysAgo));
        File.SetLastAccessTimeUtc(path, Now.AddDays(-usedDaysAgo));
        return path;
    }
}
