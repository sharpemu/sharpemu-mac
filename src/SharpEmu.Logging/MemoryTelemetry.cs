// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace SharpEmu.Logging;

/// <summary>
/// A named, monotonically increasing count that <see cref="MemoryTelemetry"/> prints with each
/// report as "name=total(+interval)". <see cref="Add"/> is a no-op unless SHARPEMU_LOG_MEMORY=1.
/// </summary>
public sealed class MemoryCounter
{
    private long _total;
    private long _reported;

    internal MemoryCounter(string name)
    {
        Name = name;
    }

    public string Name { get; }

    public long Total => Volatile.Read(ref _total);

    public void Add(long amount = 1)
    {
        if (MemoryTelemetry.Enabled)
        {
            Interlocked.Add(ref _total, amount);
        }
    }

    internal (long Total, long Delta) Take()
    {
        var total = Volatile.Read(ref _total);
        var delta = total - _reported;
        _reported = total;
        return (total, delta);
    }
}

/// <summary>
/// SHARPEMU_LOG_MEMORY=1 turns on periodic "[MEM]" lines on stderr (game.log), every
/// SHARPEMU_LOG_MEMORY_SECONDS (default 10). One line per section: "dotnet" is built in
/// (managed heap, GC counts and pauses, thread count); subsystems add theirs with
/// <see cref="Register"/> (Metal: device working set, image census, guest windows named per
/// command buffer, GPU reference census; AGC: LOD statistics) and cheap counters with
/// <see cref="Counter"/>, printed together on the "counters" line. Off, nothing runs and a
/// counter's Add is one static readonly check.
/// </summary>
public static class MemoryTelemetry
{
    public static readonly bool Enabled =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_MEMORY"), "1", StringComparison.Ordinal);

    private static readonly object Gate = new();
    private static readonly List<(string Section, Func<string> Provider)> Providers = [];
    private static readonly List<MemoryCounter> Counters = [];
    private static bool _started;

    /// <summary>The counter of this name, created on first use; keep the reference in a static field.</summary>
    public static MemoryCounter Counter(string name)
    {
        lock (Gate)
        {
            foreach (var existing in Counters)
            {
                if (existing.Name == name)
                {
                    return existing;
                }
            }

            var counter = new MemoryCounter(name);
            Counters.Add(counter);
            return counter;
        }
    }

    /// <summary>
    /// Adds a section; the provider returns the text after "[MEM] section " and runs on the
    /// reporting thread, so it has to take whatever lock its data needs. No-op when off.
    /// </summary>
    public static void Register(string section, Func<string> provider)
    {
        if (!Enabled)
        {
            return;
        }

        lock (Gate)
        {
            Providers.RemoveAll(entry => entry.Section == section);
            Providers.Add((section, provider));
        }

        Start();
    }

    /// <summary>Starts the reporting thread when enabled; safe to call more than once.</summary>
    public static void Start()
    {
        if (!Enabled)
        {
            return;
        }

        lock (Gate)
        {
            if (_started)
            {
                return;
            }

            _started = true;
        }

        var seconds = int.TryParse(
            Environment.GetEnvironmentVariable("SHARPEMU_LOG_MEMORY_SECONDS"), out var configured) && configured > 0
            ? configured
            : 10;
        new Thread(() => Run(seconds)) { IsBackground = true, Name = "memory-telemetry" }.Start();
    }

    private static void Run(int seconds)
    {
        var lastPause = TimeSpan.Zero;
        var started = Stopwatch.GetTimestamp();
        while (true)
        {
            Thread.Sleep(TimeSpan.FromSeconds(seconds));
            var elapsed = Stopwatch.GetElapsedTime(started).TotalSeconds;
            var pause = GC.GetTotalPauseDuration();
            try
            {
                Report(elapsed, pause - lastPause, pause);
            }
            catch (Exception exception)
            {
                // Even taking the provider snapshot can fail. Optional diagnostics must
                // not take down the emulation process from this background thread.
                WriteError(Console.Error, "report", exception);
            }
            lastPause = pause;
        }
    }

    private static void Report(double elapsed, TimeSpan intervalPause, TimeSpan totalPause)
    {
        var culture = CultureInfo.InvariantCulture;
        (string Section, Func<string> Provider)[] providers;
        MemoryCounter[] counters;
        lock (Gate)
        {
            providers = [.. Providers];
            counters = [.. Counters];
        }

        ReportSections(Console.Error,
            () => $"t={elapsed.ToString("F0", culture)} {Dotnet(intervalPause, totalPause)}", providers);

        if (counters.Length != 0)
        {
            var line = new StringBuilder("[MEM] counters");
            foreach (var counter in counters)
            {
                var (total, delta) = counter.Take();
                line.Append(' ').Append(counter.Name).Append('=').Append(total).Append("(+").Append(delta).Append(')');
            }

            Console.Error.WriteLine(line.ToString());
        }
    }

    // The built-in process sampler has the same failure boundary as registered providers.
    // Process inspection can fail as host threads/process resources change underneath it.
    internal static void ReportSections(TextWriter output, Func<string> dotnet,
        IReadOnlyList<(string Section, Func<string> Provider)> providers)
    {
        WriteSection(output, "dotnet", dotnet);
        foreach (var (section, provider) in providers)
            WriteSection(output, section, provider);
    }

    private static void WriteSection(TextWriter output, string section, Func<string> provider)
    {
        try
        {
            output.WriteLine($"[MEM] {section} {provider()}");
        }
        catch (Exception exception)
        {
            WriteError(output, section, exception);
        }
    }

    private static void WriteError(TextWriter output, string section, Exception exception)
    {
        try
        {
            output.WriteLine($"[MEM] {section} error={exception.GetType().Name}: {exception.Message}");
        }
        catch (Exception)
        {
            // A closed/failed diagnostic sink cannot be used to report its own failure.
        }
    }

    private static string Dotnet(TimeSpan intervalPause, TimeSpan totalPause)
    {
        const double Mb = 1048576.0;
        var culture = CultureInfo.InvariantCulture;
        var info = GC.GetGCMemoryInfo();
        var generations = info.GenerationInfo;
        var gens = new StringBuilder();
        string[] names = ["gen0", "gen1", "gen2", "loh", "poh"];
        for (var index = 0; index < generations.Length && index < names.Length; index++)
        {
            gens.Append(' ').Append(names[index]).Append("_mb=")
                .Append((generations[index].SizeAfterBytes / Mb).ToString("F1", culture));
        }

        int threads;
        using (var process = Process.GetCurrentProcess())
        {
            threads = process.Threads.Count;
        }

        return string.Create(culture,
            $"heap_mb={info.HeapSizeBytes / Mb:F1} committed_mb={info.TotalCommittedBytes / Mb:F1} " +
            $"fragmented_mb={info.FragmentedBytes / Mb:F1} available_mb={info.TotalAvailableMemoryBytes / Mb:F0} " +
            $"live_mb={GC.GetTotalMemory(false) / Mb:F1}{gens} pinned={info.PinnedObjectsCount} " +
            $"gc0={GC.CollectionCount(0)} gc1={GC.CollectionCount(1)} gc2={GC.CollectionCount(2)} " +
            $"pause_ms={totalPause.TotalMilliseconds:F0}(+{intervalPause.TotalMilliseconds:F0}) " +
            $"alloc_mb={GC.GetTotalAllocatedBytes(false) / Mb:F0} threads={threads}");
    }
}
