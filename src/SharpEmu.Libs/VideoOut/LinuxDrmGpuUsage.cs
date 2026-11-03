// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.VideoOut;

/// <summary>
/// Linux per-process GPU busy share from the DRM client statistics in /proc/self/fdinfo
/// ("drm-engine-&lt;name&gt;: &lt;ns&gt; ns", amdgpu, i915, xe, panfrost, msm and nouveau on recent
/// kernels). Each open DRM client reports its cumulative engine time; the share is the busiest
/// engine's time delta over the wall-clock window. NaN when no driver reports it.
/// </summary>
internal sealed class LinuxDrmGpuUsage : IGpuUsage
{
    private readonly Dictionary<string, long> _previous = new(StringComparer.Ordinal);
    private long _previousTimestamp;
    private double _percent = double.NaN;
    private bool _unsupported;

    public double Percent => Volatile.Read(ref _percent);

    public string SourceName => "DRM fdinfo, busiest engine";

    public void RequestSample()
    {
        if (_unsupported || !OperatingSystem.IsLinux())
        {
            return;
        }

        var now = Stopwatch.GetTimestamp();
        var totals = ReadEngineTotals();
        if (totals is null)
        {
            _unsupported = true;
            return;
        }

        if (_previousTimestamp != 0)
        {
            var wallNs = (now - _previousTimestamp) * 1_000_000_000.0 / Stopwatch.Frequency;
            var busiest = 0.0;
            foreach (var (engine, ns) in totals)
            {
                if (_previous.TryGetValue(engine, out var before) && ns >= before && wallNs > 0)
                {
                    busiest = Math.Max(busiest, (ns - before) / wallNs * 100.0);
                }
            }

            Volatile.Write(ref _percent, totals.Count > 0 ? Math.Min(100, busiest) : double.NaN);
        }

        _previous.Clear();
        foreach (var pair in totals) _previous[pair.Key] = pair.Value;
        _previousTimestamp = now;
    }

    /// <summary>Sums engine nanoseconds per engine over distinct DRM clients; null when /proc is unreadable.</summary>
    internal static Dictionary<string, long>? ReadEngineTotals(string directory = "/proc/self/fdinfo")
    {
        var totals = new Dictionary<string, long>(StringComparer.Ordinal);
        var clients = new HashSet<string>(StringComparer.Ordinal);
        IEnumerable<string> files;
        try
        {
            files = Directory.EnumerateFiles(directory);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return null;
        }

        foreach (var file in files)
        {
            string text;
            try
            {
                text = File.ReadAllText(file);
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                continue;
            }

            if (!text.Contains("drm-engine-", StringComparison.Ordinal))
            {
                continue;
            }

            ParseFdInfo(text, clients, totals);
        }

        return totals;
    }

    /// <summary>Adds one fdinfo file's engine times, skipping a client id already counted (dup'd fds).</summary>
    internal static void ParseFdInfo(string text, HashSet<string> clients, Dictionary<string, long> totals)
    {
        string? client = null;
        var engines = new List<(string Engine, long Ns)>();
        foreach (var rawLine in text.Split('\n'))
        {
            var line = rawLine.Trim();
            var colon = line.IndexOf(':');
            if (colon <= 0) continue;
            var key = line[..colon];
            var value = line[(colon + 1)..].Trim();
            if (key == "drm-client-id")
            {
                client = value;
            }
            else if (key.StartsWith("drm-engine-", StringComparison.Ordinal) && !key.StartsWith("drm-engine-capacity-", StringComparison.Ordinal))
            {
                var number = value.Split(' ', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault();
                if (long.TryParse(number, out var ns))
                {
                    engines.Add((key["drm-engine-".Length..], ns));
                }
            }
        }

        if (client is not null && !clients.Add(client))
        {
            return;
        }

        foreach (var (engine, ns) in engines)
        {
            totals[engine] = totals.GetValueOrDefault(engine) + ns;
        }
    }

    public void Dispose()
    {
    }
}

/// <summary>Uses the backend's command-buffer intervals when it reports any, otherwise an OS counter.</summary>
internal sealed class PreferredGpuUsage(MacGpuUsage intervals, IGpuUsage? fallback) : IGpuUsage
{
    public double Percent => intervals.HasSamples || fallback is null ? intervals.Percent : fallback.Percent;

    public string SourceName => intervals.HasSamples || fallback is null ? intervals.SourceName : fallback.SourceName;

    /// <summary>Both sources, for the startup log: which one is live is only known once frames arrive.</summary>
    public string Describe() => fallback is null ? intervals.SourceName : $"{intervals.SourceName}, else {fallback.SourceName}";

    public void RequestSample()
    {
        intervals.RequestSample();
        fallback?.RequestSample();
    }

    public void Dispose()
    {
        intervals.Dispose();
        fallback?.Dispose();
    }
}
