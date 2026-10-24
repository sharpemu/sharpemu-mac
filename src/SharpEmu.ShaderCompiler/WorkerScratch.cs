// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler;

/// <summary>
/// The sites of the GPU worker path that reuse an object or a buffer a call used to allocate.
/// SHARPEMU_WORKER_SCRATCH_REUSE=0 restores the allocating form of every site at once; each site
/// counts how often it reused and how often it allocated, and the Metal host prints the counts
/// beside its other per-window counters.
/// </summary>
public static class WorkerScratch
{
    public enum Site
    {
        ReadLogRuns,
        VertexBindings,
        Count,
    }

    private static readonly string[] Names = ["readlog", "vertex_bindings"];

    // Read once; tests set it per scope through Override.
    public static readonly bool EnabledByEnvironment = PerfSwitches.Enabled("SHARPEMU_WORKER_SCRATCH_REUSE");

    [ThreadStatic]
    private static bool? _override;

    public static bool Enabled => _override ?? EnabledByEnvironment;

    /// <summary>Forces the switch on this thread until the scope is disposed; for tests.</summary>
    public static IDisposable Override(bool enabled)
    {
        var previous = _override;
        _override = enabled;
        return new Restore(previous);
    }

    private sealed class Restore(bool? previous) : IDisposable
    {
        public void Dispose() => _override = previous;
    }

    // Plain increments: the sites run on the worker (or under their own lock), and a count lost
    // to a race is not worth an interlocked add on a per-draw path.
    private static readonly long[] Reused = new long[(int)Site.Count];
    private static readonly long[] Allocated = new long[(int)Site.Count];

    public static void Note(Site site, bool reused)
    {
        if (reused) Reused[(int)site]++;
        else Allocated[(int)site]++;
    }

    public static (long Reused, long Allocated) Peek(Site site) => (Reused[(int)site], Allocated[(int)site]);

    /// <summary>The counts since the last call as " scratch_site_reused=n scratch_site_alloc=n" fields; resets.</summary>
    public static string TakeCounters()
    {
        var text = new System.Text.StringBuilder(" scratch_reuse=").Append(EnabledByEnvironment ? 1 : 0);
        for (var index = 0; index < (int)Site.Count; index++)
        {
            var reused = Interlocked.Exchange(ref Reused[index], 0);
            var allocated = Interlocked.Exchange(ref Allocated[index], 0);
            text.Append(" scratch_").Append(Names[index]).Append("_reused=").Append(reused)
                .Append(" scratch_").Append(Names[index]).Append("_alloc=").Append(allocated);
        }

        return text.ToString();
    }
}
