// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace SharpEmu.Libs.VideoOut;

// Opt-in actual contention measurements, rather than attributing sampled monitor
// frames (which include Rosetta safepoints) to an unknown lock or holder.
public static class GpuLockProfile
{
    public enum Gate { Modules, Windows, Resolve, Programs, ShaderIdentity, ShaderHeaders }
    private enum Role { Unknown, A, B, Guest, Compiler, Other }
    internal sealed class Owner { internal int Role; }
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_GPU_LOCK_PROFILE") == "1";
    private static readonly ConditionalWeakTable<object, Owner> Owners = new();
    private static readonly long[] Counts = new long[6 * 6 * 6];
    private static readonly long[] Ticks = new long[6 * 6 * 6];

    private static Role CurrentRole()
    {
        var name = Thread.CurrentThread.Name ?? "";
        if (name.Contains("issue consumer", StringComparison.OrdinalIgnoreCase)) return Role.B;
        if (name.Contains("GPU worker", StringComparison.OrdinalIgnoreCase)) return Role.A;
        if (name.Contains("BPE", StringComparison.OrdinalIgnoreCase)) return Role.Guest;
        if (name.Contains("compil", StringComparison.OrdinalIgnoreCase) || name.Contains("prewarm", StringComparison.OrdinalIgnoreCase)) return Role.Compiler;
        return Role.Other;
    }

    public static Scope Enter(object gate, Gate kind)
    {
        if (!Enabled) { Monitor.Enter(gate); return new(gate, null, 0); }
        var owner = Owners.GetValue(gate, static _ => new());
        if (!Monitor.TryEnter(gate))
        {
            // The owner observed at contention, not a claim that one holder accounts
            // for the entire wait if several handovers occur before admission.
            var observed = Volatile.Read(ref owner.Role);
            var index = (int)kind * 36 + (int)CurrentRole() * 6 + observed;
            var started = Stopwatch.GetTimestamp();
            Monitor.Enter(gate);
            Interlocked.Increment(ref Counts[index]);
            Interlocked.Add(ref Ticks[index], Stopwatch.GetTimestamp() - started);
        }
        var previous = owner.Role;
        Volatile.Write(ref owner.Role, (int)CurrentRole());
        return new(gate, owner, previous);
    }

    public readonly struct Scope : IDisposable
    {
        private readonly object _gate;
        private readonly Owner? _owner;
        private readonly int _previous;
        internal Scope(object gate, Owner? owner, int previous) { _gate = gate; _owner = owner; _previous = previous; }
        public void Dispose()
        {
            if (_owner is not null) Volatile.Write(ref _owner.Role, _previous);
            Monitor.Exit(_gate);
        }
    }

    public static string Format()
    {
        if (!Enabled) return "lock_profile=0";
        var result = new StringBuilder("lock_profile=1");
        for (var index = 0; index < Counts.Length; index++)
        {
            var count = Interlocked.Read(ref Counts[index]);
            if (count == 0) continue;
            result.Append(FormattableString.Invariant($" lock_{(Gate)(index / 36)}_{(Role)(index / 6 % 6)}_by_{(Role)(index % 6)}={count}/{Interlocked.Read(ref Ticks[index]) * 1000.0 / Stopwatch.Frequency:F3}"));
        }
        return result.ToString();
    }
}
