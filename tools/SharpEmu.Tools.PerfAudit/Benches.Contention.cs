// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.Core.Cpu;
using SharpEmu.Core.Memory;
using SharpEmu.GpuRemote;
using SharpEmu.GpuSplit;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Host.MacOS;

namespace SharpEmu.PerfAudit;

/// <summary>
/// Managed guest-memory accesses from many threads at once, each thread in its own 64 KiB of a real
/// PhysicalVirtualMemory backing, so only the shared bookkeeping (alias pins, write intervals) is shared.
/// ns/op is window x threads / ops (thread time per op); min/max are per thread. Select with <c>contend</c>;
/// <c>--qos=ui</c> starts the workers at user-interactive QoS, <c>--only=a,b</c> limits the rows.
/// </summary>
internal static unsafe class ContentionBenches
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong ThreadStride = 0x1_0000;
    public static int[] ThreadCounts = [1, 2, 4, 10, 12];
    public static double WindowMs = 200;
    public static int Windows = 5;
    public static bool UserInteractive;
    public static string[]? Only;

    private sealed record Row(string Id, string Name, Dictionary<int, (double Ns, double Min, double Max, long Refusals)> ByThreads);
    private static readonly List<Row> Rows = [];

    // A body returns the refusals (false results) of one operation: 0 or 1.
    private delegate int Op(long iteration);

    [ThreadStatic] private static long _threadLocal;

    public static void Run(string[] args)
    {
        foreach (var a in args)
        {
            if (a == "--qos=ui") UserInteractive = true;
            if (a.StartsWith("--only=", StringComparison.Ordinal)) Only = a[7..].Split(',');
            if (a.StartsWith("--window=", StringComparison.Ordinal)) WindowMs = double.Parse(a[9..], CultureInfo.InvariantCulture);
            if (a.StartsWith("--threads=", StringComparison.Ordinal)) ThreadCounts = a[10..].Split(',').Select(int.Parse).ToArray();
        }

        Console.WriteLine($"--- contend: managed guest-memory access contention (qos={(UserInteractive ? "user-interactive" : "default")}, " +
            $"window {WindowMs} ms x {Windows}, threads {string.Join('/', ThreadCounts)}) ---");
        HostMemoryAuthority.Initialize(MacOSHost.CreateMemory);
        var views = MacOSHost.CreateViewMemory();
        using var pvm = new PhysicalVirtualMemory(viewHost: views, backingBytes: 32UL * 1024 * 1024);
        const ulong Region = 8UL << 20;
        var hole = (Region + views.Granularity - 1) / views.Granularity * views.Granularity;
        if (!pvm.TryHoldRangeAtOrAbove(GuestBase, hole, views.Granularity, out var address) ||
            !pvm.TryMapBacked(address, Region, 0, GuestPageProtection.Read | GuestPageProtection.Write, out _))
        {
            throw new InvalidOperationException("PVM map failed");
        }

        using var channel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("contend"), SplitSharedState.Layout);
        var state = new SplitSharedState(channel);
        state.SetPage(address + 0x40_0000);
        // PageGuard needs 4 KiB host pages: under native arm64 (16 KiB) the rows run without the GPU hooks.
        using var gpuMemory = Environment.SystemPageSize == 4096 ? new GuestGpuMemory(pvm) : null;
        if (gpuMemory is not null)
        {
            gpuMemory.Register(address, Region, GuestPageProtection.Read | GuestPageProtection.Write);
            gpuMemory.AttachStores(null, new AgcBuilderBenches.FilterStore(state));
            GuestGpuMemoryHook.Attach(gpuMemory);
        }

        Console.WriteLine($"GPU write hooks attached={gpuMemory is not null} (host page {Environment.SystemPageSize})");
        var memory = new TrackedCpuMemory(pvm);

        // The shared-backing objects behind the PVM, for the direct rows and the ledger A/B.
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        var owner = typeof(PhysicalVirtualMemory).GetField("_backedSpace", Any)!.GetValue(pvm)!;
        var sbv = (SharedBackingViews)owner.GetType().GetField("_views", Any)!.GetValue(owner)!;
        var backing = (HostBackingObject)typeof(SharedBackingViews).GetField("_backing", Any)!.GetValue(sbv)!;
        var managedWritesField = typeof(HostBackingObject).GetField("_managedWrites", Any)
            ?? throw new MissingFieldException("HostBackingObject._managedWrites");
        var ledger = managedWritesField.GetValue(backing);
        Console.WriteLine($"ledger enrolled={ledger is not null}");

        // A private (not backed) PVM region for p7.
        var privateBase = pvm.AllocateAt(GuestBase + 0x4000_0000, 16 * ThreadStride, executable: false);

        ulong Base(int thread) => address + 0x10_0000 + (ulong)thread * ThreadStride;
        var resetState = new byte[0x38];
        for (var t = 0; t < 16; t++)
        {
            BitConverter.TryWriteBytes(resetState.AsSpan(0x10), Base(t) + 0x1000);
            memory.TryWrite(Base(t) + 0x800, resetState);
        }

        // ---- PVM paths (TrackedCpuMemory over PhysicalVirtualMemory, hooks attached)
        void Memory(string suffix, string label)
        {
            Contend("r8" + suffix, "TryRead 8 B" + label, t =>
            {
                var b = Base(t); var buf = new byte[8];
                return i => memory.TryRead(b + (ulong)((i & 255) << 3), buf) ? 0 : 1;
            });
            Contend("r36" + suffix, "TryRead 0x24 B (command-buffer state)" + label, t =>
            {
                var b = Base(t) + 0x810; var buf = new byte[0x24];
                return i => memory.TryRead(b, buf) ? 0 : 1;
            });
            Contend("w8" + suffix, "TryWrite 8 B" + label, t =>
            {
                var b = Base(t); var buf = new byte[8];
                return i => memory.TryWrite(b + (ulong)((i & 255) << 3), buf) ? 0 : 1;
            });
            Contend("pair" + suffix, "TryWritePair 8 B cursor + 32 B packet" + label, t =>
            {
                var b = Base(t); var eight = new byte[8]; var packet = new byte[32];
                return i => memory.TryWritePair(b + 0x820, eight, b + 0x1000 + (ulong)((i & 127) << 5), packet) ? 0 : 1;
            });
            Contend("emit" + suffix, "emit: r36 + pair" + label, t =>
            {
                var b = Base(t); var eight = new byte[8]; var packet = new byte[32]; var cbState = new byte[0x24];
                return i =>
                {
                    var refused = memory.TryRead(b + 0x810, cbState) ? 0 : 1;
                    return refused | (memory.TryWritePair(b + 0x820, eight, b + 0x1000 + (ulong)((i & 127) << 5), packet) ? 0 : 1);
                };
            });

            // ---- SharedBackingViews directly (no hooks); a false return counts as a refusal.
            Contend("d-r8" + suffix, "SharedBackingViews.TryReadSingleView 8 B" + label, t =>
            {
                var b = Base(t); var buf = new byte[8];
                return i => sbv.TryReadSingleView(b + (ulong)((i & 255) << 3), buf) ? 0 : 1;
            });
            Contend("d-w8" + suffix, "SharedBackingViews.TryWriteSingleView 8 B" + label, t =>
            {
                var b = Base(t); var buf = new byte[8];
                return i => sbv.TryWriteSingleView(b + (ulong)((i & 255) << 3), buf) ? 0 : 1;
            });
            Contend("d-pair" + suffix, "SharedBackingViews.TryWritePairSingleView 8 + 32 B" + label, t =>
            {
                var b = Base(t); var eight = new byte[8]; var packet = new byte[32];
                return i => sbv.TryWritePairSingleView(b + 0x820, eight, b + 0x1000 + (ulong)((i & 127) << 5), packet) ? 0 : 1;
            });
        }

        Memory("", "");
        ReportLedger(ledger, "ledger enrolled");
        if (ledger is not null)
        {
            managedWritesField.SetValue(backing, null);
            try { Memory("-nl", " [_managedWrites null]"); }
            finally { managedWritesField.SetValue(backing, ledger); }
        }

        // ---- primitives
        var shared = new long[32];
        var padded = new long[32 * 32]; // 256-byte stride per thread
        Contend("p1", "Interlocked.Increment, one shared long", t => i => { Interlocked.Increment(ref shared[16]); return 0; });
        Contend("p2", "Interlocked.Increment, private 256-byte-padded slot", t =>
        {
            var slot = 32 * (t + 1);
            return i => { Interlocked.Increment(ref padded[slot]); return 0; };
        });
        Contend("p3", "[ThreadStatic] read-modify-write", t => i => { _threadLocal++; return 0; });
        Contend("p4", "Environment.CurrentManagedThreadId", t => i => Environment.CurrentManagedThreadId == -1 ? 1 : 0);
        Contend("p5", "Interlocked.MemoryBarrier", t => i => { Interlocked.MemoryBarrier(); return 0; });
        var rwls = new ReaderWriterLockSlim(LockRecursionPolicy.NoRecursion);
        Contend("p6", "ReaderWriterLockSlim read enter/exit, shared", t => i => { rwls.EnterReadLock(); rwls.ExitReadLock(); return 0; });
        Contend("p7", "PVM.TryRead 8 B, private (not backed) region", t =>
        {
            var b = privateBase + (ulong)t * ThreadStride; var buf = new byte[8];
            return i => pvm.TryRead(b + (ulong)((i & 255) << 3), buf) ? 0 : 1;
        });
        GuestGpuMemoryHook.Attach(null);
        Console.WriteLine();
        Console.WriteLine(Table());
    }

    private static void ReportLedger(object? ledger, string label)
    {
        if (ledger is null) return;
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        var type = ledger.GetType();
        var losses = type.GetProperty("HistoryRecordLosses", Any)?.GetValue(ledger);
        var coverage = type.GetProperty("FirstCoverageLoss", Any)?.GetValue(ledger);
        Console.WriteLine($"{label}: HistoryRecordLosses={losses} FirstCoverageLoss={coverage}");
    }

    private static void Contend(string id, string name, Func<int, Op> factory)
    {
        if (Only is not null && !Only.Contains(id)) return;
        // Single-threaded warmup so the op reaches tier 1 before any window.
        var warm = factory(0);
        var watch = Stopwatch.StartNew();
        for (long i = 0; watch.Elapsed.TotalMilliseconds < Harness.WarmupMs; i++) warm(i);

        var row = new Row(id, name, []);
        foreach (var threads in ThreadCounts)
        {
            row.ByThreads[threads] = Measure(threads, factory);
            var r = row.ByThreads[threads];
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{id,-10} T={threads,-2} {r.Ns,9:F1} ns/op  per-thread {r.Min,9:F1}..{r.Max,9:F1}  refusals {r.Refusals}  {name}"));
        }

        Rows.Add(row);
    }

    private static (double Ns, double Min, double Max, long Refusals) Measure(int threads, Func<int, Op> factory)
    {
        var stop = 0;
        var barrier = new Barrier(threads + 1);
        var elapsed = new double[threads];
        var ops = new long[threads];
        var refusals = new long[threads];
        var results = new List<(double Ns, double Min, double Max, long Refusals)>();
        var bodies = Enumerable.Range(0, threads).Select(factory).ToArray();

        void Worker(int t)
        {
            var op = bodies[t];
            long iteration = 0;
            for (var window = 0; window <= Windows; window++)
            {
                barrier.SignalAndWait();
                long count = 0, refused = 0;
                var start = Stopwatch.GetTimestamp();
                while (Volatile.Read(ref stop) == 0)
                {
                    for (var k = 0; k < 32; k++) refused += op(iteration++);
                    count += 32;
                }

                elapsed[t] = Stopwatch.GetElapsedTime(start).TotalNanoseconds;
                ops[t] = count;
                refusals[t] = refused;
                barrier.SignalAndWait();
            }
        }

        var started = new List<Thread>();
        var native = new List<HostQosThread>();
        for (var t = 0; t < threads; t++)
        {
            var index = t;
            if (UserInteractive)
            {
                var thread = HostQosThread.TryStart($"contend-{index}", HostThreadQos.UserInteractive, 1 << 20, () => Worker(index), out var errno)
                    ?? throw new InvalidOperationException($"QoS thread start failed: errno {errno}");
                native.Add(thread);
            }
            else
            {
                var thread = new Thread(() => Worker(index), 1 << 20) { IsBackground = true, Name = $"contend-{index}" };
                thread.Start();
                started.Add(thread);
            }
        }

        for (var window = 0; window <= Windows; window++)
        {
            Volatile.Write(ref stop, 0);
            barrier.SignalAndWait();
            Thread.Sleep(TimeSpan.FromMilliseconds(WindowMs));
            Volatile.Write(ref stop, 1);
            barrier.SignalAndWait();
            if (window == 0) continue; // warm window
            var perThread = Enumerable.Range(0, threads).Select(t => elapsed[t] / Math.Max(ops[t], 1)).ToArray();
            results.Add((elapsed.Sum() / Math.Max(ops.Sum(), 1), perThread.Min(), perThread.Max(), refusals.Sum()));
        }

        foreach (var thread in started) thread.Join();
        foreach (var thread in native) thread.Join(10_000);
        var sorted = results.OrderBy(r => r.Ns).ToArray();
        var median = sorted[sorted.Length / 2];
        return median with { Refusals = results.Sum(r => r.Refusals) };
    }

    private static string Table()
    {
        var sb = new StringBuilder();
        sb.Append(CultureInfo.InvariantCulture, $"contend arch={RuntimeInformation.ProcessArchitecture} qos={(UserInteractive ? "ui" : "default")} (ns/op, thread time per op; T=10 per-thread min..max; refusals over all windows)\n");
        sb.Append("| id | ").Append(string.Join(" | ", ThreadCounts.Select(t => $"T={t}"))).Append(" | T10/T1 | T=10 min..max | refusals | what |\n");
        sb.Append("|---|").Append(string.Concat(ThreadCounts.Select(_ => "---:|"))).Append("---:|---|---:|---|\n");
        foreach (var row in Rows)
        {
            sb.Append(CultureInfo.InvariantCulture, $"| {row.Id} | ");
            sb.Append(string.Join(" | ", ThreadCounts.Select(t => row.ByThreads[t].Ns.ToString("F1", CultureInfo.InvariantCulture))));
            var ratio = row.ByThreads.TryGetValue(10, out var ten) && row.ByThreads.TryGetValue(1, out var one) ? ten.Ns / one.Ns : double.NaN;
            var spread = row.ByThreads.TryGetValue(10, out ten) ? $"{ten.Min:F0}..{ten.Max:F0}" : "-";
            sb.Append(CultureInfo.InvariantCulture, $" | {ratio:F1}x | {spread} | {row.ByThreads.Values.Sum(v => v.Refusals)} | {row.Name} |\n");
        }

        return sb.ToString();
    }
}
