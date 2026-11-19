// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Reflection;
using SharpEmu.Core.Memory;
using SharpEmu.HLE;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Core.Memory.Tests;

// The lock-free single-view guest write on an actual managed-write-enrolled Mac backing:
// it bypasses the PVM gate, a held publication still excludes it (no bytes, then exactly
// one write through the locked busy-receipt path), and a view-crossing miss does not
// invalidate unrelated read watches on the backing.
internal static partial class PvmMappedCopyWorker
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    private const long PublicationBit = 1L << 32;

    private static bool IsSingleViewWriteScenario(string scenario) =>
        scenario is "single-view-write-unlocked" or "single-view-write-publication-busy" or "single-view-miss-keeps-read-watch";

#if SHARPEMU_HOST_MACOS
    private static void ExecuteSingleViewWrite(PhysicalVirtualMemory pvm, IHostViewMemory views, ulong kernelPage, string scenario)
    {
        var page = GuestMemoryLayout.GuestPage;
        var extent = 3 * page;
        var alignment = Math.Max(views.Granularity, page);
        Assert.True(pvm.TryHoldRangeAtOrAbove(0x2_0000_0000, extent, alignment, out var address));
        Assert.True(pvm.TryHoldRange(address, extent));
        // Three separate views, adjacent in guest space but not in the backing, so no
        // two of them can resolve as one alias and a write crossing A|B misses the
        // single-view lookup. C is the disjoint watched view.
        var rw = GuestPageProtection.Read | GuestPageProtection.Write;
        Assert.True(pvm.TryMapBacked(address, page, page, rw, out var failure), $"map A failure={failure}");
        Assert.True(pvm.TryMapBacked(address + page, page, 3 * page, rw, out failure), $"map B failure={failure}");
        Assert.True(pvm.TryMapBacked(address + 2 * page, page, 5 * page, rw, out failure), $"map C failure={failure}");
        Assert.True(pvm.TryWriteBacking(address, new byte[checked((int)extent)]));
        var space = (GuestSpaceOwner)typeof(PhysicalVirtualMemory).GetField("_backedSpace", Private)!.GetValue(pvm)!;
        var shared = typeof(GuestSpaceOwner).GetField("_views", Private)!.GetValue(space)!;
        var backing = (HostBackingObject)typeof(SharedBackingViews).GetField("_backing", Private)!.GetValue(shared)!;
        var ledger = typeof(HostBackingObject).GetField("_managedWrites", Private)!.GetValue(backing);
        Assert.NotNull(ledger); // the Mac provider enrolls every backing at construction
        // The ledger stays dormant until its first observer; these scenarios count admitted
        // intervals, so observe it first as a mapped copy's revision capture would.
        Assert.True((bool)ledger!.GetType().GetMethod("TryCaptureRevision", Private)!.Invoke(ledger, [-1L])!);
        Stage("owned-map", scenario);
        Console.WriteLine($"pvm-single-view-write address=0x{address:X} page={page} kernel_page={kernelPage}");

        if (scenario == "single-view-write-unlocked") Unlocked(pvm, ledger!, address);
        else if (scenario == "single-view-write-publication-busy") PublicationBusy(pvm, space, ledger!, address);
        else MissKeepsReadWatch(pvm, space, ledger!, address, page);
    }

    // (a) The write completes while another thread holds the PVM write lock; (d) a warmed
    // PVM.TryWrite into a single view does not allocate.
    private static void Unlocked(PhysicalVirtualMemory pvm, object ledger, ulong address)
    {
        var gate = (ReaderWriterLockSlim)typeof(PhysicalVirtualMemory).GetField("_gate", Private)!.GetValue(pvm)!;
        byte[] payload = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        var begun = Count(ledger, "begun");
        using var held = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var holder = new Thread(() =>
        {
            gate.EnterWriteLock();
            try { held.Set(); release.Wait(); }
            finally { gate.ExitWriteLock(); }
        });
        holder.Start();
        bool finished;
        Task<bool> writer;
        try
        {
            Assert.True(held.Wait(TimeSpan.FromSeconds(10)));
            writer = Task.Factory.StartNew(() => pvm.TryWrite(address + 64, payload), TaskCreationOptions.LongRunning);
            finished = writer.Wait(TimeSpan.FromSeconds(10));
        }
        finally
        {
            release.Set();
            Assert.True(holder.Join(TimeSpan.FromSeconds(10)));
        }
        Assert.True(finished, "single-view PVM.TryWrite blocked on the PVM write lock");
        Assert.True(writer.Result);
        Assert.Equal(begun + 1, Count(ledger, "begun"));
        var readBack = new byte[payload.Length];
        Assert.True(pvm.TryRead(address + 64, readBack));
        Assert.Equal(payload, readBack);

        for (var i = 0; i < 10000; i++) Assert.True(pvm.TryWrite(address + 128, payload));
        var before = GC.GetAllocatedBytesForCurrentThread(); var allSucceeded = true;
        for (var i = 0; i < 10000; i++) allSucceeded &= pvm.TryWrite(address + 128, payload);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allSucceeded);
        Assert.Equal(0, allocated);
        Assert.Equal(0, Count(ledger, "active"));
        Assert.Equal(0, Count(ledger, "aborted"));
    }

    // (b) With a publication reservation held, the single-view path writes nothing and
    // holds nothing; PVM.TryWrite falls through to the locked path, waits on its busy
    // receipt outside the gate, and lands exactly one write once the publication ends.
    private static void PublicationBusy(PhysicalVirtualMemory pvm, GuestSpaceOwner space, object ledger, ulong address)
    {
        var admission = ledger.GetType().GetField("admission", Private)!;
        var releasePublication = ledger.GetType().GetMethod("ReleasePublication", Private)!;
        byte[] payload = [0xA1, 0xB2, 0xC3, 0xD4];
        var target = address + 256;
        var begun = Count(ledger, "begun"); var completed = Count(ledger, "completed"); var aborted = Count(ledger, "aborted");
        // Quiescent: no writer is in flight, so a plain store is the same state the
        // publication's 0 -> PublicationBit CAS publishes.
        Assert.Equal(0L, (long)admission.GetValue(ledger)!);
        admission.SetValue(ledger, PublicationBit);
        var released = false;
        try
        {
            Assert.False(space.TryWriteSingleView(target, payload));
            Assert.Equal(PublicationBit, (long)admission.GetValue(ledger)!);
            Assert.Equal(begun, Count(ledger, "begun"));
            AssertBytes(pvm, target, new byte[payload.Length]);

            using var started = new ManualResetEventSlim();
            var writer = Task.Factory.StartNew(() => { started.Set(); return pvm.TryWrite(target, payload); },
                TaskCreationOptions.LongRunning);
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            Thread.Sleep(100); // well inside the 500 ms ordinary admission wait
            Assert.False(writer.IsCompleted, "PVM.TryWrite did not wait for the held publication");
            Assert.Equal(begun, Count(ledger, "begun"));
            AssertBytes(pvm, target, new byte[payload.Length]);

            releasePublication.Invoke(ledger, [null]);
            released = true;
            Assert.True(writer.Wait(TimeSpan.FromSeconds(10)));
            Assert.True(writer.Result);
        }
        finally
        {
            if (!released) releasePublication.Invoke(ledger, [null]);
        }
        AssertBytes(pvm, target, payload);
        Assert.Equal(begun + 1, Count(ledger, "begun"));
        Assert.Equal(completed + 1, Count(ledger, "completed"));
        Assert.Equal(aborted, Count(ledger, "aborted"));
        Assert.Equal(0L, (long)admission.GetValue(ledger)!);
    }

    // (c) A view-crossing write misses the single-view lookup before any admission, so it
    // never ends an operation as failed (which would invalidate every read watch on the
    // backing); a disjoint read watch stays current through the locked multi-view write.
    private static void MissKeepsReadWatch(PhysicalVirtualMemory pvm, GuestSpaceOwner space, object ledger, ulong address, ulong page)
    {
        var watched = address + 2 * page;
        var copy = new byte[checked((int)page)];
        Assert.Equal(CpuMappedCopyResult.Copied, pvm.TryReadMappedCopyWithWriteWatch(watched, page, watched, copy, out var observation));
        Assert.NotNull(observation);
        try
        {
            Assert.True(pvm.IsMappedCopyWriteWatchCurrent(observation!));
            byte[] payload = [1, 2, 3, 4, 5, 6, 7, 8];
            var crossing = address + page - 4;
            var begun = Count(ledger, "begun");
            Assert.False(space.TryWriteSingleView(crossing, payload));
            Assert.Equal(begun, Count(ledger, "begun"));
            Assert.True(pvm.IsMappedCopyWriteWatchCurrent(observation!));

            Assert.True(pvm.TryWrite(crossing, payload));
            AssertBytes(pvm, crossing, payload);
            Assert.Equal(begun + 2, Count(ledger, "begun")); // one chunk per view
            Assert.Equal(0, Count(ledger, "aborted"));
            Assert.True(pvm.IsMappedCopyWriteWatchCurrent(observation!));

            // Control: the watch is live, an overlapping single-view write does stale it.
            Assert.True(pvm.TryWrite(watched + 8, payload));
            Assert.False(pvm.IsMappedCopyWriteWatchCurrent(observation!));
        }
        finally { pvm.ReleaseMappedCopyWriteWatch(observation!); }
    }

    private static void AssertBytes(PhysicalVirtualMemory pvm, ulong address, byte[] expected)
    {
        var actual = new byte[expected.Length];
        Assert.True(pvm.TryRead(address, actual));
        Assert.Equal(expected, actual);
    }

    private static long Count(object ledger, string name) => (long)ledger.GetType().GetField(name, Private)!.GetValue(ledger)!;
#endif
}
