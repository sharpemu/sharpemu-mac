// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using SharpEmu.HLE;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Host.Memory.Tests;

// Internal ordinary bookkeeping + actual SharedBackingViews pinned-array
// destination controls. No native permissions, signals or content authority.
public sealed unsafe class ManagedWriteIntervalTests
{
    public ManagedWriteIntervalTests(ITestOutputHelper output)
    {
        output.WriteLine($"managed-write-interval architecture={RuntimeInformation.ProcessArchitecture}; framework={RuntimeInformation.FrameworkDescription}; managed-only");
        var expected = Environment.GetEnvironmentVariable("SHARPEMU_TEST_EXPECT_ARCH");
        if (!string.IsNullOrEmpty(expected)) Assert.Equal(expected, RuntimeInformation.ProcessArchitecture.ToString());
    }

    private sealed class State : IDisposable
    {
        internal const ulong Page = 16384, Address = 0x100000000;
        internal readonly FakeProvider Host = new();
        internal readonly SharedBackingViews Views;
        internal HostBackingObject Backing => Host.Backing!;
        internal State()
        {
            Views = new(Host, Page * 2);
            Assert.True(Views.TryMapReservedRange(Address, Page, 0, HostPageProtection.ReadWrite, out _));
            Assert.True(Views.TryMapReservedRange(Address + Page, Page, Page, HostPageProtection.ReadWrite, out _));
        }
        // Enrolls through a mapped copy, whose revision capture is the ledger's first observer:
        // every write after it takes admission, identity, slot and history.
        internal object Enroll()
        {
            Assert.Equal(CpuMappedCopyResult.Copied, Views.TryReadMappedCopy(Address, Page, Address, new byte[1], out _));
            var ledger = typeof(HostBackingObject).GetField("_managedWrites", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(Backing)!;
            Assert.True((bool)ledger.GetType().GetProperty("IsObserved", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(ledger)!);
            return ledger;
        }
        // Enrolls with no observer: writes stay dormant (unless SHARPEMU_WRITE_INTERVALS_DORMANT=0).
        internal object EnrollDormant()
        {
            typeof(HostBackingObject).GetMethod("EnrollManagedWrites", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(Backing, null);
            return typeof(HostBackingObject).GetField("_managedWrites", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(Backing)!;
        }
        internal byte[] Bytes() => new ReadOnlySpan<byte>((void*)Backing.AliasBase, (int)(Page * 2)).ToArray();
        public void Dispose() { Views.Dispose(); Host.Dispose(); }
    }

    private sealed class FakeProvider : IHostViewMemory, IHostMappedCopyMemory, IDisposable
    {
        private readonly byte[] storage = new byte[checked((int)(State.Page * 3))];
        private GCHandle pin;
        private readonly object gate = new();
        internal readonly HostMappedCopyLedger Ledger;
        internal HostBackingObject? Backing;
        internal FakeProvider() { pin = GCHandle.Alloc(storage, GCHandleType.Pinned); Ledger = new(gate, State.Page); }
        public ulong PageSize => State.Page;
        public ulong Granularity => PageSize;
        public bool TryCreateBacking(ulong size, out HostBackingObject? backing, out HostViewFailure failure)
        {
            Assert.Equal(State.Page * 2, size);
            var alias = ((ulong)pin.AddrOfPinnedObject() + PageSize - 1) & ~(PageSize - 1);
            Backing = backing = new(alias, size, PageSize, _ => new(HostBackingReleaseStatus.Released));
            lock (gate) Ledger.RegisterBacking(backing);
            failure = HostViewFailure.None; return true;
        }
        public bool TryMapView(HostBackingObject backing, ulong address, ulong offset, ulong size, HostPageProtection protection, out HostViewFailure failure)
        {
            lock (gate)
            {
                var prepared = Ledger.PrepareView(backing, address, size, offset, protection);
                Assert.NotNull(prepared); Assert.True(Ledger.BeforeNativeEffect(address, size)); Ledger.PublishView(prepared);
            }
            failure = HostViewFailure.None; return true;
        }
        public bool UnmapView(ulong address, ulong size) { lock (gate) return Ledger.BeforeNativeEffect(address, size); }
        public CpuMappedCopyResult TryReadMappedCopy(HostBackingObject backing, ulong footprintAddress, ulong footprintByteCount,
            ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation) =>
            Ledger.TryReadMappedCopy(backing, footprintAddress, footprintByteCount, sourceAddress, destination, out observation);
        public bool IsMappedCopyObservationCurrent(HostBackingObject backing, CpuMappedCopyObservation observation) => Ledger.IsCurrent(backing, observation);
        public ulong ReserveHole(ulong address, ulong size) => throw new NotSupportedException();
        public ulong FindUnmappedRange(ulong address, ulong size, ulong searchEnd) => throw new NotSupportedException();
        public bool SplitHole(ulong address, ulong size) => throw new NotSupportedException();
        public bool JoinHoles(ulong address, ulong size) => throw new NotSupportedException();
        public bool FreeHole(ulong address, ulong size) => throw new NotSupportedException();
        public bool CommitPrivate(ulong address, ulong size, HostPageProtection protection) => throw new NotSupportedException();
        public bool ReleasePrivate(ulong address, ulong size) => throw new NotSupportedException();
        public bool ChangeAccess(ulong address, ulong size, HostPageProtection protection) => throw new NotSupportedException();
        public IReadOnlyList<HostPageAccess> QueryAccess(ulong address, ulong size) => throw new NotSupportedException();
        public bool FreeOwnedRange(ulong address, ulong size) => throw new NotSupportedException();
        public void Dispose() { Backing?.Dispose(); if (pin.IsAllocated) pin.Free(); }
    }

    private static object Invoke(object target, string name, params object?[] args)
    {
        try { return target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(target, args)!; }
        catch (TargetInvocationException error) when (error.InnerException is { } primary)
        { ExceptionDispatchInfo.Capture(primary).Throw(); throw; }
    }
    private static long Count(object ledger, string name) => (long)ledger.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(ledger)!;
    private static int Incomplete(object ledger) => (int)ledger.GetType().GetField("coverageIncomplete", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(ledger)!;
    private static object Begin(object ledger, ulong offset, ulong count) => Invoke(ledger, "Begin", offset, count);
    private static bool End(object ledger, object token, bool completed = true) => (bool)Invoke(ledger, "End", token, completed);

    [Fact]
    public void LegacyWritesDoNotEnrollOrCreateBookkeeping()
    {
        using var state = new State();
        Assert.True(state.Views.TryWriteBacking(State.Address, new byte[] { 1, 2, 3 }));
        Assert.True(state.Views.Clear(1, 1)); Assert.True(state.Views.TryCopyBacking(State.Address + 8, State.Address, 3));
        Assert.Null(typeof(HostBackingObject).GetField("_managedWrites", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(state.Backing));
        Assert.Equal(new byte[] { 1, 0, 3 }, state.Bytes().AsSpan(8, 3).ToArray());
    }

    [Fact]
    public void ActualClearWriteAndMemmoveCompleteBalancedPhysicalIntervals()
    {
        using var state = new State(); var ledger = state.Enroll();
        byte[] seed = Enumerable.Range(0, 256).Select(i => (byte)i).ToArray();
        Assert.True(state.Views.TryWriteBacking(State.Address, seed));
        Assert.True(state.Views.TryCopyBacking(State.Address + 7, State.Address, 128)); // actual overlap/memmove
        Assert.True(state.Views.Clear(State.Page + 12, 19));
        var expected = new byte[(int)(State.Page * 2)]; seed.CopyTo(expected, 0); Array.Copy(expected, 0, expected, 7, 128);
        Assert.Equal(expected, state.Bytes());
        Assert.Equal(3, Count(ledger, "begun")); Assert.Equal(3, Count(ledger, "completed"));
        Assert.Equal(0, Count(ledger, "active")); Assert.Equal(0, Count(ledger, "aborted"));
        Assert.Equal(0, Incomplete(ledger));
    }

    [Fact]
    public void MultiSegmentWriteAndCrossSegmentStagingPreserveEveryByte()
    {
        using var state = new State(); var ledger = state.Enroll();
        var seed = Enumerable.Range(0, (int)(State.Page * 2)).Select(i => (byte)(i * 31 + 9)).ToArray();
        Assert.True(state.Views.TryWriteBacking(State.Address, seed)); // two physical destination chunks
        Assert.True(state.Views.TryCopyBacking(State.Address + 8, State.Address, State.Page + 16));
        var expected = seed.ToArray(); Array.Copy(expected, 0, expected, 8, (int)State.Page + 16);
        Assert.Equal(expected, state.Bytes());
        Assert.Equal(4, Count(ledger, "begun")); Assert.Equal(4, Count(ledger, "completed")); Assert.Equal(0, Count(ledger, "active"));
    }

    [Fact]
    public void EmptyAndRejectedBoundsPreservePriorIntervalCounters()
    {
        using var state = new State(); var ledger = state.Enroll();
        Assert.False(state.Views.Clear(0, 0));
        _ = state.Views.TryWriteBacking(State.Address, []);
        Assert.False(state.Views.TryWriteBacking(State.Address + State.Page * 2 - 1, new byte[2]));
        Assert.False(state.Views.Clear(State.Page * 2, 1));
        Assert.Equal(0, Count(ledger, "begun")); Assert.Equal(0, Count(ledger, "active"));
    }

    [Fact]
    public void ActualManagedWriteDoesNotRetimestampOrInvalidateMappingOnlyObservation()
    {
        using var state = new State();
        Assert.Equal(CpuMappedCopyResult.Copied, state.Views.TryReadMappedCopy(State.Address, State.Page, State.Address, new byte[1], out var old));
        Assert.True(state.Views.TryWriteBacking(State.Address, new byte[] { 0xEE }));
        Assert.True(state.Views.IsMappedCopyObservationCurrent(old!)); // explicitly no content authority
    }

    [Fact]
    public void PublishedIntervalsRetainExactPhysicalOffsetAndAbortDistinctFromCompletion()
    {
        using var state = new State(); var ledger = state.Enroll();
        var token = Begin(ledger, State.Page + 17, 29);
        var active = ((ulong Offset, ulong Count, long Identity)[])Invoke(ledger, "ActiveForControl");
        Assert.Single(active); Assert.Equal(State.Page + 17, active[0].Offset); Assert.Equal(29ul, active[0].Count);
        Assert.True(End(ledger, token, false)); // scalar aborted interval, not an injected native/partial-write proof
        Assert.Equal(0, Count(ledger, "active")); Assert.Equal(0, Count(ledger, "completed")); Assert.Equal(1, Count(ledger, "aborted"));
        Assert.Empty(((ulong, ulong, long)[])Invoke(ledger, "ActiveForControl"));
    }

    [Fact]
    public void CopiedAndStaleTokenCannotConsumeLaterReusedSlot()
    {
        using var state = new State(); var ledger = state.Enroll();
        var old = Begin(ledger, 10, 20); Assert.True(End(ledger, old)); Assert.False(End(ledger, old));
        ledger.GetType().GetField("nextIdentity", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(ledger, 128L);
        var current = Begin(ledger, 30, 40); // identity129 reuses identity1's slot
        Assert.False(End(ledger, old)); Assert.Equal(1, Count(ledger, "active"));
        Assert.True(End(ledger, current)); Assert.Equal(0, Count(ledger, "active")); Assert.Equal(2, Count(ledger, "completed"));
    }

    [Fact]
    public void AbortedCompletionCannotBeReplayedAsSuccess()
    {
        using var state = new State(); var ledger = state.Enroll(); var token = Begin(ledger, 10, 30);
        Assert.True(End(ledger, token, false)); Assert.False(End(ledger, token, true));
        Assert.Equal(1, Count(ledger, "aborted")); Assert.Equal(0, Count(ledger, "completed")); Assert.Equal(0, Count(ledger, "active"));
    }

    [Fact]
    public void ForeignLedgerCannotEndAnExactToken()
    {
        using var state = new State(); using var other = new State();
        var ledger = state.Enroll(); var foreign = other.Enroll(); var token = Begin(ledger, 1, 3);
        Assert.False(End(foreign, token)); Assert.Equal(1, Count(ledger, "active")); Assert.Equal(0, Count(foreign, "active"));
        Assert.True(End(ledger, token));
    }

    [Fact]
    public void SaturationBecomesStickyIncompleteAndLegacyWritesStillSucceed()
    {
        using var state = new State(); var ledger = state.Enroll(); var tokens = new List<object>();
        try
        {
            for (var i = 0; i < 128; i++) tokens.Add(Begin(ledger, (ulong)i, 1));
            var declined = Begin(ledger, 500, 1); Assert.True(End(ledger, declined));
            Assert.Equal(1, Incomplete(ledger)); Assert.Equal(128, Count(ledger, "active"));
            Assert.True(state.Views.TryWriteBacking(State.Address + 1000, new byte[] { 0xA5 }));
            Assert.Equal(0xA5, state.Bytes()[1000]); Assert.Equal(128, Count(ledger, "begun"));
        }
        finally { foreach (var token in tokens) Assert.True(End(ledger, token)); }
        Assert.Equal(0, Count(ledger, "active")); Assert.Equal(1, Incomplete(ledger));
    }

    [Fact]
    public void IdentityExhaustionNeverWrapsIntoReusableAuthority()
    {
        using var state = new State(); var ledger = state.Enroll();
        ledger.GetType().GetField("nextIdentity", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(ledger, long.MaxValue);
        var token = Begin(ledger, 0, 1); Assert.True(End(ledger, token));
        Assert.Equal(1, Incomplete(ledger)); Assert.Equal(0, Count(ledger, "active"));
        Assert.True(state.Views.Clear(0, 1)); Assert.Equal(0, Count(ledger, "begun"));
    }

    [Fact]
    public void WarmedActualSingleViewWriteDoesNotAllocate()
    {
        using var state = new State(); var ledger = state.Enroll(); byte[] bytes = [1, 3, 5, 7];
        for (var i = 0; i < 10000; i++) Assert.True(state.Views.TryWriteBacking(State.Address, bytes));
        // A per-write allocation shows in every round; a one-off runtime allocation (tiering,
        // lazy runtime state) on a loaded runner lands in one round at most.
        var allocated = long.MaxValue; var allSucceeded = true;
        for (var round = 0; round < 5 && allocated != 0; round++)
        {
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 10000; i++) allSucceeded &= state.Views.TryWriteBacking(State.Address, bytes);
            allocated = Math.Min(allocated, GC.GetAllocatedBytesForCurrentThread() - before);
        }
        Assert.True(allSucceeded); Assert.Equal(0, allocated); Assert.Equal(0, Count(ledger, "active"));
    }

    [Fact]
    public void WarmedDormantSingleViewFastPathWriteRegistersNoIntervalAndDoesNotAllocate()
    {
        using var state = new State(); var ledger = state.EnrollDormant(); byte[] bytes = [2, 4, 6, 8];
        var dormant = (bool)ledger.GetType().GetField("DormantEnabled", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        if (!dormant) return; // SHARPEMU_WRITE_INTERVALS_DORMANT=0: the observed test covers it
        Assert.False((bool)ledger.GetType().GetProperty("IsObserved", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(ledger)!);
        Assert.True(state.Views.TryWriteSingleView(State.Address + 16, bytes));
        Assert.Equal(bytes, state.Bytes().AsSpan(16, bytes.Length).ToArray());
        Assert.False(state.Views.TryWriteSingleView(State.Address + State.Page - 2, bytes));
        for (var i = 0; i < 10000; i++) Assert.True(state.Views.TryWriteSingleView(State.Address, bytes));
        var before = GC.GetAllocatedBytesForCurrentThread(); var allSucceeded = true;
        for (var i = 0; i < 10000; i++) allSucceeded &= state.Views.TryWriteSingleView(State.Address, bytes);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allSucceeded); Assert.Equal(0, allocated);
        // No admission, identity, slot, history or counter: the ledger is exactly as enrolled.
        Assert.Equal(0, Count(ledger, "nextIdentity")); Assert.Equal(0, Count(ledger, "admission"));
        Assert.Equal(0, Count(ledger, "begun")); Assert.Equal(0, Count(ledger, "completed"));
        Assert.Equal(0, Count(ledger, "active")); Assert.Equal(0, Count(ledger, "aborted"));
        Assert.Equal(0, Incomplete(ledger));
    }

    [Fact]
    public void WarmedObservedSingleViewFastPathWriteCompletesOneIntervalAndDoesNotAllocate()
    {
        using var state = new State(); var ledger = state.Enroll(); byte[] bytes = [2, 4, 6, 8];
        // An enrolled backing no longer declines the lock-free path; each write is one
        // admitted operation with exactly one completed byte interval.
        var begun = Count(ledger, "begun");
        Assert.True(state.Views.TryWriteSingleView(State.Address + 16, bytes));
        Assert.Equal(begun + 1, Count(ledger, "begun")); Assert.Equal(0, Count(ledger, "aborted"));
        Assert.Equal(bytes, state.Bytes().AsSpan(16, bytes.Length).ToArray());
        // A view-crossing write is refused with no byte interval (the PVM worker
        // scenario single-view-miss-keeps-read-watch guards the whole-backing invalidation).
        Assert.False(state.Views.TryWriteSingleView(State.Address + State.Page - 2, bytes));
        Assert.Equal(begun + 1, Count(ledger, "begun")); Assert.Equal(0, Count(ledger, "aborted"));
        for (var i = 0; i < 10000; i++) Assert.True(state.Views.TryWriteSingleView(State.Address, bytes));
        var before = GC.GetAllocatedBytesForCurrentThread(); var allSucceeded = true;
        for (var i = 0; i < 10000; i++) allSucceeded &= state.Views.TryWriteSingleView(State.Address, bytes);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allSucceeded); Assert.Equal(0, allocated);
        Assert.Equal(0, Count(ledger, "active")); Assert.Equal(0, Count(ledger, "aborted"));
    }
}
