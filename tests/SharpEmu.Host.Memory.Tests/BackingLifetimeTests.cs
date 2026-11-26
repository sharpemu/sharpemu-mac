// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Runtime.CompilerServices;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Host.Memory.Tests;

// Ordinary managed/pinned-array controls only. No host composition, native
// permission calls, signal callbacks or guest code is installed.
public sealed class BackingLifetimeTests
{
    public BackingLifetimeTests(ITestOutputHelper output)
    {
        output.WriteLine($"ordinary-backing architecture={RuntimeInformation.ProcessArchitecture}; framework={RuntimeInformation.FrameworkDescription}; native_page={Environment.SystemPageSize}; managed-pinned-array-only");
        var expected = Environment.GetEnvironmentVariable("SHARPEMU_TEST_EXPECT_ARCH");
        if (!string.IsNullOrEmpty(expected)) Assert.Equal(expected, RuntimeInformation.ProcessArchitecture.ToString());
    }

    private static HostBackingObject Phantom(Func<HostBackingObject, HostBackingReleaseResult> release, ulong size = 4096)
        => new(0x10000, size, 4096, release);

    [Fact]
    public void IdentityAndLogicalSizeDoNotPretendNativeExtentIsPermission()
    {
        using var first = Phantom(_ => new(HostBackingReleaseStatus.Released), 1);
        using var second = Phantom(_ => new(HostBackingReleaseStatus.Released));
        Assert.NotEqual(first.Identity, second.Identity);
        Assert.Equal(1ul, first.Size); Assert.Equal(4096ul, first.NativeAliasSize);
        Assert.Equal(4096ul, first.NativePageSize);
    }

    [Fact]
    public void CloseRevokesNewAdmissionsAndTimeoutRetainsStorageUntilOldAccessExits()
    {
        var calls = 0;
        var backing = Phantom(owner => { Assert.False(Monitor.IsEntered(owner.Gate)); calls++; return new(HostBackingReleaseStatus.Released); });
        Assert.True(backing.TryEnterAccess());
        Assert.False(backing.TryDispose(TimeSpan.Zero, out var timeout));
        Assert.Equal(HostBackingReleaseStatus.TimedOut, timeout.Status);
        Assert.True(backing.IsClosing); Assert.False(backing.IsDisposed); Assert.Equal(0, calls);
        Assert.False(backing.TryEnterAccess());
        backing.ExitAccess();
        Assert.True(backing.TryDispose(TimeSpan.Zero, out _)); backing.Dispose();
        Assert.Equal(1, calls); Assert.True(backing.IsDisposed);
    }

    [Fact]
    public async Task RejectedPostCloseCountCannotVetoCleanupAfterDrainObservedZero()
    {
        using var drained = new ManualResetEventSlim(); using var proceed = new ManualResetEventSlim();
        var calls = 0;
        var backing = Phantom(owner =>
        {
            Assert.True(owner.IsClosing); Assert.Equal(1, owner.ActiveAccesses);
            calls++; return new(HostBackingReleaseStatus.Released);
        });
        // This second entrant observes open before the closer but is preempted
        // before its increment. It has never gained storage access authority.
        Assert.True(Phase<bool>(backing, "CanStartAccessEntry"));
        Assert.True(backing.TryEnterAccess());
        var releasing = Task.Run(() =>
        {
            object?[] args = [TimeSpan.FromSeconds(10), null, (Action)(() => { drained.Set(); Assert.True(proceed.Wait(10000)); })];
            return (bool)typeof(HostBackingObject).GetMethod("TryDisposeCore", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(backing, args)!;
        });
        Assert.True(SpinWait.SpinUntil(() => backing.IsClosing, 10000));
        backing.ExitAccess(); // prior admitted access finishes before drain0
        Assert.True(drained.Wait(10000));
        var rejectedState = Phase<long>(backing, "IncrementAccessCount");
        Assert.Equal(1, backing.ActiveAccesses);
        bool rejectedCompleted = false;
        try
        {
            proceed.Set(); Assert.True(await releasing.WaitAsync(TimeSpan.FromSeconds(10)));
            Assert.Equal(1, calls); Assert.True(backing.IsDisposed);
            Assert.False(Phase<bool>(backing, "CompleteAccessEntry", rejectedState)); rejectedCompleted = true;
            Assert.Equal(0, backing.ActiveAccesses); Assert.False(backing.TryEnterAccess());
        }
        finally
        {
            proceed.Set();
            if (!rejectedCompleted) Phase<bool>(backing, "CompleteAccessEntry", rejectedState);
        }
    }

    [Fact]
    public void IncrementAdmittedBeforeCloseStillRetainsStorageUntilItsFinalExit()
    {
        var calls = 0; var backing = Phantom(_ => { calls++; return new(HostBackingReleaseStatus.Released); });
        Assert.True(Phase<bool>(backing, "CanStartAccessEntry"));
        var admittedState = Phase<long>(backing, "IncrementAccessCount");
        backing.CloseAccess();
        Assert.False(backing.TryDispose(TimeSpan.Zero, out var result));
        Assert.Equal(HostBackingReleaseStatus.TimedOut, result.Status); Assert.Equal(0, calls);
        // Returned increment state was open: completion after Close remains
        // admitted, with the original counted lifetime obligation intact.
        Assert.True(Phase<bool>(backing, "CompleteAccessEntry", admittedState));
        backing.ExitAccess(); Assert.True(backing.TryDispose(TimeSpan.Zero, out _)); Assert.Equal(1, calls);
    }

    [Fact]
    public void PreNativePhaseFailureCanRetryWithoutFabricatingCompletedRelease()
    {
        var calls = 0; var backing = Phantom(_ => { calls++; return new(HostBackingReleaseStatus.Released); });
        Assert.Equal(HostBackingReleaseStatus.NotAttempted, backing.LastRelease.Status);
        Assert.False(backing.LastRelease.Completed);
        var original = new IOException("ordinary phase control failure");
        object?[] args = [TimeSpan.Zero, null, (Action)(() => throw original)];
        var error = Assert.Throws<TargetInvocationException>(() =>
            typeof(HostBackingObject).GetMethod("TryDisposeCore", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(backing, args));
        Assert.Same(original, error.InnerException); Assert.Equal(0, calls); Assert.False(backing.IsDisposed);
        Assert.Equal(HostBackingReleaseStatus.NotAttempted, backing.LastRelease.Status);
        Assert.True(backing.TryDispose(TimeSpan.Zero, out _)); Assert.Equal(1, calls);
    }

    private static T Phase<T>(HostBackingObject owner, string name, params object[] args) =>
        (T)typeof(HostBackingObject).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(owner, args)!;

    [Theory]
    [InlineData(HostBackingReleaseStatus.AliasOutcomeUnknown, false)]
    [InlineData(HostBackingReleaseStatus.StorageOutcomeUnknown, true)]
    public void UnknownNativeOutcomeNeverReclaimsAliasOrClosesReusedDescriptorAgain(HostBackingReleaseStatus status, bool aliasReleased)
    {
        var calls = 0;
        var backing = Phantom(_ => { calls++; return new(status, 13); });
        Assert.False(backing.TryDispose(TimeSpan.Zero, out var result));
        Assert.Equal(13, result.NativeError); Assert.Equal(aliasReleased, result.AliasReleased);
        Assert.False(backing.TryDispose(TimeSpan.Zero, out var repeated));
        Assert.Equal(result, repeated); Assert.Equal(1, calls); Assert.False(backing.IsDisposed);
        Assert.False(backing.TryEnterAccess());
    }

    [Fact]
    public void CallbackThrowPreservesPrimaryAndQuarantinesUnknownEffects()
    {
        var original = new IOException("provider cleanup unknown"); var calls = 0;
        var backing = Phantom(_ => { calls++; throw original; });
        Assert.Same(original, Assert.Throws<IOException>(() => backing.TryDispose(TimeSpan.Zero, out _)));
        Assert.False(backing.TryDispose(TimeSpan.Zero, out var result));
        Assert.Equal(HostBackingReleaseStatus.CallbackOutcomeUnknown, result.Status); Assert.Equal(1, calls);
    }

    [Fact]
    public async Task ConcurrentReleaseIsOneUseAndDoesNotHoldBackingGate()
    {
        using var entered = new ManualResetEventSlim(); using var finish = new ManualResetEventSlim();
        var calls = 0;
        var backing = Phantom(_ => { Interlocked.Increment(ref calls); entered.Set(); Assert.True(finish.Wait(10000)); return new(HostBackingReleaseStatus.Released); });
        var worker = Task.Run(() => backing.TryDispose(TimeSpan.FromSeconds(10), out _));
        try
        {
            Assert.True(entered.Wait(10000));
            lock (backing.Gate) Assert.Equal(0, backing.ActiveAccesses);
            Assert.False(backing.TryDispose(TimeSpan.Zero, out var busy)); Assert.Equal(HostBackingReleaseStatus.Busy, busy.Status);
        }
        finally { finish.Set(); }
        Assert.True(await worker.WaitAsync(TimeSpan.FromSeconds(10))); Assert.Equal(1, calls);
    }

    [Fact]
    public void DisposeUnderBackingGateRejectsBeforeRevocation()
    {
        using var backing = Phantom(_ => new(HostBackingReleaseStatus.Released));
        lock (backing.Gate) Assert.Throws<InvalidOperationException>(() => backing.Dispose());
        Assert.False(backing.IsClosing); Assert.True(backing.TryEnterAccess()); backing.ExitAccess();
    }

    [Fact]
    public void AliasAccessPreventsDirectExplicitBackingDisposalAndStaleAdmission()
    {
        var host = new ManagedViews(); using var store = new SharedBackingViews(host, 8192);
        Assert.True(store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(store.TryEnterAliasAccess());
        try
        {
            Assert.False(host.Backing!.TryDispose(TimeSpan.Zero, out _));
            Assert.Equal(0, host.Releases); Assert.False(store.TryEnterAliasAccess());
        }
        finally { store.ExitAliasAccess(); }
        store.Dispose(); Assert.Equal(1, host.Releases); Assert.Equal(0, host.Backing!.ActiveAccesses);
    }

    [Fact]
    public void FastAndSegmentedCopiesClearAndOverlapReleaseEveryOrdinaryAccess()
    {
        var host = new ManagedViews(); using var store = new SharedBackingViews(host, 8192);
        Assert.True(store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(store.TryMapReservedRange(0x101000, 4096, 4096, HostPageProtection.ReadWrite, out _));
        var source = Enumerable.Range(0, 8192).Select(i => (byte)(i * 17)).ToArray();
        Assert.True(store.TryWriteBacking(0x100000, source));
        var read = new byte[8192]; Assert.True(store.TryReadBacking(0x100000, read)); Assert.Equal(source, read);
        Assert.True(store.TryCopyBacking(0x100800, 0x100000, 6000));
        Assert.True(store.TryReadBacking(0x100800, read.AsSpan(0, 6000))); Assert.Equal(source.AsSpan(0, 6000).ToArray(), read.AsSpan(0, 6000).ToArray());
        Assert.True(store.Clear(0, 8192)); Assert.True(store.TryReadSingleView(0x100000, read.AsSpan(0, 4096)));
        Assert.All(read[..4096], b => Assert.Equal((byte)0, b)); Assert.Equal(0, host.Backing!.ActiveAccesses);
        Span<byte> small = stackalloc byte[8]; store.TryReadSingleView(0x100000, small);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 10000; i++) if (!store.TryReadSingleView(0x100000, small)) throw new Exception("unexpected decline");
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void FailedViewCleanupRetainsUnattemptedOwnershipAndNeverRepeatsUncertainNativeScope()
    {
        var host = new ManagedViews { FailUnmapCall = 2 }; var store = new SharedBackingViews(host, 8192);
        Assert.True(store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(store.TryMapReservedRange(0x101000, 4096, 4096, HostPageProtection.ReadWrite, out _));
        Assert.Throws<InvalidOperationException>(() => store.Dispose());
        Assert.Equal(2, host.Unmaps.Count); Assert.Equal(0, host.Releases);
        Assert.Throws<InvalidOperationException>(() => store.Dispose());
        Assert.Equal(2, host.Unmaps.Count); Assert.False(store.TryEnterAliasAccess());
        Assert.False(host.Backing!.IsDisposed); // no alias release after uncertain view cleanup
    }

    [Fact]
    public void ReentrantDisposeAfterMapLeavesNoUnrecordedSuccessfulView()
    {
        var host = new ManagedViews(); var store = new SharedBackingViews(host, 8192);
        host.AfterMap = store.Dispose;
        Assert.False(store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out var failure));
        Assert.Equal(HostViewFailure.BackingUnavailable, failure); Assert.Single(host.Unmaps);
        store.Dispose(); Assert.Equal(1, host.Releases);
    }

    [Fact]
    public async Task DisposeRevokesThenWaitsForLateMapRollbackBeforeAliasRelease()
    {
        var host = new ManagedViews(); var store = new SharedBackingViews(host, 8192);
        using var entered = new ManualResetEventSlim(); using var finish = new ManualResetEventSlim();
        host.AfterMap = () => { entered.Set(); Assert.True(finish.Wait(10000)); };
        var mapping = Task.Run(() => store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(entered.Wait(10000));
        var disposing = Task.Run(store.Dispose);
        try
        {
            Assert.True(SpinWait.SpinUntil(() => !store.IsAvailable, 10000));
            Assert.Equal(0, host.Releases); Assert.False(disposing.IsCompleted);
        }
        finally { finish.Set(); }
        Assert.False(await mapping.WaitAsync(TimeSpan.FromSeconds(10)));
        await disposing.WaitAsync(TimeSpan.FromSeconds(10)); Assert.Single(host.Unmaps); Assert.Equal(1, host.Releases);
    }

    [Fact]
    public async Task ConcurrentUnmapCompletesBeforeDisposalCaptureAndIsNotReleasedTwice()
    {
        var host = new ManagedViews(); var store = new SharedBackingViews(host, 8192);
        Assert.True(store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out _));
        using var entered = new ManualResetEventSlim(); using var finish = new ManualResetEventSlim();
        host.AfterUnmap = () => { entered.Set(); Assert.True(finish.Wait(10000)); };
        var unmapping = Task.Run(() => store.Unmap(0x100000, 4096, out _));
        Assert.True(entered.Wait(10000)); var disposing = Task.Run(store.Dispose);
        try { Assert.True(SpinWait.SpinUntil(() => !store.IsAvailable, 10000)); Assert.Equal(0, host.Releases); }
        finally { finish.Set(); }
        Assert.True(await unmapping.WaitAsync(TimeSpan.FromSeconds(10)));
        await disposing.WaitAsync(TimeSpan.FromSeconds(10)); Assert.Single(host.Unmaps); Assert.Equal(1, host.Releases);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConcurrentDisposalAndReentrantCloseCannotInvertGatesOrLosePrimary(bool failProvider)
    {
        var host = new ManagedViews(); var store = new SharedBackingViews(host, 8192);
        using var entered = new ManualResetEventSlim(); using var callback = new ManualResetEventSlim();
        var original = new IOException("provider outcome unknown after reentrant close");
        host.AfterMap = () =>
        {
            entered.Set(); Assert.True(callback.Wait(10000));
            store.Dispose(); Assert.False(store.CleanupComplete); Assert.False(host.Backing!.IsDisposed);
            if (failProvider) throw original;
        };
        var mapping = Task.Run(() => { try { Assert.False(store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out _)); return (Exception?)null; } catch (Exception e) { return e; } });
        Assert.True(entered.Wait(10000));
        var disposing = Task.Run(() => { try { store.Dispose(); return (Exception?)null; } catch (Exception e) { return e; } });
        try { Assert.True(SpinWait.SpinUntil(() => !store.IsAvailable, 10000)); Assert.Equal(0, host.Releases); }
        finally { callback.Set(); }
        var mapError = await mapping.WaitAsync(TimeSpan.FromSeconds(10));
        var disposeError = await disposing.WaitAsync(TimeSpan.FromSeconds(10));
        if (failProvider)
        {
            Assert.Same(original, mapError); Assert.IsType<InvalidOperationException>(disposeError);
            Assert.True(original.Data.Contains("BackingDeferredCleanupFailure"));
            Assert.False(store.CleanupComplete); Assert.Equal(0, host.Releases); Assert.Empty(host.Unmaps);
        }
        else { Assert.Null(mapError); Assert.Null(disposeError); Assert.True(store.CleanupComplete); Assert.Single(host.Unmaps); Assert.Equal(1, host.Releases); }
    }

    [Fact]
    public void CallbackCloseRejectsFreshPublicSameThreadMapAndUnmap()
    {
        var host = new ManagedViews(); var store = new SharedBackingViews(host, 8192);
        host.AfterMap = () =>
        {
            store.Dispose(); Assert.False(store.CleanupComplete);
            Assert.False(store.TryMapReservedRange(0x110000, 4096, 4096, HostPageProtection.ReadWrite, out _));
            Assert.False(store.Unmap(0x100000, 4096, out _)); Assert.Equal(1, host.Maps);
        };
        Assert.False(store.TryMapReservedRange(0x100000, 4096, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(store.CleanupComplete); Assert.Single(host.Unmaps); Assert.Equal(1, host.Releases);
    }

    [Fact]
    public async Task GuestDisposeDropsMappingAndProtectionGatesBeforeAliasDrain()
    {
        var host = new ManagedViews(); var owner = new GuestSpaceOwner(host, 8192);
        Assert.True(owner.TryReserveAddressRange(0x100000, 0x4000));
        Assert.True(owner.TryEnterAliasAccess());
        var disposing = Task.Run(owner.Dispose);
        try
        {
            Assert.True(SpinWait.SpinUntil(() => !owner.TryClearBacking(0, 1), 10000));
            var protection = Task.Run(() => owner.SetTransientAccess(0x100000, 4096, HostPageProtection.ReadWrite));
            Assert.False(await protection.WaitAsync(TimeSpan.FromSeconds(1)));
            Assert.False(disposing.IsCompleted); Assert.Equal(0, host.Releases);
        }
        finally { owner.ExitAliasAccess(); }
        await disposing.WaitAsync(TimeSpan.FromSeconds(10)); owner.Dispose(); Assert.Equal(1, host.Releases);
    }

    [Fact]
    public void GuestRangeUnknownCleanupIsStickyAndDoesNotRepeatReleasedScopes()
    {
        var host = new ManagedViews { FailFreeCall = 2 }; var owner = new GuestSpaceOwner(host, 8192);
        Assert.True(owner.TryReserveAddressRange(0x100000, 0x4000));
        Assert.True(owner.TryReserveAddressRange(0x110000, 0x4000));
        Assert.Throws<InvalidOperationException>(owner.Dispose); Assert.Equal(2, host.Frees.Count);
        Assert.Throws<InvalidOperationException>(owner.Dispose); Assert.Equal(2, host.Frees.Count);
        Assert.False(owner.TryReserveAddressRange(0x120000, 0x4000)); Assert.Equal(1, host.Releases);
    }

#if SHARPEMU_HOST_MACOS
    [Fact]
    public void ActualMacMapEntryRejectsClosedBackingBeforeAnyNativeCall()
    {
        if (!OperatingSystem.IsMacOS()) throw new PlatformNotSupportedException("Mac closed-entry control requires macOS.");
        var calls = 0; var backing = Phantom(_ => { calls++; return new(HostBackingReleaseStatus.Released); });
        backing.CloseAccess();
        var provider = SharpEmu.Host.MacOS.MacOSHost.CreateViewMemory();
        Assert.False(provider.TryMapView(backing, 0x100000, 0, 4096, HostPageProtection.ReadWrite, out var failure));
        Assert.Equal(HostViewFailure.BackingUnavailable, failure); Assert.Equal(0, calls);
        Assert.Equal(0, backing.ActiveAccesses); backing.Dispose(); Assert.Equal(1, calls);
    }
#endif

    [Fact]
    public void LegacyCallbackRetainsExistingOneShotSemantics()
    {
        var calls = 0; var backing = new HostBackingObject(0x10000, 4096, owner => { Assert.True(Monitor.IsEntered(owner.Gate)); calls++; });
        Assert.False(backing.HasOwnedLifetime); backing.Dispose(); backing.Dispose(); Assert.Equal(1, calls);
    }

    private sealed unsafe class ManagedViews : IHostViewMemory
    {
        private readonly byte[] _storage = GC.AllocateArray<byte>(12288, pinned: true);
        public HostBackingObject? Backing; public int Releases, Maps;
        public int FailUnmapCall, FailFreeCall; public Action? AfterMap, AfterUnmap;
        public List<HostAddressRange> Unmaps { get; } = []; public List<HostAddressRange> Frees { get; } = [];
        public ulong PageSize => 4096; public ulong Granularity => 4096;
        public bool TryCreateBacking(ulong size, out HostBackingObject? backing, out HostViewFailure failure)
        {
            var start = (ulong)Unsafe.AsPointer(ref _storage[0]); var aligned = (start + 4095) & ~4095ul;
            backing = Backing = new(aligned, size, 4096, _ => { Releases++; GC.KeepAlive(_storage); return new(HostBackingReleaseStatus.Released); });
            failure = HostViewFailure.None; return true;
        }
        public ulong ReserveHole(ulong address, ulong size) => address;
        public bool SplitHole(ulong address, ulong size) => true;
        public bool JoinHoles(ulong address, ulong size) => true;
        public bool FreeHole(ulong address, ulong size) => FreeOwnedRange(address, size);
        public bool TryMapView(HostBackingObject backing, ulong address, ulong offset, ulong size, HostPageProtection protection, out HostViewFailure failure)
        {
            Maps++;
            failure = HostViewFailure.BackingUnavailable;
            if (!backing.TryEnterAccess()) return false;
            try { failure = HostViewFailure.None; }
            finally { backing.ExitAccess(); }
            AfterMap?.Invoke(); return true;
        }
        public bool UnmapView(ulong address, ulong size) { Unmaps.Add(new(address, size)); AfterUnmap?.Invoke(); return Unmaps.Count != FailUnmapCall; }
        public bool CommitPrivate(ulong address, ulong size, HostPageProtection protection) => true;
        public bool ReleasePrivate(ulong address, ulong size) => true;
        public bool ChangeAccess(ulong address, ulong size, HostPageProtection protection) => true;
        public IReadOnlyList<HostPageAccess> QueryAccess(ulong address, ulong size) => []; // access is never tracked here
        public bool FreeOwnedRange(ulong address, ulong size) { Frees.Add(new(address, size)); return Frees.Count != FailFreeCall; }
    }
}
