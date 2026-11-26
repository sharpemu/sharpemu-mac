// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#if SHARPEMU_HOST_MACOS
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Host.Memory.Tests;

// Separately authorized, fresh ordinary processes only. No signals, guest code,
// native permission probes, recovery owners, or injected cleanup outcomes.
[SupportedOSPlatform("macos")]
internal static class MacBackingLifecycleWorker
{
    internal static int Run(string scenario)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_BACKING_NATIVE") != "1") return 80;
        if (!OperatingSystem.IsMacOS()) return 81;
        try
        {
            var expected = Environment.GetEnvironmentVariable("SHARPEMU_TEST_EXPECT_ARCH");
            Assert.False(string.IsNullOrEmpty(expected)); Assert.Equal(expected, RuntimeInformation.ProcessArchitecture.ToString());
            var host = new ObservedProvider();
            Assert.Equal((ulong)Environment.SystemPageSize, host.PageSize);
            Assert.Equal(RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? 16384ul : 4096ul, host.PageSize);
            switch (scenario)
            {
                case "roundtrip": Roundtrip(host); break;
                case "held-access": HeldAccess(host); break;
                case "copied-view-close": CopiedViewClose(host); break;
                case "pending-map-close": PendingMapClose(host); break;
                default: return 82;
            }
            Console.WriteLine($"backing-native-pass case={scenario} architecture={RuntimeInformation.ProcessArchitecture} framework={RuntimeInformation.FrameworkDescription} provider_posix_page={host.PageSize}; not-universal-kernel-granularity; ordinary-alias-descriptor-only");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 83; }
    }

    private static unsafe void Roundtrip(ObservedProvider host)
    {
        var backing = host.Create(host.PageSize + 7);
        using var first = new Hole(host); using var second = new Hole(host);
        try
        {
            Assert.Equal(host.PageSize + 7, backing.Size); Assert.Equal(2 * host.PageSize, backing.NativeAliasSize);
            Assert.Equal(host.PageSize, backing.NativePageSize); Assert.True(backing.Identity > 0);
            Assert.True(host.TryMapView(backing, first.Address, 0, host.PageSize, HostPageProtection.ReadWrite, out _));
            Assert.True(host.TryMapView(backing, second.Address, 0, host.PageSize, HostPageProtection.ReadWrite, out _));
            Assert.True(backing.TryEnterAccess());
            try
            {
                ((byte*)backing.AliasBase)[0] = 0x51; ((byte*)backing.AliasBase)[host.PageSize - 1] = 0xA7;
                ((byte*)backing.AliasBase)[backing.Size - 1] = 0xC9; // within ftruncate logical EOF, never padded bytes
                Assert.Equal(0x51, ((byte*)first.Address)[0]); Assert.Equal(0xA7, ((byte*)second.Address)[host.PageSize - 1]);
                ((byte*)second.Address)[17] = 0x92; Assert.Equal(0x92, ((byte*)first.Address)[17]); Assert.Equal(0x92, ((byte*)backing.AliasBase)[17]);
                Assert.Equal(0xC9, ((byte*)backing.AliasBase)[backing.Size - 1]);
            }
            finally { backing.ExitAccess(); }
            Assert.True(host.UnmapView(first.Address, host.PageSize)); Assert.True(host.UnmapView(second.Address, host.PageSize));
            Released(backing); // real Mac munmap + close, after both views restored
            Assert.Equal(2, host.MapCalls); Assert.Equal(2, host.UnmapCalls);
        }
        finally { CleanupBacking(backing); }
    }

    private static unsafe void HeldAccess(ObservedProvider host)
    {
        var backing = host.Create(host.PageSize);
        using var hole = new Hole(host);
        var held = false;
        try
        {
            Assert.True(backing.TryEnterAccess()); held = true;
            ((byte*)backing.AliasBase)[0] = 0x37; ((byte*)backing.AliasBase)[backing.Size - 1] = 0xD2;
            Assert.False(backing.TryDispose(TimeSpan.FromMilliseconds(20), out var timeout));
            Assert.Equal(HostBackingReleaseStatus.TimedOut, timeout.Status); Assert.False(backing.IsDisposed); Assert.Equal(1, backing.ActiveAccesses);
            Assert.False(backing.TryEnterAccess());
            Assert.False(host.TryMapView(backing, hole.Address, 0, host.PageSize, HostPageProtection.ReadWrite, out var failure));
            Assert.Equal(HostViewFailure.BackingUnavailable, failure);
            // This genuinely admitted access still retains the CPU alias. No new
            // access is inferred from the timeout or from the raw address.
            Assert.Equal(0x37, ((byte*)backing.AliasBase)[0]); Assert.Equal(0xD2, ((byte*)backing.AliasBase)[backing.Size - 1]);
            backing.ExitAccess(); held = false; Released(backing);
            Assert.False(backing.TryEnterAccess()); Released(backing); // completed one-use cleanup is idempotent
        }
        finally { if (held) backing.ExitAccess(); CleanupBacking(backing); }
    }

    private static unsafe void CopiedViewClose(ObservedProvider host)
    {
        var views = new SharedBackingViews(host, 2 * host.PageSize);
        var backing = host.Backing!;
        using var first = new Hole(host); using var second = new Hole(host);
        var held = false; Task? closing = null; Exception? primary = null;
        try
        {
            Assert.True(views.TryMapReservedRange(first.Address, host.PageSize, 0, HostPageProtection.ReadWrite, out _));
            Assert.True(views.TryMapReservedRange(second.Address, host.PageSize, host.PageSize, HostPageProtection.ReadWrite, out _));
            var seed = Enumerable.Range(0, 256).Select(i => (byte)(i ^ 0x6D)).ToArray(); var copy = new byte[seed.Length];
            Assert.True(views.TryWriteBacking(first.Address, seed));
            Assert.True(views.TryCopyBacking(second.Address, first.Address, (ulong)seed.Length));
            Assert.True(views.TryReadSingleView(second.Address, copy)); Assert.Equal(seed, copy);
            Assert.True(views.TryEnterAliasAccess()); held = true;
            Assert.True(views.TryResolveAlias(second.Address, (ulong)seed.Length, out var alias));
            Assert.Equal(seed, new ReadOnlySpan<byte>((void*)second.Address, seed.Length).ToArray());
            closing = Task.Run(views.Dispose);
            Assert.True(SpinWait.SpinUntil(() => !views.IsAvailable, 10000));
            Assert.False(views.CleanupComplete); Assert.False(backing.IsDisposed); Assert.Equal(HostBackingReleaseStatus.NotAttempted, backing.LastRelease.Status);
            Assert.False(views.TryEnterAliasAccess()); Assert.False(views.TryWriteBacking(first.Address, seed));
            // Resolve before close, then use only under the already admitted
            // ordinary access. Guest virtual view rights are not certified here.
            Assert.Equal(seed, new ReadOnlySpan<byte>((void*)alias, seed.Length).ToArray());
            Assert.False(closing.IsCompleted);
            views.ExitAliasAccess(); held = false; closing.GetAwaiter().GetResult();
            Assert.True(views.CleanupComplete); AssertReleased(backing); Assert.Equal(2, host.UnmapCalls);
        }
        catch (Exception error) { primary = error; throw; }
        finally
        {
            if (held) views.ExitAliasAccess();
            FinishOwnedViews(host, views, primary, closing);
        }
    }

    private static void PendingMapClose(ObservedProvider host)
    {
        var views = new SharedBackingViews(host, host.PageSize); var backing = host.Backing!;
        using var hole = new Hole(host);
        using var requested = new ManualResetEventSlim(); using var proceed = new ManualResetEventSlim(); using var peerStarted = new ManualResetEventSlim();
        Task? mapping = null; Task? peer = null; Exception? primary = null;
        host.AfterSuccessfulMap = () =>
        {
            // Actual MAP_FIXED has returned and its provider gates are gone.
            // Reentrant Dispose requests close only, while this pending operation
            // still owns its exact late-map rollback obligation.
            views.Dispose(); Assert.False(views.CleanupComplete); Assert.False(backing.IsDisposed);
            requested.Set(); Assert.True(proceed.Wait(10000));
        };
        try
        {
            mapping = Task.Run(() =>
            {
                Assert.False(views.TryMapReservedRange(hole.Address, host.PageSize, 0, HostPageProtection.ReadWrite, out var failure));
                Assert.Equal(HostViewFailure.BackingUnavailable, failure); Assert.True(views.CleanupComplete);
            });
            Assert.True(requested.Wait(10000));
            peer = Task.Run(() => { peerStarted.Set(); views.Dispose(); Assert.True(views.CleanupComplete); });
            Assert.True(peerStarted.Wait(1000)); Assert.False(peer.IsCompleted);
            Assert.False(views.IsAvailable); Assert.False(views.CleanupComplete); Assert.Equal(HostBackingReleaseStatus.NotAttempted, backing.LastRelease.Status);
            proceed.Set(); mapping.GetAwaiter().GetResult(); peer.GetAwaiter().GetResult();
            Assert.Equal(1, host.MapCalls); Assert.Equal(1, host.UnmapCalls); AssertReleased(backing);
        }
        catch (Exception error) { primary = error; throw; }
        finally
        {
            proceed.Set();
            FinishOwnedViews(host, views, primary, mapping, peer);
        }
    }

    private static void FinishOwnedViews(ObservedProvider host, SharedBackingViews views, Exception? primary, params Task?[] tasks)
    {
        Exception? cleanup = null;
        foreach (var task in tasks)
        {
            if (task is null) continue;
            try { task.WaitAsync(TimeSpan.FromSeconds(10)).GetAwaiter().GetResult(); }
            catch (Exception error) when (error is not OutOfMemoryException and not AccessViolationException)
            {
                cleanup ??= error;
                if (!task.IsCompleted) host.QuarantineHoleCleanup = true;
            }
        }
        // Pending-provider or uncertain cleanup paths retain their sticky exact
        // obligations; Dispose must not guess/repeat an unknown syscall. Attempt
        // all ordinary joins/cleanup even if an earlier owned task failed.
        try { views.Dispose(); }
        catch (Exception error) when (error is not OutOfMemoryException and not AccessViolationException)
        { cleanup ??= error; }
        if (cleanup is null) return;
        if (primary is not null) primary.Data["BackingWorkerCleanupFailure"] = cleanup;
        else System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(cleanup).Throw();
    }

    private static void Released(HostBackingObject backing)
    {
        Assert.True(backing.TryDispose(TimeSpan.FromSeconds(10), out var result));
        Assert.Equal(HostBackingReleaseStatus.Released, result.Status); Assert.Equal(0, result.NativeError); AssertReleased(backing);
        Console.WriteLine($"native-backing-release identity={backing.Identity} logical_bytes={backing.Size} alias_extent={backing.NativeAliasSize} status={result.Status} native_error={result.NativeError}; actual-provider-cleanup-not-permission-receipt");
    }
    private static void AssertReleased(HostBackingObject backing)
    {
        Assert.True(backing.IsClosing); Assert.True(backing.IsDisposed); Assert.Equal(0, backing.ActiveAccesses);
        Assert.Equal(HostBackingReleaseStatus.Released, backing.LastRelease.Status); Assert.Equal(0, backing.LastRelease.NativeError);
        // No pointer dereference, fd inspection, or persistent-vacancy assertion
        // follows successful release; the kernel may immediately reuse either.
    }
    private static void CleanupBacking(HostBackingObject backing)
    {
        if (!backing.IsDisposed && backing.LastRelease.Status is HostBackingReleaseStatus.NotAttempted or HostBackingReleaseStatus.TimedOut or HostBackingReleaseStatus.Busy)
            backing.Dispose(); // only no-native-effect state may attempt cleanup
    }

    // Each successful kernel-selected Mach allocation is exclusive fixture ownership.
    // MAP_FIXED/placeholder restoration stay within it; the full32KiB receipt is released once.
    // Unknown map/unmap outcomes stop cleanup rather than reclaim reused holes.
    private sealed class Hole : IDisposable
    {
        private readonly ObservedProvider _host;
        internal readonly ulong Address;
        private bool _released;
        private bool _quarantined; // exact immutable allocation scope, no repeat after thrown free
        private readonly ulong _allocationAddress;
        private const ulong AllocationExtent = 32768; // full Mach-owned receipt, on both CLRs
        internal Hole(ObservedProvider host)
        {
            _host = host;
            ulong allocated = 0;
            var nativeReturn = MachVmAllocate(MachTaskSelf(), ref allocated, AllocationExtent, 1); // VM_FLAGS_ANYWHERE
            if (nativeReturn != 0) throw new InvalidOperationException($"Fixture Mach allocation failed: {nativeReturn}.");
            _allocationAddress = allocated;
            // Keep the WHOLE successful allocation, including any portion outside
            // the tested POSIX view. Select an aligned16KiB window inside32KiB;
            // Rosetta's4KiB API is not treated as universal Mach granularity.
            try
            {
                // Output may fail: successful allocation ownership is already
                // inside this exact-scope cleanup guard before any formatting.
                Console.WriteLine($"fixture-mach-allocation return={nativeReturn} requested_extent={AllocationExtent} provider_posix_page={host.PageSize}");
                Assert.NotEqual(0ul, allocated); Assert.Equal(0ul, allocated % host.PageSize);
                Address = checked((allocated + 16383) & ~16383ul);
                Assert.True(Address >= allocated && Address - allocated <= AllocationExtent - 16384);
                Assert.True(host.PageSize <= 16384);
                host.OwnedHoles.Add(Address); Assert.True(host.OwnedHoles.Count <= 2);
                if (!host.InitializeOwnedHole(Address, allocated, AllocationExtent))
                    throw new InvalidOperationException("Fixture owned-placeholder restoration uncertain; retained until process exit.");
            }
            catch (Exception error)
            {
                // No address is released then reused as a vacancy hint. If the
                // provider conversion was attempted, any unknown scope stays
                // quarantined. Before conversion only, the exact Mach receipt
                // permits one cleanup attempt; capture its result with primary.
                if (!host.UnknownHoles.Contains(Address))
                {
                    var release = MachVmDeallocate(MachTaskSelf(), allocated, AllocationExtent);
                    if (release != 0) error.Data["FixtureMachSetupCleanupReturn"] = release;
                }
                throw;
            }
        }

        public void Dispose()
        {
            if (_released || _quarantined || _host.QuarantineHoleCleanup || _host.UnknownHoles.Contains(Address)) return;
            try
            {
                if (!_host.FreeHole(_allocationAddress, AllocationExtent))
                    throw new InvalidOperationException("Fixture-owned hole cleanup uncertain; never repeat.");
                _released = true; _host.OwnedHoles.Remove(Address);
            }
            catch
            {
                // Free may have succeeded in the kernel before a provider
                // bookkeeping throw. Retain the full exact obligation but never
                // retry this syscall against a possibly reused released extent.
                _quarantined = true;
                throw;
            }
        }
    }

    // ABI/VM_FLAGS_ANYWHERE verified in the installed Apple SDK mach_vm.h and
    // vm_statistics.h. No VM_FLAGS_OVERWRITE or release/reacquire gap.
    [DllImport("libSystem.dylib", EntryPoint = "mach_task_self")] private static extern uint MachTaskSelf();
    [DllImport("libSystem.dylib", EntryPoint = "mach_vm_allocate")] private static extern int MachVmAllocate(uint task, ref ulong address, ulong size, int flags);
    [DllImport("libSystem.dylib", EntryPoint = "mach_vm_deallocate")] private static extern int MachVmDeallocate(uint task, ulong address, ulong size);

    // This observes real provider outcomes without replacing any syscall result
    // or release callback. A callback runs only after actual Map returns success.
    private sealed class ObservedProvider : IHostViewMemory
    {
        private readonly IHostViewMemory _inner = SharpEmu.Host.MacOS.MacOSHost.CreateViewMemory();
        internal readonly HashSet<ulong> OwnedHoles = []; internal readonly HashSet<ulong> UnknownHoles = [];
        internal HostBackingObject? Backing; internal Action? AfterSuccessfulMap;
        internal volatile bool QuarantineHoleCleanup; // a timed-out owned task may still touch a reservation
        internal int MapCalls, UnmapCalls;
        public ulong PageSize => _inner.PageSize; public ulong Granularity => _inner.Granularity;
        internal HostBackingObject Create(ulong size) { Assert.True(TryCreateBacking(size, out var owner, out _)); return owner!; }
        internal bool InitializeOwnedHole(ulong selected, ulong allocation, ulong extent)
        {
            // Still own this entire successful Mach allocation. Convert it
            // directly; MAP_FIXED cannot target an intervening reused gap.
            try
            {
                var ok = _inner.UnmapView(allocation, extent);
                if (!ok) UnknownHoles.Add(selected);
                return ok;
            }
            catch { UnknownHoles.Add(selected); throw; }
        }
        public bool TryCreateBacking(ulong size, out HostBackingObject? backing, out HostViewFailure failure)
        {
            Assert.True(size <= 4 * PageSize);
            var ok = _inner.TryCreateBacking(size, out backing, out failure);
            if (ok) Backing = backing; return ok;
        }
        public ulong ReserveHole(ulong address, ulong size) => _inner.ReserveHole(address, size);
        public ulong FindUnmappedRange(ulong address, ulong size, ulong searchEnd) => _inner.FindUnmappedRange(address, size, searchEnd);
        public bool SplitHole(ulong address, ulong size) => _inner.SplitHole(address, size);
        public bool JoinHoles(ulong address, ulong size) => _inner.JoinHoles(address, size);
        public bool FreeHole(ulong address, ulong size) => _inner.FreeHole(address, size);
        public bool TryMapView(HostBackingObject backing, ulong address, ulong offset, ulong size, HostPageProtection protection, out HostViewFailure failure)
        {
            Assert.Contains(address, OwnedHoles); Assert.Equal(PageSize, size); MapCalls++;
            bool ok;
            try { ok = _inner.TryMapView(backing, address, offset, size, protection, out failure); }
            catch { UnknownHoles.Add(address); throw; }
            // Expected closed-owner rejection precedes syscall; other failed
            // outcomes are conservatively quarantined rather than retried.
            if (!ok && failure != HostViewFailure.BackingUnavailable) UnknownHoles.Add(address);
            if (ok)
            {
                try { AfterSuccessfulMap?.Invoke(); }
                catch { UnknownHoles.Add(address); throw; }
            }
            return ok;
        }
        public bool UnmapView(ulong address, ulong size)
        {
            Assert.Contains(address, OwnedHoles); Assert.Equal(PageSize, size); UnmapCalls++;
            try { var ok = _inner.UnmapView(address, size); if (!ok) UnknownHoles.Add(address); return ok; }
            catch { UnknownHoles.Add(address); throw; }
        }
        public bool CommitPrivate(ulong address, ulong size, HostPageProtection protection) => throw new NotSupportedException();
        public bool ReleasePrivate(ulong address, ulong size) => throw new NotSupportedException();
        public bool ChangeAccess(ulong address, ulong size, HostPageProtection protection) => throw new NotSupportedException();
        public IReadOnlyList<HostPageAccess> QueryAccess(ulong address, ulong size) => _inner.QueryAccess(address, size);
        public bool FreeOwnedRange(ulong address, ulong size) => throw new NotSupportedException();
    }
}

[SupportedOSPlatform("macos")]
public sealed class MacBackingLifecycleTests(ITestOutputHelper output)
{
    [BackingNativeTheory]
    [InlineData("roundtrip")]
    [InlineData("held-access")]
    [InlineData("copied-view-close")]
    [InlineData("pending-map-close")]
    public async Task RealMacBackingLifecycleWorker(string scenario)
    {
        Assert.Equal("1", Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_BACKING_NATIVE"));
        var expected = Environment.GetEnvironmentVariable("SHARPEMU_TEST_EXPECT_ARCH");
        Assert.Equal(RuntimeInformation.ProcessArchitecture.ToString(), expected);
        var runtime = new DirectoryInfo(RuntimeEnvironment.GetRuntimeDirectory());
        var host = Path.Combine(runtime.Parent!.Parent!.Parent!.FullName, "dotnet");
        var start = new ProcessStartInfo(host) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true };
        start.ArgumentList.Add(typeof(MacBackingLifecycleTests).Assembly.Location); start.ArgumentList.Add("--mac-backing-lifecycle=" + scenario);
        using var worker = Process.Start(start)!;
        var stdout = worker.StandardOutput.ReadToEndAsync(); var stderr = worker.StandardError.ReadToEndAsync();
        try
        {
            await worker.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(30));
            var captured = await stdout.WaitAsync(TimeSpan.FromSeconds(10)); var errors = await stderr.WaitAsync(TimeSpan.FromSeconds(10));
            output.WriteLine(captured); if (errors.Length != 0) output.WriteLine(errors);
            Assert.Equal(0, worker.ExitCode); Assert.Empty(errors); Assert.Contains("backing-native-pass case=" + scenario, captured);
            Assert.Contains("architecture=" + expected, captured); Assert.Contains("ordinary-alias-descriptor-only", captured);
        }
        finally
        {
            if (!worker.HasExited) { worker.Kill(entireProcessTree: true); await worker.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(10)); }
        }
    }
}
internal sealed class BackingNativeTheoryAttribute : TheoryAttribute
{
    public BackingNativeTheoryAttribute()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_BACKING_NATIVE") != "1")
            Skip = "Actual Mac backing workers require separately authorized SHARPEMU_TEST_REQUIRE_BACKING_NATIVE=1.";
    }
}
#endif
