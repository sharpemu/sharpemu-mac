// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Iced.Intel;
using SharpEmu.Core.Loader;
using SharpEmu.Core.Memory;
using SharpEmu.HLE;
using Xunit;
using static Iced.Intel.AssemblerRegisters;

namespace SharpEmu.Libs.Tests.Loader;

[Collection(GuestRedZonePatcherCollection.Name)]
public sealed unsafe class SpinLoopNapPatcherTests
{
    // A threshold far longer than any test loop, so the probe tracks but never sleeps: the patched
    // code must run exactly as the original. The runtime caches the probe, so this must win the
    // first EnsureProbe of the process; the collection serializes these tests to keep it so.
    static SpinLoopNapPatcherTests()
    {
        Environment.SetEnvironmentVariable(SpinNapRuntime.IdleEnvironmentVariable, "100000");
    }

    [Fact]
    public void ProbeIsPositionIndependentAndNonZero()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        Assert.NotEmpty(SpinNapRuntime.EnsureProbeBytes(1_000_000_000));
        Assert.Same(SpinNapRuntime.EnsureProbeBytes(1_000_000_000), SpinNapRuntime.EnsureProbeBytes(1_000_000_000));
    }

    [Fact]
    public void TicksForScalesWithFrequencyAndClamps()
    {
        Assert.Equal(500_000L, SpinNapRuntime.TicksFor(500, 1_000_000_000));
        Assert.Equal(1_500_000L, SpinNapRuntime.TicksFor(500, 3_000_000_000));
        Assert.Equal((long)int.MaxValue, SpinNapRuntime.TicksFor(100_000, 100_000_000_000));
    }

    // BuildProbe(table 0x600000000000, threshold 500000, gap 50000, usleep(250) at 0x1122334455667788,
    // control 0x700000001000) as assembled before SHARPEMU_PAUSE_DELAY_NS existed (main bf7a409ce).
    private const string ProbeBeforePauseDelay =
        "9C50515253555657415041514152415341544155415641574989E349C1EB1A0F3148C1E2204809C24989D24C89D84825" +
        "FF1F000048C1E00548BB00000000006000004801C3488B034C39D80F857F0000004C89D1482B4B1048B850C300000000" +
        "00004839C172044C8953084C8953104C89D1482B4B0848B820A10700000000004839C1725648BE00100000007000004D" +
        "89D54889E54883E4F04881EC00020000480FAE04244989E44883EC20FCBFFA000000B9FA00000048B888776655443322" +
        "11FFD0490FAE0C244889EC4D89EAEB0B4C891B4C8953084C895310415F415E415D415C415B415A415941585F5E5D5B5A" +
        "59589DC3";

    // BuildLockedProbe(table 0x600000000000, generations 0x600000100000, threshold 500000, gap 50000,
    // density 5000, 4 sites, usleep(250) at 0x1122334455667788, control 0x700000001000) as assembled
    // before per-thread probe state existed (f2d75faa8).
    private const string LockedProbeBeforePrivateState =
        "9C50515253555657415041514152415341544155415641574989E349C1EB1A0F3148C1E2204809C24989D24C89D84825" +
        "FF03000048B900001000006000004C8B0CC148C1E00848BB00000000006000004801C34C8B8424800000004C391B0F85" +
        "F10000004C89D0482B43104C89531048B950C30000000000004839C80F83D30000004C3B4B280F85C900000048FF4318" +
        "31C94883F9047323488B44CB404C39C0740F4885C0740548FFC1EBE64C8944CB4048FF84CB80000000EB0448FF43204C" +
        "89D0482B430848B920A10700000000004839C80F82ED000000488B4B1848BA8813000000000000480FAFCA4839C1725F" +
        "488B4B20488D0C894801C9483B4B18774E48BE0010000000700000F048FF46084D89D54889E54883E4F04881EC000200" +
        "00480FAE04244989E44883EC20FCBFFA000000B9FA00000048B88877665544332211FFD0490FAE0C244889EC4D89EA4C" +
        "895308EB0F4C891B4C8953084C8953104C894B2831C04889431848894320488943404889838000000048894348488983" +
        "880000004889435048898390000000488943584889839800000048894360488983A000000048894368488983A8000000" +
        "48894370488983B000000048894378488983B8000000415F415E415D415C415B415A415941585F5E5D5B5A59589DC3";

    private static byte[] FixedProbe(long delayTicks, ulong statsAddress, bool sharedState = true) =>
        SpinNapRuntime.BuildProbeForTest(
            0x0000600000000000UL, 500_000, 50_000, 0x1122334455667788UL, 250, 0x0000700000001000UL, delayTicks, statsAddress,
            sharedState);

    private static byte[] FixedLockedProbe(bool sharedState) =>
        SpinNapRuntime.BuildLockedProbeForTest(
            0x0000600000000000UL, 0x0000600000100000UL, 500_000, 50_000, 5_000, 4, 0x1122334455667788UL, 250,
            0x0000700000001000UL, sharedState);

    private static bool Contains(byte[] code, string hex) =>
        Convert.ToHexString(code).Contains(hex, StringComparison.Ordinal);

    private static int CountRdtsc(byte[] code)
    {
        var count = 0;
        for (var index = 0; index + 1 < code.Length; index++)
        {
            if (code[index] == 0x0F && code[index + 1] == 0x31)
            {
                count++;
            }
        }

        return count;
    }

    // SHARPEMU_SPIN_NAP_SHARED_STATE=1 restores the old keying: with no pause delay and no counters,
    // both probes are the old bytes exactly.
    [Fact]
    public void SharedStateProbesAreByteForByteTheOldOnes()
    {
        Assert.Equal(ProbeBeforePauseDelay, Convert.ToHexString(FixedProbe(0, 0, sharedState: true)));
        Assert.Equal(LockedProbeBeforePrivateState, Convert.ToHexString(FixedLockedProbe(sharedState: true)));
        Assert.NotEqual(ProbeBeforePauseDelay, Convert.ToHexString(FixedProbe(0, 0, sharedState: false)));
        Assert.NotEqual(LockedProbeBeforePrivateState, Convert.ToHexString(FixedLockedProbe(sharedState: false)));
    }

    [Fact]
    public void SharedStateIsOffByDefault()
    {
        using (EnvironmentVariableScope.Set(SpinNapRuntime.SharedStateEnvironmentVariable, ""))
        {
            Assert.False(SpinNapRuntime.SharedStateEnabled);
        }

        using (EnvironmentVariableScope.Set(SpinNapRuntime.SharedStateEnvironmentVariable, "1"))
        {
            Assert.True(SpinNapRuntime.SharedStateEnabled);
        }
    }

    // The private probes key on the host thread pointer with a segment-relative load and never read
    // rsp for it; the shared ones do `mov r11, rsp; shr r11, 26` and no segment load. On Linux the load
    // is based (xor r11d, r11d ; mov r11, fs:[r11]): the guest-TLS scan rewrites a base-less
    // `mov r11, fs:[0]` into a call that returns the guest's TLS base.
    [Fact]
    public void PrivateProbesKeyOnTheHostThreadPointer()
    {
        var threadKeyLoad =
            OperatingSystem.IsWindows() ? "654C8B1C2530000000" :      // mov r11, gs:[0x30]
            OperatingSystem.IsMacOS() ? "654C8B1C2500000000" :        // mov r11, gs:[0]
            "4531DB644D8B1B";                                         // xor r11d, r11d ; mov r11, fs:[r11]
        const string baseLessFsLoad = "644C8B1C2500000000";           // mov r11, fs:[0]
        const string rspKey = "4989E349C1EB1A";                       // mov r11, rsp ; shr r11, 26
        foreach (var probe in new[] { FixedProbe(0, 0, sharedState: false), FixedLockedProbe(sharedState: false) })
        {
            Assert.True(Contains(probe, threadKeyLoad));
            Assert.False(Contains(probe, baseLessFsLoad));
            Assert.False(Contains(probe, rspKey));
        }

        foreach (var probe in new[] { FixedProbe(0, 0, sharedState: true), FixedLockedProbe(sharedState: true) })
        {
            Assert.False(Contains(probe, threadKeyLoad));
            Assert.True(Contains(probe, rspKey));
        }
    }

    [Fact]
    public void TheThreadKeyIsTheHostThreadPointer()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var mine = SpinNapRuntime.CurrentThreadKey();
        Assert.NotEqual(0UL, mine);
        Assert.Equal(mine, SpinNapRuntime.CurrentThreadKey());
        ulong other = 0;
        RunOnNewThread(() => other = SpinNapRuntime.CurrentThreadKey());
        Assert.NotEqual(0UL, other);
        Assert.NotEqual(mine, other);
        if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            Assert.Equal((ulong)pthread_self(), mine);
        }
    }

    [DllImport("libc")]
    private static extern nint pthread_self();

    // The backend's guest-TLS scan runs over every executable guest region, the spin-nap arena
    // included, after the loader copied the probes there. Run the real scan over each probe in an
    // executable page: it must leave every byte alone. The control, a base-less `mov r11, fs:[0]`,
    // shows the scan does patch what it matches.
    [Fact]
    public void TheGuestTlsScanLeavesTheProbesAlone()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
        var probes = new List<byte[]>();
        foreach (var sharedState in new[] { false, true })
        {
            probes.Add(FixedProbe(0, 0, sharedState));
            probes.Add(FixedProbe(19, 0x0000710000000000UL, sharedState));
            probes.Add(FixedLockedProbe(sharedState));
            probes.Add(SpinNapRuntime.BuildNappingLockedProbeForTest(tscFrequency, sharedState));
        }

        foreach (var probe in probes)
        {
            using var scan = new TlsScanFixture();
            scan.Write(probe);
            scan.Scan(probe.Length);
            Assert.Equal(probe, scan.Read(probe.Length));
        }

        using (var control = new TlsScanFixture())
        {
            byte[] baseLessLoad = [0x64, 0x4C, 0x8B, 0x1C, 0x25, 0, 0, 0, 0, 0xC3]; // mov r11, fs:[0] ; ret
            control.Write(baseLessLoad);
            control.Scan(baseLessLoad.Length);
            Assert.NotEqual(baseLessLoad, control.Read(baseLessLoad.Length));
        }
    }

    // End to end: after the TLS scan has run over them in place, the probes still key on the host
    // thread pointer, and NoteHleEntry bumps the HLE generation in the slot the locked probe uses
    // for the same thread (a patched key load would put the probe in another slot, so the thread's
    // own HLE calls would never end its window).
    [Fact]
    public void AfterTheTlsScanTheProbesAndNoteHleEntryShareTheThreadsSlot()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
        Assert.NotEmpty(SpinNapRuntime.EnsureLockedProbeBytes(1_000_000_000));
        if (SpinNapRuntime.RuntimeLockedSharedState)
            return; // the process built the runtime probe with SHARPEMU_SPIN_NAP_SHARED_STATE=1

        var lockedTable = SpinNapRuntime.RuntimeLockedTable;
        var pauseTable = SpinNapRuntime.AllocatePauseTableForTest(sharedState: false);
        var locked = SpinNapRuntime.BuildRunnableLockedProbeForTest(
            tscFrequency, lockedTable, null, sharedState: false, int.MaxValue, int.MaxValue, int.MaxValue, 8);
        var pause = SpinNapRuntime.BuildRunnableProbeForTest(
            tscFrequency, pauseTable, sharedState: false, long.MaxValue, long.MaxValue, 0, 0);
        using var lockedScan = new TlsScanFixture();
        using var pauseScan = new TlsScanFixture();
        lockedScan.Write(locked);
        lockedScan.Scan(locked.Length);
        pauseScan.Write(pause);
        pauseScan.Scan(pause.Length);

        using var threads = new WorkerThreads();
        var (thread, _) = threads.TwoWithDistinctSlots(key => (nint)SpinNapRuntime.LockedSlot(lockedTable, false, key));
        var lockedSlot = (ulong*)SpinNapRuntime.LockedSlot(lockedTable, false, thread.Key);
        var pauseSlot = (ulong*)SpinNapRuntime.PauseSlot(pauseTable, false, thread.Key);
        thread.Run(() =>
        {
            ((delegate* unmanaged<void>)lockedScan.Address)();
            ((delegate* unmanaged<void>)lockedScan.Address)();
            ((delegate* unmanaged<void>)pauseScan.Address)();
        });
        Assert.Equal(thread.Key, lockedSlot[0]);
        Assert.Equal(1UL, lockedSlot[3]);
        Assert.Equal(thread.Key, pauseSlot[0]);

        var generation = (long*)((byte*)lockedSlot + SpinNapRuntime.LockedHleGenerationOffsetForTest);
        var before = *generation;
        thread.Run(() =>
        {
            SpinNapRuntime.NoteHleEntry(0);
            ((delegate* unmanaged<void>)lockedScan.Address)();
        });
        Assert.Equal(before + 1, *generation);
        Assert.Equal(0UL, lockedSlot[3]); // the thread's own HLE call restarted its window
    }

    // The backend's TLS scan over one executable page, with a stand-in TLS handler on a later page
    // (as TlsLoadPatchBoundaryTests does, through HostMemory so it runs on every host).
    private sealed class TlsScanFixture : IDisposable
    {
        private const int Bytes = 0x4000;
        private readonly SharpEmu.Core.Cpu.Native.DirectExecutionBackend _backend = new(new ModuleManager());

        public TlsScanFixture()
        {
            Address = (nint)HostMemory.Alloc(
                null, Bytes, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
            Assert.NotEqual(0, Address);
            new Span<byte>((void*)Address, Bytes).Fill(0x90);
            var handler = new Span<byte>((void*)(Address + 0x3000), 11);
            handler[0] = 0x48; // mov rax, imm64 ; ret
            handler[1] = 0xB8;
            BinaryPrimitives.WriteUInt64LittleEndian(handler[2..], 0x00007FFE00000000UL);
            handler[10] = 0xC3;
            SetField("_tlsHandlerAddress", Address + 0x3000);
            SetField("_tlsPatchStubOffset", 0x100);
        }

        public nint Address { get; }

        public void Write(byte[] code) => code.CopyTo(new Span<byte>((void*)Address, code.Length));

        public byte[] Read(int length) => new ReadOnlySpan<byte>((void*)Address, length).ToArray();

        public void Scan(int length) =>
            typeof(SharpEmu.Core.Cpu.Native.DirectExecutionBackend)
                .GetMethod("PatchTlsPatternsInRange", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .Invoke(_backend, [(ulong)Address, (ulong)Address + (ulong)length]);

        private void SetField(string name, object value) =>
            typeof(SharpEmu.Core.Cpu.Native.DirectExecutionBackend)
                .GetField(name, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .SetValue(_backend, value);

        public void Dispose()
        {
            SetField("_tlsHandlerAddress", (nint)0);
            _backend.Dispose();
            _ = HostMemory.Free((void*)Address, 0, HostMemory.MEM_RELEASE);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OnlyADelayAddsAnRdtsc(bool sharedState)
    {
        Assert.Equal(1, CountRdtsc(FixedProbe(0, 0, sharedState)));
        Assert.Equal(1, CountRdtsc(FixedProbe(0, 0x0000710000000000UL, sharedState))); // counters alone: no second clock read
        Assert.Equal(2, CountRdtsc(FixedProbe(19, 0, sharedState)));
        Assert.Equal(2, CountRdtsc(FixedProbe(19, 0x0000710000000000UL, sharedState)));
    }

    // The delay loop's elapsed compare is unsigned (jb), so a TSC that steps back ends the wait
    // instead of stretching it; the old signed jl is gone.
    [Fact]
    public void TheDelayLoopComparesUnsigned()
    {
        foreach (var sharedState in new[] { false, true })
        {
            var probe = FixedProbe(19, 0, sharedState);
            Assert.True(Contains(probe, "4883F91372"));  // cmp rcx, 19 ; jb
            Assert.False(Contains(probe, "4883F9137C")); // cmp rcx, 19 ; jl
        }
    }

    [Theory]
    [InlineData(19u, 1_000_000_000UL, 19L)]      // Rosetta reports a 1 GHz TSC
    [InlineData(19u, 3_500_000_000UL, 67L)]      // PS5 Zen 2 at 3.5 GHz: 66.5 rounds up
    [InlineData(38u, 2_445_000_000UL, 93L)]      // 92.91 rounds up
    [InlineData(19u, 24_000_000UL, 1L)]          // a 24 MHz counter: never rounds a delay to nothing
    [InlineData(19u, 10_000_000UL, 1L)]
    [InlineData(1_000u, 3_000_000_000UL, 3_000L)] // exact: no extra tick
    [InlineData(0u, 3_500_000_000UL, 0L)]
    [InlineData(100_000u, 100_000_000_000_000UL, int.MaxValue)]
    public void PauseDelayNanosecondsConvertToTscTicks(uint nanoseconds, ulong tscFrequency, long ticks)
    {
        Assert.Equal(ticks, SpinNapRuntime.TicksForNanoseconds(nanoseconds, tscFrequency));
    }

    [Fact]
    public void PauseDelayIsOffByDefaultAndCountsFollowIt()
    {
        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseDelayEnvironmentVariable, ""))
        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseStatsEnvironmentVariable, ""))
        {
            Assert.Equal(0u, SpinNapRuntime.PauseDelayNanoseconds);
            Assert.False(SpinNapRuntime.PauseStatsEnabled);
        }

        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseDelayEnvironmentVariable, "19"))
        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseStatsEnvironmentVariable, ""))
        {
            Assert.Equal(19u, SpinNapRuntime.PauseDelayNanoseconds);
            Assert.True(SpinNapRuntime.PauseStatsEnabled);
        }

        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseDelayEnvironmentVariable, "19"))
        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseStatsEnvironmentVariable, "0"))
        {
            Assert.False(SpinNapRuntime.PauseStatsEnabled);
        }

        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseDelayEnvironmentVariable, "0"))
        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseStatsEnvironmentVariable, "1"))
        {
            Assert.True(SpinNapRuntime.PauseStatsEnabled);
        }

        using (EnvironmentVariableScope.Set(SpinNapRuntime.PauseDelayEnvironmentVariable, "1000000"))
        {
            Assert.Equal(0u, SpinNapRuntime.PauseDelayNanoseconds); // out of range: off
        }
    }

    // A probe call must last at least the delay, measured on the host clock rather than the TSC the
    // probe uses. The bounds are lower bounds only (with 10% for TSC calibration error), so a slow or
    // preempted CI machine can only make the call longer.
    [Fact]
    public void ADelayingProbeCallTakesAtLeastTheDelay()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
        const uint delayNanoseconds = 50_000;
        var probe = SpinNapRuntime.BuildDelayingProbeForTest(tscFrequency, delayNanoseconds, out var stats);
        var page = (byte*)HostMemory.Alloc(
            null, 4096, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
        Assert.True(page != null);
        try
        {
            probe.CopyTo(new Span<byte>(page, probe.Length));
            var call = (delegate* unmanaged<void>)page;
            call(); // warm up the page and the counters' line

            var single = System.Diagnostics.Stopwatch.StartNew();
            call();
            single.Stop();
            Assert.True(
                single.Elapsed.TotalMicroseconds >= delayNanoseconds / 1000.0 * 0.9,
                $"one call took {single.Elapsed.TotalMicroseconds:F1} us");

            const int calls = 20;
            var many = System.Diagnostics.Stopwatch.StartNew();
            for (var index = 0; index < calls; index++)
            {
                call();
            }

            many.Stop();
            Assert.True(
                many.Elapsed.TotalMicroseconds >= calls * delayNanoseconds / 1000.0 * 0.9,
                $"{calls} calls took {many.Elapsed.TotalMicroseconds:F1} us");

            var (entries, delayedTicks) = SpinNapRuntime.ReadPauseStats((byte*)stats);
            Assert.Equal(calls + 2, entries);
            Assert.True(delayedTicks >= (calls + 2) * SpinNapRuntime.TicksForNanoseconds(delayNanoseconds, tscFrequency));
        }
        finally
        {
            _ = HostMemory.Free(page, 0, HostMemory.MEM_RELEASE);
        }
    }

    // The delaying probe still hands the guest back every register and flag: the loop counter in
    // ecx and the flags the jnz reads survive four delayed probe calls.
    [Fact]
    public void ADelayingProbeInAPatchedLoopStaysTransparent()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        byte[] function =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00, // mov ecx, 4
            0xF3, 0x90,                   // loop: pause
            0xFF, 0xC9,                   // dec ecx
            0x75, 0xFA,                   // jnz loop
            0xB8, 0x44, 0x33, 0x22, 0x11, // mov eax, 0x11223344
            0xC3,                         // ret
        ];

        var probe = SpinNapRuntime.BuildDelayingProbeForTest(
            SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency, 1_000, out var stats);
        using var image = SpinImage.Create(function, probe: probe);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
        Assert.Equal(4, SpinNapRuntime.ReadPauseStats((byte*)stats).Entries);
    }

    // A delay longer than the streak gap must not read as a gap on every entry: the gap is measured
    // from where the entry began, before its delay. With the gap measured after the delay (the old
    // probe), every entry here restarted the streak and the probe never napped.
    [Fact]
    public void ADelayLongerThanTheGapStillNaps()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
        var probe = SpinNapRuntime.BuildRunnableProbeForTest(
            tscFrequency, SpinNapRuntime.AllocatePauseTableForTest(sharedState: false), sharedState: false,
            thresholdTicks: SpinNapRuntime.TicksFor(200, tscFrequency),
            gapTicks: SpinNapRuntime.TicksFor(20, tscFrequency),
            delayTicks: SpinNapRuntime.TicksForNanoseconds(50_000, tscFrequency),
            statsAddress: (ulong)SpinNapRuntime.AllocatePauseStatsForTest());
        using var code = new ExecutableCode(probe);
        var napsBefore = SpinNapRuntime.PauseNapCount;
        var call = (delegate* unmanaged<void>)code.Address;
        for (var index = 0; index < 40; index++)
        {
            call();
        }

        Assert.True(SpinNapRuntime.PauseNapCount - napsBefore >= 1, "a 50 us delay with a 20 us gap never napped");
    }

    // Every general register and the arithmetic, sign, overflow and direction flags come back as the
    // guest left them, on the tracking path and through a nap, for each probe in each keying mode.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProbesPreserveEveryRegisterAndFlag(bool sharedState)
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
        var probes = new[]
        {
            // Naps from the second call on (threshold 0, a gap that never breaks the streak).
            SpinNapRuntime.BuildRunnableProbeForTest(
                tscFrequency, SpinNapRuntime.AllocatePauseTableForTest(sharedState), sharedState, 0, int.MaxValue, 0, 0),
            // The same with a delay and counters.
            SpinNapRuntime.BuildRunnableProbeForTest(
                tscFrequency, SpinNapRuntime.AllocatePauseTableForTest(sharedState), sharedState, 0, int.MaxValue,
                SpinNapRuntime.TicksForNanoseconds(1_000, tscFrequency), (ulong)SpinNapRuntime.AllocatePauseStatsForTest()),
            // Never naps.
            SpinNapRuntime.BuildRunnableProbeForTest(
                tscFrequency, SpinNapRuntime.AllocatePauseTableForTest(sharedState), sharedState, int.MaxValue, int.MaxValue, 0, 0),
            // Naps on every hit after the first.
            SpinNapRuntime.BuildNappingLockedProbeForTest(tscFrequency, sharedState),
        };

        foreach (var probe in probes)
        {
            using var code = new ExecutableCode(probe);
            foreach (var flags in new ulong[] { RegisterHarness.AllTestedFlags, 0 })
            {
                for (var call = 0; call < 3; call++)
                {
                    var (registers, flagsOut) = RegisterHarness.Call(code.Address, flags);
                    Assert.Equal(RegisterHarness.Inputs, registers);
                    Assert.Equal(flags, flagsOut & RegisterHarness.AllTestedFlags);
                }
            }
        }
    }

    // Two guest threads whose stacks are 16 KiB apart (two fibers from one pool, in the same 64 MiB
    // bucket) keep separate pause streaks: the second thread's first pause starts its own streak and
    // leaves the first's alone. With the shared state they are one streak: the second thread joins
    // the first's, which is the old behaviour the off-switch restores.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TwoThreadsOnNeighbouringStacksKeepIndependentPauseStreaks(bool sharedState)
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
        var table = SpinNapRuntime.AllocatePauseTableForTest(sharedState);
        using var probe = new ExecutableCode(SpinNapRuntime.BuildRunnableProbeForTest(
            tscFrequency, table, sharedState, int.MaxValue, int.MaxValue, 0, 0));
        using var stacks = new TwoStacks();
        using var threads = new WorkerThreads();
        var (threadA, threadB) = threads.TwoWithDistinctSlots(key => (nint)SpinNapRuntime.PauseSlot(table, false, key));

        threadA.Run(() => CallOnStack(probe.Address, stacks.TopA));
        var keyA = sharedState ? stacks.SharedKeyA : threadA.Key;
        var slotA = (ulong*)SpinNapRuntime.PauseSlot(table, sharedState, keyA);
        var epochA = slotA[1];
        Assert.Equal(keyA, slotA[0]);
        Assert.NotEqual(0UL, epochA);

        threadB.Run(() => CallOnStack(probe.Address, stacks.TopB));
        var keyB = sharedState ? stacks.SharedKeyB : threadB.Key;
        var slotB = (ulong*)SpinNapRuntime.PauseSlot(table, sharedState, keyB);

        if (sharedState)
        {
            Assert.Equal(keyA, keyB); // one bucket for both stacks
            Assert.Equal(epochA, slotB[1]); // B joined A's streak
            Assert.True(slotB[2] > epochA); // and moved its `last`
            return;
        }

        Assert.True(slotA != slotB);
        Assert.Equal(keyB, slotB[0]);
        Assert.True(slotB[1] > epochA); // B's own streak, started at B's first pause
        Assert.Equal(keyA, slotA[0]);
        Assert.Equal(epochA, slotA[1]); // A's streak untouched
        Assert.Equal(epochA, slotA[2]);

        // The same thread on another stack (a fiber switch) stays in its own streak.
        threadA.Run(() => CallOnStack(probe.Address, stacks.TopB));
        Assert.Equal(epochA, slotA[1]);
        Assert.True(slotA[2] > epochA);
    }

    // One thread's locked-op nap restarts only its own window. Thread A builds a window through a
    // probe that never naps; thread B, on a stack 16 KiB away, naps through a probe over the same
    // table. A's hits survive with private state; with the shared state B's nap clears them.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OneThreadsLockedNapDoesNotResetAnothersWindow(bool sharedState)
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
        var table = SpinNapRuntime.AllocateLockedTableForTest(sharedState);
        var generations = (long*)NativeMemory.AllocZeroed(1024, sizeof(long));
        using var tracker = new ExecutableCode(SpinNapRuntime.BuildRunnableLockedProbeForTest(
            tscFrequency, table, generations, sharedState, int.MaxValue, int.MaxValue, int.MaxValue, 8));
        using var napper = new ExecutableCode(SpinNapRuntime.BuildRunnableLockedProbeForTest(
            tscFrequency, table, generations, sharedState, 0, int.MaxValue, int.MaxValue, 8));
        using var stacks = new TwoStacks();
        using var threads = new WorkerThreads();
        var (threadA, threadB) = threads.TwoWithDistinctSlots(key => (nint)SpinNapRuntime.LockedSlot(table, false, key));

        threadA.Run(() =>
        {
            for (var hit = 0; hit < 3; hit++)
            {
                CallOnStack(tracker.Address, stacks.TopA); // opens the window, then 2 hits
            }
        });
        var slotA = (ulong*)SpinNapRuntime.LockedSlot(table, sharedState, sharedState ? stacks.SharedKeyA : threadA.Key);
        Assert.Equal(2UL, slotA[3]);
        var startA = slotA[1];

        var napsBefore = SpinNapRuntime.LockedNapCount;
        threadB.Run(() =>
        {
            CallOnStack(napper.Address, stacks.TopB);
            CallOnStack(napper.Address, stacks.TopB);
        });
        Assert.True(SpinNapRuntime.LockedNapCount - napsBefore >= 1);

        threadA.Run(() => CallOnStack(tracker.Address, stacks.TopA));
        if (sharedState)
        {
            Assert.Equal(1UL, slotA[3]); // B's nap restarted the one shared window
            Assert.NotEqual(startA, slotA[1]);
        }
        else
        {
            Assert.Equal(3UL, slotA[3]);
            Assert.Equal(startA, slotA[1]);
        }
    }

    // An HLE call ends the calling thread's locked-op window and nobody else's: NoteHleEntry bumps
    // the generation in the caller's own slot of the runtime table, which the probe compares.
    [Fact]
    public void AnHleCallEndsOnlyTheCallersLockedWindow()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        Assert.NotEmpty(SpinNapRuntime.EnsureLockedProbeBytes(1_000_000_000));
        if (SpinNapRuntime.RuntimeLockedSharedState)
            return; // the process built the runtime probe with SHARPEMU_SPIN_NAP_SHARED_STATE=1

        var table = SpinNapRuntime.RuntimeLockedTable;
        using var tracker = new ExecutableCode(SpinNapRuntime.BuildRunnableLockedProbeForTest(
            SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency, table, null, sharedState: false,
            int.MaxValue, int.MaxValue, int.MaxValue, 8));
        var call = (delegate* unmanaged<void>)tracker.Address;
        using var threads = new WorkerThreads();
        var (threadA, threadB) = threads.TwoWithDistinctSlots(key => (nint)SpinNapRuntime.LockedSlot(table, false, key));
        var slotA = (ulong*)SpinNapRuntime.LockedSlot(table, false, threadA.Key);
        var slotB = SpinNapRuntime.LockedSlot(table, false, threadB.Key);
        var generationA = (long*)((byte*)slotA + SpinNapRuntime.LockedHleGenerationOffsetForTest);
        var generationB = (long*)(slotB + SpinNapRuntime.LockedHleGenerationOffsetForTest);

        threadA.Run(() =>
        {
            call();
            call();
            call();
        });
        Assert.Equal(2UL, slotA[3]);
        var generationABefore = *generationA;
        var generationBBefore = *generationB;

        threadB.Run(() =>
        {
            for (var index = 0; index < 3; index++)
            {
                SpinNapRuntime.NoteHleEntry(0);
            }
        });
        Assert.Equal(3, *generationB - generationBBefore);
        Assert.Equal(generationABefore, *generationA);

        threadA.Run(() => call());
        Assert.Equal(3UL, slotA[3]); // B's HLE calls did not touch A's window

        threadA.Run(() =>
        {
            SpinNapRuntime.NoteHleEntry(0);
            call();
        });
        Assert.Equal(generationABefore + 1, *generationA);
        Assert.Equal(0UL, slotA[3]); // A's own HLE call restarted it
    }

    // SHARPEMU_SPIN_NAP_PROBE_BENCH_OUTPUT=<path>: ns per pause-probe call (the probe never naps) for 1, 4
    // and 8 threads calling it back to back on stacks 64 KiB apart in one 64 MiB bucket, private vs shared
    // state, with no delay and with the 19 ns Zen 2 delay. Best of 3 runs of 2M calls per thread; the time
    // is wall-clock per thread, so it includes cache-line contention between the threads. Run it on the
    // Mac to see what the key load and the probe cost under Rosetta.
    [Fact]
    public void MeasureProbeCostWhenRequested()
    {
        var path = Environment.GetEnvironmentVariable("SHARPEMU_SPIN_NAP_PROBE_BENCH_OUTPUT");
        if (path is null || RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // runner(stackTop, probe, calls): switch to the stack, call the probe `calls` times, switch back.
        var a = new Assembler(64);
        var loop = a.CreateLabel();
        var (top, probeRegister, count) = OperatingSystem.IsWindows() ? (rcx, rdx, r8) : (rdi, rsi, rdx);
        a.push(rbx);
        a.push(rsi);
        a.mov(rbx, count);
        a.mov(rsi, probeRegister);
        a.mov(rax, rsp);
        a.mov(rsp, top);
        a.push(rax);
        a.push(rax);
        a.Label(ref loop);
        a.call(rsi);
        a.dec(rbx);
        a.jnz(loop);
        a.pop(rax);
        a.pop(rsp);
        a.pop(rsi);
        a.pop(rbx);
        a.ret();
        using var runnerCode = new ExecutableCode(Assemble(a));
        var runner = (delegate* unmanaged<ulong, nint, long, void>)runnerCode.Address;

        const int MaxThreads = 8;
        const ulong Spacing = 0x10000;
        const int Bytes = 4 << 20;
        var block = (byte*)NativeMemory.Alloc(Bytes);
        try
        {
            var highest = ((ulong)block + Bytes - 64) & ~15UL;
            var lowest = highest - (MaxThreads - 1) * Spacing;
            if (lowest >> SpinNapRuntime.StackKeyShift != highest >> SpinNapRuntime.StackKeyShift)
            {
                highest = ((highest >> SpinNapRuntime.StackKeyShift) << SpinNapRuntime.StackKeyShift) - 64;
            }

            Assert.True(highest - MaxThreads * Spacing >= (ulong)block);
            var tscFrequency = SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.TscFrequency;
            const long Calls = 2_000_000;
            var lines = new List<string>();
            foreach (var threads in new[] { 1, 4, MaxThreads })
            foreach (var sharedState in new[] { false, true })
            foreach (var delayNanoseconds in new uint[] { 0, SpinNapRuntime.Zen2PauseDelayNanoseconds })
            {
                var best = double.MaxValue;
                for (var repetition = 0; repetition < 3; repetition++)
                {
                    using var probe = new ExecutableCode(SpinNapRuntime.BuildRunnableProbeForTest(
                        tscFrequency, SpinNapRuntime.AllocatePauseTableForTest(sharedState), sharedState, long.MaxValue,
                        long.MaxValue, SpinNapRuntime.TicksForNanoseconds(delayNanoseconds, tscFrequency), 0));
                    using var start = new Barrier(threads + 1);
                    var workers = new Thread[threads];
                    for (var index = 0; index < threads; index++)
                    {
                        var stackTop = highest - (ulong)index * Spacing;
                        var probeAddress = probe.Address;
                        workers[index] = new Thread(() =>
                        {
                            start.SignalAndWait();
                            runner(stackTop, probeAddress, Calls);
                        });
                        workers[index].Start();
                    }

                    start.SignalAndWait();
                    var clock = System.Diagnostics.Stopwatch.StartNew();
                    foreach (var worker in workers)
                    {
                        worker.Join();
                    }

                    best = Math.Min(best, clock.Elapsed.TotalNanoseconds / Calls);
                }

                lines.Add(FormattableString.Invariant(
                    $"threads={threads} state={(sharedState ? "shared" : "private")} delay_ns={delayNanoseconds} ns_per_call={best:F1}"));
            }

            File.WriteAllLines(path, lines);
        }
        finally
        {
            NativeMemory.Free(block);
        }
    }

    [Fact]
    public void PatchesAPauseInABackwardLoopAndStaysTransparent()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // mov ecx, 4 ; loop: pause ; dec ecx ; jnz loop ; mov eax, 0x11223344 ; ret
        byte[] function =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00, // mov ecx, 4
            0xF3, 0x90,                   // loop: pause
            0xFF, 0xC9,                   // dec ecx
            0x75, 0xFA,                   // jnz loop (next ip 11, -6 -> the pause at 5)
            0xB8, 0x44, 0x33, 0x22, 0x11, // mov eax, 0x11223344
            0xC3,                         // ret
        ];

        using var image = SpinImage.Create(function);
        Assert.Equal(1, image.Result.PauseSites);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0, image.Result.FailedSites);
        Assert.Equal(0, image.Result.NonLoopSites);

        // The pause (offset 5) is now the start of a jmp into the trampoline.
        Assert.Equal(0xE9, image.ReadByte(5));
        var trampoline = (ulong)((long)image.Base + 5 + 5 + BinaryPrimitives.ReadInt32LittleEndian(image.Read(6, 4)));

        // Trampoline lead: lea rsp,[rsp-256]; call rel32; lea rsp,[rsp+256].
        var lead = image.ReadAbsolute(trampoline, 21);
        Assert.Equal(new byte[] { 0x48, 0x8D, 0xA4, 0x24, 0x00, 0xFF, 0xFF, 0xFF }, lead[..8]);
        Assert.Equal(0xE8, lead[8]);
        var probe = (ulong)((long)trampoline + 8 + 5 + BinaryPrimitives.ReadInt32LittleEndian(lead.AsSpan(9, 4)));
        // The probe is copied to the arena start, below the first trampoline; the call must reach it.
        Assert.True(probe < trampoline);
        Assert.Equal(SpinNapRuntime.EnsureProbeBytes(1_000_000_000), image.ReadAbsolute(probe, SpinNapRuntime.EnsureProbeBytes(1_000_000_000).Length));
        Assert.Equal(new byte[] { 0x48, 0x8D, 0xA4, 0x24, 0x00, 0x01, 0x00, 0x00 }, lead[13..21]);

        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    // The loader reserves the spin-nap range before the red-zone pass (GuestRedZonePatcher.ReserveSpinNapArena):
    // a reserved range that is big enough and in reach takes the trampolines; a smaller one is left alone and
    // the pass allocates as before.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TrampolinesGoToTheReservedArenaWhenItFits(bool bigEnough)
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        byte[] function =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00, // mov ecx, 4
            0xF3, 0x90,                   // loop: pause
            0xFF, 0xC9,                   // dec ecx
            0x75, 0xFA,                   // jnz loop
            0xB8, 0x44, 0x33, 0x22, 0x11, // mov eax, 0x11223344
            0xC3,                         // ret
        ];

        var arenaSize = bigEnough ? GuestRedZonePatcher.SpinNapArenaBytes : 0x1000UL;
        using var image = SpinImage.Create(function, reserveAt: 0x200000, reservedSize: arenaSize);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0xE9, image.ReadByte(5));
        var trampoline = (ulong)((long)image.Base + 5 + 5 + BinaryPrimitives.ReadInt32LittleEndian(image.Read(6, 4)));
        var arena = image.Arena!.Value;
        Assert.Equal(bigEnough, trampoline >= arena.Address && trampoline < arena.Address + arena.Size);
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void TheSpinNapArenaIsReservedJustAfterTheImage()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var memory = new PhysicalVirtualMemory();
        const ulong holeSize = 0x4000000, imageSize = 0x10000;
        var hole = (ulong)HostMemory.Alloc(null, (nuint)holeSize, HostMemory.MEM_RESERVE, HostMemory.PAGE_NOACCESS);
        Assert.NotEqual(0UL, hole);
        Assert.True(HostMemory.Free((void*)hole, 0, HostMemory.MEM_RELEASE));
        var imageBase = memory.AllocateAt(hole, imageSize);
        var arena = GuestRedZonePatcher.ReserveSpinNapArena(memory, imageBase, imageSize);
        Assert.NotNull(arena);
        Assert.InRange(arena.Value.Address, imageBase + imageSize, imageBase + holeSize - arena.Value.Size);
        Assert.Equal(GuestRedZonePatcher.SpinNapArenaBytes, arena.Value.Size);
    }

    [Fact]
    public void LeavesAPauseThatDoesNotCloseALoopAlone()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // mov eax, 7 ; pause ; ret  (no backward branch)
        byte[] function = [0xB8, 0x07, 0x00, 0x00, 0x00, 0xF3, 0x90, 0xC3];

        using var image = SpinImage.Create(function);
        Assert.Equal(1, image.Result.PauseSites);
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(1, image.Result.NonLoopSites);
        Assert.Equal(0xF3, image.ReadByte(5));
        Assert.Equal(7u, ((delegate* unmanaged<uint>)image.Base)());
    }

    // Data cell the locked-loop tests' RIP-relative operands point at (inside the image, past the code).
    private const int DataOffset = 0x800;

    private static void Rel32(byte[] code, int at, int nextIp) =>
        BinaryPrimitives.WriteInt32LittleEndian(code.AsSpan(at, 4), DataOffset - nextIp);

    // mov ecx,4 ; loop: mov eax,3 ; lock xadd [rip+data],eax ; dec ecx ; jnz loop ; mov eax,[rip+data] ; ret
    // Returns 12 (4 x 3) only if the relocated RIP-relative lock xadd still hits the right cell.
    private static byte[] LockedXaddLoop()
    {
        byte[] function =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00,       // 0: mov ecx, 4
            0xB8, 0x03, 0x00, 0x00, 0x00,       // 5: loop: mov eax, 3
            0xF0, 0x0F, 0xC1, 0x05, 0, 0, 0, 0, // 10: lock xadd [rip+data], eax (next ip 18)
            0xFF, 0xC9,                         // 18: dec ecx
            0x75, 0xEF,                         // 20: jnz loop (22 - 17 = 5)
            0x8B, 0x05, 0, 0, 0, 0,             // 22: mov eax, [rip+data] (next ip 28)
            0xC3,                               // 28: ret
        ];
        Rel32(function, 14, 18);
        Rel32(function, 24, 28);
        return function;
    }

    [Fact]
    public void LockedXaddLoopIsRecognizedRelocatedAndStaysTransparent()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var image = SpinImage.Create(LockedXaddLoop(), locked: true);
        Assert.Equal(1, image.Result.LockedSites);
        Assert.Equal(0, image.Result.LockedNonLoopSites);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0, image.Result.FailedSites);
        Assert.Equal(0xE9, image.ReadByte(10)); // the lock xadd is now a jmp into the trampoline

        // The trampoline's call reaches the locked probe (the second probe in the arena).
        var trampoline = (ulong)((long)image.Base + 10 + 5 + BinaryPrimitives.ReadInt32LittleEndian(image.Read(11, 4)));
        var lead = image.ReadAbsolute(trampoline, 21);
        var probe = (ulong)((long)trampoline + 8 + 5 + BinaryPrimitives.ReadInt32LittleEndian(lead.AsSpan(9, 4)));
        var lockedProbe = SpinNapRuntime.EnsureLockedProbeBytes(1_000_000_000);
        Assert.Equal(lockedProbe, image.ReadAbsolute(probe, lockedProbe.Length));

        Assert.Equal(12u, ((delegate* unmanaged<uint>)image.Base)());
    }

    // The probe runs in the middle of arbitrary guest code, not at a call boundary, so a nap must
    // hand back every vector register too. A nap that called into the host without saving them
    // returned to guest code with XMM0-15 holding whatever the host left there.
    [Fact]
    public void NappingLockedProbePreservesVectorRegisters()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // mov rax,C ; movq xmm0,rax ; movq xmm15,rax ; mov ecx,4
        // loop: mov eax,3 ; lock xadd [rip+data],eax ; dec ecx ; jnz loop
        // movq rax,xmm0 ; movq rdx,xmm15 ; add rax,rdx ; ret
        byte[] function =
        [
            0x48, 0xB8, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // 0: mov rax, 0x0102030405060708
            0x66, 0x48, 0x0F, 0x6E, 0xC0,       // 10: movq xmm0, rax
            0x66, 0x4C, 0x0F, 0x6E, 0xF8,       // 15: movq xmm15, rax
            0xB9, 0x04, 0x00, 0x00, 0x00,       // 20: mov ecx, 4
            0xB8, 0x03, 0x00, 0x00, 0x00,       // 25: loop: mov eax, 3
            0xF0, 0x0F, 0xC1, 0x05, 0, 0, 0, 0, // 30: lock xadd [rip+data], eax (next ip 38)
            0xFF, 0xC9,                         // 38: dec ecx
            0x75, 0xEF,                         // 40: jnz loop (42 - 17 = 25)
            0x66, 0x48, 0x0F, 0x7E, 0xC0,       // 42: movq rax, xmm0
            0x66, 0x4C, 0x0F, 0x7E, 0xFA,       // 47: movq rdx, xmm15
            0x48, 0x01, 0xD0,                   // 52: add rax, rdx
            0xC3,                               // 55: ret
        ];
        Rel32(function, 34, 38);

        using var image = SpinImage.Create(
            function, locked: true, lockedProbe: SpinNapRuntime.BuildNappingLockedProbeForTest(1_000_000_000));
        Assert.Equal(1, image.Result.LockedSites);
        Assert.Equal(0xE9, image.ReadByte(30));

        var napsBefore = SpinNapRuntime.LockedNapCount;
        Assert.Equal(0x020406080A0C0E10UL, ((delegate* unmanaged<ulong>)image.Base)());
        // Four hits: the first opens the window, each later one naps.
        Assert.Equal(3, SpinNapRuntime.LockedNapCount - napsBefore);
    }

    [Fact]
    public void LockedOpsAreLeftAloneWithoutTheLockedProbe()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var image = SpinImage.Create(LockedXaddLoop(), locked: false);
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(0xF0, image.ReadByte(10));
    }

    [Fact]
    public void XchgMemoryLoopIsRecognized()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // mov ecx,4 ; loop: mov eax,3 ; xchg [rip+data],eax ; dec ecx ; jnz loop ; mov eax,[rip+data] ; ret
        // The cell ends at 3 (each pass swaps 3 in), proving the relocated xchg still runs.
        byte[] function =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00,       // 0: mov ecx, 4
            0xB8, 0x03, 0x00, 0x00, 0x00,       // 5: loop: mov eax, 3
            0x87, 0x05, 0, 0, 0, 0,             // 10: xchg [rip+data], eax (6 bytes, next ip 16)
            0xFF, 0xC9,                         // 16: dec ecx
            0x75, 0xF1,                         // 18: jnz loop (20 - 15 = 5)
            0x8B, 0x05, 0, 0, 0, 0,             // 20: mov eax, [rip+data] (next ip 26)
            0xC3,                               // 26: ret
        ];
        Rel32(function, 12, 16);
        Rel32(function, 22, 26);
        using var image = SpinImage.Create(function, locked: true);
        Assert.Equal(1, image.Result.LockedSites);
        Assert.Equal(0xE9, image.ReadByte(10));
        Assert.Equal(3u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void LockedLoopContainingACallIsNotRecognized()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // loop: call helper ; lock xadd ; dec ; jnz loop ; ... ; helper: ret
        byte[] function =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00,       // 0: mov ecx, 4
            0xE8, 0x17, 0x00, 0x00, 0x00,       // 5: loop: call helper (next 10, +0x17 -> 33)
            0xB8, 0x03, 0x00, 0x00, 0x00,       // 10: mov eax, 3
            0xF0, 0x0F, 0xC1, 0x05, 0, 0, 0, 0, // 15: lock xadd [rip+data], eax (next ip 23)
            0xFF, 0xC9,                         // 23: dec ecx
            0x75, 0xEA,                         // 25: jnz loop (27 - 22 = 5)
            0x8B, 0x05, 0, 0, 0, 0,             // 27: mov eax, [rip+data]
            0xC3,                               // 33: ret (also the helper)
        ];
        Rel32(function, 19, 23);
        Rel32(function, 29, 33);

        using var image = SpinImage.Create(function, locked: true);
        Assert.Equal(0, image.Result.LockedSites);
        Assert.Equal(1, image.Result.LockedNonLoopSites);
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(0xF0, image.ReadByte(15));
        Assert.Equal(12u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void LockedOpOutsideAnyLoopOrInALongLoopIsNotRecognized()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // lock xadd with no backward branch at all.
        byte[] straight =
        [
            0xB8, 0x03, 0x00, 0x00, 0x00,       // 0: mov eax, 3
            0xF0, 0x0F, 0xC1, 0x05, 0, 0, 0, 0, // 5: lock xadd [rip+data], eax (next ip 13)
            0xC3,                               // 13: ret
        ];
        Rel32(straight, 9, 13);
        using (var image = SpinImage.Create(straight, locked: true))
        {
            Assert.Equal(0, image.Result.LockedSites);
            Assert.Equal(1, image.Result.LockedNonLoopSites);
        }

        // The same loop with a body padded past 256 bytes with nops.
        var padded = new List<byte>
        {
            0xB9, 0x04, 0x00, 0x00, 0x00,       // 0: mov ecx, 4
            0xB8, 0x03, 0x00, 0x00, 0x00,       // 5: loop: mov eax, 3
            0xF0, 0x0F, 0xC1, 0x05, 0, 0, 0, 0, // 10: lock xadd
        };
        padded.AddRange(Enumerable.Repeat((byte)0x90, 300));
        padded.AddRange([0xFF, 0xC9]);          // dec ecx
        padded.AddRange([0x0F, 0x85, 0, 0, 0, 0]); // jnz loop (rel32)
        var jnzNext = padded.Count;
        BinaryPrimitives.WriteInt32LittleEndian(CollectionsMarshal.AsSpan(padded)[(jnzNext - 4)..], 5 - jnzNext);
        padded.Add(0xC3);
        var paddedBytes = padded.ToArray();
        Rel32(paddedBytes, 14, 18);
        using (var image = SpinImage.Create(paddedBytes, locked: true))
        {
            Assert.Equal(0, image.Result.LockedSites);
            Assert.Equal(1, image.Result.LockedNonLoopSites);
        }
    }

    // do { dec ecx ; jz exit ; pause ; jmp short loop ; <alignment nop> } : the compiled shape of `while (!flag) pause;`.
    // The jump is 4 bytes with the pause, one short of the 5-byte patch jump, so the pass takes the nop after it.
    private static byte[] PauseJmpLoop() =>
    [
        0xB9, 0x04, 0x00, 0x00, 0x00, // 0: mov ecx, 4
        0xFF, 0xC9,                   // 5: loop: dec ecx
        0x74, 0x07,                   // 7: jz exit (next ip 9, +7 -> 16)
        0xF3, 0x90,                   // 9: pause
        0xEB, 0xF8,                   // 11: jmp short loop (next ip 13, -8 -> 5)
        0x0F, 0x1F, 0x00,             // 13: nop dword [rax] (alignment padding)
        0xB8, 0x44, 0x33, 0x22, 0x11, // 16: exit: mov eax, 0x11223344
        0xC3,                         // 21: ret
    ];

    [Fact]
    public void PatchesAPauseJmpLoopByTakingTheAlignmentNopsAfterTheJump()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var image = SpinImage.Create(PauseJmpLoop());
        Assert.Equal(1, image.Result.PauseSites);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0, image.Result.UnrelocatableSites);
        Assert.Equal(0xE9, image.ReadByte(9)); // the pause is now a jmp into the trampoline
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void LeavesAPauseJmpLoopAloneWhenTheShapeSwitchIsOff()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var _ = EnvironmentVariableScope.Set(SpinNapRuntime.WideShapesEnvironmentVariable, "0");
        using var image = SpinImage.Create(PauseJmpLoop());
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(1, image.Result.UnrelocatableSites);
        Assert.Equal(0xF3, image.ReadByte(9));
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void DoesNotTakeABranchTargetAsLoopPadding()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // The same loop, but the exit branch lands on the nop that follows the jump: it is live code, not padding.
        byte[] function =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00, // 0: mov ecx, 4
            0xFF, 0xC9,                   // 5: loop: dec ecx
            0x74, 0x04,                   // 7: jz 13 (next ip 9, +4)
            0xF3, 0x90,                   // 9: pause
            0xEB, 0xF8,                   // 11: jmp short loop
            0x0F, 0x1F, 0x00,             // 13: nop dword [rax] (a branch target)
            0xB8, 0x44, 0x33, 0x22, 0x11, // 16: mov eax, 0x11223344
            0xC3,                         // 21: ret
        ];

        using var image = SpinImage.Create(function);
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(0xF3, image.ReadByte(9));
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    private const int HoistedTableOffset = 0x2000;
    private const int HoistedPauseOffset = 40;

    // switch (kind) inside a spin loop, as clang -O2 compiles it: the table base is hoisted out of the loop into
    // a callee-saved register, and the bounds check is a 64-bit compare of the zero-extended selector. The
    // dispatch is only recognised by the wide-shape rule. `clobberAfterEpiloguePop` adds a second write of r12
    // (a mov after the pop), which must make the pass distrust the hoisted base.
    private static (byte[] Code, byte[] Table) HoistedSwitchSpinLoop(bool clobberAfterEpiloguePop = false)
    {
        var code = new List<byte>();
        code.AddRange([0x41, 0x54]);                                // push r12
        var tableLoad = code.Count;
        code.AddRange([0x4C, 0x8D, 0x25, 0, 0, 0, 0]);              // lea r12, [rip + table]
        code.AddRange([0xB9, 0x03, 0x00, 0x00, 0x00]);              // mov ecx, 3
        var loop = code.Count;
        code.AddRange([0xB8, 0x01, 0x00, 0x00, 0x00]);              // loop: mov eax, 1   (the selector)
        code.AddRange([0x48, 0x83, 0xF8, 0x01]);                    // cmp rax, 1
        var boundsBranch = code.Count;
        code.AddRange([0x77, 0x00]);                                // ja exit
        code.AddRange([0x49, 0x63, 0x04, 0x84]);                    // movsxd rax, [r12 + rax*4]
        code.AddRange([0x4C, 0x01, 0xE0]);                          // add rax, r12
        code.AddRange([0xFF, 0xE0]);                                // jmp rax
        var case0 = code.Count;
        var case0Jump = code.Count;
        code.AddRange([0xEB, 0x00]);                                // case 0: jmp short exit
        var case1 = code.Count;
        code.AddRange([0xFF, 0xC9]);                                // case 1: dec ecx
        var exitBranch = code.Count;
        code.AddRange([0x74, 0x00]);                                // jz exit
        var pause = code.Count;
        code.AddRange([0xF3, 0x90]);                                // pause
        code.AddRange([0xEB, unchecked((byte)(loop - (code.Count + 2)))]);     // jmp short loop
        code.AddRange([0x0F, 0x1F, 0x00]);                          // nop dword [rax]
        var exit = code.Count;
        code.AddRange([0xB8, 0x44, 0x33, 0x22, 0x11]);              // exit: mov eax, 0x11223344
        code.AddRange([0x41, 0x5C]);                                // pop r12 (epilogue)
        if (clobberAfterEpiloguePop)
        {
            code.AddRange([0x45, 0x31, 0xE4]);                      // xor r12d, r12d
        }

        code.Add(0xC3);                                             // ret
        code[boundsBranch + 1] = (byte)(exit - (boundsBranch + 2));
        code[case0Jump + 1] = (byte)(exit - (case0Jump + 2));
        code[exitBranch + 1] = (byte)(exit - (exitBranch + 2));
        var bytes = code.ToArray();
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(tableLoad + 3), HoistedTableOffset - (tableLoad + 7));
        Assert.Equal(HoistedPauseOffset, pause);
        var table = new byte[2 * sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(table.AsSpan(0), case0 - HoistedTableOffset);
        BinaryPrimitives.WriteInt32LittleEndian(table.AsSpan(4), case1 - HoistedTableOffset);
        return (bytes, table);
    }

    [Fact]
    public void PatchesASpinLoopInAFunctionWithAHoistedJumpTableBase()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var (code, table) = HoistedSwitchSpinLoop();
        using var image = SpinImage.Create(code, readOnlyData: (HoistedTableOffset, table));
        Assert.Equal(1, image.Result.PauseSites);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0xE9, image.ReadByte(HoistedPauseOffset));
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void LeavesTheSpinLoopAloneWhenTheHoistedTableRegisterIsWrittenAgain()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        var (code, table) = HoistedSwitchSpinLoop(clobberAfterEpiloguePop: true);
        using var image = SpinImage.Create(code, readOnlyData: (HoistedTableOffset, table));
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(1, image.Result.UnrelocatableSites);
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void LeavesTheSpinLoopAloneWhenTheShapeSwitchIsOff()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var _ = EnvironmentVariableScope.Set(SpinNapRuntime.WideShapesEnvironmentVariable, "0");
        var (code, table) = HoistedSwitchSpinLoop();
        using var image = SpinImage.Create(code, readOnlyData: (HoistedTableOffset, table));
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    private static void RunOnNewThread(Action action)
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try
            {
                action();
            }
            catch (Exception exception)
            {
                failure = exception;
            }
        });
        thread.Start();
        thread.Join();
        if (failure is not null)
        {
            throw new Xunit.Sdk.XunitException($"thread failed: {failure}");
        }
    }

    // Threads that stay alive until disposed, so their thread pointers (the private keys) cannot be
    // reused by a later thread, as a thread library's cache of thread stacks would.
    private sealed class WorkerThreads : IDisposable
    {
        private readonly List<WorkerThread> _threads = [];

        // Two live threads whose private slots differ (a thread pointer hashing onto another live
        // thread's slot is the rare collision the table tolerates; these tests want two slots).
        public (WorkerThread A, WorkerThread B) TwoWithDistinctSlots(Func<ulong, nint> slotOf)
        {
            var a = Start();
            for (var attempt = 0; attempt < 16; attempt++)
            {
                var b = Start();
                if (b.Key != a.Key && slotOf(b.Key) != slotOf(a.Key))
                {
                    return (a, b);
                }
            }

            throw new Xunit.Sdk.XunitException("no two live threads with distinct slots");
        }

        private WorkerThread Start()
        {
            var thread = new WorkerThread();
            _threads.Add(thread);
            return thread;
        }

        public void Dispose()
        {
            foreach (var thread in _threads)
            {
                thread.Dispose();
            }
        }
    }

    private sealed class WorkerThread : IDisposable
    {
        private readonly System.Collections.Concurrent.BlockingCollection<(Action Work, ManualResetEventSlim Done)> _work = new();
        private readonly Thread _thread;
        private Exception? _failure;

        public WorkerThread()
        {
            _thread = new Thread(() =>
            {
                foreach (var (work, done) in _work.GetConsumingEnumerable())
                {
                    try
                    {
                        work();
                    }
                    catch (Exception exception)
                    {
                        _failure = exception;
                    }

                    done.Set();
                }
            })
            { IsBackground = true };
            _thread.Start();
            ulong key = 0;
            Run(() => key = SpinNapRuntime.CurrentThreadKey());
            Key = key;
        }

        public ulong Key { get; }

        public void Run(Action work)
        {
            using var done = new ManualResetEventSlim();
            _work.Add((work, done));
            done.Wait();
            if (_failure is { } failure)
            {
                _failure = null;
                throw new Xunit.Sdk.XunitException($"worker thread failed: {failure}");
            }
        }

        public void Dispose()
        {
            _work.CompleteAdding();
            _thread.Join();
            _work.Dispose();
        }
    }

    // Calls `target` with rsp switched to `stackTop` (16-byte aligned), so a test picks the guest stack
    // a probe sees: mov rax, rsp ; mov rsp, top ; push rax ; push rax ; call target ; pop rax ; pop rsp.
    private static void CallOnStack(nint target, ulong stackTop)
    {
        var call = (delegate* unmanaged<ulong, nint, void>)CallOnStackCode.Value.Address;
        call(stackTop, target);
    }

    private static readonly Lazy<ExecutableCode> CallOnStackCode = new(() =>
    {
        var a = new Assembler(64);
        var top = OperatingSystem.IsWindows() ? rcx : rdi;
        var target = OperatingSystem.IsWindows() ? rdx : rsi;
        a.mov(rax, rsp);
        a.mov(rsp, top);
        a.push(rax);
        a.push(rax);
        a.call(target);
        a.pop(rax);
        a.pop(rsp);
        a.ret();
        return new ExecutableCode(Assemble(a));
    });

    private static byte[] Assemble(Assembler a)
    {
        using var stream = new MemoryStream();
        a.Assemble(new StreamCodeWriter(stream), 0);
        return stream.ToArray();
    }

    // Two 16-byte aligned stack tops 16 KiB apart (two neighbouring fiber stacks) in one 64 MiB bucket,
    // each with 256 KiB below it; the probe sees rsp = top - 152 (two pushes, the call, 16 saves).
    private sealed class TwoStacks : IDisposable
    {
        private const int Bytes = 1 << 20;
        private readonly byte* _block;

        public TwoStacks()
        {
            _block = (byte*)NativeMemory.Alloc(Bytes);
            var top = ((ulong)_block + Bytes - 64) & ~15UL;
            if ((top - 152) >> SpinNapRuntime.StackKeyShift != (top - 0x4000 - 152) >> SpinNapRuntime.StackKeyShift)
            {
                top -= 0x40000; // the 64 MiB boundary sits between the two tops; move both below it
            }

            TopA = top;
            TopB = top - 0x4000;
            Assert.True(TopB - 0x40000 >= (ulong)_block);
        }

        public ulong TopA { get; }

        public ulong TopB { get; }

        public ulong SharedKeyA => (TopA - 152) >> SpinNapRuntime.StackKeyShift;

        public ulong SharedKeyB => (TopB - 152) >> SpinNapRuntime.StackKeyShift;

        public void Dispose() => NativeMemory.Free(_block);
    }

    private sealed class ExecutableCode : IDisposable
    {
        public ExecutableCode(byte[] code)
        {
            Assert.True(code.Length <= 4096);
            var page = (byte*)HostMemory.Alloc(
                null, 4096, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
            Assert.True(page != null);
            code.CopyTo(new Span<byte>(page, code.Length));
            Address = (nint)page;
        }

        public nint Address { get; }

        public void Dispose() => _ = HostMemory.Free((void*)Address, 0, HostMemory.MEM_RELEASE);
    }

    // Loads a distinct value into each of the 15 general registers other than rsp and the given flags,
    // calls the probe, and returns what came back. Block: [0] probe, [1] flags in, [2..17) registers in
    // (rax rcx rdx rbx rbp rsi rdi r8-r15), [17] flags out, [18..33) registers out.
    private static class RegisterHarness
    {
        // CF PF AF ZF SF DF OF
        public const ulong AllTestedFlags = 0x0001 | 0x0004 | 0x0010 | 0x0040 | 0x0080 | 0x0400 | 0x0800;

        private static readonly AssemblerRegister64[] Registers =
            [rax, rcx, rdx, rbx, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15];

        public static readonly ulong[] Inputs =
            [.. Enumerable.Range(0, 15).Select(index => 0xA5A5_0000_0000_0000UL | ((ulong)(index + 1) * 0x0001_0203_0405_0607UL & 0xFFFF_FFFF_FFFF))];

        private static readonly Lazy<ExecutableCode> Code = new(() =>
        {
            var a = new Assembler(64);
            AssemblerRegister64[] saved = [rbx, rbp, r12, r13, r14, r15, rdi, rsi];
            foreach (var register in saved)
            {
                a.push(register);
            }

            a.mov(rax, OperatingSystem.IsWindows() ? rcx : rdi);
            a.push(rax);                         // the block, for after the call
            a.push(__qword_ptr[rax]);            // the probe, called through [rsp]
            a.push(__qword_ptr[rax + 8]);
            a.popfq();
            for (var index = 1; index < Registers.Length; index++)
            {
                a.mov(Registers[index], __qword_ptr[rax + 16 + index * 8]);
            }

            a.mov(rax, __qword_ptr[rax + 16]);
            a.call(__qword_ptr[rsp]);
            a.lea(rsp, __[rsp + 8]);             // drop the probe address without touching the flags
            a.pushfq();
            a.push(rax);
            a.mov(rax, __qword_ptr[rsp + 16]);
            for (var index = 1; index < Registers.Length; index++)
            {
                a.mov(__qword_ptr[rax + 144 + index * 8], Registers[index]);
            }

            a.pop(rcx);
            a.mov(__qword_ptr[rax + 144], rcx);  // rax out
            a.pop(rcx);
            a.mov(__qword_ptr[rax + 136], rcx);  // flags out
            a.pop(rcx);                          // the block
            a.cld();
            for (var index = saved.Length - 1; index >= 0; index--)
            {
                a.pop(saved[index]);
            }

            a.ret();
            return new ExecutableCode(Assemble(a));
        });

        public static (ulong[] Registers, ulong Flags) Call(nint probe, ulong flags)
        {
            var block = new ulong[33];
            block[0] = (ulong)probe;
            block[1] = 0x2 | flags; // bit 1 is always set
            Inputs.CopyTo(block, 2);
            fixed (ulong* pointer = block)
            {
                ((delegate* unmanaged<ulong*, void>)Code.Value.Address)(pointer);
            }

            return (block[18..33], block[17]);
        }
    }

    private sealed class EnvironmentVariableScope : IDisposable
    {
        private readonly string _name;
        private readonly string? _previous;

        private EnvironmentVariableScope(string name, string? previous)
        {
            _name = name;
            _previous = previous;
        }

        public static EnvironmentVariableScope Set(string name, string value)
        {
            var scope = new EnvironmentVariableScope(name, Environment.GetEnvironmentVariable(name));
            Environment.SetEnvironmentVariable(name, value);
            return scope;
        }

        public void Dispose() => Environment.SetEnvironmentVariable(_name, _previous);
    }

    private sealed class SpinImage : IDisposable
    {
        private const ulong ImageSize = 0x10000;
        private readonly PhysicalVirtualMemory _memory;

        private SpinImage(PhysicalVirtualMemory memory, ulong imageBase, GuestRedZonePatcher.SpinPatchResult result,
            GuestRedZonePatcher.SpinNapArena? arena = null)
        {
            _memory = memory;
            Base = imageBase;
            Result = result;
            Arena = arena;
        }

        public ulong Base { get; }

        public GuestRedZonePatcher.SpinNapArena? Arena { get; }

        public GuestRedZonePatcher.SpinPatchResult Result { get; }

        public static SpinImage Create(byte[] function, bool locked = false, byte[]? lockedProbe = null,
            ulong reserveAt = 0, ulong reservedSize = 0, (ulong Offset, byte[] Bytes)? readOnlyData = null,
            byte[]? probe = null)
        {
            var memory = new PhysicalVirtualMemory();
            const ulong holeSize = 0x4000000;
            var hole = (ulong)HostMemory.Alloc(null, (nuint)holeSize, HostMemory.MEM_RESERVE, HostMemory.PAGE_NOACCESS);
            Assert.NotEqual(0UL, hole);
            Assert.True(HostMemory.Free((void*)hole, 0, HostMemory.MEM_RELEASE));
            var imageBase = memory.AllocateAt(hole, ImageSize);
            Assert.True(memory.TryWrite(imageBase, function));

            var exceptionFrameHeader = new byte[32];
            exceptionFrameHeader[0] = 1;
            exceptionFrameHeader[2] = 3;
            BinaryPrimitives.WriteUInt32LittleEndian(exceptionFrameHeader.AsSpan(12), 1);
            BinaryPrimitives.WriteUInt64LittleEndian(exceptionFrameHeader.AsSpan(16), imageBase);
            Assert.True(memory.TryWrite(imageBase + 0x1000, exceptionFrameHeader));
            var headers = new List<ProgramHeader>
            {
                CreateProgramHeader(ProgramHeaderType.Load, ProgramHeaderFlags.Read | ProgramHeaderFlags.Execute,
                    0, (ulong)function.Length),
                CreateProgramHeader(ProgramHeaderType.GnuEhFrame, ProgramHeaderFlags.Read, 0x1000, (ulong)exceptionFrameHeader.Length),
            };
            if (readOnlyData is { } table)
            {
                Assert.True(memory.TryWrite(imageBase + table.Offset, table.Bytes));
                headers.Add(CreateProgramHeader(ProgramHeaderType.Load, ProgramHeaderFlags.Read, table.Offset, (ulong)table.Bytes.Length));
            }

            GuestRedZonePatcher.SpinNapArena? arena = null;
            if (reservedSize != 0)
            {
                Assert.True(memory.TryAllocateWithinRange(imageBase + reserveAt, imageBase + holeSize, reservedSize, out var arenaAddress));
                arena = new GuestRedZonePatcher.SpinNapArena(arenaAddress, reservedSize);
            }

            probe ??= SpinNapRuntime.EnsureProbeBytes(1_000_000_000);
            lockedProbe ??= locked ? SpinNapRuntime.EnsureLockedProbeBytes(1_000_000_000) : [];
            var result = GuestRedZonePatcher.PatchSpinLoops(memory, memory, headers, imageBase, ImageSize, probe, "test", lockedProbe, arena);
            return new SpinImage(memory, imageBase, result, arena);
        }

        public byte ReadByte(ulong offset) => Read(offset, 1)[0];

        public byte[] Read(ulong offset, int length)
        {
            var value = new byte[length];
            Assert.True(_memory.TryRead(Base + offset, value));
            return value;
        }

        public byte[] ReadAbsolute(ulong address, int length)
        {
            var value = new byte[length];
            Assert.True(_memory.TryRead(address, value));
            return value;
        }

        public void Dispose() => _memory.Dispose();
    }

    private static ProgramHeader CreateProgramHeader(
        ProgramHeaderType type, ProgramHeaderFlags flags, ulong address, ulong size)
    {
        Span<byte> bytes = stackalloc byte[56];
        bytes.Clear();
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, (uint)type);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes[4..], (uint)flags);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes[16..], address);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes[32..], size);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes[40..], size);
        return MemoryMarshal.Read<ProgramHeader>(bytes);
    }
}
