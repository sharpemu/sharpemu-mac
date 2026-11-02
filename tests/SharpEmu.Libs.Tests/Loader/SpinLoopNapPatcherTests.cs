// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpEmu.Core.Loader;
using SharpEmu.Core.Memory;
using SharpEmu.HLE;
using Xunit;

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

    private static byte[] FixedProbe(long delayTicks, ulong statsAddress) =>
        SpinNapRuntime.BuildProbeForTest(
            0x0000600000000000UL, 500_000, 50_000, 0x1122334455667788UL, 250, 0x0000700000001000UL, delayTicks, statsAddress);

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

    [Fact]
    public void WithoutAPauseDelayTheProbeIsByteForByteUnchanged()
    {
        Assert.Equal(ProbeBeforePauseDelay, Convert.ToHexString(FixedProbe(0, 0)));
    }

    [Fact]
    public void OnlyADelayAddsAnRdtsc()
    {
        Assert.Equal(1, CountRdtsc(FixedProbe(0, 0)));
        Assert.Equal(1, CountRdtsc(FixedProbe(0, 0x0000710000000000UL))); // counters alone: no second clock read
        Assert.Equal(2, CountRdtsc(FixedProbe(19, 0)));
        Assert.Equal(2, CountRdtsc(FixedProbe(19, 0x0000710000000000UL)));
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
