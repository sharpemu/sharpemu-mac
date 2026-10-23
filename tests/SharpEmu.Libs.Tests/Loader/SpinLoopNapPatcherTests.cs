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

    // The rdtsc wrapper DS's WaitForJobs calls from its idle loop (app+0x863a20): rdtsc; shl rdx,32; or rax,rdx; ret.
    private static readonly byte[] RdtscWrapper = [0x0F, 0x31, 0x48, 0xC1, 0xE2, 0x20, 0x48, 0x09, 0xD0, 0xC3];

    // mov ecx,4 ; loop: pause ; mov rax,rcx ; call callee ; dec ecx ; jnz loop ; mov eax,0x11223344 ; ret ; (pad) ; callee at 0x20
    private static byte[] LoopCallingCallee(byte[] callee)
    {
        var function = new byte[0x20 + callee.Length];
        byte[] head =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00, // 0: mov ecx, 4
            0xF3, 0x90,                   // 5: loop: pause
            0x48, 0x89, 0xC8,             // 7: mov rax, rcx (the pause needs 3 relocatable bytes after it)
            0xE8, 0x11, 0x00, 0x00, 0x00, // 10: call 0x20 (next ip 15)
            0xFF, 0xC9,                   // 15: dec ecx
            0x75, 0xF2,                   // 17: jnz loop (19 - 14 = 5)
            0xB8, 0x44, 0x33, 0x22, 0x11, // 19: mov eax, 0x11223344
            0xC3,                         // 24: ret
        ];
        head.CopyTo(function, 0);
        Array.Fill(function, (byte)0xCC, head.Length, 0x20 - head.Length);
        callee.CopyTo(function, 0x20);
        return function;
    }

    [Fact]
    public void PatchesAPauseLoopThatCallsASmallLeafAndStaysTransparent()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var image = SpinImage.Create(LoopCallingCallee(RdtscWrapper));
        Assert.Equal(1, image.Result.PauseSites);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0, image.Result.NonLoopSites);
        Assert.Equal(0xE9, image.ReadByte(5)); // the pause is now a jmp into the trampoline
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Theory]
    [InlineData("call", new byte[] { 0xE8, 0x00, 0x00, 0x00, 0x00, 0xC3 })]
    [InlineData("indirect jump", new byte[] { 0xFF, 0xE0, 0xC3 })]
    [InlineData("indirect call", new byte[] { 0xFF, 0xD0, 0xC3 })]
    [InlineData("syscall", new byte[] { 0x0F, 0x05, 0xC3 })]
    [InlineData("no return", new byte[] { 0x90, 0x90, 0x90 })]
    public void LeavesAPauseLoopThatCallsANonLeafAlone(string reason, byte[] callee)
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var image = SpinImage.Create(LoopCallingCallee(callee));
        Assert.True(image.Result.NonLoopSites == 1, reason);
        Assert.Equal(0, image.Result.PatchedSites);
        Assert.Equal(0xF3, image.ReadByte(5));
    }

    [Theory]
    [InlineData(true, new byte[] { 0x0F, 0x31, 0x48, 0xC1, 0xE2, 0x20, 0x48, 0x09, 0xD0, 0xC3 })]
    // push rbp ; mov rbp,rsp ; sub rsp,0x20 ; rdtsc ; leave ; ret
    [InlineData(true, new byte[] { 0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x20, 0x0F, 0x31, 0xC9, 0xC3 })]
    // cmp edi,0 ; je +2 ; nop ; xor eax,eax ; ret  (forward branch inside the callee)
    [InlineData(true, new byte[] { 0x83, 0xFF, 0x00, 0x74, 0x01, 0x90, 0x31, 0xC0, 0xC3 })]
    // sub rsp,0x1000 ; ret  (a stack adjustment beyond a leaf frame)
    [InlineData(false, new byte[] { 0x48, 0x81, 0xEC, 0x00, 0x10, 0x00, 0x00, 0xC3 })]
    // dec ecx ; jnz -4 ; ret  (a backward branch: the callee is a loop of its own)
    [InlineData(false, new byte[] { 0xFF, 0xC9, 0x75, 0xFC, 0xC3 })]
    // jmp +0x40 (tail call out of the function)
    [InlineData(false, new byte[] { 0xEB, 0x40, 0xC3 })]
    public void SmallLeafFunctionRule(bool expected, byte[] callee)
    {
        var padded = new byte[0x80];
        Array.Fill(padded, (byte)0xCC);
        callee.CopyTo(padded, 0x10);
        Assert.Equal(expected, GuestRedZonePatcher.IsSmallLeafFunction(padded, 0x1000, 0x1010));
        Assert.False(GuestRedZonePatcher.IsSmallLeafFunction(padded, 0x1000, 0x2000));
    }

    [Fact]
    public void PatchesAPauseWhoseBackEdgeIsOutOfLineFarFromIt()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // The shape of DS's WaitForJobs idle loop: the path from the pause back to itself runs through a
        // forward branch and a far out-of-line jmp, so no backward branch sits within 96 bytes of the pause.
        //   mov ecx,4 ; loop: pause ; dec ecx ; jnz far ; mov eax,0x11223344 ; ret ; ... ; far: jmp loop
        var function = new byte[0x80];
        Array.Fill(function, (byte)0xCC);
        byte[] head =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00,             // 0: mov ecx, 4
            0xF3, 0x90,                               // 5: loop: pause
            0xFF, 0xC9,                               // 7: dec ecx
            0x0F, 0x85, 0x61, 0x00, 0x00, 0x00,       // 9: jnz 0x70 (next ip 15)
            0xB8, 0x44, 0x33, 0x22, 0x11,             // 15: mov eax, 0x11223344
            0xC3,                                     // 20: ret
        ];
        head.CopyTo(function, 0);
        function[0x70] = 0xEB; // jmp loop (next ip 0x72, 5 - 0x72 = -0x6D)
        function[0x71] = 0x93;

        using var image = SpinImage.Create(function);
        Assert.Equal(1, image.Result.PauseSites);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0, image.Result.NonLoopSites);
        Assert.Equal(0x11223344u, ((delegate* unmanaged<uint>)image.Base)());
    }

    [Fact]
    public void PatchesTheShapeOfDsIdleWaitLoopAndStaysTransparent()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        // WaitForJobs' idle loop (Demon's Souls app+0x86eb39): a pause, then a memory load, a call to the rdtsc
        // wrapper, a compare against the load and a conditional exit; the back edge is an out-of-line jmp
        // reached through forward branches only, more than 96 bytes from the pause.
        var function = new byte[0x80];
        Array.Fill(function, (byte)0xCC);
        byte[] head =
        [
            0xB9, 0x04, 0x00, 0x00, 0x00,             // 0:  mov ecx, 4
            0xF3, 0x90,                               // 5:  loop: pause
            0x48, 0x89, 0xC8,                         // 7:  mov rax, rcx
            0x4C, 0x8B, 0x07,                         // 10: mov r8, [rdi]   (the polled cell: 0)
            0xE8, 0x16, 0x00, 0x00, 0x00,             // 13: call rdtsc (next ip 18 + 0x16 = 0x28)
            0x4C, 0x29, 0xC0,                         // 18: sub rax, r8
            0x48, 0x85, 0xC0,                         // 21: test rax, rax
            0x0F, 0x8E, 0x2E, 0x00, 0x00, 0x00,       // 24: jle done (next ip 30 + 0x2e = 0x4c)
            0xFF, 0xC9,                               // 30: dec ecx
            0x74, 0x2A,                               // 32: jz done (next ip 34 + 0x2a = 0x4c)
            0xE9, 0x39, 0x00, 0x00, 0x00,             // 34: jmp far (next ip 39 + 0x39 = 0x60)
        ];
        head.CopyTo(function, 0);
        RdtscWrapper.CopyTo(function, 0x28);
        // done at 0x4c: mov eax, 0x11223344 ; ret
        new byte[] { 0xB8, 0x44, 0x33, 0x22, 0x11, 0xC3 }.CopyTo(function, 0x4c);
        // far at 0x60: jmp loop (next ip 0x65 ; 5 - 0x65 = -0x60)
        new byte[] { 0xE9, 0xA0, 0xFF, 0xFF, 0xFF }.CopyTo(function, 0x60);

        var cell = 0UL;
        using var image = SpinImage.Create(function);
        Assert.Equal(1, image.Result.PauseSites);
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0, image.Result.NonLoopSites);
        Assert.Equal(0xE9, image.ReadByte(5));
        Assert.Equal(0x11223344u, ((delegate* unmanaged<ulong*, uint>)image.Base)(&cell));
    }

    // pause ; mov al,[rdi] ; test al,al ; jnz out ; dec ecx ; jnz loop ; mov eax,0x11223344 ; ret ; out: mov eax,0x55 ; ret
    // The span the patcher relocates is pause + mov + test (6 bytes): the test comes after the memory access,
    // so it sits outside the shifted core and must still be emitted for the jnz that follows to see its flags.
    private static byte[] PauseLoadTestLoop() =>
    [
        0x31, 0xD2,                   // 0:  xor edx, edx  (ZF = 1 on entry)
        0xB9, 0x04, 0x00, 0x00, 0x00, // 2:  mov ecx, 4
        0xF3, 0x90,                   // 7:  loop: pause
        0x8A, 0x07,                   // 9:  mov al, [rdi]
        0x84, 0xC0,                   // 11: test al, al
        0x75, 0x0A,                   // 13: jnz out (next ip 15 + 10 = 25)
        0xFF, 0xC9,                   // 15: dec ecx
        0x75, 0xF4,                   // 17: jnz loop (19 - 12 = 7)
        0xB8, 0x44, 0x33, 0x22, 0x11, // 19: mov eax, 0x11223344
        0xC3,                         // 24: ret
        0xB8, 0x55, 0x00, 0x00, 0x00, // 25: out: mov eax, 0x55
        0xC3,                         // 30: ret
    ];

    [Theory]
    [InlineData((byte)0, 0x11223344u)]
    [InlineData((byte)7, 0x55u)]
    public void KeepsTheInstructionsAfterTheMemoryAccessOfTheSpan(byte cellValue, uint expected)
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;

        using var image = SpinImage.Create(PauseLoadTestLoop());
        Assert.Equal(1, image.Result.PatchedSites);
        Assert.Equal(0xE9, image.ReadByte(7));
        // The span (7..12) covers the test: its original byte is a nop now, so only the trampoline can run it.
        Assert.Equal(0x90, image.ReadByte(12));
        Assert.Equal(expected, ((delegate* unmanaged<byte*, uint>)image.Base)(&cellValue));
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

    private sealed class SpinImage : IDisposable
    {
        private const ulong ImageSize = 0x10000;
        private readonly PhysicalVirtualMemory _memory;

        private SpinImage(PhysicalVirtualMemory memory, ulong imageBase, GuestRedZonePatcher.SpinPatchResult result)
        {
            _memory = memory;
            Base = imageBase;
            Result = result;
        }

        public ulong Base { get; }

        public GuestRedZonePatcher.SpinPatchResult Result { get; }

        public static SpinImage Create(byte[] function, bool locked = false, byte[]? lockedProbe = null)
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
            ProgramHeader[] headers =
            [
                CreateProgramHeader(ProgramHeaderType.Load, ProgramHeaderFlags.Read | ProgramHeaderFlags.Execute,
                    0, (ulong)function.Length),
                CreateProgramHeader(ProgramHeaderType.GnuEhFrame, ProgramHeaderFlags.Read, 0x1000, (ulong)exceptionFrameHeader.Length),
            ];

            var probe = SpinNapRuntime.EnsureProbeBytes(1_000_000_000);
            lockedProbe ??= locked ? SpinNapRuntime.EnsureLockedProbeBytes(1_000_000_000) : [];
            var result = GuestRedZonePatcher.PatchSpinLoops(memory, memory, headers, imageBase, ImageSize, probe, "test", lockedProbe);
            return new SpinImage(memory, imageBase, result);
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
