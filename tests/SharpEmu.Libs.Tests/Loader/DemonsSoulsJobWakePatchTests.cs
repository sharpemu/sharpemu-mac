// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Iced.Intel;
using SharpEmu.Core.Loader;
using SharpEmu.Libs.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class DemonsSoulsJobWakePatchTests
{
    private const ulong NapStub = 0x7FF008000UL;
    private const ulong RegionStart = 0x860000;

    private static readonly DemonsSoulsIdleNapPatch.WakeLayout Layout =
        new(ImageBase: 0x800000000, CodeAddress: 0x803000000, WaitStub: 0x7FF000000, WakeStub: 0x7FF004000, StateAddress: 0x123456789A0);

    [Fact]
    public void WakeModeKeepsTheNapBlocksAndRetargetsOnlyTheIdleAndReadySites()
    {
        var nap = DemonsSoulsIdleNapPatch.BuildPatches(500_000, 250, NapStub);
        var wake = DemonsSoulsIdleNapPatch.BuildPatches(500_000, 250, NapStub, Layout);
        _ = DemonsSoulsIdleNapPatch.BuildWakeCode(Layout, 500_000, 250, out var entries);

        Assert.Equal(11, wake.Count);
        Assert.All(wake, patch => Assert.Equal(patch.Old!.Length, patch.New.Length));
        Assert.Equal(wake.Select(patch => patch.Offset).Distinct().Count(), wake.Count);

        // Report block, its tail, the deadlock branch and RunJob's stamp are the nap's own.
        foreach (var offset in new[] { DemonsSoulsIdleNapPatch.DeadlockBranch, DemonsSoulsIdleNapPatch.ReportBlock, DemonsSoulsIdleNapPatch.ReportTail, DemonsSoulsIdleNapPatch.RunJobCall })
        {
            Assert.Equal(Find(nap, offset).New, Find(wake, offset).New);
        }

        var code = Layout.CodeAddress;
        Assert.Equal(code + (ulong)entries.WaitTsc, CallTarget(wake, DemonsSoulsIdleNapPatch.WaitTscCall));
        Assert.Equal(code + (ulong)entries.ChooseJob, CallTarget(wake, DemonsSoulsIdleNapPatch.ChooseJobCall));
        Assert.Equal(code + (ulong)entries.SubmitRelease, CallTarget(wake, DemonsSoulsIdleNapPatch.SubmitReleaseCall));
        foreach (var site in DemonsSoulsIdleNapPatch.CounterChangedCalls)
        {
            Assert.Equal(code + (ulong)entries.CounterChanged, CallTarget(wake, site));
            Assert.Equal(Layout.ImageBase + 0x86DCB0, OldCallTarget(wake, site));
        }

        Assert.Equal(Layout.ImageBase + 0x8718C0, OldCallTarget(wake, DemonsSoulsIdleNapPatch.SubmitReleaseCall));
    }

    [Fact]
    public void WakeTrampolinesAssembleToTheIntendedCode()
    {
        var code = DemonsSoulsIdleNapPatch.BuildWakeCode(Layout, 500_000, 250, out var entries);

        Assert.Equal(new DemonsSoulsIdleNapPatch.WakeEntries(0x00, 0x60, 0xA0, 0xF0), entries);
        Assert.Equal(
            [
                "push rbx", "push r12", "push r13", "mov rbx,rdi", "mov rax,123456789A0h", "mov r13,[rax]",
                "call 000000080086ECD0h", "test al,al", "jne short 000000080300005Ah", "call 0000000800864CF0h",
                "mov r12,rax", "call 0000000800863A20h", "sub rax,[rbx+r12*8+0F00h]", "cmp rax,7A120h",
                "jl short 0000000803000058h", "mov rdi,rbx", "mov esi,r12d", "mov rdx,r13", "mov ecx,0FAh",
                "xor r8d,r8d", "mov rax,7FF000000h", "call rax", "xor eax,eax", "pop r13", "pop r12", "pop rbx", "ret",
            ],
            Disassemble(code, entries.ChooseJob));
        Assert.Equal(
            [
                "call 0000000800863A20h", "push rax", "sub rax,r12", "cmp rax,7A120h", "jl short 000000080300009Bh",
                "mov rax,123456789A0h", "mov rdx,[rax]", "mov rdi,rbx", "mov esi,r14d", "mov ecx,0FAh", "mov r8d,1",
                "mov rax,7FF000000h", "call rax", "pop rax", "ret",
            ],
            Disassemble(code, entries.WaitTsc));
        Assert.Equal(
            [
                "push rdi", "push rsi", "push rax", "mov rcx,[rbp-98h]", "mov esi,[rcx+0DA0h]", "mov rax,[rcx+8]",
                "sub rax,[rcx]", "mov rcx,123456789A0h", "lock inc qword [rcx]", "test [rcx+8],rsi",
                "je short 00000008030000E7h", "xor edx,edx", "mov ecx,158h", "div rcx", "mov rdx,rax", "mov rdi,r12",
                "mov rax,7FF004000h", "call rax", "pop rax", "pop rsi", "pop rdi", "jmp 00000008008718C0h",
            ],
            Disassemble(code, entries.SubmitRelease));
        Assert.Equal(
            [
                "push rdi", "call 000000080086DCB0h", "pop rdi", "mov rax,123456789A0h", "lock inc qword [rax]",
                "cmp qword [rax+10h],0", "jne short 000000080300010Dh", "ret",
            ],
            Disassemble(code, entries.CounterChanged));
        Assert.Equal(
            ["sub rsp,8", "mov rsi,0FFFFFFFFFFFFFFFFh", "mov edx,1", "mov rax,7FF004000h", "call rax", "add rsp,8", "ret"],
            Disassemble(code, entries.CounterChanged + 0x1D));
    }

    [Fact]
    public void ApplyWiresWakeOnReadyWhenEveryByteMatchesAndNapsOtherwise()
    {
        using var memory = new SharpEmu.Core.Memory.PhysicalVirtualMemory();
        Assert.True(memory.TryAllocateAtOrAbove(0x200000000UL, 0x20000, executable: false, 0x4000, out var region));
        Assert.True(memory.TryAllocateAtOrAbove(region + 0x100000, DemonsSoulsIdleNapPatch.WakeCodeBytes, executable: false, 0x4000, out var codeBlock));
        var imageBase = region - RegionStart;
        ProgramHeader[] headers = [GuestBytePatcherTests.CreateLoad(RegionStart, 0x20000)];
        var requested = new List<string>();
        ulong AddStub(string nid)
        {
            requested.Add(nid);
            return 0x7FF000000UL + ((ulong)requested.Count * 0x4000);
        }

        var layout = Layout with { ImageBase = imageBase, CodeAddress = codeBlock };
        foreach (var patch in DemonsSoulsIdleNapPatch.BuildPatches(1, 1, 0, layout))
        {
            Assert.True(memory.TryWrite(imageBase + patch.Offset, patch.Old!));
        }

        foreach (var (offset, bytes) in DemonsSoulsIdleNapPatch.WakeContext.Skip(1))
        {
            Assert.True(memory.TryWrite(imageBase + offset, bytes));
        }

        // One context byte off: the nap goes in, the wake stays out.
        Assert.True(DemonsSoulsIdleNapPatch.Apply(memory, headers, imageBase, "PPSA01342", AddStub, _ => codeBlock));
        Assert.Equal([IdleNapExports.Nid], requested);
        Assert.Equal(imageBase + DemonsSoulsIdleNapPatch.ReportBlock, ReadCallTarget(memory, imageBase, DemonsSoulsIdleNapPatch.WaitTscCall));

        // Restore the image and the missing context byte: with a host wait, the wake goes in.
        foreach (var patch in DemonsSoulsIdleNapPatch.BuildPatches(1, 1, 0, layout))
        {
            Assert.True(memory.TryWrite(imageBase + patch.Offset, patch.Old!));
        }

        var (firstOffset, firstBytes) = DemonsSoulsIdleNapPatch.WakeContext[0];
        Assert.True(memory.TryWrite(imageBase + firstOffset, firstBytes));
        requested.Clear();
        Assert.True(DemonsSoulsIdleNapPatch.Apply(memory, headers, imageBase, "PPSA01342", AddStub, _ => codeBlock));
        if (!JobWakeRuntime.IsSupported)
        {
            Assert.Equal([IdleNapExports.Nid], requested);
            return;
        }

        Assert.Equal([IdleNapExports.Nid, JobWakeExports.WaitNid, JobWakeExports.WakeNid], requested);
        var applied = layout with { WaitStub = 0x7FF008000, WakeStub = 0x7FF00C000, StateAddress = JobWakeRuntime.StateAddress };
        var expected = DemonsSoulsIdleNapPatch.BuildWakeCode(applied, DemonsSoulsIdleNapPatch.ThresholdTicks(500, KernelRuntimeCompatExports.TscFrequency), 250, out var entries);
        var written = new byte[expected.Length];
        Assert.True(memory.TryRead(codeBlock, written));
        Assert.Equal(expected, written);
        Assert.Equal(codeBlock + (ulong)entries.WaitTsc, ReadCallTarget(memory, imageBase, DemonsSoulsIdleNapPatch.WaitTscCall));
        Assert.Equal(codeBlock + (ulong)entries.SubmitRelease, ReadCallTarget(memory, imageBase, DemonsSoulsIdleNapPatch.SubmitReleaseCall));
        Assert.Equal(codeBlock + (ulong)entries.CounterChanged, ReadCallTarget(memory, imageBase, 0x86F906));
    }

    private static List<string> Disassemble(byte[] code, int entry)
    {
        var decoder = Decoder.Create(64, new ByteArrayCodeReader(code[entry..]));
        decoder.IP = Layout.CodeAddress + (ulong)entry;
        var formatter = new NasmFormatter();
        var text = new StringOutput();
        var lines = new List<string>();
        while (true)
        {
            decoder.Decode(out var instruction);
            formatter.Format(instruction, text);
            lines.Add(text.ToStringAndReset());
            if (instruction.FlowControl is FlowControl.Return or FlowControl.UnconditionalBranch)
            {
                return lines;
            }
        }
    }

    private static GuestBytePatcher.Patch Find(List<GuestBytePatcher.Patch> patches, ulong offset) =>
        patches.Single(patch => patch.Offset == offset);

    private static ulong CallTarget(List<GuestBytePatcher.Patch> patches, ulong offset) =>
        Target(Find(patches, offset).New, Layout.ImageBase + offset);

    private static ulong OldCallTarget(List<GuestBytePatcher.Patch> patches, ulong offset) =>
        Layout.ImageBase + Target(Find(patches, offset).Old!, offset);

    private static ulong ReadCallTarget(SharpEmu.Core.Memory.PhysicalVirtualMemory memory, ulong imageBase, ulong offset)
    {
        var call = new byte[5];
        Assert.True(memory.TryRead(imageBase + offset, call));
        return Target(call, imageBase + offset);
    }

    private static ulong Target(byte[] call, ulong at)
    {
        Assert.Equal(0xE8, call[0]);
        return at + 5 + (ulong)(long)BinaryPrimitives.ReadInt32LittleEndian(call.AsSpan(1));
    }
}
