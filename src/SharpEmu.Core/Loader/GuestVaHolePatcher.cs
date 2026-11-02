// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Iced.Intel;
using SharpEmu.Core.Memory;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Core.Loader;

// Workaround for macOS 27, which refuses host mappings in [0x10_0000_0000, 0x70_0000_0000) - a
// permanent guard region every process starts with (vmmap: Guard fc0000000-7000000000, ---/---),
// which neither MAP_FIXED nor mach_vm_allocate with VM_FLAGS_OVERWRITE can replace. Titles that
// hardcode the start of the PS5 user area (0x10_0000_0000) as the base of an arena they map with
// sceKernelBatchMap would map fixed inside that window.
//
// The kernel-side memory calls already move any guest address inside the window by one shift
// (GuestVaHole.Shift, 0x10_0000_0000 -> 0x80_0000_0000 unless SHARPEMU_VA_HOLE_BASE says otherwise),
// so a title that uses the address it gets back needs nothing more. This patch is the other half of
// the same scheme: where the loader finds a hardcoded base stored as an arena base, it moves the
// base by that same shift in the code, so the title asks for the moved address itself and every
// address it derives from that base stays consistent with the kernel's relocation. Without it a
// title that mixes a patched base with constants it computes from the unpatched one would split
// between two places:
//  - GTA V Enhanced (RAGE) stores its streaming arena base in a global;
//  - Ghost of Tsushima maps its arenas one batch entry at a time (0x10_0000_0000 for libc and its
//    direct-memory pools, 0x11_0000_0000 and 0x15_0000_0000 for two more) from a base it keeps in a
//    global after the call.
// Every such base moves by the same amount, so arenas keep their order and spacing. Other loads of
// these constants stay as they are: the user area base also feeds range checks ("below the user
// area"), which an arena moved higher still passes, and the others are packed 32-bit pairs in
// struct initialisers, which must not change.
internal static class GuestVaHolePatcher
{
    // A signature byte that matches anything.
    private const short AnyByte = -1;

    // movabs rax, 0x1000000000
    // lea    rdi, [rip+disp32]
    // mov    esi, imm32
    // mov    r64, r64
    // mov    [rip+disp32], rax
    // RAGE (GTA V Enhanced) stores its streaming arena base this way.
    private static readonly short[] RageArenaBaseSignature =
    [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x48, 0x8D, 0x3D, AnyByte, AnyByte, AnyByte, AnyByte,
        0xBE, AnyByte, AnyByte, AnyByte, AnyByte,
        0x48, 0x89, AnyByte,
        0x48, 0x89, 0x05,
    ];

    // movabs rax, base; vmovups [rbx], ymm0; mov qword [rbx+0x20], 0; vxorps xmm0, xmm0, xmm0;
    // mov [rbx+0x28], rax - Ghost of Tsushima's engine describes each of its two virtual-memory regions
    // this way, [0x20, 0x30) and [0x30, 0xFC) x 2^32, the second with its size a literal right after
    // (movabs rax, size; mov [rbx+0x30], rax). It maps everything it hands out inside them.
    private static readonly short[] RegionDescriptorSignature =
    [
        0x48, 0xB8, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte,
        0xC5, 0xFC, 0x11, 0x03,
        0x48, 0xC7, 0x43, 0x20, 0x00, 0x00, 0x00, 0x00,
        0xC5, 0xF8, 0x57, 0xC0,
        0x48, 0x89, 0x43, 0x28,
    ];

    private static readonly short[] RegionSizeSignature =
    [
        0x48, 0xB8, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte, AnyByte,
        0x48, 0x89, 0x43, 0x30,
    ];

    // SHARPEMU_VA_HOLE_REGIONS=1 (experimental): where those regions go instead, clear of the mappings
    // placed anywhere (searched from 0x70_0000_0000 on macOS) and of where the arenas of
    // [0x10, 0x20) x 2^32 move (0x80-0x8F). The first keeps its 64 GiB; the second keeps its end, so a
    // literal size shrinks by as much as its base moves up.
    private static readonly (ulong Base, ulong NewBase)[] RegionMoves =
    [
        (0x20_0000_0000UL, 0x90_0000_0000UL),
        (0x30_0000_0000UL, 0xA0_0000_0000UL),
    ];

    // The window macOS 27 guards, of which the user area's arenas are 4 GiB-aligned bases.
    private const ulong WindowStart = 0x10_0000_0000UL;
    private const ulong WindowEnd = 0x70_0000_0000UL;
    private const ulong ArenaAlignment = 0x1_0000_0000UL;

    // An arena base loaded for a one-entry sceKernelBatchMap: a movabs of a 4 GiB-aligned base in the
    // window, then the batch setup - at most one store of the base into the entry (mov [rbp+disp8], r64),
    // lea rdi, [rbp+disp8]; mov esi, 1; xor edx, edx - or, for a base chosen on another path, a store of
    // another register into the arena's record (mov [r+disp32], r64) and a jmp to such a setup.
    internal readonly record struct ArenaBaseSite(int Offset, ulong Base);

    // Every arena-base load in code, each a movabs whose immediate starts two bytes in.
    internal static List<ArenaBaseSite> FindArenaBaseSites(ReadOnlySpan<byte> code)
    {
        var sites = new List<ArenaBaseSite>();
        for (var offset = 0; offset + 10 <= code.Length; offset++)
        {
            if ((code[offset] != 0x48 && code[offset] != 0x49) || code[offset + 1] < 0xB8 || code[offset + 1] > 0xBF)
            {
                continue;
            }

            var immediate = BitConverter.ToUInt64(code.Slice(offset + 2, 8));
            if (Matches(code, offset, RageArenaBaseSignature) ||
                (immediate is >= WindowStart and < WindowEnd && immediate % ArenaAlignment == 0 &&
                 (FeedsOneEntryBatchMap(code, offset + 10) || FeedsOneEntryBatchMapThroughCode(code, offset))))
            {
                sites.Add(new ArenaBaseSite(offset, immediate));
            }
        }

        return sites;
    }

    private static bool FeedsOneEntryBatchMap(ReadOnlySpan<byte> code, int next)
    {
        if (IsOneEntryBatchSetup(code, next))
        {
            return true;
        }

        // An unrelated store first (Ghost of Tsushima's 0x12_0000_0000 arena: mov [r12+0x4d0], r14),
        // the base then stored into the entry after the setup.
        if (StoreLength(code, next) is > 0 and var length && IsOneEntryBatchSetup(code, next + length))
        {
            return true;
        }

        // mov [r64+disp32], r64 (REX.WB 89 /r, mod 10) and jmp rel32 to the shared setup.
        if (next + 12 > code.Length || code[next] != 0x49 || code[next + 1] != 0x89 || (code[next + 2] & 0xC0) != 0x80 ||
            code[next + 7] != 0xE9)
        {
            return false;
        }

        var target = next + 12 + BitConverter.ToInt32(code.Slice(next + 8, 4));
        return target >= 0 && IsOneEntryBatchSetup(code, target);
    }

    // The same one-entry batch setup further on (Ghost of Tsushima's 0x16_0000_0000 arena keeps its base
    // in r15 across an unrelated call, then sets up with lea rdi, [rbp-0x170]): the base register is
    // followed through straight-line code until a call finds it stored at [rbp+disp], rdi pointing there
    // and esi 1. A branch, a write to the register, or an earlier call it does not survive (only
    // callee-saved registers do) ends the search.
    private static bool FeedsOneEntryBatchMapThroughCode(ReadOnlySpan<byte> code, int offset)
    {
        var length = Math.Min(160, code.Length - offset);
        var decoder = Decoder.Create(64, new ByteArrayCodeReader(code.Slice(offset, length).ToArray()));
        decoder.Decode(out var movabs);
        var baseRegister = movabs.Op0Register.GetFullRegister();
        var calleeSaved = baseRegister is Register.RBX or Register.R12 or Register.R13 or Register.R14 or Register.R15;
        var info = new InstructionInfoFactory();
        long? stored = null, pointed = null;
        var oneEntry = false;
        for (var count = 0; count < 24 && decoder.IP < (ulong)length; count++)
        {
            decoder.Decode(out var instruction);
            if (instruction.IsInvalid)
            {
                return false;
            }

            if (instruction.FlowControl == FlowControl.Call)
            {
                if (stored is { } slot && pointed == slot && oneEntry)
                {
                    return true;
                }

                if (!calleeSaved)
                {
                    return false;
                }

                // The setup belongs to the call it precedes.
                pointed = null;
                oneEntry = false;
                continue;
            }

            if (instruction.FlowControl != FlowControl.Next)
            {
                return false;
            }

            if (instruction.Mnemonic == Mnemonic.Mov && instruction.Op0Kind == OpKind.Memory &&
                instruction.Op1Kind == OpKind.Register && instruction.Op1Register == baseRegister &&
                instruction.MemoryBase == Register.RBP && instruction.MemoryIndex == Register.None)
            {
                stored = (long)instruction.MemoryDisplacement64;
                continue;
            }

            if (instruction.Mnemonic == Mnemonic.Lea && instruction.Op0Register == Register.RDI &&
                instruction.MemoryBase == Register.RBP && instruction.MemoryIndex == Register.None)
            {
                pointed = (long)instruction.MemoryDisplacement64;
                continue;
            }

            if (instruction.Mnemonic == Mnemonic.Mov && instruction.Op0Register == Register.ESI &&
                instruction.Op1Kind == OpKind.Immediate32 && instruction.Immediate32 == 1)
            {
                oneEntry = true;
                continue;
            }

            foreach (var used in info.GetInfo(instruction).GetUsedRegisters())
            {
                if (used.Register.GetFullRegister() == baseRegister && used.Access is not (OpAccess.Read or OpAccess.CondRead))
                {
                    return false;
                }
            }
        }

        return false;
    }

    // The length of mov [base+disp], r64 (REX.W 89 /r, mod 01 or 10, a SIB byte when rm is 100), or 0.
    private static int StoreLength(ReadOnlySpan<byte> code, int at)
    {
        if (at + 3 > code.Length || (code[at] & 0xF8) != 0x48 || code[at + 1] != 0x89)
        {
            return 0;
        }

        var mod = code[at + 2] >> 6;
        if (mod is not (1 or 2))
        {
            return 0;
        }

        var length = 3 + ((code[at + 2] & 7) == 4 ? 1 : 0) + (mod == 1 ? 1 : 4);
        return at + length <= code.Length ? length : 0;
    }

    private static bool IsOneEntryBatchSetup(ReadOnlySpan<byte> code, int at)
    {
        // An optional mov [rbp+disp8], r64 (REX.W or REX.WR 89, mod 01 rm 101).
        if (at + 4 <= code.Length && (code[at] is 0x48 or 0x4C) && code[at + 1] == 0x89 && (code[at + 2] & 0xC7) == 0x45)
        {
            at += 4;
        }

        ReadOnlySpan<byte> setup = [0x48, 0x8D, 0x7D];
        ReadOnlySpan<byte> oneEntry = [0xBE, 0x01, 0x00, 0x00, 0x00, 0x31, 0xD2];
        return at + 11 <= code.Length && code.Slice(at, 3).SequenceEqual(setup) && code.Slice(at + 4, 7).SequenceEqual(oneEntry);
    }

    public static int Patch(IVirtualMemory memory, IReadOnlyList<ProgramHeader> programHeaders, ulong imageBase)
    {
        if (!OperatingSystem.IsMacOSVersionAtLeast(27) ||
            Environment.GetEnvironmentVariable("SHARPEMU_DISABLE_VA_HOLE_PATCH") == "1")
        {
            return 0;
        }

        var moveRegions = Environment.GetEnvironmentVariable("SHARPEMU_VA_HOLE_REGIONS") == "1";
        var patched = 0;
        foreach (var header in programHeaders)
        {
            if (header.HeaderType != ProgramHeaderType.Load ||
                (header.Flags & ProgramHeaderFlags.Execute) == 0 ||
                header.FileSize == 0 ||
                header.FileSize > int.MaxValue)
            {
                continue;
            }

            var segmentStart = imageBase + header.VirtualAddress;
            var bytes = GC.AllocateUninitializedArray<byte>((int)header.FileSize);
            if (!memory.TryRead(segmentStart, bytes))
            {
                continue;
            }

            Span<byte> immediate = stackalloc byte[8];
            foreach (var found in FindArenaBaseSites(bytes))
            {
                var site = segmentStart + (ulong)found.Offset;
                var moved = unchecked(found.Base + GuestVaHole.Shift);
                BitConverter.TryWriteBytes(immediate, moved);
                var ok = memory.TryWrite(site + 2, immediate);
                Console.Error.WriteLine(
                    $"[LOADER][INFO] VA hole patch: arena base 0x{found.Base:X} -> 0x{moved:X} at 0x{site:X16} ok={ok}");
                if (ok)
                {
                    patched++;
                }
            }

            if (moveRegions)
            {
                patched += MoveRegionDescriptors(memory, segmentStart, bytes);
            }
        }

        return patched;
    }

    private static int MoveRegionDescriptors(IVirtualMemory memory, ulong segmentStart, ReadOnlySpan<byte> code)
    {
        var patched = 0;
        Span<byte> immediate = stackalloc byte[8];
        for (var offset = 0; offset + RegionDescriptorSignature.Length <= code.Length; offset++)
        {
            if (code[offset] != 0x48 || code[offset + 1] != 0xB8 || !Matches(code, offset, RegionDescriptorSignature))
            {
                continue;
            }

            var regionBase = BitConverter.ToUInt64(code.Slice(offset + 2, 8));
            var move = Array.FindIndex(RegionMoves, m => m.Base == regionBase);
            if (move < 0)
            {
                continue;
            }

            var newBase = RegionMoves[move].NewBase;
            BitConverter.TryWriteBytes(immediate, newBase);
            var site = segmentStart + (ulong)offset;
            var ok = memory.TryWrite(site + 2, immediate);
            var sizeAt = offset + RegionDescriptorSignature.Length;
            var sizeNote = string.Empty;
            if (ok && Matches(code, sizeAt, RegionSizeSignature))
            {
                var size = BitConverter.ToUInt64(code.Slice(sizeAt + 2, 8));
                var newSize = regionBase + size > newBase ? regionBase + size - newBase : 0;
                BitConverter.TryWriteBytes(immediate, newSize);
                ok = newSize != 0 && memory.TryWrite(segmentStart + (ulong)sizeAt + 2, immediate);
                sizeNote = $" size 0x{size:X} -> 0x{newSize:X}";
            }

            Console.Error.WriteLine(
                $"[LOADER][INFO] VA hole patch: VM region base 0x{regionBase:X} -> 0x{newBase:X}{sizeNote} at 0x{site:X16} ok={ok}");
            if (ok)
            {
                patched++;
            }
        }

        return patched;
    }

    private static bool Matches(ReadOnlySpan<byte> code, int offset, short[] signature)
    {
        if (offset + signature.Length > code.Length)
        {
            return false;
        }

        for (var index = 0; index < signature.Length; index++)
        {
            var actual = code[offset + index];
            var expected = signature[index];
            if (expected != AnyByte && actual != expected)
            {
                return false;
            }
        }

        return true;
    }
}
