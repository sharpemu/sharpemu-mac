// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Threading;
using Iced.Intel;

namespace SharpEmu.Core.Cpu.Native;

// Instructions Rosetta 2 does not translate, in a hot loop. Each execution raises #UD and goes
// through the software fallback in the fault handler, and the signal round trip is what costs:
// Black Myth: Wukong runs two SSE4a INSERTQ/EXTRQ sites some 300,000 times while it loads a
// level, about a minute of loading. A site that keeps faulting is rewritten once into a jump
// to a stub of instructions Rosetta does translate, which computes the same result and leaves
// the flags and every other register as the instruction would:
// - ADCX/ADOX: ADC computes the sum and carry; the stub then puts back every flag except the
//   one ADCX (CF) or ADOX (OF) defines.
// - EXTRQ/INSERTQ with immediate fields: the field masks are constants of the site, so the
//   stub does the bit-field work in general-purpose registers and MOVQ writes the low quadword
//   back, clearing the high one as the software fallback does.
//
// The rewrite happens while other threads may be running the same code, so it goes in an
// order no thread can observe half-done: the first byte becomes 0x06 (PUSH ES, #UD in 64-bit
// mode), then the jump's displacement and padding are written behind it, then the first byte
// becomes the jump's opcode. A thread that reaches the site in between faults on the 0x06 (or
// on the original instruction), and the fault handler sends it to the stub.
public sealed partial class DirectExecutionBackend
{
    /// <summary>Faults at one site before it is rewritten.</summary>
    private const int HotSitePatchThreshold = 16;

    /// <summary>Room for the longest stub: saved flags and registers, three 64-bit
    /// constants, the red zone step and the jump back.</summary>
    private const int HotSiteStubBytes = 128;

    private const byte InvalidOpcodeByte = 0x06;

    private static readonly bool HotSitePatchEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_HOT_SITE_PATCH") != "0";

    private readonly ConcurrentDictionary<ulong, int> _hotSiteFaultCounts = new();
    private readonly ConcurrentDictionary<ulong, nint> _patchedHotSites = new();
    private readonly ConcurrentDictionary<ulong, byte> _unpatchableHotSites = new();
    private static int _hotSitePatchAnnounced;
    private static long _hotSitesPatched;

    /// <summary>A fault at a site that was rewritten (or is being rewritten) resumes in its
    /// stub.</summary>
    private unsafe bool TryRedirectPatchedHotSite(void* contextRecord, ulong rip)
    {
        if (_patchedHotSites.IsEmpty || !_patchedHotSites.TryGetValue(rip, out var stub))
        {
            return false;
        }

        WriteCtxU64(contextRecord, CTX_RIP, (ulong)stub);
        return true;
    }

    /// <summary>Counts a fault of an instruction the caller just emulated in software and
    /// rewrites the site once it is hot.</summary>
    private void NoteHotSiteFault(ulong rip, in Instruction instruction)
    {
        if (!HotSitePatchEnabled || _unpatchableHotSites.ContainsKey(rip) || _patchedHotSites.ContainsKey(rip))
        {
            return;
        }

        if (_hotSiteFaultCounts.AddOrUpdate(rip, 1, static (_, count) => count + 1) != HotSitePatchThreshold)
        {
            return;
        }

        if (!TryPatchHotSite(rip, in instruction))
        {
            _unpatchableHotSites.TryAdd(rip, 0);
            if (_unpatchableHotSites.Count <= 8)
            {
                Console.Error.WriteLine(
                    $"[LOADER][WARN] Hot {instruction.Mnemonic} at 0x{rip:X16} could not be rewritten; it keeps faulting.");
            }
        }
    }

    // Stubs of hot sites, in pages placed near them: the jump into a stub and the jump back are
    // rel32. They used to come from the TLS stub region active at the time, which once every
    // module has loaded is the last one's: GTA V's hot INSERTQ at 0x802780034 was terabytes from
    // it, so it was never rewritten and trapped some 800 times a second for the whole game.
    private const int HotSiteStubPageBytes = 0x4000;
    private readonly List<(nint Base, int Used)> _hotSiteStubPages = [];

    /// <summary>Room for one stub within rel32 reach of <paramref name="site"/>, writable;
    /// 0 when none can be placed. Called under ImportStubPatchGate.</summary>
    private unsafe nint AllocateHotSiteStub(ulong site)
    {
        nint stub = 0;
        for (var index = 0; index < _hotSiteStubPages.Count && stub == 0; index++)
        {
            var (pageBase, used) = _hotSiteStubPages[index];
            if (used + HotSiteStubBytes <= HotSiteStubPageBytes &&
                WithinTlsReach(pageBase, site) && WithinTlsReach(pageBase + HotSiteStubPageBytes, site))
            {
                _hotSiteStubPages[index] = (pageBase, used + HotSiteStubBytes);
                stub = pageBase + used;
            }
        }

        if (stub == 0)
        {
            stub = (nint)TryAllocateNear(site, HotSiteStubPageBytes);
            if (stub == 0)
            {
                return 0;
            }

            _hotSiteStubPages.Add((stub, HotSiteStubBytes));
        }

        // An earlier stub sealed the page execute-read.
        uint previousProtection = 0;
        return VirtualProtect((void*)stub, HotSiteStubBytes, 64u, &previousProtection) ? stub : 0;
    }

    private unsafe bool TryPatchHotSite(ulong rip, in Instruction instruction)
    {
        var length = instruction.Length;
        List<Instruction> stubInstructions;
        var built = instruction.Mnemonic is Mnemonic.Adcx or Mnemonic.Adox
            ? TryBuildAdxStub(in instruction, out stubInstructions)
            : TryBuildSse4aStub(in instruction, out stubInstructions);
        if (length < 5 || !built)
        {
            return false;
        }

        lock (ImportStubPatchGate)
        {
            if (_patchedHotSites.ContainsKey(rip))
            {
                return true;
            }

            var stub = AllocateHotSiteStub(rip);
            if (stub == 0)
            {
                return false;
            }

            stubInstructions.Add(Instruction.CreateBranch(Code.Jmp_rel32_64, rip + (ulong)length));
            var writer = new StubCodeWriter();
            var block = new InstructionBlock(writer, stubInstructions, (ulong)stub);
            if (!BlockEncoder.TryEncode(64, block, out _, out _, BlockEncoderOptions.None) ||
                writer.Bytes.Count > HotSiteStubBytes)
            {
                return false;
            }

            var displacement = (long)stub - ((long)rip + 5);
            if (displacement is < int.MinValue or > int.MaxValue)
            {
                return false;
            }

            var code = new Span<byte>((void*)stub, HotSiteStubBytes);
            code.Fill(0xCC);
            for (var index = 0; index < writer.Bytes.Count; index++)
            {
                code[index] = writer.Bytes[index];
            }

            uint previousProtection = 0;
            VirtualProtect((void*)stub, HotSiteStubBytes, 32u, &previousProtection);
            FlushInstructionCache(GetCurrentProcess(), (void*)stub, HotSiteStubBytes);

            // From here a fault at the site resumes in the stub.
            _patchedHotSites[rip] = stub;

            Span<byte> tail = stackalloc byte[length - 1];
            tail.Fill(0x90);
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(tail, (int)displacement);
            if (!WriteTlsInstructionLocked((nint)rip, [InvalidOpcodeByte]) ||
                !WriteTlsInstructionLocked((nint)(rip + 1), tail) ||
                !WriteTlsInstructionLocked((nint)rip, [0xE9]))
            {
                // The map keeps redirecting whatever state the bytes were left in.
                return true;
            }
        }

        var patched = Interlocked.Increment(ref _hotSitesPatched);
        if (Interlocked.Exchange(ref _hotSitePatchAnnounced, 1) == 0)
        {
            Console.Error.WriteLine(
                $"[LOADER][INFO] Rewrote a hot {instruction.Mnemonic} at 0x{rip:X16} into a jump to a stub " +
                "Rosetta translates; later hot sites of untranslated instructions are rewritten the same way.");
        }
        else if (patched <= 16)
        {
            Console.Error.WriteLine($"[LOADER][INFO] Rewrote hot {instruction.Mnemonic} site #{patched} at 0x{rip:X16}.");
        }

        return true;
    }

    /// <summary>The stub body for ADCX/ADOX dst, src (register or memory), without the jump
    /// back. Fails on a form it does not model: a segment override, or RSP as an operand.</summary>
    internal static bool TryBuildAdxStub(in Instruction instruction, out List<Instruction> stub)
    {
        stub = [];
        var adox = instruction.Mnemonic == Mnemonic.Adox;
        if (instruction.Mnemonic is not (Mnemonic.Adcx or Mnemonic.Adox) ||
            instruction.Op0Kind != OpKind.Register)
        {
            return false;
        }

        var destination = instruction.Op0Register;
        var wide = destination.IsGPR64();
        if (!wide && !destination.IsGPR32())
        {
            return false;
        }

        Instruction add;
        var used = new HashSet<Register> { destination.GetFullRegister() };
        if (instruction.Op1Kind == OpKind.Register)
        {
            var source = instruction.Op1Register;
            if (source.GetFullRegister() == Register.RSP || destination.GetFullRegister() == Register.RSP)
            {
                return false;
            }

            used.Add(source.GetFullRegister());
            add = Instruction.Create(wide ? Code.Adc_r64_rm64 : Code.Adc_r32_rm32, destination, source);
        }
        else if (instruction.Op1Kind == OpKind.Memory &&
            instruction.SegmentPrefix == Register.None &&
            destination.GetFullRegister() != Register.RSP)
        {
            MemoryOperand memory;
            if (instruction.IsIPRelativeMemoryOperand)
            {
                memory = new MemoryOperand(Register.RIP, (long)instruction.IPRelativeMemoryAddress);
            }
            else
            {
                var baseRegister = instruction.MemoryBase;
                var index = instruction.MemoryIndex;
                if (index == Register.RSP)
                {
                    return false;
                }

                // The stub has moved RSP down past the red zone and one flag image.
                var displacement = (long)instruction.MemoryDisplacement64;
                if (baseRegister == Register.RSP)
                {
                    displacement += 128 + 8;
                }

                used.Add(baseRegister);
                used.Add(index);
                memory = new MemoryOperand(baseRegister, index, instruction.MemoryIndexScale, displacement, DisplacementSize(displacement));
            }

            add = Instruction.Create(wide ? Code.Adc_r64_rm64 : Code.Adc_r32_rm32, destination, memory);
        }
        else
        {
            return false;
        }

        var scratch = Register.None;
        foreach (var candidate in new[] { Register.RAX, Register.RCX, Register.RDX, Register.RBX, Register.RSI, Register.RDI })
        {
            if (!used.Contains(candidate))
            {
                scratch = candidate;
                break;
            }
        }

        if (scratch == Register.None)
        {
            return false;
        }

        var flagBit = adox ? 11 : 0;
        stub.Add(Instruction.Create(Code.Lea_r64_m, Register.RSP, new MemoryOperand(Register.RSP, -128, DisplacementSize(-128))));
        stub.Add(Instruction.Create(Code.Pushfq));
        if (adox)
        {
            // ADC adds CF; ADOX adds OF, so CF takes OF's value first.
            stub.Add(Instruction.Create(Code.Push_r64, scratch));
            stub.Add(Instruction.Create(Code.Mov_r64_rm64, scratch, new MemoryOperand(Register.RSP, 8, DisplacementSize(8))));
            stub.Add(Instruction.Create(Code.Bt_rm64_imm8, scratch, 11));
            stub.Add(Instruction.Create(Code.Pop_r64, scratch));
        }

        stub.Add(add);
        // [rsp] = the flags ADC produced, [rsp + 8] = the flags before.
        stub.Add(Instruction.Create(Code.Pushfq));
        stub.Add(Instruction.Create(Code.Push_r64, scratch));
        stub.Add(Instruction.Create(Code.Mov_r64_rm64, scratch, new MemoryOperand(Register.RSP, 8, DisplacementSize(8))));
        stub.Add(Instruction.Create(Code.And_rm64_imm8, scratch, 1));
        if (adox)
        {
            stub.Add(Instruction.Create(Code.Shl_rm64_imm8, scratch, flagBit));
        }

        stub.Add(Instruction.Create(Code.And_rm64_imm32, new MemoryOperand(Register.RSP, 16, DisplacementSize(16)), ~(1 << flagBit)));
        stub.Add(Instruction.Create(Code.Or_rm64_r64, new MemoryOperand(Register.RSP, 16, DisplacementSize(16)), scratch));
        stub.Add(Instruction.Create(Code.Pop_r64, scratch));
        stub.Add(Instruction.Create(Code.Lea_r64_m, Register.RSP, new MemoryOperand(Register.RSP, 8, DisplacementSize(8))));
        stub.Add(Instruction.Create(Code.Popfq));
        stub.Add(Instruction.Create(Code.Lea_r64_m, Register.RSP, new MemoryOperand(Register.RSP, 128, DisplacementSize(128))));
        return true;
    }

    /// <summary>The stub body for EXTRQ xmm, imm, imm and INSERTQ xmm, xmm, imm, imm, without
    /// the jump back. The register forms take their fields from an XMM register at run time
    /// and are left to the fault handler (they are too short to hold a jump anyway).</summary>
    internal static bool TryBuildSse4aStub(in Instruction instruction, out List<Instruction> stub)
    {
        stub = [];
        var extract = instruction.Mnemonic == Mnemonic.Extrq && instruction.OpCount == 3;
        var insert = instruction.Mnemonic == Mnemonic.Insertq && instruction.OpCount == 4;
        if ((!extract && !insert) ||
            instruction.Op0Kind != OpKind.Register || !instruction.Op0Register.IsXMM() ||
            (insert && (instruction.Op1Kind != OpKind.Register || !instruction.Op1Register.IsXMM())))
        {
            return false;
        }

        var destination = instruction.Op0Register;
        var length = (int)instruction.GetImmediate(extract ? 1 : 2) & 0x3F;
        var index = (int)instruction.GetImmediate(extract ? 2 : 3) & 0x3F;
        if (length == 0)
        {
            length = 64;
        }

        // The software fallback's limits (Sse4aBitFieldEmulator): a field past bit 63 is cut.
        length = Math.Min(length, 64 - index);
        var fieldMask = length == 64 ? ulong.MaxValue : (1UL << length) - 1;

        stub.Add(Instruction.Create(Code.Lea_r64_m, Register.RSP, new MemoryOperand(Register.RSP, -128, DisplacementSize(-128))));
        stub.Add(Instruction.Create(Code.Pushfq));
        stub.Add(Instruction.Create(Code.Push_r64, Register.RAX));
        stub.Add(Instruction.Create(Code.Push_r64, Register.RCX));
        if (extract)
        {
            // low = (low >> index) & mask; MOVQ clears the high quadword.
            stub.Add(Instruction.Create(Code.Movq_rm64_xmm, Register.RAX, destination));
            if (index != 0)
            {
                stub.Add(Instruction.Create(Code.Shr_rm64_imm8, Register.RAX, index));
            }

            stub.Add(Instruction.Create(Code.Mov_r64_imm64, Register.RCX, fieldMask));
            stub.Add(Instruction.Create(Code.And_rm64_r64, Register.RAX, Register.RCX));
            stub.Add(Instruction.Create(Code.Movq_xmm_rm64, destination, Register.RAX));
        }
        else
        {
            // low = (low & ~(mask << index)) | ((source & mask) << index); the source is read
            // before the destination is written, so the two may be the same register.
            stub.Add(Instruction.Create(Code.Push_r64, Register.RDX));
            stub.Add(Instruction.Create(Code.Movq_rm64_xmm, Register.RAX, instruction.Op1Register));
            stub.Add(Instruction.Create(Code.Mov_r64_imm64, Register.RCX, fieldMask));
            stub.Add(Instruction.Create(Code.And_rm64_r64, Register.RAX, Register.RCX));
            if (index != 0)
            {
                stub.Add(Instruction.Create(Code.Shl_rm64_imm8, Register.RAX, index));
            }

            stub.Add(Instruction.Create(Code.Movq_rm64_xmm, Register.RCX, destination));
            stub.Add(Instruction.Create(Code.Mov_r64_imm64, Register.RDX, ~(fieldMask << index)));
            stub.Add(Instruction.Create(Code.And_rm64_r64, Register.RCX, Register.RDX));
            stub.Add(Instruction.Create(Code.Or_rm64_r64, Register.RCX, Register.RAX));
            stub.Add(Instruction.Create(Code.Movq_xmm_rm64, destination, Register.RCX));
            stub.Add(Instruction.Create(Code.Pop_r64, Register.RDX));
        }

        stub.Add(Instruction.Create(Code.Pop_r64, Register.RCX));
        stub.Add(Instruction.Create(Code.Pop_r64, Register.RAX));
        stub.Add(Instruction.Create(Code.Popfq));
        stub.Add(Instruction.Create(Code.Lea_r64_m, Register.RSP, new MemoryOperand(Register.RSP, 128, DisplacementSize(128))));
        return true;
    }

    /// <summary>Iced's displacement size for 64-bit addressing: none, disp8, or disp32
    /// (which it calls 8).</summary>
    private static int DisplacementSize(long displacement) =>
        displacement == 0 ? 0 : displacement is >= sbyte.MinValue and <= sbyte.MaxValue ? 1 : 8;

    private sealed class StubCodeWriter : CodeWriter
    {
        public List<byte> Bytes { get; } = [];

        public override void WriteByte(byte value) => Bytes.Add(value);
    }
}
