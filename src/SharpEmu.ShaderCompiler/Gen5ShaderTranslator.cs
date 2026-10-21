// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.ShaderCompiler.Ir;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace SharpEmu.ShaderCompiler;

public static partial class Gen5ShaderTranslator
{
    private static int _dppVectorsValidated;

    // Some shipped Gen5 shaders exceed 64 KiB of encoded instructions before
    // reaching S_ENDPGM. Keep the malformed-shader guard, but size it to the
    // largest declared shader object accepted by the decoder (1 MiB).
    private const int MaxInstructions = (int)(MaximumDeclaredShaderSizeBytes / sizeof(uint));
    private const ulong FusedContinuationAlignment = 0x100;
    private const ulong ShaderSizeOffset = 0x44;
    private const uint MaximumDeclaredShaderSizeBytes = 1024 * 1024;
    private const uint NullScalarRegister = 125;
    private static readonly ConditionalWeakTable<object, FusedProgramRegistry> _fusedProgramsByMemory = new();

    private sealed class FusedProgramRegistry
    {
        private readonly object _gate = new();
        private Dictionary<ulong, FusedShaderParts> _programs = new();

        public bool TryGet(ulong entryAddress, out FusedShaderParts? parts) =>
            Volatile.Read(ref _programs).TryGetValue(entryAddress, out parts);

        public void Set(ulong entryAddress, FusedShaderParts parts)
        {
            lock (_gate)
            {
                var next = new Dictionary<ulong, FusedShaderParts>(_programs) { [entryAddress] = parts };
                Volatile.Write(ref _programs, next);
            }
        }

        public bool Remove(ulong entryAddress)
        {
            lock (_gate)
            {
                if (!_programs.ContainsKey(entryAddress)) return false;
                var next = new Dictionary<ulong, FusedShaderParts>(_programs);
                next.Remove(entryAddress);
                Volatile.Write(ref _programs, next);
                return true;
            }
        }
    }

    private sealed record FusedShaderParts(
        ulong EntryHeaderAddress,
        ulong ContinuationAddress,
        ulong ContinuationHeaderAddress);

    private static FusedProgramRegistry GetFusedPrograms(ICpuMemory memory)
    {
        // Each native guest thread has its own tracking wrapper, but all
        // shader creation and render contexts share the underlying memory.
        while (memory is ICpuMemoryWrapper wrapper)
        {
            memory = wrapper.Inner;
        }

        return _fusedProgramsByMemory.GetValue(memory, static _ => new FusedProgramRegistry());
    }

    /// <summary>
    /// Records the two code objects that AGC joins into one hardware shader.
    /// The entry code transfers control to the continuation with S_SETPC_B64, or with an
    /// S_SWAPPC_B64 that writes the return address to NULL.
    /// </summary>
    public static void RegisterFusedProgram(
        CpuContext ctx,
        ulong entryAddress,
        ulong entryHeaderAddress,
        ulong continuationAddress,
        ulong continuationHeaderAddress)
    {
        if (entryAddress == 0 ||
            entryHeaderAddress == 0 ||
            continuationAddress == 0 ||
            continuationHeaderAddress == 0)
        {
            return;
        }

        var registry = GetFusedPrograms(ctx.Memory);
        registry.Set(entryAddress, new FusedShaderParts(
            entryHeaderAddress,
            continuationAddress,
            continuationHeaderAddress));
    }

    public static bool UnregisterFusedProgram(CpuContext ctx, ulong entryAddress)
    {
        return GetFusedPrograms(ctx.Memory).Remove(entryAddress);
    }

    // The continuation registered for an entry address, when the guest joined two code objects.
    public static bool TryGetFusedProgramParts(
        CpuContext ctx,
        ulong entryAddress,
        out ulong continuationAddress,
        out ulong continuationHeaderAddress)
    {
        var registry = GetFusedPrograms(ctx.Memory);
        registry.TryGet(entryAddress, out var parts);
        continuationAddress = parts?.ContinuationAddress ?? 0;
        continuationHeaderAddress = parts?.ContinuationHeaderAddress ?? 0;
        return parts is not null;
    }

    public static bool TryGetFusedProgramParts(
        CpuContext ctx,
        ulong entryAddress,
        out ulong entryHeaderAddress,
        out ulong continuationAddress,
        out ulong continuationHeaderAddress)
    {
        var registry = GetFusedPrograms(ctx.Memory);
        registry.TryGet(entryAddress, out var parts);
        entryHeaderAddress = parts?.EntryHeaderAddress ?? 0;
        continuationAddress = parts?.ContinuationAddress ?? 0;
        continuationHeaderAddress = parts?.ContinuationHeaderAddress ?? 0;
        return parts is not null;
    }

    public static string Describe(CpuContext ctx, ulong exportShaderAddress, ulong pixelShaderAddress)
    {
        var es = TryDecodeProgram(ctx, exportShaderAddress, out var esProgram, out var esError)
            ? ShaderDecodeInfo.Create(esProgram).ToString()
            : $"error={esError}";
        var ps = TryDecodeProgram(ctx, pixelShaderAddress, out var psProgram, out var psError)
            ? ShaderDecodeInfo.Create(psProgram).ToString()
            : $"error={psError}";
        return $"es[{es}] ps[{ps}]";
    }

    public static string DescribeWords(CpuContext ctx, ulong shaderAddress) =>
        TryDecodeProgram(ctx, shaderAddress, out var program, out var error)
            ? string.Join(',', program.Instructions.SelectMany(instruction => instruction.Words)
                .Select(word => $"{word:X8}"))
            : $"error={error}";

    // Public contract entry: emitter test suites and tools drive the decoder directly
    // from raw instruction words.
    public static bool TryDecodeProgram(
        CpuContext ctx,
        ulong address,
        out Gen5ShaderProgram program,
        out string error)
    {
        ValidateDppControlVectors();
        var registry = GetFusedPrograms(ctx.Memory);
        registry.TryGet(address, out var fusedParts);
        if (fusedParts is not null)
        {
            return TryDecodeFusedProgram(ctx, address, fusedParts, out program, out error);
        }

        return TryDecodeProgramSegment(
            ctx,
            address,
            maximumBytes: null,
            // An unregistered S_SETPC transfers control to a separately
            // allocated code object.  The bytes that follow it in this
            // object are padding or unrelated data, so decoding through them
            // produces bogus ALU instructions and can reject an otherwise
            // valid shader.  Registered fused programs take the dedicated
            // path above and splice their continuation explicitly.
            stopAtSetProgramCounter: true,
            out program,
            out _,
            out error);
    }

    public static bool TryDecodeFunction(CpuContext context, ulong address,
        out Gen5ShaderProgram program, out string error)
    {
        if ((address & 3) != 0 || address >= (1ul << 48))
        {
            program = new Gen5ShaderProgram(address, []);
            error = "invalid function address";
            return false;
        }
        return TryDecodeProgramSegment(context, address, null, true,
            out program, out _, out error, coverForwardBranches: true);
    }

    public static bool TryDecodeFunction(CpuContext context, ulong address, uint maximumBytes,
        out Gen5ShaderProgram program, out string error)
    {
        program = new Gen5ShaderProgram(address, []);
        if (maximumBytes < sizeof(uint) || (address & 3) != 0 ||
            address > ulong.MaxValue - maximumBytes)
        {
            error = "invalid function address or byte limit";
            return false;
        }

        return TryDecodeProgramSegment(context, address, maximumBytes, true,
            out program, out _, out error, coverForwardBranches: true);
    }

    private enum ProgramTermination
    {
        None,
        EndProgram,
        SetProgramCounter,
    }

    private static bool TryDecodeFusedProgram(
        CpuContext ctx,
        ulong entryAddress,
        FusedShaderParts parts,
        out Gen5ShaderProgram program,
        out string error)
    {
        program = new Gen5ShaderProgram(entryAddress, []);
        if (!TryReadDeclaredShaderSize(ctx, parts.EntryHeaderAddress, out var entrySize, out error) ||
            !TryReadDeclaredShaderSize(
                ctx,
                parts.ContinuationHeaderAddress,
                out var continuationSize,
                out error))
        {
            return false;
        }

        var continuationDistance = unchecked(parts.ContinuationAddress - entryAddress);
        if (parts.ContinuationAddress == entryAddress || (continuationDistance & (sizeof(uint) - 1)) != 0)
        {
            error = $"invalid-fused-layout entry=0x{entryAddress:X} " +
                $"continuation=0x{parts.ContinuationAddress:X}";
            return false;
        }

        if (!TryDecodeProgramSegment(
                ctx,
                entryAddress,
                entrySize,
                stopAtSetProgramCounter: true,
                out var entryProgram,
                out var entryTermination,
                out error))
        {
            error = $"fused-entry: {error}";
            return false;
        }

        if (entryTermination != ProgramTermination.SetProgramCounter ||
            entryProgram.Instructions.Count == 0)
        {
            error = $"fused-entry-missing-setpc entry=0x{entryAddress:X}";
            return false;
        }

        if (!TryDecodeProgramSegment(
                ctx,
                parts.ContinuationAddress,
                continuationSize,
                stopAtSetProgramCounter: false,
                out var continuationProgram,
                out _,
                out error))
        {
            error = $"fused-continuation: {error}";
            return false;
        }

        var lastEntryInstruction = entryProgram.Instructions[^1];
        var entryEnd = (ulong)lastEntryInstruction.Pc + (ulong)lastEntryInstruction.Words.Count * sizeof(uint);
        var adjacent = parts.ContinuationAddress > entryAddress && continuationDistance <= uint.MaxValue &&
            continuationDistance >= entryEnd;
        var continuationPc = adjacent
            ? (uint)continuationDistance
            : (uint)((entryEnd + FusedContinuationAlignment - 1) & ~(FusedContinuationAlignment - 1));
        var instructions = new List<Gen5ShaderInstruction>(
            entryProgram.Instructions.Count + continuationProgram.Instructions.Count);
        instructions.AddRange(entryProgram.Instructions.Take(entryProgram.Instructions.Count - 1));

        var setProgramCounter = entryProgram.Instructions[^1];
        instructions.Add(setProgramCounter with
        {
            Encoding = Gen5ShaderEncoding.Sopp,
            Opcode = "SNop",
            Words = [0xBF800000u],
            Sources = [],
            Destinations = [],
            Control = null,
        });

        foreach (var instruction in continuationProgram.Instructions)
        {
            var rebasedPc = (ulong)continuationPc + instruction.Pc;
            if (rebasedPc > uint.MaxValue)
            {
                error = $"fused-continuation-pc-overflow pc=0x{instruction.Pc:X} " +
                    $"base=0x{continuationPc:X}";
                return false;
            }

            instructions.Add(adjacent
                ? instruction with { Pc = (uint)rebasedPc }
                : instruction with { Pc = (uint)rebasedPc, AddressOffset = unchecked(continuationDistance + instruction.ProgramOffset) });
        }

        program = new Gen5ShaderProgram(entryAddress, instructions) { IsFusedProgram = true };
        error = string.Empty;
        return true;
    }

    private static bool TryReadDeclaredShaderSize(
        CpuContext ctx,
        ulong headerAddress,
        out uint sizeBytes,
        out string error)
    {
        if (!TryReadUInt32(ctx, headerAddress + ShaderSizeOffset, out sizeBytes) ||
            sizeBytes == 0 ||
            (sizeBytes & (sizeof(uint) - 1)) != 0 ||
            sizeBytes > MaximumDeclaredShaderSizeBytes)
        {
            error = $"invalid-shader-size header=0x{headerAddress:X} size=0x{sizeBytes:X}";
            return false;
        }

        error = string.Empty;
        return true;
    }

    // S_SWAPPC_B64 stores the address of the next instruction in its destination. With the NULL destination
    // it stores nothing, so it jumps like S_SETPC_B64 does and the code after it is not reached from here.
    private static bool IsSwapProgramCounterWithoutReturnAddress(Gen5ShaderInstruction instruction) =>
        string.Equals(instruction.Opcode, "SSwappcB64", StringComparison.Ordinal) &&
        instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: NullScalarRegister }];

    private static bool TryDecodeProgramSegment(
        CpuContext ctx,
        ulong address,
        uint? maximumBytes,
        bool stopAtSetProgramCounter,
        out Gen5ShaderProgram program,
        out ProgramTermination termination,
        out string error,
        bool coverForwardBranches = false)
    {
        program = new Gen5ShaderProgram(address, []);
        termination = ProgramTermination.None;
        error = string.Empty;
        if (address == 0)
        {
            error = "missing";
            return false;
        }

        var instructions = new List<Gen5ShaderInstruction>();
        var instructionCount = 0;
        var lastPc = 0u;
        var lastWord = 0u;
        uint furthestForwardBranchTarget = 0;
        for (uint pc = 0;
             instructionCount < MaxInstructions &&
             (!maximumBytes.HasValue || pc < maximumBytes.Value);)
        {
            if (maximumBytes.HasValue && maximumBytes.Value - pc < sizeof(uint))
            {
                error = $"truncated-word pc=0x{pc:X} limit=0x{maximumBytes.Value:X}";
                return false;
            }

            if (!TryReadUInt32(ctx, address + pc, out var word))
            {
                error = $"read-failed pc=0x{pc:X}";
                return false;
            }

            lastPc = pc;
            lastWord = word;

            if (!TryDecodeInstruction(
                    ctx,
                    address,
                    pc,
                    word,
                    out var encoding,
                    out var name,
                    out var sizeDwords,
                    out error))
            {
                return false;
            }

            // S_CODE_END pads the space after the last instruction. Code placed after the final
            // S_ENDPGM that ends in a backward branch runs straight into it.
            if (string.Equals(name, "SCodeEnd", StringComparison.Ordinal))
            {
                if (pc < furthestForwardBranchTarget || instructions.Count == 0)
                {
                    error = $"code-end-inside-program pc=0x{pc:X} branchTarget=0x{furthestForwardBranchTarget:X}";
                    return false;
                }

                program = new Gen5ShaderProgram(address, instructions);
                termination = ProgramTermination.EndProgram;
                return true;
            }

            var instructionBytes = checked(sizeDwords * sizeof(uint));
            if (maximumBytes.HasValue && instructionBytes > maximumBytes.Value - pc)
            {
                error = $"truncated-instruction pc=0x{pc:X} dwords={sizeDwords} " +
                    $"limit=0x{maximumBytes.Value:X}";
                return false;
            }

            var words = new uint[sizeDwords];
            words[0] = word;
            for (uint wordIndex = 1; wordIndex < sizeDwords; wordIndex++)
            {
                if (!TryReadUInt32(ctx, address + pc + wordIndex * sizeof(uint), out words[wordIndex]))
                {
                    error = $"read-failed pc=0x{pc + wordIndex * sizeof(uint):X}";
                    return false;
                }
            }

            var instruction = CreateInstruction(pc, encoding, name, words);
            if (instruction.Control is Gen5GlobalMemoryControl
                {
                    UsesFlatAddress: true,
                })
            {
                instruction = ResolveFlatAddressBase(instructions, instruction);
            }
            instructions.Add(instruction);
            instructionCount++;

            if (Gen5IrBranchResolver.Instance.TryGetBranchTarget(
                    instruction,
                    out var branchTargetPc) &&
                branchTargetPc > instruction.Pc &&
                // S_CBRANCH_CDBG* only fire under an attached GPU debugger, and
                // the translators emit them as never taken. Their targets are
                // debugger stubs placed after S_ENDPGM that read devkit-only
                // addresses, so they must not extend the decoded program.
                !instruction.Opcode.StartsWith("SCbranchCdbg", StringComparison.Ordinal))
            {
                furthestForwardBranchTarget = Math.Max(
                    furthestForwardBranchTarget,
                    branchTargetPc);
            }

            pc += sizeDwords * sizeof(uint);
            // An unconditional backward S_BRANCH with no forward target
            // pending is the last reachable instruction too: compilers place
            // out-of-line loop blocks after S_ENDPGM, and the code object's
            // metadata trailer follows them.
            if ((string.Equals(name, "SEndpgm", StringComparison.Ordinal) ||
                 string.Equals(name, "SBranch", StringComparison.Ordinal)) &&
                pc > furthestForwardBranchTarget)
            {
                program = new Gen5ShaderProgram(address, instructions);
                termination = ProgramTermination.EndProgram;
                return true;
            }

            if (stopAtSetProgramCounter &&
                (string.Equals(name, "SSetpcB64", StringComparison.Ordinal) ||
                 IsSwapProgramCounterWithoutReturnAddress(instruction)) &&
                (!coverForwardBranches || pc > furthestForwardBranchTarget))
            {
                program = new Gen5ShaderProgram(address, instructions);
                termination = ProgramTermination.SetProgramCounter;
                return true;
            }
        }

        error = maximumBytes.HasValue
            ? $"unterminated limit=0x{maximumBytes.Value:X} pc=0x{lastPc:X} word=0x{lastWord:X8} count={instructionCount}"
            : $"unterminated pc=0x{lastPc:X} word=0x{lastWord:X8} count={instructionCount}";
        return false;
    }

    [Conditional("DEBUG")]
    private static void ValidateDppControlVectors()
    {
        if (System.Threading.Interlocked.Exchange(ref _dppVectorsValidated, 1) != 0)
        {
            return;
        }

        static (uint Lane, bool InRange) Resolve(uint control, uint lane)
        {
            var rowBase = lane & ~15u;
            var rowLane = lane & 15u;
            return control switch
            {
                >= 0x101 and <= 0x10F => (
                    rowBase + ((rowLane + (control & 15)) & 15),
                    rowLane + (control & 15) < 16),
                >= 0x111 and <= 0x11F => (
                    rowBase + ((rowLane - (control & 15)) & 15),
                    rowLane >= (control & 15)),
                >= 0x121 and <= 0x12F => (
                    rowBase + ((rowLane - (control & 15)) & 15),
                    true),
                0x140 => (rowBase + 15 - rowLane, true),
                0x141 => ((lane & ~7u) + 7 - (lane & 7), true),
                >= 0x150 and <= 0x15F => (rowBase + (control & 15), true),
                >= 0x160 and <= 0x16F => (rowBase + (rowLane ^ (control & 15)), true),
                _ => (lane, false),
            };
        }

        Debug.Assert(Resolve(0x101, 0) == (1u, true));
        Debug.Assert(Resolve(0x101, 15) == (0u, false));
        Debug.Assert(Resolve(0x112, 1) == (15u, false));
        Debug.Assert(Resolve(0x123, 0) == (13u, true));
        Debug.Assert(Resolve(0x140, 18) == (29u, true));
        Debug.Assert(Resolve(0x141, 9) == (14u, true));
        Debug.Assert(Resolve(0x153, 20) == (19u, true));
        Debug.Assert(Resolve(0x163, 22) == (21u, true));
        const uint dpp8 =
            (7u << 0) | (6u << 3) | (5u << 6) | (4u << 9) |
            (3u << 12) | (2u << 15) | (1u << 18) | (0u << 21);
        Debug.Assert(((dpp8 >> (0 * 3)) & 7) == 7);
        Debug.Assert(((dpp8 >> (7 * 3)) & 7) == 0);
    }

    private static bool TryDecodeInstruction(
        CpuContext ctx,
        ulong baseAddress,
        uint pc,
        uint word,
        out Gen5ShaderEncoding encoding,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        encoding = Gen5ShaderEncoding.Vop2;
        name = string.Empty;
        sizeDwords = 1;
        error = string.Empty;

        if ((word & 0x80000000u) == 0)
        {
            var vopOpcode = (word >> 25) & 0x3F;
            encoding = vopOpcode switch
            {
                0x3E => Gen5ShaderEncoding.Vopc,
                0x3F => Gen5ShaderEncoding.Vop1,
                _ => Gen5ShaderEncoding.Vop2,
            };
            return DecodeVop2(word, out name, out sizeDwords, out error);
        }

        if ((word & 0xF8000000u) == 0xC0000000u)
        {
            encoding = Gen5ShaderEncoding.Smrd;
            return DecodeSmrd(word, out name, out sizeDwords, out error);
        }

        if ((word & 0xC0000000u) == 0x80000000u)
        {
            var sopOpcode = (word >> 23) & 0x7F;
            encoding = sopOpcode switch
            {
                0x7D => Gen5ShaderEncoding.Sop1,
                0x7E => Gen5ShaderEncoding.Sopc,
                0x7F => Gen5ShaderEncoding.Sopp,
                >= 0x60 => Gen5ShaderEncoding.Sopk,
                _ => Gen5ShaderEncoding.Sop2,
            };
            return DecodeSop(word, out name, out sizeDwords, out error);
        }

        // gfx10 moved VOP3P (packed 16-bit math) to its own 0b110011000 prefix
        // (word0 top byte 0xCC), separate from the VOP3 block. Match the full
        // 9-bit prefix here, before the coarse major-opcode switch, so packed
        // instructions are not misread as one of the neighbouring encodings.
        if ((word & 0xFF800000u) == 0xCC000000u)
        {
            encoding = Gen5ShaderEncoding.Vop3p;
            if (!ctx.TryReadUInt32(baseAddress + pc + sizeof(uint), out var vop3pExtra))
            {
                error = $"vop3p-extra-read-failed pc=0x{pc:X}";
                return false;
            }

            return DecodeVop3p(word, vop3pExtra, out name, out sizeDwords, out error);
        }

        switch (word >> 26)
        {
            case 0x33:
                encoding = Gen5ShaderEncoding.Smem;
                return DecodeSmem(word, out name, out sizeDwords, out error);
            case 0x32:
                encoding = Gen5ShaderEncoding.Vintrp;
                return DecodeVintrp(word, out name, out sizeDwords, out error);
            case 0x34:
            case 0x35:
                encoding = Gen5ShaderEncoding.Vop3;
                if (!TryReadUInt32(ctx, baseAddress + pc + sizeof(uint), out var vop3Extra))
                {
                    error = $"vop3-extra-read-failed pc=0x{pc:X}";
                    return false;
                }

                return DecodeVop3(
                    word,
                    vop3Extra,
                    IsVop3BOpcode((word >> 16) & 0x3FF),
                    out name,
                    out sizeDwords,
                    out error);
            case 0x36:
                encoding = Gen5ShaderEncoding.Ds;
                return DecodeDs(word, out name, out sizeDwords, out error);
            case 0x37:
                encoding = Gen5ShaderEncoding.Flat;
                return DecodeFlat(word, out name, out sizeDwords, out error);
            case 0x38:
                encoding = Gen5ShaderEncoding.Mubuf;
                if (!TryReadUInt32(ctx, baseAddress + pc + sizeof(uint), out var mubufExtra))
                {
                    error = $"mubuf-extra-read-failed pc=0x{pc:X}";
                    return false;
                }

                return DecodeMubuf(word, mubufExtra, out name, out sizeDwords, out error);
            case 0x3A:
                encoding = Gen5ShaderEncoding.Mtbuf;
                if (!TryReadUInt32(ctx, baseAddress + pc + sizeof(uint), out var mtbufExtra))
                {
                    error = $"mtbuf-extra-read-failed pc=0x{pc:X}";
                    return false;
                }

                return DecodeMtbuf(word, mtbufExtra, out name, out sizeDwords, out error);
            case 0x3C:
                encoding = Gen5ShaderEncoding.Mimg;
                return DecodeMimg(word, out name, out sizeDwords, out error);
            case 0x3D:
                encoding = Gen5ShaderEncoding.Smem;
                return DecodeSmem(word, out name, out sizeDwords, out error);
            case 0x3E:
                encoding = Gen5ShaderEncoding.Exp;
                name = "Exp";
                sizeDwords = 2;
                return true;
            case 0x3F:
                encoding = Gen5ShaderEncoding.Vop3p;
                return DecodeRaw2(word, "Vop3p", out name, out sizeDwords, out error);
            default:
                error = $"unknown-top pc=0x{pc:X} word=0x{word:X8}";
                return false;
        }
    }

    // Kept beside the production decoder so offline compatibility tools use
    // precisely the same opcode tables and instruction-width rules as runtime
    // shader translation.
    internal static bool TryDecodeInstructionForPreflight(
        CpuContext ctx,
        uint pc,
        uint word,
        out string name,
        out uint sizeDwords,
        out string error) =>
        TryDecodeInstruction(
            ctx,
            0,
            pc,
            word,
            out _,
            out name,
            out sizeDwords,
            out error);

    private static bool DecodeSop(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 23) & 0x7F;
        return opcode switch
        {
            0x7D => DecodeSop1(word, out name, out sizeDwords, out error),
            0x7E => DecodeSopc(word, out name, out sizeDwords, out error),
            0x7F => DecodeSopp(word, out name, out sizeDwords, out error),
            >= 0x60 => DecodeSopk(word, out name, out sizeDwords, out error),
            _ => DecodeSop2(word, out name, out sizeDwords, out error),
        };
    }

    private static bool DecodeSop1(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 8) & 0xFF;
        var src0 = word & 0xFF;
        sizeDwords = 1 + (src0 == 0xFF ? 1u : 0u);
        error = string.Empty;
        name = opcode switch
        {
            0x03 => "SMovB32",
            0x04 => "SMovB64",
            0x05 => "SCmovB32",
            0x06 => "SCmovB64",
            0x07 => "SNotB32",
            0x08 => "SNotB64",
            0x09 => "SWqmB32",
            0x0A => "SWqmB64",
            0x0B => "SBrevB32",
            0x0C => "SBrevB64",
            0x0D => "SBcnt0I32B32",
            0x0E => "SBcnt0I32B64",
            0x0F => "SBcnt1I32B32",
            0x10 => "SBcnt1I32B64",
            0x11 => "SFF0I32B32",
            0x12 => "SFF0I32B64",
            0x13 => "SFF1I32B32",
            0x14 => "SFF1I32B64",
            0x15 => "SFlbitI32B32",
            0x16 => "SFlbitI32B64",
            0x17 => "SFlbitI32",
            0x18 => "SFlbitI32I64",
            0x19 => "SSextI32I8",
            0x1A => "SSextI32I16",
            0x1B => "SBitset0B32",
            0x1C => "SBitset0B64",
            0x1D => "SBitset1B32",
            0x1E => "SBitset1B64",
            0x1F => "SGetpcB64",
            0x20 => "SSetpcB64",
            0x21 => "SSwappcB64",
            0x22 => "SRfeB64",
            0x24 => "SAndSaveexecB64",
            0x25 => "SOrSaveexecB64",
            0x26 => "SXorSaveexecB64",
            0x27 => "SAndn2SaveexecB64",
            0x28 => "SOrn2SaveexecB64",
            0x29 => "SNandSaveexecB64",
            0x2A => "SNorSaveexecB64",
            0x2B => "SXnorSaveexecB64",
            0x2C => "SQuadmaskB32",
            0x2D => "SQuadmaskB64",
            0x2E => "SMovrelsB32",
            0x2F => "SMovrelsB64",
            0x30 => "SMovreldB32",
            0x31 => "SMovreldB64",
            0x34 => "SAbsI32",
            0x37 => "SAndn1SaveexecB64",
            0x38 => "SOrn1SaveexecB64",
            0x39 => "SAndn1WrexecB64",
            0x3A => "SAndn2WrexecB64",
            0x3B => "SBitreplicateB64B32",
            0x3C => "SAndSaveexecB32",
            0x3D => "SOrSaveexecB32",
            0x3E => "SXorSaveexecB32",
            0x3F => "SAndn2SaveexecB32",
            0x40 => "SOrn2SaveexecB32",
            0x41 => "SNandSaveexecB32",
            0x42 => "SNorSaveexecB32",
            0x43 => "SXnorSaveexecB32",
            0x44 => "SAndn1SaveexecB32",
            0x45 => "SOrn1SaveexecB32",
            0x46 => "SAndn1WrexecB32",
            0x47 => "SAndn2WrexecB32",
            0x49 => "SMovrelsd2B32",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-sop1 op=0x{opcode:X2} word=0x{word:X8}", out error);
    }

    private static bool DecodeSop2(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 23) & 0x7F;
        var src0 = word & 0xFF;
        var src1 = (word >> 8) & 0xFF;
        sizeDwords = src0 == 0xFF || src1 == 0xFF ? 2u : 1u;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "SAddU32",
            0x01 => "SSubU32",
            0x02 => "SAddI32",
            0x03 => "SSubI32",
            0x04 => "SAddcU32",
            0x05 => "SSubbU32",
            0x06 => "SMinI32",
            0x07 => "SMinU32",
            0x08 => "SMaxI32",
            0x09 => "SMaxU32",
            0x0A => "SCselectB32",
            0x0B => "SCselectB64",
            0x0E => "SAndB32",
            0x0F => "SAndB64",
            0x10 => "SOrB32",
            0x11 => "SOrB64",
            0x12 => "SXorB32",
            0x13 => "SXorB64",
            0x14 => "SAndn2B32",
            0x15 => "SAndn2B64",
            0x16 => "SOrn2B32",
            0x17 => "SOrn2B64",
            0x18 => "SNandB32",
            0x19 => "SNandB64",
            0x1A => "SNorB32",
            0x1B => "SNorB64",
            0x1C => "SXnorB32",
            0x1D => "SXnorB64",
            0x1E => "SLshlB32",
            0x1F => "SLshlB64",
            0x20 => "SLshrB32",
            0x21 => "SLshrB64",
            0x22 => "SAshrI32",
            0x23 => "SAshrI64",
            0x24 => "SBfmB32",
            0x25 => "SBfmB64",
            0x26 => "SMulI32",
            0x27 => "SBfeU32",
            0x28 => "SBfeI32",
            0x29 => "SBfeU64",
            0x2A => "SBfeI64",
            0x2C => "SAbsdiffI32",
            0x2E => "SLshl1AddU32",
            0x2F => "SLshl2AddU32",
            0x30 => "SLshl3AddU32",
            0x31 => "SLshl4AddU32",
            0x32 => "SPackLlB32B16",
            0x33 => "SPackLhB32B16",
            0x34 => "SPackHhB32B16",
            0x35 => "SMulHiU32",
            0x36 => "SMulHiI32",
            // RDNA scalar floating-point forms use the extended SOP2 opcode
            // range. They keep the raw SGPR bit representation and do not
            // alter SCC.
            0x40 => "SAddF32",
            0x41 => "SSubF32",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-sop2 op=0x{opcode:X2}", out error);
    }

    private static bool DecodeSopc(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 16) & 0x7F;
        var src0 = word & 0xFF;
        var src1 = (word >> 8) & 0xFF;
        sizeDwords = src0 == 0xFF || src1 == 0xFF ? 2u : 1u;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "SCmpEqI32",
            0x01 => "SCmpLgI32",
            0x02 => "SCmpGtI32",
            0x03 => "SCmpGeI32",
            0x04 => "SCmpLtI32",
            0x05 => "SCmpLeI32",
            0x06 => "SCmpEqU32",
            0x07 => "SCmpLgU32",
            0x08 => "SCmpGtU32",
            0x09 => "SCmpGeU32",
            0x0A => "SCmpLtU32",
            0x0B => "SCmpLeU32",
            0x0C => "SBitcmp0B32",
            0x0D => "SBitcmp1B32",
            0x0E => "SBitcmp0B64",
            0x0F => "SBitcmp1B64",
            0x12 => "SCmpEqU64",
            0x13 => "SCmpLgU64",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-sopc op=0x{opcode:X2}", out error);
    }

    private static bool DecodeSopp(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 16) & 0x7F;
        sizeDwords = 1;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "SNop",
            0x01 => "SEndpgm",
            0x02 => "SBranch",
            0x03 => "SWakeup",
            0x04 => "SCbranchScc0",
            0x05 => "SCbranchScc1",
            0x06 => "SCbranchVccz",
            0x07 => "SCbranchVccnz",
            0x08 => "SCbranchExecz",
            0x09 => "SCbranchExecnz",
            0x0A => "SBarrier",
            0x0B => "SSetkill",
            0x0C => "SWaitcnt",
            0x0D => "SSethalt",
            0x0E => "SSleep",
            0x0F => "SSetprio",
            0x10 => "SSendmsg",
            0x11 => "SSendmsghalt",
            0x12 => "STrap",
            0x13 => "SIcacheInv",
            0x14 => "SIncperflevel",
            0x15 => "SDecperflevel",
            0x16 => "STtraceData",
            0x17 => "SCbranchCdbgsys",
            0x18 => "SCbranchCdbguser",
            0x19 => "SCbranchCdbgsysOrUser",
            0x1A => "SCbranchCdbgsysAndUser",
            0x1B => "SEndpgmSaved",
            // Ends the wave after signalling that it left its POPS ordered section.
            // Without POPS ordering on the host this is S_ENDPGM.
            0x1E => "SEndpgm",
            0x1F => "SCodeEnd",
            0x20 => "SInstPrefetch",
            0x21 => "SClause",
            0x22 => "SWaitIdle",
            0x23 => "SWaitcntDepctr",
            0x24 => "SRoundMode",
            0x25 => "SDenormMode",
            0x28 => "STtracedataImm",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-sopp op=0x{opcode:X2}", out error);
    }

    private static bool DecodeSopk(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = ((word >> 23) & 0x7F) - 0x60;
        // S_SETREG_IMM32_B32 carries its value in a literal dword.
        sizeDwords = opcode == 0x15 ? 2u : 1u;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "SMovkI32",
            0x01 => "SVersion",
            0x02 => "SCmovkI32",
            0x03 => "SCmpkEqI32",
            0x04 => "SCmpkLgI32",
            0x05 => "SCmpkGtI32",
            0x06 => "SCmpkGeI32",
            0x07 => "SCmpkLtI32",
            0x08 => "SCmpkLeI32",
            0x09 => "SCmpkEqU32",
            0x0A => "SCmpkLgU32",
            0x0B => "SCmpkGtU32",
            0x0C => "SCmpkGeU32",
            0x0D => "SCmpkLtU32",
            0x0E => "SCmpkLeU32",
            0x0F => "SAddkI32",
            0x10 => "SMulkI32",
            0x12 => "SGetregB32",
            0x13 => "SSetregB32",
            0x15 => "SSetregImm32B32",
            0x16 => "SCallB64",
            // RDNA2 uses four SOPK forms to wait for one counter.
            // The selected counter does not change the translated operation.
            0x17 or 0x18 or 0x19 or 0x1A => "SWaitcnt",
            0x1B => "SSubvectorLoopBegin",
            0x1C => "SSubvectorLoopEnd",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-sopk op=0x{opcode:X2}", out error);
    }

    private static bool DecodeVop1(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 9) & 0xFF;
        var src0 = word & 0x1FF;
        sizeDwords = src0 is 0xE9 or 0xEA or 0xF9 or 0xFA or 0xFF ? 2u : 1u;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "VNop",
            0x01 => "VMovB32",
            0x02 => "VReadfirstlaneB32",
            0x03 => "VCvtI32F64",
            0x04 => "VCvtF64I32",
            0x05 => "VCvtF32I32",
            0x06 => "VCvtF32U32",
            0x07 => "VCvtU32F32",
            0x08 => "VCvtI32F32",
            0x0A => "VCvtF16F32",
            0x0B => "VCvtF32F16",
            0x0C => "VCvtRpiI32F32",
            0x0D => "VCvtFlrI32F32",
            0x0E => "VCvtOffF32I4",
            0x0F => "VCvtF32F64",
            0x10 => "VCvtF64F32",
            0x11 => "VCvtF32Ubyte0",
            0x12 => "VCvtF32Ubyte1",
            0x13 => "VCvtF32Ubyte2",
            0x14 => "VCvtF32Ubyte3",
            0x15 => "VCvtU32F64",
            0x16 => "VCvtF64U32",
            0x17 => "VTruncF64",
            0x18 => "VCeilF64",
            0x19 => "VRndneF64",
            0x1A => "VFloorF64",
            0x1B => "VPipeflush",
            0x20 => "VFractF32",
            0x21 => "VTruncF32",
            0x22 => "VCeilF32",
            0x23 => "VRndneF32",
            0x24 => "VFloorF32",
            0x25 => "VExpF32",
            0x27 => "VLogF32",
            0x2A => "VRcpF32",
            0x2B => "VRcpIflagF32",
            0x2E => "VRsqF32",
            0x2F => "VRcpF64",
            0x31 => "VRsqF64",
            0x33 => "VSqrtF32",
            0x34 => "VSqrtF64",
            0x35 => "VSinF32",
            0x36 => "VCosF32",
            0x37 => "VNotB32",
            0x38 => "VBfrevB32",
            0x39 => "VFfbhU32",
            0x3A => "VFfblB32",
            0x3B => "VFfbhI32",
            0x3C => "VFrexpExpI32F64",
            0x3D => "VFrexpMantF64",
            0x3E => "VFractF64",
            0x3F => "VFrexpExpI32F32",
            0x40 => "VFrexpMantF32",
            0x41 => "VClrexcp",
            0x42 => "VMovreldB32",
            0x43 => "VMovrelsB32",
            0x44 => "VMovrelsdB32",
            0x48 => "VMovrelsd2B32",
            0x50 => "VCvtF16U16",
            0x51 => "VCvtF16I16",
            0x52 => "VCvtU16F16",
            0x53 => "VCvtI16F16",
            0x54 => "VRcpF16",
            0x55 => "VSqrtF16",
            0x56 => "VRsqF16",
            0x57 => "VLogF16",
            0x58 => "VExpF16",
            0x59 => "VFrexpMantF16",
            0x5A => "VFrexpExpI16F16",
            0x5B => "VFloorF16",
            0x5C => "VCeilF16",
            0x5D => "VTruncF16",
            0x5E => "VRndneF16",
            0x5F => "VFractF16",
            0x60 => "VSinF16",
            0x61 => "VCosF16",
            0x62 => "VSatPkU8I16",
            0x63 => "VCvtNormI16F16",
            0x64 => "VCvtNormU16F16",
            0x65 => "VSwapB32",
            0x68 => "VSwaprelB32",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-vop1 op=0x{opcode:X2}", out error);
    }

    private static bool DecodeVop2(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 25) & 0x3F;
        if (opcode == 0x3E)
        {
            return DecodeVopc(word, out name, out sizeDwords, out error);
        }

        if (opcode == 0x3F)
        {
            return DecodeVop1(word, out name, out sizeDwords, out error);
        }

        var src0 = word & 0x1FF;
        sizeDwords = opcode is 0x20 or 0x21 or 0x2C or 0x2D or 0x37 or 0x38 ||
            src0 is 0xE9 or 0xEA or 0xF9 or 0xFA or 0xFF ? 2u : 1u;
        error = string.Empty;
        name = opcode switch
        {
            // Some Gen5 shader streams use the VOP2 opcode-zero form as a
            // padding/no-op instruction. Keep it as VNop so the translator
            // can consume the instruction without manufacturing a result.
            0x00 => "VNop",
            0x01 => "VCndmaskB32",
            0x02 => "VDot2cF32F16",
            0x03 => "VAddF32",
            0x04 => "VSubF32",
            0x05 => "VSubrevF32",
            // GFX10.1 V_MAC_LEGACY_F32 / V_MUL_LEGACY_F32 (DX9 zero rule); GFX10.3
            // reuses 0x06 for the fused V_FMAC_LEGACY_F32. Both run as one fma here.
            0x06 => "VMacLegacyF32",
            0x07 => "VMulLegacyF32",
            0x08 => "VMulF32",
            0x09 => "VMulI32I24",
            0x0A => "VMulHiI32I24",
            0x0B => "VMulU32U24",
            0x0C => "VMulHiU32U24",
            0x0D => "VDot4cI32I8",
            0x0F => "VMinF32",
            0x10 => "VMaxF32",
            0x11 => "VMinI32",
            0x12 => "VMaxI32",
            0x13 => "VMinU32",
            0x14 => "VMaxU32",
            0x15 => "VLshrB32",
            0x16 => "VLshrrevB32",
            0x17 => "VAshrI32",
            0x18 => "VAshrrevI32",
            0x19 => "VLshlB32",
            0x1A => "VLshlrevB32",
            0x1B => "VAndB32",
            0x1C => "VOrB32",
            0x1D => "VXorB32",
            0x1E => "VXnorB32",
            0x1F => "VMacF32",
            0x20 => "VMadMkF32",
            0x21 => "VMadAkF32",
            0x22 => "VBcntU32B32",
            0x23 => "VMbcntLoU32B32",
            0x24 => "VMbcntHiU32B32",
            0x25 => "VAddI32",
            0x26 => "VSubI32",
            0x27 => "VSubrevI32",
            0x28 => "VAddcU32",
            0x29 => "VSubbU32",
            0x2A => "VSubbrevU32",
            0x2B => "VFmacF32",
            0x2C => "VFmaMkF32",
            0x2D => "VFmaAkF32",
            0x2F => "VCvtPkrtzF16F32",
            0x30 => "VCvtPkU16U32",
            0x31 => "VCvtPkI16I32",
            0x32 => "VAddF16",
            0x33 => "VSubF16",
            0x34 => "VSubrevF16",
            0x35 => "VMulF16",
            0x36 => "VFmacF16",
            // RDNA encodes FMAMK/FMAAK as VOP2 instructions with a literal
            // dword immediately following the opcode word.
            0x37 => "VFmaMkF16",
            0x38 => "VFmaAkF16",
            0x39 => "VMaxF16",
            0x3A => "VMinF16",
            0x3B => "VLdexpF16",
            0x3C => "VPkFmacF16",
            _ => string.Empty,
        };

        return FinishDecode(
            name,
            $"unknown-vop2 op=0x{opcode:X2} word=0x{word:X8}",
            out error);
    }

    private static bool DecodeVopc(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 17) & 0xFF;
        var src0 = word & 0x1FF;
        sizeDwords = src0 is 0xE9 or 0xEA or 0xF9 or 0xFA or 0xFF ? 2u : 1u;
        error = string.Empty;
        name = VopcOpcodeName(opcode);
        return FinishDecode(name, $"unknown-vopc op=0x{opcode:X2}", out error);
    }

    // VOPC names, shared by the VOP3 encoding of the same compares (VOP3 opcodes 0x000-0x0FF).
    private static string VopcOpcodeName(uint opcode) =>
        opcode switch
        {
            0x00 => "VCmpFF32",
            0x01 => "VCmpLtF32",
            0x02 => "VCmpEqF32",
            0x03 => "VCmpLeF32",
            0x04 => "VCmpGtF32",
            0x05 => "VCmpLgF32",
            0x06 => "VCmpGeF32",
            0x07 => "VCmpOF32",
            0x08 => "VCmpUF32",
            0x09 => "VCmpNgeF32",
            0x0A => "VCmpNlgF32",
            0x0B => "VCmpNgtF32",
            0x0C => "VCmpNleF32",
            0x0D => "VCmpNeqF32",
            0x0E => "VCmpNltF32",
            0x0F => "VCmpTruF32",
            0x10 => "VCmpxFF32",
            0x11 => "VCmpxLtF32",
            0x12 => "VCmpxEqF32",
            0x13 => "VCmpxLeF32",
            0x14 => "VCmpxGtF32",
            0x15 => "VCmpxLgF32",
            0x16 => "VCmpxGeF32",
            0x17 => "VCmpxOF32",
            0x18 => "VCmpxUF32",
            0x19 => "VCmpxNgeF32",
            0x1A => "VCmpxNlgF32",
            0x1B => "VCmpxNgtF32",
            0x1C => "VCmpxNleF32",
            0x1D => "VCmpxNeqF32",
            0x1E => "VCmpxNltF32",
            0x1F => "VCmpxTruF32",
            0x20 => "VCmpFF64",
            0x21 => "VCmpLtF64",
            0x22 => "VCmpEqF64",
            0x23 => "VCmpLeF64",
            0x24 => "VCmpGtF64",
            0x25 => "VCmpLgF64",
            0x26 => "VCmpGeF64",
            0x27 => "VCmpOF64",
            0x28 => "VCmpUF64",
            0x29 => "VCmpNgeF64",
            0x2A => "VCmpNlgF64",
            0x2B => "VCmpNgtF64",
            0x2C => "VCmpNleF64",
            0x2D => "VCmpNeqF64",
            0x2E => "VCmpNltF64",
            0x2F => "VCmpTruF64",
            0x30 => "VCmpxFF64",
            0x31 => "VCmpxLtF64",
            0x32 => "VCmpxEqF64",
            0x33 => "VCmpxLeF64",
            0x34 => "VCmpxGtF64",
            0x35 => "VCmpxLgF64",
            0x36 => "VCmpxGeF64",
            0x37 => "VCmpxOF64",
            0x38 => "VCmpxUF64",
            0x39 => "VCmpxNgeF64",
            0x3A => "VCmpxNlgF64",
            0x3B => "VCmpxNgtF64",
            0x3C => "VCmpxNleF64",
            0x3D => "VCmpxNeqF64",
            0x3E => "VCmpxNltF64",
            0x3F => "VCmpxTruF64",
            0x80 => "VCmpFI32",
            0x81 => "VCmpLtI32",
            0x82 => "VCmpEqI32",
            0x83 => "VCmpLeI32",
            0x84 => "VCmpGtI32",
            0x85 => "VCmpNeI32",
            0x86 => "VCmpGeI32",
            0x87 => "VCmpTI32",
            0x88 => "VCmpClassF32",
            0x89 => "VCmpLtI16",
            0x8A => "VCmpEqI16",
            0x8B => "VCmpLeI16",
            0x8C => "VCmpGtI16",
            0x8D => "VCmpNeI16",
            0x8E => "VCmpGeI16",
            0x8F => "VCmpClassF16",
            0x98 => "VCmpxClassF32",
            0xA9 => "VCmpLtU16",
            0xAA => "VCmpEqU16",
            0xAB => "VCmpLeU16",
            0xAC => "VCmpGtU16",
            0xAD => "VCmpNeU16",
            0xAE => "VCmpGeU16",
            0x99 => "VCmpxLtI16",
            0x9A => "VCmpxEqI16",
            0x9B => "VCmpxLeI16",
            0x9C => "VCmpxGtI16",
            0x9D => "VCmpxNeI16",
            0x9E => "VCmpxGeI16",
            0x9F => "VCmpxClassF16",
            0xB9 => "VCmpxLtU16",
            0xBA => "VCmpxEqU16",
            0xBB => "VCmpxLeU16",
            0xBC => "VCmpxGtU16",
            0xBD => "VCmpxNeU16",
            0xBE => "VCmpxGeU16",
            0x90 => "VCmpxFI32",
            0x91 => "VCmpxLtI32",
            0x92 => "VCmpxEqI32",
            0x93 => "VCmpxLeI32",
            0x94 => "VCmpxGtI32",
            0x95 => "VCmpxNeI32",
            0x96 => "VCmpxGeI32",
            0x97 => "VCmpxTI32",
            0xA0 => "VCmpFI64",
            0xA1 => "VCmpLtI64",
            0xA2 => "VCmpEqI64",
            0xA3 => "VCmpLeI64",
            0xA4 => "VCmpGtI64",
            0xA5 => "VCmpNeI64",
            0xA6 => "VCmpGeI64",
            0xA7 => "VCmpTI64",
            0xA8 => "VCmpClassF64",
            0xB0 => "VCmpxFI64",
            0xB1 => "VCmpxLtI64",
            0xB2 => "VCmpxEqI64",
            0xB3 => "VCmpxLeI64",
            0xB4 => "VCmpxGtI64",
            0xB5 => "VCmpxNeI64",
            0xB6 => "VCmpxGeI64",
            0xB7 => "VCmpxTI64",
            0xB8 => "VCmpxClassF64",
            0xC0 => "VCmpFU32",
            0xC1 => "VCmpLtU32",
            0xC2 => "VCmpEqU32",
            0xC3 => "VCmpLeU32",
            0xC4 => "VCmpGtU32",
            0xC5 => "VCmpNeU32",
            0xC6 => "VCmpGeU32",
            0xC7 => "VCmpTU32",
            0xC8 => "VCmpFF16",
            0xC9 => "VCmpLtF16",
            0xCA => "VCmpEqF16",
            0xCB => "VCmpLeF16",
            0xCC => "VCmpGtF16",
            0xCD => "VCmpLgF16",
            0xCE => "VCmpGeF16",
            0xCF => "VCmpOF16",
            0xD0 => "VCmpxFU32",
            0xD1 => "VCmpxLtU32",
            0xD2 => "VCmpxEqU32",
            0xD3 => "VCmpxLeU32",
            0xD4 => "VCmpxGtU32",
            0xD5 => "VCmpxNeU32",
            0xD6 => "VCmpxGeU32",
            0xD7 => "VCmpxTU32",
            0xD8 => "VCmpxFF16",
            0xD9 => "VCmpxLtF16",
            0xDA => "VCmpxEqF16",
            0xDB => "VCmpxLeF16",
            0xDC => "VCmpxGtF16",
            0xDD => "VCmpxLgF16",
            0xDE => "VCmpxGeF16",
            0xDF => "VCmpxOF16",
            0xE0 => "VCmpFU64",
            0xE1 => "VCmpLtU64",
            0xE2 => "VCmpEqU64",
            0xE3 => "VCmpLeU64",
            0xE4 => "VCmpGtU64",
            0xE5 => "VCmpNeU64",
            0xE6 => "VCmpGeU64",
            0xE7 => "VCmpTU64",
            0xE8 => "VCmpUF16",
            0xE9 => "VCmpNgeF16",
            0xEA => "VCmpNlgF16",
            0xEB => "VCmpNgtF16",
            0xEC => "VCmpNleF16",
            0xED => "VCmpNeqF16",
            0xEE => "VCmpNltF16",
            0xEF => "VCmpTruF16",
            0xF0 => "VCmpxFU64",
            0xF1 => "VCmpxLtU64",
            0xF2 => "VCmpxEqU64",
            0xF3 => "VCmpxLeU64",
            0xF4 => "VCmpxGtU64",
            0xF5 => "VCmpxNeU64",
            0xF6 => "VCmpxGeU64",
            0xF7 => "VCmpxTU64",
            0xF8 => "VCmpxUF16",
            0xF9 => "VCmpxNgeF16",
            0xFA => "VCmpxNlgF16",
            0xFB => "VCmpxNgtF16",
            0xFC => "VCmpxNleF16",
            0xFD => "VCmpxNeqF16",
            0xFE => "VCmpxNltF16",
            0xFF => "VCmpxTruF16",
            _ => string.Empty,
        };

    private static bool DecodeVop3(
        uint word,
        uint extra,
        bool isVop3B,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        var opcode = (word >> 16) & 0x3FF;
        var src0 = extra & 0x1FF;
        var src1 = (extra >> 9) & 0x1FF;
        var src2 = (extra >> 18) & 0x1FF;
        sizeDwords = src0 == 0xFF || src1 == 0xFF || src2 == 0xFF ? 3u : 2u;
        error = string.Empty;
        if (!isVop3B && Vop3InterpolationName(opcode) is { } interpolation)
        {
            // src0 names the attribute, not an operand, so it never selects a literal.
            sizeDwords = 2;
            name = interpolation;
            return true;
        }
        if (!isVop3B && opcode < 0x100 && VopcOpcodeName(opcode) is { Length: > 0 } compare)
        {
            name = compare;
            return true;
        }

        name = isVop3B
            ? opcode switch
            {
                0x128 => "VAddCoCiU32",
                0x129 => "VSubCoCiU32",
                0x12A => "VSubrevCoCiU32",
                0x16D => "VDivScaleF32",
                0x16E => "VDivScaleF64",
                0x177 => "VMadI64I32",
                0x30F => "VAddCoU32",
                0x310 => "VSubCoU32",
                0x319 => "VSubrevCoU32",
                0x176 => "VMadU64U32",
                _ => $"Vop3bRaw{opcode:X3}",
            }
            : opcode switch
        {
            0x0B5 => "VCmpxNeI64",
            0x101 => "VCndmaskB32",
            0x103 => "VAddF32",
            0x104 => "VSubF32",
            0x105 => "VSubrevF32",
            0x106 => "VMacLegacyF32",
            0x107 => "VMulLegacyF32",
            0x108 => "VMulF32",
            0x109 => "VMulI32I24",
            0x10A => "VMulHiI32I24",
            0x10B => "VMulU32U24",
            0x10C => "VMulHiU32U24",
            0x10F => "VMinF32",
            0x110 => "VMaxF32",
            0x111 => "VMinI32",
            0x112 => "VMaxI32",
            0x113 => "VMinU32",
            0x114 => "VMaxU32",
            0x116 => "VLshrrevB32",
            0x118 => "VAshrrevI32",
            0x11A => "VLshlrevB32",
            0x11B => "VAndB32",
            0x11C => "VOrB32",
            0x11D => "VXorB32",
            0x11E => "VXnorB32",
            0x11F => "VMacF32",
            // V_ADD/SUB/SUBREV_NC_U32: the clamp bit saturates unsigned.
            0x125 => "VAddU32",
            0x126 => "VSubU32",
            0x127 => "VSubrevU32",
            0x12B => "VFmacF32",
            0x12F => "VCvtPkrtzF16F32",
            0x132 => "VAddF16",
            0x133 => "VSubF16",
            0x134 => "VSubrevF16",
            0x135 => "VMulF16",
            0x136 => "VFmacF16",
            0x139 => "VMaxF16",
            0x13A => "VMinF16",
            0x13B => "VLdexpF16",
            0x140 => "VMadLegacyF32",
            0x141 => "VMadF32",
            0x142 => "VMadI32I24",
            0x143 => "VMadU32U24",
            0x144 => "VCubeidF32",
            0x145 => "VCubescF32",
            0x146 => "VCubetcF32",
            0x147 => "VCubemaF32",
            0x14A => "VBfiB32",
            0x14E => "VAlignbitB32",
            0x14F => "VAlignbyteB32",
            0x14B => "VFmaF32",
            0x14C => "VFmaF64",
            0x14D => "VLerpU8",
            0x150 => "VMullitF32",
            0x151 => "VMin3F32",
            0x152 => "VMin3I32",
            0x153 => "VMin3U32",
            0x154 => "VMax3F32",
            0x155 => "VMax3I32",
            0x156 => "VMax3U32",
            0x157 => "VMed3F32",
            0x158 => "VMed3I32",
            0x159 => "VMed3U32",
            0x15A => "VSadU8",
            0x15B => "VSadHiU8",
            0x15C => "VSadU16",
            0x15D => "VSadU32",
            0x15E => "VCvtPkU8F32",
            0x15F => "VDivFixupF32",
            0x160 => "VDivFixupF64",
            0x148 => "VBfeU32",
            0x149 => "VBfeI32",
            0x164 => "VAddF64",
            0x165 => "VMulF64",
            0x166 => "VMinF64",
            0x167 => "VMaxF64",
            0x168 => "VLdexpF64",
            0x169 => "VMulLoU32",
            0x16A => "VMulHiU32",
            0x16B => "VMulLoI32",
            0x16C => "VMulHiI32",
            0x16F => "VDivFmasF32",
            0x170 => "VDivFmasF64",
            0x171 => "VMsadU8",
            0x172 => "VQsadPkU16U8",
            0x173 => "VMqsadPkU16U8",
            0x174 => "VTrigPreopF64",
            0x175 => "VMqsadU32U8",
            // VOP3-encoded VOP1 (opcode + 0x180).
            0x181 => "VMovB32",
            0x183 => "VCvtI32F64",
            0x184 => "VCvtF64I32",
            0x185 => "VCvtF32I32",
            0x186 => "VCvtF32U32",
            0x187 => "VCvtU32F32",
            0x188 => "VCvtI32F32",
            0x18A => "VCvtF16F32",
            0x18B => "VCvtF32F16",
            0x18C => "VCvtRpiI32F32",
            0x18D => "VCvtFlrI32F32",
            0x18E => "VCvtOffF32I4",
            0x18F => "VCvtF32F64",
            0x190 => "VCvtF64F32",
            0x191 => "VCvtF32Ubyte0",
            0x192 => "VCvtF32Ubyte1",
            0x193 => "VCvtF32Ubyte2",
            0x194 => "VCvtF32Ubyte3",
            0x195 => "VCvtU32F64",
            0x196 => "VCvtF64U32",
            0x197 => "VTruncF64",
            0x198 => "VCeilF64",
            0x199 => "VRndneF64",
            0x19A => "VFloorF64",
            0x1A0 => "VFractF32",
            0x1A1 => "VTruncF32",
            0x1A2 => "VCeilF32",
            0x1A3 => "VRndneF32",
            0x1A4 => "VFloorF32",
            0x1A5 => "VExpF32",
            0x1A7 => "VLogF32",
            0x1AA => "VRcpF32",
            0x1AB => "VRcpIflagF32",
            0x1AE => "VRsqF32",
            0x1AF => "VRcpF64",
            0x1B1 => "VRsqF64",
            0x1B3 => "VSqrtF32",
            0x1B4 => "VSqrtF64",
            0x1B5 => "VSinF32",
            0x1B6 => "VCosF32",
            0x1B7 => "VNotB32",
            0x1B8 => "VBfrevB32",
            0x1B9 => "VFfbhU32",
            0x1BA => "VFfblB32",
            0x1BB => "VFfbhI32",
            0x1BC => "VFrexpExpI32F64",
            0x1BD => "VFrexpMantF64",
            0x1BE => "VFractF64",
            0x1BF => "VFrexpExpI32F32",
            0x1C0 => "VFrexpMantF32",
            0x1C2 => "VMovreldB32",
            0x1C3 => "VMovrelsB32",
            0x1C4 => "VMovrelsdB32",
            0x1C8 => "VMovrelsd2B32",
            0x1D0 => "VCvtF16U16",
            0x1D1 => "VCvtF16I16",
            0x1D3 => "VCvtI16F16",
            0x1D9 => "VFrexpMantF16",
            0x1DA => "VFrexpExpI16F16",
            0x1E2 => "VSatPkU8I16",
            0x1E3 => "VCvtNormI16F16",
            0x1E4 => "VCvtNormU16F16",
            // VOP3-encoded half-precision transcendentals and rounding.
            0x1D4 => "VRcpF16",
            0x1D5 => "VSqrtF16",
            0x1D7 => "VLogF16",
            0x1D8 => "VExpF16",
            0x1DB => "VFloorF16",
            0x1DC => "VCeilF16",
            0x1DD => "VTruncF16",
            0x1DE => "VRndneF16",
            0x1DF => "VFractF16",
            0x1E0 => "VSinF16",
            0x1E1 => "VCosF16",
            0x1D6 => "VRsqF16",
            0x301 => "VAshrrevI64",
            0x303 => "VAddNcU16",
            0x305 => "VMulLoU16",
            0x311 => "VPackB32F16",
            0x312 => "VCvtPknormI16F16",
            0x313 => "VCvtPknormU16F16",
            0x340 => "VMadU16",
            0x344 => "VPermB32",
            0x34B => "VFmaF16",
            0x351 => "VMin3F16",
            0x352 => "VMin3I16",
            0x353 => "VMin3U16",
            0x354 => "VMax3F16",
            0x355 => "VMax3I16",
            0x356 => "VMax3U16",
            0x357 => "VMed3F16",
            0x358 => "VMed3I16",
            0x359 => "VMed3U16",
            0x35E => "VMadI16",
            0x35F => "VDivFixupF16",
            0x360 => "VReadlaneB32",
            0x361 => "VWritelaneB32",
            0x362 => "VLdexpF32",
            0x363 => "VBfmB32",
            0x364 => "VBcntU32B32",
            0x365 => "VMbcntLoU32B32",
            0x366 => "VMbcntHiU32B32",
            0x368 => "VCvtPknormI16F32",
            0x369 => "VCvtPknormU16F32",
            0x36A => "VCvtPkU16U32",
            0x36B => "VCvtPkI16I32",
            // GFX10 reverse 64-bit shifts.
            0x2FF => "VLshlrevB64",
            0x300 => "VLshrrevB64",
            0x373 => "VMadU32U16",
            0x375 => "VMadI32I16",
            // V_ADD/SUB_NC_I32: the clamp bit saturates signed.
            0x376 => "VSubI32",
            0x37F => "VAddI32",
            0x346 => "VLshlAddU32",
            0x345 => "VXadU32",
            0x347 => "VAddLshlU32",
            0x36D => "VAdd3U32",
            0x36F => "VLshlOrU32",
            0x178 => "VXor3B32",
            // VOP1 opcodes 0x50-0x53 are available through VOP3 as opcodes 0x1D0-0x1D3.
            0x1D2 => "VCvtU16F16",
            0x371 => "VAndOrB32",
            0x372 => "VOr3U32",
            0x377 => "VPermlane16B32",
            0x378 => "VPermlanex16B32",
            // GFX10 16-bit integer ALU, VOP3-only.
            0x304 => "VSubNcU16",
            0x307 => "VLshrrevB16",
            0x308 => "VAshrrevI16",
            0x309 => "VMaxU16",
            0x30A => "VMaxI16",
            0x30B => "VMinU16",
            0x30C => "VMinI16",
            0x30D => "VAddNcI16",
            0x30E => "VSubNcI16",
            0x314 => "VLshlrevB16",
            // VOP3-encoded 64-bit VOPC (opcode < 0x100): V_CMP_*_U64 / V_CMPX_*_U64.
            0x0E0 => "VCmpFU64",
            0x0E1 => "VCmpLtU64",
            0x0E2 => "VCmpEqU64",
            0x0E3 => "VCmpLeU64",
            0x0E4 => "VCmpGtU64",
            0x0E5 => "VCmpNeU64",
            0x0E6 => "VCmpGeU64",
            0x0E7 => "VCmpTU64",
            0x0F0 => "VCmpxFU64",
            0x0F1 => "VCmpxLtU64",
            0x0F2 => "VCmpxEqU64",
            0x0F3 => "VCmpxLeU64",
            0x0F4 => "VCmpxGtU64",
            0x0F5 => "VCmpxNeU64",
            0x0F6 => "VCmpxGeU64",
            0x0F7 => "VCmpxTU64",
            _ => $"Vop3Raw{opcode:X3}",
        };

        return FinishDecode(name, $"unknown-vop3 op=0x{opcode:X3}", out error);
    }

    private static bool IsVop3BOpcode(uint opcode) =>
        opcode is 0x128 or 0x129 or 0x12A or 0x16D or 0x16E or 0x176 or 0x177 or 0x30F or 0x310 or 0x319;

    private static bool DecodeRaw2(
        uint word,
        string prefix,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        name = $"{prefix}Raw{word >> 24:X2}";
        sizeDwords = 2;
        error = string.Empty;
        return true;
    }

    private static bool DecodeVop3p(
        uint word,
        uint extra,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        var opcode = (word >> 16) & 0x7F;
        var src0 = extra & 0x1FF;
        var src1 = (extra >> 9) & 0x1FF;
        var src2 = (extra >> 18) & 0x1FF;
        sizeDwords = src0 == 0xFF || src1 == 0xFF || src2 == 0xFF ? 3u : 2u;
        error = string.Empty;

        // Opcode numbers taken from LLVM's AMDGPU VOP3PInstructions.td and the
        // gfx9/gfx10 MC test encodings; they are unchanged across gfx9 and gfx10.
        // The mix ops (0x20/0x21/0x22) are V_MAD_MIX_* on gfx9 and V_FMA_MIX_*
        // (fused) on the gfx10 the PS5 targets; both share these opcodes. Any
        // remaining packed opcode (integer, ...) stays opaque here and fails
        // loudly at emission rather than being silently mis-emitted.
        name = opcode switch
        {
            0x00 => "VPkMadI16",
            0x01 => "VPkMulLoU16",
            0x02 => "VPkAddI16",
            0x03 => "VPkSubI16",
            0x04 => "VPkLshlrevB16",
            0x05 => "VPkLshrrevB16",
            0x06 => "VPkAshrrevI16",
            0x07 => "VPkMaxI16",
            0x08 => "VPkMinI16",
            0x09 => "VPkMadU16",
            0x0A => "VPkAddU16",
            0x0B => "VPkSubU16",
            0x0C => "VPkMaxU16",
            0x0D => "VPkMinU16",
            0x0E => "VPkFmaF16",
            0x0F => "VPkAddF16",
            0x10 => "VPkMulF16",
            0x11 => "VPkMinF16",
            0x12 => "VPkMaxF16",
            0x13 => "VDot2F32F16",
            0x14 => "VDot2I32I16",
            0x15 => "VDot2U32U16",
            0x16 => "VDot4I32I8",
            0x17 => "VDot4U32U8",
            0x18 => "VDot8I32I4",
            0x19 => "VDot8U32U4",
            0x20 => "VFmaMixF32",
            0x21 => "VFmaMixloF16",
            0x22 => "VFmaMixhiF16",
            _ => $"Vop3pRaw{opcode:X2}",
        };

        return true;
    }

    private static bool DecodeDs(
        uint word,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        var opcode = (word >> 18) & 0xFF;
        var gds = ((word >> 17) & 1) != 0;
        sizeDwords = 2;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "DsAddU32",
            0x01 => "DsSubU32",
            0x02 => "DsRsubU32",
            0x03 => "DsIncU32",
            0x04 => "DsDecU32",
            0x05 => "DsMinI32",
            0x06 => "DsMaxI32",
            0x07 => "DsMinU32",
            0x08 => "DsMaxU32",
            0x09 => "DsAndB32",
            0x0A => "DsOrB32",
            0x0B => "DsXorB32",
            0x0C => "DsMskorB32",
            0x0D => "DsWriteB32",
            0x0E => "DsWrite2B32",
            0x0F => "DsWrite2St64B32",
            0x10 => "DsCmpstB32",
            0x11 => "DsCmpstF32",
            0x12 => "DsMinF32",
            0x13 => "DsMaxF32",
            0x14 => "DsNop",
            0x15 => "DsAddF32",
            // Global wave sync (GDS only): DATA0 of INIT, SEMA_BR and BARRIER is encoded in
            // the ADDR field, as LLVM's DS_GWS_1D does for gfx10.
            0x18 when gds => "DsGwsSemaReleaseAll",
            0x19 when gds => "DsGwsInit",
            0x1A when gds => "DsGwsSemaV",
            0x1B when gds => "DsGwsSemaBr",
            0x1C when gds => "DsGwsSemaP",
            0x1D when gds => "DsGwsBarrier",
            0x1E => "DsWriteB8",
            0x1F => "DsWriteB16",
            0x20 => "DsAddRtnU32",
            0x21 => "DsSubRtnU32",
            0x22 => "DsRsubRtnU32",
            0x23 => "DsIncRtnU32",
            0x24 => "DsDecRtnU32",
            0x25 => "DsMinRtnI32",
            0x26 => "DsMaxRtnI32",
            0x27 => "DsMinRtnU32",
            0x28 => "DsMaxRtnU32",
            0x29 => "DsAndRtnB32",
            0x2A => "DsOrRtnB32",
            0x2B => "DsXorRtnB32",
            0x2C => "DsMskorRtnB32",
            0x2D => "DsWrxchgRtnB32",
            0x2E => "DsWrxchg2RtnB32",
            0x2F => "DsWrxchg2St64RtnB32",
            0x30 => "DsCmpstRtnB32",
            0x31 => "DsCmpstRtnF32",
            0x32 => "DsMinRtnF32",
            0x33 => "DsMaxRtnF32",
            0x34 => "DsWrapRtnB32",
            0x35 => "DsSwizzleB32",
            0x36 => "DsReadB32",
            0x37 => "DsRead2B32",
            0x38 => "DsRead2St64B32",
            0x39 => "DsReadI8",
            0x3A => "DsReadU8",
            0x3B => "DsReadI16",
            0x3C => "DsReadU16",
            0x3D => "DsConsume",
            0x3E => "DsAppend",
            // GDS only; the value to add is in the ADDR field (llvm.amdgcn.ds.ordered.add).
            0x3F when gds => "DsOrderedCount",
            // gfx10 groups the 64-bit LDS atomics at 0x40..0x53, the returning forms at
            // 0x60..0x73. They are named so the translator can say which one it cannot
            // run; it takes only the forms whose halves can run as 32-bit atomics.
            0x40 => "DsAddU64",
            0x41 => "DsSubU64",
            0x42 => "DsRsubU64",
            0x43 => "DsIncU64",
            0x44 => "DsDecU64",
            0x45 => "DsMinI64",
            0x46 => "DsMaxI64",
            0x47 => "DsMinU64",
            0x48 => "DsMaxU64",
            0x49 => "DsAndB64",
            0x4A => "DsOrB64",
            0x4B => "DsXorB64",
            0x4C => "DsMskorB64",
            0x4D => "DsWriteB64",
            0x4E => "DsWrite2B64",
            0x4F => "DsWrite2St64B64",
            0x50 => "DsCmpstB64",
            0x51 => "DsCmpstF64",
            0x52 => "DsMinF64",
            0x53 => "DsMaxF64",
            0x55 => "DsAddRtnF32",
            0x60 => "DsAddRtnU64",
            0x61 => "DsSubRtnU64",
            0x62 => "DsRsubRtnU64",
            0x63 => "DsIncRtnU64",
            0x64 => "DsDecRtnU64",
            0x65 => "DsMinRtnI64",
            0x66 => "DsMaxRtnI64",
            0x67 => "DsMinRtnU64",
            0x68 => "DsMaxRtnU64",
            0x69 => "DsAndRtnB64",
            0x6A => "DsOrRtnB64",
            0x6B => "DsXorRtnB64",
            0x6C => "DsMskorRtnB64",
            0x6D => "DsWrxchgRtnB64",
            0x6E => "DsWrxchg2RtnB64",
            0x6F => "DsWrxchg2St64RtnB64",
            0x70 => "DsCmpstRtnB64",
            0x71 => "DsCmpstRtnF64",
            0x72 => "DsMinRtnF64",
            0x73 => "DsMaxRtnF64",
            0x76 => "DsReadB64",
            0x77 => "DsRead2B64",
            0x78 => "DsRead2St64B64",
            0x7E => "DsCondxchg32RtnB64",
            0xA0 => "DsWriteB8D16Hi",
            0xA1 => "DsWriteB16D16Hi",
            0xA2 => "DsReadU8D16",
            0xA3 => "DsReadU8D16Hi",
            0xA4 => "DsReadI8D16",
            0xA5 => "DsReadI8D16Hi",
            0xA6 => "DsReadU16D16",
            0xA7 => "DsReadU16D16Hi",
            0xB0 => "DsWriteAddtidB32",
            0xB1 => "DsReadAddtidB32",
            0xB2 => "DsPermuteB32",
            0xB3 => "DsBpermuteB32",
            0xDE => "DsWriteB96",
            0xDF => "DsWriteB128",
            0xFE => "DsReadB96",
            0xFF => "DsReadB128",
            _ => string.Empty,
        };

        return FinishDecode(
            name,
            $"unknown-ds op=0x{opcode:X2} word=0x{word:X8}",
            out error);
    }

    private static bool DecodeBuffer(
        uint word,
        string prefix,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        // GFX10 MIMG uses bit 0 as opcode bit 7.  Treating the opcode as only
        // bits 18..24 makes the 0x80+ family (including sample-adjust forms)
        // decode as the corresponding low opcode.
        var opcode = ((word >> 18) & 0x7F) | ((word & 1) << 7);
        name = $"{prefix}Raw{opcode:X2}";
        sizeDwords = 2;
        error = string.Empty;
        return true;
    }

    private static bool DecodeMtbuf(
        uint word,
        uint extra,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        // The fourth opcode bit lives in the second word and selects the D16 forms.
        var opcode = ((word >> 16) & 0x7) | (((extra >> 21) & 1) << 3);
        name = opcode switch
        {
            0x00 => "TBufferLoadFormatX",
            0x01 => "TBufferLoadFormatXy",
            0x02 => "TBufferLoadFormatXyz",
            0x03 => "TBufferLoadFormatXyzw",
            0x04 => "TBufferStoreFormatX",
            0x05 => "TBufferStoreFormatXy",
            0x06 => "TBufferStoreFormatXyz",
            0x07 => "TBufferStoreFormatXyzw",
            0x08 => "TBufferLoadFormatD16X",
            0x09 => "TBufferLoadFormatD16Xy",
            0x0A => "TBufferLoadFormatD16Xyz",
            0x0B => "TBufferLoadFormatD16Xyzw",
            0x0C => "TBufferStoreFormatD16X",
            0x0D => "TBufferStoreFormatD16Xy",
            0x0E => "TBufferStoreFormatD16Xyz",
            0x0F => "TBufferStoreFormatD16Xyzw",
            _ => string.Empty,
        };
        sizeDwords = (extra >> 24) == 0xFF ? 3u : 2u;
        error = string.Empty;
        if (((extra >> 23) & 1) != 0 && name.StartsWith("TBufferStore", StringComparison.Ordinal))
        {
            // TFE writes a load's status dword; a store has none to write.
            name = string.Empty;
            error = $"unsupported-mtbuf-tfe op=0x{opcode:X2} word=0x{word:X8} extra=0x{extra:X8}";
            return false;
        }

        return true;
    }

    private static bool DecodeMubuf(
        uint word,
        uint extra,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        // GFX10 MUBUF: OFFSET [11:0], OP [24:18] with op[7] in bit 25 (BUFInstructions.td
        // MUBUF_Real_gfx10). Bit 0 is offset[0]; only MIMG takes opcode bit 7 from bit 0.
        var opcode = (word >> 18) & 0xFF;
        name = opcode switch
        {
            0x00 => "BufferLoadFormatX",
            0x01 => "BufferLoadFormatXy",
            0x02 => "BufferLoadFormatXyz",
            0x03 => "BufferLoadFormatXyzw",
            0x04 => "BufferStoreFormatX",
            0x05 => "BufferStoreFormatXy",
            0x06 => "BufferStoreFormatXyz",
            0x07 => "BufferStoreFormatXyzw",
            0x08 => "BufferLoadUbyte",
            0x09 => "BufferLoadSbyte",
            0x0A => "BufferLoadUshort",
            0x0B => "BufferLoadSshort",
            0x0C => "BufferLoadDword",
            0x0D => "BufferLoadDwordx2",
            0x0E => "BufferLoadDwordx4",
            0x0F => "BufferLoadDwordx3",
            0x18 => "BufferStoreByte",
            0x19 => "BufferStoreByteD16Hi",
            0x1A => "BufferStoreShort",
            0x1B => "BufferStoreShortD16Hi",
            0x1C => "BufferStoreDword",
            0x1D => "BufferStoreDwordx2",
            0x1E => "BufferStoreDwordx4",
            0x1F => "BufferStoreDwordx3",
            0x20 => "BufferLoadUbyteD16",
            0x21 => "BufferLoadUbyteD16Hi",
            0x22 => "BufferLoadSbyteD16",
            0x23 => "BufferLoadSbyteD16Hi",
            0x24 => "BufferLoadShortD16",
            0x25 => "BufferLoadShortD16Hi",
            // BUFFER_LOAD/STORE_FORMAT_D16_HI_X: one formatted component in the high half.
            0x26 => "BufferLoadFormatD16HiX",
            0x27 => "BufferStoreFormatD16HiX",
            0x30 => "BufferAtomicSwap",
            0x31 => "BufferAtomicCmpswap",
            0x32 => "BufferAtomicAdd",
            0x33 => "BufferAtomicSub",
            // gfx10.3: memory = memory >= data ? memory - data : 0 (a subtract clamped at zero).
            0x34 => "BufferAtomicCsub",
            0x35 => "BufferAtomicSmin",
            0x36 => "BufferAtomicUmin",
            0x37 => "BufferAtomicSmax",
            0x38 => "BufferAtomicUmax",
            0x39 => "BufferAtomicAnd",
            0x3A => "BufferAtomicOr",
            0x3B => "BufferAtomicXor",
            0x3C => "BufferAtomicInc",
            0x3D => "BufferAtomicDec",
            // gfx10 op 0x3E is BUFFER_ATOMIC_FCMPSWAP; BUFFER_WBINVL1 only exists on gfx6/7/8/9.
            0x3E => "BufferAtomicFcmpswap",
            0x3F => "BufferAtomicFmin",
            0x40 => "BufferAtomicFmax",
            // GFX9 buffer atomic encodings used by the guest shaders.
            0x41 => "BufferAtomicCmpswap",
            0x42 => "BufferAtomicAdd",
            0x43 => "BufferAtomicSub",
            0x45 => "BufferAtomicSmin",
            0x46 => "BufferAtomicUmin",
            0x47 => "BufferAtomicSmax",
            0x48 => "BufferAtomicUmax",
            0x49 => "BufferAtomicAnd",
            0x4A => "BufferAtomicOr",
            0x4B => "BufferAtomicXor",
            0x4C => "BufferAtomicInc",
            // The 64-bit atomics (BUFInstructions.td MUBUF_Real_Atomics_gfx10): the 32-bit
            // opcode + 0x20, with no X2 form of CSUB (0x54).
            0x50 => "BufferAtomicSwapX2",
            0x51 => "BufferAtomicCmpswapX2",
            0x52 => "BufferAtomicAddX2",
            0x53 => "BufferAtomicSubX2",
            0x55 => "BufferAtomicSminX2",
            0x56 => "BufferAtomicUminX2",
            0x57 => "BufferAtomicSmaxX2",
            0x58 => "BufferAtomicUmaxX2",
            0x59 => "BufferAtomicAndX2",
            0x5A => "BufferAtomicOrX2",
            0x5B => "BufferAtomicXorX2",
            0x5C => "BufferAtomicIncX2",
            0x5D => "BufferAtomicDecX2",
            0x5E => "BufferAtomicFcmpswapX2",
            0x5F => "BufferAtomicFminX2",
            0x60 => "BufferAtomicFmaxX2",
            // Cache invalidations: the host's memory is already coherent at these levels.
            0x71 => "BufferGl0Inv",
            0x72 => "BufferGl1Inv",
            0x80 => "BufferLoadFormatD16X",
            0x81 => "BufferLoadFormatD16Xy",
            0x82 => "BufferLoadFormatD16Xyz",
            0x83 => "BufferLoadFormatD16Xyzw",
            0x84 => "BufferStoreFormatD16X",
            0x85 => "BufferStoreFormatD16Xy",
            0x86 => "BufferStoreFormatD16Xyz",
            0x87 => "BufferStoreFormatD16Xyzw",
            _ => $"MubufRaw{opcode:X2}",
        };
        sizeDwords = (extra >> 24) == 0xFF ? 3u : 2u;
        error = string.Empty;

        // TFE (extra bit 23) appends a status VGPR after a load's data; LDS (word bit 16)
        // returns a load's data to LDS instead (RDNA2 ISA 8.1.9: only
        // BUFFER_LOAD_{UBYTE,SBYTE,USHORT,SSHORT,DWORD,FORMAT_X}, and never with TFE). TFE
        // on a store or an atomic has no status to write and LLVM rejects it; both are
        // refused rather than guessed at.
        var tfe = ((extra >> 23) & 1) != 0;
        var lds = ((word >> 16) & 1) != 0;
        var isLoad = name.StartsWith("BufferLoad", StringComparison.Ordinal);
        if (lds && (tfe || name is not ("BufferLoadUbyte" or "BufferLoadSbyte" or "BufferLoadUshort" or
            "BufferLoadSshort" or "BufferLoadDword" or "BufferLoadFormatX")))
        {
            name = string.Empty;
            error = $"unsupported-mubuf-lds op=0x{opcode:X2} word=0x{word:X8} extra=0x{extra:X8}";
            return false;
        }

        if (tfe && !isLoad)
        {
            name = string.Empty;
            error = $"unsupported-mubuf-tfe op=0x{opcode:X2} word=0x{word:X8} extra=0x{extra:X8}";
            return false;
        }

        return true;
    }

    private static bool DecodeFlat(
        uint word,
        out string name,
        out uint sizeDwords,
        out string error)
    {
        var segment = (word >> 14) & 0x3;
        // GFX10 FLAT: OFFSET [11:0], DLC [12], SEG [15:14], GLC [16], SLC [17], OP [24:18]
        // (FLATInstructions.td:2990); there is no opcode bit 7, bit 0 is offset[0].
        var opcode = (word >> 18) & 0x7F;
        sizeDwords = 2;
        error = string.Empty;
        var prefix = segment switch
        {
            0x0 => "Flat",
            0x1 => "Scratch",
            0x2 => "Global",
            _ => string.Empty,
        };
        var suffix = opcode switch
        {
            0x08 => "LoadUbyte",
            0x09 => "LoadSbyte",
            0x0A => "LoadUshort",
            0x0B => "LoadSshort",
            0x0C => "LoadDword",
            0x0D => "LoadDwordx2",
            0x0E => "LoadDwordx4",
            0x0F => "LoadDwordx3",
            // gfx10.3 GLOBAL only: SADDR + offset + lane * 4, no VADDR.
            0x16 when segment == 0x2 => "LoadDwordAddtid",
            0x17 when segment == 0x2 => "StoreDwordAddtid",
            0x18 => "StoreByte",
            0x19 => "StoreByteD16Hi",
            0x1A => "StoreShort",
            0x1B => "StoreShortD16Hi",
            0x1C => "StoreDword",
            0x1D => "StoreDwordx2",
            0x1E => "StoreDwordx4",
            0x1F => "StoreDwordx3",
            0x20 => "LoadUbyteD16",
            0x21 => "LoadUbyteD16Hi",
            0x22 => "LoadSbyteD16",
            0x23 => "LoadSbyteD16Hi",
            0x24 => "LoadShortD16",
            0x25 => "LoadShortD16Hi",
            // GFX10 FLAT/GLOBAL atomics (FLATInstructions.td FLAT_Real_GlblAtomics_gfx10):
            // the 32-bit forms at 0x30.., their 64-bit X2 forms at 0x50...
            0x30 => "AtomicSwap",
            0x31 => "AtomicCmpswap",
            0x32 => "AtomicAdd",
            0x33 => "AtomicSub",
            // gfx10.3 GLOBAL only: a subtract clamped at zero.
            0x34 when segment == 0x2 => "AtomicCsub",
            0x35 => "AtomicSmin",
            0x36 => "AtomicUmin",
            0x37 => "AtomicSmax",
            0x38 => "AtomicUMax",
            0x39 => "AtomicAnd",
            0x3A => "AtomicOr",
            0x3B => "AtomicXor",
            0x3C => "AtomicInc",
            0x3D => "AtomicDec",
            0x3E => "AtomicFcmpswap",
            0x3F => "AtomicFmin",
            0x40 => "AtomicFmax",
            0x50 => "AtomicSwapX2",
            0x51 => "AtomicCmpswapX2",
            0x52 => "AtomicAddX2",
            0x53 => "AtomicSubX2",
            0x55 => "AtomicSminX2",
            0x56 => "AtomicUminX2",
            0x57 => "AtomicSmaxX2",
            0x58 => "AtomicUmaxX2",
            0x59 => "AtomicAndX2",
            0x5A => "AtomicOrX2",
            0x5B => "AtomicXorX2",
            0x5C => "AtomicIncX2",
            0x5D => "AtomicDecX2",
            0x5E => "AtomicFcmpswapX2",
            0x5F => "AtomicFminX2",
            0x60 => "AtomicFmaxX2",
            _ => string.Empty,
        };
        name = prefix.Length != 0 && suffix.Length != 0
            ? prefix + suffix
            : string.Empty;

        return FinishDecode(
            name,
            $"unknown-flat segment=0x{segment:X1} op=0x{opcode:X2} word=0x{word:X8}",
            out error);
    }

    private static bool DecodeSmrd(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 22) & 0x1F;
        var offset = word & 0xFF;
        var immediateOffset = ((word >> 8) & 1) != 0;
        sizeDwords = !immediateOffset && offset == 0xFF ? 2u : 1u;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "SLoadDword",
            0x01 => "SLoadDwordx2",
            0x02 => "SLoadDwordx4",
            0x03 => "SLoadDwordx8",
            0x04 => "SLoadDwordx16",
            0x08 => "SBufferLoadDword",
            0x09 => "SBufferLoadDwordx2",
            0x0A => "SBufferLoadDwordx4",
            0x0B => "SBufferLoadDwordx8",
            0x0C => "SBufferLoadDwordx16",
            0x15 => "ScratchStoreDword",
            0x16 => "ScratchStoreDwordx2",
            0x17 => "ScratchStoreDwordx4",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-smrd op=0x{opcode:X2} word=0x{word:X8}", out error);
    }

    private static bool DecodeSmem(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 18) & 0xFF;
        sizeDwords = 2;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "SLoadDword",
            0x01 => "SLoadDwordx2",
            0x02 => "SLoadDwordx4",
            0x03 => "SLoadDwordx8",
            0x04 => "SLoadDwordx16",
            0x08 => "SBufferLoadDword",
            0x09 => "SBufferLoadDwordx2",
            0x0A => "SBufferLoadDwordx4",
            0x0B => "SBufferLoadDwordx8",
            0x0C => "SBufferLoadDwordx16",
            // Cache control and counters (RDNA2 ISA SMEM opcode table; S_DCACHE_WB is gfx10.1's).
            0x1F => "SGl1Inv",
            0x20 => "SDcacheInv",
            0x21 => "SDcacheWb",
            0x24 => "SMemtime",
            0x25 => "SMemrealtime",
            0x26 => "SAtcProbe",
            0x27 => "SAtcProbeBuffer",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-smem op=0x{opcode:X2}", out error);
    }

    private static bool DecodeMimg(uint word, out string name, out uint sizeDwords, out string error)
    {
        // RDNA2 MIMG OP[7] is bit 0 while OP[6:0] occupies bits 24:18. Ignoring bit 0
        // aliases the 0x80+ opcodes (MSAA_LOAD, the _G16 and BVH forms) to 0x00-0x7F.
        var opcode = ((word >> 18) & 0x7F) | ((word & 1) << 7);
        sizeDwords = 2 + ((word >> 1) & 0x3);
        name = Gen5ImageOpcodes.NameOf(opcode) ?? string.Empty;

        // TFE [16] and LWE [17] are accepted: the status dword they add after the data is
        // written only when a fetch is NACKed (an unmapped PRT page) or, for LWE, when the
        // LOD falls below the T#'s MIN_LOD_WARN (RDNA2 ISA Table 41). Guest texture memory
        // is always resident here, so the status VGPR keeps its value; LLVM's AddMemOpInit
        // zeroes that dword before every TFE/LWE instruction for the same reason.
        return FinishDecode(name, $"unknown-mimg op=0x{opcode:X2}", out error);
    }

    private static bool DecodeVintrp(uint word, out string name, out uint sizeDwords, out string error)
    {
        var opcode = (word >> 16) & 0x3;
        sizeDwords = 1;
        error = string.Empty;
        name = opcode switch
        {
            0x00 => "VInterpP1F32",
            0x01 => "VInterpP2F32",
            0x02 => "VInterpMovF32",
            _ => string.Empty,
        };

        return FinishDecode(name, $"unknown-vintrp op=0x{opcode:X1}", out error);
    }

    // VOP3-encoded interpolation: the f32 forms of VINTRP and the f16 forms only VOP3 has.
    private static string? Vop3InterpolationName(uint opcode) => opcode switch
    {
        0x200 => "VInterpP1F32",
        0x201 => "VInterpP2F32",
        0x202 => "VInterpMovF32",
        0x342 => "VInterpP1llF16",
        0x343 => "VInterpP1lvF16",
        0x35A => "VInterpP2F16",
        _ => null,
    };

    private static bool FinishDecode(string name, string decodeError, out string error)
    {
        error = string.Empty;
        if (name.Length != 0)
        {
            return true;
        }

        error = decodeError;
        return false;
    }

    private static string ClassifyInstruction(string name)
    {
        if (name.StartsWith("Image", StringComparison.Ordinal))
        {
            return "image";
        }

        if (name.StartsWith("Global", StringComparison.Ordinal))
        {
            return "global_memory";
        }

        if (name.StartsWith("VInterp", StringComparison.Ordinal))
        {
            return "interp";
        }

        if (string.Equals(name, "Exp", StringComparison.Ordinal))
        {
            return "export";
        }

        if (name.StartsWith("SLoad", StringComparison.Ordinal) ||
            name.StartsWith("SBufferLoad", StringComparison.Ordinal))
        {
            return "scalar_load";
        }

        if (name.StartsWith('V'))
        {
            return "valu";
        }

        if (name.StartsWith('S'))
        {
            return "salu";
        }

        return "other";
    }

    private static bool IsMimgInstruction(string name) =>
        name.StartsWith("Image", StringComparison.Ordinal);

    public static bool IsImageLoadOperation(string name) =>
        name.StartsWith("ImageLoad", StringComparison.Ordinal);

    public static bool IsStorageImageOperation(string name) =>
        name.StartsWith("ImageStore", StringComparison.Ordinal) ||
        name.StartsWith("ImageAtomic", StringComparison.Ordinal);

    public const uint IdentityImageDstSelect = 0xFACu;

    public static uint GetImageDescriptorDstSelect(
        IReadOnlyList<uint> resourceDescriptor) =>
        resourceDescriptor.Count > 3
            ? resourceDescriptor[3] & 0xFFFu
            : IdentityImageDstSelect;

    /// <summary>
    /// Gets the sequential store-source index for one physical image channel.
    /// A negative result means that the channel must contain zero.
    /// </summary>
    public static int GetImageStoreSourceIndex(
        uint dstSelect,
        uint dmask,
        int destinationComponent)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(destinationComponent);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(destinationComponent, 3);

        var logicalComponent = -1;
        var physicalSelector = 4u + (uint)destinationComponent;
        for (var component = 0; component < 4; component++)
        {
            if (((dstSelect >> (component * 3)) & 0x7u) == physicalSelector)
            {
                logicalComponent = component;
                break;
            }
        }

        if (logicalComponent < 0)
        {
            return -1;
        }

        var effectiveDmask = dmask & 0xFu;
        if (effectiveDmask == 0)
        {
            effectiveDmask = 1;
        }

        if ((effectiveDmask & (1u << logicalComponent)) == 0)
        {
            return -1;
        }

        var sourceIndex = 0;
        for (var component = 0; component < logicalComponent; component++)
        {
            sourceIndex += (int)((effectiveDmask >> component) & 1u);
        }

        return sourceIndex;
    }

    public static bool IsDataShareAtomic(string name) => name switch
    {
        "DsAddU32" or "DsSubU32" or "DsRsubU32" or "DsIncU32" or "DsDecU32" or
        "DsMinI32" or "DsMaxI32" or "DsMinU32" or "DsMaxU32" or
        "DsMinF32" or "DsMaxF32" or "DsAddF32" or "DsCmpstF32" or
        "DsAndB32" or "DsOrB32" or "DsXorB32" or "DsMskorB32" or "DsCmpstB32" or
        "DsAddRtnU32" or "DsSubRtnU32" or "DsRsubRtnU32" or "DsIncRtnU32" or "DsDecRtnU32" or
        "DsMinRtnI32" or "DsMaxRtnI32" or "DsMinRtnU32" or "DsMaxRtnU32" or
        "DsAndRtnB32" or "DsOrRtnB32" or "DsXorRtnB32" or "DsMskorRtnB32" or
        "DsWrxchgRtnB32" or "DsWrxchg2RtnB32" or "DsWrxchg2St64RtnB32" or
        "DsCmpstRtnB32" or "DsCmpstRtnF32" or "DsMinRtnF32" or "DsMaxRtnF32" or
        "DsAddRtnF32" or "DsWrapRtnB32" => true,
        _ => IsDataShareAtomic64(name),
    };

    /// <summary>The 64-bit LDS atomics: a register pair per data operand and, for the
    /// returning forms, a pair of destination registers.</summary>
    public static bool IsDataShareAtomic64(string name) => name switch
    {
        "DsAddU64" or "DsSubU64" or "DsRsubU64" or "DsIncU64" or "DsDecU64" or
        "DsMinI64" or "DsMaxI64" or "DsMinU64" or "DsMaxU64" or
        "DsAndB64" or "DsOrB64" or "DsXorB64" or "DsMskorB64" or
        "DsCmpstB64" or "DsCmpstF64" or "DsMinF64" or "DsMaxF64" or
        "DsAddRtnU64" or "DsSubRtnU64" or "DsRsubRtnU64" or "DsIncRtnU64" or "DsDecRtnU64" or
        "DsMinRtnI64" or "DsMaxRtnI64" or "DsMinRtnU64" or "DsMaxRtnU64" or
        "DsAndRtnB64" or "DsOrRtnB64" or "DsXorRtnB64" or "DsMskorRtnB64" or
        "DsWrxchgRtnB64" or "DsWrxchg2RtnB64" or "DsWrxchg2St64RtnB64" or
        "DsCmpstRtnB64" or "DsCmpstRtnF64" or "DsMinRtnF64" or "DsMaxRtnF64" => true,
        _ => false,
    };

    /// <summary>The LDS atomics that read a second data operand (DATA1) besides DATA0.</summary>
    public static bool IsDataShareTwoOperandAtomic(string name) =>
        name.StartsWith("DsMskor", StringComparison.Ordinal) ||
        name.StartsWith("DsCmpst", StringComparison.Ordinal) ||
        name.StartsWith("DsWrxchg2", StringComparison.Ordinal) ||
        name is "DsMinF32" or "DsMaxF32" or "DsMinRtnF32" or "DsMaxRtnF32" or
            "DsMinF64" or "DsMaxF64" or "DsMinRtnF64" or "DsMaxRtnF64" or "DsWrapRtnB32";

    private static Gen5ShaderInstruction ResolveFlatAddressBase(
        IReadOnlyList<Gen5ShaderInstruction> precedingInstructions,
        Gen5ShaderInstruction instruction)
    {
        if (instruction.Control is not Gen5GlobalMemoryControl
            {
                UsesFlatAddress: true,
            } control ||
            !TryFindVectorDefinition(
                precedingInstructions,
                control.VectorAddress,
                out var lowDefinition) ||
            !TryFindVectorDefinition(
                precedingInstructions,
                control.VectorAddress + 1,
                out var highDefinition))
        {
            return instruction;
        }

        foreach (var lowSource in lowDefinition.Sources)
        {
            if (lowSource.Kind != Gen5OperandKind.ScalarRegister)
            {
                continue;
            }

            foreach (var highSource in highDefinition.Sources)
            {
                if (highSource.Kind != Gen5OperandKind.ScalarRegister ||
                    highSource.Value != lowSource.Value + 1)
                {
                    continue;
                }

                return instruction with
                {
                    Sources =
                    [
                        .. instruction.Sources,
                        Gen5Operand.Scalar(lowSource.Value),
                    ],
                    Control = control with
                    {
                        ScalarAddress = lowSource.Value,
                    },
                };
            }
        }

        return instruction;
    }

    private static bool TryFindVectorDefinition(
        IReadOnlyList<Gen5ShaderInstruction> instructions,
        uint register,
        out Gen5ShaderInstruction definition)
    {
        for (var index = instructions.Count - 1; index >= 0; index--)
        {
            var candidate = instructions[index];
            foreach (var destination in candidate.Destinations)
            {
                if (destination.Kind == Gen5OperandKind.VectorRegister &&
                    destination.Value == register)
                {
                    definition = candidate;
                    return true;
                }
            }
        }

        definition = default!;
        return false;
    }

    private static Gen5ShaderInstruction CreateInstruction(
        uint pc,
        Gen5ShaderEncoding encoding,
        string opcode,
        uint[] words)
    {
        var word = words[0];
        var isSdwa =
            encoding is Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or Gen5ShaderEncoding.Vopc &&
            (word & 0x1FF) == 0xF9;
        var isDpp =
            encoding is Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or Gen5ShaderEncoding.Vopc &&
            (word & 0x1FF) == 0xFA;
        var isDpp8 =
            encoding is Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or Gen5ShaderEncoding.Vopc &&
            (word & 0x1FF) is 0xE9 or 0xEA;
        var literal = !isSdwa && !isDpp && !isDpp8 &&
            words.Length > MinimumEncodingDwords(encoding)
            ? words[^1]
            : (uint?)null;
        IReadOnlyList<Gen5Operand> sources = [];
        IReadOnlyList<Gen5Operand> destinations = [];
        Gen5InstructionControl? control = null;

        switch (encoding)
        {
            case Gen5ShaderEncoding.Sopp when opcode == "STrap":
                sources = [new Gen5Operand(Gen5OperandKind.LiteralConstant, word & 0xFF)];
                break;
            case Gen5ShaderEncoding.Sop1:
                sources = [Gen5Operand.Source(word & 0xFF, literal)];
                destinations = [Gen5Operand.Scalar((word >> 16) & 0x7F)];
                if (opcode is "SCmovB64" or "SCmovB32")
                {
                    // CMOV is CSELECT with the old destination as the false
                    // source. Reuse its SCC, register-pair and EXEC handling.
                    opcode = opcode == "SCmovB64" ? "SCselectB64" : "SCselectB32";
                    encoding = Gen5ShaderEncoding.Sop2;
                    sources = [sources[0], destinations[0]];
                }
                break;
            case Gen5ShaderEncoding.Sop2:
                sources =
                [
                    Gen5Operand.Source(word & 0xFF, literal),
                    Gen5Operand.Source((word >> 8) & 0xFF, literal),
                ];
                destinations = [Gen5Operand.Scalar((word >> 16) & 0x7F)];
                break;
            case Gen5ShaderEncoding.Sopc:
                sources =
                [
                    Gen5Operand.Source(word & 0xFF, literal),
                    Gen5Operand.Source((word >> 8) & 0xFF, literal),
                ];
                break;
            case Gen5ShaderEncoding.Sopk:
                sources = [new Gen5Operand(Gen5OperandKind.EncodedConstant, word & 0xFFFF)];
                if (opcode == "SWaitcnt")
                {
                    var scalarSource = (word >> 16) & 0x7F;
                    if (scalarSource != 125)
                    {
                        sources = [.. sources, Gen5Operand.Scalar(scalarSource)];
                    }
                }
                else if (opcode == "SSetregB32")
                {
                    sources =
                    [
                        Gen5Operand.Scalar((word >> 16) & 0x7F),
                        new Gen5Operand(Gen5OperandKind.EncodedConstant, word & 0xFFFF),
                    ];
                }
                else if (opcode == "SSetregImm32B32")
                {
                    // The value is the literal dword; SIMM16 names the register field.
                    sources =
                    [
                        new Gen5Operand(Gen5OperandKind.LiteralConstant, literal ?? 0),
                        new Gen5Operand(Gen5OperandKind.EncodedConstant, word & 0xFFFF),
                    ];
                }
                else if (opcode == "SVersion")
                {
                    // A code-object version marker: nothing is written.
                }
                else
                {
                    destinations = [Gen5Operand.Scalar((word >> 16) & 0x7F)];
                }
                break;
            case Gen5ShaderEncoding.Smrd:
            {
                var scalarBase = ((word >> 9) & 0x3F) * 2;
                var scalarDestination = (word >> 15) & 0x7F;
                var immediate = ((word >> 8) & 1) != 0;
                var offset = word & 0xFF;
                if (opcode.StartsWith("ScratchStore", StringComparison.Ordinal))
                {
                    var storeCount = ScalarMemoryDwordCount(opcode);
                    uint? storeDynamicOffsetRegister = null;
                    var storeImmediateOffsetBytes = 0;
                    if (immediate)
                    {
                        storeImmediateOffsetBytes = checked((int)(offset * sizeof(uint)));
                    }
                    else if (offset == 0xFF && literal.HasValue)
                    {
                        storeImmediateOffsetBytes = unchecked((int)literal.Value);
                    }
                    else
                    {
                        storeDynamicOffsetRegister = offset;
                    }

                    sources = [
                        Gen5Operand.Scalar(scalarBase),
                        .. Enumerable.Range(0, checked((int)storeCount)).Select(index =>
                            Gen5Operand.Scalar(scalarDestination + (uint)index)),
                    ];
                    if (storeDynamicOffsetRegister.HasValue)
                    {
                        sources = [.. sources, Gen5Operand.Scalar(storeDynamicOffsetRegister.Value)];
                    }

                    destinations = [];
                    control = new Gen5GlobalMemoryControl(
                        storeCount,
                        0,
                        scalarDestination,
                        0,
                        scalarBase,
                        storeImmediateOffsetBytes,
                        false,
                        false,
                        false,
                        storeDynamicOffsetRegister,
                        true);
                    break;
                }

                var count = ScalarLoadDwordCount(opcode);
                uint? dynamicOffsetRegister = null;
                var immediateOffsetBytes = 0;
                if (immediate)
                {
                    immediateOffsetBytes = checked((int)(offset * sizeof(uint)));
                }
                else if (offset == 0xFF && literal.HasValue)
                {
                    immediateOffsetBytes = unchecked((int)literal.Value);
                }
                else
                {
                    dynamicOffsetRegister = offset;
                }

                sources = dynamicOffsetRegister.HasValue
                    ? [Gen5Operand.Scalar(scalarBase), Gen5Operand.Scalar(dynamicOffsetRegister.Value)]
                    : [Gen5Operand.Scalar(scalarBase)];
                destinations = Enumerable
                    .Range((int)scalarDestination, checked((int)count))
                    .Select(index => Gen5Operand.Scalar((uint)index))
                    .ToArray();
                control = new Gen5ScalarMemoryControl(
                    count,
                    immediateOffsetBytes,
                    dynamicOffsetRegister);
                break;
            }
            case Gen5ShaderEncoding.Smem:
            {
                var extra = words[1];
                var scalarBase = (word & 0x3F) * 2;
                var scalarDestination = (word >> 6) & 0x7F;
                var scalarOffset = (extra >> 25) & 0x7F;
                var offset = SignExtend(extra & 0x1FFFFF, 21);
                if (opcode is "SMemtime" or "SMemrealtime")
                {
                    // A 64-bit counter into the SDATA pair; no memory is read.
                    sources = [];
                    destinations = [Gen5Operand.Scalar(scalarDestination), Gen5Operand.Scalar(scalarDestination + 1)];
                    break;
                }

                if (opcode is "SGl1Inv" or "SDcacheInv" or "SDcacheWb" or "SAtcProbe" or "SAtcProbeBuffer")
                {
                    // Cache maintenance and probes: no register is written. A probe's SDATA
                    // is its probe level, not a register.
                    sources = opcode.StartsWith("SAtcProbe", StringComparison.Ordinal)
                        ? [Gen5Operand.Scalar(scalarBase)]
                        : [];
                    destinations = [];
                    break;
                }

                var count = ScalarLoadDwordCount(opcode);
                var scalarOffsetOperand = Gen5Operand.Source(scalarOffset);
                var dynamicOffsetRegister = scalarOffsetOperand.Kind ==
                    Gen5OperandKind.ScalarRegister
                    ? scalarOffsetOperand.Value
                    : (uint?)null;
                sources =
                [
                    Gen5Operand.Scalar(scalarBase),
                    scalarOffsetOperand,
                ];
                destinations = Enumerable
                    .Range((int)scalarDestination, checked((int)count))
                    .Select(index => Gen5Operand.Scalar((uint)index))
                    .ToArray();
                control = new Gen5ScalarMemoryControl(
                    count,
                    offset,
                    dynamicOffsetRegister,
                    ((word >> 16) & 1) != 0,
                    ((word >> 14) & 1) != 0);
                break;
            }
            case Gen5ShaderEncoding.Vop1:
                if (isDpp8)
                {
                    var extra = words[1];
                    sources = [Gen5Operand.Vector(extra & 0xFF)];
                    control = new Gen5Dpp8Control(
                        extra >> 8,
                        (word & 0x1FF) == 0xEA);
                }
                else if (isDpp)
                {
                    var extra = words[1];
                    sources = [Gen5Operand.Vector(extra & 0xFF)];
                    control = CreateDppControl(extra);
                }
                else if (isSdwa)
                {
                    var extra = words[1];
                    var source0 = (extra & 0xFF) +
                        ((((extra >> 23) & 1) == 0) ? 256u : 0u);
                    sources = [Gen5Operand.Source(source0)];
                    control = CreateSdwaControl(extra, isCompare: false, hasSource1: false);
                }
                else
                {
                    sources = [Gen5Operand.Source(word & 0x1FF, literal)];
                }

                if (opcode == "VMovrelsB32")
                {
                    sources = [sources[0], Gen5Operand.Scalar(124)];
                }

                if (opcode == "VSwapB32")
                {
                    // V_SWAP_B32 writes both of its operands: vdst takes src0 and
                    // the src0 VGPR takes the old vdst.
                    destinations =
                    [
                        Gen5Operand.Vector((word >> 17) & 0xFF),
                        Gen5Operand.Vector(word & 0xFF),
                    ];
                    break;
                }

                // V_READFIRSTLANE_B32 is encoded as VOP1, but its destination
                // field names an SGPR rather than a VGPR. Treating it like an
                // ordinary vector destination leaves every invocation with a
                // different value and corrupts scalar addresses derived from
                // lane data.
                destinations = opcode == "VReadfirstlaneB32"
                    ? [Gen5Operand.Scalar((word >> 17) & 0x7F)]
                    : [Gen5Operand.Vector((word >> 17) & 0xFF)];
                break;
            case Gen5ShaderEncoding.Vop2:
                if (isDpp8)
                {
                    var extra = words[1];
                    sources =
                    [
                        Gen5Operand.Vector(extra & 0xFF),
                        Gen5Operand.Vector((word >> 9) & 0xFF),
                    ];
                    control = new Gen5Dpp8Control(
                        extra >> 8,
                        (word & 0x1FF) == 0xEA);
                }
                else if (isDpp)
                {
                    var extra = words[1];
                    sources =
                    [
                        Gen5Operand.Vector(extra & 0xFF),
                        Gen5Operand.Vector((word >> 9) & 0xFF),
                    ];
                    control = CreateDppControl(extra);
                }
                else if (isSdwa)
                {
                    var extra = words[1];
                    var source0 = (extra & 0xFF) + ((((extra >> 23) & 1) == 0) ? 256u : 0u);
                    var source1 =
                        ((word >> 9) & 0xFF) +
                        ((((extra >> 31) & 1) == 0) ? 256u : 0u);
                    sources =
                    [
                        Gen5Operand.Source(source0),
                        Gen5Operand.Source(source1),
                    ];
                    control = CreateSdwaControl(extra, isCompare: false, hasSource1: true);
                }
                else
                {
                    sources =
                    [
                        Gen5Operand.Source(word & 0x1FF, literal),
                        Gen5Operand.Vector((word >> 9) & 0xFF),
                    ];
                    if ((opcode is "VMadMkF32" or "VFmaMkF32" or "VFmaMkF16") && literal.HasValue)
                    {
                        sources =
                        [
                            sources[0],
                            new Gen5Operand(Gen5OperandKind.LiteralConstant, literal.Value),
                            sources[1],
                        ];
                    }
                    else if ((opcode is "VMadAkF32" or "VFmaAkF32" or "VFmaAkF16") && literal.HasValue)
                    {
                        sources =
                        [
                            .. sources,
                            new Gen5Operand(Gen5OperandKind.LiteralConstant, literal.Value),
                        ];
                    }
                }

                destinations = [Gen5Operand.Vector((word >> 17) & 0xFF)];
                break;
            case Gen5ShaderEncoding.Vopc:
                if (isDpp8)
                {
                    var extra = words[1];
                    sources =
                    [
                        Gen5Operand.Vector(extra & 0xFF),
                        Gen5Operand.Vector((word >> 9) & 0xFF),
                    ];
                    control = new Gen5Dpp8Control(
                        extra >> 8,
                        (word & 0x1FF) == 0xEA);
                }
                else if (isDpp)
                {
                    var extra = words[1];
                    sources =
                    [
                        Gen5Operand.Vector(extra & 0xFF),
                        Gen5Operand.Vector((word >> 9) & 0xFF),
                    ];
                    control = CreateDppControl(extra);
                }
                else if (isSdwa)
                {
                    var extra = words[1];
                    var source0 = (extra & 0xFF) +
                        ((((extra >> 23) & 1) == 0) ? 256u : 0u);
                    var source1 =
                        ((word >> 9) & 0xFF) +
                        ((((extra >> 31) & 1) == 0) ? 256u : 0u);
                    sources =
                    [
                        Gen5Operand.Source(source0),
                        Gen5Operand.Source(source1),
                    ];
                    var sdwa = CreateSdwaControl(extra, isCompare: true, hasSource1: true);
                    control = sdwa;
                    if (sdwa.ScalarDestination is { } scalarDestination &&
                        scalarDestination != 106)
                    {
                        destinations = [Gen5Operand.Scalar(scalarDestination)];
                    }
                }
                else
                {
                    sources =
                    [
                        Gen5Operand.Source(word & 0x1FF, literal),
                        Gen5Operand.Vector((word >> 9) & 0xFF),
                    ];
                }
                break;
            case Gen5ShaderEncoding.Vop3:
            {
                var extra = words[1];
                sources =
                [
                    Gen5Operand.Source(extra & 0x1FF, literal),
                    Gen5Operand.Source((extra >> 9) & 0x1FF, literal),
                    Gen5Operand.Source((extra >> 18) & 0x1FF, literal),
                ];
                destinations = [Gen5Operand.Vector(word & 0xFF)];
                if (Vop3InterpolationName((word >> 16) & 0x3FF) is not null)
                {
                    // src0 is attr [5:0], chan [7:6] and HIGH [8]; src1 the I/J VGPR (or
                    // V_INTERP_MOV's parameter); src2 the f16 forms' S2.
                    var parameter = (extra >> 9) & 0x1FF;
                    sources = opcode switch
                    {
                        "VInterpMovF32" => [],
                        "VInterpP1lvF16" or "VInterpP2F16" => [sources[1], sources[2]],
                        _ => [sources[1]],
                    };
                    control = new Gen5InterpolationControl(extra & 0x3F, (extra >> 6) & 0x3)
                    {
                        Parameter = opcode == "VInterpMovF32" ? parameter : 0,
                        High = ((extra >> 8) & 1) != 0,
                        Modifiers = new Gen5Vop3Control(
                            (word >> 8) & 0x7,
                            (extra >> 29) & 0x7,
                            (extra >> 27) & 0x3,
                            ((word >> 15) & 1) != 0,
                            (word >> 11) & 0xF,
                            null),
                    };
                    break;
                }

                if (opcode == "VReadlaneB32")
                {
                    // V_READLANE uses the VOP3A vdst byte even though the
                    // destination register is scalar. Bits 8-14 are the
                    // distinct sdst field used by VOP3B encodings.
                    destinations = [Gen5Operand.Scalar(word & 0xFF)];
                }
                var isVop3B = IsVop3BOpcode((word >> 16) & 0x3FF);
                if (opcode.StartsWith("VCmp", StringComparison.Ordinal))
                {
                    // VOP3 compares take two sources and write their mask to the SGPR pair in
                    // the vdst byte; GFX10 VCMPX writes EXEC only.
                    var isExecCompare = opcode.StartsWith("VCmpx", StringComparison.Ordinal);
                    sources = [sources[0], sources[1]];
                    destinations = isExecCompare ? [] : [Gen5Operand.Scalar(word & 0xFF)];
                    control = new Gen5Vop3Control(
                        (word >> 8) & 0x7,
                        (extra >> 29) & 0x7,
                        0,
                        false,
                        0,
                        isExecCompare ? null : word & 0xFF);
                    break;
                }

                control = new Gen5Vop3Control(
                    isVop3B ? 0 : (word >> 8) & 0x7,
                    (extra >> 29) & 0x7,
                    (extra >> 27) & 0x3,
                    ((word >> 15) & 1) != 0,
                    isVop3B ? 0 : (word >> 11) & 0xF,
                    isVop3B ? (word >> 8) & 0x7F : null);
                break;
            }
            case Gen5ShaderEncoding.Vop3p:
            {
                var extra = words[1];
                sources =
                [
                    Gen5Operand.Source(extra & 0x1FF, literal),
                    Gen5Operand.Source((extra >> 9) & 0x1FF, literal),
                    Gen5Operand.Source((extra >> 18) & 0x1FF, literal),
                ];
                destinations = [Gen5Operand.Vector(word & 0xFF)];

                // op_sel_hi is split across both dwords: bits [1:0] live in word1
                // [28:27], bit [2] in word0 [14].
                var opSelHi = ((extra >> 27) & 0x3) | (((word >> 14) & 0x1) << 2);
                control = new Gen5Vop3pControl(
                    (word >> 11) & 0x7,
                    opSelHi,
                    (extra >> 29) & 0x7,
                    (word >> 8) & 0x7,
                    ((word >> 15) & 1) != 0);
                break;
            }
            case Gen5ShaderEncoding.Ds:
            {
                var extra = words[1];
                var vectorAddress = extra & 0xFF;
                var vectorData0 = (extra >> 8) & 0xFF;
                var vectorData1 = (extra >> 16) & 0xFF;
                var vectorDestination = (extra >> 24) & 0xFF;
                // GFX10 DS: offset0 [7:0], offset1 [15:8], bit 16 reserved, GDS [17], op [25:18].
                control = new Gen5DataShareControl(
                    word & 0xFF,
                    (word >> 8) & 0xFF,
                    ((word >> 17) & 1) != 0);
                sources = opcode switch
                {
                    "DsAppend" or "DsConsume" or "DsReadAddtidB32" => [Gen5Operand.Scalar(124)],
                    // GWS resources are offset0[5:0] + M0[21:16]; DATA0, where an op takes
                    // one, sits in the ADDR field. DS_ORDERED_COUNT adds its ADDR-field VGPR
                    // and reads the ordered wave id from M0.
                    "DsGwsSemaReleaseAll" or "DsGwsSemaV" or "DsGwsSemaP" => [Gen5Operand.Scalar(124)],
                    "DsGwsInit" or "DsGwsSemaBr" or "DsGwsBarrier" or "DsOrderedCount" =>
                        [Gen5Operand.Vector(vectorAddress), Gen5Operand.Scalar(124)],
                    // DS_CONDXCHG32_RTN_B64: two conditional dword exchanges from DATA0[0:1].
                    "DsCondxchg32RtnB64" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData0 + 1),
                    ],
                    "DsWriteAddtidB32" => [Gen5Operand.Scalar(124), Gen5Operand.Vector(vectorData0)],
                    "DsWriteB32" or "DsWriteB8" or "DsWriteB16" or
                    "DsWriteB8D16Hi" or "DsWriteB16D16Hi" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                    ],
                    "DsWriteB64" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData0 + 1),
                    ],
                    "DsWrite2B64" or "DsWrite2St64B64" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData0 + 1),
                        Gen5Operand.Vector(vectorData1),
                        Gen5Operand.Vector(vectorData1 + 1),
                    ],
                    "DsWriteB96" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData0 + 1),
                        Gen5Operand.Vector(vectorData0 + 2),
                    ],
                    "DsWriteB128" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData0 + 1),
                        Gen5Operand.Vector(vectorData0 + 2),
                        Gen5Operand.Vector(vectorData0 + 3),
                    ],
                    "DsWrite2B32" or "DsWrite2St64B32" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData1),
                    ],
                    "DsSwizzleB32" => [Gen5Operand.Vector(vectorAddress)],
                    "DsBpermuteB32" or "DsPermuteB32" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                    ],
                    "DsNop" => [],
                    // A D16 read keeps the other half of its destination.
                    "DsReadU8D16" or "DsReadU8D16Hi" or "DsReadI8D16" or "DsReadI8D16Hi" or
                    "DsReadU16D16" or "DsReadU16D16Hi" => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorDestination),
                    ],
                    // 64-bit LDS atomics take a register pair per data operand.
                    _ when IsDataShareAtomic64(opcode) && IsDataShareTwoOperandAtomic(opcode) => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData0 + 1),
                        Gen5Operand.Vector(vectorData1),
                        Gen5Operand.Vector(vectorData1 + 1),
                    ],
                    _ when IsDataShareAtomic64(opcode) => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData0 + 1),
                    ],
                    // DS_CMPST operand order is reversed vs buffer/image cmpswap:
                    // DATA0 holds the comparator, DATA1 holds the new value. GFX10
                    // DS_MIN/MAX_F32 use DATA0 as the replacement value and DATA1 as the
                    // floating-point compare operand; DS_MSKOR DATA0 as the mask.
                    _ when IsDataShareAtomic(opcode) && IsDataShareTwoOperandAtomic(opcode) => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                        Gen5Operand.Vector(vectorData1),
                    ],
                    _ when IsDataShareAtomic(opcode) => [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorData0),
                    ],
                    _ => [Gen5Operand.Vector(vectorAddress)],
                };
                destinations = opcode switch
                {
                    "DsAppend" or "DsConsume" or "DsOrderedCount" => [
                        Gen5Operand.Vector(vectorDestination),
                    ],
                    "DsCondxchg32RtnB64" => [
                        Gen5Operand.Vector(vectorDestination),
                        Gen5Operand.Vector(vectorDestination + 1),
                    ],
                    "DsReadB32" or "DsReadI8" or "DsReadU8" or "DsReadI16" or "DsReadU16" or
                    "DsReadU8D16" or "DsReadU8D16Hi" or "DsReadI8D16" or "DsReadI8D16Hi" or
                    "DsReadU16D16" or "DsReadU16D16Hi" or "DsReadAddtidB32" or
                    "DsSwizzleB32" or "DsBpermuteB32" or "DsPermuteB32" => [
                        Gen5Operand.Vector(vectorDestination),
                    ],
                    "DsReadB64" or "DsRead2B32" or "DsRead2St64B32" => [
                        Gen5Operand.Vector(vectorDestination),
                        Gen5Operand.Vector(vectorDestination + 1),
                    ],
                    "DsReadB96" => [
                        Gen5Operand.Vector(vectorDestination),
                        Gen5Operand.Vector(vectorDestination + 1),
                        Gen5Operand.Vector(vectorDestination + 2),
                    ],
                    "DsReadB128" or "DsRead2B64" or "DsRead2St64B64" or "DsWrxchg2RtnB64" or "DsWrxchg2St64RtnB64" => [
                        Gen5Operand.Vector(vectorDestination),
                        Gen5Operand.Vector(vectorDestination + 1),
                        Gen5Operand.Vector(vectorDestination + 2),
                        Gen5Operand.Vector(vectorDestination + 3),
                    ],
                    // A returning 64-bit atomic, or a WRXCHG2 of two dwords, returns a pair.
                    _ when opcode.Contains("Rtn", StringComparison.Ordinal) &&
                        (IsDataShareAtomic64(opcode) || opcode.StartsWith("DsWrxchg2", StringComparison.Ordinal)) => [
                        Gen5Operand.Vector(vectorDestination),
                        Gen5Operand.Vector(vectorDestination + 1),
                    ],
                    _ when IsDataShareAtomic(opcode) &&
                        opcode.Contains("Rtn", StringComparison.Ordinal) => [
                        Gen5Operand.Vector(vectorDestination),
                    ],
                    _ => [],
                };
                break;
            }
            case Gen5ShaderEncoding.Vintrp:
                sources = [Gen5Operand.Vector(word & 0xFF)];
                destinations = [Gen5Operand.Vector((word >> 18) & 0xFF)];
                control = new Gen5InterpolationControl(
                    (word >> 10) & 0x3F,
                    (word >> 8) & 0x3)
                {
                    Parameter = opcode == "VInterpMovF32" ? word & 0xFF : 0,
                };
                break;
            case Gen5ShaderEncoding.Flat:
            {
                var extra = words[1];
                var vectorAddress = extra & 0xFF;
                var sourceVectorRegister = (extra >> 8) & 0xFF;
                var destinationVectorRegister = (extra >> 24) & 0xFF;
                var scalarAddress = (extra >> 16) & 0x7F;
                var usesFlatAddress = opcode.StartsWith(
                    "Flat",
                    StringComparison.Ordinal);
                var usesScratchAddress = opcode.StartsWith(
                    "Scratch",
                    StringComparison.Ordinal);
                var memoryOpcode = usesFlatAddress
                    ? "Global" + opcode["Flat".Length..]
                    : usesScratchAddress
                        ? "Global" + opcode["Scratch".Length..]
                        : opcode;
                var dwordCount = memoryOpcode switch
                {
                    "GlobalLoadUbyte" or
                    "GlobalLoadSbyte" or
                    "GlobalLoadUshort" or
                    "GlobalLoadSshort" or
                    "GlobalLoadUbyteD16" or
                    "GlobalLoadUbyteD16Hi" or
                    "GlobalLoadSbyteD16" or
                    "GlobalLoadSbyteD16Hi" or
                    "GlobalLoadShortD16" or
                    "GlobalLoadShortD16Hi" or
                    "GlobalStoreByte" or
                    "GlobalStoreByteD16Hi" or
                    "GlobalStoreShort" or
                    "GlobalStoreShortD16Hi" or
                    "GlobalStoreDword" or
                    "GlobalLoadDwordAddtid" or
                    "GlobalStoreDwordAddtid" => 1u,
                    // An atomic's count is the width of the memory it operates on.
                    _ when memoryOpcode.StartsWith("GlobalAtomic", StringComparison.Ordinal) =>
                        memoryOpcode.EndsWith("X2", StringComparison.Ordinal) ? 2u : 1u,
                    "GlobalLoadDword" => 1u,
                    "GlobalLoadDwordx2" => 2u,
                    "GlobalLoadDwordx3" => 3u,
                    "GlobalLoadDwordx4" => 4u,
                    "GlobalStoreDwordx2" => 2u,
                    "GlobalStoreDwordx3" => 3u,
                    "GlobalStoreDwordx4" => 4u,
                    _ => 0u,
                };
                sources = usesFlatAddress
                    ?
                    [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Vector(vectorAddress + 1),
                    ]
                    : usesScratchAddress && scalarAddress < 125
                        ? [Gen5Operand.Scalar(scalarAddress)]
                        : usesScratchAddress
                            ? [Gen5Operand.Vector(vectorAddress)]
                    :
                    [
                        Gen5Operand.Vector(vectorAddress),
                        Gen5Operand.Scalar(scalarAddress),
                    ];
                var addThreadId = memoryOpcode is "GlobalLoadDwordAddtid" or "GlobalStoreDwordAddtid";
                if (addThreadId)
                {
                    // The address is SADDR + offset + lane * 4; there is no VADDR to read.
                    sources = [Gen5Operand.Scalar(scalarAddress)];
                }

                var isLoad = memoryOpcode.StartsWith("GlobalLoad", StringComparison.Ordinal);
                var isStore = memoryOpcode.StartsWith("GlobalStore", StringComparison.Ordinal);
                var isAtomic = memoryOpcode.StartsWith("GlobalAtomic", StringComparison.Ordinal);
                var globallyCoherent = ((word >> 16) & 1) != 0;
                if (isStore || isAtomic)
                {
                    // CMPSWAP and FCMPSWAP read the comparator from the registers after the new value.
                    var sourceCount = memoryOpcode.StartsWith("GlobalAtomicCmpswap", StringComparison.Ordinal) ||
                        memoryOpcode.StartsWith("GlobalAtomicFcmpswap", StringComparison.Ordinal)
                        ? dwordCount * 2
                        : dwordCount;
                    sources = [.. sources, .. Enumerable
                        .Range((int)sourceVectorRegister, checked((int)sourceCount))
                        .Select(index => Gen5Operand.Vector((uint)index))];
                }
                else if (isLoad && memoryOpcode.Contains("D16", StringComparison.Ordinal))
                {
                    // A partial load preserves the other half of the destination.
                    sources = [.. sources, Gen5Operand.Vector(destinationVectorRegister)];
                }

                destinations = isLoad || (isAtomic && globallyCoherent)
                    ? Enumerable
                        .Range((int)destinationVectorRegister, checked((int)dwordCount))
                        .Select(index => Gen5Operand.Vector((uint)index))
                        .ToArray()
                    : [];
                control = new Gen5GlobalMemoryControl(
                    dwordCount,
                    vectorAddress,
                    sourceVectorRegister,
                    destinationVectorRegister,
                    usesFlatAddress ? uint.MaxValue : scalarAddress,
                    FlatOffset(word, usesFlatAddress),
                    globallyCoherent,
                    ((word >> 17) & 1) != 0,
                    usesFlatAddress,
                    Dlc: ((word >> 12) & 1) != 0,
                    AddThreadId: addThreadId);
                break;
            }
            case Gen5ShaderEncoding.Mubuf:
            {
                var extra = words[1];
                if (opcode is "BufferGl0Inv" or "BufferGl1Inv")
                {
                    // Cache invalidations: the encoding's address, data and resource fields
                    // are unused (LLVM prints no operand), so no memory access is planned.
                    sources = [];
                    destinations = [];
                    break;
                }

                var vectorAddress = extra & 0xFF;
                var vectorData = (extra >> 8) & 0xFF;
                var scalarResource = ((extra >> 16) & 0x1F) * 4;
                var scalarOffset = (extra >> 24) & 0xFF;
                var dwordCount = opcode switch
                {
                    "BufferLoadFormatX" => 1u,
                    "BufferLoadFormatXy" => 2u,
                    "BufferLoadFormatXyz" => 3u,
                    "BufferLoadFormatXyzw" => 4u,
                    "BufferStoreFormatX" => 1u,
                    "BufferStoreFormatXy" => 2u,
                    "BufferStoreFormatXyz" => 3u,
                    "BufferStoreFormatXyzw" => 4u,
                    "BufferLoadUbyte" or
                    "BufferLoadSbyte" or
                    "BufferLoadUshort" or
                    "BufferLoadSshort" or
                    "BufferStoreByte" or
                    "BufferStoreByteD16Hi" or
                    "BufferStoreShort" or
                    "BufferStoreShortD16Hi" or
                    "BufferLoadUbyteD16" or
                    "BufferLoadUbyteD16Hi" or
                    "BufferLoadSbyteD16" or
                    "BufferLoadSbyteD16Hi" or
                    "BufferLoadShortD16" or
                    "BufferLoadShortD16Hi" => 1u,
                    // D16 formatted components pack two to a register.
                    "BufferLoadFormatD16X" or "BufferLoadFormatD16Xy" or "BufferLoadFormatD16HiX" or
                    "BufferStoreFormatD16X" or "BufferStoreFormatD16Xy" or "BufferStoreFormatD16HiX" => 1u,
                    "BufferLoadFormatD16Xyz" or "BufferLoadFormatD16Xyzw" or
                    "BufferStoreFormatD16Xyz" or "BufferStoreFormatD16Xyzw" => 2u,
                    "BufferLoadDword" => 1u,
                    "BufferLoadDwordx2" => 2u,
                    "BufferLoadDwordx3" => 3u,
                    "BufferLoadDwordx4" => 4u,
                    "BufferStoreDword" => 1u,
                    "BufferStoreDwordx2" => 2u,
                    "BufferStoreDwordx3" => 3u,
                    "BufferStoreDwordx4" => 4u,
                    // An atomic's count is its data registers: CMPSWAP's new value and
                    // comparator, two per operand for the X2 forms.
                    "BufferAtomicCmpswap" or "BufferAtomicFcmpswap" => 2u,
                    "BufferAtomicCmpswapX2" or "BufferAtomicFcmpswapX2" => 4u,
                    _ when opcode.StartsWith("BufferAtomic", StringComparison.Ordinal) =>
                        opcode.EndsWith("X2", StringComparison.Ordinal) ? 2u : 1u,
                    _ => 0u,
                };
                var tfe = ((extra >> 23) & 1) != 0;
                var lds = ((word >> 16) & 1) != 0;
                sources =
                [
                    Gen5Operand.Vector(vectorAddress),
                    Gen5Operand.Scalar(scalarResource),
                    Gen5Operand.Source(scalarOffset, literal),
                ];
                if (lds)
                {
                    // A load to LDS writes at M0[15:0] + offset + lane * 4 and no VGPR.
                    sources = [.. sources, Gen5Operand.Scalar(124)];
                }

                destinations = lds
                    ? []
                    : Enumerable
                        .Range((int)vectorData, checked((int)(dwordCount + (tfe ? 1u : 0u))))
                        .Select(index => Gen5Operand.Vector((uint)index))
                        .ToArray();
                control = new Gen5BufferMemoryControl(
                    dwordCount,
                    vectorAddress,
                    vectorData,
                    scalarResource,
                    (int)(word & 0xFFF),
                    ((word >> 13) & 1) != 0,
                    ((word >> 12) & 1) != 0,
                    ((word >> 14) & 1) != 0,
                    ((extra >> 22) & 1) != 0,
                    Dlc: ((word >> 15) & 1) != 0,
                    Tfe: tfe,
                    Lds: lds);
                break;
            }
            case Gen5ShaderEncoding.Mtbuf:
            {
                var extra = words[1];
                var vectorAddress = extra & 0xFF;
                var vectorData = (extra >> 8) & 0xFF;
                var scalarResource = ((extra >> 16) & 0x1F) * 4;
                var scalarOffset = (extra >> 24) & 0xFF;
                var dwordCount = opcode switch
                {
                    "TBufferLoadFormatX" or "TBufferStoreFormatX" => 1u,
                    "TBufferLoadFormatXy" or "TBufferStoreFormatXy" => 2u,
                    "TBufferLoadFormatXyz" or "TBufferStoreFormatXyz" => 3u,
                    "TBufferLoadFormatXyzw" or "TBufferStoreFormatXyzw" => 4u,
                    "TBufferLoadFormatD16X" or "TBufferLoadFormatD16Xy" or
                    "TBufferStoreFormatD16X" or "TBufferStoreFormatD16Xy" => 1u,
                    "TBufferLoadFormatD16Xyz" or "TBufferLoadFormatD16Xyzw" or
                    "TBufferStoreFormatD16Xyz" or "TBufferStoreFormatD16Xyzw" => 2u,
                    _ => 0u,
                };
                var typedTfe = ((extra >> 23) & 1) != 0;
                sources =
                [
                    Gen5Operand.Vector(vectorAddress),
                    Gen5Operand.Scalar(scalarResource),
                    Gen5Operand.Source(scalarOffset, literal),
                ];
                destinations = Enumerable
                    .Range((int)vectorData, checked((int)(dwordCount + (typedTfe ? 1u : 0u))))
                    .Select(index => Gen5Operand.Vector((uint)index))
                    .ToArray();
                control = new Gen5BufferMemoryControl(
                    dwordCount,
                    vectorAddress,
                    vectorData,
                    scalarResource,
                    (int)(word & 0xFFF),
                    ((word >> 13) & 1) != 0,
                    ((word >> 12) & 1) != 0,
                    ((word >> 14) & 1) != 0,
                    ((extra >> 22) & 1) != 0,
                    Typed: true,
                    TypedFormat: (word >> 19) & 0x7F,
                    Dlc: ((word >> 15) & 1) != 0,
                    Tfe: typedTfe);
                break;
            }
            case Gen5ShaderEncoding.Mimg:
            {
                var extra = words[1];
                var vectorAddress = extra & 0xFF;
                var vectorData = (extra >> 8) & 0xFF;
                var scalarResource = ((extra >> 16) & 0x1F) * 4;
                var scalarSampler = ((extra >> 21) & 0x1F) * 4;
                var addressRegisters = new List<uint>(1 + Math.Max(0, words.Length - 2) * 4)
                {
                    vectorAddress,
                };
                for (var wordIndex = 2; wordIndex < words.Length; wordIndex++)
                {
                    for (var shift = 0; shift < 32; shift += 8)
                    {
                        addressRegisters.Add((words[wordIndex] >> shift) & 0xFF);
                    }
                }

                if (opcode.StartsWith("ImageBvh", StringComparison.Ordinal))
                {
                    var a16 = ((extra >> 30) & 1) != 0;
                    var rayControl = new Gen5RayIntersectControl(vectorAddress, addressRegisters, vectorData, scalarResource, a16);
                    // Node pointer (2 dwords for BVH64), extent, origin, then direction and
                    // inverse direction, which A16 packs as halves into three dwords.
                    var addressCount = (opcode == "ImageBvh64IntersectRay" ? 2 : 1) + 1 + 3 + (a16 ? 3 : 6);
                    var raySources = new List<Gen5Operand>(addressCount + 1);
                    for (var component = 0; component < addressCount; component++)
                    {
                        raySources.Add(Gen5Operand.Vector(rayControl.GetAddressRegister(component)));
                    }

                    raySources.Add(Gen5Operand.Scalar(scalarResource));
                    sources = raySources;
                    destinations = Enumerable
                        .Range((int)vectorData, (int)Gen5RayIntersectControl.ResultDwords)
                        .Select(index => Gen5Operand.Vector((uint)index))
                        .ToArray();
                    control = rayControl;
                    break;
                }

                var imageSources = new List<Gen5Operand>(addressRegisters.Count + 2);
                foreach (var addressRegister in addressRegisters)
                {
                    imageSources.Add(Gen5Operand.Vector(addressRegister));
                }

                imageSources.Add(Gen5Operand.Scalar(scalarResource));
                imageSources.Add(Gen5Operand.Scalar(scalarSampler));
                sources = imageSources;
                destinations = opcode.StartsWith("ImageStore", StringComparison.Ordinal)
                    ? []
                    : opcode is "ImageBvhIntersectRay" or "ImageBvh64IntersectRay"
                        ? Enumerable.Range((int)vectorData, 4)
                            .Select(index => Gen5Operand.Vector((uint)index))
                            .ToArray()
                        : [Gen5Operand.Vector(vectorData)];
                var dimension = (word >> 3) & 0x7;
                control = new Gen5ImageControl(
                    (word >> 8) & 0xF,
                    vectorAddress,
                    addressRegisters,
                    vectorData,
                    scalarResource,
                    scalarSampler,
                    dimension,
                    dimension is 4 or 5 or 7,
                    ((word >> 13) & 1) != 0,
                    ((word >> 25) & 1) != 0,
                    ((extra >> 30) & 1) != 0,
                    ((extra >> 31) & 1) != 0)
                {
                    // gfx10 MIMG word 0 (SIInstrFormats.td MIMGe_gfx10): DLC [7], UNRM [12],
                    // R128 [15], TFE [16], LWE [17].
                    Unorm = ((word >> 12) & 1) != 0,
                    Dlc = ((word >> 7) & 1) != 0,
                    R128 = ((word >> 15) & 1) != 0,
                    Tfe = ((word >> 16) & 1) != 0,
                    Lwe = ((word >> 17) & 1) != 0,
                };
                break;
            }
            case Gen5ShaderEncoding.Exp:
            {
                var extra = words[1];
                sources =
                [
                    Gen5Operand.Vector(extra & 0xFF),
                    Gen5Operand.Vector((extra >> 8) & 0xFF),
                    Gen5Operand.Vector((extra >> 16) & 0xFF),
                    Gen5Operand.Vector((extra >> 24) & 0xFF),
                ];
                control = new Gen5ExportControl(
                    (word >> 4) & 0x3F,
                    word & 0xF,
                    ((word >> 10) & 1) != 0,
                    ((word >> 11) & 1) != 0,
                    ((word >> 12) & 1) != 0);
                break;
            }
        }

        return new Gen5ShaderInstruction(pc, encoding, opcode, words, sources, destinations, control);
    }

    private static Gen5DppControl CreateDppControl(uint word) =>
        new(
            (word >> 8) & 0x1FF,
            ((word >> 18) & 1) != 0,
            ((word >> 19) & 1) != 0,
            ((word >> 21) & 1) | (((word >> 23) & 1) << 1),
            ((word >> 20) & 1) | (((word >> 22) & 1) << 1),
            (word >> 24) & 0xF,
            (word >> 28) & 0xF);

    private static Gen5SdwaControl CreateSdwaControl(
        uint word,
        bool isCompare,
        bool hasSource1)
    {
        var scalarDestination = isCompare
            ? ((word >> 15) & 1) != 0
                ? (word >> 8) & 0x7Fu
                : 106u
            : (uint?)null;
        return new Gen5SdwaControl(
            isCompare ? 6u : (word >> 8) & 0x7u,
            isCompare ? 0u : (word >> 11) & 0x3u,
            (word >> 16) & 0x7u,
            hasSource1 ? (word >> 24) & 0x7u : 6u,
            ((word >> 19) & 1) != 0,
            hasSource1 && ((word >> 27) & 1) != 0,
            ((word >> 21) & 1) | (hasSource1 ? ((word >> 29) & 1) << 1 : 0),
            ((word >> 20) & 1) | (hasSource1 ? ((word >> 28) & 1) << 1 : 0),
            isCompare ? 0u : (word >> 14) & 0x3u,
            !isCompare && ((word >> 13) & 1) != 0,
            scalarDestination);
    }

    private static int MinimumEncodingDwords(Gen5ShaderEncoding encoding) => encoding switch
    {
        Gen5ShaderEncoding.Vop3 or
        Gen5ShaderEncoding.Smem or
        Gen5ShaderEncoding.Mubuf or
        Gen5ShaderEncoding.Mtbuf or
        Gen5ShaderEncoding.Ds or
        Gen5ShaderEncoding.Flat or
        Gen5ShaderEncoding.Vop3p or
        Gen5ShaderEncoding.Mimg or
        Gen5ShaderEncoding.Exp => 2,
        _ => 1,
    };

    private static uint ScalarLoadDwordCount(string opcode) => opcode switch
    {
        "SLoadDword" or "SBufferLoadDword" => 1,
        "SLoadDwordx2" or "SBufferLoadDwordx2" => 2,
        "SLoadDwordx4" or "SBufferLoadDwordx4" => 4,
        "SLoadDwordx8" or "SBufferLoadDwordx8" => 8,
        "SLoadDwordx16" or "SBufferLoadDwordx16" => 16,
        _ => 0,
    };

    private static uint ScalarMemoryDwordCount(string opcode) => opcode switch
    {
        "ScratchStoreDword" => 1,
        "ScratchStoreDwordx2" => 2,
        "ScratchStoreDwordx4" => 4,
        _ => 0,
    };

    // RDNA2 ISA FLAT field table: OFFSET [11:0]; Scratch/Global are 12-bit signed, Flat
    // "must be positive". DLC is bit 12 and never part of the address.
    internal static int FlatOffset(uint word, bool isFlatSegment) =>
        isFlatSegment ? (int)(word & 0x7FF) : SignExtend(word & 0xFFF, 12);

    private static int SignExtend(uint value, int bits)
    {
        var shift = 32 - bits;
        return (int)(value << shift) >> shift;
    }

    private static void AddFeatureCount(Dictionary<string, int> counts, string key)
    {
        counts.TryGetValue(key, out var count);
        counts[key] = count + 1;
    }

    private static bool TryReadUInt32(CpuContext ctx, ulong address, out uint value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (!ctx.Memory.TryRead(address, bytes))
        {
            value = 0;
            return false;
        }

        value = BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }

    private readonly record struct ShaderDecodeInfo(
        int InstructionCount,
        Dictionary<string, int> Counts,
        Dictionary<string, int> FeatureCounts,
        Dictionary<string, int> MimgCounts,
        List<string> Details)
    {
        public static ShaderDecodeInfo Create(Gen5ShaderProgram program)
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            var featureCounts = new Dictionary<string, int>(StringComparer.Ordinal);
            var mimgCounts = new Dictionary<string, int>(StringComparer.Ordinal);
            var details = new List<string>();
            foreach (var instruction in program.Instructions)
            {
                AddFeatureCount(counts, instruction.Opcode);
                AddFeatureCount(featureCounts, ClassifyInstruction(instruction.Opcode));
                if (instruction.Control is Gen5ImageControl image)
                {
                    AddFeatureCount(mimgCounts, $"{instruction.Opcode}/dmask=0x{image.Dmask:X}");
                }

                if (details.Count < 16 && DescribeInstruction(instruction) is { } detail)
                {
                    details.Add(detail);
                }
            }

            return new ShaderDecodeInfo(
                program.Instructions.Count,
                counts,
                featureCounts,
                mimgCounts,
                details);
        }

        public override string ToString()
        {
            var builder = new StringBuilder();
            builder.Append("ins=");
            builder.Append(InstructionCount);
            AppendCounts(builder, " features=", FeatureCounts, 16);
            builder.Append(" ops=");
            AppendCounts(builder, string.Empty, Counts, 128);
            AppendCounts(builder, " mimg=", MimgCounts, 12);
            AppendDetails(builder, Details, 10);

            return builder.ToString();
        }

        private static string? DescribeInstruction(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is Gen5ImageControl image)
            {
                var addressRegisters = string.Join(
                    '/',
                    image.AddressRegisters.Select(register => $"v{register}"));
                return
                    $"{instruction.Opcode}@0x{instruction.Pc:X}:dm=0x{image.Dmask:X}," +
                    $"va={addressRegisters},vd=v{image.VectorData}," +
                    $"sr=s{image.ScalarResource},ss=s{image.ScalarSampler}," +
                    $"dim={image.Dimension},da={(image.IsArray ? 1 : 0)}," +
                    $"a16={(image.A16 ? 1 : 0)},d16={(image.D16 ? 1 : 0)}," +
                    $"glc={(image.Glc ? 1 : 0)}," +
                    $"slc={(image.Slc ? 1 : 0)}";
            }

            if (instruction.Control is Gen5ExportControl export)
            {
                return
                    $"Exp@0x{instruction.Pc:X}:target=0x{export.Target:X}," +
                    $"en=0x{export.EnableMask:X},compr={(export.Compressed ? 1 : 0)}," +
                    $"done={(export.Done ? 1 : 0)},vm={(export.ValidMask ? 1 : 0)}," +
                    $"src={string.Join('/', instruction.Sources)}";
            }

            if (instruction.Control is Gen5InterpolationControl interpolation)
            {
                return
                    $"{instruction.Opcode}@0x{instruction.Pc:X}:" +
                    $"attr={interpolation.Attribute},chan={interpolation.Channel}," +
                    $"src={instruction.Sources[0]},dst={instruction.Destinations[0]}";
            }

            return null;
        }

        private static void AppendCounts(
            StringBuilder builder,
            string prefix,
            Dictionary<string, int> counts,
            int limit)
        {
            if (counts.Count == 0)
            {
                return;
            }

            builder.Append(prefix);
            var written = 0;
            foreach (var (name, count) in counts)
            {
                if (written != 0)
                {
                    builder.Append(',');
                }

                builder.Append(name);
                builder.Append(':');
                builder.Append(count);
                written++;
                if (written == limit && counts.Count > written)
                {
                    builder.Append(",...");
                    break;
                }
            }
        }

        private static void AppendDetails(StringBuilder builder, List<string> details, int limit)
        {
            if (details.Count == 0)
            {
                return;
            }

            builder.Append(" detail=");
            var written = 0;
            foreach (var detail in details)
            {
                if (written != 0)
                {
                    builder.Append(';');
                }

                builder.Append(detail);
                written++;
                if (written == limit && details.Count > written)
                {
                    builder.Append(";...");
                    break;
                }
            }
        }
    }
}
