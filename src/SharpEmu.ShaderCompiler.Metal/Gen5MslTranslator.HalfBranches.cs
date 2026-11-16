// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal;

// Forward EXEC branches each half of a one-wave wave64 compute group takes on its own.
//
// The guest compiler closes nearly every if with S_AND_SAVEEXEC and an S_CBRANCH_EXECZ over the
// body: the wave skips it when no lane of all 64 is left. Bridged across two simdgroups, that
// test needs the other half's lanes, so the S_AND_SAVEEXEC's EXEC is exchanged through
// threadgroup memory at a threadgroup barrier first. Demon's Souls' tile-lighting kernels run
// dozens of those per thread.
//
// A half whose own lanes are all off can skip such a body by itself when running it could not
// have changed anything the other half or the code after it sees:
//   - every lane of that half is off throughout, so the body's vector work writes nothing there;
//   - the body holds no read of the other half's lanes, LDS access or exchange, so the running
//     half waits on nobody (DPP and its row-local shuffles stay inside the half), and no EXEC
//     write that could turn on a lane that was off when it began: only narrowing writes and
//     restores of masks that, on every path to the restore, the body last wrote with lanes of
//     its own EXEC;
//   - every scalar register, VCC or SCC value the body writes is dead where it joins the code
//     after it, so the halves differ only in values nothing reads (each operand counts the
//     registers it really reads: one for a 32-bit operand, a buffer resource's four);
//   - the body is entered only through the branch and leaves only to the join, and every
//     S_CBRANCH_EXECZ inside it is half-local too (one the wave takes as a whole exchanges).
// Then the branch tests this half's EXEC (simd_ballot) and the mask-ballot elision drops the
// exchange no reader is left for. The body's emitted text is checked again once the ballots
// are resolved; one that still exchanges has the program translated without these branches.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const int SccLiveSlot = 128;

        /// <summary>Blocks whose closing S_CBRANCH_EXECZ each half takes on its own (null when none).</summary>
        private bool[]? _halfLocalBranchBlocks;

        /// <summary>The blocks those branches skip, which must hold no exchange.</summary>
        private bool[]? _halfLocalBranchBodies;

        /// <summary>Set when a half-local branch body still exchanges once the mask ballots are
        /// resolved; the program is then compiled again without half-local branches.</summary>
        internal bool HalfLocalBranchExchanged { get; private set; }

        private static readonly bool HalfBranchReport =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_HALF_BRANCH_REPORT") == "1";

        /// <summary>Why the last body was refused, for <see cref="HalfBranchReport"/>.</summary>
        private string? _halfBranchRefusal;

        private bool RefuseHalfBranch(string reason)
        {
            _halfBranchRefusal = reason;
            return false;
        }

        private void ReportHalfBranch(Gen5ShaderInstruction branch, string verdict)
        {
            if (HalfBranchReport)
            {
                Console.Error.WriteLine($"[MSL][HALF-BRANCH] 0x{_request.Hash:X16} pc=0x{branch.Pc:X4} {verdict}");
            }
        }

        private bool IsHalfLocalBranchBlock(int block) =>
            _halfLocalBranchBlocks is { } blocks && block < blocks.Length && blocks[block];

        private bool HalfLocalBranchesApply =>
            _options.HalfLocalExecBranches && _stage == Gen5MslStage.Compute && _waveLaneCount == 64 &&
            ComputeThreads == 64 && IsWave64 && !IsBridgedComputeWaves && !IsBridgedMeshWave64;

        private void FindHalfLocalBranches(IReadOnlyList<ShaderBlock> blocks)
        {
            if (!HalfLocalBranchesApply)
            {
                if (HalfBranchReport && _stage == Gen5MslStage.Compute && IsWave64)
                {
                    Console.Error.WriteLine(
                        $"[MSL][HALF-BRANCH] 0x{_request.Hash:X16} program not applicable: threads={ComputeThreads} " +
                        $"bridged={IsBridgedComputeWaves} option={_options.HalfLocalExecBranches}");
                }

                return;
            }

            var instructions = _request.Program.Instructions;
            foreach (var instruction in instructions)
            {
                if (instruction.Opcode.StartsWith("SMovrel", StringComparison.Ordinal) ||
                    instruction.Opcode is "SSetpcB64" or "SSwappcB64" or "SRfeB64" or "SGetpcB64")
                {
                    if (HalfBranchReport)
                    {
                        Console.Error.WriteLine($"[MSL][HALF-BRANCH] 0x{_request.Hash:X16} program refused: {instruction.Opcode}");
                    }

                    return;
                }
            }

            var count = blocks.Count;
            var successors = new int[count][];
            var takenTargets = new int[count];
            for (var block = 0; block < count; block++)
            {
                var last = instructions[blocks[block].EndIndex - 1];
                var next = block + 1 < count ? block + 1 : ExitTarget;
                takenTargets[block] = int.MinValue;
                if (Ir.Gen5IrBranchResolver.IsTerminator(last))
                {
                    successors[block] = [];
                    continue;
                }

                var branch = last.Opcode == "SBranch" || last.Opcode.StartsWith("SCbranch", StringComparison.Ordinal);
                if (!branch)
                {
                    successors[block] = next == ExitTarget ? [] : [next];
                    continue;
                }

                int taken;
                if (IsExitBranchTarget(instructions, last))
                {
                    taken = ExitTarget;
                }
                else if (!TryGetBranchTargetBlock(blocks, last, out taken))
                {
                    if (HalfBranchReport)
                    {
                        Console.Error.WriteLine($"[MSL][HALF-BRANCH] 0x{_request.Hash:X16} program refused: unresolved branch target");
                    }

                    return;
                }

                takenTargets[block] = taken;
                var list = new List<int>(2);
                if (taken != ExitTarget)
                {
                    list.Add(taken);
                }

                if (last.Opcode != "SBranch" && next != ExitTarget)
                {
                    list.Add(next);
                }

                successors[block] = [.. list];
            }

            var (liveIn, sccLiveIn) = ScalarLiveness(blocks, successors);
            var accepted = new bool[count];
            var candidate = new bool[count];
            for (var block = 0; block + 1 < count; block++)
            {
                var last = instructions[blocks[block].EndIndex - 1];
                var join = takenTargets[block];
                if (last.Opcode != "SCbranchExecz")
                {
                    continue;
                }

                candidate[block] = true;
                if (join <= block + 1)
                {
                    ReportHalfBranch(last, "backward or empty body");
                    continue;
                }

                if (!IsHalfLocalBody(blocks, successors, block + 1, join, liveIn, sccLiveIn))
                {
                    ReportHalfBranch(last, _halfBranchRefusal ?? "unrecorded");
                    continue;
                }

                accepted[block] = true;
            }

            // A body that holds an S_CBRANCH_EXECZ the wave still takes as a whole holds that
            // branch's exchange, so it cannot be skipped by one half either.
            for (var changed = true; changed;)
            {
                changed = false;
                for (var block = 0; block < count; block++)
                {
                    if (!accepted[block])
                    {
                        continue;
                    }

                    for (var inside = block + 1; inside < takenTargets[block]; inside++)
                    {
                        if (candidate[inside] && !accepted[inside])
                        {
                            accepted[block] = false;
                            changed = true;
                            ReportHalfBranch(instructions[blocks[block].EndIndex - 1], "nested whole-wave execz");
                            break;
                        }
                    }
                }
            }

            bool[]? branches = null;
            bool[]? bodies = null;
            for (var block = 0; block < count; block++)
            {
                if (!accepted[block])
                {
                    continue;
                }

                ReportHalfBranch(instructions[blocks[block].EndIndex - 1], "accepted");
                branches ??= new bool[count];
                bodies ??= new bool[count];
                branches[block] = true;
                for (var inside = block + 1; inside < takenTargets[block]; inside++)
                {
                    bodies[inside] = true;
                }
            }

            _halfLocalBranchBlocks = branches;
            _halfLocalBranchBodies = bodies;
        }

        /// <summary>Whether blocks [first, join) form a body a half may skip alone (see the file
        /// comment).</summary>
        private bool IsHalfLocalBody(IReadOnlyList<ShaderBlock> blocks, int[][] successors, int first, int join,
            UInt128[] liveIn, bool[] sccLiveIn)
        {
            var instructions = _request.Program.Instructions;
            for (var block = 0; block < blocks.Count; block++)
            {
                var inside = block >= first && block < join;
                foreach (var successor in successors[block])
                {
                    if (inside ? successor < first || successor > join
                        : successor > first && successor < join || (successor == first && block != first - 1))
                    {
                        return RefuseHalfBranch(inside ? "body leaves other than to the join" : "body entered other than by the branch");
                    }
                }

                if (inside && successors[block].Length == 0)
                {
                    // An exit in the body would end the half that ran it, and the guest wave with it.
                    return RefuseHalfBranch("exit in body");
                }
            }

            // The scalar registers and SCC the body may write.
            UInt128 writes = 0;
            var writesScc = false;
            for (var block = first; block < join; block++)
            {
                for (var index = blocks[block].StartIndex; index < blocks[block].EndIndex; index++)
                {
                    var instruction = instructions[index];
                    if (!IsHalfLocalBodyInstruction(instruction))
                    {
                        return RefuseHalfBranch($"instruction {instruction.Encoding}:{instruction.Opcode}{(instruction.Control is Gen5DppControl or Gen5Dpp8Control ? "+dpp" : string.Empty)}");
                    }

                    var (_, _, mayWrite, mayWriteScc) = ScalarEffects(instruction);
                    writes |= mayWrite;
                    writesScc |= mayWriteScc;
                }
            }

            // Every EXEC write must leave EXEC a subset of what it was when the body began. A write
            // may restore a pair only where that pair is known to hold such a subset: on every path
            // from the body's start to the write, the pair's last write is one with lanes of EXEC
            // (WritesExecSubsetPair). That is a must-analysis over the body's blocks: nothing is
            // known at the start (a pair from outside the body may hold anything), the paths meet
            // by intersection, and any other write of a register forgets its pair.
            var bodyCount = join - first;
            var predecessors = new List<int>[bodyCount];
            for (var block = first; block < join; block++)
            {
                predecessors[block - first] = [];
            }

            for (var block = first; block < join; block++)
            {
                foreach (var successor in successors[block])
                {
                    if (successor >= first && successor < join)
                    {
                        predecessors[successor - first].Add(block);
                    }
                }
            }

            var known = new UInt128[bodyCount];
            Array.Fill(known, UInt128.MaxValue);
            known[0] = 0;
            for (var changed = true; changed;)
            {
                changed = false;
                for (var block = first; block < join; block++)
                {
                    var state = block == first ? 0 : UInt128.MaxValue;
                    foreach (var predecessor in predecessors[block - first])
                    {
                        state &= SubsetPairsAfter(blocks[predecessor], known[predecessor - first], out _);
                    }

                    if (state != known[block - first])
                    {
                        known[block - first] = state;
                        changed = true;
                    }
                }
            }

            for (var block = first; block < join; block++)
            {
                SubsetPairsAfter(blocks[block], known[block - first], out var widening);
                if (widening is not null)
                {
                    return RefuseHalfBranch($"widening exec write {widening.Opcode}");
                }
            }

            var execBits = UInt128.One << (int)ExecLoRegister | UInt128.One << (int)ExecHiRegister;
            var liveWrites = writes & ~execBits & liveIn[join];
            if (liveWrites != 0 || (writesScc && sccLiveIn[join]))
            {
                if (HalfBranchReport)
                {
                    var names = new List<string>();
                    for (var register = 0; register < 128; register++)
                    {
                        if ((liveWrites & (UInt128.One << register)) != 0)
                        {
                            names.Add($"s{register}");
                        }
                    }

                    if (writesScc && sccLiveIn[join])
                    {
                        names.Add("scc");
                    }

                    return RefuseHalfBranch($"scalar writes live at join: {string.Join(',', names)}");
                }

                return false;
            }

            return true;
        }

        /// <summary>Runs one block of a half-local body over <paramref name="subset"/>, the register
        /// pairs known to hold a subset of the body's starting EXEC on entry, and returns the pairs
        /// known after it. <paramref name="widening"/> is the first EXEC write that could turn on a
        /// lane the body began without, or null.</summary>
        private UInt128 SubsetPairsAfter(ShaderBlock block, UInt128 subset, out Gen5ShaderInstruction? widening)
        {
            widening = null;
            var instructions = _request.Program.Instructions;
            var execBits = UInt128.One << (int)ExecLoRegister | UInt128.One << (int)ExecHiRegister;
            for (var index = block.StartIndex; index < block.EndIndex; index++)
            {
                var instruction = instructions[index];
                if (WritesExec(instruction) && !NarrowsOrRestoresExec(instruction, subset))
                {
                    widening ??= instruction;
                }

                var (_, _, mayWrite, _) = ScalarEffects(instruction);
                var nonExec = mayWrite & ~execBits;
                if (nonExec == 0)
                {
                    continue;
                }

                if (WritesExecSubsetPair(instruction, subset, out var pair))
                {
                    var pairBits = UInt128.One << (int)pair | UInt128.One << (int)(pair + 1);
                    if ((nonExec & ~pairBits) == 0)
                    {
                        subset = (subset & ~nonExec) | pairBits;
                        continue;
                    }
                }

                subset &= ~nonExec;
            }

            return subset;
        }

        /// <summary>Work a half may run, or skip, without its partner in a forward EXEC body.</summary>
        private static bool IsHalfLocalBodyInstruction(Gen5ShaderInstruction instruction)
        {
            var opcode = instruction.Opcode;
            switch (instruction.Encoding)
            {
                case Gen5ShaderEncoding.Sopp:
                    return opcode is "SBranch" or "SCbranchExecz" or "SCbranchExecnz" or "SCbranchScc0" or "SCbranchScc1" or
                        "SNop" or "SInstPrefetch" or "SClause" or "STtracedata" or "SSetprio" ||
                        opcode.StartsWith("SWaitcnt", StringComparison.Ordinal);
                case Gen5ShaderEncoding.Sop1:
                case Gen5ShaderEncoding.Sop2:
                case Gen5ShaderEncoding.Sopc:
                case Gen5ShaderEncoding.Sopk:
                    return !opcode.StartsWith("SMovrel", StringComparison.Ordinal) &&
                        !opcode.StartsWith("SGetreg", StringComparison.Ordinal) &&
                        !opcode.StartsWith("SSetreg", StringComparison.Ordinal) &&
                        !opcode.StartsWith("SCbranch", StringComparison.Ordinal) &&
                        opcode is not ("SSetpcB64" or "SSwappcB64" or "SGetpcB64" or "SRfeB64");
                case Gen5ShaderEncoding.Smrd:
                case Gen5ShaderEncoding.Smem:
                    return opcode.StartsWith("SLoad", StringComparison.Ordinal) ||
                        opcode.StartsWith("SBufferLoad", StringComparison.Ordinal);
                case Gen5ShaderEncoding.Vop1:
                case Gen5ShaderEncoding.Vop2:
                case Gen5ShaderEncoding.Vop3:
                case Gen5ShaderEncoding.Vop3p:
                case Gen5ShaderEncoding.Vopc:
                    // DPP8 reads lanes of its own group of eight and the RDNA DPP16 controls lanes of
                    // its own row of sixteen, so they stay in the half and are emitted as shuffles
                    // inside that half's simdgroup. Other DPP16 controls (GFX9's wave shifts and
                    // row broadcasts) are not taken.
                    return (instruction.Control is not Gen5DppControl dpp || IsRowLocalDppControl(dpp.Control)) &&
                        !opcode.StartsWith("VReadlane", StringComparison.Ordinal) &&
                        !opcode.StartsWith("VReadfirstlane", StringComparison.Ordinal) &&
                        !opcode.StartsWith("VWritelane", StringComparison.Ordinal) &&
                        !opcode.StartsWith("VPermlane", StringComparison.Ordinal) &&
                        !opcode.StartsWith("VMbcnt", StringComparison.Ordinal) &&
                        !opcode.StartsWith("VMovrel", StringComparison.Ordinal) &&
                        !opcode.StartsWith("VSwaprel", StringComparison.Ordinal);
                case Gen5ShaderEncoding.Mubuf:
                case Gen5ShaderEncoding.Mtbuf:
                    return instruction.Control is Gen5BufferMemoryControl { Lds: false };
                case Gen5ShaderEncoding.Flat:
                    return opcode.StartsWith("Global", StringComparison.Ordinal) ||
                        opcode.StartsWith("Scratch", StringComparison.Ordinal);
                case Gen5ShaderEncoding.Mimg:
                    return opcode.StartsWith("Image", StringComparison.Ordinal);
                default:
                    return false;
            }
        }

        /// <summary>quad_perm, row_shl, row_shr, row_ror, row_mirror, row_half_mirror, row_share and
        /// row_xmask: the DPP16 controls whose source lane is in the reading lane's row.</summary>
        private static bool IsRowLocalDppControl(uint control) =>
            control is <= 0xFF or (>= 0x101 and <= 0x10F) or (>= 0x111 and <= 0x11F) or (>= 0x121 and <= 0x12F) or
                0x140 or 0x141 or (>= 0x150 and <= 0x16F);

        private static bool WritesExec(Gen5ShaderInstruction instruction)
        {
            var opcode = instruction.Opcode;
            if (opcode.StartsWith("VCmpx", StringComparison.Ordinal) || opcode.Contains("Saveexec", StringComparison.Ordinal))
            {
                return true;
            }

            if (instruction.Control is Gen5Vop3Control { ScalarDestination: ExecLoRegister or ExecHiRegister } or
                Gen5SdwaControl { ScalarDestination: ExecLoRegister or ExecHiRegister })
            {
                return true;
            }

            foreach (var destination in instruction.Destinations)
            {
                if (destination is { Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister or ExecHiRegister })
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsExecOrSubset(Gen5Operand operand, UInt128 subset) =>
            operand.Kind == Gen5OperandKind.ScalarRegister &&
            (operand.Value == ExecLoRegister ||
             (operand.Value < 127 && (subset & (UInt128.One << (int)operand.Value)) != 0 &&
              (subset & (UInt128.One << (int)(operand.Value + 1))) != 0));

        /// <summary>A write of EXEC that leaves it a subset of what it was when the body began.</summary>
        private static bool NarrowsOrRestoresExec(Gen5ShaderInstruction instruction, UInt128 subset)
        {
            var opcode = instruction.Opcode;
            var sources = instruction.Sources;
            if (opcode.StartsWith("VCmpx", StringComparison.Ordinal) || opcode.StartsWith("VCmp", StringComparison.Ordinal))
            {
                // A compare writes 0 for every lane EXEC has off.
                return true;
            }

            switch (opcode)
            {
                case "SAndSaveexecB64" or "SAndn1SaveexecB64":
                    return true;
                case "SOrSaveexecB64" or "SAndn2SaveexecB64" or "SXorSaveexecB64":
                    return sources.Count == 1 && IsExecOrSubset(sources[0], subset);
            }

            if (instruction.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister }])
            {
                return false;
            }

            return opcode switch
            {
                "SMovB64" => sources.Count == 1 && IsExecOrSubset(sources[0], subset),
                "SAndB64" => sources.Count == 2 && (IsExecOrSubset(sources[0], subset) || IsExecOrSubset(sources[1], subset)),
                "SAndn2B64" => sources.Count == 2 && IsExecOrSubset(sources[0], subset),
                "SOrB64" or "SXorB64" => sources.Count == 2 && IsExecOrSubset(sources[0], subset) && IsExecOrSubset(sources[1], subset),
                _ => false,
            };
        }

        /// <summary>Whether this writes an SGPR pair (not EXEC) only with lanes of the EXEC the body
        /// began with: the EXEC a SAVEEXEC saves, a compare's mask, or a mask op of EXEC and such pairs.</summary>
        private static bool WritesExecSubsetPair(Gen5ShaderInstruction instruction, UInt128 subset, out uint pair)
        {
            pair = 0;
            var opcode = instruction.Opcode;
            var sources = instruction.Sources;
            if (opcode.StartsWith("VCmp", StringComparison.Ordinal) && !opcode.StartsWith("VCmpx", StringComparison.Ordinal))
            {
                var scalarDestination = instruction.Control switch
                {
                    Gen5Vop3Control control => control.ScalarDestination,
                    Gen5SdwaControl control => control.ScalarDestination,
                    _ => null,
                };
                pair = scalarDestination ?? VccLoRegister;
                return pair < ExecLoRegister;
            }

            if (instruction.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLoRegister } destination])
            {
                return false;
            }

            pair = destination.Value;
            if (opcode.EndsWith("SaveexecB64", StringComparison.Ordinal))
            {
                return true;
            }

            return opcode switch
            {
                "SMovB64" => sources.Count == 1 && IsExecOrSubset(sources[0], subset),
                "SAndB64" => sources.Count == 2 && (IsExecOrSubset(sources[0], subset) || IsExecOrSubset(sources[1], subset)),
                "SAndn2B64" => sources.Count == 2 && IsExecOrSubset(sources[0], subset),
                "SOrB64" or "SXorB64" => sources.Count == 2 && IsExecOrSubset(sources[0], subset) && IsExecOrSubset(sources[1], subset),
                _ => false,
            };
        }

        /// <summary>Scalar registers (and SCC) live into each block. Reads are over-approximated
        /// (a memory instruction's scalar operand reads the eight registers from it, any other the
        /// pair), writes kill only what an instruction certainly writes, so a register is never
        /// missed as live.</summary>
        private (UInt128[] LiveIn, bool[] SccLiveIn) ScalarLiveness(IReadOnlyList<ShaderBlock> blocks, int[][] successors)
        {
            var instructions = _request.Program.Instructions;
            var count = blocks.Count;
            var liveIn = new UInt128[count];
            var sccIn = new bool[count];
            for (var changed = true; changed;)
            {
                changed = false;
                for (var block = count - 1; block >= 0; block--)
                {
                    UInt128 live = 0;
                    var scc = false;
                    foreach (var successor in successors[block])
                    {
                        live |= liveIn[successor];
                        scc |= sccIn[successor];
                    }

                    for (var index = blocks[block].EndIndex - 1; index >= blocks[block].StartIndex; index--)
                    {
                        var (reads, readsScc, _, _) = ScalarEffects(instructions[index]);
                        var (kills, killsScc) = DefiniteScalarWrites(instructions[index]);
                        live = (live & ~kills) | reads;
                        scc = (scc && !killsScc) || readsScc;
                    }

                    if (live != liveIn[block] || scc != sccIn[block])
                    {
                        liveIn[block] = live;
                        sccIn[block] = scc;
                        changed = true;
                    }
                }
            }

            return (liveIn, sccIn);
        }

        private static UInt128 Bits(uint first, uint count)
        {
            UInt128 bits = 0;
            for (var register = first; register < first + count && register < 128; register++)
            {
                bits |= UInt128.One << (int)register;
            }

            return bits;
        }

        /// <summary>Whether a scalar opcode certainly writes a register pair (S_BCNT1_I32_B64 and
        /// the like write one register).</summary>
        private static bool IsWideScalarOpcode(string opcode) =>
            opcode.StartsWith('S') && !opcode.Contains("32", StringComparison.Ordinal) &&
            (opcode.EndsWith("B64", StringComparison.Ordinal) || opcode.EndsWith("U64", StringComparison.Ordinal) ||
             opcode.EndsWith("I64", StringComparison.Ordinal));

        /// <summary>How many registers from a scalar destination an opcode may write, at most.</summary>
        private static uint MayScalarDestinationWidth(string opcode)
        {
            foreach (var (suffix, width) in (ReadOnlySpan<(string, uint)>)[("x16", 16u), ("x8", 8u), ("x4", 4u), ("x3", 3u), ("x2", 2u)])
            {
                if (opcode.Contains(suffix, StringComparison.OrdinalIgnoreCase))
                {
                    return width;
                }
            }

            return opcode.Contains("64", StringComparison.Ordinal) ? 2u : 1u;
        }

        private static uint? VectorScalarDestination(Gen5ShaderInstruction instruction) =>
            instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal) ? null : instruction.Control switch
            {
                Gen5Vop3Control control => control.ScalarDestination,
                Gen5SdwaControl control => control.ScalarDestination,
                _ => null,
            };

        private static bool WritesVccImplicitly(Gen5ShaderInstruction instruction)
        {
            var opcode = instruction.Opcode;
            if (!opcode.StartsWith('V') || VectorScalarDestination(instruction) is not null)
            {
                return false;
            }

            return (opcode.StartsWith("VCmp", StringComparison.Ordinal) && !opcode.StartsWith("VCmpx", StringComparison.Ordinal)) ||
                opcode.Contains("Co", StringComparison.Ordinal) ||
                opcode is "VAddcU32" or "VSubbU32" or "VSubbrevU32" ||
                opcode.StartsWith("VDivScale", StringComparison.Ordinal);
        }

        /// <summary>What an instruction may read and write among the scalar registers and SCC,
        /// over-approximated both ways.</summary>
        private static (UInt128 Reads, bool ReadsScc, UInt128 Writes, bool WritesScc) ScalarEffects(Gen5ShaderInstruction instruction)
        {
            var opcode = instruction.Opcode;
            var memory = instruction.Encoding is Gen5ShaderEncoding.Smrd or Gen5ShaderEncoding.Smem or Gen5ShaderEncoding.Mubuf or
                Gen5ShaderEncoding.Mtbuf or Gen5ShaderEncoding.Mimg or Gen5ShaderEncoding.Flat or Gen5ShaderEncoding.Ds;
            UInt128 reads = 0;
            var readsScc = false;
            for (var sourceIndex = 0; sourceIndex < instruction.Sources.Count; sourceIndex++)
            {
                var source = instruction.Sources[sourceIndex];
                if (source.Kind == Gen5OperandKind.ScalarRegister)
                {
                    reads |= Bits(source.Value, ScalarSourceWidth(instruction, sourceIndex, memory));
                }
                else if (source.Kind == Gen5OperandKind.EncodedConstant)
                {
                    readsScc |= source.Value == 253;
                    if (source.Value == 251)
                    {
                        reads |= Bits(VccLoRegister, 2);
                    }
                }
            }

            if ((opcode is "VCndmaskB32" or "VAddcU32" or "VSubbU32" or "VSubbrevU32" or "VAddCoCiU32" or "VSubCoCiU32" or
                    "VSubrevCoCiU32" && instruction.Sources.Count <= 2) ||
                opcode.StartsWith("VDivFmas", StringComparison.Ordinal) || opcode is "SCbranchVccz" or "SCbranchVccnz")
            {
                reads |= Bits(VccLoRegister, 2);
            }

            if (instruction.Encoding is Gen5ShaderEncoding.Ds or Gen5ShaderEncoding.Vintrp ||
                instruction.Control is Gen5BufferMemoryControl { Lds: true } ||
                opcode.StartsWith("SMovrel", StringComparison.Ordinal) || opcode.StartsWith("VMovrel", StringComparison.Ordinal) ||
                opcode.StartsWith("VSwaprel", StringComparison.Ordinal) || opcode.StartsWith("SSendmsg", StringComparison.Ordinal))
            {
                reads |= Bits(124, 1);
            }

            readsScc |= opcode.StartsWith("SCselect", StringComparison.Ordinal) || opcode.StartsWith("SAddc", StringComparison.Ordinal) ||
                opcode.StartsWith("SSubb", StringComparison.Ordinal) || opcode.StartsWith("SCmov", StringComparison.Ordinal) ||
                opcode.StartsWith("SCbranchScc", StringComparison.Ordinal);

            var (writes, _) = DefiniteScalarWrites(instruction);
            var width = MayScalarDestinationWidth(opcode);
            if (IsEnumeratedScalarDestination(instruction, width, out var firstDestination))
            {
                // A load that lists every register it writes (S_LOAD_DWORDX4 s16..s19) writes
                // those and no more: the width already counts them.
                writes |= Bits(firstDestination, width);
            }
            else
            {
                foreach (var destination in instruction.Destinations)
                {
                    if (destination.Kind == Gen5OperandKind.ScalarRegister)
                    {
                        writes |= Bits(destination.Value, width);
                    }
                }
            }

            var writesScc = instruction.Encoding is Gen5ShaderEncoding.Sop1 or Gen5ShaderEncoding.Sop2 or
                Gen5ShaderEncoding.Sopc or Gen5ShaderEncoding.Sopk;
            return (reads, readsScc, writes, writesScc);
        }

        /// <summary>How many registers from a scalar source operand an instruction may read, at most.
        /// A 32-bit ALU operand reads one register, a 64-bit one (an opcode naming 64 anywhere, a
        /// lane mask, a carry, the SAD family's packed operands) two. A memory operand reads its
        /// descriptor or address in full: an S_LOAD base pair, an S_BUFFER_LOAD or MUBUF/MTBUF
        /// buffer resource (4), an image resource (8) and sampler (4), and one register for a
        /// scalar offset or M0. Anything else reads eight. Under-reading would let a body's write
        /// go unseen, so every shape not named here keeps the wider count.</summary>
        private static uint ScalarSourceWidth(Gen5ShaderInstruction instruction, int sourceIndex, bool memory)
        {
            var opcode = instruction.Opcode;
            if (memory)
            {
                switch (instruction.Encoding)
                {
                    case Gen5ShaderEncoding.Smrd or Gen5ShaderEncoding.Smem
                        when opcode.StartsWith("SLoad", StringComparison.Ordinal):
                        return sourceIndex == 0 ? 2u : 1u;
                    case Gen5ShaderEncoding.Smrd or Gen5ShaderEncoding.Smem
                        when opcode.StartsWith("SBufferLoad", StringComparison.Ordinal):
                        return sourceIndex == 0 ? 4u : 1u;
                    case Gen5ShaderEncoding.Mubuf or Gen5ShaderEncoding.Mtbuf when sourceIndex >= 1:
                        // Vector address, buffer resource, scalar offset, then M0 for an LDS load.
                        return sourceIndex == 1 ? 4u : 1u;
                    case Gen5ShaderEncoding.Mimg when instruction.Control is Gen5ImageControl &&
                        sourceIndex >= instruction.Sources.Count - 2 &&
                        instruction.Sources[^2].Kind == Gen5OperandKind.ScalarRegister &&
                        instruction.Sources[^1].Kind == Gen5OperandKind.ScalarRegister:
                        // The vector addresses, then the image resource and the sampler.
                        return sourceIndex == instruction.Sources.Count - 2 ? 8u : 4u;
                    default:
                        return 8u;
                }
            }

            if (opcode.Contains("64", StringComparison.Ordinal) ||
                opcode.StartsWith("VCndmask", StringComparison.Ordinal) ||
                opcode.Contains("Co", StringComparison.Ordinal) ||
                opcode.Contains("Addc", StringComparison.Ordinal) ||
                opcode.Contains("Subb", StringComparison.Ordinal) ||
                opcode.Contains("DivFmas", StringComparison.Ordinal) ||
                opcode.Contains("sad", StringComparison.OrdinalIgnoreCase))
            {
                return 2u;
            }

            return instruction.Encoding is Gen5ShaderEncoding.Sop1 or Gen5ShaderEncoding.Sop2 or Gen5ShaderEncoding.Sopc or
                Gen5ShaderEncoding.Sopk or Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or Gen5ShaderEncoding.Vop3 or
                Gen5ShaderEncoding.Vop3p or Gen5ShaderEncoding.Vopc
                ? 1u
                : 2u;
        }

        /// <summary>Whether the destinations are exactly <paramref name="width"/> consecutive scalar
        /// registers, the whole write the opcode makes, listed one by one.</summary>
        private static bool IsEnumeratedScalarDestination(Gen5ShaderInstruction instruction, uint width, out uint first)
        {
            first = 0;
            var destinations = instruction.Destinations;
            if (width < 2 || destinations.Count != width)
            {
                return false;
            }

            for (var index = 0; index < destinations.Count; index++)
            {
                if (destinations[index].Kind != Gen5OperandKind.ScalarRegister ||
                    destinations[index].Value != destinations[0].Value + (uint)index)
                {
                    return false;
                }
            }

            first = destinations[0].Value;
            return true;
        }

        /// <summary>The scalar registers an instruction certainly writes (its named destinations,
        /// a VOP3 scalar destination, the VCC a compare or carry writes unnamed), and whether it
        /// certainly writes SCC.</summary>
        private static (UInt128 Writes, bool WritesScc) DefiniteScalarWrites(Gen5ShaderInstruction instruction)
        {
            var opcode = instruction.Opcode;
            var wide = IsWideScalarOpcode(opcode);
            UInt128 writes = 0;
            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind == Gen5OperandKind.ScalarRegister)
                {
                    writes |= Bits(destination.Value, wide ? 2u : 1u);
                }
            }

            if (VectorScalarDestination(instruction) is { } scalarDestination)
            {
                writes |= Bits(scalarDestination, 2);
            }

            if (WritesVccImplicitly(instruction))
            {
                writes |= Bits(VccLoRegister, 2);
            }

            if (opcode.StartsWith("VCmpx", StringComparison.Ordinal) || opcode.Contains("Saveexec", StringComparison.Ordinal))
            {
                writes |= Bits(ExecLoRegister, 2);
            }

            var writesScc = opcode.StartsWith("SCmp", StringComparison.Ordinal) || opcode.StartsWith("SBitcmp", StringComparison.Ordinal) ||
                opcode.StartsWith("SAnd", StringComparison.Ordinal) || opcode.StartsWith("SOr", StringComparison.Ordinal) ||
                opcode.StartsWith("SXor", StringComparison.Ordinal) || opcode.StartsWith("SNand", StringComparison.Ordinal) ||
                opcode.StartsWith("SNor", StringComparison.Ordinal) || opcode.StartsWith("SXnor", StringComparison.Ordinal) ||
                opcode.StartsWith("SNot", StringComparison.Ordinal) || opcode.StartsWith("SWqm", StringComparison.Ordinal) ||
                opcode.StartsWith("SBcnt", StringComparison.Ordinal) || opcode.StartsWith("SAbs", StringComparison.Ordinal) ||
                opcode.StartsWith("SAddU", StringComparison.Ordinal) || opcode.StartsWith("SAddI", StringComparison.Ordinal) ||
                opcode.StartsWith("SSubU", StringComparison.Ordinal) || opcode.StartsWith("SSubI", StringComparison.Ordinal) ||
                opcode.StartsWith("SAddc", StringComparison.Ordinal) || opcode.StartsWith("SSubb", StringComparison.Ordinal) ||
                opcode.StartsWith("SMin", StringComparison.Ordinal) || opcode.StartsWith("SMax", StringComparison.Ordinal) ||
                opcode.StartsWith("SLshl", StringComparison.Ordinal) || opcode.StartsWith("SLshr", StringComparison.Ordinal) ||
                opcode.StartsWith("SAshr", StringComparison.Ordinal) || opcode.StartsWith("SBfeU", StringComparison.Ordinal) ||
                opcode.StartsWith("SBfeI", StringComparison.Ordinal) || opcode.StartsWith("SAddk", StringComparison.Ordinal) ||
                opcode.StartsWith("SQuadmask", StringComparison.Ordinal);
            return (writes, writesScc);
        }

        /// <summary>After the mask ballots are resolved: a body a half may skip must hold no exchange.</summary>
        private bool HalfLocalBranchesHoldNoExchange()
        {
            if (_halfLocalBranchBodies is not { } bodies)
            {
                return true;
            }

            for (var block = 0; block < bodies.Length; block++)
            {
                if (bodies[block] &&
                    (_blockText[block].Contains(WaveSetFlip, StringComparison.Ordinal) ||
                     _blockText[block].Contains(Wave64Barrier, StringComparison.Ordinal) ||
                     _blockText[block].Contains("threadgroup_barrier", StringComparison.Ordinal) ||
                     _blockText[block].Contains("sharpemu_wave_scratch", StringComparison.Ordinal) ||
                     _blockText[block].Contains("simd_broadcast", StringComparison.Ordinal)))
                {
                    if (HalfBranchReport)
                    {
                        Console.Error.WriteLine($"[MSL][HALF-BRANCH] 0x{_request.Hash:X16} body block {block} still exchanges");
                    }

                    HalfLocalBranchExchanged = true;
                    return false;
                }
            }

            return true;
        }
    }
}
