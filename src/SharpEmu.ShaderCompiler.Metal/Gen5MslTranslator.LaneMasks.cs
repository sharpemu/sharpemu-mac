// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal;

// Lane-form wave64 masks.
//
// A 64-bit mask in a scalar register pair (a V_CMP result, EXEC saved by S_AND_SAVEEXEC) is the
// ballot of one bool per lane, and building it takes an exchange between the two simdgroups of a
// guest wave64: a threadgroup barrier. The pair is only data to a scalar consumer. The consumers
// that dominate Demon's Souls' compute shaders — the next logic op on masks, V_CNDMASK, the EXEC
// update — need only each lane's own bit, which the producer already had in hand.
//
// So a mask op in a straight-line run keeps the per-lane bools:
//   * a producer (V_CMP to an SGPR pair, a 64-bit mask op) also stores the bool it ballotted in a
//     per-pair variable, ml<N>;
//   * a consumer that reads that pair as a mask (V_CNDMASK's operand, the logic ops below) takes
//     the variable instead of the dwords wherever every path to it ends in a producer with no
//     other write to the pair in between, and the dwords otherwise (a lane-bit sentinel that
//     Gen5MslTranslator.LaneVariables resolves once the control flow is known);
//   * the ballot itself is a marker, and Gen5MslTranslator.MaskBallots expands it only if some
//     scalar reader (a branch on VCCZ/EXECZ, a move, a bit count, a 64-bit operand of anything
//     that is not a mask op, SCC when it is read) can still see the pair.
// The exchange therefore happens once, at the last producer, and only where the pair is read.
//
// A 64-bit mask op is emitted as a MASKOP marker with two spellings: the lane form (per-lane bool
// math plus ballot markers) and the pair form (the original ulong math over the pairs). Pair form
// is free when the pairs it reads are materialized for another reader anyway, so the resolver
// picks it then and the lane form otherwise (see PrefersPairForm); either way no barrier is added
// where the old code had none. SCC (= the result is non-zero) is a marker too: it needs the whole
// wave, so it reads the result pair, but only if SCC is read.
//
// Uniform control flow is untouched: every exchange stays at statement level where the producer
// or the consumer already was, and scalar branches only read pairs that are materialized, which
// are the same in every thread.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private sealed class MaskOp
        {
            public required string[] LaneLines { get; init; }

            public required string[] PairLines { get; init; }

            /// <summary>The general pairs (low registers) the op's operands read as lane bits;
            /// each is lazy, in the lane form, where its lane variable is valid.</summary>
            public required uint[] SourceRegisters { get; init; }

            /// <summary>VCC and EXEC halves the op reads through their per-lane bools.</summary>
            public required uint[] FixedLazy { get; init; }

            public required HashSet<uint> Dests { get; init; }

            /// <summary>The pair whose lane variable the op sets, or -1.</summary>
            public required int LaneVariable { get; init; }

            // Fragment scalar masks can carry bits belonging to absent/retired lanes,
            // and wave32's high dword can be scratch data. Only the original pair
            // expression preserves those bits when the result or SCC is observed.
            public bool PreserveScalarBits { get; init; }
            public bool SetsScc { get; init; }

            /// <summary>FixedLazy plus the source pairs found valid; set when the bodies resolve.</summary>
            public List<uint> Lazy { get; } = [];
        }

        private readonly List<MaskOp> _maskOps = [];

        /// <summary>The pairs that have a lane variable, declared once at the top.</summary>
        private readonly SortedSet<uint> _laneVariables = [];

        private bool _pairFormOnly;

        private bool LaneMasksEnabled =>
            _options.DeadMaskBallotElision &&
            ((IsWave64 && !IsNarrowWave64 && !ClampsExecToLanes) || FragmentLaneMasksEnabled);

        private bool FragmentLaneMasksEnabled =>
            _options.DeadMaskBallotElision && _options.FragmentLaneMasks && _stage == Gen5MslStage.Pixel && !IsSingleLaneStage;

        private static string LaneVariableName(uint register) => $"ml{register}";

        /// <summary>This lane's bit of the mask pair at <paramref name="register"/>: a sentinel
        /// the body resolver turns into the lane variable where it is valid and into
        /// <paramref name="fromDwords"/> where it is not.</summary>
        private string LaneBitOfPair(uint register, string fromDwords) =>
            LaneMasksEnabled ? $"{LaneBitMarker}({register}, {fromDwords})" : fromDwords;

        private static bool IsWriteReference(ReadOnlySpan<char> text, RegisterReference reference)
        {
            if (reference.Start > 0 && text[reference.Start - 1] == '&' &&
                (reference.Start < 2 || text[reference.Start - 2] != '&'))
            {
                return true;
            }

            var at = reference.Start + reference.Length;
            while (at < text.Length && text[at] == ' ')
            {
                at++;
            }

            if (at >= text.Length)
            {
                return false;
            }

            var character = text[at];
            var next = at + 1 < text.Length ? text[at + 1] : '\0';
            if (character == '=')
            {
                return next != '=';
            }

            if (next == '=' && "+-*/%&|^".Contains(character))
            {
                return true;
            }

            if ((character == '<' || character == '>') && next == character &&
                at + 2 < text.Length && text[at + 2] == '=')
            {
                return true;
            }

            return (character == '+' || character == '-') && next == character;
        }

        /// <summary>Runs <paramref name="emit"/> and returns the lines it wrote (at indentation
        /// zero) instead of leaving them in the body.</summary>
        private string[] CaptureLines(Func<bool> emit, out bool ok)
        {
            var start = _body.Length;
            var indent = _indent;
            _indent = 0;
            try
            {
                ok = emit();
            }
            finally
            {
                _indent = indent;
            }

            var captured = _body.ToString(start, _body.Length - start);
            _body.Length = start;
            return [.. SplitLines(captured)];
        }

        /// <summary>This lane's bit of a mask operand as an expression, and the pairs the
        /// expression takes from a per-lane value rather than from the dwords.</summary>
        private bool TryLaneSource(
            Gen5Operand operand,
            SortedSet<uint> fixedLazy,
            SortedSet<uint> sources,
            out string expression)
        {
            expression = string.Empty;
            switch (operand.Kind)
            {
                case Gen5OperandKind.ScalarRegister:
                {
                    var register = operand.Value;
                    if (register == VccLoRegister)
                    {
                        fixedLazy.Add(VccLoRegister);
                        fixedLazy.Add(VccHiRegister);
                        expression = "(vcc)";
                        return true;
                    }

                    if (register == ExecLoRegister)
                    {
                        fixedLazy.Add(ExecLoRegister);
                        fixedLazy.Add(ExecHiRegister);
                        expression = "(exec)";
                        return true;
                    }

                    if (register + 1 >= ScalarRegisterFileCount)
                    {
                        return false;
                    }

                    sources.Add(register);
                    expression = $"({LaneBitOfPair(register, LaneBit($"s[{register}]", $"s[{register + 1}]"))})";
                    return true;
                }

                case Gen5OperandKind.EncodedConstant when operand.Value == 128:
                    expression = "(false)";
                    return true;

                case Gen5OperandKind.EncodedConstant when operand.Value == 193:
                    expression = "(true)";
                    return true;

                default:
                    return false;
            }
        }

        private static string CombineLaneBits(string kind, string left, string right) => kind switch
        {
            "Mov" => left,
            "Not" => $"!{left}",
            "And" => $"({left} && {right})",
            "Or" => $"({left} || {right})",
            "Xor" => $"({left} != {right})",
            "Nand" => $"!({left} && {right})",
            "Nor" => $"!({left} || {right})",
            "Xnor" => $"({left} == {right})",
            "Andn1" => $"(!{left} && {right})",
            "Andn2" => $"({left} && !{right})",
            "Orn1" => $"(!{left} || {right})",
            "Orn2" => $"({left} || !{right})",
            _ => string.Empty,
        };

        /// <summary>Emits a 64-bit mask op (S_AND/OR/XOR/NAND/NOR/XNOR/ANDN1/ANDN2/ORN1/ORN2, S_MOV,
        /// S_NOT and the S_*_SAVEEXEC forms) as a mask-op marker. Returns false to leave the op to
        /// the pair-form emitter when an operand or the destination is not a plain mask.</summary>
        private bool TryEmitLaneMaskOp(Gen5ShaderInstruction instruction, uint destination)
        {
            if (_pairFormOnly || !LaneMasksEnabled)
            {
                return false;
            }

            var opcode = instruction.Opcode;
            var lowOnly = FragmentLaneMasksEnabled && opcode.EndsWith("B32", StringComparison.Ordinal);
            var suffix = lowOnly ? "B32" : "B64";
            var ballotMarker = lowOnly ? MaskBallot32Marker : MaskBallotMarker;
            var sccMarker = lowOnly ? SccAny32Marker : SccAnyMarker;
            var saveexec = opcode.EndsWith("Saveexec" + suffix, StringComparison.Ordinal);
            string kind;
            if (saveexec)
            {
                kind = opcode[1..opcode.IndexOf("Saveexec", StringComparison.Ordinal)];
            }
            else if (opcode == "SMov" + suffix)
            {
                kind = "Mov";
            }
            else if (opcode == "SNot" + suffix)
            {
                kind = "Not";
            }
            else if (opcode.Length > 4 && opcode[0] == 'S' && opcode.EndsWith(suffix, StringComparison.Ordinal))
            {
                kind = opcode[1..^3];
            }
            else
            {
                return false;
            }

            var twoSources = !saveexec && kind is not ("Mov" or "Not");
            if (CombineLaneBits(kind, "a", "b").Length == 0 ||
                instruction.Sources.Count < (twoSources ? 2 : 1))
            {
                return false;
            }

            if (destination + 1 >= ScalarRegisterFileCount ||
                destination == NullRegister || destination + 1 == NullRegister)
            {
                return false;
            }

            bool Touches(uint register) => destination <= register && register <= destination + 1;
            var exactVcc = destination == VccLoRegister;
            var exactExec = destination == ExecLoRegister;
            var touchesSpecial = Touches(VccLoRegister) || Touches(VccHiRegister) ||
                Touches(ExecLoRegister) || Touches(ExecHiRegister);
            if (saveexec
                ? Touches(ExecLoRegister) || Touches(ExecHiRegister) || (touchesSpecial && !exactVcc)
                : touchesSpecial && !exactVcc && !exactExec)
            {
                return false;
            }

            if (FragmentLaneMasksEnabled && !saveexec && !exactExec && !exactVcc &&
                !instruction.Sources.Any(source => source.Kind == Gen5OperandKind.ScalarRegister &&
                    (source.Value is ExecLoRegister or VccLoRegister || _laneVariables.Contains(source.Value))))
            {
                return false;
            }

            var fixedLazy = new SortedSet<uint>();
            var sources = new SortedSet<uint>();
            if (!TryLaneSource(instruction.Sources[0], fixedLazy, sources, out var left))
            {
                return false;
            }

            var right = string.Empty;
            if (twoSources && !TryLaneSource(instruction.Sources[1], fixedLazy, sources, out right))
            {
                return false;
            }

            var pairLines = CaptureLines(() =>
            {
                _pairFormOnly = true;
                try
                {
                    return lowOnly
                        ? TryEmitScalarAlu(instruction, out _)
                        : TryEmitScalar64(instruction, destination, out _);
                }
                finally
                {
                    _pairFormOnly = false;
                }
            }, out var pairOk);
            if (!pairOk)
            {
                return false;
            }

            // The bools are plain statements: later consumers refer to them whichever form
            // the op takes.
            var laneLines = new List<string>();
            var dests = new HashSet<uint> { destination };
            if (!lowOnly) dests.Add(destination + 1);
            string remembered;
            if (saveexec)
            {
                var old = Temp("bool", "exec");
                var next = Temp("bool", CombineLaneBits(kind, left, $"({old})"));
                if (exactVcc)
                {
                    laneLines.Add($"vcc = {old};");
                }

                laneLines.Add($"{ballotMarker}({destination}, {old});");
                laneLines.Add($"exec = {next};");
                laneLines.Add($"{ballotMarker}({ExecLoRegister}, {next});");
                laneLines.Add($"{sccMarker}({ExecLoRegister})");
                dests.Add(ExecLoRegister);
                if (!lowOnly) dests.Add(ExecHiRegister);
                remembered = old;
            }
            else
            {
                var result = Temp("bool", CombineLaneBits(kind, left, right));
                if (exactVcc)
                {
                    laneLines.Add($"vcc = {result};");
                }
                else if (exactExec)
                {
                    laneLines.Add($"exec = {result};");
                }

                laneLines.Add($"{ballotMarker}({destination}, {result});");
                if (kind != "Mov")
                {
                    laneLines.Add($"{sccMarker}({destination})");
                }

                remembered = result;
            }

            var variable = -1;
            if (destination != VccLoRegister && destination != ExecLoRegister)
            {
                variable = (int)destination;
                _laneVariables.Add(destination);
                Line($"{LaneVariableName(destination)} = {remembered};");
            }

            _maskOps.Add(new MaskOp
            {
                LaneLines = [.. laneLines],
                PairLines = pairLines,
                SourceRegisters = [.. sources],
                FixedLazy = [.. fixedLazy],
                Dests = dests,
                LaneVariable = variable,
                PreserveScalarBits = FragmentLaneMasksEnabled,
                SetsScc = kind != "Mov" || saveexec,
            });
            Line($"{MaskOpMarker}({_maskOps.Count - 1})");
            return true;
        }
    }
}
