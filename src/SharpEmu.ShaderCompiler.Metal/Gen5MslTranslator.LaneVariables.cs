// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.ShaderCompiler.Metal;

// Which lane variables are valid where (see Gen5MslTranslator.LaneMasks.cs).
//
// A mask producer stores its per-lane bool in ml<N> next to the ballot of the pair (s[N], s[N+1]).
// A consumer asks for "this lane's bit of pair N" and leaves a sentinel; once every block's text
// and the control flow exist, a forward must-analysis finds the places where ml<N> still equals
// that bit on every path: every path reaching the consumer ran a producer of the pair last, with
// no other write to either register since. There the sentinel becomes ml<N>; elsewhere it becomes
// the bit extracted from the dwords, which is what it always was.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        /// <summary>Marks a lane bit of a mask pair: the pair's low register and the expression
        /// that reads it from the dwords.</summary>
        private const string LaneBitMarker = "SHARPEMU_LB";

        private const int LaneVariableSlots = (int)ScalarRegisterFileCount;

        private static void InvalidateLaneVariable(bool[] valid, int register)
        {
            if (register >= 0 && register < LaneVariableSlots)
            {
                valid[register] = false;
            }

            if (register > 0 && register - 1 < LaneVariableSlots)
            {
                valid[register - 1] = false;
            }
        }

        /// <summary>Resolves every lane-bit sentinel in the block texts and records, for each mask
        /// op, which of its operands came from a lane variable.</summary>
        private void ResolveLaneBits(int count)
        {
            foreach (var op in _maskOps)
            {
                op.Lazy.Clear();
                op.Lazy.AddRange(op.FixedLazy);
            }

            var validIn = new bool[count][];
            for (var index = 0; index < count; index++)
            {
                validIn[index] = new bool[LaneVariableSlots];
                if (index != 0 && !_maskScalarDynamic)
                {
                    Array.Fill(validIn[index], true);
                }
            }

            // Optimistic start, narrowed over every edge until nothing changes: the entry
            // block has no valid variable, and a block is valid only where all its
            // predecessors leave it so.
            var changed = !_maskScalarDynamic;
            while (changed)
            {
                changed = false;
                for (var index = 0; index < count; index++)
                {
                    var valid = SimulateLaneVariables(index, validIn[index], rewrite: false, out _);
                    var terminator = _blockTerminators[index];
                    changed |= Narrow(terminator.TrueTarget, valid);
                    if (terminator.Kind == TerminatorKind.CondJump)
                    {
                        changed |= Narrow(terminator.FalseTarget, valid);
                    }
                }
            }

            for (var index = 0; index < count; index++)
            {
                SimulateLaneVariables(index, validIn[index], rewrite: true, out var text);
                _blockText[index] = text;
            }

            bool Narrow(int target, bool[] valid)
            {
                if (target == ExitTarget || target < 0 || target >= count || target == 0)
                {
                    return false;
                }

                var narrowed = false;
                for (var slot = 0; slot < LaneVariableSlots; slot++)
                {
                    if (validIn[target][slot] && !valid[slot])
                    {
                        validIn[target][slot] = false;
                        narrowed = true;
                    }
                }

                return narrowed;
            }
        }

        /// <summary>Walks a block with the variables valid at its start. With
        /// <paramref name="rewrite"/> it also returns the block text with its sentinels
        /// resolved.</summary>
        private bool[] SimulateLaneVariables(int block, bool[] validAtEntry, bool rewrite, out string resolved)
        {
            var valid = (bool[])validAtEntry.Clone();
            var source = _blockText[block];
            var hasSentinel = rewrite && source.Contains(LaneBitMarker, StringComparison.Ordinal);
            var text = rewrite && (hasSentinel || !IsNormalizedText(source)) ? new StringBuilder(source.Length) : null;
            var start = 0;
            var depth = 0;
            while (start < source.Length)
            {
                var line = NextLine(source, ref start);

                // Depth is that of the line as written, before a sentinel is resolved.
                var lineDepth = depth;
                foreach (var character in line)
                {
                    depth += character == '{' ? 1 : character == '}' ? -1 : 0;
                }

                depth = Math.Max(depth, 0);
                if (line.Contains(LaneBitMarker, StringComparison.Ordinal))
                {
                    line = ResolveLaneBitSentinels(line.ToString(), valid, rewrite);
                }

                if (text is not null)
                {
                    text.Append(line);
                    text.AppendLine();
                }

                ApplyLaneVariableEffects(line, lineDepth, valid, rewrite);
            }

            resolved = text?.ToString() ?? (rewrite ? source : string.Empty);
            return valid;
        }

        private string ResolveLaneBitSentinels(string line, bool[] valid, bool rewrite)
        {
            var result = new StringBuilder();
            var cursor = 0;
            while (true)
            {
                var at = line.IndexOf(LaneBitMarker + "(", cursor, StringComparison.Ordinal);
                if (at < 0)
                {
                    break;
                }

                var number = at + LaneBitMarker.Length + 1;
                var comma = line.IndexOf(',', number);
                var register = int.Parse(line.AsSpan(number, comma - number));
                var start = comma + 1;
                while (line[start] == ' ')
                {
                    start++;
                }

                var depth = 1;
                var end = start;
                while (depth > 0)
                {
                    depth += line[end] == '(' ? 1 : line[end] == ')' ? -1 : 0;
                    end++;
                }

                // `end` is one past the sentinel's closing parenthesis.
                var expression = line[start..(end - 1)];
                result.Append(line, cursor, at - cursor);
                var useVariable = register < LaneVariableSlots && valid[register];
                if (useVariable)
                {
                    _laneVariables.Add((uint)register);
                }

                result.Append(useVariable ? LaneVariableName((uint)register) : expression);
                cursor = end;
            }

            result.Append(line, cursor, line.Length - cursor);
            return result.ToString();
        }

        /// <summary>What a line does to which variables are valid. A producer's marker makes its
        /// pair's variable valid (and drops the neighbouring pairs it overlaps), any other write
        /// to a register drops the pairs containing it, and a run-time register index drops all
        /// of them.</summary>
        private void ApplyLaneVariableEffects(ReadOnlySpan<char> line, int depth, bool[] valid, bool recordLazy)
        {
            var trimmed = line.Trim();
            if (TryReadBallotMarker(trimmed, out var lo, out var condition, out var lowOnly))
            {
                InvalidateLaneVariable(valid, (int)lo);
                if (!lowOnly) InvalidateLaneVariable(valid, (int)lo + 1);
                if (depth == 0 && lo < LaneVariableSlots && condition == LaneVariableName(lo))
                {
                    valid[lo] = true;
                }

                return;
            }

            if (TryReadMarkerArgument(trimmed, MaskOpMarker, out var id))
            {
                var op = _maskOps[(int)id];
                if (recordLazy)
                {
                    foreach (var register in op.SourceRegisters)
                    {
                        if (register < LaneVariableSlots && valid[register])
                        {
                            op.Lazy.Add(register);
                            op.Lazy.Add(register + 1);
                        }
                    }
                }

                foreach (var register in op.Dests)
                {
                    InvalidateLaneVariable(valid, (int)register);
                }

                if (op.LaneVariable >= 0 && depth == 0)
                {
                    valid[op.LaneVariable] = true;
                }

                return;
            }

            if (!trimmed.Contains("s[", StringComparison.Ordinal))
            {
                return;
            }

            foreach (var reference in ScanRegisterReferences(trimmed))
            {
                if (reference.File != 's')
                {
                    continue;
                }

                if (reference.Index < 0)
                {
                    Array.Clear(valid);
                    return;
                }

                if (IsWriteReference(trimmed, reference))
                {
                    InvalidateLaneVariable(valid, reference.Index);
                }
            }
        }
    }
}
