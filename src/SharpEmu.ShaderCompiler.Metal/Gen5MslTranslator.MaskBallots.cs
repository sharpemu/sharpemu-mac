// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.ShaderCompiler.Metal;

// Dead mask-ballot elision.
//
// A guest wave64 runs as two 32-wide Apple simdgroups, so the 64-bit mask a V_CMP or
// V_CMPX produces spans both of them. simd_ballot only sees its own simdgroup, so the
// two halves are exchanged through threadgroup memory: write your half, barrier, read
// both (Gen5MslTranslator.WaveExchange.cs). This elision dates from when every exchange
// also needed a second barrier so the next ballot could not overwrite a slot someone was
// still reading.
//
// That was two threadgroup barriers per comparison, and a threadgroup barrier makes every
// simdgroup wait. Profiling Octopath Traveler II found 27 of them in the dispatch that
// spends 19% of the frame's GPU time, 17 in the next one and 8 in the one after.
//
// Most of them are for nothing. The per-lane result — the vcc/exec bool the EXEC guards
// and V_CNDMASK read — is just this lane's own condition and needs no exchange at all;
// the emitter has already stored it before the ballot runs. The 64-bit pair is only
// needed when the program reads it as data: an S_CBRANCH_VCCZ, a 64-bit operand, VCC
// moved into another register. None of those three dispatches branches on VCCZ or reads
// VCC as a mask; they use s106/s107 as ordinary scratch registers, so each V_CMP's
// ballot is overwritten before anything looks at it.
//
// The same holds for every scalar register pair, not just VCC and EXEC, and for the 64-bit
// mask ops that consume a pair only as per-lane bits (Gen5MslTranslator.LaneMasks.cs and
// .LaneVariables.cs): they keep the bools, SCC is a marker that reads the result only if it is
// read, and the 64-bit pair is exchanged only where a scalar reader remains.
//
// So the ballot is emitted as a marker and resolved once every block's text exists, by
// asking whether the pair is live afterwards. Taking the answer from the emitted MSL
// rather than from the IR is what makes it safe: every way the translator can read one
// of these registers spells it s[106], s[107], s[126] or s[127], so a reader cannot hide
// from a scan of the text, and anything unrecognised reads as a use and keeps the ballot.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        /// <summary>Marks a wave64 mask ballot: the low register of the pair and the
        /// condition. Resolved by <see cref="ResolveMaskBallots"/>.</summary>
        private const string MaskBallotMarker = "SHARPEMU_BALLOT";
        private const string MaskBallot32Marker = "SHARPEMU_BALLOT32";

        /// <summary>Marks a 64-bit mask op that has a lane form and a pair form (see
        /// Gen5MslTranslator.LaneMasks.cs); the argument indexes <c>_maskOps</c>.</summary>
        private const string MaskOpMarker = "SHARPEMU_MASKOP";

        /// <summary>Marks SCC = "the mask pair written by the op is non-zero"; the argument is
        /// the pair's low register. It reads the pair only if SCC is read afterwards.</summary>
        private const string SccAnyMarker = "SHARPEMU_SCCANY";
        private const string SccAny32Marker = "SHARPEMU_SCCANY32";

        /// <summary>The slot of SCC in the liveness arrays, after the scalar registers.</summary>
        private const int SccSlot = (int)ScalarRegisterFileCount;

        private const int MaskSlotCount = SccSlot + 1;

        /// <summary>VCC and EXEC, low and high: the registers the elision tracked before it
        /// followed every scalar register. When a shader addresses the scalar file
        /// dynamically these are the only pairs still elided.</summary>
        private static bool IsClassicMaskSlot(int slot) =>
            slot is (int)VccLoRegister or (int)VccHiRegister or (int)ExecLoRegister or (int)ExecHiRegister;

        private enum MaskLineKind { Plain, Ballot, SccAny, MaskOp }

        /// <summary>One line of a block, parsed once for the liveness passes.</summary>
        private sealed class MaskLine
        {
            /// <summary>The line as written. Null for a plain line of a block, which stays
            /// as written and is read back from the block text by <see cref="Start"/> and
            /// <see cref="Length"/>.</summary>
            public string? Text { get; init; }

            public int Start { get; init; }

            public int Length { get; init; }

            public MaskLineKind Kind { get; init; }

            public int Depth { get; init; }

            /// <summary>Slots a plain line reads, excluding its own assignment target.</summary>
            public int[] Reads { get; init; } = [];

            /// <summary>The slot an always-run assignment overwrites, or -1.</summary>
            public int Def { get; init; } = -1;

            public bool ReadsScc { get; init; }

            public bool DefinesScc { get; init; }

            public uint Lo { get; init; }
            public bool LowOnly { get; init; }

            public string Condition { get; init; } = string.Empty;

            public MaskOp? Op { get; init; }

            public MaskLine[] LaneLines { get; init; } = [];

            public MaskLine[] PairLines { get; init; } = [];
        }

        /// <summary>Whether rebuilding the text line by line, as the passes over a block do,
        /// would give the same text back: newline-terminated lines with no carriage returns.</summary>
        private static bool IsNormalizedText(string text) =>
            text.Length == 0 || (text[^1] == '\n' && text.IndexOf('\r') < 0 && Environment.NewLine == "\n");

        private readonly List<int> _readsScratch = [];

        private MaskLine ParseMaskLine(string line, int depth) => ParseMaskLine(line, line, 0, depth);

        /// <summary>Parses one line. <paramref name="text"/> is the line as a string when the
        /// caller has one; a block's plain lines are left as a range of the block text.</summary>
        private MaskLine ParseMaskLine(ReadOnlySpan<char> line, string? text, int start, int depth)
        {
            var trimmed = line.Trim();
            if (TryReadBallotMarker(trimmed, out var loRegister, out var condition, out var lowOnly))
            {
                return new MaskLine
                {
                    Text = text ?? line.ToString(),
                    Kind = MaskLineKind.Ballot,
                    Depth = depth,
                    Lo = loRegister,
                    Condition = condition,
                    LowOnly = lowOnly,
                };
            }

            var sccLowOnly = TryReadMarkerArgument(trimmed, SccAny32Marker, out var sccLo);
            if (sccLowOnly || TryReadMarkerArgument(trimmed, SccAnyMarker, out sccLo))
            {
                return new MaskLine { Text = text ?? line.ToString(), Kind = MaskLineKind.SccAny,
                    Depth = depth, Lo = sccLo, LowOnly = sccLowOnly };
            }

            if (TryReadMarkerArgument(trimmed, MaskOpMarker, out var id))
            {
                var op = _maskOps[(int)id];
                var pad = line[..(line.Length - line.TrimStart().Length)].ToString();
                return new MaskLine
                {
                    Text = text ?? line.ToString(),
                    Kind = MaskLineKind.MaskOp,
                    Depth = depth,
                    Op = op,
                    LaneLines = [.. op.LaneLines.Select(text => ParseMaskLine(pad + text, depth))],
                    PairLines = [.. op.PairLines.Select(text => ParseMaskLine(pad + text, depth))],
                };
            }

            var definition = depth == 0 ? DefinitionTarget(trimmed) : -1;
            var reads = _readsScratch;
            reads.Clear();
            foreach (var reference in ScanRegisterReferences(trimmed))
            {
                if (reference.File != 's' || reference.Index < 0 ||
                    (definition >= 0 && reference.Start == 0))
                {
                    continue;
                }

                reads.Add(reference.Index);
            }

            bool definesScc;
            bool readsScc;
            if (depth == 0 && trimmed.StartsWith("scc = ", StringComparison.Ordinal) && trimmed.EndsWith(';'))
            {
                definesScc = true;
                readsScc = ContainsIdentifier(trimmed[6..], "scc");
            }
            else
            {
                definesScc = false;
                readsScc = ContainsIdentifier(trimmed, "scc");
            }

            return new MaskLine
            {
                Text = text,
                Start = start,
                Length = line.Length,
                Kind = MaskLineKind.Plain,
                Depth = depth,
                Reads = reads.Count == 0 ? [] : [.. reads],
                Def = definition >= ScalarSlotBase && definition < VectorSlotBase ? definition : -1,
                ReadsScc = readsScc,
                DefinesScc = definesScc,
            };
        }

        private static bool ContainsIdentifier(ReadOnlySpan<char> text, string name)
        {
            var from = 0;
            while (from <= text.Length - name.Length)
            {
                var at = text[from..].IndexOf(name, StringComparison.Ordinal);
                if (at < 0)
                {
                    return false;
                }

                at += from;
                var before = at == 0 || !IsIdentifierCharacter(text[at - 1]);
                var afterIndex = at + name.Length;
                var after = afterIndex >= text.Length || !IsIdentifierCharacter(text[afterIndex]);
                if (before && after)
                {
                    return true;
                }

                from = at + 1;
            }

            return false;
        }

        private static bool TryReadMarkerArgument(ReadOnlySpan<char> trimmed, string marker, out uint value)
        {
            value = 0;
            if (!trimmed.StartsWith(marker, StringComparison.Ordinal) ||
                trimmed.Length <= marker.Length + 2 ||
                trimmed[marker.Length] != '(' || !trimmed.EndsWith(')'))
            {
                return false;
            }

            return uint.TryParse(
                trimmed.Slice(marker.Length + 1, trimmed.Length - marker.Length - 2),
                out value);
        }

        private bool _maskScalarDynamic;

        private bool IsTrackedSlot(int slot) =>
            !_maskScalarDynamic || IsClassicMaskSlot(slot) || slot == SccSlot;

        /// <summary>Replaces every mask-ballot marker with the cross-simdgroup exchange,
        /// or with nothing where the pair it writes is dead, and chooses for each 64-bit
        /// mask op between its lane form and its pair form.</summary>
        private void ResolveMaskBallots(IReadOnlyList<ShaderBlock> blocks)
        {
            var anyMarker = false;
            foreach (var text in _blockText)
            {
                anyMarker |= text.Contains(MaskBallotMarker, StringComparison.Ordinal) ||
                    text.Contains(MaskOpMarker, StringComparison.Ordinal) ||
                    text.Contains(LaneBitMarker, StringComparison.Ordinal);
            }

            if (!anyMarker)
            {
                return;
            }

            var count = blocks.Count;

            // The elision follows every scalar register by its literal index. A run-time
            // index (S_MOVREL and friends) can name any of them, so then only VCC and EXEC
            // are tracked, as before, and every mask op keeps its pair form.
            _maskScalarDynamic = false;
            for (var index = 0; index < count && !_maskScalarDynamic; index++)
            {
                var terminator = _blockTerminators[index];
                var texts = new List<string> { _blockText[index], terminator.Condition };
                texts.AddRange(terminator.DispatcherLines);
                foreach (var text in texts)
                {
                    foreach (var reference in ScanRegisterReferences(text))
                    {
                        if (reference.File == 's' && reference.Index < 0)
                        {
                            _maskScalarDynamic = true;
                            break;
                        }
                    }
                }
            }

            if (_maskScalarDynamic) Statistics.NoteMaskDynamic();
            ResolveLaneBits(count);

            var lines = new List<MaskLine>[count];
            var bodyCounts = new int[count];
            for (var index = 0; index < count; index++)
            {
                var blockText = _blockText[index];
                lines[index] = new List<MaskLine>(blockText.AsSpan().Count('\n') + 1);
                var lineStart = 0;
                var lineDepth = 0;
                while (lineStart < blockText.Length)
                {
                    var thisStart = lineStart;
                    var line = NextLine(blockText, ref lineStart);
                    lines[index].Add(ParseMaskLine(line, null, thisStart, lineDepth));
                    foreach (var character in line)
                    {
                        lineDepth += character == '{' ? 1 : character == '}' ? -1 : 0;
                    }

                    lineDepth = Math.Max(lineDepth, 0);
                }

                bodyCounts[index] = lines[index].Count;

                // The branch condition and dispatcher read at the end of the block, and the
                // VCCZ and EXECZ forms read the pair.
                var terminator = _blockTerminators[index];
                if (terminator.Condition.Length != 0)
                {
                    lines[index].Add(ParseMaskLine(terminator.Condition, 0));
                }

                foreach (var dispatcher in terminator.DispatcherLines)
                {
                    lines[index].Add(ParseMaskLine(dispatcher, 0));
                }
            }

            var liveIn = new bool[count][];
            for (var index = 0; index < count; index++)
            {
                liveIn[index] = new bool[MaskSlotCount];
            }

            var changed = true;
            while (changed)
            {
                changed = false;
                for (var index = count - 1; index >= 0; index--)
                {
                    var live = LiveOutOf(index, count, liveIn);
                    for (var at = lines[index].Count - 1; at >= 0; at--)
                    {
                        StepBackward(lines[index][at], live, null, decide: false);
                    }

                    for (var slot = 0; slot < MaskSlotCount; slot++)
                    {
                        if (live[slot] && !liveIn[index][slot])
                        {
                            liveIn[index][slot] = true;
                            changed = true;
                        }
                    }
                }
            }

            for (var index = 0; index < count; index++)
            {
                var live = LiveOutOf(index, count, liveIn);
                var resolved = new List<string?>?[lines[index].Count];
                for (var at = lines[index].Count - 1; at >= 0; at--)
                {
                    // A plain line stays as written: only the liveness step runs for it.
                    var plain = lines[index][at].Kind == MaskLineKind.Plain;
                    resolved[at] = plain ? null : [];
                    StepBackward(lines[index][at], live, resolved[at], decide: true);
                }

                var blockText = _blockText[index];
                var untouched = true;
                for (var at = 0; at < bodyCounts[index] && untouched; at++)
                {
                    untouched = resolved[at] is null;
                }

                if (untouched && IsNormalizedText(blockText))
                {
                    // Nothing in the block was a marker, so it is already the text it would be rebuilt to.
                    continue;
                }

                var text = new StringBuilder(blockText.Length);
                for (var at = 0; at < bodyCounts[index]; at++)
                {
                    if (resolved[at] is not { } parts)
                    {
                        var plain = lines[index][at];
                        if (plain.Text is { } plainText)
                        {
                            text.AppendLine(plainText);
                        }
                        else
                        {
                            text.Append(blockText, plain.Start, plain.Length);
                            text.AppendLine();
                        }

                        continue;
                    }

                    foreach (var line in parts)
                    {
                        if (line is not null)
                        {
                            text.AppendLine(line);
                        }
                    }
                }

                _blockText[index] = text.ToString();
            }
        }

        /// <summary>Whether the translated body ended up referring to the wave64 bridge,
        /// which is only known once the mask ballots are resolved.</summary>
        private bool _waveScratchReferenced;

        private void NoteWaveScratchUse()
        {
            foreach (var text in _blockText)
            {
                _waveScratchReferenced |=
                    text.Contains(WaveScratchName, StringComparison.Ordinal);
            }
        }

        private const string WaveScratchName = "sharpemu_wave_scratch";

        /// <summary>Whether the bridge has to be declared. The pre-pass over the program
        /// says whether the entry seed wants it; the emitted body says whether anything
        /// else does. Both have to be asked: a plain V_CMP in a wave64 program emits a
        /// ballot exchange without being a cross-lane op, so a program with one and no
        /// V_CMPX used to reference an array nobody had declared and fail to compile.</summary>
        private bool DeclaresWaveScratch => _usesWaveScratch || _waveScratchReferenced;

        private bool[] LiveOutOf(int block, int count, bool[][] liveIn)
        {
            var live = new bool[MaskSlotCount];
            var terminator = _blockTerminators[block];
            Union(live, Successor(terminator.TrueTarget, count, liveIn));
            if (terminator.Kind == TerminatorKind.CondJump)
            {
                Union(live, Successor(terminator.FalseTarget, count, liveIn));
            }

            return live;
        }

        /// <summary>One backward liveness step over a line. With <paramref name="resolved"/>
        /// it also records the text the line becomes: a ballot marker is its exchange or
        /// nothing, an SCC marker a read of the pair or nothing, a mask op its lane or pair
        /// form. A write that always runs ends the previous value's life; one under a
        /// condition leaves the other path reading what was there, so it keeps the ballot
        /// that produced it alive. Ballot, SCC and mask-op markers are statement-level, so
        /// they always run.</summary>
        private void StepBackward(MaskLine line, bool[] live, List<string?>? resolved, bool decide)
        {
            switch (line.Kind)
            {
                case MaskLineKind.Ballot:
                {
                    var lo = (int)line.Lo;
                    var hi = lo + 1;
                    var hasHigh = !line.LowOnly && hi < (int)ScalarRegisterFileCount;
                    var needed = !IsTrackedSlot(lo) || (hasHigh && !IsTrackedSlot(hi)) ||
                        live[lo] || (hasHigh && live[hi]);
                    resolved?.Add(needed ? ExpandMaskBallot(line.Text!, line.Lo, line.Condition, line.LowOnly) : null);
                    if (decide) Statistics.NoteBallot(needed, _stage == Gen5MslStage.Pixel);
                    live[lo] = false;
                    if (hasHigh)
                    {
                        live[hi] = false;
                    }

                    return;
                }

                case MaskLineKind.SccAny:
                {
                    if (!live[SccSlot])
                    {
                        resolved?.Add(null);
                        return;
                    }

                    var lo = (int)line.Lo;
                    var indent = line.Text!.Length - line.Text.AsSpan().TrimStart().Length;
                    var mask = line.LowOnly ? $"s[{lo}]" : $"(s[{lo}] | s[{lo + 1}])";
                    resolved?.Add($"{line.Text[..indent]}scc = ({mask} != 0u);");
                    live[SccSlot] = false;
                    live[lo] = true;
                    if (!line.LowOnly) live[lo + 1] = true;
                    return;
                }

                case MaskLineKind.MaskOp:
                {
                    var op = line.Op!;
                    var scalarObserved = op.PreserveScalarBits &&
                        (op.Dests.Any(register => live[register]) || (op.SetsScc && live[SccSlot]));
                    // This choice also participates in the liveness fixed point: pair
                    // reads can reach predecessors and must retain their producers.
                    var pair = scalarObserved || (decide && PrefersPairForm(op, live));
                    if (decide && scalarObserved) Statistics.NoteMaskPairObserved();
                    var sub = pair ? line.PairLines : line.LaneLines;
                    var parts = new List<string?>[sub.Length];
                    for (var at = sub.Length - 1; at >= 0; at--)
                    {
                        parts[at] = [];
                        StepBackward(sub[at], live, resolved is null ? null : parts[at], decide);
                    }

                    if (resolved is not null)
                    {
                        foreach (var part in parts)
                        {
                            resolved.AddRange(part);
                        }
                    }

                    return;
                }

                default:
                {
                    if (line.Def >= 0)
                    {
                        live[line.Def] = false;
                    }

                    if (line.DefinesScc)
                    {
                        live[SccSlot] = false;
                    }

                    foreach (var read in line.Reads)
                    {
                        live[read] = true;
                    }

                    if (line.ReadsScc)
                    {
                        live[SccSlot] = true;
                    }

                    resolved?.Add(line.Text!);
                    return;
                }
            }
        }

        /// <summary>The pair form of a mask op is free when every pair the lane form would
        /// have taken from a per-lane value is already materialized for another reader: it
        /// computes its result from those pairs with no exchange at all. Otherwise the lane
        /// form drops the producers' exchanges and pays one for its result, and only if
        /// that result is read as a pair.</summary>
        private bool PrefersPairForm(MaskOp op, bool[] live)
        {
            if (_maskScalarDynamic)
            {
                return true;
            }

            foreach (var register in op.Lazy)
            {
                if (op.Dests.Contains(register) || !live[register])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>The cross-simdgroup exchange, at the marker's own indentation.</summary>
        private string ExpandMaskBallot(string markerLine, uint loRegister, string condition, bool lowOnly)
        {
            var indent = markerLine.Length - markerLine.AsSpan().TrimStart().Length;
            var pad = markerLine[..indent];
            var text = new StringBuilder();

            if (!IsWave64)
            {
                text.Append($"{pad}s[{loRegister}] = sharpemu_ballot({condition});");
                if (!lowOnly && loRegister + 1 < ScalarRegisterFileCount)
                {
                    text.AppendLine();
                    text.Append($"{pad}s[{loRegister + 1}] = 0u;");
                }
                return text.ToString();
            }

            foreach (var statement in WaveBallotExchange(condition))
            {
                text.AppendLine($"{pad}{statement}");
            }

            text.Append($"{pad}s[{loRegister}] = {WaveSlot(0)};");
            if (loRegister + 1 < ScalarRegisterFileCount)
            {
                text.AppendLine();
                text.Append($"{pad}s[{loRegister + 1}] = {WaveSlot(1)};");
            }

            return text.ToString();
        }

        private static bool TryReadBallotMarker(ReadOnlySpan<char> trimmed, out uint loRegister, out string condition, out bool lowOnly)
        {
            lowOnly = trimmed.StartsWith(MaskBallot32Marker, StringComparison.Ordinal);
            loRegister = 0;
            condition = string.Empty;
            if (!trimmed.StartsWith(MaskBallotMarker, StringComparison.Ordinal))
            {
                return false;
            }

            var open = lowOnly ? MaskBallot32Marker.Length : MaskBallotMarker.Length;
            var close = trimmed.LastIndexOf(')');
            if (open >= trimmed.Length || trimmed[open] != '(' || close <= open)
            {
                return false;
            }

            var arguments = trimmed[(open + 1)..close];
            var comma = arguments.IndexOf(',');
            if (comma < 0 || arguments[(comma + 1)..].Contains(','))
            {
                return false;
            }

            if (!uint.TryParse(arguments[..comma].Trim(), out loRegister))
            {
                return false;
            }

            condition = arguments[(comma + 1)..].Trim().ToString();
            return true;
        }
    }
}
