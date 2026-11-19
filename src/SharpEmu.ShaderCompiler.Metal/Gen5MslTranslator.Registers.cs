// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.ShaderCompiler.Metal;

// Guest register allocation.
//
// The emitters write every SGPR and VGPR access as an index into one of two
// arrays, s[] and v[], which used to be declared at their architectural sizes:
// "uint s[128] = {}; uint v[256] = {};" — 384 dwords of per-thread state, all
// zero-initialized, in every single shader. A shader that touches twelve
// registers still paid for all of them: 1.5 KB of per-thread storage the Apple
// GPU compiler has to promote or spill, and the zero-fill at entry.
//
// This pass reads back the MSL the emitters just produced and finds which
// registers it actually names. That catches every implicit one for free — the
// VCC and EXEC halves, M0, the extra dwords a 64-bit or multi-dword op writes,
// the architectural registers the launch state seeds — because the emitters have
// already written them out by index. Each one named becomes its own local
// ("uint s14, v3;"), which the compiler keeps in a register with no aliasing
// question to answer.
//
// Dynamic indexing (V_MOVREL* and friends address the register file through M0)
// is the exception: that file stays an array, since any element may be the one
// addressed. The other file is still scalarized.
//
// Initialization follows liveness: a register is zero-filled only when some path
// from entry reads it before writing it. Registers the program defines before
// use — the overwhelming majority — are declared without an initializer, and the
// launch state's seeded inputs count as definitions, so they are never clobbered.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const int ScalarSlotBase = 0;
        private const int VectorSlotBase = (int)ScalarRegisterFileCount;
        private const int RegisterSlotCount =
            (int)ScalarRegisterFileCount + (int)VectorRegisterFileCount;

        /// <summary>Which guest registers this shader declares, how, and which of them
        /// need an initial value.</summary>
        private sealed class RegisterPlan
        {
            public bool ScalarArray { get; init; }

            public bool VectorArray { get; init; }

            public SortedSet<uint> Scalars { get; } = [];

            public SortedSet<uint> Vectors { get; } = [];

            public HashSet<uint> ScalarNeedsInitialValue { get; } = [];

            public HashSet<uint> VectorNeedsInitialValue { get; } = [];

            /// <summary>Rewrites s[14] into s14 for whichever file was scalarized, and
            /// resolves the dynamic-access markers over the registers the program
            /// touches. The index form stays for a file kept as an array.</summary>
            public void RewriteInto(StringBuilder target, string text)
            {
                text = ExpandDynamicVectorAccess(text);
                if (ScalarArray && VectorArray)
                {
                    target.Append(text);
                    return;
                }

                RewriteSegment(target, text);
            }

            /// <summary>The same for text that is still in a builder, without gathering it into one
            /// string first: a body is the size of the shader, which is far into the large object heap.
            /// It is taken a line-aligned segment at a time, which a register reference never crosses.</summary>
            public void RewriteInto(StringBuilder target, StringBuilder text)
            {
                if (ContainsDynamicMarker(text))
                {
                    RewriteInto(target, text.ToString());
                    return;
                }

                if (ScalarArray && VectorArray)
                {
                    target.Append(text);
                    return;
                }

                var buffer = System.Buffers.ArrayPool<char>.Shared.Rent(1024);
                try
                {
                    var carried = 0;
                    foreach (var chunk in text.GetChunks())
                    {
                        var span = chunk.Span;
                        var lastNewline = span.LastIndexOf('\n');
                        if (lastNewline < 0)
                        {
                            Carry(ref buffer, carried, span);
                            carried += span.Length;
                            continue;
                        }

                        var segment = span[..(lastNewline + 1)];
                        if (carried == 0)
                        {
                            RewriteSegment(target, segment);
                        }
                        else
                        {
                            Carry(ref buffer, carried, segment);
                            RewriteSegment(target, buffer.AsSpan(0, carried + segment.Length));
                        }

                        var rest = span[(lastNewline + 1)..];
                        carried = 0;
                        Carry(ref buffer, carried, rest);
                        carried = rest.Length;
                    }

                    if (carried != 0)
                    {
                        RewriteSegment(target, buffer.AsSpan(0, carried));
                    }
                }
                finally
                {
                    System.Buffers.ArrayPool<char>.Shared.Return(buffer);
                }
            }

            private static void Carry(ref char[] buffer, int length, ReadOnlySpan<char> value)
            {
                if (buffer.Length < length + value.Length)
                {
                    var larger = System.Buffers.ArrayPool<char>.Shared.Rent(Math.Max(buffer.Length * 2, length + value.Length));
                    buffer.AsSpan(0, length).CopyTo(larger);
                    System.Buffers.ArrayPool<char>.Shared.Return(buffer);
                    buffer = larger;
                }

                value.CopyTo(buffer.AsSpan(length));
            }

            private static bool ContainsDynamicMarker(StringBuilder text)
            {
                // A marker can straddle two chunks, so the end of one is searched with the start of the next.
                Span<char> seam = stackalloc char[2 * DynamicVectorWrite.Length];
                var tail = 0;
                foreach (var chunk in text.GetChunks())
                {
                    var span = chunk.Span;
                    if (span.Contains(DynamicVectorRead, StringComparison.Ordinal) ||
                        span.Contains(DynamicVectorWrite, StringComparison.Ordinal))
                    {
                        return true;
                    }

                    if (tail != 0)
                    {
                        var head = Math.Min(span.Length, DynamicVectorWrite.Length);
                        span[..head].CopyTo(seam[tail..]);
                        var joined = seam[..(tail + head)];
                        if (joined.Contains(DynamicVectorRead, StringComparison.Ordinal) ||
                            joined.Contains(DynamicVectorWrite, StringComparison.Ordinal))
                        {
                            return true;
                        }
                    }

                    tail = Math.Min(span.Length, DynamicVectorWrite.Length - 1);
                    span[^tail..].CopyTo(seam);
                }

                return false;
            }

            private void RewriteSegment(StringBuilder target, ReadOnlySpan<char> text)
            {
                var cursor = 0;
                foreach (var reference in ScanRegisterReferences(text))
                {
                    if (reference.Index < 0 ||
                        (reference.File == 's' ? ScalarArray : VectorArray))
                    {
                        continue;
                    }

                    target.Append(text[cursor..reference.Start]);
                    target.Append(reference.File);
                    target.Append(reference.Index);
                    cursor = reference.Start + reference.Length;
                }

                target.Append(text[cursor..]);
            }

            /// <summary>Turns each marked run-time register access into a choice among the
            /// registers the program actually touches.
            ///
            /// A relative move can name any register in the file, but a register the
            /// program never writes holds whatever the previous wave left there — the
            /// guest reads undefined data, so any value answers it. That is what makes
            /// the expansion possible: only the registers this program uses have to be
            /// reachable, and for the Octopath vertex shader that is v0..v31 rather than
            /// all 256. A read becomes a switch over them and a write becomes one masked
            /// assignment each, which is more instructions at the relative moves and in
            /// exchange keeps the whole register file in hardware registers instead of
            /// thread-private memory.</summary>
            private string ExpandDynamicVectorAccess(string text)
            {
                if (!text.Contains(DynamicVectorRead, StringComparison.Ordinal) &&
                    !text.Contains(DynamicVectorWrite, StringComparison.Ordinal))
                {
                    return text;
                }

                var result = new StringBuilder(text.Length);
                var cursor = 0;
                while (cursor < text.Length)
                {
                    var read = text.IndexOf(DynamicVectorRead, cursor, StringComparison.Ordinal);
                    var write = text.IndexOf(DynamicVectorWrite, cursor, StringComparison.Ordinal);
                    var at = read < 0 ? write : write < 0 ? read : Math.Min(read, write);
                    if (at < 0)
                    {
                        break;
                    }

                    var isRead = at == read;
                    var marker = isRead ? DynamicVectorRead : DynamicVectorWrite;
                    if (!TryReadMarkerArguments(text, at + marker.Length, out var arguments, out var end))
                    {
                        // Not a call after all; copy it through and carry on.
                        result.Append(text, cursor, at + marker.Length - cursor);
                        cursor = at + marker.Length;
                        continue;
                    }

                    result.Append(text, cursor, at - cursor);
                    result.Append(isRead
                        ? ExpandRead(arguments[0], arguments[1])
                        : ExpandWrite(arguments[0], arguments[1]));
                    cursor = end;
                }

                result.Append(text, cursor, text.Length - cursor);
                return result.ToString();
            }

            /// <summary>Assigns the register <paramref name="index"/> names to
            /// <paramref name="target"/>. The number comes from M0, an SGPR, so every lane
            /// of a wave agrees on it and the switch is a uniform branch: a few compares
            /// whatever the program's register count, where a select chain paid a compare
            /// and a move per register at every access.</summary>
            private string ExpandRead(string target, string index)
            {
                if (VectorArray)
                {
                    return $"{target} = v[{index}];";
                }

                if (Vectors.Count == 0)
                {
                    return $"{target} = 0u;";
                }

                // The lowest register is the fallback arm: an index outside the set is a
                // register the program never writes, so its value is undefined anyway.
                var statements = new StringBuilder($"switch ({index}) {{");
                foreach (var register in Vectors)
                {
                    if (register != Vectors.Min)
                    {
                        statements.Append($" case {register}u: {target} = v{register}; break;");
                    }
                }

                return statements.Append($" default: {target} = v{Vectors.Min}; break; }}").ToString();
            }

            private string ExpandWrite(string index, string value)
            {
                if (VectorArray)
                {
                    return $"v[{index}] = {value};";
                }

                var statements = new StringBuilder("{");
                foreach (var register in Vectors)
                {
                    statements.Append($" v{register} = {index} == {register}u ? {value} : v{register};");
                }

                return statements.Append(" }").ToString();
            }

            /// <summary>Splits a marker's parenthesised argument list, respecting nesting.</summary>
            private static bool TryReadMarkerArguments(
                string text,
                int open,
                out string[] arguments,
                out int end)
            {
                arguments = [];
                end = open;
                if (open >= text.Length || text[open] != '(')
                {
                    return false;
                }

                var parts = new List<string>();
                var depth = 0;
                var start = open + 1;
                for (var index = open; index < text.Length; index++)
                {
                    var character = text[index];
                    if (character == '(')
                    {
                        depth++;
                    }
                    else if (character == ')')
                    {
                        depth--;
                        if (depth == 0)
                        {
                            parts.Add(text[start..index].Trim());
                            end = index + 1;
                            arguments = [.. parts];
                            return true;
                        }
                    }
                    else if (character == ',' && depth == 1)
                    {
                        parts.Add(text[start..index].Trim());
                        start = index + 1;
                    }
                }

                return false;
            }
        }

        /// <summary>Marks a register read whose number is only known at run time, as the
        /// statement "SHARPEMU_VREAD(target, number);". Resolved by
        /// <see cref="RegisterPlan.RewriteInto"/> once the register set is known.</summary>
        private const string DynamicVectorRead = "SHARPEMU_VREAD";

        /// <summary>Marks the matching write; takes the register number and the value.</summary>
        private const string DynamicVectorWrite = "SHARPEMU_VWRITE";

        /// <summary>One s[...] or v[...] occurrence. <see cref="Index"/> is -1 when the
        /// subscript is not a decimal literal, which is how dynamic indexing shows up.</summary>
        private readonly record struct RegisterReference(
            char File,
            int Index,
            int Start,
            int Length);

        /// <summary>Finds every s[...]/v[...] in generated MSL. The name must stand alone:
        /// "sharpemu_lds[" and "buffers[" end in s and v but are preceded by an
        /// identifier character, so they are skipped.</summary>
        private static RegisterReferenceScan ScanRegisterReferences(ReadOnlySpan<char> text) => new(text);

        private readonly ref struct RegisterReferenceScan
        {
            private readonly ReadOnlySpan<char> _text;

            public RegisterReferenceScan(ReadOnlySpan<char> text) => _text = text;

            public RegisterReferenceEnumerator GetEnumerator() => new(_text);
        }

        private ref struct RegisterReferenceEnumerator
        {
            private readonly ReadOnlySpan<char> _text;
            private int _next;

            public RegisterReferenceEnumerator(ReadOnlySpan<char> text)
            {
                _text = text;
                _next = 0;
                Current = default;
            }

            public RegisterReference Current { get; private set; }

            public bool MoveNext()
            {
                var text = _text;
                for (var index = _next; index + 1 < text.Length; index++)
                {
                    var file = text[index];
                    if ((file != 's' && file != 'v') || text[index + 1] != '[')
                    {
                        continue;
                    }

                    if (index > 0 && IsIdentifierCharacter(text[index - 1]))
                    {
                        continue;
                    }

                    var cursor = index + 2;
                    var value = 0L;
                    var digits = 0;
                    while (cursor < text.Length && text[cursor] >= '0' && text[cursor] <= '9')
                    {
                        value = (value * 10) + (text[cursor] - '0');
                        cursor++;
                        digits++;
                    }

                    if (digits != 0 && cursor < text.Length && text[cursor] == ']' &&
                        value < (file == 's' ? ScalarRegisterFileCount : VectorRegisterFileCount))
                    {
                        Current = new RegisterReference(file, (int)value, index, cursor + 1 - index);
                        _next = cursor + 1;
                        return true;
                    }

                    // Not a literal subscript: the file is addressed dynamically.
                    Current = new RegisterReference(file, -1, index, 2);
                    _next = index + 1;
                    return true;
                }

                _next = text.Length;
                return false;
            }
        }

        private static bool IsIdentifierCharacter(char value) =>
            value == '_' || char.IsLetterOrDigit(value);

        private RegisterPlan BuildRegisterPlan(
            IReadOnlyList<ShaderBlock> blocks,
            string initialState)
        {
            if (!_options.RegisterAllocation)
            {
                return new RegisterPlan { ScalarArray = true, VectorArray = true };
            }

            // Everything the finished function can name, whichever form the body took.
            var texts = new List<string>(blocks.Count + 1) { initialState };
            for (var index = 0; index < blocks.Count; index++)
            {
                var terminator = _blockTerminators[index];
                texts.Add(_blockText[index]);
                texts.Add(terminator.Condition);
                foreach (var line in terminator.DispatcherLines)
                {
                    texts.Add(line);
                }
            }

            var dynamicScalar = false;
            var dynamicVector = false;
            var markedVectorAccess = false;
            var markedVectorWrite = false;
            var scalars = new SortedSet<uint>();
            var vectors = new SortedSet<uint>();
            foreach (var text in texts)
            {
                markedVectorWrite |= text.Contains(DynamicVectorWrite, StringComparison.Ordinal);
                markedVectorAccess |=
                    markedVectorWrite ||
                    text.Contains(DynamicVectorRead, StringComparison.Ordinal);
                foreach (var reference in ScanRegisterReferences(text))
                {
                    if (reference.Index < 0)
                    {
                        dynamicScalar |= reference.File == 's';
                        dynamicVector |= reference.File == 'v';
                    }
                    else
                    {
                        (reference.File == 's' ? scalars : vectors).Add((uint)reference.Index);
                    }
                }
            }

            // A run-time read is resolved however many registers the program touches: it
            // is a uniform switch over them, a few compares, against the whole file living
            // in thread-private memory. (It was a select chain, a compare per register, and
            // past 64 registers the array stayed — which put all 124 registers of Silent
            // Hill's 4096-point FFT in memory for the 8 reads at its very end.)
            //
            // A run-time *write* keeps the array. Resolving a read over the
            // program's own registers is sound because a register it never writes holds
            // undefined data, so any answer will do; a write breaks that, since the
            // register it lands on is one the program does mean to write and may read
            // back through the same number later. Only a write whose register the offset
            // analysis pinned down is resolved, and that one never reaches here.
            if (markedVectorWrite)
            {
                dynamicVector = true;
            }

            var plan = new RegisterPlan { ScalarArray = dynamicScalar, VectorArray = dynamicVector };
            plan.Scalars.UnionWith(scalars);
            plan.Vectors.UnionWith(vectors);

            var live = LiveAtEntry(blocks, initialState);
            if (markedVectorAccess)
            {
                // A run-time register number is resolved after this pass, so liveness
                // cannot see which registers it reads. It may read any of them, so they
                // all get an initial value.
                for (var register = 0; register < VectorRegisterFileCount; register++)
                {
                    live[VectorSlotBase + register] = true;
                }
            }

            foreach (var register in plan.Scalars)
            {
                if (live[ScalarSlotBase + (int)register])
                {
                    plan.ScalarNeedsInitialValue.Add(register);
                }
            }

            foreach (var register in plan.Vectors)
            {
                if (live[VectorSlotBase + (int)register])
                {
                    plan.VectorNeedsInitialValue.Add(register);
                }
            }

            return plan;
        }

        /// <summary>Backward liveness over the block CFG, with the launch state as a
        /// pseudo-block ahead of the entry: the result is the set of registers some path
        /// reads before writing, which is exactly the set that needs an initial value.
        /// The CFG used is the dispatcher's (every block, successors from the
        /// terminators), so it over-approximates the structured form, never the reverse.</summary>
        private bool[] LiveAtEntry(IReadOnlyList<ShaderBlock> blocks, string initialState)
        {
            var count = blocks.Count;
            if (_options.ForceRegisterInitialization)
            {
                var all = new bool[RegisterSlotCount];
                Array.Fill(all, true);
                return all;
            }

            var exposed = new bool[count + 1][];
            var killed = new bool[count + 1][];
            for (var index = 0; index < count; index++)
            {
                var terminator = _blockTerminators[index];

                // The branch condition reads at the end of the block.
                Summarize(
                    _blockText[index],
                    terminator.Condition.Length != 0 ? terminator.Condition : null,
                    out exposed[index],
                    out killed[index]);
            }

            Summarize(initialState, null, out exposed[count], out killed[count]);

            var liveIn = new bool[count + 1][];
            for (var index = 0; index <= count; index++)
            {
                liveIn[index] = new bool[RegisterSlotCount];
            }

            var changed = true;
            while (changed)
            {
                changed = false;
                for (var index = count - 1; index >= 0; index--)
                {
                    var terminator = _blockTerminators[index];
                    var outLive = new bool[RegisterSlotCount];
                    Union(outLive, Successor(terminator.TrueTarget, count, liveIn));
                    if (terminator.Kind == TerminatorKind.CondJump)
                    {
                        Union(outLive, Successor(terminator.FalseTarget, count, liveIn));
                    }

                    for (var slot = 0; slot < RegisterSlotCount; slot++)
                    {
                        var value = exposed[index][slot] || (outLive[slot] && !killed[index][slot]);
                        if (value && !liveIn[index][slot])
                        {
                            liveIn[index][slot] = true;
                            changed = true;
                        }
                    }
                }
            }

            var result = new bool[RegisterSlotCount];
            for (var slot = 0; slot < RegisterSlotCount; slot++)
            {
                result[slot] = exposed[count][slot] ||
                    (count != 0 && liveIn[0][slot] && !killed[count][slot]);
            }

            return result;
        }

        private static bool[]? Successor(int target, int count, bool[][] liveIn) =>
            target == ExitTarget || target >= count ? null : liveIn[target];

        private static void Union(bool[] target, bool[]? source)
        {
            if (source is null)
            {
                return;
            }

            for (var slot = 0; slot < target.Length; slot++)
            {
                target[slot] |= source[slot];
            }
        }

        /// <summary>The next line of <paramref name="text"/> from <paramref name="start"/>, without
        /// its terminator; the cursor moves past it. The span form of <see cref="SplitLines"/>.</summary>
        private static ReadOnlySpan<char> NextLine(string text, scoped ref int start)
        {
            var end = text.IndexOf('\n', start);
            if (end < 0)
            {
                end = text.Length;
            }

            var line = text.AsSpan(start, end - start).TrimEnd('\r');
            start = end + 1;
            return line;
        }

        private static IEnumerable<string> SplitLines(string text)
        {
            var start = 0;
            while (start < text.Length)
            {
                var end = text.IndexOf('\n', start);
                if (end < 0)
                {
                    end = text.Length;
                }

                yield return text[start..end].TrimEnd('\r');
                start = end + 1;
            }
        }

        /// <summary>Per-block gen/kill: which registers are read before being written in
        /// this block, and which are definitely written by it. "Definitely" means an
        /// unconditional top-level assignment — a statement at brace depth zero of the
        /// form "s[14] = ...;". An EXEC-guarded store ("if (exec) { v[3] = ...; }") is
        /// not a definition, so a register only ever written under EXEC still gets an
        /// initial value.</summary>
        private static void Summarize(string text, string? tail, out bool[] exposed, out bool[] killed)
        {
            exposed = new bool[RegisterSlotCount];
            killed = new bool[RegisterSlotCount];
            var depth = 0;
            var start = 0;
            var tailPending = tail is not null;
            while (true)
            {
                ReadOnlySpan<char> line;
                if (start < text.Length)
                {
                    line = NextLine(text, ref start);
                }
                else if (tailPending)
                {
                    tailPending = false;
                    line = tail;
                }
                else
                {
                    break;
                }

                var trimmed = line.Trim();
                var definition = depth == 0 ? DefinitionTarget(trimmed) : -1;
                foreach (var reference in ScanRegisterReferences(trimmed))
                {
                    if (reference.Index < 0)
                    {
                        // A dynamically indexed file keeps its array, which is always
                        // zero-filled, so liveness over it does not matter.
                        continue;
                    }

                    if (definition >= 0 && reference.Start == 0)
                    {
                        continue;
                    }

                    var slot = Slot(reference.File, reference.Index);
                    if (!killed[slot])
                    {
                        exposed[slot] = true;
                    }
                }

                if (definition >= 0)
                {
                    killed[definition] = true;
                }

                foreach (var character in line)
                {
                    depth += character == '{' ? 1 : character == '}' ? -1 : 0;
                }

                depth = Math.Max(depth, 0);
            }
        }

        private static int Slot(char file, int index) =>
            (file == 's' ? ScalarSlotBase : VectorSlotBase) + index;

        /// <summary>The register slot a line unconditionally assigns, or -1.</summary>
        private static int DefinitionTarget(ReadOnlySpan<char> trimmed)
        {
            if (trimmed.Length == 0 || !trimmed.EndsWith(';') ||
                (trimmed[0] != 's' && trimmed[0] != 'v') ||
                trimmed.Length < 2 || trimmed[1] != '[')
            {
                return -1;
            }

            var close = trimmed.IndexOf(']');
            if (close < 2 || !trimmed[(close + 1)..].StartsWith(" = "))
            {
                return -1;
            }

            if (!uint.TryParse(trimmed.Slice(2, close - 2), out var register))
            {
                return -1;
            }

            var limit = trimmed[0] == 's' ? ScalarRegisterFileCount : VectorRegisterFileCount;
            return register < limit ? Slot(trimmed[0], (int)register) : -1;
        }

        private void EmitRegisterFile(StringBuilder source, RegisterPlan plan, bool structured)
        {
            if (plan.ScalarArray)
            {
                source.AppendLine($"    uint s[{ScalarRegisterFileCount}] = {{}};");
            }
            else
            {
                AppendRegisterDeclarations(source, 's', plan.Scalars, plan.ScalarNeedsInitialValue);
            }

            if (plan.VectorArray)
            {
                source.AppendLine($"    uint v[{VectorRegisterFileCount}] = {{}};");
            }
            else
            {
                AppendRegisterDeclarations(source, 'v', plan.Vectors, plan.VectorNeedsInitialValue);
            }

            source.AppendLine("    bool exec = true;");
            DeclareGuestExec(source);
            if (_usesPixelValidMask)
            {
                source.AppendLine("    bool pixel_valid_mask_active = true;");
            }

            source.AppendLine("    bool vcc = false;");
            source.AppendLine("    bool scc = false;");
            foreach (var register in _laneVariables)
            {
                source.AppendLine($"    bool {LaneVariableName(register)} = false;");
            }

            if (_usesShaderClock)
            {
                source.AppendLine($"    ulong {ShaderClockName} = 0ul;");
            }

            if (!structured)
            {
                source.AppendLine("    uint pc = 0u;");
                source.AppendLine("    bool active = true;");
                source.AppendLine("    uint steps = 0u;");
                return;
            }

            if (_stage == Gen5MslStage.Compute)
            {
                // Still carries the partial-group guard the launch state computes.
                source.AppendLine("    bool active = true;");
            }

            if (_structuredUsesSteps)
            {
                source.AppendLine("    uint steps = 0u;");
            }
        }

        /// <summary>One declaration list per file, in register order, wrapped to stay
        /// readable. Only a register some path reads before writing carries "= 0u", so a
        /// dumped shader shows at a glance which of its registers are inputs.</summary>
        private static void AppendRegisterDeclarations(
            StringBuilder source,
            char file,
            SortedSet<uint> registers,
            HashSet<uint> needsInitialValue)
        {
            var line = new StringBuilder();
            foreach (var register in registers)
            {
                var name = needsInitialValue.Contains(register)
                    ? $"{file}{register} = 0u"
                    : $"{file}{register}";
                if (line.Length == 0)
                {
                    line.Append("    uint ").Append(name);
                    continue;
                }

                if (line.Length + name.Length + 2 > 96)
                {
                    source.Append(line).AppendLine(";");
                    line.Clear();
                    line.Append("    uint ").Append(name);
                    continue;
                }

                line.Append(", ").Append(name);
            }

            if (line.Length != 0)
            {
                source.Append(line).AppendLine(";");
            }
        }
    }
}
