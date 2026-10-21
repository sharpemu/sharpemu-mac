// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.ShaderCompiler.Metal;

// The structured control-flow assembler: emits each block once, nested in real MSL
// control flow, instead of running them from a PC dispatcher.
//
// Why it matters: the dispatcher's while (active) { switch (pc) } makes every guest
// register live across the loop back edge, because any block may run after any
// other. The Apple GPU compiler then has to keep — or spill — the whole guest
// register file between every pair of blocks, which is what makes translated
// shaders slow enough to be the frame budget. Nesting the blocks in if/else and
// while gives the compiler the program's real CFG, so a value's live range is the
// span it is actually used over.
//
// The recovery is the textbook one: dominators find the back edges, so a header
// with a single entry and a single way out becomes while (true) with continue and
// break; post-dominators find where a scalar branch's two arms reconverge, so a
// conditional becomes if or if/else with that join as the region's end; S_ENDPGM
// and a branch past the program end become an early return with the stage
// epilogue inlined. Metal has no goto, so an edge none of those shapes can
// express — irreducible flow, a loop with several exits, a jump into another
// region — fails the whole shader back to the dispatcher, which is always correct.
//
// Lane semantics are untouched: the conditions are the same scalar-branch
// predicates the dispatcher evaluated (SCC, the VCC/EXEC register pair), each
// Metal thread still follows its own path, and per-lane masking stays in the
// "if (exec)" guards the instruction emitters write.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        /// <summary>A region end no block can match, for a loop with no way out but a
        /// return and for the function's own outermost region.</summary>
        private const int NoFollow = int.MinValue;

        /// <summary>True when the assembled structured body needs the step counter
        /// (it has at least one back edge and the cap is enabled).</summary>
        private bool _structuredUsesSteps;

        // A loop with several ways out keeps one Follow no block can match; the edges out are
        // in Exits, and control resumes at Continuation (the exits' nearest common
        // post-dominator) after the loop dispatches to the exit it left through.
        private sealed record LoopInfo(int Header, bool[] Body, int Follow, int[]? Exits = null, int Continuation = NoFollow)
        {
            public bool MultiExit => Exits is { Length: > 1 };
        }

        private const int MultiExitFollow = int.MinValue + 1;

        // SHARPEMU_MSL_MULTI_EXIT_LOOPS=0 sends a loop with several exits to the dispatcher again.
        private static readonly bool MultiExitLoops =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_MULTI_EXIT_LOOPS") != "0";

        /// <summary>The loops enclosing the region being emitted, innermost last.</summary>
        private readonly List<LoopInfo> _openLoops = [];

        private int[] _idom = [];
        private int[] _ipdom = [];
        private List<int>[] _successors = [];
        private Dictionary<int, LoopInfo> _loopsByHeader = [];
        private bool[] _structureEmitted = [];
        private bool _structureFailed;

        // Why the last structuring attempt gave up, for SHARPEMU_MSL_STRUCTURE_REPORT: which
        // shape sends a program to the dispatcher.
        private string? _structureFailure;

        private void FailStructure(string reason)
        {
            _structureFailed = true;
            _structureFailure ??= reason;
        }

        /// <summary>Set for the second assembly attempt, which also duplicates short loop
        /// latches and picks a loop body's joins by reachability (see
        /// <see cref="TryBuildStructuredBody"/>).</summary>
        private bool _relaxedStructure;

        /// <summary>The most block text a latch may carry and still be copied to every
        /// place that jumps to it.</summary>
        private const int MaxDuplicatedLatchChars = 2048;

        /// <summary>Assembles the structured body, first exactly as before and then, if
        /// that fails, once more with two relaxations for loop bodies that continue from
        /// several places. Persistent-thread kernels (UE5's Nanite culling) have that
        /// shape: one big loop whose short latch block is reached from several nested
        /// arms, and arms that skip ahead by continuing. The retry copies such a latch to
        /// each arrival as "latch; continue;", and when one arm of a branch can only leave
        /// the iteration, joins at the other arm instead of at the post-dominator (which a
        /// continue arm pushes past the loop). A shader the first attempt already
        /// structures keeps byte-identical output.</summary>
        // SHARPEMU_MSL_RELAXED_STRUCTURE=0 skips the retry, so a program only it structures
        // runs from the dispatcher instead.
        private static readonly bool RelaxedStructureRetry =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_RELAXED_STRUCTURE") != "0";

        private bool TryBuildStructuredBody(IReadOnlyList<ShaderBlock> blocks, out string body)
        {
            var built = TryBuildStructuredBody(blocks, out StringBuilder? text);
            body = built ? text!.ToString() : string.Empty;
            return built;
        }

        private static readonly bool StructureReport =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_STRUCTURE_REPORT") == "1";

        private bool TryBuildStructuredBody(
            IReadOnlyList<ShaderBlock> blocks,
            [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out StringBuilder? body)
        {
            var built = TryBuildStructuredBodyCore(blocks, out body);
            if (!built && StructureReport)
            {
                Console.Error.WriteLine(
                    $"[MSL][STRUCTURE] 0x{_request.Hash:X16} stage={_stage} blocks={blocks.Count} keeps the dispatcher: " +
                    $"{_structureFailure ?? "unrecorded shape"}");
            }

            return built;
        }

        private bool TryBuildStructuredBodyCore(
            IReadOnlyList<ShaderBlock> blocks,
            [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out StringBuilder? body)
        {
            _structureFailure = null;
            _relaxedStructure = false;
            if (TryAssembleStructuredBody(blocks, out body))
            {
                return true;
            }

            if (!RelaxedStructureRetry)
            {
                return false;
            }

            // The retry's reason is the one that decides; the first attempt's goes with it.
            var strictFailure = _structureFailure;
            _structureFailure = null;
            _relaxedStructure = true;
            try
            {
                var built = TryAssembleStructuredBody(blocks, out body);
                if (!built)
                {
                    _structureFailure = $"{_structureFailure ?? "unrecorded shape"} (first attempt: {strictFailure ?? "unrecorded shape"})";
                }

                return built;
            }
            finally
            {
                _relaxedStructure = false;
            }
        }

        private bool TryAssembleStructuredBody(
            IReadOnlyList<ShaderBlock> blocks,
            [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out StringBuilder? body)
        {
            body = null;
            var count = blocks.Count;
            _successors = new List<int>[count];
            for (var index = 0; index < count; index++)
            {
                var terminator = _blockTerminators[index];
                _successors[index] = terminator.Kind == TerminatorKind.CondJump
                    ? [terminator.TrueTarget, terminator.FalseTarget]
                    : [terminator.TrueTarget];
            }

            _idom = Dominators(count);
            _ipdom = PostDominators(count);
            if (!TryFindLoops(count, out _loopsByHeader))
            {
                return false;
            }

            _structureEmitted = new bool[count];
            _regionDepth = 0;
            _structureFailed = false;
            _structuredUsesSteps = false;
            _openLoops.Clear();

            var text = new StringBuilder();
            if (_stage == Gen5MslStage.Compute)
            {
                // The partial-group guard the launch state computed: lanes outside the
                // guest dispatch run nothing, which is what "active = false" meant.
                AppendIndented(text, "if (!active)", 1);
                AppendIndented(text, "{", 1);
                AppendExit(text, 2);
                AppendIndented(text, "}", 1);
                text.AppendLine();
            }

            EmitRegion(text, 0, NoFollow, 1);
            if (_structureFailed)
            {
                return false;
            }

            // The stage epilogue follows the body, so an exit that already sits at the
            // end of it would emit the epilogue twice over.
            var epilogue = new StringBuilder();
            AppendExit(epilogue, 1);
            var tail = epilogue.ToString();
            if (text.Length >= tail.Length &&
                string.Equals(text.ToString(text.Length - tail.Length, tail.Length), tail, StringComparison.Ordinal))
            {
                text.Length -= tail.Length;
            }

            body = text;
            return true;
        }

        // ---- region emission ----

        /// <summary>Emits the region that starts at <paramref name="node"/> and ends when
        /// control reaches <paramref name="follow"/>. Returns true when the emitted text
        /// can fall out of its end into <paramref name="follow"/>.</summary>
        private bool EmitRegion(
            StringBuilder text,
            int node,
            int follow,
            int indent,
            bool allowLoopHeader = false)
        {
            // Regions nest only where the guest program's conditionals nest, which is
            // shallow in practice; the guard is there so a pathological program takes the
            // dispatcher rather than the stack.
            if (++_regionDepth > MaxRegionDepth)
            {
                FailStructure($"region depth over {MaxRegionDepth}");
                _regionDepth--;
                return false;
            }

            try
            {
                return EmitRegionCore(text, node, follow, indent, allowLoopHeader);
            }
            finally
            {
                _regionDepth--;
            }
        }

        private const int MaxRegionDepth = 128;

        private int _regionDepth;

        private bool EmitRegionCore(
            StringBuilder text,
            int node,
            int follow,
            int indent,
            bool allowLoopHeader)
        {
            var first = true;
            while (true)
            {
                if (_structureFailed)
                {
                    return false;
                }

                if (node == follow)
                {
                    return true;
                }

                if (node == ExitTarget)
                {
                    AppendExit(text, indent);
                    return false;
                }

                var atHeader = first && allowLoopHeader;
                first = false;
                if (!atHeader && _structureEmitted[node] && IsDuplicableTail(node, follow))
                {
                    // Each copy is a scope of its own, so the block's temporaries do not
                    // collide with another copy's.
                    AppendIndented(
                        text,
                        Gen5MslDump.Enabled ? $"{{ // tail {node}" : "{",
                        indent);
                    AppendBlockText(text, _blockText[node], indent + 1);
                    AppendIndented(text, "}", indent);
                    node = _blockTerminators[node].TrueTarget;
                    continue;
                }

                if (!atHeader && !TryEnterBlock(text, node, indent))
                {
                    return false;
                }

                if (!atHeader && _loopsByHeader.TryGetValue(node, out var loop))
                {
                    if (!EmitLoop(text, loop, indent))
                    {
                        return false;
                    }

                    var next = loop.MultiExit ? loop.Continuation : loop.Follow;
                    if (next == ExitTarget || next == NoFollow)
                    {
                        return false;
                    }

                    node = next;
                    continue;
                }

                _structureEmitted[node] = true;
                if (indent > 1 && _meshBarrierBlocks is { } barrierBlocks && barrierBlocks[node])
                {
                    _meshBarrierNested = true;
                }

                AppendIndented(
                    text,
                    Gen5MslDump.Enabled ? $"{{ // block {node}" : "{",
                    indent);
                AppendBlockText(text, _blockText[node], indent + 1);
                AppendIndented(text, "}", indent);

                var terminator = _blockTerminators[node];
                if (terminator.Kind == TerminatorKind.Jump ||
                    terminator.TrueTarget == terminator.FalseTarget)
                {
                    // The head of the loop turns the edge into sequencing, a fall out of
                    // the region, a continue, a break, a return, or a failure.
                    node = terminator.TrueTarget;
                    continue;
                }

                // A scalar branch: its arms reconverge at the immediate post-dominator
                // when that block belongs to this region, and otherwise both arms leave
                // the region on their own.
                var join = _ipdom[node];
                if (_relaxedStructure &&
                    TryFindLoopBodyJoin(node, terminator.TrueTarget, terminator.FalseTarget, follow, out var bodyJoin))
                {
                    join = bodyJoin;
                }

                if (join < 0 || join == follow || _structureEmitted[join] ||
                    EnclosesLoopBoundary(join))
                {
                    join = follow;
                }

                // Both arms are written straight into the body: a conditional whose arms
                // came out empty is cut back off, and an empty taken arm swaps to the
                // negated form, so the text matches what separate buffers would give.
                var start = text.Length;
                AppendIndented(text, $"if ({terminator.Condition})", indent);
                AppendIndented(text, "{", indent);
                var takenStart = text.Length;
                var takenFell = EmitRegion(text, terminator.TrueTarget, join, indent + 1);
                if (_structureFailed)
                {
                    return false;
                }

                var takenEmpty = text.Length == takenStart;
                bool fallenFell;
                if (takenEmpty)
                {
                    text.Length = start;
                    AppendIndented(text, $"if (!({terminator.Condition}))", indent);
                    AppendIndented(text, "{", indent);
                    var fallenStart = text.Length;
                    fallenFell = EmitRegion(text, terminator.FalseTarget, join, indent + 1);
                    if (_structureFailed)
                    {
                        return false;
                    }

                    if (text.Length == fallenStart)
                    {
                        text.Length = start;
                    }
                    else
                    {
                        AppendIndented(text, "}", indent);
                    }
                }
                else
                {
                    AppendIndented(text, "}", indent);
                    var elseStart = text.Length;
                    AppendIndented(text, "else", indent);
                    AppendIndented(text, "{", indent);
                    var fallenStart = text.Length;
                    fallenFell = EmitRegion(text, terminator.FalseTarget, join, indent + 1);
                    if (_structureFailed)
                    {
                        return false;
                    }

                    if (text.Length == fallenStart)
                    {
                        text.Length = elseStart;
                    }
                    else
                    {
                        AppendIndented(text, "}", indent);
                    }
                }

                if (join == follow)
                {
                    return takenFell || fallenFell;
                }

                if (!takenFell && !fallenFell)
                {
                    return false;
                }

                node = join;
            }
        }

        private bool EmitLoop(StringBuilder text, LoopInfo loop, int indent)
        {
            if (loop.MultiExit)
            {
                // Which way out the loop took: each exit edge records its number and
                // breaks, and the dispatch after the loop runs that exit's region.
                AppendIndented(text, $"uint {LoopExitVariable(loop)} = 0u;", indent);
            }

            AppendIndented(text, "while (true)", indent);
            AppendIndented(text, "{", indent);
            _openLoops.Add(loop);
            var fell = EmitRegion(text, loop.Header, loop.Follow, indent + 1, allowLoopHeader: true);
            _openLoops.RemoveAt(_openLoops.Count - 1);
            if (_structureFailed)
            {
                return false;
            }

            if (fell)
            {
                // Falling out of a while (true) body would loop again; the edge that
                // leaves the loop is a break.
                AppendIndented(text, "break;", indent + 1);
            }

            AppendIndented(text, "}", indent);
            return !loop.MultiExit || EmitLoopExits(text, loop, indent);
        }

        private static string LoopExitVariable(LoopInfo loop) => $"sharpemu_loop_exit_{loop.Header}";

        // Each exit's region runs up to the continuation, where every way out rejoins. An exit
        // that is the continuation itself needs nothing; two regions that share a block before
        // it fail back to the dispatcher (the second meets an emitted block), as before.
        private bool EmitLoopExits(StringBuilder text, LoopInfo loop, int indent)
        {
            var first = true;
            for (var index = 0; index < loop.Exits!.Length; index++)
            {
                var exit = loop.Exits[index];
                if (exit == loop.Continuation)
                {
                    continue;
                }

                var region = new StringBuilder();
                EmitRegion(region, exit, loop.Continuation, indent + 1);
                if (_structureFailed)
                {
                    return false;
                }

                AppendIndented(text, $"{(first ? "if" : "else if")} ({LoopExitVariable(loop)} == {index + 1}u)", indent);
                AppendIndented(text, "{", indent);
                text.Append(region);
                AppendIndented(text, "}", indent);
                first = false;
            }

            return true;
        }

        /// <summary>Resolves arriving at <paramref name="node"/> from the statement
        /// just emitted. A back edge to the enclosing loop's header is a continue and
        /// the edge that leaves it is a break; an edge to an outer loop's header or exit,
        /// or back into a region already emitted, is one MSL cannot write without a
        /// labelled break or a goto, so the shader falls back to the dispatcher.
        /// Returns false when the arrival was a statement or a failure, true when the
        /// block itself should now be emitted inline.</summary>
        private bool TryEnterBlock(StringBuilder text, int node, int indent)
        {
            var innermost = _openLoops.Count != 0 ? _openLoops[^1] : null;
            if (innermost is not null && node == innermost.Header)
            {
                AppendBackEdge(text, indent);
                AppendIndented(text, "continue;", indent);
                return false;
            }

            if (innermost is not null && innermost.MultiExit && Array.IndexOf(innermost.Exits!, node) is var exitIndex and >= 0)
            {
                AppendIndented(text, $"{LoopExitVariable(innermost)} = {exitIndex + 1}u;", indent);
                AppendIndented(text, "break;", indent);
                return false;
            }

            if (innermost is not null && node == innermost.Follow)
            {
                AppendIndented(text, "break;", indent);
                return false;
            }

            if (innermost is not null && IsDuplicableLatch(node, innermost))
            {
                // Each copy is a scope of its own, so the block's temporaries do not
                // collide with another copy's.
                AppendIndented(
                    text,
                    Gen5MslDump.Enabled ? $"{{ // latch {node}" : "{",
                    indent);
                AppendBlockText(text, _blockText[node], indent + 1);
                AppendIndented(text, "}", indent);
                AppendBackEdge(text, indent);
                AppendIndented(text, "continue;", indent);
                return false;
            }

            foreach (var loop in _openLoops)
            {
                if (node == loop.Header || node == loop.Follow || (loop.MultiExit && Array.IndexOf(loop.Exits!, node) >= 0))
                {
                    FailStructure($"edge to an outer loop's {(node == loop.Header ? "header" : "exit")}: block {node} (loop at {loop.Header})");
                    return false;
                }
            }

            if (_structureEmitted[node])
            {
                FailStructure($"edge back into an emitted region: block {node}");
                return false;
            }

            return true;
        }

        /// <summary>The dispatcher charged every block against the step cap so a
        /// mistranslated loop-exit condition could not wedge the GPU queue. The
        /// structured form charges every loop iteration, which keeps the same valve, except in
        /// a vertex or pixel program, whose runaway loop the GPU watchdog ends
        /// (<see cref="Gen5MslCodegenOptions.GraphicsLoopStepCap"/>).</summary>
        private void AppendBackEdge(StringBuilder text, int indent)
        {
            if (MaxDispatcherSteps <= 0 ||
                (_stage is Gen5MslStage.Vertex or Gen5MslStage.Pixel && !_options.GraphicsLoopStepCap))
            {
                return;
            }

            _structuredUsesSteps = true;
            AppendIndented(text, $"if (++steps >= {MaxDispatcherSteps}u)", indent);
            AppendIndented(text, "{", indent);
            AppendIndented(text, StepCapReportCode, indent + 1);
            AppendExit(text, indent + 1);
            AppendIndented(text, "}", indent);
        }

        /// <summary>Leaving the program. With no goto the stage epilogue is inlined at
        /// every exit instead of being jumped to.</summary>
        private void AppendExit(StringBuilder text, int indent)
        {
            switch (_stage)
            {
                case Gen5MslStage.Pixel:
                    // EXP.VM publishes EXEC as the pixel-valid mask; a program without
                    // VM falls back to the final EXEC value.
                    AppendIndented(
                        text,
                        _usesPixelValidMask
                            ? "if (!pixel_valid_mask_active)"
                            : "if (!exec)",
                        indent);
                    AppendIndented(text, "{", indent);
                    AppendIndented(text, "discard_fragment();", indent + 1);
                    AppendIndented(text, "}", indent);
                    AppendIndented(text, "return sharpemu_out;", indent);
                    return;
                case Gen5MslStage.Vertex:
                    AppendIndented(text, "return sharpemu_out;", indent);
                    return;
                case Gen5MslStage.Mesh:
                    // The mesh epilogue publishes the whole workgroup's output after a
                    // threadgroup barrier, so every thread has to reach it: an exit at the
                    // end of the program falls into it, and one anywhere else would leave
                    // the barrier to a part of the group, so that shape stays dispatched.
                    if (indent > 1)
                    {
                        FailStructure("mesh exit inside a conditional");
                    }

                    return;
                default:
                    AppendIndented(text, "return;", indent);
                    return;
            }
        }

        /// <summary>Whether arriving at <paramref name="node"/> inside
        /// <paramref name="loop"/> may be written as a copy of the block followed by a
        /// continue: the retry only, a short block of the loop's body that goes straight
        /// back to its header.</summary>
        private bool IsDuplicableLatch(int node, LoopInfo loop)
        {
            if (!_relaxedStructure || node == loop.Header || node < 0 || !loop.Body[node])
            {
                return false;
            }

            var terminator = _blockTerminators[node];
            var target = terminator.Kind == TerminatorKind.Jump || terminator.TrueTarget == terminator.FalseTarget
                ? terminator.TrueTarget
                : ExitTarget;
            return target == loop.Header && _blockText[node].Length <= MaxDuplicatedLatchChars;
        }

        /// <summary>Whether a block already emitted may be emitted again where a second
        /// region reaches it: the retry only, a short block that is not a loop boundary
        /// and goes straight on to the end of the region. Two arms of a branch that each
        /// either run a shared tail or skip past it to the join (the shape EXEC-skip
        /// branches leave) both reach that tail before the join.</summary>
        private bool IsDuplicableTail(int node, int follow)
        {
            if (!_relaxedStructure || node < 0 || _loopsByHeader.ContainsKey(node) ||
                EnclosesLoopBoundary(node) || _blockText[node].Length > MaxDuplicatedLatchChars ||
                _meshBarrierBlocks is { } barrierBlocks && barrierBlocks[node])
            {
                return false;
            }

            var terminator = _blockTerminators[node];
            return (terminator.Kind == TerminatorKind.Jump || terminator.TrueTarget == terminator.FalseTarget) &&
                terminator.TrueTarget == follow;
        }

        /// <summary>The blocks of the innermost open loop's body that <paramref name="start"/>
        /// reaches without leaving the iteration: the walk stops at the header, at a
        /// duplicable latch (both continue), at the loop's exit (a break) and at the
        /// program's end.</summary>
        private HashSet<int> IterationReach(int start, LoopInfo loop)
        {
            var reached = new HashSet<int>();
            var pending = new Stack<int>();
            pending.Push(start);
            while (pending.Count != 0)
            {
                var node = pending.Pop();
                if (node < 0 || node == loop.Header || node == loop.Follow || !loop.Body[node] ||
                    IsDuplicableLatch(node, loop) || !reached.Add(node))
                {
                    continue;
                }

                foreach (var successor in _successors[node])
                {
                    pending.Push(successor);
                }
            }

            return reached;
        }

        /// <summary>The retry's join for a branch inside a loop body. When one arm can
        /// only leave the iteration (continue, break or return), the branch is an
        /// if-without-else and the other arm's target is where the code goes on. When
        /// both arms go on, they join at the first block both reach that neither can get
        /// past without passing through: a path that skips a candidate and leaves the
        /// iteration is fine, one that skips it and runs on below it is not. Either
        /// answer is only a proposal: the region emitter still rejects a block reached
        /// twice.</summary>
        private bool TryFindLoopBodyJoin(int node, int taken, int fallen, int follow, out int join)
        {
            join = -1;
            LoopInfo loop;
            if (_openLoops.Count != 0)
            {
                loop = _openLoops[^1];
            }
            else if (_ipdom[node] < 0 && ExitArmJoins)
            {
                // Outside any loop a branch whose arm can end the program (a discard, an
                // early S_ENDPGM) has no post-dominator, and the arms used to be emitted
                // separately until they met the same block twice. The same search finds
                // where the arms that go on rejoin, with the region's end as the only
                // way out besides the program's end.
                loop = new LoopInfo(NoFollow, WholeProgramBody(), follow);
            }
            else
            {
                return false;
            }

            var takenReach = IterationReach(taken, loop);
            var fallenReach = IterationReach(fallen, loop);
            if (takenReach.Count == 0 && fallenReach.Count != 0)
            {
                join = fallen;
                return true;
            }

            if (fallenReach.Count == 0 && takenReach.Count != 0)
            {
                join = taken;
                return true;
            }

            var candidates = new List<int>();
            foreach (var candidate in takenReach)
            {
                if (candidate > node && fallenReach.Contains(candidate))
                {
                    candidates.Add(candidate);
                }
            }

            candidates.Sort();
            foreach (var candidate in candidates)
            {
                if (IsLoopBodyJoin(candidate, taken, fallen, loop))
                {
                    join = candidate;
                    return true;
                }
            }

            return false;
        }

        // SHARPEMU_MSL_EXIT_ARM_JOINS=0 leaves a branch with an exiting arm outside loops to
        // the post-dominator alone again.
        private static readonly bool ExitArmJoins =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_EXIT_ARM_JOINS") != "0";

        private bool[]? _wholeProgramBody;

        private bool[] WholeProgramBody()
        {
            if (_wholeProgramBody is null || _wholeProgramBody.Length != _successors.Length)
            {
                _wholeProgramBody = new bool[_successors.Length];
                Array.Fill(_wholeProgramBody, true);
            }

            return _wholeProgramBody;
        }

        /// <summary>Whether the arms starting at <paramref name="taken"/> and
        /// <paramref name="fallen"/> reach nothing beyond <paramref name="candidate"/>
        /// except through it, within the current iteration.</summary>
        private bool IsLoopBodyJoin(int candidate, int taken, int fallen, LoopInfo loop)
        {
            var beyond = new HashSet<int>();
            foreach (var successor in _successors[candidate])
            {
                beyond.UnionWith(IterationReach(successor, loop));
            }

            var reached = new HashSet<int>();
            var pending = new Stack<int>();
            pending.Push(taken);
            pending.Push(fallen);
            while (pending.Count != 0)
            {
                var node = pending.Pop();
                if (node < 0 || node == candidate || node == loop.Header || node == loop.Follow ||
                    !loop.Body[node] || IsDuplicableLatch(node, loop) || !reached.Add(node))
                {
                    continue;
                }

                if (beyond.Contains(node))
                {
                    return false;
                }

                foreach (var successor in _successors[node])
                {
                    pending.Push(successor);
                }
            }

            return true;
        }

        private bool EnclosesLoopBoundary(int node)
        {
            foreach (var loop in _openLoops)
            {
                if (node == loop.Header || node == loop.Follow || (loop.MultiExit && Array.IndexOf(loop.Exits!, node) >= 0))
                {
                    return true;
                }
            }

            return false;
        }

        // ---- CFG analysis ----

        /// <summary>Natural loops, keyed by header. Fails on irreducible flow (a back
        /// edge whose target does not dominate its source) and on a loop with more than
        /// one way out, neither of which while/break/continue can express.</summary>
        private bool TryFindLoops(int count, out Dictionary<int, LoopInfo> loops)
        {
            loops = [];
            var predecessors = new List<int>[count];
            for (var index = 0; index < count; index++)
            {
                predecessors[index] = [];
            }

            for (var index = 0; index < count; index++)
            {
                foreach (var successor in _successors[index])
                {
                    if (successor != ExitTarget)
                    {
                        predecessors[successor].Add(index);
                    }
                }
            }

            var bodies = new Dictionary<int, bool[]>();
            for (var index = 0; index < count; index++)
            {
                foreach (var successor in _successors[index])
                {
                    if (successor == ExitTarget || _idom[index] == -1)
                    {
                        continue;
                    }

                    if (!Dominates(successor, index))
                    {
                        // Only an edge whose target dominates its source closes a
                        // natural loop. Anything else retreating is irreducible flow,
                        // which the emitter rejects when it meets the block twice.
                        continue;
                    }

                    if (!bodies.TryGetValue(successor, out var body))
                    {
                        body = new bool[count];
                        body[successor] = true;
                        bodies[successor] = body;
                    }

                    CollectLoopBody(body, successor, index, predecessors);
                }
            }

            foreach (var (header, body) in bodies)
            {
                var exits = new List<int>();
                for (var index = 0; index < count; index++)
                {
                    if (!body[index])
                    {
                        continue;
                    }

                    foreach (var successor in _successors[index])
                    {
                        if (successor == ExitTarget || body[successor] || exits.Contains(successor))
                        {
                            continue;
                        }

                        exits.Add(successor);
                    }
                }

                if (exits.Count <= 1)
                {
                    loops[header] = new LoopInfo(header, body, exits.Count == 1 ? exits[0] : NoFollow);
                    continue;
                }

                // Several distinct exits: only one can be a plain break target, so each
                // records which it was and the loop dispatches after it (EmitLoopExits).
                var continuation = CommonPostDominator(exits);
                if (!MultiExitLoops || (continuation >= 0 && body[continuation]))
                {
                    _structureFailure ??= $"loop with several exits: {exits[0]} and {exits[1]}";
                    return false;
                }

                loops[header] = new LoopInfo(header, body, MultiExitFollow, [.. exits], continuation);
            }

            // Loops must nest: one body contains the other or they are disjoint.
            foreach (var outer in loops.Values)
            {
                foreach (var inner in loops.Values)
                {
                    if (ReferenceEquals(outer, inner))
                    {
                        continue;
                    }

                    var overlap = false;
                    var innerInOuter = true;
                    var outerInInner = true;
                    for (var index = 0; index < count; index++)
                    {
                        overlap |= outer.Body[index] && inner.Body[index];
                        innerInOuter &= !inner.Body[index] || outer.Body[index];
                        outerInInner &= !outer.Body[index] || inner.Body[index];
                    }

                    if (overlap && !innerInOuter && !outerInInner)
                    {
                        _structureFailure ??= "partially overlapping loops";
                        return false;
                    }
                }
            }

            return true;
        }

        // The nearest block that post-dominates every node given, or ExitTarget when only the
        // program's end does.
        private int CommonPostDominator(List<int> nodes)
        {
            var chain = new List<int>();
            for (var cursor = nodes[0]; cursor >= 0 && chain.Count <= _ipdom.Length; cursor = _ipdom[cursor])
            {
                chain.Add(cursor);
            }

            foreach (var candidate in chain)
            {
                var common = true;
                for (var index = 1; index < nodes.Count && common; index++)
                {
                    var reached = false;
                    var steps = 0;
                    for (var cursor = nodes[index]; cursor >= 0 && steps <= _ipdom.Length; cursor = _ipdom[cursor], steps++)
                    {
                        if (cursor == candidate)
                        {
                            reached = true;
                            break;
                        }
                    }

                    common = reached;
                }

                if (common)
                {
                    return candidate;
                }
            }

            return ExitTarget;
        }

        private static void CollectLoopBody(
            bool[] body,
            int header,
            int latch,
            List<int>[] predecessors)
        {
            var stack = new Stack<int>();
            if (!body[latch])
            {
                body[latch] = true;
                stack.Push(latch);
            }

            while (stack.Count != 0)
            {
                var node = stack.Pop();
                foreach (var predecessor in predecessors[node])
                {
                    if (predecessor == header || body[predecessor])
                    {
                        continue;
                    }

                    body[predecessor] = true;
                    stack.Push(predecessor);
                }
            }
        }

        private bool Dominates(int dominator, int node)
        {
            for (var cursor = node; cursor != -1; cursor = _idom[cursor])
            {
                if (cursor == dominator)
                {
                    return true;
                }

                if (_idom[cursor] == cursor)
                {
                    break;
                }
            }

            return false;
        }

        /// <summary>Immediate dominators by the Cooper-Harvey-Kennedy iteration;
        /// unreachable blocks get -1.</summary>
        private int[] Dominators(int count)
        {
            var order = ReversePostorder(count, forward: true, out var index);
            var idom = new int[count];
            Array.Fill(idom, -1);
            idom[0] = 0;
            var predecessors = EdgeLists(count, forward: false);
            var changed = true;
            while (changed)
            {
                changed = false;
                foreach (var node in order)
                {
                    if (node == 0)
                    {
                        continue;
                    }

                    var candidate = -1;
                    foreach (var predecessor in predecessors[node])
                    {
                        if (idom[predecessor] == -1)
                        {
                            continue;
                        }

                        candidate = candidate == -1
                            ? predecessor
                            : Intersect(candidate, predecessor, idom, index);
                    }

                    if (candidate != -1 && idom[node] != candidate)
                    {
                        idom[node] = candidate;
                        changed = true;
                    }
                }
            }

            return idom;
        }

        /// <summary>Immediate post-dominators, computed as dominators of the reverse CFG
        /// rooted at the single exit. A block that cannot reach the exit gets -1, and a
        /// block whose post-dominator is the exit itself gets <see cref="ExitTarget"/>.</summary>
        private int[] PostDominators(int count)
        {
            // The exit is node "count" in the reversed graph.
            var successors = new List<int>[count + 1];
            var predecessors = new List<int>[count + 1];
            for (var node = 0; node <= count; node++)
            {
                successors[node] = [];
                predecessors[node] = [];
            }

            for (var node = 0; node < count; node++)
            {
                foreach (var target in _successors[node])
                {
                    var resolved = target == ExitTarget ? count : target;
                    // Reversed: the post-dominator problem is dominance from the exit.
                    successors[resolved].Add(node);
                    predecessors[node].Add(resolved);
                }
            }

            var order = new List<int>(count + 1);
            var index = new int[count + 1];
            Array.Fill(index, -1);
            var visited = new bool[count + 1];
            var postorder = new List<int>(count + 1);
            var stack = new Stack<(int Node, int Next)>();
            visited[count] = true;
            stack.Push((count, 0));
            while (stack.Count != 0)
            {
                var (node, next) = stack.Pop();
                if (next < successors[node].Count)
                {
                    stack.Push((node, next + 1));
                    var successor = successors[node][next];
                    if (!visited[successor])
                    {
                        visited[successor] = true;
                        stack.Push((successor, 0));
                    }

                    continue;
                }

                postorder.Add(node);
            }

            postorder.Reverse();
            foreach (var node in postorder)
            {
                index[node] = order.Count;
                order.Add(node);
            }

            var idom = new int[count + 1];
            Array.Fill(idom, -1);
            idom[count] = count;
            var changed = true;
            while (changed)
            {
                changed = false;
                foreach (var node in order)
                {
                    if (node == count)
                    {
                        continue;
                    }

                    var candidate = -1;
                    foreach (var predecessor in predecessors[node])
                    {
                        if (idom[predecessor] == -1)
                        {
                            continue;
                        }

                        candidate = candidate == -1
                            ? predecessor
                            : Intersect(candidate, predecessor, idom, index);
                    }

                    if (candidate != -1 && idom[node] != candidate)
                    {
                        idom[node] = candidate;
                        changed = true;
                    }
                }
            }

            var result = new int[count];
            for (var node = 0; node < count; node++)
            {
                result[node] = idom[node] == -1 ? -1 : idom[node] == count ? ExitTarget : idom[node];
            }

            return result;
        }

        private static int Intersect(int left, int right, int[] idom, int[] index)
        {
            while (left != right)
            {
                while (index[left] > index[right])
                {
                    left = idom[left];
                }

                while (index[right] > index[left])
                {
                    right = idom[right];
                }
            }

            return left;
        }

        private List<int>[] EdgeLists(int count, bool forward)
        {
            var lists = new List<int>[count];
            for (var node = 0; node < count; node++)
            {
                lists[node] = [];
            }

            for (var node = 0; node < count; node++)
            {
                foreach (var target in _successors[node])
                {
                    if (target == ExitTarget)
                    {
                        continue;
                    }

                    if (forward)
                    {
                        lists[node].Add(target);
                    }
                    else
                    {
                        lists[target].Add(node);
                    }
                }
            }

            return lists;
        }

        private List<int> ReversePostorder(int count, bool forward, out int[] index)
        {
            var successors = EdgeLists(count, forward);
            var visited = new bool[count];
            var postorder = new List<int>(count);
            var stack = new Stack<(int Node, int Next)>();
            visited[0] = true;
            stack.Push((0, 0));
            while (stack.Count != 0)
            {
                var (node, next) = stack.Pop();
                if (next < successors[node].Count)
                {
                    stack.Push((node, next + 1));
                    var successor = successors[node][next];
                    if (!visited[successor])
                    {
                        visited[successor] = true;
                        stack.Push((successor, 0));
                    }

                    continue;
                }

                postorder.Add(node);
            }

            postorder.Reverse();
            index = new int[count];
            Array.Fill(index, -1);
            for (var position = 0; position < postorder.Count; position++)
            {
                index[postorder[position]] = position;
            }

            return postorder;
        }
    }
}
