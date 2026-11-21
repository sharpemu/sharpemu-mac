// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.ShaderCompiler.Metal;

// Control-flow emission. A decoded Gen5 program is a list of basic blocks; this
// file turns that list into MSL statements two different ways.
//
// The original form is a PC dispatcher — while (active) { switch (pc) { ... } } —
// ported from the SPIR-V translator. It is universal but ruinous on Apple GPUs:
// every guest register is live across the loop back edge, so the whole register
// file has to stay resident between any two blocks and the compiler either keeps
// 384 dwords in registers (destroying occupancy) or spills them to stack memory.
//
// The structured form emits the blocks once each, in reverse postorder, as a
// linear sequence of labelled statements: fallthrough where the successor is the
// next block, real if/else on the scalar branch condition, while (true) with
// break/continue around natural loops, and a forward or backward goto for the
// leftover edges a tree of loops and conditionals cannot express. No pc variable
// exists, so the compiler sees the program's real CFG and live ranges collapse to
// the span each value is actually used over.
//
// Lane semantics are untouched: every control transfer here comes from the same
// scalar-branch condition the dispatcher evaluated (SCC, the VCC/EXEC register
// pair, or an unconditional branch), each Metal thread follows its own path
// exactly as it did when it carried its own pc, and per-lane masking stays in the
// "if (exec)" guards the instruction emitters write.
public static partial class Gen5MslTranslator
{
    /// <summary>Successor id standing for "leave the program" (S_ENDPGM, or a branch
    /// past the last instruction).</summary>
    private const int ExitTarget = -1;

    private enum TerminatorKind
    {
        /// <summary>Unconditional transfer to <see cref="BlockTerminator.TrueTarget"/>;
        /// also covers plain fallthrough into the next block.</summary>
        Jump,

        /// <summary>Scalar branch: <see cref="BlockTerminator.Condition"/> selects between
        /// the two targets.</summary>
        CondJump,
    }

    /// <summary>How a block leaves, independent of how it is emitted. The dispatcher
    /// lines are kept verbatim so dispatcher output is byte-identical to the form that
    /// shipped before the structured emitter existed.</summary>
    private sealed class BlockTerminator
    {
        public TerminatorKind Kind;
        public int TrueTarget = ExitTarget;
        public int FalseTarget = ExitTarget;
        public string Condition = string.Empty;
        public List<string> DispatcherLines = [];

        public static BlockTerminator Exit(params string[] lines) => new()
        {
            Kind = TerminatorKind.Jump,
            TrueTarget = ExitTarget,
            DispatcherLines = [.. lines],
        };
    }

    private sealed partial class CompilationContext
    {
        /// <summary>Per-block statement text, each at indentation level zero, in guest
        /// block order. The assemblers below re-indent it into place.</summary>
        private readonly List<string> _blockText = [];

        private readonly List<BlockTerminator> _blockTerminators = [];

        /// <summary>Emits every block's straight-line body into <see cref="_blockText"/>
        /// and records how each one leaves.</summary>
        private bool TryEmitBlocks(IReadOnlyList<ShaderBlock> blocks, out string error)
        {
            for (var index = 0; index < blocks.Count; index++)
            {
                _body.Clear();
                _indent = 0;
                ForgetHeapMarks();
                if (!TryEmitBlockBody(blocks, index, out var terminator, out error))
                {
                    error = $"block=0x{blocks[index].StartPc:X}: {error}";
                    return false;
                }

                _blockText.Add(_body.ToString());
                _blockTerminators.Add(terminator);
            }

            _body.Clear();
            error = string.Empty;
            return true;
        }

        // ---- dispatcher assembly ----

        /// <summary>The PC dispatcher's switch body: one case per block, the block's
        /// statements, then the pc/active updates its terminator needs.</summary>
        private string BuildDispatcherSwitchBody(IReadOnlyList<ShaderBlock> blocks, int baseIndent)
        {
            var body = new StringBuilder();
            for (var index = 0; index < blocks.Count; index++)
            {
                AppendIndented(
                    body,
                    Gen5MslDump.Enabled
                        ? $"case {index}u: // pc=0x{blocks[index].StartPc:X4}"
                        : $"case {index}u:",
                    baseIndent);
                AppendIndented(body, "{", baseIndent);
                AppendBlockText(body, _blockText[index], baseIndent + 1);
                foreach (var line in _blockTerminators[index].DispatcherLines)
                {
                    AppendIndented(body, line, baseIndent + 1);
                }

                AppendIndented(body, "}", baseIndent);
                AppendIndented(body, "break;", baseIndent);
            }

            return body.ToString();
        }

        private static void AppendIndented(StringBuilder target, string text, int indent)
        {
            for (var level = 0; level < indent; level++)
            {
                target.Append("    ");
            }

            target.AppendLine(text);
        }

        /// <summary>Re-indents a block body (already newline-terminated, relative
        /// indentation preserved) to sit at <paramref name="indent"/>.</summary>
        private static void AppendBlockText(StringBuilder target, string text, int indent)
        {
            var start = 0;
            while (start < text.Length)
            {
                var end = text.IndexOf('\n', start);
                if (end < 0)
                {
                    end = text.Length;
                }

                var line = text.AsSpan(start, end - start).TrimEnd('\r');
                if (line.Length != 0)
                {
                    for (var level = 0; level < indent; level++)
                    {
                        target.Append("    ");
                    }

                    target.Append(line);
                }

                target.AppendLine();
                start = end + 1;
            }
        }
    }
}
