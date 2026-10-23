// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Ir;

public static partial class Gen5ExecFullAnalysis
{
    // Pointwise identities over the actual entry-mask halves, kept as independent
    // symbols so copies between halves cannot invent lanes in a partial wave. A reduced Boolean decision diagram
    // proves identities such as (entry & p) | (entry & ~p) == entry, including copies
    // through SGPRs and joins. It reasons about bits only, never about scalar branch
    // outcomes. Unknown arithmetic/relative writes destroy the corresponding facts.
    private static IReadOnlySet<uint> AnalyzeEntryIdentities(
        Gen5ShaderProgram program, bool wave32, bool clampExecHigh, IReadOnlySet<uint> full)
    {
        var instructions = program.Instructions;
        var result = new HashSet<uint>(full);
        if (instructions.Count == 0 || instructions.Any(static instruction =>
                instruction.Opcode.StartsWith("SSetpc", StringComparison.Ordinal) ||
                instruction.Opcode.StartsWith("SSwappc", StringComparison.Ordinal) ||
                instruction.Opcode.StartsWith("SCall", StringComparison.Ordinal) ||
                instruction.Opcode.StartsWith("SRfe", StringComparison.Ordinal) ||
                instruction.Opcode.StartsWith("SSubvector", StringComparison.Ordinal) ||
                instruction.Opcode.StartsWith("SCbranchG", StringComparison.Ordinal) ||
                instruction.Opcode == "SCbranchJoin"))
            return result;

        var byPc = new Dictionary<uint, int>(instructions.Count);
        for (var index = 0; index < instructions.Count; index++)
            if (!byPc.TryAdd(instructions[index].Pc, index))
                return new HashSet<uint>(); // synthetic/ambiguous PCs cannot carry a PC proof

        var masks = new MaskIdentities();
        var entryLow = masks.Variable(0);
        var entryHigh = wave32 ? 0 : masks.Variable(1);
        var states = new int[instructions.Count][];
        states[0] = UnknownMasks();
        states[0][ExecLow] = entryLow;
        states[0][ExecHigh] = entryHigh;
        var queue = new Queue<int>();
        var queued = new bool[instructions.Count];
        queue.Enqueue(0);
        queued[0] = true;
        while (queue.TryDequeue(out var index))
        {
            queued[index] = false;
            var instruction = instructions[index];
            var input = (int[])states[index].Clone();
            var output = TransferIdentities(instruction, input, index, masks);
            if (clampExecHigh) output[ExecHigh] = 0;
            void Flow(int successor)
            {
                // No symbolic value from a previous iteration may alias a producer
                // in this iteration. Every cycle crosses a retreating edge. The full
                // mask analysis still supplies its loop-invariant facts in the result.
                var edge = successor <= index ? UnknownMasks() : output;
                var changed = false;
                if (states[successor] is not { } previous)
                {
                    states[successor] = (int[])edge.Clone();
                    changed = true;
                }
                else
                {
                    for (var register = 0; register < previous.Length; register++)
                        if (previous[register] != -1 && previous[register] != edge[register])
                        {
                            previous[register] = -1;
                            changed = true;
                        }
                }
                if (changed && !queued[successor])
                {
                    queued[successor] = true;
                    queue.Enqueue(successor);
                }
            }
            if (Gen5IrBranchResolver.IsTerminator(instruction)) continue;
            if (Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var pc))
            {
                if (!byPc.TryGetValue(pc, out var target)) return full;
                Flow(target);
                if (Gen5IrBranchResolver.IsUnconditionalBranch(instruction)) continue;
            }
            if (index + 1 < instructions.Count) Flow(index + 1);
        }
        for (var index = 0; index < states.Length; index++)
            if (states[index] is { } state && masks.Covers(entryLow, state[ExecLow]) &&
                (wave32 || masks.Covers(entryHigh, state[ExecHigh])))
                result.Add(instructions[index].Pc);
        return result;
    }

    private static int[] UnknownMasks()
    {
        var state = new int[128];
        Array.Fill(state, -1);
        return state;
    }

    private static int[] TransferIdentities(Gen5ShaderInstruction instruction, int[] input, int index, MaskIdentities masks)
    {
        var output = (int[])input.Clone();
        var opcode = instruction.Opcode;
        if (opcode.Contains("Movrel", StringComparison.Ordinal)) return UnknownMasks();
        void Forget(uint register, uint width)
        {
            for (var part = 0u; part < width && register + part < 128; part++)
                output[register + part] = masks.Variable(2 + index * 128 + (int)(register + part));
        }
        foreach (var destination in instruction.Destinations)
            if (destination.Kind == Gen5OperandKind.ScalarRegister)
                Forget(destination.Value, instruction.Control is Gen5ScalarMemoryControl memory
                    ? memory.DestinationCount : instruction.DestinationWidth);
        var scalarDestination = instruction.Control switch
        {
            Gen5Vop3Control control => control.ScalarDestination,
            Gen5SdwaControl control => control.ScalarDestination,
            _ => null,
        };
        if (opcode.StartsWith("VCmpx", StringComparison.Ordinal) || opcode.StartsWith("VCmpsx", StringComparison.Ordinal))
            Forget(ExecLow, 2);
        else if (scalarDestination is { } scalar) Forget(scalar, 2);
        else if (opcode.StartsWith("VCmp", StringComparison.Ordinal) ||
                 (opcode.StartsWith('V') && opcode.Contains("Co", StringComparison.Ordinal))) Forget(106, 2);
        if (opcode.Contains("Wrexec", StringComparison.Ordinal)) Forget(ExecLow, 2);

        if (!opcode.StartsWith('S') || instruction.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister } dest, ..])
            return output;
        var width = opcode.EndsWith("B64", StringComparison.Ordinal) ? 2u : 1u;
        var save = opcode.Contains("Saveexec", StringComparison.Ordinal);
        if (save) Forget(ExecLow, width);
        var kind = save ? opcode[1..opcode.IndexOf("Saveexec", StringComparison.Ordinal)]
            : opcode.EndsWith("B32", StringComparison.Ordinal) || opcode.EndsWith("B64", StringComparison.Ordinal)
                ? opcode[1..^3] : "";
        int Read(Gen5Operand operand, uint part) => operand.Kind switch
        {
            Gen5OperandKind.ScalarRegister when operand.Value + part < 128 => input[operand.Value + part],
            Gen5OperandKind.EncodedConstant when operand.Value == 128 => 0,
            Gen5OperandKind.EncodedConstant when operand.Value == 193 => 1,
            Gen5OperandKind.LiteralConstant when operand.Value == 0 => 0,
            // Literal dwords zero-extend in RawSource64; only inline -1 sign-extends.
            Gen5OperandKind.LiteralConstant when operand.Value == uint.MaxValue => part == 0 ? 1 : 0,
            _ => -1,
        };
        if (instruction.Sources.Count == 0) return output;
        for (var part = 0u; part < width; part++)
        {
            var left = Read(instruction.Sources[0], part);
            var right = save ? input[ExecLow + part]
                : instruction.Sources.Count > 1 ? Read(instruction.Sources[1], part) : -1;
            var value = kind switch
            {
                "Mov" => left,
                "Wqm" when left == 1 => 1,
                "Not" => masks.Not(left),
                "And" => masks.Apply('&', left, right),
                "Or" => masks.Apply('|', left, right),
                "Xor" => masks.Apply('^', left, right),
                "Andn1" => masks.Apply('&', masks.Not(left), right),
                "Andn2" => masks.Apply('&', left, masks.Not(right)),
                "Orn1" => masks.Apply('|', masks.Not(left), right),
                "Orn2" => masks.Apply('|', left, masks.Not(right)),
                "Nand" => masks.Not(masks.Apply('&', left, right)),
                "Nor" => masks.Not(masks.Apply('|', left, right)),
                "Xnor" => masks.Not(masks.Apply('^', left, right)),
                _ => -1,
            };
            if (save)
            {
                if (dest.Value + part < 128) output[dest.Value + part] = input[ExecLow + part];
                output[ExecLow + part] = value;
            }
            else if (dest.Value + part < 128 && kind is "Mov" or "Wqm" or "Not" or "And" or "Or" or
                     "Xor" or "Andn1" or "Andn2" or "Orn1" or "Orn2" or "Nand" or "Nor" or "Xnor")
                output[dest.Value + part] = value;
        }
        return output;
    }

    private sealed class MaskIdentities
    {
        private readonly List<(int Variable, int Low, int High)> _nodes = [(int.MaxValue, 0, 0), (int.MaxValue, 1, 1)];
        private readonly Dictionary<(int, int, int), int> _unique = [];
        private readonly Dictionary<(char, int, int), int> _cache = [];
        public int Variable(int variable) => Node(variable, 0, 1);
        public int Not(int value) => Apply('^', value, 1);
        public bool Covers(int entry, int value) => value >= 0 && Apply('&', entry, Not(value)) == 0;
        private int Node(int variable, int low, int high)
        {
            if (low == high) return low;
            if (low < 0 || high < 0 || _nodes.Count >= 8192) return -1;
            if (_unique.TryGetValue((variable, low, high), out var existing)) return existing;
            var id = _nodes.Count;
            _nodes.Add((variable, low, high));
            _unique[(variable, low, high)] = id;
            return id;
        }
        public int Apply(char op, int left, int right, int depth = 0)
        {
            if (op == '&' && (left == 0 || right == 0)) return 0;
            if (op == '|' && (left == 1 || right == 1)) return 1;
            if (left < 0 || right < 0) return -1;
            if (left > right) (left, right) = (right, left);
            if (left < 2 && right < 2) return op switch { '&' => left & right, '|' => left | right, _ => left ^ right };
            if (left == right) return op == '^' ? 0 : left;
            if (_cache.TryGetValue((op, left, right), out var cached)) return cached;
            var variable = Math.Min(_nodes[left].Variable, _nodes[right].Variable);
            // A deeply constructed adversarial expression is refused conservatively.
            if (depth >= 128 || _nodes.Count >= 8192 || _cache.Count >= 32768) return -1;
            var a = _nodes[left];
            var b = _nodes[right];
            var low = Apply(op, a.Variable == variable ? a.Low : left, b.Variable == variable ? b.Low : right, depth + 1);
            var high = Apply(op, a.Variable == variable ? a.High : left, b.Variable == variable ? b.High : right, depth + 1);
            var result = Node(variable, low, high);
            _cache[(op, left, right)] = result;
            return result;
        }
    }
}
