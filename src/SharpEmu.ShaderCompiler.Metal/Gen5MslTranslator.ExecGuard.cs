// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal;

// EXEC guard elision.
//
// A vector instruction only writes the lanes EXEC has set, so every VALU result and
// every vector memory store is emitted behind "if (exec)". That is the single most
// common construct in translated Metal: a profile of Octopath Traveler II's shaders
// counted 7951 of them across 67 programs, 505 in the dispatch that spends 19% of the
// frame's GPU time. Each one is a per-lane branch or select around the write, and it
// also keeps the register's previous value live up to the write, which is what the
// old value is for when a lane is masked off.
//
// EXEC starts full and most programs never narrow it. Gen5ExecFullAnalysis — the same
// analysis the SPIR-V translator uses, in the shared IR — says which instructions are
// reached only with every lane set: it follows EXEC through S_WQM of a full mask and
// through a saved copy moved or ORed back, and gives up at V_CMPX or S_AND_SAVEEXEC.
// Where it says full, the guard is dead: no lane keeps its old value, so neither the
// branch nor the old value needs to exist.
//
// Wave64 compute is the one case to be careful with. A 64-lane guest wave runs as two
// 32-wide Apple simdgroups, so "every lane set" means both halves; the analysis is
// told the wave size and the host can withhold the elision per request
// (EnableExecGuardElision), which is how the Vulkan backend handles a driver that
// needs the guards.
//
// EXEC runs. Where EXEC is not provably full, a guard per write is a select per VALU
// result: about a quarter of the GPU time of Black Myth: Wukong's hottest pixel
// programs (2.5 -> 1.9 ms on the shader bench with every guard removed). A run of
// consecutive VALU instructions that only read and write their own lane's registers
// leaves EXEC alone, so one "if (exec) { ... }" around the run writes exactly the lanes
// the per-write guards did: the hardware masks the lanes off instead, and a simdgroup
// whose lanes are all off skips the run. The run ends before anything that is not
// lane-local - a lane read across the wave, a write to VCC, SCC or a scalar register (a
// lane that skipped it would keep a stale copy of a value every lane must agree on), a
// derivative, a barrier - so those keep running in every lane. The emitted text of a
// run is checked for such constructs as well, and a program where one slipped through
// is translated again with a guard per write (ExecRunRefused).
public static partial class Gen5MslTranslator
{
    private static long _execRunsRefused;

    /// <summary>Programs translated again with a guard per write because an EXEC run met
    /// something that is not lane-local.</summary>
    public static long ExecRunsRefused => Interlocked.Read(ref _execRunsRefused);

    private sealed partial class CompilationContext
    {
        private bool _execRunOpen;
        private int _execRunStart;

        /// <summary>A run's text held a construct that is not lane-local; the program is
        /// not used and is translated again without runs.</summary>
        internal bool ExecRunRefused { get; private set; }

        /// <summary>Pcs the analysis proved run with every EXEC lane set, computed on
        /// first use because most programs never ask.</summary>
        private IReadOnlySet<uint>? _execFullPcs;

        /// <summary>Set while the instruction being emitted is one of those: its vector
        /// writes need no guard.</summary>
        private bool _execKnownFull;
        private Ir.Gen5VectorObservationPlan? _observationPlan;
        private bool _observationPlanBuilt;
        private Ir.VectorRegisterSet _unobservedAluWrites;

        private Ir.Gen5VectorObservationPlan? _laneObservationPlan;
        private bool _laneObservationPlanBuilt;

        private void BeginInstruction(Gen5ShaderInstruction instruction)
        {
            _execKnownFull = _execRunOpen || IsExecKnownFull(instruction.Pc);
            _unobservedAluWrites = default;
            if (IsUnobservedAluInstruction(instruction))
            {
                _unobservedAluWrites = _laneObservationPlan!.UnobservedWrites(instruction.Pc);
                return;
            }

            // Only pure ALU destination writes participate. Memory, exports and
            // mask updates retain their guards even when their result is dead.
            if (_options.UnobservedAluWrites && _options.ExecGuardElision && _request.EnableExecGuardElision &&
                UnobservedWritesApply() &&
                instruction.Encoding is (Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or
                    Gen5ShaderEncoding.Vop3 or Gen5ShaderEncoding.Vop3p) &&
                instruction.Control is not (Gen5DppControl or Gen5Dpp8Control))
            {
                if (!_observationPlanBuilt)
                {
                    _observationPlan = Ir.Gen5VectorObservationAnalysis.Analyze(
                        _request.Program, wave32: _waveLaneCount == 32, trackLaneSpills: _options.ExecLaneSpills);
                    _observationPlanBuilt = true;
                }
                _unobservedAluWrites = _observationPlan?.UnobservedWrites(instruction.Pc) ?? default;
            }
        }

        private bool UnobservedWritesApply() =>
            _options.UnobservedAluAllStages
                ? _stage is Gen5MslStage.Compute or Gen5MslStage.Pixel or Gen5MslStage.Vertex ||
                    (_stage == Gen5MslStage.Mesh && _waveLaneCount == 32 && Mesh.ThreadsPerThreadgroup <= 32)
                : _waveLaneCount == 32 && (_stage == Gen5MslStage.Compute ||
                    (_stage == Gen5MslStage.Mesh && Mesh.ThreadsPerThreadgroup <= 32));

        private bool IsExecKnownFull(uint pc)
        {
            if (!_options.ExecGuardElision || !_request.EnableExecGuardElision)
            {
                return false;
            }

            _execFullPcs ??= Ir.Gen5ExecFullAnalysis.Analyze(
                _request.Program,
                wave32: _waveLaneCount == 32,
                trackLaneSpills: _options.ExecLaneSpills);
            return _execFullPcs.Contains(pc);
        }

        private bool ExecRunsApply => _options.ExecRuns && _options.ExecGuardElision && _request.EnableExecGuardElision;

        /// <summary>
        /// Whether a lane-local ALU instruction writes only registers no masked-off lane reads
        /// back (<see cref="Ir.Gen5LaneObservationAnalysis"/>): it is emitted unguarded, outside
        /// any EXEC run, so its destination's old value need not stay alive for the lanes
        /// EXEC clears. Multi-wave mesh shaders are left out, as for the older analysis.
        /// </summary>
        private bool IsUnobservedAluInstruction(Gen5ShaderInstruction instruction)
        {
            if (!_options.LaneExactUnobservedWrites || !_options.ExecGuardElision || !_request.EnableExecGuardElision ||
                !(_stage is Gen5MslStage.Compute or Gen5MslStage.Pixel or Gen5MslStage.Vertex ||
                    (_stage == Gen5MslStage.Mesh && _waveLaneCount == 32 && Mesh.ThreadsPerThreadgroup <= 32)) ||
                IsExecKnownFull(instruction.Pc) || !IsLaneLocalAlu(instruction))
            {
                return false;
            }

            if (!_laneObservationPlanBuilt)
            {
                _laneObservationPlan = Ir.Gen5LaneObservationAnalysis.Analyze(
                    _request.Program, wave32: _waveLaneCount == 32, trackLaneSpills: _options.ExecLaneSpills);
                _laneObservationPlanBuilt = true;
            }

            var dead = _laneObservationPlan?.UnobservedWrites(instruction.Pc) ?? default;
            if (dead.IsEmpty)
            {
                return false;
            }

            var wide = instruction.Opcode.Contains("64", StringComparison.Ordinal);
            var any = false;
            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind != Gen5OperandKind.VectorRegister ||
                    !dead.Contains(destination.Value) || (wide && !dead.Contains(destination.Value + 1)))
                {
                    return false;
                }

                any = true;
            }

            return any;
        }

        /// <summary>
        /// Whether an instruction may sit inside an EXEC run: a VALU operation whose every
        /// operand and result is a register of its own lane (no DPP, lane reads or writes,
        /// relative register numbers, interpolation), with no scalar result - no compare,
        /// carry, VCC write or scalar destination.
        /// </summary>
        private bool IsExecRunInstruction(Gen5ShaderInstruction instruction) =>
            ExecRunsApply && !IsExecKnownFull(instruction.Pc) && IsLaneLocalAlu(instruction);

        /// <summary>The instructions an EXEC run may hold, whatever EXEC is.</summary>
        private static bool IsLaneLocalAlu(Gen5ShaderInstruction instruction)
        {
            if (instruction.Encoding is not (Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or
                    Gen5ShaderEncoding.Vop3 or Gen5ShaderEncoding.Vop3p) ||
                instruction.Control is Gen5DppControl or Gen5Dpp8Control or
                    Gen5Vop3Control { ScalarDestination: not null } or Gen5SdwaControl { ScalarDestination: not null })
            {
                return false;
            }

            var opcode = instruction.Opcode;
            if (opcode.StartsWith("VCmp", StringComparison.Ordinal) ||
                opcode.StartsWith("VReadlane", StringComparison.Ordinal) ||
                opcode.StartsWith("VReadfirstlane", StringComparison.Ordinal) ||
                opcode.StartsWith("VWritelane", StringComparison.Ordinal) ||
                opcode.StartsWith("VPermlane", StringComparison.Ordinal) ||
                opcode.StartsWith("VMovrel", StringComparison.Ordinal) ||
                opcode.StartsWith("VSwaprel", StringComparison.Ordinal) ||
                opcode.StartsWith("VInterp", StringComparison.Ordinal) ||
                opcode.StartsWith("VDivScale", StringComparison.Ordinal) ||
                opcode.Contains("Co", StringComparison.Ordinal) ||
                opcode is "VAddcU32" or "VSubbU32" or "VSubbrevU32" or "VMadU64U32" or "VMadI64I32" or "VNop")
            {
                return false;
            }

            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind != Gen5OperandKind.VectorRegister)
                {
                    return false;
                }
            }

            return true;
        }

        private void OpenExecRun()
        {
            if (_execRunOpen)
            {
                return;
            }

            Line("if (exec)");
            Line("{");
            _indent++;
            _execRunOpen = true;
            _execRunStart = _body.Length;
        }

        private void CloseExecRun()
        {
            if (!_execRunOpen)
            {
                return;
            }

            CheckExecRun(_execRunStart);
            _indent--;
            Line("}");
            _execRunOpen = false;
        }

        // What must never run in only some lanes: a cross-lane read, a wave-uniform result,
        // a derivative, a barrier or control flow out of the run.
        private static readonly string[] NotLaneLocal =
        [
            "sharpemu_ballot", "simd_", "quad_", "threadgroup_barrier", "sharpemu_wave", "sharpemu_lane <",
            "vcc =", "exec =", "scc =", "atomic_", "sharpemu_lds", "dfdx", "dfdy", "fwidth", ".sample(",
            "discard_fragment", "return", "break;", "continue;", "goto ", "pc =", "active =",
        ];

        private void CheckExecRun(int start)
        {
            if (ExecRunRefused)
            {
                return;
            }

            var text = _body.ToString(start, _body.Length - start);
            foreach (var token in NotLaneLocal)
            {
                if (text.Contains(token, StringComparison.Ordinal))
                {
                    ExecRunRefused = true;
                    return;
                }
            }

            // A scalar register written inside the run ("s[12] = ..." before renaming).
            for (var index = text.IndexOf("s[", StringComparison.Ordinal); index >= 0;
                 index = text.IndexOf("s[", index + 2, StringComparison.Ordinal))
            {
                if (index > 0 && (char.IsLetterOrDigit(text[index - 1]) || text[index - 1] == '_'))
                {
                    continue;
                }

                var close = text.IndexOf(']', index);
                if (close > 0 && close + 3 < text.Length && text.AsSpan(close + 1).TrimStart(' ').StartsWith("= "))
                {
                    ExecRunRefused = true;
                    return;
                }
            }
        }

        /// <summary>One statement behind the per-lane EXEC guard, or the statement alone
        /// where every lane is provably set.</summary>
        private string ExecGuarded(string statement) =>
            _execKnownFull ? statement : $"if (exec) {{ {statement} }}";

        /// <summary>Opens a multi-statement EXEC guard, or nothing where every lane is
        /// provably set. Paired with <see cref="EndExecGuard"/>.</summary>
        private void BeginExecGuard()
        {
            if (_execKnownFull)
            {
                return;
            }

            Line("if (exec)");
            Line("{");
            _indent++;
        }

        private void EndExecGuard()
        {
            if (_execKnownFull)
            {
                return;
            }

            _indent--;
            Line("}");
        }

        /// <summary>The EXEC term of a condition that also guards on something else, as a
        /// prefix that disappears when every lane is provably set.</summary>
        private string ExecCondition(string rest) =>
            _execKnownFull ? rest : $"exec && {rest}";

        /// <summary>A value that survives only where EXEC has the lane set — the form an
        /// export or a stage output takes, since those assign unconditionally.</summary>
        private string ExecSelect(string value, string previous) =>
            _execKnownFull ? value : $"exec ? {value} : {previous}";
    }
}
