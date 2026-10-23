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
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        /// <summary>Pcs the analysis proved run with every EXEC lane set, computed on
        /// first use because most programs never ask.</summary>
        private IReadOnlySet<uint>? _execFullPcs;

        /// <summary>Set while the instruction being emitted is one of those: its vector
        /// writes need no guard.</summary>
        private bool _execKnownFull;

        private void BeginInstruction(Gen5ShaderInstruction instruction)
        {
            _execKnownFull = _insidePredicatedRegion || IsExecKnownFull(instruction.Pc);
        }

        private bool IsExecKnownFull(uint pc)
        {
            if (!_options.ExecGuardElision || !_request.EnableExecGuardElision)
            {
                return false;
            }

            _execFullPcs ??= Ir.Gen5ExecFullAnalysis.Analyze(
                _request.Program,
                wave32: _waveLaneCount == 32 || _stage == Gen5MslStage.Pixel,
                reconvergence: _options.ExecReconvergence,
                clampExecHigh: ClampsExecToLanes || IsNarrowWave64);
            return _execFullPcs.Contains(pc);
        }

        /// <summary>One statement behind the per-lane EXEC guard, or the statement alone
        /// where every lane is provably set.</summary>
        private string ExecGuarded(string statement)
        {
            NoteExecGuard();
            return _execKnownFull ? statement : $"if (exec) {{ {statement} }}";
        }

        /// <summary>Opens a multi-statement EXEC guard, or nothing where every lane is
        /// provably set. Paired with <see cref="EndExecGuard"/>.</summary>
        private void BeginExecGuard()
        {
            NoteExecGuard();
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
        private string ExecCondition(string rest)
        {
            NoteExecGuard();
            return _execKnownFull ? rest : $"exec && {rest}";
        }

        /// <summary>A value that survives only where EXEC has the lane set — the form an
        /// export or a stage output takes, since those assign unconditionally.</summary>
        private string ExecSelect(string value, string previous)
        {
            NoteExecGuard();
            return _execKnownFull ? value : $"exec ? {value} : {previous}";
        }

        private bool _insidePredicatedRegion;
        private bool _predicatedGuardCounted;

        private void NoteExecGuard()
        {
            // The first guarded site is replaced by the region's one outer guard.
            // Later sites are removed. Do not subtract before any site is counted.
            if (_insidePredicatedRegion && !_predicatedGuardCounted)
            {
                _predicatedGuardCounted = true;
                return;
            }
            Statistics.NoteGuard(_execKnownFull);
        }

        // Only local, pure vector ALU. Scalar instructions, mask/carry results, DPP,
        // subgroup operations, memory, samples and derivatives end a region. Nothing
        // moves across them, and inactive lanes keep every previous VGPR value.
        private bool CanShareExecGuard(Gen5ShaderInstruction instruction) =>
            _options.ExecReconvergence && _options.ExecGuardElision && _request.EnableExecGuardElision &&
            !IsExecKnownFull(instruction.Pc) &&
            instruction.Control is null or Gen5Vop3Control { ScalarDestination: null } &&
            instruction.Opcode is "VMovB32" or "VAddF32" or "VSubF32" or "VSubrevF32" or
                "VMulF32" or "VFmaF32" or "VFmacF32" or "VMacF32" or "VMinF32" or "VMaxF32" or
                "VAddU32" or "VSubU32" or "VSubrevU32" or "VAndB32" or "VOrB32" or "VXorB32" or
                "VLshlrevB32" or "VLshrrevB32" or "VAshrrevI32" or "VCndmaskB32";

        private void EndPredicatedRegion()
        {
            if (!_insidePredicatedRegion) return;
            _insidePredicatedRegion = false;
            _indent--;
            Line("}");
        }
    }
}
