// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Ir;

/// <summary>
/// Recognises a persistent-threads kernel: one whose groups each loop over a shared work queue
/// until it is empty, polling coherent memory for work other groups publish. Such a kernel
/// sizes its dispatch to the groups the console GPU keeps resident at once (UE5's Nanite
/// culling launches 1440), and on a host that keeps fewer resident the groups that run spin on
/// work held by groups that cannot start. Every group is interchangeable there, so a host can
/// launch only as many as it keeps resident and compute the same result.
/// <para>
/// The shape it looks for: the outermost loop spans most of the program, polls coherent memory
/// (a GLC load or a returning atomic) and writes device memory inside, and little follows it.
/// A kernel that waits on another group only briefly - a decoupled look-back scan polls its
/// predecessor's flag in a small loop and does its work outside it - does not have that shape,
/// and must not be shrunk, since each of its groups takes one tile.
/// </para>
/// </summary>
public static class Gen5PersistentThreadsAnalysis
{
    /// <summary>Whether the outermost loop holds at least half the program.</summary>
    private const double MinimumLoopShare = 0.5;

    /// <summary>The most instructions that may follow the outermost loop's end.</summary>
    private const double MaximumTailShare = 0.1;

    public static bool IsPersistent(Gen5ShaderProgram program)
    {
        var instructions = program.Instructions;
        if (instructions.Count < 16)
        {
            return false;
        }

        var successors = Gen5Wave64HalfMaskAnalysis.BuildSuccessors(instructions);
        if (successors is null)
        {
            return false;
        }

        // The widest back edge: code is laid out with loop bodies contiguous, so the span from
        // the target to the branch is the loop.
        var loopStart = -1;
        var loopEnd = -1;
        for (var index = 0; index < instructions.Count; index++)
        {
            foreach (var successor in successors[index])
            {
                if (successor <= index && (loopEnd - loopStart) < (index - successor))
                {
                    loopStart = successor;
                    loopEnd = index;
                }
            }
        }

        if (loopStart < 0 ||
            loopEnd - loopStart + 1 < instructions.Count * MinimumLoopShare ||
            instructions.Count - 1 - loopEnd > instructions.Count * MaximumTailShare)
        {
            return false;
        }

        var polls = false;
        var writes = false;
        for (var index = loopStart; index <= loopEnd; index++)
        {
            var instruction = instructions[index];
            var opcode = instruction.Opcode;
            var store = opcode.Contains("Store", StringComparison.Ordinal);
            var atomic = opcode.Contains("Atomic", StringComparison.Ordinal);
            var glc = instruction.Control is Gen5BufferMemoryControl { Glc: true } or Gen5GlobalMemoryControl { Glc: true };
            if (glc && !store)
            {
                polls = true;
            }

            if ((store || atomic) &&
                instruction.Control is Gen5BufferMemoryControl or Gen5GlobalMemoryControl)
            {
                writes = true;
            }
        }

        return polls && writes;
    }
}
