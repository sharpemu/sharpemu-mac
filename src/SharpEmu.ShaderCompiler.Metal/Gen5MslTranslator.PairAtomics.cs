// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const string PairLocksExpression = $"{ResourcesName}.{Gen5MslArgumentLayout.PairLocksName}";
        private const string PairStepReportExpression = $"{ResourcesName}.{Gen5MslArgumentLayout.StepReportName}";

        /// <summary>A guest dword-pair atomic (an *_X2 atomic) under the pair's sequence lock
        /// (Templates/pair_atomic_prelude.msl), so a GLC LOAD_DWORDX2 never sees one dword
        /// from before it and one from after. The body runs each dword's 32-bit atomic; it is
        /// emitted inside the branch that took the lock, which is what keeps lanes of one
        /// SIMD-group that contend for a slot from waiting on each other.</summary>
        private void EmitPairLocked(string firstWord, Action body)
        {
            _usesPairLocks = true;
            var lockWord = Temp("device atomic_uint*", $"sharpemu_pair_lock({PairLocksExpression}, {firstWord})");
            var sequence = Temp("uint", "0u");
            var tries = Temp("uint", "0u");
            var done = Temp("bool", "false");
            Line($"while (!{done})");
            Line("{");
            _indent++;
            Line($"if (sharpemu_pair_try_lock({lockWord}, {sequence}, {tries}, {PairStepReportExpression}))");
            Line("{");
            _indent++;
            body();
            Line($"sharpemu_pair_unlock({lockWord}, {sequence});");
            Line($"{done} = true;");
            _indent--;
            Line("}");
            _indent--;
            Line("}");
        }

        /// <summary>A body that runs inside the lock of the shared-LDS qword holding the
        /// dword at <paramref name="wordIndex"/> (Templates/pair_atomic_prelude.msl,
        /// sharpemu_lds_try_lock). Taken and released inside one branch of one loop trip,
        /// like <see cref="EmitPairLocked"/>, so lanes of one SIMD-group contending for it
        /// cannot deadlock each other.</summary>
        private void EmitLdsLocked(string wordIndex, Action body)
        {
            _usesPairLocks = true;
            var lockWord = Temp(
                "device atomic_uint*",
                $"sharpemu_lds_pair_lock({PairLocksExpression}, sharpemu_group_id, ({wordIndex}) >> 1u)");
            var sequence = Temp("uint", "0u");
            var tries = Temp("uint", "0u");
            var done = Temp("bool", "false");
            Line($"while (!{done})");
            Line("{");
            _indent++;
            Line($"if (sharpemu_lds_try_lock({lockWord}, {sequence}, {tries}, {PairStepReportExpression}))");
            Line("{");
            _indent++;
            body();
            Line($"sharpemu_lds_unlock({lockWord}, {sequence});");
            Line($"{done} = true;");
            _indent--;
            Line("}");
            _indent--;
            Line("}");
        }

        /// <summary>A GLC LOAD_DWORDX2 read as one snapshot (see <see cref="EmitPairLocked"/>)
        /// into two destination registers. The helper call is the whole read, so it runs
        /// only for lanes EXEC has set unless the caller already guards on it.</summary>
        private void StorePairLoad(uint destination, string pairExpression, bool guardWithExec = true)
        {
            _usesPairLocks = true;
            var pair = Temp("uint2", "uint2(0u)");
            if (guardWithExec) NoteExecGuard();
            Line(guardWithExec && !_execKnownFull
                ? $"if (exec) {{ {pair} = {pairExpression}; }}"
                : $"{pair} = {pairExpression};");
            StoreVector(destination, $"{pair}.x", guardWithExec);
            StoreVector(destination + 1, $"{pair}.y", guardWithExec);
        }

        private static string PairLockArguments => $"{PairLocksExpression}, {PairStepReportExpression}";

        /// <summary>The operation suffixes of the guest *_ATOMIC_*_X2 opcodes (RDNA2 ISA 12.14,
        /// 12.18), all of which <see cref="EmitQwordAtomic"/> performs.</summary>
        private static bool IsQwordAtomic(string name) => name is
            "Swap" or "Cmpswap" or "Add" or "Sub" or "Smin" or "Umin" or "Smax" or "Umax" or
            "And" or "Or" or "Xor" or "Inc" or "Dec" or "Fcmpswap" or "Fmin" or "Fmax";

        /// <summary>
        /// One guest 64-bit atomic on the qword whose dwords are <paramref name="first"/> and
        /// <paramref name="second"/>, emitted where the caller has decided the lane performs it.
        /// <paramref name="source"/> and <paramref name="comparator"/> are ulong values the caller
        /// read before any destination register is written (CMPSWAP_X2 and FCMPSWAP_X2 take DATA
        /// from VDATA[0:1] and the comparator from VDATA[2:3]); the pre-operation qword lands in
        /// <paramref name="originalLow"/> and <paramref name="originalHigh"/>.
        ///
        /// Metal has no 64-bit exchange, compare-exchange or fetch operation, so the whole
        /// read-modify-write runs under the pair's sequence lock (<see cref="EmitPairLocked"/>):
        /// other X2 atomics and GLC LOAD_DWORDX2 take the same lock, so to them it is one
        /// access. Inside it:
        /// - AND/OR/XOR/SWAP act on each dword on its own, so each half is a 32-bit atomic;
        /// - ADD/SUB fetch-add the low dword and carry (borrow) into a fetch-add of the high one;
        /// both fetches return the dword the lock left there, so the returned qword is exact;
        /// - every other operation loads both dwords, computes the new qword and stores it.
        /// A 32-bit atomic or a plain store that does not take the lock and lands on the same
        /// qword between that load and store is overwritten, which a guest mixing dword and
        /// qword atomics on one location could observe; no guest pattern doing so is known.
        /// The float forms compare on the bits (MSL has no double): FCMPSWAP_X2 swaps when the
        /// qword equals the comparator as a double (a NaN equals nothing, -0 equals +0);
        /// FMIN_X2/FMAX_X2 replace the qword with DATA when DATA orders below (above) it,
        /// the ISA's "(src &lt; tmp) ? src : tmp", so a NaN on either side keeps memory, as
        /// the 32-bit BUFFER_ATOMIC_FMIN/FMAX here do.
        /// </summary>
        private void EmitQwordAtomic(
            string name,
            string first,
            string second,
            string source,
            string comparator,
            string originalLow,
            string originalHigh)
        {
            _usesFloat64Atomics |= name is "Fcmpswap" or "Fmin" or "Fmax";
            var low = $"(uint)({source})";
            var high = $"(uint)(({source}) >> 32)";
            var lowWord = $"(device atomic_uint*)({first})";
            var highWord = $"(device atomic_uint*)({second})";
            EmitPairLocked(first, () =>
            {
                switch (name)
                {
                    case "And" or "Or" or "Xor":
                    {
                        var function = $"atomic_fetch_{name.ToLowerInvariant()}_explicit";
                        Line($"{originalLow} = {function}({lowWord}, {low}, memory_order_relaxed);");
                        Line($"{originalHigh} = {function}({highWord}, {high}, memory_order_relaxed);");
                        return;
                    }
                    case "Swap":
                        Line($"{originalLow} = atomic_exchange_explicit({lowWord}, {low}, memory_order_relaxed);");
                        Line($"{originalHigh} = atomic_exchange_explicit({highWord}, {high}, memory_order_relaxed);");
                        return;
                    case "Add":
                    {
                        Line($"{originalLow} = atomic_fetch_add_explicit({lowWord}, {low}, memory_order_relaxed);");
                        var carry = Temp("uint", $"({originalLow} + {low}) < {originalLow} ? 1u : 0u");
                        Line($"{originalHigh} = atomic_fetch_add_explicit({highWord}, {high} + {carry}, memory_order_relaxed);");
                        return;
                    }
                    case "Sub":
                    {
                        Line($"{originalLow} = atomic_fetch_sub_explicit({lowWord}, {low}, memory_order_relaxed);");
                        var borrow = Temp("uint", $"{originalLow} < {low} ? 1u : 0u");
                        Line($"{originalHigh} = atomic_fetch_sub_explicit({highWord}, {high} + {borrow}, memory_order_relaxed);");
                        return;
                    }
                }

                Line($"{originalLow} = atomic_load_explicit({lowWord}, memory_order_relaxed);");
                Line($"{originalHigh} = atomic_load_explicit({highWord}, memory_order_relaxed);");
                var old = Temp("ulong", $"(ulong){originalLow} | ((ulong){originalHigh} << 32)");
                var next = Temp("ulong", QwordAtomicResult(name, old, source, comparator));
                Line($"atomic_store_explicit({lowWord}, (uint){next}, memory_order_relaxed);");
                Line($"atomic_store_explicit({highWord}, (uint)({next} >> 32), memory_order_relaxed);");
            });
        }

        /// <summary>The register pair starting at <paramref name="register"/> as a ulong, read now.</summary>
        private string QwordSource(uint register) =>
            Temp("ulong", $"(ulong)v[{register}] | ((ulong)v[{register + 1}] << 32)");

        /// <summary>The qword a read-modify-write X2 atomic leaves (RDNA2 ISA 12.14).</summary>
        private static string QwordAtomicResult(string name, string old, string source, string comparator) => name switch
        {
            "Cmpswap" => $"{old} == {comparator} ? {source} : {old}",
            "Smin" => $"as_type<ulong>(min(as_type<long>({old}), as_type<long>({source})))",
            "Smax" => $"as_type<ulong>(max(as_type<long>({old}), as_type<long>({source})))",
            "Umin" => $"min({old}, {source})",
            "Umax" => $"max({old}, {source})",
            "Inc" => $"{old} >= {source} ? 0ul : {old} + 1ul",
            "Dec" => $"({old} == 0ul || {old} > {source}) ? {source} : {old} - 1ul",
            "Fcmpswap" => $"sharpemu_f64_eq({old}, {comparator}) ? {source} : {old}",
            "Fmin" => $"sharpemu_f64_lt({source}, {old}) ? {source} : {old}",
            "Fmax" => $"sharpemu_f64_lt({old}, {source}) ? {source} : {old}",
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "not a qword atomic"),
        };
    }
}
