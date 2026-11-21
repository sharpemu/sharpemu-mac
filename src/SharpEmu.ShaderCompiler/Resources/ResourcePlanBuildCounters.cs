// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

/// <summary>
/// How many compiled resource plans were built, for the periodic render report. Each build
/// makes a <see cref="System.Reflection.Emit.DynamicMethod"/> or an expression delegate that
/// the finalizer thread later has to tear down, so a steady stream of them is churn.
/// </summary>
public static class ResourcePlanBuildCounters
{
    private static long _evaluatorMethods;
    private static long _transactions;
    private static long _totalMethods;

    internal static void RecordEvaluatorMethod()
    {
        Interlocked.Increment(ref _evaluatorMethods);
        Interlocked.Increment(ref _totalMethods);
    }

    internal static void RecordTransaction()
    {
        Interlocked.Increment(ref _transactions);
        Interlocked.Increment(ref _totalMethods);
    }

    private static long _descriptorTableRefusals;

    // A plan whose device-address ranges stay on the interpreter for its device-descriptor tables
    // (CompiledDeviceAddressRanges.DescriptorTablesRefusal); counted for the whole session.
    internal static void RecordDescriptorTableRefusal() => Interlocked.Increment(ref _descriptorTableRefusals);

    internal static long DescriptorTableRefusals => Interlocked.Read(ref _descriptorTableRefusals);

    /// <summary>The builds since the previous call, as one report line.</summary>
    public static string TakeReport()
    {
        var methods = Interlocked.Exchange(ref _evaluatorMethods, 0);
        var transactions = Interlocked.Exchange(ref _transactions, 0);
        return FormattableString.Invariant(
            $"[PERF][PLAN_BUILDS] dynamic_methods={methods} transactions={transactions} total={Volatile.Read(ref _totalMethods)} ") +
            FormattableString.Invariant($"descriptor_table_refusals={DescriptorTableRefusals}");
    }
}
