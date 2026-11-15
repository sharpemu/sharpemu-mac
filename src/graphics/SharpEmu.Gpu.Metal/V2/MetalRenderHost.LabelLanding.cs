// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Why held label copies landed, and what they were, for the WAITS line: the counts that say
/// which landings the frame could do without and which barrier each one paid.
/// </summary>
/// <remarks>
/// label_land     what landed the held copies: the commit, a storage binding that reads or writes them, a planned or
///                unplanned device-address range, indirect arguments, a persistent-thread kernel, a staged gather, a
///                join, a transfer, a copy that could not wait, a compute encoder opening, the image cache, other.
/// label_batch    copies per landing.
/// label_barrier  what ordered the landing: a shared event, a device-only event, the encoder fence alone, or a
///                dispatch inside the open compute encoder.
/// label_mix      the landed copies: command-processor writes, causal ones, CPU-visible and device-only end-of-pipe
///                labels, and the rest (GDS stores, occlusion dumps).
/// label_force_by the first held copy a forcing range overlapped, and label_force_range that range's size.
/// label_in_encoder_refused why a landing with a compute encoder open did not fold into it.
/// </remarks>
internal sealed partial class MetalRenderHost
{
    internal enum LabelLandCause : byte
    {
        None,
        Commit,
        BindRead,
        BindWritten,
        DaPlanned,
        DaUnplanned,
        Indirect,
        Persistent,
        Gather,
        Join,
        Transfer,
        CannotWait,
        EnsureCompute,
        Image,
        Other,
    }

    // What a held copy is: it decides whether the copy needs the shared event (CpuVisible) and is counted by label_mix.
    private enum LabelCopyKind : byte
    {
        Other,
        CpWrite,
        CpCausal,
        EndOfPipe,
    }

    private static readonly string[] LabelLandNames =
        ["commit", "bind_read", "bind_written", "da_planned", "da_unplanned", "indirect", "persistent", "gather", "join",
         "transfer", "cannot_wait", "ensure_compute", "image", "other"];

    private static readonly string[] LabelBatchNames = ["1", "2-5", "6-15", "16+"];
    private static readonly string[] LabelBarrierNames = ["shared", "device", "fence_only", "in_encoder"];
    private static readonly string[] LabelMixNames = ["cp", "cp_causal", "eop_cpu", "eop_dev", "other"];
    private static readonly string[] LabelForceByNames = ["cp", "eop_cpu", "eop_dev", "other"];
    private static readonly string[] LabelForceRangeNames = ["<64K", "<16M", "<1G", ">=1G"];
    private static readonly string[] LabelRefusedNames = ["occlusion", "scatter", "no_fence", "staged"];

    private const int BarrierKindShared = 0, BarrierKindDevice = 1, BarrierKindFenceOnly = 2, BarrierKindInEncoder = 3;

    private readonly long[] _labelLand = new long[LabelLandNames.Length];
    private readonly long[] _labelBatch = new long[LabelBatchNames.Length];
    private readonly long[] _labelBarrierKinds = new long[LabelBarrierNames.Length];
    private readonly long[] _labelMix = new long[LabelMixNames.Length];
    private readonly long[] _labelForceBy = new long[LabelForceByNames.Length];
    private readonly long[] _labelForceRange = new long[LabelForceRangeNames.Length];
    private readonly long[] _labelInEncoderRefused = new long[LabelRefusedNames.Length];

    // What asked for the held copies to land next (set with _landDeferredLabels), None for a plain boundary.
    private LabelLandCause _labelLandCause;

    // The label_force_by class of the copy the last DeferredLabelCopiesTouch(address, size) found, -1 for none.
    private int _labelTouchClass = -1;

    // The landing cause a site asks for with _landDeferredLabels; the first one asked keeps it until the copies land.
    private void RequestLabelLanding(LabelLandCause cause)
    {
        _landDeferredLabels = true;
        if (_labelLandCause == LabelLandCause.None)
        {
            _labelLandCause = cause;
        }
    }

    private static int LabelMixIndex(in DeferredLabelCopy copy) => copy.Kind switch
    {
        LabelCopyKind.CpWrite => 0,
        LabelCopyKind.CpCausal => 1,
        LabelCopyKind.EndOfPipe => copy.CpuVisible ? 2 : 3,
        _ => 4,
    };

    private static int LabelForceIndex(in DeferredLabelCopy copy) => copy.Kind switch
    {
        LabelCopyKind.CpWrite or LabelCopyKind.CpCausal => 0,
        LabelCopyKind.EndOfPipe => copy.CpuVisible ? 1 : 2,
        _ => 3,
    };

    // A range that forced the held copies to land: the first copy it overlapped, and how wide it was.
    private void NoteLabelForce(ulong size)
    {
        if (_labelTouchClass >= 0)
        {
            _labelForceBy[_labelTouchClass]++;
        }

        _labelForceRange[size switch
        {
            < 64UL << 10 => 0,
            < 16UL << 20 => 1,
            < 1UL << 30 => 2,
            _ => 3,
        }]++;
    }

    // One landing of held copies: why, how many, and what they were.
    private void NoteLabelLanding(ReadOnlySpan<DeferredLabelCopy> copies, LabelLandCause cause)
    {
        _labelLand[(int)(cause == LabelLandCause.None ? LabelLandCause.Other : cause) - 1]++;
        _labelBatch[copies.Length switch
        {
            1 => 0,
            <= 5 => 1,
            <= 15 => 2,
            _ => 3,
        }]++;
        foreach (ref readonly var copy in copies)
        {
            _labelMix[LabelMixIndex(in copy)]++;
        }
    }

    // SHARPEMU_METAL_V2_CP_WRITE_DEVICE_ORDER (on; =0 off): a command-processor write its queue made with nothing causal
    // before it (TakeCommandProcessorWriteCausal: no read of a value not yet landed, no flush, cache action or acquire
    // since its previous write) is held device-ordered rather than CPU-visible. On the console such a WRITE_DATA waits
    // for no earlier work and publishes nothing for the CPU, so it needs no shared event, and a batch of them lands
    // inside the open compute encoder. Every Demon's Souls batch held one and paid the shared event for it.
    // =verify keeps every write CPU-visible and counts (cp_write_order_suspects, logged once) the ones that would have
    // been device-ordered while some queue had read a pending value since the last shared barrier: a causal read site
    // the interpreter does not flag shows up there.
    private static readonly string? CpWriteDeviceOrderMode =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CP_WRITE_DEVICE_ORDER");

    internal bool CpWritesDeviceOrdered { get; set; } = CpWriteDeviceOrderMode is not ("0" or "verify");

    private static readonly bool CpWriteDeviceOrderVerify = CpWriteDeviceOrderMode == "verify";

    // A causal command-processor write is held: every later one is held CPU-visible behind it until the held copies
    // land, so none overtakes it and the order of the writes is kept across the landing.
    private bool _causalCpWriteHeld;
    private long _pendingReadsAtSharedBarrier;
    private long _cpWriteOrderSuspects;

    // Command-processor writes recorded as a dispatch in the open compute encoder rather than held (label_cp_in_encoder).
    private long _labelCpInEncoder;

    private (bool CpuVisible, LabelCopyKind Kind) CommandProcessorWriteClass(bool causal)
    {
        var ordered = causal || _causalCpWriteHeld;
        if (CpWriteDeviceOrderVerify && !ordered &&
            SharpEmu.Libs.Gpu.GpuCommands.GpuCommandInterpreter.PendingValueReads != _pendingReadsAtSharedBarrier)
        {
            _cpWriteOrderSuspects++;
            MetalV2Diagnostics.WarnOnce("cp-write-order-suspect",
                "A command-processor write its queue did not flag as causal followed a read of a pending value since the " +
                "last shared label barrier (SHARPEMU_METAL_V2_CP_WRITE_DEVICE_ORDER=verify).");
        }

        return (!CpWritesDeviceOrdered || ordered, ordered ? LabelCopyKind.CpCausal : LabelCopyKind.CpWrite);
    }

    // heldBefore: the held copies before the write was recorded; more now means it was held.
    private void NoteCommandProcessorWriteHeld(bool causal, int heldBefore)
    {
        if (causal && _deferredLabelCopies.Count > heldBefore)
        {
            _causalCpWriteHeld = true;
        }
    }

    private string TakeLabelLandingCounters()
    {
        var text = new System.Text.StringBuilder(256);
        text.Append(" cp_write_order_suspects=").Append(Interlocked.Exchange(ref _cpWriteOrderSuspects, 0));
        text.Append(" label_cp_in_encoder=").Append(Interlocked.Exchange(ref _labelCpInEncoder, 0));
        Append(text, "label_land", LabelLandNames, _labelLand);
        Append(text, "label_batch", LabelBatchNames, _labelBatch);
        Append(text, "label_barrier", LabelBarrierNames, _labelBarrierKinds);
        Append(text, "label_mix", LabelMixNames, _labelMix);
        Append(text, "label_force_by", LabelForceByNames, _labelForceBy);
        Append(text, "label_force_range", LabelForceRangeNames, _labelForceRange);
        Append(text, "label_in_encoder_refused", LabelRefusedNames, _labelInEncoderRefused);
        return text.ToString();

        static void Append(System.Text.StringBuilder text, string field, string[] names, long[] counts)
        {
            text.Append(' ').Append(field).Append("=[");
            for (var index = 0; index < names.Length; index++)
            {
                if (index != 0) text.Append(',');
                text.Append(names[index]).Append(':').Append(Interlocked.Exchange(ref counts[index], 0));
            }

            text.Append(']');
        }
    }

    /// <summary>Landings per cause since the last take, for tests; same order as <see cref="LabelLandCause"/> minus None.</summary>
    internal long[] TakeLabelLandCounts()
    {
        var counts = new long[_labelLand.Length];
        for (var index = 0; index < counts.Length; index++) counts[index] = Interlocked.Exchange(ref _labelLand[index], 0);
        return counts;
    }

    /// <summary>label_barrier since the last take: shared, device, fence_only, in_encoder.</summary>
    internal long[] TakeLabelBarrierKinds()
    {
        var counts = new long[_labelBarrierKinds.Length];
        for (var index = 0; index < counts.Length; index++) counts[index] = Interlocked.Exchange(ref _labelBarrierKinds[index], 0);
        return counts;
    }

    /// <summary>label_in_encoder_refused since the last take: occlusion, scatter, no_fence, staged.</summary>
    internal long[] TakeLabelInEncoderRefusals()
    {
        var counts = new long[_labelInEncoderRefused.Length];
        for (var index = 0; index < counts.Length; index++) counts[index] = Interlocked.Exchange(ref _labelInEncoderRefused[index], 0);
        return counts;
    }
}
