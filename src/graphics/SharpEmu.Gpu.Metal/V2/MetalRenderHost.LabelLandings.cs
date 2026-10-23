// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    internal bool LabelsInEncoder { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABEL_IN_ENCODER") != "0";

    private nint _labelCopyPipeline;
    internal long InEncoderLabelWrites { get; private set; }
    internal long LabelLandingBatches { get; private set; }
    internal long LabelLandingsCoalesced { get; private set; }
    private readonly Dictionary<string, long> _labelLandingRefusals = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _encoderEnds = new(StringComparer.Ordinal);
    private string? _encoderEndReason;
    private ulong _landingReportFrame;
    private long _reportedInEncoderWrites, _reportedLandingBatches, _reportedCoalesced;

    private void CountLabelLanding(int copies)
    {
        LabelLandingBatches++;
        LabelLandingsCoalesced += Math.Max(0, copies - 1);
    }

    private bool RefuseInEncoderLanding(string reason)
    {
        _labelLandingRefusals[reason] = _labelLandingRefusals.GetValueOrDefault(reason) + 1;
        return false;
    }

    // EOP labels can be polled directly by guest CPU threads, without any host read hook.
    // A compute memory barrier cannot publish earlier stores to those threads. Only CP
    // copies with immutable staged sources are eligible; older EOP/host publications and
    // staged writebacks keep the original shared-event + blit ordering.
    private bool TryLandCopiesInCompute()
    {
        if (!LabelsInEncoder) return RefuseInEncoderLanding("disabled");
        if (_computeEncoder == 0) return RefuseInEncoderLanding("no-compute");
        if (_heldCopiesRequireCpuBarrier) return RefuseInEncoderLanding("cpu-observation");
        if (_stagedStorageCopies is { Count: > 0 }) return RefuseInEncoderLanding("staged-stores");
        if (Volatile.Read(ref _owedHostLabels) != 0) return RefuseInEncoderLanding("host-label");

        var copies = CollectionsMarshal.AsSpan(_deferredLabelCopies);
        var pipeline = LabelCopyPipeline();
        var table = _guestMemory.Allocate((ulong)(sizeof(ulong) + copies.Length * 3 * sizeof(ulong)), 256, out var bytes);
        if (!table.IsValid) return RefuseInEncoderLanding("staging");
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, (ulong)copies.Length);
        for (var index = 0; index < copies.Length; index++)
        {
            ref readonly var copy = ref copies[index];
            var at = bytes[(sizeof(ulong) + index * 3 * sizeof(ulong))..];
            BinaryPrimitives.WriteUInt64LittleEndian(at,
                MetalNative.SendGpuAddress(copy.StagingBuffer, SelGpuAddress) + copy.StagingOffset);
            BinaryPrimitives.WriteUInt64LittleEndian(at[8..],
                MetalNative.SendGpuAddress(copy.DestinationBuffer, SelGpuAddress) + copy.DestinationOffset);
            BinaryPrimitives.WriteUInt64LittleEndian(at[16..], (ulong)copy.Bytes);
        }

        // Flush retained guest dispatches before changing pipeline/bindings. Concurrent
        // dispatches require barriers on both sides, including WAW and aliased buffers.
        ComputeScopeBarrier();
        foreach (ref readonly var copy in copies)
        {
            MetalNative.SendUseResourceUsage(_computeEncoder, SelUseResourceUsage, copy.StagingBuffer, Mtl.ResourceUsageRead);
            MetalNative.SendUseResourceUsage(_computeEncoder, SelUseResourceUsage, copy.DestinationBuffer, Mtl.ResourceUsageWrite);
        }
        MetalNative.SendVoid(_computeEncoder, SelSetComputePipelineState, pipeline);
        MetalNative.SendSetBuffer(_computeEncoder, SelSetComputeBuffer, table.Buffer, (nuint)table.Offset, 0);
        var one = new MtlSize { Width = 1, Height = 1, Depth = 1 };
        MetalNative.SendDispatch(_computeEncoder, SelDispatchThreadgroups, one, one);
        ComputeScopeBarrier();
        _boundComputePipeline = 0;
        BindStageBuffers(_computeEncoder, StageCompute, SelSetComputeBuffer);
        InEncoderLabelWrites += copies.Length;
        CountLabelLanding(copies.Length);
        if (PassProfileEnabled) PassProfileNoteWork($"cp-copies n={copies.Length}", "label");
        ClearHeldLabelCopies();
        return true;
    }

    // One thread applies the entire run in record order. Parallel stores would race for
    // overlapping WRITE_DATA packets, including an unaligned partial overwrite of a qword.
    private const string LabelCopySource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void sharpemu_cp_copies(constant ulong* table [[buffer(0)]]) {
            for (ulong i = 0; i < table[0]; ++i) {
                const ulong at = 1 + i * 3;
                const ulong src = table[at], dst = table[at + 1], bytes = table[at + 2];
                ulong b = 0;
                // Keep naturally aligned label words whole, and avoid a byte loop for fills.
                if (((src | dst) & 7) == 0) {
                    for (; b + 8 <= bytes; b += 8)
                        reinterpret_cast<device ulong*>(dst + b)[0] = reinterpret_cast<const device ulong*>(src + b)[0];
                }
                if (((src | dst) & 3) == 0) {
                    for (; b + 4 <= bytes; b += 4)
                        reinterpret_cast<device uint*>(dst + b)[0] = reinterpret_cast<const device uint*>(src + b)[0];
                }
                for (; b < bytes; ++b)
                    reinterpret_cast<device uchar*>(dst + b)[0] = reinterpret_cast<const device uchar*>(src + b)[0];
            }
        }
        """;

    private nint LabelCopyPipeline()
    {
        if (_labelCopyPipeline != 0) return _labelCopyPipeline;
        var options = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        nint library = 0, function = 0;
        try
        {
            MetalNative.ApplyMslCompileOptions(options);
            nint error = 0;
            library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithSource:options:error:"),
                MetalNative.NsString(LabelCopySource), options, ref error);
            if (library == 0) throw Fatal($"The CP copy kernel did not compile: {MetalNative.DescribeError(error)}.");
            function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("sharpemu_cp_copies"));
            if (function == 0) throw Fatal("The CP copy kernel is missing.");
            _labelCopyPipeline = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
            if (_labelCopyPipeline == 0) throw Fatal($"The CP copy pipeline did not build: {MetalNative.DescribeError(error)}.");
            return _labelCopyPipeline;
        }
        finally
        {
            if (function != 0) MetalNative.SendVoid(function, SelRelease);
            if (library != 0) MetalNative.SendVoid(library, SelRelease);
            MetalNative.SendVoid(options, SelRelease);
        }
    }

    private void NoteEncoderEnded(string kind)
    {
        var reason = kind + ":" + (_encoderEndReason ?? "other");
        _encoderEnds[reason] = _encoderEnds.GetValueOrDefault(reason) + 1;
        _encoderEndReason = null;
    }

    private string TakeLabelLandingCounters()
    {
        var frames = _bindingPresentId - _landingReportFrame;
        _landingReportFrame = _bindingPresentId;
        var writes = InEncoderLabelWrites - _reportedInEncoderWrites;
        var batches = LabelLandingBatches - _reportedLandingBatches;
        var coalesced = LabelLandingsCoalesced - _reportedCoalesced;
        _reportedInEncoderWrites = InEncoderLabelWrites;
        _reportedLandingBatches = LabelLandingBatches;
        _reportedCoalesced = LabelLandingsCoalesced;
        var ends = string.Join(',', _encoderEnds.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}"));
        var perFrame = frames == 0 ? "n/a" : string.Join(',', _encoderEnds.OrderBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => $"{p.Key}={((double)p.Value / frames).ToString("F2", CultureInfo.InvariantCulture)}"));
        var refused = string.Join(',', _labelLandingRefusals.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}"));
        _encoderEnds.Clear();
        _labelLandingRefusals.Clear();
        return $"landing_frames={frames} label_in_encoder={writes} label_landing_batches={batches} " +
            $"label_landings_coalesced={coalesced} label_landing_refused=[{refused}] encoder_ends=[{ends}] encoder_ends_per_frame=[{perFrame}]";
    }
}
