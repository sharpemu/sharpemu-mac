// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// GET_LOD_STATS measured on the GPU (sceAgcDcbGetLodStats). A pixel shader sampling a T#
/// with MIP_STATS_ENABLE records the LOD it asked for into a resident counter bank
/// (Templates/mip_stats_prelude.msl); the packet becomes a one-threadgroup kernel, recorded
/// where the packet sits in the stream, that packs the bank into the guest's MipStatsBuffer
/// through its window and clears the bank for the next window. A label the guest writes
/// after the packet lands after the kernel, as the work before it does.
/// </summary>
/// <remarks>
/// This is the hardware's behaviour, so it is the default: a texture streamer (Demon's
/// Souls) keeps only the mips the screen asks for. The bound-slot fallback answers "mip 0"
/// for every texture a draw bound, and the streamer then raises every one to full detail.
/// SHARPEMU_LOD_STATS=fallback goes back to it.
/// </remarks>
internal sealed partial class MetalRenderHost
{
    internal static readonly bool GpuLodStatistics =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOD_STATS"), "fallback", StringComparison.Ordinal);

    /// <summary>
    /// A pixel pipeline carries the counters only once the title uses LOD statistics
    /// (MipStatistics.TitleUsesCounters): the sites sit behind a function constant
    /// (Gen5MslShader.MipStatisticsFunctionConstantName) that a pipeline made before is
    /// specialized false. SHARPEMU_LOD_STATS=always has them from the first draw, and
    /// =fallback never records, so neither splits pipelines.
    /// </summary>
    internal static readonly bool LodStatisticsFollowDemand =
        GpuLodStatistics && Gen5MslCodegenOptions.FromEnvironment.MipStatisticsFunctionConstant;

    public bool PixelPipelinesFollowLodStatistics => LodStatisticsFollowDemand;

    // A pixel pipeline was made with the counters compiled out. The first report after that is
    // answered from the bound-slot fallback: the bank may lack the samples of the draws
    // through it. Any thread.
    private bool _lodStatsCompiledOut;
    private bool _lodStatsPrimed;

    /// <summary>
    /// The function of a pixel (or any) library entry that declares the LOD statistics function
    /// constant, specialized with <paramref name="enabled"/>; the caller owns it, or zero.
    /// </summary>
    internal static unsafe nint NewMipStatisticsFunction(nint library, string entry, bool enabled, out nint error)
    {
        error = 0;
        var specializedName = entry + (enabled ? "_sharpemu_mip_on" : "_sharpemu_mip_off");
        // Binary-archive AIR is specialized. The plain-key library contains both named
        // values, so loading one must select that value rather than re-specialize baked AIR.
        var stored = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(specializedName));
        if (stored != 0)
        {
            return stored;
        }

        var values = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLFunctionConstantValues"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        try
        {
            var value = enabled ? (byte)1 : (byte)0;
            MetalNative.SendSetConstantValue(
                values, MetalNative.Selector("setConstantValue:type:atIndex:"), (nint)(&value), MetalDataTypeBool,
                Gen5MslShader.MipStatisticsFunctionConstantIndex);
            if (MetalBinaryArchive.PlainPixelKey)
            {
                var descriptor = MetalNative.Send(
                    MetalNative.Send(MetalNative.Class("MTLFunctionDescriptor"), MetalNative.Selector("alloc")),
                    MetalNative.Selector("init"));
                try
                {
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setName:"), MetalNative.NsString(entry));
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setSpecializedName:"), MetalNative.NsString(specializedName));
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setConstantValues:"), values);
                    return MetalNative.Send(library, MetalNative.Selector("newFunctionWithDescriptor:error:"), descriptor, ref error);
                }
                finally
                {
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("release"));
                }
            }

            return MetalNative.Send(
                library,
                MetalNative.Selector("newFunctionWithName:constantValues:error:"),
                MetalNative.NsString(entry),
                values,
                ref error);
        }
        finally
        {
            MetalNative.SendVoid(values, MetalNative.Selector("release"));
        }
    }

    private const nuint MetalDataTypeBool = 53;

    // After the bank: the slots the last report said were sampled, those whose finest mip
    // was coarser than 0, and the reports packed ([MEM] lod).
    private const int LodStatsTallyWords = 4;
    private const int LodStatsThreads = MipStatistics.SlotCount;
    private const string LodStatsFunction = "sharpemu_lod_stats_pack";

    private static readonly int LodStatsBankWords = (int)(MipStatistics.CounterBufferBytes / sizeof(uint));
    private static readonly nint SelLodStatsComputeEncoder = MetalNative.Selector("computeCommandEncoder");
    private static readonly nint SelLodStatsSetPipeline = MetalNative.Selector("setComputePipelineState:");
    private static readonly nint SelLodStatsSetBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
    private static readonly nint SelLodStatsSetBytes = MetalNative.Selector("setBytes:length:atIndex:");
    private static readonly nint SelLodStatsDispatch = MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:");
    private static readonly nint SelLodStatsEndEncoding = MetalNative.Selector("endEncoding");

    // One thread per slot. Each reads and, when asked, clears its own three words, so
    // nothing is shared but the tally; the header, the label the guest may poll, is
    // stored last, after a device barrier over the slots.
    internal const string LodStatsPackSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct LodStatsArguments
        {
            uint reset;
            uint pad;
        };

        constant uint lod_stats_entries = 257u;
        constant uint lod_stats_first_word = 16u;
        constant uint lod_stats_max_samples = 0xFFFFFFu;

        kernel void sharpemu_lod_stats_pack(
            device atomic_uint* bank [[buffer(0)]],
            device uint* report [[buffer(1)]],
            constant LodStatsArguments& args [[buffer(2)]],
            uint slot [[thread_position_in_threadgroup]])
        {
            threadgroup atomic_uint sampled;
            threadgroup atomic_uint coarse;
            if (slot == 0u)
            {
                atomic_store_explicit(&sampled, 0u, memory_order_relaxed);
                atomic_store_explicit(&coarse, 0u, memory_order_relaxed);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint finest = atomic_load_explicit(bank + slot, memory_order_relaxed);
            uint clamped = min(atomic_load_explicit(bank + lod_stats_entries + slot, memory_order_relaxed), lod_stats_max_samples);
            uint unclamped = min(atomic_load_explicit(bank + 2u * lod_stats_entries + slot, memory_order_relaxed), lod_stats_max_samples);
            if (args.reset != 0u)
            {
                atomic_store_explicit(bank + slot, 0xFFFFFFFFu, memory_order_relaxed);
                atomic_store_explicit(bank + lod_stats_entries + slot, 0u, memory_order_relaxed);
                atomic_store_explicit(bank + 2u * lod_stats_entries + slot, 0u, memory_order_relaxed);
            }

            uint level = finest == 0xFFFFFFFFu ? 15u : min(finest, 14u);
            report[lod_stats_first_word + slot * 2u] = clamped | (slot << 24);
            report[lod_stats_first_word + slot * 2u + 1u] = unclamped | (level << 24);
            if (slot > 0u && slot < lod_stats_first_word)
            {
                report[slot] = 0u;
            }

            if (level != 15u)
            {
                atomic_fetch_add_explicit(&sampled, 1u, memory_order_relaxed);
                if (level != 0u)
                {
                    atomic_fetch_add_explicit(&coarse, 1u, memory_order_relaxed);
                }
            }

            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (slot == 0u)
            {
                report[0] = lod_stats_first_word * 4u;
                device atomic_uint* tally = bank + 3u * lod_stats_entries;
                atomic_store_explicit(tally, atomic_load_explicit(&sampled, memory_order_relaxed), memory_order_relaxed);
                atomic_store_explicit(tally + 1, atomic_load_explicit(&coarse, memory_order_relaxed), memory_order_relaxed);
                atomic_fetch_add_explicit(tally + 2, 1u, memory_order_relaxed);
            }
        }
        """;

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct LodStatsArguments
    {
        public uint Reset;
        public uint Pad;
    }

    private nint _lodStatsBuffer;
    private nint _lodStatsContents;
    private ulong _lodStatsGpuAddress;
    private nint _lodStatsPipeline;
    private long _lodStatsReports;

    /// <summary>Pixel programs carry the recording at every sampled-image site, guarded at
    /// run time by the bound T#'s MIP_STATS_ENABLE, unless SHARPEMU_LOD_STATS=fallback.
    /// The shader prewarm sets it too, before it translates anything: a pixel program
    /// translated without it is MSL the game never asks for. Set only at command-stream
    /// creation, it was still off while the prewarm translated the graphics list, and 9 of
    /// 343 recorded Demon's Souls pixel libraries matched what the game compiled
    /// (run 20261018-203322-ds-ds-memopt-check; 725 of 769 with it on).</summary>
    internal static void EnableLodStatistics() => ResourceMaterializer.RecordMipStatistics = GpuLodStatistics;

    /// <summary>
    /// Portable buffers (PortableBufferWord): every program reads its buffers' strides and
    /// read formats at run time, so a translation does not depend on the draw's vertex
    /// layout and a shader precompile can make the library a draw asks for.
    /// SHARPEMU_METAL_V2_PORTABLE_BUFFERS=0 bakes them in again.
    /// </summary>
    internal static readonly bool PortableBuffers =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PORTABLE_BUFFERS") != "0";

    /// <summary>
    /// The process-wide translation switches, set by the command stream and by the shader
    /// prewarm before either translates anything, so both make the same MSL.
    /// </summary>
    internal static void ApplyTranslationSwitches()
    {
        EnableLodStatistics();
        PortableBufferWord.Enabled = PortableBuffers;
    }

    /// <summary>GET_LOD_STATS reports this host packed on the GPU.</summary>
    internal long LodStatisticsReports => Interlocked.Read(ref _lodStatsReports);

    /// <summary>The counter bank and its tally words as the CPU sees them (shared storage).</summary>
    internal unsafe Span<uint> LodStatisticsBank
    {
        get
        {
            EnsureLodStatisticsBank();
            return new Span<uint>((void*)_lodStatsContents, LodStatsBankWords + LodStatsTallyWords);
        }
    }

    /// <summary>The bank's GPU address for a program's mip_stats field.</summary>
    private ulong LodStatisticsAddress()
    {
        EnsureLodStatisticsBank();
        return _lodStatsGpuAddress;
    }

    private unsafe void EnsureLodStatisticsBank()
    {
        if (_lodStatsBuffer != 0)
        {
            return;
        }

        var words = LodStatsBankWords + LodStatsTallyWords;
        _lodStatsBuffer = _guestMemory.CreateResidentSharedBuffer(
            (ulong)words * sizeof(uint), "SharpEmu LodStatistics", out _lodStatsContents, out _lodStatsGpuAddress);
        if (_lodStatsBuffer == 0 || _lodStatsContents == 0)
        {
            throw Fatal("The LOD statistics counter bank could not be created.");
        }

        var bank = new Span<uint>((void*)_lodStatsContents, words);
        bank[..(int)MipStatistics.CounterWords].Fill(uint.MaxValue);
        bank[(int)MipStatistics.CounterWords..].Clear();
    }

    bool ICommandStreamHost.TryRecordLodStatistics(ulong destination, uint size, bool resetCounters)
    {
        // A buffer too small for the slots gets the fallback's "no counters" header, and a
        // host that publishes labels as it parses them would publish the guest's fence
        // before the GPU wrote the report.
        if (!ResourceMaterializer.RecordMipStatistics || size < MipStatistics.ReportBytes || !DefersEndOfPipeLabels)
        {
            return false;
        }

        if (!_lodStatsPrimed)
        {
            _lodStatsPrimed = true;
            if (Volatile.Read(ref _lodStatsCompiledOut))
            {
                return false;
            }
        }

        const ulong reportBytes = MipStatistics.ReportBytes;
        var target = _guestMemory.ResolveWritable(destination, reportBytes);
        if (!target.IsValid)
        {
            MetalV2Diagnostics.WarnOnce("lod-stats-without-window",
                $"A GET_LOD_STATS buffer has no window, so it gets the bound-slot fallback: address=0x{destination:X16}.");
            return false;
        }

        EnsureLodStatisticsBank();
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
        EncodeLodStatisticsPack(context.CommandBuffer, in target, resetCounters);
        context.ProfileEncoded("lod stats");
        var serial = context.SubmissionSerial;
        Interlocked.Increment(ref _lodStatsReports);
        // The draws' bound-slot count ([MEM] lod max_slots_bound) starts its window here as well.
        MipStatistics.Shared.ClearBoundSlots();

        _imageCache.InvalidateMemoryFromGpu(destination, reportBytes);
        NoteGuestMemoryWrite(destination, reportBytes, MetalGpuWriteOrigin.Copy);
        // A command processor polling the header suspends until the submission that
        // writes it completes; the worker commits what it has before it parks.
        _pendingLabelWrites++;
        if (_labelWakeSerial != serial)
        {
            _labelWakeSerial = serial;
            _completions.Enqueue(serial, () => _commandStream?.NotifyCompletedGpuTick(serial));
        }

        if (LodStatsTelemetry.Enabled)
        {
            _completions.Enqueue(serial, NoteLodStatisticsTally);
        }

        return true;
    }

    private unsafe void EncodeLodStatisticsPack(nint commandBuffer, in MetalGuestRange target, bool reset)
    {
        var encoder = MetalNative.Send(commandBuffer, SelLodStatsComputeEncoder);
        MetalMemoryTelemetry.ComputeEncoders.Add();
        if (encoder == 0)
        {
            throw Fatal("A Metal compute encoder could not be created for a GET_LOD_STATS report.");
        }

        try
        {
            var arguments = new LodStatsArguments { Reset = reset ? 1u : 0u };
            MetalNative.SendVoid(encoder, SelLodStatsSetPipeline, LodStatisticsPipeline());
            MetalNative.SendSetBuffer(encoder, SelLodStatsSetBuffer, _lodStatsBuffer, 0, 0);
            MetalNative.SendSetBuffer(encoder, SelLodStatsSetBuffer, target.Buffer, (nuint)target.Offset, 1);
            MetalNative.SendSetBytes(encoder, SelLodStatsSetBytes, (nint)(&arguments), (nuint)sizeof(LodStatsArguments), 2);
            MetalNative.SendDispatch(encoder, SelLodStatsDispatch,
                new MtlSize { Width = 1, Height = 1, Depth = 1 },
                new MtlSize { Width = LodStatsThreads, Height = 1, Depth = 1 });
        }
        finally
        {
            MetalNative.SendVoid(encoder, SelLodStatsEndEncoding);
        }
    }

    // Read when a report's submission has completed; a later report may already have
    // overwritten the words, which a ten-second telemetry line does not mind.
    private unsafe void NoteLodStatisticsTally()
    {
        if (_lodStatsContents == 0)
        {
            return;
        }

        var tally = new Span<uint>((void*)_lodStatsContents, LodStatsBankWords + LodStatsTallyWords)[LodStatsBankWords..];
        LodStatsTelemetry.NoteReportSlots((int)Volatile.Read(ref tally[0]), (int)Volatile.Read(ref tally[1]));
    }

    private nint LodStatisticsPipeline()
    {
        if (_lodStatsPipeline != 0)
        {
            return _lodStatsPipeline;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var options = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
            MetalNative.ApplyMslCompileOptions(options);
            nint error = 0;
            var library = MetalNative.Send(
                _device, MetalNative.Selector("newLibraryWithSource:options:error:"), MetalNative.NsString(LodStatsPackSource), options, ref error);
            MetalNative.SendVoid(options, MetalNative.Selector("release"));
            if (library == 0)
            {
                throw Fatal($"The GET_LOD_STATS pack kernel did not compile: {MetalNative.DescribeError(error)}.");
            }

            var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(LodStatsFunction));
            MetalNative.SendVoid(library, MetalNative.Selector("release"));
            if (function == 0)
            {
                throw Fatal("The GET_LOD_STATS pack function is missing.");
            }

            error = 0;
            var pipeline = MetalNative.Send(
                _device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
            MetalNative.SendVoid(function, MetalNative.Selector("release"));
            if (pipeline == 0)
            {
                throw Fatal($"The GET_LOD_STATS pack pipeline did not build: {MetalNative.DescribeError(error)}.");
            }

            _lodStatsPipeline = pipeline;
            return pipeline;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }
}
