// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.Gpu.Metal.V2;

// One-shot warnings and the environment switches of the v2 backend. Anything v2
// cannot do exactly is reported once by key, never per draw and never fatally.
internal static class MetalV2Diagnostics
{
    private static readonly ConcurrentDictionary<string, byte> _reported = new(StringComparer.Ordinal);

    public static readonly bool Verbose =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRACE"), "1", StringComparison.Ordinal);

    // Naming a range that has to be staged costs two extra backing lookups, so it is
    // on only while someone is hunting ring fallbacks. Defaults on under the trace
    // switch; SHARPEMU_METAL_V2_REPORT_UNBACKED=1 turns it on by itself.
    public static readonly bool ReportUnbackedRanges = Verbose ||
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_REPORT_UNBACKED"), "1", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_IGNORE_TESTS=1 drops face culling, the depth compare and the
    // stencil test from every draw. A pass whose target never changes either had its
    // fragments thrown away by one of those three or never had a fragment to throw
    // away, and nothing in a target dump tells the two apart. A comma-separated list of
    // depth, stencil, cull and bounds drops only those, to tell which one it was.
    private static readonly string[] IgnoredTests =
        (Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IGNORE_TESTS") ?? "").Split(',', StringSplitOptions.TrimEntries);

    public static readonly bool IgnoreFragmentTests = IgnoredTests.Contains("1");
    public static readonly bool IgnoreDepthTest = IgnoreFragmentTests || IgnoredTests.Contains("depth");
    public static readonly bool IgnoreStencilTest = IgnoreFragmentTests || IgnoredTests.Contains("stencil");
    public static readonly bool IgnoreCulling = IgnoreFragmentTests || IgnoredTests.Contains("cull");
    public static readonly bool IgnoreDepthBounds = IgnoreFragmentTests || IgnoredTests.Contains("bounds");

    // equal: a depth compare of EQUAL passes ALWAYS instead - a pass drawn against a prepass's depth
    // that its own positions do not reproduce bit for bit loses its fragments to EQUAL.
    public static readonly bool LoosenDepthEqual = IgnoredTests.Contains("equal");

    // SHARPEMU_METAL_V2_LOOSEN_EQUAL_VS=<vertex hash>[,...]: depth EQUAL passes ALWAYS only for draws with these vertex
    // programs, for finding the draws whose positions miss their depth prepass.
    private static readonly HashSet<ulong> LoosenEqualVertexHashes = ParseHashes(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LOOSEN_EQUAL_VS"));

    // SHARPEMU_METAL_V2_DEPTH_ALWAYS_VS=<vertex hash>[,...]: draws with these vertex programs pass the depth test
    // whatever their compare (writes unchanged), to see the whole of geometry the scene's depth hides.
    private static readonly HashSet<ulong> DepthAlwaysVertexHashes = ParseHashes(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEPTH_ALWAYS_VS"));

    public static bool PassesDepthFor(ulong vertexHash) =>
        DepthAlwaysVertexHashes.Count != 0 && DepthAlwaysVertexHashes.Contains(vertexHash);

    // SHARPEMU_METAL_V2_FORCE_INDIRECT_INSTANCES=<vertex hash>:<instances>[,...]: indexed indirect draws with these vertex
    // programs run that many instances whatever their arguments' count, to see what a zero count leaves out.
    private static readonly Dictionary<ulong, uint> ForcedIndirectInstanceCounts = ParseForcedInstances(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FORCE_INDIRECT_INSTANCES"));

    public static uint? ForcedIndirectInstances(ulong vertexHash) =>
        ForcedIndirectInstanceCounts.Count != 0 && ForcedIndirectInstanceCounts.TryGetValue(vertexHash, out var count) ? count : null;

    private static Dictionary<ulong, uint> ParseForcedInstances(string? text)
    {
        var counts = new Dictionary<ulong, uint>();
        foreach (var part in (text ?? string.Empty).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var pieces = part.Split(':');
            if (pieces.Length == 2 &&
                ulong.TryParse(pieces[0].Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                    System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var hash) &&
                uint.TryParse(pieces[1], out var count))
            {
                counts[hash] = count;
            }
        }

        return counts;
    }

    // SHARPEMU_METAL_V2_LOOSEN_EQUAL_TO=ge|le: what a loosened EQUAL becomes (default ALWAYS). GREATER-OR-EQUAL
    // or LESS-OR-EQUAL keeps the prepass depth occluding everything else, and which of the two lets a
    // draw's fragments through says on which side of its prepass depth its own positions land.
    public static readonly SharpEmu.Libs.Gpu.Abstractions.GpuCompareOp LoosenEqualTo =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LOOSEN_EQUAL_TO") switch
        {
            "ge" => SharpEmu.Libs.Gpu.Abstractions.GpuCompareOp.GreaterOrEqual,
            "le" => SharpEmu.Libs.Gpu.Abstractions.GpuCompareOp.LessOrEqual,
            _ => SharpEmu.Libs.Gpu.Abstractions.GpuCompareOp.Always,
        };

    // SHARPEMU_METAL_V2_LOOSEN_EQUAL_WRITE=1: a loosened EQUAL also writes depth, so a snapshot after the pass holds
    // the depth the draw's own positions give, to set beside the prepass depth it was meant to match.
    public static readonly bool LoosenEqualWrites = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LOOSEN_EQUAL_WRITE") == "1";

    // SHARPEMU_METAL_V2_LOG_STENCIL_VS=<vertex hash>[,...] (or "all"): each depth/stencil state made for draws with
    // these vertex programs is logged once with its stencil compare, operations, masks and reference.
    private static readonly string? LogStencilVs = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LOG_STENCIL_VS");
    private static readonly HashSet<ulong> LogStencilVertexHashes = ParseHashes(LogStencilVs);

    public static bool LogsStencilFor(ulong vertexHash) =>
        LogStencilVs == "all" || (LogStencilVertexHashes.Count != 0 && LogStencilVertexHashes.Contains(vertexHash));

    public static bool LoosensEqualFor(ulong vertexHash) =>
        LoosenEqualVertexHashes.Count != 0 && LoosenEqualVertexHashes.Contains(vertexHash);

    private static HashSet<ulong> ParseHashes(string? text)
    {
        var hashes = new HashSet<ulong>();
        foreach (var part in (text ?? string.Empty).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var digits = part.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ? part[2..] : part;
            if (ulong.TryParse(digits, System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var hash))
            {
                hashes.Add(hash);
            }
        }

        return hashes;
    }

    // a2c: alpha to coverage stays off - an MRT0 alpha that reads 0 drops every sample of a draw
    // that never meant its alpha as coverage.
    public static readonly bool IgnoreAlphaToCoverage = IgnoredTests.Contains("a2c");

    // SHARPEMU_METAL_V2_DRAIN_CALLERS=1 counts which host entry points drain the deferred draws, in the
    // PERF line's prefetch block (a drain inside a draining draw counts too).
    public static readonly bool CountDrainCallers =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DRAIN_CALLERS") == "1";

    // SHARPEMU_METAL_V2_DEFER_BARRIERS=1 (experiment): a barrier queues behind the deferred draws instead
    // of draining them, and an operand or poll read nothing queued can produce leaves them queued
    // (MetalRenderHost.TryDeferBehindDraws, QueuedWorkLeaves). It cuts the render thread's waits on the
    // program resolver by about a third in Grand Theft Auto V's street, but its frame rate did not
    // move (gdab A/B: 14.74/15.25 against 14.95/15.32 median/mean fps).
    public static readonly bool DeferBarriers =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEFER_BARRIERS") == "1";

    // SHARPEMU_METAL_V2_DEFER_EOP=1 queues end-of-pipe writes behind the deferred draws too. It halves
    // the render thread's waits on the program resolver but lowered GTA V's frame rate (gdb-4/5): the
    // labels a title paces its frames on land later.
    public static readonly bool DeferEndOfPipe =
        DeferBarriers && Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEFER_EOP") == "1";

    // SHARPEMU_METAL_V2_REPORT_VERTEX_LAYOUT=1 keeps the vertex descriptor each
    // pipeline was built with, for the per-draw report: a stride or an attribute format
    // that does not match what the program fetches is invisible in everything else.
    public static readonly bool ReportVertexLayout =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_REPORT_VERTEX_LAYOUT"), "1", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_DEFER_STORE=1 lets a pass wait until it knows what follows it
    // before naming its store actions, so a store the next operation clears outright is
    // discarded. Off by default on the measurement: across Octopath and Sarah not one
    // store was ever proven dead — no pass in either ends because a clear arrived, and
    // the clears that do happen are materialising a surface about to be sampled rather
    // than one just rendered into. Asking Metal for MTLStoreActionUnknown changes what
    // the driver can decide before the pass runs, and that is not worth paying for a
    // saving neither title has.
    public static readonly bool DeferStoreActions =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEFER_STORE"), "1", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_DEFER_LABELS=0 makes an end-of-pipe label's bytes end the open
    // render pass to be copied, as they did before. A label is four or eight bytes and
    // a pass end on a tile-based GPU costs the whole framebuffer twice, so the copy
    // waits for a boundary that was going to happen anyway — but it also means the
    // guest sees the label slightly later, so the old behaviour stays reachable.
    public static readonly bool DeferLabelWrites =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEFER_LABELS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_DEFER_COMPUTE_COPIES=0 makes an end-of-pipe label or a
    // command-processor write recorded while a compute encoder is open end that encoder to
    // be copied, as it did before. Games put a label or a WRITE_DATA after nearly every
    // dispatch, and ending the encoder for each one split a frame into a hundred compute
    // encoders whose boundaries cost more GPU time than the dispatches they held.
    public static readonly bool DeferComputeCopies =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEFER_COMPUTE_COPIES"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_TRANSFER_IN_ENCODER=0 makes an image upload or writeback recorded
    // while a compute encoder is open end that encoder, as it did before. A transfer the
    // one-pass tiler encodes is nothing but compute dispatches, so it is recorded into the
    // open compute encoder instead; the encoder boundary it saves was a GPU idle gap of
    // 0.5-4 ms per transfer at the Demon's Souls title screen.
    public static readonly bool TransferInEncoder =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRANSFER_IN_ENCODER"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_DEFER_FRAGMENT_WRITE_PASS_END=0 ends the render pass after every draw
    // whose fragment stage stores to memory, as it did before. Nothing in the same pass reads
    // those stores until the guest flushes or acquires, so the pass now stays open until a
    // dispatch, an indirect draw or such a guest sync packet; ending it per draw reloaded and
    // stored the whole depth target 36 times a frame in Demon's Souls.
    public static readonly bool DeferFragmentWritePassEnd =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEFER_FRAGMENT_WRITE_PASS_END"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_SKIP_SERIAL_COMPUTE_BARRIERS=0 records a memory barrier on the open
    // compute encoder again for every barrier the executor asks for (two per dispatch). The
    // encoder is serial, and Metal documents a barrier there as "allowed, but ignored"
    // (MTLComputeCommandEncoder.h, memoryBarrierWithScope:): each dispatch completes, its
    // stores visible, before the next starts, for untracked buffers as for tracked textures.
    // Perf-audit bench 8b1/8b2: 2,700 dependent dispatches on untracked buffers read back
    // right without the barriers and take the same GPU time with them.
    public static readonly bool SkipSerialComputeBarriers =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SKIP_SERIAL_COMPUTE_BARRIERS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_ELIDE_GLOBAL_BARRIERS=0 records the guest acquire or release barrier in
    // an open render pass every time, as it did before. The executor already orders a draw's
    // buffer stores from the vertex stage with a barrier right after the draw, and ends the pass
    // over fragment stores before their consumer; attachments are ordered per fragment in draw
    // order. So the barrier is recorded only while the pass holds stores nothing ordered yet
    // (deferred fragment stores, or vertex stores through a device address or storage image).
    public static readonly bool ElideGlobalBarriers =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ELIDE_GLOBAL_BARRIERS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_KERNEL_FILLS=0 runs a title's own compute memset or memcpy over the
    // exact bytes of a cached image as the dispatch it is, with the writeback before it
    // and the upload after it, instead of as a clear or texture copy of the image
    // (MetalImageCache.KernelFills).
    public static readonly bool KernelFills =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_KERNEL_FILLS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_FOLD_CLEARS=0 encodes every clear as its own render pass the
    // moment it is asked for, the way it did before a clear could become the load action
    // of the pass that follows it. A fast-clear that is materialised and then rendered
    // into costs a 4K store and a 4K load for a result the load action gives away, which
    // is what the switch exists to be able to measure.
    // SHARPEMU_METAL_V2_COPY_READ_THROUGH=0 makes a host read of bytes a title's compute
    // memcpy is still writing wait for that submission, as before. Otherwise the read is
    // answered from the copy's source words when nothing later writes the range and the
    // source itself is not pending (MetalRenderHost.CopyReads).
    public static readonly bool CopyReadThrough =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COPY_READ_THROUGH"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_DATA_LOADS_DEVICE=0 keeps every data-only scalar load flattened into
    // the host-evaluated table, as before. Otherwise a load whose dword the GPU is still
    // writing when the draw is bound (a constant table copied on the GPU) makes that shader
    // be planned again with the load read on the device, instead of the host waiting for the
    // writer's submission on every bind.
    public static readonly bool DataLoadsOnDevice =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DATA_LOADS_DEVICE"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_OVERLAY_PAGE_FILTER=0 makes every guest read take the pending
    // command-processor write overlay's lock to learn that no pending write is on its pages,
    // as before the overlay kept its lock-free page counts.
    public static readonly bool OverlayPageFilter =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_OVERLAY_PAGE_FILTER"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_LABEL_OVERLAY=0 makes a resource materialization's read of an
    // end-of-pipe label wait for the GPU to land it again, as before the label overlay. By
    // default the value the label will hold is read as written (MetalRenderHost.LabelOverlay).
    public static readonly bool LabelOverlay =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABEL_OVERLAY"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_WAIT_TRACE=1 (or SHARPEMU_METAL_V2_EOP_TRACE=1) names, on each
    // SLOW-WAIT line, the shader whose resources were being materialized when the host waited.
    public static readonly bool WaitTrace =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WAIT_TRACE"), "1", StringComparison.Ordinal) ||
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EOP_TRACE"), "1", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_WAIT_STACK=1 adds the emulator frames that asked for the read to each SLOW-WAIT line.
    public static readonly bool WaitStack =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WAIT_STACK"), "1", StringComparison.Ordinal);

    public static readonly bool FoldClearsIntoPasses =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FOLD_CLEARS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_FLUSH_COALESCE=0 commits the command buffer at the end of every
    // guest submission that made progress, as it did before. Each commit leaves the GPU
    // idle until the next buffer is scheduled; on Demon's Souls' title screen ~17 commits
    // a frame were made only because a submission ended. The three limits below bound
    // how long a coalesced buffer stays open (submissions, encoders, microseconds).
    // SHARPEMU_METAL_V2_ELIDE_LOADS=0 loads every colour attachment a pass does not clear,
    // as before. Otherwise a pass begun for a draw that provably overwrites every pixel of
    // an attachment does not read it into tile memory first (MetalPassLoads).
    public static readonly bool ElideLoads =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ELIDE_LOADS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_DCC_CLAIM_INDEX=0 walks every DCC clear claim on every recorded GPU
    // store, as before, instead of skipping stores no claimed slice starts in
    // (MetalImageCache.Tracking.cs).
    public static readonly bool DccClaimPageIndex =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DCC_CLAIM_INDEX"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_JOIN_DEPTH_ONLY=0 opens a pass of its own for every depth-only draw,
    // as before. Otherwise a depth-only draw into the depth attachment of the open pass
    // stays in that pass, drawn with a pipeline that writes none of its colour attachments
    // (MetalRenderHost.DepthOnlyJoin.cs).
    public static readonly bool JoinDepthOnlyDraws =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_JOIN_DEPTH_ONLY"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_JOIN_SUBSETS=0 opens a pass of its own for a draw into some of the open
    // pass's colour attachments (MetalRenderHost.SubsetJoin.cs).
    public static readonly bool JoinColorSubsets =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_JOIN_SUBSETS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_DCC_CLAIM_FIRST=1 settles a DCC slice whose claim stands from the
    // claim alone (MetalImageCache.Tracking.cs) instead of querying pending writes and reading
    // the slice first. Off by default: with it on, SH renders its dark room and everything
    // after it as neon noise, and the verify mode below does not catch it.
    // SHARPEMU_METAL_V2_DCC_CLAIM_FIRST_VERIFY=1 (or SHARPEMU_PERF_VERIFY=1) takes the full
    // path for every slice and reports each standing claim it disagrees with.
    public static readonly bool DccClaimFirst =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DCC_CLAIM_FIRST"), "1", StringComparison.Ordinal);

    public static readonly bool DccClaimFirstVerify = PerfSwitches.Verify("SHARPEMU_METAL_V2_DCC_CLAIM_FIRST_VERIFY");

    // SHARPEMU_METAL_V2_DCC_RUN_MARKERS=0 scans every slice of a DCC run whose slices a settled
    // scan found claimed whole, instead of settling it from the marker (MetalImageCache.Tracking.cs).
    // Unlike the claim-first path above it still queries pending writes and reads each slice's
    // first byte, which is all the full path reads before a claimed repeat.
    public static readonly bool DccRunMarkers =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DCC_RUN_MARKERS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_LOSSY_TEXTURES=1 creates sampled-only textures with Apple GPU lossy
    // compression (about half the memory) where the device and the format allow it
    // (MetalImageCache.Lossy.cs). Off by default: with it on, SH rendered dark scenes as
    // pink/blue false colour, black specks and opaque lenses in 5 of 8 runs, and in none of 3
    // with it off.
    public static readonly bool LossyTextures =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LOSSY_TEXTURES"), "1", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_COMPRESSIBLE_TARGETS=0 creates every colour image with the
    // shader-write and pixel-format-view usages, as before. Otherwise an image first asked
    // for as a colour target is created with ShaderRead | RenderTarget only, and gains the
    // other two the first time something needs them (MetalImageCache.EnsureTextureUsage).
    // Apple GPUs keep a texture losslessly compressed only without those two usages:
    // measured on an M4, a 1920x1080 RGBA16F full-screen pass costs 2.2-3.4x as much
    // GPU time with either of them as without (243 vs 523-580 us).
    public static readonly bool CompressibleTargets =
        !string.Equals(
            Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COMPRESSIBLE_TARGETS"), "0", StringComparison.Ordinal);

    // SHARPEMU_METAL_V2_SKIP_OVERWRITTEN_REFRESH=0 uploads a CPU-dirty render target when it
    // is acquired, as before. Otherwise the upload waits for the pass to begin and is
    // dropped when that pass clears the whole image or its first draw overwrites it
    // (MetalImageCache.DeferredRefresh). The switch turns off the wait and the drop together.
    public static readonly bool SkipOverwrittenRefresh =
        !string.Equals(
            Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SKIP_OVERWRITTEN_REFRESH"), "0", StringComparison.Ordinal);

    public static readonly bool CoalesceFlushes =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FLUSH_COALESCE"), "0", StringComparison.Ordinal);

    public static readonly int CoalesceMaxSubmissions =
        PositiveInt("SHARPEMU_METAL_V2_FLUSH_COALESCE_SUBMISSIONS", 8);

    public static readonly int CoalesceMaxEncoders =
        PositiveInt("SHARPEMU_METAL_V2_FLUSH_COALESCE_ENCODERS", 64);

    public static readonly int CoalesceMaxMicroseconds =
        PositiveInt("SHARPEMU_METAL_V2_FLUSH_COALESCE_US", 2000);

    private static int PositiveInt(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out var value) && value > 0
            ? value
            : fallback;

    // ConcurrentDictionary.TryAdd takes the bucket lock even when the key exists, and
    // the hot bind paths ask every time; the lock-free lookup answers the repeats.
    public static void WarnOnce(string key, string message)
    {
        if (!_reported.ContainsKey(key) && _reported.TryAdd(key, 0))
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] {message}");
        }
    }

    public static void InfoOnce(string key, string message)
    {
        if (!_reported.ContainsKey(key) && _reported.TryAdd(key, 0))
        {
            Console.Error.WriteLine($"[METAL-V2][INFO] {message}");
        }
    }

    // An interpolated message is formatted only while its key has not been reported: many
    // of these sit on per-draw and per-transfer paths, and the message was built (and its
    // holes evaluated) on every call only to be dropped.
    public static void WarnOnce(string key, [InterpolatedStringHandlerArgument(nameof(key))] ref OnceMessageHandler message)
    {
        if (message.Wanted)
        {
            WarnOnce(key, message.ToStringAndClear());
        }
    }

    public static void InfoOnce(string key, [InterpolatedStringHandlerArgument(nameof(key))] ref OnceMessageHandler message)
    {
        if (message.Wanted)
        {
            InfoOnce(key, message.ToStringAndClear());
        }
    }

    [InterpolatedStringHandler]
    public ref struct OnceMessageHandler
    {
        private DefaultInterpolatedStringHandler _inner;

        public OnceMessageHandler(int literalLength, int formattedCount, string key, out bool shouldAppend)
        {
            Wanted = shouldAppend = !_reported.ContainsKey(key);
            _inner = shouldAppend ? new DefaultInterpolatedStringHandler(literalLength, formattedCount) : default;
        }

        public bool Wanted { get; }

        public void AppendLiteral(string value) => _inner.AppendLiteral(value);

        public void AppendFormatted<T>(T value) => _inner.AppendFormatted(value);

        public void AppendFormatted<T>(T value, string? format) => _inner.AppendFormatted(value, format);

        public void AppendFormatted<T>(T value, int alignment) => _inner.AppendFormatted(value, alignment);

        public void AppendFormatted<T>(T value, int alignment, string? format) => _inner.AppendFormatted(value, alignment, format);

        public void AppendFormatted(ReadOnlySpan<char> value) => _inner.AppendFormatted(value);

        public void AppendFormatted(string? value) => _inner.AppendFormatted(value);

        public string ToStringAndClear() => _inner.ToStringAndClear();
    }

    public static void Trace(string message)
    {
        if (Verbose)
        {
            Console.Error.WriteLine($"[METAL-V2][TRACE] {message}");
        }
    }
}
