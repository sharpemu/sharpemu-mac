// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using System.Runtime.InteropServices;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.Gpu.Metal.V2;

// Tessellated draws run as a hull kernel and a patch draw through Metal's tessellator
// (SharpEmu.Libs.Gpu.Rendering.RenderExecutor.Tessellation.cs). This side binds the kernel's launch,
// turns the factors the hull program wrote to the guest's ring into Metal's (half precision, a small
// kernel in the hull's compute encoder) and draws the patches; the domain program is the pipeline's
// post-tessellation vertex function (CreateRenderPipelineState). Encoders run in order
// (EncoderOrdering), so the pass that draws the patches reads what both kernels wrote.
internal sealed partial class MetalRenderHost
{
    public bool SupportsTessellationEmulation => true;

    // SHARPEMU_METAL_V2_TESSELLATION_FLIP_WINDING=1 winds the tessellator's triangles the other way
    // than SetTessellatorState maps the guest's VGT_TF_PARAM.TOPOLOGY, for checking the mapping.
    private static readonly bool FlipTessellationWinding =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TESSELLATION_FLIP_WINDING") == "1";

    // The launch the next dispatch binds: patch count, first instance and patches of one instance, and
    // the vertex ids in the ring.
    private bool _hullLaunchPending;
    private uint _hullPatchCount;
    private uint _hullFirstInstance;
    private uint _hullInstancePatches;
    private nint _hullVertexBuffer;
    private ulong _hullVertexOffset;

    // The factors the next DrawPatches hands the tessellator, and the bytes of one patch's.
    private MetalGuestRange _tessellationFactors;
    private uint _tessellationFactorBytes;
    private nint _tessellationFactorPipeline;

    public void BindHullLaunch(ReadOnlySpan<uint> vertexIds, uint patchCount, uint firstInstance, uint instancePatches)
    {
        var binding = UploadTransient(MemoryMarshal.AsBytes(vertexIds.IsEmpty ? [0u] : vertexIds), 16);
        _hullVertexBuffer = (nint)binding.Handle;
        _hullVertexOffset = binding.Offset;
        _hullPatchCount = patchCount;
        _hullFirstInstance = firstInstance;
        _hullInstancePatches = instancePatches;
        _hullLaunchPending = true;
    }

    // Right before the dispatch: the encoder may be a new one by then.
    private unsafe void ApplyHullLaunch(nint encoder)
    {
        if (!_hullLaunchPending)
        {
            return;
        }

        _hullLaunchPending = false;
        var launch = stackalloc uint[4] { _hullPatchCount, _hullFirstInstance, _hullInstancePatches, 0 };
        MetalNative.SendSetBytes(encoder, SelHullSetBytes, (nint)launch, 4 * sizeof(uint), Gen5TessellationLaunch.HullLaunchBufferIndex);
        MetalNative.SendSetBuffer(
            encoder, SelSetComputeBuffer, _hullVertexBuffer, (nuint)_hullVertexOffset, Gen5TessellationLaunch.HullVertexBufferIndex);
    }

    // SHARPEMU_TRACE_TESS_FACTORS=1 (diagnostic): once each patch draw's submission completes, how many of the
    // factors the tessellator read cull their patch (zero or not finite), with the draw's first instance.
    private static readonly bool TraceTessFactors = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_TESS_FACTORS") == "1";
    private int _tracedTessFactors;

    private unsafe void TraceTessellationFactors(MetalGuestRange factors, uint patches, uint firstInstance, uint instances)
    {
        if (!TraceTessFactors || !factors.IsValid || _tracedTessFactors++ > 4000)
        {
            return;
        }

        var contents = MetalNative.Send(factors.Buffer, MetalNative.Selector("contents"));
        if (contents == 0)
        {
            return;
        }

        var flip = Interlocked.Read(ref _flips);
        var count = (int)Math.Min((ulong)patches * (_tessellationFactorBytes / sizeof(ushort)), 1u << 20);
        var address = contents + (nint)factors.Offset;
        _completions.Enqueue(SubmissionSerial, () =>
        {
            var halves = new ReadOnlySpan<ushort>((void*)address, count);
            int culled = 0, finite = 0;
            float max = 0;
            foreach (var bits in halves)
            {
                var value = (float)BitConverter.UInt16BitsToHalf(bits);
                if (!float.IsFinite(value) || value <= 0) culled++;
                else { finite++; max = Math.Max(max, value); }
            }

            Console.Error.WriteLine($"[METAL-V2][TESS-FACTORS] flip={flip} first={firstInstance} instances={instances} " +
                $"factors={count} culling={culled} max={max}");
            // SHARPEMU_TRACE_TESS_FACTORS_DIR=<dir>: the half factors themselves, per patch draw.
            if (Environment.GetEnvironmentVariable("SHARPEMU_TRACE_TESS_FACTORS_DIR") is { Length: > 0 } directory)
            {
                Directory.CreateDirectory(directory);
                File.WriteAllBytes(Path.Combine(directory, $"factors-flip{flip}-first{firstInstance}.bin"),
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(halves).ToArray());
            }
        });
    }

    public unsafe void PrepareTessellationFactors(ulong ringAddress, uint patchCount, bool quad, float minFactor, float maxFactor)
    {
        // The ring as a buffer read sees the hull kernel's stores, staged or in place.
        var factorsPerPatch = quad ? 6u : 4u;
        var ring = ObtainBuffer(ringAddress, (ulong)patchCount * factorsPerPatch * sizeof(float), isWritten: false);
        _tessellationFactors = _guestMemory.Allocate((ulong)patchCount * factorsPerPatch * sizeof(ushort), 256, out _);
        _tessellationFactorBytes = factorsPerPatch * sizeof(ushort);
        var encoder = EnsureComputeEncoder();
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers);
        MetalNative.SendVoid(encoder, SelSetComputePipelineState, TessellationFactorPipeline());
        // The hull's own dispatch, or the next one, binds its pipeline again.
        _boundComputePipeline = 0;
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, (nint)ring.Handle, (nuint)ring.Offset, 0);
        MetalNative.SendSetBuffer(
            encoder, SelSetComputeBuffer, _tessellationFactors.Buffer, (nuint)_tessellationFactors.Offset, 1);
        var limits = stackalloc uint[4]
        {
            patchCount,
            BitConverter.SingleToUInt32Bits(float.IsFinite(minFactor) ? Math.Max(minFactor, 1f) : 1f),
            BitConverter.SingleToUInt32Bits(float.IsFinite(maxFactor) ? Math.Clamp(maxFactor, 1f, 64f) : 64f),
            quad ? 1u : 0u,
        };
        MetalNative.SendSetBytes(encoder, SelHullSetBytes, (nint)limits, 4 * sizeof(uint), 2);
        MetalNative.SendDispatch(
            encoder,
            SelDispatchThreadgroups,
            new MtlSize { Width = (patchCount + 63) / 64, Height = 1, Depth = 1 },
            new MtlSize { Width = 64, Height = 1, Depth = 1 });
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers);
    }

    public unsafe void DrawPatches(uint controlPoints, uint patchCount, uint instanceCount, uint firstInstance)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        if (!_tessellationFactors.IsValid)
        {
            throw Fatal("A patch draw has no tessellation factors prepared.");
        }

        ApplyGraphicsState();
        var entry = _boundGraphicsPipeline!;
        // Each instance's patches have their own factors, and their own place in the off-chip buffer.
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetTessellationFactorBuffer, _tessellationFactors.Buffer, (nuint)_tessellationFactors.Offset,
            (nuint)(patchCount * _tessellationFactorBytes));
        var factors = _tessellationFactors;
        _tessellationFactors = default;
        TraceTessellationFactors(factors, patchCount * instanceCount, firstInstance, instanceCount);
        // The batch is drawn from instance 0. The tessellator reads each instance's factors at its instance id times the
        // stride, base instance included, and the factors hold this batch's instances only: drawn from the batch's first
        // guest instance, a second batch (Ghost of Tsushima's menu particles past 4096) read factors past their end, mostly
        // zero, so its patches were culled in most frames and its wind flashed in and out. The domain needs only its
        // patch's place among the batch's (the hull, which reads the guest's instances, has the first instance itself).
        var launch = stackalloc uint[2] { patchCount, 0 };
        MetalV2Native.SendSetBytes(
            _renderEncoder, SelSetVertexBytes, (nint)launch, 2 * sizeof(uint), Gen5TessellationLaunch.DomainLaunchBufferIndex);
        // A vertex-buffer slot another pipeline may use; forget it.
        _boundVertexBuffers[Gen5TessellationLaunch.DomainLaunchBufferIndex - VertexBufferSlotBase] = -1;
        var passes = FacePassCount(entry);
        for (var pass = 0; pass < passes; pass++)
        {
            if (passes > 1)
            {
                SelectFacePass(pass);
            }

            MetalV2Native.SendDrawPatches(
                _renderEncoder,
                SelDrawPatches,
                controlPoints,
                0,
                patchCount,
                0,
                0,
                Math.Max(instanceCount, 1),
                0);
        }

        EndFacePasses(entry, passes);
        CountDraw(started);
        ProfileWork(entry, "draw-patches", WorkShape.Vertices(patchCount * controlPoints, instanceCount));
        if (FlushAfterPatchDraws)
        {
            FlushCommandBuffer();
        }
    }

    // SHARPEMU_DIAG_FLUSH_AFTER_PATCHES=1 (diagnostic): commit the command buffer after every patch draw, so each
    // tessellation batch's hull, factor conversion and draw run in a command buffer of their own.
    private static readonly bool FlushAfterPatchDraws =
        Environment.GetEnvironmentVariable("SHARPEMU_DIAG_FLUSH_AFTER_PATCHES") == "1";

    // The tessellator state of a domain program's pipeline (Gen5TessellationLaunch): factors per
    // patch, the guest's partition and winding, and a maximum factor that admits the guest's largest
    // (Metal's fractional-odd factors stop one below its maximum). The guest's clockwise output is
    // Metal's counter-clockwise, found by looking: with the winding taken as written, Grand Theft
    // Auto V's palm trunks showed the insides of their far halves (dark, the wrong faces culled).
    private static void SetTessellatorState(nint descriptor, Gen5TessellationLaunch tessellation)
    {
        var (partition, needed) = tessellation.Partition switch
        {
            Gen5TessellationPartition.Pow2 => (Mtl.TessellationPartitionModePow2, BitOperations.RoundUpToPowerOf2(tessellation.MaxFactor)),
            Gen5TessellationPartition.FractionalOdd => (Mtl.TessellationPartitionModeFractionalOdd, tessellation.MaxFactor + 1),
            Gen5TessellationPartition.FractionalEven => (Mtl.TessellationPartitionModeFractionalEven, tessellation.MaxFactor),
            _ => (Mtl.TessellationPartitionModeInteger, tessellation.MaxFactor),
        };
        var clockwise = tessellation.OutputClockwise == FlipTessellationWinding;
        // Each instance's patches have factors of their own (the hull kernel runs over every instance's patches, and
        // DrawPatches binds them with a stride of one instance). Per patch alone, Metal ignores the stride and every
        // instance takes the first instance's factors: Ghost of Tsushima's menu particles, one patch an instance, were
        // drawn or culled whole by the first particle of each batch, so its wind and dots flashed in and out.
        MetalV2Native.SendVoidValue(
            descriptor, MetalNative.Selector("setTessellationFactorStepFunction:"), Mtl.TessellationFactorStepFunctionPerPatchAndPerInstance);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setTessellationPartitionMode:"), partition);
        MetalV2Native.SendVoidValue(
            descriptor,
            MetalNative.Selector("setTessellationOutputWindingOrder:"),
            (nuint)(clockwise ? Mtl.WindingClockwise : Mtl.WindingCounterClockwise));
        MetalV2Native.SendVoidValue(
            descriptor, MetalNative.Selector("setTessellationControlPointIndexType:"), Mtl.ControlPointIndexTypeNone);
        MetalV2Native.SendVoidValue(
            descriptor, MetalNative.Selector("setMaxTessellationFactor:"), (nuint)Math.Clamp((needed + 1) & ~1u, 2u, 64u));
    }

    private nint TessellationFactorPipeline()
    {
        if (_tessellationFactorPipeline != 0)
        {
            return _tessellationFactorPipeline;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var options = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            MetalNative.ApplyMslCompileOptions(options);
            nint error = 0;
            var library = MetalNative.Send(
                _device, MetalNative.Selector("newLibraryWithSource:options:error:"),
                MetalNative.NsString(TessellationFactorSource), options, ref error);
            MetalNative.SendVoid(options, SelRelease);
            if (library == 0)
            {
                throw Fatal($"The tessellation factor kernel did not compile: {MetalNative.DescribeError(error)}.");
            }

            var function = MetalNative.Send(
                library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("sharpemu_tessellation_factors"));
            MetalNative.SendVoid(library, SelRelease);
            if (function == 0)
            {
                throw Fatal("The tessellation factor function is missing.");
            }

            error = 0;
            var pipeline = MetalNative.Send(
                _device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
            MetalNative.SendVoid(function, SelRelease);
            if (pipeline == 0)
            {
                throw Fatal($"The tessellation factor pipeline did not build: {MetalNative.DescribeError(error)}.");
            }

            _tessellationFactorPipeline = pipeline;
            return pipeline;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    // One thread per patch: the guest's four factors (three edges, the inside) or a quad's six (four
    // edges, two inside; the order is Metal's too) clamped to the guest's levels, as
    // MTLTriangleTessellationFactorsHalf or MTLQuadTessellationFactorsHalf. An edge not above zero
    // (or NaN) culls the patch, which Metal does for a zero edge.
    private const string TessellationFactorSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct sharpemu_tessellation_limits
        {
            uint patches;
            float min_factor;
            float max_factor;
            uint quad;
        };

        kernel void sharpemu_tessellation_factors(
            device const float* guest [[buffer(0)]],
            device half* factors [[buffer(1)]],
            constant sharpemu_tessellation_limits& limits [[buffer(2)]],
            uint patch [[thread_position_in_grid]])
        {
            if (patch >= limits.patches)
            {
                return;
            }

            uint edges = limits.quad != 0u ? 4u : 3u;
            uint count = limits.quad != 0u ? 6u : 4u;
            device const float* levels = guest + patch * count;
            device half* output = factors + patch * count;
            bool culled = false;
            for (uint edge = 0u; edge < edges; edge++)
            {
                culled = culled || !(levels[edge] > 0.0f);
            }

            for (uint factor = 0u; factor < count; factor++)
            {
                float level = levels[factor];
                output[factor] = culled ? 0.0h : half(isnan(level) ? limits.min_factor : clamp(level, limits.min_factor, limits.max_factor));
            }
        }
        """;

    private static readonly nint SelHullSetBytes = MetalNative.Selector("setBytes:length:atIndex:");
}
