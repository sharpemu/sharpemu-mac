// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

// Render passes, the pending draw state Metal needs at encoder level, draws,
// dispatches, barriers and the clears that replace a draw.
internal sealed partial class MetalRenderHost
{
    // MSL reserves buffer 0 for the stage argument buffer and buffer 1 for push data,
    // so vertex streams start at 2. Metal allows 31 buffer arguments per stage.
    private const int VertexBufferSlotBase = 2;
    private const int MaxVertexBufferSlots = 29;
    private const int MaxColorAttachments = RenderingState.ColorAttachmentCapacity;

    private static readonly nint SelRenderEncoder = MetalNative.Selector("renderCommandEncoderWithDescriptor:");
    private static readonly nint SelColorAttachments = MetalNative.Selector("colorAttachments");
    private static readonly nint SelDepthAttachment = MetalNative.Selector("depthAttachment");
    private static readonly nint SelStencilAttachment = MetalNative.Selector("stencilAttachment");
    private static readonly nint SelObjectAtIndex = MetalNative.Selector("objectAtIndexedSubscript:");
    private static readonly nint SelSetTexture = MetalNative.Selector("setTexture:");
    private static readonly nint SelSetLevel = MetalNative.Selector("setLevel:");
    private static readonly nint SelSetSlice = MetalNative.Selector("setSlice:");
    private static readonly nint SelSetDepthPlane = MetalNative.Selector("setDepthPlane:");
    private static readonly nint SelTextureType = MetalNative.Selector("textureType");
    private static readonly nint SelSetLoadAction = MetalNative.Selector("setLoadAction:");
    private static readonly nint SelSetStoreAction = MetalNative.Selector("setStoreAction:");
    private static readonly nint SelSetColorStoreAction = MetalNative.Selector("setColorStoreAction:atIndex:");
    private static readonly nint SelSetDepthStoreAction = MetalNative.Selector("setDepthStoreAction:");
    private static readonly nint SelSetStencilStoreAction = MetalNative.Selector("setStencilStoreAction:");
    private static readonly nint SelSetClearColor = MetalNative.Selector("setClearColor:");
    private static readonly nint SelSetClearDepth = MetalNative.Selector("setClearDepth:");
    private static readonly nint SelSetClearStencil = MetalNative.Selector("setClearStencil:");
    private static readonly nint SelSetRenderTargetWidth = MetalNative.Selector("setRenderTargetWidth:");
    private static readonly nint SelSetRenderTargetHeight = MetalNative.Selector("setRenderTargetHeight:");
    private static readonly nint SelSetRenderTargetArrayLength = MetalNative.Selector("setRenderTargetArrayLength:");
    private static readonly nint SelSetDefaultRasterSampleCount = MetalNative.Selector("setDefaultRasterSampleCount:");
    private static readonly nint SelSetRenderPipelineState = MetalNative.Selector("setRenderPipelineState:");
    private static readonly nint SelSetDepthStencilState = MetalNative.Selector("setDepthStencilState:");
    private static readonly nint SelSetViewport = MetalNative.Selector("setViewport:");
    private static readonly nint SelSetScissorRect = MetalNative.Selector("setScissorRect:");
    private static readonly nint SelSetViewports = MetalNative.Selector("setViewports:count:");
    private static readonly nint SelSetScissorRects = MetalNative.Selector("setScissorRects:count:");

    // The guest's viewport index is four bits wide, and Metal takes sixteen viewports.
    private const int ViewportSlotCount = 16;
    private static readonly nint SelSetCullMode = MetalNative.Selector("setCullMode:");
    private static readonly nint SelSetFrontFacingWinding = MetalNative.Selector("setFrontFacingWinding:");
    private static readonly nint SelSetDepthClipMode = MetalNative.Selector("setDepthClipMode:");
    private static readonly nint SelSetTriangleFillMode = MetalNative.Selector("setTriangleFillMode:");
    private static readonly nint SelSetBlendColor = MetalNative.Selector("setBlendColorRed:green:blue:alpha:");
    private static readonly nint SelSetDepthBias = MetalNative.Selector("setDepthBias:slopeScale:clamp:");
    private static readonly nint SelSetDepthTestBounds = MetalNative.Selector("setDepthTestMinBound:maxBound:");
    private static readonly nint SelSetStencilReferences =
        MetalNative.Selector("setStencilFrontReferenceValue:backReferenceValue:");
    private static readonly nint SelSetVertexBuffer = MetalNative.Selector("setVertexBuffer:offset:atIndex:");
    private static readonly nint SelSetFragmentBuffer = MetalNative.Selector("setFragmentBuffer:offset:atIndex:");
    private static readonly nint SelSetComputeBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
    private static readonly nint SelSetComputePipelineState = MetalNative.Selector("setComputePipelineState:");
    private static readonly nint SelUseResourceStages = MetalNative.Selector("useResource:usage:stages:");
    private static readonly nint SelUseResourceUsage = MetalNative.Selector("useResource:usage:");
    private static readonly nint SelDrawPrimitives =
        MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:");
    private static readonly nint SelDrawIndexedPrimitives = MetalNative.Selector(
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:");
    private static readonly nint SelDrawIndirect = MetalNative.Selector(
        "drawPrimitives:indirectBuffer:indirectBufferOffset:");
    private static readonly nint SelDrawIndexedIndirect = MetalNative.Selector(
        "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:");
    private static readonly nint SelDrawPatches = MetalNative.Selector(
        "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:");
    private static readonly nint SelDrawIndexedPatches = MetalNative.Selector(
        "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:" +
        "controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:");
    private static readonly nint SelSetTessellationFactorBuffer =
        MetalNative.Selector("setTessellationFactorBuffer:offset:instanceStride:");
    private static readonly nint SelSetVertexBytes = MetalNative.Selector("setVertexBytes:length:atIndex:");
    private static readonly nint SelDrawMeshThreadgroups = MetalNative.Selector(
        "drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:");
    private static readonly nint SelDispatchThreadgroups =
        MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:");
    private static readonly nint SelDispatchIndirect =
        MetalNative.Selector("dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:");
    private static readonly nint SelRenderMemoryBarrier =
        MetalNative.Selector("memoryBarrierWithScope:afterStages:beforeStages:");
    private static readonly nint SelComputeMemoryBarrier = MetalNative.Selector("memoryBarrierWithScope:");

    // The two encoders the host holds open across calls, and across worker passes when a
    // coalesced flush keeps the buffer open. The command buffer hands them back
    // autoreleased, so each is retained when opened and released once, right after
    // endEncoding (the command buffer keeps its own reference to what it recorded).
    private nint _renderEncoder;
    private nint _computeEncoder;

    /// <summary>Takes the host's own reference on an encoder it will hold in a field.</summary>
    private static nint RetainEncoder(nint encoder)
    {
        if (encoder != 0)
        {
            MetalNative.SendVoid(encoder, SelRetain);
        }

        return encoder;
    }

    /// <summary>Closes an encoder opened through <see cref="RetainEncoder"/> and gives its reference back.</summary>
    // Two encoders of one command buffer can run at the same time on the GPU. The guest windows
    // are untracked (MetalGuestMemory), so Metal orders nothing between them, and an encoder can
    // read what the one before it is still writing. Black Myth: Wukong's Nanite passes did once a
    // larger transient budget let more of them share a command buffer: black blocks over the army,
    // black specks around Wukong and a black box over his eye in three of four cinematic runs; a
    // commit was the only thing that had ordered them.
    internal enum EncoderOrderingMode
    {
        None,
        // Each render and compute encoder waits on one fence before its first stage and updates
        // it after its last, so each starts only once the one before it has finished.
        Fence,
        // An event signal and wait after each: the same order, but every barrier splits the
        // command buffer, and in Wukong's gameplay the GPU sat idle ~46 ms of every frame on them.
        Event,
    }

    // SHARPEMU_METAL_V2_ENCODER_BARRIER=0 leaves the encoders unordered, =event orders them with events.
    internal EncoderOrderingMode EncoderOrdering { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ENCODER_BARRIER") switch
        {
            "0" => EncoderOrderingMode.None,
            "event" => EncoderOrderingMode.Event,
            _ => EncoderOrderingMode.Fence,
        };

    private static readonly nint SelComputeWaitForFence = MetalNative.Selector("waitForFence:");
    private static readonly nint SelComputeUpdateFence = MetalNative.Selector("updateFence:");
    private static readonly nint SelRenderWaitForFence = MetalNative.Selector("waitForFence:beforeStages:");
    private static readonly nint SelRenderUpdateFence = MetalNative.Selector("updateFence:afterStages:");
    private static readonly nint SelNewEncoderFence = MetalNative.Selector("newFence");

    private nint _encoderFence;
    private long _encoderBarriers;

    /// <summary>Orderings encoded after render and compute encoders since the last call.</summary>
    internal long TakeEncoderBarriers() => Interlocked.Exchange(ref _encoderBarriers, 0);

    private nint EncoderFence()
    {
        if (_encoderFence == 0)
        {
            _encoderFence = MetalNative.Send(_device, SelNewEncoderFence);
            if (_encoderFence == 0)
            {
                throw Fatal("A Metal fence could not be created to order encoders.");
            }
        }

        return _encoderFence;
    }

    // Right after a render or compute encoder is created, before anything is encoded into it.
    private void WaitForPreviousEncoder(nint encoder, bool render)
    {
        if (EncoderOrdering != EncoderOrderingMode.Fence || encoder == 0)
        {
            return;
        }

        if (render)
        {
            MetalV2Native.SendWaitForFence(encoder, SelRenderWaitForFence, EncoderFence(),
                Mtl.RenderStageVertex | Mtl.RenderStageObject | Mtl.RenderStageMesh);
        }
        else
        {
            MetalNative.SendVoid(encoder, SelComputeWaitForFence, EncoderFence());
        }
    }

    // Right before a render, compute or blit encoder ends.
    private void UpdateEncoderFence(nint encoder, bool render)
    {
        if (EncoderOrdering != EncoderOrderingMode.Fence || encoder == 0)
        {
            return;
        }

        if (render)
        {
            // Same shape as waitForFence:beforeStages: (encoder, selector, fence, stages).
            MetalV2Native.SendWaitForFence(encoder, SelRenderUpdateFence, EncoderFence(), Mtl.RenderStageFragment);
        }
        else
        {
            // Blit and compute encoders take the same updateFence: and waitForFence:.
            MetalNative.SendVoid(encoder, SelComputeUpdateFence, EncoderFence());
        }

        Interlocked.Increment(ref _encoderBarriers);
    }

    void IMetalSubmissionContext.EncoderStarted(nint encoder, bool render) => WaitForPreviousEncoder(encoder, render);

    void IMetalSubmissionContext.EncoderEnding(nint encoder, bool render) => UpdateEncoderFence(encoder, render);

    private void EndAndReleaseEncoder(nint encoder, bool render)
    {
        UpdateEncoderFence(encoder, render);
        MetalNative.SendVoid(encoder, SelEndEncoding);
        MetalNative.SendVoid(encoder, SelRelease);
        if (EncoderOrdering == EncoderOrderingMode.Event && _commandBuffer != 0)
        {
            EncodeEventBarrier(_commandBuffer);
            Interlocked.Increment(ref _encoderBarriers);
        }
    }
    private long _renderPasses;
    private long _passEndClear;
    private long _passEndTarget;
    private long _passEndTransfer;
    private long _passEndCompute;
    private long _passEndFlush;
    private long _passEndPresent;
    private long _passEndOther;
    private long _targetlessPasses;

    // Sized from the enum: a hard-coded length indexed by a cause is one new cause away
    // from throwing on the first transfer of that kind.
    private readonly long[] _transferCauses =
        new long[Enum.GetValues<MetalTransferCause>().Length];

    /// <summary>Transfers by cause since the last call; every one ended the open pass.</summary>
    internal string TakeTransferCauses()
    {
        var builder = new System.Text.StringBuilder();
        for (var cause = 0; cause < _transferCauses.Length; cause++)
        {
            var count = _transferCauses[cause];
            _transferCauses[cause] = 0;
            if (count != 0)
            {
                builder.Append(builder.Length == 0 ? string.Empty : " ")
                    .Append((MetalTransferCause)cause).Append('=').Append(count);
            }
        }

        return builder.Length == 0 ? "none" : builder.ToString();
    }

    // What the passes cost in memory traffic, which on a tile-based GPU is the whole
    // story: a pass loads every attachment it does not clear and stores every one it
    // keeps, so the bill is attachment bytes times passes, and each mid-frame split
    // pays it again. Counted as the descriptor is built, from the real pixel size.
    private long _loadBytes;
    private long _storeBytes;
    private long _clearedBytes;
    private long _passesEmpty;
    private long _discardedBytes;

    // The standalone clear passes: a load-clear encoder with no draws, which ends the
    // open pass and then stores a whole attachment of its own. They are render passes
    // too, and the ones the pass counters never saw.
    private long _clearPasses;
    private long _clearDepthPasses;
    private long _clearPixels;

    internal (long Passes, long DepthPasses, long Pixels) TakeClearPasses()
    {
        var passes = Interlocked.Exchange(ref _clearPasses, 0);
        var depth = Interlocked.Exchange(ref _clearDepthPasses, 0);
        var pixels = Interlocked.Exchange(ref _clearPixels, 0);
        return (passes, depth, pixels);
    }

    // A pass that renders into the same attachments as the pass before last: the
    // target set left them and came straight back, so the pair stored and reloaded
    // the whole framebuffer for whatever happened in between. This is the merge
    // opportunity, counted rather than assumed.
    private long _passesReturned;
    private ulong _previousPassKey;
    private ulong _passBeforeKey;

    internal (long Load, long Store, long Cleared, long Discarded, long Empty, long Returned)
        TakeAttachmentBandwidth()
    {
        var load = Interlocked.Exchange(ref _loadBytes, 0);
        var store = Interlocked.Exchange(ref _storeBytes, 0);
        var cleared = Interlocked.Exchange(ref _clearedBytes, 0);
        var discarded = Interlocked.Exchange(ref _discardedBytes, 0);
        var empty = Interlocked.Exchange(ref _passesEmpty, 0);
        var returned = Interlocked.Exchange(ref _passesReturned, 0);
        return (load, store, cleared, discarded, empty, returned);
    }

    private void NotePassIdentity(in RenderingState state)
    {
        var first = state.ColorAttachmentCount != 0 ? state.ColorAttachments[0].View.Handle : 0;
        var key = first ^ (state.DepthStencilAttachment.View.Handle * 0x9E3779B97F4A7C15UL);
        if (key != 0 && key == _passBeforeKey && key != _previousPassKey)
        {
            _passesReturned++;
        }

        _passBeforeKey = _previousPassKey;
        _previousPassKey = key;
    }

    private static long AttachmentBytes(in RenderingState state, uint texelBytes) =>
        MetalPassStores.AttachmentBytes(state.Width, state.Height, state.Samples, texelBytes);

    private void NoteAttachmentTraffic(in RenderingState state)
    {
        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            var bytes = AttachmentBytes(
                in state, index < MaxColorAttachments ? _attachmentTexelBytes[index] : 0);
            if (state.ColorAttachments[index].IsClear ||
                (index < MaxColorAttachments && _overrideClear[index]))
            {
                _clearedBytes += bytes;
            }
            else if (index < MaxColorAttachments && _elidedLoads[index])
            {
                _loadsElided++;
                _loadElidedBytes += bytes;
            }
            else
            {
                _loadBytes += bytes;
            }

            _storeBytes += bytes;
        }

        ref readonly var depthStencil = ref state.DepthStencilAttachment;
        if (depthStencil.HasDepth && depthStencil.View.Handle != 0)
        {
            var bytes = AttachmentBytes(in state, _depthTexelBytes);
            if (depthStencil.DepthClear)
            {
                _clearedBytes += bytes;
            }
            else
            {
                _loadBytes += bytes;
            }

            _storeBytes += bytes;
        }

        if (depthStencil.HasStencil && depthStencil.View.Handle != 0)
        {
            // Metal keeps stencil in its own plane, one byte a pixel, whatever the
            // combined format's element size says.
            var bytes = AttachmentBytes(in state, 1);
            if (depthStencil.StencilClear)
            {
                _clearedBytes += bytes;
            }
            else
            {
                _loadBytes += bytes;
            }

            _storeBytes += bytes;
        }
    }

    internal (long Clear, long Target, long Transfer, long Compute, long Flush, long Present, long Other,
        long Targetless) TakePassBreakdown()
    {
        var clear = Interlocked.Exchange(ref _passEndClear, 0);
        var target = Interlocked.Exchange(ref _passEndTarget, 0);
        var transfer = Interlocked.Exchange(ref _passEndTransfer, 0);
        var compute = Interlocked.Exchange(ref _passEndCompute, 0);
        var flush = Interlocked.Exchange(ref _passEndFlush, 0);
        var present = Interlocked.Exchange(ref _passEndPresent, 0);
        var other = Interlocked.Exchange(ref _passEndOther, 0);
        var targetless = Interlocked.Exchange(ref _targetlessPasses, 0);
        return (clear, target, transfer, compute, flush, present, other, targetless);
    }
    private bool _renderingActive;
    private RenderingState _renderingState;
    private nint _boundComputePipeline;
    private MetalPipelineEntry? _boundGraphicsPipeline;

    // What the open encoder already has. Metal keeps encoder state between draws, so
    // these let a draw send only what changed; every one of them is reset when an
    // encoder is created, because a new encoder starts with none of it.
    private nint _boundDepthStencilState;
    private float _boundBlendRed = float.NaN;
    private float _boundBlendGreen = float.NaN;
    private float _boundBlendBlue = float.NaN;
    private float _boundBlendAlpha = float.NaN;
    private float _boundBiasConstant = float.NaN;
    private float _boundBiasSlope = float.NaN;
    private float _boundBiasClamp = float.NaN;

    // A new encoder starts with the bounds test off, which Metal spells [0, 1], so a
    // draw without the test never has to send it.
    private float _boundDepthMinBound;
    private float _boundDepthMaxBound = 1f;
    private uint _boundStencilFront = uint.MaxValue;
    private uint _boundStencilBack = uint.MaxValue;
    private readonly nint[] _boundVertexBuffers = new nint[MaxVertexBufferSlots];
    private readonly ulong[] _boundVertexOffsets = new ulong[MaxVertexBufferSlots];

    // The pipeline state on the encoder, which for a rectangle list is one of the bound
    // entry's per-index-type states rather than always its first.
    private nint _boundRenderState;

    // The viewport and scissor last sent on the encoder, and whether they went to every
    // viewport slot. Unknown until the first draw of an encoder sends them.
    private bool _boundViewportKnown;
    private bool _boundViewportAllSlots;
    private MtlViewport _boundViewport;
    private MtlScissorRect _boundScissor;

    private void ForgetEncoderState()
    {
        _graphicsBindings.Reset();
        _boundViewportKnown = false;
        _boundGraphicsPipeline = null;
        _boundRenderState = 0;
        _boundDepthStencilState = 0;
        _boundBlendRed = float.NaN;
        _boundBlendGreen = float.NaN;
        _boundBlendBlue = float.NaN;
        _boundBlendAlpha = float.NaN;
        _boundBiasConstant = float.NaN;
        _boundBiasSlope = float.NaN;
        _boundBiasClamp = float.NaN;
        _boundDepthMinBound = 0f;
        _boundDepthMaxBound = 1f;
        _boundStencilFront = uint.MaxValue;
        _boundStencilBack = uint.MaxValue;
        Array.Clear(_boundVertexBuffers);
        Array.Clear(_boundVertexOffsets);
        ForgetResidentResources();
    }
    private MetalPipelineEntry? _pendingGraphicsPipeline;

    private readonly nint[] _vertexBuffers = new nint[MaxVertexBufferSlots];
    private readonly ulong[] _vertexOffsets = new ulong[MaxVertexBufferSlots];
    private int _vertexBufferCount;
    private nint _indexBuffer;
    private ulong _indexOffset;
    private nuint _indexType;
    private DynamicDrawState _dynamicState;

    // The attachment subresource each acquired view points at; a created view starts
    // at its own level zero, and the image texture keeps the real level and slice.
    private readonly uint[] _attachmentLevels = new uint[MaxColorAttachments];
    private readonly uint[] _attachmentSlices = new uint[MaxColorAttachments];
    private readonly uint[] _attachmentDepthPlanes = new uint[MaxColorAttachments];
    private int _attachmentCount;
    private uint _depthAttachmentLevel;
    private uint _depthAttachmentSlice;

    // How wide one pixel of each attachment is, taken from the image the view was
    // made from. A tile-based GPU's cost is the attachment's bytes, not its pixels,
    // so this is what makes a pass split measurable rather than guessed at.
    private readonly uint[] _attachmentTexelBytes = new uint[MaxColorAttachments];
    private uint _depthTexelBytes;

    // The subresource each attachment view addresses. A store can only be discarded
    // when the pass covered all of it, because discarding throws the whole
    // subresource away and not just the part the render area touched.
    private readonly uint[] _attachmentWidths = new uint[MaxColorAttachments];
    private readonly uint[] _attachmentHeights = new uint[MaxColorAttachments];
    private uint _depthAttachmentWidth;
    private uint _depthAttachmentHeight;

    // The image texture behind each attachment view, and the subresource of it the
    // view really addresses. A clear names the texture and its own subresource, so a
    // clear that follows a pass can only be matched against the pass through these.
    private readonly nint[] _attachmentTextures = new nint[MaxColorAttachments];
    private readonly uint[] _attachmentRealLevels = new uint[MaxColorAttachments];
    private readonly uint[] _attachmentRealLayers = new uint[MaxColorAttachments];
    private nint _depthAttachmentTexture;
    private uint _depthAttachmentRealLevel;
    private uint _depthAttachmentRealLayer;

    // The pixel sizes the open pass was created with. The live arrays above belong to
    // whatever attachments were acquired most recently, which by the time a pass ends
    // is the pass about to begin.
    private readonly uint[] _passTexelBytes = new uint[MaxColorAttachments];
    private uint _passDepthTexelBytes;
    private readonly nint[] _passTextures = new nint[MaxColorAttachments];
    private readonly uint[] _passRealLevels = new uint[MaxColorAttachments];
    private readonly uint[] _passRealLayers = new uint[MaxColorAttachments];
    private readonly uint[] _passWidths = new uint[MaxColorAttachments];
    private readonly uint[] _passHeights = new uint[MaxColorAttachments];
    private nint _passDepthTexture;
    private uint _passDepthRealLevel;
    private uint _passDepthRealLayer;
    private uint _passDepthWidth;
    private uint _passDepthHeight;

    // A clear that is about to be encoded, which is how the pass it ends learns that
    // its stores are dead. Set just before the clear takes the pass's place and read
    // once, as the pass resolves its store actions.
    private nint _clearedTexture;
    private uint _clearedBaseLevel;
    private uint _clearedLevelCount;
    private uint _clearedBaseLayer;
    private uint _clearedLayerCount;
    private uint _clearedWidth;
    private uint _clearedHeight;
    private bool _clearedDepth;
    private bool _clearedStencil;

    /// <summary>
    /// Records the clear that is about to be encoded in place of the open pass, so the
    /// pass can discard the stores it overwrites. The clear is the very next thing the
    /// GPU does to that texture — nothing else is encoded between ending the pass and
    /// encoding the clear — so a store it covers cannot be observed.
    /// </summary>
    private void NoteClearAboutToBeEncoded(
        nint texture,
        in SubresourceRange range,
        uint width,
        uint height,
        bool depth,
        bool stencil)
    {
        if (!_renderingActive || _renderEncoder == 0)
        {
            return;
        }

        _clearedTexture = texture;
        _clearedBaseLevel = range.BaseLevel;
        _clearedLevelCount = Math.Max(range.LevelCount, 1);
        _clearedBaseLayer = range.BaseLayer;
        _clearedLayerCount = Math.Max(range.LayerCount, 1);
        _clearedWidth = width;
        _clearedHeight = height;
        _clearedDepth = depth;
        _clearedStencil = stencil;
    }

    private void ForgetNotedClear() => _clearedTexture = 0;

    private bool NotedClearCoversAttachment(
        nint texture,
        uint realLevel,
        uint realLayer,
        uint subresourceWidth,
        uint subresourceHeight)
    {
        return _clearedTexture != 0 && texture == _clearedTexture &&
            realLevel >= _clearedBaseLevel && realLevel < _clearedBaseLevel + _clearedLevelCount &&
            realLayer >= _clearedBaseLayer &&
            realLayer + _renderingState.Layers <= _clearedBaseLayer + _clearedLayerCount &&
            MetalPassStores.ClearCoversAttachment(
                _clearedWidth,
                _clearedHeight,
                _renderingState.Layers,
                _renderingState.Samples,
                _renderingState.Layers,
                _renderingState.Samples,
                subresourceWidth,
                subresourceHeight);
    }

    // The subresources the open pass was created with, so the next draw can tell
    // whether it renders into the very same attachments.
    private readonly uint[] _passLevels = new uint[MaxColorAttachments];
    private readonly uint[] _passSlices = new uint[MaxColorAttachments];
    private readonly uint[] _passDepthPlanes = new uint[MaxColorAttachments];
    private int _passAttachmentCount;
    private uint _passDepthLevel;
    private uint _passDepthSlice;

    /// <summary>
    /// Whether the next draw can keep the open render pass. Only what Metal bakes
    /// into the pass matters: the attachments, their subresources, the render area
    /// and whether a load clear has to happen. The shared RenderingState also
    /// carries the clear values of attachments that are not being cleared, and
    /// comparing those would end the pass between ordinary draws — which on a
    /// tile-based GPU means storing and reloading the whole framebuffer each time.
    /// </summary>
    private bool CanContinuePass(in RenderingState state)
    {
        if (!_renderingActive || _renderEncoder == 0 ||
            state.ColorAttachmentCount != _renderingState.ColorAttachmentCount ||
            state.Width != _renderingState.Width || state.Height != _renderingState.Height ||
            state.Layers != _renderingState.Layers || state.Samples != _renderingState.Samples ||
            _attachmentCount != _passAttachmentCount)
        {
            return false;
        }

        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            ref readonly var next = ref state.ColorAttachments[index];
            ref readonly var open = ref _renderingState.ColorAttachments[index];
            if (next.IsClear || next.View.Handle != open.View.Handle || next.Format != open.Format)
            {
                return false;
            }

            if (index < _attachmentCount &&
                (_attachmentLevels[index] != _passLevels[index] ||
                 _attachmentSlices[index] != _passSlices[index] ||
                 _attachmentDepthPlanes[index] != _passDepthPlanes[index]))
            {
                return false;
            }
        }

        ref readonly var nextDepth = ref state.DepthStencilAttachment;
        ref readonly var openDepth = ref _renderingState.DepthStencilAttachment;
        if (nextDepth.DepthClear || nextDepth.StencilClear ||
            nextDepth.View.Handle != openDepth.View.Handle ||
            nextDepth.HasDepth != openDepth.HasDepth || nextDepth.HasStencil != openDepth.HasStencil)
        {
            return false;
        }

        return nextDepth.View.Handle == 0 ||
            (_depthAttachmentLevel == _passDepthLevel && _depthAttachmentSlice == _passDepthSlice);
    }

    // What a pass asks for while it is open. Unknown is a promise to name the real
    // action before encoding ends, which is the only way to let the work that follows
    // the pass decide: a store whose pixels the very next pass clears is dead, and a
    // tile-based GPU charges the full attachment for it.
    private static nuint PendingStoreAction =>
        MetalV2Diagnostics.DeferStoreActions ? Mtl.StoreActionUnknown : Mtl.StoreActionStore;

    /// <summary>
    /// Names every store action the open pass left as Unknown. A colour, depth or
    /// stencil attachment whose whole subresource the next pass clears is discarded;
    /// everything else is stored, including every attachment of a pass that ends with
    /// nothing known to follow it.
    /// </summary>
    private void ResolveStoreActions(in RenderingState next, bool hasNext)
    {
        ref readonly var open = ref _renderingState;
        for (var index = 0; index < open.ColorAttachmentCount; index++)
        {
            var discard = hasNext
                ? NextPassClearsColorAttachment(in next, index)
                : index < MaxColorAttachments && NotedClearCoversAttachment(
                    _passTextures[index],
                    _passRealLevels[index],
                    _passRealLayers[index],
                    _passWidths[index],
                    _passHeights[index]);
            MetalV2Native.SendSetStoreActionAtIndex(
                _renderEncoder,
                SelSetColorStoreAction,
                discard ? Mtl.StoreActionDontCare : Mtl.StoreActionStore,
                (nuint)index);
            if (discard)
            {
                _discardedBytes += AttachmentBytes(
                    in open, index < MaxColorAttachments ? _passTexelBytes[index] : 0);
            }
        }

        ref readonly var openDepth = ref open.DepthStencilAttachment;
        if (openDepth.HasDepth && openDepth.View.Handle != 0)
        {
            var discard = hasNext
                ? NextPassClearsDepthStencil(in next, stencil: false)
                : _clearedDepth && NotedClearCoversAttachment(
                    _passDepthTexture, _passDepthRealLevel, _passDepthRealLayer,
                    _passDepthWidth, _passDepthHeight);
            MetalV2Native.SendVoidValue(
                _renderEncoder,
                SelSetDepthStoreAction,
                discard ? Mtl.StoreActionDontCare : Mtl.StoreActionStore);
            if (discard)
            {
                _discardedBytes += AttachmentBytes(in open, _passDepthTexelBytes);
            }
        }

        if (openDepth.HasStencil && openDepth.View.Handle != 0)
        {
            var discard = hasNext
                ? NextPassClearsDepthStencil(in next, stencil: true)
                : _clearedStencil && NotedClearCoversAttachment(
                    _passDepthTexture, _passDepthRealLevel, _passDepthRealLayer,
                    _passDepthWidth, _passDepthHeight);
            MetalV2Native.SendVoidValue(
                _renderEncoder,
                SelSetStencilStoreAction,
                discard ? Mtl.StoreActionDontCare : Mtl.StoreActionStore);
            if (discard)
            {
                _discardedBytes += AttachmentBytes(in open, 1);
            }
        }

        ForgetNotedClear();
    }

    // Discarding throws away the whole subresource, so the clear that follows has to
    // cover the whole subresource: its render area at least as large as the mip, and
    // at least as many layers as the pass being discarded wrote.
    private bool NextPassClearsColorAttachment(in RenderingState next, int openIndex)
    {
        var handle = _renderingState.ColorAttachments[openIndex].View.Handle;
        if (handle == 0)
        {
            return false;
        }

        for (var index = 0; index < next.ColorAttachmentCount && index < MaxColorAttachments; index++)
        {
            ref readonly var candidate = ref next.ColorAttachments[index];
            if ((!candidate.IsClear && !_overrideClear[index]) || candidate.View.Handle != handle ||
                _attachmentLevels[index] != _passLevels[openIndex] ||
                _attachmentSlices[index] != _passSlices[openIndex] ||
                _attachmentDepthPlanes[index] != _passDepthPlanes[openIndex])
            {
                continue;
            }

            return MetalPassStores.ClearCoversAttachment(
                next.Width,
                next.Height,
                next.Layers,
                next.Samples,
                _renderingState.Layers,
                _renderingState.Samples,
                _attachmentWidths[index],
                _attachmentHeights[index]);
        }

        return false;
    }

    private bool NextPassClearsDepthStencil(in RenderingState next, bool stencil)
    {
        ref readonly var nextDepth = ref next.DepthStencilAttachment;
        var handle = _renderingState.DepthStencilAttachment.View.Handle;
        return handle != 0 &&
            nextDepth.View.Handle == handle &&
            (stencil ? nextDepth.StencilClear : nextDepth.DepthClear) &&
            _depthAttachmentLevel == _passDepthLevel &&
            _depthAttachmentSlice == _passDepthSlice &&
            MetalPassStores.ClearCoversAttachment(
                next.Width,
                next.Height,
                next.Layers,
                next.Samples,
                _renderingState.Layers,
                _renderingState.Samples,
                _depthAttachmentWidth,
                _depthAttachmentHeight);
    }

    // The draw the next BeginRendering is for (IRenderHost.NoteUpcomingDraw), taken by
    // that BeginRendering whether or not it opens a pass, so it never outlives its draw.
    private MetalPipelineEntry? _upcomingPipeline;
    private uint _upcomingVertexCount;
    private uint _upcomingFirstVertex;

    // The colour attachments of the pass being opened whose load became DontCare, and the
    // attachments and bytes that saved since the last report.
    private readonly bool[] _elidedLoads = new bool[MaxColorAttachments];
    private long _loadsElided;
    private long _loadElidedBytes;

    public void NoteUpcomingDraw(in PipelineHandle pipeline, uint vertexCount, uint firstVertex)
    {
        _upcomingPipeline = _pipelineEntries.TryGetValue(pipeline.Pipeline, out var entry) && entry.Graphics
            ? entry
            : null;
        _upcomingVertexCount = vertexCount;
        _upcomingFirstVertex = firstVertex;
    }

    internal (long Attachments, long Bytes) TakeElidedLoads() =>
        (Interlocked.Exchange(ref _loadsElided, 0), Interlocked.Exchange(ref _loadElidedBytes, 0));

    /// <summary>
    /// Decides, for a pass about to be opened, which colour attachments its first draw
    /// overwrites whole (MetalPassLoads), so their load can be DontCare. Every attachment
    /// and the depth attachment has to be exactly the render area: a load action speaks
    /// for the whole subresource, and a smaller attachment would shrink the area Metal
    /// renders.
    /// </summary>
    private void ChooseElidedLoads(
        in RenderingState state,
        MetalPipelineEntry? entry,
        uint vertexCount,
        uint firstVertex)
    {
        Array.Clear(_elidedLoads);
        if (!MetalV2Diagnostics.ElideLoads || entry is null || entry.ConstantPositions is null ||
            entry.OverwriteMask == 0 || entry.Mesh is not null || entry.CullsEverything || vertexCount == 0)
        {
            return;
        }

        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            if (state.ColorAttachments[index].View.Handle == 0 || index >= _attachmentCount ||
                _attachmentWidths[index] != state.Width || _attachmentHeights[index] != state.Height)
            {
                return;
            }
        }

        ref readonly var depthStencil = ref state.DepthStencilAttachment;
        var depthAttached = depthStencil.View.Handle != 0 && (depthStencil.HasDepth || depthStencil.HasStencil);
        if (depthAttached && (_depthAttachmentWidth != state.Width || _depthAttachmentHeight != state.Height))
        {
            return;
        }

        var rejects =
            (depthAttached && depthStencil.HasDepth && _dynamicState.DepthTestEnabled &&
                _dynamicState.DepthCompare != GpuCompareOp.Always) ||
            (depthAttached && depthStencil.HasDepth && _dynamicState.DepthBoundsTestEnabled) ||
            (depthAttached && depthStencil.HasStencil && _dynamicState.StencilTestEnabled);
        var viewport = HostViewport(in _dynamicState);
        var scissor = _dynamicState.Scissor;
        var draw = new MetalPassLoads.Draw(
            entry.ConstantPositions,
            entry.PrimitiveType,
            entry.RectangleStates is not null,
            vertexCount,
            firstVertex,
            entry.CullMode == Mtl.CullModeNone,
            entry.DepthClipMode == Mtl.DepthClipModeClamp,
            entry.OverwriteMask,
            new MetalPassLoads.Viewport(viewport.OriginX, viewport.OriginY, viewport.Width, viewport.Height),
            scissor.Left,
            scissor.Top,
            scissor.Right,
            scissor.Bottom,
            rejects,
            state.Samples,
            state.Layers,
            state.Width,
            state.Height);
        for (var index = 0; index < state.ColorAttachmentCount && index < MaxColorAttachments; index++)
        {
            if (state.ColorAttachments[index].IsClear || _overrideClear[index])
            {
                continue;
            }

            _elidedLoads[index] = MetalPassLoads.OverwritesAttachment(
                in draw, index, _attachmentWidths[index], _attachmentHeights[index]);
        }
    }

    /// <summary>
    /// Decides, for the pass about to begin, the upload of every attachment whose acquire
    /// left it waiting (MetalImageCache.DeferredRefresh): dropped when the pass writes all
    /// of the attachment before reading any of it - a clear load action or an elided
    /// load - and made now otherwise. Runs after the previous pass ended and before this
    /// one's encoder exists, so an upload made here lands between the two. As for an
    /// elided load, every attachment has to be exactly the render area, so a load action
    /// speaks for the whole subresource.
    /// </summary>
    private void ResolveDeferredRefreshes(in RenderingState state)
    {
        if (!_imageCache.HasDeferredRefreshes)
        {
            return;
        }

        var exact = true;
        for (var index = 0; index < state.ColorAttachmentCount && exact; index++)
        {
            exact = index < _attachmentCount && index < MaxColorAttachments &&
                _attachmentWidths[index] == state.Width && _attachmentHeights[index] == state.Height;
        }

        ref readonly var depthStencil = ref state.DepthStencilAttachment;
        var depthAttached = depthStencil.View.Handle != 0 && (depthStencil.HasDepth || depthStencil.HasStencil);
        exact &= !depthAttached ||
            (_depthAttachmentWidth == state.Width && _depthAttachmentHeight == state.Height);
        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            ref readonly var attachment = ref state.ColorAttachments[index];
            var overwritten = exact && index < MaxColorAttachments &&
                (attachment.IsClear || _overrideClear[index] || _elidedLoads[index]);
            _imageCache.ResolveDeferredRefresh((nint)attachment.View.Handle, overwritten);
        }

        if (depthAttached)
        {
            _imageCache.ResolveDeferredRefresh(
                (nint)depthStencil.View.Handle,
                exact && depthStencil.HasDepth && depthStencil.DepthClear,
                stencilOverwritten: depthStencil.HasStencil && depthStencil.StencilClear);
        }

        // Whatever this pass does not name - a target the draw dropped - is uploaded as
        // its acquire would have.
        _imageCache.DrainDeferredRefreshes();
    }

    public void BeginRendering(in RenderingState state)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.RenderPassBegin);
        _joinedPipeline = null;
        _joinedVariant = null;
        // Before anything else, because this runs ahead of every draw and is therefore
        // the last moment at which a waiting clear is still invisible. Whatever this
        // pass cannot perform itself is encoded below, where no encoder is open.
        if (_pendingClears.Count != 0 && _imageCache.HasDeferredRefreshes)
        {
            // A clear that waited from before an attachment's acquire lands before that
            // attachment's upload, which is where the acquire used to make it - not as
            // this pass's load action over the upload. Clears are queued only after every
            // waiting upload is made (ClearAttachmentTexture), so a waiting clear is older.
            const int MaxRoots = 16;
            if (_pendingClears.Count > MaxRoots)
            {
                _imageCache.DrainDeferredRefreshes();
            }
            else
            {
                Span<nint> roots = stackalloc nint[_pendingClears.Count];
                for (var index = 0; index < roots.Length; index++)
                {
                    roots[index] = _pendingClears[index].Root;
                }

                _imageCache.DrainDeferredRefreshesOf(roots);
            }
        }

        var foldedClear = FoldPendingClearsInto(in state);
        var upcoming = _upcomingPipeline;
        var upcomingVertexCount = _upcomingVertexCount;
        var upcomingFirstVertex = _upcomingFirstVertex;
        _upcomingPipeline = null;
        _upcomingVertexCount = 0;
        var launch = _requestedLaunch;
        _requestedLaunch = null;
        // An attachment still waiting for its upload was not the open pass's target when
        // that pass began, so the pass cannot be continued: it is resolved below instead.
        var continues = !foldedClear && _pendingClears.Count == 0 && !_imageCache.HasDeferredRefreshes &&
            CanContinuePass(in state);
        if (continues && (launch is not { } requested || TryTakePreparedLaunch(requested)))
        {
            return;
        }

        if (!continues && launch is null && !foldedClear && _pendingClears.Count == 0 &&
            !_imageCache.HasDeferredRefreshes && (TryJoinDepthOnly(in state, upcoming) || TryJoinColorSubset(in state, upcoming)))
        {
            return;
        }

        if (!continues && CountJoinRefusals && _renderingActive && state.ColorAttachmentCount != 0)
        {
            _ = RefuseSubsetJoin(launch is not null ? "launch" : foldedClear ? "folded-clear" :
                _pendingClears.Count != 0 ? "pending-clear" : _imageCache.HasDeferredRefreshes ? "deferred-refresh" : "tried");
        }

        if (continues)
        {
            // Only the indirect launch this draw needs written ends the pass.
            _passEndIndirect++;
            PassProfileNoteEnd("indirect");
        }
        else if (_renderingActive)
        {
            var clears = state.DepthStencilAttachment.DepthClear || state.DepthStencilAttachment.StencilClear;
            for (var index = 0; index < state.ColorAttachmentCount && !clears; index++)
            {
                clears |= state.ColorAttachments[index].IsClear;
            }

            if (clears)
            {
                _passEndClear++;
            }
            else
            {
                _passEndTarget++;
            }

            PassProfileNoteEnd(clears ? "clear" : "target");
        }
        else if (_computeEncoder != 0)
        {
            PassProfileNoteEnd("render");
        }

        EndRenderingBefore(in state);
        EndComputeEncoder();
        NativeOcclusionMakeRoom();
        EncodeStagedStores();
        EncodeDeferredLabelCopies();
        MaterializePendingClears();
        if (launch is { } pending)
        {
            EncodeIndirectLaunches(pending);
        }

        ChooseElidedLoads(in state, upcoming, upcomingVertexCount, upcomingFirstVertex);
        ResolveDeferredRefreshes(in state);
        var commandBuffer = EnsureCommandBuffer();
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLRenderPassDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        if (descriptor == 0)
        {
            throw Fatal("A Metal render pass descriptor could not be created.");
        }

        try
        {
            var colors = MetalNative.Send(descriptor, SelColorAttachments);
            for (var index = 0; index < state.ColorAttachmentCount; index++)
            {
                ref readonly var attachment = ref state.ColorAttachments[index];
                var slot = MetalNative.SendAtIndex(colors, SelObjectAtIndex, (nuint)index);
                MetalNative.SendVoid(slot, SelSetTexture, (nint)attachment.View.Handle);
                MetalV2Native.SendVoidValue(slot, SelSetLevel, index < _attachmentCount ? _attachmentLevels[index] : 0);
                MetalV2Native.SendVoidValue(slot, SelSetSlice, index < _attachmentCount ? _attachmentSlices[index] : 0);
                MetalV2Native.SendVoidValue(
                    slot, SelSetDepthPlane, index < _attachmentCount ? _attachmentDepthPlanes[index] : 0);
                var clears = attachment.IsClear || (index < MaxColorAttachments && _overrideClear[index]);
                var elided = !clears && index < MaxColorAttachments && _elidedLoads[index];
                MetalV2Native.SendVoidValue(
                    slot,
                    SelSetLoadAction,
                    clears ? Mtl.LoadActionClear : elided ? Mtl.LoadActionDontCare : Mtl.LoadActionLoad);
                if (clears || elided)
                {
                    NoteColorTargetRebuilt(index);
                }

                NoteColorTargetDrawn(index);

                MetalV2Native.SendVoidValue(slot, SelSetStoreAction, PendingStoreAction);
                if (clears)
                {
                    MetalNative.SendVoidClearColor(
                        slot,
                        SelSetClearColor,
                        attachment.IsClear ? ClearColorOf(in attachment) : _overrideClearColor[index]);
                }
            }

            ref readonly var depthStencil = ref state.DepthStencilAttachment;
            var attached = state.ColorAttachmentCount != 0;
            if (depthStencil.View.Handle != 0 && (depthStencil.HasDepth || depthStencil.HasStencil) &&
                (!depthStencil.HasDepth || depthStencil.DepthClear) && (!depthStencil.HasStencil || depthStencil.StencilClear))
            {
                NoteDepthTargetRebuilt();
            }

            if (depthStencil.View.Handle != 0 && depthStencil.HasDepth && depthStencil.DepthClear)
            {
                NoteDepthCleared();
            }

            if (depthStencil.HasDepth && depthStencil.View.Handle != 0)
            {
                attached = true;
                var depth = MetalNative.Send(descriptor, SelDepthAttachment);
                MetalNative.SendVoid(depth, SelSetTexture, (nint)depthStencil.View.Handle);
                MetalV2Native.SendVoidValue(depth, SelSetLevel, _depthAttachmentLevel);
                MetalV2Native.SendVoidValue(depth, SelSetSlice, _depthAttachmentSlice);
                MetalV2Native.SendVoidValue(
                    depth, SelSetLoadAction, depthStencil.DepthClear ? Mtl.LoadActionClear : Mtl.LoadActionLoad);
                MetalV2Native.SendVoidValue(depth, SelSetStoreAction, PendingStoreAction);
                if (depthStencil.DepthClear)
                {
                    MetalNative.SendVoidDouble(
                        depth, SelSetClearDepth, BitConverter.UInt32BitsToSingle(depthStencil.ClearWord0));
                }
            }

            if (depthStencil.HasStencil && depthStencil.View.Handle != 0)
            {
                attached = true;
                var stencil = MetalNative.Send(descriptor, SelStencilAttachment);
                MetalNative.SendVoid(stencil, SelSetTexture, (nint)depthStencil.View.Handle);
                MetalV2Native.SendVoidValue(stencil, SelSetLevel, _depthAttachmentLevel);
                MetalV2Native.SendVoidValue(stencil, SelSetSlice, _depthAttachmentSlice);
                MetalV2Native.SendVoidValue(
                    stencil, SelSetLoadAction, depthStencil.StencilClear ? Mtl.LoadActionClear : Mtl.LoadActionLoad);
                MetalV2Native.SendVoidValue(stencil, SelSetStoreAction, PendingStoreAction);
                if (depthStencil.StencilClear)
                {
                    MetalV2Native.SendVoidValue(stencil, SelSetClearStencil, (nuint)depthStencil.ClearWord1);
                }
            }

            // A draw without attachments is either vertex-only (its output is what the vertex
            // stage stores, no fragment ever runs) or a pixel program that stores to memory.
            // The shared state asks for the maximum framebuffer in the vertex-only case, and a
            // tile-based GPU would walk every one of a 16384x16384 area's tiles for it, so
            // that pass takes the smallest area Metal accepts. A pixel program needs the real
            // area, or its fragments run for a single pixel.
            if (!attached)
            {
                if (state.RasterizesWithoutAttachments)
                {
                    ConfigureAttachmentlessPass(descriptor, state.Samples, state.Width, state.Height);
                }
                else
                {
                    ConfigureAttachmentlessPass(descriptor, state.Samples);
                }

                _targetlessPasses++;
            }

            if (state.Layers > 1)
            {
                MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetArrayLength, state.Layers);
            }

            _tinyPass = IsTinyPass(in state);
            if (PpProbe && _tinyPass)
            {
                EncodeTinyPassProbe(commandBuffer);
            }

            if (PassProfileEnabled)
            {
                PassProfileAttachRender(descriptor);
            }

            NativeOcclusionAttach(descriptor);
            OcclusionAuditAttach(descriptor);
            _renderEncoder = RetainEncoder(MetalNative.Send(commandBuffer, SelRenderEncoder, descriptor));
            MetalMemoryTelemetry.RenderEncoders.Add();
            LabelEncoder(_renderEncoder, $"render {state.Width}x{state.Height}");
            if (_renderEncoder == 0)
            {
                throw Fatal(
                    $"A Metal render command encoder could not be created: colors={state.ColorAttachmentCount} " +
                    $"depth={depthStencil.HasDepth} stencil={depthStencil.HasStencil} area={state.Width}x{state.Height}.");
            }

            WaitForPreviousEncoder(_renderEncoder, render: true);
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }

        NoteRecordedWork();
        _renderingActive = true;
        _renderingState = state;
        ForgetEncoderState();
        if (launch is not null)
        {
            WaitForIndirectLaunches();
        }

        _renderPasses++;
        var loadBefore = _loadBytes;
        var storeBefore = _storeBytes;
        NoteAttachmentTraffic(in state);
        if (PassProfileEnabled)
        {
            PassProfileDescribe(in state, _loadBytes - loadBefore, _storeBytes - storeBefore);
        }
        NotePassIdentity(in state);
        if (MetalV2Diagnostics.Verbose || _renderPasses < 4096)
        {
            var first = state.ColorAttachmentCount != 0 ? state.ColorAttachments[0] : default;
            ref readonly var passDepth = ref state.DepthStencilAttachment;
            MetalV2Diagnostics.InfoOnce(
                $"pass:{state.Width}x{state.Height}:{state.Samples}:{state.ColorAttachmentCount}:" +
                $"{(int)first.Format}:{first.IsClear}:{passDepth.HasDepth}:{passDepth.DepthClear}",
                $"render pass {state.Width}x{state.Height} samples={state.Samples} layers={state.Layers} " +
                $"colors={state.ColorAttachmentCount} format={(int)first.Format} clear={first.IsClear} " +
                $"depth={passDepth.HasDepth} depth_clear={passDepth.DepthClear} stencil={passDepth.HasStencil}.");
        }
        _passAttachmentCount = _attachmentCount;
        Array.Copy(_attachmentLevels, _passLevels, MaxColorAttachments);
        Array.Copy(_attachmentSlices, _passSlices, MaxColorAttachments);
        Array.Copy(_attachmentDepthPlanes, _passDepthPlanes, MaxColorAttachments);
        Array.Copy(_attachmentTexelBytes, _passTexelBytes, MaxColorAttachments);
        Array.Copy(_attachmentTextures, _passTextures, MaxColorAttachments);
        Array.Copy(_attachmentRealLevels, _passRealLevels, MaxColorAttachments);
        Array.Copy(_attachmentRealLayers, _passRealLayers, MaxColorAttachments);
        Array.Copy(_attachmentWidths, _passWidths, MaxColorAttachments);
        Array.Copy(_attachmentHeights, _passHeights, MaxColorAttachments);
        _passDepthTexture = _depthAttachmentTexture;
        _passDepthRealLevel = _depthAttachmentRealLevel;
        _passDepthRealLayer = _depthAttachmentRealLayer;
        _passDepthWidth = _depthAttachmentWidth;
        _passDepthHeight = _depthAttachmentHeight;
        _passDepthLevel = _depthAttachmentLevel;
        _passDepthSlice = _depthAttachmentSlice;
        _passDepthTexelBytes = _depthTexelBytes;
    }

    // A pass with no attachment takes its area and its sample count from the
    // descriptor. The sample count defaults to zero, which Metal does not accept:
    // the x86-64 AGX driver divides by it while it sizes the pass's tiles, and the
    // process dies of an integer divide fault inside renderCommandEncoderWithDescriptor:.
    // A zero sample count in the shared state means unspecified, as it does for
    // the pipeline's raster sample count, which this one must equal.
    internal static void ConfigureAttachmentlessPass(nint descriptor, uint samples, uint width = 1, uint height = 1)
    {
        MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetWidth, Math.Max(width, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetHeight, Math.Max(height, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetDefaultRasterSampleCount, Math.Max(samples, 1));
    }

    private static MtlClearColor ClearColorOf(in RenderingAttachment attachment) => new()
    {
        Red = BitConverter.UInt32BitsToSingle(attachment.ClearWord0),
        Green = BitConverter.UInt32BitsToSingle(attachment.ClearWord1),
        Blue = BitConverter.UInt32BitsToSingle(attachment.ClearWord2),
        Alpha = BitConverter.UInt32BitsToSingle(attachment.ClearWord3),
    };

    public void EndRendering() => FinishRendering(default, hasNext: false);

    // The executor ends the pass to release fragment stores, to order a write the pass cannot,
    // or because a dispatch is next; the TBDR line and the pass profile say which.
    void IRenderHost.EndRendering(string why)
    {
        if (_renderingActive)
        {
            switch (why)
            {
                case "release":
                    _passEndRelease++;
                    break;
                case "barrier":
                    _passEndBarrier++;
                    break;
                default:
                    _passEndDispatch++;
                    break;
            }

            PassProfileNoteEnd(why);
        }

        EndRendering();
    }

    // The pass is ending because another one is about to begin, and that one is what
    // decides whether this pass's stores are worth making.
    private void EndRenderingBefore(in RenderingState next) => FinishRendering(in next, hasNext: true);

    private void FinishRendering(in RenderingState next, bool hasNext)
    {
        if (!_renderingActive)
        {
            return;
        }

        var ending = _renderingState;
        if (_renderEncoder != 0)
        {
            if (MetalV2Diagnostics.DeferStoreActions)
            {
                ResolveStoreActions(in next, hasNext);
            }

            OcclusionAuditClose();
            EndAndReleaseEncoder(_renderEncoder, render: true);
            _renderEncoder = 0;
            NativeOcclusionFinish();
            PassProfileClose();
        }

        _renderingActive = false;
        _renderingState = default;
        _renderPassSerial++;
        ForgetNotedClear();

        ForgetEncoderState();
        DumpPassTargets(in ending);
        if (_passDrawCount == 0)
        {
            _passesEmpty++;
        }

        _passDrawCount = 0;
        _passLastProfile = null;
    }

    // Under SHARPEMU_METAL_V2_DUMP_TARGETS, remember what the pass just finished writing.
    // Only remember: a dump reads back through its own command buffer and waits, and
    // EndRendering runs inside EndEncoders, which runs inside the flush. Dumping here
    // would nest a flush inside a flush. They are drained at the next flip instead.
    // SHARPEMU_METAL_V2_DUMP_FLIPS_ONLY=1: a sampled frame writes only what it presents, not every
    // pass target, so a long run can be followed picture by picture at the cost of one readback.
    private static readonly bool DumpFlipsOnly =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_FLIPS_ONLY") == "1";

    private void DumpPassTargets(in RenderingState ending)
    {
        if (!DumpTargetsEnabled || DumpFlipsOnly)
        {
            return;
        }

        for (var index = 0; index < ending.ColorAttachmentCount; index++)
        {
            NotePendingDump((nint)ending.ColorAttachments[index].View.Handle, $"pass{index}");
        }

        // The depth the pass leaves behind says whether its geometry rasterised at all,
        // which a colour target that never changes cannot.
        if (ending.DepthStencilAttachment.HasDepth)
        {
            NotePendingDump((nint)ending.DepthStencilAttachment.View.Handle, "passdepth");
        }
    }

    private void EndComputeEncoder()
    {
        if (_computeEncoder == 0)
        {
            return;
        }

        _transferSharesCompute = false;
        EndAndReleaseEncoder(_computeEncoder, render: false);
        _computeEncoder = 0;
        _boundComputePipeline = 0;
        PassProfileClose();
        SnapshotComputeStorage();
    }

    internal nint EnsureComputeEncoder()
    {
        NoteRecordedWork();
        if (_computeEncoder == 0)
        {
            if (_renderingActive)
            {
                _passEndCompute++;
                PassProfileNoteEnd("compute");
            }

            EndRendering();
            EncodeStagedStores();
            EncodeDeferredLabelCopies();
        }

        // Only the clears this dispatch can see are encoded ahead of it; the rest keep
        // waiting for the pass that absorbs them. A clear is a pass of its own, so one
        // the dispatch sees ends an open compute encoder and this opens another.
        MaterializePendingClearsForCompute();
        if (_computeEncoder != 0)
        {
            return _computeEncoder;
        }

        var computeBuffer = EnsureCommandBuffer();
        _computeEncoder = PassProfileEnabled ? PassProfileComputeEncoder(computeBuffer) : 0;
        if (_computeEncoder == 0)
        {
            _computeEncoder = MetalNative.Send(computeBuffer, SelComputeEncoder);
            MetalMemoryTelemetry.ComputeEncoders.Add();
        }

        RetainEncoder(_computeEncoder);
        LabelEncoder(_computeEncoder, "compute");

        if (_computeEncoder == 0)
        {
            throw Fatal("A Metal compute command encoder could not be created.");
        }

        WaitForPreviousEncoder(_computeEncoder, render: false);

        // A new encoder has none of the residency the last one was given.
        ForgetResidentResources();
        return _computeEncoder;
    }

    // An encoder opened again after the pipeline was bound has no pipeline state.
    private void EnsureComputePipelineState(nint encoder, MetalPipelineEntry entry)
    {
        UsePipelineState(entry);
        if (_boundComputePipeline != entry.State)
        {
            MetalNative.SendVoid(encoder, SelSetComputePipelineState, entry.State);
            _boundComputePipeline = entry.State;
            SignpostPipeline(encoder, entry);
        }
    }

    public void BindVertexBuffers(ReadOnlySpan<BufferBinding> bindings, VertexInputInfo input)
    {
        _ = input;
        var count = Math.Min(bindings.Length, MaxVertexBufferSlots);
        if (bindings.Length > MaxVertexBufferSlots)
        {
            MetalV2Diagnostics.WarnOnce(
                "vertex-slots",
                $"A draw binds more vertex streams than Metal has buffer slots; the extra streams are dropped: " +
                $"streams={bindings.Length} slots={MaxVertexBufferSlots}.");
        }

        for (var index = 0; index < count; index++)
        {
            _vertexBuffers[index] = (nint)bindings[index].Handle;
            _vertexOffsets[index] = bindings[index].Offset;
        }

        _vertexBufferCount = count;
    }

    private ulong _indexGuestAddress;
    private ulong _indexGuestSize;

    public void NoteIndexSource(ulong guestAddress, ulong size)
    {
        _indexGuestAddress = guestAddress;
        _indexGuestSize = size;
    }

    public void BindIndexBuffer(BufferBinding binding, GpuIndexType type)
    {
        _indexGuestAddress = 0;
        _indexGuestSize = 0;
        _indexBuffer = (nint)binding.Handle;
        _indexOffset = binding.Offset;
        _indexType = MetalV2Formats.IndexType(type);
    }

    public void SetDynamicState(in DynamicDrawState state) => _dynamicState = state;

    public void BindPipeline(GpuBindPoint bindPoint, in PipelineHandle pipeline)
    {
        var entry = RequirePipelineEntry(in pipeline);
        if (bindPoint == GpuBindPoint.Graphics)
        {
            // A depth-only draw BeginRendering joined to the open pass draws with the
            // variant made for that pass's colour attachments.
            _pendingGraphicsPipeline = ReferenceEquals(entry, _joinedPipeline) ? _joinedVariant : entry;
            _joinedPipeline = null;
            _joinedVariant = null;
            return;
        }

        EnsureComputePipelineState(EnsureComputeEncoder(), entry);
        _pendingComputePipeline = entry;
    }

    private MetalPipelineEntry? _pendingComputePipeline;

    // Everything the executor recorded before the pass opened lands on the encoder here.
    private void ApplyGraphicsState()
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.GraphicsStateApply);
        if (_renderEncoder == 0)
        {
            throw Fatal("A draw has no open Metal render encoder.");
        }

        var entry = _pendingGraphicsPipeline ??
            throw Fatal("A draw has no bound Metal render pipeline state.");
        if (!ReferenceEquals(_boundGraphicsPipeline, entry))
        {
            // Every later draw of this encoder with the entry (its rectangle states too) is
            // in the same command buffer.
            UsePipelineState(entry);
            MetalNative.SendVoid(_renderEncoder, SelSetRenderPipelineState, entry.State);
            _boundRenderState = entry.State;
            SignpostPipeline(_renderEncoder, entry);
            MetalV2Native.SendVoidValue(
                _renderEncoder, SelSetCullMode, (nuint)(MetalV2Diagnostics.IgnoreCulling ? Mtl.CullModeNone : entry.CullMode));
            MetalV2Native.SendVoidValue(_renderEncoder, SelSetFrontFacingWinding, (nuint)entry.Winding);
            MetalV2Native.SendVoidValue(_renderEncoder, SelSetDepthClipMode, (nuint)entry.DepthClipMode);
            MetalV2Native.SendVoidValue(_renderEncoder, SelSetTriangleFillMode, (nuint)entry.FillMode);
            _boundGraphicsPipeline = entry;
        }

        // Encoder state survives between draws, so only what changed is sent. A draw
        // costs twenty-odd message sends otherwise, and consecutive draws in one pass
        // share almost all of their state.
        var depthStencilState = GetDepthStencilState(entry, in _dynamicState);
        if (_boundDepthStencilState != depthStencilState)
        {
            MetalNative.SendVoid(_renderEncoder, SelSetDepthStencilState, depthStencilState);
            _boundDepthStencilState = depthStencilState;
        }

        NativeOcclusionPrepareDraw();
        OcclusionAuditPrepareDraw();
        ApplyViewportAndScissor();
        if (_boundBlendRed != _dynamicState.BlendRed || _boundBlendGreen != _dynamicState.BlendGreen ||
            _boundBlendBlue != _dynamicState.BlendBlue || _boundBlendAlpha != _dynamicState.BlendAlpha)
        {
            MetalNative.SendVoidBlendColor(
                _renderEncoder,
                SelSetBlendColor,
                _dynamicState.BlendRed,
                _dynamicState.BlendGreen,
                _dynamicState.BlendBlue,
                _dynamicState.BlendAlpha);
            _boundBlendRed = _dynamicState.BlendRed;
            _boundBlendGreen = _dynamicState.BlendGreen;
            _boundBlendBlue = _dynamicState.BlendBlue;
            _boundBlendAlpha = _dynamicState.BlendAlpha;
        }

        // Points and lines take their own bias; triangles the front one, which a draw split
        // by facing (FacePassCount) replaces with the back one for its back faces.
        if (!_dynamicState.DepthBiasEnabled)
        {
            SetDepthBias(0f, 0f, 0f);
        }
        else if (RasterizesTriangles(entry))
        {
            SetDepthBias(_dynamicState.DepthBiasConstantFactor, _dynamicState.DepthBiasSlopeFactor, _dynamicState.DepthBiasClamp);
        }
        else
        {
            SetDepthBias(
                _dynamicState.DepthBiasPointLineConstantFactor, _dynamicState.DepthBiasPointLineSlopeFactor, _dynamicState.DepthBiasClamp);
        }

        ApplyDepthBounds(entry);

        if (_dynamicState.StencilTestEnabled &&
            (_boundStencilFront != _dynamicState.FrontStencil.Reference ||
                _boundStencilBack != _dynamicState.BackStencil.Reference))
        {
            MetalV2Native.SendStencilReferences(
                _renderEncoder,
                SelSetStencilReferences,
                _dynamicState.FrontStencil.Reference,
                _dynamicState.BackStencil.Reference);
            _boundStencilFront = _dynamicState.FrontStencil.Reference;
            _boundStencilBack = _dynamicState.BackStencil.Reference;
        }

        for (var index = 0; index < _vertexBufferCount; index++)
        {
            if (_boundVertexBuffers[index] == _vertexBuffers[index] &&
                _boundVertexOffsets[index] == _vertexOffsets[index])
            {
                continue;
            }

            MetalNative.SendSetBuffer(
                _renderEncoder,
                SelSetVertexBuffer,
                _vertexBuffers[index],
                (nuint)_vertexOffsets[index],
                (nuint)(VertexBufferSlotBase + index));
            _boundVertexBuffers[index] = _vertexBuffers[index];
            _boundVertexOffsets[index] = _vertexOffsets[index];
        }

        ApplyStageBindings(graphics: true);
    }

    private void SetDepthBias(float constant, float slope, float clamp)
    {
        if (_boundBiasConstant != constant || _boundBiasSlope != slope || _boundBiasClamp != clamp)
        {
            MetalV2Native.SendDepthBias(_renderEncoder, SelSetDepthBias, constant, slope, clamp);
            _boundBiasConstant = constant;
            _boundBiasSlope = slope;
            _boundBiasClamp = clamp;
        }
    }

    // Whether the pipeline's rasterizer sees triangles (a rectangle list's patches tessellate
    // to triangles), the only primitives with a facing.
    private static bool RasterizesTriangles(MetalPipelineEntry entry) =>
        entry.Mesh is { } mesh
            ? mesh.Topology == Gen5MslMeshTopology.Triangle
            : entry.RectangleStates is not null ||
                entry.PrimitiveType is Mtl.PrimitiveTriangle or Mtl.PrimitiveTriangleStrip;

    // How many times the draw is sent: twice when its front and back faces take different
    // polygon offsets, which Metal's one bias per draw cannot express - once for the front
    // faces under a host cull of the back ones with the front bias, then once for the back
    // faces under a cull of the front ones with the back bias. The cull follows the
    // pipeline's own front-face winding, so each pass keeps exactly the triangles the
    // guest's facing gives that bias. Once otherwise.
    private int FacePassCount(MetalPipelineEntry entry)
    {
        if (!_dynamicState.DepthBiasPerFace || !_dynamicState.DepthBiasEnabled || !RasterizesTriangles(entry) ||
            MetalV2Diagnostics.IgnoreCulling)
        {
            return 1;
        }

        if (entry.RectangleStates is not null)
        {
            throw Fatal(
                "A rectangle list's front and back faces take different polygon offsets, and a rectangle's facing is not " +
                $"modelled: pipeline={entry.Id} {entry.Profile}.");
        }

        if (entry.CullMode != Mtl.CullModeNone || entry.CullsEverything)
        {
            throw Fatal(
                "A draw split by facing for its polygon offsets has a pipeline that culls a face: " +
                $"pipeline={entry.Id} cull={entry.CullMode} all={entry.CullsEverything}.");
        }

        return 2;
    }

    // Sets the cull and the bias of one pass of a draw split by facing (pass 0: front faces).
    private void SelectFacePass(int pass)
    {
        MetalV2Native.SendVoidValue(_renderEncoder, SelSetCullMode, (nuint)(pass == 0 ? Mtl.CullModeBack : Mtl.CullModeFront));
        if (pass == 0)
        {
            SetDepthBias(_dynamicState.DepthBiasConstantFactor, _dynamicState.DepthBiasSlopeFactor, _dynamicState.DepthBiasClamp);
        }
        else
        {
            SetDepthBias(_dynamicState.DepthBiasBackConstantFactor, _dynamicState.DepthBiasBackSlopeFactor, _dynamicState.DepthBiasClamp);
        }
    }

    // Restores the pipeline's own cull after a draw split by facing.
    private void EndFacePasses(MetalPipelineEntry entry, int passes)
    {
        if (passes > 1)
        {
            MetalV2Native.SendVoidValue(_renderEncoder, SelSetCullMode, (nuint)entry.CullMode);
        }
    }

    // Guests program a negative viewport height to get y-up rendering out of
    // Vulkan's y-down NDC. Metal's NDC is already y-up and rejects a negative
    // height, so the same on-screen mapping is the flipped positive rect.
    private static MtlViewport HostViewport(in DynamicDrawState state)
    {
        var originY = (double)state.ViewportY;
        var viewportHeight = (double)state.ViewportHeight;
        if (viewportHeight < 0)
        {
            originY += viewportHeight;
            viewportHeight = -viewportHeight;
        }

        return new MtlViewport
        {
            OriginX = state.ViewportX,
            OriginY = originY,
            Width = Math.Max(state.ViewportWidth, 0),
            Height = viewportHeight,
            ZNear = Math.Clamp(state.ViewportMinDepth, 0f, 1f),
            ZFar = Math.Clamp(state.ViewportMaxDepth, 0f, 1f),
        };
    }

    private void ApplyViewportAndScissor()
    {
        var width = Math.Max(_renderingState.Width, 1u);
        var height = Math.Max(_renderingState.Height, 1u);
        var viewport = HostViewport(in _dynamicState);
        var scissor = _dynamicState.Scissor;
        var left = (uint)Math.Clamp(scissor.Left, 0, (int)width);
        var top = (uint)Math.Clamp(scissor.Top, 0, (int)height);
        var right = (uint)Math.Clamp(scissor.Right, 0, (int)width);
        var bottom = (uint)Math.Clamp(scissor.Bottom, 0, (int)height);
        var scissorWidth = right > left ? right - left : 0;
        var scissorHeight = bottom > top ? bottom - top : 0;
        if (_boundGraphicsPipeline is { CullsEverything: true } && !MetalV2Diagnostics.IgnoreCulling)
        {
            // Metal has no "cull both faces"; a one-pixel scissor is the cheapest
            // equivalent, and the draw then covers nothing it would have kept.
            left = 0;
            top = 0;
            scissorWidth = 1;
            scissorHeight = 1;
        }

        // Metal refuses an empty scissor rectangle; an empty guest rectangle becomes
        // one pixel at its corner, which the clamped viewport keeps out of the image.
        var scissorRect = new MtlScissorRect
        {
            X = left,
            Y = top,
            Width = Math.Max(scissorWidth, 1),
            Height = Math.Max(scissorHeight, 1),
        };
        var allSlots = _boundGraphicsPipeline is { WritesViewportIndex: true };
        if (_boundViewportKnown && _boundViewportAllSlots == allSlots &&
            _boundViewport.OriginX == viewport.OriginX && _boundViewport.OriginY == viewport.OriginY &&
            _boundViewport.Width == viewport.Width && _boundViewport.Height == viewport.Height &&
            _boundViewport.ZNear == viewport.ZNear && _boundViewport.ZFar == viewport.ZFar &&
            _boundScissor.X == scissorRect.X && _boundScissor.Y == scissorRect.Y &&
            _boundScissor.Width == scissorRect.Width && _boundScissor.Height == scissorRect.Height)
        {
            return;
        }

        _boundViewportKnown = true;
        _boundViewportAllSlots = allSlots;
        _boundViewport = viewport;
        _boundScissor = scissorRect;
        if (!allSlots)
        {
            MetalNative.SendVoidViewport(_renderEncoder, SelSetViewport, viewport);
            MetalNative.SendVoidScissor(_renderEncoder, SelSetScissorRect, scissorRect);
            return;
        }

        // A program that writes the viewport index can name any of the sixteen slots,
        // and Metal leaves a slot that was never set undefined. The draw state carries
        // the first guest viewport only — the one the Vulkan presenter sets — so every
        // slot gets it, which is what the Vulkan path's single viewport amounts to.
        unsafe
        {
            var viewports = stackalloc MtlViewport[ViewportSlotCount];
            var scissors = stackalloc MtlScissorRect[ViewportSlotCount];
            for (var index = 0; index < ViewportSlotCount; index++)
            {
                viewports[index] = viewport;
                scissors[index] = scissorRect;
            }

            MetalV2Native.SendVoidArray(_renderEncoder, SelSetViewports, (nint)viewports, ViewportSlotCount);
            MetalV2Native.SendVoidArray(_renderEncoder, SelSetScissorRects, (nint)scissors, ViewportSlotCount);
        }
    }

    public void Draw(uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        ApplyGraphicsState();
        var entry = _boundGraphicsPipeline!;
        if (entry.Mesh is { } mesh)
        {
            DrawMesh(mesh, vertexCount, instanceCount, indexed: false, firstIndexOrVertex: firstVertex,
                baseVertex: 0, firstInstance: firstInstance);
            CountDraw(started);
            ProfileWork(entry, "draw-mesh", WorkShape.Vertices(vertexCount, instanceCount));
            return;
        }

        if (entry.RectangleStates is not null)
        {
            DrawRectangles(entry, vertexCount, instanceCount, firstInstance, indexed: false, firstVertex, 0, 0);
            CountDraw(started);
            ProfileWork(entry, "draw-rectangles", WorkShape.Vertices(vertexCount, instanceCount));
            return;
        }

        var passes = FacePassCount(entry);
        for (var pass = 0; pass < passes; pass++)
        {
            if (passes > 1)
            {
                SelectFacePass(pass);
            }

            MetalV2Native.SendDrawPrimitivesBaseInstance(
                _renderEncoder,
                SelDrawPrimitives,
                entry.PrimitiveType,
                firstVertex,
                vertexCount,
                Math.Max(instanceCount, 1),
                firstInstance);
            for (var repeat = 1; _tinyPass && repeat < PpRepeat; repeat++)
            {
                MetalV2Native.SendDrawPrimitivesBaseInstance(
                    _renderEncoder, SelDrawPrimitives, entry.PrimitiveType, firstVertex, vertexCount,
                    Math.Max(instanceCount, 1), firstInstance);
            }
        }

        EndFacePasses(entry, passes);

        CountDraw(started);
        ProfileWork(entry, "draw", WorkShape.Vertices(vertexCount, instanceCount));
    }

    void IRenderHost.DrawIndexed(uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint firstInstance)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_indexBuffer == 0)
        {
            throw Fatal("An indexed draw has no bound index buffer.");
        }

        ApplyGraphicsState();
        var entry = _boundGraphicsPipeline!;
        if (entry.Mesh is { } mesh)
        {
            DrawMesh(mesh, indexCount, instanceCount, indexed: true, firstIndexOrVertex: firstIndex,
                baseVertex: vertexOffset, firstInstance: firstInstance);
            CountDraw(started);
            ProfileWork(entry, "draw-indexed-mesh", WorkShape.Indices(indexCount, instanceCount));
            return;
        }

        if (entry.RectangleStates is not null)
        {
            DrawRectangles(entry, indexCount, instanceCount, firstInstance, indexed: true, 0, firstIndex, vertexOffset);
            CountDraw(started);
            ProfileWork(entry, "draw-indexed-rectangles", WorkShape.Indices(indexCount, instanceCount));
            return;
        }

        var indexBytes = _indexType == Mtl.IndexTypeUInt32 ? 4UL : 2UL;
        var passes = FacePassCount(entry);
        for (var pass = 0; pass < passes; pass++)
        {
            if (passes > 1)
            {
                SelectFacePass(pass);
            }

            MetalV2Native.SendDrawIndexedBaseVertex(
                _renderEncoder,
                SelDrawIndexedPrimitives,
                entry.PrimitiveType,
                indexCount,
                _indexType,
                _indexBuffer,
                (nuint)(_indexOffset + (firstIndex * indexBytes)),
                Math.Max(instanceCount, 1),
                vertexOffset,
                firstInstance);
            for (var repeat = 1; _tinyPass && repeat < PpRepeat; repeat++)
            {
                MetalV2Native.SendDrawIndexedBaseVertex(
                    _renderEncoder, SelDrawIndexedPrimitives, entry.PrimitiveType, indexCount, _indexType, _indexBuffer,
                    (nuint)(_indexOffset + (firstIndex * indexBytes)), Math.Max(instanceCount, 1), vertexOffset,
                    firstInstance);
            }
        }

        EndFacePasses(entry, passes);

        CountDraw(started);
        ProfileWork(entry, "draw-indexed", WorkShape.Indices(indexCount, instanceCount));
    }

    void IRenderHost.DrawIndirect(BufferBinding arguments)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        // The guest's argument layout is (vertexCount, instanceCount, firstVertex,
        // firstInstance), which is exactly MTLDrawPrimitivesIndirectArguments.
        ApplyGraphicsState();
        var entry = _boundGraphicsPipeline!;
        if (entry.Mesh is { } mesh)
        {
            DrawMeshIndirect(entry, mesh, arguments, indexed: false);
            CountDraw(started);
            ProfileWork(entry, "draw-mesh-indirect", "indirect");
            return;
        }

        if (entry.RectangleStates is not null)
        {
            DrawRectanglesIndirect(entry, arguments);
            CountDraw(started);
            ProfileWork(entry, "draw-rectangles-indirect", "indirect");
            return;
        }

        var passes = FacePassCount(entry);
        for (var pass = 0; pass < passes; pass++)
        {
            if (passes > 1)
            {
                SelectFacePass(pass);
            }

            MetalV2Native.SendDrawIndirect(
                _renderEncoder,
                SelDrawIndirect,
                entry.PrimitiveType,
                (nint)arguments.Handle,
                (nuint)arguments.Offset);
        }

        EndFacePasses(entry, passes);
        Interlocked.Increment(ref _indirectGpuDraws);
        CountDraw(started);
        ProfileWork(entry, "draw-indirect", "indirect");
    }

    void IRenderHost.DrawIndexedIndirect(BufferBinding arguments)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_indexBuffer == 0)
        {
            throw Fatal("An indexed indirect draw has no bound index buffer.");
        }

        // The guest's argument layout is (indexCount, instanceCount, firstIndex,
        // vertexOffset, firstInstance), which is exactly MTLDrawIndexedPrimitivesIndirectArguments.
        ApplyGraphicsState();
        var entry = _boundGraphicsPipeline!;
        if (entry.Mesh is { } mesh)
        {
            DrawMeshIndirect(entry, mesh, arguments, indexed: true);
            CountDraw(started);
            ProfileWork(entry, "draw-indexed-mesh-indirect", "indirect");
            return;
        }

        if (entry.RectangleStates is not null)
        {
            // TakesIndirectIndexedRectangleListArguments let it through: the pass builds its
            // patch draw from the record (MetalRenderHost.RectangleRebase.cs).
            DrawRectanglesIndexedIndirect(entry, arguments);
            CountDraw(started);
            ProfileWork(entry, "draw-indexed-rect-indirect", "indirect");
            return;
        }

        var passes = FacePassCount(entry);
        for (var pass = 0; pass < passes; pass++)
        {
            if (passes > 1)
            {
                SelectFacePass(pass);
            }

            MetalV2Native.SendDrawIndexedIndirect(
                _renderEncoder,
                SelDrawIndexedIndirect,
                entry.PrimitiveType,
                _indexType,
                _indexBuffer,
                (nuint)_indexOffset,
                (nint)arguments.Handle,
                (nuint)arguments.Offset);
        }

        EndFacePasses(entry, passes);
        Interlocked.Increment(ref _indirectGpuDraws);
        CountDraw(started);
        ProfileWork(entry, "draw-indexed-indirect", "indirect");
    }

    // A persistent-threads kernel (Gen5MslShader.PersistentThreads) launches at most this many
    // workgroups, few enough that all of them stay resident on any Apple GPU, so none waits on
    // queue work held by a group that cannot start. SHARPEMU_METAL_V2_PERSISTENT_GROUPS=<n>
    // changes it; 0 launches the guest's count.
    private static readonly uint PersistentThreadGroups =
        uint.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PERSISTENT_GROUPS"), out var persistentGroups)
            ? persistentGroups
            : 64;

    private readonly HashSet<ulong> _clampAnnounced = [];

    public void Dispatch(uint groupsX, uint groupsY, uint groupsZ)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        // A persistent-thread kernel spins on memory other work publishes, which can be what a
        // label copy held for the commit writes: the copy would land only after the kernel had
        // waited out its budget for it (UE5's Nanite culling took twice as long per dispatch).
        // Everything held lands ahead of such a dispatch instead.
        if (LabelsAtCommit && _deferredLabelCopies.Count != 0 && _pendingComputePipeline is { PersistentThreads: true })
        {
            OrderAfterDeferredCopies(0, ulong.MaxValue);
        }

        var encoder = EnsureComputeEncoder();
        var entry = _pendingComputePipeline ?? throw Fatal("A dispatch has no bound Metal compute pipeline state.");
        if (entry.PersistentThreads && PersistentThreadGroups != 0 && groupsY <= 1 && groupsZ <= 1 &&
            groupsX > PersistentThreadGroups)
        {
            // UE5's Nanite culling sizes its dispatch to the groups the console GPU keeps
            // resident (1440) and shares the work through a queue every group drains, so
            // fewer groups compute the same thing, while groups the host cannot keep resident
            // only wait behind the spinning ones.
            if (_clampAnnounced.Add(entry.VertexHash))
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][CLAMP] hash=0x{entry.VertexHash:X16} groups {groupsX} -> {PersistentThreadGroups} (persistent threads)");
            }

            groupsX = PersistentThreadGroups;
        }

        EnsureComputePipelineState(encoder, entry);
        ApplyStageBindings(graphics: false);
        ApplyHullLaunch(encoder);
        MetalNative.SendDispatch(
            encoder,
            SelDispatchThreadgroups,
            new MtlSize { Width = Math.Max(groupsX, 1), Height = Math.Max(groupsY, 1), Depth = Math.Max(groupsZ, 1) },
            entry.ThreadsPerThreadgroup);
        CountDraw(started);
        ProfileWork(
            entry,
            "dispatch",
            WorkShape.Groups(groupsX, groupsY, groupsZ, entry.ThreadsPerThreadgroup));
    }

    public bool TryDispatchIndirect(ulong argumentsAddress)
    {
        var range = _guestMemory.Resolve(argumentsAddress, 3 * sizeof(uint));
        if (!range.IsValid)
        {
            return false;
        }

        // The GPU reads the group counts where the dispatch sits in the stream.
        OrderAfterDeferredCopies(argumentsAddress, 3 * sizeof(uint));
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var encoder = EnsureComputeEncoder();
        var entry = _pendingComputePipeline ??
            throw Fatal("An indirect dispatch has no bound Metal compute pipeline state.");
        EnsureComputePipelineState(encoder, entry);
        ApplyStageBindings(graphics: false);
        MetalV2Native.SendDispatchIndirect(
            encoder,
            SelDispatchIndirect,
            range.Buffer,
            (nuint)range.Offset,
            entry.ThreadsPerThreadgroup);
        CountDraw(started);
        ProfileWork(
            entry,
            "dispatch-indirect",
            WorkShape.Threads(entry.ThreadsPerThreadgroup));
        return true;
    }

    // A rectangle list: one three-control-point patch per rectangle, through the vertex
    // program's post-tessellation form (Gen5MslRectangleList), which returns the four
    // corners with the fourth derived from the other three's outputs. Metal's patch draws
    // take no base vertex, so the per-vertex buffers are rebound at the first vertex (or
    // the indexed draw's vertex offset) and the program gets the same vertex indices
    // through the draw block.
    private void DrawRectangles(
        MetalPipelineEntry entry,
        uint count,
        uint instanceCount,
        uint firstInstance,
        bool indexed,
        uint firstVertex,
        uint firstIndex,
        int vertexOffset)
    {
        var patches = MetalV2Formats.RectanglePatchCount(count);
        if (patches == 0)
        {
            return;
        }

        var index32 = _indexType == Mtl.IndexTypeUInt32;
        var state = RectangleState(entry, indexed ? index32 ? 2 : 1 : 0);
        if (_boundRenderState != state)
        {
            MetalNative.SendVoid(_renderEncoder, SelSetRenderPipelineState, state);
            _boundRenderState = state;
        }

        var vertexBase = indexed ? vertexOffset : (long)firstVertex;
        var strides = entry.RectangleVertexStrides!;
        for (var binding = 0; vertexBase != 0 && binding < Math.Min(strides.Length, _vertexBufferCount); binding++)
        {
            if (strides[binding] == 0)
            {
                continue;
            }

            var offset = (long)_vertexOffsets[binding] + (vertexBase * strides[binding]);
            if (offset < 0)
            {
                throw Fatal(
                    $"A rectangle list's vertex offset reaches before its vertex buffer: binding={binding} " +
                    $"offset={_vertexOffsets[binding]} stride={strides[binding]} base={vertexBase}.");
            }

            MetalNative.SendSetBuffer(
                _renderEncoder, SelSetVertexBuffer, _vertexBuffers[binding], (nuint)offset, (nuint)(VertexBufferSlotBase + binding));
            _boundVertexBuffers[binding] = _vertexBuffers[binding];
            _boundVertexOffsets[binding] = (ulong)offset;
        }

        // vertex_base, vertex_offset, index_type, first_index; see Gen5RectDraw.
        Span<uint> block = stackalloc uint[(int)(Gen5MslRectangleList.DrawInfoByteSize / sizeof(uint))];
        block[0] = indexed ? 0 : firstVertex;
        block[1] = unchecked((uint)vertexOffset);
        block[2] = indexed ? index32 ? 2u : 1u : 0u;
        block[3] = indexed ? firstIndex : 0;
        unsafe
        {
            fixed (uint* words = block)
            {
                MetalV2Native.SendSetBytes(
                    _renderEncoder,
                    SelSetVertexBytes,
                    (nint)words,
                    Gen5MslRectangleList.DrawInfoByteSize,
                    Gen5MslRectangleList.DrawInfoBufferIndex);
            }
        }

        // The index slot is read as a dword stream and must be bound even for an
        // automatic draw; the null buffer serves when the draw has no index buffer.
        MetalNative.SendSetBuffer(
            _renderEncoder,
            SelSetVertexBuffer,
            indexed ? _indexBuffer : _guestMemory.NullBuffer,
            indexed ? (nuint)_indexOffset : 0,
            Gen5MslRectangleList.IndexBufferIndex);
        // Both slots are vertex-buffer slots another pipeline may use; forget them.
        _boundVertexBuffers[Gen5MslRectangleList.IndexBufferIndex - VertexBufferSlotBase] = -1;
        _boundVertexBuffers[Gen5MslRectangleList.DrawInfoBufferIndex - VertexBufferSlotBase] = -1;

        // MTLQuadTessellationFactorsHalf: four edge and two inside factors, all 1.0.
        var factors = _guestMemory.Allocate(6 * sizeof(ushort), 256, out var factorBytes);
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(factorBytes).Fill(0x3C00);
        MetalNative.SendSetBuffer(_renderEncoder, SelSetTessellationFactorBuffer, factors.Buffer, (nuint)factors.Offset, 0);
        if (!indexed)
        {
            MetalV2Native.SendDrawPatches(
                _renderEncoder,
                SelDrawPatches,
                Gen5MslRectangleList.VerticesPerRectangle,
                0,
                patches,
                0,
                0,
                Math.Max(instanceCount, 1),
                firstInstance);
            return;
        }

        MetalV2Native.SendDrawIndexedPatches(
            _renderEncoder,
            SelDrawIndexedPatches,
            Gen5MslRectangleList.VerticesPerRectangle,
            0,
            patches,
            0,
            0,
            _indexBuffer,
            (nuint)(_indexOffset + (firstIndex * (index32 ? 4UL : 2UL))),
            Math.Max(instanceCount, 1),
            firstInstance);
    }

    // One NGG subgroup per threadgroup, one instance per grid row. The counts the
    // geometry engine would have put in registers travel in the draw-info block the
    // mesh prologue reads.
    private void DrawMesh(
        Gen5MslMeshInfo mesh,
        uint count,
        uint instanceCount,
        bool indexed,
        uint firstIndexOrVertex,
        int baseVertex,
        uint firstInstance)
    {
        // The subgroup split and the draw-info words are Gen5MslMeshLaunch's, which the
        // indirect launch kernel computes the same way on the GPU (DrawMeshIndirect).
        var input = MeshInputTopology(_boundGraphicsPipeline?.StaticParameters?.Topology);
        var subgroups = Gen5MslMeshLaunch.SubgroupCount(mesh, input, count);
        if (subgroups == 0 || instanceCount == 0)
        {
            return;
        }

        // Cached meshlets replace the whole-triangle subgroups of an indexed triangle list: a
        // threadgroup then holds the unique vertices of up to MaxPrimitives triangles, as the
        // geometry engine packs a subgroup, instead of three own vertices per triangle.
        MetalMeshletCache.Meshlets meshlets = default;
        var useMeshlets = !mesh.HasGeometryHalf && !mesh.Amplified && indexed &&
            TryGetMeshlets(mesh, count, firstIndexOrVertex, out meshlets);
        if (useMeshlets)
        {
            subgroups = meshlets.Count;
        }

        var drawInfo = _guestMemory.Allocate(Gen5MslMeshInfo.DrawInfoByteSize, 256, out var span);
        span.Clear();
        var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(span);
        Gen5MslMeshLaunch.FillDrawInfo(
            mesh,
            input,
            words,
            count,
            instanceCount,
            indexed,
            _indexType == Mtl.IndexTypeUInt32 ? 2u : 1u,
            firstIndexOrVertex,
            baseVertex,
            firstInstance);
        if (useMeshlets)
        {
            words[16] = meshlets.Count;
            words[17] = meshlets.VertexBase;
            words[18] = meshlets.PrimitiveBase;
        }

        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetMeshBuffer, drawInfo.Buffer, (nuint)drawInfo.Offset, (nuint)mesh.DrawInfoBufferIndex);
        // The index slot is read as a dword stream and must be bound even for an
        // automatic draw; the null buffer serves when the draw has no index buffer.
        var indexBuffer = indexed && _indexBuffer != 0 ? _indexBuffer : _guestMemory.NullBuffer;
        var indexOffset = indexed && _indexBuffer != 0 ? _indexOffset : 0;
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetMeshBuffer, indexBuffer, (nuint)indexOffset, (nuint)mesh.IndexBufferIndex);
        if (!mesh.HasGeometryHalf)
        {
            // Always bound, like the index slot: the null buffer when no meshlets are used.
            MetalNative.SendSetBuffer(
                _renderEncoder,
                SelSetMeshBuffer,
                useMeshlets ? meshlets.Buffer : _guestMemory.NullBuffer,
                0,
                Gen5MslMeshInfo.MeshletBufferIndex);
        }

        if (mesh.DeviceLds)
        {
            MetalNative.SendSetBuffer(
                _renderEncoder, SelSetMeshBuffer, _guestMemory.MeshLdsPool, 0, Gen5MslMeshInfo.LdsPoolBufferIndex);
        }

        if (PassProfileEnabled)
        {
            PassProfileNoteGroups((ulong)subgroups * instanceCount);
        }

        var entry = _boundGraphicsPipeline!;
        var passes = FacePassCount(entry);
        for (var pass = 0; pass < passes; pass++)
        {
            if (passes > 1)
            {
                SelectFacePass(pass);
            }

            MetalNative.SendDrawMesh(
                _renderEncoder,
                SelDrawMeshThreadgroups,
                new MtlSize { Width = subgroups, Height = instanceCount, Depth = 1 },
                new MtlSize { Width = 1, Height = 1, Depth = 1 },
                new MtlSize { Width = Math.Max(mesh.ThreadsPerThreadgroup, 1), Height = 1, Depth = 1 });
        }

        EndFacePasses(entry, passes);
    }

    private MetalMeshletCache? _meshletCache;

    private MetalMeshletCache MeshletCache => _meshletCache ??= new MetalMeshletCache(_guestMemory, _memory);

    // The cached meshlets of an indexed triangle-list draw, or false (counted by reason)
    // when it runs as whole-triangle subgroups.
    private bool TryGetMeshlets(Gen5MslMeshInfo mesh, uint indexCount, uint firstIndex, out MetalMeshletCache.Meshlets meshlets)
    {
        meshlets = default;
        var cache = MeshletCache;
        if (MetalMeshletCache.IsDisabled)
        {
            cache.NoteFallback(MetalMeshletCache.Fallback.Disabled);
            return false;
        }

        if (_boundGraphicsPipeline?.StaticParameters?.Topology != GpuPrimitiveTopology.TriangleList ||
            mesh.VerticesPerPrimitive != 3 || mesh.Topology != Gen5MslMeshTopology.Triangle)
        {
            cache.NoteFallback(MetalMeshletCache.Fallback.NotTriangleList);
            return false;
        }

        var index32 = _indexType == Mtl.IndexTypeUInt32;
        var elementBytes = index32 ? 4UL : 2UL;
        var rangeBytes = (ulong)indexCount * elementBytes;
        var rangeStart = (ulong)firstIndex * elementBytes;
        if (_indexGuestAddress == 0 || rangeStart + rangeBytes > _indexGuestSize)
        {
            cache.NoteFallback(MetalMeshletCache.Fallback.NoGuestAddress);
            return false;
        }

        var address = _indexGuestAddress + rangeStart;

        // The same predicate a host-side guest write uses: bytes the GPU is still writing
        // are not the bytes the draw will read, so they are neither hashed nor cached.
        if (_gpuWrites.PendingSerial(address, rangeBytes, CompletedSerial) != 0 ||
            _imageCache.HasGpuModifiedImageBytes(address, rangeBytes))
        {
            cache.NoteFallback(MetalMeshletCache.Fallback.GpuWrites);
            return false;
        }

        return cache.TryGet(
            address,
            indexCount,
            index32,
            mesh.MaxVertices,
            mesh.MaxPrimitives,
            SubmissionSerial,
            CompletedSerial,
            out meshlets);
    }

    private static Gen5MslMeshInputTopology MeshInputTopology(GpuPrimitiveTopology? topology) => topology switch
    {
        GpuPrimitiveTopology.TriangleStrip => Gen5MslMeshInputTopology.TriangleStrip,
        GpuPrimitiveTopology.PointList => Gen5MslMeshInputTopology.PointList,
        _ => Gen5MslMeshInputTopology.TriangleList,
    };

    private void CountDraw(long startedTicks)
    {
        NoteRecordedWork();
        OcclusionAuditCountDraw();
        Interlocked.Increment(ref _drawCount);
        NotePassDraw(_boundGraphicsPipeline?.Profile);
        Interlocked.Add(ref _drawTicks, System.Diagnostics.Stopwatch.GetTimestamp() - startedTicks);
        _recordedCommands++;
    }

    // Shader stores become visible to later work. Inside a render pass a scoped
    // barrier is enough; across encoders Metal's own tracking orders the windows.
    // The barrier itself records no guest write range: what the stores reached was
    // recorded where the storage buffers and images were bound.
    public void ShaderWriteBarrier(GpuPipelineStage sourceStages)
    {
        // Inside a pass the barrier names the stages that stored: a vertex-only
        // barrier is the form Apple GPUs order mid-pass, and it is the only one the
        // executor leaves the pass open for.
        nuint after = 0;
        if ((sourceStages & GpuPipelineStage.VertexShaderBit) != 0)
        {
            after |= Mtl.RenderStageVertex;
        }

        if ((sourceStages & GpuPipelineStage.FragmentShaderBit) != 0)
        {
            after |= Mtl.RenderStageFragment;
        }

        RecordMemoryBarrier(after == 0 ? Mtl.RenderStageVertex | Mtl.RenderStageFragment : after);
    }

    public bool OrdersVertexWritesInsidePass => true;

    public bool DefersFragmentWritePassEnd => MetalV2Diagnostics.DeferFragmentWritePassEnd;

    public long RenderPassSerial => _renderPassSerial;

    public void ShaderWriteHazardBarrier() => RecordMemoryBarrier();

    public void ShaderAccessBarrier() => RecordMemoryBarrier();

    private void RecordMemoryBarrier() => RecordMemoryBarrier(Mtl.RenderStageVertex | Mtl.RenderStageFragment);

    private void RecordMemoryBarrier(nuint afterStages,
        nuint scope = Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures | Mtl.BarrierScopeRenderTargets)
    {
        if (_renderEncoder != 0)
        {
            MetalV2Native.SendMemoryBarrierStages(
                _renderEncoder,
                SelRenderMemoryBarrier,
                scope,
                afterStages,
                Mtl.RenderStageVertex | Mtl.RenderStageFragment);
            return;
        }

        // Every compute encoder this host opens is serial, where Metal ignores the barrier
        // (MetalV2Diagnostics.SkipSerialComputeBarriers).
        if (_computeEncoder != 0 && !MetalV2Diagnostics.SkipSerialComputeBarriers)
        {
            // A compute encoder has no render targets to order against, and Metal
            // rejects a scope that names them: validation fails the assertion rather
            // than ignoring the bit, which ends the process.
            MetalV2Native.SendMemoryBarrierScope(
                _computeEncoder, SelComputeMemoryBarrier, scope & ~Mtl.BarrierScopeRenderTargets);
        }
    }

    /// <summary>
    /// A guest cache flush. Ending the render pass here would store and reload the
    /// whole attachment on a tile-based GPU, which is what the guest's per-draw
    /// barriers would cost; a scoped memory barrier on the open encoder expresses
    /// the same ordering without leaving the pass. Only a barrier with no encoder
    /// open has nothing to attach to, and then there is nothing to order either.
    /// </summary>
    public void EmitGlobalBarrier()
    {
        if (PostToWorker(EmitGlobalBarrier))
        {
            return;
        }

        if (TryDeferBehindDraws(_emitOrderedGlobalBarrier ??= EmitOrderedGlobalBarrier))
        {
            return;
        }

        DrainDraws();
        EmitOrderedGlobalBarrier();
    }

    private Action? _emitOrderedGlobalBarrier;
    private long _globalBarriersRecorded;
    private long _globalBarriersElided;

    /// <summary>Guest barriers recorded and elided in an open render pass, for tests.</summary>
    internal (long Recorded, long Elided) GlobalBarrierCounts => (_globalBarriersRecorded, _globalBarriersElided);

    private void EmitOrderedGlobalBarrier()
    {
        if (_renderEncoder != 0 && MetalV2Diagnostics.ElideGlobalBarriers)
        {
            // What the guest's flush or invalidate must make visible to later work in this pass
            // is shader stores; the executor has ordered every one but those it reports here.
            // Attachments need no barrier: their loads and stores run in draw order per fragment.
            if (_translation is not { HasUnorderedStoresInOpenPass: false })
            {
                RecordMemoryBarrier(Mtl.RenderStageVertex | Mtl.RenderStageFragment,
                    Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
                _globalBarriersRecorded++;
            }
            else
            {
                _globalBarriersElided++;
            }

            return;
        }

        if (_renderEncoder != 0 || _computeEncoder != 0)
        {
            RecordMemoryBarrier();
            return;
        }

        EndRendering();
        EndComputeEncoder();
    }

    // A barrier or release only orders GPU commands, so behind deferred draws it joins their queue
    // instead of draining them: a drain at every barrier left the program resolver, which only
    // starts a draw's programs once it is submitted, no lead over the draws the drain runs.
    private bool TryDeferBehindDraws(Action ordered, ulong writeAddress = 0, ulong writeSize = 0)
    {
        if (!MetalV2Diagnostics.DeferBarriers || _translation is not { HasPendingDraws: true, IsDraining: false } translation ||
            !_relay.IsGpuQueueThread)
        {
            return false;
        }

        translation.EnqueuePendingAction(ordered, writeAddress, writeSize);
        _deferredBarriers++;
        return true;
    }

    private long _deferredBarriers;

    // Whether the stream may read these bytes with the deferred work still queued.
    private bool QueuedWorkLeaves(ulong address, ulong size) =>
        MetalV2Diagnostics.DeferBarriers && _translation is { HasPendingDraws: true, IsDraining: false } translation &&
        _relay.IsGpuQueueThread && !translation.PendingWorkMayWrite(address, size);

    // How a guest clear arrives: with no pass open, with one open that has drawn
    // nothing yet, or with one open that has. Only the first two can become the next
    // pass's load action; the third has to be a real clear.
    private long _clearNoPass;
    private long _clearEmptyPass;
    private long _clearDrawnPass;

    internal string TakeClearArrivals()
    {
        var none = Interlocked.Exchange(ref _clearNoPass, 0);
        var empty = Interlocked.Exchange(ref _clearEmptyPass, 0);
        var drawn = Interlocked.Exchange(ref _clearDrawnPass, 0);
        return $"no_pass={none} empty_pass={empty} drawn_pass={drawn}";
    }

    // Replaces a draw whose only effect is one colour over the whole target.
    public void ClearColorTargets(ReadOnlySpan<ColorTargetState> targets, SolidColorClear clear)
    {
        if (!_renderingActive)
        {
            _clearNoPass++;
        }
        else if (_passDrawCount == 0)
        {
            _clearEmptyPass++;
        }
        else
        {
            _clearDrawnPass++;
        }

        Span<float> logical = stackalloc float[4] { clear.Red, clear.Green, clear.Blue, clear.Alpha };
        foreach (ref readonly var target in targets)
        {
            var acquisition = ((IRenderHost)this).AcquireColorAttachment(in target);
            if (acquisition.View.Handle == 0)
            {
                continue;
            }

            var mapping = target.Resolution.ExportMapping;
            var color = default(GpuClearColor);
            color.Float32_0 = logical[(int)mapping.Map(0)];
            color.Float32_1 = logical[(int)mapping.Map(1)];
            color.Float32_2 = logical[(int)mapping.Map(2)];
            color.Float32_3 = logical[(int)mapping.Map(3)];
            var view = target.Resolution.Request.View;
            var range = new SubresourceRange(0, 1, 0, Math.Max(view.LayerCount, 1));
            ClearAttachmentTexture(
                (nint)acquisition.View.Handle,
                in range,
                target.Resolution.Extent.Width,
                target.Resolution.Extent.Height,
                in color);
        }

        // No pass follows to decide a target this clear acquired and skipped.
        _imageCache.DrainDeferredRefreshes();
        _recordedCommands++;
    }

    void IMetalImageCacheOwner.ClearTextureColor(
        nint texture,
        ulong pixelFormat,
        in SubresourceRange range,
        uint width,
        uint height,
        in GpuClearColor color)
    {
        _ = pixelFormat;
        ClearAttachmentTexture(texture, in range, width, height, in color);
    }

    void IMetalImageCacheOwner.ClearTextureDepthStencil(
        nint texture,
        ulong pixelFormat,
        in SubresourceRange range,
        uint width,
        uint height,
        float depth,
        byte stencil,
        bool clearDepth,
        bool clearStencil)
    {
        _imageCache.DrainDeferredRefreshes();
        _clearPasses++;
        _clearDepthPasses++;
        _clearPixels += (long)Math.Max(width, 1) * Math.Max(height, 1) * Math.Max(range.LayerCount, 1);
        NoteClearAboutToBeEncoded(texture, in range, width, height, clearDepth, clearStencil);
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.DepthClear, texture);
        var commandBuffer = context.CommandBuffer;
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLRenderPassDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        try
        {
            if (clearDepth && MetalV2Formats.IsDepthFormat(pixelFormat))
            {
                var attachment = MetalNative.Send(descriptor, SelDepthAttachment);
                ConfigureClearAttachment(attachment, texture, in range);
                MetalNative.SendVoidDouble(attachment, SelSetClearDepth, depth);
            }

            if (clearStencil && MetalV2Formats.HasStencil(pixelFormat))
            {
                var attachment = MetalNative.Send(descriptor, SelStencilAttachment);
                ConfigureClearAttachment(attachment, texture, in range);
                MetalV2Native.SendVoidValue(attachment, SelSetClearStencil, stencil);
            }

            MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetWidth, Math.Max(width, 1));
            MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetHeight, Math.Max(height, 1));
            if (range.LayerCount > 1)
            {
                MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetArrayLength, range.LayerCount);
            }

            var encoder = MetalNative.Send(commandBuffer, SelRenderEncoder, descriptor);
            MetalMemoryTelemetry.RenderEncoders.Add();
            if (encoder != 0)
            {
                WaitForPreviousEncoder(encoder, render: true);
                UpdateEncoderFence(encoder, render: true);
                MetalNative.SendVoid(encoder, SelEndEncoding);
            }
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded($"clear {width}x{height} layers={range.LayerCount}");
        }
    }

    /// <summary>A clear of one subresource, either folded into a pass or encoded alone.</summary>
    private void ClearAttachmentTexture(
        nint texture,
        in SubresourceRange range,
        uint width,
        uint height,
        in GpuClearColor color)
    {
        // An attachment acquired before this clear was asked for is uploaded before it,
        // as its acquire used to, and never over it.
        _imageCache.DrainDeferredRefreshes();
        if (texture == 0)
        {
            return;
        }

        // A clear of a surface that is about to be the next pass's attachment is the
        // pass's load action, and a load action is free: the pass would otherwise clear
        // the surface here, store all of it, and then load all of it straight back. The
        // clear waits to find out, and BeginRendering runs before every draw, so nothing
        // can look at the surface before the wait is over.
        if (MetalV2Diagnostics.FoldClearsIntoPasses)
        {
            // Retained for as long as it waits. A clear is asked for while its image is
            // being looked up and the lookups that follow can replace that image's
            // texture; asking the cache whether it still exists is not an option, because
            // the cache's lock is held over the call that queued this and is fatal to
            // take twice. Clearing a texture nothing will read costs nothing.
            MetalNative.Send(texture, SelRetain);
            _pendingClears.Add(new PendingClear(texture, RootTexture(texture), range, width, height, color));
            return;
        }

        EncodeClearPass(texture, in range, width, height, in color);
    }

    // A clear that has been asked for and not yet encoded. It is either folded into the
    // load action of the pass that follows it or encoded as its own pass, and one of the
    // two always happens before anything can observe the surface.
    //
    // Root is the texture that owns the pixels: the clear may name a view of it, and
    // whatever else touches the same pixels may name the texture or another view.
    private readonly record struct PendingClear(
        nint Texture,
        nint Root,
        SubresourceRange Range,
        uint Width,
        uint Height,
        GpuClearColor Color);

    private readonly List<PendingClear> _pendingClears = [];
    private readonly bool[] _overrideClear = new bool[MaxColorAttachments];
    private readonly MtlClearColor[] _overrideClearColor = new MtlClearColor[MaxColorAttachments];
    private long _clearsFolded;
    private long _clearsEncoded;

    internal (long Folded, long Encoded) TakeClearFolding() =>
        (Interlocked.Exchange(ref _clearsFolded, 0), Interlocked.Exchange(ref _clearsEncoded, 0));

    /// <summary>
    /// Encodes every clear that is still waiting. Called wherever something that could
    /// observe a surface is about to be recorded and the pass about to begin — the only
    /// thing that can absorb a clear — is not it.
    /// </summary>
    private void MaterializePendingClears()
    {
        if (_pendingClears.Count == 0 || _encodingPendingClears)
        {
            return;
        }

        _encodingPendingClears = true;
        try
        {
            foreach (ref readonly var pending in
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_pendingClears))
            {
                EncodeClearPass(
                    pending.Texture, pending.Range, pending.Width, pending.Height, pending.Color);
                MetalNative.SendVoid(pending.Texture, SelRelease);
            }

            _pendingClears.Clear();
        }
        finally
        {
            _encodingPendingClears = false;
        }
    }

    private bool _encodingPendingClears;

    /// <summary>How many clears are waiting for a pass to absorb them.</summary>
    internal int PendingClearCount => _pendingClears.Count;

    /// <summary>
    /// Encodes, in the order they were asked for, the waiting clears of the textures
    /// something about to be recorded reads or writes, and leaves every other one
    /// waiting. A texture and each of its views share the pixels, so they are compared
    /// by the texture that owns them.
    /// </summary>
    private void MaterializePendingClearsOf(ReadOnlySpan<nint> textures)
    {
        if (_pendingClears.Count == 0 || _encodingPendingClears || textures.IsEmpty)
        {
            return;
        }

        Span<nint> roots = textures.Length <= 16 ? stackalloc nint[textures.Length] : new nint[textures.Length];
        for (var index = 0; index < textures.Length; index++)
        {
            // A dispatch's used resources include buffers (the step report), which have
            // no parentTexture; only textures can be, or be views of, a cleared texture.
            roots[index] = textures[index] == 0 || !IsTexture(textures[index]) ? 0 : RootTexture(textures[index]);
        }

        _encodingPendingClears = true;
        try
        {
            var kept = 0;
            for (var index = 0; index < _pendingClears.Count; index++)
            {
                var pending = _pendingClears[index];
                if (!roots.Contains(pending.Root))
                {
                    _pendingClears[kept++] = pending;
                    continue;
                }

                EncodeClearPass(
                    pending.Texture, pending.Range, pending.Width, pending.Height, pending.Color);
                MetalNative.SendVoid(pending.Texture, SelRelease);
            }

            _pendingClears.RemoveRange(kept, _pendingClears.Count - kept);
        }
        finally
        {
            _encodingPendingClears = false;
        }
    }

    /// <summary>
    /// Encodes the waiting clears a compute dispatch about to be recorded can see: the
    /// ones of every texture its bindings name, as a sampled or a storage image. Its
    /// buffers and device addresses reach guest memory, which a clear only reaches
    /// through the writeback of its texture - a transfer that names the texture and so
    /// encodes the clear first. Outside a resource preparation the bindings are not
    /// known, and every clear is encoded.
    /// </summary>
    private void MaterializePendingClearsForCompute()
    {
        if (_pendingClears.Count == 0)
        {
            return;
        }

        if (_preparation is null)
        {
            MaterializePendingClears();
            return;
        }

        MaterializePendingClearsOf(new ReadOnlySpan<nint>(_usedResources, 0, _usedResourceCount));
    }

    // Whether a clear still waiting for its pass is of this texture or what it views.
    private bool PendingClearTouches(nint texture)
    {
        if (_pendingClears.Count == 0)
        {
            return false;
        }

        var root = RootTexture(texture);
        foreach (ref readonly var pending in System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_pendingClears))
        {
            if (pending.Root == root)
            {
                return true;
            }
        }

        return false;
    }

    // The texture a view was made from, or the texture itself.
    private static nint RootTexture(nint texture)
    {
        for (var depth = 0; depth < 4 && texture != 0; depth++)
        {
            var parent = MetalNative.Send(texture, SelParentTexture);
            if (parent == 0)
            {
                break;
            }

            texture = parent;
        }

        return texture;
    }

    private static readonly nint SelParentTexture = MetalNative.Selector("parentTexture");

    private static bool IsTexture(nint resource) =>
        MetalNative.SendBool(resource, SelRespondsToSelector, SelParentTexture);

    private static readonly nint SelRespondsToSelector = MetalNative.Selector("respondsToSelector:");

    // DB_DEPTH_CONTROL.DEPTH_BOUNDS_ENABLE: the DB discards a fragment whose stored depth
    // lies outside [DB_DEPTH_BOUNDS_MIN, DB_DEPTH_BOUNDS_MAX], both ends inclusive, the
    // registers being raw floats the DB compares as written. Metal's encoder test is the
    // same compare on the stored depth but takes only bounds within [0, 1] with min <= max.
    // A stored depth is itself within [0, 1], so a bound beyond that range compares the
    // same as the range's end; a range that holds no stored depth at all (min > max after
    // that, which includes min > 1 and max < 0) discards every fragment, which the encoder
    // cannot express; the executor drops such a draw when that leaves nothing observable,
    // so one reaching here stores to memory or clears.
    private void ApplyDepthBounds(MetalPipelineEntry entry)
    {
        var minBound = 0f;
        var maxBound = 1f;
        if (_dynamicState.DepthBoundsTestEnabled && entry.HasDepth && !MetalV2Diagnostics.IgnoreDepthBounds)
        {
            var guestMin = _dynamicState.DepthMinBounds;
            var guestMax = _dynamicState.DepthMaxBounds;
            if (float.IsNaN(guestMin) || float.IsNaN(guestMax))
            {
                throw Fatal($"A draw tests depth bounds that are not numbers: min={guestMin} max={guestMax}.");
            }

            minBound = Math.Max(guestMin, 0f);
            maxBound = Math.Min(guestMax, 1f);
            if (minBound > maxBound)
            {
                throw Fatal(
                    $"A draw that stores to memory or clears tests depth bounds that hold no stored depth, which " +
                    $"discards every fragment and which Metal's depth bounds test cannot express: " +
                    $"min={guestMin} max={guestMax}.");
            }
        }

        if (_boundDepthMinBound == minBound && _boundDepthMaxBound == maxBound)
        {
            return;
        }

        if (!DepthBoundsTestSupported)
        {
            // Documented deviation: before Apple10 the device has no depth bounds test, and
            // refusing would stop every title that bounds its light and shadow passes (UE4).
            // Such a draw shades every fragment its other tests pass, a superset of the
            // hardware's; an empty range was already dropped exactly by the executor.
            MetalV2Diagnostics.WarnOnce(
                "depth-bounds-unsupported",
                $"This Metal device has no depth bounds test (GPU family Apple10); depth-bounded draws run " +
                $"without it: first min={minBound} max={maxBound}.");
            return;
        }

        MetalV2Native.SendDepthTestBounds(_renderEncoder, SelSetDepthTestBounds, minBound, maxBound);
        _boundDepthMinBound = minBound;
        _boundDepthMaxBound = maxBound;
    }

    /// <summary>
    /// Hands the pass about to begin every waiting clear it can perform itself, as a
    /// load action over the same pixels, and encodes the ones it cannot.
    /// </summary>
    private bool FoldPendingClearsInto(in RenderingState state)
    {
        Array.Clear(_overrideClear);
        if (_pendingClears.Count == 0)
        {
            return false;
        }

        var folded = false;
        for (var index = _pendingClears.Count - 1; index >= 0; index--)
        {
            var pending = _pendingClears[index];
            var attachment = MatchingAttachment(in state, in pending);
            if (attachment < 0)
            {
                continue;
            }

            if (!state.ColorAttachments[attachment].IsClear && !_overrideClear[attachment])
            {
                // The pass clears nothing here, so it clears this instead. The walk runs
                // newest first, so the clear that was asked for last is the one that takes
                // the attachment and an older clear of the same pixels is only dropped —
                // which is what it would have been, overwritten where it lay. A pass that
                // clears this attachment itself supersedes all of them the same way.
                _overrideClear[attachment] = true;
                _overrideClearColor[attachment] = new MtlClearColor
                {
                    Red = pending.Color.Float32_0,
                    Green = pending.Color.Float32_1,
                    Blue = pending.Color.Float32_2,
                    Alpha = pending.Color.Float32_3,
                };
                folded = true;
            }

            MetalNative.SendVoid(pending.Texture, SelRelease);
            _pendingClears.RemoveAt(index);
            Interlocked.Increment(ref _clearsFolded);
        }

        return folded;
    }

    /// <summary>
    /// The colour attachment of the pass about to begin that a waiting clear covers
    /// exactly. Exactly is the whole rule: a load action clears the render area, so it
    /// can only stand in for a clear of the same pixels — one subresource, the same mip,
    /// the same layer, and a render area that is the whole of it.
    /// </summary>
    private int MatchingAttachment(in RenderingState state, in PendingClear pending)
    {
        for (var index = 0; index < state.ColorAttachmentCount && index < MaxColorAttachments; index++)
        {
            if (_attachmentTextures[index] == pending.Texture &&
                _attachmentRealLevels[index] == pending.Range.BaseLevel &&
                _attachmentRealLayers[index] == pending.Range.BaseLayer &&
                MetalPassStores.ClearIsExactlyTheAttachment(
                    pending.Range.LevelCount,
                    pending.Range.LayerCount,
                    state.Layers,
                    state.Width,
                    state.Height,
                    pending.Width,
                    pending.Height,
                    _attachmentWidths[index],
                    _attachmentHeights[index]))
            {
                return index;
            }
        }

        return -1;
    }

    // One load-clear render pass with no draws: the cheapest full clear in Metal.
    private void EncodeClearPass(
        nint texture,
        in SubresourceRange range,
        uint width,
        uint height,
        in GpuClearColor color)
    {
        Interlocked.Increment(ref _clearsEncoded);
        _clearPasses++;
        _clearPixels += (long)Math.Max(width, 1) * Math.Max(height, 1) * Math.Max(range.LayerCount, 1);
        NoteClearAboutToBeEncoded(texture, in range, width, height, depth: false, stencil: false);
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.ColorClear, texture);
        var commandBuffer = context.CommandBuffer;
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLRenderPassDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        try
        {
            var colors = MetalNative.Send(descriptor, SelColorAttachments);
            var attachment = MetalNative.SendAtIndex(colors, SelObjectAtIndex, 0);
            ConfigureClearAttachment(attachment, texture, in range);
            MetalNative.SendVoidClearColor(attachment, SelSetClearColor, new MtlClearColor
            {
                Red = color.Float32_0,
                Green = color.Float32_1,
                Blue = color.Float32_2,
                Alpha = color.Float32_3,
            });
            MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetWidth, Math.Max(width, 1));
            MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetHeight, Math.Max(height, 1));
            if (range.LayerCount > 1)
            {
                MetalV2Native.SendVoidValue(descriptor, SelSetRenderTargetArrayLength, range.LayerCount);
            }

            var encoder = MetalNative.Send(commandBuffer, SelRenderEncoder, descriptor);
            MetalMemoryTelemetry.RenderEncoders.Add();
            if (encoder == 0)
            {
                MetalV2Diagnostics.WarnOnce("clear-encoder", "A Metal clear pass could not be encoded.");
                return;
            }

            WaitForPreviousEncoder(encoder, render: true);
            UpdateEncoderFence(encoder, render: true);
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded($"clear {width}x{height} layers={range.LayerCount}");
        }
    }

    private static void ConfigureClearAttachment(nint attachment, nint texture, in SubresourceRange range)
    {
        MetalNative.SendVoid(attachment, SelSetTexture, texture);
        MetalV2Native.SendVoidValue(attachment, SelSetLevel, range.BaseLevel);
        // A volume's layers are its depth planes, as on the attachments of a pass; its
        // slice index must stay zero. A DCC clear of one slice of a volume clears one plane.
        var volume = (ulong)MetalNative.Send(texture, SelTextureType) == Mtl.TextureType3D;
        MetalV2Native.SendVoidValue(attachment, volume ? SelSetDepthPlane : SelSetSlice, range.BaseLayer);
        MetalV2Native.SendVoidValue(attachment, SelSetLoadAction, Mtl.LoadActionClear);
        MetalV2Native.SendVoidValue(attachment, SelSetStoreAction, Mtl.StoreActionStore);
    }

    public bool TryRetainTargetlessDraw(RegisterBanks banks, GraphicsPrograms programs, in TargetlessDrawArguments arguments)
    {
        _ = banks;
        _ = programs;
        _ = arguments;
        // v2 renders a targetless draw as it is; nothing is retained for a later flip.
        return false;
    }
}
