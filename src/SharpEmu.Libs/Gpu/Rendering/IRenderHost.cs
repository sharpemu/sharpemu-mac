// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Libs.Gpu.Rendering;

// A host buffer and the byte offset a guest address maps to inside it.
public readonly record struct BufferBinding(ulong Handle, ulong Offset);

// The packet arguments of a draw the host keeps until the next flip.
public readonly record struct TargetlessDrawArguments(ulong SubmitId, bool Indexed, GpuCommands.DrawIndexedArguments Indexed_, GpuCommands.DrawAutoArguments Auto);

public readonly record struct RenderHostLimits(uint MaxFramebufferWidth, uint MaxFramebufferHeight, uint MaxViewportWidth, uint MaxViewportHeight);

public readonly record struct ColorAttachmentAcquisition(
    ResourceSlotIdentifier Image,
    GpuImageHandle View,
    GpuImageLayout Layout,
    uint Samples,
    bool MetadataClear,
    GpuClearColor MetadataClearValue);

public readonly record struct DepthAttachmentAcquisition(GpuImageHandle View, uint Samples, bool MetadataClear);

public readonly record struct ScissorRectangle(int Left, int Top, int Right, int Bottom)
{
    public bool IsSet => Left != 0 || Top != 0 || Right != 0 || Bottom != 0;

    public bool IsValid => Right > Left && Bottom > Top;

    public ScissorRectangle Offset(int x, int y) => new(Left + x, Top + y, Right + x, Bottom + y);

    public ScissorRectangle Intersect(in ScissorRectangle other) =>
        new(Math.Max(Left, other.Left), Math.Max(Top, other.Top), Math.Min(Right, other.Right), Math.Min(Bottom, other.Bottom));
}

// The state the host records before every draw.
public readonly record struct DynamicDrawState(
    float ViewportX,
    float ViewportY,
    float ViewportWidth,
    float ViewportHeight,
    float ViewportMinDepth,
    float ViewportMaxDepth,
    ScissorRectangle Scissor,
    float LineWidth,
    float BlendRed,
    float BlendGreen,
    float BlendBlue,
    float BlendAlpha,
    bool DepthTestEnabled,
    bool DepthWriteEnabled,
    GpuCompareOp DepthCompare,
    bool DepthBiasEnabled,
    float DepthBiasConstantFactor,
    float DepthBiasClamp,
    float DepthBiasSlopeFactor,
    bool DepthBoundsTestEnabled,
    float DepthMinBounds,
    float DepthMaxBounds,
    bool StencilTestEnabled,
    StencilMasks FrontStencil,
    StencilMasks BackStencil,
    uint ColorWriteCount,
    byte ColorWriteEnableMask,
    // Set when front- and back-facing triangles take different biases: DepthBiasConstantFactor
    // and DepthBiasSlopeFactor then bias the front faces only, and these two the back faces.
    bool DepthBiasPerFace = false,
    float DepthBiasBackConstantFactor = 0f,
    float DepthBiasBackSlopeFactor = 0f,
    // The bias of points and lines, which take the front offset under POLY_OFFSET_PARA_ENABLE
    // whatever the triangle faces take.
    float DepthBiasPointLineConstantFactor = 0f,
    float DepthBiasPointLineSlopeFactor = 0f,
    uint DepthCountControl = 0);

// The host-side descriptors of one shader stage, prepared before the draw or dispatch records.
public interface IPreparedBindings
{
    ShaderStageResources Stage { get; }
}

// Keeps every prepared binding and the stream ring contents intact until the draw or dispatch is recorded.
public interface IResourcePreparation : IDisposable
{
}

// Everything the executor needs from the renderer, the caches and the pipeline objects.
public interface IRenderHost
{
    int BeginImageSnapshot(IPreparedBindings? pixelBindings) => 0;

    int BeginComputeImageSnapshot(IPreparedBindings bindings) => 0;

    void EndComputeImageSnapshot(int sequence, IPreparedBindings bindings) { }

    void EndImageSnapshot(int sequence, ReadOnlySpan<ColorTargetState> targets) { }

    // Records a draw's color-target base for the image-clear trace; see docs/image-clear-tracing.md.
    void TraceDrawTarget(string disposition, ulong address, ulong pixelShaderAddress, ulong vertexShaderAddress) { }

    RenderHostLimits Limits { get; }

    bool SupportsNativeRectangles => false;

    IImageFormatSupport FormatSupport { get; }

    // True while a command buffer is recording.
    bool IsRecording { get; }

    void RunPendingOperations();

    void SetDebugInformation(RecordedOperation operation, ulong submitId, uint argument0, uint argument1, uint argument2, uint argument3, ulong argument4);

    bool TryReadGuest(ulong address, Span<byte> destination);

    // The part of the range that is mapped from its start; zero when the start is unmapped.
    ulong ClampMappedSize(ulong address, ulong size);

    ResourceSlotIdentifier FindImage(ref ImageRequest request, bool exactFormat);

    void BindRenderTarget(ResourceSlotIdentifier image);

    void ResetBindings();

    // Finds the image again when it changed, makes the view, sets the layout and resolves a metadata clear.
    ColorAttachmentAcquisition AcquireColorAttachment(in ColorTargetState target);

    // Makes the view and consumes the metadata clear state; the layout follows in a second call.
    DepthAttachmentAcquisition AcquireDepthAttachment(in DepthAttachmentState depth);

    void TransitionDepthAttachment(in DepthAttachmentState depth, GpuImageLayout layout, GpuImageAspect writeAspects);

    BufferBinding NullBuffer { get; }

    BufferBinding ObtainBuffer(ulong address, ulong size, bool isWritten);

    void TraceVertexBuffer(ulong shaderHash, VertexAttributeResource attribute, VertexInputBuffer input,
        BufferBinding binding, ulong requestedBytes, ulong acquiredBytes) { }

    // Copies host bytes into the stream ring for the current recording.
    BufferBinding UploadTransient(ReadOnlySpan<byte> data, uint alignment);

    ulong UploadMeshVertexIndices(ReadOnlySpan<uint> indices) =>
        throw new NotSupportedException("The renderer does not support indexed mesh inputs.");

    bool SupportsIndirectMeshDraw => false;

    IIndirectMeshDraw PrepareIndirectMeshDraw(in IndirectMeshDrawRequest request) =>
        throw new NotSupportedException("The renderer does not support indirect mesh inputs.");

    void BindVertexBuffers(ReadOnlySpan<BufferBinding> bindings, VertexInputInfo input);

    void BindIndexBuffer(BufferBinding binding, GpuIndexType type);

    // The guest address and byte size the index buffer just bound was read from; address 0
    // when the indices are host data (an upload or an expanded primitive-restart stream) and
    // so name no guest range. A backend that derives per-range data from the indices uses it.
    void NoteIndexSource(ulong guestAddress, ulong size)
    {
    }

    // A draw or dispatch whose program reaches guest memory through the device-address page
    // table is about to be prepared: called before BeginPreparation, so a backend can settle
    // guest memory such a program may reach (and end its command buffer) while that is safe.
    // The stages carry the programs and the device-address ranges evaluated for this draw.
    void BeforeDeviceAddressPreparation(ReadOnlySpan<ShaderStageResources> stages)
    {
    }

    // Opens the preparation scope; a ring wrap inside it is refused, and disposal releases it on every path.
    IResourcePreparation BeginPreparation();

    IPreparedBindings PrepareBindings(ShaderStageResources stage);

    void PrepareDeviceAddresses();

    // True when a draw or dispatch whose pipeline is still compiling may be left for
    // later: the packet suspends and runs again once the compile finishes, so the
    // command stream serves its other queues meanwhile. A host that cannot do that
    // keeps waiting for the compile in place, because a one-shot dispatch that is
    // skipped never runs, and a later packet may be waiting on what it would write.
    bool SupportsPipelineSuspend => false;

    // The last draw or dispatch could not run: its pipeline is still compiling.
    void NotePipelineNotReady()
    {
    }

    // True when a draw whose pipeline is still compiling may be left out of the frame instead
    // of waiting for the compile (RenderExecutor.MayLeaveOutWhileCompiling decides which).
    bool LeavesOutDrawsWhileCompiling => false;

    // The host's part of that decision, for the attachments the draw has just acquired: true
    // only for targets it knows are rebuilt from nothing every frame, so that the frame being
    // recorded is the only one that misses the draw.
    bool MayLeaveOutWhileCompiling(in RenderingState rendering) => false;

    void BindResources(IPreparedBindings prepared);

    void CommitBindings(GpuBindPoint bindPoint, in PipelineHandle pipeline, ReadOnlySpan<IPreparedBindings> stages);

    void SetDynamicState(in DynamicDrawState state);

    void BeginRendering(in RenderingState state);

    void EndRendering();

    // Ends the pass for a stated reason ("release", "barrier" or "dispatch"), for a host that
    // reports why passes ended; the others just end it.
    void EndRendering(string why) => EndRendering();

    // Whether a texture bound for the draw being prepared reads the depth attachment's subresources.
    bool SamplesDepthAttachment(in DepthAttachmentState depth) => true;

    // The next draw stores to buffers or storage images; called before its BeginRendering.
    void PrepareMemoryWritingDraw() { }

    // The first host draw call the next BeginRendering is for, so a host can choose its
    // render pass's load actions from what that draw covers: its pipeline and, for a direct
    // non-indexed draw, the vertices it draws. vertexCount is zero when the draw is indexed
    // or indirect and its vertices are not known here.
    void NoteUpcomingDraw(in PipelineHandle pipeline, uint vertexCount, uint firstVertex) { }

    void BindPipeline(GpuBindPoint bindPoint, in PipelineHandle pipeline);

    void Draw(uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance);

    void DrawIndexed(uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint firstInstance);

    // One indexed draw whose counts the GPU reads from the buffer (VkDrawIndexedIndirectCommand layout).
    void DrawIndexedIndirect(BufferBinding arguments) =>
        throw new NotSupportedException("The render host does not draw from indirect arguments.");

    // Called before the BeginRendering of a draw whose counts the GPU reads from the buffer
    // (guest address argumentsAddress), so a host that must turn them into a launch of its
    // own (a mesh grid) can encode that work where no pass is open. indexLimit bounds an
    // indexed draw's count, as the bound index range does.
    void PrepareIndirectDraw(in PipelineHandle pipeline, ulong argumentsAddress, BufferBinding arguments, bool indexed, uint indexLimit) { }

    // One non-indexed draw whose counts the GPU reads from the buffer (VkDrawIndirectCommand layout).
    void DrawIndirect(BufferBinding arguments) =>
        throw new NotSupportedException("The render host does not draw from indirect arguments.");

    // True when the host can hand this vertex program an indirect draw with its arguments
    // (argumentsSize bytes at argumentsAddress) still in guest memory. A host that runs the
    // program as a mesh pipeline sizes the launch grid from the counts; unless it builds that
    // grid on the GPU in PrepareIndirectDraw, it answers false and the executor resolves such
    // a draw from guest memory before the pass begins. No draw is dropped either way.
    bool TakesIndirectDrawArguments(ShaderProgram vertexProgram, ulong argumentsAddress, ulong argumentsSize) => true;

    // True when a targetless indirect draw that cannot be retained for replay may keep its
    // counts on the GPU; otherwise every targetless indirect draw is resolved first.
    bool TakesTargetlessIndirectDrawArguments => false;

    // Called before the executor reads an indirect draw's arguments on the CPU; a host may count it.
    void NoteIndirectArgumentsReadOnCpu(ulong address, ulong size) { }

    // The same question for a rectangle list. A host that draws each rectangle as one
    // patch sizes the draw from the index count, which an indirect patch draw cannot
    // read in the guest's argument layout, so the executor resolves such a draw first.
    bool TakesIndirectRectangleListArguments => true;

    // True when the host draws a GpuPrimitiveTopology.TriangleFan pipeline as fans. A host
    // without fans answers false: the executor then draws triangle fans, polygons and quad
    // lists as indexed triangle lists it builds from the draw's vertices or indices, which
    // needs their counts, so such an indirect draw is resolved on the CPU first.
    bool DrawsTriangleFans => true;

    // The same for a non-indexed rectangle list with this vertex input; a host may build
    // its patch draw on the GPU in PrepareIndirectDraw.
    bool TakesIndirectAutoRectangleListArguments(VertexInputInfo vertexInput, ulong argumentsAddress, ulong argumentsSize) =>
        TakesIndirectRectangleListArguments;

    // The same for an indexed rectangle list with this vertex input, whose count the bound
    // index range limits to indexLimit (uint.MaxValue when unbounded).
    bool TakesIndirectIndexedRectangleListArguments(
        VertexInputInfo vertexInput, ulong argumentsAddress, ulong argumentsSize, uint indexLimit) =>
        TakesIndirectRectangleListArguments;

    void InsertDrawTraceMarker(string label) { }

    void DrawMeshTasks(uint groupCountX, uint groupCountY, uint groupCountZ) =>
        throw new NotSupportedException("The render host does not support mesh shaders.");

    void Dispatch(uint groupsX, uint groupsY, uint groupsZ);

    // Executes a dispatch from three uint32 group counts in guest memory when the backend can
    // consume the guest buffer directly. Backends without native indirect dispatch return false.
    bool TryDispatchIndirect(ulong argumentsAddress);

    // Orders the buffer writes of the given shader stages before every later access.
    void ShaderWriteBarrier(GpuPipelineStage sourceStages);

    // True when ShaderWriteBarrier orders vertex-stage buffer writes inside the open render
    // pass, so a draw whose vertex shader writes buffers need not end the pass. Fragment-stage
    // writes always end it: a tile-based GPU cannot make them visible mid-pass.
    bool OrdersVertexWritesInsidePass => false;

    // True when a draw whose fragment stage stores to memory may leave the render pass open:
    // the executor then ends the pass and issues the fragment barrier only when a consumer of
    // the stores can appear (a dispatch, an indirect draw, or the guest's own cache flush or
    // acquire), instead of after every such draw.
    bool DefersFragmentWritePassEnd => false;

    // Advances whenever an open render pass ends, whoever ends it. A host that defers fragment
    // write pass ends reports it, so the executor knows whether the pass that stored is still open.
    long RenderPassSerial => 0;

    void ShaderWriteHazardBarrier();

    void ShaderAccessBarrier();

    // Clears the bound color targets to one colour in place of the draw.
    void ClearColorTargets(ReadOnlySpan<ColorTargetState> targets, SolidColorClear clear);

    // Keeps a draw without targets for the next flip; false leaves the draw to run as it is.
    bool TryRetainTargetlessDraw(GpuCommands.Registers.RegisterBanks banks, GraphicsPrograms programs, in TargetlessDrawArguments arguments);

    void MarkGpuWritten(ResourceSlotIdentifier image);

    void ResolveImage(ResourceSlotIdentifier source, uint sourceMip, uint sourceLayer, ResourceSlotIdentifier destination, uint destinationMip, uint destinationLayer);

    void CopyDepthStencilImage(ResourceSlotIdentifier source, ResourceSlotIdentifier destination, in SubresourceRange range, in GpuExtent3D extent, GpuImageAspect aspects);

    bool IsMetadata(ulong address);

    // Whether a buffer binding of this size at the address spans the whole metadata surface
    // registered there, as a clear of it does. A small buffer that merely starts where a
    // surface's metadata once lived (transient memory reused for counters) does not.
    bool CoversMetadata(ulong address, ulong size) => IsMetadata(address);

    bool ClearMetadata(ulong address);

    bool TryClearImageFromBuffer(ulong address, ulong size, uint packedClear);

    bool TryAbsorbDccFill(ulong address, ulong size, uint fillValue);

    bool TryFillDccMetadata(ulong address, ulong size, uint fillValue);

    bool TryCopyWordsOnHost(ulong destination, ulong source, ulong sourceWords, ulong words);

    // A compute fill of repeating dwords over [address, address + size), the pattern starting at
    // address, that a host may do as an image clear instead; false leaves the dispatch to run.
    bool TryClearImageFromKernelFill(ulong address, ulong size, ReadOnlySpan<uint> pattern) => false;

    // A recognized fill or copy kernel was left to run as a dispatch, and why (a short key).
    void NoteKernelFillFallback(string reason) { }

    Exception Fatal(string message);
}
