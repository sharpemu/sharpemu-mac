// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

// The Metal ABI constants v2 names. Declared locally for the same reason
// MetalNative is: the backend talks to the OS through objc_msgSend only, so no
// binding package owns these values.
internal static class Mtl
{
    // MTLResourceOptions. The storage mode occupies bits 4..7, the hazard mode bits 8..9.
    public const nuint ResourceStorageModeShared = 0 << 4;
    public const nuint ResourceStorageModeManaged = 1 << 4;
    public const nuint ResourceStorageModePrivate = 2 << 4;
    public const nuint ResourceHazardTrackingModeUntracked = 2 << 8;

    // MTLStorageMode, as MTLTextureDescriptor.storageMode takes it.
    public const nuint StorageModeShared = 0;
    public const nuint StorageModePrivate = 2;
    public const nuint StorageModeMemoryless = 3;

    // MTLTextureType.
    public const ulong TextureType1D = 0;
    public const ulong TextureType1DArray = 1;
    public const ulong TextureType2D = 2;
    public const ulong TextureType2DArray = 3;
    public const ulong TextureType2DMultisample = 4;
    public const ulong TextureTypeCube = 5;
    public const ulong TextureType3D = 7;
    public const ulong TextureType2DMultisampleArray = 8;

    // MTLTextureUsage.
    public const ulong TextureUsageUnknown = 0;
    public const ulong TextureUsageShaderRead = 1;
    public const ulong TextureUsageShaderWrite = 2;
    public const ulong TextureUsageRenderTarget = 4;
    public const ulong TextureUsagePixelFormatView = 16;

    // MTLLoadAction / MTLStoreAction.
    public const nuint LoadActionDontCare = 0;
    public const nuint LoadActionLoad = 1;
    public const nuint LoadActionClear = 2;
    public const nuint StoreActionDontCare = 0;
    public const nuint StoreActionStore = 1;
    public const nuint StoreActionStoreAndMultisampleResolve = 3;

    // A descriptor that asks for Unknown promises to name the real action on the
    // encoder before encoding ends, which is how a store can be discarded once the
    // work that follows the pass is known.
    public const nuint StoreActionUnknown = 4;

    // MTLPrimitiveType.
    public const nuint PrimitivePoint = 0;
    public const nuint PrimitiveLine = 1;
    public const nuint PrimitiveLineStrip = 2;
    public const nuint PrimitiveTriangle = 3;
    public const nuint PrimitiveTriangleStrip = 4;

    // MTLIndexType.
    public const nuint IndexTypeUInt16 = 0;
    public const nuint IndexTypeUInt32 = 1;

    // MTLBlendOperation.
    public const nint BlendOperationAdd = 0;
    public const nint BlendOperationSubtract = 1;
    public const nint BlendOperationReverseSubtract = 2;
    public const nint BlendOperationMin = 3;
    public const nint BlendOperationMax = 4;

    // MTLColorWriteMask. Metal orders the bits alpha-first.
    public const nint ColorWriteMaskNone = 0;
    public const nint ColorWriteMaskAlpha = 1 << 0;
    public const nint ColorWriteMaskBlue = 1 << 1;
    public const nint ColorWriteMaskGreen = 1 << 2;
    public const nint ColorWriteMaskRed = 1 << 3;

    // MTLWinding / MTLCullMode / MTLTriangleFillMode / MTLDepthClipMode.
    public const nint WindingClockwise = 0;
    public const nint WindingCounterClockwise = 1;
    public const nint CullModeNone = 0;
    public const nint CullModeFront = 1;
    public const nint CullModeBack = 2;
    public const nint DepthClipModeClip = 0;
    public const nint DepthClipModeClamp = 1;

    // MTLBarrierScope and MTLRenderStages.
    public const nuint BarrierScopeBuffers = 1;
    public const nuint BarrierScopeTextures = 2;
    public const nuint BarrierScopeRenderTargets = 4;
    public const nuint RenderStageVertex = 1;
    public const nuint RenderStageFragment = 2;
    public const nuint RenderStageTile = 4;
    public const nuint RenderStageObject = 8;
    public const nuint RenderStageMesh = 16;

    // MTLResourceUsage.
    public const nuint ResourceUsageRead = 1;
    public const nuint ResourceUsageWrite = 2;

    // MTLSampler* enumerations.
    public const nint SamplerFilterNearest = 0;
    public const nint SamplerFilterLinear = 1;
    public const nint SamplerMipFilterNotMipmapped = 0;
    public const nint SamplerMipFilterNearest = 1;
    public const nint SamplerMipFilterLinear = 2;
    public const nint SamplerAddressClampToEdge = 0;
    public const nint SamplerAddressMirrorClampToEdge = 1;
    public const nint SamplerAddressRepeat = 2;
    public const nint SamplerAddressMirrorRepeat = 3;
    public const nint SamplerAddressClampToZero = 4;
    public const nint SamplerAddressClampToBorderColor = 5;
    public const nint SamplerBorderTransparentBlack = 0;
    public const nint SamplerBorderOpaqueBlack = 1;
    public const nint SamplerBorderOpaqueWhite = 2;

    // MTLVertexStepFunction.
    public const nint StepFunctionPerVertex = 1;
    public const nint StepFunctionPerInstance = 2;
    public const nint StepFunctionPerPatchControlPoint = 4;

    // Tessellation, which a rectangle list draws through (see Gen5MslRectangleList):
    // MTLTessellationFactorStepFunction, MTLTessellationPartitionMode and
    // MTLTessellationControlPointIndexType.
    public const nuint TessellationFactorStepFunctionConstant = 0;
    public const nuint TessellationPartitionModeInteger = 1;
    public const nuint ControlPointIndexTypeNone = 0;
    public const nuint ControlPointIndexTypeUInt16 = 1;
    public const nuint ControlPointIndexTypeUInt32 = 2;

    // MTLMutability, for pipeline buffer arguments the shader only reads.
    public const nuint MutabilityImmutable = 1;
}

// MTLVertexFormat, only the formats a guest vertex buffer descriptor maps to.
internal enum MtlVertexFormat : uint
{
    Invalid = 0,
    UChar2 = 1,
    UChar3 = 2,
    UChar4 = 3,
    Char2 = 4,
    Char3 = 5,
    Char4 = 6,
    UChar2Normalized = 7,
    UChar3Normalized = 8,
    UChar4Normalized = 9,
    Char2Normalized = 10,
    Char3Normalized = 11,
    Char4Normalized = 12,
    UShort2 = 13,
    UShort3 = 14,
    UShort4 = 15,
    Short2 = 16,
    Short3 = 17,
    Short4 = 18,
    UShort2Normalized = 19,
    UShort3Normalized = 20,
    UShort4Normalized = 21,
    Short2Normalized = 22,
    Short3Normalized = 23,
    Short4Normalized = 24,
    Half2 = 25,
    Half3 = 26,
    Half4 = 27,
    Float = 28,
    Float2 = 29,
    Float3 = 30,
    Float4 = 31,
    Int = 32,
    Int2 = 33,
    Int3 = 34,
    Int4 = 35,
    UInt = 36,
    UInt2 = 37,
    UInt3 = 38,
    UInt4 = 39,
    Int1010102Normalized = 40,
    UInt1010102Normalized = 41,
    UChar = 45,
    Char = 46,
    UCharNormalized = 47,
    CharNormalized = 48,
    UShort = 49,
    Short = 50,
    UShortNormalized = 51,
    ShortNormalized = 52,
    Half = 53,
    FloatRG11B10 = 54,
    FloatRGB9E5 = 55,
}

// The depth and stencil pixel formats v2 uses on top of MtlPixelFormat.
internal static class MtlDepthFormats
{
    public const ulong Depth16Unorm = 250;
    public const ulong Depth32Float = 252;
    public const ulong Stencil8 = 253;
    public const ulong Depth32FloatStencil8 = 260;
}

// The objc_msgSend signatures v2 needs beyond MetalNative's. Kept in one place
// so every native call in the backend goes through a declared ABI shape, and so
// the selector identities can be cached as static readonly fields by the callers.
internal static partial class MetalV2Native
{
    private const string ObjCLibrary = "/usr/lib/libobjc.A.dylib";

    // newBufferWithBytesNoCopy:length:options:deallocator: — a nil deallocator block
    // leaves the host pages to their owner, which is what a guest-memory window needs.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendNewBufferNoCopy(
        nint receiver,
        nint selector,
        nint pointer,
        nuint length,
        nuint options,
        nint deallocator);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendWithValue(nint receiver, nint selector, nuint value);

    // -[MTLDevice sparseTileSizeWithTextureType:pixelFormat:sampleCount:] returns a 24-byte
    // MTLSize. x86_64 returns a struct that size through a hidden result pointer passed
    // first, which only objc_msgSend_stret expects: objc_msgSend takes that pointer for the
    // receiver and messages the stack (the Demon's Souls GPU-worker crashes under Rosetta).
    // arm64 returns it through x8 and has objc_msgSend alone.
    public static MtlSize SendSparseTileSize(
        nint receiver, nint selector, nuint textureType, nuint pixelFormat, nuint sampleCount)
    {
        if (RuntimeInformation.ProcessArchitecture == Architecture.X64)
        {
            SendSparseTileSizeStret(out var size, receiver, selector, textureType, pixelFormat, sampleCount);
            return size;
        }

        return SendSparseTileSizeArm64(receiver, selector, textureType, pixelFormat, sampleCount);
    }

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend_stret")]
    private static partial void SendSparseTileSizeStret(
        out MtlSize result, nint receiver, nint selector, nuint textureType, nuint pixelFormat, nuint sampleCount);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial MtlSize SendSparseTileSizeArm64(
        nint receiver, nint selector, nuint textureType, nuint pixelFormat, nuint sampleCount);

    // -[MTLResourceStateCommandEncoder updateTextureMapping:mode:region:mipLevel:slice:]
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendUpdateTextureMapping(
        nint receiver, nint selector, nint texture, nuint mode, MtlRegion region, nuint mipLevel, nuint slice);

    // Ties one object's lifetime to another's: OBJC_ASSOCIATION_RETAIN releases the value
    // when the object is deallocated.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_setAssociatedObject")]
    public static partial void SetAssociatedObject(nint obj, nint key, nint value, nuint policy);

    // A selector taking one NSRange, which the ABI passes as its two words.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendRange(nint receiver, nint selector, nuint location, nuint length);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidValue(nint receiver, nint selector, nuint value);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidValue64(nint receiver, nint selector, ulong value);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial ulong SendValue64(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nuint SendValue(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendWithError(nint receiver, nint selector, nint argument, ref nint error);

    // -[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:]
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendNewBufferTexture(
        nint receiver,
        nint selector,
        nint descriptor,
        nuint offset,
        nuint bytesPerRow);

    // -newTextureViewWithPixelFormat:textureType:levels:slices:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendNewTextureView(
        nint receiver,
        nint selector,
        nuint pixelFormat,
        nuint textureType,
        NsRange levels,
        NsRange slices);

    // -newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendNewSwizzledTextureView(
        nint receiver,
        nint selector,
        nuint pixelFormat,
        nuint textureType,
        NsRange levels,
        NsRange slices,
        MtlTextureSwizzleChannels swizzle);

    // -drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:
    //  instanceCount:baseVertex:baseInstance:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawIndexedBaseVertex(
        nint receiver,
        nint selector,
        nuint primitiveType,
        nuint indexCount,
        nuint indexType,
        nint indexBuffer,
        nuint indexBufferOffset,
        nuint instanceCount,
        nint baseVertex,
        nuint baseInstance);

    // -drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawPrimitivesBaseInstance(
        nint receiver,
        nint selector,
        nuint primitiveType,
        nuint vertexStart,
        nuint vertexCount,
        nuint instanceCount,
        nuint baseInstance);

    // -drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    //  instanceCount:baseInstance:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawPatches(
        nint receiver,
        nint selector,
        nuint numberOfPatchControlPoints,
        nuint patchStart,
        nuint patchCount,
        nint patchIndexBuffer,
        nuint patchIndexBufferOffset,
        nuint instanceCount,
        nuint baseInstance);

    // -drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    //  controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawIndexedPatches(
        nint receiver,
        nint selector,
        nuint numberOfPatchControlPoints,
        nuint patchStart,
        nuint patchCount,
        nint patchIndexBuffer,
        nuint patchIndexBufferOffset,
        nint controlPointIndexBuffer,
        nuint controlPointIndexBufferOffset,
        nuint instanceCount,
        nuint baseInstance);

    // -drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:
    //  threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawMeshIndirect(
        nint receiver,
        nint selector,
        nint indirectBuffer,
        nuint indirectBufferOffset,
        MtlSize threadsPerObjectThreadgroup,
        MtlSize threadsPerMeshThreadgroup);

    // -drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawPatchesIndirect(
        nint receiver,
        nint selector,
        nuint numberOfPatchControlPoints,
        nint patchIndexBuffer,
        nuint patchIndexBufferOffset,
        nint indirectBuffer,
        nuint indirectBufferOffset);

    // -drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:
    //  controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawIndexedPatchesIndirect(
        nint receiver,
        nint selector,
        nuint numberOfPatchControlPoints,
        nint patchIndexBuffer,
        nuint patchIndexBufferOffset,
        nint controlPointIndexBuffer,
        nuint controlPointIndexBufferOffset,
        nint indirectBuffer,
        nuint indirectBufferOffset);

    // -waitForFence:beforeStages:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendWaitForFence(nint receiver, nint selector, nint fence, nuint stages);

    // -drawPrimitives:indirectBuffer:indirectBufferOffset:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawIndirect(
        nint receiver,
        nint selector,
        nuint primitiveType,
        nint indirectBuffer,
        nuint indirectBufferOffset);

    // -drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:
    //  indirectBuffer:indirectBufferOffset:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawIndexedIndirect(
        nint receiver,
        nint selector,
        nuint primitiveType,
        nuint indexType,
        nint indexBuffer,
        nuint indexBufferOffset,
        nint indirectBuffer,
        nuint indirectBufferOffset);

    // -dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDispatchIndirect(
        nint receiver,
        nint selector,
        nint indirectBuffer,
        nuint indirectBufferOffset,
        MtlSize threadsPerThreadgroup);

    // -memoryBarrierWithScope:afterStages:beforeStages:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendMemoryBarrierStages(
        nint receiver,
        nint selector,
        nuint scope,
        nuint afterStages,
        nuint beforeStages);

    // -memoryBarrierWithScope: on a compute encoder.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendMemoryBarrierScope(nint receiver, nint selector, nuint scope);

    // -useResources:count:usage:stages: and -useResources:count:usage:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendUseResources(
        nint receiver,
        nint selector,
        nint resources,
        nuint count,
        nuint usage,
        nuint stages);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendUseResourcesNoStages(
        nint receiver,
        nint selector,
        nint resources,
        nuint count,
        nuint usage);

    // -setViewports:count: / -setScissorRects:count:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidArray(nint receiver, nint selector, nint items, nuint count);

    // -encodeSignalEvent:value: / -encodeWaitForEvent:value:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendEventValue(nint receiver, nint selector, nint @event, ulong value);

    // -waitUntilSignaledValue:timeoutMS:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static partial bool SendWaitUntilSignaledValue(
        nint receiver,
        nint selector,
        ulong value,
        ulong timeoutMilliseconds);

    // -setDepthBias:slopeScale:clamp:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDepthBias(
        nint receiver,
        nint selector,
        float depthBias,
        float slopeScale,
        float clamp);

    // -setDepthTestMinBound:maxBound:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDepthTestBounds(nint receiver, nint selector, float minBound, float maxBound);

    // -setStencilFrontReferenceValue:backReferenceValue:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendStencilReferences(nint receiver, nint selector, uint front, uint back);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidFloatValue(nint receiver, nint selector, float value);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidUInt(nint receiver, nint selector, uint value);

    // -setVertexBytes:length:atIndex: and friends.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetBytes(
        nint receiver,
        nint selector,
        nint bytes,
        nuint length,
        nuint index);

    // -setColorStoreAction:atIndex: on a render encoder, which a descriptor's
    // MTLStoreActionUnknown defers to.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetStoreActionAtIndex(
        nint receiver,
        nint selector,
        nuint action,
        nuint index);

    // -setBufferOffset:atIndex:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetBufferOffset(nint receiver, nint selector, nuint offset, nuint index);

    // -setTexture:atIndex: / -setSamplerState:atIndex: on a compute or render encoder.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetResourceAtIndex(nint receiver, nint selector, nint resource, nuint index);

    // -copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:
    //  toTexture:destinationSlice:destinationLevel:destinationOrigin:options:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyBufferToTextureOptions(
        nint receiver,
        nint selector,
        nint sourceBuffer,
        nuint sourceOffset,
        nuint sourceBytesPerRow,
        nuint sourceBytesPerImage,
        MtlSize sourceSize,
        nint destinationTexture,
        nuint destinationSlice,
        nuint destinationLevel,
        MtlOrigin destinationOrigin,
        nuint options);

    // -copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:
    //  toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyTextureToBufferOptions(
        nint receiver,
        nint selector,
        nint sourceTexture,
        nuint sourceSlice,
        nuint sourceLevel,
        MtlOrigin sourceOrigin,
        MtlSize sourceSize,
        nint destinationBuffer,
        nuint destinationOffset,
        nuint destinationBytesPerRow,
        nuint destinationBytesPerImage,
        nuint options);
}
