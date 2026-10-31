// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Abstractions;

// The backend-neutral vocabulary of the shared GPU engine.
//
// Every member carries the numeric value of the identically named Vulkan enumerant, so the
// Vulkan backend converts with a plain cast (see Gpu/Vulkan/VulkanTypeConversions.cs) and the
// recorded Vulkan stream is unchanged. The NeutralGpuVocabularyParityTests architecture test
// keeps the two in step; only the members the engine actually names are listed.

// Matches the Vulkan enum VkImageType.
public enum GpuImageType
{
    Type1D = 0,
    Type2D = 1,
    Type3D = 2,
}

// Matches the Vulkan enum VkImageTiling.
public enum GpuImageTiling
{
    Optimal = 0,
    Linear = 1,
}

// Matches the Vulkan enum VkImageUsageFlags.
[Flags]
public enum GpuImageUsage
{
    None = 0,
    TransferSrcBit = 1,
    TransferDstBit = 2,
    SampledBit = 4,
    StorageBit = 8,
    ColorAttachmentBit = 16,
    DepthStencilAttachmentBit = 32,
}

// Matches the Vulkan enum VkImageCreateFlags.
[Flags]
public enum GpuImageCreateFlags
{
    None = 0,
    CreateMutableFormatBit = 8,
    CreateCubeCompatibleBit = 16,
    Create2DArrayCompatibleBit = 32,
    CreateBlockTexelViewCompatibleBit = 128,
    CreateExtendedUsageBit = 256,
}

// Matches the Vulkan enum VkSampleCountFlags.
[Flags]
public enum GpuSampleCount
{
    None = 0,
    Count1Bit = 1,
    Count2Bit = 2,
    Count4Bit = 4,
    Count8Bit = 8,
    Count16Bit = 16,
    Count32Bit = 32,
    Count64Bit = 64,
}

// Matches the Vulkan enum VkIndexType.
public enum GpuIndexType
{
    Uint16 = 0,
    Uint32 = 1,
}

// Matches the Vulkan enum VkImageLayout.
public enum GpuImageLayout
{
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    DepthStencilReadOnlyOptimal = 4,
    ShaderReadOnlyOptimal = 5,
    TransferSrcOptimal = 6,
    TransferDstOptimal = 7,

    // Promoted extension layouts keep their extension enumerant values.
    PresentSrcKhr = 1000001002,
    DepthReadOnlyStencilAttachmentOptimal = 1000117000,
    DepthAttachmentStencilReadOnlyOptimal = 1000117001,
    DepthAttachmentOptimal = 1000241000,
    DepthReadOnlyOptimal = 1000241001,
    StencilAttachmentOptimal = 1000241002,
    StencilReadOnlyOptimal = 1000241003,
}

// Matches the Vulkan enum VkImageAspectFlags.
[Flags]
public enum GpuImageAspect
{
    None = 0,
    ColorBit = 1,
    DepthBit = 2,
    StencilBit = 4,
}

// Matches the Vulkan enum VkPipelineStageFlags.
[Flags]
public enum GpuPipelineStage
{
    None = 0,
    TopOfPipeBit = 1,
    DrawIndirectBit = 2,
    VertexInputBit = 4,
    VertexShaderBit = 8,
    TessellationControlShaderBit = 16,
    TessellationEvaluationShaderBit = 32,
    GeometryShaderBit = 64,
    FragmentShaderBit = 128,
    EarlyFragmentTestsBit = 256,
    LateFragmentTestsBit = 512,
    ColorAttachmentOutputBit = 1024,
    ComputeShaderBit = 2048,
    TransferBit = 4096,
    BottomOfPipeBit = 8192,
    HostBit = 16384,
    AllGraphicsBit = 32768,
    AllCommandsBit = 65536,
    MeshShaderBitExt = 0x00100000,
}

// Matches the Vulkan enum VkPipelineBindPoint.
public enum GpuBindPoint
{
    Graphics = 0,
    Compute = 1,
}

// Matches the Vulkan enum VkPrimitiveTopology.
public enum GpuPrimitiveTopology
{
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
    TriangleFan = 5,
    PatchList = 10,
}

// Matches the Vulkan enum VkCompareOp.
public enum GpuCompareOp
{
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    Always = 7,
}

// Matches the Vulkan enum VkStencilOp.
public enum GpuStencilOp
{
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementAndClamp = 3,
    DecrementAndClamp = 4,
    Invert = 5,
    IncrementAndWrap = 6,
    DecrementAndWrap = 7,
}

// Matches the Vulkan enum VkImageViewType.
public enum GpuImageViewType
{
    Type1D = 0,
    Type2D = 1,
    Type3D = 2,
    TypeCube = 3,
    Type1DArray = 4,
    Type2DArray = 5,
    TypeCubeArray = 6,
}

// Matches the Vulkan enum VkComponentSwizzle.
public enum GpuComponentSwizzle
{
    Identity = 0,
    Zero = 1,
    One = 2,
    R = 3,
    G = 4,
    B = 5,
    A = 6,
}
