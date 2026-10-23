// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// A linear byte range inside a staging buffer, or any MTLBuffer the tiler reads or writes.
internal readonly record struct MetalBufferSpan(nint Buffer, ulong Offset, ulong Size)
{
    public bool IsValid => Buffer != 0 && Size != 0;
}

internal enum MetalDepthConversion
{
    Widen,
    Narrow,
}

// The 16-bit depth conversion window, in the layout the Vulkan DepthConversionLayout uses.
internal struct MetalDepthConversionLayout
{
    public uint Width;
    public uint Height;
    public uint Layers;
    public ulong SourceRowStride;
    public ulong TargetRowStride;
    public ulong SourceSliceStride;
    public ulong TargetSliceStride;
}

// The 13 dwords every block-copy kernel reads; the MSL TileArgs struct, field for field.
[StructLayout(LayoutKind.Sequential, Pack = 4)]
internal struct MetalTileArguments
{
    public uint SourceBase;
    public uint DestinationBase;
    public uint Width;
    public uint Height;
    public uint Depth;
    public uint SurfaceZ;
    public uint PitchBytes;
    public uint SliceBytes;
    public uint BlocksPerRow;
    public uint BlocksPerSlice;
    public uint TailX;
    public uint TailY;
    public uint Tail;
    public uint BaseY;
}

// The MSL RepackArgs struct.
[StructLayout(LayoutKind.Sequential, Pack = 4)]
internal struct MetalRepackArguments
{
    public uint SourceBase;
    public uint DestinationBase;
    public uint RowWords;
    public uint Rows;
    public uint Slices;
    public uint RowBytes;
    public uint SourceRowBytes;
    public uint TargetRowBytes;
    public uint SourceSliceBytes;
    public uint TargetSliceBytes;
    public uint SourceWords;
}

// Runs the tiling, detiling, depth conversion, channel swap and row repack compute
// passes, and owns the staging buffers they stage through. The Metal twin of the Vulkan
// backend's GpuTiler: same arguments, same per-element addressing, no descriptor sets.
internal sealed unsafe partial class MetalTiler : IDisposable
{
    private const uint KindCount = 10;
    private const uint ElementSizeCount = 5;
    private const uint VariantCount = 2;
    private const uint PipelineCount = KindCount * VariantCount * ElementSizeCount * VariantCount;

    // Device buffer bindings take an offset aligned to the largest type the kernel loads,
    // which is uint4. The residual rides along in the arguments, exactly as the Vulkan
    // tiler folds a descriptor offset into SourceBase.
    private const ulong BindAlignment = 16;

    private static readonly nint SelAlloc = MetalNative.Selector("alloc");
    private static readonly nint SelInit = MetalNative.Selector("init");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelNewLibrary = MetalNative.Selector("newLibraryWithSource:options:error:");
    private static readonly nint SelNewFunction = MetalNative.Selector("newFunctionWithName:");
    private static readonly nint SelNewSpecialized = MetalNative.Selector("newFunctionWithName:constantValues:error:");
    private static readonly nint SelNewComputePipeline = MetalNative.Selector("newComputePipelineStateWithFunction:error:");
    private static readonly nint SelSetConstantValue = MetalNative.Selector("setConstantValue:type:atIndex:");
    private static readonly nint SelComputeEncoder = MetalNative.Selector("computeCommandEncoder");
    private static readonly nint SelBlitEncoder = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelEndEncoding = MetalNative.Selector("endEncoding");
    private static readonly nint SelSetPipeline = MetalNative.Selector("setComputePipelineState:");
    private static readonly nint SelSetBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
    private static readonly nint SelSetBytes = MetalNative.Selector("setBytes:length:atIndex:");
    private static readonly nint SelDispatch = MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:");
    private static readonly nint SelSetTexture = MetalNative.Selector("setTexture:atIndex:");

    private const uint DataTypeUInt = 33;
    private const uint DataTypeBool = 53;

    private readonly nint _device;
    private readonly nint[] _blockLibraries = new nint[KindCount * VariantCount];
    private readonly nint[] _blockPipelines = new nint[PipelineCount];
    private readonly Dictionary<string, nint> _conversionPipelines = new(StringComparer.Ordinal);
    private readonly MetalStagingPool _staging;
    private readonly Dictionary<(TileBlockKind Kind, MetalTileDirection Direction), nint> _textureLibraries = [];
    private nint _conversionLibrary;
    private nint _depthWritebackLibrary;
    private bool _disposed;

    public MetalTiler(nint device)
    {
        if (device == 0)
        {
            throw GpuFatal.Fatal("The Metal tiler needs a device.");
        }

        _device = device;
        _staging = new MetalStagingPool(device);
    }

    /// <summary>The slabs <see cref="Acquire"/> lends from.</summary>
    internal MetalStagingPool Staging => _staging;

    public ulong StagingBytes => _staging.TotalBytes;

    public uint StagingBufferCount => _staging.SlabCount;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var pipeline in _blockPipelines)
        {
            Release(pipeline);
        }

        foreach (var pipeline in _conversionPipelines.Values)
        {
            Release(pipeline);
        }

        _conversionPipelines.Clear();
        foreach (var library in _blockLibraries)
        {
            Release(library);
        }

        Release(_conversionLibrary);
        _conversionLibrary = 0;
        Release(_depthWritebackLibrary);
        _depthWritebackLibrary = 0;
        foreach (var library in _textureLibraries.Values)
        {
            Release(library);
        }

        _textureLibraries.Clear();
        _staging.Dispose();
    }

    private static void Release(nint handle)
    {
        if (handle != 0)
        {
            MetalNative.SendVoid(handle, SelRelease);
        }
    }

    // Staging buffers are shared storage: on unified memory a detile writes bytes the
    // blit reads without a copy, and a readback is visible to the CPU with no extra pass.
    // A slab returns to the pool once the command buffer that used it completes; nothing
    // caps the size, so a 4096x4096 surface simply gets a bigger slab. The pool's size
    // classes, idle release, cap and purgeable state are MetalStagingPool's.
    public MetalBufferSpan Acquire(IMetalSubmissionContext context, ulong size) =>
        _staging.Acquire(context.SubmissionSerial, context.CompletedSerial, size);

    /// <summary>The submission the slabs in <see cref="RecordingBytes"/> were handed to.</summary>
    public ulong RecordingSerial => _staging.RecordingSerial;

    /// <summary>
    /// Shared memory the slabs handed to the submission being recorded hold. None of it
    /// returns before that command buffer completes, so a buffer that keeps recording
    /// transfers holds every slab it was given; the host commits it at a budget.
    /// </summary>
    public ulong RecordingBytes => _staging.RecordingBytes;

    /// <summary>Releases and makes purgeable the slabs idle since <paramref name="completedSerial"/>.</summary>
    public void TrimStaging(ulong completedSerial) => _staging.Trim(completedSerial);

    public nint BeginCompute(IMetalSubmissionContext context)
    {
        context.ReleaseSharedEncoder();
        var encoder = MetalNative.Send(context.CommandBuffer, SelComputeEncoder);
        MetalMemoryTelemetry.ComputeEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("The tiler could not open a compute encoder.");
        }

        context.EncoderStarted(encoder, render: false);
        return encoder;
    }

    public nint BeginBlit(IMetalSubmissionContext context)
    {
        context.ReleaseSharedEncoder();
        var encoder = MetalNative.Send(context.CommandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("The tiler could not open a blit encoder.");
        }

        context.EncoderStarted(encoder, render: false);
        return encoder;
    }

    // Ends an encoder outside any ordering (tests that drive the tiler on their own).
    public static void EndEncoding(nint encoder) => MetalNative.SendVoid(encoder, SelEndEncoding);

    // Ends an encoder BeginCompute or BeginBlit opened.
    public static void EndEncoding(IMetalSubmissionContext context, nint encoder)
    {
        context.EncoderEnding(encoder, render: false);
        MetalNative.SendVoid(encoder, SelEndEncoding);
    }

    private nint Library(TileBlockKind kind, MetalTileDirection direction)
    {
        var slot = (uint)kind * VariantCount + (direction == MetalTileDirection.Tile ? 1u : 0u);
        if (_blockLibraries[slot] != 0)
        {
            return _blockLibraries[slot];
        }

        _blockLibraries[slot] = CompileLibrary(MetalTilerShaders.CreateBlockCopyLibrary(kind, direction),
            $"tiler kind={kind} direction={direction}");
        return _blockLibraries[slot];
    }

    private nint ConversionLibrary()
    {
        if (_conversionLibrary == 0)
        {
            _conversionLibrary = CompileLibrary(MetalTilerShaders.CreateConversionLibrary(), "tiler conversions");
        }

        return _conversionLibrary;
    }

    private nint CompileLibrary(string source, string owner)
    {
        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), SelAlloc), SelInit);
            MetalNative.ApplyMslCompileOptions(options);
            nint error = 0;
            var library = MetalNative.Send(_device, SelNewLibrary, MetalNative.NsString(source), options, ref error);
            Release(options);
            if (library == 0)
            {
                throw GpuFatal.Fatal($"The tiler library did not compile: owner={owner} error={MetalNative.DescribeError(error)}.");
            }

            return library;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    private nint CreatePipeline(nint library, string functionName, uint elementBytes, bool? fastCopy, string owner)
    {
        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            nint error = 0;
            nint function;
            if (elementBytes == 0)
            {
                function = MetalNative.Send(library, SelNewFunction, MetalNative.NsString(functionName));
            }
            else
            {
                var values = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLFunctionConstantValues"), SelAlloc), SelInit);
                var bytes = elementBytes;
                MetalNative.SendSetConstantValue(values, SelSetConstantValue, (nint)(&bytes), DataTypeUInt, MetalTilerShaders.ElementBytesConstant);
                if (fastCopy is { } wanted)
                {
                    var fast = wanted ? (byte)1 : (byte)0;
                    MetalNative.SendSetConstantValue(values, SelSetConstantValue, (nint)(&fast), DataTypeBool, MetalTilerShaders.FastCopyConstant);
                }

                function = MetalNative.Send(library, SelNewSpecialized, MetalNative.NsString(functionName), values, ref error);
                Release(values);
            }

            if (function == 0)
            {
                throw GpuFatal.Fatal($"The tiler function is missing: owner={owner} error={MetalNative.DescribeError(error)}.");
            }

            error = 0;
            var pipeline = MetalNative.Send(_device, SelNewComputePipeline, function, ref error);
            Release(function);
            if (pipeline == 0)
            {
                throw GpuFatal.Fatal($"The tiler pipeline did not build: owner={owner} error={MetalNative.DescribeError(error)}.");
            }

            return pipeline;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    private nint BlockPipeline(TileBlockKind kind, MetalTileDirection direction, uint elementBytes, bool fastCopy)
    {
        var elementIndex = (uint)BitOperations.TrailingZeroCount(elementBytes);
        if ((uint)kind >= KindCount || elementIndex >= ElementSizeCount || !BitOperations.IsPow2(elementBytes))
        {
            throw GpuFatal.Fatal($"The tile transfer has no pipeline: kind={(uint)kind} bytes={elementBytes}.");
        }

        var slot = (((uint)kind * VariantCount + (direction == MetalTileDirection.Tile ? 1u : 0u)) * ElementSizeCount + elementIndex)
                   * VariantCount + (fastCopy ? 1u : 0u);
        if (_blockPipelines[slot] != 0)
        {
            return _blockPipelines[slot];
        }

        _blockPipelines[slot] = CreatePipeline(Library(kind, direction), MetalTilerShaders.KernelName(kind, direction),
            elementBytes, fastCopy, $"tiler kind={kind} direction={direction} bytes={elementBytes} fast={fastCopy}");
        return _blockPipelines[slot];
    }

    private nint ConversionPipeline(string functionName)
    {
        if (_conversionPipelines.TryGetValue(functionName, out var pipeline))
        {
            return pipeline;
        }

        pipeline = CreatePipeline(ConversionLibrary(), functionName, 0, false, $"tiler {functionName}");
        _conversionPipelines[functionName] = pipeline;
        return pipeline;
    }

    private nint DepthWritebackLibrary()
    {
        if (_depthWritebackLibrary == 0)
        {
            _depthWritebackLibrary = CompileLibrary(MetalTilerShaders.CreateDepthWritebackLibrary(), "tiler depth writeback");
        }

        return _depthWritebackLibrary;
    }

    private nint DepthWritebackPipeline(bool arrayed, uint elementBytes)
    {
        var name = arrayed ? MetalTilerShaders.DepthWritebackArray : MetalTilerShaders.DepthWriteback;
        var key = $"{name}/{elementBytes}";
        if (_conversionPipelines.TryGetValue(key, out var pipeline))
        {
            return pipeline;
        }

        pipeline = CreatePipeline(DepthWritebackLibrary(), name, elementBytes, null, $"tiler {key}");
        _conversionPipelines[key] = pipeline;
        return pipeline;
    }

    private nint TextureCopyPipeline(TileBlockKind kind, MetalTileDirection direction, uint elementBytes)
    {
        var name = MetalTilerShaders.TextureKernelName(kind, direction);
        var key = $"{name}/{elementBytes}";
        if (_conversionPipelines.TryGetValue(key, out var pipeline))
        {
            return pipeline;
        }

        if (!_textureLibraries.TryGetValue((kind, direction), out var library))
        {
            library = CompileLibrary(MetalTilerShaders.CreateTextureBlockCopyLibrary(kind, direction),
                $"tiler texture kind={kind} direction={direction}");
            _textureLibraries[(kind, direction)] = library;
        }

        pipeline = CreatePipeline(library, name, elementBytes, null, $"tiler {key}");
        _conversionPipelines[key] = pipeline;
        return pipeline;
    }

    /// <summary>
    /// Moves tiled guest bytes straight into a texture (detile) or straight out of one
    /// (tile), one dispatch per transfer, through an unsigned-integer 2D-array view whose
    /// texel is the transfer's element. The caller decides that every transfer qualifies
    /// (<see cref="MetalTilerShaders.SupportsTextureCopy"/>, no mip tail, one slice deep);
    /// one that does not is a caller bug, not a fallback.
    /// </summary>
    public void EncodeTextureBlockCopies(
        nint encoder, MetalBufferSpan tiled, nint view, List<MetalTileTransfer> transfers, MetalTileDirection direction)
    {
        if (view == 0 || transfers.Count == 0 || !tiled.IsValid)
        {
            throw GpuFatal.Fatal(
                $"The texture tile transfer is not encodable: view=0x{view:x} transfers={transfers.Count} tiled=0x{tiled.Size:x}.");
        }

        var tile = direction == MetalTileDirection.Tile;
        var tiledBase = FoldBase(tiled.Offset, tiled.Size, out var tiledBind, "texture tile transfer");
        if (tiledBase % 4 != 0)
        {
            throw GpuFatal.Fatal($"The texture tile transfer guest range is not dword aligned: offset=0x{tiled.Offset:x}.");
        }

        MetalV2Native.SendSetResourceAtIndex(encoder, SelSetTexture, view, MetalTilerShaders.TextureIndex);
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, tiled.Buffer, (nuint)tiledBind,
            tile ? MetalTilerShaders.TargetIndex : MetalTilerShaders.SourceIndex);
        var pipelineKind = (TileBlockKind)uint.MaxValue;
        uint pipelineBytes = 0;
        var subresource = stackalloc uint[2];
        foreach (ref readonly var transfer in CollectionsMarshal.AsSpan(transfers))
        {
            if (!MetalTilerShaders.SupportsTextureCopy(transfer.BytesPerElement) || transfer.Tail || transfer.Depth != 1)
            {
                throw GpuFatal.Fatal(
                    $"The texture tile transfer does not qualify: kind={(uint)transfer.Kind} bytes={transfer.BytesPerElement} " +
                    $"tail={transfer.Tail} depth={transfer.Depth}.");
            }

            var arguments = Prepare(transfer, tile, tile ? 0u : tiledBase, tile ? tiledBase : 0u, tiled.Size,
                ulong.MaxValue, out _);
            if (pipelineKind != transfer.Kind || pipelineBytes != transfer.BytesPerElement)
            {
                MetalNative.SendVoid(encoder, SelSetPipeline, TextureCopyPipeline(transfer.Kind, direction, transfer.BytesPerElement));
                pipelineKind = transfer.Kind;
                pipelineBytes = transfer.BytesPerElement;
            }

            subresource[0] = transfer.Slice;
            subresource[1] = transfer.Level;
            MetalNative.SendSetBytes(encoder, SelSetBytes, (nint)subresource, (nuint)sizeof(uint) * 2,
                MetalTilerShaders.SubresourceIndex);
            Dispatch(encoder, arguments, MetalTilerShaders.Threadgroup(transfer.Kind));
        }
    }

    private static void BindTarget(nint encoder, nint buffer, ulong offset)
    {
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, buffer, (nuint)offset, MetalTilerShaders.TargetIndex);
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, buffer, (nuint)offset, MetalTilerShaders.TargetAtomicIndex);
    }

    private static uint FoldBase(ulong offset, ulong capacity, out ulong bindOffset, string owner)
    {
        bindOffset = offset & ~(BindAlignment - 1);
        var residual = offset - bindOffset;
        if (residual + capacity > uint.MaxValue)
        {
            throw GpuFatal.Fatal($"A tiler binding exceeds 32 bits: owner={owner} offset=0x{offset:x} capacity=0x{capacity:x}.");
        }

        return (uint)residual;
    }

    private static bool TryMultiply(ulong left, ulong right, out ulong result)
    {
        result = left * right;
        return left == 0 || right <= ulong.MaxValue / left;
    }

    private static bool TryAdd(ulong left, ulong right, out ulong result)
    {
        result = left + right;
        return right <= ulong.MaxValue - left;
    }

    private static bool IsValidRange(ulong offset, ulong size, ulong capacity) =>
        size != 0 && offset <= capacity && size <= capacity - offset;

    // Validates one transfer and turns it into the dwords the kernel reads. The checks
    // mirror GpuTiler.Prepare so a layout that the Vulkan backend rejects is rejected here.
    private static MetalTileArguments Prepare(
        in MetalTileTransfer transfer,
        bool tile,
        uint sourceBase,
        uint destinationBase,
        ulong tiledCapacity,
        ulong linearCapacity,
        out TileBlockLayout block)
    {
        var tiledWidth = transfer.TiledWidth != 0 ? transfer.TiledWidth : transfer.Pitch;
        var tiledHeight = transfer.TiledHeight != 0 ? transfer.TiledHeight : transfer.Height;
        if (!TileGeometry.TryGetBlockLayout(transfer.Kind, transfer.BytesPerElement, out block) ||
            transfer.Width == 0 || transfer.Height == 0 || transfer.Depth == 0 || transfer.Pitch < transfer.Width ||
            (!transfer.Tail && (tiledWidth < transfer.Width || tiledHeight < transfer.Height)) ||
            !IsValidRange(transfer.LinearOffset, transfer.LinearSize, linearCapacity) ||
            !IsValidRange(transfer.TiledOffset, transfer.TiledSize, tiledCapacity) ||
            (block.BlockDepth == 1 && transfer.Depth != 1))
        {
            throw GpuFatal.Fatal(
                $"The tile transfer is invalid: kind={(uint)transfer.Kind} bytes={transfer.BytesPerElement} " +
                $"size={transfer.Width}x{transfer.Height}x{transfer.Depth} pitch={transfer.Pitch} " +
                $"tiled={tiledWidth}x{tiledHeight} tail={transfer.Tail} " +
                $"linear=0x{transfer.LinearOffset:x}+0x{transfer.LinearSize:x}/0x{linearCapacity:x} " +
                $"tiledRange=0x{transfer.TiledOffset:x}+0x{transfer.TiledSize:x}/0x{tiledCapacity:x}.");
        }

        if (!TryMultiply(transfer.Pitch, transfer.BytesPerElement, out var pitchBytes) || pitchBytes > uint.MaxValue)
        {
            throw GpuFatal.Fatal($"The tile transfer pitch exceeds 32 bits: pitch={transfer.Pitch} bytes={transfer.BytesPerElement}.");
        }

        var sliceBytes = transfer.LinearSliceStride;
        if (!TryMultiply(pitchBytes, transfer.Height, out var minimumSlice))
        {
            throw GpuFatal.Fatal($"The tile transfer slice size overflows: pitchBytes={pitchBytes} height={transfer.Height}.");
        }

        if (sliceBytes == 0)
        {
            sliceBytes = minimumSlice;
        }

        ulong linearUsed = 0;
        if ((transfer.Depth > 1 && sliceBytes < minimumSlice) ||
            !TryMultiply(transfer.Depth - 1u, sliceBytes, out var bytes) || !TryAdd(linearUsed, bytes, out linearUsed) ||
            !TryMultiply(transfer.Height - 1u, pitchBytes, out bytes) || !TryAdd(linearUsed, bytes, out linearUsed) ||
            !TryMultiply(transfer.Width, transfer.BytesPerElement, out bytes) || !TryAdd(linearUsed, bytes, out linearUsed) ||
            linearUsed > transfer.LinearSize || sliceBytes > uint.MaxValue)
        {
            throw GpuFatal.Fatal(
                $"The tile transfer linear extent does not fit its range: size={transfer.Width}x{transfer.Height}x{transfer.Depth} " +
                $"pitchBytes={pitchBytes} sliceBytes={sliceBytes} minimumSlice={minimumSlice} used={linearUsed} linearSize={transfer.LinearSize}.");
        }

        var columns = ((ulong)tiledWidth + block.BlockWidth - 1) / block.BlockWidth;
        var rows = ((ulong)tiledHeight + block.BlockHeight - 1) / block.BlockHeight;
        if (!TryMultiply(columns, rows, out var blocksPerSlice) || columns > uint.MaxValue || blocksPerSlice > uint.MaxValue)
        {
            throw GpuFatal.Fatal($"The tile transfer block grid exceeds 32 bits: columns={columns} rows={rows}.");
        }

        if (transfer.Tail)
        {
            if (transfer.Kind == TileBlockKind.Standard256B || transfer.Depth > block.BlockDepth ||
                transfer.TailX >= block.BlockWidth || transfer.Width > block.BlockWidth - transfer.TailX ||
                transfer.TailY >= block.BlockHeight || transfer.Height > block.BlockHeight - transfer.TailY ||
                transfer.TiledSize < block.BlockSize)
            {
                throw GpuFatal.Fatal(
                    $"The tail tile transfer does not fit one block: kind={(uint)transfer.Kind} tail={transfer.TailX},{transfer.TailY} " +
                    $"size={transfer.Width}x{transfer.Height}x{transfer.Depth} " +
                    $"block={block.BlockWidth}x{block.BlockHeight}x{block.BlockDepth} blockSize={block.BlockSize} tiledSize={transfer.TiledSize}.");
            }
        }
        else
        {
            var slices = ((ulong)transfer.Depth + block.BlockDepth - 1) / block.BlockDepth;
            if (!TryMultiply(blocksPerSlice, slices, out var tiledUsed) ||
                !TryMultiply(tiledUsed, block.BlockSize, out tiledUsed) || tiledUsed > transfer.TiledSize)
            {
                throw GpuFatal.Fatal(
                    $"The tile transfer tiled extent does not fit its range: blocksPerSlice={blocksPerSlice} slices={slices} " +
                    $"blockSize={block.BlockSize} tiledSize={transfer.TiledSize}.");
            }
        }

        var alignment = Math.Min(transfer.BytesPerElement, 4u);
        if (((transfer.LinearOffset | transfer.TiledOffset | pitchBytes | sliceBytes) & (alignment - 1u)) != 0)
        {
            throw GpuFatal.Fatal(
                $"The tile transfer offsets are not aligned to the element: alignment={alignment} " +
                $"linear=0x{transfer.LinearOffset:x} tiled=0x{transfer.TiledOffset:x} pitchBytes={pitchBytes} sliceBytes={sliceBytes}.");
        }

        var source = sourceBase + (tile ? transfer.LinearOffset : transfer.TiledOffset);
        var destination = destinationBase + (tile ? transfer.TiledOffset : transfer.LinearOffset);
        if (source > uint.MaxValue || destination > uint.MaxValue)
        {
            throw GpuFatal.Fatal($"The tile transfer base exceeds 32 bits: source=0x{source:x} destination=0x{destination:x}.");
        }

        return new MetalTileArguments
        {
            SourceBase = (uint)source,
            DestinationBase = (uint)destination,
            Width = transfer.Width,
            Height = transfer.Height,
            Depth = transfer.Depth,
            SurfaceZ = transfer.SurfaceZ,
            PitchBytes = (uint)pitchBytes,
            SliceBytes = (uint)sliceBytes,
            BlocksPerRow = (uint)columns,
            BlocksPerSlice = (uint)blocksPerSlice,
            TailX = transfer.TailX,
            TailY = transfer.TailY,
            Tail = transfer.Tail ? 1u : 0u,
            BaseY = transfer.BaseY,
        };
    }

    // The fast variant of a block copy. Four bytes and wider means whole-element vector
    // loads, which need the element's own alignment on both sides; the block offsets and
    // block sizes are always element-aligned, so only the folded bases and the linear
    // strides can break it. Sub-dword means one dword per element group, which needs the
    // destination side dword-aligned: the tiled side inherits that from the block size,
    // the linear side from its base and its strides. A tail transfer starts at an
    // arbitrary element inside its block, so no group of it is dword-aligned.
    private static bool AllowsFastCopy(in MetalTileTransfer transfer, in MetalTileArguments arguments, bool tile)
    {
        var bytes = transfer.BytesPerElement;
        if (bytes >= 4)
        {
            return bytes == 4 ||
                   ((arguments.SourceBase | arguments.DestinationBase | arguments.PitchBytes | arguments.SliceBytes) & (bytes - 1)) == 0;
        }

        if (transfer.Tail || WideGroupOf(transfer).Width == 0)
        {
            return false;
        }

        return tile
            ? arguments.DestinationBase % 4 == 0
            : ((arguments.DestinationBase | arguments.PitchBytes | arguments.SliceBytes) & 3u) == 0;
    }

    private static (uint Width, uint Height) WideGroupOf(in MetalTileTransfer transfer) =>
        MetalTilerShaders.WideGroup(transfer.Kind, transfer.BytesPerElement);

    private static void Dispatch(nint encoder, MetalTileArguments arguments, (uint X, uint Y, uint Z) group) =>
        Dispatch(encoder, arguments, group, (1, 1));

    // One thread per element, or per element group when the fast sub-dword variant is
    // bound: the grid shrinks by the group's extent and the kernel scales the index back.
    private static void Dispatch(
        nint encoder, MetalTileArguments arguments, (uint X, uint Y, uint Z) group, (uint Width, uint Height) wide)
    {
        var threadsX = (arguments.Width + wide.Width - 1) / wide.Width;
        var threadsY = (arguments.Height + wide.Height - 1) / wide.Height;
        MetalNative.SendSetBytes(encoder, SelSetBytes, (nint)(&arguments),
            (nuint)MetalTilerShaders.ArgumentBytes, MetalTilerShaders.ArgumentIndex);
        MetalNative.SendDispatch(encoder, SelDispatch,
            new MtlSize
            {
                Width = (nuint)((threadsX + group.X - 1) / group.X),
                Height = (nuint)((threadsY + group.Y - 1) / group.Y),
                Depth = (nuint)((arguments.Depth + group.Z - 1) / group.Z),
            },
            new MtlSize { Width = group.X, Height = group.Y, Depth = group.Z });
    }

    // Detiles the guest surface into a linear staging range, one dispatch per subresource.
    public void EncodeDetile(nint encoder, MetalBufferSpan tiled, MetalBufferSpan linear, List<MetalTileTransfer> transfers)
    {
        EncodeBlockCopies(encoder, tiled, linear, transfers, tile: false);
    }

    // Retiles a linear staging range into the guest surface.
    public void EncodeTile(nint encoder, MetalBufferSpan linear, MetalBufferSpan tiled, List<MetalTileTransfer> transfers)
    {
        EncodeBlockCopies(encoder, tiled, linear, transfers, tile: true);
    }

    private void EncodeBlockCopies(nint encoder, MetalBufferSpan tiled, MetalBufferSpan linear, List<MetalTileTransfer> transfers, bool tile)
    {
        if (transfers.Count == 0 || !tiled.IsValid || !linear.IsValid)
        {
            throw GpuFatal.Fatal(
                $"The tile transfer batch is empty or has no capacity: transfers={transfers.Count} tiled=0x{tiled.Size:x} linear=0x{linear.Size:x}.");
        }

        var tiledBase = FoldBase(tiled.Offset, tiled.Size, out var tiledBind, "tiled");
        var linearBase = FoldBase(linear.Offset, linear.Size, out var linearBind, "linear");
        var source = tile ? linear.Buffer : tiled.Buffer;
        var sourceBind = tile ? linearBind : tiledBind;
        var target = tile ? tiled.Buffer : linear.Buffer;
        var targetBind = tile ? tiledBind : linearBind;
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, source, (nuint)sourceBind, MetalTilerShaders.SourceIndex);
        BindTarget(encoder, target, targetBind);

        var pipelineKind = (TileBlockKind)uint.MaxValue;
        uint pipelineBytes = 0;
        var pipelineFast = false;
        var bound = false;
        var direction = tile ? MetalTileDirection.Tile : MetalTileDirection.Detile;
        foreach (ref readonly var transfer in System.Runtime.InteropServices.CollectionsMarshal.AsSpan(transfers))
        {
            var arguments = Prepare(transfer, tile, tile ? linearBase : tiledBase, tile ? tiledBase : linearBase,
                tiled.Size, linear.Size, out _);
            var fast = AllowsFastCopy(transfer, arguments, tile);
            if (!bound || pipelineKind != transfer.Kind || pipelineBytes != transfer.BytesPerElement || pipelineFast != fast)
            {
                MetalNative.SendVoid(encoder, SelSetPipeline, BlockPipeline(transfer.Kind, direction, transfer.BytesPerElement, fast));
                pipelineKind = transfer.Kind;
                pipelineBytes = transfer.BytesPerElement;
                pipelineFast = fast;
                bound = true;
            }

            var wide = fast && transfer.BytesPerElement < 4 ? WideGroupOf(transfer) : (1u, 1u);
            Dispatch(encoder, arguments, MetalTilerShaders.Threadgroup(transfer.Kind), wide);
        }
    }

    // Tiles the guest depth plane out of the Metal depth texture in one pass: the kernel
    // reads the texel and writes it where the guest surface's own 64 KiB addressing puts
    // it, so neither a blit into a staging buffer nor a separate retile is needed. One
    // dispatch per transfer, which for a depth surface is one per array layer; the layer
    // rides in surfaceZ, as both the texture slice and the folded address bits.
    public void EncodeDepthTextureWriteback(
        nint encoder, nint texture, MetalBufferSpan guest, uint elementBytes, bool arrayed, List<MetalTileTransfer> transfers)
    {
        if (texture == 0 || transfers.Count == 0 || !guest.IsValid || elementBytes is not (2 or 4))
        {
            throw GpuFatal.Fatal(
                $"The depth texture writeback is not encodable: texture=0x{texture:x} transfers={transfers.Count} " +
                $"guest=0x{guest.Size:x} bytes={elementBytes}.");
        }

        var tiledBase = FoldBase(guest.Offset, guest.Size, out var tiledBind, "depth writeback");
        if (tiledBase % 4 != 0)
        {
            throw GpuFatal.Fatal($"The depth writeback target is not dword aligned: offset=0x{guest.Offset:x}.");
        }

        MetalV2Native.SendSetResourceAtIndex(encoder, SelSetTexture, texture, MetalTilerShaders.TextureIndex);
        BindTarget(encoder, guest.Buffer, tiledBind);
        MetalNative.SendVoid(encoder, SelSetPipeline, DepthWritebackPipeline(arrayed, elementBytes));
        foreach (ref readonly var transfer in CollectionsMarshal.AsSpan(transfers))
        {
            if (transfer.BytesPerElement != elementBytes || transfer.Kind != TileBlockKind.Depth64KB || transfer.Tail)
            {
                throw GpuFatal.Fatal(
                    $"The depth writeback transfer does not match its plane: kind={(uint)transfer.Kind} " +
                    $"bytes={transfer.BytesPerElement} tail={transfer.Tail} planeBytes={elementBytes}.");
            }

            var arguments = Prepare(transfer, tile: true, 0, tiledBase, guest.Size,
                transfer.LinearOffset + transfer.LinearSize, out _);
            Dispatch(encoder, arguments, MetalTilerShaders.Threadgroup(TileBlockKind.Depth64KB),
                elementBytes == 2 ? (2u, 1u) : (1u, 1u));
        }
    }

    private static ulong RequiredBytes(uint height, uint layers, ulong rowStride, ulong sliceStride, ulong active)
    {
        if (height == 0 || layers == 0 || rowStride < active || height - 1 > (ulong.MaxValue - active) / rowStride)
        {
            throw GpuFatal.Fatal($"The depth conversion layout is invalid: height={height} layers={layers} rowStride={rowStride} active={active}.");
        }

        var slice = (ulong)(height - 1) * rowStride + active;
        if (sliceStride < slice || layers - 1 > (ulong.MaxValue - slice) / sliceStride)
        {
            throw GpuFatal.Fatal($"The depth conversion slice stride is too small: sliceStride={sliceStride} slice={slice} layers={layers}.");
        }

        return (ulong)(layers - 1) * sliceStride + slice;
    }

    // Widens 16-bit depth to the 32-bit plane Metal blits, or narrows it back.
    public void EncodeDepthConversion(
        nint encoder,
        MetalBufferSpan source,
        MetalBufferSpan target,
        MetalDepthConversion direction,
        bool float32,
        in MetalDepthConversionLayout layout)
    {
        var widen = direction == MetalDepthConversion.Widen;
        var sourceElement = widen ? 2UL : 4UL;
        var targetElement = widen ? 4UL : 2UL;
        if (layout.Width == 0 || layout.SourceRowStride > uint.MaxValue || layout.TargetRowStride > uint.MaxValue)
        {
            throw GpuFatal.Fatal(
                $"The depth conversion layout is invalid: width={layout.Width} sourceRowStride={layout.SourceRowStride} targetRowStride={layout.TargetRowStride}.");
        }

        var sourceRequired = RequiredBytes(layout.Height, layout.Layers, layout.SourceRowStride, layout.SourceSliceStride, layout.Width * sourceElement);
        var targetRequired = RequiredBytes(layout.Height, layout.Layers, layout.TargetRowStride, layout.TargetSliceStride, layout.Width * targetElement);
        if (source.Size < sourceRequired || target.Size < targetRequired)
        {
            throw GpuFatal.Fatal(
                $"The depth conversion buffers are too small: source={source.Size}/{sourceRequired} target={target.Size}/{targetRequired}.");
        }

        var name = widen
            ? float32 ? MetalTilerShaders.DepthWidenFloat32 : MetalTilerShaders.DepthWidenFixed24
            : float32 ? MetalTilerShaders.DepthNarrowFloat32 : MetalTilerShaders.DepthNarrowFixed24;
        var sourceBase = FoldBase(source.Offset, source.Size, out var sourceBind, "depth source");
        var targetBase = FoldBase(target.Offset, target.Size, out var targetBind, "depth target");
        MetalNative.SendVoid(encoder, SelSetPipeline, ConversionPipeline(name));
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, source.Buffer, (nuint)sourceBind, MetalTilerShaders.SourceIndex);
        BindTarget(encoder, target.Buffer, targetBind);
        for (uint layer = 0; layer < layout.Layers; layer++)
        {
            var arguments = new MetalTileArguments
            {
                SourceBase = (uint)(sourceBase + layout.SourceSliceStride * layer),
                DestinationBase = (uint)(targetBase + layout.TargetSliceStride * layer),
                Width = layout.Width,
                Height = layout.Height,
                Depth = 1,
                PitchBytes = (uint)layout.SourceRowStride,
                SliceBytes = (uint)layout.TargetRowStride,
            };
            Dispatch(encoder, arguments, (64, 1, 1));
        }
    }

    // Swaps the 16-bit blue and red channels of a BGRA16 range in place of a copy.
    public void EncodeSwapBgra16(nint encoder, MetalBufferSpan source, MetalBufferSpan target)
    {
        if (source.Size == 0 || source.Size % 8 != 0 || source.Size / 8 > uint.MaxValue || target.Size < source.Size)
        {
            throw GpuFatal.Fatal($"The BGRA16 swap sizes are invalid: source={source.Size} target={target.Size}.");
        }

        var sourceBase = FoldBase(source.Offset, source.Size, out var sourceBind, "swap source");
        var targetBase = FoldBase(target.Offset, target.Size, out var targetBind, "swap target");
        MetalNative.SendVoid(encoder, SelSetPipeline, ConversionPipeline(MetalTilerShaders.SwapBgra16));
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, source.Buffer, (nuint)sourceBind, MetalTilerShaders.SourceIndex);
        BindTarget(encoder, target.Buffer, targetBind);
        Dispatch(encoder, new MetalTileArguments
        {
            SourceBase = sourceBase,
            DestinationBase = targetBase,
            Width = (uint)(source.Size / 8),
            Height = 1,
            Depth = 1,
        }, (64, 1, 1));
    }

    // Copies row-strided bytes when a blit cannot address the guest range directly:
    // Metal wants a buffer-texture copy offset and row pitch that are multiples of the
    // texel size, and a guest linear surface is under no such obligation.
    public void EncodeRepackRows(
        nint encoder,
        MetalBufferSpan source,
        MetalBufferSpan target,
        uint rowBytes,
        uint rows,
        uint slices,
        ulong sourceRowBytes,
        ulong targetRowBytes,
        ulong sourceSliceBytes,
        ulong targetSliceBytes)
    {
        if (rowBytes == 0 || rows == 0 || slices == 0 ||
            sourceRowBytes < rowBytes || targetRowBytes < rowBytes ||
            sourceRowBytes > uint.MaxValue || targetRowBytes > uint.MaxValue ||
            sourceSliceBytes > uint.MaxValue || targetSliceBytes > uint.MaxValue)
        {
            throw GpuFatal.Fatal(
                $"The row repack window is invalid: rowBytes={rowBytes} rows={rows} slices={slices} " +
                $"sourceRow={sourceRowBytes} targetRow={targetRowBytes} sourceSlice={sourceSliceBytes} targetSlice={targetSliceBytes}.");
        }

        var sourceBase = FoldBase(source.Offset, source.Size, out var sourceBind, "repack source");
        var targetBase = FoldBase(target.Offset, target.Size, out var targetBind, "repack target");
        // The widest destination row spans one extra dword when it starts mid-dword.
        var rowWords = (rowBytes + 3u) / 4u + 1u;
        var arguments = new MetalRepackArguments
        {
            SourceBase = sourceBase,
            DestinationBase = targetBase,
            RowWords = rowWords,
            Rows = rows,
            Slices = slices,
            RowBytes = rowBytes,
            SourceRowBytes = (uint)sourceRowBytes,
            TargetRowBytes = (uint)targetRowBytes,
            SourceSliceBytes = (uint)sourceSliceBytes,
            TargetSliceBytes = (uint)targetSliceBytes,
            SourceWords = (uint)((sourceBase + source.Size + 3) / 4),
        };
        MetalNative.SendVoid(encoder, SelSetPipeline, ConversionPipeline(MetalTilerShaders.RepackRows));
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, source.Buffer, (nuint)sourceBind, MetalTilerShaders.SourceIndex);
        BindTarget(encoder, target.Buffer, targetBind);
        MetalNative.SendSetBytes(encoder, SelSetBytes, (nint)(&arguments), (nuint)sizeof(MetalRepackArguments), MetalTilerShaders.ArgumentIndex);
        MetalNative.SendDispatch(encoder, SelDispatch,
            new MtlSize { Width = (nuint)((rowWords + 63) / 64), Height = rows, Depth = slices },
            new MtlSize { Width = 64, Height = 1, Depth = 1 });
    }
}
