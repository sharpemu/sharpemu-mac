// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A real device, a real queue and one command buffer at a time, plus the serials
/// <see cref="IMetalSubmissionContext"/> promises. Submitting waits, so a staging slab
/// is reclaimed for the next case exactly as it would be in a frame.
/// </summary>
internal sealed unsafe class MetalTransferHarness : IMetalSubmissionContext, IDisposable
{
    private static readonly nint SelCommandBuffer = MetalNative.Selector("commandBuffer");
    private static readonly nint SelCommit = MetalNative.Selector("commit");
    private static readonly nint SelWait = MetalNative.Selector("waitUntilCompleted");
    private static readonly nint SelRetain = MetalNative.Selector("retain");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelContents = MetalNative.Selector("contents");
    private static readonly nint SelNewBufferWithBytes = MetalNative.Selector("newBufferWithBytes:length:options:");
    private static readonly nint SelNewBufferWithLength = MetalNative.Selector("newBufferWithLength:options:");
    private static readonly nint SelNewTexture = MetalNative.Selector("newTextureWithDescriptor:");

    private readonly nint _queue;
    private readonly List<nint> _owned = [];

    public MetalTransferHarness(nint device)
    {
        Device = device;
        _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        Transfer = new MetalImageTransfer(device);
    }

    public nint Device { get; }

    public nint CommandBuffer { get; private set; }

    public ulong SubmissionSerial { get; private set; } = 1;

    public ulong CompletedSerial { get; private set; }

    public MetalImageTransfer Transfer { get; }

    public MetalTiler Tiler => Transfer.Tiler;

    /// <summary>The encoders the transfer and the tiler opened through this context.</summary>
    public int EncodersStarted { get; private set; }

    void IMetalSubmissionContext.EncoderStarted(nint encoder, bool render) => EncodersStarted++;

    public void Begin() => CommandBuffer = MetalNative.Send(MetalNative.Send(_queue, SelCommandBuffer), SelRetain);

    public void Submit()
    {
        MetalNative.SendVoid(CommandBuffer, SelCommit);
        MetalNative.SendVoid(CommandBuffer, SelWait);
        MetalNative.SendVoid(CommandBuffer, SelRelease);
        CommandBuffer = 0;
        CompletedSerial = SubmissionSerial;
        SubmissionSerial++;
    }

    public nint Buffer(ReadOnlySpan<byte> bytes)
    {
        nint buffer;
        fixed (byte* pointer = bytes)
        {
            buffer = MetalNative.SendBuffer(Device, SelNewBufferWithBytes, (nint)pointer, (nuint)bytes.Length, 0);
        }

        _owned.Add(buffer);
        return buffer;
    }

    public nint Buffer(ulong size)
    {
        var buffer = MetalNative.SendNewBuffer(Device, SelNewBufferWithLength, (nuint)((size + 3) & ~3UL), 0);
        _owned.Add(buffer);
        return buffer;
    }

    public byte[] Read(nint buffer, ulong offset, ulong size)
    {
        var bytes = new byte[size];
        var contents = MetalNative.Send(buffer, SelContents);
        new ReadOnlySpan<byte>((byte*)contents + offset, (int)size).CopyTo(bytes);
        return bytes;
    }

    /// <summary>A private texture shaped exactly as the transfer asks for.</summary>
    public nint Texture(in MetalTextureShape shape)
    {
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLTextureDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setTextureType:"), (nint)shape.TextureType);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setPixelFormat:"), (nint)shape.PixelFormat);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setWidth:"), (nint)shape.Width);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setHeight:"), (nint)shape.Height);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setDepth:"), (nint)shape.Depth);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setMipmapLevelCount:"), (nint)shape.MipLevels);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setArrayLength:"), (nint)shape.ArrayLength);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setSampleCount:"), (nint)shape.SampleCount);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setUsage:"), (nint)shape.Usage);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setStorageMode:"), 2);
        var texture = MetalNative.Send(Device, SelNewTexture, descriptor);
        MetalNative.SendVoid(descriptor, SelRelease);
        _owned.Add(texture);
        return texture;
    }

    public void Dispose()
    {
        Transfer.Dispose();
        foreach (var handle in _owned)
        {
            if (handle != 0)
            {
                MetalNative.SendVoid(handle, SelRelease);
            }
        }

        _owned.Clear();
        if (_queue != 0)
        {
            MetalNative.SendVoid(_queue, SelRelease);
        }
    }
}

/// <summary>
/// The CPU twin of the tiler kernels, assembled from the shared tile tables only:
/// <see cref="TileGeometry.TryGetBlockOffset"/> places an element inside its block and
/// <see cref="TileGeometry.TryGetBlockXor"/> folds in the per-block and per-layer address
/// bits. Nothing here reads the MSL kernels or the Metal backend, so it is an independent
/// oracle for what the GPU must produce.
/// </summary>
internal static class MetalTileOracle
{
    public static byte[] Pattern(int length, uint seed)
    {
        var bytes = new byte[length];
        var state = seed * 2654435761u + 1u;
        for (var index = 0; index < length; index++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bytes[index] = (byte)(state >> 7);
        }

        return bytes;
    }

    public static ulong SliceBytes(in MetalTileTransfer transfer) =>
        transfer.LinearSliceStride != 0
            ? transfer.LinearSliceStride
            : (ulong)transfer.Pitch * transfer.BytesPerElement * transfer.Height;

    // Moves one transfer's elements between the linear and tiled byte arrays.
    public static void Transfer(byte[] linear, byte[] tiled, in MetalTileTransfer transfer, bool toTiled)
    {
        if (!TileGeometry.TryGetBlockLayout(transfer.Kind, transfer.BytesPerElement, out var block))
        {
            throw new InvalidOperationException($"No block layout for {transfer.Kind}/{transfer.BytesPerElement}.");
        }

        var tiledWidth = transfer.TiledWidth != 0 ? transfer.TiledWidth : transfer.Pitch;
        var tiledHeight = transfer.TiledHeight != 0 ? transfer.TiledHeight : transfer.Height;
        var columns = (tiledWidth + block.BlockWidth - 1) / block.BlockWidth;
        var rows = (tiledHeight + block.BlockHeight - 1) / block.BlockHeight;
        var pitchBytes = (ulong)transfer.Pitch * transfer.BytesPerElement;
        var sliceBytes = SliceBytes(transfer);
        var thin = block.BlockDepth == 1;
        var bytes = (int)transfer.BytesPerElement;
        for (uint z = 0; z < transfer.Depth; z++)
        {
            for (uint y = 0; y < transfer.Height; y++)
            {
                for (uint x = 0; x < transfer.Width; x++)
                {
                    uint blockX = 0, blockY = 0, blockZ = 0, elementX, elementY;
                    if (transfer.Tail)
                    {
                        elementX = x + transfer.TailX;
                        elementY = y + transfer.TailY;
                    }
                    else
                    {
                        blockX = x / block.BlockWidth;
                        blockY = y / block.BlockHeight;
                        blockZ = z / block.BlockDepth;
                        elementX = x % block.BlockWidth;
                        elementY = y % block.BlockHeight;
                    }

                    var elementZ = (z + transfer.SurfaceZ) % block.BlockDepth;
                    if (!TileGeometry.TryGetBlockOffset(block, elementX, elementY, elementZ, out var offset) ||
                        !TileGeometry.TryGetBlockXor(block, blockX, blockY, thin ? transfer.SurfaceZ : blockZ, out var blockXor))
                    {
                        throw new InvalidOperationException(
                            $"No block offset for {transfer.Kind}/{transfer.BytesPerElement} at ({elementX},{elementY},{elementZ}).");
                    }

                    var blockIndex = transfer.Tail ? 0 : blockZ * columns * rows + blockY * columns + blockX;
                    var tiledIndex = (int)(transfer.TiledOffset + blockIndex * (ulong)block.BlockSize + (offset ^ blockXor));
                    var linearIndex = (int)(transfer.LinearOffset + z * sliceBytes + y * pitchBytes + x * transfer.BytesPerElement);
                    if (toTiled)
                    {
                        Array.Copy(linear, linearIndex, tiled, tiledIndex, bytes);
                    }
                    else
                    {
                        Array.Copy(tiled, tiledIndex, linear, linearIndex, bytes);
                    }
                }
            }
        }
    }

    // Bytes between the row width and the pitch are never transferred.
    public static void MaskPadding(byte[] bytes, in MetalTileTransfer transfer)
    {
        var pitchBytes = (ulong)transfer.Pitch * transfer.BytesPerElement;
        var rowBytes = (ulong)transfer.Width * transfer.BytesPerElement;
        var sliceBytes = SliceBytes(transfer);
        for (uint z = 0; z < transfer.Depth; z++)
        {
            for (uint y = 0; y < transfer.Height; y++)
            {
                var start = (int)(transfer.LinearOffset + z * sliceBytes + y * pitchBytes + rowBytes);
                var length = (int)Math.Min((ulong)(bytes.Length - start), pitchBytes - rowBytes);
                if (length > 0)
                {
                    Array.Clear(bytes, start, length);
                }
            }
        }
    }
}

/// <summary>One tile transfer case with its linear bytes and the tiled bytes the CPU
/// twin produces; the Metal port of the Vulkan backend's TilerCases.</summary>
internal sealed record MetalTilerCase(MetalTileTransfer Transfer, TileBlockLayout Block, byte[] Linear, byte[] Tiled, string Name);

internal static class MetalTilerCases
{
    public enum Shape
    {
        Blocks,
        Tail,
        Layered,
    }

    private static uint AlignUp(uint value, uint alignment) => (value + alignment - 1) / alignment * alignment;

    // Null when the kind and size cannot form the requested shape.
    public static MetalTilerCase? Create(TileBlockKind kind, uint bytes, Shape shape, uint seed)
    {
        if (!TileGeometry.TryGetBlockLayout(kind, bytes, out var block))
        {
            return null;
        }

        var transfer = new MetalTileTransfer { Kind = kind, BytesPerElement = bytes };
        switch (shape)
        {
            case Shape.Blocks:
                transfer.Width = block.BlockWidth + 3;
                transfer.Height = block.BlockHeight + 5;
                transfer.Depth = block.BlockDepth == 1 ? 1u : block.BlockDepth + 2;
                break;
            case Shape.Tail:
                if (kind == TileBlockKind.Standard256B)
                {
                    return null;
                }

                transfer.Width = Math.Min(5, block.BlockWidth / 2);
                transfer.Height = Math.Min(6, block.BlockHeight / 2);
                transfer.Depth = block.BlockDepth == 1 ? 1u : 2u;
                transfer.TailX = block.BlockWidth / 2;
                transfer.TailY = block.BlockHeight / 4;
                transfer.Tail = true;
                break;
            case Shape.Layered:
                if (kind is not (TileBlockKind.RenderTarget64KB or TileBlockKind.Depth64KB or TileBlockKind.RenderTarget64KBGen5))
                {
                    return null;
                }

                transfer.Width = block.BlockWidth + 1;
                transfer.Height = 9;
                transfer.Depth = 1;
                transfer.SurfaceZ = 3;
                break;
        }

        transfer.Pitch = transfer.Width + 2;
        transfer.TiledWidth = transfer.Tail ? block.BlockWidth : AlignUp(transfer.Width, block.BlockWidth);
        transfer.TiledHeight = transfer.Tail ? block.BlockHeight : AlignUp(transfer.Height, block.BlockHeight);
        var columns = transfer.TiledWidth / block.BlockWidth;
        var rows = transfer.TiledHeight / block.BlockHeight;
        var slabs = (transfer.Depth + block.BlockDepth - 1) / block.BlockDepth;
        transfer.TiledSize = transfer.Tail ? block.BlockSize : (ulong)columns * rows * slabs * block.BlockSize;
        transfer.LinearSize = (ulong)transfer.Pitch * bytes * transfer.Height * transfer.Depth;

        var linear = MetalTileOracle.Pattern((int)transfer.LinearSize, seed);
        var tiled = new byte[transfer.TiledSize];
        MetalTileOracle.Transfer(linear, tiled, transfer, toTiled: true);
        return new MetalTilerCase(transfer, block, linear, tiled, $"{kind}/{bytes}/{shape}");
    }

    public static IEnumerable<MetalTilerCase> All()
    {
        uint seed = 1;
        foreach (var kind in Enum.GetValues<TileBlockKind>())
        {
            for (uint bytes = 1; bytes <= 16; bytes *= 2)
            {
                foreach (var shape in Enum.GetValues<Shape>())
                {
                    if (Create(kind, bytes, shape, seed++) is { } created)
                    {
                        yield return created;
                    }
                }
            }
        }
    }
}
