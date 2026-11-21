// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

internal static partial class MetalNative
{
    public const nuint PixelFormatRgba32Uint = 123;
    public const nuint PixelFormatRgba32Float = 125;
    public const nuint TextureType2D = 2;
    public const nuint TextureType2DArray = 3;

    /// <summary>A 1x1 texture of 16-byte texels, one per layer, filled with the given words.</summary>
    public static nint CreateTexelTexture(nuint pixelFormat, nuint textureType, uint layers, uint[] texels)
    {
        var descriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(descriptor, Selector("setPixelFormat:"), pixelFormat);
        SendInteger(descriptor, Selector("setTextureType:"), textureType);
        SendInteger(descriptor, Selector("setWidth:"), 1);
        SendInteger(descriptor, Selector("setHeight:"), 1);
        SendInteger(descriptor, Selector("setDepth:"), 1);
        SendInteger(descriptor, Selector("setArrayLength:"), layers);
        SendInteger(descriptor, Selector("setMipmapLevelCount:"), 1);
        SendInteger(descriptor, Selector("setStorageMode:"), 0);
        SendInteger(descriptor, Selector("setUsage:"), 1);
        var texture = Send(Device.Value, Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0) throw new InvalidOperationException("Failed to create a texel texture.");
        unsafe
        {
            fixed (uint* words = texels)
            {
                for (uint layer = 0; layer < layers; layer++)
                {
                    SendReplace(texture, Selector("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
                        new MtlRegion { Width = 1, Height = 1, Depth = 1 }, 0, layer, (nint)(words + layer * 4), 16, 0);
                }
            }
        }

        return texture;
    }

    public static ulong ResourceId(nint texture) => SendULong(texture, Selector("gpuResourceID"));

    /// <summary>A bindless handle table as the v2 host writes it: 16 bytes per entry.</summary>
    public static DeviceTable CreateBindlessTable(byte[] entries, IReadOnlyList<nint> textures) =>
        new(DescriptorBindingKind.Buffers, SharedBuffer(entries), (uint)(entries.Length / 16), textures);

    /// <summary>The bindless heap's element table as the v2 host writes it: 8 bytes per element, slot-major.</summary>
    public static DeviceTable CreateHeapElements(byte[] elements, IReadOnlyList<nint> residents) =>
        new(DescriptorBindingKind.Buffers, SharedBuffer(elements), (uint)(elements.Length / 8), residents);

    /// <summary>The bindless heap's feedback words (mark bitsets, then element classes), as given.</summary>
    public static DeviceTable CreateHeapFeedback(uint[] words) => CreateWordTable(DescriptorBindingKind.Buffers, words);

    /// <summary>The words of a table's buffer as the GPU left them.</summary>
    public static uint[] ReadWords(DeviceTable table)
    {
        var bytes = ReadSharedBuffer(table.Buffer, (int)table.Count * sizeof(uint));
        var words = new uint[table.Count];
        Buffer.BlockCopy(bytes, 0, words, 0, bytes.Length);
        return words;
    }

    /// <summary>A normalized-coordinate sampler with the given min/mag filter (0 nearest, 1 linear), clamped to edge.</summary>
    public static nint CreateFilterSampler(nuint filter)
    {
        var descriptor = Send(Send(objc_getClass("MTLSamplerDescriptor"), Selector("alloc")), Selector("init"));
        SendVoidBool(descriptor, Selector("setSupportArgumentBuffers:"), true);
        SendInteger(descriptor, Selector("setNormalizedCoordinates:"), 1);
        SendInteger(descriptor, Selector("setMinFilter:"), filter);
        SendInteger(descriptor, Selector("setMagFilter:"), filter);
        return Send(Device.Value, Selector("newSamplerStateWithDescriptor:"), descriptor);
    }

    /// <summary>A read-only table bound to one pointer field, holding the given words.</summary>
    public static DeviceTable CreateWordTable(DescriptorBindingKind kind, uint[] words)
    {
        var bytes = new byte[Math.Max(words.Length, 1) * sizeof(uint)];
        Buffer.BlockCopy(words, 0, bytes, 0, words.Length * sizeof(uint));
        return new DeviceTable(kind, SharedBuffer(bytes), (uint)words.Length, []);
    }
}
