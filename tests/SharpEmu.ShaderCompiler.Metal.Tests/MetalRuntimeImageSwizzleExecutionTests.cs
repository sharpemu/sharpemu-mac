// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// A store through a storage image whose destination select is shader data writes, bit for bit,
/// the texel an MSL module with the select baked in writes: every permutation of XYZW and a sample
/// of other selects, two dmasks, and common storage formats (8-bit unorm, 16-bit float with 32-bit
/// and D16 data, 32-bit uint and float). One module serves every select. The Metal twin of
/// RuntimeImageSwizzleDeviceTests (Vulkan).
/// </summary>
public sealed class MetalRuntimeImageSwizzleExecutionTests(ITestOutputHelper output)
{
    private const uint Size = 2;

    // X001, XYZ1, XXXX, 0000, 1111, ZYX0, XYXY.
    private static readonly uint[] OtherSelects = [0x204, 0x3AC, 0x924, 0x000, 0x249, 0x12E, 0xB2C];

    // Guest format, MTLPixelFormat, bytes per texel, D16 data.
    public static TheoryData<uint, uint, uint, bool> Formats() => new()
    {
        { 56, 70, 4, false }, // RGBA8Unorm
        { 71, 115, 8, false }, // RGBA16Float
        { 71, 115, 8, true },
        { 75, 123, 16, false }, // RGBA32Uint
        { 77, 125, 16, false }, // RGBA32Float
    };

    [Theory]
    [MemberData(nameof(Formats))]
    public void RuntimeSelectStoresWriteTheBakedTexelBitForBit(uint guestFormat, uint pixelFormat, uint texelBytes, bool d16)
    {
        var device = DeviceGate();
        var selects = Permutations().Concat(OtherSelects).ToArray();
        var stores = 0;
        foreach (var dmask in new[] { 0xFu, 0x5u })
        {
            string? shared = null;
            foreach (var select in selects)
            {
                var baked = Store(guestFormat, pixelFormat, texelBytes, d16, dmask, select, runtime: false, device);
                var runtime = Store(guestFormat, pixelFormat, texelBytes, d16, dmask, select, runtime: true, device);
                Assert.NotEqual(baked.Source, runtime.Source);
                shared ??= runtime.Source;
                Assert.Equal(shared, runtime.Source);
                if (!device)
                {
                    continue;
                }

                Assert.Equal(Convert.ToHexString(baked.Texel), Convert.ToHexString(runtime.Texel));
                if (select == 0xF2E && dmask == 0xF)
                {
                    // ZYXW: the X channel holds the third component.
                    Assert.Equal(Convert.ToHexString(Texel(guestFormat, dmask, select)), Convert.ToHexString(baked.Texel));
                }

                stores++;
            }
        }

        output.WriteLine($"Compared {stores} baked and runtime-select stores of format {guestFormat} (d16={d16}).");
    }

    // A shader the translation disk cache rebuilt from its payload (Gen5MslShaderPayload) compiles to a library and
    // stores, through the runtime select, the texel the fresh translation stores.
    [Theory]
    [MemberData(nameof(Formats))]
    public void AShaderRebuiltFromItsCachePayloadStoresTheTexelTheTranslationStores(uint guestFormat, uint pixelFormat,
        uint texelBytes, bool d16)
    {
        var device = DeviceGate();
        var rebuilt = 0;
        foreach (var select in new[] { 0xF2Eu, 0x204u, 0x12Eu })
        {
            var fresh = Store(guestFormat, pixelFormat, texelBytes, d16, 0xF, select, runtime: true, device);
            var cached = Store(guestFormat, pixelFormat, texelBytes, d16, 0xF, select, runtime: true, device, shader =>
            {
                Assert.True(Gen5MslShaderPayload.TryRead(Gen5MslShaderPayload.Write(shader, ShaderStage.Compute, 0x5EED),
                    out var read, out _, out _, out var error), error);
                Assert.NotSame(shader, read);
                rebuilt++;
                return read;
            });
            Assert.Equal(fresh.Source, cached.Source);
            Assert.Equal(Convert.ToHexString(fresh.Texel), Convert.ToHexString(cached.Texel));
        }

        Assert.Equal(3, rebuilt);
        output.WriteLine($"Compared 3 fresh and rebuilt stores of format {guestFormat} (d16={d16}, device={device}).");
    }

    // The 24 orderings of X, Y, Z and W.
    private static IEnumerable<uint> Permutations()
    {
        foreach (var x in new uint[] { 4, 5, 6, 7 })
        foreach (var y in new uint[] { 4, 5, 6, 7 })
        foreach (var z in new uint[] { 4, 5, 6, 7 })
        foreach (var w in new uint[] { 4, 5, 6, 7 })
        {
            if (new[] { x, y, z, w }.Distinct().Count() == 4)
            {
                yield return x | (y << 3) | (z << 6) | (w << 9);
            }
        }
    }

    // The data registers' values: distinct per component, exact in the format.
    private static uint[] Data(uint guestFormat, bool d16)
    {
        if (d16)
        {
            return [Half(1.5f) | (Half(2.5f) << 16), Half(3.5f) | (Half(4.5f) << 16)];
        }

        return guestFormat switch
        {
            56 => [Bits(0.2f), Bits(0.4f), Bits(0.6f), Bits(0.8f)],
            75 => [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444],
            _ => [Bits(1.5f), Bits(2.5f), Bits(3.5f), Bits(4.5f)],
        };

        static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);
        static uint Half(float value) => BitConverter.HalfToUInt16Bits((Half)value);
    }

    // The bytes of the texel a store of every component through select writes.
    private static byte[] Texel(uint guestFormat, uint dmask, uint select)
    {
        var bytes = new List<byte>();
        float[] floats = guestFormat == 56 ? [0.2f, 0.4f, 0.6f, 0.8f] : [1.5f, 2.5f, 3.5f, 4.5f];
        uint[] uints = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        for (var channel = 0; channel < 4; channel++)
        {
            var component = Gen5ShaderTranslator.GetImageStoreSourceIndex(select, dmask, channel);
            switch (guestFormat)
            {
                case 56:
                    bytes.Add(component < 0 ? (byte)0 : (byte)Math.Round(floats[component] * 255));
                    break;
                case 71:
                    bytes.AddRange(BitConverter.GetBytes(component < 0 ? (ushort)0 : BitConverter.HalfToUInt16Bits((Half)floats[component])));
                    break;
                case 75:
                    bytes.AddRange(BitConverter.GetBytes(component < 0 ? 0u : uints[component]));
                    break;
                default:
                    bytes.AddRange(BitConverter.GetBytes(component < 0 ? 0f : floats[component]));
                    break;
            }
        }

        return [.. bytes];
    }

    private static (byte[] Texel, string Source) Store(uint guestFormat, uint pixelFormat, uint texelBytes, bool d16,
        uint dmask, uint select, bool runtime, bool device, Func<Gen5MslShader, Gen5MslShader>? rebuild = null)
    {
        BindingLayout? layout = null;
        ResourceSnapshot? snapshot = null;
        ShaderCompileRequest? request = null;
        Assert.True(Gen5ImageFixtures.TryCompile("ImageStore", Gen5ImageFixtures.ImageType2D, out var shader, out var error,
            unifiedFormat: guestFormat, dstSelect: select, dmask: dmask, d16: d16, addressValues: [1, 0],
            storeValues: Data(guestFormat, d16), runtimeImageSwizzles: runtime,
            inspect: (compiled, resources) => (request, layout, snapshot) = (compiled, compiled.Bindings, resources)), error);
        Assert.Equal(runtime ? 1u : 0u, layout!.StorageImageSwizzleCount);
        shader = rebuild is null ? shader : rebuild(shader!);
        if (!device)
        {
            return ([], shader!.Source);
        }

        // The shader data the descriptor path writes (WriteStageShaderData): here only the store
        // channel maps, which the fixture's compute kernel reads from push data.
        var push = new byte[PushData.ByteSize];
        if (runtime)
        {
            Assert.True(layout.UsesPushData);
            var shaderData = new uint[layout.ShaderDataDwordCount];
            RuntimeImageSwizzle.WriteStoreMaps(shaderData, layout, request!.Resources.Info.Images, snapshot!.Images);
            Buffer.BlockCopy(shaderData, 0, push, (int)layout.PushDataStartDword * 4, shaderData.Length * 4);
        }

        var initial = Enumerable.Repeat((byte)0xA5, (int)(Size * Size * texelBytes)).ToArray();
        var texture = MetalNative.CreateRawTexture(pixelFormat, Size, Size, texelBytes, initial);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error + shader.Source);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16], push,
            shader.ArgumentLayout!, 1, out _, out error, texture), error);
        var bytes = MetalNative.ReadRawTexture(texture, Size, Size, texelBytes);
        return (bytes.AsSpan((int)texelBytes, (int)texelBytes).ToArray(), shader.Source);
    }

    private bool DeviceGate()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Texture execution requires the exclusive Metal device gate; only the sources are compared.");
            return false;
        }

        Assert.True(MetalNative.IsAvailable, "The required gate needs a Metal device.");
        return true;
    }
}
