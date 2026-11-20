// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// An integer image store whose select is baked into the module (SHARPEMU_RUNTIME_IMAGE_SWIZZLE=0, or an indirect
/// table's candidate) keeps the low bits of each register, as the hardware and upstream's store through the unsigned
/// view do, where a Metal integer texture write saturates: the store narrows each component to the width its image's
/// width slot (ImageStoreWidth) gives, which the host writes from the bound T# (Gen5MslTranslator.WriteStoreWidths).
/// </summary>
public sealed class MetalIntegerStoreWidthTests(ITestOutputHelper output)
{
    private const uint Size = 2;

    // Guest unified format, MTLPixelFormat of the view the host binds, bytes per texel, component widths (X first).
    // A write-only 16_SINT image is viewed and stored as R16Uint (upstream 99f9fe930).
    public static TheoryData<uint, uint, uint, uint[]> Formats() => new()
    {
        { 5, 13, 1, [8] }, // 8_UINT -> R8Uint
        { 11, 23, 2, [16] }, // 16_UINT -> R16Uint
        { 12, 23, 2, [16] }, // 16_SINT, write-only -> R16Uint
        { 54, 91, 4, [10, 10, 10, 2] }, // 2_10_10_10_UINT (X, Y, Z 10 bits, W 2) -> RGB10A2Uint
    };

    // Wider than any of the formats, with distinct low bits per component.
    private static readonly uint[] Values = [0xFFFF_82AB, 0x0001_2345, 0xFFFF_FC01, 0xFFFF_FFFE];

    [Theory]
    [MemberData(nameof(Formats))]
    public void AnIntegerStoreWithTheSelectInTheModuleKeepsTheLowBits(uint guestFormat, uint pixelFormat, uint texelBytes, uint[] widths)
    {
        BindingLayout? layout = null;
        ResourceSnapshot? snapshot = null;
        ShaderCompileRequest? request = null;
        Assert.True(Gen5ImageFixtures.TryCompile("ImageStore", Gen5ImageFixtures.ImageType2D, out var shader, out var error,
            unifiedFormat: guestFormat, dmask: 0xF, addressValues: [1, 0], storeValues: Values, runtimeImageSwizzles: false,
            inspect: (compiled, resources) => (request, layout, snapshot) = (compiled, compiled.Bindings, resources)), error);
        Assert.Equal(0u, layout!.StorageImageSwizzleCount);
        Assert.Equal(1u, layout.StorageImageWidthCount);
        Assert.Contains("extract_bits", shader!.Source, StringComparison.Ordinal);

        // The shader data the descriptor path writes (MetalRenderHost.WriteStageShaderData): here only the width slot.
        var shaderData = new uint[layout.ShaderDataDwordCount];
        Gen5MslTranslator.WriteStoreWidths(shaderData, layout, request!.Resources.Info.Images, request.Resources.Info.UsesRuntimeImageSwizzles,
            snapshot!.Images);
        var expectedCode = widths.Length == 4 ? 3u : widths[0] == 8 ? 1u : 2u;
        Assert.Equal(expectedCode, shaderData[layout.StorageImageWidthsDword] & 3u);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Texture execution requires the exclusive Metal device gate; only the layout and source are checked.");
            return;
        }

        Assert.True(MetalNative.IsAvailable, "The required gate needs a Metal device.");
        Assert.True(layout.UsesPushData);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), error + shader.Source);
        byte[] Store(uint[] data)
        {
            var push = new byte[PushData.ByteSize];
            Buffer.BlockCopy(data, 0, push, (int)layout.PushDataStartDword * 4, data.Length * 4);
            var texture = MetalNative.CreateRawTexture(pixelFormat, Size, Size, texelBytes,
                Enumerable.Repeat((byte)0xA5, (int)(Size * Size * texelBytes)).ToArray());
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16], push,
                shader.ArgumentLayout!, 1, out _, out error, texture), error);
            return MetalNative.ReadRawTexture(texture, Size, Size, texelBytes).AsSpan((int)texelBytes, (int)texelBytes).ToArray();
        }

        var texel = Store(shaderData);
        // Width code 0 stores each register whole, as every store without a slot did: Metal saturates it.
        var whole = Store(new uint[shaderData.Length]);

        // Each component's low bits, packed from bit 0 (X first): a saturating write stores all ones instead.
        var packed = 0ul;
        var shift = 0;
        for (var component = 0; component < widths.Length; component++)
        {
            packed |= (Values[component] & ((1ul << (int)widths[component]) - 1)) << shift;
            shift += (int)widths[component];
        }

        var expected = BitConverter.GetBytes(packed).AsSpan(0, (int)texelBytes).ToArray();
        output.WriteLine($"format {guestFormat}: narrowed {Convert.ToHexString(texel)}, whole {Convert.ToHexString(whole)}");
        Assert.Equal(Convert.ToHexString(expected), Convert.ToHexString(texel));
        Assert.NotEqual(Convert.ToHexString(expected), Convert.ToHexString(whole));
    }
}
