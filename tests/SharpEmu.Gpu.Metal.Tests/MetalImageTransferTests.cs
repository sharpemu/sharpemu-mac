// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The Metal v2 GPU image transfer: the generated MSL kernels against the shared CPU
/// tile tables, and whole images against the layout the shared TileGeometry defines.
/// Device work is behind the exclusive device gate, as every other native Metal case is.
/// </summary>
public sealed partial class MetalImageTransferTests(ITestOutputHelper output)
{
    [Fact]
    public void TilerSourcesCoverEveryKindDirectionAndElementSize()
    {
        var variants = 0;
        foreach (var kind in Enum.GetValues<TileBlockKind>())
        {
            foreach (var direction in Enum.GetValues<MetalTileDirection>())
            {
                var source = MetalTilerShaders.CreateBlockCopyLibrary(kind, direction);
                Assert.Contains($"kernel void {MetalTilerShaders.KernelName(kind, direction)}(", source);
                for (uint bytes = 1; bytes <= 16; bytes *= 2)
                {
                    if (!MetalTilerShaders.Supports(kind, bytes))
                    {
                        Assert.DoesNotContain($"if (kElementBytes == {bytes}u)", source);
                        continue;
                    }

                    Assert.Contains($"if (kElementBytes == {bytes}u)", source);
                    variants++;
                }
            }
        }

        // Nine kinds take all five element sizes; Depth64KB stops at eight bytes.
        Assert.Equal(2 * (9 * 5 + 4), variants);
        var conversions = MetalTilerShaders.CreateConversionLibrary();
        foreach (var name in new[]
                 {
                     MetalTilerShaders.DepthWidenFixed24, MetalTilerShaders.DepthWidenFloat32,
                     MetalTilerShaders.DepthNarrowFixed24, MetalTilerShaders.DepthNarrowFloat32,
                     MetalTilerShaders.SwapBgra16, MetalTilerShaders.RepackRows,
                 })
        {
            Assert.Contains($"kernel void {name}(", conversions);
        }
    }

    [Fact]
    public void CpuTwinCasesCoverEveryKindAndElementSize()
    {
        var cases = MetalTilerCases.All().ToList();
        Assert.Equal(44 + 39 + 9 + 15, cases.Count);
        Assert.All(cases, tilerCase => Assert.Equal((int)tilerCase.Transfer.TiledSize, tilerCase.Tiled.Length));
    }

    [Theory]
    [InlineData((int)GpuFormat.R8G8B8A8Unorm, (uint)MtlPixelFormat.Rgba8Unorm)]
    [InlineData((int)GpuFormat.B8G8R8A8Srgb, (uint)MtlPixelFormat.Bgra8UnormSrgb)]
    [InlineData((int)GpuFormat.BC7SrgbBlock, (uint)MtlPixelFormat.Bc7RgbaUnormSrgb)]
    [InlineData((int)GpuFormat.D32Sfloat, (uint)MtlPixelFormat.Depth32Float)]
    [InlineData((int)GpuFormat.D16UnormS8Uint, (uint)MtlPixelFormat.Depth32FloatStencil8)]
    [InlineData((int)GpuFormat.A2R10G10B10UnormPack32, (uint)MtlPixelFormat.Bgr10A2Unorm)]
    public void HostFormatsMapToTheirMetalEquivalent(int hostFormat, uint expected)
    {
        Assert.True(MetalGuestFormats.TryDecodeHostFormat((GpuFormat)hostFormat, out var format));
        Assert.Equal(expected, (uint)format);
    }

    [Fact]
    public void ShapeForDescribesCubesAsArraysAndVolumesAsThreeDimensional()
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var cube = Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 64, 64, 3, 6, GuestTileMode.Standard64KB);
        var cubeShape = transfer.ShapeFor(cube);
        Assert.Equal((ulong)MtlTextureType.Type2DArray, cubeShape.TextureType);
        Assert.Equal(6u, cubeShape.ArrayLength);
        Assert.Equal(1u, cubeShape.Depth);
        Assert.Equal((ulong)MtlPixelFormat.Rgba8Unorm, cubeShape.PixelFormat);
        Assert.True((cubeShape.Usage & (ulong)MtlTextureUsage.ShaderWrite) != 0);

        var volume = Volume(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 32, 32, 16, 2);
        var volumeShape = transfer.ShapeFor(volume);
        Assert.Equal((ulong)MtlTextureType.Type3D, volumeShape.TextureType);
        Assert.Equal(16u, volumeShape.Depth);
        Assert.Equal(1u, volumeShape.ArrayLength);

        var compressed = Color2D(GuestPixelFormat.Bc3UNorm, GpuFormat.BC3UnormBlock, 64, 64, 3, 1, GuestTileMode.Standard64KB);
        var compressedShape = transfer.ShapeFor(compressed);
        // Sampled only, plus the view usage for its sRGB twin (MetalImageTransferTests.SrgbTwinViews).
        Assert.Equal((ulong)(MtlTextureUsage.ShaderRead | MtlTextureUsage.PixelFormatView), compressedShape.Usage);
        Assert.Equal((ulong)MtlPixelFormat.Bc3Rgba, compressedShape.PixelFormat);
        var twinless = Color2D(GuestPixelFormat.Bc5UNorm, GpuFormat.BC5UnormBlock, 64, 64, 3, 1, GuestTileMode.Standard64KB);
        Assert.Equal((ulong)MtlTextureUsage.ShaderRead, transfer.ShapeFor(twinless).Usage);
    }

    [Fact]
    public void DetileAndTileMatchTheCpuTwinAndInvertEachOther()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var failures = new List<string>();
        var batch = new List<MetalTileTransfer>(1);
        var cases = 0;
        foreach (var tilerCase in MetalTilerCases.All())
        {
            var transfer = tilerCase.Transfer;
            batch.Clear();
            batch.Add(transfer);
            var tiledSpan = (transfer.TiledSize + 3) & ~3UL;
            var linearSpan = (transfer.LinearSize + 3) & ~3UL;
            var tiledSource = harness.Buffer(tilerCase.Tiled);
            var detiled = harness.Buffer(linearSpan);
            var retiled = harness.Buffer(tiledSpan);
            harness.Begin();
            var encoder = harness.Tiler.BeginCompute(harness);
            harness.Tiler.EncodeDetile(encoder,
                new MetalBufferSpan(tiledSource, 0, tiledSpan), new MetalBufferSpan(detiled, 0, linearSpan), batch);
            MetalTiler.EndEncoding(encoder);
            encoder = harness.Tiler.BeginCompute(harness);
            harness.Tiler.EncodeTile(encoder,
                new MetalBufferSpan(detiled, 0, linearSpan), new MetalBufferSpan(retiled, 0, tiledSpan), batch);
            MetalTiler.EndEncoding(encoder);
            harness.Submit();

            var actual = harness.Read(detiled, 0, transfer.LinearSize);
            var expected = tilerCase.Linear.ToArray();
            MetalTileOracle.MaskPadding(actual, transfer);
            MetalTileOracle.MaskPadding(expected, transfer);
            if (!actual.AsSpan().SequenceEqual(expected))
            {
                failures.Add($"{tilerCase.Name}: detile differs from the CPU twin");
            }

            if (!harness.Read(retiled, 0, transfer.TiledSize).AsSpan().SequenceEqual(tilerCase.Tiled))
            {
                failures.Add($"{tilerCase.Name}: retile does not invert detile");
            }

            cases++;
        }

        output.WriteLine($"tiler cases={cases}");
        Assert.Equal(44 + 39 + 9 + 15, cases);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void DepthConversionsWidenAndNarrowEveryRow()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        const uint width = 37;
        const uint height = 11;
        const uint layers = 2;
        var layout = new MetalDepthConversionLayout
        {
            Width = width,
            Height = height,
            Layers = layers,
            SourceRowStride = 80,
            TargetRowStride = 160,
            SourceSliceStride = 1024,
            TargetSliceStride = 2048,
        };
        var source = new byte[layers * 1024];
        var values = new ushort[layers, height, width];
        var random = new Random(41);
        for (var layer = 0; layer < layers; layer++)
        {
            for (var row = 0; row < height; row++)
            {
                for (var column = 0; column < width; column++)
                {
                    var value = (ushort)random.Next(0, 65536);
                    values[layer, row, column] = value;
                    BitConverter.TryWriteBytes(source.AsSpan(layer * 1024 + row * 80 + column * 2), value);
                }
            }
        }

        var sourceBuffer = harness.Buffer(source);
        var wide = harness.Buffer(layers * 2048UL);
        var narrow = harness.Buffer(layers * 1024UL);
        harness.Begin();
        var encoder = harness.Tiler.BeginCompute(harness);
        harness.Tiler.EncodeDepthConversion(encoder, new MetalBufferSpan(sourceBuffer, 0, layers * 1024UL),
            new MetalBufferSpan(wide, 0, layers * 2048UL), MetalDepthConversion.Widen, float32: true, layout);
        MetalTiler.EndEncoding(encoder);
        encoder = harness.Tiler.BeginCompute(harness);
        harness.Tiler.EncodeDepthConversion(encoder, new MetalBufferSpan(wide, 0, layers * 2048UL),
            new MetalBufferSpan(narrow, 0, layers * 1024UL), MetalDepthConversion.Narrow, float32: true,
            layout with
            {
                SourceRowStride = 160,
                TargetRowStride = 80,
                SourceSliceStride = 2048,
                TargetSliceStride = 1024,
            });
        MetalTiler.EndEncoding(encoder);
        harness.Submit();

        var wideBytes = harness.Read(wide, 0, layers * 2048UL);
        var narrowBytes = harness.Read(narrow, 0, layers * 1024UL);
        for (var layer = 0; layer < layers; layer++)
        {
            for (var row = 0; row < height; row++)
            {
                for (var column = 0; column < width; column++)
                {
                    uint value = values[layer, row, column];
                    Assert.Equal(BitConverter.SingleToUInt32Bits(value * (1.0f / 65535.0f)),
                        BitConverter.ToUInt32(wideBytes, layer * 2048 + row * 160 + column * 4));
                    Assert.Equal(value, BitConverter.ToUInt16(narrowBytes, layer * 1024 + row * 80 + column * 2));
                }

                Assert.Equal(0u, BitConverter.ToUInt32(wideBytes, layer * 2048 + row * 160 + (int)width * 4));
            }
        }
    }

    [Fact]
    public void SwapBgra16ExchangesTheOuterChannels()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var pixels = MetalTileOracle.Pattern(8 * 13, 19);
        var input = harness.Buffer(pixels);
        var output = harness.Buffer((ulong)pixels.Length);
        harness.Begin();
        var encoder = harness.Tiler.BeginCompute(harness);
        harness.Tiler.EncodeSwapBgra16(encoder, new MetalBufferSpan(input, 0, (ulong)pixels.Length),
            new MetalBufferSpan(output, 0, (ulong)pixels.Length));
        MetalTiler.EndEncoding(encoder);
        harness.Submit();

        var swapped = harness.Read(output, 0, (ulong)pixels.Length);
        for (var pixel = 0; pixel < 13; pixel++)
        {
            var offset = pixel * 8;
            Assert.Equal(pixels[(offset + 4)..(offset + 6)], swapped[offset..(offset + 2)]);
            Assert.Equal(pixels[(offset + 2)..(offset + 4)], swapped[(offset + 2)..(offset + 4)]);
            Assert.Equal(pixels[offset..(offset + 2)], swapped[(offset + 4)..(offset + 6)]);
            Assert.Equal(pixels[(offset + 6)..(offset + 8)], swapped[(offset + 6)..(offset + 8)]);
        }
    }

    // One pass of a 4096x4096 RGBA8 64 KiB-tiled detile moves 64 MiB in and 64 MiB out,
    // so the kernel is bandwidth-bound and the number is the figure of merit.
    [Fact]
    public void DetileThroughputIsBandwidthBound()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        const uint extent = 4096;
        const uint bytes = 4;
        if (!TileGeometry.TryGetBlockLayout(TileBlockKind.Standard64KB, bytes, out var block))
        {
            return;
        }

        var transfer = new MetalTileTransfer
        {
            Kind = TileBlockKind.Standard64KB,
            BytesPerElement = bytes,
            Width = extent,
            Height = extent,
            Depth = 1,
            Pitch = extent,
            TiledWidth = extent,
            TiledHeight = extent,
            TiledSize = (ulong)(extent / block.BlockWidth) * (extent / block.BlockHeight) * block.BlockSize,
            LinearSize = (ulong)extent * extent * bytes,
        };
        var batch = new List<MetalTileTransfer> { transfer };
        var tiled = harness.Buffer(transfer.TiledSize);
        var linear = harness.Buffer(transfer.LinearSize);

        // One warm pass compiles the library and the pipeline; the measured passes do not.
        for (var round = 0; round < 2; round++)
        {
            const int passes = 8;
            harness.Begin();
            var encoder = harness.Tiler.BeginCompute(harness);
            for (var pass = 0; pass < passes; pass++)
            {
                harness.Tiler.EncodeDetile(encoder, new MetalBufferSpan(tiled, 0, transfer.TiledSize),
                    new MetalBufferSpan(linear, 0, transfer.LinearSize), batch);
            }

            MetalTiler.EndEncoding(encoder);
            var started = Stopwatch.GetTimestamp();
            harness.Submit();
            var elapsed = Stopwatch.GetElapsedTime(started).TotalSeconds;
            if (round == 0)
            {
                continue;
            }

            var traffic = (transfer.TiledSize + transfer.LinearSize) * (ulong)passes;
            output.WriteLine($"detile {extent}x{extent} rgba8 64KB: passes={passes} " +
                             $"traffic={traffic / (1024.0 * 1024.0):F0} MiB elapsed={elapsed * 1000:F2} ms " +
                             $"throughput={traffic / elapsed / 1e9:F2} GB/s");
            Assert.True(elapsed > 0);
        }

        output.WriteLine($"staging buffers={harness.Tiler.StagingBufferCount} bytes={harness.Tiler.StagingBytes}");
    }
}
