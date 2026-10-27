// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalSamplerViewClassesTests
{
    [Fact]
    public void ASharedSamplerRequiresEveryPairedViewToBeInteger()
    {
        ImageRequest[] images = [Image(GpuFormat.R32Uint), Image(GpuFormat.R32Sint), Image(GpuFormat.R32Sfloat)];
        List<SampledImagePair> pairs = [Pair(0, 0), Pair(1, 0), Pair(0, 1), Pair(2, 1), Pair(2, 2), Pair(1, 2)];
        var classes = new byte[4];
        MetalSamplerViewClasses.Resolve(pairs, images, classes);
        Assert.Equal(MetalSamplerViewClasses.AllInteger, classes[0]);
        Assert.NotEqual(MetalSamplerViewClasses.AllInteger, classes[1]);
        Assert.NotEqual(MetalSamplerViewClasses.AllInteger, classes[2]);
        Assert.Equal(0, classes[3]); // No pairing does not force integer filtering.
    }

    [Fact]
    public void InvalidImagesDoNotEstablishAPairingAndUnusedSamplerPairsAreIgnored()
    {
        ImageRequest[] images = [Image(GpuFormat.R32Uint)];
        List<SampledImagePair> pairs = [Pair(uint.MaxValue, 0), Pair(1, 0), Pair(0, uint.MaxValue), Pair(1, 1), Pair(0, 1)];
        var classes = new byte[2];
        MetalSamplerViewClasses.Resolve(pairs, images, classes);
        Assert.Equal(0, classes[0]);
        Assert.Equal(MetalSamplerViewClasses.AllInteger, classes[1]);
    }

    [Fact]
    public void ReusedScratchReflectsCurrentFormatsAndCurrentImageAndSamplerCounts()
    {
        ImageRequest[] images = [Image(GpuFormat.R32Uint), Image(GpuFormat.R32Sint)];
        List<SampledImagePair> pairs = [Pair(0, 0), Pair(1, 1)];
        var classes = new byte[3];
        MetalSamplerViewClasses.Resolve(pairs, images, classes);
        Assert.Equal(MetalSamplerViewClasses.AllInteger, classes[0]);
        Assert.Equal(MetalSamplerViewClasses.AllInteger, classes[1]);

        images[0] = Image(GpuFormat.R8G8B8A8Unorm);
        classes[2] = 99;
        MetalSamplerViewClasses.Resolve(pairs, images.AsSpan(0, 1), classes.AsSpan(0, 2));
        Assert.NotEqual(MetalSamplerViewClasses.AllInteger, classes[0]);
        Assert.Equal(0, classes[1]); // Stale capacity is not a bound image.
        Assert.Equal(99, classes[2]); // Stale sampler capacity is untouched.

        MetalSamplerViewClasses.Resolve([], images, classes);
        Assert.Equal(new byte[3], classes);
        MetalSamplerViewClasses.Resolve(pairs, images, Span<byte>.Empty);
    }

    [Fact]
    public void ClassificationMatchesThePreviousPerSamplerWalkForMixedPairGraphs()
    {
        var random = new Random(137);
        for (var sample = 0; sample < 100; sample++)
        {
            var images = Enumerable.Range(0, random.Next(0, 12))
                .Select(_ => Image(random.Next(3) switch
                {
                    0 => GpuFormat.R32Uint,
                    1 => GpuFormat.R32Sint,
                    _ => GpuFormat.R32Sfloat,
                })).ToArray();
            var pairs = Enumerable.Range(0, random.Next(0, 40))
                .Select(_ => Pair((uint)random.Next(0, 15), (uint)random.Next(0, 12))).ToList();
            var classes = new byte[random.Next(0, 10)];
            MetalSamplerViewClasses.Resolve(pairs, images, classes);
            for (var sampler = 0; sampler < classes.Length; sampler++)
            {
                var views = pairs.Where(pair => pair.Sampler == sampler && pair.Image < images.Length)
                    .Select(pair => images[(int)pair.Image].View.Format).ToArray();
                Assert.Equal(views.Length != 0 && views.All(ViewFormatRules.IsIntegerFormat),
                    classes[sampler] == MetalSamplerViewClasses.AllInteger);
            }
        }
    }

    private static ImageRequest Image(GpuFormat format) =>
        new() { View = ImageViewDescription.Default with { Format = format } };

    private static SampledImagePair Pair(uint image, uint sampler) => new() { Image = image, Sampler = sampler };
}
