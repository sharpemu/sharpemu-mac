// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The dword-wide variant of the one- and two-byte block copies. It is chosen only when
// the destination side of the copy is dword-aligned, so every kind is exercised both
// ways: a pitch that is a multiple of four takes the wide path, a pitch that is not
// falls back to the masked atomics, and both have to agree with the CPU twin.
public sealed partial class MetalImageTransferTests
{
    // The tiled depth writeback reads the depth texture in a compute kernel instead of
    // blitting its plane into a staging buffer, so every shape that kernel has a variant
    // for has to come back byte for byte: both element sizes, the layered texture its
    // array variant binds, and a width whose last element pair the row cuts in half.
    [Fact]
    public void TiledDepthPlanesWriteBackStraightOutOfTheirTexture()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        RoundTrip(harness, Depth(GuestDepthFormat.Z32Float, stencil: false, 97, 33, 1), 61, "z32/odd width");
        RoundTrip(harness, Depth(GuestDepthFormat.Z32Float, stencil: false, 96, 48, 3), 62, "z32/3 layers");
        RoundTrip(harness, Depth(GuestDepthFormat.Z32Float, stencil: true, 96, 48, 2), 63, "z32+s8/2 layers");
        // A 16-bit depth surface with no stencil keeps Metal's own Depth16Unorm texture,
        // so the writeback kernel narrows what the texture normalised.
        RoundTrip(harness, Depth(GuestDepthFormat.Z16, stencil: false, 97, 33, 2), 64, "z16/odd width/2 layers");
        // With stencil there is no 16-bit combined format, so the plane was widened on
        // the way in and the same kernel narrows it on the way out.
        RoundTrip(harness, Depth(GuestDepthFormat.Z16, stencil: true, 96, 48, 2), 65, "z16+s8/2 layers widened");
    }

    [Fact]
    public void WideGroupsFollowEachKindsOwnAddressing()
    {
        foreach (var kind in Enum.GetValues<TileBlockKind>())
        {
            for (uint bytes = 1; bytes <= 2; bytes++)
            {
                if (!MetalTilerShaders.Supports(kind, bytes))
                {
                    continue;
                }

                // The linear side's dword is always 4/bytes elements along x.
                Assert.Equal((4 / bytes, 1u), MetalTilerShaders.DwordGroup(kind, bytes, tiledSide: false));

                // The tiled side's dword follows the kind's addressing. The one-byte
                // depth and GCN 1D-thin kinds interleave x and y in the two low address
                // bits, so their dword is a 2x2 quad; every other kind keeps x's low bits
                // there and takes a run along x.
                var quad = kind is TileBlockKind.Depth64KB or TileBlockKind.GcnThin1D && bytes == 1;
                Assert.Equal(quad ? (2u, 2u) : (4 / bytes, 1u), MetalTilerShaders.DwordGroup(kind, bytes, tiledSide: true));

                // One thread covers whole dwords on both sides, so the quad kinds'
                // one-byte copies take two of each and everything else takes one.
                Assert.Equal(quad ? (4u, 2u) : (4 / bytes, 1u), MetalTilerShaders.WideGroup(kind, bytes));
            }
        }

        // Four bytes and wider copy whole elements, so they have no element group.
        Assert.Equal((0u, 0u), MetalTilerShaders.WideGroup(TileBlockKind.Standard64KB, 4));
    }

    [Fact]
    public void SubDwordBlockCopiesMatchTheCpuTwinOnBothPaths()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var failures = new List<string>();
        var cases = 0;
        uint seed = 71;
        foreach (var kind in Enum.GetValues<TileBlockKind>())
        {
            for (uint bytes = 1; bytes <= 2; bytes++)
            {
                if (!TileGeometry.TryGetBlockLayout(kind, bytes, out var block))
                {
                    continue;
                }

                // Both a pitch the wide variant can take and one it cannot, and a width
                // whose last element group is cut short by the row.
                foreach (var aligned in new[] { true, false })
                {
                    var transfer = new MetalTileTransfer
                    {
                        Kind = kind,
                        BytesPerElement = bytes,
                        Width = block.BlockWidth + 3,
                        Height = block.BlockHeight + 5,
                        Depth = block.BlockDepth == 1 ? 1u : block.BlockDepth + 2,
                    };
                    var pitchElements = (transfer.Width + 7) / 8 * 8;
                    transfer.Pitch = aligned ? pitchElements : pitchElements + 1;
                    transfer.TiledWidth = (transfer.Width + block.BlockWidth - 1) / block.BlockWidth * block.BlockWidth;
                    transfer.TiledHeight = (transfer.Height + block.BlockHeight - 1) / block.BlockHeight * block.BlockHeight;
                    var columns = transfer.TiledWidth / block.BlockWidth;
                    var rows = transfer.TiledHeight / block.BlockHeight;
                    var slabs = (transfer.Depth + block.BlockDepth - 1) / block.BlockDepth;
                    transfer.TiledSize = (ulong)columns * rows * slabs * block.BlockSize;
                    transfer.LinearSize = (ulong)transfer.Pitch * bytes * transfer.Height * transfer.Depth;
                    if (!aligned && transfer.Pitch * bytes % 4 == 0)
                    {
                        continue;
                    }

                    var name = $"{kind}/{bytes}/pitch={transfer.Pitch}";
                    var linear = MetalTileOracle.Pattern((int)transfer.LinearSize, seed++);
                    var tiled = new byte[transfer.TiledSize];
                    MetalTileOracle.Transfer(linear, tiled, transfer, toTiled: true);

                    var batch = new List<MetalTileTransfer> { transfer };
                    var tiledSpan = (transfer.TiledSize + 3) & ~3UL;
                    var linearSpan = (transfer.LinearSize + 3) & ~3UL;
                    var tiledSource = harness.Buffer(tiled);
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
                    var expected = linear.ToArray();
                    // A detiled row's padding is never written, so it has to be zero on
                    // the GPU side and is masked out of the reference.
                    MetalTileOracle.MaskPadding(expected, transfer);
                    var padding = actual.ToArray();
                    MetalTileOracle.MaskPadding(padding, transfer);
                    if (!actual.AsSpan().SequenceEqual(padding))
                    {
                        failures.Add($"{name}: the detile wrote into the row padding");
                    }

                    if (!padding.AsSpan().SequenceEqual(expected))
                    {
                        failures.Add($"{name}: detile differs from the CPU twin");
                    }

                    if (!harness.Read(retiled, 0, transfer.TiledSize).AsSpan().SequenceEqual(tiled))
                    {
                        failures.Add($"{name}: retile does not invert detile");
                    }

                    cases++;
                }
            }
        }

        output.WriteLine($"sub-dword cases={cases}");
        Assert.True(cases >= 30, $"cases={cases}");
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
