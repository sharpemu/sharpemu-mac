// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// A compute shader that binds 4 KiB inside a GPU-written 1080p RGBA16F render target as a
/// storage buffer publishes the run it binds, not the 16 MiB target (MetalImageCache.ConsumerRuns.cs).
/// Every scene runs with the run publication on and off: the bytes the shader sees, and the
/// whole target read back afterwards, must be the same both ways.
/// </summary>
public sealed class ConsumerRunSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Width = 1920;
    private const uint Height = 1080;
    private const uint BytesPerTexel = 8;
    private const uint Dwords = 1024;
    private const uint BufferFormat = 20;

    // The bound bytes start inside a row, a third of the way down the target.
    private const ulong BoundOffset = (Width * BytesPerTexel * 360) + 0x1F00;

    /// <summary>buffer_load_dword from s[0:3], buffer_store_dword to s[4:7], one dword a thread.</summary>
    private static readonly uint[] CopyDwords =
    [
        0x8F098608,                // s_lshl_b32 s9, s8, 6
        0x4A020009,                // v_add_nc_u32_e32 v1, s9, v0
        0xE0302000, 0x80000201,    // buffer_load_dword v2, v1, s[0:3], 0 idxen
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xE0702000, 0x80010201,    // buffer_store_dword v2, v1, s[4:7], 0 idxen
        0xBF810000,                // s_endpgm
    ];

    /// <summary>buffer_load_dword from s[0:3], plus 1000, buffer_store_dword back to s[0:3].</summary>
    private static readonly uint[] AddInPlace =
    [
        0x8F098608,                // s_lshl_b32 s9, s8, 6
        0x4A020009,                // v_add_nc_u32_e32 v1, s9, v0
        0xE0302000, 0x80000201,    // buffer_load_dword v2, v1, s[0:3], 0 idxen
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0x4A0404FF, 0x000003E8,    // v_add_nc_u32_e32 v2, 0x3e8, v2
        0xE0702000, 0x80000201,    // buffer_store_dword v2, v1, s[0:3], 0 idxen
        0xBF810000,                // s_endpgm
    ];

    private readonly record struct Outcome(byte[] Read, byte[] Again, byte[] Whole, long BindRetiled, long WholeRetiled);

    private Outcome? Run(bool subRange, bool written, uint tileMode)
    {
        using var rig = CreateRig();
        if (rig is null) return null;
        var cache = rig.Host.ImageCache;
        cache.SubRangeConsumers = subRange;
        var size = (ulong)Width * Height * BytesPerTexel;
        // Room for a tiled target's padding rows too.
        var target = rig.Memory.Allocate(size + (64 * 1024 * 16), 0x10000);
        var fill = new byte[size + (64 * 1024 * 16)];
        Array.Fill(fill, (byte)0x11);
        rig.Memory.Write(target, fill);

        // A gradient, so every row and column of the target holds different texels.
        var state = new SceneState(Width, Height).ColorTarget(0, target, Width, Height, format: 12, numberType: 7,
            tileMode: tileMode);
        SceneVertex[] quad =
        [
            SceneVertex.At(-1, -1, 0.5f, 0, 0, 1, 1), SceneVertex.At(1, -1, 0.5f, 1, 0, 0, 1),
            SceneVertex.At(-1, 1, 0.5f, 0, 1, 0, 1), SceneVertex.At(-1, 1, 0.5f, 0, 1, 0, 1),
            SceneVertex.At(1, -1, 0.5f, 1, 0, 0, 1), SceneVertex.At(1, 1, 0.5f, 1, 1, 1, 0.5f),
        ];
        var draw = new ScenePm4();
        rig.DrawMesh(draw, state, quad);
        rig.Submit(draw);

        var bound = target + BoundOffset;
        var copy = rig.Memory.Allocate(Dwords * 4, 256);
        var before = cache.WritebackBytes;
        var pm4 = new ScenePm4();
        if (written)
        {
            rig.Dispatch(pm4, AddInPlace,
                [.. Descriptors.Buffer(bound, 4, Dwords, BufferFormat), .. Descriptors.Buffer(copy, 4, Dwords, BufferFormat)],
                Dwords / 64);
            pm4.Event(ScenePm4.EventCsPartialFlush);
        }

        rig.Dispatch(pm4, CopyDwords,
            [.. Descriptors.Buffer(bound, 4, Dwords, BufferFormat), .. Descriptors.Buffer(copy, 4, Dwords, BufferFormat)],
            Dwords / 64);
        rig.Submit(pm4);
        var bindRetiled = cache.WritebackBytes - before;
        var read = rig.Read(copy, (int)(Dwords * 4));

        // The same bytes bound again publish nothing more.
        var again = rig.Memory.Allocate(Dwords * 4, 256);
        var repeat = new ScenePm4();
        rig.Dispatch(repeat, CopyDwords,
            [.. Descriptors.Buffer(bound, 4, Dwords, BufferFormat), .. Descriptors.Buffer(again, 4, Dwords, BufferFormat)],
            Dwords / 64);
        rig.Submit(repeat);
        var repeatRetiled = cache.WritebackBytes - before - bindRetiled;
        if (subRange && !written)
        {
            Assert.Equal(0, repeatRetiled);
        }

        before = cache.WritebackBytes;
        var whole = rig.Read(target, (int)size);
        var wholeRetiled = cache.WritebackBytes - before;
        Assert.Empty(rig.Fatals);
        return new Outcome(read, rig.Read(again, (int)(Dwords * 4)), whole, bindRetiled, wholeRetiled);
    }

    [Theory]
    [InlineData(false, 0u)]
    [InlineData(true, 0u)]
    [InlineData(false, 27u)]
    [InlineData(true, 27u)]
    public void AComputeBindingOfFourKibInsideATargetRetilesItsRunAndReadsWhatAWholeWritebackDoes(bool written, uint tileMode)
    {
        if (Run(subRange: false, written, tileMode) is not { } whole) return;
        var run = Run(subRange: true, written, tileMode)!.Value;
        var size = (long)Width * Height * BytesPerTexel;
        Output.WriteLine(
            $"written={written} tile={tileMode}: bind retiled {whole.BindRetiled} bytes whole, {run.BindRetiled} as a run " +
            $"({(double)whole.BindRetiled / Math.Max(1, run.BindRetiled):F0}x); the later full read retiled " +
            $"{whole.WholeRetiled} and {run.WholeRetiled}");

        // The bound bytes are the GPU's, not the fill the guest held before the draw.
        Assert.Contains(whole.Read, value => value != 0x11);
        var expected = whole.Whole.AsSpan((int)BoundOffset, (int)(Dwords * 4)).ToArray();
        Assert.Equal(expected, whole.Again);
        Assert.Equal(whole.Read, run.Read);
        Assert.Equal(whole.Again, run.Again);
        Assert.Equal(whole.Whole, run.Whole);
        if (!written)
        {
            Assert.Equal(expected, run.Read);
        }

        Assert.True(whole.BindRetiled >= size, $"the whole writeback retiled {whole.BindRetiled} of {size} bytes");
        Assert.True(run.BindRetiled > 0 && run.BindRetiled <= size / 8,
            $"the binding retiled {run.BindRetiled} of {size} bytes");
        Assert.True(run.WholeRetiled > 0, "the full read found nothing left to publish");
    }
}
