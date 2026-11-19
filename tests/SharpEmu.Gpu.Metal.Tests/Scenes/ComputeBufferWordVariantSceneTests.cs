// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Compute pipelines specialized for a dispatch's buffer words (MetalRenderHost.BufferWordVariants),
/// through the whole command path: PM4 dispatch, translation with the words as function
/// constants, CommitBindings(Compute), BindPipeline(Compute), Dispatch. The program loads a
/// formatted record by index (the stride and the format word are run-time words) and stores
/// what it read: the first dispatch of a layout runs the generic pipeline while its variant
/// builds, later ones the variant, which writes the same bytes; a pipeline keeps a bounded
/// number of variants; the compute switch keeps every dispatch generic; and a variant's
/// library file is stored and found under a key of its own.
/// </summary>
public sealed class ComputeBufferWordVariantSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Index = 2;
    private const uint Records = 8;
    private const int InputBytes = 1024;
    private const uint Float32x4 = 77;
    private const uint Unorm8x4 = 56;
    private const uint Unorm16x2 = 23;
    private static readonly TimeSpan BuildTimeout = TimeSpan.FromSeconds(60);
    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    /// <summary>
    /// BUFFER_LOAD_FORMAT_XYZW of record s0 from the V# in s[8:11] (IDXEN: the stride multiplies
    /// the index), its four dwords stored as read to the V# in s[12:15].
    /// </summary>
    private static readonly uint[] LoadIndexedFormat =
    [
        0x7E020200,             // v_mov_b32 v1, s0
        0xE00C2000, 0x80020401, // buffer_load_format_xyzw v[4:7], v1, s[8:11] idxen
        0xE0700000, 0x80030400, // buffer_store_dword v4, off, s[12:15]
        0xE0700004, 0x80030500, // buffer_store_dword v5, off, s[12:15] offset:4
        0xE0700008, 0x80030600, // buffer_store_dword v6, off, s[12:15] offset:8
        0xE070000C, 0x80030700, // buffer_store_dword v7, off, s[12:15] offset:12
        0xBF810000,             // s_endpgm
    ];

    [Fact]
    public void AComputeVariantIsStoredUnderAnotherArchiveKeyThanTheGenericPipeline()
    {
        var library = new UInt128(0x1111, 0x2222);
        var first = new UInt128(1, 2);
        var second = new UInt128(3, 4);

        Assert.Equal(library, MetalRenderHost.ComputeBufferWordArchiveKey(library, default));
        Assert.NotEqual(library, MetalRenderHost.ComputeBufferWordArchiveKey(library, first));
        Assert.NotEqual(
            MetalRenderHost.ComputeBufferWordArchiveKey(library, first),
            MetalRenderHost.ComputeBufferWordArchiveKey(library, second));
        // Another program's variant of the same values is another file.
        Assert.NotEqual(
            MetalRenderHost.ComputeBufferWordArchiveKey(library, first),
            MetalRenderHost.ComputeBufferWordArchiveKey(new UInt128(0x3333, 0x4444), first));
        // A module with no key stores no variant: the values alone would name one file for every program.
        Assert.Equal(default, MetalRenderHost.ComputeBufferWordArchiveKey(default, first));
        // Not the key a render variant's pair is listed under.
        Assert.NotEqual(
            MetalRenderHost.BufferWordArchiveKey(library, first), MetalRenderHost.ComputeBufferWordArchiveKey(library, first));
    }

    [Fact]
    public void TheComputeSwitchFollowsTheConstantsAndItsVariable() =>
        Assert.Equal(
            MetalRenderHost.BufferWordConstants &&
                Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BUFFER_WORD_COMPUTE") != "0",
            MetalRenderHost.BufferWordComputeVariants);

    [Fact]
    public void TheFirstDispatchOfALayoutRunsGenericAndTheVariantWritesTheSameBytes()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        RequireVariants(rig);
        using var gate = new ManualResetEventSlim(false);
        rig.Host.BeforeBufferWordVariantBuild = () => gate.Wait(BuildTimeout);
        try
        {
            var input = Input(rig);
            foreach (var (format, stride) in new[] { (Float32x4, 16u), (Unorm8x4, 12u), (Unorm16x2, 20u), (Float32x4, 36u) })
            {
                gate.Reset();
                var before = rig.Host.BufferWordComputeCounters;
                var generic = Dispatch(rig, input, format, stride);
                var during = rig.Host.BufferWordComputeCounters;
                Assert.Equal(
                    (before.DrawsVariant, before.DrawsGeneric + 1, before.DrawsPending + 1, before.Built),
                    (during.DrawsVariant, during.DrawsGeneric, during.DrawsPending, during.Built));
                Assert.Equal(1, rig.Host.BufferWordVariantsPending);

                gate.Set();
                Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
                var specialized = Dispatch(rig, input, format, stride);
                var after = rig.Host.BufferWordComputeCounters;
                Output.WriteLine(
                    $"format={format} stride={stride} generic={string.Join(",", generic.Select(Hex))} " +
                    $"built={after.Built} variant={after.DrawsVariant} failed={after.Failed}");
                Assert.Equal((before.Built + 1, before.Failed, during.DrawsVariant + 1), (after.Built, after.Failed, after.DrawsVariant));
                Assert.Equal(generic, specialized);
                Assert.Contains(generic, word => word != 0);
                if (format == Float32x4)
                {
                    // The record's own four dwords: the stride picked the record.
                    Assert.Equal(rig.Read<uint>(input + Index * stride, 4), generic);
                }
            }

            // Draws kept their own counters.
            Assert.Equal(0L, rig.Host.BufferWordCounters.DrawsGeneric + rig.Host.BufferWordCounters.DrawsVariant);
            Assert.Empty(rig.Fatals);
        }
        finally
        {
            gate.Set();
        }
    }

    [Fact]
    public void LayoutsPastThePipelinesCapRunGenericAndAreCounted()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        RequireVariants(rig);
        using var gate = new ManualResetEventSlim(false);
        rig.Host.BeforeBufferWordVariantBuild = () => gate.Wait(BuildTimeout);
        try
        {
            var input = Input(rig);
            // A stride that changes from dispatch to dispatch: one layout each.
            static uint Stride(int layout) => 16 + 4 * (uint)layout;
            var results = new List<uint[]>();
            for (var layout = 0; layout <= MetalRenderHost.MaxBufferWordVariantsPerPipeline; layout++)
            {
                results.Add(Dispatch(rig, input, Float32x4, Stride(layout)));
            }

            var counters = rig.Host.BufferWordComputeCounters;
            Assert.Equal(MetalRenderHost.MaxBufferWordVariantsPerPipeline + 1L, counters.DrawsGeneric);
            Assert.Equal(1L, counters.DrawsCapped);
            Assert.Equal(MetalRenderHost.MaxBufferWordVariantsPerPipeline, rig.Host.BufferWordVariantsPending);

            gate.Set();
            Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
            // The layout past the cap stays generic, now and later; the others run their variants.
            var capped = MetalRenderHost.MaxBufferWordVariantsPerPipeline;
            Assert.Equal(results[capped], Dispatch(rig, input, Float32x4, Stride(capped)));
            Assert.Equal(results[0], Dispatch(rig, input, Float32x4, Stride(0)));
            counters = rig.Host.BufferWordComputeCounters;
            Assert.Equal((long)MetalRenderHost.MaxBufferWordVariantsPerPipeline, counters.Built);
            Assert.Equal(2L, counters.DrawsCapped);
            Assert.Equal(1L, counters.DrawsVariant);
            for (var layout = 0; layout < results.Count; layout++)
            {
                Assert.Equal(rig.Read<uint>(input + Index * Stride(layout), 4), results[layout]);
            }
        }
        finally
        {
            gate.Set();
        }
    }

    [Fact]
    public void TheComputeSwitchOffKeepsEveryDispatchGeneric()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        RequireVariants(rig);
        rig.Host.ComputeBufferWordVariants = false;
        var input = Input(rig);
        var first = Dispatch(rig, input, Unorm8x4, 12);
        Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
        var second = Dispatch(rig, input, Unorm8x4, 12);

        var counters = rig.Host.BufferWordComputeCounters;
        Assert.Equal((0L, 0L, 2L, 0L), (counters.Built, counters.DrawsVariant, counters.DrawsGeneric, counters.DrawsPending));
        Assert.Equal(2L, rig.Host.BufferWordComputeDispatchesOff);
        Assert.Equal(0, rig.Host.BufferWordVariantsPending);
        Assert.Equal(first, second);
    }

    [Fact]
    public void TheNextHostFindsTheComputeVariantInTheArchive()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            Output.WriteLine("Scenes run on the device only behind the exclusive device gate (SHARPEMU_TEST_REQUIRE_DEVICE=1).");
            return;
        }

        var root = Path.Combine(Path.GetTempPath(), $"sharpemu-compute-buffer-words-{Guid.NewGuid():N}");
        try
        {
            var first = Run(root);
            Output.WriteLine($"first: {first}");
            // The generic pipeline's library file, and the variant's of its own.
            Assert.Equal(2, first.LibrariesWritten);

            var second = Run(root);
            Output.WriteLine($"second: {second}");
            // Both pipelines come from the archive: the generic one from its library's file,
            // the variant, function and pipeline, from the file stored under its key (the
            // generic file's library would ignore the values); nothing is written again.
            Assert.Equal((2, 2, 0, 0), (second.LibraryHits, second.PipelineHits, second.PipelineMisses, second.LibrariesWritten));
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    private (int LibraryHits, int LibraryMisses, int Rejected, int LibrariesWritten,
        int PipelineHits, int PipelineMisses, int PipelinesAdded, int LoadedSegments) Run(string root)
    {
        var archive = MetalBinaryArchive.Open(root, Device.Value, "translator-a", 64L << 20);
        var previous = MetalBinaryArchive.Install(archive);
        try
        {
            using (var rig = CreateRig()!)
            {
                RequireVariants(rig);
                var input = Input(rig);
                var generic = Dispatch(rig, input, Unorm8x4, 12);
                Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
                Assert.Equal(generic, Dispatch(rig, input, Unorm8x4, 12));
                var counters = rig.Host.BufferWordComputeCounters;
                Assert.Equal((1L, 1L, 0L), (counters.Built, counters.DrawsVariant, counters.Failed));
            }

            archive.Flush(wait: true);
            return archive.Counters;
        }
        finally
        {
            MetalBinaryArchive.Install(previous);
            archive.Dispose();
        }
    }

    // The scenes translate as the host does: with portable buffers and the words as constants.
    private static void RequireVariants(SceneRig rig)
    {
        Assert.True(MetalRenderHost.BufferWordConstants, "Buffer-word constants are off in this process.");
        Assert.True(rig.Host.ComputeBufferWordVariants || !MetalRenderHost.BufferWordComputeVariants);
    }

    // A byte pattern no two records share, so a wrong stride or format reads other bytes.
    private static ulong Input(SceneRig rig)
    {
        var address = rig.Memory.Allocate(InputBytes, 256);
        var bytes = new byte[InputBytes];
        for (var at = 0; at < bytes.Length; at++)
        {
            bytes[at] = (byte)((at * 37) + 11);
        }

        rig.Memory.Write<byte>(address, bytes);
        return address;
    }

    // One dispatch of the program over record Index of the input, read as format/stride; the four dwords it stored.
    private static uint[] Dispatch(SceneRig rig, ulong input, uint format, uint stride)
    {
        var output = rig.Memory.Allocate(16, 256);
        rig.Memory.Write<uint>(output, [0, 0, 0, 0]);
        var userData = new uint[16];
        userData[0] = Index;
        Descriptors.Buffer(input, stride, Records, format).CopyTo(userData, 8);
        Descriptors.Buffer(output, 4, 4, format: 20).CopyTo(userData, 12);
        rig.Submit(rig.Dispatch(new ScenePm4(), LoadIndexedFormat, userData, 1));
        return rig.Read<uint>(output, 4);
    }

    private static string Hex(uint value) => $"0x{value:X8}";
}
