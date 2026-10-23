// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed partial class ComputeSceneTests
{
    [Fact]
    public void SegmentedDispatchesSurviveAnArchiveWrittenWithoutBatching()
    {
        using (var probe = CreateRig())
            if (probe is null) return;
        var device = MetalNative.MTLCreateSystemDefaultDevice();
        var root = Path.Combine(Path.GetTempPath(), $"sharpemu-batch-archive-{Guid.NewGuid():N}");
        try
        {
            // First persist only the ordinary pipeline. Next boot recovers the
            // missing batch kernel; the third boot must hit both archived variants.
            for (var boot = 0; boot < 3; boot++)
            {
                using var archive = MetalBinaryArchive.Open(root, device, "batch-test", 64L << 20);
                var previous = MetalBinaryArchive.Install(archive);
                try
                {
                    using (var rig = CreateRig()!)
                    {
                        rig.Host.ConcurrentComputeEnabled = true;
                        rig.Host.ComputeBatchEnabled = boot != 0;
                        var (pm4, outputs) = ArchiveBatchStream(rig);
                        rig.Submit(pm4);
                        var before = rig.Host.ComputeBatchedDispatches;
                        rig.Submit(pm4);
                        Assert.Equal(boot == 0 ? 0 : 4, rig.Host.ComputeBatchedDispatches - before);
                        Assert.Equal(boot == 1 ? 1 : 0, rig.Host.ComputeBatchRecompiles);
                        Assert.Equal(0, rig.Host.ComputeBatchFallbacks);
                        Assert.Empty(rig.Fatals);
                        foreach (var result in outputs)
                            Assert.All(rig.Read<uint>(result, 64), value => Assert.Equal(18u, value));
                    }
                    Assert.Equal(boot == 0 ? 0 : boot == 1 ? 1 : 2, archive.Counters.LibraryHits);
                    Assert.Equal(boot < 2 ? 1 : 0, archive.Counters.LibrariesWritten);
                    if (boot == 2) Assert.Equal(2, archive.Counters.PipelineHits);
                }
                finally { MetalBinaryArchive.Install(previous); }
            }
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
            MetalNative.SendVoid(device, MetalNative.Selector("release"));
        }
    }

    [Fact]
    public void AMissingSegmentedEntryFallsBackWithTheOriginalDispatchBindings()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.ConcurrentComputeEnabled = true;
        rig.Host.ComputeBatchEnabled = false;
        var (pm4, outputs) = ArchiveBatchStream(rig);
        rig.Submit(pm4);

        // Simulate stale reflection claiming an optional function absent even from
        // the source. Both the archive/library lookup and recompile must be harmless.
        var modules = (System.Collections.IEnumerable)typeof(MetalRenderHost)
            .GetField("_modules", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(rig.Host)!;
        foreach (var module in modules)
        {
            if (module is null) continue;
            var compiled = module.GetType().GetField("Compiled")!;
            var shader = (MetalCompiledGuestShader)compiled.GetValue(module)!;
            compiled.SetValue(module, new MetalCompiledGuestShader(shader.AcquireShader() with
            {
                ComputeBatchEntryPoint = "missing_batch_entry",
            }));
        }
        foreach (var result in outputs) rig.Memory.Write<uint>(result, new uint[64]);
        rig.Host.ComputeBatchEnabled = true;
        rig.Submit(pm4);
        Assert.Equal(0, rig.Host.ComputeBatchedDispatches);
        Assert.Equal(1, rig.Host.ComputeBatchRecompiles);
        Assert.Equal(4, rig.Host.ComputeBatchFallbacks);
        Assert.Empty(rig.Fatals);
        foreach (var result in outputs)
            Assert.All(rig.Read<uint>(result, 64), value => Assert.Equal(18u, value));
    }

    private static (ScenePm4 Stream, ulong[] Outputs) ArchiveBatchStream(SceneRig rig)
    {
        rig.Program(IncrementBuffer);
        rig.Memory.Allocate(4096, 4096);
        var input = Fill(rig, 64, 17);
        var outputs = Enumerable.Range(0, 4).Select(_ => Fill(rig, 64, 0)).ToArray();
        var pm4 = new ScenePm4();
        foreach (var result in outputs)
            Setup(rig, pm4, IncrementBuffer, [.. Vsharp(input, 4, 64), .. Vsharp(result, 4, 64)], 64)
                .DispatchDirect(1, 1, 1);
        return (pm4, outputs);
    }
}
