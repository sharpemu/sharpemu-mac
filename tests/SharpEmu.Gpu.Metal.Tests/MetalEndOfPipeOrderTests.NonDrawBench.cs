// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Text.Json;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalEndOfPipeOrderTests
{
    // SHARPEMU_NONDRAW_BENCH_OUTPUT=<path>: the worker's cost of one end-of-pipe label and one
    // occlusion dump, with no encoder open (the copy is its own transfer) and with a compute
    // encoder open (the copy is held for the encoder's end, as it is behind an open render pass),
    // and of the commit that lands what was held. Calls run in batches of 128, a commit between.
    [Fact]
    public void MeasureEndOfPipeAndOcclusionDumpWhenRequested()
    {
        var path = Environment.GetEnvironmentVariable("SHARPEMU_NONDRAW_BENCH_OUTPUT");
        if (path is null || !RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.NativeOcclusionEnabled = true;
        const ulong labels = GuestBase + 0x20000;
        const ulong queries = GuestBase + 0x30000;
        const int batch = 128;
        var results = new List<object>();
        // SHARPEMU_NONDRAW_BENCH_ROUNDS lengthens a run for a sampling profiler; ..._ONLY keeps the cases whose name contains it.
        var rounds = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_NONDRAW_BENCH_ROUNDS"), out var requested) ? requested : 14;
        var only = Environment.GetEnvironmentVariable("SHARPEMU_NONDRAW_BENCH_ONLY");

        void Measure(string name, bool computeOpen, Action<int> call)
        {
            if (only is not null && !name.Contains(only, StringComparison.Ordinal)) return;
            var callNs = new List<double>();
            var commitNs = new List<double>();
            long allocated = 0;
            long calls = 0;
            for (var round = 0; round < rounds; round++)
            {
                if (computeOpen) _ = host.EnsureComputeEncoder();
                var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
                var started = Stopwatch.GetTimestamp();
                for (var index = 0; index < batch; index++) call(index);
                var called = Stopwatch.GetTimestamp();
                var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
                var serial = host.FlushCommandBuffer();
                var flushed = Stopwatch.GetTimestamp();
                WaitForGpu(host, serial);
                WaitForCompletions(host);
                if (round < 4) continue;
                callNs.Add((called - started) * 1e9 / Stopwatch.Frequency / batch);
                commitNs.Add((flushed - called) * 1e9 / Stopwatch.Frequency);
                allocated += allocatedAfter - allocatedBefore;
                calls += batch;
            }

            callNs.Sort();
            commitNs.Sort();
            results.Add(new
            {
                name,
                computeOpen,
                nsPerCallMedian = callNs[callNs.Count / 2],
                nsPerCallBest = callNs[0],
                commitMicrosecondsMedian = commitNs[commitNs.Count / 2] / 1000,
                allocatedBytesPerCall = (double)allocated / calls,
            });
        }

        foreach (var computeOpen in new[] { false, true })
        {
            var open = computeOpen ? "held" : "immediate";
            Measure($"eop_write32_{open}", computeOpen, index =>
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, labels + (ulong)index * 16, (uint)index)));
            Measure($"eop_write64_{open}", computeOpen, index =>
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, labels + (ulong)index * 16, (ulong)index)));
            Measure($"eop_interrupt_only_{open}", computeOpen, index =>
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.InterruptOnly, 1, EventId: 0, ContextId: (uint)index)));
            Measure($"occlusion_dump_16_blocks_{open}", computeOpen, index =>
                Assert.True(host.DumpOcclusionCounter(0, queries + (ulong)index * 256, 0xFFFF, 16)));
            Measure($"occlusion_dump_1_block_{open}", computeOpen, index =>
                Assert.True(host.DumpOcclusionCounter(0, queries + (ulong)index * 256, 1, 8)));
            Measure($"cp_write_8_bytes_{open}", computeOpen, index =>
                host.RecordCommandProcessorWrite(labels + (ulong)index * 16, BitConverter.GetBytes((ulong)index), "WRITE_DATA"));
        }

        // The dump as it runs behind an open render pass: a visibility batch is open, so the copies are held for the
        // encoder's end. The batch is made by hand (no pass is drawn here) and rewound each round.
        {
            var hostType = typeof(SharpEmu.Gpu.Metal.V2.MetalRenderHost);
            const System.Reflection.BindingFlags any = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var batchType = hostType.GetNestedType("OcclusionBatch", System.Reflection.BindingFlags.NonPublic)!;
            var newBuffer = hostType.GetMethod("NewOcclusionBuffer", any)!;
            var counts = (nint)newBuffer.Invoke(host, [4096 * 8])!;
            var resultBuffer = (nint)newBuffer.Invoke(host, [4096 * 8])!;
            var openBatch = Activator.CreateInstance(batchType, any | System.Reflection.BindingFlags.Public, null, [counts, resultBuffer], null)!;
            hostType.GetField("_oqOpen", any)!.SetValue(host, openBatch);
            var slot = batchType.GetField("Slot", any | System.Reflection.BindingFlags.Public)!;
            foreach (var coalesce in new[] { false, true, false, true })
            {
                host.CoalesceOcclusionDumps = coalesce;
                Measure($"occlusion_dump_16_blocks_batch_open_coalesce_{(coalesce ? "on" : "off")}", true, index =>
                {
                    if (index == 0) slot.SetValue(openBatch, 0);
                    Assert.True(host.DumpOcclusionCounter(0, queries + (ulong)index * 256, 0xFFFF, 16));
                });
            }

            hostType.GetField("_oqOpen", any)!.SetValue(host, null);
            host.CoalesceOcclusionDumps = true;
        }

        File.WriteAllText(path, JsonSerializer.Serialize(new {
            architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(), results },
            new JsonSerializerOptions { WriteIndented = true }));
    }

    // SHARPEMU_LABEL_SEPARATOR_BENCH_OUTPUT=<path>: the GPU time of 200 pairs of tiny dispatches with each thing a
    // held-copy landing can put between the two of a pair: nothing, an encoder split (the fence chain), a split with a
    // blit encoder, the same with a device-only event before the blit, with a shared event, and a word-copy dispatch
    // inside the open encoder between two memory barriers. One command buffer per case, GPUStartTime to GPUEndTime,
    // cases interleaved per round; per pair is the case's time over 200, and the separator's cost is that minus the
    // case with nothing.
    [Fact]
    public void MeasureLabelSeparatorsWhenRequested()
    {
        var path = Environment.GetEnvironmentVariable("SHARPEMU_LABEL_SEPARATOR_BENCH_OUTPUT");
        if (path is null || !RequireDevice()) return;
        const int pairs = 200;
        const int rounds = 13;
        const int warmup = 3;
        var device = Device.Value;
        var pool = SharpEmu.Gpu.Metal.MetalNative.objc_autoreleasePoolPush();
        try
        {
            var queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            var tick = SeparatorBenchPipeline(device,
                "kernel void bench(device uint* words [[buffer(0)]], uint index [[thread_position_in_grid]]) { words[index] += 1u; }");
            var copy = SeparatorBenchPipeline(device,
                "kernel void bench(device const uint* source [[buffer(0)]], device uint* destination [[buffer(1)]], " +
                "uint index [[thread_position_in_grid]]) { if (index < 2u) destination[index] = source[index]; }");
            var selNewBuffer = MetalNative.Selector("newBufferWithLength:options:");
            var words = MetalNative.SendNewBuffer(device, selNewBuffer, 4096, 0);
            var staging = MetalNative.SendNewBuffer(device, selNewBuffer, 4096, 0);
            var labels = MetalNative.SendNewBuffer(device, selNewBuffer, 4096, 0);
            var fence = MetalNative.Send(device, MetalNative.Selector("newFence"));
            var selCommandBuffer = MetalNative.Selector("commandBuffer");
            var selCompute = MetalNative.Selector("computeCommandEncoder");
            var selBlit = MetalNative.Selector("blitCommandEncoder");
            var selPipeline = MetalNative.Selector("setComputePipelineState:");
            var selSetBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
            var selDispatch = MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:");
            var selWaitFence = MetalNative.Selector("waitForFence:");
            var selUpdateFence = MetalNative.Selector("updateFence:");
            var selEnd = MetalNative.Selector("endEncoding");
            var selCopy = MetalNative.Selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
            var selSignal = MetalNative.Selector("encodeSignalEvent:value:");
            var selWait = MetalNative.Selector("encodeWaitForEvent:value:");
            var selBarrier = MetalNative.Selector("memoryBarrierWithScope:");
            var selRelease = MetalNative.Selector("release");
            var group = new MtlSize { Width = 1, Height = 1, Depth = 1 };
            var threads = new MtlSize { Width = 64, Height = 1, Depth = 1 };
            // split_nofence leaves the fence out (Metal's own hazard tracking still orders the two encoders over the
            // tracked buffer); split_untracked also takes an untracked buffer, so nothing orders them: what a split
            // costs once the fence chain is taken away.
            var untracked = MetalNative.SendNewBuffer(device, selNewBuffer, 4096, 1 << 8);
            string[] cases = ["none", "split", "split_nofence", "split_untracked", "split_blit", "device_event", "shared_event", "in_encoder"];
            var fenced = true;
            var target = words;
            var samples = cases.ToDictionary(name => name, _ => new List<double>());

            nint OpenCompute(nint commandBuffer)
            {
                var encoder = MetalNative.Send(commandBuffer, selCompute);
                if (fenced) MetalNative.SendVoid(encoder, selWaitFence, fence);
                MetalNative.SendVoid(encoder, selPipeline, tick);
                MetalNative.SendSetBuffer(encoder, selSetBuffer, target, 0, 0);
                return encoder;
            }

            void EndCompute(nint encoder)
            {
                if (fenced) MetalNative.SendVoid(encoder, selUpdateFence, fence);
                MetalNative.SendVoid(encoder, selEnd);
            }

            void Blit(nint commandBuffer)
            {
                var blit = MetalNative.Send(commandBuffer, selBlit);
                MetalNative.SendVoid(blit, selWaitFence, fence);
                MetalNative.SendCopyBuffer(blit, selCopy, staging, 0, labels, 0, 8);
                MetalNative.SendVoid(blit, selUpdateFence, fence);
                MetalNative.SendVoid(blit, selEnd);
            }

            for (var round = 0; round < rounds; round++)
            {
                for (var which = 0; which < cases.Length; which++)
                {
                    var name = cases[(which + round) % cases.Length];
                    var commandBuffer = MetalNative.Send(queue, selCommandBuffer);
                    var @event = name switch
                    {
                        "device_event" => MetalNative.Send(device, MetalNative.Selector("newEvent")),
                        "shared_event" => MetalNative.Send(device, MetalNative.Selector("newSharedEvent")),
                        _ => 0,
                    };
                    var value = 0UL;
                    fenced = name is not ("split_nofence" or "split_untracked");
                    target = name == "split_untracked" ? untracked : words;
                    var encoder = OpenCompute(commandBuffer);
                    for (var pair = 0; pair < pairs; pair++)
                    {
                        MetalNative.SendDispatch(encoder, selDispatch, group, threads);
                        switch (name)
                        {
                            case "none":
                                break;
                            case "in_encoder":
                                SharpEmu.Gpu.Metal.V2.MetalV2Native.SendMemoryBarrierScope(encoder, selBarrier, SharpEmu.Gpu.Metal.V2.Mtl.BarrierScopeBuffers);
                                MetalNative.SendVoid(encoder, selPipeline, copy);
                                MetalNative.SendSetBuffer(encoder, selSetBuffer, staging, 0, 0);
                                MetalNative.SendSetBuffer(encoder, selSetBuffer, labels, 0, 1);
                                MetalNative.SendDispatch(encoder, selDispatch, group, threads);
                                SharpEmu.Gpu.Metal.V2.MetalV2Native.SendMemoryBarrierScope(encoder, selBarrier, SharpEmu.Gpu.Metal.V2.Mtl.BarrierScopeBuffers);
                                MetalNative.SendVoid(encoder, selPipeline, tick);
                                MetalNative.SendSetBuffer(encoder, selSetBuffer, words, 0, 0);
                                break;
                            default:
                                EndCompute(encoder);
                                if (@event != 0)
                                {
                                    value++;
                                    SharpEmu.Gpu.Metal.V2.MetalV2Native.SendEventValue(commandBuffer, selSignal, @event, value);
                                    SharpEmu.Gpu.Metal.V2.MetalV2Native.SendEventValue(commandBuffer, selWait, @event, value);
                                }

                                if (name is not ("split" or "split_nofence" or "split_untracked"))
                                {
                                    Blit(commandBuffer);
                                }

                                encoder = OpenCompute(commandBuffer);
                                break;
                        }

                        MetalNative.SendDispatch(encoder, selDispatch, group, threads);
                    }

                    EndCompute(encoder);
                    MetalNative.SendVoid(commandBuffer, MetalNative.Selector("commit"));
                    MetalNative.SendVoid(commandBuffer, MetalNative.Selector("waitUntilCompleted"));
                    Assert.Equal(4u, (uint)MetalNative.SendUInteger(commandBuffer, MetalNative.Selector("status")));
                    var elapsed = MetalNative.SendDouble(commandBuffer, MetalNative.Selector("GPUEndTime")) -
                        MetalNative.SendDouble(commandBuffer, MetalNative.Selector("GPUStartTime"));
                    if (@event != 0) MetalNative.SendVoid(@event, selRelease);
                    if (round >= warmup) samples[name].Add(elapsed * 1e6 / pairs);
                }
            }

            var none = Median(samples["none"]);
            var results = cases.Select(name => new
            {
                name,
                microsecondsPerPairMedian = Median(samples[name]),
                microsecondsPerPairBest = samples[name].Min(),
                separatorMicroseconds = Median(samples[name]) - none,
            }).ToList();
            foreach (var result in results)
            {
                output.WriteLine($"SEPARATOR {result.name}: {result.microsecondsPerPairMedian:F2} us/pair " +
                    $"(best {result.microsecondsPerPairBest:F2}), separator {result.separatorMicroseconds:F2} us");
            }

            File.WriteAllText(path, JsonSerializer.Serialize(new {
                architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(), pairs,
                rounds = rounds - warmup, results }, new JsonSerializerOptions { WriteIndented = true }));
            foreach (var handle in new[] { words, untracked, staging, labels, fence, tick, copy, queue }) MetalNative.SendVoid(handle, selRelease);
        }
        finally
        {
            SharpEmu.Gpu.Metal.MetalNative.objc_autoreleasePoolPop(pool);
        }

        static double Median(List<double> values)
        {
            var sorted = values.Order().ToList();
            return sorted[sorted.Count / 2];
        }
    }

    private static nint SeparatorBenchPipeline(nint device, string body)
    {
        var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalNative.ApplyMslCompileOptions(options);
        nint error = 0;
        var library = MetalNative.Send(device, MetalNative.Selector("newLibraryWithSource:options:error:"),
            MetalNative.NsString("#include <metal_stdlib>\nusing namespace metal;\n" + body), options, ref error);
        Assert.True(library != 0, MetalNative.DescribeError(error));
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("bench"));
        nint pipelineError = 0;
        var pipeline = MetalNative.Send(device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref pipelineError);
        Assert.True(pipeline != 0, MetalNative.DescribeError(pipelineError));
        return pipeline;
    }
}
