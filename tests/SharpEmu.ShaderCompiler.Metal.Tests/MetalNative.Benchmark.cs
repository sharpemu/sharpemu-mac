// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// Timed dispatch, for measuring what a codegen change does on the device rather
// than arguing about it from the emitted text. Two numbers come back:
//
//   * GPU time, from the command buffer's own GPUStartTime/GPUEndTime, over a
//     dispatch wide enough that launch overhead does not dominate;
//   * the pipeline's maxTotalThreadsPerThreadgroup, which the driver derives from
//     the function's register footprint. It is the occupancy ceiling, and it moves
//     the moment register pressure does — a direct read on the thing the PC
//     dispatcher was wrecking.
internal static partial class MetalNative
{
    internal readonly record struct BenchmarkResult(
        double GpuMilliseconds,
        uint MaxThreadsPerThreadgroup,
        uint ThreadExecutionWidth,
        bool StepCapHit);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial double SendDouble(nint receiver, nint selector);

    /// <summary>Compiles and times a translated compute shader. The argument buffer is
    /// filled the same way the correctness path fills it, so the kernel sees the layout
    /// it was compiled against; the dispatch is <paramref name="threadgroups"/> groups of
    /// <paramref name="threadsPerThreadgroup"/> threads, repeated
    /// <paramref name="iterations"/> times in one command buffer.</summary>
    public static bool TryBenchmark(
        Gen5MslShader shader,
        byte[] bufferContents,
        byte[] pushData,
        uint threadgroups,
        uint threadsPerThreadgroup,
        uint iterations,
        out BenchmarkResult result,
        out string error,
        IReadOnlyList<DeviceTable>? tables = null,
        nuint? mathMode = null,
        bool groupSizeIsMultipleOfWidth = false)
    {
        result = default;
        if (!TryCompileLibrary(shader.Source, out var library, out error, mathMode))
        {
            return false;
        }

        var buffer = SharedBuffer(bufferContents);
        var stepReport = SharedBuffer(new byte[sizeof(uint)]);
        var pairLocks = SharedBuffer(new byte[Gen5MslArgumentLayout.PairLockTableWords * sizeof(uint)]);
        if (buffer == 0 || stepReport == 0 || pairLocks == 0)
        {
            error = "failed to create the data buffers";
            return false;
        }

        var arguments = BuildArgumentBuffer(
            shader.ArgumentLayout!,
            SendULong(buffer, Selector("gpuAddress")),
            (uint)bufferContents.Length,
            SendULong(stepReport, Selector("gpuAddress")),
            tables,
            SendULong(pairLocks, Selector("gpuAddress")));
        var argumentBuffer = SharedBuffer(arguments);
        var pushBuffer = SharedBuffer(pushData.Length == 0 ? new byte[4] : pushData);
        if (argumentBuffer == 0 || pushBuffer == 0)
        {
            error = "failed to create the argument or push buffer";
            return false;
        }

        var function = Send(library, Selector("newFunctionWithName:"), NsString(shader.EntryPoint));
        if (function == 0)
        {
            error = $"entry point '{shader.EntryPoint}' not found";
            return false;
        }

        var pipeline = NewBenchmarkPipeline(function, groupSizeIsMultipleOfWidth, out error);
        if (pipeline == 0)
        {
            return false;
        }

        var maxThreads = (uint)SendULong(pipeline, Selector("maxTotalThreadsPerThreadgroup"));
        var executionWidth = (uint)SendULong(pipeline, Selector("threadExecutionWidth"));
        var queue = Send(Device.Value, Selector("newCommandQueue"));
        if (queue == 0)
        {
            error = "failed to create the command queue";
            return false;
        }

        var groups = new MtlSize { Width = threadgroups, Height = 1, Depth = 1 };
        var threads = new MtlSize { Width = threadsPerThreadgroup, Height = 1, Depth = 1 };
        var best = double.MaxValue;

        // Warm up once so shader-cache and allocation costs stay out of the number,
        // then keep the fastest of the timed runs: the GPU is shared with whatever
        // else is on the machine, so the floor is the honest comparison.
        for (var pass = 0; pass < 6; pass++)
        {
            var commandBuffer = Send(queue, Selector("commandBuffer"));
            var encoder = Send(commandBuffer, Selector("computeCommandEncoder"));
            SendVoid(encoder, Selector("setComputePipelineState:"), pipeline);
            SendSetBuffer(
                encoder,
                Selector("setBuffer:offset:atIndex:"),
                argumentBuffer,
                0,
                (nuint)Gen5MslArgumentLayout.ResourcesBufferIndex);
            SendSetBuffer(
                encoder,
                Selector("setBuffer:offset:atIndex:"),
                pushBuffer,
                0,
                (nuint)Gen5MslArgumentLayout.PushDataBufferIndex);
            SendUseResource(encoder, Selector("useResource:usage:"), buffer, ReadWriteUsage);
            SendUseResource(encoder, Selector("useResource:usage:"), stepReport, ReadWriteUsage);
            SendUseResource(encoder, Selector("useResource:usage:"), pairLocks, ReadWriteUsage);
            SendUseResource(encoder, Selector("useResource:usage:"), SharedTable.Value, ReadWriteUsage);
            foreach (var table in tables ?? [])
            {
                foreach (var resident in (nint[])[table.Buffer, .. table.Residents])
                {
                    SendUseResource(encoder, Selector("useResource:usage:"), resident, ReadWriteUsage);
                }
            }
            for (var iteration = 0u; iteration < iterations; iteration++)
            {
                SendDispatch(encoder, Selector("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads);
            }

            SendVoid(encoder, Selector("endEncoding"));
            SendVoid(commandBuffer, Selector("commit"));
            SendVoid(commandBuffer, Selector("waitUntilCompleted"));
            var commandError = Send(commandBuffer, Selector("error"));
            if (commandError != 0)
            {
                error = "GPU execution failed: " + DescribeError(commandError);
                return false;
            }

            if (pass == 0)
            {
                continue;
            }

            var elapsed = SendDouble(commandBuffer, Selector("GPUEndTime")) -
                SendDouble(commandBuffer, Selector("GPUStartTime"));
            best = Math.Min(best, elapsed * 1000.0);
        }

        var stepContents = Send(stepReport, Selector("contents"));
        var hits = stepContents == 0 ? 0 : unchecked((uint)Marshal.ReadInt32(stepContents));
        result = new BenchmarkResult(best, maxThreads, executionWidth, hits != 0);
        return true;
    }

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint SendComputeDescriptor(
        nint receiver, nint selector, nint descriptor, nuint options, nint reflection, ref nint error);

    // A compute pipeline the way the emulator makes one, or through a descriptor that
    // declares the threadgroup size a multiple of the SIMD width.
    private static nint NewBenchmarkPipeline(nint function, bool groupSizeIsMultipleOfWidth, out string error)
    {
        error = string.Empty;
        nint nsError = 0;
        nint pipeline;
        if (groupSizeIsMultipleOfWidth)
        {
            var descriptor = Send(Send(objc_getClass("MTLComputePipelineDescriptor"), Selector("alloc")), Selector("init"));
            SendVoid(descriptor, Selector("setComputeFunction:"), function);
            SendVoidBool(descriptor, Selector("setThreadGroupSizeIsMultipleOfThreadExecutionWidth:"), true);
            pipeline = SendComputeDescriptor(
                Device.Value, Selector("newComputePipelineStateWithDescriptor:options:reflection:error:"), descriptor, 0, 0, ref nsError);
        }
        else
        {
            pipeline = Send(Device.Value, Selector("newComputePipelineStateWithFunction:error:"), function, ref nsError);
        }

        if (pipeline == 0)
        {
            error = $"pipeline creation failed: {DescribeError(nsError)}";
        }

        return pipeline;
    }

    /// <summary>Compiles and times a hand-written kernel with one buffer at indices 0 and 1,
    /// the same way as <see cref="TryBenchmark"/> (warm-up, then the best of five).</summary>
    public static bool TryBenchmarkKernel(
        string source,
        string entryPoint,
        byte[] bufferContents,
        uint threadgroups,
        uint threadsPerThreadgroup,
        uint iterations,
        out double gpuMilliseconds,
        out uint maxThreads,
        out string error,
        bool groupSizeIsMultipleOfWidth = false)
    {
        gpuMilliseconds = 0;
        maxThreads = 0;
        if (!TryCompileLibrary(source, out var library, out error))
        {
            return false;
        }

        var function = Send(library, Selector("newFunctionWithName:"), NsString(entryPoint));
        if (function == 0)
        {
            error = $"entry point '{entryPoint}' not found";
            return false;
        }

        var pipeline = NewBenchmarkPipeline(function, groupSizeIsMultipleOfWidth, out error);
        var buffer = SharedBuffer(bufferContents);
        var queue = Send(Device.Value, Selector("newCommandQueue"));
        if (pipeline == 0 || buffer == 0 || queue == 0)
        {
            error = error.Length != 0 ? error : "failed to create the buffer or queue";
            return false;
        }

        maxThreads = (uint)SendULong(pipeline, Selector("maxTotalThreadsPerThreadgroup"));
        var best = double.MaxValue;
        for (var pass = 0; pass < 6; pass++)
        {
            var commandBuffer = Send(queue, Selector("commandBuffer"));
            var encoder = Send(commandBuffer, Selector("computeCommandEncoder"));
            SendVoid(encoder, Selector("setComputePipelineState:"), pipeline);
            SendSetBuffer(encoder, Selector("setBuffer:offset:atIndex:"), buffer, 0, 0);
            SendSetBuffer(encoder, Selector("setBuffer:offset:atIndex:"), buffer, 0, 1);
            for (var iteration = 0u; iteration < iterations; iteration++)
            {
                SendDispatch(
                    encoder,
                    Selector("dispatchThreadgroups:threadsPerThreadgroup:"),
                    new MtlSize { Width = threadgroups, Height = 1, Depth = 1 },
                    new MtlSize { Width = threadsPerThreadgroup, Height = 1, Depth = 1 });
            }

            SendVoid(encoder, Selector("endEncoding"));
            SendVoid(commandBuffer, Selector("commit"));
            SendVoid(commandBuffer, Selector("waitUntilCompleted"));
            var commandError = Send(commandBuffer, Selector("error"));
            if (commandError != 0)
            {
                error = "GPU execution failed: " + DescribeError(commandError);
                return false;
            }

            if (pass != 0)
            {
                best = Math.Min(best, (SendDouble(commandBuffer, Selector("GPUEndTime")) -
                    SendDouble(commandBuffer, Selector("GPUStartTime"))) * 1000.0);
            }
        }

        gpuMilliseconds = best;
        return true;
    }

    private static byte[] BuildArgumentBuffer(
        Gen5MslArgumentLayout layout,
        ulong bufferAddress,
        uint bufferBytes,
        ulong stepReportAddress,
        IReadOnlyList<DeviceTable>? tables = null,
        ulong pairLocksAddress = 0)
    {
        var arguments = new byte[Math.Max(layout.ByteSize, 8)];
        foreach (var field in layout.Fields)
        {
            for (var element = 0; element < field.Count; element++)
            {
                switch (field.FieldKind)
                {
                    case MslArgumentFieldKind.BufferPointers:
                        BitConverter.TryWriteBytes(
                            arguments.AsSpan((int)field.ByteOffset + (element * 8)), bufferAddress);
                        break;
                    case MslArgumentFieldKind.BufferByteCounts:
                        BitConverter.TryWriteBytes(
                            arguments.AsSpan((int)field.ByteOffset + (element * 4)), bufferBytes);
                        break;
                    case MslArgumentFieldKind.StepReport:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset), stepReportAddress);
                        break;
                    case MslArgumentFieldKind.PairLocks:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset), pairLocksAddress);
                        break;
                    case MslArgumentFieldKind.Pointer:
                        BitConverter.TryWriteBytes(
                            arguments.AsSpan((int)field.ByteOffset),
                            FindTable(tables, field.Kind) is { } table ? SendULong(table.Buffer, Selector("gpuAddress")) : SharedTableAddress());
                        break;
                    case MslArgumentFieldKind.Word:
                        BitConverter.TryWriteBytes(
                            arguments.AsSpan((int)field.ByteOffset),
                            FindTable(tables, field.Kind) is { } counted ? counted.Count : SharedTableDwords);
                        break;
                }
            }
        }

        return arguments;
    }
}
