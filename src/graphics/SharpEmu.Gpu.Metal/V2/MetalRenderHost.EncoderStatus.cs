// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

// SHARPEMU_METAL_V2_ENCODER_STATUS=1 (diagnostic): command buffers are made with
// MTLCommandBufferErrorOptionEncoderExecutionStatus, render and compute encoders carry labels and
// every pipeline bind leaves a debug signpost naming its shaders (with
// SHARPEMU_METAL_V2_PROFILE_SHADERS=1, the vertex and pixel hashes; otherwise the vertex or compute
// one), so a submission that faults or hangs names the encoders that did not complete and the
// pipelines they ran.
internal sealed partial class MetalRenderHost
{
    private static readonly bool EncoderStatusEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ENCODER_STATUS") == "1";

    private const nuint ErrorOptionEncoderExecutionStatus = 1;
    private static readonly nint SelCommandBufferWithDescriptor = MetalNative.Selector("commandBufferWithDescriptor:");
    private static readonly nint SelInsertDebugSignpost = MetalNative.Selector("insertDebugSignpost:");
    private static readonly nint SelEncoderSetLabel = MetalNative.Selector("setLabel:");
    private static readonly Lazy<nint> EncoderStatusDescriptor = new(() =>
    {
        var descriptor = MetalNative.Send(MetalNative.Class("MTLCommandBufferDescriptor"), MetalNative.Selector("new"));
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setErrorOptions:"), ErrorOptionEncoderExecutionStatus);
        return descriptor;
    });

    private static long _labelledEncoders;

    private nint NewCommandBuffer() => EncoderStatusEnabled
        ? MetalNative.Send(RecordQueue, SelCommandBufferWithDescriptor, EncoderStatusDescriptor.Value)
        : MetalNative.Send(RecordQueue, SelCommandBuffer);

    private static void LabelEncoder(nint encoder, string kind)
    {
        if (!EncoderStatusEnabled || encoder == 0)
        {
            return;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            MetalNative.SendVoid(encoder, SelEncoderSetLabel,
                MetalNative.NsString($"{kind}#{Interlocked.Increment(ref _labelledEncoders)}"));
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    private static void SignpostPipeline(nint encoder, MetalPipelineEntry entry)
    {
        if (!EncoderStatusEnabled || encoder == 0)
        {
            return;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var text = entry.Profile ?? $"{(entry.Graphics ? "vs" : "cs")}=0x{entry.VertexHash:X16}";
            MetalNative.SendVoid(encoder, SelInsertDebugSignpost, MetalNative.NsString(text));
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    // The encoders of a failed submission that did not complete, with their last signposts.
    private static string DescribeEncoderStatus(nint error)
    {
        if (!EncoderStatusEnabled || error == 0)
        {
            return string.Empty;
        }

        var userInfo = MetalNative.Send(error, MetalNative.Selector("userInfo"));
        var infos = userInfo == 0
            ? 0
            : MetalNative.Send(userInfo, MetalNative.Selector("objectForKey:"),
                MetalNative.NsString("MTLCommandBufferEncoderInfoErrorKey"));
        if (infos == 0)
        {
            return " encoders=unreported";
        }

        var count = (int)MetalV2Native.SendValue(infos, MetalNative.Selector("count"));
        var lines = new List<string>();
        for (var index = 0; index < count; index++)
        {
            var info = MetalNative.Send(infos, MetalNative.Selector("objectAtIndex:"), (nint)index);
            var state = (int)MetalV2Native.SendValue(info, MetalNative.Selector("errorState"));
            if (state == 1)
            {
                continue;
            }

            var signposts = MetalNative.Send(info, MetalNative.Selector("debugSignposts"));
            var signpostCount = signposts == 0 ? 0 : (int)MetalV2Native.SendValue(signposts, MetalNative.Selector("count"));
            var shown = new List<string>();
            for (var post = Math.Max(0, signpostCount - 12); post < signpostCount; post++)
            {
                shown.Add(ReadNsString(MetalNative.Send(signposts, MetalNative.Selector("objectAtIndex:"), (nint)post)));
            }

            var stateName = state switch { 2 => "affected", 3 => "pending", 4 => "faulted", _ => "unknown" };
            lines.Add($"#{index} {ReadNsString(MetalNative.Send(info, MetalNative.Selector("label")))} {stateName} " +
                $"signposts={signpostCount} last=[{string.Join(" | ", shown)}]");
        }

        return $" encoders={count} not_completed=[{string.Join("; ", lines)}]";
    }

    private static string ReadNsString(nint value) => value == 0
        ? "?"
        : Marshal.PtrToStringUTF8(MetalNative.Send(value, MetalNative.Selector("UTF8String"))) ?? "?";
}
