// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The budget within which a program's images bind one per slot (ShaderResourcePlan.StableImageBindings), whatever
/// descriptors a draw repeats, so one specialization serves every alias partition of its image slots
/// (IShaderPipelineHost.ComputeImageDescriptorLimits, GraphicsImageDescriptorLimits; upstream answers them from the
/// Vulkan device limits). A plan over the budget keeps compact bindings, one element per distinct descriptor.
///
/// Every stage reads its images, samplers and buffers through its own argument buffer (Gen5MslArgumentLayout), written
/// with gpuResourceIDs and gpuAddresses: the argument buffers tier 2 layout. Nothing is shared between the stages of a
/// pipeline, so compute and graphics stages have one budget, the per-stage argument buffer limits of Apple's Metal
/// Feature Set Tables ("GPU implementation limits by family", Argument buffers; the limits count what a stage reaches
/// through every argument buffer it reads):
/// <list type="bullet">
/// <item>textures per stage: 1M on Apple6 to Apple11. Sampled and storage images are both textures, so their sum is
/// held to it through Resources, which also counts the buffers.</item>
/// <item>buffers per stage: no limit on Apple6 to Apple11.</item>
/// <item>samplers per stage: 996 on Apple7 and Apple8, 500K on Apple9 to Apple11; also no more than the unique
/// argument-buffer samplers the app may create, MTLDevice.maxArgumentBufferSamplerCount (1024 or more on macOS:
/// "Improving CPU performance by using argument buffers").</item>
/// </list>
/// A device without tier 2 argument buffers or below Apple7 (not a device this app runs on) answers no budget.
/// SHARPEMU_METAL_STABLE_IMAGE_BINDINGS=0 answers no budget: every program keeps compact bindings, as before.
/// </summary>
internal static class MetalStableImageLimits
{
    internal static readonly bool Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_STABLE_IMAGE_BINDINGS") != "0";

    internal const uint TexturesPerStage = 1_000_000;
    internal const uint SamplersPerStageApple7 = 996;
    internal const uint SamplersPerStageApple9 = 500_000;

    // MTLArgumentBuffersTier2.
    private const nint ArgumentBuffersTier2 = 1;
    private const nint GpuFamilyApple7 = 1007;
    private const nint GpuFamilyApple9 = 1009;

    private static readonly nint SelArgumentBuffersSupport = MetalNative.Selector("argumentBuffersSupport");
    private static readonly nint SelMaxArgumentBufferSamplerCount = MetalNative.Selector("maxArgumentBufferSamplerCount");
    private static readonly nint SelSupportsFamily = MetalNative.Selector("supportsFamily:");
    private static readonly nint SelRespondsToSelector = MetalNative.Selector("respondsToSelector:");

    private static readonly object Gate = new();
    private static nint _device;
    private static StableImageDescriptorLimits _deviceLimits;

    /// <summary>The budget this process answers for the device: the live host and the prewarm alike.</summary>
    internal static StableImageDescriptorLimits Of(nint device) => Of(device, Enabled);

    internal static StableImageDescriptorLimits Of(nint device, bool enabled)
    {
        if (!enabled || device == 0)
        {
            return default;
        }

        lock (Gate)
        {
            if (_device != device)
            {
                _deviceLimits = Query(device);
                _device = device;
            }

            return _deviceLimits;
        }
    }

    /// <summary>The budget of a device with these capabilities.</summary>
    internal static StableImageDescriptorLimits FromCapabilities(bool tier2, bool apple7, bool apple9, ulong maxUniqueSamplers)
    {
        if (!tier2 || !apple7 || maxUniqueSamplers == 0)
        {
            return default;
        }

        var samplers = (uint)Math.Min(apple9 ? SamplersPerStageApple9 : SamplersPerStageApple7, maxUniqueSamplers);
        return new StableImageDescriptorLimits(TexturesPerStage, TexturesPerStage, samplers, uint.MaxValue, TexturesPerStage);
    }

    /// <summary>The budget as the prewarm stamp keeps it: entries made under another budget are not taken as current.</summary>
    internal static string StampPart(StableImageDescriptorLimits limits) =>
        string.Create(CultureInfo.InvariantCulture,
            $"stable_images={limits.SampledImages},{limits.StorageImages},{limits.Samplers},{limits.StorageBuffers},{limits.Resources}");

    private static StableImageDescriptorLimits Query(nint device)
    {
        var tier2 = MetalNative.Send(device, SelArgumentBuffersSupport) == ArgumentBuffersTier2;
        var families = MetalNative.SendBool(device, SelRespondsToSelector, SelSupportsFamily);
        var apple7 = families && MetalNative.SendBool(device, SelSupportsFamily, GpuFamilyApple7);
        var apple9 = families && MetalNative.SendBool(device, SelSupportsFamily, GpuFamilyApple9);
        var samplers = (ulong)MetalNative.Send(device, SelMaxArgumentBufferSamplerCount);
        var limits = FromCapabilities(tier2, apple7, apple9, samplers);
        Console.Error.WriteLine(
            $"[METAL-V2][INFO] Stable image bindings: tier2={tier2} apple7={apple7} apple9={apple9} " +
            $"max_argument_buffer_samplers={samplers} -> {StampPart(limits)}");
        return limits;
    }
}
