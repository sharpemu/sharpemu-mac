// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.CLI;

/// <summary>
/// Static composition of the graphics backend and the host platform adapters.
///
/// Which packages exist in the build is decided by the publish RID, not at runtime: the
/// SHARPEMU_GPU_* constants come from the conditional ItemGroups in SharpEmu.CLI.csproj, so an
/// osx-x64 artifact links SharpEmu.Gpu.Metal and SharpEmu.Platform.MacOS and contains no
/// Vulkan, MoltenVK or SDL at all, while win-x64/linux-x64 link SharpEmu.Gpu.Vulkan and
/// SharpEmu.Adapters.Sdl. There is deliberately no runtime probe and no fallback: a build has
/// exactly one backend, and asking for another one is an error at registration time.
/// </summary>
internal static class GpuComposition
{
    /// <summary>
    /// Installs the process's GPU backend, presenter, display catalog and launcher pad. Must run
    /// before anything touches the renderer — that is, before emulation or the GUI starts.
    /// </summary>
    public static void Register()
    {
#if SHARPEMU_GPU_METAL
        // CA1416 cannot see that SHARPEMU_GPU_METAL is only defined for osx-* RIDs, so it
        // reads these [SupportedOSPlatform("macos")] calls as reachable everywhere. Both
        // throw PlatformNotSupportedException if the process somehow is not on macOS, so a
        // duplicate OperatingSystem.IsMacOS() here would only hide the real guard.
#pragma warning disable CA1416
        SharpEmu.Platform.MacOS.MacPlatform.Register();
        if (SharpEmu.GpuSplit.SplitGpuComposition.IsRequested)
        {
            // SHARPEMU_GPU_PROCESS=1: the Metal backend runs in the native arm64 GPU host
            // (SharpEmu.GpuHost), which starts with the first use of the renderer; this process
            // talks to it through proxies (docs/macos/split-protocol.md).
            SharpEmu.GpuSplit.SplitGpuComposition.Register();
        }
        else
        {
            SharpEmu.Gpu.Metal.MetalGpuBackend.Register();
        }
#pragma warning restore CA1416
#elif SHARPEMU_GPU_VULKAN
        SharpEmu.Adapters.Sdl.SdlPlatform.Register();
        SharpEmu.Gpu.Vulkan.VulkanGpuBackend.Register();
        // RenderDoc needs its layer allowed through the Vulkan loader before the first
        // instance is created; the Metal build has no loader to configure.
        SharpEmu.Libs.VideoOut.RenderDocCapture.ApplyGraphicsLoaderEnvironment();
#else
#error No GPU backend was composed. SharpEmu.CLI.csproj must define SHARPEMU_GPU_METAL or SHARPEMU_GPU_VULKAN for the target RID.
#endif
    }

    /// <summary>
    /// Stops what the backend runs outside this process once the guest has stopped, before the
    /// runtime unmaps guest memory. Only the GPU-process split has anything to stop.
    /// </summary>
    public static void Shutdown()
    {
#if SHARPEMU_GPU_METAL
#pragma warning disable CA1416
        if (SharpEmu.GpuSplit.SplitGpuComposition.IsRequested)
        {
            SharpEmu.GpuSplit.SplitGpuComposition.Shutdown();
        }
#pragma warning restore CA1416
#endif
    }
}
