// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Diagnostics;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.Gpu;
using SharpEmu.Gpu.Vulkan.Buffers;
using SharpEmu.Gpu.Vulkan.Images;
using SharpEmu.Gpu.Vulkan.Presentation;

namespace SharpEmu.Gpu.Vulkan;

/// <summary>
/// Composition root of the Vulkan backend package, used on Windows and Linux. The application
/// calls <see cref="Register"/> once at startup; the shared engine then resolves the renderer
/// through <c>GuestGpu.Current</c> without naming Vulkan.
/// </summary>
public static class VulkanGpuBackend
{
    public const string Name = "Vulkan";

    public static void Register()
    {
        GuestGpu.Register(static () => new VulkanGuestGpuBackend(), Name);
        HostVideoHost.RegisterPresenter(
            VulkanVideoPresenter.TryConfigureVideo,
            VulkanVideoPresenter.QueryHdrOutputSupport,
            autoHdrAllowed: true);
        LoadProgressDiagnostics.RegisterBlockedCommandStreamProbe(
            VulkanVideoPresenter.SnapshotBlockedCommandStream);
        RenderDocCapture.RegisterLoaderEnvironment(VulkanLoaderEnvironment.Apply);
        // Guest threads reach Silk.NET's marshalling stubs through the flip path, so the
        // bindings have to be JIT-warmed before a guest stack can enter them.
        SharpEmu.HLE.ModuleManager.RegisterGuestReachableInterop("Silk.NET");
        // The resource caches behind the Vulkan presenter add their counters to the
        // shared per-window render report.
        RenderPhaseProfile.RegisterReportProvider(GuestBufferCache.TakeAsyncReadbackReport);
        RenderPhaseProfile.RegisterReportProvider(GpuSyncProfile.TakeReport);
        RenderPhaseProfile.RegisterReportProvider(GuestImageCache.TakeLookupReport);
        RenderPhaseProfile.RegisterReportProvider(GuestImageCache.TakeDccReport);
        VulkanBackendPerfReports.Register();
        RenderPhaseProfile.RegisterReportProvider(static () =>
        {
            SharpEmu.ShaderCompiler.Vulkan.ShaderTranslationProfile.Report();
            return string.Empty;
        });
    }
}
