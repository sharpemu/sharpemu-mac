// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Versioning;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.Gpu;
using SharpEmu.Gpu.Metal.V2;

namespace SharpEmu.Gpu.Metal;

/// <summary>
/// Composition root of the Metal backend package. The application calls <see cref="Register"/>
/// once at startup; the shared engine then resolves the renderer through <c>GuestGpu.Current</c>
/// without naming Metal, and never has to ask what OS it is on.
///
/// The renderer is the v2 host: an address-keyed image cache over no-copy guest memory.
/// </summary>
[SupportedOSPlatform("macos")]
public static class MetalGpuBackend
{
    public const string Name = "Metal";

    /// <summary>
    /// The upstream resource and image features of sync d467d5867 that the Metal backend
    /// opts out of, back to what it had before that sync. With all of them on, Demon's
    /// Souls drew large hard-edged black areas in the Outpost Passage from about 150 s
    /// in world; with these off (and the synthetic ZPASS_DUMP pairs Metal keeps) it renders
    /// as before for the whole 300 s (run 20261019-013035-ds-metal-optout). Which of them
    /// is the cause is not isolated yet.
    /// Set when the backend is created, before it plans or binds anything.
    /// </summary>
    internal static void ConfigureSharedGpuCode()
    {
        SharpEmu.Libs.Gpu.Images.ImageRequestBuilders.NativeColorClearMetadata = false;
        SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.CompiledTransactions = false;
        SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.SpecializationGuards = false;
        SharpEmu.ShaderCompiler.Resources.ResourceTracker.UnvalidatedAddressWordsAreMemory = false;
        // Not an opt-out: binding words the GPU is still writing go to the device unless asked not to.
        SharpEmu.ShaderCompiler.Resources.DeviceBindingLoads.Enabled = !V2.MetalV2Diagnostics.WaitResourceWords;
    }

    public static void Register()
    {
        if (!OperatingSystem.IsMacOS())
        {
            throw new PlatformNotSupportedException(
                "SharpEmu.Gpu.Metal was composed into a process that is not running on macOS.");
        }

        GuestGpu.Register(static () =>
        {
            ConfigureSharedGpuCode();
            return new MetalV2Backend();
        }, Name);
        // V2 presents HDR on an EDR layer for --hdr=on, but Auto stays SDR: on an EDR screen
        // SILENT HILL's HDR path loads ~35 s slower and presents black, where SDR shows the
        // Konami logo by 12 s. Auto can follow the screen, as on Vulkan, once that is fixed.
        HostVideoHost.RegisterPresenter(
            MetalV2Presenter.TryConfigureVideo,
            MetalV2Presenter.QueryHdrOutputSupport,
            autoHdrAllowed: false);
        Console.Error.WriteLine($"[LOADER][INFO] GPU backend: {Name} v2.");
    }
}
