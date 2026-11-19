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
        // The image cache takes textures described from their first resident level (MetalImageCache.ResidentLevels.cs).
        // This is the process that runs the request builders and the cache alike: the guest process with one process,
        // the GPU host with two (GpuHostMain.ComposeMetal), which reads the same environment.
        SharpEmu.Libs.Gpu.Images.ImageRequestBuilders.PartialMipChains = V2.MetalImageCache.PartialMipChains;
        SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.CompiledTransactions = CompiledTransactionsDefault();
        SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.SpecializationGuards = false;
        SharpEmu.ShaderCompiler.Resources.ResourceTracker.UnvalidatedAddressWordsAreMemory = false;
    }

    // Compiled resource transactions are back on for the titles below. Grand Theft Auto V's draws
    // write new pointers into their user data every frame, so a materialization misses its cache
    // tens of thousands of times a second: in Los Santos gameplay ~55K misses per 5 s took 1.24 s of
    // the command stream through the evaluator and 0.75-0.88 s compiled, 9.46 -> 9.99 median fps
    // with the same pictures. Elsewhere they stay off until the Demon's Souls areas they were
    // turned off for are checked again. SHARPEMU_METAL_COMPILED_TRANSACTIONS=1 or =0 decides for
    // any title.
    private static readonly HashSet<string> CompiledTransactionTitles = new(StringComparer.Ordinal) { "PPSA04264", "PPSA03420" };

    private static bool CompiledTransactionsDefault()
    {
        var titleId = VideoOutBackendServices.GetApplicationTitleId();
        var enabled = Environment.GetEnvironmentVariable("SHARPEMU_METAL_COMPILED_TRANSACTIONS") switch
        {
            "1" => true,
            "0" => false,
            _ => CompiledTransactionTitles.Contains(titleId),
        };
        Console.Error.WriteLine($"[LOADER][INFO] Compiled resource transactions: {(enabled ? "on" : "off")} (title {titleId}).");
        return enabled;
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
