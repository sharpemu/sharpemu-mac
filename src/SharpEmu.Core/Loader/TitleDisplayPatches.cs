// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Memory;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Per-title display-size patches, chosen by <see cref="EnvironmentVariable"/>:
/// <list type="bullet">
/// <item>unset or <c>1</c>: rewrite every compiled 3840x2160 size to the display while the
/// title stays in its 4K mode. The fastest. The UI lays itself out on the shrunken canvas, and
/// for Demon's Souls <see cref="ImageScaleSitesFor"/> keeps the images it sizes from their 4K
/// textures in proportion to it.</item>
/// <item><c>mode</c>: run the title in its own HD display mode, held there
/// (<see cref="DisplayModeProfile"/>), the PS5's own path for a 1080p television: the output,
/// the display buffers and the UI stay 1920x1080 and the presenter scales that to the window;
/// below 1080p only the internal scene size shrinks to the display.</item>
/// <item><c>0</c>: the title's own sizes, as a PS5 runs it: Demon's Souls outputs 4K
/// whatever the television is and the system scales it.</item>
/// </list>
/// Every site is verified against the bytes the known build carries before anything is
/// written; a different build is left untouched.
/// </summary>
internal static class TitleDisplayPatches
{
    public const string EnvironmentVariable = "SHARPEMU_TITLE_DISPLAY_PATCH";

    /// <summary>What a site encodes. 32-bit widths and heights are little-endian integers;
    /// a <see cref="Slot.WidthHeight"/> site packs width in the low and height in the high
    /// dword of one 64-bit immediate; the float slots are IEEE singles.</summary>
    internal enum Slot
    {
        Width,
        Height,
        WidthHeight,
        WidthF32,
        HeightF32,
    }

    internal enum Role
    {
        /// <summary>The output buffer, display size or UI canvas: becomes the display size.</summary>
        Output,

        /// <summary>The internal render size: never grows, shrinks to the display size.</summary>
        Internal,
    }

    internal readonly record struct Site(ulong Offset, Slot Slot, uint OldWidth, uint OldHeight, Role Role);

    internal enum Strategy
    {
        /// <summary>The title's own sizes.</summary>
        Off,

        /// <summary>Every compiled 4K size rewritten to the display; the title stays in its 4K mode.</summary>
        AllSites,

        /// <summary>The title's own HD display mode; see <see cref="DisplayModeProfile"/>.</summary>
        DisplayMode,
    }

    /// <summary>A code patch that is not a size: <see cref="New"/> replaces <see cref="Old"/>.</summary>
    internal readonly record struct CodeSite(ulong Offset, byte[] Old, byte[] New);

    /// <summary>A title that picks its display mode as "override, else a compiled default".
    /// Each <see cref="DefaultModeSites"/> offset is the imm32 of a <c>mov r32, OldDefaultMode</c>
    /// that becomes <see cref="HdMode"/>; <see cref="OverrideSites"/> drop the overrides (a saved
    /// visual-mode option among them) so the mode holds. The HD output stays
    /// <see cref="HdWidth"/> x <see cref="HdHeight"/>: the title composites its UI over the
    /// scene at that size, so the output, display buffers and UI canvas must agree on it.
    /// <see cref="HdBufferSites"/>, compiled 4K sizes the HD mode still allocates or reports,
    /// become that size. Below it only <see cref="HdInternalSites"/>, the internal scene size,
    /// shrink to the display.</summary>
    internal sealed record DisplayModeProfile(
        IReadOnlyList<ulong> DefaultModeSites,
        uint OldDefaultMode,
        uint HdMode,
        IReadOnlyList<CodeSite> OverrideSites,
        uint HdWidth,
        uint HdHeight,
        IReadOnlyList<Site> HdBufferSites,
        IReadOnlyList<Site> HdInternalSites)
    {
        /// <summary>The mode for the display; null keeps the title's own 4K mode.</summary>
        public uint? ModeFor(uint width, uint height) =>
            width >= 3840 || height >= 2160 ? null : HdMode;
    }

    // Unset keeps the fast rewrite of every site until the display mode has been checked in a run.
    internal static Strategy SelectedStrategy =>
        Environment.GetEnvironmentVariable(EnvironmentVariable) switch
        {
            "0" => Strategy.Off,
            "mode" => Strategy.DisplayMode,
            _ => Strategy.AllSites,
        };

    // Demon's Souls (PPSA01342, v01.004.000). The video-out setup creates its "DisplayBuffer"
    // textures and registers them at 3840x2160 (immediates at the call sites and in the two
    // display-size getters), the render configs in read-only data carry the output and
    // display size, the mode tables (index 2 performance, 3 cinematic) carry the display and
    // internal sizes, and the 4K entries of the UI canvas tables (0x2172C04 / 0x2172A0C) and
    // the 3840.0 / 2160.0 literals (0x1135E17, 0x1FF9C60, 0x1FFEA50, 0x1FFE588) normalise
    // positions authored in 4K pixels. The title stays in mode 3, so its UI keeps the 4K layout.
    private static readonly Site[] DemonsSouls =
    [
        new(0x94535C, Slot.Width, 3840, 2160, Role.Output),
        new(0x945361, Slot.Height, 3840, 2160, Role.Output),
        new(0x94542E, Slot.Width, 3840, 2160, Role.Output),
        new(0x945433, Slot.Height, 3840, 2160, Role.Output),
        new(0x944A48, Slot.WidthHeight, 3840, 2160, Role.Output),
        new(0x945004, Slot.WidthHeight, 3840, 2160, Role.Output),
        new(0x945579, Slot.WidthHeight, 3840, 2160, Role.Output),
        new(0x9E7F88, Slot.Width, 3840, 2160, Role.Output),
        new(0x9E7FB8, Slot.Height, 3840, 2160, Role.Output),
        new(0x20010AC, Slot.WidthHeight, 3840, 2160, Role.Output),
        new(0x20010B4, Slot.WidthHeight, 3840, 2160, Role.Output),
        new(0x20013EC, Slot.WidthHeight, 3840, 2160, Role.Output),
        new(0x20013F4, Slot.WidthHeight, 3840, 2160, Role.Output),
        new(0x20010A4, Slot.Width, 2560, 1440, Role.Internal),
        new(0x20010A8, Slot.Height, 2560, 1440, Role.Internal),
        new(0x20013E4, Slot.WidthHeight, 3840, 2160, Role.Internal),
        new(0x1FFA5D8, Slot.Width, 3840, 2160, Role.Output),
        new(0x1FFA5DC, Slot.Width, 3840, 2160, Role.Output),
        new(0x1FFBE38, Slot.Height, 3840, 2160, Role.Output),
        new(0x1FFBE3C, Slot.Height, 3840, 2160, Role.Output),
        new(0x1FFBE28, Slot.Width, 2560, 1440, Role.Internal),
        new(0x1FFBE2C, Slot.Width, 3840, 2160, Role.Internal),
        new(0x1FFD758, Slot.Height, 2560, 1440, Role.Internal),
        new(0x1FFD75C, Slot.Height, 3840, 2160, Role.Internal),
        new(0x1135E17, Slot.WidthF32, 3840, 2160, Role.Output),
        new(0x1135E1B, Slot.HeightF32, 3840, 2160, Role.Output),
        new(0x1FF9C60, Slot.WidthF32, 3840, 2160, Role.Output),
        new(0x1FF9C64, Slot.HeightF32, 3840, 2160, Role.Output),
        new(0x1FFEA50, Slot.WidthF32, 3840, 2160, Role.Output),
        new(0x1FFE588, Slot.HeightF32, 3840, 2160, Role.Output),
        new(0x2172C04, Slot.WidthF32, 3840, 2160, Role.Output),
        new(0x2172A0C, Slot.HeightF32, 3840, 2160, Role.Output),
    ];

    // Demon's Souls display modes: 1 and 2 output 1920x1080 (internal 1600x900, 1920x1080),
    // 3 and 4 output 3840x2160 (internal 2560x1440, 3840x2160); tables at 0x1FFA5D0 / 0x1FFBE30
    // (output width / height) and 0x1FFBE20 / 0x1FFD750 (internal width / height). The mode is
    // r_Wants4K, else r_Wants4K_Game, else a compiled 3, worked out in three places: the
    // video-out mode set (0xA63E40), the renderer (0xBC8055) and the UI (0xEB436B). Every 4K
    // decision keys off mode 3 or 4: the registered display buffers (0x9449B0), the
    // display-size getters (0x9E7F80 / 0x9E7FB0), the window margins (0x945745), the UI canvas
    // ({1920,3840} x {1080,2160} at 0x2172C00 / 0x2172A08, read at 0xEB4022 and 0xECBCCE) and
    // the UI's 4K layout switch (0xEB43B9). A saved visual mode (r_Wants4K_Game 3 or 4) takes
    // over from the default once the save loads, so the two cmovne at each site go. The mode
    // (video-out object +0x3C, renderer copy at cfg+0xD18) has no other reader: no quality,
    // shadow, LOD or frame-rate setting keys off it, and the flip-rate calls (0x94531C,
    // 0x94560E, 0x9456D6, 0x9457F3) do not depend on it.
    //
    // Mode 2, not 1: its internal size is the HD output, so at 1080p the scene renders at
    // 1920x1080 as the all-sites rewrite renders it. The title composites its UI over the
    // upscaled scene at the output size, drawing the canvas 1:1 into it, and the HD output
    // (table index 1), the 1920x1080 views of the display buffers (0x9453E1 / 0x94547A, which
    // 0x944CB7 hands to sceVideoOutSetBufferAttribute2), the canvas tables and the display-size
    // getters all say 1920x1080. They stay so: a smaller registered buffer shows only its corner
    // of the composite. Below 1080p only the internal scene size (table index 1) shrinks, the
    // title upscales it to 1080 as it does from 1600x900 in mode 1, and the presenter scales
    // the 1920x1080 buffer to the window.
    //
    // The HD mode still allocates its "DisplayBuffer" textures at 3840x2160 (0x94535B /
    // 0x94542D) and takes its 1920x1080 views as mip 0 of them; the buffers it registers with
    // video out are those textures' storage (0x944D4D) under their tiling (0x944CFA). They
    // become 1920x1080, the size of the view, as the all-sites rewrite makes them the display
    // size. So does the fixed 3840x2160 ceiling every render config carries (0x944A46 in the
    // mode set, 0x945002 in its getter, 0x945577 at set-up); its only known reader, 0x953500,
    // sizes a render target from it (0x953F37) only under a flag the configs leave clear.
    private static readonly byte[] Nop3 = [0x0F, 0x1F, 0x00];

    // Demon's Souls sizes a UI image from its texture: the texture's width and height divided by
    // the canvas height over the height the UI is authored at (0xEA5FD0: [ui+8] / [ui]), so one
    // texel stays one canvas pixel, on the understanding that the UI's textures are made for the
    // canvas. They are the 4K ones whatever the canvas, so on the 1920x1080 canvas the all-sites
    // rewrite makes, the header emblem, the list diamonds and the highlight bars come out twice
    // their size while the text and layout, scaled by the canvas, do not. The three callers that
    // divide a texture size by that ratio (0xF380B9, 0xF33BE0 and 0x1675F50) call a stand-in
    // placed in the int3 padding after it instead, which divides 2160, the canvas the
    // textures are made for, by the authored height and joins it at 0xEA5FD5. Its other caller,
    // pixel snapping (0xEC855A / 0xEC857A), keeps the canvas.
    private const ulong ImageScaleRatio = 0xEA5FD0;
    private const ulong ImageScaleStandIn = 0xEA5FE1;

    private static readonly CodeSite[] DemonsSoulsImageScaleSites =
    [
        // mov eax, 0x45070000 (2160.0f); vmovd xmm0, eax; jmp 0xEA5FD5 (mov eax, [rdi]; ...)
        new(ImageScaleStandIn,
            [0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC],
            [0xB8, 0x00, 0x00, 0x07, 0x45, 0xC5, 0xF9, 0x6E, 0xC0, 0xEB, 0xE9]),
        RedirectedCall(0xF380B9),
        RedirectedCall(0xF33BE0),
        RedirectedCall(0x1675F50),
    ];

    private static CodeSite RedirectedCall(ulong offset) =>
        new(offset, CallRel32(offset, ImageScaleRatio), CallRel32(offset, ImageScaleStandIn));

    private static byte[] CallRel32(ulong offset, ulong target)
    {
        var bytes = new byte[5];
        bytes[0] = 0xE8;
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(1), checked((int)((long)target - (long)(offset + 5))));
        return bytes;
    }

    /// <summary>The code patches that keep a title's texture-sized UI images in proportion to the
    /// canvas the all-sites rewrite shrinks; null for a title without them.</summary>
    internal static IReadOnlyList<CodeSite>? ImageScaleSitesFor(string? titleId) =>
        titleId switch
        {
            "PPSA01342" => DemonsSoulsImageScaleSites,
            _ => null,
        };

    private static readonly DisplayModeProfile DemonsSoulsModes = new(
        [0xA63E41, 0xBC8056, 0xEB436C],
        OldDefaultMode: 3,
        HdMode: 2,
        [
            new(0xA63E47, [0x0F, 0x45, 0xF1], Nop3),
            new(0xA63E4C, [0x0F, 0x45, 0xF0], Nop3),
            new(0xBC806F, [0x0F, 0x45, 0xF1], Nop3),
            new(0xBC8074, [0x0F, 0x45, 0xF0], Nop3),
            new(0xEB4372, [0x0F, 0x45, 0xC1], Nop3),
            new(0xEB437A, [0x0F, 0x45, 0xC3], Nop3),
        ],
        HdWidth: 1920,
        HdHeight: 1080,
        [
            new(0x94535C, Slot.Width, 3840, 2160, Role.Output),
            new(0x945361, Slot.Height, 3840, 2160, Role.Output),
            new(0x94542E, Slot.Width, 3840, 2160, Role.Output),
            new(0x945433, Slot.Height, 3840, 2160, Role.Output),
            new(0x944A48, Slot.WidthHeight, 3840, 2160, Role.Output),
            new(0x945004, Slot.WidthHeight, 3840, 2160, Role.Output),
            new(0x945579, Slot.WidthHeight, 3840, 2160, Role.Output),
        ],
        [
            new(0x1FFBE24, Slot.Width, 1920, 1080, Role.Internal),
            new(0x1FFD754, Slot.Height, 1920, 1080, Role.Internal),
        ]);

    internal static DisplayModeProfile? DisplayModeFor(string? titleId) =>
        titleId switch
        {
            "PPSA01342" => DemonsSoulsModes,
            _ => null,
        };

    internal static IReadOnlyList<Site>? SitesFor(string? titleId) =>
        titleId switch
        {
            "PPSA01342" => DemonsSouls,
            _ => null,
        };

    /// <summary>Rewrites the title's compiled display sizes to <paramref name="width"/> x
    /// <paramref name="height"/>. Returns true when the sites were rewritten.</summary>
    public static bool Apply(
        IVirtualMemory memory,
        IReadOnlyList<ProgramHeader> programHeaders,
        ulong imageBase,
        string? titleId,
        uint width,
        uint height)
    {
        ArgumentNullException.ThrowIfNull(memory);
        ArgumentNullException.ThrowIfNull(programHeaders);
        var strategy = SelectedStrategy;
        if (strategy == Strategy.Off || !EngineResolution.IsEnabled || width == 0 || height == 0)
        {
            return false;
        }

        List<GuestBytePatcher.Patch>? patches;
        if (strategy == Strategy.DisplayMode)
        {
            var profile = DisplayModeFor(titleId);
            patches = profile is null ? null : BuildDisplayModePatches(profile, width, height);
        }
        else
        {
            var sites = SitesFor(titleId);
            patches = sites is null ? null : BuildPatches(sites, width, height);
            if (patches is { Count: > 0 } && ImageScaleSitesFor(titleId) is { } imageSites)
            {
                patches.AddRange(imageSites.Select(site => CreatePatch(site.Offset, site.Old, site.New)));
            }
        }

        if (patches is null || patches.Count == 0)
        {
            return false;
        }

        // A build other than the one the sites were taken from keeps every byte: verify all
        // of them before the patcher (which verifies again) writes any.
        var current = new byte[patches.Max(patch => patch.New.Length)];
        foreach (var patch in patches)
        {
            var span = current.AsSpan(0, patch.New.Length);
            if (!memory.TryRead(imageBase + patch.Offset, span) || !span.SequenceEqual(patch.Old))
            {
                Console.Error.WriteLine(
                    $"[LOADER][WARN] {titleId}: display size patch skipped, the executable does not carry the " +
                    $"expected bytes at 0x{patch.Offset:X} (a different build?).");
                return false;
            }
        }

        GuestBytePatcher.Apply(memory, programHeaders, imageBase, patches);
        if (strategy == Strategy.DisplayMode)
        {
            var profile = DisplayModeFor(titleId)!;
            EngineResolution.Report(
                "Bluepoint",
                $"display mode {profile.ModeFor(width, height)}",
                profile.HdWidth,
                profile.HdHeight,
                $"scene {Math.Min(width, profile.HdWidth)}x{Math.Min(height, profile.HdHeight)}");
        }
        else
        {
            EngineResolution.Report("Bluepoint", "compiled display sizes", width, height, $"{patches.Count} sites");
        }

        Console.Error.WriteLine(strategy == Strategy.DisplayMode
            ? $"[LOADER][INFO] {titleId}: starts in its display mode {DisplayModeFor(titleId)!.ModeFor(width, height)} " +
              $"for a {width}x{height} display ({patches.Count} sites, {EnvironmentVariable}=mode)."
            : $"[LOADER][INFO] {titleId}: compiled display sizes rewritten to {width}x{height} at {patches.Count} sites " +
              $"({EnvironmentVariable}=mode runs its HD mode, =0 keeps its own sizes).");
        return true;
    }

    /// <summary>The rewrites that hold the title in its HD mode and, below the HD size, shrink
    /// that mode's internal scene size to the display. The output, display buffers and
    /// everything the UI lays out in are the HD size.</summary>
    internal static List<GuestBytePatcher.Patch> BuildDisplayModePatches(
        DisplayModeProfile profile, uint width, uint height)
    {
        var patches = new List<GuestBytePatcher.Patch>();
        if (profile.ModeFor(width, height) is not { } mode)
        {
            return patches;
        }

        var old = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(old, profile.OldDefaultMode);
        var replacement = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(replacement, mode);
        foreach (var offset in profile.DefaultModeSites)
        {
            patches.Add(CreatePatch(offset, old, replacement));
        }

        foreach (var site in profile.OverrideSites)
        {
            patches.Add(CreatePatch(site.Offset, site.Old, site.New));
        }

        patches.AddRange(BuildPatches(profile.HdBufferSites, profile.HdWidth, profile.HdHeight));

        // The scene only shrinks: a display between 1080p and 4K keeps the HD size.
        var sceneWidth = Math.Min(width, profile.HdWidth);
        var sceneHeight = Math.Min(height, profile.HdHeight);
        patches.AddRange(BuildPatches(profile.HdInternalSites, sceneWidth, sceneHeight));
        return patches;
    }

    private static GuestBytePatcher.Patch CreatePatch(ulong offset, byte[] old, byte[] replacement) =>
        new(
            offset,
            replacement,
            old,
            $"0x{offset:X}={Convert.ToHexString(replacement).ToLowerInvariant()}?{Convert.ToHexString(old).ToLowerInvariant()}");

    internal static List<GuestBytePatcher.Patch> BuildPatches(IReadOnlyList<Site> sites, uint width, uint height)
    {
        var patches = new List<GuestBytePatcher.Patch>(sites.Count);
        foreach (var site in sites)
        {
            var newWidth = site.Role == Role.Internal ? Math.Min(width, site.OldWidth) : width;
            var newHeight = site.Role == Role.Internal ? Math.Min(height, site.OldHeight) : height;
            if (newWidth == site.OldWidth && newHeight == site.OldHeight)
            {
                continue;
            }

            var old = Encode(site.Slot, site.OldWidth, site.OldHeight);
            var replacement = Encode(site.Slot, newWidth, newHeight);
            if (old.AsSpan().SequenceEqual(replacement))
            {
                continue;
            }

            patches.Add(new GuestBytePatcher.Patch(
                site.Offset,
                replacement,
                old,
                $"0x{site.Offset:X}={Convert.ToHexString(replacement).ToLowerInvariant()}?{Convert.ToHexString(old).ToLowerInvariant()}"));
        }

        return patches;
    }

    private static byte[] Encode(Slot slot, uint width, uint height)
    {
        switch (slot)
        {
            case Slot.Width:
            {
                var bytes = new byte[4];
                BinaryPrimitives.WriteUInt32LittleEndian(bytes, width);
                return bytes;
            }

            case Slot.Height:
            {
                var bytes = new byte[4];
                BinaryPrimitives.WriteUInt32LittleEndian(bytes, height);
                return bytes;
            }

            case Slot.WidthHeight:
            {
                var bytes = new byte[8];
                BinaryPrimitives.WriteUInt64LittleEndian(bytes, width | ((ulong)height << 32));
                return bytes;
            }

            case Slot.WidthF32:
            {
                var bytes = new byte[4];
                BinaryPrimitives.WriteSingleLittleEndian(bytes, width);
                return bytes;
            }

            case Slot.HeightF32:
            {
                var bytes = new byte[4];
                BinaryPrimitives.WriteSingleLittleEndian(bytes, height);
                return bytes;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(slot), slot, null);
        }
    }
}
