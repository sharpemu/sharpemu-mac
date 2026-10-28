// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>What a one-pass texture transfer asks a texture before it reads or writes it.</summary>
internal readonly record struct MetalTileTextureInfo(
    MtlPixelFormat Format, MtlTextureUsage Usage, MtlTextureType Type, nuint Levels, nuint Slices);

/// <summary>
/// The properties and the unsigned-integer view the tiler's one-pass kernels move a texture
/// through, made once for each texture a cached image owns instead of on every transfer:
/// creating the view alone was about 10 µs of the recording thread, the bulk of a render
/// target's transfer. A texture never changes its format, usage, type or extent, so both
/// stay valid for as long as the texture lives.
///
/// The cached images count their references to a texture (<see cref="MetalCachedImage"/>
/// moves an old texture to the image that retires it), and the last one to let go drops the
/// entry and releases the view, before the texture itself is released, so a new texture
/// that reuses the handle never finds an old entry. A texture no cached image owns has no
/// entry: its transfers read the properties and make a view each time, as before, since
/// nothing would ever release one held for it.
/// </summary>
internal static class MetalTileViews
{
    private static readonly nint SelPixelFormat = MetalNative.Selector("pixelFormat");
    private static readonly nint SelUsage = MetalNative.Selector("usage");
    private static readonly nint SelTextureType = MetalNative.Selector("textureType");
    private static readonly nint SelMipmapLevelCount = MetalNative.Selector("mipmapLevelCount");
    private static readonly nint SelArrayLength = MetalNative.Selector("arrayLength");
    private static readonly nint SelRetain = MetalNative.Selector("retain");
    private static readonly nint SelRelease = MetalNative.Selector("release");

    private sealed class Entry
    {
        public int References;
        public bool HasInfo;
        public MetalTileTextureInfo Info;
        public nint View;
    }

    private static readonly Lock Gate = new();
    private static readonly Dictionary<nint, Entry> Entries = [];

    /// <summary>The view kept for a texture, or zero; for tests.</summary>
    internal static nint KeptView(nint texture)
    {
        lock (Gate)
        {
            return Entries.TryGetValue(texture, out var entry) ? entry.View : 0;
        }
    }

    public static MetalTileTextureInfo Read(nint texture) => new(
        (MtlPixelFormat)(uint)MetalV2Native.SendValue(texture, SelPixelFormat),
        (MtlTextureUsage)(uint)MetalV2Native.SendValue(texture, SelUsage),
        (MtlTextureType)(uint)MetalV2Native.SendValue(texture, SelTextureType),
        MetalV2Native.SendValue(texture, SelMipmapLevelCount),
        MetalV2Native.SendValue(texture, SelArrayLength));

    /// <summary>A cached image now refers to this texture.</summary>
    public static void Reference(nint texture)
    {
        if (texture == 0)
        {
            return;
        }

        lock (Gate)
        {
            if (!Entries.TryGetValue(texture, out var entry))
            {
                entry = new Entry();
                Entries[texture] = entry;
            }

            entry.References++;
        }
    }

    /// <summary>A cached image no longer refers to this texture; the last one drops its view.</summary>
    public static void Unreference(nint texture)
    {
        if (texture == 0)
        {
            return;
        }

        nint view;
        lock (Gate)
        {
            if (!Entries.TryGetValue(texture, out var entry) || --entry.References > 0)
            {
                return;
            }

            Entries.Remove(texture);
            view = entry.View;
        }

        if (view != 0)
        {
            // Any command buffer that encoded the view holds it until it completes.
            MetalNative.SendVoid(view, SelRelease);
        }
    }

    /// <summary>
    /// The texture's properties and, when one was made, its integer view, retained for the
    /// caller, who releases it. False when no cached image owns the texture; true with
    /// <paramref name="hasInfo"/> false when one does but nothing was asked of it yet.
    /// </summary>
    public static bool TryLookup(nint texture, out MetalTileTextureInfo info, out bool hasInfo, out nint view)
    {
        lock (Gate)
        {
            if (!Entries.TryGetValue(texture, out var entry))
            {
                info = default;
                hasInfo = false;
                view = 0;
                return false;
            }

            info = entry.Info;
            hasInfo = entry.HasInfo;
            view = entry.View;
            if (view != 0)
            {
                MetalNative.SendVoid(view, SelRetain);
            }

            return true;
        }
    }

    /// <summary>Keeps the properties of a texture a cached image owns.</summary>
    public static void Remember(nint texture, in MetalTileTextureInfo info)
    {
        lock (Gate)
        {
            if (Entries.TryGetValue(texture, out var entry))
            {
                entry.Info = info;
                entry.HasInfo = true;
            }
        }
    }

    /// <summary>
    /// Keeps a view the caller made of a texture a cached image owns, taking a reference of
    /// its own, or does nothing when the texture has no entry or already kept one.
    /// </summary>
    public static void RememberView(nint texture, nint view)
    {
        lock (Gate)
        {
            if (Entries.TryGetValue(texture, out var entry) && entry.View == 0)
            {
                MetalNative.SendVoid(view, SelRetain);
                entry.View = view;
            }
        }
    }
}
