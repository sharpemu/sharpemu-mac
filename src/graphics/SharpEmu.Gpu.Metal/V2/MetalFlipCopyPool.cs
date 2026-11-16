// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// SHARPEMU_FLIP_AHEAD: the presenter-owned textures a GPU flip's frame is copied into, in the
/// stream at the flip (MetalRenderHost.FlipCopy.cs). Without the copy the presenter samples the
/// display image's own texture until the next frame replaces it, so a guest told of flip N before
/// N is shown could render the next frame into the buffer still on screen.
/// <para>A texture goes back to the pool only once the presenter can no longer read it: it is not
/// the presented frame, not queued, not being encoded by a presentation, and the last
/// presentation command buffer that sampled it has completed.</para>
/// </summary>
internal static partial class MetalFlipCopyPool
{
    // The frame on screen, the frames the presenter queues (MaxPendingFrames = 4), the one the
    // stream is writing, and one whose last presentation may still be on the GPU.
    internal const int Capacity = 7;

    internal enum Refusal
    {
        PoolExhausted,
        TextureType,
        Multisampled,
        Layout,
        TextureCreate,
        Encoder,
        Count,
    }

    private static readonly string[] RefusalNames =
        ["pool_exhausted", "texture_type", "multisampled", "layout", "texture_create", "encoder"];

    private sealed class Entry
    {
        public nint Texture;
        public nint Presented;
        public ulong PixelFormat;
        public uint Width;
        public uint Height;
        public MtlTextureSwizzleChannels Swizzle;
        public nint LastSampledBy;
    }

    private static readonly object _gate = new();
    private static readonly List<Entry> _entries = new();
    private static readonly long[] _refusals = new long[(int)Refusal.Count];
    private static long _copies;
    private static int _highWater;
    private static ulong _copyBytes;

    private static readonly nint SelSwizzle = MetalNative.Selector("swizzle");
    private static readonly nint SelStatus = MetalNative.Selector("status");
    private static readonly nint SelRetain = MetalNative.Selector("retain");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelSetUsage = MetalNative.Selector("setUsage:");
    private static readonly nint SelSetStorageMode = MetalNative.Selector("setStorageMode:");
    private static readonly nint SelNewTexture = MetalNative.Selector("newTextureWithDescriptor:");
    private static readonly nint SelNewSwizzledView =
        MetalNative.Selector("newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:");

    // MTLCommandBufferStatusCompleted; Error (5) is past it too.
    private const nuint CommandBufferCompleted = 4;

    [LibraryImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static partial MtlTextureSwizzleChannels SendSwizzle(nint receiver, nint selector);

    /// <summary>The swizzle the presenter samples a texture through.</summary>
    internal static MtlTextureSwizzleChannels SwizzleOf(nint texture) => SendSwizzle(texture, SelSwizzle);

    /// <summary>
    /// A texture of this shape the presenter cannot read any more, or a new one. destination is
    /// what the copy writes, presented what the presenter is handed (the same texture, or a view
    /// of it with the source's swizzle).
    /// </summary>
    internal static bool TryAcquire(nint device, ulong pixelFormat, uint width, uint height,
        MtlTextureSwizzleChannels swizzle, out nint destination, out nint presented, out Refusal refusal)
    {
        destination = 0;
        presented = 0;
        refusal = Refusal.Count;
        lock (_gate)
        {
            Entry? free = null;
            Entry? reclaim = null;
            var held = 0;
            foreach (var entry in _entries)
            {
                if (IsReadableLocked(entry))
                {
                    held++;
                    continue;
                }

                var matches = entry.PixelFormat == pixelFormat && entry.Width == width && entry.Height == height &&
                    SameSwizzle(entry.Swizzle, swizzle);
                if (matches)
                {
                    free ??= entry;
                }
                else
                {
                    reclaim ??= entry;
                }
            }

            if (free is null)
            {
                if (_entries.Count >= Capacity)
                {
                    if (reclaim is null)
                    {
                        refusal = Refusal.PoolExhausted;
                        return false;
                    }

                    ReleaseLocked(reclaim);
                    _entries.Remove(reclaim);
                }

                free = Create(device, pixelFormat, width, height, swizzle);
                if (free is null)
                {
                    refusal = Refusal.TextureCreate;
                    return false;
                }

                _entries.Add(free);
            }

            _highWater = Math.Max(_highWater, held + 1);
            destination = free.Texture;
            presented = free.Presented;
            return true;
        }
    }

    internal static void NoteCopy(ulong bytes)
    {
        Interlocked.Increment(ref _copies);
        Interlocked.Exchange(ref _copyBytes, bytes);
    }

    internal static void NoteRefusal(Refusal refusal) => Interlocked.Increment(ref _refusals[(int)refusal]);

    internal static string RefusalName(Refusal refusal) => RefusalNames[(int)refusal];

    /// <summary>A presentation command buffer, now committed, sampled this texture.</summary>
    internal static void NoteSampled(nint texture, nint commandBuffer)
    {
        if (texture == 0 || commandBuffer == 0)
        {
            return;
        }

        lock (_gate)
        {
            foreach (var entry in _entries)
            {
                if (entry.Presented != texture)
                {
                    continue;
                }

                MetalNative.SendVoid(commandBuffer, SelRetain);
                if (entry.LastSampledBy != 0)
                {
                    MetalNative.SendVoid(entry.LastSampledBy, SelRelease);
                }

                entry.LastSampledBy = commandBuffer;
                return;
            }
        }
    }

    /// <summary>Copies, refusals by reason, the most textures the presenter could read at once plus the one written.</summary>
    internal static string TakeReport()
    {
        var copies = Interlocked.Exchange(ref _copies, 0);
        var refusals = new System.Text.StringBuilder();
        for (var index = 0; index < (int)Refusal.Count; index++)
        {
            var count = Interlocked.Exchange(ref _refusals[index], 0);
            if (count != 0)
            {
                refusals.Append(refusals.Length == 0 ? "" : ",").Append(RefusalNames[index]).Append(':').Append(count);
            }
        }

        int pool;
        int high;
        lock (_gate)
        {
            pool = _entries.Count;
            high = _highWater;
        }

        return $"flip_copies={copies} flip_copy_refused={(refusals.Length == 0 ? "0" : refusals.ToString())} " +
            $"flip_copy_pool={pool} flip_copy_pool_high={high} flip_copy_mb={Interlocked.Read(ref _copyBytes) / (1024.0 * 1024.0):F1}";
    }

    // Caller holds the gate.
    private static bool IsReadableLocked(Entry entry)
    {
        if (MetalV2Presenter.IsFlipCopyHeld(entry.Presented))
        {
            return true;
        }

        if (entry.LastSampledBy == 0)
        {
            return false;
        }

        if (MetalV2Native.SendValue(entry.LastSampledBy, SelStatus) < CommandBufferCompleted)
        {
            return true;
        }

        MetalNative.SendVoid(entry.LastSampledBy, SelRelease);
        entry.LastSampledBy = 0;
        return false;
    }

    private static bool SameSwizzle(MtlTextureSwizzleChannels left, MtlTextureSwizzleChannels right) =>
        left.Red == right.Red && left.Green == right.Green && left.Blue == right.Blue && left.Alpha == right.Alpha;

    // MTLTextureSwizzleRed..Alpha are 2..5.
    private static bool IsIdentity(MtlTextureSwizzleChannels swizzle) =>
        swizzle.Red == 2 && swizzle.Green == 3 && swizzle.Blue == 4 && swizzle.Alpha == 5;

    private static Entry? Create(nint device, ulong pixelFormat, uint width, uint height, MtlTextureSwizzleChannels swizzle)
    {
        var descriptor = MetalNative.SendTextureDescriptor(
            MetalNative.Class("MTLTextureDescriptor"),
            MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            (nuint)pixelFormat,
            width,
            height,
            mipmapped: false);
        if (descriptor == 0)
        {
            return null;
        }

        var identity = IsIdentity(swizzle);
        MetalV2Native.SendVoidValue(descriptor, SelSetUsage,
            (nuint)(Mtl.TextureUsageShaderRead | (identity ? 0 : Mtl.TextureUsagePixelFormatView)));
        MetalV2Native.SendVoidValue(descriptor, SelSetStorageMode, Mtl.StorageModePrivate);
        var texture = MetalNative.Send(device, SelNewTexture, descriptor);
        if (texture == 0)
        {
            return null;
        }

        var presented = texture;
        if (!identity)
        {
            presented = MetalV2Native.SendNewSwizzledTextureView(
                texture, SelNewSwizzledView, (nuint)pixelFormat, (nuint)Mtl.TextureType2D,
                new NsRange { Location = 0, Length = 1 }, new NsRange { Location = 0, Length = 1 }, swizzle);
            if (presented == 0)
            {
                MetalNative.SendVoid(texture, SelRelease);
                return null;
            }
        }

        return new Entry
        {
            Texture = texture,
            Presented = presented,
            PixelFormat = pixelFormat,
            Width = width,
            Height = height,
            Swizzle = swizzle,
        };
    }

    // Caller holds the gate; the entry is not readable.
    private static void ReleaseLocked(Entry entry)
    {
        if (entry.Presented != entry.Texture)
        {
            MetalNative.SendVoid(entry.Presented, SelRelease);
        }

        MetalNative.SendVoid(entry.Texture, SelRelease);
        if (entry.LastSampledBy != 0)
        {
            MetalNative.SendVoid(entry.LastSampledBy, SelRelease);
        }
    }
}
