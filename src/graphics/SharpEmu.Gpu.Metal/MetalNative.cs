// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu;

namespace SharpEmu.Gpu.Metal;

[StructLayout(LayoutKind.Sequential)]
internal struct CGSize
{
    public double Width;
    public double Height;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MtlClearColor
{
    public double Red;
    public double Green;
    public double Blue;
    public double Alpha;
}

/// <summary>MTLTextureSwizzle.</summary>
internal enum MtlTextureSwizzle : byte
{
    Zero = 0,
    One = 1,
    Red = 2,
    Green = 3,
    Blue = 4,
    Alpha = 5,
}

/// <summary>MTLTextureSwizzleChannels: one MTLTextureSwizzle byte per output
/// channel (Zero=0, One=1, Red=2, Green=3, Blue=4, Alpha=5).</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct MtlTextureSwizzleChannels
{
    public byte Red;
    public byte Green;
    public byte Blue;
    public byte Alpha;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MtlRegion
{
    public nuint X;
    public nuint Y;
    public nuint Z;
    public nuint Width;
    public nuint Height;
    public nuint Depth;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MtlSize
{
    public nuint Width;
    public nuint Height;
    public nuint Depth;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MtlOrigin
{
    public nuint X;
    public nuint Y;
    public nuint Z;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NsRange
{
    public nuint Location;
    public nuint Length;
}

internal struct MtlScissorRect
{
    public nuint X;
    public nuint Y;
    public nuint Width;
    public nuint Height;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MtlViewport
{
    public double OriginX;
    public double OriginY;
    public double Width;
    public double Height;
    public double ZNear;
    public double ZFar;
}

/// <summary>
/// Objective-C runtime access for the Metal presenter: QuartzCore and Metal
/// through objc_msgSend, with one LibraryImport overload per distinct native
/// signature. Dependency-free by design — this plus the OS frameworks is the entire
/// Metal path, which is what keeps it NativeAOT-clean.
/// </summary>
internal static partial class MetalNative
{
    private const string ObjCLibrary = "/usr/lib/libobjc.A.dylib";
    private const string MetalFramework = "/System/Library/Frameworks/Metal.framework/Metal";
    private const string QuartzCoreFramework = "/System/Library/Frameworks/QuartzCore.framework/QuartzCore";

    private static bool _frameworksLoaded;

    /// <summary>
    /// Makes QuartzCore classes visible to objc_getClass; Metal is
    /// pulled in by its own LibraryImport. Call once before any Class() lookup.
    /// </summary>
    public static void EnsureFrameworksLoaded()
    {
        if (_frameworksLoaded)
        {
            return;
        }

        ApplyShaderCacheSize();
        NativeLibrary.Load(QuartzCoreFramework);
        _frameworksLoaded = true;
    }

    /// <summary>
    /// The size SharpEmu gives macOS's per-app Metal shader cache, in bytes, unless
    /// MTL_SHADER_CACHE_SIZE is already set: 3 GiB.
    /// </summary>
    /// <remarks>
    /// The cache keeps the compiled form of every MSL library and evicts the oldest past its
    /// size, which is about 1 GiB by default for the whole bundle, every title and every
    /// build together. Filled to that, it dropped Demon's Souls' libraries within half an
    /// hour of other titles running: the next start compiled the 143 prewarmed computes
    /// cold (43 s instead of 6 s) and some 50 graphics libraries on the GPU worker at about
    /// a second each, drawing nothing for 45 s (runs 20261016-174800-ds-base-2,
    /// 20261016-180157-ds-fixA-1). Metal reads the variable when it first opens the cache,
    /// so it is set before the first device; Environment.SetEnvironmentVariable would not
    /// reach the native environment. The size must stay below
    /// <see cref="FirstShaderCacheSizeThatDisablesTheCache"/>.
    /// </remarks>
    internal const long DefaultShaderCacheBytes = 3L << 30;

    /// <summary>
    /// The smallest MTL_SHADER_CACHE_SIZE that turns the Metal shader cache off: 4 GiB.
    /// </summary>
    /// <remarks>
    /// Measured on macOS 27.2 with a standalone app compiling the same salted DS compute
    /// shader in two processes: 1, 2, 3 GiB and 4294967295 bytes hit in the second process
    /// (~1 ms); 4294967296 and 8 GiB compiled cold in both (~760 ms) and never wrote the cache
    /// file. A 4 GiB default made every SharpEmu start compile every shader cold: the warm
    /// Demon's Souls boot took 55 s of prewarm and drew its first frame at 154 s
    /// (run 20261017-031602-ds-main-2). Processes with 3 GiB and with the default size
    /// share one cache in both directions.
    /// </remarks>
    internal const long FirstShaderCacheSizeThatDisablesTheCache = 1L << 32;

    private const string ShaderCacheSizeVariable = "MTL_SHADER_CACHE_SIZE";

    private static void ApplyShaderCacheSize()
    {
        var configured = getenv(ShaderCacheSizeVariable);
        if (configured != 0)
        {
            var text = Marshal.PtrToStringUTF8(configured);
            if (DisablesShaderCache(text))
            {
                Console.Error.WriteLine(
                    $"[METAL][WARN] {ShaderCacheSizeVariable}={text} is 4 GiB or more, which turns macOS's " +
                    "Metal shader cache off: every start compiles every shader again. Use a smaller size.");
            }

            return;
        }

        var value = DefaultShaderCacheBytes.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (setenv(ShaderCacheSizeVariable, value, 0) != 0)
        {
            Console.Error.WriteLine(
                $"[METAL][WARN] {ShaderCacheSizeVariable} could not be set (errno {Marshal.GetLastPInvokeError()}); " +
                "the Metal shader cache keeps its default size.");
        }
    }

    /// <summary>Whether a MTL_SHADER_CACHE_SIZE value turns the Metal shader cache off.</summary>
    internal static bool DisablesShaderCache(string? value) =>
        ulong.TryParse(value, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out var bytes) &&
        bytes >= FirstShaderCacheSizeThatDisablesTheCache;

    [LibraryImport("/usr/lib/libSystem.dylib", StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint getenv(string name);

    [LibraryImport("/usr/lib/libSystem.dylib", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial int setenv(string name, string value, int overwrite);

    [LibraryImport(MetalFramework)]
    public static partial nint MTLCreateSystemDefaultDevice();

    [LibraryImport(ObjCLibrary, StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint objc_getClass(string name);

    [LibraryImport(ObjCLibrary, StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint sel_registerName(string name);

    [LibraryImport(ObjCLibrary)]
    public static partial nint objc_autoreleasePoolPush();

    [LibraryImport(ObjCLibrary)]
    public static partial void objc_autoreleasePoolPop(nint pool);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoidFloat(nint receiver, nint selector, float value);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint Send(nint receiver, nint selector);

    /// <summary>objc_msgSend for -gpuResourceID. MTLResourceID is a one-field
    /// 8-byte struct, returned in a register on the x86-64 ABI, so it maps to a
    /// ulong return — the value written into a Tier 2 argument buffer slot.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial ulong SendGpuResourceId(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint Send(nint receiver, nint selector, nint argument);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint Send(nint receiver, nint selector, nint argument, ref nint error);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint Send(nint receiver, nint selector, nint argument0, nint argument1, ref nint error);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendAtIndex(nint receiver, nint selector, nuint index);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static partial bool SendBool(nint receiver, nint selector);

    /// <summary>One-argument BOOL sends, e.g. respondsToSelector:.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static partial bool SendBool(nint receiver, nint selector, nint argument);


    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial double SendDouble(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoid(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoid(nint receiver, nint selector, nint argument);

    /// <summary>Two-object-argument void sends, e.g. setObject:forKey:.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoid(nint receiver, nint selector, nint argument0, nint argument1);

    /// <summary>setSwizzle: on MTLTextureDescriptor. Four one-byte
    /// MTLTextureSwizzle values, passed packed like the framework expects.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidSwizzle(
        nint receiver,
        nint selector,
        MtlTextureSwizzleChannels channels);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidBool(nint receiver, nint selector, [MarshalAs(UnmanagedType.I1)] bool argument);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidNUInt(nint receiver, nint selector, nuint argument);

    // MTLLanguageVersion3_2: (major << 16) | minor. Texture atomics arrived in
    // MSL 3.1 and MTLCompileOptions still defaults to an older version, so every
    // site that compiles MSL asks for this one rather than taking the default.
    public const nuint MslLanguageVersion = (3 << 16) | 2;

    /// <summary>Pins the MSL version a library is compiled against.</summary>
    public static void SetMslLanguageVersion(nint options) =>
        SetMslLanguageVersion(options, MslLanguageVersion);

    /// <summary>Pins a library to a specific MSL version, for source that needs a newer one.</summary>
    public static void SetMslLanguageVersion(nint options, nuint version) =>
        SendVoidNUInt(options, Selector("setLanguageVersion:"), version);

    /// <summary>
    /// The compile options every translated-guest MSL library is built with: fast
    /// math off, because the translated GCN float semantics do not survive it, an
    /// explicit language version so MSL 3.1+ features are available, and invariance
    /// preserved, so a [[position, invariant]] output is computed the same way in every
    /// program (a depth-EQUAL pass against a prepass depends on it).
    /// </summary>
    public static void ApplyMslCompileOptions(nint options)
    {
        SendVoidBool(options, Selector("setFastMathEnabled:"), false);
        SendVoidBool(options, Selector("setPreserveInvariance:"), true);
        SetMslLanguageVersion(options);
    }

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidDouble(nint receiver, nint selector, double argument);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidSize(nint receiver, nint selector, CGSize size);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidClearColor(nint receiver, nint selector, MtlClearColor color);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidBlendColor(
        nint receiver,
        nint selector,
        float red,
        float green,
        float blue,
        float alpha);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidViewport(nint receiver, nint selector, MtlViewport viewport);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetAtIndex(nint receiver, nint selector, nint value, nuint index);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidCopyTexture(nint receiver, nint selector, nint source, nint destination);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyTextureSlice(nint receiver, nint selector, nint source,
        nuint sourceSlice, nuint sourceLevel, MtlOrigin sourceOrigin, MtlSize sourceSize,
        nint destination, nuint destinationSlice, nuint destinationLevel, MtlOrigin destinationOrigin);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendBuffer(nint receiver, nint selector, nint bytes, nuint length, nuint options);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendNewBuffer(nint receiver, nint selector, nuint length, nuint options);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyTextureToBuffer(
        nint receiver,
        nint selector,
        nint sourceTexture,
        nuint sourceSlice,
        nuint sourceLevel,
        MtlOrigin sourceOrigin,
        MtlSize sourceSize,
        nint destinationBuffer,
        nuint destinationOffset,
        nuint destinationBytesPerRow,
        nuint destinationBytesPerImage);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyBufferToTexture(
        nint receiver,
        nint selector,
        nint sourceBuffer,
        nuint sourceOffset,
        nuint sourceBytesPerRow,
        nuint sourceBytesPerImage,
        MtlSize sourceSize,
        nint destinationTexture,
        nuint destinationSlice,
        nuint destinationLevel,
        MtlOrigin destinationOrigin);

    /// <summary>
    /// copyFromBuffer:...:toTexture:...:options: — the depth/stencil-aware upload. A
    /// combined depth-stencil texture has one plane per aspect, so the plane is chosen
    /// by MTLBlitOption rather than by the pixel format.
    /// </summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyBufferToTextureOptions(
        nint receiver,
        nint selector,
        nint sourceBuffer,
        nuint sourceOffset,
        nuint sourceBytesPerRow,
        nuint sourceBytesPerImage,
        MtlSize sourceSize,
        nint destinationTexture,
        nuint destinationSlice,
        nuint destinationLevel,
        MtlOrigin destinationOrigin,
        nuint options);

    /// <summary>copyFromTexture:...:toBuffer:...:options: — the readback twin of
    /// <see cref="SendCopyBufferToTextureOptions"/>.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyTextureToBufferOptions(
        nint receiver,
        nint selector,
        nint sourceTexture,
        nuint sourceSlice,
        nuint sourceLevel,
        MtlOrigin sourceOrigin,
        MtlSize sourceSize,
        nint destinationBuffer,
        nuint destinationOffset,
        nuint destinationBytesPerRow,
        nuint destinationBytesPerImage,
        nuint options);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDispatch(
        nint receiver,
        nint selector,
        MtlSize threadgroups,
        MtlSize threadsPerThreadgroup);

    /// <summary>setBytes:length:atIndex: — per-dispatch constants without a ring
    /// buffer; Metal copies them into the encoder's own argument storage.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetBytes(nint receiver, nint selector, nint bytes, nuint length, nuint index);

    /// <summary>setConstantValue:type:atIndex: on MTLFunctionConstantValues.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetConstantValue(nint receiver, nint selector, nint value, nuint type, nuint index);

    /// <summary>objc_msgSend for the NSUInteger-returning property getters.</summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nuint SendUInteger(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendSetBuffer(nint receiver, nint selector, nint buffer, nuint offset, nuint index);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendFillBuffer(nint receiver, nint selector, nint buffer, NsRange range, byte value);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendCopyBuffer(nint receiver, nint selector, nint source, nuint sourceOffset, nint destination, nuint destinationOffset, nuint size);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendUseResource(nint receiver, nint selector, nint resource, nuint usage, nuint stages);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendUseResourceUsage(nint receiver, nint selector, nint resource, nuint usage);

    // useResources:count:usage: — a C array of resources in one message.
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static unsafe partial void SendUseResources(nint receiver, nint selector, nint* resources, nuint count, nuint usage);

    // useResources:count:usage:stages:
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static unsafe partial void SendUseResourcesStages(
        nint receiver, nint selector, nint* resources, nuint count, nuint usage, nuint stages);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial ulong SendGpuAddress(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendVoidScissor(nint receiver, nint selector, MtlScissorRect rect);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawPrimitivesInstanced(
        nint receiver,
        nint selector,
        nuint primitiveType,
        nuint vertexStart,
        nuint vertexCount,
        nuint instanceCount);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawIndexedPrimitives(
        nint receiver,
        nint selector,
        nuint primitiveType,
        nuint indexCount,
        nuint indexType,
        nint indexBuffer,
        nuint indexBufferOffset,
        nuint instanceCount);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial nint SendTextureDescriptor(
        nint receiver,
        nint selector,
        nuint pixelFormat,
        nuint width,
        nuint height,
        [MarshalAs(UnmanagedType.I1)] bool mipmapped);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendReplaceRegion(
        nint receiver,
        nint selector,
        MtlRegion region,
        nuint mipmapLevel,
        nint bytes,
        nuint bytesPerRow);

    /// <summary>
    /// The slice- and volume-aware upload. One array layer takes a slice with
    /// Depth 1 and no image stride; a volume takes slice 0, the region's full
    /// Depth and the byte stride between depth slices.
    /// </summary>
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendReplaceRegionSlice(
        nint receiver,
        nint selector,
        MtlRegion region,
        nuint mipmapLevel,
        nuint slice,
        nint bytes,
        nuint bytesPerRow,
        nuint bytesPerImage);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    public static partial void SendDrawPrimitives(
        nint receiver,
        nint selector,
        nuint primitiveType,
        nuint vertexStart,
        nuint vertexCount);

    public static nint Class(string name) => objc_getClass(name);

    // Objective-C selector identities are process-wide and permanent. Reuse the
    // native identity instead of marshalling/registering constant names on every
    // encoder call. Concurrent first use may register twice; both yield the same
    // identity, and no native object retain/release belongs to this cache.
    private static readonly ConcurrentDictionary<string, nint> Selectors = new(StringComparer.Ordinal);
    public static nint Selector(string name) => Selectors.TryGetValue(name, out var selector)
        ? selector
        : Selectors.GetOrAdd(name, static key => sel_registerName(key));

    /// <summary>Autoreleased NSString — only valid inside an autorelease pool
    /// unless the caller retains it.</summary>
    public static nint NsString(string value)
    {
        var utf8 = Marshal.StringToCoTaskMemUTF8(value);
        try
        {
            return Send(Class("NSString"), Selector("stringWithUTF8String:"), utf8);
        }
        finally
        {
            Marshal.FreeCoTaskMem(utf8);
        }
    }

    /// <summary>Reads an NSString's UTF-8 contents, or null if the handle is nil.</summary>
    public static string? ReadNsString(nint nsString)
    {
        if (nsString == 0)
        {
            return null;
        }

        var utf8 = Send(nsString, Selector("UTF8String"));
        return utf8 == 0 ? null : Marshal.PtrToStringUTF8(utf8);
    }

    public static string DescribeError(nint error)
    {
        if (error == 0)
        {
            return "unknown error";
        }

        var description = Send(error, Selector("localizedDescription"));
        var utf8 = Send(description, Selector("UTF8String"));
        return Marshal.PtrToStringUTF8(utf8) ?? "unknown error";
    }
}
