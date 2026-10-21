// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Minimal Metal.framework access via objc_msgSend — just enough to compile MSL source
/// with the OS runtime compiler and dispatch a single-thread compute kernel. Kept
/// dependency-free on purpose; object lifetimes lean on process teardown, which is fine
/// for a test host.
/// </summary>
internal static partial class MetalNative
{
    private const string ObjCLibrary = "/usr/lib/libobjc.A.dylib";
    private const string MetalFramework = "/System/Library/Frameworks/Metal.framework/Metal";

    // MTLLanguageVersion3_2: (major << 16) | minor. Texture atomics arrived in
    // MSL 3.1 and the default version is older, so the backend asks for this one
    // explicitly and the harness has to match it.
    private const nuint MslLanguageVersion = (3 << 16) | 2;

    [StructLayout(LayoutKind.Sequential)]
    private struct MtlSize
    {
        public nuint Width;
        public nuint Height;
        public nuint Depth;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MtlRegion
    {
        public nuint X, Y, Z, Width, Height, Depth;
    }

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendInteger(nint receiver, nint selector, nuint value);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendReplace(nint receiver, nint selector, MtlRegion region,
        nuint level, nuint slice, nint bytes, nuint rowBytes, nuint imageBytes);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendReadTexture(nint receiver, nint selector, nint bytes,
        nuint rowBytes, nuint imageBytes, MtlRegion region, nuint level, nuint slice);

    [LibraryImport(MetalFramework)]
    private static partial nint MTLCreateSystemDefaultDevice();

    [LibraryImport(ObjCLibrary, StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint objc_getClass(string name);

    [LibraryImport(ObjCLibrary, StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint sel_registerName(string name);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint Send(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint Send(nint receiver, nint selector, nint argument);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint Send(nint receiver, nint selector, nint argument0, nint argument1, ref nint error);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint Send(nint receiver, nint selector, nint argument, ref nint error);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint SendBuffer(nint receiver, nint selector, nint bytes, nuint length, nuint options);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendVoid(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendVoid(nint receiver, nint selector, nint argument);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendVoidBool(nint receiver, nint selector, [MarshalAs(UnmanagedType.I1)] bool argument);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendVoidUInt(nint receiver, nint selector, nuint argument);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendSetBuffer(nint receiver, nint selector, nint buffer, nuint offset, nuint index);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDispatch(nint receiver, nint selector, MtlSize threadgroups, MtlSize threadsPerThreadgroup);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendUseResource(nint receiver, nint selector, nint resource, nuint usage);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial ulong SendULong(nint receiver, nint selector);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDraw(nint receiver, nint selector, nuint primitive, nuint start, nuint count);

    // MTLResourceUsageRead | MTLResourceUsageWrite for buffers reached through an argument buffer.
    private const nuint ReadWriteUsage = 3;

    private static readonly Lazy<nint> Device = new(() =>
        OperatingSystem.IsMacOS() ? MTLCreateSystemDefaultDevice() : 0);

    // Whether device tests may execute. Hosted CI macOS runners only expose "Apple Paravirtual
    // device": it compiles MSL, but lacks parts of the Metal API the execution tests drive (its
    // render encoder has no setMeshBuffer, and calling it kills the test host). That device does
    // not count as available unless the device gate demands one, in which case it is used and
    // any gap fails loudly. Compile-only paths keep using Device directly.
    public static bool IsAvailable { get; } = Device.Value != 0 && (
        Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1" ||
        !(Marshal.PtrToStringUTF8(Send(Send(Device.Value, Selector("name")), Selector("UTF8String")))
            ?.Contains("Paravirtual", StringComparison.OrdinalIgnoreCase) ?? false));

    private static nint Selector(string name) => sel_registerName(name);

    private static nint NsString(string value)
    {
        var utf8 = Marshal.StringToCoTaskMemUTF8(value);
        try
        {
            return Send(objc_getClass("NSString"), Selector("stringWithUTF8String:"), utf8);
        }
        finally
        {
            Marshal.FreeCoTaskMem(utf8);
        }
    }

    private static string DescribeError(nint error)
    {
        if (error == 0)
        {
            return "unknown error";
        }

        var description = Send(error, Selector("localizedDescription"));
        var utf8 = Send(description, Selector("UTF8String"));
        return Marshal.PtrToStringUTF8(utf8) ?? "unknown error";
    }

    /// <summary>Compiles MSL source with the OS runtime compiler.</summary>
    public static bool TryCompileLibrary(string source, out nint library, out string error)
    {
        library = 0;
        error = string.Empty;

        // Metal defaults to fast-math; GCN float semantics do not survive it, so the
        // harness compiles the way a real Metal backend must: fast-math off, and
        // the same language version the backend asks for so a feature that needs
        // MSL 3.1 or later (texture atomics) compiles here too.
        var options = Send(Send(objc_getClass("MTLCompileOptions"), Selector("alloc")), Selector("init"));
        SendVoidBool(options, Selector("setFastMathEnabled:"), false);
        SendVoidUInt(options, Selector("setLanguageVersion:"), MslLanguageVersion);

        nint nsError = 0;
        library = Send(
            Device.Value,
            Selector("newLibraryWithSource:options:error:"),
            NsString(source),
            options,
            ref nsError);
        if (library == 0)
        {
            error = DescribeError(nsError);
            return false;
        }

        return true;
    }

    /// <summary>Runs a kernel compiled through a compile request: the argument buffer
    /// is filled directly from the layout with the data buffer's address and length
    /// (every buffer field points at the one data buffer), push data at slot 1.</summary>
    public static bool TryExecuteWithArgumentBuffer(
        nint library,
        string entryPoint,
        byte[] bufferContents,
        byte[] pushData,
        Gen5MslArgumentLayout layout,
        uint threadsPerThreadgroup,
        out byte[] result,
        out string error,
        nint texture = 0,
        nint sampler = 0)
    {
        return TryExecuteWithArgumentBuffer(
            library, entryPoint, bufferContents, pushData, layout, threadsPerThreadgroup,
            out result, out _, out error, texture, sampler);
    }

    /// <summary>
    /// As above, and also reports the step-report word the dispatcher increments when it
    /// runs out of steps — the same word the v2 host binds and reads back per shader hash.
    /// </summary>
    public static bool TryExecuteWithArgumentBuffer(
        nint library,
        string entryPoint,
        byte[] bufferContents,
        byte[] pushData,
        Gen5MslArgumentLayout layout,
        uint threadsPerThreadgroup,
        out byte[] result,
        out uint stepCapHits,
        out string error,
        nint texture = 0,
        nint sampler = 0,
        uint threadgroupCount = 1,
        IReadOnlyList<DeviceTable>? tables = null,
        IReadOnlyList<DeviceTable>? bindlessTables = null,
        Func<DescriptorBindingKind, nint>? textureOfClass = null)
    {
        result = [];
        stepCapHits = 0;
        var buffer = SharedBuffer(bufferContents);
        if (buffer == 0)
        {
            error = "failed to create the data buffer";
            return false;
        }

        var address = SendULong(buffer, Selector("gpuAddress"));
        var stepReport = SharedBuffer(new byte[sizeof(uint)]);
        if (stepReport == 0)
        {
            error = "failed to create the step-report buffer";
            return false;
        }

        var stepReportAddress = SendULong(stepReport, Selector("gpuAddress"));
        // The zeroed pair-lock table the v2 host binds for every program.
        var pairLocks = SharedBuffer(new byte[Gen5MslArgumentLayout.PairLockTableWords * sizeof(uint)]);
        if (pairLocks == 0)
        {
            error = "failed to create the pair-lock table";
            return false;
        }

        var pairLocksAddress = SendULong(pairLocks, Selector("gpuAddress"));
        var arguments = new byte[Math.Max(layout.ByteSize, 8)];
        foreach (var field in layout.Fields)
        {
            for (var element = 0; element < field.Count; element++)
            {
                switch (field.FieldKind)
                {
                    case MslArgumentFieldKind.BufferPointers:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset + (element * 8)), address);
                        break;
                    case MslArgumentFieldKind.Textures:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset + (element * 8)),
                            SendULong(textureOfClass?.Invoke(field.Kind) ?? texture, Selector("gpuResourceID")));
                        break;
                    case MslArgumentFieldKind.BindlessImageTables:
                        // One device handle table per slot, in slot order.
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset + (element * 8)),
                            SendULong(bindlessTables![element].Buffer, Selector("gpuAddress")));
                        break;
                    case MslArgumentFieldKind.Samplers:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset + (element * 8)),
                            SendULong(sampler, Selector("gpuResourceID")));
                        break;
                    case MslArgumentFieldKind.BufferByteCounts:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset + (element * 4)), (uint)bufferContents.Length);
                        break;
                    case MslArgumentFieldKind.StepReport:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset), stepReportAddress);
                        break;
                    case MslArgumentFieldKind.PairLocks:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset), pairLocksAddress);
                        break;
                    case MslArgumentFieldKind.Pointer:
                        // A read-only table the host fills before the encode: the user
                        // data words, the flattened resource table, the address ranges.
                        // Zero-filled here, which is what an untouched one holds, unless
                        // the caller supplies the binding's table.
                        BitConverter.TryWriteBytes(
                            arguments.AsSpan((int)field.ByteOffset),
                            FindTable(tables, field.Kind) is { } table ? SendULong(table.Buffer, Selector("gpuAddress")) : SharedTableAddress());
                        break;
                    case MslArgumentFieldKind.Word:
                        BitConverter.TryWriteBytes(
                            arguments.AsSpan((int)field.ByteOffset),
                            FindTable(tables, field.Kind) is { } counted ? counted.Count : SharedTableDwords);
                        break;
                }
            }
        }

        var argumentBuffer = SharedBuffer(arguments);
        var pushBuffer = SharedBuffer(pushData.Length == 0 ? new byte[4] : pushData);
        if (argumentBuffer == 0 || pushBuffer == 0)
        {
            error = "failed to create the argument or push buffer";
            return false;
        }

        if (!RunKernel(
                library,
                entryPoint,
                [(argumentBuffer, (nuint)Gen5MslArgumentLayout.ResourcesBufferIndex), (pushBuffer, (nuint)Gen5MslArgumentLayout.PushDataBufferIndex)],
                [
                    buffer, stepReport, pairLocks, SharedTable.Value,
                    .. texture == 0 ? Array.Empty<nint>() : [texture],
                    .. (tables ?? []).SelectMany(static table => (nint[])[table.Buffer, .. table.Residents]),
                    .. (bindlessTables ?? []).SelectMany(static table => (nint[])[table.Buffer, .. table.Residents]),
                    .. textureOfClass is null ? Array.Empty<nint>() : layout.Fields
                        .Where(static field => field.FieldKind == MslArgumentFieldKind.Textures)
                        .Select(field => textureOfClass(field.Kind)).Distinct(),
                ],
                threadsPerThreadgroup,
                buffer,
                bufferContents.Length,
                out result,
                out error,
                threadgroupCount))
        {
            return false;
        }

        var stepContents = Send(stepReport, Selector("contents"));
        if (stepContents == 0)
        {
            error = "step-report contents unavailable after execution";
            return false;
        }

        stepCapHits = unchecked((uint)Marshal.ReadInt32(stepContents));
        return true;
    }

    /// <summary>Float RGBA fixture with one distinct value per texel/mip/layer.
    /// Texture type is the actual MSL backing (1D guest images use height-one2D).</summary>
    public static nint CreateFloatTexture(uint type, uint width, uint height, uint layers,
        uint depth, IReadOnlyList<byte[]> mipBytes)
    {
        var descriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(descriptor, Selector("setPixelFormat:"), 125); // RGBA32Float
        SendInteger(descriptor, Selector("setTextureType:"), type);
        SendInteger(descriptor, Selector("setWidth:"), width);
        SendInteger(descriptor, Selector("setHeight:"), height);
        SendInteger(descriptor, Selector("setDepth:"), depth);
        SendInteger(descriptor, Selector("setArrayLength:"), layers);
        SendInteger(descriptor, Selector("setMipmapLevelCount:"), (nuint)mipBytes.Count);
        SendInteger(descriptor, Selector("setStorageMode:"), 0);
        SendInteger(descriptor, Selector("setUsage:"), 3);
        var texture = Send(Device.Value, Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0) throw new InvalidOperationException("Failed to create fixture texture.");
        for (uint level = 0; level < mipBytes.Count; level++)
        {
            var w = Math.Max(width >> (int)level, 1u);
            var h = Math.Max(height >> (int)level, 1u);
            var d = Math.Max(depth >> (int)level, 1u);
            var imageBytes = w * h * 16;
            unsafe
            {
                fixed (byte* bytes = mipBytes[(int)level])
                {
                    for (uint layer = 0; layer < layers; layer++)
                    {
                        SendReplace(texture, Selector("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
                            new MtlRegion { Width = w, Height = h, Depth = d }, level, layer,
                            (nint)(bytes + layer * imageBytes), w * 16, type == 7 ? imageBytes : 0);
                    }
                }
            }
        }
        return texture;
    }

    /// <summary>
    /// A single-level R32Uint 2D texture a shader may run atomics on. Metal
    /// allows a texture atomic only on an R32Uint/R32Sint (or R64Uint) texture
    /// created with MTLTextureUsageShaderAtomic, which is usage bit 5.
    /// </summary>
    public static nint CreateAtomicUintTexture(
        uint width, uint height, IReadOnlyList<uint> texels, nuint usage = 1 | 2 | 32)
    {
        const nuint R32Uint = 53;
        var descriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(descriptor, Selector("setPixelFormat:"), R32Uint);
        SendInteger(descriptor, Selector("setTextureType:"), 2); // MTLTextureType2D
        SendInteger(descriptor, Selector("setWidth:"), width);
        SendInteger(descriptor, Selector("setHeight:"), height);
        SendInteger(descriptor, Selector("setDepth:"), 1);
        SendInteger(descriptor, Selector("setArrayLength:"), 1);
        SendInteger(descriptor, Selector("setMipmapLevelCount:"), 1);
        SendInteger(descriptor, Selector("setStorageMode:"), 0);
        SendInteger(descriptor, Selector("setUsage:"), usage);
        var texture = Send(Device.Value, Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0)
        {
            throw new InvalidOperationException(
                "Failed to create an atomic R32Uint fixture texture; MTLTextureUsageShaderAtomic needs macOS 14 or later.");
        }

        var bytes = new byte[width * height * sizeof(uint)];
        for (var index = 0; index < texels.Count; index++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint)), texels[index]);
        }

        unsafe
        {
            fixed (byte* raw = bytes)
            {
                SendReplace(texture, Selector("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
                    new MtlRegion { Width = width, Height = height, Depth = 1 }, 0, 0,
                    (nint)raw, width * sizeof(uint), 0);
            }
        }

        return texture;
    }

    /// <summary>
    /// A single-level RG32Uint 2D texture for 64-bit texture atomics, which Metal runs on
    /// RG32Uint textures created with MTLTextureUsageShaderAtomic. Each texel is one ulong.
    /// </summary>
    public static nint CreateAtomicUlongTexture(
        uint width, uint height, IReadOnlyList<ulong> texels, nuint usage = 1 | 2 | 32)
    {
        const nuint RG32Uint = 103;
        var descriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(descriptor, Selector("setPixelFormat:"), RG32Uint);
        SendInteger(descriptor, Selector("setTextureType:"), 2); // MTLTextureType2D
        SendInteger(descriptor, Selector("setWidth:"), width);
        SendInteger(descriptor, Selector("setHeight:"), height);
        SendInteger(descriptor, Selector("setDepth:"), 1);
        SendInteger(descriptor, Selector("setArrayLength:"), 1);
        SendInteger(descriptor, Selector("setMipmapLevelCount:"), 1);
        SendInteger(descriptor, Selector("setStorageMode:"), 0);
        SendInteger(descriptor, Selector("setUsage:"), usage);
        var texture = Send(Device.Value, Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0)
        {
            throw new InvalidOperationException("Failed to create an atomic RG32Uint fixture texture.");
        }

        var bytes = new byte[width * height * sizeof(ulong)];
        for (var index = 0; index < texels.Count; index++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(index * sizeof(ulong)), texels[index]);
        }

        unsafe
        {
            fixed (byte* raw = bytes)
            {
                SendReplace(texture, Selector("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
                    new MtlRegion { Width = width, Height = height, Depth = 1 }, 0, 0,
                    (nint)raw, width * sizeof(ulong), 0);
            }
        }

        return texture;
    }

    public static ulong[] ReadAtomicUlongTexture(nint texture, uint width, uint height)
    {
        var bytes = new byte[width * height * sizeof(ulong)];
        unsafe
        {
            fixed (byte* raw = bytes)
            {
                SendReadTexture(texture, Selector("getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:"),
                    (nint)raw, width * sizeof(ulong), 0,
                    new MtlRegion { Width = width, Height = height, Depth = 1 }, 0, 0);
            }
        }

        var texels = new ulong[width * height];
        for (var index = 0; index < texels.Length; index++)
        {
            texels[index] = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(index * sizeof(ulong)));
        }

        return texels;
    }

    public static uint[] ReadAtomicUintTexture(nint texture, uint width, uint height)
    {
        var bytes = new byte[width * height * sizeof(uint)];
        unsafe
        {
            fixed (byte* raw = bytes)
            {
                SendReadTexture(texture, Selector("getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:"),
                    (nint)raw, width * sizeof(uint), 0,
                    new MtlRegion { Width = width, Height = height, Depth = 1 }, 0, 0);
            }
        }

        var texels = new uint[width * height];
        for (var index = 0; index < texels.Length; index++)
        {
            texels[index] = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint)));
        }

        return texels;
    }

    public static byte[] ReadFloatTexture(nint texture, uint width, uint height, uint layers, uint depth)
    {
        var data = new byte[width * height * layers * depth * 16];
        unsafe
        {
            fixed (byte* bytes = data)
            {
                for (uint layer = 0; layer < layers; layer++)
                {
                    SendReadTexture(texture, Selector("getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:"),
                        (nint)(bytes + layer * width * height * 16), width * 16, depth > 1 ? width * height * 16 : 0,
                        new MtlRegion { Width = width, Height = height, Depth = depth }, 0, layer);
                }
            }
        }
        return data;
    }

    /// <summary>
    /// A private RGBA32Float multisampled texture (2D, or a 2D array when
    /// <paramref name="layers"/> exceeds one) whose sample s of texel (x, y) in layer l holds
    /// l * 1000 + s * 100 + y * 10 + x plus the channel index. A multisampled texture cannot
    /// be uploaded, so each layer is drawn by a per-sample fragment shader.
    /// </summary>
    public static nint CreateMultisampleFloatTexture(uint width, uint height, uint samples, uint layers)
    {
        const string source = """
            #include <metal_stdlib>
            using namespace metal;
            struct Fill { float4 position [[position]]; };
            vertex Fill fill_vertex(uint id [[vertex_id]])
            {
                float2 corner = float2((id << 1) & 2u, id & 2u);
                return { float4(corner * 2.0f - 1.0f, 0.0f, 1.0f) };
            }
            fragment float4 fill_fragment(Fill in [[stage_in]], uint sample [[sample_id]], constant uint& layer [[buffer(0)]])
            {
                float2 texel = floor(in.position.xy);
                float value = float(layer) * 1000.0f + float(sample) * 100.0f + texel.y * 10.0f + texel.x;
                return float4(value, value + 1.0f, value + 2.0f, value + 3.0f);
            }
            """;
        if (!TryCompileLibrary(source, out var library, out var error))
        {
            throw new InvalidOperationException($"MSAA fill shader: {error}");
        }

        var descriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(descriptor, Selector("setPixelFormat:"), 125); // RGBA32Float
        SendInteger(descriptor, Selector("setTextureType:"), layers > 1 ? 8u : 4u); // 2DMultisample(Array)
        SendInteger(descriptor, Selector("setWidth:"), width);
        SendInteger(descriptor, Selector("setHeight:"), height);
        SendInteger(descriptor, Selector("setDepth:"), 1);
        SendInteger(descriptor, Selector("setArrayLength:"), layers);
        SendInteger(descriptor, Selector("setSampleCount:"), samples);
        SendInteger(descriptor, Selector("setStorageMode:"), 2); // private
        SendInteger(descriptor, Selector("setUsage:"), 1 | 4); // shader read, render target
        var texture = Send(Device.Value, Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0) throw new InvalidOperationException("Failed to create the multisampled fixture texture.");

        var pipelineDescriptor = Send(Send(objc_getClass("MTLRenderPipelineDescriptor"), Selector("alloc")), Selector("init"));
        SendVoid(pipelineDescriptor, Selector("setVertexFunction:"), Send(library, Selector("newFunctionWithName:"), NsString("fill_vertex")));
        SendVoid(pipelineDescriptor, Selector("setFragmentFunction:"), Send(library, Selector("newFunctionWithName:"), NsString("fill_fragment")));
        SendInteger(pipelineDescriptor, Selector("setRasterSampleCount:"), samples);
        var colorDescriptor = Send(Send(pipelineDescriptor, Selector("colorAttachments")), Selector("objectAtIndexedSubscript:"), 0);
        SendInteger(colorDescriptor, Selector("setPixelFormat:"), 125);
        nint pipelineError = 0;
        var pipeline = Send(Device.Value, Selector("newRenderPipelineStateWithDescriptor:error:"), pipelineDescriptor, ref pipelineError);
        if (pipeline == 0) throw new InvalidOperationException($"MSAA fill pipeline: {DescribeError(pipelineError)}");

        var queue = SharedQueue.Value;
        var commandBuffer = Send(queue, Selector("commandBuffer"));
        for (uint layer = 0; layer < layers; layer++)
        {
            var pass = Send(objc_getClass("MTLRenderPassDescriptor"), Selector("renderPassDescriptor"));
            var attachment = Send(Send(pass, Selector("colorAttachments")), Selector("objectAtIndexedSubscript:"), 0);
            SendVoid(attachment, Selector("setTexture:"), texture);
            SendInteger(attachment, Selector("setSlice:"), layer);
            SendInteger(attachment, Selector("setLoadAction:"), 0); // don't care: every sample is drawn
            SendInteger(attachment, Selector("setStoreAction:"), 1); // store
            var encoder = Send(commandBuffer, Selector("renderCommandEncoderWithDescriptor:"), pass);
            SendVoid(encoder, Selector("setRenderPipelineState:"), pipeline);
            unsafe
            {
                var value = layer;
                SendSetBuffer(encoder, Selector("setFragmentBytes:length:atIndex:"), (nint)(&value), sizeof(uint), 0);
            }

            SendDraw(encoder, Selector("drawPrimitives:vertexStart:vertexCount:"), 3, 0, 3); // triangles
            SendVoid(encoder, Selector("endEncoding"));
        }

        SendVoid(commandBuffer, Selector("commit"));
        SendVoid(commandBuffer, Selector("waitUntilCompleted"));
        return texture;
    }

    /// <summary>A single-level shared 2D texture of any pixel format, filled from raw
    /// bytes, readable and writable by shaders.</summary>
    public static nint CreateRawTexture(nuint pixelFormat, uint width, uint height, uint bytesPerPixel, byte[] bytes)
    {
        var descriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(descriptor, Selector("setPixelFormat:"), pixelFormat);
        SendInteger(descriptor, Selector("setTextureType:"), 2);
        SendInteger(descriptor, Selector("setWidth:"), width);
        SendInteger(descriptor, Selector("setHeight:"), height);
        SendInteger(descriptor, Selector("setStorageMode:"), 0);
        SendInteger(descriptor, Selector("setUsage:"), 3);
        var texture = Send(Device.Value, Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0) throw new InvalidOperationException($"Failed to create a pixel format {pixelFormat} fixture texture.");
        unsafe
        {
            fixed (byte* raw = bytes)
            {
                SendReplace(texture, Selector("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
                    new MtlRegion { Width = width, Height = height, Depth = 1 }, 0, 0,
                    (nint)raw, width * bytesPerPixel, 0);
            }
        }

        return texture;
    }

    public static byte[] ReadRawTexture(nint texture, uint width, uint height, uint bytesPerPixel)
    {
        var bytes = new byte[width * height * bytesPerPixel];
        unsafe
        {
            fixed (byte* raw = bytes)
            {
                SendReadTexture(texture, Selector("getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:"),
                    (nint)raw, width * bytesPerPixel, 0,
                    new MtlRegion { Width = width, Height = height, Depth = 1 }, 0, 0);
            }
        }

        return bytes;
    }

    public static nint CreatePointSampler()
    {
        var descriptor = Send(Send(objc_getClass("MTLSamplerDescriptor"), Selector("alloc")), Selector("init"));
        SendVoidBool(descriptor, Selector("setSupportArgumentBuffers:"), true);
        SendInteger(descriptor, Selector("setNormalizedCoordinates:"), 1);
        SendInteger(descriptor, Selector("setMipFilter:"), 1);
        return Send(Device.Value, Selector("newSamplerStateWithDescriptor:"), descriptor);
    }

    // A zero-filled table for every read-only pointer field of an argument buffer, kept
    // for the process so one allocation serves every execution.
    // One command queue for the process: a queue per execution runs the device out of queues
    // partway through a large conformance category.
    private static readonly Lazy<nint> SharedQueue = new(() => Send(Device.Value, Selector("newCommandQueue")));

    private const uint SharedTableDwords = 256;

    private static readonly Lazy<nint> SharedTable =
        new(() => SharedBuffer(new byte[SharedTableDwords * sizeof(uint)]));

    private static ulong SharedTableAddress() =>
        SendULong(SharedTable.Value, Selector("gpuAddress"));

    // options 0 = MTLResourceStorageModeShared: CPU-visible for readback.
    private static nint SharedBuffer(byte[] contents)
    {
        unsafe
        {
            fixed (byte* bytes = contents)
            {
                return SendBuffer(
                    Device.Value,
                    Selector("newBufferWithBytes:length:options:"),
                    (nint)bytes,
                    (nuint)contents.Length,
                    0);
            }
        }
    }

    private static bool RunKernel(
        nint library,
        string entryPoint,
        IReadOnlyList<(nint Buffer, nuint Index)> bindings,
        IReadOnlyList<nint> residentBuffers,
        uint threadsPerThreadgroup,
        nint resultBuffer,
        int resultLength,
        out byte[] result,
        out string error,
        uint threadgroupCount = 1)
    {
        result = [];
        error = string.Empty;

        var function = Send(library, Selector("newFunctionWithName:"), NsString(entryPoint));
        if (function == 0)
        {
            error = $"entry point '{entryPoint}' not found in the compiled library";
            return false;
        }

        nint nsError = 0;
        var pipeline = Send(
            Device.Value,
            Selector("newComputePipelineStateWithFunction:error:"),
            function,
            ref nsError);
        if (pipeline == 0)
        {
            error = $"pipeline creation failed: {DescribeError(nsError)}";
            return false;
        }

        var queue = SharedQueue.Value;
        if (queue == 0)
        {
            error = "failed to create the command queue";
            return false;
        }

        var commandBuffer = Send(queue, Selector("commandBuffer"));
        var encoder = Send(commandBuffer, Selector("computeCommandEncoder"));
        SendVoid(encoder, Selector("setComputePipelineState:"), pipeline);
        foreach (var (buffer, index) in bindings)
        {
            SendSetBuffer(encoder, Selector("setBuffer:offset:atIndex:"), buffer, 0, index);
        }

        foreach (var resident in residentBuffers)
        {
            SendUseResource(encoder, Selector("useResource:usage:"), resident, ReadWriteUsage);
        }

        var groups = new MtlSize { Width = threadgroupCount, Height = 1, Depth = 1 };
        var threads = new MtlSize { Width = threadsPerThreadgroup, Height = 1, Depth = 1 };
        SendDispatch(encoder, Selector("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads);
        SendVoid(encoder, Selector("endEncoding"));
        SendVoid(commandBuffer, Selector("commit"));
        SendVoid(commandBuffer, Selector("waitUntilCompleted"));

        var commandError = Send(commandBuffer, Selector("error"));
        if (commandError != 0)
        {
            error = "GPU execution failed: " + DescribeError(commandError);
            return false;
        }
        var contentsPointer = Send(resultBuffer, Selector("contents"));
        if (contentsPointer == 0)
        {
            error = "buffer contents unavailable after execution";
            return false;
        }

        result = new byte[resultLength];
        Marshal.Copy(contentsPointer, result, 0, result.Length);
        return true;
    }
}
