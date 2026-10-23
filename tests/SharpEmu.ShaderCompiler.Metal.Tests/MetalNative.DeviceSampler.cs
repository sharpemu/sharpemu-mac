// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

internal static partial class MetalNative
{
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendSamplerFloat(nint receiver, nint selector, float value);

    // Hardware reference for the explicit-LOD sampler tests. Independent of the
    // generated tap/filter helpers, and configured from the same four guest words.
    internal static nint CreateReferenceSampler(uint[] words)
    {
        var descriptor = Send(Send(objc_getClass("MTLSamplerDescriptor"), Selector("alloc")), Selector("init"));
        SendVoidBool(descriptor, Selector("setSupportArgumentBuffers:"), true);
        SendInteger(descriptor, Selector("setNormalizedCoordinates:"), (words[0] & 0x8000) == 0 ? 1u : 0u);
        SendInteger(descriptor, Selector("setMagFilter:"), (words[2] >> 20) & 1);
        SendInteger(descriptor, Selector("setMinFilter:"), (words[2] >> 22) & 1);
        var mip = (words[2] >> 26) & 3;
        SendInteger(descriptor, Selector("setMipFilter:"), mip == 3 ? 1u : mip);
        if (mip != 0)
        {
            SendSamplerFloat(descriptor, Selector("setLodMinClamp:"), (words[1] & 0xFFF) / 256f);
            SendSamplerFloat(descriptor, Selector("setLodMaxClamp:"), ((words[1] >> 12) & 0xFFF) / 256f);
        }
        for (var axis = 0; axis < 3; axis++)
        {
            var mode = (words[0] >> (axis * 3)) & 7;
            var metalMode = mode switch { 0 => 2u, 1 => 3u, 2 => 0u, 3 or 5 or 7 => 1u, _ => 5u };
            SendInteger(descriptor, Selector(new[] { "setSAddressMode:", "setTAddressMode:", "setRAddressMode:" }[axis]), metalMode);
        }
        var border = words[3] >> 30;
        SendInteger(descriptor, Selector("setBorderColor:"), border is 1 or 2 ? border : 0);
        var sampler = Send(Device.Value, Selector("newSamplerStateWithDescriptor:"), descriptor);
        SendVoid(descriptor, Selector("release"));
        return sampler;
    }
}
