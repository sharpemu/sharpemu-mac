// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

// An integer image store keeps the low bits of each register (upstream stores a write-only SINT image through its UINT
// view, 99f9fe930), where a Metal write saturates a value its view format cannot hold, so the MSL store narrows each
// component to the width of the format the host binds. A written image with a store channel map (RuntimeImageSwizzle)
// carries that width above its map; every other written integer storage image (the select baked into the module with
// SHARPEMU_RUNTIME_IMAGE_SWIZZLE=0, an indirect table's candidate) takes a width slot here: 2 bits per image, 16 to a
// dword (BindingLayout.StorageImageWidthsDword), written by the host from the image's bound T#. A host whose stores keep
// the low bits by themselves leaves the slots zero, which narrows nothing.
public static class ImageStoreWidth
{
    public const int BitsPerSlot = 2;
    public const int SlotsPerDword = 32 / BitsPerSlot;

    // A written integer storage image whose stores have no store channel map to carry the width.
    public static bool TakesSlot(ImageResource image, bool runtimeImageSwizzles) =>
        image.Written && image.ResourceClass == ImageResourceClass.Storage &&
        image.NumericClass is ImageNumericClass.Uint or ImageNumericClass.Sint &&
        !(runtimeImageSwizzles && RuntimeImageSwizzle.Serves(image));

    // The image's width slot: its rank among the images that take one, or -1 for any other image.
    public static int Slot(IReadOnlyList<ImageResource> images, int index, bool runtimeImageSwizzles)
    {
        if ((uint)index >= (uint)images.Count || !TakesSlot(images[index], runtimeImageSwizzles))
        {
            return -1;
        }

        var slot = 0;
        for (var other = 0; other < index; other++)
        {
            slot += TakesSlot(images[other], runtimeImageSwizzles) ? 1 : 0;
        }

        return slot;
    }

    public static uint SlotCount(IReadOnlyList<ImageResource> images, bool runtimeImageSwizzles)
    {
        var count = 0u;
        foreach (var image in images)
        {
            count += TakesSlot(image, runtimeImageSwizzles) ? 1u : 0u;
        }

        return count;
    }

    public static uint DwordCount(uint slots) => (slots + SlotsPerDword - 1) / SlotsPerDword;
}
