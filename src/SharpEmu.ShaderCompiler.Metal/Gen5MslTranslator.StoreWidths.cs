// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// An integer image store keeps the low bits of each register, where a Metal write saturates a value its view format
// cannot hold (0xFFFF8000 written to an R16Uint view stores 0xFFFF; the hardware, and upstream's store through the
// unsigned view, 0x8000). Each store narrows its components to the width of the format the host binds: the width code
// (StoreComponentWidthCode) of an image with a store channel map sits above its map, and every other written integer
// storage image has a width slot of its own (ImageStoreWidth, BindingLayout.StorageImageWidthsDword).
public static partial class Gen5MslTranslator
{
    /// <summary>
    /// Writes the store width code of every written integer storage image from its bound T#: above the store channel
    /// map of an image that has one (its bits 12-13), in the image's width slot otherwise. A null or absent descriptor
    /// gives code 0 (stored whole; the write lands on a null texture).
    /// </summary>
    public static void WriteStoreWidths(uint[] shaderData, BindingLayout layout, IReadOnlyList<ImageResource> images,
        bool runtimeImageSwizzles, IReadOnlyList<uint[]> descriptors)
    {
        if (layout.StorageImageWidthCount != 0)
        {
            Array.Clear(shaderData, (int)layout.StorageImageWidthsDword, (int)ImageStoreWidth.DwordCount(layout.StorageImageWidthCount));
        }

        var mapSlot = 0;
        var widthSlot = 0;
        for (var index = 0; index < images.Count; index++)
        {
            var image = images[index];
            var hasMap = layout.StorageImageSwizzleCount != 0 && image.Written && RuntimeImageSwizzle.Serves(image);
            var hasWidth = ImageStoreWidth.TakesSlot(image, runtimeImageSwizzles);
            if (!hasMap && !hasWidth)
            {
                continue;
            }

            var code = index < descriptors.Count && descriptors[index] is { Length: >= 8 } words
                ? StoreComponentWidthCode(GuestImageFormat.FormatOf(words))
                : 0u;
            if (hasMap)
            {
                if (mapSlot < layout.StorageImageSwizzleCount)
                {
                    shaderData[(int)layout.StorageImageSwizzlesDword + (mapSlot / 2)] |= code << (StoreWidthCodeShift + (mapSlot % 2 * 16));
                }

                mapSlot++;
                continue;
            }

            if (widthSlot >= layout.StorageImageWidthCount)
            {
                throw new InvalidOperationException(
                    $"The store width slots do not match the layout: image={index} slot={widthSlot} layout={layout.StorageImageWidthCount}.");
            }

            shaderData[(int)layout.StorageImageWidthsDword + (widthSlot / ImageStoreWidth.SlotsPerDword)] |=
                code << (widthSlot % ImageStoreWidth.SlotsPerDword * ImageStoreWidth.BitsPerSlot);
            widthSlot++;
        }

        if (widthSlot != layout.StorageImageWidthCount)
        {
            throw new InvalidOperationException(
                $"The store width slots do not match the layout: images={widthSlot} layout={layout.StorageImageWidthCount}.");
        }
    }

    private sealed partial class CompilationContext
    {
        // The width code of an integer store with no store channel map, read from the image's width slot; null, with the
        // error, when the layout gives the image no slot: its store would saturate, so the program is refused.
        private string? StoreWidthCode(int resourceIndex, out string error)
        {
            error = string.Empty;
            var slot = resourceIndex < 0 ? -1
                : ImageStoreWidth.Slot(_request.Resources.Info.Images, resourceIndex, _request.Resources.Info.UsesRuntimeImageSwizzles);
            if (slot < 0 || slot >= _request.Bindings.StorageImageWidthCount)
            {
                error = $"integer image store to image {resourceIndex} has no store width slot (slots={_request.Bindings.StorageImageWidthCount})";
                return null;
            }

            var dword = _request.Bindings.StorageImageWidthsDword + (uint)(slot / ImageStoreWidth.SlotsPerDword);
            var shift = slot % ImageStoreWidth.SlotsPerDword * ImageStoreWidth.BitsPerSlot;
            return Temp("uint", $"({ShaderDataDword(dword)} >> {shift}u) & 3u");
        }

        // One integer component of a store, as uint bits, narrowed to the bits of the physical channel the width code
        // gives (1: 8, 2: 16, 3: 10_10_10_2, 0: whole) and, for an int texture, sign-extended from them, so the write
        // keeps the register's low bits instead of saturating; the result is the texel component of kind.
        private string NarrowStoreComponent(string kind, string raw, string width, uint channel)
        {
            var bits = Temp("uint",
                $"{width} == 1u ? 8u : {width} == 2u ? 16u : {width} == 3u ? {(channel == 3 ? 2 : 10)}u : 32u");
            var value = Temp("uint", raw);
            return kind == "int"
                ? $"({bits} == 32u ? as_type<int>({value}) : extract_bits(as_type<int>({value}), 0u, {bits}))"
                : $"({bits} == 32u ? {value} : extract_bits({value}, 0u, {bits}))";
        }
    }
}
