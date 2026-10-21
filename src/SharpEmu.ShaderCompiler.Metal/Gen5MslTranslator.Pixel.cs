// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {

        private static bool UsesSampler(string opcode) => Gen5ImageOpcodes.UsesSampler(opcode);

        // ---- image instruction emission ----

        private bool TryEmitImage(
            Gen5ShaderInstruction instruction,
            Gen5ImageControl image,
            out string error)
        {
            error = string.Empty;
            if (instruction.Opcode is "ImageBvhIntersectRay" or "ImageBvh64IntersectRay")
            {
                // The decoder gives the BVH opcodes a Gen5RayIntersectControl
                // (TryEmitRayIntersect); with an image control the operands are not the ray's.
                error = $"{instruction.Opcode}: a BVH intersection needs its ray-intersect control, not an image control";
                return false;
            }

            string texture;
            string samplerName;
            MslImageClass imageClass;
            uint dstSelect;
            string mipLevel;
            int compareFunction;
            uint packedFormat;
            SamplerResource? samplerInfo;
            int resourceIndex;
            {
                if (TryGetImageElementCases(instruction, image, out var selector, out var elements, out error))
                {
                    // One case per descriptor over a constant element, like a switch on the selector.
                    for (var index = 0; index < elements.Count; index++)
                    {
                        Line($"if ({selector} == {index}u)");
                        Line("{");
                        _indent++;
                        var emitted =
                            TryResolveLayoutImage(instruction, image, out texture, out samplerName, out imageClass, out dstSelect, out mipLevel, out compareFunction, out packedFormat, out samplerInfo, out error, out resourceIndex, elements[index]) &&
                            EmitImageOperation(instruction, image, texture, samplerName, imageClass, dstSelect, mipLevel, compareFunction, packedFormat, samplerInfo, resourceIndex, out error);
                        _indent--;
                        Line("}");
                        if (!emitted)
                        {
                            return false;
                        }
                    }

                    return true;
                }

                if (error.Length != 0)
                {
                    return false;
                }

                if (!TryResolveLayoutImage(instruction, image, out texture, out samplerName, out imageClass, out dstSelect, out mipLevel, out compareFunction, out packedFormat, out samplerInfo, out error, out resourceIndex))
                {
                    return false;
                }
            }

            return EmitImageOperation(instruction, image, texture, samplerName, imageClass, dstSelect, mipLevel, compareFunction, packedFormat, samplerInfo, resourceIndex, out error);
        }

        // One image operation over a resolved texture; its results are register writes.
        private bool EmitImageOperation(
            Gen5ShaderInstruction instruction,
            Gen5ImageControl image,
            string texture,
            string samplerName,
            MslImageClass imageClass,
            uint dstSelect,
            string mipLevel,
            int compareFunction,
            uint packedFormat,
            SamplerResource? samplerInfo,
            int resourceIndex,
            out string error)
        {
            error = string.Empty;
            var kind = imageClass.ComponentKind;
            var isStorage = imageClass.IsStorage;
            var flags = Gen5ImageOpcodes.FlagsOf(instruction.Opcode);
            var packed = (flags & Gen5ImageOp.Packed) != 0;
            if (packed && image.D16)
            {
                error = $"{instruction.Opcode} moves raw texel dwords and has no D16 form";
                return false;
            }
            if (instruction.Opcode == "ImageGetResinfo")
            {
                EmitImageGetResinfo(image, texture, imageClass);
                return true;
            }

            if (instruction.Opcode == "ImageGetLod")
            {
                EmitImageGetLod(image, texture, samplerName, imageClass);
                return true;
            }

            if (instruction.Opcode.StartsWith("ImageAtomic", StringComparison.Ordinal))
            {
                return TryEmitImageAtomic(instruction, image, texture, imageClass, out error);
            }

            if ((flags & Gen5ImageOp.Store) != 0)
            {
                if (!isStorage)
                {
                    error = "image store is not bound as storage";
                    return false;
                }

                // Raw signed spatial address components plus the layer, all bounds-checked
                // together so an out-of-range texel drops the write instead of wrapping.
                var spatial = new string[imageClass.SpatialComponents];
                var guards = new List<string>(4);
                for (var component = 0; component < spatial.Length; component++)
                {
                    spatial[component] = Temp("int", $"as_type<int>({ImageIntegerAddress(image, component)})");
                    guards.Add($"{spatial[component]} >= 0");
                    guards.Add($"{spatial[component]} < (int){ImageExtentQuery(texture, imageClass, component, null)}");
                }

                var layerArgument = string.Empty;
                if (imageClass.Arrayed)
                {
                    var layer = Temp("int", $"as_type<int>({ImageIntegerAddress(image, imageClass.SpatialComponents)})");
                    guards.Add($"{layer} >= 0");
                    guards.Add($"{layer} < (int){texture}.get_array_size()");
                    layerArgument = $", (uint){layer}";
                }

                var components = new string[4];
                if (packed)
                {
                    if (!TryUnpackImageTexel(image, kind, packedFormat, out components, out error))
                    {
                        return false;
                    }
                }
                else
                {
                    for (var component = 0; component < 4; component++)
                    {
                        var sourceIndex = Gen5ShaderTranslator.GetImageStoreSourceIndex(
                            dstSelect,
                            image.Dmask,
                            component);
                        components[component] = sourceIndex >= 0
                            ? ImageTexelComponent(
                                kind,
                                ImageStoreComponent(image, kind, (uint)sourceIndex))
                            : kind == "float" ? "0.0f" : "0";
                    }
                }

                var coordinate = spatial.Length == 1
                    ? $"uint2((uint){spatial[0]}, 0u)"
                    : $"{imageClass.IntegerCoordinateType}({string.Join(", ", spatial.Select(part => $"(uint){part}"))})";

                // Bounds-checked, EXEC-guarded write.
                Line($"if ({ExecCondition(string.Join(" && ", guards))})");
                Line("{");
                _indent++;
                Line($"{texture}.write({VectorLiteral(kind)}({components[0]}, {components[1]}, {components[2]}, {components[3]}), {coordinate}{layerArgument});");
                _indent--;
                Line("}");
                return true;
            }

            if (isStorage && (flags & Gen5ImageOp.Load) == 0)
            {
                error = $"unsupported storage image opcode {instruction.Opcode}";
                return false;
            }

            string sampled;
            var writeAllComponents = false;
            var resultKind = kind;
            if ((flags & Gen5ImageOp.Load) != 0)
            {
                // A storage element is its own mip view; multisampled textures
                // carry a sample operand rather than a mip level.
                // The guest reads a zero texel, through the view's swizzle, anywhere outside
                // the image. Metal does the same for a coordinate past the extent, but a mip,
                // layer or sample past the last one is undefined - garbage, or a GPU fault, for
                // UE5's passes that march off a volume. So those are clamped, and a read they
                // would have taken out of range goes past the extent instead, where Metal
                // returns the swizzled zero the guest gets (an edge texel is not).
                var readsMip = !isStorage && imageClass.SupportsMipLevels && !imageClass.Multisampled;
                var outside = new List<string>();
                var mip = readsMip ? Temp("uint", mipLevel) : null;
                if (mip is not null)
                {
                    outside.Add($"{mip} >= {texture}.get_num_mip_levels()");
                }

                var layer = imageClass.Arrayed
                    ? Temp("uint", BuildIntegerLayer(image, imageClass.SpatialComponents, texture))
                    : null;
                if (layer is not null)
                {
                    outside.Add($"{layer} >= {texture}.get_array_size()");
                }

                // 2DMsaa/2DMsaaArray take a sample index where a mip level would go; it is
                // the address component right after the coordinates, like the mip operand.
                var sampleIndex = imageClass.Multisampled
                    ? Temp("uint", ImageIntegerAddress(image, imageClass.CoordinateComponents))
                    : null;
                if (sampleIndex is not null)
                {
                    outside.Add($"{sampleIndex} >= {texture}.get_num_samples()");
                }

                var away = outside.Count == 0 ? null : Temp("bool", string.Join(" || ", outside));
                var spatial = new string[imageClass.SpatialComponents];
                for (var component = 0; component < spatial.Length; component++)
                {
                    var address = ImageIntegerAddress(image, component);
                    spatial[component] = Temp("uint", away is null ? address : $"{away} ? 0xFFFFFFFFu : ({address})");
                }

                var coordinate = spatial.Length == 1
                    ? $"uint2({spatial[0]}, 0u)"
                    : $"{imageClass.IntegerCoordinateType}({string.Join(", ", spatial)})";
                var arguments = coordinate;
                if (layer is not null)
                {
                    arguments += $", ({away} ? 0u : {layer})";
                }

                if (sampleIndex is not null)
                {
                    arguments += $", ({away} ? 0u : {sampleIndex})";
                }
                else if (mip is not null)
                {
                    arguments += $", ({away} ? 0u : {mip})";
                }

                sampled = Temp($"vec<{kind}, 4>", $"{texture}.read({arguments})");
                if (packed)
                {
                    // DMASK selects texel dwords rather than components.
                    if (!TryPackImageTexel(sampled, kind, packedFormat, (flags & Gen5ImageOp.PackedSigned) != 0, out sampled, out error))
                    {
                        return false;
                    }

                    resultKind = "uint";
                }
            }
            else if (instruction.Opcode == "ImageMsaaLoad")
            {
                if (!imageClass.Multisampled)
                {
                    error = "image_msaa_load reads a multisampled image";
                    return false;
                }

                sampled = EmitImageMsaaLoad(image, texture, imageClass);
                writeAllComponents = true;
            }
            else if (instruction.Opcode.StartsWith("ImageSample", StringComparison.Ordinal))
            {
                if (!TryEmitImageSample(instruction, image, texture, samplerName, imageClass, compareFunction, samplerInfo, resourceIndex, out sampled, out error))
                {
                    return false;
                }
            }
            else if (instruction.Opcode.StartsWith("ImageGather4", StringComparison.Ordinal))
            {
                if (!TryEmitImageGather(instruction, image, texture, samplerName, imageClass, compareFunction, samplerInfo, out sampled, out error))
                {
                    return false;
                }

                writeAllComponents = true;
            }
            else
            {
                error = $"unsupported image opcode {instruction.Opcode}";
                return false;
            }

            var outputValues = new List<string>(4);
            for (var component = 0; component < 4; component++)
            {
                if (!writeAllComponents && (image.Dmask & (1u << component)) == 0)
                {
                    continue;
                }

                var value = $"{sampled}[{component}]";
                outputValues.Add(resultKind == "uint" ? value : AsUInt(value));
            }

            if (image.D16)
            {
                for (var index = 0; index < outputValues.Count; index += 2)
                {
                    var low = outputValues[index];
                    var high = index + 1 < outputValues.Count ? outputValues[index + 1] : "0u";
                    StoreVector(
                        image.VectorData + (uint)(index / 2),
                        PackImageD16(kind, low, high));
                }
            }
            else
            {
                for (var index = 0; index < outputValues.Count; index++)
                {
                    StoreVector(image.VectorData + (uint)index, outputValues[index]);
                }
            }

            return true;
        }

        // IMAGE_MSAA_LOAD: the one DMASK component of four consecutive fragments starting at
        // the address's fragment ID, in four registers (two under D16). The host keeps an MSAA
        // surface as its samples with no FMASK, so a fragment ID is the sample index; a fragment
        // past the texture's sample count reads zero, as Metal leaves that read undefined.
        private string EmitImageMsaaLoad(Gen5ImageControl image, string texture, MslImageClass imageClass)
        {
            var kind = imageClass.ComponentKind;
            var spatial = Enumerable.Range(0, imageClass.SpatialComponents)
                .Select(component => Temp("uint", ImageIntegerAddress(image, component)))
                .ToArray();
            var arguments = $"{imageClass.IntegerCoordinateType}({string.Join(", ", spatial)})";
            if (imageClass.Arrayed)
            {
                arguments += $", {BuildIntegerLayer(image, imageClass.SpatialComponents, texture)}";
            }

            var fragment = Temp("uint", ImageIntegerAddress(image, imageClass.CoordinateComponents));
            var samples = Temp("uint", $"{texture}.get_num_samples()");
            var component = "xyzw"[(int)GatherComponent(image.Dmask)];
            var zero = $"vec<{kind}, 4>(0)";
            var texels = new string[4];
            for (var index = 0; index < texels.Length; index++)
            {
                var sample = Temp("uint", $"{fragment} + {index}u");
                texels[index] = Temp(kind, $"({sample} < {samples} ? {texture}.read({arguments}, {sample}) : {zero}).{component}");
            }

            return Temp($"vec<{kind}, 4>", $"{VectorLiteral(kind)}({string.Join(", ", texels)})");
        }

        // The size query reports the spatial extent, then the layer count for an arrayed
        // image, then the mip count in w. Components past the image's own rank read zero,
        // mirroring the SPIR-V backend's OpImageQuerySize{Lod} handling.
        private void EmitImageGetResinfo(Gen5ImageControl image, string texture, MslImageClass imageClass)
        {
            // Storage elements and multisampled textures have no queryable mip chain.
            var queryable = !imageClass.IsStorage && !imageClass.Multisampled;
            var lod = queryable ? Temp("uint", ImageIntegerAddress(image, 0)) : null;
            var sizes = new string[imageClass.CoordinateComponents];
            for (var component = 0; component < imageClass.SpatialComponents; component++)
            {
                sizes[component] = Temp("uint", ImageExtentQuery(texture, imageClass, component, lod));
            }

            if (imageClass.Arrayed)
            {
                sizes[imageClass.SpatialComponents] = Temp("uint", $"{texture}.get_array_size()");
            }

            var levels = (image.Dmask & 0x8) != 0 && queryable
                ? Temp("uint", $"{texture}.get_num_mip_levels()")
                : "1u";
            uint outputIndex = 0;
            for (var component = 0; component < 4; component++)
            {
                if ((image.Dmask & (1u << component)) == 0)
                {
                    continue;
                }

                StoreVector(
                    image.VectorData + outputIndex++,
                    component < sizes.Length ? sizes[component]
                        : component == 3 ? levels
                        : "0u");
            }
        }

        // IMAGE_ATOMIC_* over a read_write storage texture. Metal's texture atomics
        // (MSL 3.1) take and return the whole texel as a 4-vector, and the guest
        // binding for an atomic image is always single-channel uint, so only x
        // carries data. The ops Metal has no member for -- the signed minimum and
        // maximum over a uint texture, the wrap-clamping increment and decrement,
        // compare-swap and the float forms -- read the texel, derive the
        // replacement and compare-exchange until the texel they computed from is
        // still the one in memory.
        private bool TryEmitImageAtomic(
            Gen5ShaderInstruction instruction,
            Gen5ImageControl image,
            string texture,
            MslImageClass imageClass,
            out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            if (!imageClass.IsStorage || imageClass.Multisampled)
            {
                error = $"{opcode} needs a single-sampled storage image binding";
                return false;
            }

            // BindingLayout only gives an atomic binding to a uint image, so this
            // is the one component kind the texel vector can have, or ulong for
            // a 64-bit atomic.
            var wide = ResourceTracker.Is64BitImageAtomic(opcode, image.Dmask);
            if (imageClass.ComponentKind != (wide ? "ulong" : "uint"))
            {
                error = $"{opcode} on a {imageClass.ComponentKind} image has no Metal texture atomic";
                return false;
            }

            // The address is bounds-checked like a store: Metal leaves an atomic on
            // an out-of-range texel undefined, so an invalid guest address drops.
            var spatial = new string[imageClass.SpatialComponents];
            var guards = new List<string>(4);
            for (var component = 0; component < spatial.Length; component++)
            {
                spatial[component] = Temp("int", $"as_type<int>({ImageIntegerAddress(image, component)})");
                guards.Add($"{spatial[component]} >= 0");
                guards.Add($"{spatial[component]} < (int){ImageExtentQuery(texture, imageClass, component, null)}");
            }

            var layerArgument = string.Empty;
            if (imageClass.Arrayed)
            {
                var layer = Temp("int", $"as_type<int>({ImageIntegerAddress(image, imageClass.SpatialComponents)})");
                guards.Add($"{layer} >= 0");
                guards.Add($"{layer} < (int){texture}.get_array_size()");
                layerArgument = $", (uint){layer}";
            }

            var coordinate = spatial.Length == 1
                ? $"uint2((uint){spatial[0]}, 0u)"
                : $"{imageClass.IntegerCoordinateType}({string.Join(", ", spatial.Select(part => $"(uint){part}"))})";
            var data = Temp("uint", $"v[{image.VectorData}]");
            // IMAGE_ATOMIC_CMPSWAP and FCMPSWAP read the new value from VDATA and
            // the comparator from the register after it. VDATA is an 8-bit field,
            // so the pair can name one register past the file.
            var comparator = Temp(
                "uint",
                image.VectorData + 1 < VectorRegisterFileCount ? $"v[{image.VectorData + 1}]" : "0u");
            var access = $"{coordinate}{layerArgument}";
            var suffix = opcode["ImageAtomic".Length..];
            if (wide)
            {
                // Metal's 64-bit texture atomics (MSL 3.1, RG32Uint textures) are only
                // the unsigned minimum and maximum, and neither returns the old texel.
                // That covers a visibility buffer's depth-and-ID maximum.
                var method = suffix switch { "Umax" => "atomic_max", "Umin" => "atomic_min", _ => null };
                if (method is null || image.Glc)
                {
                    error = $"{opcode} on 64-bit texels{(image.Glc ? " returning the old value" : string.Empty)} has no Metal texture atomic";
                    return false;
                }

                var wideData = Temp(
                    "ulong",
                    image.VectorData + 1 < VectorRegisterFileCount
                        ? $"as_type<ulong>(uint2(v[{image.VectorData}], v[{image.VectorData + 1}]))"
                        : $"(ulong)v[{image.VectorData}]");
                Line($"if ({ExecCondition(string.Join(" && ", guards))})");
                Line("{");
                _indent++;
                Line($"{texture}.{method}({access}, ulong4({wideData}));");
                _indent--;
                Line("}");
                return true;
            }

            Line($"if ({ExecCondition(string.Join(" && ", guards))})");
            Line("{");
            _indent++;
            var direct = suffix switch
            {
                "Swap" => $"{texture}.atomic_exchange({access}, uint4({data}))",
                "Add" => $"{texture}.atomic_fetch_add({access}, uint4({data}))",
                "Sub" => $"{texture}.atomic_fetch_sub({access}, uint4({data}))",
                "Umin" => $"{texture}.atomic_fetch_min({access}, uint4({data}))",
                "Umax" => $"{texture}.atomic_fetch_max({access}, uint4({data}))",
                "And" => $"{texture}.atomic_fetch_and({access}, uint4({data}))",
                "Or" => $"{texture}.atomic_fetch_or({access}, uint4({data}))",
                "Xor" => $"{texture}.atomic_fetch_xor({access}, uint4({data}))",
                _ => string.Empty,
            };

            string original;
            if (direct.Length > 0)
            {
                original = Temp("uint", $"{direct}.x");
            }
            else
            {
                var replacement = suffix switch
                {
                    "Smin" => (Func<string, string>)(value => $"(uint)min(as_type<int>({value}), as_type<int>({data}))"),
                    "Smax" => value => $"(uint)max(as_type<int>({value}), as_type<int>({data}))",
                    // IMAGE_ATOMIC_INC/DEC wrap against DATA rather than counting
                    // freely: INC resets at the limit, DEC reloads it.
                    "Inc" => value => $"{value} >= {data} ? 0u : {value} + 1u",
                    "Dec" => value => $"({value} == 0u || {value} > {data}) ? {data} : {value} - 1u",
                    "Cmpswap" => value => $"{value} == {comparator} ? {data} : {value}",
                    "Fcmpswap" => value =>
                        $"as_type<float>({value}) == as_type<float>({comparator}) ? {data} : {value}",
                    "Fmin" => value =>
                        $"as_type<float>({data}) < as_type<float>({value}) ? {data} : {value}",
                    "Fmax" => value =>
                        $"as_type<float>({data}) > as_type<float>({value}) ? {data} : {value}",
                    _ => null!,
                };
                if (replacement is null)
                {
                    _indent--;
                    Line("}");
                    error = $"unsupported image atomic {opcode}";
                    return false;
                }

                original = EmitImageAtomicExchangeLoop(texture, access, replacement);
            }

            if (image.Glc)
            {
                Line($"v[{image.VectorData}] = {original};");
            }

            _indent--;
            Line("}");
            return true;
        }

        // Reads the texel, derives the replacement from what it observed and swaps
        // it in, retrying until the texel it computed from is still in memory. A
        // failed exchange refreshes the observed value, so the replacement is
        // always derived from the texel the swap is compared against.
        private string EmitImageAtomicExchangeLoop(
            string texture,
            string access,
            Func<string, string> replacement)
        {
            var observed = Temp("uint4", $"{texture}.atomic_load({access})");
            var original = Temp("uint", $"{observed}.x");
            var exchanged = Temp("bool", "false");
            Line($"while (!{exchanged})");
            Line("{");
            _indent++;
            Line($"{original} = {observed}.x;");
            var next = Temp("uint", replacement(original));
            Line($"{exchanged} = {texture}.atomic_compare_exchange_weak({access}, &{observed}, uint4({next}));");
            _indent--;
            Line("}");
            return original;
        }

        // IMAGE_GET_LOD reports the level the sampler would pick for the address:
        // the clamped level in x and the unclamped one in y, which is what the
        // SPIR-V translator's OpImageQueryLod returns. A multisampled texture has
        // no mip chain to pick from, so both halves read zero.
        private void EmitImageGetLod(
            Gen5ImageControl image,
            string texture,
            string samplerName,
            MslImageClass imageClass)
        {
            var levels = new string[2];
            if (imageClass.SupportsSampleLod)
            {
                var coordinates = SampleCoordinates(image, SampleAddressOf(image, Gen5ImageOp.None), imageClass);
                if (imageClass.SpatialComponents == 1)
                {
                    coordinates = Temp("float2", $"float2({coordinates}, 0.5f)");
                }

                levels[0] = Temp("float", $"{texture}.calculate_clamped_lod({samplerName}, {coordinates})");
                levels[1] = Temp("float", $"{texture}.calculate_unclamped_lod({samplerName}, {coordinates})");
            }
            else
            {
                levels[0] = "0.0f";
                levels[1] = "0.0f";
            }

            // A zero DMASK still returns the clamped level, like the size query.
            var mask = image.Dmask != 0 ? image.Dmask : 1u;
            uint outputIndex = 0;
            for (var component = 0; component < 2; component++)
            {
                if ((mask & (1u << component)) == 0)
                {
                    continue;
                }

                StoreVector(image.VectorData + outputIndex++, AsUInt(levels[component]));
            }
        }

        // The extent accessor uses the guest's logical rank. Height-one 2D
        // backing preserves 1D mip queries; multisampled textures take no LOD.
        private static string ImageExtentQuery(string texture, MslImageClass imageClass, int component, string? lod)
        {
            var accessor = component switch
            {
                0 => "get_width",
                1 => "get_height",
                _ => "get_depth",
            };
            return lod is null || !imageClass.SupportsMipLevels
                ? $"{texture}.{accessor}()"
                : $"{texture}.{accessor}({lod})";
        }

        // The array layer of a sampled access. Hardware rounds the float layer to the
        // nearest slice, and Metal wants an integer, so the rounding is explicit.
        //
        // Guest cube layers advance by 8 per cube (6 faces plus 2 padding slices) while
        // the host 2D-array view packs 6, so the padding is subtracted. The float-to-integer
        // conversion truncates there, matching the SPIR-V backend's OpConvertFToU.
        private string BuildSampleLayer(Gen5ImageControl image, int component, MslImageClass imageClass, string texture)
        {
            var raw = ImageFloatAddress(image, component);
            string layer;
            if (imageClass.Cube)
            {
                var guest = Temp("uint", $"(uint)max({raw}, 0.0f)");
                layer = Temp("uint", $"{guest} - (({guest} >> 3u) << 1u)");
            }
            else
            {
                layer = Temp("uint", $"(uint)max(rint({raw}), 0.0f)");
            }

            return Temp("uint", $"min({layer}, {texture}.get_array_size() - 1u)");
        }

        // Integer layer accesses retain invalid values so the native read returns
        // zero rather than reading the nearest layer. Cube layers are raw here.
        private string BuildIntegerLayer(Gen5ImageControl image, int component, string texture) =>
            Temp("uint", ImageIntegerAddress(image, component));

        // MIMG UNRM: the address is in texels of mip 0, so it is divided by that level's extent to
        // reach the normalized coordinates the sampler expects (clamp/addressing stays the sampler's).
        private string NormalizeUnnormalizedCoordinates(string coordinates, string texture, MslImageClass imageClass)
        {
            var count = imageClass.SpatialComponents;
            if (count == 1)
            {
                return Temp("float", $"{coordinates} / (float){ImageExtentQuery(texture, imageClass, 0, "0u")}");
            }

            var extents = string.Join(
                ", ",
                Enumerable.Range(0, count).Select(component => $"(float){ImageExtentQuery(texture, imageClass, component, "0u")}"));
            return Temp(
                imageClass.FloatCoordinateType,
                $"{coordinates} / {imageClass.FloatCoordinateType}({extents})");
        }

        private static string VectorLiteral(string kind) => $"vec<{kind}, 4>";

        private static int ImageAddressRegister(Gen5ImageControl image, int component) =>
            image.A16 ? component / 2 : component;

        private static int ImageFullAddressSlots(Gen5ImageControl image) =>
            image.A16 ? 2 : 1;

        private static int AlignFullImageAddress(Gen5ImageControl image, int component) =>
            image.A16 ? (component + 1) & ~1 : component;

        /// <summary>Float address component, unpacking A16 half pairs.</summary>
        private string ImageFloatAddress(Gen5ImageControl image, int component)
        {
            var register = image.GetAddressRegister(ImageAddressRegister(image, component));
            return image.A16
                ? $"(float)as_type<half2>(v[{register}])[{component & 1}]"
                : $"as_type<float>(v[{register}])";
        }

        /// <summary>Integer address component, unpacking A16 16-bit pairs.</summary>
        private string ImageIntegerAddress(Gen5ImageControl image, int component)
        {
            var register = image.GetAddressRegister(ImageAddressRegister(image, component));
            return image.A16
                ? $"((v[{register}] >> {(component & 1) * 16}) & 0xFFFFu)"
                : $"v[{register}]";
        }

        /// <summary>One store-source component, unpacking D16 halves.</summary>
        private string ImageStoreComponent(Gen5ImageControl image, string kind, uint component)
        {
            if (!image.D16)
            {
                return $"v[{image.VectorData + component}]";
            }

            var packed = $"v[{image.VectorData + (component / 2)}]";
            if (kind == "float")
            {
                return AsUInt($"(float)as_type<half2>({packed})[{component & 1}]");
            }

            var low = $"(({packed} >> {(component & 1) * 16}) & 0xFFFFu)";
            return kind == "int"
                ? $"(uint)extract_bits(as_type<int>({low}), 0u, 16u)"
                : low;
        }

        private static string ImageTexelComponent(string kind, string raw) => kind switch
        {
            "int" => $"as_type<int>({raw})",
            "uint" => raw,
            _ => $"as_type<float>({raw})",
        };

        private string PackImageD16(string kind, string low, string high)
        {
            if (kind == "float")
            {
                return $"(((uint)as_type<ushort>(half(as_type<float>({low})))) | (((uint)as_type<ushort>(half(as_type<float>({high})))) << 16))";
            }

            return $"((({low}) & 0xFFFFu) | ((({high}) & 0xFFFFu) << 16))";
        }

        // ---- exports ----

        private bool TryEmitExport(
            Gen5ShaderInstruction instruction,
            Gen5ExportControl export,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 4)
            {
                error = "missing export sources";
                return false;
            }

            if (_stage == Gen5MslStage.Mesh)
            {
                return TryEmitMeshExport(instruction, export);
            }

            if (_stage == Gen5MslStage.Vertex)
            {
                return TryEmitVertexExport(instruction, export);
            }

            if (_stage != Gen5MslStage.Pixel)
            {
                // Compute programs have no export interface.
                return true;
            }

            // EXP.VM communicates the current EXEC mask independently of the
            // export data target. NULL exports are the hardware-defined way to
            // update only fragment validity, and the final VM export wins.
            if (export.ValidMask && _usesPixelValidMask)
            {
                // EXEC put back to the entry mask is every live fragment of the wave, this
                // lane included; saying so lets the end-of-program discard fold away.
                _entryMaskExports ??= FindEntryMaskExports(_request.Program.Instructions);
                if (_entryMaskExports.Contains(instruction.Pc))
                {
                    Line("pixel_valid_mask_active = true;");
                }
                else
                {
                    Line("pixel_valid_mask_active = exec;");
                    _validMaskFromExec = true;
                }
            }

            if (export.Target == MrtzExportTarget)
            {
                if (_pixelDepthExport && MrtzDepth(instruction) is { } depthValue)
                {
                    const string depth = "sharpemu_out.sharpemu_depth";
                    Line($"{depth} = {ExecSelect(depthValue, depth)};");
                }

                if (_pixelStencilExport && MrtzStencil(instruction) is { } stencilValue)
                {
                    const string stencil = "sharpemu_out.sharpemu_stencil";
                    Line($"{stencil} = {ExecSelect(stencilValue, stencil)};");
                }

                if (_pixelSampleMaskExport && MrtzSampleMask(instruction) is { } maskValue)
                {
                    const string coverage = "sharpemu_out.sharpemu_coverage";
                    Line($"{coverage} = {ExecSelect(maskValue, coverage)};");
                }

                return true;
            }

            Gen5PixelOutputBinding? binding = null;
            foreach (var candidate in _pixelOutputBindings)
            {
                if (candidate.ExportTarget == export.Target)
                {
                    binding = candidate;
                    break;
                }
            }

            if (binding is null)
            {
                return true;
            }

            var field = $"sharpemu_out.mrt{binding.Value.GuestSlot}";
            var componentType = binding.Value.Kind switch
            {
                Gen5PixelOutputKind.Uint => "uint",
                Gen5PixelOutputKind.Sint => "int",
                _ => "float",
            };
            var format = ColorExportFormat(binding.Value, export);
            if (format == Gen5ColorExportFormat.Zero)
            {
                // SPI_SHADER_ZERO: the export carries nothing to the target.
                return true;
            }

            var values = new string[4];
            for (var component = 0; component < 4; component++)
            {
                var source = ColorExportSource(format, component);
                if (source < 0)
                {
                    // A channel the format does not carry: green and blue are 0, alpha 1.
                    values[component] = component == 3
                        ? binding.Value.Kind == Gen5PixelOutputKind.Float ? "1.0f" : $"{componentType}(1)"
                        : $"{componentType}(0)";
                    continue;
                }

                if ((export.EnableMask & (1u << source)) == 0)
                {
                    var outputComponent = (uint)component;
                    if (!binding.Value.ComponentMapping.IsIdentity)
                    {
                        for (var physicalComponent = 0u; physicalComponent < 4; physicalComponent++)
                        {
                            if (binding.Value.ComponentMapping.Map(physicalComponent) == component)
                            {
                                outputComponent = physicalComponent;
                                break;
                            }
                        }
                    }

                    values[component] = $"{field}[{outputComponent}]";
                    continue;
                }

                values[component] = IsHalfExportFormat(format)
                    ? HalfExportValue(format, binding.Value.Kind, $"v[{instruction.Sources[source >> 1].Value}]", source & 1)
                    : binding.Value.Kind switch
                    {
                        Gen5PixelOutputKind.Uint => $"v[{instruction.Sources[source].Value}]",
                        Gen5PixelOutputKind.Sint => $"as_type<int>(v[{instruction.Sources[source].Value}])",
                        _ => $"as_type<float>(v[{instruction.Sources[source].Value}])",
                    };
            }

            if (binding.Value.ComponentMapping != Gen5ColorComponentMapping.Identity)
            {
                var mappedValues = new string[4];
                for (var component = 0; component < mappedValues.Length; component++)
                {
                    mappedValues[component] = values[
                        binding.Value.ComponentMapping.Map((uint)component)];
                }

                values = mappedValues;
            }

            // A lane removed from EXEC keeps the previous output value; killed
            // fragments are discarded in the epilogue.
            var value = $"vec<{componentType}, 4>({values[0]}, {values[1]}, {values[2]}, {values[3]})";
            if (binding.Value.ClampToUnit && binding.Value.Kind == Gen5PixelOutputKind.Float)
            {
                value = $"saturate({value})";
            }

            Line($"{field} = {ExecSelect(value, field)};");
            return true;
        }

        // The MRTZ layout: SPI_SHADER_Z_FORMAT, or the export's own when the registers name
        // none or the format disagrees with COMPR.
        private Gen5ColorExportFormat MrtzFormat(Gen5ExportControl export) =>
            _mrtzFormat is { } format &&
            (format == Gen5ColorExportFormat.Zero || export.Compressed == IsHalfExportFormat(format))
                ? format
                : export.Compressed ? Gen5ColorExportFormat.Uint16Abgr : Gen5ColorExportFormat.Abgr32;

        // Mesa's ac_get_spi_shader_z_format and its MRTZ export: the 32-bit formats carry depth
        // in X, stencil in Y and the sample mask in Z; UINT16_ABGR (compressed, no depth)
        // carries stencil in X[23:16] and the sample mask in Y[15:0].
        private string? MrtzDepth(Gen5ShaderInstruction instruction) =>
            instruction.Control is Gen5ExportControl export && (export.EnableMask & 1u) != 0 &&
            MrtzFormat(export) is Gen5ColorExportFormat.R32 or Gen5ColorExportFormat.GR32 or
                Gen5ColorExportFormat.AR32 or Gen5ColorExportFormat.Abgr32
                ? $"as_type<float>(v[{instruction.Sources[0].Value}])"
                : null;

        // The stencil reference is the low 8 bits, the width of every stencil attachment.
        private string? MrtzStencil(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5ExportControl export)
            {
                return null;
            }

            var format = MrtzFormat(export);
            if (IsHalfExportFormat(format))
            {
                return (export.EnableMask & 0x3u) != 0
                    ? $"((v[{instruction.Sources[0].Value}] >> 16) & 0xFFu)"
                    : null;
            }

            return format is Gen5ColorExportFormat.GR32 or Gen5ColorExportFormat.Abgr32 && (export.EnableMask & 0x2u) != 0
                ? $"(v[{instruction.Sources[1].Value}] & 0xFFu)"
                : null;
        }

        private string? MrtzSampleMask(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5ExportControl export)
            {
                return null;
            }

            var format = MrtzFormat(export);
            if (IsHalfExportFormat(format))
            {
                return (export.EnableMask & 0xCu) != 0
                    ? $"(v[{instruction.Sources[1].Value}] & 0xFFFFu)"
                    : null;
            }

            return format == Gen5ColorExportFormat.Abgr32 && (export.EnableMask & 0x4u) != 0
                ? $"v[{instruction.Sources[2].Value}]"
                : null;
        }

        private static bool IsHalfExportFormat(Gen5ColorExportFormat format) =>
            format is >= Gen5ColorExportFormat.Fp16Abgr and <= Gen5ColorExportFormat.Sint16Abgr;

        // SPI_SHADER_COL_FORMAT for the binding's export. Without a format from the registers,
        // or when it disagrees with COMPR, the export's own layout decides: four 32-bit
        // channels, or two 16-bit channels a VGPR read as the target's kind.
        private static Gen5ColorExportFormat ColorExportFormat(Gen5PixelOutputBinding binding, Gen5ExportControl export)
        {
            if (binding.ExportFormat is { } format &&
                (format == Gen5ColorExportFormat.Zero || export.Compressed == IsHalfExportFormat(format)))
            {
                return format;
            }

            return !export.Compressed
                ? Gen5ColorExportFormat.Abgr32
                : binding.Kind switch
                {
                    Gen5PixelOutputKind.Uint => Gen5ColorExportFormat.Uint16Abgr,
                    Gen5PixelOutputKind.Sint => Gen5ColorExportFormat.Sint16Abgr,
                    _ => Gen5ColorExportFormat.Fp16Abgr,
                };
        }

        // The export channel (EN bit) that carries an RGBA channel, or -1. A 16-bit format packs
        // channel c into VGPR c / 2, half c % 2. GFX10 32_AR puts alpha in channel 1, as Mesa's
        // emit_ps_color_export, ACO and LLPC write it (before GFX10 it was channel 3).
        private static int ColorExportSource(Gen5ColorExportFormat format, int component) => format switch
        {
            Gen5ColorExportFormat.R32 => component == 0 ? 0 : -1,
            Gen5ColorExportFormat.GR32 => component < 2 ? component : -1,
            Gen5ColorExportFormat.AR32 => component switch { 0 => 0, 3 => 1, _ => -1 },
            _ => component,
        };

        // One 16-bit channel of a compressed export, converted to the target's kind: the CB
        // reads FP16 as half, UNORM16/SNORM16 as normalized, UINT16/SINT16 as integers.
        private static string HalfExportValue(Gen5ColorExportFormat format, Gen5PixelOutputKind kind, string packed, int half)
        {
            var unsigned = $"as_type<ushort2>({packed})[{half}]";
            var signed = $"as_type<short2>({packed})[{half}]";
            return (format, kind) switch
            {
                (Gen5ColorExportFormat.Sint16Abgr, Gen5PixelOutputKind.Float) => $"float({signed})",
                (Gen5ColorExportFormat.Sint16Abgr, Gen5PixelOutputKind.Uint) => $"uint(int({signed}))",
                (_, Gen5PixelOutputKind.Uint) => $"uint({unsigned})",
                (Gen5ColorExportFormat.Uint16Abgr, Gen5PixelOutputKind.Sint) => $"int({unsigned})",
                (_, Gen5PixelOutputKind.Sint) => $"int({signed})",
                (Gen5ColorExportFormat.Unorm16Abgr, _) => $"(float({unsigned}) * (1.0f / 65535.0f))",
                (Gen5ColorExportFormat.Snorm16Abgr, _) => $"max(float({signed}) * (1.0f / 32767.0f), -1.0f)",
                (Gen5ColorExportFormat.Uint16Abgr, _) => $"float({unsigned})",
                _ => $"(float)as_type<half2>({packed})[{half}]",
            };
        }

        private bool TryEmitVertexExport(
            Gen5ShaderInstruction instruction,
            Gen5ExportControl export)
        {
            // Target 12 is POS0; 13..15 are POS1..POS3 (point size, layer, viewport
            // index, clip and cull distances); 32..63 are the param outputs. MRTZ is
            // ignored like the SPIR-V side.
            if (export.Target is >= 13 and < 16)
            {
                EmitAuxPositionExport(instruction, export);
                return true;
            }

            string field;
            if (export.Target == 12)
            {
                field = "sharpemu_out.sharpemu_position";
            }
            else if (export.Target is >= 32 and < 64 &&
                     _vertexOutputs.Contains(export.Target - 32))
            {
                field = $"sharpemu_out.param{export.Target - 32}";
            }
            else
            {
                return true;
            }

            var values = new string[4];
            for (var component = 0; component < 4; component++)
            {
                if ((export.EnableMask & (1u << component)) == 0)
                {
                    values[component] = component == 3 ? "1.0f" : "0.0f";
                    continue;
                }

                if (export.Compressed)
                {
                    var packed = $"v[{instruction.Sources[component >> 1].Value}]";
                    values[component] = $"(float)as_type<half2>({packed})[{component & 1}]";
                    continue;
                }

                values[component] = $"as_type<float>(v[{instruction.Sources[component].Value}])";
            }

            var exported = $"float4({values[0]}, {values[1]}, {values[2]}, {values[3]})";
            if (export.Target == 12)
            {
                exported = ToHostClipSpace(exported);
            }

            Line($"{field} = {ExecSelect(exported, field)};");
            if (export.Target == 12)
            {
                EmitInvalidPositionDistance();
            }
            else
            {
                EmitInterpolantExports(export.Target - 32, field);
            }

            return true;
        }

        /// <summary>
        /// Vertex attribute fetch: the evaluator captured this buffer load as a
        /// fixed-function vertex input, so read the stage_in field instead of
        /// guest memory (bound via MTLVertexDescriptor by the backend).
        /// </summary>
        private bool TryEmitVertexInputFetch(
            Gen5BufferMemoryControl control,
            ShaderVertexInput input,
            out string error)
        {
            error = string.Empty;
            if (control.DwordCount == 0)
            {
                error =
                    $"invalid vertex input fetch components={control.DwordCount}";
                return false;
            }

            for (uint component = 0; component < control.DwordCount; component++)
            {
                var selector = (input.DestinationSelect >> (int)(component * 3)) & 0x7u;
                if (selector == 0)
                {
                    StoreVector(control.VectorData + component, "0u");
                }
                else if (selector == 1)
                {
                    StoreVector(
                        control.VectorData + component,
                        input.NumberFormat is 4u or 5u ? "1u" : "0x3f800000u");
                }
                else if (selector is >= 4u and <= 7u)
                {
                    var sourceComponent = selector - 4u;
                    if (sourceComponent >= input.ComponentCount)
                    {
                        error =
                            $"vertex input destination selector exceeds source components: selector={selector} components={input.ComponentCount}";
                        return false;
                    }

                    var value = input.ComponentCount == 1
                        ? $"sharpemu_vin.in{input.Location}"
                        : $"sharpemu_vin.in{input.Location}[{sourceComponent}]";
                    StoreVector(control.VectorData + component, AsUInt(value));
                }
                else
                {
                    error = $"unsupported vertex input destination selector={selector}";
                    return false;
                }
            }

            return true;
        }

        // ---- interpolation / pixel inputs ----

        // V_INTERP_MOV's selector picks one of the triangle's parameters: P0 is the
        // provoking vertex's value and P10/P20 the two deltas from it. Reading P0
        // alone is exactly what a flat input gives, so an attribute read only that
        // way is declared flat. The deltas need the three vertex values, which
        // vertex_value inputs give (see Gen5MslTranslator.Interpolation.cs); a host
        // without them keeps ordinary interpolation for the remaining attributes --
        // the same fallback the SPIR-V translator takes on a device without
        // fragmentShaderBarycentric.
        private bool ReadsOnlyTheProvokingVertex(uint attribute)
        {
            var read = false;
            foreach (var instruction in _request.Program.Instructions)
            {
                if (instruction.Control is not Gen5InterpolationControl control ||
                    control.Attribute != attribute)
                {
                    continue;
                }

                if (instruction.Opcode != "VInterpMovF32" ||
                    control.Parameter != ProvokingVertexParameter)
                {
                    return false;
                }

                read = true;
            }

            return read;
        }

        private const uint ProvokingVertexParameter = 2;

        private bool TryEmitInterpolation(
            Gen5ShaderInstruction instruction,
            Gen5InterpolationControl interpolation,
            out string error)
        {
            error = string.Empty;
            if (_stage != Gen5MslStage.Pixel ||
                !_pixelAttributes.Contains(interpolation.Attribute) ||
                instruction.Destinations.Count == 0 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.VectorRegister)
            {
                error = "invalid interpolated attribute";
                return false;
            }

            var destination = instruction.Destinations[0].Value;
            if (_perVertexAttributes.Contains(interpolation.Attribute))
            {
                return TryEmitInterpolationParameter(instruction, interpolation, destination, out error);
            }

            if (IsManuallyInterpolated(interpolation.Attribute))
            {
                // Only feeds a hand interpolation whose result the host's replaces.
                StoreVector(destination, AsUInt($"{PixelInputField(interpolation.Attribute)}[{interpolation.Channel}]"));
                return true;
            }

            if (IsHalfInterpolation(instruction.Opcode) ||
                interpolation.Modifiers is { } modifiers && ((modifiers.AbsoluteMask | modifiers.NegateMask) & 0x6u) != 0)
            {
                // Fixed-function interpolation cannot split a 32-bit parameter into halves or
                // weight it with modified barycentrics.
                error = $"{instruction.Opcode} of attribute {interpolation.Attribute} needs per-vertex pixel inputs";
                return false;
            }

            if (_perVertexSourcedAttributes.TryGetValue(interpolation.Attribute, out var flatSource))
            {
                EmitPerVertexSourcedInterpolation(interpolation, destination, flatSource);
                return true;
            }

            StoreVector(
                destination,
                AsUInt(ApplyInterpolationOutputModifiers(
                    interpolation, $"{PixelInputField(interpolation.Attribute)}[{interpolation.Channel}]")));
            return true;
        }

        /// <summary>
        /// Seeds pixel input VGPRs in SPI_PS_INPUT_ADDR compact order: the
        /// interpolation slots reserve registers even though V_INTERP reads MSL
        /// varyings directly (they carry the barycentrics when the program
        /// interpolates per-vertex inputs by hand), and the position and system
        /// inputs land in the hardware-selected VGPRs.
        /// </summary>
        private void EmitPixelInputState(StringBuilder source)
        {
            EmitBarycentricReconstruction(source);
            uint vgpr = 0;

            void Advance(int bit, uint dwordCount)
            {
                if ((_pixelInputAddress & (1u << bit)) == 0)
                {
                    return;
                }

                if (BarycentricInputRegisters(bit) is { } registers)
                {
                    for (var index = 0; index < registers.Length; index++)
                    {
                        source.AppendLine($"    v[{vgpr + (uint)index}] = as_type<uint>({registers[index]});");
                    }
                }

                vgpr += dwordCount;
            }

            void Input(int bit, string? value)
            {
                var mask = 1u << bit;
                if ((_pixelInputAddress & mask) == 0)
                {
                    return;
                }

                if ((_pixelInputEnable & mask) != 0 && value is not null)
                {
                    source.AppendLine($"    v[{vgpr}] = {value};");
                }

                vgpr++;
            }

            const string fragCoord = "sharpemu_in.sharpemu_frag_coord";
            Advance(0, 2);  // PERSP_SAMPLE
            Advance(1, 2);  // PERSP_CENTER
            Advance(2, 2);  // PERSP_CENTROID
            Advance(3, 3);  // PERSP_PULL_MODEL
            Advance(4, 2);  // LINEAR_SAMPLE
            Advance(5, 2);  // LINEAR_CENTER
            Advance(6, 2);  // LINEAR_CENTROID
            // LINE_STIPPLE_TEX reads 0: line stipple is not emulated (as AnyPS5).
            Input(7, "0u");
            Input(8, $"as_type<uint>({fragCoord}.x)");
            Input(9, $"as_type<uint>({fragCoord}.y)");
            Input(10, $"as_type<uint>({fragCoord}.z)");
            // POS_W_FLOAT is W; [[position]].w is 1/W, like Vulkan's FragCoord.w.
            Input(11, $"as_type<uint>(1.0f / {fragCoord}.w)");

            // FRONT_FACE, ANCILLARY, SAMPLE_COVERAGE and POS_FIXED_PT follow the
            // position inputs.
            Input(12, PixelSystemInput(12));
            Input(13, PixelSystemInput(13));
            Input(14, PixelSystemInput(14));
            Input(15, PixelSystemInput(15));
        }
    }
}
