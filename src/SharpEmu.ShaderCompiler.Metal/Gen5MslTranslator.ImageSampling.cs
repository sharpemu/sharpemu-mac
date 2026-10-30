// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        // Where a sample's, gather's or LOD query's address operands sit, laid out by the
        // instruction's own DIM (RDNA2 ISA 8.2.5): {offset}{bias}{z-compare}{derivatives}{body}.
        // The first four are dword indices, -1 when absent: the offsets, the bias (a half in
        // the low bits under A16, LLVM packImage16bitOpsToDwords), the f32 reference and the
        // derivatives all occupy whole dwords. Body is the first body component in address
        // component units, which are packed halves under A16; the _L LOD or _CL clamp follows
        // the coordinates there.
        private readonly record struct SampleAddress(
            int OffsetDword,
            int BiasDword,
            int ReferenceDword,
            int GradientDword,
            int GradientSetDwords,
            bool G16,
            int Body,
            int Spatial,
            int Coordinates)
        {
            /// <summary>The body component after the coordinates: the _L LOD or the _CL clamp.</summary>
            public int Trailing => Body + Coordinates;
        }

        private static SampleAddress SampleAddressOf(Gen5ImageControl image, Gen5ImageOp flags)
        {
            // Spatial axes and body coordinates (axes plus a slice or face) per MIMG DIM.
            var (spatial, coordinates) = image.Dimension switch
            {
                0 => (1, 1),
                2 => (3, 3),
                3 => (2, 3),
                4 => (1, 2),
                5 or 7 => (2, 3),
                _ => (2, 2),
            };
            var dword = 0;
            var offset = (flags & Gen5ImageOp.Offset) != 0 ? dword++ : -1;
            var bias = (flags & Gen5ImageOp.Bias) != 0 ? dword++ : -1;
            var reference = (flags & Gen5ImageOp.Compare) != 0 ? dword++ : -1;

            // f32 derivatives take a dword each, d/dh for every axis then d/dv. _G16 packs each
            // set into halves padded to whole dwords (LLVM packImage16bitOpsToDwords):
            // 1D dx/dh; dx/dv. 2D dy/dh,dx/dh; dy/dv,dx/dv. 3D dy/dh,dx/dh; -,dz/dh; dy/dv,dx/dv; -,dz/dv.
            var g16 = (flags & Gen5ImageOp.G16) != 0;
            var setDwords = g16 ? (spatial + 1) / 2 : spatial;
            var gradients = -1;
            if ((flags & Gen5ImageOp.Derivatives) != 0)
            {
                gradients = dword;
                dword += 2 * setDwords;
            }

            return new SampleAddress(offset, bias, reference, gradients, setDwords, g16, image.A16 ? dword * 2 : dword, spatial, coordinates);
        }

        private static string SampleAddressDword(Gen5ImageControl image, int dword) =>
            $"v[{image.GetAddressRegister(dword)}]";

        private static string SampleBias(Gen5ImageControl image, SampleAddress address) =>
            image.A16
                ? $"(float)as_type<half2>({SampleAddressDword(image, address.BiasDword)})[0]"
                : $"as_type<float>({SampleAddressDword(image, address.BiasDword)})";

        private static string SampleReference(Gen5ImageControl image, SampleAddress address) =>
            $"as_type<float>({SampleAddressDword(image, address.ReferenceDword)})";

        // One derivative: set 0 is d/dh, set 1 is d/dv. An axis the instruction lacks is 0.
        private static string SampleGradient(Gen5ImageControl image, SampleAddress address, int set, int component)
        {
            if (component >= address.Spatial)
            {
                return "0.0f";
            }

            var first = address.GradientDword + (set * address.GradientSetDwords);
            return address.G16
                ? $"(float)as_type<half2>({SampleAddressDword(image, first + (component / 2))})[{component & 1}]"
                : $"as_type<float>({SampleAddressDword(image, first + component)})";
        }

        // The host texture's spatial coordinates from the instruction's body. A host axis the
        // instruction does not address reads 0, and an axis the host lacks is ignored: the T#
        // decides the addressing, the DIM only how many VGPRs are read (a 2D DIM through a 1D
        // T# reads x, a 3D DIM through a 2D T# reads u and v; AnyPS5's RDNA2 measurements).
        //
        // Cube bodies carry face coordinates in [1, 2] that the host 2D-array view of the
        // faces samples at (s - 1, t - 1), whichever DIM addresses the cube.
        private string SampleCoordinates(Gen5ImageControl image, SampleAddress address, MslImageClass imageClass)
        {
            var count = imageClass.SpatialComponents;
            var parts = new string[count];
            for (var component = 0; component < count; component++)
            {
                var value = component < address.Spatial ? ImageFloatAddress(image, address.Body + component) : "0.0f";
                parts[component] = imageClass.Cube && component < 2 ? $"({value}) - 1.0f" : value;
            }

            return count == 1
                ? Temp("float", parts[0])
                : Temp(imageClass.FloatCoordinateType, $"{imageClass.FloatCoordinateType}({string.Join(", ", parts)})");
        }

        // The host layer of an arrayed view comes from the body component after the host's own
        // axes when the instruction has one there: an array slice, a cube face, or the r of a 3D
        // DIM through a 2D-array T# (which RDNA2 reads as the layer). Without it the access
        // reads the view's first layer, as a 2D DIM through a cube T# reads its BASE_ARRAY face.
        private string SampleLayerArgument(Gen5ImageControl image, SampleAddress address, MslImageClass imageClass, string texture)
        {
            if (!imageClass.Arrayed)
            {
                return string.Empty;
            }

            return imageClass.SpatialComponents < address.Coordinates
                ? $", {BuildSampleLayer(image, address.Body + imageClass.SpatialComponents, imageClass, texture)}"
                : ", 0u";
        }

        // The _O offsets: one dword of six-bit signed texel offsets, x [5:0], y [13:8], z [21:16],
        // as an int2 (a 1D image is a height-one 2D texture here) or int3. Cube maps take none:
        // the offsets of a cube access are ignored, as Kyty's translator does.
        private string? SampleOffsets(Gen5ImageControl image, SampleAddress address, MslImageClass imageClass)
        {
            if (address.OffsetDword < 0 || imageClass.Cube)
            {
                return null;
            }

            var packed = Temp("int", $"as_type<int>({SampleAddressDword(image, address.OffsetDword)})");
            var axes = Math.Max(imageClass.SpatialComponents, 2);
            var parts = Enumerable.Range(0, axes).Select(component =>
                component < imageClass.SpatialComponents && component < address.Spatial
                    ? $"extract_bits({packed}, {component * 8}u, 6u)"
                    : "0");
            return Temp($"int{axes}", $"int{axes}({string.Join(", ", parts)})");
        }

        // Metal applies a texel offset only within -8..7 (outside it the result is wrong); the
        // guest's six-bit offsets reach -32..31. An offset outside the native range moves the
        // normalized coordinate by offset / extent of `level` instead and the native offset is
        // zero. That is exact for a read of one level, which every gather is; a sample blending
        // two levels moves the coarser one by half as much as the hardware does.
        private (string Coordinates, string NativeOffset) SplitSampleOffset(
            string coordinates,
            string offset,
            MslImageClass imageClass,
            string texture,
            string level)
        {
            var axes = Math.Max(imageClass.SpatialComponents, 2);
            var offsetType = $"int{axes}";
            var inRange = Temp("bool", $"all({offset} >= {offsetType}(-8)) && all({offset} <= {offsetType}(7))");
            var native = Temp(offsetType, $"{inRange} ? {offset} : {offsetType}(0)");
            var count = imageClass.SpatialComponents;
            string moved;
            if (count == 1)
            {
                moved = $"{coordinates} + (float){offset}.x / (float){ImageExtentQuery(texture, imageClass, 0, level)}";
            }
            else
            {
                var extents = string.Join(
                    ", ",
                    Enumerable.Range(0, count).Select(component => $"(float){ImageExtentQuery(texture, imageClass, component, level)}"));
                moved = $"{coordinates} + {imageClass.FloatCoordinateType}({offset}) / {imageClass.FloatCoordinateType}({extents})";
            }

            return (Temp(imageClass.FloatCoordinateType, $"{inRange} ? {coordinates} : {moved}"), native);
        }

        // A mip index from a float level: nearest, inside the view's levels.
        // MIRROR_ONCE_BORDER (S# clamp 7) has no host address mode. The host samples
        // mirror-once-to-edge, which matches it inside the mirrored copy, and texel index
        // floor(u * size) leaves that copy, [-size, size - 1], exactly when the normalized
        // coordinate is below -1 or at least 1, at every level; the S# border colour is
        // substituted there. That is exact only where each result is one texel: the host
        // sampler refuses filtering across the edge, and offsets (which move the edge per
        // level), compares, cubes and register-table border colours are refused here.
        private bool TryMirrorOnceBorderTest(
            string opcode, SamplerResource? samplerInfo, MslImageClass imageClass, string coordinates, bool offsetOrCompare,
            out string? pastMirror, out string error)
        {
            pastMirror = null;
            error = string.Empty;
            var spatialComponents = imageClass.SpatialComponents;
            var axes = (samplerInfo?.MirrorOnceBorderAxes ?? 0) & ((1u << spatialComponents) - 1);
            if (axes == 0)
            {
                return true;
            }

            if (imageClass.Cube || offsetOrCompare || imageClass.DepthCompare || samplerInfo!.BorderColorType == 3)
            {
                // Documented deviation: past the mirrored copy these read the edge texel
                // (mirror-once-to-edge) instead of the border colour.
                ShaderCompilerWarnings.WarnOnce(
                    "mirror-once-border-shape",
                    $"{opcode} through a MIRROR_ONCE_BORDER sampler with a cube, an offset, a compare or a border colour " +
                    "table reads the edge texel past the mirrored copy instead of the border colour.");
                return true;
            }

            var tests = new List<string>(3);
            for (var axis = 0; axis < spatialComponents; axis++)
            {
                if ((axes & (1u << axis)) != 0)
                {
                    var component = spatialComponents == 1 ? coordinates : $"{coordinates}.{"xyz"[axis]}";
                    tests.Add($"{component} < -1.0f || {component} >= 1.0f");
                }
            }

            pastMirror = Temp("bool", string.Join(" || ", tests));
            return true;
        }

        // The S# BORDER_COLOR_TYPE constant colours: transparent black, opaque black, opaque white.
        private static string BorderColour(string kind, uint type)
        {
            var one = kind == "float" ? "1.0f" : "1";
            var zero = kind == "float" ? "0.0f" : "0";
            return type switch
            {
                0 => $"{VectorLiteral(kind)}({zero}, {zero}, {zero}, {zero})",
                1 => $"{VectorLiteral(kind)}({zero}, {zero}, {zero}, {one})",
                _ => $"{VectorLiteral(kind)}({one}, {one}, {one}, {one})",
            };
        }

        // An S# LOD_BIAS (signed 6.8 fixed point) as an exact float expression.
        private static string SamplerLodBiasLiteral(int bias) => $"({bias}.0f / 256.0f)";

        // A pixel-stage sample of an image the materializer made eligible
        // (ImageResource.MipStatistics: GPU lod statistics are on and the image is sampled)
        // can count into the GET_LOD_STATS bank. Whether it does is the bound T#'s
        // MIP_STATS_ENABLE, runtime state in the image's shader-data field, so the program is
        // the same either way. Storage, multisampled and cube views are left out, as the
        // SPIR-V backend leaves them.
        private bool RecordsMipStatistics(int resourceIndex, MslImageClass imageClass) =>
            _hasMipStatistics && _stage == Gen5MslStage.Pixel &&
            resourceIndex >= 0 && resourceIndex < _request.Resources.Info.Images.Count &&
            _request.Resources.Info.Images[resourceIndex].MipStatistics &&
            !imageClass.IsStorage && !imageClass.Multisampled && !imageClass.Cube;

        // The requested LOD, relative to the view, goes to the counter of the image's field
        // (MipStatistics.ImageField, which the host writes into the shader data). A field with
        // bit 15 set (MipStatistics.NoCounterField: the bound T# has MIP_STATS_ENABLE clear)
        // skips everything, the LOD computation included: one shader-data load and a branch
        // uniform across the draw. requestedLod emits that computation, inside the branch.
        // The whole site is also behind a function constant, which the host sets once the title
        // uses LOD statistics: until then the pipeline has no site at all.
        // Only lanes in EXEC that are not helper threads count (Templates/mip_stats_prelude.msl).
        private void RecordMipStatistics(int resourceIndex, Func<string> requestedLod)
        {
            _usesMipStatistics = true;
            if (UsesMipStatisticsConstant)
            {
                Line($"if ({Gen5MslShader.MipStatisticsFunctionConstantName})");
                Line("{");
                _indent++;
            }

            var field = Temp("uint", ShaderDataDword(_request.Bindings.MipStatisticsDword + (uint)resourceIndex));
            Line($"if (({field} & 0x8000u) == 0u)");
            Line("{");
            _indent++;
            var lod = requestedLod();
            Line($"sharpemu_mip_stats({ResourcesName}.{Gen5MslArgumentLayout.MipStatisticsName}, {field}, {lod}, " +
                 $"{ExecCondition("!simd_is_helper_thread()")});");
            _indent--;
            Line("}");
            if (UsesMipStatisticsConstant)
            {
                _indent--;
                Line("}");
            }
        }

        // The isotropic LOD of explicit derivatives: log2 of the longer screen-axis footprint
        // in texels of the view's first level.
        private string GradientLod(string texture, MslImageClass imageClass, string gradientX, string gradientY)
        {
            var scalar = imageClass.SpatialComponents == 1;
            var type = scalar ? "float" : imageClass.FloatCoordinateType;
            var size = imageClass.SpatialComponents switch
            {
                1 => $"(float){texture}.get_width()",
                2 => $"float2({texture}.get_width(), {texture}.get_height())",
                _ => $"float3({texture}.get_width(), {texture}.get_height(), {texture}.get_depth())",
            };
            var x = Temp(type, $"{gradientX} * {size}");
            var y = Temp(type, $"{gradientY} * {size}");
            return scalar
                ? Temp("float", $"0.5f * log2(max({x} * {x}, {y} * {y}))")
                : Temp("float", $"0.5f * log2(max(dot({x}, {x}), dot({y}, {y})))");
        }

        private string NearestMip(string level, string texture) =>
            Temp("uint", $"min((uint)max(rint({level}), 0.0f), {texture}.get_num_mip_levels() - 1u)");

        // The guest S# DEPTH_COMPARE_FUNC (word 0 bits 14:12), reference OP texel as Vulkan
        // and Metal compare: never, less, equal, less-equal, greater, not-equal, greater-equal,
        // always. Not-equal is unordered, so a NaN on either side passes it.
        private static string CompareExpression(int function, string reference, string texel) => function switch
        {
            0 => "false",
            1 => $"{reference} < {texel}",
            2 => $"{reference} == {texel}",
            3 => $"{reference} <= {texel}",
            4 => $"{reference} > {texel}",
            5 => $"{reference} != {texel}",
            6 => $"{reference} >= {texel}",
            7 => "true",
            _ => throw new ArgumentOutOfRangeException(nameof(function), function, "a depth compare function is three bits"),
        };

        // The component a gather returns: DMASK has one bit set, naming it.
        private static uint GatherComponent(uint dmask)
        {
            uint component = 0;
            while (component < 3 && (dmask & (1u << (int)component)) == 0)
            {
                component++;
            }

            return component;
        }

        private bool TryEmitImageSample(
            Gen5ShaderInstruction instruction,
            Gen5ImageControl image,
            string texture,
            string samplerName,
            MslImageClass imageClass,
            int compareFunction,
            SamplerResource? samplerInfo,
            int resourceIndex,
            out string sampled,
            out string error)
        {
            sampled = string.Empty;
            error = string.Empty;
            var kind = imageClass.ComponentKind;
            var opcode = instruction.Opcode;
            var flags = Gen5ImageOpcodes.FlagsOf(opcode);
            var address = SampleAddressOf(image, flags);
            var hasCompare = (flags & Gen5ImageOp.Compare) != 0;
            var hasGradients = (flags & Gen5ImageOp.Derivatives) != 0;
            var hasZeroLod = (flags & Gen5ImageOp.LodZero) != 0;
            var hasLod = (flags & Gen5ImageOp.Lod) != 0;
            var hasBias = (flags & Gen5ImageOp.Bias) != 0;
            var hasClamp = (flags & Gen5ImageOp.Clamp) != 0;
            var spatialComponents = imageClass.SpatialComponents;
            var unorm = image.Unorm;
            if (unorm && imageClass.Cube)
            {
                // Cube face coordinates are normalized [1, 2] values; what UNRM does to them
                // is not documented or measured.
                error = $"unnormalized-coordinate cube sampling is not supported for {opcode}";
                return false;
            }

            if (hasCompare && !imageClass.DepthCompare && compareFunction < 0)
            {
                error = $"{opcode} compares a color texture without its sampler's compare function";
                return false;
            }

            var offset = SampleOffsets(image, address, imageClass);
            var bias = hasBias ? Temp("float", SampleBias(image, address)) : "0.0f";
            var reference = hasCompare ? Temp("float", SampleReference(image, address)) : "0.0f";

            // The S# LOD_BIAS adds to every LOD the sampler computes or is given (_L), before the
            // _CL and MIN/MAX_LOD clamps; _LZ forces level 0 and unnormalized coordinates force
            // mip 0, so neither is biased.
            var samplerLodBias = samplerInfo?.LodBias ?? 0;
            var samplerBias = samplerLodBias != 0 && !unorm && !hasZeroLod ? SamplerLodBiasLiteral(samplerLodBias) : null;
            if (samplerBias is not null && !hasGradients && !hasLod && _stage == Gen5MslStage.Pixel)
            {
                bias = hasBias ? Temp("float", $"{bias} + {samplerBias}") : samplerBias;
                hasBias = true;
            }

            // Derivatives are one host-spatial vector per screen axis, with no array layer and
            // no cube bias: they have no coordinate origin.
            var gradientType = spatialComponents == 3 ? "gradient3d" : "gradient2d";
            var gradientX = string.Empty;
            var gradientY = string.Empty;
            if (hasGradients)
            {
                string Gradients(int set)
                {
                    var parts = Enumerable.Range(0, spatialComponents)
                        .Select(component => SampleGradient(image, address, set, component))
                        .ToArray();
                    return spatialComponents == 1
                        ? Temp("float", parts[0])
                        : Temp(imageClass.FloatCoordinateType, $"{imageClass.FloatCoordinateType}({string.Join(", ", parts)})");
                }

                gradientX = Gradients(0);
                gradientY = Gradients(1);
                if (samplerBias is not null)
                {
                    // Scaling both derivatives by 2^bias raises the LOD they give by the bias
                    // and leaves the anisotropy ratio alone; a gradient sample takes no bias.
                    var scale = Temp("float", $"metal::exp2({samplerBias})");
                    gradientX = Temp(spatialComponents == 1 ? "float" : imageClass.FloatCoordinateType, $"{gradientX} * {scale}");
                    gradientY = Temp(spatialComponents == 1 ? "float" : imageClass.FloatCoordinateType, $"{gradientY} * {scale}");
                }
            }

            var coordinates = SampleCoordinates(image, address, imageClass);
            var layerArgument = SampleLayerArgument(image, address, imageClass, texture);
            // Implicit-LOD and bias sampling need the derivatives a fragment quad provides,
            // so Metal only offers them in a fragment function. Outside one the level is
            // explicit, which is what the SPIR-V backend does for the same reason.
            // Unnormalized coordinates address mip 0 only: the level is forced to zero.
            var useGradients = hasGradients && !unorm;
            var explicitLod = unorm || hasGradients || hasZeroLod || hasLod || _stage != Gen5MslStage.Pixel;
            var lod = hasZeroLod || unorm
                ? "0.0f"
                : hasLod
                    ? Temp("float", samplerBias is null
                        ? ImageFloatAddress(image, address.Trailing)
                        : $"{ImageFloatAddress(image, address.Trailing)} + {samplerBias}")
                    : explicitLod
                        ? "0.0f"
                        : bias;
            // The LOD the instruction asks for, before the _CL clamp raises it.
            var givenLod = hasZeroLod || unorm || hasLod ? lod : null;
            string? clamp = null;
            if (hasClamp)
            {
                clamp = Temp("float", ImageFloatAddress(image, address.Trailing));
                if (explicitLod && !useGradients)
                {
                    lod = Temp("float", $"max({lod}, {clamp})");
                }
            }

            if (RecordsMipStatistics(resourceIndex, imageClass))
            {
                var planar = spatialComponents == 1 ? $"float2({coordinates}, 0.5f)" : coordinates;
                RecordMipStatistics(resourceIndex, () => givenLod ?? (useGradients
                    ? GradientLod(texture, imageClass, gradientX, gradientY)
                    : Temp("float", $"{texture}.calculate_unclamped_lod({samplerName}, {planar}) + {bias}")));
            }

            if (unorm)
            {
                coordinates = NormalizeUnnormalizedCoordinates(coordinates, texture, imageClass);
            }

            if (!TryMirrorOnceBorderTest(opcode, samplerInfo, imageClass, coordinates, offset is not null || hasCompare, out var pastMirror, out error))
            {
                return false;
            }

            if (samplerInfo is { Reduction: not 0 } reducing && kind == "float")
            {
                if (hasCompare || imageClass.DepthCompare || imageClass.Cube || imageClass.Dimension == ImageDimension.Dim3D ||
                    pastMirror is not null)
                {
                    ShaderCompilerWarnings.WarnOnce(
                        $"sampler-reduction:{opcode}:{imageClass.Dimension}:{imageClass.Cube}",
                        $"{opcode} through a min/max filter-mode sampler of a {(imageClass.Cube ? "cube" : imageClass.Dimension)} image " +
                        $"{(hasCompare ? "with a compare " : string.Empty)}is filtered as a blend.");
                }
                else
                {
                    // The level the hardware filters at: the explicit one, the one the derivatives give, or in a fragment
                    // function the one the sampler computes; raised to the _CL clamp.
                    var level = explicitLod && !useGradients
                        ? lod
                        : useGradients
                            ? GradientLod(texture, imageClass, gradientX, gradientY)
                            : _stage == Gen5MslStage.Pixel
                                ? Temp("float", $"{texture}.calculate_unclamped_lod({samplerName}, {(spatialComponents == 1 ? $"float2({coordinates}, 0.5f)" : coordinates)}) + {bias}")
                                : "0.0f";
                    if (clamp is not null && !(explicitLod && !useGradients))
                    {
                        level = Temp("float", $"max({level}, {clamp})");
                    }

                    var planar = spatialComponents == 1 ? Temp("float2", $"float2({coordinates}, 0.5f)") : coordinates;
                    sampled = EmitReducedSample(texture, samplerName, imageClass, planar, layerArgument, level, offset, reducing);
                    sampled = ForceDegamma(sampled, samplerInfo, kind);
                    return true;
                }
            }

            var offsetArgument = string.Empty;
            if (offset is not null)
            {
                // The level an out-of-range offset is scaled by: the explicit one, or in a
                // fragment function the one the sampler picks for the coordinates.
                var foldLevel = explicitLod && !useGradients
                    ? lod
                    : _stage == Gen5MslStage.Pixel && imageClass.SupportsSampleLod
                        ? Temp("float", $"{texture}.calculate_clamped_lod({samplerName}, {(spatialComponents == 1 ? $"float2({coordinates}, 0.5f)" : coordinates)})")
                        : "0.0f";
                (coordinates, var nativeOffset) = SplitSampleOffset(
                    coordinates, offset, imageClass, texture, NearestMip(foldLevel, texture));
                offsetArgument = $", {nativeOffset}";
            }

            if (spatialComponents == 1)
            {
                coordinates = Temp("float2", $"float2({coordinates}, 0.5f)");
                if (useGradients)
                {
                    gradientX = $"float2({gradientX}, 0.0f)";
                    gradientY = $"float2({gradientY}, 0.0f)";
                }
            }

            var clampArgument = clamp is null ? string.Empty : $", metal::min_lod_clamp({clamp})";
            var samplerArguments = !imageClass.SupportsSampleLod
                ? string.Empty
                : useGradients
                    ? $", metal::{gradientType}({gradientX}, {gradientY}){clampArgument}"
                    : explicitLod
                        ? $", metal::level({lod})"
                        : hasBias
                            ? $", metal::bias({bias}){clampArgument}"
                            : clampArgument;
            if (imageClass.DepthCompare)
            {
                // Native depth samplers compare each contributing texel before filtering,
                // with the compare function of the S# the host sampler was built from.
                var comparison = hasCompare
                    ? Temp("float", $"{texture}.sample_compare({samplerName}, {coordinates}{layerArgument}, {reference}{samplerArguments}{offsetArgument})")
                    : Temp("float", $"{texture}.sample({samplerName}, {coordinates}{layerArgument}{samplerArguments}{offsetArgument})");
                sampled = Temp("float4", hasCompare
                    ? $"float4({comparison}, {comparison}, {comparison}, 1.0f)"
                    : $"float4({comparison}, 0.0f, 0.0f, 1.0f)");
                return true;
            }

            sampled = Temp(
                $"vec<{kind}, 4>",
                $"{texture}.sample({samplerName}, {coordinates}{layerArgument}{samplerArguments}{offsetArgument})");
            if (pastMirror is not null)
            {
                sampled = Temp($"vec<{kind}, 4>", $"{pastMirror} ? {BorderColour(kind, samplerInfo!.BorderColorType)} : {sampled}");
            }

            sampled = ForceDegamma(sampled, samplerInfo, kind);
            if (hasCompare)
            {
                // A color format has no host depth view: its first channel is compared with
                // the S# function after filtering (exact for point sampling).
                var passes = Temp("bool", CompareExpression(compareFunction, reference, $"(float){sampled}[0]"));
                var one = kind == "float" ? "1.0f" : "1";
                var zero = kind == "float" ? "0.0f" : "0";
                sampled = Temp(
                    $"vec<{kind}, 4>",
                    $"{VectorLiteral(kind)}({passes} ? {one} : {zero}, {passes} ? {one} : {zero}, {passes} ? {one} : {zero}, {one})");
            }

            return true;
        }

        // FORCE_DEGAMMA converts the filtered colour from sRGB to linear whatever the texture's format; alpha is linear
        // already. Per sampler: one texture sampled through two samplers may decode in one only.
        private string ForceDegamma(string sampled, SamplerResource? samplerInfo, string kind)
        {
            if (samplerInfo is not { ForceDegamma: true } || kind != "float")
            {
                return sampled;
            }

            var encoded = Temp("float3", $"metal::max({sampled}.rgb, 0.0f)");
            var linear = Temp("float3",
                $"metal::select(metal::powr(({encoded} + 0.055f) / 1.055f, 2.4f), {encoded} / 12.92f, {encoded} <= 0.04045f)");
            return Temp("vec<float, 4>", $"float4({linear}, {sampled}.a)");
        }

        // A min/max FILTER_MODE sampler returns, per component, the minimum or maximum of the texels its filter weighs
        // rather than their weighted sum. Metal samplers only blend, so each texel of the footprint is read by an
        // explicit-level sample at its centre through the same host sampler (its addressing and border apply, and a
        // linear filter there weighs that texel alone), and the texels of nonzero weight are reduced: the 2x2 of the
        // nearest level, or of the two levels a linear mip filter blends, the second one only when it weighs.
        private string EmitReducedSample(
            string texture,
            string samplerName,
            MslImageClass imageClass,
            string coordinates,
            string layerArgument,
            string level,
            string? offset,
            SamplerResource sampler)
        {
            var result = Temp("vec<float, 4>", sampler.Reduction == 1 ? "float4(INFINITY)" : "float4(-INFINITY)");
            if (!sampler.ReductionAcrossLevels)
            {
                EmitReducedFootprint(result, texture, samplerName, imageClass, coordinates, layerArgument, NearestMip(level, texture), offset, sampler.Reduction, null);
                return result;
            }

            var last = Temp("float", $"(float)({texture}.get_num_mip_levels() - 1u)");
            var clamped = Temp("float", $"metal::clamp({level}, 0.0f, {last})");
            var first = Temp("uint", $"(uint)metal::floor({clamped})");
            var second = Temp("uint", $"metal::min({first} + 1u, {texture}.get_num_mip_levels() - 1u)");
            EmitReducedFootprint(result, texture, samplerName, imageClass, coordinates, layerArgument, first, offset, sampler.Reduction, null);
            EmitReducedFootprint(
                result, texture, samplerName, imageClass, coordinates, layerArgument, second, offset, sampler.Reduction,
                $"{clamped} > (float){first}");
            return result;
        }

        private void EmitReducedFootprint(
            string result,
            string texture,
            string samplerName,
            MslImageClass imageClass,
            string coordinates,
            string layerArgument,
            string mip,
            string? offset,
            uint reduction,
            string? weighs)
        {
            var oneDimensional = imageClass.SpatialComponents == 1;
            var size = Temp(
                "float2",
                oneDimensional
                    ? $"float2((float){ImageExtentQuery(texture, imageClass, 0, mip)}, 1.0f)"
                    : $"float2((float){ImageExtentQuery(texture, imageClass, 0, mip)}, (float){ImageExtentQuery(texture, imageClass, 1, mip)})");
            var position = Temp("float2", $"{coordinates} * {size} - 0.5f");
            var corner = Temp("float2", $"floor({position})");
            var fraction = Temp("float2", $"{position} - {corner}");
            if (offset is not null)
            {
                Line($"{corner} += float2({offset});");
            }

            if (oneDimensional)
            {
                // One row: the second footprint row weighs nothing.
                Line($"{corner}.y = 0.0f;");
                Line($"{fraction}.y = 0.0f;");
            }

            var reduce = reduction == 1 ? "metal::fmin" : "metal::fmax";
            for (var texel = 0; texel < 4; texel++)
            {
                var column = texel & 1;
                var row = texel >> 1;
                // The first column and row always weigh (1 - fraction > 0); the second ones only past the centre.
                var conditions = new List<string>();
                if (weighs is not null)
                {
                    conditions.Add(weighs);
                }

                if (column == 1)
                {
                    conditions.Add($"{fraction}.x > 0.0f");
                }

                if (row == 1)
                {
                    conditions.Add($"{fraction}.y > 0.0f");
                }

                var centre = $"({corner} + float2({column}.0f, {row}.0f) + 0.5f) / {size}";
                var value = $"{texture}.sample({samplerName}, {centre}{layerArgument}, metal::level((float){mip}))";
                Line(conditions.Count == 0
                    ? $"{result} = {reduce}({result}, {value});"
                    : $"if ({string.Join(" && ", conditions)}) {result} = {reduce}({result}, {value});");
            }
        }

        // GATHER4*: the four texels of the 2x2 footprint a bilinear filter would read, one
        // component each, in the order (i0, j1), (i1, j1), (i1, j0), (i0, j0). The hardware
        // gathers from one level: level zero for _LZ, the _L LOD, or the computed LOD plus the
        // _B bias, raised to the _CL clamp, rounded to the nearest level (Kyty's GatherMip
        // models the same nearest-level choice). Metal's gather reads only the base level and
        // offsets inside -8..7, so other levels and offsets read the four texels with explicit
        // level samples at the texel centres, through the same host sampler (its addressing,
        // border and compare state apply). A compare gather tests each texel against the
        // reference with the S# function.
        private bool TryEmitImageGather(
            Gen5ShaderInstruction instruction,
            Gen5ImageControl image,
            string texture,
            string samplerName,
            MslImageClass imageClass,
            int compareFunction,
            SamplerResource? samplerInfo,
            out string sampled,
            out string error)
        {
            sampled = string.Empty;
            error = string.Empty;
            var kind = imageClass.ComponentKind;
            var opcode = instruction.Opcode;
            var flags = Gen5ImageOpcodes.FlagsOf(opcode);
            var address = SampleAddressOf(image, flags);
            var hasCompare = (flags & Gen5ImageOp.Compare) != 0;
            if ((flags & Gen5ImageOp.Horizontal) != 0)
            {
                if (imageClass.Dimension is ImageDimension.Dim1D or ImageDimension.Dim1DArray)
                {
                    error = "unsupported 1D horizontal image gather";
                    return false;
                }

                if (image.Unorm && imageClass.Cube)
                {
                    error = $"unnormalized-coordinate cube gather is not supported for {opcode}";
                    return false;
                }

                sampled = EmitHorizontalGather(image, address, texture, samplerName, imageClass);
                return true;
            }

            // The ISA gathers from 2D, 2D-array and cube images (Table 43); a 1D image is a
            // height-one 2D texture here and gathers from its one row.
            if (imageClass.Dimension == ImageDimension.Dim3D)
            {
                error = $"image dimension {imageClass.Dimension} does not support {opcode} on Metal";
                return false;
            }

            if (image.Unorm && imageClass.Cube)
            {
                error = $"unnormalized-coordinate cube gather is not supported for {opcode}";
                return false;
            }

            if (hasCompare && !imageClass.DepthCompare && compareFunction < 0)
            {
                error = $"{opcode} compares a color texture without its sampler's compare function";
                return false;
            }

            if (imageClass.DepthCompare && !hasCompare)
            {
                error = $"{opcode} gathers depth from a depth-compare binding";
                return false;
            }

            if ((samplerInfo?.MirrorOnceBorderAxes ?? 0) != 0)
            {
                // Documented deviation: a gather past the mirrored copy reads edge texels.
                ShaderCompilerWarnings.WarnOnce(
                    "mirror-once-border-gather",
                    $"{opcode} through a MIRROR_ONCE_BORDER sampler reads edge texels past the mirrored copy instead of the border colour.");
            }

            var offset = SampleOffsets(image, address, imageClass);
            var bias = (flags & Gen5ImageOp.Bias) != 0 ? Temp("float", SampleBias(image, address)) : "0.0f";
            var reference = hasCompare ? Temp("float", SampleReference(image, address)) : "0.0f";
            var coordinates = SampleCoordinates(image, address, imageClass);
            var layerArgument = SampleLayerArgument(image, address, imageClass, texture);
            var oneDimensional = imageClass.SpatialComponents == 1;
            var planar = oneDimensional ? Temp("float2", $"float2({coordinates}, 0.5f)") : coordinates;

            // The float level, or null for level zero. Outside a fragment function there are
            // no quad derivatives: the computed LOD is that of a zero gradient, below every level.
            string? level = null;
            if (!image.Unorm && (flags & Gen5ImageOp.LodZero) == 0)
            {
                if ((flags & Gen5ImageOp.Lod) != 0)
                {
                    level = Temp("float", ImageFloatAddress(image, address.Trailing));
                }
                else if (_stage == Gen5MslStage.Pixel)
                {
                    level = Temp("float", $"{texture}.calculate_unclamped_lod({samplerName}, {planar}) + {bias}");
                }

                // The S# LOD_BIAS raises the given or computed LOD, as for a sample.
                if (level is not null && samplerInfo is { LodBias: not 0 } biased)
                {
                    level = Temp("float", $"{level} + {SamplerLodBiasLiteral(biased.LodBias)}");
                }

                if ((flags & Gen5ImageOp.Clamp) != 0)
                {
                    var clamp = Temp("float", ImageFloatAddress(image, address.Trailing));
                    level = Temp("float", level is null ? $"max(0.0f, {clamp})" : $"max({level}, {clamp})");
                }
            }

            var mip = level is null ? "0u" : NearestMip(level, texture);
            var normalized = image.Unorm ? NormalizeUnnormalizedCoordinates(coordinates, texture, imageClass) : coordinates;
            var component = hasCompare ? 0u : GatherComponent(image.Dmask);
            var resultType = imageClass.DepthCompare ? "float4" : $"vec<{kind}, 4>";

            string Native(string nativeOffset) => imageClass.DepthCompare
                ? $"{texture}.gather_compare({samplerName}, {normalized}{layerArgument}, {reference}, {nativeOffset})"
                : $"{texture}.gather({samplerName}, {normalized}{layerArgument}, {nativeOffset}, component::{"xyzw"[(int)component]})";

            if (imageClass.SupportsGather && level is null && offset is null)
            {
                sampled = Temp(resultType, Native("int2(0)"));
            }
            else
            {
                sampled = Temp(resultType, $"{resultType}(0)");
                string? native = null;
                if (imageClass.SupportsGather)
                {
                    var inRange = offset is null
                        ? "true"
                        : Temp("bool", $"all({offset} >= int2(-8)) && all({offset} <= int2(7))");
                    native = Temp("bool", $"{mip} == 0u && {inRange}");
                    Line($"if ({native})");
                    Line("{");
                    _indent++;
                    Line($"{sampled} = {Native(offset ?? "int2(0)")};");
                    _indent--;
                    Line("}");
                    Line("else");
                }

                Line("{");
                _indent++;
                EmitEmulatedGather(
                    sampled, texture, samplerName, imageClass, image.Unorm ? coordinates : planar, layerArgument,
                    mip, offset, image.Unorm, component, hasCompare && imageClass.DepthCompare ? reference : null);
                _indent--;
                Line("}");
            }

            if (hasCompare && !imageClass.DepthCompare)
            {
                var one = kind == "float" ? "1.0f" : "1";
                var zero = kind == "float" ? "0.0f" : "0";
                var gathered = sampled;
                var tests = Enumerable.Range(0, 4)
                    .Select(texel => $"({CompareExpression(compareFunction, reference, $"(float){gathered}[{texel}]")}) ? {one} : {zero}");
                sampled = Temp(resultType, $"{VectorLiteral(kind)}({string.Join(", ", tests)})");
            }

            return true;
        }

        // The four footprint texels of one level, each read by an explicit-level sample at its
        // centre: a linear filter there weighs the one texel fully, and a compare sampler tests
        // just that texel. The footprint's corner is floor(u * size - 0.5) plus the offsets, in
        // texels of the level (or of mip 0 for unnormalized coordinates, which force level 0).
        private void EmitEmulatedGather(
            string result,
            string texture,
            string samplerName,
            MslImageClass imageClass,
            string coordinates,
            string layerArgument,
            string mip,
            string? offset,
            bool unnormalized,
            uint component,
            string? depthReference)
        {
            var oneDimensional = imageClass.SpatialComponents == 1;
            var size = Temp(
                "float2",
                oneDimensional
                    ? $"float2((float){ImageExtentQuery(texture, imageClass, 0, mip)}, 1.0f)"
                    : $"float2((float){ImageExtentQuery(texture, imageClass, 0, mip)}, (float){ImageExtentQuery(texture, imageClass, 1, mip)})");
            var position = unnormalized
                ? oneDimensional ? $"float2({coordinates}, 0.5f)" : coordinates
                : $"{coordinates} * {size}";
            var corner = Temp("float2", $"floor({position} - 0.5f){(offset is null ? string.Empty : $" + float2({offset})")}");
            if (oneDimensional)
            {
                // One row: both footprint rows are row 0.
                Line($"{corner}.y = 0.0f;");
            }

            int[] columns = [0, 1, 1, 0];
            int[] rows = oneDimensional ? [0, 0, 0, 0] : [1, 1, 0, 0];
            for (var texel = 0; texel < 4; texel++)
            {
                var centre = $"({corner} + float2({columns[texel]}.0f, {rows[texel]}.0f) + 0.5f) / {size}";
                var value = depthReference is not null
                    ? $"{texture}.sample_compare({samplerName}, {centre}{layerArgument}, {depthReference}, metal::level((float){mip}))"
                    : $"{texture}.sample({samplerName}, {centre}{layerArgument}, metal::level((float){mip}))[{component}]";
                Line($"{result}[{texel}] = {value};");
            }
        }

        // GATHER4H returns four horizontally adjacent texels rather than a 2x2 footprint, which
        // the SPIR-V translator expresses as a gather with the constant offsets {-1,0}, {0,0},
        // {1,0} and {2,0}. Metal's gather takes a single offset, so each of the four texels is
        // point-sampled at the footprint corner the gather would have taken it from. UNRM
        // coordinates are already texels of mip 0.
        private string EmitHorizontalGather(
            Gen5ImageControl image,
            SampleAddress address,
            string texture,
            string samplerName,
            MslImageClass imageClass)
        {
            var kind = imageClass.ComponentKind;
            var size = Temp("float2", $"float2({texture}.get_width(0), {texture}.get_height(0))");
            var coordinates = SampleCoordinates(image, address, imageClass);
            var layerArgument = SampleLayerArgument(image, address, imageClass, texture);
            // The gather's own corner texel: the lower-left of the 2x2 footprint.
            var position = image.Unorm ? coordinates : $"{coordinates} * {size}";
            var corner = Temp("float2", $"floor({position} - 0.5f) + 0.5f");
            var component = GatherComponent(image.Dmask);
            var texels = new string[4];
            for (var texel = 0; texel < texels.Length; texel++)
            {
                var sample = Temp(
                    $"vec<{kind}, 4>",
                    $"{texture}.sample({samplerName}, ({corner} + float2({texel - 1}.0f, 0.0f)) / {size}" +
                    $"{layerArgument}, metal::level(0.0f))");
                texels[texel] = $"{sample}[{component}]";
            }

            return Temp($"vec<{kind}, 4>", $"{VectorLiteral(kind)}({string.Join(", ", texels)})");
        }
    }
}
