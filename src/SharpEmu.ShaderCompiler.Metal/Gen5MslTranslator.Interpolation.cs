// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        // Pixel input slots V_INTERP_MOV reads one vertex of (P10/P20, or P1/P2 over
        // them), declared as metal::vertex_value<float4>: MSL 4.0 (macOS 26) on an Apple10
        // GPU. The request says whether the host can run that. Paired with a mesh function
        // that hands the vertex values over (PixelPerVertexInputsFromPrimitive), a slot is
        // instead three float4 inputs of per-primitive data, which every Apple7+ GPU reads.
        private readonly HashSet<uint> _perVertexAttributes = [];

        // Slots read only through P0, the provoking vertex, which a flat input gives.
        private readonly HashSet<uint> _flatParameterAttributes = [];

        // Slots that share a per-vertex input with a V_INTERP_MOV slot; true for a flat slot.
        private readonly Dictionary<uint, bool> _perVertexSourcedAttributes = [];

        // Slots that read another slot's declared input, keyed by slot.
        private readonly Dictionary<uint, uint> _sharedPixelInputs = [];

        // The [[barycentric_coord]] qualifiers, or null when the input is not declared.
        // MSL 3.2 allows only one barycentric input; the MSL 4.0 path accepts both.
        private string? _perspectiveBarycentric;
        private string? _linearBarycentric;

        private bool ReconstructPerspectiveBarycentrics => PerVertexInputsFromPrimitive &&
            _perspectiveBarycentric is not null && _linearBarycentric is not null;

        private string PerspectiveBarycentricField => ReconstructPerspectiveBarycentrics
            ? "sharpemu_bary_persp"
            : "sharpemu_in.sharpemu_bary_persp";
        private const string LinearBarycentricField = "sharpemu_in.sharpemu_bary_linear";
        private const string VertexWInputName = "sharpemu_vertex_w";

        private static readonly string[] VertexIndexNames = ["first", "second", "third"];

        // The host interpolants that replace the program's hand interpolations, and where
        // (ShaderCompileRequest.PixelManualInterpolation); the slots they read become flat.
        private Gen5ManualInterpolationPlan? _manualInterpolation;
        private ILookup<uint, Gen5InterpolationRewrite>? _interpolationRewrites;
        private string? _manualInterpolationFailure;

        private bool IsManuallyInterpolated(uint attribute) =>
            _manualInterpolation is { } plan && attribute < 32 && (plan.Attributes & (1u << (int)attribute)) != 0;

        /// <summary>True when the program reads every vertex's value of some input.</summary>
        private bool UsesPerVertexPixelInputs => _perVertexAttributes.Count != 0;

        /// <summary>True when those values come from per-primitive data, not vertex_value.</summary>
        private bool PerVertexInputsFromPrimitive => _request.PixelPerVertexInputsFromPrimitive;

        /// <summary>True when the program declares a vertex_value input, which only MSL 4.0 compiles.</summary>
        private bool UsesVertexValueInputs => UsesPerVertexPixelInputs && !PerVertexInputsFromPrimitive;

        private uint EnabledPixelInputs => _pixelInputAddress & _pixelInputEnable;

        // SPI_PS_INPUT_CNTL for a slot: OFFSET in [4:0], FLAT_SHADE in bit 10.
        private uint PixelInputControl(uint attribute) =>
            attribute < (uint)_pixelInputCntl.Length ? _pixelInputCntl[attribute] : attribute;

        /// <summary>
        /// Decides how every V_INTERP slot is declared, before any block is emitted.
        /// Mirrors the SPIR-V translator's DeclareInterpolationParameters and its slot
        /// grouping, except that a slot read only through P0 stays a flat input even
        /// when per-vertex inputs exist: flat is exactly P0 and needs no MSL 4.0.
        /// </summary>
        private void ClassifyPixelInputs()
        {
            if (_request.PixelManualInterpolation)
            {
                _manualInterpolation = Gen5ManualInterpolation.Analyze(
                    _request.Program, _pixelInputCntl, _request.PixelCustomInterpolationMask, _pixelInputAddress,
                    out var reason);
                _manualInterpolationFailure = _manualInterpolation is null ? reason : null;
                _interpolationRewrites = _manualInterpolation?.Rewrites.ToLookup(rewrite => rewrite.Pc);
            }

            foreach (var attribute in _pixelAttributes)
            {
                if (!NeedsVertexValues(attribute))
                {
                    continue;
                }

                if (ReadsOnlyTheProvokingVertex(attribute) || IsManuallyInterpolated(attribute))
                {
                    _flatParameterAttributes.Add(attribute);
                }
                else if (_request.SupportsPerVertexPixelInputs || PerVertexInputsFromPrimitive)
                {
                    _perVertexAttributes.Add(attribute);
                }
            }

            // Several PS input slots may read one VS parameter (the guest compiler emits a
            // slot per use). Slots of one kind share one host input; when a V_INTERP_MOV
            // slot is among them, the others read the per-vertex values (interpolated with
            // the barycentrics, or vertex 0 for a flat slot).
            bool IsFlatSlot(uint attribute) =>
                (PixelInputControl(attribute) & 0x400u) != 0 ||
                _flatParameterAttributes.Contains(attribute);

            foreach (var group in _pixelAttributes.GroupBy(attribute =>
                         PixelInputControl(attribute) & 0x1Fu))
            {
                var perVertexOwner = group.Where(_perVertexAttributes.Contains).Cast<uint?>().FirstOrDefault();
                var owners = new Dictionary<bool, uint>();
                foreach (var attribute in group)
                {
                    if (perVertexOwner is { } perVertex)
                    {
                        if (attribute != perVertex)
                        {
                            _sharedPixelInputs.Add(attribute, perVertex);
                            if (!_perVertexAttributes.Contains(attribute))
                            {
                                _perVertexSourcedAttributes.Add(attribute, IsFlatSlot(attribute));
                            }
                        }

                        continue;
                    }

                    // Smooth slots of one parameter share one input, interpolated where the
                    // first of them is: the vertex program writes the location once.
                    var flat = IsFlatSlot(attribute);
                    if (owners.TryGetValue(flat, out var owner))
                    {
                        _sharedPixelInputs.Add(attribute, owner);
                    }
                    else
                    {
                        owners.Add(flat, attribute);
                    }
                }
            }

            if (!UsesPerVertexPixelInputs)
            {
                return;
            }

            // A V_INTERP_P1/P2 over a per-vertex slot reads the I/J barycentrics from the
            // input VGPRs, so those are seeded from [[barycentric_coord]]. MSL 3.2 only
            // accepts center_perspective/center_no_perspective on these built-ins, even
            // when SAMPLE or CENTROID is the only enabled guest location. Use the same
            // center approximation as mixed locations; exact multisample locations need
            // a separate implementation. Ordinary user interpolants keep their qualifiers.
            var enabled = EnabledPixelInputs;
            if ((enabled & 0xFu) != 0 || _perVertexSourcedAttributes.ContainsValue(false))
            {
                _perspectiveBarycentric = "center_perspective";
            }

            if ((enabled & 0x70u) != 0)
            {
                _linearBarycentric = "center_no_perspective";
            }
        }

        /// <summary>The Gen5PsIn members for the V_INTERP slots and the pixel system inputs.</summary>
        private void EmitPixelInputMembers(StringBuilder source)
        {
            var attributes = _pixelAttributes.Where(attribute => !_sharedPixelInputs.ContainsKey(attribute)).ToArray();
            var locations = Gen5PixelInputMapping.ResolveLocations(_pixelInputCntl, attributes);
            for (var index = 0; index < attributes.Length; index++)
            {
                var attribute = attributes[index];
                if (_perVertexAttributes.Contains(attribute) && PerVertexInputsFromPrimitive)
                {
                    // Named by the param the slot reads (SPI_PS_INPUT_CNTL.OFFSET), as the
                    // mesh function names the primitive's copies of it.
                    for (var vertex = 0u; vertex < 3u; vertex++)
                    {
                        source.AppendLine(
                            $"    float4 attr{attribute}_v{vertex} " +
                            $"[[user({PerVertexInputName(PixelInputControl(attribute) & 0x1Fu, vertex)})]];");
                    }

                    continue;
                }

                if (_perVertexAttributes.Contains(attribute))
                {
                    source.AppendLine(
                        $"    vertex_value<float4> attr{attribute} [[user(locn{locations[index]})]];");
                    continue;
                }

                var qualifier = (PixelInputControl(attribute) & 0x400u) != 0 ||
                    _flatParameterAttributes.Contains(attribute)
                    ? ", flat"
                    : InterpolationQualifier(attribute) is { } mode ? $", {mode}" : string.Empty;
                source.AppendLine($"    float4 attr{attribute} [[user(locn{locations[index]}){qualifier}]];");
            }

            foreach (var interpolant in _manualInterpolation?.Interpolants ?? [])
            {
                var mode = BarycentricQualifier((int)interpolant.Barycentric) is { } qualifier ? $", {qualifier}" : string.Empty;
                source.AppendLine(
                    $"    float4 {InterpolantField(interpolant.Location, interpolant.Kind)} " +
                    $"[[user({InterpolantName(interpolant.Location, interpolant.Kind)}){mode}]];");
            }

            if (ReconstructPerspectiveBarycentrics)
            {
                source.AppendLine($"    float3 sharpemu_vertex_w [[user({VertexWInputName}), flat]];");
            }
            else if (_perspectiveBarycentric is not null)
            {
                source.AppendLine($"    float3 sharpemu_bary_persp [[barycentric_coord, {_perspectiveBarycentric}]];");
            }

            if (_linearBarycentric is not null)
            {
                source.AppendLine($"    float3 sharpemu_bary_linear [[barycentric_coord, {_linearBarycentric}]];");
            }

            // FRONT_FACE and ANCILLARY are VGPRs the hardware fills for pixel shaders that
            // enable them; SAMPLE_COVERAGE is the pixel's coverage mask.
            var enabled = EnabledPixelInputs;
            if ((enabled & (1u << 12)) != 0)
            {
                source.AppendLine("    bool sharpemu_front_facing [[front_facing]];");
            }

            if ((enabled & (1u << 13)) != 0)
            {
                source.AppendLine("    uint sharpemu_layer [[render_target_array_index]];");
            }

            if ((enabled & (1u << 14)) != 0)
            {
                source.AppendLine("    uint sharpemu_sample_mask [[sample_mask]];");
            }
        }

        private void EmitBarycentricReconstruction(StringBuilder source)
        {
            if (!ReconstructPerspectiveBarycentrics) return;

            // Perspective weights are (linear_i / clip_w_i) / sum(linear_j / clip_w_j).
            // The mesh hands clip W in primitive vertex order, so the 3.2 fragment needs
            // only one built-in instead of the illegal pair of barycentric declarations.
            source.AppendLine($"    float3 sharpemu_bary_persp = {LinearBarycentricField} / sharpemu_in.sharpemu_vertex_w;");
            source.AppendLine("    sharpemu_bary_persp /= dot(sharpemu_bary_persp, float3(1.0f));");
        }

        private string PixelInputField(uint attribute) =>
            $"sharpemu_in.attr{(_sharedPixelInputs.TryGetValue(attribute, out var owner) ? owner : attribute)}";

        private string VertexValue(string field, uint vertex, uint channel) => PerVertexInputsFromPrimitive
            ? $"{field}_v{vertex}[{channel}]"
            : $"{field}.get(vertex_index::{VertexIndexNames[vertex]})[{channel}]";

        // A smooth or flat slot read from a per-vertex input: the vertices weighted by the
        // perspective barycentrics, or vertex 0 (the provoking vertex) for a flat slot.
        private void EmitPerVertexSourcedInterpolation(
            Gen5InterpolationControl interpolation,
            uint destination,
            bool flat)
        {
            var field = PixelInputField(interpolation.Attribute);
            var value = VertexValue(field, 0, interpolation.Channel);
            if (!flat)
            {
                value = $"({value} * {PerspectiveBarycentricField}.x + " +
                    $"{VertexValue(field, 1, interpolation.Channel)} * {PerspectiveBarycentricField}.y + " +
                    $"{VertexValue(field, 2, interpolation.Channel)} * {PerspectiveBarycentricField}.z)";
            }

            StoreVector(destination, AsUInt(value));
        }

        // V_INTERP_MOV P10/P20/P0 and V_INTERP_P1/P2 over the three vertex values. A
        // custom-interpolated slot hands the raw vertex values instead of the deltas.
        private bool TryEmitInterpolationParameter(
            Gen5ShaderInstruction instruction,
            Gen5InterpolationControl interpolation,
            uint destination,
            out string error)
        {
            error = string.Empty;
            var custom = interpolation.Attribute < 32 &&
                (_request.PixelCustomInterpolationMask & (1u << (int)interpolation.Attribute)) != 0;
            var field = PixelInputField(interpolation.Attribute);

            string Parameter(uint mode)
            {
                var value = VertexValue(field, (mode + 1) % 3, interpolation.Channel);
                return !custom && mode < 2
                    ? $"({value} - {VertexValue(field, 0, interpolation.Channel)})"
                    : value;
            }

            // The f16 forms read one half of each vertex's 32-bit parameter (HIGH picks the
            // upper one). P10/P20 are the vertex differences rounded to f16, as AnyPS5 reads
            // them; a custom-interpolated slot hands the raw vertex halves instead.
            var half = interpolation.High ? 1 : 0;
            string HalfVertex(uint vertex) =>
                $"(float)as_type<half2>(as_type<uint>({VertexValue(field, vertex, interpolation.Channel)}))[{half}]";
            string HalfParameter(uint mode)
            {
                var value = HalfVertex((mode + 1) % 3);
                return !custom && mode < 2 ? $"(float)half({value} - {HalfVertex(0)})" : value;
            }

            string result;
            switch (instruction.Opcode)
            {
                case "VInterpMovF32":
                {
                    var mode = interpolation.Parameter;
                    if (mode >= 3)
                    {
                        error = "reserved interpolation parameter selector";
                        return false;
                    }

                    result = Parameter(mode);
                    break;
                }

                case "VInterpP1F32" or "VInterpP2F32":
                {
                    var firstPhase = instruction.Opcode == "VInterpP1F32";
                    var source = InterpolationSource(instruction, interpolation, 0);
                    result = $"({Parameter(firstPhase ? 0u : 1u)} * {source} + " +
                        $"{(firstPhase ? Parameter(2) : AsFloat($"v[{destination}]"))})";
                    break;
                }

                // P10 * I + P0 (or + S2's half), rounded once to f32 (AnyPS5 measured RDNA2).
                case "VInterpP1llF16":
                    result = $"fma({HalfParameter(0)}, {InterpolationSource(instruction, interpolation, 0)}, {HalfParameter(2)})";
                    break;
                case "VInterpP1lvF16":
                {
                    var origin = InterpolationSourceModifiers(interpolation, 1,
                        $"(float)as_type<half2>({SourceExpression(instruction.Sources[1], instruction)})[{half}]");
                    result = $"fma({HalfParameter(0)}, {InterpolationSource(instruction, interpolation, 0)}, {origin})";
                    break;
                }

                // P20 * J + S2.f32, rounded to f16 into the low half (the high one with
                // op_sel[3]); the other half is kept, like the other f16 VOP3 results.
                case "VInterpP2F16":
                {
                    var sum = $"fma({HalfParameter(1)}, {InterpolationSource(instruction, interpolation, 0)}, " +
                        $"{InterpolationSource(instruction, interpolation, 1)})";
                    var packed = $"(uint)as_type<ushort>(half({ApplyInterpolationOutputModifiers(interpolation, sum)}))";
                    StoreVector(destination, ((interpolation.Modifiers?.OperandSelect ?? 0) & 8) != 0
                        ? $"((v[{destination}] & 0x0000FFFFu) | ({packed} << 16))"
                        : $"((v[{destination}] & 0xFFFF0000u) | {packed})");
                    return true;
                }

                default:
                    error = $"unsupported interpolation opcode {instruction.Opcode}";
                    return false;
            }

            StoreVector(destination, AsUInt(ApplyInterpolationOutputModifiers(interpolation, result)));
            return true;
        }

        private static bool IsHalfInterpolation(string opcode) =>
            opcode is "VInterpP1llF16" or "VInterpP1lvF16" or "VInterpP2F16";

        // A slot needs every vertex's value when a program reads its parameters (V_INTERP_MOV),
        // its 16-bit halves (the f16 forms), or modifies the I/J it is interpolated with.
        private bool NeedsVertexValues(uint attribute) =>
            _request.Program.Instructions.Any(instruction =>
                instruction.Control is Gen5InterpolationControl control &&
                control.Attribute == attribute &&
                (instruction.Opcode == "VInterpMovF32" || IsHalfInterpolation(instruction.Opcode) ||
                 control.Modifiers is { } modifiers && ((modifiers.AbsoluteMask | modifiers.NegateMask) & 0x6u) != 0));

        // Source operand index (0: I/J, 1: S2) as a float with its VOP3 abs/neg, which count
        // from src1 because src0 holds the attribute.
        private string InterpolationSource(Gen5ShaderInstruction instruction, Gen5InterpolationControl interpolation, int index) =>
            InterpolationSourceModifiers(interpolation, index,
                AsFloat(SourceExpression(instruction.Sources[index], instruction)));

        private static string InterpolationSourceModifiers(Gen5InterpolationControl interpolation, int index, string value)
        {
            var bit = 1u << (index + 1);
            if (interpolation.Modifiers is not { } modifiers)
            {
                return value;
            }

            if ((modifiers.AbsoluteMask & bit) != 0)
            {
                value = $"fabs({value})";
            }

            return (modifiers.NegateMask & bit) != 0 ? $"(-{value})" : value;
        }

        // The VOP3 forms' omod (ignored under IEEE mode, RDNA2 ISA 6.2.2) and clamp.
        private string ApplyInterpolationOutputModifiers(Gen5InterpolationControl interpolation, string expression)
        {
            if (interpolation.Modifiers is not { } modifiers)
            {
                return expression;
            }

            expression = (_request.IeeeMode ? 0u : modifiers.OutputModifier) switch
            {
                1 => $"(({expression}) * 2.0f)",
                2 => $"(({expression}) * 4.0f)",
                3 => $"(({expression}) * 0.5f)",
                _ => expression,
            };
            return modifiers.Clamp ? $"clamp({expression}, 0.0f, 1.0f)" : expression;
        }

        // The interpolation location of a host-interpolated slot: which barycentric input the
        // I/J VGPR of its V_INTERP_P1/P2 is. SPI_PS_INPUT_ADDR lays those out in bit order
        // (PERSP_SAMPLE/CENTER/CENTROID two VGPRs each, PULL_MODEL three, LINEAR_SAMPLE/
        // CENTER/CENTROID two each). PERSP_SAMPLE and LINEAR_SAMPLE interpolate at the
        // sample, which on a single-sample target is the pixel center. Null for the pixel
        // center with perspective, Metal's default (and the pull model's).
        private string? InterpolationQualifier(uint attribute)
        {
            foreach (var instruction in _request.Program.Instructions)
            {
                if (instruction.Control is not Gen5InterpolationControl control || control.Attribute != attribute ||
                    instruction.Opcode is not ("VInterpP1F32" or "VInterpP2F32") ||
                    instruction.Sources.Count == 0 || instruction.Sources[0].Kind != Gen5OperandKind.VectorRegister)
                {
                    continue;
                }

                var vgpr = instruction.Sources[0].Value;
                var start = 0u;
                for (var bit = 0; bit < 7; bit++)
                {
                    if ((_pixelInputAddress & (1u << bit)) == 0)
                    {
                        continue;
                    }

                    var count = bit == 3 ? 3u : 2u;
                    if (vgpr >= start && vgpr < start + count)
                    {
                        return BarycentricQualifier(bit);
                    }

                    start += count;
                }
            }

            return null;
        }

        // The Metal qualifier of an SPI_PS_INPUT_ENA barycentric (null for PERSP_CENTER, the
        // default, and the pull model's).
        private static string? BarycentricQualifier(int bit) => bit switch
        {
            0 => "sample_perspective",
            2 => "centroid_perspective",
            4 => "sample_no_perspective",
            5 => "center_no_perspective",
            6 => "centroid_no_perspective",
            _ => null,
        };

        /// <summary>The [[user]] name of a host interpolant (ShaderCompileRequest.VertexInterpolants):
        /// param <paramref name="location"/>'s channels unpacked as <paramref name="kind"/>.</summary>
        internal static string InterpolantName(uint location, Gen5InterpolantKind kind) => $"locn{location}_i{(int)kind}";

        internal static string InterpolantField(uint location, Gen5InterpolantKind kind) => $"interp{location}_{(int)kind}";

        // A param's four channels as an interpolant kind, from the float4 that carries its bits.
        internal static string UnpackInterpolant(string param, Gen5InterpolantKind kind)
        {
            if (kind == Gen5InterpolantKind.Float)
            {
                return param;
            }

            var channels = new string[4];
            for (var channel = 0; channel < 4; channel++)
            {
                var bits = $"as_type<uint>({param}[{channel}])";
                channels[channel] = kind switch
                {
                    Gen5InterpolantKind.HalfLow => $"(float)as_type<half2>({bits})[0]",
                    Gen5InterpolantKind.HalfHigh => $"(float)as_type<half2>({bits})[1]",
                    Gen5InterpolantKind.UInt => $"(float){bits}",
                    Gen5InterpolantKind.Int => $"(float)as_type<int>({bits})",
                    _ => $"(float)(({bits} >> {8 * (kind - Gen5InterpolantKind.Byte0)}u) & 0xFFu)",
                };
            }

            return $"float4({string.Join(", ", channels)})";
        }

        // After a hand interpolation completes, its register takes the host's interpolation of
        // the same channel instead.
        private void EmitInterpolationRewrites(uint pc)
        {
            if (_interpolationRewrites is null)
            {
                return;
            }

            foreach (var rewrite in _interpolationRewrites[pc])
            {
                var value = $"sharpemu_in.{InterpolantField(rewrite.Location, rewrite.Kind)}[{rewrite.Channel}]";
                if (rewrite.Scale != 1f)
                {
                    value = $"({value} * as_type<float>(0x{BitConverter.SingleToUInt32Bits(rewrite.Scale):X8}u))";
                }

                if (rewrite.Offset != 0f)
                {
                    value = $"({value} + as_type<float>(0x{BitConverter.SingleToUInt32Bits(rewrite.Offset):X8}u))";
                }

                if (rewrite.Clamp)
                {
                    value = $"clamp({value}, 0.0f, 1.0f)";
                }

                StoreVector(rewrite.Register, AsUInt(value));
            }
        }

        // The param exports of a vertex stage also write the interpolants unpacked from them.
        private void EmitInterpolantExports(uint location, string param)
        {
            foreach (var encoded in _request.VertexInterpolants)
            {
                var (interpolantLocation, kind) = Gen5Interpolant.Decode(encoded);
                if (interpolantLocation == location)
                {
                    Line($"sharpemu_out.{InterpolantField(location, kind)} = {UnpackInterpolant(param, kind)};");
                }
            }
        }

        private void EmitInterpolantOutputMembers(StringBuilder source)
        {
            foreach (var encoded in _request.VertexInterpolants)
            {
                var (location, kind) = Gen5Interpolant.Decode(encoded);
                source.AppendLine($"    float4 {InterpolantField(location, kind)} [[user({InterpolantName(location, kind)})]];");
            }
        }

        /// <summary>
        /// The I/J barycentrics an interpolation slot's VGPRs carry when the program
        /// interpolates per-vertex inputs by hand, or the I/W, J/W, 1/W of the pull
        /// model. Null when the slot keeps its registers unwritten.
        /// </summary>
        private string[]? BarycentricInputRegisters(int bit)
        {
            if (bit > 6 || !UsesPerVertexPixelInputs || (EnabledPixelInputs & (1u << bit)) == 0)
            {
                return null;
            }

            if (bit == 3)
            {
                // [[position]].w is 1/W, and the perspective-correct barycentric times 1/W
                // is the screen-linear I/W the pull model interpolates.
                const string inverseW = "sharpemu_in.sharpemu_frag_coord.w";
                return
                [
                    $"{PerspectiveBarycentricField}.y * {inverseW}",
                    $"{PerspectiveBarycentricField}.z * {inverseW}",
                    inverseW,
                ];
            }

            var field = bit < 4 ? PerspectiveBarycentricField : LinearBarycentricField;
            return [$"{field}.y", $"{field}.z"];
        }

        // FRONT_FACE is 0xFFFFFFFF for a front face and 0 for a back face with
        // SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS (Mesa's RADV sets it and ACO tests the input with
        // v_cmp_lg_u32 0), and +1.0/-1.0 without it (Mesa's fgt 0 and fsign reads), as AnyPS5
        // reads both. ANCILLARY carries the render target array index in bits [28:16].
        // POS_FIXED_PT holds the integer pixel position: x in bits [15:0] and y in bits [31:16].
        private string? PixelSystemInput(int bit) => bit switch
        {
            12 => _request.PixelFrontFaceAllBits
                ? "(sharpemu_in.sharpemu_front_facing ? 0xFFFFFFFFu : 0u)"
                : "(sharpemu_in.sharpemu_front_facing ? 0x3F800000u : 0xBF800000u)",
            13 => "(sharpemu_in.sharpemu_layer << 16)",
            14 => "sharpemu_in.sharpemu_sample_mask",
            15 => "((uint(sharpemu_in.sharpemu_frag_coord.x) & 0xFFFFu) | " +
                "(uint(sharpemu_in.sharpemu_frag_coord.y) << 16))",
            _ => null,
        };
    }
}
