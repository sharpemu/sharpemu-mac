// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics.CodeAnalysis;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

// The less common GFX10 integer, f16 and scalar ALU opcodes: the legacy (DX9) multiplies,
// frexp, the 16-bit conversions and normalizations, the SAD family, byte permute, the
// 16-bit three-operand and packed integer ops, the dot products, the relative scalar
// moves and the EXEC write-back ops. Each follows the RDNA2 ISA pseudo code; anything
// whose behaviour the ISA leaves open is refused with the reason rather than guessed.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        // ---- legacy (DX9) multiplies ----

        /// <summary>
        /// V_MUL_LEGACY_F32 / V_MAC_LEGACY_F32 / V_MAD_LEGACY_F32: "0.0 times anything produces
        /// 0.0" — even infinity or NaN — and the product is then +0. The test is on the
        /// operand after its abs/neg, by bits, so -0 counts. On gfx10.1 (the PS5's gfx1013) the
        /// MAC/MAD forms are not fused: the product is rounded before the add (gfx10.3 replaced
        /// them with V_FMAC/FMA_LEGACY_F32), and they flush their denormal sources in every
        /// FLOAT_MODE, so a denormal factor takes the zero rule too. The zero product is then
        /// added to the addend (+0 + -0 = +0).
        /// </summary>
        private string EmitLegacyMultiply(Gen5ShaderInstruction instruction, string? addend)
        {
            var left = Temp("float", F(instruction, 0));
            var right = Temp("float", F(instruction, 1));
            const string zeroMask = "0x7F800000u";
            var zero = Temp(
                "bool",
                $"(as_type<uint>({left}) & {zeroMask}) == 0u || (as_type<uint>({right}) & {zeroMask}) == 0u");
            if (addend is null)
            {
                return FloatResult(instruction, $"({zero} ? 0.0f : ({left} * {right}))");
            }

            var c = Temp("float", addend);
            // Its own statement: Metal contracts a * b + c written as one expression into an fma.
            var product = Temp("float", $"{zero} ? 0.0f : ({left} * {right})");
            return FloatResult(instruction, $"({product} + {c})");
        }

        /// <summary>V_MAD_F32 / V_MADMK_F32 / V_MADAK_F32 / V_MAC_F32: the product is rounded
        /// to f32 before the add. The product is its own statement because Metal contracts
        /// a * b + c written as one expression into a single-rounding fma.</summary>
        private string EmitUnfusedMultiplyAdd(Gen5ShaderInstruction instruction, string addend)
        {
            var product = Temp("float", $"{F(instruction, 0)} * {F(instruction, 1)}");
            var c = Temp("float", addend);
            return FloatResult(instruction, $"({product} + {c})");
        }

        // ---- frexp ----

        /// <summary>V_FREXP_MANT_F32 / V_FREXP_EXP_I32_F32 on the operand's bits: the
        /// significand in [0.5, 1) with the input's sign and the matching exponent, denormal
        /// inputs normalized exactly. Infinity and NaN return the input and exponent 0, and
        /// so does a zero (frexp(0) = 0 * 2^0).</summary>
        private string EmitFrexpF32(Gen5ShaderInstruction instruction, bool mantissa)
        {
            var bits = FlushInputDenormalF32(AsUInt(F(instruction, 0)));
            var biased = Temp("uint", $"({bits} >> 23) & 0xFFu");
            var fraction = Temp("uint", $"{bits} & 0x7FFFFFu");
            var special = Temp("bool", $"{biased} == 0xFFu || ({bits} & 0x7FFFFFFFu) == 0u");
            // A denormal's leading one sits at bit p = 31 - clz; it becomes the implicit bit.
            var lead = Temp("uint", $"31u - (uint)clz({fraction} | 1u)");
            if (!mantissa)
            {
                return AsUInt(
                    $"({special} ? 0 : ({biased} == 0u ? (int){lead} - 148 : (int){biased} - 126))");
            }

            var normalized = Temp(
                "uint",
                $"{biased} == 0u ? (({fraction} << (23u - {lead})) & 0x7FFFFFu) : {fraction}");
            var result = Temp(
                "uint",
                $"{special} ? {bits} : (({bits} & 0x80000000u) | (126u << 23) | {normalized})");
            return HasFloatOutputModifiers(instruction)
                ? FloatResult(instruction, AsFloat(result))
                : result;
        }

        /// <summary>The f16 forms: V_FREXP_MANT_F16 returns an f16 and V_FREXP_EXP_I16_F16 an
        /// i16, both into the op_sel-selected destination half.</summary>
        private string EmitFrexpF16(Gen5ShaderInstruction instruction, uint destination, bool mantissa)
        {
            var bits = ReadMinimumHalfBits(instruction, 0);
            var biased = Temp("uint", $"({bits} >> 10) & 0x1Fu");
            var fraction = Temp("uint", $"{bits} & 0x3FFu");
            var special = Temp("bool", $"{biased} == 0x1Fu || ({bits} & 0x7FFFu) == 0u");
            var lead = Temp("uint", $"31u - (uint)clz({fraction} | 1u)");
            if (!mantissa)
            {
                return Int16Result(
                    instruction,
                    destination,
                    AsUInt($"({special} ? 0 : ({biased} == 0u ? (int){lead} - 23 : (int){biased} - 14))"));
            }

            var normalized = Temp(
                "uint",
                $"{biased} == 0u ? (({fraction} << (10u - {lead})) & 0x3FFu) : {fraction}");
            var result = Temp(
                "uint",
                $"{special} ? {bits} : (({bits} & 0x8000u) | (14u << 10) | {normalized})");
            return HasFloatOutputModifiers(instruction, half: true)
                ? Float16Result(instruction, destination, $"(float)as_type<half>((ushort){result})")
                : Int16Result(instruction, destination, result);
        }

        /// <summary>Whether omod or clamp would change a float result (omod only counts outside
        /// IEEE mode). Bit-exact paths skip the float round trip when neither applies.</summary>
        private bool HasFloatOutputModifiers(Gen5ShaderInstruction instruction, bool half = false)
        {
            var (outputModifier, clamp) = FloatOutputModifiers(instruction, half);
            return outputModifier != 0 || clamp;
        }

        // ---- 16-bit conversions ----

        /// <summary>V_CVT_I16_F16: truncate toward zero, saturate to the i16 range, NaN is 0.</summary>
        private string EmitCvtI16F16(Gen5ShaderInstruction instruction, uint destination)
        {
            var value = Temp("float", F16(instruction, 0));
            return Int16Result(
                instruction,
                destination,
                AsUInt($"(isnan({value}) ? 0 : (int)clamp(trunc({value}), -32768.0f, 32767.0f))"));
        }

        /// <summary>
        /// flt16_to_snorm16 / flt16_to_unorm16 on f16 bits: clamp to [-1, 1] (unorm [0, 1]),
        /// scale by 32767 (65535) and round to nearest even, NaN giving 0. The product of an
        /// 11-bit significand and the 15/16-bit scale needs 27 bits, more than an f32 holds,
        /// so it is formed as an integer and the rounding shift is done by hand.
        /// </summary>
        private string NormalizeHalfBits(string bits, bool signed)
        {
            var magnitude = Temp("uint", $"{bits} & 0x7FFFu");
            var negative = Temp("bool", $"({bits} & 0x8000u) != 0u");
            var scale = signed ? 32767u : 65535u;
            var biased = Temp("uint", $"{magnitude} >> 10");
            // value = significand * 2^(max(biased, 1) - 25); below 1.0 the exponent is negative.
            var significand = Temp("uint", $"({magnitude} & 0x3FFu) | ({biased} != 0u ? 0x400u : 0u)");
            // Only read below 1.0, where biased <= 14; the clamp keeps the shifts defined.
            var shift = Temp("uint", $"25u - clamp({biased}, 1u, 14u)");
            var product = Temp("uint", $"{significand} * {scale}u");
            var truncated = Temp("uint", $"{product} >> {shift}");
            var remainder = Temp("uint", $"{product} & ((1u << {shift}) - 1u)");
            var halfway = Temp("uint", $"1u << ({shift} - 1u)");
            var rounded = Temp(
                "uint",
                $"{truncated} + (({remainder} > {halfway} || ({remainder} == {halfway} && ({truncated} & 1u) != 0u)) ? 1u : 0u)");
            // magnitude >= 1.0 (0x3C00) saturates; NaN is 0.
            var level = Temp(
                "uint",
                $"{magnitude} > 0x7C00u ? 0u : ({magnitude} >= 0x3C00u ? {scale}u : {rounded})");
            return signed
                ? $"({negative} ? (0u - {level}) & 0xFFFFu : {level})"
                : $"({negative} ? 0u : {level})";
        }

        /// <summary>V_SAT_PK_U8_I16: each signed 16-bit half saturated to 0..255, the two bytes
        /// in the low half and zeroes above (D = {16'b0, sat8(S0[31:16]), sat8(S0[15:0])}).</summary>
        private string EmitSaturatePackU8(Gen5ShaderInstruction instruction)
        {
            var source = Temp("uint", RawSource(instruction, 0));
            string Saturated(string half) => $"(uint)clamp(as_type<int>({half} << 16) >> 16, 0, 255)";
            return $"({Saturated($"({source} & 0xFFFFu)")} | ({Saturated($"({source} >> 16)")} << 8))";
        }

        // ---- integer multiplies / dots on VOP2 ----

        /// <summary>V_MUL_HI_I32_I24: bits 63:32 of the 48-bit signed product.</summary>
        private string EmitSignedMultiplyHigh24(Gen5ShaderInstruction instruction)
        {
            var left = $"(long)(as_type<int>(({RawSource(instruction, 0)}) << 8u) >> 8)";
            var right = $"(long)(as_type<int>(({RawSource(instruction, 1)}) << 8u) >> 8)";
            return $"(uint)(as_type<ulong>({left} * {right}) >> 32)";
        }

        /// <summary>V_DOT4C_I32_I8: the signed byte products of S0 and S1 added to the old
        /// destination, wrapping in 32 bits.</summary>
        private string EmitDot4cI32I8(Gen5ShaderInstruction instruction, uint destination)
        {
            var left = Temp("uint", RawSource(instruction, 0));
            var right = Temp("uint", RawSource(instruction, 1));
            var products = string.Join(
                " + ",
                Enumerable.Range(0, 4).Select(index =>
                    $"({SignedField(left, index * 8, 8)} * {SignedField(right, index * 8, 8)})"));
            return $"(as_type<uint>({products}) + v[{destination}])";
        }

        /// <summary>The signed field of <paramref name="width"/> bits at <paramref name="offset"/>, as an int.</summary>
        private static string SignedField(string value, int offset, int width) =>
            $"(as_type<int>({value} << {32 - offset - width}u) >> {32 - width})";

        /// <summary>The unsigned field of <paramref name="width"/> bits at <paramref name="offset"/>.</summary>
        private static string UnsignedField(string value, int offset, int width) =>
            $"(({value} >> {offset}u) & 0x{(1u << width) - 1u:X}u)";

        // ---- SAD family / lerp / permute ----

        /// <summary>V_LERP_U8: per byte (S0 + S1 + S2's round bit) >> 1.</summary>
        private string EmitLerpU8(Gen5ShaderInstruction instruction)
        {
            var a = Temp("uint", RawSource(instruction, 0));
            var b = Temp("uint", RawSource(instruction, 1));
            var c = Temp("uint", RawSource(instruction, 2));
            var bytes = Enumerable.Range(0, 4).Select(index =>
                $"((({UnsignedField(a, index * 8, 8)} + {UnsignedField(b, index * 8, 8)} + (({c} >> {index * 8}u) & 1u)) >> 1u) << {index * 8}u)");
            return $"({string.Join(" | ", bytes)})";
        }

        /// <summary>SAD_U8 / MSAD_U8 of the four bytes of <paramref name="reference"/> against
        /// <paramref name="window"/>: the sum of the unsigned byte differences; the masked form
        /// leaves out the bytes whose reference is zero.</summary>
        private static string ByteSad(string window, string reference, bool masked)
        {
            var terms = Enumerable.Range(0, 4).Select(index =>
            {
                var x = UnsignedField(window, index * 8, 8);
                var y = UnsignedField(reference, index * 8, 8);
                var difference = $"(max({x}, {y}) - min({x}, {y}))";
                return masked ? $"({y} == 0u ? 0u : {difference})" : difference;
            });
            return $"({string.Join(" + ", terms)})";
        }

        /// <summary>V_MSAD_U8: D = S2 + MSAD_U8(S0, S1), overflow wrapping.</summary>
        private string EmitMsadU8(Gen5ShaderInstruction instruction)
        {
            var window = Temp("uint", RawSource(instruction, 0));
            var reference = Temp("uint", RawSource(instruction, 1));
            return $"(({RawSource(instruction, 2)}) + {ByteSad(window, reference, masked: true)})";
        }

        /// <summary>
        /// V_QSAD_PK_U16_U8 / V_MQSAD_PK_U16_U8 (two 16-bit accumulators per dword, D and S2
        /// register pairs) and V_MQSAD_U32_U8 (four 32-bit accumulators, D and S2 register
        /// quads): result k is the (masked) SAD of S0[8k+31:8k] against S1 plus accumulator k.
        /// The 16-bit sums wrap within their field. Every result is formed before any
        /// destination register is written, so D may overlap a source.
        /// </summary>
        private bool TryEmitQuadSad(Gen5ShaderInstruction instruction, uint destination, out string error)
        {
            error = string.Empty;
            var masked = instruction.Opcode != "VQsadPkU16U8";
            var wide = instruction.Opcode == "VMqsadU32U8";
            var window = Temp("ulong", RawSource64(instruction, 0));
            var reference = Temp("uint", RawSource(instruction, 1));
            string[] accumulators;
            if (wide)
            {
                if (!TryRawSource128(instruction, 2, out accumulators, out error))
                {
                    return false;
                }
            }
            else
            {
                var pair = Temp("ulong", RawSource64(instruction, 2));
                accumulators = [Temp("uint", $"(uint){pair}"), Temp("uint", $"(uint)({pair} >> 32)")];
            }

            var sads = new string[4];
            for (var index = 0; index < 4; index++)
            {
                sads[index] = Temp("uint", ByteSad($"(uint)({window} >> {index * 8}u)", reference, masked));
            }

            string[] results;
            if (wide)
            {
                results = [.. Enumerable.Range(0, 4).Select(index => Temp("uint", $"{accumulators[index]} + {sads[index]}"))];
            }
            else
            {
                string Lane(int index) =>
                    $"((({accumulators[index / 2]} >> {index % 2 * 16}u) + {sads[index]}) & 0xFFFFu)";
                results =
                [
                    Temp("uint", $"{Lane(0)} | ({Lane(1)} << 16)"),
                    Temp("uint", $"{Lane(2)} | ({Lane(3)} << 16)"),
                ];
            }

            for (var index = 0; index < results.Length; index++)
            {
                StoreVector(destination + (uint)index, results[index]);
            }

            return true;
        }

        /// <summary>A 128-bit operand as four dwords: a VGPR or SGPR quad. A constant has no
        /// defined 128-bit reading, so it is refused.</summary>
        private bool TryRawSource128(
            Gen5ShaderInstruction instruction,
            int sourceIndex,
            out string[] dwords,
            out string error)
        {
            error = string.Empty;
            var operand = instruction.Sources[sourceIndex];
            switch (operand.Kind)
            {
                case Gen5OperandKind.VectorRegister:
                    dwords = [.. Enumerable.Range(0, 4).Select(index => Temp("uint", VectorExpression(operand.Value + (uint)index)))];
                    return true;
                case Gen5OperandKind.ScalarRegister:
                    dwords = [.. Enumerable.Range(0, 4).Select(index => Temp("uint", ScalarExpression(operand.Value + (uint)index)))];
                    return true;
                default:
                    dwords = [];
                    error = $"{instruction.Opcode}: a constant 128-bit accumulator (src{sourceIndex}) is not supported";
                    return false;
            }
        }

        /// <summary>V_PERM_B32: each result byte picks a byte of {S0, S1} (S1 is bytes 0-3),
        /// replicates the sign of byte 1/3/5/7 for selectors 8-11, or is 0x00 (12) or 0xFF (13+).</summary>
        private string EmitPermB32(Gen5ShaderInstruction instruction)
        {
            var bytes = Temp("ulong", $"((ulong)({RawSource(instruction, 0)}) << 32) | (ulong)({RawSource(instruction, 1)})");
            var selectors = Temp("uint", RawSource(instruction, 2));
            var result = new string[4];
            for (var index = 0; index < 4; index++)
            {
                var selector = Temp("uint", UnsignedField(selectors, index * 8, 8));
                // Selector 8 + k reads the top bit of byte 2k + 1, bit 16k + 15.
                var signBit = $"(uint)(({bytes} >> ((({selector} - 8u) & 3u) * 16u + 15u)) & 1ul)";
                var picked = Temp(
                    "uint",
                    $"{selector} >= 13u ? 0xFFu : {selector} == 12u ? 0u : {selector} >= 8u ? {signBit} * 0xFFu : " +
                    $"(uint)(({bytes} >> (({selector} & 7u) * 8u)) & 0xFFul)");
                result[index] = $"({picked} << {index * 8}u)";
            }

            return $"({string.Join(" | ", result)})";
        }

        // ---- 64-bit integer ----

        /// <summary>V_MAD_I64_I32: {carry, D.i64} = S0.i32 * S1.i32 + S2.i64 as a 65-bit signed
        /// sum. The carry is bit 64 of that sum — the sign of the exact result — formed from
        /// the operands' signs and the unsigned carry out of the low 64 bits.</summary>
        private string EmitMadI64I32(Gen5ShaderInstruction instruction, uint destination)
        {
            var product = Temp(
                "long",
                $"(long)as_type<int>({RawSource(instruction, 0)}) * (long)as_type<int>({RawSource(instruction, 1)})");
            var addend = Temp("ulong", RawSource64(instruction, 2));
            var wide = Temp("ulong", $"as_type<ulong>({product}) + {addend}");
            StoreCarryOut(
                instruction,
                $"({product} < 0l) != (as_type<long>({addend}) < 0l) != ({wide} < {addend})");
            StoreVector(destination + 1, $"(uint)({wide} >> 32)");
            return $"(uint){wide}";
        }

        /// <summary>V_ASHRREV_I64: D.i64 = S1.i64 >> S0[5:0], arithmetic.</summary>
        private string EmitArithmeticShiftRev64(Gen5ShaderInstruction instruction, uint destination)
        {
            var shift = Temp("uint", $"({RawSource(instruction, 0)}) & 63u");
            var shifted = Temp("ulong", $"as_type<ulong>(as_type<long>({RawSource64(instruction, 1)}) >> {shift})");
            StoreVector(destination + 1, $"(uint)({shifted} >> 32)");
            return $"(uint){shifted}";
        }

        // ---- VOP3 16-bit integer ----

        /// <summary>V_MUL_LO_U16 / V_MAD_U16 / V_MAD_I16 / V_MAD_I32_I16 and the 16-bit
        /// MIN3/MAX3/MED3: op_sel picks each 16-bit source half (Int16Source), the 16-bit
        /// results land in the op_sel[3] half (Int16Result), and clamp saturates to the
        /// result type.</summary>
        private string EmitInt16Vop3(Gen5ShaderInstruction instruction, uint destination, bool clamp)
        {
            var opcode = instruction.Opcode;
            var signed = opcode.EndsWith("I16", StringComparison.Ordinal);
            var a = Temp("uint", Int16Source(instruction, 0, signed));
            var b = Temp("uint", Int16Source(instruction, 1, signed));
            string Int(string value) => $"as_type<int>({value})";
            switch (opcode)
            {
                case "VMulLoU16":
                    return Int16Result(instruction, destination, clamp ? $"min({a} * {b}, 0xFFFFu)" : $"({a} * {b})");
                case "VMadU16":
                {
                    var c = Int16Source(instruction, 2, signed: false);
                    var sum = Temp("uint", $"{a} * {b} + ({c})");
                    return Int16Result(instruction, destination, clamp ? $"min({sum}, 0xFFFFu)" : sum);
                }
                case "VMadI16":
                {
                    var c = Int16Source(instruction, 2, signed: true);
                    var sum = Temp("int", $"{Int(a)} * {Int(b)} + {Int(c)}");
                    return Int16Result(instruction, destination, AsUInt(clamp ? $"clamp({sum}, -32768, 32767)" : sum));
                }
                case "VMadI32I16":
                {
                    var c = Temp("uint", RawSource(instruction, 2));
                    return clamp
                        ? AsUInt($"(int)clamp((long){Int(a)} * (long){Int(b)} + (long){Int(c)}, -2147483648l, 2147483647l)")
                        : $"(as_type<uint>({Int(a)} * {Int(b)}) + {c})";
                }
            }

            var c3 = Temp("uint", Int16Source(instruction, 2, signed));
            string Min(string x, string y) => signed ? AsUInt($"min({Int(x)}, {Int(y)})") : $"min({x}, {y})";
            string Max(string x, string y) => signed ? AsUInt($"max({Int(x)}, {Int(y)})") : $"max({x}, {y})";
            var value = opcode switch
            {
                "VMin3I16" or "VMin3U16" => Min(Min(a, b), c3),
                "VMax3I16" or "VMax3U16" => Max(Max(a, b), c3),
                _ => Max(Min(a, b), Min(Max(a, b), c3)),
            };
            return Int16Result(instruction, destination, value);
        }

        /// <summary>V_PACK_B32_F16: the two op_sel-selected f16 halves, abs/neg on their sign
        /// bits, as a bit move. Clamp, omod and a destination op_sel have no defined meaning
        /// for a 32-bit pack and are refused.</summary>
        private bool TryEmitPackB32F16(Gen5ShaderInstruction instruction, [NotNullWhen(true)] out string? value, out string error)
        {
            value = null;
            error = string.Empty;
            if (instruction.Control is Gen5Vop3Control control &&
                (control.Clamp || control.OutputModifier != 0 || (control.OperandSelect & ~3u) != 0))
            {
                error = $"{instruction.Opcode}: clamp, omod and op_sel beyond the two sources are not supported";
                return false;
            }

            var low = ReadMinimumHalfBits(instruction, 0);
            var high = ReadMinimumHalfBits(instruction, 1);
            value = $"({low} | ({high} << 16))";
            return true;
        }

        /// <summary>V_CVT_PKNORM_I16_F16 / _U16_F16: D = {norm(S1.f16), norm(S0.f16)}.</summary>
        private bool TryEmitPackNormF16(Gen5ShaderInstruction instruction, bool signed, [NotNullWhen(true)] out string? value, out string error)
        {
            value = null;
            error = string.Empty;
            if (instruction.Control is Gen5Vop3Control control &&
                ((control.OperandSelect & ~3u) != 0 || FloatOutputModifiers(instruction).OutputModifier != 0))
            {
                error = $"{instruction.Opcode}: omod and op_sel beyond the two sources are not supported";
                return false;
            }

            var low = NormalizeHalfBits(ReadMinimumHalfBits(instruction, 0), signed);
            var high = NormalizeHalfBits(ReadMinimumHalfBits(instruction, 1), signed);
            value = $"({low} | ({high} << 16))";
            return true;
        }

        // ---- VOP3P packed 16-bit integer ----

        /// <summary>
        /// The V_PK_* 16-bit integer ops. Each result lane reads the source halves its op_sel
        /// (low lane) or op_sel_hi (high lane) bit names, as EmitPackedI16Arithmetic does, and
        /// the hardware's clamp saturates each lane to the type's range. neg_lo/neg_hi negate
        /// floating-point inputs only and an integer op ignores them: the PS5 compiler leaves
        /// them set on v_pk_add_u16 and v_pk_max_i16 in Black Myth: Wukong's FSR3 passes, which
        /// refusing them dropped every frame.
        /// </summary>
        private bool TryEmitPackedInt16(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            [NotNullWhen(true)] out string? value,
            out string error)
        {
            value = null;
            error = string.Empty;
            var opcode = instruction.Opcode;
            var ternary = opcode is "VPkMadI16" or "VPkMadU16";
            var sources = new string[ternary ? 3 : 2];
            for (var index = 0; index < sources.Length; index++)
            {
                sources[index] = Temp("uint", RawSource(instruction, index));
            }

            string Lane(uint mask)
            {
                string Half(int index) => ((mask >> index) & 1) != 0
                    ? $"(({sources[index]} >> 16) & 0xFFFFu)"
                    : $"({sources[index]} & 0xFFFFu)";
                string Signed(int index) => $"(as_type<int>({Half(index)} << 16u) >> 16)";
                var clamp = control.Clamp;
                var lane = opcode switch
                {
                    "VPkMadI16" => clamp
                        ? $"as_type<uint>(clamp({Signed(0)} * {Signed(1)} + {Signed(2)}, -32768, 32767))"
                        : $"({Half(0)} * {Half(1)} + {Half(2)})",
                    "VPkMadU16" => clamp
                        ? $"min({Half(0)} * {Half(1)} + {Half(2)}, 0xFFFFu)"
                        : $"({Half(0)} * {Half(1)} + {Half(2)})",
                    "VPkMulLoU16" => clamp ? $"min({Half(0)} * {Half(1)}, 0xFFFFu)" : $"({Half(0)} * {Half(1)})",
                    "VPkAddU16" => clamp ? $"min({Half(0)} + {Half(1)}, 0xFFFFu)" : $"({Half(0)} + {Half(1)})",
                    "VPkSubU16" => clamp
                        ? $"({Half(0)} < {Half(1)} ? 0u : {Half(0)} - {Half(1)})"
                        : $"({Half(0)} - {Half(1)})",
                    // The rev shifts take the count from src0 and the value from src1.
                    "VPkLshlrevB16" => $"({Half(1)} << ({Half(0)} & 15u))",
                    "VPkLshrrevB16" => $"({Half(1)} >> ({Half(0)} & 15u))",
                    "VPkAshrrevI16" => $"as_type<uint>({Signed(1)} >> ({Half(0)} & 15u))",
                    "VPkMaxI16" => $"as_type<uint>(max({Signed(0)}, {Signed(1)}))",
                    "VPkMinI16" => $"as_type<uint>(min({Signed(0)}, {Signed(1)}))",
                    "VPkMaxU16" => $"max({Half(0)}, {Half(1)})",
                    _ => $"min({Half(0)}, {Half(1)})",
                };
                return $"({lane} & 0xFFFFu)";
            }

            value =$"({Lane(control.OpSelMask)} | ({Lane(control.OpSelHiMask)} << 16))";
            return true;
        }

        // ---- VOP3P dot products ----

        /// <summary>
        /// V_DOT2_F32_F16, V_DOT2_{I32_I16,U32_U16}, V_DOT4_{I32_I8,U32_U8},
        /// V_DOT8_{I32_I4,U32_U4}: the products of corresponding elements of S0 and S1 summed
        /// with the 32-bit S2. The dot2 forms read element 0 from the op_sel-selected and
        /// element 1 from the op_sel_hi-selected half, like a packed lane, and the f16 form
        /// takes neg_lo/neg_hi on those elements; S2 must use the default selection and no
        /// negate. The byte and nibble forms have no defined op_sel or negate, so only the
        /// defaults are accepted. Clamp saturates the integer forms to their 32-bit range and
        /// the f16 form to [0, 1].
        /// </summary>
        private bool TryEmitDot(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            [NotNullWhen(true)] out string? value,
            out string error)
        {
            value = null;
            error = string.Empty;
            var opcode = instruction.Opcode;
            var dot2 = opcode.StartsWith("VDot2", StringComparison.Ordinal);
            var half = opcode == "VDot2F32F16";
            if ((control.OpSelMask & 4) != 0 || (control.OpSelHiMask & 4) == 0 ||
                ((control.NegLoMask | control.NegHiMask) & 4) != 0)
            {
                error = $"{opcode}: op_sel/op_sel_hi/neg on the 32-bit accumulator (src2) are not supported";
                return false;
            }

            if (!half && (control.NegLoMask | control.NegHiMask) != 0)
            {
                error = $"{opcode}: neg_lo/neg_hi on an integer dot product are not supported";
                return false;
            }

            if (!dot2 && (control.OpSelMask != 0 || control.OpSelHiMask != 7))
            {
                error = $"{opcode}: op_sel/op_sel_hi other than the default are not supported";
                return false;
            }

            if (half)
            {
                var sources = PackedSources(instruction, control, 2);
                var products = Enumerable.Range(0, 2).Select(element =>
                    $"{PackedF16Operand(instruction, control, sources, 0, element == 1)} * " +
                    $"{PackedF16Operand(instruction, control, sources, 1, element == 1)}").ToArray();
                // The association of EmitDot2cF32F16: the two exact products, then the addend.
                var dot = $"({products[0]} + {products[1]} + as_type<float>({RawSource(instruction, 2)}))";
                value = AsUInt(control.Clamp ? ClampToUnitInterval(dot) : dot);
                return true;
            }

            var signed = opcode.Contains("I32", StringComparison.Ordinal);
            var left = Temp("uint", RawSource(instruction, 0));
            var right = Temp("uint", RawSource(instruction, 1));
            var accumulator = Temp("uint", RawSource(instruction, 2));
            var width = opcode.EndsWith("16", StringComparison.Ordinal) ? 16
                : opcode.EndsWith("8", StringComparison.Ordinal) ? 8 : 4;
            var terms = new List<string>();
            for (var element = 0; element < 32 / width; element++)
            {
                string Element(string source, int index)
                {
                    // dot2: element 0 from the op_sel half, element 1 from the op_sel_hi half.
                    var offset = dot2
                        ? ((((element == 0 ? control.OpSelMask : control.OpSelHiMask) >> index) & 1) != 0 ? 16 : 0)
                        : element * width;
                    return signed
                        ? $"(long){SignedField(source, offset, width)}"
                        : $"(long){UnsignedField(source, offset, width)}";
                }

                terms.Add($"{Element(left, 0)} * {Element(right, 1)}");
            }

            var sum = Temp(
                "long",
                $"{string.Join(" + ", terms)} + {(signed ? $"(long)as_type<int>({accumulator})" : $"(long){accumulator}")}");
            value = !control.Clamp
                ? $"(uint){sum}"
                : signed
                    ? $"as_type<uint>((int)clamp({sum}, -2147483648l, 2147483647l))"
                    : $"(uint)clamp({sum}, 0l, 4294967295l)";
            return true;
        }

        // ---- f16 class ----

        /// <summary>V_CMP_CLASS_F16 on the f16 bits (after abs/neg): unlike the f32 path, a
        /// signalling NaN (quiet bit 9 clear, mask bit 0) and a quiet NaN (mask bit 1) are
        /// told apart.</summary>
        private string EmitCompareClassF16(Gen5ShaderInstruction instruction)
        {
            var bits = ReadMinimumHalfBits(instruction, 0);
            var mask = Temp("uint", RawSource(instruction, 1));
            var exponent = Temp("uint", $"{bits} & 0x7C00u");
            var fraction = Temp("uint", $"{bits} & 0x3FFu");
            var negative = Temp("bool", $"({bits} & 0x8000u) != 0u");
            var nan = Temp("bool", $"{exponent} == 0x7C00u && {fraction} != 0u");
            var infinite = Temp("bool", $"{exponent} == 0x7C00u && {fraction} == 0u");
            var zero = Temp("bool", $"{exponent} == 0u && {fraction} == 0u");
            var subnormal = Temp("bool", $"{exponent} == 0u && {fraction} != 0u");
            var normal = Temp("bool", $"{exponent} != 0u && {exponent} != 0x7C00u");
            string Bit(int index) => $"(({mask} >> {index}u) & 1u) != 0u";
            return Temp(
                "bool",
                $"({Bit(0)} && {nan} && ({fraction} & 0x200u) == 0u) || " +
                $"({Bit(1)} && {nan} && ({fraction} & 0x200u) != 0u) || " +
                $"({Bit(2)} && {infinite} && {negative}) || " +
                $"({Bit(3)} && {normal} && {negative}) || " +
                $"({Bit(4)} && {subnormal} && {negative}) || " +
                $"({Bit(5)} && {zero} && {negative}) || " +
                $"({Bit(6)} && {zero} && !{negative}) || " +
                $"({Bit(7)} && {subnormal} && !{negative}) || " +
                $"({Bit(8)} && {normal} && !{negative}) || " +
                $"({Bit(9)} && {infinite} && !{negative})");
        }

        // ---- VGPR swaps ----

        /// <summary>V_SWAP_B32: vdst takes src0 and the src0 VGPR the old vdst, both read
        /// before either is written. An untyped move: no modifiers.</summary>
        private bool TryEmitSwap(Gen5ShaderInstruction instruction, out string error)
        {
            error = string.Empty;
            if (instruction.Control is not null ||
                instruction.Sources is not [{ Kind: Gen5OperandKind.VectorRegister } source, ..] ||
                instruction.Destinations is not
                    [{ Kind: Gen5OperandKind.VectorRegister } destination, { Kind: Gen5OperandKind.VectorRegister } other] ||
                other.Value != source.Value)
            {
                error = "v_swap_b32 expects two plain VGPR operands and no modifiers";
                return false;
            }

            var incoming = Temp("uint", VectorExpression(source.Value));
            var outgoing = Temp("uint", VectorExpression(destination.Value));
            StoreVector(destination.Value, incoming);
            StoreVector(source.Value, outgoing);
            return true;
        }

        /// <summary>V_SWAPREL_B32: swaps VGPR[src0 + M0[9:0]] with VGPR[vdst + M0[25:16]],
        /// through the same run-time register access as the relative moves.</summary>
        private bool TryEmitSwapRelative(Gen5ShaderInstruction instruction, out string error)
        {
            error = string.Empty;
            if (instruction.Control is not null ||
                instruction.Sources is not [{ Kind: Gen5OperandKind.VectorRegister } source, ..] ||
                instruction.Destinations is not [{ Kind: Gen5OperandKind.VectorRegister } destination, ..])
            {
                error = "v_swaprel_b32 expects two plain VGPR operands and no modifiers";
                return false;
            }

            var m0 = Temp("uint", ScalarExpression(M0Register));
            var sourceIndex = Temp(
                "uint",
                $"({source.Value}u + ({m0} & 0x3FFu)) & {VectorRegisterFileCount - 1}u");
            var destinationIndex = Temp(
                "uint",
                $"({destination.Value}u + (({m0} >> 16u) & 0x3FFu)) & {VectorRegisterFileCount - 1}u");
            string Read(string index)
            {
                if (!_options.BoundedRelativeMoves)
                {
                    return Temp("uint", $"v[{index}]");
                }

                var value = $"t{_nextTemp++}";
                Line($"uint {value};");
                Line($"{DynamicVectorRead}({value}, {index});");
                return value;
            }

            var incoming = Read(sourceIndex);
            var outgoing = Read(destinationIndex);
            string Write(string index, string value) => _options.BoundedRelativeMoves
                ? $"{DynamicVectorWrite}({index}, {value});"
                : $"v[{index}] = {value};";
            Line(ExecGuarded(Write(destinationIndex, incoming)));
            Line(ExecGuarded(Write(sourceIndex, outgoing)));
            return true;
        }

        // ---- scalar ----

        /// <summary>S_GETREG_B32 of HW_REG_MODE: FP_ROUND and FP_DENORM (bits 7:0) are the
        /// launch FLOAT_MODE, DX10_CLAMP (bit 8) and IEEE (bit 9) the launch state; the
        /// exception enables and the rest read 0. Any other register is refused.</summary>
        private bool TryEmitGetreg(Gen5ShaderInstruction instruction, out string error)
        {
            error = string.Empty;
            const uint ModeRegister = 1;
            var simm16 = instruction.Sources.Count > 0
                ? instruction.Sources[0].Value & 0xFFFFu
                : instruction.Words[0] & 0xFFFFu;
            var id = simm16 & 0x3Fu;
            var offset = (simm16 >> 6) & 0x1Fu;
            var size = ((simm16 >> 11) & 0x1Fu) + 1u;
            if (id != ModeRegister)
            {
                error = $"s_getreg_b32 of hardware register {id} is not supported (only HW_REG_MODE)";
                return false;
            }

            var field = (uint)((((1ul << (int)size) - 1ul) << (int)offset) & 0xFFFFFFFFul);

            if (instruction.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister } destination, ..])
            {
                error = "s_getreg_b32 expects a scalar destination";
                return false;
            }

            var mode = (_request.FloatMode & 0xFFu) | (_request.Dx10Clamp ? 0x100u : 0u) | (_request.IeeeMode ? 0x200u : 0u);
            StoreScalar(destination.Value, FormatUInt((mode & field) >> (int)offset));
            return true;
        }

        /// <summary>
        /// S_MOVRELS / S_MOVRELD / S_MOVRELSD_2 over the SGPR file: the register number in the
        /// instruction plus M0 (or one of M0's halves). The access is a switch over the 128
        /// registers rather than an s[] subscript, so every register stays named — the mask
        /// ballot and register allocation passes see each one — and a write that lands on VCC
        /// or EXEC goes through StoreScalar, which keeps the per-lane view and the EXEC lane
        /// clamp in step. A number past the file reads 0 and drops the write.
        /// </summary>
        private bool TryEmitScalarMoveRelative(Gen5ShaderInstruction instruction, out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            var wide = opcode.EndsWith("B64", StringComparison.Ordinal);
            if (instruction.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister } destination, ..] ||
                instruction.Sources.Count == 0)
            {
                error = $"{opcode} expects a scalar destination and source";
                return false;
            }

            var source = instruction.Sources[0];
            var relativeSource = opcode is not ("SMovreldB32" or "SMovreldB64");
            if (relativeSource && source.Kind != Gen5OperandKind.ScalarRegister)
            {
                error = $"{opcode} source must be a scalar register";
                return false;
            }

            var m0 = Temp("uint", ScalarExpression(M0Register));
            var sourceOffset = opcode == "SMovrelsd2B32" ? Temp("uint", $"{m0} & 0x3FFu") : m0;
            var destinationOffset = opcode == "SMovrelsd2B32" ? Temp("uint", $"({m0} >> 16u) & 0x3FFu") : m0;

            string low;
            string? high = null;
            if (relativeSource)
            {
                var index = Temp("uint", $"{source.Value}u + {sourceOffset}");
                low = $"t{_nextTemp++}";
                Line($"uint {low} = 0u;");
                if (wide)
                {
                    high = $"t{_nextTemp++}";
                    Line($"uint {high} = 0u;");
                }

                Line($"switch ({index}) {{");
                for (uint register = 0; register < ScalarRegisterFileCount; register++)
                {
                    Line(wide
                        ? $"case {register}u: {low} = {ScalarExpression(register)}; {high} = {ScalarExpression(register + 1)}; break;"
                        : $"case {register}u: {low} = {ScalarExpression(register)}; break;");
                }

                Line("default: break;");
                Line("}");
            }
            else if (wide)
            {
                var pair = Temp("ulong", RawSource64(instruction, 0));
                low = Temp("uint", $"(uint){pair}");
                high = Temp("uint", $"(uint)({pair} >> 32)");
            }
            else
            {
                low = Temp("uint", RawSource(instruction, 0));
            }

            if (opcode is "SMovrelsB32" or "SMovrelsB64")
            {
                StoreScalar(destination.Value, low);
                if (high is not null)
                {
                    StoreScalar(destination.Value + 1, high);
                }

                return true;
            }

            var target = Temp("uint", $"{destination.Value}u + {destinationOffset}");
            Line($"switch ({target}) {{");
            for (uint register = 0; register < ScalarRegisterFileCount; register++)
            {
                Line($"case {register}u: {{");
                _indent++;
                StoreScalar(register, low);
                if (high is not null)
                {
                    StoreScalar(register + 1, high);
                }

                Line("break;");
                _indent--;
                Line("}");
            }

            Line("default: break;");
            Line("}");
            return true;
        }

        /// <summary>S_ANDN1_WREXEC / S_ANDN2_WREXEC (B32 and B64): EXEC = ~S0 & EXEC or
        /// S0 & ~EXEC, then D = the new EXEC and SCC = EXEC != 0. Unlike SAVEEXEC the
        /// destination takes the result, after the same lane clamp the saveexec forms apply.</summary>
        private void EmitWriteExec(Gen5ShaderInstruction instruction, uint destination)
        {
            var andn1 = instruction.Opcode.StartsWith("SAndn1", StringComparison.Ordinal);
            if (instruction.Opcode.EndsWith("B64", StringComparison.Ordinal))
            {
                var source = Temp("ulong", RawSource64(instruction, 0));
                var oldExec = Temp("ulong", Scalar64Expression(ExecLoRegister));
                var mask = Temp(
                    "ulong",
                    ClampExecToLanes64(andn1 ? $"(~{source} & {oldExec})" : $"({source} & ~{oldExec})"));
                Line($"s[{ExecLoRegister}] = (uint){mask};");
                Line(IsNarrowWave64
                    ? $"s[{ExecHiRegister}] = 0u;"
                    : $"s[{ExecHiRegister}] = (uint)({mask} >> 32);");
                Line($"exec = (({mask} >> sharpemu_lane) & 1ul) != 0ul;");
                Line($"scc = {mask} != 0ul;");
                StoreScalar64(destination, mask);
                return;
            }

            var left = Temp("uint", RawSource(instruction, 0));
            var old = Temp("uint", $"s[{ExecLoRegister}]");
            var narrow = Temp("uint", ClampExecToLanes(andn1 ? $"(~{left} & {old})" : $"({left} & ~{old})"));
            Line($"s[{ExecLoRegister}] = {narrow};");
            Line($"s[{ExecHiRegister}] = 0u;");
            Line($"exec = (({narrow} >> sharpemu_lane) & 1u) != 0u;");
            Line($"scc = {narrow} != 0u;");
            StoreScalar(destination, narrow);
        }
    }
}
