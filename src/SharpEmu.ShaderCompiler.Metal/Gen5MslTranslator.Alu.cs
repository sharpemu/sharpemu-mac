// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const string TauLiteral = "6.2831853071795862f";

        // ---- vector ALU ----

        private bool TryEmitVectorAlu(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            _sixteenBitSdwaResult = false;
            if (instruction.Opcode == "VNop")
            {
                return true;
            }

            if (instruction.Control is Gen5SdwaControl sdwa &&
                (sdwa.Source0Select == 7 ||
                 sdwa.Source1Select == 7 ||
                 sdwa.DestinationSelect == 7 ||
                 sdwa.DestinationUnused == 3))
            {
                error = $"reserved SDWA selector/modifier in {instruction.Opcode}";
                return false;
            }

            if (instruction.Control is Gen5DppControl dppControl &&
                !IsSupportedDppControl(dppControl.Control))
            {
                error = $"unsupported DPP16 control 0x{dppControl.Control:X3}";
                return false;
            }

            if (IsFloat64Opcode(instruction.Opcode))
            {
                return TryEmitFloat64(instruction, out error);
            }

            if (instruction.Opcode.StartsWith("VCmp", StringComparison.Ordinal))
            {
                return TryEmitVectorCompare(instruction, out error);
            }

            switch (instruction.Opcode)
            {
                case "VReadfirstlaneB32":
                {
                    if (instruction.Destinations.Count == 0 ||
                        instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister ||
                        instruction.Sources.Count == 0)
                    {
                        error = "invalid read-first-lane operands";
                        return false;
                    }

                    // Under the single-lane model the "first active lane" is
                    // always this lane. Real lanes broadcast from the
                    // first guest-active lane (the ballot of EXEC), matching
                    // the SPIR-V translator — SPIR-V's own BroadcastFirst uses
                    // the first host-active invocation, which may be a lane the
                    // guest has masked off.
                    var value = RawSource(instruction, 0);
                    if (IsSingleLaneStage)
                    {
                        StoreScalar(instruction.Destinations[0].Value, Temp("uint", value));
                        return true;
                    }

                    if (IsWave64)
                    {
                        StoreScalar(
                            instruction.Destinations[0].Value,
                            EmitWave64ReadFirstLane(value));
                        return true;
                    }

                    var mask = Temp("uint", "sharpemu_ballot(exec)");
                    var firstLane = Temp("uint", $"{mask} == 0u ? 0u : (uint)ctz({mask})");
                    StoreScalar(
                        instruction.Destinations[0].Value,
                        Temp("uint", ShuffleLane(value, firstLane)));
                    return true;
                }
                case "VReadlaneB32":
                {
                    if (instruction.Destinations.Count == 0 ||
                        instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister)
                    {
                        error = "VReadlaneB32 expects scalar destination";
                        return false;
                    }

                    if (LaneSpillPlan() is { } spills &&
                        spills.CopyReads.TryGetValue(instruction.Pc, out var spillCopy))
                    {
                        // A reload of a spilled SGPR: every path parked it with a
                        // constant-lane writelane, which kept a copy in every thread.
                        StoreScalar(instruction.Destinations[0].Value, $"sharpemu_spill{spillCopy}");
                        return true;
                    }

                    var value = RawSource(instruction, 0);
                    if (IsWave64)
                    {
                        // Lanes 32..63 live in the other simdgroup. Reading lane & 31 of
                        // its own half gave each half a different scalar, and a loop
                        // bounded by it ran a different trip count per half, so the
                        // barriers inside it never met (a GPU hang).
                        var wideLane = Temp("uint", $"({RawSource(instruction, 1)}) & 63u");
                        StoreScalar(
                            instruction.Destinations[0].Value,
                            EmitWave64ReadLane(value, wideLane));
                        return true;
                    }

                    if (IsNarrowWave64)
                    {
                        // The guest wave has 64 lanes but only the low 32 exist here; the
                        // upper half is architecturally inactive and reads as zero.
                        var narrowLane = Temp("uint", $"({RawSource(instruction, 1)}) & 63u");
                        StoreScalar(
                            instruction.Destinations[0].Value,
                            Temp("uint", $"{narrowLane} >= 32u ? 0u : {ShuffleLane(value, narrowLane)}"));
                        return true;
                    }

                    var lane = Temp("uint", $"({RawSource(instruction, 1)}) & 31u");
                    if (_stage == Gen5MslStage.Pixel && !IsSingleLaneStage)
                    {
                        // A fragment that was never launched has no thread to shuffle from
                        // (undefined in MSL); its guest lane holds the scan value below it.
                        // Ghost of Tsushima reads a wave minimum from lane 31 and loops until
                        // it reaches 255, so a stray value there hung the GPU.
                        lane = Temp("uint", $"sharpemu_present_lane({lane})");
                    }

                    StoreScalar(
                        instruction.Destinations[0].Value,
                        Temp("uint", ShuffleLane(value, lane)));
                    return true;
                }
                case "VMovrelsB32":
                case "VMovreldB32":
                case "VMovrelsdB32":
                case "VMovrelsd2B32":
                    return TryEmitMoveRelative(instruction, out error);
                case "VSwapB32":
                    return TryEmitSwap(instruction, out error);
                case "VSwaprelB32":
                    return TryEmitSwapRelative(instruction, out error);
                case "VWritelaneB32":
                {
                    // vdst[lane(src1)] = src0; a writelane lands regardless of EXEC.
                    var destination = DestinationVector(instruction);
                    var source = RawSource(instruction, 0);
                    var lane = RawSource(instruction, 1);
                    // Under wave64 sharpemu_lane is the full 0..63 index, so only the
                    // one matching lane writes; wave32 has 32 lanes.
                    // A narrow wave64 masks with 63 as well: a lane >= 32 matches no thread, so the
                    // write is a no-op, as on a wave whose upper half is inactive.
                    var laneMask = IsWave64 || IsNarrowWave64 ? "63u" : "31u";
                    StoreVector(
                        destination,
                        $"(sharpemu_lane == (({lane}) & {laneMask})) ? ({source}) : v[{destination}]",
                        guardWithExec: false);
                    if (LaneSpillPlan() is { } spills &&
                        spills.CopyWrites.TryGetValue(instruction.Pc, out var spillCopy))
                    {
                        // The parked value is scalar, so every thread keeps it.
                        Line($"sharpemu_spill{spillCopy} = {source};");
                    }

                    return true;
                }
                case "VCndmaskB32":
                {
                    // dst = mask-bit(lane) ? src1 : src0. VOP3 abs/neg act on the
                    // sign bit, matching the SPIR-V translator; the mask is VCC
                    // for VOP2 and an explicit SGPR operand for VOP3.
                    var mask = instruction.Sources.Count > 2
                        ? MaskBitExpression(instruction.Sources[2])
                        : "vcc";
                    var destination = DestinationVector(instruction);
                    var selected = Temp(
                        "uint",
                        $"({mask}) ? ({SignModifiedSource(instruction, 1)}) : ({SignModifiedSource(instruction, 0)})");
                    if (instruction.Control is Gen5DppControl dpp)
                    {
                        selected = Temp("uint", $"({EmitDppWriteEnabled(dpp)}) ? {selected} : v[{destination}]");
                    }

                    // An SDWA select writes one byte or word of the destination and zeroes,
                    // sign-extends or preserves the rest, like every other SDWA result.
                    if (instruction.Control is Gen5SdwaControl { ScalarDestination: null } sdwaSelect)
                    {
                        selected = ApplySdwaDestination(sdwaSelect, selected, $"v[{destination}]");
                    }

                    StoreVector(destination, selected);
                    return true;
                }
            }

            return TryEmitVectorValue(instruction, out error);
        }

        private bool TryEmitVectorValue(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            var destination = DestinationVector(instruction);
            // The clamp bit saturates an integer result to the type's range (ISA 6.5).
            // The add/sub family and V_MAD_U32_U16 implement it; the other integer ops the
            // hardware can clamp are refused rather than silently wrapped.
            var intClamp = instruction.Control is Gen5Vop3Control { Clamp: true } or Gen5SdwaControl { Clamp: true };
            if (intClamp)
            {
                if (instruction.Opcode is "VMulU32U24" or "VMulI32I24" or "VMadU32U24" or "VMadI32I24" or
                    "VSadU32" or "VMadU64U32" or "VMadI64I32" or
                    "VMsadU8" or "VQsadPkU16U8" or "VMqsadPkU16U8" or "VMqsadU32U8")
                {
                    error = $"{instruction.Opcode}: clamped integer results are not supported";
                    return false;
                }

                // An SDWA destination select takes its field from the clamped 32-bit result
                // (ISA 6.5 clamps to the op's type; 13.3.7 DST_SEL then places the low bits),
                // which the common write-back below does.
            }

            string? expression = instruction.Opcode switch
            {
                "VMovB32" => SignModifiedSource(instruction, 0),

                // ---- float arithmetic ----
                "VAddF32" => FloatResult(instruction, $"{F(instruction, 0)} + {F(instruction, 1)}"),
                "VAddF16" => Float16Result(
                    instruction,
                    destination,
                    $"{F16(instruction, 0)} + {F16(instruction, 1)}"),
                "VSubF32" => FloatResult(instruction, $"{F(instruction, 0)} - {F(instruction, 1)}"),
                "VSubrevF32" => FloatResult(instruction, $"{F(instruction, 1)} - {F(instruction, 0)}"),
                "VSubF16" => Float16Result(
                    instruction,
                    destination,
                    $"{F16(instruction, 0)} - {F16(instruction, 1)}"),
                "VSubrevF16" => Float16Result(
                    instruction,
                    destination,
                    $"{F16(instruction, 1)} - {F16(instruction, 0)}"),
                "VMulF32" => FloatResult(instruction, $"{F(instruction, 0)} * {F(instruction, 1)}"),
                "VMulF16" => Float16Result(
                    instruction,
                    destination,
                    $"{F16(instruction, 0)} * {F16(instruction, 1)}"),
                "VMinF32" or "VMaxF32" or "VMin3F32" or "VMax3F32" or "VMed3F32" => MinMaxF32Result(instruction),
                "VMinF16" or "VMaxF16" or "VMin3F16" or "VMax3F16" or "VMed3F16" => MinMaxF16Result(instruction, destination),
                // The decoder normalizes mk/ak literal placement, so every MAD/FMA form reads
                // (src0, src1, src2). The FMA forms round once; V_MAD_F32, V_MADMK/MADAK_F32
                // and V_MAC_F32 (gfx10.1) round the product before the add.
                "VFmaF32" or "VFmaAkF32" or "VFmaMkF32" =>
                    FloatResult(instruction, $"fma({F(instruction, 0)}, {F(instruction, 1)}, {F(instruction, 2)})"),
                "VMadF32" or "VMadAkF32" or "VMadMkF32" => EmitUnfusedMultiplyAdd(instruction, F(instruction, 2)),
                "VFmaF16" or "VFmaMkF16" or "VFmaAkF16" => Float16Result(
                    instruction,
                    destination,
                    $"fma({F16(instruction, 0)}, {F16(instruction, 1)}, {F16(instruction, 2)})"),
                "VFmacF32" =>
                    FloatResult(instruction, $"fma({F(instruction, 0)}, {F(instruction, 1)}, as_type<float>(v[{destination}]))"),
                "VMacF32" => EmitUnfusedMultiplyAdd(instruction, $"as_type<float>(v[{destination}])"),
                // DX9 legacy multiplies: a zero operand makes the product +0 (see EmitLegacyMultiply).
                // V_MULLIT_F32's ISA text only adds "specific INF, NaN, overflow rules" that it
                // does not spell out; it is the legacy multiply here.
                "VMulLegacyF32" or "VMullitF32" => EmitLegacyMultiply(instruction, null),
                "VMadLegacyF32" => EmitLegacyMultiply(instruction, F(instruction, 2)),
                "VPkFmacF16" => EmitPackedF16Fmac(instruction, destination),
                "VDot2cF32F16" => EmitDot2cF32F16(instruction, destination),
                "VFloorF32" => FloatResult(instruction, $"floor({F(instruction, 0)})"),
                "VCeilF32" => FloatResult(instruction, $"ceil({F(instruction, 0)})"),
                "VTruncF32" => FloatResult(instruction, $"trunc({F(instruction, 0)})"),
                "VRndneF32" => FloatResult(instruction, $"rint({F(instruction, 0)})"),
                "VFractF32" => EmitFractF32(instruction),
                // fast::sqrt exceeded the one-ULP budget on input 0x5E606496. Keep
                // square root precise even when the reciprocal fast paths are enabled.
                "VSqrtF32" => InvalidNanF32Result(instruction, "sqrt"),
                "VRsqF32" => InvalidNanF32Result(instruction, _options.FastReciprocals ? "fast::rsqrt" : "rsqrt"),
                "VRcpF16" => Float16Reciprocal(instruction, destination),
                "VSqrtF16" => Float16SquareRoot(instruction, destination),
                "VLogF16" => Float16Logarithm(instruction, destination),
                "VExpF16" => Float16Result(instruction, destination, $"exp2({F16(instruction, 0)})"),
                "VFloorF16" => Float16Result(instruction, destination, $"floor({F16(instruction, 0)})"),
                "VCeilF16" => Float16Result(instruction, destination, $"ceil({F16(instruction, 0)})"),
                "VTruncF16" => Float16Result(instruction, destination, $"trunc({F16(instruction, 0)})"),
                "VRndneF16" => Float16Result(instruction, destination, $"rint({F16(instruction, 0)})"),
                "VFractF16" => Float16Result(instruction, destination, $"fract({F16(instruction, 0)})"),
                "VSinF16" => Float16Result(instruction, destination, $"sin({F16(instruction, 0)} * {TauLiteral})"),
                "VCosF16" => Float16Result(instruction, destination, $"cos({F16(instruction, 0)} * {TauLiteral})"),
                "VRsqF16" => Float16Result(
                    instruction,
                    destination,
                    $"rsqrt({F16(instruction, 0)})"),
                // F32 denormal inputs reach rcp/rsq/sqrt/log/exp as a signed zero, as
                // on the guest; the IEEE-flag reciprocal keeps the unflushed input.
                "VRcpF32" => FloatResult(
                    instruction,
                    _options.FastReciprocals
                        ? $"fast::divide(1.0f, sharpemu_flush_f32_denorm({F(instruction, 0)}))"
                        : $"(1.0f / sharpemu_flush_f32_denorm({F(instruction, 0)}))"),
                "VRcpIflagF32" => FloatResult(instruction, $"(1.0f / {F(instruction, 0)})"),
                "VLogF32" => InvalidNanF32Result(instruction, "fast::log2"),
                "VExpF32" => FloatResult(instruction, $"fast::exp2(sharpemu_flush_f32_denorm({F(instruction, 0)}))"),
                // GCN sin/cos take revolutions; reduce to one cycle first, then mirror
                // the SPIR-V Tau prescale. Only sine keeps the sign of a zero input.
                "VSinF32" => InvalidNanF32Result(instruction, "sin", cycle: "true"),
                "VCosF32" => InvalidNanF32Result(instruction, "cos", cycle: "false"),
                "VLdexpF32" =>
                    FloatResult(instruction, $"ldexp({F(instruction, 0)}, as_type<int>({RawSource(instruction, 1)}))"),
                // D.f16 = S0.f16 * 2^S1.i16. Past +-100 the f16 result is already 0 or infinity, so
                // the exponent is clamped there and the f32 scaling stays exact.
                "VLdexpF16" => Float16Result(
                    instruction,
                    destination,
                    $"({F16(instruction, 0)} * as_type<float>((uint)(clamp(as_type<int>({Int16Source(instruction, 1, signed: true)}), -100, 100) + 127) << 23))"),
                "VFrexpMantF32" => EmitFrexpF32(instruction, mantissa: true),
                "VFrexpExpI32F32" => EmitFrexpF32(instruction, mantissa: false),
                "VFrexpMantF16" => EmitFrexpF16(instruction, destination, mantissa: true),
                "VFrexpExpI16F16" => EmitFrexpF16(instruction, destination, mantissa: false),

                // ---- conversions ----
                "VCvtF32I32" => FloatResult(instruction, $"(float)as_type<int>({RawSource(instruction, 0)})"),
                "VCvtF32U32" => FloatResult(instruction, $"(float)({RawSource(instruction, 0)})"),
                "VCvtU32F32" => SaturatedFloatToUint(Temp("float", F(instruction, 0))),
                // gfx10 16-bit results keep the other half of the VGPR (LLVM zeroesHigh16BitsOfDest).
                "VCvtU16F16" => Int16Result(
                    instruction,
                    destination,
                    $"(uint)clamp(trunc(isnan({F16(instruction, 0)}) ? 0.0f : {F16(instruction, 0)}), 0.0f, 65535.0f)"),
                "VCvtI32F32" => SaturatedFloatToInt(Temp("float", F(instruction, 0))),
                // u16/i16 -> f16 rounds to nearest even (65520 and up become +inf, as IEEE does).
                "VCvtF16U16" => Float16Result(instruction, destination, $"(float)({Int16Source(instruction, 0, signed: false)})"),
                "VCvtF16I16" => Float16Result(
                    instruction,
                    destination,
                    $"(float)as_type<int>({Int16Source(instruction, 0, signed: true)})"),
                "VCvtI16F16" => EmitCvtI16F16(instruction, destination),
                "VCvtNormI16F16" => Int16Result(
                    instruction,
                    destination,
                    NormalizeHalfBits(ReadMinimumHalfBits(instruction, 0), signed: true)),
                "VCvtNormU16F16" => Int16Result(
                    instruction,
                    destination,
                    NormalizeHalfBits(ReadMinimumHalfBits(instruction, 0), signed: false)),
                "VSatPkU8I16" => EmitSaturatePackU8(instruction),
                // RPI is floor(x + 0.5) (round to nearest, ties up); FLR is floor(x).
                "VCvtRpiI32F32" => SaturatedFloatToInt(Temp("float", $"floor({F(instruction, 0)} + 0.5f)")),
                "VCvtFlrI32F32" => SaturatedFloatToInt(Temp("float", $"floor({F(instruction, 0)})")),
                "VCvtF32Ubyte0" => FloatResult(instruction, $"(float)(({RawSource(instruction, 0)}) & 0xFFu)"),
                "VCvtF32Ubyte1" => FloatResult(instruction, $"(float)((({RawSource(instruction, 0)}) >> 8) & 0xFFu)"),
                "VCvtF32Ubyte2" => FloatResult(instruction, $"(float)((({RawSource(instruction, 0)}) >> 16) & 0xFFu)"),
                "VCvtF32Ubyte3" => FloatResult(instruction, $"(float)((({RawSource(instruction, 0)}) >> 24) & 0xFFu)"),
                // Clamp and omod apply to the f32 result before it is rounded to f16. The half
                // goes to the op_sel[3]-selected half and, as for every gfx10 16-bit result,
                // the other half of the VGPR is kept.
                "VCvtF16F32" => Float16Result(instruction, destination, F(instruction, 0)),
                // The source is a half: an SDWA or VOP3 negate flips its sign bit 15, not
                // bit 31 of the register, and VOP3 op_sel can pick the high half.
                "VCvtF32F16" => FloatResult(instruction, F16(instruction, 0)),
                "VCvtOffF32I4" =>
                    FloatResult(instruction, $"sharpemu_off_i4_table[({RawSource(instruction, 0)}) & 15u]"),
                "VCvtPkU8F32" =>
                    EmitCvtPkU8F32(instruction),
                "VCvtPkrtzF16F32" =>
                    EmitCvtPkrtzF16F32(instruction),
                "VCvtPkU16U32" =>
                    $"(min({RawSource(instruction, 0)}, 65535u) | (min({RawSource(instruction, 1)}, 65535u) << 16u))",
                "VCvtPkI16I32" =>
                    $"((as_type<uint>(clamp(as_type<int>({RawSource(instruction, 0)}), -32768, 32767)) & 0xFFFFu) | " +
                    $"(as_type<uint>(clamp(as_type<int>({RawSource(instruction, 1)}), -32768, 32767)) << 16u))",
                "VCvtPknormI16F32" =>
                    $"pack_float_to_snorm2x16(float2({F(instruction, 0)}, {F(instruction, 1)}))",
                "VCvtPknormU16F32" =>
                    $"pack_float_to_unorm2x16(float2({F(instruction, 0)}, {F(instruction, 1)}))",

                // ---- integer arithmetic ----
                "VAddU32" or "VAddI32" =>
                    intClamp
                        ? SaturatedAddSub(instruction, subtract: false, reverse: false, signed: IsSignedNoCarryAddSub(instruction))
                        : $"(({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)}))",
                "VSubU32" or "VSubI32" =>
                    intClamp
                        ? SaturatedAddSub(instruction, subtract: true, reverse: false, signed: IsSignedNoCarryAddSub(instruction))
                        : $"(({RawSource(instruction, 0)}) - ({RawSource(instruction, 1)}))",
                "VSubrevU32" or "VSubrevI32" =>
                    intClamp
                        ? SaturatedAddSub(instruction, subtract: true, reverse: true, signed: IsSignedNoCarryAddSub(instruction))
                        : $"(({RawSource(instruction, 1)}) - ({RawSource(instruction, 0)}))",
                "VMulI32I24" =>
                    EmitSignedMultiply24(instruction),
                // The 24-bit product wraps in 32 bits before the addend joins it.
                "VMadI32I24" =>
                    $"({EmitSignedMultiply24(instruction)} + ({RawSource(instruction, 2)}))",
                "VMulLoU32" or "VMulLoI32" =>
                    $"(({RawSource(instruction, 0)}) * ({RawSource(instruction, 1)}))",
                // D.u = S0.u[23:0] * S1.u[23:0]: bits 31:24 of each source are ignored.
                "VMulU32U24" =>
                    $"((({RawSource(instruction, 0)}) & 0xFFFFFFu) * (({RawSource(instruction, 1)}) & 0xFFFFFFu))",
                "VMulHiU32" =>
                    $"mulhi({RawSource(instruction, 0)}, {RawSource(instruction, 1)})",
                "VMulHiU32U24" =>
                    $"mulhi(({RawSource(instruction, 0)}) & 0xFFFFFFu, ({RawSource(instruction, 1)}) & 0xFFFFFFu)",
                "VMulHiI32" =>
                    AsUInt($"mulhi(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)}))"),
                "VMulHiI32I24" => EmitSignedMultiplyHigh24(instruction),
                "VDot4cI32I8" => EmitDot4cI32I8(instruction, destination),
                "VLerpU8" => EmitLerpU8(instruction),
                "VMsadU8" => EmitMsadU8(instruction),
                "VPermB32" => EmitPermB32(instruction),
                "VMulLoU16" or "VMadU16" or "VMadI16" or "VMadI32I16" or
                    "VMin3I16" or "VMin3U16" or "VMax3I16" or "VMax3U16" or "VMed3I16" or "VMed3U16" =>
                    EmitInt16Vop3(instruction, destination, intClamp),
                "VMadU32U24" =>
                    $"(((({RawSource(instruction, 0)}) & 0xFFFFFFu) * (({RawSource(instruction, 1)}) & 0xFFFFFFu)) + ({RawSource(instruction, 2)}))",
                // The multiplicands are the op_sel-selected halves; the addend is whole.
                "VMadU32U16" => intClamp
                    ? $"((uint)min((ulong)({Int16Source(instruction, 0, signed: false)}) * (ulong)({Int16Source(instruction, 1, signed: false)}) + (ulong)({RawSource(instruction, 2)}), 0xFFFFFFFFul))"
                    : $"((({Int16Source(instruction, 0, signed: false)}) * ({Int16Source(instruction, 1, signed: false)})) + ({RawSource(instruction, 2)}))",
                "VAdd3U32" =>
                    $"(({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)}) + ({RawSource(instruction, 2)}))",
                "VXadU32" =>
                    $"((({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)})) + ({RawSource(instruction, 2)}))",
                "VSadU32" =>
                    $"((max({RawSource(instruction, 0)}, {RawSource(instruction, 1)}) - min({RawSource(instruction, 0)}, {RawSource(instruction, 1)})) + ({RawSource(instruction, 2)}))",
                "VAddLshlU32" =>
                    $"((({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)})) << (({RawSource(instruction, 2)}) & 31u))",
                "VLshlAddU32" =>
                    $"((({RawSource(instruction, 0)}) << (({RawSource(instruction, 1)}) & 31u)) + ({RawSource(instruction, 2)}))",
                "VMinU32" => $"min({RawSource(instruction, 0)}, {RawSource(instruction, 1)})",
                "VMaxU32" => $"max({RawSource(instruction, 0)}, {RawSource(instruction, 1)})",
                "VMinI32" =>
                    AsUInt($"min(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)}))"),
                "VMaxI32" =>
                    AsUInt($"max(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)}))"),
                "VMin3U32" =>
                    $"min(min({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), {RawSource(instruction, 2)})",
                "VMax3U32" =>
                    $"max(max({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), {RawSource(instruction, 2)})",
                "VMin3I32" =>
                    AsUInt($"min(min(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), as_type<int>({RawSource(instruction, 2)}))"),
                "VMax3I32" =>
                    AsUInt($"max(max(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), as_type<int>({RawSource(instruction, 2)}))"),
                "VMed3U32" =>
                    $"max(min({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), min(max({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), {RawSource(instruction, 2)}))",
                "VMed3I32" =>
                    AsUInt($"max(min(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), min(max(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), as_type<int>({RawSource(instruction, 2)})))"),

                // ---- bitwise ----
                "VAndB32" => $"(({RawSource(instruction, 0)}) & ({RawSource(instruction, 1)}))",
                "VOrB32" => $"(({RawSource(instruction, 0)}) | ({RawSource(instruction, 1)}))",
                "VXorB32" => $"(({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)}))",
                "VXor3B32" => $"(({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)}) ^ ({RawSource(instruction, 2)}))",
                "VXnorB32" => $"~(({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)}))",
                "VNotB32" => $"~({RawSource(instruction, 0)})",
                "VAndOrB32" =>
                    $"((({RawSource(instruction, 0)}) & ({RawSource(instruction, 1)})) | ({RawSource(instruction, 2)}))",
                "VOr3U32" =>
                    $"(({RawSource(instruction, 0)}) | ({RawSource(instruction, 1)}) | ({RawSource(instruction, 2)}))",
                "VLshlOrU32" =>
                    $"((({RawSource(instruction, 0)}) << (({RawSource(instruction, 1)}) & 31u)) | ({RawSource(instruction, 2)}))",
                "VLshlB32" => $"(({RawSource(instruction, 0)}) << (({RawSource(instruction, 1)}) & 31u))",
                "VLshlrevB32" => $"(({RawSource(instruction, 1)}) << (({RawSource(instruction, 0)}) & 31u))",
                "VLshrB32" => $"(({RawSource(instruction, 0)}) >> (({RawSource(instruction, 1)}) & 31u))",
                "VLshrrevB32" => $"(({RawSource(instruction, 1)}) >> (({RawSource(instruction, 0)}) & 31u))",
                "VLshrrevB64" => EmitShiftRev64(instruction, destination, left: false),
                "VLshlrevB64" => EmitShiftRev64(instruction, destination, left: true),
                "VAshrI32" =>
                    AsUInt($"(as_type<int>({RawSource(instruction, 0)}) >> (({RawSource(instruction, 1)}) & 31u))"),
                "VAshrrevI32" =>
                    AsUInt($"(as_type<int>({RawSource(instruction, 1)}) >> (({RawSource(instruction, 0)}) & 31u))"),
                // extract_bits is undefined once offset + width passes 32, which the ISA
                // allows: the field is just the shifted value masked to `width` bits.
                "VBfeU32" =>
                    $"((({RawSource(instruction, 0)}) >> (({RawSource(instruction, 1)}) & 31u)) & ((1u << (({RawSource(instruction, 2)}) & 31u)) - 1u))",
                "VBfeI32" => EmitBfeI32(instruction),
                "VBfiB32" =>
                    $"((({RawSource(instruction, 0)}) & ({RawSource(instruction, 1)})) | (~({RawSource(instruction, 0)}) & ({RawSource(instruction, 2)})))",
                "VAlignbitB32" =>
                    $"uint(((ulong({RawSource(instruction, 0)}) << 32) | ulong({RawSource(instruction, 1)})) >> (({RawSource(instruction, 2)}) & 31u))",
                "VAlignbyteB32" =>
                    $"uint(((ulong({RawSource(instruction, 0)}) << 32) | ulong({RawSource(instruction, 1)})) >> ((({RawSource(instruction, 2)}) & 3u) * 8u))",
                "VBfmB32" =>
                    $"(((1u << (({RawSource(instruction, 0)}) & 31u)) - 1u) << (({RawSource(instruction, 1)}) & 31u))",
                "VBfrevB32" => $"reverse_bits({RawSource(instruction, 0)})",
                "VBcntU32B32" => $"(popcount({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)}))",
                "VFfblB32" =>
                    $"(({RawSource(instruction, 0)}) == 0u ? 0xFFFFFFFFu : (uint)ctz({RawSource(instruction, 0)}))",
                "VFfbhU32" => EmitFindFirstBitHigh(instruction, signed: false),
                "VFfbhI32" => EmitFindFirstBitHigh(instruction, signed: true),

                // ---- wave / lane ----
                // mbcnt reads the mask dword the guest passes (no cross-lane
                // op), so only the per-lane thread-mask math differs by wave
                // size. Wave64 lanes 32..63 count the whole low half in mbcnt_lo
                // and their own partial in mbcnt_hi; a 1u << lane for lane>=32
                // would be undefined, so those are split out.
                "VMbcntLoU32B32" => IsWave64
                    ? $"((sharpemu_lane >= 32u ? popcount({RawSource(instruction, 0)}) : popcount(({RawSource(instruction, 0)}) & ((1u << sharpemu_lane) - 1u))) + ({RawSource(instruction, 1)}))"
                    : $"(popcount(({RawSource(instruction, 0)}) & ((1u << sharpemu_lane) - 1u)) + ({RawSource(instruction, 1)}))",
                "VMbcntHiU32B32" => IsWave64
                    ? $"((sharpemu_lane >= 32u ? popcount(({RawSource(instruction, 0)}) & ((1u << (sharpemu_lane - 32u)) - 1u)) : 0u) + ({RawSource(instruction, 1)}))"
                    // Wave32: the high mask half holds no lanes; pass the addend.
                    : RawSource(instruction, 1),
                "VPermlane16B32" => EmitPermlane16(instruction, exchangeRows: false),
                "VPermlanex16B32" => EmitPermlane16(instruction, exchangeRows: true),

                // ---- cube map helpers ----
                "VCubeidF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Id),
                "VCubescF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Sc),
                "VCubetcF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Tc),
                "VCubemaF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Ma),

                _ => null,
            };

            if (expression is null)
            {
                switch (instruction.Opcode)
                {
                    case "VAddCoU32":
                    {
                        var left = Temp("uint", RawSource(instruction, 0));
                        var right = Temp("uint", RawSource(instruction, 1));
                        var sum = Temp("uint", $"{left} + {right}");
                        var carry = Temp("bool", $"{sum} < {left}");
                        StoreCarryOut(instruction, carry);
                        expression = intClamp ? $"({carry} ? 0xFFFFFFFFu : {sum})" : sum;
                        break;
                    }
                    case "VSubCoU32":
                    case "VSubrevCoU32":
                    {
                        var reverse = instruction.Opcode == "VSubrevCoU32";
                        var left = Temp("uint", RawSource(instruction, reverse ? 1 : 0));
                        var right = Temp("uint", RawSource(instruction, reverse ? 0 : 1));
                        var borrowOut = Temp("bool", $"{left} < {right}");
                        StoreCarryOut(instruction, borrowOut);
                        expression = intClamp ? $"({borrowOut} ? 0u : ({left} - {right}))" : $"({left} - {right})";
                        break;
                    }
                    case "VAddcU32":
                    case "VAddCoCiU32":
                    {
                        var left = Temp("uint", RawSource(instruction, 0));
                        var right = Temp("uint", RawSource(instruction, 1));
                        var carryIn = instruction.Sources.Count > 2
                            ? MaskBitExpression(instruction.Sources[2])
                            : "vcc";
                        var partial = Temp("uint", $"{left} + {right}");
                        var sum = Temp("uint", $"{partial} + (({carryIn}) ? 1u : 0u)");
                        var carry = Temp("bool", $"({partial} < {left}) || ({sum} < {partial})");
                        StoreCarryOut(instruction, carry);
                        expression = intClamp ? $"({carry} ? 0xFFFFFFFFu : {sum})" : sum;
                        break;
                    }
                    case "VSubbU32":
                    case "VSubbrevU32":
                    case "VSubCoCiU32":
                    case "VSubrevCoCiU32":
                    {
                        var reverse = instruction.Opcode is "VSubbrevU32" or "VSubrevCoCiU32";
                        var left = Temp("uint", RawSource(instruction, reverse ? 1 : 0));
                        var right = Temp("uint", RawSource(instruction, reverse ? 0 : 1));
                        var borrowIn = instruction.Sources.Count > 2
                            ? MaskBitExpression(instruction.Sources[2])
                            : "vcc";
                        var borrow = Temp("uint", $"({borrowIn}) ? 1u : 0u");
                        var partial = Temp("uint", $"{left} - {right}");
                        var borrowOut = Temp("bool", $"({left} < {right}) || ({partial} < {borrow})");
                        StoreCarryOut(instruction, borrowOut);
                        expression = intClamp ? $"({borrowOut} ? 0u : ({partial} - {borrow}))" : $"({partial} - {borrow})";
                        break;
                    }
                    case "VMadU64U32":
                    {
                        // 64-bit product+addend into a VGPR pair, carry to SDST.
                        var product = Temp(
                            "ulong",
                            $"(ulong)({RawSource(instruction, 0)}) * (ulong)({RawSource(instruction, 1)})");
                        var addend = Temp("ulong", RawSource64(instruction, 2));
                        var wide = Temp("ulong", $"{product} + {addend}");
                        StoreCarryOut(instruction, $"{wide} < {addend}");
                        StoreVector(destination + 1, $"(uint)({wide} >> 32)");
                        expression = $"(uint){wide}";
                        break;
                    }
                    case "VMadI64I32":
                        expression = EmitMadI64I32(instruction, destination);
                        break;
                    case "VAshrrevI64":
                        expression = EmitArithmeticShiftRev64(instruction, destination);
                        break;
                    case "VQsadPkU16U8":
                    case "VMqsadPkU16U8":
                    case "VMqsadU32U8":
                        return TryEmitQuadSad(instruction, destination, out error);
                    case "VPackB32F16":
                        if (!TryEmitPackB32F16(instruction, out expression, out error))
                        {
                            return false;
                        }

                        break;
                    case "VCvtPknormI16F16":
                    case "VCvtPknormU16F16":
                        if (!TryEmitPackNormF16(instruction, instruction.Opcode == "VCvtPknormI16F16", out expression, out error))
                        {
                            return false;
                        }

                        break;
                    case "VFmacF16":
                    case "VMacLegacyF32":
                    {
                        // The accumulator is the destination itself (its low f16 half for FMAC_F16).
                        // A VOP3 form that selects, negates or takes the absolute value of it, or
                        // writes the high half, has no plain reading and is refused.
                        if (instruction.Control is Gen5Vop3Control tied &&
                            (((tied.AbsoluteMask | tied.NegateMask) & 4) != 0 || (tied.OperandSelect & 0xC) != 0))
                        {
                            error = $"{instruction.Opcode}: modifiers or op_sel on the accumulator are not supported";
                            return false;
                        }

                        expression = instruction.Opcode == "VFmacF16"
                            // A half fma rounds once; a float fma then a narrowing could round twice.
                            ? Float16Result(
                                instruction,
                                destination,
                                $"(float)fma(half({F16(instruction, 0)}), half({F16(instruction, 1)}), as_type<half>((ushort)(v[{destination}] & 0xFFFFu)))")
                            : EmitLegacyMultiply(instruction, $"as_type<float>(v[{destination}])");
                        break;
                    }
                    case "VPkMadI16":
                    case "VPkMulLoU16":
                    case "VPkLshlrevB16":
                    case "VPkLshrrevB16":
                    case "VPkAshrrevI16":
                    case "VPkMaxI16":
                    case "VPkMinI16":
                    case "VPkMadU16":
                    case "VPkAddU16":
                    case "VPkSubU16":
                    case "VPkMaxU16":
                    case "VPkMinU16":
                    case "VDot2F32F16":
                    case "VDot2I32I16":
                    case "VDot2U32U16":
                    case "VDot4I32I8":
                    case "VDot4U32U8":
                    case "VDot8I32I4":
                    case "VDot8U32U4":
                    {
                        if (instruction.Control is not Gen5Vop3pControl packedInteger)
                        {
                            error = $"missing vop3p control for {instruction.Opcode}";
                            return false;
                        }

                        if (instruction.Opcode.StartsWith("VDot", StringComparison.Ordinal))
                        {
                            if (!TryEmitDot(instruction, packedInteger, out expression, out error))
                            {
                                return false;
                            }
                        }
                        else if (!TryEmitPackedInt16(instruction, packedInteger, out expression, out error))
                        {
                            return false;
                        }

                        break;
                    }
                    case "VPkAddF16":
                    case "VPkMulF16":
                    case "VPkMinF16":
                    case "VPkMaxF16":
                    case "VPkFmaF16":
                    case "VPkAddI16":
                    case "VPkSubI16":
                    case "VFmaMixF32":
                    case "VFmaMixloF16":
                    case "VFmaMixhiF16":
                    {
                        if (instruction.Control is not Gen5Vop3pControl packed)
                        {
                            error = $"missing vop3p control for {instruction.Opcode}";
                            return false;
                        }

                        expression = instruction.Opcode switch
                        {
                            "VPkAddI16" => EmitPackedI16Arithmetic(instruction, packed, subtract: false),
                            "VPkSubI16" => EmitPackedI16Arithmetic(instruction, packed, subtract: true),
                            "VFmaMixF32" or "VFmaMixloF16" or "VFmaMixhiF16" =>
                                EmitFmaMix(instruction, packed, destination),
                            _ => EmitPackedF16(instruction, packed),
                        };
                        break;
                    }
                    case "VAddNcU16":
                    case "VAddNcI16":
                    case "VSubNcU16":
                    case "VSubNcI16":
                    case "VLshrrevB16":
                    case "VLshlrevB16":
                    case "VAshrrevI16":
                    case "VMaxU16":
                    case "VMinU16":
                    case "VMaxI16":
                    case "VMinI16":
                    {
                        // The hardware saturates these to the 16-bit signed or
                        // unsigned range when clamp is set; the SPIR-V translator
                        // refuses that case rather than mis-emitting it, and both
                        // backends have to agree on what a program may rely on.
                        if (instruction.Control is Gen5Vop3Control { Clamp: true })
                        {
                            error = $"{instruction.Opcode}: clamped 16-bit integer results are not supported";
                            return false;
                        }

                        var signed = instruction.Opcode.EndsWith("I16", StringComparison.Ordinal);
                        var left = Int16Source(instruction, 0, signed);
                        var right = Int16Source(instruction, 1, signed);
                        expression = Int16Result(
                            instruction,
                            destination,
                            instruction.Opcode switch
                            {
                                "VAddNcU16" or "VAddNcI16" => $"(({left}) + ({right}))",
                                "VSubNcU16" or "VSubNcI16" => $"(({left}) - ({right}))",
                                // The rev shifts take the shift amount from src0 and
                                // the value from src1, with only four bits of count.
                                "VLshrrevB16" => $"(({right}) >> (({left}) & 15u))",
                                "VLshlrevB16" => $"(({right}) << (({left}) & 15u))",
                                "VAshrrevI16" => AsUInt($"(as_type<int>({right}) >> (({left}) & 15u))"),
                                "VMaxU16" => $"max({left}, {right})",
                                "VMinU16" => $"min({left}, {right})",
                                "VMaxI16" => AsUInt($"max(as_type<int>({left}), as_type<int>({right}))"),
                                _ => AsUInt($"min(as_type<int>({left}), as_type<int>({right}))"),
                            });
                        break;
                    }
                    default:
                        error = $"unsupported vector opcode {instruction.Opcode}";
                        return false;
                }
            }

            var result = Temp("uint", expression);
            if (instruction.Control is Gen5DppControl dpp)
            {
                var writeEnabled = EmitDppWriteEnabled(dpp);
                result = Temp("uint", $"({writeEnabled}) ? {result} : v[{destination}]");
            }

            if (instruction.Control is Gen5SdwaControl { ScalarDestination: null } sdwaDestination)
            {
                result = ApplySdwaDestination(sdwaDestination, result, $"v[{destination}]", _sixteenBitSdwaResult);
            }

            _sixteenBitSdwaResult = false;
            StoreVector(destination, result);
            return true;
        }

        // D.u = (unsigned)S0.f: out-of-range values (infinity included) saturate and NaN
        // is 0 (ISA 12.8, V_CVT_U32_F32). A bare MSL float-to-uint cast is undefined there.
        private static string SaturatedFloatToUint(string x) =>
            $"({x} >= 4294967296.0f ? 0xFFFFFFFFu : (uint)clamp(isnan({x}) ? 0.0f : {x}, 0.0f, 4294967040.0f))";

        // V_CVT_I32_F32 and the rounding variants: saturate to INT_MIN..INT_MAX, NaN is 0.
        private string SaturatedFloatToInt(string x) =>
            AsUInt($"({x} >= 2147483648.0f ? 2147483647 : (int)clamp(isnan({x}) ? 0.0f : {x}, -2147483648.0f, 2147483520.0f))");

        // ISA 12.12: D.i = (S0.i >> S1[4:0]) & ((1 << S2[4:0]) - 1), the field sign-extended
        // from its top bit; a zero width gives 0 and offset + width may pass 32.
        private string EmitBfeI32(Gen5ShaderInstruction instruction)
        {
            var shifted = Temp("int", $"as_type<int>({RawSource(instruction, 0)}) >> (({RawSource(instruction, 1)}) & 31u)");
            var width = Temp("uint", $"({RawSource(instruction, 2)}) & 31u");
            var up = Temp("uint", $"(32u - {width}) & 31u");
            return AsUInt($"({width} == 0u ? 0 : (as_type<int>(as_type<uint>({shifted}) << {up}) >> {up}))");
        }

        /// <summary>Whether a no-carry add/sub saturates as signed under clamp. Only the
        /// VOP3-only V_ADD_NC_I32 (0x37F) and V_SUB_NC_I32 (0x376) are signed. The decoder names
        /// the VOP2 opcodes 0x25-0x27 VAddI32/VSubI32/VSubrevI32, but on gfx10 they are
        /// V_ADD_NC_U32, V_SUB_NC_U32 and V_SUBREV_NC_U32 (their VOP3 forms 0x125-0x127 decode
        /// as the U32 names), so an SDWA clamp on them saturates to 0..0xFFFFFFFF.</summary>
        private static bool IsSignedNoCarryAddSub(Gen5ShaderInstruction instruction) =>
            instruction.Encoding == Gen5ShaderEncoding.Vop3 && instruction.Opcode is "VAddI32" or "VSubI32";

        // The clamp bit on the 32-bit add/sub family saturates the result: unsigned
        // forms to 0..0xFFFFFFFF, signed forms to INT_MIN..INT_MAX.
        private string SaturatedAddSub(
            Gen5ShaderInstruction instruction,
            bool subtract,
            bool reverse,
            bool signed)
        {
            var left = Temp("uint", RawSource(instruction, reverse ? 1 : 0));
            var right = Temp("uint", RawSource(instruction, reverse ? 0 : 1));
            if (signed)
            {
                var op = subtract ? "-" : "+";
                return AsUInt(
                    $"((int)clamp((long)as_type<int>({left}) {op} (long)as_type<int>({right}), -2147483648l, 2147483647l))");
            }

            return subtract
                ? $"({left} < {right} ? 0u : {left} - {right})"
                : $"((uint)min((ulong){left} + (ulong){right}, 0xFFFFFFFFul))";
        }

        private string EmitSignedMultiply24(Gen5ShaderInstruction instruction)
        {
            var signedLeft = $"(as_type<int>(({RawSource(instruction, 0)}) << 8u) >> 8)";
            var signedRight = $"(as_type<int>(({RawSource(instruction, 1)}) << 8u) >> 8)";
            // Unsigned multiplication preserves the low result bits on overflow.
            return $"({AsUInt(signedLeft)} * {AsUInt(signedRight)})";
        }

        // The relative moves address the VGPR file relative to the wave's own
        // allocation base, so the encoded register number and M0 simply add.
        // V_MOVRELSD2 is the one form that takes separate source and destination
        // offsets, out of M0's two halves.
        //
        // Indexing the register file at run time is what costs: it forces the file to
        // stay an array, so none of it can be scalarized and all 256 dwords sit in
        // per-thread memory. In Octopath Traveler II that happens to exactly one shader
        // out of 67 — and it is the one that spends 44% of the frame's GPU time, a
        // vertex program run 12.5 million times a frame with four relative moves in it.
        //
        // Gen5MoveRelativeOffsets (shared IR, already used by the SPIR-V backend) says
        // which values M0 can hold at each relative move when straight-line scalar code
        // produces it. Knowing them turns the move into a select among constant register
        // numbers, and the file scalarizes like any other program's.
        private bool TryEmitMoveRelative(Gen5ShaderInstruction instruction, out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            // The VOP3 (e64) encoding is accepted without modifiers: a relative move is untyped.
            if (instruction.Control is not (null or
                    Gen5Vop3Control { AbsoluteMask: 0, NegateMask: 0, OutputModifier: 0, Clamp: false, OperandSelect: 0 }) ||
                instruction.Sources.Count == 0 ||
                instruction.Destinations.Count == 0 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.VectorRegister)
            {
                error = $"{opcode} expects an unmodified operand and a VGPR destination";
                return false;
            }

            if (_options.BoundedRelativeMoves)
            {
                _moveRelativeOffsets ??= Ir.Gen5MoveRelativeOffsets.Analyze(_request.Program);
                if (_moveRelativeOffsets.TryGetValue(instruction.Pc, out var offsets) &&
                    offsets.Length != 0)
                {
                    return TryEmitBoundedMoveRelative(instruction, offsets, out error);
                }
            }

            // M0 that no straight-line reasoning bounds — the waterfall idiom
            // (v_readfirstlane into M0, v_cmpx, v_movrels) is the common source, and it
            // is what the hot Octopath vertex shader does. The register number is then
            // only known at run time, so the access is marked here and resolved once the
            // set of registers the program touches is known: see Gen5MslTranslator.Registers.
            var m0 = Temp("uint", ScalarExpression(M0Register));
            var sourceOffset = m0;
            var destinationOffset = m0;
            if (opcode == "VMovrelsd2B32")
            {
                sourceOffset = Temp("uint", $"{m0} & 0x3FFu");
                destinationOffset = Temp("uint", $"({m0} >> 16u) & 0x3FFu");
            }

            string value;
            if (opcode == "VMovreldB32")
            {
                // Only the destination is relative here: src0 is an ordinary
                // operand and may be an SGPR or an inline/literal constant.
                value = Temp("uint", RawSource(instruction, 0));
            }
            else
            {
                var source = instruction.Sources[0];
                if (source.Kind != Gen5OperandKind.VectorRegister)
                {
                    error = $"{opcode} source must be a vector register";
                    return false;
                }

                var sourceIndex = Temp(
                    "uint",
                    $"({source.Value}u + {sourceOffset}) & {VectorRegisterFileCount - 1}u");
                if (_options.BoundedRelativeMoves)
                {
                    // A statement rather than an expression: the resolved read is a
                    // switch on the register number (see Gen5MslTranslator.Registers).
                    value = $"t{_nextTemp++}";
                    Line($"uint {value};");
                    Line($"{DynamicVectorRead}({value}, {sourceIndex});");
                }
                else
                {
                    value = Temp("uint", $"v[{sourceIndex}]");
                }
            }

            var destination = instruction.Destinations[0].Value;
            if (opcode == "VMovrelsB32")
            {
                StoreVector(destination, value);
                return true;
            }

            var target = Temp(
                "uint",
                $"({destination}u + {destinationOffset}) & {VectorRegisterFileCount - 1}u");
            Line(ExecGuarded(_options.BoundedRelativeMoves
                ? $"{DynamicVectorWrite}({target}, {value});"
                : $"v[{target}] = {value};"));
            return true;
        }

        private IReadOnlyDictionary<uint, uint[]>? _moveRelativeOffsets;

        /// <summary>A relative move whose M0 values are known: the source becomes a chain
        /// of selects over constant register numbers and the destination a write per
        /// reachable register, so nothing indexes the register file at run time.</summary>
        private bool TryEmitBoundedMoveRelative(
            Gen5ShaderInstruction instruction,
            uint[] offsets,
            out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            var m0 = Temp("uint", ScalarExpression(M0Register));
            var sourceOffsets = offsets;
            var destinationOffsets = offsets;
            var sourceSelector = m0;
            var destinationSelector = m0;
            if (opcode == "VMovrelsd2B32")
            {
                // M0's halves are independent offsets, so each side selects on its own.
                sourceSelector = Temp("uint", $"{m0} & 0x3FFu");
                destinationSelector = Temp("uint", $"({m0} >> 16u) & 0x3FFu");
                sourceOffsets = Distinct(offsets, value => value & 0x3FFu);
                destinationOffsets = Distinct(offsets, value => (value >> 16) & 0x3FFu);
            }

            string value;
            if (opcode == "VMovreldB32")
            {
                value = Temp("uint", RawSource(instruction, 0));
            }
            else
            {
                var source = instruction.Sources[0];
                if (source.Kind != Gen5OperandKind.VectorRegister)
                {
                    error = $"{opcode} source must be a vector register";
                    return false;
                }

                var selected = VectorExpression(source.Value + sourceOffsets[0]);
                for (var index = 1; index < sourceOffsets.Length; index++)
                {
                    selected =
                        $"({sourceSelector} == {FormatUInt(sourceOffsets[index])} " +
                        $"? {VectorExpression(source.Value + sourceOffsets[index])} : {selected})";
                }

                value = Temp("uint", selected);
            }

            var destination = instruction.Destinations[0].Value;
            if (opcode == "VMovrelsB32")
            {
                StoreVector(destination, value);
                return true;
            }

            // Only the register M0 actually selects takes the value; the others keep
            // theirs, which is what writing through the index would have done.
            foreach (var offset in destinationOffsets)
            {
                var register = (destination + offset) & (VectorRegisterFileCount - 1);
                StoreVector(
                    register,
                    $"{destinationSelector} == {FormatUInt(offset)} ? {value} : {VectorExpression(register)}");
            }

            return true;
        }

        private static uint[] Distinct(uint[] values, Func<uint, uint> project)
        {
            var projected = new SortedSet<uint>();
            foreach (var value in values)
            {
                projected.Add(project(value));
            }

            return [.. projected];
        }

        // D.u64 = S1.u64 << or >> (S0.u32 & 0x3F), landing in the register pair.
        private string EmitShiftRev64(Gen5ShaderInstruction instruction, uint destination, bool left)
        {
            var shift = Temp("uint", $"({RawSource(instruction, 0)}) & 63u");
            var shifted = Temp(
                "ulong",
                $"({RawSource64(instruction, 1)}) {(left ? "<<" : ">>")} {shift}");
            StoreVector(destination + 1, $"(uint)({shifted} >> 32)");
            return $"(uint){shifted}";
        }

        private string EmitCvtPkU8F32(Gen5ShaderInstruction instruction)
        {
            // flt32_to_uint8 saturates to 0..255, maps NaN to 0 and rounds to nearest even.
            var source = Temp("float", F(instruction, 0));
            var converted = Temp("uint", $"(uint)rint(clamp(isnan({source}) ? 0.0f : {source}, 0.0f, 255.0f))");
            var offset = Temp("uint", $"(({RawSource(instruction, 1)}) & 3u) << 3");
            var baseValue = Temp("uint", RawSource(instruction, 2));
            return $"(({baseValue} & ~(0xFFu << {offset})) | (({converted} & 0xFFu) << {offset}))";
        }

        private string EmitCvtPkrtzF16F32(Gen5ShaderInstruction instruction)
        {
            var first = TruncateFloat32ForPack(F(instruction, 0));
            var second = TruncateFloat32ForPack(F(instruction, 1));
            return $"(((uint)as_type<ushort>(half({first}))) | (((uint)as_type<ushort>(half({second}))) << 16))";
        }

        // Rounds an f32 toward zero onto an f16 value, so half() then converts it exactly
        // (V_CVT_PKRTZ_F16_F32). Toward zero, a finite value beyond the f16 range becomes
        // +-65504, never infinity: Silent Hill's deferred lighting packs values past 65504
        // and its fog pass scales them back down, while an infinity stays infinite, turns
        // into NaN and blacks out the title-menu scene. Subnormal results keep whole 2^-24
        // steps; NaN and infinity pass through to half() unchanged.
        private string TruncateFloat32ForPack(string source)
        {
            var value = Temp("float", source);
            var normal = $"as_type<float>(as_type<uint>(clamp({value}, -65504.0f, 65504.0f)) & 0xFFFFE000u)";
            var subnormal = $"(trunc({value} * 16777216.0f) * 5.9604644775390625e-08f)";
            var finite = $"(fabs({value}) < 6.103515625e-05f ? {subnormal} : {normal})";
            return Temp("float", $"(isnan({value}) || isinf({value})) ? {value} : {finite}");
        }

        private string EmitFindFirstBitHigh(Gen5ShaderInstruction instruction, bool signed)
        {
            var source = Temp("uint", RawSource(instruction, 0));
            if (!signed)
            {
                // V_FFBH_U32 counts from bit 31, so 31 - findMSB is just the
                // leading-zero count; a zero source has no set bit and reports ~0.
                return $"({source} == 0u ? 0xFFFFFFFFu : (uint)clz({source}))";
            }

            // V_FFBH_I32 looks for the first bit that differs from the sign, which
            // is the first clear bit of a negative source: complementing turns it
            // back into a leading-zero count. 0 and -1 have no such bit at all.
            var magnitude = Temp(
                "uint",
                $"(({source} & 0x80000000u) != 0u) ? ~{source} : {source}");
            return $"({magnitude} == 0u ? 0xFFFFFFFFu : (uint)clz({magnitude}))";
        }

        // ---- VOP3P: packed and mixed 16-bit math ----

        // Packed f16 (VOP3P). Each source register carries two f16 values, one per
        // result lane, and the two lanes are computed independently: op_sel /
        // op_sel_hi pick which half of each source feeds the lane, neg_lo / neg_hi
        // negate it, the op runs in f32 and the result is rounded back to f16. The
        // SPIR-V backend has to open-code every f16 <-> f32 conversion because
        // SPIR-V without the Float16 capability cannot express them exactly; Metal
        // has a native `half` whose conversions are IEEE (the translator's libraries
        // are built with fast math off), so the conversions here are plain casts.
        // VOP3P carries neither abs nor omod, and F/FloatResult only look at
        // Gen5Vop3Control, so every modifier below is decoded by hand.
        private string EmitPackedF16(Gen5ShaderInstruction instruction, Gen5Vop3pControl control)
        {
            var sources = PackedSources(instruction, control, instruction.Opcode == "VPkFmaF16" ? 3 : 2);
            var low = EmitPackedF16Lane(instruction, control, sources, highLane: false);
            var high = EmitPackedF16Lane(instruction, control, sources, highLane: true);
            return $"(({low}) | (({high}) << 16))";
        }

        // Both lanes read the same sources, so each one is hoisted into a single
        // temp. An inline constant is one f32 value rather than a packed pair: both
        // lanes take it whole, which is why it is held as a float and not as raw bits.
        // A 32-bit literal is a packed pair like a register and op_sel picks its half.
        private string[] PackedSources(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            int count)
        {
            var sources = new string[count];
            for (var index = 0; index < count; index++)
            {
                sources[index] = IsPackedConstant(instruction.Sources[index])
                    ? Temp("float", F(instruction, index))
                    : Temp("uint", RawSource(instruction, index));
            }

            return sources;
        }

        private static bool IsPackedConstant(Gen5Operand operand) =>
            operand.Kind is Gen5OperandKind.EncodedConstant;

        private string EmitPackedF16Lane(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            string[] sources,
            bool highLane)
        {
            var left = PackedF16Operand(instruction, control, sources, 0, highLane);
            var right = PackedF16Operand(instruction, control, sources, 1, highLane);
            var value = instruction.Opcode switch
            {
                "VPkAddF16" => $"({left} + {right})",
                "VPkMulF16" => $"({left} * {right})",
                // V_MIN/MAX_F16 per lane: a NaN operand yields the other one (quieted when
                // signalling in IEEE mode) and -0 orders below +0, which MSL's fmin/fmax
                // leave unspecified.
                "VPkMinF16" => PackedHalfMinMax("sharpemu_min_f16", left, right),
                "VPkMaxF16" => PackedHalfMinMax("sharpemu_max_f16", left, right),
                // Use half fma itself: float fma followed by half conversion can
                // double-round at a half midpoint (MSL specification section 6.6).
                _ => $"(float)fma(half({left}), half({right}), half({PackedF16Operand(instruction, control, sources, 2, highLane)}))",
            };
            if (control.Clamp)
            {
                value = ClampToUnitInterval(value);
            }

            // 0 and 1 are exact in both formats and the clamp is monotonic, so
            // clamping ahead of the narrowing gives the f16 the hardware clamps to.
            return $"((uint)as_type<ushort>(half({value})))";
        }

        private string PackedHalfMinMax(string helper, string left, string right) =>
            $"(float)as_type<half>((ushort){helper}((uint)as_type<ushort>(half({left})), " +
            $"(uint)as_type<ushort>(half({right})), {(_request.IeeeMode ? "true" : "false")}))";

        /// <summary>One packed-f16 source half widened to f32, with its lane's negate applied.</summary>
        private string PackedF16Operand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            string[] sources,
            int index,
            bool highLane)
        {
            var negateMask = highLane ? control.NegHiMask : control.NegLoMask;
            // An inline constant has no half to select; only the lane's negate applies.
            var value = IsPackedConstant(instruction.Sources[index])
                ? sources[index]
                : HalfToFloat(
                    sources[index],
                    (((highLane ? control.OpSelHiMask : control.OpSelMask) >> index) & 1) != 0);
            return ((negateMask >> index) & 1) != 0 ? $"(-{value})" : value;
        }

        /// <summary>Widens one 16-bit half of a packed register to f32.</summary>
        private static string HalfToFloat(string raw, bool highHalf) =>
            $"(float)as_type<half2>({raw}).{(highHalf ? "y" : "x")}";

        /// <summary>
        /// The VOP3P clamp modifier saturates to [0, 1] and turns NaN into 0. MSL's
        /// clamp() is unspecified for a NaN input, so the bounds are applied as
        /// ordered compares, which are false for NaN and therefore pick the 0 bound.
        /// </summary>
        private string ClampToUnitInterval(string value)
        {
            var bounded = Temp("float", value);
            var lowerBounded = Temp("float", $"({bounded} > 0.0f) ? {bounded} : 0.0f");
            var clamped = $"(({lowerBounded} < 1.0f) ? {lowerBounded} : 1.0f)";
            // Without DX10_CLAMP a NaN passes through the clamp.
            return _request.Dx10Clamp ? clamped : $"(isnan({bounded}) ? {bounded} : {clamped})";
        }

        // V_PK_ADD/SUB_I16 operate on the op_sel-selected halves of src0 and src1
        // and wrap in 16 bits per lane, so the signed and unsigned readings of the
        // operands share one bit-level implementation.
        private string EmitPackedI16Arithmetic(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            bool subtract)
        {
            var left = Temp("uint", RawSource(instruction, 0));
            var right = Temp("uint", RawSource(instruction, 1));
            var op = subtract ? "-" : "+";

            static string SelectHalf(string value, uint mask, int index) =>
                ((mask >> index) & 1) != 0
                    ? $"(({value} >> 16) & 0xFFFFu)"
                    : $"({value} & 0xFFFFu)";

            static string Signed16(string half) => $"(as_type<int>({half} << 16u) >> 16)";

            // With clamp each lane saturates to the signed 16-bit range (ISA 6.5).
            string Lane(uint mask) => control.Clamp
                ? $"(as_type<uint>(clamp({Signed16(SelectHalf(left, mask, 0))} {op} {Signed16(SelectHalf(right, mask, 1))}, -32768, 32767)) & 0xFFFFu)"
                : $"(({SelectHalf(left, mask, 0)} {op} {SelectHalf(right, mask, 1)}) & 0xFFFFu)";

            return $"({Lane(control.OpSelMask)} | ({Lane(control.OpSelHiMask)} << 16))";
        }

        // V_FMA_MIX_F32 / _MIXLO_F16 / _MIXHI_F16. Unlike the v_pk_* ops these are a
        // single f32 fma whose three sources are decoded independently: op_sel_hi
        // says a register source is an f16 half (picked by op_sel) widened to f32
        // rather than a whole f32, neg_hi is the absolute-value modifier and neg_lo
        // negates. _MIXLO / _MIXHI round the f32 result to f16 and merge it into one
        // half of vdst, leaving the other half as it was.
        private string EmitFmaMix(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            uint destination)
        {
            var value =
                $"fma({FmaMixOperand(instruction, control, 0)}, " +
                $"{FmaMixOperand(instruction, control, 1)}, " +
                $"{FmaMixOperand(instruction, control, 2)})";
            if (control.Clamp)
            {
                value = ClampToUnitInterval(value);
            }

            if (instruction.Opcode == "VFmaMixF32")
            {
                return AsUInt($"({value})");
            }

            var packed = $"((uint)as_type<ushort>(half({value})))";
            return instruction.Opcode == "VFmaMixloF16"
                ? $"((v[{destination}] & 0xFFFF0000u) | {packed})"
                : $"((v[{destination}] & 0x0000FFFFu) | ({packed} << 16))";
        }

        private string FmaMixOperand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            int index)
        {
            var operand = instruction.Sources[index];
            // op_sel_hi narrows a register or a literal to one of its halves; an inline
            // constant keeps its f32 value either way. A half-selected literal is an f16
            // bit pattern: Silent Hill's luminance weights 0x34CD/0x38B8/0x2F0A read as
            // f32 denormals made the luminance zero and the lighting divided by it.
            var halfSource = ((control.OpSelHiMask >> index) & 1) != 0;
            var readAsHalf = halfSource &&
                operand.Kind is Gen5OperandKind.VectorRegister or Gen5OperandKind.ScalarRegister or
                    Gen5OperandKind.LiteralConstant;
            // An f16 source given an inline constant takes the constant's f16 value: an
            // integer one is a bit pattern (1 is the denormal 0x0001) and 1/(2*pi) is 0x3118.
            var value = readAsHalf
                ? HalfToFloat(
                    Temp("uint", RawSource(instruction, index)),
                    ((control.OpSelMask >> index) & 1) != 0)
                : halfSource && operand.Kind == Gen5OperandKind.EncodedConstant &&
                    TryHalfInlineConstant(operand.Value, out var halfBits)
                    ? $"(float)as_type<half>((ushort){FormatUInt(halfBits)})"
                    : F(instruction, index);

            // abs first, then negate — the order the hardware applies them in.
            if (((control.NegHiMask >> index) & 1) != 0)
            {
                value = $"fabs({value})";
            }

            return ((control.NegLoMask >> index) & 1) != 0 ? $"(-{value})" : value;
        }

        // V_PK_FMAC_F16 (VOP2 0x3C): each half of the destination accumulates its own
        // fused f16 multiply-add, D.h = fma(S0.h, S1.h, D.h). There is no op_sel and
        // no neg on this encoding, so the halves are read straight out of the raw
        // bits — including for an inline-constant source, whose 32 bits are read as a
        // packed pair like any register. Metal's half fma rounds exactly once, which
        // is what makes the fused form expressible directly.
        private string EmitPackedF16Fmac(Gen5ShaderInstruction instruction, uint destination)
        {
            var left = Temp("half2", $"as_type<half2>({PackedFmacSource(instruction, 0)})");
            var right = Temp("half2", $"as_type<half2>({PackedFmacSource(instruction, 1)})");
            var accumulator = Temp("half2", $"as_type<half2>(v[{destination}])");
            return AsUInt($"fma({left}, {right}, {accumulator})");
        }

        // Before gfx11 a float inline constant (0.5 .. 1/2pi, encodings 240-248) is an f16
        // in the low 16 bits with zero above, not the f32 the other encodings widen it to.
        private string PackedFmacSource(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            if (operand.Kind == Gen5OperandKind.EncodedConstant && operand.Value is >= 240 and <= 248)
            {
                ReadOnlySpan<ushort> halves = [0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118];
                return $"0x{halves[(int)(operand.Value - 240)]:X4}u";
            }

            return RawSource(instruction, sourceIndex);
        }

        // RDNA2 ISA section 12.7: DOT2C accumulates the two half products
        // into the old float destination. The C means accumulate, not clamp.
        private string EmitDot2cF32F16(Gen5ShaderInstruction instruction, uint destination)
        {
            var left = Temp("float2", $"(float2)as_type<half2>({RawSource(instruction, 0)})");
            var right = Temp("float2", $"(float2)as_type<half2>({RawSource(instruction, 1)})");
            var dot = $"({left}.x * {right}.x + {left}.y * {right}.y + as_type<float>(v[{destination}]))";
            return FloatResult(instruction, dot);
        }

        // ---- 16-bit integer VALU ----

        /// <summary>
        /// The 16-bit half of a source feeding a VOP3 16-bit integer op: op_sel bit
        /// `sourceIndex` takes the high half, and the signed forms sign-extend it so
        /// the surrounding 32-bit op sees the right value.
        /// </summary>
        private string Int16Source(Gen5ShaderInstruction instruction, int sourceIndex, bool signed)
        {
            var raw = RawSource(instruction, sourceIndex, applySdwaIntegerModifiers: false);
            var shift = instruction.Control is Gen5Vop3Control control &&
                (control.OperandSelect & (1u << sourceIndex)) != 0
                    ? 16
                    : 0;
            var half = $"((({raw}) >> {shift}) & 0xFFFFu)";
            return signed ? AsUInt($"(as_type<int>({half} << 16) >> 16)") : half;
        }

        /// <summary>Writes the low 16 bits of a result into the op_sel-selected
        /// destination half, preserving the other half.</summary>
        private string Int16Result(
            Gen5ShaderInstruction instruction,
            uint destination,
            string value)
        {
            var half = $"(({value}) & 0xFFFFu)";
            if (instruction.Control is Gen5SdwaControl)
            {
                return SixteenBitSdwaResult(half);
            }

            return ((instruction.Control as Gen5Vop3Control)?.OperandSelect ?? 0) switch
            {
                var select when (select & 8) != 0 =>
                    $"((v[{destination}] & 0x0000FFFFu) | ({half} << 16))",
                _ => $"((v[{destination}] & 0xFFFF0000u) | {half})",
            };
        }

        // ---- DPP / SDWA machinery ----

        private static bool IsSupportedDppControl(uint control) =>
            control <= 0xFF ||
            control is >= 0x101 and <= 0x10F or
                >= 0x111 and <= 0x11F or
                >= 0x121 and <= 0x12F or
                0x140 or 0x141 or
                >= 0x150 and <= 0x15F or
                >= 0x160 and <= 0x16F;

        /// <summary>Target lane + in-range flag for a DPP16 control.</summary>
        private (string TargetLane, string InRange) EmitDppSourceLane(Gen5DppControl control)
        {
            var dpp = control.Control;
            if (dpp <= 0xFF)
            {
                // Quad permute: two selector bits per lane-in-quad.
                var selected = Temp(
                    "uint",
                    $"({dpp}u >> ((sharpemu_lane & 3u) * 2u)) & 3u");
                return (Temp("uint", $"(sharpemu_lane & 0xFFFFFFFCu) + {selected}"), "true");
            }

            if (dpp is >= 0x101 and <= 0x10F)
            {
                // row_shl
                var shifted = Temp("uint", $"(sharpemu_lane & 15u) + {dpp & 15}u");
                var inRange = Temp("bool", $"{shifted} < 16u");
                return (Temp("uint", $"(sharpemu_lane & 0xFFFFFFF0u) + ({shifted} & 15u)"), inRange);
            }

            if (dpp is >= 0x111 and <= 0x11F)
            {
                // row_shr
                var inRange = Temp("bool", $"(sharpemu_lane & 15u) >= {dpp & 15}u");
                return (
                    Temp("uint", $"(sharpemu_lane & 0xFFFFFFF0u) + (((sharpemu_lane & 15u) - {dpp & 15}u) & 15u)"),
                    inRange);
            }

            if (dpp is >= 0x121 and <= 0x12F)
            {
                // row_ror
                return (
                    Temp("uint", $"(sharpemu_lane & 0xFFFFFFF0u) + (((sharpemu_lane & 15u) - {dpp & 15}u) & 15u)"),
                    "true");
            }

            var target = dpp switch
            {
                0x140 => "(sharpemu_lane & 0xFFFFFFF0u) + (15u - (sharpemu_lane & 15u))",
                0x141 => "(sharpemu_lane & 0xFFFFFFF8u) + (7u - (sharpemu_lane & 7u))",
                >= 0x150 and <= 0x15F => $"(sharpemu_lane & 0xFFFFFFF0u) + {dpp & 15}u",
                >= 0x160 and <= 0x16F => $"(sharpemu_lane & 0xFFFFFFF0u) + ((sharpemu_lane & 15u) ^ {dpp & 15}u)",
                _ => "sharpemu_lane",
            };
            return (Temp("uint", target), "true");
        }

        // Under the single-lane model (the vertex stage) every shuffle-select
        // resolves to the lane's own value (the register conceptually holds this
        // thread's value in every lane), the SPIR-V translator's no-subgroup
        // fallback; compute, mesh and pixel lanes are real simdgroup threads and
        // shuffle for real.
        private bool IsSingleLaneStage => !HasRealWaveLanes;

        private string ShuffleLane(string value, string targetLane) =>
            IsSingleLaneStage ? value : $"simd_shuffle({value}, (ushort){targetLane})";

        // Whether the source lane has its EXEC bit, read from the ballot rather than by
        // shuffling the bit over: a simdgroup is often only partly filled (a fragment
        // simdgroup on a primitive's edge, a partial compute group), simd_shuffle from a
        // lane that is not running is undefined, and simd_ballot reports it as 0 - the
        // EXEC-clear lane it stands for.
        private string LaneActiveExpression(string targetLane) =>
            IsSingleLaneStage ? "exec" : $"((sharpemu_ballot(exec) >> ({targetLane})) & 1u) != 0u";

        private string ApplyDppSource(Gen5DppControl control, string value)
        {
            var stored = Temp("uint", value);
            var (targetLane, inRange) = EmitDppSourceLane(control);
            var safeTarget = Temp("uint", $"(({inRange}) ? {targetLane} : sharpemu_lane) & 31u");
            var shuffled = Temp("uint", ShuffleLane(stored, safeTarget));
            if (control.FetchInactive)
            {
                // FI only lets an EXEC-inactive source lane be read; a lane whose source is
                // out of the row (row_shl/row_shr) still reads zero, for bound_ctrl to write.
                return Temp("uint", $"({inRange}) ? {shuffled} : 0u");
            }

            var sourceActive = Temp("bool", LaneActiveExpression(safeTarget));
            return Temp("uint", $"(({inRange}) && {sourceActive}) ? {shuffled} : 0u");
        }

        private string ApplyDpp8Source(Gen5Dpp8Control control, string value)
        {
            var stored = Temp("uint", value);
            var selector = Temp(
                "uint",
                $"({control.LaneSelectors}u >> ((sharpemu_lane & 7u) * 3u)) & 7u");
            var targetLane = Temp("uint", $"((sharpemu_lane & 0xFFFFFFF8u) + {selector}) & 31u");
            var shuffled = Temp("uint", ShuffleLane(stored, targetLane));
            if (control.FetchInactive)
            {
                return shuffled;
            }

            var sourceActive = Temp("bool", LaneActiveExpression(targetLane));
            return Temp("uint", $"{sourceActive} ? {shuffled} : 0u");
        }

        private string EmitDppWriteEnabled(Gen5DppControl control)
        {
            var (targetLane, inRange) = EmitDppSourceLane(control);
            // A wave64 in a workgroup wider than one wave runs as 32-wide simdgroups whose
            // lane index is 0..31; the thread's position in the workgroup tells which half
            // of its guest wave it is, and the upper half is rows 2 and 3.
            var row = _waveLaneCount == 64 && _stage == Gen5MslStage.Compute && !IsWave64
                ? $"((sharpemu_lane >> 4) + ((((sharpemu_local_id.x + sharpemu_local_id.y * {_localSizeX}u + " +
                    $"sharpemu_local_id.z * {_localSizeX * _localSizeY}u) >> 5) & 1u) << 1))"
                : "(sharpemu_lane >> 4)";
            var rowEnabled = $"(({control.RowMask}u >> {row}) & 1u) != 0u";
            // Each BANK_MASK bit gates four consecutive lanes (bit 24: lanes 0-3, 16-19...).
            var bankEnabled = $"(({control.BankMask}u >> ((sharpemu_lane >> 2) & 3u)) & 1u) != 0u";
            // An invalid source lane (out of the row, or EXEC-inactive without FI) reads 0 with
            // BOUND_CTRL and otherwise leaves the destination lane unwritten: with FI clear,
            // "interaction with inactive lanes is controlled by bound_ctrl" (LLVM's DPP syntax;
            // llvm.amdgcn.update.dpp keeps `old` for those lanes).
            var sourceAllows = control.BoundControl
                ? "true"
                : control.FetchInactive
                    ? inRange
                    // The ballot sits in its own statement: under the && only some lanes would run it.
                    : $"({inRange}) && {Temp("bool", LaneActiveExpression(Temp("uint", $"(({inRange}) ? {targetLane} : sharpemu_lane) & 31u")))}";
            return Temp("bool", $"({rowEnabled}) && ({bankEnabled}) && ({sourceAllows})");
        }

        private string ApplySdwaDestination(
            Gen5SdwaControl control,
            string value,
            string previous,
            bool sixteenBitResult = false)
        {
            var (shift, width) = control.DestinationSelect switch
            {
                0 => (0u, 8u),
                1 => (8u, 8u),
                2 => (16u, 8u),
                3 => (24u, 8u),
                4 => (0u, 16u),
                5 => (16u, 16u),
                // A 16-bit result is the low word of a DWORD selection, and DST_UNUSED says
                // what the high word becomes: UNUSED_PAD zeroes it. GFX10 code asks for a
                // zero-extended half this way, since a plain 16-bit write keeps the high word
                // (Black Myth: Wukong's BC6H encoder reads V_CVT_F16_F32's whole register as
                // an integer next, and kept garbage endpoints from the high word otherwise).
                _ when sixteenBitResult => (0u, 16u),
                _ => (0u, 32u),
            };
            if (width == 32)
            {
                return value;
            }

            var lowMask = width == 8 ? 0xFFu : 0xFFFFu;
            var fieldMask = lowMask << (int)shift;
            var upperStart = shift + width;
            var upperMask = upperStart == 32 ? 0u : uint.MaxValue << (int)upperStart;
            var positioned = Temp("uint", $"(({value}) & 0x{lowMask:X}u) << {shift}");
            return control.DestinationUnused switch
            {
                // 0: unused bits zeroed. 1: sign-extend upward. 2: preserve.
                0 => positioned,
                1 => Temp(
                    "uint",
                    $"{positioned} | ((({positioned} & 0x{1u << (int)(shift + width - 1):X}u) != 0u) ? 0x{upperMask:X}u : 0u)"),
                2 => Temp("uint", $"(({previous}) & 0x{~fieldMask:X}u) | {positioned}"),
                _ => throw new InvalidOperationException("reserved SDWA destination-unused mode"),
            };
        }

        // ---- compares ----

        private bool TryEmitVectorCompare(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            string condition;
            if (opcode is "VCmpClassF32" or "VCmpxClassF32")
            {
                condition = EmitCompareClass(instruction);
            }
            else if (opcode is "VCmpClassF16" or "VCmpxClassF16")
            {
                condition = EmitCompareClassF16(instruction);
            }
            else if (opcode is
                     "VCmpTruF32" or "VCmpxTruF32" or
                     "VCmpTruF16" or "VCmpxTruF16" or
                     "VCmpTI32" or "VCmpTU32" or "VCmpxTI32" or "VCmpxTU32" or
                     // The 64-bit F/T forms are constant compares like their 32-bit
                     // siblings; without them TrimCompare hands "F"/"T" to the
                     // integer operator table and the opcode reads as unsupported.
                     "VCmpTI64" or "VCmpxTI64" or "VCmpTU64" or "VCmpxTU64")
            {
                condition = "true";
            }
            else if (opcode is
                     "VCmpFF32" or "VCmpxFF32" or
                     "VCmpFF16" or "VCmpxFF16" or
                     "VCmpFI32" or "VCmpFU32" or "VCmpxFI32" or "VCmpxFU32" or
                     "VCmpFI64" or "VCmpxFI64" or "VCmpFU64" or "VCmpxFU64")
            {
                condition = "false";
            }
            else if (opcode is
                     "VCmpOF32" or "VCmpxOF32" or
                     "VCmpOF16" or "VCmpxOF16")
            {
                var left = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? F16(instruction, 0)
                    : F(instruction, 0);
                var right = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? F16(instruction, 1)
                    : F(instruction, 1);
                condition = $"(!isnan({left}) && !isnan({right}))";
            }
            else if (opcode is
                     "VCmpUF32" or "VCmpxUF32" or
                     "VCmpUF16" or "VCmpxUF16")
            {
                var left = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? F16(instruction, 0)
                    : F(instruction, 0);
                var right = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? F16(instruction, 1)
                    : F(instruction, 1);
                condition = $"(isnan({left}) || isnan({right}))";
            }
            else if (opcode.EndsWith("F32", StringComparison.Ordinal) ||
                     opcode.EndsWith("F16", StringComparison.Ordinal))
            {
                // Ordered compares are the plain C operators (false on NaN);
                // the Nxx forms are their unordered negations (true on NaN).
                var (op, unordered) = TrimCompare(opcode) switch
                {
                    "Lt" => ("<", false),
                    "Eq" => ("==", false),
                    "Le" => ("<=", false),
                    "Gt" => (">", false),
                    // Less-or-greater: ordered, so false on NaN (C's != is true there).
                    "Lg" => ("<>", false),
                    "Ge" => (">=", false),
                    "Neq" => ("==", true),
                    "Nlt" => ("<", true),
                    "Nle" => ("<=", true),
                    "Ngt" => (">", true),
                    "Nge" => (">=", true),
                    "Nlg" => ("<>", true),
                    _ => (string.Empty, false),
                };
                if (op.Length == 0)
                {
                    error = $"unsupported float compare {opcode}";
                    return false;
                }

                var left = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? F16(instruction, 0)
                    : F(instruction, 0);
                var right = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? F16(instruction, 1)
                    : F(instruction, 1);
                var comparison = op == "<>"
                    ? $"({left} < {right} || {left} > {right})"
                    : $"({left} {op} {right})";
                condition = unordered ? $"(!{comparison})" : comparison;
            }
            else
            {
                var signed16 = opcode.EndsWith("I16", StringComparison.Ordinal);
                var unsigned16 = opcode.EndsWith("U16", StringComparison.Ordinal);
                var signed = signed16 ||
                    opcode.EndsWith("I32", StringComparison.Ordinal) ||
                    opcode.EndsWith("I64", StringComparison.Ordinal);
                var wide = opcode.EndsWith("I64", StringComparison.Ordinal) ||
                    opcode.EndsWith("U64", StringComparison.Ordinal);
                var op = TrimCompare(opcode) switch
                {
                    "Eq" => "==",
                    "Ne" => "!=",
                    "Lt" => "<",
                    "Le" => "<=",
                    "Gt" => ">",
                    "Ge" => ">=",
                    _ => string.Empty,
                };
                if (op.Length == 0)
                {
                    error = $"unsupported integer compare {opcode}";
                    return false;
                }

                var left = wide ? RawSource64(instruction, 0) : RawSource(instruction, 0);
                var right = wide ? RawSource64(instruction, 1) : RawSource(instruction, 1);
                condition = signed
                    ? signed16
                        ? $"(int(short(({left}) & 0xFFFFu)) {op} int(short(({right}) & 0xFFFFu)))"
                        : wide
                        ? $"(as_type<long>({left}) {op} as_type<long>({right}))"
                        : $"(as_type<int>({left}) {op} as_type<int>({right}))"
                    // A 16-bit compare reads only the low half; whatever the
                    // upper half carries must not change the result.
                    : unsigned16
                        ? $"((({left}) & 0xFFFFu) {op} (({right}) & 0xFFFFu))"
                        : $"(({left}) {op} ({right}))";
            }

            // Only EXEC-enabled lanes can pass; balloting the raw condition
            // would leak results from disabled lanes into saveexec/branches.
            var active = Temp("bool", ExecCondition(condition));
            if (instruction.Control is Gen5DppControl compareDpp)
            {
                var writeEnabled = EmitDppWriteEnabled(compareDpp);
                // A lane whose write is disabled keeps what it had: its VCC bit for V_CMP,
                // its EXEC bit for V_CMPX (VCC is unrelated to a V_CMPX).
                var kept = opcode.StartsWith("VCmpx", StringComparison.Ordinal) ? "exec" : "vcc";
                active = Temp("bool", $"({writeEnabled}) ? {active} : {kept}");
            }

            if (opcode.StartsWith("VCmpx", StringComparison.Ordinal))
            {
                // GFX10 VCMPX writes EXEC only.
                Line($"exec = {active};");
                EmitBallotStore(ExecLoRegister, "exec");
            }
            else
            {
                StoreMaskBit(CompareMaskDestination(instruction), active);
            }

            return true;
        }

        /// <summary>V_CMP_CLASS_F32 on the f32 bits after abs/neg. A signalling NaN (quiet bit
        /// 22 clear) is class bit 0 and a quiet NaN bit 1. Float compares are not used: the
        /// host flushes f32 denormals in them, so a denormal read as a zero (bits 5/6) instead
        /// of a subnormal (bits 4/7).</summary>
        private string EmitCompareClass(Gen5ShaderInstruction instruction)
        {
            var bits = Temp("uint", AsUInt(F(instruction, 0)));
            var mask = Temp("uint", RawSource(instruction, 1));
            var exponent = Temp("uint", $"{bits} & 0x7F800000u");
            var fraction = Temp("uint", $"{bits} & 0x7FFFFFu");
            var negative = Temp("bool", $"({bits} & 0x80000000u) != 0u");
            var nan = Temp("bool", $"{exponent} == 0x7F800000u && {fraction} != 0u");
            var infinite = Temp("bool", $"{exponent} == 0x7F800000u && {fraction} == 0u");
            var zero = Temp("bool", $"{exponent} == 0u && {fraction} == 0u");
            var subnormal = Temp("bool", $"{exponent} == 0u && {fraction} != 0u");
            var normal = Temp("bool", $"{exponent} != 0u && {exponent} != 0x7F800000u");
            string Bit(int index) => $"(({mask} >> {index}u) & 1u) != 0u";
            // Class bits: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -subnormal,
            // 5 -zero, 6 +zero, 7 +subnormal, 8 +normal, 9 +inf.
            return Temp(
                "bool",
                $"({Bit(0)} && {nan} && ({fraction} & 0x400000u) == 0u) || " +
                $"({Bit(1)} && {nan} && ({fraction} & 0x400000u) != 0u) || " +
                $"({Bit(2)} && {infinite} && {negative}) || " +
                $"({Bit(3)} && {normal} && {negative}) || " +
                $"({Bit(4)} && {subnormal} && {negative}) || " +
                $"({Bit(5)} && {zero} && {negative}) || " +
                $"({Bit(6)} && {zero} && !{negative}) || " +
                $"({Bit(7)} && {subnormal} && !{negative}) || " +
                $"({Bit(8)} && {normal} && !{negative}) || " +
                $"({Bit(9)} && {infinite} && !{negative})");
        }

        /// <summary>The register a V_CMP writes its mask to: the VOP3 encoding names it in
        /// the vdst byte (an SGPR pair, VCC, EXEC or NULL), SDWA in its sdst field, and
        /// the VOPC encoding always writes VCC.</summary>
        private static uint CompareMaskDestination(Gen5ShaderInstruction instruction) =>
            instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister } destination, ..]
                ? destination.Value
                : instruction.Control is Gen5SdwaControl { ScalarDestination: { } scalarDestination }
                    ? scalarDestination
                    : VccLoRegister;

        private static string TrimCompare(string opcode)
        {
            var trimmed = opcode.StartsWith("VCmpx", StringComparison.Ordinal)
                ? opcode["VCmpx".Length..]
                : opcode["VCmp".Length..];
            return trimmed[..^3];
        }

        private void StoreCarryOut(Gen5ShaderInstruction instruction, string carryCondition)
        {
            var active = Temp("bool", ExecCondition($"({carryCondition})"));
            var target = instruction.Control is Gen5Vop3Control { ScalarDestination: { } register }
                ? register
                : VccLoRegister;
            StoreMaskBit(target, active);
        }

        /// <summary>
        /// Writes this lane's bit of a wave mask: VCC/EXEC update the per-lane
        /// bool and mirror the ballot into their architectural SGPRs; a plain
        /// SGPR receives the ballot of the per-lane condition (one SGPR in
        /// wave32, a pair in wave64); NULL discards it.
        /// </summary>
        private void StoreMaskBit(uint register, string condition)
        {
            switch (register)
            {
                case NullRegister:
                    return;
                case VccLoRegister:
                    Line($"vcc = {condition};");
                    EmitBallotStore(VccLoRegister, "vcc");
                    return;
                case ExecLoRegister:
                    Line($"exec = {condition};");
                    EmitBallotStore(ExecLoRegister, "exec");
                    return;
                default:
                    if (register >= ScalarRegisterFileCount)
                    {
                        return;
                    }

                    if (!IsWave64 && !IsNarrowWave64)
                    {
                        // A wave32 mask is the single SGPR; the next one is not part of it.
                        // A narrow wave64 mask is a pair whose high half has no lanes, so it
                        // takes EmitBallotStore's zeroing instead.
                        Line($"s[{register}] = sharpemu_ballot({condition});");
                        return;
                    }

                    EmitBallotStore(register, condition);
                    return;
            }
        }

        /// <summary>Broadcasts <paramref name="value"/> from the first guest-active
        /// lane (lowest set bit of the 64-lane EXEC mask) to all lanes — mirroring the
        /// SPIR-V translator's BroadcastFirstWave64Active — in one exchange: each half
        /// publishes its EXEC ballot and the value of its own first active lane, and the
        /// low half's wins unless it has none. Returns the temp holding the result.</summary>
        private string EmitWave64ReadFirstLane(string value)
        {
            Line(WaveSetFlip);
            var ballot = Temp("uint", "sharpemu_ballot(exec)");
            // The ballot only has lanes that are running, so the shuffle reads one.
            var halfFirst = Temp("uint", ShuffleLane(value, $"({ballot} == 0u ? 0u : (uint)ctz({ballot}))"));
            Line($"{WaveHalfBallotSlot} = {ballot};");
            Line($"{WaveHalfValueSlot} = {halfFirst};");
            Line(WaveExchangeBarrier);
            var lo = Temp("uint", WaveSlot(0));
            var hi = Temp("uint", WaveSlot(1));
            // With no active lane in either half the first lane is lane 0, which the low
            // half's shuffle already read into slot 2.
            return Temp("uint", $"({lo} != 0u || {hi} == 0u) ? {WaveSlot(2)} : {WaveSlot(3)}");
        }

        /// <summary>Broadcasts <paramref name="value"/> from guest lane
        /// <paramref name="lane"/> (0..63) to all lanes. Like V_READLANE it reads the
        /// lane whether or not EXEC has it.</summary>
        private string EmitWave64ReadLane(string value, string lane) =>
            EmitWave64BroadcastFromLane(value, lane);

        /// <summary>Stores the wave ballot of <paramref name="condition"/> into the
        /// mask register pair (low, low+1). Wave32 fills the low dword and clears
        /// the high; wave64 bridges both 32-wide halves through threadgroup
        /// scratch so the pair holds the full 64-lane mask. The bridging barrier
        /// is safe because the guest program's scalar PC keeps all 64 lanes in
        /// lockstep through the dispatcher (one wave per threadgroup).</summary>
        private void EmitBallotStore(uint loRegister, string condition)
        {
            var hiRegister = loRegister + 1;
            if (!IsWave64)
            {
                Line($"s[{loRegister}] = sharpemu_ballot({condition});");
                if (hiRegister < ScalarRegisterFileCount &&
                    !(loRegister == VccLoRegister && VccIsLowHalfOnly))
                {
                    Line($"s[{hiRegister}] = 0u;");
                }

                return;
            }

            if (_options.DeadMaskBallotElision && hiRegister < ScalarRegisterFileCount)
            {
                // The exchange costs a threadgroup barrier, and it is only worth it if
                // the 64-bit pair is read before it is overwritten. That is not knowable
                // until every block has been emitted, so leave a marker for
                // Gen5MslTranslator.MaskBallots to resolve. Remember the per-lane bool
                // too: a consumer that wants this lane's bit of the mask takes it from
                // there (Gen5MslTranslator.LaneMasks.cs). VCC and EXEC already have one.
                var laneCondition = condition;
                if (LaneMasksEnabled && loRegister != VccLoRegister && loRegister != ExecLoRegister)
                {
                    laneCondition = LaneVariableName(loRegister);
                    _laneVariables.Add(loRegister);
                    Line($"{laneCondition} = {condition};");
                }

                Line($"{MaskBallotMarker}({loRegister}, {laneCondition});");
                return;
            }

            foreach (var statement in WaveBallotExchange(condition))
            {
                Line(statement);
            }

            Line($"s[{loRegister}] = {WaveSlot(0)};");
            if (hiRegister < ScalarRegisterFileCount)
            {
                Line($"s[{hiRegister}] = {WaveSlot(1)};");
            }
        }

        // ---- permlane / cube ----

        private string EmitPermlane16(Gen5ShaderInstruction instruction, bool exchangeRows)
        {
            if (instruction.Control is not Gen5Vop3Control control ||
                (control.OperandSelect & ~3u) != 0 ||
                control.AbsoluteMask != 0 ||
                control.NegateMask != 0 ||
                control.OutputModifier != 0 ||
                control.Clamp)
            {
                throw new NotSupportedException(
                    $"invalid permlane modifiers for {instruction.Opcode}");
            }

            var value = Temp("uint", RawSource(instruction, 0));
            var selectorLow = Temp("uint", RawSource(instruction, 1));
            var selectorHigh = Temp("uint", RawSource(instruction, 2));
            var localLane = Temp("uint", "sharpemu_lane & 15u");
            var selector = Temp(
                "uint",
                $"({localLane} < 8u ? ({selectorLow} >> ({localLane} << 2)) : ({selectorHigh} >> (({localLane} - 8u) << 2))) & 15u");
            var rowBase = exchangeRows
                ? "((sharpemu_lane & 0xFFFFFFF0u) ^ 16u)"
                : "(sharpemu_lane & 0xFFFFFFF0u)";
            var targetLane = Temp("uint", $"({rowBase} + {selector}) & 31u");
            var shuffled = Temp("uint", ShuffleLane(value, targetLane));
            var fetchInactive = (control.OperandSelect & 1) != 0;
            if (fetchInactive)
            {
                return shuffled;
            }

            // op_sel[1] is the DPP BOUND_CTRL bit: an EXEC-inactive source lane (FI clear) gives
            // 0 with it and otherwise leaves the destination lane as it was.
            var sourceActive = Temp("bool", LaneActiveExpression(targetLane));
            var invalid = (control.OperandSelect & 2) != 0 ? "0u" : $"v[{DestinationVector(instruction)}]";
            if (_stage == Gen5MslStage.Pixel && !IsSingleLaneStage)
            {
                // A partner fragment that was never launched is not an EXEC-clear guest lane:
                // the guest wave has it, and its reductions enable it and give it the identity.
                // The reading lane's own value stands in for that identity in an idempotent
                // reduction (min, max, and, or), where BOUND_CTRL's zero made Ghost of
                // Tsushima's wave minimum 0 forever.
                var sourcePresent = Temp("bool", $"((sharpemu_ballot(true) >> ({targetLane})) & 1u) != 0u");
                return Temp("uint", $"{sourceActive} ? {shuffled} : ({sourcePresent} ? {invalid} : {value})");
            }

            return Temp("uint", $"{sourceActive} ? {shuffled} : {invalid}");
        }

        private enum CubeCoordinate
        {
            Id,
            Sc,
            Tc,
            Ma,
        }

        private string EmitCubeCoordinate(
            Gen5ShaderInstruction instruction,
            CubeCoordinate coordinate)
        {
            // The coordinates are f32 operands, so a denormal is a zero of its sign.
            var x = Temp("float", $"sharpemu_flush_f32_denorm({F(instruction, 0)})");
            var y = Temp("float", $"sharpemu_flush_f32_denorm({F(instruction, 1)})");
            var z = Temp("float", $"sharpemu_flush_f32_denorm({F(instruction, 2)})");
            // The conditions are the ISA's pseudo-code, ordered compares included: with a NaN
            // coordinate they are false, where fmax would skip the NaN.
            var isZMax = Temp("bool", $"fabs({z}) >= fabs({x}) && fabs({z}) >= fabs({y})");
            var yGeX = Temp("bool", $"fabs({y}) >= fabs({x})");
            if (coordinate == CubeCoordinate.Ma)
            {
                // 2.0 * the major axis coordinate itself, sign included (ISA 12.12 V_CUBEMA_F32).
                return FloatResult(instruction, $"2.0f * ({isZMax} ? {z} : ({yGeX} ? {y} : {x}))");
            }

            var isYMax = Temp("bool", $"!{isZMax} && {yGeX}");
            switch (coordinate)
            {
                case CubeCoordinate.Id:
                {
                    var zCase = $"({z} < 0.0f ? 5.0f : 4.0f)";
                    var yCase = $"({y} < 0.0f ? 3.0f : 2.0f)";
                    var xCase = $"({x} < 0.0f ? 1.0f : 0.0f)";
                    return FloatResult(
                        instruction,
                        $"({isZMax} ? {zCase} : ({yGeX} ? {yCase} : {xCase}))");
                }
                case CubeCoordinate.Sc:
                {
                    var zCase = $"({z} < 0.0f ? (-{x}) : {x})";
                    var xCase = $"({x} < 0.0f ? {z} : (-{z}))";
                    return FloatResult(
                        instruction,
                        $"({isZMax} ? {zCase} : ({isYMax} ? {x} : {xCase}))");
                }
                default:
                {
                    var yCase = $"({y} < 0.0f ? (-{z}) : {z})";
                    return FloatResult(
                        instruction,
                        $"({isYMax} ? {yCase} : (-{y}))");
                }
            }
        }

        // ---- scalar ALU ----

        private bool TryEmitScalarAlu(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Encoding == Gen5ShaderEncoding.Sopc)
            {
                return TryEmitScalarCompare(instruction, out error);
            }

            if (instruction.Destinations.Count == 0 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister)
            {
                error = "missing scalar destination";
                return false;
            }

            var destination = instruction.Destinations[0].Value;
            if (instruction.Encoding == Gen5ShaderEncoding.Sopk)
            {
                var immediate = unchecked((uint)(short)(instruction.Words[0] & 0xFFFF));
                if (instruction.Opcode.StartsWith("SCmpk", StringComparison.Ordinal))
                {
                    // The unsigned compares take SIMM16 zero-extended; only the signed ones sign-extend it.
                    if (instruction.Opcode.EndsWith("U32", StringComparison.Ordinal))
                    {
                        immediate = instruction.Words[0] & 0xFFFF;
                    }
                    return TryEmitScalarCompareK(instruction, destination, immediate, out error);
                }

                var current = ScalarExpression(destination);
                var value = instruction.Opcode switch
                {
                    "SMovkI32" => FormatUInt(immediate),
                    // S_CMOVK_I32: if SCC, D = signext(SIMM16); SCC is left alone.
                    "SCmovkI32" => $"(scc ? {FormatUInt(immediate)} : {current})",
                    "SAddkI32" => $"({current} + {FormatUInt(immediate)})",
                    "SMulkI32" => $"({current} * {FormatUInt(immediate)})",
                    _ => string.Empty,
                };
                if (value.Length == 0)
                {
                    error = $"unsupported scalar immediate {instruction.Opcode}";
                    return false;
                }

                var stored = Temp("uint", value);
                // RDNA2 ISA: S_ADDK_I32 writes SCC = signed overflow, exactly like
                // S_ADD_I32. S_MOVK_I32 and S_MULK_I32 leave SCC alone.
                if (instruction.Opcode == "SAddkI32")
                {
                    var addend = FormatUInt(immediate);
                    Line(
                        $"scc = ((~({current} ^ {addend}) & ({current} ^ {stored})) >> 31) != 0u;");
                }

                StoreScalar(destination, stored);
                return true;
            }

            if (instruction.Opcode == "SGetpcB64")
            {
                {
                    // The shader base is pushed per draw; the program offset is added to it.
                    var (baseLow, baseHigh) = ShaderBaseWords();
                    var offset = unchecked(instruction.ProgramOffset + (ulong)(instruction.Words.Count * sizeof(uint)));
                    var address = Temp("ulong", $"((ulong){baseLow} | ((ulong){baseHigh} << 32)) + {offset}ul");
                    StoreScalar(destination, $"(uint){address}");
                    StoreScalar(destination + 1, $"(uint)({address} >> 32)");
                    return true;
                }
            }

            if (instruction.Opcode is "SQuadmaskB32" or "SQuadmaskB64")
            {
                var wide = instruction.Opcode == "SQuadmaskB64";
                var source = Temp(wide ? "ulong" : "uint", wide ? RawSource64(instruction, 0) : RawSource(instruction, 0));
                var result = Temp("uint", "0u");
                for (var quad = 0; quad < (wide ? 16 : 8); quad++)
                    Line($"{result} |= (({source} >> {quad * 4}) & 15{(wide ? "ul" : "u")}) != 0 ? {1u << quad}u : 0u;");
                StoreScalar(destination, result);
                if (wide) StoreScalar(destination + 1, "0u");
                Line($"scc = {result} != 0u;");
                return true;
            }

            switch (instruction.Opcode)
            {
                case "SMovrelsB32":
                case "SMovrelsB64":
                case "SMovreldB32":
                case "SMovreldB64":
                case "SMovrelsd2B32":
                    return TryEmitScalarMoveRelative(instruction, out error);
                case "SAndn1WrexecB32":
                case "SAndn1WrexecB64":
                case "SAndn2WrexecB32":
                case "SAndn2WrexecB64":
                    EmitWriteExec(instruction, destination);
                    return true;
                case "SBcnt0I32B64":
                {
                    var wide = Temp("ulong", RawSource64(instruction, 0));
                    var zeros = Temp("uint", $"(uint)popcount(~{wide})");
                    StoreScalar(destination, zeros);
                    Line($"scc = {zeros} != 0u;");
                    return true;
                }
                case "SFF0I32B64":
                {
                    var wide = Temp("ulong", RawSource64(instruction, 0));
                    StoreScalar(
                        destination,
                        Temp("uint", $"{wide} == 0xFFFFFFFFFFFFFFFFul ? 0xFFFFFFFFu : (uint)ctz(~{wide})"));
                    return true;
                }
                case "SFlbitI32I64":
                {
                    // The first bit below the sign that differs from it; -1 for 0 and -1.
                    var wide = Temp("ulong", RawSource64(instruction, 0));
                    var magnitude = Temp("ulong", $"(as_type<long>({wide}) < 0l) ? ~{wide} : {wide}");
                    var high = Temp("uint", $"(uint)({magnitude} >> 32)");
                    var low = Temp("uint", $"(uint){magnitude}");
                    StoreScalar(
                        destination,
                        Temp("uint", $"{high} != 0u ? (uint)clz({high}) : ({low} != 0u ? 32u + (uint)clz({low}) : 0xFFFFFFFFu)"));
                    return true;
                }
                case "SBrevB64":
                {
                    var wide = Temp("ulong", RawSource64(instruction, 0));
                    StoreScalar(destination, Temp("uint", $"reverse_bits((uint)({wide} >> 32))"));
                    StoreScalar(destination + 1, Temp("uint", $"reverse_bits((uint){wide})"));
                    return true;
                }
            }

            if (instruction.Opcode == "SBcnt1I32B64")
            {
                var wide = Temp("ulong", RawSource64(instruction, 0));
                var bitCountResult = Temp("uint", $"(uint)popcount({wide})");
                StoreScalar(destination, bitCountResult);
                Line($"scc = {bitCountResult} != 0u;");
                return true;
            }

            if (instruction.Opcode == "SFF1I32B64")
            {
                var wide = Temp("ulong", RawSource64(instruction, 0));
                var result = Temp(
                    "uint",
                    $"{wide} == 0ul ? 0xFFFFFFFFu : (uint)ctz({wide})");
                StoreScalar(destination, result);
                return true;
            }

            if (instruction.Opcode == "SFlbitI32B64")
            {
                var source = Temp("ulong", RawSource64(instruction, 0));
                var low = Temp("uint", $"(uint){source}");
                var high = Temp("uint", $"(uint)({source} >> 32)");
                var result = Temp("uint",
                    $"{high} != 0u ? clz({high}) : ({low} != 0u ? 32u + clz({low}) : 0xFFFFFFFFu)");
                StoreScalar(destination, result);
                return true;
            }

            if (instruction.Opcode.EndsWith("B64", StringComparison.Ordinal) ||
                instruction.Opcode == "SAshrI64" ||
                instruction.Opcode is "SBfeU64" or "SBfeI64")
            {
                return TryEmitScalar64(instruction, destination, out error);
            }

            var left = Temp("uint", RawSource(instruction, 0));
            if (instruction.Opcode.EndsWith("SaveexecB32", StringComparison.Ordinal))
            {
                var oldExec = Temp("uint", $"s[{ExecLoRegister}]");
                var operation = instruction.Opcode[1..instruction.Opcode.IndexOf(
                    "Saveexec",
                    StringComparison.Ordinal)];
                var combined = operation switch
                {
                    "And" => $"({left} & {oldExec})",
                    "Or" => $"({left} | {oldExec})",
                    "Xor" => $"({left} ^ {oldExec})",
                    "Nand" => $"~({left} & {oldExec})",
                    "Nor" => $"~({left} | {oldExec})",
                    "Xnor" => $"~({left} ^ {oldExec})",
                    "Andn1" => $"(~{left} & {oldExec})",
                    "Andn2" => $"({left} & ~{oldExec})",
                    "Orn1" => $"(~{left} | {oldExec})",
                    "Orn2" => $"({left} | ~{oldExec})",
                    _ => string.Empty,
                };
                if (combined.Length == 0)
                {
                    error = $"unsupported scalar 32-bit saveexec opcode {instruction.Opcode}";
                    return false;
                }

                var mask = Temp("uint", ClampExecToLanes(combined));
                StoreScalar(destination, oldExec);
                Line($"s[{ExecLoRegister}] = {mask};");
                Line($"s[{ExecHiRegister}] = 0u;");
                Line($"exec = (({mask} >> sharpemu_lane) & 1u) != 0u;");
                Line($"scc = {mask} != 0u;");
                return true;
            }

            switch (instruction.Opcode)
            {
                case "SMovB32":
                    StoreScalar(destination, left);
                    return true;
                case "SNotB32":
                {
                    var result = Temp("uint", $"~{left}");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SAbsI32":
                {
                    var sign = Temp("uint", $"0u - ({left} >> 31u)");
                    var result = Temp("uint", $"({left} ^ {sign}) - {sign}");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SWqmB32" when IsSingleLaneStage:
                {
                    // Whole-quad mode adds the lane's quad neighbours, which a single-lane
                    // stage does not have: Metal runs a fragment's helper lanes itself.
                    var result = Temp("uint", left);
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SWqmB32":
                {
                    var quadAny = Temp(
                        "uint",
                        $"({left} | ({left} >> 1) | ({left} >> 2) | ({left} >> 3)) & 0x11111111u");
                    var result = Temp("uint", $"{quadAny} * 0xFu");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SBrevB32":
                {
                    var result = Temp("uint", $"reverse_bits({left})");
                    StoreScalar(destination, result);
                    return true;
                }
                case "SBitreplicateB64B32":
                {
                    var source = Temp("uint", left);
                    for (uint half = 0; half < 2; half++)
                    {
                        var expanded = Temp("uint", "0u");
                        for (uint bit = 0; bit < 16; bit++)
                            Line($"{expanded} |= ((({source} >> {half * 16 + bit}u) & 1u) * 3u) << {bit * 2}u;");
                        StoreScalar(destination + half, expanded);
                    }
                    return true;
                }
                case "SBcnt1I32B32":
                case "SBcnt0I32B32":
                {
                    var result = Temp("uint", instruction.Opcode == "SBcnt1I32B32" ? $"popcount({left})" : $"popcount(~{left})");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SFF0I32B32":
                    StoreScalar(destination, Temp("uint", $"{left} == 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint)ctz(~{left})"));
                    return true;
                case "SFlbitI32":
                {
                    // The first bit below the sign that differs from it; -1 for 0 and -1.
                    var magnitude = Temp("uint", $"(({left} & 0x80000000u) != 0u) ? ~{left} : {left}");
                    StoreScalar(destination, Temp("uint", $"{magnitude} == 0u ? 0xFFFFFFFFu : (uint)clz({magnitude})"));
                    return true;
                }
                case "SSextI32I8":
                    StoreScalar(destination, Temp("uint", AsUInt($"(as_type<int>({left} << 24u) >> 24)")));
                    return true;
                case "SSextI32I16":
                    StoreScalar(destination, Temp("uint", AsUInt($"(as_type<int>({left} << 16u) >> 16)")));
                    return true;
                case "SFF1I32B32":
                case "SFlbitI32B32":
                {
                    var result = Temp(
                        "uint",
                        $"{left} == 0u ? 0xFFFFFFFFu : (uint){(instruction.Opcode == "SFlbitI32B32" ? "clz" : "ctz")}({left})");
                    StoreScalar(destination, result);
                    return true;
                }
                case "SBitset0B32":
                    StoreScalar(
                        destination,
                        $"{ScalarExpression(destination)} & ~(1u << ({left} & 31u))");
                    return true;
                case "SBitset1B32":
                    StoreScalar(
                        destination,
                        $"{ScalarExpression(destination)} | (1u << ({left} & 31u))");
                    return true;
            }

            if (instruction.Sources.Count < 2)
            {
                error = $"missing scalar source for {instruction.Opcode}";
                return false;
            }

            var right = Temp("uint", RawSource(instruction, 1));
            string resultExpression;
            string sccStatement;
            switch (instruction.Opcode)
            {
                case "SAddU32":
                    resultExpression = $"({left} + {right})";
                    sccStatement = "RESULT < " + left;
                    break;
                case "SSubU32":
                    resultExpression = $"({left} - {right})";
                    sccStatement = $"{right} > {left}";
                    break;
                case "SAddI32":
                    resultExpression = $"({left} + {right})";
                    sccStatement = $"((~({left} ^ {right}) & ({left} ^ RESULT)) >> 31) != 0u";
                    break;
                case "SSubI32":
                    resultExpression = $"({left} - {right})";
                    sccStatement = $"(((({left} ^ {right})) & ({left} ^ RESULT)) >> 31) != 0u";
                    break;
                case "SAddF32":
                    resultExpression = $"as_type<uint>(as_type<float>({left}) + as_type<float>({right}))";
                    sccStatement = string.Empty;
                    break;
                case "SSubF32":
                    resultExpression = $"as_type<uint>(as_type<float>({left}) - as_type<float>({right}))";
                    sccStatement = string.Empty;
                    break;
                case "SAddcU32":
                {
                    var partial = Temp("uint", $"{left} + {right}");
                    var sum = Temp("uint", $"{partial} + (scc ? 1u : 0u)");
                    Line($"scc = ({partial} < {left}) || ({sum} < {partial});");
                    StoreScalar(destination, sum);
                    return true;
                }
                case "SSubbU32":
                {
                    var borrow = Temp("uint", "scc ? 1u : 0u");
                    var partial = Temp("uint", $"{left} - {right}");
                    var difference = Temp("uint", $"{partial} - {borrow}");
                    Line($"scc = ({right} > {left}) || (({borrow} == 1u) && ({right} == {left}));");
                    StoreScalar(destination, difference);
                    return true;
                }
                case "SMulI32":
                    resultExpression = $"({left} * {right})";
                    sccStatement = string.Empty;
                    break;
                // D = |S0 - S1| of the wrapped 32-bit difference; SCC = D != 0.
                case "SAbsdiffI32":
                    resultExpression = $"(as_type<int>({left} - {right}) < 0 ? {right} - {left} : {left} - {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SMulHiU32":
                    resultExpression = $"mulhi({left}, {right})";
                    sccStatement = string.Empty;
                    break;
                case "SMulHiI32":
                    resultExpression = $"as_type<uint>(mulhi(as_type<int>({left}), as_type<int>({right})))";
                    sccStatement = string.Empty;
                    break;
                case "SAndB32":
                    resultExpression = $"({left} & {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SOrB32":
                    resultExpression = $"({left} | {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SXorB32":
                    resultExpression = $"({left} ^ {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SNandB32":
                    resultExpression = $"~({left} & {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SNorB32":
                    resultExpression = $"~({left} | {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SXnorB32":
                    resultExpression = $"~({left} ^ {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SAndn2B32":
                    resultExpression = $"({left} & ~{right})";
                    sccStatement = "NONZERO";
                    break;
                case "SOrn2B32":
                    resultExpression = $"({left} | ~{right})";
                    sccStatement = "NONZERO";
                    break;
                case "SLshlB32":
                    resultExpression = $"({left} << ({right} & 31u))";
                    sccStatement = "NONZERO";
                    break;
                case "SLshrB32":
                    resultExpression = $"({left} >> ({right} & 31u))";
                    sccStatement = "NONZERO";
                    break;
                case "SAshrI32":
                    resultExpression = $"(uint)(as_type<int>({left}) >> ({right} & 31u))";
                    sccStatement = "NONZERO";
                    break;
                case "SBfmB32":
                    resultExpression = $"(((1u << ({left} & 31u)) - 1u) << ({right} & 31u))";
                    sccStatement = string.Empty;
                    break;
                case "SBfeU32":
                case "SBfeI32":
                {
                    // Width clamps to the bits remaining above the offset.
                    var offset = Temp("uint", $"{right} & 31u");
                    var width = Temp(
                        "uint",
                        $"min(({right} >> 16) & 0x7Fu, 32u - {offset})");
                    var result = instruction.Opcode == "SBfeI32"
                        ? Temp(
                            "uint",
                            $"{width} == 0u ? 0u : (uint)extract_bits(as_type<int>({left}), {offset}, {width})")
                        : Temp(
                            "uint",
                            $"{width} == 0u ? 0u : extract_bits({left}, {offset}, {width})");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SCselectB32":
                    resultExpression = $"(scc ? {left} : {right})";
                    sccStatement = string.Empty;
                    break;
                case "SMinU32":
                    resultExpression = $"min({left}, {right})";
                    sccStatement = $"{left} < {right}";
                    break;
                case "SMaxU32":
                    resultExpression = $"max({left}, {right})";
                    sccStatement = $"{left} > {right}";
                    break;
                case "SMinI32":
                    resultExpression = $"(uint)min(as_type<int>({left}), as_type<int>({right}))";
                    sccStatement = $"as_type<int>({left}) < as_type<int>({right})";
                    break;
                case "SMaxI32":
                    resultExpression = $"(uint)max(as_type<int>({left}), as_type<int>({right}))";
                    sccStatement = $"as_type<int>({left}) > as_type<int>({right})";
                    break;
                case "SLshl1AddU32":
                case "SLshl2AddU32":
                case "SLshl3AddU32":
                case "SLshl4AddU32":
                {
                    var shift = (uint)(instruction.Opcode[5] - '0');
                    resultExpression = $"(({left} << {shift}) + {right})";
                    // SCC is the unsigned carry out of the 64-bit sum.
                    sccStatement = $"((((ulong){left}) << {shift}) + (ulong){right}) > 0xFFFFFFFFul";
                    break;
                }
                case "SPackLlB32B16":
                    resultExpression = $"(({left} & 0xFFFFu) | ({right} << 16))";
                    sccStatement = string.Empty;
                    break;
                case "SPackLhB32B16":
                    resultExpression = $"(({left} & 0xFFFFu) | ({right} & 0xFFFF0000u))";
                    sccStatement = string.Empty;
                    break;
                case "SPackHhB32B16":
                    resultExpression = $"(({left} >> 16) | ({right} & 0xFFFF0000u))";
                    sccStatement = string.Empty;
                    break;
                default:
                    error = $"unsupported scalar opcode {instruction.Opcode}";
                    return false;
            }

            var value2 = Temp("uint", resultExpression);
            StoreScalar(destination, value2);
            if (sccStatement == "NONZERO")
            {
                Line($"scc = {value2} != 0u;");
            }
            else if (sccStatement.Length != 0)
            {
                Line($"scc = {sccStatement.Replace("RESULT", value2)};");
            }

            return true;
        }

        private bool TryEmitScalarCompare(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 2)
            {
                error = "missing scalar compare source";
                return false;
            }

            if (instruction.Opcode is "SCmpEqU64" or "SCmpLgU64")
            {
                var wideLeft = Temp("ulong", RawSource64(instruction, 0));
                var wideRight = Temp("ulong", RawSource64(instruction, 1));
                var comparison = instruction.Opcode == "SCmpEqU64" ? "==" : "!=";
                Line($"scc = {wideLeft} {comparison} {wideRight};");
                return true;
            }

            if (instruction.Opcode is "SBitcmp0B64" or "SBitcmp1B64")
            {
                // SCC = (S0.u64[S1.u[5:0]] == 0/1) (ISA 12.4).
                var wideValue = Temp("ulong", RawSource64(instruction, 0));
                var bitIndex = Temp("uint", RawSource(instruction, 1));
                var bitSet = $"(({wideValue} >> ({bitIndex} & 63u)) & 1ul) != 0ul";
                Line(instruction.Opcode == "SBitcmp1B64" ? $"scc = {bitSet};" : $"scc = !({bitSet});");
                return true;
            }

            var left = Temp("uint", RawSource(instruction, 0));
            var right = Temp("uint", RawSource(instruction, 1));
            if (instruction.Opcode is "SBitcmp0B32" or "SBitcmp1B32")
            {
                var isSet = $"(({left} >> ({right} & 31u)) & 1u) != 0u";
                Line(instruction.Opcode == "SBitcmp1B32"
                    ? $"scc = {isSet};"
                    : $"scc = !({isSet});");
                return true;
            }

            return TryEmitScalarCompareCore(instruction.Opcode, "SCmp", left, right, out error);
        }

        private bool TryEmitScalarCompareK(
            Gen5ShaderInstruction instruction,
            uint destination,
            uint immediate,
            out string error) =>
            TryEmitScalarCompareCore(
                instruction.Opcode,
                "SCmpk",
                ScalarExpression(destination),
                FormatUInt(immediate),
                out error);

        private bool TryEmitScalarCompareCore(
            string opcode,
            string prefix,
            string left,
            string right,
            out string error)
        {
            error = string.Empty;
            var suffix = opcode[prefix.Length..];
            var signed = suffix.EndsWith("I32", StringComparison.Ordinal);
            var op = suffix[..^3] switch
            {
                "Eq" => "==",
                "Lg" => "!=",
                "Gt" => ">",
                "Ge" => ">=",
                "Lt" => "<",
                "Le" => "<=",
                _ => string.Empty,
            };
            if (op.Length == 0)
            {
                error = $"unsupported scalar compare {opcode}";
                return false;
            }

            Line(signed
                ? $"scc = as_type<int>({left}) {op} as_type<int>({right});"
                : $"scc = ({left}) {op} ({right});");
            return true;
        }

        // ---- 64-bit scalar ops over register pairs ----

        private bool TryEmitScalar64(
            Gen5ShaderInstruction instruction,
            uint destination,
            out string error)
        {
            error = string.Empty;

            // S_SETPC_B64 and S_SWAPPC_B64 are dynamic call/return operations.
            // The Gen5 translator has already linearized the reachable shader
            // body, so there is no host program counter to exchange here. Their
            // single pair operand must not be treated as a binary 64-bit op.
            if (instruction.Opcode is "SSetpcB64" or "SSwappcB64")
            {
                return true;
            }

            // S_BITSET0/1_B64 take a 32-bit bit index and update the 64-bit value
            // already held at the destination: they have no 64-bit source operand
            // and neither form writes SCC.
            if (instruction.Opcode is "SBitset0B64" or "SBitset1B64")
            {
                var bit = Temp("ulong", $"1ul << (ulong)({RawSource(instruction, 0)} & 63u)");
                StoreScalar64(
                    destination,
                    instruction.Opcode == "SBitset1B64"
                        ? $"{Scalar64Expression(destination)} | {bit}"
                        : $"{Scalar64Expression(destination)} & ~{bit}");
                return true;
            }

            if (TryEmitLaneMaskOp(instruction, destination))
            {
                return true;
            }

            var left = Temp("ulong", RawSource64(instruction, 0));
            if (instruction.Opcode.EndsWith("SaveexecB64", StringComparison.Ordinal))
            {
                var oldExec = Temp("ulong", Scalar64Expression(ExecLoRegister));
                var operation = instruction.Opcode[1..instruction.Opcode.IndexOf(
                    "Saveexec",
                    StringComparison.Ordinal)];
                var combined = operation switch
                {
                    "And" => $"({left} & {oldExec})",
                    "Or" => $"({left} | {oldExec})",
                    "Xor" => $"({left} ^ {oldExec})",
                    "Nand" => $"~({left} & {oldExec})",
                    "Nor" => $"~({left} | {oldExec})",
                    "Xnor" => $"~({left} ^ {oldExec})",
                    "Andn1" => $"(~{left} & {oldExec})",
                    "Andn2" => $"({left} & ~{oldExec})",
                    "Orn1" => $"(~{left} | {oldExec})",
                    "Orn2" => $"({left} | ~{oldExec})",
                    _ => string.Empty,
                };
                if (combined.Length == 0)
                {
                    error = $"unsupported scalar 64-bit saveexec opcode {instruction.Opcode}";
                    return false;
                }

                var mask = Temp("ulong", ClampExecToLanes64(combined));
                StoreScalar64(destination, oldExec);
                Line($"s[{ExecLoRegister}] = (uint){mask};");
                Line(IsNarrowWave64
                    ? $"s[{ExecHiRegister}] = 0u;"
                    : $"s[{ExecHiRegister}] = (uint)({mask} >> 32);");
                Line($"exec = (({mask} >> sharpemu_lane) & 1ul) != 0ul;");
                Line($"scc = {mask} != 0ul;");
                return true;
            }

            string value;
            var setsScc = true;
            switch (instruction.Opcode)
            {
                case "SMovB64":
                    value = left;
                    setsScc = false;
                    break;
                case "SCmovB64":
                    value = $"(scc ? {left} : {Scalar64Expression(destination)})";
                    setsScc = false;
                    break;
                case "SNotB64":
                    value = $"~{left}";
                    break;
                case "SWqmB64" when IsSingleLaneStage:
                    value = left;
                    break;
                case "SWqmB64":
                {
                    // Whole-quad mode: each 4-lane group becomes all-ones if any
                    // of its bits is set.
                    var quadAny = Temp(
                        "ulong",
                        $"({left} | ({left} >> 1) | ({left} >> 2) | ({left} >> 3)) & 0x1111111111111111ul");
                    value = $"({quadAny} * 0xFul)";
                    break;
                }
                case "SLshlB64" or "SLshrB64" or "SAshrI64":
                {
                    var shift = Temp("uint", $"({RawSource(instruction, 1)}) & 63u");
                    value = instruction.Opcode switch
                    {
                        "SLshlB64" => $"({left} << {shift})",
                        "SLshrB64" => $"({left} >> {shift})",
                        _ => $"(ulong)(as_type<long>({left}) >> {shift})",
                    };
                    break;
                }
                case "SBfmB64":
                {
                    var width = Temp("ulong", $"(ulong)(({RawSource(instruction, 0)}) & 63u)");
                    var offset = Temp("ulong", $"(ulong)(({RawSource(instruction, 1)}) & 63u)");
                    value = $"((((1ul << {width}) - 1ul)) << {offset})";
                    // S_BFM_B64 has no SCC result (ISA 12.1), like S_BFM_B32.
                    setsScc = false;
                    break;
                }
                case "SBfeU64" or "SBfeI64":
                {
                    var control = Temp("uint", RawSource(instruction, 1));
                    var offset = Temp("uint", $"{control} & 63u");
                    var width = Temp("uint", $"min(({control} >> 16) & 0x7Fu, 64u - {offset})");
                    var mask = Temp(
                        "ulong",
                        $"{width} >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << {width}) - 1ul)");
                    var extracted = Temp("ulong", $"({left} >> {offset}) & {mask}");
                    if (instruction.Opcode == "SBfeI64")
                    {
                        var signBit = Temp(
                            "ulong",
                            $"{width} == 0u ? 0ul : (1ul << ({width} - 1u))");
                        extracted = Temp(
                            "ulong",
                            $"{width} == 0u ? 0ul : (({extracted} ^ {signBit}) - {signBit})");
                    }

                    value = extracted;
                    break;
                }
                default:
                {
                    if (instruction.Sources.Count < 2)
                    {
                        error = "missing scalar 64-bit source";
                        return false;
                    }

                    var right = Temp("ulong", RawSource64(instruction, 1));
                    value = instruction.Opcode switch
                    {
                        "SAndB64" => $"({left} & {right})",
                        "SOrB64" => $"({left} | {right})",
                        "SXorB64" => $"({left} ^ {right})",
                        "SNandB64" => $"~({left} & {right})",
                        "SNorB64" => $"~({left} | {right})",
                        "SXnorB64" => $"~({left} ^ {right})",
                        "SAndn1B64" => $"(~{left} & {right})",
                        "SAndn2B64" => $"({left} & ~{right})",
                        "SOrn1B64" => $"(~{left} | {right})",
                        "SOrn2B64" => $"({left} | ~{right})",
                        "SCselectB64" => $"(scc ? {left} : {right})",
                        _ => string.Empty,
                    };
                    if (value.Length == 0)
                    {
                        error = $"unsupported scalar 64-bit opcode {instruction.Opcode}";
                        return false;
                    }

                    setsScc = instruction.Opcode != "SCselectB64";
                    break;
                }
            }

            var stored = Temp("ulong", value);
            StoreScalar64(destination, stored);
            if (setsScc)
            {
                Line($"scc = {stored} != 0ul;");
            }

            return true;
        }

        // ---- operand helpers ----

        private uint DestinationVector(Gen5ShaderInstruction instruction)
        {
            var destination = instruction.Destinations[0];
            return destination.Kind == Gen5OperandKind.VectorRegister
                ? destination.Value
                : throw new NotSupportedException(
                    $"vector destination expected in {instruction.Opcode}");
        }

        /// <summary>
        /// Raw 32-bit source with DPP/DPP8 lane remapping on src0 and SDWA
        /// byte/word selection + integer modifiers, mirroring GetRawSource.
        /// </summary>
        private string RawSource(
            Gen5ShaderInstruction instruction,
            int sourceIndex,
            bool applySdwaIntegerModifiers = true)
        {
            var value = SourceExpression(instruction.Sources[sourceIndex], instruction);
            if (sourceIndex == 0 && instruction.Control is Gen5DppControl dpp)
            {
                value = ApplyDppSource(dpp, value);
            }
            else if (sourceIndex == 0 && instruction.Control is Gen5Dpp8Control dpp8)
            {
                value = ApplyDpp8Source(dpp8, value);
            }

            if (instruction.Control is Gen5SdwaControl sdwa)
            {
                var selector = sourceIndex switch
                {
                    0 => sdwa.Source0Select,
                    1 => sdwa.Source1Select,
                    _ => 6u,
                };
                value = selector switch
                {
                    0 => $"(({value}) & 0xFFu)",
                    1 => $"((({value}) >> 8) & 0xFFu)",
                    2 => $"((({value}) >> 16) & 0xFFu)",
                    3 => $"((({value}) >> 24) & 0xFFu)",
                    4 => $"(({value}) & 0xFFFFu)",
                    5 => $"((({value}) >> 16) & 0xFFFFu)",
                    _ => value,
                };
                var signExtend = sourceIndex switch
                {
                    0 => sdwa.Source0SignExtend,
                    1 => sdwa.Source1SignExtend,
                    _ => false,
                };
                if (signExtend && selector != 6)
                {
                    var width = selector <= 3 ? 8u : 16u;
                    value = $"(uint)extract_bits(as_type<int>({value}), 0u, {width}u)";
                }

                if (applySdwaIntegerModifiers)
                {
                    // SDWA ABS/NEG are floating-point sign-bit modifiers even on
                    // a bit-move opcode: ABS clears the sign bit, NEG flips it.
                    // Two's-complement negating the raw bits instead turns 1.0
                    // into -4.0 and -3.0 into 1.5, which silently skews every
                    // pass that y-flips its clip position with an SDWA-negated
                    // V_MOV_B32 - the whole of UE's DrawRectangle.
                    var signBit = selector switch
                    {
                        <= 3 => 0x80u,
                        4 or 5 => 0x8000u,
                        _ => 0x80000000u,
                    };

                    if ((sdwa.AbsoluteMask & (1u << sourceIndex)) != 0)
                    {
                        value = $"(({value}) & {FormatUInt(~signBit)})";
                    }

                    if ((sdwa.NegateMask & (1u << sourceIndex)) != 0)
                    {
                        value = $"(({value}) ^ {FormatUInt(signBit)})";
                    }
                }
            }

            return value;
        }

        private string SignModifiedSource(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var value = RawSource(instruction, sourceIndex);
            // RawSource already applied an SDWA abs/neg to the selected byte or word.
            if (instruction.Control is not (Gen5Vop3Control or Gen5DppControl))
            {
                return value;
            }

            var (absoluteMask, negateMask) = SourceModifierMasks(instruction);
            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                value = $"(({value}) & 0x7FFFFFFFu)";
            }

            return (negateMask & (1u << sourceIndex)) != 0
                ? $"(({value}) ^ 0x80000000u)"
                : value;
        }

        /// <summary>The per-source abs/neg masks of whichever encoding carries them.</summary>
        private static (uint Absolute, uint Negate) SourceModifierMasks(Gen5ShaderInstruction instruction) =>
            instruction.Control switch
            {
                Gen5Vop3Control control => (control.AbsoluteMask, control.NegateMask),
                Gen5SdwaControl control => (control.AbsoluteMask, control.NegateMask),
                Gen5DppControl control => (control.AbsoluteMask, control.NegateMask),
                _ => (0u, 0u),
            };

        /// <summary>The output modifier and clamp of a VOP3 or SDWA float result. Output
        /// modifiers are ignored when MODE.IEEE is set or when FLOAT_MODE keeps output
        /// denormals of the result's precision (RDNA2 ISA 6.2.2): bit 5 for f32, bit 7 for
        /// f16 and f64.</summary>
        private (uint OutputModifier, bool Clamp) FloatOutputModifiers(Gen5ShaderInstruction instruction, bool halfOrDouble = false)
        {
            var (outputModifier, clamp) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.OutputModifier, control.Clamp),
                Gen5SdwaControl control => (control.OutputModifier, control.Clamp),
                _ => (0u, false),
            };
            return (OutputModifierApplies(halfOrDouble) ? outputModifier : 0u, clamp);
        }

        /// <summary>V_SQRT/RSQ/LOG/SIN/COS_F32: a NaN made from a number (sqrt of a negative, sin of an
        /// infinity) is 0xFFC00000, which RDNA2 ISA 12.8 gives in its functional examples
        /// (V_SQRT_F32(0xff800000) => 0xffc00000, likewise V_RSQ, V_LOG, V_SIN, V_COS); Metal's NaN
        /// is positive. A NaN operand passes through. Sin and cos reduce the operand to one cycle first.</summary>
        private string InvalidNanF32Result(Gen5ShaderInstruction instruction, string function, string? cycle = null)
        {
            var operand = Temp("float", F(instruction, 0));
            var argument = cycle is null
                ? $"sharpemu_flush_f32_denorm({operand})"
                : $"sharpemu_trig_cycle_f32({operand}, {cycle}) * {TauLiteral}";
            return FloatResult(instruction, $"sharpemu_invalid_nan_f32({function}({argument}), {operand})");
        }

        /// <summary>V_FRACT_F32: D = S0 + -floor(S0) (RDNA2 ISA 12.8), so an infinity gives inf + -inf, a
        /// NaN (0xFFC00000 as V_SQRT_F32 and the other invalid results); Metal's fract(inf) is 0.
        /// Metal's fract already stops short of 1.0 for a tiny negative operand.</summary>
        private string EmitFractF32(Gen5ShaderInstruction instruction)
        {
            var operand = Temp("float", F(instruction, 0));
            return FloatResult(instruction, $"(isinf({operand}) ? as_type<float>(0xFFC00000u) : fract({operand}))");
        }

        private bool OutputModifierApplies(bool halfOrDouble) =>
            !_request.IeeeMode && (_request.FloatMode & (halfOrDouble ? 0x80u : 0x20u)) == 0;

        /// <summary>Scales a float result by omod, which flushes a zero or denormal result to
        /// +0, then saturates it to [0, 1] for clamp (a NaN gives 0 under DX10_CLAMP and
        /// passes through without it).</summary>
        private string ApplyFloatOutputModifiers(Gen5ShaderInstruction instruction, string expression, bool half = false)
        {
            var (outputModifier, clamp) = FloatOutputModifiers(instruction, half);
            if (outputModifier != 0)
            {
                var scale = outputModifier switch { 1 => "2.0f", 2 => "4.0f", _ => "0.5f" };
                var scaled = Temp("float", $"({expression}) * {scale}");
                // For f16 the zero test runs after rounding to half, so a result that rounds
                // to an f16 denormal is flushed as well.
                var zero = half
                    ? $"(as_type<ushort>(half({scaled})) & 0x7C00u) == 0u"
                    : $"(as_type<uint>({scaled}) & 0x7FFFFFFFu) == 0u"; // a bit test: the relaxed math mode ignores the sign of a zero
                expression = Temp("float", $"({zero}) ? 0.0f : {scaled}");
            }

            return clamp ? ClampFloatToUnitInterval(expression) : expression;
        }

        private string ClampFloatToUnitInterval(string expression)
        {
            if (_request.Dx10Clamp)
            {
                return $"clamp({expression}, 0.0f, 1.0f)";
            }

            var value = Temp("float", expression);
            return $"(isnan({value}) ? {value} : clamp({value}, 0.0f, 1.0f))";
        }

        /// <summary>64-bit source: SGPR/VGPR pair, sign-extended inline, or zero-extended 32-bit.</summary>
        private string RawSource64(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            switch (operand.Kind)
            {
                case Gen5OperandKind.ScalarRegister:
                    return Scalar64Expression(operand.Value);
                case Gen5OperandKind.VectorRegister:
                    return $"((ulong)v[{operand.Value}] | ((ulong)v[{operand.Value + 1}] << 32))";
                case Gen5OperandKind.EncodedConstant when operand.Value is >= 193 and <= 208:
                {
                    // Inline negatives sign-extend: -1 denotes a full 64-bit mask.
                    var signed = -(long)(operand.Value - 192);
                    return $"0x{unchecked((ulong)signed):X}ul";
                }
                default:
                    return $"(ulong)({RawSource(instruction, sourceIndex)})";
            }
        }

        private string Scalar64Expression(uint register) => register switch
        {
            // VCC/EXEC read their architectural SGPR pairs like any other
            // register — programs park plain data there (see StoreScalar).
            _ when register + 1 < ScalarRegisterFileCount =>
                $"((ulong)s[{register}] | ((ulong)s[{register + 1}] << 32))",
            _ => "0ul",
        };

        private void StoreScalar64(uint register, string ulongValue)
        {
            StoreScalar(register, $"(uint)({ulongValue})");
            StoreScalar(register + 1, $"(uint)(({ulongValue}) >> 32)");
        }

        /// <summary>V_MIN/MAX/MIN3/MAX3/MED3_F32 through the prelude's bit-level helpers (ISA
        /// 12.7, 12.12): a NaN operand yields the other one (quieted when signalling in IEEE
        /// mode), -0 orders below +0, and V_MED3 on a NaN operand answers MIN3, so a clamp
        /// compiled to med3(hi, x, lo) flushes a NaN x to lo. Metal's fmin/fmax return either
        /// zero and compare f32 denormals as zero.</summary>
        private string MinMaxF32Result(Gen5ShaderInstruction instruction)
        {
            var ieee = _request.IeeeMode ? "true" : "false";
            string Bits(int index) => FlushInputDenormalF32(AsUInt(F(instruction, index)));
            if (_options.NativeFloatMinMax && !_request.IeeeMode && TryNativeMinMaxF32(instruction, Bits) is { } native)
            {
                var nativeResult = Temp("uint", native);
                return HasFloatOutputModifiers(instruction) ? FloatResult(instruction, AsFloat(nativeResult)) : nativeResult;
            }

            var bits = instruction.Opcode switch
            {
                "VMinF32" => $"sharpemu_min_f32({Bits(0)}, {Bits(1)}, {ieee})",
                "VMaxF32" => $"sharpemu_max_f32({Bits(0)}, {Bits(1)}, {ieee})",
                "VMin3F32" => $"sharpemu_min_f32(sharpemu_min_f32({Bits(0)}, {Bits(1)}, {ieee}), {Bits(2)}, {ieee})",
                "VMax3F32" => $"sharpemu_max_f32(sharpemu_max_f32({Bits(0)}, {Bits(1)}, {ieee}), {Bits(2)}, {ieee})",
                _ => $"sharpemu_med3_f32({Bits(0)}, {Bits(1)}, {Bits(2)}, {ieee})",
            };
            var result = Temp("uint", bits);
            return HasFloatOutputModifiers(instruction) ? FloatResult(instruction, AsFloat(result)) : result;
        }

        /// <summary>V_MIN/MAX/MIN3/MAX3/MED3_F32 outside IEEE mode without the bit-level helpers, or
        /// null where that is not provably the helpers' answer (see
        /// <see cref="Gen5MslCodegenOptions.NativeFloatMinMax"/>):
        ///   * against a non-NaN inline constant c, the answer is x exactly when x is a number that
        ///     orders above (max) or below (min) c, and c otherwise. The x that do are one or two
        ///     ranges of bit patterns, so the select tests the bits: no float compare, which a
        ///     denormal x could not survive;
        ///   * between two registers while FLOAT_MODE flushes f32 input denormals, fmax/fmin give
        ///     the number when one operand is NaN, and only a NaN src0 (answer src1) and two
        ///     operands that compare equal (the same bits, or +0 and -0, where the AND or OR of
        ///     the bits is the ordered answer) need the bits;
        ///   * V_MED3 with two constants lo &lt; hi, hi not a zero, is hi above hi, x between them,
        ///     and lo below lo or for a NaN x (MIN3 then answers lo).</summary>
        private string? TryNativeMinMaxF32(Gen5ShaderInstruction instruction, Func<int, string> bits)
        {
            var count = instruction.Opcode is "VMinF32" or "VMaxF32" ? 2 : 3;
            if (instruction.Sources.Count < count)
            {
                return null;
            }

            var constants = new uint?[count];
            for (var index = 0; index < count; index++)
            {
                constants[index] = TryConstantF32Operand(instruction, index, out var value) ? value : null;
            }

            var flushed = (_request.FloatMode & 0x10u) == 0;
            string? Pair(string? left, uint? leftConstant, string? right, uint? rightConstant, bool max)
            {
                if (rightConstant is { } c)
                {
                    return leftConstant is null ? MinMaxAgainstConstant(left!, c, max) : null;
                }

                if (leftConstant is { } d)
                {
                    return MinMaxAgainstConstant(right!, d, max);
                }

                if (!flushed)
                {
                    return null;
                }

                var a = Temp("uint", left!);
                var b = Temp("uint", right!);
                var fa = AsFloat(a);
                var fb = AsFloat(b);
                return $"(isnan({fa}) ? {b} : ({fa} == {fb} ? ({a} {(max ? "&" : "|")} {b}) : as_type<uint>({(max ? "fmax" : "fmin")}({fa}, {fb}))))";
            }

            string? Operand(int index) => constants[index] is null ? bits(index) : null;
            switch (instruction.Opcode)
            {
                case "VMinF32":
                case "VMaxF32":
                    return Pair(Operand(0), constants[0], Operand(1), constants[1], instruction.Opcode == "VMaxF32");
                case "VMin3F32":
                case "VMax3F32":
                {
                    var max = instruction.Opcode == "VMax3F32";
                    if (constants[0] is not null && constants[1] is not null)
                    {
                        return null;
                    }

                    var first = Pair(Operand(0), constants[0], Operand(1), constants[1], max);
                    return first is null ? null : Pair(Temp("uint", first), null, Operand(2), constants[2], max);
                }

                case "VMed3F32":
                {
                    var variable = -1;
                    var known = new List<uint>();
                    for (var index = 0; index < 3; index++)
                    {
                        if (constants[index] is { } value)
                        {
                            known.Add(value);
                        }
                        else
                        {
                            variable = index;
                        }
                    }

                    if (known.Count != 2)
                    {
                        return null;
                    }

                    var (lo, hi) = OrderKeyF32(known[0]) < OrderKeyF32(known[1]) ? (known[0], known[1]) : (known[1], known[0]);
                    if (OrderKeyF32(lo) == OrderKeyF32(hi) || (hi & 0x7FFFFFFFu) == 0)
                    {
                        return null;
                    }

                    var x = Temp("uint", bits(variable));
                    return $"({BitRangesTest(x, AboveF32(hi))} ? {FormatUInt(hi)} : ({BitRangesTest(x, AboveF32(lo))} ? {x} : {FormatUInt(lo)}))";
                }
            }

            return null;
        }

        /// <summary>Min or max of <paramref name="value"/> and the non-NaN constant
        /// <paramref name="constant"/>, as the ISA orders them.</summary>
        private string MinMaxAgainstConstant(string value, uint constant, bool max)
        {
            var x = Temp("uint", value);
            return $"({BitRangesTest(x, max ? AboveF32(constant) : BelowF32(constant))} ? {x} : {FormatUInt(constant)})";
        }

        /// <summary>An operand that is an inline constant or literal, as the f32 bits the
        /// instruction reads: its abs and neg applied, and a denormal flushed to a zero of its
        /// sign while FLOAT_MODE flushes f32 input denormals. False for a NaN, and for an SDWA
        /// operand, whose selects the bits would have to follow.</summary>
        private bool TryConstantF32Operand(Gen5ShaderInstruction instruction, int index, out uint bits)
        {
            bits = 0;
            if (instruction.Control is not (null or Gen5Vop3Control))
            {
                return false;
            }

            var operand = instruction.Sources[index];
            if (operand.Kind == Gen5OperandKind.LiteralConstant)
            {
                bits = operand.Value;
            }
            else if (operand.Kind != Gen5OperandKind.EncodedConstant || operand.Value is 251 or 252 or 253 ||
                     !Gen5InlineConstants.TryDecode(operand.Value, out bits))
            {
                return false;
            }

            var (absolute, negate) = SourceModifierMasks(instruction);
            if ((absolute & (1u << index)) != 0)
            {
                bits &= 0x7FFFFFFFu;
            }

            if ((negate & (1u << index)) != 0)
            {
                bits ^= 0x80000000u;
            }

            if ((_request.FloatMode & 0x10u) == 0 && (bits & 0x7F800000u) == 0)
            {
                bits &= 0x80000000u;
            }

            return (bits & 0x7FFFFFFFu) <= 0x7F800000u;
        }

        /// <summary>The ISA's f32 order as an unsigned key (sharpemu_order_f32): -0 below +0.</summary>
        private static uint OrderKeyF32(uint bits) => (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;

        /// <summary>The non-NaN bit patterns that order above the non-NaN <paramref name="c"/>:
        /// the positive numbers past it, or, for a negative c, every positive number and the
        /// negative ones of smaller magnitude.</summary>
        private static (uint Low, uint High)[] AboveF32(uint c) => (c & 0x80000000u) == 0
            ? c == 0x7F800000u ? [] : [(c + 1, 0x7F800000u)]
            : c == 0x80000000u ? [(0u, 0x7F800000u)] : [(0u, 0x7F800000u), (0x80000000u, c - 1)];

        /// <summary>The non-NaN bit patterns that order below the non-NaN <paramref name="c"/>.</summary>
        private static (uint Low, uint High)[] BelowF32(uint c) => (c & 0x80000000u) != 0
            ? c == 0xFF800000u ? [] : [(c + 1, 0xFF800000u)]
            : c == 0u ? [(0x80000000u, 0xFF800000u)] : [(0x80000000u, 0xFF800000u), (0u, c - 1)];

        private static string BitRangesTest(string value, (uint Low, uint High)[] ranges)
        {
            if (ranges.Length == 0)
            {
                return "false";
            }

            return string.Join(" || ", ranges.Select(range => range.Low == 0
                ? $"{value} <= {FormatUInt(range.High)}"
                : $"({value} - {FormatUInt(range.Low)}) <= {FormatUInt(range.High - range.Low)}"));
        }

        /// <summary>The f32 bits of an operand read by a bit-exact path, with a denormal flushed
        /// to a zero of its sign when FLOAT_MODE flushes f32 input denormals (bit 4 clear), as
        /// the float paths get from Metal.</summary>
        /// <summary>The f32 bits of a result of a bit-exact path, with a denormal flushed to a zero of
        /// its sign when FLOAT_MODE flushes f32 output denormals (bit 5 clear; RDNA2 ISA 6.4).</summary>
        private string FlushOutputDenormalF32(string bits)
        {
            var value = Temp("uint", bits);
            return (_request.FloatMode & 0x20u) != 0
                ? value
                : Temp("uint", $"(({value} & 0x7F800000u) == 0u) ? ({value} & 0x80000000u) : {value}");
        }

        private string FlushInputDenormalF32(string bits)
        {
            var value = Temp("uint", bits);
            return (_request.FloatMode & 0x10u) != 0
                ? value
                : Temp("uint", $"(({value} & 0x7F800000u) == 0u) ? ({value} & 0x80000000u) : {value}");
        }

        private string F(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var expression = AsFloat(
                RawSource(instruction, sourceIndex, applySdwaIntegerModifiers: false));
            var (absoluteMask, negateMask) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.AbsoluteMask, control.NegateMask),
                Gen5SdwaControl control => (control.AbsoluteMask, control.NegateMask),
                Gen5DppControl control => (control.AbsoluteMask, control.NegateMask),
                _ => (0u, 0u),
            };
            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                expression = $"fabs({expression})";
            }

            if ((negateMask & (1u << sourceIndex)) != 0)
            {
                expression = $"(-{expression})";
            }

            return expression;
        }

        /// <summary>The 16-bit value an inline constant gives a 16-bit float operand. The integer
        /// constants 128..208 are their two's-complement bit patterns, as they are for f32
        /// operands (`v_add_f16 v0, 1, v1` adds the denormal 0x0001; `1.0` is constant 242), and
        /// 240..248 are the f16 encodings of 0.5, -0.5, 1, -1, 2, -2, 4, -4 and 1/(2*pi),
        /// whose f16 form 0x3118 is not the f32 1/(2*pi) rounded.</summary>
        private static bool TryHalfInlineConstant(uint encoded, out uint bits)
        {
            bits = encoded switch
            {
                >= 128 and <= 192 => encoded - 128,
                >= 193 and <= 208 => (0x10000u - (encoded - 192)) & 0xFFFFu,
                240 => 0x3800u,
                241 => 0xB800u,
                242 => 0x3C00u,
                243 => 0xBC00u,
                244 => 0x4000u,
                245 => 0xC000u,
                246 => 0x4400u,
                247 => 0xC400u,
                248 => 0x3118u,
                _ => uint.MaxValue,
            };
            return bits != uint.MaxValue;
        }

        /// <summary>Reads the selected 16-bit half as a widened float.</summary>
        private string F16(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            string expression;
            if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                TryHalfInlineConstant(operand.Value, out var halfBits))
            {
                expression = $"(float)as_type<half>((ushort){FormatUInt(halfBits)})";
            }
            else if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                Gen5InlineConstants.TryDecode(operand.Value, out var inline))
            {
                expression = AsFloat(FormatUInt(inline));
            }
            else
            {
                var raw = RawSource(
                    instruction,
                    sourceIndex,
                    applySdwaIntegerModifiers: false);
                var shift = instruction.Control is Gen5Vop3Control control &&
                    (control.OperandSelect & (1u << sourceIndex)) != 0
                        ? 16
                        : 0;
                expression =
                    $"(float)as_type<half>((ushort)((({raw}) >> {shift}) & 0xFFFFu))";
            }

            var (absoluteMask, negateMask) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.AbsoluteMask, control.NegateMask),
                Gen5SdwaControl control => (control.AbsoluteMask, control.NegateMask),
                Gen5DppControl control => (control.AbsoluteMask, control.NegateMask),
                _ => (0u, 0u),
            };
            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                expression = $"fabs({expression})";
            }

            if ((negateMask & (1u << sourceIndex)) != 0)
            {
                expression = $"(-{expression})";
            }

            return expression;
        }

        private string Float16Logarithm(Gen5ShaderInstruction instruction, uint destination)
        {
            var source = Temp("float", F16(instruction, 0));
            return Float16Result(instruction, destination,
                $"({source} == 0.0f ? as_type<float>(0xFF800000u) : " +
                $"({source} > 0.0f ? (isinf({source}) ? {source} : log2({source})) : as_type<float>(0xFFC00000u)))");
        }

        private string Float16Reciprocal(Gen5ShaderInstruction instruction, uint destination)
        {
            var source = Temp("float", F16(instruction, 0));
            var bits = Temp("uint", $"as_type<uint>({source})");
            var magnitude = Temp("uint", $"({bits} & 0x7FFFFFFFu)");
            return Float16Result(instruction, destination,
                $"({magnitude} == 0u ? as_type<float>(({bits} & 0x80000000u) | 0x7F800000u) : " +
                $"({magnitude} == 0x7F800000u ? as_type<float>({bits} & 0x80000000u) : (1.0f / {source})))");
        }

        private string Float16SquareRoot(Gen5ShaderInstruction instruction, uint destination)
        {
            var source = Temp("float", F16(instruction, 0));
            return Float16Result(instruction, destination,
                $"({source} == 0.0f ? {source} : ({source} > 0.0f ? " +
                $"(isinf({source}) ? {source} : sqrt({source})) : as_type<float>(0xFFC00000u)))");
        }

        private string ReadMinimumHalfBits(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            string bits;
            if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                TryHalfInlineConstant(operand.Value, out var halfBits))
            {
                bits = FormatUInt(halfBits);
            }
            else if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                Gen5InlineConstants.TryDecode(operand.Value, out var inline))
            {
                bits = FormatUInt(BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(inline)));
            }
            else
            {
                var raw = RawSource(instruction, sourceIndex, applySdwaIntegerModifiers: false);
                var shift = instruction.Control is Gen5Vop3Control selection &&
                    (selection.OperandSelect & (1u << sourceIndex)) != 0 ? 16 : 0;
                bits = $"((({raw}) >> {shift}) & 0xFFFFu)";
            }
            var (absolute, negate) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.AbsoluteMask, control.NegateMask),
                Gen5SdwaControl control => (control.AbsoluteMask, control.NegateMask),
                Gen5DppControl control => (control.AbsoluteMask, control.NegateMask),
                _ => (0u, 0u),
            };
            if ((absolute & (1u << sourceIndex)) != 0) bits = $"({bits} & 0x7FFFu)";
            if ((negate & (1u << sourceIndex)) != 0) bits = $"({bits} ^ 0x8000u)";
            return Temp("uint", bits);
        }

        /// <summary>The f16 forms of <see cref="MinMaxF32Result"/>, on the f16 bits, into the
        /// op_sel-selected destination half.</summary>
        private string MinMaxF16Result(Gen5ShaderInstruction instruction, uint destination)
        {
            var ieee = _request.IeeeMode ? "true" : "false";
            string Bits(int index) => ReadMinimumHalfBits(instruction, index);
            var bits = Temp("uint", instruction.Opcode switch
            {
                "VMinF16" => $"sharpemu_min_f16({Bits(0)}, {Bits(1)}, {ieee})",
                "VMaxF16" => $"sharpemu_max_f16({Bits(0)}, {Bits(1)}, {ieee})",
                "VMin3F16" => $"sharpemu_min_f16(sharpemu_min_f16({Bits(0)}, {Bits(1)}, {ieee}), {Bits(2)}, {ieee})",
                "VMax3F16" => $"sharpemu_max_f16(sharpemu_max_f16({Bits(0)}, {Bits(1)}, {ieee}), {Bits(2)}, {ieee})",
                _ => $"sharpemu_med3_f16({Bits(0)}, {Bits(1)}, {Bits(2)}, {ieee})",
            });
            var control = instruction.Control as Gen5Vop3Control;
            var (modifier, clamp) = FloatOutputModifiers(instruction, halfOrDouble: true);
            if (modifier != 0)
            {
                // omod flushes a zero or denormal result to +0.
                var scale = modifier == 1 ? "2.0f" : modifier == 2 ? "4.0f" : "0.5f";
                var scaled = Temp("uint", $"(uint)as_type<ushort>(half((float)as_type<half>((ushort){bits}) * {scale}))");
                bits = Temp("uint", $"(({bits} & 0x7FFFu) > 0x7C00u ? {bits} : ({scaled} & 0x7C00u) == 0u ? 0u : {scaled})");
            }
            if (clamp)
            {
                // A NaN clamps to 0 under DX10_CLAMP and passes through without it.
                var nan = _request.Dx10Clamp ? "0u" : bits;
                bits = Temp("uint", $"(({bits} & 0x7FFFu) > 0x7C00u ? {nan} : ({bits} & 0x8000u) != 0u ? 0u : min({bits}, 0x3C00u))");
            }
            if (instruction.Control is Gen5SdwaControl)
            {
                return SixteenBitSdwaResult(bits);
            }

            return ((control?.OperandSelect ?? 0) & 8) != 0
                ? $"((v[{destination}] & 0x0000FFFFu) | ({bits} << 16))"
                : $"((v[{destination}] & 0xFFFF0000u) | {bits})";
        }

        /// <summary>
        /// Rounds to f16 and preserves the unselected VGPR half. An SDWA word picks the field and what
        /// becomes of the rest of the register itself.
        /// </summary>
        private string Float16Result(
            Gen5ShaderInstruction instruction,
            uint destination,
            string expression)
        {
            var control = instruction.Control as Gen5Vop3Control;
            expression = ApplyFloatOutputModifiers(instruction, expression, half: true);

            var packed = $"(uint)as_type<ushort>(half({expression}))";
            if (instruction.Control is Gen5SdwaControl)
            {
                return SixteenBitSdwaResult($"({packed})");
            }

            return ((control?.OperandSelect ?? 0) & 8) != 0
                ? $"((v[{destination}] & 0x0000FFFFu) | (({packed}) << 16))"
                : $"((v[{destination}] & 0xFFFF0000u) | ({packed}))";
        }

        // Set while an SDWA instruction's 16-bit result is being built: the result is the
        // bare zero-extended half, and ApplySdwaDestination places it.
        private bool _sixteenBitSdwaResult;

        private string SixteenBitSdwaResult(string half)
        {
            _sixteenBitSdwaResult = true;
            return half;
        }

        /// <summary>
        /// Wraps a float expression with VOP3/SDWA output modifiers and clamp,
        /// then bitcasts back to the register file's uint domain.
        /// </summary>
        private string FloatResult(Gen5ShaderInstruction instruction, string expression) =>
            AsUInt($"({ApplyFloatOutputModifiers(instruction, expression)})");

        /// <summary>The lane's bit of a mask operand (VCC/EXEC/SGPR mask).</summary>
        private string MaskBitExpression(Gen5Operand operand) => operand switch
        {
            { Kind: Gen5OperandKind.ScalarRegister, Value: VccLoRegister } => "vcc",
            { Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister } => "exec",
            { Kind: Gen5OperandKind.ScalarRegister } scalar =>
                IsWave64
                    ? $"({LaneBitOfPair(scalar.Value, LaneBit($"s[{scalar.Value}]", $"s[{scalar.Value + 1}]"))})"
                    : $"(((s[{scalar.Value}] >> sharpemu_lane) & 1u) != 0u)",
            _ => throw new NotSupportedException("mask operand must be a scalar register"),
        };
    }
}
