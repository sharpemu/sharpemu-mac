// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

// The clip position of a vertex program that computes it from the vertex index alone.
//
// Post-processing, composites and the final present draw one full-screen triangle or
// quad whose corners the vertex program makes out of the vertex index: x = (id & 1) * 4 - 1
// and the like, with no vertex buffer and no constant behind them. Such a draw overwrites
// every pixel of its target, and a host that can see that before the render pass begins
// can skip loading the target into tile memory. This works the position out ahead of time
// for the first few vertex indices by running the program over what is known at entry:
// the vertex index, VCC and EXEC. Everything else — user data, memory, the instance index,
// every other register — is unknown, and anything computed from an unknown is unknown.
//
// Values are tracked bit by bit, because the EXEC a primitive shader writes is a shift of
// all-ones by a count read from a system register, and the only bit a single-lane vertex
// cares about, bit 0, is one whatever the count. Arithmetic needs whole known operands.
// The program has to be straight-line: a branch, an unknown EXEC at the position export,
// or an auxiliary position export (clip and cull distances, layer, viewport) gives up.
// So does anything whose result this does not model, unless it only writes registers
// that then become unknown.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        /// <summary>The vertex indices <see cref="Gen5MslShader.ConstantPositions"/> covers.</summary>
        internal const int ConstantPositionVertices = Gen5MslShader.ConstantPositionVertexCount;

        private readonly record struct KnownBits(uint Known, uint Value)
        {
            public static readonly KnownBits Unknown = new(0, 0);

            public static KnownBits Of(uint value) => new(uint.MaxValue, value);

            public bool IsConstant => Known == uint.MaxValue;

            public bool? Bit0 => (Known & 1u) == 0 ? null : (Value & 1u) != 0;
        }

        /// <summary>The host clip position of vertex indices 0..5, four floats each, or
        /// null when it depends on anything but the index.</summary>
        internal static float[]? EvaluateConstantPositions(
            IReadOnlyList<Gen5ShaderInstruction> instructions,
            ShaderClipSpaceTransform clip)
        {
            if (clip.UserClipPlanes is { Length: > 0 })
            {
                // The program clips planes these positions do not describe.
                return null;
            }

            var positions = new float[ConstantPositionVertices * 4];
            for (uint vertex = 0; vertex < ConstantPositionVertices; vertex++)
            {
                if (!TryEvaluatePosition(instructions, vertex, out var position))
                {
                    return null;
                }

                // HostPlaneDistances uses the guest position, before the host clip-space
                // transform. A known nonnegative distance at every vertex cannot clip
                // any part of a primitive. Keep user/auxiliary clip planes conservative.
                var guestZ = position[2];
                var guestW = position[3];
                if (((clip.DepthPlanes & ShaderDepthPlanes.NearAtZero) != 0 && !(guestZ >= 0f)) ||
                    ((clip.DepthPlanes & ShaderDepthPlanes.NearAtNegativeW) != 0 && !(guestZ + guestW >= 0f)) ||
                    ((clip.DepthPlanes & ShaderDepthPlanes.Far) != 0 && !(guestW - guestZ >= 0f)))
                {
                    return null;
                }

                // HostClipSpaceStatements, in the same order and precision.
                var x = position[0];
                var y = position[1];
                var z = position[2];
                var w = position[3];
                if (clip.WReciprocal)
                {
                    w = 1.0f / w;
                }

                if (clip.XyPreDivided)
                {
                    x *= w;
                    y *= w;
                }

                if (clip.ZPreDivided)
                {
                    z *= w;
                }

                if (clip.Enabled)
                {
                    x = ((x * clip.ScaleX) + (clip.OffsetX * w)) / clip.HalfExtentX - w;
                    y = ((y * clip.ScaleY) + (clip.OffsetY * w)) / clip.HalfExtentY - w;
                }

                if (clip.PositionXMirror)
                {
                    x = -x;
                }

                if (clip.PositionYDown)
                {
                    y = -y;
                }

                if (clip.DepthRemap)
                {
                    z = (z * clip.DepthScale) + (w * clip.DepthOffset);
                }
                else if (clip.DepthNegativeOneToOne)
                {
                    z = (z + w) * 0.5f;
                }

                positions[(vertex * 4) + 0] = x;
                positions[(vertex * 4) + 1] = y;
                positions[(vertex * 4) + 2] = z;
                positions[(vertex * 4) + 3] = w;
            }

            return positions;
        }

        private static bool TryEvaluatePosition(
            IReadOnlyList<Gen5ShaderInstruction> instructions,
            uint vertex,
            out float[] position)
        {
            position = [];
            var scalars = new KnownBits[ScalarRegisterFileCount];
            var vectors = new KnownBits[VectorRegisterFileCount];
            scalars[VccLoRegister] = KnownBits.Of(0);
            scalars[VccHiRegister] = KnownBits.Of(0);
            // A single-lane stage's entry ballot is its own lane, bit 0.
            scalars[ExecLoRegister] = new KnownBits(1u, 1u);
            scalars[ExecHiRegister] = KnownBits.Of(0);
            vectors[5] = KnownBits.Of(vertex);
            var exported = new KnownBits[] { KnownBits.Of(0), KnownBits.Of(0), KnownBits.Of(0), KnownBits.Of(0) };

            foreach (var instruction in instructions)
            {
                var opcode = instruction.Opcode;
                if (opcode == "SEndpgm")
                {
                    break;
                }

                if (IsBranch(opcode) || IsIndirectControlFlow(opcode) ||
                    opcode.Contains("Movrel", StringComparison.Ordinal) ||
                    opcode.Contains("exec", StringComparison.OrdinalIgnoreCase) ||
                    opcode.Contains("Cmpx", StringComparison.Ordinal) ||
                    opcode.Contains("Cmpsx", StringComparison.Ordinal))
                {
                    return false;
                }

                if (instruction.Control is Gen5ExportControl export)
                {
                    if (export.Target is >= 13 and < 16)
                    {
                        return false;
                    }

                    if (export.Target != 12)
                    {
                        continue;
                    }

                    if (export.Compressed)
                    {
                        return false;
                    }

                    var exec = scalars[ExecLoRegister].Bit0;
                    if (exec == false)
                    {
                        continue;
                    }

                    for (var component = 0; component < 4; component++)
                    {
                        exported[component] = exec is null
                            ? KnownBits.Unknown
                            : (export.EnableMask & (1u << component)) == 0
                                ? KnownBits.Of(component == 3 ? 0x3F800000u : 0u)
                                : ReadVector(vectors, instruction.Sources[component]);
                    }

                    continue;
                }

                if (instruction.Encoding is Gen5ShaderEncoding.Sop1 or Gen5ShaderEncoding.Sop2 or
                        Gen5ShaderEncoding.Sopk or Gen5ShaderEncoding.Sopc or Gen5ShaderEncoding.Sopp &&
                    TryEvaluateScalar(instruction, scalars, vectors, out var scalarHandled))
                {
                    if (!scalarHandled)
                    {
                        ForgetDestinations(instruction, scalars, vectors);
                    }

                    continue;
                }

                if (opcode.StartsWith('V') && instruction.Encoding is Gen5ShaderEncoding.Vop1 or
                        Gen5ShaderEncoding.Vop2 or Gen5ShaderEncoding.Vop3 &&
                    TryEvaluateVector(instruction, scalars, vectors, out var value))
                {
                    if (instruction.Destinations is not [{ Kind: Gen5OperandKind.VectorRegister, Value: var target }, ..] ||
                        target >= VectorRegisterFileCount)
                    {
                        return false;
                    }

                    vectors[target] = scalars[ExecLoRegister].Bit0 switch
                    {
                        true => value,
                        false => vectors[target],
                        null => KnownBits.Unknown,
                    };
                    continue;
                }

                ForgetDestinations(instruction, scalars, vectors);
            }

            if (!exported[0].IsConstant || !exported[1].IsConstant ||
                !exported[2].IsConstant || !exported[3].IsConstant)
            {
                return false;
            }

            position =
            [
                BitConverter.UInt32BitsToSingle(exported[0].Value),
                BitConverter.UInt32BitsToSingle(exported[1].Value),
                BitConverter.UInt32BitsToSingle(exported[2].Value),
                BitConverter.UInt32BitsToSingle(exported[3].Value),
            ];
            return true;
        }

        /// <summary>Makes every register the instruction could write unknown.</summary>
        private static void ForgetDestinations(
            Gen5ShaderInstruction instruction,
            KnownBits[] scalars,
            KnownBits[] vectors)
        {
            void ForgetScalar(uint first, uint count)
            {
                for (var register = first; register < first + count && register < ScalarRegisterFileCount; register++)
                {
                    scalars[register] = KnownBits.Unknown;
                }
            }

            void ForgetVector(uint first, uint count)
            {
                for (var register = first; register < first + count && register < VectorRegisterFileCount; register++)
                {
                    vectors[register] = KnownBits.Unknown;
                }
            }

            // Memory results land in a run of registers the destination list may name only
            // the first of; eight covers every load, with its TFE/LWE status dword.
            const uint MemoryRun = 8;
            var vectorOp = instruction.Opcode.StartsWith('V');
            var vectorRun = !vectorOp
                ? MemoryRun
                : instruction.Opcode.Contains("sad", StringComparison.OrdinalIgnoreCase)
                    ? 4u
                    : instruction.Opcode.Contains("64", StringComparison.Ordinal) ? 2u : 1u;
            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind == Gen5OperandKind.ScalarRegister)
                {
                    ForgetScalar(destination.Value, vectorOp ? 2u : Math.Max(instruction.DestinationWidth, 1u));
                }
                else if (destination.Kind == Gen5OperandKind.VectorRegister)
                {
                    ForgetVector(destination.Value, vectorRun);
                }
            }

            switch (instruction.Control)
            {
                case Gen5ScalarMemoryControl memory when instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister } first, ..]:
                    ForgetScalar(first.Value, Math.Max(memory.DestinationCount, 1u));
                    break;
                case Gen5BufferMemoryControl buffer:
                    ForgetVector(buffer.VectorData, MemoryRun);
                    break;
                case Gen5ImageControl image:
                    ForgetVector(image.VectorData, MemoryRun);
                    break;
                case Gen5RayIntersectControl ray:
                    ForgetVector(ray.VectorData, MemoryRun);
                    break;
                case Gen5GlobalMemoryControl global:
                    ForgetVector(global.DestinationVectorRegister, MemoryRun);
                    break;
                case Gen5Vop3Control { ScalarDestination: { } pair }:
                    ForgetScalar(pair, 2);
                    break;
                case Gen5SdwaControl { ScalarDestination: { } pair }:
                    ForgetScalar(pair, 2);
                    break;
            }

            // A data-share read or anything else in memory encodings with no destination
            // listed: nothing about the vector file survives it.
            if (instruction.Destinations.Count == 0 &&
                instruction.Encoding is Gen5ShaderEncoding.Ds or Gen5ShaderEncoding.Flat or
                    Gen5ShaderEncoding.Mubuf or Gen5ShaderEncoding.Mtbuf or Gen5ShaderEncoding.Mimg &&
                instruction.Control is not (Gen5BufferMemoryControl or Gen5ImageControl or Gen5GlobalMemoryControl))
            {
                ForgetVector(0, VectorRegisterFileCount);
            }

            if (vectorOp)
            {
                // Comparisons and carries write VCC without naming it.
                ForgetScalar(VccLoRegister, 2);
            }
        }

        private static KnownBits ReadScalarOperand(KnownBits[] scalars, KnownBits[] vectors, Gen5Operand operand) =>
            operand.Kind switch
            {
                Gen5OperandKind.ScalarRegister => operand.Value < ScalarRegisterFileCount
                    ? scalars[operand.Value]
                    : KnownBits.Unknown,
                Gen5OperandKind.VectorRegister => operand.Value < VectorRegisterFileCount
                    ? vectors[operand.Value]
                    : KnownBits.Unknown,
                Gen5OperandKind.LiteralConstant => KnownBits.Of(operand.Value),
                // Integers and the plain float constants only; apertures, 1/(2pi) and the
                // condition sources are not worth trusting here.
                Gen5OperandKind.EncodedConstant when
                    operand.Value is (>= 128 and <= 208) or (>= 240 and <= 247) &&
                    Gen5InlineConstants.TryDecode(operand.Value, out var constant) => KnownBits.Of(constant),
                _ => KnownBits.Unknown,
            };

        private static KnownBits ReadVector(KnownBits[] vectors, Gen5Operand operand) =>
            operand.Kind == Gen5OperandKind.VectorRegister && operand.Value < VectorRegisterFileCount
                ? vectors[operand.Value]
                : KnownBits.Unknown;

        /// <summary>A 64-bit scalar operand as its two halves.</summary>
        private static (KnownBits Low, KnownBits High) ReadScalar64(KnownBits[] scalars, Gen5Operand operand) =>
            operand switch
            {
                { Kind: Gen5OperandKind.ScalarRegister } when operand.Value + 1 < ScalarRegisterFileCount =>
                    (scalars[operand.Value], scalars[operand.Value + 1]),
                { Kind: Gen5OperandKind.EncodedConstant, Value: >= 128 and <= 192 } =>
                    (KnownBits.Of(operand.Value - 128), KnownBits.Of(0)),
                { Kind: Gen5OperandKind.EncodedConstant, Value: >= 193 and <= 208 } =>
                    (KnownBits.Of(unchecked((uint)-(int)(operand.Value - 192))), KnownBits.Of(uint.MaxValue)),
                _ => (KnownBits.Unknown, KnownBits.Unknown),
            };

        /// <summary>Writes one SGPR the way StoreScalar does in a single-lane stage: EXEC
        /// keeps only the lane's bit and its high half reads zero.</summary>
        private static void StoreScalarValue(KnownBits[] scalars, uint register, KnownBits value)
        {
            if (register >= ScalarRegisterFileCount)
            {
                return;
            }

            scalars[register] = register switch
            {
                ExecLoRegister => And(value, KnownBits.Of(1u)),
                ExecHiRegister => KnownBits.Of(0),
                _ => value,
            };
        }

        private static bool TryEvaluateScalar(
            Gen5ShaderInstruction instruction,
            KnownBits[] scalars,
            KnownBits[] vectors,
            out bool handled)
        {
            handled = false;
            var opcode = instruction.Opcode;
            if (instruction.Encoding == Gen5ShaderEncoding.Sopp || instruction.Destinations.Count == 0)
            {
                // s_waitcnt, s_nop, s_sendmsg, s_ttracedata, s_inst_prefetch, comparisons:
                // nothing but SCC or the outside world changes.
                handled = instruction.Destinations.Count == 0;
                return true;
            }

            if (instruction.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister, Value: var destination }, ..])
            {
                return true;
            }

            KnownBits Source(int index) => index < instruction.Sources.Count
                ? ReadScalarOperand(scalars, vectors, instruction.Sources[index])
                : KnownBits.Unknown;

            KnownBits? result = opcode switch
            {
                "SMovB32" => Source(0),
                "SAndB32" => And(Source(0), Source(1)),
                "SOrB32" => Or(Source(0), Source(1)),
                "SXorB32" => Xor(Source(0), Source(1)),
                "SLshlB32" => ShiftLeft(Source(0), Source(1)),
                "SLshrB32" => ShiftRight(Source(0), Source(1)),
                "SBfeU32" => BitFieldExtract(Source(0), Source(1)),
                "SAddU32" or "SAddI32" => Arithmetic(Source(0), Source(1), static (a, b) => a + b),
                "SSubU32" or "SSubI32" => Arithmetic(Source(0), Source(1), static (a, b) => a - b),
                "SMulI32" => Arithmetic(Source(0), Source(1), static (a, b) => a * b),
                _ => null,
            };
            if (result is { } single)
            {
                StoreScalarValue(scalars, destination, single);
                handled = true;
                return true;
            }

            if (instruction.Sources.Count == 0)
            {
                return true;
            }

            var (low, high) = ReadScalar64(scalars, instruction.Sources[0]);
            (KnownBits Low, KnownBits High)? wide = opcode switch
            {
                "SMovB64" => (low, high),
                "SLshrB64" when instruction.Sources.Count > 1 =>
                    ShiftRight64(low, high, ReadScalarOperand(scalars, vectors, instruction.Sources[1])),
                _ => null,
            };
            if (wide is { } pair)
            {
                StoreScalarValue(scalars, destination, pair.Low);
                StoreScalarValue(scalars, destination + 1, pair.High);
                handled = true;
            }

            return true;
        }

        private static bool TryEvaluateVector(
            Gen5ShaderInstruction instruction,
            KnownBits[] scalars,
            KnownBits[] vectors,
            out KnownBits value)
        {
            value = KnownBits.Unknown;
            // Source and output modifiers, op_sel, DPP and SDWA are not modelled.
            if (instruction.Control is not (null or Gen5Vop3Control
                {
                    AbsoluteMask: 0, NegateMask: 0, OutputModifier: 0, Clamp: false, OperandSelect: 0, ScalarDestination: null,
                }))
            {
                return false;
            }

            KnownBits Source(int index) => index < instruction.Sources.Count
                ? ReadScalarOperand(scalars, vectors, instruction.Sources[index])
                : KnownBits.Unknown;

            static float AsFloat(uint bits) => BitConverter.UInt32BitsToSingle(bits);
            static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);

            KnownBits? result = instruction.Opcode switch
            {
                "VMovB32" => Source(0),
                "VAddF32" => Arithmetic(Source(0), Source(1), static (a, b) => Bits(AsFloat(a) + AsFloat(b))),
                "VSubF32" => Arithmetic(Source(0), Source(1), static (a, b) => Bits(AsFloat(a) - AsFloat(b))),
                "VSubrevF32" => Arithmetic(Source(0), Source(1), static (a, b) => Bits(AsFloat(b) - AsFloat(a))),
                "VMulF32" => Arithmetic(Source(0), Source(1), static (a, b) => Bits(AsFloat(a) * AsFloat(b))),
                "VFmaF32" or "VFmaAkF32" or "VFmaMkF32" => Fma(Source(0), Source(1), Source(2)),
                // V_MAD_F32 and V_MAC_F32 round the product before the add.
                "VMadF32" or "VMadAkF32" or "VMadMkF32" =>
                    Arithmetic(Arithmetic(Source(0), Source(1), static (a, b) => Bits(AsFloat(a) * AsFloat(b))), Source(2),
                        static (p, c) => Bits(AsFloat(p) + AsFloat(c))),
                "VFmacF32" when instruction.Destinations is [{ Kind: Gen5OperandKind.VectorRegister } target, ..] =>
                    Fma(Source(0), Source(1), ReadVector(vectors, target)),
                "VMacF32" when instruction.Destinations is [{ Kind: Gen5OperandKind.VectorRegister } target, ..] =>
                    Arithmetic(Arithmetic(Source(0), Source(1), static (a, b) => Bits(AsFloat(a) * AsFloat(b))),
                        ReadVector(vectors, target), static (p, c) => Bits(AsFloat(p) + AsFloat(c))),
                "VCvtF32I32" => Arithmetic(Source(0), KnownBits.Of(0), static (a, _) => Bits((float)unchecked((int)a))),
                "VCvtF32U32" => Arithmetic(Source(0), KnownBits.Of(0), static (a, _) => Bits((float)a)),
                "VAddU32" or "VAddI32" => Arithmetic(Source(0), Source(1), static (a, b) => a + b),
                "VSubU32" or "VSubI32" => Arithmetic(Source(0), Source(1), static (a, b) => a - b),
                "VSubrevU32" or "VSubrevI32" => Arithmetic(Source(0), Source(1), static (a, b) => b - a),
                "VMulLoU32" or "VMulLoI32" => Arithmetic(Source(0), Source(1), static (a, b) => a * b),
                "VMulU32U24" => Arithmetic(Source(0), Source(1), static (a, b) => (a & 0xFFFFFFu) * (b & 0xFFFFFFu)),
                "VMadU32U24" => Arithmetic(
                    Arithmetic(Source(0), Source(1), static (a, b) => (a & 0xFFFFFFu) * (b & 0xFFFFFFu)),
                    Source(2),
                    static (a, b) => a + b),
                "VLshlAddU32" => Arithmetic(
                    ShiftLeft(Source(0), Source(1)), Source(2), static (a, b) => a + b),
                "VAddLshlU32" => ShiftLeft(
                    Arithmetic(Source(0), Source(1), static (a, b) => a + b), Source(2)),
                "VLshlOrU32" => Or(ShiftLeft(Source(0), Source(1)), Source(2)),
                "VAndOrB32" => Or(And(Source(0), Source(1)), Source(2)),
                "VAndB32" => And(Source(0), Source(1)),
                "VOrB32" => Or(Source(0), Source(1)),
                "VXorB32" => Xor(Source(0), Source(1)),
                "VLshlrevB32" => ShiftLeft(Source(1), Source(0)),
                "VLshrrevB32" => ShiftRight(Source(1), Source(0)),
                "VAshrrevI32" => Arithmetic(Source(1), Source(0), static (a, b) => unchecked((uint)((int)a >> (int)(b & 31u)))),
                "VBfeU32" => Arithmetic(
                    Arithmetic(Source(0), Source(1), static (a, b) => a >> (int)(b & 31u)),
                    Source(2),
                    static (a, b) => a & ((1u << (int)(b & 31u)) - 1u)),
                "VCndmaskB32" => Select(
                    instruction.Sources.Count > 2
                        ? MaskBit(scalars, instruction.Sources[2])
                        : scalars[VccLoRegister].Bit0,
                    Source(1),
                    Source(0)),
                _ => null,
            };
            if (result is not { } known)
            {
                return false;
            }

            value = known;
            return true;
        }

        private static bool? MaskBit(KnownBits[] scalars, Gen5Operand operand) =>
            operand.Kind == Gen5OperandKind.ScalarRegister && operand.Value < ScalarRegisterFileCount
                ? scalars[operand.Value].Bit0
                : null;

        private static KnownBits Select(bool? condition, KnownBits whenTrue, KnownBits whenFalse) => condition switch
        {
            true => whenTrue,
            false => whenFalse,
            null => KnownBits.Unknown,
        };

        private static KnownBits Fma(KnownBits a, KnownBits b, KnownBits c) =>
            a.IsConstant && b.IsConstant && c.IsConstant
                ? KnownBits.Of(BitConverter.SingleToUInt32Bits(MathF.FusedMultiplyAdd(
                    BitConverter.UInt32BitsToSingle(a.Value),
                    BitConverter.UInt32BitsToSingle(b.Value),
                    BitConverter.UInt32BitsToSingle(c.Value))))
                : KnownBits.Unknown;

        private static KnownBits Arithmetic(KnownBits a, KnownBits b, Func<uint, uint, uint> operation) =>
            a.IsConstant && b.IsConstant ? KnownBits.Of(unchecked(operation(a.Value, b.Value))) : KnownBits.Unknown;

        private static KnownBits And(KnownBits a, KnownBits b)
        {
            var known = (a.Known & b.Known) | (a.Known & ~a.Value) | (b.Known & ~b.Value);
            return new KnownBits(known, a.Value & b.Value & known);
        }

        private static KnownBits Or(KnownBits a, KnownBits b)
        {
            var known = (a.Known & b.Known) | (a.Known & a.Value) | (b.Known & b.Value);
            return new KnownBits(known, (a.Value | b.Value) & known);
        }

        private static KnownBits Xor(KnownBits a, KnownBits b)
        {
            var known = a.Known & b.Known;
            return new KnownBits(known, (a.Value ^ b.Value) & known);
        }

        private static KnownBits ShiftLeft(KnownBits value, KnownBits amount) =>
            ShiftByEveryCount(value, amount, static (v, count) => new KnownBits(
                (v.Known << count) | ((1u << count) - 1u),
                v.Value << count));

        private static KnownBits ShiftRight(KnownBits value, KnownBits amount) =>
            ShiftByEveryCount(value, amount, static (v, count) => new KnownBits(
                (v.Known >> count) | ~(uint.MaxValue >> count),
                v.Value >> count));

        /// <summary>A 32-bit shift by a count masked to five bits: the bits every possible
        /// count agrees on are known, which for a known count is all of them.</summary>
        private static KnownBits ShiftByEveryCount(
            KnownBits value,
            KnownBits amount,
            Func<KnownBits, int, KnownBits> shift)
        {
            KnownBits? merged = null;
            for (var count = 0u; count < 32u; count++)
            {
                if ((count & amount.Known) != (amount.Value & amount.Known & 31u))
                {
                    continue;
                }

                var shifted = shift(value, (int)count);
                merged = merged is not { } previous
                    ? shifted
                    : Agree(previous, shifted);
            }

            return merged ?? KnownBits.Unknown;
        }

        private static KnownBits Agree(KnownBits a, KnownBits b)
        {
            var known = a.Known & b.Known & ~(a.Value ^ b.Value);
            return new KnownBits(known, a.Value & known);
        }

        /// <summary>S_LSHR_B64 by a count masked to six bits, merged over every count the
        /// known bits of the amount allow.</summary>
        private static (KnownBits Low, KnownBits High) ShiftRight64(KnownBits low, KnownBits high, KnownBits amount)
        {
            var known = ((ulong)high.Known << 32) | low.Known;
            var value = ((ulong)high.Value << 32) | low.Value;
            ulong? mergedKnown = null;
            ulong mergedValue = 0;
            for (var count = 0u; count < 64u; count++)
            {
                if ((count & amount.Known) != (amount.Value & amount.Known & 63u))
                {
                    continue;
                }

                var shiftedKnown = (known >> (int)count) | ~(ulong.MaxValue >> (int)count);
                var shiftedValue = value >> (int)count;
                if (mergedKnown is not { } previous)
                {
                    mergedKnown = shiftedKnown;
                    mergedValue = shiftedValue;
                    continue;
                }

                mergedKnown = previous & shiftedKnown & ~(mergedValue ^ shiftedValue);
                mergedValue &= mergedKnown.Value;
            }

            var finalKnown = mergedKnown ?? 0;
            mergedValue &= finalKnown;
            return (
                new KnownBits((uint)finalKnown, (uint)mergedValue),
                new KnownBits((uint)(finalKnown >> 32), (uint)(mergedValue >> 32)));
        }

        private static KnownBits BitFieldExtract(KnownBits value, KnownBits control)
        {
            if (!control.IsConstant)
            {
                return KnownBits.Unknown;
            }

            var offset = (int)(control.Value & 31u);
            var width = (int)Math.Min((control.Value >> 16) & 0x7Fu, 32u - (uint)offset);
            if (width == 0)
            {
                return KnownBits.Of(0);
            }

            var mask = width == 32 ? uint.MaxValue : (1u << width) - 1u;
            var shifted = ShiftRight(value, KnownBits.Of((uint)offset));
            return And(shifted, KnownBits.Of(mask));
        }
    }
}
