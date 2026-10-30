// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler;

public enum Gen5ShaderEncoding
{
    Sop1,
    Sop2,
    Sopc,
    Sopp,
    Sopk,
    Smrd,
    Smem,
    Mubuf,
    Mtbuf,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vintrp,
    Ds,
    Flat,
    Vop3p,
    Mimg,
    Exp,
}

public enum Gen5OperandKind
{
    ScalarRegister,
    VectorRegister,
    EncodedConstant,
    LiteralConstant,
}

public enum Gen5PixelOutputKind
{
    Float,
    Uint,
    Sint,
}

/// <summary>
/// Selects the logical shader component for each physical color component.
/// Each component uses two bits in <see cref="Packed"/>.
/// </summary>
public readonly record struct Gen5ColorComponentMapping
{
    public const byte IdentityPacked = 0xE4;
    private readonly byte _encoded;

    public Gen5ColorComponentMapping(byte packed)
    {
        _encoded = (byte)(packed ^ IdentityPacked);
    }

    public byte Packed => (byte)(_encoded ^ IdentityPacked);

    public static Gen5ColorComponentMapping Identity { get; } = new(IdentityPacked);

    public uint Map(uint physicalComponent) =>
        physicalComponent < 4
            ? (uint)(Packed >> checked((int)(physicalComponent * 2))) & 0x3u
            : physicalComponent;

    public uint ApplyMask(uint logicalMask)
    {
        var mappedMask = 0u;
        for (var physicalComponent = 0u; physicalComponent < 4; physicalComponent++)
        {
            mappedMask |= ((logicalMask >> checked((int)Map(physicalComponent))) & 1u)
                << checked((int)physicalComponent);
        }

        return mappedMask;
    }

    public Gen5ColorComponentMapping Then(Gen5ColorComponentMapping next)
    {
        var packed = 0u;
        for (var physicalComponent = 0u; physicalComponent < 4; physicalComponent++)
        {
            packed |= next.Map(Map(physicalComponent))
                << checked((int)(physicalComponent * 2));
        }

        return new Gen5ColorComponentMapping(checked((byte)packed));
    }

    public bool IsIdentity => Packed == IdentityPacked;

    public static bool TryResolveRenderTarget(
        uint componentSwap,
        uint componentCount,
        out Gen5ColorComponentMapping mapping)
    {
        var packed = (componentSwap, componentCount) switch
        {
            (0, >= 1 and <= 4) => IdentityPacked,
            (1, 1) => 0xE1,
            (1, 2) => 0x6C,
            (1, 3) => 0xB4,
            (1, 4) => 0xC6,
            (2, 1) => 0xC6,
            (2, 2) => 0xE1,
            (2, 3) => 0xC6,
            (2, 4) => 0x1B,
            (3, 1) => 0x27,
            (3, 2) => 0x63,
            (3, 3) => 0x87,
            (3, 4) => 0x93,
            _ => -1,
        };
        mapping = packed >= 0
            ? new Gen5ColorComponentMapping(checked((byte)packed))
            : default;
        return packed >= 0;
    }
}

// SPI_SHADER_COL_FORMAT (and SPI_SHADER_Z_FORMAT): how the export's VGPRs carry the target's
// channels. The 16-bit formats come from compressed (COMPR) exports, two channels a VGPR.
public enum Gen5ColorExportFormat
{
    Zero = 0,
    R32 = 1,
    GR32 = 2,
    AR32 = 3,
    Fp16Abgr = 4,
    Unorm16Abgr = 5,
    Snorm16Abgr = 6,
    Uint16Abgr = 7,
    Sint16Abgr = 8,
    Abgr32 = 9,
}

public readonly record struct Gen5PixelOutputBinding(
    uint GuestSlot,
    uint HostLocation,
    Gen5PixelOutputKind Kind,
    Gen5ColorComponentMapping ComponentMapping)
{
    public Gen5PixelOutputBinding(
        uint guestSlot,
        uint hostLocation,
        Gen5PixelOutputKind kind)
        : this(guestSlot, hostLocation, kind, Gen5ColorComponentMapping.Identity)
    {
    }

    private readonly uint? _exportTarget;

    // The fragment output index: 1 is the second source of dual-source blending (the MRT1
    // export), which shares host location 0 with the first and is not a render target.
    public uint BlendSourceIndex { get; init; }
    // The blend reads source alpha but the component mapping moves the shader's alpha out of
    // the host alpha channel: the program also writes that alpha to the dual-source output
    // (Location 0, Index 1) so the pipeline can blend with SRC1_ALPHA.
    public bool DualSourceAlpha { get; init; }

    // SPI_SHADER_COL_FORMAT of the export target; null when the registers name no format, and
    // then the export's own layout (COMPR or not) decides.
    public Gen5ColorExportFormat? ExportFormat { get; init; }

    // CB_COLOR_INFO.BLEND_CLAMP on a float target written without blending: the CB clamps
    // the exported colour to [0, 1], so the program saturates the output.
    public bool ClampToUnit { get; init; }

    // The EXP MRT target that feeds this slot. It differs from the slot when the pixel
    // program skips targets, because the hardware packs color exports into the slots
    // that CB_SHADER_MASK enables.
    public uint ExportTarget
    {
        get => _exportTarget ?? GuestSlot;
        init => _exportTarget = value;
    }
}

public enum Gen5TessellationRole : byte
{
    // The merged local/hull program: one guest wave64 per threadgroup of PatchesPerGroup patches.
    Hull,
    // The domain (export) program: a post-tessellation vertex function over triangle or quad patches.
    Domain,
}

// VGT_TF_PARAM.PARTITIONING, in the register's order.
public enum Gen5TessellationPartition : byte
{
    Integer,
    Pow2,
    FractionalOdd,
    FractionalEven,
}

// A tessellated draw on a host whose hull stage is a compute kernel. The hull program runs as a
// kernel whose threadgroup is one guest wave of PatchesPerGroup patches (ControlPoints lanes each,
// one per input and output control point): its local half reads the control points' vertices by the
// ids the host computed from the index buffer, and its hull half hands each patch to the domain stage
// through the guest's off-chip buffer, OffchipBlockBytes per threadgroup, and writes its tessellation
// factors, FactorBytesPerPatch per patch, to the guest's factor ring. The host's tessellator then
// subdivides every patch by those factors (Partition, triangles wound OutputClockwise, factors up to
// MaxFactor) and the domain program runs as its post-tessellation vertex function, at each point the
// tessellator makes. Partition, winding and MaxFactor are pipeline state the domain program carries.
public sealed record Gen5TessellationLaunch(
    Gen5TessellationRole Role,
    uint PatchesPerGroup,
    uint ControlPoints,
    uint OffchipBlockBytes,
    uint FactorBytesPerPatch,
    Gen5TessellationPartition Partition = Gen5TessellationPartition.Integer,
    bool OutputClockwise = false,
    uint MaxFactor = 16,
    // VGT_TF_PARAM.TYPE's quad domain (Ghost of Tsushima) rather than its triangle one: four edge
    // factors and two inside ones per patch, and the domain program's u and v span a square.
    bool Quad = false)
{
    // The hull kernel's launch dwords (patch count, first instance, patches of one instance) and one
    // instance's control points' vertex ids.
    public const int HullLaunchBufferIndex = 2;
    public const int HullVertexBufferIndex = 3;

    // The domain program's launch dwords (patches of one instance, the draw's first instance): a
    // vertex-buffer slot, which a domain program (it fetches no vertex attributes) leaves free.
    public const int DomainLaunchBufferIndex = 2;
}

public readonly record struct Gen5ComputeSystemRegisters(
    uint? WorkGroupXRegister,
    uint? WorkGroupYRegister,
    uint? WorkGroupZRegister,
    uint? ThreadGroupSizeRegister);

// The geometry-engine state one NGG primitive-shader (merged ES/GS) subgroup runs under.
// A primitive shader is one workgroup of waves that share LDS, so the emitter needs the
// subgroup's vertex/primitive budget to size its output and its workgroup.
//
// VerticesPerSubgroup/PrimitivesPerSubgroup come from VGT_GS_ONCHIP_CNTL
// (ES_VERTS_PER_SUBGRP / GS_PRIMS_PER_SUBGRP); when the title leaves them zero the
// emitter falls back to MaxOutputPerSubgroup (GE_MAX_OUTPUT_PER_SUBGROUP). The group
// sizes are GE_CNTL.VERT_GRP_SIZE / PRIM_GRP_SIZE, which bound what the geometry engine
// packs into one subgroup. OutputPrimitiveType is VGT_GS_OUT_PRIM_TYPE
// (0 point, 1 line strip, 2 triangle strip). LocalDataShareBytes is the LDS the stage
// allocates, decoded from the geometry stage's resource register.
public readonly record struct Gen5NggStageInfo(
    uint VerticesPerSubgroup,
    uint PrimitivesPerSubgroup,
    uint MaxOutputPerSubgroup,
    uint OutputPrimitiveType,
    uint PrimitiveAmplificationFactor,
    uint LocalDataShareBytes,
    uint VertexGroupSize,
    uint PrimitiveGroupSize)
{
    /// <summary>VGT_GS_MAX_VERT_OUT of a fused program, whose GS half runs after the ES
    /// half in the same workgroup; zero for a primitive shader without a GS half. A GS
    /// half reads its input vertices back from LDS through the GS vertex offsets, so the
    /// launch state has to assemble those from the draw's input topology, and every input
    /// primitive can emit up to this many vertices into the subgroup's output.</summary>
    public uint GeometryVerticesOut { get; init; }

    /// <summary>True when the stage runs as wave64 (VGT_SHADER_STAGES_EN.GS_W32_EN clear).
    /// A mesh program without a GS half translates such a wave as a narrow wave64: one
    /// 32-lane Metal simdgroup whose upper 32 guest lanes are architecturally inactive.</summary>
    public bool IsWave64 { get; init; }

    /// <summary>True in NGG passthrough mode (VGT_SHADER_STAGES_EN.PRIMGEN_PASSTHRU_EN): the
    /// geometry engine hands every lane its primitive already packed the way the primitive
    /// export takes it (three 10-bit vertex fields) in v0, instead of the GS vertex offsets,
    /// and the program exports that register as it is.</summary>
    public bool Passthrough { get; init; }

    /// <summary>VGT_GS_MAX_VERT_OUT of a merged ES/GS subgroup that amplifies a single input
    /// vertex (GS_EN, GE_NGG_SUBGRP_CNTL.PRIM_AMP_FACTOR above 1, ES_VERTS_PER_SUBGRP 1):
    /// how Prospero runs a mesh shader. Every point of a point-list draw launches one
    /// workgroup, whose first lane's ES vertex index is the point (the workgroup index),
    /// and the workgroup emits up to this many vertices and PRIM_AMP_FACTOR primitives, one
    /// per thread. Black Myth: Wukong's Nanite hardware rasterizer is one. Zero otherwise.</summary>
    public uint AmplifiedVerticesOut { get; init; }

    /// <summary>The most primitives one amplified subgroup emits
    /// (GE_NGG_SUBGRP_CNTL.PRIM_AMP_FACTOR).</summary>
    public uint AmplifiedPrimitivesOut => PrimitiveAmplificationFactor & 0x1FFu;
}

public readonly record struct Gen5Operand(Gen5OperandKind Kind, uint Value)
{
    public static Gen5Operand Scalar(uint index) =>
        new(Gen5OperandKind.ScalarRegister, index);

    public static Gen5Operand Vector(uint index) =>
        new(Gen5OperandKind.VectorRegister, index);

    public static Gen5Operand Source(uint encoded, uint? literal = null)
    {
        if (encoded >= 256)
        {
            return Vector(encoded - 256);
        }

        if (encoded is 249 or 255 && literal.HasValue)
        {
            return new(Gen5OperandKind.LiteralConstant, literal.Value);
        }

        // 108..123 are the temporary SGPRs (TTMP0..TTMP15).  They use the
        // same scalar register file as ordinary SGPRs; treating them as
        // encoded constants loses values in shader continuations that spill
        // through a temporary register.
        if (encoded <= 105 || encoded is 106 or 107 or (>= 108 and <= 123) or 124 or 126 or 127)
        {
            return Scalar(encoded);
        }

        return new(Gen5OperandKind.EncodedConstant, encoded);
    }

    public override string ToString() => Kind switch
    {
        Gen5OperandKind.ScalarRegister => $"s{Value}",
        Gen5OperandKind.VectorRegister => $"v{Value}",
        Gen5OperandKind.LiteralConstant => $"0x{Value:X8}",
        _ => $"src[{Value}]",
    };
}

public abstract record Gen5InstructionControl;

public sealed record Gen5ImageControl(
    uint Dmask,
    uint VectorAddress,
    IReadOnlyList<uint> AddressRegisters,
    uint VectorData,
    uint ScalarResource,
    uint ScalarSampler,
    uint Dimension,
    bool IsArray,
    bool Glc,
    bool Slc,
    bool A16,
    bool D16) : Gen5InstructionControl
{
    /// <summary>MIMG UNRM (word bit 12): the address is in texels rather than normalized.</summary>
    public bool Unorm { get; init; }

    /// <summary>MIMG DLC (word bit 7): a cache policy, no effect on the result.</summary>
    public bool Dlc { get; init; }

    /// <summary>MIMG R128 (word bit 15): the T# is four dwords rather than eight.</summary>
    public bool R128 { get; init; }

    /// <summary>MIMG TFE (word bit 16): a PRT fault-status dword follows the data.</summary>
    public bool Tfe { get; init; }

    /// <summary>MIMG LWE (word bit 17): an LOD-warning status dword follows the data.</summary>
    public bool Lwe { get; init; }

    public uint GetAddressRegister(int component) =>
        component < AddressRegisters.Count
            ? AddressRegisters[component]
            : VectorAddress + (uint)component;
}

public sealed record Gen5RayIntersectControl(
    uint VectorAddress,
    IReadOnlyList<uint> AddressRegisters,
    uint VectorData,
    uint ScalarResource,
    bool A16) : Gen5InstructionControl
{
    public const uint ResultDwords = 4;

    public uint GetAddressRegister(int component) =>
        component < AddressRegisters.Count
            ? AddressRegisters[component]
            : VectorAddress + (uint)component;
}

// Dlc (FLAT word 0 bit 12) only steers the GL1 cache policy. AddThreadId marks
// GLOBAL_LOAD/STORE_DWORD_ADDTID, whose address is SADDR + offset + lane * 4 with no VADDR.
public sealed record Gen5GlobalMemoryControl(
    uint DwordCount,
    uint VectorAddress,
    uint SourceVectorRegister,
    uint DestinationVectorRegister,
    uint ScalarAddress,
    int OffsetBytes,
    bool Glc,
    bool Slc,
    bool UsesFlatAddress = false,
    uint? DynamicOffsetRegister = null,
    bool SourceIsScalar = false,
    bool Dlc = false,
    bool AddThreadId = false) : Gen5InstructionControl;

// A typed access carries the unified format from the instruction; a formatted
// untyped access reads the descriptor format when it executes. Dlc only steers the GL1
// cache policy. Tfe (a load's texel-fault enable) writes a status dword to the VGPR after
// the data; Lds (MUBUF word 0 bit 16) returns a load's data to LDS at M0 instead of VGPRs.
public sealed record Gen5BufferMemoryControl(
    uint DwordCount,
    uint VectorAddress,
    uint VectorData,
    uint ScalarResource,
    int OffsetBytes,
    bool IndexEnabled,
    bool OffsetEnabled,
    bool Glc,
    bool Slc,
    bool Typed = false,
    uint TypedFormat = 0,
    bool Dlc = false,
    bool Tfe = false,
    bool Lds = false) : Gen5InstructionControl;

public sealed record Gen5ExportControl(
    uint Target,
    uint EnableMask,
    bool Compressed,
    bool Done,
    bool ValidMask) : Gen5InstructionControl;

public sealed record Gen5InterpolationControl(
    uint Attribute,
    uint Channel) : Gen5InstructionControl
{
    // V_INTERP_MOV's parameter: 0 is P10, 1 P20 and 2 P0 (the VINTRP vsrc field, VOP3 src1).
    public uint Parameter { get; init; }

    // The f16 forms' HIGH bit (VOP3 src0[8]): the attribute's upper 16 bits.
    public bool High { get; init; }

    // The VOP3 encoding's modifiers (abs/neg on the I/J and S2 operands, clamp, omod);
    // null for VINTRP.
    public Gen5Vop3Control? Modifiers { get; init; }
}

public sealed record Gen5Vop3Control(
    uint AbsoluteMask,
    uint NegateMask,
    uint OutputModifier,
    bool Clamp,
    uint OperandSelect,
    uint? ScalarDestination) : Gen5InstructionControl;

public sealed record Gen5SdwaControl(
    uint DestinationSelect,
    uint DestinationUnused,
    uint Source0Select,
    uint Source1Select,
    bool Source0SignExtend,
    bool Source1SignExtend,
    uint AbsoluteMask,
    uint NegateMask,
    uint OutputModifier,
    bool Clamp,
    uint? ScalarDestination) : Gen5InstructionControl;

// Packed (VOP3P) source and destination modifiers. Each mask holds one bit per
// source operand. OpSel/OpSelHi pick which 16-bit half of a source feeds the low
// and high result lanes respectively; NegLo/NegHi negate the value routed to each
// lane. Clamp saturates each output half to [0, 1].
public sealed record Gen5Vop3pControl(
    uint OpSelMask,
    uint OpSelHiMask,
    uint NegLoMask,
    uint NegHiMask,
    bool Clamp) : Gen5InstructionControl;

public sealed record Gen5DppControl(
    uint Control,
    bool FetchInactive,
    bool BoundControl,
    uint AbsoluteMask,
    uint NegateMask,
    uint BankMask,
    uint RowMask) : Gen5InstructionControl;

public sealed record Gen5Dpp8Control(
    uint LaneSelectors,
    bool FetchInactive) : Gen5InstructionControl;

// Glc is the SMEM GLC bit (gfx10 word 0 bit 16): the load misses the scalar cache and
// reads the device-coherent value. Dlc (word 0 bit 14) only steers the GL1 cache policy.
public sealed record Gen5ScalarMemoryControl(
    uint DestinationCount,
    int ImmediateOffsetBytes,
    uint? DynamicOffsetRegister,
    bool Glc = false,
    bool Dlc = false) : Gen5InstructionControl;

public sealed record Gen5DataShareControl(
    uint Offset0,
    uint Offset1,
    bool Gds) : Gen5InstructionControl
{
    // Single-address DS instructions encode one 16-bit byte offset across
    // OFFSET0 and OFFSET1. The paired read2/write2 forms instead interpret
    // them as two independent scaled 8-bit offsets.
    public uint SingleOffsetBytes => Offset0 | (Offset1 << 8);
}

public sealed record Gen5ShaderInstruction(
    uint Pc,
    Gen5ShaderEncoding Encoding,
    string Opcode,
    IReadOnlyList<uint> Words,
    IReadOnlyList<Gen5Operand> Sources,
    IReadOnlyList<Gen5Operand> Destinations,
    Gen5InstructionControl? Control)
{
    public ulong? AddressOffset { get; init; }

    public ulong ProgramOffset => AddressOffset ?? Pc;
    public uint DestinationWidth => Opcode is "SBcnt1I32B64" or "SFF1I32B64" or "SFlbitI32B64"
        ? 1u : Opcode.Contains("64", StringComparison.Ordinal) ? 2u : 1u;
}

public sealed record Gen5ShaderProgram(
    ulong Address,
    IReadOnlyList<Gen5ShaderInstruction> Instructions)
{
    public IReadOnlySet<uint> FunctionBufferAccesses { get; init; } = new HashSet<uint>();
    public bool IsFusedProgram { get; init; }

    private const uint PixelColorTargetCount = 8;
    private const int PixelColorMaskBits = 4;
    private readonly uint _pixelColorExportMasks = ComputePixelColorExportMasks(Instructions);
    private readonly uint _parameterExportMask = ComputeParameterExportMask(Instructions);

    public uint PixelColorExportMasks => _pixelColorExportMasks;

    public uint ParameterExportMask => _parameterExportMask;

    private static uint ComputePixelColorExportMasks(
        IReadOnlyList<Gen5ShaderInstruction> instructions)
    {
        var masks = 0u;
        foreach (var instruction in instructions)
        {
            if (instruction.Control is Gen5ExportControl export &&
                export.Target < PixelColorTargetCount)
            {
                masks |= (export.EnableMask & 0xFu) <<
                    (int)(export.Target * PixelColorMaskBits);
            }
        }

        return masks;
    }

    private static uint ComputeParameterExportMask(
        IReadOnlyList<Gen5ShaderInstruction> instructions)
    {
        var mask = 0u;
        foreach (var instruction in instructions)
        {
            if (instruction.Control is Gen5ExportControl export &&
                export.Target is >= 32 and < 64 &&
                export.EnableMask != 0)
            {
                mask |= 1u << (int)(export.Target - 32);
            }
        }

        return mask;
    }

    public IEnumerable<Gen5ImageControl> ImageResources =>
        Instructions
            .Select(instruction => instruction.Control)
            .OfType<Gen5ImageControl>();
}
