// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Agc;

/// <summary>The x86-64 register encodings an <see cref="AgcPacketStubPart"/> may read (SysV integer arguments 2-5).</summary>
public enum AgcStubRegister : byte
{
    Rcx = 1,
    Rdx = 2,
    Rsi = 6,
    R8 = 8,
}

/// <summary>One term of a packet dword: <c>((register &gt;&gt; ShiftRight) &amp; Mask) &lt;&lt; ShiftLeft</c>, in 32 bits.</summary>
public readonly record struct AgcPacketStubPart(AgcStubRegister Register, int ShiftRight, uint Mask, int ShiftLeft);

public enum AgcPacketStubWordKind : byte
{
    /// <summary>The constant ORed with every part.</summary>
    Parts,

    /// <summary><c>max(first part, 1)</c> (the constant is 0).</summary>
    AtLeastOne,

    /// <summary>
    /// The draw initiator of a modifier in the first part's register: 0 when bit 32 is set, else
    /// <c>((uint)modifier &gt;&gt; 3) &amp; 0x20</c>; the constant is ORed in either way.
    /// </summary>
    DrawInitiator,

    /// <summary>
    /// The OR of the word's <see cref="AgcPacketStubWord.Slots"/> and the constant: the user-SGPR
    /// locations an indirect draw's modifier selects (see <c>AgcExports.IndexIndirectRegisterLocations</c>).
    /// </summary>
    ModifierSgprSlots,
}

/// <summary>
/// One register location an indirect draw's modifier (the low 32 bits of <paramref name="Register"/>)
/// selects, shifted left by <paramref name="ShiftLeft"/>: when <c>modifier &amp; EnableMask</c> is
/// clear or <c>modifier &amp; DisableMask</c> is not, <paramref name="Disabled"/>; else, for
/// <paramref name="FieldShift"/> -1, the flag 1; else the stage's user-SGPR base (0x10C for stages
/// 3 and 5 in bits 29..31, 0x8C otherwise) plus <c>(modifier &gt;&gt; FieldShift) &amp; 0x1F</c>.
/// </summary>
public readonly record struct AgcModifierSgprSlot(AgcStubRegister Register, uint EnableMask, int FieldShift, uint Disabled, int ShiftLeft)
{
    /// <summary>Modifier bits that, when any is set, disable the slot (the user-VGPR bit 4).</summary>
    public uint DisableMask { get; init; }
}

public sealed record AgcPacketStubWord(uint Constant, AgcPacketStubPart[] Parts, AgcPacketStubWordKind Kind = AgcPacketStubWordKind.Parts)
{
    /// <summary>The slots of a <see cref="AgcPacketStubWordKind.ModifierSgprSlots"/> word.</summary>
    public AgcModifierSgprSlot[] Slots { get; init; } = [];
}

/// <summary>
/// A command builder whose whole effect is "append these dwords at the command buffer's cursor and
/// advance the cursor", for a buffer that has the room: the one case a native stub in the guest can
/// do exactly as the export does. Every dword is a function of the integer arguments in registers.
/// The stub declines (the export runs) for a null buffer or one without room for the packet and the
/// buffer's reserve, which includes everything that would call the buffer's full callback.
/// </summary>
/// <param name="SizeDwords">The dwords the builder advances the cursor by (1 to 15).</param>
/// <param name="ReturnDwordOffset">The packet address the builder returns, in dwords from the cursor it started at.</param>
public sealed record AgcPacketStubSpec(uint SizeDwords, uint ReturnDwordOffset, AgcPacketStubWord[] Words);

public static partial class AgcExports
{
    // SHARPEMU_AGC_PACKET_STUBS=0 keeps every builder on the managed exports.
    public static readonly bool PacketStubsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_AGC_PACKET_STUBS") != "0";

    /// <summary>True while AGC call tracing wants every builder call in the export.</summary>
    public static bool PacketStubsBlockedByTracing => _traceAgc;

    private static AgcPacketStubWord Const(uint value) => new(value, []);

    private static AgcPacketStubWord Arg(AgcStubRegister register, int shiftRight = 0, uint mask = 0xFFFF_FFFFu, int shiftLeft = 0, uint constant = 0) =>
        new(constant, [new AgcPacketStubPart(register, shiftRight, mask, shiftLeft)]);

    private static AgcPacketStubSpec DirectRegisterSpec(uint op) => new(3, 0,
        [Const(Pm4(3, op, 0)), Arg(AgcStubRegister.Rsi, 0, 0xFFFFu), Arg(AgcStubRegister.Rsi, 32)]);

    private static AgcPacketStubSpec IndirectRegistersSpec(uint op) => new(5, 0,
        [Const(Pm4(5, op, 0)), Arg(AgcStubRegister.Rsi, 0, 0xFFFF_FFFCu), Arg(AgcStubRegister.Rsi, 32), Const(0x8000_0000u), Arg(AgcStubRegister.Rdx, 0, 0x3FFFu)]);

    private static AgcPacketStubWord DrawInitiatorOfRcx() =>
        new(0, [new AgcPacketStubPart(AgcStubRegister.Rcx, 0, 0xFFFF_FFFFu, 0)], AgcPacketStubWordKind.DrawInitiator);

    private static AgcPacketStubWord Slots(params AgcModifierSgprSlot[] slots) =>
        new(0, [], AgcPacketStubWordKind.ModifierSgprSlots) { Slots = slots };

    // The modifier slots of IndexIndirectRegisterLocations (rdx = modifier): base vertex (bit 0) and
    // first instance (bit 2), each none (0x280) when the modifier passes the values in VGPRs (bit 4).
    // A single draw has no draw-index slot.
    private static readonly AgcModifierSgprSlot BaseVertexSlot =
        new(AgcStubRegister.Rdx, DrawModifierStartVertex, 9, 0x280, 0) { DisableMask = DrawModifierUserVgprs };
    private static readonly AgcModifierSgprSlot FirstInstanceSlot =
        new(AgcStubRegister.Rdx, DrawModifierStartInstance, 19, 0x280, 0) { DisableMask = DrawModifierUserVgprs };

    // IndirectDrawInitiator of rdx.
    private static AgcPacketStubWord IndirectDrawInitiatorOfRdx() =>
        new(2, [new AgcPacketStubPart(AgcStubRegister.Rdx, 0, 0xFFFF_FFFFu, 0)], AgcPacketStubWordKind.DrawInitiator);

    // The leaf builders that append one fixed-shape packet, by NID. Each entry mirrors its export's
    // TryEmitCommandDwords call word for word (AgcPacketStubTests runs each stub against its export).
    private static readonly Dictionary<string, AgcPacketStubSpec> PacketStubSpecs = new()
    {
        ["pFLArOT53+w"] = DirectRegisterSpec(ItSetShReg),                    // sceAgcDcbSetShRegisterDirect
        ["LHFXRrlTPD8"] = DirectRegisterSpec(ItSetContextReg),               // sceAgcDcbSetCxRegisterDirect
        ["w4-d0n60hdo"] = DirectRegisterSpec(ItSetUconfigReg),               // sceAgcDcbSetUcRegisterDirect
        ["-HOOCn0JY48"] = IndirectRegistersSpec(ItSetShRegIndirect),         // sceAgcDcbSetShRegistersIndirect
        ["ZvwO9euwYzc"] = IndirectRegistersSpec(ItSetContextRegIndirect),    // sceAgcDcbSetCxRegistersIndirect
        ["hvUfkUIQcOE"] = IndirectRegistersSpec(ItSetUconfigRegIndirect),    // sceAgcDcbSetUcRegistersIndirect
        ["tSBxhAPyytQ"] = new(2, 0, [Const(Pm4(2, ItNumInstances, 0)), Arg(AgcStubRegister.Rsi)]),   // sceAgcDcbSetNumInstances
        ["8N2tmT3jmC8"] = new(2, 0, [Const(Pm4(2, ItNop, RIndexCount)), Arg(AgcStubRegister.Rsi)]),  // sceAgcDcbSetIndexCount
        ["GIIW2J37e70"] = new(3, 0,                                                                  // sceAgcDcbSetIndexSize
        [
            Const(Pm4(3, ItSetUconfigRegIndex, 0)),
            Const(0x2000_0000u | VgtIndexType),
            new AgcPacketStubWord(0x400u, [new(AgcStubRegister.Rsi, 0, 3, 0), new(AgcStubRegister.Rdx, 0, 3, 6)]),
        ]),
        ["l4fM9K-Lyks"] = new(5, 0,                                                                  // sceAgcDcbSetIndexBuffer
        [
            Const(Pm4(3, ItIndexBase, 0)), Arg(AgcStubRegister.Rsi), Arg(AgcStubRegister.Rsi, 32),
            Const(Pm4(2, ItIndexBufferSize, 0)), Arg(AgcStubRegister.Rdx),
        ]),
        ["Yw0jKSqop+E"] = new(7, 0,                                                                  // sceAgcDcbDrawIndexAuto
            [Const(Pm4(7, ItNop, RDrawIndexAuto)), Arg(AgcStubRegister.Rsi), Const(0), Const(0), Const(0), Const(0), Const(0)]),
        ["B+aG9DUnTKA"] = new(5, 0,                                                                  // sceAgcDcbDrawIndexOffset
        [
            Const(Pm4(5, ItDrawIndexOffset2, 0)),
            new AgcPacketStubWord(0, [new(AgcStubRegister.Rdx, 0, 0xFFFF_FFFFu, 0)], AgcPacketStubWordKind.AtLeastOne),
            Arg(AgcStubRegister.Rsi), Arg(AgcStubRegister.Rdx), DrawInitiatorOfRcx(),
        ]),
        ["q88lQ+GP5Yk"] = new(11, 5,                                                                 // sceAgcDcbDrawIndex: INDEX_BASE, INDEX_BUFFER_SIZE, DRAW_INDEX_2
        [
            Const(Pm4(3, ItIndexBase, 0)), Arg(AgcStubRegister.Rdx), Arg(AgcStubRegister.Rdx, 32),
            Const(Pm4(2, ItIndexBufferSize, 0)), Arg(AgcStubRegister.Rsi),
            Const(Pm4(6, ItDrawIndex2, 0)), Arg(AgcStubRegister.Rsi), Arg(AgcStubRegister.Rdx), Arg(AgcStubRegister.Rdx, 32), Arg(AgcStubRegister.Rsi),
            DrawInitiatorOfRcx(),
        ]),
        ["t1vNu082-jM"] = new(5, 0,                                                                  // sceAgcDcbDrawIndexIndirect
        [
            Const(Pm4(5, ItDrawIndexIndirect, 0)), Arg(AgcStubRegister.Rsi),
            Slots(BaseVertexSlot), Slots(FirstInstanceSlot), IndirectDrawInitiatorOfRdx(),
        ]),
        ["1q1titRBL6o"] = new(5, 0,                                                                  // sceAgcDcbDrawIndirect
        [
            Const(Pm4(5, ItDrawIndirect, 0)), Arg(AgcStubRegister.Rsi),
            Slots(BaseVertexSlot), Slots(FirstInstanceSlot), IndirectDrawInitiatorOfRdx(),
        ]),
        ["k3GhuSNmBLU"] = new(5, 0,                                                                  // sceAgcCbDispatch
        [
            Const(Pm4(5, ItDispatchDirect, 0)), Arg(AgcStubRegister.Rsi), Arg(AgcStubRegister.Rdx), Arg(AgcStubRegister.Rcx),
            Arg(AgcStubRegister.R8, 0, 0xA038u, 0, 0x41u),
        ]),
    };

    /// <summary>The packet stub for a builder import, when it has one.</summary>
    public static bool TryGetPacketStubSpec(string nid, out AgcPacketStubSpec spec) =>
        PacketStubSpecs.TryGetValue(nid, out spec!);

    /// <summary>The marker NOP header and payload sceAgcCbSetShRegisterRangeDirect puts in front of its packet.</summary>
    public static readonly uint RangePacketNopHeader = Pm4(2, ItNop, RZero);

    public const uint RangePacketMarker = CbSetShRegisterRangeMarker;

    /// <summary>The SET_SH_REG header of a zero-length payload; the stub ORs the value count shifted left by 16.</summary>
    public static readonly uint SetShRegHeaderWithoutCount = Pm4(2, ItSetShReg, 0);

    /// <summary>The most values a variable-length packet stub copies; a larger packet is the export's.</summary>
    public const int RangeStubMaxValues = 64;

    /// <summary>
    /// The most values sceAgcCbSetShRegisterRangeDirect reserves for a null source (the packet's
    /// count field limit, <c>TryGetCbSetShRegisterRangeDirectLayout</c>); the stub copies nothing then.
    /// </summary>
    public const int RangeStubMaxReservedValues = 0x3FFF;

    public const int RegistersStubMaxRegisters = 32;

    private const string CbSetShRegisterRangeDirectNid = "n2fD4A+pb+g";
    private const string CbSetShRegistersDirectNid = "UZbQjYAwwXM";

    // SHARPEMU_AGC_PATCH_STUBS=0 keeps the sceAgcSet{Cx,Sh,Uc}RegIndirectPatch* patchers on the managed
    // exports (SHARPEMU_AGC_PACKET_STUBS=0 does too).
    public static readonly bool PatchStubsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_AGC_PATCH_STUBS") != "0";

    /// <summary>The header bits the patchers test (length and opcode); the stub masks the header with this.</summary>
    public const uint RegIndirectPatchHeaderMask = 0x3FFF_FF00u;

    /// <summary>The count field of a SET_*_REG_INDIRECT packet (its fifth dword).</summary>
    public const uint RegIndirectPatchCountMask = 0x3FFFu;

    // The register-indirect patchers, by NID. Each runs SetIndirectPatchAddress /
    // SetIndirectPatchRegisterCount / AddIndirectPatchRegisters on the 5-dword packet the
    // sceAgcDcbSet*RegistersIndirect builders write (AgcPacketStubTests runs each stub against its export).
    private static readonly Dictionary<string, AgcRegIndirectPatchStubSpec> RegIndirectPatchStubSpecs = new()
    {
        ["vcmNN+AAXnY"] = new(AgcRegIndirectPatchStub.SetAddress, Pm4(5, ItSetContextRegIndirect, 0)),       // sceAgcSetCxRegIndirectPatchSetAddress
        ["Qrj4c+61z4A"] = new(AgcRegIndirectPatchStub.SetAddress, Pm4(5, ItSetShRegIndirect, 0)),            // sceAgcSetShRegIndirectPatchSetAddress
        ["6lNcCp+fxi4"] = new(AgcRegIndirectPatchStub.SetAddress, Pm4(5, ItSetUconfigRegIndirect, 0)),       // sceAgcSetUcRegIndirectPatchSetAddress
        ["whb1RL7K4Ss"] = new(AgcRegIndirectPatchStub.SetNumRegisters, Pm4(5, ItSetContextRegIndirect, 0)),  // sceAgcSetCxRegIndirectPatchSetNumRegisters
        ["nCUgItdN2ms"] = new(AgcRegIndirectPatchStub.SetNumRegisters, Pm4(5, ItSetShRegIndirect, 0)),       // sceAgcSetShRegIndirectPatchSetNumRegisters
        ["fRG-JOH5+sI"] = new(AgcRegIndirectPatchStub.SetNumRegisters, Pm4(5, ItSetUconfigRegIndirect, 0)),  // sceAgcSetUcRegIndirectPatchSetNumRegisters
        ["d-6uF9sZDIU"] = new(AgcRegIndirectPatchStub.AddRegisters, Pm4(5, ItSetContextRegIndirect, 0)),     // sceAgcSetCxRegIndirectPatchAddRegisters
        ["z2duB-hHQSM"] = new(AgcRegIndirectPatchStub.AddRegisters, Pm4(5, ItSetShRegIndirect, 0)),          // sceAgcSetShRegIndirectPatchAddRegisters
        ["vRoArM9zaIk"] = new(AgcRegIndirectPatchStub.AddRegisters, Pm4(5, ItSetUconfigRegIndirect, 0)),     // sceAgcSetUcRegIndirectPatchAddRegisters
    };

    /// <summary>The native stub for a register-indirect patcher import, when it has one.</summary>
    public static bool TryGetRegIndirectPatchStub(string nid, out AgcRegIndirectPatchStubSpec spec) =>
        RegIndirectPatchStubSpecs.TryGetValue(nid, out spec);

    private const string GetDataPacketPayloadAddressNid = "V++UgBtQhn0";

    /// <summary>The header bits sceAgcGetDataPacketPayloadAddress tests for a type-0 packet (its count field).</summary>
    public const uint PayloadAddressCountMask = 0x3FFF_0000u;

    /// <summary>
    /// Whether <paramref name="nid"/> is sceAgcGetDataPacketPayloadAddress, whose native stub
    /// (EmitGuestAgcPayloadAddressStub) answers the export's success path.
    /// </summary>
    public static bool TryGetPayloadAddressStub(string nid) => nid == GetDataPacketPayloadAddressNid;

    /// <summary>The variable-length packet stub for a builder import, when it has one.</summary>
    public static bool TryGetVariablePacketStub(string nid, out AgcVariablePacketStub kind)
    {
        switch (nid)
        {
            case CbSetShRegisterRangeDirectNid:
                kind = AgcVariablePacketStub.ShRegisterRange;
                return true;
            case CbSetShRegistersDirectNid:
                kind = AgcVariablePacketStub.ShRegisters;
                return true;
            default:
                kind = default;
                return false;
        }
    }
}

public enum AgcVariablePacketStub
{
    /// <summary>sceAgcCbSetShRegisterRangeDirect(cb, offset, values, count): marker NOP + SET_SH_REG with the values copied in.</summary>
    ShRegisterRange,

    /// <summary>sceAgcCbSetShRegistersDirect(cb, (offset, value) pairs, count) for one run of consecutive ascending offsets.</summary>
    ShRegisters,
}

/// <summary>What a register-indirect patcher does to the packet in RDI, given the argument in RSI.</summary>
public enum AgcRegIndirectPatchStub : byte
{
    /// <summary>Set*RegIndirectPatchSetAddress(packet, registers): address dwords at +4 (low two bits kept) and +8.</summary>
    SetAddress,

    /// <summary>Set*RegIndirectPatchSetNumRegisters(packet, count): the count field at +16.</summary>
    SetNumRegisters,

    /// <summary>Set*RegIndirectPatchAddRegisters(packet, count): count added into the field at +16, wrapping inside it.</summary>
    AddRegisters,
}

/// <summary>
/// A register-indirect patcher as native code: it patches a SET_*_REG_INDIRECT packet whose header,
/// masked with <c>AgcExports.RegIndirectPatchHeaderMask</c>, equals <paramref name="Header"/>'s (length 5
/// and the space's opcode), and declines (the export runs) for anything else: a null or out-of-range
/// packet pointer, a packet that crosses a page, the legacy NOP layout, a null address.
/// </summary>
/// <param name="Header">The space's SET_*_REG_INDIRECT header as the builder writes it.</param>
public readonly record struct AgcRegIndirectPatchStubSpec(AgcRegIndirectPatchStub Kind, uint Header);
