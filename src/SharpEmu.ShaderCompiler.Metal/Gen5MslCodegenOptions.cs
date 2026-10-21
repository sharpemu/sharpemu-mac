// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal;

/// <summary>
/// Codegen switches, passed per compile so a harness can translate the same program two
/// ways in one process without racing other compiles. The emulator uses
/// <see cref="FromEnvironment"/>, which is how each half of the change can be taken out
/// during bring-up.
/// </summary>
/// <param name="StructuredControlFlow">Nest the blocks in if/else and while instead of
/// running them from a PC dispatcher. <c>SHARPEMU_MSL_STRUCTURED=0</c> forces the
/// dispatcher.</param>
/// <param name="RegisterAllocation">Declare only the guest registers a program names,
/// instead of both register files at their architectural size.
/// <c>SHARPEMU_MSL_REGALLOC=0</c> forces the full files.</param>
/// <param name="ForceRegisterInitialization">Zero-fill every declared register instead of
/// only the ones some path reads before writing. <c>SHARPEMU_MSL_REGINIT=1</c> turns this
/// on.</param>
/// <param name="ExecGuardElision">Drop the per-lane EXEC guard where the program proves
/// every lane is set. <c>SHARPEMU_MSL_EXECGUARD=0</c> keeps every guard.</param>
/// <param name="BoundedRelativeMoves">Resolve V_MOVREL*'s register — from M0's known
/// values where straight-line code produces them, and otherwise over the registers the
/// program touches — so the register file needs no run-time index and can be scalarized.
/// <c>SHARPEMU_MSL_MOVREL=0</c> indexes the file instead, which keeps it an array.</param>
/// <param name="RawDwordAccess">Read and write a raw buffer dword with one indexed
/// access, which is how the GCN ISA addresses it and what the SPIR-V backend emits,
/// instead of assembling it from bytes. <c>SHARPEMU_MSL_DWORDACCESS=0</c> goes back to
/// the byte path.</param>
/// <param name="UniformTablesInConstantSpace">Point at the read-only uniform tables (the
/// user-data words, the flattened resource table) through the constant address space,
/// which is Apple's uniform datapath, instead of the device one.
/// <c>SHARPEMU_MSL_CONSTANTTABLES=0</c> goes back to device.</param>
/// <param name="DispatchThreadgroupMemory">Declare the threadgroup memory the dispatch
/// allocated rather than the full 32 KB. Threadgroup memory is a hard occupancy limit, so
/// this is the difference between one threadgroup per core and many.
/// <c>SHARPEMU_MSL_LDSSIZE=0</c> declares all of it.</param>
/// <param name="DeadMaskBallotElision">Skip a wave64 VCC or EXEC ballot whose 64-bit pair
/// nothing reads before it is overwritten, which saves the two threadgroup barriers the
/// cross-simdgroup exchange costs. <c>SHARPEMU_MSL_MASKBALLOT=0</c> materializes every
/// ballot.</param>
/// <param name="ElideByteBias">Drop the per-buffer byte bias (the packed memory-offset
/// bytes of the shader data) from every buffer address. The Metal host never writes
/// those bytes, so each bias is zero and the read and add are dead work.
/// <c>SHARPEMU_MSL_BYTEBIAS=1</c> keeps them.</param>
/// <param name="MipStatisticsFunctionConstant">Guard each GET_LOD_STATS counter site with the
/// function constant <see cref="Gen5MslShader.MipStatisticsFunctionConstantName"/>, so the
/// host compiles the counters out of a pixel pipeline until the title asks for LOD
/// statistics. The MSL text does not depend on when that happens, only the function the host
/// makes of it. <c>SHARPEMU_LOD_STATS=always</c> leaves the sites unguarded: a pipeline has
/// the counters from the first draw.</param>
/// <param name="GraphicsLoopStepCap">Charge the step cap on every loop iteration of a structured
/// vertex or pixel program too. The cap is for a mistranslated exit condition that would wedge
/// the queue; a graphics program's runaway loop is ended by the GPU watchdog, and the counter
/// is a register and a compare on every back edge. Compute and mesh programs keep the cap
/// either way. <c>SHARPEMU_SHADER_MAX_STEPS_GRAPHICS=1</c> restores it.</param>
/// <param name="LaneSpillCopies">Serve a wave64 V_READLANE_B32 that reloads an SGPR the
/// guest compiler spilled to a VGPR lane from a per-thread copy taken at the spill, which
/// saves the cross-simdgroup exchange and its barrier
/// (see <see cref="Ir.Gen5LaneSpillAnalysis"/>). <c>SHARPEMU_MSL_SPILLCOPY=0</c> reads every
/// lane through the exchange.</param>
/// <param name="PersistentWaitFences">Make every S_WAITCNT of a persistent-thread kernel a
/// device-scope memory fence. Such a kernel hands work between workgroups through memory: a
/// group stores an item, waits for its stores (S_WAITCNT_VSCNT), then publishes it with an
/// atomic, and a consumer loads the published index and waits before it loads the item. The
/// guest GPU's L2 makes a completed store visible to every CU; Metal orders nothing between a
/// plain store and a relaxed atomic, so another workgroup could take the index before the item,
/// and nothing makes the store leave the producer's cache soon: a consumer polled until it did.
/// UE5's two Nanite culling kernels took 58 and 34 ms of a 260 ms Black Myth: Wukong cinematic
/// frame doing that at 64 workgroups, and 0.1 ms with the fences; with more workgroups they
/// also handed out stale clusters (black speckles over the army).
/// <c>SHARPEMU_MSL_PERSISTENT_FENCES=0</c> leaves S_WAITCNT a no-op there too.</param>
/// <param name="PersistentRecordLocks">Make a persistent-thread kernel's multi-dword buffer
/// records one unit to other workgroups: a STORE_DWORDX2-X4 takes the sequence lock of its
/// first dword and a GLC LOAD_DWORDX3/X4 reads under it, as a GLC LOAD_DWORDX2 already does. The
/// guest writes and reads such a record with one instruction, which other CUs see whole; here
/// each dword is its own access, and UE5's Nanite culling read queue records half written.
/// <c>SHARPEMU_MSL_PERSISTENT_RECORDS=0</c> accesses them dword by dword.</param>
public sealed record Gen5MslCodegenOptions(
    bool StructuredControlFlow = true,
    bool RegisterAllocation = true,
    bool ForceRegisterInitialization = false,
    bool ExecGuardElision = true,
    bool BoundedRelativeMoves = true,
    bool RawDwordAccess = true,
    bool UniformTablesInConstantSpace = true,
    bool DispatchThreadgroupMemory = true,
    bool DeadMaskBallotElision = true,
    bool ElideByteBias = true,
    bool MipStatisticsFunctionConstant = true,
    bool GraphicsLoopStepCap = false,
    bool LaneSpillCopies = true,
    bool PersistentWaitFences = true,
    bool PersistentRecordLocks = true)
{
    public static Gen5MslCodegenOptions FromEnvironment { get; } = new(
        StructuredControlFlow: Environment.GetEnvironmentVariable("SHARPEMU_MSL_STRUCTURED") != "0",
        RegisterAllocation: Environment.GetEnvironmentVariable("SHARPEMU_MSL_REGALLOC") != "0",
        ForceRegisterInitialization: Environment.GetEnvironmentVariable("SHARPEMU_MSL_REGINIT") == "1",
        ExecGuardElision: Environment.GetEnvironmentVariable("SHARPEMU_MSL_EXECGUARD") != "0",
        BoundedRelativeMoves: Environment.GetEnvironmentVariable("SHARPEMU_MSL_MOVREL") != "0",
        RawDwordAccess: Environment.GetEnvironmentVariable("SHARPEMU_MSL_DWORDACCESS") != "0",
        UniformTablesInConstantSpace:
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_CONSTANTTABLES") != "0",
        DispatchThreadgroupMemory: Environment.GetEnvironmentVariable("SHARPEMU_MSL_LDSSIZE") != "0",
        DeadMaskBallotElision:
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_MASKBALLOT") != "0",
        ElideByteBias: Environment.GetEnvironmentVariable("SHARPEMU_MSL_BYTEBIAS") != "1",
        MipStatisticsFunctionConstant: Environment.GetEnvironmentVariable("SHARPEMU_LOD_STATS") != "always",
        GraphicsLoopStepCap: Environment.GetEnvironmentVariable("SHARPEMU_SHADER_MAX_STEPS_GRAPHICS") == "1",
        LaneSpillCopies: Environment.GetEnvironmentVariable("SHARPEMU_MSL_SPILLCOPY") != "0",
        PersistentWaitFences: Environment.GetEnvironmentVariable("SHARPEMU_MSL_PERSISTENT_FENCES") != "0",
        PersistentRecordLocks: Environment.GetEnvironmentVariable("SHARPEMU_MSL_PERSISTENT_RECORDS") != "0");
}
