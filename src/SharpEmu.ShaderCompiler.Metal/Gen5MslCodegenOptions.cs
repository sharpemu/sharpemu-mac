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
/// <param name="LaneSpillCopies">Serve a V_READLANE_B32 that reloads an SGPR the guest compiler
/// spilled to a VGPR lane from a per-thread copy taken at the spill: for wave64 it saves the
/// cross-simdgroup exchange and its barrier, and it keeps the spill when the lane is missing from
/// a partial simdgroup (see <see cref="Ir.Gen5LaneSpillAnalysis"/>). <c>SHARPEMU_MSL_SPILLCOPY=0</c>
/// reads every lane, except in a stage that models a single lane, where only the copy is right.</param>
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
/// <param name="DeviceAddressSpans">Resolve a multi-dword global or device-buffer access (GLOBAL_LOAD/STORE
/// DWORDX2-X4, BUFFER_LOAD/STORE through a device descriptor) once for the whole span instead of
/// walking the page table per dword: a span inside one 16 KB page indexes off the one pointer and
/// one that crosses a page falls back to the per-dword path. <c>SHARPEMU_MSL_DEVICESPAN=0</c>
/// resolves every dword alone.</param>
/// <param name="EarlyFragmentTests">Declare <c>[[early_fragment_tests]]</c> on a pixel function whose
/// guest DB_SHADER_CONTROL asks for early Z (<see cref="ShaderCompileRequest.PixelEarlyDepth"/>): Metal
/// only tests depth before the shader on its own for a function that writes no memory, so a shader
/// with buffer or image stores ran for fragments the guest hardware would have rejected first.
/// <c>SHARPEMU_MSL_EARLYZ=0</c> leaves the choice to Metal.</param>
/// <param name="VectorDwordLoads">Read the dwords of an S_BUFFER_LOAD/S_LOAD_DWORDXn from a bound
/// buffer, and of a BUFFER_LOAD_DWORDXn with a contiguous address, in runs of up to four behind one
/// bounds check (sharpemu_load_dwords) instead of one checked load per dword. A run that crosses
/// the end of the binding still reads each dword alone, so nothing a shader sees changes.
/// <c>SHARPEMU_MSL_VECLOAD=0</c> loads dword by dword.</param>
/// <param name="ThreadgroupSizeHint">Declare a compute kernel's workgroup size as
/// <c>[[max_total_threads_per_threadgroup(N)]]</c>, so the compiler budgets registers for the N
/// threads the dispatch launches rather than the device's 1024. Only a size the dispatch is
/// known to use (more than one thread) is declared. <c>SHARPEMU_MSL_TGHINT=0</c> leaves it
/// out.</param>
/// <param name="RuntimeDwordFormatLoads">Read identity-swizzled 32-bit formatted buffer
/// components as one checked run when the current descriptor permits it. Other formats,
/// swizzles and bounds retain the general decoder. <c>SHARPEMU_MSL_RUNTIME_DWORD_FORMAT=0</c>
/// disables this run-time fast path.</param>
/// <param name="NativeFloatMinMax">Lower V_MIN/MAX/MIN3/MAX3/MED3_F32 outside IEEE mode to float
/// compares and selects on the operand bits, where that gives the bit-level helpers' answer for
/// every input: against an inline constant, and between two registers while FLOAT_MODE flushes
/// f32 input denormals. Everything else keeps the helpers. <c>SHARPEMU_MSL_FMINMAX=0</c> keeps
/// the helpers everywhere.</param>
/// <param name="ConstantTableSlots">Read a flattened-table word at a slot fixed at translation
/// time without the bounds select: the shader reports the highest such slot
/// (<see cref="Gen5MslShader.FlattenedTableMinimumWords"/>) and the host zero-pads the table it
/// uploads to it, so a slot past the evaluated table still reads 0.
/// <c>SHARPEMU_MSL_TABLEPAD=0</c> keeps the select.</param>
/// <param name="FastReciprocals">Lower V_RCP/V_RSQ_F32 to the fast:: forms, retaining
/// input flush and special results (0, infinity, negative, NaN). Device sweeps check a
/// one-ULP budget against the precise forms. V_SQRT_F32 stays precise because fast::sqrt
/// exceeded that budget. <c>SHARPEMU_MSL_FASTRCP=0</c> keeps the precise forms.</param>
/// <param name="ContractPixelCompute">Leave out <c>#pragma clang fp contract(off)</c> in pixel and
/// compute programs, so the relaxed math mode may fuse a multiply and an add. V_MAD_F32 rounds
/// the product on the guest, so a fused result can differ by an ULP; vertex and mesh programs,
/// which feed positions, keep the pragma either way. Off by default;
/// <c>SHARPEMU_MSL_CONTRACT=1</c> turns it on.</param>
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
    bool PersistentRecordLocks = true,
    bool DeviceAddressSpans = true,
    bool EarlyFragmentTests = true,
    bool VectorDwordLoads = true,
    bool ThreadgroupSizeHint = true,
    bool NativeFloatMinMax = true,
    bool ConstantTableSlots = true,
    bool FastReciprocals = true,
    bool ContractPixelCompute = false,
    bool RuntimeDwordFormatLoads = true)
{
    /// <summary>Diagnostic threshold for moving mesh LDS from threadgroup memory to the
    /// device pool, in bytes. The supported mesh allocation is at most 16 KiB. Zero
    /// selects the device pool for every nonempty allocation; 16384 keeps all allocations
    /// in threadgroup memory. The default preserves the existing 12 KiB policy.</summary>
    public uint MeshDeviceLdsThresholdBytes { get; init; } = 12288;

    /// <summary>Experimental structured mesh exits, enabled with SHARPEMU_MSL_MESH_EXIT_JOIN=1.
    /// A single hardware wave may also publish its output at a loop-limit exit.</summary>
    public bool MeshExitJoin { get; init; }

    /// <summary>Experimental constant indexing of provably invariant private scratch slots.
    /// Enabled with SHARPEMU_MSL_STATIC_SCRATCH=1.</summary>
    public bool StaticScratchSlots { get; init; }

    /// <summary>Experimental removal of pure ALU guards when inactive values cannot
    /// be observed. SHARPEMU_MSL_UNOBSERVED_ALU=1 enables this for wave32 compute
    /// and single-wave mesh shaders.</summary>
    public bool UnobservedAluWrites { get; init; }

    /// <summary>Experimental separation of fused geometry input and output capacities,
    /// with complete wave64 execution. SHARPEMU_MSL_GEOMETRY_CAPACITY=1.</summary>
    public bool GeometryOutputCapacity { get; init; }

    /// <summary>Experimental single-run loads for statically known packed formats.
    /// SHARPEMU_MSL_PACKED_FORMAT_LOADS=1; partial trailing elements retain the decoder.</summary>
    public bool PackedFormatLoads { get; init; }

    /// <summary>Experimental guarded forwarding of the last 128-bit LDS write in a
    /// single-wave mesh subgroup. SHARPEMU_MSL_MESH_LDS_FORWARDING=1.</summary>
    public bool MeshLdsForwarding { get; init; }

    /// <summary>Experimental constant-address-space bindings for scalar-only buffers
    /// in mesh shaders without device-memory writes. SHARPEMU_MSL_SCALAR_CONSTANT_BUFFERS=1.</summary>
    public bool ScalarConstantBuffers { get; init; }

    /// <summary>Experimental compact threadgroup bank for frequently addressed LDS
    /// regions, retaining device storage for other addresses. SHARPEMU_MSL_MESH_HOT_LDS=1.</summary>
    public bool MeshHotLds { get; init; }

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
        PersistentRecordLocks: Environment.GetEnvironmentVariable("SHARPEMU_MSL_PERSISTENT_RECORDS") != "0",
        DeviceAddressSpans: Environment.GetEnvironmentVariable("SHARPEMU_MSL_DEVICESPAN") != "0",
        EarlyFragmentTests: Environment.GetEnvironmentVariable("SHARPEMU_MSL_EARLYZ") != "0",
        VectorDwordLoads: Environment.GetEnvironmentVariable("SHARPEMU_MSL_VECLOAD") != "0",
        ThreadgroupSizeHint: Environment.GetEnvironmentVariable("SHARPEMU_MSL_TGHINT") != "0",
        NativeFloatMinMax: Environment.GetEnvironmentVariable("SHARPEMU_MSL_FMINMAX") != "0",
        ConstantTableSlots: Environment.GetEnvironmentVariable("SHARPEMU_MSL_TABLEPAD") != "0",
        FastReciprocals: Environment.GetEnvironmentVariable("SHARPEMU_MSL_FASTRCP") != "0",
        ContractPixelCompute: Environment.GetEnvironmentVariable("SHARPEMU_MSL_CONTRACT") == "1",
        RuntimeDwordFormatLoads: Environment.GetEnvironmentVariable("SHARPEMU_MSL_RUNTIME_DWORD_FORMAT") != "0")
    {
        UnobservedAluWrites = Environment.GetEnvironmentVariable("SHARPEMU_MSL_UNOBSERVED_ALU") == "1",
        GeometryOutputCapacity = Environment.GetEnvironmentVariable("SHARPEMU_MSL_GEOMETRY_CAPACITY") == "1",
        PackedFormatLoads = Environment.GetEnvironmentVariable("SHARPEMU_MSL_PACKED_FORMAT_LOADS") == "1",
        MeshLdsForwarding = Environment.GetEnvironmentVariable("SHARPEMU_MSL_MESH_LDS_FORWARDING") == "1",
        ScalarConstantBuffers = Environment.GetEnvironmentVariable("SHARPEMU_MSL_SCALAR_CONSTANT_BUFFERS") == "1",
        MeshHotLds = Environment.GetEnvironmentVariable("SHARPEMU_MSL_MESH_HOT_LDS") == "1",
        StaticScratchSlots = Environment.GetEnvironmentVariable("SHARPEMU_MSL_STATIC_SCRATCH") == "1",
        MeshExitJoin = Environment.GetEnvironmentVariable("SHARPEMU_MSL_MESH_EXIT_JOIN") == "1",
        MeshDeviceLdsThresholdBytes = uint.TryParse(
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_MESH_LDS_THRESHOLD"), out var threshold)
                ? Math.Min(threshold, 16384u) : 12288u,
    };
}
