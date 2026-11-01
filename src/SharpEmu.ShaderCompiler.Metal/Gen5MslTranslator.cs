// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Text;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

/// <summary>The optional helper blocks a translated module carries ahead of its entry point.</summary>
[Flags]
internal enum Gen5MslPreludes
{
    None = 0,
    FormatLoads = 1 << 0,
    RealWaveLanes = 1 << 1,
    BufferOffset = 1 << 2,
    CoherentLoads = 1 << 3,
    PairLocks = 1 << 4,
    Float64 = 1 << 5,
    Bvh = 1 << 6,
    MipStatistics = 1 << 7,
    ScalarConstants = 1 << 8,
    DecodedFormatLoads = 1 << 9,
    All = (1 << 10) - 1,
}

// Compiles shader requests into Metal source with argument-buffer bindings.
public static partial class Gen5MslTranslator
{
    /// <summary>
    /// Every helper block a module with <paramref name="preludes"/> carries, in emission
    /// order, without the per-request resource struct (no prelude refers to it).
    /// </summary>
    internal static string RenderModulePreludes(Gen5MslPreludes preludes) =>
        RenderModulePreludes(preludes, Gen5MslCodegenOptions.FromEnvironment);

    internal static string RenderModulePreludes(Gen5MslPreludes preludes, Gen5MslCodegenOptions options)
    {
        var source = new StringBuilder();
        source.AppendLine("#include <metal_stdlib>");
        source.AppendLine();
        source.AppendLine("using namespace metal;");
        source.AppendLine();
        CompilationContext.AppendLeadingPreludes(source, preludes);
        CompilationContext.AppendTrailingPreludes(source, preludes, options.BranchlessDwordLoads);
        return source.ToString();
    }

    private const uint ScalarRegisterFileCount = 128;
    private const uint VectorRegisterFileCount = 256;
    // The most LDS a dispatch can ask for, which is also Metal's threadgroup-memory
    // limit: 8192 dwords is 32 KB either way.
    private const uint LdsDwordCount = 8192;

    /// <summary>
    /// The threadgroup memory a compute dispatch gets, from what it allocated
    /// (COMPUTE_PGM_RSRC2.LDS_SIZE), rounded up to a power of two because LDS addresses
    /// are masked into bounds rather than clamped. A dispatch that does not say falls
    /// back to the whole 32 KB.
    ///
    /// This matters more than its size suggests. Threadgroup memory is a hard occupancy
    /// limit on Apple GPUs: a threadgroup that asks for all 32 KB is the only one that
    /// fits on a core, so a 64-thread guest workgroup leaves the core running 64 threads
    /// where it could run hundreds. Octopath Traveler II's second- and fourth-heaviest
    /// dispatches are 8x8x1 workgroups that were each declaring the full 32 KB — the
    /// SPIR-V backend has always sized this from the allocation, and now so does this one.
    /// </summary>
    private static uint ComputeLdsDwordCount(ShaderCompileRequest request)
    {
        var declared = request.LocalDataShareDwords;
        return declared == 0 || declared > LdsDwordCount
            ? LdsDwordCount
            : System.Numerics.BitOperations.RoundUpToPowerOf2(declared);
    }
    // Graphics stages model LDS as per-invocation scratch; a full 32 KB array
    // per fragment/vertex invocation risks Metal compile limits, and
    // per-invocation write-then-read correctness only needs deterministic
    // address masking (mirrors the SPIR-V translator's Private-array choice).
    private const uint PrivateLdsDwordCount = 2048;
    private const uint VccLoRegister = 106;
    private const uint VccHiRegister = 107;
    private const uint M0Register = 124;
    private const uint NullRegister = 125;
    private const uint ExecLoRegister = 126;
    private const uint ExecHiRegister = 127;
    // EXP target MRTZ: depth, stencil, sample mask and MRT0 alpha per SPI_SHADER_Z_FORMAT.
    private const uint MrtzExportTarget = 8;

    private static bool ValidatePixelOutputs(IReadOnlyList<Gen5PixelOutputBinding> outputs, out string error)
    {
        error = string.Empty;
        // Synthetic dual-source alpha (an alpha copy at index one) has no Metal emission: the
        // Metal path feeds dual-source blending from the program's own second export instead.
        if (outputs.Any(output => output.DualSourceAlpha))
        {
            error = "synthetic dual-source alpha is not supported by the Metal backend";
            return false;
        }
        var dualSource = outputs.Any(output => output.BlendSourceIndex != 0);
        if (dualSource && (outputs.Count != 2 ||
            outputs.Any(output => output.HostLocation != 0 || output.Kind != Gen5PixelOutputKind.Float) ||
            !outputs.Select(output => output.BlendSourceIndex).Order().SequenceEqual(new uint[] { 0, 1 })))
        {
            error = "dual-source pixel outputs need one target and its second source, at host location zero with indices zero and one";
            return false;
        }
        if (outputs.Count > 8)
        {
            error = "pixel outputs must contain at most eight guest slots in the 0..7 range";
            return false;
        }

        for (var index = 0; index < outputs.Count; index++)
        {
            if (outputs[index].GuestSlot > 7)
            {
                error = "pixel outputs must contain at most eight guest slots in the 0..7 range";
                return false;
            }

            for (var other = index + 1; other < outputs.Count; other++)
            {
                if (outputs[other].GuestSlot == outputs[index].GuestSlot ||
                    (outputs[other].HostLocation == outputs[index].HostLocation &&
                     outputs[other].BlendSourceIndex == outputs[index].BlendSourceIndex))
                {
                    error = "pixel output guest slots and host locations must be unique";
                    return false;
                }
            }
        }

        // Host locations must be dense 0..N-1 so [[color(n)]] attachments match.
        for (uint location = 0; location < (dualSource ? 1 : outputs.Count); location++)
        {
            var found = false;
            foreach (var output in outputs)
            {
                found |= output.HostLocation == location;
            }

            if (!found)
            {
                error = "pixel output host locations must be dense in the 0..N-1 range";
                return false;
            }
        }

        return true;
    }

    /// <summary>The records a SHARPEMU_MSL_PC_PROBE in log mode keeps per flush.</summary>
    public const int PcProbeLogRecords = 65536;

    /// <summary>The words of one SHARPEMU_MSL_PC_PROBE log record of a program (probe index, then registers).</summary>
    public static int PcProbeLogRecordWords(ulong hash) => CompilationContext.ProbeRecordWords(hash);

    /// <summary>How the translated shaders in this process came out. A shader whose
    /// control flow no MSL construct expresses falls back to the PC dispatcher, so the
    /// ratio is what says whether the structured emitter is covering a game's shaders;
    /// the render host folds it into its periodic performance line.</summary>
    public static class Statistics
    {
        private static long _structured;
        private static long _dispatcher;

        public static (long Structured, long Dispatcher) ControlFlowForms =>
            (Interlocked.Read(ref _structured), Interlocked.Read(ref _dispatcher));

        internal static void NoteControlFlowForm(bool structured)
        {
            Interlocked.Increment(ref structured ? ref _structured : ref _dispatcher);
        }
    }

    private sealed partial class CompilationContext
    {
        // Safety valve for the PC-dispatcher loop, mirroring the SPIR-V
        // translator: a mistranslated loop-exit condition must terminate the
        // invocation instead of wedging the GPU queue.
        private static readonly int _maxDispatcherSteps =
            int.TryParse(
                Environment.GetEnvironmentVariable("SHARPEMU_SHADER_MAX_STEPS"),
                out var maxSteps) && maxSteps >= 0
                ? maxSteps
                : 100_000;

        // A program that reads memory coherently (a GLC load, or an atomic returning its
        // pre-op value) can be polling for another workgroup's writes: a persistent-thread
        // kernel spins until a shared queue drains, so its trip count follows the whole
        // dispatch's work rather than its own, and the faster each iteration runs the more
        // of them one wait takes: UE5's Nanite culling needs up to ~370k iterations on an
        // M5 Pro once structured, more on a smaller GPU, and cutting it off leaves its
        // never-reset queues corrupt for every later frame. Such programs get a budget
        // well past that; a genuinely stuck one still ends, just later. Only a persistent-thread
        // kernel waits that long by design: any other program that polls (GTA V runs one whose
        // wait never ends here) keeps the plain cap, since each of its capped dispatches holds
        // the queue 64 times as long and stalled GTA's prologue load.
        private const int PollingStepMultiplier = 64;

        private int MaxDispatcherSteps =>
            _maxDispatcherSteps > 0 && PollsCoherentMemory && IsPersistentThreadsKernel()
                ? (int)Math.Min((long)_maxDispatcherSteps * PollingStepMultiplier, int.MaxValue)
                : _maxDispatcherSteps;

        private bool? _pollsCoherentMemory;

        private bool PollsCoherentMemory => _pollsCoherentMemory ??= _request.Program.Instructions.Any(
            static instruction =>
                instruction.Control is Gen5BufferMemoryControl { Glc: true } or Gen5GlobalMemoryControl { Glc: true } &&
                !instruction.Opcode.Contains("Store", StringComparison.Ordinal));

        // The call the dispatcher makes when it runs out of steps. The step-report word
        // is the last field of every stage argument buffer, so it is always in scope.
        private const string StepCapReportCall =
            $"sharpemu_note_step_cap({ResourcesName}.{Gen5MslArgumentLayout.StepReportName});";

        // SHARPEMU_MSL_STEPCAP_PROBE=s40,v41,...: a capped invocation also stores up to seven
        // named registers (or `exec`) into the words after its step-report counter, so a loop
        // that never exits says which value kept it going. SHARPEMU_MSL_STEPCAP_PROBE_HASH
        // limits it to one program, since a register another program never names would not
        // compile there.
        private static readonly string[] StepCapProbeRegisters =
            (Environment.GetEnvironmentVariable("SHARPEMU_MSL_STEPCAP_PROBE") ?? string.Empty)
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Take(7)
                .ToArray();

        private static readonly ulong StepCapProbeHash =
            ulong.TryParse(
                (Environment.GetEnvironmentVariable("SHARPEMU_MSL_STEPCAP_PROBE_HASH") ?? string.Empty)
                    .Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                System.Globalization.NumberStyles.HexNumber,
                System.Globalization.CultureInfo.InvariantCulture,
                out var probeHash)
                ? probeHash
                : 0;

        // SHARPEMU_MSL_PC_PROBE=<hash>:<pc>:<nth>:<reg>/<reg>/...[:<condition>]: when an active lane of that program
        // reaches the instruction at pc, it counts itself in the program's step-report word, and the
        // nth one stores up to seven registers (state before the instruction) into the probe words,
        // which the host prints as a [METAL-V2][STEP-CAP] line. The optional condition is an MSL
        // expression over the register names (as_type<float>(v40) <= 0.0f) that a lane must also
        // meet. With nth = log, every qualifying lane (up to PcProbeLogRecords per flush) stores up
        // to 24 registers as one record, which the host appends to SHARPEMU_MSL_PC_PROBE_LOG. A
        // diagnostic: it reads lanes' values without touching the program's results.
        // Several probes may be given, separated by ';' (one per program): the host keeps a log region
        // per probed program, so two programs of the same frame (a prepass and its base pass) can be
        // compared record for record.
        private static readonly (ulong Hash, uint Pc, uint Nth, string[] Registers, string? Condition)[] PcProbes =
            (Environment.GetEnvironmentVariable("SHARPEMU_MSL_PC_PROBE") ?? string.Empty)
                .Split(';', StringSplitOptions.RemoveEmptyEntries)
                .Select(ParsePcProbe)
                .Where(static probe => probe is not null)
                .Select(static probe => probe!.Value)
                .ToArray();

        private static (ulong Hash, uint Pc, uint Nth, string[] Registers, string? Condition)? ParsePcProbe(string? value)
        {
            var parts = (value ?? string.Empty).Split(':');
            if (parts.Length is not (4 or 5) ||
                !ulong.TryParse(parts[0].Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                    System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var hash) ||
                !uint.TryParse(parts[1].Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                    System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var pc) ||
                !(uint.TryParse(parts[2], out var nth) || parts[2] == "log"))
            {
                return null;
            }

            var log = parts[2] == "log";
            return (hash, pc, log ? uint.MaxValue : nth,
                parts[3].Split('/', StringSplitOptions.RemoveEmptyEntries).Take(log ? 24 : 7).ToArray(),
                parts.Length == 5 ? parts[4] : null);
        }

        /// <summary>
        /// The words of one log record of a probed program: the index of the probe that wrote it among
        /// all the probes given, then the registers, padded to the largest probe of that program (all
        /// the log probes of a program share its report region, so several PCs of one program stay apart).
        /// </summary>
        internal static int ProbeRecordWords(ulong hash) =>
            1 + PcProbes.Where(probe => probe.Hash == hash && probe.Nth == uint.MaxValue)
                .Select(static probe => probe.Registers.Length).DefaultIfEmpty(0).Max();

        private string PcProbeCode((ulong Hash, uint Pc, uint Nth, string[] Registers, string? Condition) probe)
        {
            var report = $"{ResourcesName}.{Gen5MslArgumentLayout.StepReportName}";
            var condition = probe.Condition is null ? string.Empty : $" && ({probe.Condition})";
            var log = probe.Nth == uint.MaxValue;
            var recordWords = ProbeRecordWords(probe.Hash);
            var text = new StringBuilder($"if (exec{condition} && {report} != nullptr) {{ uint sharpemu_probe_n = atomic_fetch_add_explicit({report}, 1u, memory_order_relaxed); ");
            text.Append(log
                ? $"if (sharpemu_probe_n < {PcProbeLogRecords}u) {{ atomic_store_explicit({report} + 1u + sharpemu_probe_n * {recordWords}u, {Array.IndexOf(PcProbes, probe)}u, memory_order_relaxed);"
                : $"if (sharpemu_probe_n == {probe.Nth}u) {{");
            for (var index = 0; index < probe.Registers.Length; index++)
            {
                var name = probe.Registers[index];
                var value = name == "exec" ? "(exec ? 1u : 0u)" : $"as_type<uint>({name})";
                var word = log ? $"1u + sharpemu_probe_n * {recordWords}u + {index + 1}u" : $"{index + 1}";
                text.Append($" atomic_store_explicit({report} + {word}, {value}, memory_order_relaxed);");
            }

            text.Append(" } }");
            return text.ToString();
        }

        private string StepCapReportCode
        {
            get
            {
                if (StepCapProbeRegisters.Length == 0 || (StepCapProbeHash != 0 && StepCapProbeHash != _request.Hash))
                {
                    return StepCapReportCall;
                }

                var text = new StringBuilder(StepCapReportCall);
                text.Append($" if ({ResourcesName}.{Gen5MslArgumentLayout.StepReportName} != nullptr) {{");
                for (var index = 0; index < StepCapProbeRegisters.Length; index++)
                {
                    var name = StepCapProbeRegisters[index];
                    var value = name == "exec" ? "(exec ? 1u : 0u)" : $"as_type<uint>({name})";
                    text.Append($" atomic_store_explicit({ResourcesName}.{Gen5MslArgumentLayout.StepReportName} + {index + 1}, {value}, memory_order_relaxed);");
                }

                text.Append(" }");
                return text.ToString();
            }
        }

        private readonly Gen5MslStage _stage;
        private readonly uint _localSizeX;
        private readonly uint _localSizeY;
        private readonly uint _localSizeZ;
        private readonly uint _waveLaneCount;
        private readonly IReadOnlyList<Gen5PixelOutputBinding> _pixelOutputBindings;
        private readonly bool _usesPixelValidMask;

        // The EXP.VM instructions whose EXEC is the restored entry EXEC (see
        // Gen5MslTranslator.EntryExec.cs), and whether any other one published EXEC.
        private HashSet<uint>? _entryMaskExports;
        private bool _validMaskFromExec;
        private readonly bool _pixelDepthExport;
        private readonly bool _pixelStencilExport;
        private readonly bool _pixelSampleMaskExport;
        private readonly Gen5ColorExportFormat? _mrtzFormat;

        // A translator-side probe, off unless someone is hunting a draw that covers
        // nothing; see the vertex epilogue.
        // SHARPEMU_MSL_DEBUG_VS_WORD=<n>: publish buffer n's byte count and byte bias
        // through the step-report word, which the host prints after every flush.
        private static readonly int DebugVertexWord =
            int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_MSL_DEBUG_VS_WORD"), out var watched) &&
            watched >= 0
                ? watched
                : -1;

        // Narrows the debug switches to these programs (comma-separated), so what they report is not
        // mixed with every other shader in the frame; unset (or nothing parsable) means every program.
        private static readonly HashSet<ulong> DebugHashes =
            (Environment.GetEnvironmentVariable("SHARPEMU_MSL_DEBUG_HASH") ?? string.Empty)
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(static text => ulong.TryParse(
                    text.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ? text[2..] : text,
                    System.Globalization.NumberStyles.HexNumber,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out var hash) ? hash : 0UL)
                .Where(static hash => hash != 0)
                .ToHashSet();

        private bool DebugsThisProgram => DebugHashes.Count == 0 || DebugHashes.Contains(_request.Hash);

        private static readonly bool DebugVertexPosition =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_DEBUG_VS_POSITION") == "1";

        // SHARPEMU_MSL_DEBUG_NO_DISCARD=1 (diagnostic): a pixel program never discards, whatever its valid mask
        // says (SHARPEMU_MSL_DEBUG_HASH narrows it to one program). Geometry that appears with it was being killed
        // by its own pixel program; geometry still missing never reached the pixel stage.
        internal static readonly bool DebugNoDiscard =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_DEBUG_NO_DISCARD") == "1";
        private readonly uint _pixelInputEnable;
        private readonly uint _pixelInputAddress;
        private readonly uint[] _pixelInputCntl;
        private readonly SortedSet<uint> _pixelAttributes = [];
        private readonly SortedSet<uint> _vertexOutputs = [];
        private readonly Dictionary<uint, ShaderVertexInput> _vertexInputsByPc = [];
        private readonly int _requiredVertexOutputCount;
        private readonly StringBuilder _body = new();
        private int _indent;
        private int _nextTemp;
        private bool _usesLds;
        private uint _scratchDwordCount;
        private bool _usesFormatLoads;
        private int _formatBufferLoads;
        private bool _usesDecodedFormatLoads;
        private bool _usesBufferOffset;
        private bool _usesCoherentLoads;
        private bool _usesPairLocks;
        // An F64 atomic (FCMPSWAP_X2, FMIN/FMAX_X2, the DS F64 forms) compares on the
        // software binary64 helpers, so the program carries float64_prelude.
        private bool _usesFloat64Atomics;
        private bool _usesShaderClock;
        private bool _usesWaveScratch;

        /// <summary>True when this stage emulates a 64-lane guest wave across two
        /// 32-wide Apple simdgroups (compute only; graphics stages run a wave as one
        /// simdgroup or a single lane regardless of guest wave size, and the host
        /// requests wave32 for them, as it does for the SPIR-V translator).
        /// Only a one-wave (64-thread) workgroup is bridged, the SPIR-V translator's
        /// scope: the bridge's barriers are threadgroup-wide, and the waves of a larger
        /// workgroup take their own trip counts through the wave-uniform loops that hold
        /// them, so a barrier inside one would never be met by the other waves and the
        /// GPU hangs. Larger workgroups run each 32-wide simdgroup as a wave, unless the
        /// program combines the halves itself (<see cref="IsBridgedComputeWaves"/>).</summary>
        private bool IsWave64 =>
            (_waveLaneCount == 64 && _stage == Gen5MslStage.Compute &&
                ((ulong)_localSizeX * _localSizeY * _localSizeZ == 64 || IsBridgedComputeWaves)) ||
            IsBridgedMeshWave64;

        /// <summary>A compute workgroup of several guest wave64s whose program reads a lane of
        /// the upper half by number (V_READLANE of a constant lane 32..63), so it combines the
        /// halves itself and 32-wide simdgroups run as waves get it wrong. Ghost of Tsushima's
        /// tiled light culler (csB52D742B, 16x16 groups) totals each tile's lights over a wave
        /// from lanes 31 and 63 and places them with V_MBCNT over a 64-bit VCC: as eight wave32s
        /// the total doubled and each half wrote its lights over the other's, so the next list
        /// started past slots nothing wrote, the lighting read stale entries there, and the fire
        /// light came and went in 16x16 blocks every frame. Such waves are bridged like the mesh
        /// path's: slots of their own and a pair barrier for each exchange, while the guest
        /// S_BARRIER stays a threadgroup barrier. Other programs keep one simdgroup a wave, whose
        /// exchanges cost nothing.</summary>
        private readonly bool _bridgedComputeWaves;

        private bool IsBridgedComputeWaves => _bridgedComputeWaves;

        /// <summary>Guest waves in a bridged compute workgroup.</summary>
        private uint BridgedComputeWaves => _localSizeX * _localSizeY * _localSizeZ / 64u;

        // SHARPEMU_BRIDGE_COMPUTE_WAVES=0 runs every multi-wave compute workgroup as wave32s, as before.
        private static readonly bool BridgeComputeWavesEnabled =
            Environment.GetEnvironmentVariable("SHARPEMU_BRIDGE_COMPUTE_WAVES") != "0";

        private static bool BridgesComputeWaves(ShaderCompileRequest request, Gen5MslStage stage, uint waveLanes,
            ulong threads, uint ldsDwords)
        {
            if (!BridgeComputeWavesEnabled || stage != Gen5MslStage.Compute || waveLanes != 64 || threads <= 64 ||
                threads % 64 != 0)
            {
                return false;
            }

            // Each wave's slots and pair counter are threadgroup memory beside the dispatch's LDS.
            var usesLds = request.Program.Instructions.Any(instruction =>
                instruction.Control is Gen5DataShareControl { Gds: false } or Gen5BufferMemoryControl { Lds: true });
            if ((usesLds ? ldsDwords : 0u) + (WaveScratchDwords + 1u) * (threads / 64) > LdsDwordCount)
            {
                return false;
            }

            return request.Program.Instructions.Any(instruction =>
                instruction.Opcode == "VReadlaneB32" && instruction.Sources.Count > 1 &&
                instruction.Sources[1] is var lane &&
                ((lane.Kind == Gen5OperandKind.EncodedConstant && lane.Value is >= 160 and <= 191) ||
                    (lane.Kind == Gen5OperandKind.LiteralConstant && (lane.Value & 63u) >= 32)));
        }

        /// <summary>An amplified wave64 mesh subgroup (Prospero's mesh shader): its threads are
        /// numbered (merged-wave subgroup id &lt;&lt; 6) + lane and each exports its own vertex and
        /// primitive, so a guest wave has to be all 64 lanes - two bridged simdgroups, several
        /// waves per threadgroup. Their exchanges meet at a pair barrier of the wave's two
        /// simdgroups instead of a threadgroup barrier, since the other waves need not be at
        /// the same exchange (see <see cref="Wave64Barrier"/>).</summary>
        private bool IsBridgedMeshWave64 => _stage == Gen5MslStage.Mesh && _meshInfo is { GuestWaveLanes: 64 };

        /// <summary>Where the two halves of a bridged wave64 meet: a threadgroup barrier when
        /// the wave is the whole threadgroup (compute), else the wave's own pair barrier. The
        /// pair barrier fences this half's threadgroup writes, counts it in, and waits for the
        /// other half to count in for the same exchange; both halves run the same sequence of
        /// exchanges (their scalar state is the same), so each waits only for its partner.
        /// The wait gives up after <see cref="PairBarrierSpinLimit"/> polls and reports a step
        /// cap: a half whose partner never comes then runs on with stale exchanges instead of
        /// spinning until the GPU watchdog discards the whole command buffer.</summary>
        private string Wave64Barrier => IsBridgedMeshWave64 || IsBridgedComputeWaves
            ? "{ simdgroup_barrier(mem_flags::mem_threadgroup); sharpemu_wave_gen += 2u; " +
              "if ((sharpemu_lane & 31u) == 0u) { atomic_fetch_add_explicit(&sharpemu_wave_pair[sharpemu_tg_index >> 6u], 1u, memory_order_relaxed); } " +
              "for (uint sharpemu_pair_spins = 0u; atomic_load_explicit(&sharpemu_wave_pair[sharpemu_tg_index >> 6u], memory_order_relaxed) < sharpemu_wave_gen;) " +
              $"{{ if (++sharpemu_pair_spins == {PairBarrierSpinLimit}u) {{ {StepCapReportCall} break; }} }} " +
              "simdgroup_barrier(mem_flags::mem_threadgroup); }"
            : "threadgroup_barrier(mem_flags::mem_threadgroup);";

        // Polls of the partner's count before a pair barrier gives up: milliseconds, where a
        // partner that is coming arrives within one block of guest code.
        private const uint PairBarrierSpinLimit = 1u << 20;

        /// <summary>A guest wave32 program: VCC is VCC_LO alone, so a mask write (V_CMP,
        /// a carry-out) leaves s107 untouched and VCCZ reads s106 only (EXECZ likewise reads
        /// EXEC_LO only). In wave32, VCC_HI is an ordinary SGPR the compiler allocates: GTA V's
        /// BVH rotation pass parks `s_movk_i32 s107, 0x204` before a V_CMP to VCC and reads it
        /// after (and keeps a saved EXEC there), and zeroing it there broke the tree it rotates.
        /// The guest wave size comes from the dispatch initiator's wave32 bit for compute,
        /// VGT_SHADER_STAGES_EN.GS_W32_EN for an NGG primitive shader and SPI_PS_IN_CONTROL.
        /// PS_W32_EN for a pixel shader; a legacy vertex shader's is not carried, and its high
        /// VCC half, which has no lanes, stays clear.</summary>
        private bool VccIsLowHalfOnly => _stage switch
        {
            Gen5MslStage.Compute => _waveLaneCount == 32,
            Gen5MslStage.Mesh or Gen5MslStage.Vertex => _request.PrimitiveShader is { IsWave64: false },
            Gen5MslStage.Pixel => _request.PixelGuestWave32,
            _ => false,
        };

        /// <summary>The VCCZ status bit: the whole pair, or VCC_LO alone in wave32.</summary>
        private string VccZeroCondition => VccIsLowHalfOnly
            ? $"s[{VccLoRegister}] == 0u"
            : $"(s[{VccLoRegister}] | s[{VccHiRegister}]) == 0u";

        /// <summary>True for the stages whose Metal threads map one-to-one onto guest wave
        /// lanes, so ballots, shuffles and lane indices are real: compute dispatches, NGG
        /// primitive shaders emitted as mesh functions, and pixel shaders when the host
        /// enables graphics subgroup operations.
        ///
        /// A pixel shader's guest wave32 is the fragment's 32-wide Apple SIMD-group, the
        /// one-subgroup-per-wave mapping the SPIR-V translator uses for graphics stages.
        /// Apple packs fragments into it as 2x2 quads at lanes 4n..4n+3 ordered top-left,
        /// top-right, bottom-left, bottom-right ([[thread_index_in_quadgroup]] is
        /// lane &amp; 3), which is the GCN pixel-quad layout that DPP quad_perm and S_WQM
        /// assume, so guest lanes need no remapping.
        ///
        /// The vertex stage stays a single logical lane. MSL documents SIMD-group
        /// functions only for kernel and fragment functions (MSL spec 6.10) and gives a
        /// vertex function no [[thread_index_in_simdgroup]], so there is no supported
        /// lane index to shuffle or ballot against.</summary>
        private bool HasRealWaveLanes =>
            _stage is Gen5MslStage.Compute or Gen5MslStage.Mesh ||
            (_stage == Gen5MslStage.Pixel && _request.EnableGraphicsSubgroupOperations);

        /// <summary>True where guest LDS is threadgroup memory the wave's lanes share.
        /// Every other stage keeps it as per-invocation scratch, real lanes or not: a
        /// fragment function has no threadgroup memory to put it in.</summary>
        private bool HasSharedLds => _stage is Gen5MslStage.Compute or Gen5MslStage.Mesh;

        public bool TryCompile(out Gen5MslShader shader, out string error)
        {
            shader = default!;
            error = string.Empty;
            try
            {
                // A 64-lane guest wave that uses cross-lane ops needs the
                // threadgroup-scratch bridge (ballots span both 32-wide halves,
                // read-first-lane broadcasts across them). Programs without such
                // ops are wave-size-agnostic and need no scratch.
                _usesWaveScratch = IsWave64 && UsesWaveSensitiveOperations();

                // The mesh stage parks a wave at S_BARRIER and resumes it from the next
                // block, so a barrier has to end its block (see Gen5MslTranslator.Mesh.cs).
                var blocks = BuildBasicBlocks(
                    _request.Program.Instructions,
                    splitAfterBarrier: _stage == Gen5MslStage.Mesh);
                if (blocks.Count == 0)
                {
                    error = "shader contains no executable blocks";
                    return false;
                }

                {
                    DeclareLayoutBindings();
                }

                foreach (var instruction in _request.Program.Instructions)
                {
                    _usesLds |= instruction.Control is Gen5DataShareControl { Gds: false } &&
                        instruction.Opcode is not ("DsSwizzleB32" or "DsBpermuteB32" or "DsPermuteB32" or "DsNop");
                    _usesLds |= instruction.Control is Gen5BufferMemoryControl { Lds: true };
                    if (instruction.Opcode.StartsWith("Scratch", StringComparison.Ordinal))
                    {
                        _scratchDwordCount = Math.Max(_request.ScratchDwords, 1u);
                    }

                    _usesFormatLoads |= IsFormatBufferLoad(instruction.Opcode) ||
                        IsFormatBufferStore(instruction.Opcode);
                    if (instruction.Opcode.StartsWith("BufferLoadFormat", StringComparison.Ordinal))
                    {
                        _formatBufferLoads++;
                    }
                    _usesShaderClock |= instruction.Opcode is "SMemtime" or "SMemrealtime";
                    if (instruction.Control is Gen5InterpolationControl interpolationControl)
                    {
                        _pixelAttributes.Add(interpolationControl.Attribute);
                    }

                    if (_stage is Gen5MslStage.Vertex or Gen5MslStage.Mesh &&
                        instruction.Control is Gen5ExportControl { Target: >= 32 and < 64 } vertexExport)
                    {
                        _vertexOutputs.Add(vertexExport.Target - 32);
                    }
                }

                // A FLAT access whose address can land in an aperture needs the
                // storage that aperture names, even when no DS or SCRATCH opcode
                // in the program asked for it.
                foreach (var memory in _request.Memory.Entries)
                {
                    _usesLds |= memory.AddressSpace
                        is FlatAddressSpace.Shared or FlatAddressSpace.SharedOrPrivate;
                    if (memory.AddressSpace is FlatAddressSpace.Private or FlatAddressSpace.SharedOrPrivate)
                    {
                        _scratchDwordCount = Math.Max(_request.ScratchDwords, 1u);
                    }
                }

                _staticScratchSlots = CanUseStaticScratchSlots();
                AssignLaneSlotLds(blocks);

                if (_stage is Gen5MslStage.Vertex or Gen5MslStage.Mesh)
                {
                    DeclareAuxPositionOutputs();
                }

                if (_stage == Gen5MslStage.Pixel)
                {
                    ClassifyPixelInputs();
                    if (_manualInterpolationFailure is { } failure)
                    {
                        error = $"no manual interpolation plan: {failure}";
                        return false;
                    }
                }

                if (_stage == Gen5MslStage.Mesh)
                {
                    // The paired pixel program is translated against the same param
                    // interface as the vertex path, so cover every location it reads.
                    for (uint location = 0; location < _requiredVertexOutputCount; location++)
                    {
                        _vertexOutputs.Add(location);
                    }

                    // The mesh entry has no [[stage_in]]: a primitive shader fetches its
                    // own attributes through buffer loads, which the resource path
                    // translates unchanged.
                }

                if (_stage == Gen5MslStage.Vertex)
                {
                    // Cover every location the paired fragment shader reads,
                    // even ones this vertex program never exports, so Metal's
                    // exact vertex-out/fragment-in interface match succeeds.
                    // Extras stay zero-filled.
                    for (uint location = 0; location < _requiredVertexOutputCount; location++)
                    {
                        _vertexOutputs.Add(location);
                    }

                    foreach (var input in _request.VertexInputs)
                    {
                        if (input.ComponentCount is >= 1 and <= 4)
                        {
                            _vertexInputsByPc.TryAdd(input.Pc, input);
                            foreach (var aliasPc in input.AliasPcs ?? [])
                            {
                                _vertexInputsByPc.TryAdd(aliasPc, input);
                            }
                        }
                    }
                }

                // Emit the block bodies first: block translation discovers nothing
                // that changes the signature in the compute stage, but keeping the
                // order body-then-wrap matches how the pixel/vertex stages need it
                // (their IO discovery happens during block translation).
                if (!TryEmitBlocks(blocks, out error))
                {
                    return false;
                }

                // Needs every block's text, so it cannot run during emission.
                ResolveMaskBallots(blocks);
                NoteWaveScratchUse();

                var source = new StringBuilder();
                EmitModule(source, blocks);
                shader = new Gen5MslShader(
                    source.ToString(),
                    EntryPointName,
                    _stage,
                    AttributeCount: _stage switch
                    {
                        Gen5MslStage.Pixel => (uint)_pixelAttributes.Count,
                        Gen5MslStage.Vertex or Gen5MslStage.Mesh => (uint)_vertexOutputs.Count,
                        _ => 0,
                    },
                    _stage == Gen5MslStage.Mesh ? Mesh.ThreadsPerThreadgroup : _localSizeX,
                    _localSizeY,
                    _localSizeZ,
                    ArgumentLayout: _argumentLayout,
                    MeshInfo: _meshInfo,
                    WritesLayer: _writesLayer,
                    WritesViewportIndex: _writesViewportIndex)
                {
                    UsesPerVertexInputs = UsesVertexValueInputs,
                    UsesMipStatisticsConstant = UsesMipStatisticsConstant,
                    ConstantPositions = _stage == Gen5MslStage.Vertex && !DebugVertexPosition && _request.Tessellation is null
                        ? EvaluateConstantPositions(_request.Program.Instructions, _request.ClipSpace)
                        : null,
                    CannotDiscard = _stage == Gen5MslStage.Pixel && _usesPixelValidMask && !_validMaskFromExec,
                    ColorOutputMask = _stage == Gen5MslStage.Pixel ? ColorOutputLocations(_pixelOutputBindings) : 0,
                    PersistentThreads = IsPersistentThreadsKernel(),
                    Tessellation = _request.Tessellation,
                    FlattenedTableMinimumWords = _flattenedTableMinimumWords,
                };
                return true;
            }
            catch (Exception exception)
            {
                error = exception.Message;
                return false;
            }
        }

        /// <summary>Mirrors the SPIR-V translator's subgroup-usage predicates:
        /// the ops whose results depend on the wave width or on other lanes.
        /// A program without them is wave-size-agnostic.</summary>
        private bool UsesWaveSensitiveOperations()
        {
            foreach (var instruction in _request.Program.Instructions)
            {
                if (instruction.Control is Gen5DppControl or Gen5Dpp8Control ||
                    instruction.Opcode is "VPermlane16B32" or "VPermlanex16B32"
                        or "VReadlaneB32" or "VReadfirstlaneB32"
                        or "VMbcntLoU32B32" or "VMbcntHiU32B32"
                        or "DsAppend" or "DsConsume" ||
                    instruction.Opcode.Contains("Saveexec", StringComparison.Ordinal) ||
                    instruction.Opcode.StartsWith("SCbranchExec", StringComparison.Ordinal) ||
                    instruction.Opcode.StartsWith("SCbranchVcc", StringComparison.Ordinal) ||
                    instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal))
                {
                    return true;
                }

                foreach (var operand in instruction.Sources)
                {
                    if (IsWaveMaskOperand(operand))
                    {
                        return true;
                    }
                }

                foreach (var operand in instruction.Destinations)
                {
                    if (IsWaveMaskOperand(operand))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        private static bool IsWaveMaskOperand(Gen5Operand operand) =>
            operand.Kind == Gen5OperandKind.ScalarRegister &&
            operand.Value is VccLoRegister or VccHiRegister or ExecLoRegister or ExecHiRegister;

        /// <summary>Rewrites the trailing comma of the last emitted parameter
        /// line into the closing parenthesis. Every stage emits each entry
        /// parameter with a trailing comma so optional parameters never need
        /// to know whether they are last.</summary>
        private static void CloseParameterList(StringBuilder source)
        {
            var index = source.Length - 1;
            while (index >= 0 && (source[index] == '\n' || source[index] == '\r'))
            {
                index--;
            }

            if (index >= 0 && source[index] == ',')
            {
                source.Remove(index, source.Length - index);
                source.AppendLine(")");
            }
        }

        private string EntryPointName => _stage switch
        {
            Gen5MslStage.Vertex => "gen5_vs",
            Gen5MslStage.Pixel => "gen5_ps",
            Gen5MslStage.Mesh => "gen5_ms",
            _ => "gen5_cs",
        };

        /// <summary>The workgroup size of a compute kernel as its thread limit
        /// (<see cref="Gen5MslCodegenOptions.ThreadgroupSizeHint"/>). The host dispatches the
        /// threadgroup size the shader reports (<see cref="Gen5MslShader.ThreadgroupSizeX"/>),
        /// except for a one-thread size, which it may replace with the dispatch's, so that one
        /// is not declared.</summary>
        private string ThreadgroupSizeAttribute()
        {
            var threads = (ulong)_localSizeX * _localSizeY * _localSizeZ;
            return _options.ThreadgroupSizeHint && _stage == Gen5MslStage.Compute && threads is > 1 and <= 1024
                ? $"[[max_total_threads_per_threadgroup({threads})]] "
                : string.Empty;
        }

        private void EmitModule(StringBuilder source, IReadOnlyList<ShaderBlock> blocks)
        {
            source.AppendLine("// Generated by SharpEmu Gen5MslTranslator.");
            source.AppendLine("#include <metal_stdlib>");
            source.AppendLine();
            source.AppendLine("using namespace metal;");
            source.AppendLine();
            // Every guest float op rounds on its own: GCN fuses a multiply and an add only
            // where the program says V_FMA, and its legacy V_MAD/V_MAC round the product
            // first. Letting the compiler contract a*b+c makes one program compute a value a
            // different way than another program that spells it the same, which a depth
            // EQUAL pass against a prepass cannot survive.
            // Gen5MslCodegenOptions.ContractPixelCompute lifts it where no position or depth
            // comes out of the program.
            if (!(_options.ContractPixelCompute &&
                  ((_stage == Gen5MslStage.Compute && _request.Tessellation is null) ||
                   (_stage == Gen5MslStage.Pixel && !_pixelDepthExport))))
            {
                source.AppendLine("#pragma clang fp contract(off)");
                source.AppendLine();
            }

            // Uniforms: dispatch bounds plus per-buffer byte lengths. Metal has
            // no OpArrayLength equivalent, so buffer extents travel with the
            // dispatch instead of being queried in-shader.
            var preludes = ModulePreludes();
            AppendLeadingPreludes(source, preludes);

            // A request binds one argument buffer per stage; its lengths replace the uniforms.
            EmitResourceStruct(source);

            AppendTrailingPreludes(source, preludes, _options.BranchlessDwordLoads);
            if (UsesMipStatisticsConstant)
            {
                source.AppendLine(
                    $"constant bool {Gen5MslShader.MipStatisticsFunctionConstantName} " +
                    $"[[function_constant({Gen5MslShader.MipStatisticsFunctionConstantIndex})]];");
            }

            source.AppendLine();

            if (_stage == Gen5MslStage.Mesh)
            {
                // The mesh vertex payload, the merged ES/GS launch block and the mesh
                // output type.
                EmitMeshTypes(source);
                EmitMeshSignature(source);
            }
            else if (_stage == Gen5MslStage.Vertex)
            {
                // Stage IO structs: fetched attributes from the evaluated vertex
                // inputs (bound via MTLVertexDescriptor), position plus the
                // param outputs the paired fragment shader reads.
                if (_vertexInputsByPc.Count != 0)
                {
                    source.AppendLine("struct Gen5VsIn");
                    source.AppendLine("{");
                    var declared = new HashSet<uint>();
                    foreach (var input in _vertexInputsByPc.Values)
                    {
                        if (!declared.Add(input.Location))
                        {
                            continue;
                        }

                        var componentType = input.NumberFormat switch
                        {
                            4u => "uint",
                            5u => "int",
                            _ => "float",
                        };
                        var fieldType = input.ComponentCount == 1
                            ? componentType
                            : $"{componentType}{input.ComponentCount}";
                        source.AppendLine(
                            $"    {fieldType} in{input.Location} [[attribute({input.Location})]];");
                    }

                    source.AppendLine("};");
                    source.AppendLine();
                }

                source.AppendLine("struct Gen5VsOut");
                source.AppendLine("{");
                // Invariant, so two programs that compute a position the same way get the same
                // bits: UE's base pass draws with depth EQUAL against its depth prepass, and a
                // position the compiler fused or scheduled differently in one of them fails
                // the test on almost every pixel (the mesh never shows). Takes effect with
                // MTLCompileOptions.preserveInvariance, which the host sets.
                source.AppendLine("    float4 sharpemu_position [[position, invariant]];");
                EmitAuxPositionFields(source, perVertexLayer: true);
                foreach (var location in _vertexOutputs)
                {
                    source.AppendLine($"    float4 param{location} [[user(locn{location})]];");
                }

                EmitInterpolantOutputMembers(source);
                source.AppendLine("};");
                source.AppendLine();
                if (_request.Tessellation is { Role: Gen5TessellationRole.Domain } tessellator)
                {
                    if (_vertexInputsByPc.Count != 0)
                    {
                        throw new NotSupportedException("A domain program that fetches vertex attributes is not supported.");
                    }

                    // The tessellator's state is pipeline state; it is written into the source so
                    // two domain programs that differ only in it never share a library or a state.
                    source.AppendLine(
                        $"// tessellator: partition={tessellator.Partition} " +
                        $"output={(tessellator.OutputClockwise ? "cw" : "ccw")} max={tessellator.MaxFactor}");
                    source.AppendLine($"[[patch({(tessellator.Quad ? "quad" : "triangle")}, {tessellator.ControlPoints})]]");
                }

                source.AppendLine($"vertex Gen5VsOut {EntryPointName}(");
                if (_vertexInputsByPc.Count != 0)
                {
                    source.AppendLine("    Gen5VsIn sharpemu_vin [[stage_in]],");
                }
            }
            else if (_stage == Gen5MslStage.Pixel)
            {
                // Stage IO structs: interpolated attributes discovered from the
                // program's V_INTERP controls, MRT outputs from the bindings.
                source.AppendLine("struct Gen5PsIn");
                source.AppendLine("{");
                source.AppendLine("    float4 sharpemu_frag_coord [[position]];");
                EmitPixelInputMembers(source);
                source.AppendLine("};");
                source.AppendLine();
                source.AppendLine("struct Gen5PsOut");
                source.AppendLine("{");
                foreach (var binding in _pixelOutputBindings)
                {
                    var fieldType = binding.Kind switch
                    {
                        Gen5PixelOutputKind.Uint => "uint4",
                        Gen5PixelOutputKind.Sint => "int4",
                        _ => "float4",
                    };
                    source.AppendLine(
                        $"    {fieldType} mrt{binding.GuestSlot} [[color({binding.HostLocation}){(binding.BlendSourceIndex != 0 ? $", index({binding.BlendSourceIndex})" : string.Empty)}]];");
                }

                if (_pixelDepthExport)
                {
                    source.AppendLine("    float sharpemu_depth [[depth(any)]];");
                }

                if (_pixelStencilExport)
                {
                    source.AppendLine("    uint sharpemu_stencil [[stencil]];");
                }

                if (_pixelSampleMaskExport)
                {
                    source.AppendLine("    uint sharpemu_coverage [[sample_mask]];");
                }

                source.AppendLine("};");
                source.AppendLine();
                // The guest asked for the depth test before the shader and nothing the shader does
                // can change its outcome (the request clears kill and every export), so the test
                // runs first here too: Metal does that on its own only for a function that
                // writes no memory, and a shader with buffer or image stores would run, and
                // store, for fragments the guest hardware rejects before it starts.
                var earlyFragmentTests = _options.EarlyFragmentTests && _request.PixelEarlyDepth &&
                    !_pixelDepthExport && !_pixelStencilExport && !_pixelSampleMaskExport;
                source.AppendLine($"{(earlyFragmentTests ? "[[early_fragment_tests]] " : string.Empty)}fragment Gen5PsOut {EntryPointName}(");
                source.AppendLine("    Gen5PsIn sharpemu_in [[stage_in]],");
            }
            else if (_stage != Gen5MslStage.Mesh)
            {
                source.AppendLine($"{ThreadgroupSizeAttribute()}kernel void {EntryPointName}(");
            }

            {
                EmitResourceParameters(source);
            }

            if (_stage == Gen5MslStage.Mesh)
            {
                EmitMeshSystemParameters(source);
            }

            if (_stage == Gen5MslStage.Compute)
            {
                source.AppendLine("    uint3 sharpemu_local_id [[thread_position_in_threadgroup]],");
                source.AppendLine("    uint3 sharpemu_group_id [[threadgroup_position_in_grid]],");
            }

            if (_stage == Gen5MslStage.Vertex && _request.Tessellation is { Role: Gen5TessellationRole.Domain } patchDomain)
            {
                source.AppendLine("    uint sharpemu_patch_id [[patch_id]],");
                source.AppendLine($"    {(patchDomain.Quad ? "float2" : "float3")} sharpemu_patch_position [[position_in_patch]],");
                source.AppendLine("    uint sharpemu_instance_id [[instance_id]],");
                source.AppendLine($"    constant uint* sharpemu_domain_launch [[buffer({Gen5TessellationLaunch.DomainLaunchBufferIndex})]],");
            }
            else if (_stage == Gen5MslStage.Vertex)
            {
                source.AppendLine("    uint sharpemu_vertex_id [[vertex_id]],");
                source.AppendLine("    uint sharpemu_instance_id [[instance_id]],");
            }
            else if (_stage == Gen5MslStage.Compute && IsWave64)
            {
                // A 64-lane guest wave is two 32-wide Apple simdgroups. Metal
                // packs a threadgroup's threads into simdgroups in ascending
                // thread_index order, so thread_index_in_threadgroup & 63 is the
                // guest lane and its low bit-5 selects the half. Both halves sit
                // in one threadgroup, so a threadgroup_barrier rendezvous bridges
                // the wave for 64-wide ballots (see EmitWave64Ballot).
                source.AppendLine("    uint sharpemu_tg_index [[thread_index_in_threadgroup]],");
            }
            else if (_stage == Gen5MslStage.Compute || (_stage == Gen5MslStage.Pixel && HasRealWaveLanes))
            {
                // Compute threads and fragments map one-to-one onto guest lanes, so
                // wave ops address the invocation's real simdgroup lane.
                source.AppendLine("    uint sharpemu_lane [[thread_index_in_simdgroup]],");
            }

            if (_stage == Gen5MslStage.Compute && _request.Tessellation is { Role: Gen5TessellationRole.Hull })
            {
                source.AppendLine($"    constant uint* sharpemu_hull_launch [[buffer({Gen5TessellationLaunch.HullLaunchBufferIndex})]],");
                source.AppendLine($"    device const uint* sharpemu_hull_vertices [[buffer({Gen5TessellationLaunch.HullVertexBufferIndex})]],");
            }

            CloseParameterList(source);
            source.AppendLine("{");
            if (IsBridgedComputeWaves)
            {
                // Each guest wave owns both slot sets of its own, at wave * WaveScratchDwords,
                // and a pair-barrier counter its two halves count into, zeroed before any wave
                // reaches an exchange.
                var waves = BridgedComputeWaves;
                source.AppendLine("    const uint sharpemu_lane = sharpemu_tg_index & 63u;");
                source.AppendLine($"    threadgroup uint sharpemu_wave_scratch[{WaveScratchDwords * waves}];");
                source.AppendLine($"    uint sharpemu_wave_set = (sharpemu_tg_index >> 6u) * {WaveScratchDwords}u;");
                source.AppendLine($"    threadgroup atomic_uint sharpemu_wave_pair[{waves}];");
                source.AppendLine("    uint sharpemu_wave_gen = 0u;");
                source.AppendLine($"    if (sharpemu_tg_index < {waves}u) {{ atomic_store_explicit(&sharpemu_wave_pair[sharpemu_tg_index], 0u, memory_order_relaxed); }}");
                source.AppendLine("    threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            else if (_stage == Gen5MslStage.Compute && IsWave64)
            {
                source.AppendLine("    uint sharpemu_lane = sharpemu_tg_index & 63u;");
                if (DeclaresWaveScratch && !AliasesWaveScratchIntoLds)
                {
                    // Two sets of slots, alternated per exchange (see
                    // Gen5MslTranslator.WaveExchange.cs). Indexed only by half,
                    // so correct for a one-wave (64-thread) workgroup — larger
                    // workgroups would need per-wave scratch (matches the SPIR-V
                    // translator's bridge scope).
                    source.AppendLine($"    threadgroup uint sharpemu_wave_scratch[{WaveScratchDwords}];");
                }

                if (DeclaresWaveScratch)
                {
                    source.AppendLine("    uint sharpemu_wave_set = 0u;");
                }

            }
            else if (IsBridgedMeshWave64)
            {
                // Each guest wave owns both slot sets of its own, at wave * WaveScratchDwords,
                // and a pair-barrier counter its two halves count into (zeroed by the mesh
                // declarations' seed block before the program starts).
                var waves = Mesh.WavesPerThreadgroup;
                source.AppendLine("    const uint sharpemu_lane = sharpemu_tg_index & 63u;");
                source.AppendLine($"    threadgroup uint sharpemu_wave_scratch[{WaveScratchDwords * waves}];");
                source.AppendLine($"    uint sharpemu_wave_set = (sharpemu_tg_index >> 6u) * {WaveScratchDwords}u;");
                source.AppendLine($"    threadgroup atomic_uint sharpemu_wave_pair[{waves}];");
                source.AppendLine("    uint sharpemu_wave_gen = 0u;");
            }
            if (!HasRealWaveLanes)
            {
                // The vertex stage, and a pixel stage whose host turned graphics
                // subgroups off, model a single logical wave lane — the SPIR-V
                // translator's no-subgroup fallback. Ballots degrade to bit 0 in the
                // prelude and shuffle-family selects resolve to the lane's own value.
                source.AppendLine("    const uint sharpemu_lane = 0u;");
            }

            if (LaneSpillPlan() is { } spills)
            {
                // The per-thread copies of spilled SGPRs (see Ir.Gen5LaneSpillAnalysis).
                for (var copy = 0; copy < spills.CopyCount; copy++)
                {
                    source.AppendLine($"    uint sharpemu_spill{copy} = 0u;");
                }
            }
            if (_usesLds)
            {
                if (_stage == Gen5MslStage.Mesh)
                {
                    // A primitive shader's LDS is the real thing: its waves exchange
                    // vertex data through it. Sized from the stage's own allocation
                    // because the mesh output payload shares the threadgroup budget.
                    //
                    // Above the device threshold the array is a slot of the shared device
                    // pool instead; the pointer is formed once the slot is acquired.
                    source.AppendLine(UsesDeviceLds
                        ? "    threadgroup uint sharpemu_lds_slot;"
                        : $"    threadgroup uint sharpemu_lds[{_meshLdsDwords}];");
                }
                else if (_stage == Gen5MslStage.Compute)
                {
                    // Guest LDS as workgroup-shared memory, sized from what the
                    // dispatch allocated; the address is masked into bounds like the
                    // SPIR-V translator, which is why it is a power of two.
                    source.AppendLine($"    threadgroup uint sharpemu_lds[{_computeLdsDwords}];");
                }
                else
                {
                    // Graphics stages model LDS as per-invocation scratch (the
                    // SPIR-V translator's Private-array trick), sized smaller
                    // because only write-then-read correctness is needed. It is
                    // left uninitialised like the threadgroup arrays above: a guest
                    // reading LDS it never wrote sees a previous wave's leftovers
                    // on hardware, and zero-filling 8 KB per fragment or vertex
                    // costs more than the shader that uses it.
                    source.AppendLine(_ldsLaneSlots is not null
                        ? $"    thread uint sharpemu_lds[{_ldsLaneSlotCount}];"
                        : $"    thread uint sharpemu_lds[{PrivateLdsDwordCount}];");
                }
            }

            if (_scratchDwordCount != 0)
            {
                // Guest scratch is per-invocation private storage, so it is a
                // thread-local array like the SPIR-V translator's Private one.
                // The dword index wraps into it, so an out-of-range guest
                // address never escapes the allocation. Hardware scratch is
                // uninitialised (it holds whatever the last wave on that slot
                // left), so the array is too rather than being zero-filled on
                // every invocation.
                source.AppendLine($"    thread uint sharpemu_scratch[{_scratchDwordCount}];");
            }

            if (DeclaresWaveScratch && AliasesWaveScratchIntoLds)
            {
                // Only a dispatch that asked for the whole 32 KB has no room left for
                // the bridge, and then it aliases the top dwords of the LDS
                // allocation: a separate array would push total threadgroup memory
                // past Metal's limit and the pipeline would not compile. Guest LDS
                // accesses are bounds-masked into the same allocation, so this trades
                // a rare top-of-LDS collision for a compilable pipeline.
                source.AppendLine(
                    $"    threadgroup uint* sharpemu_wave_scratch = &sharpemu_lds[{_computeLdsDwords - WaveScratchDwords}u];");
            }

            if (_stage == Gen5MslStage.Mesh)
            {
                // The rendezvous counters and the export allocation have to exist before
                // the register file so the launch state can read the allocation.
                EmitMeshDeclarations(source);
            }

            // The launch state and the translated body are assembled before the
            // register declarations because the declarations are sized from them:
            // only the SGPRs and VGPRs this program actually touches get a local.
            var initialState = new StringBuilder();
            EmitInitialState(initialState);

            var initialStateText = initialState.ToString();
            var body = new StringBuilder();
            StringBuilder? structuredBuilder = null;
            var structured = false;
            var meshBody = string.Empty;
            if (_stage == Gen5MslStage.Mesh &&
                _options.StructuredControlFlow && !ProgramHasBarrier() &&
                TryBuildStructuredBody(blocks, out meshBody))
            {
                // A mesh program without S_BARRIER has nothing to rendezvous over, so it
                // nests like any other stage instead of keeping every guest register live
                // across a dispatcher loop.
                structured = true;
                body.Append(meshBody);
            }
            else if (_stage == Gen5MslStage.Mesh &&
                _options.StructuredControlFlow && ProgramHasBarrier() &&
                TryBuildMeshBarrierBody(blocks, out meshBody, out var linearBarrierBody))
            {
                // Every S_BARRIER is workgroup-uniform, so it is emitted in place and the
                // body needs no rendezvous (see Gen5MslTranslator.MeshBarrier.cs).
                structured = !linearBarrierBody;
                body.Append(meshBody);
            }
            else if (_stage == Gen5MslStage.Mesh)
            {
                // A barrier that is not workgroup-uniform keeps the dispatcher: the barrier
                // parks a wave between blocks, which needs a resumable pc (see
                // Gen5MslTranslator.Mesh.cs).
                EmitMeshDispatcher(body, BuildDispatcherSwitchBody(blocks, baseIndent: 3));
            }
            else if (_options.StructuredControlFlow &&
                TryBuildStructuredBody(blocks, out structuredBuilder))
            {
                structured = true;
            }
            else
            {
                body.AppendLine("    while (active)");
                body.AppendLine("    {");
                if (MaxDispatcherSteps > 0)
                {
                    body.AppendLine("        uint from_pc = pc;");
                }

                body.AppendLine("        switch (pc)");
                body.AppendLine("        {");
                body.Append(BuildDispatcherSwitchBody(blocks, baseIndent: 2));
                body.AppendLine("        default:");
                body.AppendLine("            active = false;");
                body.AppendLine("            break;");
                body.AppendLine("        }");
                if (MaxDispatcherSteps > 0)
                {
                    // Blocks are numbered in address order, so every trip around a guest
                    // loop moves pc to the block it left or an earlier one. Charging only
                    // those moves gives the cap the structured form's meaning, one step
                    // per loop iteration. Charging every block made a loop's budget
                    // shrink with its block count, and cut off persistent-thread kernels
                    // (UE5's Nanite culling needs over a million block moves) that wait
                    // on other workgroups for longer than a single loop takes.
                    body.AppendLine($"        if (pc <= from_pc && ++steps >= {MaxDispatcherSteps}u)");
                    body.AppendLine("        {");
                    body.AppendLine("            active = false;");
                    body.AppendLine($"            {StepCapReportCode}");
                    body.AppendLine("        }");
                }

                body.AppendLine("    }");
            }

            if (_stage != Gen5MslStage.Mesh)
            {
                Statistics.NoteControlFlowForm(structured);
            }

            var plan = BuildRegisterPlan(blocks, initialStateText);
            EmitRegisterFile(source, plan, structured);
            if (_stage == Gen5MslStage.Pixel)
            {
                source.AppendLine("    Gen5PsOut sharpemu_out = {};");
                if (_pixelDepthExport)
                {
                    // Lanes that never export MRTZ under EXEC keep the rasterized depth.
                    source.AppendLine("    sharpemu_out.sharpemu_depth = sharpemu_in.sharpemu_frag_coord.z;");
                }

                if (_pixelSampleMaskExport)
                {
                    // Lanes that never export MRTZ keep their whole coverage.
                    source.AppendLine("    sharpemu_out.sharpemu_coverage = 0xFFFFFFFFu;");
                }
            }
            else if (_stage == Gen5MslStage.Vertex)
            {
                // Zero-initialized: param outputs the program never exports
                // stay (0,0,0,0) to satisfy the fragment interface.
                source.AppendLine("    Gen5VsOut sharpemu_out = {};");
            }

            plan.RewriteInto(source, initialStateText);
            source.AppendLine();
            plan.RewriteInto(source, structuredBuilder ?? body);
            if (_stage == Gen5MslStage.Mesh)
            {
                source.AppendLine();
                EmitMeshEpilogue(source);
                source.AppendLine("}");
                return;
            }

            if (_stage == Gen5MslStage.Pixel)
            {
                // EXP.VM publishes EXEC as the pixel-valid mask. EXEC can be
                // restored afterward, so only malformed shaders without VM
                // fall back to the final EXEC value.
                if (!(DebugNoDiscard && DebugsThisProgram))
                {
                    source.AppendLine(
                        _usesPixelValidMask
                            ? "    if (!pixel_valid_mask_active)"
                            : "    if (!exec)");
                    source.AppendLine("    {");
                    source.AppendLine("        discard_fragment();");
                    source.AppendLine("    }");
                }

                source.AppendLine("    return sharpemu_out;");
            }
            else if (_stage == Gen5MslStage.Vertex)
            {
                if (DebugVertexPosition && DebugsThisProgram && _request.Tessellation is null)
                {
                    // SHARPEMU_MSL_DEBUG_VS_POSITION=1 throws the program's own clip
                    // position away and puts every triple of vertices across the whole
                    // of NDC. A draw that still covers nothing with this in place was
                    // never about the vertex arithmetic.
                    source.AppendLine(
                        "    sharpemu_out.sharpemu_position = float4(" +
                        "(float)(sharpemu_vertex_id % 3u) - 0.5f, " +
                        "(float)((sharpemu_vertex_id / 3u) % 3u) - 1.0f, 0.5f, 1.0f);");
                }

                if (DebugVertexWord >= 0 && DebugsThisProgram &&
                    DebugVertexWord < _request.Resources.Info.Buffers.Count)
                {
                    source.AppendLine(
                        $"    if ({ResourcesName}.{Gen5MslArgumentLayout.StepReportName} != nullptr) " +
                        $"atomic_store_explicit({ResourcesName}.{Gen5MslArgumentLayout.StepReportName}, 0x40000000u | " +
                        $"({(_options.ElideByteBias ? "0u" : $"bias[{DebugVertexWord}]")} << 20u) | " +
                        $"({ResourcesName}.buffer_bytes[{DebugVertexWord}] & 0xFFFFFu) | " +
                        $"(b{DebugVertexWord} == nullptr ? 0x80000000u : 0u), memory_order_relaxed);");
                }

                source.AppendLine("    return sharpemu_out;");
            }

            source.AppendLine("}");
        }

        // Shared helpers: MSL allows free functions, so unaligned and subdword
        // access is a byte-pointer cast instead of the manual word-combining
        // the SPIR-V translator inlines at every site. All access is
        // range-checked against the binding's byte length; loads outside the
        // buffer produce zero and stores are dropped, matching the SPIR-V
        // translator's robust-access behavior. The static text lives in
        // Templates/prelude.msl; only the wave-ballot expression varies.
        //
        // A single-lane stage models one logical wave lane (lane 0), so a ballot
        // is just that lane's bit: "value ? 1 : 0" — the SPIR-V translator's
        // no-subgroup fallback. Stages with real lanes mirror its subgroup path:
        // threads map one-to-one onto simdgroup lanes and ballots are real, so
        // masks parked in VCC/EXEC hold each lane's actual bit. That is also what
        // keeps cross-lane ops defined inside the translated control flow (the
        // structured nest or the while(active){switch(pc)} dispatcher): every
        // scalar register, and so every branch and the pc, derives from uniform
        // inputs, ballots and lane broadcasts, so a simdgroup's lanes never take
        // different paths — only per-lane EXEC guards differ, and cross-lane ops
        // are emitted outside them. The two models must not be mixed: a real
        // ballot over a per-lane EXEC bit lets lanes branch apart, and then they
        // corrupt each other's EXEC/VCC reconstruction (whole quads discarded).
        //
        // In a fragment, Metal's running lanes include helper threads (they are
        // active for SIMD-group functions, MSL spec 6.10.2), so the entry EXEC
        // ballot holds them too — the SPIR-V translator's subgroup model, and what
        // keeps the EXEC guard elision's "EXEC starts full" premise true. Helper
        // threads' stores and atomics have no effect and their outputs are
        // dropped, so only a guest ballot of the entry mask can tell.
        private Gen5MslPreludes ModulePreludes()
        {
            var preludes = Gen5MslPreludes.None;
            if (_usesFormatLoads)
            {
                preludes |= Gen5MslPreludes.FormatLoads;
            }

            if (HasRealWaveLanes)
            {
                preludes |= Gen5MslPreludes.RealWaveLanes;
            }

            if (_usesBufferOffset)
            {
                preludes |= Gen5MslPreludes.BufferOffset;
            }

            if (_usesDecodedFormatLoads)
            {
                preludes |= Gen5MslPreludes.DecodedFormatLoads;
            }

            if (_usesCoherentLoads)
            {
                preludes |= Gen5MslPreludes.CoherentLoads;
            }

            if (_usesPairLocks)
            {
                preludes |= Gen5MslPreludes.PairLocks;
            }

            if (UsesFloat64Prelude())
            {
                preludes |= Gen5MslPreludes.Float64;
            }

            if (UsesBvhPrelude())
            {
                preludes |= Gen5MslPreludes.Bvh;
            }

            if (_usesMipStatistics)
            {
                preludes |= Gen5MslPreludes.MipStatistics;
            }

            if (Enumerable.Range(0, _request.Resources.Info.Buffers.Count).Any(IsScalarConstantBuffer))
                preludes |= Gen5MslPreludes.ScalarConstants;

            return preludes;
        }

        // The helpers ahead of the resource struct: the unified-format decoder and the
        // device-address page table.
        internal static void AppendLeadingPreludes(StringBuilder source, Gen5MslPreludes preludes)
        {
            // The decoder (format_load_prelude.msl) reads the format tables.
            if ((preludes & (Gen5MslPreludes.FormatLoads | Gen5MslPreludes.DecodedFormatLoads)) != 0)
            {
                EmitFormatLoadPrelude(source);
            }

            EmitResourcePrelude(source);
        }

        // The helpers after the resource struct. A pair lock reads through the coherent
        // helpers, so it always carries coherent_load_prelude.
        internal static void AppendTrailingPreludes(StringBuilder source, Gen5MslPreludes preludes, bool branchlessDwordLoads)
        {
            var dwordLoads = branchlessDwordLoads ? "dword_loads_branchless" : "dword_loads_checked";
            source.Append(MslTemplates.Render(
                "prelude",
                ("ballot_return", (preludes & Gen5MslPreludes.RealWaveLanes) != 0
                    ? "(uint)(uint64_t)simd_ballot(value)"
                    : "value ? 1u : 0u"),
                ("dword_loads_device", MslTemplates.Render(dwordLoads, ("space", "device")))));
            if ((preludes & Gen5MslPreludes.ScalarConstants) != 0)
            {
                source.Append(MslTemplates.Render(
                    "scalar_constant_prelude",
                    ("dword_loads_constant", MslTemplates.Render(dwordLoads, ("space", "constant")))));
            }

            if ((preludes & Gen5MslPreludes.BufferOffset) != 0)
            {
                source.Append(MslTemplates.Render("buffer_offset_prelude"));
            }

            if ((preludes & Gen5MslPreludes.DecodedFormatLoads) != 0)
            {
                source.Append(MslTemplates.Render("format_load_prelude"));
            }

            if ((preludes & (Gen5MslPreludes.CoherentLoads | Gen5MslPreludes.PairLocks)) != 0)
            {
                source.Append(MslTemplates.Render("coherent_load_prelude"));
            }

            if ((preludes & Gen5MslPreludes.PairLocks) != 0)
            {
                source.Append(MslTemplates.Render("pair_atomic_prelude"));
            }

            if ((preludes & Gen5MslPreludes.Float64) != 0)
            {
                source.Append(MslTemplates.Render("float64_prelude"));
            }

            if ((preludes & Gen5MslPreludes.Bvh) != 0)
            {
                source.Append(MslTemplates.Render("bvh_prelude"));
            }

            if ((preludes & Gen5MslPreludes.MipStatistics) != 0)
            {
                source.Append(MslTemplates.Render("mip_stats_prelude"));
            }
        }

        // The GFX10 unified-format table is baked from the same authoritative
        // decoder descriptor evaluation uses (dataFormat | numberFormat << 8);
        // the descriptor is read at execution time, so decoding stays dynamic —
        // compiled shaders may be reused with new SRDs. The static conversion
        // functions live in Templates/format_prelude.msl.
        private static void EmitFormatLoadPrelude(StringBuilder source)
        {
            var table = new StringBuilder();
            for (uint format = 0; format < 128; format++)
            {
                Gfx10UnifiedFormat.TryDecode(format, out var dataFormat, out var numberFormat);
                if ((format & 15) == 0)
                {
                    if (format != 0)
                    {
                        table.AppendLine();
                    }

                    table.Append("   ");
                }

                table.Append($" 0x{dataFormat | (numberFormat << 8):X}u,");
            }

            // Component layouts, keyed component * 16 + data format (format_prelude.msl).
            var layouts = new uint[64];
            foreach (var (component, format, bytes, bitOffset, bitCount) in Gfx10UnifiedFormat.ComponentLayouts)
            {
                var key = component * 16 + format;
                if (key >= layouts.Length || bytes > 0xFF || bitOffset > 0xFF || bitCount > 32)
                {
                    throw new InvalidOperationException(
                        $"A GFX10 component layout does not fit the packed table: component={component} format={format}.");
                }

                layouts[key] = bytes | (bitOffset << 8) | (bitCount << 16);
            }

            var layoutTable = new StringBuilder();
            for (var key = 0; key < layouts.Length; key++)
            {
                layoutTable.Append(key % 16 == 0 ? (key == 0 ? "   " : Environment.NewLine + "   ") : string.Empty);
                layoutTable.Append($" 0x{layouts[key]:X}u,");
            }

            source.AppendLine(MslTemplates.Render(
                "format_prelude",
                ("format_table", table.ToString()),
                ("layout_table", layoutTable.ToString())));
        }

        private void EmitRegisterFile(StringBuilder source)
        {
            source.AppendLine($"    uint s[{ScalarRegisterFileCount}] = {{}};");
            source.AppendLine($"    uint v[{VectorRegisterFileCount}] = {{}};");
            source.AppendLine("    bool exec = true;");
            if (_usesPixelValidMask)
            {
                source.AppendLine("    bool pixel_valid_mask_active = true;");
            }
            source.AppendLine("    bool vcc = false;");
            source.AppendLine("    bool scc = false;");
            source.AppendLine("    uint pc = 0u;");
            source.AppendLine("    bool active = true;");
            source.AppendLine("    uint steps = 0u;");
        }

        private void EmitInitialState(StringBuilder source)
        {
            {
                EmitLayoutInitialState(source);
            }

            if (_stage == Gen5MslStage.Compute && _request.Tessellation is { Role: Gen5TessellationRole.Hull } hull)
            {
                EmitHullLaunchState(source, hull);
            }
            else if (_stage == Gen5MslStage.Compute)
            {
                source.AppendLine("    v[0] = sharpemu_local_id.x;");
                source.AppendLine("    v[1] = sharpemu_local_id.y;");
                source.AppendLine("    v[2] = sharpemu_local_id.z;");

                // Partial-group guard: lanes whose global id falls outside the
                // guest dispatch stay inactive, matching the SPIR-V bounds
                // check driven by the same uniform.
                source.AppendLine(
                    $"    active = (sharpemu_group_id.x * {_localSizeX}u + sharpemu_local_id.x) < {ComputeThreadLimit(0)}");
                source.AppendLine(
                    $"        && (sharpemu_group_id.y * {_localSizeY}u + sharpemu_local_id.y) < {ComputeThreadLimit(1)}");
                source.AppendLine(
                    $"        && (sharpemu_group_id.z * {_localSizeZ}u + sharpemu_local_id.z) < {ComputeThreadLimit(2)};");

                if (_request.ComputeSystemRegisters is { } registers)
                {
                    EmitComputeSystemRegister(source, registers.WorkGroupXRegister, "sharpemu_group_id.x");
                    EmitComputeSystemRegister(source, registers.WorkGroupYRegister, "sharpemu_group_id.y");
                    EmitComputeSystemRegister(source, registers.WorkGroupZRegister, "sharpemu_group_id.z");
                    if (registers.ThreadGroupSizeRegister is { } sizeRegister &&
                        sizeRegister < ScalarRegisterFileCount)
                    {
                        source.AppendLine(
                            $"    s[{sizeRegister}] = {checked(_localSizeX * _localSizeY * _localSizeZ)}u;");
                    }
                }
            }
            else if (_stage == Gen5MslStage.Pixel)
            {
                EmitPixelInputState(source);
            }
            else if (_stage == Gen5MslStage.Vertex && _request.Tessellation is { Role: Gen5TessellationRole.Domain } domain)
            {
                EmitDomainLaunchState(source, domain);
            }
            else if (_stage == Gen5MslStage.Vertex)
            {
                // Hardware-selected VGPRs for the vertex and instance indices.
                source.AppendLine("    v[5] = sharpemu_vertex_id;");
                source.AppendLine("    v[8] = sharpemu_instance_id;");
            }
            else if (_stage == Gen5MslStage.Mesh)
            {
                EmitMeshInitialState(source);
            }

            // VCC/EXEC live in their architectural SGPRs (see StoreScalar);
            // establish the entry state over whatever the initial-scalar block
            // carried so the register file and the bool views agree from the
            // first instruction.
            source.AppendLine($"    s[{VccLoRegister}] = 0u;");
            source.AppendLine($"    s[{VccHiRegister}] = 0u;");
            if (IsWave64)
            {
                // A bridged wave is exactly one 64-thread threadgroup, and
                // dispatchThreadgroups always launches every thread of it, so at
                // entry (before the dispatcher masks any off) all 64 lanes are
                // active: the mask is a constant and needs no ballot, no bridge and
                // none of its two threadgroup barriers.
                source.AppendLine($"    s[{ExecLoRegister}] = 0xFFFFFFFFu;");
                source.AppendLine($"    s[{ExecHiRegister}] = 0xFFFFFFFFu;");
            }
            else
            {
                source.AppendLine($"    s[{ExecLoRegister}] = sharpemu_ballot(true);");
                source.AppendLine($"    s[{ExecHiRegister}] = 0u;");
            }
        }

        // The merged local/hull wave as the hardware's hull stage starts it (Gen5TessellationLaunch):
        // one wave64 per threadgroup over PatchesPerGroup patches of the draw's instances, instance
        // after instance. s2 is the threadgroup's off-chip offset, s3 the merged wave info (local and
        // hull thread counts, one control point each), s4 its tessellation-factor offset; v0 the patch
        // within its instance, v1 the patch within the group and the control point, v2 the local
        // half's vertex index, v3 its thread within the group (its LDS slot), v5 the instance. s[0:1]
        // and the user data come from the layout like any user data.
        private static void EmitHullLaunchState(StringBuilder source, Gen5TessellationLaunch hull)
        {
            var patches = hull.PatchesPerGroup;
            var points = hull.ControlPoints;
            source.AppendLine("    uint sharpemu_hull_lane = sharpemu_tg_index & 63u;");
            source.AppendLine($"    uint sharpemu_hull_first = sharpemu_group_id.x * {patches}u;");
            source.AppendLine("    uint sharpemu_hull_patches = sharpemu_hull_launch[0] > sharpemu_hull_first");
            source.AppendLine($"        ? min({patches}u, sharpemu_hull_launch[0] - sharpemu_hull_first) : 0u;");
            source.AppendLine($"    uint sharpemu_hull_points = sharpemu_hull_patches * {points}u;");
            source.AppendLine($"    s[2] = sharpemu_group_id.x * {hull.OffchipBlockBytes}u;");
            source.AppendLine("    s[3] = sharpemu_hull_points | (sharpemu_hull_points << 8);");
            source.AppendLine($"    s[4] = sharpemu_hull_first * {hull.FactorBytesPerPatch}u;");
            source.AppendLine($"    uint sharpemu_hull_patch = sharpemu_hull_first + sharpemu_hull_lane / {points}u;");
            source.AppendLine("    uint sharpemu_hull_instance_patches = max(sharpemu_hull_launch[2], 1u);");
            source.AppendLine("    uint sharpemu_hull_instance_patch = sharpemu_hull_patch % sharpemu_hull_instance_patches;");
            source.AppendLine("    v[0] = sharpemu_hull_instance_patch;");
            source.AppendLine($"    v[1] = (sharpemu_hull_lane / {points}u) | ((sharpemu_hull_lane % {points}u) << 8);");
            source.AppendLine("    v[2] = sharpemu_hull_lane < sharpemu_hull_points");
            source.AppendLine($"        ? sharpemu_hull_vertices[sharpemu_hull_instance_patch * {points}u + sharpemu_hull_lane % {points}u] : 0u;");
            source.AppendLine("    v[3] = sharpemu_hull_lane;");
            source.AppendLine("    v[5] = sharpemu_hull_launch[1] + sharpemu_hull_patch / sharpemu_hull_instance_patches;");
            source.AppendLine("    active = true;");
        }

        // The domain program at one point the tessellator made: its u and v in v5/v6 (barycentric, or a quad's), the
        // patch within its hull threadgroup in v7, the patch in v8 and that group's off-chip offset in s4.
        // The hull kernel ran over every instance's patches, instance after instance.
        private static void EmitDomainLaunchState(StringBuilder source, Gen5TessellationLaunch domain)
        {
            var patches = domain.PatchesPerGroup;
            source.AppendLine("    uint sharpemu_domain_patch = (sharpemu_instance_id - sharpemu_domain_launch[1]) * sharpemu_domain_launch[0] + sharpemu_patch_id;");
            source.AppendLine("    v[5] = as_type<uint>(sharpemu_patch_position.x);");
            source.AppendLine("    v[6] = as_type<uint>(sharpemu_patch_position.y);");
            source.AppendLine($"    v[7] = sharpemu_domain_patch % {patches}u;");
            source.AppendLine("    v[8] = sharpemu_patch_id;");
            source.AppendLine($"    s[4] = (sharpemu_domain_patch / {patches}u) * {domain.OffchipBlockBytes}u;");
        }

        private static void EmitComputeSystemRegister(
            StringBuilder source,
            uint? scalarRegister,
            string expression)
        {
            if (scalarRegister is { } register && register < ScalarRegisterFileCount)
            {
                source.AppendLine($"    s[{register}] = {expression};");
            }
        }

        // ---- block bodies ----

        /// <summary>Emits one block's straight-line statements and classifies how it
        /// leaves. The terminator is not emitted into the body: the dispatcher and the
        /// structured assembler express the same edges differently.</summary>
        private bool TryEmitBlockBody(
            IReadOnlyList<ShaderBlock> blocks,
            int blockIndex,
            out BlockTerminator terminator,
            out string error)
        {
            _ldsPhase = LdsPhase.None;
            var emitted = TryEmitBlockBodyCore(blocks, blockIndex, out terminator, out error);
            // Without a flow analysis the phase rule orders only within a block, so the
            // block hands over with no half's LDS access still in flight.
            if (emitted && _ldsPhase != LdsPhase.None && !(terminator.Kind == TerminatorKind.Jump && terminator.TrueTarget == ExitTarget))
            {
                Line(Wave64Barrier);
            }

            return emitted;
        }

        private enum LdsPhase { None, Read, Write }

        private LdsPhase _ldsPhase;
        private bool _ldsOrderingBuilt;
        private IReadOnlySet<uint>? _ldsBarriersBefore;

        /// <summary>The pcs that need a threadgroup barrier before them so one 32-wide half
        /// of a guest wave64 cannot overtake the other's LDS accesses (a guest wave's LDS
        /// ops are in order on hardware, so its kernels use no S_BARRIER). Null when the
        /// ordering is not needed or the program has dynamic control flow. These sit at
        /// statement level between instructions, outside any per-lane exec guard, in blocks
        /// every lane reaches: the branches of a bridged wave64 are wave-uniform.</summary>
        private IReadOnlySet<uint>? LdsBarriersBefore()
        {
            if (!_ldsOrderingBuilt)
            {
                _ldsOrderingBuilt = true;
                if (IsWave64 && _usesLds)
                {
                    var sharedFlatPcs = _request.Memory.Entries
                        .Where(static memory => memory.AddressSpace is FlatAddressSpace.Shared or FlatAddressSpace.SharedOrPrivate)
                        .Select(static memory => memory.Pc)
                        .ToHashSet();
                    _ldsBarriersBefore = Ir.Gen5Wave64SharedMemoryOrdering.BarriersBefore(_request.Program, sharedFlatPcs);
                }
            }

            return _ldsBarriersBefore;
        }

        /// <summary>A compute kernel that drains a shared queue and never reads its workgroup
        /// ID, so its groups are interchangeable.</summary>
        private bool LocksPersistentRecords => _options.PersistentRecordLocks && IsPersistentThreadsKernel();

        private bool? _persistentThreads;

        private bool IsPersistentThreadsKernel() => _persistentThreads ??=
            _stage == Gen5MslStage.Compute &&
            _request.ComputeSystemRegisters is not { WorkGroupXRegister: not null } &&
            _request.ComputeSystemRegisters is not { WorkGroupYRegister: not null } &&
            _request.ComputeSystemRegisters is not { WorkGroupZRegister: not null } &&
            Ir.Gen5PersistentThreadsAnalysis.IsPersistent(_request.Program);

        private bool _laneSpillPlanBuilt;
        private Ir.Gen5LaneSpillPlan? _laneSpillPlan;

        /// <summary>The lane spills whose reloads read a per-thread copy instead of the lane:
        /// for a wave64 compute program an optimization (no cross-half exchange), for a stage
        /// that models a single lane (sharpemu_lane is 0 in every thread) what makes them work
        /// at all - there a writelane to lane 1 lands nowhere and its reload read lane 0's
        /// spill (Black Myth: Wukong's hand base pass placed vertices from the wrong scalar).
        /// Null when there are none, or for wave64 compute when the option is off.</summary>
        private Ir.Gen5LaneSpillPlan? LaneSpillPlan()
        {
            if (!_laneSpillPlanBuilt)
            {
                _laneSpillPlanBuilt = true;
                if (!HasRealWaveLanes)
                {
                    // Which VGPRs the launch defines differs by stage; treating them all as
                    // defined only asks that a writelane reach each reload first.
                    _laneSpillPlan = Ir.Gen5LaneSpillAnalysis.Analyze(_request.Program);
                }
                else if (_options.LaneSpillCopies)
                {
                    // Real lanes too: the parked value is scalar, so the copy is what the lane holds,
                    // and a lane a partial simdgroup (or the inactive half of a narrow wave64) does
                    // not have keeps nothing to shuffle from - the hardware's lanes always exist. A
                    // compute launch loads the thread IDs into v0..v2; any other stage's inputs are
                    // treated as defined.
                    _laneSpillPlan = _stage == Gen5MslStage.Compute
                        ? Ir.Gen5LaneSpillAnalysis.Analyze(_request.Program, firstUndefinedVector: 3)
                        : Ir.Gen5LaneSpillAnalysis.Analyze(_request.Program);
                }
            }

            return _laneSpillPlan;
        }

        private bool ProgramHasBarrier()
        {
            foreach (var instruction in _request.Program.Instructions)
            {
                if (instruction.Opcode == "SBarrier")
                {
                    return true;
                }
            }

            return false;
        }

        private bool TryEmitBlockBodyCore(
            IReadOnlyList<ShaderBlock> blocks,
            int blockIndex,
            out BlockTerminator terminator,
            out string error)
        {
            error = string.Empty;
            terminator = BlockTerminator.Exit("active = false;");
            var block = blocks[blockIndex];
            var instructions = _request.Program.Instructions;
            for (var index = block.StartIndex; index < block.EndIndex; index++)
            {
                var instruction = instructions[index];
                var isTerminator = index == block.EndIndex - 1;
                // Comments cost nothing in the compiled library and are the only way
                // to read a dumped MSL body against the guest program it came from.
                if (Gen5MslDump.Enabled)
                {
                    Line($"// 0x{instruction.Pc:X4} {instruction.Opcode}");
                }

                foreach (var pcProbe in PcProbes)
                {
                    if (pcProbe.Hash == _request.Hash && pcProbe.Pc == instruction.Pc)
                    {
                        Line(PcProbeCode(pcProbe));
                    }
                }

                if (IsWave64 && _usesLds)
                {
                    var barriers = LdsBarriersBefore();
                    if (barriers is not null)
                    {
                        if (barriers.Contains(instruction.Pc))
                        {
                            Line(Wave64Barrier);
                        }
                    }
                    else
                    {
                        var nextPhase = instruction.Control is Gen5DataShareControl { Gds: false }
                            ? instruction.Opcode.StartsWith("DsRead", StringComparison.Ordinal) ? LdsPhase.Read
                            : instruction.Opcode.StartsWith("DsWrite", StringComparison.Ordinal) ? LdsPhase.Write : LdsPhase.None
                            : LdsPhase.None;
                        if (instruction.Opcode == "SBarrier")
                        {
                            _ldsPhase = LdsPhase.None;
                        }

                        if (nextPhase != LdsPhase.None)
                        {
                            if (_ldsPhase != LdsPhase.None && _ldsPhase != nextPhase)
                            {
                                Line(Wave64Barrier);
                            }

                            _ldsPhase = nextPhase;
                        }
                    }
                }

                if (instruction.Opcode == "SEndpgm")
                {
                    terminator = BlockTerminator.Exit("active = false;");
                    return true;
                }

                if (instruction.Opcode == "SBranch")
                {
                    // A branch to (or past) the program's end is an exit — the
                    // pattern sprite alpha-kill shaders use to skip their tail.
                    if (IsExitBranchTarget(instructions, instruction))
                    {
                        terminator = BlockTerminator.Exit("active = false;");
                        return true;
                    }

                    if (!TryGetBranchTargetBlock(blocks, instruction, out var target))
                    {
                        error = $"branch target outside program at pc=0x{instruction.Pc:X}";
                        return false;
                    }

                    terminator = new BlockTerminator
                    {
                        Kind = TerminatorKind.Jump,
                        TrueTarget = target,
                        DispatcherLines = [$"pc = {target}u;"],
                    };
                    return true;
                }

                if (instruction.Opcode.StartsWith("SCbranch", StringComparison.Ordinal))
                {
                    if (!TryGetBranchCondition(instruction.Opcode, out var condition))
                    {
                        error = $"unsupported conditional branch {instruction.Opcode}";
                        return false;
                    }

                    var fallthrough = blockIndex + 1;
                    if (IsExitBranchTarget(instructions, instruction))
                    {
                        // Taken → exit; not taken → fall through (or exit when
                        // this is the last block anyway).
                        if (fallthrough >= blocks.Count)
                        {
                            terminator = BlockTerminator.Exit("active = false;");
                        }
                        else
                        {
                            terminator = new BlockTerminator
                            {
                                Kind = TerminatorKind.CondJump,
                                TrueTarget = ExitTarget,
                                FalseTarget = fallthrough,
                                Condition = condition,
                                DispatcherLines =
                                [
                                    $"pc = {fallthrough}u;",
                                    $"active = !({condition});",
                                ],
                            };
                        }

                        return true;
                    }

                    if (!TryGetBranchTargetBlock(blocks, instruction, out var target))
                    {
                        error = $"branch target outside program at pc=0x{instruction.Pc:X}";
                        return false;
                    }

                    terminator = new BlockTerminator
                    {
                        Kind = TerminatorKind.CondJump,
                        TrueTarget = target,
                        FalseTarget = fallthrough >= blocks.Count ? ExitTarget : fallthrough,
                        Condition = condition,
                        DispatcherLines = fallthrough >= blocks.Count
                            ?
                            [
                                $"pc = ({condition}) ? {target}u : 0xFFFFFFFFu;",
                                $"active = ({condition});",
                            ]
                            : [$"pc = ({condition}) ? {target}u : {fallthrough}u;"],
                    };
                    return true;
                }

                if (!isTerminator && CanFuseWave64ReadLanes(instruction, instructions[index + 1]))
                {
                    // One exchange for both reads (see EmitFusedWave64ReadLanes).
                    var second = instructions[++index];
                    isTerminator = index == block.EndIndex - 1;
                    if (Gen5MslDump.Enabled)
                    {
                        Line($"// 0x{second.Pc:X4} {second.Opcode} (fused)");
                    }

                    BeginInstruction(instruction);
                    EmitFusedWave64ReadLanes(instruction, second);
                    EmitInterpolationRewrites(instruction.Pc);
                    EmitInterpolationRewrites(second.Pc);
                }
                else if (!TryEmitInstruction(instruction, out error))
                {
                    error = $"pc=0x{instruction.Pc:X4} {instruction.Opcode}: {error}";
                    return false;
                }
                else
                {
                    EmitInterpolationRewrites(instruction.Pc);
                }

                if (isTerminator)
                {
                    // Fall through to the next block (or exit at program end).
                    terminator = blockIndex + 1 < blocks.Count
                        ? new BlockTerminator
                        {
                            Kind = TerminatorKind.Jump,
                            TrueTarget = blockIndex + 1,
                            DispatcherLines = [$"pc = {blockIndex + 1}u;"],
                        }
                        : BlockTerminator.Exit("active = false;");
                }
            }

            return true;
        }

        private bool TryGetBranchCondition(string opcode, out string condition)
        {
            condition = opcode switch
            {
                "SCbranchScc0" => "!scc",
                "SCbranchScc1" => "scc",
                // VCCZ/EXECZ test the full architectural register pair, which
                // also covers programs that parked plain data in VCC; a wave32
                // compute program's VCC is VCC_LO alone (see VccIsLowHalfOnly).
                "SCbranchVccz" => VccZeroCondition,
                "SCbranchVccnz" => VccIsLowHalfOnly
                    ? $"s[{VccLoRegister}] != 0u"
                    : $"(s[{VccLoRegister}] | s[{VccHiRegister}]) != 0u",
                "SCbranchExecz" => IsSingleLaneStage ? "!exec"
                    : VccIsLowHalfOnly ? $"s[{ExecLoRegister}] == 0u"
                    : $"(s[{ExecLoRegister}] | s[{ExecHiRegister}]) == 0u",
                "SCbranchExecnz" => IsSingleLaneStage ? "exec"
                    : VccIsLowHalfOnly ? $"s[{ExecLoRegister}] != 0u"
                    : $"(s[{ExecLoRegister}] | s[{ExecHiRegister}]) != 0u",
                // The emulator does not expose a shader debug session.
                "SCbranchCdbgsys" or
                "SCbranchCdbguser" or
                "SCbranchCdbgsysOrUser" or
                "SCbranchCdbgsysAndUser" => "false",
                _ => string.Empty,
            };
            return condition.Length != 0;
        }

        private static bool TryGetBranchTargetBlock(
            IReadOnlyList<ShaderBlock> blocks,
            Gen5ShaderInstruction instruction,
            out int block)
        {
            block = -1;
            return TryGetBranchTargetPc(instruction, out var targetPc) &&
                TryFindBlock(blocks, targetPc, out block);
        }

        /// <summary>True when the branch lands at or past the last instruction's
        /// end — an exit, matching the SPIR-V translator's handling.</summary>
        private static bool IsExitBranchTarget(
            IReadOnlyList<Gen5ShaderInstruction> instructions,
            Gen5ShaderInstruction instruction)
        {
            if (instructions.Count == 0 ||
                !TryGetBranchTargetPc(instruction, out var targetPc))
            {
                return false;
            }

            var last = instructions[^1];
            var lastEndPc = last.Pc + (uint)(last.Words.Count * sizeof(uint));
            return targetPc >= lastEndPc;
        }

        // ---- instruction dispatch ----

        private static bool SetregTouchesIeeeOrClamp(uint simm16)
        {
            const uint ModeRegister = 1;
            if ((simm16 & 0x3Fu) != ModeRegister)
            {
                return false;
            }

            var offset = (simm16 >> 6) & 0x1Fu;
            var size = ((simm16 >> 11) & 0x1Fu) + 1u;
            var touched = (uint)(((1ul << (int)size) - 1ul) << (int)offset);
            return (touched & 0x300u) != 0;
        }

        private bool TryEmitInstruction(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            BeginInstruction(instruction);
            switch (instruction.Opcode)
            {
                // No shader trap handler is installed, so S_TRAP has no effect.
                case "STrap":
                    return true;
                // S_GL1_INV and S_DCACHE_INV drop the GL1 and scalar data caches so later loads
                // see memory written since. Metal gives a shader no such cache to drop: every
                // scalar load here reads its host binding or device memory when it runs, the
                // coherence BUFFER_WBINVL1 already relies on. S_DCACHE_WB has nothing to write
                // back, as no scalar store is emitted, and S_ATC_PROBE* only warm a cache.
                case "SGl1Inv":
                case "SDcacheInv":
                case "SDcacheWb":
                case "SAtcProbe":
                case "SAtcProbeBuffer":
                // BUFFER_GL0_INV / BUFFER_GL1_INV drop the per-CU and shader-array caches for the
                // same reason S_GL1_INV does; device memory here has no such cache, and a GLC
                // access already reads coherently (coherent_load_prelude).
                case "BufferGl0Inv":
                case "BufferGl1Inv":
                    return true;
                case "SMemtime":
                case "SMemrealtime":
                    return TryEmitShaderClock(instruction, out error);
                // S_SETREG_B32 simm16 = hwreg id[5:0], offset[10:6], size-1[15:11]. The translator
                // honours MODE only through the launch-time IEEE / DX10_CLAMP / FLOAT_MODE state, so a
                // write to MODE bits 8..9 would silently desync from it and is refused. Writes to
                // FP_ROUND / FP_DENORM (bits 0..7) are not tracked (Metal has no per-instruction
                // rounding and its f32 denormal flushing is fixed), and the other registers (TRAPSTS,
                // HW_ID, ...) do not affect results, so those writes stay no-ops.
                case "SSetregB32":
                case "SSetregImm32B32":
                    if (instruction.Sources.Count > 1 && SetregTouchesIeeeOrClamp(instruction.Sources[1].Value))
                    {
                        error = $"{(instruction.Opcode == "SSetregB32" ? "s_setreg_b32" : "s_setreg_imm32_b32")} " +
                            "writes MODE.IEEE/DX10_CLAMP, which is fixed at launch and not supported";
                        return false;
                    }

                    return true;
                case "SGetregB32":
                    return TryEmitGetreg(instruction, out error);
                // S_ROUND_MODE sets MODE.FP_ROUND. Metal rounds every result to nearest even, so
                // only the immediate that selects that for all precisions can be honoured.
                case "SRoundMode":
                    if ((instruction.Words[0] & 0xFFFFu) != 0)
                    {
                        error = $"s_round_mode 0x{instruction.Words[0] & 0xFFFFu:X}: only round-to-nearest-even (0) is supported";
                        return false;
                    }

                    return true;
                // Not tracked, as for an S_SETREG_B32 of MODE.FP_DENORM: Metal flushes f32 denormals
                // and keeps f16 ones whatever the mode says.
                case "SDenormMode":
                // Wave scheduling, power and debug hints with no effect on results: S_WAKEUP and
                // S_SLEEP pace waves, S_SETKILL and S_INCPERFLEVEL/S_DECPERFLEVEL drive the
                // debugger and performance counters, S_ICACHE_INV drops the instruction cache,
                // S_WAIT_IDLE waits for outstanding work like S_WAITCNT, S_TTRACEDATA_IMM writes
                // the thread trace, S_VERSION is a code-object marker and V_PIPEFLUSH /
                // V_CLREXCP flush the VALU pipe and clear the (unobservable) exception state.
                case "SWakeup":
                case "SSetkill":
                case "SSleep":
                case "SIcacheInv":
                case "SIncperflevel":
                case "SDecperflevel":
                case "SWaitIdle":
                case "STtracedataImm":
                case "SVersion":
                case "VPipeflush":
                case "VClrexcp":
                    return true;
                // Debugger halts and context-switch / trap-handler returns: there is no host
                // agent to resume a halted wave or a saved context to return to.
                case "SSethalt":
                case "SSendmsghalt":
                case "SEndpgmSaved":
                case "SRfeB64":
                    error = $"{instruction.Opcode} halts the wave or returns from a trap handler, which has no host equivalent";
                    return false;
                // A call to a PC-relative subroutine and the subvector loops re-enter code with a
                // saved PC or a halved EXEC; the block builder does not follow either.
                case "SCallB64":
                case "SSubvectorLoopBegin":
                case "SSubvectorLoopEnd":
                    error = $"{instruction.Opcode} is subroutine/subvector control flow the translator does not follow";
                    return false;
                // The emitted MSL has no counters to wait on, but a persistent-thread kernel's
                // waits are where its stores become visible to other workgroups
                // (Gen5MslCodegenOptions.PersistentWaitFences).
                case "SWaitcnt" when _options.PersistentWaitFences && IsPersistentThreadsKernel():
                    Line("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);");
                    return true;
                case "SNop":
                case "SWaitcnt":
                // A dependency-counter wait: the emitted MSL has no hardware
                // counters to wait on, so it is a hint like S_WAITCNT.
                case "SWaitcntDepctr":
                case "SInstPrefetch":
                case "STtraceData":
                // Wave scheduling priority hint; no effect on results.
                case "SSetprio":
                case "SClause":
                case "VNop":
                // NGG shaders bracket their exports with s_sendmsg
                // (GS_ALLOC_REQ/DEALLOC) to reserve hardware export space. The mesh
                // stage turns the allocation request into the output's primitive count;
                // the per-vertex path translates exports directly, so it is moot there.
                case "SSendmsg":
                    if (_stage == Gen5MslStage.Mesh && Gen5NggExports.IsAllocationRequest(instruction))
                    {
                        EmitMeshAllocationRequest();
                    }

                    return true;
                case "SBarrier":
                    if (_stage == Gen5MslStage.Mesh)
                    {
                        // Park the wave; the dispatcher's loop body is the only
                        // convergent barrier point (see Gen5MslTranslator.Mesh.cs).
                        Line("sharpemu_mesh_wait = true;");
                        return true;
                    }

                    // The device fence only orders this program's own buffer writes for
                    // the rest of the group. A program with none — an LDS-only kernel
                    // like a UE5 FFT pass, 48 barriers per thread — skips it.
                    Line(ProgramWritesDeviceMemory()
                        ? "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
                        : "threadgroup_barrier(mem_flags::mem_threadgroup);");
                    return true;
            }

            if (instruction.Control is Gen5ImageControl imageControl)
            {
                return TryEmitImage(instruction, imageControl, out error);
            }

            if (instruction.Control is Gen5RayIntersectControl rayIntersect)
            {
                return TryEmitRayIntersect(instruction, rayIntersect, out error);
            }

            if (instruction.Control is Gen5ExportControl exportControl)
            {
                return TryEmitExport(instruction, exportControl, out error);
            }

            if (instruction.Control is Gen5InterpolationControl interpolationControl)
            {
                return TryEmitInterpolation(instruction, interpolationControl, out error);
            }

            if (instruction.Control is Gen5DataShareControl dataShare)
            {
                return TryEmitDataShare(instruction, dataShare, out error);
            }

            if (instruction.Control is Gen5ScalarMemoryControl scalarMemory)
            {
                return TryEmitScalarMemory(instruction, scalarMemory, out error);
            }

            if (instruction.Control is Gen5GlobalMemoryControl globalMemory)
            {
                return TryEmitGlobalMemory(instruction, globalMemory, out error);
            }

            if (instruction.Control is Gen5BufferMemoryControl bufferMemory)
            {
                return TryEmitBufferMemory(instruction, bufferMemory, out error);
            }

            if (instruction.Opcode.StartsWith("V", StringComparison.Ordinal))
            {
                return TryEmitVectorAlu(instruction, out error);
            }

            if (instruction.Opcode.StartsWith("S", StringComparison.Ordinal))
            {
                return TryEmitScalarAlu(instruction, out error);
            }

            error = "unsupported instruction";
            return false;
        }

        // ---- memory ----

        private bool TryEmitScalarMemory(
            Gen5ShaderInstruction instruction,
            Gen5ScalarMemoryControl control,
            out string error)
        {
            error = string.Empty;
            {
                return TryEmitLayoutScalarMemory(instruction, control, out error);
            }
        }

        private const string ShaderClockName = "sharpemu_shader_clock";

        // S_MEMTIME / S_MEMREALTIME write a 64-bit counter to the SDATA pair. A Metal shader
        // has no clock to read: MSL declares none, and the AGX compiler fails the pipeline
        // on llvm.readcyclecounter (__builtin_readcyclecounter; checked on an M4, macOS 27).
        // The stand-in is the invocation's own tick counter, advanced by one on every read of
        // either instruction: a read returns more than every earlier read in the invocation,
        // so a delta between two reads is positive and a wait for the counter to pass a mark
        // ends. It counts reads, not time, and is not comparable between invocations or with
        // the 100 MHz reference clock the command processor's timestamps report. Every lane
        // of a wave runs the same scalar instructions, so the value is wave-uniform.
        private bool TryEmitShaderClock(Gen5ShaderInstruction instruction, out string error)
        {
            error = string.Empty;
            if (instruction.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister } low, _])
            {
                error = $"{instruction.Opcode} has no scalar register pair destination";
                return false;
            }

            Line($"{ShaderClockName} += 1ul;");
            StoreScalar64(low.Value, ShaderClockName);
            return true;
        }

        private bool TryEmitGlobalMemory(
            Gen5ShaderInstruction instruction,
            Gen5GlobalMemoryControl control,
            out string error)
        {
            error = string.Empty;
            {
                return TryEmitLayoutGlobalMemory(instruction, control, out error);
            }
        }

        private bool TryEmitBufferMemory(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            out string error)
        {
            error = string.Empty;
            if (instruction.Opcode is "BufferWbinvl1" or "BufferWbinvl1Vol")
            {
                // The host storage path is already coherent; the guest instruction
                // invalidates a cache level and does not access the buffer itself.
                return true;
            }

            if (control.Lds)
            {
                return TryEmitBufferLoadToLds(instruction, control, out error);
            }

            if (control.Tfe)
            {
                return TryEmitBufferLoadWithStatus(instruction, control, out error);
            }

            return TryEmitBufferMemoryAccess(instruction, control, out error);
        }

        // TFE on a MUBUF/MTBUF load writes a status dword to the VGPR after the data (RDNA2
        // ISA 8.1.3, "TFE"): zero when every page the access touched was resident, nonzero
        // after a fault, whose data then reads zero. A bound buffer is resident whole; the
        // device-descriptor path reports a dword the page table does not map
        // (_bufferTfeStatus). TFE has no meaning on a store or an atomic.
        private bool TryEmitBufferLoadWithStatus(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            out string error)
        {
            if (!instruction.Opcode.StartsWith("BufferLoad", StringComparison.Ordinal) &&
                !instruction.Opcode.StartsWith("TBufferLoad", StringComparison.Ordinal))
            {
                error = $"TFE on the non-load buffer opcode {instruction.Opcode}";
                return false;
            }

            _bufferTfeStatus = null;
            if (!TryEmitBufferMemoryAccess(instruction, control, out error))
            {
                return false;
            }

            StoreVector(control.VectorData + control.DwordCount, _bufferTfeStatus ?? "0u");
            _bufferTfeStatus = null;
            return true;
        }

        // A MUBUF load with LDS=1 writes its one dword per lane to LDS instead of VDATA, at
        // M0[15:0] + inst_offset + lane * 4 (RDNA2 ISA 8.1.9; LLVM applies the instruction
        // offset to both addresses). gfx10 allows it on LOAD_FORMAT_X, (S|U)BYTE,
        // (S|U)SHORT and DWORD. The dword is the value the same load would put in a VGPR: it
        // is loaded into VDATA as usual, copied to LDS, and VDATA is restored, since the
        // instruction leaves it untouched.
        private bool TryEmitBufferLoadToLds(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            out string error)
        {
            if (instruction.Opcode is not ("BufferLoadFormatX" or "BufferLoadUbyte" or "BufferLoadSbyte" or
                "BufferLoadUshort" or "BufferLoadSshort" or "BufferLoadDword") || control.Tfe)
            {
                error = $"buffer {instruction.Opcode} cannot return its data to LDS";
                return false;
            }

            if (!HasSharedLds || !HasRealWaveLanes)
            {
                error = $"buffer {instruction.Opcode} to LDS in a stage without shared LDS and wave lanes";
                return false;
            }

            if (control.VectorData >= VectorRegisterFileCount || instruction.Sources.Count < 4)
            {
                error = $"buffer {instruction.Opcode} to LDS has no staging register or M0 source";
                return false;
            }

            // M0 and the VDATA value to restore are read before the load runs.
            var ldsAddress = Temp(
                "uint",
                $"({SourceExpression(instruction.Sources[3], instruction)} & 0xFFFFu) + 0x{unchecked((uint)control.OffsetBytes):X}u + (sharpemu_lane << 2u)");
            var staged = Temp("uint", $"v[{control.VectorData}]");
            if (!TryEmitBufferMemoryAccess(instruction, control with { Lds = false }, out error))
            {
                return false;
            }

            Line(ExecGuarded($"sharpemu_lds[{LdsIndex(ldsAddress, 0)}] = v[{control.VectorData}];"));
            Line($"v[{control.VectorData}] = {staged};");
            return true;
        }

        private bool TryEmitBufferMemoryAccess(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            out string error)
        {
            error = string.Empty;
            if (FormatComponentCount(instruction.Opcode, control, out var formatD16) == 0 && formatD16.Enabled)
            {
                error = $"unsupported buffer opcode {instruction.Opcode}";
                return false;
            }

            if (_stage == Gen5MslStage.Vertex &&
                _vertexInputsByPc.TryGetValue(instruction.Pc, out var vertexInput))
            {
                return TryEmitVertexInputFetch(control, vertexInput, out error);
            }

            if (_request.Memory.TryGetIndex(instruction.Pc, 0, out var accessMemoryIndex))
            {
                var accessMemory = _request.Memory[accessMemoryIndex];
                if (accessMemory.PlanningOnly)
                {
                    // The dead linear tail of a fused shader: its descriptor is supplied by the
                    // continuation object and is intentionally absent from this plan.
                    return true;
                }

                var strategy = accessMemory.BufferDescriptor?.ChooseStrategy(
                        control.Typed,
                        accessMemory.Formatted,
                        accessMemory.Access == MemoryAccess.Atomic)
                    ?? BufferLoweringStrategy.NativeBinding;
                if (accessMemory.DeviceDescriptor || strategy == BufferLoweringStrategy.PhysicalStorageBuffer)
                {
                    return TryEmitDeviceDescriptorBufferMemory(instruction, control, out error);
                }

                if (strategy == BufferLoweringStrategy.BoundedCandidateTable)
                {
                    return TryEmitBoundedCandidateTableMemory(instruction, control, accessMemoryIndex, out error);
                }
            }

            if (!TryResolveLayoutBuffer(instruction.Pc, out var resolvedBinding, out _))
            {
                error = "missing buffer-memory binding";
                return false;
            }

            return EmitResolvedBufferMemory(instruction, control, resolvedBinding, out error);
        }

        // One buffer operation against a resolved binding: the dense binding path and every
        // arm of a bounded candidate table share it.
        private bool EmitResolvedBufferMemory(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            int bindingIndex,
            out string error)
        {
            error = string.Empty;
            var buffers = _request.Resources.Info.Buffers;
            if (bindingIndex < 0 || bindingIndex >= buffers.Count)
            {
                error = $"buffer binding {bindingIndex} is out of range";
                return false;
            }

            // The dense buffer, its stride and its format come from the specialization.
            var specialized = buffers[bindingIndex];
            // PackedStride holds the V#'s stride (bits 0-13), SWIZZLE_ENABLE (bit 14),
            // INDEX_STRIDE (bits 16-17) and ADD_TID_ENABLE (bit 20); the materializer keeps
            // the swizzle only for a nonzero stride.
            var packedStride = specialized.PackedStride;
            var addThreadId = (packedStride & (1u << 20)) != 0;
            if (addThreadId && !HasRealWaveLanes)
            {
                // ADD_TID_ENABLE adds the lane in the wave to the index; a stage that models
                // one lane has no lane id to add.
                error = $"buffer {instruction.Opcode} adds the thread id (packed stride 0x{packedStride:X}) in a stage without wave lanes";
                return false;
            }

            var stride = FormatUInt(packedStride & 0x3FFF);
            var descriptorFormat = specialized.DescriptorFormat;
            var descriptorWord3 = FormatUInt((specialized.DescriptorFormat << 12) | (specialized.DescriptorSwizzle & 0xFFF));
            if (_request.Bindings is { BufferWordCount: not 0 } portable && (uint)bindingIndex < portable.MemoryOffsetCount)
            {
                // Portable buffers (PortableBufferWord): the V#'s stride, and the format word
                // of a formatted buffer the program only reads, come from the shader data.
                var words = portable.FirstBufferWordDword() + (uint)bindingIndex * PortableBufferWord.DwordCount;
                stride = Temp("uint", $"({ShaderDataDword(words)} & 0x{PortableBufferWord.StrideMask:X}u)");
                if (PortableBufferWord.ReadsFormatAtRuntime(specialized))
                {
                    descriptorWord3 = Temp("uint", ShaderDataDword(words + 1));
                }
            }

            var scalarOffset = instruction.Sources.Count > 2
                ? SourceExpression(instruction.Sources[2], instruction)
                : "0u";
            var vectorIndex = BufferIndex(control, addThreadId ? "true" : null);
            var vectorOffset = control.OffsetEnabled
                ? $"v[{control.VectorAddress + (control.IndexEnabled ? 1u : 0u)}]"
                : "0u";
            BufferAddress address;
            if ((packedStride & (1u << 14)) != 0)
            {
                var indexStride = 8u << (int)((packedStride >> 16) & 3);
                address = PlacedBufferAddress(
                    scalarOffset,
                    vectorIndex,
                    $"(0x{unchecked((uint)control.OffsetBytes):X}u + {vectorOffset})",
                    stride,
                    FormatUInt(indexStride),
                    placed => ApplyByteBias(bindingIndex, placed));
            }
            else
            {
                address = new BufferAddress(
                    Temp(
                        "uint",
                        ApplyByteBias(
                            bindingIndex,
                            $"(0x{unchecked((uint)control.OffsetBytes):X}u + {scalarOffset} + {vectorOffset} + ({vectorIndex} * {stride}))")),
                    null);
            }

            var components = FormatComponentCount(instruction.Opcode, control, out var d16);
            if (instruction.Opcode.StartsWith("BufferStoreFormat", StringComparison.Ordinal))
            {
                if (descriptorFormat == 0)
                {
                    return true;
                }

                if (!TryEmitBufferFormatStore(bindingIndex, address, control, descriptorWord3, descriptorFormat, components, d16))
                {
                    error = $"unsupported buffer store format {descriptorFormat}";
                    return false;
                }

                return true;
            }

            if (control.Typed &&
                instruction.Opcode.StartsWith("TBufferStore", StringComparison.Ordinal) &&
                TryEmitBufferFormatStore(bindingIndex, address, control, descriptorWord3, control.TypedFormat, components, d16))
            {
                return true;
            }

            // A typed load converts with the instruction format, a formatted untyped load
            // with the descriptor format and swizzle; raw accesses take the byte path below.
            if (IsFormatBufferLoad(instruction.Opcode))
            {
                if (!control.Typed)
                {
                    EmitBufferFormatLoad(
                        bindingIndex,
                        address,
                        descriptorWord3,
                        control.VectorData,
                        components,
                        d16,
                        control.Glc);
                    return true;
                }

                if (TryEmitTypedBufferFormatLoad(BindingFormatMemory(bindingIndex, control.Glc), address, control, descriptorWord3, components, d16))
                {
                    return true;
                }
            }

            // A D16 access packs halves, so the raw dword path below cannot stand in for a
            // typed format that does not decode.
            if (d16.Enabled)
            {
                error = $"unsupported typed format {control.TypedFormat} for {instruction.Opcode}";
                return false;
            }

            return TryEmitResolvedMemoryAccess(
                instruction.Opcode,
                bindingIndex,
                address,
                control.VectorData,
                control.VectorData,
                control.DwordCount,
                control.Glc,
                out error);
        }

        // Where the bytes of one buffer access land. A linear access is one byte address
        // (Base) that its bytes follow contiguously. A placed one - a swizzled V#, or a V#
        // held in scalar registers, whose words decide at run time - has no such address:
        // each dword is placed on its own from its unswizzled byte offset (RDNA2 ISA 8.1.5,
        // the swizzle element is one dword on gfx10), so Place maps a byte distance into
        // the access to the address of that dword.
        private readonly record struct BufferAddress(string Base, Func<string, string>? Place)
        {
            public bool Contiguous => Place is null;

            public string At(uint bytes) =>
                Place is not null ? Place($"{bytes}u") : bytes == 0 ? Base : $"({Base} + {bytes}u)";

            public string At(string bytes) =>
                Place is not null ? Place(bytes) : $"({Base} + {bytes})";
        }

        // The index of a MUBUF/MTBUF access: the VGPR index when IDXEN is set, plus the lane
        // in the wave (thread_id[5:0]) when the V# sets ADD_TID_ENABLE (RDNA2 ISA 8.1.5).
        // addThreadId is null for a V# without it, "true" for one known to set it, or a
        // run-time condition.
        private static string BufferIndex(Gen5BufferMemoryControl control, string? addThreadId)
        {
            var vectorIndex = control.IndexEnabled ? $"v[{control.VectorAddress}]" : "0u";
            return addThreadId switch
            {
                null => vectorIndex,
                "true" => control.IndexEnabled ? $"({vectorIndex} + sharpemu_lane)" : "sharpemu_lane",
                _ => $"({vectorIndex} + ({addThreadId} ? sharpemu_lane : 0u))",
            };
        }

        // The one placement of a buffer access's dwords that the bound and the scalar
        // register V# paths share. The index, the unswizzled offset (VGPR offset plus
        // instruction offset) and the SGPR offset are taken now, before a load writes a
        // destination that may alias them; each dword then lands at
        // soffset + sharpemu_buffer_offset(index, offset + distance), the SGPR offset added
        // outside the swizzle. A zero index stride places linearly; finish adds what the
        // path adds after (a binding's byte bias).
        private BufferAddress PlacedBufferAddress(
            string scalarOffset,
            string index,
            string offset,
            string stride,
            string indexStride,
            Func<string, string> finish)
        {
            _usesBufferOffset = true;
            var scalar = Temp("uint", scalarOffset);
            var placedIndex = Temp("uint", index);
            var placedOffset = Temp("uint", offset);
            return new BufferAddress(
                placedOffset,
                distance => Temp(
                    "uint",
                    finish($"({scalar} + sharpemu_buffer_offset({placedIndex}, {placedOffset} + {distance}, {stride}, {indexStride}))")));
        }

        // Whether all bytes of an access are in range: one check over a linear access; for
        // a placed one each dword is checked at its own address, the access in range only
        // when all are (formatted accesses and atomics are all or nothing, RDNA2 ISA 8.1.5
        // Table 35 notes).
        private string AccessInBounds(FormatMemory memory, BufferAddress address, uint bytes)
        {
            if (address.Contiguous)
            {
                return memory.ElementInBounds(address.Base, $"{bytes}u");
            }

            var parts = new List<string>();
            for (uint distance = 0; distance < bytes; distance += sizeof(uint))
            {
                parts.Add(memory.ElementInBounds(address.At(distance), $"{Math.Min(sizeof(uint), bytes - distance)}u"));
            }

            return parts.Count == 1 ? parts[0] : Temp("bool", string.Join(" && ", parts));
        }

        private void EmitBufferFormatLoad(
            int bindingIndex,
            BufferAddress byteAddress,
            string descriptorWord3,
            uint vectorData,
            uint componentCount,
            FormatD16 d16,
            bool glc)
        {
            // A descriptor word the specialization fixed is decoded here; a format of whole
            // dwords then loads as one run. Anything else, and every D16 load, decodes on the
            // device below. A placed access has no run: its dwords are not contiguous.
            if (!d16.Enabled &&
                byteAddress.Contiguous &&
                TryParseUIntLiteral(descriptorWord3, out var literalWord3) &&
                TryEmitLiteralDwordFormatLoad(bindingIndex, byteAddress.Base, literalWord3, vectorData, componentCount, glc))
            {
                return;
            }

            EmitDecodedBufferFormatLoad(
                BindingFormatMemory(bindingIndex, glc),
                byteAddress,
                descriptorWord3,
                vectorData,
                componentCount,
                d16);
        }

        // The D16 forms of the formatted accesses (BUFFER/TBUFFER_LOAD/STORE_FORMAT_D16_*,
        // RDNA2 ISA 12.9/12.14): each component travels as a 16-bit half, two to a register
        // with X in the low half, so XYZ and XYZW take two registers. The Hi form moves its one
        // component through the high half of its register.
        private readonly record struct FormatD16(bool Enabled, bool High);

        // How many components a formatted access transfers: one per register, or for a D16
        // form the count its suffix names. Zero for a D16 suffix that is not known.
        private static uint FormatComponentCount(string opcode, Gen5BufferMemoryControl control, out FormatD16 d16)
        {
            const string Marker = "FormatD16";
            var at = opcode.IndexOf(Marker, StringComparison.Ordinal);
            if (at < 0)
            {
                d16 = default;
                return control.DwordCount;
            }

            var suffix = opcode[(at + Marker.Length)..];
            var high = suffix.StartsWith("Hi", StringComparison.Ordinal);
            d16 = new FormatD16(true, high);
            return (high ? suffix[2..] : suffix) switch
            {
                "X" => 1u,
                "Xy" when !high => 2u,
                "Xyz" when !high => 3u,
                "Xyzw" when !high => 4u,
                _ => 0u,
            };
        }

        // The converted components of a formatted load into their registers. A D16 load
        // narrows each with sharpemu_format_d16 and packs the halves; a register it fills only
        // half of (D16_X, the Z of D16_XYZ, D16_HI_X) keeps its other half, as the hardware's
        // D16 loads preserve the bits they do not write.
        private void StoreFormatLoadComponents(uint vectorData, IReadOnlyList<string> values, FormatD16 d16, string numberFormat)
        {
            if (!d16.Enabled)
            {
                for (var component = 0; component < values.Count; component++)
                {
                    StoreVector(vectorData + (uint)component, values[component]);
                }

                return;
            }

            var halves = new string[values.Count];
            for (var component = 0; component < values.Count; component++)
            {
                halves[component] = Temp("uint", $"sharpemu_format_d16({values[component]}, {numberFormat})");
            }

            if (d16.High)
            {
                StoreVector(vectorData, $"(v[{vectorData}] & 0x0000FFFFu) | ({halves[0]} << 16)");
                return;
            }

            for (var component = 0; component < halves.Length; component += 2)
            {
                var register = vectorData + (uint)(component / 2);
                StoreVector(
                    register,
                    component + 1 < halves.Length
                        ? $"{halves[component]} | ({halves[component + 1]} << 16)"
                        : $"(v[{register}] & 0xFFFF0000u) | {halves[component]}");
            }
        }

        // The register value a formatted store transfers for one component: the register, or
        // for a D16 store its half widened back to the 32-bit value the number format encodes
        // from (sharpemu_format_d16_expand).
        private string FormatStoreComponent(uint vectorData, uint component, FormatD16 d16, string numberFormat)
        {
            if (!d16.Enabled)
            {
                return $"v[{vectorData + component}]";
            }

            var register = vectorData + (component / 2);
            var half = d16.High || (component & 1) != 0
                ? $"(v[{register}] >> 16u)"
                : $"(v[{register}] & 0xFFFFu)";
            return Temp("uint", $"sharpemu_format_d16_expand({half}, {numberFormat})");
        }

        // BUFFER_LOAD_FORMAT_* with the format decoded on the device: the bound path and a
        // V# held in scalar registers (read through the device-address page table) share it.
        private void EmitDecodedBufferFormatLoad(
            FormatMemory memory,
            BufferAddress byteAddress,
            string descriptorWord3,
            uint vectorData,
            uint componentCount,
            FormatD16 d16)
        {
            var packedRun = _options.PackedFormatLoads && !d16.Enabled && byteAddress.Contiguous &&
                memory.LoadWords is not null && TryParseUIntLiteral(descriptorWord3, out var packedWord3) &&
                BeginLiteralPackedFormatLoad(memory, byteAddress.Base, packedWord3, vectorData, componentCount);
            // Format and destination swizzle come from descriptor word 3, a register
            // or a specialization constant; the prelude table decodes the unified format.
            var word3 = Temp("uint", descriptorWord3);
            var entry = Temp("uint", $"sharpemu_gfx10_formats[({word3} >> 12) & 0x7Fu]");
            var dataFormat = Temp("uint", $"{entry} & 0xFFu");
            var numberFormat = Temp("uint", $"({entry} >> 8) & 0xFFu");
            // Portable vertex layouts keep word 3 dynamic. The common whole-dword case
            // needs neither four layout lookups nor four numeric conversions. Check the
            // current word, including every selected swizzle, on each invocation: this
            // must not specialize a descriptor that can change between draws.
            var dwordRun = _options.RuntimeDwordFormatLoads && !d16.Enabled &&
                byteAddress.Contiguous && memory.LoadWords is not null &&
                !TryParseUIntLiteral(descriptorWord3, out _);
            if (dwordRun)
            {
                var formats = componentCount switch
                {
                    1 => $"{dataFormat} == 4u || {dataFormat} == 11u || {dataFormat} == 13u || {dataFormat} == 14u",
                    2 => $"{dataFormat} == 11u || {dataFormat} == 13u || {dataFormat} == 14u",
                    _ => $"{dataFormat} == 13u || {dataFormat} == 14u",
                };
                var selectorMask = (1u << (int)(componentCount * 3)) - 1;
                // A float3 XYZW fetch may read one extra dword only when it is in
                // bounds; the final short element retains the general decoder.
                var inRange = memory.ElementInBounds(byteAddress.Base, $"{componentCount * sizeof(uint)}u");
                Line($"if (({formats}) && ({numberFormat} == 4u || {numberFormat} == 5u || {numberFormat} == 7u) && " +
                    $"({word3} & 0x{selectorMask:X}u) == 0x{0xFACu & selectorMask:X}u && {inRange})");
                Line("{");
                _indent++;
                var words = Temp("uint4", memory.LoadWords!(componentCount, byteAddress.Base));
                if (componentCount == 4)
                {
                    Line($"{words}.w = {dataFormat} == 13u ? sharpemu_format_one({numberFormat}) : {words}.w;");
                }
                for (uint component = 0; component < componentCount; component++)
                {
                    StoreVector(vectorData + component, $"{words}[{component}]");
                }
                _indent--;
                Line("}");
                Line("else");
                Line("{");
                _indent++;
            }
            // A program with many of these loads calls the decoder (format_load_prelude.msl).
            if (memory.Decode is not null && byteAddress.Contiguous &&
                _options.DecodedFormatLoadThreshold > 0 && _formatBufferLoads >= _options.DecodedFormatLoadThreshold)
            {
                _usesDecodedFormatLoads = true;
                var decoded = Temp("uint4", memory.Decode(byteAddress.Base, word3, componentCount));
                var decodedValues = new string[componentCount];
                for (var destination = 0; destination < decodedValues.Length; destination++)
                {
                    decodedValues[destination] = $"{decoded}[{destination}]";
                }

                StoreFormatLoadComponents(vectorData, decodedValues, d16, numberFormat);
                if (dwordRun || packedRun)
                {
                    _indent--;
                    Line("}");
                }

                return;
            }

            // A format the table cannot decode reads every destination as zero there.
            var valid = memory.ZeroInvalidFormat ? Temp("bool", $"{dataFormat} != 0u") : null;
            var canonical = new string[4];
            var componentBounds = new string[4];
            for (var component = 0; component < 4; component++)
            {
                var missing = component == 3
                    ? $"sharpemu_format_one({numberFormat})"
                    : "0u";
                canonical[component] = LoadFormatComponent(
                    memory,
                    byteAddress,
                    dataFormat,
                    numberFormat,
                    component,
                    missing,
                    out componentBounds[component]);
            }

            // Only selected memory components contribute to the shared bounds check.
            var selectors = new string[componentCount];
            var inBounds = Temp("bool", "true");
            for (uint destination = 0; destination < componentCount; destination++)
            {
                var selector = Temp("uint", $"({word3} >> {destination * 3}u) & 7u");
                selectors[destination] = selector;
                Line($"{inBounds} = {inBounds} && ({selector} == 4u ? {componentBounds[0]} : " +
                    $"{selector} == 5u ? {componentBounds[1]} : {selector} == 6u ? {componentBounds[2]} : " +
                    $"{selector} == 7u ? {componentBounds[3]} : true);");
            }

            var values = new string[componentCount];
            for (uint destination = 0; destination < componentCount; destination++)
            {
                var selector = selectors[destination];
                var constant = Temp("uint", $"{selector} == 1u ? sharpemu_format_one({numberFormat}) : 0u");
                var value = $"!{inBounds} ? {constant} : " +
                    $"{selector} == 4u ? {canonical[0]} : " +
                    $"{selector} == 5u ? {canonical[1]} : " +
                    $"{selector} == 6u ? {canonical[2]} : " +
                    $"{selector} == 7u ? {canonical[3]} : {constant}";
                values[destination] = valid is null ? value : $"{valid} ? ({value}) : 0u";
            }

            StoreFormatLoadComponents(vectorData, values, d16, numberFormat);
            if (dwordRun || packedRun)
            {
                _indent--;
                Line("}");
            }
        }

        private static bool TryParseUIntLiteral(string text, out uint value)
        {
            value = 0;
            if (text.Length < 2 || text[^1] != 'u')
            {
                return false;
            }

            var digits = text[..^1];
            return digits.StartsWith("0x", StringComparison.Ordinal)
                ? uint.TryParse(digits.AsSpan(2), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out value)
                : uint.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out value);
        }

        // BUFFER_LOAD_FORMAT_* through a descriptor word 3 known at translate time, for a
        // format whose components are whole dwords at byte offsets 0, 4, 8 and 12: the
        // selectors are constants, so only the selected components are loaded, as one run
        // from one address, and only they are bounds checked (the same rule the descriptor
        // table form applies at run time). A component the format lacks reads as the
        // missing value, W as one and the rest as zero; an out of range access reads the
        // selector's constant.
        private bool TryEmitLiteralDwordFormatLoad(
            int bindingIndex,
            string byteAddress,
            uint word3,
            uint vectorData,
            uint componentCount,
            bool glc)
        {
            if (!Gfx10UnifiedFormat.TryDecode((word3 >> 12) & 0x7Fu, out var dataFormat, out var numberFormat))
            {
                return false;
            }

            var formatComponents = Gfx10UnifiedFormat.ComponentCount(dataFormat);
            if (formatComponents == 0 || !Gfx10UnifiedFormat.HasWholeDwordComponents(dataFormat))
            {
                return false;
            }

            for (uint component = 0; component < formatComponents; component++)
            {
                if (!Gfx10UnifiedFormat.TryGetComponentLayout(
                        dataFormat, component, out var offset, out var bit, out var bits) ||
                    offset != component * sizeof(uint) || bit != 0 || bits != 32)
                {
                    return false;
                }
            }

            // The last selected component the format has decides the run and its bounds.
            var run = 0;
            for (uint destination = 0; destination < componentCount; destination++)
            {
                var selector = (word3 >> (int)(destination * 3)) & 7u;
                if (selector >= 4 && selector - 4 < formatComponents)
                {
                    run = Math.Max(run, (int)(selector - 4) + 1);
                }
            }

            string? words = null;
            var inBounds = "true";
            if (run > 0)
            {
                inBounds = ElementInBounds(bindingIndex, byteAddress, $"{run * sizeof(uint)}u");
                _usesCoherentLoads |= glc;
                words = Temp(
                    "uint4",
                    $"sharpemu_load_words{(glc ? "_coherent" : string.Empty)}<{run}>(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress})");
            }

            var one = $"sharpemu_format_one({numberFormat}u)";
            for (uint destination = 0; destination < componentCount; destination++)
            {
                var selector = (word3 >> (int)(destination * 3)) & 7u;
                var constant = selector == 1 ? one : "0u";
                string value;
                if (selector < 4)
                {
                    value = constant;
                }
                else if (selector - 4 >= formatComponents)
                {
                    value = selector == 7 ? one : "0u";
                }
                else
                {
                    value = Temp(
                        "uint",
                        $"sharpemu_format_convert({words}[{selector - 4}], 32u, {numberFormat}u, {dataFormat}u)");
                }

                // An access out of range reads every destination as its constant, a
                // selected component the format lacks included.
                var guarded = selector >= 4 && run > 0 && constant != value;
                StoreVector(vectorData + destination, guarded ? $"{inBounds} ? {value} : {constant}" : value);
            }

            return true;
        }

        // Check the required range as one access, without an overflowing end address.
        private string ElementInBounds(int bindingIndex, string byteAddress, string elementBytes)
        {
            var bytes = BufferBytes(bindingIndex);
            return Temp("bool", $"{elementBytes} <= {bytes} && {byteAddress} <= {bytes} - {elementBytes}");
        }

        // Component i of a typed load comes from memory component i; components the
        // format does not have read as zero. An unbound descriptor reads as zero.
        private bool TryEmitTypedBufferFormatLoad(
            FormatMemory memory,
            BufferAddress byteAddress,
            Gen5BufferMemoryControl control,
            string descriptorWord3,
            uint transferred,
            FormatD16 d16)
        {
            if (!Gfx10UnifiedFormat.TryDecode(control.TypedFormat, out var dataFormat, out var numberFormat))
            {
                return false;
            }

            var componentCount = Gfx10UnifiedFormat.ComponentCount(dataFormat);
            if (componentCount == 0)
            {
                return false;
            }

            // All transferred components must be bound and inside the binding.
            var accessBytes = Gfx10UnifiedFormat.GetAccessByteSize(dataFormat, transferred);
            var inBounds = AccessInBounds(memory, byteAddress, accessBytes);
            var valid = Temp(
                "bool",
                $"(({descriptorWord3} >> 12) & 0x7Fu) != 0u && {inBounds}");

            // Formats of whole dword components sit at byte offsets 0, 4, 8 and 12 of the
            // element: the address is added once and the run is loaded by one helper, with a
            // plain aligned read when it is aligned and inside the binding. A placed access
            // has no run: each dword lands on its own.
            string? dwordRun = null;
            var runCount = Math.Min(transferred, componentCount);
            if (memory.LoadWords is not null && byteAddress.Contiguous &&
                Gfx10UnifiedFormat.HasWholeDwordComponents(dataFormat) && runCount is >= 1 and <= 4)
            {
                var consecutive = true;
                for (uint component = 0; component < runCount; component++)
                {
                    consecutive &= Gfx10UnifiedFormat.TryGetComponentLayout(
                            dataFormat, component, out var componentOffset, out var componentBit, out var componentBits) &&
                        componentOffset == component * sizeof(uint) && componentBit == 0 && componentBits == 32;
                }

                if (consecutive)
                {
                    dwordRun = Temp("uint4", memory.LoadWords(runCount, byteAddress.Base));
                }
            }

            var values = new string[transferred];
            for (uint destination = 0; destination < transferred; destination++)
            {
                var value = dwordRun is not null && destination < runCount
                    ? Temp(
                        "uint",
                        $"sharpemu_format_convert({dwordRun}[{destination}], 32u, {numberFormat}u, {dataFormat}u)")
                    : destination < componentCount
                    ? LoadFormatComponent(
                        memory,
                        byteAddress,
                        $"{dataFormat}u",
                        $"{numberFormat}u",
                        (int)destination,
                        "0u",
                        out _)
                    : "0u";
                values[destination] = $"{valid} ? {value} : 0u";
            }

            StoreFormatLoadComponents(control.VectorData, values, d16, $"{numberFormat}u");
            return true;
        }

        // A formatted store converts each register with the selected number format and places
        // the bits at the component's offset; all transferred components are stored or dropped.
        private bool TryEmitBufferFormatStore(
            int bindingIndex,
            BufferAddress byteAddress,
            Gen5BufferMemoryControl control,
            string descriptorWord3,
            uint unifiedFormat,
            uint transferred,
            FormatD16 d16)
        {
            if (!Gfx10UnifiedFormat.TryDecode(unifiedFormat, out var dataFormat, out var numberFormat))
            {
                return false;
            }

            var componentCount = Math.Min(transferred, Gfx10UnifiedFormat.ComponentCount(dataFormat));
            if (componentCount == 0)
            {
                return false;
            }

            var elementBytes = Gfx10UnifiedFormat.GetAccessByteSize(dataFormat, componentCount);
            var inBounds = AccessInBounds(BindingFormatMemory(bindingIndex), byteAddress, elementBytes);
            var allowed = Temp(
                "bool",
                $"(({descriptorWord3} >> 12) & 0x7Fu) != 0u && {inBounds}");
            Line($"if ({ExecCondition(allowed)})");
            Line("{");
            _indent++;
            if (Gfx10UnifiedFormat.HasWholeDwordComponents(dataFormat))
            {
                // Dword components are dword aligned and keep their register bits (a D16
                // store's widened half).
                for (uint component = 0; component < componentCount; component++)
                {
                    Gfx10UnifiedFormat.TryGetComponentLayout(dataFormat, component, out var byteOffset, out _, out _);
                    Line(StoreDwordStatement(
                        bindingIndex,
                        byteAddress.At(byteOffset),
                        FormatStoreComponent(control.VectorData, component, d16, $"{numberFormat}u")));
                }
            }
            else
            {
                var dwordCount = (elementBytes + 3) / 4;
                var values = new string[4];
                var masks = new uint[4];
                Array.Fill(values, "0u");
                for (uint component = 0; component < componentCount; component++)
                {
                    Gfx10UnifiedFormat.TryGetComponentLayout(dataFormat, component, out var byteOffset, out var bitOffset, out var bitCount);
                    var encoded = Temp(
                        "uint",
                        $"sharpemu_format_encode({FormatStoreComponent(control.VectorData, component, d16, $"{numberFormat}u")}, {bitCount}u, {numberFormat}u, {dataFormat}u)");
                    var dword = (int)(byteOffset / 4);
                    var bit = ((byteOffset & 3) * 8) + bitOffset;
                    var placed = bit == 0 ? encoded : $"({encoded} << {bit}u)";
                    values[dword] = values[dword] == "0u" ? placed : $"{values[dword]} | {placed}";
                    masks[dword] |= ((1u << (int)bitCount) - 1) << (int)bit;
                }

                if (byteAddress.Contiguous)
                {
                    Line(
                        $"sharpemu_store_element(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress.Base}, {dwordCount}u, " +
                        $"uint4({values[0]}, {values[1]}, {values[2]}, {values[3]}), " +
                        $"uint4(0x{masks[0]:X}u, 0x{masks[1]:X}u, 0x{masks[2]:X}u, 0x{masks[3]:X}u));");
                }
                else
                {
                    // A placed element's dwords land apart: each merges at its own address.
                    for (uint dword = 0; dword < dwordCount; dword++)
                    {
                        Line(
                            $"sharpemu_store_element(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress.At(dword * sizeof(uint))}, 1u, " +
                            $"uint4({values[dword]}, 0u, 0u, 0u), uint4(0x{masks[dword]:X}u, 0u, 0u, 0u));");
                    }
                }
            }

            _indent--;
            Line("}");
            return true;
        }

        // Where a formatted load reads its element's bytes: a host binding, or a V# held in
        // scalar registers and read through the device-address page table. The bounds check
        // takes (address, bytes); the word load takes (address, bytes, in bounds) and reads
        // the unaligned word holding those bytes; LoadWords, when present, loads a run of
        // whole dwords. ZeroInvalidFormat reads a format the table cannot decode as zero.
        // Decode, when present, is the whole general decoder of a contiguous element as one
        // call (address, word 3, components): the destination values as a uint4.
        private readonly record struct FormatMemory(
            Func<string, string, string> ElementInBounds,
            Func<string, string, string, string> LoadWord,
            Func<uint, string, string>? LoadWords,
            bool ZeroInvalidFormat,
            Func<string, string, uint, string>? Decode = null);

        // A GLC load reads its words through the coherent helpers (coherent_load_prelude).
        private FormatMemory BindingFormatMemory(int bindingIndex, bool glc = false)
        {
            _usesCoherentLoads |= glc;
            return new(
                (address, bytes) => ElementInBounds(bindingIndex, address, bytes),
                (address, _, _) => glc
                    ? $"sharpemu_load_word_coherent(b{bindingIndex}, {BufferBytes(bindingIndex)}, {address})"
                    : LoadWord(bindingIndex, address),
                (count, address) => $"sharpemu_load_words{(glc ? "_coherent" : string.Empty)}<{count}>(b{bindingIndex}, {BufferBytes(bindingIndex)}, {address})",
                ZeroInvalidFormat: false,
                Decode: glc
                    ? null
                    : (address, word3, count) =>
                        $"sharpemu_format_load_decoded(b{bindingIndex}, {BufferBytes(bindingIndex)}, {address}, {word3}, {count}u)");
        }

        private string LoadFormatComponent(
            FormatMemory memory,
            BufferAddress byteAddress,
            string dataFormat,
            string numberFormat,
            int component,
            string missing,
            out string componentInBounds)
        {
            var byteOffset = Temp("uint", "0u");
            var bitOffset = Temp("uint", "0u");
            var bitCount = Temp("uint", "0u");
            Line($"sharpemu_format_layout({dataFormat}, {component}u, {byteOffset}, {bitOffset}, {bitCount});");
            var componentBytes = Temp("uint", $"({bitOffset} + {bitCount} + 7u) >> 3u");
            var componentAddress = byteAddress.At(byteOffset);
            var rangeInBounds = memory.ElementInBounds(componentAddress, componentBytes);
            componentInBounds = Temp("bool", $"{bitCount} == 0u || {rangeInBounds}");
            var packed = Temp(
                "uint",
                memory.LoadWord(componentAddress, componentBytes, rangeInBounds));
            var raw = Temp(
                "uint",
                $"{bitCount} == 0u ? 0u : extract_bits({packed}, {bitOffset}, {bitCount})");
            return Temp(
                "uint",
                $"{bitCount} == 0u ? {missing} : sharpemu_format_convert({raw}, {bitCount}, {numberFormat}, {dataFormat})");
        }

        private string LdsIndex(string address, uint offsetBytes)
        {
            var wordMask = _stage switch
            {
                Gen5MslStage.Compute => _computeLdsDwords - 1,
                Gen5MslStage.Mesh => (uint)(System.Numerics.BitOperations.RoundUpToPowerOf2(_meshLdsDwords) - 1),
                _ => PrivateLdsDwordCount - 1,
            };
            var index = offsetBytes == 0
                ? $"((({address}) >> 2) & {wordMask}u)"
                : $"(((({address}) + {offsetBytes}u) >> 2) & {wordMask}u)";

            // The mesh LDS is declared at exactly the guest allocation, so the masked index
            // can name a dword past it; clamp to the last one.
            return _stage == Gen5MslStage.Mesh && !UsesDeviceLds && wordMask != _meshLdsDwords - 1
                ? $"min({index}, {_meshLdsDwords - 1}u)"
                : index;
        }

        private bool TryEmitDataShare(
            Gen5ShaderInstruction instruction,
            Gen5DataShareControl control,
            out string error)
        {
            error = string.Empty;
            if (control.Gds)
            {
                {
                    return TryEmitGlobalDataShare(instruction, control, out error);
                }
            }

            // Any intervening LDS writer may alias the cached record. Even a masked
            // write invalidates every lane's copy, not just the writing lane's copy.
            if (UseMeshLdsForwarding && !instruction.Opcode.StartsWith("DsRead", StringComparison.Ordinal) &&
                instruction.Opcode is not ("DsSwizzleB32" or "DsBpermuteB32" or "DsNop"))
                Line("sharpemu_lds_forward_valid = false;");

            if (TryEmitMeshHotLds(instruction, control)) return true;

            void StoreLds(string index, string value)
            {
                // Exec-guarded like every other lane-visible write.
                Line(ExecGuarded($"sharpemu_lds[{index}] = {value};"));
            }

            switch (instruction.Opcode)
            {
                case "DsSwizzleB32":
                    return TryEmitDataShareSwizzle(instruction, control, out error);
                case "DsBpermuteB32":
                    return TryEmitDataShareBpermute(instruction, control, out error);
                case "DsPermuteB32":
                    return TryEmitDataSharePermute(instruction, control, out error);
                case "DsNop":
                    return true;
                case "DsWriteB8":
                case "DsWriteB16":
                case "DsWriteB8D16Hi":
                case "DsWriteB16D16Hi":
                    return TryEmitDataShareSubwordWrite(instruction, control, out error);
                case "DsReadU8":
                case "DsReadI16":
                case "DsReadU16":
                case "DsReadU8D16":
                case "DsReadU8D16Hi":
                case "DsReadI8D16":
                case "DsReadI8D16Hi":
                case "DsReadU16D16":
                case "DsReadU16D16Hi":
                    return TryEmitDataShareSubwordRead(instruction, control, out error);
                case "DsAppend":
                case "DsConsume":
                {
                    if (instruction.Sources.Count < 1 || instruction.Destinations.Count < 1)
                    {
                        error = $"missing {instruction.Opcode} operand";
                        return false;
                    }

                    var offset = control.SingleOffsetBytes;
                    var m0 = Temp("uint", RawSource(instruction, 0));
                    var baseAddress = Temp("uint", $"{m0} >> 16u");
                    var sizeBytes = Temp("uint", $"{m0} & 0xFFFFu");
                    var inBounds = Temp("bool", $"{offset + 3}u < {sizeBytes}");
                    var index = LdsIndex(baseAddress, offset);
                    var destination = instruction.Destinations[0].Value;
                    var operation = instruction.Opcode == "DsAppend" ? "add" : "sub";

                    // Per-invocation LDS keeps the one-lane model: the counter is this lane's own.
                    if (!HasSharedLds)
                    {
                        var original = Temp("uint", $"sharpemu_lds[{index}]");
                        var assignment = operation == "add" ? "+=" : "-=";
                        Line($"if ({ExecCondition(inBounds)}) {{ sharpemu_lds[{index}] {assignment} 1u; }}");
                        StoreVector(destination, $"{inBounds} ? {original} : 0u");
                        return true;
                    }

                    string count;
                    string first;
                    if (IsWave64)
                    {
                        var broadcast = EmitWave64AppendAtomic(
                            waveCount => $"atomic_fetch_{operation}_explicit(({LdsAddressSpace} atomic_uint*)&sharpemu_lds[{index}], {waveCount}, memory_order_relaxed)",
                            inBounds);
                        StoreVector(destination, $"{inBounds} ? {broadcast} : 0u");
                        return true;
                    }

                    var mask = Temp("uint", "sharpemu_ballot(exec)");
                    count = Temp("uint", $"popcount({mask})");
                    first = Temp("uint", $"{mask} == 0u ? 0u : (uint)ctz({mask})");
                    var firstValue = Temp("uint", "0u");
                    Line($"if ({ExecCondition($"{inBounds} && sharpemu_lane == {first}")})");
                    Line("{");
                    _indent++;
                    Line($"{firstValue} = atomic_fetch_{operation}_explicit(({LdsAddressSpace} atomic_uint*)&sharpemu_lds[{index}], {count}, memory_order_relaxed);");
                    _indent--;
                    Line("}");
                    var result = Temp("uint", $"simd_broadcast({firstValue}, {first})");
                    StoreVector(destination, $"{inBounds} ? {result} : 0u");
                    return true;
                }
                case "DsWriteB32":
                case "DsWriteAddtidB32":
                {
                    if (LaneSlotLdsElement(instruction) is { } slot)
                    {
                        Line(ExecGuarded($"{slot} = {RawSource(instruction, 1)};"));
                        return true;
                    }

                    var address = Temp("uint", RawSource(instruction, 0));
                    if (instruction.Opcode == "DsWriteAddtidB32")
                    {
                        address = Temp("uint", $"({address} & 0xFFFFu) + (sharpemu_lane << 2u)");
                    }
                    StoreLds(LdsIndex(address, control.SingleOffsetBytes), RawSource(instruction, 1));
                    return true;
                }
                case "DsWriteB64":
                {
                    var address = Temp("uint", RawSource(instruction, 0));
                    var offset = control.SingleOffsetBytes;
                    StoreLds(LdsIndex(address, offset), RawSource(instruction, 1));
                    StoreLds(LdsIndex(address, offset + sizeof(uint)), RawSource(instruction, 2));
                    return true;
                }
                case "DsWriteB96":
                case "DsWriteB128":
                {
                    var dwordCount = instruction.Opcode == "DsWriteB128" ? 4 : 3;
                    var address = Temp("uint", RawSource(instruction, 0));
                    var offset = control.SingleOffsetBytes;
                    for (var dword = 0; dword < dwordCount; dword++)
                    {
                        StoreLds(
                            LdsIndex(address, offset + (uint)(dword * sizeof(uint))),
                            RawSource(instruction, 1 + dword));
                    }
                    RememberMeshLdsWrite(instruction, address, offset);

                    return true;
                }
                case "DsWrite2B64":
                case "DsWrite2St64B64":
                    return TryEmitDataShareWritePair64(instruction, control, out error);
                case "DsWrite2B32":
                case "DsWrite2St64B32":
                {
                    var st64 = instruction.Opcode == "DsWrite2St64B32";
                    var address = Temp("uint", RawSource(instruction, 0));
                    StoreLds(
                        LdsIndex(address, EffectiveDsPairOffsetBytes(control.Offset0, st64)),
                        RawSource(instruction, 1));
                    StoreLds(
                        LdsIndex(address, EffectiveDsPairOffsetBytes(control.Offset1, st64)),
                        RawSource(instruction, 2));
                    return true;
                }
                case "DsReadB32":
                case "DsReadAddtidB32":
                {
                    if (LaneSlotLdsElement(instruction) is { } slot)
                    {
                        StoreVector(instruction.Destinations[0].Value, slot);
                        return true;
                    }

                    var address = Temp("uint", RawSource(instruction, 0));
                    if (instruction.Opcode == "DsReadAddtidB32")
                    {
                        address = Temp("uint", $"({address} & 0xFFFFu) + (sharpemu_lane << 2u)");
                    }
                    StoreVector(
                        instruction.Destinations[0].Value,
                        $"sharpemu_lds[{LdsIndex(address, control.SingleOffsetBytes)}]");
                    return true;
                }
                case "DsReadI8":
                {
                    if (instruction.Destinations.Count < 1 || instruction.Sources.Count < 1)
                    {
                        error = "missing LDS signed byte read operand";
                        return false;
                    }

                    // The byte lives in the word the address selects; which byte
                    // comes from the low two bits of the offset address.
                    var address = Temp("uint", RawSource(instruction, 0));
                    var byteAddress = Temp("uint", $"{address} + {control.SingleOffsetBytes}u");
                    var word = Temp("uint", $"sharpemu_lds[{LdsIndex(address, control.SingleOffsetBytes)}]");
                    var shift = Temp("uint", $"({byteAddress} & 3u) << 3u");
                    StoreVector(
                        instruction.Destinations[0].Value,
                        $"(uint)(((int)(({word} >> {shift}) << 24u)) >> 24)");
                    return true;
                }
                case "DsReadB64":
                case "DsReadB96":
                case "DsReadB128":
                {
                    var dwordCount = instruction.Opcode switch { "DsReadB64" => 2, "DsReadB96" => 3, _ => 4 };
                    if (instruction.Destinations.Count < dwordCount)
                    {
                        error = "missing LDS read operand";
                        return false;
                    }

                    var address = Temp("uint", RawSource(instruction, 0));
                    var offset = control.SingleOffsetBytes;
                    if (TryForwardMeshLdsRead(instruction, address, offset)) return true;
                    for (var dword = 0; dword < dwordCount; dword++)
                    {
                        StoreVector(
                            instruction.Destinations[dword].Value,
                            $"sharpemu_lds[{LdsIndex(address, offset + (uint)(dword * sizeof(uint)))}]");
                    }

                    return true;
                }
                case "DsRead2B64":
                case "DsRead2St64B64":
                    return TryEmitDataShareReadPair64(instruction, control, out error);
                case "DsRead2B32":
                case "DsRead2St64B32":
                {
                    if (instruction.Destinations.Count < 2)
                    {
                        error = "missing LDS read2 operand";
                        return false;
                    }

                    var st64 = instruction.Opcode == "DsRead2St64B32";
                    var address = Temp("uint", RawSource(instruction, 0));
                    StoreVector(
                        instruction.Destinations[0].Value,
                        $"sharpemu_lds[{LdsIndex(address, EffectiveDsPairOffsetBytes(control.Offset0, st64))}]");
                    StoreVector(
                        instruction.Destinations[1].Value,
                        $"sharpemu_lds[{LdsIndex(address, EffectiveDsPairOffsetBytes(control.Offset1, st64))}]");
                    return true;
                }
                case "DsMskorB32":
                    return TryEmitDataShareAtomic(instruction, control, out error);
                case "DsCondxchg32RtnB64":
                {
                    // Two conditional exchanges, one per dword of the qword (the GFX9 ISA's
                    // DS_CONDXCHG32_RTN_B64; RDNA2 only lists it): a DATA dword with bit 31
                    // set stores its bits 30:0, and both pre-operation dwords are returned.
                    // Each dword is its own atomic, so in a program without LDS qword locks a
                    // concurrent writer can land between them; with them both run under the
                    // one lock of their qword.
                    if (instruction.Sources.Count < 3 || instruction.Destinations.Count < 2)
                    {
                        error = $"missing LDS operands for {instruction.Opcode}";
                        return false;
                    }

                    var address = Temp("uint", RawSource(instruction, 0));
                    var lowIndex = Temp("uint", LdsIndex(address, control.SingleOffsetBytes));
                    var highIndex = Temp("uint", LdsIndex(address, control.SingleOffsetBytes + sizeof(uint)));
                    var low = Temp("uint", RawSource(instruction, 1));
                    var high = Temp("uint", RawSource(instruction, 2));
                    var originalLow = Temp("uint", "0u");
                    var originalHigh = Temp("uint", "0u");
                    BeginExecGuard();
                    void Exchange()
                    {
                        Line($"{originalLow} = {EmitDataShareAtomicWord("Condxchg", lowIndex, low, "0u")};");
                        Line($"{originalHigh} = {EmitDataShareAtomicWord("Condxchg", highIndex, high, "0u")};");
                    }

                    if (UsesLdsQwordLocks)
                    {
                        EmitLdsLocked(lowIndex, Exchange);
                    }
                    else
                    {
                        Exchange();
                    }

                    Line($"v[{instruction.Destinations[0].Value}] = {originalLow};");
                    Line($"v[{instruction.Destinations[1].Value}] = {originalHigh};");
                    EndExecGuard();
                    return true;
                }
                default:
                    if (Gen5ShaderTranslator.IsDataShareAtomic(instruction.Opcode))
                    {
                        return TryEmitDataShareAtomic(instruction, control, out error);
                    }

                    error = $"unsupported LDS opcode {instruction.Opcode}";
                    return false;
            }
        }

        private static uint EffectiveDsPairOffsetBytes(uint offset, bool st64) =>
            offset * (st64 ? 256u : sizeof(uint));

        // DS_WRITE_B8/B16 and their D16_HI forms (which store DATA0 bits [23:16] or
        // [31:16]): the byte or short merges into the LDS word it lands in, and a short at
        // byte 3 of a word spills its high byte into the next. Threadgroup LDS merges with a
        // compare-and-swap, so a lane writing a neighbouring byte of the same word at the
        // same time is not lost; the private array of a graphics stage merges in place.
        private bool TryEmitDataShareSubwordWrite(
            Gen5ShaderInstruction instruction,
            Gen5DataShareControl control,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 2)
            {
                error = $"missing LDS operands for {instruction.Opcode}";
                return false;
            }

            var bytes = instruction.Opcode.StartsWith("DsWriteB8", StringComparison.Ordinal) ? 1u : 2u;
            var mask = bytes == 1 ? 0xFFu : 0xFFFFu;
            var sourceShift = instruction.Opcode.EndsWith("D16Hi", StringComparison.Ordinal) ? 16 : 0;
            var address = Temp("uint", RawSource(instruction, 0));
            var byteAddress = Temp("uint", $"{address} + {control.SingleOffsetBytes}u");
            var data = RawSource(instruction, 1);
            var value = Temp("uint", sourceShift == 0 ? $"{data} & 0x{mask:X}u" : $"({data} >> {sourceShift}u) & 0x{mask:X}u");
            var shift = Temp("uint", $"({byteAddress} & 3u) << 3u");
            BeginExecGuard();
            EmitLdsMaskedWrite(LdsIndex(byteAddress, 0), $"{value} << {shift}", $"0x{mask:X}u << {shift}");
            if (bytes == 2)
            {
                var spills = Temp("bool", $"{shift} == 24u");
                EmitLdsMaskedWrite(
                    LdsIndex(byteAddress, sizeof(uint)),
                    $"{spills} ? ({value} >> 8u) : 0u",
                    $"{spills} ? 0xFFu : 0u");
            }

            EndExecGuard();
            return true;
        }

        // Merges the masked bits of one LDS word; a zero mask leaves the word alone.
        private void EmitLdsMaskedWrite(string index, string value, string mask)
        {
            var wordIndex = Temp("uint", index);
            var bits = Temp("uint", value);
            var keep = Temp("uint", mask);
            if (!HasSharedLds)
            {
                Line($"sharpemu_lds[{wordIndex}] = (sharpemu_lds[{wordIndex}] & ~{keep}) | ({bits} & {keep});");
                return;
            }

            Line($"if ({keep} != 0u)");
            Line("{");
            _indent++;
            var pointer = Temp($"{LdsAddressSpace} atomic_uint*", $"({LdsAddressSpace} atomic_uint*)&sharpemu_lds[{wordIndex}]");
            var expected = Temp("uint", $"atomic_load_explicit({pointer}, memory_order_relaxed)");
            Line($"while (!atomic_compare_exchange_weak_explicit({pointer}, &{expected}, ({expected} & ~{keep}) | ({bits} & {keep}), memory_order_relaxed, memory_order_relaxed))");
            Line("{");
            Line("}");
            _indent--;
            Line("}");
        }

        // DS_READ_U8/I8/U16/I16 and the D16 forms: the byte or short at the address, read
        // from the word it lands in (a short at byte 3 takes its high byte from the next
        // word), zero or sign extended. The D16 forms extend to 16 bits and write the low
        // (or, for _HI, the high) half of the destination, keeping the other half.
        private bool TryEmitDataShareSubwordRead(
            Gen5ShaderInstruction instruction,
            Gen5DataShareControl control,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 1 || instruction.Destinations.Count < 1)
            {
                error = $"missing LDS operands for {instruction.Opcode}";
                return false;
            }

            var opcode = instruction.Opcode;
            var bits = opcode.Contains("16", StringComparison.Ordinal) &&
                !opcode.StartsWith("DsReadU8", StringComparison.Ordinal) &&
                !opcode.StartsWith("DsReadI8", StringComparison.Ordinal) ? 16u : 8u;
            var signExtend = opcode.StartsWith("DsReadI", StringComparison.Ordinal);
            var d16 = opcode.Contains("D16", StringComparison.Ordinal);
            var d16High = opcode.EndsWith("D16Hi", StringComparison.Ordinal);
            var address = Temp("uint", RawSource(instruction, 0));
            var byteAddress = Temp("uint", $"{address} + {control.SingleOffsetBytes}u");
            var shift = Temp("uint", $"({byteAddress} & 3u) << 3u");
            var low = Temp("uint", $"sharpemu_lds[{LdsIndex(byteAddress, 0)}]");
            var raw = bits == 8
                ? Temp("uint", $"({low} >> {shift}) & 0xFFu")
                : Temp(
                    "uint",
                    $"(uint)(((ulong){low} | ((ulong)sharpemu_lds[{LdsIndex(byteAddress, sizeof(uint))}] << 32)) >> {shift}) & 0xFFFFu");
            var extended = signExtend
                ? Temp("uint", $"(uint)extract_bits(as_type<int>({raw}), 0u, {bits}u)")
                : raw;
            var destination = instruction.Destinations[0].Value;
            StoreVector(
                destination,
                !d16 ? extended
                : d16High ? $"(v[{destination}] & 0x0000FFFFu) | (({extended} & 0xFFFFu) << 16)"
                : $"(v[{destination}] & 0xFFFF0000u) | ({extended} & 0xFFFFu)");
            return true;
        }

        // LDS atomics. Compute and mesh stages back LDS with real threadgroup
        // memory, so each guest atomic becomes a threadgroup atomic over the same
        // word. Graphics stages model one logical lane over a private LDS array
        // (the SPIR-V translator's Private-array fallback), where a plain
        // read-modify-write already has the atomic's semantics.
        private bool TryEmitDataShareAtomic(
            Gen5ShaderInstruction instruction,
            Gen5DataShareControl control,
            out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            if (instruction.Sources.Count < 1)
            {
                error = $"missing LDS address operand for {opcode}";
                return false;
            }

            if (Gen5ShaderTranslator.IsDataShareAtomic64(opcode))
            {
                return TryEmitDataShareAtomic64(instruction, control, out error);
            }

            if (opcode is "DsWrxchg2RtnB32" or "DsWrxchg2St64RtnB32")
            {
                return TryEmitDataShareExchangePair(instruction, control, out error);
            }

            // DS_MSKOR, DS_WRAP and DS_MIN/MAX_F32 take a second data register; DS_CMPST
            // reverses the buffer order and reads DATA0 as the comparator.
            var takesThreeSources = Gen5ShaderTranslator.IsDataShareTwoOperandAtomic(opcode);
            if (instruction.Sources.Count < (takesThreeSources ? 3 : 2))
            {
                error = $"missing LDS operands for {opcode}";
                return false;
            }

            var name = opcode switch
            {
                "DsMskorB32" or "DsMskorRtnB32" => "Mskor",
                "DsMinF32" or "DsMinRtnF32" => "Fmin",
                "DsMaxF32" or "DsMaxRtnF32" => "Fmax",
                "DsRsubU32" or "DsRsubRtnU32" => "Rsub",
                "DsAddF32" or "DsAddRtnF32" => "Fadd",
                "DsCmpstF32" or "DsCmpstRtnF32" => "Fcmpswap",
                "DsWrapRtnB32" => "Wrap",
                _ => DataShareAtomicName(opcode),
            };
            if (name.Length == 0)
            {
                error = $"unsupported LDS opcode {opcode}";
                return false;
            }

            var address = Temp("uint", RawSource(instruction, 0));
            var index = Temp("uint", LdsIndex(address, control.SingleOffsetBytes));
            var first = Temp("uint", RawSource(instruction, 1));
            var second = takesThreeSources ? Temp("uint", RawSource(instruction, 2)) : "0u";
            BeginExecGuard();
            var original = EmitLockedDataShareAtomicWord(name, index, first, second);
            if (instruction.Destinations.Count > 0)
            {
                Line($"v[{instruction.Destinations[0].Value}] = {original};");
            }

            EndExecGuard();
            return true;
        }

        // EmitDataShareAtomicWord inside the lock of the word's qword when the program
        // locks its shared LDS (UsesLdsQwordLocks).
        private string EmitLockedDataShareAtomicWord(string name, string index, string first, string second)
        {
            if (!UsesLdsQwordLocks)
            {
                return EmitDataShareAtomicWord(name, index, first, second);
            }

            var original = Temp("uint", "0u");
            EmitLdsLocked(index, () => Line($"{original} = {EmitDataShareAtomicWord(name, index, first, second)};"));
            return original;
        }

        // Emits the atomic over one LDS word and names the pre-operation value.
        // "first" is DATA0, "second" DATA1 for the three-operand forms.
        private string EmitDataShareAtomicWord(string name, string index, string first, string second)
        {
            var word = $"sharpemu_lds[{index}]";
            if (!HasSharedLds)
            {
                // One logical lane over private scratch: read, compute, write back.
                var observed = Temp("uint", word);
                var replacement = name switch
                {
                    "Add" => $"{observed} + {first}",
                    "Sub" => $"{observed} - {first}",
                    "Inc" => $"{observed} >= {first} ? 0u : {observed} + 1u",
                    "Dec" => $"({observed} == 0u || {observed} > {first}) ? {first} : {observed} - 1u",
                    "Smin" => $"(uint)min(as_type<int>({observed}), as_type<int>({first}))",
                    "Smax" => $"(uint)max(as_type<int>({observed}), as_type<int>({first}))",
                    "Umin" => $"min({observed}, {first})",
                    "Umax" => $"max({observed}, {first})",
                    "And" => $"{observed} & {first}",
                    "Or" => $"{observed} | {first}",
                    "Xor" => $"{observed} ^ {first}",
                    "Swap" => first,
                    "Cmpswap" => $"{observed} == {first} ? {second} : {observed}",
                    "Mskor" => $"({observed} & ~{first}) | {second}",
                    "Fmin" => DataShareFloatMinMax(false, observed, first),
                    "Fmax" => DataShareFloatMinMax(true, observed, first),
                    _ => DataShareCompareExchangeValue(name, observed, first, second),
                };
                Line($"{word} = {replacement};");
                return observed;
            }

            var pointer = $"({LdsAddressSpace} atomic_uint*)&{word}";
            var direct = name switch
            {
                "Add" => $"atomic_fetch_add_explicit({pointer}, {first}, memory_order_relaxed)",
                "Sub" => $"atomic_fetch_sub_explicit({pointer}, {first}, memory_order_relaxed)",
                "Umin" => $"atomic_fetch_min_explicit({pointer}, {first}, memory_order_relaxed)",
                "Umax" => $"atomic_fetch_max_explicit({pointer}, {first}, memory_order_relaxed)",
                "Smin" => $"(uint)atomic_fetch_min_explicit(({LdsAddressSpace} atomic_int*)&{word}, as_type<int>({first}), memory_order_relaxed)",
                "Smax" => $"(uint)atomic_fetch_max_explicit(({LdsAddressSpace} atomic_int*)&{word}, as_type<int>({first}), memory_order_relaxed)",
                "And" => $"atomic_fetch_and_explicit({pointer}, {first}, memory_order_relaxed)",
                "Or" => $"atomic_fetch_or_explicit({pointer}, {first}, memory_order_relaxed)",
                "Xor" => $"atomic_fetch_xor_explicit({pointer}, {first}, memory_order_relaxed)",
                "Swap" => $"atomic_exchange_explicit({pointer}, {first}, memory_order_relaxed)",
                _ => string.Empty,
            };
            if (direct.Length > 0)
            {
                return Temp("uint", direct);
            }

            // The remaining forms have no single Metal atomic: they read, derive
            // the replacement from what they observed and swap it in, retrying
            // until the word they computed from is still the one in memory.
            var pointerName = Temp($"{LdsAddressSpace} atomic_uint*", pointer);
            var expected = Temp("uint", $"atomic_load_explicit({pointerName}, memory_order_relaxed)");
            var next = name switch
            {
                "Cmpswap" => $"{expected} == {first} ? {second} : {expected}",
                // DS_INC/DEC_U32 wrap against DATA0, so they are not a plain add/sub of one.
                "Inc" => $"{expected} >= {first} ? 0u : {expected} + 1u",
                "Dec" => $"({expected} == 0u || {expected} > {first}) ? {first} : {expected} - 1u",
                "Mskor" => $"({expected} & ~{first}) | {second}",
                // DS_MIN/MAX_F32 keep the smaller/larger of DATA0 and memory; see
                // DataShareFloatMinMax for why DATA1 is not the compare operand.
                "Fmin" => DataShareFloatMinMax(false, expected, first),
                "Fmax" => DataShareFloatMinMax(true, expected, first),
                _ => DataShareCompareExchangeValue(name, expected, first, second),
            };
            Line($"while (!atomic_compare_exchange_weak_explicit({pointerName}, &{expected}, {next}, memory_order_relaxed, memory_order_relaxed))");
            Line("{");
            Line("}");
            return expected;
        }

        // DS_MIN/MAX_F32 (and the F64 forms): memory becomes DATA0 when DATA0 orders below
        // (above) it, a NaN on either side keeping memory as BUFFER_ATOMIC_FMIN/FMAX do.
        // The ISA's pseudo-code names a separate compare operand (DATA2), but LLVM encodes
        // these with DATA0 alone (DATA1 is v0 or garbage), radeonsi and RADV lower float
        // atomicMin/Max to them, and the PS5 emulators used as references model a plain
        // min/max of DATA0; so DATA0 is the operand, and DATA1 is read but unused.
        private static string DataShareFloatMinMax(bool max, string observed, string value) => max
            ? $"as_type<float>({value}) > as_type<float>({observed}) ? {value} : {observed}"
            : $"as_type<float>({value}) < as_type<float>({observed}) ? {value} : {observed}";

        // The replacement value of the LDS atomics that only a compare-and-swap loop (or the
        // private array's plain read-modify-write) expresses, from the observed word.
        private static string DataShareCompareExchangeValue(string name, string observed, string first, string second) => name switch
        {
            // DS_RSUB: memory = DATA0 - memory.
            "Rsub" => $"{first} - {observed}",
            // DS_ADD_F32: an IEEE single-precision add.
            "Fadd" => $"as_type<uint>(as_type<float>({observed}) + as_type<float>({first}))",
            // DS_CMPST_F32 compares as floats (so -0 matches +0 and NaN matches nothing),
            // with DS_CMPST's DATA0 comparator and DATA1 new value.
            "Fcmpswap" => $"as_type<float>({observed}) == as_type<float>({first}) ? {second} : {observed}",
            // DS_CONDXCHG32_RTN_B64, per dword: DATA with bit 31 set writes its bits 30:0.
            "Condxchg" => $"({first} & 0x80000000u) != 0u ? ({first} & 0x7FFFFFFFu) : {observed}",
            // DS_WRAP_RTN_B32: memory >= DATA0 ? memory - DATA0 : memory + DATA1.
            _ => $"{observed} >= {first} ? {observed} - {first} : {observed} + {second}",
        };

        // DS_WRXCHG2(ST64)_RTN_B32: two independent exchanges, DATA0 at OFFSET0 and DATA1 at
        // OFFSET1 (dword units, 64 dwords for ST64), each returning its pre-exchange word.
        private bool TryEmitDataShareExchangePair(
            Gen5ShaderInstruction instruction,
            Gen5DataShareControl control,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 3 || instruction.Destinations.Count < 2)
            {
                error = $"missing LDS operands for {instruction.Opcode}";
                return false;
            }

            var st64 = instruction.Opcode == "DsWrxchg2St64RtnB32";
            var address = Temp("uint", RawSource(instruction, 0));
            var firstIndex = Temp("uint", LdsIndex(address, EffectiveDsPairOffsetBytes(control.Offset0, st64)));
            var secondIndex = Temp("uint", LdsIndex(address, EffectiveDsPairOffsetBytes(control.Offset1, st64)));
            var first = Temp("uint", RawSource(instruction, 1));
            var second = Temp("uint", RawSource(instruction, 2));
            BeginExecGuard();
            var firstOriginal = EmitLockedDataShareAtomicWord("Swap", firstIndex, first, "0u");
            var secondOriginal = EmitLockedDataShareAtomicWord("Swap", secondIndex, second, "0u");
            Line($"v[{instruction.Destinations[0].Value}] = {firstOriginal};");
            Line($"v[{instruction.Destinations[1].Value}] = {secondOriginal};");
            EndExecGuard();
            return true;
        }

        // The 64-bit LDS atomics, over a pair of words in the uint LDS array.
        //
        // Graphics stages give each invocation its own LDS array (one logical lane), so a
        // plain 64-bit read-modify-write is the atomic and every form runs exactly.
        //
        // Compute and mesh LDS is threadgroup memory (or, for a large mesh allocation, a
        // device pool slot), whose Metal atomics are 32-bit only. A program whose 64-bit
        // LDS atomics are all non-returning ADD/SUB/AND/OR/XOR runs each as a 32-bit atomic
        // per half - OR/AND/XOR are bitwise independent, ADD/SUB fold each lane's own carry
        // or borrow out of the low word into the high one - so memory ends at the value the
        // 64-bit atomics would leave however the lanes interleave. Any other 64-bit form
        // makes the program take the LDS qword locks (UsesLdsQwordLocks, EmitLdsLocked):
        // every LDS atomic of it, 32- or 64-bit, then runs inside the lock of the qword it
        // touches, the 64-bit ones as one load, compute and store of both words.
        private bool TryEmitDataShareAtomic64(
            Gen5ShaderInstruction instruction,
            Gen5DataShareControl control,
            out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            var returns = opcode.Contains("Rtn", StringComparison.Ordinal);
            var operation = DataShareQwordOperation(opcode);
            var twoOperands = Gen5ShaderTranslator.IsDataShareTwoOperandAtomic(opcode);
            if (operation.Length == 0)
            {
                error = $"unsupported LDS opcode {opcode}";
                return false;
            }

            if (instruction.Sources.Count < (twoOperands ? 5 : 3) || (returns && instruction.Destinations.Count < (operation == "Wrxchg2" ? 4 : 2)))
            {
                error = $"missing LDS operands for {opcode}";
                return false;
            }

            _usesFloat64Atomics |= operation is "CmpstF" or "MinF" or "MaxF";
            var address = Temp("uint", RawSource(instruction, 0));
            var data0 = Temp("ulong", $"(ulong){RawSource(instruction, 1)} | ((ulong){RawSource(instruction, 2)} << 32)");
            var data1 = twoOperands
                ? Temp("ulong", $"(ulong){RawSource(instruction, 3)} | ((ulong){RawSource(instruction, 4)} << 32)")
                : "0ul";
            if (HasSharedLds && !UsesLdsQwordLocks)
            {
                // Only the splittable forms reach here (see UsesLdsQwordLocks).
                var lowIndex = Temp("uint", LdsIndex(address, control.SingleOffsetBytes));
                var highIndex = Temp("uint", LdsIndex(address, control.SingleOffsetBytes + sizeof(uint)));
                var low = Temp("uint", $"(uint){data0}");
                var high = Temp("uint", $"(uint)({data0} >> 32)");
                BeginExecGuard();
                switch (operation)
                {
                    case "Add":
                    {
                        var originalLow = EmitDataShareAtomicWord("Add", lowIndex, low, "0u");
                        var carry = Temp("uint", $"({originalLow} + {low}) < {originalLow} ? 1u : 0u");
                        EmitDataShareAtomicWord("Add", highIndex, Temp("uint", $"{high} + {carry}"), "0u");
                        break;
                    }
                    case "Sub":
                    {
                        var originalLow = EmitDataShareAtomicWord("Sub", lowIndex, low, "0u");
                        var borrow = Temp("uint", $"{originalLow} < {low} ? 1u : 0u");
                        EmitDataShareAtomicWord("Sub", highIndex, Temp("uint", $"{high} + {borrow}"), "0u");
                        break;
                    }
                    default:
                        EmitDataShareAtomicWord(operation, lowIndex, low, "0u");
                        EmitDataShareAtomicWord(operation, highIndex, high, "0u");
                        break;
                }

                EndExecGuard();
                return true;
            }

            // Read the pair, compute, write it back: in place for the private array, inside
            // the qword's lock with 32-bit atomic loads and stores for shared LDS.
            var st64 = opcode == "DsWrxchg2St64RtnB64";
            var pairs = operation == "Wrxchg2"
                ? new[] { (Offset: control.Offset0 * (st64 ? 512u : 8u), Value: data0), (Offset: control.Offset1 * (st64 ? 512u : 8u), Value: data1) }
                : new[] { (Offset: control.SingleOffsetBytes, Value: data0) };
            var originals = new List<string>();
            BeginExecGuard();
            foreach (var (offset, value) in pairs)
            {
                var lowIndex = Temp("uint", LdsIndex(address, offset));
                var highIndex = Temp("uint", LdsIndex(address, offset + sizeof(uint)));
                var observed = Temp("ulong", "0ul");
                void ReadModifyWrite()
                {
                    var next = Temp("ulong", DataShareQwordResult(operation, observed, value, data1));
                    if (HasSharedLds)
                    {
                        Line(LdsWordStore(lowIndex, $"(uint){next}"));
                        Line(LdsWordStore(highIndex, $"(uint)({next} >> 32)"));
                    }
                    else
                    {
                        Line($"sharpemu_lds[{lowIndex}] = (uint){next};");
                        Line($"sharpemu_lds[{highIndex}] = (uint)({next} >> 32);");
                    }
                }

                if (HasSharedLds)
                {
                    EmitLdsLocked(lowIndex, () =>
                    {
                        Line($"{observed} = (ulong){LdsWordLoad(lowIndex)} | ((ulong){LdsWordLoad(highIndex)} << 32);");
                        ReadModifyWrite();
                    });
                }
                else
                {
                    Line($"{observed} = (ulong)sharpemu_lds[{lowIndex}] | ((ulong)sharpemu_lds[{highIndex}] << 32);");
                    ReadModifyWrite();
                }

                originals.Add(observed);
            }

            if (returns)
            {
                for (var pair = 0; pair < originals.Count; pair++)
                {
                    Line($"v[{instruction.Destinations[2 * pair].Value}] = (uint){originals[pair]};");
                    Line($"v[{instruction.Destinations[(2 * pair) + 1].Value}] = (uint)({originals[pair]} >> 32);");
                }
            }

            EndExecGuard();
            return true;
        }

        private static string DataShareQwordOperation(string opcode) =>
            opcode.Replace("Rtn", string.Empty, StringComparison.Ordinal) switch
            {
                "DsAddU64" => "Add",
                "DsSubU64" => "Sub",
                "DsRsubU64" => "Rsub",
                "DsIncU64" => "Inc",
                "DsDecU64" => "Dec",
                "DsMinI64" => "MinI",
                "DsMaxI64" => "MaxI",
                "DsMinU64" => "MinU",
                "DsMaxU64" => "MaxU",
                "DsAndB64" => "And",
                "DsOrB64" => "Or",
                "DsXorB64" => "Xor",
                "DsMskorB64" => "Mskor",
                "DsWrxchgB64" => "Wrxchg",
                "DsWrxchg2B64" or "DsWrxchg2St64B64" => "Wrxchg2",
                "DsCmpstB64" => "Cmpst",
                "DsCmpstF64" => "CmpstF",
                "DsMinF64" => "MinF",
                "DsMaxF64" => "MaxF",
                _ => string.Empty,
            };

        // The qword a 64-bit DS atomic leaves (RDNA2 ISA 12.13) from the one it observed;
        // "value" is DATA0, "data1" DATA1 for the two-operand forms.
        private static string DataShareQwordResult(string operation, string observed, string value, string data1) => operation switch
        {
            "Add" => $"{observed} + {value}",
            "Sub" => $"{observed} - {value}",
            "Rsub" => $"{value} - {observed}",
            "Inc" => $"{observed} >= {value} ? 0ul : {observed} + 1ul",
            "Dec" => $"({observed} == 0ul || {observed} > {value}) ? {value} : {observed} - 1ul",
            "MinI" => $"as_type<ulong>(min(as_type<long>({observed}), as_type<long>({value})))",
            "MaxI" => $"as_type<ulong>(max(as_type<long>({observed}), as_type<long>({value})))",
            "MinU" => $"min({observed}, {value})",
            "MaxU" => $"max({observed}, {value})",
            "And" => $"{observed} & {value}",
            "Or" => $"{observed} | {value}",
            "Xor" => $"{observed} ^ {value}",
            // DS_MSKOR: DATA0 is the mask cleared, DATA1 the bits set.
            "Mskor" => $"({observed} & ~{value}) | {data1}",
            "Wrxchg" or "Wrxchg2" => value,
            // DS_CMPST: DATA0 is the comparator, DATA1 the new value; the F64 form compares
            // as doubles (a NaN equals nothing, -0 equals +0).
            "Cmpst" => $"{observed} == {value} ? {data1} : {observed}",
            "CmpstF" => $"sharpemu_f64_eq({observed}, {value}) ? {data1} : {observed}",
            // DS_MIN/MAX_F64 keep the smaller/larger of DATA0 and memory (see
            // DataShareFloatMinMax for why DATA0 and not DATA1).
            "MinF" => $"sharpemu_f64_lt({value}, {observed}) ? {value} : {observed}",
            "MaxF" => $"sharpemu_f64_lt({observed}, {value}) ? {value} : {observed}",
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "not a 64-bit DS atomic"),
        };

        private string LdsWordLoad(string index) =>
            $"atomic_load_explicit(({LdsAddressSpace} atomic_uint*)&sharpemu_lds[{index}], memory_order_relaxed)";

        private string LdsWordStore(string index, string value) =>
            $"atomic_store_explicit(({LdsAddressSpace} atomic_uint*)&sharpemu_lds[{index}], {value}, memory_order_relaxed);";

        /// <summary>
        /// True when this program's shared LDS needs the qword locks: it has a 64-bit LDS
        /// atomic that is not a non-returning ADD/SUB/AND/OR/XOR (those split exactly, see
        /// TryEmitDataShareAtomic64). Then every LDS atomic of the program takes the lock of
        /// the qword it touches, so a 32-bit atomic on one half never lands between a locked
        /// 64-bit read and write. Plain DS_READ/DS_WRITE do not lock: a 64-bit read may see
        /// one half from before a concurrent 64-bit atomic and one from after.
        /// </summary>
        private bool UsesLdsQwordLocks =>
            _usesLdsQwordLocks ??= HasSharedLds && _request.Program.Instructions.Any(instruction =>
                instruction.Control is Gen5DataShareControl { Gds: false } &&
                Gen5ShaderTranslator.IsDataShareAtomic64(instruction.Opcode) &&
                (instruction.Opcode.Contains("Rtn", StringComparison.Ordinal) ||
                 DataShareQwordOperation(instruction.Opcode) is not ("Add" or "Sub" or "And" or "Or" or "Xor")));

        private bool? _usesLdsQwordLocks;

        // The status a TFE buffer load writes after its data, when the access path computed
        // one (see TryEmitBufferLoadWithStatus); null is resident.
        private string? _bufferTfeStatus;

        private bool TryEmitResolvedMemoryAccess(
            string opcode,
            int bindingIndex,
            BufferAddress byteAddress,
            uint sourceVectorRegister,
            uint destinationVectorRegister,
            uint dwordCount,
            bool glc,
            out string error)
        {
            error = string.Empty;
            if (opcode.StartsWith("BufferAtomic", StringComparison.Ordinal) ||
                opcode.StartsWith("GlobalAtomic", StringComparison.Ordinal))
            {
                var atomic = opcode.StartsWith("BufferAtomic", StringComparison.Ordinal)
                    ? opcode["BufferAtomic".Length..] : opcode["GlobalAtomic".Length..];
                // The X2 forms are one 64-bit atomic on a dword pair (EmitQwordAtomic).
                var words = atomic.EndsWith("X2", StringComparison.Ordinal) ? 2u : 1u;
                if (words == 2)
                {
                    atomic = atomic[..^2];
                }

                // MUBUF CMPSWAP reads the new value from VDATA and the comparator
                // from the register after it; SWAP reads VDATA and, with GLC,
                // replaces that same register with the pre-operation word
                // (RDNA2 sections 8.1/12.14).
                if (words == 2
                        ? !IsQwordAtomic(atomic)
                        : !TryGetAtomicExpression(atomic, $"v[{sourceVectorRegister}]", $"v[{sourceVectorRegister + 1}]", out _))
                {
                    error = $"unsupported memory opcode {opcode}";
                    return false;
                }

                BeginExecGuard();
                var bytes = BufferBytes(bindingIndex);
                var width = words * sizeof(uint);
                // A placed access's words land apart; each is checked and aligned at its own address.
                var wordAddresses = new string[words];
                for (uint word = 0; word < words; word++)
                {
                    wordAddresses[word] = byteAddress.Contiguous ? byteAddress.Base : byteAddress.At(word * sizeof(uint));
                }

                Line(byteAddress.Contiguous
                    ? $"if ({width}u <= {bytes} && {byteAddress.Base} <= {bytes} - {width}u && ({byteAddress.Base} & 3u) == 0u)"
                    : "if (" + string.Join(
                        " && ",
                        wordAddresses.Select(address => $"4u <= {bytes} && {address} <= {bytes} - 4u && ({address} & 3u) == 0u")) + ")");
                Line("{");
                _indent++;
                string WordPointer(uint word) => byteAddress.Contiguous
                    ? $"b{bindingIndex} + ({byteAddress.Base} >> 2) + {word}u"
                    : $"b{bindingIndex} + ({wordAddresses[word]} >> 2)";
                string Operate(uint word, string pointer)
                {
                    TryGetAtomicExpression(
                        atomic,
                        $"v[{sourceVectorRegister + word}]",
                        $"v[{sourceVectorRegister + 1}]",
                        out var build);
                    return build(pointer);
                }

                if (words == 2)
                {
                    // Every source register is read before a destination, which shares
                    // VDATA, is written.
                    var originals = new[] { Temp("uint", "0u"), Temp("uint", "0u") };
                    var first = Temp("device uint*", WordPointer(0));
                    var second = Temp("device uint*", WordPointer(1));
                    EmitQwordAtomic(
                        atomic,
                        first,
                        second,
                        QwordSource(sourceVectorRegister),
                        atomic is "Cmpswap" or "Fcmpswap" ? QwordSource(sourceVectorRegister + 2) : "0ul",
                        originals[0],
                        originals[1]);
                    if (glc)
                    {
                        Line($"v[{destinationVectorRegister}] = {originals[0]};");
                        Line($"v[{destinationVectorRegister + 1}] = {originals[1]};");
                    }
                }
                else
                {
                    var original = Temp("uint", Operate(0, WordPointer(0)));
                    if (glc)
                    {
                        Line($"v[{destinationVectorRegister}] = {original};");
                    }
                }

                _indent--;
                Line("}");
                if (glc)
                {
                    var cleared = string.Concat(
                        Enumerable.Range(0, (int)words)
                            .Select(word => $" v[{destinationVectorRegister + word}] = 0u;"));
                    Line($"else {{{cleared} }}");
                }

                EndExecGuard();
                return true;
            }

            if (opcode.StartsWith("GlobalStore", StringComparison.Ordinal) ||
                opcode.StartsWith("BufferStore", StringComparison.Ordinal) ||
                opcode.StartsWith("TBufferStore", StringComparison.Ordinal))
            {
                BeginExecGuard();
                if (TryGetSubdwordStoreInfo(opcode, out var storeBytes, out var sourceShift))
                {
                    var source = sourceShift == 0
                        ? $"v[{sourceVectorRegister}]"
                        : $"(v[{sourceVectorRegister}] >> {sourceShift})";
                    Line($"sharpemu_store_bytes(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress.At(0)}, {source}, {storeBytes}u);");
                }
                else if (dwordCount >= 2 && byteAddress.Contiguous && LocksPersistentRecords)
                {
                    // One record to other workgroups (Gen5MslCodegenOptions.PersistentRecordLocks).
                    EmitPairLocked($"b{bindingIndex} + (({byteAddress.Base}) >> 2)", () =>
                    {
                        for (uint index = 0; index < dwordCount; index++)
                        {
                            Line(StoreDwordStatement(bindingIndex, $"{byteAddress.Base} + {index * 4}u", $"v[{sourceVectorRegister + index}]"));
                        }
                    });
                }
                else
                {
                    for (uint index = 0; index < dwordCount; index++)
                    {
                        Line(StoreDwordStatement(
                            bindingIndex,
                            byteAddress.Contiguous ? $"{byteAddress.Base} + {index * 4}u" : byteAddress.At(index * 4),
                            $"v[{sourceVectorRegister + index}]"));
                    }
                }

                EndExecGuard();
                return true;
            }

            if (TryGetSubdwordLoadInfo(opcode, out var loadBytes, out var signExtend, out var d16, out var d16High))
            {
                _usesCoherentLoads |= glc;
                var loaded = Temp(
                    "uint",
                    $"sharpemu_load_bytes{(glc ? "_coherent" : string.Empty)}(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress.At(0)}, {loadBytes}u, {(signExtend ? "true" : "false")})");
                if (!d16)
                {
                    StoreVector(destinationVectorRegister, loaded);
                    return true;
                }

                // D16 loads merge into one half of the destination register.
                StoreVector(
                    destinationVectorRegister,
                    d16High
                        ? $"(v[{destinationVectorRegister}] & 0x0000FFFFu) | (({loaded} & 0xFFFFu) << 16)"
                        : $"(v[{destinationVectorRegister}] & 0xFFFF0000u) | ({loaded} & 0xFFFFu)");
                return true;
            }

            if (opcode.StartsWith("GlobalLoad", StringComparison.Ordinal) ||
                opcode.StartsWith("BufferLoad", StringComparison.Ordinal) ||
                opcode.StartsWith("TBufferLoad", StringComparison.Ordinal))
            {
                _usesCoherentLoads |= glc;
                string DwordAddress(uint index) =>
                    byteAddress.Contiguous ? $"({byteAddress.Base} + {index * 4}u)" : byteAddress.At(index * 4);
                if (glc && dwordCount == 2)
                {
                    // A GLC LOAD_DWORDX2 is one snapshot of the pair (see EmitPairLocked).
                    StorePairLoad(
                        destinationVectorRegister,
                        $"sharpemu_load_pair_coherent(b{bindingIndex}, {BufferBytes(bindingIndex)}, {DwordAddress(0)}, {DwordAddress(1)}, {PairLockArguments})");
                    return true;
                }

                if (glc && dwordCount is 3 or 4 && byteAddress.Contiguous && LocksPersistentRecords)
                {
                    // One snapshot of the record (Gen5MslCodegenOptions.PersistentRecordLocks).
                    _usesPairLocks = true;
                    var record = Temp("uint4", "uint4(0u)");
                    var load = $"sharpemu_load_record_coherent(b{bindingIndex}, {BufferBytes(bindingIndex)}, {DwordAddress(0)}, {dwordCount}u, {PairLockArguments})";
                    Line(!_execKnownFull ? $"if (exec) {{ {record} = {load}; }}" : $"{record} = {load};");
                    for (uint index = 0; index < dwordCount; index++)
                    {
                        StoreVector(destinationVectorRegister + index, $"{record}[{index}]");
                    }

                    return true;
                }

                if (!glc && dwordCount > 1 && byteAddress.Contiguous && _options.VectorDwordLoads && _options.RawDwordAccess)
                {
                    // Every dword is loaded before the first destination is written, as the
                    // hardware reads its address operands once.
                    var runs = new List<string>();
                    for (uint first = 0; first < dwordCount; first += 4)
                    {
                        var load = LoadDwords(bindingIndex, DwordAddress(first), Math.Min(4u, dwordCount - first));
                        if (_execKnownFull)
                        {
                            runs.Add(Temp("uint4", load));
                            continue;
                        }

                        var run = Temp("uint4", "uint4(0u)");
                        Line($"if (exec) {{ {run} = {load}; }}");
                        runs.Add(run);
                    }

                    for (uint index = 0; index < dwordCount; index++)
                    {
                        StoreVector(destinationVectorRegister + index, $"{runs[(int)(index / 4)]}[{index % 4}]");
                    }

                    return true;
                }

                for (uint index = 0; index < dwordCount; index++)
                {
                    var address = DwordAddress(index);
                    StoreVector(
                        destinationVectorRegister + index,
                        glc
                            ? $"sharpemu_load_dword_coherent(b{bindingIndex}, {BufferBytes(bindingIndex)}, {address})"
                            : LoadDword(bindingIndex, address));
                }

                return true;
            }

            error = $"unsupported memory opcode {opcode}";
            return false;
        }

        private static bool TryGetSubdwordLoadInfo(
            string opcode,
            out uint byteCount,
            out bool signExtend,
            out bool d16,
            out bool d16High)
        {
            byteCount = opcode.Contains("byte", StringComparison.OrdinalIgnoreCase) ? 1u : 2u;
            signExtend = opcode.Contains("Sbyte", StringComparison.Ordinal) ||
                opcode.Contains("Sshort", StringComparison.Ordinal);
            d16 = opcode.Contains("D16", StringComparison.Ordinal);
            d16High = opcode.EndsWith("D16Hi", StringComparison.Ordinal);
            return opcode.Contains("LoadUbyte", StringComparison.Ordinal) ||
                opcode.Contains("LoadSbyte", StringComparison.Ordinal) ||
                opcode.Contains("LoadUshort", StringComparison.Ordinal) ||
                opcode.Contains("LoadSshort", StringComparison.Ordinal) ||
                opcode.Contains("LoadShortD16", StringComparison.Ordinal);
        }

        private static bool TryGetSubdwordStoreInfo(
            string opcode,
            out uint byteCount,
            out uint sourceShift)
        {
            byteCount = opcode.Contains("StoreByte", StringComparison.Ordinal) ? 1u : 2u;
            sourceShift = opcode.EndsWith("D16Hi", StringComparison.Ordinal) ? 16u : 0u;
            return opcode.Contains("StoreByte", StringComparison.Ordinal) ||
                opcode.Contains("StoreShort", StringComparison.Ordinal);
        }

        private static bool IsFormatBufferLoad(string opcode) =>
            opcode.StartsWith("BufferLoadFormat", StringComparison.Ordinal) ||
            opcode.StartsWith("TBufferLoadFormat", StringComparison.Ordinal);

        // BUFFER_STORE_FORMAT_* and TBUFFER_STORE_FORMAT_* encode through the format prelude.
        private static bool IsFormatBufferStore(string opcode) =>
            opcode.StartsWith("BufferStoreFormat", StringComparison.Ordinal) ||
            opcode.StartsWith("TBufferStoreFormat", StringComparison.Ordinal);

        private string BufferBytes(int bindingIndex) =>
            $"{ResourcesName}.buffer_bytes[{bindingIndex}]";

        /// <summary>A raw dword load, the form S_BUFFER_LOAD_DWORD* and
        /// BUFFER_LOAD_DWORD* take: the byte address truncates to the dword it lands in
        /// and the dword is read, out of range reading zero.
        ///
        /// This is what the SPIR-V backend does for the same opcodes, and it is the whole
        /// of it — a shift, a compare and a load. The byte-combining
        /// <see cref="LoadWord"/> below is for a format access, whose component really can
        /// straddle two dwords because the descriptor's stride is a byte count. Using it
        /// for raw dword loads cost an alignment branch and a four-iteration byte loop at
        /// each of them, which Octopath Traveler II's shaders do 1661 times, 65 of them in
        /// the dispatch that spends 19% of the frame's GPU time.</summary>
        private string LoadDword(int bindingIndex, string byteAddress) =>
            _options.RawDwordAccess
                ? $"sharpemu_load_dword(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress})"
                : LoadWord(bindingIndex, byteAddress);

        /// <summary>A run of one to four raw dwords from <paramref name="byteAddress"/> on, as a
        /// uint4 (see <see cref="Gen5MslCodegenOptions.VectorDwordLoads"/>).</summary>
        private string LoadDwords(int bindingIndex, string byteAddress, uint count) =>
            $"sharpemu_load_dwords<{count}>(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress})";

        private string StoreDwordStatement(int bindingIndex, string byteAddress, string value) =>
            _options.RawDwordAccess
                ? $"sharpemu_store_dword(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress}, {value});"
                : $"sharpemu_store_bytes(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress}, {value}, 4u);";

        private string LoadWord(int bindingIndex, string byteAddress) =>
            $"sharpemu_load_word(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress})";

        private string ApplyByteBias(int bindingIndex, string byteAddress) =>
            _options.ElideByteBias ? byteAddress : $"({byteAddress} + bias[{bindingIndex}])";

        // ---- writer helpers ----

        private void Line(string text)
        {
            for (var index = 0; index < _indent; index++)
            {
                _body.Append("    ");
            }

            _body.AppendLine(text);
        }

        private bool? _writesDeviceMemory;

        /// <summary>Whether any instruction stores to or atomically updates buffer or
        /// global memory. LDS (Ds*) stays in threadgroup memory and image writes go to
        /// textures, which the barrier's device fence never covered.</summary>
        private bool ProgramWritesDeviceMemory() =>
            _writesDeviceMemory ??= _request.Program.Instructions.Any(static instruction =>
                !instruction.Opcode.StartsWith("Ds", StringComparison.Ordinal) &&
                !instruction.Opcode.StartsWith("Image", StringComparison.Ordinal) &&
                // Scratch is thread-private (a thread-local array), never device memory.
                !instruction.Opcode.StartsWith("Scratch", StringComparison.Ordinal) &&
                (instruction.Opcode.Contains("Store", StringComparison.Ordinal) ||
                 instruction.Opcode.Contains("Atomic", StringComparison.Ordinal)));

        private string Temp(string type, string expression)
        {
            var name = $"t{_nextTemp++}";
            for (var index = 0; index < _indent; index++)
            {
                _body.Append("    ");
            }

            _body.Append(type).Append(' ').Append(name).Append(" = ").Append(expression).Append(';').AppendLine();
            return name;
        }

        /// <summary>The per-lane view after a write of the low dword of a mask pair. The lanes
        /// of the high half keep the view they had, which is its bit of the high dword (the view
        /// is kept in step with the pair), so the high dword is not read: a pair whose last
        /// ballot nothing else reads can stay unexchanged (Gen5MslTranslator.LaneMasks.cs).</summary>
        private string LowHalfWriteBit(string low, string view, string high) =>
            LaneMasksEnabled
                ? $"sharpemu_lane < 32u ? (((({low}) >> (sharpemu_lane & 31u)) & 1u) != 0u) : {view}"
                : LaneBit(low, high);

        /// <summary>The same for a write of the high dword.</summary>
        private string HighHalfWriteBit(string low, string high, string view) =>
            LaneMasksEnabled
                ? $"sharpemu_lane < 32u ? {view} : (((({high}) >> (sharpemu_lane & 31u)) & 1u) != 0u)"
                : LaneBit(low, high);

        /// <summary>This lane's bit of a 64-bit mask held as two dwords.
        ///
        /// A 64-lane guest wave runs as two 32-wide Apple simdgroups, so its lane index
        /// goes up to 63 while each half of the mask is a 32-bit word. Shifting a uint by
        /// 32 or more is undefined in MSL — in practice the count wraps, so lanes 32..63
        /// were reading the low half's bits and seeing another lane's EXEC. The half has
        /// to be selected before the shift.</summary>
        private string LaneBit(string low, string high) =>
            IsWave64
                ? $"(((sharpemu_lane < 32u ? ({low}) : ({high})) >> (sharpemu_lane & 31u)) & 1u) != 0u"
                : $"(({low}) >> sharpemu_lane & 1u) != 0u";

        /// <summary>True for the graphics stages, where a guest wave's lanes are only the
        /// host threads that are running: one lane in a single-lane stage, the fragments
        /// (and their helpers) Metal packed into the simdgroup in a pixel stage. Any
        /// other EXEC bit is a lane that does not exist - whole-quad mode, an all-ones
        /// move or a wave64 high half puts them there - and since no thread can ever
        /// clear it again, an EXECZ loop exit would wait on it for ever. Compute and
        /// mesh keep EXEC as written, as the SPIR-V translator does.</summary>
        private bool ClampsExecToLanes => IsSingleLaneStage || _stage == Gen5MslStage.Pixel;

        /// <summary>A new EXEC low dword limited to the lanes that exist (see
        /// <see cref="ClampsExecToLanes"/>). A pixel stage ballots the running lanes at
        /// every EXEC write rather than once at entry: helper threads may stop once the
        /// gradients they exist for are done (MSL spec 6.10.3), and a bit left for one
        /// would never clear. EXEC is only written at scalar points, where the whole
        /// simdgroup is converged.</summary>
        private string ClampExecToLanes(string expression) =>
            IsSingleLaneStage ? $"({expression}) & 1u"
            : ClampsExecToLanes ? $"({expression}) & sharpemu_ballot(true)"
            : expression;

        /// <summary>The 64-bit form of <see cref="ClampExecToLanes"/>, for a whole EXEC
        /// pair; the high half never has a lane in these stages.</summary>
        private string ClampExecToLanes64(string expression) =>
            IsSingleLaneStage ? $"({expression}) & 1ul"
            : ClampsExecToLanes ? $"({expression}) & (ulong)sharpemu_ballot(true)"
            : IsNarrowWave64 ? $"({expression}) & 0xFFFFFFFFul"
            : expression;

        // VCC (s106:s107) and EXEC (s126:s127) are architectural SGPRs: programs
        // freely use them as scratch data registers (s_buffer_load into s[106],
        // then v_rcp_f32 of that value is real RDNA2 code). The register file
        // holds their raw 32-bit values as the source of truth; the bools
        // vcc/exec are cached per-lane views kept in sync at every write so
        // control flow stays cheap. Reading them back as data returns the file.
        private void StoreScalar(uint register, string expression)
        {
            switch (register)
            {
                case VccLoRegister:
                {
                    var value = Temp("uint", expression);
                    Line($"s[{VccLoRegister}] = {value};");
                    Line($"vcc = {LowHalfWriteBit(value, "vcc", ScalarExpression(VccHiRegister))};");
                    return;
                }

                case ExecLoRegister:
                {
                    var value = Temp("uint", ClampExecToLanes(expression));
                    Line($"s[{ExecLoRegister}] = {value};");
                    Line($"exec = {LowHalfWriteBit(value, "exec", ScalarExpression(ExecHiRegister))};");
                    return;
                }

                case VccHiRegister:
                case ExecHiRegister:
                {
                    var value = Temp("uint", register == ExecHiRegister && (ClampsExecToLanes || IsNarrowWave64) ? "0u" : expression);
                    Line($"s[{register}] = {value};");
                    if (IsWave64)
                    {
                        // A 64-lane wave keeps lanes 32..63 here, so writing this half
                        // moves the mask under those lanes and the cached view has to
                        // follow. Wave32 carries no lanes here and only keeps the data.
                        var low = ScalarExpression(
                            register == VccHiRegister ? VccLoRegister : ExecLoRegister);
                        Line(register == VccHiRegister
                            ? $"vcc = {HighHalfWriteBit(low, value, "vcc")};"
                            : $"exec = {HighHalfWriteBit(low, value, "exec")};");
                    }

                    return;
                }
            }

            if (register < ScalarRegisterFileCount)
            {
                Line($"s[{register}] = {expression};");
            }
        }

        private void StoreVector(uint register, string expression, bool guardWithExec = true)
        {
            if (register >= VectorRegisterFileCount)
            {
                return;
            }

            if (guardWithExec && !_execKnownFull && !_unobservedAluWrites.Contains(register))
            {
                Line($"if (exec) {{ v[{register}] = {expression}; }}");
            }
            else
            {
                Line($"v[{register}] = {expression};");
            }
        }

        /// <summary>One VGPR by number, wrapped into the file the way a relative move's
        /// register arithmetic wraps.</summary>
        private static string VectorExpression(uint register) =>
            $"v[{register & (VectorRegisterFileCount - 1)}]";

        private string ScalarExpression(uint register) =>
            register < ScalarRegisterFileCount ? $"s[{register}]" : "0u";

        private string SourceExpression(
            Gen5Operand operand,
            Gen5ShaderInstruction instruction)
        {
            switch (operand.Kind)
            {
                case Gen5OperandKind.ScalarRegister:
                    return ScalarExpression(operand.Value);
                case Gen5OperandKind.VectorRegister:
                    return $"v[{operand.Value}]";
                case Gen5OperandKind.LiteralConstant:
                    return FormatUInt(operand.Value);
                case Gen5OperandKind.EncodedConstant:
                    // 251/252/253 read the VCCZ/EXECZ/SCC status bits as data.
                    if (operand.Value == 251)
                    {
                        return $"({VccZeroCondition} ? 1u : 0u)";
                    }

                    if (operand.Value == 252)
                    {
                        return $"((s[{ExecLoRegister}] | s[{ExecHiRegister}]) == 0u ? 1u : 0u)";
                    }

                    if (operand.Value == 253)
                    {
                        return "(scc ? 1u : 0u)";
                    }

                    if (Gen5InlineConstants.TryDecode(operand.Value, out var constant))
                    {
                        return FormatUInt(constant);
                    }

                    throw new NotSupportedException(
                        $"unsupported encoded constant {operand.Value} in {instruction.Opcode}");
                default:
                    throw new NotSupportedException($"unsupported operand kind {operand.Kind}");
            }
        }

        private static string FormatUInt(uint value) =>
            value <= 9 ? $"{value}u" : $"0x{value.ToString("X", CultureInfo.InvariantCulture)}u";

        private static string AsFloat(string expression) => $"as_type<float>({expression})";

        /// <summary>The uint bits of an expression. A value that was just reinterpreted
        /// as a float is handed back as it came: the register file is uint, so an
        /// instruction that reads a float and writes uint bits would otherwise emit
        /// as_type&lt;uint&gt;(as_type&lt;float&gt;(v0)) — a no-op the Metal compiler folds
        /// but which 120 lines of a game's shaders carry through the compiler first.</summary>
        private static string AsUInt(string expression)
        {
            const string cast = "as_type<float>(";
            if (expression.StartsWith(cast, StringComparison.Ordinal) &&
                expression.EndsWith(')') &&
                MatchingParenthesis(expression, cast.Length - 1) == expression.Length - 1)
            {
                return expression[cast.Length..^1];
            }

            return $"as_type<uint>({expression})";
        }

        /// <summary>The index of the parenthesis closing the one at
        /// <paramref name="open"/>, or -1.</summary>
        private static int MatchingParenthesis(string expression, int open)
        {
            var depth = 0;
            for (var index = open; index < expression.Length; index++)
            {
                depth += expression[index] == '(' ? 1 : expression[index] == ')' ? -1 : 0;
                if (depth == 0)
                {
                    return index;
                }
            }

            return -1;
        }

        // ---- basic blocks (ports BuildBasicBlocks from the SPIR-V translator) ----

        private readonly record struct ShaderBlock(
            uint StartPc,
            int StartIndex,
            int EndIndex);

        private static uint ColorOutputLocations(IReadOnlyList<Gen5PixelOutputBinding> bindings)
        {
            var mask = 0u;
            foreach (var binding in bindings)
            {
                if (binding.BlendSourceIndex == 0 && binding.HostLocation < 32)
                {
                    mask |= 1u << (int)binding.HostLocation;
                }
            }

            return mask;
        }

        private static IReadOnlyList<ShaderBlock> BuildBasicBlocks(
            IReadOnlyList<Gen5ShaderInstruction> instructions,
            bool splitAfterBarrier = false)
        {
            if (instructions.Count == 0)
            {
                return [];
            }

            var leaders = new SortedSet<uint> { instructions[0].Pc };
            for (var index = 0; index < instructions.Count; index++)
            {
                var instruction = instructions[index];
                if (IsBranch(instruction.Opcode) &&
                    TryGetBranchTargetPc(instruction, out var targetPc))
                {
                    leaders.Add(targetPc);
                }

                if ((IsBranch(instruction.Opcode) || instruction.Opcode == "SEndpgm" ||
                        (splitAfterBarrier && instruction.Opcode == "SBarrier")) &&
                    index + 1 < instructions.Count)
                {
                    leaders.Add(instructions[index + 1].Pc);
                }
            }

            var starts = new List<uint>(leaders.Count);
            foreach (var pc in leaders)
            {
                if (FindInstructionIndex(instructions, pc) >= 0)
                {
                    starts.Add(pc);
                }
            }

            var blocks = new List<ShaderBlock>(starts.Count);
            for (var index = 0; index < starts.Count; index++)
            {
                var startIndex = FindInstructionIndex(instructions, starts[index]);
                var endIndex = index + 1 < starts.Count
                    ? FindInstructionIndex(instructions, starts[index + 1])
                    : instructions.Count;
                if (startIndex >= 0 && endIndex > startIndex)
                {
                    blocks.Add(new ShaderBlock(starts[index], startIndex, endIndex));
                }
            }

            return blocks;
        }

        private static bool IsBranch(string opcode) =>
            opcode == "SBranch" ||
            opcode.StartsWith("SCbranch", StringComparison.Ordinal);

        private static bool TryGetBranchTargetPc(
            Gen5ShaderInstruction instruction,
            out uint targetPc)
        {
            targetPc = 0;
            if (instruction.Control is ShaderLinkedBranchControl linked)
            {
                targetPc = linked.TargetPc;
                return true;
            }
            if (instruction.Encoding != Gen5ShaderEncoding.Sopp ||
                instruction.Words.Count == 0)
            {
                return false;
            }

            var offset = unchecked((short)(instruction.Words[0] & 0xFFFF));
            var nextPc = (long)instruction.Pc +
                (instruction.Words.Count * sizeof(uint));
            var target = nextPc + (offset * sizeof(uint));
            if (target < 0 || target > uint.MaxValue)
            {
                return false;
            }

            targetPc = (uint)target;
            return true;
        }

        private static int FindInstructionIndex(
            IReadOnlyList<Gen5ShaderInstruction> instructions,
            uint pc)
        {
            for (var index = 0; index < instructions.Count; index++)
            {
                if (instructions[index].Pc == pc)
                {
                    return index;
                }
            }

            return -1;
        }

        private static bool TryFindBlock(
            IReadOnlyList<ShaderBlock> blocks,
            uint pc,
            out int block)
        {
            for (var index = 0; index < blocks.Count; index++)
            {
                if (blocks[index].StartPc == pc)
                {
                    block = index;
                    return true;
                }
            }

            block = -1;
            return false;
        }
    }
}
