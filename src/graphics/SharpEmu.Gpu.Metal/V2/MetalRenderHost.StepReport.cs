// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

// The dispatcher step cap, made observable. Every translated program bounds its
// while(active){switch(pc)} loop so a mistranslated loop-exit condition terminates the
// invocation instead of wedging the queue, but that bound used to be silent: a shader
// that spun to the cap only looked slow, and nothing said which one it was. The last
// field of every stage argument buffer is now a pointer to one word of this buffer,
// private to the program's hash, which the dispatcher increments when it gives up. The
// host reads the words back after each flush, so hitting the cap reads as
// [METAL-V2][STEP-CAP] with the hash to dump with SHARPEMU_SHADER_MSL_DUMP_HASH.
internal sealed partial class MetalRenderHost
{
    private const int StepReportSlots = 8192;

    // Each program's report is a counter followed by seven probe words a capped invocation
    // may fill (SHARPEMU_MSL_STEPCAP_PROBE).
    private const int StepReportStride = 8;
    private const int MaxStepCapReports = 64;
    private long _stepCapTotalTicks;

    // SHARPEMU_MSL_PC_PROBE in log mode (<hash>:<pc>:log:<regs>..., several separated by ';'): each
    // probed program's report pointer goes to its own region after the slots, a counter followed by
    // Gen5MslTranslator.PcProbeLogRecords records of one word per register. The records of the flushes
    // around a dump-trigger frame are appended to SHARPEMU_MSL_PC_PROBE_LOG as (count, registers |
    // probe index << 16) headers followed by the words.
    private static readonly (ulong Hash, int Registers)[] ProbeLogs = ParseProbeLogs();
    private static readonly string? ProbeLogPath = Environment.GetEnvironmentVariable("SHARPEMU_MSL_PC_PROBE_LOG");

    // SHARPEMU_MSL_PC_PROBE_LOG_FLUSHES=<n>: how many flushes after the dump flip are logged (default 8).
    private static readonly int ProbeLogFlushes =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_MSL_PC_PROBE_LOG_FLUSHES"), out var flushes) && flushes > 0
            ? flushes
            : 8;
    private int _probeLogFlushes;

    // One region per probed program; its records are Gen5MslTranslator.PcProbeLogRecordWords long
    // (the probe index, then the registers), so a program probed at several PCs keeps them apart.
    private static (ulong Hash, int Registers)[] ParseProbeLogs()
    {
        var logs = new List<(ulong, int)>();
        foreach (var spec in (Environment.GetEnvironmentVariable("SHARPEMU_MSL_PC_PROBE") ?? string.Empty)
            .Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = spec.Split(':');
            if (parts.Length is (4 or 5) && parts[2] == "log" &&
                ulong.TryParse(parts[0].Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var hash) &&
                !logs.Exists(log => log.Item1 == hash))
            {
                logs.Add((hash, Gen5MslTranslator.PcProbeLogRecordWords(hash)));
            }
        }

        return logs.ToArray();
    }

    private static int ProbeLogWords(int index) => 1 + Gen5MslTranslator.PcProbeLogRecords * ProbeLogs[index].Registers;

    private static int ProbeLogOffset(int index)
    {
        var offset = StepReportSlots * StepReportStride;
        for (var i = 0; i < index; i++)
        {
            offset += ProbeLogWords(i);
        }

        return offset;
    }

    private static int ProbeLogTotalWords => ProbeLogs.Length == 0 ? 0 : ProbeLogOffset(ProbeLogs.Length) - StepReportSlots * StepReportStride;

    private nint _stepReportBuffer;
    private nint _stepReportContents;
    private ulong _stepReportGpuAddress;
    private int _stepReportUsed;
    private int _stepCapReports;
    private readonly Dictionary<ulong, int> _stepReportSlotByHash = new();
    private readonly ulong[] _stepReportHashBySlot = new ulong[StepReportSlots];
    private readonly string[] _stepReportStageBySlot = new string[StepReportSlots];
    private readonly long[] _stepCapTotalBySlot = new long[StepReportSlots];

    /// <summary>
    /// The GPU address of the step-report word reserved for one program, or zero when
    /// no slot is left; the shader null-checks the pointer, so zero just disables the
    /// report for that program.
    /// </summary>
    private ulong StepReportAddress(ulong hash, string stage)
    {
        for (var index = 0; index < ProbeLogs.Length; index++)
        {
            if (ProbeLogs[index].Hash == hash)
            {
                EnsureStepReportBuffer();
                NoteUsedResource(_stepReportBuffer, Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
                return _stepReportGpuAddress + (ulong)(ProbeLogOffset(index) * sizeof(uint));
            }
        }

        if (!_stepReportSlotByHash.TryGetValue(hash, out var slot))
        {
            if (_stepReportUsed == StepReportSlots)
            {
                MetalV2Diagnostics.WarnOnce(
                    "step-report-full",
                    $"The step-report buffer has no slot left; later programs cannot report the " +
                    $"dispatcher step cap: slots={StepReportSlots}.");
                return 0;
            }

            EnsureStepReportBuffer();
            slot = _stepReportUsed++;
            _stepReportSlotByHash[hash] = slot;
            _stepReportHashBySlot[slot] = hash;
            _stepReportStageBySlot[slot] = stage;
        }

        NoteUsedResource(_stepReportBuffer, Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
        return _stepReportGpuAddress + (ulong)(slot * StepReportStride * sizeof(uint));
    }

    private void EnsureStepReportBuffer()
    {
        if (_stepReportBuffer != 0)
        {
            return;
        }

        _stepReportBuffer = _guestMemory.CreateReportBuffer(
            (ulong)((StepReportSlots * StepReportStride + ProbeLogTotalWords) * sizeof(uint)),
            "SharpEmu StepReport",
            out _stepReportContents,
            out _stepReportGpuAddress);
        StepReportWords.Clear();
        for (var index = 0; index < ProbeLogs.Length; index++)
        {
            ProbeLogWordsSpan(index).Clear();
        }
    }

    private unsafe Span<uint> StepReportWords =>
        _stepReportContents == 0
            ? []
            : new Span<uint>((void*)_stepReportContents, StepReportSlots * StepReportStride);

    private unsafe Span<uint> ProbeLogWordsSpan(int index) =>
        _stepReportContents == 0
            ? []
            : new Span<uint>((uint*)_stepReportContents + ProbeLogOffset(index), ProbeLogWords(index));

    private void DrainProbeLog()
    {
        if (ProbeLogs.Length == 0 || _stepReportContents == 0)
        {
            return;
        }

        if (_dumpThisFlip)
        {
            _probeLogFlushes = ProbeLogFlushes;
        }

        var write = _probeLogFlushes > 0 && ProbeLogPath is not null;
        if (_probeLogFlushes > 0)
        {
            _probeLogFlushes--;
        }

        for (var index = 0; index < ProbeLogs.Length; index++)
        {
            var words = ProbeLogWordsSpan(index);
            var count = Math.Min(words[0], (uint)Gen5MslTranslator.PcProbeLogRecords);
            words[0] = 0;
            if (count == 0 || !write)
            {
                continue;
            }

            using var file = new FileStream(ProbeLogPath!, FileMode.Append, FileAccess.Write);
            Span<uint> header = [count, (uint)ProbeLogs[index].Registers | ((uint)index << 16)];
            file.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(header));
            file.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.Slice(1, (int)count * ProbeLogs[index].Registers)));
        }
    }

    /// <summary>
    /// Reads back and clears the step-report words. Called once per flush: the counts
    /// may belong to the submission before last, which is immaterial for a diagnostic
    /// that only has to name the shader.
    /// </summary>
    private void ReportStepCaps()
    {
        DrainProbeLog();
        if (_stepReportUsed == 0 || _stepReportContents == 0)
        {
            return;
        }

        var words = StepReportWords;
        for (var slot = 0; slot < _stepReportUsed; slot++)
        {
            var hits = words[slot * StepReportStride];
            if (hits == 0)
            {
                continue;
            }

            words[slot * StepReportStride] = 0;
            var probe = words.Slice(slot * StepReportStride + 1, StepReportStride - 1);
            var probeText = probe.IndexOfAnyExcept(0u) < 0
                ? string.Empty
                : " probe=[" + string.Join(" ", probe.ToArray().Select(static word => $"{word:X8}")) + "]";
            probe.Clear();
            _stepCapTotalBySlot[slot] += hits;
            if (_stepCapReports >= MaxStepCapReports)
            {
                // Past the cap, a probed program's running total every two seconds says when it still runs.
                var now = System.Diagnostics.Stopwatch.GetTimestamp();
                if (now - _stepCapTotalTicks > 2 * System.Diagnostics.Stopwatch.Frequency)
                {
                    _stepCapTotalTicks = now;
                    Console.Error.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "[METAL-V2][STEP-CAP] running hash=0x{0:X16} total={1}", _stepReportHashBySlot[slot], _stepCapTotalBySlot[slot]));
                }

                continue;
            }

            _stepCapReports++;
            Console.Error.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "[METAL-V2][STEP-CAP] hash=0x{0:X16} stage={1} invocations={2} total={3}{5}{4}",
                _stepReportHashBySlot[slot],
                _stepReportStageBySlot[slot],
                hits,
                _stepCapTotalBySlot[slot],
                _stepCapReports == MaxStepCapReports ? " (further reports suppressed)" : string.Empty,
                probeText));
        }
    }

    private void DisposeStepReport()
    {
        if (_stepReportBuffer == 0)
        {
            return;
        }

        for (var slot = 0; slot < _stepReportUsed; slot++)
        {
            if (_stepCapTotalBySlot[slot] != 0)
            {
                Console.Error.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "[METAL-V2][STEP-CAP] final hash=0x{0:X16} stage={1} total={2}",
                    _stepReportHashBySlot[slot],
                    _stepReportStageBySlot[slot],
                    _stepCapTotalBySlot[slot]));
            }
        }

        MetalNative.SendVoid(_stepReportBuffer, MetalNative.Selector("release"));
        _stepReportBuffer = 0;
        _stepReportContents = 0;
        _stepReportGpuAddress = 0;
    }

    private static string StepReportStageName(int stageIndex) => stageIndex switch
    {
        StageVertex => "vertex",
        StageFragment => "fragment",
        _ => "compute",
    };
}
