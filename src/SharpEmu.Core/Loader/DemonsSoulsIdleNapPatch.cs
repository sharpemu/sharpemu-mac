// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Memory;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Lets Demon's Souls' idle job workers sleep. The title runs 13 job workers that look for work
/// in pure guest spins (ChooseJobToRun re-polling parked jobs, and WaitForJobs' pause loop),
/// with no kernel call on the way, so on a host every idle worker stays runnable and the
/// critical render job gets a fraction of a core. The patch puts three trampolines into
/// WaitForJobs' deadlock report, which is dead once its CheckForDeadlock result is ignored:
/// RunJob's wrapper stamps the worker's slot at [S+0xF00+8w] with the TSC at job end,
/// ChooseJobToRun's wrapper and WaitForJobs' TSC read then nap through
/// <see cref="IdleNapExports"/> when the worker has gone longer than the idle threshold since
/// then. Every byte is verified against v01.004.000 before anything is written; a different
/// build is left untouched.
/// </summary>
internal static class DemonsSoulsIdleNapPatch
{
    public const string EnvironmentVariable = "SHARPEMU_DS_IDLE_NAP";
    public const string IdleEnvironmentVariable = "SHARPEMU_DS_IDLE_NAP_IDLE_US";
    public const string NapEnvironmentVariable = "SHARPEMU_DS_IDLE_NAP_US";
    public const string TitleId = "PPSA01342";

    internal const uint DefaultIdleMicroseconds = 500;
    internal const uint DefaultNapMicroseconds = 250;

    // Callees, as image offsets.
    private const ulong ReadTsc = 0x863A20;
    private const ulong WorkerIndex = 0x864CF0;
    private const ulong ChooseJobToRun = 0x86ECD0;
    private const ulong RunJob = 0x86F5C0;

    // Sites: the worker loop's RunJob and ChooseJobToRun calls, WaitForJobs' TSC read, the
    // branch around the deadlock report, the report itself and its 10-byte flag-and-join tail.
    internal const ulong RunJobCall = 0x86E936;
    internal const ulong ChooseJobCall = 0x86E963;
    internal const ulong WaitTscCall = 0x86EB49;
    internal const ulong DeadlockBranch = 0x86EBE2;
    internal const ulong ReportBlock = 0x86EBE8;
    internal const ulong ReportTail = 0x86EC76;
    private const ulong AfterReport = 0x86EC80;

    private static readonly byte[] OriginalReportBlock = Convert.FromHexString(
        "488B0549F8880241FFC48038000F84820000004183FC02727C803D8801890200756CC5F96F4590488D056BD5" +
        "8601BF05000000BE01000000488D151BD586014C8D4DB031C94531C04889042431C0C5FA7F45B0C745C0D102" +
        "0000E8D95E000084C0742BCD41EB");

    // je AfterReport: taken when CheckForDeadlock reports no deadlock.
    private static readonly byte[] OriginalDeadlockBranch = Convert.FromHexString("0F8498000000");

    private static readonly byte[] OriginalReportTail = Convert.FromHexString("C60513018902014489");

    internal static bool IsEnabled =>
        !string.Equals(Environment.GetEnvironmentVariable(EnvironmentVariable), "0", StringComparison.Ordinal);

    /// <summary>Installs the nap when <paramref name="titleId"/> is Demon's Souls and the
    /// executable is the known build. <paramref name="addImportStub"/> maps an import stub for
    /// a NID and returns its guest address (0 on failure). Returns true when installed.</summary>
    public static bool Apply(
        IVirtualMemory memory,
        IReadOnlyList<ProgramHeader> programHeaders,
        ulong imageBase,
        string? titleId,
        Func<string, ulong> addImportStub)
    {
        ArgumentNullException.ThrowIfNull(memory);
        ArgumentNullException.ThrowIfNull(programHeaders);
        ArgumentNullException.ThrowIfNull(addImportStub);
        if (!IsEnabled || !string.Equals(titleId, TitleId, StringComparison.Ordinal))
        {
            return false;
        }

        var idleMicroseconds = ReadMicroseconds(IdleEnvironmentVariable, DefaultIdleMicroseconds);
        var napMicroseconds = ReadMicroseconds(NapEnvironmentVariable, DefaultNapMicroseconds);
        var thresholdTicks = ThresholdTicks(idleMicroseconds, KernelRuntimeCompatExports.TscFrequency);

        // Verify before the stub is mapped; the patcher verifies again before writing.
        foreach (var patch in BuildPatches(thresholdTicks, napMicroseconds, napStub: 0))
        {
            var current = new byte[patch.Old!.Length];
            if (!memory.TryRead(imageBase + patch.Offset, current) || !current.AsSpan().SequenceEqual(patch.Old))
            {
                Console.Error.WriteLine(
                    $"[LOADER][WARN] {titleId}: idle job-worker nap skipped, the executable does not carry the " +
                    $"expected bytes at 0x{patch.Offset:X} (a different build?).");
                return false;
            }
        }

        var napStub = addImportStub(IdleNapExports.Nid);
        if (napStub == 0)
        {
            Console.Error.WriteLine($"[LOADER][WARN] {titleId}: idle job-worker nap skipped, no import stub for it.");
            return false;
        }

        GuestBytePatcher.Apply(memory, programHeaders, imageBase, BuildPatches(thresholdTicks, napMicroseconds, napStub));
        Console.Error.WriteLine(
            $"[LOADER][INFO] {titleId}: idle job workers nap {napMicroseconds} us once idle for {idleMicroseconds} us " +
            $"({thresholdTicks} TSC ticks; set {EnvironmentVariable}=0 to keep them spinning).");
        return true;
    }

    internal static long ThresholdTicks(uint idleMicroseconds, ulong tscFrequency) =>
        (long)UInt128.Min((UInt128)idleMicroseconds * tscFrequency / 1_000_000, int.MaxValue);

    internal static List<GuestBytePatcher.Patch> BuildPatches(long thresholdTicks, uint napMicroseconds, ulong napStub)
    {
        var code = new Assembler(ReportBlock);

        // T3: replaces WaitForJobs' TSC read; also called by T1 with the slot in r12. Naps when
        // now - r12 reaches the threshold and returns the pre-nap TSC either way.
        var t3 = code.Here;
        code.Call(ReadTsc);
        code.Emit(0x50);                                    // push rax
        code.Emit(0x4C, 0x29, 0xE0);                        // sub rax, r12
        code.Emit(0x48, 0x3D).Int32((int)thresholdTicks);  // cmp rax, threshold
        code.Emit(0x7C, 0x11);                              // jl  (past the nap)
        code.Emit(0xBF).Int32((int)napMicroseconds);       // mov edi, nap
        code.Emit(0x48, 0xB8).Int64((long)napStub);        // mov rax, stub
        code.Emit(0xFF, 0xD0);                              // call rax
        code.Emit(0x58, 0xC3);                              // pop rax; ret

        // T1: wraps ChooseJobToRun(S, ...); on no job, loads the worker's slot and calls T3.
        var t1 = code.Here;
        code.Emit(0x53, 0x41, 0x54);                        // push rbx; push r12
        code.Emit(0x48, 0x89, 0xFB);                        // mov rbx, rdi
        code.Emit(0x50);                                    // push rax (alignment)
        code.Call(ChooseJobToRun);
        code.Emit(0x84, 0xC0);                              // test al, al
        code.Emit(0x75, 0x14);                              // jnz (to the epilogue)
        code.Call(WorkerIndex);
        code.Emit(0x4C, 0x8B, 0xA4, 0xC3).Int32(0xF00);    // mov r12, [rbx+rax*8+0xF00]
        code.Call(t3);
        code.Emit(0x31, 0xC0);                              // xor eax, eax
        code.Emit(0x59, 0x41, 0x5C, 0x5B, 0xC3);            // pop rcx; pop r12; pop rbx; ret

        // T2: wraps RunJob(S, job); stamps the worker's slot with the TSC at job end. Its last
        // instructions live in the report's tail.
        var t2 = code.Here;
        code.Emit(0x53);                                    // push rbx
        code.Emit(0x48, 0x89, 0xFB);                        // mov rbx, rdi
        code.Call(RunJob);
        code.Call(WorkerIndex);
        code.Emit(0x48, 0x8D, 0x1C, 0xC3);                  // lea rbx, [rbx+rax*8]
        code.Call(ReadTsc);
        code.Emit(0xEB).Rel8(ReportTail);                   // jmp tail
        var block = code.ToArray();
        if (block.Length > OriginalReportBlock.Length)
        {
            throw new InvalidOperationException($"idle nap trampolines need {block.Length} bytes, the report has {OriginalReportBlock.Length}");
        }

        var tail = new Assembler(ReportTail);
        tail.Emit(0x48, 0x89, 0x83).Int32(0xF00);          // mov [rbx+0xF00], rax
        tail.Emit(0x5B, 0xC3);                              // pop rbx; ret

        return
        [
            Site(DeadlockBranch, new Assembler(DeadlockBranch).Jmp(AfterReport).Emit(0x90).ToArray(), OriginalDeadlockBranch),
            Site(ReportBlock, block, OriginalReportBlock.AsSpan(0, block.Length).ToArray()),
            Site(ReportTail, tail.ToArray(), OriginalReportTail),
            Site(WaitTscCall, Call(WaitTscCall, t3), Call(WaitTscCall, ReadTsc)),
            Site(ChooseJobCall, Call(ChooseJobCall, t1), Call(ChooseJobCall, ChooseJobToRun)),
            Site(RunJobCall, Call(RunJobCall, t2), Call(RunJobCall, RunJob)),
        ];
    }

    private static GuestBytePatcher.Patch Site(ulong offset, byte[] replacement, byte[] old) =>
        new(
            offset,
            replacement,
            old,
            $"{TitleId} idle nap 0x{offset:X}={Convert.ToHexString(replacement).ToLowerInvariant()}?{Convert.ToHexString(old).ToLowerInvariant()}");

    private static byte[] Call(ulong at, ulong target) => new Assembler(at).Call(target).ToArray();

    private static uint ReadMicroseconds(string variable, uint fallback) =>
        uint.TryParse(Environment.GetEnvironmentVariable(variable), out var value) && value is > 0 and <= 100_000
            ? value
            : fallback;

    private sealed class Assembler(ulong origin)
    {
        private readonly List<byte> _bytes = [];

        public ulong Here => origin + (ulong)_bytes.Count;

        public Assembler Emit(params byte[] bytes)
        {
            _bytes.AddRange(bytes);
            return this;
        }

        public Assembler Int32(int value)
        {
            Span<byte> bytes = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
            _bytes.AddRange(bytes);
            return this;
        }

        public Assembler Int64(long value)
        {
            Span<byte> bytes = stackalloc byte[8];
            BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
            _bytes.AddRange(bytes);
            return this;
        }

        public Assembler Call(ulong target) => Emit(0xE8).Rel32(target);

        public Assembler Jmp(ulong target) => Emit(0xE9).Rel32(target);

        public Assembler Rel32(ulong target) => Int32(checked((int)((long)target - (long)(Here + 4))));

        public Assembler Rel8(ulong target) => Emit(unchecked((byte)checked((sbyte)((long)target - (long)(Here + 1)))));

        public byte[] ToArray() => [.. _bytes];
    }
}
