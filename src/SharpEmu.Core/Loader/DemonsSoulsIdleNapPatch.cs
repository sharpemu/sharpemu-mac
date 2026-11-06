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
/// <para>Wake-on-ready (default where <see cref="JobWakeRuntime.IsSupported"/>): the idle
/// worker blocks in <see cref="JobWakeExports"/> instead of sleeping a fixed nap, with the nap
/// length as its timeout, and the paths that make jobs ready wake it. Workers find work only by
/// polling: a job becomes visible to worker w when count[g][w] is raised, by Submit
/// (app+0x86E210, at 0x86E593/0x86E61A, for every worker in the batch's mask [batch+0xDA0]) or
/// by CounterChanged (app+0x86DCB0, at 0x86DF19, which re-queues jobs whose dependency counter
/// it was told changed). Jobs still waiting on dependencies are polled by ChooseJobToRun, and
/// those dependencies are job counters whose updates go through CounterChanged too. So Submit's
/// last call (the batch release at 0x86E7BD) and every CounterChanged call site get a
/// trampoline that bumps the wake generation and wakes sleepers with queued work: up to one per
/// submitted job when a worker in the batch's mask sleeps (Submit), or one when a worker sleeps
/// re-polling jobs that wait on dependencies (CounterChanged). The waits read the generation
/// before they look for work, so a wake cannot fall between the look and the sleep; work made
/// ready some other way (a GPU label a predicate polls) waits for the timeout, as it waited
/// for the nap. The trampolines live in a small code block the loader
/// allocates within rel32 reach; the guest's own control flow and data are unchanged.
/// SHARPEMU_DS_JOB_WAKE=0 restores the timed nap byte for byte.</para>
/// </summary>
internal static class DemonsSoulsIdleNapPatch
{
    public const string EnvironmentVariable = "SHARPEMU_DS_IDLE_NAP";
    public const string IdleEnvironmentVariable = "SHARPEMU_DS_IDLE_NAP_IDLE_US";
    public const string NapEnvironmentVariable = "SHARPEMU_DS_IDLE_NAP_US";
    public const string WakeEnvironmentVariable = "SHARPEMU_DS_JOB_WAKE";
    public const string TitleId = "PPSA01342";

    internal const uint DefaultIdleMicroseconds = 500;
    internal const uint DefaultNapMicroseconds = 250;

    // Callees, as image offsets.
    private const ulong ReadTsc = 0x863A20;
    private const ulong WorkerIndex = 0x864CF0;
    private const ulong ChooseJobToRun = 0x86ECD0;
    private const ulong RunJob = 0x86F5C0;
    private const ulong CounterChanged = 0x86DCB0;
    private const ulong ReleaseBatch = 0x8718C0;

    // Sites: the worker loop's RunJob and ChooseJobToRun calls, WaitForJobs' TSC read, the
    // branch around the deadlock report, the report itself and its 10-byte flag-and-join tail.
    internal const ulong RunJobCall = 0x86E936;
    internal const ulong ChooseJobCall = 0x86E963;
    internal const ulong WaitTscCall = 0x86EB49;
    internal const ulong DeadlockBranch = 0x86EBE2;
    internal const ulong ReportBlock = 0x86EBE8;
    internal const ulong ReportTail = 0x86EC76;
    private const ulong AfterReport = 0x86EC80;

    // Wake-on-ready sites: Submit's batch release (its last call, after every job of the batch
    // is queued) and the four calls of CounterChanged (worker loop on job start, RunJob's two
    // completion counters, and the public counter add at 0x869177).
    internal const ulong SubmitReleaseCall = 0x86E7BD;
    internal static readonly ulong[] CounterChangedCalls = [0x869177, 0x86E997, 0x86F8E7, 0x86F906];

    // Bytes the wake trampolines rely on without patching them, verified with the sites.
    internal static readonly (ulong Offset, byte[] Bytes)[] WakeContext =
    [
        // Submit: mov r12, rdi (S); mov [rbp-0x98], rsi (the batch); mov rbx, [rbp-0x98];
        // mov rcx, [rbx+8]; mov r15, [rbx] (jobs begin..end); add r15, 0x158 (job stride);
        // mov edx, [rbx+0xDA0] (worker mask); its tail from the loop exit through the release
        // call (lea rdi, [r12+0xDB8] keeps S in r12 there).
        (0x86E224, Convert.FromHexString("4989FC")),
        (0x86E2B7, Convert.FromHexString("4889B568FFFFFF")),
        (0x86E2CE, Convert.FromHexString("488B9D68FFFFFF488B4B084C8B3B")),
        (0x86E373, Convert.FromHexString("4981C758010000")),
        (0x86E56E, Convert.FromHexString("8B93A00D0000")),
        (0x86E77E, Convert.FromHexString(
            "498B8424480E0000498DBC24B80D0000BEFFFFFFFF4829C3722548C1FB0648B887B56F58FB86B56F" +
            "418B8C245C0E0000BEFFFFFFFF480FAFC34839C80F42F0")),

        // WaitForJobs: r14 = WorkerIndex(); its scan of [S+0xFF8] groups at [S+0x1000] +
        // g*0x9220 + 0x91E0 + 4w and of the mailbox [S+0xF80+8w] (GuestJobQueues reads the
        // same); r12 = [S+0xF00+8w] before the TSC call.
        (0x86EA60, Convert.FromHexString("E88B62FFFF803B000F852E020000488B55A84989C6")),
        (0x86EAAE, Convert.FromHexString(
            "8B83F80F00004885C07522EB620F1F44000041B7014489F824010A030F85CC0100008B83F80F0000" +
            "4885C07442488B8B00100000488B55A04869C020920000488D8C11E091000031D2660F1F84000000" +
            "0000833C11007FBA4A83BCF3800F000000751F")),
        (0x86EB41, Convert.FromHexString("4E8BA4F3000F0000")),
    ];

    private static readonly byte[] OriginalReportBlock = Convert.FromHexString(
        "488B0549F8880241FFC48038000F84820000004183FC02727C803D8801890200756CC5F96F4590488D056BD5" +
        "8601BF05000000BE01000000488D151BD586014C8D4DB031C94531C04889042431C0C5FA7F45B0C745C0D102" +
        "0000E8D95E000084C0742BCD41EB");

    // je AfterReport: taken when CheckForDeadlock reports no deadlock.
    private static readonly byte[] OriginalDeadlockBranch = Convert.FromHexString("0F8498000000");

    private static readonly byte[] OriginalReportTail = Convert.FromHexString("C60513018902014489");

    internal static bool IsEnabled =>
        !string.Equals(Environment.GetEnvironmentVariable(EnvironmentVariable), "0", StringComparison.Ordinal);

    internal static bool IsWakeEnabled =>
        !string.Equals(Environment.GetEnvironmentVariable(WakeEnvironmentVariable), "0", StringComparison.Ordinal);

    /// <summary>Where the wake trampolines went and what they call; null in nap mode.</summary>
    internal sealed record WakeLayout(ulong ImageBase, ulong CodeAddress, ulong WaitStub, ulong WakeStub, ulong StateAddress);

    // Offsets of the wake trampolines inside their code block.
    internal readonly record struct WakeEntries(int ChooseJob, int WaitTsc, int SubmitRelease, int CounterChanged);

    /// <summary>Installs the nap when <paramref name="titleId"/> is Demon's Souls and the
    /// executable is the known build. <paramref name="addImportStub"/> maps an import stub for
    /// a NID and returns its guest address (0 on failure); <paramref name="allocateCode"/>
    /// returns a writable, executable block of the given size within rel32 reach of the image
    /// (0 on failure, or null for no wake-on-ready). Returns true when installed.</summary>
    public static bool Apply(
        IVirtualMemory memory,
        IReadOnlyList<ProgramHeader> programHeaders,
        ulong imageBase,
        string? titleId,
        Func<string, ulong> addImportStub,
        Func<ulong, ulong>? allocateCode = null)
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
            if (!Matches(memory, imageBase + patch.Offset, patch.Old!))
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

        var wake = PrepareWake(memory, imageBase, titleId, thresholdTicks, napMicroseconds, addImportStub, allocateCode);
        GuestBytePatcher.Apply(memory, programHeaders, imageBase, BuildPatches(thresholdTicks, napMicroseconds, napStub, wake));
        Console.Error.WriteLine(wake is null
            ? $"[LOADER][INFO] {titleId}: idle job workers nap {napMicroseconds} us once idle for {idleMicroseconds} us " +
              $"({thresholdTicks} TSC ticks; set {EnvironmentVariable}=0 to keep them spinning)."
            : $"[LOADER][INFO] {titleId}: idle job workers wait for ready jobs (timeout {napMicroseconds} us) once idle " +
              $"for {idleMicroseconds} us ({thresholdTicks} TSC ticks; trampolines at 0x{wake.CodeAddress:X}; set " +
              $"{WakeEnvironmentVariable}=0 for the timed nap, {EnvironmentVariable}=0 to keep them spinning).");
        return true;
    }

    // Verifies the wake sites, maps the wait and wake stubs and writes the trampolines; null
    // (nap mode) when switched off or anything is missing.
    private static WakeLayout? PrepareWake(
        IVirtualMemory memory,
        ulong imageBase,
        string? titleId,
        long thresholdTicks,
        uint timeoutMicroseconds,
        Func<string, ulong> addImportStub,
        Func<ulong, ulong>? allocateCode)
    {
        if (!IsWakeEnabled || allocateCode is null)
        {
            return null;
        }

        string? skipped = null;
        foreach (var (offset, bytes) in WakeContext)
        {
            if (!Matches(memory, imageBase + offset, bytes))
            {
                skipped = $"the executable does not carry the expected bytes at 0x{offset:X}";
                break;
            }
        }

        foreach (var (offset, old) in WakeSites())
        {
            if (skipped is null && !Matches(memory, imageBase + offset, old))
            {
                skipped = $"the executable does not carry the expected bytes at 0x{offset:X}";
            }
        }

        if (skipped is null && !JobWakeRuntime.IsSupported)
        {
            skipped = "this host has no wait with an early wake (Darwin ulock)";
        }

        ulong stateAddress = 0;
        if (skipped is null && !JobWakeRuntime.TryInitialize(out stateAddress))
        {
            skipped = "its shared state could not be allocated";
        }

        WakeLayout? layout = null;
        if (skipped is null)
        {
            var waitStub = addImportStub(JobWakeExports.WaitNid);
            var wakeStub = waitStub == 0 ? 0 : addImportStub(JobWakeExports.WakeNid);
            var codeAddress = wakeStub == 0 ? 0 : allocateCode(WakeCodeBytes);
            if (codeAddress == 0)
            {
                skipped = "no import stubs or code block for it";
            }
            else
            {
                layout = new WakeLayout(imageBase, codeAddress, waitStub, wakeStub, stateAddress);
                byte[]? code;
                try
                {
                    // Also encodes every call between the block and the image.
                    _ = BuildPatches(thresholdTicks, timeoutMicroseconds, napStub: 0, layout);
                    code = BuildWakeCode(layout, thresholdTicks, timeoutMicroseconds, out _);
                }
                catch (OverflowException)
                {
                    code = null;
                }

                if (code is null)
                {
                    skipped = $"its code block at 0x{codeAddress:X} is out of rel32 reach of the image";
                    layout = null;
                }
                else if (!memory.TryWrite(codeAddress, code))
                {
                    skipped = $"its trampolines could not be written at 0x{codeAddress:X}";
                    layout = null;
                }
            }
        }

        if (skipped is not null)
        {
            Console.Error.WriteLine($"[LOADER][WARN] {titleId}: job-worker wake-on-ready skipped ({skipped}); idle workers nap instead.");
        }

        return layout;
    }

    private static bool Matches(IVirtualMemory memory, ulong address, byte[] expected)
    {
        var current = new byte[expected.Length];
        return memory.TryRead(address, current) && current.AsSpan().SequenceEqual(expected);
    }

    // The original calls the wake sites carry (the nap's own sites are in BuildPatches).
    private static IEnumerable<(ulong Offset, byte[] Old)> WakeSites()
    {
        yield return (SubmitReleaseCall, Call(SubmitReleaseCall, ReleaseBatch));
        foreach (var site in CounterChangedCalls)
        {
            yield return (site, Call(site, CounterChanged));
        }
    }

    internal static long ThresholdTicks(uint idleMicroseconds, ulong tscFrequency) =>
        (long)UInt128.Min((UInt128)idleMicroseconds * tscFrequency / 1_000_000, int.MaxValue);

    internal static List<GuestBytePatcher.Patch> BuildPatches(
        long thresholdTicks,
        uint napMicroseconds,
        ulong napStub,
        WakeLayout? wake = null)
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

        List<GuestBytePatcher.Patch> patches =
        [
            Site(DeadlockBranch, new Assembler(DeadlockBranch).Jmp(AfterReport).Emit(0x90).ToArray(), OriginalDeadlockBranch),
            Site(ReportBlock, block, OriginalReportBlock.AsSpan(0, block.Length).ToArray()),
            Site(ReportTail, tail.ToArray(), OriginalReportTail),
        ];
        if (wake is null)
        {
            patches.Add(Site(WaitTscCall, Call(WaitTscCall, t3), Call(WaitTscCall, ReadTsc)));
            patches.Add(Site(ChooseJobCall, Call(ChooseJobCall, t1), Call(ChooseJobCall, ChooseJobToRun)));
            patches.Add(Site(RunJobCall, Call(RunJobCall, t2), Call(RunJobCall, RunJob)));
            return patches;
        }

        // Wake mode keeps the report block (RunJob's stamp still runs there; the nap's T1 and
        // T3 are left unreferenced) and points the two idle sites at the wait trampolines.
        _ = BuildWakeCode(wake, thresholdTicks, napMicroseconds, out var entries);
        var wakeBlock = wake.CodeAddress;
        patches.Add(Site(WaitTscCall, FarCall(wake, WaitTscCall, wakeBlock + (ulong)entries.WaitTsc), Call(WaitTscCall, ReadTsc)));
        patches.Add(Site(ChooseJobCall, FarCall(wake, ChooseJobCall, wakeBlock + (ulong)entries.ChooseJob), Call(ChooseJobCall, ChooseJobToRun)));
        patches.Add(Site(RunJobCall, Call(RunJobCall, t2), Call(RunJobCall, RunJob)));
        foreach (var (offset, old) in WakeSites())
        {
            var entry = offset == SubmitReleaseCall ? entries.SubmitRelease : entries.CounterChanged;
            patches.Add(Site(offset, FarCall(wake, offset, wakeBlock + (ulong)entry), old));
        }

        return patches;
    }

    internal const ulong WakeCodeBytes = 0x1000;

    /// <summary>
    /// The wake trampolines, assembled at <see cref="WakeLayout.CodeAddress"/> (absolute
    /// addresses throughout). Each keeps the System V ABI of the call it replaces.
    /// </summary>
    internal static byte[] BuildWakeCode(WakeLayout wake, long thresholdTicks, uint timeoutMicroseconds, out WakeEntries entries)
    {
        var image = wake.ImageBase;
        var code = new Assembler(wake.CodeAddress);

        // ChooseJobToRun(S, &job) wrapper: reads the generation before the poll; on no job,
        // and once idle past the threshold since the worker's last job ended, waits.
        var chooseJob = code.Here;
        code.Emit(0x53, 0x41, 0x54, 0x41, 0x55);            // push rbx; push r12; push r13
        code.Emit(0x48, 0x89, 0xFB);                        // mov rbx, rdi
        code.Emit(0x48, 0xB8).Int64((long)wake.StateAddress); // mov rax, state
        code.Emit(0x4C, 0x8B, 0x28);                        // mov r13, [rax] (generation)
        code.Call(image + ChooseJobToRun);
        code.Emit(0x84, 0xC0);                              // test al, al
        var toOut = code.Jcc8(0x75);                        // jnz out
        code.Call(image + WorkerIndex);
        code.Emit(0x49, 0x89, 0xC4);                        // mov r12, rax (worker)
        code.Call(image + ReadTsc);
        code.Emit(0x4A, 0x2B, 0x84, 0xE3).Int32(0xF00);    // sub rax, [rbx+r12*8+0xF00]
        code.Emit(0x48, 0x3D).Int32((int)thresholdTicks);  // cmp rax, threshold
        var toNoJob = code.Jcc8(0x7C);                      // jl no_job
        code.Emit(0x48, 0x89, 0xDF);                        // mov rdi, rbx
        code.Emit(0x44, 0x89, 0xE6);                        // mov esi, r12d
        code.Emit(0x4C, 0x89, 0xEA);                        // mov rdx, r13
        code.Emit(0xB9).Int32((int)timeoutMicroseconds);   // mov ecx, timeout
        code.Emit(0x45, 0x31, 0xC0);                        // xor r8d, r8d (no queue check)
        code.Emit(0x48, 0xB8).Int64((long)wake.WaitStub);  // mov rax, wait
        code.Emit(0xFF, 0xD0);                              // call rax
        code.Bind(toNoJob);
        code.Emit(0x31, 0xC0);                              // no_job: xor eax, eax
        code.Bind(toOut);
        code.Emit(0x41, 0x5D, 0x41, 0x5C, 0x5B, 0xC3);      // out: pop r13; pop r12; pop rbx; ret

        // WaitForJobs' TSC read (rbx = S, r12 = the worker's last-job TSC, r14 = worker):
        // returns the TSC; once idle past the threshold, first waits, re-checking the queues
        // after it is published as asleep (its scan ran before the generation was read).
        var waitTsc = code.Align(16).Here;
        code.Call(image + ReadTsc);
        code.Emit(0x50);                                    // push rax
        code.Emit(0x4C, 0x29, 0xE0);                        // sub rax, r12
        code.Emit(0x48, 0x3D).Int32((int)thresholdTicks);  // cmp rax, threshold
        var toBusy = code.Jcc8(0x7C);                       // jl busy
        code.Emit(0x48, 0xB8).Int64((long)wake.StateAddress); // mov rax, state
        code.Emit(0x48, 0x8B, 0x10);                        // mov rdx, [rax] (generation)
        code.Emit(0x48, 0x89, 0xDF);                        // mov rdi, rbx
        code.Emit(0x44, 0x89, 0xF6);                        // mov esi, r14d
        code.Emit(0xB9).Int32((int)timeoutMicroseconds);   // mov ecx, timeout
        code.Emit(0x41, 0xB8).Int32(1);                    // mov r8d, 1 (check the queues)
        code.Emit(0x48, 0xB8).Int64((long)wake.WaitStub);  // mov rax, wait
        code.Emit(0xFF, 0xD0);                              // call rax
        code.Bind(toBusy);
        code.Emit(0x58, 0xC3);                              // busy: pop rax; ret

        // Submit's ReleaseBatch(S+0xDB8, index) call (rbp = Submit's frame, r12 = S): bumps
        // the generation and, when a worker in the batch's mask sleeps, wakes one per job.
        var submitRelease = code.Align(16).Here;
        code.Emit(0x57, 0x56, 0x50);                        // push rdi; push rsi; push rax
        code.Emit(0x48, 0x8B, 0x8D).Int32(-0x98);          // mov rcx, [rbp-0x98] (batch)
        code.Emit(0x8B, 0xB1).Int32(0xDA0);                // mov esi, [rcx+0xDA0] (mask)
        code.Emit(0x48, 0x8B, 0x41, 0x08);                  // mov rax, [rcx+8] (jobs end)
        code.Emit(0x48, 0x2B, 0x01);                        // sub rax, [rcx] (jobs begin)
        code.Emit(0x48, 0xB9).Int64((long)wake.StateAddress); // mov rcx, state
        code.Emit(0xF0, 0x48, 0xFF, 0x01);                  // lock inc qword [rcx]
        code.Emit(0x48, 0x85, 0x71, 0x08);                  // test [rcx+8], rsi
        var toRelease = code.Jcc8(0x74);                    // jz release
        code.Emit(0x31, 0xD2);                              // xor edx, edx
        code.Emit(0xB9).Int32(0x158);                      // mov ecx, job stride
        code.Emit(0x48, 0xF7, 0xF1);                        // div rcx
        code.Emit(0x48, 0x89, 0xC2);                        // mov rdx, rax (jobs)
        code.Emit(0x4C, 0x89, 0xE7);                        // mov rdi, r12
        code.Emit(0x48, 0xB8).Int64((long)wake.WakeStub);  // mov rax, wake
        code.Emit(0xFF, 0xD0);                              // call rax
        code.Bind(toRelease);
        code.Emit(0x58, 0x5E, 0x5F);                        // release: pop rax; pop rsi; pop rdi
        code.Jmp(image + ReleaseBatch);

        // CounterChanged(S, counter, value) wrapper: after it re-queued what the change
        // unblocked, bumps the generation and, when a worker sleeps re-polling jobs that wait
        // on dependencies, wakes one sleeper with queued work. (A change with only queue-empty
        // sleepers skips the call: the worker that made it polls next itself, and a second
        // ready job waits for a timeout as it waited for the nap.)
        var counterChanged = code.Align(16).Here;
        code.Emit(0x57);                                    // push rdi
        code.Call(image + CounterChanged);
        code.Emit(0x5F);                                    // pop rdi
        code.Emit(0x48, 0xB8).Int64((long)wake.StateAddress); // mov rax, state
        code.Emit(0xF0, 0x48, 0xFF, 0x00);                  // lock inc qword [rax]
        code.Emit(0x48, 0x83, 0x78, 0x10, 0x00);            // cmp qword [rax+16], 0 (pollers)
        code.Emit(0x75, 0x01, 0xC3);                        // jnz wake; ret
        code.Emit(0x48, 0x83, 0xEC, 0x08);                  // wake: sub rsp, 8
        code.Emit(0x48, 0xC7, 0xC6).Int32(-1);             // mov rsi, -1 (any worker)
        code.Emit(0xBA).Int32(1);                          // mov edx, 1
        code.Emit(0x48, 0xB8).Int64((long)wake.WakeStub);  // mov rax, wake
        code.Emit(0xFF, 0xD0);                              // call rax
        code.Emit(0x48, 0x83, 0xC4, 0x08, 0xC3);            // add rsp, 8; ret

        var bytes = code.ToArray();
        if ((ulong)bytes.Length > WakeCodeBytes)
        {
            throw new InvalidOperationException($"wake trampolines need {bytes.Length} bytes, the block has {WakeCodeBytes}");
        }

        var origin = wake.CodeAddress;
        entries = new WakeEntries(
            (int)(chooseJob - origin), (int)(waitTsc - origin), (int)(submitRelease - origin), (int)(counterChanged - origin));
        return bytes;
    }

    // A call from image offset `at` to an absolute address, encoded for the loaded image.
    private static byte[] FarCall(WakeLayout wake, ulong at, ulong target) =>
        new Assembler(wake.ImageBase + at).Call(target).ToArray();

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

        // A short conditional jump forward whose target Bind fills in; returns its rel8 index.
        public int Jcc8(byte opcode)
        {
            Emit(opcode, 0);
            return _bytes.Count - 1;
        }

        public void Bind(int rel8) =>
            _bytes[rel8] = unchecked((byte)checked((sbyte)(_bytes.Count - (rel8 + 1))));

        public Assembler Align(int alignment)
        {
            while ((Here & (ulong)(alignment - 1)) != 0)
            {
                _bytes.Add(0xCC);
            }

            return this;
        }

        public byte[] ToArray() => [.. _bytes];
    }
}
