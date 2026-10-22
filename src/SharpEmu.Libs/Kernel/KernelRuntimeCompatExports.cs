// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.DsTrace;
using SharpEmu.Libs.Fiber;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace SharpEmu.Libs.Kernel;

public static class KernelRuntimeCompatExports
{
    private const ulong TlsErrnoOffset = 0x40;

    /// <summary>
    /// The guest's pid. A console title runs as a small fixed pid in its own
    /// process namespace, not the host's; getpid, sysctl kern.proc.* and kill
    /// all agree on this value (an intentional deviation).
    /// </summary>
    internal const int GuestProcessId = 74;

    /// <summary>The guest's parent pid (the console's SceSysCore-style launcher).</summary>
    internal const int GuestParentProcessId = 1;
    private const ulong TlsStackChkGuardBaseOffset = 0x800;
    private const ulong StackChkGuardFieldOffset = 0x10;
    private const ulong TlsProcParamOffset = 0x100;
    private const ulong TlsMallocReplaceOffset = 0x200;
    private const ulong TlsNewReplaceOffset = 0x300;
    private const int MallocReplaceSize = 0x70;
    private const int NewReplaceSize = 0x68;
    private const int OrbisTimesecSize = sizeof(long) + sizeof(uint) + sizeof(uint);
    // PS4/PS5 libkernel module-info ABI. The extended form is consumed by
    // sceKernelGetModuleInfoFromAddr and ends at byte 0x1A8; libc commonly
    // places its stack canary immediately after that caller-owned buffer.
    private const int ModuleInfoNameMaxBytes = 256;
    private const int ModuleInfoSize = 0x160;
    private const int ModuleInfoExSize = 0x1A8;
    private const ulong ModuleInfoNameOffset = 0x08;
    private const ulong ModuleInfoExHandleOffset = 0x108;
    private const ulong ModuleInfoExInitProcOffset = 0x128;
    private const ulong ModuleInfoExSegmentsOffset = 0x160;
    private const ulong ModuleInfoExSegmentCountOffset = 0x1A0;
    private const int ModuleInfoSegmentSize = 16;
    private const ulong GuestExecutableAddressFloor = 0x0000000800000000UL;
    private const ulong GuestExecutableAddressCeiling = 0x0000000880000000UL;
    private const ulong HostUnwindBoundarySize = 0x10_0000UL;
    private const ulong DefaultKernelTscFrequency = 10_000_000UL;
    private const ulong PrtAreaStartAddress = 0x0000001000000000UL;
    private const ulong PrtAreaSize = 0x000000EC00000000UL;
    private const int AioInitParamSize = 0x3C;
    private const uint MemCommit = 0x1000;
    private const uint MemReserve = 0x2000;
    private const uint PageExecuteReadWrite = 0x40;
    private static readonly object _stateGate = new();
    private static readonly long _processStartCounter = Stopwatch.GetTimestamp();
    private static readonly RdtscDelegate? _rdtscReader = CreateRdtscReader();
    private static readonly ulong _kernelTscFrequency = ResolveKernelTscFrequency();
    private static readonly ulong _stackChkGuardValue = 0xC0DEC0DECAFEBA00UL;
    private static readonly nint _stackChkGuardObjectAddress =
        HleDataSymbols.TryGetAddress("f7uOxY9mM1U", out var stackChkGuardAddress)
            ? unchecked((nint)stackChkGuardAddress)
            : AllocateStackChkGuardObject();
    private static ulong _applicationHeapApiAddress;
    private static ulong _processProcParamAddress;
    private static uint _gpoStateBits;
    // sceSysmodule load counts by module id (Load increments, Unload
    // decrements; IsLoaded is "count > 0").
    private static readonly Dictionary<uint, int> _sysmoduleLoadCounts = new();

    // SDK error_table.csv: 0x805A1000 "tried to load an unknown PRX",
    // 0x805A1001 "unloaded a PRX already unloaded, or the module is not loaded".
    internal const int SceSysmoduleErrorInvalidValue = unchecked((int)0x805A1000);
    internal const int SceSysmoduleErrorUnloaded = unchecked((int)0x805A1001);
    private static readonly object _prtApertureGate = new();
    private static readonly (ulong Base, ulong Size)[] _prtApertures = new (ulong Base, ulong Size)[3];
    private static int _stackChkFailCount;
    private static long _usleepTraceCount;
    // SHARPEMU_USLEEP_SHORT_PUMP=1 restores the old behaviour of pumping the guest
    // scheduler on every sceKernelUsleep, including sub-millisecond ones.
    private static readonly bool _usleepShortPump =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_USLEEP_SHORT_PUMP"), "1", StringComparison.Ordinal);
    private static readonly bool _traceUsleep =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_USLEEP"), "1", StringComparison.Ordinal);
    private static readonly bool _traceGuestThreads =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_GUEST_THREADS"), "1", StringComparison.Ordinal);

    // Adaptive nap for usleep(<1 ms) polling loops (a worker pool waiting for jobs). The
    // first short sleeps of a streak sleep exactly what they ask for, so a wait that ends
    // quickly stays fast; once one thread has been polling for SHARPEMU_USLEEP_NAP_AFTER_US (default 500 us)
    // its short sleeps really sleep, for max(requested, SHARPEMU_USLEEP_NAP_US) (default
    // 250 us; 0 turns naps off). A streak ends when the thread spends more than 2 ms
    // between short sleeps or sleeps for 1 ms or more. Per thread, not per title.
    internal static readonly long UsleepNapMicroseconds =
        ReadNonNegativeEnvironmentInteger("SHARPEMU_USLEEP_NAP_US", 250);
    internal static readonly long _usleepNapAfterTicks =
        ReadNonNegativeEnvironmentInteger("SHARPEMU_USLEEP_NAP_AFTER_US", 500) * Stopwatch.Frequency / 1_000_000;
    internal static readonly long _usleepStreakGapTicks = Stopwatch.Frequency * 2 / 1000;

    [ThreadStatic]
    private static long _shortUsleepStreakStart;

    [ThreadStatic]
    private static long _lastShortUsleep;

    private static long ReadNonNegativeEnvironmentInteger(string name, long fallback) =>
        long.TryParse(Environment.GetEnvironmentVariable(name), out var value) && value >= 0
            ? value
            : fallback;

    /// <summary>
    /// The sub-millisecond usleep policy (see <see cref="UsleepNapMicroseconds"/>): returns
    /// true when this call napped, false when it slept just the requested time.
    /// </summary>
    internal static bool ShortUsleep(ulong micros)
    {
        var now = Stopwatch.GetTimestamp();
        if (_lastShortUsleep == 0 || now - _lastShortUsleep > _usleepStreakGapTicks)
        {
            _shortUsleepStreakStart = now;
        }

        _lastShortUsleep = now;
        if (UsleepNapMicroseconds > 0 && now - _shortUsleepStreakStart >= _usleepNapAfterTicks)
        {
            HostTiming.SleepMicroseconds(Math.Max((long)micros, UsleepNapMicroseconds));
            _lastShortUsleep = Stopwatch.GetTimestamp();
            return true;
        }

        // Like the FreeBSD kernel (kern_clock_nanosleep always blocks for a nonzero
        // interval), a short sleep never returns before its requested time. A bare yield
        // returned in well under a microsecond, so a guest that counts usleep(1) calls as
        // its timeout ("GPU hang after N polls") gave up long before real hardware would,
        // and before the streak above was old enough to nap. The wait is an
        // absolute-deadline host wait (mach_wait_until on macOS), never a millisecond
        // round-up.
        HostTiming.SleepMicroseconds((long)micros);
        return false;
    }

    private static readonly bool _stopwatchTicksAreNanoseconds =
        Stopwatch.Frequency == 1_000_000_000L;


    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate ulong RdtscDelegate();

    [SysAbiExport(
        Nid = "uvT2iYBBnkY",
        ExportName = "sceKernelSync",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelSync(CpuContext ctx) => ctx.SetReturn(0);

    [SysAbiExport(
        Nid = "1jfXLRVzisc",
        ExportName = "sceKernelUsleep",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelUsleep(CpuContext ctx)
    {
        var micros = ctx[CpuRegister.Rdi];
        var dsUsleep = DsTrace.Sched && DsTrace.Armed;
        var dsEnter = dsUsleep ? Stopwatch.GetTimestamp() : 0;
        TraceUsleepSpin(ctx, micros);
        if (micros == 0)
        {
            ctx[CpuRegister.Rax] = 0;
            if (dsUsleep)
            {
                DsSched.Usleep(ctx, 0, dsEnter, 0, 0);
            }
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        // Sub-millisecond usleep is a worker polling backoff. Pump takes the global
        // guest-thread gate and scans every thread, so many workers spinning on
        // usleep(1) contend on it; the 1 ms dispatcher still pumps. The short sleep
        // below still blocks. SHARPEMU_USLEEP_SHORT_PUMP=1 restores the old always-Pump path.
        long dsPump = 0;
        if (micros >= 1000 || _usleepShortPump)
        {
            dsPump = dsUsleep ? Stopwatch.GetTimestamp() : 0;
            GuestThreadExecution.Scheduler?.Pump(ctx, "sceKernelUsleep");
            if (dsUsleep)
            {
                dsPump = Stopwatch.GetTimestamp() - dsPump;
            }
        }

        var dsBranch = 2;
        if (micros < 1000)
        {
            dsBranch = ShortUsleep(micros) ? 1 : 0;
        }
        else
        {
            _lastShortUsleep = 0;
            // Precise sleep: rounding microseconds up to Thread.Sleep
            // milliseconds (which itself overshoots by a scheduler quantum)
            // hard-caps games that pace their frame loop with usleep.
            micros = Audio.AudioOut2Exports.ProducerSleepMicroseconds(micros);
            HostTiming.SleepMicroseconds((long)Math.Min(micros, long.MaxValue));
        }

        ctx[CpuRegister.Rax] = 0;
        if (dsUsleep)
        {
            DsSched.Usleep(ctx, micros, dsEnter, dsPump, dsBranch);
        }
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static void TraceUsleepSpin(CpuContext ctx, ulong micros)
    {
        if (!_traceUsleep)
        {
            return;
        }

        var count = Interlocked.Increment(ref _usleepTraceCount);
        if (count > 32 && count % 10000 != 0)
        {
            return;
        }

        var rbx = ctx[CpuRegister.Rbx];
        var r12 = ctx[CpuRegister.R12];
        var r13 = ctx[CpuRegister.R13];
        var lockAddress = rbx == 0 ? 0 : rbx + 0xF78;
        var lockText = "unreadable";
        if (lockAddress != 0 && ctx.TryReadUInt64(lockAddress, out var lockValue))
        {
            lockText = $"0x{lockValue:X16}";
        }

        var schedulerText = "unreadable";
        if (r12 != 0 && ctx.TryReadUInt64(r12 + 8, out var schedulerAddress))
        {
            schedulerText = $"0x{schedulerAddress:X16}";
        }

        var waitValueText = "unreadable";
        if (r13 != 0 && ctx.TryReadUInt64(r13, out var waitValue))
        {
            waitValueText = $"0x{waitValue:X16}";
        }

        var callerReturnText = "unreadable";
        var rbp = ctx[CpuRegister.Rbp];
        if (rbp != 0 && ctx.TryReadUInt64(rbp + 8, out var callerReturn))
        {
            callerReturnText = $"0x{callerReturn:X16}";
        }

        var returnRip = GuestThreadExecution.TryGetCurrentImportCallFrame(out var frame)
            ? frame.ReturnRip
            : 0UL;
        var thread = GuestThreadExecution.CurrentGuestThreadHandle;
        var fiber = FiberExports.GetCurrentFiberAddressForDiagnostics(ctx);

        Console.Error.WriteLine(
            $"[LOADER][TRACE] usleep#{count}: usec={micros} ret=0x{returnRip:X16} caller={callerReturnText} thread=0x{thread:X16} fiber=0x{fiber:X16} rbx=0x{rbx:X16} lock@+F78=0x{lockAddress:X16}:{lockText} r12=0x{r12:X16} scheduler@+8={schedulerText} r13=0x{r13:X16}:{waitValueText} r14=0x{ctx[CpuRegister.R14]:X16} r15=0x{ctx[CpuRegister.R15]:X16}");

        if (count % 100000 == 0 &&
            _traceGuestThreads &&
            GuestThreadExecution.Scheduler is { } scheduler)
        {
            foreach (var snapshot in scheduler.SnapshotThreads())
            {
                Console.Error.WriteLine(
                    $"[LOADER][TRACE] guest_thread.snapshot handle=0x{snapshot.ThreadHandle:X16} name='{snapshot.Name}' " +
                    $"state={snapshot.State} imports={snapshot.ImportCount} nid={snapshot.LastImportNid ?? "none"} " +
                    $"ret=0x{snapshot.LastReturnRip:X16} block={snapshot.BlockReason ?? "none"}");
            }
        }
    }

    [SysAbiExport(
        Nid = "QBi7HCK03hw",
        ExportName = "sceKernelClockGettime",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelClockGettime(CpuContext ctx)
    {
        var clockId = unchecked((int)ctx[CpuRegister.Rdi]);
        var timeAddress = ctx[CpuRegister.Rsi];
        if (timeAddress == 0)
        {
            return KernelErrno.Sce(OrbisErrno.EFAULT);
        }

        var errno = KernelClocks.TryGetTime(clockId, out var seconds, out var nanoseconds);
        if (errno != 0)
        {
            return KernelErrno.Sce(errno);
        }

        if (!KernelClocks.TryWriteTimespec(ctx, timeAddress, seconds, nanoseconds))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "wRYVA5Zolso",
        ExportName = "sceKernelClockGetres",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelClockGetres(CpuContext ctx)
    {
        var clockId = unchecked((int)ctx[CpuRegister.Rdi]);
        var resolutionAddress = ctx[CpuRegister.Rsi];
        var errno = KernelClocks.TryGetResolution(clockId, out var seconds, out var nanoseconds);
        if (errno != 0)
        {
            return KernelErrno.Sce(errno);
        }

        if (resolutionAddress != 0 &&
            !KernelClocks.TryWriteTimespec(ctx, resolutionAddress, seconds, nanoseconds))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    /// <summary>
    /// Reads a clock for an internal deadline computation (condvar timed
    /// waits). An id the guest could not have validated falls back to the
    /// monotonic clock instead of failing the wait.
    /// </summary>
    internal static void GetClockTime(int clockId, out long seconds, out long nanoseconds)
    {
        if (KernelClocks.TryGetTime(clockId, out seconds, out nanoseconds) != 0)
        {
            ReadMonotonicSinceStart(out seconds, out nanoseconds);
        }
    }

    internal static void ReadMonotonicSinceStart(out long seconds, out long nanoseconds)
    {
        var processStartCounter = _processStartCounter;
        var elapsedTicks = Stopwatch.GetTimestamp() - processStartCounter;
        if (_stopwatchTicksAreNanoseconds)
        {
            // Constant divisors let the JIT strength-reduce the division;
            // games call this thousands of times per second.
            seconds = elapsedTicks / 1_000_000_000L;
            nanoseconds = elapsedTicks - seconds * 1_000_000_000L;
        }
        else
        {
            seconds = elapsedTicks / Stopwatch.Frequency;
            nanoseconds = (elapsedTicks % Stopwatch.Frequency) * 1_000_000_000L / Stopwatch.Frequency;
        }
    }

    [SysAbiExport(
        Nid = "ejekcaNQNq0",
        ExportName = "sceKernelGettimeofday",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGettimeofday(CpuContext ctx)
    {
        var timeAddress = ctx[CpuRegister.Rdi];
        if (timeAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var now = DateTimeOffset.UtcNow;
        var seconds = now.ToUnixTimeSeconds();
        var microseconds = (now.Ticks % TimeSpan.TicksPerSecond) / 10;
        if (!ctx.TryWriteUInt64(timeAddress, unchecked((ulong)seconds)) ||
            !ctx.TryWriteUInt64(timeAddress + sizeof(long), unchecked((ulong)microseconds)))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "n88vx3C5nW8",
        ExportName = "gettimeofday",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixGettimeofday(CpuContext ctx)
    {
        var timeAddress = ctx[CpuRegister.Rdi];
        var timezoneAddress = ctx[CpuRegister.Rsi];
        var now = DateTimeOffset.UtcNow;
        var seconds = now.ToUnixTimeSeconds();
        var microseconds = (now.Ticks % TimeSpan.TicksPerSecond) / 10;

        if (timeAddress != 0 &&
            (!ctx.TryWriteUInt64(timeAddress, unchecked((ulong)seconds)) ||
             !ctx.TryWriteUInt64(timeAddress + sizeof(long), unchecked((ulong)microseconds))))
        {
            return KernelErrno.Posix(ctx, OrbisErrno.EFAULT);
        }

        if (timezoneAddress != 0 &&
            (!TryWriteInt32(ctx, timezoneAddress, 0) ||
             !TryWriteInt32(ctx, timezoneAddress + sizeof(int), 0)))
        {
            return KernelErrno.Posix(ctx, OrbisErrno.EFAULT);
        }

        ctx[CpuRegister.Rax] = 0;
        return 0;
    }

    [SysAbiExport(
        Nid = "-2IRUCO--PM",
        ExportName = "sceKernelReadTsc",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelReadTsc(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = ReadTscCounter();
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    internal static ulong TscFrequency => _kernelTscFrequency;

    internal static ulong ReadTscCounter() =>
        TryReadHostTsc(out var counter) ? counter : unchecked((ulong)Math.Max(0, Stopwatch.GetTimestamp()));

    [SysAbiExport(
        Nid = "1j3S3n-tTW4",
        ExportName = "sceKernelGetTscFrequency",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetTscFrequency(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = _kernelTscFrequency;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "4J2sUJmuHZQ",
        ExportName = "sceKernelGetProcessTime",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetProcessTime(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = ReadProcessTimeMicroseconds();
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    internal static ulong ReadProcessTimeMicroseconds()
    {
        var elapsedTicks = Stopwatch.GetTimestamp() - _processStartCounter;
        var micros = elapsedTicks * 1_000_000L / Stopwatch.Frequency;
        return unchecked((ulong)Math.Max(0, micros));
    }

    [SysAbiExport(
        Nid = "fgxnMeTNUtY",
        ExportName = "sceKernelGetProcessTimeCounter",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetProcessTimeCounter(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = ReadProcessTimeCounter();
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    internal static ulong ReadProcessTimeCounter()
        => ReadProcessTimeCounterAt(Stopwatch.GetTimestamp());

    internal static ulong ReadProcessTimeCounterAt(long timestamp)
    {
        var elapsedTicks = Math.Max(0, timestamp - _processStartCounter);
        if (DiagProcessTimeScale is > 0.0 and not 1.0)
        {
            var after = (long)(DiagProcessTimeScaleAfterSeconds * Stopwatch.Frequency);
            if (elapsedTicks > after)
            {
                elapsedTicks = after + (long)((elapsedTicks - after) * DiagProcessTimeScale);
            }
        }

        if (_stopwatchTicksAreNanoseconds)
        {
            return unchecked((ulong)elapsedTicks);
        }

        var whole = elapsedTicks / Stopwatch.Frequency;
        var remainder = elapsedTicks % Stopwatch.Frequency;
        return unchecked((ulong)(whole * (long)ProcessTimeCounterFrequency +
            remainder * (long)ProcessTimeCounterFrequency / Stopwatch.Frequency));
    }

    /// <summary>
    /// <c>sceKernelGetProcessTimeCounterFrequency</c>: a fixed 1 GHz on every
    /// host (an intentional deviation). The counter is
    /// scaled to it, so counter / frequency stays real seconds.
    /// </summary>
    internal const ulong ProcessTimeCounterFrequency = 1_000_000_000UL;

    /// <summary>
    /// True when the host Stopwatch already ticks in nanoseconds, so a native
    /// stub reading the raw host counter agrees with
    /// <see cref="ProcessTimeCounterFrequency"/>.
    /// </summary>
    public static bool HostCounterMatchesProcessTimeFrequency => _stopwatchTicksAreNanoseconds && DiagProcessTimeScale is 0.0 or 1.0;

    // [WIP] diag: SHARPEMU_DIAG_PROCESS_TIME_SCALE=<factor> slows (factor < 1) the process-time
    // counter the guest paces its frames with, so each frame advances less game time.
    internal static readonly double DiagProcessTimeScale =
        double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DIAG_PROCESS_TIME_SCALE"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture,
            out var scale) && scale > 0 ? scale : 1.0;

    // SHARPEMU_DIAG_PROCESS_TIME_SCALE_AFTER=<seconds>: the scale applies from then on.
    internal static readonly double DiagProcessTimeScaleAfterSeconds =
        double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DIAG_PROCESS_TIME_SCALE_AFTER"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture,
            out var seconds) && seconds > 0 ? seconds : 0.0;

    [SysAbiExport(
        Nid = "BNowx2l588E",
        ExportName = "sceKernelGetProcessTimeCounterFrequency",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetProcessTimeCounterFrequency(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = ProcessTimeCounterFrequency;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "6xVpy0Fdq+I",
        ExportName = "_sigprocmask",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int Sigprocmask(CpuContext ctx)
    {
        _ = ctx;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "959qrazPIrg",
        ExportName = "sceKernelGetProcParam",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetProcParam(CpuContext ctx)
    {
        ulong address;
        lock (_stateGate)
        {
            address = _processProcParamAddress;
        }

        if (address == 0)
        {
            address = GetTlsScratchAddress(ctx, TlsProcParamOffset);
        }

        if (address != 0)
        {
            if (address == GetTlsScratchAddress(ctx, TlsProcParamOffset))
            {
                _ = ctx.Memory.TryWrite(address, new byte[0x80]);
            }
        }

        TraceProcParam(ctx, address);

        ctx[CpuRegister.Rax] = address;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // SceProcParam header: u64 size, u32 magic, u32 entry count, u64 SDK
    // version (0xMMmmppxx, e.g. 0x04000031 for SDK 4.00).
    private const ulong ProcParamSdkVersionOffset = 0x10;
    private const ulong ProcParamMinimumSizeWithSdkVersion = 0x18;

    /// <summary>
    /// The SDK version the title was built with, from its PT_SCE_PROCPARAM
    /// segment. False when the title has no proc param or it is too short.
    /// </summary>
    internal static bool TryGetProcessSdkVersion(CpuContext ctx, out uint sdkVersion)
    {
        sdkVersion = 0;
        ulong address;
        lock (_stateGate)
        {
            address = _processProcParamAddress;
        }

        if (address == 0 ||
            !ctx.TryReadUInt64(address, out var size) ||
            size < ProcParamMinimumSizeWithSdkVersion ||
            !ctx.TryReadUInt64(address + ProcParamSdkVersionOffset, out var version) ||
            version == 0)
        {
            return false;
        }

        sdkVersion = unchecked((uint)version);
        return true;
    }

    public static void ConfigureProcessProcParamAddress(ulong procParamAddress)
    {
        lock (_stateGate)
        {
            _processProcParamAddress = procParamAddress;
        }
    }

    private static void TraceProcParam(CpuContext ctx, ulong address)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_PROC_PARAM"), "1", StringComparison.Ordinal))
        {
            return;
        }

        if (address == 0)
        {
            Console.Error.WriteLine("[LOADER][TRACE] proc_param: address=0");
            return;
        }

        const int dumpSize = 0x200;
        var buffer = GC.AllocateUninitializedArray<byte>(dumpSize);
        if (!ctx.Memory.TryRead(address, buffer))
        {
            Console.Error.WriteLine($"[LOADER][TRACE] proc_param: address=0x{address:X16} unreadable");
            return;
        }

        Console.Error.WriteLine($"[LOADER][TRACE] proc_param: address=0x{address:X16} size=0x{dumpSize:X}");
        for (var offset = 0; offset < dumpSize; offset += 16)
        {
            var slice = buffer.AsSpan(offset, 16);
            var hex = Convert.ToHexString(slice);
            Console.Error.WriteLine($"[LOADER][TRACE] proc_param[{offset:X3}]: {hex}");
        }

        TraceProcParamPointers(ctx, address, buffer);
    }

    private static void TraceProcParamPointers(CpuContext ctx, ulong baseAddress, ReadOnlySpan<byte> buffer)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_PROC_PARAM_PTRS"), "1", StringComparison.Ordinal))
        {
            return;
        }

        if (buffer.Length < 0x50)
        {
            return;
        }

        for (var offset = 0x20; offset <= 0x48; offset += 8)
        {
            var ptr = BinaryPrimitives.ReadUInt64LittleEndian(buffer.Slice(offset, 8));
            Console.Error.WriteLine($"[LOADER][TRACE] proc_param.ptr@{offset:X2}: 0x{ptr:X16}");
            if (ptr == 0)
            {
                continue;
            }

            TraceProcParamPointerTarget(ctx, ptr);
        }
    }

    private static void TraceProcParamPointerTarget(CpuContext ctx, ulong address)
    {
        const int maxAsciiBytes = 256;
        const int maxWideChars = 128;

        if (TryReadUtf8CString(ctx, address, maxAsciiBytes, out var asciiValue))
        {
            Console.Error.WriteLine($"[LOADER][TRACE] proc_param.ptr.target ascii@0x{address:X16}: \"{asciiValue}\"");
            return;
        }

        if (TryReadUtf16CString(ctx, address, maxWideChars, out var wideValue))
        {
            Console.Error.WriteLine($"[LOADER][TRACE] proc_param.ptr.target wide@0x{address:X16}: \"{wideValue}\"");
            return;
        }

        var preview = GC.AllocateUninitializedArray<byte>(64);
        if (ctx.Memory.TryRead(address, preview))
        {
            var hex = Convert.ToHexString(preview);
            Console.Error.WriteLine($"[LOADER][TRACE] proc_param.ptr.target hex@0x{address:X16}: {hex}");
            TraceProcParamEmbeddedPointers(ctx, address, preview);
        }
        else
        {
            Console.Error.WriteLine($"[LOADER][TRACE] proc_param.ptr.target unreadable@0x{address:X16}");
        }
    }

    private static bool TryReadUtf8CString(CpuContext ctx, ulong address, int maxBytes, out string value)
    {
        value = string.Empty;
        var buffer = GC.AllocateUninitializedArray<byte>(maxBytes);
        if (!ctx.Memory.TryRead(address, buffer))
        {
            return false;
        }

        var length = Array.IndexOf(buffer, (byte)0);
        if (length < 0)
        {
            length = maxBytes;
        }

        if (length == 0)
        {
            return false;
        }

        var text = Encoding.UTF8.GetString(buffer, 0, length);
        if (!IsMostlyPrintable(text))
        {
            return false;
        }

        value = text;
        return true;
    }

    private static bool TryReadUtf16CString(CpuContext ctx, ulong address, int maxChars, out string value)
    {
        value = string.Empty;
        var maxBytes = maxChars * 2;
        var buffer = GC.AllocateUninitializedArray<byte>(maxBytes);
        if (!ctx.Memory.TryRead(address, buffer))
        {
            return false;
        }

        var lengthBytes = -1;
        for (var i = 0; i + 1 < buffer.Length; i += 2)
        {
            if (buffer[i] == 0 && buffer[i + 1] == 0)
            {
                lengthBytes = i;
                break;
            }
        }

        if (lengthBytes <= 0)
        {
            return false;
        }

        var text = Encoding.Unicode.GetString(buffer, 0, lengthBytes);
        if (!IsMostlyPrintable(text))
        {
            return false;
        }

        value = text;
        return true;
    }

    private static bool IsMostlyPrintable(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var printable = 0;
        for (var i = 0; i < text.Length; i++)
        {
            var ch = text[i];
            if (ch == '\0')
            {
                continue;
            }

            if (!char.IsControl(ch) || ch == '\r' || ch == '\n' || ch == '\t')
            {
                printable++;
            }
        }

        return printable >= Math.Max(4, text.Length * 3 / 4);
    }

    private static void TraceProcParamEmbeddedPointers(CpuContext ctx, ulong baseAddress, ReadOnlySpan<byte> data)
    {
        const int maxCandidates = 12;
        var found = 0;
        Span<byte> probe = stackalloc byte[2];

        for (var offset = 0; offset + 8 <= data.Length; offset += 8)
        {
            var candidate = BinaryPrimitives.ReadUInt64LittleEndian(data.Slice(offset, 8));
            if (candidate == 0)
            {
                continue;
            }

            if (!ctx.Memory.TryRead(candidate, probe))
            {
                continue;
            }

            if (TryReadUtf8CString(ctx, candidate, 256, out var ascii))
            {
                Console.Error.WriteLine($"[LOADER][TRACE] proc_param.ptr.embed@0x{baseAddress:X16}+0x{offset:X2} -> 0x{candidate:X16} ascii \"{ascii}\"");
                if (++found >= maxCandidates)
                {
                    return;
                }
                continue;
            }

            if (TryReadUtf16CString(ctx, candidate, 128, out var wide))
            {
                Console.Error.WriteLine($"[LOADER][TRACE] proc_param.ptr.embed@0x{baseAddress:X16}+0x{offset:X2} -> 0x{candidate:X16} wide \"{wide}\"");
                if (++found >= maxCandidates)
                {
                    return;
                }
            }
        }
    }

    [SysAbiExport(
        Nid = "9BcDykPmo1I",
        ExportName = "__error",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int ErrorAddress(CpuContext ctx)
    {
        var address = GetTlsScratchAddress(ctx, TlsErrnoOffset);
        ctx[CpuRegister.Rax] = address;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    internal static bool TrySetErrno(CpuContext ctx, int value)
    {
        var address = GetTlsScratchAddress(ctx, TlsErrnoOffset);
        return address != 0 && TryWriteInt32(ctx, address, value);
    }

    internal static bool TryGetErrno(CpuContext ctx, out int value)
    {
        value = 0;
        var address = GetTlsScratchAddress(ctx, TlsErrnoOffset);
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        if (address == 0 || !ctx.Memory.TryRead(address, bytes))
        {
            return false;
        }

        value = BinaryPrimitives.ReadInt32LittleEndian(bytes);
        return true;
    }

    [SysAbiExport(
        Nid = "bnZxYgAFeA0",
        ExportName = "sceKernelGetSanitizerNewReplaceExternal",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetSanitizerNewReplaceExternal(CpuContext ctx)
    {
        var address = GetTlsScratchAddress(ctx, TlsNewReplaceOffset);
        if (address != 0)
        {
            if (!ctx.Memory.TryWrite(address, new byte[NewReplaceSize]) ||
                !ctx.TryWriteUInt64(address, NewReplaceSize))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }
        }

        ctx[CpuRegister.Rax] = address;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "py6L8jiVAN8",
        ExportName = "sceKernelGetSanitizerMallocReplaceExternal",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetSanitizerMallocReplaceExternal(CpuContext ctx)
    {
        var address = GetTlsScratchAddress(ctx, TlsMallocReplaceOffset);
        if (address != 0)
        {
            if (!ctx.Memory.TryWrite(address, new byte[MallocReplaceSize]) ||
                !ctx.TryWriteUInt64(address, MallocReplaceSize))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }
        }

        ctx[CpuRegister.Rax] = address;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "jh+8XiK4LeE",
        ExportName = "sceKernelIsAddressSanitizerEnabled",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelIsAddressSanitizerEnabled(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "ca7v6Cxulzs",
        ExportName = "sceKernelSetGPO",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelSetGpo(CpuContext ctx)
    {
        _gpoStateBits = unchecked((uint)ctx[CpuRegister.Rdi]);
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "4oXYe9Xmk0Q",
        ExportName = "sceKernelGetGPI",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetGpi(CpuContext ctx)
    {
        var bitMask = _gpoStateBits;
        if (string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_ALLOC_IMPORTS"), "1", StringComparison.Ordinal))
        {
            Console.Error.WriteLine(
                $"[LOADER][TRACE] get_gpi: mask=0x{bitMask:X8}");
        }
        ctx[CpuRegister.Rax] = bitMask;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "7oxv3PPCumo",
        ExportName = "sceKernelReserveVirtualRange",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelReserveVirtualRange(CpuContext ctx) =>
        KernelMemoryCompatExports.ReserveBackingRange(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi],
            ctx[CpuRegister.Rdx], ctx[CpuRegister.Rcx]);

    [SysAbiExport(
        Nid = "BohYr-F7-is",
        ExportName = "sceKernelSetPrtAperture",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelSetPrtAperture(CpuContext ctx)
    {
        var rawId = ctx[CpuRegister.Rdi];
        var apertureBase = ctx[CpuRegister.Rsi];
        var apertureSize = ctx[CpuRegister.Rdx];
        var apertureId = unchecked((int)rawId);

        if (apertureId < 0 || apertureId >= _prtApertures.Length)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if ((apertureBase & 0xFFFUL) != 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (apertureBase < PrtAreaStartAddress)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (apertureSize > PrtAreaSize)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (apertureBase - PrtAreaStartAddress > PrtAreaSize - apertureSize)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        lock (_prtApertureGate)
        {
            _prtApertures[apertureId] = (apertureBase, apertureSize);
        }

        Console.Error.WriteLine(
            $"[LOADER][TRACE] set_prt_aperture: id={apertureId} base=0x{apertureBase:X16} size=0x{apertureSize:X16}");
        if (string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_ALLOC_IMPORTS"), "1", StringComparison.Ordinal))
        {
            Console.Error.WriteLine(
                $"[LOADER][TRACE] set_prt_aperture raw: rdi=0x{rawId:X16} rsi=0x{apertureBase:X16} rdx=0x{apertureSize:X16} rcx=0x{ctx[CpuRegister.Rcx]:X16} r8=0x{ctx[CpuRegister.R8]:X16} r9=0x{ctx[CpuRegister.R9]:X16}");
        }
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // sceKernelGetPrtAperture(id, *base, *size): the aperture SetPrtAperture stored, zero when unset.
    [SysAbiExport(
        Nid = "L0v2Go5jOuM",
        ExportName = "sceKernelGetPrtAperture",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetPrtAperture(CpuContext ctx)
    {
        var apertureId = unchecked((int)ctx[CpuRegister.Rdi]);
        var baseAddress = ctx[CpuRegister.Rsi];
        var sizeAddress = ctx[CpuRegister.Rdx];
        if (apertureId < 0 || apertureId >= _prtApertures.Length)
            return unchecked((int)0x80020016);
        if (baseAddress == 0 || sizeAddress == 0)
            return unchecked((int)0x8002000E);
        (ulong Base, ulong Size) aperture;
        lock (_prtApertureGate)
            aperture = _prtApertures[apertureId];
        return ctx.TryWriteUInt64(baseAddress, aperture.Base) && ctx.TryWriteUInt64(sizeAddress, aperture.Size)
            ? 0
            : unchecked((int)0x8002000E);
    }

    internal static bool OverlapsPrtAperture(ulong address, ulong size)
    {
        if (size == 0)
            return false;
        var end = address > ulong.MaxValue - size ? ulong.MaxValue : address + size;
        lock (_prtApertureGate)
        {
            return _prtApertures.Any(aperture => aperture.Size != 0 &&
                aperture.Base < end && address < aperture.Base + aperture.Size);
        }
    }

    internal static bool ContainsPrtRange(ulong address, ulong size)
    {
        if (size == 0 || address > ulong.MaxValue - size)
            return false;
        lock (_prtApertureGate)
        {
            return _prtApertures.Any(aperture => address >= aperture.Base &&
                size <= aperture.Size && address - aperture.Base <= aperture.Size - size);
        }
    }

    [SysAbiExport(
        Nid = "f7KBOafysXo",
        ExportName = "sceKernelGetModuleInfoFromAddr",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetModuleInfoFromAddr(CpuContext ctx)
    {
        var queriedAddress = ctx[CpuRegister.Rdi];
        var flags = unchecked((int)ctx[CpuRegister.Rsi]);
        var outInfoAddress = ctx[CpuRegister.Rdx];
        if (outInfoAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (flags is < 0 or >= 3)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!ctx.TryReadUInt64(outInfoAddress, out var callerSize))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (callerSize != ModuleInfoExSize)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!KernelModuleRegistry.TryGetModuleByAddress(queriedAddress, out var module))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        if (!TryWriteModuleInfoEx(ctx, outInfoAddress, module))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "RpQJJVKTiFM",
        ExportName = "sceKernelGetModuleInfoForUnwind",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetModuleInfoForUnwind(CpuContext ctx)
    {
        var queriedAddress = ctx[CpuRegister.Rdi];
        var flags = unchecked((int)ctx[CpuRegister.Rsi]);
        var outInfoAddress = ctx[CpuRegister.Rdx];
        if (outInfoAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (flags is < 0 or >= 3)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!ctx.TryReadUInt64(outInfoAddress, out var callerSize))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (callerSize < 0x130)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!KernelModuleRegistry.TryGetModuleByAddress(queriedAddress, out var module))
        {
            if (queriedAddress >= GuestExecutableAddressFloor &&
                queriedAddress < GuestExecutableAddressCeiling)
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
            }

            if (!TryWriteHostBoundaryModuleInfoForUnwind(ctx, outInfoAddress, queriedAddress))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }
        }
        else if (!TryWriteModuleInfoForUnwind(ctx, outInfoAddress, module))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "crb5j7mkk1c",
        ExportName = "_is_signal_return",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelIsSignalReturn(CpuContext ctx)
    {
        // SharpEmu calls guest exception handlers without a signal-return frame.
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "kUpgrXIrz7Q",
        ExportName = "sceKernelGetModuleInfo",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetModuleInfo(CpuContext ctx)
    {
        return KernelGetModuleInfoByHandleCore(
            ctx,
            unchecked((int)ctx[CpuRegister.Rdi]),
            ctx[CpuRegister.Rsi]);
    }

    [SysAbiExport(
        Nid = "QgsKEUfkqMA",
        ExportName = "sceKernelGetModuleInfo2",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetModuleInfo2(CpuContext ctx)
    {
        return KernelGetModuleInfoByHandleCore(
            ctx,
            unchecked((int)ctx[CpuRegister.Rdi]),
            ctx[CpuRegister.Rsi]);
    }

    [SysAbiExport(
        Nid = "HZO7xOos4xc",
        ExportName = "sceKernelGetModuleInfoInternal",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetModuleInfoInternal(CpuContext ctx)
    {
        return KernelGetModuleInfoExByHandleCore(
            ctx,
            unchecked((int)ctx[CpuRegister.Rdi]),
            ctx[CpuRegister.Rsi]);
    }

    [SysAbiExport(
        Nid = "IuxnUuXk6Bg",
        ExportName = "sceKernelGetModuleList",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetModuleList(CpuContext ctx)
    {
        return KernelGetModuleListCore(
            ctx,
            handlesAddress: ctx[CpuRegister.Rdi],
            capacity: ctx[CpuRegister.Rsi],
            outCountAddress: ctx[CpuRegister.Rdx],
            includeSystemModules: true);
    }

    [SysAbiExport(
        Nid = "ZzzC3ZGVAkc",
        ExportName = "sceKernelGetModuleList2",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetModuleList2(CpuContext ctx)
    {
        return KernelGetModuleListCore(
            ctx,
            handlesAddress: ctx[CpuRegister.Rdi],
            capacity: ctx[CpuRegister.Rsi],
            outCountAddress: ctx[CpuRegister.Rdx],
            includeSystemModules: false);
    }

    [SysAbiExport(
        Nid = "Fjc4-n1+y2g",
        ExportName = "__elf_phdr_match_addr",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int ElfPhdrMatchAddr(CpuContext ctx)
    {
        var moduleInfoAddress = ctx[CpuRegister.Rdi];
        _ = ctx[CpuRegister.Rsi]; // dtor virtual address
        if (moduleInfoAddress == 0)
        {
            ctx[CpuRegister.Rax] = 0;
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        ctx[CpuRegister.Rax] = 1;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "OMDRKKAZ8I4",
        ExportName = "sceKernelDebugRaiseException",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDebugRaiseException(CpuContext ctx)
    {
        return KernelDebugRaiseExceptionCore(ctx, "sceKernelDebugRaiseException");
    }

    [SysAbiExport(
        Nid = "zE-wXIZjLoM",
        ExportName = "sceKernelDebugRaiseExceptionOnReleaseMode",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDebugRaiseExceptionOnReleaseMode(CpuContext ctx)
    {
        return KernelDebugRaiseExceptionCore(ctx, "sceKernelDebugRaiseExceptionOnReleaseMode");
    }

    private static int KernelDebugRaiseExceptionCore(CpuContext ctx, string exportName)
    {
        var error = unchecked((uint)ctx[CpuRegister.Rdi]);
        var unknown = unchecked((long)ctx[CpuRegister.Rsi]);
        if (unknown != 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        // libkernel treats this as a fatal debug exception. Callers commonly place
        // UD2 immediately after the noreturn call, so returning normally turns the
        // guest's intended termination into a host illegal-instruction fault.
        Console.Error.WriteLine($"[LOADER][ERROR] {exportName}: error=0x{error:X8}");
        GuestThreadExecution.RequestCurrentEntryExit(exportName, unchecked((ulong)error));
        ctx[CpuRegister.Rax] = error;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "cfwBSQyr5Ys",
        ExportName = "sceKernelDebugWriteCppExceptionInfo",
        Target = Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDebugWriteCppExceptionInfo(CpuContext ctx)
    {
        _ = ctx;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "f7uOxY9mM1U",
        ExportName = "__stack_chk_guard",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int StackCheckGuard(CpuContext ctx)
    {
        var baseAddress = _stackChkGuardObjectAddress != 0
            ? unchecked((ulong)_stackChkGuardObjectAddress)
            : GetTlsScratchAddress(ctx, TlsStackChkGuardBaseOffset);
        if (baseAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (_stackChkGuardObjectAddress != 0)
        {
            try
            {
                Marshal.WriteInt64(_stackChkGuardObjectAddress, unchecked((long)_stackChkGuardValue));
                Marshal.WriteInt64(IntPtr.Add(_stackChkGuardObjectAddress, (int)sizeof(ulong)), unchecked((long)_stackChkGuardValue));
            }
            catch
            {
            }
        }

        if (ctx.FsBase != 0)
        {
            _ = ctx.TryWriteUInt64(ctx.FsBase + 0x28, _stackChkGuardValue);
            _ = ctx.TryWriteUInt64(ctx.FsBase + TlsStackChkGuardBaseOffset, _stackChkGuardValue);
            _ = ctx.TryWriteUInt64(ctx.FsBase + TlsStackChkGuardBaseOffset + StackChkGuardFieldOffset, _stackChkGuardValue);
        }

        ctx[CpuRegister.Rax] = baseAddress;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "Ou3iL1abvng",
        ExportName = "__stack_chk_fail",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int StackCheckFail(CpuContext ctx)
    {
        var count = Interlocked.Increment(ref _stackChkFailCount);
        Console.Error.WriteLine(
            $"[LOADER][ERROR] __stack_chk_fail#{count}: rip=0x{ctx.Rip:X16} rdi=0x{ctx[CpuRegister.Rdi]:X16}");
        var result = (int)OrbisGen2Result.ORBIS_GEN2_ERROR_CPU_TRAP;
        GuestThreadExecution.RequestCurrentEntryExit("__stack_chk_fail", result);
        ctx[CpuRegister.Rax] = unchecked((ulong)result);
        return result;
    }

    [SysAbiExport(
        Nid = "p5EcQeEeJAE",
        ExportName = "_sceKernelRtldSetApplicationHeapAPI",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelRtldSetApplicationHeapApi(CpuContext ctx)
    {
        lock (_stateGate)
        {
            _applicationHeapApiAddress = ctx[CpuRegister.Rdi];
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "QKd0qM58Qes",
        ExportName = "sceKernelStopUnloadModule",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelStopUnloadModule(CpuContext ctx)
    {
        var resultAddress = ctx[CpuRegister.R9];
        if (resultAddress != 0 && !TryWriteInt32(ctx, resultAddress, 0))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    /// <summary>
    /// sceKernelDlclose(handle): 0 for a handle the registry knows, ESRCH
    /// otherwise. Modules stay mapped: nothing is unloaded (deviations.md).
    /// The NID is not in the PS5 catalog; it is exported for titles built
    /// against older SDKs.
    /// </summary>
    [SysAbiExport(
        Nid = "cuC7gViUT4Q",
        ExportName = "sceKernelDlclose",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDlclose(CpuContext ctx)
    {
        var handle = unchecked((int)ctx[CpuRegister.Rdi]);
        var result = KernelModuleRegistry.TryGetModuleByHandle(handle, out _)
            ? 0
            : OrbisErrno.ToSceKernelError(OrbisErrno.ESRCH);
        ctx[CpuRegister.Rax] = unchecked((ulong)(long)result);
        return result;
    }

    [SysAbiExport(
        Nid = "wzvqT4UqKX8",
        ExportName = "sceKernelLoadStartModule",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelLoadStartModule(CpuContext ctx)
    {
        var modulePathAddress = ctx[CpuRegister.Rdi];
        var argumentSize = ctx[CpuRegister.Rsi];
        var argumentAddress = ctx[CpuRegister.Rdx];
        var resultAddress = ctx[CpuRegister.R9];
        if (resultAddress != 0 && !TryWriteInt32(ctx, resultAddress, 0))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (!TryReadUtf8Z(ctx, modulePathAddress, 512, out var modulePath) ||
            string.IsNullOrWhiteSpace(modulePath))
        {
            return ReturnModuleLoadError(
                ctx,
                (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        int handle;
        var hasDirectory = modulePath.Contains('/') || modulePath.Contains('\\');
        var isApp0Path = modulePath.StartsWith("/app0/", StringComparison.OrdinalIgnoreCase) ||
            modulePath.StartsWith("app0/", StringComparison.OrdinalIgnoreCase);
        var loadResult = hasDirectory
            ? KernelModuleRegistry.LoadModule(modulePath)
            : default;
        if (loadResult.Succeeded)
        {
            handle = loadResult.Handle;
        }
        else if (!isApp0Path &&
            KernelModuleRegistry.TryFindByPathOrName(modulePath, out var moduleByPath))
        {
            handle = moduleByPath.Handle;
        }
        else
        {
            loadResult = hasDirectory
                ? loadResult
                : KernelModuleRegistry.LoadModule(modulePath);
            if (!loadResult.Succeeded)
            {
                return ReturnModuleLoadError(ctx, loadResult.Error);
            }

            handle = loadResult.Handle;
        }

        if (KernelModuleRegistry.TryBeginModuleStart(handle, out var moduleToStart))
        {
            var scheduler = GuestThreadExecution.Scheduler;
            string? startError = null;
            ulong startResult = 0;
            var started = scheduler is not null && scheduler.TryCallGuestFunction(
                ctx,
                moduleToStart.InitEntryPoint,
                argumentSize,
                argumentAddress,
                0,
                0,
                0,
                $"sceKernelLoadStartModule:{moduleToStart.Name}",
                out startResult,
                out startError);
            KernelModuleRegistry.CompleteModuleStart(handle, started);
            if (!started)
            {
                Console.Error.WriteLine(
                    $"[LOADER][ERROR] sceKernelLoadStartModule failed to start '{moduleToStart.Name}' " +
                    $"at 0x{moduleToStart.InitEntryPoint:X16}: {startError ?? "guest scheduler unavailable"}");
                return ReturnModuleLoadError(
                    ctx,
                    (int)OrbisGen2Result.ORBIS_GEN2_ERROR_CPU_TRAP);
            }

            if (resultAddress != 0 && !TryWriteInt32(ctx, resultAddress, unchecked((int)startResult)))
            {
                return ReturnModuleLoadError(
                    ctx,
                    (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }

            Console.Error.WriteLine(
                $"[LOADER][INFO] sceKernelLoadStartModule started '{moduleToStart.Name}' " +
                $"at 0x{moduleToStart.InitEntryPoint:X16}");
        }

        ctx[CpuRegister.Rax] = unchecked((uint)handle);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int ReturnModuleLoadError(CpuContext ctx, int error)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)(long)error);
        return error;
    }

    [SysAbiExport(
        Nid = "g8cM39EUZ6o",
        ExportName = "sceSysmoduleLoadModule",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSysmodule")]
    public static int SysmoduleLoadModule(CpuContext ctx) =>
        SysmoduleResult(ctx, LoadSysmodule(unchecked((ushort)ctx[CpuRegister.Rdi])));

    /// <summary>
    /// One load of a system module: 0, or INVALID_VALUE for id 0. Ids are
    /// not otherwise validated: no id table is available here, so an unknown
    /// id loads as an HLE module (deviations.md).
    /// </summary>
    internal static int LoadSysmodule(uint moduleId)
    {
        if ((moduleId & 0x7FFFFFFF) == 0)
        {
            return SceSysmoduleErrorInvalidValue;
        }

        // A module already mapped from the title's sce_module keeps its real
        // handle (KernelModuleRegistry.MarkSysmoduleLoaded); otherwise the
        // HLE library stands in under a synthetic handle.
        _ = KernelModuleRegistry.MarkSysmoduleLoaded(unchecked((int)moduleId));
        lock (_stateGate)
        {
            _sysmoduleLoadCounts[moduleId] = _sysmoduleLoadCounts.GetValueOrDefault(moduleId) + 1;
        }

        return 0;
    }

    /// <summary>One unload: 0, INVALID_VALUE for id 0, UNLOADED when not loaded.</summary>
    internal static int UnloadSysmodule(uint moduleId)
    {
        if ((moduleId & 0x7FFFFFFF) == 0)
        {
            return SceSysmoduleErrorInvalidValue;
        }

        lock (_stateGate)
        {
            if (!_sysmoduleLoadCounts.TryGetValue(moduleId, out var count) || count < 1)
            {
                return SceSysmoduleErrorUnloaded;
            }

            if (count == 1)
            {
                _sysmoduleLoadCounts.Remove(moduleId);
                KernelModuleRegistry.MarkSysmoduleUnloaded(unchecked((int)moduleId));
            }
            else
            {
                _sysmoduleLoadCounts[moduleId] = count - 1;
            }
        }

        return 0;
    }

    /// <summary>0 when loaded, INVALID_VALUE for id 0, otherwise UNLOADED.</summary>
    internal static int QuerySysmodule(uint moduleId)
    {
        if ((moduleId & 0x7FFFFFFF) == 0)
        {
            return SceSysmoduleErrorInvalidValue;
        }

        lock (_stateGate)
        {
            return _sysmoduleLoadCounts.GetValueOrDefault(moduleId) > 0 ? 0 : SceSysmoduleErrorUnloaded;
        }
    }

    private static int SysmoduleResult(CpuContext ctx, int result)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)(long)result);
        return result;
    }

    [SysAbiExport(
        Nid = "nu4a0-arQis",
        ExportName = "sceKernelAioInitializeParam",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAioInitializeParam(CpuContext ctx)
    {
        var paramAddress = ctx[CpuRegister.Rdi];
        if (paramAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        Span<byte> zero = stackalloc byte[AioInitParamSize];
        zero.Clear();
        if (!ctx.Memory.TryWrite(paramAddress, zero))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "-o5uEDpN+oY",
        ExportName = "sceKernelConvertUtcToLocaltime",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelConvertUtcToLocaltime(CpuContext ctx)
    {
        var utcSeconds = unchecked((long)ctx[CpuRegister.Rdi]);
        var localTimeAddress = ctx[CpuRegister.Rsi];
        var timesecAddress = ctx[CpuRegister.Rdx];
        var dstSecondsAddress = ctx[CpuRegister.Rcx];

        var timezone = GetStandardTimezone();
        var localSeconds = ConvertUtcToLocaltimeSeconds(
            utcSeconds,
            timezone.MinutesWest,
            timezone.DstSeconds);
        var westSeconds = unchecked((uint)(-timezone.MinutesWest * 60));

        if (localTimeAddress != 0 &&
            !ctx.TryWriteUInt64(localTimeAddress, unchecked((ulong)localSeconds)))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (timesecAddress != 0)
        {
            Span<byte> timesec = stackalloc byte[OrbisTimesecSize];
            BinaryPrimitives.WriteInt64LittleEndian(timesec, utcSeconds);
            BinaryPrimitives.WriteUInt32LittleEndian(timesec.Slice(sizeof(long), sizeof(uint)), westSeconds);
            BinaryPrimitives.WriteUInt32LittleEndian(
                timesec.Slice(sizeof(long) + sizeof(uint), sizeof(uint)),
                unchecked((uint)timezone.DstSeconds));
            if (!ctx.Memory.TryWrite(timesecAddress, timesec))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }
        }

        if (dstSecondsAddress != 0 &&
            !ctx.TryWriteUInt64(dstSecondsAddress, unchecked((ulong)timezone.DstSeconds)))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "0NTHN1NKONI",
        ExportName = "sceKernelConvertLocaltimeToUtc",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelConvertLocaltimeToUtc(CpuContext ctx)
    {
        var localSeconds = unchecked((long)ctx[CpuRegister.Rdi]);
        var utcTimeAddress = ctx[CpuRegister.Rdx];
        var timeResultAddress = ctx[CpuRegister.Rcx];
        var dstSecondsAddress = ctx[CpuRegister.R8];

        if (timeResultAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var timezone = GetStandardTimezone();
        var utcSeconds = ConvertLocaltimeToUtcSeconds(
            localSeconds,
            timezone.MinutesWest,
            timezone.DstSeconds);
        // Both conversion directions return seconds and UTC offsets in the same layout.
        Span<byte> timeResult = stackalloc byte[OrbisTimesecSize];
        BinaryPrimitives.WriteInt64LittleEndian(timeResult, utcSeconds);
        BinaryPrimitives.WriteInt32LittleEndian(timeResult.Slice(sizeof(long), sizeof(int)), -timezone.MinutesWest * 60);
        BinaryPrimitives.WriteInt32LittleEndian(
            timeResult.Slice(sizeof(long) + sizeof(int), sizeof(int)),
            timezone.DstSeconds);
        if (!ctx.Memory.TryWrite(timeResultAddress, timeResult))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (utcTimeAddress != 0 && !ctx.TryWriteUInt64(utcTimeAddress, unchecked((ulong)utcSeconds)))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (dstSecondsAddress != 0 && !TryWriteInt32(ctx, dstSecondsAddress, timezone.DstSeconds))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    internal static long ConvertLocaltimeToUtcSeconds(long localSeconds, int minutesWest, int dstSeconds) =>
        unchecked(localSeconds + (long)minutesWest * 60 - dstSeconds);

    internal static long ConvertUtcToLocaltimeSeconds(long utcSeconds, int minutesWest, int dstSeconds) =>
        unchecked(utcSeconds - (long)minutesWest * 60 + dstSeconds);

    private static (int MinutesWest, int DstSeconds) GetStandardTimezone()
    {
        // Use the standard UTC offset; daylight-saving settings are not modeled.
        var timezone = TimeZoneInfo.Local;
        var baseOffset = timezone.BaseUtcOffset;
        return (unchecked((int)-baseOffset.TotalMinutes), 0);
    }

    [SysAbiExport(
        Nid = "vYU8P9Td2Zo",
        ExportName = "sceKernelAioInitializeImpl",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAioInitializeImpl(CpuContext ctx)
    {
        var paramAddress = ctx[CpuRegister.Rdi];
        var size = unchecked((int)ctx[CpuRegister.Rsi]);
        if (paramAddress == 0 || size < AioInitParamSize)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "eR2bZFAAU0Q",
        ExportName = "sceSysmoduleUnloadModule",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSysmodule")]
    public static int SysmoduleUnloadModule(CpuContext ctx) =>
        SysmoduleResult(ctx, UnloadSysmodule(unchecked((ushort)ctx[CpuRegister.Rdi])));

    [SysAbiExport(
        Nid = "hHrGoGoNf+s",
        ExportName = "sceSysmoduleLoadModuleInternalWithArg",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSysmodule")]
    public static int SysmoduleLoadModuleInternalWithArg(CpuContext ctx)
    {
        // Internal ids are 32-bit (0x80000000 | id).
        var moduleId = unchecked((uint)ctx[CpuRegister.Rdi]);
        var resultAddress = ctx[CpuRegister.R8];
        var result = LoadSysmodule(moduleId);
        if (result != 0)
        {
            return SysmoduleResult(ctx, result);
        }

        if (resultAddress != 0 && !TryWriteInt32(ctx, resultAddress, 0))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "4fU5yvOkVG4",
        ExportName = "sceSysmoduleGetModuleInfoForUnwind",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSysmodule")]
    public static int SysmoduleGetModuleInfoForUnwind(CpuContext ctx) =>
        KernelGetModuleInfoForUnwind(ctx);

    [SysAbiExport(
        Nid = "fMP5NHUOaMk",
        ExportName = "sceSysmoduleIsLoaded",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSysmodule")]
    public static int SysmoduleIsLoaded(CpuContext ctx)
    {
        return SysmoduleResult(ctx, QuerySysmodule(unchecked((ushort)ctx[CpuRegister.Rdi])));
    }

    [SysAbiExport(
        Nid = "WslcK1FQcGI",
        ExportName = "sceKernelIsNeoMode",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelIsNeoMode(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "Xjoosiw+XPI",
        ExportName = "sceKernelUuidCreate",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelUuidCreate(CpuContext ctx)
    {
        var uuidAddress = ctx[CpuRegister.Rdi];
        if (uuidAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        Span<byte> uuid = stackalloc byte[16];
        RandomNumberGenerator.Fill(uuid);
        if (!ctx.Memory.TryWrite(uuidAddress, uuid))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int KernelGetModuleInfoByHandleCore(CpuContext ctx, int handle, ulong outInfoAddress)
    {
        if (outInfoAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!KernelModuleRegistry.TryGetModuleByHandle(handle, out var module))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        if (!ctx.TryReadUInt64(outInfoAddress, out var callerSize))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (callerSize != ModuleInfoSize)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryWriteModuleInfo(ctx, outInfoAddress, module))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int KernelGetModuleInfoExByHandleCore(CpuContext ctx, int handle, ulong outInfoAddress)
    {
        if (outInfoAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!KernelModuleRegistry.TryGetModuleByHandle(handle, out var module))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        if (!ctx.TryReadUInt64(outInfoAddress, out var callerSize))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (callerSize != ModuleInfoExSize)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryWriteModuleInfoEx(ctx, outInfoAddress, module))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int KernelGetModuleListCore(
        CpuContext ctx,
        ulong handlesAddress,
        ulong capacity,
        ulong outCountAddress,
        bool includeSystemModules)
    {
        if (outCountAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var handles = KernelModuleRegistry.GetModuleHandles(includeSystemModules);
        if (!ctx.TryWriteUInt64(outCountAddress, unchecked((ulong)handles.Length)))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (handlesAddress == 0 || capacity == 0 || handles.Length == 0)
        {
            ctx[CpuRegister.Rax] = 0;
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        var writableCount = (int)Math.Min(Math.Min(capacity, (ulong)int.MaxValue), (ulong)handles.Length);
        for (var i = 0; i < writableCount; i++)
        {
            if (!TryWriteInt32(ctx, handlesAddress + (ulong)(i * sizeof(int)), handles[i]))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }
        }

        if ((ulong)handles.Length > capacity)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static bool TryWriteModuleInfo(
        CpuContext ctx,
        ulong outInfoAddress,
        KernelModuleRegistry.ModuleEntry module)
    {
        var payload = new byte[ModuleInfoSize];
        BinaryPrimitives.WriteUInt64LittleEndian(payload, ModuleInfoSize);
        WriteModuleName(payload, module.Name);
        // segments[4] at 0x108, segment_count at 0x148; the 20-byte
        // fingerprint at 0x14C stays zero (not read from the SELF).
        WriteModuleSegments(payload, 0x108, 0x148, module);
        return ctx.Memory.TryWrite(outInfoAddress, payload);
    }

    private static bool TryWriteModuleInfoEx(
        CpuContext ctx,
        ulong outInfoAddress,
        KernelModuleRegistry.ModuleEntry module)
    {
        var payload = new byte[ModuleInfoExSize];
        BinaryPrimitives.WriteUInt64LittleEndian(payload, ModuleInfoExSize);
        WriteModuleName(payload, module.Name);
        BinaryPrimitives.WriteInt32LittleEndian(
            payload.AsSpan((int)ModuleInfoExHandleOffset),
            module.Handle);
        BinaryPrimitives.WriteUInt64LittleEndian(
            payload.AsSpan((int)ModuleInfoExInitProcOffset),
            module.EntryPoint);
        // eh_frame_hdr_addr 0x148, eh_frame_addr 0x150, eh_frame_hdr_size
        // 0x158 (unknown, 0), eh_frame_size 0x15C.
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(0x148), module.EhFrameHeaderAddress);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(0x150), module.EhFrameAddress);
        BinaryPrimitives.WriteUInt32LittleEndian(
            payload.AsSpan(0x15C),
            (uint)Math.Min(module.EhFrameSize, uint.MaxValue));
        WriteModuleSegments(
            payload,
            (int)ModuleInfoExSegmentsOffset,
            (int)ModuleInfoExSegmentCountOffset,
            module);
        // ref_count: every module the registry reports is loaded once.
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x1A4), 1);
        return ctx.Memory.TryWrite(outInfoAddress, payload);
    }

    private static bool TryWriteModuleInfoForUnwind(
        CpuContext ctx,
        ulong outInfoAddress,
        KernelModuleRegistry.ModuleEntry module)
    {
        const int unwindInfoSize = 0x130;
        var payload = new byte[unwindInfoSize];
        BinaryPrimitives.WriteUInt64LittleEndian(payload, unwindInfoSize);
        WriteModuleName(payload, module.Name);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(0x108), module.EhFrameHeaderAddress);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(0x110), module.EhFrameAddress);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(0x118), module.EhFrameSize);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(0x120), module.BaseAddress);
        BinaryPrimitives.WriteUInt64LittleEndian(
            payload.AsSpan(0x128),
            module.EndAddress - module.BaseAddress);
        return ctx.Memory.TryWrite(outInfoAddress, payload);
    }

    private static bool TryWriteHostBoundaryModuleInfoForUnwind(
        CpuContext ctx,
        ulong outInfoAddress,
        ulong queriedAddress)
    {
        var boundaryBase = queriedAddress & ~(HostUnwindBoundarySize - 1);
        var module = new KernelModuleRegistry.ModuleEntry(
            Handle: 0,
            Name: "SharpEmuHostBoundary",
            Path: string.Empty,
            BaseAddress: boundaryBase,
            EndAddress: boundaryBase + HostUnwindBoundarySize,
            EntryPoint: 0,
            InitEntryPoint: 0,
            EhFrameHeaderAddress: 0,
            EhFrameAddress: 0,
            EhFrameSize: 0,
            StartState: KernelModuleRegistry.ModuleStartState.Started,
            IsMain: false,
            IsSystemModule: false);
        return TryWriteModuleInfoForUnwind(ctx, outInfoAddress, module);
    }

    private static void WriteModuleName(Span<byte> payload, string moduleName)
    {
        if (string.IsNullOrWhiteSpace(moduleName))
        {
            return;
        }

        var encoded = Encoding.UTF8.GetBytes(moduleName);
        var payloadLength = Math.Min(encoded.Length, ModuleInfoNameMaxBytes - 1);
        if (payloadLength > 0)
        {
            encoded.AsSpan(0, payloadLength).CopyTo(
                payload.Slice((int)ModuleInfoNameOffset, payloadLength));
        }
    }

    // SceKernelModuleSegmentInfo[4]: { u64 address; u32 size; i32 prot; }.
    internal const int ModuleInfoMaxSegments = 4;

    private static void WriteModuleSegments(
        Span<byte> payload,
        int offset,
        int countOffset,
        KernelModuleRegistry.ModuleEntry module)
    {
        var segments = KernelModuleRegistry.GetModuleSegments(module.Handle);
        if (segments.Length == 0)
        {
            // No program headers known (synthetic or host modules): one
            // aggregate readable and executable range.
            segments =
            [
                new KernelModuleRegistry.ModuleSegment(module.BaseAddress, module.EndAddress - module.BaseAddress, 5),
            ];
        }

        var count = Math.Min(segments.Length, ModuleInfoMaxSegments);
        for (var i = 0; i < count; i++)
        {
            var slot = offset + i * ModuleInfoSegmentSize;
            Debug.Assert(slot >= 0 && slot + ModuleInfoSegmentSize <= payload.Length);
            BinaryPrimitives.WriteUInt64LittleEndian(payload.Slice(slot), segments[i].Address);
            BinaryPrimitives.WriteUInt32LittleEndian(
                payload.Slice(slot + sizeof(ulong)),
                (uint)Math.Min(segments[i].Size, uint.MaxValue));
            BinaryPrimitives.WriteInt32LittleEndian(payload.Slice(slot + 12), segments[i].Protection);
        }

        BinaryPrimitives.WriteUInt32LittleEndian(payload.Slice(countOffset), (uint)count);
    }

    private static bool TryReadUtf8Z(CpuContext ctx, ulong address, int maxLength, out string value)
    {
        value = string.Empty;
        if (address == 0 || maxLength <= 0)
        {
            return false;
        }

        var bytes = new List<byte>(Math.Min(maxLength, 64));
        Span<byte> one = stackalloc byte[1];
        for (var i = 0; i < maxLength; i++)
        {
            if (!ctx.Memory.TryRead(address + (ulong)i, one))
            {
                return false;
            }

            if (one[0] == 0)
            {
                value = Encoding.UTF8.GetString(bytes.ToArray());
                return true;
            }

            bytes.Add(one[0]);
        }

        value = Encoding.UTF8.GetString(bytes.ToArray());
        return true;
    }

    private static ulong GetTlsScratchAddress(CpuContext ctx, ulong offset)
    {
        if (ctx.FsBase == 0)
        {
            return 0;
        }

        return unchecked(ctx.FsBase + offset);
    }

    private static bool TryWriteInt32(CpuContext ctx, ulong address, int value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        return ctx.Memory.TryWrite(address, bytes);
    }

    private static nint AllocateStackChkGuardObject()
    {
        try
        {
            var memory = Marshal.AllocHGlobal(sizeof(ulong) * 2);
            Marshal.WriteInt64(memory, unchecked((long)_stackChkGuardValue));
            Marshal.WriteInt64(IntPtr.Add(memory, sizeof(ulong)), unchecked((long)_stackChkGuardValue));
            return memory;
        }
        catch
        {
            return 0;
        }
    }

    private static ulong ResolveKernelTscFrequency()
    {
        var (frequencyHz, source) = SelectKernelTscFrequency(
            _rdtscReader is not null,
            Environment.GetEnvironmentVariable("SHARPEMU_TSC_FREQ_HZ"),
            TryCalibrateHostTscFrequency,
            TryResolveCpuidTscFrequency,
            Stopwatch.Frequency);
        TraceKernelTscFrequency(source, frequencyHz);
        return frequencyHz;
    }

    internal delegate bool TryGetFrequency(out ulong frequencyHz);

    // sceKernelReadTsc only returns the CPU's RDTSC when the host RDTSC reader is available
    // (currently 64-bit Windows); otherwise it falls back to the QPC-based Stopwatch. The
    // calibrated and CPUID frequencies both describe RDTSC, so reporting either while ReadTsc is
    // actually returning Stopwatch ticks makes sceKernelGetTscFrequency disagree with the counter,
    // and a guest computing elapsed = readTscDelta / frequency gets the wrong time on Linux and
    // macOS. Gate those on rdtscAvailable; otherwise report Stopwatch's own frequency so the pair
    // stays consistent.
    internal static (ulong FrequencyHz, string Source) SelectKernelTscFrequency(
        bool rdtscAvailable,
        string? overrideHzText,
        TryGetFrequency tryCalibrate,
        TryGetFrequency tryResolveCpuid,
        long stopwatchFrequency)
    {
        const ulong minSane = 1_000_000UL;

        if (!string.IsNullOrWhiteSpace(overrideHzText) &&
            ulong.TryParse(overrideHzText, out var overrideHz) &&
            overrideHz >= minSane)
        {
            return (overrideHz, "env");
        }

        if (rdtscAvailable)
        {
            if (tryCalibrate(out ulong calibratedHz) && calibratedHz >= minSane)
            {
                return (calibratedHz, "calibrated-rdtsc");
            }

            if (tryResolveCpuid(out ulong cpuidHz) && cpuidHz >= minSane)
            {
                return (cpuidHz, "cpuid");
            }
        }

        var hostQpc = stopwatchFrequency > 0
            ? unchecked((ulong)stopwatchFrequency)
            : DefaultKernelTscFrequency;
        return hostQpc >= minSane ? (hostQpc, "qpc") : (DefaultKernelTscFrequency, "default");
    }

    private static void TraceKernelTscFrequency(string source, ulong frequencyHz)
    {
        Console.Error.WriteLine($"[LOADER][INFO] Kernel TSC frequency: {frequencyHz} Hz ({source})");
    }

    private static bool TryResolveCpuidTscFrequency(out ulong frequencyHz)
    {
        frequencyHz = 0;
        if (!X86Base.IsSupported)
        {
            return false;
        }

        try
        {
            var leaf15 = X86Base.CpuId(unchecked((int)0x15), 0);
            uint denominator = unchecked((uint)leaf15.Eax);
            uint numerator = unchecked((uint)leaf15.Ebx);
            uint crystalHz = unchecked((uint)leaf15.Ecx);
            if (denominator != 0 && numerator != 0 && crystalHz != 0)
            {
                frequencyHz = ((ulong)crystalHz * numerator) / denominator;
                if (frequencyHz != 0)
                {
                    return true;
                }
            }

            var leaf16 = X86Base.CpuId(unchecked((int)0x16), 0);
            uint baseMHz = unchecked((uint)leaf16.Eax);
            if (baseMHz != 0)
            {
                frequencyHz = (ulong)baseMHz * 1_000_000UL;
                return true;
            }
        }
        catch
        {
        }

        return false;
    }

    private static bool TryCalibrateHostTscFrequency(out ulong frequencyHz)
    {
        frequencyHz = 0;
        if (!TryReadHostTsc(out _))
        {
            return false;
        }

        const int sampleCount = 3;
        Span<ulong> estimates = stackalloc ulong[sampleCount];
        int validSamples = 0;
        for (int i = 0; i < sampleCount; i++)
        {
            if (!TryCalibrateHostTscFrequencySample(out ulong estimate))
            {
                continue;
            }

            estimates[validSamples++] = estimate;
        }

        if (validSamples == 0)
        {
            return false;
        }

        estimates[..validSamples].Sort();
        frequencyHz = estimates[validSamples / 2];
        return frequencyHz != 0;
    }

    private static bool TryCalibrateHostTscFrequencySample(out ulong frequencyHz)
    {
        frequencyHz = 0;
        if (!TryReadHostTsc(out ulong startTsc))
        {
            return false;
        }

        long startQpc = Stopwatch.GetTimestamp();
        long minimumQpcDelta = Math.Max(Stopwatch.Frequency / 20, 1);
        long endQpc;
        ulong endTsc;
        do
        {
            Thread.SpinWait(256);
            endQpc = Stopwatch.GetTimestamp();
            if (!TryReadHostTsc(out endTsc))
            {
                return false;
            }
        }
        while (endQpc - startQpc < minimumQpcDelta);

        ulong deltaTsc = endTsc - startTsc;
        long deltaQpc = endQpc - startQpc;
        if (deltaTsc == 0 || deltaQpc <= 0)
        {
            return false;
        }

        frequencyHz = unchecked((deltaTsc * (ulong)Stopwatch.Frequency) / (ulong)deltaQpc);
        return frequencyHz != 0;
    }

    private static bool TryReadHostTsc(out ulong counter)
    {
        counter = 0;
        if (_rdtscReader is null)
        {
            return false;
        }

        try
        {
            counter = _rdtscReader();
            return counter != 0;
        }
        catch
        {
            counter = 0;
            return false;
        }
    }

    private static RdtscDelegate? CreateRdtscReader()
    {
        if (!Environment.Is64BitProcess ||
            RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return null;
        }

        try
        {
            nint stubAddress = VirtualAlloc(nint.Zero, (nuint)16, MemCommit | MemReserve, PageExecuteReadWrite);
            if (stubAddress == 0)
            {
                return null;
            }

            Span<byte> stub = stackalloc byte[]
            {
                0x0F, 0x31,
                0x48, 0xC1, 0xE2, 0x20,
                0x48, 0x09, 0xD0,
                0xC3,
            };

            Marshal.Copy(stub.ToArray(), 0, stubAddress, stub.Length);
            return Marshal.GetDelegateForFunctionPointer<RdtscDelegate>(stubAddress);
        }
        catch
        {
            return null;
        }
    }

    private static unsafe nint VirtualAlloc(nint lpAddress, nuint dwSize, uint flAllocationType, uint flProtect) =>
        (nint)HostMemory.Alloc((void*)lpAddress, dwSize, flAllocationType, flProtect);


    private static ulong AlignUp(ulong value, ulong alignment)
    {
        if (alignment <= 1)
        {
            return value;
        }

        var mask = alignment - 1;
        return (value + mask) & ~mask;
    }

    [SysAbiExport(
        Nid = "QvsZxomvUHs",
        ExportName = "sceKernelNanosleep",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelNanosleep(CpuContext ctx) => NanosleepCore(ctx, posix: false);

    [SysAbiExport(
        Nid = "NhpspxdjEKU",
        ExportName = "_nanosleep",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixNanosleepUnderscore(CpuContext ctx) => NanosleepCore(ctx, posix: true);

    [SysAbiExport(
        Nid = "yS8U2TGCe1A",
        ExportName = "nanosleep",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixNanosleep(CpuContext ctx) => NanosleepCore(ctx, posix: true);

    private static int NanosleepCore(CpuContext ctx, bool posix)
    {
        const int eFault = 14;
        const int eInvalid = 22;
        var requestAddress = ctx[CpuRegister.Rdi];
        var remainAddress = ctx[CpuRegister.Rsi];

        if (requestAddress == 0)
        {
            return NanosleepFailure(ctx, posix, eInvalid, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        Span<byte> timespecBuffer = stackalloc byte[16];
        if (!ctx.Memory.TryRead(requestAddress, timespecBuffer))
        {
            return NanosleepFailure(ctx, posix, eFault, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var tvSec = BinaryPrimitives.ReadInt64LittleEndian(timespecBuffer);
        var tvNsec = BinaryPrimitives.ReadInt64LittleEndian(timespecBuffer[sizeof(long)..]);
        if (tvSec < 0 || tvNsec < 0 || tvNsec >= 1_000_000_000L)
        {
            return NanosleepFailure(ctx, posix, eInvalid, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        if (tvSec == 0 && tvNsec == 0)
        {
            WriteRemainingTime(ctx, remainAddress, 0, 0);
            ctx[CpuRegister.Rax] = 0;
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        // Same policy as usleep: a sub-millisecond sleep neither pumps the scheduler (its
        // gate) nor rounds to a millisecond; longer sleeps wait for an absolute deadline
        // (mach_wait_until on macOS) instead of Thread.Sleep's millisecond granularity.
        if (tvSec == 0 && tvNsec < 1_000_000L)
        {
            _ = ShortUsleep((ulong)((tvNsec + 999) / 1000));
        }
        else
        {
            GuestThreadExecution.Scheduler?.Pump(ctx, posix ? "nanosleep" : "sceKernelNanosleep");
            HostTiming.SleepUntil(SharpEmu.Libs.Threading.GuestTime.DeadlineFromRelativeNanoseconds(tvSec, tvNsec));
        }

        WriteRemainingTime(ctx, remainAddress, 0, 0);
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int NanosleepFailure(
        CpuContext ctx,
        bool posix,
        int errnoValue,
        OrbisGen2Result sceResult)
    {
        if (posix)
        {
            TrySetErrno(ctx, errnoValue);
            ctx[CpuRegister.Rax] = unchecked((ulong)(-1L));
        }
        else
        {
            // sceKernelNanosleep reports the sce code, not the bare errno.
            _ = sceResult;
            return SharpEmu.Libs.Threading.GuestErrno.ReturnSce(ctx, errnoValue);
        }

        return (int)sceResult;
    }

    private static void WriteRemainingTime(CpuContext ctx, ulong remainAddress, long seconds, long nanoseconds)
    {
        if (remainAddress == 0)
        {
            return;
        }

        Span<byte> remainBuffer = stackalloc byte[16];
        BinaryPrimitives.WriteInt64LittleEndian(remainBuffer, seconds);
        BinaryPrimitives.WriteInt64LittleEndian(remainBuffer[sizeof(long)..], nanoseconds);
        ctx.Memory.TryWrite(remainAddress, remainBuffer);
    }

    [SysAbiExport(
        Nid = "QcteRwbsnV0",
        ExportName = "usleep",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixUsleep(CpuContext ctx) => KernelUsleep(ctx);

    [SysAbiExport(
        Nid = "HoLVWNanBBc",
        ExportName = "getpid",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int GetProcessId(CpuContext ctx)
    {
        var processId = GuestProcessId;
        ctx[CpuRegister.Rax] = unchecked((uint)processId);
        return processId;
    }
}
