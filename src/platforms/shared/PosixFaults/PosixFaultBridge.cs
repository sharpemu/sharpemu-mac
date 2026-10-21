// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Iced.Intel;
using SharpEmu.Core.Cpu.Native.Platform;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using static Iced.Intel.AssemblerRegisters;

#if SHARPEMU_CPU_MACOS_X64
namespace SharpEmu.Cpu.DirectExecution.MacOS.X64;
#elif SHARPEMU_CPU_LINUX_X64
namespace SharpEmu.Cpu.DirectExecution.Linux.X64;
#else
#error A POSIX fault bridge must be compiled by exactly one OS execution package.
#endif

/// <summary>
/// POSIX bridge from a <c>sigaction</c> fault handler to the shared Win64
/// recovery chain.
/// </summary>
/// <remarks>
/// The handler rebuilds the <c>EXCEPTION_POINTERS</c> view the shared handlers
/// expect (Win64 CONTEXT register offsets) from the signal's mcontext, runs the
/// same recovery chain the Windows vectored path uses (unresolved-import trap
/// sentinels, demand-paging of lazily-committed guest pages, fault
/// diagnostics), and writes register changes back into the mcontext so
/// sigreturn resumes the repaired guest. Unrecovered faults are forwarded to
/// the previously installed handler so the .NET runtime keeps turning its own
/// faults into managed exceptions.
///
/// <para>Linked by source into SharpEmu.Cpu.DirectExecution.MacOS.X64 and
/// SharpEmu.Cpu.DirectExecution.Linux.X64 — there is no shared POSIX execution
/// assembly, for the same reason there is no Host.Posix one. Everything that
/// differs between Darwin and Linux (signal numbers, <c>struct sigaction</c>
/// and <c>siginfo_t</c> layout, <c>ucontext</c>/<c>mcontext</c> layout, the
/// general-register order, where the vector registers and the page-fault error
/// code live, and the <c>sigaction</c> declaration itself) lives in each
/// package's own <see cref="PosixSignalLayout"/>, so this file contains no OS
/// check and each package compiles it against constants, not branches.</para>
///
/// <para>The shared recovery entries are raw function pointers published before
/// interception starts. The bridge creates no delegates or interface dispatch
/// during a signal. Existing optional tracing and shared diagnostic/recovery
/// work are preserved; this extraction does not claim those older paths are
/// async-signal-safe.</para>
/// </remarks>
internal static unsafe class PosixFaultBridge
{
    private const int PosixSigIll = 4;
    private const int PosixSigTrap = 5;
    private const int PosixSigAbort = 6;
    private const int PosixSigFpe = 8;
    private const int PosixSigSegv = 11;

    // Inside an FXSAVE image the XMM registers start at +160 (32-byte header
    // plus 8 legacy x87/MMX slots of 16 bytes) - the same relative position
    // they occupy in the Win64 CONTEXT's FltSave area.
    internal const int FxsaveXmmOffset = 160;
    internal const int XmmBlockSize = 16 * 16;

    // Host stack below the saved host RSP that is still treated as the host frames in use.
    private const ulong HostStackSearchBytes = 0x100_0000;
    private const int SignalStackGap = 0x1000;

    private static bool _handlersInstalled;
    private static nint _signalTrampoline;
    private static bool _rawRecoveryEnabled;
    private static bool _signalWarmup;
    private static readonly nint[] _previousActions = new nint[32];
    private static int _signalTraceCount;
    private static long _perfSignalCount;
    private static readonly bool _perfSignalCounter =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_PERF_MEM"), "1", StringComparison.Ordinal);

    // The shared recovery entries, stored as raw function pointers so the
    // handler never has to build a callable while handling a fault.
    private static delegate* managed<void*, int> _rawRecovery;
    private static delegate* managed<void*, int> _recovery;

    // [0] the host RSP TLS slot (uint.MaxValue while unknown), [8] the Win64 TlsGetValue stub.
    private static nint _signalStackCells;

    [ThreadStatic]
    private static int _signalHandlerDepth;

    private static readonly bool PosixGpuFaultFastPath =
        Environment.GetEnvironmentVariable("SHARPEMU_POSIX_GPU_FAULT_FAST_PATH") != "0";

    // CPU packages are registered outward by the application and are not a
    // dependency of the HLE export assemblies ModuleManager walks. Calling the
    // native probe directly warms the kernel routine but misses these newly
    // extracted concrete interface entries. Execute their dispatch once on a
    // normal host stack, before a signal can cause their first JIT/translation.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.NoOptimization)]
    internal static void WarmUpPlatformEntries(IDirectExecutionPlatform platform)
    {
        byte source = 0x53, destination = 0;
        _ = platform.Capabilities;
        _ = platform.TryReadHostMemory((ulong)(&source), &destination, 1);
        _ = platform.TryReadExecutableHostMemory((ulong)(&source), &destination, 1);
    }

    internal static void Install(in DirectExecutionFaultHandlers handlers)
    {
        if (string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_DISABLE_POSIX_SIGNALS"), "1", StringComparison.Ordinal))
        {
            Console.Error.WriteLine("[LOADER][WARN] POSIX signal exception bridge disabled by SHARPEMU_DISABLE_POSIX_SIGNALS=1; guest faults will not be recovered.");
            return;
        }

        _rawRecovery = handlers.RawRecovery;
        _recovery = handlers.Recovery;
        if (_handlersInstalled)
        {
            PublishStackState(handlers.HostRspSlotTlsIndex, handlers.TlsGetValueAddress);
            return;
        }

        _rawRecoveryEnabled = !string.Equals(
            Environment.GetEnvironmentVariable("SHARPEMU_DISABLE_RAW_HANDLER"), "1", StringComparison.Ordinal);
        if (!_rawRecoveryEnabled)
        {
            Console.Error.WriteLine("[LOADER][INFO] Raw sentinel recovery disabled by SHARPEMU_DISABLE_RAW_HANDLER=1");
        }

        InstallGuestCodeReader();
        GuestImageWriteTracker.InitializeNativeFaultDiagnostics();
        WarmUpSignalPath(handlers.WarmUp);
        GuestImageWriteTracker.WarmUp();

        _signalTrampoline = CreateSignalTrampoline();
        PublishStackState(handlers.HostRspSlotTlsIndex, handlers.TlsGetValueAddress);
        if (!InstallSignalHandler(PosixSigSegv) ||
            !InstallSignalHandler(PosixSignalLayout.SigBus) ||
            !InstallSignalHandler(PosixSigIll) ||
            !InstallSignalHandler(PosixSigTrap) ||
            !InstallSignalHandler(PosixSigAbort) ||
            !InstallSignalHandler(PosixSigFpe) ||
            !InstallSignalHandler(PosixSignalLayout.SigSys))
        {
            throw new InvalidOperationException("Failed to install POSIX fault signal handlers");
        }

        _handlersInstalled = true;
        Console.Error.WriteLine("[LOADER][INFO] POSIX signal exception bridge installed (SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGSYS)");
    }

    /// <summary>
    /// The signal handlers stay installed for the life of the process: a guest
    /// worker thread can still fault after the session that installed them
    /// asked to stop, and removing a handler a faulting thread is about to
    /// enter cannot be made safe.
    /// </summary>
    internal static void Remove()
    {
    }

    /// <summary>
    /// Test hook: arms the shared recovery entries without installing the
    /// signal handlers, which is process-wide and cannot be undone. Lets a
    /// fixture drive <see cref="TryHandleFault"/> with a fabricated frame.
    /// </summary>
    internal static void UseSharedRecoveryEntriesForTest(nint rawRecovery, nint recovery)
    {
        _rawRecovery = (delegate* managed<void*, int>)rawRecovery;
        _recovery = (delegate* managed<void*, int>)recovery;
    }

    private static void PublishStackState(uint hostRspSlotTlsIndex, nint tlsGetValueAddress)
    {
        if (_signalStackCells == 0)
        {
            return;
        }

        *(nint*)(_signalStackCells + 8) = tlsGetValueAddress;
        *(uint*)_signalStackCells = hostRspSlotTlsIndex;
    }

    /// <summary>
    /// Runs the signal-recovery path once with fabricated inputs before the
    /// handlers are installed. The first entry into the handler must not
    /// require JIT compilation (a fault can interrupt arbitrary runtime
    /// states), and under Rosetta 2 the signal trampoline cannot enter x86
    /// code that has never been executed (and therefore never translated): a
    /// cold handler is silently never invoked and the faulting instruction
    /// retries forever.
    /// </summary>
    private static void WarmUpSignalPath(delegate* managed<void> warmUp)
    {
        byte* testUserContext = stackalloc byte[512];
        new Span<byte>(testUserContext, 512).Clear();
        byte* testMachineContext = stackalloc byte[1024];
        new Span<byte>(testMachineContext, 1024).Clear();
        PosixSignalLayout.PrepareWarmUpContext(testUserContext, testMachineContext, 1024);

        _signalWarmup = true;
        try
        {
            ((delegate* unmanaged<int, nint, nint, void>)&HandlePosixSignal)(PosixSigSegv, 0, (nint)testUserContext);

            // Warm the branches the fabricated fault above skips without
            // spamming diagnostics: the benign-exception path through the
            // shared recovery chain, the lazy-commit probe, and the chain
            // helper (signal 0 has no saved action and sigaction(0, ...) fails
            // with EINVAL).
            warmUp();
            ChainPreviousAction(0, 0, 0);
        }
        finally
        {
            _signalWarmup = false;
        }
    }

    private static bool InstallSignalHandler(int signal)
    {
        byte* action = stackalloc byte[PosixSignalLayout.SigactionSize];
        new Span<byte>(action, PosixSignalLayout.SigactionSize).Clear();
        *(nint*)action = _signalTrampoline;
        // No SA_ONSTACK: the runtime's alternate stacks are far too small for
        // the recovery/diagnostic path (JIT compilation of cold handler code
        // can run inside the signal frame). Guest faults deliver onto the 2MB
        // guest stack, host faults onto the regular thread stack — the same
        // stacks Windows dispatches exceptions on.
        *(int*)(action + PosixSignalLayout.SigactionFlagsOffset) = PosixSignalLayout.SaSigInfo | PosixSignalLayout.SaNoDefer;

        var previous = (byte*)NativeMemory.AllocZeroed((nuint)PosixSignalLayout.SigactionSize);
        if (PosixSignalLayout.Sigaction(signal, action, previous) != 0)
        {
            NativeMemory.Free(previous);
            Console.Error.WriteLine($"[LOADER][ERROR] sigaction({signal}) failed: errno={Marshal.GetLastPInvokeError()}");
            return false;
        }

        _previousActions[signal] = (nint)previous;
        return true;
    }

    // Neutral diagnostic route/reason scalars; no recovery ABI changes.
    private const int TraceEarlyImage = 1, TraceGpuFast = 2, TraceRawCore = 3,
        TraceCoreRecovery = 4, TraceNoCoreCallback = 5, TraceBridgeException = 6;
    private const int TraceRecovered = 0, TraceDeclined = 1, TraceMissingContext = 2,
        TraceHandlerException = 3, TraceWarmup = 4;

    [UnmanagedCallersOnly]
    private static void HandlePosixSignal(int signal, nint siginfo, nint ucontext)
    {
        if (_signalHandlerDepth > 0)
        {
            // A fault inside our own fault handler (diagnostics touched an
            // unmapped address): restore the default action and return so the
            // re-executed instruction terminates the process.
            RestoreDefaultAction(signal);
            return;
        }

        _signalHandlerDepth++;
        if (_perfSignalCounter)
        {
            var n = Interlocked.Increment(ref _perfSignalCount);
            if (n % 100000 == 0)
            {
                Console.Error.WriteLine($"[PERF][MEM] posix_faults={n}");
            }
        }
        long invocation = 0;
        var recovered = false;
        var route = TraceNoCoreCallback;
        var reason = TraceDeclined;
        try
        {
            invocation = BeginNativeSignalDiagnostics(signal, siginfo, ucontext);
            // Only a classified write can belong to guest-image tracking.
            // Use the same normalized address as the later fault bridge:
            // Darwin can provide the address in mcontext when si_addr is zero.
            if (TryHandleTrackedImageWriteFaultForInvocation(signal, siginfo, ucontext, invocation))
            {
                recovered = true;
                route = TraceEarlyImage;
                reason = TraceRecovered;
                return;
            }

            // GPU-tracked pages fault on every first CPU access after the GPU or an upload used them,
            // thousands of times a second in a frame. Resolving them needs only the address and the
            // access kind, so they skip the Win64 context conversion and the other handlers.
            if (PosixGpuFaultFastPath && !_signalWarmup &&
                (signal == PosixSigSegv || signal == PosixSignalLayout.SigBus) &&
                siginfo != 0 &&
                TryResolveGpuFault(siginfo, ucontext))
            {
                recovered = true;
                route = TraceGpuFast;
                reason = TraceRecovered;
                return;
            }

            if (TryHandleFaultWithDiagnosticRoute(signal, siginfo, ucontext, out route, out reason))
            {
                recovered = true;
                return;
            }
        }
        catch
        {
            // A managed exception must never unwind out of a signal frame.
            route = TraceBridgeException;
            reason = TraceHandlerException;
        }
        finally
        {
            try
            {
                GuestImageWriteTracker.RecordNativeFaultDisposition(invocation, recovered, route, reason);
            }
            catch
            {
                // Optional diagnostics must never unwind from a native signal.
                // A missing final slot remains explicitly unknown to the reader.
            }
            _signalHandlerDepth--;
        }

        ChainPreviousAction(signal, siginfo, ucontext);
    }

    internal static bool TryHandleTrackedImageWriteFault(int signal, nint siginfo, nint ucontext) =>
        TryHandleTrackedImageWriteFaultForInvocation(signal, siginfo, ucontext,
            BeginNativeSignalDiagnostics(signal, siginfo, ucontext));

    private static long BeginNativeSignalDiagnostics(int signal, nint siginfo, nint ucontext)
    {
        if (!GuestImageWriteTracker.NativeFaultDiagnosticsEnabled) return 0;
        byte* registers = PosixSignalLayout.GetRegisterBase(ucontext);
        var rawAddress = siginfo == 0 ? 0 : *(ulong*)((byte*)siginfo + PosixSignalLayout.SigInfoAddressOffset);
        var signalCode = siginfo == 0 ? 0 : *(int*)((byte*)siginfo + PosixSignalLayout.SigInfoCodeOffset);
        var address = registers == null ? rawAddress : GetFaultAddress(siginfo, registers);
        var rip = registers == null ? 0 : *(ulong*)(registers + PosixSignalLayout.RegisterOffsets[16]);
        var access = registers != null && (signal == PosixSigSegv || signal == PosixSignalLayout.SigBus)
            ? (int)GetAccessType(registers, address, rip) : -1;
        return GuestImageWriteTracker.RecordNativeImageFaultEntry(
            signal, rawAddress, signalCode, address, access,
            registers == null ? 0 : PosixSignalLayout.GetPageFaultErrorCode(registers), rip,
            registers == null ? 0 : PosixSignalLayout.GetTrapNumber(registers), registers != null);
    }

    private static bool TryHandleTrackedImageWriteFaultForInvocation(
        int signal, nint siginfo, nint ucontext, long invocation)
    {
        var handled = false;
        if (signal == PosixSigSegv || signal == PosixSignalLayout.SigBus)
        {
            byte* registers = PosixSignalLayout.GetRegisterBase(ucontext);
            if (registers != null)
            {
                ulong address = GetFaultAddress(siginfo, registers);
                ulong rip = *(ulong*)(registers + PosixSignalLayout.RegisterOffsets[16]);
                handled = GetAccessType(registers, address, rip) == 1 &&
                    GuestImageWriteTracker.TryHandleWriteFaultWithNativeDiagnostics(address, invocation);
            }
        }
        GuestImageWriteTracker.RecordNativeImageFaultResult(handled, invocation);
        return handled;
    }

    // Guest code is identity-mapped, and a diagnostic only ever reads at an address the
    // guest has just executed, so a direct read is safe; the indirection keeps the HLE
    // layer free of unsafe code and of a disassembler reference.
    private static void InstallGuestCodeReader()
    {
        GuestCodeReader.Reader ??= (address, count) =>
        {
            if (address < 0x10000 || count <= 0 || count > 64)
            {
                return null;
            }

            var bytes = new byte[count];
            new ReadOnlySpan<byte>((void*)address, count).CopyTo(bytes);
            return bytes;
        };

        GuestCodeReader.Disassembler ??= (address, bytes) =>
        {
            var decoder = Decoder.Create(64, new ByteArrayCodeReader(bytes), address);
            var instruction = decoder.Decode();
            var formatter = new NasmFormatter();
            var output = new StringOutput();
            formatter.Format(instruction, output);
            return output.ToStringAndReset();
        };
    }

    private static bool TryResolveGpuFault(nint siginfo, nint ucontext)
    {
        byte* registers = PosixSignalLayout.GetRegisterBase(ucontext);
        if (registers == null)
        {
            return false;
        }

        ulong faultAddress = GetFaultAddress(siginfo, registers);
        ReadOnlySpan<int> offsets = PosixSignalLayout.RegisterOffsets;
        ulong rip = *(ulong*)(registers + offsets[16]);
        var kind = GetAccessType(registers, faultAddress, rip) switch
        {
            0 => FaultKind.Read,
            1 => FaultKind.Write,
            _ => FaultKind.Unknown,
        };
        if (kind == FaultKind.Unknown)
        {
            return false;
        }

        // A watch that fires knows the address but not the instruction. Publish the
        // machine state so a recovery step that cares can name the guest code.
        if (GuestSlotTrap.Armed)
        {
            Span<ulong> general = stackalloc ulong[16];
            for (var index = 0; index < general.Length && index < offsets.Length; index++)
            {
                general[index] = *(ulong*)(registers + offsets[index]);
            }

            GuestFaultContext.Publish(rip, general);
        }

        try
        {
            return GuestGpuMemoryHook.TryResolveFault(kind, faultAddress);
        }
        finally
        {
            GuestFaultContext.Clear();
        }
    }

    /// <summary>
    /// Rebuilds the Win64 fault view from a signal frame and runs the shared
    /// recovery chain over it. Internal rather than private so the per-OS
    /// fault fixtures can drive it with a fabricated frame.
    /// </summary>
    internal static bool TryHandleFault(int signal, nint siginfo, nint ucontext) =>
        TryHandleFaultWithDiagnosticRoute(signal, siginfo, ucontext, out _, out _);

    private static bool TryHandleFaultWithDiagnosticRoute(
        int signal, nint siginfo, nint ucontext, out int route, out int reason)
    {
        route = TraceNoCoreCallback;
        reason = TraceMissingContext;
        byte* registers = PosixSignalLayout.GetRegisterBase(ucontext);
        if (registers == null)
        {
            return false;
        }

        byte* contextRecord = stackalloc byte[DirectExecutionFaultFrame.ContextSize];
        new Span<byte>(contextRecord, DirectExecutionFaultFrame.ContextSize).Clear();
        ReadOnlySpan<int> offsets = PosixSignalLayout.RegisterOffsets;
        for (int i = 0; i < offsets.Length; i++)
        {
            DirectExecutionFaultFrame.WriteRegister(contextRecord, DirectExecutionFaultFrame.RaxOffset + i * 8, *(ulong*)(registers + offsets[i]));
        }

        // EFLAGS too: instruction recovery reads CF/OF as inputs (ADCX/ADOX carry in
        // through them) and writes the flags its instruction defines.
        *(uint*)(contextRecord + DirectExecutionFaultFrame.EflagsOffset) =
            (uint)*(ulong*)(registers + PosixSignalLayout.RflagsOffset);

        // Copy the vector registers that instruction recovery can change.
        // Keep all other floating-point state unchanged.
        byte* vectorRegisters = PosixSignalLayout.GetVectorRegisterAddress(ucontext, registers);
        if (vectorRegisters != null)
        {
            Buffer.MemoryCopy(
                vectorRegisters,
                contextRecord + DirectExecutionFaultFrame.Xmm0Offset,
                XmmBlockSize,
                XmmBlockSize);
        }
        // Tell the shared recovery chain whether it may change vector registers.
        DirectExecutionFaultContext.VectorRegistersBridged = vectorRegisters != null;

        DirectExecutionFaultFrame.Record record = default;
        record.ExceptionAddress = (void*)DirectExecutionFaultFrame.ReadRegister(contextRecord, DirectExecutionFaultFrame.RipOffset);
        if (signal == PosixSigIll)
        {
            record.ExceptionCode = 3221225501u;
        }
        else if (signal == PosixSigTrap)
        {
            record.ExceptionCode = 2147483651u;
        }
        else if (signal == PosixSigAbort)
        {
            record.ExceptionCode = 1073741845u;
        }
        else if (signal == PosixSignalLayout.SigSys)
        {
            // A raw guest syscall the host kernel refused; RIP is past it.
            record.ExceptionCode = DirectExecutionFaultFrame.HostSyscallTrapCode;
        }
        else if (signal == PosixSigFpe)
        {
            // STATUS_INTEGER_DIVIDE_BY_ZERO: #DE from div/idiv (zero divisor or quotient overflow).
            record.ExceptionCode = 3221225620u;
        }
        else
        {
            ulong faultAddress = GetFaultAddress(siginfo, registers);
            record.ExceptionCode = 3221225477u;
            record.NumberParameters = 2;
            record.ExceptionInformation[0] = GetAccessType(registers, faultAddress, DirectExecutionFaultFrame.ReadRegister(contextRecord, DirectExecutionFaultFrame.RipOffset));
            record.ExceptionInformation[1] = faultAddress;
        }

        DirectExecutionFaultFrame.Pointers pointers;
        pointers.ExceptionRecord = &record;
        pointers.ContextRecord = contextRecord;

        int traceIndex = _signalWarmup ? 0 : Interlocked.Increment(ref _signalTraceCount);
        bool traceSignal = traceIndex > 0 && (traceIndex <= 16 || traceIndex % 1024 == 0 ||
            string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_POSIX_SIGNALS"), "1", StringComparison.Ordinal));
        if (traceSignal)
        {
            Console.Error.WriteLine(
                $"[LOADER][TRACE] posix-signal#{traceIndex}: sig={signal} rip=0x{DirectExecutionFaultFrame.ReadRegister(contextRecord, DirectExecutionFaultFrame.RipOffset):X16} " +
                $"fault=0x{record.ExceptionInformation[1]:X16} access={record.ExceptionInformation[0]} rsp=0x{DirectExecutionFaultFrame.ReadRegister(contextRecord, DirectExecutionFaultFrame.RspOffset):X16}");
            Console.Error.Flush();
        }

        // Sentinel recovery runs first: on Windows both vectored handlers see
        // every fault anyway, and recovering here avoids dumping the full
        // recovery-chain diagnostics for each recoverable trap.
        int disposition = 0;
        if (_rawRecoveryEnabled && _rawRecovery != null)
        {
            route = TraceRawCore;
            disposition = _rawRecovery(&pointers);
        }
        if (disposition != -1 && !_signalWarmup && _recovery != null)
        {
            route = TraceCoreRecovery;
            disposition = _recovery(&pointers);
        }
        reason = _signalWarmup ? TraceWarmup : disposition == -1 ? TraceRecovered : TraceDeclined;
        if (traceSignal)
        {
            Console.Error.WriteLine(
                $"[LOADER][TRACE] posix-signal#{traceIndex}: recovered={disposition == -1} new_rip=0x{DirectExecutionFaultFrame.ReadRegister(contextRecord, DirectExecutionFaultFrame.RipOffset):X16}");
            Console.Error.Flush();
        }
        if (disposition != -1 && !_signalWarmup)
        {
            return false;
        }

        for (int i = 0; i < offsets.Length; i++)
        {
            *(ulong*)(registers + offsets[i]) = DirectExecutionFaultFrame.ReadRegister(contextRecord, DirectExecutionFaultFrame.RaxOffset + i * 8);
        }

        var rflags = (ulong*)(registers + PosixSignalLayout.RflagsOffset);
        *rflags = (*rflags & 0xFFFF_FFFF_0000_0000UL) |
            *(uint*)(contextRecord + DirectExecutionFaultFrame.EflagsOffset);
        if (vectorRegisters != null)
        {
            Buffer.MemoryCopy(
                contextRecord + DirectExecutionFaultFrame.Xmm0Offset,
                vectorRegisters,
                XmmBlockSize,
                XmmBlockSize);
        }
        return true;
    }

    private static ulong GetFaultAddress(nint siginfo, byte* registers)
    {
        ulong address = siginfo != 0 ? *(ulong*)((byte*)siginfo + PosixSignalLayout.SigInfoAddressOffset) : 0;
        if (address == 0)
        {
            address = PosixSignalLayout.GetFallbackFaultAddress(registers);
        }

        return address;
    }

    private static ulong GetAccessType(byte* registers, ulong faultAddress, ulong rip)
    {
        // x86 page-fault error code: bit 1 = write access, bit 4 = instruction
        // fetch. Fall back to comparing the fault address against RIP when
        // the error code is not populated (e.g. under Rosetta 2 translation).
        ulong error = PosixSignalLayout.GetPageFaultErrorCode(registers);
        if ((error & 0x10) != 0)
        {
            return 8;
        }
        if ((error & 0x2) != 0)
        {
            return 1;
        }

        return faultAddress != 0 && faultAddress == rip ? 8u : 0u;
    }

    private static void RestoreDefaultAction(int signal)
    {
        byte* action = stackalloc byte[PosixSignalLayout.SigactionSize];
        new Span<byte>(action, PosixSignalLayout.SigactionSize).Clear();
        _ = PosixSignalLayout.Sigaction(signal, action, null);
    }

    private static void ChainPreviousAction(int signal, nint siginfo, nint ucontext)
    {
        byte* previous = (uint)signal < (uint)_previousActions.Length
            ? (byte*)_previousActions[signal]
            : null;
        nint handler = previous != null ? *(nint*)previous : 0;
        if (handler == 0)
        {
            // SIG_DFL (or nothing saved): reinstate the default action and
            // return, so re-executing the faulting instruction terminates the
            // process with the original fault context intact.
            RestoreDefaultAction(signal);
            return;
        }
        if (handler == 1)
        {
            // SIG_IGN
            return;
        }

        int flags = *(int*)(previous + PosixSignalLayout.SigactionFlagsOffset);
        if ((flags & PosixSignalLayout.SaSigInfo) != 0)
        {
            ((delegate* unmanaged<int, nint, nint, void>)handler)(signal, siginfo, ucontext);
        }
        else
        {
            ((delegate* unmanaged<int, void>)handler)(signal);
        }
    }

    // Guest faults are delivered on the guest stack. The runtime only knows the thread's own stack,
    // so managed code that runs on the guest stack can make a garbage collection misread the frames
    // of that thread. Windows swaps the stack bounds in the TEB while guest code runs; POSIX cannot,
    // so the signal handler moves to the host stack below the frames that entered guest code first.
    private static nint CreateSignalTrampoline()
    {
        _signalStackCells = (nint)NativeMemory.AllocZeroed(16);
        *(uint*)_signalStackCells = uint.MaxValue;

        var a = new Assembler(64);
        var direct = a.CreateLabel();
        var call = a.CreateLabel();
        var done = a.CreateLabel();
        a.push(rbx);
        a.push(r12);
        a.push(r13);
        a.push(r14);
        a.push(r15);
        a.mov(r12d, edi);
        a.mov(r13, rsi);
        a.mov(r14, rdx);
        a.mov(rbx, rsp);

        // The saved host RSP of this thread, when it entered guest code.
        a.mov(rax, (ulong)_signalStackCells);
        a.mov(ecx, __dword_ptr[rax]);
        a.cmp(ecx, -1);
        a.je(direct);
        a.mov(rax, __qword_ptr[rax + 8]);
        a.test(rax, rax);
        a.je(direct);
        a.sub(rsp, 48);
        a.call(rax);
        a.add(rsp, 48);
        a.test(rax, rax);
        a.je(direct);
        a.mov(r15, __qword_ptr[rax]);
        a.test(r15, r15);
        a.je(direct);

        // Already on the host stack below the saved RSP: an import or host code faulted.
        a.cmp(rbx, r15);
        a.jae(call);
        a.mov(rax, r15);
        a.sub(rax, rbx);
        a.cmp(rax, (int)HostStackSearchBytes);
        a.jb(direct);

        a.Label(ref call);
        a.lea(rsp, __[r15 - SignalStackGap]);
        a.and(rsp, -16);
        a.Label(ref direct);
        a.mov(edi, r12d);
        a.mov(rsi, r13);
        a.mov(rdx, r14);
        a.mov(rax, (ulong)(nint)(delegate* unmanaged<int, nint, nint, void>)&HandlePosixSignal);
        a.call(rax);
        a.Label(ref done);
        a.mov(rsp, rbx);
        a.pop(r15);
        a.pop(r14);
        a.pop(r13);
        a.pop(r12);
        a.pop(rbx);
        a.ret();

        var page = (byte*)HostMemory.Alloc(null, 4096, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
        if (page == null)
        {
            throw new OutOfMemoryException("Failed to allocate the POSIX signal trampoline page");
        }

        var writer = new SpanCodeWriter(new Span<byte>(page, 4096));
        a.Assemble(writer, (ulong)page);
        if (!HostMemory.Protect(page, 4096, HostMemory.PAGE_EXECUTE_READ, out _))
        {
            throw new InvalidOperationException("Failed to protect the POSIX signal trampoline page");
        }

        return (nint)page;
    }

    private sealed class SpanCodeWriter : CodeWriter
    {
        private readonly byte* _start;
        private readonly int _length;
        private int _offset;

        public SpanCodeWriter(Span<byte> destination)
        {
            fixed (byte* start = destination)
            {
                _start = start;
            }

            _length = destination.Length;
        }

        public override void WriteByte(byte value)
        {
            if (_offset >= _length)
            {
                throw new InvalidOperationException("The POSIX signal trampoline does not fit its page.");
            }

            _start[_offset++] = value;
        }
    }
}
