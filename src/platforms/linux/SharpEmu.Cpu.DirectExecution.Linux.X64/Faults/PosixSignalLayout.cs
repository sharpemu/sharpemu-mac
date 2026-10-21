// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Cpu.DirectExecution.Linux.X64;

/// <summary>
/// Everything about the Linux signal ABI that <see cref="PosixFaultBridge"/>
/// needs: signal numbers, the <c>struct sigaction</c> and <c>siginfo_t</c>
/// layouts, the <c>ucontext_t</c>/<c>mcontext_t</c> layout and the
/// general-register order.
/// </summary>
/// <remarks>
/// The macOS package has a file of the same name in the same namespace with
/// its own values, which is how the shared bridge compiles against constants
/// instead of OS branches. macOS and Linux share no fault context layout.
/// </remarks>
internal static unsafe class PosixSignalLayout
{
    internal const int SigBus = 7;
    internal const int SigSys = 31;

    // struct sigaction: the handler pointer leads, then glibc/musl pack
    // { mask(128), flags(4), restorer(8) }.
    internal const int SigactionSize = 152;
    internal const int SigactionFlagsOffset = 136;

    internal const int SaSigInfo = 0x0004;
    internal const int SaNoDefer = 0x40000000;

    // siginfo_t.si_addr: { signo, errno, code, pad32, addr }.
    internal const int SigInfoCodeOffset = 8;
    internal const int SigInfoAddressOffset = 16;

    // glibc embeds mcontext_t inline in ucontext_t at +40, with the general
    // registers in gregs[23].
    private const int UcontextGregsOffset = 40;
    private const int GregsErrOffset = 19 * 8;
    private const int GregsTrapNumberOffset = 20 * 8;

    // The kernel's x86-64 sigcontext places the FXSAVE-image pointer right
    // after the general registers it hands to the handler: err(152)
    // trapno(160) oldmask(168) cr2(176) fpstate(184), all relative to
    // GetRegisterBase. glibc and musl both overlay this kernel layout verbatim
    // (glibc's mcontext_t.fpregs is the same slot), so the offset is
    // libc-independent.
    private const int GregsFpstateOffset = 184;

    /// <summary>
    /// Byte offsets of the general registers relative to
    /// <see cref="GetRegisterBase"/>, ordered to match the contiguous Win64
    /// CONTEXT block CTX_RAX..CTX_RIP (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
    /// r8..r15, rip). Verified against the x86-64 platform headers.
    /// </summary>
    internal static ReadOnlySpan<int> RegisterOffsets =>
        [104, 112, 96, 88, 120, 80, 72, 64, 0, 8, 16, 24, 32, 40, 48, 56, 128];

    /// <summary>gregs[REG_EFL], right after REG_RIP.</summary>
    internal const int RflagsOffset = 136;

    internal static byte* GetRegisterBase(nint ucontext) =>
        ucontext == 0 ? null : (byte*)ucontext + UcontextGregsOffset;

    internal static byte* GetVectorRegisterAddress(nint userContextAddress, byte* machineContext)
    {
        _ = userContextAddress;
        byte* floatingPointState = *(byte**)(machineContext + GregsFpstateOffset);
        return floatingPointState != null ? floatingPointState + PosixFaultBridge.FxsaveXmmOffset : null;
    }

    /// <summary>
    /// Linux carries the faulting address only in siginfo, so there is no
    /// fallback to read from the machine context.
    /// </summary>
    internal static ulong GetFallbackFaultAddress(byte* registers)
    {
        _ = registers;
        return 0;
    }

    internal static ulong GetPageFaultErrorCode(byte* registers) =>
        *(ulong*)(registers + GregsErrOffset);

    // Linux sigcontext_64 and glibc REG_TRAPNO agree on this uint64 slot.
    // Optional first-entry telemetry only; never a recovery classifier.
    internal static ulong GetTrapNumber(byte* registers) =>
        *(ulong*)(registers + GregsTrapNumberOffset);

    /// <summary>
    /// Nothing to fabricate: the Linux machine context is embedded in the
    /// user context the bridge's warm-up pass already zeroed.
    /// </summary>
    internal static void PrepareWarmUpContext(byte* userContext, byte* machineContext, ulong machineContextSize)
    {
        _ = userContext;
        _ = machineContext;
        _ = machineContextSize;
    }

    [DllImport("libc", EntryPoint = "sigaction", SetLastError = true)]
    internal static extern int Sigaction(int signum, void* act, void* oldact);
}
