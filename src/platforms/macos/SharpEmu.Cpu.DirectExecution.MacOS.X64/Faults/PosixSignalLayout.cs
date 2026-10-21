// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Cpu.DirectExecution.MacOS.X64;

namespace SharpEmu.Cpu.DirectExecution.MacOS.X64;

/// <summary>
/// Everything about the Darwin signal ABI that
/// <see cref="PosixFaultBridge"/> needs: signal numbers, the
/// <c>struct sigaction</c> and <c>siginfo_t</c> layouts, the
/// <c>ucontext_t</c>/<c>__darwin_mcontext64</c> layout and the general-register
/// order.
/// </summary>
/// <remarks>
/// The Linux package has a file of the same name in the same namespace with
/// its own values, which is how the shared bridge compiles against constants
/// instead of OS branches. Rosetta 2 delivers the regular x86-64 layout to
/// translated processes, so nothing here is Rosetta-specific.
/// </remarks>
internal static unsafe class PosixSignalLayout
{
    internal const int SigBus = 10;
    internal const int SigSys = 12;

    // struct sigaction: the handler pointer leads, then { mask(4), flags(4) }.
    internal const int SigactionSize = 16;
    internal const int SigactionFlagsOffset = 12;

    internal const int SaSigInfo = 0x0040;
    internal const int SaNoDefer = 0x0010;

    // siginfo_t.si_addr: { signo, errno, code, pid, uid, status, addr }.
    internal const int SigInfoCodeOffset = 8;
    internal const int SigInfoAddressOffset = 24;

    // ucontext_t stores a pointer to __darwin_mcontext64 at +48 and the size of
    // that machine context at +40; the general registers live in its __ss
    // thread state after the 16-byte exception state.
    private const int UcontextMcontextOffset = 48;
    private const int UserContextMachineContextSizeOffset = 40;

    // Vector registers sit at the same offset in both signal frame formats.
    // The system headers i386/_mcontext.h and mach/i386/_structs.h define it.
    private const int MachineContextVectorRegistersOffset = 352;
    private const int McontextTrapNumberOffset = 0;
    private const int McontextErrOffset = 4;
    private const int McontextFaultAddressOffset = 8;

    /// <summary>
    /// Byte offsets of the general registers relative to
    /// <see cref="GetRegisterBase"/>, ordered to match the contiguous Win64
    /// CONTEXT block CTX_RAX..CTX_RIP (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
    /// r8..r15, rip). Verified against the x86-64 platform headers.
    /// </summary>
    internal static ReadOnlySpan<int> RegisterOffsets =>
        [16, 32, 40, 24, 72, 64, 56, 48, 80, 88, 96, 104, 112, 120, 128, 136, 144];

    /// <summary>__ss.__rflags, right after __rip.</summary>
    internal const int RflagsOffset = 152;

    internal static byte* GetRegisterBase(nint ucontext) =>
        ucontext == 0 ? null : *(byte**)((byte*)ucontext + UcontextMcontextOffset);

    internal static byte* GetVectorRegisterAddress(nint userContextAddress, byte* machineContext)
    {
        var machineContextSize = *(ulong*)((byte*)userContextAddress + UserContextMachineContextSizeOffset);
        return machineContextSize >= MachineContextVectorRegistersOffset + PosixFaultBridge.XmmBlockSize
            ? machineContext + MachineContextVectorRegistersOffset
            : null;
    }

    /// <summary>
    /// Darwin repeats the faulting address in the machine context, which is
    /// where it has to be read when siginfo does not carry one.
    /// </summary>
    internal static ulong GetFallbackFaultAddress(byte* registers) =>
        *(ulong*)(registers + McontextFaultAddressOffset);

    internal static ulong GetPageFaultErrorCode(byte* registers) =>
        *(uint*)(registers + McontextErrOffset);

    // Darwin's __es begins { uint16 trapno, uint16 cpu, uint32 err }.
    // Optional first-entry telemetry only; never a recovery classifier.
    internal static ulong GetTrapNumber(byte* registers) =>
        *(ushort*)(registers + McontextTrapNumberOffset);

    /// <summary>
    /// Fabricates the minimum of a Darwin signal frame that the bridge's
    /// warm-up pass has to walk: the machine-context pointer and its size.
    /// </summary>
    internal static void PrepareWarmUpContext(byte* userContext, byte* machineContext, ulong machineContextSize)
    {
        *(byte**)(userContext + UcontextMcontextOffset) = machineContext;
        *(ulong*)(userContext + UserContextMachineContextSizeOffset) = machineContextSize;

        // Load the Mach memory functions before the first signal uses them:
        // binding a lazy P/Invoke inside a signal frame needs the JIT.
        byte readProbe = 0;
        _ = MacOSHostMemoryProbe.TryRead((ulong)machineContext, &readProbe, 1);
    }

    [DllImport("libc", EntryPoint = "sigaction", SetLastError = true)]
    internal static extern int Sigaction(int signum, void* act, void* oldact);
}
