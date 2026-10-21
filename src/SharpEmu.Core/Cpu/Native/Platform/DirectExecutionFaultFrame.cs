// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Core.Cpu.Native.Platform;

/// <summary>
/// The normalized x64 fault frame exchanged between an OS bridge and shared
/// recovery. Its layout deliberately matches Win64 EXCEPTION_POINTERS/CONTEXT;
/// POSIX bridges translate their native frame into this layout. This is an
/// interop contract, not a host operating-system dependency.
/// </summary>
public static unsafe class DirectExecutionFaultFrame
{
    public const int ContextSize = 0x4D0;
    public const int EflagsOffset = 68;
    public const int RaxOffset = 120;
    public const int RspOffset = 152;
    public const int RipOffset = 248;
    public const int Xmm0Offset = 0x1A0;

    /// <summary>
    /// Exception code a POSIX bridge reports for SIGSYS: the host kernel
    /// refused a syscall (on macOS, a raw guest syscall with no class bits).
    /// RIP is past the instruction. Windows has no equivalent.
    /// </summary>
    public const uint HostSyscallTrapCode = 0xE0535953u;

    [StructLayout(LayoutKind.Sequential)]
    public struct Pointers
    {
        public Record* ExceptionRecord;
        public void* ContextRecord;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Record
    {
        public uint ExceptionCode;
        public uint ExceptionFlags;
        public Record* ExceptionRecord;
        public void* ExceptionAddress;
        public uint NumberParameters;
        public fixed ulong ExceptionInformation[15];
    }

    public static ulong ReadRegister(void* context, int offset) => *(ulong*)((byte*)context + offset);
    public static void WriteRegister(void* context, int offset, ulong value) => *(ulong*)((byte*)context + offset) = value;
}
