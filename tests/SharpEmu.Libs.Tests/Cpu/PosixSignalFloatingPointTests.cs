// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#if SHARPEMU_CPU_MACOS_X64 || SHARPEMU_CPU_LINUX_X64
using System.Runtime.InteropServices;
using Iced.Intel;
using SharpEmu.HLE;
using Xunit;
using static Iced.Intel.AssemblerRegisters;
#if SHARPEMU_CPU_MACOS_X64
using Bridge = SharpEmu.Cpu.DirectExecution.MacOS.X64.PosixFaultBridge;
#else
using Bridge = SharpEmu.Cpu.DirectExecution.Linux.X64.PosixFaultBridge;
#endif

namespace SharpEmu.Libs.Tests.Cpu;

/// <summary>
/// The signal trampoline's floating-point switch: a guest that faults runs with the console's MXCSR (0x9FC0,
/// flush-to-zero and denormals-are-zero) and, on macOS and under Rosetta, the signal handler starts with it. The
/// trampoline's prologue must hand the managed handler the host's MXCSR and x87 control word and its epilogue must put
/// the interrupted ones back. Runs the exact emitted sequence around a probe, without a signal.
/// </summary>
public sealed unsafe class PosixSignalFloatingPointTests
{
    private const uint GuestMxcsr = CpuContext.InitialMxcsr;
    private const ushort GuestFpuControlWord = 0x027F;

    [NativeX64Fact]
    public void TheHandlerRunsWithTheHostFloatingPointControlAndTheGuestsComeBack()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return;
        }

        var a = new Assembler(64);
        // rdi: four results (inside MXCSR, inside control word, after MXCSR, after control word), 4 bytes each.
        a.push(rbx);
        a.mov(rbx, rdi);
        a.sub(rsp, 16);
        a.stmxcsr(__dword_ptr[rsp + 8]);
        a.fnstcw(__word_ptr[rsp + 12]);
        // The guest's state when it faults.
        a.mov(__dword_ptr[rsp], GuestMxcsr);
        a.ldmxcsr(__dword_ptr[rsp]);
        a.mov(__word_ptr[rsp + 4], GuestFpuControlWord);
        a.fldcw(__word_ptr[rsp + 4]);

        Bridge.EmitEnterHostFloatingPoint(a);
        a.stmxcsr(__dword_ptr[rbx]);
        a.fnstcw(__word_ptr[rbx + 4]);
        Bridge.EmitLeaveHostFloatingPoint(a);

        a.stmxcsr(__dword_ptr[rbx + 8]);
        a.fnstcw(__word_ptr[rbx + 12]);
        // The test runner's own state back.
        a.ldmxcsr(__dword_ptr[rsp + 8]);
        a.fldcw(__word_ptr[rsp + 12]);
        a.add(rsp, 16);
        a.pop(rbx);
        a.ret();

        var size = (nuint)Environment.SystemPageSize;
        var page = (byte*)HostMemory.Alloc(null, size, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
        Assert.True(page != null);
        try
        {
            a.Assemble(new PointerCodeWriter(page, (int)size), (ulong)page);
            var results = stackalloc uint[4];
            new Span<uint>(results, 4).Clear();
            ((delegate* unmanaged<uint*, void>)page)(results);

            // Status flags (bits 0-5) may be set by the probe itself; the control bits are what matter.
            Assert.Equal(Bridge.HostMxcsr, results[0] & ~0x3Fu);
            Assert.Equal(Bridge.HostFpuControlWord, (ushort)results[1]);
            Assert.Equal(GuestMxcsr, results[2] & ~0x3Fu);
            Assert.Equal(GuestFpuControlWord, (ushort)results[3]);
        }
        finally
        {
            Assert.True(HostMemory.Free(page, 0, HostMemory.MEM_RELEASE));
        }
    }
}

internal sealed unsafe class PointerCodeWriter(byte* target, int capacity) : CodeWriter
{
    private int _offset;

    public override void WriteByte(byte value)
    {
        Assert.True(_offset < capacity);
        target[_offset++] = value;
    }
}
#endif
