// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpEmu.Core.Cpu.Native;
using SharpEmu.Core.Cpu.Native.Platform;
using SharpEmu.HLE;
using Xunit;

namespace SharpEmu.Libs.Tests.Cpu;

/// <summary>
/// What a guest import costs to cross into managed code, measured through the real import trampoline:
/// native code calls the trampoline the backend emits for an import, which switches to the host
/// stack and enters the managed gateway (the reverse P/Invoke under Rosetta), and the gateway
/// dispatches a managed export that only sets RAX. Opt-in (SHARPEMU_GATEWAY_BENCH_OUTPUT names the
/// report file): ordinary runs do not collect timing data.
/// </summary>
public sealed unsafe class GatewayCrossingMeasurementTests
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.NonPublic;

    [NativeX64Fact]
    public void GatewayCrossingMeasurement()
    {
        var outputPath = Environment.GetEnvironmentVariable("SHARPEMU_GATEWAY_BENCH_OUTPUT");
        if (string.IsNullOrEmpty(outputPath))
        {
            return;
        }

        var lines = new List<string>();
        Exception? failure = null;
        // The gateway runs 64 KiB below the caller on the same thread stack, as it runs below the
        // frame that entered guest code in a game: give the thread room.
        var thread = new Thread(() =>
        {
            try
            {
                Measure(lines);
            }
            catch (Exception exception)
            {
                failure = exception;
            }
        }, 16 << 20);
        thread.Start();
        thread.Join();
        File.WriteAllLines(outputPath, lines);
        Assert.Null(failure);
    }

    private static void Measure(List<string> lines)
    {
        Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
        var modules = new ModuleManager();
        modules.Freeze();
        using var backend = new DirectExecutionBackend(modules);
        var type = typeof(DirectExecutionBackend);
        type.GetField("_cpuContext", Instance)!.SetValue(backend, new CpuContext(new FakeCpuMemory(0x1_0000_0000, 0x1000), Generation.Gen5));
        // Entry 0 is classified a trivial leaf by its NID; entry 1 takes the full import path.
        var entryType = type.GetNestedType("ImportStubEntry", BindingFlags.NonPublic)!;
        var entries = Array.CreateInstance(entryType, 2);
        var nids = new[] { "V++UgBtQhn0", "gateway-bench-full-path" };
        for (var index = 0; index < nids.Length; index++)
        {
            var export = new ExportedFunction("bench", nids[index], "gateway bench", Generation.Gen5, c =>
            {
                c[CpuRegister.Rax] = 0;
                return 0;
            });
            entries.SetValue(Activator.CreateInstance(entryType, [0x200000UL, nids[index], export, false, false, false, false, 0UL]), index);
        }

        type.GetField("_importEntries", Instance)!.SetValue(backend, entries);
        Assert.True(DirectExecutionBackend.IsTrivialLeafImport(nids[0]));
        var createTrampoline = type.GetMethod("CreateImportHandlerTrampoline", Instance)!;
        var trivialTrampoline = (nint)createTrampoline.Invoke(backend, [0])!;
        var fullTrampoline = (nint)createTrampoline.Invoke(backend, [1])!;
        Assert.NotEqual(0, trivialTrampoline);
        Assert.NotEqual(0, fullTrampoline);

        var hostRspSlot = (ulong*)NativeMemory.AllocZeroed(sizeof(ulong));
        var hostRspTlsIndex = (uint)type.GetField("_hostRspSlotTlsIndex", Instance)!.GetValue(backend)!;
        Assert.True(DirectExecutionPlatform.Current.SetTlsValue(hostRspTlsIndex, (nint)hostRspSlot));

        // A native stub with no managed crossing at all, for the floor of the same loop.
        var nativeOnly = (byte*)HostMemory.Alloc(null, 4096, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
        nativeOnly[0] = 0x31; nativeOnly[1] = 0xC0; nativeOnly[2] = 0xC3;              // xor eax, eax / ret

        // The caller: rdi = count, rsi = target, rdx = host RSP slot. It points the slot 64 KiB
        // below its own frame and calls the target count times, as guest code calls an import.
        var loop = (byte*)HostMemory.Alloc(null, 4096, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
        byte[] code =
        [
            0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x55,   // push rbx, r12-r15, rbp
            0x48, 0x83, 0xEC, 0x08,                                       // sub rsp, 8 (16-align the calls)
            0x48, 0x89, 0xFB,                                             // mov rbx, rdi
            0x49, 0x89, 0xF4,                                             // mov r12, rsi
            0x48, 0x8D, 0x84, 0x24, 0x00, 0x00, 0xFF, 0xFF,               // lea rax, [rsp - 0x10000]
            0x48, 0x83, 0xE0, 0xF0,                                       // and rax, -16
            0x48, 0x89, 0x02,                                             // mov [rdx], rax
            // loop:
            0x31, 0xFF,                                                   // xor edi, edi
            0x31, 0xF6,                                                   // xor esi, esi
            0x41, 0xFF, 0xD4,                                             // call r12
            0x48, 0xFF, 0xCB,                                             // dec rbx
            0x75, 0xF4,                                                   // jnz loop
            0x48, 0x83, 0xC4, 0x08,                                       // add rsp, 8
            0x5D, 0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B,   // pop rbp, r15-r12, rbx
            0xC3,
        ];
        code.CopyTo(new Span<byte>(loop, code.Length));
        Assert.True(HostMemory.Protect(loop, 4096, HostMemory.PAGE_EXECUTE_READ, out _));
        Assert.True(HostMemory.Protect(nativeOnly, 4096, HostMemory.PAGE_EXECUTE_READ, out _));
        HostMemory.FlushInstructionCache(loop, 4096);
        HostMemory.FlushInstructionCache(nativeOnly, 4096);
        var run = (delegate* unmanaged<ulong, nint, ulong*, void>)loop;

        double NanosPerCall(nint target, ulong count)
        {
            var start = Stopwatch.GetTimestamp();
            run(count, target, hostRspSlot);
            return Stopwatch.GetElapsedTime(start).TotalNanoseconds / count;
        }

        // The managed half on its own: DispatchImportCore as the gateway calls it, with the per-call
        // diagnostics block on (a backend no session configured) and off (_leafDiagnosticsOff).
        var dispatch = type.GetMethod("DispatchImportCore", Instance)!.CreateDelegate<Func<int, nint, ulong>>(backend);
        var pack = Marshal.AllocHGlobal(512);
        new Span<byte>((void*)pack, 512).Clear();
        *(ulong*)(pack + 192 + 96) = 0x800010000UL;
        var diagnosticsOff = type.GetField("_leafDiagnosticsOff", Instance)!;
        double ManagedNanosPerCall(int index, int count)
        {
            var start = Stopwatch.GetTimestamp();
            for (var i = 0; i < count; i++)
            {
                dispatch(index, pack + 192);
            }

            return Stopwatch.GetElapsedTime(start).TotalNanoseconds / count;
        }

        try
        {
            // Warm up every path until tiered compilation has settled.
            for (var warm = 0; warm < 5; warm++)
            {
                NanosPerCall((nint)nativeOnly, 200_000);
                NanosPerCall(trivialTrampoline, 400_000);
                NanosPerCall(fullTrampoline, 400_000);
                ManagedNanosPerCall(0, 400_000);
                ManagedNanosPerCall(1, 400_000);
            }

            const ulong Calls = 400_000;
            foreach (var round in Enumerable.Range(0, 6))
            {
                lines.Add(
                    $"round {round}: native_stub_ns={NanosPerCall((nint)nativeOnly, Calls):F1} " +
                    $"trampoline_trivial_leaf_ns={NanosPerCall(trivialTrampoline, Calls):F1} " +
                    $"trampoline_full_path_ns={NanosPerCall(fullTrampoline, Calls):F1}");
            }

            foreach (var off in new[] { false, true, false, true })
            {
                diagnosticsOff.SetValue(backend, off);
                lines.Add(
                    $"managed DispatchImportCore leaf_diagnostics_off={off}: trivial_leaf_ns={ManagedNanosPerCall(0, 400_000):F1} " +
                    $"full_path_ns={ManagedNanosPerCall(1, 400_000):F1}");
            }
        }
        finally
        {
            Marshal.FreeHGlobal(pack);
            HostMemory.Free(loop, 0, HostMemory.MEM_RELEASE);
            HostMemory.Free(nativeOnly, 0, HostMemory.MEM_RELEASE);
            DirectExecutionPlatform.Current.SetTlsValue(hostRspTlsIndex, 0);
            NativeMemory.Free(hostRspSlot);
        }
    }
}
