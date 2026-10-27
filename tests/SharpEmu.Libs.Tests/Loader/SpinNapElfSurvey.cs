// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpEmu.Core.Loader;
using SharpEmu.Core.Memory;
using SharpEmu.HLE;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Libs.Tests.Loader;

// TEMPORARY survey (not for commit): SHARPEMU_SPIN_SURVEY_ELF=<flat elf> runs the spin pass over it.
[Collection(GuestRedZonePatcherCollection.Name)]
public sealed unsafe class SpinNapElfSurvey(ITestOutputHelper output)
{
    [Fact]
    public void Survey()
    {
        var paths = Environment.GetEnvironmentVariable("SHARPEMU_SPIN_SURVEY_ELF");
        if (string.IsNullOrEmpty(paths) || RuntimeInformation.ProcessArchitecture != Architecture.X64)
            return;
        foreach (var path in paths.Split(':'))
        {
            var file = File.ReadAllBytes(path);
            var phoff = BinaryPrimitives.ReadUInt64LittleEndian(file.AsSpan(0x20));
            var phnum = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(0x38));
            var headers = new List<ProgramHeader>();
            ulong span = 0;
            for (var i = 0; i < phnum; i++)
            {
                var h = MemoryMarshal.Read<ProgramHeader>(file.AsSpan((int)phoff + i * 56, 56));
                headers.Add(h);
                if (h.HeaderType == ProgramHeaderType.Load)
                    span = Math.Max(span, h.VirtualAddress + h.MemorySize);
            }

            span = (span + 0xFFFFF) & ~0xFFFFFUL;
            using var memory = new PhysicalVirtualMemory();
            var holeSize = span + 0x8000000;
            var hole = (ulong)HostMemory.Alloc(null, (nuint)holeSize, HostMemory.MEM_RESERVE, HostMemory.PAGE_NOACCESS);
            Assert.NotEqual(0UL, hole);
            Assert.True(HostMemory.Free((void*)hole, 0, HostMemory.MEM_RELEASE));
            var imageBase = memory.AllocateAt(hole, span);
            foreach (var h in headers)
            {
                if (h.HeaderType != ProgramHeaderType.Load || h.FileSize == 0)
                    continue;
                Assert.True(memory.TryWrite(imageBase + h.VirtualAddress, file.AsSpan((int)h.Offset, (int)h.FileSize)));
            }

            var probe = SpinNapRuntime.EnsureProbeBytes(1_000_000_000);
            GuestRedZonePatcher.SurveyLog = new List<string>();
            var result = GuestRedZonePatcher.PatchSpinLoops(memory, memory, headers, imageBase, span, probe, Path.GetFileName(path));
            output.WriteLine($"{path}: {result}");
            Console.Error.WriteLine($"SURVEY {path}: {result}");
            File.WriteAllLines("/tmp/r2k1-survey-" + Path.GetFileName(path) + ".txt", GuestRedZonePatcher.SurveyLog);
            GuestRedZonePatcher.SurveyLog = null;
        }
    }
}
