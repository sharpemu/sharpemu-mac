// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

// Device-free, DS-shaped call counts, not a recorded game workload. Twelve immutable
// plans, eight static V#s, a changing root CB and four table scalars per stage. Every
// frame has 15,178 stage calls; scalar offsets cycle through eight ring positions.
var frames = args.Length == 0 ? 30 : int.Parse(args[0]);
var verify = args.Contains("--verify");
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var modes = new[] { "full", "legacy", "packets" };
Console.WriteLine($"arch={RuntimeInformation.ProcessArchitecture} frames={frames} calls_per_frame=15178 verify={verify} synthetic=1");
foreach (var mode in modes)
{
    var memory = new Memory();
    var plan = MakePlan();
    var plans = Enumerable.Range(0, 12).Select(_ => MakePlan()).ToArray();
    var inputs = new ResourceRuntimeInputs
    {
        ReadMemory = memory.Read, ReadCleanMemory = memory.Read, ReadResidentMemory = memory.Resident,
        DependencySource = memory,
    };
    ResidentGuestBytesReader resident = memory.Resident;
    var cache = new ResourceMaterializationCache(dependencyBypass: mode == "packets", verifyDependencyBypass: verify);
    var roots = Enumerable.Range(0, 8).Select(i => new uint[] { 0x1000, 0, 0, 0, 0x10000u + (uint)i * 256, 0, 256, 0x20000 }).ToArray();
    var snapshot = new ResourceSnapshot();
    var specialization = new ResourceSpecialization();
    long initialBypassed = 0, initialPatched = 0, initialFull = 0;
    for (var frame = -3; frame < frames; frame++)
    {
        if (frame == 0)
        {
            memory.Reads = 0;
            memory.Allocations = GC.GetAllocatedBytesForCurrentThread();
            memory.Started = Stopwatch.GetTimestamp();
            initialBypassed = cache.BypassedCalls;
            initialPatched = cache.PatchedCalls;
            initialFull = cache.FullPathCalls;
        }
        // One exact interval changes each frame. All other stages can keep their
        // already-certified dependencies until their own intervals change.
        memory.Write(0x1400, (uint)(frame + 10));
        for (var call = 0; call < 15178; call++)
        {
            inputs.UserData = roots[(call / 12) & 7];
            plan = plans[call % plans.Length];
            var ok = mode == "full"
                ? ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization)
                : cache.Materialize(plan, inputs, resident, ref snapshot, ref specialization, out _);
            if (!ok) throw new InvalidOperationException("Materialization failed");
        }
    }
    var elapsed = Stopwatch.GetElapsedTime(memory.Started).TotalMicroseconds;
    var allocated = GC.GetAllocatedBytesForCurrentThread() - memory.Allocations;
    var calls = frames * 15178L;
    Console.WriteLine($"mode={mode} us/call={elapsed / calls:F3} ms/frame={elapsed / frames / 1000:F3} " +
        $"bytes/call={allocated / (double)calls:F2} reads/call={memory.Reads / (double)calls:F2} " +
        $"bypassed={cache.BypassedCalls - initialBypassed} patched={cache.PatchedCalls - initialPatched} full_path={cache.FullPathCalls - initialFull}");
}

static ShaderResourcePlan MakePlan()
{
    var instructions = new List<Gen5ShaderInstruction>();
    uint pc = 0;
    for (uint descriptor = 0; descriptor < 8; descriptor++)
    {
        instructions.Add(ScalarLoad(pc, 0, 16 + descriptor * 4, 4, immediateOffset: (int)descriptor * 32)); pc += 8;
        instructions.Add(BufferLoad(pc, 16 + descriptor * 4)); pc += 8;
    }
    instructions.Add(BufferLoad(pc, 4)); pc += 8;
    instructions.Add(ScalarLoad(pc, 0, 48, 4, immediateOffset: 0x400)); pc += 8;
    for (uint word = 0; word < 4; word++)
    {
        instructions.Add(Vop1(pc, "VMovB32", word, Gen5Operand.Scalar(48 + word))); pc += 4;
    }
    instructions.Add(EndProgram(pc));
    return Extract(Program(instructions.ToArray()), userDataCount: 8);
}

sealed class Memory : IResourceDependencySource
{
    private readonly uint[] _words = new uint[0x200];
    private readonly long[] _versions = new long[0x200];
    public long Reads, Allocations, Started;
    public ResourceDependencyEpoch DependencyEpoch => default;
    public Memory()
    {
        for (var descriptor = 0; descriptor < 8; descriptor++)
        {
            var offset = descriptor * 8;
            _words[offset] = 0x20000u + (uint)descriptor * 4096;
            _words[offset + 2] = 256;
            _words[offset + 3] = 0x20000;
        }
    }
    public bool Read(ulong address, out uint value)
    {
        Reads++;
        var index = (address - 0x1000) / 4;
        value = index < (ulong)_words.Length ? _words[index] : 0;
        return index < (ulong)_words.Length;
    }
    public bool Resident(ulong address, Span<byte> bytes, bool clean)
    {
        for (var offset = 0; offset < bytes.Length; offset += 4)
        {
            if (!Read(address + (ulong)offset, out var word)) return false;
            BitConverter.TryWriteBytes(bytes[offset..], word);
        }
        return true;
    }
    public void Write(ulong address, uint word)
    {
        var index = (address - 0x1000) / 4;
        _versions[index]++;
        _words[index] = word;
    }
    public bool TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean, out ResourceDependency? dependency)
    {
        dependency = null;
        var start = checked((int)((address - 0x1000) / 4));
        var count = expected.Length / 4;
        if (start < 0 || start > _words.Length - count || !MemoryMarshal.AsBytes(_words.AsSpan(start, count)).SequenceEqual(expected)) return false;
        dependency = new Token(this, start, _versions.AsSpan(start, count).ToArray());
        return true;
    }
    private sealed class Token(Memory memory, int start, long[] versions) : ResourceDependency
    {
        public override ResourceDependencyStatus Validate() => memory._versions.AsSpan(start, versions.Length).SequenceEqual(versions)
            ? ResourceDependencyStatus.Current : ResourceDependencyStatus.CpuWrite;
        public override void Dispose() { }
    }
}
