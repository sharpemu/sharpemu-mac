// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

public sealed class ResourceRetryReadLogTests
{
    [Fact]
    public void PacketReadsPopulateRetryLogAndLaterChangesOrUnavailableBytesVetoRetry()
    {
        using var memory = new Memory();
        var plan = Extract(Program(ScalarLoad(0, 0, 16), Vop1(8, "VMovB32", 0, Gen5Operand.Scalar(16)),
            EndProgram(12)), userDataCount: 2);
        var log = new GuestReadLog();
        var inputs = new ResourceRuntimeInputs { UserData = [0x1000, 0], ReadMemory = memory.Read,
            ReadCleanMemory = memory.Read, DependencySource = memory };
        var cache = new ResourceMaterializationCache(dependencyBypass: true, verifyDependencyBypass: true);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(cache.Materialize(plan, inputs, memory.Resident, ref snapshot, ref specialization, out _));
        inputs.ObserveCachedRead = (address, bytes) => log.Bytes(address, true, bytes);
        Assert.True(cache.Materialize(plan, inputs, memory.Resident, ref snapshot, ref specialization, out _));
        Assert.Equal(1, cache.BypassedCalls);
        Assert.NotEqual(0, log.Count);
        Assert.True(log.Complete);
        Assert.True(log.Holds(memory.Resident));
        memory.Available = false;
        Assert.False(log.Holds(memory.Resident));
        memory.Available = true;
        memory.Write(22);
        Assert.False(log.Holds(memory.Resident));
        log.Clear();
        Assert.True(cache.Materialize(plan, inputs, memory.Resident, ref snapshot, ref specialization, out _));
        Assert.Equal(2, cache.FullPathCalls);
        Assert.Equal(22u, snapshot.FlattenedResourceTable[0]);
    }

    private sealed class Memory : IResourceDependencySource, IDisposable
    {
        private readonly GuestWriteGenerations _versions = new();
        private uint _word = 11;
        internal bool Available = true;
        public ResourceDependencyEpoch DependencyEpoch => default;
        public bool Read(ulong address, out uint word) { word = _word; return address == 0x1000; }
        public bool Resident(ulong address, Span<byte> bytes, bool clean)
        {
            if (!Available || address != 0x1000 || bytes.Length != 4) return false;
            BitConverter.TryWriteBytes(bytes, _word);
            return true;
        }
        public void Write(uint word) { _versions.NoteWrite(0x1000, 4); _word = word; }
        public bool TryCaptureDependency(ulong address, ReadOnlySpan<byte> bytes, bool clean, out ResourceDependency? dependency)
        {
            var token = _versions.Register(address, (ulong)bytes.Length);
            dependency = new Token(token);
            return address == 0x1000 && bytes.Length == 4 && BitConverter.ToUInt32(bytes) == _word && token.IsCurrent;
        }
        public void Dispose() => _versions.Dispose();
        private sealed class Token(GuestWriteGenerations.Token token) : ResourceDependency
        {
            public override ResourceDependencyStatus Validate() => token.IsCurrent ? ResourceDependencyStatus.Current : ResourceDependencyStatus.CpuWrite;
            public override void Dispose() { }
        }
    }
}
