// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ResourceDependencyBypassTests
{
    // Models interval versions plus a writer that can pause between admission and
    // landing. Reads during a full evaluation observe the current ordered memory.
    internal sealed class Memory : IResourceDependencySource
    {
        internal readonly Dictionary<ulong, uint> Words = new();
        private readonly List<Token> _tokens = [];
        internal int Reads, ResidentReads, Captures;
        internal ResourceDependencyEpoch Epoch;
        internal bool RefuseCapture, CorruptCapture, Pending, PublicationsPending;
        internal bool ByteValidation;
        internal ResourceCaptureFailure Failure = ResourceCaptureFailure.PendingWriter;
        internal long Policy, Retry;
        internal int ByteValidations;
        public bool DependencyReadsReady => !PublicationsPending;
        public ResourceDependencyEpoch DependencyEpoch => Epoch;
        public ResourceDependencyFailure CaptureFailure => new(ResourceRangeClass.ChunkedBacking, Failure);
        public long CapturePolicyVersion => Policy;
        public long CaptureRetryVersion => Retry;
        public bool UsesDependencyByteValidation => ByteValidation;
        public bool TryValidateDependencyBytes(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths,
            ReadOnlySpan<int> offsets, ReadOnlySpan<byte> expected)
        {
            ByteValidations++;
            if (Pending) return false;
            for (var range = 0; range < addresses.Length; range++)
                for (var offset = 0; offset < lengths[range]; offset += 4)
                    if (!Words.TryGetValue(addresses[range] + (ulong)offset, out var word) ||
                        word != BitConverter.ToUInt32(expected[(offsets[range] + offset)..])) return false;
            return true;
        }

        internal bool Read(ulong address, out uint word) { Reads++; return Words.TryGetValue(address, out word); }
        internal bool Resident(ulong address, Span<byte> bytes, bool clean)
        {
            ResidentReads++;
            for (var offset = 0; offset < bytes.Length; offset += 4)
            {
                if (!Read(address + (ulong)offset, out var word)) return false;
                BitConverter.TryWriteBytes(bytes[offset..], word);
            }
            return true;
        }
        public bool TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean, out ResourceDependency? dependency)
        {
            dependency = null;
            Captures++;
            if (RefuseCapture || Pending) return false;
            for (var offset = 0; offset < expected.Length; offset += 4)
                if (!Words.TryGetValue(address + (ulong)offset, out var word) || BitConverter.ToUInt32(expected[offset..]) != word) return false;
            var token = new Token(this, address, (ulong)expected.Length);
            _tokens.Add(token);
            dependency = token;
            return true;
        }
        internal void Write(ulong address, uint word)
        {
            foreach (var token in _tokens)
                if (address >= token.Address && address - token.Address < token.Size) token.Status = ResourceDependencyStatus.CpuWrite;
            Words[address] = word;
        }
        private sealed class Token(Memory owner, ulong address, ulong size) : ResourceDependency
        {
            internal readonly ulong Address = address, Size = size;
            internal ResourceDependencyStatus Status;
            public override ResourceDependencyStatus Validate() => owner.Pending ? ResourceDependencyStatus.PendingWriter :
                owner.CorruptCapture ? ResourceDependencyStatus.Current : Status;
            public override void Dispose() => Status = ResourceDependencyStatus.Unavailable;
        }
    }

    internal static ShaderResourcePlan TablePlan() => Extract(Program(
        ScalarLoad(0, 0, 16, 4), ScalarLoad(8, 0, 20, immediateOffset: 0x40),
        Vop1(16, "VMovB32", 0, Gen5Operand.Scalar(20)),
        BufferLoad(20, 4), BufferLoad(28, 16), EndProgram(36)), userDataCount: 8);

    internal static Memory TableMemory()
    {
        var memory = new Memory();
        foreach (var pointer in new ulong[] { 0x1000, 0x2000 })
        {
            memory.Words[pointer] = 0x8000;
            memory.Words[pointer + 4] = 0;
            memory.Words[pointer + 8] = 256;
            memory.Words[pointer + 12] = 0x2_0000;
            memory.Words[pointer + 0x40] = 11;
        }
        return memory;
    }
    internal static uint[] Roots(uint cb = 0x10000, uint table = 0x1000) => [table, 0, 0, 0, cb, 0, 256, 0x2_0000];

    internal sealed class Stream(ShaderResourcePlan plan, Memory memory, bool bypass = true, bool verify = true)
    {
        internal readonly ResourceMaterializationCache Cache = new(dependencyBypass: bypass, verifyDependencyBypass: verify);
        internal ResourceSnapshot Snapshot = new();
        internal ResourceSpecialization Specialization = new();
        private readonly ResidentGuestBytesReader _resident = memory.Resident;
        internal readonly ResourceRuntimeInputs Inputs = new()
        {
            ReadMemory = memory.Read, ReadCleanMemory = memory.Read, ReadResidentMemory = memory.Resident,
            DependencySource = memory,
        };
        internal void Run(uint[] roots)
        {
            Inputs.UserData = roots;
            if (!Cache.Materialize(plan, Inputs, _resident, ref Snapshot, ref Specialization, out var failure))
                Assert.Fail(failure.ToString());
        }
        internal void EqualFull()
        {
            var full = new ResourceSnapshot();
            var specialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(plan, Inputs, ref full, ref specialization));
            Assert.Equal(specialization, Specialization);
            Assert.Equal(full.UserData, Snapshot.UserData);
            Assert.Equal(full.FlattenedResourceTable, Snapshot.FlattenedResourceTable);
            Assert.Equal(full.DeviceAddressRanges, Snapshot.DeviceAddressRanges);
            Assert.Equal(full.Buffers.SelectMany(x => x), Snapshot.Buffers.SelectMany(x => x));
            Assert.Equal(full.Images.SelectMany(x => x), Snapshot.Images.SelectMany(x => x));
            Assert.Equal(full.Samplers.SelectMany(x => x), Snapshot.Samplers.SelectMany(x => x));
        }
    }

    [Fact]
    public void UnchangedAndScalarOnlyCallsNeverEnterOldValidatorOrEvaluator()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory, verify: false);
        var roots = Roots();
        stream.Run(roots);
        var previous = stream.Snapshot;
        var reads = memory.Reads;
        var resident = memory.ResidentReads;
        stream.Run(roots);
        Assert.Equal(1, stream.Cache.BypassedCalls);
        stream.Run(Roots(0x11000));
        Assert.Equal(1, stream.Cache.PatchedCalls);
        Assert.Equal(reads, memory.Reads);
        Assert.Equal(resident, memory.ResidentReads);
        Assert.Equal(0x10000u, previous.Buffers[0][0]); // Published output was not patched in place.
        stream.EqualFull();
        Assert.Equal(1, stream.Cache.FullPathCalls);
    }

    [Fact]
    public void PendingHostPublicationsVetoCertificatesWithoutChangingTheGpuEpoch()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory, verify: false);
        var roots = Roots();
        stream.Run(roots);
        stream.Run(roots);
        Assert.Equal(1, stream.Cache.BypassedCalls);
        var epoch = memory.Epoch;
        var reads = memory.Reads;
        memory.PublicationsPending = true;
        stream.Run(roots);
        stream.Run(roots);
        Assert.Equal(epoch, memory.Epoch);
        Assert.Equal(1, stream.Cache.BypassedCalls);
        Assert.Equal(3, stream.Cache.FullPathCalls);
        Assert.True(memory.Reads > reads);
        memory.Write(0x1040, 77);
        memory.PublicationsPending = false;
        stream.Run(roots);
        stream.Run(roots);
        Assert.Equal(2, stream.Cache.BypassedCalls);
        stream.EqualFull();
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("gpu")]
    [InlineData("pending")]
    [InlineData("remap")]
    [InlineData("image")]
    [InlineData("overlay")]
    [InlineData("label")]
    [InlineData("pointer")]
    public void ChangedDependenciesTakeFullPathAndMatchEveryByte(string change)
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory);
        stream.Run(Roots());
        stream.Run(Roots());
        Assert.Equal(1, stream.Cache.BypassedCalls);
        switch (change)
        {
            case "cpu": memory.Write(0x1040, 22); break;
            case "gpu": memory.Epoch = memory.Epoch with { GpuWrites = 1 }; memory.Words[0x1040] = 33; break;
            case "pending": memory.Pending = true; memory.Words[0x1040] = 44; break;
            case "remap": memory.Epoch = memory.Epoch with { Mapping = 1 }; memory.Words[0x1040] = 55; break;
            case "image": memory.Epoch = memory.Epoch with { Images = 1 }; break;
            case "overlay": memory.Epoch = memory.Epoch with { Overlay = 1 }; break;
            case "label": memory.Epoch = memory.Epoch with { Labels = 1 }; break;
        }
        stream.Run(change == "pointer" ? Roots(table: 0x2000) : Roots());
        stream.EqualFull();
        Assert.Equal(2, stream.Cache.FullPathCalls);
        Assert.Equal(1, stream.Cache.BypassedCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OverlappingCpuAndGpuWritersInBothOrdersCannotReuseOldWords(bool cpuFirst)
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory);
        stream.Run(Roots());
        void Cpu() => memory.Write(0x1040, 22);
        void Gpu() { memory.Epoch = memory.Epoch with { GpuWrites = 1 }; memory.Words[0x1040] = 33; }
        if (cpuFirst) { Cpu(); Gpu(); } else { Gpu(); Cpu(); }
        stream.Run(Roots());
        stream.EqualFull();
        Assert.Equal(cpuFirst ? 33u : 22u, stream.Snapshot.FlattenedResourceTable.Last());
        Assert.Equal(0, stream.Cache.BypassedCalls);
    }

    [Fact]
    public void AdmissionBeforeLandedAndDisjointStoresAreDistinguished()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory);
        stream.Run(Roots());
        memory.Write(0x2000, 77);
        stream.Run(Roots());
        Assert.Equal(1, stream.Cache.BypassedCalls);
        memory.Pending = true; // Writer admitted; bytes not landed yet.
        stream.Run(Roots());
        memory.Write(0x1040, 88);
        memory.Pending = false;
        stream.Run(Roots());
        stream.EqualFull();
        Assert.Equal(3, stream.Cache.FullPathCalls);
    }

    [Fact]
    public void MutableRootStorageCannotChangeAnExistingPacket()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory);
        var roots = Roots();
        stream.Run(roots);
        roots[4] = 0x12000;
        stream.Run(roots);
        Assert.Equal(1, stream.Cache.PatchedCalls);
        stream.EqualFull();
    }

    [Fact]
    public void UnreadInlineRootsPatchOnlyUserDataWithoutMemoryReads()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory, verify: false);
        stream.Run(Roots());
        var previous = stream.Snapshot;
        var reads = memory.Reads;
        var roots = Roots();
        roots[2] = 99;
        stream.Run(roots);
        Assert.Equal(1, stream.Cache.PatchedCalls);
        Assert.Same(previous.Buffers, stream.Snapshot.Buffers);
        Assert.Equal(reads, memory.Reads);
        stream.EqualFull();
    }

    [Fact]
    public void ABufferBaseThatAlsoAddressesHostReadsCannotBePatched()
    {
        var plan = Extract(Program(ScalarLoad(0, 4, 12), Vop1(8, "VMovB32", 0, Gen5Operand.Scalar(12)),
            BufferLoad(12, 4), EndProgram(20)), userDataCount: 8);
        var memory = TableMemory();
        memory.Words[0x10000] = 11;
        memory.Words[0x11000] = 22;
        var stream = new Stream(plan, memory);
        stream.Run(Roots());
        stream.Run(Roots(0x11000));
        stream.EqualFull();
        Assert.Equal(0, stream.Cache.PatchedCalls);
    }

    [Fact]
    public void VerificationModeDetectsABrokenVersionProvider()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory);
        stream.Run(Roots());
        memory.CorruptCapture = true;
        memory.Words[0x1040] = 99; // Deliberately omit the notification.
        Assert.Contains("verification mismatch", Assert.Throws<InvalidOperationException>(() => stream.Run(Roots())).Message);
    }

    [Fact]
    public void AnUncertifiedRangeCanEnrollAfterItsWriterFinishes()
    {
        var memory = TableMemory();
        memory.RefuseCapture = true;
        var stream = new Stream(TablePlan(), memory);
        stream.Run(Roots());
        stream.Run(Roots());
        Assert.Equal(0, stream.Cache.BypassedCalls);
        memory.RefuseCapture = false;
        memory.Retry++; // Completion of the original writer, without a new GPU write.
        stream.Run(Roots());
        stream.Run(Roots());
        Assert.Equal(1, stream.Cache.BypassedCalls);
        stream.EqualFull();
    }

    [Fact]
    public void PhysicalRefusalsAreConstantTimePerEntryUntilMappingOrPolicyChanges()
    {
        var memory = TableMemory();
        memory.RefuseCapture = true;
        memory.Failure = ResourceCaptureFailure.PhysicalWatch;
        var stream = new Stream(TablePlan(), memory, verify: false);
        stream.Run(Roots());
        var captures = memory.Captures;
        for (var i = 0; i < 100; i++)
        {
            memory.Epoch = memory.Epoch with { GpuWrites = (ulong)i, Overlay = i };
            memory.Retry++;
            stream.Run(Roots());
        }
        Assert.Equal(captures, memory.Captures);
        memory.Epoch = memory.Epoch with { Mapping = 1 };
        stream.Run(Roots());
        Assert.Equal(captures + 1, memory.Captures);
        memory.Policy++;
        stream.Run(Roots());
        Assert.Equal(captures + 2, memory.Captures);
        // A different table is not negatively cached by the first one's shader key.
        stream.Run(Roots(table: 0x2000));
        Assert.True(memory.Captures > captures + 2);
        stream.EqualFull();
    }

    [Fact]
    public void BytePacketsValidateEveryReadWithoutPhysicalCaptureOrLegacyWalk()
    {
        var memory = TableMemory();
        memory.ByteValidation = true;
        memory.RefuseCapture = true;
        var stream = new Stream(TablePlan(), memory, verify: false);
        stream.Run(Roots());
        var reads = memory.Reads;
        var resident = memory.ResidentReads;
        for (var i = 0; i < 100; i++)
        {
            memory.Epoch = memory.Epoch with { GpuWrites = (ulong)i, Overlay = i, Labels = i };
            stream.Run(Roots());
        }
        Assert.Equal(100, stream.Cache.BypassedCalls);
        Assert.Equal(100, stream.Cache.ByteValidatedCalls);
        Assert.Equal(0, memory.Captures);
        Assert.Equal(reads, memory.Reads);
        Assert.Equal(resident, memory.ResidentReads);
        stream.EqualFull();
        memory.Words[0x1040] = 88; // Native/alias store with no notification.
        stream.Run(Roots());
        Assert.Equal(2, stream.Cache.FullPathCalls);
        stream.EqualFull();
        stream.Run(Roots(0x11000));
        Assert.Equal(1, stream.Cache.PatchedCalls);
        stream.EqualFull();
    }

    [Fact]
    public void BytePacketsKeepMappingImagePendingAndPolicyVetoes()
    {
        var memory = TableMemory();
        memory.ByteValidation = true;
        var stream = new Stream(TablePlan(), memory);
        stream.Run(Roots());
        stream.Run(Roots());
        memory.Pending = true;
        stream.Run(Roots());
        Assert.Equal(2, stream.Cache.FullPathCalls);
        memory.Pending = false;
        stream.Run(Roots());
        memory.Epoch = memory.Epoch with { Images = 1 };
        stream.Run(Roots());
        Assert.Equal(3, stream.Cache.FullPathCalls);
        memory.Epoch = memory.Epoch with { Mapping = 1 };
        stream.Run(Roots());
        Assert.Equal(4, stream.Cache.FullPathCalls);
        memory.Policy++;
        stream.Run(Roots());
        Assert.Equal(5, stream.Cache.FullPathCalls);
        stream.EqualFull();
    }

    [Fact]
    public void BytePacketsEnrollOnlyAfterTheCompleteObservedReadSetIsSafe()
    {
        var memory = TableMemory();
        memory.ByteValidation = true;
        memory.Pending = true;
        var stream = new Stream(TablePlan(), memory, verify: false);
        stream.Run(Roots());
        Assert.True(memory.ByteValidations > 0);
        Assert.Equal(1, stream.Cache.FullPathCalls);
        var validations = memory.ByteValidations;
        stream.Run(Roots());
        Assert.Equal(validations, memory.ByteValidations);
        memory.Pending = false;
        memory.Retry++;
        stream.Run(Roots());
        stream.Run(Roots());
        Assert.Equal(3, stream.Cache.FullPathCalls);
        Assert.Equal(1, stream.Cache.ByteValidatedCalls);
    }

    [Fact]
    public void AnImageDescriptorIdentityChangeReevaluatesItsCompleteResult()
    {
        var plan = Extract(Program(Image(0, "ImageLoad", 0), EndProgram(8)), userDataCount: 8);
        var memory = new Memory();
        var stream = new Stream(plan, memory);
        uint[] roots = [0x2000, 20u << 20, 0, 0xFACu | (9u << 28), 0, 0, 0, 0];
        stream.Run(roots);
        stream.Run(roots);
        Assert.Equal(1, stream.Cache.BypassedCalls);
        memory.Epoch = memory.Epoch with { Images = 1 };
        stream.Run(roots);
        stream.EqualFull();
        Assert.Equal(2, stream.Cache.FullPathCalls);
    }

    [Fact]
    public void AnInvalidBufferDescriptorCannotBeResurrectedByPatchingItsBase()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory);
        var roots = Roots(cb: 0);
        roots[7] = 1u << 30; // Full specialization clears this non-V# descriptor.
        stream.Run(roots);
        var changed = (uint[])roots.Clone();
        changed[4] = 0x11000;
        stream.Run(changed);
        stream.EqualFull();
        Assert.Equal(0, stream.Cache.PatchedCalls);
        Assert.All(stream.Snapshot.Buffers[0], word => Assert.Equal(0u, word));
    }

    [Fact]
    public void KillSwitchKeepsTheLegacyPathAndItsExactResults()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory, bypass: false);
        stream.Run(Roots());
        stream.Run(Roots());
        stream.EqualFull();
        Assert.Equal(2, stream.Cache.FullPathCalls);
        Assert.Equal(0, stream.Cache.BypassedCalls);
    }

    [Fact]
    public void RepeatedPatchStreamAllocatesNothingAfterWarmup()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory, verify: false);
        var roots = Enumerable.Range(0, 256).Select(i => Roots(0x10000u + (uint)i * 256)).ToArray();
        foreach (var input in roots) stream.Run(input);
        for (var i = 0; i < 100; i++) foreach (var input in roots) stream.Run(input);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++) foreach (var input in roots) stream.Run(input);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Equal(1, stream.Cache.FullPathCalls);
    }
    [Fact]
    public void BypassesReplayEveryCertifiedReadForRetryLogsIncludingScalarPatches()
    {
        var memory = TableMemory();
        var stream = new Stream(TablePlan(), memory, verify: false);
        stream.Run(Roots());
        var observed = new Dictionary<ulong, byte[]>();
        stream.Inputs.ObserveCachedRead = (address, bytes) => observed[address] = bytes.ToArray();
        var reads = memory.Reads;
        stream.Run(Roots());
        Assert.Equal(1, stream.Cache.BypassedCalls);
        Assert.Equal(reads, memory.Reads);
        Assert.NotEmpty(observed);
        foreach (var (address, bytes) in observed)
            for (var offset = 0; offset < bytes.Length; offset += 4)
                Assert.Equal(memory.Words[address + (ulong)offset], BitConverter.ToUInt32(bytes, offset));
        Assert.Contains(observed, item => item.Key <= 0x1040 && item.Key + (ulong)item.Value.Length > 0x1040);
        observed.Clear();
        stream.Run(Roots(0x11000));
        Assert.Equal(1, stream.Cache.PatchedCalls);
        Assert.NotEmpty(observed);
        Assert.Equal(reads, memory.Reads);
        stream.EqualFull();
    }
}
