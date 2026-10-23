// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// ShaderProgramCache.BackgroundReplan is process-wide: these tests do not run beside each other.
[Collection(nameof(BindingReplanTests))]
public sealed class BindingReplanTests
{
    private const ulong CodeAddress = PipelineTestGuest.MemoryBase + 0x1_0000;
    private const ulong HeaderAddress = PipelineTestGuest.MemoryBase + 0x8000;
    private const ulong BufferAddress = PipelineTestGuest.MemoryBase + 0x4_0000;
    private const ulong TableAddress = PipelineTestGuest.MemoryBase + 0x2_0000;

    // s_load_dwordx4 s[0:3], s[0:1], 0; buffer_load_format_x v0, off, s[0:3], 0; s_nop N;
    // s_endpgm: the V# is read from the table s[0:1] points at. The s_nop keeps each test's hash
    // apart: the learned binding loads are process-wide.
    private static uint[] TableBoundLoadProgram(uint nop) => [0xF4080000, 0xFA000000, 0xE0000000, 0x80000000, 0xBF800000 | nop, 0xBF810000];

    private sealed class Bind
    {
        public required PipelineTestGuest Guest { get; init; }
        public required ShaderProgramCache Programs { get; init; }
        public required ShaderSource Source { get; init; }
        public bool Pending { get; set; }
        public List<ulong> PendingReads { get; } = [];

        public ShaderProgram Run()
        {
            var cursor = 0u;
            return Programs.GetOrCompile(Source, PipelineTestGuest.ComputeOptions(threadsX: 64), ref cursor, out _);
        }
    }

    private static Bind Create(uint nop)
    {
        var guest = new PipelineTestGuest();
        guest.Host.BackgroundShaderCompilationSupported = true;
        Bind? bind = null;
        guest.Host.ResourceWordPending = address => bind!.Pending && address >= TableAddress && address < TableAddress + 16;
        var programs = new ShaderProgramCache(guest.Context, guest.Compiler, guest.Host);
        guest.RegisterProgram(CodeAddress, HeaderAddress, TableBoundLoadProgram(nop));
        guest.WriteWords(TableAddress, PipelineTestGuest.BufferDescriptor(BufferAddress, 4, 64, BufferDescriptorWords.Format32UInt));
        var source = guest.Source(CodeAddress, ShaderStage.Compute, [unchecked((uint)TableAddress), (uint)(TableAddress >> 32)]);
        bind = new Bind { Guest = guest, Programs = programs, Source = source };
        return bind;
    }

    // Demon's Souls' logo movies: each frame a V# the GPU was still writing re-planned one
    // compute shader, the dispatch switched to the new program at once and waited 1.8-2.1 s for
    // its pipeline. The bind now keeps the program it has (reading the word on the host) while
    // the new one compiles in the background, and the key switches once it is ready.
    [Fact]
    public void APendingWordReplan_KeepsTheCurrentProgram_UntilTheReplannedOneIsReady()
    {
        var previous = ShaderProgramCache.BackgroundReplan;
        ShaderProgramCache.BackgroundReplan = true;
        var bind = Create(nop: 2);
        try
        {
            var first = bind.Run();
            Assert.Single(bind.Guest.Compiler.Requests);
            Assert.Single(bind.Guest.Compiler.Requests[0].Resources.Info.Buffers);

            bind.Pending = true;
            var kept = bind.Run();

            // The bind got the program it had, and the re-plan is learned and translating elsewhere.
            Assert.Equal(first, kept);
            Assert.True(DeviceBindingLoads.CountFor(Assert.Single(bind.Programs.Entries).Plan.DeviceBindingKey) > 0);

            var deadline = Stopwatch.StartNew();
            var switched = kept;
            while (switched == first && deadline.Elapsed < TimeSpan.FromSeconds(10))
            {
                Thread.Sleep(5);
                switched = bind.Run();
            }

            Assert.NotEqual(first, switched);
            Assert.Equal(2, bind.Guest.Compiler.Requests.Count);
            // The program switched to is the re-planned one: the V# is read on the device.
            Assert.Empty(bind.Guest.Compiler.Requests[1].Resources.Info.Buffers);
            // Its module was made by the background translation, not again by the bind.
            Assert.Equal(2, bind.Guest.Host.Modules.Count);
            Assert.Equal(switched.Id, bind.Guest.Host.Modules[1].ProgramId);
            Assert.Equal(switched, bind.Run());
        }
        finally
        {
            ShaderProgramCache.BackgroundReplan = previous;
            foreach (var entry in bind.Programs.Entries) DeviceBindingLoads.Forget(entry.Plan.DeviceBindingKey);
        }
    }

    // SHARPEMU_METAL_V2_REPLAN_BACKGROUND=0: the bind switches to the re-planned program at once.
    [Fact]
    public void WithoutBackgroundReplans_TheBindTakesTheReplannedProgramAtOnce()
    {
        var previous = ShaderProgramCache.BackgroundReplan;
        ShaderProgramCache.BackgroundReplan = false;
        var bind = Create(nop: 3);
        try
        {
            var first = bind.Run();
            bind.Pending = true;
            var replanned = bind.Run();

            Assert.NotEqual(first, replanned);
            Assert.Equal(2, bind.Guest.Compiler.Requests.Count);
            Assert.Empty(bind.Guest.Compiler.Requests[1].Resources.Info.Buffers);
        }
        finally
        {
            ShaderProgramCache.BackgroundReplan = previous;
            foreach (var entry in bind.Programs.Entries) DeviceBindingLoads.Forget(entry.Plan.DeviceBindingKey);
        }
    }

    // A program the key never compiled cannot serve while another compiles: the first bind
    // takes the re-planned program as before (one compile, not two).
    [Fact]
    public void AFirstBindWithAPendingWord_CompilesTheReplannedProgramOnly()
    {
        var previous = ShaderProgramCache.BackgroundReplan;
        ShaderProgramCache.BackgroundReplan = true;
        var bind = Create(nop: 4);
        try
        {
            bind.Pending = true;
            bind.Run();

            Assert.Single(bind.Guest.Compiler.Requests);
            Assert.Empty(bind.Guest.Compiler.Requests[0].Resources.Info.Buffers);
        }
        finally
        {
            ShaderProgramCache.BackgroundReplan = previous;
            foreach (var entry in bind.Programs.Entries) DeviceBindingLoads.Forget(entry.Plan.DeviceBindingKey);
        }
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SuccessorWaitsForBothItsLibraryAndComputePipeline(bool libraryFirst)
    {
        var bind = Create(libraryFirst ? 11u : 12u);
        try
        {
            var first = bind.Run();
            bind.Guest.Host.SuccessorLibraryReady = false;
            bind.Guest.Host.SuccessorPipelineReady = false;
            bind.Pending = true;
            Assert.Equal(first, bind.Run());
            var entry = Assert.Single(bind.Programs.Entries);
            Assert.NotNull(entry.SuccessorWork);
            Assert.True(SpinWait.SpinUntil(() => entry.SuccessorWork.IsCompleted, TimeSpan.FromSeconds(10)));
            Assert.Equal(first, bind.Run());
            bind.Guest.Host.SuccessorLibraryReady = libraryFirst;
            bind.Guest.Host.SuccessorPipelineReady = !libraryFirst;
            Assert.Equal(first, bind.Run());
            // The old module still names its original buffer layout and materialization.
            Assert.Single(bind.Guest.Compiler.Requests[0].Resources.Info.Buffers);
            bind.Guest.Host.SuccessorLibraryReady = true;
            bind.Guest.Host.SuccessorPipelineReady = true;
            var switched = bind.Run();
            Assert.NotEqual(first, switched);
            Assert.Empty(bind.Guest.Compiler.Requests[1].Resources.Info.Buffers);
            Assert.Equal(switched, bind.Run());
        }
        finally { foreach (var entry in bind.Programs.Entries) DeviceBindingLoads.Forget(entry.Plan.DeviceBindingKey); }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void FailedOrCancelledSuccessorKeepsTheOriginalProgram(int failure)
    {
        var bind = Create(13u + (uint)failure);
        try
        {
            var first = bind.Run();
            if (failure == 1) bind.Guest.Host.CancelBackgroundTranslation = true;
            else if (failure == 2) bind.Guest.Compiler.FatalRejection = true;
            else bind.Guest.Compiler.Rejection = "forced successor rejection";
            bind.Pending = true;
            Assert.Equal(first, bind.Run());
            var entry = Assert.Single(bind.Programs.Entries);
            Assert.True(SpinWait.SpinUntil(() => entry.SuccessorWork!.IsCompleted, TimeSpan.FromSeconds(10)));
            Assert.Equal(first, bind.Run());
            Assert.Null(entry.Successor);
            Assert.Single(bind.Guest.Host.Modules);
            Assert.Equal(first, bind.Run());
        }
        finally { foreach (var entry in bind.Programs.Entries) DeviceBindingLoads.Forget(entry.Plan.DeviceBindingKey); }
    }

}
