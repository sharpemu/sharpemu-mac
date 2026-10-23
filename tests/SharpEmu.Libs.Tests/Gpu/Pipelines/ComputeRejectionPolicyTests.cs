// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

[Collection(SchedulingStateCollection.Name)]
public sealed class ComputeRejectionPolicyTests
{
    private const string Variable = "SHARPEMU_STRICT_COMPUTE";
    private const ulong CodeAddress = PipelineTestGuest.MemoryBase + 0x1000;
    private const ulong HeaderAddress = PipelineTestGuest.MemoryBase + 0x8000;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnlinkableShaderCallsCompileTheCallerAlone(bool strict)
    {
        using var fatal = new FatalScope();
        var guest = new PipelineTestGuest();
        guest.Compiler.LinkedShaderCallsSupported = false;
        guest.RegisterProgram(CodeAddress, HeaderAddress, [0xBE8E2100, 0xBF810000]);
        var source = guest.Source(CodeAddress, ShaderStage.Compute, []);
        var cursor = 0u;
        Assert.True(guest.Programs.TryGetProgram(source, PipelineTestGuest.ComputeOptions(32),
            strict, ref cursor, out _, out _, out var rejection), rejection);
        var request = Assert.Single(guest.Compiler.Requests);
        Assert.Contains(request.Program.Instructions, static instruction => instruction.Opcode == "SSwappcB64");
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData("0", false)]
    [InlineData("1", false)]
    [InlineData(null, true)]
    [InlineData("0", true)]
    [InlineData("1", true)]
    [InlineData("invalid", false)]
    [InlineData("invalid", true)]
    public void ComputeRejectionUsesTheConfiguredPolicy(string? value, bool rejectPlan)
    {
        var previous = Environment.GetEnvironmentVariable(Variable);
        try
        {
            Environment.SetEnvironmentVariable(Variable, value);
            using var fatal = new FatalScope();
            var guest = new PipelineTestGuest();
            guest.RegisterProgram(CodeAddress, HeaderAddress, rejectPlan
                ? [0x7E100500, 0xF42C0402, 0x10000000, 0xF0000108, 0x00040000, 0xBF810000]
                : PipelineTestGuest.EndProgram);
            if (!rejectPlan) guest.Compiler.Rejection = "The test instruction is not supported.";
            var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
            var registers = Registers();
            ComputeProgram Lookup() => cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1);
            if (value != "0")
                Assert.Throws<SchedulerFatalException>(() => Lookup());
            else
            {
                var result = Lookup();
                Assert.False(result.Available);
                Assert.False(result.Consumed);
                Assert.False(result.Program.IsValid);
                Assert.False(Lookup().Available);
            }
            Assert.Empty(guest.Host.Modules);
            Assert.Empty(guest.Host.ComputePipelines);
            Assert.Equal(0, guest.Compiler.Compilations);
            if (!rejectPlan && value == "0")
            {
                guest.Compiler.Rejection = null;
                Assert.True(Lookup().Available);
                Assert.Single(guest.Host.Modules);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(Variable, previous);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void UnreadableTablesDrainOnceThenRetryOrSkipEvenInStrictMode(bool strict, bool repaired)
    {
        using var fatal = new FatalScope();
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(CodeAddress, HeaderAddress,
            [0xF4080000, 0xFA000000, 0xE0000000, 0x80000000, 0xBF810000]);
        var table = PipelineTestGuest.MemoryBase + 0x40000;
        guest.WriteWords(table, 0x1000u, 4u << 16, 16, 0x21014FAC);
        var pending = true;
        guest.Host.RefuseTableRead = address => address >= table && address < table + 16 && pending;
        guest.Host.SynchronizeTables = () => { if (repaired) pending = false; };
        var source = guest.Source(CodeAddress, ShaderStage.Compute, [(uint)table, (uint)(table >> 32)]);
        var cursor = 0u;
        var result = guest.Programs.TryGetProgram(source, PipelineTestGuest.ComputeOptions(1), strict,
            ref cursor, out var program, out _, out var rejection);
        Assert.Equal(repaired, result);
        Assert.Equal(repaired, program.IsValid);
        Assert.Equal(1, guest.Host.TableSynchronizations);
        Assert.Equal(Environment.CurrentManagedThreadId, guest.Host.TableSynchronizationThread);
        Assert.Equal(repaired ? 0 : 1, guest.Programs.UnreadablePointerSkips);
        if (!repaired)
        {
            Assert.Contains("UnreadableGuestPointer", rejection);
            Assert.Empty(guest.Compiler.Requests);
            // No permanent shader rejection: a later bind can recover.
            pending = false;
            Assert.True(guest.Programs.TryGetProgram(source, PipelineTestGuest.ComputeOptions(1), strict,
                ref cursor, out _, out _, out _));
            Assert.Equal(1, guest.Host.TableSynchronizations);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnUnmappedNonNullPointerIsSkippedAfterOneDrain(bool strict)
    {
        using var fatal = new FatalScope();
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(CodeAddress, HeaderAddress,
            [0xF4080000, 0xFA000000, 0xE0000000, 0x80000000, 0xBF810000]);
        var source = guest.Source(CodeAddress, ShaderStage.Compute, [0x3F000000u, 0u]);
        var cursor = 0u;
        Assert.False(guest.Programs.TryGetProgram(source, PipelineTestGuest.ComputeOptions(1), strict,
            ref cursor, out _, out _, out var rejection));
        Assert.Contains("UnreadableGuestPointer", rejection);
        Assert.Equal(1, guest.Host.TableSynchronizations);
        Assert.Equal(0u, cursor);
        Assert.Empty(guest.Compiler.Requests);
    }

    [Fact]
    public void UnreadableComputeTableUsesTheUnavailableProgramPath()
    {
        using var fatal = new FatalScope();
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(CodeAddress, HeaderAddress,
            [0xF4080000, 0xFA000000, 0xE0000000, 0x80000000, 0xBF810000]);
        var registers = Registers();
        registers.UserScalars.Set(0, 0x3F000000, UserScalarKind.Unknown);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var result = cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1);
        Assert.False(result.Available);
        Assert.False(result.Consumed);
        Assert.False(result.Program.IsValid);
        Assert.Equal(1, guest.Host.TableSynchronizations);
        Assert.Empty(guest.Host.ComputePipelines);
        Assert.Empty(guest.Compiler.Requests);
    }

    [Fact]
    public void UnreadablePointerFatalDebugSwitchRestoresTheOldPolicy()
    {
        const string variable = "SHARPEMU_GPU_MATERIALIZE_FATAL";
        var previous = Environment.GetEnvironmentVariable(variable);
        try
        {
            Environment.SetEnvironmentVariable(variable, "1");
            using var fatal = new FatalScope();
            var guest = new PipelineTestGuest();
            guest.RegisterProgram(CodeAddress, HeaderAddress,
                [0xF4080000, 0xFA000000, 0xE0000000, 0x80000000, 0xBF810000]);
            var source = guest.Source(CodeAddress, ShaderStage.Compute, [0x3F000000u, 0u]);
            var cursor = 0u;
            Assert.Throws<SchedulerFatalException>(() => guest.Programs.TryGetProgram(source,
                PipelineTestGuest.ComputeOptions(1), false, ref cursor, out _, out _, out _));
            Assert.Equal(0, guest.Host.TableSynchronizations);
        }
        finally { Environment.SetEnvironmentVariable(variable, previous); }
    }

    [Fact]
    public void UnexpectedCompilerErrorsAreNotConvertedIntoSkippedDispatches()
    {
        var guest = new PipelineTestGuest(_ => throw new InvalidOperationException("Unexpected compiler error."));
        guest.RegisterProgram(CodeAddress, HeaderAddress, PipelineTestGuest.EndProgram);
        var source = guest.Source(CodeAddress, ShaderStage.Compute, []);
        var cursor = 0u;
        Assert.Throws<InvalidOperationException>(() => guest.Programs.TryGetProgram(
            source, PipelineTestGuest.ComputeOptions(1), false, ref cursor, out _, out _, out _));
    }

    [Fact]
    public void DirectProgramLookupRemainsStrict()
    {
        using var fatal = new FatalScope();
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(CodeAddress, HeaderAddress, PipelineTestGuest.EndProgram);
        guest.Compiler.Rejection = "The test instruction is not supported.";
        var source = guest.Source(CodeAddress, ShaderStage.Pixel, []);
        var options = new StageCompileOptions { PixelInfo = new PixelInputInfo() };
        var cursor = 0u;
        Assert.Throws<SchedulerFatalException>(() => guest.Programs.GetOrCompile(source, options, ref cursor, out _));
        Assert.Throws<SchedulerFatalException>(() => guest.Programs.TryGetProgram(
            source, options, true, ref cursor, out _, out _, out _));
    }

    private static ComputeStageRegisters Registers() => new()
    {
        Address = CodeAddress, ThreadsX = 1, ThreadsY = 1, ThreadsZ = 1, UserScalarCount = 8,
    };
}
