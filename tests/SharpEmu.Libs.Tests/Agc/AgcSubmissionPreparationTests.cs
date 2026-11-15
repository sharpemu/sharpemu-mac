// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Buffers.Binary;
using System.Reflection;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Tests.Gpu.GpuCommands;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

[Collection(SchedulingStateCollection.Name)]
public sealed class AgcSubmissionPreparationTests
{
    private const ulong Base = 0x10000000, Command = Base + 0x100, Packet = Base + 0x300;

    [Theory]
    [InlineData("dcb")] [InlineData("acb")] [InlineData("multi")]
    public void ActualExportCompletedPrefixCallbackRunsAfterBothGatesAndAllAdmissions(string kind)
    {
        var memory = Memory(); var backend = new PreparingBackend(); using var scope = new BackendScope(backend);
        var host = new RecordingCommandStreamHost(); var queue = new CommandStreamQueue(host);
        var completions = 0;
        host.EndPrefixCapture = complete => complete ? new CommandPrefixCompletionTests.Completion(() =>
        {
            CheckGates(memory, false);
            Assert.Equal(kind == "multi" ? 2 : 1, backend.Admissions);
            completions++;
        }) : null;
        backend.CreateReceipt = (code, address, count, id, snapshots) =>
        {
            CheckGates(memory, true);
            host.WriteWords(address, [0xC0001000u, 0u]);
            return queue.EnqueueWithPreparation(code, address, count, id, snapshots);
        };
        var worker = new CommandStreamWorker(queue, host, static () => false, true, "actual AGC prefix"); worker.Start();
        try
        {
            Assert.Equal(0, Submit(memory, kind));
            Assert.Equal(kind == "multi" ? 2 : 1, completions);
        }
        finally { worker.Stop(); }
    }

    [Theory]
    [InlineData("dcb", false)] [InlineData("dcb", true)]
    [InlineData("acb", false)] [InlineData("acb", true)]
    [InlineData("multi", false)] [InlineData("multi", true)]
    public void ActualExportWaitsOnlyAfterBothAgcGatesAndAllMultiAdmissions(string kind, bool deferred)
    {
        var memory = Memory(); var backend = new PreparingBackend(); using var scope = new BackendScope(backend);
        backend.OnAdmission = () => CheckGates(memory, true);
        backend.OnWait = () =>
        {
            CheckGates(memory, false);
            Assert.Equal(kind == "multi" ? 2 : 1, backend.Admissions);
            return deferred ? GuestSubmissionPreparationResult.DeferredDependency : GuestSubmissionPreparationResult.PreparedPrefix;
        };
        Assert.Equal(0, Submit(memory, kind));
        Assert.Equal(kind == "multi" ? 2 : 1, backend.Waits);
        Assert.Equal(kind == "acb" ? 0x20u : 0u, backend.LastQueue);
    }

    [Theory]
    [InlineData("dcb")] [InlineData("acb")] [InlineData("multi")]
    public void HostsWithoutOptionalPreparationKeepImmediateSubmissionContract(string kind)
    {
        var backend = new SubmissionBackend(); using var scope = new BackendScope(backend);
        Assert.Equal(0, Submit(Memory(), kind));
        Assert.Equal(kind == "multi" ? 2 : 1, backend.Admissions);
    }

    [Fact]
    public void OriginalPreparationFailureLeavesAgcGatesReleasedAndIsNotSuccessfulSubmit()
    {
        var memory = Memory(); var backend = new PreparingBackend(); using var scope = new BackendScope(backend);
        var original = new NotSupportedException("original accepted library error");
        backend.OnWait = () => { CheckGates(memory, false); throw original; };
        Assert.Same(original, Assert.Throws<NotSupportedException>(() => Submit(memory, "dcb")));
        CheckGates(memory, false);
    }

    [Fact]
    public void CancelledPreparationUsesExistingCancelledExportRoute()
    {
        var backend = new PreparingBackend(); using var scope = new BackendScope(backend);
        backend.OnWait = () => throw new OperationCanceledException("exact accepted session closed");
        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_CANCELED, Submit(Memory(), "dcb"));
        Assert.Equal(1, backend.Admissions); Assert.Equal(1, backend.Waits);
    }

    // A set-shader-register packet: only the submit-time command-buffer walk shadows it.
    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 1)]
    public void SubmitWalksTheCommandBufferOnlyForAHostThatReadsGeometrySnapshots(bool consumesSnapshots, int shadowedRegisters)
    {
        var memory = Memory();
        Write(memory, Command, 0xC0010000u | (0x76u << 8)); Write(memory, Command + 4, 0x48u); Write(memory, Command + 8, 1u);
        Write(memory, Packet + 8, 3u);
        SubmissionBackend backend = consumesSnapshots ? new SnapshotConsumingBackend() : new SubmissionBackend();
        using var scope = new BackendScope(backend);
        Assert.Equal(0, Submit(memory, "dcb"));
        Assert.Equal(1, backend.Admissions);
        Assert.Equal(shadowedRegisters, AgcExports.GetGeometryCaptureRegistersForTests(memory).Count);
    }

    // The split backend's host reads the stream from guest memory: a copy made at submit was
    // dropped unread on every submit.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SubmitCopiesTheCommandBufferOnlyForAHostThatReadsTheCopy(bool consumesCopy)
    {
        var memory = Memory();
        SubmissionBackend backend = consumesCopy ? new SubmissionBackend() : new StreamReadingBackend();
        using var scope = new BackendScope(backend);
        Assert.Equal(0, Submit(memory, "dcb"));
        Assert.Equal(1, backend.Admissions);
        Assert.Equal(consumesCopy, backend.LastSnapshots is SubmittedCommandBuffer);
        if (!consumesCopy) Assert.Null(backend.LastSnapshots);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SubmitOfAnUnmappedCommandBufferIsAMemoryFaultWhetherOrNotTheHostReadsTheCopy(bool consumesCopy)
    {
        var memory = Memory();
        Write(memory, Packet, Base + 0x100000); // Outside the 0x1000 bytes that are mapped.
        SubmissionBackend backend = consumesCopy ? new SubmissionBackend() : new StreamReadingBackend();
        using var scope = new BackendScope(backend);
        Assert.Equal(unchecked((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT), Submit(memory, "dcb"));
        Assert.Equal(0, backend.Admissions);
    }

    private static FakeCpuMemory Memory()
    {
        var memory = new FakeCpuMemory(Base, 0x1000);
        Write(memory, Command, 0xC0001000u); Write(memory, Command + 4, 0u); // Two-word NOP.
        Write(memory, Packet, Command); Write(memory, Packet + 8, 2u);
        Write(memory, Packet + 0x40, Command); Write(memory, Packet + 0x48, Command);
        Write(memory, Packet + 0x60, 2u); Write(memory, Packet + 0x64, 2u);
        return memory;
    }
    private static int Submit(FakeCpuMemory memory, string kind)
    {
        var context = new CpuContext(memory, Generation.Gen5);
        if (kind == "multi")
        { context[CpuRegister.Rdi] = Packet + 0x40; context[CpuRegister.Rsi] = Packet + 0x60; context[CpuRegister.Rdx] = 2; return AgcExports.DriverSubmitMultiDcbs(context); }
        if (kind == "acb")
        { context[CpuRegister.Rdi] = 0x20; context[CpuRegister.Rsi] = Packet; return AgcExports.DriverSubmitAcb(context); }
        context[CpuRegister.Rdi] = Packet; return AgcExports.DriverSubmitDcb(context);
    }
    private static void Write(FakeCpuMemory memory, ulong address, uint value)
    { Span<byte> bytes = stackalloc byte[4]; BinaryPrimitives.WriteUInt32LittleEndian(bytes, value); Assert.True(memory.TryWrite(address, bytes)); }
    private static void Write(FakeCpuMemory memory, ulong address, ulong value)
    { Span<byte> bytes = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(bytes, value); Assert.True(memory.TryWrite(address, bytes)); }
    private static void CheckGates(FakeCpuMemory memory, bool entered)
    {
        var table = typeof(AgcExports).GetField("_submittedGpuStates", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        object?[] args = [memory, null];
        Assert.True((bool)table.GetType().GetMethod("TryGetValue")!.Invoke(table, args)!);
        var state = args[1]!;
        foreach (var name in new[] { "Gate", "CommandSubmissionGate" })
            Assert.Equal(entered, Monitor.IsEntered(state.GetType().GetProperty(name)!.GetValue(state)!));
    }
    private sealed class BackendScope : IDisposable
    {
        private readonly FieldInfo field = typeof(GuestGpu).GetField("_instance", BindingFlags.Static | BindingFlags.NonPublic)!;
        private readonly object? previous;
        public BackendScope(IGuestGpuBackend backend) { previous = field.GetValue(null); field.SetValue(null, backend); }
        public void Dispose() => field.SetValue(null, previous);
    }
    private class SubmissionBackend : IGuestGpuBackend
    {
        public int Admissions; public uint LastQueue; public object? LastSnapshots;
        public Action? OnAdmission;
        public string BackendName => "managed preparation control";
        public void SubmitCommandStream(ICpuMemory memory, uint queue, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
        { OnAdmission?.Invoke(); LastQueue = queue; LastSnapshots = geometrySnapshots; Admissions++; }
        public void EnsureStarted(uint width, uint height) => throw new NotSupportedException();
        public bool TryCompileProgram(ShaderCompileRequest request, out IGuestCompiledShader? shader, out string error) => throw new NotSupportedException();
        public IGuestCompiledShader GetDepthOnlyFragmentShader() => throw new NotSupportedException();
        public void HideSplashScreen() => throw new NotSupportedException();
        public void Submit(byte[] pixels, uint width, uint height) => throw new NotSupportedException();
        public void SubmitGuestDraw(GuestDrawKind kind, uint width, uint height) => throw new NotSupportedException();
        public bool TrySubmitGuestImage(int handle, int index, ulong address, uint width, uint height, uint pitch, ulong id) => throw new NotSupportedException();
        public IdleOutcome SubmitDone(ICpuMemory memory) => throw new NotSupportedException();
        public void RegisterKnownDisplayBuffer(ulong address, uint format) => throw new NotSupportedException();
        public bool IsGpuGuestImageAvailable(ulong address, uint format, uint type) => false;
        public void CountShaderCompilation() => throw new NotSupportedException();
        public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() => throw new NotSupportedException();
        public void RequestClose() => throw new NotSupportedException();
    }
    private sealed class SnapshotConsumingBackend : SubmissionBackend, IGuestGpuBackend
    {
        public bool ConsumesSubmittedGeometrySnapshots => true;
    }
    private sealed class StreamReadingBackend : SubmissionBackend, IGuestGpuBackend
    {
        public bool ConsumesSubmittedCommandBytes => false;
    }
    private sealed class PreparingBackend : SubmissionBackend, IGuestSubmissionPreparationBackend
    {
        public int Waits; public Func<GuestSubmissionPreparationResult>? OnWait;
        public Func<uint, ulong, uint, ulong, object?, IGuestSubmissionPreparation>? CreateReceipt;
        public IGuestSubmissionPreparation SubmitCommandStreamWithPreparation(ICpuMemory memory, uint queue, ulong address, uint dwordCount, ulong submissionId, object? snapshots)
        { SubmitCommandStream(memory, queue, address, dwordCount, submissionId, snapshots); return CreateReceipt?.Invoke(queue, address, dwordCount, submissionId, snapshots) ?? new Receipt(this); }
        private sealed class Receipt(PreparingBackend backend) : IGuestSubmissionPreparation
        {
            public GuestSubmissionPreparationResult WaitUntilPrepared() { backend.Waits++; return backend.OnWait?.Invoke() ?? GuestSubmissionPreparationResult.PreparedPrefix; }
        }
    }
}
