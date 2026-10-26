// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Buffers.Binary;
using System.Diagnostics;
using System.Reflection;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Tests.Gpu.GpuCommands;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Libs.Tests.Agc;

// A guest render thread blocked in sceAgcSuspendPoint (the worker one boundary behind) and a second
// guest thread submitting: the frame-boundary wait no longer holds CommandSubmissionGate.
[Collection(SchedulingStateCollection.Name)]
public sealed class AgcSuspendPointGateTests(ITestOutputHelper output)
{
    private const ulong Base = 0x10000000;
    private const ulong FirstDcb = Base + 0x100, SecondDcb = Base + 0x200, Acb = Base + 0x280;
    private const ulong FirstPacket = Base + 0x300, SecondPacket = Base + 0x340, AcbPacket = Base + 0x380;
    private static readonly TimeSpan WorkerLag = TimeSpan.FromMilliseconds(250);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ASubmitFromAnotherThreadDoesNotWaitForTheFrameBoundary(bool split)
    {
        var (memory, host, queue) = Setup();
        using var scope = new BackendScope(new QueueBackend(queue));
        var previous = AgcExports.SplitSubmitDoneWait;
        AgcExports.SplitSubmitDoneWait = split;
        try
        {
            // Frame 1, then frame 2's boundary: the worker has run nothing, so the second blocks.
            Assert.Equal(0, SubmitDcb(memory, FirstPacket));
            Assert.Equal(0, SuspendPoint(memory));
            Assert.Equal(0, SubmitDcb(memory, FirstPacket));
            var render = Task.Run(() => SuspendPoint(memory));
            await Task.WhenAny(render, Task.Delay(100));
            Assert.False(render.IsCompleted);

            // Another guest thread's async compute submit, measured while the boundary waits.
            var watch = Stopwatch.StartNew();
            var compute = Task.Run(() => { var result = SubmitAcb(memory); watch.Stop(); return result; });
            await Task.WhenAny(compute, Task.Delay(WorkerLag));
            Assert.Equal(split, compute.IsCompleted);

            Drain(queue, render);
            Assert.Equal(0, await render.WaitAsync(TimeSpan.FromSeconds(10)));
            Assert.Equal(0, await compute.WaitAsync(TimeSpan.FromSeconds(10)));
            output.WriteLine(FormattableString.Invariant(
                $"split={split} acb_submit_latency_ms={watch.Elapsed.TotalMilliseconds:F3} worker_lag_ms={WorkerLag.TotalMilliseconds}"));
            if (!split)
            {
                Assert.True(watch.Elapsed >= WorkerLag * 0.8, $"the gated submit returned after {watch.Elapsed}");
            }
        }
        finally
        {
            AgcExports.SplitSubmitDoneWait = previous;
        }
    }

    // The capture reset stays where the boundary is in submit order: a DCB submitted while the
    // boundary is waited on keeps the registers it shadowed.
    [Fact]
    public async Task ADcbSubmittedDuringTheWaitKeepsItsGeometryCapture()
    {
        var (memory, _, queue) = Setup();
        using var scope = new BackendScope(new QueueBackend(queue));
        Assert.Equal(0, SubmitDcb(memory, FirstPacket));
        Assert.Equal(0, SuspendPoint(memory));
        Assert.Equal(0, SubmitDcb(memory, FirstPacket));
        var render = Task.Run(() => SuspendPoint(memory));
        await Task.WhenAny(render, Task.Delay(100));
        Assert.False(render.IsCompleted);

        Assert.Equal(0, await Task.Run(() => SubmitDcb(memory, SecondPacket)).WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.False(render.IsCompleted);
        Drain(queue, render);
        Assert.Equal(0, await render.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(new Dictionary<uint, uint> { [0x49] = 2 }, AgcExports.GetGeometryCaptureRegistersForTests(memory));
    }

    [Fact]
    public async Task ConcurrentBoundariesEachWaitOnlyForTheirOwn()
    {
        var queue = new CommandStreamQueue(new RecordingCommandStreamHost(), boundariesInFlight: 1);
        Assert.Equal(IdleOutcome.Completed, queue.BeginDone(out var first));
        Assert.Equal(IdleOutcome.Completed, queue.BeginDone(out var second));
        Assert.Equal(IdleOutcome.Completed, queue.BeginDone(out var third));
        Assert.Equal(IdleOutcome.Completed, queue.WaitDone(first));

        var secondWait = Task.Run(() => queue.WaitDone(second));
        var thirdWait = Task.Run(() => queue.WaitDone(third));
        await Task.WhenAny(secondWait, Task.Delay(100));
        Assert.False(secondWait.IsCompleted);

        // The first boundary retiring is enough for the second; the third still has two ahead.
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(IdleOutcome.Completed, await secondWait.WaitAsync(TimeSpan.FromSeconds(10)));
        await Task.WhenAny(thirdWait, Task.Delay(100));
        Assert.False(thirdWait.IsCompleted);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(IdleOutcome.Completed, await thirdWait.WaitAsync(TimeSpan.FromSeconds(10)));
    }

    private static (FakeCpuMemory Memory, RecordingCommandStreamHost Host, CommandStreamQueue Queue) Setup()
    {
        var memory = new FakeCpuMemory(Base, 0x1000);
        var host = new RecordingCommandStreamHost();
        // Two DCBs that each set one shader register, and a NOP compute buffer.
        uint[] first = [0xC0010000u | (0x76u << 8), 0x48u, 1u];
        uint[] second = [0xC0010000u | (0x76u << 8), 0x49u, 2u];
        uint[] nop = [0xC0001000u, 0u];
        Write(memory, host, FirstDcb, first);
        Write(memory, host, SecondDcb, second);
        Write(memory, host, Acb, nop);
        WritePacket(memory, FirstPacket, FirstDcb, (uint)first.Length);
        WritePacket(memory, SecondPacket, SecondDcb, (uint)second.Length);
        WritePacket(memory, AcbPacket, Acb, (uint)nop.Length);
        return (memory, host, new CommandStreamQueue(host, boundariesInFlight: 1));
    }

    // The worker, late: runs everything queued until the guest's boundary wait returns.
    private static void Drain(CommandStreamQueue queue, Task render)
    {
        var watch = Stopwatch.StartNew();
        while (!render.IsCompleted && watch.Elapsed < TimeSpan.FromSeconds(10))
        {
            if (queue.ProcessOne() is SliceResult.NoWork or SliceResult.AllBlocked)
            {
                Thread.Sleep(1);
            }
        }
    }

    private static int SubmitDcb(FakeCpuMemory memory, ulong packet)
    {
        var context = new CpuContext(memory, Generation.Gen5);
        context[CpuRegister.Rdi] = packet;
        return AgcExports.DriverSubmitDcb(context);
    }

    private static int SubmitAcb(FakeCpuMemory memory)
    {
        var context = new CpuContext(memory, Generation.Gen5);
        context[CpuRegister.Rdi] = 0x20;
        context[CpuRegister.Rsi] = AcbPacket;
        return AgcExports.DriverSubmitAcb(context);
    }

    private static int SuspendPoint(FakeCpuMemory memory) => AgcExports.SuspendPoint(new CpuContext(memory, Generation.Gen5));

    private static void Write(FakeCpuMemory memory, RecordingCommandStreamHost host, ulong address, uint[] words)
    {
        Span<byte> bytes = stackalloc byte[4];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes, words[index]);
            Assert.True(memory.TryWrite(address + (ulong)index * 4, bytes));
        }

        host.WriteWords(address, words);
    }

    private static void WritePacket(FakeCpuMemory memory, ulong packet, ulong address, uint dwords)
    {
        Span<byte> bytes = stackalloc byte[12];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, address);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes[8..], dwords);
        Assert.True(memory.TryWrite(packet, bytes));
    }

    private sealed class BackendScope : IDisposable
    {
        private readonly FieldInfo _field = typeof(GuestGpu).GetField("_instance", BindingFlags.Static | BindingFlags.NonPublic)!;
        private readonly object? _previous;
        public BackendScope(IGuestGpuBackend backend) { _previous = _field.GetValue(null); _field.SetValue(null, backend); }
        public void Dispose() => _field.SetValue(null, _previous);
    }

    // A host whose queue is run by the test as a worker that is behind.
    private sealed class QueueBackend(CommandStreamQueue queue) : IGuestGpuBackend
    {
        public string BackendName => "suspend point gate";
        public bool ConsumesSubmittedGeometrySnapshots => true;
        public void SubmitCommandStream(ICpuMemory memory, uint queueId, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
        {
            if (queueId == 0) queue.EnqueueGraphics(address, dwordCount, submissionId, geometrySnapshots);
            else queue.EnqueueCompute(queueId, address, dwordCount, submissionId, geometrySnapshots);
        }
        public IdleOutcome SubmitDone(ICpuMemory memory) => queue.Done();
        public IdleOutcome BeginSubmitDone(ICpuMemory memory, out Func<IdleOutcome>? wait)
        {
            var outcome = queue.BeginDone(out var ticket);
            wait = ticket != 0 ? () => queue.WaitDone(ticket) : null;
            return outcome;
        }
        public void EnsureStarted(uint width, uint height) => throw new NotSupportedException();
        public bool TryCompileProgram(ShaderCompileRequest request, out IGuestCompiledShader? shader, out string error) => throw new NotSupportedException();
        public IGuestCompiledShader GetDepthOnlyFragmentShader() => throw new NotSupportedException();
        public void HideSplashScreen() => throw new NotSupportedException();
        public void Submit(byte[] pixels, uint width, uint height) => throw new NotSupportedException();
        public void SubmitGuestDraw(GuestDrawKind kind, uint width, uint height) => throw new NotSupportedException();
        public bool TrySubmitGuestImage(int handle, int index, ulong address, uint width, uint height, uint pitch, ulong id) => throw new NotSupportedException();
        public void RegisterKnownDisplayBuffer(ulong address, uint format) => throw new NotSupportedException();
        public bool IsGpuGuestImageAvailable(ulong address, uint format, uint type) => false;
        public void CountShaderCompilation() => throw new NotSupportedException();
        public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() => throw new NotSupportedException();
        public void RequestClose() => throw new NotSupportedException();
    }
}
