// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

[Collection(SchedulingStateCollection.Name)]
public sealed class OrderedCommandStreamHostTests
{
    private sealed class FlipTarget : ICommandStreamFlipTarget
    {
        public ulong Prepare(int handle, int index, int mode, long argument) => 1;
        public void Complete(ulong requestId) { }
        public void Presented(ulong requestId) { }
        public bool IsDone(int handle, int index) => true;
    }

    // The same RenderExecutor and recording renderer used by the draw/compute suites:
    // the comparison below includes its emitted resources, state and semantic commands.
    private sealed class EncodedHost : ManagedCommandStreamHost, ICommandIssueRenderer, ICommandStreamHost
    {
        internal readonly RecordingRenderHost Render;
        internal readonly FakePipelineProvider Pipelines = new();
        internal readonly RenderExecutor Executor;
        internal readonly List<string> Semantics = [];
        internal Action? BeforeDraw;
        internal Action? BeforeRead;
        internal bool PipelineNotReady;
        internal bool Delay;
        internal int Draws;
        internal EncodedHost(RecordingRenderHost render) : base(render.GuestMemory, new EndOfPipeEvents(), new FlipTarget())
        { Render = render; Executor = new(render, Pipelines); }
        public void BeginIssuedSubmission(int queueId, ulong submitId) => Semantics.Add($"begin {queueId} {submitId}");
        public void ResetIssuedQueue(int queueId) => Semantics.Add($"reset {queueId}");
        public void DrawIssued(int queueId, ulong submitId, RegisterBanks banks, in IssueProgramIdentities programs,
            bool indexed, in DrawIndexedArguments indexedArguments, in DrawAutoArguments autoArguments)
        {
            BeforeDraw?.Invoke();
            if (Delay && (++Draws & 3) == 0) Thread.Yield();
            Semantics.Add($"draw {queueId}/{submitId} {banks.Context.LineWidth} {banks.Context.ScreenViewport.Viewports[0].XScale} {banks.Shader.Vertex.ExportUserScalars.Values[0]}");
            if (indexed) Executor.DrawIndexed(submitId, banks, indexedArguments);
            else Executor.DrawAuto(submitId, banks, autoArguments);
        }
        public void DispatchIssued(int queueId, ulong submitId, RegisterBanks banks, in IssueProgramIdentities programs,
            uint x, uint y, uint z, uint initiator, ulong indirect)
        {
            Semantics.Add($"dispatch {queueId}/{submitId} {x}/{y}/{z} {initiator} {indirect}");
            if (!PipelineNotReady) Executor.Dispatch(submitId, banks, x, y, z, initiator, indirect);
        }
        public bool ConsumePipelineNotReady() { var result = PipelineNotReady; PipelineNotReady = false; return result; }
        public override bool TryReadGuest(ulong address, Span<byte> destination)
        { BeforeRead?.Invoke(); return base.TryReadGuest(address, destination); }
        public override bool DefersEndOfPipeLabels => true;
        public override void RecordEndOfPipe(in EndOfPipeWrite write)
        {
            Semantics.Add($"eop {write.Kind} {write.Destination} {write.Value}");
            if (write.Kind == EndOfPipeWriteKind.Write32) Memory.TryWrite(write.Destination, BitConverter.GetBytes((uint)write.Value));
            base.RecordEndOfPipe(write);
        }
        public override ulong PrepareFlip(int handle, int index, int flipMode, long flipArgument)
        { Semantics.Add($"flip {handle}/{index}/{flipMode}/{flipArgument}"); return base.PrepareFlip(handle, index, flipMode, flipArgument); }
        public override void Flush() => Semantics.Add("flush");
    }

    private sealed class Runner : IDisposable
    {
        internal readonly OrderedCommandStreamHost Host;
        internal readonly CommandStreamQueue Queue;
        private readonly Thread? _consumer;
        internal Runner(ICommandStreamHost target, bool split, int items = 64)
        {
            Host = new(target, split, itemsPerBlock: items, blocks: 2);
            Queue = new(Host);
            Host.AttachQueue(Queue);
            if (split) { _consumer = new Thread(() => Host.Consume()); _consumer.Start(); }
        }
        public void Dispose()
        {
            Host.Stop();
            if (_consumer is not null) Assert.True(_consumer.Join(5000));
        }
    }

    [Fact]
    public void RecordedMixedPm4StreamEmitsTheSameSemanticCommandsInSerialAndSplit()
    {
        using var fatal = new FatalScope();
        static (string[] Encoded, string[] Semantics) Replay(bool split)
        {
            var target = new EncodedHost(new()) { Delay = split };
            using var runner = new Runner(target, split, 7);
            Banks().CopyTo(runner.Queue.GetInterpreter(0).TypedRegisters);
            var packets = new List<uint[]>();
            for (uint draw = 0; draw < 300; draw++)
            {
                packets.Add(StreamRunner.Packet(PacketOpcode.NumInstances, 1 + draw % 3));
                packets.Add(StreamRunner.Packet(PacketOpcode.SetContextRegister, ContextRegisterOffset.PaSuLineCntl, 8 + draw % 4));
                packets.Add(StreamRunner.Packet(PacketOpcode.DrawIndexAuto, 3 + 3 * (draw % 5), 0));
                if (draw % 3 == 0) packets.Add(StreamRunner.Packet(PacketOpcode.DispatchDirect, 1 + draw % 7, 2, 1, 1));
            }
            var words = StreamRunner.Concat(packets.ToArray());
            const ulong commands = RecordingRenderHost.MemoryBase + 0x800000;
            target.Memory.TryWrite(commands, System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan()));
            runner.Queue.EnqueueGraphics(commands, (uint)words.Length, 1, null);
            Assert.Equal(SliceResult.Completed, runner.Queue.ProcessOne());
            runner.Host.Drain();
            Assert.Equal(300, target.Semantics.Count(s => s.StartsWith("draw ")));
            Assert.Equal(100, target.Semantics.Count(s => s.StartsWith("dispatch ")));
            Assert.NotEmpty(target.Render.DynamicStates);
            return (target.Render.Calls.ToArray(), target.Semantics.ToArray());
        }
        var serial = Replay(false);
        var split = Replay(true);
        Assert.Equal(serial.Encoded, split.Encoded);
        Assert.Equal(serial.Semantics, split.Semantics);
    }

    [Fact]
    public void ProducerCanFillAnotherBlockWhileConsumerIsInsideADraw()
    {
        using var fatal = new FatalScope();
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var target = new EncodedHost(new());
        target.BeforeDraw = () => { entered.Set(); Assert.True(release.Wait(TimeSpan.FromSeconds(10))); };
        using var runner = new Runner(target, true, 16);
        Banks().CopyTo(runner.Queue.GetInterpreter(0).TypedRegisters);
        try
        {
            for (var i = 0; i < 16; i++) runner.Host.DrawAuto(1, Auto(3));
            Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
            for (var i = 0; i < 16; i++) runner.Host.DrawAuto(1, Auto(6));
            Assert.Contains("issue_blocks=2", runner.Host.FormatProfile());
            Assert.Contains("issue_sync_stalls=0", runner.Host.FormatProfile());
        }
        finally { release.Set(); }
        for (var i = 0; i < 4096; i++) runner.Host.DrawAuto(1, Auto(3));
        runner.Host.Drain();
        Assert.Equal(4128, target.Semantics.Count(s => s.StartsWith("draw ")));
        Assert.Contains("ack_Sync=1/", runner.Host.FormatProfile());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AsynchronousCommandProcessorWritesOwnLargePayloadsUntilConsumption(bool split)
    {
        using var fatal = new FatalScope();
        var target = new EncodedHost(new());
        using var runner = new Runner(target, split, 16);
        const ulong address = RecordingRenderHost.MemoryBase + 0x900000;
        var bytes = Enumerable.Range(0, 2048).Select(i => (byte)i).ToArray();
        var expected = bytes.ToArray();
        runner.Host.RecordCommandProcessorWrite(address, bytes, "owned-payload");
        Array.Fill(bytes, (byte)99);
        runner.Host.Drain();
        var read = new byte[expected.Length];
        Assert.True(target.Memory.TryRead(address, read));
        Assert.Equal(expected, read);
        Assert.DoesNotContain("ack_RecordCommandProcessorWrite=", runner.Host.FormatProfile());
    }

    [Fact]
    public void QueuedBanksStayIndependentOfNestedWritesResetAndPushPop()
    {
        using var fatal = new FatalScope();
        var target = new EncodedHost(new());
        using var runner = new Runner(target, true);
        var live = runner.Queue.GetInterpreter(0).TypedRegisters;
        Banks().CopyTo(live);
        runner.Host.BeginSubmission(0, 1, null);
        live.Context.LineWidth = 2;
        live.Shader.Vertex.ExportUserScalars.Values[0] = 11;
        runner.Host.DrawAuto(1, Auto(3));
        live.ApplyContextState(ContextStateOperation.PushClear);
        live.Context.ScreenViewport.Viewports[0].XScale = 100;
        live.ApplyContextState(ContextStateOperation.Pop);
        live.Shader.Vertex.ExportUserScalars.Values[0] = 22;
        runner.Host.DrawAuto(1, Auto(6));
        live.Reset();
        runner.Host.Drain();
        Assert.Contains("draw 0/1 2 32 11", target.Semantics);
        Assert.Contains("draw 0/1 2 32 22", target.Semantics);
    }

    private sealed class FetchBoundsProbe : IProgramPrefetch
    {
        internal readonly List<VertexFetchBounds> Bounds = [];
        public bool TryTakeGraphics(RegisterBanks banks, in GraphicsProgramInputs inputs, out GraphicsPrograms programs)
        {
            Bounds.Add(inputs.FetchBounds);
            programs = null!;
            return false;
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void VertexFetchBoundsUseEachIssuedBanksAndPacketSnapshot(bool split)
    {
        var previous = AffineVertexFetchResolver.Enabled;
        AffineVertexFetchResolver.Enabled = true;
        try
        {
            using var fatal = new FatalScope();
            var target = new EncodedHost(new());
            var probe = new FetchBoundsProbe();
            target.Executor.Prefetch = probe;
            using var runner = new Runner(target, split);
            var live = runner.Queue.GetInterpreter(0).TypedRegisters;
            Banks().CopyTo(live);
            runner.Host.BeginSubmission(0, 1, null);
            live.UserConfig.IndexOffset = 4;
            runner.Host.DrawAuto(1, Auto(3, instances: 2, firstVertex: 7));
            live.UserConfig.IndexOffset = 20;
            runner.Host.DrawAuto(1, Auto(6, instances: 5, firstVertex: 9));
            live.Reset();
            runner.Host.Drain();
            Assert.Equal(new[]
            {
                new VertexFetchBounds(11, 3, 0, 2, VerticesKnown: true, InstancesKnown: true),
                new VertexFetchBounds(29, 6, 0, 5, VerticesKnown: true, InstancesKnown: true),
            }, probe.Bounds);
        }
        finally { AffineVertexFetchResolver.Enabled = previous; }
    }

    [Fact]
    public void UnresolvedWriterBlocksLaterGuestReadUntilBHasAppliedItsEffects()
    {
        using var fatal = new FatalScope();
        using var writerEntered = new ManualResetEventSlim();
        using var releaseWriter = new ManualResetEventSlim();
        using var readEntered = new ManualResetEventSlim();
        var target = new EncodedHost(new());
        const ulong address = RecordingRenderHost.MemoryBase + 0x900000;
        target.BeforeDraw = () =>
        {
            writerEntered.Set();
            Assert.True(releaseWriter.Wait(TimeSpan.FromSeconds(5)));
            target.Memory.TryWrite(address, BitConverter.GetBytes(123u));
        };
        target.BeforeRead = readEntered.Set;
        using var runner = new Runner(target, true, 1);
        Banks().CopyTo(runner.Queue.GetInterpreter(0).TypedRegisters);
        runner.Host.BeginSubmission(0, 1, null);
        runner.Host.DrawAuto(1, Auto(3));
        var read = Task.Run(() =>
        {
            Span<byte> bytes = stackalloc byte[4];
            Assert.True(runner.Host.TryReadGuestOperand(address, bytes));
            return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        });
        try
        {
            Assert.True(writerEntered.Wait(TimeSpan.FromSeconds(5)));
            Assert.False(readEntered.IsSet);
        }
        finally { releaseWriter.Set(); }
        Assert.True(read.Wait(TimeSpan.FromSeconds(5)));
        Assert.Equal(123u, read.Result);
        Assert.Contains("unresolved_writer_stalls=1", runner.Host.FormatProfile());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LabelsWaitOperandsFlipAndSuspendRelayObserveTheEncodedPrefix(bool split)
    {
        using var fatal = new FatalScope();
        var target = new EncodedHost(new());
        using var runner = new Runner(target, split);
        Banks().CopyTo(runner.Queue.GetInterpreter(0).TypedRegisters);
        runner.Host.BeginSubmission(0, 9, null);
        runner.Host.DrawAuto(9, Auto(3));
        const ulong label = RecordingRenderHost.MemoryBase + 0x900000;
        runner.Host.RecordEndOfPipe(new(EndOfPipeWriteKind.Write32, 9, label, 77));
        Span<byte> bytes = stackalloc byte[4];
        Assert.True(runner.Host.TryReadGuestWaitOperand(label, bytes, true));
        Assert.Equal(77u, BinaryPrimitives.ReadUInt32LittleEndian(bytes));
        runner.Host.PrepareFlip(1, 0, 0, 4);
        ((ICommandStreamHost)runner.Host).FlushFrame();
        runner.Host.ExecuteGuestMemoryOperation(() => target.Semantics.Add("relay/unmap"));
        Assert.Equal(new[] { "begin", "draw", "eop", "flip", "flush", "relay/unmap" },
            target.Semantics.Select(s => s.Split(' ')[0]));
    }

    [Fact]
    public void PipelineNotReadyIsAcknowledgedAndTheSameDispatchPacketIsRetried()
    {
        using var fatal = new FatalScope();
        var target = new EncodedHost(new()) { PipelineNotReady = true };
        using var runner = new Runner(target, true);
        Banks().CopyTo(runner.Queue.GetInterpreter(0).TypedRegisters);
        const ulong commands = RecordingRenderHost.MemoryBase + 0x800000;
        var words = StreamRunner.Packet(PacketOpcode.DispatchDirect, 2, 3, 4, 1);
        target.Memory.TryWrite(commands, System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan()));
        runner.Queue.EnqueueGraphics(commands, (uint)words.Length, 1, null);
        Assert.Equal(SliceResult.BlockedWithoutProgress, runner.Queue.ProcessOne());
        runner.Queue.NotifyPipelineReady();
        Assert.Equal(SliceResult.Completed, runner.Queue.ProcessOne());
        runner.Host.Drain();
        Assert.Equal(2, target.Semantics.Count(s => s.StartsWith("dispatch ")));
        Assert.Single(target.Render.Calls, s => s.StartsWith("dispatch "));
    }

    [Fact]
    public void PreallocatedBankCopyCoversEveryPublicFieldWithoutSharingNestedObjects()
    {
        static void Seed(object value)
        {
            foreach (var field in value.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance))
            {
                var type = field.FieldType;
                if (type == typeof(uint)) field.SetValue(value, 17u);
                else if (type == typeof(ulong)) field.SetValue(value, 0x12345678UL);
                else if (type == typeof(int)) field.SetValue(value, 13);
                else if (type == typeof(float)) field.SetValue(value, 2.5f);
                else if (type == typeof(bool)) field.SetValue(value, true);
                else if (type == typeof(byte)) field.SetValue(value, (byte)3);
                else if (type == typeof(ushort)) field.SetValue(value, (ushort)7);
                else if (type.IsEnum) field.SetValue(value, Enum.ToObject(type, 1));
                else if (field.GetValue(value) is Array array)
                {
                    for (var index = 0; index < array.Length; index++)
                    {
                        var element = array.GetValue(index)!;
                        if (element is uint) array.SetValue(99u, index);
                        else if (element is int) array.SetValue(21, index);
                        else if (element is float) array.SetValue(3.5f, index);
                        else if (element is byte) array.SetValue((byte)5, index);
                        else if (element is bool) array.SetValue(true, index);
                        else { Seed(element); array.SetValue(element, index); }
                    }
                }
                else if (field.GetValue(value) is Dictionary<uint, uint> dictionary) dictionary[51] = 76;
                else if (field.GetValue(value) is { } nested && !type.IsPrimitive)
                { Seed(nested); if (type.IsValueType) field.SetValue(value, nested); }
            }
        }
        static void Compare(object expected, object actual)
        {
            Assert.Equal(expected.GetType(), actual.GetType());
            if (expected.GetType().IsValueType) { Assert.Equal(expected, actual); return; }
            Assert.NotSame(expected, actual);
            if (expected is Array a && actual is Array b)
            {
                Assert.Equal(a.Length, b.Length);
                for (var index = 0; index < a.Length; index++) Compare(a.GetValue(index)!, b.GetValue(index)!);
                return;
            }
            if (expected is Dictionary<uint, uint> x && actual is Dictionary<uint, uint> y)
            { Assert.Equal(x.OrderBy(p => p.Key), y.OrderBy(p => p.Key)); return; }
            foreach (var field in expected.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance))
                Compare(field.GetValue(expected)!, field.GetValue(actual)!);
        }
        var source = Banks();
        Seed(source.Context); Seed(source.Shader); Seed(source.UserConfig);
        var destination = new RegisterBanks(source.Fatal);
        source.CopyTo(destination);
        Compare(source.Context, destination.Context);
        Compare(source.Shader, destination.Shader);
        Compare(source.UserConfig, destination.UserConfig);
    }

    [Fact]
    public void WarmSerialIssuePathDoesNotAllocatePerDrawOrWakePerDraw()
    {
        var target = new RecordingCommandStreamHost();
        using var runner = new Runner(target, false);
        // The allocation assertion covers publication/copying, with a nonallocating sink.
        // The recording host allocates on execution, so leave one block unpublished.
        runner.Host.DrawAuto(1, Auto(3));
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 32; i++) runner.Host.DrawAuto(1, Auto(3));
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
        Assert.Contains("issue_blocks=0", runner.Host.FormatProfile());
        runner.Host.Drain();
        Assert.Contains("issue_blocks=1", runner.Host.FormatProfile());
        Assert.Contains("issue_wakes=0", runner.Host.FormatProfile());
    }
}
