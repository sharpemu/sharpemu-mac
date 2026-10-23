// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class WorkerPipelineTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private sealed class Flips : ICommandStreamFlipTarget
    {
        public ulong Prepare(int handle, int index, int mode, long argument) => 1;
        public void Complete(ulong id) { }
        public void Presented(ulong id) { }
        public bool IsDone(int handle, int index) => true;
    }

    // The probe pauses a real Metal draw, rather than replacing encoding with a
    // sleep. All commands and immutable bank records reach the native renderer.
    private sealed class Probe(MetalRenderHost target, ICpuMemory memory) : ManagedCommandStreamHost(memory, new EndOfPipeEvents(), new Flips()), ICommandIssueRenderer
    {
        internal Action? BeforeFirstDraw;
        internal readonly List<(ulong Packet, float Width)> Commands = [];
        private ICommandIssueRenderer Renderer => target;
        public bool EncodesDispatchWithoutProducerRetry => Renderer.EncodesDispatchWithoutProducerRetry;
        public void BeginIssuedSubmission(int queue, ulong submit) => Renderer.BeginIssuedSubmission(queue, submit);
        public void ResetIssuedQueue(int queue) => Renderer.ResetIssuedQueue(queue);
        public void DrawIssued(int queue, ulong submit, RegisterBanks banks, in IssueProgramIdentities programs,
            bool indexed, in DrawIndexedArguments indexedArguments, in DrawAutoArguments autoArguments)
        {
            if (Commands.Count == 0) BeforeFirstDraw?.Invoke();
            Commands.Add((autoArguments.PacketAddress, banks.Context.LineWidth));
            Renderer.DrawIssued(queue, submit, banks, programs, indexed, indexedArguments, autoArguments);
        }
        public void DispatchIssued(int queue, ulong submit, RegisterBanks banks, in IssueProgramIdentities programs,
            uint x, uint y, uint z, uint initiator, ulong indirect) =>
            Renderer.DispatchIssued(queue, submit, banks, programs, x, y, z, initiator, indirect);
    }

    [Fact]
    public void MetalIssueStreamOverlapsProducerAndEncoderWithOneFinalAcknowledgement()
    {
        (byte[] Pixels, (ulong, float)[] Commands)? Replay(bool split)
        {
            using var rig = CreateRig();
            if (rig is null) return null;
            const uint size = 16;
            var target = rig.ColorTarget(size, size);
            var state = new SceneState(size, size).ColorTarget(0, target, size, size);
            rig.Submit(rig.DrawMesh(new ScenePm4(), state, SceneDraws.Quad(0.5f, 1f, 0.25f, 0f)));
            var probe = new Probe(rig.Host, rig.Memory);
            Thread? consumer = null;
            var producerThread = Environment.CurrentManagedThreadId;
            var captures = 0;
            var issues = new OrderedCommandStreamHost(probe, split, synchronizeCode: (address, bytes) =>
            {
                Assert.Equal(split ? consumer!.ManagedThreadId : producerThread, Environment.CurrentManagedThreadId);
                Interlocked.Increment(ref captures);
                rig.Host.SynchronizeShaderCode(address, bytes);
            }, blocks: 4, itemsPerBlock: 32);
            issues.AttachQueue(rig.Host.CommandStream);
            using var entered = new ManualResetEventSlim();
            using var release = new ManualResetEventSlim();
            if (split)
            {
                probe.BeforeFirstDraw = () => { entered.Set(); Assert.True(release.Wait(TimeSpan.FromSeconds(10))); };
                consumer = new Thread(() => issues.Consume(rig.Relay.BindCurrentThread, () => MetalAutoreleasePool.Enter()))
                { Name = "Metal test issue consumer" };
                consumer.Start();
            }
            try
            {
                issues.BeginSubmission(0, 100, null);
                for (ulong index = 0; index < 1024; index++)
                {
                    var banks = rig.Host.CommandStream.GetInterpreter(0).TypedRegisters;
                    banks.Context.LineWidth = 8u + (uint)(index % 4);
                    issues.DrawAuto(100, new(index, ScenePm4.OpDrawIndexAuto, 6, 1, 0, 0, DrawOffsetSource.Packet) { MustExecute = true });
                    if (split && index == 62)
                    {
                        Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
                        // B is still in the first draw; A has already published two
                        // complete blocks. No timing threshold is needed to prove it.
                        var overlap = issues.FormatProfile();
                        Assert.Contains("issue_blocks=2 ", overlap);
                        Assert.Contains("issue_sync_stalls=0 ", overlap);
                        output.WriteLine("overlap " + overlap);
                        release.Set();
                    }
                }
                issues.Drain();
                var profile = issues.FormatProfile();
                output.WriteLine((split ? "split " : "serial ") + profile);
                Assert.Contains("issue_blocks=33 ", profile);
                Assert.Contains("ack_Sync=1/", profile);
                Assert.DoesNotContain("ack_SynchronizeShaderCode=", profile);
                Assert.Equal(1024, probe.Commands.Count);
                Assert.True(captures > 0);
            }
            finally
            {
                release.Set();
                issues.Stop();
                if (consumer is not null) Assert.True(consumer.Join(10000));
                rig.Relay.BindCurrentThread();
            }
            rig.Drain();
            Assert.Empty(rig.Fatals);
            return (rig.ReadImage(target, size, size), probe.Commands.ToArray());
        }
        var serial = Replay(false);
        if (serial is null) return;
        var split = Replay(true)!.Value;
        Assert.Equal(serial.Value.Commands, split.Commands);
        Assert.Equal(serial.Value.Pixels, split.Pixels);
        Assert.Equal(0xFF0040FFu, SceneDraws.Pixel(split.Pixels, 16, 8, 8));
    }
}
