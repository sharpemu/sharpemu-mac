// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using System.Reflection;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class RecipeReplaySceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    [Fact]
    public void ChangedScalarTableBytesCannotReuseOldArgumentPointerContents()
    {
        using var rig = CreateRig(); if (rig is null) return;
        rig.Host.RecipeReplayEnabled = true;
        var table = rig.Memory.Upload<uint>(Enumerable.Range(0, 24).Select(i => 0x1000u + (uint)i).ToArray(), 16);
        var destination = rig.Memory.Allocate(32, 256);
        var tape = rig.Dispatch(new ScenePm4(), ComputeScenePrograms.ScalarLoadImmediateOffset,
            [Descriptors.Lo(table), Descriptors.Hi(table), 0, 0, .. Descriptors.Buffer(destination, 4, 8, format: 20)], 1);
        rig.Submit(tape); rig.Submit(tape);
        Assert.Equal(new uint[] { 0x1004, 0x1005, 0x1006, 0x1007 }, rig.Read<uint>(destination, 4));
        rig.Memory.Write<uint>(table + 16, new uint[] { 19, 23, 29, 31 });
        rig.Submit(tape);
        Assert.Equal(new uint[] { 19, 23, 29, 31 }, rig.Read<uint>(destination, 4));
        rig.Host.RecipeReplayEnabled = false;
        rig.Submit(tape);
        Assert.Equal(new uint[] { 19, 23, 29, 31 }, rig.Read<uint>(destination, 4));
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void RelocationBeyondTheSnapshotMemoStillReusesSelectionAndWritesEveryCurrentDestination()
    {
        using var rig = CreateRig(); if (rig is null) return;
        rig.Host.RecipeReplayEnabled = true;
        var destinations = new List<ulong>();
        for (var i = 0; i < 40; i++)
        {
            var address = rig.Memory.Allocate(64 * 4, 256); destinations.Add(address);
            var tape = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(address, 4, 64, format: 20), 1);
            rig.Submit(tape);
        }
        Assert.True(rig.Host.ReplayCounters.Issues >= 35, $"Relocations did not reuse preparation: {rig.Host.ReplayCounters}");
        foreach (var address in destinations)
            Assert.Equal(Enumerable.Range(0, 64).Select(i => (uint)i ^ 0x5A5Au), rig.Read<uint>(address, 64));
        Assert.Empty(rig.Fatals);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReusedDrawAndDispatchDecisionsMatchReferenceCommandsAndDeviceOutput(bool split)
    {
        using var rig = CreateRig(split); if (rig is null) return;
        rig.Host.RecipeReplayEnabled = true;
        const uint size = 32;
        var target = rig.ColorTarget(size, size);
        var state = new SceneState(size, size).ColorTarget(0, target, size, size);
        var buffer = rig.Memory.Allocate(64 * 4, 256);
        var tape = rig.DrawMesh(new ScenePm4(), state, SceneDraws.Quad(0.5f, 0.25f, 0.5f, 1));
        rig.Dispatch(tape, ScenePrograms.StoreGlobalId, Descriptors.Buffer(buffer, 4, 64, format: 20), 1);
        rig.Submit(tape); // Warm translation, pipelines and acquired resources.
        for (var i = 0; i < 3; i++) rig.Submit(tape);
        MetalRecipeCommands replay = null!;
        rig.Host.RunOnIssueConsumer(() => replay = rig.Host.CaptureReplayCommands());
        var replayCommands = replay.Commands.TakeLast(2).ToArray();
        var replayArguments = CaptureArguments(rig.Host, replay, replayCommands);
        Assert.Equal(2, replayCommands.Length);
        Assert.Equal(MetalRecipeOperation.Draw, replayCommands[0].Operation);
        Assert.Equal(MetalRecipeOperation.Dispatch, replayCommands[1].Operation);
        Assert.True(rig.Host.ReplayCounters.Stages > 0, $"No stage replay: {rig.Host.ReplayCounters}");
        Assert.True(rig.Host.ReplayCounters.Issues > 0, $"No complete issued replay: {rig.Host.ReplayCounters}");
        Assert.True(rig.Host.ReplayCounters.Arguments > 0, $"No argument replay: {rig.Host.ReplayCounters}");
        var replayPixels = rig.ReadImage(target, size, size);
        var replayCompute = rig.Read<uint>(buffer, 64);

        rig.Host.CaptureReferenceCommands = true;
        rig.Host.RecipeReplayEnabled = false;
        rig.Submit(tape);
        MetalRecipeCommands reference = null!;
        rig.Host.RunOnIssueConsumer(() => reference = rig.Host.CaptureReplayCommands());
        var referenceCommands = reference.Commands.TakeLast(2).ToArray();
        var referenceArguments = CaptureArguments(rig.Host, reference, referenceCommands);
        Assert.Equal(replayArguments.Select(bytes => Convert.ToHexString(bytes)), referenceArguments.Select(bytes => Convert.ToHexString(bytes)));
        Assert.Equal(replayCommands.Select(Shape), referenceCommands.Select(Shape));
        for (var i = 0; i < 2; i++)
        {
            var a = replay.Bindings.Skip(replayCommands[i].FirstBinding).Take(replayCommands[i].BindingCount);
            var b = reference.Bindings.Skip(referenceCommands[i].FirstBinding).Take(referenceCommands[i].BindingCount);
            Assert.Equal(a.Select(binding => (binding.Stage, binding.Index)), b.Select(binding => (binding.Stage, binding.Index)));
            // Vertex windows are the same actual no-copy resource. Argument slots
            // legitimately use different serial-safe storage in the reference path.
            Assert.Equal(a.Where(binding => binding.Index >= 2), b.Where(binding => binding.Index >= 2));
        }
        Assert.Equal(replayPixels, rig.ReadImage(target, size, size));
        Assert.Equal(replayCompute, rig.Read<uint>(buffer, 64));
        Assert.Equal(0x5A5Au, replayCompute[0]);
        Assert.Equal(63u ^ 0x5A5Au, replayCompute[63]);
        Assert.Empty(rig.Fatals);
    }

    private static object Shape(MetalRecipeCommand command) =>
        (command.Compute, command.Pipeline, command.Operation, command.A, command.B, command.C, command.D, command.E, command.IndexType);

    // Compare argument fields and their actual pointer targets, rather than
    // declaring two differently allocated native buffers equivalent by name.
    private static unsafe byte[][] CaptureArguments(MetalRenderHost host, MetalRecipeCommands sequence, MetalRecipeCommand[] commands)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var stages = (MetalRenderHost.MetalStageBindings[])typeof(MetalRenderHost).GetField("_stageBindings", flags)!.GetValue(host)!;
        var result = new List<byte[]>();
        foreach (var command in commands)
        foreach (var binding in sequence.Bindings.Skip(command.FirstBinding).Take(command.BindingCount))
        {
            if (binding.Index >= 2) continue;
            var contents = (byte*)MetalNative.Send(binding.Buffer, MetalNative.Selector("contents")) + binding.Offset;
            if (binding.Index == 1)
            {
                result.Add(new ReadOnlySpan<byte>(contents, (int)PushData.ByteSize).ToArray());
                continue;
            }
            var stage = stages[binding.Stage];
            var layout = Gen5MslArgumentLayout.Build(stage.Program.Bindings!);
            var argument = new ReadOnlySpan<byte>(contents, (int)layout.ByteSize).ToArray();
            foreach (var field in layout.Fields.Where(field => field.FieldKind == MslArgumentFieldKind.Pointer))
            {
                var size = field.Kind switch
                {
                    DescriptorBindingKind.ShaderData => (int)stage.Program.Bindings!.ShaderDataDwordCount * 4,
                    DescriptorBindingKind.FlattenedResourceTable => stage.StageResources.Resources.FlattenedResourceTable.Length * 4,
                    DescriptorBindingKind.DeviceAddressPageTable => TableBytes(host, flags),
                    _ => 0,
                };
                if (size == 0) continue; // Stable process-owned pointers compare verbatim.
                var pointer = BitConverter.ToUInt64(argument, (int)field.ByteOffset);
                result.Add(ReadGpuPointer(host, pointer, size, flags));
                argument.AsSpan((int)field.ByteOffset, sizeof(ulong)).Clear();
            }
            result.Add(argument);
        }
        return result.ToArray();
    }
    private static int TableBytes(MetalRenderHost host, BindingFlags flags)
    {
        var table = typeof(MetalRenderHost).GetField("_deviceAddressTableBuilder", flags)!.GetValue(host)!;
        return Convert.ToInt32(table.GetType().GetProperty("ByteSize")!.GetValue(table));
    }

    private static unsafe byte[] ReadGpuPointer(MetalRenderHost host, ulong pointer, int size, BindingFlags flags)
    {
        // GPU virtual addresses need not equal CPU contents pointers. Resolve the
        // address against the actual live test buffers before dereferencing it.
        var buffers = new HashSet<nint>();
        var memory = (MetalGuestMemory)typeof(MetalRenderHost).GetField("_guestMemory", flags)!.GetValue(host)!;
        foreach (var buffer in memory.ResidentBuffers) buffers.Add(buffer);
        foreach (var ring in (System.Collections.IEnumerable)typeof(MetalGuestMemory).GetField("_rings", flags)!.GetValue(memory)!)
            buffers.Add((nint)ring.GetType().GetField("Buffer")!.GetValue(ring)!);
        var arena = typeof(MetalRenderHost).GetField("_replayArena", flags)!.GetValue(host);
        if (arena is not null)
            foreach (var name in new[] { "_slabs", "_retired" })
                foreach (var slab in (IEnumerable<MetalRecipeArena.Slab>)typeof(MetalRecipeArena).GetField(name, flags)!.GetValue(arena)!)
                    buffers.Add(slab.Buffer);
        foreach (var buffer in buffers)
        {
            var address = MetalNative.SendGpuAddress(buffer, MetalNative.Selector("gpuAddress"));
            var length = (ulong)MetalNative.Send(buffer, MetalNative.Selector("length"));
            if (pointer < address || pointer - address > length || (ulong)size > length - (pointer - address)) continue;
            var contents = MetalNative.Send(buffer, MetalNative.Selector("contents"));
            Assert.NotEqual(0, contents);
            return new ReadOnlySpan<byte>((byte*)contents + (pointer - address), size).ToArray();
        }
        throw new InvalidOperationException($"No live test buffer contains GPU pointer 0x{pointer:X}+{size}.");
    }
}
