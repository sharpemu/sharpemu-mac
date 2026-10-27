// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

// A retained snapshot is a candidate: the draw uses it only when the state it was captured
// under is the state the interpreter reached, which the prepass cannot know in advance.
[Collection(AgcCommandBufferChainCollection.Name)]
public sealed class AgcGeometrySnapshotValidationTests
{
    private const ulong BaseAddress = 0x1_0000_0000;
    private const int MemorySize = 0x1_0000;
    private const ulong CommandAddress = BaseAddress + 0x1000;
    private const ulong PredicateAddress = BaseAddress + 0x8000;
    private const ulong IndexAddress = BaseAddress + 0x9000;
    private const ulong PatchedIndexAddress = BaseAddress + 0xA000;

    private const uint ItConditionalExecute = 0x22;
    private const uint ItDrawIndex2 = 0x27;
    private const uint ItIndexType = 0x2A;
    private const uint ItWriteData = 0x37;
    private const uint ItSetShaderRegister = 0x76;
    private const uint UserDataRegister = 0x8C;
    private const uint UserDataValue = 0x1234;
    private const uint IndexCount = 6;

    private static uint[] SetShaderRegister(uint offset, uint value) => [Pm4Header(3, ItSetShaderRegister), offset, value];

    private static uint[] IndexType(uint type) => [Pm4Header(2, ItIndexType), type];

    private static uint[] DrawIndex2(ulong indexAddress) =>
        [Pm4Header(6, ItDrawIndex2), IndexCount, unchecked((uint)indexAddress), (uint)(indexAddress >> 32), IndexCount, 0];

    private static uint[] ConditionalExecute(ulong predicateAddress, uint skippedDwords) =>
        [Pm4Header(5, ItConditionalExecute), unchecked((uint)predicateAddress), (uint)(predicateAddress >> 32), 0, skippedDwords];

    private static uint[] WriteData(ulong destination, uint value) =>
        [Pm4Header(5, ItWriteData), 2u << 8, unchecked((uint)destination), (uint)(destination >> 32), value];

    private static uint Pm4Header(uint dwords, uint opcode) =>
        0xC000_0000u | ((dwords - 2) << 16) | (opcode << 8);

    private static (FakeCpuMemory Memory, uint Dwords, ulong DrawPacketAddress) WriteStream(params uint[][] packets)
    {
        var memory = new FakeCpuMemory(BaseAddress, MemorySize);
        var words = packets.SelectMany(static packet => packet).ToArray();
        var drawOffset = 0u;
        foreach (var packet in packets)
        {
            if (packet[0] == Pm4Header(6, ItDrawIndex2))
            {
                break;
            }

            drawOffset += (uint)packet.Length;
        }

        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes, words[index]);
            Assert.True(memory.TryWrite(CommandAddress + ((ulong)index * sizeof(uint)), bytes));
        }

        return (memory, (uint)words.Length, CommandAddress + ((ulong)drawOffset * sizeof(uint)));
    }

    private static object CapturedAsThePrepassWould(ulong drawPacketAddress, ulong indexAddress) =>
        AgcExports.CreateGeometrySnapshotsForTests(
            drawPacketAddress,
            indexSize: 0,
            indexAddress,
            IndexCount,
            instanceCount: 1,
            [(UserDataRegister, UserDataValue)]);

    private static void SubmitDriver(FakeCpuMemory memory, uint dwords)
    {
        var context = new CpuContext(memory, Generation.Gen5);
        const ulong submissionAddress = BaseAddress + 0x40;
        context[CpuRegister.Rdi] = submissionAddress;
        Assert.True(context.TryWriteUInt64(submissionAddress, CommandAddress));
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(length, dwords);
        Assert.True(memory.TryWrite(submissionAddress + 8, length));
        Assert.Equal(0, AgcExports.DriverSubmitDcb(context));
    }

    [Fact]
    public void ExecutedShaderRegistersKeepTheInterpreterAliasAcrossClearAndReset()
    {
        var (memory, dwords, _) = WriteStream(SetShaderRegister(UserDataRegister, 0));
        SubmitDriver(memory, dwords);
        var interpreter = AgcExports.GetHeadlessCommandStreamForTests(memory).Queue.GetInterpreter(0);
        var alias = AgcExports.GetExecutedShaderRegistersForTests(memory);
        Assert.Same(interpreter.Registers.Shader, alias);
        Assert.True(alias.TryGetValue(UserDataRegister, out var zero));
        Assert.Equal(0u, zero);
        alias.Clear();
        Assert.Empty(interpreter.Registers.Shader);
        interpreter.Registers.Shader[UserDataRegister] = 9;
        Assert.Equal(9u, alias[UserDataRegister]);
        interpreter.Registers.Reset();
        Assert.Same(alias, interpreter.Registers.Shader);
        Assert.Empty(alias);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CaptureRetainsOverflowShaderKeysEvenWhenExecutionSkipsTheWrite(bool indirect)
    {
        const ulong table = PredicateAddress + 0x100;
        uint[] write = indirect
            ? [Pm4Header(5, PacketOpcode.SetShaderRegisterIndirect), unchecked((uint)table), (uint)(table >> 32), 0, 1]
            : SetShaderRegister(uint.MaxValue, 0);
        var (memory, dwords, _) = WriteStream(
            ConditionalExecute(PredicateAddress, (uint)write.Length), write,
            IndexType(0), DrawIndex2(IndexAddress));
        Span<byte> zero = stackalloc byte[4];
        zero.Clear();
        Assert.True(memory.TryWrite(PredicateAddress, zero));
        Span<byte> entry = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(entry, uint.MaxValue);
        BinaryPrimitives.WriteUInt32LittleEndian(entry[4..], 0);
        Assert.True(memory.TryWrite(table, entry));
        SubmitDriver(memory, dwords);
        var captured = AgcExports.GetGeometryCaptureRegistersForTests(memory);
        var key = indirect ? uint.MaxValue & ~0x7000_0000u : uint.MaxValue;
        Assert.True(captured.TryGetValue(key, out var value));
        Assert.Equal(0u, value);
        var executed = AgcExports.GetExecutedShaderRegistersForTests(memory);
        Assert.False(executed.ContainsKey(key));
    }

    [Fact]
    public void MatchingState_UsesTheSnapshot()
    {
        var (memory, dwords, drawPacket) = WriteStream(
            SetShaderRegister(UserDataRegister, UserDataValue),
            IndexType(0),
            DrawIndex2(IndexAddress));

        AgcExports.GetHeadlessCommandStreamForTests(memory).Submit(0, CommandAddress, dwords, 1, CapturedAsThePrepassWould(drawPacket, IndexAddress));

        Assert.Equal(AgcExports.GeometrySnapshotDecision.Accepted, AgcExports.LastGeometrySnapshotDecisionForTests);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CaptureRegisters_MatchInterpreterAcrossFrameBoundary(bool endFrame)
    {
        var (memory, dwords, _) = WriteStream(
            SetShaderRegister(0x48, 1), SetShaderRegister(0x49, 2),
            SetShaderRegister(0x4A, 3), SetShaderRegister(0x4B, 4));
        var context = new CpuContext(memory, Generation.Gen5);
        const ulong submissionAddress = BaseAddress + 0x40;
        context[CpuRegister.Rdi] = submissionAddress;
        Assert.True(context.TryWriteUInt64(submissionAddress, CommandAddress));
        Span<byte> lengthBytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(lengthBytes, dwords);
        Assert.True(memory.TryWrite(submissionAddress + 8, lengthBytes));
        Assert.Equal(0, AgcExports.DriverSubmitDcb(context));
        if (endFrame)
        {
            Assert.Equal(0, AgcExports.SuspendPoint(context));
        }

        var nextPacket = SetShaderRegister(0xC8, 5);
        for (var index = 0; index < nextPacket.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(lengthBytes, nextPacket[index]);
            Assert.True(memory.TryWrite(CommandAddress + (ulong)(index * 4), lengthBytes));
        }

        BinaryPrimitives.WriteUInt32LittleEndian(lengthBytes, (uint)nextPacket.Length);
        Assert.True(memory.TryWrite(submissionAddress + 8, lengthBytes));
        Assert.Equal(0, AgcExports.DriverSubmitDcb(context));
        var interpreterRegisters = AgcExports.GetHeadlessCommandStreamForTests(memory).Queue.GetInterpreter(0).Registers.Shader;
        var capturedRegisters = AgcExports.GetGeometryCaptureRegistersForTests(memory);
        Assert.Equal(endFrame ? 1 : 5, interpreterRegisters.Count);
        Assert.Equal(interpreterRegisters.OrderBy(static entry => entry.Key), capturedRegisters.OrderBy(static entry => entry.Key));
    }

    // The prepass applied the register write; the interpreter skipped it through the predicate.
    [Fact]
    public void RegisterWriteSkippedByConditionalExecute_RejectsTheSnapshot()
    {
        var (memory, dwords, drawPacket) = WriteStream(
            ConditionalExecute(PredicateAddress, 3),
            SetShaderRegister(UserDataRegister, UserDataValue),
            IndexType(0),
            DrawIndex2(IndexAddress));
        Span<byte> zero = stackalloc byte[sizeof(uint)];
        Assert.True(memory.TryWrite(PredicateAddress, zero));

        AgcExports.GetHeadlessCommandStreamForTests(memory).Submit(0, CommandAddress, dwords, 1, CapturedAsThePrepassWould(drawPacket, IndexAddress));

        Assert.Equal(AgcExports.GeometrySnapshotDecision.Rejected, AgcExports.LastGeometrySnapshotDecisionForTests);
    }

    // The stream patches the draw's index address before the draw executes.
    [Fact]
    public void IndexAddressPatchedAfterCapture_RejectsTheSnapshot()
    {
        var (memory, dwords, drawPacket) = WriteStream(
            SetShaderRegister(UserDataRegister, UserDataValue),
            IndexType(0),
            WriteData(CommandAddress + ((3 + 2 + 5 + 2) * sizeof(uint)), unchecked((uint)PatchedIndexAddress)),
            DrawIndex2(IndexAddress));

        AgcExports.GetHeadlessCommandStreamForTests(memory).Submit(0, CommandAddress, dwords, 1, CapturedAsThePrepassWould(drawPacket, IndexAddress));

        Assert.Equal(AgcExports.GeometrySnapshotDecision.Rejected, AgcExports.LastGeometrySnapshotDecisionForTests);
        Assert.True(AgcExports.TryGetGraphicsIndexStateForTests(new CpuContext(memory, Generation.Gen5), out var address, out _, out _));
        Assert.Equal(PatchedIndexAddress, address);
    }
}
