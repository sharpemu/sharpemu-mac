// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

[Collection(SchedulingStateCollection.Name)]
public sealed class DrawPathReadTests
{
    private sealed class Memory(bool scalarOnly) : ICpuMemory
    {
        public readonly FakeCpuMemory Inner = new(0x1000, 0x1000);
        public bool Readable = true;
        public bool TryRead(ulong address, Span<byte> bytes) => Readable && (!scalarOnly || bytes.Length <= 4) && Inner.TryRead(address, bytes);
        public bool TryWrite(ulong address, ReadOnlySpan<byte> bytes) => Inner.TryWrite(address, bytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DrawPathPixelSemanticsStayFreshAndFallBackToScalarReads(bool scalarOnly)
    {
        using var fatal = new FatalScope();
        var memory = new Memory(scalarOnly);
        var context = new CpuContext(memory, Generation.Gen5);
        var shader = new RegisteredShader(0x2000, 0x2000, 4, 0, 0, 0x1000, 2, 0, 0);
        PixelInputInfo Resolve() => PixelStageInputResolver.Resolve(context, shader, new ShaderInterfaceRegisters(),
            new byte[8], new ColorComponentMap[8], 2, false);
        Assert.True(memory.TryWrite(0x1000, new byte[] { 0, 0, 0, 1, 0, 0, 0, 0 }));
        var first = Resolve();
        Assert.Equal(1u, first.CustomInterpolationMask);
        Assert.True(memory.TryWrite(0x1000, new byte[] { 0, 0, 0, 0, 0, 0, 0, 1 }));
        Assert.Equal(2u, Resolve().CustomInterpolationMask);
        Assert.Equal(1u, first.CustomInterpolationMask);
        memory.Readable = false;
        Assert.Throws<SchedulerFatalException>(() => Resolve());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DrawPathVertexDescriptorsFallBackToScalarReads(bool scalarOnly)
    {
        using var fatal = new FatalScope();
        var memory = new Memory(scalarOnly);
        var context = new CpuContext(memory, Generation.Gen5);
        Assert.True(memory.TryWrite(0x1000, new byte[4]));
        // Attribute table at 0x1000, one 16-byte descriptor at 0x1100.
        Assert.True(memory.TryWrite(0x1100, new byte[] { 0, 0x18, 0, 0, 0, 0, 16, 0, 4, 0, 0, 0, 0xAC, 0x0F, 0, 0 }));
        var resources = VertexInputResolver.ApplySemantics(context, new uint[] { 4u << 16 }, 0x1000, 0x1100, 0x2000);
        Assert.Equal(0x1800ul, Assert.Single(resources).Descriptor.Address);
        Assert.True(memory.TryWrite(0x1100, new byte[] { 0, 0x19, 0, 0 }));
        Assert.Equal(0x1900ul, Assert.Single(VertexInputResolver.ApplySemantics(context,
            new uint[] { 4u << 16 }, 0x1000, 0x1100, 0x2000)).Descriptor.Address);
        memory.Readable = false;
        Assert.Throws<SchedulerFatalException>(() => VertexInputResolver.ApplySemantics(context,
            new uint[] { 4u << 16 }, 0x1000, 0x1100, 0x2000));
    }
}
