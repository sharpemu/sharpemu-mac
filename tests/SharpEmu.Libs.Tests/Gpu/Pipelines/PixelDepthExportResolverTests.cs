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

// Z_EXPORT_ENABLE reaches the pixel program only when a depth aspect is bound: the DB
// drops the export otherwise, and Metal rejects a depth-writing fragment function in a
// pipeline without a depth attachment.
[Collection(SchedulingStateCollection.Name)]
public sealed class PixelDepthExportResolverTests : IDisposable
{
    private const ulong MemoryBase = 0x1_0000_0000;

    private readonly FatalScope _fatal = new();
    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x1000);

    [Theory]
    [InlineData(true, true, true)]
    [InlineData(true, false, false)]
    [InlineData(false, true, false)]
    public void DepthExportFollowsTheBoundDepthAspect(bool zExportEnable, bool depthAspectBound, bool expected)
    {
        var shaderInterface = new ShaderInterfaceRegisters
        {
            DepthShaderControl = DepthShaderControlRegisters.Decode(zExportEnable ? 1u : 0u),
        };
        var info = PixelStageInputResolver.Resolve(
            new CpuContext(_memory, Generation.Gen5),
            new RegisteredShader(MemoryBase, MemoryBase, 16, 0, 0, 0, 0, 0, 0),
            shaderInterface,
            new byte[PixelInputInfo.TargetCount],
            new ColorComponentMap[PixelInputInfo.TargetCount],
            inputCount: 0,
            depthAspectBound);

        Assert.Equal(expected, info.DepthExportEnable);
    }

    // STENCIL_TEST_VAL_EXPORT_ENABLE (bit 1) needs a bound stencil aspect the same way, and
    // SPI_SHADER_Z_FORMAT reaches the program when the registers name one.
    [Theory]
    [InlineData(0x2u, true, 7u, true, SharpEmu.ShaderCompiler.Gen5ColorExportFormat.Uint16Abgr)]
    [InlineData(0x2u, false, 2u, false, SharpEmu.ShaderCompiler.Gen5ColorExportFormat.GR32)]
    [InlineData(0x0u, true, 0u, false, null)]
    public void StencilExportFollowsTheBoundStencilAspect(
        uint control, bool stencilAspectBound, uint zFormat, bool expected, SharpEmu.ShaderCompiler.Gen5ColorExportFormat? format)
    {
        var shaderInterface = new ShaderInterfaceRegisters
        {
            DepthShaderControl = DepthShaderControlRegisters.Decode(control),
            DepthExportFormat = zFormat,
        };
        var info = PixelStageInputResolver.Resolve(
            new CpuContext(_memory, Generation.Gen5),
            new RegisteredShader(MemoryBase, MemoryBase, 16, 0, 0, 0, 0, 0, 0),
            shaderInterface,
            new byte[PixelInputInfo.TargetCount],
            new ColorComponentMap[PixelInputInfo.TargetCount],
            inputCount: 0,
            depthAspectBound: true,
            stencilAspectBound);

        Assert.Equal(expected, info.StencilExportEnable);
        Assert.Equal(format, info.DepthExportFormat);
        Assert.False(info.EarlyDepth && control != 0);
    }

    // DB_SHADER_CONTROL: Z_ORDER bits 4-5, KILL 6, MASK_EXPORT 8, EXEC_ON_HIER_FAIL 9, EXEC_ON_NOOP 10,
    // DEPTH_BEFORE_SHADER 12. Metal tests depth early only where the guest does and skips the shader for
    // the fragments it rejects: EXEC_ON_* make that shader run for them, so those stay late.
    [Theory]
    [InlineData(0x000u, false)]            // LATE_Z
    [InlineData(0x010u, true)]             // EARLY_Z_THEN_LATE_Z
    [InlineData(0x020u, true)]             // RE_Z
    [InlineData(0x030u, true)]             // EARLY_Z_THEN_RE_Z
    [InlineData(0x210u, false)]            // + EXEC_ON_HIER_FAIL
    [InlineData(0x410u, false)]            // + EXEC_ON_NOOP
    [InlineData(0x620u, false)]            // RE_Z + both
    [InlineData(0x1000u, true)]            // DEPTH_BEFORE_SHADER alone
    [InlineData(0x1630u, true)]            // DEPTH_BEFORE_SHADER wins over EXEC_ON_*
    [InlineData(0x050u, false)]            // KILL_ENABLE
    [InlineData(0x1050u, false)]           // KILL_ENABLE, even with DEPTH_BEFORE_SHADER
    [InlineData(0x011u, false)]            // Z_EXPORT_ENABLE
    [InlineData(0x012u, false)]            // STENCIL_TEST_VAL_EXPORT_ENABLE
    [InlineData(0x110u, false)]            // MASK_EXPORT_ENABLE
    [InlineData(0x8000u, false)]           // DUAL_QUAD_DISABLE and friends do not matter
    [InlineData(0x8010u, true)]
    public void EarlyDepthFollowsTheGuestsDepthOrderAndExecuteBits(uint control, bool early)
    {
        var shaderInterface = new ShaderInterfaceRegisters { DepthShaderControl = DepthShaderControlRegisters.Decode(control) };
        var info = PixelStageInputResolver.Resolve(
            new CpuContext(_memory, Generation.Gen5),
            new RegisteredShader(MemoryBase, MemoryBase, 16, 0, 0, 0, 0, 0, 0),
            shaderInterface,
            new byte[PixelInputInfo.TargetCount],
            new ColorComponentMap[PixelInputInfo.TargetCount],
            inputCount: 0,
            depthAspectBound: true,
            stencilAspectBound: true);

        Assert.Equal(early, info.EarlyDepth);
    }

    [Fact]
    public void TheDecodedBitsMatchTheHardwareRegister()
    {
        var control = DepthShaderControlRegisters.Decode((1u << 9) | (1u << 12));
        Assert.True(control.ExecuteOnHierFail);
        Assert.True(control.DepthBeforeShader);
        Assert.False(control.ExecuteOnNoop);
        Assert.Equal(0u, control.RemainingBits);
        Assert.True(DepthShaderControlRegisters.Decode(1u << 10).ExecuteOnNoop);
    }

    public void Dispose() => _fatal.Dispose();
}
