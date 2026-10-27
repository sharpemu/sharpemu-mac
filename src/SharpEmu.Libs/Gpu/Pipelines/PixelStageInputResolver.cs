// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The pixel stage's static inputs: interpolators, system inputs and the color targets it exports to.
public static class PixelStageInputResolver
{
    private const uint InputPerspectiveSample = 0x0001;
    private const uint InputPerspectiveCenter = 0x0002;
    private const uint InputPerspectiveCentroid = 0x0004;
    private const uint InputPerspectivePull = 0x0008;
    private const uint InputLinearSample = 0x0010;
    private const uint InputLinearCenter = 0x0020;
    private const uint InputLinearCentroid = 0x0040;
    private const uint InputLineStipple = 0x0080;
    private const uint InputPositionX = 0x0100;
    private const uint InputPositionY = 0x0200;
    private const uint InputPositionZ = 0x0400;
    private const uint InputPositionW = 0x0800;
    private const uint InputFrontFace = 0x1000;
    private const uint InputAncillary = 0x2000;
    private const uint InputSampleCoverage = 0x4000;
    private const uint InputPositionFixedPoint = 0x8000;
    private const uint SupportedInputBits =
        InputPerspectiveSample | InputPerspectiveCenter | InputPerspectiveCentroid | InputPerspectivePull |
        InputLinearSample | InputLinearCenter | InputLinearCentroid | InputLineStipple |
        InputPositionX | InputPositionY | InputPositionZ | InputPositionW | InputFrontFace |
        InputAncillary | InputSampleCoverage | InputPositionFixedPoint;

    // The first register after the interpolation inputs the hardware fills.
    public static uint CountSystemInputRegisters(ShaderInterfaceRegisters shaderInterface, ulong shaderAddress)
    {
        var enable = shaderInterface.PixelInputEnable;
        var address = shaderInterface.PixelInputAddress;
        if ((enable & ~SupportedInputBits) != 0 || (address & ~SupportedInputBits) != 0 || enable != address)
        {
            throw GpuFatal.Fatal(
                $"The pixel input registers are not supported: shader=0x{shaderAddress:X16} enable=0x{enable:X8} address=0x{address:X8}.");
        }

        var registers = 0u;
        if ((address & InputPerspectiveSample) != 0) registers += 2;
        if ((address & InputPerspectiveCenter) != 0) registers += 2;
        if ((address & InputPerspectiveCentroid) != 0) registers += 2;
        if ((address & InputPerspectivePull) != 0) registers += 3;
        if ((address & InputLinearSample) != 0) registers += 2;
        if ((address & InputLinearCenter) != 0) registers += 2;
        if ((address & InputLinearCentroid) != 0) registers += 2;
        if ((address & InputLineStipple) != 0) registers += 1;
        return registers;
    }

    // The semantics a pixel program's header declares are static, but the guest owns the table, so every
    // draw still reads it. One read for all of them: this was one guest read per input (up to 32), each
    // taking the memory read lock, to decode a few words.
    private static uint CustomInterpolationMask(CpuContext context, RegisteredShader shader, uint semanticCount)
    {
        if (semanticCount == 0)
        {
            return 0;
        }

        Span<uint> words = stackalloc uint[PixelInputInfo.InterpolatorCount];
        words = words[..(int)semanticCount];
        if (!context.Memory.TryRead(shader.InputSemanticsAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes(words)))
        {
            // A word is unmapped: read them one at a time to name it.
            for (var index = 0u; index < semanticCount; index++)
            {
                if (!context.TryReadUInt32(shader.InputSemanticsAddress + index * sizeof(uint), out words[(int)index]))
                {
                    throw GpuFatal.Fatal($"The pixel input semantics are unreadable: shader=0x{shader.CodeAddress:X16} index={index}.");
                }
            }
        }

        var customMask = 0u;
        for (var index = 0; index < words.Length; index++)
        {
            var semantic = new ShaderInputSemantic(words[index]);
            if (semantic.IsCustom && semantic.IsHalfFloat == 0)
            {
                customMask |= 1u << index;
            }
        }

        return customMask;
    }

    public static PixelInputInfo Resolve(
        CpuContext context,
        RegisteredShader shader,
        ShaderInterfaceRegisters shaderInterface,
        ReadOnlySpan<byte> targetOutputModes,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        uint inputCount,
        bool depthAspectBound,
        bool stencilAspectBound = false,
        PixelInputArrays? reuse = null)
    {
        var activeInputs = shaderInterface.PixelInputEnable & shaderInterface.PixelInputAddress;
        var semanticCount = Math.Min(Math.Min(shader.InputSemanticsCount, inputCount), (uint)PixelInputInfo.InterpolatorCount);
        var customMask = CustomInterpolationMask(context, shader, semanticCount);

        Span<uint> interpolators = stackalloc uint[PixelInputInfo.InterpolatorCount];
        for (var index = 0u; index < inputCount; index++)
        {
            // An unwritten slot maps attribute n to parameter n.
            interpolators[(int)index] = (shaderInterface.PixelInterpolatorWritten & (1u << (int)index)) != 0
                ? shaderInterface.PixelInterpolatorSettings[index]
                : index;
        }

        Span<byte> outputModes = stackalloc byte[PixelInputInfo.TargetCount];
        Span<ColorComponentMap> mappings = stackalloc ColorComponentMap[PixelInputInfo.TargetCount];
        for (var index = 0; index < PixelInputInfo.TargetCount; index++)
        {
            outputModes[index] = targetOutputModes[index];
            mappings[index] = outputModes[index] != 0 ? targetExportMapping[index] : default;
        }

        reuse ??= new PixelInputArrays();
        var control = shaderInterface.DepthShaderControl;
        return new PixelInputInfo
        {
            InputCount = inputCount,
            SystemInputBase = CountSystemInputRegisters(shaderInterface, shader.CodeAddress),
            CustomInterpolationMask = customMask,
            PerspectiveCenterRegister = (activeInputs & InputPerspectiveCenter) != 0
                ? ((activeInputs & InputPerspectiveSample) != 0 ? 2u : 0u)
                : PixelInputInfo.NoPerspectiveCenterRegister,
            InterpolatorSettings = PixelInputArrays.Share(interpolators, ref reuse.Interpolators),
            TargetOutputModes = PixelInputArrays.Share(outputModes, ref reuse.OutputModes),
            TargetExportMappings = PixelInputArrays.Share(mappings, ref reuse.ExportMappings),
            TargetExportFormats = PixelInputArrays.Share<byte>(shaderInterface.TargetOutputModes, ref reuse.ExportFormats),
            ScratchDwords = shader.ScratchDwords,
            PositionX = (activeInputs & InputPositionX) != 0,
            PositionY = (activeInputs & InputPositionY) != 0,
            PositionZ = (activeInputs & InputPositionZ) != 0,
            PositionW = (activeInputs & InputPositionW) != 0,
            FrontFace = (activeInputs & InputFrontFace) != 0,
            FrontFaceAllBits = ((shaderInterface.BarycentricControl >> 24) & 1u) != 0,
            SampleShading = (activeInputs & (InputPerspectiveSample | InputLinearSample)) != 0,
            NoPerspective = (activeInputs & InputLinearCenter) != 0,
            KillEnable = control.KillEnable,
            // Without a depth aspect the DB drops the exported depth, and Metal rejects a
            // fragment function that writes depth into a pipeline without one.
            DepthExportEnable = control.DepthExportEnable && depthAspectBound,
            // Likewise Metal rejects a stencil-writing fragment function without a stencil
            // attachment.
            StencilExportEnable = control.StencilExportEnable && stencilAspectBound,
            SampleMaskExportEnable = control.MaskExportEnable,
            DepthExportFormat = shaderInterface.DepthExportFormat is var zFormat and > 0 and <= 9
                ? (SharpEmu.ShaderCompiler.Gen5ColorExportFormat)zFormat
                : null,
            EarlyDepth = TestsDepthBeforeShader(control),
            ExecuteOnNoop = control.ExecuteOnNoop,
        };
    }

    // Whether the pixel function may declare [[early_fragment_tests]]: the guest tests depth before
    // the shader AND skips the shader for a fragment that test rejects. That only changes what a
    // shader with memory writes does (Metal already tests early for one without), and then it
    // matters: the stores of a rejected fragment must happen exactly when the guest's do.
    // - DEPTH_BEFORE_SHADER forces the test ahead of the shader whatever the other bits say
    //   (Mesa RADV sets it with [earlydepthstencil] next to EXEC_ON_* = writes_memory).
    // - Z_ORDER 1..3 (EARLY_Z_THEN_LATE_Z, RE_Z, EARLY_Z_THEN_RE_Z) test early too, but EXEC_ON_HIER_FAIL
    //   and EXEC_ON_NOOP make the shader run for a rejected fragment, and then its stores happen:
    //   those shaders stay on Metal's late order.
    // - A shader that kills or exports depth, stencil or coverage is decided after it ran, and Metal
    //   rejects depth or stencil outputs with the attribute.
    internal static bool TestsDepthBeforeShader(in DepthShaderControlRegisters control) =>
        !control.KillEnable && !control.DepthExportEnable && !control.StencilExportEnable && !control.MaskExportEnable &&
        (control.DepthBeforeShader ||
         (control.DepthExportOrder != 0 && !control.ExecuteOnHierFail && !control.ExecuteOnNoop));
}

// The arrays of the last pixel input resolved. Nothing writes a resolved input's arrays, and
// consecutive draws of a pass resolve the same interpolators and targets: a draw whose arrays
// come out equal shares the last ones instead of allocating its own. Used under one lock.
public sealed class PixelInputArrays
{
    internal uint[]? Interpolators;
    internal byte[]? OutputModes;
    internal byte[]? ExportFormats;
    internal ColorComponentMap[]? ExportMappings;

    internal static T[] Share<T>(ReadOnlySpan<T> values, ref T[]? last) where T : IEquatable<T> =>
        last is not null && values.SequenceEqual(last) ? last : last = values.ToArray();
}
