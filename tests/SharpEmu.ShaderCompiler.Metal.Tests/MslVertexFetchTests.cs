// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslVertexFetchTests
{
    public static TheoryData<uint, uint> Formats => new()
    {
        { 1, 47 }, { 2, 48 }, { 5, 45 }, { 6, 46 },
        { 7, 51 }, { 8, 52 }, { 11, 49 }, { 12, 50 }, { 13, 53 },
        { 14, 7 }, { 15, 10 }, { 18, 1 }, { 19, 4 },
        { 20, 36 }, { 21, 32 }, { 22, 28 },
        { 23, 19 }, { 24, 22 }, { 27, 13 }, { 28, 16 }, { 29, 25 },
        { 36, 54 }, { 50, 41 }, { 51, 40 },
        { 56, 9 }, { 57, 12 }, { 60, 3 }, { 61, 6 },
        { 62, 37 }, { 63, 33 }, { 64, 29 },
        { 65, 21 }, { 66, 24 }, { 69, 15 }, { 70, 18 }, { 71, 27 },
        { 72, 38 }, { 73, 34 }, { 74, 30 }, { 75, 39 }, { 76, 35 }, { 77, 31 },
    };

    [Theory]
    [MemberData(nameof(Formats))]
    public void NativeVertexFetchMatchesShaderFetchBits(uint format, uint nativeFormat)
    {
        if (!MetalNative.IsAvailable) return;
        foreach (var (stride, instance, divisor, indexed, swizzle) in new[]
        {
            (64u, false, 1u, false, 0xFACu),
            (20u, false, 1u, true, 0xFACu),
            (64u, true, 1u, true, 0xFACu),
            (64u, true, 2u, false, 0xFACu),
            (0u, false, 1u, true, 0xFACu),
            (64u, false, 1u, true, 7u | (4u << 3) | (1u << 6)),
        })
        {
            var bytes = new byte[1024];
            new Random(7321).NextBytes(bytes);
            // Finite values, signed zero, denormals and integer edge bits in every width.
            uint[] edges = [0, 0x80000000, 0xFFFFFFFF, 0x007FFFFF, 0x3F800000, 0xFFFF8000, 0x7F7F7F7F, 0x01234567];
            for (var i = 0; i < edges.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 4), edges[i]);
            var portable = Compile(format, stride, instance, divisor, swizzle, fixedFetch: false);
            var native = Compile(format, stride, instance, divisor, swizzle, fixedFetch: true);
            var baseInstance = divisor == 1 ? 3u : 0u;
            Assert.True(MetalNative.TryCaptureVertexFetch(portable, bytes, 0, stride, instance, divisor, 2, baseInstance, indexed,
                out var shaderResult, out var error), error);
            Assert.True(MetalNative.TryCaptureVertexFetch(native, bytes, nativeFormat, stride, instance, divisor, 2, baseInstance, indexed,
                out var nativeResult, out error), error);
            Assert.True(shaderResult.AsSpan().SequenceEqual(nativeResult),
                $"format={format} stride={stride} instance={instance} divisor={divisor} indexed={indexed} swizzle=0x{swizzle:X}: " +
                string.Join(", ", Enumerable.Range(0, shaderResult.Length / 4)
                    .Where(i => BitConverter.ToUInt32(shaderResult, i * 4) != BitConverter.ToUInt32(nativeResult, i * 4))
                    .Take(8).Select(i => $"[{i}]={BitConverter.ToUInt32(shaderResult, i * 4):X8}/{BitConverter.ToUInt32(nativeResult, i * 4):X8}")));
        }
    }

    private static Gen5MslShader Compile(uint format, uint stride, bool instance, uint divisor, uint swizzle, bool fixedFetch)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources, Gen5Operand[] destinations, Gen5InstructionControl? control = null)
        {
            instructions.Add(new(pc, encoding, opcode, [0u, 0u], sources, destinations, control));
            pc += 8;
        }
        uint[] descriptor = [0x200000, stride << 16, stride == 0 ? 1024u : 16u, (format << 12) | swizzle | (2u << 28)];
        for (uint i = 0; i < 4; i++)
            Add("SMovB32", Gen5ShaderEncoding.Sop1, [new(Gen5OperandKind.LiteralConstant, descriptor[i])], [Gen5Operand.Scalar(8 + i)]);
        var address = instance ? 8u : 5u;
        if (divisor != 1)
        {
            Add("VLshrrevB32", Gen5ShaderEncoding.Vop2, [new(Gen5OperandKind.LiteralConstant, 1), Gen5Operand.Vector(address)], [Gen5Operand.Vector(9)]);
            address = 9;
        }
        var fetchPc = pc;
        Add("BufferLoadFormatXyzw", Gen5ShaderEncoding.Mubuf,
            [Gen5Operand.Vector(address), Gen5Operand.Scalar(8), new(Gen5OperandKind.EncodedConstant, 128)],
            Enumerable.Range(10, 4).Select(i => Gen5Operand.Vector((uint)i)).ToArray(),
            new Gen5BufferMemoryControl(4, address, 10, 8, 0, true, false, false, false));
        Add("Exp", Gen5ShaderEncoding.Exp, Enumerable.Range(10, 4).Select(i => Gen5Operand.Vector((uint)i)).ToArray(), [],
            new Gen5ExportControl(32, 15, false, true, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], []);
        var program = new Gen5ShaderProgram(0x100000, instructions);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Vertex, 1, 0, 0,
            fixedFunctionVertexLoads: fixedFetch ? new HashSet<uint> { fetchPc } : null);
        var specialization = ResourceSpecialization.Default(plan.Info);
        if (!fixedFetch) specialization.Buffers[0] = new(stride, format, swizzle);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, [], false, false, false, 0, portableBuffers: false);
        Gfx10UnifiedFormat.TryDecode(format, out _, out var number);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            WaveSize = 32,
            RequiredVertexOutputCount = 1,
            VertexInputs = fixedFetch ? [new ShaderVertexInput(fetchPc, 0, 4, 4, number, swizzle, instance, [])] : [],
        };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }
}
