// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// A compute program whose binding is only ever read by scalar buffer loads gets it in the
// constant address space (Gen5MslCodegenOptions.ScalarConstantComputeBuffers), the way a mesh
// program does: the loads are the same dword reads, and constant is Apple's uniform path.
// Anything that could make the binding change under the program — a store or atomic to a
// buffer, a vector or coherent access of the binding, a dropped scalar cache — keeps device.
public sealed class Gen5MslComputeConstantBufferTests
{
    private const uint NullOperand = 125;
    private const uint ReadDescriptor = 8;
    private const uint WriteDescriptor = 12;
    private const string ConstantBinding = "constant uint* b0 =";
    private const string DeviceBinding = "device uint* b0 =";

    public enum Extra { None, StoreToOtherBuffer, VectorLoadOfSameBinding, ScalarCacheInvalidate, CoherentScalarLoad }

    [Fact]
    public void AScalarOnlyReadOnlyBindingIsConstant()
    {
        var source = Compile(Extra.None);
        Assert.Contains(ConstantBinding, source, StringComparison.Ordinal);
        Assert.DoesNotContain(DeviceBinding, source, StringComparison.Ordinal);
        AssertCompilesOnDevice(source);
    }

    [Theory]
    [InlineData(Extra.StoreToOtherBuffer)]
    [InlineData(Extra.VectorLoadOfSameBinding)]
    [InlineData(Extra.ScalarCacheInvalidate)]
    [InlineData(Extra.CoherentScalarLoad)]
    public void AnythingThatCouldChangeOrBypassTheBindingKeepsDevice(Extra extra)
    {
        var source = Compile(extra);
        Assert.DoesNotContain(ConstantBinding, source, StringComparison.Ordinal);
        Assert.Contains(DeviceBinding, source, StringComparison.Ordinal);
        AssertCompilesOnDevice(source);
    }

    [Fact]
    public void TheComputeSwitchTakesOnlyComputeBackToDevice()
    {
        var off = new Gen5MslCodegenOptions { ScalarConstantComputeBuffers = false };
        Assert.Contains(DeviceBinding, Compile(Extra.None, off), StringComparison.Ordinal);
        var meshOnly = new Gen5MslCodegenOptions { ScalarConstantBuffers = false };
        Assert.Contains(DeviceBinding, Compile(Extra.None, meshOnly), StringComparison.Ordinal);
    }

    [Fact]
    public void RawDwordAccessIsRequired()
    {
        var bytewise = new Gen5MslCodegenOptions(RawDwordAccess: false);
        Assert.DoesNotContain(ConstantBinding, Compile(Extra.None, bytewise), StringComparison.Ordinal);
    }

    [Fact]
    public void EnvironmentSwitchDefaultsOn()
    {
        Assert.Equal(
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_SCALAR_CONSTANT_COMPUTE") != "0",
            Gen5MslCodegenOptions.FromEnvironment.ScalarConstantComputeBuffers);
    }

    private static string Compile(Extra extra, Gen5MslCodegenOptions? options = null)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            ScalarBufferLoad(ReadDescriptor, 32, extra == Extra.CoherentScalarLoad),
        };
        if (extra == Extra.StoreToOtherBuffer)
        {
            instructions.Add(new(0, Gen5ShaderEncoding.Vop1, "VMovB32", [0u], [Gen5Operand.Scalar(32)], [Gen5Operand.Vector(1)], null));
            instructions.Add(new(0, Gen5ShaderEncoding.Mubuf, "BufferStoreDword", [0u, 0u],
                [Gen5Operand.Vector(1), Gen5Operand.Scalar(WriteDescriptor), Gen5Operand.Source(128)], [],
                new Gen5BufferMemoryControl(1, 0, 1, WriteDescriptor, 0, false, false, false, false)));
        }
        else if (extra == Extra.VectorLoadOfSameBinding)
        {
            instructions.Add(new(0, Gen5ShaderEncoding.Mubuf, "BufferLoadDword", [0u, 0u],
                [Gen5Operand.Vector(0), Gen5Operand.Scalar(ReadDescriptor), Gen5Operand.Source(128)], [Gen5Operand.Vector(1)],
                new Gen5BufferMemoryControl(1, 0, 1, ReadDescriptor, 0, false, false, false, false)));
        }
        else if (extra == Extra.ScalarCacheInvalidate)
        {
            instructions.Add(new(0, Gen5ShaderEncoding.Smem, "SDcacheInv", [0u, 0u], [], [], new Gen5ScalarMemoryControl(0, 0, null)));
        }

        instructions.Add(new(0, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null));
        var program = new Gen5ShaderProgram(0x2000, instructions.Select((instruction, index) => instruction with { Pc = (uint)index * 8 }).ToArray());

        var userData = new uint[16];
        foreach (var register in new[] { ReadDescriptor, WriteDescriptor })
        {
            userData[register] = register == ReadDescriptor ? 0x200000u : 0x300000u;
            userData[register + 2] = 0x1000;
            userData[register + 3] = DescriptorConstants.IdentityDestinationSelect;
        }

        var request = Gen5ComputeFixtures.RequestOrThrow(
            program, ShaderStage.Compute, waveLaneCount: 64, localSizeX: 64,
            runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options), error);
        return shader.Source;
    }

    private static Gen5ShaderInstruction ScalarBufferLoad(uint resourceRegister, uint destination, bool coherent) =>
        new(
            0,
            Gen5ShaderEncoding.Smem,
            "SBufferLoadDword",
            [0u, 0u],
            [Gen5Operand.Scalar(resourceRegister), Gen5Operand.Source(NullOperand)],
            [Gen5Operand.Scalar(destination)],
            new Gen5ScalarMemoryControl(1, 16, null, Glc: coherent));

    private static void AssertCompilesOnDevice(string source)
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the compute constant-buffer test without a Metal device.");
            return;
        }

        Assert.True(MetalNative.TryCompileLibrary(source, out _, out var error), error);
    }
}
