// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class Gen5MslComputeBatchTests
{
    [Theory]
    [InlineData(32u)]
    [InlineData(64u)]
    public void BatchEntrySelectsEachRequestsUserDataAndResources(uint wave)
    {
        var fixture = new Gen5ComputeFixture("batch-user-data",
        [
            wave == 64 ? 0x8F118610u : 0x8F118510u, // s_lshl_b32 s17, s16, log2(wave)
            0x4A020011,             // v_add_nc_u32 v1, s17, v0 (global thread)
            0x4A040200,             // v_add_nc_u32 v2, s0, v1
            0x34020282,             // v_lshlrev_b32 v1, 2, v1
            0xE0701000, 0x80020201, // buffer_store_dword v2, v1, s[8:11], 0 offen
            0xBF810000,
        ], 8, 4096);
        var original = Gen5ComputeFixtures.CreateComputeRequest(fixture, wave, wave);
        var plan = ShaderResourcePlan.Extract(original.Program, original.Stage, original.Hash,
            original.UserDataBase, original.UserDataCount);
        var request = new ShaderCompileRequest(plan, original.Resources, original.Bindings)
        {
            WaveSize = wave,
            LocalSizeX = wave,
            ComputeSystemRegisters = new(16, null, null, null),
        };
        Assert.True(request.Bindings.UsesPushData);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error,
            new Gen5MslCodegenOptions(ComputeBatchEntryPoint: true)), error);
        Assert.NotNull(shader.ComputeBatchEntryPoint);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var ordinary, out error), error);
        // The normal entry and all helpers keep their exact generated source.
        Assert.StartsWith(ordinary.Source, shader.Source);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
        Assert.True(MetalNative.IsAvailable);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), error);
        uint[] groups = [1, 3, 2, 7, 1];
        var pushes = new List<byte[]>();
        var arguments = new List<byte[]>();
        var expected = new List<byte[]>();
        for (var index = 0; index < groups.Length; index++)
        {
            var push = new byte[PushData.ByteSize];
            for (var user = 0; user < request.Bindings.UserDataRegisters.Count; user++)
            {
                var register = request.Bindings.UserDataRegisters[user];
                var value = register == 0 ? (uint)(index * 37 + 11) : 0;
                BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan(((int)request.Bindings.PushDataStartDword + user) * 4), value);
            }
            pushes.Add(push);
            arguments.Add(new byte[shader.ArgumentLayout!.ByteSize]);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[groups[index] * wave * 4], push,
                shader.ArgumentLayout, wave, out var bytes, out _, out error, threadgroupCount: groups[index]), error);
            for (var lane = 0; lane < groups[index] * wave; lane++)
                Assert.Equal((uint)(index * 37 + 11 + lane), BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(lane * 4)));
            expected.Add(bytes);
        }
        Assert.True(MetalNative.ExecuteComputeBatch(shader, arguments, pushes, groups,
            groups.Select(count => (int)(count * wave * 4)).ToArray(), out var batched, out error), error);
        for (var index = 0; index < groups.Length; index++) Assert.Equal(expected[index], batched[index]);
    }
}
