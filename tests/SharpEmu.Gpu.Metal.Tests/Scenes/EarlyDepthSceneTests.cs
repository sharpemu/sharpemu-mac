// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// DB_SHADER_CONTROL decides whether a pixel shader with memory writes runs for a fragment the
/// depth test rejects. A fully occluded quad draws with a pixel program that adds one to a
/// buffer counter per invocation; the counter shows how many invocations ran. The rule: the
/// shader is skipped (Metal [[early_fragment_tests]]) when the guest tests depth first without
/// EXEC_ON_HIER_FAIL / EXEC_ON_NOOP, or with DEPTH_BEFORE_SHADER; it runs for every fragment when
/// the order is LATE_Z, when either EXEC_ON_* bit is set, or when it can kill.
/// </summary>
public sealed class EarlyDepthSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 16;
    private const uint BufferFormat = 20;
    private const uint ZOrderEarly = 1u << 4;
    private const uint ZOrderRe = 2u << 4;
    private const uint ZOrderEarlyThenRe = 3u << 4;
    private const uint KillEnable = 1u << 6;
    private const uint ExecOnHierFail = 1u << 9;
    private const uint ExecOnNoop = 1u << 10;
    private const uint DepthBeforeShader = 1u << 12;

    /// <summary>
    /// s[0:3] = counter V#, s4 = primitive mask. Adds 1 to record 0, then MRT0 = interpolated PARAM0.
    /// Assembled with llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding.
    /// </summary>
    private static readonly uint[] CountingPixel =
    [
        0xBEFC0304,                // s_mov_b32 m0, s4
        0x7E0C0281,                // v_mov_b32_e32 v6, 1
        0x7E0E0280,                // v_mov_b32_e32 v7, 0
        0xE0C82000, 0x80000607,    // buffer_atomic_add v6, v7, s[0:3], 0 idxen
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xC8140300,                // v_interp_p1_f32_e32 v5, v0, attr0.w
        0xC8150301,                // v_interp_p2_f32_e32 v5, v1, attr0.w
        0xF800180F, 0x05040302,    // exp mrt0 v2, v3, v4, v5 done vm
        0xBF810000,                // s_endpgm
    ];

    // Control value, whether the occluded fragments' shader runs.
    public static TheoryData<uint, bool> Cases => new()
    {
        { 0u, true },                                           // LATE_Z
        { ZOrderEarly, false },                                 // EARLY_Z_THEN_LATE_Z
        { ZOrderRe, false },                                    // RE_Z
        { ZOrderEarlyThenRe, false },                           // EARLY_Z_THEN_RE_Z
        { ZOrderEarly | ExecOnHierFail, true },
        { ZOrderEarly | ExecOnNoop, true },
        { ZOrderRe | ExecOnHierFail | ExecOnNoop, true },
        { ZOrderEarlyThenRe | ExecOnNoop, true },
        { DepthBeforeShader, false },                           // forced, whatever Z_ORDER says
        { ZOrderEarly | ExecOnHierFail | ExecOnNoop | DepthBeforeShader, false },
        { ZOrderEarly | KillEnable, true },                     // a kill is decided after the shader
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void AnOccludedFragmentStoresWhenTheGuestShaderWouldRun(uint control, bool occludedShaderRuns)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var occludedCounter = rig.Memory.Allocate(64, 256);
        var visibleCounter = rig.Memory.Allocate(64, 256);
        rig.Memory.Write<uint>(occludedCounter, new uint[16]);
        rig.Memory.Write<uint>(visibleCounter, new uint[16]);
        var d = new DepthRig(rig, Size, Size, zFormat: 3);

        // The near quad owns the depth; the far quad is occluded; the last is nearer still and visible.
        d.Draw(d.State().DepthControl(true, true, CompareFunc.Always), d.Full(0.25f, 1, 0, 0));
        foreach (var (z, counter) in new[] { (0.75f, occludedCounter), (0.1f, visibleCounter) })
        {
            var state = d.State().DepthControl(true, true, CompareFunc.Less);
            state.Context[Cx.DbShaderControl] = control;
            rig.DrawMesh(d.Pm4, state, d.Full(z, 0, 1, 0), CountingPixel, Descriptors.Buffer(counter, 4, 16, BufferFormat));
        }

        rig.Submit(d.Pm4);
        // Control: a fragment that passes the test always runs, so the program and the counter work.
        Assert.Equal(Size * Size, rig.Read<uint>(visibleCounter, 1)[0]);
        Assert.Equal(occludedShaderRuns ? Size * Size : 0u, rig.Read<uint>(occludedCounter, 1)[0]);
    }
}
