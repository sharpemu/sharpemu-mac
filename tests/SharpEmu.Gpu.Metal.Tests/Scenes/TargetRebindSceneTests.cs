// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Draws that write their target registers again before each draw (DepthRig.State emits the targets and the depth
/// state every time), as a title's material binds do. With the same values the draws keep the targets' version and
/// the previous draw's attachments (RenderExecutor.ResolveTargetLayout): they must stay in one render pass and draw
/// what acquiring the targets again draws. Another value, or another target, still acquires them again.
/// </summary>
public sealed class TargetRebindSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 64;
    private const uint Red = 0xFF0000FF;
    private const uint Green = 0xFF00FF00;
    private const uint Blue = 0xFFFF0000;

    private static readonly FieldInfo RenderPasses =
        typeof(MetalRenderHost).GetField("_renderPasses", BindingFlags.Instance | BindingFlags.NonPublic)!;

    [Fact]
    public void DrawsRebindingTheSameTargetsStayInOnePass()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var d = new DepthRig(rig, Size, Size, zFormat: 3);
        // The depth memory starts at zero: the first draw writes 0.5 everywhere.
        d.Draw(d.State().DepthControl(true, true, CompareFunc.Always), d.Full(0.5f, 1, 0, 0));
        for (var i = 0; i < 4; i++)
        {
            // Nearer, then farther: the depth the earlier draws wrote decides.
            d.Draw(d.State().DepthControl(true, true, CompareFunc.LessEqual), d.Rect(i % 2 == 0 ? 0.25f : 0.75f, 0, 1, 0, i * 16, 0, (i * 16) + 16, 64));
        }

        var colour = d.QueueColorReadback();
        rig.Submit(d.Pm4);
        var colours = colour();

        Assert.Equal(1L, (long)RenderPasses.GetValue(rig.Host)!);
        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(i % 2 == 0 ? Green : Red, colours[(32 * Size) + (i * 16) + 8]);
        }
    }

    // Depth writes turned off under the same targets: a value the depth acquisition reads, so the draw acquires the
    // depth target again; the pass goes on, and the later draws no longer write depth.
    [Fact]
    public void ADepthStateChangeUnderTheSameTargetsIsAcquiredAndKeepsThePass()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var d = new DepthRig(rig, Size, Size, zFormat: 3);
        d.Draw(d.State().DepthControl(true, true, CompareFunc.Always), d.Full(0.5f, 1, 0, 0));
        d.Draw(d.State().DepthControl(true, false, CompareFunc.Always), d.Rect(0.25f, 0, 1, 0, 0, 0, 32, 64));
        // Passes only where depth is still 0.5: the read-only draw before it wrote none.
        d.Draw(d.State().DepthControl(true, true, CompareFunc.Greater), d.Rect(0.4f, 0, 0, 1, 0, 0, 64, 64));

        var depth = d.QueueDepthReadback();
        var colour = d.QueueColorReadback();
        rig.Submit(d.Pm4);
        var colours = colour();
        var depths = depth();

        Assert.Equal(Green, colours[(32 * Size) + 8]);
        Assert.Equal(Red, colours[(32 * Size) + 40]);
        Assert.Equal(0.5f, depths[(32 * Size) + 8]);
        Assert.Equal(0.5f, depths[(32 * Size) + 40]);
    }

    // Another colour target between two draws into the first: three passes, each target with its own draws.
    [Fact]
    public void AnotherTargetBetweenRebindsEndsThePass()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var d = new DepthRig(rig, Size, Size, zFormat: 3);
        var other = rig.ColorTarget(Size, Size);
        d.Draw(d.State().DepthControl(false, false, CompareFunc.Always), d.Full(0.5f, 1, 0, 0));
        d.Draw(new SceneState(Size, Size).ColorTarget(0, other, Size, Size).DepthControl(false, false, CompareFunc.Always), d.Full(0.5f, 0, 0, 1));
        d.Draw(d.State().DepthControl(false, false, CompareFunc.Always), d.Rect(0.5f, 0, 1, 0, 0, 0, 32, 64));

        var colour = d.QueueColorReadback();
        rig.Submit(d.Pm4);
        var colours = colour();
        var others = rig.ReadImage<uint>(other, Size, Size);

        Assert.Equal(3L, (long)RenderPasses.GetValue(rig.Host)!);
        Assert.Equal(Green, colours[(32 * Size) + 8]);
        Assert.Equal(Red, colours[(32 * Size) + 40]);
        Assert.Equal(Blue, others[(32 * Size) + 40]);
    }
}
