// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// What a v2 start prewarms. The driver's shader cache is the OS's and evicts by itself,
// so a stamp left by an earlier finished prewarm must not skip the recorded pipelines.
public sealed class MetalShaderPrewarmPendingTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "SharpEmuTests", Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    // The startup prewarm translates before the command stream exists; with the
    // mip-statistics switch still off its pixel libraries were not the game's (9 of 343
    // matched in Demon's Souls).
    [Fact]
    public void ThePrewarmTranslatesWithTheCommandStreamsMipStatisticsSwitch()
    {
        var previous = ResourceMaterializer.RecordMipStatistics;
        try
        {
            ResourceMaterializer.RecordMipStatistics = !MetalRenderHost.GpuLodStatistics;
            MetalShaderPrewarm.TranslateAsTheGameDoes();
            Assert.Equal(MetalRenderHost.GpuLodStatistics, ResourceMaterializer.RecordMipStatistics);
        }
        finally
        {
            ResourceMaterializer.RecordMipStatistics = previous;
        }
    }

    [Fact]
    public void ACurrentStampStillPrewarmsEveryRecordedCompute()
    {
        using (var list = Open())
        {
            Record(list, 0x1000);
            Record(list, 0x2000);
            list.WriteStamp("build-a");
        }

        using var reopened = Open();
        Assert.True(reopened.IsStampCurrent("build-a"));
        var pending = MetalShaderPrewarm.PendingComputes(reopened, "build-a", out var resumed);
        Assert.Equal(0, resumed);
        Assert.Equal(new ulong[] { 0x1000, 0x2000 }, pending.Select(item => item.Record.Address).Order());
    }

    [Fact]
    public void AnInterruptedPrewarmResumesWithTheComputesItHadNotFinished()
    {
        using (var list = Open())
        {
            Record(list, 0x1000);
            Record(list, 0x2000);
        }

        using var reopened = Open();
        var finished = reopened.LoadedComputes().Single(item => item.Record.Address == 0x1000);
        reopened.AppendProgress("build-a", [ShaderPrewarmList.Identity(finished.Record)]);
        var pending = MetalShaderPrewarm.PendingComputes(reopened, "build-a", out var resumed);
        Assert.Equal(1, resumed);
        Assert.Equal(0x2000UL, Assert.Single(pending).Record.Address);
    }

    [Fact]
    public void WithNoFirstFrameListTheGuestIsHeldForEveryCompute()
    {
        using (var list = Open())
        {
            Record(list, 0x1000);
            Record(list, 0x2000);
        }

        using var reopened = Open();
        Assert.Null(reopened.ReadFirstFrameComputes());
        var ordered = MetalShaderPrewarm.HoldFirst(reopened.LoadedComputes(), null, out var held);
        Assert.Equal(2, held);
        Assert.All(ordered, item => Assert.True(item.Held));
    }

    [Fact]
    public void TheComputesBeforeTheFirstFrameAreHeldForAndCompiledFirst()
    {
        using (var list = Open())
        {
            Record(list, 0x1000);
            Record(list, 0x2000);
            list.WriteFirstFrameComputes();
            // Created after the first frame: not in the next start's hold.
            Record(list, 0x3000);
            list.WriteFirstFrameComputes();
        }

        using (var reopened = Open())
        {
            // A later session first uses 0x3000 and 0x1000 before its frame.
            Record(reopened, 0x3000);
            Record(reopened, 0x1000);
            var firstFrame = reopened.ReadFirstFrameComputes()!;
            var ordered = MetalShaderPrewarm.HoldFirst(reopened.LoadedComputes(), firstFrame, out var held);
            Assert.Equal(2, held);
            Assert.Equal(new ulong[] { 0x1000, 0x2000, 0x3000 }, ordered.Select(item => item.Record.Address));
            Assert.Equal(new[] { true, true, false }, ordered.Select(item => item.Held));
            reopened.WriteFirstFrameComputes();
        }

        using var third = Open();
        var held3 = MetalShaderPrewarm.HoldFirst(third.LoadedComputes(), third.ReadFirstFrameComputes(), out var count);
        Assert.Equal(2, count);
        Assert.Equal(new ulong[] { 0x1000, 0x3000, 0x2000 }, held3.Select(item => item.Record.Address));
    }

    // Linked records never hold the guest and compile after every other compute, in the linked lane's order (the order
    // function stands in for LinkedPrewarmOrder, whose order ShaderCallUnionStoreTests checks), even when the first frame
    // or a list without one would hold them.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LinkedRecordsAreNeverHeldForAndCompileLastInTheirLanesOrder(bool firstFrameList)
    {
        using (var list = Open())
        {
            Record(list, 0x1000);
            Record(list, 0x2000);
            Record(list, 0x3000);
            Record(list, 0x4000);
            list.WriteFirstFrameComputes();
            Record(list, 0x5000);
        }

        using var reopened = Open();
        var linked = new HashSet<ulong> { 0x1000, 0x3000, 0x5000 };
        var ordered = MetalShaderPrewarm.HoldFirstLinkedLast(reopened.LoadedComputes(),
            firstFrameList ? reopened.ReadFirstFrameComputes() : null, record => linked.Contains(record.Address),
            records => records.OrderByDescending(item => item.Record.Address), out var held);

        Assert.Equal(2, held);
        Assert.Equal(new ulong[] { 0x2000, 0x4000, 0x5000, 0x3000, 0x1000 }, ordered.Select(item => item.Record.Address));
        Assert.Equal(new[] { true, true, false, false, false }, ordered.Select(item => item.Held));
    }

    [Fact]
    public void AFirstFrameBeforeAnyComputeHoldsForNone()
    {
        using (var list = Open())
        {
            list.WriteFirstFrameComputes();
            Record(list, 0x1000);
        }

        using var reopened = Open();
        var ordered = MetalShaderPrewarm.HoldFirst(reopened.LoadedComputes(), reopened.ReadFirstFrameComputes(), out var held);
        Assert.Equal(0, held);
        Assert.False(Assert.Single(ordered).Held);
    }

    private ShaderPrewarmList Open() =>
        ShaderPrewarmList.Open(_directory) ?? throw new InvalidOperationException("The list did not open.");

    private static void Record(ShaderPrewarmList list, ulong address)
    {
        var code = new ShaderCodeCapture
        {
            Hash = address,
            CodeSize = 4,
            Address = address,
            Generation = Generation.Gen5,
            Ranges = [new CodeRange(address, [0, 0, 0, 0])],
        };
        list.RecordCompute(code, new ComputePrewarmRecord
        {
            Hash = address,
            CodeSize = 4,
            Address = address,
            UserDataBase = 0,
            UserDataCount = 0,
            PushDataCursor = 0,
            Info = new ComputeInputInfo { ThreadsX = 64, ThreadsY = 1, ThreadsZ = 1 },
            Specialization = new ResourceSpecialization(),
        });
    }
}
