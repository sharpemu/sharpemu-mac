// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// The resolve memo (GraphicsResolveMemo) hands a draw the programs of an earlier draw with its key only while every
// guest read of that resolution gives the same answer; verify mode resolves each hit again and stops on a difference.
public sealed partial class KeptStageInputsTests
{
    private const ulong LoadingPixelAddress = PipelineTestGuest.MemoryBase + 0x3000;
    private const ulong LoadingPixelHeader = PipelineTestGuest.MemoryBase + 0x8200;
    private const ulong Descriptor = PipelineTestGuest.MemoryBase + 0xD000;
    private const ulong OtherDescriptor = PipelineTestGuest.MemoryBase + 0xD100;

    private ShaderPipelineCache MemoCache(ResolveMemoMode mode = ResolveMemoMode.Verify) =>
        new(_guest.Context, _guest.Host, _guest.Compiler, _guest.Registry) { ResolveMemoMode = mode };

    private GraphicsPrograms Programs(ShaderPipelineCache cache, bool pixelActive)
    {
        var programs = cache.GetGraphicsPrograms(_banks, [], pixelActive, depthBound: false);
        Assert.True(programs.Available);
        return programs;
    }

    private static string Report() => GraphicsResolveMemo.TakeReport();

    // A pixel program that loads a V# through the pointer in s[0:1] and reads through it.
    private void UseLoadingPixelProgram(ulong descriptor)
    {
        _guest.RegisterProgram(LoadingPixelAddress, LoadingPixelHeader, [0xF4080000, 0xFA000000, 0xE0000000, 0x80000000, 0xBF810000],
            PixelUserDataBlock);
        _guest.WriteWords(Descriptor, PipelineTestGuest.BufferDescriptor(StreamBase, 16, 4, Format32x4Float));
        _guest.WriteWords(OtherDescriptor, PipelineTestGuest.BufferDescriptor(StreamBase, 16, 4, Format32x4Float));
        _banks.Shader.Pixel.Address = LoadingPixelAddress;
        PointPixelAt(descriptor);
    }

    private void PointPixelAt(ulong descriptor)
    {
        _banks.Shader.Pixel.UserScalars.Set(0, (uint)descriptor, UserScalarKind.Unknown);
        _banks.Shader.Pixel.UserScalars.Set(1, (uint)(descriptor >> 32), UserScalarKind.Unknown);
        _banks.MarkWritten(1u << (int)RegisterStamp.PixelUserData);
    }

    [Fact]
    public void ResolveMemo_HandsADrawSeenBeforeItsProgramsAndVerifiesThem()
    {
        var cache = MemoCache();
        Report();
        var vertexOnly = Programs(cache, pixelActive: false);
        var withPixel = Programs(cache, pixelActive: true);
        Assert.NotSame(vertexOnly, withPixel);
        // Not only the last draw: both keys are held.
        Assert.Same(vertexOnly, Programs(cache, pixelActive: false));
        Assert.Same(withPixel, Programs(cache, pixelActive: true));
        var report = Report();
        Assert.Contains("hits=2,", report);
        Assert.Contains("verify_checked=2,", report);
        Assert.Contains("verify_failed=0)", report);
    }

    [Fact]
    public void ResolveMemo_TakesAnotherKeyForOtherUserDataAndComesBackToTheFirst()
    {
        var cache = MemoCache();
        UseLoadingPixelProgram(Descriptor);
        var first = Programs(cache, pixelActive: true);
        PointPixelAt(OtherDescriptor);
        var other = Programs(cache, pixelActive: true);
        Assert.NotSame(first, other);
        PointPixelAt(Descriptor);
        Assert.Same(first, Programs(cache, pixelActive: true));
        PointPixelAt(OtherDescriptor);
        Assert.Same(other, Programs(cache, pixelActive: true));
    }

    [Fact]
    public void ResolveMemo_MissesWhenAStagesDescriptorWordChanges()
    {
        var cache = MemoCache();
        UseLoadingPixelProgram(Descriptor);
        var first = Programs(cache, pixelActive: true);
        Assert.Same(first, Programs(cache, pixelActive: true));
        Report();
        _guest.WriteWords(Descriptor, PipelineTestGuest.BufferDescriptor(StreamBase + 0x100, 16, 4, Format32x4Float));
        var changed = Programs(cache, pixelActive: true);
        Assert.NotSame(first, changed);
        Assert.Contains("words=1", Report());
        Assert.Same(changed, Programs(cache, pixelActive: true));
    }

    // A words miss whose new words keep the stage's specialization (another buffer base) is rebuilt from its revalidation:
    // verify mode resolves the same draw in full and finds no difference; the result held before is left as it was.
    [Fact]
    public void ResolveMemo_RebuildsAWordsMissFromItsRevalidationAsTheFullResolutionWould()
    {
        var cache = MemoCache();
        Assert.True(cache.ResolveMemoRebuild);
        UseLoadingPixelProgram(Descriptor);
        var first = Programs(cache, pixelActive: true);
        var firstBuffers = first.PixelInput.Stage.Resources.Buffers.Select(words => words.ToArray()).ToArray();
        var firstVertexStage = first.VertexInput.Stage;
        Report();
        _guest.WriteWords(Descriptor, PipelineTestGuest.BufferDescriptor(StreamBase + 0x100, 16, 4, Format32x4Float));
        var rebuilt = Programs(cache, pixelActive: true);
        var report = Report();
        Assert.Contains("words=1", report);
        Assert.Contains("words_rebuilt=1,", report);
        Assert.Contains("verify_checked=1,", report);
        Assert.Contains("verify_failed=0)", report);
        Assert.NotSame(first, rebuilt);
        Assert.NotSame(first.PixelInput, rebuilt.PixelInput);
        Assert.Same(first.PixelInput.Stage.Program, rebuilt.PixelInput.Stage.Program);
        Assert.Equal(first.Pixel, rebuilt.Pixel);
        Assert.Equal(first.Vertex, rebuilt.Vertex);
        Assert.Contains(rebuilt.PixelInput.Stage.Resources.Buffers, words => words[0] == unchecked((uint)(StreamBase + 0x100)));
        // The earlier result is untouched (a worker may still hold it), and the unchanged vertex stage is kept as it was.
        Assert.Equal(firstBuffers, first.PixelInput.Stage.Resources.Buffers);
        Assert.Equal(firstVertexStage, rebuilt.VertexInput.Stage);
        // The rebuilt entry is the one the next draw takes.
        Assert.Same(rebuilt, Programs(cache, pixelActive: true));
        Assert.Contains("hits=1,", Report());
    }

    // Words that specialize the program otherwise (another stride) may pick another permutation: not rebuilt, resolved in
    // full (and counted); with rebuild off every words miss resolves in full.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ResolveMemo_ResolvesAWordsMissInFullWhenItsSpecializationChangesOrRebuildIsOff(bool rebuild)
    {
        var cache = MemoCache();
        cache.ResolveMemoRebuild = rebuild;
        UseLoadingPixelProgram(Descriptor);
        var first = Programs(cache, pixelActive: true);
        Report();
        _guest.WriteWords(Descriptor, rebuild
            ? PipelineTestGuest.BufferDescriptor(StreamBase, 32, 4, Format32x4Float)
            : PipelineTestGuest.BufferDescriptor(StreamBase + 0x100, 16, 4, Format32x4Float));
        var changed = Programs(cache, pixelActive: true);
        var report = Report();
        Assert.NotSame(first, changed);
        Assert.Contains("words=1", report);
        Assert.Contains("words_rebuilt=0,", report);
        Assert.Contains(rebuild ? "specialization=1" : "specialization=0", report);
        Assert.Same(changed, Programs(cache, pixelActive: true));
        Assert.Contains("verify_failed=0)", Report());
    }

    [Fact]
    public void ResolveMemo_MissesWhenTheVertexTableWordsChange()
    {
        var cache = MemoCache();
        var first = Programs(cache, pixelActive: false);
        Assert.Same(first, Programs(cache, pixelActive: false));
        Report();
        WriteTables(AttributeTable, BufferTable, stride: 32);
        var changed = Programs(cache, pixelActive: false);
        Assert.NotSame(first, changed);
        Assert.Equal(32u, changed.VertexInput.Buffers[0].Stride);
        Assert.Contains("tables=1", Report());
    }

    [Fact]
    public void ResolveMemo_MissesWhenAContextRegisterTheResolutionReadsChanges()
    {
        var cache = MemoCache();
        var first = Programs(cache, pixelActive: true);
        WriteContext(ContextRegisterOffset.CbShaderMask, 0x0000_000F);
        var masked = Programs(cache, pixelActive: true);
        Assert.NotSame(first, masked);
        WriteContext(ContextRegisterOffset.PaClClipCntl, 1u << 16);
        WriteContext(ContextRegisterOffset.PaClViewportXScale, Float(32f));
        var unclipped = Programs(cache, pixelActive: true);
        Assert.NotSame(masked, unclipped);
        WriteContext(ContextRegisterOffset.PaClViewportXScale, Float(64f));
        Assert.Equal(64f, Programs(cache, pixelActive: true).VertexInput.ClipSpace.ScaleX);
    }

    [Fact]
    public void ResolveMemo_KeepsTheProgramsOverAGlobalRegistrationButNotOverANewHeader()
    {
        var cache = MemoCache();
        var first = Programs(cache, pixelActive: false);
        // Another header registered elsewhere: this draw's header record is unchanged.
        ShaderHeaderRegistry.NoteRegistration();
        Assert.Same(first, Programs(cache, pixelActive: false));
        Report();
        _guest.RegisterProgram(VertexAddress, VertexHeader, [0xBF800000, 0xBF810000], UserDataBlock, Semantics, 2);
        Assert.NotSame(first, Programs(cache, pixelActive: false));
        var report = Report();
        Assert.True(report.Contains("header=1") || report.Contains("identity=1") || report.Contains("epoch=1"), report);
    }

    [Fact]
    public void ResolveMemo_MissesWhenThePixelInputSemanticsChange()
    {
        const ulong PixelSemantics = PipelineTestGuest.MemoryBase + 0x9300;
        _guest.WriteWords(PixelSemantics, 0);
        _guest.RegisterProgram(PixelAddress, PixelHeader, PipelineTestGuest.EndProgram, PixelUserDataBlock, PixelSemantics, 1);
        WriteContext(ContextRegisterOffset.SpiPsInControl, 1);
        var cache = MemoCache();
        var first = Programs(cache, pixelActive: true);
        Assert.Same(first, Programs(cache, pixelActive: true));
        Report();
        _guest.WriteWords(PixelSemantics, 1u << 24);
        var custom = Programs(cache, pixelActive: true);
        Assert.NotSame(first, custom);
        Assert.Equal(1u, custom.PixelInput.CustomInterpolationMask);
        Assert.Contains("semantics=1", Report());
    }

    [Fact]
    public void ResolveMemo_MissesOnceADeviceDataLoadIsLearned()
    {
        var cache = MemoCache();
        var first = Programs(cache, pixelActive: false);
        Assert.Same(first, Programs(cache, pixelActive: false));
        DeviceDataLoads.Learn(0x5EED_0000_0000_0000UL | (ulong)Environment.TickCount64, [3]);
        Assert.NotSame(first, Programs(cache, pixelActive: false));
    }

    [Fact]
    public void ResolveMemo_OffResolvesAsTheKeptProgramsDid()
    {
        var cache = MemoCache(ResolveMemoMode.Off);
        _ = Programs(cache, pixelActive: true);
        var cached = Programs(cache, pixelActive: true);
        Assert.Same(cached, Programs(cache, pixelActive: true));
        Assert.NotSame(cached, Programs(cache, pixelActive: false));
        Assert.NotSame(cached, Programs(cache, pixelActive: true));
    }

    // Verify on a thread beside the guest: a resolution reads the descriptor once (the hit's revalidation, the full
    // resolution, the re-check and the second full resolution each read it in turn), and a guest store between two of
    // those reads makes the full resolution differ from a hit that was current when it was made. The answers of the
    // descriptor's first word, read after the two draws that store and first verify the entry (reads 1-3): the hit's
    // revalidation (4), its full resolution (5), the difference's report reading where the words lie (6), the re-check
    // (7), the second full resolution (8).
    private (GraphicsPrograms Hit, GraphicsPrograms Third, string Report) VerifyWithDescriptorReads(Func<int, bool> moved)
    {
        var cache = MemoCache();
        UseLoadingPixelProgram(Descriptor);
        var hit = Programs(cache, pixelActive: true);
        Assert.Same(hit, Programs(cache, pixelActive: true));
        Report();
        var reads = 0;
        _guest.Host.WordRead = (address, word) => address == Descriptor && moved(++reads + 3) ? word + 0x100 : word;
        try
        {
            var third = Programs(cache, pixelActive: true);
            return (hit, third, Report());
        }
        finally
        {
            _guest.Host.WordRead = null;
        }
    }

    // The store lands after the hit read its words and stays: the re-check misses, so the difference is a race (counted),
    // not a mismatch, and the draw takes the full resolution's programs, which read the store.
    [Fact]
    public void ResolveMemo_VerifyCountsAStoreBetweenTheHitAndTheFullResolutionAsARace()
    {
        var failed = VerifyFailedCount();
        var (hit, third, report) = VerifyWithDescriptorReads(read => read >= 5);
        Assert.Contains("verify_raced=(changed=1,came_and_went=0)", report);
        Assert.Equal(failed, VerifyFailedCount());
        Assert.NotSame(hit, third);
        Assert.Contains(third.PixelInput.Stage.Resources.Buffers, words => words[0] == unchecked((uint)(StreamBase + 0x100)));
    }

    // The store is seen by the full resolution only: the re-check holds and a second full resolution agrees with the hit.
    [Fact]
    public void ResolveMemo_VerifyCountsADifferenceThatCameAndWentAsARace()
    {
        var failed = VerifyFailedCount();
        var (hit, third, report) = VerifyWithDescriptorReads(read => read == 5);
        Assert.Contains("verify_raced=(changed=0,came_and_went=1)", report);
        Assert.Equal(failed, VerifyFailedCount());
        Assert.Same(hit, third);
    }

    // A difference that outlives the re-check (the hit still revalidates, a second full resolution still differs) is a
    // mismatch: stopped as before, and on a speculative thread counted apart from plain abandons. Here the store is seen by
    // every read but the re-check's (7), as a writer that does not hold still would be.
    [Fact]
    public void ResolveMemo_VerifyStopsWhenTheHitStillRevalidatesAndTheDifferenceStays()
    {
        var failed = VerifyFailedCount();
        GpuFatal.TakeSpeculativeVerifyFailures();
        var thrown = Assert.Throws<GpuFatal.AbandonedWorkException>(() =>
            GpuFatal.Speculatively(() => VerifyWithDescriptorReads(read => read is >= 5 and not 7)));
        Assert.Contains("verify mismatch field=pixel.buffers", thrown.Message);
        Assert.Contains("recheck=current again=pixel.buffers", thrown.Message);
        // The report names the permutation keys and where the differing words are read from.
        Assert.Matches("pixel=\\[same_program=True specialization=(same_object|equal) words=buffers\\[0\\] ", thrown.Message);
        Assert.Contains($"read_at=w0@0x{Descriptor:X}", thrown.Message);
        var counted = VerifyFailedCount();
        GraphicsResolveMemo.ForgetVerifyFailureForTest();
        Assert.Equal(failed + 1, counted);
        Assert.Equal(1, GpuFatal.TakeSpeculativeVerifyFailures());
    }

    private static long VerifyFailedCount()
    {
        var report = GraphicsResolveMemo.TakeReport();
        var at = report.IndexOf("verify_failed=", StringComparison.Ordinal) + "verify_failed=".Length;
        return long.Parse(report[at..report.IndexOf(')', at)], System.Globalization.CultureInfo.InvariantCulture);
    }

    // The comparer behind verify names every field of what a resolution hands out: a field added later fails here
    // until the comparer (and the memo's key, if the resolution reads something new for it) covers it.
    [Fact]
    public void ResolveMemo_TheVerifyComparerCoversEveryFieldOfAResolution()
    {
        // Every settable property and public field (computed getters derive from them).
        static string[] Names(Type type) => type.GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.SetMethod is not null).Select(property => property.Name)
            .Concat(type.GetFields(BindingFlags.Instance | BindingFlags.Public).Select(field => field.Name))
            .Order(StringComparer.Ordinal).ToArray();

        Assert.Equal(["Available", "DisableBlending", "DropUnwrittenColorTargets", "MeshInput", "Pixel", "PixelInput", "PositionStream",
            "SolidClear", "Tessellation", "Vertex", "VertexInput"], Names(typeof(GraphicsPrograms)));
        Assert.Equal(["Attributes", "Buffers", "ClipSpace", "Dx10Clamp", "FetchAttributeRegister", "FetchBufferRegister", "FetchEmbedded",
            "FloatMode", "IeeeMode", "PositionExportControl", "ScratchDwords", "Stage", "StartInstanceUserData", "Tessellation"],
            Names(typeof(VertexInputInfo)));
        Assert.Equal(["CustomInterpolationMask", "DepthExportEnable", "DepthExportFormat", "Dx10Clamp", "EarlyDepth", "ExecuteOnNoop",
            "FloatMode", "FrontFace", "FrontFaceAllBits", "GuestWave32", "IeeeMode", "InputCount", "InterpolatorSettings", "KillEnable",
            "NoPerspective", "PerspectiveCenterRegister", "PositionW", "PositionX", "PositionY", "PositionZ", "SampleMaskExportEnable",
            "SampleShading", "ScratchDwords", "Stage", "StencilExportEnable", "SystemInputBase", "TargetExportFormats",
            "TargetExportMappings", "TargetOutputModes"], Names(typeof(PixelInputInfo)));
        Assert.Equal(["CallArguments", "CallRecordRanges", "CallRecords", "CallTargets", "MeshDraw", "Program", "Resources", "ShaderBase",
            "TessellationData", "ThreadLimits"], Names(typeof(ShaderStageResources)));
        // Origin and BindingPacket are the snapshot's bookkeeping, not what the draw binds.
        Assert.Equal(["BindingPacket", "BindlessImages", "Buffers", "DeviceAddressRanges", "FlattenedResourceTable", "Images",
            "MipStatistics", "Origin", "Samplers", "UserData"], Names(typeof(ResourceSnapshot)));
    }
}
