// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// The dispatcher's step cap keeps a mistranslated loop-exit condition from wedging the
// single Metal queue, and the step-report word is how the host learns it happened: the
// last field of every stage argument buffer points at one word per shader hash, and the
// dispatcher increments it on the way out. Without it a capped shader is indistinguishable
// from a slow one.
public sealed class Gen5MslStepCapReportTests(ITestOutputHelper output)
{
    // s_branch -1 jumps to itself: the dispatcher can only leave through the cap.
    private static readonly Gen5ComputeFixture SelfBranch = new("step-cap", [0xBF82FFFFu, 0xBF810000u], 0, 0);

    // One pass through straight-line ALU; the cap is never reached.
    private static readonly Gen5ComputeFixture Terminating = Gen5ComputeFixtures.Fmac;

    // 60000 trips around a loop whose exit test and back edge sit in different blocks,
    // so each trip is two block moves: 120000 of them, but only 60000 iterations.
    private static readonly Gen5ComputeFixture TwoBlockLoop = new(
        "two-block-loop",
        [
            0xBE8003FF, 0x0000EA60, // s_mov_b32 s0, 60000
            0x81808100,             // loop: s_sub_i32 s0, s0, 1
            0xBF068000,             // s_cmp_eq_u32 s0, 0
            0xBF850001,             // s_cbranch_scc1 done
            0xBF82FFFC,             // s_branch loop
            0xBF810000,             // done: s_endpgm
        ],
        0,
        0);

    private static readonly Gen5MslCodegenOptions Dispatcher = new(StructuredControlFlow: false);

    [Fact]
    public void TheDispatcherReportsEveryInvocationThatRanOutOfSteps()
    {
        if (SkipWithoutMetalDevice()) return;

        const uint threads = 8;
        var hits = Execute(SelfBranch, threads);
        Assert.Equal(threads, hits);
    }

    [Fact]
    public void AProgramThatTerminatesReportsNothing()
    {
        if (SkipWithoutMetalDevice()) return;

        Assert.Equal(0u, Execute(Terminating, threads: 4));
    }

    [Fact]
    public void TheDispatcherChargesLoopIterationsNotBlocks()
    {
        if (SkipWithoutMetalDevice()) return;

        // Persistent-thread kernels poll for work across many blocks per trip; charging
        // every block cut them off long before their loops were done.
        Assert.Equal(0u, Execute(TwoBlockLoop, threads: 4, Dispatcher));
        Assert.Equal(4u, Execute(SelfBranch, threads: 4, Dispatcher));
    }

    private uint Execute(Gen5ComputeFixture fixture, uint threads, Gen5MslCodegenOptions? options = null)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, localSizeX: threads);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var emitError, options), $"[{fixture.Name}] {emitError}");
        if (options is { StructuredControlFlow: false })
        {
            Assert.Contains("switch (pc)", shader.Source, StringComparison.Ordinal);
        }

        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError),
            $"[{fixture.Name}] {compileError}");
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                new byte[32],
                new byte[Resources.PushData.ByteSize],
                shader.ArgumentLayout!,
                threads,
                out _,
                out var hits,
                out var runError),
            $"[{fixture.Name}] {runError}");
        return hits;
    }

    private bool SkipWithoutMetalDevice()
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            "The required gate cannot run the step-cap report test without a Metal device.");
        output.WriteLine("No Metal device on this host; the step-cap report test skipped.");
        return true;
    }
}
