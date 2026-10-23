// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using System.Diagnostics;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalRecipeClassifierTests(ITestOutputHelper output)
{
    private static RecipeHash Hash(ulong value) { var hash = new RecipeHash(); hash.Add(value); return hash; }
    private static RecipeObservation Issue(ulong structure = 1, ulong value = 2, ulong offset = 3, ulong dependency = 4, ulong pass = 5,
        long ticks = 10, int calls = 2) => new(new(Hash(pass), Hash(structure), Hash(value), Hash(offset), Hash(dependency)), ticks, calls, 1, 0, 0, 0);
    private static RecipeChange[] Compare(RecipeObservation[] current, params RecipeObservation[] previous)
    {
        var result = new RecipeChange[current.Length];
        MetalRecipeClassifier.Compare(current, previous, result);
        return result;
    }
    [Fact]
    public void AllocationExpressionsUsePublishedBoundsAndOffsetsWithoutAddressMasking()
    {
        ViewRecord[] views = [new(0x12345000, 0x1000, 0xA000, HostPageProtection.ReadWrite),
            new(0x12346000, 0x1000, 0xB000, HostPageProtection.ReadWrite)];
        Assert.True(RecipeAddressExpression.TryResolve(views, 0x12345123, 16, out var first, out var firstOffset));
        Assert.True(RecipeAddressExpression.TryResolve(views, 0x12345456, 16, out var second, out var secondOffset));
        Assert.Equal(first, second); Assert.Equal(0x123UL, firstOffset); Assert.Equal(0x456UL, secondOffset);
        Assert.True(RecipeAddressExpression.TryResolve(views, 0x12346123, 16, out var other, out _));
        Assert.NotEqual(first, other);
        Assert.False(RecipeAddressExpression.TryResolve(views, 0x12345FF8, 16, out _, out _));
        Assert.False(RecipeAddressExpression.TryResolve(views, 0x12348000, 16, out _, out _));
        Assert.False(RecipeAddressExpression.TryResolve(views, ulong.MaxValue - 1, 16, out _, out _));
        var remapped = views.ToArray(); remapped[0] = remapped[0] with { Offset = 0xC000 };
        Assert.True(RecipeAddressExpression.TryResolve(remapped, 0x12345123, 16, out var newBacking, out _));
        Assert.NotEqual(first, newBacking);
    }
    [Fact]
    public void WarmMatchingCostAtTheBaselineIssueCountIsReported()
    {
        const int count = 9046, frames = 12;
        var classifier = new MetalRecipeClassifier();
        void Frame(int number)
        {
            for (var i = count - 1; i >= 0; i--)
                classifier.Record(Issue(structure: (ulong)(i % 32), value: (ulong)i, offset: (ulong)(number % 3)));
            classifier.EndFrame();
        }
        for (var i = 0; i < 6; i++) Frame(i);
        var bytes = GC.GetAllocatedBytesForCurrentThread(); var start = Stopwatch.GetTimestamp();
        for (var i = 0; i < frames; i++) Frame(i);
        var elapsed = Stopwatch.GetElapsedTime(start);
        output.WriteLine($"recipe_synthetic issues_per_frame={count} history=4 frames={frames} ms_per_frame={elapsed.TotalMilliseconds / frames:F3} bytes_per_frame={(GC.GetAllocatedBytesForCurrentThread() - bytes) / frames} includes_capture=0 includes_renderer=0");
        Assert.Equal(18, classifier.FrameId);
    }
    [Fact]
    public void ExactRepeatIncludesAllDependencyVersions() =>
        Assert.Equal([RecipeChange.Exact], Compare([Issue()], Issue()));
    [Fact]
    public void OnlyAnExplicitOffsetChangeIsRelocation() =>
        Assert.Equal([RecipeChange.Relocation], Compare([Issue(offset: 123456789)], Issue(offset: 17)));
    [Theory]
    [InlineData(2, 2, 4, 5)] // shader/layout/specialization identity
    [InlineData(1, 3, 4, 5)] // raster/descriptor/resource/allocation identity
    [InlineData(1, 2, 5, 5)] // dependency changed at the SAME address
    [InlineData(1, 2, 4, 6)] // target/subresource/format pass identity
    public void SemanticInputsCannotBeHiddenByRelocation(ulong structure, ulong value, ulong dependency, ulong pass) =>
        Assert.Equal([RecipeChange.Semantic], Compare([Issue(structure, value, dependency: dependency, pass: pass)], Issue()));
    [Fact]
    public void PartialCapturesCannotCreditReuseEvenWithEqualFingerprints()
    {
        Assert.Equal([RecipeChange.Semantic], Compare([Issue() with { Complete = false }], Issue()));
        Assert.Equal([RecipeChange.Semantic], Compare([Issue()], Issue() with { Complete = false }));
    }
    [Fact]
    public void ReorderedIssuesMatchByStructureAcrossFrames() =>
        Assert.Equal([RecipeChange.Exact, RecipeChange.Exact], Compare([Issue(structure: 22), Issue(structure: 11)], Issue(structure: 11), Issue(structure: 22)));
    [Fact]
    public void DuplicateCurrentIssuesCannotReuseOnePreviousIssueTwice() =>
        Assert.Equal([RecipeChange.Exact, RecipeChange.Semantic], Compare([Issue(), Issue()], Issue()));
    [Fact]
    public void ExactMatchesReserveTheirInventoryBeforeRelocations() =>
        Assert.Equal([RecipeChange.Semantic, RecipeChange.Exact], Compare([Issue(offset: 9), Issue(offset: 3)], Issue(offset: 3)));
    [Fact]
    public void ExactAndRelocationConsumeTheSameInventory() =>
        Assert.Equal([RecipeChange.Exact, RecipeChange.Relocation, RecipeChange.Semantic],
            Compare([Issue(offset: 3), Issue(offset: 9), Issue(offset: 10)], Issue(offset: 3), Issue(offset: 4)));
    [Fact]
    public void NonconsecutiveFramesHaveIndependentDenominatorsAndTimeTotals()
    {
        var classifier = new MetalRecipeClassifier();
        classifier.Record(Issue(ticks: 100, calls: 2)); classifier.EndFrame();
        classifier.Record(Issue(structure: 22)); classifier.EndFrame();
        classifier.Record(Issue(ticks: 300, calls: 3)); classifier.EndFrame();
        using var output = new StringWriter(); classifier.Report(output);
        var lines = output.ToString().Split('\n');
        var lag1 = Assert.Single(lines, line => line.Contains("pass=all lag=1 "));
        var lag2 = Assert.Single(lines, line => line.Contains("pass=all lag=2 "));
        Assert.Contains("n=2 exact=0 relocation=0 semantic=2", lag1);
        Assert.Contains("n=1 exact=1 relocation=0 semantic=0", lag2);
        Assert.Contains("exact_resmat_calls=3", lag2);
        Assert.Contains("actual_bypass=0 saved_ms=0", output.ToString());
    }
    [Fact]
    public void CurrentFrameNeverMatchesItselfAndEmptyFramesKeepTheirLag()
    {
        var classifier = new MetalRecipeClassifier();
        classifier.Record(Issue()); classifier.Record(Issue()); classifier.EndFrame();
        classifier.EndFrame();
        classifier.Record(Issue()); classifier.EndFrame();
        using var output = new StringWriter(); classifier.Report(output);
        Assert.Contains("pass=all lag=1 n=1 exact=0 relocation=0 semantic=1", output.ToString());
        Assert.Contains("pass=all lag=2 n=1 exact=1 relocation=0 semantic=0", output.ToString());
    }
    [Fact]
    public void EveryClassRetainsItsOwnMaterializationCostAndPassTotals()
    {
        var classifier = new MetalRecipeClassifier();
        classifier.Record(Issue(structure: 1)); classifier.Record(Issue(structure: 2)); classifier.Record(Issue(structure: 3)); classifier.EndFrame();
        classifier.Record(Issue(structure: 1, calls: 11));
        classifier.Record(Issue(structure: 2, offset: 88, calls: 22));
        classifier.Record(Issue(structure: 3, dependency: 99, calls: 33)); classifier.EndFrame();
        using var output = new StringWriter(); classifier.Report(output);
        var line = Assert.Single(output.ToString().Split('\n'), line => line.Contains("pass=all lag=1 "));
        Assert.Contains("n=3 exact=1 relocation=1 semantic=1 value_repeat=3", line);
        Assert.Contains("exact_resmat_calls=11 relocation_resmat_calls=22 semantic_resmat_calls=33", line);
        Assert.Equal(2, output.ToString().Split('\n').Count(line => line.Contains("lag=1 n=3")));
    }
    [Fact]
    public void DependencyFreeTimeBoundCannotLoseAnExpensiveExactCandidate()
    {
        var classifier = new MetalRecipeClassifier();
        classifier.Record(Issue()); classifier.EndFrame();
        classifier.Record(Issue(dependency: 999, ticks: 1, calls: 1));
        classifier.Record(Issue(ticks: Stopwatch.Frequency, calls: 99)); classifier.EndFrame();
        using var output = new StringWriter(); classifier.Report(output);
        var row = Assert.Single(output.ToString().Split('\n'), line => line.Contains("pass=all lag=1 "));
        Assert.Contains("exact=1 relocation=0 semantic=1 value_repeat=1", row);
        Assert.Contains("value_repeat_calls=99", row);
    }
    [Fact]
    public void HistoryAndIssueRetentionAreBoundedAndDropsAreVisible()
    {
        var classifier = new MetalRecipeClassifier();
        for (var i = 0; i <= MetalRecipeClassifier.MaximumIssues; i++) classifier.Record(Issue());
        classifier.EndFrame();
        for (var frame = 0; frame < MetalRecipeClassifier.HistoryFrames + 2; frame++) classifier.EndFrame();
        classifier.Record(Issue()); classifier.EndFrame();
        using var output = new StringWriter(); classifier.Report(output);
        Assert.Contains("dropped=1", output.ToString());
        Assert.DoesNotContain("exact=1 ", output.ToString());
    }
}
