// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalHostLookupTests
{
    private sealed class Dependency : ResourceDependency
    {
        internal bool Current = true;
        public override ResourceDependencyStatus Validate() => Current ? ResourceDependencyStatus.Current : ResourceDependencyStatus.CpuWrite;
        public override void Dispose() => Current = false;
    }
    private sealed class Source : IResourceDependencySource
    {
        public ResourceDependencyEpoch DependencyEpoch { get; set; }
        public bool DependencyReadsReady { get; set; } = true;
        internal Dependency? Last;
        internal bool Capturable = true;
        internal int Captures;
        public bool TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean, out ResourceDependency? dependency)
        {
            Captures++;
            dependency = Capturable ? Last = new() : null;
            return dependency is not null;
        }
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)]
    public void MetadataCodeRequiresEveryDependencyEpochAndOriginalReceipt(int field)
    {
        var source = new Source();
        using var cache = new MetalMetadataCodeCache();
        cache.Keep(source, 4096, 4096, 0xCC, source.DependencyEpoch);
        Assert.True(cache.TryGet(source, 4096, 4096, out var code));
        Assert.Equal(0xCC, code);
        source.DependencyEpoch = field switch
        {
            0 => new(1, 0, 0, 0, 0), 1 => new(0, 1, 0, 0, 0), 2 => new(0, 0, 1, 0, 0),
            3 => new(0, 0, 0, 1, 0), _ => new(0, 0, 0, 0, 1),
        };
        Assert.False(cache.TryGet(source, 4096, 4096, out _));
        cache.Keep(source, 4096, 4096, 0xDD, source.DependencyEpoch);
        Assert.True(cache.TryGet(source, 4096, 4096, out _));
        source.DependencyReadsReady = false;
        Assert.False(cache.TryGet(source, 4096, 4096, out _));
        source.DependencyReadsReady = true;
        source.Last!.Current = false;
        Assert.False(cache.TryGet(source, 4096, 4096, out _));
        Assert.False(cache.TryGet(source, 4096, 8192, out _));
    }

    [Fact]
    public void MetadataCodeRefusalIsBoundedAndCannotRecertifyAcrossAnEpochRace()
    {
        var source = new Source { Capturable = false };
        using var cache = new MetalMetadataCodeCache();
        for (var i = 0; i < 20; i++) cache.Keep(source, 4096, 4096, 0, source.DependencyEpoch);
        Assert.Equal(1, source.Captures);
        Assert.False(cache.TryGet(source, 4096, 4096, out _));
        var oldEpoch = source.DependencyEpoch;
        source.DependencyEpoch = new(1, 0, 0, 0);
        cache.Keep(source, 4096, 4096, 0, source.DependencyEpoch);
        Assert.Equal(1, source.Captures); // An unrelated GPU epoch cannot repair a missing physical watch.
        source.DependencyEpoch = new(1, 0, 1, 0);
        source.Capturable = true;
        cache.Keep(source, 4096, 4096, 0, oldEpoch);
        Assert.False(cache.TryGet(source, 4096, 4096, out _));
        cache.Keep(source, 4096, 4096, 0, source.DependencyEpoch);
        Assert.True(cache.TryGet(source, 4096, 4096, out _));
    }

    [Fact]
    public void ResidentWalkExpiresOnCollectionEncoderAndStageWidening()
    {
        var walk = new MetalResidencyWalk();
        Assert.False(walk.IsCurrent(1, 1, 0));
        walk.Record(1, 1, Mtl.RenderStageVertex);
        Assert.True(walk.IsCurrent(1, 1, Mtl.RenderStageVertex));
        Assert.False(walk.IsCurrent(1, 1, Mtl.RenderStageFragment));
        walk.Record(1, 1, Mtl.RenderStageFragment);
        Assert.True(walk.IsCurrent(1, 1, Mtl.RenderStageVertex | Mtl.RenderStageFragment));
        Assert.False(walk.IsCurrent(2, 1, 0));
        Assert.False(walk.IsCurrent(1, 2, 0));
        walk.Record(2, 2, 0); // A compute encoder also requires its own walk.
        Assert.True(walk.IsCurrent(2, 2, 0));
        Assert.False(walk.IsCurrent(2, 2, Mtl.RenderStageVertex));
        walk.Reset();
        Assert.False(walk.IsCurrent(2, 2, 0));
    }

    [Fact]
    public void NamedWindowVersionObservesAllMutationPaths()
    {
        var set = new MetalWindowSet();
        void Changes(Action action) { var version = set.Version; action(); Assert.NotEqual(version, set.Version); }
        Changes(() => set.Add(700));
        var same = set.Version; set.Add(700); Assert.Equal(same, set.Version);
        Changes(() => set.Remove(700));
        var other = new MetalWindowSet(); other.Add(7);
        Changes(() => set.UnionWith(other));
        Changes(() => set.CopyFrom(other));
        Changes(() => set.SetIntersection(other, other));
        Changes(() => set.SetDifference(other, other));
        Changes(set.Clear);
    }
}
