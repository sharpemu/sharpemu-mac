// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The meshlet cache over a small guest backing: why it builds, which built meshlets it hands
/// back for the bytes at a range, and when a buffer it no longer needs can be written again.
/// Buffers come from the device, so every test is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalMeshletCacheTests(ITestOutputHelper output)
{
    private const ulong RangeAddress = MetalGuestWindowGranuleTests.GuestBase + 0x10000;
    private const uint MaxVertices = 64;
    private const uint MaxPrimitives = 32;

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    private sealed class Rig : IDisposable
    {
        public readonly MetalGuestWindowGranuleTests.Backing Backing = new();
        public readonly MetalGuestMemory Memory;
        public readonly MetalMeshletCache Cache;

        public Rig(int? maxVersions = null)
        {
            Memory = new MetalGuestMemory(
                Device.Value, Backing, new MetalGuestWindowGranuleTests.BackingMemory(Backing),
                MetalGuestWindowGranuleTests.GranuleBytes, MetalGuestWindowGranuleTests.MaxWindowBytes);
            Cache = new MetalMeshletCache(Memory, new MetalGuestWindowGranuleTests.BackingMemory(Backing), maxVersions);
        }

        public void Write(uint[] indices, ulong address = RangeAddress)
        {
            var bytes = new byte[indices.Length * sizeof(uint)];
            Buffer.BlockCopy(indices, 0, bytes, 0, bytes.Length);
            Assert.True(Backing.TryWriteBacking(address, bytes));
        }

        public MetalMeshletCache.Meshlets Get(uint indexCount, ulong serial, ulong completed, ulong address = RangeAddress)
        {
            Assert.True(Cache.TryGet(address, indexCount, true, MaxVertices, MaxPrimitives, serial, completed, out var meshlets));
            return meshlets;
        }

        public void Dispose()
        {
            Memory.Dispose();
            Backing.Dispose();
        }
    }

    // A triangle list of `triangles` triangles over a vertex pool, shaped by the seed.
    private static uint[] Indices(int seed, int triangles = 300)
    {
        var random = new Random(seed);
        var indices = new uint[triangles * 3];
        for (var i = 0; i < indices.Length; i++)
        {
            indices[i] = (uint)random.Next(0, 400);
        }

        return indices;
    }

    // The words the cache handed back, as the mesh prologue would read them.
    private static uint[] Words(MetalMeshletCache.Meshlets meshlets, int triangles)
    {
        var contents = MetalNative.Send(meshlets.Buffer, MetalNative.Selector("contents"));
        return new ReadOnlySpan<uint>((void*)contents, (int)meshlets.PrimitiveBase + triangles).ToArray();
    }

    private static void AssertBuiltFrom(uint[] indices, MetalMeshletCache.Meshlets meshlets)
    {
        var expected = MeshletBuilder.Build(indices, MaxVertices, MaxPrimitives);
        Assert.Equal(expected.MeshletCount, meshlets.Count);
        Assert.Equal(expected.VertexBase, meshlets.VertexBase);
        Assert.Equal(expected.PrimitiveBase, meshlets.PrimitiveBase);
        Assert.Equal(expected.Words, Words(meshlets, indices.Length / 3));
    }

    // Content A, then B, then A again at one range, each under a new submission: the first build
    // finds a key it never saw, the second a hash the key never had, the third one it had. One
    // version per key, so the third lookup builds.
    [Fact]
    public void RebuildReasonsAreCountedByKind()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var rig = new Rig(maxVersions: 1);
        var a = Indices(1);
        var b = Indices(2);
        rig.Write(a);
        AssertBuiltFrom(a, rig.Get((uint)a.Length, 1, 0));
        rig.Write(b);
        AssertBuiltFrom(b, rig.Get((uint)b.Length, 2, 1));
        rig.Write(a);
        AssertBuiltFrom(a, rig.Get((uint)a.Length, 3, 2));

        Assert.Equal(1, rig.Cache.Builds(MetalMeshletCache.BuildReason.New));
        Assert.Equal(1, rig.Cache.Builds(MetalMeshletCache.BuildReason.Changed));
        Assert.Equal(1, rig.Cache.Builds(MetalMeshletCache.BuildReason.Recurring));
        Assert.Equal(0, rig.Cache.Evictions);
        var counters = rig.Cache.TakeCounters();
        output.WriteLine(counters);
        Assert.Contains("meshlet_builds_by=[new=1 changed=1 recurring=1]", counters);
        Assert.Contains("meshlet_evictions=0", counters);
        Assert.Contains($"meshlet_hash_kb={3 * a.Length * sizeof(uint) / 1024}", counters);
        Assert.Equal(0, rig.Cache.Builds(MetalMeshletCache.BuildReason.New));
    }

    [Fact]
    public void AReturningIndexPatternReusesItsEarlierMeshlets()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var rig = new Rig(maxVersions: 4);
        var a = Indices(1);
        var b = Indices(2);
        rig.Write(a);
        var first = rig.Get((uint)a.Length, 1, 0);
        rig.Write(b);
        var second = rig.Get((uint)b.Length, 2, 1);
        rig.Write(a);
        var third = rig.Get((uint)a.Length, 3, 2);

        Assert.NotEqual(first.Buffer, second.Buffer);
        Assert.Equal(first, third);
        AssertBuiltFrom(a, third);
        Assert.Equal(1, rig.Cache.VersionHits);
        Assert.Equal(2, rig.Cache.VersionCount);
        Assert.Equal(1, rig.Cache.EntryCount);
        var counters = rig.Cache.TakeCounters();
        Assert.Contains("meshlet_builds=2 ", counters);
        Assert.Contains("meshlet_version_hits=1 meshlet_versions=2", counters);
    }

    // Contents cycling through and back over one range, one lookup per submission: whatever
    // version is handed back holds exactly the meshlets of the bytes there now.
    [Fact]
    public void AChangedRangeNeverReturnsAnotherVersion()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var rig = new Rig(maxVersions: 4);
        var contents = Enumerable.Range(10, 6).Select(seed => Indices(seed)).ToArray();
        var order = new[] { 0, 1, 2, 0, 3, 1, 4, 5, 0, 2, 2, 5, 1, 3, 0 };
        var serial = 1UL;
        foreach (var index in order)
        {
            rig.Write(contents[index]);
            AssertBuiltFrom(contents[index], rig.Get((uint)contents[index].Length, serial, serial - 1));
            serial++;
        }

        Assert.True(rig.Cache.VersionCount <= 4);
        Assert.True(rig.Cache.VersionHits > 0);
    }

    // Five contents at one range under submissions that have not completed: the key keeps four
    // versions, and the one that made way is released only once its last submission completes.
    [Fact]
    public void VersionsPerKeyAreBoundedAndRetiredOnlyAfterCompletion()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var rig = new Rig(maxVersions: 4);
        var contents = Enumerable.Range(20, 5).Select(seed => Indices(seed)).ToArray();
        for (var index = 0; index < contents.Length; index++)
        {
            rig.Write(contents[index]);
            rig.Get((uint)contents[index].Length, (ulong)index + 1, 0);
        }

        Assert.Equal(4, rig.Cache.VersionCount);
        Assert.Equal(1, rig.Cache.RetiredCount);
        // Its last use was submission 1: still bound while nothing completed.
        rig.Write(contents[4]);
        rig.Get((uint)contents[4].Length, 6, 0);
        Assert.Equal(1, rig.Cache.RetiredCount);
        rig.Get((uint)contents[4].Length, 7, 1);
        Assert.Equal(0, rig.Cache.RetiredCount);

        // The first content is gone and is built again; the others are still versions of the key.
        rig.Write(contents[0]);
        rig.Get((uint)contents[0].Length, 8, 1);
        Assert.Contains("meshlet_builds=6 ", rig.Cache.TakeCounters());
    }

    [Fact]
    public void TheSameSubmissionSkipReturnsTheVersionItValidated()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var rig = new Rig(maxVersions: 4);
        var a = Indices(1);
        var b = Indices(2);
        rig.Write(a);
        var first = rig.Get((uint)a.Length, 1, 0);
        rig.Write(b);
        var second = rig.Get((uint)b.Length, 2, 1);
        rig.Write(a);
        Assert.Equal(first, rig.Get((uint)a.Length, 3, 2));
        _ = rig.Cache.TakeCounters();

        // Later draws of submission 3 skip the hash and get the version submission 3 validated,
        // not the one built last.
        var again = rig.Get((uint)a.Length, 3, 2);
        Assert.Equal(first, again);
        Assert.NotEqual(second, again);
        var counters = rig.Cache.TakeCounters();
        Assert.Contains("meshlet_validations_skipped=1 meshlet_builds=0 ", counters);
        Assert.Contains("meshlet_hash_kb=0 ", counters);
    }

    [Fact]
    public void WithVersionsOffOneEntryPerKey()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var rig = new Rig(maxVersions: 1);
        var a = Indices(1);
        var b = Indices(2);
        var serial = 1UL;
        foreach (var content in new[] { a, b, a, b })
        {
            rig.Write(content);
            AssertBuiltFrom(content, rig.Get((uint)content.Length, serial, serial - 1));
            Assert.Equal(1, rig.Cache.VersionCount);
            serial++;
        }

        Assert.Equal(0, rig.Cache.VersionHits);
        Assert.Contains("meshlet_builds=4 ", rig.Cache.TakeCounters());
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Meshlet buffers are created on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }
}
