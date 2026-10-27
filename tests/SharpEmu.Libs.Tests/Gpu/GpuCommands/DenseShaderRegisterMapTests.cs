// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

public sealed class DenseShaderRegisterMapTests
{
    [Fact]
    public void PresenceCountAndOverflowDistinguishUnwrittenFromZero()
    {
        var map = new DenseShaderRegisterMap();
        foreach (var key in new[] { 0u, 63u, 64u, 767u, 768u, uint.MaxValue })
        {
            Assert.False(map.TryGetValue(key, out var missing));
            Assert.Equal(0u, missing);
            Assert.Throws<KeyNotFoundException>(() => map[key]);
            map[key] = 0;
            Assert.True(map.ContainsKey(key));
            Assert.True(map.TryGetValue(key, out var zero));
            Assert.Equal(0u, zero);
            map[key] = key;
            Assert.Equal(key, map[key]);
        }
        Assert.Equal(6, map.Count);
        Assert.Throws<ArgumentException>(() => map.Add(0, 1));
        Assert.Equal(6, map.Count);
    }

    [Fact]
    public void InsertionOrderAndLiveViewsSurviveClearThroughAnAlias()
    {
        var map = new DenseShaderRegisterMap();
        IDictionary<uint, uint> alias = map;
        IReadOnlyDictionary<uint, uint> readOnly = map;
        var keys = alias.Keys;
        var values = alias.Values;
        alias[900] = 9;
        alias[2] = 2;
        alias[1] = 1;
        alias[2] = 22;
        Assert.Equal(new uint[] { 900, 2, 1 }, keys);
        Assert.Equal(new uint[] { 9, 22, 1 }, values);
        Assert.Equal(22u, readOnly[2]);
        Assert.True(alias.Remove(2));
        Assert.False(alias.Remove(2));
        alias.Add(2, 23);
        Assert.Equal(new uint[] { 900, 1, 2 }, readOnly.Keys);
        Assert.Equal(3, new Dictionary<uint, uint>(map).Count);
        alias.Clear();
        Assert.Empty(map);
        Assert.Empty(keys);
        Assert.Empty(values);
        Assert.Equal(0, readOnly.Count);
        Assert.False(map.ContainsKey(900));
        Assert.False(map.ContainsKey(1));
        alias[1] = 0;
        Assert.Single(map);
        Assert.Equal(new uint[] { 1 }, keys);
        Assert.Equal(new uint[] { 0 }, values);
        Assert.Throws<NotSupportedException>(() => keys.Add(3));
        Assert.Throws<NotSupportedException>(() => values.Clear());
    }

    [Fact]
    public void PairAndCopyContractsPreserveExactMembership()
    {
        ICollection<KeyValuePair<uint, uint>> map = new DenseShaderRegisterMap();
        map.Add(new(12, 34));
        Assert.False(map.Remove(new(12, 35)));
        Assert.True(map.Contains(new(12, 34)));
        var copied = new KeyValuePair<uint, uint>[3];
        map.CopyTo(copied, 1);
        Assert.Equal(new KeyValuePair<uint, uint>(12, 34), copied[1]);
        Assert.Throws<ArgumentNullException>(() => map.CopyTo(null!, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => map.CopyTo(copied, -1));
        Assert.Throws<ArgumentException>(() => map.CopyTo(copied, 3));
        Assert.True(map.Remove(new(12, 34)));
        Assert.Empty(map);
    }

    [Fact]
    public void EnumerationDetectsInsertionButAllowsOverwriteRemovalAndClear()
    {
        var map = new DenseShaderRegisterMap { [1] = 1, [2] = 2 };
        var overwritten = map.GetEnumerator();
        map[1] = 7;
        Assert.True(overwritten.MoveNext());
        Assert.Equal(7u, overwritten.Current.Value);
        map.Remove(2);
        Assert.False(overwritten.MoveNext());
        var cleared = map.GetEnumerator();
        map.Clear();
        Assert.False(cleared.MoveNext());
        var inserted = map.GetEnumerator();
        map[3] = 3;
        Assert.Throws<InvalidOperationException>(() => inserted.MoveNext());
        var keys = map.Keys.GetEnumerator();
        map[4] = 4;
        Assert.Throws<InvalidOperationException>(() => keys.MoveNext());
    }

    [Fact]
    public void ReinsertCompactionPreservesLiveInsertionOrder()
    {
        var map = new DenseShaderRegisterMap { [1] = 1, [900] = 9, [2] = 0 };
        for (var i = 0; i < 2000; i++)
        {
            Assert.True(map.Remove(2));
            map[2] = (uint)i;
        }
        Assert.Equal(new uint[] { 1, 900, 2 }, map.Keys);
        Assert.Equal(3, map.Count);
        Assert.Equal(1999u, map[2]);
    }

    [Fact]
    public void SeededMixedOperationsMatchDictionaryValuesAndCounts()
    {
        var random = new System.Random(7721);
        var map = new DenseShaderRegisterMap();
        var reference = new Dictionary<uint, uint>();
        for (var i = 0; i < 10000; i++)
        {
            var key = i % 17 == 0 ? uint.MaxValue : (uint)random.Next(0, 900);
            switch (random.Next(0, 12))
            {
                case 0:
                    Assert.Equal(reference.Remove(key), map.Remove(key));
                    break;
                case 1:
                    map.Clear(); reference.Clear();
                    break;
                default:
                    var value = (uint)random.Next();
                    map[key] = value; reference[key] = value;
                    break;
            }
            Assert.Equal(reference.Count, map.Count);
            Assert.Equal(reference.TryGetValue(key, out var expected), map.TryGetValue(key, out var actual));
            Assert.Equal(expected, actual);
            if (i % 100 == 0)
                Assert.Equal(reference.OrderBy(x => x.Key), map.OrderBy(x => x.Key));
        }
    }
}
