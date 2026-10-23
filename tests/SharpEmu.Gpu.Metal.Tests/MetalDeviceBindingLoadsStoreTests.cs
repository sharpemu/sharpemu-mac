// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Binding loads a start left to the device, and the ones it refused, are seeded into the next
// one, so a shader is planned with them from its first compile instead of being learned again
// one re-plan (and one new program) at a time.
public sealed class MetalDeviceBindingLoadsStoreTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "SharpEmuTests", Guid.NewGuid().ToString("N"));

    public MetalDeviceBindingLoadsStoreTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    [Fact]
    public void TheLastWholeLineOfAKeyWinsAndTornLinesAreSkipped()
    {
        var path = Path.Combine(_directory, MetalDeviceBindingLoadsStore.FileName("build"));
        File.WriteAllText(path,
            MetalDeviceBindingLoadsStore.Line(0xABCD, [3, 1], []) + "\n" +
            MetalDeviceBindingLoadsStore.Line(0xABCD, [1], [3]) + "\n" +
            MetalDeviceBindingLoadsStore.Line(0x1234, [0, 5], []) + "\n" +
            "000000000000ABCD 1,7");

        var keys = MetalDeviceBindingLoadsStore.Load(path, out var lines);

        Assert.Equal(4, lines);
        Assert.Equal(2, keys.Count);
        Assert.Equal([1], keys[0xABCD].Learned);
        Assert.Equal([3], keys[0xABCD].Refused);
        Assert.Equal([0, 5], keys[0x1234].Learned);
        Assert.Empty(keys[0x1234].Refused);
        Assert.NotEqual(MetalDeviceBindingLoadsStore.FileName("a"), MetalDeviceDataLoadsStore.FileName("a"));
    }

    [Fact]
    public void ASeededKeyPlansWithItsLoadsAndKeepsItsRefusals()
    {
        const ulong key = 0x5EED_B1D0_0000_0001;
        var changes = new List<ulong>();
        void OnChanged(ulong changed, int[] _, int[] __) => changes.Add(changed);
        DeviceBindingLoads.Changed += OnChanged;
        try
        {
            var learned = DeviceBindingLoads.LearnedLoads;
            DeviceBindingLoads.Seed(key, [4, 9, 2], [2]);

            Assert.Equal([4, 9], DeviceBindingLoads.For(key)!.Order());
            Assert.True(DeviceBindingLoads.IsRefused(key, 2));
            Assert.True(DeviceBindingLoads.IsCurrent(key, [9, 4]));
            Assert.False(DeviceBindingLoads.IsCurrent(key, [4]));
            Assert.False(DeviceBindingLoads.IsCurrent(key, []));
            Assert.Equal(learned, DeviceBindingLoads.LearnedLoads);
            Assert.DoesNotContain(key, changes);

            // A refused load is not learned again; a new one is, and the store hears of it.
            Assert.False(DeviceBindingLoads.Learn(key, 2));
            Assert.True(DeviceBindingLoads.Learn(key, 7));
            Assert.Contains(key, changes);
        }
        finally
        {
            DeviceBindingLoads.Changed -= OnChanged;
            DeviceBindingLoads.Forget(key);
        }

        Assert.Null(DeviceBindingLoads.For(key));
        Assert.True(DeviceBindingLoads.IsCurrent(key, []));
    }
}
