// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class UsedResourceListTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.NonPublic;

    // The resources a draw tells its encoder about: each once, in first-use order, with every
    // usage it was noted with; a new draw starts an empty list. The handles never reach Metal.
    [Fact]
    public void EachResourceIsListedOnceInFirstUseOrderWithItsUsagesJoined()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        var random = new System.Random(7);
        // Few resources, then enough to grow the list and its index twice, then few again.
        foreach (var distinct in new[] { 6, 300, 9, 1000, 3 })
        {
            var expected = new List<nint>();
            var usages = new Dictionary<nint, nuint>();
            using (host.BeginPreparation())
            {
                for (var note = 0; note < distinct * 3; note++)
                {
                    var resource = (nint)(0x6_0000_0000L + random.Next(distinct) * 0x40L);
                    var usage = (nuint)(1 << random.Next(2));
                    host.NoteUsedResource(resource, usage);
                    if (!usages.TryGetValue(resource, out var known))
                    {
                        expected.Add(resource);
                    }

                    usages[resource] = known | usage;
                }

                var count = (int)Field("_usedResourceCount").GetValue(host)!;
                var resources = (nint[])Field("_usedResources").GetValue(host)!;
                var noted = (nuint[])Field("_usedResourceUsage").GetValue(host)!;
                Assert.Equal(expected, resources[..count]);
                Assert.Equal(expected.Select(resource => usages[resource]), noted[..count]);
            }
        }
    }

    private static FieldInfo Field(string name) => typeof(MetalRenderHost).GetField(name, Instance)!;
}
