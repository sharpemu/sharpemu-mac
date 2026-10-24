// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Metal;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The device-address table over granule-sized windows, read by the real resolver
/// (sharpemu_resolve_device_address from resource_prelude.msl): runs are cut at granule
/// and window edges, every entry points into its own window, a shader reads the guest bytes
/// at addresses in each of them - also after a range moved a window edge off a granule edge -
/// and a lookup in a window the submission did not name both performs the access and
/// reports its granule, which the host then names.
/// </summary>
public sealed unsafe class MetalDeviceAddressWindowTests(ITestOutputHelper output)
{
    private const ulong GuestBase = MetalGuestWindowGranuleTests.GuestBase;
    private const ulong Granule = MetalGuestWindowGranuleTests.GranuleBytes;

    // Two mapped spans: one across granules 0..2, one inside granule 3.
    private static readonly (ulong Address, ulong Size)[] Spans =
    [
        (GuestBase + 0x0F0000, 0x200000),
        (GuestBase + 0x380000, 0x040000),
    ];

    // Addresses on both sides of every granule edge the first span crosses, inside the
    // second span, and two the table does not map.
    private static readonly ulong[] Probes =
    [
        GuestBase + 0x0F0004, GuestBase + 0x0FFFFC, GuestBase + 0x100000, GuestBase + 0x1FFFFC,
        GuestBase + 0x200000, GuestBase + 0x2EFFFC, GuestBase + 0x380010, GuestBase + 0x3BFFFC,
        GuestBase + 0x2F0000, GuestBase + 0x000010,
    ];

    private static readonly bool[] Mapped = [true, true, true, true, true, true, true, true, false, false];

    private static string KernelSource() =>
        "#include <metal_stdlib>\nusing namespace metal;\n" + Gen5MslTranslator.DeviceAddressPrelude() + """

        kernel void probe(device const Gen5AddressRange* ranges [[buffer(0)]],
                          constant uint& range_count [[buffer(1)]],
                          device uint* faults [[buffer(2)]],
                          device const ulong* addresses [[buffer(3)]],
                          device uint* results [[buffer(4)]],
                          uint tid [[thread_position_in_grid]])
        {
            Gen5RangeCache cache = { 0ul, 0ul, nullptr };
            results[tid] = sharpemu_load_device_dword(ranges, range_count, faults, 0u, cache, addresses[tid]);
        }
        """;

    private static uint Pattern(ulong address) => (uint)((address - GuestBase) * 2654435761UL) ^ 0x5A5A0000u;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ATableOverManyWindowsResolvesEveryMappedAddress(bool movedEdge)
    {
        if (!RequireDevice())
        {
            return;
        }

        using var fixture = new Fixture(movedEdge: movedEdge);
        var table = fixture.Table;
        // A range across the edge at granule 1 moves it to 0x108000, which cuts granule 1's
        // run in two windows.
        var entries = movedEdge ? 5 : 4;
        Assert.Equal(entries, table.Count);
        Assert.Equal(movedEdge ? [0, 1, 1, 2, 3] : [0, 1, 2, 3], table.EntryGranules.ToArray());
        Assert.Equal([0, 1, 2, 3], table.Granules.ToArray());
        Assert.Equal(4, table.Windows.Count);
        Assert.Equal(entries, table.Entries.Select(entry => entry.Base).Distinct().Count());
        output.WriteLine($"entries={table.Count} windows={table.Windows.Count}");

        var results = fixture.Run(namedWindows: null, nameWindows: table.Windows);
        for (var index = 0; index < Probes.Length; index++)
        {
            Assert.Equal(Mapped[index] ? Pattern(Probes[index]) : 0u, results[index]);
        }

        Assert.Equal(0, fixture.Reach.TakeFeedback());
        Assert.Equal((0L, 0L), fixture.Reach.TakeCounters());
    }

    // The submission reaches granules 0 and 1 and names their windows; the table flags the
    // entries in the windows of 2 and 3. A shader that reads there still gets the guest
    // bytes, and reports both granules, which the host adds to the reach set and the title's
    // profile: the next table flags nothing and the next run reports no miss.
    [Fact]
    public void AMissInAnUnnamedGranuleNamesItFromTheNextSubmission()
    {
        if (!RequireDevice())
        {
            return;
        }

        var directory = Path.Combine(Path.GetTempPath(), $"sharpemu-reach-{Guid.NewGuid():N}");
        try
        {
            var profile = MetalDeviceAddressReachProfile.Open(directory, Granule);
            Assert.NotNull(profile);
            using var fixture = new Fixture(profile);
            fixture.Reach.Reached.Add(0);
            fixture.Reach.Reached.Add(1);
            var named = new MetalWindowSet();
            named.SetIntersection(fixture.Table.Granules, fixture.Reach.Reached);
            Assert.Equal([0, 1], named.ToArray());
            var namedWindows = new MetalWindowSet();
            fixture.Table.AddWindowsOfGranules(named, namedWindows);
            Assert.Equal(2, namedWindows.Count);

            var results = fixture.Run(namedWindows, nameWindows: fixture.Table.Windows);
            for (var index = 0; index < Probes.Length; index++)
            {
                Assert.Equal(Mapped[index] ? Pattern(Probes[index]) : 0u, results[index]);
            }

            Assert.Equal(2, fixture.Reach.TakeFeedback());
            Assert.Equal([0, 1, 2, 3], fixture.Reach.Reached.ToArray());
            var (misses, learned) = fixture.Reach.TakeCounters();
            Assert.Equal(2L, learned);
            Assert.True(misses >= 4, $"misses={misses}");
            output.WriteLine($"misses={misses} learned={learned}");

            named.SetIntersection(fixture.Table.Granules, fixture.Reach.Reached);
            namedWindows.Clear();
            fixture.Table.AddWindowsOfGranules(named, namedWindows);
            fixture.Run(namedWindows, nameWindows: namedWindows);
            Assert.Equal(0, fixture.Reach.TakeFeedback());
            Assert.Equal(0L, fixture.Reach.TakeCounters().Misses);

            // The profile holds what was learned, at the granule size it was learned at.
            var (bits, granules) = MetalDeviceAddressReachProfile.Load(Path.Combine(directory, MetalDeviceAddressReachProfile.FileName));
            Assert.Equal(20, bits);
            Assert.Equal(new HashSet<ulong> { 2, 3 }, granules);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void CapturedRangesNameAllIntersectingWindowsBeforeTheirFirstUse()
    {
        if (!RequireDevice()) return;
        using var fixture = new Fixture(movedEdge: true);
        var windows = new MetalWindowSet();
        Assert.True(fixture.Table.AddWindowsOfRange(GuestBase + 0x0F0004, 0x200000 - 4, windows));
        Assert.Equal([0, 1, 2], windows.ToArray());
        Assert.False(fixture.Table.AddWindowsOfRange(GuestBase + 0x0F0004, 4, windows));
        Assert.False(fixture.Table.AddWindowsOfRange(GuestBase + 0x2F0000, 0x90000, windows));
        Assert.True(fixture.Table.AddWindowsOfRange(GuestBase + 0x380000, 4, windows));
        Assert.Equal([0, 1, 2, 3], windows.ToArray());
        Assert.False(fixture.Table.AddWindowsOfRange(GuestBase, 0, windows));
        Assert.False(fixture.Table.AddWindowsOfRange(ulong.MaxValue - 3, 8, windows));
    }

    private sealed class Fixture : IDisposable
    {
        private readonly MetalGuestWindowGranuleTests.Backing _backing = new();
        private readonly MetalTransferHarness _harness;
        private readonly nint _pipeline;

        public Fixture(MetalDeviceAddressReachProfile? profile = null, bool movedEdge = false)
        {
            for (var address = GuestBase; address < GuestBase + MetalGuestWindowGranuleTests.BackingSize; address += 4)
            {
                _backing.TryWriteBacking(address, BitConverter.GetBytes(Pattern(address)));
            }

            Memory = new MetalGuestMemory(
                Device.Value, _backing, new MetalGuestWindowGranuleTests.BackingMemory(_backing),
                Granule, MetalGuestWindowGranuleTests.MaxWindowBytes);
            _harness = new MetalTransferHarness(Device.Value);
            if (movedEdge)
            {
                Memory.BeginSubmission(1, 0);
                Assert.True(Memory.TryResolveInPlace(GuestBase + 0x0F8000, 0x10000, out _));
                Assert.Equal(1L, Memory.WindowLayoutCounts.Moves);
                Memory.BeginSubmission(2, 1);
            }

            Table.Rebuild(Memory, visit =>
            {
                foreach (var (address, size) in Spans)
                {
                    visit(address, size);
                }
            }, (int)Gen5MslTranslator.MaxAddressRangeCount);
            Reach = new MetalDeviceAddressReach(Memory, profile);
            _pipeline = MetalGuestWindowResidencyTests.ComputePipeline(KernelSource(), "probe");
        }

        public MetalGuestMemory Memory { get; }

        public MetalDeviceAddressTable Table { get; } = new();

        public MetalDeviceAddressReach Reach { get; }

        public uint[] Run(MetalWindowSet? namedWindows, MetalWindowSet nameWindows)
        {
            var tableBuffer = _harness.Buffer((ulong)Table.ByteSize);
            Table.Write(new Span<byte>((void*)MetalNative.Send(tableBuffer, MetalNative.Selector("contents")), Table.ByteSize), namedWindows);
            var addresses = _harness.Buffer(MemoryMarshal.AsBytes(Probes.AsSpan()));
            var results = _harness.Buffer((ulong)(Probes.Length * sizeof(uint)));
            var count = (uint)Table.Count | MetalDeviceAddressIndex.IndexedFlag;

            _harness.Begin();
            var encoder = MetalNative.Send(_harness.CommandBuffer, MetalNative.Selector("computeCommandEncoder"));
            MetalNative.SendVoid(encoder, MetalNative.Selector("setComputePipelineState:"), _pipeline);
            var setBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
            MetalNative.SendSetBuffer(encoder, setBuffer, tableBuffer, 0, 0);
            MetalNative.SendSetBytes(encoder, MetalNative.Selector("setBytes:length:atIndex:"), (nint)(&count), sizeof(uint), 1);
            MetalNative.SendSetBuffer(encoder, setBuffer, Reach.Buffer, 0, 2);
            MetalNative.SendSetBuffer(encoder, setBuffer, addresses, 0, 3);
            MetalNative.SendSetBuffer(encoder, setBuffer, results, 0, 4);
            var windows = new List<nint>();
            foreach (var id in nameWindows)
            {
                windows.Add(Memory.WindowBuffer(id));
            }

            var resident = windows.ToArray();
            fixed (nint* resources = resident)
            {
                MetalNative.SendUseResources(encoder, MetalNative.Selector("useResources:count:usage:"), resources,
                    (nuint)resident.Length, Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
            }

            MetalNative.SendDispatch(encoder, MetalNative.Selector("dispatchThreads:threadsPerThreadgroup:"),
                new MtlSize { Width = (nuint)Probes.Length, Height = 1, Depth = 1 },
                new MtlSize { Width = (nuint)Probes.Length, Height = 1, Depth = 1 });
            MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
            _harness.Submit();
            var bytes = _harness.Read(results, 0, (ulong)(Probes.Length * sizeof(uint)));
            return MemoryMarshal.Cast<byte, uint>(bytes).ToArray();
        }

        public void Dispose()
        {
            MetalNative.SendVoid(_pipeline, MetalNative.Selector("release"));
            _harness.Dispose();
            Memory.Dispose();
            _backing.Dispose();
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Device-address windows run on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static Lazy<nint> Device => MetalGuestWindowResidencyTests.Device;
}
