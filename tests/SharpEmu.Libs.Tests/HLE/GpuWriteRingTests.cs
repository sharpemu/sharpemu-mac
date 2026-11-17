// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

[Collection(nameof(GpuWriteRingTests))]
public sealed class GpuWriteRingTests
{
    // Addresses no other test records, so a shared static ring still answers only for these.
    private const ulong Base = 0x7F12_3450_0000;

    [Fact]
    public void ReportsOnlyTheWritesThatIntersectARangeNewestFirst()
    {
        var was = GpuWriteRing.Enabled;
        GpuWriteRing.Enabled = true;
        try
        {
            GpuWriteRing.Record(Base + 0x0800, 0x400, 7, "gpu:image-texels", "image addr=0x1 format=R8G8B8A8Unorm");
            GpuWriteRing.Record(Base + 0x2000, 0x100, 8, "host-copy", "from=0x2");
            GpuWriteRing.Record(Base + 0x0C00, 0x10, 9, "gpu:storage", "dispatch", "cs#1");

            var lines = GpuWriteRing.Describe(Base + 0x0A00, Base + 0x0C01);

            Assert.Equal(2, lines.Count);
            Assert.Contains("kind=gpu:storage producer=dispatch cs#1", lines[0]);
            Assert.Contains("serial=7 src=0x0", lines[1]);
            Assert.Contains("kind=gpu:image-texels", lines[1]);
            Assert.Empty(GpuWriteRing.Describe(Base + 0x0C10, Base + 0x2000));

            // A label-sized write is not kept.
            GpuWriteRing.Record(Base + 0x3000, 8, 10, "gpu:eop_label", null);
            Assert.Empty(GpuWriteRing.Describe(Base + 0x3000, Base + 0x3008));
        }
        finally
        {
            GpuWriteRing.Enabled = was;
        }
    }

    [Fact]
    public void AWriteIsAlsoFoundByTheBackingItsAddressResolvedToWhenRecorded()
    {
        var was = GpuWriteRing.Enabled;
        var resolver = GpuWriteRing.BackingResolver;
        GpuWriteRing.Enabled = true;
        try
        {
            // The address maps backing 0x9_0000_0000 when the write is recorded, and nothing afterwards.
            GpuWriteRing.BackingResolver = address => address - (Base + 0xC000) + 0x9_0000_0000;
            GpuWriteRing.Record(Base + 0xC800, 0x400, 11, "gpu:image-texels", "image addr=0x3");
            GpuWriteRing.BackingResolver = _ => ulong.MaxValue;
            GpuWriteRing.Record(Base + 0xD800, 0x400, 12, "gpu:image-texels", "image addr=0x4");

            var lines = GpuWriteRing.Describe(0x9_0000_0A00, 0x9_0000_0A40, byBacking: true);

            Assert.Single(lines);
            Assert.Contains($"[0x{Base + 0xC800:X16}", lines[0]);
            Assert.Contains("backing=0x900000800 serial=11", lines[0]);
            Assert.Empty(GpuWriteRing.Describe(0x9_0000_0C00, 0x9_0000_1000, byBacking: true));
            Assert.Contains("backing=? serial=12", Assert.Single(GpuWriteRing.Describe(Base + 0xD800, Base + 0xD810)));
        }
        finally
        {
            GpuWriteRing.BackingResolver = resolver;
            GpuWriteRing.Enabled = was;
        }
    }

    [Fact]
    public void ACopyIntoARangeNamesTheSourceBytesAndTheBackingTheyMappedThen()
    {
        var was = GpuWriteRing.HleEnabled;
        var resolver = GpuWriteRing.BackingResolver;
        GpuWriteRing.HleEnabled = true;
        try
        {
            GpuWriteRing.BackingResolver = address => address - Base + 0x5_0000_0000;
            GpuWriteRing.RecordHle(Base + 0x20000, 0x8000, Base + 0x90000, GpuWriteRing.CopyKind, null);
            GpuWriteRing.RecordHle(Base + 0x20000, 0x8000, 0x1234, "hle:ampr-read", "/game/data.bin");

            var sources = GpuWriteRing.CopySourcesOf(Base + 0x21000, Base + 0x21400);

            var source = Assert.Single(sources);
            Assert.Equal((Base + 0x91000, 0x400UL, 0x5_0009_1000UL), source);
            Assert.Empty(GpuWriteRing.CopySourcesOf(Base + 0x28000, Base + 0x29000));
        }
        finally
        {
            GpuWriteRing.BackingResolver = resolver;
            GpuWriteRing.HleEnabled = was;
        }
    }

    [Fact]
    public void HleWritesNameTheirSourceAndThreadAndFollowTheirOwnSwitch()
    {
        var was = GpuWriteRing.HleEnabled;
        try
        {
            GpuWriteRing.HleEnabled = false;
            GpuWriteRing.RecordHle(Base + 0xB000, 0x400, 0x1234, "hle:memcpy", null);
            Assert.Empty(GpuWriteRing.Describe(Base + 0xB000, Base + 0xB400));

            GpuWriteRing.HleEnabled = true;
            GpuWriteRing.RecordHle(Base + 0xB000, 0x400, 0x1234, "hle:memcpy", null);
            GpuWriteRing.RecordHle(Base + 0xB400, 0x8000, 0x20000, "hle:ampr-read", "/game/data.bin");

            var lines = GpuWriteRing.Describe(Base + 0xB3F0, Base + 0xB410);

            Assert.Equal(2, lines.Count);
            Assert.Contains("src=0x20000", lines[0]);
            Assert.Contains("kind=hle:ampr-read producer=/game/data.bin", lines[0]);
            Assert.Contains($"src=0x1234 tid={Environment.CurrentManagedThreadId} kind=hle:memcpy", lines[1]);
        }
        finally
        {
            GpuWriteRing.HleEnabled = was;
        }
    }

    [Fact]
    public void RecordsNothingWhileOff()
    {
        var was = GpuWriteRing.Enabled;
        GpuWriteRing.Enabled = false;
        try
        {
            GpuWriteRing.Record(Base + 0x9000, 0x400, 1, "gpu:copy", null);
            Assert.Empty(GpuWriteRing.Describe(Base + 0x9000, Base + 0x9400));
        }
        finally
        {
            GpuWriteRing.Enabled = was;
        }
    }

    [Fact]
    public void FaultRangesLeftByOneProcessAreAnsweredAndRemovedByTheOther()
    {
        var was = GpuWriteRing.Enabled;
        GpuWriteRing.Enabled = true;
        var path = GpuWriteRing.FaultRangesPath(Environment.ProcessId);
        try
        {
            GpuWriteRing.Record(Base + 0x5000, 0x400, 3, "gpu:image-texels", "image");
            GpuWriteRing.ReportFaultRanges([(Base + 0x4F00, 0x200UL, "e0a=[[rbp]]")]);
            Assert.True(File.Exists(path));
            Assert.StartsWith($"{Base + 0x4F00:X} 200 e0a=[[rbp]]", File.ReadAllText(path));

            GpuWriteRing.ReportFaultRangesOf(Environment.ProcessId);
            Assert.False(File.Exists(path));
        }
        finally
        {
            GpuWriteRing.Enabled = was;
            File.Delete(path);
        }
    }
}
