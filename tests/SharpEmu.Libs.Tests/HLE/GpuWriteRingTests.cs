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
