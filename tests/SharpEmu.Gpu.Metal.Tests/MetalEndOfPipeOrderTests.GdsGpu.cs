// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalEndOfPipeOrderTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GdsFillCopyAndCpuObservationKeepTheirPacketOrder(bool enabled)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.GpuWordsEnabled = enabled;
        host.FillBuffer(0, 16, 0x12345678, true);
        host.CopyBuffer(WindowedLabel, 0, 16, false, true);
        // The first copy must snapshot the original GDS bytes before this fill.
        host.FillBuffer(0, 16, 0xAABBCCDD, true);
        host.CopyBuffer(32, WindowedLabel, 16, true, false);
        host.FlushCommandBuffer();
        var recording = host.SubmissionSerial;
        Span<uint> words = stackalloc uint[12];
        host.ReadGds(words, 0, 12);
        if (enabled) Assert.Equal(recording, host.SubmissionSerial);
        Assert.All(words[..4].ToArray(), value => Assert.Equal(0xAABBCCDDu, value));
        Assert.All(words[8..].ToArray(), value => Assert.Equal(0x12345678u, value));
        Assert.Equal(0x12345678u, rig.Backing.Read32(WindowedLabel));
    }

    [Fact]
    public void GdsCopyBehindAHostReportRetainsTheCompletionSnapshot()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.GpuWordsEnabled = true;
        host.FillBuffer(0, 8, 0x12345678, true);
        host.RecordEndOfPipe(new(SharpEmu.Libs.Gpu.GpuCommands.EndOfPipeWriteKind.GdsWrite32,
            1, UnwindowedLabel, GdsWordOffset: 0, GdsWordCount: 2));
        // The non-windowed report is CPU-published. A later GDS fill must not
        // overwrite its source before that callback has copied the old words.
        host.FillBuffer(0, 8, 0xAABBCCDD, true);
        host.SynchronizeGuestTables();
        Assert.Equal(0x12345678u, rig.Backing.Read32(UnwindowedLabel));
        Span<uint> words = stackalloc uint[2];
        host.ReadGds(words, 0, 2);
        Assert.Equal(0xAABBCCDDu, words[0]);
    }
}
