// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalClearFoldingTests
{
    [Fact]
    public void FullWalkPendingQueriesAgreeBeforeAndAfterARecordedStoreCompletes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var word = host.DataWritePending;
        var range = host.DataRangeWritePending;
        Assert.NotNull(word); Assert.NotNull(range);
        rig.WriteBacking32(PendingLine, 0x12345678);
        ulong serial;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            serial = host.SubmissionSerial;
            host.NoteGuestMemoryWrite(GuestBase + 0x40000, 4, MetalGpuWriteOrigin.StorageBuffer);
            host.NoteGuestMemoryWrite(PendingLine, 4, MetalGpuWriteOrigin.StorageBuffer);
            Assert.True(word(PendingLine));
            Assert.True(range(PendingLine, 4));
            // Within the live write bounds, but outside every recorded write.
            Assert.False(word(LineTable));
            Assert.False(range(LineTable, 16));
            Assert.Equal(serial, host.SubmissionSerial);
            Assert.True(host.RenderEncoderOpen);
            host.EndRendering();
        }
        Assert.True(host.TryReadGuestWord(PendingLine, out var value));
        Assert.Equal(0x12345678u, value);
        Assert.True(host.CompletedSerial >= serial);
        Assert.False(word(PendingLine));
        Assert.False(range(PendingLine, 4));
    }

    [Fact]
    public void FullWalkPendingRangeKeepsUnboundedWritersConservativeWithoutLearningAWord()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var word = host.DataWritePending;
        var range = host.DataRangeWritePending;
        Assert.NotNull(word); Assert.NotNull(range);
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            host.NoteUnboundedGuestMemoryWrite();
            Assert.True(range(LineTable, 16));
            Assert.False(word(LineTable));
            Assert.True(host.RenderEncoderOpen);
            host.EndRendering();
        }
    }
}
