// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Bytes a recorded GPU write may still change are refused to the resident read and compare,
/// as the pipeline host contract asks: the materializer then reads them through the word
/// read, which waits for the write. Once the write completes they read resident again.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong ResidentPending = GuestBase + 0x68000;

    [Fact]
    public void BytesAPendingGpuWriteCoversAreRefusedToResidentReadsUntilTheWriteCompletes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        rig.WriteBacking32(ResidentPending, 0x11111111);
        rig.WriteBacking32(ResidentPending + 4, 0x22222222);
        rig.WriteBacking32(ResidentPending + 0x100, 0x33333333);
        rig.WriteBacking32(ResidentPending + 0x104, 0x44444444);
        rig.AliasAccess = true;
        var recorded = new byte[8];
        Assert.True(host.TryReadResidentGuestBytes(ResidentPending, recorded, clean: false));

        ulong serial;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            serial = host.SubmissionSerial;
            host.NoteGuestMemoryWrite(ResidentPending + 4, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);

            // A run overlapping the pending word is refused, read or compared.
            var bytes = new byte[8];
            Assert.False(host.TryReadResidentGuestBytes(ResidentPending, bytes, clean: false));
            Assert.False(host.TryCompareResidentGuestBytes(
                [ResidentPending], [recorded.Length], [0], recorded));

            // A run elsewhere reads resident.
            Assert.True(host.TryReadResidentGuestBytes(ResidentPending + 0x100, bytes, clean: false));
            Assert.Equal(0x33333333u, BitConverter.ToUInt32(bytes, 0));
            Assert.Equal(0x44444444u, BitConverter.ToUInt32(bytes, 4));
            host.EndRendering();
        }

        // The word read waits for the write; afterwards the run reads resident again.
        host.BeginGuestWordReads();
        Assert.True(host.TryReadGuestWord(ResidentPending + 4, out _));
        host.EndGuestWordReads();
        Assert.True(host.CompletedSerial >= serial);
        var reread = new byte[8];
        Assert.True(host.TryReadResidentGuestBytes(ResidentPending, reread, clean: false));
        Assert.True(host.TryCompareResidentGuestBytes([ResidentPending], [reread.Length], [0], reread));
    }
}
