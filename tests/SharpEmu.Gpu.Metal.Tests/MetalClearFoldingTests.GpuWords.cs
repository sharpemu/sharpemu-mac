// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalClearFoldingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DisjointFinalWordDoesNotDrainAnOwedHostPublication(bool gpuWritten)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var host = rig.Host;
        host.GpuWordsEnabled = true;
        rig.EnableStablePublicationAliases();
        host.FinalGuestTablesEnabled = true;
        rig.WriteBacking32(PendingTable, 17);
        rig.WriteBacking32(LineTable, 0xA5A5A5A5);
        if (gpuWritten) host.FillBufferInOrder(LineTable, 0x2000, 0xA5A5A5A5);
        var serial = host.SubmissionSerial;
        host.Completions.Enqueue(serial, () => { entered.Set(); release.Wait(TimeSpan.FromSeconds(10)); });
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.AtomicAdd32, 1, PendingTable, 1));
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        // Open later unrelated work. Neither the publication nor the byte read may
        // commit it simply to observe the older resource.
        var recording = host.SubmissionSerial;
        Task<uint>? read = null;
        try
        {
            read = Task.Run(() =>
            {
                Assert.True(host.TryReadGuestWord(LineTable, out var value));
                return value;
            });
            Assert.True(read.Wait(TimeSpan.FromSeconds(3)), "Disjoint read waited for a host callback.");
            Assert.Equal(0xA5A5A5A5u, read.Result);
            Assert.Equal(recording, host.SubmissionSerial);
        }
        finally { release.Set(); read?.Wait(TimeSpan.FromSeconds(10)); }
        host.SynchronizeGuestTables();
        Assert.True(host.TryReadGuestWord(PendingTable, out var label));
        Assert.Equal(18u, label);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OverlappingHostPublicationCannotBeReplacedByItsGpuSignal(bool enabled)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var host = rig.Host;
        host.GpuWordsEnabled = enabled;
        rig.WriteBacking32(PendingTable, 17);
        var serial = host.SubmissionSerial;
        host.Completions.Enqueue(serial, () => { entered.Set(); release.Wait(TimeSpan.FromSeconds(10)); });
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.AtomicAdd32, 1, PendingTable, 1));
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        var read = Task.Run(() =>
        {
            Assert.True(host.TryReadGuestWord(PendingTable, out var value));
            return value;
        });
        try { Assert.False(read.Wait(TimeSpan.FromMilliseconds(100))); }
        finally { release.Set(); Assert.True(read.Wait(TimeSpan.FromSeconds(10))); }
        Assert.Equal(18u, read.Result);
    }
}
