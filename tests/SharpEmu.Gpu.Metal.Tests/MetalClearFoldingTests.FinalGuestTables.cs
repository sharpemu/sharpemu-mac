// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalClearFoldingTests
{
    [Fact]
    public void AHostPublicationVetoesResidentReadsAndCertificateCaptureUntilItLands()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var host = rig.Host;
        host.FinalGuestTablesEnabled = true;
        rig.WriteBacking32(PendingTable, 17);
        var serial = host.SubmissionSerial;
        host.Completions.Enqueue(serial, () => { entered.Set(); release.Wait(TimeSpan.FromSeconds(10)); });
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.AtomicAdd32, 1, PendingTable, 1));
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        try
        {
            var source = (IResourceDependencySource)host;
            Assert.False(source.DependencyReadsReady);
            Assert.False(source.TryCaptureDependency(PendingTable, BitConverter.GetBytes(17u), false, out _));
            Assert.Equal(ResourceCaptureFailure.PendingWriter, source.CaptureFailure.Reason);
            Assert.False(host.TryReadResidentGuestBytes(PendingTable, new byte[4], false));
            Assert.False(host.TryReadCleanGuestWord(PendingTable, out _));
        }
        finally { release.Set(); }
        host.SynchronizeGuestTables();
        Assert.True(((IResourceDependencySource)host).DependencyReadsReady);
        Assert.True(host.TryReadGuestWord(PendingTable, out var value));
        Assert.Equal(18u, value);
    }

    [Fact]
    public void ADrainWaitsForAPublicationAlreadyRemovedFromThePendingList()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var finished = new ManualResetEventSlim();
        var host = rig.Host;
        var serial = host.SubmissionSerial;
        host.Completions.Enqueue(serial, () => { entered.Set(); release.Wait(TimeSpan.FromSeconds(10)); });
        host.FlushCommandBuffer();
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        var waiter = new Thread(() => { host.Completions.WaitFor(serial); finished.Set(); });
        waiter.Start();
        try { Assert.False(finished.Wait(TimeSpan.FromMilliseconds(100))); }
        finally { release.Set(); Assert.True(waiter.Join(TimeSpan.FromSeconds(10))); }
        Assert.True(finished.IsSet);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ARecycledCopySourceCannotAnswerAFinalDescriptorRead(bool deferBindingWords)
    {
        if (!RequireDevice()) return;
        var previous = DeviceBindingLoads.Enabled;
        DeviceBindingLoads.Enabled = deferBindingWords;
        try
        {
            using var rig = new Rig(Device.Value);
            var serial = RecordPendingCopy(rig);
            var host = rig.Host;
            host.FinalGuestTablesEnabled = true;
            rig.WriteBacking32(CopySource, 0x3F000000);
            Assert.False(host.TryReadResidentGuestBytes(CopyDestination, new byte[16], false));
            Assert.False(host.TryReadCleanGuestWord(CopyDestination, out _));
            if (deferBindingWords) Assert.True(host.ResourceWordPending!(CopyDestination));
            else Assert.Null(host.ResourceWordPending);
            Assert.True(host.TryReadGuestWord(CopyDestination, out var word));
            // The test records a copy intent without executing a dispatch. Waiting must
            // read the actual destination, never substitute the recycled source.
            Assert.Equal(0x11111111u, word);
            Assert.True(host.CompletedSerial >= serial);
            Assert.Equal(0, host.CopyReadCounters.Reads);
            Assert.Equal("memory-final", host.LastGuestTableRead.Path);
            Assert.Equal(serial, host.LastGuestTableRead.PendingSerial);
        }
        finally { DeviceBindingLoads.Enabled = previous; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FinalWordReadsObserveCpuRecyclingWithinAMaterializationScope(bool readHints)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = true;
        host.FinalReadHintsEnabled = readHints;
        host.BeginGuestWordReads();
        try
        {
            rig.WriteBacking32(LineTable, 0x12345678);
            Assert.True(host.TryReadGuestWord(LineTable, out var before));
            Assert.Equal(0x12345678u, before);
            // Raw guest CPU writes do not bump the Metal GPU/overlay versions.
            rig.WriteBacking32(LineTable, 0x87654321);
            Assert.True(host.TryReadGuestWord(LineTable, out var after));
            Assert.Equal(0x87654321u, after);
            Assert.True(host.TryReadCleanGuestWord(LineTable, out var clean));
            Assert.Equal(0x87654321u, clean);
            var resident = new byte[sizeof(uint)];
            Assert.True(host.TryReadResidentGuestBytes(LineTable, resident, false));
            Assert.Equal(0x87654321u, BitConverter.ToUInt32(resident));
        }
        finally { host.EndGuestWordReads(); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FinalReadHintsExpireWhenAGpuWriterAppearsInTheSameScope(bool readHints)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = true;
        host.FinalReadHintsEnabled = readHints;
        host.BeginGuestWordReads();
        try
        {
            rig.WriteBacking32(CopyDestination, 17);
            Assert.True(host.TryReadGuestWord(CopyDestination, out var before));
            Assert.Equal(17u, before);

            var serial = RecordPendingCopy(rig);
            host.FinalGuestTablesEnabled = true;
            rig.WriteBacking32(CopyDestination, 29);
            Assert.False(host.TryReadCleanGuestWord(CopyDestination, out _));
            Assert.False(host.TryReadResidentGuestBytes(CopyDestination, new byte[sizeof(uint)], false));
            Assert.True(host.TryReadGuestWord(CopyDestination, out var after));
            Assert.Equal(29u, after);
            Assert.True(host.CompletedSerial >= serial);
            Assert.Equal(0, host.CopyReadCounters.Reads);
        }
        finally { host.EndGuestWordReads(); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FinalDescriptorReadsWaitForPriorWritersOnTheIssueConsumer(bool split)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = true;
        var ordered = new OrderedCommandStreamHost(host, split);
        var consumer = split ? new Thread(() => ordered.Consume()) : null;
        consumer?.Start();
        try
        {
            ordered.ExecuteGuestMemoryOperation(() =>
            {
                // GPU transfer, partial CP overlay, then a label. Every final read
                // waits directly on the consumer, with no relay request to itself.
                host.FillBufferInOrder(PendingTable, 0x2000, 0xA5A5A5A5);
                Assert.False(host.TryReadResidentGuestBytes(PendingTable, new byte[16], false));
                Assert.True(host.TryReadGuestWord(PendingTable, out var filled));
                Assert.Equal(0xA5A5A5A5u, filled);
                host.RecordCommandProcessorWrite(PendingTable + 1, new byte[] { 0x12, 0x34 }, "TABLE_WRITE");
                Assert.False(host.TryReadResidentGuestBytes(PendingTable, new byte[4], false));
                Assert.True(host.TryReadGuestWord(PendingTable, out var patched));
                Assert.Equal(0xA53412A5u, patched);
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, PendingTable, 0xFACE1234));
                Assert.False(host.TryReadCleanGuestWord(PendingTable, out _));
                Assert.True(host.TryReadGuestWord(PendingTable, out var label));
                Assert.Equal(0xFACE1234u, label);
                host.SynchronizeGuestTables();
            });
        }
        finally
        {
            ordered.Stop();
            if (consumer is not null) Assert.True(consumer.Join(TimeSpan.FromSeconds(10)));
        }
    }
}
