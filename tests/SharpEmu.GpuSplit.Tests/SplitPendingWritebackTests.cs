// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using Xunit;

namespace SharpEmu.GpuSplit.Tests;

/// <summary>
/// The GPU process's store for one image at the start of a page, kept as the Metal image cache
/// keeps its images (MetalImageCache.PendingWritebacks.cs): a render makes it GPU-modified, a
/// writeback is recorded under the lock (the image's claim ends, the writeback's begins) and lands
/// later, a managed write's call writes the image back or waits for the writeback before the write
/// lands, and the write's record marks the image stale (dropping the GPU's bytes if it is still
/// GPU-modified then). The pending summary is published as the lock is let go.
/// </summary>
internal sealed unsafe class WritebackImageStore : IGuestImageStore
{
    public const ulong ImageBytes = 2048;
    // What a render changes; the rest of the image keeps what the last reload read.
    public const int GpuBytes = 1024;

    private readonly object _gate = new();
    private readonly byte[] _texture = new byte[ImageBytes];
    private readonly ManualResetEventSlim _landed = new(true);
    private IPendingWritebackPages? _pending;
    private ulong _address;
    private bool _modified;
    private bool _stale = true;
    private bool _inFlight;
    private int _precise;
    private int _landedWrites;
    private int _lostGpuBytes;

    public int PreciseCalls => Volatile.Read(ref _precise);

    public int LandedWrites => Volatile.Read(ref _landedWrites);

    public int LostGpuBytes => Volatile.Read(ref _lostGpuBytes);

    /// <summary>Spins between the image's claim ending and its writeback's beginning, to widen that gap.</summary>
    public int GapSpins { get; set; }

    public void Attach(ulong address, IPendingWritebackPages pending)
    {
        _address = address;
        _pending = pending;
        pending.Activate();
    }

    public void Render(byte value)
    {
        lock (_gate)
        {
            if (_stale)
            {
                new ReadOnlySpan<byte>((void*)_address, (int)ImageBytes).CopyTo(_texture);
                _stale = false;
            }

            if (!_modified)
            {
                _pending!.Add(_address, ImageBytes);
                _modified = true;
            }

            _texture.AsSpan(0, GpuBytes).Fill(value);
            _pending!.Publish();
        }
    }

    /// <summary>The bytes the writeback will store, or null when the image holds nothing to write back.</summary>
    public byte[]? RecordWriteback()
    {
        lock (_gate)
        {
            byte[]? snapshot = null;
            if (_modified)
            {
                snapshot = (byte[])_texture.Clone();
                _modified = false;
                _pending!.Remove(_address, ImageBytes);
                Thread.SpinWait(GapSpins);
                _pending.Add(_address, ImageBytes);
                _inFlight = true;
                _landed.Reset();
            }

            _pending!.Publish();
            return snapshot;
        }
    }

    public void LandWriteback(byte[] snapshot)
    {
        snapshot.CopyTo(new Span<byte>((void*)_address, (int)ImageBytes));
        lock (_gate)
        {
            _inFlight = false;
            _pending!.Remove(_address, ImageBytes);
            _pending.Publish();
            _landed.Set();
        }
    }

    public bool MarkPreciseCpuWrite(ulong address, ulong size)
    {
        Interlocked.Increment(ref _precise);
        while (true)
        {
            _landed.Wait();
            lock (_gate)
            {
                if (_inFlight)
                {
                    continue;
                }

                if (_modified)
                {
                    _texture.CopyTo(new Span<byte>((void*)_address, (int)ImageBytes));
                    _modified = false;
                    _pending!.Remove(_address, ImageBytes);
                }

                _stale = true;
                _pending!.Publish();
                return true;
            }
        }
    }

    public bool MarkCpuWriteLanded(ulong address, ulong size)
    {
        Interlocked.Increment(ref _landedWrites);
        lock (_gate)
        {
            if (_modified)
            {
                // The write is in guest memory already: nothing may go under it, so the GPU's bytes are lost.
                Interlocked.Increment(ref _lostGpuBytes);
                _modified = false;
                _pending!.Remove(_address, ImageBytes);
            }

            _stale = true;
            _pending!.Publish();
            return true;
        }
    }

    public bool MarkCpuWrite(ulong address, ulong size) => false;

    public void Unregister(ulong address, ulong size)
    {
    }
}

[Collection(SplitCollection.Name)]
public sealed unsafe class SplitPendingWritebackTests
{
    private const GuestPageProtection ReadWrite = InProcessSplit.ReadWrite;

    private static InProcessSplit Start(WritebackImageStore store, out ulong image)
    {
        var split = new InProcessSplit(gpuStore: store);
        image = split.Guest.Address;
        split.GuestMemory.Register(image, SharedGuestMemory.ViewBytes, ReadWrite);
        var page = image;
        var watcher = new Thread(() => split.GpuMemory.Pages.AddWatch(page, 0x1000, blockReads: false));
        watcher.Start();
        Assert.True(watcher.Join(10_000));
        var pending = ((IPendingWritebackSource)split.Environment!.AddressSpace).PendingWritebacks;
        Assert.NotNull(pending);
        store.Attach(image, pending);
        return split;
    }

    // A managed write the way PhysicalVirtualMemory makes one.
    private static void ManagedWrite(InProcessSplit split, ulong address, ulong value)
    {
        Assert.True(split.GuestMemory.MarkCpuWrite(address, 8));
        *(ulong*)address = value;
        split.GuestMemory.MarkCpuWriteLanded(address, 8);
    }

    [MacOSFact]
    public void AWatchedWriteCallsOnlyWhereTheGpuHoldsBytesGuestMemoryLacks()
    {
        var store = new WritebackImageStore();
        using var split = Start(store, out var image);
        var slot = image + (ulong)WritebackImageStore.GpuBytes;

        // Watched, nothing GPU-written: no call, and the records invalidate the image.
        ManagedWrite(split, slot, 1);
        Wait.Until(() => store.LandedWrites >= 1, "the write's record");
        Assert.Equal(0, store.PreciseCalls);

        // GPU-written: the call writes the image back before the write lands over its page.
        store.Render(0x5A);
        ManagedWrite(split, slot, 2);
        Assert.Equal(1, store.PreciseCalls);
        Assert.True(new ReadOnlySpan<byte>((void*)image, WritebackImageStore.GpuBytes).IndexOfAnyExcept((byte)0x5A) < 0);
        Assert.Equal(2UL, *(ulong*)slot);

        // A recorded writeback not landed yet: the call waits for it.
        store.Render(0x6B);
        var snapshot = store.RecordWriteback()!;
        var writer = Task.Run(() => ManagedWrite(split, slot, 3));
        Assert.False(writer.Wait(100));
        store.LandWriteback(snapshot);
        Assert.True(writer.Wait(10_000));
        Assert.Equal(2, store.PreciseCalls);
        Assert.Equal(3UL, *(ulong*)slot);

        // Landed and published: no call again.
        ManagedWrite(split, slot, 4);
        _ = split.Session.Backend.SubmitDone(null!);
        Assert.Equal(2, store.PreciseCalls);
        Assert.Equal(0, store.LostGpuBytes);
    }

    // CPU writes racing a GPU writeback into the same image: neither side's bytes may be lost.
    [MacOSFact]
    public void ManagedWritesRacingAWritebackLoseNeitherTheGpusBytesNorTheirOwn()
    {
        const int Rounds = 1500;
        var store = new WritebackImageStore { GapSpins = 2_000 };
        using var split = Start(store, out var image);
        var random = new Random(18);
        for (var round = 1; round <= Rounds; round++)
        {
            var value = (byte)(round % 251 + 1);
            var slot = image + (ulong)WritebackImageStore.GpuBytes + (ulong)(round % 128) * 8;
            var cpuValue = 0xC0DE_0000_0000UL | (uint)round;
            store.Render(value);
            var gpuDelay = random.Next(0, 4_000);
            var landDelay = random.Next(0, 4_000);
            var cpuDelay = random.Next(0, 6_000);
            using var go = new ManualResetEventSlim();
            var gpu = Task.Run(() =>
            {
                go.Wait();
                Thread.SpinWait(gpuDelay);
                if (store.RecordWriteback() is { } snapshot)
                {
                    Thread.SpinWait(landDelay);
                    store.LandWriteback(snapshot);
                }
            });
            var cpu = Task.Run(() =>
            {
                go.Wait();
                Thread.SpinWait(cpuDelay);
                ManagedWrite(split, slot, cpuValue);
            });
            go.Set();
            Assert.True(Task.WaitAll([gpu, cpu], 10_000));
            _ = split.Session.Backend.SubmitDone(null!);

            Assert.Equal(cpuValue, *(ulong*)slot);
            var gpuBytes = new ReadOnlySpan<byte>((void*)image, WritebackImageStore.GpuBytes);
            Assert.True(gpuBytes.IndexOfAnyExcept(value) < 0, $"round {round}: the GPU's bytes were lost");
            Assert.Equal(0, store.LostGpuBytes);
        }

        // Both paths ran: writes that met a pending writeback called, the others went as records.
        Assert.InRange(store.PreciseCalls, 1, Rounds - 1);
    }
}
