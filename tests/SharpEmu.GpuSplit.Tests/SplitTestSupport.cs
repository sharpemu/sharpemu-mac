// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Host.MacOS;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.GpuSplit.Tests;

/// <summary>The split is Darwin-only (Mach rendezvous, shm, ulock).</summary>
public sealed class MacOSFactAttribute : FactAttribute
{
    public MacOSFactAttribute()
    {
        if (!OperatingSystem.IsMacOS())
        {
            Skip = "The two-process split exists only on macOS.";
        }
    }
}

// Process-wide state (the backing switch, Mach names, the replica window at 16 TiB, the video-out
// bridge and the hooks) is shared by every test here: they run one at a time.
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class SplitCollection
{
    public const string Name = "Split";
}

internal static class Wait
{
    public static void Until(Func<bool> condition, string what, int milliseconds = 10_000)
    {
        var clock = Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.ElapsedMilliseconds > milliseconds)
            {
                Assert.Fail($"Timed out waiting for {what}.");
            }

            Thread.Sleep(2);
        }
    }
}

/// <summary>
/// A shareable guest backing in this process, as the guest process makes it under
/// <c>SHARPEMU_GPU_PROCESS=1</c>, with one guest view of <see cref="ViewBytes"/> at
/// <see cref="Address"/> (backing offset <see cref="ViewOffset"/>).
/// </summary>
internal sealed unsafe class SharedGuestMemory : IDisposable
{
    public const ulong Chunk = 0x10000;
    public const ulong ViewBytes = 4 * Chunk;
    public const ulong ViewOffset = 8 * Chunk;
    public const ulong BackingBytes = 256 * Chunk;

    private readonly ulong _holeBytes;

    public SharedGuestMemory(bool chunked = false)
    {
        Host = MacOSHost.CreateViewMemory();
        var backing = Environment.GetEnvironmentVariable("SHARPEMU_GUEST_BACKING");
        var shareable = Environment.GetEnvironmentVariable("SHARPEMU_GUEST_BACKING_SHAREABLE");
        try
        {
            Environment.SetEnvironmentVariable("SHARPEMU_GUEST_BACKING", chunked ? null : "shm");
            Environment.SetEnvironmentVariable("SHARPEMU_GUEST_BACKING_SHAREABLE", "1");
            Views = new SharedBackingViews(Host, BackingBytes);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPEMU_GUEST_BACKING", backing);
            Environment.SetEnvironmentVariable("SHARPEMU_GUEST_BACKING_SHAREABLE", shareable);
        }

        Assert.True(Views.IsAvailable);
        Export = Assert.IsType<GuestBackingExport>(Views.BackingExport);
        Table = Assert.IsType<GuestViewTable>(Views.ViewTable);
        _holeBytes = (ViewBytes + Host.Granularity - 1) & ~(Host.Granularity - 1);
        Address = Host.FindUnmappedRange(0xF0_0000_0000, _holeBytes, 0xFC_0000_0000);
        Assert.NotEqual(0UL, Address);
        Assert.Equal(Address, Host.ReserveHole(Address, _holeBytes));
        Assert.True(Views.TryMapReservedRange(Address, ViewBytes, ViewOffset, HostPageProtection.ReadWrite, out _));
    }

    public IHostViewMemory Host { get; }

    public SharedBackingViews Views { get; }

    public GuestBackingExport Export { get; }

    public GuestViewTable Table { get; }

    /// <summary>The guest address of the view.</summary>
    public ulong Address { get; }

    public ulong* Pointer(ulong offset) => (ulong*)(Address + offset);

    /// <summary>The backing as the GPU process maps it: mirror, read-only table and replica.</summary>
    public (HostBackingObject Mirror, HostSharedRegion Table, GuestReplica Replica) MapLikeTheGpuProcess()
    {
        var mirrorHost = MacOSHost.CreateGuestMirrorMemory();
        Assert.True(mirrorHost.TryMapMirror(Export, out var mirror));
        Assert.True(mirrorHost.TryMapSharedRegion(Table.Region.Descriptor, Table.Region.Size, writable: false, out var table));
        var replica = GuestReplica.TryAttach(mirrorHost, mirror!, table!, GuestReplica.DefaultBase, windowStart: 0xF0_0000_0000, windowEnd: 0xFC_0000_0000);
        Assert.NotNull(replica);
        return (mirror!, table!, replica!);
    }

    public void Dispose()
    {
        if (Views.Contains(Address, ViewBytes))
        {
            _ = Views.Unmap(Address, ViewBytes, out _);
        }

        _ = Host.FreeHole(Address, _holeBytes);
        Views.Dispose();
    }
}

/// <summary>The guest process's address space as far as the split touches it: protections, recorded.</summary>
internal sealed class RecordingGuestSpace : IGuestAddressSpace, IGuestDerivedProtectionAddressSpace
{
    public ConcurrentQueue<(ulong Address, ulong Size, GuestPageProtection Guest, GuestPageProtection Access)> Protections { get; } = new();

    public ulong ProtectionPageSize => TrackerLayout.PageBytes;

    public bool Has(ulong address, GuestPageProtection access) =>
        Protections.Any(protection => protection.Address <= address && address < protection.Address + protection.Size && protection.Access == access);

    public (ulong Address, ulong Size, GuestPageProtection Guest, GuestPageProtection Access) Last(ulong address) =>
        Protections.Last(protection => protection.Address <= address && address < protection.Address + protection.Size);

    public bool TryApplyDerivedProtection(ulong address, ulong size, GuestPageProtection guestAccess, GuestPageProtection derivedAccess,
        bool ordinaryEnrollment)
    {
        Protections.Enqueue((address, size, guestAccess, derivedAccess));
        return true;
    }

    public bool TryProtect(ulong address, ulong size, GuestPageProtection protection)
    {
        Protections.Enqueue((address, size, protection, protection));
        return true;
    }

    public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();

    public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();

    public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) =>
        throw new NotSupportedException();

    public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();

    public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();

    public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
}

/// <summary>The guest's video-out state, event queues and input, recorded.</summary>
internal sealed class RecordingGuestSide : ISplitGuestSide
{
    public ConcurrentQueue<string> Calls { get; } = new();

    public ConcurrentQueue<(GpuSplitFlipEvent Kind, ulong RequestId)> FlipEvents { get; } = new();

    public ConcurrentQueue<(GpuSplitNotice Kind, string Text)> Notices { get; } = new();

    public ConcurrentQueue<(int EventId, uint ContextId)> Interrupts { get; } = new();

    public ConcurrentQueue<int[]> Inputs { get; } = new();

    public string? LostReason { get; private set; }

    public bool Hdr { get; set; } = true;

    public void TriggerInterrupt(int eventId, uint contextId) => Interrupts.Enqueue((eventId, contextId));

    public void FlipEvent(GpuSplitFlipEvent kind, ulong requestId) => FlipEvents.Enqueue((kind, requestId));

    public void Notice(GpuSplitNotice kind, string text) => Notices.Enqueue((kind, text));

    public void Input(bool focused, ReadOnlySpan<int> keys, in HostGamepadState gamepad, string? gamepadName) =>
        Inputs.Enqueue(keys.ToArray());

    public int TryReserveFlipRequest(int handle, int bufferIndex, int flipMode, long flipArg, out ulong requestId)
    {
        Calls.Enqueue($"reserve {handle} {bufferIndex} {flipMode} {flipArg}");
        requestId = 77;
        return 0;
    }

    public bool IsFlipDone(int handle, int bufferIndex)
    {
        Calls.Enqueue($"done {handle} {bufferIndex}");
        return true;
    }

    public bool CanPresentFlip(ulong requestId, long timestamp)
    {
        Calls.Enqueue($"can {requestId} {timestamp}");
        return true;
    }

    public long NextFlipPresentTimestamp(ulong requestId, long timestamp)
    {
        Calls.Enqueue($"next {requestId} {timestamp}");
        return timestamp + 100;
    }

    public bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out VideoOutExports.DisplayBufferInfo info)
    {
        Calls.Enqueue($"buffer {handle} {bufferIndex}");
        info = new VideoOutExports.DisplayBufferInfo(0x5000, 0x8000_0000, 0, 1920, 1080, 1920, 3);
        return true;
    }

    public int GetFlipEventCount(ulong requestId)
    {
        Calls.Enqueue($"events {requestId}");
        return 2;
    }

    public bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId)
    {
        Calls.Enqueue($"release {bufferIndex} {requestId}");
        return bufferIndex < 0;
    }

    public bool IsFlipPresentationPending(ulong requestId)
    {
        Calls.Enqueue($"pending {requestId}");
        return true;
    }

    public bool IsHdrOutputRequested => Hdr;

    public string GetWindowTitle() => "SharpEmu · test window";

    public string GetApplicationTitleId() => "PPSA01234";

    public void OnGpuProcessLost(string reason) => LostReason = reason;
}

/// <summary>The GPU process's backend, recorded; it can run bracket work on a worker thread of its own.</summary>
internal sealed class RecordingBackend : IGuestGpuBackend
{
    public ConcurrentQueue<string> Calls { get; } = new();

    public ConcurrentQueue<(uint Queue, ulong Address, uint Dwords, ulong SubmissionId, ICpuMemory Memory)> Submissions { get; } = new();

    public ConcurrentQueue<byte[]> Frames { get; } = new();

    public ThreadRelay? Worker { get; set; }

    public int SubmissionDelayMilliseconds { get; set; }

    // Holds the waiting half of a split SubmitDone until set.
    public ManualResetEventSlim BoundaryRelease { get; } = new(true);

    public string BackendName => "Recording";

    public void EnsureStarted(uint width, uint height) => Calls.Enqueue($"start {width}x{height}");

    public bool TryCompileProgram(ShaderCompileRequest request, out IGuestCompiledShader? shader, out string error)
    {
        shader = null;
        error = "none";
        return false;
    }

    public IGuestCompiledShader GetDepthOnlyFragmentShader() => throw new NotSupportedException();

    public void HideSplashScreen() => Calls.Enqueue("hide-splash");

    public void Submit(byte[] bgraFrame, uint width, uint height)
    {
        Calls.Enqueue($"frame {width}x{height} {bgraFrame.Length}");
        Frames.Enqueue(bgraFrame);
    }

    public void SubmitGuestDraw(GuestDrawKind drawKind, uint width, uint height)
    {
    }

    public bool TrySubmitGuestImage(int videoOutHandle, int displayBufferIndex, ulong address, uint width, uint height, uint pitchInPixel,
        ulong flipRequestId)
    {
        Calls.Enqueue($"flip {videoOutHandle} {displayBufferIndex} 0x{address:X} {flipRequestId}");
        return flipRequestId % 2 == 0;
    }

    public void SubmitCommandStream(ICpuMemory memory, uint queue, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
    {
        if (SubmissionDelayMilliseconds > 0)
        {
            Thread.Sleep(SubmissionDelayMilliseconds);
        }

        Submissions.Enqueue((queue, address, dwordCount, submissionId, memory));
    }

    public IdleOutcome SubmitDone(ICpuMemory memory)
    {
        Calls.Enqueue($"done {Submissions.Count}");
        return IdleOutcome.Completed;
    }

    public IdleOutcome BeginSubmitDone(ICpuMemory memory, out Func<IdleOutcome>? wait)
    {
        Calls.Enqueue($"boundary {Submissions.Count}");
        wait = () =>
        {
            BoundaryRelease.Wait();
            Calls.Enqueue("boundary-waited");
            return IdleOutcome.Completed;
        };
        return IdleOutcome.Completed;
    }

    public void RunAfterPendingCommandStreams(Action work)
    {
        Calls.Enqueue("after-pending");
        if (Worker is { } worker)
        {
            worker.RunOnWorker(work);
        }
        else
        {
            work();
        }
    }

    public void RegisterKnownDisplayBuffer(ulong address, uint guestFormat)
    {
    }

    public bool IsGpuGuestImageAvailable(ulong address, uint format, uint numberType) => false;

    public void CountShaderCompilation() => Calls.Enqueue("count-shader");

    public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() => (11, 2.5, 3, 4);

    public void RequestClose() => Calls.Enqueue("close");
}

/// <summary>
/// A GPU worker stand-in: one thread that runs what it is handed, which counts as the GPU queue
/// thread, like the Metal worker's relay.
/// </summary>
internal sealed class ThreadRelay : IGpuQueueRelay, IDisposable
{
    private readonly BlockingCollection<Action> _work = new();
    private readonly Thread _thread;

    public ThreadRelay()
    {
        _thread = new Thread(() =>
        {
            foreach (var work in _work.GetConsumingEnumerable())
            {
                work();
            }
        })
        {
            IsBackground = true,
            Name = "test GPU worker",
        };
        _thread.Start();
    }

    public int ThreadId => _thread.ManagedThreadId;

    public bool IsGpuQueueThread => Environment.CurrentManagedThreadId == _thread.ManagedThreadId;

    public void RunOnWorker(Action work)
    {
        if (IsGpuQueueThread)
        {
            work();
            return;
        }

        using var done = new ManualResetEventSlim(false);
        _work.Add(() =>
        {
            try
            {
                work();
            }
            finally
            {
                done.Set();
            }
        });
        done.Wait();
    }

    public void Post(Action work) => RunOnWorker(work);

    public void RunOnGpuQueue(Action work) => RunOnWorker(work);

    public bool TryRunOnGpuQueue(Action work)
    {
        RunOnWorker(work);
        return true;
    }

    public void Dispose() => _work.CompleteAdding();
}

/// <summary>The GPU process's image store, recorded.</summary>
internal sealed class RecordingImageStore : IGuestImageStore
{
    public ConcurrentQueue<(string Call, ulong Address, ulong Size, int ThreadId)> Calls { get; } = new();

    public Func<ulong, ulong, bool>? OnFault { get; set; }

    public bool MarkCpuWrite(ulong address, ulong size)
    {
        Calls.Enqueue(("fault", address, size, Environment.CurrentManagedThreadId));
        return OnFault?.Invoke(address, size) ?? false;
    }

    public bool MarkPreciseCpuWrite(ulong address, ulong size)
    {
        Calls.Enqueue(("precise", address, size, Environment.CurrentManagedThreadId));
        return true;
    }

    public bool MarkCpuWriteLanded(ulong address, ulong size)
    {
        Calls.Enqueue(("landed", address, size, Environment.CurrentManagedThreadId));
        return true;
    }

    public void Unregister(ulong address, ulong size) =>
        Calls.Enqueue(("unregister", address, size, Environment.CurrentManagedThreadId));

    public bool Saw(string call, ulong address) => Calls.Any(entry => entry.Call == call && entry.Address == address);
}
