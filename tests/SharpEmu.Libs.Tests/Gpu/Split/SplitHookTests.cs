// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.Tests.Memory.GpuMemory;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Split;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class SplitHookCollection
{
    public const string Name = "SplitHooks";
}

/// <summary>
/// The seams the two-process split hooks into, each with nothing on the other side: the page
/// summary's watch observer, the span observer, the video-out bridge, the interrupt redirect, the
/// shader-header and fused-program mirrors, and the window input snapshot.
/// </summary>
[Collection(SplitHookCollection.Name)]
public sealed class SplitHookTests : IDisposable
{
    private const GuestPageProtection ReadWrite = GuestPageProtection.Read | GuestPageProtection.Write;

    private readonly Action<string> _previousFatal = PageGuard.OnFatal;
    private readonly List<string> _fatals = new();

    public SplitHookTests()
    {
        PageGuard.OnFatal = _fatals.Add;
    }

    public void Dispose()
    {
        PageGuard.OnFatal = _previousFatal;
        VideoOutBackendServices.Bridge = null;
        EndOfPipeEvents.Redirect = null;
        AgcBackendServices.ShaderHeaderObserver = null;
        Gen5ShaderTranslator.FusedProgramObserver = null;
    }

    [Fact]
    public void TheWatchObserverHearsEachPagesFirstAndLastWatchBeforeItsProtection()
    {
        var space = new ObservedAddressSpace();
        using var guard = new PageGuard(space);

        guard.AddWatch(0x10000, 0x2000, blockReads: false);
        guard.AddWatch(0x10000, 0x1000, blockReads: true);
        guard.RemoveWatch(0x10000, 0x1000, blockReads: false);
        guard.RemoveWatch(0x10000, 0x1000, blockReads: true);
        guard.RemoveWatch(0x11000, 0x1000, blockReads: false);

        Assert.Equal(
            new[]
            {
                "watch 0x10000 True", "watch 0x11000 True", "protect 0x10000+0x2000 Read",
                // The read watch joins a watched page and leaves it watched: no summary change.
                "protect 0x10000+0x1000 None",
                "watch 0x10000 False", "protect 0x10000+0x1000 Read, Write",
                "watch 0x11000 False", "protect 0x11000+0x1000 Read, Write",
            },
            space.Events);
        Assert.Empty(_fatals);
    }

    [Fact]
    public void ASpanObserverGetsTheSpansSoFarThenEveryChange()
    {
        using var memory = new GuestGpuMemory(new RecordingAddressSpace());
        memory.Register(0x100000, 0x4000, ReadWrite);
        memory.NoteProtected(0x101000, 0x1000, GuestPageProtection.Read);
        var observer = new RecordingSpanObserver();

        memory.AttachSpanObserver(observer);
        Assert.Equal(
            new[]
            {
                "register 0x100000+0x1000 Read, Write", "register 0x101000+0x1000 Read",
                "register 0x102000+0x2000 Read, Write",
            },
            observer.Events);

        observer.Events.Clear();
        memory.Register(0x200000, 0x1000, GuestPageProtection.Read);
        Assert.True(memory.NoteProtected(0x200000, 0x1000, ReadWrite));
        Assert.False(memory.NoteProtected(0x900000, 0x1000, ReadWrite));
        Assert.Equal(new[] { "register 0x200000+0x1000 Read", "protect 0x200000+0x1000 Read, Write" }, observer.Events);

        memory.AttachSpanObserver(null);
        memory.Register(0x300000, 0x1000, ReadWrite);
        Assert.Equal(2, observer.Events.Count);
    }

    [Fact]
    public void AnObserverThatAppliesTheProtectionsReplacesTheManagersOwn()
    {
        var space = new RecordingAddressSpace();
        using var memory = new GuestGpuMemory(space);
        var observer = new RecordingSpanObserver { Applies = true };
        memory.AttachSpanObserver(observer);

        memory.Register(0x400000, 0x2000, ReadWrite);
        Assert.True(memory.NoteProtected(0x400000, 0x1000, GuestPageProtection.Read));
        Assert.Empty(space.Protects);
        Assert.Equal(GuestPageProtection.Read, memory.Pages.Permissions.Lookup(0x400000));

        // One that does not (a failed call) leaves them to the guard, as without an observer.
        observer.Applies = false;
        memory.Register(0x500000, 0x1000, ReadWrite);
        Assert.Equal(new[] { (0x500000UL, 0x1000UL, ReadWrite) }, space.Protects);
    }

    [Fact]
    public void TheVideoOutBridgeTakesEveryStateCall()
    {
        var bridge = new RecordingBridge();
        VideoOutBackendServices.Bridge = bridge;

        Assert.Equal(0, VideoOutBackendServices.TryReserveFlipRequest(1, 2, 3, 4, gpuQueued: true, out var requestId));
        Assert.Equal(77ul, requestId);
        Assert.True(VideoOutBackendServices.TryGetDisplayBufferInfo(1, 2, out var buffer));
        Assert.Equal(0x5000ul, buffer.Address);
        Assert.True(VideoOutBackendServices.CanPresentFlip(77, 9));
        Assert.Equal(19, VideoOutBackendServices.NextFlipPresentTimestamp(77, 9));
        Assert.True(VideoOutBackendServices.IsFlipDone(1, 2));
        Assert.Equal("PPSA00000", VideoOutBackendServices.GetApplicationTitleId());
        Assert.True(VideoOutBackendServices.IsHdrOutputRequested);
        VideoOutBackendServices.CompleteFlipOnGpu(77);
        VideoOutBackendServices.MarkFlipPresented(77);
        VideoOutBackendServices.DiscardFlip(78);
        VideoOutBackendServices.CancelFlip(79);
        VideoOutBackendServices.CancelOutstandingFlips();
        VideoOutBackendServices.NotifyPresentationWindowClosed();
        VideoOutBackendServices.SetSelectedGpuName("Apple M5");
        VideoOutBackendServices.TraceGpuFlip(null!, 1, 2, 3, 4, 0x5000, 1UL);

        Assert.Equal(
            new[]
            {
                "reserve 1 2 3 4 True", "buffer 1 2", "can 77 9", "next 77 9", "done 1 2", "title", "hdr",
                "gpu 77", "presented 77", "discard 78", "cancel 79", "cancel-all", "closed", "gpu-name Apple M5",
                "trace 1 2 0x5000",
            },
            bridge.Calls);

        // A property of the format alone stays local.
        Assert.False(VideoOutBackendServices.IsHdrPixelFormat(0x80000000));
    }

    [Fact]
    public void AnInterruptGoesToTheRedirectWhenThereIsOne()
    {
        var redirect = new RecordingSink();
        EndOfPipeEvents.Redirect = redirect;
        new EndOfPipeEvents().TriggerInterrupt(42, 7);
        Assert.Equal(new[] { (42, 7u) }, redirect.Interrupts);
    }

    [Fact]
    public void ShaderHeadersAreReportedSnapshottedAndMirrored()
    {
        var reported = new List<(ulong, ulong)>();
        AgcBackendServices.ShaderHeaderObserver = (code, header) => reported.Add((code, header));
        AgcBackendServices.MirrorShaderHeader(0xDEAD_0000, 0xBEEF_0000);

        Assert.Equal(0xBEEF_0000ul, AgcExports.GetShaderHeaderAddress(0xDEAD_0000));
        Assert.Contains(new KeyValuePair<ulong, ulong>(0xDEAD_0000, 0xBEEF_0000), AgcBackendServices.SnapshotShaderHeaders());
        // Mirroring is the GPU process's write: it reports nothing back.
        Assert.Empty(reported);
    }

    [Fact]
    public void FusedProgramsAreReportedAndSnapshotted()
    {
        var memory = new FakeCpuMemory(0x1000, 0x1000);
        var context = new CpuContext(memory, Generation.Gen5);
        var reported = new List<(ulong, ulong, ulong, ulong)>();
        Gen5ShaderTranslator.FusedProgramObserver = (entry, header, continuation, continuationHeader) =>
            reported.Add((entry, header, continuation, continuationHeader));

        Gen5ShaderTranslator.RegisterFusedProgram(context, 0x1000, 0x1100, 0x2000, 0x2100);
        Assert.Equal(new[] { (0x1000ul, 0x1100ul, 0x2000ul, 0x2100ul) }, Gen5ShaderTranslator.SnapshotFusedPrograms(memory));
        Assert.True(Gen5ShaderTranslator.UnregisterFusedProgram(context, 0x1000));
        Assert.False(Gen5ShaderTranslator.UnregisterFusedProgram(context, 0x1000));
        Assert.Empty(Gen5ShaderTranslator.SnapshotFusedPrograms(memory));
        Assert.Equal(new[] { (0x1000ul, 0x1100ul, 0x2000ul, 0x2100ul), (0x1000ul, 0ul, 0ul, 0ul) }, reported);
    }

    [Fact]
    public void WindowInputCopiesItsStateAndCountsChanges()
    {
        HostWindowInput.Connect();
        try
        {
            var before = HostWindowInput.Version;
            HostWindowInput.SetKey(30, down: true);
            HostWindowInput.SetGamepad("pad", new HostGamepadState(true, HostGamepadButtons.Cross, 1, 2, 3, 4, 5, 6));
            Assert.True(HostWindowInput.Version >= before + 2);

            Span<int> keys = stackalloc int[8];
            var count = HostWindowInput.CopyState(keys, out var focused, out var gamepad, out var name);
            Assert.Equal(1, count);
            Assert.Equal(30, keys[0]);
            Assert.True(focused);
            Assert.Equal(HostGamepadButtons.Cross, gamepad.Buttons);
            Assert.Equal("pad", name);

            HostWindowInput.SetKeys([40, 41]);
            count = HostWindowInput.CopyState(keys, out _, out _, out _);
            Assert.Equal(new[] { 40, 41 }, keys[..count].ToArray().Order());
        }
        finally
        {
            HostWindowInput.Disconnect();
        }
    }

    private sealed class ObservedAddressSpace : IGuestAddressSpace, IPageWatchObserver
    {
        public List<string> Events { get; } = new();

        public ulong ProtectionPageSize => 4096;

        public void OnPageWatchChanged(ulong page, bool watched) => Events.Add($"watch 0x{page:X} {watched}");

        public bool TryProtect(ulong address, ulong size, GuestPageProtection protection)
        {
            Events.Add($"protect 0x{address:X}+0x{size:X} {protection}");
            return true;
        }

        public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();

        public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();

        public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) => throw new NotSupportedException();

        public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();

        public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();

        public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
    }

    private sealed class RecordingSpanObserver : IGuestSpanObserver
    {
        public List<string> Events { get; } = new();

        public bool Applies { get; set; }

        public bool OnSpanRegistered(ulong address, ulong size, GuestPageProtection protection)
        {
            Events.Add($"register 0x{address:X}+0x{size:X} {protection}");
            return Applies;
        }

        public bool OnSpanProtected(ulong address, ulong size, GuestPageProtection protection)
        {
            Events.Add($"protect 0x{address:X}+0x{size:X} {protection}");
            return Applies;
        }
    }

    private sealed class RecordingSink : IEndOfPipeSink
    {
        public List<(int, uint)> Interrupts { get; } = new();

        public void TriggerInterrupt(int eventId, uint contextId) => Interrupts.Add((eventId, contextId));
    }

    private sealed class RecordingBridge : IVideoOutBackendBridge
    {
        public List<string> Calls { get; } = new();

        public bool IsHdrOutputRequested
        {
            get
            {
                Calls.Add("hdr");
                return true;
            }
        }

        public bool CanPresentFlip(ulong requestId, long timestamp)
        {
            Calls.Add($"can {requestId} {timestamp}");
            return true;
        }

        public long NextFlipPresentTimestamp(ulong requestId, long timestamp)
        {
            Calls.Add($"next {requestId} {timestamp}");
            return timestamp + 10;
        }

        public void CancelFlip(ulong requestId) => Calls.Add($"cancel {requestId}");

        public void CancelOutstandingFlips() => Calls.Add("cancel-all");

        public void CompleteFlip(ulong requestId) => Calls.Add($"complete {requestId}");

        public void CompleteFlipOnGpu(ulong requestId) => Calls.Add($"gpu {requestId}");

        public void DiscardFlip(ulong requestId) => Calls.Add($"discard {requestId}");

        public string GetApplicationTitleId()
        {
            Calls.Add("title");
            return "PPSA00000";
        }

        public int GetFlipEventCount(ulong requestId) => 0;

        public string GetWindowTitle() => "window";

        public bool IsFlipDone(int handle, int bufferIndex)
        {
            Calls.Add($"done {handle} {bufferIndex}");
            return true;
        }

        public bool IsFlipPresentationPending(ulong requestId) => false;

        public void MarkFlipPresented(ulong requestId) => Calls.Add($"presented {requestId}");

        public void NotifyPresentationWindowClosed() => Calls.Add("closed");

        public bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId) => false;

        public void ReportPresentedFrame() => Calls.Add("frame");

        public void SetSelectedGpuName(string gpuName) => Calls.Add($"gpu-name {gpuName}");

        public void TraceGpuFlip(int handle, int bufferIndex, int flipMode, long flipArg, ulong address, ulong requestId) =>
            Calls.Add($"trace {handle} {bufferIndex} 0x{address:X}");

        public bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out VideoOutExports.DisplayBufferInfo info)
        {
            Calls.Add($"buffer {handle} {bufferIndex}");
            info = new VideoOutExports.DisplayBufferInfo(0x5000, 0, 0, 1920, 1080, 1920);
            return true;
        }

        public int TryReserveFlipRequest(int handle, int bufferIndex, int flipMode, long flipArg, bool gpuQueued, out ulong requestId)
        {
            Calls.Add($"reserve {handle} {bufferIndex} {flipMode} {flipArg} {gpuQueued}");
            requestId = 77;
            return 0;
        }
    }
}
