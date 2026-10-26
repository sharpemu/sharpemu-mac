// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Host.MacOS;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.GpuSplit.Tests;

/// <summary>
/// Both halves of the split in one process, over one real region and one real guest memory
/// service: process A's session and proxies, process B's host service and address space, a
/// recording backend in place of Metal and a recording guest in place of the emulator.
/// </summary>
internal sealed class InProcessSplit : IDisposable
{
    public const GuestPageProtection ReadWrite = GuestPageProtection.Read | GuestPageProtection.Write;

    private readonly IDisposable _server;
    private readonly GpuRemoteChannel _hostChannel;
    private readonly GpuProcessClient _client;
    private readonly Thread _loop;

    public InProcessSplit(Action<GuestGpuMemory>? beforeStart = null, TimeSpan? inputPoll = null)
    {
        Guest = new SharedGuestMemory();
        _server = MacOSHost.ServeGuestMemory(Guest.Export, Guest.Table.Region, out var service);
        _hostChannel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("sesplit"), SplitSharedState.Layout);
        var hello = new HelloMessage { ProtocolVersion = GpuRemoteProtocol.Version, ProcessId = System.Environment.ProcessId, Translated = -1 };
        Assert.True(_hostChannel.Outbound.TryWrite((uint)GpuRemoteMessage.Hello, in hello));
        _client = GpuProcessClient.Attach(_hostChannel.Name, parentProcessId: 0, TimeSpan.FromSeconds(5));
        Host = new SplitGpuHost(_client.Channel, _client.OutboundLock, new SplitGpuHostOptions
        {
            ComposeBackend = environment =>
            {
                Environment = environment;
                environment.Memory.AttachStores(null, GpuStore);
                return Backend;
            },
            ConfigureVideo = options => ConfiguredVideo = options,
            QueryHdrOutputSupport = static () => true,
            InstallGlobals = false,
            InputPollInterval = inputPoll ?? TimeSpan.Zero,
        });
        _client.Service = Host;
        _loop = new Thread(() => _client.Run()) { IsBackground = true, Name = "test GPU host ring" };
        _loop.Start();

        Assert.True(_hostChannel.Inbound.WaitForData(5_000_000_000));
        Assert.True(_hostChannel.Inbound.TryRead(out var type, out _));
        Assert.Equal((uint)GpuRemoteMessage.HelloAck, type);
        _hostChannel.Inbound.Advance();

        GuestMemory = new GuestGpuMemory(GuestSpace);
        beforeStart?.Invoke(GuestMemory);
        Session = new SplitGpuSession(_hostChannel, new object(), GuestMemory, Guest.Table, GuestSide);
        Session.Begin(service, new HostVideoOptions { Width = 1600, Height = 900, VSync = false }, "PPSA01234", installGlobals: false);
        Assert.True(Host.IsReady);
    }

    public SharedGuestMemory Guest { get; }

    public RecordingGuestSpace GuestSpace { get; } = new();

    public RecordingGuestSide GuestSide { get; } = new();

    public GuestGpuMemory GuestMemory { get; }

    public SplitGpuSession Session { get; }

    public SplitGpuHost Host { get; }

    public RecordingBackend Backend { get; } = new();

    public RecordingImageStore GpuStore { get; } = new();

    public SplitGpuEnvironment? Environment { get; private set; }

    public HostVideoOptions? ConfiguredVideo { get; private set; }

    public GuestGpuMemory GpuMemory => Host.Memory!;

    /// <summary>Process B dies: what the parent watch reports in A.</summary>
    public void KillGpuSide() => _hostChannel.MarkPeerExited();

    public void Dispose()
    {
        Session.Dispose();
        _hostChannel.Outbound.TryWrite((uint)GpuRemoteMessage.Shutdown, ReadOnlySpan<byte>.Empty);
        _hostChannel.MarkPeerExited();
        _client.Channel.MarkPeerExited();
        _ = _loop.Join(5_000);
        Host.Dispose();
        Host.GuestMemory?.Dispose();
        _client.Dispose();
        _hostChannel.Dispose();
        _server.Dispose();
        Guest.Dispose();
    }
}

[Collection(SplitCollection.Name)]
public sealed unsafe class SplitInProcessTests
{
    private const GuestPageProtection ReadWrite = InProcessSplit.ReadWrite;

    [MacOSFact]
    public void TheStartReplaysSpansAndComposesTheBackendOverTheSharedMemory()
    {
        ulong address = 0;
        using var split = new InProcessSplit(memory =>
        {
            // Registered before the GPU process exists: the start replays it.
            address = 0xF0_0000_0000;
            memory.Register(address, 0x20000, ReadWrite);
            memory.NoteProtected(address + 0x10000, 0x10000, GuestPageProtection.Read);
        });

        Assert.True(split.GpuMemory.Covers(address, 0x20000));
        Assert.Equal(ReadWrite, split.GpuMemory.Pages.Permissions.Lookup(address));
        Assert.Equal(GuestPageProtection.Read, split.GpuMemory.Pages.Permissions.Lookup(address + 0x10000));
        Assert.Equal(1600, split.Environment!.VideoOptions!.Width);
        Assert.False(split.Environment.VideoOptions.VSync);
        Assert.Equal("PPSA01234", split.Host.VideoOut!.GetApplicationTitleId());
        Assert.True(split.Host.VideoOut.IsHdrOutputRequested);

        // A registration after the start crosses as it happens.
        split.GuestMemory.Register(split.Guest.Address, SharedGuestMemory.ViewBytes, ReadWrite);
        Assert.True(split.GpuMemory.Covers(split.Guest.Address, SharedGuestMemory.ViewBytes));
        Assert.True(split.GuestMemory.NoteProtected(split.Guest.Address, 0x1000, GuestPageProtection.Read));
        Assert.Equal(GuestPageProtection.Read, split.GpuMemory.Pages.Permissions.Lookup(split.Guest.Address));
    }

    [MacOSFact]
    public void AFrameBoundaryWaitsForTheSubmissionsBeforeIt()
    {
        using var split = new InProcessSplit();
        split.Backend.SubmissionDelayMilliseconds = 30;
        for (var index = 0; index < 3; index++)
        {
            split.Session.Backend.SubmitCommandStream(null!, queue: 0, address: 0x1000 + (ulong)index, dwordCount: 16, submissionId: (ulong)index, null);
        }

        var watch = Stopwatch.StartNew();
        Assert.Equal(IdleOutcome.Completed, split.Session.Backend.SubmitDone(null!));
        Assert.True(watch.ElapsedMilliseconds >= 50, $"SubmitDone returned after {watch.ElapsedMilliseconds} ms");
        Assert.Contains("done 3", split.Backend.Calls);
        var submissions = split.Backend.Submissions.ToArray();
        Assert.Equal(new ulong[] { 0, 1, 2 }, submissions.Select(submission => submission.SubmissionId));
        Assert.All(submissions, submission => Assert.Same(split.Host.AddressSpace, submission.Memory));
        Assert.True(split.Host.GuestMemory!.Replica.ReplayedSequence >= split.Guest.Table.LogSequence);
    }

    [MacOSFact]
    public void ASplitFrameBoundaryIsOrderedOnTheRingAndWaitsOutsideIt()
    {
        using var split = new InProcessSplit();
        var backend = split.Session.Backend;
        backend.SubmitCommandStream(null!, queue: 0, address: 0x1000, dwordCount: 16, submissionId: 1, null);
        backend.SubmitCommandStream(null!, queue: 0x20, address: 0x2000, dwordCount: 16, submissionId: 2, null);
        split.Backend.BoundaryRelease.Reset();

        // The ordering half only writes the record; the GPU side queues it behind both submissions.
        Assert.Equal(IdleOutcome.Completed, backend.BeginSubmitDone(null!, out var wait));
        Assert.NotNull(wait);
        var waiter = Task.Run(wait!);

        // A submission sent while the boundary is still waited on reaches the GPU side after it.
        backend.SubmitCommandStream(null!, queue: 0x20, address: 0x3000, dwordCount: 16, submissionId: 3, null);
        Wait.Until(() => split.Backend.Submissions.Count == 3, "the submission behind the boundary");
        Assert.Contains("boundary 2", split.Backend.Calls);
        Assert.False(waiter.IsCompleted);

        split.Backend.BoundaryRelease.Set();
        Assert.True(waiter.Wait(TimeSpan.FromSeconds(10)));
        Assert.Equal(IdleOutcome.Completed, waiter.Result);
        Assert.Contains("boundary-waited", split.Backend.Calls);
    }

    [MacOSFact]
    public void ExportFlipsAreAcceptedOrRefusedBack()
    {
        using var split = new InProcessSplit();
        Assert.True(split.Session.Backend.TrySubmitGuestImage(1, 2, 0x5000, 1920, 1080, 1920, flipRequestId: 8));
        Assert.True(split.Session.Backend.TrySubmitGuestImage(1, 3, 0x6000, 1920, 1080, 1920, flipRequestId: 7));
        Wait.Until(() => split.GuestSide.FlipEvents.Contains((GpuSplitFlipEvent.Refused, 7UL)), "the refused flip");
        Assert.DoesNotContain(split.GuestSide.FlipEvents, flip => flip.RequestId == 8);
        Assert.Contains("flip 1 2 0x5000 8", split.Backend.Calls);
    }

    [MacOSFact]
    public void WatchesCrossAsProtectionsAndFilterTheManagedWrites()
    {
        using var split = new InProcessSplit();
        var address = split.Guest.Address;
        split.GuestMemory.Register(address, SharedGuestMemory.ViewBytes, ReadWrite);
        // The GPU side owns the protections: the span's protection is applied once, by its request.
        Assert.Single(split.GuestSpace.Protections, protection => protection.Address == address);

        // B arms a watch: A applies it before AddWatch returns, and the summary shows it.
        var watcher = new Thread(() => split.GpuMemory.Pages.AddWatch(address, 0x1000, blockReads: false));
        watcher.Start();
        Assert.True(watcher.Join(10_000));
        Assert.Equal(GuestPageProtection.Read, split.GuestSpace.Last(address).Access);

        // A managed write under the watch crosses before it lands (a call) and after (a record).
        Assert.True(split.GuestMemory.MarkCpuWrite(address + 8, 8));
        Assert.True(split.GpuStore.Saw("precise", address + 8));
        split.GuestMemory.MarkCpuWriteLanded(address + 8, 8);
        Wait.Until(() => split.GpuStore.Saw("landed", address + 8), "the landed write");

        // Outside every watched group nothing crosses.
        Assert.True(split.GuestMemory.MarkCpuWrite(address + 3 * SharedGuestMemory.Chunk, 8));
        split.GuestMemory.MarkCpuWriteLanded(address + 3 * SharedGuestMemory.Chunk, 8);
        _ = split.Session.Backend.SubmitDone(null!);
        Assert.DoesNotContain(split.GpuStore.Calls, call => call.Address == address + 3 * SharedGuestMemory.Chunk);

        // A write fault: B's store drops the watch, and A has the page writable again before the
        // fault returns, so the retried store passes.
        split.GpuStore.OnFault = (faulted, _) =>
        {
            split.GpuMemory.Pages.RemoveWatch(faulted & ~0xFFFUL, 0x1000, blockReads: false);
            return true;
        };
        Assert.True(split.GuestMemory.TryResolveFault(FaultKind.Write, address + 16));
        Assert.True(split.GpuStore.Saw("fault", address + 16));
        Assert.Equal(ReadWrite, split.GuestSpace.Last(address).Access);
        Assert.True(split.Session.ProtectCounts.Applied >= 2);

        // A watch removed before the fault reaches the store: the store declines, and the page's
        // restoration lets the store retry once - not twice for the same restoration.
        var page = address + 0x2000;
        var watcherAgain = new Thread(() => split.GpuMemory.Pages.AddWatch(page, 0x1000, blockReads: false));
        watcherAgain.Start();
        Assert.True(watcherAgain.Join(10_000));
        var remover = new Thread(() => split.GpuMemory.Pages.RemoveWatch(page, 0x1000, blockReads: false));
        remover.Start();
        Assert.True(remover.Join(10_000));
        split.GpuStore.OnFault = null;
        Assert.True(split.GuestMemory.TryResolveFault(FaultKind.Write, page + 8));
        Assert.False(split.GuestMemory.TryResolveFault(FaultKind.Write, page + 8));
    }

    [MacOSFact]
    public void AMappingChangeParksTheWorkerAndUnregistersOnIt()
    {
        using var split = new InProcessSplit();
        using var worker = new ThreadRelay();
        var address = split.Guest.Address;
        split.GuestMemory.Register(address, SharedGuestMemory.ViewBytes, ReadWrite);
        split.GpuMemory.AttachGpuQueue(worker, null);
        try
        {
            var ranInside = false;
            split.GuestMemory.RunMappingChange(() =>
            {
                // A's change runs while B's worker is parked, and A is the GPU queue thread meanwhile.
                ranInside = true;
                split.GuestMemory.Unregister(address + SharedGuestMemory.Chunk, SharedGuestMemory.Chunk);
            });

            Assert.True(ranInside);
            var unregistered = Assert.Single(split.GpuStore.Calls, call => call.Call == "unregister");
            Assert.Equal(address + SharedGuestMemory.Chunk, unregistered.Address);
            Assert.Equal(worker.ThreadId, unregistered.ThreadId);
            Assert.False(split.GpuMemory.Covers(address + SharedGuestMemory.Chunk, 8));
            Assert.False(split.GuestMemory.Covers(address + SharedGuestMemory.Chunk, 8));
            Assert.True(split.GpuMemory.Covers(address, 8));

            // Outside a mapping change an unregistration brackets itself.
            split.GuestMemory.Unregister(address, SharedGuestMemory.Chunk);
            Assert.False(split.GpuMemory.Covers(address, 8));

            // An ordered video-out change runs behind the backend's accepted streams.
            split.Backend.Worker = worker;
            var ran = false;
            split.Session.Backend.RunAfterPendingCommandStreams(() => ran = true);
            Assert.True(ran);
            Assert.Contains("after-pending", split.Backend.Calls);
        }
        finally
        {
            split.GpuMemory.AttachGpuQueue(null, null);
        }
    }

    [MacOSFact]
    public void TheGpuSideReachesTheGuestsVideoOutState()
    {
        using var split = new InProcessSplit();
        var bridge = split.Host.VideoOut!;
        split.GuestSide.Hdr = false;
        string? failure = null;
        var caller = new Thread(() =>
        {
            try
            {
                Assert.Equal(0, bridge.TryReserveFlipRequest(1, 2, 1, 42, gpuQueued: true, out var request));
                Assert.Equal(77UL, request);
                Assert.False(bridge.IsHdrOutputRequested);
                Assert.True(bridge.IsFlipDone(1, 2));
                Assert.True(bridge.CanPresentFlip(77, 1000));
                Assert.Equal(1100, bridge.NextFlipPresentTimestamp(77, 1000));
                Assert.True(bridge.TryGetDisplayBufferInfo(1, 2, out var buffer));
                Assert.Equal(new VideoOutExports.DisplayBufferInfo(0x5000, 0x8000_0000, 0, 1920, 1080, 1920, 3), buffer);
                Assert.Equal(2, bridge.GetFlipEventCount(77));
                Assert.True(bridge.ReleaseNoBufferFlip(-1, 77));
                Assert.True(bridge.IsFlipPresentationPending(77));
                Assert.Equal("SharpEmu · test window", bridge.GetWindowTitle());

                // A flip step is a call: the guest's state moved before it returns.
                bridge.MarkFlipPresented(77);
                Assert.Contains((GpuSplitFlipEvent.Presented, 77UL), split.GuestSide.FlipEvents);
                bridge.CompleteFlipOnGpu(78);
                bridge.DiscardFlip(79);
                bridge.CancelFlip(80);
                Assert.Equal(
                    new[] { GpuSplitFlipEvent.Presented, GpuSplitFlipEvent.CompleteOnGpu, GpuSplitFlipEvent.Discarded, GpuSplitFlipEvent.Cancelled },
                    split.GuestSide.FlipEvents.Select(flip => flip.Kind));
            }
            catch (Exception exception)
            {
                failure = exception.ToString();
            }
        });
        caller.Start();
        Assert.True(caller.Join(20_000));
        Assert.Null(failure);
        Assert.Equal(
            new[] { "reserve 1 2 1 42", "done 1 2", "can 77 1000", "next 77 1000", "buffer 1 2", "events 77", "release -1 77", "pending 77" },
            split.GuestSide.Calls);

        bridge.SetSelectedGpuName("Apple M5 Pro");
        bridge.NotifyPresentationWindowClosed();
        bridge.CancelOutstandingFlips();
        Wait.Until(() => split.GuestSide.Notices.Count == 3, "the notices");
        Assert.Equal(
            new[] { (GpuSplitNotice.SelectedGpuName, "Apple M5 Pro"), (GpuSplitNotice.PresentationWindowClosed, ""), (GpuSplitNotice.CancelOutstandingFlips, "") },
            split.GuestSide.Notices);
    }

    [MacOSFact]
    public void InterruptsAndWindowInputReachTheGuest()
    {
        using var split = new InProcessSplit(inputPoll: TimeSpan.FromMilliseconds(2));
        split.Host.SendInterrupt(5, 2);
        Wait.Until(() => split.GuestSide.Interrupts.Contains((5, 2u)), "the interrupt");

        // From a GPU thread an interrupt is a call: delivered before it returns.
        var completion = new Thread(() => split.Host.TriggerInterrupt(6, 1));
        completion.Start();
        Assert.True(completion.Join(10_000));
        Assert.Contains((6, 1u), split.GuestSide.Interrupts);

        HostWindowInput.Connect();
        try
        {
            HostWindowInput.SetKey(33, down: true);
            Wait.Until(() => split.GuestSide.Inputs.Any(keys => keys.Contains(33)), "the key");
        }
        finally
        {
            HostWindowInput.Disconnect();
        }
    }

    [MacOSFact]
    public void ControlCallsFramesQueriesAndMirrorsCross()
    {
        using var split = new InProcessSplit();
        var backend = split.Session.Backend;
        backend.EnsureStarted(1280, 720);
        backend.HideSplashScreen();
        backend.CountShaderCompilation();
        var frame = new byte[600 * 400 * 4];
        frame[^1] = 0xA5;
        backend.Submit(frame, 600, 400);
        split.Session.SendVideoOptions(new HostVideoOptions { Width = 2560, Height = 1440 });
        Assert.True(split.Session.Writer.Write(GpuRemoteMessage.ShaderHeader, new ShaderHeaderMessage { CodeAddress = 0xC0DE_0000, HeaderAddress = 0xC0DE_1000 }));
        Assert.True(split.Session.Writer.Write(GpuRemoteMessage.FusedProgram, new FusedProgramMessage
        {
            Entry = 0xF00D_0000,
            EntryHeader = 0xF00D_1000,
            Continuation = 0xF00D_2000,
            ContinuationHeader = 0xF00D_3000,
        }));
        Assert.Equal(IdleOutcome.Completed, backend.SubmitDone(null!));

        Assert.Equal(new[] { "start 1280x720", "hide-splash", "count-shader", "frame 600x400 960000" }, split.Backend.Calls.Take(4));
        Assert.Equal(0xA5, Assert.Single(split.Backend.Frames)[^1]);
        Assert.Equal(2560, split.ConfiguredVideo!.Width);
        Assert.Contains(new KeyValuePair<ulong, ulong>(0xC0DE_0000, 0xC0DE_1000), AgcBackendServices.SnapshotShaderHeaders());
        Assert.Contains((0xF00D_0000UL, 0xF00D_1000UL, 0xF00D_2000UL, 0xF00D_3000UL), Gen5ShaderTranslator.SnapshotFusedPrograms(split.Host.AddressSpace!));

        Assert.Equal((11L, 2.5, 3L, 4L), backend.ReadAndResetPerfCounters());
        Assert.True(backend.QueryHdrOutputSupport());

        // The pad lives with the window: its output crosses to the GPU side's input source.
        var source = new RecordingInputSource();
        HostWindowInputSource.Set(source);
        try
        {
            var output = new SplitGamepadOutput(split.Session.Writer);
            output.SetRumble(10, 20);
            output.SetTriggerRumble(null, 7);
            output.SetLightbar(1, 2, 3);
            output.SetAdaptiveTriggerEffect(new HostAdaptiveTriggerEffect(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), null);
            output.ResetLightbar();
            _ = backend.SubmitDone(null!);
            Assert.Equal(new[] { "rumble 10 20", "triggers  7", "lightbar 1 2 3", "adaptive 11 none", "reset" }, source.Calls);
        }
        finally
        {
            HostWindowInputSource.Clear(source);
        }

        backend.RequestClose();
        Wait.Until(() => split.Backend.Calls.Contains("close"), "the close");
        HostSessionControl.ResetShutdownRequest();
    }

    [MacOSFact]
    public void TheGuestStopsWhenTheGpuProcessDies()
    {
        using var split = new InProcessSplit();
        split.GuestMemory.Register(split.Guest.Address, SharedGuestMemory.ViewBytes, ReadWrite);
        var watcher = new Thread(() => split.GpuMemory.Pages.AddWatch(split.Guest.Address, 0x1000, blockReads: false));
        watcher.Start();
        Assert.True(watcher.Join(10_000));
        Assert.Equal(GuestPageProtection.Read, split.GuestSpace.Last(split.Guest.Address).Access);
        split.KillGpuSide();
        Wait.Until(() => split.GuestSide.LostReason is not null, "the loss report");
        Assert.Contains("exited unexpectedly", split.GuestSide.LostReason);
        Assert.False(split.Session.IsAlive);
        // The dead side's watch comes off before the guest is stopped.
        Assert.Equal(ReadWrite, split.GuestSpace.Last(split.Guest.Address).Access);

        // Nothing waits for the dead GPU process any more.
        var watch = Stopwatch.StartNew();
        Assert.Equal(IdleOutcome.Cancelled, split.Session.Backend.SubmitDone(null!));
        split.GuestMemory.Unregister(split.Guest.Address, SharedGuestMemory.Chunk);
        Assert.False(split.GuestMemory.Covers(split.Guest.Address, 8));
        Assert.False(split.Session.Backend.TrySubmitGuestImage(1, 0, 0, 1, 1, 1, 2));
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(2), $"calls to a dead GPU process took {watch.Elapsed}");
    }

    private sealed class RecordingInputSource : IHostWindowInputSource
    {
        public List<string> Calls { get; } = new();

        public bool HasKeyboardFocus => true;

        public bool IsKeyDown(int virtualKey) => false;

        public int GetGamepadStates(Span<HostGamepadState> destination) => 0;

        public string? DescribeConnectedGamepad() => null;

        public void SetRumble(byte largeMotor, byte smallMotor) => Calls.Add($"rumble {largeMotor} {smallMotor}");

        public void SetTriggerRumble(byte? leftTrigger, byte? rightTrigger) => Calls.Add($"triggers {leftTrigger} {rightTrigger}");

        public void SetAdaptiveTriggerEffect(HostAdaptiveTriggerEffect? leftTrigger, HostAdaptiveTriggerEffect? rightTrigger) =>
            Calls.Add($"adaptive {leftTrigger?.B10.ToString() ?? "none"} {rightTrigger?.B10.ToString() ?? "none"}");

        public void SetLightbar(byte red, byte green, byte blue) => Calls.Add($"lightbar {red} {green} {blue}");

        public void ResetLightbar() => Calls.Add("reset");
    }
}
