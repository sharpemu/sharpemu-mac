// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Versioning;
using SharpEmu.HLE;
using SharpEmu.HLE.DsTrace;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Platform.MacOS.Windowing;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The window and the GPU worker of the v2 backend. Two threads: the AppKit main
/// thread owns the CAMetalLayer and presents, and the GPU worker owns the command
/// stream and all recording. A flip hands the cached image's MTLTexture over, so a
/// presented frame never travels through the CPU.
/// </summary>
[SupportedOSPlatform("macos")]
internal static partial class MetalV2Presenter
{
    private const ulong PixelFormatBgra8Unorm = 80;

    private static readonly object _gate = new();
    private static readonly object _frameGate = new();

    private static nint _device;
    private static nint _queue;
    private static MetalRenderHost? _host;
    private static MetalGpuWorkerRelay? _relay;
    private static Thread? _worker;
    private static SharpEmu.HLE.Host.HostQosThread? _workerQos;
    // CoreCLR passes explicit scheduling parameters to every thread it creates, and Darwin
    // then refuses pthread_set_qos_class_self_np on it (EPERM): the UserInteractive request
    // at the top of RunWorker never applied, and the worker ran at base priority 26 next
    // to a title's spinning job threads. A thread created natively with the class in its
    // attributes is the one route that works. SHARPEMU_METAL_V2_WORKER_QOS_THREAD=0 keeps
    // the managed thread.
    private static readonly bool WorkerQosThreadEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WORKER_QOS_THREAD"), "0", StringComparison.Ordinal);
    private const int WorkerStackBytes = 16 * 1024 * 1024;
    private const int PacerStackBytes = 4 * 1024 * 1024;
    private static Thread? _windowThread;
    private static IMetalHostWindow? _hostWindow;
    private static HostVideoOptions _videoOptions = HostVideoOptions.Default;
    private static nint _metalLayer;
    private static nint _presentQueue;
    private static nint _presentPipeline;
    private static nint _presentLinearPipeline;
    private static nint _presentSampler;
    private static double _drawableWidth;
    private static double _drawableHeight;
    private static bool _closed;
    private static bool _closeRequested;
    private static bool _splashHidden;
    private static bool _windowStarted;
    private static Exception? _startupFailure;

    // The frame the next present shows: a guest texture plus the serial it needs.
    // Published flips waiting for their submission and their refresh, oldest first.
    private readonly record struct PendingFrame(
        nint Texture, uint Width, uint Height, ulong RequestId, ulong Serial, bool IsHdr);

    private const int MaxPendingFrames = 4;
    private static readonly Queue<PendingFrame> _pendingFrames = new(MaxPendingFrames);
    private static nint _presentTexture;
    private static uint _presentWidth;
    private static uint _presentHeight;
    private static bool _presentIsHdr;
    private static byte[]? _pendingHostFrame;
    private static uint _pendingHostWidth;
    private static uint _pendingHostHeight;
    private static nint _hostFrameTexture;
    private static uint _hostFrameWidth;
    private static uint _hostFrameHeight;
    private static long _guestFrameSequence;
    private static long _presentedGuestFrames;
    // Independent of the resettable statistics counters and the published flip queue.
    private static long _adoptedGuestFrameSequence;
    private static long _hostFrameSequence;
    private static readonly bool ReusePresentedFrame =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_REUSE_PRESENTED_FRAME") == "1";
    private static readonly MetalPresentationCache PresentationCache = new();
    private static int _firstGuestFrameNoted;
    private static long _publishedGuestFrames;
    private static long _adoptionsBlocked;
    private static long _hostPresents;
    private static long _statsTimestamp;

    public static bool TryConfigureVideo(HostVideoOptions options)
    {
        lock (_gate)
        {
            if (_windowStarted)
            {
                return false;
            }

            _videoOptions = options.Normalize();
            return true;
        }
    }

    public static MetalRenderHost Host =>
        _host ?? throw GpuFatal.Fatal("The Metal v2 render host has not been created.");

    public static bool IsStarted => Volatile.Read(ref _host) is not null;

    // MTLGPUFamily.Metal3, the family that has the mesh stage.
    private const nint GpuFamilyMetal3 = 5001;

    /// <summary>Whether this device can build mesh render pipelines.</summary>
    public static bool SupportsMeshPipelines()
    {
        lock (_gate)
        {
            try
            {
                EnsureDeviceLocked();
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[METAL-V2][WARN] The Metal device is unavailable: {exception.Message}");
                return false;
            }

            var supportsFamily = MetalNative.Selector("supportsFamily:");
            if (MetalNative.Class("MTLMeshRenderPipelineDescriptor") == 0 ||
                !MetalNative.SendBool(_device, MetalNative.Selector("respondsToSelector:"), supportsFamily) ||
                !MetalNative.SendBool(_device, supportsFamily, GpuFamilyMetal3))
            {
                Console.Error.WriteLine("[METAL-V2][INFO] Mesh render pipelines are unavailable on this device.");
                return false;
            }

            return true;
        }
    }

    /// <summary>Creates the device, the render host and the GPU worker, once.</summary>
    public static MetalRenderHost EnsureHost()
    {
        lock (_gate)
        {
            if (_startupFailure is { } failure)
            {
                throw new InvalidOperationException("The Metal v2 backend could not start.", failure);
            }

            if (_host is { } existing)
            {
                return existing;
            }

            try
            {
                EnsureDeviceLocked();
                if (GuestGpuMemoryHook.Current is not { AddressSpace: ICpuMemory guest and IGuestBackedSpace backing } memory)
                {
                    throw GpuFatal.Fatal("Metal v2 requires the guest GPU memory manager with backed virtual memory.");
                }

                _relay = new MetalGpuWorkerRelay(Wake, TryEnqueueControlBarrier);
                var host = new MetalRenderHost(_device, _queue, guest, backing, memory.Pages, _relay);
                host.InitializeCommandStream(new FlipTarget());
                _host = host;
                // v2 has no buffer store: guest memory is the GPU's memory, so only the
                // image store answers a CPU write fault. The relay lets an unmap wait
                // for the worker to stop touching the range.
                memory.AttachStores(null, host.ImageCache);
                memory.AttachGpuQueue(_relay, null);
                var qosErrno = -1;
                _workerQos = WorkerQosThreadEnabled
                    ? SharpEmu.HLE.Host.HostQosThread.TryStart("SharpEmu Metal v2 GPU worker",
                        SharpEmu.HLE.Host.HostThreadQos.UserInteractive, WorkerStackBytes, RunWorker, out qosErrno)
                    : null;
                if (_workerQos is null)
                {
                    if (WorkerQosThreadEnabled)
                    {
                        Console.Error.WriteLine(
                            $"[METAL-V2][WARN] GPU worker: native QoS thread create failed (errno={qosErrno}); using a managed thread.");
                    }
                    _worker = new Thread(RunWorker)
                    {
                        IsBackground = true,
                        Name = "SharpEmu Metal v2 GPU worker",
                    };
                    _worker.Start();
                }
                else
                {
                    Console.Error.WriteLine("[METAL-V2][INFO] GPU worker runs on a native UserInteractive thread.");
                }
                // The pacer's wake latency is a refresh for the guest: same native
                // UserInteractive thread as the worker, same managed fallback.
                var pacerErrno = -1;
                if (!WorkerQosThreadEnabled ||
                    SharpEmu.HLE.Host.HostQosThread.TryStart("SharpEmu Metal v2 flip pacer",
                        SharpEmu.HLE.Host.HostThreadQos.UserInteractive, PacerStackBytes, RunFlipPacer, out pacerErrno) is null)
                {
                    if (WorkerQosThreadEnabled)
                    {
                        Console.Error.WriteLine(
                            $"[METAL-V2][WARN] flip pacer: native QoS thread create failed (errno={pacerErrno}); using a managed thread.");
                    }
                    new Thread(RunFlipPacer)
                    {
                        IsBackground = true,
                        Name = "SharpEmu Metal v2 flip pacer",
                    }.Start();
                }
                Console.Error.WriteLine("[METAL-V2][INFO] GPU worker started.");
                return host;
            }
            catch (Exception exception)
            {
                _startupFailure = exception;
                throw;
            }
        }
    }

    private static void EnsureDeviceLocked()
    {
        if (_device != 0)
        {
            return;
        }

        MetalNative.EnsureFrameworksLoaded();
        _device = MetalNative.MTLCreateSystemDefaultDevice();
        if (_device == 0)
        {
            throw GpuFatal.Fatal("No Metal device is available.");
        }

        _queue = MetalNative.Send(_device, MetalNative.Selector("newCommandQueue"));
        _presentQueue = MetalNative.Send(_device, MetalNative.Selector("newCommandQueue"));
        if (_queue == 0 || _presentQueue == 0)
        {
            throw GpuFatal.Fatal("A Metal command queue could not be created.");
        }

        var name = MetalNative.ReadNsString(MetalNative.Send(_device, MetalNative.Selector("name")));
        if (!string.IsNullOrEmpty(name))
        {
            VideoOutBackendServices.SetSelectedGpuName(name);
        }
    }

    public static void EnsureStarted(uint width, uint height)
    {
        if (width == 0 || height == 0)
        {
            return;
        }

        lock (_gate)
        {
            if (_closed || _windowStarted)
            {
                return;
            }

            EnsureDeviceLocked();
            _windowStarted = true;
            // Before the prewarm, whose libraries it may already hold.
            MetalBinaryArchive.Start(_device);
            MetalShaderPrewarm.Start(_device);
            if (HostMainThread.IsAvailable)
            {
                HostMainThread.SetShutdownRequestHandler(RequestClose);
                HostMainThread.Post(RunWindow);
                return;
            }

            _windowThread = new Thread(RunWindow)
            {
                IsBackground = true,
                Name = "SharpEmu Metal v2 VideoOut",
            };
            _windowThread.Start();
        }
    }

    public static void HideSplashScreen()
    {
        lock (_frameGate)
        {
            _splashHidden = true;
            _pendingHostFrame = null;
        }
    }

    /// <summary>Presents one CPU-produced BGRA frame (the splash and the fallbacks).</summary>
    public static void Submit(byte[] bgraFrame, uint width, uint height)
    {
        if (bgraFrame.Length != checked((int)(width * height * 4)))
        {
            return;
        }

        lock (_frameGate)
        {
            if (_splashHidden && _presentTexture != 0)
            {
                return;
            }

            _pendingHostFrame = bgraFrame;
            _pendingHostWidth = width;
            _pendingHostHeight = height;
        }

        EnsureStarted(width, height);
    }

    public static void RequestClose()
    {
        Volatile.Write(ref _closeRequested, true);
        // The guest leaves right after this, and the runtime then frees the page
        // guard, which refuses to go while an image watch is outstanding. Dropping
        // the watches here does not depend on the worker reaching its own teardown.
        Volatile.Read(ref _host)?.ImageCache.StopWatching();
        try
        {
            Volatile.Read(ref _hostWindow)?.Close();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] The window could not be closed: {exception.Message}");
        }

        StopWorker();
    }

    /// <summary>
    /// Stops the GPU worker and waits for its teardown. The page watches the image
    /// cache holds have to come off before the runtime releases the page guard, and
    /// only the worker may touch the cache, so the caller waits for it here.
    /// </summary>
    public static void StopWorker()
    {
        Thread? worker;
        SharpEmu.HLE.Host.HostQosThread? workerQos;
        lock (_gate)
        {
            worker = _worker;
            _worker = null;
            workerQos = _workerQos;
            _workerQos = null;
        }

        Volatile.Write(ref _closeRequested, true);
        MetalShaderPrewarm.Stop();
        _host?.CommandStream.Wake();
        if (workerQos is not null)
        {
            if (!workerQos.IsCurrentThread && !workerQos.Join(2000))
            {
                Console.Error.WriteLine("[METAL-V2][WARN] The GPU worker did not stop within two seconds.");
            }
            return;
        }
        if (worker is null || worker == Thread.CurrentThread)
        {
            return;
        }

        if (!worker.Join(TimeSpan.FromSeconds(2)))
        {
            Console.Error.WriteLine("[METAL-V2][WARN] The GPU worker did not stop within five seconds.");
        }
    }

    private static void Wake()
    {
        _host?.CommandStream.Wake();
        _host?.WakeWorker();
    }

    // ---- the GPU worker ----

    // An ordered video-out change (a buffer unregister) must land behind the command
    // streams the queue already accepted: one of them may still carry a flip to the
    // slot, and reserving that flip after the unregister is a fatal InvalidIndex.
    // The relay alone only orders work before the next slice, not behind the queue.
    private static bool TryEnqueueControlBarrier(Action work, Action cancel) =>
        _host is { } host && host.CommandStream.TryEnqueueControlBarrier(() => host.RunControlBarrier(work), cancel);

    /// <summary>
    /// One pass of the worker: the relay's commands, a budget of command-stream slices and
    /// the counters. The pass runs in an autorelease pool of its own, because this thread
    /// has none: what Metal autoreleases while recording (command buffers, encoders,
    /// descriptors and strings from class factories) is freed when the pass ends. Whatever
    /// has to outlive the pass is retained by its owner. The park is outside the pool, so
    /// the pool never waits out the idle interval.
    /// </summary>
    /// <returns>True when nothing is runnable and the worker should park.</returns>
    internal static bool RunWorkerStep(MetalRenderHost host, MetalGpuWorkerRelay relay)
    {
        using var pool = MetalAutoreleasePool.Enter();
        host.RunPendingCommands();
        host.RunCommandStreamSlices(64);
        host.ReportCounters();
        if ((host.UsesCommandThread ? host.CommandThreadHasWork : host.CommandStream.HasUnblockedPending) ||
            relay.HasPendingCommands)
        {
            return false;
        }

        // A buffer a coalesced Stream flush left open goes out before the park:
        // nothing else would commit it until the guest submits again. With the command thread the
        // worker parks between its items all the time, and the thread asks for the commit itself.
        if (!host.UsesCommandThread)
        {
            host.CommitCoalescedFlush();
        }

        host.PollGpuCompletions();
        host.KeepQueueWarmIfIdle();
        return true;
    }

    private static void RunWorker()
    {
        var host = _host!;
        var qosApplied = SharpEmu.HLE.Host.HostThreadQualityOfService.TryApplyToCurrentThread(SharpEmu.HLE.Host.HostThreadQos.UserInteractive);
        if (DsTrace.On)
        {
            DsTrace.RegisterThread(2, 0, Thread.CurrentThread.Name ?? "MetalV2Worker", qosApplied ? 1 : 0);
        }
        _relay!.BindCurrentThread();
        try
        {
            while (!Volatile.Read(ref _closeRequested) && !HostSessionControl.IsShutdownRequested)
            {
                if (RunWorkerStep(host, _relay))
                {
                    var idleStarted = System.Diagnostics.Stopwatch.GetTimestamp();
                    var dsIdle = DsTrace.Gpu && DsTrace.Armed;
                    var dsIdleCpu = dsIdle ? DsTrace.SelfCpuNs() : 0UL;
                    // The park is idle time, not loop overhead: charged to Idle it stays out of
                    // Unattributed and out of the worker's per-draw figure.
                    using (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Measure(
                        SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.Idle))
                    {
                        if (host.UsesCommandThread)
                        {
                            // The command thread waits on the queue; the worker on what it hands over.
                            host.WaitForCommandThreadWork(CommandStreamQueue.AllBlockedRetryMilliseconds);
                        }
                        else
                        {
                            _ = host.CommandStream.WaitForWork(CommandStreamQueue.AllBlockedRetryMilliseconds);
                        }
                    }
                    var idleEnded = System.Diagnostics.Stopwatch.GetTimestamp();
                    host.NoteIdle(idleEnded - idleStarted);
                    if (dsIdle)
                    {
                        DsEmitIdle(host.CommandStream, idleStarted, idleEnded, dsIdleCpu);
                    }
                }
            }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][ERROR] The GPU worker stopped: {exception}");
            host.CommandStream.Fail(exception);
        }
        finally
        {
            Shutdown(host);
        }
    }

    // DSTRACE §4.7 WIDLE: one worker park. reason 0 = nothing pending (NoWork), 1 = every
    // pending head blocked (AllBlocked); WaitForWork captured both under its own gate.
    private static void DsEmitIdle(CommandStreamQueue stream, long started, long ended, ulong cpuStarted)
    {
        var allBlocked = stream.DsIdleHadPending;
        var snapshot = allBlocked ? stream.DsIdleSnapshot : default;
        DsTrace.EmitAt(
            ended,
            DsEv.WIDLE,
            (ulong)started,
            (ulong)DsTrace.TicksToNs(ended - started),
            allBlocked ? 1UL : 0UL,
            (ulong)snapshot.Outstanding,
            snapshot.SampleWaitAddress,
            (ulong)(snapshot.OldestAgeMilliseconds * 1000.0),
            DsTrace.SelfCpuNs() - cpuStarted);
    }

    // The worker owns the teardown: the page watches have to come off before the
    // page guard is released, and only this thread may touch the image cache.
    private static void Shutdown(MetalRenderHost host)
    {
        using var pool = MetalAutoreleasePool.Enter();
        try
        {
            host.CommandStream.StopAccepting();
            host.CommandStream.DiscardAll();
            host.StopCommandThread();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] The command stream could not stop: {exception.Message}");
        }

        try
        {
            _relay?.StopAcceptingWork();
            host.ImageCache.StopWatching();
            host.FlushAndWaitForGpu();
            host.ImageCache.Shutdown();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] The image cache could not drain: {exception.Message}");
        }
        finally
        {
            GuestGpuMemoryHook.Current?.AttachStores(null, null);
            GuestGpuMemoryHook.Current?.AttachGpuQueue(null, null);
            VideoOutBackendServices.CancelOutstandingFlips();
        }

        Console.Error.WriteLine("[METAL-V2][INFO] GPU worker stopped.");
    }

    // ---- the window and presentation ----

    private static void RunWindow()
    {
        try
        {
            using var window = new MacHostWindow(VideoOutBackendServices.GetWindowTitle(), _videoOptions);
            Volatile.Write(ref _hostWindow, window);
            var pool = MetalNative.objc_autoreleasePoolPush();
            try
            {
                _metalLayer = window.CreateMetalLayer();
                ConfigureMetalLayer(_device, _metalLayer);
                if (!TryCreatePresentPipeline(_device, out _presentPipeline, out var error))
                {
                    Console.Error.WriteLine($"[METAL-V2][ERROR] The present pipeline failed: {error}");
                    return;
                }

                if (!TryCreatePresentPipeline(
                        _device, out _presentLinearPipeline, out error, fragmentEntry: "present_linear_fs"))
                {
                    Console.Error.WriteLine($"[METAL-V2][ERROR] The linear present pipeline failed: {error}");
                    return;
                }

                _presentSampler =CreateLinearSampler(_device);
                InitializeHdrOutput(window);
                SyncDrawableSize();
            }
            finally
            {
                MetalNative.objc_autoreleasePoolPop(pool);
            }

            Console.Error.WriteLine("[METAL-V2][INFO] VideoOut presenter started.");
            window.Run(static () => { }, static _ => PresentFrame(), static () => { });
            if (window.ClosedByUser)
            {
                Console.Error.WriteLine("[METAL-V2][WARN] The window closed; requesting emulator shutdown.");
                VideoOutBackendServices.NotifyPresentationWindowClosed();
            }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][ERROR] The VideoOut presenter failed: {exception}");
        }
        finally
        {
            Volatile.Write(ref _hostWindow, null);
            _metalLayer = 0;
            lock (_gate)
            {
                _closed = true;
            }

            ReleaseHdrSupportWaiters();
        }
    }

    private static void ConfigureMetalLayer(nint device, nint layer)
    {
        MetalNative.SendVoid(layer, MetalNative.Selector("setDevice:"), device);
        MetalV2Native.SendVoidValue(layer, MetalNative.Selector("setPixelFormat:"), (nuint)PixelFormatBgra8Unorm);
        MetalNative.SendVoidBool(layer, MetalNative.Selector("setOpaque:"), true);
        MetalNative.SendVoidBool(layer, MetalNative.Selector("setFramebufferOnly:"), true);
        MetalV2Native.SendVoidValue(layer, MetalNative.Selector("setMaximumDrawableCount:"), 3);
        var displaySync = MetalNative.Selector("setDisplaySyncEnabled:");
        if (MetalNative.SendBool(layer, MetalNative.Selector("respondsToSelector:"), displaySync))
        {
            MetalNative.SendVoidBool(layer, displaySync, _videoOptions.VSync);
        }
    }

    private static void SyncDrawableSize()
    {
        var window = Volatile.Read(ref _hostWindow);
        if (window is null || _metalLayer == 0)
        {
            return;
        }

        var (pixelWidth, pixelHeight) = window.PixelSize;
        if (pixelWidth == _drawableWidth && pixelHeight == _drawableHeight)
        {
            return;
        }

        _drawableWidth = pixelWidth;
        _drawableHeight = pixelHeight;
        SharpEmu.Libs.Media.MoviePresentationServices.SetPresentationSize((uint)pixelWidth, (uint)pixelHeight);
        MetalNative.SendVoidSize(
            _metalLayer,
            MetalNative.Selector("setDrawableSize:"),
            new CGSize { Width = pixelWidth, Height = pixelHeight });
    }

    private static void PresentFrame()
    {
        if (Volatile.Read(ref _closeRequested))
        {
            return;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var window = Volatile.Read(ref _hostWindow);
            if (window is null || window.IsMinimized)
            {
                Thread.Sleep(8);
                return;
            }

            if (window.ConsumeSurfaceRestore())
            {
                _drawableWidth = 0;
                PresentationCache.Invalidate();
            }

            SyncDrawableSize();
            AdoptPendingFrame();
            UploadPendingHostFrame();
            UpdateOutputColor(window);
            // The GPU worker promotes frames too, when it publishes one and when it asks
            // for a flip slot, and a promotion releases the frame it replaces. So the
            // presented frame is read under the gate and held until the encoder has bound
            // it (the command buffer then keeps its own reference): read bare, a promotion
            // in between released it and the bind retained freed memory.
            nint presented;
            uint presentedWidth, presentedHeight;
            bool presentedHdr;
            long presentedSequence;
            lock (_frameGate)
            {
                presented = _presentTexture;
                presentedWidth = _presentWidth;
                presentedHeight = _presentHeight;
                presentedHdr = _presentIsHdr;
                presentedSequence = _adoptedGuestFrameSequence;
                if (presented != 0)
                {
                    MetalNative.Send(presented, MetalNative.Selector("retain"));
                }
            }

            try
            {
                var texture = presented != 0 ? presented : _hostFrameTexture;
                var width = presented != 0 ? presentedWidth : _hostFrameWidth;
                var height = presented != 0 ? presentedHeight : _hostFrameHeight;
                var movie = SelectHostMovieFallback(ref texture, ref width, ref height);
                var stamp = new MetalPresentationStamp(
                    presentedSequence, _hostFrameSequence, movie ? _movieFallbackSerial : 0,
                    texture, _drawableWidth, _drawableHeight, _layerHdrActive,
                    !movie && presented != 0 && presentedHdr, _edrCurrentHeadroom);
                // Keep polling movies, window state and guest flips. The flip pacer is
                // independent, so reusing a drawable never delays guest flip completion.
                // Before a guest/movie image exists, the prewarm overlay may animate.
                if (ReusePresentedFrame && !PresentationCache.NeedsPresent(in stamp, presented == 0 && !movie))
                {
                    ReportPresentationStatistics(presented: false);
                    // nextDrawable normally paces this loop; skipping it must not spin.
                    Thread.Sleep(8);
                    return;
                }

                var drawable = MetalNative.Send(_metalLayer, MetalNative.Selector("nextDrawable"));
                if (drawable == 0)
                {
                    return;
                }

                var drawableTexture = MetalNative.Send(drawable, MetalNative.Selector("texture"));
                var commandBuffer = MetalNative.Send(_presentQueue, MetalNative.Selector("commandBuffer"));
                MetalMemoryTelemetry.CommandBuffers.Add();
                if (drawableTexture == 0 || commandBuffer == 0)
                {
                    return;
                }

                var pass = CreateClearPass(drawableTexture);
                var encoder = MetalNative.Send(
                    commandBuffer, MetalNative.Selector("renderCommandEncoderWithDescriptor:"), pass);
                MetalMemoryTelemetry.RenderEncoders.Add();
                if (encoder == 0)
                {
                    return;
                }

                if (texture != 0 && width != 0 && height != 0)
                {
                    EncodePresent(encoder, texture, width, height, !movie && presented != 0 && presentedHdr);
                }

                if (presented == 0 && !movie)
                {
                    EncodePrewarmProgress(encoder);
                }

                MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
                MetalNative.SendVoid(commandBuffer, MetalNative.Selector("presentDrawable:"), drawable);
                MetalMemoryTelemetry.Commits.Add();
                MetalNative.SendVoid(commandBuffer, MetalNative.Selector("commit"));
                PresentationCache.RecordPresented(in stamp);
                PerfOverlay.RecordPresent();
                ReportPresentationStatistics();
            }
            finally
            {
                if (presented != 0)
                {
                    MetalNative.SendVoid(presented, MetalNative.Selector("release"));
                }
            }
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    // A published flip becomes the presented frame once its submission completes.
    // Promotion also happens on publication, so a frame that finished while the guest
    // was already drawing the next one is still shown instead of being replaced
    // unseen: with one slot and a completion wait, the newest frame is never ready
    // and the display starves.
    private static void AdoptPendingFrame()
    {
        lock (_frameGate)
        {
            if (!TryPromotePendingLocked())
            {
                Interlocked.Increment(ref _adoptionsBlocked);
            }
        }
    }

    private static readonly AutoResetEvent _flipPacerSignal = new(false);

    // Wakes the pacer: a frame was published or a flip submission completed.
    internal static void SignalFlipPacer() => _flipPacerSignal.Set();

    // Presents each queued flip at the refresh video-out allows, independent of the
    // window's drawable cadence: the guest's flip event and buffer release follow
    // presentation, so a frame promoted a host tick late costs the guest a refresh.
    private static void RunFlipPacer()
    {
        var qosApplied = SharpEmu.HLE.Host.HostThreadQualityOfService.TryApplyToCurrentThread(
            SharpEmu.HLE.Host.HostThreadQos.UserInteractive);
        if (DsTrace.On)
        {
            DsTrace.RegisterThread(4, 0, Thread.CurrentThread.Name ?? "FlipPacer", qosApplied ? 1 : 0);
        }
        while (Volatile.Read(ref _host) is { } host)
        {
            var wakeAt = long.MaxValue;
            var waitingForGpu = false;
            // This thread has no pool: what the promotion and the event read autorelease is
            // freed at the end of each pass, not held for the life of the thread.
            using (MetalAutoreleasePool.Enter())
            {
                lock (_frameGate)
                {
                    while (TryPromotePendingLocked())
                    {
                    }

                    if (_pendingFrames.TryPeek(out var frame))
                    {
                        if (host.Completions.SignaledValue < frame.Serial)
                        {
                            waitingForGpu = true;
                        }
                        else
                        {
                            wakeAt = VideoOutBackendServices.NextFlipPresentTimestamp(
                                frame.RequestId, System.Diagnostics.Stopwatch.GetTimestamp());
                        }
                    }
                }
            }

            if (waitingForGpu)
            {
                // Every finished command buffer signals the pacer (OnGpuCompleted), so
                // the wait ends when the GPU does. Without the completion handler the
                // shared event is polled, and so is this.
                _flipPacerSignal.WaitOne(host.Completions.WantsCompletionHandler ? GpuWaitBackstopMilliseconds : 1);
            }
            else if (wakeAt == long.MaxValue)
            {
                _flipPacerSignal.WaitOne(wakeAt == long.MaxValue && _pendingFrames.Count > 0 ? 1 : 100);
            }
            else
            {
                SleepUntil(wakeAt);
            }
        }
    }

    // A finished command buffer the pacer was not told about is caught by this backstop.
    private const int GpuWaitBackstopMilliseconds = 8;

    // Coarse sleep for the bulk (macOS overshoots by up to ~1.5 ms, and a signal ends it
    // early), then mach_wait_until for the tail: the kernel wakes the thread at the
    // deadline, where the spin it replaces held a core for the last two milliseconds of
    // every refresh.
    private static void SleepUntil(long target)
    {
        var frequency = System.Diagnostics.Stopwatch.Frequency;
        var remaining = target - System.Diagnostics.Stopwatch.GetTimestamp();
        if (remaining <= 0)
        {
            return;
        }

        var microseconds = remaining * 1_000_000 / frequency;
        if (microseconds > 2000)
        {
            _flipPacerSignal.WaitOne((int)((microseconds - 1500) / 1000));
            return;
        }

        MachWaitFor(remaining * 1_000_000_000 / frequency);
    }

    private static void MachWaitFor(long nanoseconds)
    {
        if (MachTimebase.Numer == 0 || MachTimebase.Denom == 0)
        {
            throw new InvalidOperationException("mach_timebase_info returned no timebase.");
        }

        var ticks = (ulong)nanoseconds * MachTimebase.Denom / MachTimebase.Numer;
        _ = mach_wait_until(mach_absolute_time() + ticks);
        Interlocked.Increment(ref _pacerTimedWaits);
    }

    private static long _pacerTimedWaits;

    private struct MachTimebaseInfo
    {
        public uint Numer;
        public uint Denom;
    }

    private static readonly MachTimebaseInfo MachTimebase = ReadMachTimebase();

    private static MachTimebaseInfo ReadMachTimebase()
    {
        _ = mach_timebase_info(out var info);
        return info;
    }

    [System.Runtime.InteropServices.DllImport("/usr/lib/libSystem.dylib")]
    private static extern int mach_timebase_info(out MachTimebaseInfo info);

    [System.Runtime.InteropServices.DllImport("/usr/lib/libSystem.dylib")]
    private static extern ulong mach_absolute_time();

    [System.Runtime.InteropServices.DllImport("/usr/lib/libSystem.dylib")]
    private static extern int mach_wait_until(ulong deadline);

    // Caller holds the frame gate. Moves the oldest pending frame into the presented
    // slot once the GPU finished the submission that drew it and video-out lets it
    // reach the display (vblank, flip rate and flip mode).
    private static bool TryPromotePendingLocked()
    {
        if (!_pendingFrames.TryPeek(out var frame))
        {
            return false;
        }

        var host = Volatile.Read(ref _host);
        if (host is not null && host.Completions.SignaledValue < frame.Serial)
        {
            return false;
        }

        if (frame.RequestId != 0 &&
            !VideoOutBackendServices.CanPresentFlip(frame.RequestId, System.Diagnostics.Stopwatch.GetTimestamp()))
        {
            return false;
        }

        _ = _pendingFrames.Dequeue();
        if (_presentTexture != 0)
        {
            MetalNative.SendVoid(_presentTexture, MetalNative.Selector("release"));
        }

        _presentTexture = frame.Texture;
        _presentWidth = frame.Width;
        _presentHeight = frame.Height;
        _presentIsHdr = frame.IsHdr;
        Interlocked.Increment(ref _adoptedGuestFrameSequence);
        if (Interlocked.Increment(ref _presentedGuestFrames) == 1 && Interlocked.Exchange(ref _firstGuestFrameNoted, 1) == 0)
        {
            MetalShaderPrewarm.NoteFirstGuestFrame();
        }
        if (DsTrace.Gpu && DsTrace.Armed)
        {
            DsTrace.Emit(DsEv.PRESENT, frame.RequestId);
        }

        PerfOverlay.RecordGuestPresent();
        if (frame.RequestId != 0)
        {
            VideoOutBackendServices.MarkFlipPresented(frame.RequestId);
            // A flip packet suspended on a full queue can run now. On the worker
            // itself the caller is the slot check, which proceeds anyway.
            if (_relay is { IsGpuQueueThread: false } relay)
            {
                relay.TryPost(static () => Volatile.Read(ref _host)?.RetryBlockedFlips());
            }
        }

        return true;
    }

    private static unsafe void UploadPendingHostFrame()
    {
        byte[]? frame;
        uint width;
        uint height;
        lock (_frameGate)
        {
            frame = _pendingHostFrame;
            width = _pendingHostWidth;
            height = _pendingHostHeight;
            _pendingHostFrame = null;
        }

        if (frame is null || width == 0 || height == 0)
        {
            return;
        }

        if (_hostFrameTexture == 0 || _hostFrameWidth != width || _hostFrameHeight != height)
        {
            if (_hostFrameTexture != 0)
            {
                MetalNative.SendVoid(_hostFrameTexture, MetalNative.Selector("release"));
            }

            var descriptor = MetalNative.SendTextureDescriptor(
                MetalNative.Class("MTLTextureDescriptor"),
                MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
                (nuint)PixelFormatBgra8Unorm,
                width,
                height,
                mipmapped: false);
            _hostFrameTexture = MetalNative.Send(
                _device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
            _hostFrameWidth = width;
            _hostFrameHeight = height;
        }

        if (_hostFrameTexture == 0)
        {
            return;
        }

        fixed (byte* pixels = frame)
        {
            MetalNative.SendReplaceRegion(
                _hostFrameTexture,
                MetalNative.Selector("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                new MtlRegion { Width = width, Height = height, Depth = 1 },
                0,
                (nint)pixels,
                width * 4);
        }
        _hostFrameSequence++;
    }

    private static nint CreateClearPass(nint targetTexture)
    {
        var pass = MetalNative.Send(
            MetalNative.Class("MTLRenderPassDescriptor"), MetalNative.Selector("renderPassDescriptor"));
        var attachment = MetalNative.SendAtIndex(
            MetalNative.Send(pass, MetalNative.Selector("colorAttachments")),
            MetalNative.Selector("objectAtIndexedSubscript:"),
            0);
        MetalNative.SendVoid(attachment, MetalNative.Selector("setTexture:"), targetTexture);
        MetalV2Native.SendVoidValue(attachment, MetalNative.Selector("setLoadAction:"), Mtl.LoadActionClear);
        MetalV2Native.SendVoidValue(attachment, MetalNative.Selector("setStoreAction:"), Mtl.StoreActionStore);
        MetalNative.SendVoidClearColor(
            attachment,
            MetalNative.Selector("setClearColor:"),
            new MtlClearColor { Red = 0, Green = 0, Blue = 0, Alpha = 1 });
        return pass;
    }

    // Float VideoOut flip buffers hold linear light (the Vulkan presenter's
    // IsLinearFloatPresentSource); only an SDR drawable needs them encoded, an
    // extended-range layer takes the linear values as they are.
    internal static bool NeedsLinearEncode(ulong pixelFormat, bool hdr) =>
        !hdr && pixelFormat is MtlFormat.Rgba16Float or MtlFormat.Rgba32Float;

    private static void EncodePresent(nint encoder, nint texture, uint frameWidth, uint frameHeight, bool isHdr)
    {
        if (_layerHdrActive)
        {
            EncodeHdrPresentState(encoder, _hdrPresentPipeline, isHdr, _edrCurrentHeadroom);
        }
        else
        {
            // Linear float flips hold scRGB light, which a BGRA8 drawable shows as an
            // sRGB-encoded frame only after the encode, as the Vulkan presenter does.
            var format = (ulong)MetalNative.Send(texture, MetalNative.Selector("pixelFormat"));
            var pipeline = NeedsLinearEncode(format, _layerHdrActive) ? _presentLinearPipeline : _presentPipeline;
            MetalNative.SendVoid(encoder, MetalNative.Selector("setRenderPipelineState:"), pipeline);
        }

        var scaleX = _drawableWidth / frameWidth;
        var scaleY = _drawableHeight / frameHeight;
        var viewportWidth = _drawableWidth;
        var viewportHeight = _drawableHeight;
        if (_videoOptions.ScalingMode != HostScalingMode.Stretch)
        {
            var scale = _videoOptions.ScalingMode == HostScalingMode.Cover
                ? Math.Max(scaleX, scaleY)
                : Math.Min(scaleX, scaleY);
            if (_videoOptions.ScalingMode == HostScalingMode.Integer && scale >= 1)
            {
                scale = Math.Floor(scale);
            }

            viewportWidth = frameWidth * scale;
            viewportHeight = frameHeight * scale;
        }

        MetalNative.SendVoidViewport(
            encoder,
            MetalNative.Selector("setViewport:"),
            new MtlViewport
            {
                OriginX = (_drawableWidth - viewportWidth) * 0.5,
                OriginY = (_drawableHeight - viewportHeight) * 0.5,
                Width = viewportWidth,
                Height = viewportHeight,
                ZNear = 0,
                ZFar = 1,
            });
        MetalNative.SendSetAtIndex(
            encoder, MetalNative.Selector("setFragmentTexture:atIndex:"), texture, 0);
        MetalNative.SendSetAtIndex(
            encoder, MetalNative.Selector("setFragmentSamplerState:atIndex:"), _presentSampler, 0);
        MetalNative.SendDrawPrimitives(
            encoder, MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:"), Mtl.PrimitiveTriangle, 0, 3);
    }

    private static bool TryCreatePresentPipeline(
        nint device,
        out nint pipeline,
        out string error,
        string? fragmentSource = null,
        string fragmentEntry = "present_fs",
        ulong pixelFormat = PixelFormatBgra8Unorm)
    {
        pipeline = 0;
        error = string.Empty;
        var options = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        // The present shaders are hand-written and keep Metal's default fast math;
        // only the language version is pinned, so every library in the backend is
        // compiled against the same MSL version.
        MetalNative.SetMslLanguageVersion(options);
        nint vertexLibrary = 0;
        nint fragmentLibrary = 0;
        try
        {
            nint compileError = 0;
            vertexLibrary = MetalNative.Send(
                device,
                MetalNative.Selector("newLibraryWithSource:options:error:"),
                MetalNative.NsString(MslFixedShaders.CreateFullscreenVertex(1)),
                options,
                ref compileError);
            fragmentLibrary = MetalNative.Send(
                device,
                MetalNative.Selector("newLibraryWithSource:options:error:"),
                MetalNative.NsString(fragmentSource ?? MslFixedShaders.CreatePresentFragment()),
                options,
                ref compileError);
            if (vertexLibrary == 0 || fragmentLibrary == 0)
            {
                error = MetalNative.DescribeError(compileError);
                return false;
            }

            var newFunction = MetalNative.Selector("newFunctionWithName:");
            var vertexFunction = MetalNative.Send(vertexLibrary, newFunction, MetalNative.NsString("fullscreen_vs"));
            var fragmentFunction = MetalNative.Send(fragmentLibrary, newFunction, MetalNative.NsString(fragmentEntry));
            if (vertexFunction == 0 || fragmentFunction == 0)
            {
                error = "the present entry points are missing from the compiled libraries";
                return false;
            }

            var descriptor = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLRenderPipelineDescriptor"), MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), vertexFunction);
            MetalNative.SendVoid(descriptor, MetalNative.Selector("setFragmentFunction:"), fragmentFunction);
            var attachment = MetalNative.SendAtIndex(
                MetalNative.Send(descriptor, MetalNative.Selector("colorAttachments")),
                MetalNative.Selector("objectAtIndexedSubscript:"),
                0);
            MetalV2Native.SendVoidValue(
                attachment, MetalNative.Selector("setPixelFormat:"), (nuint)pixelFormat);
            nint pipelineError = 0;
            pipeline = MetalNative.Send(
                device,
                MetalNative.Selector("newRenderPipelineStateWithDescriptor:error:"),
                descriptor,
                ref pipelineError);
            MetalNative.SendVoid(descriptor, MetalNative.Selector("release"));
            MetalNative.SendVoid(vertexFunction, MetalNative.Selector("release"));
            MetalNative.SendVoid(fragmentFunction, MetalNative.Selector("release"));
            if (pipeline == 0)
            {
                error = MetalNative.DescribeError(pipelineError);
                return false;
            }

            return true;
        }
        finally
        {
            if (options != 0)
            {
                MetalNative.SendVoid(options, MetalNative.Selector("release"));
            }

            if (vertexLibrary != 0)
            {
                MetalNative.SendVoid(vertexLibrary, MetalNative.Selector("release"));
            }

            if (fragmentLibrary != 0)
            {
                MetalNative.SendVoid(fragmentLibrary, MetalNative.Selector("release"));
            }
        }
    }

    private static nint CreateLinearSampler(nint device)
    {
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLSamplerDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        try
        {
            MetalV2Native.SendVoidValue(
                descriptor, MetalNative.Selector("setMinFilter:"), (nuint)Mtl.SamplerFilterLinear);
            MetalV2Native.SendVoidValue(
                descriptor, MetalNative.Selector("setMagFilter:"), (nuint)Mtl.SamplerFilterLinear);
            return MetalNative.Send(device, MetalNative.Selector("newSamplerStateWithDescriptor:"), descriptor);
        }
        finally
        {
            MetalNative.SendVoid(descriptor, MetalNative.Selector("release"));
        }
    }

    private static void ReportPresentationStatistics(bool presented = true)
    {
        if (presented) Interlocked.Increment(ref _hostPresents);
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_statsTimestamp == 0)
        {
            _statsTimestamp = now;
            return;
        }

        if (now - _statsTimestamp < 2 * System.Diagnostics.Stopwatch.Frequency)
        {
            return;
        }

        var seconds = (now - _statsTimestamp) / (double)System.Diagnostics.Stopwatch.Frequency;
        var guestFrames = Interlocked.Exchange(ref _presentedGuestFrames, 0);
        var hostPresents = Interlocked.Exchange(ref _hostPresents, 0);
        _statsTimestamp = now;
        var guestFps = (guestFrames / seconds).ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
        var hostHz = (hostPresents / seconds).ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
        var window = seconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);
        var published = Interlocked.Exchange(ref _publishedGuestFrames, 0);
        var blocked = Interlocked.Exchange(ref _adoptionsBlocked, 0);
        // Where a flip stops. published=0 on its own cannot tell a title that never flips
        // from one whose every flip is released or discarded on the way to the presenter,
        // and those want opposite investigations.
        var flips = _host?.TakeFlipCounters() ?? default;
        Console.Error.WriteLine(
            $"[METAL-PRESENT] guest_frames={guestFrames} guest_fps={guestFps} host_presents={hostPresents} " +
            $"host_hz={hostHz} seconds={window} published={published} adopt_blocked={blocked} " +
            $"flip_attempts={flips.Attempts} flip_no_buffer={flips.WithoutBuffer} " +
            $"flip_discarded={flips.Discarded} flip_published={flips.Published} " +
            $"flip_refreshed={flips.Refreshed} {DescribeGpuMemory()}");
    }

    private static readonly nint SelCurrentAllocatedSize = MetalNative.Selector("currentAllocatedSize");

    // The device's whole allocation (MTLDevice.currentAllocatedSize: every buffer, texture
    // and heap the process holds, including the no-copy guest windows) next to what the
    // image cache itself believes it holds, so a growth in one can be told from the other.
    private static string DescribeGpuMemory()
    {
        const double MiB = 1024d * 1024d;
        var allocated = _device == 0 ? 0UL : MetalV2Native.SendValue64(_device, SelCurrentAllocatedSize);
        var cache = _host?.ImageCache;
        return string.Create(
            System.Globalization.CultureInfo.InvariantCulture,
            $"gpu_alloc_mb={allocated / MiB:F0} gpu_images={cache?.ImageCount ?? 0} " +
            $"gpu_image_cache_mb={(cache?.TotalUsedMemory ?? 0) / MiB:F0}");
    }

    // The flip seam the command stream publishes through.
    private sealed class FlipTarget : IMetalFlipTarget
    {
        // Flip packets suspend while the queue is full, as in the Vulkan presenter.
        public bool HasFlipSlot()
        {
            lock (_frameGate)
            {
                return _pendingFrames.Count < MaxPendingFrames || TryPromotePendingLocked();
            }
        }

        public void PublishGuestFrame(nint texture, uint width, uint height, ulong requestId, ulong serial, bool isHdr)
        {
            // The presenter owns one retain per queued or presented frame.
            MetalNative.Send(texture, MetalNative.Selector("retain"));
            lock (_frameGate)
            {
                _pendingFrames.Enqueue(new PendingFrame(texture, width, height, requestId, serial, isHdr));
                _ = TryPromotePendingLocked();
                _splashHidden = true;
            }

            SignalFlipPacer();

            Interlocked.Increment(ref _guestFrameSequence);
            Interlocked.Increment(ref _publishedGuestFrames);
        }
    }
}
