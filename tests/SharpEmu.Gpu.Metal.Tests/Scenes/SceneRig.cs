// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// A whole emulated GPU for one scene: guest memory, the Metal v2 render host with its
/// command stream and draw translation, and the v2 backend as the process's guest GPU.
/// A scene writes its programs, buffers and textures into guest memory, submits a PM4
/// stream exactly as the guest's command buffer would carry it, and reads its targets back
/// through the host's synchronized guest reads.
/// </summary>
internal sealed unsafe class SceneRig : IDisposable
{
    private const BindingFlags Statics = BindingFlags.Static | BindingFlags.NonPublic;

    private static readonly FieldInfo HeaderTable = typeof(AgcExports).GetField("_shaderHeadersByCode", Statics)!;
    private static readonly FieldInfo FatalHook = typeof(SharpEmu.Libs.Gpu.Scheduling.GpuFatal).GetField("OnFatal", Statics)!;
    private static readonly FieldInfo[] BackendFields =
        new[] { "_factory", "_backendName", "_instance" }.Select(name => typeof(GuestGpu).GetField(name, Statics)!).ToArray();

    private readonly nint _queue;
    private readonly SceneAddressSpace _space = new();
    private readonly PageGuard _pages;
    // One registry for every live rig: a scene may hold two rigs at once, the hook takes
    // one manager, and every rig's memory sits at the same guest span.
    private static readonly object RegistryGate = new();
    private static GuestGpuMemory? _registry;
    private static int _registryUsers;
    private readonly MetalGpuWorkerRelay _relay = new(static () => { });
    private readonly object?[] _savedBackend;
    private readonly object? _savedFatal;
    private readonly List<ulong> _registeredShaders = [];
    private readonly Dictionary<uint[], ulong> _programs = new(ReferenceEqualityComparer.Instance);
    private ulong _submissionId = 1;

    /// <summary>The registered copy of a program, placed on first use.</summary>
    public ulong Program(uint[] code)
    {
        if (!_programs.TryGetValue(code, out var address))
        {
            address = RegisterShader(code);
            _programs[code] = address;
        }

        return address;
    }

    public SceneRig(nint device)
    {
        _savedBackend = BackendFields.Select(field => field.GetValue(null)).ToArray();
        // A loud refusal ends the process in the emulator; here it fails the scene instead.
        _savedFatal = FatalHook.GetValue(null);
        FatalHook.SetValue(null, (Action<string>)(message =>
        {
            lock (Fatals)
            {
                Fatals.Add(message);
            }

            throw new SceneFatalException(message);
        }));
        BackendFields[0].SetValue(null, (Func<IGuestGpuBackend>)(() => new MetalV2Backend()));
        BackendFields[1].SetValue(null, "Metal");
        BackendFields[2].SetValue(null, new MetalV2Backend());
        _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        _pages = new PageGuard(_space);
        // The guest GPU memory registry the emulator attaches at boot: the device-address
        // page table a shader dereferences raw guest pointers through is built from its
        // spans, so the scene's block is registered as the guest's mapped memory.
        lock (RegistryGate)
        {
            if (_registryUsers++ == 0)
            {
                _registry = new GuestGpuMemory(new SceneAddressSpace());
                _registry.Register(SceneMemory.Base, SceneMemory.Size, SharpEmu.HLE.GuestPageProtection.Read | SharpEmu.HLE.GuestPageProtection.Write);
                GuestGpuMemoryHook.Attach(_registry);
            }
        }
        Host = new MetalRenderHost(device, _queue, Memory, Memory, _pages, _relay);
        _relay.BindCurrentThread();
        Host.InitializeCommandStream(new FlipTarget());
    }

    public SceneMemory Memory { get; } = new();

    /// <summary>Every GPU fatal raised while this rig ran (each also threw <see cref="SceneFatalException"/>).</summary>
    public List<string> Fatals { get; } = [];

    public MetalRenderHost Host { get; }

    private static ConcurrentDictionary<ulong, ulong> Headers => (ConcurrentDictionary<ulong, ulong>)HeaderTable.GetValue(null)!;

    /// <summary>
    /// Places a program in guest memory and registers its AGC shader header, as
    /// sceAgcCreateShader leaves it: code size, user-data block and input semantics.
    /// </summary>
    public ulong RegisterShader(uint[] code, uint[]? inputSemantics = null)
    {
        var codeAddress = Memory.Upload<uint>(code, 256);
        // The user-data block: no direct resources (offset table null, count 0 at +0x2C).
        var userData = Memory.Upload(new byte[0x40], 16);
        var semantics = inputSemantics is { Length: > 0 } ? Memory.Upload<uint>(inputSemantics, 16) : 0UL;
        var header = new byte[0x60];
        var words = MemoryMarshal.Cast<byte, uint>(header.AsSpan());
        words[0] = 0x34333231;
        words[1] = 0x18;
        BitConverter.TryWriteBytes(header.AsSpan(0x08), userData);
        BitConverter.TryWriteBytes(header.AsSpan(0x10), codeAddress);
        BitConverter.TryWriteBytes(header.AsSpan(0x30), semantics);
        BitConverter.TryWriteBytes(header.AsSpan(0x44), (uint)(code.Length * 4));
        BitConverter.TryWriteBytes(header.AsSpan(0x50), (ushort)(inputSemantics?.Length ?? 0));
        var headerAddress = Memory.Upload(header, 16);
        Headers[codeAddress] = headerAddress;
        _registeredShaders.Add(codeAddress);
        return codeAddress;
    }

    /// <summary>Writes the stream into guest memory, runs it on the graphics queue to completion and waits for the GPU.</summary>
    public void Submit(ScenePm4 stream) => Submit(stream.ToArray());

    /// <summary>Runs <paramref name="words"/> on the graphics queue (0) or a compute queue (0x20 to 0x57).</summary>
    public void Submit(uint[] words, int queue = 0)
    {
        Enqueue(words, queue);
        Drain();
    }

    /// <summary>Hands <paramref name="words"/> to the command stream without running it.</summary>
    public void Enqueue(uint[] words, int queue = 0)
    {
        var address = Memory.Upload<uint>(words, 256);
        if (queue == 0)
        {
            Host.CommandStream.EnqueueGraphics(address, (uint)words.Length, _submissionId++, null);
        }
        else
        {
            Host.CommandStream.EnqueueCompute((uint)queue, address, (uint)words.Length, _submissionId++, null);
        }
    }

    /// <summary>The relay the rig's host posts GPU-queue work through, for driving the worker's own pass.</summary>
    internal MetalGpuWorkerRelay Relay => _relay;

    /// <summary>Records all queued packets without waiting for their GPU work to finish.</summary>
    public void RecordPending()
    {
        var recorded = 0;
        if (!Host.CommandStream.TryEnqueueControlBarrier(
                () => Volatile.Write(ref recorded, 1), () => Volatile.Write(ref recorded, -1)))
            throw new InvalidOperationException("The scene could not enqueue its recording barrier.");
        var deadline = DateTime.UtcNow.AddSeconds(30);
        do
        {
            if (DateTime.UtcNow > deadline || Volatile.Read(ref recorded) < 0)
                throw new TimeoutException("The scene's recording barrier did not finish.");
            Host.RunPendingCommands();
            Host.RunCommandStreamSlices(64);
            Thread.Yield();
        } while (Volatile.Read(ref recorded) != 1 || Host.CommandThreadHasWork || _relay.HasPendingCommands);
    }

    /// <summary>Runs the worker loop until no submission is runnable, then waits for the GPU.</summary>
    public void Drain()
    {
        // HasPending excludes a slice the producer is currently interpreting. A
        // trailing control barrier proves it handed every earlier item to the worker.
        var producerDrained = Host.UsesCommandThread ? 0 : 1;
        if (Host.UsesCommandThread && !Host.CommandStream.TryEnqueueControlBarrier(
                () => Volatile.Write(ref producerDrained, 1),
                () => Volatile.Write(ref producerDrained, -1)))
            throw new InvalidOperationException("The scene could not enqueue its producer barrier.");
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (true)
        {
            if (DateTime.UtcNow > deadline || Volatile.Read(ref producerDrained) < 0)
                throw new TimeoutException("The scene's command stream did not finish its producer barrier.");
            Host.RunPendingCommands();
            Host.RunCommandStreamSlices(64);
            if (Host.CommandStream.HasUnblockedPending || Host.CommandThreadHasWork || _relay.HasPendingCommands)
            {
                continue;
            }

            Host.CommitCoalescedFlush();
            Host.FlushAndWaitForGpu();
            Host.PollGpuCompletions();
            // A producer may have consumed the final packet while its worker items
            // still own the draws and result copies. Stream-empty alone is not done.
            if (Volatile.Read(ref producerDrained) == 1 && !Host.CommandStream.HasPending &&
                !Host.CommandThreadHasWork && !_relay.HasPendingCommands)
            {
                return;
            }

            if (DateTime.UtcNow > deadline)
            {
                throw new TimeoutException("The scene's command stream stayed blocked.");
            }

            Host.CommandStream.RetryBlocked();
        }
    }

    /// <summary>Guest bytes after every GPU write to them landed (the host's synchronized read).</summary>
    public byte[] Read(ulong address, int length)
    {
        var bytes = new byte[length];
        if (!Host.TryReadGuest(address, bytes))
        {
            throw new InvalidOperationException($"The scene read failed: 0x{address:X16}+0x{length:X}.");
        }

        return bytes;
    }

    public T[] Read<T>(ulong address, int count) where T : unmanaged =>
        MemoryMarshal.Cast<byte, T>(Read(address, count * Marshal.SizeOf<T>())).ToArray();

    public void Dispose()
    {
        try
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
        }
        finally
        {
            foreach (var code in _registeredShaders)
            {
                Headers.TryRemove(code, out _);
            }

            FatalHook.SetValue(null, _savedFatal);
            for (var index = 0; index < BackendFields.Length; index++)
            {
                BackendFields[index].SetValue(null, _savedBackend[index]);
            }

            lock (RegistryGate)
            {
                if (--_registryUsers == 0)
                {
                    GuestGpuMemoryHook.Attach(null);
                    _registry!.Dispose();
                    _registry = null;
                }
            }

            _pages.Dispose();
            Memory.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private sealed class FlipTarget : IMetalFlipTarget
    {
        public bool HasFlipSlot() => true;

        public void PublishGuestFrame(nint texture, uint width, uint height, ulong requestId, ulong serial, bool isHdr)
        {
        }
    }
}

/// <summary>A GPU fatal (a loud refusal) raised by the emulated GPU while a scene ran.</summary>
internal sealed class SceneFatalException(string message) : Exception(message);
