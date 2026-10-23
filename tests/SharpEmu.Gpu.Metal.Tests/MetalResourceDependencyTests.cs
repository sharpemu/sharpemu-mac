// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.Tests.Scenes;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalResourceDependencyTests
{
    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
    private static long _nextMapping;
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void UnobservedBackingNativeAndStagedRangesEnrollAndBypassExactly(bool native, bool staged)
    {
        using var rig = new Rig(native, staged);
        var source = (IResourceDependencySource)rig.Host;
        rig.Host.ResourceGenerationsEnabled = false;
        Assert.False(source.TryCaptureDependency(rig.Address, BitConverter.GetBytes(0x8000u), false, out _));
        Assert.Equal(ResourceCaptureFailure.NativeWatch, source.CaptureFailure.Reason);
        rig.Host.ResourceGenerationsEnabled = true;
        using var dependency = Capture(rig);
        Assert.Equal(ResourceDependencyStatus.Current, dependency.Validate());
        rig.Run();
        rig.Run();
        Assert.Equal(1, rig.Cache.BypassedCalls);
        rig.Memory.TryWrite(rig.Address + 0x4000, BitConverter.GetBytes(88u));
        rig.Run();
        Assert.Equal(2, rig.Cache.BypassedCalls);
        rig.Write(99);
        Assert.Equal(ResourceDependencyStatus.CpuWrite, dependency.Validate());
        rig.Run();
        Assert.Equal(99u, rig.Snapshot.FlattenedResourceTable[0]);
        Assert.Equal(2, rig.Cache.FullPathCalls);
        Assert.Contains(native ? "NativeMapped/None" : staged ? "Staged/None" : "GuestBacking/None", rig.Host.TakeDependencyCaptureReport());
    }

    [Fact]
    public void AliasWritesAndReleasedNativeWatchesCannotCertifyOldBytes()
    {
        using var rig = new Rig();
        using var first = Capture(rig);
        // A second native VA cannot inherit the first VA's page guard. The
        // provider proof expires at mapping admission and refuses recapture.
        Assert.True(rig.Memory.Views.TryMapReservedRange(rig.AliasAddress, Rig.Size, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(rig.Memory.TryWrite(rig.AliasAddress, BitConverter.GetBytes(11u)));
        Assert.Equal(ResourceDependencyStatus.Mapping, first.Validate());
        var source = (IResourceDependencySource)rig.Host;
        Assert.False(source.TryCaptureDependency(rig.Address, BitConverter.GetBytes(11u), false, out _));
        Assert.Equal(ResourceCaptureFailure.NativeAlias, source.CaptureFailure.Reason);
        Assert.True(rig.Memory.Views.Unmap(rig.AliasAddress, Rig.Size, out _));
        rig.Write(0x8000);
        using var second = Capture(rig);
        Assert.True(rig.Manager.WriteGenerations.ReleaseFaultWatch(rig.Address));
        Assert.Equal(ResourceDependencyStatus.CpuWrite, second.Validate());
        rig.Write(33);
        rig.Run();
        rig.Run();
        Assert.Equal(1, rig.Cache.BypassedCalls);
        Assert.Equal(33u, rig.Snapshot.FlattenedResourceTable[0]);
    }

    [Fact]
    public void PendingGpuWritesAndRemapsRefuseStaleCertificates()
    {
        using var rig = new Rig();
        using var first = Capture(rig);
        rig.Host.NoteGuestMemoryWrite(rig.Address, 4, MetalGpuWriteOrigin.Copy);
        Assert.Equal(ResourceDependencyStatus.CpuWrite, first.Validate());
        Assert.False(((IResourceDependencySource)rig.Host).TryCaptureDependency(rig.Address, BitConverter.GetBytes(0x8000u), false, out _));
        Assert.Equal(ResourceCaptureFailure.PendingWriter, ((IResourceDependencySource)rig.Host).CaptureFailure.Reason);
        rig.Host.FlushAndWaitForGpu();
        using var next = Capture(rig);
        Assert.True(rig.Memory.Views.Unmap(rig.Address, Rig.Size, out _));
        Assert.True(rig.Memory.Views.TryMapReservedRange(rig.Address, Rig.Size, 0, HostPageProtection.ReadWrite, out _));
        Assert.Equal(ResourceDependencyStatus.Mapping, next.Validate());
        // Mapping ABA cannot be hidden by unchanged bytes or a current page generation.
        rig.Run();
        rig.Run();
        Assert.Equal(1, rig.Cache.BypassedCalls);
    }

    [Fact]
    public void AManagedWriteDuringCaptureRejectsTheOriginalPhysicalObservation()
    {
        using var rig = new Rig();
        rig.Memory.AfterCapture = () => rig.Write(77);
        var source = (IResourceDependencySource)rig.Host;
        Assert.False(source.TryCaptureDependency(rig.Address, BitConverter.GetBytes(0x8000u), false, out var dependency));
        Assert.Null(dependency);
        Assert.Equal(ResourceCaptureFailure.Changed, source.CaptureFailure.Reason);
    }

    [Fact]
    public void DisjointGpuEpochChangesShareTheOriginalPhysicalWatch()
    {
        using var rig = new Rig();
        using var first = Capture(rig);
        for (var i = 0; i < 256; i++)
        {
            rig.Host.NoteGuestMemoryWrite(rig.Address + 0x4000, 4, MetalGpuWriteOrigin.Copy);
            using var next = Capture(rig);
            Assert.Same(first, next);
        }
        Assert.Equal(ResourceDependencyStatus.Current, first.Validate());
    }

    [Fact]
    public void DeviceCopyOverAnEnrolledTableReevaluatesThenBypassesTheNewBytes()
    {
        using var rig = new Rig();
        rig.Run();
        rig.Run();
        Assert.Equal(1, rig.Cache.BypassedCalls);
        Assert.True(rig.Memory.TryWrite(rig.Address + 0x4000, BitConverter.GetBytes(88u)));
        rig.Host.CopyGuestMemory(rig.Address, rig.Address + 0x4000, 4, allowHost: false);
        rig.Run();
        Assert.Equal(88u, rig.Snapshot.FlattenedResourceTable[0]);
        rig.Run();
        Assert.Equal(2, rig.Cache.BypassedCalls);
    }

    [Fact]
    public void PendingLabelBytesKeepTheirOrderedValueAndReportTheOverlayClass()
    {
        using var rig = new Rig();
        rig.Run();
        rig.Host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, rig.Address, 55));
        var source = (IResourceDependencySource)rig.Host;
        Assert.False(source.TryCaptureDependency(rig.Address, BitConverter.GetBytes(0x8000u), false, out _));
        Assert.Equal(ResourceRangeClass.Overlay, source.CaptureFailure.Range);
        rig.Run();
        Assert.Equal(55u, rig.Snapshot.FlattenedResourceTable[0]);
        rig.Host.FlushAndWaitForGpu();
        rig.Run();
        rig.Run();
        Assert.True(rig.Cache.BypassedCalls > 0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void IssuedTableWritesPrecedeCaptureAndInvalidateTheOriginalDependency(bool split)
    {
        using var rig = new Rig();
        var ordered = new OrderedCommandStreamHost(rig.Host, split);
        var consumer = split ? new Thread(() => ordered.Consume()) : null;
        consumer?.Start();
        try
        {
            ResourceDependency? original = null;
            ordered.ExecuteGuestMemoryOperation(() =>
            {
                original = Capture(rig);
                rig.Run();
                rig.Run();
                Assert.Equal(1, rig.Cache.BypassedCalls);
            });
            ordered.RecordCommandProcessorWrite(rig.Address, BitConverter.GetBytes(88u), "TABLE_WRITE");
            ordered.FlushAndWait();
            ordered.ExecuteGuestMemoryOperation(() =>
            {
                Assert.Equal(ResourceDependencyStatus.CpuWrite, original!.Validate());
                rig.Run();
                rig.Run();
                Assert.Equal(88u, rig.Snapshot.FlattenedResourceTable[0]);
                Assert.Equal(2, rig.Cache.FullPathCalls);
                original.Dispose();
            });
        }
        finally
        {
            ordered.Stop();
            if (consumer is not null) Assert.True(consumer.Join(TimeSpan.FromSeconds(10)));
        }
    }

    private static ResourceDependency Capture(Rig rig)
    {
        Assert.True(((IResourceDependencySource)rig.Host).TryCaptureDependency(rig.Address,
            BitConverter.GetBytes(0x8000u), false, out var dependency));
        return Assert.IsAssignableFrom<ResourceDependency>(dependency);
    }

    private sealed class Rig : IDisposable
    {
        public const ulong Size = 1024 * 1024;
        private readonly IHostViewMemory _viewsHost = SharpEmu.Host.MacOS.MacOSHost.CreateViewMemory();
        private readonly GuestGpuMemory? _previous = GuestGpuMemoryHook.Current;
        private readonly nint _queue;
        private readonly PageGuard _pages = new(new SceneAddressSpace());
        public readonly ResourceMaterializationCache Cache = new(dependencyBypass: true, verifyDependencyBypass: true);
        public ResourceSnapshot Snapshot = new();
        private ResourceSpecialization _specialization = new();
        private readonly ShaderResourcePlan _plan = Extract(Program(ScalarLoad(0, 0, 16),
            Vop1(8, "VMovB32", 0, Gen5Operand.Scalar(16)), EndProgram(12)), userDataCount: 2);
        private readonly ResourceRuntimeInputs _inputs;
        private readonly ResidentGuestBytesReader _resident;
        public readonly GuestGpuMemory Manager = new(new SceneAddressSpace());
        public ulong Address { get; }
        public ulong AliasAddress => Address + Size;
        public Memory Memory { get; }
        public MetalRenderHost Host { get; }

        public Rig(bool native = false, bool staged = false)
        {
            for (var attempt = 0; attempt < 256; attempt++)
            {
                var candidate = 0xF0_0000_0000 + (ulong)Interlocked.Increment(ref _nextMapping) * 4 * Size;
                if (_viewsHost.ReserveHole(candidate, 2 * Size) != candidate) continue;
                Address = candidate;
                break;
            }
            Assert.NotEqual(0UL, Address);
            var views = new SharedBackingViews(_viewsHost, Size);
            Assert.True(views.TryMapReservedRange(Address, Size, 0, HostPageProtection.ReadWrite, out _));
            Memory = new Memory(views, native);
            Manager.Register(Address, 2 * Size, GuestPageProtection.Read | GuestPageProtection.Write);
            GuestGpuMemoryHook.Attach(null);
            GuestGpuMemoryHook.Attach(Manager);
            _queue = MetalNative.Send(Device.Value, MetalNative.Selector("newCommandQueue"));
            Host = new MetalRenderHost(Device.Value, _queue, Memory, Memory, _pages, new MetalGpuWorkerRelay(static () => { }));
            if (staged)
            {
                var cache = new MetalStagedRangeCache(new StagedBackend(), 4 * Size);
                cache.Acquire(Address, Size, false, 1, 0, out _);
                typeof(MetalRenderHost).GetField("_stagedRanges", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.SetValue(Host, cache);
            }
            Write(0x8000);
            _resident = Host.TryReadResidentGuestBytes;
            GuestWordReader read = Host.TryReadGuestWord;
            _inputs = new() { UserData = [(uint)Address, (uint)(Address >> 32)], ReadMemory = read, ReadCleanMemory = read,
                ReadResidentMemory = _resident, DependencySource = Host };
        }
        public void Write(uint word)
        {
            Manager.MarkCpuWrite(Address, 4);
            Assert.True(Memory.TryWrite(Address, BitConverter.GetBytes(word)));
            Manager.MarkCpuWriteLanded(Address, 4);
        }
        public void Run() => Assert.True(Cache.Materialize(_plan, _inputs, _resident, ref Snapshot, ref _specialization, out _));
        public void Dispose()
        {
            typeof(ResourceMaterializationCache).GetMethod("ClearPackets", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.Invoke(Cache, null);
            Host.Dispose();
            Manager.Dispose();
            GuestGpuMemoryHook.Attach(null);
            GuestGpuMemoryHook.Attach(_previous);
            _pages.Dispose();
            Memory.Views.Dispose();
            _viewsHost.FreeOwnedRange(Address, 2 * Size);
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private sealed class StagedBackend : IStagedRangeBackend
    {
        public (nint Handle, nint Contents, ulong GpuAddress) CreateBuffer(ulong capacity) => (1, 0, 1);
        public void ReleaseBuffer(nint handle) { }
        public void AddWatch(ulong address, ulong size) { }
        public void RemoveWatch(ulong address, ulong size) { }
    }

    private sealed class Memory(SharedBackingViews views, bool native) : ICpuMemory, IGuestBackedSpace, ICpuMappedReadWatchSource
    {
        public SharedBackingViews Views => views;
        public Action? AfterCapture;
        public bool TryRead(ulong address, Span<byte> bytes) => views.TryReadBacking(address, bytes);
        public bool TryWrite(ulong address, ReadOnlySpan<byte> bytes) => views.TryWriteBacking(address, bytes);
        public bool CanRead(ulong address, ulong size) => views.Contains(address, size);
        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong start, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong offset, GuestPageProtection access, out HostViewFailure failure) { failure = default; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => views.Unmap(address, size, out _);
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => !native && views.Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => !native && views.Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => (views.AliasBase, views.Size);
        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias) => views.TryResolveAlias(address, size, out alias);
        public bool TryReadBacking(ulong address, Span<byte> bytes) => TryRead(address, bytes);
        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> bytes) => TryWrite(address, bytes);
        public CpuMappedCopyResult TryReadMappedCopy(ulong address, ulong size, ulong source, Span<byte> bytes, out CpuMappedCopyObservation? observation) =>
            views.TryReadMappedCopy(address, size, source, bytes, out observation);
        public CpuMappedCopyResult TryReadMappedCopyWithWriteWatch(ulong address, ulong size, ulong source, Span<byte> bytes, out CpuMappedCopyObservation? observation)
        {
            var result = views.TryReadMappedCopyWithWriteWatch(address, size, source, bytes, out observation);
            AfterCapture?.Invoke();
            return result;
        }
        public bool IsMappedCopyObservationCurrent(CpuMappedCopyObservation observation) => views.IsMappedCopyObservationCurrent(observation);
        public bool IsMappedCopyWriteWatchCurrent(CpuMappedCopyObservation observation) => views.IsMappedCopyWriteWatchCurrent(observation);
        public void ReleaseMappedCopyWriteWatch(CpuMappedCopyObservation observation) => views.ReleaseMappedCopyWriteWatch(observation);
    }
}
