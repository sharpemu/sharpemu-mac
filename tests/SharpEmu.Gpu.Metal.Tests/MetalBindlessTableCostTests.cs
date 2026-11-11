// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The host side of a bindless image table the size GTA V's ray-traced pass binds: every
/// distinct descriptor resolved to a texture view, the device table of handles written, and
/// the textures declared to an encoder. The first bind creates the textures; later binds of the
/// same table find them and leave the device table as it is; a new table of the same descriptors
/// takes them from the lookup memo, and a pass after one image changed redoes that entry alone.
/// Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalBindlessTableCostTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong TextureBase = GuestBase + 0x10_0000;
    // Streamed textures 64 KiB apart, as a title's are: the image cache indexes them by
    // 1 MiB page, so this many share each page.
    private const ulong TextureStride = 0x1_0000;
    private const int Entries = 5120;
    private const int Distinct = 4441;
    private const int SteadyBinds = 50;
    private const BindingFlags Members = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    [Fact]
    public void AFiveThousandEntryTableIsLookedUpOnlyWhileTheImageSetChanges()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Bindless table binds are measured on the device only behind the exclusive device gate.");
            return;
        }

        Assert.True(OperatingSystem.IsMacOS());
        var device = MetalNative.MTLCreateSystemDefaultDevice();
        Assert.NotEqual(0, device);
        var queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        using var backing = new Backing();
        var pages = new PageGuard(new AddressSpace());
        var host = new MetalRenderHost(device, queue, backing, backing, pages, new MetalGpuWorkerRelay(static () => { }));
        try
        {
            var info = new ShaderResourceInfo();
            info.Images.Add(new ImageResource
            {
                ResourceClass = SharpEmu.ShaderCompiler.Resources.ImageResourceClass.Sampled,
                NumericClass = ImageNumericClass.Float,
                Dimension = ImageDimension.Dim2D,
                Read = true,
                BindlessSlot = 0,
            });
            var layout = BindingLayout.Allocate(info, [], false, false, false);
            Assert.Equal(1u, layout.BindlessImageTableCount);

            var descriptors = new uint[Distinct][];
            for (var texture = 0; texture < Distinct; texture++)
                descriptors[texture] = Texture(TextureBase + (ulong)texture * TextureStride);
            var entries = new int[Entries];
            for (var entry = 0; entry < Entries; entry++)
                entries[entry] = entry % 9 == 8 ? BindlessImageTable.NullEntry : entry % Distinct;
            var table = new BindlessImageTable
            {
                Slot = 0,
                VariantImages = [0],
                Descriptors = descriptors,
                DescriptorVariants = new byte[Distinct],
                Entries = entries,
                Address = GuestBase,
            };
            var snapshot = new ResourceSnapshot { BindlessImages = [table] };

            var hostType = typeof(MetalRenderHost);
            var stageType = hostType.GetNestedType("MetalStageBindings", BindingFlags.NonPublic)!;
            var stage = Activator.CreateInstance(stageType, nonPublic: true)!;
            stageType.GetMethod("EnsureCapacity")!.Invoke(stage, [1, 0, 0, 0]);
            stageType.GetField("Program")!.SetValue(stage, new ShaderProgramInfo { Hash = 0x9FBC, Bindings = layout });
            stageType.GetField("ImageCount")!.SetValue(stage, 1);
            // The root's null texture: any handle, since nothing executes.
            ((ulong[])stageType.GetField("TextureIds")!.GetValue(stage)!)[0] = 1;
            var bind = hostType.GetMethod("BindBindlessImages", Members)!;
            var declare = hostType.GetMethod("DeclareBindlessImages", Members)!;

            TimeSpan Bind(out TimeSpan declared)
            {
                var watch = Stopwatch.StartNew();
                using (host.BeginPreparation())
                {
                    bind.Invoke(host, [stage, info, snapshot]);
                    var bound = watch.Elapsed;
                    var buffer = MetalNative.Send(queue, MetalNative.Selector("commandBuffer"));
                    var encoder = MetalNative.Send(buffer, MetalNative.Selector("computeCommandEncoder"));
                    watch.Restart();
                    declare.Invoke(host, [encoder, false, (nuint)0]);
                    declared = watch.Elapsed;
                    MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
                    return bound;
                }
            }

            var cold = Bind(out var coldDeclare);
            Assert.Equal((1L, (long)Distinct, (long)Distinct), Counts(host.TakeBindlessCost()));
            var addresses = (ulong[])stageType.GetField("BindlessTableAddresses")!.GetValue(stage)!;
            var address = addresses[0];
            Assert.NotEqual(0ul, address);

            var steady = TimeSpan.Zero;
            var steadyDeclare = TimeSpan.Zero;
            for (var pass = 0; pass < SteadyBinds; pass++)
            {
                steady += Bind(out var declared);
                steadyDeclare += declared;
            }

            var (measured, tables, resolved, lookups) = host.TakeBindlessCost();
            Assert.Equal((long)SteadyBinds, tables);
            Assert.Equal((long)SteadyBinds * Distinct, resolved);
            // The first bind created the textures, so the next looks every descriptor up once
            // more with the image set holding still; every later bind reuses what it found.
            Assert.Equal((long)Distinct, lookups);
            // Nothing the table names changed, so the same device table serves every bind.
            Assert.Equal(address, addresses[0]);

            output.WriteLine($"{Entries} entries, {Distinct} textures: first bind {cold.TotalMilliseconds:F2} ms " +
                $"(declare {coldDeclare.TotalMilliseconds:F3} ms); steady bind " +
                $"{steady.TotalMilliseconds / SteadyBinds:F3} ms (resolve {measured.TotalMilliseconds / SteadyBinds:F3} ms, " +
                $"declare {steadyDeclare.TotalMilliseconds / SteadyBinds:F3} ms) per dispatch.");
        }
        finally
        {
            host.ImageCache.StopWatching();
            host.Dispose();
            pages.Dispose();
            MetalNative.SendVoid(queue, MetalNative.Selector("release"));
        }
    }

    // A table the materializer makes anew for descriptors another table already resolved - a small table is a new
    // object on every draw - takes their images and views from the lookup memo instead of looking them up again.
    [Fact]
    public void ANewTableOfDescriptorsAnotherTableResolvedLooksNothingUp()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig();
        const int count = 96;
        var first = rig.Table(count);
        rig.Bind(first);
        rig.Bind(first);
        rig.Host.TakeBindlessCost();

        var second = rig.Table(count);
        rig.Bind(second);
        Assert.Equal((1L, (long)count, 0L), Counts(rig.Host.TakeBindlessCost()));
        Assert.Equal(rig.ResourceIds(first), rig.ResourceIds(second));
    }

    // A table whose memo missed because one of its images changed (the binding epoch moved) resolves that entry again
    // and leaves the others as they were: their stamps still hold.
    [Fact]
    public void APassAfterOneImageChangedResolvesOnlyThatEntryAgain()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig();
        const int count = 96;
        var table = rig.Table(count);
        rig.Bind(table);
        rig.Bind(table);
        rig.Bind(table);
        rig.Host.TakeBindlessCost();
        var skipped = rig.Host.BindlessIncrementalSkipped;

        // A pass writes texture 7: its image is GPU-written now and its binding version moves, which moves the
        // binding epoch every table memo reads.
        rig.Host.ImageCache.GetImage(rig.Field<ResourceSlotIdentifier[]>(table, "Images")[7]).MarkGpuModified();
        rig.Bind(table);
        Assert.Equal((1L, (long)count, 0L), Counts(rig.Host.TakeBindlessCost()));
        Assert.Equal(count - 1, rig.Host.BindlessIncrementalSkipped - skipped);
        Assert.DoesNotContain(0ul, rig.ResourceIds(table));
        // A texture a pass wrote is a hazard: declared to each encoder that reads it, the others resident only.
        Assert.Equal([rig.Field<nint[]>(table, "Textures")[7]], rig.Field<List<nint>>(table, "Declared"));

        // Nothing changed since: the table memo serves the next bind without a pass.
        rig.Bind(table);
        Assert.Equal(count - 1, rig.Host.BindlessIncrementalSkipped - skipped);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1")
        {
            return true;
        }

        output.WriteLine("Bindless table binds run on the device only behind the exclusive device gate.");
        return false;
    }

    // A render host and one compute stage with a single sampled-image bindless table, bound as BindResources would.
    private sealed class Rig : IDisposable
    {
        private readonly nint _queue;
        private readonly Backing _backing = new();
        private readonly PageGuard _pages = new(new AddressSpace());
        private readonly ShaderResourceInfo _info = new();
        private readonly object _stage;
        private readonly MethodInfo _bind;

        public Rig()
        {
            Assert.True(OperatingSystem.IsMacOS());
            var device = MetalNative.MTLCreateSystemDefaultDevice();
            Assert.NotEqual(0, device);
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            Host = new MetalRenderHost(device, _queue, _backing, _backing, _pages, new MetalGpuWorkerRelay(static () => { }));
            _info.Images.Add(new ImageResource
            {
                ResourceClass = SharpEmu.ShaderCompiler.Resources.ImageResourceClass.Sampled,
                NumericClass = ImageNumericClass.Float,
                Dimension = ImageDimension.Dim2D,
                Read = true,
                BindlessSlot = 0,
            });
            var layout = BindingLayout.Allocate(_info, [], false, false, false);
            var stageType = typeof(MetalRenderHost).GetNestedType("MetalStageBindings", BindingFlags.NonPublic)!;
            _stage = Activator.CreateInstance(stageType, nonPublic: true)!;
            stageType.GetMethod("EnsureCapacity")!.Invoke(_stage, [1, 0, 0, 0]);
            stageType.GetField("Program")!.SetValue(_stage, new ShaderProgramInfo { Hash = 0x9FBD, Bindings = layout });
            stageType.GetField("ImageCount")!.SetValue(_stage, 1);
            ((ulong[])stageType.GetField("TextureIds")!.GetValue(_stage)!)[0] = 1;
            _bind = typeof(MetalRenderHost).GetMethod("BindBindlessImages", Members)!;
        }

        public MetalRenderHost Host { get; }

        // count distinct textures, one entry each.
        public BindlessImageTable Table(int count)
        {
            var descriptors = new uint[count][];
            for (var texture = 0; texture < count; texture++)
                descriptors[texture] = Texture(TextureBase + (ulong)texture * TextureStride);
            return new BindlessImageTable
            {
                Slot = 0,
                VariantImages = [0],
                Descriptors = descriptors,
                DescriptorVariants = new byte[count],
                Entries = Enumerable.Range(0, count).ToArray(),
                Address = GuestBase,
            };
        }

        public void Bind(BindlessImageTable table)
        {
            using (Host.BeginPreparation())
            {
                _bind.Invoke(Host, [_stage, _info, new ResourceSnapshot { BindlessImages = [table] }]);
            }
        }

        // The texture handles the table's device table names, entry by entry.
        public ulong[] ResourceIds(BindlessImageTable table) => Field<ulong[]>(table, "ResourceIds").ToArray();

        // A field of the table's device table.
        public T Field<T>(BindlessImageTable table, string name)
        {
            var tables = (System.Collections.IDictionary)typeof(MetalRenderHost).GetField("_bindlessTables", Members)!.GetValue(Host)!;
            var device = tables[table]!;
            return (T)device.GetType().GetField(name)!.GetValue(device)!;
        }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            _pages.Dispose();
            _backing.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private static (long Tables, long Descriptors, long Lookups) Counts(
        (TimeSpan Elapsed, long Tables, long Descriptors, long Lookups) cost) =>
        (cost.Tables, cost.Descriptors, cost.Lookups);

    // An 8x8 linear RGBA8 2D texture.
    private static uint[] Texture(ulong address)
    {
        var shifted = address >> 8;
        const uint last = 7;
        return
        [
            (uint)shifted,
            ((uint)(shifted >> 32) & 0xFF) | ((uint)GuestPixelFormat.Bits8_8_8_8UNorm << 20) | ((last & 0x3) << 30),
            ((last >> 2) & 0xFFF) | (last << 14),
            DescriptorConstants.IdentityImageSwizzle | ((uint)GuestTileMode.Linear << 20) | ((uint)GuestImageType.Color2D << 28),
            0, 0, 0, 0,
        ];
    }

    private sealed class AddressSpace : IGuestAddressSpace
    {
        public ulong ProtectionPageSize => 4096;
        public bool TryProtect(ulong address, ulong size, GuestPageProtection protection) => true;
        public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();
        public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();
        public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) => throw new NotSupportedException();
        public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();
        public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();
        public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
    }

    // One page-aligned native block seen as guest memory at GuestBase, repeated across the
    // guest range the textures span: each texture reads the same bytes, which is all a
    // measurement of binding needs.
    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        public const ulong Size = 4 * 1024 * 1024;
        private const ulong Reach = 512UL << 20;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Fill(_block, (nuint)Size, 0x5A);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private static ulong Offset(ulong address) => (address - GuestBase) % Size;

        private bool Contains(ulong address, ulong length) =>
            address >= GuestBase && address - GuestBase < Reach && Offset(address) + length <= Size;

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, Size);

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            alias = Contains(address, size) ? (ulong)_block + Offset(address) : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => TryWrite(address, data);

        public bool TryReadBacking(ulong address, Span<byte> data) => TryRead(address, data);

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!Contains(virtualAddress, (ulong)destination.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + Offset(virtualAddress), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
        {
            if (!Contains(virtualAddress, (ulong)source.Length)) return false;
            source.CopyTo(new Span<byte>((byte*)_block + Offset(virtualAddress), source.Length));
            return true;
        }

        public bool CanRead(ulong address, ulong size) => Contains(address, size);
    }
}
