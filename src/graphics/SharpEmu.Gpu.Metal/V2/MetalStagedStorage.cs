// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Storage bindings whose guest range no one window covers. Such a range is contiguous
/// to the guest but not in the backing - it crosses from one mapping into another whose
/// pages sit elsewhere - so the shader is handed a copy in the upload ring. Two things
/// keep that copy honest. A copy the shader writes is carried back into guest memory,
/// piece by piece through the windows, once the work that writes it has been recorded:
/// without that the stores went nowhere, which is how Silent Hill's lightmap table, filled
/// by a compute scatter, stayed zero and every lightmapped surface came out black. And a
/// copy of bytes the GPU is still writing is gathered on the GPU, after that work, rather
/// than snapshotted by the CPU before it has run.
/// </summary>
/// <remarks>
/// Every binding of the same range shares one copy until the submission ends, which is
/// what the guest sees - one buffer - and keeps a pass of draws over the same table from
/// gathering it once per draw. A copy is dropped as soon as anything else writes its
/// bytes. A binding of a different range over a copy's bytes first has the copy's stores
/// written back, so it reads or writes after them in stream order.
///
/// Only the bytes the shader changed are written back. Each copy keeps a second, pristine
/// copy of what it was staged from, and the writeback kernel stores a byte into guest
/// memory only where the two differ. Copying the whole range back would also put back the
/// staged value of every byte the shader left alone, over whatever anything else wrote
/// there in the meantime.
/// </remarks>
internal sealed unsafe class MetalStagedStorage(MetalGuestMemory memory, MetalStagedRangeCache? cache = null)
{
    private const ulong PageBytes = 4096;
    private const uint ThreadsPerGroup = 256;

    private static readonly nint SelCopyBuffer =
        MetalNative.Selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
    private static readonly nint SelBlitEncoder = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelComputeEncoder = MetalNative.Selector("computeCommandEncoder");
    private static readonly nint SelEndEncoding = MetalNative.Selector("endEncoding");
    private static readonly nint SelSetPipeline = MetalNative.Selector("setComputePipelineState:");
    private static readonly nint SelSetBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
    private static readonly nint SelSetBytes = MetalNative.Selector("setBytes:length:atIndex:");
    private static readonly nint SelDispatch = MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:");
    private static readonly nint SelAlloc = MetalNative.Selector("alloc");
    private static readonly nint SelInit = MetalNative.Selector("init");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelNewLibrary = MetalNative.Selector("newLibraryWithSource:options:error:");
    private static readonly nint SelNewFunction = MetalNative.Selector("newFunctionWithName:");
    private static readonly nint SelNewComputePipeline = MetalNative.Selector("newComputePipelineStateWithFunction:error:");

    private const string WritebackFunction = "staged_store_writeback";

    // One thread per four bytes: a byte the shader changed goes to guest memory, and the
    // pristine copy takes it too, so a later writeback of the same copy sends only what
    // changed after this one.
    private const string WritebackSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct StagedStoreArguments
        {
            ulong copy;
            ulong pristine;
            ulong guest;
            uint bytes;
            uint pad;
        };

        kernel void staged_store_writeback(
            device const uchar* copy [[buffer(0)]],
            device uchar* pristine [[buffer(1)]],
            device uchar* guest [[buffer(2)]],
            constant StagedStoreArguments& args [[buffer(3)]],
            uint id [[thread_position_in_grid]])
        {
            for (uint lane = 0; lane < 4u; lane++)
            {
                uint index = id * 4u + lane;
                if (index >= args.bytes)
                {
                    return;
                }

                uchar value = copy[args.copy + index];
                if (value != pristine[args.pristine + index])
                {
                    guest[args.guest + index] = value;
                    pristine[args.pristine + index] = value;
                }
            }
        }
        """;

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct WritebackArguments
    {
        public ulong Copy;
        public ulong Pristine;
        public ulong Guest;
        public uint Bytes;
        public uint Pad;
    }

    private nint _writebackPipeline;

    // The entries bound in the command buffer being recorded: a ring copy lives here alone,
    // a cached range lives on in the cache and is listed for its stores, flags and sharing.
    private readonly List<MetalStagedRangeCache.Entry> _copies = new();

    public int Count => _copies.Count;

    /// <summary>The copy an earlier binding of exactly this range made, which this one shares.</summary>
    public bool TryShare(ulong address, ulong size, bool written, out MetalGuestRange staging)
    {
        foreach (var copy in _copies)
        {
            if (copy.Address != address || copy.Size != size || copy.Removed)
            {
                continue;
            }

            // A cached copy something has written since is not the guest's bytes any more.
            if (copy.Cached && !copy.Dirty && Volatile.Read(ref copy.Stale) != 0)
            {
                continue;
            }

            if (written)
            {
                // A copy staged for reading has no pristine bytes to tell its stores by.
                if (!copy.Pristine.IsValid)
                {
                    continue;
                }

                copy.Dirty = true;
                copy.Armed = false;
            }

            staging = copy.Staging;
            return true;
        }

        staging = default;
        return false;
    }

    /// <summary>
    /// A cached range was bound for this command buffer: its stores, if the binding
    /// writes, are written back after the work that makes them.
    /// </summary>
    public void Bind(MetalStagedRangeCache.Entry entry, bool written)
    {
        if (!entry.InList)
        {
            entry.InList = true;
            _copies.Add(entry);
        }

        if (written)
        {
            entry.Dirty = true;
            entry.Armed = false;
        }
    }

    /// <summary>
    /// True when the range overlaps stores recorded into another copy. They have to be
    /// written back before a binding of these bytes is recorded.
    /// </summary>
    public bool HasRecordedStoresOver(ulong address, ulong size)
    {
        foreach (var copy in _copies)
        {
            if (copy.Dirty && copy.Armed && Overlaps(copy, address, size))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>True when some copy has recorded stores that guest memory does not have yet.</summary>
    public bool HasRecordedStores
    {
        get
        {
            foreach (var copy in _copies)
            {
                if (copy.Dirty && copy.Armed)
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>
    /// Keeps a staged copy for the later bindings of the same range. The pristine range
    /// holds the same bytes as the copy before any shader stores into it.
    /// </summary>
    public void Add(ulong address, ulong size, MetalGuestRange staging, MetalGuestRange pristine, bool written) =>
        _copies.Add(new MetalStagedRangeCache.Entry
        {
            Address = address,
            Size = size,
            Staging = staging,
            Pristine = pristine,
            Dirty = written,
            InList = true,
        });

    /// <summary>The draw or dispatch the copies were bound for has been recorded.</summary>
    public void Arm()
    {
        foreach (var copy in _copies)
        {
            copy.Armed = true;
        }
    }

    /// <summary>
    /// Something other than these copies wrote the range: a copy over it no longer holds
    /// the guest's bytes. One with stores of its own is kept, since those stores are newer.
    /// </summary>
    public void Forget(ulong address, ulong size)
    {
        if (_copies.Count == 0)
        {
            return;
        }

        // RemoveAll with a lambda over address and size allocated its closure on every call, even
        // for the early return above; this keeps the survivors in order in place.
        var kept = 0;
        for (var index = 0; index < _copies.Count; index++)
        {
            var copy = _copies[index];
            if (copy.Dirty || !Overlaps(copy, address, size))
            {
                _copies[kept++] = copy;
                continue;
            }

            copy.InList = false;
        }

        if (kept != _copies.Count)
        {
            _copies.RemoveRange(kept, _copies.Count - kept);
        }
    }

    /// <summary>
    /// The submission is over and the ring bytes behind a finished copy may be reused. A
    /// copy still waiting for its draw stays, so its stores are still written back.
    /// </summary>
    public void EndSubmission()
    {
        if (_copies.Count == 0)
        {
            return;
        }

        _copies.RemoveAll(copy =>
        {
            if (copy.Dirty && !copy.Armed)
            {
                return false;
            }

            // Stores recorded but never written back are lost, as they always were; a cached
            // copy must not go on standing for guest bytes it never reached.
            if (copy.Cached && copy.Dirty)
            {
                copy.Dirty = false;
                Volatile.Write(ref copy.Stale, 1);
            }

            copy.Armed = false;
            copy.InList = false;
            return true;
        });
    }

    /// <summary>
    /// Stores the bytes every armed copy changed into guest memory through the windows,
    /// with a compute encoder opened on the command buffer after the work that stored.
    /// </summary>
    /// <returns>The ranges written back, which the caller records as GPU writes.</returns>
    public List<(ulong Address, ulong Size)> EncodeWritebacks(nint commandBuffer, ulong submissionSerial = 0)
    {
        var written = new List<(ulong, ulong)>();
        nint encoder = 0;
        try
        {
            foreach (var copy in _copies)
            {
                if (!copy.Dirty || !copy.Armed || copy.Removed)
                {
                    continue;
                }

                // This submission reads the copy's buffers now, so they are not reusable before it completes.
                if (copy.Cached)
                {
                    cache?.Touch(copy, submissionSerial);
                    cache?.MarkSelfWriteback(copy);
                }

                if (encoder == 0)
                {
                    encoder = MetalNative.Send(commandBuffer, SelComputeEncoder);
                    MetalMemoryTelemetry.ComputeEncoders.Add();
                    if (encoder == 0)
                    {
                        throw GpuFatal.Fatal("A Metal compute encoder could not be created for staged storage stores.");
                    }

                    MetalNative.SendVoid(encoder, SelSetPipeline, WritebackPipeline());
                }

                var missed = EncodeWriteback(encoder, copy);
                if (missed != 0)
                {
                    MetalV2Diagnostics.WarnOnce(
                        $"staged-storage-unwritten:{copy.Size:X}",
                        $"A staged storage buffer has bytes no window covers, so the shader's stores to them " +
                        $"stay in the copy: address=0x{copy.Address:X16} size=0x{copy.Size:X} unwritten=0x{missed:X} " +
                        $"({memory.DescribeResolve(copy.Address, copy.Size)}).");
                }

                written.Add((copy.Address, copy.Staging.Size));
            }
        }
        finally
        {
            if (encoder != 0)
            {
                MetalNative.SendVoid(encoder, SelEndEncoding);
            }
        }

        return written;
    }

    /// <summary>The copies whose stores were just encoded are now what guest memory holds.</summary>
    public void MarkWrittenBack()
    {
        foreach (var copy in _copies)
        {
            if (!copy.Armed)
            {
                continue;
            }

            if (copy.Cached && copy.Dirty)
            {
                cache?.OnWrittenBack(copy);
            }

            copy.Dirty = false;
        }
    }

    /// <summary>
    /// Fills a staged copy from the windows on the GPU, after whatever the GPU still has to
    /// write there, then its pristine copy from it, when it has one. Bytes no window covers keep what the
    /// CPU staged. Each step is an encoder of its own, which is what orders them.
    /// </summary>
    public void EncodeGather(nint commandBuffer, ulong address, in MetalGuestRange staging, in MetalGuestRange pristine)
    {
        var size = staging.Size;
        var gather = BlitEncoder(commandBuffer);
        try
        {
            for (var offset = 0UL; offset < size;)
            {
                var run = memory.ResolveWritablePrefix(address + offset, size - offset);
                if (!run.IsValid)
                {
                    offset += Math.Min(size - offset, PageBytes - ((address + offset) & (PageBytes - 1)));
                    continue;
                }

                MetalNative.SendCopyBuffer(
                    gather, SelCopyBuffer,
                    run.Buffer, (nuint)run.Offset,
                    staging.Buffer, (nuint)(staging.Offset + offset), (nuint)run.Size);
                offset += run.Size;
            }
        }
        finally
        {
            MetalNative.SendVoid(gather, SelEndEncoding);
        }

        if (!pristine.IsValid)
        {
            return;
        }

        var snapshot = BlitEncoder(commandBuffer);
        try
        {
            MetalNative.SendCopyBuffer(
                snapshot, SelCopyBuffer,
                staging.Buffer, (nuint)staging.Offset,
                pristine.Buffer, (nuint)pristine.Offset, (nuint)size);
        }
        finally
        {
            MetalNative.SendVoid(snapshot, SelEndEncoding);
        }
    }

    private static nint BlitEncoder(nint commandBuffer)
    {
        var encoder = MetalNative.Send(commandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("A Metal blit encoder could not be created for a staged storage buffer.");
        }

        return encoder;
    }

    // One dispatch per run of the range one window covers. Returns the bytes no window covers.
    private ulong EncodeWriteback(nint encoder, MetalStagedRangeCache.Entry copy)
    {
        var size = copy.Staging.Size;
        var missed = 0UL;
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, copy.Staging.Buffer, 0, 0);
        MetalNative.SendSetBuffer(encoder, SelSetBuffer, copy.Pristine.Buffer, 0, 1);
        for (var offset = 0UL; offset < size;)
        {
            var run = memory.ResolveWritablePrefix(copy.Address + offset, size - offset);
            if (!run.IsValid)
            {
                var step = Math.Min(size - offset, PageBytes - ((copy.Address + offset) & (PageBytes - 1)));
                missed += step;
                offset += step;
                continue;
            }

            var arguments = new WritebackArguments
            {
                Copy = copy.Staging.Offset + offset,
                Pristine = copy.Pristine.Offset + offset,
                Guest = run.Offset,
                Bytes = checked((uint)run.Size),
            };
            MetalNative.SendSetBuffer(encoder, SelSetBuffer, run.Buffer, 0, 2);
            MetalNative.SendSetBytes(encoder, SelSetBytes, (nint)(&arguments), (nuint)sizeof(WritebackArguments), 3);
            var threads = (run.Size + 3) / 4;
            MetalNative.SendDispatch(encoder, SelDispatch,
                new MtlSize { Width = (nuint)((threads + ThreadsPerGroup - 1) / ThreadsPerGroup), Height = 1, Depth = 1 },
                new MtlSize { Width = ThreadsPerGroup, Height = 1, Depth = 1 });
            offset += run.Size;
        }

        return missed;
    }

    private nint WritebackPipeline()
    {
        if (_writebackPipeline != 0)
        {
            return _writebackPipeline;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var device = memory.Device;
            var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), SelAlloc), SelInit);
            MetalNative.ApplyMslCompileOptions(options);
            nint error = 0;
            var library = MetalNative.Send(device, SelNewLibrary, MetalNative.NsString(WritebackSource), options, ref error);
            MetalNative.SendVoid(options, SelRelease);
            if (library == 0)
            {
                throw GpuFatal.Fatal($"The staged storage writeback did not compile: {MetalNative.DescribeError(error)}.");
            }

            var function = MetalNative.Send(library, SelNewFunction, MetalNative.NsString(WritebackFunction));
            MetalNative.SendVoid(library, SelRelease);
            if (function == 0)
            {
                throw GpuFatal.Fatal("The staged storage writeback function is missing.");
            }

            error = 0;
            var pipeline = MetalNative.Send(device, SelNewComputePipeline, function, ref error);
            MetalNative.SendVoid(function, SelRelease);
            if (pipeline == 0)
            {
                throw GpuFatal.Fatal($"The staged storage writeback pipeline did not build: {MetalNative.DescribeError(error)}.");
            }

            _writebackPipeline = pipeline;
            return pipeline;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    private static bool Overlaps(MetalStagedRangeCache.Entry copy, ulong address, ulong size) =>
        address < copy.Address + copy.Size && copy.Address < address + size;
}
