// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// Bulk heap record reads, upstream's VulkanVideoPresenter.BindlessHeap.cs (VisitHeapRuns, 7cf0ac568 / 99f9fe930):
// revalidation reads the records of a view's entries in runs, one guest read per run of nearby records, instead of one
// read per record. SHARPEMU_HEAP_BULK_READS=0 reads each record on its own, as before. A run that cannot be read (a page
// a recorded GPU write may still own, an unmapped one) falls back to the per-record read of each of its entries, which
// waits for such a write, so every entry compares the words its own read would return.
internal sealed partial class MetalRenderHost
{
    internal static readonly bool HeapBulkReadsConfigured = Environment.GetEnvironmentVariable("SHARPEMU_HEAP_BULK_READS") != "0";

    // Tests set it on their own host, so a test of one setting never changes another test's host.
    internal bool? HeapBulkReadsOverride;

    internal bool HeapBulkReadsActive => HeapBulkReadsOverride ?? HeapBulkReadsConfigured;

    // A run spans at most this many bytes and skips at most HeapRunGapBytes between two records it reads.
    internal const uint HeapRunMaxBytes = 64 << 10;
    internal const uint HeapRunGapBytes = 1 << 10;

    private byte[]? _heapRunBytes;

    // Run reads that succeeded and runs that fell back to per-record reads, for tests.
    internal long HeapRunReads, HeapRunFallbacks;

    // The end (exclusive) of the run of sorted record indexes that starts at start: every index in it is below records,
    // the run spans at most HeapRunMaxBytes, and no two neighbors are more than HeapRunGapBytes of records apart.
    internal static int HeapRecordRunEnd(ReadOnlySpan<uint> indexes, int start, uint recordBytes, uint records)
    {
        var first = indexes[start];
        var last = first;
        var end = start + 1;
        while (end < indexes.Length)
        {
            var next = indexes[end];
            if (next >= records || (ulong)(next - last) * recordBytes > HeapRunGapBytes + recordBytes ||
                ((ulong)next - first + 1) * recordBytes > HeapRunMaxBytes)
            {
                break;
            }

            last = next;
            end++;
        }

        return end;
    }

    // What revalidation in runs reads and visits: the host implements it over its views, guest memory and entries
    // (HeapRunVisitor); tests implement it over fakes.
    internal interface IHeapRunVisitor
    {
        // The view's heap as it is now: a resolve may move it.
        BindlessHeapDescriptor Heap(bool sampler);

        bool TryRead(ulong address, Span<byte> destination);

        // The record of the entry at position, read on its own as ReadHeapRecord reads it; the word count.
        int ReadRecord(int position, Span<uint> words);

        // Compares the entry at position with words and resolves it again when they changed; true when it did.
        bool Visit(int position, ReadOnlySpan<uint> words);
    }

    // Visits the entries at positions [from, to) of one heap view, whose sorted record indexes are indexes, each with the
    // words of its record. Records are read in runs (HeapRecordRunEnd) into buffer. A visit that resolved ends its run: the
    // next entry starts a new one, read after that resolve, with the heap as the resolve left it, so no entry compares
    // bytes read before an earlier entry's resolve, and buffer is never read across one. A run that cannot be read falls
    // back to one read per entry, each right before its visit. Entries past the heap's records get no words, as
    // ReadHeapRecord gives them.
    internal static void VisitHeapRuns<T>(ref T visitor, ReadOnlySpan<uint> indexes, int from, int to, bool sampler,
        Span<byte> buffer, Span<uint> scratch) where T : IHeapRunVisitor
    {
        indexes = indexes[..to];
        var start = from;
        while (start < to)
        {
            var heap = visitor.Heap(sampler);
            var bytes = heap.DescriptorBytes;
            if (heap.Base == 0 || indexes[start] >= heap.Records)
            {
                _ = visitor.Visit(start, ReadOnlySpan<uint>.Empty);
                start++;
                continue;
            }

            // The records of a heap are its stride apart; each holds the descriptor in its first bytes.
            var stride = Math.Max(heap.Stride, bytes);
            var end = HeapRecordRunEnd(indexes, start, stride, heap.Records);
            var first = indexes[start];
            var run = buffer[..checked((int)(((ulong)indexes[end - 1] - first) * stride + bytes))];
            if (!visitor.TryRead(heap.Base + (ulong)first * stride, run))
            {
                for (; start < end; start++)
                {
                    _ = visitor.Visit(start, scratch[..visitor.ReadRecord(start, scratch)]);
                }

                continue;
            }

            while (start < end)
            {
                var offset = checked((int)((indexes[start] - first) * stride));
                var resolved = visitor.Visit(start, MemoryMarshal.Cast<byte, uint>(run.Slice(offset, (int)bytes)));
                start++;
                if (resolved)
                {
                    break;
                }
            }
        }
    }

    // The host's side of VisitHeapRuns, over the entries of one view in record order: the view's descriptors are its
    // heap's records, a stride apart (BindlessHeapViewTable.Region.Descriptors).
    private readonly struct HeapRunVisitor(MetalRenderHost owner, HeapEntry[] order, BindlessHeapViewTable.Region region) : IHeapRunVisitor
    {
        public BindlessHeapDescriptor Heap(bool sampler) => region.Descriptors with { Sampler = sampler };

        public bool TryRead(ulong address, Span<byte> destination)
        {
            // The bytes as the per-record read's first attempt sees them; a run any GPU write may still own falls back.
            var read = owner.TryReadResidentGuestBytes(address, destination, clean: true);
            if (read)
            {
                owner.HeapRunReads++;
            }
            else
            {
                owner.HeapRunFallbacks++;
            }

            return read;
        }

        public int ReadRecord(int position, Span<uint> words) => owner.ReadHeapRecord(order[position], words);

        public bool Visit(int position, ReadOnlySpan<uint> words) => owner.RevalidateHeapEntry(order[position], words);
    }

    // The entries of one view, in record order, compared with their records read in runs (HeapRecordRunEnd).
    private void RevalidateHeapRegionInRuns(HeapRegionEntries entries, ulong serial)
    {
        entries.ValidatedSerial = serial;
        // Every entry of a view's list holds that view's region while the view keeps it.
        if (entries.Order.Find(static entry => entry.Region is not null)?.Region is not { } region)
        {
            return;
        }

        var order = entries.Order.ToArray();
        var records = new uint[order.Length];
        for (var position = 0; position < order.Length; position++)
        {
            records[position] = order[position].Index - region.View.First;
        }

        Span<uint> scratch = stackalloc uint[8];
        var visitor = new HeapRunVisitor(this, order, region);
        VisitHeapRuns(ref visitor, records, 0, order.Length, region.Heap.Sampler,
            (_heapRunBytes ??= new byte[HeapRunMaxBytes]).AsSpan(), scratch);
    }

    // One entry compared with the words of its record (RevalidateHeapRegion's step): resolved again when they changed, or
    // when it had no texture the last time; true when it was resolved.
    private bool RevalidateHeapEntry(HeapEntry entry, ReadOnlySpan<uint> words)
    {
        if (entry.Region is null || !_heapEntries.TryGetValue((entry.Slot, entry.Index), out var current) ||
            !ReferenceEquals(current, entry))
        {
            // Released by a resolve of an earlier entry (the view lost its run).
            return false;
        }

        // An R128 T# is its four words; the rest of the T# is zero.
        Span<uint> extended = stackalloc uint[8];
        if (words.Length == 4 && entry.Region is { Heap: { Sampler: false, R128: true } })
        {
            extended.Clear();
            words.CopyTo(extended);
            words = extended;
        }

        // A T# that had no texture when it was last looked at (its memory was not mapped) is looked at again.
        if (words.SequenceEqual(entry.Words) &&
            (entry.Resolved || entry.Slot >= BindlessHeap.ImageKinds.Length ||
             HeapRecordClass(entry, words) == ImageNumericClass.Unsupported))
        {
            return false;
        }

        entry.Words = words.ToArray();
        ResolveHeapEntry(entry);
        return true;
    }
}
