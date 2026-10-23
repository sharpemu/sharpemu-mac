// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Hashing;

namespace SharpEmu.Gpu.Metal.V2;

// Immutable argument storage. A slab is NEVER patched or recycled while any
// recorded/submitted serial can read it, including unretained command buffers.
internal sealed class MetalRecipeArena(nint device) : IDisposable
{
    internal sealed class Slab(nint buffer, nint contents, ulong gpuAddress)
    {
        internal readonly nint Buffer = buffer, Contents = contents;
        internal readonly ulong GpuAddress = gpuAddress;
        internal ulong Used, LastSerial;
    }
    internal readonly record struct Record(Slab Owner, ulong Offset, int Length)
    {
        internal MetalGuestRange Range => new(Owner.Buffer, Offset, (ulong)Length, Owner.GpuAddress + Offset);
        internal void Use(ulong serial) => Owner.LastSerial = Math.Max(Owner.LastSerial, serial);
    }
    private const ulong SlabBytes = 1024 * 1024;
    private const int MaxSlabs = 16;
    private readonly List<Slab> _slabs = new(), _retired = new();
    private readonly Dictionary<ulong, List<Record>> _records = new();
    internal long Generation { get; private set; }
    internal long Hits, UploadedBytes, Refused;

    internal Slab[] References(ReadOnlySpan<nint> buffers)
    {
        var result = new List<Slab>();
        foreach (var slab in _slabs)
            if (buffers.Contains(slab.Buffer)) result.Add(slab);
        foreach (var slab in _retired)
            if (buffers.Contains(slab.Buffer)) result.Add(slab);
        return result.ToArray();
    }

    internal unsafe bool Intern(ReadOnlySpan<byte> bytes, ulong serial, ulong completed, out Record record)
    {
        record = default;
        if (bytes.Length == 0 || bytes.Length > (int)SlabBytes) return false;
        var hash = XxHash3.HashToUInt64(bytes);
        if (_records.TryGetValue(hash, out var bucket))
            foreach (var candidate in bucket)
                if (candidate.Length == bytes.Length && bytes.SequenceEqual(
                    new ReadOnlySpan<byte>((byte*)candidate.Owner.Contents + candidate.Offset, candidate.Length)))
                {
                    record = candidate; record.Use(serial); Hits++; return true;
                }
        var size = ((ulong)bytes.Length + 255) & ~255UL;
        var slab = _slabs.Count == 0 ? null : _slabs[^1];
        if (slab is null || slab.Used + size > SlabBytes)
        {
            for (var i = _retired.Count - 1; i >= 0; i--)
                if (_retired[i].LastSerial <= completed)
                {
                    MetalNative.SendVoid(_retired[i].Buffer, MetalNative.Selector("release"));
                    _retired.RemoveAt(i);
                }
            if (_slabs.Count >= MaxSlabs)
            {
                // Every cached argument reference carries Generation. Old versions
                // retire by actual serial, not frame count or a triple-buffer guess.
                Generation++;
                _records.Clear(); bucket = null;
                _retired.AddRange(_slabs); _slabs.Clear();
            }
            if (_slabs.Count + _retired.Count >= MaxSlabs * 2) { Refused++; return false; }
            var buffer = MetalNative.SendNewBuffer(device, MetalNative.Selector("newBufferWithLength:options:"),
                (nuint)SlabBytes, Mtl.ResourceStorageModeShared);
            if (buffer == 0) { Refused++; return false; }
            slab = new(buffer, MetalNative.Send(buffer, MetalNative.Selector("contents")),
                MetalNative.SendGpuAddress(buffer, MetalNative.Selector("gpuAddress")));
            _slabs.Add(slab);
        }
        record = new(slab, slab.Used, bytes.Length);
        bytes.CopyTo(new Span<byte>((byte*)slab.Contents + slab.Used, bytes.Length));
        slab.Used += size; record.Use(serial); UploadedBytes += bytes.Length;
        if (bucket is null) _records[hash] = bucket = new();
        bucket.Add(record);
        return true;
    }

    // The host drains its submission prefix before disposing the arena.
    public void Dispose()
    {
        foreach (var slab in _slabs.Concat(_retired))
            MetalNative.SendVoid(slab.Buffer, MetalNative.Selector("release"));
        _slabs.Clear(); _retired.Clear(); _records.Clear();
    }
}
