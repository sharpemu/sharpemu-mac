// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.IO.Hashing;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The binary information block a compiled shader can carry after its first two words.
public readonly record struct ShaderBinaryInfo(uint Hash0, uint Hash1)
{
    public const uint MarkerWord = 0xBEEB03FF;
    public const int ByteSize = 28;

    public ulong DeclaredHash => ((ulong)Hash1 << 32) | Hash0;
}

// A program is identified by its content hash, never by its address.
public static class ShaderIdentity
{
    // The declared hash from the marker block, or zero when the code carries none.
    public static bool TryReadDeclaredHash(ICpuMemory memory, ulong codeAddress, out ulong declaredHash)
    {
        declaredHash = 0;
        Span<byte> head = stackalloc byte[2 * sizeof(uint)];
        if (!memory.TryRead(codeAddress, head))
        {
            return false;
        }

        if (BinaryPrimitives.ReadUInt32LittleEndian(head) != ShaderBinaryInfo.MarkerWord)
        {
            return true;
        }

        var infoAddress = codeAddress + ((ulong)BinaryPrimitives.ReadUInt32LittleEndian(head[4..]) + 1) * 2 * sizeof(uint);
        Span<byte> info = stackalloc byte[ShaderBinaryInfo.ByteSize];
        if (!memory.TryRead(infoAddress, info))
        {
            return false;
        }

        declaredHash = new ShaderBinaryInfo(
            BinaryPrimitives.ReadUInt32LittleEndian(info[16..]),
            BinaryPrimitives.ReadUInt32LittleEndian(info[20..])).DeclaredHash;
        return true;
    }

    // The declared hash when it is not zero, else the content hash of every code range.
    public static ulong Compute(ICpuMemory memory, ulong codeAddress, ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges, string label)
    {
        if (!TryReadDeclaredHash(memory, codeAddress, out var declaredHash))
        {
            throw Scheduling.GpuFatal.Fatal($"The shader code is unreadable: label={label} shader=0x{codeAddress:X16}.");
        }

        if (declaredHash != 0)
        {
            return declaredHash;
        }

        return ComputeContent(memory, ranges, label);
    }

    // Runtime-linked fetch code is not covered by the entry shader's declared hash.
    public static ulong ComputeContent(ICpuMemory memory, ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges, string label)
    {

        // Shaders without a declared hash are rehashed on every draw, so the hasher and
        // the code buffer are reused per thread instead of allocated per call.
        var hash = _hasher ??= new XxHash3();
        hash.Reset();
        foreach (var (address, sizeBytes) in ranges)
        {
            var buffer = _codeBuffer;
            if (buffer is null || buffer.Length < sizeBytes)
            {
                buffer = _codeBuffer = new byte[Math.Max(sizeBytes, 4096u)];
            }

            var code = buffer.AsSpan(0, (int)sizeBytes);
            if (!memory.TryRead(address, code))
            {
                throw Scheduling.GpuFatal.Fatal($"The shader code is unreadable: label={label} shader=0x{address:X16} size=0x{sizeBytes:X8}.");
            }

            hash.Append(code);
        }

        return hash.GetCurrentHashAsUInt64();
    }

    // Compute with the content hashes of the shaders already hashed remembered, so a shader without a
    // declared hash is not read out and hashed again on every draw. The remembered hash stands only while
    // the guest still holds the bytes it was made from, which one in-place compare per code range settles;
    // memory that cannot compare in place, and any difference, take the full path as before.
    public static ulong Compute(ICpuMemory memory, ulong codeAddress, ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges, string label,
        ShaderIdentityMemo memo)
    {
        if (!TryReadDeclaredHash(memory, codeAddress, out var declaredHash))
        {
            throw Scheduling.GpuFatal.Fatal($"The shader code is unreadable: label={label} shader=0x{codeAddress:X16}.");
        }

        if (declaredHash != 0)
        {
            return declaredHash;
        }

        if (memo.TryGet(memory, codeAddress, ranges, out var remembered, out var compared))
        {
            return remembered;
        }

        // A memory that cannot compare in place would miss every draw and pay for a copy, a fresh hasher
        // and a map write each time; it takes the plain path, as before the memo.
        if (!compared || !memo.CanKeep(ranges))
        {
            return Compute(memory, codeAddress, ranges, label);
        }

        // The hash is made from the very bytes kept, so the two cannot disagree if the guest writes meanwhile.
        var code = new byte[ranges.Length][];
        var hash = new XxHash3();
        for (var index = 0; index < ranges.Length; index++)
        {
            var (address, sizeBytes) = ranges[index];
            code[index] = new byte[sizeBytes];
            if (!memory.TryRead(address, code[index]))
            {
                throw Scheduling.GpuFatal.Fatal($"The shader code is unreadable: label={label} shader=0x{address:X16} size=0x{sizeBytes:X8}.");
            }

            hash.Append(code[index]);
        }

        var contentHash = hash.GetCurrentHashAsUInt64();
        memo.Add(codeAddress, ranges, code, contentHash);
        return contentHash;
    }

    [ThreadStatic]
    private static XxHash3? _hasher;

    [ThreadStatic]
    private static byte[]? _codeBuffer;
}

// The content hashes of shaders without a declared hash, with the code bytes each was made from
// (ShaderIdentity.Compute). Entries are immutable and the map is concurrent: the render thread and a
// thread resolving draws ahead of it share one.
public sealed class ShaderIdentityMemo
{
    private const int MaxEntries = 4096;
    private const long MaxEntryBytes = 1 << 20;
    private const long MaxTotalBytes = 64 << 20;

    private sealed class Entry(ulong codeAddress, (ulong Address, uint SizeBytes)[] ranges, byte[][] code, ulong hash)
    {
        public readonly ulong CodeAddress = codeAddress;
        public readonly (ulong Address, uint SizeBytes)[] Ranges = ranges;
        public readonly byte[][] Code = code;
        public readonly ulong Hash = hash;
    }

    private readonly System.Collections.Concurrent.ConcurrentDictionary<ulong, Entry> _entries = new();
    private long _bytes;

    public int Count => _entries.Count;

    internal bool CanKeep(ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges)
    {
        var total = 0L;
        foreach (var (_, sizeBytes) in ranges)
        {
            total += sizeBytes;
        }

        return total <= MaxEntryBytes;
    }

    // compared is false when the remembered entry matches the ranges but the memory could not compare
    // them in place (no in-place compare, or unmapped); it is true otherwise.
    internal bool TryGet(ICpuMemory memory, ulong codeAddress, ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges, out ulong hash,
        out bool compared)
    {
        hash = 0;
        compared = true;
        if (!_entries.TryGetValue(codeAddress, out var entry) || entry.Ranges.Length != ranges.Length)
        {
            return false;
        }

        for (var index = 0; index < ranges.Length; index++)
        {
            if (entry.Ranges[index] != ranges[index])
            {
                return false;
            }
        }

        for (var index = 0; index < ranges.Length; index++)
        {
            if (!memory.TryCompare(ranges[index].Address, entry.Code[index], out var equal))
            {
                compared = false;
                return false;
            }

            if (!equal)
            {
                return false;
            }
        }

        hash = entry.Hash;
        return true;
    }

    internal void Add(ulong codeAddress, ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges, byte[][] code, ulong hash)
    {
        var size = 0L;
        foreach (var bytes in code)
        {
            size += bytes.Length;
        }

        // Bounded: dropping the entries only costs hashing the code again.
        if (_entries.Count >= MaxEntries || Interlocked.Add(ref _bytes, size) > MaxTotalBytes)
        {
            _entries.Clear();
            Interlocked.Exchange(ref _bytes, size);
        }

        _entries[codeAddress] = new Entry(codeAddress, ranges.ToArray(), code, hash);
    }
}
