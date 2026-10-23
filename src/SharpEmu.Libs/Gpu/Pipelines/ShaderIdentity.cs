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

    public static ulong ComputeContent(ICpuMemory memory, ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges, string label, Func<string, Exception>? invalid = null)
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
                throw (invalid ?? Scheduling.GpuFatal.Fatal)($"The shader code is unreadable: label={label} shader=0x{address:X16} size=0x{sizeBytes:X8}.");
            }

            hash.Append(code);
        }

        return hash.GetCurrentHashAsUInt64();
    }

    [ThreadStatic]
    private static XxHash3? _hasher;

    [ThreadStatic]
    private static byte[]? _codeBuffer;
}

// The ready identity is immutable. Tokens describe the exact observation that produced
// it; a new observation is never substituted into an old identity after a write.
public sealed record ReadyShaderIdentity(ulong Hash, ulong ContentHash);

public sealed class ShaderIdentityCache(ICpuMemory memory, Action<ulong, ulong> synchronize,
    SharpEmu.HLE.GpuMemory.GuestWriteGenerations? generations, Func<string, Exception>? invalid = null)
{
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_PROGRAM_INTERN") != "0";
    private sealed record Entry(ReadyShaderIdentity Identity, SharpEmu.HLE.GpuMemory.GuestWriteGenerations.Token[] Tokens);
    private readonly Dictionary<RegisteredShader, Entry> _entries = new(ReferenceEqualityComparer.Instance);
    private readonly object _gate = new();

    public ReadyShaderIdentity Get(RegisteredShader shader, string label)
    {
        // Identity capture contains no compilation. A short capture gate also single-flights
        // concurrent readers; program compilation remains in the existing compile futures.
        lock (_gate)
        {
            if (Enabled && generations is not null && _entries.TryGetValue(shader, out var entry) &&
                entry.Tokens.Length != 0 && Array.TrueForAll(entry.Tokens, static token => token.IsCurrent))
            {
                ProgramInternCounters.CodeHit();
                return entry.Identity;
            }
            ProgramInternCounters.CodeMiss();
            Span<(ulong Address, uint SizeBytes)> ranges = stackalloc (ulong, uint)[RegisteredShader.MaxCodeRanges];
            ranges = ranges[..shader.CopyCodeRanges(ranges)];
            foreach (var (address, size) in ranges) synchronize(address, size);
            var tokens = Enabled && generations is not null
                ? ranges.ToArray().Select(range => generations.Register(range.Address, range.SizeBytes)).ToArray()
                : [];
            var content = ShaderIdentity.ComputeContent(memory, ranges, label, invalid);
            if (!ShaderIdentity.TryReadDeclaredHash(memory, shader.CodeAddress, out var declared))
                throw (invalid ?? Scheduling.GpuFatal.Fatal)($"The shader code is unreadable: label={label} shader=0x{shader.CodeAddress:X16}.");
            var hash = declared == 0 ? content : declared;
            var identity = _entries.TryGetValue(shader, out var previous) && previous.Identity.Hash == hash && previous.Identity.ContentHash == content
                ? previous.Identity : new ReadyShaderIdentity(hash, content);
            if ((tokens.Length != 0 || previous is null || !ReferenceEquals(identity, previous.Identity)) &&
                Array.TrueForAll(tokens, static token => token.IsCurrent))
            {
                if (_entries.Count >= 4096) _entries.Clear();
                _entries[shader] = new Entry(identity, tokens);
            }
            return identity;
        }
    }
}

// Interval counts are drained alongside materialization counters in Metal's PERF line.
public static class ProgramInternCounters
{
    private static long _codeHits, _codeMisses, _headerHits, _headerMisses;
    internal static void CodeHit() => Interlocked.Increment(ref _codeHits);
    internal static void CodeMiss() => Interlocked.Increment(ref _codeMisses);
    internal static void HeaderHit() => Interlocked.Increment(ref _headerHits);
    internal static void HeaderMiss() => Interlocked.Increment(ref _headerMisses);
    public static string TakeReport() =>
        $"program_intern_hit={Interlocked.Exchange(ref _codeHits, 0)} program_intern_miss={Interlocked.Exchange(ref _codeMisses, 0)} " +
        $"header_intern_hit={Interlocked.Exchange(ref _headerHits, 0)} header_intern_miss={Interlocked.Exchange(ref _headerMisses, 0)}";
}
