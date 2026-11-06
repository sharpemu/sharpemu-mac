// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Hashing;
using SharpEmu.HLE;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Splits an indexed triangle list into meshlets the way the geometry engine packs NGG
/// subgroups: each holds a bounded number of unique vertices and triangles, so a mesh
/// threadgroup runs one without reading the index stream.
///
/// A built buffer is [meshlet count x uint4 {vertexOffset, primOffset, firstTriangle,
/// vertexCount | primCount &lt;&lt; 16}] [vertex list: the raw index values] [prim list:
/// one dword per triangle, local i0 | i1 &lt;&lt; 8 | i2 &lt;&lt; 16]. The offsets are
/// relative to the start of their list; the draw-info block carries the lists' bases.
/// </summary>
public static class MeshletBuilder
{
    /// <summary>Local indices are 8-bit, so a meshlet holds at most this many vertices.</summary>
    public const uint MaxLocalVertices = 255;

    public readonly record struct Result(uint[] Words, uint MeshletCount, uint VertexBase, uint PrimitiveBase);

    /// <summary>
    /// The lists one build fills, kept by a caller that builds repeatedly so a rebuild
    /// allocates nothing: an index range can be hundreds of thousands of indices, and
    /// fresh lists of that size landed on the large object heap on every rebuild.
    /// </summary>
    public sealed class Scratch
    {
        internal readonly List<uint> Headers = [];
        internal readonly List<uint> VertexList = [];
        internal readonly List<uint> PrimitiveList = [];
        internal readonly uint[] Current = new uint[MaxLocalVertices];

        // The current meshlet's vertices by index value: open addressing over a table four times the
        // most a meshlet holds, its entries current only under this meshlet's generation (a flush
        // starts the next one instead of clearing the table).
        internal const int TableBits = 10;
        internal readonly uint[] TableKeys = new uint[1 << TableBits];
        internal readonly uint[] TableLocals = new uint[1 << TableBits];
        internal readonly uint[] TableGenerations = new uint[1 << TableBits];
        internal uint Generation;

        public uint MeshletCount => (uint)(Headers.Count / 4);

        public uint VertexBase => MeshletCount * 4;

        public uint PrimitiveBase => VertexBase + (uint)VertexList.Count;

        /// <summary>The number of words <see cref="CopyTo"/> writes.</summary>
        public int WordCount => Headers.Count + VertexList.Count + PrimitiveList.Count;

        /// <summary>Writes the built buffer: headers, then the vertex list, then the prim list.</summary>
        public void CopyTo(Span<uint> words)
        {
            System.Runtime.InteropServices.CollectionsMarshal.AsSpan(Headers).CopyTo(words);
            System.Runtime.InteropServices.CollectionsMarshal.AsSpan(VertexList).CopyTo(words[(int)VertexBase..]);
            System.Runtime.InteropServices.CollectionsMarshal.AsSpan(PrimitiveList).CopyTo(words[(int)PrimitiveBase..]);
        }

        internal void Reset(int indexCount, int triangles)
        {
            Headers.Clear();
            VertexList.Clear();
            PrimitiveList.Clear();
            VertexList.EnsureCapacity(indexCount);
            PrimitiveList.EnsureCapacity(triangles);
        }
    }

    public static Result Build(ReadOnlySpan<uint> indices, uint maxVertices, uint maxPrimitives)
    {
        var scratch = new Scratch();
        BuildInto(indices, maxVertices, maxPrimitives, scratch);
        var words = new uint[scratch.WordCount];
        scratch.CopyTo(words);
        return new Result(words, scratch.MeshletCount, scratch.VertexBase, scratch.PrimitiveBase);
    }

    /// <summary>Builds the meshlets into <paramref name="scratch"/>, replacing what it held.</summary>
    public static void BuildInto(ReadOnlySpan<uint> indices, uint maxVertices, uint maxPrimitives, Scratch scratch)
    {
        maxVertices = Math.Clamp(maxVertices, 3, MaxLocalVertices);
        maxPrimitives = Math.Clamp(maxPrimitives, 1, MaxLocalVertices);
        var triangles = indices.Length / 3;
        scratch.Reset(indices.Length, triangles);
        var headers = scratch.Headers;
        var vertexList = scratch.VertexList;
        var primitiveList = scratch.PrimitiveList;

        var current = scratch.Current;
        var keys = scratch.TableKeys;
        var locals = scratch.TableLocals;
        var generations = scratch.TableGenerations;
        const int tableMask = (1 << Scratch.TableBits) - 1;
        var currentVertices = 0u;
        var currentPrimitives = 0u;
        var meshletVertexStart = 0u;
        var meshletPrimitiveStart = 0u;
        var meshletFirstTriangle = 0u;
        NextGeneration(scratch);
        var generation = scratch.Generation;

        void Flush()
        {
            if (currentPrimitives == 0)
            {
                return;
            }

            headers.Add(meshletVertexStart);
            headers.Add(meshletPrimitiveStart);
            headers.Add(meshletFirstTriangle);
            headers.Add(currentVertices | (currentPrimitives << 16));
            for (var i = 0u; i < currentVertices; i++)
            {
                vertexList.Add(current[i]);
            }

            meshletVertexStart += currentVertices;
            meshletPrimitiveStart += currentPrimitives;
            currentVertices = 0;
            currentPrimitives = 0;
            NextGeneration(scratch);
            generation = scratch.Generation;
        }

        // The local index of a vertex value in the current meshlet, or uint.MaxValue.
        uint Find(uint value)
        {
            var slot = (int)((value * 0x9E3779B1u) >> (32 - Scratch.TableBits));
            while (generations[slot] == generation)
            {
                if (keys[slot] == value)
                {
                    return locals[slot];
                }

                slot = (slot + 1) & tableMask;
            }

            return uint.MaxValue;
        }

        uint Add(uint value)
        {
            var slot = (int)((value * 0x9E3779B1u) >> (32 - Scratch.TableBits));
            while (generations[slot] == generation)
            {
                slot = (slot + 1) & tableMask;
            }

            generations[slot] = generation;
            keys[slot] = value;
            locals[slot] = currentVertices;
            current[currentVertices] = value;
            return currentVertices++;
        }

        for (var triangle = 0; triangle < triangles; triangle++)
        {
            var a = indices[triangle * 3];
            var b = indices[(triangle * 3) + 1];
            var c = indices[(triangle * 3) + 2];

            // New unique vertices this triangle would add to the current meshlet.
            var foundA = Find(a);
            var foundB = b == a ? foundA : Find(b);
            var foundC = c == a ? foundA : c == b ? foundB : Find(c);
            var added = (foundA == uint.MaxValue ? 1u : 0u) + (b != a && foundB == uint.MaxValue ? 1u : 0u) +
                (c != a && c != b && foundC == uint.MaxValue ? 1u : 0u);

            if (currentPrimitives == maxPrimitives || currentVertices + added > maxVertices)
            {
                Flush();
                foundA = foundB = foundC = uint.MaxValue;
            }

            if (currentPrimitives == 0)
            {
                meshletFirstTriangle = (uint)triangle;
            }

            // In order a, b, c, as each would be added: a repeated value takes the slot its first use got.
            var la = foundA != uint.MaxValue ? foundA : Add(a);
            var lb = b == a ? la : foundB != uint.MaxValue ? foundB : Add(b);
            var lc = c == a ? la : c == b ? lb : foundC != uint.MaxValue ? foundC : Add(c);
            primitiveList.Add(la | (lb << 8) | (lc << 16));
            currentPrimitives++;
        }

        Flush();
    }

    // Starts a new generation of the vertex table, clearing it only when the counter wraps.
    private static void NextGeneration(Scratch scratch)
    {
        if (++scratch.Generation == 0)
        {
            Array.Clear(scratch.TableGenerations);
            scratch.Generation = 1;
        }
    }
}

/// <summary>
/// Meshlets of the indexed triangle-list draws a non-GS mesh pipeline runs, built once per
/// index range and validated against the guest bytes on every use: an index buffer the guest
/// rewrites in place hashes differently and is rebuilt. Entries live in shared Metal buffers
/// the residency set names, under a byte budget with least-recently-used eviction of entries
/// no in-flight submission can still read.
/// </summary>
internal sealed class MetalMeshletCache
{
    private const ulong BudgetBytes = 128UL * 1024 * 1024;
    private const int HashChunkBytes = 64 * 1024;

    public enum Fallback
    {
        Disabled,
        NotTriangleList,
        NoGuestAddress,
        GpuWrites,
        Unreadable,
        OverBudget,
    }

    public readonly record struct Meshlets(nint Buffer, uint Count, uint VertexBase, uint PrimitiveBase);

    private readonly record struct Key(
        ulong Address, uint IndexCount, bool Index32, uint MaxVertices, uint MaxPrimitives);

    private sealed class Entry
    {
        public required nint Buffer;
        public required nint Contents;
        public required ulong Bytes;
        public required ulong Hash;
        public required uint Count;
        public required uint VertexBase;
        public required uint PrimitiveBase;
        public ulong ValidatedSerial;
        public ulong LastUsedSerial;
        public ulong LastUsedTick;
    }

    private static readonly bool Disabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_MESHLETS") == "0";

    private readonly MetalGuestMemory _guestMemory;
    private readonly ICpuMemory _memory;
    private readonly Dictionary<Key, Entry> _entries = [];
    private readonly byte[] _chunk = new byte[HashChunkBytes];
    private readonly MeshletBuilder.Scratch _scratch = new();
    private readonly System.IO.Hashing.XxHash3 _hasher = new();
    private uint[] _indices = [];
    private readonly List<(nint Buffer, ulong Serial)> _retired = [];
    private ulong _bytes;
    private ulong _tick;

    private long _draws;
    private readonly long[] _fallbacks = new long[Enum.GetValues<Fallback>().Length];
    private long _buildTicks;
    private long _hashTicks;
    private long _builds;
    private long _validationsSkipped;

    public MetalMeshletCache(MetalGuestMemory guestMemory, ICpuMemory memory)
    {
        _guestMemory = guestMemory;
        _memory = memory;
    }

    public static bool IsDisabled => Disabled;

    public void NoteFallback(Fallback reason) => _fallbacks[(int)reason]++;

    /// <summary>Returns the meshlets of the index range, building them when the range is new
    /// or its bytes changed. The caller has already established that the draw is eligible
    /// (indexed triangle list, a guest-backed index buffer no GPU work is writing).</summary>
    public bool TryGet(
        ulong rangeAddress,
        uint indexCount,
        bool index32,
        uint maxVertices,
        uint maxPrimitives,
        ulong submissionSerial,
        ulong completedSerial,
        out Meshlets meshlets)
    {
        meshlets = default;
        ReleaseRetired(completedSerial);
        var elementBytes = index32 ? 4UL : 2UL;
        var rangeBytes = indexCount * elementBytes;
        var key = new Key(rangeAddress, indexCount, index32, maxVertices, maxPrimitives);

        // An entry already validated under this submission is not hashed again: the guest
        // does not rewrite an index range between the draws one command buffer records, and
        // hashing the bytes on every use was the cost that mattered (tens of milliseconds a
        // frame on the worker). Ranges the GPU writes never get here.
        if (_entries.TryGetValue(key, out var validated) && validated.ValidatedSerial == submissionSerial)
        {
            Touch(validated, submissionSerial);
            meshlets = new Meshlets(validated.Buffer, validated.Count, validated.VertexBase, validated.PrimitiveBase);
            _draws++;
            _validationsSkipped++;
            return true;
        }

        var hashStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        if (!TryHash(rangeAddress, rangeBytes, out var hash))
        {
            NoteFallback(Fallback.Unreadable);
            return false;
        }

        _hashTicks += System.Diagnostics.Stopwatch.GetTimestamp() - hashStarted;

        if (_entries.TryGetValue(key, out var entry))
        {
            if (entry.Hash == hash)
            {
                entry.ValidatedSerial = submissionSerial;
                Touch(entry, submissionSerial);
                meshlets = new Meshlets(entry.Buffer, entry.Count, entry.VertexBase, entry.PrimitiveBase);
                _draws++;
                return true;
            }

            Remove(key, entry);
        }

        var buildStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        // The index words and the build lists are scratch reused across rebuilds: a dynamic
        // index range is rebuilt every frame, and arrays of its size went to the LOH each time.
        if (_indices.Length < indexCount)
        {
            _indices = new uint[Math.Max(indexCount, (uint)Math.Min((ulong)_indices.Length * 2, int.MaxValue / sizeof(uint)))];
        }

        var indices = _indices.AsSpan(0, (int)indexCount);
        if (index32)
        {
            if (!_memory.TryRead(rangeAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes(indices)))
            {
                NoteFallback(Fallback.Unreadable);
                return false;
            }
        }
        else
        {
            // The 16-bit indices are read into the back half of the words and widened in
            // place front to back: word i is written after halfword i, which it never overtakes.
            var halves = System.Runtime.InteropServices.MemoryMarshal.Cast<uint, ushort>(indices)[(int)indexCount..];
            if (!_memory.TryRead(rangeAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes(halves)))
            {
                NoteFallback(Fallback.Unreadable);
                return false;
            }

            for (var i = 0; i < indices.Length; i++)
            {
                indices[i] = halves[i];
            }
        }

        var built = _scratch;
        MeshletBuilder.BuildInto(indices, maxVertices, maxPrimitives, built);
        var wordCount = built.WordCount;
        var size = (ulong)wordCount * sizeof(uint);
        if (!MakeRoom(size, completedSerial))
        {
            ReleaseRetired(completedSerial);
            NoteFallback(Fallback.OverBudget);
            return false;
        }

        var buffer = _guestMemory.CreateMeshletBuffer(size, out var contents);
        unsafe
        {
            built.CopyTo(new Span<uint>((void*)contents, wordCount));
        }

        entry = new Entry
        {
            Buffer = buffer,
            Contents = contents,
            Bytes = size,
            Hash = hash,
            Count = built.MeshletCount,
            VertexBase = built.VertexBase,
            PrimitiveBase = built.PrimitiveBase,
            ValidatedSerial = submissionSerial,
        };
        _entries[key] = entry;
        _bytes += size;
        Touch(entry, submissionSerial);
        _buildTicks += System.Diagnostics.Stopwatch.GetTimestamp() - buildStarted;
        _builds++;
        _draws++;
        meshlets = new Meshlets(buffer, entry.Count, entry.VertexBase, entry.PrimitiveBase);
        return true;
    }

    private void Touch(Entry entry, ulong serial)
    {
        entry.LastUsedSerial = serial;
        entry.LastUsedTick = ++_tick;
    }

    private bool TryHash(ulong address, ulong bytes, out ulong hash)
    {
        var hasher = _hasher;
        hasher.Reset();
        for (ulong offset = 0; offset < bytes;)
        {
            var count = (int)Math.Min((ulong)_chunk.Length, bytes - offset);
            var span = _chunk.AsSpan(0, count);
            if (!_memory.TryRead(address + offset, span))
            {
                hash = 0;
                return false;
            }

            hasher.Append(span);
            offset += (ulong)count;
        }

        hash = hasher.GetCurrentHashAsUInt64();
        return true;
    }

    // Evicts least-recently-used entries no in-flight submission can still read until the
    // new one fits. False when everything left is still in use.
    private bool MakeRoom(ulong size, ulong completedSerial)
    {
        if (size > BudgetBytes)
        {
            return false;
        }

        while (_bytes + size > BudgetBytes)
        {
            Key? victimKey = null;
            Entry? victim = null;
            foreach (var (key, entry) in _entries)
            {
                if (entry.LastUsedSerial <= completedSerial && (victim is null || entry.LastUsedTick < victim.LastUsedTick))
                {
                    victimKey = key;
                    victim = entry;
                }
            }

            if (victim is null)
            {
                return false;
            }

            Remove(victimKey!.Value, victim);
        }

        return true;
    }

    // The buffer may still be bound by recorded work, so it is released only once the
    // serial it was last used under has completed.
    private void Remove(Key key, Entry entry)
    {
        _entries.Remove(key);
        _bytes -= entry.Bytes;
        _retired.Add((entry.Buffer, entry.LastUsedSerial));
    }

    private void ReleaseRetired(ulong completedSerial)
    {
        for (var i = _retired.Count - 1; i >= 0; i--)
        {
            if (_retired[i].Serial <= completedSerial)
            {
                _guestMemory.ReleaseMeshletBuffer(_retired[i].Buffer);
                _retired.RemoveAt(i);
            }
        }
    }

    /// <summary>The window's counters as the PERF line's text, then reset.</summary>
    public string TakeCounters()
    {
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        var frequency = (double)System.Diagnostics.Stopwatch.Frequency;
        var text =
            $"meshlet_draws={_draws} meshlet_validations_skipped={_validationsSkipped} meshlet_builds={_builds} meshlet_mb={_bytes / (1024 * 1024)} " +
            $"meshlet_build_ms={(_buildTicks * 1000.0 / frequency).ToString("F1", culture)} " +
            $"meshlet_hash_ms={(_hashTicks * 1000.0 / frequency).ToString("F1", culture)} " +
            $"meshlet_fallbacks={string.Join('/', _fallbacks)}";
        _draws = 0;
        _builds = 0;
        _validationsSkipped = 0;
        _buildTicks = 0;
        _hashTicks = 0;
        Array.Clear(_fallbacks);
        return text;
    }
}
