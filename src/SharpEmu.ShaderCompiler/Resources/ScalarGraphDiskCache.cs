// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SharpEmu.ShaderCompiler.Resources;

internal static class ScalarGraphDiskCache
{
    private const int Version = 5;
    private const int MaxEntryBytes = 32 * 1024 * 1024;
    private const long MaxCacheBytes = 256L * 1024 * 1024;
    private static readonly string CachePath = Environment.GetEnvironmentVariable("SHARPEMU_SHADER_ANALYSIS_CACHE_PATH") ??
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpEmu", "shader-analysis");
    private static readonly bool Trace = Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_SHADER_PLANNING") == "1";
    private static readonly object WriteLock = new();

    [ThreadStatic]
    private static MemoryStream? _snapshotBuffer;

    // Pruning lists and stats every entry: the shared cache held 42,740 graphs (255 MB), which
    // took 160-360 ms to list under Rosetta. Listing for every written graph, under WriteLock,
    // on the GPU worker and the prewarm threads, charged that to every shader planned without
    // a cached graph, which is every shader after a compiler build. The size is now listed once per
    // process and directory and then counted forward, and a prune goes down to three
    // quarters of the cap so the next one is many entries away. Other processes writing the
    // same directory are only seen at the next listing, so the cap can be overshot by what
    // they wrote in between.
    private const long PruneTargetBytes = MaxCacheBytes / 4 * 3;
    private static readonly Dictionary<string, long> KnownBytes = new(StringComparer.Ordinal);
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> Listings = new(StringComparer.Ordinal);

    /// <summary>How many times this process has listed <paramref name="directory"/>.</summary>
    internal static int ListingsOf(string directory) => Listings.GetValueOrDefault(directory);

    internal static ScalarValueGraph Build(Gen5ShaderProgram program, uint userDataBase, uint userDataCount,
        IReadOnlySet<uint>? fixedFunctionVertexLoads, uint waveSize, string? cachePath = null, ulong excludedUserDataRegisters = 0, int dynamicUserData = -1)
    {
        var directory = cachePath ?? CachePath;
        if (directory == "0") return ScalarValueGraph.Build(program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
        var key = Key(program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
        var path = Path.Combine(directory, key + ".graph");
        try
        {
            using var file = File.OpenRead(path);
            if (file.Length is > 32 and <= MaxEntryBytes)
            {
                Span<byte> digest = stackalloc byte[32];
                file.ReadExactly(digest);
                var length = checked((int)file.Length - 32);

                // Entries run to hundreds of kilobytes; a pooled buffer keeps each read off the large object heap.
                var bytes = System.Buffers.ArrayPool<byte>.Shared.Rent(length);
                try
                {
                    file.ReadExactly(bytes.AsSpan(0, length));
                    if (CryptographicOperations.FixedTimeEquals(digest, SHA256.HashData(bytes.AsSpan(0, length))))
                    {
                        using var reader = new BinaryReader(new MemoryStream(bytes, 0, length, false));
                        if (reader.ReadString() != key) throw new InvalidDataException("Graph cache key mismatch.");
                        var cached = ScalarValueGraph.ReadSnapshot(reader, program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
                        if (Trace) Console.Error.WriteLine($"[SHADER_ANALYSIS_CACHE] hit key={key} bytes={length}");
                        return cached;
                    }
                }
                finally
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(bytes);
                }
            }
        }
        catch (Exception error) when (IsCacheFailure(error)) { }

        var graph = ScalarValueGraph.Build(program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
        try
        {
            // One buffer per thread, kept between shaders, instead of one grown by doubling for each.
            var stream = _snapshotBuffer ??= new MemoryStream();
            stream.SetLength(0);
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(key);
                graph.WriteSnapshot(writer);
            }
            if (stream.Length <= MaxEntryBytes - 32)
            {
                var bytes = stream.GetBuffer().AsSpan(0, (int)stream.Length);
                lock (WriteLock)
                {
                    Directory.CreateDirectory(directory);
                    // Keep a bounded cache across compiler versions too. No live
                    // analysis objects or guest memory are retained by the cache.
                    var entryBytes = stream.Length + 32;
                    if (!KnownBytes.TryGetValue(directory, out var known) || known + entryBytes > MaxCacheBytes)
                    {
                        known = Prune(directory, MaxCacheBytes - entryBytes, PruneTargetBytes - entryBytes);
                    }
                    KnownBytes[directory] = known + entryBytes;
                    var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
                    try
                    {
                        using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                        {
                            file.Write(SHA256.HashData(bytes));
                            file.Write(bytes);
                        }
                        File.Move(temporary, path, true);
                    }
                    finally { File.Delete(temporary); }
                }
                if (Trace) Console.Error.WriteLine($"[SHADER_ANALYSIS_CACHE] miss key={key} bytes={bytes.Length}");
            }
            if (stream.Capacity > 4 * 1024 * 1024) _snapshotBuffer = null;
        }
        catch (Exception error) when (IsCacheFailure(error)) { }
        return graph;
    }

    internal static string Key(Gen5ShaderProgram program, uint userDataBase, uint userDataCount,
        IReadOnlySet<uint>? fixedFunctionVertexLoads, uint waveSize, ulong excludedUserDataRegisters = 0, int dynamicUserData = -1)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var stream = new HashingStream(hash);
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
        {
            writer.Write(Version);
            // The analysis code, not the build: see AssemblyCodeIdentity.
            writer.Write(AssemblyCodeIdentity.Of(typeof(ScalarValueGraph).Assembly));
            writer.Write(userDataBase); writer.Write(userDataCount); writer.Write(waveSize);
            writer.Write(excludedUserDataRegisters);
            if (dynamicUserData >= 0) writer.Write(dynamicUserData);
            var loads = fixedFunctionVertexLoads?.Order().ToArray() ?? [];
            writer.Write(loads.Length);
            foreach (var load in loads) writer.Write(load);
            writer.Write(program.Instructions.Count);
            foreach (var instruction in program.Instructions)
            {
                writer.Write(instruction.Pc); writer.Write(instruction.ProgramOffset); writer.Write((int)instruction.Encoding); writer.Write(instruction.Opcode);
                writer.Write(instruction.Words.Count);
                foreach (var word in instruction.Words) writer.Write(word);
                writer.Write(instruction.Sources.Count);
                foreach (var operand in instruction.Sources) { writer.Write((int)operand.Kind); writer.Write(operand.Value); }
                writer.Write(instruction.Destinations.Count);
                foreach (var operand in instruction.Destinations) { writer.Write((int)operand.Kind); writer.Write(operand.Value); }
                // Controls are a closed set of decoded record types. Include the
                // runtime type and every property, including variable NSA arrays.
                writer.Write(instruction.Control?.GetType().FullName ?? "");
                writer.Write(instruction.Control is { } control ? JsonSerializer.Serialize(control, control.GetType()) : "");
            }
        }
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    /// <summary>A write-only stream that feeds what it is given to a hash instead of keeping it.</summary>
    private sealed class HashingStream(IncrementalHash hash) : Stream
    {
        public override bool CanRead => false;

        public override bool CanSeek => false;

        public override bool CanWrite => true;

        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => hash.AppendData(buffer, offset, count);

        public override void Write(ReadOnlySpan<byte> buffer) => hash.AppendData(buffer);
    }

    /// <summary>
    /// Lists the directory's graphs and, when they hold more than <paramref name="limit"/>
    /// bytes, deletes the oldest until at most <paramref name="target"/> bytes are left.
    /// Returns the bytes left.
    /// </summary>
    internal static long Prune(string directory, long limit, long target)
    {
        Listings.AddOrUpdate(directory, 1, static (_, count) => count + 1);
        var entries = new DirectoryInfo(directory).GetFiles("*.graph").OrderBy(f => f.LastWriteTimeUtc).ToArray();
        var total = entries.Sum(f => f.Length);
        if (total <= limit) return total;
        foreach (var entry in entries)
        {
            if (total <= target) break;
            total -= entry.Length;
            entry.Delete();
        }
        return total;
    }

    private static bool IsCacheFailure(Exception error) => error is IOException or UnauthorizedAccessException or
        InvalidDataException or ArgumentException or OverflowException;
}
