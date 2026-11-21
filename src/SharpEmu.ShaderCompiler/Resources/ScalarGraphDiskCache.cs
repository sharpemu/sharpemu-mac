// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SharpEmu.ShaderCompiler.Resources;

// The running game's folder for the scalar graph disk cache: the presenter names it beside the game's pipeline cache.
public static class ShaderAnalysisCache
{
    public static void UseGameFolder(string folder) => ScalarGraphDiskCache.UseGameFolder(folder);
}

internal static class ScalarGraphDiskCache
{
    // 5 on both parents, with different layouts (the fork's uncompressed entries, upstream's zlib-compressed ones);
    // 6: the merged zlib-compressed entries, so neither parent's entries are taken.
    private const int Version = 6;
    private const int MaxEntryBytes = 32 * 1024 * 1024;
    // Each game keeps its own cache, up to 8 GiB of compressed entries, trimmed by last use. One small folder shared by
    // every game and trimmed by write time cannot hold one game's graphs: a game's opening and half an hour of play can
    // build hundreds of megabytes of them, and play keeps adding graphs. A warm graph takes about a third off planning.
    private const long MaxCacheBytes = 8L * 1024 * 1024 * 1024;
    // A trim that passes the cap stops at this size, so the writes after it do not each list and trim the folder again.
    private const long TrimTargetBytes = MaxCacheBytes / 10 * 9;
    // SHARPEMU_SHADER_ANALYSIS_CACHE_PATH names the folder ("0" turns the cache off). Otherwise it is the running game's
    // folder (UseGameFolder), kept like its pipeline binaries and prewarm list, so no game trims another's graphs; until a
    // game names its folder the shared folder below is used.
    private static readonly string? CachePath =
        Environment.GetEnvironmentVariable("SHARPEMU_SHADER_ANALYSIS_CACHE_PATH") is { Length: > 0 } configured ? configured : null;
    private static volatile string? _gameFolder;

    internal static void UseGameFolder(string folder) => _gameFolder = folder;

    // A host that names no game folder (both hosts name one once their device is up) keeps the one folder every game
    // shared before the per-game folders, at the cap it had then.
    private static readonly string SharedCachePath =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpEmu", "shader-analysis");
    private const long SharedMaxCacheBytes = 256L * 1024 * 1024;
    private static readonly bool Trace = Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_SHADER_PLANNING") == "1";
    private static readonly object WriteLock = new();
    // Each folder's size as last listed plus what this process wrote since (under WriteLock).
    private static readonly Dictionary<string, long> CountedBytes = new(StringComparer.OrdinalIgnoreCase);

    [ThreadStatic]
    private static MemoryStream? _snapshotBuffer;

    // Pruning lists and stats every entry: the shared cache held 42,740 graphs (255 MB), which
    // took 160-360 ms to list under Rosetta. Listing for every written graph, under WriteLock,
    // on the GPU worker and the prewarm threads, charged that to every shader planned without
    // a cached graph, which is every shader after a compiler build. The size is listed once per
    // process and directory and then counted forward (CountedBytes), and a trim goes down below
    // the cap so the next one is many entries away. Other processes writing the
    // same directory are only seen at the next listing, so the cap can be overshot by what
    // they wrote in between.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> Listings = new(StringComparer.Ordinal);

    /// <summary>How many times this process has listed <paramref name="directory"/>.</summary>
    internal static int ListingsOf(string directory) => Listings.GetValueOrDefault(directory);

    internal static ScalarValueGraph Build(Gen5ShaderProgram program, uint userDataBase, uint userDataCount,
        IReadOnlySet<uint>? fixedFunctionVertexLoads, uint waveSize, string? cachePath = null, ulong excludedUserDataRegisters = 0, int dynamicUserData = -1)
    {
        var directory = cachePath ?? CachePath ?? _gameFolder ?? SharedCachePath;
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
                        using var reader = new BinaryReader(Inflate(bytes, length));
                        if (reader.ReadString() != key) throw new InvalidDataException("Graph cache key mismatch.");
                        var cached = ScalarValueGraph.ReadSnapshot(reader, program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
                        MarkUsed(path);
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
                using var compressed = new MemoryStream();
                using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
                {
                    zlib.Write(stream.GetBuffer(), 0, (int)stream.Length);
                }

                var bytes = compressed.GetBuffer().AsSpan(0, (int)compressed.Length);
                lock (WriteLock)
                {
                    Directory.CreateDirectory(directory);
                    // Keep a bounded cache across compiler versions too. No live
                    // analysis objects or guest memory are retained by the cache.
                    var entryBytes = compressed.Length + 32;
                    var capBytes = directory == SharedCachePath ? SharedMaxCacheBytes : MaxCacheBytes;
                    if (!CountedBytes.TryGetValue(directory, out var counted) || counted + entryBytes > capBytes)
                    {
                        counted = Trim(directory, entryBytes, capBytes, capBytes == MaxCacheBytes ? TrimTargetBytes : capBytes / 10 * 9);
                    }

                    CountedBytes[directory] = counted + entryBytes;
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

    // An entry's snapshot, inflated (at most MaxEntryBytes; the reader needs a seekable stream).
    private static MemoryStream Inflate(byte[] compressed, int length)
    {
        var inflated = new MemoryStream();
        using (var zlib = new ZLibStream(new MemoryStream(compressed, 0, length, false), CompressionMode.Decompress))
        {
            var buffer = new byte[81920];
            int read;
            while ((read = zlib.Read(buffer, 0, buffer.Length)) > 0)
            {
                if (inflated.Length + read > MaxEntryBytes) throw new InvalidDataException("Graph cache entry too large.");
                inflated.Write(buffer, 0, read);
            }
        }

        inflated.Position = 0;
        return inflated;
    }

    // A hit marks its entry used: Trim drops the entries used longest ago, not the ones written longest ago.
    private static void MarkUsed(string path)
    {
        try
        {
            File.SetLastAccessTimeUtc(path, DateTime.UtcNow);
        }
        catch (Exception error) when (IsCacheFailure(error)) { }
    }

    // Lists the folder; when an entry of entryBytes would take it past capBytes, deletes the entries used longest ago until
    // it fits within targetBytes. Returns the bytes that remain.
    internal static long Trim(string directory, long entryBytes, long capBytes, long targetBytes)
    {
        Listings.AddOrUpdate(directory, 1, static (_, count) => count + 1);
        var entries = new DirectoryInfo(directory).GetFiles("*.graph");
        var total = entries.Sum(static entry => entry.Length);
        if (total + entryBytes <= capBytes) return total;
        foreach (var entry in entries.OrderBy(LastUse))
        {
            if (total + entryBytes <= targetBytes) break;
            total -= entry.Length;
            entry.Delete();
        }

        return total;
    }

    private static DateTime LastUse(FileInfo entry) =>
        entry.LastAccessTimeUtc > entry.LastWriteTimeUtc ? entry.LastAccessTimeUtc : entry.LastWriteTimeUtc;

    private static bool IsCacheFailure(Exception error) => error is IOException or UnauthorizedAccessException or
        InvalidDataException or ArgumentException or OverflowException;
}
