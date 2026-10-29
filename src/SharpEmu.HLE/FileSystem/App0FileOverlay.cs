// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;

namespace SharpEmu.HLE.FileSystem;

/// <summary>
/// Read-time substitutions for the files of the title's /app0, keyed by the host path the guest's
/// open resolves to. An entry either replaces a file's whole content (any length, for a file
/// served from an archive-mounted app0; on disk only at the file's own length) or overwrites byte
/// ranges of it, which is how a value inside a large container (an ini inside a .pak) changes
/// without copying the container. Every read path that serves /app0 consults it: archive streams
/// (<see cref="ZArchiveFileStream"/>), host files opened for reading
/// (<see cref="App0OverlayFileStream"/>) and positional readers that call <see cref="Entry.Apply"/>.
/// Nothing is written to disk.
/// </summary>
public static class App0FileOverlay
{
    public readonly record struct Patch(long Offset, byte[] Bytes);

    public sealed class Entry
    {
        internal Entry(string reason, byte[]? content, Patch[] patches)
        {
            Reason = reason;
            Content = content;
            Patches = patches;
        }

        public string Reason { get; }

        /// <summary>The whole file, when the entry replaces it; null when it patches ranges.</summary>
        public byte[]? Content { get; }

        public IReadOnlyList<Patch> Patches { get; }

        /// <summary>Reads the replacement content at <paramref name="offset"/>.</summary>
        public int ReadContent(Span<byte> buffer, long offset)
        {
            var content = Content ?? throw new InvalidOperationException("The entry patches ranges.");
            if (offset >= content.Length)
            {
                return 0;
            }

            var count = (int)Math.Min(buffer.Length, content.Length - offset);
            content.AsSpan((int)offset, count).CopyTo(buffer);
            return count;
        }

        /// <summary>Overwrites the patched bytes that fall inside <paramref name="read"/> bytes
        /// read into <paramref name="buffer"/> from file offset <paramref name="offset"/>.</summary>
        public void Apply(Span<byte> buffer, long offset, int read)
        {
            if (read <= 0)
            {
                return;
            }

            var end = offset + read;
            foreach (var patch in Patches)
            {
                var patchEnd = patch.Offset + patch.Bytes.Length;
                if (patchEnd <= offset || patch.Offset >= end)
                {
                    continue;
                }

                var from = Math.Max(offset, patch.Offset);
                var to = Math.Min(end, patchEnd);
                patch.Bytes.AsSpan((int)(from - patch.Offset), (int)(to - from))
                    .CopyTo(buffer[(int)(from - offset)..]);
            }
        }
    }

    private static readonly ConcurrentDictionary<string, Entry> Entries = new(StringComparer.OrdinalIgnoreCase);

    public static bool IsEmpty => Entries.IsEmpty;

    /// <summary>Serves <paramref name="content"/> in place of the file at <paramref name="hostPath"/>.</summary>
    public static void Replace(string hostPath, byte[] content, string reason)
    {
        ArgumentNullException.ThrowIfNull(content);
        Entries[Normalize(hostPath)] = new Entry(reason, content, []);
    }

    /// <summary>Overwrites byte ranges of the file at <paramref name="hostPath"/> on every read.</summary>
    public static void AddPatches(string hostPath, IReadOnlyList<Patch> patches, string reason)
    {
        ArgumentNullException.ThrowIfNull(patches);
        if (patches.Count == 0)
        {
            return;
        }

        var key = Normalize(hostPath);
        Entries.AddOrUpdate(
            key,
            _ => new Entry(reason, null, [.. patches]),
            (_, existing) => existing.Content is null
                ? new Entry(existing.Reason + "; " + reason, null, [.. existing.Patches, .. patches])
                : throw new InvalidOperationException($"{hostPath} is already replaced whole."));
    }

    public static bool TryGet(string? hostPath, out Entry entry)
    {
        entry = null!;
        if (Entries.IsEmpty || string.IsNullOrEmpty(hostPath))
        {
            return false;
        }

        string key;
        try
        {
            key = Normalize(hostPath);
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }

        return Entries.TryGetValue(key, out entry!);
    }

    public static void Clear() => Entries.Clear();

    private static string Normalize(string hostPath) =>
        Path.TrimEndingDirectorySeparator(Path.GetFullPath(hostPath));
}

/// <summary>A host file opened for reading whose bytes come through an <see cref="App0FileOverlay"/>
/// entry. Still a <see cref="FileStream"/>, so descriptor code that asks for the handle (fstat)
/// keeps working; positional readers must use <see cref="ReadAt"/> rather than the raw handle.</summary>
public sealed class App0OverlayFileStream : FileStream
{
    private readonly App0FileOverlay.Entry _entry;

    public App0OverlayFileStream(string path, FileShare share, App0FileOverlay.Entry entry)
        : base(path, FileMode.Open, FileAccess.Read, share)
    {
        _entry = entry;
        if (entry.Content is { } content && content.Length != base.Length)
        {
            throw new IOException($"{path}: a replacement of a host file must keep its length.");
        }
    }

    public int ReadAt(Span<byte> buffer, long offset)
    {
        if (_entry.Content is not null)
        {
            return _entry.ReadContent(buffer, offset);
        }

        var read = RandomAccess.Read(SafeFileHandle, buffer, offset);
        _entry.Apply(buffer, offset, read);
        return read;
    }

    public override int Read(Span<byte> buffer)
    {
        var position = Position;
        var read = ReadAt(buffer, position);
        Position = position + read;
        return read;
    }

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override int ReadByte()
    {
        Span<byte> one = stackalloc byte[1];
        return Read(one) == 1 ? one[0] : -1;
    }

    public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(Read(buffer.Span));

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        Task.FromResult(Read(buffer.AsSpan(offset, count)));
}
