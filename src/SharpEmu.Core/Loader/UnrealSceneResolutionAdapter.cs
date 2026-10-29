// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using SharpEmu.HLE.FileSystem;

namespace SharpEmu.Core.Loader;

/// <summary>
/// The scene size of an Unreal title on PS5 is <c>r.setres</c> (1920x1080 in the engine's
/// Engine/Platforms/PS5/Config/PS5Engine.ini on every title seen), independent of the back
/// buffer <see cref="UnrealBackBufferPatch"/> sizes. The ini lives in the title's .pak. When the
/// pak index is readable (not encrypted) and the ini entry stored plainly, the value is patched
/// in place as the title reads the pak, kept at its length (the width is zero-padded, which the
/// engine's parser accepts). A pak with an encrypted index cannot be searched, so the size goes
/// on the engine command line instead (<c>-ResX= -ResY=</c> appended to ue4commandline.txt or
/// uecommandline.txt in /app0).
/// </summary>
internal static class UnrealSceneResolutionAdapter
{
    internal const uint PakMagic = 0x5A6F12E1;
    private const string SetResKey = "r.setres";
    private static readonly string[] CommandLineFiles = ["ue4commandline.txt", "uecommandline.txt"];

    internal readonly record struct IniEntry(string Path, long DataOffset, long Size);

    internal enum PakIndex
    {
        NotAPak,
        Encrypted,
        Unsupported,
        Readable,
    }

    internal static void Apply(IReadOnlyList<EngineResolution.Segment> segments, string? app0Root, uint width, uint height)
    {
        if (string.IsNullOrEmpty(app0Root) || !UnrealBackBufferPatch.HasUnrealSignature(segments))
        {
            return;
        }

        var patched = new List<string>();
        var already = 0;
        var encrypted = 0;
        foreach (var pakPath in App0Archive.EnumerateFilesRecursive(app0Root, "*.pak"))
        {
            using var stream = App0Archive.OpenReadOrFile(pakPath);
            var kind = ReadIniEntries(stream, out var inis);
            if (kind == PakIndex.Encrypted)
            {
                encrypted++;
                continue;
            }

            if (kind != PakIndex.Readable)
            {
                continue;
            }

            var patches = new List<App0FileOverlay.Patch>();
            foreach (var ini in inis)
            {
                if (ini.Size <= 0 || ini.Size > 1 << 20)
                {
                    continue;
                }

                var content = new byte[ini.Size];
                stream.Position = ini.DataOffset;
                stream.ReadExactly(content);
                foreach (var (offset, length, oldWidth, oldHeight) in FindSetRes(content))
                {
                    if (oldWidth == width && oldHeight == height)
                    {
                        already++;
                        continue;
                    }

                    if (EncodeSetRes(width, height, length) is not { } value)
                    {
                        EngineResolution.Warn("UE", $"{ini.Path}: r.setres={oldWidth}x{oldHeight} has no room for {width}x{height}.");
                        continue;
                    }

                    patches.Add(new App0FileOverlay.Patch(ini.DataOffset + offset, value));
                    patched.Add($"{ini.Path} {oldWidth}x{oldHeight}");
                }
            }

            App0FileOverlay.AddPatches(pakPath, patches, "Unreal r.setres");
        }

        if (patched.Count > 0)
        {
            EngineResolution.Report("UE", "r.setres (pak ini)", width, height, string.Join(", ", patched));
            return;
        }

        if (already > 0)
        {
            EngineResolution.Report("UE", "r.setres (pak ini)", width, height, "the title's own size");
            return;
        }

        if (encrypted > 0)
        {
            ApplyCommandLine(app0Root, width, height);
        }
    }

    /// <summary>Appends <c>-ResX=W -ResY=H</c> to the engine's command-line file.</summary>
    private static void ApplyCommandLine(string app0Root, uint width, uint height)
    {
        foreach (var name in CommandLineFiles)
        {
            var path = Path.Combine(app0Root, name);
            if (EngineResolution.TryReadApp0File(path, 64 * 1024) is not { } original)
            {
                continue;
            }

            if (AppendResolution(original, width, height) is not { } rewritten)
            {
                EngineResolution.Report("UE", $"{name} -ResX/-ResY", width, height, "already on the command line");
                return;
            }

            if (!EngineResolution.TryReplaceApp0File(path, rewritten, "Unreal -ResX/-ResY"))
            {
                EngineResolution.Warn("UE", $"{name} on disk cannot change length; the scene size stays the title's own.");
                return;
            }

            EngineResolution.Report("UE", $"{name} -ResX/-ResY", width, height, "the pak index is encrypted");
            return;
        }

        EngineResolution.Warn("UE", "the pak index is encrypted and /app0 has no command-line file; the scene size stays the title's own.");
    }

    /// <summary>The command line with <c>-ResX=</c>/<c>-ResY=</c> appended; null when it already sets them.</summary>
    internal static byte[]? AppendResolution(byte[] original, uint width, uint height)
    {
        var text = Encoding.UTF8.GetString(original);
        if (text.Contains("-ResX=", StringComparison.OrdinalIgnoreCase) ||
            text.Contains("-ResY=", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        var body = text.TrimEnd('\r', '\n', ' ', '\t', '\0');
        return Encoding.UTF8.GetBytes(
            $"{body} -ResX={width.ToString(CultureInfo.InvariantCulture)} -ResY={height.ToString(CultureInfo.InvariantCulture)}");
    }

    /// <summary>Every <c>r.setres=WxH</c> line: the value's offset and length in
    /// <paramref name="ini"/> and the size it names.</summary>
    internal static List<(int Offset, int Length, uint Width, uint Height)> FindSetRes(ReadOnlySpan<byte> ini)
    {
        var found = new List<(int, int, uint, uint)>();
        var lineStart = 0;
        while (lineStart < ini.Length)
        {
            var newline = ini[lineStart..].IndexOf((byte)'\n');
            var lineEnd = newline < 0 ? ini.Length : lineStart + newline;
            var line = ini[lineStart..lineEnd];
            var indent = 0;
            while (indent < line.Length && line[indent] is (byte)' ' or (byte)'\t')
            {
                indent++;
            }

            var keyLength = SetResKey.Length + 1;
            if (line.Length - indent > keyLength &&
                Ascii.EqualsIgnoreCase(line.Slice(indent, SetResKey.Length), SetResKey) &&
                line[indent + SetResKey.Length] == (byte)'=')
            {
                var valueStart = indent + keyLength;
                var valueEnd = valueStart;
                while (valueEnd < line.Length && (char.IsAsciiDigit((char)line[valueEnd]) || line[valueEnd] is (byte)'x' or (byte)'X'))
                {
                    valueEnd++;
                }

                var value = Encoding.ASCII.GetString(line[valueStart..valueEnd]);
                var separator = value.IndexOfAny(['x', 'X']);
                if (separator > 0 &&
                    uint.TryParse(value.AsSpan(0, separator), NumberStyles.None, CultureInfo.InvariantCulture, out var w) &&
                    uint.TryParse(value.AsSpan(separator + 1), NumberStyles.None, CultureInfo.InvariantCulture, out var h))
                {
                    found.Add((lineStart + valueStart, valueEnd - valueStart, w, h));
                }
            }

            lineStart = lineEnd + 1;
        }

        return found;
    }

    /// <summary><c>WxH</c> in exactly <paramref name="length"/> characters, the width zero-padded;
    /// null when it does not fit.</summary>
    internal static byte[]? EncodeSetRes(uint width, uint height, int length)
    {
        var heightText = height.ToString(CultureInfo.InvariantCulture);
        var widthLength = length - 1 - heightText.Length;
        var widthText = width.ToString(CultureInfo.InvariantCulture);
        if (widthLength < widthText.Length)
        {
            return null;
        }

        return Encoding.ASCII.GetBytes(widthText.PadLeft(widthLength, '0') + "x" + heightText);
    }

    /// <summary>
    /// Reads a version 10 or 11 pak's index and lists its .ini entries stored uncompressed and
    /// unencrypted, with the offset of their data. The footer is the last 221 bytes:
    /// key GUID (16), encrypted-index flag (1), magic, version, index offset and size, index
    /// hash (20), five compression-method names (32 each).
    /// </summary>
    internal static PakIndex ReadIniEntries(Stream stream, out List<IniEntry> inis)
    {
        inis = [];
        const int TailLength = 512;
        var length = stream.Length;
        if (length < 64)
        {
            return PakIndex.NotAPak;
        }

        var tailLength = (int)Math.Min(TailLength, length);
        var tail = new byte[tailLength];
        stream.Position = length - tailLength;
        stream.ReadExactly(tail);
        var magic = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(magic, PakMagic);
        var at = tail.AsSpan().LastIndexOf(magic);
        if (at < 1 || at + 24 > tail.Length)
        {
            return PakIndex.NotAPak;
        }

        var version = BinaryPrimitives.ReadUInt32LittleEndian(tail.AsSpan(at + 4));
        var indexOffset = BinaryPrimitives.ReadInt64LittleEndian(tail.AsSpan(at + 8));
        var indexSize = BinaryPrimitives.ReadInt64LittleEndian(tail.AsSpan(at + 16));
        if (tail[at - 1] != 0)
        {
            return PakIndex.Encrypted;
        }

        if (version is not (10 or 11) || indexOffset < 0 || indexSize <= 0 || indexSize > 256 << 20 ||
            indexOffset + indexSize > length)
        {
            return PakIndex.Unsupported;
        }

        var index = ReadAt(stream, indexOffset, (int)indexSize);
        var reader = new Reader(index);
        var mount = reader.String();
        _ = reader.Int32(); // entry count
        _ = reader.UInt64(); // path hash seed
        if (reader.Int32() != 0)
        {
            reader.Skip(8 + 8 + 20); // path hash index offset, size, hash
        }

        if (reader.Int32() == 0)
        {
            return PakIndex.Unsupported;
        }

        var directoryOffset = reader.Int64();
        var directorySize = reader.Int64();
        reader.Skip(20);
        var encodedSize = reader.Int32();
        var encoded = reader.Bytes(encodedSize);
        if (directoryOffset < 0 || directorySize <= 0 || directorySize > 256 << 20 || directoryOffset + directorySize > length)
        {
            return PakIndex.Unsupported;
        }

        var directories = new Reader(ReadAt(stream, directoryOffset, (int)directorySize));
        var directoryCount = directories.Int32();
        for (var d = 0; d < directoryCount; d++)
        {
            var directory = directories.String();
            var fileCount = directories.Int32();
            for (var f = 0; f < fileCount; f++)
            {
                var file = directories.String();
                var location = directories.Int32();
                if (location < 0 || !file.EndsWith(".ini", StringComparison.OrdinalIgnoreCase) ||
                    !TryDecodeEntry(encoded, location, out var offset, out var size, out var compressed, out var encryptedEntry) ||
                    compressed || encryptedEntry)
                {
                    continue;
                }

                // The serialized entry header (offset, sizes, compression method, hash, flags,
                // block size: 53 bytes uncompressed) precedes the data.
                inis.Add(new IniEntry(mount + directory + file, offset + 53, size));
            }
        }

        return PakIndex.Readable;
    }

    /// <summary>Decodes one bit-packed FPakEntry of the v10+ encoded entry blob.</summary>
    internal static bool TryDecodeEntry(
        ReadOnlySpan<byte> encoded,
        int location,
        out long offset,
        out long size,
        out bool compressed,
        out bool encrypted)
    {
        offset = size = 0;
        compressed = encrypted = false;
        if (location < 0 || location + 4 > encoded.Length)
        {
            return false;
        }

        var bits = BinaryPrimitives.ReadUInt32LittleEndian(encoded[location..]);
        var at = location + 4;
        compressed = ((bits >> 23) & 0x3F) != 0;
        encrypted = ((bits >> 22) & 1) != 0;
        if (!TryReadPacked(encoded, ref at, (bits & (1u << 31)) != 0, out offset) ||
            !TryReadPacked(encoded, ref at, (bits & (1u << 30)) != 0, out size))
        {
            return false;
        }

        return true;
    }

    private static bool TryReadPacked(ReadOnlySpan<byte> bytes, ref int at, bool is32, out long value)
    {
        var width = is32 ? 4 : 8;
        if (at + width > bytes.Length)
        {
            value = 0;
            return false;
        }

        value = is32
            ? BinaryPrimitives.ReadUInt32LittleEndian(bytes[at..])
            : BinaryPrimitives.ReadInt64LittleEndian(bytes[at..]);
        at += width;
        return true;
    }

    private static byte[] ReadAt(Stream stream, long offset, int count)
    {
        var bytes = new byte[count];
        stream.Position = offset;
        stream.ReadExactly(bytes);
        return bytes;
    }

    private sealed class Reader(byte[] bytes)
    {
        private int _at;

        public int Int32()
        {
            var value = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(_at));
            _at += 4;
            return value;
        }

        public long Int64()
        {
            var value = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(_at));
            _at += 8;
            return value;
        }

        public ulong UInt64()
        {
            var value = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(_at));
            _at += 8;
            return value;
        }

        public void Skip(int count) => _at += count;

        public byte[] Bytes(int count)
        {
            var value = bytes.AsSpan(_at, count).ToArray();
            _at += count;
            return value;
        }

        /// <summary>An FString: a length (negative for UTF-16 code units) including the NUL.</summary>
        public string String()
        {
            var count = Int32();
            if (count == 0)
            {
                return string.Empty;
            }

            string value;
            if (count < 0)
            {
                value = Encoding.Unicode.GetString(bytes, _at, -count * 2);
                _at += -count * 2;
            }
            else
            {
                value = Encoding.UTF8.GetString(bytes, _at, count);
                _at += count;
            }

            return value.TrimEnd('\0');
        }
    }
}
