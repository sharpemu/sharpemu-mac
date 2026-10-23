// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.IO.Compression;
using System.IO.Hashing;
using System.Text;

namespace SharpEmu.Libs.Gpu.Pipelines;

/// <summary>
/// The vertex and pixel programs a title translated (graphics-prewarm-*.bin), recorded as
/// what the translator was handed: the guest code and the translation inputs of the
/// shader corpus (stage options, specialization, push-data start, host flags), so a start
/// can translate them again offline. The compute prewarm has always worked this way. A
/// graphics program was only recorded as the MSL it translated to, which a translator
/// change leaves unusable: every library then compiled on demand on the GPU worker, one
/// draw's pair at a time, with the screen black for minutes. The list is kept apart from
/// the compute list (an older build reading a record kind it does not know ends its read
/// there and drops the rest). Each record schema keeps its own list, as the compute
/// list's do.
/// </summary>
public sealed class GraphicsPrewarmList : IDisposable
{
    public const int MaxOtherLists = 3;
    private const string FilePrefix = "graphics-prewarm";
    private const uint Magic = 0x50575047;
    private const uint FormatVersion = 2;
    private const byte CodeKind = 1;
    private const byte GraphicsKind = 2;
    private const int RecordHeaderBytes = sizeof(uint) + sizeof(ulong);
    // A graphics record's payload: the key of its guest code, its identity, then the compressed body.
    private const int PrefixBytes = sizeof(ulong) + sizeof(uint) + 2 * sizeof(ulong) + sizeof(ulong);

    public static string Stem => $"{FilePrefix}-{SchemaFingerprint.ToString("x8", CultureInfo.InvariantCulture)}";
    public static string FileName => Stem + ".bin";
    public static string StampFileName => Stem + ".stamp";
    public static string FirstFrameFileName => Stem + ".first-frame";

    /// <summary>One recorded program: its identity, its guest code and its compressed record body.</summary>
    public readonly record struct Item(ulong Identity, ShaderCodeCapture Code, byte[] Body);

    private readonly object _gate = new();
    private readonly FileStream _file;
    private readonly string _stampPath;
    private readonly string _firstFramePath;
    private readonly HashSet<(ulong Hash, uint CodeSize, ulong Address, ulong ContinuationAddressOffset)> _codes = [];
    private readonly HashSet<ulong> _identities = [];
    private readonly Dictionary<(ulong Hash, uint CodeSize, ulong Address, ulong ContinuationAddressOffset), ShaderCodeCapture> _loadedCodes = new();
    private readonly List<(ulong Identity, (ulong, uint, ulong, ulong) CodeKey, byte[] Body)> _loaded = [];
    // The programs this session created, in order, until its first guest frame.
    private readonly List<ulong> _sessionPrograms = [];
    private readonly HashSet<ulong> _sessionProgramSet = [];
    private bool _firstFrameWritten;
    private bool _failed;

    private GraphicsPrewarmList(FileStream file, string stampPath, string firstFramePath)
    {
        _file = file;
        _stampPath = stampPath;
        _firstFramePath = firstFramePath;
    }

    public int LoadedCount => _loaded.Count;

    public static GraphicsPrewarmList? Open(string directory)
    {
        try
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, FileName);
            var file = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
            var list = new GraphicsPrewarmList(
                file, Path.Combine(directory, StampFileName), Path.Combine(directory, FirstFrameFileName));
            try
            {
                list.Load();
            }
            catch
            {
                file.Dispose();
                throw;
            }

            PruneOtherLists(directory);
            return list;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Graphics prewarm list unavailable: {exception.Message}");
            return null;
        }
    }

    /// <summary>The recorded programs whose guest code is on the list, in the order they were first recorded.</summary>
    public IReadOnlyList<Item> Loaded()
    {
        var result = new List<Item>(_loaded.Count);
        foreach (var (identity, codeKey, body) in _loaded)
        {
            if (_loadedCodes.TryGetValue(codeKey, out var code))
            {
                result.Add(new Item(identity, code, body));
            }
        }

        return result;
    }

    /// <summary>The record of a listed program, with the guest code it was recorded with.</summary>
    public static ShaderCorpusRecord Decode(in Item item)
    {
        using var input = new ZLibStream(new MemoryStream(item.Body), CompressionMode.Decompress);
        using var json = new MemoryStream();
        input.CopyTo(json);
        return ShaderCorpus.DeserializeBody(json.ToArray()) with { Code = item.Code };
    }

    /// <summary>
    /// The translator stamp and the number of libraries the last complete prewarm made from
    /// this list; null when none has finished. The libraries themselves are in the host's
    /// library list, which a start only trusts while it holds that many.
    /// </summary>
    public (string Stamp, int Libraries)? ReadComplete()
    {
        try
        {
            if (!File.Exists(_stampPath))
            {
                return null;
            }

            var lines = File.ReadAllLines(_stampPath);
            return lines.Length == 2 && int.TryParse(lines[1], NumberStyles.None, CultureInfo.InvariantCulture, out var libraries)
                ? (lines[0], libraries)
                : null;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    public void WriteComplete(string stamp, int libraries)
    {
        try
        {
            File.WriteAllLines(_stampPath, [stamp, libraries.ToString(CultureInfo.InvariantCulture)]);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Graphics prewarm stamp write failed: {exception.Message}");
        }
    }

    /// <summary>
    /// The programs the last session that reached a guest frame created before its first
    /// one, by identity; null when no session has. A prewarm can hold the guest for these
    /// alone and compile the rest while it runs.
    /// </summary>
    public HashSet<ulong>? ReadFirstFrame()
    {
        try
        {
            if (!File.Exists(_firstFramePath))
            {
                return null;
            }

            var identities = new HashSet<ulong>();
            foreach (var line in File.ReadAllLines(_firstFramePath))
            {
                if (ulong.TryParse(line, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var identity))
                {
                    identities.Add(identity);
                }
            }

            return identities;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>The guest showed its first frame: what this session created until now is what the next start holds it for. Once per session.</summary>
    public void WriteFirstFrame()
    {
        ulong[] identities;
        lock (_gate)
        {
            if (_firstFrameWritten)
            {
                return;
            }

            _firstFrameWritten = true;
            identities = _sessionPrograms.ToArray();
            _sessionPrograms.Clear();
            _sessionProgramSet.Clear();
        }

        try
        {
            var temporary = _firstFramePath + ".tmp";
            File.WriteAllLines(temporary, identities.Select(identity => identity.ToString("X16", CultureInfo.InvariantCulture)));
            File.Move(temporary, _firstFramePath, overwrite: true);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Graphics prewarm first-frame list write failed: {exception.Message}");
        }
    }

    /// <summary>Records one permutation the shader cache is about to translate.</summary>
    public void Record(ShaderCodeCapture code, ShaderCorpusRecord record)
    {
        byte[] body;
        ulong identity;
        try
        {
            // The values of the user data are the draw's; the translation reads their count.
            var json = ShaderCorpus.SerializeBody(record with { UserData = new uint[record.UserData.Length] });
            // Where the guest loaded the program is not part of what it translates to (a fused
            // program keeps the distance to its continuation); the same program at another
            // address in each session would otherwise be recorded again.
            var registered = record.Registered;
            var placed = ShaderCorpus.SerializeBody(record with
            {
                UserData = new uint[record.UserData.Length],
                Registered = registered with
                {
                    CodeAddress = 0,
                    HeaderAddress = 0,
                    UserDataAddress = 0,
                    InputSemanticsAddress = 0,
                    ContinuationAddress = registered.ContinuationAddress == 0 ? 0 : registered.ContinuationAddress - registered.CodeAddress,
                },
            });
            identity = XxHash3.HashToUInt64(placed);
            body = Compress(json);
        }
        catch (Exception exception) when (exception is NotSupportedException or InvalidOperationException or System.Text.Json.JsonException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Graphics prewarm could not record a program: {exception.Message}");
            return;
        }

        lock (_gate)
        {
            if (!_firstFrameWritten && _sessionProgramSet.Add(identity))
            {
                _sessionPrograms.Add(identity);
            }

            if (_failed || !_identities.Add(identity))
            {
                return;
            }

            if (_codes.Add(code.Key))
            {
                Append(CodeKind, ShaderPrewarmList.SerializeCode(code));
            }

            var payload = new byte[PrefixBytes + body.Length];
            using (var stream = new MemoryStream(payload))
            using (var writer = new BinaryWriter(stream))
            {
                var key = code.Key;
                writer.Write(key.Hash);
                writer.Write(key.CodeSize);
                writer.Write(key.Address);
                writer.Write(key.ContinuationAddressOffset);
                writer.Write(identity);
            }

            body.CopyTo(payload, PrefixBytes);
            Append(GraphicsKind, payload);
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _failed = true;
            _file.Dispose();
        }
    }

    private static byte[] Compress(byte[] json)
    {
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(output, CompressionLevel.Fastest, leaveOpen: true))
        {
            zlib.Write(json);
        }

        return output.ToArray();
    }

    private void Append(byte kind, byte[] payload)
    {
        try
        {
            var record = new byte[RecordHeaderBytes + 1 + payload.Length];
            record[RecordHeaderBytes] = kind;
            payload.CopyTo(record, RecordHeaderBytes + 1);
            var body = record.AsSpan(RecordHeaderBytes);
            BitConverter.TryWriteBytes(record.AsSpan(0, sizeof(uint)), (uint)body.Length);
            BitConverter.TryWriteBytes(record.AsSpan(sizeof(uint), sizeof(ulong)), XxHash3.HashToUInt64(body));
            _file.Write(record);
            _file.Flush();
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or ObjectDisposedException)
        {
            _failed = true;
            Console.Error.WriteLine($"[LOADER][WARN] Graphics prewarm list write failed: {exception.Message}");
        }
    }

    private void Load()
    {
        var content = new byte[_file.Length];
        _file.Position = 0;
        _file.ReadExactly(content);
        var validEnd = 0L;
        if (content.Length >= 3 * sizeof(uint) &&
            BitConverter.ToUInt32(content, 0) == Magic &&
            BitConverter.ToUInt32(content, sizeof(uint)) == FormatVersion &&
            BitConverter.ToUInt32(content, 2 * sizeof(uint)) == SchemaFingerprint)
        {
            validEnd = 3 * sizeof(uint);
            var offset = (int)validEnd;
            while (content.Length - offset >= RecordHeaderBytes + 1)
            {
                var length = BitConverter.ToUInt32(content, offset);
                var checksum = BitConverter.ToUInt64(content, offset + sizeof(uint));
                if (length == 0 || length > content.Length - offset - RecordHeaderBytes)
                {
                    break;
                }

                var body = content.AsSpan(offset + RecordHeaderBytes, (int)length);
                if (XxHash3.HashToUInt64(body) != checksum || !TryLoadRecord(body[0], body[1..].ToArray()))
                {
                    break;
                }

                offset += RecordHeaderBytes + (int)length;
                validEnd = offset;
            }
        }

        if (validEnd == 0)
        {
            _file.SetLength(0);
            _file.Position = 0;
            Span<byte> header = stackalloc byte[3 * sizeof(uint)];
            BitConverter.TryWriteBytes(header, Magic);
            BitConverter.TryWriteBytes(header[sizeof(uint)..], FormatVersion);
            BitConverter.TryWriteBytes(header[(2 * sizeof(uint))..], SchemaFingerprint);
            _file.Write(header);
            _file.Flush();
            return;
        }

        _file.SetLength(validEnd);
        _file.Position = validEnd;
    }

    private bool TryLoadRecord(byte kind, byte[] payload)
    {
        try
        {
            switch (kind)
            {
                case CodeKind:
                {
                    var code = ShaderPrewarmList.DeserializeCode(payload);
                    _codes.Add(code.Key);
                    _loadedCodes[code.Key] = code;
                    return true;
                }

                case GraphicsKind when payload.Length > PrefixBytes:
                {
                    using var reader = new BinaryReader(new MemoryStream(payload, 0, PrefixBytes));
                    var codeKey = (reader.ReadUInt64(), reader.ReadUInt32(), reader.ReadUInt64(), reader.ReadUInt64());
                    var identity = reader.ReadUInt64();
                    var body = payload.AsSpan(PrefixBytes).ToArray();
                    if (_identities.Add(identity))
                    {
                        _loaded.Add((identity, codeKey, body));
                    }

                    return true;
                }

                default:
                    return false;
            }
        }
        catch (Exception exception) when (exception is EndOfStreamException or ArgumentException or IOException or InvalidDataException)
        {
            return false;
        }
    }

    private static void PruneOtherLists(string directory)
    {
        try
        {
            var others = new DirectoryInfo(directory)
                .EnumerateFiles(FilePrefix + "*.bin")
                .Where(file => !string.Equals(file.Name, FileName, StringComparison.Ordinal))
                .OrderByDescending(file => file.LastWriteTimeUtc)
                .Skip(MaxOtherLists);
            foreach (var file in others)
            {
                var stem = Path.GetFileNameWithoutExtension(file.Name);
                foreach (var extension in new[] { ".bin", ".stamp", ".first-frame" })
                {
                    File.Delete(Path.Combine(directory, stem + extension));
                }
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Graphics prewarm could not remove older lists: {exception.Message}");
        }
    }

    private static readonly uint SchemaFingerprint = ComputeSchemaFingerprint();

    // Every type the record body serializes, by its properties, so a record from a build whose
    // translation inputs differ is not read as this one's.
    private static uint ComputeSchemaFingerprint()
    {
        var types = new SortedDictionary<string, Type>(StringComparer.Ordinal);
        var pending = new Stack<Type>();
        pending.Push(typeof(ShaderCorpusRecord));
        while (pending.TryPop(out var type))
        {
            if (type.IsArray || type.IsGenericType)
            {
                foreach (var inner in type.IsArray ? [type.GetElementType()!] : type.GetGenericArguments())
                {
                    pending.Push(inner);
                }

                continue;
            }

            if (type.FullName is not { } name || !name.StartsWith("SharpEmu.", StringComparison.Ordinal) ||
                type == typeof(ShaderCodeCapture) || !types.TryAdd(name, type))
            {
                continue;
            }

            foreach (var property in type.GetProperties())
            {
                pending.Push(property.PropertyType);
            }
        }

        var builder = new StringBuilder();
        foreach (var (name, type) in types)
        {
            builder.Append(name).Append('{');
            foreach (var property in type.GetProperties().OrderBy(static property => property.Name, StringComparer.Ordinal))
            {
                builder.Append(property.PropertyType.ToString()).Append(' ').Append(property.Name).Append(';');
            }

            builder.Append('}');
        }

        return (uint)XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(builder.ToString()));
    }
}
