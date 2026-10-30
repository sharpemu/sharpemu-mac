// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.IO.Hashing;
using System.Security.Cryptography;
using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

internal sealed class RecordingCpuMemory(ICpuMemory inner) : ICpuMemory, ICpuMemoryWrapper
{
    private readonly List<(ulong Address, byte[] Bytes)> _reads = [];

    public ICpuMemory Inner => inner;

    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        if (!inner.TryRead(virtualAddress, destination))
        {
            return false;
        }

        _reads.Add((virtualAddress, destination.ToArray()));
        return true;
    }

    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;

    public CodeRange[] TakeRanges()
    {
        _reads.Sort(static (left, right) => left.Address.CompareTo(right.Address));
        var ranges = new List<CodeRange>();
        ulong start = 0;
        var bytes = new List<byte>();
        foreach (var (address, data) in _reads)
        {
            var end = start + (ulong)bytes.Count;
            if (bytes.Count != 0 && address <= end)
            {
                var overlap = (int)(end - address);
                if (overlap < data.Length)
                {
                    bytes.AddRange(data.AsSpan(overlap).ToArray());
                }

                continue;
            }

            if (bytes.Count != 0)
            {
                ranges.Add(new CodeRange(start, bytes.ToArray()));
            }

            start = address;
            bytes.Clear();
            bytes.AddRange(data);
        }

        if (bytes.Count != 0)
        {
            ranges.Add(new CodeRange(start, bytes.ToArray()));
        }

        _reads.Clear();
        return ranges.ToArray();
    }
}

internal sealed class ReplayCpuMemory(IReadOnlyList<CodeRange> ranges) : ICpuMemory
{
    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        foreach (var range in ranges)
        {
            if (virtualAddress < range.Address)
            {
                continue;
            }

            var offset = virtualAddress - range.Address;
            if (offset + (ulong)destination.Length <= (ulong)range.Bytes.Length)
            {
                range.Bytes.AsSpan((int)offset, destination.Length).CopyTo(destination);
                return true;
            }
        }

        return false;
    }

    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;
}

public readonly record struct CodeRange(ulong Address, byte[] Bytes);

public readonly record struct FusedCodeParts(ulong EntryHeaderAddress, ulong ContinuationAddress, ulong ContinuationHeaderAddress);

public sealed class ShaderCodeCapture
{
    public required ulong Hash { get; init; }
    public required uint CodeSize { get; init; }
    public required ulong Address { get; init; }
    public required Generation Generation { get; init; }
    public FusedCodeParts? Fused { get; init; }
    public required CodeRange[] Ranges { get; init; }

    public (ulong Hash, uint CodeSize, ulong Address, ulong ContinuationAddressOffset) Key =>
        (Hash, CodeSize, Address, Fused is { } fused ? unchecked(fused.ContinuationAddress - Address) : 0);

    public CpuContext CreateContext()
    {
        var context = new CpuContext(new ReplayCpuMemory(Ranges), Generation);
        if (Fused is { } fused)
        {
            Gen5ShaderTranslator.RegisterFusedProgram(
                context, Address, fused.EntryHeaderAddress, fused.ContinuationAddress, fused.ContinuationHeaderAddress);
        }

        return context;
    }
}

public sealed class ComputePrewarmRecord
{
    public ulong ContinuationAddressOffset { get; init; }
    public required ulong Hash { get; init; }
    public required uint CodeSize { get; init; }
    public required ulong Address { get; init; }
    public required uint UserDataBase { get; init; }
    public required uint UserDataCount { get; init; }
    public required uint PushDataCursor { get; init; }
    public required ComputeInputInfo Info { get; init; }
    public Gen5ComputeSystemRegisters? SystemRegisters { get; init; }
    public required ResourceSpecialization Specialization { get; init; }

    public (ulong Hash, uint CodeSize, ulong Address, ulong ContinuationAddressOffset) CodeKey =>
        (Hash, CodeSize, Address, ContinuationAddressOffset);
}

public sealed class ShaderPrewarmList : IDisposable
{
    // Each record schema keeps its own list, stamp and progress: builds whose records
    // differ share the title's directory, and one list for all of them was emptied by
    // whichever build opened it last, so the next start of the other had nothing to
    // prewarm. The lists of other schemas are kept, the most recently written first,
    // up to MaxOtherLists.
    public const int MaxOtherLists = 3;
    private const string FilePrefix = "shader-prewarm";
    private const string LegacyStem = FilePrefix;

    public static string Stem => $"{FilePrefix}-{SchemaFingerprint.ToString("x8", CultureInfo.InvariantCulture)}";
    public static string FileName => Stem + ".bin";
    public static string StampFileName => Stem + ".stamp";
    public static string ProgressFileName => Stem + ".progress";
    public static string FirstFrameFileName => Stem + ".first-frame";

    private const uint Magic = 0x57504553;
    private const uint FormatVersion = 8;
    private const byte CodeKind = 1;
    private const byte ComputeKind = 2;
    private const int RecordHeaderBytes = sizeof(uint) + sizeof(ulong);

    private readonly object _gate = new();
    private readonly FileStream _file;
    private readonly string _stampPath;
    private readonly string _progressPath;
    private readonly object _progressGate = new();
    private readonly HashSet<(ulong Hash, uint CodeSize, ulong Address, ulong ContinuationAddressOffset)> _codes = [];
    private readonly HashSet<ulong> _computes = [];
    private readonly Dictionary<(ulong Hash, uint CodeSize, ulong Address, ulong ContinuationAddressOffset), ShaderCodeCapture> _loadedCodes = new();
    private readonly List<ComputePrewarmRecord> _loadedComputes = [];
    // The computes this session created, in order, until its first guest frame.
    private readonly List<ulong> _sessionComputes = [];
    private readonly HashSet<ulong> _sessionComputeSet = [];
    private string? _firstFramePath;
    private bool _firstFrameWritten;
    private bool _failed;

    private ShaderPrewarmList(FileStream file, string stampPath, string progressPath)
    {
        _file = file;
        _stampPath = stampPath;
        _progressPath = progressPath;
    }

    private string _loadedCoverage = string.Empty;

    public int LoadedComputeCount => _loadedComputes.Count;
    public string StampPath => _stampPath;

    public string CoverageStamp(string stamp) => $"{stamp}|coverage-v1|{_loadedCoverage}";

    public static string CacheStamp(string coverageStamp, ReadOnlySpan<byte> cache) =>
        $"{coverageStamp}|cache-v1|{Convert.ToHexString(SHA256.HashData(cache))}";

    public static ShaderPrewarmList? Open(string directory)
    {
        try
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, FileName);
            AdoptLegacyList(directory, path);
            var file = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
            var list = new ShaderPrewarmList(file, Path.Combine(directory, StampFileName), Path.Combine(directory, ProgressFileName))
            {
                _firstFramePath = Path.Combine(directory, FirstFrameFileName),
            };
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
            Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm list unavailable: {exception.Message}");
            return null;
        }
    }

    // A list from before the per-schema names is taken over when its schema is this one.
    private static void AdoptLegacyList(string directory, string path)
    {
        try
        {
            AdoptLegacyListCore(directory, path);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm could not adopt the older list: {exception.Message}");
        }
    }

    private static void AdoptLegacyListCore(string directory, string path)
    {
        var legacy = Path.Combine(directory, LegacyStem + ".bin");
        if (File.Exists(path) || !File.Exists(legacy) || ReadSchema(legacy) != SchemaFingerprint)
        {
            return;
        }

        // Copied, not moved: a build from before the per-schema names may still run on
        // this title, and it keeps its list until the bound on other lists removes it.
        foreach (var (from, to) in new[] { (".stamp", StampFileName), (".progress", ProgressFileName) })
        {
            var source = Path.Combine(directory, LegacyStem + from);
            if (File.Exists(source))
            {
                File.Copy(source, Path.Combine(directory, to), overwrite: true);
            }
        }

        File.Copy(legacy, path);
    }

    private static uint? ReadSchema(string path)
    {
        Span<byte> header = stackalloc byte[3 * sizeof(uint)];
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        if (stream.ReadAtLeast(header, header.Length, throwOnEndOfStream: false) < header.Length ||
            BitConverter.ToUInt32(header) != Magic ||
            BitConverter.ToUInt32(header[sizeof(uint)..]) != FormatVersion)
        {
            return null;
        }

        return BitConverter.ToUInt32(header[(2 * sizeof(uint))..]);
    }

    private static void PruneOtherLists(string directory)
    {
        try
        {
            PruneOtherListsCore(directory);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm could not remove older lists: {exception.Message}");
        }
    }

    private static void PruneOtherListsCore(string directory)
    {
        var others = new DirectoryInfo(directory)
            .EnumerateFiles(FilePrefix + "*.bin")
            .Where(file => !string.Equals(file.Name, FileName, StringComparison.Ordinal))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .Skip(MaxOtherLists);
        foreach (var file in others)
        {
            var stem = Path.GetFileNameWithoutExtension(file.Name);
            foreach (var extension in new[] { ".bin", ".stamp", ".progress", ".first-frame" })
            {
                File.Delete(Path.Combine(directory, stem + extension));
            }
        }
    }

    public IReadOnlyList<(ComputePrewarmRecord Record, ShaderCodeCapture Code)> LoadedComputes()
    {
        var result = new List<(ComputePrewarmRecord, ShaderCodeCapture)>(_loadedComputes.Count);
        foreach (var record in _loadedComputes)
        {
            if (_loadedCodes.TryGetValue(record.CodeKey, out var code))
            {
                result.Add((record, code));
            }
        }

        return result;
    }

    public bool IsStampCurrent(string stamp)
    {
        try
        {
            return File.Exists(_stampPath) && File.ReadAllText(_stampPath) == stamp;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    public void WriteStamp(string stamp)
        => WriteStamp(_stampPath, stamp);

    public static void WriteStamp(string path, string stamp)
    {
        try
        {
            var temporaryPath = path + $".{Environment.ProcessId}.tmp";
            File.WriteAllText(temporaryPath, stamp);
            File.Move(temporaryPath, path, overwrite: true);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm stamp write failed: {exception.Message}");
        }
    }

    // Where the guest loaded the shader is not part of what it compiles to: the same shader at another
    // address in each session made a new record every time, and the list grew with every start.
    public static ulong Identity(ComputePrewarmRecord record) => XxHash3.HashToUInt64(SerializeCompute(record, withAddress: false));

    public HashSet<ulong> ReadProgress(string stamp)
    {
        var completed = new HashSet<ulong>();
        lock (_progressGate)
        {
            try
            {
                if (!File.Exists(_progressPath))
                {
                    return completed;
                }

                var lines = File.ReadAllLines(_progressPath);
                if (lines.Length == 0 || lines[0] != ProgressKey(stamp))
                {
                    File.Delete(_progressPath);
                    return completed;
                }

                for (var index = 1; index < lines.Length; index++)
                {
                    if (ulong.TryParse(lines[index], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var identity))
                    {
                        completed.Add(identity);
                    }
                }
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                completed.Clear();
            }
        }

        return completed;
    }

    public void AppendProgress(string stamp, IReadOnlyCollection<ulong> identities)
    {
        if (identities.Count == 0)
        {
            return;
        }

        lock (_progressGate)
        {
            try
            {
                var lines = new List<string>(identities.Count + 1);
                if (!File.Exists(_progressPath))
                {
                    lines.Add(ProgressKey(stamp));
                }

                foreach (var identity in identities)
                {
                    lines.Add(identity.ToString("X16", CultureInfo.InvariantCulture));
                }

                File.AppendAllLines(_progressPath, lines);
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm progress write failed: {exception.Message}");
            }
        }
    }

    public void ClearProgress()
    {
        lock (_progressGate)
        {
            try
            {
                File.Delete(_progressPath);
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm progress clear failed: {exception.Message}");
            }
        }
    }

    /// <summary>
    /// The computes the last session that reached a guest frame created before its first
    /// one, by identity; null when no session has. A prewarm can hold the guest for these
    /// alone and compile the rest while it runs.
    /// </summary>
    public HashSet<ulong>? ReadFirstFrameComputes()
    {
        try
        {
            if (_firstFramePath is not { } path || !File.Exists(path))
            {
                return null;
            }

            var identities = new HashSet<ulong>();
            foreach (var line in File.ReadAllLines(path))
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

    /// <summary>
    /// The guest showed its first frame: the computes this session created until now are
    /// the ones the next start's prewarm holds the guest for. Once per session.
    /// </summary>
    public void WriteFirstFrameComputes()
    {
        ulong[] identities;
        lock (_gate)
        {
            if (_firstFrameWritten || _firstFramePath is null)
            {
                return;
            }

            _firstFrameWritten = true;
            identities = _sessionComputes.ToArray();
            _sessionComputes.Clear();
            _sessionComputeSet.Clear();
        }

        try
        {
            var temporary = _firstFramePath + ".tmp";
            File.WriteAllLines(temporary, identities.Select(identity => identity.ToString("X16", CultureInfo.InvariantCulture)));
            File.Move(temporary, _firstFramePath, overwrite: true);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm first-frame list write failed: {exception.Message}");
        }
    }

    private static string ProgressKey(string stamp) =>
        XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(stamp)).ToString("X16", CultureInfo.InvariantCulture);

    public void RecordCompute(ShaderCodeCapture code, ComputePrewarmRecord record)
    {
        var payload = SerializeCompute(record);
        var identity = Identity(record);
        lock (_gate)
        {
            if (!_firstFrameWritten && _sessionComputeSet.Add(identity))
            {
                _sessionComputes.Add(identity);
            }

            if (_failed || !_computes.Add(identity))
            {
                return;
            }

            if (_codes.Add(code.Key))
            {
                Append(CodeKind, SerializeCode(code));
            }

            Append(ComputeKind, payload);
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
            Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm list write failed: {exception.Message}");
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

        _loadedCoverage = Convert.ToHexString(SHA256.HashData(content.AsSpan(0, checked((int)validEnd))));
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
                    var code = DeserializeCode(payload);
                    _codes.Add(code.Key);
                    _loadedCodes[code.Key] = code;
                    return true;
                }

                case ComputeKind:
                {
                    var record = DeserializeCompute(payload);
                    if (_computes.Add(Identity(record)))
                    {
                        _loadedComputes.Add(record);
                    }

                    return true;
                }

                default:
                    return false;
            }
        }
        catch (Exception exception) when (exception is EndOfStreamException or ArgumentException or IOException)
        {
            return false;
        }
    }

    private static readonly uint SchemaFingerprint = ComputeSchemaFingerprint();

    private static uint ComputeSchemaFingerprint()
    {
        var builder = new StringBuilder();
        foreach (var type in new[]
        {
            typeof(ComputeInputInfo), typeof(Gen5ComputeSystemRegisters), typeof(ResourceSpecialization),
            typeof(BufferSpecialization), typeof(ImageSpecialization), typeof(BufferCandidateTableSpecialization),
        })
        {
            builder.Append(type.FullName).Append('{');
            foreach (var property in type.GetProperties().OrderBy(static property => property.Name, StringComparer.Ordinal))
            {
                builder.Append(property.PropertyType.FullName).Append(' ').Append(property.Name).Append(';');
            }

            builder.Append('}');
        }

        return (uint)XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(builder.ToString()));
    }

    internal static byte[] SerializeCode(ShaderCodeCapture code)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write(code.Hash);
        writer.Write(code.CodeSize);
        writer.Write(code.Address);
        writer.Write((int)code.Generation);
        writer.Write(code.Fused is not null);
        if (code.Fused is { } fused)
        {
            writer.Write(fused.EntryHeaderAddress);
            writer.Write(fused.ContinuationAddress);
            writer.Write(fused.ContinuationHeaderAddress);
        }

        writer.Write(code.Ranges.Length);
        foreach (var range in code.Ranges)
        {
            writer.Write(range.Address);
            writer.Write(range.Bytes.Length);
            writer.Write(range.Bytes);
        }

        writer.Flush();
        return stream.ToArray();
    }

    internal static ShaderCodeCapture DeserializeCode(byte[] payload)
    {
        using var reader = new BinaryReader(new MemoryStream(payload));
        var hash = reader.ReadUInt64();
        var codeSize = reader.ReadUInt32();
        var address = reader.ReadUInt64();
        var generation = (Generation)reader.ReadInt32();
        FusedCodeParts? fused = reader.ReadBoolean()
            ? new FusedCodeParts(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64())
            : null;
        var ranges = new CodeRange[CheckedCount(reader.ReadInt32(), payload.Length)];
        for (var index = 0; index < ranges.Length; index++)
        {
            var rangeAddress = reader.ReadUInt64();
            var bytes = reader.ReadBytes(CheckedCount(reader.ReadInt32(), payload.Length));
            ranges[index] = new CodeRange(rangeAddress, bytes);
        }

        return new ShaderCodeCapture
        {
            Hash = hash,
            CodeSize = codeSize,
            Address = address,
            Generation = generation,
            Fused = fused,
            Ranges = ranges,
        };
    }

    private static byte[] SerializeCompute(ComputePrewarmRecord record, bool withAddress = true)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write(record.Hash);
        writer.Write(record.CodeSize);
        writer.Write(withAddress ? record.Address : 0UL);
        writer.Write(record.ContinuationAddressOffset);
        writer.Write(record.UserDataBase);
        writer.Write(record.UserDataCount);
        writer.Write(record.PushDataCursor);

        var info = record.Info;
        writer.Write(info.ThreadsX);
        writer.Write(info.ThreadsY);
        writer.Write(info.ThreadsZ);
        writer.Write(info.DispatchThreadDimensions);
        writer.Write(info.GroupIdX);
        writer.Write(info.GroupIdY);
        writer.Write(info.GroupIdZ);
        writer.Write(info.ThreadIdCount);
        writer.Write(info.ThreadGroupSizeEnabled);
        writer.Write(info.WaveSize);
        writer.Write(info.LocalDataShareDwords);
        writer.Write(info.ScratchDwords);
        writer.Write(info.IeeeMode);
        writer.Write(info.FloatMode);
        writer.Write(info.Dx10Clamp);
        writer.Write(info.NeedsLocalDataShareBarriers);
        writer.Write(info.WorkgroupRegister);

        writer.Write(record.SystemRegisters is not null);
        if (record.SystemRegisters is { } registers)
        {
            WriteOptional(writer, registers.WorkGroupXRegister);
            WriteOptional(writer, registers.WorkGroupYRegister);
            WriteOptional(writer, registers.WorkGroupZRegister);
            WriteOptional(writer, registers.ThreadGroupSizeRegister);
        }

        var specialization = record.Specialization;
        writer.Write(specialization.BaseBufferCount);
        writer.Write(specialization.Buffers.Count);
        foreach (var buffer in specialization.Buffers)
        {
            writer.Write(buffer.PackedStride);
            writer.Write(buffer.DescriptorFormat);
            writer.Write(buffer.DescriptorSwizzle);
        }

        writer.Write(specialization.Images.Count);
        foreach (var image in specialization.Images)
        {
            writer.Write((int)image.NumericClass);
            writer.Write((int)image.Dimension);
            writer.Write(image.MipCount);
            writer.Write(image.ConversionFormat);
            writer.Write(image.ShaderSwizzle);
            writer.Write(image.IndirectRoot);
            writer.Write(image.IndirectMappingOffset);
            writer.Write(image.IndirectSearchIterations);
            writer.Write(image.Cube);
            writer.Write(image.EmulatedCompareFunction);
            writer.Write(image.PackedFormat);
        }

        writer.Write(specialization.BufferCandidateTables.Count);
        foreach (var table in specialization.BufferCandidateTables)
        {
            writer.Write(table.FirstCandidate);
            writer.Write(table.CandidateCount);
            writer.Write(table.MappingOffset);
            writer.Write(table.SearchIterations);
        }

        // Which descriptors alias which: ResourceMaterializer merges image and sampler
        // bindings whose descriptors are identical, so the merged program has fewer elements
        // in a descriptor field than the same record without them, and a translation made
        // for one argument-buffer layout must never stand in for the other.
        WriteWords(writer, specialization.ImageDescriptorGroups);
        WriteWords(writer, specialization.SamplerDescriptorGroups);
        writer.Write(specialization.Samplers.Count);
        foreach (var sampler in specialization.Samplers)
        {
            writer.Write(sampler.LodBias);
            writer.Write(sampler.MirrorOnceBorderAxes);
            writer.Write(sampler.BorderColorType);
            writer.Write(sampler.ForceDegamma);
            writer.Write(sampler.Reduction);
            writer.Write(sampler.ReductionAcrossLevels);
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static ComputePrewarmRecord DeserializeCompute(byte[] payload)
    {
        using var reader = new BinaryReader(new MemoryStream(payload));
        var hash = reader.ReadUInt64();
        var codeSize = reader.ReadUInt32();
        var address = reader.ReadUInt64();
        var continuationAddressOffset = reader.ReadUInt64();
        var userDataBase = reader.ReadUInt32();
        var userDataCount = reader.ReadUInt32();
        var pushDataCursor = reader.ReadUInt32();
        var info = new ComputeInputInfo
        {
            ThreadsX = reader.ReadUInt32(),
            ThreadsY = reader.ReadUInt32(),
            ThreadsZ = reader.ReadUInt32(),
            DispatchThreadDimensions = reader.ReadBoolean(),
            GroupIdX = reader.ReadBoolean(),
            GroupIdY = reader.ReadBoolean(),
            GroupIdZ = reader.ReadBoolean(),
            ThreadIdCount = reader.ReadInt32(),
            ThreadGroupSizeEnabled = reader.ReadBoolean(),
            WaveSize = reader.ReadUInt32(),
            LocalDataShareDwords = reader.ReadUInt32(),
            ScratchDwords = reader.ReadUInt32(),
            IeeeMode = reader.ReadBoolean(),
            FloatMode = reader.ReadUInt32(),
            Dx10Clamp = reader.ReadBoolean(),
            NeedsLocalDataShareBarriers = reader.ReadBoolean(),
            WorkgroupRegister = reader.ReadInt32(),
        };

        Gen5ComputeSystemRegisters? registers = reader.ReadBoolean()
            ? new Gen5ComputeSystemRegisters(ReadOptional(reader), ReadOptional(reader), ReadOptional(reader), ReadOptional(reader))
            : null;

        var specialization = new ResourceSpecialization { BaseBufferCount = reader.ReadInt32() };
        for (var count = CheckedCount(reader.ReadInt32(), payload.Length); count > 0; count--)
        {
            specialization.Buffers.Add(new BufferSpecialization(reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32()));
        }

        for (var count = CheckedCount(reader.ReadInt32(), payload.Length); count > 0; count--)
        {
            specialization.Images.Add(new ImageSpecialization(
                (ImageNumericClass)reader.ReadInt32(),
                (ImageDimension)reader.ReadInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadBoolean(),
                reader.ReadInt32(),
                PackedFormat: reader.ReadUInt32()));
        }

        for (var count = CheckedCount(reader.ReadInt32(), payload.Length); count > 0; count--)
        {
            specialization.BufferCandidateTables.Add(new BufferCandidateTableSpecialization(
                reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32()));
        }

        ReadWords(reader, payload.Length, specialization.ImageDescriptorGroups);
        ReadWords(reader, payload.Length, specialization.SamplerDescriptorGroups);
        for (var count = CheckedCount(reader.ReadInt32(), payload.Length); count > 0; count--)
        {
            specialization.Samplers.Add(new SamplerSpecialization(
                reader.ReadInt32(), reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadBoolean(), reader.ReadUInt32(), reader.ReadBoolean()));
        }

        return new ComputePrewarmRecord
        {
            Hash = hash,
            CodeSize = codeSize,
            Address = address,
            UserDataBase = userDataBase,
            ContinuationAddressOffset = continuationAddressOffset,
            UserDataCount = userDataCount,
            PushDataCursor = pushDataCursor,
            Info = info,
            SystemRegisters = registers,
            Specialization = specialization,
        };
    }

    private static void WriteWords(BinaryWriter writer, List<uint> words)
    {
        writer.Write(words.Count);
        foreach (var word in words)
        {
            writer.Write(word);
        }
    }

    private static void ReadWords(BinaryReader reader, int payloadLength, List<uint> words)
    {
        for (var count = CheckedCount(reader.ReadInt32(), payloadLength); count > 0; count--)
        {
            words.Add(reader.ReadUInt32());
        }
    }

    private static void WriteOptional(BinaryWriter writer, uint? value)
    {
        writer.Write(value.HasValue);
        writer.Write(value.GetValueOrDefault());
    }

    private static uint? ReadOptional(BinaryReader reader)
    {
        var hasValue = reader.ReadBoolean();
        var value = reader.ReadUInt32();
        return hasValue ? value : null;
    }

    private static int CheckedCount(int count, int limit) =>
        count >= 0 && count <= limit ? count : throw new ArgumentException("The prewarm record count is out of range.");
}
