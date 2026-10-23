// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.IO.Hashing;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Logging;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The host capabilities the request builder reads, as the host reported them when the program was captured.
public sealed record ShaderCorpusHostFlags
{
    public bool GraphicsSubgroupOperations { get; init; }
    public bool SharedInt64Atomics { get; init; }
    public bool SignedZeroInfNanPreserve { get; init; }
    public bool ClipDistance { get; init; }
    public bool PerVertexPixelInputs { get; init; }
    public bool MipStatisticsSubgroupReduction { get; init; }
    public bool ExecGuardElision { get; init; }
    public bool ComputeWave64SubgroupNative { get; init; }
    public bool ComputeRunsOn32LaneSubgroups { get; init; }
}

// One program as the translator was about to see it. Options is null when the guest code did not
// decode; Specialization is null when the record was taken before the resources were materialized,
// which still replays the decode and the resource plan.
public sealed record ShaderCorpusRecord
{
    [JsonIgnore]
    public ShaderCodeCapture Code { get; init; } = null!;

    public ShaderStage Stage { get; init; }
    public ulong Hash { get; init; }
    public RegisteredShader Registered { get; init; } = null!;
    public uint[] UserData { get; init; } = [];
    public uint UserDataBase { get; init; }
    public ulong ExcludedUserDataRegisters { get; init; }
    public StageCompileOptions? Options { get; init; }
    public ResourceSpecialization? Specialization { get; init; }
    // The binding loads the specialization's plan left to the device (DeviceBindingLoads); the
    // replay plans with exactly these. Null (older records): with the learned ones.
    public int[]? DeviceBindingMemory { get; init; }
    public int[]? DeviceDataMemory { get; init; }
    public string DeviceLoadCompiler { get; init; } = ShaderResourcePlan.DeviceLoadCompilerIdentity;
    public uint PushDataCursor { get; init; }
    public ShaderCorpusHostFlags Host { get; init; } = new();
    public string? DecodeError { get; init; }

    // Identical code at two addresses shares one capture, which holds the first one's address; the replay
    // decodes where the bytes were captured.
    public ShaderSource CreateSource() =>
        new(Registered with
        {
            CodeAddress = Code.Address,
            ContinuationAddress = Code.Fused?.ContinuationAddress ?? 0,
        }, Hash, UserData, UserDataBase, Stage) { ExcludedUserDataRegisters = ExcludedUserDataRegisters };
}

// SHARPEMU_SHADER_CORPUS_DIR (a path, or 1 for shader-corpus under the logs folder): every distinct program is written before its translation is
// attempted, so a run that dies on the first shader the translator rejects still leaves the
// whole set that was submitted up to that point. File = header, the prewarm list's code record
// (the guest bytes), then the rest of the translation inputs as JSON.
public static class ShaderCorpus
{
    public const string DirectoryVariable = "SHARPEMU_SHADER_CORPUS_DIR";
    public const string FileExtension = ".corpus";

    private const uint Magic = 0x50434553;
    private const uint FormatVersion = 2;

    private static string? CorpusDirectory = SharpEmuPaths.ResolveDumpPath(DirectoryVariable, "shader-corpus");

    private static readonly object Gate = new();
    private static readonly HashSet<string> Written = [];
    private static readonly HashSet<string> PlannedStems = [];
    private static bool _directoryReady;
    private static bool _directoryFailed;

    private static readonly JsonSerializerOptions JsonOptions = CreateJsonOptions();

    internal static bool Enabled => CorpusDirectory is not null;

    // Points the writer somewhere else and forgets what it wrote; the environment variable is read once.
    internal static void UseDirectory(string? directory)
    {
        lock (Gate)
        {
            CorpusDirectory = directory;
            Written.Clear();
            PlannedStems.Clear();
            _directoryReady = false;
            _directoryFailed = false;
        }
    }

    // The guest code did not decode: all that can be kept is the bytes the decoder read.
    internal static void RecordDecodeFailure(ShaderSource source, ShaderCodeCapture code, string error, ShaderCorpusHostFlags host) =>
        Record($"{source.Label}-{source.Hash:X16}-decode", planStem: null, new ShaderCorpusRecord
        {
            Code = code,
            Stage = source.Stage,
            Hash = source.Hash,
            Registered = source.Registered,
            UserData = source.UserData,
            UserDataBase = source.UserDataBase,
            ExcludedUserDataRegisters = source.ExcludedUserDataRegisters,
            Host = host,
            DecodeError = error,
        });

    // Taken before the resource plan is extracted: the plan can reject a program without it ever
    // reaching a compile, and a rejection by a fatal error leaves no later chance to write.
    internal static void RecordPlan(ShaderSource source, StageCompileOptions options, ShaderCodeCapture code, ShaderCorpusHostFlags host)
    {
        try
        {
            var record = Create(source, options, code, host, specialization: null, pushDataCursor: 0);
            var stem = Stem(record);
            lock (Gate)
            {
                if (!PlannedStems.Add(stem) || ExistsForStem(stem))
                {
                    return;
                }
            }

            Record($"{stem}-plan", planStem: null, record);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[GPU][WARN] Shader corpus record failed: {exception.GetType().Name}: {exception.Message}");
        }
    }

    // Taken before the translation of one permutation: everything its compile request is made of.
    internal static void RecordCompile(ShaderSource source, StageCompileOptions options, ShaderCodeCapture code, ShaderCorpusHostFlags host,
        ResourceSpecialization specialization, uint pushDataCursor, int[]? deviceBindingMemory = null, int[]? deviceDataMemory = null)
    {
        try
        {
            var record = Create(source, options, code, host, specialization, pushDataCursor, deviceBindingMemory, deviceDataMemory);
            var stem = Stem(record);
            var variant = XxHash3.HashToUInt64(JsonSerializer.SerializeToUtf8Bytes(record with { UserData = new uint[source.UserData.Length] }, JsonOptions));
            Record($"{stem}-{variant:X16}", stem, record);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[GPU][WARN] Shader corpus record failed: {exception.GetType().Name}: {exception.Message}");
        }
    }

    public static ShaderCorpusRecord Load(string path)
    {
        var content = File.ReadAllBytes(path);
        if (content.Length < 3 * sizeof(uint) ||
            BitConverter.ToUInt32(content, 0) != Magic ||
            BitConverter.ToUInt32(content, sizeof(uint)) != FormatVersion)
        {
            throw new InvalidDataException($"{path} is not a version {FormatVersion} shader corpus record.");
        }

        var codeLength = BitConverter.ToInt32(content, 2 * sizeof(uint));
        var codeStart = 3 * sizeof(uint);
        if (codeLength < 0 || codeLength > content.Length - codeStart)
        {
            throw new InvalidDataException($"{path} is truncated.");
        }

        var code = ShaderPrewarmList.DeserializeCode(content.AsSpan(codeStart, codeLength).ToArray());
        var record = JsonSerializer.Deserialize<ShaderCorpusRecord>(content.AsSpan(codeStart + codeLength), JsonOptions)
            ?? throw new InvalidDataException($"{path} has no record body.");
        return record with { Code = code };
    }

    /// <summary>Rebuilds what the translator is handed. A record without options only decodes, one without a
    /// specialization stops after the resource plan; both succeed with a null request.</summary>
    public static bool TryBuildRequest(ShaderCorpusRecord record, out ShaderCompileRequest? request, out string error) =>
        ShaderProgramCache.TryBuildCorpusRequest(record, out request, out error);

    internal static ShaderCorpusRecord Create(ShaderSource source, StageCompileOptions options, ShaderCodeCapture code,
        ShaderCorpusHostFlags host, ResourceSpecialization? specialization, uint pushDataCursor,
        int[]? deviceBindingMemory = null, int[]? deviceDataMemory = null) =>
        new()
        {
            Code = code,
            Stage = source.Stage,
            Hash = source.Hash,
            Registered = source.Registered,
            UserData = source.UserData,
            UserDataBase = source.UserDataBase,
            ExcludedUserDataRegisters = source.ExcludedUserDataRegisters,
            Options = options,
            Specialization = specialization,
            DeviceBindingMemory = deviceBindingMemory,
            DeviceDataMemory = deviceDataMemory,
            PushDataCursor = pushDataCursor,
            Host = host,
        };

    // The program and the inputs of its stage, without the specialization (and the binding loads
    // of its plan) and without the user-data values the translation never reads (only their count).
    private static string Stem(ShaderCorpusRecord record)
    {
        var keyed = record with
        {
            Specialization = null, DeviceBindingMemory = null, DeviceDataMemory = null, PushDataCursor = 0, UserData = new uint[record.UserData.Length],
        };
        var key = XxHash3.HashToUInt64(JsonSerializer.SerializeToUtf8Bytes(keyed, JsonOptions));
        return $"{LabelOf(record.Stage)}-{record.Hash:X16}-{key:X16}";
    }

    // The record body as the corpus files and the graphics prewarm list keep it, without the guest code.
    internal static byte[] SerializeBody(ShaderCorpusRecord record) => JsonSerializer.SerializeToUtf8Bytes(record, JsonOptions);

    internal static ShaderCorpusRecord DeserializeBody(ReadOnlySpan<byte> body) =>
        JsonSerializer.Deserialize<ShaderCorpusRecord>(body, JsonOptions) ?? throw new InvalidDataException("The record body is empty.");

    private static string LabelOf(ShaderStage stage) => stage switch
    {
        ShaderStage.Vertex => "vertex",
        ShaderStage.Pixel => "pixel",
        _ => "compute",
    };

    private static bool ExistsForStem(string stem)
    {
        if (!TryPrepareDirectory())
        {
            return true;
        }

        try
        {
            return System.IO.Directory.EnumerateFiles(CorpusDirectory!, stem + "-*" + FileExtension).Any();
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return true;
        }
    }

    private static void Record(string name, string? planStem, ShaderCorpusRecord record)
    {
        try
        {
            lock (Gate)
            {
                if (!Written.Add(name) || !TryPrepareDirectory())
                {
                    return;
                }
            }

            var path = Path.Combine(CorpusDirectory!, name + FileExtension);
            if (File.Exists(path))
            {
                return;
            }

            var code = ShaderPrewarmList.SerializeCode(record.Code);
            var body = JsonSerializer.SerializeToUtf8Bytes(record, JsonOptions);
            var bytes = new byte[3 * sizeof(uint) + code.Length + body.Length];
            BitConverter.TryWriteBytes(bytes.AsSpan(0, sizeof(uint)), Magic);
            BitConverter.TryWriteBytes(bytes.AsSpan(sizeof(uint), sizeof(uint)), FormatVersion);
            BitConverter.TryWriteBytes(bytes.AsSpan(2 * sizeof(uint), sizeof(uint)), code.Length);
            code.CopyTo(bytes, 3 * sizeof(uint));
            body.CopyTo(bytes, 3 * sizeof(uint) + code.Length);

            // A run that dies on the shader being recorded must not leave half a file.
            var temporary = path + ".tmp";
            File.WriteAllBytes(temporary, bytes);
            File.Move(temporary, path, overwrite: true);

            // The full record replays everything the plan-only one did.
            if (planStem is not null)
            {
                File.Delete(Path.Combine(CorpusDirectory!, $"{planStem}-plan{FileExtension}"));
            }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[GPU][WARN] Shader corpus write failed for {name}: {exception.GetType().Name}: {exception.Message}");
        }
    }

    private static bool TryPrepareDirectory()
    {
        lock (Gate)
        {
            if (_directoryReady || _directoryFailed)
            {
                return _directoryReady;
            }

            try
            {
                System.IO.Directory.CreateDirectory(CorpusDirectory!);
                _directoryReady = true;
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                _directoryFailed = true;
                Console.Error.WriteLine($"[GPU][WARN] Shader corpus directory unavailable: {exception.Message}");
            }

            return _directoryReady;
        }
    }

    private static JsonSerializerOptions CreateJsonOptions()
    {
        var resolver = new DefaultJsonTypeInfoResolver();
        resolver.Modifiers.Add(static info =>
        {
            // The resources a stage was bound with are runtime state, not a translation input.
            if (info.Type == typeof(VertexInputInfo) || info.Type == typeof(PixelInputInfo) || info.Type == typeof(ComputeInputInfo))
            {
                for (var index = info.Properties.Count - 1; index >= 0; index--)
                {
                    if (info.Properties[index].Name == nameof(VertexInputInfo.Stage))
                    {
                        info.Properties.RemoveAt(index);
                    }
                }
            }
        });
        var options = new JsonSerializerOptions { TypeInfoResolver = resolver };
        options.Converters.Add(new JsonStringEnumConverter());
        options.Converters.Add(new ColorComponentMappingConverter());
        return options;
    }

    private sealed class ColorComponentMappingConverter : JsonConverter<Gen5ColorComponentMapping>
    {
        public override Gen5ColorComponentMapping Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
            new(reader.GetByte());

        public override void Write(Utf8JsonWriter writer, Gen5ColorComponentMapping value, JsonSerializerOptions options) =>
            writer.WriteNumberValue(value.Packed);
    }
}
