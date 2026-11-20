// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.IO.Hashing;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.ShaderCache;

public enum StaticShaderKind : byte
{
    Compute,
    Graphics,
}

/// <summary>
/// One unit of the static precompile: a compute program, or a pixel program (with the
/// geometry program it was paired with, or none) under one draw state of its signature.
/// <see cref="Id"/> is stable across starts, so a finished unit is not done again.
/// </summary>
public readonly record struct StaticShaderJob(
    ulong Id, StaticShaderKind Kind, ulong Program, ulong Partner, int State, int Uses, bool Created);

/// <summary>What a unit translates: a compute record and its code, or a pixel record and, when
/// its vertex half derived, a vertex record.</summary>
public sealed record StaticShaderBuild(
    ComputePrewarmRecord? Compute,
    ShaderCodeCapture? ComputeCode,
    ShaderCorpusRecord? Pixel,
    ShaderCorpusRecord? Vertex);

/// <summary>
/// Plans the static precompile of a title's programs the game has not drawn or dispatched
/// yet, from its shader inventory (ShaderInventory, GameShaderScanner) and the draw states
/// recorded programs were used with (the graphics prewarm list's records and the title's
/// seed). A pixel program takes the states of the recorded programs of its pixel signature
/// (StaticStageInputs.PixelSignature), the most used first and at most
/// <see cref="StatesPerSignature"/>, crossed with every geometry program it was paired with
/// (at most <see cref="VerticesPerPixel"/>); its vertex half takes the most used vertex
/// state. On Demon's Souls a state taken from another program of the same signature
/// reproduced the draw's MSL for 325 of 326 pixel records that derive at all. Backend-neutral:
/// the backend translates the builds and stores what it compiles.
/// </summary>
public sealed class StaticShaderPlanner : IDisposable
{
    public const int StatesPerSignature = 8;
    public const int VerticesPerPixel = 4;
    private const int VertexStates = 1;

    private readonly ShaderInventoryFile _inventory;
    private readonly Dictionary<ulong, List<(ShaderCorpusRecord State, int Uses, ulong Key)>> _palettes = new();
    private readonly List<(ShaderCorpusRecord State, int Uses, ulong Key)> _vertexStates = [];
    private readonly HashSet<(ulong Vertex, ulong Pixel)> _seedPairs = [];
    private readonly Dictionary<ulong, List<ulong>> _seedVerticesOf = new();
    private readonly ConcurrentDictionary<ulong, ulong> _signatures = new();
    private readonly ShaderCorpusHostFlags _host;
    private readonly object _seedGate = new();

    private StaticShaderPlanner(ShaderInventoryFile inventory, ShaderCorpusHostFlags host)
    {
        _inventory = inventory;
        _host = host;
    }

    public int Programs => _inventory.ProgramCount;

    public int Pairs => _inventory.PairCount + _seedPairs.Count;

    public int Signatures => _palettes.Count;

    public int States => _palettes.Values.Sum(static palette => palette.Count) + _vertexStates.Count;

    public int FirstVariantVotes { get; private set; }

    public int SecondVariantVotes { get; private set; }

    /// <summary>Which of the two interpolant mappings the title's draws matched; the second when
    /// nothing voted (it is the one Demon's Souls uses).</summary>
    public bool SecondInterpolantVariant => SecondVariantVotes >= FirstVariantVotes;

    /// <summary>The created programs captured so far and those dropped while nothing recorded them.</summary>
    public static (long Captured, long Dropped) CaptureCounters => (ShaderInventory.Captured, ShaderInventory.Dropped);

    /// <summary>Stops capturing created programs (the precompile is off).</summary>
    public static void DisableCapture() => ShaderInventory.Disable();

    /// <summary>The title's seed: user/shader_seeds/&lt;TITLE&gt;.seed, as upstream keeps it.</summary>
    public static string SeedPath(string userDirectory, string titleId) => ShaderSeedFile.PathFor(userDirectory, titleId);

    /// <summary>The inventory file's path, for the log.</summary>
    public string InventoryPath => _inventory.Path;

    /// <summary>
    /// Opens the inventory in <paramref name="directory"/> (pipeline_cache/&lt;TITLE&gt;) and
    /// learns the palettes from <paramref name="recorded"/> and the seed at
    /// <paramref name="seedPath"/>. Null when the inventory cannot be opened.
    /// </summary>
    /// With <paramref name="imageLimits"/>, every state predicts under that stable image budget (the current host's)
    /// instead of the one its recording session answered.
    public static StaticShaderPlanner? Open(string directory, IEnumerable<ShaderCorpusRecord> recorded, string? seedPath,
        StableImageDescriptorLimits? imageLimits = null)
    {
        if (ShaderInventoryFile.Open(directory) is not { } inventory)
        {
            return null;
        }

        var records = recorded.Where(static record => record.Options is not null).ToList();
        if (imageLimits is { } limits)
        {
            records = records
                .Select(record => record with { Host = record.Host with { ComputeImageLimits = limits, GraphicsImageLimits = limits } })
                .ToList();
        }

        var host = records.FirstOrDefault()?.Host ??
            (imageLimits is { } budget ? new ShaderCorpusHostFlags { ComputeImageLimits = budget, GraphicsImageLimits = budget }
                : new ShaderCorpusHostFlags());
        var planner = new StaticShaderPlanner(inventory, host);
        var seed = seedPath is null ? null : ShaderSeedFile.Load(seedPath);
        planner.Learn(records, seed);
        return planner;
    }

    /// <summary>
    /// Records the programs the game creates into the inventory from now on, and hands the
    /// units of each one (once it is in the inventory) to <paramref name="created"/>.
    /// </summary>
    public void AttachCapture(Action<StaticShaderJob> created) =>
        ShaderInventory.Attach(_inventory, (program, _) =>
        {
            foreach (var job in JobsOf(program, created: true))
            {
                created(job);
            }
        });

    /// <summary>
    /// Scans app0 for AGC programs into the inventory: the files that name shaders and the
    /// executables, or with <paramref name="everything"/> every file but media. Files
    /// scanned before under the same size, time and scanner version are skipped.
    /// </summary>
    public ShaderScanResult Scan(string root, bool everything, int threads, Action<long, long>? progress, CancellationToken cancellation) =>
        GameShaderScanner.Scan(root, _inventory, threads, progress, cancellation, everything ? null : GameShaderScanner.IsLikelyShaderFile);

    /// <summary>Every unit of the inventory but those in <paramref name="done"/>: programs the game
    /// created first, then by how often their state was used.</summary>
    public List<StaticShaderJob> Plan(Func<ulong, bool> done)
    {
        var jobs = new List<(StaticShaderJob Job, int Order)>();
        var order = 0;
        foreach (var program in _inventory.Programs())
        {
            foreach (var job in JobsOf(program, program.Source == InventorySource.Created))
            {
                if (!done(job.Id))
                {
                    jobs.Add((job, order++));
                }
            }
        }

        jobs.Sort(static (left, right) =>
            left.Job.Created != right.Job.Created ? (left.Job.Created ? -1 : 1)
            : left.Job.Uses != right.Job.Uses ? right.Job.Uses.CompareTo(left.Job.Uses)
            : left.Order.CompareTo(right.Order));
        return jobs.Select(static entry => entry.Job).ToList();
    }

    /// <summary>The records a unit translates; false with the reason when its programs do not derive.</summary>
    public bool TryBuild(in StaticShaderJob job, out StaticShaderBuild build, out string reason)
    {
        build = null!;
        if (!_inventory.TryRead(job.Program, out var header, out var code))
        {
            reason = "not in the inventory";
            return false;
        }

        if (job.Kind == StaticShaderKind.Compute)
        {
            if (!StaticStageInputs.TryCompute(header, code, computeWave64Supported: false, _host, out var compute, out reason))
            {
                return false;
            }

            build = new StaticShaderBuild(compute, StaticStageInputs.WithHeader(code, header), null, null);
            return true;
        }

        if (!_palettes.TryGetValue(Signature(job.Program, header), out var palette) || job.State >= palette.Count)
        {
            reason = "no draw state";
            return false;
        }

        AgcShaderHeader? vertexHeader = null;
        ShaderCodeCapture? vertexCode = null;
        if (job.Partner != 0)
        {
            if (!_inventory.TryRead(job.Partner, out var partnerHeader, out var partnerCode))
            {
                reason = "the geometry program is not in the inventory";
                return false;
            }

            vertexHeader = partnerHeader;
            vertexCode = partnerCode;
        }

        var state = palette[job.State].State;
        if (!StaticStageInputs.TryPixel(header, code, vertexHeader, SecondInterpolantVariant, state,
                state.Options!.PerVertexInputsFromPrimitive, out var pixel, out reason))
        {
            return false;
        }

        ShaderCorpusRecord? vertex = null;
        if (vertexHeader is not null && _vertexStates.Count != 0 &&
            StaticStageInputs.TryVertex(vertexHeader, vertexCode!, header, pixel, _vertexStates[0].State, out var derived, out _))
        {
            vertex = derived;
        }

        build = new StaticShaderBuild(null, null, pixel.Record, vertex);
        reason = string.Empty;
        return true;
    }

    /// <summary>The seed this title's inventory and states make, for learn mode.</summary>
    public bool TrySaveSeed(string path, string titleId, string gameVersion)
    {
        var seed = new ShaderSeedFile
        {
            TitleId = titleId,
            GameVersion = gameVersion,
            FirstVariantVotes = FirstVariantVotes,
            SecondVariantVotes = SecondVariantVotes,
        };
        foreach (var pair in _inventory.Pairs())
        {
            seed.Pairs.Add(pair);
        }

        lock (_seedGate)
        {
            seed.Pairs.UnionWith(_seedPairs);
        }

        foreach (var (signature, palette) in _palettes)
        {
            seed.States.AddRange(palette.Select(entry => new SeedState(signature, entry.Uses, StateOnly(entry.State))));
        }

        seed.States.AddRange(_vertexStates.Select(static entry => new SeedState(0, entry.Uses, StateOnly(entry.State))));
        return seed.TrySave(path);
    }

    public void Dispose()
    {
        ShaderInventory.Detach(_inventory);
        _inventory.Dispose();
    }

    private IEnumerable<StaticShaderJob> JobsOf(InventoryProgram program, bool created)
    {
        if (program.Type == AgcShaderType.Compute)
        {
            yield return new StaticShaderJob(JobId(StaticShaderKind.Compute, program.Identity, 0, 0), StaticShaderKind.Compute,
                program.Identity, 0, 0, 0, created);
            yield break;
        }

        if (program.Type != AgcShaderType.Pixel || !_inventory.TryReadHeader(program.Identity, out var header) ||
            !_palettes.TryGetValue(Signature(program.Identity, header), out var palette))
        {
            yield break;
        }

        var vertices = VerticesOf(program.Identity);
        if (vertices.Count == 0 && header.InputSemanticCount == 0)
        {
            vertices = [0];
        }

        for (var index = 0; index < palette.Count; index++)
        {
            foreach (var vertex in vertices.Take(VerticesPerPixel))
            {
                yield return new StaticShaderJob(JobId(StaticShaderKind.Graphics, program.Identity, vertex, palette[index].Key),
                    StaticShaderKind.Graphics, program.Identity, vertex, index, palette[index].Uses, created);
            }
        }
    }

    private List<ulong> VerticesOf(ulong pixel)
    {
        var vertices = _inventory.VerticesOf(pixel).ToList();
        lock (_seedGate)
        {
            if (_seedVerticesOf.TryGetValue(pixel, out var seeded))
            {
                vertices.AddRange(seeded.Where(vertex => !vertices.Contains(vertex) && _inventory.Contains(vertex)));
            }
        }

        return vertices;
    }

    private ulong Signature(ulong identity, AgcShaderHeader header) =>
        _signatures.GetOrAdd(identity, static (_, header) => StaticStageInputs.PixelSignature(header), header);

    private void Learn(List<ShaderCorpusRecord> records, ShaderSeedFile? seed)
    {
        var byCode = new Dictionary<(ulong Hash, uint CodeSize), ulong>();
        foreach (var program in _inventory.Programs())
        {
            if (program.Type == AgcShaderType.Pixel)
            {
                byCode.TryAdd((program.Hash, program.CodeSize), program.Identity);
            }
        }

        var pixelStates = new Dictionary<ulong, Dictionary<ulong, (ShaderCorpusRecord State, int Uses)>>();
        var vertexStates = new Dictionary<ulong, (ShaderCorpusRecord State, int Uses)>();
        void AddPixel(ulong signature, ShaderCorpusRecord state, int uses)
        {
            if (!pixelStates.TryGetValue(signature, out var group))
            {
                pixelStates[signature] = group = new Dictionary<ulong, (ShaderCorpusRecord, int)>();
            }

            var key = PixelStateKey(state);
            group[key] = group.TryGetValue(key, out var known) ? (known.State, known.Uses + uses) : (state, uses);
        }

        void AddVertex(ShaderCorpusRecord state, int uses)
        {
            var key = VertexStateKey(state);
            vertexStates[key] = vertexStates.TryGetValue(key, out var known) ? (known.State, known.Uses + uses) : (state, uses);
        }

        foreach (var record in records)
        {
            if (record.Stage == ShaderStage.Vertex && record.Options!.VertexInfo is not null)
            {
                AddVertex(record, 1);
                continue;
            }

            if (record.Stage != ShaderStage.Pixel || record.Options!.PixelInfo is not { } pixelInfo ||
                !byCode.TryGetValue((record.Hash, record.Code.CodeSize), out var identity) ||
                !_inventory.TryReadHeader(identity, out var header))
            {
                continue;
            }

            AddPixel(Signature(identity, header), record, 1);
            Vote(identity, header, pixelInfo);
        }

        if (seed is not null)
        {
            FirstVariantVotes += seed.FirstVariantVotes;
            SecondVariantVotes += seed.SecondVariantVotes;
            foreach (var (signature, uses, state) in seed.States)
            {
                if (state.Options is null)
                {
                    continue;
                }

                if (state.Stage == ShaderStage.Pixel && state.Options.PixelInfo is not null)
                {
                    AddPixel(signature, state, uses);
                }
                else if (state.Stage == ShaderStage.Vertex && state.Options.VertexInfo is not null)
                {
                    AddVertex(state, uses);
                }
            }

            foreach (var pair in seed.Pairs)
            {
                _seedPairs.Add(pair);
                if (!_seedVerticesOf.TryGetValue(pair.Pixel, out var vertices))
                {
                    _seedVerticesOf[pair.Pixel] = vertices = [];
                }

                vertices.Add(pair.Vertex);
            }
        }

        foreach (var (signature, group) in pixelStates)
        {
            _palettes[signature] = group
                .OrderByDescending(static entry => entry.Value.Uses)
                .ThenBy(static entry => entry.Key)
                .Take(StatesPerSignature)
                .Select(static entry => (entry.Value.State, entry.Value.Uses, entry.Key))
                .ToList();
        }

        _vertexStates.AddRange(vertexStates
            .OrderByDescending(static entry => entry.Value.Uses)
            .ThenBy(static entry => entry.Key)
            .Take(VertexStates)
            .Select(static entry => (entry.Value.State, entry.Value.Uses, entry.Key)));
    }

    // Which interpolant mapping the recorded draw's SPI_PS_INPUT_CNTL values are.
    private void Vote(ulong pixel, AgcShaderHeader header, Rendering.PixelInputInfo info)
    {
        if (header.InputSemanticCount == 0 || info.InputCount == 0)
        {
            return;
        }

        foreach (var vertex in _inventory.VerticesOf(pixel))
        {
            if (!_inventory.TryReadHeader(vertex, out var vertexHeader))
            {
                continue;
            }

            var count = (int)Math.Min(info.InputCount, (uint)info.InterpolatorSettings.Length);
            var recorded = info.InterpolatorSettings.AsSpan(0, count);
            var first = AgcExports.ComputeInterpolantMapping(header.InputSemantics, vertexHeader.OutputSemantics, false);
            var second = AgcExports.ComputeInterpolantMapping(header.InputSemantics, vertexHeader.OutputSemantics, true);
            var firstMatches = first.Length >= count && recorded.SequenceEqual(first.AsSpan(0, count));
            var secondMatches = second.Length >= count && recorded.SequenceEqual(second.AsSpan(0, count));
            if (firstMatches != secondMatches)
            {
                FirstVariantVotes += firstMatches ? 1 : 0;
                SecondVariantVotes += secondMatches ? 1 : 0;
                return;
            }
        }
    }

    // What a pixel program takes from a draw: its color targets' output modes and export
    // routing, the depth and stencil exports, the per-vertex input path and the host.
    private static ulong PixelStateKey(ShaderCorpusRecord record)
    {
        var options = record.Options!;
        var info = options.PixelInfo!;
        var text = new StringBuilder()
            .AppendJoin(',', options.PixelOutputs.Select(static output => output.ToString())).Append('|')
            .Append(Convert.ToHexString(info.TargetOutputModes)).Append('|')
            .AppendJoin(',', info.TargetExportMappings.Select(static mapping => mapping.Packed)).Append('|')
            .Append(info.DepthExportEnable).Append(info.StencilExportEnable).Append(options.PerVertexInputsFromPrimitive)
            .Append('|').Append(HostKey(record.Host));
        return XxHash3.HashToUInt64(MemoryMarshal.AsBytes(text.ToString().AsSpan()));
    }

    // What a geometry program takes: the clip space and the primitive-shader mode.
    private static ulong VertexStateKey(ShaderCorpusRecord record)
    {
        var options = record.Options!;
        var primitive = options.PrimitiveShader;
        var text = new StringBuilder()
            .Append(options.VertexInfo!.ClipSpace).Append('|')
            .Append(primitive is not { } shader ? "-" : $"{shader.OutputPrimitiveType},{shader.IsWave64},{shader.Passthrough},{shader.AmplifiedVerticesOut}")
            .Append('|').Append(HostKey(record.Host));
        return XxHash3.HashToUInt64(MemoryMarshal.AsBytes(text.ToString().AsSpan()));
    }

    private static string HostKey(ShaderCorpusHostFlags host) => host.ToString();

    private static ulong JobId(StaticShaderKind kind, ulong program, ulong partner, ulong state)
    {
        Span<ulong> values = [(ulong)kind, program, partner, state];
        return XxHash3.HashToUInt64(MemoryMarshal.AsBytes(values));
    }

    // A state without the program it was recorded with: what a seed keeps.
    private static ShaderCorpusRecord StateOnly(ShaderCorpusRecord record) => record with
    {
        Specialization = null,
        Registered = record.Registered with { CodeAddress = 0, HeaderAddress = 0, UserDataAddress = 0, InputSemanticsAddress = 0 },
    };
}
