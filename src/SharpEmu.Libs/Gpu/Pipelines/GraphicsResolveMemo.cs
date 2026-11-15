// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Hashing;
using System.Runtime.InteropServices;
using SharpEmu.HLE.Diagnostics;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Libs.Gpu.Pipelines;

// SHARPEMU_RESOLVE_MEMO: on (default), 0 (the kept programs of the last draw only, as before), shadow (counts what the
// memo would hit and keeps the kept path), verify (every hit also resolves in full and stops on any difference).
internal enum ResolveMemoMode
{
    Off,
    Shadow,
    On,
    Verify,
}

// Why a draw found no usable memo entry; each falls through to the full resolution.
internal enum ResolveMemoMiss
{
    // No entry has the draw's key.
    New,
    // The program cache replaced or refused an entry since (ShaderProgramCache.ContentEpoch), or the vertex table
    // metadata was dropped.
    Epoch,
    // The code address names another header record.
    Header,
    // The code's identity changed.
    Identity,
    // The vertex attribute or buffer table words changed.
    Tables,
    // The pixel input semantics changed.
    Semantics,
    // A stage's descriptor words no longer evaluate to its snapshot.
    Words,
    // Device data loads were learned (DeviceDataLoads.Generation).
    Generation,
    Count,
}

// Why a draw's resolution is never kept in the memo.
internal enum ResolveMemoUncacheable
{
    Frontend,
    Mesh,
    Tessellation,
    Calls,
    Unavailable,
    Tables,
    Trace,
    // A snapshot with no origin to revalidate it from.
    Origin,
    // A user data count over the bank (the full resolution stops on it).
    Key,
    Count,
}

// The graphics programs of draws already resolved, keyed by everything their resolution reads from the registers and
// the draw (ResolveKey), many per frame: a draw whose key matches an entry takes its programs once every guest read the
// full resolution makes still gives the same answer (ShaderPipelineCache.TryTakeResolveMemo). Used under the pipeline
// cache's _resolveGate only.
internal sealed class GraphicsResolveMemo
{
    internal static readonly ResolveMemoMode DefaultMode = ParseMode(Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_MEMO"));

    // SHARPEMU_RESOLVE_MEMO_ENTRIES: at least twice a frame's draws (Demon's Souls ~7300).
    internal static readonly int DefaultCapacity =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_MEMO_ENTRIES"), out var entries) && entries >= 2 ? entries : 16384;

    // The memo only counts what it would hit for now (shadow); 0 turns that off too.
    internal static ResolveMemoMode ParseMode(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        "0" or "off" => ResolveMemoMode.Off,
        _ => ResolveMemoMode.Shadow,
    };

    // One way of a set: the key it was resolved under and what the hit re-checks.
    internal sealed class Entry
    {
        public required byte[] Key;
        public required ulong Hash;
        public long Used;
        public GraphicsPrograms? Programs;
        public RegisteredShader? VertexShader;
        public ulong VertexAddress;
        public ulong VertexHash;
        public RegisteredShader? PixelShader;
        public ulong PixelAddress;
        public ulong PixelHash;
        public VertexTableWords? TableWords;
        public ulong AttributeTable;
        public ulong BufferTable;
        public uint PixelSemanticCount;
        public uint PixelCustomMask;
        public long ContentEpoch;
        public int DeviceDataGeneration;
        public long TablesGeneration;
        // GraphicsProgramsComparer.Fingerprint at insertion (verify: nothing wrote the shared result since).
        public ulong Fingerprint;
    }

    private const int MaxKeyBytes = 4096;
    private readonly byte[] _key = new byte[MaxKeyBytes];
    private int _keyLength;
    private int _userDataStart;
    private readonly Entry?[] _ways;
    private readonly int _setMask;
    private long _clock;
    private int _count;
    // The register and user-data halves of the keys stored, to tell a miss's cause (key_miss); bounded.
    private readonly HashSet<ulong> _registerHalves = [];
    private readonly HashSet<ulong> _userDataHalves = [];

    public GraphicsResolveMemo(ResolveMemoMode mode, int capacity)
    {
        Mode = mode;
        var sets = 1;
        while (sets * 2 < capacity) sets <<= 1;
        _setMask = sets - 1;
        _ways = new Entry?[sets * 2];
    }

    public ResolveMemoMode Mode { get; }

    public int Count => _count;

    public ulong Hash { get; private set; }

    public ReadOnlySpan<byte> Key => _key.AsSpan(0, _keyLength);

    // The draw's key: every register and draw input the resolution reads, the user data words last. False when the user
    // data count is over the bank (the full resolution stops there).
    public bool Fill(RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags,
        PixelOutputBlendBits blendBits)
    {
        var key = _key.AsSpan();
        var at = 0;
        var vertex = banks.Shader.Vertex;
        var pixel = banks.Shader.Pixel;
        var context = banks.Context;
        var shaderInterface = context.ShaderInterface;

        Put(key, ref at, flags);
        Put(key, ref at, targetExportMapping.Length);
        PutSpan(key, ref at, targetExportMapping);

        // The vertex stage: every plain register (the hull, legacy and export user scalars are not read).
        Put(key, ref at, vertex.ExportAddress);
        Put(key, ref at, vertex.LocalAddress);
        Put(key, ref at, vertex.HullAddress);
        Put(key, ref at, vertex.HullUserDataAddress);
        Put(key, ref at, vertex.HullResource1);
        Put(key, ref at, vertex.HullResource2);
        Put(key, ref at, vertex.GeometryAddress);
        Put(key, ref at, vertex.GeometryUserDataAddress);
        Put(key, ref at, vertex.GeometryResource1);
        Put(key, ref at, vertex.GeometryResource2);
        Put(key, ref at, vertex.ExportResource1);
        Put(key, ref at, vertex.ExportResource2);
        Put(key, ref at, vertex.LegacyVertexAddress);
        Put(key, ref at, vertex.LegacyVertexResource1);
        Put(key, ref at, vertex.LegacyVertexResource2);
        Put(key, ref at, vertex.GeometryUserScalars.Count);
        if (flags.PixelActive)
        {
            Put(key, ref at, pixel.Address);
            Put(key, ref at, pixel.UserDataAddress);
            Put(key, ref at, pixel.AuxiliaryTableAddress);
            Put(key, ref at, pixel.Resource1);
            Put(key, ref at, pixel.Resource2);
            Put(key, ref at, pixel.UserScalars.Count);
        }

        // The context registers the resolution reads (ShaderPipelineCache, HostViewportTransform, ColorTargetResolver,
        // PixelExportRouting): the blend state as the bits it resolves from (PixelOutputBlend), the viewport's transform.
        Put(key, ref at, context.ShaderStages);
        Put(key, ref at, context.RenderTargetMask);
        PutSpan(key, ref at, context.ColorTargets.AsSpan());
        Put(key, ref at, blendBits.Value);
        Put(key, ref at, context.Clip);
        Put(key, ref at, context.DepthRenderOverride);
        Put(key, ref at, context.DepthTarget.ZInfo);
        Put(key, ref at, context.ScreenViewport.TransformControl);
        ref readonly var viewport = ref context.ScreenViewport.Primary;
        Put(key, ref at, viewport.MinDepth);
        Put(key, ref at, viewport.MaxDepth);
        Put(key, ref at, viewport.XScale);
        Put(key, ref at, viewport.XOffset);
        Put(key, ref at, viewport.YScale);
        Put(key, ref at, viewport.YOffset);
        Put(key, ref at, viewport.ZScale);
        Put(key, ref at, viewport.ZOffset);
        PutSpan(key, ref at, context.UserClipPlanes.AsSpan());

        // The whole shader interface.
        Put(key, ref at, shaderInterface.VertexOutputConfiguration);
        Put(key, ref at, shaderInterface.PositionExportFormat);
        Put(key, ref at, shaderInterface.VertexOutputControl);
        PutSpan(key, ref at, shaderInterface.PixelInterpolatorSettings.AsSpan());
        Put(key, ref at, shaderInterface.PixelInterpolatorWritten);
        Put(key, ref at, shaderInterface.IndexExportFormat);
        Put(key, ref at, shaderInterface.PrimitiveShaderSubgroupControl);
        Put(key, ref at, shaderInterface.GeometryInstanceCount);
        Put(key, ref at, shaderInterface.GeometryOnChipControl);
        Put(key, ref at, shaderInterface.MaxTessellationLevel);
        Put(key, ref at, shaderInterface.MinTessellationLevel);
        Put(key, ref at, shaderInterface.MaxOutputPerSubgroup);
        Put(key, ref at, shaderInterface.ExportRingItemSize);
        Put(key, ref at, shaderInterface.GeometryMaxVerticesOut);
        Put(key, ref at, shaderInterface.GeometryOutputPrimitiveType);
        Put(key, ref at, shaderInterface.PrimitiveIdEnable);
        Put(key, ref at, shaderInterface.VertexReuseOff);
        Put(key, ref at, shaderInterface.TessellationDistribution);
        Put(key, ref at, shaderInterface.LocalHullConfiguration);
        Put(key, ref at, shaderInterface.TessellationFactorParameter);
        Put(key, ref at, shaderInterface.DepthExportFormat);
        PutSpan(key, ref at, shaderInterface.TargetOutputModes.AsSpan());
        Put(key, ref at, shaderInterface.PixelInputEnable);
        Put(key, ref at, shaderInterface.PixelInputAddress);
        Put(key, ref at, shaderInterface.PixelInputControl);
        Put(key, ref at, shaderInterface.BarycentricControl);
        Put(key, ref at, shaderInterface.ColorShaderMask);
        Put(key, ref at, shaderInterface.ScanShaderControl);
        Put(key, ref at, shaderInterface.DepthShaderControl);

        // The user data words each stage resolves with (ShaderPipelineCache.UserData): its declared count, else the
        // registers the guest wrote.
        _userDataStart = at;
        if (!PutUserData(key, ref at, vertex.GeometryUserScalars, vertex.GeometryResource2.UserScalarCount) ||
            (flags.PixelActive && !PutUserData(key, ref at, pixel.UserScalars, pixel.Resource2.UserScalarCount)))
        {
            return false;
        }

        _keyLength = at;
        Hash = XxHash3.HashToUInt64(key[..at]);
        return true;
    }

    private static bool PutUserData(Span<byte> key, ref int at, UserScalarRegisters registers, uint declared)
    {
        var count = declared != 0 ? declared : registers.Count;
        if (count > UserScalarRegisters.Capacity)
        {
            return false;
        }

        Put(key, ref at, count);
        PutSpan(key, ref at, registers.ValueSpan((int)count));
        return true;
    }

    private static void Put<T>(Span<byte> key, ref int at, in T value) where T : unmanaged
    {
        MemoryMarshal.Write(key[at..], in value);
        at += System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
    }

    private static void PutSpan<T>(Span<byte> key, ref int at, ReadOnlySpan<T> values) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(values);
        bytes.CopyTo(key[at..]);
        at += bytes.Length;
    }

    // The entry with the filled key, or null.
    public Entry? Find()
    {
        var set = (int)(Hash & (ulong)_setMask) * 2;
        var key = Key;
        for (var way = set; way < set + 2; way++)
        {
            if (_ways[way] is { } entry && entry.Hash == Hash && key.SequenceEqual(entry.Key))
            {
                entry.Used = ++_clock;
                return entry;
            }
        }

        return null;
    }

    // Why the filled key has no entry: only its user data or only its registers differ from a key stored.
    public ResolveMemoKeyMiss ClassifyMiss()
    {
        var key = Key;
        var registers = XxHash3.HashToUInt64(key[.._userDataStart]);
        var userData = XxHash3.HashToUInt64(key[_userDataStart..]);
        return _registerHalves.Contains(registers) ? ResolveMemoKeyMiss.UserDataOnly
            : _userDataHalves.Contains(userData) ? ResolveMemoKeyMiss.RegistersOnly
            : ResolveMemoKeyMiss.New;
    }

    // A new entry for the filled key, in place of the set's older way; the caller fills it.
    public Entry Store()
    {
        var key = Key;
        var set = (int)(Hash & (ulong)_setMask) * 2;
        var way = set;
        for (var candidate = set; candidate < set + 2; candidate++)
        {
            if (_ways[candidate] is not { } held)
            {
                way = candidate;
                break;
            }

            if (held.Hash == Hash && key.SequenceEqual(held.Key))
            {
                way = candidate;
                break;
            }

            if (held.Used < (_ways[way]?.Used ?? long.MinValue))
            {
                way = candidate;
            }
        }

        if (_ways[way] is { } replaced)
        {
            if (replaced.Hash != Hash || !key.SequenceEqual(replaced.Key))
            {
                _evictions.Increment();
            }
        }
        else
        {
            _count++;
            _entries.Increment();
        }

        if (_registerHalves.Count >= _ways.Length * 2)
        {
            _registerHalves.Clear();
            _userDataHalves.Clear();
        }

        _registerHalves.Add(XxHash3.HashToUInt64(key[.._userDataStart]));
        _userDataHalves.Add(XxHash3.HashToUInt64(key[_userDataStart..]));
        var entry = new Entry { Key = key.ToArray(), Hash = Hash, Used = ++_clock };
        _ways[way] = entry;
        return entry;
    }

    private static readonly ThreadCounter _hits = new();
    private static readonly ThreadCounter _evictions = new();
    private static readonly ThreadCounter _entries = new();
    private static readonly ThreadCounter _verifyChecked = new();
    private static readonly ThreadCounter _shadowWouldHit = new();
    private static readonly ThreadCounter[] _misses = NewCounters((int)ResolveMemoMiss.Count);
    private static readonly ThreadCounter[] _keyMisses = NewCounters((int)ResolveMemoKeyMiss.Count);
    private static readonly ThreadCounter[] _uncacheable = NewCounters((int)ResolveMemoUncacheable.Count);
    private static long _verifyFailed;

    private static ThreadCounter[] NewCounters(int count) => Enumerable.Range(0, count).Select(_ => new ThreadCounter()).ToArray();

    public static void NoteHit() => _hits.Increment();

    public static void NoteShadowHit() => _shadowWouldHit.Increment();

    public static void NoteMiss(ResolveMemoMiss cause) => _misses[(int)cause].Increment();

    public static void NoteKeyMiss(ResolveMemoKeyMiss cause) => _keyMisses[(int)cause].Increment();

    public static void NoteUncacheable(ResolveMemoUncacheable cause) => _uncacheable[(int)cause].Increment();

    public static void NoteVerified() => _verifyChecked.Increment();

    public static void NoteVerifyFailed() => Interlocked.Increment(ref _verifyFailed);

    private static string Take(ThreadCounter[] counters, params string[] names) =>
        string.Join(",", names.Select((name, index) => FormattableString.Invariant($"{name}={counters[index].Take()}")));

    // The resolve_memo=(...) part of [PERF][KEPT_PROGRAMS]; entries counts every live memo.
    public static string TakeReport() => FormattableString.Invariant(
        $"resolve_memo=(mode={DefaultMode.ToString().ToLowerInvariant()},hits={_hits.Take()},would_hit={_shadowWouldHit.Take()},") +
        "miss_by=(" + Take(_misses, "new", "epoch", "header", "identity", "tables", "semantics", "words", "generation") + ")," +
        "key_miss=(" + Take(_keyMisses, "new", "user_data_only", "regs_only") + ")," +
        "uncacheable_by=(" + Take(_uncacheable, "frontend", "mesh", "tess", "calls", "unavailable", "tables", "trace", "origin", "key") + ")," +
        FormattableString.Invariant($"evictions={_evictions.Take()},entries={_entries.Total},verify_checked={_verifyChecked.Take()},") +
        FormattableString.Invariant($"verify_failed={Interlocked.Read(ref _verifyFailed)})");
}

internal enum ResolveMemoKeyMiss
{
    New,
    UserDataOnly,
    RegistersOnly,
    Count,
}

// ShaderPipelineCache.PixelOutputBlend of a draw's context: the blend state its resolution reads.
internal readonly record struct PixelOutputBlendBits(uint Value);
