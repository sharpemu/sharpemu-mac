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
    // A program entry, the device data loads or the vertex table metadata changed while the draw resolved, its vertex
    // source is not the draw's, or another draw's key was filled while it resolved.
    Changed,
    Count,
}

// Why a words miss was not rebuilt from its revalidation (ShaderPipelineCache.TryTakeResolveMemo) and resolved in full.
internal enum ResolveMemoRebuildRefusal
{
    // A snapshot of the materialization cache, which is revalidated only.
    Origin,
    // A reader that tells table words apart, or no program to evaluate.
    Refused,
    // The program could not evaluate the words.
    Evaluate,
    // The materialization from the evaluated words failed.
    Failed,
    // The words specialize the program otherwise.
    Specialization,
    // Device data loads were learned while the words were read.
    Generation,
    Count,
}

// The graphics programs of draws already resolved, keyed by everything their resolution reads from the registers and
// the draw (ResolveKey), many per frame: a draw whose key matches an entry takes its programs once every guest read the
// full resolution makes still gives the same answer (ShaderPipelineCache.TryTakeResolveMemo). Used under its resolver
// shard's gate only (one memo per shard; with one shard that gate is the pipeline cache's _resolveGate).
internal sealed class GraphicsResolveMemo
{
    internal static readonly ResolveMemoMode DefaultMode = ParseMode(Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_MEMO"));

    // SHARPEMU_RESOLVE_MEMO_ENTRIES: at least twice a frame's draws (Demon's Souls ~7300).
    internal static readonly int DefaultCapacity =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_MEMO_ENTRIES"), out var entries) && entries >= 2 ? entries : 16384;

    // SHARPEMU_RESOLVE_MEMO_REBUILD=0: an entry whose descriptor words changed resolves in full again instead of taking the
    // snapshot its revalidation evaluated (ShaderPipelineCache.TryTakeResolveMemo).
    internal static readonly bool DefaultRebuild = Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_MEMO_REBUILD") != "0";

    // SHARPEMU_RESOLVE_MEMO_HOLD_CHURN=0: every rebuilt words miss replaces its entry's programs, as before. On (default), a
    // rebuild right after another rebuild of the same entry is kept only when it evaluated the same words as that one
    // (Entry.RebuildSignature): a draw whose descriptor words move on every take (Demon's Souls: ring-allocated constant
    // buffers, ~1/3 of its draws) is handed its rebuilt programs without the long-lived entry pointing at them, so they die
    // young instead of being promoted by every gen0 collection (~56 MB each in game) and compacted. Its next take
    // revalidates the programs the entry kept and rebuilds again, the work a kept rebuild would have cost it (its words
    // moved again); words that settle are kept on their second rebuild and hit from then on, and a change after a hit is
    // kept at once as before (words that change every other take keep their hits). Only what is kept changes, never a
    // result.
    internal static readonly bool DefaultHoldChurn = Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_MEMO_HOLD_CHURN") != "0";

    // SHARPEMU_RESOLVE_MEMO_ADMIT_SEEN=0: every full resolution under a new key is kept, as before. On (default), it is kept
    // only when its key was missed before (SeenBefore): a draw whose user data names a new ring address every frame
    // (Demon's Souls: ~6% of its draws, key_miss user_data_only ~= evictions) resolves in full every frame anyway, and
    // keeping each of those resolutions made the long-lived memo point at a new entry, key and program graph per draw
    // (promoted by every gen0 collection) while evicting entries that would have hit. A key that comes back is kept on
    // its second miss and hits from its third take. Only what is kept changes, never a result.
    internal static readonly bool DefaultAdmitSeen = Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_MEMO_ADMIT_SEEN") != "0";

    internal static ResolveMemoMode ParseMode(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        "0" or "off" => ResolveMemoMode.Off,
        "shadow" => ResolveMemoMode.Shadow,
        "verify" => ResolveMemoMode.Verify,
        _ => ResolveMemoMode.On,
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
        // The signature of the words this entry's last rebuild evaluated when it followed another rebuild (HoldChurn), else 0.
        public ulong RebuildSignature;
        // Whether the entry's last take rebuilt (kept or not) rather than hit (HoldChurn).
        public bool RebuiltLast;
    }

    private const int MaxKeyBytes = 4096;
    // Ways per set. A frame's draws come back in the same order every frame, so a set holding more of a frame's keys
    // than it has ways misses all of them every frame (LRU): at two ways ~23% of the bench's GameShaped draws (and the
    // evictions ~= miss_by new of Demon's Souls) missed so; eight ways at the same capacity leave few such sets.
    internal const int Ways = 8;
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
        while (sets * Ways < capacity) sets <<= 1;
        _setMask = sets - 1;
        _ways = new Entry?[sets * Ways];
        // Eight bits per key the filter holds before it is cleared.
        _seen = new ulong[Math.Max(1, _ways.Length * 2 * 8 / 64)];
        _seenMask = _seen.Length * 64 - 1;
    }

    // Whether the filled key was missed before (since the filter was last cleared); records it when not.
    public bool SeenBefore()
    {
        var first = (int)(Hash & (ulong)_seenMask);
        var second = (int)((Hash >> 32) & (ulong)_seenMask);
        if ((_seen[first >> 6] & (1UL << first)) != 0 && (_seen[second >> 6] & (1UL << second)) != 0)
        {
            return true;
        }

        if (++_seenAdded > _ways.Length * 2)
        {
            Array.Clear(_seen);
            _seenAdded = 1;
        }

        _seen[first >> 6] |= 1UL << first;
        _seen[second >> 6] |= 1UL << second;
        return false;
    }

    public ResolveMemoMode Mode { get; }

    // Whether a words miss is rebuilt from its revalidation (DefaultRebuild; tests set it).
    public bool Rebuild { get; set; } = DefaultRebuild;

    // Whether a rebuild that follows another one with other words is handed out without being kept (DefaultHoldChurn;
    // tests set it).
    public bool HoldChurn { get; set; } = DefaultHoldChurn;

    // Whether a new key's resolution is kept only on the key's second miss (DefaultAdmitSeen; tests set it).
    public bool AdmitSeen { get; set; } = DefaultAdmitSeen;

    // The keys missed since the filter was last cleared, as a two-probe Bloom filter over the key hash: a false positive
    // keeps a resolution on its first miss (as with AdmitSeen off), and a key is forgotten only when the filter is
    // cleared, after twice the memo's capacity of first misses (~3% false positives at that fill).
    private readonly ulong[] _seen;
    private readonly int _seenMask;
    private int _seenAdded;

    public int Count => _count;

    public ulong Hash { get; private set; }

    public ReadOnlySpan<byte> Key => _key.AsSpan(0, _keyLength);

    // How many keys were filled: a store belongs to the key filled before its resolution only while this has not moved
    // (a guest read inside a resolution can resolve another draw on the same thread, which fills its own key).
    public long FillSerial { get; private set; }

    // The draw's key: every register and draw input the resolution reads, the user data words last. False when the user
    // data count is over the bank (the full resolution stops there).
    public bool Fill(RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags,
        PixelOutputBlendBits blendBits)
    {
        FillSerial++;
        ref var key = ref MemoryMarshal.GetArrayDataReference(_key);
        var at = 0;
        var vertex = banks.Shader.Vertex;
        var pixel = banks.Shader.Pixel;
        var context = banks.Context;
        var shaderInterface = context.ShaderInterface;

        at = Put(ref key, at, flags);
        at = Put(ref key, at, targetExportMapping.Length);
        at = PutSpan(ref key, at, targetExportMapping);

        // The vertex stage: every plain register (the hull, legacy and export user scalars are not read).
        at = Put(ref key, at, vertex.ExportAddress);
        at = Put(ref key, at, vertex.LocalAddress);
        at = Put(ref key, at, vertex.HullAddress);
        at = Put(ref key, at, vertex.HullUserDataAddress);
        at = Put(ref key, at, vertex.HullResource1);
        at = Put(ref key, at, vertex.HullResource2);
        at = Put(ref key, at, vertex.GeometryAddress);
        at = Put(ref key, at, vertex.GeometryUserDataAddress);
        at = Put(ref key, at, vertex.GeometryResource1);
        at = Put(ref key, at, vertex.GeometryResource2);
        at = Put(ref key, at, vertex.ExportResource1);
        at = Put(ref key, at, vertex.ExportResource2);
        at = Put(ref key, at, vertex.LegacyVertexAddress);
        at = Put(ref key, at, vertex.LegacyVertexResource1);
        at = Put(ref key, at, vertex.LegacyVertexResource2);
        at = Put(ref key, at, vertex.GeometryUserScalars.Count);
        if (flags.PixelActive)
        {
            at = Put(ref key, at, pixel.Address);
            at = Put(ref key, at, pixel.UserDataAddress);
            at = Put(ref key, at, pixel.AuxiliaryTableAddress);
            at = Put(ref key, at, pixel.Resource1);
            at = Put(ref key, at, pixel.Resource2);
            at = Put(ref key, at, pixel.UserScalars.Count);
        }

        // The context registers the resolution reads (ShaderPipelineCache, HostViewportTransform, ColorTargetResolver,
        // PixelExportRouting): the blend state as the bits it resolves from (PixelOutputBlend), the viewport's transform.
        at = Put(ref key, at, context.ShaderStages);
        at = Put(ref key, at, context.RenderTargetMask);
        at = PutSpan(ref key, at, context.ColorTargets.AsSpan());
        at = Put(ref key, at, blendBits.Value);
        at = Put(ref key, at, context.Clip);
        at = Put(ref key, at, context.DepthRenderOverride);
        at = Put(ref key, at, context.DepthTarget.ZInfo);
        at = Put(ref key, at, context.ScreenViewport.TransformControl);
        ref readonly var viewport = ref context.ScreenViewport.Primary;
        at = Put(ref key, at, viewport.MinDepth);
        at = Put(ref key, at, viewport.MaxDepth);
        at = Put(ref key, at, viewport.XScale);
        at = Put(ref key, at, viewport.XOffset);
        at = Put(ref key, at, viewport.YScale);
        at = Put(ref key, at, viewport.YOffset);
        at = Put(ref key, at, viewport.ZScale);
        at = Put(ref key, at, viewport.ZOffset);
        at = PutSpan(ref key, at, context.UserClipPlanes.AsSpan());

        // The whole shader interface.
        at = Put(ref key, at, shaderInterface.VertexOutputConfiguration);
        at = Put(ref key, at, shaderInterface.PositionExportFormat);
        at = Put(ref key, at, shaderInterface.VertexOutputControl);
        at = PutSpan(ref key, at, shaderInterface.PixelInterpolatorSettings.AsSpan());
        at = Put(ref key, at, shaderInterface.PixelInterpolatorWritten);
        at = Put(ref key, at, shaderInterface.IndexExportFormat);
        at = Put(ref key, at, shaderInterface.PrimitiveShaderSubgroupControl);
        at = Put(ref key, at, shaderInterface.GeometryInstanceCount);
        at = Put(ref key, at, shaderInterface.GeometryOnChipControl);
        at = Put(ref key, at, shaderInterface.MaxTessellationLevel);
        at = Put(ref key, at, shaderInterface.MinTessellationLevel);
        at = Put(ref key, at, shaderInterface.MaxOutputPerSubgroup);
        at = Put(ref key, at, shaderInterface.ExportRingItemSize);
        at = Put(ref key, at, shaderInterface.GeometryMaxVerticesOut);
        at = Put(ref key, at, shaderInterface.GeometryOutputPrimitiveType);
        at = Put(ref key, at, shaderInterface.PrimitiveIdEnable);
        at = Put(ref key, at, shaderInterface.VertexReuseOff);
        at = Put(ref key, at, shaderInterface.TessellationDistribution);
        at = Put(ref key, at, shaderInterface.LocalHullConfiguration);
        at = Put(ref key, at, shaderInterface.TessellationFactorParameter);
        at = Put(ref key, at, shaderInterface.DepthExportFormat);
        at = PutSpan(ref key, at, shaderInterface.TargetOutputModes.AsSpan());
        at = Put(ref key, at, shaderInterface.PixelInputEnable);
        at = Put(ref key, at, shaderInterface.PixelInputAddress);
        at = Put(ref key, at, shaderInterface.PixelInputControl);
        at = Put(ref key, at, shaderInterface.BarycentricControl);
        at = Put(ref key, at, shaderInterface.ColorShaderMask);
        at = Put(ref key, at, shaderInterface.ScanShaderControl);
        at = Put(ref key, at, shaderInterface.DepthShaderControl);

        // The user data words each stage resolves with (ShaderPipelineCache.UserData): its declared count, else the
        // registers the guest wrote.
        _userDataStart = at;
        at = PutUserData(ref key, at, vertex.GeometryUserScalars, vertex.GeometryResource2.UserScalarCount);
        if (at >= 0 && flags.PixelActive)
        {
            at = PutUserData(ref key, at, pixel.UserScalars, pixel.Resource2.UserScalarCount);
        }

        if (at < 0)
        {
            return false;
        }

        _keyLength = at;
        Hash = XxHash3.HashToUInt64(_key.AsSpan(0, at));
        return true;
    }

    // The key's offset after the user data, or -1 when the count is over the bank. The writers take the key's first byte
    // and return the offset rather than take a span and a ref to the offset: the offset's address and a span per field
    // put every field's write through memory (the inlined spans were more locals than the JIT keeps in registers).
    // Each write still stops at the key's end.
    private static int PutUserData(ref byte key, int at, UserScalarRegisters registers, uint declared)
    {
        var count = declared != 0 ? declared : registers.Count;
        if (count > UserScalarRegisters.Capacity)
        {
            return -1;
        }

        at = Put(ref key, at, count);
        return PutSpan(ref key, at, registers.ValueSpan((int)count));
    }

    private static int Put<T>(ref byte key, int at, T value) where T : unmanaged
    {
        var size = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        if ((ulong)(uint)at + (ulong)size > MaxKeyBytes)
        {
            ThrowKeyOverflow();
        }

        System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref System.Runtime.CompilerServices.Unsafe.Add(ref key, at), value);
        return at + size;
    }

    private static int PutSpan<T>(ref byte key, int at, ReadOnlySpan<T> values) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(values);
        if ((ulong)(uint)at + (ulong)(uint)bytes.Length > MaxKeyBytes)
        {
            ThrowKeyOverflow();
        }

        bytes.CopyTo(MemoryMarshal.CreateSpan(ref System.Runtime.CompilerServices.Unsafe.Add(ref key, at), bytes.Length));
        return at + bytes.Length;
    }

    [System.Diagnostics.CodeAnalysis.DoesNotReturn]
    private static void ThrowKeyOverflow() => throw new InvalidOperationException("The resolve memo key is over its buffer.");

    // The entry with the filled key, or null.
    public Entry? Find()
    {
        var set = (int)(Hash & (ulong)_setMask) * Ways;
        var key = Key;
        for (var way = set; way < set + Ways; way++)
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

    // A new entry for the filled key, in place of the set's least recently used way; the caller fills it.
    public Entry Store()
    {
        var key = Key;
        var set = (int)(Hash & (ulong)_setMask) * Ways;
        var way = set;
        for (var candidate = set; candidate < set + Ways; candidate++)
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
    private static readonly ThreadCounter _rebuilt = new();
    private static readonly ThreadCounter _rebuildHeld = new();
    private static readonly ThreadCounter _notAdmitted = new();
    private static readonly ThreadCounter[] _rebuildRefused = NewCounters((int)ResolveMemoRebuildRefusal.Count);
    private static readonly ThreadCounter _evictions = new();
    private static readonly ThreadCounter _entries = new();
    private static readonly ThreadCounter _verifyChecked = new();
    private static readonly ThreadCounter _shadowWouldHit = new();
    private static readonly ThreadCounter[] _misses = NewCounters((int)ResolveMemoMiss.Count);
    private static readonly ThreadCounter[] _keyMisses = NewCounters((int)ResolveMemoKeyMiss.Count);
    private static readonly ThreadCounter[] _uncacheable = NewCounters((int)ResolveMemoUncacheable.Count);
    private static long _verifyFailed;
    private static readonly ThreadCounter[] _verifyRaced = NewCounters((int)ResolveMemoVerifyRace.Count);

    private static ThreadCounter[] NewCounters(int count) => Enumerable.Range(0, count).Select(_ => new ThreadCounter()).ToArray();

    public static void NoteHit() => _hits.Increment();

    public static void NoteRebuilt() => _rebuilt.Increment();

    public static void NoteRebuildHeld() => _rebuildHeld.Increment();

    public static void NoteNotAdmitted() => _notAdmitted.Increment();

    public static void NoteRebuildRefused(ResolveMemoRebuildRefusal cause) => _rebuildRefused[(int)cause].Increment();

    public static void NoteShadowHit() => _shadowWouldHit.Increment();

    public static void NoteMiss(ResolveMemoMiss cause) => _misses[(int)cause].Increment();

    public static void NoteKeyMiss(ResolveMemoKeyMiss cause) => _keyMisses[(int)cause].Increment();

    public static void NoteUncacheable(ResolveMemoUncacheable cause) => _uncacheable[(int)cause].Increment();

    public static void NoteVerified() => _verifyChecked.Increment();

    public static void NoteVerifyFailed() => Interlocked.Increment(ref _verifyFailed);

    // Tests that provoke a mismatch on purpose take it back out of the run-long count the other tests read.
    internal static void ForgetVerifyFailureForTest() => Interlocked.Decrement(ref _verifyFailed);

    public static void NoteVerifyRaced(ResolveMemoVerifyRace how) => _verifyRaced[(int)how].Increment();

    private static string Take(ThreadCounter[] counters, params string[] names) =>
        string.Join(",", names.Select((name, index) => FormattableString.Invariant($"{name}={counters[index].Take()}")));

    // The resolve_memo=(...) part of the prefetch counters (ShaderPipelineCache.TakeGraphicsResolveReport); entries counts every live memo.
    public static string TakeReport() => FormattableString.Invariant(
        $"resolve_memo=(mode={DefaultMode.ToString().ToLowerInvariant()},hits={_hits.Take()},would_hit={_shadowWouldHit.Take()},") +
        "miss_by=(" + Take(_misses, "new", "epoch", "header", "identity", "tables", "semantics", "words", "generation") + ")," +
        FormattableString.Invariant($"words_rebuilt={_rebuilt.Take()},rebuild={(DefaultRebuild ? 1 : 0)},") +
        FormattableString.Invariant($"rebuilt_not_kept={_rebuildHeld.Take()},hold_churn={(DefaultHoldChurn ? 1 : 0)},") +
        "rebuild_refused_by=(" + Take(_rebuildRefused, "origin", "refused", "evaluate", "failed", "specialization", "generation") + ")," +
        "key_miss=(" + Take(_keyMisses, "new", "user_data_only", "regs_only") + ")," +
        "uncacheable_by=(" + Take(_uncacheable, "frontend", "mesh", "tess", "calls", "unavailable", "tables", "trace", "origin", "key", "changed") + ")," +
        FormattableString.Invariant($"first_miss_not_kept={_notAdmitted.Take()},admit_seen={(DefaultAdmitSeen ? 1 : 0)},") +
        FormattableString.Invariant($"evictions={_evictions.Take()},entries={_entries.Total},verify_checked={_verifyChecked.Take()},") +
        "verify_raced=(" + Take(_verifyRaced, "changed", "came_and_went", "superseded") + ")," +
        FormattableString.Invariant($"verify_failed={Interlocked.Read(ref _verifyFailed)})");
}

// How a verify difference turned out to be a race (ShaderPipelineCache.VerifyResolveMemo): the hit's revalidation misses
// when made again (its words changed since), or it still holds and a second full resolution agrees with the hit, or the
// resolution's reads straddle a GPU write recorded since it began (superseded: the take refuses it).
internal enum ResolveMemoVerifyRace
{
    Changed,
    CameAndWent,
    // Ahead of the worker, a GPU write recorded since the resolution began covers what it read: the take refuses it.
    Superseded,
    Count,
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
