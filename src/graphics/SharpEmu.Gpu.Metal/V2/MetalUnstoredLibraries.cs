// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Stores the libraries a session compiled from MSL that no pipeline stored. A library file
/// is written when a pipeline is made from it (MetalBinaryArchive.NoteRenderPipeline,
/// CaptureCompute), but the prewarms compile every program past sessions listed, and the
/// runtime every module the game creates, drawn or not: those never made a pipeline, so they
/// missed the archive and compiled from MSL again at every start (a warm second start of
/// The Oregon Trail: 22 such, all from the library list; Silent Hill: The Short Message:
/// 166 from the list and 34 the game created and did not draw). Each is stored as the static
/// precompile stores a guess (MetalStaticPrecompile.Store) on one Utility thread, once the
/// guest runs and <see cref="Delay"/> after it compiled, unless a pipeline stored it first.
/// SHARPEMU_METAL_V2_STORE_UNSTORED=0 turns it off.
/// </summary>
internal static class MetalUnstoredLibraries
{
    internal const string Variable = "SHARPEMU_METAL_V2_STORE_UNSTORED";

    /// <summary>How long a library waits for a pipeline to store it; settable for tests.</summary>
    internal static TimeSpan Delay { get; set; } = TimeSpan.FromSeconds(20);

    // Sources wait in memory until stored; past this many the rest wait for a later start.
    private const int MaxPending = 4096;

    private static readonly Regex EntryPoint = new(@"\b(gen5_(?:vs|ps|cs|ms))\s*\(", RegexOptions.CultureInvariant);
    private static readonly string MipStatisticsDeclaration =
        Gen5MslShader.MipStatisticsFunctionConstantName + " [[function_constant(";

    private static readonly ConcurrentDictionary<UInt128, Entry> _pending = new();
    private static readonly object _gate = new();
    // Set at Stop, so the thread's wait between passes ends at once and shutdown, where the
    // archive closes in the last moments of the process, is not held up.
    private static readonly ManualResetEventSlim _stop = new(false);
    private static MetalCompileThread? _thread;
    private static volatile bool _stopping;
    private static int _noted;
    private static int _stored;
    private static int _storedByPipeline;
    private static int _notStorable;
    private static int _failed;
    private static int _reported;

    public static bool Enabled { get; } = Environment.GetEnvironmentVariable(Variable) != "0";

    private sealed record Entry(MetalBinaryArchive Archive, nint Device, string Source, bool UsesPerVertexInputs, long Ticks);

    /// <summary>A library just compiled from <paramref name="source"/> after missing the archive. Any thread.</summary>
    public static void Note(MetalBinaryArchive archive, nint device, UInt128 key, string source, bool usesPerVertexInputs)
    {
        if (!Enabled || _stopping || key == default || _pending.Count >= MaxPending ||
            !_pending.TryAdd(key, new Entry(archive, device, source, usesPerVertexInputs, Stopwatch.GetTimestamp())))
        {
            return;
        }

        Interlocked.Increment(ref _noted);
        if (Volatile.Read(ref _thread) is not null)
        {
            return;
        }

        lock (_gate)
        {
            if (_thread is null && !_stopping)
            {
                _thread = MetalCompileThread.Start(
                    "SharpEmu unstored library store",
                    SharpEmu.HLE.Host.HostThreadQos.Utility,
                    Run);
            }
        }
    }

    /// <summary>The entry point a translated source declares, or null when it declares none or several.</summary>
    internal static string? EntryPointOf(string source)
    {
        string? found = null;
        foreach (Match match in EntryPoint.Matches(source))
        {
            var name = match.Groups[1].Value;
            if (found is not null && found != name)
            {
                return null;
            }

            found = name;
        }

        return found;
    }

    /// <summary>Whether a translated source declares the LOD statistics function constant.</summary>
    internal static bool UsesMipStatisticsConstant(string source) =>
        source.Contains(MipStatisticsDeclaration, StringComparison.Ordinal);

    /// <summary>Stops the thread (what is left waits for a later start) and logs what it did.</summary>
    public static void Stop()
    {
        _stopping = true;
        _stop.Set();
        MetalCompileThread? thread;
        lock (_gate)
        {
            thread = _thread;
        }

        if (thread is not null && !thread.IsCurrentThread)
        {
            thread.Join(TimeSpan.FromSeconds(10));
        }

        if (Volatile.Read(ref _noted) != 0 && Interlocked.Exchange(ref _reported, 1) == 0)
        {
            Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"[METAL-V2][ARCHIVE] unstored libraries: compiled_from_msl={Volatile.Read(ref _noted)} " +
                $"stored={Volatile.Read(ref _stored)} stored_by_pipeline={Volatile.Read(ref _storedByPipeline)} " +
                $"not_storable={Volatile.Read(ref _notStorable)} failed={Volatile.Read(ref _failed)} left={_pending.Count}"));
        }

        _pending.Clear();
    }

    private static void Run()
    {
        while (!_stopping)
        {
            _stop.Wait(1000);
            if (!_stopping && !MetalShaderPrewarm.Gate.IsHolding)
            {
                Drain(Delay);
            }
        }
    }

    /// <summary>
    /// Stores the libraries noted at least <paramref name="delay"/> ago that no pipeline
    /// stored meanwhile, on the caller's thread; the number written.
    /// </summary>
    internal static int Drain(TimeSpan delay)
    {
        var written = 0;
        foreach (var (key, entry) in _pending)
        {
            if (_stopping)
            {
                break;
            }

            if (Stopwatch.GetElapsedTime(entry.Ticks) >= delay && _pending.TryRemove(key, out _) && Store(key, entry))
            {
                written++;
            }
        }

        return written;
    }

    private static bool Store(UInt128 key, Entry entry)
    {
        if (entry.Archive.HasLibrary(key))
        {
            Interlocked.Increment(ref _storedByPipeline);
            return false;
        }

        var entryPoint = EntryPointOf(entry.Source);
        if (entryPoint is null)
        {
            Interlocked.Increment(ref _notStorable);
            return false;
        }

        try
        {
            // Waits while the game compiles, as the static precompile does.
            using var slot = MetalCompileScheduler.Shared.EnterPrewarm(static () => _stopping);
            if (!slot.Entered)
            {
                return false;
            }

            var bytes = MetalStaticPrecompile.Store(
                entry.Archive,
                entry.Device,
                entry.Source,
                entryPoint,
                entry.UsesPerVertexInputs,
                UsesMipStatisticsConstant(entry.Source),
                key,
                out var error);
            if (bytes > 0)
            {
                Interlocked.Increment(ref _stored);
                return true;
            }
            else if (error is null)
            {
                // A vertex function fetching through a vertex descriptor, or stored meanwhile.
                Interlocked.Increment(ref _notStorable);
            }
            else
            {
                Interlocked.Increment(ref _failed);
                MetalV2Diagnostics.Trace($"unstored library 0x{key:X32} not stored: {error}");
            }
        }
        catch (Exception exception) when (!_stopping)
        {
            Interlocked.Increment(ref _failed);
            MetalV2Diagnostics.Trace($"unstored library 0x{key:X32} not stored: {exception.Message}");
        }

        return false;
    }
}
