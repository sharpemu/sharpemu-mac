// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO.Compression;
using System.IO.Hashing;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Records the MSL source of every graphics library a title compiles (msl-libraries-*.bin,
/// next to the shader prewarm list) and, on the next start, compiles those sources again on
/// background threads while the game loads, keeping the libraries: the module the game
/// creates later takes its library from here instead of running the MSL front end on the
/// GPU worker. The front end costs about 130 ms a graphics library whenever macOS's
/// per-app Metal shader cache no longer holds it, which on a machine running more than one
/// title is most of the time (the cache is capped near 1 GiB for every title under the
/// bundle), and Silent Hill: The Short Message created some 200 of them in one burst after
/// its content warning, drawing nothing for 36 s while the guest waited on the GPU.
///
/// The compute prewarm holds the game while it runs; this one does not. It starts once the
/// compute prewarm is done, and a module whose source is still compiling here waits for
/// that compile rather than starting a second one. A source no start has used for
/// <see cref="MaxSessionsUnused"/> starts that used the list is dropped, which bounds the
/// list after translator changes. SHARPEMU_SHADER_PREWARM=0 turns it off with the compute
/// prewarm; SHARPEMU_METAL_V2_LIBRARY_PREWARM=0 turns off only this one.
///
/// Each translator build keeps its own list (the file name carries a hash of the compute
/// prewarm's stamp: translators, device and OS). The list holds MSL text, and a source
/// another translator wrote is one this build never asks for: after a translator change
/// gave every Demon's Souls shader new MSL, one shared list kept 1106 such sources for
/// eight more starts and compiled them all at every start, 114 s on four threads while
/// the game booted, and none was taken (hits=0). The lists of other builds are kept, the
/// most recently written first, up to <see cref="MaxOtherLists"/>.
///
/// A new translator therefore starts with no list. The graphics prewarm (GraphicsPrewarmList,
/// run by MetalShaderPrewarm) translates the recorded vertex and pixel programs again and
/// <see cref="Offer"/>s each source here, which compiles and lists it like one the list held.
/// </summary>
internal static class MetalLibraryPrewarm
{
    internal const string FilePrefix = "msl-libraries-v1";
    internal const int MaxOtherLists = 3;
    internal const int MaxSessionsUnused = 8;
    internal const int MaxEntries = 4096;

    private const uint Magic = 0x4C4C534D;
    private const uint FormatVersion = 1;
    private const int HeaderBytes = 2 * sizeof(uint);
    private const int RecordHeaderBytes = sizeof(uint) + sizeof(ulong);
    private const int MaxRecordBytes = 64 << 20;
    private const byte SessionKind = 1;
    private const byte SourceKind = 2;
    private const byte UseKind = 3;

    private const int Queued = 0;
    private const int Compiling = 1;
    private const int Ready = 2;
    private const int Claimed = 3;

    private static readonly nint SelRelease = MetalNative.Selector("release");

    internal static readonly bool Disabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LIBRARY_PREWARM") == "0";

    /// <summary>
    /// Whether a compiled library is kept for the module that asks for it. The hold-until-done
    /// and precompile-only modes compile and release instead (MetalShaderPrecompile): the
    /// guest is held meanwhile, so nothing would take them before they went idle, and the
    /// module that asks later is a hit in macOS's shader cache.
    /// </summary>
    internal static bool KeepLibraries { get; set; } = !MetalShaderPrecompile.Current.HoldsUntilComplete;

    private sealed class Slot(ulong key, bool usesPerVertexInputs, byte[] compressed)
    {
        public readonly ulong Key = key;
        public readonly bool UsesPerVertexInputs = usesPerVertexInputs;
        public byte[]? Compressed = compressed;
        public int State;
        public nint Library;
        public long ReadyTicks;
        public readonly ManualResetEventSlim Done = new(false);
    }

    /// <summary>One recorded source as the list file holds it, for the loader and its tests.</summary>
    internal readonly record struct Entry(ulong Key, bool UsesPerVertexInputs, byte[] Compressed, int LastSession, int Order);

    private static readonly object _sync = new();
    private static ConcurrentDictionary<ulong, Slot>? _slots;
    private static ConcurrentDictionary<ulong, byte>? _written;
    private static BlockingCollection<(byte Kind, ulong Key, bool UsesPerVertexInputs, string? Source)>? _writes;
    private static MetalCompileThread? _writer;
    private static MetalCompileThread[] _threads = [];
    private static List<Slot> _order = [];
    private static nint _device;
    private static volatile bool _stopping;
    private static bool _started;
    private static volatile bool _running;
    private static int _compiled;
    private static int _failed;
    private static long _compileTicks;
    private static int _hits;
    private static int _waits;
    private static long _waitTicks;
    private static int _claimed;
    private static int _misses;
    private static int _held;

    /// <summary>The prewarmed libraries held for a module that has not asked for them yet.</summary>
    public static int HeldLibraries => Volatile.Read(ref _held);

    /// <summary>
    /// Releases the prewarmed libraries no module took within the idle time of a module
    /// (ready before <paramref name="readyBefore"/>, a Stopwatch timestamp): the module that
    /// asks later compiles its source, a hit in macOS's shader cache since this process
    /// compiled it. Any thread.
    /// </summary>
    public static int ReleaseIdle(long readyBefore)
    {
        if (Volatile.Read(ref _held) == 0 || Volatile.Read(ref _slots) is not { } slots)
        {
            return 0;
        }

        var released = 0;
        foreach (var slot in slots.Values)
        {
            if (Volatile.Read(ref slot.State) != Ready || slot.ReadyTicks > readyBefore)
            {
                continue;
            }

            lock (slot)
            {
                if (slot.Library == 0)
                {
                    continue;
                }

                MetalNative.SendVoid(slot.Library, SelRelease);
                slot.Library = 0;
            }

            Interlocked.Decrement(ref _held);
            released++;
        }

        return released;
    }

    /// <summary>Whether a list is open: modules then record into it and take from it.</summary>
    public static bool Active => Volatile.Read(ref _slots) is not null;

    /// <summary>The sources this start loaded from its list, before any offered ones.</summary>
    public static int ListedSources
    {
        get
        {
            lock (_sync)
            {
                return _order.Count;
            }
        }
    }

    /// <summary>
    /// Loads the title's list and queues every source it holds, without compiling yet; the
    /// caller is the compute prewarm's start thread, which holds the game until it returns,
    /// so no module is created before the queue exists.
    /// </summary>
    public static void Open(string directory, nint device, string stamp)
    {
        if (Disabled)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, FileName(stamp));
            var (entries, sessions) = Load(path, out var rewrite);
            var kept = Select(entries, sessions);
            if (rewrite || kept.Count != entries.Count)
            {
                Rewrite(path, kept, sessions);
            }

            PruneOtherLists(directory, path);

            var slots = new ConcurrentDictionary<ulong, Slot>();
            var order = new List<Slot>(kept.Count);
            foreach (var entry in kept)
            {
                var slot = new Slot(entry.Key, entry.UsesPerVertexInputs, entry.Compressed);
                if (slots.TryAdd(entry.Key, slot))
                {
                    order.Add(slot);
                }
            }

            var writes = new BlockingCollection<(byte, ulong, bool, string?)>();
            var writer = MetalCompileThread.Create(
                "SharpEmu Metal v2 library list", SharpEmu.HLE.Host.HostThreadQos.Utility, () => WriteLoop(path, writes));
            lock (_sync)
            {
                _stopping = false;
                _started = false;
                _device = device;
                _order = order;
                _written = new ConcurrentDictionary<ulong, byte>();
                _writes = writes;
                _writer = writer;
                Volatile.Write(ref _slots, slots);
            }

            writer.Start();
            Console.Error.WriteLine(
                $"[LOADER][INFO] Library prewarm list: sources={order.Count} dropped={entries.Count - kept.Count} " +
                $"sessions={sessions}");
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Library prewarm list unavailable: {exception.Message}");
        }
    }

    /// <summary>
    /// Starts compiling the queued sources; called once the computes the guest is held for
    /// are done. Later calls do nothing.
    /// </summary>
    public static void Begin()
    {
        List<Slot> order;
        lock (_sync)
        {
            if (_started || _stopping || _slots is null)
            {
                return;
            }

            _started = true;
            order = _order;
        }

        if (order.Count == 0)
        {
            return;
        }

        var work = new ConcurrentQueue<Slot>(order);
        var workerCount = MetalShaderPrewarm.WorkerCount;
        var remaining = workerCount;
        _running = true;
        var started = Stopwatch.GetTimestamp();
        Console.Error.WriteLine($"[LOADER][INFO] Library prewarm started: libraries={order.Count} workers={workerCount}");
        var threads = new MetalCompileThread[workerCount];
        for (var index = 0; index < workerCount; index++)
        {
            // UserInitiated, as the compute prewarm's while it holds the guest: a library the
            // game takes from here waits for it, and the compiler service runs the request at
            // this thread's class.
            threads[index] = MetalCompileThread.Create($"SharpEmu library prewarm {index}", SharpEmu.HLE.Host.HostThreadQos.UserInitiated, () =>
            {
                while (!_stopping && work.TryDequeue(out var slot))
                {
                    CompileSlot(slot);
                }

                if (Interlocked.Decrement(ref remaining) == 0)
                {
                    _running = false;
                    var compiled = Volatile.Read(ref _compiled);
                    Console.Error.WriteLine(string.Create(
                        System.Globalization.CultureInfo.InvariantCulture,
                        $"[LOADER][INFO] Library prewarm done: compiled={compiled} failed={Volatile.Read(ref _failed)} " +
                        $"claimed={Volatile.Read(ref _claimed)} ms={Stopwatch.GetElapsedTime(started).TotalMilliseconds:F0} " +
                        $"library_avg_ms={(compiled == 0 ? 0 : TicksToMs(Interlocked.Read(ref _compileTicks)) / compiled):F1}"));
                }
            });
        }

        lock (_sync)
        {
            _threads = threads;
        }

        foreach (var thread in threads)
        {
            thread.Start();
        }
    }

    /// <summary>True while the queued sources are compiling.</summary>
    public static bool Running => _running;

    /// <summary>
    /// Compiles every queued source to completion on the calling thread and
    /// <see cref="MetalShaderPrewarm.WorkerCount"/> others, instead of <see cref="Begin"/>'s
    /// background threads; for the hold-until-done and precompile-only modes, which hold
    /// the guest until it returns. <paramref name="compiled"/> runs once per source. The
    /// number of sources it took; zero when the list is not open or already began.
    /// </summary>
    public static int Precompile(Action? compiled)
    {
        List<Slot> order;
        lock (_sync)
        {
            if (_started || _stopping || _slots is null)
            {
                return 0;
            }

            _started = true;
            order = _order;
        }

        if (order.Count == 0)
        {
            return 0;
        }

        var work = new ConcurrentQueue<Slot>(order);
        _running = true;
        var started = Stopwatch.GetTimestamp();
        void drain()
        {
            while (!_stopping && work.TryDequeue(out var slot))
            {
                CompileSlot(slot);
                compiled?.Invoke();
            }
        }

        var threads = Enumerable.Range(0, Math.Max(0, MetalShaderPrewarm.WorkerCount - 1))
            .Select(index => MetalCompileThread.Create(
                $"SharpEmu library precompile {index}", SharpEmu.HLE.Host.HostThreadQos.UserInitiated, drain))
            .ToArray();
        foreach (var thread in threads)
        {
            thread.Start();
        }

        drain();
        foreach (var thread in threads)
        {
            thread.Join();
        }

        _running = false;
        Console.Error.WriteLine(string.Create(
            System.Globalization.CultureInfo.InvariantCulture,
            $"[LOADER][INFO] Library precompile done: libraries={order.Count} compiled={Volatile.Read(ref _compiled)} " +
            $"failed={Volatile.Read(ref _failed)} held={Volatile.Read(ref _held)} " +
            $"ms={Stopwatch.GetElapsedTime(started).TotalMilliseconds:F0}"));
        return order.Count;
    }

    private static void CompileSlot(Slot slot)
    {
        // Waits while the game compiles (MetalCompileScheduler). The slot stays queued until
        // then, so a module that asks for this source meanwhile claims it and compiles it
        // itself instead of waiting for a compile that has not started.
        using var prewarm = MetalCompileScheduler.Shared.EnterPrewarm(static () => _stopping);
        if (!prewarm.Entered || Interlocked.CompareExchange(ref slot.State, Compiling, Queued) != Queued)
        {
            return;
        }

        var started = Stopwatch.GetTimestamp();
        try
        {
            var source = Decompress(slot.Compressed!);
            var library = MetalRenderHost.TryCompileLibrary(_device, source, slot.UsesPerVertexInputs, out var error);
            if (library == 0)
            {
                // The runtime compiles it again and reports the error where it knows the shader.
                Interlocked.Increment(ref _failed);
                MetalV2Diagnostics.Trace($"library prewarm failed key=0x{slot.Key:X16} {error}");
            }
            else
            {
                Interlocked.Increment(ref _compiled);
            }

            if (library != 0 && !KeepLibraries)
            {
                // Compiled for macOS's shader cache only; see KeepLibraries.
                MetalNative.SendVoid(library, SelRelease);
                library = 0;
            }

            slot.Library = library;
            if (library != 0)
            {
                slot.ReadyTicks = Stopwatch.GetTimestamp();
                Interlocked.Increment(ref _held);
            }
        }
        catch (Exception exception)
        {
            // Nothing here may end the process; the runtime compiles the source itself.
            Interlocked.Increment(ref _failed);
            MetalV2Diagnostics.Trace($"library prewarm failed key=0x{slot.Key:X16} {exception.Message}");
        }
        finally
        {
            Interlocked.Add(ref _compileTicks, Stopwatch.GetTimestamp() - started);
            slot.Compressed = null;
            Volatile.Write(ref slot.State, Ready);
            slot.Done.Set();
        }
    }

    /// <summary>
    /// A source the graphics prewarm translated from its own list: compiled here and kept for
    /// the module that asks for it, and written to the list so the next start with this
    /// translator has it without translating anything. True when this call compiled it; a
    /// source the list already holds, or one the game compiled itself first, is left to
    /// whoever has it. Any thread.
    /// </summary>
    public static bool Offer(string source, bool usesPerVertexInputs)
    {
        if (Volatile.Read(ref _slots) is not { } slots || _stopping)
        {
            return false;
        }

        var key = Key(source, usesPerVertexInputs);
        if (_written is { } written && written.ContainsKey(key))
        {
            return false;
        }

        var slot = new Slot(key, usesPerVertexInputs, Compress(source));
        if (!slots.TryAdd(key, slot))
        {
            return false;
        }

        try
        {
            _writes?.Add((SourceKind, key, usesPerVertexInputs, source));
        }
        catch (InvalidOperationException)
        {
            // The list closed with the session.
        }

        _written?.TryAdd(key, 0);
        CompileSlot(slot);
        return true;
    }

    /// <summary>The list file of one build: the prefix and a hash of its prewarm stamp.</summary>
    internal static string FileName(string stamp) =>
        string.Create(
            System.Globalization.CultureInfo.InvariantCulture,
            $"{FilePrefix}-{(uint)XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(stamp)):x8}.bin");

    /// <summary>
    /// Removes the lists of other builds beyond the <see cref="MaxOtherLists"/> most
    /// recently written, the unstamped list of older builds among them.
    /// </summary>
    internal static void PruneOtherLists(string directory, string current)
    {
        try
        {
            var others = new DirectoryInfo(directory)
                .EnumerateFiles(FilePrefix + "*.bin")
                .Where(file => !string.Equals(file.Name, Path.GetFileName(current), StringComparison.Ordinal))
                .OrderByDescending(file => file.LastWriteTimeUtc)
                .Skip(MaxOtherLists);
            foreach (var file in others)
            {
                file.Delete();
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Library prewarm could not remove older lists: {exception.Message}");
        }
    }

    /// <summary>The list key of a source: its text and whether it pins the per-vertex-inputs MSL version.</summary>
    public static ulong Key(string source, bool usesPerVertexInputs) =>
        XxHash3.HashToUInt64(MemoryMarshal.AsBytes(source.AsSpan()), usesPerVertexInputs ? 1 : 0);

    /// <summary>
    /// The prewarmed library for this source, retained for the caller, waiting for it when
    /// its compile is under way. False when the caller has to compile it, and then
    /// <see cref="NoteCompiled"/> is owed.
    /// </summary>
    public static bool TryTake(ulong key, out nint library)
    {
        library = 0;
        if (Volatile.Read(ref _slots) is not { } slots)
        {
            return false;
        }

        if (!slots.TryGetValue(key, out var slot))
        {
            Interlocked.Increment(ref _misses);
            return false;
        }

        if (Interlocked.CompareExchange(ref slot.State, Claimed, Queued) == Queued)
        {
            Interlocked.Increment(ref _claimed);
            return false;
        }

        if (Volatile.Read(ref slot.State) == Compiling)
        {
            var started = Stopwatch.GetTimestamp();
            slot.Done.Wait();
            var waited = Stopwatch.GetTimestamp() - started;
            Interlocked.Add(ref _waitTicks, waited);
            Interlocked.Increment(ref _waits);
            MetalCompileScheduler.NotePrewarmWait(waited);
        }

        lock (slot)
        {
            if (slot.Library == 0)
            {
                return false;
            }

            // The slot's reference goes to the module, so the library lives as long as the
            // module keeps it and not for the session (MetalRenderHost.ShaderCache releases
            // an idle module's). A second module of this source compiles it, which macOS's
            // shader cache answers now that this process has compiled it once.
            library = slot.Library;
            slot.Library = 0;
            Interlocked.Decrement(ref _held);
        }

        var hits = Interlocked.Increment(ref _hits);
        if (hits is 1 or 10 or 50 or 100 or 200 or 500 or 1000)
        {
            Console.Error.WriteLine(Summary("Library prewarm hits"));
        }

        return true;
    }

    /// <summary>Records a source the caller compiled itself, or marks a listed one used.</summary>
    public static void NoteCompiled(ulong key, string source, bool usesPerVertexInputs)
    {
        if (Volatile.Read(ref _slots) is not { } slots || _writes is not { } writes || _written is not { } written)
        {
            return;
        }

        if (!written.TryAdd(key, 0))
        {
            return;
        }

        try
        {
            writes.Add(slots.ContainsKey(key)
                ? (UseKind, key, usesPerVertexInputs, null)
                : (SourceKind, key, usesPerVertexInputs, source));
        }
        catch (InvalidOperationException)
        {
            // The list closed with the session.
        }
    }

    /// <summary>Marks a listed source used by a module that took its prewarmed library.</summary>
    public static void NoteTaken(ulong key) => NoteCompiled(key, string.Empty, false);

    private static string Summary(string label) =>
        string.Create(
            System.Globalization.CultureInfo.InvariantCulture,
            $"[LOADER][INFO] {label}: hits={Volatile.Read(ref _hits)} waits={Volatile.Read(ref _waits)} " +
            $"wait_ms={TicksToMs(Interlocked.Read(ref _waitTicks)):F0} claimed={Volatile.Read(ref _claimed)} " +
            $"misses={Volatile.Read(ref _misses)}");

    private static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    /// <summary>Stops the workers, closes the list and releases the libraries nobody took.</summary>
    public static void Stop()
    {
        MetalCompileThread[] threads;
        MetalCompileThread? writer;
        BlockingCollection<(byte, ulong, bool, string?)>? writes;
        ConcurrentDictionary<ulong, Slot>? slots;
        lock (_sync)
        {
            _stopping = true;
            threads = _threads;
            writer = _writer;
            writes = _writes;
            slots = _slots;
            _threads = [];
            _writer = null;
            _writes = null;
            Volatile.Write(ref _slots, null);
        }

        if (slots is null)
        {
            return;
        }

        Console.Error.WriteLine(Summary("Library prewarm session"));
        foreach (var thread in threads)
        {
            if (!thread.IsCurrentThread)
            {
                thread.Join(TimeSpan.FromSeconds(10));
            }
        }

        writes?.CompleteAdding();
        writer?.Join(TimeSpan.FromSeconds(10));
        foreach (var slot in slots.Values)
        {
            lock (slot)
            {
                if (slot.Library != 0 && Volatile.Read(ref slot.State) == Ready)
                {
                    MetalNative.SendVoid(slot.Library, SelRelease);
                    slot.Library = 0;
                    Interlocked.Decrement(ref _held);
                }
            }
        }
    }

    private static void WriteLoop(string path, BlockingCollection<(byte Kind, ulong Key, bool UsesPerVertexInputs, string? Source)> writes)
    {
        try
        {
            using var file = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
            if (file.Length < HeaderBytes)
            {
                file.SetLength(0);
                WriteHeader(file);
            }

            file.Seek(0, SeekOrigin.End);
            var sessionWritten = false;
            foreach (var (kind, key, usesPerVertexInputs, source) in writes.GetConsumingEnumerable())
            {
                if (!sessionWritten)
                {
                    WriteRecord(file, [SessionKind]);
                    sessionWritten = true;
                }

                WriteRecord(file, kind == SourceKind
                    ? SourceBody(key, usesPerVertexInputs, Compress(source!))
                    : UseBody(key));
                file.Flush();
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Library prewarm list could not be written: {exception.Message}");
        }
    }

    /// <summary>
    /// Reads the list: every source with the session it was last used in (sessions count
    /// from one) and its order of use within it. A torn or corrupt tail ends the read and
    /// asks for a rewrite.
    /// </summary>
    internal static (List<Entry> Entries, int Sessions) Load(string path, out bool rewrite)
    {
        rewrite = false;
        var entries = new Dictionary<ulong, Entry>();
        var sessions = 0;
        var records = 0;
        if (!File.Exists(path))
        {
            return ([], 0);
        }

        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < HeaderBytes ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes) != Magic ||
            BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(sizeof(uint))) != FormatVersion)
        {
            rewrite = bytes.Length != 0;
            return ([], 0);
        }

        var offset = HeaderBytes;
        while (offset < bytes.Length)
        {
            if (bytes.Length - offset < RecordHeaderBytes)
            {
                rewrite = true;
                break;
            }

            var length = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset));
            var checksum = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(offset + sizeof(uint)));
            if (length <= 0 || length > MaxRecordBytes || bytes.Length - offset - RecordHeaderBytes < length)
            {
                rewrite = true;
                break;
            }

            var body = bytes.AsSpan(offset + RecordHeaderBytes, length);
            if (XxHash3.HashToUInt64(body) != checksum)
            {
                rewrite = true;
                break;
            }

            offset += RecordHeaderBytes + length;
            records++;
            switch (body[0])
            {
                case SessionKind:
                    sessions++;
                    break;
                case SourceKind when body.Length > 1 + sizeof(ulong) + 1:
                {
                    var key = BinaryPrimitives.ReadUInt64LittleEndian(body[1..]);
                    entries[key] = new Entry(key, body[1 + sizeof(ulong)] != 0, body[(2 + sizeof(ulong))..].ToArray(),
                        Math.Max(sessions, 1), records);
                    break;
                }
                case UseKind when body.Length == 1 + sizeof(ulong):
                {
                    var key = BinaryPrimitives.ReadUInt64LittleEndian(body[1..]);
                    if (entries.TryGetValue(key, out var entry))
                    {
                        entries[key] = entry with { LastSession = Math.Max(sessions, 1), Order = records };
                    }

                    break;
                }
                default:
                    rewrite = true;
                    break;
            }
        }

        // Every use leaves a record behind; the list is compacted once they outnumber the sources.
        if (records > 2 * entries.Count + MaxSessionsUnused)
        {
            rewrite = true;
        }

        return (entries.Values.ToList(), sessions);
    }

    /// <summary>
    /// The sources this start compiles, in the order the most recent start that used them
    /// needed them: those used in recent starts first, the least recent last, none unused for
    /// <see cref="MaxSessionsUnused"/> starts, at most <see cref="MaxEntries"/>.
    /// </summary>
    internal static List<Entry> Select(List<Entry> entries, int sessions) =>
        entries
            .Where(entry => sessions - entry.LastSession < MaxSessionsUnused)
            .OrderByDescending(entry => entry.LastSession)
            .ThenBy(entry => entry.Order)
            .Take(MaxEntries)
            .ToList();

    // Writes the kept sources back with the session markers that keep their ages.
    internal static void Rewrite(string path, List<Entry> kept, int sessions)
    {
        var temporary = path + ".tmp";
        using (var file = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            WriteHeader(file);
            var first = Math.Max(1, sessions - MaxSessionsUnused + 1);
            for (var session = first; session <= sessions; session++)
            {
                WriteRecord(file, [SessionKind]);
                foreach (var entry in kept.Where(entry => entry.LastSession == session).OrderBy(entry => entry.Order))
                {
                    WriteRecord(file, SourceBody(entry.Key, entry.UsesPerVertexInputs, entry.Compressed));
                }
            }
        }

        File.Move(temporary, path, overwrite: true);
    }

    private static void WriteHeader(Stream file)
    {
        Span<byte> header = stackalloc byte[HeaderBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(header, Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(header[sizeof(uint)..], FormatVersion);
        file.Write(header);
    }

    internal static void WriteRecord(Stream file, ReadOnlySpan<byte> body)
    {
        Span<byte> header = stackalloc byte[RecordHeaderBytes];
        BinaryPrimitives.WriteInt32LittleEndian(header, body.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(header[sizeof(uint)..], XxHash3.HashToUInt64(body));
        file.Write(header);
        file.Write(body);
    }

    internal static byte[] SourceBody(ulong key, bool usesPerVertexInputs, byte[] compressed)
    {
        var body = new byte[2 + sizeof(ulong) + compressed.Length];
        body[0] = SourceKind;
        BinaryPrimitives.WriteUInt64LittleEndian(body.AsSpan(1), key);
        body[1 + sizeof(ulong)] = usesPerVertexInputs ? (byte)1 : (byte)0;
        compressed.CopyTo(body, 2 + sizeof(ulong));
        return body;
    }

    internal static byte[] UseBody(ulong key)
    {
        var body = new byte[1 + sizeof(ulong)];
        body[0] = UseKind;
        BinaryPrimitives.WriteUInt64LittleEndian(body.AsSpan(1), key);
        return body;
    }

    internal static byte[] Compress(string source)
    {
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(output, CompressionLevel.Fastest, leaveOpen: true))
        {
            zlib.Write(Encoding.UTF8.GetBytes(source));
        }

        return output.ToArray();
    }

    internal static string Decompress(byte[] compressed)
    {
        using var input = new ZLibStream(new MemoryStream(compressed), CompressionMode.Decompress);
        using var reader = new StreamReader(input, Encoding.UTF8);
        return reader.ReadToEnd();
    }
}
