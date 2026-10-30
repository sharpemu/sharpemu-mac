// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO.Hashing;
using System.Runtime.InteropServices;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Logging;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The title's own copy of what the driver compiled, kept in MTLBinaryArchive files under
/// the title's pipeline cache, so a start whose macOS Metal cache no longer holds a shader
/// (another bundle id, another title pushed it out, the cache was cleared) does not run the
/// compilers again.
///
/// A serialized archive is a fat file: an air64 slice, a metallib with the AIR of every
/// function in it, and a slice of GPU binaries for this device. That gives two caches:
///
/// <list type="bullet">
/// <item>Libraries. Each library the host compiles from MSL gets one archive file, named by
/// a hash of its source and compile options and written from the first pipeline made from
/// it. Opened as a library (newLibraryWithURL) its AIR slice stands in for the MSL front end:
/// measured on 159 Demon's Souls compute libraries with the macOS cache cleared, 0.13-0.44 ms
/// a library against 330-540 ms through newLibraryWithSource. A compute library's file also
/// holds its pipeline's binary, and opened as an archive (about 6 ms) it stands in for the
/// back end: 0.12 ms a pipeline against 106-181 ms. The name is the content, so a translator
/// change only misses the libraries whose MSL it changed.</item>
/// <item>Render pipelines. Pipelines no loaded archive had are added to a segment
/// (pipelines-STAMP-SEQ.mtla, a new one every <see cref="SegmentBytes"/>) that is serialized
/// off the GPU worker, every <see cref="FlushInterval"/> and at shutdown. An open archive is
/// read into memory whole (33 MB of footprint for a 28 MB file) and a title's segments add up
/// to hundreds of MB, so the next start opens none of them: each segment ends in the list of
/// shader pairs (vertex or mesh library key, pixel library key) of its pipelines, read from
/// the file's tail, and a pipeline creation opens the segments of its pair when it asks, the
/// least recently used closing again past <see cref="ResidentSegmentBytes"/>. It asks them
/// with FailOnBinaryArchiveMiss and compiles only on a miss: 0.2-0.6 ms against 7-21 ms.</item>
/// </list>
///
/// Every file ends in a trailer (magic, format, payload length, XxHash64 of the payload);
/// Metal ignores bytes past the slices, and a file whose trailer does not match is deleted
/// unread: a truncated archive still opens as a library, and a flipped byte goes unnoticed
/// by Metal. A directory written for another device, OS or format is removed at start. The
/// files are bounded to <see cref="DiskBudgetBytes"/>, least recently used first (a segment
/// is used when it is opened).
/// SHARPEMU_METAL_V2_BINARY_ARCHIVE=0 turns it off.
///
/// The archiver packs a whole archive or none: one pipeline it refuses fails the
/// serialization of every pipeline in the segment, so a failed segment is searched for the
/// pipelines it refuses (<see cref="RebuildWithoutRefused"/>) and written without them.
/// serializeToURL also reports success and writes nothing when an archive is unchanged since
/// its last serialization, which is what adding a pipeline it already holds leaves it.
/// </summary>
internal sealed partial class MetalBinaryArchive : IDisposable
{
    internal const uint FormatVersion = 1;
    internal const string Extension = ".mtla";
    internal const string SegmentPrefix = "pipelines-";
    // A title's working set: Black Myth: Wukong's opening alone builds over a thousand render
    // pipelines (about a gigabyte of GPU binaries) and three thousand libraries, so a 1 GiB
    // bound evicted every run's pipelines before the next run could open them, and each boot
    // compiled them all again (minutes of loading, seconds-long stalls in play).
    // SHARPEMU_METAL_V2_BINARY_ARCHIVE_BUDGET_MB=<n> sets it.
    internal static readonly long DiskBudgetBytes =
        long.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BINARY_ARCHIVE_BUDGET_MB"), out var budgetMb) && budgetMb > 0
            ? budgetMb << 20
            : 8L << 30;
    internal const long SegmentBytes = 8L << 20;
    internal const long ResidentSegmentBytes = 32L << 20;
    internal const int TrailerBytes = 24;
    internal static readonly TimeSpan FlushInterval = TimeSpan.FromSeconds(15);
    private const uint TrailerMagic = 0x41424553; // "SEBA"
    private const uint IndexMagic = 0x58444953; // "SIDX"
    private const int FlushAfterPipelines = 64;

    // A queued capture keeps its descriptor's functions alive; past this many the newest
    // are dropped (their pipelines are stored by a later start).
    // SHARPEMU_METAL_V2_BINARY_ARCHIVE_QUEUE=<n> sets it.
    private static readonly int MaxQueuedJobs =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BINARY_ARCHIVE_QUEUE"), out var queued) && queued > 0 ? queued : 256;

    // MTLPipelineOptionFailOnBinaryArchiveMiss.
    internal const nuint FailOnBinaryArchiveMiss = 1 << 2;

    internal static readonly bool Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BINARY_ARCHIVE") != "0";

    private static readonly nint SelRetain = MetalNative.Selector("retain");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelAlloc = MetalNative.Selector("alloc");
    private static readonly nint SelInit = MetalNative.Selector("init");

    internal enum PipelineKind : byte
    {
        Compute,
        Render,
        Mesh,
    }

    private enum JobKind : byte
    {
        Capture,
        Flush,
    }

    private readonly record struct Job(
        JobKind Kind, PipelineKind Pipeline, nint Descriptor, UInt128 Key0, UInt128 Key1, bool AddToSegment,
        ManualResetEventSlim? Done, ulong Pair = 0);

    // A pipeline of the current segment, retained to search the segment when the archiver
    // refuses one of them.
    private readonly record struct SegmentEntry(PipelineKind Kind, nint Descriptor, ulong Pair);

    private sealed class ResidentSegment(nint archive, long bytes)
    {
        public nint Archive { get; } = archive;

        public long Bytes { get; } = bytes;

        public long Tick { get; set; }
    }

    private static MetalBinaryArchive? _current;
    private static int _exitHooked;

    private readonly nint _device;
    private readonly string _root;
    private readonly string _directory;
    private readonly string _libraries;
    private readonly string _translatorStamp;
    private readonly long _budget;
    private readonly ConcurrentDictionary<UInt128, byte> _known = new();
    private readonly ConcurrentDictionary<UInt128, byte> _pending = new();
    private readonly BlockingCollection<Job> _jobs = new(MaxQueuedJobs);
    private readonly MetalCompileThread _writer;
    private readonly ManualResetEventSlim _loaded = new(false);
    private readonly object _archivesLock = new();
    private readonly object _trimLock = new();

    // The segments on disk by shader pair, newest first, and the ones open now; under
    // _archivesLock.
    private readonly Dictionary<ulong, List<int>> _segmentsByPair = new();

    // Segments another translator build wrote, by the negative id they are listed under. Their
    // pairs are keyed by the MSL of both shaders, so a pair this build still makes is the same
    // pipeline; a translator change then recompiles only the pipelines whose MSL it changed.
    private readonly Dictionary<int, string> _foreignSegments = new();
    private readonly Dictionary<int, ResidentSegment> _resident = new();
    private long _residentBytes;
    private long _residentTick;
    private int _disposed;

    // The writer thread's own.
    private nint _segment;
    private int _segmentSequence;
    private int _segmentDirty;
    private int _segmentPipelines;
    private long _segmentBytes;
    private long _bytesPerPipeline;
    private readonly List<SegmentEntry> _segmentEntries = [];
    private readonly HashSet<ulong> _segmentPairs = [];
    private long _lastFlushTicks;
    private long _writtenSinceTrim;

    private int _libraryHits;
    private int _libraryMisses;
    private int _rejected;
    private int _librariesWritten;
    private int _pipelineHits;
    private int _pipelineMisses;
    private int _pipelinesAdded;
    private int _loadedSegments;
    private int _segmentsOpened;
    private int _capturesDropped;
    private long _acquireTicks;
    private int _acquires;
    private int _maxSegmentsPerPair;
    private int _pipelinesRefused;
    private int _librariesStored;
    private long _libraryBytesStored;
    private long _loadedSegmentBytes;

    // Adds the time an AcquirePipelineArchives call took to the closed line's acquire_ms.
    private readonly struct AcquireTimer(MetalBinaryArchive archive, long started) : IDisposable
    {
        public void Dispose()
        {
            Interlocked.Add(ref archive._acquireTicks, Stopwatch.GetTimestamp() - started);
            Interlocked.Increment(ref archive._acquires);
        }
    }

    private MetalBinaryArchive(nint device, string root, string translatorStamp, long budget)
    {
        _device = device;
        _root = root;
        _translatorStamp = Hex64(XxHash3.HashToUInt64(MemoryMarshal.AsBytes(translatorStamp.AsSpan())));
        _budget = budget;
        _directory = Path.Combine(root, Hex64(XxHash3.HashToUInt64(MemoryMarshal.AsBytes(DeviceStamp(device).AsSpan()))));
        _libraries = Path.Combine(_directory, "libraries");
        _lastFlushTicks = Stopwatch.GetTimestamp();
        // Pipelines asked for before the segments are listed miss them, so the listing runs at the
        // class of the compiles that wait; the writes after it at Utility (RunWriter).
        _writer = MetalCompileThread.Create("SharpEmu Metal v2 binary archive", SharpEmu.HLE.Host.HostThreadQos.UserInitiated, RunWriter);
    }

    /// <summary>The process's archive, once the presenter started it, for this device.</summary>
    public static MetalBinaryArchive? For(nint device) =>
        Volatile.Read(ref _current) is { } archive && archive._device == device ? archive : null;

    /// <summary>The library directory, for tests.</summary>
    internal string LibraryDirectory => _libraries;

    /// <summary>The device, OS and format directory, for tests.</summary>
    internal string Directory => _directory;

    /// <summary>What the open segments may add up to; for tests. SHARPEMU_METAL_V2_BINARY_ARCHIVE_RESIDENT_MB=<n> sets it.</summary>
    internal long ResidentLimit { get; set; } =
        long.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BINARY_ARCHIVE_RESIDENT_MB"), out var residentMb) && residentMb > 0
            ? residentMb << 20
            : ResidentSegmentBytes;

    /// <summary>Segments opened for a pipeline lookup and pipelines the archiver refused, for tests.</summary>
    internal (int SegmentsOpened, int PipelinesRefused) SegmentCounters =>
        (Volatile.Read(ref _segmentsOpened), Volatile.Read(ref _pipelinesRefused));

    /// <summary>The descriptors the archiver is taken to refuse, for tests: the real refusals
    /// cannot be made on demand.</summary>
    internal Func<nint, bool>? RefusedByArchiver { get; set; }

    // Tests: the archiver refuses an archive of more pipelines than this, as it refused a whole
    // segment of Black Myth: Wukong's every part of which it took.
    internal int? RefusesArchivesOver { get; set; }

    public (int LibraryHits, int LibraryMisses, int Rejected, int LibrariesWritten,
        int PipelineHits, int PipelineMisses, int PipelinesAdded, int LoadedSegments) Counters =>
        (Volatile.Read(ref _libraryHits), Volatile.Read(ref _libraryMisses), Volatile.Read(ref _rejected),
            Volatile.Read(ref _librariesWritten), Volatile.Read(ref _pipelineHits), Volatile.Read(ref _pipelineMisses),
            Volatile.Read(ref _pipelinesAdded), Volatile.Read(ref _loadedSegments));

    /// <summary>The library files <see cref="StoreLibrary"/> wrote and their bytes on disk.</summary>
    public (int Libraries, long Bytes) Stored => (Volatile.Read(ref _librariesStored), Interlocked.Read(ref _libraryBytesStored));

    /// <summary>The budget the archive keeps its files under, in bytes.</summary>
    public long Budget => _budget;

    /// <summary>What the archive's files take on disk now, in bytes.</summary>
    public long DiskBytes()
    {
        try
        {
            return new DirectoryInfo(_directory).EnumerateFiles("*" + Extension, SearchOption.AllDirectories).Sum(static file => file.Length);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return 0;
        }
    }

    /// <summary>
    /// Opens the title's archive for the process before the shader prewarm starts, so the
    /// prewarm's libraries are taken from it too. Failure leaves the backend without one.
    /// </summary>
    public static void Start(nint device)
    {
        if (!Enabled || device == 0 || Volatile.Read(ref _current) is not null)
        {
            return;
        }

        try
        {
            var root = SharpEmuPaths.UserPath(
                "pipeline_cache",
                MetalShaderPrewarm.SanitizeTitleId(VideoOutBackendServices.GetApplicationTitleId()),
                "metal-v2",
                "binary-archive");
            var archive = Open(root, device, TranslatorStamp(), DiskBudgetBytes);
            if (Interlocked.CompareExchange(ref _current, archive, null) is not null)
            {
                archive.Dispose();
                return;
            }

            if (Interlocked.Exchange(ref _exitHooked, 1) == 0)
            {
                AppDomain.CurrentDomain.ProcessExit += static (_, _) => Stop();
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Metal binary archive could not open: {exception.Message}");
        }
    }

    /// <summary>Makes <paramref name="archive"/> the process's archive, for tests; returns the one it replaced.</summary>
    internal static MetalBinaryArchive? Install(MetalBinaryArchive? archive) => Interlocked.Exchange(ref _current, archive);

    /// <summary>Writes what is pending and closes the process's archive.</summary>
    public static void Stop() => Interlocked.Exchange(ref _current, null)?.Dispose();

    /// <summary>
    /// Opens (creating) the archive under <paramref name="root"/>: removes directories of
    /// other devices, OSes and formats, lists the library files and starts the writer, which
    /// loads the segments of <paramref name="translatorStamp"/>.
    /// </summary>
    internal static MetalBinaryArchive Open(string root, nint device, string translatorStamp, long budget)
    {
        var archive = new MetalBinaryArchive(device, root, translatorStamp, budget);
        archive.Prepare();
        archive._writer.Start();
        return archive;
    }

    /// <summary>The translators decide the MSL; segments of another stamp hold pipelines of
    /// functions this build no longer makes.</summary>
    internal static string TranslatorStamp() =>
        string.Join(
            '|',
            AssemblyCodeIdentity.Of(typeof(Gen5ShaderTranslator).Assembly),
            AssemblyCodeIdentity.Of(typeof(Gen5MslTranslator).Assembly));

    // The device and the OS decide what the driver makes of AIR, and what a binary slice is
    // for; the format is this file's.
    private static string DeviceStamp(nint device) =>
        string.Join(
            '|',
            FormatVersion.ToString(CultureInfo.InvariantCulture),
            MetalNative.ReadNsString(MetalNative.Send(device, MetalNative.Selector("name"))),
            RuntimeInformation.OSDescription,
            RuntimeInformation.OSArchitecture);

    /// <summary>
    /// The library key of a source: its text and the options it compiles with (the math mode,
    /// the MSL version the per-vertex-inputs flag picks). A binary built in the safe math
    /// mode is another binary than the relaxed one, so the mode is part of the seed; the
    /// safe seed is the one every library was keyed with before the mode existed.
    /// </summary>
    public static UInt128 LibraryKey(string source, bool usesPerVertexInputs) =>
        LibraryKey(source, usesPerVertexInputs, MslMathMode.Default);

    internal static UInt128 LibraryKey(string source, bool usesPerVertexInputs, nuint mathMode)
    {
        var version = usesPerVertexInputs ? Gen5MslShader.PerVertexInputsLanguageVersion : MetalNative.MslLanguageVersion;
        var seed = (long)(((ulong)FormatVersion << 48) ^ version ^ ((ulong)mathMode << 40));
        return XxHash128.HashToUInt128(MemoryMarshal.AsBytes(source.AsSpan()), seed);
    }

    /// <summary>
    /// The library stored for <paramref name="key"/>, owned by the caller, or zero. A file
    /// that fails its trailer or that Metal will not open is deleted. Any thread.
    /// </summary>
    public nint TryLoadLibrary(UInt128 key)
    {
        if (key == default || !_known.ContainsKey(key))
        {
            Interlocked.Increment(ref _libraryMisses);
            return 0;
        }

        var path = LibraryPath(key);
        if (!TryVerify(path, out _))
        {
            Reject(key, path);
            return 0;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            nint error = 0;
            var library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithURL:error:"), FileUrl(path), ref error);
            if (library == 0)
            {
                Reject(key, path);
                return 0;
            }

            Touch(path);
            Interlocked.Increment(ref _libraryHits);
            return library;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    /// <summary>
    /// The library file of <paramref name="key"/> opened as an archive, for the compute
    /// pipeline its binary slice holds; owned by the caller, or zero.
    /// </summary>
    public nint OpenLibraryArchive(UInt128 key) =>
        key != default && _known.ContainsKey(key) ? OpenArchive(LibraryPath(key)) : 0;

    /// <summary>
    /// The segments holding pipelines of the shader pair as an NSArray for a descriptor's
    /// binaryArchives, retained for the caller, or zero before the segments are listed or
    /// when none holds the pair. Opens the ones that are closed (reading each whole).
    /// </summary>
    public nint AcquirePipelineArchives(UInt128 vertexKey, UInt128 pixelKey)
    {
        if (!_loaded.IsSet)
        {
            return 0;
        }

        var pair = PairKey(vertexKey, pixelKey);
        var acquireStarted = Stopwatch.GetTimestamp();
        using var acquireTimer = new AcquireTimer(this, acquireStarted);
        lock (_archivesLock)
        {
            if (Volatile.Read(ref _disposed) != 0 || !_segmentsByPair.TryGetValue(pair, out var sequences))
            {
                return 0;
            }

            var array = MetalNative.Send(MetalNative.Send(MetalNative.Class("NSMutableArray"), SelAlloc), SelInit);
            var added = 0;
            if (sequences.Count > _maxSegmentsPerPair) _maxSegmentsPerPair = sequences.Count;
            foreach (var sequence in sequences.ToArray())
            {
                var archive = OpenSegment(sequence, sequences);
                if (archive != 0)
                {
                    MetalNative.SendVoid(array, MetalNative.Selector("addObject:"), archive);
                    added++;
                }
            }

            if (added == 0)
            {
                MetalNative.SendVoid(array, SelRelease);
                return 0;
            }

            return array;
        }
    }

    /// <summary>
    /// The archive key of a pixel library for a pipeline made with its LOD statistics counters
    /// (Gen5MslShader.UsesMipStatisticsConstant): another function of the same library, so
    /// another pair. The key itself without them.
    /// </summary>
    internal static UInt128 LodStatisticsKey(UInt128 key, bool counters) =>
        key == default || !counters ? key : key ^ new UInt128(0x4C4F44535441545FUL, 0x434F554E54455253UL);

    /// <summary>The key a shader pair is listed under in a segment's tail.</summary>
    internal static ulong PairKey(UInt128 vertexKey, UInt128 pixelKey)
    {
        Span<UInt128> keys = [vertexKey, pixelKey];
        return XxHash3.HashToUInt64(MemoryMarshal.AsBytes(keys));
    }

    /// <summary>Counts a pipeline created from (or missed by) an archive.</summary>
    public void NotePipelineLookup(bool hit) =>
        Interlocked.Increment(ref hit ? ref _pipelineHits : ref _pipelineMisses);

    /// <summary>
    /// Stores the library of <paramref name="key"/> with the compute pipeline of
    /// <paramref name="function"/>, off the caller's thread, unless it is stored already.
    /// </summary>
    public void CaptureCompute(UInt128 key, nint function)
    {
        if (!NeedsLibrary(key) || function == 0 || !_pending.TryAdd(key, 0))
        {
            return;
        }

        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLComputePipelineDescriptor"), SelAlloc), SelInit);
        if (descriptor == 0)
        {
            _pending.TryRemove(key, out _);
            return;
        }

        MetalNative.SendVoid(descriptor, MetalNative.Selector("setComputeFunction:"), function);
        if (!TryEnqueue(new Job(JobKind.Capture, PipelineKind.Compute, descriptor, key, default, false, null)))
        {
            MetalNative.SendVoid(descriptor, SelRelease);
            _pending.TryRemove(key, out _);
        }
    }

    /// <summary>Whether the library file of <paramref name="key"/> is on disk (or being written).</summary>
    public bool HasLibrary(UInt128 key) => key != default && (_known.ContainsKey(key) || _pending.ContainsKey(key));

    /// <summary>Whether the library file of <paramref name="key"/> is written (a compute's holds its pipeline's binary too).</summary>
    public bool IsStored(UInt128 key) => key != default && _known.ContainsKey(key);

    /// <summary>
    /// Stores the library file of <paramref name="key"/> from a pipeline that is never made:
    /// the functions of <paramref name="descriptor"/> added to an archive of their own and
    /// written, on the caller's thread (which waits for the driver's back end). For the
    /// static precompile, which compiles programs the game has not drawn yet and keeps none
    /// of them; the runtime's <see cref="TryLoadLibrary"/> then opens the file as the
    /// library. The bytes written, or -1 when the file was there already or failed.
    /// </summary>
    public long StoreLibrary(UInt128 key, PipelineKind kind, nint descriptor)
    {
        if (descriptor == 0 || !NeedsLibrary(key) || !_pending.TryAdd(key, 0))
        {
            return -1;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        var archive = NewArchive(null);
        try
        {
            if (archive == 0 || !Add(archive, kind, descriptor))
            {
                return -1;
            }

            var bytes = Serialize(archive, LibraryPath(key));
            if (bytes < 0)
            {
                return -1;
            }

            _known.TryAdd(key, 0);
            Interlocked.Increment(ref _librariesWritten);
            Interlocked.Increment(ref _librariesStored);
            Interlocked.Add(ref _libraryBytesStored, bytes);
            NoteWritten(bytes);
            return bytes;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] {exception.Message}");
            return -1;
        }
        finally
        {
            if (archive != 0)
            {
                MetalNative.SendVoid(archive, SelRelease);
            }

            MetalNative.objc_autoreleasePoolPop(pool);
            Forget(key);
        }
    }

    /// <summary>
    /// Takes a render or mesh pipeline just created from <paramref name="descriptor"/>:
    /// added to the current segment when no loaded archive had it, and written as the
    /// library file of either stage key that has none. The descriptor is retained until
    /// the writer is done with it. The two stages' entry points must differ, so each file's
    /// library has one function of the name its module asks for.
    /// </summary>
    public void NoteRenderPipeline(nint descriptor, PipelineKind kind, UInt128 vertexKey, UInt128 pixelKey, bool missed)
    {
        if (descriptor == 0 || Volatile.Read(ref _disposed) != 0)
        {
            return;
        }

        // Before the segments are loaded nothing could hit; a pipeline they hold would be
        // stored twice, and one they do not is stored by a later start.
        missed &= _loaded.IsSet;
        var vertex = NeedsLibrary(vertexKey) && _pending.TryAdd(vertexKey, 0) ? vertexKey : default;
        var pixel = pixelKey != vertexKey && NeedsLibrary(pixelKey) && _pending.TryAdd(pixelKey, 0) ? pixelKey : default;
        if (!missed && vertex == default && pixel == default)
        {
            return;
        }

        MetalNative.Send(descriptor, SelRetain);
        if (!TryEnqueue(new Job(JobKind.Capture, kind, descriptor, vertex, pixel, missed, null, PairKey(vertexKey, pixelKey))))
        {
            Interlocked.Increment(ref _capturesDropped);
            MetalNative.SendVoid(descriptor, SelRelease);
            Forget(vertex);
            Forget(pixel);
        }
    }

    /// <summary>Serializes the current segment now; waits when <paramref name="wait"/>.</summary>
    public void Flush(bool wait, TimeSpan? timeout = null)
    {
        // Not disposed: a wait that times out leaves the writer to set it later.
        var done = wait ? new ManualResetEventSlim(false) : null;
        var limit = timeout ?? TimeSpan.FromSeconds(30);
        if (!TryEnqueue(new Job(JobKind.Flush, default, 0, default, default, false, done), wait ? limit : TimeSpan.Zero))
        {
            return;
        }

        done?.Wait(limit);
    }

    /// <summary>Waits for the writer to have loaded the pipeline segments, for tests.</summary>
    internal bool WaitLoaded(TimeSpan timeout) => _loaded.Wait(timeout);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _jobs.CompleteAdding();
        if (!_writer.Join(TimeSpan.FromSeconds(30)))
        {
            Console.Error.WriteLine("[METAL-V2][ARCHIVE][WARN] The writer did not finish in 30 s; pending pipelines are dropped.");
            return;
        }

        lock (_archivesLock)
        {
            foreach (var segment in _resident.Values)
            {
                MetalNative.SendVoid(segment.Archive, SelRelease);
            }

            _resident.Clear();
            _residentBytes = 0;
            _segmentsByPair.Clear();
        }

        var (hits, misses, rejected, written, pipelineHits, pipelineMisses, added, segments) = Counters;
        Console.Error.WriteLine(
            $"[METAL-V2][ARCHIVE] closed library_hits={hits} library_misses={misses} rejected={rejected} " +
            $"libraries_written={written} pipeline_hits={pipelineHits} pipeline_misses={pipelineMisses} " +
            $"pipelines_added={added} pipelines_refused={Volatile.Read(ref _pipelinesRefused)} " +
            $"segments_listed={segments} segments_opened={Volatile.Read(ref _segmentsOpened)} " +
            $"captures_dropped={Volatile.Read(ref _capturesDropped)} acquires={Volatile.Read(ref _acquires)} " +
            $"acquire_ms={Volatile.Read(ref _acquireTicks) * 1000.0 / Stopwatch.Frequency:F0} max_segments_per_pair={_maxSegmentsPerPair}");
    }

    private bool NeedsLibrary(UInt128 key) =>
        key != default && Volatile.Read(ref _disposed) == 0 && !_known.ContainsKey(key) && !_pending.ContainsKey(key);

    private void Forget(UInt128 key)
    {
        if (key != default)
        {
            _pending.TryRemove(key, out _);
        }
    }

    private bool TryEnqueue(Job job) => TryEnqueue(job, TimeSpan.Zero);

    private bool TryEnqueue(Job job, TimeSpan timeout)
    {
        try
        {
            return _jobs.TryAdd(job, timeout);
        }
        catch (InvalidOperationException)
        {
            // Closed: shutdown has begun.
            return false;
        }
    }

    private string LibraryPath(UInt128 key) =>
        Path.Combine(_libraries, key.ToString("x32", CultureInfo.InvariantCulture) + Extension);

    private string SegmentPath(int sequence) =>
        Path.Combine(_directory, $"{SegmentPrefix}{_translatorStamp}-{sequence.ToString("D5", CultureInfo.InvariantCulture)}{Extension}");

    // Where a listed segment lives: this build's own by sequence, another build's by its file.
    private string SegmentFile(int sequence) =>
        sequence < 0 && _foreignSegments.TryGetValue(sequence, out var path) ? path : SegmentPath(sequence);

    private void Reject(UInt128 key, string path)
    {
        Interlocked.Increment(ref _rejected);
        _known.TryRemove(key, out _);
        TryDelete(path);
    }

    // Removes what other devices, OSes and formats wrote and what an interrupted write
    // left, then lists the libraries on disk.
    private void Prepare()
    {
        System.IO.Directory.CreateDirectory(_libraries);
        foreach (var other in System.IO.Directory.EnumerateDirectories(_root))
        {
            if (!string.Equals(Path.GetFullPath(other), Path.GetFullPath(_directory), StringComparison.Ordinal))
            {
                TryDeleteDirectory(other);
            }
        }

        foreach (var path in System.IO.Directory.EnumerateFiles(_directory, "*", SearchOption.AllDirectories))
        {
            if (!path.EndsWith(Extension, StringComparison.Ordinal))
            {
                TryDelete(path);
            }
        }

        foreach (var path in System.IO.Directory.EnumerateFiles(_libraries, "*" + Extension))
        {
            if (UInt128.TryParse(Path.GetFileNameWithoutExtension(path), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var key))
            {
                _known.TryAdd(key, 0);
            }
            else
            {
                TryDelete(path);
            }
        }
    }

    private void RunWriter()
    {
        try
        {
            using (MetalAutoreleasePool.Enter())
            {
                LoadSegments();
                lock (_trimLock)
                {
                    Trim();
                }
            }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] Loading the pipeline segments failed: {exception.Message}");
        }
        finally
        {
            _loaded.Set();
        }

        MetalCompileThread.ApplyToCurrent(SharpEmu.HLE.Host.HostThreadQos.Utility);
        try
        {
            while (!_jobs.IsCompleted)
            {
                if (_jobs.TryTake(out var job, TimeSpan.FromSeconds(1)))
                {
                    using (MetalAutoreleasePool.Enter())
                    {
                        Run(job);
                    }
                }

                if (_segmentDirty != 0 &&
                    (_segmentDirty >= FlushThreshold() ||
                     Stopwatch.GetElapsedTime(_lastFlushTicks) >= FlushInterval))
                {
                    using (MetalAutoreleasePool.Enter())
                    {
                        FlushSegment();
                    }
                }
            }
        }
        catch (InvalidOperationException)
        {
            // Completed while taking.
        }

        using (MetalAutoreleasePool.Enter())
        {
            FlushSegment();
            if (_segment != 0)
            {
                CloseSegment();
            }
        }
    }

    private void Run(Job job)
    {
        try
        {
            if (job.Kind == JobKind.Flush)
            {
                FlushSegment();
                return;
            }

            if (job.AddToSegment)
            {
                AddToSegment(job.Pipeline, job.Descriptor, job.Pair);
            }

            if (job.Key0 != default || job.Key1 != default)
            {
                WriteLibraries(job.Pipeline, job.Descriptor, job.Key0, job.Key1);
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] {exception.Message}");
        }
        finally
        {
            if (job.Descriptor != 0)
            {
                MetalNative.SendVoid(job.Descriptor, SelRelease);
            }

            Forget(job.Key0);
            Forget(job.Key1);
            job.Done?.Set();
        }
    }

    private void AddToSegment(PipelineKind kind, nint descriptor, ulong pair)
    {
        if (_segment == 0)
        {
            _segment = NewArchive(null);
            if (_segment == 0)
            {
                return;
            }
        }

        if (Add(_segment, kind, descriptor))
        {
            MetalNative.Send(descriptor, SelRetain);
            _segmentEntries.Add(new SegmentEntry(kind, descriptor, pair));
            _segmentPairs.Add(pair);
            _segmentPipelines++;
            _segmentDirty++;
            Interlocked.Increment(ref _pipelinesAdded);
        }
    }

    // A flush also serializes what was added since the last one, so it is sized by what the
    // pipelines took so far: a segment ends within a few pipelines of SegmentBytes instead of
    // a flush interval's worth past it.
    private int FlushThreshold() =>
        _bytesPerPipeline <= 0
            ? FlushAfterPipelines
            : (int)Math.Clamp((SegmentBytes - _segmentBytes) / _bytesPerPipeline, 1, FlushAfterPipelines);

    private void FlushSegment()
    {
        _lastFlushTicks = Stopwatch.GetTimestamp();
        if (_segment == 0 || _segmentDirty == 0)
        {
            return;
        }

        _segmentDirty = 0;
        var path = SegmentPath(_segmentSequence);
        var bytes = WriteSegment(_segment, _segmentEntries, path);
        if (bytes < 0)
        {
            bytes = RebuildWithoutRefused(path);
        }

        if (bytes < 0 || bytes >= SegmentBytes || _closeAfterFlush)
        {
            // Full, unwritable or written in pieces: the next pipeline starts the next segment,
            // and this one's memory goes with its archive.
            _closeAfterFlush = false;
            CloseSegment();
        }
        else
        {
            _segmentBytes = bytes;
            _bytesPerPipeline = Math.Max(bytes / Math.Max(_segmentPipelines, 1), 1);
        }

        NoteWritten(Math.Max(bytes, 0));
    }

    private void CloseSegment()
    {
        MetalNative.SendVoid(_segment, SelRelease);
        _segment = 0;
        ReleaseEntries(_segmentEntries);
        _segmentEntries.Clear();
        _segmentPairs.Clear();
        _segmentPipelines = 0;
        _segmentBytes = 0;
        _segmentSequence++;
    }

    private static void ReleaseEntries(List<SegmentEntry> entries)
    {
        foreach (var entry in entries)
        {
            MetalNative.SendVoid(entry.Descriptor, SelRelease);
        }
    }

    // The archiver packs all the pipelines of an archive or none. Tests stand in for the
    // pipelines it refuses.
    private long WriteSegment(nint archive, List<SegmentEntry> entries, string path)
    {
        if ((RefusedByArchiver is { } refused && entries.Exists(entry => refused(entry.Descriptor))) ||
            entries.Count > RefusesArchivesOver)
        {
            Console.Error.WriteLine("[METAL-V2][ARCHIVE][WARN] An archive could not be written: a pipeline was refused.");
            return -1;
        }

        return Serialize(archive, path, entries.Select(static entry => entry.Pair).Distinct().ToArray());
    }

    /// <summary>
    /// Writes the current segment again without the pipelines the archiver refuses, found by
    /// halves: the pipelines of the other half are archived again (the driver has them cached)
    /// into archives of their own and written to a scratch file. The refused are described once
    /// and dropped; the segment goes on from the archive made without them. The segment's
    /// length, or -1 when none of it could be written.
    /// </summary>
    private long RebuildWithoutRefused(string path)
    {
        var entries = _segmentEntries.ToArray();
        var refused = new List<SegmentEntry>();
        FindRefused(entries, 0, entries.Length, path + ".probe", refused);
        TryDelete(path + ".probe");
        if (refused.Count == 0)
        {
            return WriteWithoutRefusal(entries, path);
        }

        var good = entries.Where(entry => !refused.Contains(entry)).ToList();
        foreach (var entry in refused)
        {
            Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] The archiver refused a pipeline, which is not stored: {Describe(entry)}");
        }

        Interlocked.Add(ref _pipelinesRefused, refused.Count);
        var archive = good.Count == 0 ? 0 : NewArchive(null);
        long bytes = -1;
        if (archive != 0)
        {
            foreach (var entry in good)
            {
                Add(archive, entry.Kind, entry.Descriptor);
            }

            bytes = WriteSegment(archive, good, path);
        }

        if (bytes < 0)
        {
            if (archive != 0)
            {
                MetalNative.SendVoid(archive, SelRelease);
            }

            return -1;
        }

        MetalNative.SendVoid(_segment, SelRelease);
        _segment = archive;
        ReleaseEntries(refused);
        _segmentEntries.Clear();
        _segmentEntries.AddRange(good);
        _segmentPairs.Clear();
        _segmentPairs.UnionWith(good.Select(static entry => entry.Pair));
        _segmentPipelines = good.Count;
        return bytes;
    }

    /// <summary>
    /// Writes a segment no pipeline of which the archiver refuses alone, though it refused the
    /// segment's archive whole: in Black Myth: Wukong's gameplay every half archived and the whole
    /// did not ("expecting 'fragment' stage in pipeline no. 5"), and the segment was dropped, so
    /// its dozens of pipelines were built again on the next run, each a stall of 0.3-1.4 s. The
    /// segment is written from a fresh archive of its pipelines, and failing that in pieces, each
    /// a segment of its own at the next sequence numbers, halved only as far as the archiver needs.
    /// </summary>
    private long WriteWithoutRefusal(SegmentEntry[] entries, string path)
    {
        var fresh = NewArchive(null);
        if (fresh == 0)
        {
            return -1;
        }

        foreach (var entry in entries)
        {
            Add(fresh, entry.Kind, entry.Descriptor);
        }

        var bytes = WriteSegment(fresh, entries.ToList(), path);
        if (bytes >= 0)
        {
            Interlocked.Increment(ref _segmentsRewrittenFresh);
            MetalNative.SendVoid(_segment, SelRelease);
            _segment = fresh;
            return bytes;
        }

        MetalNative.SendVoid(fresh, SelRelease);
        var sequence = _segmentSequence;
        var written = WritePieces(entries, 0, entries.Length, ref sequence);
        Interlocked.Increment(ref _segmentsWrittenInPieces);
        // The pieces are segments of their own, so this one ends: the next pipeline starts the
        // segment after the last piece (CloseSegment moves past the current sequence).
        _segmentSequence = Math.Max(sequence - 1, _segmentSequence);
        _closeAfterFlush = true;
        return written;
    }

    // Writes entries[start..start + count] as one segment at the sequence, or in halves when the
    // archiver refuses it whole; a single pipeline it refuses is dropped. The bytes written.
    private long WritePieces(SegmentEntry[] entries, int start, int count, ref int sequence)
    {
        var piece = entries.AsSpan(start, count).ToArray();
        var archive = NewArchive(null);
        if (archive == 0)
        {
            return 0;
        }

        try
        {
            foreach (var entry in piece)
            {
                Add(archive, entry.Kind, entry.Descriptor);
            }

            var refusedHere = (RefusedByArchiver is { } refused && piece.Any(entry => refused(entry.Descriptor))) ||
                count > RefusesArchivesOver;
            var bytes = refusedHere
                ? -1
                : Serialize(archive, SegmentPath(sequence), piece.Select(static entry => entry.Pair).Distinct().ToArray(), quiet: true);
            if (bytes >= 0)
            {
                sequence++;
                return bytes;
            }
        }
        finally
        {
            MetalNative.SendVoid(archive, SelRelease);
        }

        if (count == 1)
        {
            Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] The archiver refused a pipeline, which is not stored: {Describe(entries[start])}");
            Interlocked.Increment(ref _pipelinesRefused);
            return 0;
        }

        var half = count / 2;
        return WritePieces(entries, start, half, ref sequence) + WritePieces(entries, start + half, count - half, ref sequence);
    }

    private long _segmentsRewrittenFresh;
    private long _segmentsWrittenInPieces;
    private bool _closeAfterFlush;

    // Narrows entries[start..start + count], which do not archive as a whole, to the ones
    // that do not archive alone.
    private void FindRefused(SegmentEntry[] entries, int start, int count, string probe, List<SegmentEntry> refused)
    {
        if (count == 1)
        {
            refused.Add(entries[start]);
            return;
        }

        var half = count / 2;
        if (!Archives(entries, start, half, probe))
        {
            FindRefused(entries, start, half, probe, refused);
        }

        if (!Archives(entries, start + half, count - half, probe))
        {
            FindRefused(entries, start + half, count - half, probe, refused);
        }
    }

    private bool Archives(SegmentEntry[] entries, int start, int count, string probe)
    {
        if ((RefusedByArchiver is { } refused && entries.Skip(start).Take(count).Any(entry => refused(entry.Descriptor))) ||
            count > RefusesArchivesOver)
        {
            return false;
        }

        var archive = NewArchive(null);
        if (archive == 0)
        {
            return true;
        }

        try
        {
            for (var index = start; index < start + count; index++)
            {
                Add(archive, entries[index].Kind, entries[index].Descriptor);
            }

            return Serialize(archive, probe, null, quiet: true) >= 0;
        }
        finally
        {
            MetalNative.SendVoid(archive, SelRelease);
        }
    }

    // What tells a refused pipeline apart: its stages and the attachments it draws to.
    private static string Describe(SegmentEntry entry)
    {
        static string Name(nint descriptor, string selector)
        {
            var function = MetalNative.Send(descriptor, MetalNative.Selector(selector));
            return function == 0 ? "none" : MetalNative.ReadNsString(MetalNative.Send(function, MetalNative.Selector("name"))) ?? "?";
        }

        var descriptor = entry.Descriptor;
        var colors = MetalNative.Send(descriptor, MetalNative.Selector("colorAttachments"));
        var formats = new List<string>();
        for (nuint index = 0; index < 8; index++)
        {
            var format = MetalNative.SendUInteger(
                MetalNative.SendAtIndex(colors, MetalNative.Selector("objectAtIndexedSubscript:"), index),
                MetalNative.Selector("pixelFormat"));
            if (format != 0)
            {
                formats.Add($"{index}:{format}");
            }
        }

        return $"{entry.Kind} {(entry.Kind == PipelineKind.Mesh ? "mesh" : "vertex")}=" +
               $"{Name(descriptor, entry.Kind == PipelineKind.Mesh ? "meshFunction" : "vertexFunction")} " +
               $"fragment={Name(descriptor, "fragmentFunction")} colors=[{string.Join(' ', formats)}] " +
               $"depth={MetalNative.SendUInteger(descriptor, MetalNative.Selector("depthAttachmentPixelFormat"))} " +
               $"stencil={MetalNative.SendUInteger(descriptor, MetalNative.Selector("stencilAttachmentPixelFormat"))} " +
               $"samples={MetalNative.SendUInteger(descriptor, MetalNative.Selector(entry.Kind == PipelineKind.Mesh ? "rasterSampleCount" : "sampleCount"))}";
    }

    private void WriteLibraries(PipelineKind kind, nint descriptor, UInt128 first, UInt128 second)
    {
        if (kind != PipelineKind.Compute && !StageEntryPointsDiffer(kind, descriptor))
        {
            return;
        }

        var archive = NewArchive(null);
        if (archive == 0)
        {
            return;
        }

        try
        {
            if (!Add(archive, kind, descriptor))
            {
                return;
            }

            var target = first != default ? first : second;
            var path = LibraryPath(target);
            var bytes = Serialize(archive, path);
            if (bytes < 0)
            {
                return;
            }

            _known.TryAdd(target, 0);
            Interlocked.Increment(ref _librariesWritten);
            NoteWritten(bytes);
            if (first != default && second != default)
            {
                var copy = LibraryPath(second);
                File.Copy(path, copy, overwrite: true);
                _known.TryAdd(second, 0);
                Interlocked.Increment(ref _librariesWritten);
                NoteWritten(bytes);
            }
        }
        finally
        {
            MetalNative.SendVoid(archive, SelRelease);
        }
    }

    // A file's library is found by entry point name, so a stage whose name another stage
    // in the same pipeline shares cannot be told apart in it.
    private static bool StageEntryPointsDiffer(PipelineKind kind, nint descriptor)
    {
        var first = MetalNative.Send(descriptor, MetalNative.Selector(kind == PipelineKind.Mesh ? "meshFunction" : "vertexFunction"));
        var second = MetalNative.Send(descriptor, MetalNative.Selector("fragmentFunction"));
        if (first == 0 || second == 0)
        {
            return true;
        }

        return !string.Equals(
            MetalNative.ReadNsString(MetalNative.Send(first, MetalNative.Selector("name"))),
            MetalNative.ReadNsString(MetalNative.Send(second, MetalNative.Selector("name"))),
            StringComparison.Ordinal);
    }

    private bool Add(nint archive, PipelineKind kind, nint descriptor)
    {
        var selector = kind switch
        {
            PipelineKind.Compute => "addComputePipelineFunctionsWithDescriptor:error:",
            PipelineKind.Mesh => "addMeshRenderPipelineFunctionsWithDescriptor:error:",
            _ => "addRenderPipelineFunctionsWithDescriptor:error:",
        };
        nint error = 0;
        if (SendBoolError(archive, MetalNative.Selector(selector), descriptor, ref error))
        {
            return true;
        }

        Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] A {kind} pipeline could not be archived: {MetalNative.DescribeError(error)}");
        return false;
    }

    private void LoadSegments()
    {
        var segments = new List<(int Sequence, string Path)>();
        var foreign = new List<FileInfo>();
        foreach (var path in System.IO.Directory.EnumerateFiles(_directory, SegmentPrefix + "*" + Extension))
        {
            var name = Path.GetFileNameWithoutExtension(path).AsSpan(SegmentPrefix.Length);
            var dash = name.IndexOf('-');
            if (dash < 0 || !int.TryParse(name[(dash + 1)..], NumberStyles.None, CultureInfo.InvariantCulture, out var sequence))
            {
                TryDelete(path);
                continue;
            }

            if (!name[..dash].SequenceEqual(_translatorStamp.AsSpan()))
            {
                // Another translator's pipelines. The ones whose shaders it changed are never
                // asked for again and age out under the budget like any unused file.
                foreign.Add(new FileInfo(path));
                continue;
            }

            segments.Add((sequence, path));
        }

        segments.Sort(static (left, right) => right.Sequence.CompareTo(left.Sequence));
        _segmentSequence = segments.Count == 0 ? 0 : segments[0].Sequence + 1;

        // After this build's own, newest first: a pair asks its segments in list order.
        foreign.Sort(static (left, right) => right.LastWriteTimeUtc.CompareTo(left.LastWriteTimeUtc));
        foreach (var file in foreign)
        {
            int id;
            lock (_archivesLock)
            {
                id = -1 - _foreignSegments.Count;
                _foreignSegments[id] = file.FullName;
            }

            segments.Add((id, file.FullName));
        }

        long listedBytes = 0;
        var listed = 0;
        foreach (var (sequence, path) in segments)
        {
            if (!TryReadPairs(path, out var pairs, out var length))
            {
                // Not a segment of this format, or one written before segments listed their
                // pairs: nothing says what it holds.
                Interlocked.Increment(ref _rejected);
                TryDelete(path);
                continue;
            }

            lock (_archivesLock)
            {
                foreach (var pair in pairs)
                {
                    if (!_segmentsByPair.TryGetValue(pair, out var sequences))
                    {
                        _segmentsByPair[pair] = sequences = [];
                    }

                    // Newest first, as the segments are taken.
                    sequences.Add(sequence);
                }
            }

            listedBytes += length;
            listed++;
        }

        _loadedSegments = listed;
        _loadedSegmentBytes = listedBytes;
        Console.Error.WriteLine(
            $"[METAL-V2][ARCHIVE] opened {_directory} libraries={_known.Count} segments={listed} " +
            $"segment_mb={listedBytes / 1048576.0:F1} other_translator_segments={foreign.Count}");
    }

    // Under _archivesLock. Opens a segment for a lookup, closing the least recently used
    // past ResidentLimit (never the ones of the lookup: they are in wanted); zero when its
    // file does not check out any more.
    private nint OpenSegment(int sequence, List<int> wanted)
    {
        if (_resident.TryGetValue(sequence, out var open))
        {
            open.Tick = ++_residentTick;
            return open.Archive;
        }

        var path = SegmentFile(sequence);
        if (!TryVerify(path, out var length))
        {
            Interlocked.Increment(ref _rejected);
            DropSegment(sequence, path);
            return 0;
        }

        while (_residentBytes + length > ResidentLimit)
        {
            var victim = -1;
            var oldest = long.MaxValue;
            foreach (var (key, segment) in _resident)
            {
                if (segment.Tick < oldest && !wanted.Contains(key))
                {
                    victim = key;
                    oldest = segment.Tick;
                }
            }

            if (victim < 0)
            {
                break;
            }

            MetalNative.SendVoid(_resident[victim].Archive, SelRelease);
            _residentBytes -= _resident[victim].Bytes;
            _resident.Remove(victim);
        }

        var archive = OpenArchive(path);
        if (archive == 0)
        {
            Interlocked.Increment(ref _rejected);
            DropSegment(sequence, path);
            return 0;
        }

        _resident[sequence] = new ResidentSegment(archive, length) { Tick = ++_residentTick };
        _residentBytes += length;
        Interlocked.Increment(ref _segmentsOpened);
        Touch(path);
        return archive;
    }

    private void DropSegment(int sequence, string path)
    {
        TryDelete(path);
        foreach (var pair in _segmentsByPair.Where(entry => entry.Value.Contains(sequence)).Select(static entry => entry.Key).ToArray())
        {
            _segmentsByPair[pair].Remove(sequence);
            if (_segmentsByPair[pair].Count == 0)
            {
                _segmentsByPair.Remove(pair);
            }
        }
    }

    // The writer's and the static precompile's threads (StoreLibrary).
    private void NoteWritten(long bytes)
    {
        if (Interlocked.Add(ref _writtenSinceTrim, bytes) < _budget / 16)
        {
            return;
        }

        lock (_trimLock)
        {
            if (Interlocked.Read(ref _writtenSinceTrim) >= _budget / 16)
            {
                Interlocked.Exchange(ref _writtenSinceTrim, 0);
                Trim();
            }
        }
    }

    // Least recently used first: a library's file is touched when it is loaded, a segment
    // when it is written.
    private void Trim()
    {
        var files = new DirectoryInfo(_directory).GetFiles("*" + Extension, SearchOption.AllDirectories);
        var total = 0L;
        foreach (var file in files)
        {
            total += file.Length;
        }

        if (total <= _budget)
        {
            return;
        }

        var current = Path.GetFullPath(SegmentPath(_segmentSequence));
        Array.Sort(files, static (left, right) => left.LastWriteTimeUtc.CompareTo(right.LastWriteTimeUtc));
        foreach (var file in files)
        {
            if (total <= _budget - _budget / 10)
            {
                break;
            }

            if (string.Equals(file.FullName, current, StringComparison.Ordinal))
            {
                continue;
            }

            if (UInt128.TryParse(Path.GetFileNameWithoutExtension(file.Name), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var key))
            {
                _known.TryRemove(key, out _);
            }

            total -= file.Length;
            TryDelete(file.FullName);
        }
    }

    private nint NewArchive(string? path)
    {
        var descriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLBinaryArchiveDescriptor"), SelAlloc), SelInit);
        if (descriptor == 0)
        {
            return 0;
        }

        try
        {
            if (path is not null)
            {
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setUrl:"), FileUrl(path));
            }

            nint error = 0;
            return MetalNative.Send(_device, MetalNative.Selector("newBinaryArchiveWithDescriptor:error:"), descriptor, ref error);
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }
    }

    private nint OpenArchive(string path)
    {
        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            return NewArchive(path);
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    // Serializes to a temporary name, appends the trailer (and the pairs of a segment) and
    // moves it into place, so a reader never sees a half-written file. The final length, or -1.
    private long Serialize(nint archive, string path, ulong[]? pairs = null, bool quiet = false)
    {
        var temporary = $"{path}.{Environment.ProcessId.ToString(CultureInfo.InvariantCulture)}.tmp";
        nint error = 0;
        if (!SendBoolError(archive, MetalNative.Selector("serializeToURL:error:"), FileUrl(temporary), ref error))
        {
            if (!quiet)
            {
                Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] An archive could not be written: {MetalNative.DescribeError(error)}");
            }

            TryDelete(temporary);
            return -1;
        }

        if (!File.Exists(temporary))
        {
            // Success with no file: the archive holds what it held when it was last
            // serialized (the pipeline it was given again was in it already), and that file
            // is still there.
            return File.Exists(path) ? new FileInfo(path).Length : -1;
        }

        try
        {
            AppendTrailer(temporary, pairs);
            File.Move(temporary, path, overwrite: true);
            return new FileInfo(path).Length;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[METAL-V2][ARCHIVE][WARN] An archive could not be written: {exception.Message}");
            TryDelete(temporary);
            return -1;
        }
    }

    /// <summary>
    /// Appends the trailer that <see cref="TryVerify"/> checks to a serialized archive; a
    /// segment's first lists the shader pairs of its pipelines (<see cref="TryReadPairs"/>) in
    /// the bytes the trailer covers.
    /// </summary>
    internal static void AppendTrailer(string path, IReadOnlyCollection<ulong>? pairs = null)
    {
        var file = File.ReadAllBytes(path);
        var payload = file;
        if (pairs is { Count: > 0 })
        {
            payload = new byte[file.Length + pairs.Count * 8 + 8];
            file.CopyTo(payload, 0);
            var tail = payload.AsSpan(file.Length);
            var offset = 0;
            foreach (var pair in pairs)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(tail[offset..], pair);
                offset += 8;
            }

            BinaryPrimitives.WriteUInt32LittleEndian(tail[offset..], (uint)pairs.Count);
            BinaryPrimitives.WriteUInt32LittleEndian(tail[(offset + 4)..], IndexMagic);
        }

        Span<byte> trailer = stackalloc byte[TrailerBytes];
        WriteTrailer(payload, trailer);
        using var stream = new FileStream(path, FileMode.Append, FileAccess.Write);
        stream.Write(payload.AsSpan(file.Length));
        stream.Write(trailer);
    }

    /// <summary>
    /// The shader pairs a segment lists, read from the end of the file without checking the
    /// rest (<see cref="TryVerify"/> does when the segment is opened); false when the tail is
    /// not a trailer after a list.
    /// </summary>
    internal static bool TryReadPairs(string path, out ulong[] pairs, out long length)
    {
        pairs = [];
        length = 0;
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            length = stream.Length;
            Span<byte> tail = stackalloc byte[TrailerBytes + 8];
            if (length < tail.Length)
            {
                return false;
            }

            stream.Seek(-tail.Length, SeekOrigin.End);
            stream.ReadExactly(tail);
            var trailer = tail[8..];
            if (BinaryPrimitives.ReadUInt32LittleEndian(trailer) != TrailerMagic ||
                BinaryPrimitives.ReadUInt32LittleEndian(trailer[4..]) != FormatVersion ||
                BinaryPrimitives.ReadInt64LittleEndian(trailer[8..]) != length - TrailerBytes ||
                BinaryPrimitives.ReadUInt32LittleEndian(tail[4..]) != IndexMagic)
            {
                return false;
            }

            var count = BinaryPrimitives.ReadUInt32LittleEndian(tail);
            if (count == 0 || count * 8L + tail.Length > length)
            {
                return false;
            }

            var keys = new byte[count * 8];
            stream.Seek(-(tail.Length + keys.Length), SeekOrigin.End);
            stream.ReadExactly(keys);
            pairs = new ulong[count];
            for (var index = 0; index < pairs.Length; index++)
            {
                pairs[index] = BinaryPrimitives.ReadUInt64LittleEndian(keys.AsSpan(index * 8));
            }

            return true;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    internal static void WriteTrailer(ReadOnlySpan<byte> payload, Span<byte> trailer)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(trailer, TrailerMagic);
        BinaryPrimitives.WriteUInt32LittleEndian(trailer[4..], FormatVersion);
        BinaryPrimitives.WriteInt64LittleEndian(trailer[8..], payload.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(trailer[16..], XxHash64.HashToUInt64(payload));
    }

    /// <summary>Whether a file is a whole archive this format wrote; its length when it is.</summary>
    // The segment is hashed through a pooled buffer rather than read whole: segments run to
    // tens of megabytes, and one array per segment opened during play was a large object each
    // (Black Myth: Wukong's full collections).
    internal static bool TryVerify(string path, out long length)
    {
        length = 0;
        var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(1 << 20);
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1);
            var total = stream.Length;
            if (total <= TrailerBytes)
            {
                return false;
            }

            Span<byte> trailer = stackalloc byte[TrailerBytes];
            stream.Seek(total - TrailerBytes, SeekOrigin.Begin);
            stream.ReadExactly(trailer);
            var payloadLength = total - TrailerBytes;
            if (BinaryPrimitives.ReadUInt32LittleEndian(trailer) != TrailerMagic ||
                BinaryPrimitives.ReadUInt32LittleEndian(trailer[4..]) != FormatVersion ||
                BinaryPrimitives.ReadInt64LittleEndian(trailer[8..]) != payloadLength)
            {
                return false;
            }

            stream.Seek(0, SeekOrigin.Begin);
            var hash = new XxHash64();
            for (var remaining = payloadLength; remaining > 0;)
            {
                var chunk = buffer.AsSpan(0, (int)Math.Min(remaining, buffer.Length));
                stream.ReadExactly(chunk);
                hash.Append(chunk);
                remaining -= chunk.Length;
            }

            if (BinaryPrimitives.ReadUInt64LittleEndian(trailer[16..]) != hash.GetCurrentHashAsUInt64())
            {
                return false;
            }

            length = total;
            return true;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return false;
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // Autoreleased.
    private static nint FileUrl(string path) =>
        MetalNative.Send(MetalNative.Class("NSURL"), MetalNative.Selector("fileURLWithPath:"), MetalNative.NsString(path));

    private static void Touch(string path)
    {
        try
        {
            File.SetLastWriteTimeUtc(path, DateTime.UtcNow);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // Only the eviction order depends on it.
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // Another start may hold it; the next trim tries again.
        }
    }

    private static void TryDeleteDirectory(string path)
    {
        try
        {
            System.IO.Directory.Delete(path, recursive: true);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // The next start tries again.
        }
    }

    private static string Hex64(ulong value) => value.ToString("x16", CultureInfo.InvariantCulture);

    [LibraryImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    [return: MarshalAs(UnmanagedType.I1)]
    private static partial bool SendBoolError(nint receiver, nint selector, nint argument, ref nint error);
}
