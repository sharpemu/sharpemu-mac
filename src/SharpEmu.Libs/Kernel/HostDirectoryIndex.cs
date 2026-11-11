// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Collections.Frozen;
using System.Diagnostics;
using System.IO.Enumeration;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.Libs.Ampr;

namespace SharpEmu.Libs.Kernel;

/// <summary>
/// Per-directory metadata index over the host paths a mount resolves to.
/// </summary>
/// <remarks>
/// Resolving a guest path has to answer three questions about every path
/// component: does it exist, is it a symlink (which would escape the mount),
/// and — for the leaf — how large is it. Asking the host one component at a
/// time costs one <c>lstat</c> per component, and the game image lives on an
/// external volume where a cold random <c>lstat</c> measures ~180 us. Demon's
/// Souls resolves ~222k distinct asset paths during boot, so that is ~40 s of
/// pure metadata latency.
///
/// Listing the component's parent directory instead answers the same questions
/// for every sibling at once, and a <c>readdir</c> followed by stats in
/// directory order measures ~37 us per entry on the same volume — 5x cheaper
/// per file, and the cost is paid once per directory rather than once per
/// asset. Asset tables address files in directory-sized clusters, so one
/// listing typically serves hundreds of resolutions.
///
/// A listing is only authoritative about an absent name if it covers every
/// spelling the host filesystem would accept, so case folding is probed per
/// volume with <c>pathconf(_PC_CASE_SENSITIVE)</c>. When the probe cannot
/// answer, a name that is not in the listing reports <see cref="Verdict.Unknown"/>
/// and the caller falls back to its own per-component stat.
/// </remarks>
internal static partial class HostDirectoryIndex
{
    internal enum Verdict
    {
        /// <summary>The index cannot answer; the caller must stat the path itself.</summary>
        Unknown,
        /// <summary>The parent directory was listed and does not contain the name.</summary>
        Absent,
        Directory,
        File,
        /// <summary>A symlink (any type). Resolving through it would leave the mount.</summary>
        Symlink,
    }

    internal readonly record struct Entry(Verdict Verdict, long Length);

    // Guards total memory: a pathological root (or a guest walking a host-wide
    // mount) must not grow this without bound. Demon's Souls needs ~223k.
    private const int MaxIndexedEntries = 1 << 21;

    // A directory this large costs more to list than the handful of files a
    // caller is likely to want from it, so it is left to per-component stats.
    private const int MaxDirectoryEntries = 1 << 17;

    private sealed class DirectoryListing
    {
        public required FrozenDictionary<string, Entry> Ordinal { get; init; }

        /// <summary>Non-null only where the host filesystem folds case.</summary>
        public FrozenDictionary<string, Entry>? IgnoreCase { get; init; }

        /// <summary>
        /// False when the listing could not be completed (I/O error, entry cap),
        /// in which case a missing name means "not known" rather than "absent".
        /// </summary>
        public required bool Complete { get; init; }

        public static readonly DirectoryListing Incomplete = new()
        {
            Ordinal = FrozenDictionary<string, Entry>.Empty,
            Complete = false,
        };
    }

    private static readonly ConcurrentDictionary<string, DirectoryListing> _directories =
        new(HostFsPath.Comparer);

    // Volume root (or the deepest probed directory) -> whether names fold case.
    private static readonly ConcurrentDictionary<string, bool> _caseFolding =
        new(HostFsPath.Comparer);

    private static int _indexedEntries;

    internal static bool Enabled { get; } = !string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_DISABLE_DIR_INDEX"),
        "1",
        StringComparison.Ordinal);

    /// <summary>
    /// Describes <paramref name="hostPath"/> from its parent directory's listing,
    /// building that listing on first use.
    /// </summary>
    internal static Entry Describe(string hostPath)
    {
        if (!Enabled || string.IsNullOrEmpty(hostPath))
        {
            return default;
        }

        if (TryDescribeFromCachedListing(hostPath, out var described))
        {
            return described;
        }

        string? directory;
        string name;
        try
        {
            directory = Path.GetDirectoryName(hostPath);
            name = Path.GetFileName(hostPath);
        }
        catch (ArgumentException)
        {
            return default;
        }

        if (string.IsNullOrEmpty(directory) || name.Length == 0)
        {
            // A volume root has no parent to list, and the mount roots
            // themselves are verified by the caller's containment check.
            return default;
        }

        var listing = GetListing(directory);
        if (listing.Ordinal.TryGetValue(name, out var entry))
        {
            return entry;
        }

        if (listing.IgnoreCase is { } folded && folded.TryGetValue(name, out entry))
        {
            return entry;
        }

        return listing.Complete ? new Entry(Verdict.Absent, 0) : default;
    }

    /// <summary>
    /// <see cref="Describe"/> for a path whose parent is already listed, looked up
    /// by slices of <paramref name="hostPath"/> instead of new directory and name
    /// strings: the path resolver asks this once per asset it resolves.
    /// </summary>
    private static bool TryDescribeFromCachedListing(string hostPath, out Entry entry)
    {
        entry = default;
        // GetDirectoryName(string) also collapses repeated separators (and flips
        // them on Windows); only a path it would return unchanged is sliced.
        ReadOnlySpan<char> doubleSeparator = ['/', '/'];
        if (OperatingSystem.IsWindows() || hostPath.AsSpan().IndexOf(doubleSeparator) >= 0)
        {
            return false;
        }

        var directory = Path.GetDirectoryName(hostPath.AsSpan());
        var name = Path.GetFileName(hostPath.AsSpan());
        if (directory.IsEmpty || name.IsEmpty ||
            !_directories.GetAlternateLookup<ReadOnlySpan<char>>().TryGetValue(directory, out var listing))
        {
            return false;
        }

        if (listing.Ordinal.GetAlternateLookup<ReadOnlySpan<char>>().TryGetValue(name, out entry))
        {
            return true;
        }

        if (listing.IgnoreCase is { } folded &&
            folded.GetAlternateLookup<ReadOnlySpan<char>>().TryGetValue(name, out entry))
        {
            return true;
        }

        entry = listing.Complete ? new Entry(Verdict.Absent, 0) : default;
        return true;
    }

    /// <summary>
    /// Drops the listing that would describe <paramref name="hostPath"/>, after a
    /// create, delete, rename or truncate changed what the host holds there.
    /// </summary>
    internal static void Invalidate(string hostPath)
    {
        if (!Enabled || string.IsNullOrEmpty(hostPath))
        {
            return;
        }

        try
        {
            var directory = Path.GetDirectoryName(hostPath);
            if (!string.IsNullOrEmpty(directory) &&
                _directories.TryRemove(directory, out var removed))
            {
                Interlocked.Add(ref _indexedEntries, -removed.Ordinal.Count);
            }

            // The path itself may be a directory whose own listing is now stale.
            if (_directories.TryRemove(hostPath, out var self))
            {
                Interlocked.Add(ref _indexedEntries, -self.Ordinal.Count);
            }
        }
        catch (ArgumentException)
        {
            // An unusable path indexes nothing, so there is nothing to drop.
        }
    }

    private static int _warmStarted;

    /// <summary>
    /// Lists every directory under <paramref name="hostRoot"/> on a background
    /// thread so the guest's first resolve into each one is a dictionary probe.
    /// </summary>
    /// <remarks>
    /// Building a listing on demand moves the cold directory-metadata read onto
    /// whichever guest thread resolves the first path in that directory, which
    /// put ~4 s of external-volume latency on the engine's initialization phase.
    /// The work is the same either way (and strictly less than the per-file
    /// stats it replaces), so it runs here, overlapped with module load, rather
    /// than in front of the guest.
    /// </remarks>
    internal static void BeginWarm(string? hostRoot, bool persistSnapshot = false)
    {
        if (!Enabled || string.IsNullOrWhiteSpace(hostRoot) || !Directory.Exists(hostRoot))
        {
            return;
        }

        if (Interlocked.Exchange(ref _warmStarted, 1) != 0)
        {
            return;
        }

        if (persistSnapshot)
        {
            _snapshotPhase.Reset();
        }

        // A dedicated thread, not the pool: the walk blocks on device latency for
        // seconds, and a pool worker parked that long makes the pool inject
        // threads for everything else queued behind it.
        var warmer = new Thread(static state =>
        {
            var (root, persist) = ((string, bool))state!;
            Warm(root, persist);
        })
        {
            IsBackground = true,
            Name = "dir-index-warm",
            Priority = ThreadPriority.BelowNormal,
        };
        warmer.Start((hostRoot, persistSnapshot));
    }

    // setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_THROTTLE): the
    // kernel defers this thread's disk requests behind foreground ones.
    private const int IopolTypeDisk = 0;
    private const int IopolScopeThread = 1;
    private const int IopolThrottle = 1;

    [LibraryImport("libc", EntryPoint = "setiopolicy_np")]
    private static partial int SetIoPolicy(int iotype, int scope, int policy);

    private static void ThrottleThreadIo()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        try
        {
            // The game image is on a slow external volume, and the warm walk
            // reads metadata the guest has not asked for yet. Letting it queue
            // ahead of the asset reads the guest is blocked on trades startup
            // latency for index coverage nobody needed yet.
            _ = SetIoPolicy(IopolTypeDisk, IopolScopeThread, IopolThrottle);
        }
        catch (Exception exception) when (
            exception is DllNotFoundException or EntryPointNotFoundException)
        {
            // Without the policy the walk simply competes normally.
        }
    }

    // Unset from BeginWarm until the snapshot has been seeded or ruled out.
    private static readonly ManualResetEventSlim _snapshotPhase = new(initialState: true);

    /// <summary>
    /// Blocks until the persisted app0 listings are seeded (or known to be
    /// unusable), so the guest's first resolves find them in place.
    /// </summary>
    /// <returns>False when <paramref name="timeout"/> expired first.</returns>
    internal static bool WaitForSnapshot(TimeSpan timeout) => _snapshotPhase.Wait(timeout);

    private static void Warm(string root, bool persistSnapshot)
    {
        try
        {
            WarmWithSnapshot(root, persistSnapshot);
        }
        finally
        {
            _snapshotPhase.Set();
        }
    }

    private static void WarmWithSnapshot(string root, bool persistSnapshot)
    {
        string? snapshotPath = null;
        long fingerprint = 0;
        // A writable app0 changes under the index without touching the dump
        // fingerprint, so only the read-only image may be described from disk.
        if (persistSnapshot && !KernelMemoryCompatExports.IsApp0Writable)
        {
            try
            {
                snapshotPath = AmprFileRegistry.GetDirectoryIndexSnapshotPath(root);
                fingerprint = AmprFileRegistry.GetDumpFingerprint(root);
            }
            catch (Exception exception) when (
                exception is IOException or UnauthorizedAccessException or ArgumentException)
            {
                snapshotPath = null;
            }
        }

        try
        {
            _ = WarmCore(root, snapshotPath, fingerprint);
        }
        catch (Exception exception)
        {
            // Warming is opportunistic: a dump that moves or a mount that
            // hiccups must leave resolution working, just without the index.
            Console.Error.WriteLine(
                $"[LOADER][WARN] dir_index.warm_failed root={root}: {exception.Message}");
        }
    }

    /// <summary>
    /// Lists every directory under <paramref name="root"/>, or seeds those
    /// listings from the snapshot a previous walk of the same dump saved.
    /// </summary>
    /// <returns>True when the listings came from the snapshot.</returns>
    /// <remarks>
    /// Listing the tree is ~20k directory reads and one stat per file (~223k for
    /// Demon's Souls) on the external game volume. The guest's resource scan
    /// needs those answers while the same volume streams its first assets, and
    /// whenever the host had dropped the metadata from its cache the scan paid
    /// that contention on every path: GatherResourceFileInfo took 33-80 s
    /// instead of ~9 s. The image is read-only, so the answers a complete walk
    /// produced stay valid for as long as the dump fingerprint the app0 id
    /// cache trusts does.
    /// </remarks>
    internal static bool WarmCore(string root, string? snapshotPath, long fingerprint)
    {
        root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
        var seeded = snapshotPath is not null && TryLoadSnapshot(root, snapshotPath, fingerprint);
        _snapshotPhase.Set();
        if (seeded)
        {
            return true;
        }

        ThrottleThreadIo();
        var walked = snapshotPath is null ? null : new List<(string Relative, DirectoryListing Listing)>();
        var complete = true;
        // Breadth-first so the shallow directories a title opens first are ready
        // before the deep asset trees it streams later.
        var pending = new Queue<(string Directory, string Relative)>();
        pending.Enqueue((root, string.Empty));
        while (pending.Count > 0)
        {
            if (Volatile.Read(ref _indexedEntries) >= MaxIndexedEntries)
            {
                complete = false;
                break;
            }

            var (directory, relative) = pending.Dequeue();
            var listing = GetListing(directory);
            complete &= listing.Complete;
            walked?.Add((relative, listing));
            foreach (var (name, entry) in listing.Ordinal)
            {
                if (entry.Verdict == Verdict.Directory)
                {
                    pending.Enqueue((
                        Path.Combine(directory, name),
                        relative.Length == 0 ? name : relative + "/" + name));
                }
            }
        }

        if (walked is not null && complete)
        {
            TrySaveSnapshot(root, snapshotPath!, fingerprint, walked);
        }

        return false;
    }

    private const uint SnapshotMagic = 0x31584944u; // 'DIX1'
    private const uint SnapshotVersion = 1;

    private static bool TryLoadSnapshot(string root, string snapshotPath, long fingerprint)
    {
        if (fingerprint == 0 ||
            string.Equals(
                Environment.GetEnvironmentVariable("SHARPEMU_AMPR_REINDEX"),
                "1",
                StringComparison.Ordinal) ||
            !File.Exists(snapshotPath))
        {
            return false;
        }

        var stopwatch = Stopwatch.StartNew();
        var parsed = new List<(string Directory, DirectoryListing Listing)>();
        var total = 0;
        try
        {
            using var stream = new FileStream(
                snapshotPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                1 << 16,
                FileOptions.SequentialScan);
            using var reader = new BinaryReader(stream, Encoding.UTF8);
            if (reader.ReadUInt32() != SnapshotMagic ||
                reader.ReadUInt32() != SnapshotVersion ||
                !string.Equals(reader.ReadString(), root, HostFsPath.Comparison) ||
                reader.ReadInt64() != fingerprint)
            {
                return false;
            }

            var directoryCount = reader.ReadInt32();
            if (directoryCount <= 0 || directoryCount > MaxIndexedEntries)
            {
                return false;
            }

            for (var i = 0; i < directoryCount; i++)
            {
                var relative = reader.ReadString();
                var foldsCase = reader.ReadBoolean();
                var entryCount = reader.ReadInt32();
                total += entryCount;
                if (!IsSnapshotRelative(relative) ||
                    entryCount < 0 ||
                    entryCount > MaxDirectoryEntries ||
                    total > MaxIndexedEntries)
                {
                    return false;
                }

                var entries = new Dictionary<string, Entry>(entryCount, HostFsPath.Comparer);
                for (var j = 0; j < entryCount; j++)
                {
                    var name = reader.ReadString();
                    var verdict = (Verdict)reader.ReadByte();
                    var length = reader.ReadInt64();
                    if (!IsSnapshotName(name) ||
                        verdict is not (Verdict.Directory or Verdict.File or Verdict.Symlink) ||
                        length < 0)
                    {
                        return false;
                    }

                    entries[name] = new Entry(verdict, length);
                }

                var directory = relative.Length == 0
                    ? root
                    : Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar));
                parsed.Add((directory, CreateListing(entries, foldsCase)));
            }

            // A snapshot cut short must not seed half a tree whose missing
            // directories would then be listed as if they did not exist.
            if (reader.ReadUInt32() != SnapshotMagic)
            {
                return false;
            }
        }
        catch (Exception exception) when (
            exception is IOException or UnauthorizedAccessException or
                ArgumentException or NotSupportedException or EndOfStreamException)
        {
            Console.Error.WriteLine(
                $"[LOADER][WARN] dir_index.snapshot_load_failed path={snapshotPath}: {exception.Message}");
            return false;
        }

        // The host volume may have been reformatted or the dump copied under the
        // same path; a case-folding answer that no longer holds would turn
        // misspelled lookups into wrong hits or misses.
        var rootFolds = parsed.Count > 0 && parsed[0].Directory == root && parsed[0].Listing.IgnoreCase is not null;
        if (FoldsCase(root) != rootFolds)
        {
            return false;
        }

        foreach (var (directory, listing) in parsed)
        {
            // A listing the guest built while this loaded is equivalent; keep it.
            if (_directories.TryAdd(directory, listing))
            {
                Interlocked.Add(ref _indexedEntries, listing.Ordinal.Count);
            }
        }

        Console.Error.WriteLine(
            $"[LOADER][INFO] dir_index.snapshot_hit root={root} dirs={parsed.Count} entries={total} " +
            $"elapsed_ms={stopwatch.Elapsed.TotalMilliseconds:F1}");
        return true;
    }

    private static void TrySaveSnapshot(
        string root,
        string snapshotPath,
        long fingerprint,
        List<(string Relative, DirectoryListing Listing)> directories)
    {
        if (fingerprint == 0)
        {
            return;
        }

        var tempPath = $"{snapshotPath}.{Environment.ProcessId}.tmp";
        var total = 0;
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16))
            using (var writer = new BinaryWriter(stream, Encoding.UTF8))
            {
                writer.Write(SnapshotMagic);
                writer.Write(SnapshotVersion);
                writer.Write(root);
                writer.Write(fingerprint);
                writer.Write(directories.Count);
                foreach (var (relative, listing) in directories)
                {
                    writer.Write(relative);
                    writer.Write(listing.IgnoreCase is not null);
                    writer.Write(listing.Ordinal.Count);
                    foreach (var (name, entry) in listing.Ordinal)
                    {
                        writer.Write(name);
                        writer.Write((byte)entry.Verdict);
                        writer.Write(entry.Length);
                    }

                    total += listing.Ordinal.Count;
                }

                writer.Write(SnapshotMagic);
            }

            File.Move(tempPath, snapshotPath, overwrite: true);
            Console.Error.WriteLine(
                $"[LOADER][INFO] dir_index.snapshot_saved path={snapshotPath} dirs={directories.Count} entries={total}");
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            try
            {
                File.Delete(tempPath);
            }
            catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException)
            {
                // Nothing more to do; the next walk overwrites it.
            }

            Console.Error.WriteLine(
                $"[LOADER][WARN] dir_index.snapshot_save_failed path={snapshotPath}: {exception.Message}");
        }
    }

    // A snapshot names entries relative to the root it was taken of; anything
    // that could climb out of that root or alias another entry is refused.
    private static bool IsSnapshotName(string name) =>
        name.Length is > 0 and <= MaxHostPathComponentLength &&
        name is not ("." or "..") &&
        name.IndexOf('\0') < 0 &&
        name.IndexOf(Path.DirectorySeparatorChar) < 0 &&
        name.IndexOf(Path.AltDirectorySeparatorChar) < 0;

    private static bool IsSnapshotRelative(string relative)
    {
        if (relative.Length == 0)
        {
            return true;
        }

        foreach (var segment in relative.Split('/'))
        {
            if (!IsSnapshotName(segment))
            {
                return false;
            }
        }

        return true;
    }

    private const int MaxHostPathComponentLength = 255;

    /// <summary>Test hook: forget every listing and case-folding probe.</summary>
    internal static void ClearForTests()
    {
        _directories.Clear();
        _caseFolding.Clear();
        Volatile.Write(ref _indexedEntries, 0);
        Volatile.Write(ref _warmStarted, 0);
        _snapshotPhase.Set();
    }

    private static DirectoryListing GetListing(string directory)
    {
        if (_directories.TryGetValue(directory, out var cached))
        {
            return cached;
        }

        var listing = BuildListing(directory);
        // A racing build of the same directory just loses its copy; both are
        // equivalent, and double-counting the cap is corrected on the discard.
        if (_directories.TryAdd(directory, listing))
        {
            Interlocked.Add(ref _indexedEntries, listing.Ordinal.Count);
            return listing;
        }

        return _directories.TryGetValue(directory, out cached) ? cached : listing;
    }

    private static DirectoryListing BuildListing(string directory)
    {
        if (Volatile.Read(ref _indexedEntries) >= MaxIndexedEntries)
        {
            return DirectoryListing.Incomplete;
        }

        var entries = new Dictionary<string, Entry>(HostFsPath.Comparer);
        try
        {
            var enumerable = new FileSystemEnumerable<(string Name, Entry Entry)>(
                directory,
                static (ref FileSystemEntry entry) =>
                {
                    var attributes = entry.Attributes;
                    var verdict = (attributes & FileAttributes.ReparsePoint) != 0
                        ? Verdict.Symlink
                        : entry.IsDirectory
                            ? Verdict.Directory
                            : Verdict.File;
                    var length = verdict == Verdict.File ? entry.Length : 0;
                    return (entry.FileName.ToString(), new Entry(verdict, length));
                },
                new EnumerationOptions
                {
                    RecurseSubdirectories = false,
                    // The guest filesystem has no notion of hidden or system
                    // entries, so every name the host holds must be visible.
                    AttributesToSkip = 0,
                    IgnoreInaccessible = true,
                    ReturnSpecialDirectories = false,
                });

            foreach (var (name, entry) in enumerable)
            {
                if (entries.Count >= MaxDirectoryEntries)
                {
                    return DirectoryListing.Incomplete;
                }

                // A case-folding host can only hand back one spelling per name,
                // so a collision here means the host is case-sensitive and both
                // files are real. Keeping the first is enough: the ordinal map
                // holds each spelling under its own key.
                entries[name] = entry;
            }
        }
        catch (Exception exception) when (
            exception is IOException or UnauthorizedAccessException or
                ArgumentException or NotSupportedException)
        {
            // The directory may not exist, may be a file, or may be unreadable.
            // None of that is an error here: the caller stats the path itself.
            return DirectoryListing.Incomplete;
        }

        return CreateListing(entries, FoldsCase(directory));
    }

    private static DirectoryListing CreateListing(Dictionary<string, Entry> entries, bool foldsCase)
    {
        var ordinal = entries.ToFrozenDictionary(HostFsPath.Comparer);
        FrozenDictionary<string, Entry>? ignoreCase = null;
        if (foldsCase)
        {
            var folded = new Dictionary<string, Entry>(
                entries.Count,
                StringComparer.OrdinalIgnoreCase);
            foreach (var (name, entry) in entries)
            {
                folded[name] = entry;
            }

            ignoreCase = folded.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);
        }

        return new DirectoryListing
        {
            Ordinal = ordinal,
            IgnoreCase = ignoreCase,
            Complete = true,
        };
    }

    // POSIX _PC_CASE_SENSITIVE as macOS numbers it. Linux does not define the
    // name at all, which pathconf reports as EINVAL (-1).
    private const int PcCaseSensitive = 11;

    [LibraryImport("libc", EntryPoint = "pathconf", StringMarshalling = StringMarshalling.Utf8)]
    private static partial long PathConf(string path, int name);

    /// <summary>
    /// Whether the volume holding <paramref name="directory"/> folds name case.
    /// Probed once per directory and memoized; an unanswerable probe reports
    /// false, which only costs the caller an extra stat on a case mismatch.
    /// </summary>
    internal static bool FoldsCase(string directory)
    {
        if (OperatingSystem.IsWindows())
        {
            return true;
        }

        if (_caseFolding.TryGetValue(directory, out var folds))
        {
            return folds;
        }

        folds = false;
        if (OperatingSystem.IsMacOS())
        {
            try
            {
                // 0 = names are folded, 1 = distinct by case, -1 = no answer.
                folds = PathConf(directory, PcCaseSensitive) == 0;
            }
            catch (Exception exception) when (
                exception is DllNotFoundException or EntryPointNotFoundException)
            {
                folds = false;
            }
        }

        _caseFolding[directory] = folds;
        return folds;
    }
}
