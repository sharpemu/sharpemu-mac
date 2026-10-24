// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Logging;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// Which guest memory windows an encoder that runs a device-address program names resident.
//
// Such a program can dereference any mapped guest page, so the safe answer is every window
// the address table spans - 11 GiB in Demon's Souls in-world. The driver wires every page of
// a named window, touched or not, and that much on top of the image cache does not fit under
// the machine's wire limit. The programs reach far less.
//
// So the encoder names the windows holding the granules the title's programs have been seen
// to reach, kept per title next to the shader cache. Every table entry in a window outside
// that set carries its granule in its page count's upper half; a lookup that lands in one
// reports the granule in a feedback bitmap, and the host names the windows holding that
// granule from the next submission on and adds it to the title's profile. The access itself
// still goes through: the window's pages stay mapped for the life of the buffer (a window a
// cut replaced is kept until the submission that replaced it completes), so performing it
// keeps the result the guest expects, where a deterministic zero would return wrong data on
// every first touch. With 1 GiB windows a cold Demon's Souls run missed only while it learned
// (none after ~280 s); the learning is now per granule (32 MiB by default), so a reach that
// touches a few pages of a gigabyte no longer names the whole gigabyte. The profile is kept
// in granules rather than windows because windows move. Misses are counted in the PERF line
// (da_misses). SHARPEMU_METAL_V2_DA_REACH=0 names every window the table spans instead.
internal sealed partial class MetalRenderHost
{
    private static readonly bool DeviceAddressReachEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DA_REACH") != "0";

    private MetalDeviceAddressReach? _reach;

    // The granules an encoder running a device-address program reaches this submission: the
    // table's granules, narrowed to the reach set when it is on; and the windows it names,
    // those the table's entries in those granules lie in. The table's flags were written
    // against the same windows.
    private readonly MetalWindowSet _deviceAddressNamed = new();
    private readonly MetalWindowSet _deviceAddressNamedWindows = new();

    private MetalDeviceAddressReach Reach => _reach ??= OpenReach();

    // The title's directory next to its shader cache: the reach profile and the single-copy kill switch.
    private static string TitleProfileDirectory() =>
        SharpEmuPaths.UserPath(
            "pipeline_cache",
            MetalShaderPrewarm.SanitizeTitleId(VideoOutBackendServices.GetApplicationTitleId()),
            "metal-v2");

    private MetalDeviceAddressReach OpenReach()
    {
        var profile = MetalDeviceAddressReachProfile.Open(TitleProfileDirectory(), _guestMemory.GranuleBytes);
        var reach = new MetalDeviceAddressReach(_guestMemory, profile);
        if (profile is not null)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][INFO] device-address reach seeded: granules={reach.Reached.Count} " +
                $"granule_mb={_guestMemory.GranuleBytes >> 20} reach_mb={_guestMemory.GranuleSetBytes(reach.Reached) >> 20}");
        }

        return reach;
    }

    /// <summary>The feedback buffer a device-address program reports misses into.</summary>
    private ulong ReachFeedbackAddress() => Reach.FeedbackAddress;

    /// <summary>
    /// Takes the granules completed or running submissions reported reaching outside their
    /// named set, names them from now on, and settles the windows this submission names,
    /// whose cuts then stay until it ends. Called once per submission, as its table is
    /// written, so the table's flags and the windows its encoders name agree.
    /// </summary>
    private void BeginDeviceAddressReach()
    {
        var table = _deviceAddressTableBuilder;
        if (DeviceAddressReachEnabled)
        {
            Reach.TakeFeedback();
            _deviceAddressNamed.SetIntersection(table.Granules, Reach.Reached);
            _deviceAddressNamedWindows.Clear();
            table.AddWindowsOfGranules(_deviceAddressNamed, _deviceAddressNamedWindows);
        }
        else
        {
            _deviceAddressNamed.CopyFrom(table.Granules);
            _deviceAddressNamedWindows.CopyFrom(table.Windows);
        }

        _guestMemory.PinWindows(_deviceAddressNamedWindows);
        if (DeviceAddressBufferReuse)
            ResolveDeviceAddressBuffers();
    }

    /// <summary>The windows an encoder that runs a device-address program names.</summary>
    private MetalWindowSet DeviceAddressNamedWindows() => _deviceAddressNamedWindows;

    private string TakeDeviceAddressReachCounters()
    {
        var (misses, learned) = _reach?.TakeCounters() ?? (0, 0);
        return string.Create(
            CultureInfo.InvariantCulture,
            $" da_misses={misses} da_learned={learned} da_named={_deviceAddressNamedWindows.Count} " +
            $"da_named_mb={_guestMemory.WindowSetBytes(_deviceAddressNamedWindows) >> 20} " +
            $"da_buffer_builds={Interlocked.Exchange(ref _deviceAddressBufferBuilds, 0)} " +
            $"da_buffer_reuses={Interlocked.Exchange(ref _deviceAddressBufferReuses, 0)} " +
            $"da_buffer_lookups={Interlocked.Exchange(ref _deviceAddressBufferLookups, 0)}");
    }

    // The device-address side of the [MEM] singlecopy section: the table's trap entries and the
    // stages that store with no evaluated range.
    private string DescribeSingleCopyDeviceAddresses()
    {
        var table = _deviceAddressTableBuilder;
        return string.Create(
            CultureInfo.InvariantCulture,
            $" device_address_writes={DeviceAddressWrites} da_table_entries={table.Count} da_trap_entries={table.TrapEntryCount} " +
            $"da_trap_mb={table.TrapBytes >> 20} da_trap_unflagged={table.TrapUnflagged} " +
            $"da_trap_overflowed={(table.TrapOverflowed ? 1 : 0)}");
    }
}

/// <summary>
/// The device-address reach of a title: the granules its programs were seen to reach, and
/// the feedback words the programs report into (resource_prelude.msl): word 0 counts lookups
/// that landed outside the named set; word 1 counts accesses to single-copy trap entries
/// (MetalDeviceAddressTable.TrapFlag), words 2 and 3 locate the bitmap of backing pages they
/// reached (first word, word count) and word 4 is set in the shadow mode, where such an
/// access goes through; from word 8 one bit per granule, then the trap bitmap.
/// </summary>
internal sealed class MetalDeviceAddressReach
{
    internal const int MissCountWord = 0;
    internal const int TrapCountWord = 1;
    internal const int TrapBitmapWord = 2;
    internal const int TrapBitmapCountWord = 3;
    internal const int TrapShadowWord = 4;
    internal const int HeaderWords = 8;

    // Submissions whose trap bitmap words may land after their count was taken: the bitmap is
    // scanned for that many more takes after a count.
    private const int TrapRescans = 8;

    private readonly MetalGuestMemory _memory;
    private readonly MetalDeviceAddressReachProfile? _profile;
    private readonly nint _contents;
    private readonly int _bitmapWords;
    private readonly int _trapFirstWord;
    private readonly int _trapWords;
    private readonly List<ulong> _learnedGranules = new();
    private long _misses;
    private long _learned;
    private int _trapRescans;

    public MetalDeviceAddressReach(MetalGuestMemory memory, MetalDeviceAddressReachProfile? profile)
    {
        _memory = memory;
        _profile = profile;
        _bitmapWords = (memory.GranuleCount + 31) / 32;
        // One bit per 16 KiB page of the backing, when single-copy chunks can be trapped.
        var trapPages = memory.TrapChunkBytes == 0 ? 0UL : (memory.AliasBytes + DeviceAddressPaging.PageSize - 1) >> DeviceAddressPaging.PageBits;
        _trapFirstWord = HeaderWords + _bitmapWords;
        _trapWords = (int)((trapPages + 31) / 32);
        var words = _trapFirstWord + _trapWords;
        Buffer = memory.CreateResidentSharedBuffer(
            (ulong)words * sizeof(uint), "device-address.reach", out _contents, out var address);
        FeedbackAddress = address;
        unsafe
        {
            var span = new Span<uint>((void*)_contents, words);
            span.Clear();
            span[TrapBitmapWord] = (uint)_trapFirstWord;
            span[TrapBitmapCountWord] = (uint)_trapWords;
            span[TrapShadowWord] = memory.IsSingleCopyShadow ? 1u : 0u;
        }

        if (profile is not null)
        {
            foreach (var granule in profile.Granules)
            {
                if (granule < (ulong)memory.GranuleCount)
                {
                    Reached.Add((int)granule);
                }
            }
        }
    }

    public nint Buffer { get; }

    public ulong FeedbackAddress { get; }

    /// <summary>The granules the title's device-address programs are known to reach.</summary>
    public MetalWindowSet Reached { get; } = new();

    /// <summary>Reads and clears the feedback; returns how many granules it added.</summary>
    public unsafe int TakeFeedback()
    {
        var words = (int*)_contents;
        Interlocked.Add(ref _misses, (uint)Interlocked.Exchange(ref words[MissCountWord], 0));
        var added = 0;
        for (var word = 0; word < _bitmapWords; word++)
        {
            if (Volatile.Read(ref words[HeaderWords + word]) == 0)
            {
                continue;
            }

            for (var bits = (uint)Interlocked.Exchange(ref words[HeaderWords + word], 0); bits != 0; bits &= bits - 1)
            {
                var granule = (word * 32) + System.Numerics.BitOperations.TrailingZeroCount(bits);
                if (granule < _memory.GranuleCount && Reached.Add(granule))
                {
                    added++;
                    _learnedGranules.Add((ulong)granule);
                }
            }
        }

        if (added != 0)
        {
            Interlocked.Add(ref _learned, added);
            _profile?.Learn(_learnedGranules);
            _learnedGranules.Clear();
        }

        return added;
    }

    public (long Misses, long Learned) TakeCounters() =>
        (Interlocked.Exchange(ref _misses, 0), Interlocked.Exchange(ref _learned, 0));

    /// <summary>
    /// Takes the trap hits submissions reported: how many accesses reached a trap entry, and
    /// the backing pages they reached (added to <paramref name="pages"/>). The bitmap is only
    /// scanned after a count, and for a few takes more, since a bit may land after its count.
    /// </summary>
    public unsafe long TakeTrapHits(List<ulong> pages)
    {
        if (_trapWords == 0)
        {
            return 0;
        }

        var words = (int*)_contents;
        var hits = (long)(uint)Interlocked.Exchange(ref words[TrapCountWord], 0);
        if (hits != 0)
        {
            _trapRescans = TrapRescans;
        }
        else if (_trapRescans == 0)
        {
            return 0;
        }
        else
        {
            _trapRescans--;
        }

        var bitmap = new ReadOnlySpan<int>(words + _trapFirstWord, _trapWords);
        for (var start = 0; start < bitmap.Length;)
        {
            var found = bitmap[start..].IndexOfAnyExcept(0);
            if (found < 0)
            {
                break;
            }

            var word = start + found;
            for (var bits = (uint)Interlocked.Exchange(ref words[_trapFirstWord + word], 0); bits != 0; bits &= bits - 1)
            {
                pages.Add(((ulong)word * 32) + (ulong)System.Numerics.BitOperations.TrailingZeroCount(bits));
            }

            start = word + 1;
        }

        return hits;
    }
}

/// <summary>
/// A title's device-address reach, persisted: the granules of the guest backing its programs
/// were seen to reach. The first line names the granule size (<c>granule_bits=N</c>); every
/// later line is the hexadecimal granule numbers of one learn, appended as they are learned and
/// read back as a union. A profile written at another granule size is converted on open (a
/// coarser granule becomes all the finer ones it holds, a finer one the coarser one holding it)
/// and rewritten. The v2 profile of the 1 GiB layout is not read: it learned whole windows, so
/// it would name gigabytes the programs never reached. A granule the layout of a later start no
/// longer maps there only names a window more; one it misses is learned again on first touch.
/// </summary>
internal sealed class MetalDeviceAddressReachProfile
{
    internal const string FileName = "device-address-reach-v3.txt";
    private const string Header = "granule_bits=";

    private readonly string _path;

    private MetalDeviceAddressReachProfile(string path, HashSet<ulong> granules)
    {
        _path = path;
        Granules = granules;
    }

    public HashSet<ulong> Granules { get; }

    public static MetalDeviceAddressReachProfile? Open(string directory, ulong granuleBytes)
    {
        try
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, FileName);
            var bits = System.Numerics.BitOperations.TrailingZeroCount(granuleBytes);
            var (storedBits, stored) = Load(path);
            var granules = Convert(stored, storedBits ?? bits, bits);
            if (storedBits != bits)
            {
                File.WriteAllText(path, HeaderLine(bits) + "\n" + (granules.Count != 0 ? Line(granules) + "\n" : string.Empty));
            }

            return new MetalDeviceAddressReachProfile(path, granules);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Device-address reach could not be read: {exception.Message}");
            return null;
        }
    }

    internal static string HeaderLine(int granuleBits) =>
        Header + granuleBits.ToString(CultureInfo.InvariantCulture);

    /// <summary>The granule bits the file was written at (null when it has no header) and its granules.</summary>
    internal static (int? GranuleBits, HashSet<ulong> Granules) Load(string path)
    {
        var granules = new HashSet<ulong>();
        int? bits = null;
        if (!File.Exists(path))
        {
            return (bits, granules);
        }

        foreach (var line in File.ReadLines(path))
        {
            if (line.StartsWith(Header, StringComparison.Ordinal))
            {
                if (int.TryParse(line.AsSpan(Header.Length), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) &&
                    parsed is >= 14 and < 40)
                {
                    bits = parsed;
                }

                continue;
            }

            foreach (var text in line.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (ulong.TryParse(text, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var granule))
                {
                    granules.Add(granule);
                }
            }
        }

        // Granules without a header have no known size; they are dropped rather than guessed.
        return bits is null ? (null, new HashSet<ulong>()) : (bits, granules);
    }

    /// <summary>The same backing reach expressed in granules of another size.</summary>
    internal static HashSet<ulong> Convert(HashSet<ulong> granules, int fromBits, int toBits)
    {
        if (fromBits == toBits)
        {
            return granules;
        }

        var converted = new HashSet<ulong>();
        foreach (var granule in granules)
        {
            if (fromBits > toBits)
            {
                var per = 1UL << (fromBits - toBits);
                for (var fine = 0UL; fine < per; fine++)
                {
                    converted.Add((granule << (fromBits - toBits)) + fine);
                }
            }
            else
            {
                converted.Add(granule >> (toBits - fromBits));
            }
        }

        return converted;
    }

    internal static string Line(IEnumerable<ulong> granules) =>
        string.Join(',', granules.Order().Select(granule => granule.ToString("X", CultureInfo.InvariantCulture)));

    public void Learn(IReadOnlyCollection<ulong> granules)
    {
        var added = granules.Where(Granules.Add).ToList();
        if (added.Count == 0)
        {
            return;
        }

        try
        {
            File.AppendAllText(_path, Line(added) + "\n");
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Device-address reach could not be written: {exception.Message}");
        }
    }
}
