// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Rendering;

/// <summary>
/// Whether a title's word copy or fill kernel (<c>dst[i] = src[i % period]</c>, a fill being a
/// copy from its pattern) would change any word of its destination, and which 4 KiB pages it
/// would change. Destination word <c>i</c> is compared with source word <c>i % period</c>; the
/// source holds at least <c>min(period, words)</c> words.
///
/// A masked copy (<c>dst[i] = src[i &amp; mask]</c>, mask a run of low bits) is the same copy
/// with period <c>mask + 1</c>, and its source may be shorter than that when fewer words are
/// copied: the executor passes <c>min(mask + 1, words)</c>, which picks the same words.
///
/// The words are compared a source run at a time (the run lasts until the index wraps), so a
/// long period is one vectorised compare. A short period is repeated into a block first, so a
/// one-word fill is not compared one word at a time.
/// </summary>
public static class ModuloCopyCompare
{
    /// <summary>The page size the changed-page walk reports in, as the guest page tracking uses.</summary>
    public const int PageBytes = 4096;

    private const int PageWords = PageBytes / sizeof(uint);

    // A period this short is compared through a block of whole repeats at least this many words long.
    private const int RepeatedBlockWords = 256;

    /// <summary>
    /// True when every one of <paramref name="destination"/>'s words equals the source word the
    /// copy would store there: <c>destination[j] == source[(firstWord + j) % period]</c>.
    /// </summary>
    public static bool Equal(ReadOnlySpan<uint> destination, ReadOnlySpan<uint> source, ulong period, ulong firstWord = 0) =>
        FirstDifference(destination, source, period, firstWord) < 0;

    /// <summary>The index in <paramref name="destination"/> of the first word the copy would change, or -1.</summary>
    public static int FirstDifference(ReadOnlySpan<uint> destination, ReadOnlySpan<uint> source, ulong period, ulong firstWord = 0)
    {
        CheckShape(destination.Length, source.Length, period, firstWord);
        if (destination.IsEmpty)
        {
            return -1;
        }

        if (period < RepeatedBlockWords && (ulong)destination.Length > period)
        {
            return FirstDifferenceRepeated(destination, source[..(int)period], firstWord);
        }

        var done = 0;
        while (done < destination.Length)
        {
            var sourceIndex = (int)((firstWord + (ulong)done) % period);
            var run = (int)Math.Min((ulong)(destination.Length - done), period - (ulong)sourceIndex);
            var compared = destination.Slice(done, run);
            var common = compared.CommonPrefixLength(source.Slice(sourceIndex, run));
            if (common != run)
            {
                return done + common;
            }

            done += run;
        }

        return -1;
    }

    /// <summary>How many of <paramref name="destination"/>'s words the copy would change.</summary>
    public static int DifferingWords(ReadOnlySpan<uint> destination, ReadOnlySpan<uint> source, ulong period, ulong firstWord = 0)
    {
        CheckShape(destination.Length, source.Length, period, firstWord);
        // Every word is visited anyway, so the index wraps by hand instead of by division.
        var differing = 0;
        var sourceIndex = (int)(firstWord % period);
        var wrap = (int)Math.Min(period, (ulong)source.Length);
        foreach (var word in destination)
        {
            differing += word != source[sourceIndex] ? 1 : 0;
            if (++sourceIndex == wrap)
            {
                sourceIndex = 0;
            }
        }

        return differing;
    }

    /// <summary>
    /// The 4 KiB guest pages the copy would change, in address order: each is the part of the
    /// destination (at guest address <paramref name="destinationAddress"/>, 4-byte aligned) that
    /// lies on one page and holds at least one word the copy would change.
    /// </summary>
    public static ChangedPageEnumerator ChangedPages(
        ulong destinationAddress, ReadOnlySpan<uint> destination, ReadOnlySpan<uint> source, ulong period)
    {
        if ((destinationAddress & (sizeof(uint) - 1)) != 0)
        {
            throw new ArgumentException("The destination is not word aligned.", nameof(destinationAddress));
        }

        CheckShape(destination.Length, source.Length, period, 0);
        return new ChangedPageEnumerator(destinationAddress, destination, source, period);
    }

    /// <summary>One page's part of the destination: its guest address and its words in the destination.</summary>
    public readonly record struct ChangedPage(ulong Address, int FirstWord, int WordCount);

    public ref struct ChangedPageEnumerator
    {
        private readonly ulong _address;
        private readonly ReadOnlySpan<uint> _destination;
        private readonly ReadOnlySpan<uint> _source;
        private readonly ulong _period;
        private int _next;

        internal ChangedPageEnumerator(ulong address, ReadOnlySpan<uint> destination, ReadOnlySpan<uint> source, ulong period)
        {
            _address = address;
            _destination = destination;
            _source = source;
            _period = period;
        }

        public ChangedPage Current { get; private set; }

        public readonly ChangedPageEnumerator GetEnumerator() => this;

        public bool MoveNext()
        {
            while (_next < _destination.Length)
            {
                var first = _next;
                var address = _address + (ulong)first * sizeof(uint);
                var toPageEnd = (int)((PageBytes - (address & (PageBytes - 1))) / sizeof(uint));
                var count = Math.Min(toPageEnd, _destination.Length - first);
                _next = first + count;
                if (!Equal(_destination.Slice(first, count), _source, _period, (ulong)first))
                {
                    Current = new ChangedPage(address, first, count);
                    return true;
                }
            }

            return false;
        }
    }

    // The source as a block of whole repeats, so that every block-sized step of the destination
    // starts at the same phase of the period and compares as one span.
    private static int FirstDifferenceRepeated(ReadOnlySpan<uint> destination, ReadOnlySpan<uint> pattern, ulong firstWord)
    {
        var period = pattern.Length;
        var repeats = (RepeatedBlockWords + period - 1) / period;
        Span<uint> block = stackalloc uint[repeats * period + period];
        for (var at = 0; at < block.Length; at += period)
        {
            pattern.CopyTo(block[at..]);
        }

        // block[phase..] starts at the destination's first word and still holds a whole block.
        var phase = (int)(firstWord % (ulong)period);
        var window = block.Slice(phase, repeats * period);
        var done = 0;
        while (done < destination.Length)
        {
            var run = Math.Min(destination.Length - done, window.Length);
            var common = destination.Slice(done, run).CommonPrefixLength(window[..run]);
            if (common != run)
            {
                return done + common;
            }

            done += run;
        }

        return -1;
    }

    private static void CheckShape(int destinationWords, int sourceWords, ulong period, ulong firstWord)
    {
        if (period == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(period), "A copy repeats a source of at least one word.");
        }

        // The highest source word read is the period's last once the index wraps, else the last copied.
        var phase = firstWord % period;
        var needed = destinationWords == 0 ? 0 : Math.Min(period, phase + (ulong)destinationWords);
        if ((ulong)sourceWords < needed)
        {
            throw new ArgumentException("The source is shorter than the words the copy reads.", nameof(sourceWords));
        }
    }
}
