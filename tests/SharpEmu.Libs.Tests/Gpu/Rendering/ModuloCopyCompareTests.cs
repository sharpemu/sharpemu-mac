// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

/// <summary>
/// The compare against <c>dst[i] = src[i % period]</c>, checked against a word-by-word reference
/// over random shapes: periods shorter and longer than the copy, destinations that do not start or
/// end on a page, single changed words anywhere, and the masked copy's shortened source.
/// </summary>
public sealed class ModuloCopyCompareTests
{
    private const int Trials = 400;

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(1000)]
    public void APeriodShorterThanTheCopyRepeats(int period)
    {
        var random = new System.Random(period);
        for (var trial = 0; trial < Trials / 4; trial++)
        {
            var words = period + 1 + random.Next(3 * period + 64);
            var (destination, source) = Identical(random, words, period);
            AssertMatchesReference(destination, source, (ulong)period);
            Assert.True(ModuloCopyCompare.Equal(destination, source, (ulong)period));

            var changed = random.Next(words);
            destination[changed] ^= 1u << random.Next(32);
            AssertMatchesReference(destination, source, (ulong)period);
            Assert.Equal(changed, ModuloCopyCompare.FirstDifference(destination, source, (ulong)period));
            Assert.Equal(1, ModuloCopyCompare.DifferingWords(destination, source, (ulong)period));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(100)]
    public void APeriodAsLongAsTheCopyOrLongerIsAPlainCopy(int extra)
    {
        var random = new System.Random(extra + 11);
        for (var trial = 0; trial < Trials / 4; trial++)
        {
            var words = 1 + random.Next(2048);
            var period = words + extra;
            var source = RandomWords(random, period);
            var destination = source[..words];
            Assert.True(ModuloCopyCompare.Equal(destination, source, (ulong)period));
            Assert.Equal(-1, ModuloCopyCompare.FirstDifference(destination, source, (ulong)period));

            var changed = random.Next(words);
            destination[changed]++;
            Assert.Equal(changed, ModuloCopyCompare.FirstDifference(destination, source, (ulong)period));
            AssertMatchesReference(destination, source, (ulong)period);
        }
    }

    [Fact]
    public void RandomShapesAgreeWithTheReference()
    {
        var random = new System.Random(1234);
        for (var trial = 0; trial < Trials; trial++)
        {
            var period = 1 + random.Next(random.Next(2) == 0 ? 8 : 600);
            var words = 1 + random.Next(1500);
            var (destination, source) = Identical(random, words, period);
            // Some words changed, some changed back to what the copy stores.
            for (var change = random.Next(4); change > 0; change--)
            {
                destination[random.Next(words)] = (uint)random.Next(3);
            }

            AssertMatchesReference(destination, source, (ulong)period);
            var start = random.Next(words);
            AssertMatchesReference(destination[start..], source, (ulong)period, (ulong)start);
        }
    }

    [Theory]
    [InlineData(0x1000UL, 1024)]
    [InlineData(0x1004UL, 1024)]
    [InlineData(0x1FFCUL, 3)]
    [InlineData(0x1FF0UL, 2100)]
    public void ChangedPagesNameEachPageWithAChangedWordOnce(ulong address, int words)
    {
        var random = new System.Random((int)address ^ words);
        for (var trial = 0; trial < Trials / 8; trial++)
        {
            var period = 1 + random.Next(40);
            var (destination, source) = Identical(random, words, period);
            Assert.Equal(0, CountPages(address, destination, source, (ulong)period));
            var changedWords = new HashSet<int>();
            for (var change = 1 + random.Next(3); change > 0; change--)
            {
                var at = random.Next(words);
                changedWords.Add(at);
                destination[at] = ~ExpectedWord(source, (ulong)period, (ulong)at);
            }

            var expected = changedWords.Select(word => (address + (ulong)word * 4) / ModuloCopyCompare.PageBytes).Distinct().Order().ToArray();
            var pages = new List<ModuloCopyCompare.ChangedPage>();
            foreach (var page in ModuloCopyCompare.ChangedPages(address, destination, source, (ulong)period))
            {
                pages.Add(page);
            }

            Assert.Equal(expected, pages.Select(page => page.Address / ModuloCopyCompare.PageBytes).ToArray());
            foreach (var page in pages)
            {
                // The part on the page: it starts on the page or at the destination, and stays on one page.
                Assert.Equal(address + (ulong)page.FirstWord * 4, page.Address);
                Assert.True(page.FirstWord == 0 || page.Address % ModuloCopyCompare.PageBytes == 0);
                Assert.Equal(page.Address / ModuloCopyCompare.PageBytes,
                    (page.Address + (ulong)page.WordCount * 4 - 1) / ModuloCopyCompare.PageBytes);
                Assert.True(page.FirstWord + page.WordCount == words ||
                    (page.Address + (ulong)page.WordCount * 4) % ModuloCopyCompare.PageBytes == 0);
            }
        }
    }

    // A masked copy stores src[i & mask]; the executor hands the host period mask + 1 with a
    // source of min(mask + 1, words) words, which must compare like the masked index.
    [Theory]
    [InlineData(0u)]
    [InlineData(1u)]
    [InlineData(0xFu)]
    [InlineData(0xFFu)]
    [InlineData(0x3FFu)]
    public void AMaskedCopyComparesLikeItsPeriod(uint mask)
    {
        var random = new System.Random((int)mask + 5);
        for (var trial = 0; trial < Trials / 8; trial++)
        {
            var words = 1 + random.Next(3000);
            var period = (ulong)mask + 1;
            var sourceLength = (int)Math.Min(period, (ulong)words);
            var source = RandomWords(random, sourceLength);
            var destination = new uint[words];
            for (var index = 0; index < words; index++)
            {
                destination[index] = source[index & (int)mask];
            }

            Assert.True(ModuloCopyCompare.Equal(destination, source, (ulong)sourceLength));
            Assert.True(ModuloCopyCompare.Equal(destination, source, period));
            var changed = random.Next(words);
            destination[changed] ^= 0x80000000u;
            Assert.Equal(changed, ModuloCopyCompare.FirstDifference(destination, source, (ulong)sourceLength));
            Assert.Equal(changed, ModuloCopyCompare.FirstDifference(destination, source, period));
        }
    }

    [Fact]
    public void ASourceShorterThanTheWordsTheCopyReadsIsRefused()
    {
        Assert.Throws<ArgumentException>(() => ModuloCopyCompare.Equal(new uint[8], new uint[3], 4));
        Assert.Throws<ArgumentException>(() => ModuloCopyCompare.Equal(new uint[3], new uint[2], 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => ModuloCopyCompare.Equal(new uint[3], new uint[3], 0));
        Assert.Throws<ArgumentException>(() => ModuloCopyCompare.ChangedPages(0x1002, new uint[3], new uint[3], 3));
        // Fewer words than the period read only as many source words.
        Assert.True(ModuloCopyCompare.Equal(new uint[3], new uint[3], 100));
        Assert.True(ModuloCopyCompare.Equal([], [], 7));
    }

    private static void AssertMatchesReference(uint[] destination, uint[] source, ulong period, ulong firstWord = 0)
    {
        var first = -1;
        var differing = 0;
        for (var index = 0; index < destination.Length; index++)
        {
            if (destination[index] != ExpectedWord(source, period, firstWord + (ulong)index))
            {
                first = first < 0 ? index : first;
                differing++;
            }
        }

        Assert.Equal(first, ModuloCopyCompare.FirstDifference(destination, source, period, firstWord));
        Assert.Equal(first < 0, ModuloCopyCompare.Equal(destination, source, period, firstWord));
        Assert.Equal(differing, ModuloCopyCompare.DifferingWords(destination, source, period, firstWord));
    }

    private static uint ExpectedWord(uint[] source, ulong period, ulong index) => source[(int)(index % period)];

    private static int CountPages(ulong address, uint[] destination, uint[] source, ulong period)
    {
        var count = 0;
        foreach (var _ in ModuloCopyCompare.ChangedPages(address, destination, source, period))
        {
            count++;
        }

        return count;
    }

    private static (uint[] Destination, uint[] Source) Identical(System.Random random, int words, int period)
    {
        var source = RandomWords(random, period);
        var destination = new uint[words];
        for (var index = 0; index < words; index++)
        {
            destination[index] = source[index % period];
        }

        return (destination, source);
    }

    private static uint[] RandomWords(System.Random random, int count)
    {
        var words = new uint[count];
        for (var index = 0; index < count; index++)
        {
            // Few distinct values, so a changed word often lands on a value the copy also stores.
            words[index] = (uint)random.Next(4) * 0x01010101u;
        }

        return words;
    }
}
