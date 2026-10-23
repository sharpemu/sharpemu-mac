// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// A set of guest memory window ids or granule indices, one bit each. It replaces the 64-bit
/// masks of the 1 GiB layout: a 32 MiB granule over an 18 GiB backing is ~580 windows and
/// granules. It grows on demand and never shrinks, so the sets kept per submission stop
/// allocating after the first few.
/// </summary>
internal sealed class MetalWindowSet
{
    private ulong[] _words;
    public long Version { get; private set; }

    public MetalWindowSet(int capacity = 64) =>
        _words = new ulong[Math.Max(1, (capacity + 63) / 64)];

    /// <summary>The number of ids the words can hold without growing.</summary>
    public int Capacity => _words.Length * 64;

    public bool IsEmpty
    {
        get
        {
            foreach (var word in _words)
            {
                if (word != 0)
                {
                    return false;
                }
            }

            return true;
        }
    }

    public int Count
    {
        get
        {
            var count = 0;
            foreach (var word in _words)
            {
                count += BitOperations.PopCount(word);
            }

            return count;
        }
    }

    /// <summary>Adds an id; true when it was not in the set.</summary>
    public bool Add(int id)
    {
        var word = id >> 6;
        if (word >= _words.Length)
        {
            Array.Resize(ref _words, Math.Max(word + 1, _words.Length * 2));
        }

        var bit = 1UL << (id & 63);
        if ((_words[word] & bit) != 0)
        {
            return false;
        }

        _words[word] |= bit;
        Version++;
        return true;
    }

    public void Remove(int id)
    {
        var word = id >> 6;
        if (word < _words.Length)
        {
            var before = _words[word];
            _words[word] &= ~(1UL << (id & 63));
            if (before != _words[word]) Version++;
        }
    }

    public bool Contains(int id)
    {
        var word = id >> 6;
        return id >= 0 && word < _words.Length && (_words[word] & (1UL << (id & 63))) != 0;
    }

    public void Clear() { Array.Clear(_words); Version++; }

    public void CopyFrom(MetalWindowSet other)
    {
        Version++;
        if (_words.Length < other._words.Length)
        {
            _words = new ulong[other._words.Length];
        }

        Array.Clear(_words);
        other._words.CopyTo(_words, 0);
    }

    /// <summary>Adds every id of another set; true when that added any.</summary>
    public bool UnionWith(MetalWindowSet other)
    {
        if (_words.Length < other._words.Length)
        {
            Array.Resize(ref _words, other._words.Length);
        }

        var added = false;
        for (var index = 0; index < other._words.Length; index++)
        {
            var news = other._words[index] & ~_words[index];
            if (news != 0)
            {
                _words[index] |= news;
                added = true;
            }
        }

        if (added) Version++;
        return added;
    }

    /// <summary>Whether every id of this set is in another.</summary>
    public bool IsSubsetOf(MetalWindowSet other)
    {
        for (var index = 0; index < _words.Length; index++)
        {
            var theirs = index < other._words.Length ? other._words[index] : 0;
            if ((_words[index] & ~theirs) != 0)
            {
                return false;
            }
        }

        return true;
    }

    public bool SetEquals(MetalWindowSet other) => IsSubsetOf(other) && other.IsSubsetOf(this);

    /// <summary>Replaces this set with the ids in both <paramref name="left"/> and <paramref name="right"/>.</summary>
    public void SetIntersection(MetalWindowSet left, MetalWindowSet right)
    {
        Version++;
        var length = Math.Min(left._words.Length, right._words.Length);
        if (_words.Length < length)
        {
            _words = new ulong[length];
        }

        Array.Clear(_words);
        for (var index = 0; index < length; index++)
        {
            _words[index] = left._words[index] & right._words[index];
        }
    }

    /// <summary>Replaces this set with the ids in <paramref name="left"/> but not in <paramref name="right"/>.</summary>
    public void SetDifference(MetalWindowSet left, MetalWindowSet right)
    {
        CopyFrom(left);
        var length = Math.Min(_words.Length, right._words.Length);
        for (var index = 0; index < length; index++)
        {
            _words[index] &= ~right._words[index];
        }
    }

    public Enumerator GetEnumerator() => new(_words);

    /// <summary>The ids in ascending order, without allocating.</summary>
    public struct Enumerator
    {
        private readonly ulong[] _words;
        private int _word;
        private ulong _bits;

        internal Enumerator(ulong[] words)
        {
            _words = words;
            _word = -1;
            _bits = 0;
            Current = -1;
        }

        public int Current { get; private set; }

        public bool MoveNext()
        {
            while (_bits == 0)
            {
                if (++_word >= _words.Length)
                {
                    return false;
                }

                _bits = _words[_word];
            }

            Current = (_word << 6) + BitOperations.TrailingZeroCount(_bits);
            _bits &= _bits - 1;
            return true;
        }
    }
}
