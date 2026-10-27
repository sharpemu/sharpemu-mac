// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections;

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>Dense shader words with explicit presence and sparse capture keys.</summary>
/// <remarks>Not thread safe. Clear preserves this object's identity. Enumeration follows insertion order;
/// overwrites preserve position, and removal followed by insertion appends a new position.</remarks>
public sealed class DenseShaderRegisterMap : IDictionary<uint, uint>, IReadOnlyDictionary<uint, uint>
{
    private const int Capacity = (int)RegisterBankLayout.ShaderRegisterCount;
    private readonly uint[] _values = new uint[Capacity];
    private readonly ulong[] _present = new ulong[(Capacity + 63) / 64];
    private readonly int[] _positions = new int[Capacity];
    private readonly List<uint> _order = new();
    private Dictionary<uint, (uint Value, int Position)>? _overflow;
    private int _version;
    private KeyCollection? _keys;
    private ValueCollection? _valueCollection;

    public int Count { get; private set; }
    public bool IsReadOnly => false;
    public ICollection<uint> Keys => _keys ??= new KeyCollection(this);
    public ICollection<uint> Values => _valueCollection ??= new ValueCollection(this);
    IEnumerable<uint> IReadOnlyDictionary<uint, uint>.Keys => Keys;
    IEnumerable<uint> IReadOnlyDictionary<uint, uint>.Values => Values;

    public uint this[uint key]
    {
        get => TryGetValue(key, out var value) ? value : throw new KeyNotFoundException();
        set
        {
            if (key < Capacity)
            {
                if (!ContainsDense(key))
                {
                    _positions[key] = Append(key);
                    _present[key >> 6] |= 1UL << (int)(key & 63);
                }
                _values[key] = value;
            }
            else
            {
                _overflow ??= new();
                var position = _overflow.TryGetValue(key, out var previous) ? previous.Position : Append(key);
                _overflow[key] = (value, position);
            }
        }
    }

    private bool ContainsDense(uint key) => (_present[key >> 6] & (1UL << (int)(key & 63))) != 0;

    private int Append(uint key)
    {
        // Only insertion invalidates enumerators. Compact removal tombstones here,
        // so repeated remove/reinsert does not grow the order list without a bound.
        if (_order.Count > Math.Max(Capacity, (long)Count * 2))
        {
            var kept = 0;
            for (var index = 0; index < _order.Count; index++)
            {
                var existing = _order[index];
                if (!TryGetValue(existing, out var value)) continue;
                var oldPosition = existing < Capacity ? _positions[existing] : _overflow![existing].Position;
                if (oldPosition != index) continue;
                _order[kept] = existing;
                if (existing < Capacity) _positions[existing] = kept;
                else _overflow![existing] = (value, kept);
                kept++;
            }
            _order.RemoveRange(kept, _order.Count - kept);
        }
        var position = _order.Count;
        _order.Add(key);
        Count++;
        _version++;
        return position;
    }

    public bool ContainsKey(uint key) => key < Capacity ? ContainsDense(key) : _overflow?.ContainsKey(key) == true;

    public bool TryGetValue(uint key, out uint value)
    {
        if (key < Capacity && ContainsDense(key))
        {
            value = _values[key];
            return true;
        }
        if (key >= Capacity && _overflow is not null && _overflow.TryGetValue(key, out var entry))
        {
            value = entry.Value;
            return true;
        }
        value = 0;
        return false;
    }

    public void Add(uint key, uint value)
    {
        if (ContainsKey(key)) throw new ArgumentException("The key already exists.", nameof(key));
        this[key] = value;
    }

    public bool Remove(uint key)
    {
        if (key < Capacity)
        {
            if (!ContainsDense(key)) return false;
            _present[key >> 6] &= ~(1UL << (int)(key & 63));
        }
        else if (_overflow is null || !_overflow.Remove(key)) return false;
        Count--;
        return true;
    }

    public void Clear()
    {
        Array.Clear(_present);
        _overflow?.Clear();
        _order.Clear();
        Count = 0;
    }

    public void Add(KeyValuePair<uint, uint> item) => Add(item.Key, item.Value);
    public bool Contains(KeyValuePair<uint, uint> item) => TryGetValue(item.Key, out var value) && value == item.Value;
    public bool Remove(KeyValuePair<uint, uint> item) => Contains(item) && Remove(item.Key);

    public void CopyTo(KeyValuePair<uint, uint>[] array, int arrayIndex)
    {
        ValidateCopy(array, arrayIndex, Count);
        foreach (var item in this) array[arrayIndex++] = item;
    }

    private static void ValidateCopy<T>(T[] array, int index, int count)
    {
        ArgumentNullException.ThrowIfNull(array);
        if ((uint)index > array.Length) throw new ArgumentOutOfRangeException(nameof(index));
        if (array.Length - index < count) throw new ArgumentException("The destination array is too small.");
    }

    public Enumerator GetEnumerator() => new(this);
    IEnumerator<KeyValuePair<uint, uint>> IEnumerable<KeyValuePair<uint, uint>>.GetEnumerator() => GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public struct Enumerator : IEnumerator<KeyValuePair<uint, uint>>
    {
        private readonly DenseShaderRegisterMap _owner;
        private readonly int _version;
        private int _index;
        public KeyValuePair<uint, uint> Current { get; private set; }
        object IEnumerator.Current => _index == 0 || _index > _owner._order.Count
            ? throw new InvalidOperationException("The enumerator is not positioned on an entry.") : Current;

        internal Enumerator(DenseShaderRegisterMap owner)
        {
            _owner = owner;
            _version = owner._version;
            _index = 0;
            Current = default;
        }

        public bool MoveNext()
        {
            if (_version != _owner._version) throw new InvalidOperationException("The collection was modified.");
            while (_index < _owner._order.Count)
            {
                var position = _index++;
                var key = _owner._order[position];
                if (!_owner.TryGetValue(key, out var value)) continue;
                var currentPosition = key < Capacity ? _owner._positions[key] : _owner._overflow![key].Position;
                if (position != currentPosition) continue;
                Current = new(key, value);
                return true;
            }
            _index = _owner._order.Count + 1;
            Current = default;
            return false;
        }

        public void Reset()
        {
            if (_version != _owner._version) throw new InvalidOperationException("The collection was modified.");
            _index = 0;
            Current = default;
        }
        public void Dispose() { }
    }

    private abstract class ReadOnlyCollection(DenseShaderRegisterMap owner) : ICollection<uint>
    {
        protected readonly DenseShaderRegisterMap Owner = owner;
        public int Count => Owner.Count;
        public bool IsReadOnly => true;
        public void Add(uint item) => throw new NotSupportedException();
        public void Clear() => throw new NotSupportedException();
        public bool Remove(uint item) => throw new NotSupportedException();
        public abstract bool Contains(uint item);
        public abstract IEnumerator<uint> GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        public void CopyTo(uint[] array, int arrayIndex)
        {
            ValidateCopy(array, arrayIndex, Count);
            foreach (var item in this) array[arrayIndex++] = item;
        }
    }

    private sealed class KeyCollection(DenseShaderRegisterMap owner) : ReadOnlyCollection(owner)
    {
        public override bool Contains(uint item) => Owner.ContainsKey(item);
        public override IEnumerator<uint> GetEnumerator() => Project(Owner.GetEnumerator());
        private static IEnumerator<uint> Project(Enumerator source)
        {
            while (source.MoveNext()) yield return source.Current.Key;
        }
    }

    private sealed class ValueCollection(DenseShaderRegisterMap owner) : ReadOnlyCollection(owner)
    {
        public override bool Contains(uint item)
        {
            foreach (var entry in Owner) if (entry.Value == item) return true;
            return false;
        }
        public override IEnumerator<uint> GetEnumerator() => Project(Owner.GetEnumerator());
        private static IEnumerator<uint> Project(Enumerator source)
        {
            while (source.MoveNext()) yield return source.Current.Value;
        }
    }
}
