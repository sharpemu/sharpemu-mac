// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using WorkerScratch = SharpEmu.ShaderCompiler.WorkerScratch;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

/// <summary>
/// The program a dispatch prepared before it suspended on its pipeline (RenderExecutor.Dispatch),
/// kept for the attempt that runs it once the compile is done. Preparing it again is the cost of a
/// retry: the user data, the resource materialisation with its guest reads, which can wait for the
/// GPU, and the executor's consumers. The retry takes this instead while the shader source is the
/// same and every guest word the preparation read still holds what it read; the first difference,
/// or a word it cannot check without waiting, prepares the dispatch again as before. Kept, it is
/// the state the dispatch would have seen had it run in order, when it first reached the packet.
/// </summary>
public sealed class ComputeProgramRetry
{
    internal ComputeProgram? Program;
    internal ShaderSource? Source;
    internal uint Initiator;
    internal uint DimensionX;
    internal uint DimensionY;
    internal uint DimensionZ;

    internal GuestReadLog Reads { get; } = new();

    // The last GetComputeProgram took the kept program rather than preparing it.
    public bool Reused { get; internal set; }

    internal bool Matches(ShaderSource source, uint initiator, uint x, uint y, uint z) =>
        Program is not null && Source is { } kept &&
        ReferenceEquals(kept.Registered, source.Registered) && kept.Hash == source.Hash &&
        kept.UserDataBase == source.UserDataBase && kept.Stage == source.Stage &&
        kept.ExcludedUserDataRegisters == source.ExcludedUserDataRegisters &&
        kept.UserData.AsSpan().SequenceEqual(source.UserData) &&
        Initiator == initiator && DimensionX == x && DimensionY == y && DimensionZ == z;

    internal void Keep(ComputeProgram program, ShaderSource source, uint initiator, uint x, uint y, uint z)
    {
        Program = program;
        Source = source;
        Initiator = initiator;
        DimensionX = x;
        DimensionY = y;
        DimensionZ = z;
    }

    internal void Reset()
    {
        Program = null;
        Source = null;
        Reused = false;
        Reads.Clear();
    }
}

/// <summary>
/// The guest words and byte runs one preparation read, and what it read there. Bounded: a
/// preparation that reads more than the bound is not kept (<see cref="Complete"/> false).
/// </summary>
internal sealed class GuestReadLog
{
    private const int MaxBytes = 64 * 1024;

    private readonly List<(ulong Address, uint Word)> _words = [];
    // The runs live in one buffer the log keeps across Clear, so a dispatch's reads cost no array
    // each; SHARPEMU_WORKER_SCRATCH_REUSE=0 copies every run into an array of its own again.
    private readonly List<(ulong Address, int Offset, int Length)> _runs = [];
    private readonly List<(ulong Address, byte[] Bytes)> _copiedRuns = [];
    private byte[] _data = [];
    private int _dataLength;
    private readonly object _gate = new();
    private int _bytes;

    // False once a read failed or the bound was passed: the log no longer says everything
    // the preparation saw, so nothing may be reused on its word.
    public bool Complete { get; private set; } = true;

    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _words.Count + _runs.Count + _copiedRuns.Count;
            }
        }
    }

    public void Clear()
    {
        lock (_gate)
        {
            _words.Clear();
            _runs.Clear();
            _copiedRuns.Clear();
            _dataLength = 0;
            _bytes = 0;
            Complete = true;
        }
    }

    public void Word(ulong address, bool read, uint word)
    {
        lock (_gate)
        {
            if (!read || (_bytes += sizeof(uint)) > MaxBytes)
            {
                Complete = false;
                return;
            }

            _words.Add((address, word));
        }
    }

    public void Bytes(ulong address, bool read, ReadOnlySpan<byte> bytes)
    {
        lock (_gate)
        {
            if (!read || (_bytes += bytes.Length) > MaxBytes)
            {
                Complete = false;
                return;
            }

            if (!WorkerScratch.Enabled)
            {
                WorkerScratch.Note(WorkerScratch.Site.ReadLogRuns, reused: false);
                _copiedRuns.Add((address, bytes.ToArray()));
                return;
            }

            var grown = _data.Length < _dataLength + bytes.Length;
            if (grown)
            {
                Array.Resize(ref _data, Math.Max(_dataLength + bytes.Length, Math.Min(_data.Length * 2, MaxBytes)));
            }

            WorkerScratch.Note(WorkerScratch.Site.ReadLogRuns, reused: !grown);
            bytes.CopyTo(_data.AsSpan(_dataLength));
            _runs.Add((address, _dataLength, bytes.Length));
            _dataLength += bytes.Length;
        }
    }

    /// <summary>
    /// Whether every logged read would read the same now, asked without waiting: a range the GPU
    /// may still own, or one the host cannot answer from what it holds, does not hold.
    /// </summary>
    public bool Holds(ResidentGuestBytesReader read)
    {
        lock (_gate)
        {
            if (!Complete)
            {
                return false;
            }

            Span<byte> word = stackalloc byte[sizeof(uint)];
            foreach (var (address, expected) in _words)
            {
                if (!read(address, word, false) ||
                    System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(word) != expected)
                {
                    return false;
                }
            }

            var longest = 0;
            foreach (var (_, _, length) in _runs)
            {
                longest = Math.Max(longest, length);
            }

            foreach (var (_, expected) in _copiedRuns)
            {
                longest = Math.Max(longest, expected.Length);
            }

            // The comparison buffer is a stack frame for the usual run and one array for a long one.
            Span<byte> scratch = longest <= 1024 ? stackalloc byte[longest] : new byte[longest];
            foreach (var (address, offset, length) in _runs)
            {
                var now = scratch[..length];
                if (!read(address, now, false) || !now.SequenceEqual(_data.AsSpan(offset, length)))
                {
                    return false;
                }
            }

            foreach (var (address, expected) in _copiedRuns)
            {
                var now = scratch[..expected.Length];
                if (!read(address, now, false) || !now.SequenceEqual(expected))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
