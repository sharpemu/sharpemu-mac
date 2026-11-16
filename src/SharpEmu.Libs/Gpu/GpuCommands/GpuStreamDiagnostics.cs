// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>
/// Evidence for command streams the worker decodes differently from what the guest wrote:
/// a submit-time hash of each top-level DCB (SHARPEMU_DCB_HASH=1), a header-type check
/// (SHARPEMU_GPU_STRICT_HEADERS=1) and counters for the speculative ring-chunk follow.
/// </summary>
public static class GpuStreamDiagnostics
{
    private const ulong FnvOffset = 0xCBF29CE484222325UL;
    private const ulong FnvPrime = 0x100000001B3UL;
    private const int ReadChunkDwords = 256;

    public static readonly bool DcbHashEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_DCB_HASH") == "1";

    public static readonly bool StrictHeaders =
        Environment.GetEnvironmentVariable("SHARPEMU_GPU_STRICT_HEADERS") == "1";

    private static long _hashChecks;
    private static long _hashMismatches;
    private static long _ringFollows;
    private static long _ringSuspends;

    public static long HashChecks => Interlocked.Read(ref _hashChecks);

    public static long HashMismatches => Interlocked.Read(ref _hashMismatches);

    public static long RingFollows => Interlocked.Read(ref _ringFollows);

    public static long RingSuspends => Interlocked.Read(ref _ringSuspends);

    public static void RecordRingFollow() => Interlocked.Increment(ref _ringFollows);

    public static void RecordRingSuspend() => Interlocked.Increment(ref _ringSuspends);

    // FNV-1a over the little-endian bytes of each dword; no allocation.
    public static ulong Hash(ReadOnlySpan<uint> dwords) => Hash(FnvOffset, dwords);

    private static ulong Hash(ulong state, ReadOnlySpan<uint> dwords)
    {
        foreach (var dword in dwords)
        {
            var value = dword;
            for (var shift = 0; shift < 32; shift += 8)
            {
                state = (state ^ ((value >> shift) & 0xFFu)) * FnvPrime;
            }
        }

        return state;
    }

    public static bool TryHashGuest(ICommandStreamHost host, ulong address, uint dwordCount, out ulong hash)
    {
        hash = FnvOffset;
        Span<uint> chunk = stackalloc uint[ReadChunkDwords];
        for (uint done = 0; done < dwordCount;)
        {
            var count = (int)Math.Min((uint)ReadChunkDwords, dwordCount - done);
            var slice = chunk.Slice(0, count);
            if (!host.TryReadGuest(address + ((ulong)done * sizeof(uint)), MemoryMarshal.AsBytes(slice)))
            {
                return false;
            }

            hash = Hash(hash, slice);
            done += (uint)count;
        }

        return true;
    }

    // SHARPEMU_CP_HISTORY=reuse: whether the guest has changed the submission's top-level DCB since submit.
    // Small DCBs only (<= 8 Ki dwords): hashing every slice of a gameplay frame's large DCBs stalls the stream.
    internal static bool ChangedSinceSubmit(ICommandStreamHost host, CommandSubmission submission) =>
        DcbHashEnabled && submission.DcbHashed && submission.DwordCount <= 8192 &&
        TryHashGuest(host, submission.Address, submission.DwordCount, out var now) && now != submission.DcbHash;

    // Re-hashes the submission's top-level DCB and reports a difference from the submit-time hash.
    public static void Verify(ICommandStreamHost host, CommandSubmission submission, string at)
    {
        if (!DcbHashEnabled || !submission.DcbHashed)
        {
            return;
        }

        if (!TryHashGuest(host, submission.Address, submission.DwordCount, out var now))
        {
            return;
        }

        Interlocked.Increment(ref _hashChecks);
        if (now == submission.DcbHash)
        {
            return;
        }

        Interlocked.Increment(ref _hashMismatches);
        Console.Error.WriteLine(
            $"[GPU][DCBHASH] mismatch serial={submission.SubmissionId} addr=0x{submission.Address:X16} " +
            $"dwords={submission.DwordCount} submit=0x{submission.DcbHash:X16} now=0x{now:X16} at={at}");
    }
}
