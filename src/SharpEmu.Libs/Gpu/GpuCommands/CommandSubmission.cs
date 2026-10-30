// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.GpuCommands;

public enum CommandSubmissionKind
{
    Graphics,
    Compute,
    // A video-out export flip queued behind the graphics submissions before it.
    FlipPreparation,
    // A suspend point queued behind the graphics submissions of the frame it ends.
    FrameBoundary,
}

public enum SubmissionProgress
{
    Complete,
    Blocked,
}

// One command buffer view; the words are read from guest memory when each packet executes.
public struct PacketCursor
{
    public ulong Address;
    public uint DwordCount;
    public uint Offset;
    public uint DeferredAdvance;

    // Base of the ring chunk this buffer belongs to; the chunk-advance sentinel needs it.
    public ulong RingChunkBase;
    public bool FollowedChunkAdvance;

    public PacketCursor(ulong address, uint dwordCount, ulong ringChunkBase = 0, bool followedChunkAdvance = false)
    {
        Address = address;
        DwordCount = dwordCount;
        Offset = 0;
        DeferredAdvance = 0;
        RingChunkBase = ringChunkBase;
        FollowedChunkAdvance = followedChunkAdvance;
    }

    public readonly uint Remaining => DwordCount - Offset;
}

// The cursor stack of one command stream, kept across suspensions.
public sealed class PacketCursorStack
{
    public const int MaxDepth = 64;

    private readonly List<PacketCursor> _cursors = new();

    public bool Suspended { get; internal set; }

    // Why the last slice suspended; the queue uses it to wake this submission only
    // when the thing it is waiting for can have changed.
    public CommandWaitCondition SuspendCondition { get; internal set; } = CommandWaitCondition.Unknown;

    public bool MadeProgress { get; internal set; }

    public int Depth => _cursors.Count;

    public bool IsEmpty => _cursors.Count == 0;

    // The packet the top cursor points at, or zero when no cursor is loaded.
    public ulong CurrentPacketAddress => _cursors.Count == 0 ? 0 : Top.Address + ((ulong)Top.Offset * sizeof(uint));

    internal ref PacketCursor Top => ref System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_cursors)[^1];

    internal ref PacketCursor At(int index) => ref System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_cursors)[index];

    internal void Push(PacketCursor cursor) => _cursors.Add(cursor);

    internal void Pop() => _cursors.RemoveAt(_cursors.Count - 1);

    internal void Clear()
    {
        _cursors.Clear();
        Suspended = false;
        SuspendCondition = CommandWaitCondition.Unknown;
        MadeProgress = false;
    }
}

public sealed class CommandSubmission
{
    public CommandSubmission(CommandSubmissionKind kind, int queueId, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
    {
        Kind = kind;
        QueueId = queueId;
        Address = address;
        DwordCount = dwordCount;
        SubmissionId = submissionId;
        GeometrySnapshots = geometrySnapshots;
    }

    public CommandSubmissionKind Kind { get; }

    public int QueueId { get; }

    public ulong Address { get; }

    public uint DwordCount { get; }

    // The submitter's own sequence number, kept for diagnostics and the translation state.
    public ulong SubmissionId { get; }

    internal ulong AdmissionOrdinal { get; set; }

    // SHARPEMU_DCB_HASH: hash of the top-level DCB dwords when it was submitted.
    internal ulong DcbHash { get; set; }

    internal bool DcbHashed { get; set; }

    // SHARPEMU_GPU_CLOBBER_DIAG: host timestamp of the enqueue, for the age at a fatal.
    internal long EnqueuedTicks { get; set; }

    // SHARPEMU_SLICE_PROFILE: when the last slice suspended the submission, and on what address.
    internal long ParkedTicks { get; set; }

    internal ulong ParkedAddress { get; set; }

    internal uint ParkedOpcode { get; set; }

    // Snapshots captured at submit time; the processor never reads them.
    public object? GeometrySnapshots { get; }

    public int FlipHandle { get; init; }

    public int FlipIndex { get; init; }

    public ulong FlipRequestId { get; init; }

    public PacketCursorStack Commands { get; } = new();

    // Its place in the order every queue was fed; the frame run-ahead waits by it.

    internal CommandSubmissionPreparation? Preparation { get; init; }

    public bool ResetInterpreter { get; internal set; }

    public bool Started { get; internal set; }

    public bool CommandsComplete { get; internal set; }

    public bool Blocked { get; internal set; }

    // Host timestamp of the slice that left this submission blocked; diagnostics only.
    public long BlockedSinceTicks { get; internal set; }

    // Address of the packet the blocked slice stopped on; diagnostics only.
    public ulong BlockedPacketAddress { get; internal set; }

    // When this submission first stopped on its current packet, kept across retries of
    // that same packet so a head that never moves on can be named; diagnostics only.
    public long StalledSinceTicks { get; internal set; }

    public bool StallReported { get; internal set; }

    // What the blocked slice is waiting for; the queue wakes it on that alone.
    public CommandWaitCondition Wait { get; internal set; } = CommandWaitCondition.Unknown;

    // The wake generation this head was already kept runnable for. A wake that arrives
    // while the slice runs is worth exactly one more run, and without remembering which
    // one it was, a queue that is told of a completion every millisecond never parks at
    // all: Poppy Playtime spun forty thousand slices in two seconds and drew nothing.
    public ulong RacedWakeGeneration { get; internal set; }

    // The value a watched wait last saw at its address; a re-read that differs wakes it.
    internal ulong WatchedValue { get; set; }
}
