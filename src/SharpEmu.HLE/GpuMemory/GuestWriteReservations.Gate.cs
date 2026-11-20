// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.HLE.GpuMemory;

// The managed-write lease guards a CPU write against a GPU claim over its pages, and only a backend that makes GPU claims
// (the Vulkan buffer cache) gives it a meaning: the Metal backend never claims (it orders its own guest-memory writes
// another way), so on macOS every managed write paid for a reservation nothing ever waited on. A managed write therefore
// takes the lease only once a claimant has registered with these reservations (RegisterClaimant: the Vulkan cache when it
// is constructed, and any first claim before it proceeds); until then it runs as it did before the lease existed, with
// one per-thread depth counter around its copy and landed notification.
//
// No claim can meet a write that started before registration unprotected. Registration first raises _claimant, then
// makes every thread either see it or show its depth (Interlocked.MemoryBarrierProcessWide), then waits until every
// unleased depth is zero, and only then lets a claim proceed (_registered). An unleased write raises its thread's depth
// after its premark and reads _claimant afterwards: either the read sees the claimant, and the write backs out and takes
// the lease (its premark repeats, which marks the same pages again), or its depth store precedes the barrier, and
// registration waits for the write to land and be marked. The write's own side needs no fence, as for the chunk marks of
// the lease's lock-free path. The premark stays outside the depth, as it stays outside the reservation (it may relay to
// the GPU, whose claim must not wait on it).
//
// SHARPEMU_MANAGED_WRITE_LEASE_GATE=0 makes every managed write take the lease, as upstream does.
public sealed partial class GuestWriteReservations
{
    /// <summary>False: every managed write takes the lease (SHARPEMU_MANAGED_WRITE_LEASE_GATE=0). Tests switch it.</summary>
    public static bool GateEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_MANAGED_WRITE_LEASE_GATE") != "0";

    private readonly object _registration = new();
    private int _claimant;
    private int _registered;

    // Each thread's unleased writes in progress, padded so two threads never share a cache line.
    [StructLayout(LayoutKind.Explicit, Size = 128)]
    private struct UnleasedRecord
    {
        [FieldOffset(64)] public int Depth;
    }

    private static readonly UnleasedRecord[] s_unleased = new UnleasedRecord[RecordCount];
    private static int s_unleasedTaken;

    // The thread's record: its index + 1, 0 before its first unleased write, -1 when every record is taken.
    [ThreadStatic] private static int t_unleased;

    /// <summary>Whether a managed write over these reservations takes the lease: a claimant registered, or the gate is off.</summary>
    public bool LeasesWrites => Volatile.Read(ref _claimant) != 0 || !GateEnabled;

    /// <summary>Whether a backend that makes GPU claims registered with these reservations.</summary>
    public bool HasClaimant => Volatile.Read(ref _registered) != 0;

    /// <summary>
    /// A backend that makes GPU claims (<see cref="BeginGpuClaim"/>): every managed write from now on takes the lease, and
    /// this returns once every write that started without one has landed. Idempotent; the first claim calls it too.
    /// </summary>
    public void RegisterClaimant()
    {
        if (Volatile.Read(ref _registered) != 0)
        {
            return;
        }

        lock (_registration)
        {
            if (_registered != 0)
            {
                return;
            }

            Volatile.Write(ref _claimant, 1);
            // Every thread either reads the claimant from now on or has its depth store visible below.
            Interlocked.MemoryBarrierProcessWide();
            var own = t_unleased;
            var records = Math.Min(Volatile.Read(ref s_unleasedTaken), RecordCount);
            for (var index = 0; index < records; index++)
            {
                if (index + 1 == own && Volatile.Read(ref s_unleased[index].Depth) != 0)
                {
                    throw new InvalidOperationException("A managed CPU writer cannot register a GPU claimant inside its own write.");
                }

                var spin = new SpinWait();
                while (Volatile.Read(ref s_unleased[index].Depth) != 0)
                {
                    spin.SpinOnce(sleep1Threshold: -1);
                }
            }

            Volatile.Write(ref _registered, 1);
        }
    }

    /// <summary>
    /// A managed write after its premark, with no claimant registered: counts it in the thread's depth. False when a
    /// claimant registered meanwhile (or no record is left): the write takes the lease instead.
    /// </summary>
    internal bool TryEnterUnleasedWrite()
    {
        var index = t_unleased;
        if (index == 0)
        {
            index = Interlocked.Increment(ref s_unleasedTaken);
            t_unleased = index = index <= RecordCount ? index : -1;
        }

        if (index < 0)
        {
            return false;
        }

        ref var record = ref s_unleased[index - 1];
        var depth = record.Depth;
        Volatile.Write(ref record.Depth, depth + 1);
        // No fence: registration makes the depth store visible before it reads the depths.
        if (Volatile.Read(ref _claimant) == 0)
        {
            return true;
        }

        Volatile.Write(ref record.Depth, depth);
        return false;
    }

    /// <summary>The landed end of a write <see cref="TryEnterUnleasedWrite"/> counted.</summary>
    internal static void ExitUnleasedWrite()
    {
        ref var record = ref s_unleased[t_unleased - 1];
        Volatile.Write(ref record.Depth, record.Depth - 1);
    }
}
