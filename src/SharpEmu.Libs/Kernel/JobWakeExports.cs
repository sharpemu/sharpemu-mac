// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Kernel;

/// <summary>
/// The wait and wake a title patch's trampolines call for idle job workers (see
/// DemonsSoulsIdleNapPatch and <see cref="JobWakeRuntime"/>). Like <see cref="IdleNapExports"/>
/// they are reached through import stubs the loader adds for NIDs no console library exports.
/// </summary>
public static class JobWakeExports
{
    /// <summary>wait(scheduler, worker, generation, timeout_us, check_queues).</summary>
    public const string WaitNid = "SharpEmuJWt";

    /// <summary>wake(scheduler, worker_mask, count).</summary>
    public const string WakeNid = "SharpEmuJWk";

    [SysAbiExport(
        Nid = WaitNid,
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int JobWait(CpuContext ctx)
    {
        var scheduler = ctx[CpuRegister.Rdi];
        var worker = (int)(uint)ctx[CpuRegister.Rsi];
        var generation = (long)ctx[CpuRegister.Rdx];
        var timeout = (uint)ctx[CpuRegister.Rcx];
        GuestJobQueues? probe = (uint)ctx[CpuRegister.R8] != 0 ? new GuestJobQueues(ctx.Memory, scheduler) : null;
        _ = JobWakeRuntime.Wait(worker, generation, timeout, probe);
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = WakeNid,
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int JobWake(CpuContext ctx)
    {
        var scheduler = ctx[CpuRegister.Rdi];
        var mask = ctx[CpuRegister.Rsi];
        var count = (long)Math.Min(ctx[CpuRegister.Rdx], JobWakeRuntime.MaximumWorkers);
        ctx[CpuRegister.Rax] = (ulong)JobWakeRuntime.Wake(mask, count, new GuestJobQueues(ctx.Memory, scheduler));
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }
}

/// <summary>
/// The workers of Demon's Souls' job scheduler <c>S</c> that have queued work, read the way its
/// WaitForJobs (app+0x86EA30) looks: worker w has work when a queue group g has
/// <c>count[g][w] &gt; 0</c> (int at <c>[S+0x1000] + g*0x9220 + 0x91E0 + 4w</c>, groups
/// <c>[S+0xFF8]</c>) or its mailbox <c>[S+0xF80+8w]</c> is set. Workers number
/// <c>[S+0xB90]</c>. A count is raised for every job queued for the worker, runnable or still
/// waiting on its dependencies. Unreadable memory reads as no work.
/// </summary>
internal readonly struct GuestJobQueues(ICpuMemory memory, ulong scheduler) : IJobWorkProbe
{
    internal const ulong WorkerCountOffset = 0xB90;
    internal const ulong MailboxOffset = 0xF80;
    internal const ulong GroupCountOffset = 0xFF8;
    internal const ulong GroupsOffset = 0x1000;
    internal const ulong GroupStride = 0x9220;
    internal const ulong CountsOffset = 0x91E0;
    private const int MaximumGroups = 16;

    public ulong WorkMask()
    {
        Span<byte> header = stackalloc byte[8];
        if (!memory.TryRead(scheduler + WorkerCountOffset, header))
        {
            return 0;
        }

        var workers = (int)Math.Min(BinaryPrimitives.ReadUInt64LittleEndian(header), (ulong)JobWakeRuntime.MaximumWorkers);
        if (workers == 0 || !memory.TryRead(scheduler + GroupCountOffset, header[..4]))
        {
            return 0;
        }

        var groups = (int)Math.Min(BinaryPrimitives.ReadUInt32LittleEndian(header), (uint)MaximumGroups);
        Span<byte> row = stackalloc byte[JobWakeRuntime.MaximumWorkers * 8];
        var mask = 0UL;
        if (memory.TryRead(scheduler + MailboxOffset, row[..(workers * 8)]))
        {
            for (var w = 0; w < workers; w++)
            {
                if (BinaryPrimitives.ReadUInt64LittleEndian(row[(w * 8)..]) != 0)
                {
                    mask |= 1UL << w;
                }
            }
        }

        if (groups == 0 || !memory.TryRead(scheduler + GroupsOffset, header))
        {
            return mask;
        }

        var baseAddress = BinaryPrimitives.ReadUInt64LittleEndian(header);
        for (var g = 0; g < groups; g++)
        {
            if (!memory.TryRead(baseAddress + ((ulong)g * GroupStride) + CountsOffset, row[..(workers * 4)]))
            {
                continue;
            }

            for (var w = 0; w < workers; w++)
            {
                if (BinaryPrimitives.ReadInt32LittleEndian(row[(w * 4)..]) > 0)
                {
                    mask |= 1UL << w;
                }
            }
        }

        return mask;
    }
}
