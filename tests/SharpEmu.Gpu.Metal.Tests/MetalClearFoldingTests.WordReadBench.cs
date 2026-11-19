// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalClearFoldingTests
{
    /// <summary>
    /// SHARPEMU_BENCH_WORD_READS=&lt;resolutions&gt; times the guest reads of one stage's resolution on the resolving
    /// thread, shaped like Demon's Souls: 45 descriptor words a few words apart in a page that holds one pending GPU
    /// write (in a line of its own), over 250 live write records, then the worker's re-check of that resolution
    /// after an ownership version moved. Each combination of ConcurrentOwnedLines and ConcurrentScopedSerial is
    /// timed on the same host.
    /// </summary>
    [Fact]
    public void ConcurrentResolveCostOverAnOwnedDescriptorPage()
    {
        if (!int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_BENCH_WORD_READS"), out var resolutions) || resolutions <= 0)
        {
            return;
        }

        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        for (var record = 0UL; record < 250; record++)
        {
            host.NoteGuestMemoryWrite(GuestBase + 0x40000 + record * 0x100, 0x40, MetalGpuWriteOrigin.StorageBuffer);
        }

        WriteOwnedPage(rig);
        var words = new ulong[45];
        for (var index = 0; index < words.Length; index++)
        {
            words[index] = OwnedPage + (ulong)index * 5 * sizeof(uint);
        }

        var concurrent = (IConcurrentResolveHost)host;
        void Resolve()
        {
            concurrent.BeginConcurrentResolve();
            try
            {
                foreach (var address in words)
                {
                    if (!host.TryReadGuestWord(address, out _))
                    {
                        throw new InvalidOperationException($"word 0x{address:X} was not read");
                    }
                }
            }
            finally
            {
                concurrent.ReleaseAssumptions(concurrent.EndConcurrentResolve());
            }
        }

        double Time(Action body)
        {
            for (var warm = 0; warm < 2000; warm++)
            {
                body();
            }

            var stopwatch = Stopwatch.StartNew();
            for (var index = 0; index < resolutions; index++)
            {
                body();
            }

            stopwatch.Stop();
            return stopwatch.Elapsed.TotalMilliseconds * 1000 / resolutions;
        }

        foreach (var (ownedLines, scopedSerial) in new[] { (false, false), (true, false), (false, true), (true, true) })
        {
            host.ConcurrentOwnedLines = ownedLines;
            host.ConcurrentScopedSerial = scopedSerial;
            _ = host.TakeOwnerTestCounters();
            var resolve = Time(Resolve);
            var perResolve = host.TakeOwnerTestCounters();

            // One resolution's answers, checked again by the worker after a version moved.
            concurrent.BeginConcurrentResolve();
            foreach (var address in words)
            {
                Assert.True(host.TryReadGuestWord(address, out _));
            }

            var assumptions = concurrent.EndConcurrentResolve();
            host.NoteGuestMemoryWrite(GuestBase + 0x80000, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
            var recheck = Time(() => Assert.True(concurrent.IsStillCurrent(assumptions)));
            concurrent.ReleaseAssumptions(assumptions);
            output.WriteLine(
                $"owned_lines={(ownedLines ? 1 : 0)} scoped_serial={(scopedSerial ? 1 : 0)}: resolve {resolve:F2} us, " +
                $"recheck {recheck:F2} us; resolutions {perResolve}");
        }
    }
}
