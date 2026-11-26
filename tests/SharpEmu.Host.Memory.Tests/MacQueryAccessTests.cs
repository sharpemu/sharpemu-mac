// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#if SHARPEMU_HOST_MACOS
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using SharpEmu.HLE.Host;
using SharpEmu.Host.MacOS;
using Xunit;

namespace SharpEmu.Host.Memory.Tests;

/// <summary>
/// <see cref="IHostViewMemory.QueryAccess"/> on macOS answers from the region table, not the kernel. A partial unmap
/// maps the rest of a view again with what it reports, so it must say what the kernel enforces: page by page, after
/// page watches (no access, read-only) were set on single pages and runs, and nothing for a part that is unmapped.
/// </summary>
[SupportedOSPlatform("macos")]
public sealed unsafe class MacQueryAccessTests
{
    private const int VmRegionBasicInfo64 = 9;

    [Fact]
    public void QueryAccessMatchesTheKernelPageByPage()
    {
        var views = MacOSHost.CreateViewMemory();
        var page = views.PageSize;
        var size = 16 * page;
        Assert.True(views.TryCreateBacking(size, out var backing, out _));
        using var owned = backing!;
        var address = views.FindUnmappedRange(0xF0_0000_0000, size, 0xFC_0000_0000);
        Assert.NotEqual(0UL, address);
        Assert.Equal(address, views.ReserveHole(address, size));
        try
        {
            Assert.True(views.SplitHole(address, size));
            Assert.True(views.TryMapView(backing!, address, 0, 12 * page, HostPageProtection.ReadWrite, out _));
            // Page watches: a read watch on one page, a write watch on one page and on a run, a page made read-write
            // again inside the run, and a read watch at the end of the view.
            Assert.True(views.ChangeAccess(address + page, page, HostPageProtection.NoAccess));
            Assert.True(views.ChangeAccess(address + 2 * page, page, HostPageProtection.ReadOnly));
            Assert.True(views.ChangeAccess(address + 4 * page, 4 * page, HostPageProtection.ReadOnly));
            Assert.True(views.ChangeAccess(address + 6 * page, page, HostPageProtection.ReadWrite));
            Assert.True(views.ChangeAccess(address + 11 * page, page, HostPageProtection.NoAccess));

            var runs = views.QueryAccess(address, size);
            // The mapped part is described without gaps; the reserved tail past it is not described at all.
            var cursor = address;
            foreach (var run in runs)
            {
                Assert.Equal(cursor, run.Address);
                cursor += run.Size;
            }

            Assert.Equal(address + 12 * page, cursor);
            for (var offset = 0UL; offset < 12 * page; offset += page)
            {
                var run = runs.Single(candidate => candidate.Address <= address + offset &&
                                                   address + offset < candidate.Address + candidate.Size);
                Assert.True(
                    KernelProtection(address + offset) == run.Protection,
                    $"page +0x{offset:X}: table says {run.Protection}, kernel enforces {KernelProtection(address + offset)}");
            }

            // A sub-range query starts at the asked address, inside a run.
            var inner = views.QueryAccess(address + 5 * page, 2 * page);
            Assert.Equal(
                [new HostPageAccess(address + 5 * page, page, HostPageProtection.ReadOnly),
                 new HostPageAccess(address + 6 * page, page, HostPageProtection.ReadWrite)],
                inner);

            // A map failure names the native region the kernel holds there on macOS, and inside the GPU carveout says so.
            IHostMemory memory = MacOSHost.CreateMemory();
            Assert.Contains($"native[0x{address + page:X}+0x{page:X} prot=0/", memory.DescribeNativeRegion(address + page, page));
            if (OperatingSystem.IsMacOSVersionAtLeast(27))
            {
                Assert.Contains("macos27-carveout[0x1000000000,0x7000000000)", memory.DescribeNativeRegion(0x10_0000_0000, page));
            }

            Assert.True(views.UnmapView(address, 12 * page));
            Assert.Empty(views.QueryAccess(address, size));
        }
        finally
        {
            Assert.True(views.FreeHole(address, size));
        }
    }

    private static HostPageProtection KernelProtection(ulong address)
    {
        var regionAddress = address;
        ulong regionSize = 0;
        uint count = 9;
        var information = stackalloc int[9];
        var task = MachTaskSelf();
        Assert.Equal(0, MachVmRegion(task, ref regionAddress, ref regionSize, VmRegionBasicInfo64, information, ref count, out var objectName));
        if (objectName != 0) _ = MachPortDeallocate(task, objectName);
        Assert.True(regionAddress <= address && address - regionAddress < regionSize, $"0x{address:X} is not mapped");
        return (information[0] & 7) switch
        {
            0 => HostPageProtection.NoAccess,
            1 => HostPageProtection.ReadOnly,
            3 => HostPageProtection.ReadWrite,
            var other => throw new InvalidOperationException($"unexpected protection {other}"),
        };
    }

    [DllImport("libSystem.dylib", EntryPoint = "mach_task_self")]
    private static extern uint MachTaskSelf();

    [DllImport("libSystem.dylib", EntryPoint = "mach_vm_region")]
    private static extern int MachVmRegion(uint task, ref ulong address, ref ulong size, int flavor,
        int* information, ref uint informationCount, out uint objectName);

    [DllImport("libSystem.dylib", EntryPoint = "mach_port_deallocate")]
    private static extern int MachPortDeallocate(uint task, uint name);
}
#endif
