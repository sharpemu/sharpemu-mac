// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

/// <summary>
/// Process-wide factory for the host placeholder/view memory backend. Each
/// <see cref="Create"/> call returns a fresh instance: callers own their own
/// reservation bookkeeping.
/// </summary>
/// <remarks>
/// Registered separately from <see cref="HostPlatform"/> on purpose. View
/// memory is plain host virtual memory and is exercised by memory tests in
/// processes that cannot run guest code at all (a native ARM64 test host), so
/// it must not be gated behind a host platform that legitimately rejects such
/// a process.
/// </remarks>
public static class HostViewMemory
{
    private static Func<IHostViewMemory>? _factory;

    /// <summary>Registers the view-memory factory. Last registration wins.</summary>
    public static void Initialize(Func<IHostViewMemory> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);
        Volatile.Write(ref _factory, factory);
    }

    public static IHostViewMemory Create()
    {
        var factory = Volatile.Read(ref _factory) ??
            throw new InvalidOperationException(
                "No host view-memory backend has been registered. The application must call " +
                "HostViewMemory.Initialize(...) with the SharpEmu.Host.* backend for its target " +
                "OS before any guest work starts (see SharpEmu.CLI's HostComposition).");
        return factory();
    }

    /// <summary>
    /// Whether <paramref name="address"/> and <paramref name="size"/> describe a
    /// non-empty range that does not wrap the address space.
    /// </summary>
    /// <remarks>
    /// Host-package contract: the argument validation every
    /// <see cref="IHostViewMemory"/> backend applies before it calls into the OS,
    /// shared so the three per-OS packages reject the same inputs.
    /// </remarks>
    public static bool IsValidRange(ulong address, ulong size) =>
        address != 0 && size != 0 && size <= ulong.MaxValue - address;

    /// <summary>
    /// Whether <paramref name="offset"/> and <paramref name="size"/> name a
    /// page-aligned window that lies inside <paramref name="backing"/>.
    /// </summary>
    /// <remarks>Host-package contract; see <see cref="IsValidRange"/>.</remarks>
    public static bool IsValidOffset(HostBackingObject backing, ulong offset, ulong size, ulong pageSize) =>
        offset % pageSize == 0 && offset < backing.Size && size != 0 && size <= backing.Size - offset;

    // A page's access as the host reports it (the PAGE_* values; the Posix region table keeps the same values). False for an
    // access with a modifier such as a guard page, or one HostPageProtection has no value for.
    internal static bool TryGetProtection(uint raw, out HostPageProtection protection)
    {
        (var known, protection) = raw switch
        {
            HostMemory.PAGE_NOACCESS => (true, HostPageProtection.NoAccess),
            HostMemory.PAGE_READONLY => (true, HostPageProtection.ReadOnly),
            HostMemory.PAGE_READWRITE => (true, HostPageProtection.ReadWrite),
            HostMemory.PAGE_EXECUTE => (true, HostPageProtection.Execute),
            HostMemory.PAGE_EXECUTE_READ => (true, HostPageProtection.ReadExecute),
            HostMemory.PAGE_EXECUTE_READWRITE => (true, HostPageProtection.ReadWriteExecute),
            0x80 => (true, HostPageProtection.ExecuteWriteCopy),
            _ => (false, HostPageProtection.NoAccess),
        };
        return known;
    }

    // Walks the range with a host query that describes the run of equal access a page starts. A run it cannot describe (not
    // committed, an access TryGetProtection does not know) is left out, and so is the rest of the range once the query fails.
    public static IReadOnlyList<HostPageAccess> CollectAccess(ulong address, ulong size, Func<ulong, (bool Found, HostMemory.BasicInfo Info)> query)
    {
        var runs = new List<HostPageAccess>();
        var end = address + size;
        for (var current = address; current < end;)
        {
            var (found, info) = query(current);
            if (!found || info.BaseAddress + info.RegionSize <= current)
            {
                break;
            }

            var stop = Math.Min(end, info.BaseAddress + info.RegionSize);
            if (info.State == HostMemory.MEM_COMMIT && TryGetProtection(info.Protect, out var protection))
            {
                runs.Add(new HostPageAccess(current, stop - current, protection));
            }

            current = stop;
        }

        return runs;
    }
}
