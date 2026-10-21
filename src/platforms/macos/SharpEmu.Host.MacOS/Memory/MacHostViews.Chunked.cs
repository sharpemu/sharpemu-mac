// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.HLE.Host;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// The chunked guest backing (the default; SHARPEMU_GUEST_BACKING=shm restores one POSIX shm
/// object): 2 MiB purgeable named memory entries, released in 64 KiB chunks.
///
/// <para>
/// Why: under Rosetta (a 4 KiB VM map) no madvise-style call frees a page of the shm object
/// while the alias and the guest views map it, so memory the guest frees stays resident for
/// the life of the process (artifacts/memopt/single-copy-textures.md §3). A purgeable object
/// can be emptied in place: <c>mach_vm_purgable_control(EMPTY)</c> frees its pages at once,
/// every mapping stays valid and reads zero, and NONVOLATILE makes it ordinary memory again.
/// Chunks are 64 KiB because that is the granule titles allocate and BatchMap direct memory
/// in (Demon's Souls scatters a texture over 64 KiB physical pages). The flexible region uses
/// the same size: it is 448 MiB (7168 chunks) and is freed by unmapping, so one rule serves both.
/// </para>
///
/// <para>
/// Why the objects are larger than the chunks: under Rosetta a Metal no-copy window over the
/// backing costs about the square of the objects it spans (IOKit makes one memory entry per
/// object, and for a sub-page map each one re-simplifies every map entry left in the range).
/// 64 KiB objects held windows to 32 MiB, and Demon's Souls copied its larger bindings through
/// the ring every command buffer. At 2 MiB a 1 GiB window spans 512 objects and is made in
/// ~2.4 ms (artifacts/memopt/exp-window). EMPTY is whole-object and leaves wired pages alone
/// (they keep their bytes), so a release of only some chunks of an object wires the rest of
/// it first (<c>mlock</c>), empties it and unwires it. The released chunks' pages are freed
/// through every mapping and the rest keep every byte. A writer racing the release cannot
/// lose a write, because every page it can reach outside the released chunks stays wired.
/// That costs ~0.6 ms per partly released object under Rosetta, and it faults in the
/// object's other chunks: one that held no pages comes back as zero pages. A release of
/// whole objects is the plain EMPTY.
/// </para>
///
/// <para>
/// Layout: a reservation of the alias size, 64 KiB aligned, with each object mapped over it
/// (<c>mach_vm_map</c> FIXED|OVERWRITE, read-write, maximum read-write); the entry ports are
/// dropped at once, the mapping keeps each object alive. A guest view is
/// <c>mach_vm_remap(copy=FALSE)</c> of the alias range: the same objects, so it aliases the
/// alias and every other view exactly as an shm view does, for any page-aligned offset and
/// size, then <c>mprotect</c> to the view's protection. Execute is dropped exactly as the shm
/// view drops it: <c>mmap(MAP_SHARED, PROT_READ|PROT_WRITE|PROT_EXEC)</c> of the shm object
/// succeeds but maps read-write with a read-write maximum, on arm64 and under Rosetta alike
/// (artifacts/memopt/exp/chunked-rwx.c), so an executable guest view has never been executable
/// and a later protection change that asks for execute fails the same way. Holes, protection changes and unmaps
/// are the same calls as for shm. Every object mapping carries the <see cref="GuestChunkTag"/>
/// VM tag and a view inherits it from the alias, so scripts/macos/memstat.c classifies the objects as
/// guest memory whether or not a view maps them.
/// </para>
/// </summary>
internal sealed unsafe partial class MacHostViews
{
    internal const ulong GuestChunkBytes = 64 * 1024;

    // The purgeable object behind 32 chunks (see the class summary).
    internal const ulong GuestObjectBytes = 2 * 1024 * 1024;

    // VM_MEMORY_APPLICATION_SPECIFIC_11: the user tag of chunk mappings (memstat keys on it).
    internal const int GuestChunkTag = 250;

    private const int VM_FLAGS_ANYWHERE = 0x0001;
    private const int VM_FLAGS_OVERWRITE = 0x4000;
    private const int GuestChunkTagFlags = GuestChunkTag << 24;
    private const int MAP_MEM_NAMED_CREATE = 0x020000;
    private const int MAP_MEM_PURGABLE = 0x040000;
    private const int VM_PROT_NONE = 0;
    private const int VM_PROT_READ_WRITE = 3;
    private const uint VM_INHERIT_SHARE = 0;
    private const int VM_PURGABLE_SET_STATE = 0;
    private const int VM_PURGABLE_NONVOLATILE = 0;
    private const int VM_PURGABLE_EMPTY = 2;

    // Chunks emptied before one residency check: a 4096-entry mincore vector at 4 KiB pages.
    // A whole number of objects: batches end on multiples of it, so none splits an object.
    private const ulong DiscardBatchChunks = 256;

    // Partial releases wire, empty and unwire one at a time: two at once over one object would
    // each wire the chunks the other releases. Nothing is taken while it is held.
    private static readonly object PartialEmptyGate = new();

    /// <summary>Test override of SHARPEMU_GUEST_BACKING for backings created on this thread; null reads the variable.</summary>
    [ThreadStatic]
    internal static bool? ChunkedBackingForTests;

    // On unless SHARPEMU_GUEST_BACKING=shm. It was opt-in while Metal windows over 64 KiB
    // objects were capped at 32 MiB (DS in-world 0.15 FPS against 3.4 with shm, runs
    // 20261018-225430 / -230850); 2 MiB objects let them grow to the 1 GiB shm windows reach.
    private static bool UseChunkedBacking => ChunkedBackingForTests ?? !string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_GUEST_BACKING"), "shm", StringComparison.OrdinalIgnoreCase);

    // Called under HostViewRegions.Gate after the ledger preflight, like the shm path.
    private bool TryCreateChunkedBackingLocked(ulong size, out HostBackingObject? backing, out HostViewFailure failure)
    {
        backing = null;
        failure = HostViewFailure.BackingUnavailable;
        if (FailAliasMapForTests)
        {
            return false;
        }

        var aliasSize = AlignUp(size, PageSize);
        var task = mach_task_self();
        // Mach's ANYWHERE search begins at this address; keep the host alias
        // above guest application VA instead of consuming a future fixed arena.
        ulong alias = HostAliasStart;
        if (mach_vm_map(task, ref alias, aliasSize, GuestChunkBytes - 1, VM_FLAGS_ANYWHERE | GuestChunkTagFlags, 0, 0, 0,
                VM_PROT_NONE, VM_PROT_READ_WRITE, VM_INHERIT_SHARE) != KERN_SUCCESS)
        {
            return false;
        }

        for (ulong offset = 0; offset < aliasSize; offset += GuestObjectBytes)
        {
            if (!TryMapObject(task, alias + offset, Math.Min(GuestObjectBytes, aliasSize - offset)))
            {
                // Only this new reservation exists; deallocating it drops every chunk made so far.
                mach_vm_deallocate(task, alias, aliasSize);
                return false;
            }
        }

        try
        {
            backing = new HostBackingObject(alias, size, PageSize, ReleaseOwnedBackingObject)
            {
                ChunkBytes = GuestChunkBytes,
                ObjectBytes = GuestObjectBytes,
                DiscardChunks = DiscardChunks,
                CreateGpuAlias = ChunkedGpuAlias.TryCreate,
            };
            MacMappedCopyState.Ledger?.RegisterBacking(backing);
        }
        catch (Exception error)
        {
            // No guest view exists yet; the alias is the only owner of the chunks.
            var deallocated = mach_vm_deallocate(task, alias, aliasSize);
            if (deallocated != KERN_SUCCESS)
            {
                MacMappedCopyState.Ledger?.QuarantineUnknownAlias();
                try { error.Data["BackingSetupCleanup"] = $"alias=0x{alias:X};size={aliasSize};mach_vm_deallocate={deallocated}"; }
                catch { } // primary construction/publication failure remains authoritative
            }

            backing = null;
            throw;
        }

        failure = HostViewFailure.None;
        return true;
    }

    private static bool TryMapObject(uint task, ulong address, ulong length)
    {
        var entrySize = length;
        if (mach_make_memory_entry_64(task, ref entrySize, 0, MAP_MEM_NAMED_CREATE | MAP_MEM_PURGABLE | VM_PROT_READ_WRITE,
                out var entry, 0) != KERN_SUCCESS)
        {
            return false;
        }

        var target = address;
        var mapped = entrySize >= length && mach_vm_map(task, ref target, length, 0,
            VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE | GuestChunkTagFlags, entry, 0, 0, VM_PROT_READ_WRITE, VM_PROT_READ_WRITE,
            VM_INHERIT_SHARE) == KERN_SUCCESS && target == address;
        // The mapping holds the object; the port is not needed again (views remap the alias).
        mach_port_deallocate(task, entry);
        return mapped;
    }

    // Called under backing.Gate with the range validated, in place of the shm MAP_FIXED view.
    private bool TryMapChunkedViewCore(HostBackingObject backing, ulong address, ulong offset, ulong size,
        HostPageProtection protection, out HostViewFailure failure)
    {
        var task = mach_task_self();
        var target = address;
        // No tag in the flags: remap refuses one (KERN_INVALID_ARGUMENT) and copies the alias tag.
        if (mach_vm_remap(task, ref target, size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, task,
                backing.AliasBase + offset, 0, out _, out _, VM_INHERIT_SHARE) != KERN_SUCCESS)
        {
            failure = HostViewFailure.FixedMapFailed;
            return false;
        }

        if (target != address)
        {
            mach_vm_deallocate(task, target, size);
            failure = HostViewFailure.WrongHostAddress;
            return false;
        }

        // The view's maximum is the alias's read-write; execute is dropped as for shm (above).
        if (mprotect((nint)address, (nuint)size, GetNativeProtection(protection) & ~PROT_EXEC) != 0)
        {
            // Put the hole back, as an unmap would, rather than leave a read-write view.
            _ = mmap((nint)address, (nuint)size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
            failure = HostViewFailure.FixedMapFailed;
            return false;
        }

        failure = HostViewFailure.None;
        return true;
    }

    // HostBackingObject.DiscardChunks. EMPTY frees an object's pages (resident and compressed)
    // at once, through every mapping; NONVOLATILE right after makes it ordinary memory again
    // that reads zero. An object the range covers only in part has the rest of it wired first
    // (TryEmptyPartOfObject). A chunk wired by a command buffer keeps its pages and contents
    // through EMPTY, and one touched meanwhile has pages again, so whatever is still resident
    // after a batch is zero-filled instead: the caller is promised zeros either way.
    private static HostBackingDiscard DiscardChunks(HostBackingObject backing, ulong offset, ulong size)
    {
        var chunk = backing.ChunkBytes;
        if (chunk == 0 || size == 0 || offset % chunk != 0 || offset >= backing.Size || size > backing.Size - offset ||
            (size % chunk != 0 && offset + size != backing.Size))
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"0x{offset:X}+0x{size:X} is not whole chunks of the backing.");
        }

        var task = mach_task_self();
        var alias = backing.AliasBase;
        var objectBytes = Math.Max(backing.ObjectBytes, chunk);
        // The last chunk (and object) reaches over the alias padding past Size.
        var end = offset + size == backing.Size ? backing.NativeAliasSize : offset + size;
        var page = (ulong)Environment.SystemPageSize;
        var batchBytes = DiscardBatchChunks * chunk;
        var vector = stackalloc byte[(int)(batchBytes / page)];
        var wasEmptied = stackalloc bool[(int)DiscardBatchChunks];
        ulong emptied = 0, emptiedChunks = 0, zeroFilled = 0;
        for (var batch = offset; batch < end;)
        {
            // Batches end on multiples of their size, so no object is split between two.
            var batchEnd = Math.Min(end, (batch / batchBytes + 1) * batchBytes);
            for (var objectStart = batch / objectBytes * objectBytes; objectStart < batchEnd; objectStart += objectBytes)
            {
                var objectEnd = Math.Min(objectStart + objectBytes, backing.NativeAliasSize);
                var from = Math.Max(batch, objectStart);
                var to = Math.Min(batchEnd, objectEnd);
                var done = from == objectStart && to == objectEnd
                    ? TryEmptyObject(task, alias, objectStart)
                    : TryEmptyPartOfObject(task, alias, objectStart, objectEnd, from, to);
                for (var cursor = from; cursor < to; cursor += chunk)
                {
                    wasEmptied[(int)((cursor - batch) / chunk)] = done;
                }
            }

            // No residency answer means every chunk is treated as still resident.
            var known = mincore((nint)(alias + batch), (nuint)(batchEnd - batch), vector) == 0;
            var slot = 0;
            for (var cursor = batch; cursor < batchEnd; cursor += chunk, slot++)
            {
                var length = Math.Min(chunk, batchEnd - cursor);
                var resident = !known || !wasEmptied[slot];
                var first = (cursor - batch) / page;
                for (var index = first; index < first + length / page && !resident; index++)
                {
                    resident = (vector[index] & 1) != 0;
                }

                if (resident)
                {
                    NativeMemory.Clear((void*)(alias + cursor), (nuint)length);
                    zeroFilled += length;
                }
                else
                {
                    emptied += length;
                    emptiedChunks++;
                }
            }

            batch = batchEnd;
        }

        return new HostBackingDiscard(emptied, emptiedChunks, zeroFilled);
    }

    // A whole object: EMPTY, then NONVOLATILE. False when it could not be emptied.
    private static bool TryEmptyObject(uint task, ulong alias, ulong objectStart)
    {
        var state = VM_PURGABLE_EMPTY;
        if (mach_vm_purgable_control(task, alias + objectStart, VM_PURGABLE_SET_STATE, ref state) != KERN_SUCCESS)
        {
            return false;
        }

        MakeNonvolatile(task, alias, objectStart);
        return true;
    }

    // Chunks [from, to) of the object at [objectStart, objectEnd): the rest of the object is
    // wired so the EMPTY passes over it (a wired page keeps its bytes), then unwired. Every
    // page outside [from, to) is wired before the EMPTY, so a write landing there meanwhile
    // lands on a wired page and stays. False, with nothing emptied, when the rest could not be
    // wired (the caller then zero-fills the chunks).
    private static bool TryEmptyPartOfObject(uint task, ulong alias, ulong objectStart, ulong objectEnd, ulong from, ulong to)
    {
        var below = from - objectStart;
        var above = objectEnd - to;
        lock (PartialEmptyGate)
        {
            var lowWired = below == 0 || mlock((nint)(alias + objectStart), (nuint)below) == 0;
            var highWired = lowWired && (above == 0 || mlock((nint)(alias + to), (nuint)above) == 0);
            try
            {
                if (!highWired)
                {
                    return false;
                }

                var state = VM_PURGABLE_EMPTY;
                if (mach_vm_purgable_control(task, alias + objectStart, VM_PURGABLE_SET_STATE, ref state) != KERN_SUCCESS)
                {
                    return false;
                }

                MakeNonvolatile(task, alias, objectStart);
                return true;
            }
            finally
            {
                if (below != 0 && lowWired)
                {
                    _ = munlock((nint)(alias + objectStart), (nuint)below);
                }

                if (above != 0 && highWired)
                {
                    _ = munlock((nint)(alias + to), (nuint)above);
                }
            }
        }
    }

    private static void MakeNonvolatile(uint task, ulong alias, ulong objectStart)
    {
        var state = VM_PURGABLE_NONVOLATILE;
        if (mach_vm_purgable_control(task, alias + objectStart, VM_PURGABLE_SET_STATE, ref state) != KERN_SUCCESS)
        {
            // An object left empty or volatile could lose later writes: never continue.
            throw new InvalidOperationException(
                $"The guest backing object at offset 0x{objectStart:X} could not be made nonvolatile again.");
        }
    }

    // HostBackingObject.CreateGpuAlias: the alias remapped once more (copy=FALSE, so the same
    // objects) for the GPU's windows, and one non-purgeable 64 KiB dummy object whose port is
    // kept so it can be mapped over any number of punched chunks. Measured under Rosetta on an
    // M4 (artifacts/memopt/single-copy-textures.md §4): remapping an 18 GiB alias of 64 KiB
    // objects took ~0.3 s, a punch or a fill ~20 µs, and the remap keeps one map entry per
    // object, so a window over it costs what one over the alias does, plus two objects' worth
    // for every hole inside it (the dummy splits the object around it).
    private sealed class ChunkedGpuAlias : IHostBackingGpuAlias
    {
        private readonly HostBackingObject _backing;
        private readonly uint _dummy;
        private readonly ulong _dummyView;
        private int _disposed;

        private ChunkedGpuAlias(HostBackingObject backing, ulong address, uint dummy, ulong dummyView)
        {
            _backing = backing;
            Base = address;
            Size = backing.NativeAliasSize;
            _dummy = dummy;
            _dummyView = dummyView;
        }

        public ulong Base { get; }

        public ulong Size { get; }

        public static IHostBackingGpuAlias? TryCreate(HostBackingObject backing)
        {
            if (backing.ChunkBytes != GuestChunkBytes || backing.IsDisposed)
            {
                return null;
            }

            var task = mach_task_self();
            var size = backing.NativeAliasSize;
            ulong address = 0;
            if (mach_vm_remap(task, ref address, size, GuestChunkBytes - 1, VM_FLAGS_ANYWHERE, task, backing.AliasBase, 0,
                    out _, out _, VM_INHERIT_SHARE) != KERN_SUCCESS)
            {
                return null;
            }

            var entrySize = GuestChunkBytes;
            if (mach_make_memory_entry_64(task, ref entrySize, 0, MAP_MEM_NAMED_CREATE | VM_PROT_READ_WRITE, out var dummy, 0) !=
                KERN_SUCCESS)
            {
                mach_vm_deallocate(task, address, size);
                return null;
            }

            ulong view = 0;
            if (entrySize < GuestChunkBytes ||
                mach_vm_map(task, ref view, GuestChunkBytes, GuestChunkBytes - 1, VM_FLAGS_ANYWHERE | GuestChunkTagFlags, dummy, 0, 0,
                    VM_PROT_READ_WRITE, VM_PROT_READ_WRITE, VM_INHERIT_SHARE) != KERN_SUCCESS)
            {
                mach_port_deallocate(task, dummy);
                mach_vm_deallocate(task, address, size);
                return null;
            }

            return new ChunkedGpuAlias(backing, address, dummy, view);
        }

        public bool TryPunch(ulong offset, ulong size)
        {
            if (!IsWholeChunks(offset, size))
            {
                return false;
            }

            var task = mach_task_self();
            for (var cursor = offset; cursor < offset + size; cursor += GuestChunkBytes)
            {
                var length = Math.Min(GuestChunkBytes, Size - cursor);
                var target = Base + cursor;
                if (mach_vm_map(task, ref target, length, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE | GuestChunkTagFlags, _dummy, 0, 0,
                        VM_PROT_READ_WRITE, VM_PROT_READ_WRITE, VM_INHERIT_SHARE) != KERN_SUCCESS || target != Base + cursor)
                {
                    return false;
                }
            }

            return true;
        }

        public bool TryFill(ulong offset, ulong size)
        {
            if (!IsWholeChunks(offset, size))
            {
                return false;
            }

            var task = mach_task_self();
            var length = Math.Min(size, Size - offset);
            var target = Base + offset;
            return mach_vm_remap(task, ref target, length, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, task,
                       _backing.AliasBase + offset, 0, out _, out _, VM_INHERIT_SHARE) == KERN_SUCCESS &&
                   target == Base + offset;
        }

        public void FillDummy(byte value) => new Span<byte>((void*)_dummyView, (int)GuestChunkBytes).Fill(value);

        private bool IsWholeChunks(ulong offset, ulong size) =>
            Volatile.Read(ref _disposed) == 0 && size != 0 && offset % GuestChunkBytes == 0 && offset < Size &&
            size <= Size - offset && (size % GuestChunkBytes == 0 || offset + size == Size);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            var task = mach_task_self();
            mach_vm_deallocate(task, Base, Size);
            mach_vm_deallocate(task, _dummyView, GuestChunkBytes);
            mach_port_deallocate(task, _dummy);
        }
    }

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_vm_map(uint task, ref ulong address, ulong size, ulong mask, int flags,
        uint memoryObject, ulong offset, int copy, int currentProtection, int maximumProtection, uint inheritance);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_vm_remap(uint targetTask, ref ulong targetAddress, ulong size, ulong mask, int flags,
        uint sourceTask, ulong sourceAddress, int copy, out int currentProtection, out int maximumProtection,
        uint inheritance);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_make_memory_entry_64(uint task, ref ulong size, ulong offset, int permission,
        out uint objectHandle, uint parentEntry);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_vm_purgable_control(uint task, ulong address, int control, ref int state);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_port_deallocate(uint task, uint name);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int mincore(nint address, nuint length, byte* vector);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int mlock(nint address, nuint length);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int munlock(nint address, nuint length);
}
