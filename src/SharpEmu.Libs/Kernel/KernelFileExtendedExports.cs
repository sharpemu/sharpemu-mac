// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Diagnostics;
using System.Buffers;
using SharpEmu.HLE.FileSystem;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Threading;

namespace SharpEmu.Libs.Kernel;

/// <summary>
/// Positional and scatter/gather file I/O, synchronization, renaming, fcntl,
/// polling and asynchronous I/O (AIO) that complement the base file exports.
/// These share the same guest FD table and path-resolution helpers via the
/// partial <see cref="KernelMemoryCompatExports"/> class.
/// </summary>
public static partial class KernelMemoryCompatExports
{
    // fcntl commands (FreeBSD numbering used by the PS4/PS5 libc).
    private const int F_DUPFD = 0;
    private const int F_GETFD = 1;
    private const int F_SETFD = 2;
    private const int F_GETFL = 3;
    private const int F_SETFL = 4;

    private static Stream? GetOpenFile(int fd)
    {
        lock (_fdGate)
        {
            return _openFiles.TryGetValue(fd, out var stream) ? stream : null;
        }
    }

    // A descriptor is a host file or a read-only file served from a mounted archive.
    private static int ReadOpenFileAt(Stream stream, Span<byte> buffer, long offset) => stream switch
    {
        App0OverlayFileStream overlaid => overlaid.ReadAt(buffer, offset),
        FileStream file => RandomAccess.Read(file.SafeFileHandle, buffer, offset),
        ZArchiveFileStream archived => archived.ReadAt(buffer, offset),
        _ => throw new IOException("The descriptor does not support positional reads."),
    };

    private static void WriteOpenFileAt(Stream stream, ReadOnlySpan<byte> payload, long offset)
    {
        if (stream is not FileStream file)
        {
            throw new IOException("The descriptor is read-only.");
        }

        RandomAccess.Write(file.SafeFileHandle, payload, offset);
    }

    private static void FlushOpenFileToDisk(Stream stream)
    {
        if (stream is FileStream file)
        {
            file.Flush(flushToDisk: true);
        }
    }

    // ---- Positional read/write (do not move the file offset) ----

    [SysAbiExport(Nid = "ezv-RSBNKqI", ExportName = "pread",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPread(CpuContext ctx)
    {
        var result = KernelPreadCore(ctx);
        return result == (int)OrbisGen2Result.ORBIS_GEN2_OK
            ? 0
            : PosixFailure(ctx, result, notFoundErrno: Ebadf);
    }

    [SysAbiExport(Nid = "+r3rMFwItV4", ExportName = "sceKernelPread",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelPread(CpuContext ctx) => KernelPreadCore(ctx);

    private static int KernelPreadCore(CpuContext ctx)
    {
        var profile = FileIoProfile.Begin();
        var outcome = KernelPreadCoreInner(ctx);
        FileIoProfile.EndNoted(
            FileIoProfile.Op.Pread,
            profile,
            bytes: outcome == (int)OrbisGen2Result.ORBIS_GEN2_OK
                ? unchecked((long)ctx[CpuRegister.Rax])
                : 0,
            failed: outcome != (int)OrbisGen2Result.ORBIS_GEN2_OK);
        return outcome;
    }

    private static int KernelPreadCoreInner(CpuContext ctx)
    {
        var fd = unchecked((int)ctx[CpuRegister.Rdi]);
        var bufferAddress = ctx[CpuRegister.Rsi];
        var requested = (int)Math.Min(ctx[CpuRegister.Rdx], int.MaxValue);
        var offset = unchecked((long)ctx[CpuRegister.Rcx]);
        if (requested < 0 || (requested > 0 && bufferAddress == 0) || offset < 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var stream = GetOpenFile(fd);
        if (stream is null)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        FileIoProfile.NotePath(OpenFileName(stream));

        if (requested == 0)
        {
            ctx[CpuRegister.Rax] = 0;
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        // Titles that stream from .pak archives pread in chunks well past the
        // 85 KiB large-object threshold, so a fresh array per call put every
        // read's staging buffer on the LOH. The read goes straight into the
        // guest's host mapping when it hands the range out in place, and
        // through a pooled buffer otherwise.
        byte[]? buffer = null;
        try
        {
            int read;
            try
            {
                if (!ctx.Memory.TryFillInPlace(bufferAddress, requested, (Stream: stream, Offset: offset),
                        static (state, destination) => ReadOpenFileAt(state.Stream, destination, state.Offset), out read))
                {
                    buffer = ArrayPool<byte>.Shared.Rent(requested);
                    read = ReadOpenFileAt(stream, buffer.AsSpan(0, requested), offset);
                }
            }
            catch (IOException)
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
            }

            if (read > 0 && stream is FileStream file)
            {
                // pread never advances the file offset, so the kernel's own
                // read-ahead cannot see that a chunked archive read is
                // sequential; say so explicitly. A file served from a mounted
                // archive has no host handle to advise.
                AdvisePreadReadAhead(fd, file, offset, read);
            }

            SharpEmu.HLE.GuestLoadProgress.NoteFileRead(read);

            if (read > 0 && buffer is not null && !ctx.Memory.TryWrite(bufferAddress, buffer.AsSpan(0, read)))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }

            ctx[CpuRegister.Rax] = unchecked((ulong)read);
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }
        finally
        {
            if (buffer is not null)
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }

    // Per guest descriptor: where the next read would have to start to count as
    // continuing, plus the file length, so deciding whether to hint costs no
    // syscall of its own. Dropped when the descriptor closes.
    private readonly record struct PreadStream(long NextOffset, long Length);

    private static readonly ConcurrentDictionary<int, PreadStream> _preadStreams = new();

    private static void AdvisePreadReadAhead(int fd, FileStream stream, long offset, int read)
    {
        if (!HostReadAhead.Enabled)
        {
            return;
        }

        var completed = offset + read;
        long length;
        bool continuesPreviousRead;
        if (_preadStreams.TryGetValue(fd, out var tracked))
        {
            length = tracked.Length;
            continuesPreviousRead = tracked.NextOffset == offset;
        }
        else
        {
            try
            {
                length = RandomAccess.GetLength(stream.SafeFileHandle);
            }
            catch (Exception exception) when (
                exception is IOException or ObjectDisposedException)
            {
                return;
            }

            continuesPreviousRead = false;
        }

        _preadStreams[fd] = new PreadStream(completed, length);
        if (!continuesPreviousRead || completed >= length)
        {
            return;
        }

        HostReadAhead.Advise(
            stream.SafeFileHandle,
            completed,
            (int)Math.Min(HostReadAhead.WindowBytes, length - completed));
    }

    internal static void ForgetPreadStream(int fd) => _preadStreams.TryRemove(fd, out _);

    [SysAbiExport(Nid = "C2kJ-byS5rM", ExportName = "pwrite",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPwrite(CpuContext ctx)
    {
        var result = KernelPwriteCore(ctx);
        return result == (int)OrbisGen2Result.ORBIS_GEN2_OK
            ? 0
            : PosixFailure(ctx, result, notFoundErrno: Ebadf);
    }

    [SysAbiExport(Nid = "nKWi-N2HBV4", ExportName = "sceKernelPwrite",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelPwrite(CpuContext ctx) => KernelPwriteCore(ctx);

    private static int KernelPwriteCore(CpuContext ctx)
    {
        var profile = FileIoProfile.Begin();
        var outcome = KernelPwriteCoreInner(ctx);
        FileIoProfile.End(
            FileIoProfile.Op.Pwrite,
            profile,
            bytes: outcome == (int)OrbisGen2Result.ORBIS_GEN2_OK
                ? unchecked((long)ctx[CpuRegister.Rax])
                : 0,
            failed: outcome != (int)OrbisGen2Result.ORBIS_GEN2_OK);
        return outcome;
    }

    private static int KernelPwriteCoreInner(CpuContext ctx)
    {
        var fd = unchecked((int)ctx[CpuRegister.Rdi]);
        var bufferAddress = ctx[CpuRegister.Rsi];
        var requested = (int)Math.Min(ctx[CpuRegister.Rdx], int.MaxValue);
        var offset = unchecked((long)ctx[CpuRegister.Rcx]);
        if (requested < 0 || (requested > 0 && bufferAddress == 0) || offset < 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var stream = GetOpenFile(fd);
        if (stream is null)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        if (requested == 0)
        {
            ctx[CpuRegister.Rax] = 0;
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        var payload = ArrayPool<byte>.Shared.Rent(requested);
        try
        {
            if (!ctx.Memory.TryRead(bufferAddress, payload.AsSpan(0, requested)))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }

            try
            {
                WriteOpenFileAt(stream, payload.AsSpan(0, requested), offset);
            }
            catch (IOException)
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
            }

            ctx[CpuRegister.Rax] = unchecked((ulong)requested);
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(payload);
        }
    }

    // ---- Synchronization ----

    [SysAbiExport(Nid = "juWbTNM+8hw", ExportName = "fsync",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixFsync(CpuContext ctx)
    {
        var result = KernelFsyncCore(ctx);
        return result == (int)OrbisGen2Result.ORBIS_GEN2_OK
            ? 0
            : PosixFailure(ctx, result, notFoundErrno: Ebadf);
    }

    [SysAbiExport(Nid = "fTx66l5iWIA", ExportName = "sceKernelFsync",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelFsync(CpuContext ctx) => KernelFsyncCore(ctx);

    [SysAbiExport(Nid = "KIbJFQ0I1Cg", ExportName = "fdatasync",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixFdatasync(CpuContext ctx) => PosixFsync(ctx);

    [SysAbiExport(Nid = "30Rh4ixbKy4", ExportName = "sceKernelFdatasync",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelFdatasync(CpuContext ctx) => KernelFsyncCore(ctx);

    private static int KernelFsyncCore(CpuContext ctx)
    {
        var fd = unchecked((int)ctx[CpuRegister.Rdi]);
        var stream = GetOpenFile(fd);
        if (stream is null)
        {
            if (fd is 0 or 1 or 2)
            {
                ctx[CpuRegister.Rax] = 0;
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
            }

            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        try
        {
            FlushOpenFileToDisk(stream);
        }
        catch (IOException)
        {
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(Nid = "Y2OqwJQ3lr8", ExportName = "sync",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixSync(CpuContext ctx)
    {
        lock (_fdGate)
        {
            foreach (var stream in _openFiles.Values)
            {
                try { FlushOpenFileToDisk(stream); } catch (IOException) { }
            }
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // ---- Truncation ----

    [SysAbiExport(Nid = "ih4CD9-gghM", ExportName = "ftruncate",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixFtruncate(CpuContext ctx)
    {
        var result = KernelFtruncateCore(ctx);
        return result == (int)OrbisGen2Result.ORBIS_GEN2_OK
            ? 0
            : PosixFailure(ctx, result, notFoundErrno: Ebadf);
    }

    [SysAbiExport(Nid = "VW3TVZiM4-E", ExportName = "sceKernelFtruncate",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelFtruncate(CpuContext ctx) => KernelFtruncateCore(ctx);

    private static int KernelFtruncateCore(CpuContext ctx)
    {
        var fd = unchecked((int)ctx[CpuRegister.Rdi]);
        var length = unchecked((long)ctx[CpuRegister.Rsi]);
        if (length < 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var stream = GetOpenFile(fd);
        if (stream is null)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        // FreeBSD kern_ftruncate: EINVAL when the fd is not open for writing
        // (every /app0 descriptor, archive-backed files).
        if (!stream.CanWrite)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        try
        {
            lock (stream)
            {
                stream.SetLength(length);
            }
        }
        catch (Exception ex) when (ex is IOException or NotSupportedException or UnauthorizedAccessException)
        {
            return KernelErrno.Sce(HostExceptionToErrno(ex));
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(Nid = "ayrtszI7GBg", ExportName = "truncate",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixTruncate(CpuContext ctx)
    {
        var result = KernelTruncateCore(ctx);
        return result == (int)OrbisGen2Result.ORBIS_GEN2_OK
            ? 0
            : PosixFailure(ctx, result);
    }

    [SysAbiExport(Nid = "WlyEA-sLDf0", ExportName = "sceKernelTruncate",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelTruncate(CpuContext ctx) => KernelTruncateCore(ctx);

    private static int KernelTruncateCore(CpuContext ctx)
    {
        var pathAddress = ctx[CpuRegister.Rdi];
        var length = unchecked((long)ctx[CpuRegister.Rsi]);
        if (length < 0 || !TryReadNullTerminatedUtf8(ctx, pathAddress, MaxGuestStringLength, out var guestPath))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (IsReadOnlyGuestMutationPath(guestPath))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_EROFS;
        }

        var hostPath = ResolveGuestPath(guestPath);
        if (string.IsNullOrEmpty(hostPath))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        if (Directory.Exists(hostPath))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_EISDIR;
        }

        try
        {
            using var stream = new FileStream(hostPath, FileMode.Open, FileAccess.Write, GuestFileShare);
            stream.SetLength(length);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return KernelErrno.Sce(HostExceptionToErrno(ex));
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // ---- Rename ----

    [SysAbiExport(Nid = "NN01qLRhiqU", ExportName = "rename",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixRename(CpuContext ctx)
    {
        var result = KernelRenameCore(ctx);
        return result == (int)OrbisGen2Result.ORBIS_GEN2_OK
            ? 0
            : PosixFailure(ctx, result);
    }

    [SysAbiExport(Nid = "52NcYU9+lEo", ExportName = "sceKernelRename",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelRename(CpuContext ctx) => KernelRenameCore(ctx);

    private static int KernelRenameCore(CpuContext ctx)
    {
        if (!TryReadNullTerminatedUtf8(ctx, ctx[CpuRegister.Rdi], MaxGuestStringLength, out var fromGuest) ||
            !TryReadNullTerminatedUtf8(ctx, ctx[CpuRegister.Rsi], MaxGuestStringLength, out var toGuest))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (IsReadOnlyGuestMutationPath(fromGuest) || IsReadOnlyGuestMutationPath(toGuest))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_EROFS;
        }

        var fromHost = ResolveGuestPath(fromGuest);
        var toHost = ResolveGuestPath(toGuest);
        if (string.IsNullOrEmpty(fromHost) || string.IsNullOrEmpty(toHost))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        var errno = RenameHostPath(fromHost, toHost);
        if (errno != 0)
        {
            return KernelErrno.Sce(errno);
        }

        InvalidateNegativeStatCacheForPathAndAncestors(toGuest);
        AddNegativeStatCacheForGuestPath(fromGuest);
        InvalidateAprFileSizeCache(fromHost);
        InvalidateAprFileSizeCache(toHost);
        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    /// <summary>
    /// rename(2) on host paths with FreeBSD semantics: a file replaces a file,
    /// a directory replaces an empty directory, a file over a directory is
    /// EISDIR, a directory over a file ENOTDIR, over a non-empty directory
    /// ENOTEMPTY. Returns 0 or the errno.
    /// </summary>
    internal static int RenameHostPath(string fromHost, string toHost)
    {
        try
        {
            var fromIsDirectory = Directory.Exists(fromHost);
            if (!fromIsDirectory && !File.Exists(fromHost))
            {
                return OrbisErrno.ENOENT;
            }

            if (string.Equals(
                    Path.GetFullPath(fromHost),
                    Path.GetFullPath(toHost),
                    StringComparison.Ordinal))
            {
                return 0;
            }

            var toIsDirectory = Directory.Exists(toHost);
            var toIsFile = !toIsDirectory && File.Exists(toHost);
            if (fromIsDirectory)
            {
                if (toIsFile)
                {
                    return OrbisErrno.ENOTDIR;
                }

                if (toIsDirectory)
                {
                    if (Directory.EnumerateFileSystemEntries(toHost).Any())
                    {
                        return OrbisErrno.ENOTEMPTY;
                    }

                    Directory.Delete(toHost);
                }

                Directory.Move(fromHost, toHost);
                return 0;
            }

            if (toIsDirectory)
            {
                return OrbisErrno.EISDIR;
            }

            File.Move(fromHost, toHost, overwrite: true);
            return 0;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return HostExceptionToErrno(ex);
        }
    }

    // ---- Descriptor duplication ----

    [SysAbiExport(Nid = "iiQjzvfWDq0", ExportName = "dup",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixDup(CpuContext ctx)
    {
        var fd = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (_fdGate)
        {
            if (!_openFiles.TryGetValue(fd, out var stream))
            {
                return PosixFailure(
                    ctx,
                    (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND,
                    notFoundErrno: Ebadf);
            }

            // POSIX dup shares the open file description (and offset), which is
            // exactly the shared FileStream reference.
            var newFd = AllocateGuestFileDescriptor();
            if (newFd < 0)
            {
                return KernelErrno.Posix(ctx, OrbisErrno.EMFILE);
            }

            _openFiles[newFd] = stream;
            ctx[CpuRegister.Rax] = unchecked((ulong)newFd);
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(Nid = "wdUufa9g-D8", ExportName = "dup2",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixDup2(CpuContext ctx)
    {
        var oldFd = unchecked((int)ctx[CpuRegister.Rdi]);
        var newFd = unchecked((int)ctx[CpuRegister.Rsi]);
        lock (_fdGate)
        {
            if (!_openFiles.TryGetValue(oldFd, out var stream))
            {
                return PosixFailure(
                    ctx,
                    (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND,
                    notFoundErrno: Ebadf);
            }

            if (newFd < 0 || newFd >= GuestDescriptors.Limit)
            {
                return KernelErrno.Posix(ctx, OrbisErrno.EBADF);
            }

            if (oldFd == newFd)
            {
                ctx[CpuRegister.Rax] = unchecked((ulong)newFd);
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
            }

            // If newFd names an open file, dup2 closes it first; the stream
            // itself survives while another descriptor still shares it.
            if (_openFiles.TryGetValue(newFd, out var existing) &&
                !ReferenceEquals(existing, stream) &&
                !IsStreamSharedLocked(existing, newFd))
            {
                try { existing.Dispose(); } catch (IOException) { }
            }

            GuestDescriptors.TryClaim(newFd);
            _openFiles[newFd] = stream;
            ctx[CpuRegister.Rax] = unchecked((ulong)newFd);
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // ---- fcntl ----

    [SysAbiExport(Nid = "8nY19bKoiZk", ExportName = "fcntl",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixFcntl(CpuContext ctx)
    {
        var result = KernelFcntlCore(ctx);
        return result == (int)OrbisGen2Result.ORBIS_GEN2_OK
            ? 0
            : PosixFailure(ctx, result, notFoundErrno: Ebadf);
    }

    [SysAbiExport(Nid = "SoZkxZkCHaw", ExportName = "sceKernelFcntl",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelFcntl(CpuContext ctx) => KernelFcntlCore(ctx);

    private static int KernelFcntlCore(CpuContext ctx)
    {
        var fd = unchecked((int)ctx[CpuRegister.Rdi]);
        var command = unchecked((int)ctx[CpuRegister.Rsi]);
        var argument = unchecked((int)ctx[CpuRegister.Rdx]);

        switch (command)
        {
            case F_DUPFD:
                lock (_fdGate)
                {
                    if (!_openFiles.TryGetValue(fd, out var stream))
                    {
                        return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
                    }

                    // F_DUPFD: the lowest free descriptor >= argument.
                    if (argument < 0 || argument >= GuestDescriptors.Limit)
                    {
                        return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
                    }

                    var newFd = AllocateGuestFileDescriptor(argument);
                    if (newFd < 0)
                    {
                        return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_EMFILE;
                    }

                    _openFiles[newFd] = stream;
                    ctx[CpuRegister.Rax] = unchecked((ulong)newFd);
                }

                return (int)OrbisGen2Result.ORBIS_GEN2_OK;

            case F_GETFD:
            case F_GETFL:
                // No close-on-exec / status flags are tracked; report cleared.
                ctx[CpuRegister.Rax] = 0;
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;

            case F_SETFD:
            case F_SETFL:
                ctx[CpuRegister.Rax] = 0;
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;

            default:
                ctx[CpuRegister.Rax] = 0;
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }
    }
}
