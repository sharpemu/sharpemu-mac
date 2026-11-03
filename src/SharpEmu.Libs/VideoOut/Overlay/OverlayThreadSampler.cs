// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>
/// Reads the cumulative CPU time of every thread in this process, keyed by the OS thread id
/// (macOS thread_identifier_info / pthread_threadid_np, Linux tid, Windows thread id), with
/// the thread's name. One instance is driven from one thread (the telemetry builder).
/// </summary>
/// <remarks>
/// macOS: task_threads + thread_info(THREAD_EXTENDED_INFO) - nanosecond user+system time and
/// the pthread name, and valid for Rosetta-translated threads (CLOCK_THREAD_CPUTIME_ID is not).
/// Linux: /proc/self/task/&lt;tid&gt;/schedstat (nanoseconds on CPU) and comm.
/// Windows: GetThreadTimes per thread (100 ns units, updated at the scheduler tick) and
/// GetThreadDescription.
/// </remarks>
internal abstract partial class OverlayThreadSampler
{
    internal readonly record struct ThreadCpu(ulong ThreadId, long CpuNanoseconds, string Name);

    /// <summary>Fills <paramref name="into"/> with every live thread; returns false if the platform read failed.</summary>
    public abstract bool Sample(List<ThreadCpu> into);

    /// <summary>The OS id of the calling thread, in the same id space as <see cref="ThreadCpu.ThreadId"/>.</summary>
    public static ulong CurrentThreadId()
    {
        if (OperatingSystem.IsMacOS())
        {
            return MacNative.pthread_threadid_np(0, out var id) == 0 ? id : 0;
        }

        if (OperatingSystem.IsLinux())
        {
            return (ulong)LinuxNative.gettid();
        }

        if (OperatingSystem.IsWindows())
        {
            return WindowsNative.GetCurrentThreadId();
        }

        return (ulong)Environment.CurrentManagedThreadId;
    }

    public static OverlayThreadSampler? CreateForCurrentProcess()
    {
        try
        {
            if (OperatingSystem.IsMacOS()) return new MacSampler();
            if (OperatingSystem.IsLinux()) return new LinuxSampler();
            if (OperatingSystem.IsWindows()) return new WindowsSampler();
        }
        catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException or PlatformNotSupportedException)
        {
        }

        return null;
    }

    private sealed unsafe class MacSampler : OverlayThreadSampler
    {
        private const int ThreadExtendedInfo = 5;
        private const uint ThreadExtendedInfoCount = 28;
        private const int ThreadIdentifierInfo = 4;
        private const uint ThreadIdentifierInfoCount = 6;

        [StructLayout(LayoutKind.Sequential)]
        private struct ThreadExtendedInfoData
        {
            public ulong UserTime;
            public ulong SystemTime;
            public int CpuUsage;
            public int Policy;
            public int RunState;
            public int Flags;
            public int SleepTime;
            public int CurrentPriority;
            public int Priority;
            public int MaxPriority;
            public fixed byte Name[64];
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ThreadIdentifierInfoData
        {
            public ulong ThreadId;
            public ulong ThreadHandle;
            public ulong DispatchQueueAddress;
        }

        private readonly uint _task = MacNative.TaskSelf();
        // Names are decoded once per (thread, name hash) rather than per sample.
        private readonly Dictionary<ulong, (int Hash, string Name)> _names = [];

        public override bool Sample(List<ThreadCpu> into)
        {
            into.Clear();
            if (MacNative.task_threads(_task, out var array, out var count) != 0)
            {
                return false;
            }

            var ports = (uint*)array;
            try
            {
                for (var i = 0; i < count; i++)
                {
                    ThreadIdentifierInfoData identifier;
                    var identifierCount = ThreadIdentifierInfoCount;
                    if (MacNative.thread_info(ports[i], ThreadIdentifierInfo, &identifier, ref identifierCount) != 0)
                    {
                        continue;
                    }

                    ThreadExtendedInfoData info;
                    var infoCount = ThreadExtendedInfoCount;
                    if (MacNative.thread_info(ports[i], ThreadExtendedInfo, &info, ref infoCount) != 0)
                    {
                        continue;
                    }

                    var nameLength = 0;
                    while (nameLength < 64 && info.Name[nameLength] != 0) nameLength++;
                    var nameBytes = new ReadOnlySpan<byte>(info.Name, nameLength);
                    var hash = (int)System.IO.Hashing.XxHash32.HashToUInt32(nameBytes);
                    if (!_names.TryGetValue(identifier.ThreadId, out var cached) || cached.Hash != hash)
                    {
                        cached = (hash, Encoding.UTF8.GetString(nameBytes));
                        _names[identifier.ThreadId] = cached;
                    }

                    into.Add(new ThreadCpu(identifier.ThreadId, (long)(info.UserTime + info.SystemTime), cached.Name));
                }
            }
            finally
            {
                for (var i = 0; i < count; i++)
                {
                    _ = MacNative.mach_port_deallocate(_task, ports[i]);
                }

                _ = MacNative.vm_deallocate(_task, (nuint)array, (nuint)(count * sizeof(uint)));
            }

            if (_names.Count > into.Count * 2 + 64)
            {
                PruneNames(into);
            }

            return true;
        }

        private void PruneNames(List<ThreadCpu> live)
        {
            var keep = new HashSet<ulong>(live.Count);
            foreach (var thread in live) keep.Add(thread.ThreadId);
            foreach (var id in _names.Keys.ToArray())
            {
                if (!keep.Contains(id)) _names.Remove(id);
            }
        }
    }

    private sealed class LinuxSampler : OverlayThreadSampler
    {
        private readonly Dictionary<ulong, string> _names = [];
        private readonly byte[] _buffer = new byte[128];

        public override bool Sample(List<ThreadCpu> into)
        {
            into.Clear();
            try
            {
                foreach (var directory in Directory.EnumerateDirectories("/proc/self/task"))
                {
                    if (!ulong.TryParse(Path.GetFileName(directory.AsSpan()), out var tid))
                    {
                        continue;
                    }

                    var nanoseconds = ReadLeadingNumber(Path.Combine(directory, "schedstat"));
                    if (nanoseconds < 0)
                    {
                        continue;
                    }

                    if (!_names.TryGetValue(tid, out var name))
                    {
                        name = ReadText(Path.Combine(directory, "comm"));
                        _names[tid] = name;
                    }

                    into.Add(new ThreadCpu(tid, nanoseconds, name));
                }
            }
            catch (IOException)
            {
                return into.Count > 0;
            }
            catch (UnauthorizedAccessException)
            {
                return false;
            }

            if (_names.Count > into.Count * 2 + 64)
            {
                var keep = new HashSet<ulong>(into.Select(static thread => thread.ThreadId));
                foreach (var id in _names.Keys.ToArray())
                {
                    if (!keep.Contains(id)) _names.Remove(id);
                }
            }

            return true;
        }

        private long ReadLeadingNumber(string path)
        {
            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1, FileOptions.None);
                var read = stream.Read(_buffer, 0, _buffer.Length);
                long value = 0;
                var digits = 0;
                for (var i = 0; i < read && _buffer[i] is >= (byte)'0' and <= (byte)'9'; i++, digits++)
                {
                    value = value * 10 + (_buffer[i] - '0');
                }

                return digits > 0 ? value : -1;
            }
            catch (IOException)
            {
                return -1;
            }
        }

        private static string ReadText(string path)
        {
            try
            {
                return File.ReadAllText(path).Trim();
            }
            catch (IOException)
            {
                return string.Empty;
            }
        }
    }

    private sealed class WindowsSampler : OverlayThreadSampler
    {
        private const uint ThreadQueryLimitedInformation = 0x0800;
        private readonly Dictionary<ulong, (nint Handle, string Name)> _threads = [];

        public override bool Sample(List<ThreadCpu> into)
        {
            into.Clear();
            ProcessThreadCollection threads;
            try
            {
                using var process = Process.GetCurrentProcess();
                threads = process.Threads;
            }
            catch (InvalidOperationException)
            {
                return false;
            }

            foreach (ProcessThread thread in threads)
            {
                using (thread)
                {
                    var id = (ulong)thread.Id;
                    if (!_threads.TryGetValue(id, out var entry))
                    {
                        var handle = WindowsNative.OpenThread(ThreadQueryLimitedInformation, false, (uint)id);
                        entry = (handle, handle != 0 ? WindowsNative.Describe(handle) : string.Empty);
                        _threads[id] = entry;
                    }

                    if (entry.Handle != 0 &&
                        WindowsNative.GetThreadTimes(entry.Handle, out _, out _, out var kernel, out var user))
                    {
                        into.Add(new ThreadCpu(id, (kernel + user) * 100, entry.Name));
                    }
                }
            }

            if (_threads.Count > into.Count * 2 + 64)
            {
                var keep = new HashSet<ulong>(into.Select(static thread => thread.ThreadId));
                foreach (var (id, entry) in _threads.ToArray())
                {
                    if (keep.Contains(id)) continue;
                    if (entry.Handle != 0) WindowsNative.CloseHandle(entry.Handle);
                    _threads.Remove(id);
                }
            }

            return true;
        }
    }

    private static unsafe partial class MacNative
    {
        private const string LibSystem = "/usr/lib/libSystem.B.dylib";

        [LibraryImport(LibSystem)]
        internal static partial int pthread_threadid_np(nint thread, out ulong threadId);

        [LibraryImport(LibSystem)]
        internal static partial int task_threads(uint task, out nint threads, out uint count);

        [LibraryImport(LibSystem)]
        internal static partial int thread_info(uint thread, int flavor, void* info, ref uint count);

        [LibraryImport(LibSystem)]
        internal static partial int mach_port_deallocate(uint task, uint name);

        [LibraryImport(LibSystem)]
        internal static partial int vm_deallocate(uint task, nuint address, nuint size);

        // mach_task_self() is a macro over the mach_task_self_ variable.
        internal static uint TaskSelf()
        {
            var library = NativeLibrary.Load(LibSystem);
            return *(uint*)NativeLibrary.GetExport(library, "mach_task_self_");
        }
    }

    private static partial class LinuxNative
    {
        [LibraryImport("libc", SetLastError = false)]
        internal static partial int gettid();
    }

    private static unsafe partial class WindowsNative
    {
        [LibraryImport("kernel32.dll")]
        internal static partial uint GetCurrentThreadId();

        [LibraryImport("kernel32.dll")]
        internal static partial nint OpenThread(uint access, [MarshalAs(UnmanagedType.Bool)] bool inherit, uint threadId);

        [LibraryImport("kernel32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool CloseHandle(nint handle);

        [LibraryImport("kernel32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool GetThreadTimes(nint thread, out long creation, out long exit, out long kernel, out long user);

        [LibraryImport("kernel32.dll")]
        private static partial int GetThreadDescription(nint thread, out nint description);

        [LibraryImport("kernel32.dll")]
        private static partial nint LocalFree(nint memory);

        internal static string Describe(nint handle)
        {
            try
            {
                if (GetThreadDescription(handle, out var description) < 0 || description == 0)
                {
                    return string.Empty;
                }

                var text = Marshal.PtrToStringUni(description) ?? string.Empty;
                LocalFree(description);
                return text;
            }
            catch (EntryPointNotFoundException)
            {
                return string.Empty;
            }
        }
    }
}
