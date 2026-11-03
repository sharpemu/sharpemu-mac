// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Text;

namespace SharpEmu.Libs.VideoOut.Overlay;

// The GPU/guest split (macOS: the translated guest process and the native GPU host). The overlay
// draws in the GPU host, which also owns the render thread, the GPU intervals and its own thread
// CPU; the guest's HLE gateway, its guest-thread ids and its GC run in the other process. The
// guest publishes those into a small file-backed shared mapping that the GPU host reads, and the
// GPU host writes the capture demand back. Single writer per field; torn reads of a counter cost
// at most one frame's accuracy, which the overlay tolerates.
public static partial class OverlayTelemetry
{
    private static class Bridge
    {
        private const uint Magic = 0x424F4553; // "SEOB"
        private const int LayoutVersion = 1;
        internal const int MaxImports = 8192;
        private const int MaxThreads = 512;
        private const int NameBytes = 96;

        // Header fields (byte offsets).
        private const int MagicOffset = 0;
        private const int VersionOffset = 4;
        private const int DemandOffset = 8;          // written by the GPU host
        private const int ImportCountOffset = 12;
        private const int GcPauseTicksOffset = 16;   // TimeSpan ticks
        private const int Gen0Offset = 24;
        private const int Gen1Offset = 28;
        private const int Gen2Offset = 32;
        private const int ThreadCountOffset = 36;
        private const int AllocatedBytesOffset = 40;
        private const int HeartbeatOffset = 48;      // Stopwatch-free sequence number
        private const int HeaderBytes = 128;
        private const int ThreadsOffset = HeaderBytes;
        private const int ImportsOffset = ThreadsOffset + MaxThreads * sizeof(ulong);
        private const int NamesOffset = ImportsOffset + MaxImports * 2 * sizeof(long);
        private const int TotalBytes = NamesOffset + MaxImports * NameBytes;

        private static MemoryMappedFile? _file;
        private static MemoryMappedViewAccessor? _view;
        private static unsafe byte* _base;

        internal static string PathFor(int guestProcessId) =>
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"sharpemu-overlay-{guestProcessId}.bin");

        // ------------------------------------------------------------------ guest side

        private static int _guestStarted;

        /// <summary>True in the guest process of the split: it runs the gateway but not the overlay.</summary>
        internal static bool IsSplitGuest =>
            OperatingSystem.IsMacOS() &&
            string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_GPU_PROCESS"), "1", StringComparison.Ordinal) &&
            PerfOverlay.GuestProcessId == 0 && !PerfOverlay.Enabled;

        internal static void StartGuestPublisher()
        {
            if (Interlocked.Exchange(ref _guestStarted, 1) != 0)
            {
                return;
            }

            var thread = new Thread(GuestLoop)
            {
                IsBackground = true,
                Name = "SharpEmu overlay bridge",
                Priority = ThreadPriority.BelowNormal,
            };
            thread.Start();
        }

        private static unsafe void GuestLoop()
        {
            try
            {
                var path = PathFor(Environment.ProcessId);
                using var stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
                stream.SetLength(TotalBytes);
                using var file = MemoryMappedFile.CreateFromFile(stream, null, TotalBytes, MemoryMappedFileAccess.ReadWrite,
                    HandleInheritability.None, leaveOpen: true);
                using var view = file.CreateViewAccessor(0, TotalBytes, MemoryMappedFileAccess.ReadWrite);
                byte* basePointer = null;
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);
                try
                {
                    *(int*)(basePointer + VersionOffset) = LayoutVersion;
                    Volatile.Write(ref *(uint*)(basePointer + MagicOffset), Magic);
                    AppDomain.CurrentDomain.ProcessExit += (_, _) => TryDelete(path);
                    var named = new bool[MaxImports];
                    var totalsTicks = new long[MaxImports];
                    var totalsCounts = new long[MaxImports];
                    long heartbeat = 0;
                    while (true)
                    {
                        var demand = (OverlayCaptureDemand)Volatile.Read(ref *(int*)(basePointer + DemandOffset));
                        HleTimingActive = (demand & (OverlayCaptureDemand.HleCalls | OverlayCaptureDemand.Timeline)) != 0;
                        PublishGuest(basePointer, named, totalsTicks, totalsCounts);
                        Volatile.Write(ref *(long*)(basePointer + HeartbeatOffset), ++heartbeat);
                        Thread.Sleep(demand == OverlayCaptureDemand.None ? 250 : 16);
                    }
                }
                finally
                {
                    view.SafeMemoryMappedViewHandle.ReleasePointer();
                }
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[OVERLAY][BRIDGE] guest publisher stopped: {exception.GetType().Name}: {exception.Message}");
            }
        }

        private static unsafe void PublishGuest(byte* basePointer, bool[] named, long[] totalsTicks, long[] totalsCounts)
        {
            var gcPause = GC.GetTotalPauseDuration();
            Volatile.Write(ref *(long*)(basePointer + GcPauseTicksOffset), gcPause.Ticks);
            *(int*)(basePointer + Gen0Offset) = GC.CollectionCount(0);
            *(int*)(basePointer + Gen1Offset) = GC.CollectionCount(1);
            *(int*)(basePointer + Gen2Offset) = GC.CollectionCount(2);
            Volatile.Write(ref *(long*)(basePointer + AllocatedBytesOffset), GC.GetTotalAllocatedBytes(false));

            var threads = 0;
            var threadIds = (ulong*)(basePointer + ThreadsOffset);
            foreach (var id in GuestThreadIds.Keys)
            {
                if (threads == MaxThreads) break;
                threadIds[threads++] = id;
            }

            Volatile.Write(ref *(int*)(basePointer + ThreadCountOffset), threads);

            HleThreadSlots[] slots;
            lock (HleThreads)
            {
                slots = HleThreads.ToArray();
            }

            Array.Clear(totalsTicks);
            Array.Clear(totalsCounts);
            var importCount = 0;
            foreach (var slot in slots)
            {
                var arrays = slot.Arrays;
                var length = Math.Min(arrays.Ticks.Length, MaxImports);
                for (var index = 0; index < length; index++)
                {
                    var ticks = arrays.Ticks[index];
                    if (ticks == 0) continue;
                    totalsTicks[index] += ticks;
                    totalsCounts[index] += arrays.Counts[index];
                    importCount = Math.Max(importCount, index + 1);
                }
            }

            var imports = (long*)(basePointer + ImportsOffset);
            for (var index = 0; index < importCount; index++)
            {
                if (totalsTicks[index] == 0) continue;
                if (!named[index])
                {
                    named[index] = true;
                    WriteName(basePointer + NamesOffset + (long)index * NameBytes, HleName(index));
                }

                // Ticks are converted to 100 ns units so the two processes' Stopwatch frequencies never matter.
                Volatile.Write(ref imports[index * 2], totalsTicks[index] * 10_000_000 / Stopwatch.Frequency);
                Volatile.Write(ref imports[index * 2 + 1], totalsCounts[index]);
            }

            Volatile.Write(ref *(int*)(basePointer + ImportCountOffset), Math.Max(importCount, *(int*)(basePointer + ImportCountOffset)));
        }

        private static unsafe void WriteName(byte* destination, string name)
        {
            var span = new Span<byte>(destination, NameBytes);
            span.Clear();
            var written = Encoding.UTF8.GetBytes(name.AsSpan(0, Math.Min(name.Length, NameBytes / 2)), span[..(NameBytes - 1)]);
            span[written] = 0;
        }

        private static void TryDelete(string path)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

        // ------------------------------------------------------------------ GPU-host side

        private static int _openedFor;
        private static long _lastOpenAttempt;
        private static long[] _previousImportTicks = new long[MaxImports];
        private static long[] _previousImportCounts = new long[MaxImports];
        private static long _previousGcPauseTicks = -1;
        private static readonly int[] PreviousGenerations = new int[3];
        private static readonly string?[] Names = new string?[MaxImports];
        internal static readonly HashSet<ulong> GuestThreads = [];

        internal static unsafe bool IsOpen => _base != null;

        /// <summary>Opens the guest's mapping once it exists; call from the builder thread.</summary>
        internal static unsafe void TryOpen(int guestProcessId)
        {
            if (guestProcessId <= 0 || _openedFor == guestProcessId)
            {
                return;
            }

            var now = Stopwatch.GetTimestamp();
            if (now - _lastOpenAttempt < Stopwatch.Frequency)
            {
                return;
            }

            _lastOpenAttempt = now;
            var path = PathFor(guestProcessId);
            if (!File.Exists(path))
            {
                return;
            }

            try
            {
                var stream = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
                if (stream.Length < TotalBytes)
                {
                    stream.Dispose();
                    return;
                }

                var file = MemoryMappedFile.CreateFromFile(stream, null, TotalBytes, MemoryMappedFileAccess.ReadWrite,
                    HandleInheritability.None, leaveOpen: false);
                var view = file.CreateViewAccessor(0, TotalBytes, MemoryMappedFileAccess.ReadWrite);
                byte* basePointer = null;
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);
                if (Volatile.Read(ref *(uint*)(basePointer + MagicOffset)) != Magic ||
                    *(int*)(basePointer + VersionOffset) != LayoutVersion)
                {
                    view.SafeMemoryMappedViewHandle.ReleasePointer();
                    view.Dispose();
                    file.Dispose();
                    return;
                }

                _file = file;
                _view = view;
                _base = basePointer;
                _openedFor = guestProcessId;
                _previousGcPauseTicks = -1;
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
            }
        }

        internal static unsafe void WriteDemand(OverlayCaptureDemand demand)
        {
            if (_base != null)
            {
                Volatile.Write(ref *(int*)(_base + DemandOffset), (int)demand);
            }
        }

        internal static unsafe void ReadGuestThreads()
        {
            GuestThreads.Clear();
            if (_base == null) return;
            var count = Math.Min(Volatile.Read(ref *(int*)(_base + ThreadCountOffset)), MaxThreads);
            var ids = (ulong*)(_base + ThreadsOffset);
            for (var index = 0; index < count; index++) GuestThreads.Add(ids[index]);
        }

        /// <summary>The guest's GC pause and collections since the previous call.</summary>
        internal static unsafe (float PauseMs, int Collections, int HighestGeneration) TakeGuestGc()
        {
            if (_base == null) return (0, 0, -1);
            var pause = Volatile.Read(ref *(long*)(_base + GcPauseTicksOffset));
            var generations = stackalloc int[3]
            {
                *(int*)(_base + Gen0Offset), *(int*)(_base + Gen1Offset), *(int*)(_base + Gen2Offset),
            };
            var first = _previousGcPauseTicks < 0;
            var pauseMs = first ? 0f : (float)TimeSpan.FromTicks(Math.Max(0, pause - _previousGcPauseTicks)).TotalMilliseconds;
            _previousGcPauseTicks = pause;
            var highest = -1;
            var collections = 0;
            for (var generation = 0; generation < 3; generation++)
            {
                var delta = first ? 0 : generations[generation] - PreviousGenerations[generation];
                PreviousGenerations[generation] = generations[generation];
                if (delta > 0) highest = generation;
                if (generation == 0) collections = Math.Max(0, delta);
            }

            return (pauseMs, collections, highest);
        }

        /// <summary>Calls <paramref name="visit"/> with (import, milliseconds, calls) for each import that moved.</summary>
        internal static unsafe void TakeImportDeltas(Action<int, float, int> visit)
        {
            if (_base == null) return;
            var count = Math.Min(Volatile.Read(ref *(int*)(_base + ImportCountOffset)), MaxImports);
            var imports = (long*)(_base + ImportsOffset);
            for (var index = 0; index < count; index++)
            {
                var ticks = Volatile.Read(ref imports[index * 2]);
                var delta = ticks - _previousImportTicks[index];
                if (delta == 0) continue;
                var calls = Volatile.Read(ref imports[index * 2 + 1]);
                var callDelta = calls - _previousImportCounts[index];
                _previousImportTicks[index] = ticks;
                _previousImportCounts[index] = calls;
                if (delta > 0)
                {
                    visit(index, delta / 10_000f, (int)Math.Max(0, callDelta));
                }
            }
        }

        internal static unsafe string? NameOf(int import)
        {
            if (_base == null || (uint)import >= MaxImports) return null;
            if (Names[import] is { } cached) return cached;
            var pointer = _base + NamesOffset + (long)import * NameBytes;
            var length = 0;
            while (length < NameBytes && pointer[length] != 0) length++;
            if (length == 0) return null;
            var name = Encoding.UTF8.GetString(pointer, length);
            Names[import] = name;
            return name;
        }
    }
}
