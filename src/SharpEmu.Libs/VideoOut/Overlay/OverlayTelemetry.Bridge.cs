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
// GPU host writes the capture demand and the threads it wants timed back. Single writer per field;
// torn reads of a counter cost at most one frame's accuracy, which the overlay tolerates.
//
// Timestamps cross the boundary in nanoseconds of the monotonic clock both processes' Stopwatch
// reads (CLOCK_UPTIME_RAW on macOS, also under Rosetta), so a guest call lines up with the GPU
// host's frames.
public static partial class OverlayTelemetry
{
    private static class Bridge
    {
        private const uint Magic = 0x424F4553; // "SEOB"
        private const int LayoutVersion = 2;
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
        private const int MainThreadOffset = 56;     // written by the GPU host: the guest main thread
        private const int WorkerThreadOffset = 64;   // written by the GPU host: the busiest worker
        private const int CallWriteOffset = 72;      // calls ever written to the ring
        private const int HeaderBytes = 128;
        private const int ThreadsOffset = HeaderBytes;
        // Per import: main-thread ticks (100 ns), main calls, worker ticks, worker calls.
        private const int ImportsOffset = ThreadsOffset + MaxThreads * sizeof(ulong);
        private const int NamesOffset = ImportsOffset + MaxImports * 4 * sizeof(long);
        // Long calls (>= 100 µs) of the two timed threads: start ns, end ns, import, lane.
        private const int CallEntries = 4096;
        private const int CallBytes = 24;
        private const int CallsOffset = NamesOffset + MaxImports * NameBytes;
        private const int TotalBytes = CallsOffset + CallEntries * CallBytes;

        private static long ToNanoseconds(long ticks) => (long)(ticks * (1_000_000_000.0 / Stopwatch.Frequency));
        private static long FromNanoseconds(long nanoseconds) => (long)(nanoseconds * (Stopwatch.Frequency / 1_000_000_000.0));

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
                    var state = new GuestState();
                    long heartbeat = 0;
                    while (true)
                    {
                        var demand = (OverlayCaptureDemand)Volatile.Read(ref *(int*)(basePointer + DemandOffset));
                        HleTimingActive = (demand & (OverlayCaptureDemand.HleCalls | OverlayCaptureDemand.Timeline)) != 0;
                        TimelineActive = (demand & OverlayCaptureDemand.Timeline) != 0;
                        PublishGuest(basePointer, state);
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

        // The publisher thread's running totals. Each thread's growth since the last publish is
        // added to the bucket of its current role, so the buckets only grow when the GPU host
        // picks a different main thread.
        private sealed class GuestState
        {
            public readonly bool[] Named = new bool[MaxImports];
            public readonly long[] MainTicks = new long[MaxImports];
            public readonly long[] MainCounts = new long[MaxImports];
            public readonly long[] WorkerTicks = new long[MaxImports];
            public readonly long[] WorkerCounts = new long[MaxImports];
            public readonly Dictionary<HleThreadSlots, (long[] Ticks, int[] Counts)> Previous = [];
            public readonly List<(long Start, long End, int Import)> Calls = new(256);
            public long CallsWritten;
            public int ImportCount;
        }

        private static unsafe void PublishGuest(byte* basePointer, GuestState state)
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

            var mainId = Volatile.Read(ref *(ulong*)(basePointer + MainThreadOffset));
            var workerId = Volatile.Read(ref *(ulong*)(basePointer + WorkerThreadOffset));
            foreach (var slot in slots)
            {
                var arrays = slot.Arrays;
                var length = Math.Min(arrays.Ticks.Length, MaxImports);
                if (!state.Previous.TryGetValue(slot, out var previous) || previous.Ticks.Length < length)
                {
                    var grown = (new long[length], new int[length]);
                    if (previous.Ticks is not null)
                    {
                        Array.Copy(previous.Ticks, grown.Item1, previous.Ticks.Length);
                        Array.Copy(previous.Counts, grown.Item2, previous.Counts.Length);
                    }

                    previous = grown;
                    state.Previous[slot] = previous;
                }

                var isMain = slot.ThreadId == mainId && mainId != 0;
                var ticksBucket = isMain ? state.MainTicks : state.WorkerTicks;
                var countsBucket = isMain ? state.MainCounts : state.WorkerCounts;
                for (var index = 0; index < length; index++)
                {
                    var ticks = arrays.Ticks[index];
                    var delta = ticks - previous.Ticks[index];
                    if (delta <= 0) continue;
                    var count = arrays.Counts[index];
                    ticksBucket[index] += delta;
                    countsBucket[index] += count - previous.Counts[index];
                    previous.Ticks[index] = ticks;
                    previous.Counts[index] = count;
                    state.ImportCount = Math.Max(state.ImportCount, index + 1);
                }

                // Long calls: the two threads the GPU host times go into the ring; the rest are dropped.
                state.Calls.Clear();
                slot.TakeLongCalls(long.MinValue, long.MaxValue, state.Calls);
                var lane = isMain ? 0 : slot.ThreadId == workerId && workerId != 0 ? 1 : -1;
                if (lane < 0) continue;
                foreach (var (callStart, callEnd, import) in state.Calls)
                {
                    var entry = basePointer + CallsOffset + (state.CallsWritten % CallEntries) * CallBytes;
                    *(long*)entry = ToNanoseconds(callStart);
                    *(long*)(entry + 8) = ToNanoseconds(callEnd);
                    *(int*)(entry + 16) = import;
                    *(int*)(entry + 20) = lane;
                    state.CallsWritten++;
                }
            }

            Volatile.Write(ref *(long*)(basePointer + CallWriteOffset), state.CallsWritten);
            var imports = (long*)(basePointer + ImportsOffset);
            var ticksTo100ns = 10_000_000.0 / Stopwatch.Frequency;
            for (var index = 0; index < state.ImportCount; index++)
            {
                if (state.MainTicks[index] == 0 && state.WorkerTicks[index] == 0) continue;
                if (!state.Named[index])
                {
                    state.Named[index] = true;
                    WriteName(basePointer + NamesOffset + (long)index * NameBytes, HleName(index));
                }

                // Ticks are converted to 100 ns units so the two processes' Stopwatch frequencies never matter.
                Volatile.Write(ref imports[index * 4], (long)(state.MainTicks[index] * ticksTo100ns));
                Volatile.Write(ref imports[index * 4 + 1], state.MainCounts[index]);
                Volatile.Write(ref imports[index * 4 + 2], (long)(state.WorkerTicks[index] * ticksTo100ns));
                Volatile.Write(ref imports[index * 4 + 3], state.WorkerCounts[index]);
            }

            Volatile.Write(ref *(int*)(basePointer + ImportCountOffset), state.ImportCount);
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
        private static readonly long[] PreviousImport = new long[MaxImports * 4];
        private static long _callsRead;
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
                Array.Clear(PreviousImport);
                _callsRead = Volatile.Read(ref *(long*)(basePointer + CallWriteOffset));
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

        /// <summary>Tells the guest which of its threads are the main one and the busiest worker, for lane attribution and timing.</summary>
        internal static unsafe void WriteFocus(ulong mainThreadId, ulong workerThreadId)
        {
            if (_base != null)
            {
                Volatile.Write(ref *(ulong*)(_base + MainThreadOffset), mainThreadId);
                Volatile.Write(ref *(ulong*)(_base + WorkerThreadOffset), workerThreadId);
            }
        }

        /// <summary>Appends the long calls the guest published since the previous call, in this process's Stopwatch ticks.</summary>
        internal static unsafe void TakeCalls(List<(long Start, long End, int Import, OverlayLane Lane)> into)
        {
            if (_base == null) return;
            var written = Volatile.Read(ref *(long*)(_base + CallWriteOffset));
            if (written - _callsRead > CallEntries)
            {
                // The reader fell a whole ring behind; the oldest entries are already overwritten.
                _callsRead = written - CallEntries;
            }

            for (; _callsRead < written; _callsRead++)
            {
                var entry = _base + CallsOffset + (_callsRead % CallEntries) * CallBytes;
                var lane = *(int*)(entry + 20) == 0 ? OverlayLane.GuestMain : OverlayLane.GuestWorkers;
                into.Add((FromNanoseconds(*(long*)entry), FromNanoseconds(*(long*)(entry + 8)), *(int*)(entry + 16), lane));
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

        /// <summary>
        /// Calls <paramref name="visit"/> with (import, main ms, main calls, worker ms, worker calls)
        /// for each import that moved since the previous call.
        /// </summary>
        internal static unsafe void TakeImportDeltas(Action<int, float, int, float, int> visit)
        {
            if (_base == null) return;
            var count = Math.Min(Volatile.Read(ref *(int*)(_base + ImportCountOffset)), MaxImports);
            var imports = (long*)(_base + ImportsOffset);
            Span<long> delta = stackalloc long[4];
            for (var index = 0; index < count; index++)
            {
                var moved = false;
                for (var field = 0; field < 4; field++)
                {
                    var at = index * 4 + field;
                    var value = Volatile.Read(ref imports[at]);
                    delta[field] = Math.Max(0, value - PreviousImport[at]);
                    moved |= value != PreviousImport[at];
                    PreviousImport[at] = value;
                }

                if (moved && (delta[0] > 0 || delta[2] > 0))
                {
                    visit(index, delta[0] / 10_000f, (int)delta[1], delta[2] / 10_000f, (int)delta[3]);
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
