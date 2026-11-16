// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>
/// The last few thousand guest writes the command processor made or deferred to the GPU:
/// labels, write-data packets, end-of-pipe values. A wait that never completes is either
/// waiting for a write that has not happened or for one that happened and was overwritten
/// before the waiter looked; this is what tells the two apart. Diagnostics only: nothing
/// records unless <see cref="Enabled"/>, so the callers pay one branch per write otherwise.
/// SHARPEMU_CP_HISTORY=1 turns it on, as do SHARPEMU_CP_WATCH and SHARPEMU_GPU_CLOBBER_DIAG
/// (the clobber report prints the history); the stall and shader-code dumps read it when on.
/// </summary>
public static class CommandProcessorWriteHistory
{
    public readonly record struct Entry(
        long Ticks, int QueueId, ulong Address, int Size, ulong Value, string Kind, bool Deferred, ulong Packet = 0);

    // Off by default: a record costs a timestamp, an interlocked add and, at some call sites, a
    // formatted string or a guest read made only for the record. Tests set it directly.
    public static bool Enabled { get; set; } = EnabledByEnvironment(Environment.GetEnvironmentVariable);

    internal static bool EnabledByEnvironment(Func<string, string?> getEnvironmentVariable) =>
        getEnvironmentVariable("SHARPEMU_CP_HISTORY") is "1" or "reuse" ||
        !string.IsNullOrWhiteSpace(getEnvironmentVariable("SHARPEMU_CP_WATCH")) ||
        getEnvironmentVariable("SHARPEMU_GPU_CLOBBER_DIAG") == "1";

    // SHARPEMU_CP_HISTORY=reuse (diagnostic): a 64 Ki-entry history that also holds each submission's admission and the
    // slices that walk it, every command-processor write tagged with the packet that made it, and is printed whole for
    // the last ~800 events before a command-stream fatal: which landed write let the guest recycle the buffer being walked.
    public static readonly bool Reuse = Environment.GetEnvironmentVariable("SHARPEMU_CP_HISTORY") == "reuse";

    // The packet the interpreter on this thread is executing (reuse mode); a worker thread's records carry none.
    [ThreadStatic] public static ulong CurrentPacket;

    private static readonly int Capacity = Reuse ? 1 << 18 : 4096;
    private static readonly Entry[] _entries = new Entry[Capacity];
    private static long _next;

    // SHARPEMU_CP_WATCH=<hex>[,<hex>...]: every recorded write overlapping one of these
    // addresses is also printed as it happens, so a label's whole history survives the ring.
    private static readonly ulong[] Watched = ParseWatched(Environment.GetEnvironmentVariable("SHARPEMU_CP_WATCH"));

    private static ulong[] ParseWatched(string? text) => string.IsNullOrWhiteSpace(text)
        ? []
        : text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => ulong.TryParse(
                token.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ? token[2..] : token,
                System.Globalization.NumberStyles.HexNumber, null, out var address) ? address : 0)
            .Where(address => address != 0)
            .ToArray();

    public static void Record(int queueId, ulong address, int size, ulong value, string kind, bool deferred)
    {
        if (!Enabled)
        {
            return;
        }

        foreach (var watched in Watched)
        {
            if (watched >= address && watched < address + (ulong)Math.Max(size, 1))
            {
                Console.Error.WriteLine(
                    $"[GPU][CPWATCH] address=0x{address:X16} size={size} value=0x{value:X} kind={kind} " +
                    $"queue={queueId} deferred={deferred} thread={Environment.CurrentManagedThreadId}");
            }
        }

        var slot = Interlocked.Increment(ref _next) - 1;
        _entries[(int)(slot % Capacity)] = new Entry(
            System.Diagnostics.Stopwatch.GetTimestamp(), queueId, address, size, value, kind, deferred,
            Reuse && queueId >= 0 ? CurrentPacket : 0);
    }

    /// <summary>Reuse mode: a submission admitted or a slice of it started or ended (packet = the walk position).</summary>
    public static void RecordStream(string kind, int queueId, ulong buffer, uint dwords, ulong submissionId, ulong packet)
    {
        var slot = Interlocked.Increment(ref _next) - 1;
        _entries[(int)(slot % Capacity)] = new Entry(
            System.Diagnostics.Stopwatch.GetTimestamp(), queueId, buffer, (int)Math.Min(dwords * 4UL, int.MaxValue),
            submissionId, kind, false, packet);
    }

    /// <summary>Reuse mode: the history of one buffer and the last events before a fatal, oldest first.</summary>
    public static void DumpForReuse(ulong buffer, uint dwords, int recent)
    {
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        var end = Interlocked.Read(ref _next);
        var start = Math.Max(0, end - Capacity);
        var bytes = (ulong)dwords * 4;
        Console.Error.WriteLine($"[GPU][REUSE] buffer=0x{buffer:X16} dwords={dwords} events={end - start}");
        var shown = 0;
        for (var index = start; index < end && shown < 400; index++)
        {
            var entry = _entries[(int)(index % Capacity)];
            if (entry.Size != 0 && entry.Address < buffer + bytes && buffer < entry.Address + (ulong)entry.Size)
            {
                Print("buf", entry, now);
                shown++;
            }
        }

        for (var index = Math.Max(start, end - recent); index < end; index++)
        {
            Print("all", _entries[(int)(index % Capacity)], now);
        }

        // SHARPEMU_CP_HISTORY_DUMP=<file>: the whole history, for reading offline.
        var path = Environment.GetEnvironmentVariable("SHARPEMU_CP_HISTORY_DUMP");
        if (!string.IsNullOrWhiteSpace(path))
        {
            try
            {
                using var writer = new StreamWriter(path);
                for (var index = start; index < end; index++)
                {
                    writer.WriteLine(Format("all", _entries[(int)(index % Capacity)], now));
                }

                Console.Error.WriteLine($"[GPU][REUSE] history written: {path} events={end - start}");
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[GPU][REUSE] history dump failed: {exception.Message}");
            }
        }

        static void Print(string tag, in Entry entry, long now) => Console.Error.WriteLine(Format(tag, entry, now));

        static string Format(string tag, in Entry entry, long now) =>
            $"[GPU][REUSE] {tag} t_ms={(entry.Ticks - now) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F3} " +
            $"q={entry.QueueId} kind={entry.Kind} addr=0x{entry.Address:X} size={entry.Size} value=0x{entry.Value:X} " +
            $"packet=0x{entry.Packet:X}";
    }

    /// <summary>The recorded writes overlapping a range, oldest first.</summary>
    public static List<Entry> Find(ulong address, ulong size)
    {
        var found = new List<Entry>();
        var end = Interlocked.Read(ref _next);
        var start = Math.Max(0, end - Capacity);
        for (var index = start; index < end; index++)
        {
            var entry = _entries[(int)(index % Capacity)];
            if (entry.Size != 0 && entry.Address < address + size && address < entry.Address + (ulong)entry.Size)
            {
                found.Add(entry);
            }
        }

        return found;
    }
}
