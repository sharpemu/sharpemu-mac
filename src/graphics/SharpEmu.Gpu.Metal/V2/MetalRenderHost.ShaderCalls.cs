// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Logging;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// Linked shader calls (Gen5ShaderCallLinker, lowered by Gen5MslTranslator.ShaderCalls.cs). SHARPEMU_METAL_SHADER_CALLS=0
// turns all of it off, as before the port: the program cache compiles a caller alone and skips its calls.
//  - Call records of programs linked with run-time arguments (MetalCallRecordPreparation): the words table a dispatch's
//    call entries search, the heap elements of the descriptors its callees read through their records (from
//    BindlessHeap.CallRecordFirstIndex on; they need the bindless heap), and the memory they reach, ordered and flushed with
//    the stage's device-address ranges.
//  - Call marks (CallMarks): two shared bitsets; dispatches mark the active one, and once the other's submissions have
//    completed it is read, its bits stamped with the snapshot number (CallMarkedAt) and cleared, and the two swap. The
//    program cache compares the stamps with the snapshot in which it named a callee (ShaderCalleeTiers).
//  - Call faults: a call of a program linked per record that no record names claims an 8-word report
//    (Gen5MslArgumentLayout.CallFaultWords), read once per flush and logged as [UNLINKED_CALL] with a counter; under
//    SHARPEMU_STRICT_COMPUTE (on unless 0) it is fatal, as on Vulkan (BdaFaultProcessor).
internal sealed partial class MetalRenderHost
{
    internal static readonly bool ShaderCallsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_SHADER_CALLS") != "0";

    // Records live in the bindless heap, so they need it; marks exist only for programs linked with run-time records.
    public bool RuntimeCallRecordsSupported => ShaderCallsEnabled && BindlessHeapEnabled && RuntimeCallRecordsEnabled;

    // Tests link a copy per record on one host, as SHARPEMU_RUNTIME_CALL_RECORDS=0 does for the process.
    internal bool RuntimeCallRecordsEnabled { get; set; } = true;

    public bool CallMarksSupported => RuntimeCallRecordsSupported;

    public long CallMarkSnapshots => Volatile.Read(ref _callMarkSnapshots);

    public long CallMarkedAt(uint bit) => Volatile.Read(ref _callMarkedAt[bit & (CallMarks.Bits - 1)]);

    // pipeline_cache/<title>/metal-v2/linked, beside the translations and analysis caches (MetalTranslationCache).
    // Tests set it before the first dispatch (the program cache opens the store at its first runtime-record caller).
    public string? LinkedProgramCacheDirectory
    {
        get => ShaderCallsEnabled ? _linkedProgramCacheDirectory ??= SharpEmuPaths.UserPath("pipeline_cache",
            MetalShaderPrewarm.SanitizeTitleId(VideoOutBackendServices.GetApplicationTitleId()), "metal-v2", "linked") : null;
        internal set => _linkedProgramCacheDirectory = value;
    }

    private string? _linkedProgramCacheDirectory;

    // Call records.
    private readonly record struct CallRecordKey(int Slot, ulong First, ulong Second, ulong Third, ulong Fourth);

    private readonly Dictionary<CallRecordKey, HeapEntry> _callRecordEntries = new();
    private readonly uint[] _callRecordNext = new uint[BindlessHeap.ArrayCount];
    // The image arrays number their record elements in one sequence, so the class word of an element index
    // (WriteHeapElementClass) holds the numeric class of the one record image there.
    private uint _callRecordNextImage;
    private readonly Dictionary<uint, HeapEntry> _callRecordImages = new();
    private readonly HeapRegionEntries _callRecordTextures = new();
    private MetalCallRecordPreparation? _callRecordPreparation;
    private int _callRecordReports;

    internal long CallRecordElementsExhausted, UnlinkedCalls;

    // Calls that returned at once because the caller's program did not hold their callee yet: each bit a read mark snapshot
    // holds is one (caller, callee address) whose calls were skipped since the snapshot before. The program cache links the
    // callee in once its calls have settled (ShaderCalleeTiers), and the program that holds it serves from the next dispatch
    // after its pipeline is ready; until then the caller runs without the callee's effects, as on Vulkan. Counted in total
    // (MarkedCalls) and on the perf line (call_marks=); the program cache logs each caller once ([CALLEE_TIERS]).
    internal long MarkedCalls => Volatile.Read(ref _markedCalls);
    private long _markedCalls;
    private long _markedCallsWindow;

    internal long TakeMarkedCalls() => Interlocked.Exchange(ref _markedCallsWindow, 0);

    internal MetalCallRecordPreparation CallRecordPreparation => _callRecordPreparation ??= new MetalCallRecordPreparation(
        ReadCallRecordBytes, (address, size) => _memory.CanRead(address, size), CallRecordElement);

    // The stage's call record words and ranges, for a program whose callees read descriptors through their records.
    private void PrepareCallRecords(MetalStageBindings prepared, ShaderProgramInfo program, ShaderStageResources stage)
    {
        prepared.CallRecordWords = [];
        prepared.CallRecordRanges = [];
        if (program.CallRecordCallees.Count == 0)
        {
            return;
        }

        if (!RuntimeCallRecordsSupported)
        {
            throw Fatal($"A program reads call records with linked shader calls or the bindless heap off: hash=0x{program.Hash:X16}.");
        }

        PrepareBindlessHeap();
        (prepared.CallRecordWords, prepared.CallRecordRanges) = CallRecordPreparation.Prepare(program, stage);
        // Each record texture as an acquire now would give it, noted as used by this submission.
        MakeHeapRegionCurrent(_callRecordTextures, SubmissionSerial);
    }

    private bool ReadCallRecordBytes(ulong address, Span<byte> destination)
    {
        if (TryReadResidentGuestBytes(address, destination, clean: true))
        {
            return true;
        }

        // Bytes a recorded GPU write may still own: read word by word, which waits for the write.
        if ((destination.Length & 3) != 0)
        {
            return false;
        }

        var words = MemoryMarshal.Cast<byte, uint>(destination);
        for (var word = 0; word < words.Length; word++)
        {
            if (!TryReadGuestWord(address + (ulong)word * sizeof(uint), out words[word]))
            {
                return false;
            }
        }

        return true;
    }

    // The heap element that holds the descriptor of one record field: one per distinct descriptor and array, resolved when
    // it is first seen. An integer image is resolved to its integer view and its element's class word names its class, so
    // a load reads it through that class's declaration (Gen5MslTranslator.TryEmitCallRecordImage).
    private uint CallRecordElement(CallRecordField field, ReadOnlySpan<uint> words)
    {
        static ulong Pair(uint low, uint high) => low | ((ulong)high << 32);
        var key = new CallRecordKey(field.Slot, Pair(words[0], words[1]), Pair(words[2], words[3]), Pair(words[4], words[5]),
            Pair(words[6], words[7]));
        if (_callRecordEntries.TryGetValue(key, out var entry))
        {
            return entry.Index;
        }

        var image = field.Slot < BindlessHeap.ImageKinds.Length;
        var next = image ? _callRecordNextImage : _callRecordNext[field.Slot];
        if (BindlessHeap.CallRecordFirstIndex + next >= BindlessHeap.CallRecordNullIndex)
        {
            CallRecordElementsExhausted++;
            if (_callRecordReports++ < 8)
            {
                Console.Error.WriteLine($"[METAL-V2][WARN] The call record elements of a bindless heap array are used up: slot={field.Slot}; " +
                    "further records read as null.");
            }

            return BindlessHeap.CallRecordNullIndex;
        }

        if (image)
            _callRecordNextImage = next + 1;
        else
            _callRecordNext[field.Slot] = next + 1;
        entry = new HeapEntry
        {
            Slot = field.Slot,
            Index = BindlessHeap.CallRecordFirstIndex + next,
            Words = words[..(field.Sampler ? 4 : 8)].ToArray(),
        };
        _callRecordEntries.Add(key, entry);
        _callRecordTextures.Order.Add(entry);
        if (image)
            _callRecordImages.Add(entry.Index, entry);
        ResolveHeapEntry(entry);
        return entry.Index;
    }

    // Call marks.
    private readonly long[] _callMarkedAt = new long[CallMarks.Bits];
    private long _callMarkSnapshots;
    private readonly nint[] _callMarkBuffers = new nint[2];
    private readonly nint[] _callMarkContents = new nint[2];
    private readonly ulong[] _callMarkAddresses = new ulong[2];
    // The last submission that bound each bitset, and whether it holds marks not read yet.
    private readonly ulong[] _callMarkSerial = new ulong[2];
    private readonly bool[] _callMarkPending = new bool[2];
    private int _callMarkActive;

    private unsafe Span<uint> CallMarkWords(int index) => new((void*)_callMarkContents[index], (int)CallMarks.Words);

    private ulong CallMarksAddress()
    {
        if (_callMarkBuffers[0] == 0)
        {
            for (var index = 0; index < 2; index++)
            {
                _callMarkBuffers[index] = _guestMemory.CreateReportBuffer(CallMarks.Bytes, $"SharpEmu call marks {index}",
                    out _callMarkContents[index], out _callMarkAddresses[index]);
                CallMarkWords(index).Clear();
            }
        }

        var active = _callMarkActive;
        _callMarkSerial[active] = SubmissionSerial;
        _callMarkPending[active] = true;
        NoteUsedResource(_callMarkBuffers[active], Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
        return _callMarkAddresses[active];
    }

    // Once per flush: the inactive bitset, once every submission that bound it has completed, is read and cleared; then a
    // bitset bound since the last swap gives way to the cleared one.
    private void PollCallMarks()
    {
        if (_callMarkBuffers[0] == 0)
        {
            return;
        }

        var other = 1 - _callMarkActive;
        if (_callMarkPending[other] && _callMarkSerial[other] <= CompletedSerial)
        {
            var words = CallMarkWords(other);
            var snapshot = Volatile.Read(ref _callMarkSnapshots) + 1;
            for (var word = 0; word < words.Length; word++)
            {
                var held = Volatile.Read(ref words[word]);
                if (held != 0)
                {
                    var count = BitOperations.PopCount(held);
                    Interlocked.Add(ref _markedCalls, count);
                    Interlocked.Add(ref _markedCallsWindow, count);
                }

                for (var bits = held; bits != 0; bits &= bits - 1)
                {
                    Volatile.Write(ref _callMarkedAt[word * 32 + BitOperations.TrailingZeroCount(bits)], snapshot);
                }
            }

            words.Clear();
            _callMarkPending[other] = false;
            Volatile.Write(ref _callMarkSnapshots, snapshot);
        }

        if (!_callMarkPending[other] && _callMarkPending[_callMarkActive])
        {
            _callMarkActive = other;
        }
    }

    // Call faults.
    private nint _callFaultBuffer;
    private nint _callFaultContents;
    private ulong _callFaultAddress;

    internal unsafe Span<uint> CallFaultWords => new((void*)_callFaultContents, Gen5MslArgumentLayout.CallFaultWords);
    // An unlinked call stops the emulator, as any compute failure does, unless SHARPEMU_STRICT_COMPUTE=0 (tests change it).
    internal bool StrictCompute = Environment.GetEnvironmentVariable("SHARPEMU_STRICT_COMPUTE") != "0";

    internal ulong CallFaultsAddress()
    {
        if (_callFaultBuffer == 0)
        {
            _callFaultBuffer = _guestMemory.CreateReportBuffer(Gen5MslArgumentLayout.CallFaultWords * sizeof(uint), "SharpEmu call faults",
                out _callFaultContents, out _callFaultAddress);
            CallFaultWords.Clear();
        }

        NoteUsedResource(_callFaultBuffer, Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
        return _callFaultAddress;
    }

    // Once per flush: a claimed report names the caller, the call and its target; the waves that made the call ended at it.
    internal void ReportCallFaults(bool shutdown = false)
    {
        PollCallMarks();
        if (_callFaultBuffer == 0)
        {
            return;
        }

        var words = CallFaultWords;
        if (Volatile.Read(ref words[0]) != 2)
        {
            return;
        }

        var hash = (ulong)words[2] << 32 | words[1];
        var target = (ulong)words[5] << 32 | words[4];
        var pc = words[3];
        var stage = words[6];
        words[1..].Clear();
        Volatile.Write(ref words[0], 0u);
        var message = $"The shader called an unlinked function: hash=0x{hash:X16} pc=0x{pc:X} target=0x{target:X16} stage={stage}.";
        // At shutdown the host is draining and the fault is only reported; the emulator is stopping anyway.
        if (StrictCompute && !shutdown)
        {
            throw Fatal(message);
        }

        var unlinked = ++UnlinkedCalls;
        if (unlinked <= 16 || (unlinked & (unlinked - 1)) == 0)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN][UNLINKED_CALL] {message} count={unlinked}. The waves that made the call were " +
                "ended; images can be incorrect. SHARPEMU_STRICT_COMPUTE=0 is set.");
        }
    }

    private void DisposeShaderCalls()
    {
        _callRecordEntries.Clear();
        _callRecordImages.Clear();
        _callRecordTextures.Order.Clear();
        foreach (ref var buffer in _callMarkBuffers.AsSpan())
        {
            if (buffer != 0)
            {
                MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
                buffer = 0;
            }
        }

        if (_callFaultBuffer != 0)
        {
            MetalNative.SendVoid(_callFaultBuffer, MetalNative.Selector("release"));
            _callFaultBuffer = 0;
        }
    }
}
