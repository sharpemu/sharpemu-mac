// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

internal sealed class CommandStreamFatalException(string message) : Exception(message);

// Records every host call in order and serves guest memory from a fake.
internal sealed class RecordingCommandStreamHost : ICommandStreamHost, ICommandSubmissionCompletionHost
{
    internal Action<int, ulong>? BeginPrefixCapture;
    internal Func<bool, IGuestSubmissionPrefixCompletion?>? EndPrefixCapture;
    public void BeginSubmissionCompletionCapture(int queueId, ulong submissionId) => BeginPrefixCapture?.Invoke(queueId, submissionId);
    public IGuestSubmissionPrefixCompletion? EndSubmissionCompletionCapture(bool complete) => EndPrefixCapture?.Invoke(complete);
    public const ulong MemoryBase = 0x1000_0000;
    public const int MemorySize = 0x10_0000;

    public RecordingCommandStreamHost()
    {
        GuestMemory = new FakeCpuMemory(MemoryBase, MemorySize);
    }

    public FakeCpuMemory GuestMemory { get; }

    public ICpuMemory Memory => GuestMemory;

    public Func<int, ulong>? ReadOcclusionSamples { get; set; }

    public Func<int, bool>? PendingPublications { get; set; }

    public Func<int, EndOfPipeWrite, bool>? DeferredEndOfPipe { get; set; }
    public Func<ulong, ulong, bool>? PendingMemoryPublication { get; set; }
    public bool TryDeferEndOfPipe(int queueId, in EndOfPipeWrite write) => DeferredEndOfPipe?.Invoke(queueId, write) == true;
    public bool HasPendingMemoryPublication(ulong address, ulong size) => PendingMemoryPublication?.Invoke(address, size) == true;
    public DeferredLabelReader? DeferredLabelRead { get; set; }
    public delegate bool OrderedWriteDeferrer(int queueId, ulong address, ReadOnlySpan<byte> source);
    public OrderedWriteDeferrer? OrderedWrite { get; set; }
    public Func<int, bool>? CrossQueueHold { get; set; }
    public bool TryDeferOrderedWrite(int queueId, ulong address, ReadOnlySpan<byte> source) =>
        OrderedWrite?.Invoke(queueId, address, source) == true;
    public bool MustHoldForCrossQueueOrder(int queueId) => CrossQueueHold?.Invoke(queueId) == true;
    public delegate bool DeferredLabelReader(int queueId, ulong address, ulong size, out bool deferred, out ulong value);
    public bool TryReadDeferredLabel(int queueId, ulong address, ulong size, out bool deferred, out ulong value)
    {
        if (DeferredLabelRead is { } read) return read(queueId, address, size, out deferred, out value);
        deferred = false;
        value = 0;
        return !HasPendingMemoryPublication(address, size);
    }

    public bool HasPendingPublications(int queueId) => PendingPublications?.Invoke(queueId) == true;

    public bool TryReadOcclusionCounter(int queueId, out ulong value)
    {
        value = ReadOcclusionSamples?.Invoke(queueId) ?? 0;
        return ReadOcclusionSamples is not null;
    }

    public List<string> Calls { get; } = new();

    public List<(string Operation, ulong Address, ulong Size, byte[] Bytes)> TracedWrites { get; } = new();

    public void TraceGuestWrite(string operation, ulong address, ulong size)
    {
        var bytes = new byte[checked((int)size)];
        GuestMemory.TryRead(address, bytes);
        TracedWrites.Add((operation, address, size, bytes));
    }

    public List<EndOfPipeWrite> EndOfPipeWrites { get; } = new();
    public Action<EndOfPipeWrite>? EndOfPipeRecorded { get; set; }

    public List<ulong> GuestReads { get; } = new();

    public Action<ulong>? BeforeGuestRead { get; set; }
    public List<(ulong Address, int Size)> GuestReadRanges { get; } = new();

    public List<(string Subject, ulong Address, ulong Size)> WriterRequests { get; } = new();

    public void TraceWritersOf(string subject, ulong address, ulong size) =>
        WriterRequests.Add((subject, address, size));

    public List<DrawIndexedArguments> IndexedDraws { get; } = new();

    public List<DrawAutoArguments> AutoDraws { get; } = new();

    public Queue<Action> PendingCommands { get; } = new();

    public int PendingCommandRuns { get; private set; }

    public bool FlipSlot { get; set; } = true;

    // A host that leaves end-of-pipe labels to the GPU, and the tick they land by.
    public bool DefersEndOfPipeLabels { get; set; }

    public bool WritesGdsAtEndOfPipe { get; set; }

    public ulong PendingLabelTick { get; set; }

    public Func<EndOfPipeWriteKind, bool>? DefersKind { get; set; }

    public bool DefersEndOfPipeLabel(EndOfPipeWriteKind kind) => DefersKind?.Invoke(kind) ?? DefersEndOfPipeLabels;

    public bool FlipDone { get; set; } = true;

    public ulong NextFlipRequestId { get; set; } = 1;

    public byte[] Gds { get; } = new byte[ManagedCommandStreamHost.GdsBytes];

    // Qword values the GPU owns; the synchronized read lands them in memory first, as a download would.
    public Dictionary<ulong, ulong> PendingGpuValues { get; } = new();

    // Addresses read through the operand reader, which a host runs behind the work before the packet.
    public List<ulong> OperandReads { get; } = new();

    // The work before the packet, completed by an operand read (a GPU pass that writes a buffer).
    public Action<ulong>? BeforeOperandRead { get; set; }

    public bool TryReadGuestOperand(ulong address, Span<byte> destination)
    {
        OperandReads.Add(address);
        BeforeOperandRead?.Invoke(address);
        return TryReadGuest(address, destination);
    }

    public bool TryReadGuest(ulong address, Span<byte> destination)
    {
        GuestReads.Add(address);
        BeforeGuestRead?.Invoke(address);
        GuestReadRanges.Add((address, destination.Length));
        if (PendingGpuValues.Remove(address, out var pending))
        {
            WriteQword(address, pending);
        }

        return GuestMemory.TryRead(address, destination);
    }

    // A host that hands over command blocks, as the Metal host does. BlockVersion stands for the
    // counters of its token: a test bumps it where the real host would bump one.
    public bool SupportsCommandBlocks { get; set; }

    public long BlockVersion { get; set; }

    public List<(ulong Address, int Size)> BlockReads { get; } = new();

    public Func<ulong, int, bool>? RefusesBlock { get; set; }

    public bool TryReadCommandBlock(ulong address, Span<byte> block, out CommandReadToken token)
    {
        token = new CommandReadToken(BlockVersion, 0, 0, 0);
        BlockReads.Add((address, block.Length));
        return RefusesBlock?.Invoke(address, block.Length) != true && GuestMemory.TryRead(address, block);
    }

    public bool IsCommandBlockCurrent(in CommandReadToken token) => token.First == BlockVersion;

    // Polls that asked the host to commit the recording buffer so a pending write can land.
    public List<ulong> PollCommits { get; } = new();

    public bool TryReadGuestWaitOperand(ulong address, Span<byte> destination, bool commitPendingWrite)
    {
        if (commitPendingWrite)
        {
            PollCommits.Add(address);
        }

        return TryReadGuest(address, destination);
    }

    // Called before every packet, where a real host runs what other threads posted.
    public Action? OnRunPending { get; set; }

    public void RunPendingCommands()
    {
        PendingCommandRuns++;
        OnRunPending?.Invoke();
        while (PendingCommands.Count != 0)
        {
            PendingCommands.Dequeue()();
        }
    }

    public void BeginSubmission(int queueId, ulong submissionId, object? geometrySnapshots) => Calls.Add($"begin {queueId} {submissionId}");

    public void Flush() => Calls.Add("flush");

    public void FlushAndWait() => Calls.Add("flush_and_wait");

    public void SynchronizeGpu() => Calls.Add("synchronize");

    public void RunGarbageCollector() => Calls.Add("gc");

    public void EmitGlobalBarrier() => Calls.Add("barrier");

    // Counted rather than recorded in Calls: every barrier-emitting packet also releases.
    public int ShaderWriteReleases { get; private set; }

    public void ReleaseShaderWrites() => ShaderWriteReleases++;

    public void FillBuffer(ulong address, ulong size, uint value, bool isGds)
    {
        Calls.Add($"fill {address:X} {size:X} {value:X8} gds={isGds}");
        var pattern = new byte[size];
        for (var offset = 0; offset + 4 <= pattern.Length; offset += 4)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(pattern.AsSpan(offset), value);
        }

        if (isGds)
        {
            pattern.CopyTo(Gds.AsSpan((int)address));
        }
        else if (!GuestMemory.TryWrite(address, pattern))
        {
            throw Fatal($"fill outside memory {address:X}");
        }
    }

    public void CopyBuffer(ulong destination, ulong source, ulong size, bool destinationIsGds, bool sourceIsGds)
    {
        Calls.Add($"copy {destination:X} {source:X} {size:X} dstGds={destinationIsGds} srcGds={sourceIsGds}");
        var bytes = new byte[size];
        if (sourceIsGds)
        {
            Gds.AsSpan((int)source, (int)size).CopyTo(bytes);
        }
        else if (!GuestMemory.TryRead(source, bytes))
        {
            throw Fatal($"copy source outside memory {source:X}");
        }

        if (destinationIsGds)
        {
            bytes.CopyTo(Gds.AsSpan((int)destination));
        }
        else if (!GuestMemory.TryWrite(destination, bytes))
        {
            throw Fatal($"copy destination outside memory {destination:X}");
        }
    }

    public void ReadGds(Span<uint> destination, uint wordOffset, uint wordCount)
    {
        Calls.Add($"read_gds {wordOffset} {wordCount}");
        GlobalDataShare.ReadWords(Gds, destination, wordOffset, wordCount);
    }

    public void RecordEndOfPipe(in EndOfPipeWrite write)
    {
        EndOfPipeWrites.Add(write);
        Calls.Add($"eop {write.Kind}");
        EndOfPipeRecorded?.Invoke(write);
    }

    // A host that lands the graphics queue's command-processor writes in order with its labels.
    public bool DefersCommandProcessorWrites { get; set; }

    public List<(string Kind, ulong Address, byte[] Bytes)> CommandProcessorWrites { get; } = new();

    public void RecordCommandProcessorWrite(ulong address, ReadOnlySpan<byte> source, string kind)
    {
        CommandProcessorWrites.Add((kind, address, source.ToArray()));
        Calls.Add($"cp_write {kind} {address:X} {source.Length}");
    }

    // A host that takes a run of identical qword stores as one call, as the Metal host does.
    public bool RecordsQwordRuns { get; set; }

    public bool PublishesSyntheticOcclusionDumps { get; set; }

    public void RecordCommandProcessorQwords(ulong address, int count, ulong strideBytes, ulong value, string kind)
    {
        if (!RecordsQwordRuns)
        {
            var bytes = BitConverter.GetBytes(value);
            for (var index = 0; index < count; index++)
            {
                RecordCommandProcessorWrite(address + (ulong)index * strideBytes, bytes, kind);
            }

            return;
        }

        Calls.Add($"cp_qwords {kind} {address:X} {count} {strideBytes}");
        for (var index = 0; index < count; index++)
        {
            CommandProcessorWrites.Add((kind, address + (ulong)index * strideBytes, BitConverter.GetBytes(value)));
        }
    }

    public void FillBufferInOrder(ulong address, ulong size, uint value) =>
        Calls.Add($"fill_in_order {address:X} {size:X} {value:X8}");

    public void CopyBufferInOrder(ulong destination, ulong source, ulong size) =>
        Calls.Add($"copy_in_order {destination:X} {source:X} {size:X}");

    public void TriggerInterrupt(int eventId, uint contextId) => Calls.Add($"interrupt {eventId} {contextId}");

    public bool HasFlipSlot() => FlipSlot;

    public ulong PrepareFlip(int handle, int index, int flipMode, long flipArgument)
    {
        Calls.Add($"prepare_flip {handle} {index} {flipMode} {flipArgument}");
        return NextFlipRequestId++;
    }

    public void PrepareCpuFlip(int handle, int index, ulong requestId) => Calls.Add($"cpu_flip {handle} {index} {requestId}");

    public bool IsFlipDone(int handle, int index)
    {
        Calls.Add($"flip_done? {handle} {index}");
        return FlipDone;
    }

    public void DrawIndexed(ulong submitId, in DrawIndexedArguments arguments)
    {
        IndexedDraws.Add(arguments);
        Calls.Add($"draw_indexed {submitId} {arguments.IndexCount}");
    }

    public void DrawAuto(ulong submitId, in DrawAutoArguments arguments)
    {
        AutoDraws.Add(arguments);
        Calls.Add($"draw_auto {submitId} {arguments.VertexCount}");
    }

    public void DispatchDirect(ulong submitId, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator, ulong indirectArgumentsAddress = 0) =>
        Calls.Add(indirectArgumentsAddress == 0
            ? $"dispatch {submitId} {groupsX} {groupsY} {groupsZ} {dispatchInitiator:X}"
            : $"dispatch {submitId} {groupsX} {groupsY} {groupsZ} {dispatchInitiator:X} @{indirectArgumentsAddress:X}");

    public bool ResolvesIndirectDispatchOnGpu { get; set; }

    public bool ResolvesIndirectDrawOnGpu { get; set; }

    public bool ResolvesIndirectAutoDrawOnGpu { get; set; }

    /// <summary>Guest ranges GPU work still writes; null answers every range as pending.</summary>
    public List<(ulong Address, ulong Size)>? PendingGpuWrites { get; set; }

    public bool HasPendingGpuWrite(ulong address, ulong size) =>
        PendingGpuWrites is not { } pending ||
        pending.Exists(range => address < range.Address + range.Size && range.Address < address + size);

    /// <summary>Operand ranges recorded work still owes; a read of one commits it and parks.</summary>
    public List<(ulong Address, ulong Size)> OwedOperandWrites { get; } = [];

    public bool CommitPendingOperandWrite(ulong address, ulong size)
    {
        if (!OwedOperandWrites.Exists(range => address < range.Address + range.Size && range.Address < address + size))
        {
            return false;
        }

        Calls.Add($"commit_operand {address:X} {size}");
        return true;
    }

    public List<ulong> IndirectCpuReads { get; } = [];

    public void NoteIndirectArgumentsReadOnCpu(ulong address, ulong size) => IndirectCpuReads.Add(address);

    public void OnQueueReset(int queueId) => Calls.Add($"queue_reset {queueId}");

    public Exception Fatal(string message) => new CommandStreamFatalException(message);

    public uint ReadDword(ulong address)
    {
        Span<byte> bytes = stackalloc byte[4];
        Assert.True(GuestMemory.TryRead(address, bytes));
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    public ulong ReadQword(ulong address)
    {
        Span<byte> bytes = stackalloc byte[8];
        Assert.True(GuestMemory.TryRead(address, bytes));
        return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
    }

    public void WriteDword(ulong address, uint value)
    {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        Assert.True(GuestMemory.TryWrite(address, bytes));
    }

    public void WriteQword(ulong address, ulong value)
    {
        Span<byte> bytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        Assert.True(GuestMemory.TryWrite(address, bytes));
    }

    public void WriteWords(ulong address, ReadOnlySpan<uint> words) =>
        Assert.True(GuestMemory.TryWrite(address, MemoryMarshal.AsBytes(words)));
}

// Builds packet words and runs them through one processor, keeping the cursor stack for retries.
internal sealed class StreamRunner
{
    public const ulong CommandAddress = RecordingCommandStreamHost.MemoryBase + 0x1000;
    public const ulong LabelAddress = RecordingCommandStreamHost.MemoryBase + 0x8000;
    public const ulong TableAddress = RecordingCommandStreamHost.MemoryBase + 0x9000;
    public const ulong DataAddress = RecordingCommandStreamHost.MemoryBase + 0xA000;

    public StreamRunner(int queueId = 0, RecordingCommandStreamHost? host = null)
    {
        Host = host ?? new RecordingCommandStreamHost();
        Interpreter = new GpuCommandInterpreter(Host, queueId, queueId == 0 ? 0 : GpuCommandInterpreter.ComputeQueueBase + queueId - 1);
    }

    public RecordingCommandStreamHost Host { get; }

    public GpuCommandInterpreter Interpreter { get; }

    public PacketCursorStack Commands { get; private set; } = new();

    public uint Dwords { get; private set; }

    public static uint[] CustomPacket(uint opcode, uint customCode, params uint[] payload)
    {
        var words = new uint[payload.Length + 1];
        words[0] = PacketHeader.Make((uint)words.Length, opcode, customCode);
        payload.CopyTo(words, 1);
        return words;
    }

    public static uint[] Packet(uint opcode, params uint[] payload) => CustomPacket(opcode, 0, payload);

    public static uint[] Concat(params uint[][] packets)
    {
        var length = 0;
        foreach (var packet in packets)
        {
            length += packet.Length;
        }

        var words = new uint[length];
        var offset = 0;
        foreach (var packet in packets)
        {
            packet.CopyTo(words, offset);
            offset += packet.Length;
        }

        return words;
    }

    public void Load(params uint[][] packets) => Load(CommandAddress, Concat(packets));

    // Loading new packets starts a new stream; Run() without packets continues the current one.
    public void Load(ulong address, uint[] words)
    {
        Host.WriteWords(address, words);
        Dwords = (uint)words.Length;
        Commands = new PacketCursorStack();
    }

    public SubmissionProgress Run() => Interpreter.Process(Commands, CommandAddress, Dwords);

    public SubmissionProgress Run(params uint[][] packets)
    {
        Load(packets);
        return Run();
    }

    public CommandStreamFatalException RunExpectingFatal(params uint[][] packets)
    {
        Load(packets);
        return Assert.Throws<CommandStreamFatalException>(() => Run());
    }

    public static uint Low(ulong value) => (uint)value;

    public static uint High(ulong value) => (uint)(value >> 32);
}
