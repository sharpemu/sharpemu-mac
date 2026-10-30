// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpEmu.HLE.DsTrace;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Gpu.GpuCommands;

// The packet a handler executes: header fields plus its position in the cursor.
public readonly record struct PacketContext(uint Header, ulong PacketAddress, uint Offset, uint Remaining, uint Total)
{
    public uint Length => PacketHeader.Length(Header);

    public uint Opcode => PacketHeader.Opcode(Header);

    public uint CustomCode => PacketHeader.CustomCode(Header);
}

public readonly record struct FlipRequest(int Handle, int Index, int FlipMode, long FlipArgument);

// Parses one queue's command stream and keeps that queue's register and packet state.
public sealed partial class GpuCommandInterpreter
{
    public const int ComputeQueueBase = 0x20;
    public const uint ConstantRamDwords = 0x3000;
    public const uint RingChunkBytes = 0x10000;

    // SHARPEMU_GPU_STRICT_HEADERS: fail at the first header that is not a type-3 packet.
    internal bool StrictHeaders { get; set; } = GpuStreamDiagnostics.StrictHeaders;

    private readonly ICommandStreamHost _host;
    private readonly List<uint[]> _payloadScratchByDepth = new();
    private readonly uint[] _constantRam = new uint[ConstantRamDwords];
    private PacketCursorStack? _execution;
    private bool _chainRequested;
    private ulong _packetSerial;

    // DSTRACE SLICE: packets run so far, read by the queue around a slice.
    internal ulong DsPacketSerial => _packetSerial;

    // The packet the interpreter last read and how deep in indirect buffers (SHARPEMU_TRACE_BOUNDARY_WAITS).
    internal ulong LastPacketAddress;
    internal int LastPacketDepth;
    internal (uint Header, uint W1, uint W2, uint W3) LastPacketWords;
    private ulong _lastWriteAddress;
    private ulong _lastWriteValue;
    private int _lastWriteLength;
    private ulong _lastWritePacket;

    // The last packets this interpreter walked, so a stream that leaves its
    // packet boundaries says how it got there instead of only where it stopped.
    private readonly PacketTraceEntry[] _packetTrace = new PacketTraceEntry[32];
    private int _packetTraceCount;

    private struct PacketTraceEntry
    {
        public ulong Serial;
        public ulong Address;
        public uint Offset;
        public uint Header;
        public uint Length;
        public uint Consumed;
        public int Depth;
        public PacketTraceOutcome Outcome;
    }

    private enum PacketTraceOutcome : byte
    {
        Advanced,
        Filler,
        Predicated,
        DeferredAdvance,
        Suspended,
        Chained,
        Popped,
    }

    private void Trace(PacketTraceOutcome outcome, in PacketCursor cursor, uint offset,
        uint header, uint length, uint consumed, int depth)
    {
        _packetTrace[_packetTraceCount++ % _packetTrace.Length] = new PacketTraceEntry
        {
            Serial = _packetSerial,
            Address = cursor.Address,
            Offset = offset,
            Header = header,
            Length = length,
            Consumed = consumed,
            Depth = depth,
            Outcome = outcome,
        };
    }

    // An unknown opcode means either this interpreter walked off the packet grid
    // or the words under the cursor are not the ones it was handed. Reading the
    // header again separates the two: a guest that rebuilds a command buffer it
    // has already submitted changes it between two reads microseconds apart, and
    // the trace above then describes a stream that no longer exists.
    private string DescribeConcurrentRewrite(ulong packetAddress, uint header)
    {
        Span<byte> word = stackalloc byte[sizeof(uint)];
        if (!_host.TryReadGuest(packetAddress, word))
        {
            return string.Empty;
        }

        var reread = BinaryPrimitives.ReadUInt32LittleEndian(word);
        return reread == header
            ? string.Empty
            : $" The guest rewrote this packet while it was being read: header=0x{header:X8} is now 0x{reread:X8}.";
    }

    private void DumpPacketTrace()
    {
        var count = Math.Min(_packetTraceCount, _packetTrace.Length);
        Console.Error.WriteLine($"[GPU][FATAL] The last {count} packets this queue walked, oldest first:");
        var first = _packetTraceCount - count;
        for (var index = first; index < _packetTraceCount; index++)
        {
            ref var entry = ref _packetTrace[index % _packetTrace.Length];
            Console.Error.WriteLine(
                $"\tserial={entry.Serial} depth={entry.Depth} buffer=0x{entry.Address:X16} " +
                $"offset=0x{entry.Offset:X5} header=0x{entry.Header:X8} length={entry.Length} " +
                $"consumed={entry.Consumed} {entry.Outcome}");
        }
    }

    public GpuCommandInterpreter(ICommandStreamHost host, int queueId, int interruptEventId)
    {
        _host = host;
        QueueId = queueId;
        InterruptEventId = interruptEventId;
        TypedRegisters = new RegisterBanks(host.Fatal);
        UsePacketBlocks = PacketBlockReads.Enabled && host.SupportsCommandBlocks;
    }

    public int QueueId { get; }

    public int InterruptEventId { get; }

    public bool IsComputeQueue => InterruptEventId >= ComputeQueueBase;

    public CommandRegisterBanks Registers { get; } = new();

    // The typed banks the render executor reads; the dictionaries feed the translation they replace.
    public RegisterBanks TypedRegisters { get; }

    public ulong SubmitId { get; set; }

    public uint IndexTypeAndSize { get; private set; }

    public uint IndexBufferSize { get; private set; }

    public ulong IndexBaseAddress { get; private set; }

    public ulong DrawIndirectArgumentsBase { get; private set; }

    public ulong DispatchIndirectArgumentsBase { get; private set; }

    // Persistent draw state: indirect draws update it for later draws. An indirect draw
    // resolved on the GPU leaves the count in guest memory; it is read only if a later
    // draw needs it.
    public uint InstanceCount
    {
        get
        {
            if (_deferredInstanceCountAddress != 0)
            {
                _instanceCount = ReadDword(_deferredInstanceCountAddress);
                _deferredInstanceCountAddress = 0;
            }

            return _instanceCount;
        }
        private set
        {
            _instanceCount = value;
            _deferredInstanceCountAddress = 0;
        }
    }

    private uint _instanceCount = 1;
    private ulong _deferredInstanceCountAddress;

    public uint DrawIndexOffset { get; private set; }

    public uint UserDataMarker { get; private set; }

    public bool PredicateSkip { get; private set; }

    public bool ConditionalWaitEnabled { get; private set; }

    public FlipRequest PendingFlip { get; private set; }

    public uint ConstantEngineCount { get; private set; }

    public uint DrawEngineCount { get; private set; }

    public bool ConstantEngineComplete { get; set; }

    // One bank for every command processor of a queue set: a query's begin and end dumps may run
    // on different queues, and the end must still read later than the begin or the pair counts no
    // samples. CommandStreamQueue shares one bank across its interpreters; a lone one owns its own.
    private readonly ulong[] _syntheticOcclusionCounters = new ulong[4];

    // The counter of a host with no occlusion counter: one for every interpreter, never reset.
    private static long _hostlessOcclusionCounter;

    internal ulong[] SyntheticOcclusionCounterBank
    {
        init => _syntheticOcclusionCounters = value.Length == 4
            ? value
            : throw new ArgumentException("The occlusion counter bank holds four counters.", nameof(value));
    }

    private ulong _occlusionCounterStart;
    private int _selectedOcclusionCounter;
    private uint _occlusionDumpStrideBytes = 16;
    private uint _occlusionDumpInstanceMask = 0xFFFF;

    public ulong SyntheticOcclusionCounter => _syntheticOcclusionCounters[_selectedOcclusionCounter];

    public ulong AtomicReturnMeData { get; private set; }

    public bool AtomicReturnMeValid { get; private set; }

    public ulong AtomicReturnPfpData { get; private set; }

    public bool AtomicReturnPfpValid { get; private set; }

    public ReadOnlySpan<uint> ConstantRam => _constantRam;

    public void Reset()
    {
        var before = TypedRegisters.Context;
        _lastWriteLength = 0;
        Registers.Reset();
        TypedRegisters.Reset();
        IndexTypeAndSize = 0;
        IndexBufferSize = 0;
        UserDataMarker = 0;
        DrawIndirectArgumentsBase = 0;
        DispatchIndirectArgumentsBase = 0;
        Array.Clear(_constantRam);
        if (Rendering.ScissorTrace.Instance is { Active: true } trace)
            trace.ContextOperation(SubmitId, QueueId, 0, "Reset", TypedRegisters, before);
    }

    // The queue-reset packets also drop the index base, the instance count and the wait condition.
    public void ResetForQueueReset()
    {
        Reset();
        IndexBaseAddress = 0;
        InstanceCount = 1;
        DrawIndexOffset = 0;
        ConditionalWaitEnabled = false;
        _host.OnQueueReset(QueueId);
    }

    public void ResetCounters()
    {
        DrawEngineCount = 0;
        ConstantEngineCount = 0;
        ConstantEngineComplete = false;
    }

    public void SetFlip(FlipRequest flip) => PendingFlip = flip;

    public SubmissionProgress Process(PacketCursorStack execution, ulong address, uint dwordCount)
    {
        if (_execution is not null)
        {
            throw _host.Fatal($"The command interpreter is already running a stream: queue={QueueId}.");
        }

        if (execution.IsEmpty && dwordCount != 0)
        {
            execution.Push(new PacketCursor(address, dwordCount, ringChunkBase: address));
        }

        execution.Suspended = false;
        execution.SuspendCondition = CommandWaitCondition.Unknown;
        execution.MadeProgress = false;
        // Forwarding is local to this slice, never a cached signal from an older submission.
        _lastWriteLength = 0;
        // So is whatever was copied out of a command buffer: the guest may have stored into it since.
        BeginPacketBlockSlice();
        _execution = execution;
        try
        {
            RunPackets(execution, 0);
        }
        finally
        {
            _execution = null;
        }

        return execution.IsEmpty ? SubmissionProgress.Complete : SubmissionProgress.Blocked;
    }

    // Runs a nested buffer to completion before the calling packet finishes.
    public void RunIndirectBuffer(ulong address, uint dwordCount)
    {
        var execution = RequireExecution();
        if (dwordCount == 0)
        {
            return;
        }

        if (execution.Depth >= PacketCursorStack.MaxDepth)
        {
            throw _host.Fatal($"The indirect buffer nesting is too deep: depth={execution.Depth} address=0x{address:X16} dwords={dwordCount}.");
        }

        var stopDepth = execution.Depth;
        ReadBehindEarlierWork(address);
        execution.Push(new PacketCursor(address, dwordCount, ringChunkBase: address));
        RunPackets(execution, stopDepth);
    }

    // Replaces the current buffer with another one at the same depth; the chaining packet ends its buffer.
    private void ChainToBuffer(ulong address, uint dwordCount, bool followedChunkAdvance = false)
    {
        var execution = RequireExecution();
        if (!followedChunkAdvance)
        {
            ReadBehindEarlierWork(address);
        }

        execution.Pop();
        execution.Push(new PacketCursor(address, dwordCount, ringChunkBase: address, followedChunkAdvance));
        _chainRequested = true;
    }

    // Earlier work in the stream can build the buffer a packet calls or chains to (a GPU pass that writes
    // a dispatch list). An operand read completes that work first, so the packets read next hold what it
    // wrote. A ring chunk the guest still appends to is read as it is written.
    private void ReadBehindEarlierWork(ulong address)
    {
        Span<byte> first = stackalloc byte[sizeof(uint)];
        _ = _host.TryReadGuestOperand(address, first);
    }

    public void Suspend() => SuspendOn(CommandWaitCondition.Unknown);

    /// <summary>
    /// Suspends this submission and records what it is waiting for, so the queue can
    /// wake it when that alone changes instead of re-running every blocked head.
    /// </summary>
    public void SuspendOn(in CommandWaitCondition condition)
    {
        var execution = RequireExecution();
        execution.Suspended = true;
        execution.SuspendCondition = condition;
        if (DsTrace.Gpu)
        {
            DsEmitSuspend(in condition);
        }
    }

    // DSTRACE CPS: the compare operands of the memory wait about to suspend. Only the
    // queue's own (worker) thread runs this interpreter, so plain fields suffice.
    private bool _dsHasWaitOperands;
    private ulong _dsWaitReference;
    private ulong _dsWaitMask;
    private uint _dsWaitFunction;
    private ulong _dsWaitCurrent;

    private void DsNoteWaitOperands(ulong reference, ulong mask, uint function, ulong current)
    {
        _dsWaitReference = reference;
        _dsWaitMask = mask;
        _dsWaitFunction = function;
        _dsWaitCurrent = current;
        _dsHasWaitOperands = true;
    }

    private void DsEmitSuspend(in CommandWaitCondition condition)
    {
        var hasOperands = _dsHasWaitOperands;
        _dsHasWaitOperands = false;
        if (!DsTrace.Armed)
        {
            return;
        }

        DsTrace.Emit(
            DsEv.CPS,
            (ulong)QueueId,
            (ulong)condition.Kind,
            condition.Address,
            hasOperands ? _dsWaitReference : 0,
            hasOperands ? _dsWaitMask : 0,
            hasOperands ? _dsWaitFunction : 0,
            hasOperands ? _dsWaitCurrent : 0);
    }

    // A paired wait must still observe this queue's immediate label store when the CPU
    // resets the live address for the next frame between the two packets (Dead Cells).
    private bool TryReadOwnWrite(ulong address, bool is64Bit, out ulong value)
    {
        value = 0;
        if (address != _lastWriteAddress || _packetSerial - _lastWritePacket > 2 ||
            _lastWriteLength < (is64Bit ? sizeof(ulong) : sizeof(uint)))
        {
            return false;
        }
        value = is64Bit ? _lastWriteValue : (uint)_lastWriteValue;
        return true;
    }

    private PacketCursorStack RequireExecution() =>
        _execution ?? throw _host.Fatal($"No command stream is running: queue={QueueId}.");

    private void RunPackets(PacketCursorStack execution, int stopDepth)
    {
        while (execution.Depth > stopDepth)
        {
            _host.RunPendingCommands();
            var cursorIndex = execution.Depth - 1;
            ref var cursor = ref execution.At(cursorIndex);
            if (cursor.Offset > cursor.DwordCount)
            {
                DumpPacketTrace();
                throw _host.Fatal($"The packet cursor is past its buffer: offset={cursor.Offset} dwords={cursor.DwordCount} address=0x{cursor.Address:X16}.");
            }

            if (cursor.DeferredAdvance != 0)
            {
                if (cursor.DeferredAdvance > cursor.Remaining)
                {
                    DumpPacketTrace();
                    throw _host.Fatal($"The deferred advance is past the buffer: advance={cursor.DeferredAdvance} remaining={cursor.Remaining} address=0x{cursor.Address:X16}.");
                }

                Trace(PacketTraceOutcome.DeferredAdvance, in cursor, cursor.Offset, 0, 0,
                    cursor.DeferredAdvance, cursorIndex);
                cursor.Offset += cursor.DeferredAdvance;
                cursor.DeferredAdvance = 0;
                execution.MadeProgress = true;
                continue;
            }

            if (cursor.Offset == cursor.DwordCount)
            {
                Trace(PacketTraceOutcome.Popped, in cursor, cursor.Offset, 0, 0, 0, cursorIndex);
                execution.Pop();
                continue;
            }

            var packetAddress = cursor.Address + ((ulong)cursor.Offset * sizeof(uint));
            if (CommandStreamQueue.TraceBoundaryWaits)
            {
                LastPacketAddress = packetAddress;
                LastPacketDepth = cursorIndex;
            }

            // The header and the body of a packet are consecutive, and almost every
            // packet is short, so one read of a small window covers both. Reading the
            // header on its own first doubled the number of guest reads the command
            // stream makes, and a guest read is not free: the backend has to decide
            // whether the GPU may still be writing those bytes.
            ReadOnlySpan<uint> window;
            using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.CommandPacketRead))
            {
                window = ReadPacketWindow(cursorIndex, packetAddress, cursor.Remaining);
            }

            var header = window[0];
            if (CommandStreamQueue.TraceBoundaryWaits)
            {
                LastPacketWords = (header, window.Length > 1 ? window[1] : 0u, window.Length > 2 ? window[2] : 0u, window.Length > 3 ? window[3] : 0u);
            }

            if (PacketProfile.Enabled)
            {
                PacketProfile.RecordHeaderRead();
            }
            _packetSerial++;
            var total = cursor.DwordCount;
            var remaining = cursor.Remaining;
            var offset = cursor.Offset;

            if (header == PacketHeader.FillerHeader)
            {
                Trace(PacketTraceOutcome.Filler, in cursor, offset, header, 1, 1, cursorIndex);
                cursor.Offset++;
                execution.MadeProgress = true;
                continue;
            }

            // Ring memory the guest has not written yet; retry after it appends more.
            if (header == 0 && cursor.FollowedChunkAdvance)
            {
                GpuStreamDiagnostics.RecordRingSuspend();
                execution.Suspended = true;
                execution.SuspendCondition = new CommandWaitCondition(CommandWaitKind.RingAppend);
                return;
            }

            if (StrictHeaders && PacketHeader.PacketType(header) != 3)
            {
                ReportStreamContext("Strict header check");
                throw _host.Fatal(
                    $"The packet header is not type 3: offset=0x{offset:X5} header=0x{header:X8} " +
                    $"type={PacketHeader.PacketType(header)} address=0x{packetAddress:X16}.");
            }

            if (remaining < 2)
            {
                DumpPacketTrace();
                throw _host.Fatal($"The packet is shorter than two dwords: offset=0x{offset:X5} header=0x{header:X8} address=0x{packetAddress:X16}.");
            }

            var length = PacketHeader.Length(header);
            if (length > remaining)
            {
                DumpPacketTrace();
                throw _host.Fatal($"The packet is longer than the buffer: offset=0x{offset:X5} header=0x{header:X8} length={length} remaining={remaining} address=0x{packetAddress:X16}.");
            }

            if (PacketHeader.IsPredicated(header) && PredicateSkip)
            {
                Trace(PacketTraceOutcome.Predicated, in cursor, offset, header, length, length, cursorIndex);
                TraceSkippedPackets("predicated", packetAddress, length, PacketHeader.Opcode(header));
                cursor.Offset += length;
                execution.MadeProgress = true;
                continue;
            }

            var opcode = PacketHeader.Opcode(header);
            // Only a wait or an inert NOP/marker may separate the store and its paired wait.
            // Draws, dispatches and transfers may produce a newer GPU value at the same address.
            if (opcode is not (PacketOpcode.WaitRegisterMemory or PacketOpcode.WaitRegisterMemory64) &&
                !(opcode == PacketOpcode.Nop && PacketHeader.CustomCode(header) is
                    0 or PacketCustomCode.PushMarker or PacketCustomCode.PopMarker or
                    PacketCustomCode.WaitMemory32 or PacketCustomCode.WaitMemory64))
            {
                _lastWriteLength = 0;
            }
            var handler = PacketDispatchTable.Opcodes[opcode];
            if (handler is null)
            {
                DumpUnknownPacket(cursor, offset);
                DumpPacketTrace();
                throw _host.Fatal(
                    $"The packet opcode is unknown: offset=0x{offset:X5} header=0x{header:X8} " +
                    $"opcode=0x{opcode:X2} address=0x{packetAddress:X16}.{DescribeConcurrentRewrite(packetAddress, header)}");
            }

            // The window already holds the body unless the packet is a long one.
            ReadOnlySpan<uint> payload;
            if (length <= (uint)window.Length)
            {
                payload = window.Slice(1, (int)(length - 1));
            }
            else
            {
                using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.CommandPacketRead))
                {
                    payload = ReadPayload(cursorIndex, packetAddress, length - 1, remaining);
                }
            }
            var packet = new PacketContext(header & ~1u, packetAddress, offset, remaining, total);
            uint consumed;
            if (PacketProfile.Enabled)
            {
                var packetStarted = System.Diagnostics.Stopwatch.GetTimestamp();
                consumed = handler(this, in packet, payload) + 1;
                PacketProfile.RecordPacket(opcode, System.Diagnostics.Stopwatch.GetTimestamp() - packetStarted);
            }
            else
            {
                consumed = handler(this, in packet, payload) + 1;
            }

            // The header's count is the size of the packet, so a handler that
            // reports fewer dwords than its own header declares has not consumed
            // the whole packet: advancing by its count would resume inside the
            // body and read an operand as the next header. Several handlers
            // return a fixed size for the variant they know (the register NOP
            // offsets, the semaphore, the atomic), and a longer encoding of the
            // same packet would walk the stream off its boundaries from there on.
            // COND_EXEC is the one packet that reports more on purpose, because
            // it skips the packets it covers.
            if (consumed < length)
            {
                consumed = length;
            }

            if (consumed > remaining)
            {
                DumpPacketTrace();
                throw _host.Fatal($"The handler consumed more than the buffer holds: consumed={consumed} remaining={remaining} header=0x{header:X8} address=0x{packetAddress:X16}.");
            }

            if (execution.Suspended)
            {
                Trace(PacketTraceOutcome.Suspended, in execution.At(cursorIndex), offset, header,
                    length, consumed, cursorIndex);
                if (execution.Depth > cursorIndex + 1 && !_chainRequested)
                {
                    execution.At(cursorIndex).DeferredAdvance = consumed;
                }

                _chainRequested = false;
                return;
            }

            if (_chainRequested)
            {
                Trace(PacketTraceOutcome.Chained, in execution.At(cursorIndex), offset, header,
                    length, consumed, cursorIndex);
                _chainRequested = false;
                execution.MadeProgress = true;
                continue;
            }

            if (execution.Depth != cursorIndex + 1)
            {
                throw _host.Fatal($"The cursor stack changed under a packet: depth={execution.Depth} expected={cursorIndex + 1} header=0x{header:X8}.");
            }

            Trace(PacketTraceOutcome.Advanced, in execution.At(cursorIndex), offset, header,
                length, consumed, cursorIndex);
            execution.At(cursorIndex).Offset += consumed;
            execution.MadeProgress = true;
        }
    }

    // The payload lives in a per-depth scratch buffer for the handler call only.
    // How much of a packet is read in one go. Long packets (large register writes,
    // constant-RAM dumps) fall back to a second read for the rest of their body.
    private const uint PacketWindowDwords = 16;

    // The header and as much of the body as the window holds, in one guest read.
    private ReadOnlySpan<uint> ReadPacketWindow(int depth, ulong packetAddress, uint remaining)
    {
        var dwords = Math.Min(remaining, PacketWindowDwords);
        var blockWords = BlockWords(depth, packetAddress, remaining, dwords, dwords);
        if (!blockWords.IsEmpty)
        {
            return blockWords;
        }

        while (_payloadScratchByDepth.Count <= depth)
        {
            _payloadScratchByDepth.Add(new uint[PacketHeader.MaxLength]);
        }

        var scratch = _payloadScratchByDepth[depth].AsSpan(0, (int)dwords);
        RenderPhaseProfile.RecordCommandRead(
            RenderPhaseProfile.CommandReadKind.Header, scratch.Length * sizeof(uint));
        if (!_host.TryReadGuest(packetAddress, MemoryMarshal.AsBytes(scratch)))
        {
            throw _host.Fatal(
                $"The command stream cannot read a packet header: address=0x{packetAddress:X16} dwords={dwords}.");
        }

        return scratch;
    }

    private ReadOnlySpan<uint> ReadPayload(int depth, ulong packetAddress, uint payloadDwords, uint remaining)
    {
        var blockWords = BlockWords(depth, packetAddress, remaining, payloadDwords + 1, payloadDwords + 1);
        if (!blockWords.IsEmpty)
        {
            return blockWords[1..];
        }

        while (_payloadScratchByDepth.Count <= depth)
        {
            _payloadScratchByDepth.Add(new uint[PacketHeader.MaxLength]);
        }

        var scratch = _payloadScratchByDepth[depth].AsSpan(0, (int)payloadDwords);
        if (payloadDwords != 0 && PacketProfile.Enabled)
        {
            PacketProfile.RecordPayloadRead(payloadDwords);
        }

        if (payloadDwords != 0)
        {
            RenderPhaseProfile.RecordCommandRead(RenderPhaseProfile.CommandReadKind.Payload, scratch.Length * sizeof(uint));
        }
        if (payloadDwords != 0 && !_host.TryReadGuest(packetAddress + sizeof(uint), MemoryMarshal.AsBytes(scratch)))
        {
            throw _host.Fatal($"The command stream cannot read a packet: address=0x{packetAddress:X16} dwords={payloadDwords + 1}.");
        }

        return scratch;
    }

    private void DumpUnknownPacket(in PacketCursor cursor, uint offset)
    {
        var begin = offset > 8 ? offset - 8 : 0;
        var end = Math.Min(cursor.DwordCount, offset + 16);
        Console.Error.WriteLine($"[GPU][FATAL] The packet stream near the unknown packet: buffer=0x{cursor.Address:X16} dwords={cursor.DwordCount} offset=0x{offset:X5}");
        Span<byte> word = stackalloc byte[sizeof(uint)];
        for (var index = begin; index < end; index++)
        {
            var text = _host.TryReadGuest(cursor.Address + ((ulong)index * sizeof(uint)), word)
                ? BinaryPrimitives.ReadUInt32LittleEndian(word).ToString("X8")
                : "????????";
            Console.Error.WriteLine($"\t{index:X5}{(index == offset ? ":" : " ")} {text}");
        }
    }

    internal uint ReadDword(ulong address, RenderPhaseProfile.CommandReadKind readKind = RenderPhaseProfile.CommandReadKind.Operand32)
    {
        RenderPhaseProfile.RecordCommandRead(readKind, sizeof(uint));
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        // The packets themselves are command data; anything else is an operand earlier work may write.
        var read = readKind is RenderPhaseProfile.CommandReadKind.Header or RenderPhaseProfile.CommandReadKind.Payload or
            RenderPhaseProfile.CommandReadKind.RegisterTable
            ? _host.TryReadGuest(address, bytes)
            : _host.TryReadGuestOperand(address, bytes);
        if (!read)
        {
            throw _host.Fatal($"The command stream cannot read guest memory: address=0x{address:X16} size=4.");
        }

        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    internal ulong ReadQword(ulong address)
    {
        RenderPhaseProfile.RecordCommandRead(RenderPhaseProfile.CommandReadKind.Operand64, sizeof(ulong));
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        if (!_host.TryReadGuestOperand(address, bytes))
        {
            throw _host.Fatal($"The command stream cannot read guest memory: address=0x{address:X16} size=8.");
        }

        return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
    }

    /// <summary>
    /// Reads the bytes a wait packet polls, as they are in memory right now.
    /// </summary>
    /// <remarks>
    /// A poll is not an ordinary read: the hardware's command processor reads what memory
    /// holds and stalls on the value it finds, so waiting here for work that may be about
    /// to write those bytes only moves the stall onto the worker thread. An unmet compare
    /// suspends the submission instead, and the host wakes it when the write lands.
    /// </remarks>
    private ulong ReadWaitOperand(ulong address, bool is64Bit, bool commitPendingWrite)
    {
        RenderPhaseProfile.RecordCommandRead(is64Bit ? RenderPhaseProfile.CommandReadKind.Operand64 : RenderPhaseProfile.CommandReadKind.Operand32,
            is64Bit ? sizeof(ulong) : sizeof(uint));
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        var size = is64Bit ? sizeof(ulong) : sizeof(uint);
        if (!_host.TryReadGuestWaitOperand(address, bytes[..size], commitPendingWrite))
        {
            throw _host.Fatal($"The command stream cannot read guest memory: address=0x{address:X16} size={size}.");
        }

        return is64Bit ? BinaryPrimitives.ReadUInt64LittleEndian(bytes) : BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    /// <summary>
    /// Reads an operand earlier work on this queue may still be writing, or suspends the
    /// submission at this packet while that work finishes.
    /// </summary>
    /// <remarks>
    /// The command processor reads such an operand after the work ahead of it retires, and
    /// stalls in the meantime. Stalling the worker until the GPU drains holds every other
    /// queue behind this one, so the host commits the owing work and the submission parks on
    /// the bytes instead; the packet runs again, from its start, once they land. A host that
    /// cannot tell reads the bytes as they are now.
    /// </remarks>
    internal bool TryReadOperandOrSuspend(ulong address, Span<byte> destination)
    {
        // A label this queue recorded ahead of the packet is what the operand holds once the
        // work before it retires, which is when the command processor would read it.
        if (destination.Length is sizeof(uint) or sizeof(ulong) && PendingLabelLookup is { } pending &&
            pending(address, destination.Length == sizeof(ulong), out var labelValue))
        {
            if (destination.Length == sizeof(ulong))
            {
                BinaryPrimitives.WriteUInt64LittleEndian(destination, labelValue);
            }
            else
            {
                BinaryPrimitives.WriteUInt32LittleEndian(destination, (uint)labelValue);
            }

            return true;
        }

        if (_host.CommitPendingOperandWrite(address, (ulong)destination.Length))
        {
            Interlocked.Increment(ref _operandSuspends);
            SuspendOn(CommandWaitCondition.Memory(address, (ulong)destination.Length));
            return false;
        }

        ReadBytes(address, destination);
        return true;
    }

    private static long _operandSuspends;

    /// <summary>Packets since startup that parked on an operand earlier work still owed instead of draining the GPU.</summary>
    public static long OperandSuspends => Interlocked.Read(ref _operandSuspends);

    internal void ReadBytes(ulong address, Span<byte> destination)
    {
        RenderPhaseProfile.RecordCommandRead(RenderPhaseProfile.CommandReadKind.Other, destination.Length);
        if (!_host.TryReadGuestOperand(address, destination))
        {
            throw _host.Fatal($"The command stream cannot read guest memory: address=0x{address:X16} size={destination.Length}.");
        }
    }

    private static long _skippedPacketTraceCount;

    // Skipped packets are silent otherwise; a skipped producer leaves its consumer with stale memory.
    internal void TraceSkippedPackets(string kind, ulong packetAddress, uint dwords, uint detail)
    {
        if (IndexedDrawTrace.Enabled)
            SkippedIndexedDrawScanner.Scan(_host.Memory.TryRead, packetAddress, dwords, IndexTypeAndSize, IndexBaseAddress,
                (count, type, address, packet) => IndexedDrawTrace.Write(
                    IndexedDrawTrace.Capture(count, type, address, _host.Memory.TryRead),
                    $"{kind}-candidate", SubmitId, packet), IndexedDrawTrace.ScanIssue);
        if (!ImageClearTrace.Enabled) return;
        _host.TraceGuestWrite($"command-skip-{kind}", packetAddress, (ulong)dwords * sizeof(uint));
        if (Interlocked.Increment(ref _skippedPacketTraceCount) > 256) return;
        var target = string.Empty;
        Span<byte> body = stackalloc byte[3 * sizeof(uint)];
        if (detail == PacketOpcode.IndirectBuffer && dwords == 4 && _host.TryReadGuest(packetAddress + sizeof(uint), body))
        {
            var words = MemoryMarshal.Cast<byte, uint>(body);
            var targetAddress = words[0] | ((ulong)(words[1] & 0xFFFFu) << 32);
            var targetDwords = words[2] & 0xFFFFFu;
            target = $" target=0x{targetAddress:X16} targetDwords={targetDwords} chain={(words[2] >> 20) & 1u}";
            TraceSkippedTargetPackets(targetAddress, targetDwords);
        }
        Console.Error.WriteLine($"[GPU][TRACE] CommandSkip kind={kind} packet=0x{packetAddress:X16} dwords={dwords} detail=0x{detail:X8}{target}");
    }

    private static readonly HashSet<ulong> _decodedSkipTargets = new();

    // Lists each packet of a skipped buffer once, so the skipped work can be identified.
    private void TraceSkippedTargetPackets(ulong address, uint dwords)
    {
        lock (_decodedSkipTargets)
        {
            if (_decodedSkipTargets.Count >= 16 || !_decodedSkipTargets.Add(address)) return;
        }
        var buffer = new byte[Math.Min(dwords, 4096u) * sizeof(uint)];
        if (buffer.Length == 0 || !_host.TryReadGuest(address, buffer)) return;
        var words = MemoryMarshal.Cast<byte, uint>(buffer);
        var packets = new List<string>();
        for (var offset = 0; offset < words.Length;)
        {
            var header = words[offset];
            if ((header >> 30) != 3)
            {
                packets.Add($"raw:0x{header:X8}");
                offset++;
                continue;
            }
            var length = (int)PacketHeader.Length(header);
            var marker = string.Empty;
            if (PacketHeader.Opcode(header) == PacketOpcode.Nop && PacketHeader.CustomCode(header) == PacketCustomCode.PushMarker &&
                offset + length <= words.Length)
            {
                var text = MemoryMarshal.AsBytes(words.Slice(offset + 1, length - 1));
                var end = text.IndexOf((byte)0);
                marker = $"\"{System.Text.Encoding.ASCII.GetString(end < 0 ? text : text[..end])}\"";
            }
            packets.Add($"0x{PacketHeader.Opcode(header):X2}/{length}{(PacketHeader.IsPredicated(header) ? "p" : "")}" +
                (PacketHeader.Opcode(header) == PacketOpcode.Nop ? $"[r=0x{PacketHeader.CustomCode(header):X2}]" : "") + marker);
            offset += Math.Max(length, 1);
        }
        Console.Error.WriteLine($"[GPU][TRACE] CommandSkipTarget target=0x{address:X16} dwords={dwords} packets={string.Join(',', packets)}");
    }

    internal void WriteDword(ulong address, uint value, string kind = "WRITE32", [CallerMemberName] string caller = "")
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        WriteBytes(address, bytes, kind, caller);
    }

    internal void WriteQword(ulong address, ulong value, string kind = "WRITE64", [CallerMemberName] string caller = "")
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        WriteBytes(address, bytes, kind, caller);
    }

    /// <summary>
    /// Told when this interpreter writes guest memory, so the queue can wake exactly
    /// the submissions whose compare reads those bytes. The queue sets it.
    /// </summary>
    internal Action<ulong, ulong>? GuestMemoryWritten { get; init; }

    /// <summary>Told when this interpreter defers a label to the GPU; the queue keeps it
    /// visible to other queues' waits until it lands.</summary>
    internal Action<ulong, int, ulong>? PendingLabelRecorded { get; init; }

    /// <summary>The same for a run of eight-byte labels of one value at a fixed stride, as one call.</summary>
    internal Action<ulong, int, ulong, ulong>? PendingQwordsRecorded { get; init; }

    internal PendingLabelReader? PendingLabelLookup { get; init; }

    internal delegate bool PendingLabelReader(ulong address, bool is64Bit, out ulong value);

    // The caller name labels the packet handler in the image-clear trace.
    internal void WriteBytes(ulong address, ReadOnlySpan<byte> source, string kind = "WRITE_BYTES", [CallerMemberName] string caller = "")
    {
        NotePacketBlockWrite();
        if (_host.TryDeferOrderedWrite(QueueId, address, source))
        {
            _lastWriteLength = 0;
            return;
        }

        if (!_host.TryWriteGuest(address, source, kind))
        {
            throw _host.Fatal($"The command stream cannot write guest memory: address=0x{address:X16} size={source.Length}.");
        }

        GuestMemoryWritten?.Invoke(address, (ulong)source.Length);
        if (CommandProcessorWriteHistory.Enabled && source.Length is sizeof(uint) or sizeof(ulong))
        {
            CommandProcessorWriteHistory.Record(QueueId, address, source.Length,
                source.Length == sizeof(uint)
                    ? BinaryPrimitives.ReadUInt32LittleEndian(source)
                    : BinaryPrimitives.ReadUInt64LittleEndian(source),
                kind, deferred: false);
        }

        _lastWriteLength = 0;
        if (source.Length is sizeof(uint) or sizeof(ulong))
        {
            _lastWriteAddress = address;
            _lastWriteValue = source.Length == sizeof(uint)
                ? BinaryPrimitives.ReadUInt32LittleEndian(source)
                : BinaryPrimitives.ReadUInt64LittleEndian(source);
            _lastWriteLength = source.Length;
            _lastWritePacket = _packetSerial;
        }
        if (ImageClearTrace.Enabled) _host.TraceGuestWrite($"command-write-{caller}", address, (ulong)source.Length);
    }

    private static ulong Address(uint low, uint high) => low | ((ulong)high << 32);
}
