// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.DsTrace;
using SharpEmu.Libs.Gpu.GpuCommands.Packets;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public sealed partial class GpuCommandInterpreter
{
    private const uint RewindValidBit = 1u << 31;
    private const uint RewindKnownBits = 0x8100_0000u;

    // Polls the label once; an unmet condition suspends the submission at this packet.
    internal void WaitOnMemory(uint compareFunction, ulong address, ulong reference, ulong mask, uint waitOperation, bool is64Bit)
    {
        if (address == 0)
        {
            throw _host.Fatal($"The wait address is zero: compare={compareFunction} reference=0x{reference:X16}.");
        }

        if (!WaitOperation.IsValid(waitOperation))
        {
            throw _host.Fatal($"The wait operation is not supported: operation=0x{waitOperation:X8} address=0x{address:X16}.");
        }

        if (!WaitOperation.ShouldExecute(waitOperation, ConditionalWaitEnabled))
        {
            return;
        }

        // A deferred label is visible here at once; guest CPU threads see it after the dump before it.
        if (!_host.TryReadDeferredLabel(QueueId, address, is64Bit ? 8UL : 4UL, out var deferred, out var value))
        {
            Suspend();
            return;
        }

        // The first look commits nothing. Committing the recording command buffer so a
        // pending write can land ends the open render pass, and when this queue's own store
        // or a recorded label satisfies the wait the commit buys nothing: nine in ten poll
        // commits on Void Terrarium, Superliminal and Quake resumed without suspending, each
        // one storing and reloading every attachment of the pass.
        if (!deferred)
        {
            value = ReadWaitOperand(address, is64Bit, commitPendingWrite: false);
        }

        if (!WaitOperation.TryCompare(value, reference, mask, compareFunction, out var satisfied))
        {
            throw _host.Fatal($"The wait compare function is unknown: function={compareFunction} address=0x{address:X16}.");
        }

        if (!satisfied && TryReadOwnWrite(address, is64Bit, out var ownValue))
        {
            _ = WaitOperation.TryCompare(ownValue, reference, mask, compareFunction, out satisfied);
        }

        if (!satisfied && PendingLabelLookup is { } pending && pending(address, is64Bit, out var labelValue))
        {
            _ = WaitOperation.TryCompare(labelValue, reference, mask, compareFunction, out satisfied);
        }

        if (!satisfied && !deferred)
        {
            // Nothing recorded will satisfy it, so the bytes have to change: make sure the
            // work that writes them is on its way before the submission sleeps on them.
            value = ReadWaitOperand(address, is64Bit, commitPendingWrite: true);
            _ = WaitOperation.TryCompare(value, reference, mask, compareFunction, out satisfied);
        }

        if (!satisfied)
        {
            if (DsTrace.Gpu)
            {
                DsNoteWaitOperands(reference, mask, compareFunction, value);
            }

            SuspendOn(CommandWaitCondition.WatchedCompare(address, is64Bit, value, reference, mask, compareFunction));
        }
    }

    private const ulong QueryReadyBit = 1UL << 63;
    private const int ZPassDepthBlocks = 16;

    // What the last predication found, for a zpass packet that continues the set.
    private bool _predicationVisible;

    // Predication is set (not cleared): the draws that follow may be the ones it decides on,
    // so a backend never leaves one out while its pipeline compiles (DrawsMustExecute).
    private bool _predicationSet;

    // Between an occlusion query's begin and end dumps (ZPASS_DONE at a block's even, then
    // odd, qword): the draws inside are what the query counts.
    private bool _occlusionQueryOpen;

    /// <summary>
    /// Draws a backend has to run even while their pipeline is still compiling: predicated
    /// ones and those inside an occlusion query, whose outcome the guest reads back.
    /// </summary>
    internal bool DrawsMustExecute => _predicationSet || _occlusionQueryOpen;

    /// <summary>
    /// Sets the predicate the packets with header bit 0 obey.
    /// </summary>
    /// <remarks>
    /// <para>The operation picks what the address holds: 1 the zpass (occlusion) counters,
    /// sixteen depth-block pairs of begin and end qwords with bit 63 set once written; 2 a
    /// streamout-statistics record of primitives written and needed, begin then end, which
    /// counts as visible while nothing overflowed; 3 a DX12 qword and 4 a Vulkan dword that
    /// count as visible when nonzero. The boolean says which outcome draws: 0 draws when
    /// not visible (or overflowed), 1 when visible.</para>
    /// <para>The hint is PredicationZPassWaitOp: 0 (kWaitForQueryResults) stalls the
    /// command processor until the results are in, so the submission parks on them; 1
    /// (kDoNotPredicateIfQueryResultsNotReady) never stalls and leaves the tagged packets
    /// unpredicated while the results are outstanding. The continue bit ORs a zpass result
    /// into the previous one, so one predicate can span several queries.</para>
    /// </remarks>
    internal void SetPredication(uint condition, uint operation, uint waitOperation, ulong address, bool continueSet = false)
    {
        _predicationSet = operation != 0;
        if (operation == 0)
        {
            // Clearing predication reads no memory, so no GPU result can be outstanding.
            PredicateSkip = false;
            _predicationVisible = false;
            return;
        }

        if (operation > 4)
        {
            throw _host.Fatal($"The predication operation is unknown: operation=0x{operation:X8} address=0x{address:X16}.");
        }

        if (condition > 1)
        {
            throw _host.Fatal($"The predication condition is unknown: condition=0x{condition:X8} address=0x{address:X16}.");
        }

        if (address == 0)
        {
            throw _host.Fatal($"The predication address is zero: operation={operation}.");
        }

        var size = operation switch
        {
            1 => (ulong)ZPassDepthBlocks * 2 * sizeof(ulong),
            2 => 4UL * sizeof(ulong),
            3 => (ulong)sizeof(ulong),
            _ => (ulong)sizeof(uint),
        };

        if (waitOperation != 0 && _host.HasPendingGpuWrite(address, size))
        {
            // The results are still on their way, and the game asked for the packets to run
            // unpredicated in that case.
            PredicateSkip = false;
            return;
        }

        Span<ulong> words = stackalloc ulong[ZPassDepthBlocks * 2];
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(words)[..(int)size];
        if (!TryReadOperandOrSuspend(address, bytes))
        {
            return;
        }

        bool visible;
        switch (operation)
        {
            case 1:
            {
                ulong passed = 0;
                for (var block = 0; block < ZPassDepthBlocks; block++)
                {
                    var begin = words[block * 2];
                    var end = words[(block * 2) + 1];
                    if ((begin & end & QueryReadyBit) == 0)
                    {
                        if (waitOperation != 0)
                        {
                            PredicateSkip = false;
                            return;
                        }

                        // The occlusion dumps that fill the pairs have not run; the command
                        // processor would stall here until they do.
                        Interlocked.Increment(ref _predicationResultWaits);
                        SuspendOn(CommandWaitCondition.Memory(address, size));
                        return;
                    }

                    passed += (end & ~QueryReadyBit) - (begin & ~QueryReadyBit);
                }

                visible = passed != 0;
                if (continueSet)
                {
                    visible |= _predicationVisible;
                }

                break;
            }

            case 2:
                // Overflow is the streamout having needed room for more primitives than it wrote.
                visible = words[3] - words[1] == words[2] - words[0];
                break;
            case 3:
                visible = words[0] != 0;
                break;
            default:
                visible = (uint)words[0] != 0;
                break;
        }

        _predicationVisible = visible;
        PredicateSkip = condition == 0 ? visible : !visible;
        TracePredication(condition, waitOperation, address, words[0]);
        VisibilityResultTrace.Predicate(QueueId, SubmitId, address, words[0], condition, waitOperation, PredicateSkip);
    }

    private static long _predicationResultWaits;

    /// <summary>Zpass predicates since startup that parked until the occlusion results were written.</summary>
    public static long PredicationResultWaits => Interlocked.Read(ref _predicationResultWaits);

    private static long _predicationTraceCount;
    private readonly Queue<(ulong Address, ulong Value, long Evaluation, long Timestamp)> _skippedPredicates = new();

    // Each evaluation decides which later predicated packets run; see docs/image-clear-tracing.md.
    private void TracePredication(uint condition, uint waitOperation, ulong address, ulong value)
    {
        if (!Images.ImageClearTrace.Enabled) return;
        var evaluation = Interlocked.Increment(ref _predicationTraceCount);
        RecheckSkippedPredicates(evaluation);
        if (evaluation > 256) return;
        Console.Error.WriteLine($"[GPU][TRACE] Predication address=0x{address:X16} value=0x{value:X16} condition={condition} " +
            $"wait={waitOperation} skip={PredicateSkip} evaluation={evaluation}");
        if (evaluation <= 32) _host.TraceWritersOf($"predicate=0x{address:X16}", address, sizeof(ulong));
        if (PredicateSkip && _skippedPredicates.Count < 64)
            _skippedPredicates.Enqueue((address, value, evaluation, System.Diagnostics.Stopwatch.GetTimestamp()));
    }

    // A value that changes after its evaluation shows a writer that ran too late for the skip decision.
    private void RecheckSkippedPredicates(long evaluation)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        for (var remaining = _skippedPredicates.Count; remaining > 0; remaining--)
        {
            var entry = _skippedPredicates.Dequeue();
            var current = _host.TryReadGuest(entry.Address, bytes) ? System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(bytes) : entry.Value;
            if (current != entry.Value)
            {
                var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(entry.Timestamp).TotalMilliseconds;
                Console.Error.WriteLine($"[GPU][TRACE] PredicationChanged address=0x{entry.Address:X16} evaluated=0x{entry.Value:X16} " +
                    $"now=0x{current:X16} evaluation={entry.Evaluation} laterEvaluations={evaluation - entry.Evaluation} elapsedMs={elapsed:F3}");
                continue;
            }
            if (evaluation - entry.Evaluation < 64) _skippedPredicates.Enqueue(entry);
        }
    }

    internal void WaitForConstantEngine()
    {
        if (ConstantEngineCount <= DrawEngineCount && !ConstantEngineComplete)
        {
            SuspendOn(new CommandWaitCondition(CommandWaitKind.EngineCounter));
        }
    }

    internal void WaitForDrawEngineDifference(uint difference)
    {
        if (DrawEngineCount > ConstantEngineCount)
        {
            throw _host.Fatal($"The DE counter is ahead of the CE counter: de={DrawEngineCount} ce={ConstantEngineCount}.");
        }

        if (ConstantEngineCount - DrawEngineCount >= difference)
        {
            SuspendOn(new CommandWaitCondition(CommandWaitKind.EngineCounter));
        }
    }

    internal void WaitForRewind(bool valid)
    {
        if (!valid)
        {
            Suspend();
        }
    }

    internal uint WaitRegisterMemory32Packet(in PacketContext packet, ReadOnlySpan<uint> payload) =>
        WaitRegisterMemoryPacket(packet, payload, is64Bit: false);

    internal uint WaitRegisterMemory64Packet(in PacketContext packet, ReadOnlySpan<uint> payload) =>
        WaitRegisterMemoryPacket(packet, payload, is64Bit: true);

    private uint WaitRegisterMemoryPacket(in PacketContext packet, ReadOnlySpan<uint> payload, bool is64Bit)
    {
        var valueDwords = is64Bit ? 2u : 1u;
        var payloadDwords = 4u + (valueDwords * 2u);
        var opcode = is64Bit ? PacketOpcode.WaitRegisterMemory64 : PacketOpcode.WaitRegisterMemory;
        if (packet.Opcode != opcode || packet.CustomCode != 0 || packet.Length != payloadDwords + 1)
        {
            throw _host.Fatal($"The wait packet header is not supported: header=0x{packet.Header:X8} address=0x{packet.PacketAddress:X16}.");
        }

        var alignmentMask = is64Bit ? 0x7u : 0x3u;
        var control = payload[0];
        var address = (payload[1] & ~alignmentMask) | ((ulong)(payload[2] & 0x3FFFFu) << 32);
        var reference = ReadValue(payload, 3, is64Bit);
        var mask = ReadValue(payload, 3 + (int)valueDwords, is64Bit);
        if ((control & 0x10u) == 0)
        {
            if (is64Bit)
            {
                throw _host.Fatal($"A 64-bit register-space wait is not supported: control=0x{control:X8} address=0x{packet.PacketAddress:X16}.");
            }

            WaitOnRegister(control & 0x7u, payload[1] & 0x3FFFFu, (uint)reference, (uint)mask, WaitOperation.Decode(control, is64Bit), packet.PacketAddress);
            return payloadDwords;
        }

        WaitOnMemory(control & 0x7u, address, reference, mask, WaitOperation.Decode(control, is64Bit), is64Bit);
        return payloadDwords;
    }

    private static ulong ReadValue(ReadOnlySpan<uint> payload, int index, bool is64Bit) =>
        is64Bit ? payload[index] | ((ulong)payload[index + 1] << 32) : payload[index];

    // A register-space poll reads a register this queue's own packets write, so nothing
    // else can change it while the submission waits: an unmet compare can never be met.
    // The register address is a dword index: config space 0x2000-0x2BFF, persistent SH
    // 0x2C00-0x2FFF, context 0xA000-0xBFFF and user config 0xC000-0xFFFF.
    private void WaitOnRegister(uint compareFunction, uint register, uint reference, uint mask, uint waitOperation, ulong packetAddress)
    {
        if (!WaitOperation.IsValid(waitOperation))
        {
            throw _host.Fatal($"The wait operation is not supported: operation=0x{waitOperation:X8} register=0x{register:X5}.");
        }

        if (!WaitOperation.ShouldExecute(waitOperation, ConditionalWaitEnabled))
        {
            return;
        }

        uint value;
        bool written;
        if (register >= ShaderRegisterSpaceBase && register < ShaderRegisterSpaceBase + RegisterBankLayout.ShaderRegisterCount)
        {
            written = Registers.Shader.TryGetValue(register - ShaderRegisterSpaceBase, out value);
        }
        else if (register >= UserConfigRegisterSpaceBase && register < UserConfigRegisterSpaceBase + RegisterBankLayout.UserConfigRegisterCount)
        {
            written = Registers.UserConfig.TryGetValue(register - UserConfigRegisterSpaceBase, out value);
        }
        else
        {
            throw _host.Fatal($"A register-space wait polls a register that is not modelled: register=0x{register:X5} address=0x{packetAddress:X16}.");
        }

        if (!written)
        {
            throw _host.Fatal($"A register-space wait polls a register this queue never wrote: register=0x{register:X5} address=0x{packetAddress:X16}.");
        }

        if (!WaitOperation.TryCompare(value, reference, mask, compareFunction, out var satisfied))
        {
            throw _host.Fatal($"The wait compare function is unknown: function={compareFunction} register=0x{register:X5}.");
        }

        if (!satisfied)
        {
            throw _host.Fatal(
                $"A register-space wait can never be satisfied: register=0x{register:X5} value=0x{value:X8} " +
                $"reference=0x{reference:X8} mask=0x{mask:X8} function={compareFunction} address=0x{packetAddress:X16}.");
        }
    }

    private const uint ShaderRegisterSpaceBase = 0x2C00;
    private const uint UserConfigRegisterSpaceBase = 0xC000;

    // COND_EXEC skips the next count dwords when the dword it reads is zero. The two low
    // address bits are reserved, and the compute engine's form carries a cache policy in
    // bits 25-26 of the third dword, which does not change what is read.
    internal uint ConditionalExecutePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var payloadDwords = packet.Length - 1;
        if (payloadDwords < 4)
        {
            throw _host.Fatal($"The conditional-execute packet is too short: length={packet.Length} address=0x{packet.PacketAddress:X16}.");
        }

        var address = ((ulong)payload[1] << 32) | (payload[0] & 0xFFFF_FFFCu);
        var reserved = payload[2] & ~(0x3u << 25);
        var executeCount = payload[3] & 0x3FFFu;
        // Remaining counts this packet's header, so a skip that ends exactly at the end of
        // the buffer is payload + count == Remaining - 1.
        if (reserved != 0 || address == 0 || payloadDwords + executeCount >= packet.Remaining)
        {
            throw _host.Fatal(
                $"The conditional-execute packet is not supported: reserved=0x{payload[2]:X8} " +
                $"address=0x{address:X16} count={executeCount} remaining={packet.Remaining}.");
        }

        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (!TryReadOperandOrSuspend(address, bytes))
        {
            return payloadDwords;
        }

        if (address == TraceConditionalExecuteAddress && Interlocked.Increment(ref _tracedConditionalExecutes) <= 48)
        {
            TraceConditionalBlock(packet.PacketAddress, address, System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes),
                packet.PacketAddress + (payloadDwords + 1) * sizeof(uint), executeCount);
        }

        if (System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes) != 0)
        {
            return payloadDwords;
        }

        TraceSkippedPackets("conditional-execute", packet.PacketAddress + (payloadDwords + 1) * sizeof(uint), executeCount, (uint)address);
        return payloadDwords + executeCount;
    }

    // SHARPEMU_TRACE_COND_EXEC=<hex address> (diagnostic): the first COND_EXECs reading that dword, with its value
    // and the opcodes of the block they guard (an indirect buffer's own packets in braces).
    private static readonly ulong TraceConditionalExecuteAddress =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_TRACE_COND_EXEC"), System.Globalization.NumberStyles.HexNumber, null,
            out var conditionalAddress) ? conditionalAddress : 0;
    private static int _tracedConditionalExecutes;

    private void TraceConditionalBlock(ulong packetAddress, ulong address, uint value, ulong blockAddress, uint dwords)
    {
        string Describe(ulong at, uint count, int depth)
        {
            var buffer = new byte[Math.Min(count, 4096u) * sizeof(uint)];
            if (buffer.Length == 0 || !_host.TryReadGuest(at, buffer)) return "unreadable";
            var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(buffer);
            var text = new System.Text.StringBuilder();
            for (var offset = 0; offset < words.Length;)
            {
                var header = words[offset];
                if ((header >> 30) != 3)
                {
                    text.Append($" stop:0x{header:X8}");
                    break;
                }

                var opcode = (header >> 8) & 0xFF;
                var body = (int)((header >> 16) & 0x3FFF) + 1;
                text.Append(opcode == PacketOpcode.Nop ? $" 10:{PacketHeader.CustomCode(header):X2}/{body}" : $" {opcode:X2}");
                if (opcode == PacketOpcode.IndirectBuffer && body == 3 && offset + 3 < words.Length && depth == 0)
                {
                    var target = words[offset + 1] | ((ulong)(words[offset + 2] & 0xFFFF) << 32);
                    text.Append($"{{{Describe(target, words[offset + 3] & 0xFFFFF, 1)} }}");
                }

                offset += body + 1;
            }

            return text.ToString();
        }

        var raw = new byte[Math.Min(dwords, 32u) * sizeof(uint)];
        var rawText = _host.TryReadGuest(blockAddress, raw)
            ? string.Join(" ", System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(raw).ToArray().Select(static word => $"{word:X8}"))
            : "unreadable";
        Console.Error.WriteLine(
            $"[GPU][COND-EXEC] q={QueueId} packet=0x{packetAddress:X} reads=0x{address:X} value=0x{value:X} dwords={dwords}:{Describe(blockAddress, dwords, 0)} raw={rawText}");
    }

    internal uint SetPredicationPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var payloadDwords = packet.Length - 1;
        if (payloadDwords < 2)
        {
            throw _host.Fatal($"The predication packet is too short: length={packet.Length} address=0x{packet.PacketAddress:X16}.");
        }

        const uint flagsMask = 0x8007_1100u;
        uint flags;
        ulong address;
        if (payloadDwords >= 3 && (payload[0] & ~flagsMask) == 0 && payload[2] <= 0xFFFFu)
        {
            flags = payload[0];
            address = ((ulong)payload[2] << 32) | (payload[1] & 0xFFFF_FFF0u);
        }
        else
        {
            flags = payload[1];
            address = (payload[0] & 0xFFFF_FFF0u) | ((ulong)(payload[1] & 0xFFu) << 32);
        }

        SetPredication((flags >> 8) & 0x1u, (flags >> 16) & 0x7u, (flags >> 12) & 0x1u, address, (flags & (1u << 31)) != 0);
        return payloadDwords;
    }

    internal uint RewindPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Header != PacketHeader.Make(2, PacketOpcode.Rewind) || (payload[0] & ~RewindKnownBits) != 0)
        {
            throw _host.Fatal($"The rewind packet is not supported: header=0x{packet.Header:X8} body=0x{payload[0]:X8} address=0x{packet.PacketAddress:X16}.");
        }

        WaitForRewind((payload[0] & RewindValidBit) != 0);
        return 1;
    }

    internal uint IncrementCeCounterPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireExactPacket(packet, PacketHeader.Make(2, PacketOpcode.IncrementCeCounter), payload[0], 1);
        ConstantEngineCount++;
        return 1;
    }

    internal uint IncrementDeCounterPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireExactPacket(packet, PacketHeader.Make(2, PacketOpcode.IncrementDeCounter), payload[0], 0);
        DrawEngineCount++;
        return 1;
    }

    internal uint WaitOnCeCounterPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireExactPacket(packet, PacketHeader.Make(2, PacketOpcode.WaitOnCeCounter), payload[0], 1);
        WaitForConstantEngine();
        return 1;
    }

    internal uint WaitOnDeCounterDiffPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Header != PacketHeader.Make(2, PacketOpcode.WaitOnDeCounterDiff))
        {
            throw _host.Fatal($"The DE counter wait header is not supported: header=0x{packet.Header:X8}.");
        }

        WaitForDrawEngineDifference(payload[0]);
        return 1;
    }

    private void RequireExactPacket(in PacketContext packet, uint header, uint body, uint expectedBody)
    {
        if (packet.Header != header || body != expectedBody)
        {
            throw _host.Fatal($"The packet is not supported: header=0x{packet.Header:X8} body=0x{body:X8} address=0x{packet.PacketAddress:X16}.");
        }
    }
}
