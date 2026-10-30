// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Resources;

public sealed partial class ResourceMaterializationCache
{
    // A stale entry evaluated again slot by slot instead of walked again. Every flattened slot is
    // one dword loaded at an address computed from user data, constants and earlier slots, and the
    // host evaluates nothing else but functions of user data and slots: descriptors, branch
    // conditions, device ranges. So when every slot those read holds the entry's word, they come
    // out as the entry's, and the table is the slots' words for this draw. A slot's address is
    // computed for this draw from the slots before it; its word is taken from where the entry's
    // range around it now lies, read in bulk. A per-frame constant ring behind a pointer in a
    // descriptor table moves the pointer and every slot read through it, while the V#s the ring
    // holds keep their words. A descriptor slot whose word differs is a real descriptor change,
    // and the draw is materialized again.
    private sealed class SlotPlan
    {
        public required ScalarValueKind[] Kinds { get; init; }
        // The handle dwords each slot's address uses: base low and high, then records and the
        // fourth V# dword for a buffer read.
        public required ScalarValue[][] Handles { get; init; }
        public required long[] Immediates { get; init; }
        public required uint[] Offsets { get; init; }
        public required int[] FlatOffsets { get; init; }
        // Slots the host reads outside the table: descriptors, branch conditions, device ranges.
        public required bool[] Evaluated { get; init; }

        public int Count => Kinds.Length;

        // Null when some value the materialization evaluates is not a function of user data and
        // slots, or some slot's address is not computable from the slots before it.
        public static SlotPlan? Analyze(ShaderResourcePlan plan)
        {
            var count = plan.TableReads.Count;
            if (count == 0 || plan.IndirectImages.Count != 0 || plan.BufferCandidateTables.Count != 0 ||
                plan.DeviceDescriptorTables.Count != 0 || plan.RequiresSpecializationMemory)
                return null;

            var kinds = new ScalarValueKind[count];
            var handles = new ScalarValue[count][];
            var immediates = new long[count];
            var offsets = new uint[count];
            var flatOffsets = new int[count];
            for (var slot = 0; slot < count; slot++)
            {
                var read = plan.TableReads[slot];
                var value = read.Value;
                if (value.Kind is not (ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord) ||
                    value.Operands.Length < 2 || !value.Operands[1].IsConstant ||
                    (uint)value.MemoryIndex >= (uint)plan.Memory.Count || read.FlatOffset >= (uint)count)
                    return null;
                var handle = value.Operands[0];
                var used = value.Kind == ScalarValueKind.ScalarBufferWord ? 4 : 2;
                if (handle.Operands.Length < 2 || (value.Kind == ScalarValueKind.ScalarBufferWord && handle.Operands.Length != 4))
                    return null;
                handles[slot] = handle.Operands[..used];
                foreach (var operand in handles[slot])
                    if (!Computable(operand, slot, 0))
                        return null;
                kinds[slot] = value.Kind;
                // As the evaluator reads them: the immediate through int, the offset through uint.
                immediates[slot] = (int)plan.Memory[value.MemoryIndex].Offset;
                offsets[slot] = (uint)value.Operands[1].Payload;
                flatOffsets[slot] = (int)read.FlatOffset;
            }

            var evaluated = new bool[count];
            var pending = new Stack<ScalarValue>();
            foreach (var source in plan.MaterializationSources)
            {
                if (source >= plan.DescriptorSources.Count)
                    return null;
                foreach (var dword in plan.DescriptorSources[(int)source].Dwords)
                    pending.Push(dword);
            }

            foreach (var branch in plan.ResourceBranches)
                if (branch.Condition is { } condition)
                    pending.Push(condition);
            foreach (var range in plan.DeviceAddressRanges)
            {
                if (!range.Plannable)
                    continue;
                pending.Push(range.BaseLow);
                pending.Push(range.BaseHigh);
            }

            var visited = new HashSet<ScalarValue>();
            while (pending.TryPop(out var value))
            {
                if (value.IsConstant || !visited.Add(value))
                    continue;
                switch (value.Kind)
                {
                    case ScalarValueKind.ResourceTableWord:
                        if (value.Payload >= (ulong)count)
                            return null;
                        evaluated[(int)value.Payload] = true;
                        continue;
                    case ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord:
                        // A load the table does not hold reads memory no slot accounts for.
                        return null;
                    case ScalarValueKind.Phi:
                        // The evaluator takes a phi's invariant value, or fails on every draw alike.
                        if (plan.Graph.ResolveInvariantPhi(value) is { } invariant && !ReferenceEquals(invariant, value))
                            pending.Push(invariant);
                        continue;
                }

                foreach (var operand in value.Operands)
                    pending.Push(operand);
            }

            return new SlotPlan
            {
                Kinds = kinds,
                Handles = handles,
                Immediates = immediates,
                Offsets = offsets,
                FlatOffsets = flatOffsets,
                Evaluated = evaluated,
            };
        }

        private static bool Computable(ScalarValue value, int slot, int depth)
        {
            if (depth > 32)
                return false;
            switch (value.Kind)
            {
                case ScalarValueKind.Constant or ScalarValueKind.UserData or ScalarValueKind.ShaderBase or ScalarValueKind.MemoryAperture:
                    return true;
                case ScalarValueKind.ResourceTableWord:
                    return value.Payload < (ulong)slot;
                case ScalarValueKind.Select when value.Operands.Length == 3:
                    break;
                case ScalarValueKind.Operation when RuntimeValueValidator.IsUniformOperation(value.Operation):
                    break;
                default:
                    return false;
            }

            foreach (var operand in value.Operands)
                if (!Computable(operand, slot, depth + 1))
                    return false;
            return true;
        }

        // The evaluator's EvaluateWide for the kinds Computable admits.
        private static bool Compute(ScalarValue value, IReadOnlyList<uint> userData, uint userDataBase, ulong shaderBase,
            uint[] slots, out ulong result)
        {
            result = 0;
            switch (value.Kind)
            {
                case ScalarValueKind.Constant:
                    result = value.Payload;
                    return true;
                case ScalarValueKind.UserData:
                {
                    var register = value.UserDataRegister;
                    if (register < userDataBase || register - userDataBase >= (uint)userData.Count)
                        return false;
                    result = userData[(int)(register - userDataBase)];
                    return true;
                }
                case ScalarValueKind.ShaderBase:
                    result = shaderBase;
                    return true;
                case ScalarValueKind.MemoryAperture:
                    result = Gen5InlineConstants.DecodeAperture64((uint)value.Payload) >> 32;
                    return true;
                case ScalarValueKind.ResourceTableWord:
                    result = slots[(int)value.Payload];
                    return true;
                case ScalarValueKind.Select:
                {
                    if (!Compute(value.Operands[0], userData, userDataBase, shaderBase, slots, out var condition) ||
                        !Compute(value.Operands[1], userData, userDataBase, shaderBase, slots, out var whenTrue) ||
                        !Compute(value.Operands[2], userData, userDataBase, shaderBase, slots, out var whenFalse))
                        return false;
                    result = condition != 0 ? whenTrue : whenFalse;
                    return true;
                }
                default:
                {
                    Span<ulong> operands = stackalloc ulong[value.Operands.Length];
                    for (var index = 0; index < operands.Length; index++)
                        if (!Compute(value.Operands[index], userData, userDataBase, shaderBase, slots, out operands[index]))
                            return false;
                    return ScalarOperationSemantics.TryEvaluate(value.Operation, operands, out result);
                }
            }
        }

        // The slot's address for a draw with this user data, given the words of the slots before
        // it, as RuntimeValueEvaluator.EvaluateRawRead resolves it. Zero: an unbound V# reads 0.
        public bool TryAddress(int slot, IReadOnlyList<uint> userData, uint userDataBase, ulong shaderBase, uint[] slots,
            out ulong address, out bool zero)
        {
            address = 0;
            zero = false;
            var handle = Handles[slot];
            ulong records = 0;
            if (!Compute(handle[0], userData, userDataBase, shaderBase, slots, out var low) ||
                !Compute(handle[1], userData, userDataBase, shaderBase, slots, out var high) ||
                (handle.Length == 4 &&
                 (!Compute(handle[2], userData, userDataBase, shaderBase, slots, out records) ||
                  !Compute(handle[3], userData, userDataBase, shaderBase, slots, out _))))
                return false;

            var baseAddress = ((high << 32) | (uint)low) & 0x0000_FFFF_FFFF_FFFFul;
            switch (RuntimeValueEvaluator.ResolveRawAddress(Kinds[slot], baseAddress, high, records, Immediates[slot], Offsets[slot],
                        out address))
            {
                case RuntimeValueEvaluator.RawAddress.Read:
                    return true;
                case RuntimeValueEvaluator.RawAddress.Zero:
                    zero = true;
                    return true;
                default:
                    return false;
            }
        }
    }

    // Where each slot of an entry was read: its address, the byte offset of its word in the entry's
    // bytes and the range holding it (both -1 for a slot that read zero without a read), and the
    // word. Rebuilding the entry's own table from these words is what admits the model.
    private sealed class SlotModel
    {
        public required ulong[] Addresses { get; init; }
        public required int[] Offsets { get; init; }
        public required int[] Ranges { get; init; }
        public required uint[] Values { get; init; }
    }

    private static readonly SlotModel Unmodelled = new() { Addresses = [], Offsets = [], Ranges = [], Values = [] };
    private static readonly ConditionalWeakTable<ShaderResourcePlan, StrongBox<SlotPlan?>> SlotPlans = new();

    private static SlotPlan? SlotPlanOf(ShaderResourcePlan plan) =>
        SlotPlans.GetValue(plan, static plan => new StrongBox<SlotPlan?>(SlotPlan.Analyze(plan))).Value;

    private SlotModel? ModelOf(Entry entry, ShaderResourcePlan plan, SlotPlan slots)
    {
        if (entry.Slots is null)
        {
            entry.Slots = BuildModel(entry, plan, slots, entry.SlotsHint) ?? Unmodelled;
            entry.SlotsHint = null;
        }

        var model = entry.Slots;
        return ReferenceEquals(model, Unmodelled) ? null : model;
    }

    private ulong[] _modelAddresses = [];
    private int[] _modelOffsets = [];
    private int[] _modelRanges = [];
    private uint[] _modelValues = [];

    // Built in scratch: an entry rebuilt from another (hint) for a moved ring usually keeps where
    // its slots lie in its bytes and often their words, and those arrays are shared.
    private SlotModel? BuildModel(Entry entry, ShaderResourcePlan plan, SlotPlan slots, SlotModel? hint)
    {
        var table = entry.Snapshot.FlattenedResourceTable;
        if (!entry.TableRefreshable)
            return null;
        var count = slots.Count;
        if (_modelAddresses.Length < count)
        {
            var capacity = Math.Max(count, _modelAddresses.Length * 2);
            _modelAddresses = new ulong[capacity];
            _modelOffsets = new int[capacity];
            _modelRanges = new int[capacity];
            _modelValues = new uint[capacity];
        }

        var addresses = _modelAddresses.AsSpan(0, count);
        var offsets = _modelOffsets.AsSpan(0, count);
        var ranges = _modelRanges.AsSpan(0, count);
        // TryAddress reads the words of the slots before the one it computes.
        var valueArray = _modelValues;
        var values = valueArray.AsSpan(0, count);
        addresses.Clear();
        values.Clear();
        for (var slot = 0; slot < count; slot++)
        {
            if (!slots.TryAddress(slot, entry.UserData, plan.UserDataBase, entry.ShaderBase, valueArray, out var address, out var zero))
                return null;
            if (zero)
            {
                offsets[slot] = ranges[slot] = -1;
            }
            else
            {
                if (!TryFindRange(entry, address, out var range, out var offset))
                    return null;
                addresses[slot] = address;
                offsets[slot] = offset;
                ranges[slot] = range;
                values[slot] = BinaryPrimitives.ReadUInt32LittleEndian(entry.Bytes.AsSpan(offset));
            }

            if (slots.FlatOffsets[slot] >= table.Length || table[slots.FlatOffsets[slot]] != values[slot])
                return null;
        }

        return new SlotModel
        {
            Addresses = hint is not null && addresses.SequenceEqual(hint.Addresses) ? hint.Addresses : addresses.ToArray(),
            Offsets = hint is not null && offsets.SequenceEqual(hint.Offsets) ? hint.Offsets : offsets.ToArray(),
            Ranges = hint is not null && ranges.SequenceEqual(hint.Ranges) ? hint.Ranges : ranges.ToArray(),
            Values = hint is not null && values.SequenceEqual(hint.Values) ? hint.Values : values.ToArray(),
        };
    }

    private uint[] _slotValues = [];
    private ulong[] _slotAddresses = [];
    private byte[] _slotSame = [];
    private byte[] _slotMoved = [];
    private ulong[] _slotDeltas = [];
    private byte[] _slotRangeState = [];

    private const byte SameRead = 1, SameFailed = 2, MovedRead = 4, MovedFailed = 8;
    private const ulong ZeroSlot = ulong.MaxValue;

    private enum SlotRefusal
    {
        Plan,
        Model,
        UserData,
        Address,
        Unreadable,
        Descriptor,
        Count,
    }

    private static readonly string[] SlotRefusalNames = ["plan", "model", "user_data", "address", "unreadable", "descriptor"];
    private static readonly long[] _slotRefusals = new long[(int)SlotRefusal.Count];
    private static long _totalSlotRefreshes;
    public long SlotRefreshes { get; private set; }

    private static void RefuseSlots(SlotRefusal reason) => Interlocked.Increment(ref _slotRefusals[(int)reason]);

    // The A/B phase this cache last saw, and its counters and thread CPU time when it began.
    private long _slotPhase = -1;
    private long _phaseStart;
    private long _phaseCpu;
    private (long Hits, long Misses, long Refreshes, long SlotRefreshes) _phaseCounts;

    // Reports the phase that ended: the materializing thread's CPU time and this cache's counts.
    private void ReportSlotPhase(long phase)
    {
        var now = Stopwatch.GetTimestamp();
        var cpu = SlotRefreshRuntime.ThreadCpuNanoseconds();
        if (_slotPhase >= 0)
        {
            Console.Error.WriteLine(
                FormattableString.Invariant(
                    $"[PERF][SLOT_REFRESH_AB] phase={_slotPhase} arm={(SlotRefreshRuntime.Enabled(_slotPhase) ? "on" : "off")} ") +
                FormattableString.Invariant(
                    $"seconds={(now - _phaseStart) / (double)Stopwatch.Frequency:F2} thread=\"{Thread.CurrentThread.Name}\" ") +
                FormattableString.Invariant(
                    $"thread_cpu_ms={(cpu - _phaseCpu) / 1e6:F1} hits={Hits - _phaseCounts.Hits} misses={Misses - _phaseCounts.Misses} ") +
                FormattableString.Invariant(
                    $"refreshes={TableRefreshes - _phaseCounts.Refreshes} slot_refreshes={SlotRefreshes - _phaseCounts.SlotRefreshes}"));
        }

        _slotPhase = phase;
        _phaseStart = now;
        _phaseCpu = cpu;
        _phaseCounts = (Hits, Misses, TableRefreshes, SlotRefreshes);
    }

    private static string TakeSlotReport() =>
        FormattableString.Invariant($" slot_refreshes={Interlocked.Exchange(ref _totalSlotRefreshes, 0)} slot_refused=") +
        string.Join(",", SlotRefusalNames.Select((name, index) =>
            FormattableString.Invariant($"{name}:{Interlocked.Exchange(ref _slotRefusals[index], 0)}")));

    // The entry evaluated again for this draw, or false with nothing stored. The new entry records
    // each slot's word where this draw read it, which is every word the materialization depends on.
    private bool TryRefreshSlots(Entry entry, ShaderResourcePlan plan, SlotPlan slots, ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader, AddressRebase rebase, in RejectionSample rejection, out Entry refreshed,
        out bool descriptorChanged)
    {
        refreshed = null!;
        descriptorChanged = false;
        if (ModelOf(entry, plan, slots) is not { } model)
        {
            RefuseSlots(SlotRefusal.Model);
            return false;
        }

        var count = slots.Count;
        if (_slotValues.Length < count)
        {
            _slotValues = new uint[Math.Max(count, _slotValues.Length * 2)];
            _slotAddresses = new ulong[_slotValues.Length];
        }

        if (_slotSame.Length < entry.Bytes.Length)
        {
            _slotSame = new byte[Math.Max(entry.Bytes.Length, _slotSame.Length * 2)];
            _slotMoved = new byte[_slotSame.Length];
        }

        var rangeCount = entry.RangeAddresses.Length;
        if (_slotRangeState.Length < rangeCount)
        {
            _slotRangeState = new byte[Math.Max(rangeCount, _slotRangeState.Length * 2)];
            _slotDeltas = new ulong[_slotRangeState.Length];
        }

        _slotRangeState.AsSpan(0, rangeCount).Clear();
        var values = _slotValues;
        var addresses = _slotAddresses;
        Span<byte> single = stackalloc byte[sizeof(uint)];
        for (var slot = 0; slot < count; slot++)
        {
            if (!slots.TryAddress(slot, inputs.UserData, plan.UserDataBase, inputs.ShaderBase, values, out var address, out var zero))
            {
                RefuseSlots(SlotRefusal.Address);
                return false;
            }

            uint word = 0;
            if (zero)
            {
                address = ZeroSlot;
            }
            else if (slots.Evaluated[slot] && model.Ranges[slot] is var sampledRange &&
                rejection.RefutesWord(address, sampledRange < 0 || entry.RangeClean[sampledRange], model.Values[slot]))
            {
                // Validation already read this descriptor word changed: no read can refresh it.
                RefuseSlots(SlotRefusal.Descriptor);
                descriptorChanged = true;
                return false;
            }
            else if (!TryReadSlot(entry, model, slot, address, residentReader, single, out word))
            {
                RefuseSlots(SlotRefusal.Unreadable);
                return false;
            }

            if (slots.Evaluated[slot] && word != model.Values[slot])
            {
                RefuseSlots(SlotRefusal.Descriptor);
                descriptorChanged = true;
                return false;
            }

            values[slot] = word;
            addresses[slot] = address;
        }

        // A ring that moved with the same words in it (constants the title writes again every frame)
        // gives the entry's table, and with the entry's user data the entry's snapshot.
        var previous = entry.Snapshot;
        var table = previous.FlattenedResourceTable;
        for (var slot = 0; slot < count; slot++)
        {
            if (table[slots.FlatOffsets[slot]] == values[slot])
                continue;
            if (ReferenceEquals(table, previous.FlattenedResourceTable))
                table = (uint[])table.Clone();
            table[slots.FlatOffsets[slot]] = values[slot];
        }

        var userData = CurrentUserData(previous.UserData, inputs.UserData);
        var snapshot = ReferenceEquals(table, previous.FlattenedResourceTable) && ReferenceEquals(userData, previous.UserData)
            ? previous
            : new ResourceSnapshot
            {
                Buffers = previous.Buffers,
                Images = previous.Images,
                Samplers = previous.Samplers,
                FlattenedResourceTable = table,
                UserData = userData,
                DeviceAddressRanges = previous.DeviceAddressRanges,
                BindlessImages = previous.BindlessImages,
            };

        var recorder = TakeRecorder();
        try
        {
            for (var slot = 0; slot < count; slot++)
            {
                if (addresses[slot] == ZeroSlot)
                    continue;
                var range = model.Ranges[slot];
                recorder.Record(addresses[slot], values[slot], range < 0 || entry.RangeClean[range],
                    range >= 0 && entry.WordTableOnly[model.Offsets[slot] / sizeof(uint)]);
            }

            refreshed = recorder.Build(plan, inputs, snapshot, entry.Specialization, rebase.MemoryBases, entry.UsedUserData, like: entry);
        }
        finally
        {
            ReturnRecorder(recorder);
        }

        return true;
    }

    // The slot's word at its address for this draw. At the entry's own address it comes from the
    // entry's range read where it lies; at a moved one, from that range read moved by as much. A
    // range read moved by another amount, or one that does not read whole, leaves the word to a
    // read of its own.
    private bool TryReadSlot(Entry entry, SlotModel model, int slot, ulong address, ResidentGuestBytesReader residentReader,
        Span<byte> single, out uint word)
    {
        word = 0;
        var range = model.Ranges[slot];
        var clean = range < 0 || entry.RangeClean[range];
        if (range >= 0)
        {
            var state = _slotRangeState[range];
            var offset = entry.RangeOffsets[range];
            var length = entry.RangeLengths[range];
            if (address == model.Addresses[slot])
            {
                if ((state & (SameRead | SameFailed)) == 0)
                {
                    state |= residentReader(entry.RangeAddresses[range], _slotSame.AsSpan(offset, length), clean) ? SameRead : SameFailed;
                    _slotRangeState[range] = state;
                }

                if ((state & SameRead) != 0)
                {
                    word = BinaryPrimitives.ReadUInt32LittleEndian(_slotSame.AsSpan(model.Offsets[slot]));
                    return true;
                }
            }
            else
            {
                var delta = address - model.Addresses[slot];
                if ((state & (MovedRead | MovedFailed)) == 0)
                {
                    state |= residentReader(entry.RangeAddresses[range] + delta, _slotMoved.AsSpan(offset, length), clean)
                        ? MovedRead
                        : MovedFailed;
                    _slotRangeState[range] = state;
                    _slotDeltas[range] = delta;
                }

                if ((state & MovedRead) != 0 && _slotDeltas[range] == delta)
                {
                    word = BinaryPrimitives.ReadUInt32LittleEndian(_slotMoved.AsSpan(model.Offsets[slot]));
                    return true;
                }
            }
        }

        if (!residentReader(address, single, clean))
            return false;
        word = BinaryPrimitives.ReadUInt32LittleEndian(single);
        return true;
    }

    // The range holding a recorded dword and the dword's byte offset in the entry's bytes.
    private static bool TryFindRange(Entry entry, ulong address, out int range, out int offset)
    {
        offset = 0;
        range = -1;
        var addresses = entry.RangeAddresses;
        int low = 0, high = addresses.Length - 1;
        while (low <= high)
        {
            var middle = (low + high) >>> 1;
            if (addresses[middle] <= address)
            {
                range = middle;
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }

        if (range < 0)
            return false;
        var delta = address - addresses[range];
        if (delta % sizeof(uint) != 0 || delta >= (ulong)entry.RangeLengths[range])
            return false;
        offset = entry.RangeOffsets[range] + (int)delta;
        return true;
    }
}

// SHARPEMU_RESOURCE_SLOT_REFRESH=0 turns the slot refresh off. SHARPEMU_RESOURCE_SLOT_REFRESH_AB_S=N
// (diagnostic) alternates it on and off every N seconds from the first materialization, so one run
// carries interleaved A/B phases over the same scenes; even phases refresh.
internal static class SlotRefreshRuntime
{
    private static readonly bool Disabled = Environment.GetEnvironmentVariable("SHARPEMU_RESOURCE_SLOT_REFRESH") == "0";
    private static readonly long AbPeriodTicks =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_RESOURCE_SLOT_REFRESH_AB_S"), out var seconds) && seconds > 0
            ? seconds * Stopwatch.Frequency
            : 0;

    private static long _abEpoch;

    public static bool Alternating => AbPeriodTicks != 0;

    // The phase now, -1 without A/B.
    public static long Phase()
    {
        if (AbPeriodTicks == 0)
            return -1;
        var now = Stopwatch.GetTimestamp();
        var epoch = Volatile.Read(ref _abEpoch);
        if (epoch == 0)
        {
            var prior = Interlocked.CompareExchange(ref _abEpoch, now, 0);
            epoch = prior != 0 ? prior : now;
            if (prior == 0)
                Console.Error.WriteLine(FormattableString.Invariant(
                    $"[PERF][SLOT_REFRESH_AB] epoch period_s={AbPeriodTicks / Stopwatch.Frequency} (even phases refresh)"));
        }

        return (now - epoch) / AbPeriodTicks;
    }

    public static bool Enabled(long phase) => !Disabled && (phase < 0 || phase % 2 == 0);

    // CPU time of the calling thread, or -1 where it is not known. Under Rosetta the thread CPU-time
    // clock counts 24 MHz ticks as nanoseconds; the process info call reports real nanoseconds.
    public static long ThreadCpuNanoseconds()
    {
        if (!OperatingSystem.IsMacOS() || PthreadThreadId(0, out var thread) != 0)
            return -1;
        var info = new ulong[ThreadInfoBytes / sizeof(ulong)];
        return ProcessInfo(Environment.ProcessId, ThreadId64Info, thread, info, ThreadInfoBytes) == ThreadInfoBytes
            ? (long)(info[0] + info[1])
            : -1;
    }

    // PROC_PIDTHREADID64INFO fills a proc_threadinfo: user and system time first, 112 bytes in all.
    private const int ThreadId64Info = 15;
    private const int ThreadInfoBytes = 112;

    [DllImport("libc", EntryPoint = "pthread_threadid_np")]
    private static extern int PthreadThreadId(nint thread, out ulong id);

    [DllImport("libc", EntryPoint = "proc_pidinfo")]
    private static extern int ProcessInfo(int pid, int flavor, ulong argument, [Out] ulong[] buffer, int size);
}
