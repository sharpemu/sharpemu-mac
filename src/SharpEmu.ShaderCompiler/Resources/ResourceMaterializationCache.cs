// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;

namespace SharpEmu.ShaderCompiler.Resources;

// Copies guest bytes only when they are already resident on the CPU side. It never
// synchronizes: a range the GPU may still own, or that a clean read would refuse,
// returns false so the caller falls back to a full materialization.
public delegate bool ResidentGuestBytesReader(ulong address, Span<byte> destination, bool clean);

// Compares guest bytes with recorded ones where they lie: range i is lengths[i] bytes at
// addresses[i], to equal recorded[offsets[i]..]. True only when every range was read
// through the same checks ResidentGuestBytesReader makes and every byte is equal. False
// says nothing about the bytes: one differs, or the host cannot compare in place, and the
// caller then reads the ranges to find out which.
public delegate bool ResidentGuestBytesComparer(
    ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets, ReadOnlySpan<byte> recorded);

// Materialization is a pure function of the plan, the user data words the evaluator
// reads, the shader base, the compute state and the guest words it reads. This cache
// records exactly those words and reuses the result while every one of them is
// unchanged, so a draw that re-binds the same resources skips the descriptor walk. A user
// data word the evaluator never read (a per-draw constant for the shader's own ALU, say)
// does not distinguish two draws; the snapshot a hit returns carries the current user
// data. Any failed read, any GPU-owned range and any byte difference make the entry
// miss; nothing is guessed.
public sealed partial class ResourceMaterializationCache
{
    // Two generations approximate LRU: a full young generation becomes the old one,
    // and an old entry that is used again is promoted back. A generation is full at
    // _generationCapacity keys or at _generationByteBudget estimated retained bytes,
    // whichever comes first: a count alone let entries with large bindless tables or long
    // read lists hold hundreds of MB. Dropping the old generation only costs rebuilds.
    // A scan of distinct textures can miss forever: retaining a full validation entry
    // for every draw then costs more than the walk. Admission is only a retention hint;
    // it can decline to store a successful walk, never authorize a hit or skip a read.
    private static readonly Action<bool> NoteTablePhase = inTable => ReadingTable = inTable;
    private const int ColdScanThreshold = 32;
    private static readonly bool ColdAdmissionEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_RESOURCE_CACHE_ADMISSION") != "0";
    private int _consecutiveMisses;
    private bool _cacheSaturated;
    private uint[]? _admissionTags;
    public long ColdBypasses { get; private set; }

    private bool DeferColdEntry(ResourceSnapshot snapshot)
    {
        if (!ColdAdmissionEnabled || !_cacheSaturated) return false;
        var hash = new HashCode();
        foreach (var image in snapshot.Images)
        {
            hash.Add(image.Length);
            foreach (var word in image) hash.Add(word);
        }
        foreach (var sampler in snapshot.Samplers)
        {
            hash.Add(sampler.Length);
            foreach (var word in sampler) hash.Add(word);
        }
        foreach (var buffer in snapshot.Buffers)
        {
            hash.Add(buffer.Length);
            // Memory-base learning may patch the V# address. Do not let admission
            // starve those proofs when the buffer moves on every draw.
            for (var index = 1; index < buffer.Length; index++)
                hash.Add(index == 1 ? buffer[index] & 0xFFFF0000u : buffer[index]);
        }
        var tag = (uint)hash.ToHashCode();
        if (tag == 0) tag = 1;
        var tags = _admissionTags ??= new uint[512];
        var slot = tag & (uint)(tags.Length - 1);
        var repeated = tags[slot] == tag;
        tags[slot] = tag;
        return _consecutiveMisses >= ColdScanThreshold && !repeated;
    }

    private readonly int _generationCapacity;
    private readonly long _generationByteBudget;
    private Dictionary<ulong, Entry> _young = new();
    private Dictionary<ulong, Entry> _old = new();
    private long _youngBytes;
    private long _oldBytes;
    // The rebases' recent entries (AddressRebase.Learn) hold entries the generations may have
    // dropped already; past this estimate every rebase forgets them, which only delays learning.
    private readonly long _recentByteBudget;
    private long _recentBytes;
    private byte[] _scratch = new byte[256];
    private ReadRecorder? _spareRecorder;
    private ReadRecorder? _secondSpareRecorder;
    // The user data words each (plan, base, compute state) has read so far. They only index the
    // entries; Entry.Matches is what decides a hit.
    private readonly Dictionary<ulong, ulong[]> _learnedUserData = new();
    private readonly Dictionary<ulong, AddressRebase> _rebases = new();
    private const int MaxDrawKeys = 16384;

    private AddressRebase RebaseOf(ulong drawKey, ShaderResourcePlan plan)
    {
        if (!_rebases.TryGetValue(drawKey, out var rebase))
        {
            // Bounded: every replanned program or new shader base adds a key, and each rebase keeps
            // up to RecentCapacity entries alive. Dropping them is what a draw key seen for the
            // first time gets: nothing rebased and nothing learned, so entries are matched on
            // every word again and missing ones are rebuilt.
            if (_rebases.Count >= MaxDrawKeys)
            {
                _rebases.Clear();
                _learnedUserData.Clear();
                _recentBytes = 0;
            }

            _rebases.Add(drawKey, rebase = new AddressRebase(plan));
        }

        return rebase;
    }

    // Titles allocate per-draw constants linearly, so a buffer's base address changes on every
    // draw while everything else the draw binds stays the same; keyed on that address, the cache
    // misses nearly every draw. A user data word the plan uses verbatim as a buffer's base dword
    // is a candidate. It is rebased only once misses have shown it repeatedly that two draws
    // differing in nothing but candidate words materialize identically, except for those base
    // dwords, which equal the words: the same descriptors, table, ranges, specialization and
    // recorded guest bytes. One counterexample rejects the word for good. A rebased word leaves
    // the key and the match, and a hit patches the base dwords from the draw's own user data.
    private sealed class AddressRebase
    {
        private const int Confirmations = 3;

        // A word that is a base address. A buffer candidate is the base dword of a buffer descriptor
        // (Buffer >= 0): the hit patches that dword, and reads inside the descriptor's range move.
        // A raw candidate is the low dword of the handle of scalar loads; its reads lie at constant
        // byte offsets [MinOffset, MaxOffset] from the base whose high dword is user data word High
        // (or HighConstant when High < 0).
        private readonly record struct Candidate(int UserData, int Buffer, int High = -1, uint HighConstant = 0,
            long MinOffset = 0, long MaxOffset = 0);

        private const ulong RawAddressMask = 0x0000_FFFF_FFFF_FFFFul;

        private readonly Candidate[] _candidates;
        private readonly ulong[] _candidateMask;
        private ulong[] _rejected = [];
        private readonly Dictionary<int, int> _confirmed = new();
        private const int RecentCapacity = 512;
        private readonly Dictionary<ulong, Entry> _recent = new();
        // The latest entry per structure and content print (EntryPrint): the same object drawn
        // before, which _recent misses when other objects are drawn in between.
        private readonly Dictionary<ulong, Entry> _recentByPrint = new();
        private long _recentByPrintBytes;

        // The estimated bytes the recent entries hold (Entry.EstimatedBytes).
        public long RecentBytes { get; private set; }

        public void ForgetRecent()
        {
            _recent.Clear();
            _recentByPrint.Clear();
            _recentByPrintBytes = 0;
            RecentBytes = 0;
        }

        // Keeps built as the latest entry of its structure and print; the one it replaces, if any.
        private Entry? RememberByPrint(ulong structure, Entry built)
        {
            if (!_hasPrint || EntryPrint(built) is not { } print)
                return null;
            var key = MixPrint(structure, print);
            Entry? previous;
            if (_recentByPrint.Count >= RecentCapacity)
            {
                _recentByPrint.TryGetValue(key, out previous);
                _recentByPrint.Clear();
                RecentBytes -= _recentByPrintBytes;
                _recentByPrintBytes = 0;
                _recentByPrint[key] = built;
            }
            else
            {
                // One lookup: a miss whose print is new, the usual case, pays for one hash probe.
                ref var slot = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_recentByPrint, key, out _);
                previous = slot;
                slot = built;
                if (previous is not null)
                {
                    RecentBytes -= previous.EstimatedBytes;
                    _recentByPrintBytes -= previous.EstimatedBytes;
                }
            }

            RecentBytes += built.EstimatedBytes;
            _recentByPrintBytes += built.EstimatedBytes;
            return previous;
        }

        // Some candidate word the entries read differs: they are drawn through other pointers.
        private bool CandidateWordsDiffer(Entry left, Entry right)
        {
            foreach (var candidate in _candidates)
            {
                var index = candidate.UserData;
                if (index < left.UserData.Length && index < right.UserData.Length && IsUsed(left.UsedUserData, index) &&
                    left.UserData[index] != right.UserData[index])
                    return true;
            }

            return false;
        }

        private ulong StructureOf(Entry entry)
        {
            var hash = new HashCode();
            hash.Add(entry.UserData.Length);
            for (var index = 0; index < entry.UserData.Length; index++)
            {
                if (!IsUsed(entry.UsedUserData, index) || IsUsed(_candidateMask, index))
                    continue;
                hash.Add(index);
                hash.Add(entry.UserData[index]);
            }

            return (uint)hash.ToHashCode();
        }

        private readonly ShaderResourcePlan _plan;

        public AddressRebase(ShaderResourcePlan plan)
        {
            _plan = plan;
            var candidates = new List<Candidate>();

            for (var buffer = 0; buffer < plan.Info.Buffers.Count; buffer++)
            {
                var source = plan.DescriptorSources[(int)plan.Info.Buffers[buffer].Source];
                if (source.Dwords.Length != 0 && UserDataIndex(source.Dwords[0]) is { } index)
                    candidates.Add(new Candidate(index, buffer));
            }

            var values = PlanValues(plan);
            var buffered = candidates.Select(candidate => candidate.UserData).ToHashSet();
            foreach (var raw in RawCandidates(plan, values))
                if (!buffered.Contains(raw.UserData))
                    candidates.Add(raw);

            _candidates = candidates.ToArray();
            _candidateMask = MaskOf(_candidates.Select(candidate => candidate.UserData));
            _printOffsets = PrintOffsetsOf(plan);
            _hasPrint = _printOffsets.Any(offsets => offsets.Length != 0);
            var memoryBases = MemoryBaseCandidates(plan);
            MemoryBases = memoryBases.Select(candidate => candidate.Buffer).ToArray();
            _memoryState = new int[plan.Info.Buffers.Count];
            _memorySlots = new int[plan.Info.Buffers.Count];
            _memorySlots.AsSpan().Fill(-1);
            foreach (var (buffer, slot) in memoryBases)
                _memorySlots[buffer] = slot;
            _memoryLearnable = memoryBases.Count;
            (TableSlotBuffers, TableSlotOffsets, TableSlotsThroughBases) = TableSlotSources(plan, memoryBases);
            DirectSlots = DirectSlotsOf(plan, TableSlotBuffers);
        }

        // A table slot that is one dword loaded through a user data pointer at constant offsets: its
        // word is that dword. High is the user data index of the pointer's high dword, or -1 for
        // the constant HighConstant.
        public readonly record struct DirectLoad(int Low, int High, ulong HighConstant, long Immediate, uint Offset);

        // Per table slot: its direct load (default for a slot through a memory-based buffer), or
        // null when some other slot is not a direct load.
        public DirectLoad[]? DirectSlots { get; }

        private DirectLoad[]? DirectSlotsOf(ShaderResourcePlan plan, int[] slotBuffers)
        {
            var loads = new DirectLoad[plan.TableReads.Count];
            for (var slot = 0; slot < loads.Length; slot++)
            {
                if (slotBuffers[slot] >= 0)
                    continue;
                var value = plan.TableReads[slot].Value;
                if (value.Kind != ScalarValueKind.ScalarAddressWord || value.Operands.Length < 2 ||
                    value.Operands[1].Kind != ScalarValueKind.Constant || (uint)value.MemoryIndex >= (uint)plan.Memory.Count)
                    return null;
                var handle = value.Operands[0];
                if (handle.Operands.Length < 2 || UserDataIndex(handle.Operands[0]) is not { } low)
                    return null;
                var high = UserDataIndex(handle.Operands[1]);
                if (high is null && handle.Operands[1].Kind != ScalarValueKind.Constant)
                    return null;
                loads[slot] = new DirectLoad(low, high ?? -1, high is null ? handle.Operands[1].Payload : 0,
                    (int)plan.Memory[value.MemoryIndex].Offset, (uint)value.Operands[1].Payload);
            }

            return loads;
        }

        // Per table slot: the memory-based buffer it reads a dword of at a constant byte offset from
        // the base, or -1. TableSlotsThroughBases: some slot reads through such a buffer otherwise.
        public int[] TableSlotBuffers { get; } = [];
        public long[] TableSlotOffsets { get; } = [];
        public bool TableSlotsThroughBases { get; }

        private static (int[] Buffers, long[] Offsets, bool Through) TableSlotSources(ShaderResourcePlan plan,
            List<(int Buffer, int Slot)> memoryBases)
        {
            var buffers = new int[plan.TableReads.Count];
            var offsets = new long[plan.TableReads.Count];
            buffers.AsSpan().Fill(-1);
            if (memoryBases.Count == 0)
                return (buffers, offsets, false);

            int BufferOf(ScalarValue handleWord)
            {
                foreach (var (buffer, slot) in memoryBases)
                {
                    var baseWord = plan.DescriptorSources[(int)plan.Info.Buffers[buffer].Source].Dwords[0];
                    if (ReferenceEquals(handleWord, baseWord) ||
                        (slot >= 0 && handleWord.Kind == ScalarValueKind.ResourceTableWord && (int)handleWord.Payload == slot))
                        return buffer;
                }

                return -1;
            }

            var through = false;
            for (var slot = 0; slot < plan.TableReads.Count; slot++)
            {
                var value = plan.TableReads[slot].Value;
                if (value.Kind == ScalarValueKind.ScalarBufferWord && value.Operands.Length >= 2 &&
                    value.Operands[0].Operands.Length != 0 && value.Operands[1].Kind == ScalarValueKind.Constant &&
                    value.MemoryIndex >= 0 && value.MemoryIndex < plan.Memory.Count &&
                    BufferOf(value.Operands[0].Operands[0]) is var direct and >= 0)
                {
                    buffers[slot] = direct;
                    offsets[slot] = ((long)plan.Memory[value.MemoryIndex].Offset + (long)(uint)value.Operands[1].Payload) & ~3L;
                    continue;
                }

                var visited = new HashSet<ScalarValue>();
                var pending = new Stack<ScalarValue>();
                pending.Push(value);
                while (!through && pending.Count != 0)
                {
                    var current = pending.Pop();
                    if (!visited.Add(current))
                        continue;
                    if (current.Kind == ScalarValueKind.ScalarBufferWord && current.Operands.Length != 0 &&
                        current.Operands[0].Operands.Length != 0 && BufferOf(current.Operands[0].Operands[0]) >= 0)
                        through = true;
                    foreach (var operand in current.Operands)
                        pending.Push(operand);
                }
            }

            return (buffers, offsets, through);
        }

        private readonly int[] _memorySlots;

        // Memory-loaded bases, learned per buffer like the user data words: confirmations count up
        // to MemoryActive; MemoryRejected is final.
        private const int MemoryActive = Confirmations;
        private const int MemoryRejected = -1;
        private readonly int[] _memoryState;
        private int _memoryLearnable;

        public IReadOnlyList<int> MemoryBases { get; }

        public bool HasLearnableMemoryBases => _memoryLearnable != 0;


        public bool IsRejectedMemoryBase(int buffer) => _memoryState[buffer] == MemoryRejected;

        public bool AreActiveMemoryBases(List<(int Buffer, uint Word)> patches)
        {
            foreach (var (buffer, _) in patches)
                if (_memoryState[buffer] < MemoryActive)
                    return false;
            return true;
        }

        // A rebased refresh computed for a stale entry, compared with the full materialization of
        // the same draw: agreement counts towards activating its bases, one disagreement rejects
        // them for good.
        public void ConfirmMemoryBases(List<(int Buffer, uint Word)> patches, bool agrees)
        {
            foreach (var (buffer, _) in patches)
            {
                if (_memoryState[buffer] == MemoryRejected)
                    continue;
                if (!agrees)
                {
                    _memoryState[buffer] = MemoryRejected;
                    _memoryLearnable--;
                    Interlocked.Increment(ref _memoryBasesRejected);
                }
                else if (_memoryState[buffer] < MemoryActive)
                {
                    _memoryState[buffer]++;
                }
            }
        }

        // The window of buffer reads a patched base covers for this draw.
        public static bool TryMovedWindow(uint[] descriptor, uint baseWord, out ulong start, out ulong size)
        {
            start = size = 0;
            if (descriptor.Length < 3)
                return false;
            start = baseWord | ((ulong)(descriptor[1] & 0xFFFF) << 32);
            var stride = (descriptor[1] >> 16) & 0x3FFF;
            size = stride == 0 ? descriptor[2] : (ulong)descriptor[2] * stride;
            return true;
        }

        // The base dword moves, and so does the table slot that holds the same word. buffers are the
        // entry's (shared) or ApplyBuffers' copy of them; only arrays shared with the entry are copied
        // before they are written. The table is copied when a slot moves.
        public (uint[][] Buffers, uint[] Table) PatchMemoryBases(uint[][] shared, uint[][] buffers, uint[] table,
            List<(int Buffer, uint Word)> patches)
        {
            if (ReferenceEquals(buffers, shared))
                buffers = (uint[][])buffers.Clone();
            var sharedTable = table;
            foreach (var (buffer, word) in patches)
            {
                if (ReferenceEquals(buffers[buffer], shared[buffer]))
                    buffers[buffer] = (uint[])buffers[buffer].Clone();
                var previous = buffers[buffer][0];
                buffers[buffer][0] = word;
                var slot = _memorySlots[buffer];
                var flat = slot >= 0 && slot < _plan.TableReads.Count ? (int)_plan.TableReads[slot].FlatOffset : -1;
                if (flat >= 0 && flat < sharedTable.Length && sharedTable[flat] == previous)
                {
                    if (ReferenceEquals(table, sharedTable))
                        table = (uint[])table.Clone();
                    table[flat] = word;
                }
            }

            return (buffers, table);
        }

        private int? UserDataIndex(ScalarValue value)
        {
            if (value.Kind != ScalarValueKind.UserData)
                return null;
            var register = value.UserDataRegister;
            return register >= _plan.UserDataBase && register - _plan.UserDataBase < _plan.UserDataCount
                ? (int)(register - _plan.UserDataBase)
                : null;
        }

        // Every scalar load of the plan whose handle takes its low dword verbatim from user data.
        // A word qualifies only when all its loads use constant offsets and one high dword.
        // Every value the plan evaluates, operands included.
        private static HashSet<ScalarValue> PlanValues(ShaderResourcePlan plan)
        {
            var roots = new List<ScalarValue>();
            foreach (var source in plan.DescriptorSources)
                roots.AddRange(source.Dwords);
            roots.AddRange(plan.TableReads.Select(read => read.Value));
            roots.AddRange(plan.DynamicReads);
            foreach (var access in plan.Accesses)
            {
                if (access is null)
                    continue;
                foreach (var value in new[] { access.Handle, access.SamplerHandle, access.Read, access.Offset, access.Active })
                    if (value is not null)
                        roots.Add(value);
            }

            roots.AddRange(plan.Graph.Values);
            var visited = new HashSet<ScalarValue>();
            var pending = new Stack<ScalarValue>(roots);
            while (pending.Count != 0)
            {
                var value = pending.Pop();
                if (!visited.Add(value))
                    continue;
                foreach (var operand in value.Operands)
                    pending.Push(operand);
            }

            return visited;
        }

        // Buffers whose base dword is read from guest memory, directly or through a slot of the
        // flattened table: descriptor tables the title rewrites per frame or per draw point them at
        // freshly allocated constants. The slot (or -1) is where the table holds the same word.
        // Recorded reads inside such a buffer's window move with its base.
        private static List<(int Buffer, int Slot)> MemoryBaseCandidates(ShaderResourcePlan plan)
        {
            var bases = new List<(int Buffer, int Slot)>();
            for (var buffer = 0; buffer < plan.Info.Buffers.Count; buffer++)
            {
                var source = plan.DescriptorSources[(int)plan.Info.Buffers[buffer].Source];
                if (source.Dwords.Length < 2)
                    continue;
                var baseWord = source.Dwords[0];
                var slot = baseWord.Kind == ScalarValueKind.ResourceTableWord ? (int)baseWord.Payload : -1;
                if (slot < 0 && baseWord.Kind is not (ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord))
                    continue;
                bases.Add((buffer, slot));
            }

            return bases;
        }

        private IEnumerable<Candidate> RawCandidates(ShaderResourcePlan plan, HashSet<ScalarValue> values)
        {
            var found = new Dictionary<int, Candidate>();
            var disqualified = new HashSet<int>();
            foreach (var value in values)
            {
                if (value.Kind is not (ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord) ||
                    value.Operands.Length < 2 || value.Operands[0].Operands.Length < 2 ||
                    UserDataIndex(value.Operands[0].Operands[0]) is not { } index)
                    continue;

                var high = value.Operands[0].Operands[1];
                var highIndex = UserDataIndex(high);
                var offset = value.Operands[1];
                if ((highIndex is null && high.Kind != ScalarValueKind.Constant) || offset.Kind != ScalarValueKind.Constant ||
                    value.MemoryIndex >= plan.Memory.Count)
                {
                    disqualified.Add(index);
                    continue;
                }

                var relative = RelativeOffset(plan, value);
                var candidate = new Candidate(index, -1, highIndex ?? -1, highIndex is null ? (uint)high.Payload : 0, relative, relative);
                if (found.TryGetValue(index, out var existing))
                {
                    if (existing.High != candidate.High || existing.HighConstant != candidate.HighConstant)
                    {
                        disqualified.Add(index);
                        continue;
                    }

                    candidate = existing with
                    {
                        MinOffset = Math.Min(existing.MinOffset, relative),
                        MaxOffset = Math.Max(existing.MaxOffset, relative),
                    };
                }

                found[index] = candidate;
            }

            return found.Values.Where(candidate => !disqualified.Contains(candidate.UserData));
        }

        // The byte offset from its handle's base of the dword a raw scalar load reads.
        private static long RelativeOffset(ShaderResourcePlan plan, ScalarValue value)
        {
            var immediate = (long)plan.Memory[value.MemoryIndex].Offset;
            var offset = (uint)value.Operands[1].Payload;
            return value.Kind == ScalarValueKind.ScalarBufferWord
                ? (immediate + (long)offset) & ~3L
                : (immediate & ~3L) + (long)(offset & ~3u);
        }

        // The content print. A raw candidate is a pointer to a descriptor table. Once it is rebased the
        // key leaves it out, so every object a shader draws through such a pointer would share one key
        // and the few variants chained behind it; and before it is, an object whose table moves every
        // frame is under a new key every frame. The print puts back what tells objects apart: the
        // descriptor dwords the plan loads through the pointer, read where it points now. It only
        // chooses the key: Entry.Matches and Validate still decide a hit on the entry it finds.
        //
        // Per candidate, the byte offsets from its base of up to PrintWords such dwords, spread over
        // the ones within PrintSpanBytes of the first so one read covers them; empty for a buffer
        // candidate and for a pointer the plan loads no descriptor dword through. A buffer's base dword
        // is left out: a memory-loaded base moves on every draw and TryRebaseRefresh patches it.
        private readonly long[][] _printOffsets;
        private readonly bool _hasPrint;
        private const int PrintWords = 32;
        private const int PrintSpanBytes = 4096;
        private const ulong PrintSeed = 0x243F_6A88_85A3_08D3UL;
        // A word that cannot be found or read: outside the range of a word.
        private const ulong MissingWord = 1UL << 32;

        private long[][] PrintOffsetsOf(ShaderResourcePlan plan)
        {
            var offsets = new long[_candidates.Length][];
            offsets.AsSpan().Fill([]);
            if (!ContentKey.Enabled)
                return offsets;
            var bufferSources = new HashSet<int>();
            foreach (var buffer in plan.Info.Buffers)
                bufferSources.Add((int)buffer.Source);
            var found = new SortedSet<long>?[_candidates.Length];
            for (var source = 0; source < plan.DescriptorSources.Count; source++)
            {
                var dwords = plan.DescriptorSources[source].Dwords;
                for (var dword = bufferSources.Contains(source) ? 1 : 0; dword < dwords.Length; dword++)
                {
                    // A descriptor dword the plan flattened into its table is that slot's load.
                    var value = dwords[dword];
                    if (value.Kind == ScalarValueKind.ResourceTableWord && value.Payload < (ulong)plan.TableReads.Count)
                        value = plan.TableReads[(int)value.Payload].Value;
                    if (value.Kind is not (ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord) ||
                        value.Operands.Length < 2 || value.Operands[0].Operands.Length < 2 ||
                        value.Operands[1].Kind != ScalarValueKind.Constant || (uint)value.MemoryIndex >= (uint)plan.Memory.Count ||
                        UserDataIndex(value.Operands[0].Operands[0]) is not { } index)
                        continue;
                    var candidate = Array.FindIndex(_candidates, candidate => candidate.UserData == index && candidate.Buffer < 0);
                    if (candidate >= 0)
                        (found[candidate] ??= []).Add(RelativeOffset(plan, value));
                }
            }

            for (var candidate = 0; candidate < found.Length; candidate++)
            {
                if (found[candidate] is not { Count: > 0 } set)
                    continue;
                var first = set.Min;
                var all = set.Where(offset => offset - first <= PrintSpanBytes - sizeof(uint)).ToArray();
                if (all.Length > PrintWords)
                    all = Enumerable.Range(0, PrintWords).Select(index => all[(int)((long)index * all.Length / PrintWords)]).ToArray();
                offsets[candidate] = all;
            }

            return offsets;
        }

        private static ulong MixPrint(ulong print, ulong value) =>
            System.Numerics.BitOperations.RotateLeft((print ^ value) * 0x9E37_79B9_7F4A_7C15UL, 29);

        // Folds the words into the print in four independent lanes: one dependent chain of up to
        // PrintWords multiplies was most of what a print cost.
        private static ulong FoldPrint(ulong print, ReadOnlySpan<ulong> words)
        {
            ulong first = print, second = print ^ 0x1357_9BDF_2468_ACE1UL, third = print ^ 0x0F1E_2D3C_4B5A_6979UL,
                fourth = print ^ 0x7A69_5847_3625_1403UL;
            var index = 0;
            for (; index + 4 <= words.Length; index += 4)
            {
                first = MixPrint(first, words[index]);
                second = MixPrint(second, words[index + 1]);
                third = MixPrint(third, words[index + 2]);
                fourth = MixPrint(fourth, words[index + 3]);
            }

            for (; index < words.Length; index++)
                first = MixPrint(first, words[index]);
            return MixPrint(MixPrint(MixPrint(first, second), third), fourth);
        }

        // The print of the descriptors the entry read through every candidate not rejected: what the
        // print of the same object drawn again is, wherever its table then lies. Null when none has one.
        private ulong? EntryPrint(Entry entry)
        {
            var print = PrintSeed;
            var any = false;
            Span<ulong> words = stackalloc ulong[PrintWords];
            for (var index = 0; index < _candidates.Length; index++)
            {
                var offsets = _printOffsets[index];
                var candidate = _candidates[index];
                if (offsets.Length == 0 || IsUsed(_rejected, candidate.UserData))
                    continue;
                if (candidate.UserData >= entry.UserData.Length || candidate.High >= entry.UserData.Length)
                    return null;
                var addressBase = RawBase(entry.UserData, candidate, entry.UserData[candidate.UserData]);
                any = true;
                print = MixPrint(print, (ulong)index);
                var found = words[..offsets.Length];
                if (!TryRecordedRun(entry, addressBase, offsets, found))
                {
                    for (var word = 0; word < offsets.Length; word++)
                    {
                        found[word] = TryFindWord(entry, unchecked(addressBase + (ulong)offsets[word]), out var at)
                            ? System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(entry.Bytes.AsSpan(at, sizeof(uint)))
                            : MissingWord;
                    }
                }

                print = FoldPrint(print, found);
            }

            return any ? print : null;
        }

        // The words at the offsets when one recorded range holds them all, as a descriptor table's
        // usually does: one search for the run instead of one per word, with the same words found.
        private static bool TryRecordedRun(Entry entry, ulong addressBase, long[] offsets, Span<ulong> words)
        {
            var first = unchecked(addressBase + (ulong)offsets[0]);
            if (!TryFindRange(entry, first, out var range, out var at))
                return false;
            var last = unchecked(first + (ulong)(offsets[^1] - offsets[0]));
            if (last < first || last - entry.RangeAddresses[range] + sizeof(uint) > (ulong)entry.RangeLengths[range])
                return false;
            // A later range that starts inside the run would be where the search finds its words
            // from there on (overlapping ranges, which only unaligned reads make).
            if (range + 1 < entry.RangeAddresses.Length && entry.RangeAddresses[range + 1] <= last)
                return false;
            var bytes = entry.Bytes.AsSpan(at);
            for (var word = 0; word < offsets.Length; word++)
                words[word] = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes[(int)(offsets[word] - offsets[0])..]);
            return true;
        }

        // The print of the descriptors behind this draw's rebased pointers, read where they lie now;
        // zero when no rebased candidate has one. Words that cannot be read print as missing, the
        // same for the lookup and for the entry stored after it. Remembered for TryRecallPrint.
        public ulong MemoryPrint(IReadOnlyList<uint> userData, ResidentGuestBytesReader reader, ref byte[] scratch)
        {
            if (!_hasPrint || Active.Length == 0)
                return 0;
            var print = PrintSeed;
            var bases = PrintSeed;
            var any = false;
            Span<ulong> words = stackalloc ulong[PrintWords];
            for (var index = 0; index < _candidates.Length; index++)
            {
                var offsets = _printOffsets[index];
                var candidate = _candidates[index];
                if (offsets.Length == 0 || !IsUsed(Active, candidate.UserData))
                    continue;
                if (candidate.UserData >= userData.Count || candidate.High >= userData.Count)
                    return 0;
                var addressBase = RawBase(userData, candidate, userData[candidate.UserData]);
                any = true;
                print = MixPrint(print, (ulong)index);
                bases = MixPrint(bases, addressBase);
                var length = (int)(offsets[^1] - offsets[0]) + sizeof(uint);
                if (scratch.Length < length)
                    scratch = new byte[Math.Max(length, scratch.Length * 2)];
                var bytes = scratch.AsSpan(0, length);
                var start = (long)addressBase + offsets[0];
                var read = start >= 0 && reader((ulong)start, bytes, false);
                var found = words[..offsets.Length];
                for (var word = 0; word < offsets.Length; word++)
                {
                    found[word] = read
                        ? System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes[(int)(offsets[word] - offsets[0])..])
                        : MissingWord;
                }

                print = FoldPrint(print, found);
            }

            if (!any)
                return 0;
            (_printMemo ??= new (ulong, ulong[]?, ulong)[PrintMemoSlots])[MemoSlot(bases)] = (bases, Active, print);
            return print;
        }

        // The print each recent set of rebased pointers gave (MemoryPrint), so a hit through the same
        // pointers finds its key without reading the descriptors here and again in Validate. A recalled
        // print only guesses the key: Entry.Matches and Validate still decide the hit, and when there
        // is none the draw reads the print again. Allocated once a pointer with a print is rebased.
        private const int PrintMemoSlots = 32;
        private (ulong Bases, ulong[]? Active, ulong Print)[]? _printMemo;

        private static int MemoSlot(ulong bases) => (int)(bases >> 59);

        public bool TryRecallPrint(IReadOnlyList<uint> userData, out ulong print)
        {
            print = 0;
            if (_printMemo is not { } memo || Active.Length == 0)
                return false;
            var bases = PrintSeed;
            var any = false;
            for (var index = 0; index < _candidates.Length; index++)
            {
                var candidate = _candidates[index];
                if (_printOffsets[index].Length == 0 || !IsUsed(Active, candidate.UserData))
                    continue;
                if (candidate.UserData >= userData.Count || candidate.High >= userData.Count)
                    return false;
                bases = MixPrint(bases, RawBase(userData, candidate, userData[candidate.UserData]));
                any = true;
            }

            ref readonly var slot = ref memo[MemoSlot(bases)];
            if (!any || slot.Bases != bases || !ReferenceEquals(slot.Active, Active))
                return false;
            print = slot.Print;
            return true;
        }

        // The window [start, start + size) of reads a candidate's base covers in this entry.
        private bool TryWindow(Entry entry, in Candidate candidate, out ulong start, out ulong size, out ulong addressBase)
        {
            start = size = addressBase = 0;
            if (candidate.Buffer >= 0)
            {
                if (candidate.Buffer >= entry.Snapshot.Buffers.Length)
                    return false;
                var descriptor = entry.Snapshot.Buffers[candidate.Buffer];
                if (descriptor.Length < 3)
                    return false;
                start = addressBase = descriptor[0] | ((ulong)(descriptor[1] & 0xFFFF) << 32);
                var stride = (descriptor[1] >> 16) & 0x3FFF;
                size = stride == 0 ? descriptor[2] : (ulong)descriptor[2] * stride;
                return true;
            }

            // The window starts at the base itself: the read prefetch fetches a pointer's block from
            // its base, ahead of the lowest offset a load uses.
            addressBase = RawBase(entry.UserData, candidate, entry.UserData[candidate.UserData]);
            var lowest = Math.Min(0, candidate.MinOffset);
            var low = (long)addressBase + lowest;
            if (low < 0)
                return false;
            start = (ulong)low;
            size = (ulong)(candidate.MaxOffset - lowest) + sizeof(uint);
            return true;
        }

        private static ulong RawBase(IReadOnlyList<uint> userData, in Candidate candidate, uint low)
        {
            var high = candidate.High >= 0 ? userData[candidate.High] : candidate.HighConstant;
            return (((ulong)high << 32) | low) & RawAddressMask & ~3UL;
        }

        // The rebased words; replaced, never mutated, so a caller can compare references.
        public ulong[] Active { get; private set; } = [];

        // Every rebased base dword of the entry equals the word it came from.
        public bool AppliesTo(Entry entry)
        {
            if (Active.Length == 0)
                return true;
            foreach (var (userData, buffer, _, _, _, _) in _candidates)
            {
                if (buffer >= 0 && IsUsed(Active, userData) &&
                    (buffer >= entry.Snapshot.Buffers.Length || entry.Snapshot.Buffers[buffer].Length == 0 ||
                     userData >= entry.UserData.Length || entry.Snapshot.Buffers[buffer][0] != entry.UserData[userData]))
                    return false;
            }

            return true;
        }

        // The cached snapshot as a hit gives it to this draw: carrying the draw's user data, with the
        // rebased base dwords patched from it. One snapshot at most, none when neither differs.
        public ResourceSnapshot Apply(ResourceSnapshot snapshot, IReadOnlyList<uint> userData)
        {
            var buffers = ApplyBuffers(snapshot.Buffers, userData);
            var current = CurrentUserData(snapshot.UserData, userData);
            return ReferenceEquals(buffers, snapshot.Buffers) && ReferenceEquals(current, snapshot.UserData)
                ? snapshot
                : new ResourceSnapshot
                {
                    Buffers = buffers,
                    Images = snapshot.Images,
                    Samplers = snapshot.Samplers,
                    FlattenedResourceTable = snapshot.FlattenedResourceTable,
                    UserData = current,
                    DeviceAddressRanges = snapshot.DeviceAddressRanges,
                    BindlessImages = snapshot.BindlessImages,
                };
        }

        // The buffers with each rebased base dword taken from this draw's user data; the same array
        // when none moves, otherwise a copy that copies only the descriptors it patches.
        public uint[][] ApplyBuffers(uint[][] shared, IReadOnlyList<uint> userData)
        {
            if (Active.Length == 0)
                return shared;
            uint[][]? buffers = null;
            foreach (var (index, buffer, _, _, _, _) in _candidates)
            {
                if (buffer < 0 || !IsUsed(Active, index) || shared[buffer][0] == userData[index])
                    continue;
                buffers ??= (uint[][])shared.Clone();
                if (ReferenceEquals(buffers[buffer], shared[buffer]))
                    buffers[buffer] = (uint[])buffers[buffer].Clone();
                buffers[buffer][0] = userData[index];
            }

            return buffers ?? shared;
        }

        // Compares a fresh materialization with the latest one that agrees on every word but the
        // candidates: the same object drawn again, usually a frame earlier, with new constants.
        // Comparing with the plan's previous draw instead finds another object's textures in
        // the way and never concludes.
        public void Learn(Entry built,
            Func<Entry, ulong[], (ResourceSnapshot Snapshot, List<(int Buffer, uint Word)>? Patches)?>? predict = null)
        {
            if (_candidates.Length == 0)
                return;
            var structure = StructureOf(built);
            _recent.TryGetValue(structure, out var previous);
            if (_recent.Count >= RecentCapacity)
            {
                _recent.Clear();
                RecentBytes = _recentByPrintBytes;
            }
            else if (previous is not null)
            {
                RecentBytes -= previous.EstimatedBytes;
            }

            _recent[structure] = built;
            RecentBytes += built.EstimatedBytes;
            // The same object's previous draw through another pointer is the comparison that can
            // confirm the pointer; the latest draw is often another object, which never does.
            if (RememberByPrint(structure, built) is { } same && !ReferenceEquals(same, previous) &&
                CandidateWordsDiffer(same, built))
                previous = same;
            if (previous is null || previous.UserData.Length != built.UserData.Length ||
                !previous.UsedUserData.AsSpan().SequenceEqual(built.UsedUserData))
                return;

            var differing = new List<int>();
            for (var index = 0; index < built.UserData.Length; index++)
            {
                if (!IsUsed(built.UsedUserData, index) || previous.UserData[index] == built.UserData[index])
                    continue;
                if (!IsUsed(_candidateMask, index))
                    return;
                differing.Add(index);
            }

            // A word already rejected explains any mismatch; blaming the others with it would reject
            // sound bases that merely change together with it.
            if (differing.Count == 0 || differing.Any(index => IsUsed(_rejected, index)))
                return;

            // Reads must land where the previous draw's did, moved with a rebased buffer when they
            // fall inside its window. Guest bytes that changed between the two draws make the
            // comparison say nothing about the words.
            var reads = CompareMovedReads(this, previous, built, differing);
            if (reads == ReadComparison.Inconclusive)
            {
                // Changed guest bytes hide whether the words are bases. Predict this draw from the
                // previous one as a hit with the words rebased would, moved memory-loaded bases
                // included, and judge the words by the prediction instead.
                if (predict?.Invoke(previous, Or(Active, MaskOf(differing))) is not { } prediction)
                    return;

                var agrees = Difference(prediction.Snapshot, previous.Specialization, built.Snapshot, built.Specialization) is null;
                if (prediction.Patches is { } patches)
                {
                    // A wrong prediction that moved bases blames those bases, not the words.
                    ConfirmMemoryBases(patches, agrees);
                    if (!agrees)
                        return;
                }
                else if (!agrees)
                {
                    _rejected = Or(_rejected, MaskOf(differing));
                    if (Overlaps(Active, _rejected))
                        Active = AndNot(Active, _rejected);
                    return;
                }

                Confirm(differing);
                return;
            }

            if (reads == ReadComparison.Mismatch ||
                !SameExceptBases(previous, built, differing, compareTable: reads == ReadComparison.SameBytes))
            {
                _rejected = Or(_rejected, MaskOf(differing));
                if (Overlaps(Active, _rejected))
                    Active = AndNot(Active, _rejected);
                return;
            }

            Confirm(differing);
        }

        private void Confirm(List<int> differing)
        {
            var activated = new List<int>();
            foreach (var index in differing)
            {
                var count = _confirmed.GetValueOrDefault(index) + 1;
                _confirmed[index] = count;
                if (count >= Confirmations && !IsUsed(_rejected, index) && !IsUsed(Active, index))
                    activated.Add(index);
            }

            if (activated.Count != 0)
                Active = Or(Active, MaskOf(activated));
        }

        // Runs a prediction as though the words in probe were rebased already.
        public T WithActive<T>(ulong[] probe, Func<T> run)
        {
            var saved = Active;
            Active = probe;
            try
            {
                return run();
            }
            finally
            {
                Active = saved;
            }
        }

        // The flattened table is left out when only table-only bytes changed: it holds the words read
        // from memory, whose addresses the read comparison already checked.
        private bool SameExceptBases(Entry previous, Entry built, List<int> differing, bool compareTable)
        {
            if (!previous.Specialization.Equals(built.Specialization))
                return false;

            var left = previous.Snapshot;
            var right = built.Snapshot;
            if (!SameWords(left.Images, right.Images) || !SameWords(left.Samplers, right.Samplers) ||
                (compareTable
                    ? !left.FlattenedResourceTable.AsSpan().SequenceEqual(right.FlattenedResourceTable)
                    : left.FlattenedResourceTable.Length != right.FlattenedResourceTable.Length) ||
                !left.DeviceAddressRanges.AsSpan().SequenceEqual(right.DeviceAddressRanges) ||
                left.Buffers.Length != right.Buffers.Length)
                return false;

            for (var buffer = 0; buffer < left.Buffers.Length; buffer++)
            {
                var before = left.Buffers[buffer];
                var after = right.Buffers[buffer];
                if (before.Length != after.Length)
                    return false;
                for (var dword = 0; dword < before.Length; dword++)
                {
                    if (dword == 0 && BaseWord(buffer, differing) is { } index)
                    {
                        if (before[0] != previous.UserData[index] || after[0] != built.UserData[index])
                            return false;
                    }
                    else if (before[dword] != after[dword])
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        // Every read of the built draw is the previous draw's read, moved by the rebase when it lies
        // in a moved buffer's window; bytesDiffer tells whether the guest bytes read there changed.
        private enum ReadComparison
        {
            // A read moved where the words cannot explain it while every byte read alike matched.
            Mismatch,
            // Guest bytes changed where the descriptors depend on them, or a changed pointer in
            // guest memory redirected later reads: the pair says nothing about the words.
            Inconclusive,
            SameBytes,
            // Only words the flattened table alone reads changed: per-draw constants.
            TableBytesDiffer,
        }

        private static ReadComparison CompareMovedReads(AddressRebase rebase, Entry previous, Entry built, List<int> differing)
        {
            var matched = previous.RangeAddresses.Length == built.RangeAddresses.Length;
            var bytesDiffer = false;
            var descriptorBytesDiffer = false;
            var common = Math.Min(previous.RangeAddresses.Length, built.RangeAddresses.Length);
            for (var range = 0; range < common; range++)
            {
                var length = previous.RangeLengths[range];
                if (rebase.Move(previous, built.UserData, previous.RangeAddresses[range], length, differing) != built.RangeAddresses[range] ||
                    length != built.RangeLengths[range] || previous.RangeClean[range] != built.RangeClean[range])
                {
                    matched = false;
                    continue;
                }

                for (var offset = 0; offset < length; offset += sizeof(uint))
                {
                    var before = previous.RangeOffsets[range] + offset;
                    var after = built.RangeOffsets[range] + offset;
                    if (previous.Bytes.AsSpan(before, sizeof(uint)).SequenceEqual(built.Bytes.AsSpan(after, sizeof(uint))))
                        continue;
                    bytesDiffer = true;
                    descriptorBytesDiffer |= !previous.WordTableOnly[before / sizeof(uint)] || !built.WordTableOnly[after / sizeof(uint)];
                }
            }

            if (!matched)
                return bytesDiffer ? ReadComparison.Inconclusive : ReadComparison.Mismatch;
            if (descriptorBytesDiffer || (bytesDiffer && !(previous.TableRefreshable && built.TableRefreshable)))
                return ReadComparison.Inconclusive;
            return bytesDiffer ? ReadComparison.TableBytesDiffer : ReadComparison.SameBytes;
        }

        // Where a read the entry recorded lies for a draw with this user data: a read inside the
        // window of a candidate whose base word moved moves with it. Only the words in moving count
        // (the differing words, or the rebased ones on a hit).
        private ulong Move(Entry entry, IReadOnlyList<uint> userData, ulong address, int length, IReadOnlyCollection<int>? moving)
        {
            foreach (var candidate in _candidates)
            {
                var index = candidate.UserData;
                if (moving is null ? !IsUsed(Active, index) : !moving.Contains(index))
                    continue;
                if (userData[index] == entry.UserData[index] || !TryWindow(entry, candidate, out var start, out var size, out var addressBase))
                    continue;
                // Only a range wholly inside the window moves: a range that merely overlaps it can hold
                // another block the title allocated next to it, which must not move.
                if (address < start || address - start + (ulong)length > size)
                    continue;
                var movedBase = candidate.Buffer >= 0
                    ? (addressBase & ~0xFFFF_FFFFUL) | userData[index]
                    : RawBase(entry.UserData, candidate, userData[index]);
                return address - addressBase + movedBase;
            }

            return address;
        }

        // A hit reads the entry's recorded ranges where they lie for this draw.
        public ulong MoveRead(Entry entry, IReadOnlyList<uint> userData, ulong address, int length) =>
            Active.Length == 0 ? address : Move(entry, userData, address, length, null);

        // True when a rebased base of the entry differs from this draw's word.
        public bool Moves(Entry entry, IReadOnlyList<uint> userData)
        {
            foreach (var candidate in _candidates)
                if (IsUsed(Active, candidate.UserData) && entry.UserData[candidate.UserData] != userData[candidate.UserData])
                    return true;
            return false;
        }

        private int? BaseWord(int buffer, List<int> differing)
        {
            foreach (var candidate in _candidates)
                if (candidate.Buffer == buffer && differing.Contains(candidate.UserData))
                    return candidate.UserData;
            return null;
        }

        private static bool SameWords(uint[][] left, uint[][] right)
        {
            if (left.Length != right.Length)
                return false;
            for (var index = 0; index < left.Length; index++)
                if (!left[index].AsSpan().SequenceEqual(right[index]))
                    return false;
            return true;
        }

        private static ulong[] MaskOf(IEnumerable<int> indices)
        {
            var mask = Array.Empty<ulong>();
            foreach (var index in indices)
            {
                if (mask.Length <= index >> 6)
                    Array.Resize(ref mask, (index >> 6) + 1);
                mask[index >> 6] |= 1UL << (index & 63);
            }

            return mask;
        }

        private static ulong[] Or(ulong[] left, ulong[] right)
        {
            var result = new ulong[Math.Max(left.Length, right.Length)];
            for (var index = 0; index < result.Length; index++)
                result[index] = (index < left.Length ? left[index] : 0) | (index < right.Length ? right[index] : 0);
            return result;
        }

        private static ulong[] AndNot(ulong[] left, ulong[] right)
        {
            var result = (ulong[])left.Clone();
            for (var index = 0; index < result.Length && index < right.Length; index++)
                result[index] &= ~right[index];
            return result.All(word => word == 0) ? [] : result;
        }

        private static bool Overlaps(ulong[] left, ulong[] right)
        {
            for (var index = 0; index < left.Length && index < right.Length; index++)
                if ((left[index] & right[index]) != 0)
                    return true;
            return false;
        }
    }

    // SHARPEMU_RESOURCE_CACHE_MB overrides the default total.
    public const long DefaultByteBudget = 256L << 20;

    // byteBudget is the total estimated retained bytes: three quarters for the two
    // generations, one quarter for the rebases' recent entries. Negative takes the default.
    public ResourceMaterializationCache(int generationCapacity = 16384, long byteBudget = -1)
    {
        _generationCapacity = Math.Max(1, generationCapacity);
        var budget = byteBudget >= 0 ? byteBudget : ConfiguredByteBudget();
        _recentByteBudget = budget / 4;
        _generationByteBudget = Math.Max(1, (budget - _recentByteBudget) / 2);
    }

    private static long ConfiguredByteBudget() =>
        long.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_RESOURCE_CACHE_MB"), out var megabytes) && megabytes > 0
            ? megabytes << 20
            : DefaultByteBudget;

    // The estimated bytes the cache retains: both generations and the rebases' recent entries.
    public long RetainedBytes => _youngBytes + _oldBytes + _recentBytes;

    public int EntryCount => _young.Count + _old.Count;

    private static long _lastRetainedBytes;

    private static long _totalHits;
    private static long _totalMisses;
    private static long _totalUncacheable;
    private const int MaxVariants = 8;
    private static long _totalStale;
    private static long _totalStaleUnreadable;
    private static long _totalRefreshes;
    private static long _totalNotFound;
    private static readonly long[] _refreshRefusals = new long[6];
    private static readonly string[] RefreshRefusalNames = ["not_refreshable", "unreadable", "descriptor_words", "unchanged", "table_eval", "rebased"];
    private int _lastRefreshRefusal;

    // Why a rebased refresh was refused, for the render report; MemoryBasesRejected counts bases a
    // wrong prediction rejected for good.
    private enum RebaseRefusal
    {
        NoMap,
        Unreadable,
        BaseInWindow,
        RejectedBase,
        ChangedWord,
        NoBaseChange,
        Table,
        Count,
    }

    private static readonly string[] RebaseRefusalNames =
        ["no_map", "unreadable", "base_in_window", "rejected_base", "changed_word", "no_base_change", "table"];
    private static readonly long[] _rebaseRefusals = new long[(int)RebaseRefusal.Count];
    private static long _memoryBasesRejected;

    private static bool RefuseRebase(RebaseRefusal reason)
    {
        Interlocked.Increment(ref _rebaseRefusals[(int)reason]);
        return false;
    }

    private static string TakeRebaseReport() =>
        " rebase_refused=" + string.Join(",", RebaseRefusalNames.Select((name, index) =>
            FormattableString.Invariant($"{name}:{Interlocked.Exchange(ref _rebaseRefusals[index], 0)}"))) +
        FormattableString.Invariant($" memory_bases_rejected={Interlocked.Exchange(ref _memoryBasesRejected, 0)}");

    private enum Diagnostic
    {
        KeyAbsent, KeyMismatch, OldPromotion, GenerationRotation, GenerationEntriesDropped,
        Variant1, Variant2, Variant3, Variant4, Variant5, Variant6, Variant7, Variant8, BytesChanged, Unreadable,
        RefreshLayout, RefreshUnreadable, RefreshDescriptorChanged, RefreshUnchanged,
        RefreshEvaluation, RefreshReadChanged, RefreshDescriptorSkipped, RefreshOlderVariant, RejectionPrecheck, Count,
    }
    private static readonly bool DiagnosticsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE") == "1";
    private static readonly long[] DiagnosticCounts = new long[(int)Diagnostic.Count];

    private static void CountDiagnostic(Diagnostic diagnostic, int count = 1)
    {
        if (DiagnosticsEnabled) Interlocked.Add(ref DiagnosticCounts[(int)diagnostic], count);
    }

    private static bool RejectRefresh(Diagnostic reason)
    {
        CountDiagnostic(reason);
        return false;
    }

    [ThreadStatic]
    private static bool _readingTable;

    public static bool ReadingTable
    {
        get => _readingTable;
        private set => _readingTable = value;
    }

    public long Hits { get; private set; }
    public long Misses { get; private set; }
    public long Uncacheable { get; private set; }
    public long TableRefreshes { get; private set; }

    // Process-wide counters since the previous call, for the periodic render report.
    public static string TakeReport()
    {
        var hits = Interlocked.Exchange(ref _totalHits, 0);
        var misses = Interlocked.Exchange(ref _totalMisses, 0);
        var uncacheable = Interlocked.Exchange(ref _totalUncacheable, 0);
        var stale = Interlocked.Exchange(ref _totalStale, 0);
        var staleUnreadable = Interlocked.Exchange(ref _totalStaleUnreadable, 0);
        var refreshes = Interlocked.Exchange(ref _totalRefreshes, 0);
        var notFound = Interlocked.Exchange(ref _totalNotFound, 0);
        var refusals = string.Join(",", RefreshRefusalNames.Select((name, index) => FormattableString.Invariant($"{name}:{Interlocked.Exchange(ref _refreshRefusals[index], 0)}")));
        var total = hits + misses;
        var diagnostics = DiagnosticsEnabled
            ? " " + string.Join(" ", Enumerable.Range(0, (int)Diagnostic.Count).Select(index =>
                $"{(Diagnostic)index}={Interlocked.Exchange(ref DiagnosticCounts[index], 0)}"))
            : string.Empty;
        return FormattableString.Invariant(
            $"[PERF][RESOURCE_CACHE] hits={hits} misses={misses} not_found={notFound} stale={stale} stale_unreadable={staleUnreadable} refreshes={refreshes} refused={refusals} uncacheable={uncacheable} hit_rate={(total == 0 ? 0 : hits * 100.0 / total):F1}% retained_mb={Volatile.Read(ref _lastRetainedBytes) / (1024.0 * 1024.0):F1} {RawReadPrefetch.TakeReport()}{diagnostics}") +
            TakeSlotReport() + TakeRebaseReport() +
            TakeStaleDifferenceReport() + TakeMissReport() + TakeDetailReport();
    }

    private static readonly bool StaleDiffEnabled = Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE") == "1";
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> _staleDifferences = new(StringComparer.Ordinal);

    private static void NoteStaleDifference(Entry stale, Entry built)
    {
        var parts = new SortedSet<string>(StringComparer.Ordinal);
        if (!stale.Specialization.Equals(built.Specialization))
            parts.Add("spec");
        DifferingWords("buf", stale.Snapshot.Buffers, built.Snapshot.Buffers, parts);
        DifferingWords("img", stale.Snapshot.Images, built.Snapshot.Images, parts);
        DifferingWords("smp", stale.Snapshot.Samplers, built.Snapshot.Samplers, parts);
        if (!stale.Snapshot.FlattenedResourceTable.AsSpan().SequenceEqual(built.Snapshot.FlattenedResourceTable))
            parts.Add("table");
        if (!stale.Snapshot.DeviceAddressRanges.AsSpan().SequenceEqual(built.Snapshot.DeviceAddressRanges))
            parts.Add("ranges");
        if (stale.Bytes.Length != built.Bytes.Length || !stale.RangeAddresses.AsSpan().SequenceEqual(built.RangeAddresses))
            parts.Add("reads_moved");
        var signature = parts.Count == 0 ? "none" : string.Join("+", parts);
        _staleDifferences.AddOrUpdate(signature, 1, static (_, count) => count + 1);
    }

    private static void DifferingWords(string name, uint[][] left, uint[][] right, SortedSet<string> parts)
    {
        if (left.Length != right.Length)
        {
            parts.Add($"{name}.count");
            return;
        }

        for (var index = 0; index < left.Length; index++)
        {
            if (left[index].Length != right[index].Length)
            {
                parts.Add($"{name}.length");
                continue;
            }

            for (var word = 0; word < left[index].Length; word++)
                if (left[index][word] != right[index][word])
                    parts.Add($"{name}.w{word}");
        }
    }

    private static string TakeStaleDifferenceReport()
    {
        if (_staleDifferences.IsEmpty)
            return string.Empty;
        var ranked = _staleDifferences.ToArray().OrderByDescending(pair => pair.Value).Take(10).ToArray();
        _staleDifferences.Clear();
        return Environment.NewLine + string.Join(Environment.NewLine,
            ranked.Select(pair => FormattableString.Invariant($"[PERF][RESOURCE_STALE_DIFF] n={pair.Value} changed={pair.Key}")));
    }

    // The first part a cached result differs from a fresh one in, or null when they are the same.
    private static string? Difference(ResourceSnapshot cached, ResourceSpecialization cachedSpecialization,
        ResourceSnapshot fresh, ResourceSpecialization freshSpecialization)
    {
        static string? Words(string name, uint[][] left, uint[][] right)
        {
            if (left.Length != right.Length)
                return $"{name}.count {left.Length}!={right.Length}";
            for (var index = 0; index < left.Length; index++)
            {
                if (left[index].Length != right[index].Length)
                    return $"{name}[{index}].length";
                for (var word = 0; word < left[index].Length; word++)
                    if (left[index][word] != right[index][word])
                        return $"{name}[{index}][{word}] cached=0x{left[index][word]:X} fresh=0x{right[index][word]:X}";
            }

            return null;
        }

        if (!cachedSpecialization.Equals(freshSpecialization))
            return "specialization";
        if (Words("buffer", cached.Buffers, fresh.Buffers) is { } buffers)
            return buffers;
        if (Words("image", cached.Images, fresh.Images) is { } images)
            return images;
        if (Words("sampler", cached.Samplers, fresh.Samplers) is { } samplers)
            return samplers;
        if (cached.FlattenedResourceTable.Length != fresh.FlattenedResourceTable.Length)
            return "table.length";
        for (var slot = 0; slot < cached.FlattenedResourceTable.Length; slot++)
            if (cached.FlattenedResourceTable[slot] != fresh.FlattenedResourceTable[slot])
                return $"table[{slot}] cached=0x{cached.FlattenedResourceTable[slot]:X} fresh=0x{fresh.FlattenedResourceTable[slot]:X}";
        if (!cached.DeviceAddressRanges.AsSpan().SequenceEqual(fresh.DeviceAddressRanges))
            return "ranges";
        if (!cached.UserData.AsSpan().SequenceEqual(fresh.UserData))
            return "user_data";
        return null;
    }

    public bool Materialize(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader,
        ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization,
        out ResourceMaterializationFailure failure)
    {
        var missesBefore = Misses;
        var timing = new MaterializationTiming(MissDiagnosticsEnabled);
        try
        {
            var success = MaterializeCore(plan, inputs, residentReader, ref snapshot, ref specialization, out failure, ref timing);
            if (success && Misses == missesBefore) _consecutiveMisses = 0;
            return success;
        }
        finally
        {
            timing.Complete(plan);
        }
    }

    private bool MaterializeCore(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader,
        ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization,
        out ResourceMaterializationFailure failure,
        ref MaterializationTiming timing)
    {
        ulong drawKey;
        ulong[]? learned;
        AddressRebase rebase;
        ulong key;
        var phase = SlotRefreshRuntime.Phase();
        if (phase != _slotPhase)
            ReportSlotPhase(phase);
        var slotRefresh = SlotRefreshRuntime.Enabled(phase);
        using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.CacheKey))
        {
            drawKey = DrawKeyOf(plan, inputs);
            _learnedUserData.TryGetValue(drawKey, out learned);
            rebase = RebaseOf(drawKey, plan);
            key = KeyOf(drawKey, inputs.UserData, learned, rebase.Active);
        }

        if (rebase.TryRecallPrint(inputs.UserData, out var recalled) &&
            TryRecalledHit(WithPrint(key, recalled), plan, inputs, residentReader, rebase, ref snapshot, ref specialization))
        {
            failure = default;
            return true;
        }

        using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.CacheKey))
        {
            key = WithPrint(key, rebase.MemoryPrint(inputs.UserData, residentReader, ref _printScratch));
        }

        bool found;
        Entry cached;
        using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.CacheFind))
        {
            found = TryFind(key, plan, inputs, rebase.Active, out cached);
        }
        Entry? stale = null;
        (List<(int Buffer, uint Word)> Patches, ResourceSnapshot Snapshot, Entry From)? learning = null;
        if (found)
        {
            var unreadable = false;
            Entry? previous = null;
            var variantIndex = 0;
            var refreshCandidates = 0;
            var rejection = default(RejectionSample);
            for (var variant = cached; variant is not null; previous = variant, variant = variant.Next)
            {
                CountDiagnostic((Diagnostic)((int)Diagnostic.Variant1 + Math.Min(variantIndex++, MaxVariants - 1)));
                // TryFind already checked the head against this draw's unchanged inputs.
                if ((!ReferenceEquals(variant, cached) && !variant.Matches(plan, inputs, rebase.Active)) || !rebase.AppliesTo(variant))
                    continue;
                stale ??= variant;
                if (rebase.Active.Length == 0 && !rebase.HasLearnableMemoryBases && rejection.Rejects(variant))
                {
                    CountDiagnostic(Diagnostic.RejectionPrecheck);
                    CountDiagnostic(Diagnostic.RefreshDescriptorSkipped);
                    continue;
                }
                if (Validate(variant, residentReader, rebase, inputs.UserData, slotRefresh,
                    out var variantUnreadable, out var descriptorChanged, out rejection, inputs.CompareResidentMemory))
                {
                    if (previous is not null)
                    {
                        previous.Next = variant.Next;
                        variant.Next = cached;
                        Store(key, variant);
                    }

                    Hits++;
                    Interlocked.Increment(ref _totalHits);
                    snapshot = HitSnapshot(variant, inputs.UserData, rebase);
                    ResourceMaterializationProfile.RecordCacheHit(recalled: false, rebase.Active.Length != 0,
                        variant.RangeAddresses.Length, variant.Bytes.Length);
                    specialization = variant.Specialization;
                    failure = default;
                    return true;
                }

                unreadable |= variantUnreadable;
                // A rejected candidate can fall back to a full walk, but cannot accept stale bytes.
                if (!variantUnreadable && !descriptorChanged && variant.TableRefreshable)
                    refreshCandidates |= 1 << (variantIndex - 1);
                else if (descriptorChanged)
                    CountDiagnostic(Diagnostic.RefreshDescriptorSkipped);
            }

            timing.ChangePhase(MaterializationPhase.TableRefresh);
            // Slots evaluated again first: exact, and it follows a moved pointer the rebase cannot.
            var tableRefused = false;
            if (stale is not null && slotRefresh)
            {
                if (SlotPlanOf(plan) is not { } slots)
                {
                    RefuseSlots(SlotRefusal.Plan);
                }
                else
                {
                    for (Entry? previousVariant = null, variant = cached; variant is not null; previousVariant = variant, variant = variant.Next)
                    {
                        if (!variant.Matches(plan, inputs, rebase.Active) || !rebase.AppliesTo(variant))
                            continue;
                        // A rebased word that moves, or an entry whose bytes were read where another
                        // draw's rebased words moved them: the slots would be read elsewhere.
                        if (variant.MovedReads || rebase.Moves(variant, inputs.UserData))
                        {
                            RefuseSlots(SlotRefusal.UserData);
                            continue;
                        }

                        if (TryRefreshSlots(variant, plan, slots, inputs, residentReader, rebase, rejection, out var evaluated, out var descriptorChanged))
                        {
                            if (ReferenceEquals(variant, cached))
                            {
                                evaluated.Next = cached.Next;
                            }
                            else
                            {
                                previousVariant!.Next = variant.Next;
                                evaluated.Next = cached;
                            }

                            Store(key, evaluated);
                            TableRefreshes++;
                            SlotRefreshes++;
                            Interlocked.Increment(ref _totalRefreshes);
                            Interlocked.Increment(ref _totalSlotRefreshes);
                            snapshot = evaluated.Snapshot;
                            specialization = evaluated.Specialization;
                            failure = default;
                            return true;
                        }

                        // A descriptor slot that changed is a word the table refresh would refuse.
                        tableRefused |= descriptorChanged && ReferenceEquals(variant, cached);
                    }
                }
            }
            // Moved bases first: a changed base word is a descriptor word, which a table refresh
            // would read in full only to refuse.
            ResourceSnapshot rebased = null!;
            List<(int Buffer, uint Word)> patches = null!;
            // Each variant is tried: a key's variants are often the same draw with different
            // descriptors (materials), and only the one this draw binds can be rebased.
            Entry? refreshedFrom = null;
            _lastRefreshRefusal = 0;
            Entry? beforeRefreshed = null;
            if (stale is not null && rebase.HasLearnableMemoryBases)
            {
                for (Entry? previousVariant = null, variant = cached; variant is not null; previousVariant = variant, variant = variant.Next)
                {
                    if (!variant.Matches(plan, inputs, rebase.Active) || !rebase.AppliesTo(variant))
                        continue;
                    if (TryRebaseRefresh(variant, plan, inputs, residentReader, rebase, out rebased, out patches))
                    {
                        refreshedFrom = variant;
                        beforeRefreshed = previousVariant;
                        break;
                    }
                }
            }

            if (refreshedFrom is not null)
            {
                if (rebase.AreActiveMemoryBases(patches))
                {
                    if (beforeRefreshed is not null)
                    {
                        beforeRefreshed.Next = refreshedFrom.Next;
                        refreshedFrom.Next = cached;
                        Store(key, refreshedFrom);
                    }

                    Hits++;
                    Interlocked.Increment(ref _totalHits);
                    snapshot = rebased;
                    specialization = refreshedFrom.Specialization;
                    failure = default;
                    return true;
                }

                learning = (patches, rebased, refreshedFrom);
            }
            else if (tableRefused)
            {
                _lastRefreshRefusal = 2;
            }
            else
            {
                // With fixed addresses any variant that validated readable with only table words
                // changed is a candidate; with rebased words only the newest one is.
                previous = null;
                variantIndex = 0;
                for (var variant = cached; variant is not null; previous = variant, variant = variant.Next, variantIndex++)
                {
                    var fixedAddresses = rebase.Active.Length == 0 && !rebase.HasLearnableMemoryBases;
                    if (!variant.Matches(plan, inputs, rebase.Active) || !rebase.AppliesTo(variant) ||
                        (fixedAddresses ? (refreshCandidates & (1 << variantIndex)) == 0 : !ReferenceEquals(variant, cached)) ||
                        !TryRefreshTable(key, variant, plan, inputs, residentReader, rebase, out var refreshed))
                        continue;

                    if (previous is not null)
                    {
                        previous.Next = variant.Next;
                        refreshed.Next = cached;
                        CountDiagnostic(Diagnostic.RefreshOlderVariant);
                    }
                    Store(key, refreshed);
                    TableRefreshes++;
                    Interlocked.Increment(ref _totalRefreshes);
                    snapshot = HitSnapshot(refreshed, inputs.UserData, rebase);
                    specialization = refreshed.Specialization;
                    failure = default;
                    return true;
                }
            }

            Interlocked.Increment(ref _totalStale);
            Interlocked.Increment(ref _refreshRefusals[refreshedFrom is not null ? 5 : _lastRefreshRefusal]);
            if (unreadable)
                Interlocked.Increment(ref _totalStaleUnreadable);
        }
        else
        {
            Interlocked.Increment(ref _totalNotFound);
        }

        timing.ChangePhase(MaterializationPhase.Materialization);
        Misses++;
        if (_consecutiveMisses < ColdScanThreshold) _consecutiveMisses++;
        RecordMiss(plan, key, found);
        Interlocked.Increment(ref _totalMisses);
        if (ColdAdmissionEnabled && _cacheSaturated && _consecutiveMisses >= ColdScanThreshold)
        {
            // Scan misses need no validation record unless their contents recur. A
            // recurring result is walked again with recording below before admission;
            // only that fresh, fully recorded walk can ever become a cached entry.
            var walking = new ResourceRuntimeInputs
            {
                // Match the recorder's owned user-data snapshot even when retention is bypassed.
                UserData = inputs.UserData.ToArray(),
                ShaderBase = inputs.ShaderBase,
                ReadMemory = inputs.ReadMemory,
                ReadCleanMemory = inputs.ReadCleanMemory,
                ReadResidentMemory = inputs.ReadResidentMemory,
                ReadsClean = inputs.ReadsClean,
                DataWritePending = inputs.DataWritePending,
                ComputeState = inputs.ComputeState,
                TablePhase = NoteTablePhase,
            };
            // Keep the caller's outputs intact if the required recorded rewalk fails.
            var walkedSnapshot = snapshot;
            var walkedSpecialization = specialization;
            try
            {
                if (!ResourceMaterializer.Materialize(plan, walking, ref walkedSnapshot, ref walkedSpecialization, out failure))
                    return false;
            }
            finally
            {
                ReadingTable = false;
            }
            if (DeferColdEntry(walkedSnapshot))
            {
                snapshot = walkedSnapshot;
                specialization = walkedSpecialization;
                ColdBypasses++;
                return true;
            }
        }

        var recorder = TakeRecorder();
        try
        {
            ResourceRuntimeInputs recording;
            using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.RecorderWrap))
            {
                recording = new ResourceRuntimeInputs
                {
                    UserData = recorder.WrapUserData(inputs.UserData),
                    ShaderBase = inputs.ShaderBase,
                    ReadMemory = recorder.Wrap(inputs.ReadMemory, clean: false),
                    ReadCleanMemory = recorder.Wrap(inputs.ReadCleanMemory, clean: true),
                    ReadResidentMemory = recorder.WrapResident(inputs.ReadResidentMemory),
                    ReadsClean = inputs.ReadsClean,
                    DataWritePending = inputs.DataWritePending,
                    ComputeState = inputs.ComputeState,
                    TablePhase = recorder.TablePhase,
                };
            }

            if (!ResourceMaterializer.Materialize(plan, recording, ref snapshot, ref specialization, out failure))
                return false;

            if (recorder.Failed)
            {
                Uncacheable++;
                Interlocked.Increment(ref _totalUncacheable);
                return true;
            }

            if (DeferColdEntry(snapshot))
            {
                ColdBypasses++;
                return true;
            }

            timing.ChangePhase(MaterializationPhase.CacheStore);
            Entry built;
            using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.EntryBuild))
            {
                built = recorder.Build(plan, inputs, snapshot, specialization, rebase.MemoryBases, like: stale);
            }
            if (StaleDiffEnabled && stale is not null && learning is null)
                NoteStaleDifference(stale, built);
            if (learning is { } pending)
            {
                var agrees = Difference(pending.Snapshot, pending.From.Specialization, built.Snapshot, built.Specialization) is null;
                rebase.ConfirmMemoryBases(pending.Patches, agrees);
            }

            var activeBefore = rebase.Active;
            Learn(rebase, built, plan, inputs, residentReader);
            var widened = Union(learned, built.UsedUserData);
            if (!ReferenceEquals(widened, learned) || !ReferenceEquals(activeBefore, rebase.Active))
            {
                _learnedUserData[drawKey] = widened;
                key = KeyFor(drawKey, inputs, widened, rebase, residentReader);
                found = false;
            }

            if (found)
            {
                built.Next = cached;
                var depth = 1;
                for (var variant = built; variant.Next is not null; variant = variant.Next)
                {
                    if (depth++ >= MaxVariants)
                    {
                        _cacheSaturated = true;
                        variant.Next = null;
                        break;
                    }
                }
            }

            Store(key, built);
            return true;
        }
        finally
        {
            ReturnRecorder(recorder);
        }
    }

    // Refresh only table words. Descriptor bytes and the reevaluated reads must pass fresh checks.
    private bool TryRefreshTable(ulong key, Entry cached, ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader, AddressRebase rebase, out Entry refreshed)
    {
        refreshed = null!;
        _lastRefreshRefusal = 0;
        if (!cached.TableRefreshable)
            return RejectRefresh(Diagnostic.RefreshLayout);

        // Each range is compared as soon as it is read: the first changed descriptor word refuses
        // the entry before the rest is read. The words are kept only once the refresh succeeds.
        if (_refreshCurrent.Length < cached.Bytes.Length)
            _refreshCurrent = new byte[Math.Max(cached.Bytes.Length, _refreshCurrent.Length * 2)];
        var current = _refreshCurrent;
        var moves = rebase.Moves(cached, inputs.UserData);
        if (moves && _refreshMoved.Length < cached.RangeAddresses.Length)
            _refreshMoved = new ulong[Math.Max(cached.RangeAddresses.Length, _refreshMoved.Length * 2)];
        var changed = false;
        for (var index = 0; index < cached.RangeAddresses.Length; index++)
        {
            var address = cached.RangeAddresses[index];
            var start = cached.RangeOffsets[index];
            var length = cached.RangeLengths[index];
            if (moves)
                _refreshMoved[index] = address = rebase.MoveRead(cached, inputs.UserData, address, length);
            if (!ReadWithProfile(residentReader, address, current.AsSpan(start, length), cached.RangeClean[index], CacheDetail.RefreshRead))
            {
                _lastRefreshRefusal = 1;
                return false;
            }

            using var comparisonProfile = MeasureDetail(CacheDetail.RefreshCompare);
            if (current.AsSpan(start, length).SequenceEqual(cached.Bytes.AsSpan(start, length)))
                continue;
            for (var offset = start; offset < start + length; offset += sizeof(uint))
            {
                if (current.AsSpan(offset, sizeof(uint)).SequenceEqual(cached.Bytes.AsSpan(offset, sizeof(uint))))
                    continue;
                if (!cached.WordTableOnly[offset / sizeof(uint)])
                {
                    _lastRefreshRefusal = 2;
                    return RejectRefresh(Diagnostic.RefreshDescriptorChanged);
                }
                changed = true;
            }
        }

        if (!changed)
        {
            _lastRefreshRefusal = 3;
            return false;
        }

        _lastRefreshRefusal = 4;

        var recorder = TakeRecorder();
        try
        {
            var recording = recorder.TableInputs(inputs);
            var cachedTable = cached.Snapshot.FlattenedResourceTable;
            ReadingTable = true;
            bool evaluated;
            uint[] table;
            try
            {
                using var evaluationProfile = MeasureDetail(CacheDetail.RefreshEvaluate);
                evaluated = ResourceMaterializer.TryEvaluateTable(plan, recording, out table);
            }
            finally
            {
                ReadingTable = false;
            }

            if (!evaluated || recorder.Failed || table.Length != cachedTable.Length)
                return RejectRefresh(Diagnostic.RefreshEvaluation);

            using (MeasureDetail(CacheDetail.RefreshVerify))
            foreach (var (address, word, _, _) in recorder.Reads)
            {
                var found = moves
                    ? TryFindMovedWord(cached, _refreshMoved, address, out var offset)
                    : TryFindWord(cached, address, out offset);
                if (!found || System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(offset, sizeof(uint))) != word)
                    return false;
            }

            // The written device-address slots come from reads validated unchanged above.
            foreach (var slot in plan.WrittenRangeSlotByHandle.Values)
                cachedTable.AsSpan((int)slot, ShaderResourcePlan.WrittenRangeDwordCount).CopyTo(table.AsSpan((int)slot));

            var previous = cached.Snapshot;
            refreshed = new Entry
            {
                Plan = cached.Plan,
                UserData = cached.UserData,
                UsedUserData = cached.UsedUserData,
                ShaderBase = cached.ShaderBase,
                ComputeState = cached.ComputeState,
                RangeAddresses = cached.RangeAddresses,
                RangeOffsets = cached.RangeOffsets,
                RangeLengths = cached.RangeLengths,
                RangeClean = cached.RangeClean,
                WordTableOnly = cached.WordTableOnly,
                BaseWordBuffers = cached.BaseWordBuffers,
                // Derived from the ranges and base words, which the refreshed entry shares.
                Joined = cached.Joined,
                JoinedHint = cached.Joined is null ? cached.JoinedHint : null,
                BaseWordProbes = cached.BaseWordProbes,
                // From the ranges and the descriptors, which it shares too (RangeWindowsOf,
                // TableCopySourcesOf); the slots' words changed, where they lie did not.
                RangeWindows = cached.RangeWindows,
                TableCopySources = cached.TableCopySources,
                SlotsHint = cached.Slots is { } slots && !ReferenceEquals(slots, Unmodelled) ? slots : cached.SlotsHint,
                TableRefreshable = true,
                MovedReads = moves || cached.MovedReads,
                // Published entries must not share mutable scratch storage.
                Bytes = current.AsSpan(0, cached.Bytes.Length).ToArray(),
                Snapshot = new ResourceSnapshot
                {
                    Buffers = previous.Buffers,
                    Images = previous.Images,
                    Samplers = previous.Samplers,
                    FlattenedResourceTable = table,
                    UserData = previous.UserData,
                    DeviceAddressRanges = previous.DeviceAddressRanges,
                    BindlessImages = previous.BindlessImages,
                },
                Specialization = cached.Specialization,
                Next = cached.Next,
            };
            return true;
        }
        finally
        {
            ReturnRecorder(recorder);
        }
    }

    // A miss that learns predicts with a table evaluation of its own while its recorder is still
    // taken, so up to two are kept: one spare made a new recorder, its lists and scratch, per such miss.
    private ReadRecorder TakeRecorder()
    {
        if (_spareRecorder is { } recorder)
        {
            _spareRecorder = _secondSpareRecorder;
            _secondSpareRecorder = null;
            return recorder;
        }

        return new ReadRecorder();
    }

    private void ReturnRecorder(ReadRecorder recorder)
    {
        recorder.Reset();
        _secondSpareRecorder = _spareRecorder;
        _spareRecorder = recorder;
    }

    // The byte offset of a dword read at address when the entry's ranges lie at moved addresses.
    private static bool TryFindMovedWord(Entry entry, ulong[] moved, ulong address, out int offset)
    {
        // moved may be a reused buffer longer than the entry's ranges.
        for (var index = 0; index < entry.RangeAddresses.Length; index++)
        {
            if (address < moved[index])
                continue;
            var delta = address - moved[index];
            if (delta % sizeof(uint) == 0 && delta < (ulong)entry.RangeLengths[index])
            {
                offset = entry.RangeOffsets[index] + (int)delta;
                return true;
            }
        }

        offset = 0;
        return false;
    }

    // The byte offset of a recorded dword in the entry's bytes, found by its address.
    private static bool TryFindWord(Entry entry, ulong address, out int offset) =>
        TryFindRange(entry, address, out _, out offset);

    private static ulong DrawKeyOf(ShaderResourcePlan plan, ResourceRuntimeInputs inputs)
    {
        var hash = new HashCode();
        hash.Add(RuntimeHelpers.GetHashCode(plan));
        hash.Add(inputs.ShaderBase);
        hash.Add(inputs.ComputeState);
        hash.Add(inputs.UserData.Count);
        return (ulong)(uint)hash.ToHashCode() | ((ulong)(uint)RuntimeHelpers.GetHashCode(plan) << 32);
    }

    // Only the words the draw's shader has been seen to read take part, less the rebased ones: the
    // key is an index, and Entry.Matches decides whether an entry is reusable.
    private static ulong KeyOf(ulong drawKey, IReadOnlyList<uint> userData, ulong[]? learned, ulong[]? rebased = null)
    {
        var low = (uint)drawKey;
        var high = (uint)(drawKey >> 32) ^ 0x9E3779B9u;
        if (learned is not null)
        {
            for (var index = 0; index < userData.Count; index++)
            {
                if (!IsUsed(learned, index) || (rebased is not null && IsUsed(rebased, index)))
                    continue;
                var word = userData[index];
                low = (low ^ word) * 0x01000193u;
                high = ((high + word) * 0x85EBCA6Bu) ^ (high >> 13);
            }
        }

        return ((ulong)high << 32) | low;
    }

    private byte[] _printScratch = new byte[256];

    // KeyOf, and for a rebased descriptor-table pointer the print of the descriptors it points at
    // now (AddressRebase.MemoryPrint): objects drawn through moving tables keep apart, and the same
    // object finds its entry wherever its table lies this time.
    private ulong KeyFor(ulong drawKey, ResourceRuntimeInputs inputs, ulong[]? learned, AddressRebase rebase,
        ResidentGuestBytesReader residentReader) =>
        WithPrint(KeyOf(drawKey, inputs.UserData, learned, rebase.Active),
            rebase.MemoryPrint(inputs.UserData, residentReader, ref _printScratch));

    private static ulong WithPrint(ulong key, ulong print) => print == 0 ? key : key ^ (print * 0xC2B2_AE3D_27D4_EB4FUL);

    // A hit under the key a recalled print gives (AddressRebase.TryRecallPrint), decided by the
    // checks every hit passes. Anything short of a hit leaves the draw to the full path, which
    // reads the print again and finds, refreshes or rebuilds the entry under the key it gives.
    private bool TryRecalledHit(ulong key, ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader, AddressRebase rebase, ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization)
    {
        Entry cached;
        using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.CacheFind))
        {
            if (!TryFind(key, plan, inputs, rebase.Active, out cached))
                return false;
        }

        for (Entry? previous = null, variant = cached; variant is not null; previous = variant, variant = variant.Next)
        {
            // Only the head was checked by TryFind; older variants still need the full check.
            if ((!ReferenceEquals(variant, cached) && !variant.Matches(plan, inputs, rebase.Active)) || !rebase.AppliesTo(variant) ||
                !Validate(variant, residentReader, rebase, inputs.UserData, false, out _, out _, out _, inputs.CompareResidentMemory))
                continue;
            if (previous is not null)
            {
                previous.Next = variant.Next;
                variant.Next = cached;
                Store(key, variant);
            }

            Hits++;
            Interlocked.Increment(ref _totalHits);
            snapshot = HitSnapshot(variant, inputs.UserData, rebase);
            ResourceMaterializationProfile.RecordCacheHit(recalled: true, rebase.Active.Length != 0,
                variant.RangeAddresses.Length, variant.Bytes.Length);
            specialization = variant.Specialization;
            return true;
        }

        return false;
    }

    // SHARPEMU_RESOURCE_CONTENT_KEY=0: no content print, so a rebased pointer's key holds every
    // object drawn through it, and a pointer is learned only against the plan's latest draw.
    private static class ContentKey
    {
        public static readonly bool Enabled = PerfSwitches.Enabled("SHARPEMU_RESOURCE_CONTENT_KEY");
    }

    private static bool IsUsed(ulong[] mask, int index) =>
        (uint)(index >> 6) < (uint)mask.Length && (mask[index >> 6] & (1UL << (index & 63))) != 0;

    // The same array when nothing new was learned, so the caller can tell.
    private static ulong[] Union(ulong[]? learned, ulong[] used)
    {
        if (learned is null)
            return used;
        var changed = learned.Length < used.Length;
        for (var index = 0; !changed && index < used.Length; index++)
            changed = (used[index] & ~learned[index]) != 0;
        if (!changed)
            return learned;
        var merged = new ulong[Math.Max(learned.Length, used.Length)];
        for (var index = 0; index < merged.Length; index++)
            merged[index] = (index < learned.Length ? learned[index] : 0) | (index < used.Length ? used[index] : 0);
        return merged;
    }

    // A cached snapshot is shared by every draw that hits it, so it keeps the user data of the draw
    // that built it; a draw whose unread words differ gets a copy that carries its own: these words.
    private static uint[] CurrentUserData(uint[] words, IReadOnlyList<uint> userData)
    {
        if (SameUserData(words, userData))
            return words;
        // A draw's user data array is never written once made (ShaderSource.UserData), so the
        // snapshot shares it rather than copying it for every draw, as ResourceSnapshot.WithUserData does.
        var current = userData as uint[];
        if (current is null)
        {
            current = new uint[userData.Count];
            for (var index = 0; index < current.Length; index++)
                current[index] = userData[index];
        }

        return current;
    }

    private static bool SameUserData(uint[] words, IReadOnlyList<uint> userData)
    {
        if (words.Length != userData.Count)
            return false;
        if (userData is uint[] array)
            return ReferenceEquals(words, array) || words.AsSpan().SequenceEqual(array);
        for (var index = 0; index < words.Length; index++)
            if (words[index] != userData[index])
                return false;
        return true;
    }

    // The snapshot a hit on the entry gives this draw (AddressRebase.Apply). A draw that differs from
    // the entry's in user data words its plan never reads differs on every hit, and so does a rebased
    // one: the entry keeps the last HitSnapshotCapacity snapshots it made, which identical draws share.
    // Draws that cycle through a few pointers (meshes, descriptor tables) then allocate nothing.
    // A kept snapshot is never written: it is handed out as is, so a draw recorded earlier and
    // encoded later still holds exactly the snapshot it was given.
    private static ResourceSnapshot HitSnapshot(Entry entry, IReadOnlyList<uint> userData, AddressRebase rebase)
    {
        using var profile = ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.HitSnapshot);
        var kept = entry.HitSnapshots;
        if (kept is not null)
        {
            foreach (var (last, active) in kept)
                if (last is not null && ReferenceEquals(active, rebase.Active) && SameUserData(last.UserData, userData))
                    return last;
        }

        var snapshot = rebase.Apply(entry.Snapshot, userData);
        if (!ReferenceEquals(snapshot, entry.Snapshot))
        {
            kept ??= entry.HitSnapshots = new (ResourceSnapshot?, ulong[]?)[HitSnapshotCapacity];
            kept[entry.NextHitSnapshot] = (snapshot, rebase.Active);
            entry.NextHitSnapshot = (entry.NextHitSnapshot + 1) % HitSnapshotCapacity;
        }

        return snapshot;
    }

    private const int HitSnapshotCapacity = 8;

    private bool TryFind(ulong key, ShaderResourcePlan plan, ResourceRuntimeInputs inputs, ulong[] rebased, out Entry entry)
    {
        if (_young.TryGetValue(key, out entry!))
        {
            var matches = entry.Matches(plan, inputs, rebased);
            if (!matches) CountDiagnostic(Diagnostic.KeyMismatch);
            return matches;
        }
        if (!_old.TryGetValue(key, out entry!))
        {
            CountDiagnostic(Diagnostic.KeyAbsent);
            return false;
        }
        if (!entry.Matches(plan, inputs, rebased))
        {
            CountDiagnostic(Diagnostic.KeyMismatch);
            return false;
        }
        CountDiagnostic(Diagnostic.OldPromotion);
        _old.Remove(key);
        _oldBytes -= entry.StoredChainBytes;
        Store(key, entry);
        return true;
    }

    private void Store(ulong key, Entry entry)
    {
        var bytes = ChainBytes(entry);
        if (_young.TryGetValue(key, out var replaced))
        {
            _youngBytes -= replaced.StoredChainBytes;
        }
        else if (_young.Count >= _generationCapacity || (_young.Count != 0 && _youngBytes + bytes > _generationByteBudget))
        {
            _cacheSaturated |= _old.Count != 0;
            CountDiagnostic(Diagnostic.GenerationRotation);
            CountDiagnostic(Diagnostic.GenerationEntriesDropped, _old.Count);
            RecordEvictions(_old.Keys);
            // The dropped generation's table becomes the new young one: a fresh table of this
            // capacity was ~450 KB on the large object heap per rotation.
            var recycled = _old;
            recycled.Clear();
            recycled.EnsureCapacity(_generationCapacity);
            _old = _young;
            _oldBytes = _youngBytes;
            _young = recycled;
            _youngBytes = 0;
        }

        entry.StoredChainBytes = bytes;
        _young[key] = entry;
        _youngBytes += bytes;
        Volatile.Write(ref _lastRetainedBytes, RetainedBytes);
    }

    // A key holds its entry and the variants chained behind it.
    private static long ChainBytes(Entry entry)
    {
        long bytes = 0;
        for (Entry? variant = entry; variant is not null; variant = variant.Next)
            bytes += variant.EstimatedBytes;
        return bytes;
    }

    // What a hit on the previous entry would give this draw with the words in probe rebased: its
    // reads moved and its bytes unchanged, or its memory-loaded bases moved as well. Null when
    // neither path applies, which says nothing about the words.
    private (ResourceSnapshot Snapshot, List<(int Buffer, uint Word)>? Patches)? Predict(Entry previous, ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs, ResidentGuestBytesReader residentReader, AddressRebase rebase, ulong[] probe) =>
        rebase.WithActive<(ResourceSnapshot, List<(int Buffer, uint Word)>?)?>(probe, () =>
        {
            if (!previous.Matches(plan, inputs, probe) || !rebase.AppliesTo(previous))
                return null;
            if (Validate(previous, residentReader, rebase, inputs.UserData, false, out _))
                return (rebase.Apply(previous.Snapshot, inputs.UserData), null);
            if (rebase.HasLearnableMemoryBases &&
                TryRebaseRefresh(previous, plan, inputs, residentReader, rebase, out var snapshot, out var patches))
                return (snapshot, patches);
            return null;
        });

    // Its own method: a lambda in MaterializeCore that captured the parameters made every call,
    // hits included, allocate the closure on entry.
    private void Learn(AddressRebase rebase, Entry built, ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader)
    {
        var before = rebase.RecentBytes;
        rebase.Learn(built, (previous, probe) => Predict(previous, plan, inputs, residentReader, rebase, probe));
        _recentBytes += rebase.RecentBytes - before;
        if (_recentBytes <= _recentByteBudget)
            return;
        foreach (var other in _rebases.Values)
            other.ForgetRecent();
        _recentBytes = 0;
    }

    // The table for this draw without evaluating it: a slot that reads a memory-based buffer at a
    // constant offset takes the word now at that offset, which the moved window has just read; the
    // other slots read nothing that changed, so they keep the entry's words, except a base slot,
    // which the patch has already moved. current holds this draw's words in the entry's layout.
    // The table is patched in place: the caller passes a copy it owns.
    private static bool TryCopyTable(Entry entry, ShaderResourcePlan plan, AddressRebase rebase, byte[] current, uint[] table)
    {
        var sources = entry.TableCopySources ??= TableCopySourcesOf(entry, plan, rebase, table.Length);
        if (sources.Length == 0)
            return false;
        for (var slot = 0; slot < sources.Length; slot++)
        {
            var offset = sources[slot];
            if (offset >= 0)
                table[(int)plan.TableReads[slot].FlatOffset] = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(offset));
        }

        return true;
    }

    // A table whose slots are all direct loads or reads through memory-based buffers.
    private static bool CanCopyDirect(AddressRebase rebase)
    {
        if (rebase.DirectSlots is not null)
            return true;
        return false;
    }


    // Every slot takes the dword it loads for this draw. A direct slot's address is evaluated from
    // the draw's own user data, as the table evaluation would, and its dword is taken from the range
    // this draw read around it; a slot through a memory-based buffer copies as in TryCopyTable.
    // Nothing is written unless every slot is placed.
    private static bool TryCopyDirectTable(Entry entry, ShaderResourcePlan plan, AddressRebase rebase, byte[] current,
        ulong[] moved, IReadOnlyList<uint> userData, uint[] table)
    {
        var through = entry.TableCopySources ??= TableCopySourcesOf(entry, plan, rebase, table.Length);
        var direct = rebase.DirectSlots!;
        if (through.Length != direct.Length)
            return Unplaced();
        Span<int> sources = direct.Length <= 256 ? stackalloc int[direct.Length] : new int[direct.Length];
        for (var slot = 0; slot < direct.Length; slot++)
        {
            if (plan.TableReads[slot].FlatOffset >= table.Length)
                return Unplaced();
            if (rebase.TableSlotBuffers[slot] >= 0)
            {
                if (through[slot] < 0)
                    return Unplaced();
                sources[slot] = through[slot];
                continue;
            }

            var load = direct[slot];
            if (load.Low >= userData.Count || load.High >= userData.Count)
                return Unplaced();
            var high = load.High >= 0 ? userData[load.High] : load.HighConstant;
            var baseAddress = ((high << 32) | userData[load.Low]) & 0x0000_FFFF_FFFF_FFFFul;
            if (RuntimeValueEvaluator.ResolveRawAddress(ScalarValueKind.ScalarAddressWord, baseAddress, high, 0,
                    load.Immediate, load.Offset, out var address) != RuntimeValueEvaluator.RawAddress.Read ||
                !TryFindMovedWord(entry, moved, address, out sources[slot]))
                return Unplaced();
        }

        for (var slot = 0; slot < sources.Length; slot++)
            table[(int)plan.TableReads[slot].FlatOffset] = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(sources[slot]));
        return true;

        static bool Unplaced()
        {
            return false;
        }
    }

    // Per table slot: the byte offset in the entry's words of the word the slot copies, or -1 for a
    // slot that keeps its word. Empty when some slot cannot be placed.
    private static int[] TableCopySourcesOf(Entry entry, ShaderResourcePlan plan, AddressRebase rebase, int tableLength)
    {
        var sources = new int[plan.TableReads.Count];
        for (var slot = 0; slot < sources.Length; slot++)
        {
            sources[slot] = -1;
            var buffer = rebase.TableSlotBuffers[slot];
            if (buffer < 0)
                continue;
            var descriptor = entry.Snapshot.Buffers[buffer];
            if (!AddressRebase.TryMovedWindow(descriptor, descriptor[0], out var start, out var size) ||
                rebase.TableSlotOffsets[slot] < 0 || (ulong)rebase.TableSlotOffsets[slot] + sizeof(uint) > size ||
                !TryFindWord(entry, start + (ulong)rebase.TableSlotOffsets[slot], out var offset) ||
                plan.TableReads[slot].FlatOffset >= tableLength)
                return [];
            sources[slot] = offset;
        }

        return sources;
    }

    // Where each memory-loaded base word of the entry lies: its range and byte offset in it.
    // Counted first, so the one array is all it allocates: it runs for every new entry.
    private static (int Range, int Offset)[] BaseWordProbesOf(Entry entry)
    {
        var map = entry.BaseWordBuffers!;
        var count = 0;
        for (var range = 0; range < entry.RangeAddresses.Length; range++)
            for (var offset = 0; offset < entry.RangeLengths[range]; offset += sizeof(uint))
                if (map[(entry.RangeOffsets[range] + offset) / sizeof(uint)] >= 0)
                    count++;
        if (count == 0)
            return [];

        var probes = new (int Range, int Offset)[count];
        var next = 0;
        for (var range = 0; range < entry.RangeAddresses.Length; range++)
            for (var offset = 0; offset < entry.RangeLengths[range]; offset += sizeof(uint))
                if (map[(entry.RangeOffsets[range] + offset) / sizeof(uint)] >= 0)
                    probes[next++] = (range, offset);
        return probes;
    }

    private byte[] _rebaseCurrent = [];
    private byte[] _refreshCurrent = [];
    private ulong[] _refreshMoved = [];
    private ulong[] _rebaseMoved = [];
    private readonly List<(int Buffer, uint Word)> _rebaseBases = new();
    private readonly List<(int Buffer, uint Word)> _rebaseFound = new();
    private readonly List<(ulong Start, ulong Size)> _rebaseWindows = new();

    // Per recorded range: the memory-based buffer whose window wholly holds it, or NoBase. Such a
    // range is read only once the buffer's base is known: it moves with the base.
    private static int[] RangeWindowsOf(Entry entry, AddressRebase rebase)
    {
        var windows = new int[entry.RangeAddresses.Length];
        windows.AsSpan().Fill(NoBase);
        for (var index = 0; index < windows.Length; index++)
        {
            foreach (var buffer in rebase.MemoryBases)
            {
                var descriptor = entry.Snapshot.Buffers[buffer];
                if (descriptor.Length >= 3 && AddressRebase.TryMovedWindow(descriptor, descriptor[0], out var start, out var size) &&
                    entry.RangeAddresses[index] >= start && entry.RangeAddresses[index] - start + (ulong)entry.RangeLengths[index] <= size)
                {
                    windows[index] = buffer;
                    break;
                }
            }
        }

        return windows;
    }

    // A stale entry whose only changed words are memory-loaded buffer bases, and constants only the
    // table reads: the bases are patched into a copy of the snapshot and the table is evaluated
    // again for this draw, which reads the constants where the moved bases now point. Every read of
    // that evaluation is a word the entry validated unchanged, or lies in the window a patched base
    // now covers. The descriptors and specialization are the entry's; learning checks that.
    private bool TryRebaseRefresh(Entry entry, ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader, AddressRebase rebase, out ResourceSnapshot snapshot,
        out List<(int Buffer, uint Word)> patches)
    {
        snapshot = null!;
        patches = null!;
        if (entry.BaseWordBuffers is not { } map)
            return RefuseRebase(RebaseRefusal.NoMap);
        var userData = inputs.UserData;
        // The words read here are compared and copied, never kept, so the buffers are reused.
        if (_rebaseCurrent.Length < entry.Bytes.Length)
            _rebaseCurrent = new byte[Math.Max(entry.Bytes.Length, _rebaseCurrent.Length * 2)];
        if (_rebaseMoved.Length < entry.RangeAddresses.Length)
            _rebaseMoved = new ulong[Math.Max(entry.RangeAddresses.Length, _rebaseMoved.Length * 2)];
        var current = _rebaseCurrent;
        var moved = _rebaseMoved;
        var windowed = entry.RangeWindows ??= RangeWindowsOf(entry, rebase);
        for (var index = 0; index < entry.RangeAddresses.Length; index++)
        {
            if (windowed[index] != NoBase)
                continue;
            moved[index] = rebase.MoveRead(entry, userData, entry.RangeAddresses[index], entry.RangeLengths[index]);
            if (!residentReader(moved[index], current.AsSpan(entry.RangeOffsets[index], entry.RangeLengths[index]), entry.RangeClean[index]))
                return RefuseRebase(RebaseRefusal.Unreadable);
            // A changed word of a range that does not move is a base or a constant, or the entry is
            // refused below whatever the other ranges hold.
            for (var offset = entry.RangeOffsets[index]; offset < entry.RangeOffsets[index] + entry.RangeLengths[index]; offset += sizeof(uint))
            {
                var word = offset / sizeof(uint);
                if (map[word] < 0 && !entry.WordTableOnly[word] &&
                    System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(offset)) !=
                    System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(entry.Bytes.AsSpan(offset)))
                    return RefuseRebase(RebaseRefusal.ChangedWord);
            }
        }

        // The new base of each buffer whose base word changed, from the ranges read so far. A base
        // word inside a moving window would need its own base first; such an entry is not rebased.
        var rangeCount = entry.RangeAddresses.Length;
        var bases = _rebaseBases;
        bases.Clear();
        foreach (var (range, rangeOffset) in entry.BaseWordProbes ??= BaseWordProbesOf(entry))
        {
            if (windowed[range] != NoBase)
                return RefuseRebase(RebaseRefusal.BaseInWindow);
            var offset = entry.RangeOffsets[range] + rangeOffset;
            var now = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(offset));
            if (now != System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(entry.Bytes.AsSpan(offset)))
                bases.Add((map[offset / sizeof(uint)], now));
        }

        for (var index = 0; index < rangeCount; index++)
        {
            var buffer = windowed[index];
            if (buffer == NoBase)
                continue;
            var oldBase = entry.Snapshot.Buffers[buffer][0];
            var newBase = oldBase;
            foreach (var (changed, word) in bases)
                if (changed == buffer)
                    newBase = word;
            moved[index] = entry.RangeAddresses[index] - oldBase + newBase;
            if (!residentReader(moved[index], current.AsSpan(entry.RangeOffsets[index], entry.RangeLengths[index]), entry.RangeClean[index]))
                return RefuseRebase(RebaseRefusal.Unreadable);
        }

        // Reused: the patches a refresh returns are read before the next one runs (Materialize
        // confirms a pending refresh's before Learn predicts with another).
        var found = _rebaseFound;
        found.Clear();
        // A changed constant outside the moved windows is read by a slot the copy cannot place.
        var constantsOutsideWindows = false;
        for (var index = 0; index < rangeCount; index++)
        {
            // Most ranges did not change at all; only a changed one is looked at word by word.
            if (current.AsSpan(entry.RangeOffsets[index], entry.RangeLengths[index])
                .SequenceEqual(entry.Bytes.AsSpan(entry.RangeOffsets[index], entry.RangeLengths[index])))
                continue;
            for (var offset = entry.RangeOffsets[index]; offset < entry.RangeOffsets[index] + entry.RangeLengths[index]; offset += sizeof(uint))
            {
                var word = offset / sizeof(uint);
                var now = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(offset));
                if (now == System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(entry.Bytes.AsSpan(offset)))
                    continue;
                var buffer = map[word];
                if (buffer >= 0)
                {
                    if (rebase.IsRejectedMemoryBase(buffer))
                        return RefuseRebase(RebaseRefusal.RejectedBase);
                    found.Add((buffer, now));
                }
                else if (!entry.WordTableOnly[word])
                    return RefuseRebase(RebaseRefusal.ChangedWord);
                else
                    constantsOutsideWindows |= windowed[index] == NoBase;
            }
        }

        if (found.Count == 0)
            return RefuseRebase(RebaseRefusal.NoBaseChange);
        // The entry's snapshot as a hit applies it, with the moved bases patched in: built from its
        // parts, so the draw allocates only the snapshot it returns.
        var cached = entry.Snapshot;
        var (patchedBuffers, patchedTable) = rebase.PatchMemoryBases(cached.Buffers, rebase.ApplyBuffers(cached.Buffers, userData),
            cached.FlattenedResourceTable, found);
        var patchedUserData = CurrentUserData(cached.UserData, userData);
        // A changed constant outside the windows is still copied when every slot is a direct load:
        // each slot's word is then the dword at its own address, which this draw has just read.
        var copyDirect = constantsOutsideWindows && CanCopyDirect(rebase);
        if (plan.TableReads.Count != 0 && (!constantsOutsideWindows || copyDirect) && !rebase.TableSlotsThroughBases)
        {
            // The patch cloned the table when it moved a base slot; otherwise it is the entry's.
            var table = ReferenceEquals(patchedTable, cached.FlattenedResourceTable)
                ? (uint[])patchedTable.Clone()
                : patchedTable;
            // The copy writes nothing unless every slot can be placed.
            if (copyDirect
                    ? TryCopyDirectTable(entry, plan, rebase, current, moved, userData, table)
                    : TryCopyTable(entry, plan, rebase, current, table))
            {
                snapshot = PatchedSnapshot(cached, patchedBuffers, table, patchedUserData);
                patches = found;
                return true;
            }
        }

        if (plan.TableReads.Count != 0)
        {
            if (!entry.TableRefreshable)
                return RefuseRebase(RebaseRefusal.Table);
            var windows = _rebaseWindows;
            windows.Clear();
            foreach (var (buffer, baseWord) in found)
                if (AddressRebase.TryMovedWindow(patchedBuffers[buffer], baseWord, out var start, out var size))
                    windows.Add((start, size));

            var recorder = TakeRecorder();
            try
            {
                var recording = recorder.TableInputs(inputs);
                var cachedTable = entry.Snapshot.FlattenedResourceTable;
                ReadingTable = true;
                bool evaluated;
                uint[] table;
                try
                {
                    evaluated = ResourceMaterializer.TryEvaluateTable(plan, recording, out table);
                }
                finally
                {
                    ReadingTable = false;
                }

                if (!evaluated || recorder.Failed || table.Length != cachedTable.Length)
                    return RefuseRebase(RebaseRefusal.Table);

                foreach (var (address, word, _, _) in recorder.Reads)
                {
                    var inWindow = false;
                    foreach (var (start, size) in windows)
                        inWindow |= address >= start && address - start < size;
                    if (inWindow)
                        continue;
                    if (!TryFindMovedWord(entry, moved, address, out var offset) ||
                        System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(offset, sizeof(uint))) != word)
                        return RefuseRebase(RebaseRefusal.Table);
                }

                // The written device-address slots come from reads validated unchanged above.
                foreach (var slot in plan.WrittenRangeSlotByHandle.Values)
                    cachedTable.AsSpan((int)slot, ShaderResourcePlan.WrittenRangeDwordCount).CopyTo(table.AsSpan((int)slot));
                patchedTable = table;
            }
            finally
            {
                ReturnRecorder(recorder);
            }
        }

        snapshot = PatchedSnapshot(cached, patchedBuffers, patchedTable, patchedUserData);
        patches = found;
        return true;
    }

    private static ResourceSnapshot PatchedSnapshot(ResourceSnapshot cached, uint[][] buffers, uint[] table, uint[] userData) =>
        new()
        {
            Buffers = buffers,
            Images = cached.Images,
            Samplers = cached.Samplers,
            FlattenedResourceTable = table,
            UserData = userData,
            DeviceAddressRanges = cached.DeviceAddressRanges,
            BindlessImages = cached.BindlessImages,
        };

    private readonly record struct RejectionSample(int Range, ulong Address, int Length, bool Clean, int WordOffset, uint Word)
    {
        public bool Rejects(Entry entry)
        {
            if (Length == 0 || Range >= entry.RangeAddresses.Length || entry.RangeAddresses[Range] != Address ||
                entry.RangeLengths[Range] != Length || entry.RangeClean[Range] != Clean)
                return false;
            var offset = entry.RangeOffsets[Range] + WordOffset;
            // This sample can reject a variant. It cannot prove that a variant is current.
            return !entry.WordTableOnly[offset / sizeof(uint)] &&
                System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(entry.Bytes.AsSpan(offset, sizeof(uint))) != Word;
        }

        // The same refusal for a slot refresh: the sampled word, read under the same policy at
        // the slot's address, differs from the descriptor word the entry evaluated there.
        public bool RefutesWord(ulong address, bool clean, uint expected) =>
            Length != 0 && Clean == clean && Address + (ulong)WordOffset == address && Word != expected;
    }

    private static ulong MoveValidationRead(Entry entry, IReadOnlyList<uint> userData, AddressRebase rebase,
        ulong address, int length)
    {
        using var profile = ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.AddressRebasing);
        return rebase.MoveRead(entry, userData, address, length);
    }

    private bool Validate(Entry entry, ResidentGuestBytesReader residentReader, AddressRebase rebase,
        IReadOnlyList<uint> userData, bool joined, out bool unreadable) =>
        Validate(entry, residentReader, rebase, userData, joined, out unreadable, out _, out _);

    private bool Validate(Entry entry, ResidentGuestBytesReader residentReader, AddressRebase rebase, IReadOnlyList<uint> userData,
        bool joined, out bool unreadable, out bool descriptorChanged, out RejectionSample rejection,
        ResidentGuestBytesComparer? comparer = null)
    {
        using var profile = ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.HitValidation);
        unreadable = false;
        descriptorChanged = false;
        rejection = default;
        // With no recorded guest reads there are no bytes or base-word probes to validate.
        if (entry.RangeAddresses.Length == 0) return true;
        if (joined && rebase.Active.Length == 0)
            return ValidateJoined(entry, residentReader, comparer, out unreadable, out descriptorChanged, out rejection);
        if (comparer is not null && InPlaceValidateEnabled &&
            CompareValidationRanges(entry, userData, rebase, comparer))
        {
            if (!InPlaceValidateVerify) return true;
            // Verification must use the original scalar reads, not enter this fast path again.
            var read = ValidateByReading(entry, residentReader, rebase, userData,
                out unreadable, out descriptorChanged, out rejection);
            if (!read)
                Console.Error.WriteLine($"[PERF][INPLACE_VALIDATE_VERIFY] mismatch: exact-address comparison found all {entry.RangeAddresses.Length} ranges equal, " +
                    $"scalar reading found them {(unreadable ? "unreadable" : "changed")}.");
            return read;
        }
        return ValidateByReading(entry, residentReader, rebase, userData,
            out unreadable, out descriptorChanged, out rejection);
    }

    private bool ValidateByReading(Entry entry, ResidentGuestBytesReader residentReader, AddressRebase rebase,
        IReadOnlyList<uint> userData, out bool unreadable, out bool descriptorChanged, out RejectionSample rejection)
    {
        unreadable = false;
        descriptorChanged = false;
        rejection = default;
        // A memory-loaded base usually moved when anything did: its few bytes are checked before
        // ranges that can span kilobytes. The full comparison below still decides a match.
        if (entry.BaseWordBuffers is not null)
        {
            Span<byte> word = stackalloc byte[sizeof(uint)];
            foreach (var (range, offset) in entry.BaseWordProbes ??= BaseWordProbesOf(entry))
            {
                var address = MoveValidationRead(entry, userData, rebase, entry.RangeAddresses[range], entry.RangeLengths[range]) + (ulong)offset;
                if (residentReader(address, word, entry.RangeClean[range]) &&
                    !word.SequenceEqual(entry.Bytes.AsSpan(entry.RangeOffsets[range] + offset, sizeof(uint))))
                    return false;
            }
        }

        for (var index = 0; index < entry.RangeAddresses.Length; index++)
        {
            var length = entry.RangeLengths[index];
            if (_scratch.Length < length)
                _scratch = new byte[Math.Max(length, _scratch.Length * 2)];
            var current = _scratch.AsSpan(0, length);
            var address = MoveValidationRead(entry, userData, rebase, entry.RangeAddresses[index], length);
            if (!ReadWithProfile(residentReader, address, current, entry.RangeClean[index], CacheDetail.ValidateRead))
            {
                unreadable = true;
                CountDiagnostic(Diagnostic.Unreadable);
                return false;
            }

            using var comparisonProfile = MeasureDetail(CacheDetail.ValidateCompare);
            if (!current.SequenceEqual(entry.Bytes.AsSpan(entry.RangeOffsets[index], length)))
            {
                RecordChangedWords(entry, index, current);
                CountDiagnostic(Diagnostic.BytesChanged);
                var previous = entry.Bytes.AsSpan(entry.RangeOffsets[index], length);
                for (var offset = 0; offset < length; offset += sizeof(uint))
                {
                    if (!entry.WordTableOnly[(entry.RangeOffsets[index] + offset) / sizeof(uint)] &&
                        !current.Slice(offset, sizeof(uint)).SequenceEqual(previous.Slice(offset, sizeof(uint))))
                    {
                        descriptorChanged = true;
                        rejection = new RejectionSample(index, entry.RangeAddresses[index], length, entry.RangeClean[index],
                            offset, System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current[offset..]));
                        break;
                    }
                }
                return false;
            }
        }

        return true;
    }

    private sealed class Entry
    {
        public required ShaderResourcePlan Plan { get; init; }
        public required uint[] UserData { get; init; }
        // The user data words the evaluator read, one bit per word.
        public required ulong[] UsedUserData { get; init; }
        public required ulong ShaderBase { get; init; }
        public required ComputeSelectorState? ComputeState { get; init; }
        public required ulong[] RangeAddresses { get; init; }
        public required int[] RangeOffsets { get; init; }
        public required int[] RangeLengths { get; init; }
        public required bool[] RangeClean { get; init; }
        // Per recorded dword: read only while the flattened table was evaluated.
        public required bool[] WordTableOnly { get; init; }
        // The table has the plan's plain layout, so no specialization read or extended it.
        public required bool TableRefreshable { get; init; }
        public required byte[] Bytes { get; init; }
        public required ResourceSnapshot Snapshot { get; init; }
        public required ResourceSpecialization Specialization { get; init; }
        // Per recorded dword: the buffer whose memory-loaded base dword it is, or a negative value.
        public int[]? BaseWordBuffers { get; init; }
        // Filled on the first rebased refresh of the entry; see RangeWindowsOf and TableCopySourcesOf.
        public int[]? RangeWindows { get; set; }
        public int[]? TableCopySources { get; set; }
        public (int Range, int Offset)[]? BaseWordProbes { get; set; }
        // Bytes a table refresh read where rebased words moved the ranges, not at RangeAddresses.
        public bool MovedReads { get; init; }
        // Filled on the first slot refresh of the entry; see ModelOf.
        public SlotModel? Slots { get; set; }
        // Filled on the first joined validation of the entry; see JoinedReadsOf.
        public JoinedReads? Joined { get; set; }
        // The derived data of the entry this one was rebuilt from, when it could not be shared whole:
        // its arrays that come out equal are shared when this entry's are made, then dropped.
        public JoinedReads? JoinedHint { get; set; }
        public SlotModel? SlotsHint { get; set; }
        // The last snapshots hits made for draws (HitSnapshot), each with the rebased words it applied;
        // NextHitSnapshot is the slot the next one replaces.
        public (ResourceSnapshot? Snapshot, ulong[]? Active)[]? HitSnapshots { get; set; }
        public int NextHitSnapshot { get; set; }
        public Entry? Next { get; set; }

        // The chain bytes (ChainBytes) this entry was stored with as a key's head.
        public long StoredChainBytes { get; set; }

        private long _estimatedBytes = -1;

        // An upper bound on what the entry keeps alive, from its immutable arrays. Arrays it shares
        // with the entry it was rebuilt from are counted in both. The derived data filled in later
        // (RangeWindows, BaseWordProbes, TableCopySources, Joined, Slots, HitSnapshots) is counted up
        // front from the sizes it is made of, so the estimate never changes.
        public long EstimatedBytes
        {
            get
            {
                if (_estimatedBytes >= 0)
                    return _estimatedBytes;
                var ranges = RangeAddresses.Length;
                var table = Snapshot.FlattenedResourceTable.Length;
                var size = EntryObjectBytes +
                    ArrayBytes(UserData.Length, sizeof(uint)) + ArrayBytes(UsedUserData.Length, sizeof(ulong)) +
                    ArrayBytes(ranges, sizeof(ulong)) + 2 * ArrayBytes(ranges, sizeof(int)) + ArrayBytes(ranges, sizeof(bool)) +
                    ArrayBytes(WordTableOnly.Length, sizeof(bool)) + ArrayBytes(Bytes.Length, sizeof(byte)) +
                    (BaseWordBuffers is null ? 0 : ArrayBytes(BaseWordBuffers.Length, sizeof(int))) +
                    SnapshotBytes(Snapshot) + SpecializationBytes(Specialization) +
                    // RangeWindows, BaseWordProbes, TableCopySources.
                    ArrayBytes(ranges, sizeof(int)) + ArrayBytes(ranges, 2 * sizeof(int)) + ArrayBytes(table, sizeof(int)) +
                    // JoinedReads: at most one read per range.
                    ObjectBytes + ArrayBytes(ranges, sizeof(ulong)) + 4 * ArrayBytes(ranges, sizeof(int)) + ArrayBytes(ranges, sizeof(bool)) +
                    // SlotModel: at most one slot per table word.
                    ObjectBytes + 2 * ArrayBytes(table, sizeof(ulong)) + 2 * ArrayBytes(table, sizeof(int)) +
                    // HitSnapshots: the slot array, and per slot a snapshot shell with its patched buffers
                    // and the user data array it keeps alive.
                    ArrayBytes(HitSnapshotCapacity, 2 * IntPtr.Size) + HitSnapshotCapacity *
                    (ObjectBytes + JaggedBytes(Snapshot.Buffers) + ArrayBytes(UserData.Length, sizeof(uint)));
                _estimatedBytes = size;
                return size;
            }
        }

        private const long ObjectBytes = 64;
        private const long EntryObjectBytes = 256;

        private static long ArrayBytes(int length, int elementBytes) => 24 + (long)length * elementBytes;

        private static long JaggedBytes(uint[][] arrays)
        {
            var size = ArrayBytes(arrays.Length, IntPtr.Size);
            foreach (var array in arrays)
                size += ArrayBytes(array.Length, sizeof(uint));
            return size;
        }

        private static long SnapshotBytes(ResourceSnapshot snapshot)
        {
            var size = ObjectBytes + JaggedBytes(snapshot.Buffers) + JaggedBytes(snapshot.Images) + JaggedBytes(snapshot.Samplers) +
                ArrayBytes(snapshot.FlattenedResourceTable.Length, sizeof(uint)) + ArrayBytes(snapshot.UserData.Length, sizeof(uint)) +
                ArrayBytes(snapshot.DeviceAddressRanges.Length, Unsafe.SizeOf<DeviceAddressRange>()) +
                ArrayBytes(snapshot.BindlessImages.Length, IntPtr.Size);
            foreach (var table in snapshot.BindlessImages)
            {
                size += ObjectBytes + ArrayBytes(table.VariantImages.Length, sizeof(uint)) + JaggedBytes(table.Descriptors) +
                    ArrayBytes(table.DescriptorVariants.Length, sizeof(byte)) + ArrayBytes(table.Entries.Length, sizeof(int));
            }

            return size;
        }

        private static long SpecializationBytes(ResourceSpecialization specialization) =>
            ObjectBytes + 6 * ObjectBytes +
            32L * (specialization.Buffers.Count + specialization.Images.Count + specialization.BufferCandidateTables.Count +
                specialization.ImageDescriptorGroups.Count + specialization.SamplerDescriptorGroups.Count + specialization.Samplers.Count);

        // A rebased word is an address the hit patches in, so it does not distinguish entries.
        public bool Matches(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, ulong[] rebased)
        {
            if (!ReferenceEquals(Plan, plan) || ShaderBase != inputs.ShaderBase ||
                !Nullable.Equals(ComputeState, inputs.ComputeState) || UserData.Length != inputs.UserData.Count)
                return false;
            for (var index = 0; index < UserData.Length; index++)
                if (IsUsed(UsedUserData, index) && !IsUsed(rebased, index) && UserData[index] != inputs.UserData[index])
                    return false;
            return true;
        }
    }

    // A user data list that notes which words are read by index. Copying it out (the snapshot's
    // own copy) goes through the enumerator and is not a read of the evaluator.
    private sealed class UserDataRecorder : IReadOnlyList<uint>
    {
        private IReadOnlyList<uint> _inner = [];
        private ulong[] _used = [];

        public void Begin(IReadOnlyList<uint> inner)
        {
            _inner = inner;
            _used = new ulong[(inner.Count + 63) >> 6];
        }

        // The recorded set; the next Begin allocates a fresh one, so the entry keeps this array.
        public ulong[] TakeUsed() => _used;

        public int Count => _inner.Count;

        public uint this[int index]
        {
            get
            {
                _used[index >> 6] |= 1UL << (index & 63);
                return _inner[index];
            }
        }

        public IEnumerator<uint> GetEnumerator() => _inner.GetEnumerator();

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private sealed class ReadRecorder
    {
        private readonly List<(ulong Address, uint Word, bool Clean, bool Table)> _reads = new();
        private bool _inTable;
        private GuestWordReader? _reader;
        private GuestWordReader? _cleanReader;
        private ResidentGuestBytesReader? _residentReader;
        private readonly GuestWordReader _recordRead;
        private readonly GuestWordReader _recordCleanRead;
        private readonly ResidentGuestBytesReader _recordResidentRead;

        public ReadRecorder()
        {
            _recordRead = (ulong address, out uint word) => Read(_reader!, address, out word, clean: false);
            _recordCleanRead = (ulong address, out uint word) => Read(_cleanReader!, address, out word, clean: true);
            _recordResidentRead = (ulong address, Span<byte> destination, bool clean) => ReadResident(address, destination, clean);
            TablePhase = inTable =>
            {
                _inTable = inTable;
                ReadingTable = inTable;
            };
        }

        public void Reset()
        {
            _reads.Clear();
            // Retain ordinary descriptor walks without keeping unusually large
            // tables alive for the lifetime of the renderer.
            if (_reads.Capacity > 16384)
                _reads.Capacity = 0;
            if (_sortScratch.Length > 16384) _sortScratch = [];
            _reader = null;
            _cleanReader = null;
            _residentReader = null;
            _inTable = false;
            Failed = false;
            if (_tableInputs is { } table)
                table.UserData = [];
        }

        public bool Failed { get; private set; }

        private readonly UserDataRecorder _userData = new();

        public IReadOnlyList<uint> WrapUserData(IReadOnlyList<uint> inner)
        {
            _userData.Begin(inner);
            return _userData;
        }

        public List<(ulong Address, uint Word, bool Clean, bool Table)> Reads => _reads;

        public void Record(ulong address, uint word, bool clean, bool table) => _reads.Add((address, word, clean, table));

        public Action<bool> TablePhase { get; }

        public GuestWordReader? Wrap(GuestWordReader? inner, bool clean)
        {
            if (inner is null)
                return null;
            if (clean)
            {
                _cleanReader = inner;
                return _recordCleanRead;
            }
            _reader = inner;
            return _recordRead;
        }

        private bool Read(GuestWordReader inner, ulong address, out uint word, bool clean)
        {
            if (!inner(address, out word))
            {
                Failed = true;
                return false;
            }
            using (ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.ReadRecording))
                _reads.Add((address, word, clean, _inTable));
            return true;
        }

        private ResourceRuntimeInputs? _tableInputs;

        // The inputs of a table evaluation that records through this recorder: the draw's, every
        // reader wrapped, no table phase. One instance serves every evaluation with the same readers,
        // refilled per draw; nothing keeps inputs past the call (ResourceRuntimeInputs).
        public ResourceRuntimeInputs TableInputs(ResourceRuntimeInputs inputs)
        {
            var read = Wrap(inputs.ReadMemory, clean: false);
            var clean = Wrap(inputs.ReadCleanMemory, clean: true);
            var resident = WrapResident(inputs.ReadResidentMemory);
            var table = _tableInputs;
            if (table is null || !ReferenceEquals(table.ReadMemory, read) || !ReferenceEquals(table.ReadCleanMemory, clean) ||
                !ReferenceEquals(table.ReadResidentMemory, resident) || table.ReadsClean != inputs.ReadsClean ||
                !ReferenceEquals(table.DataWritePending, inputs.DataWritePending))
            {
                _tableInputs = table = new ResourceRuntimeInputs
                {
                    ReadMemory = read,
                    ReadCleanMemory = clean,
                    ReadResidentMemory = resident,
                    ReadsClean = inputs.ReadsClean,
                    DataWritePending = inputs.DataWritePending,
                };
            }

            table.UserData = inputs.UserData;
            table.ShaderBase = inputs.ShaderBase;
            table.ComputeState = inputs.ComputeState;
            return table;
        }

        public ResidentGuestBytesReader? WrapResident(ResidentGuestBytesReader? inner)
        {
            if (inner is null)
                return null;
            _residentReader = inner;
            return _recordResidentRead;
        }

        private bool ReadResident(ulong address, Span<byte> destination, bool clean)
        {
            if (!_residentReader!(address, destination, clean))
                return false;

            using var profile = ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.ReadRecording);
            var wordCount = destination.Length / sizeof(uint);
            if (wordCount >= 8)
            {
                var first = _reads.Count;
                System.Runtime.InteropServices.CollectionsMarshal.SetCount(_reads, checked(first + wordCount));
                var recorded = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_reads).Slice(first, wordCount);
                for (var index = 0; index < wordCount; index++)
                    recorded[index] = (address + (ulong)index * sizeof(uint),
                        System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(destination[(index * sizeof(uint))..]), clean, _inTable);
                return true;
            }
            for (var offset = 0; offset + sizeof(uint) <= destination.Length; offset += sizeof(uint))
                _reads.Add((address + (ulong)offset,
                    System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(destination[offset..]), clean, _inTable));
            return true;
        }

        // The range layout of the last build, filled before its retained arrays are made: an entry
        // rebuilt from the same reads (a stale entry evaluated again, often every frame) shares the
        // arrays of the entry it replaces instead of allocating them again.
        private ulong[] _scratchAddresses = [];
        private int[] _scratchOffsets = [];
        private int[] _scratchLengths = [];
        private bool[] _scratchClean = [];
        private bool[] _scratchTableOnly = [];
        private byte[] _scratchBytes = [];
        private int[] _scratchBaseWords = [];

        private (ulong Address, uint Word, bool Clean, bool Table)[] _sortScratch = [];

        // Descriptor evaluation followed by table evaluation usually records two ordered runs.
        // Merge those runs in linear time; irregular graphs still use the general sort.
        private void SortReads()
        {
            var split = -1;
            for (var index = 1; index < _reads.Count; index++)
            {
                if (_reads[index - 1].Address <= _reads[index].Address) continue;
                if (split >= 0)
                {
                    ResourceMaterializationProfile.CountRead(ResourceMaterializationProfile.ReadCounter.SortedReadLists);
                    _reads.Sort(static (left, right) => left.Address.CompareTo(right.Address));
                    return;
                }
                split = index;
            }
            if (split < 0)
            {
                ResourceMaterializationProfile.CountRead(ResourceMaterializationProfile.ReadCounter.OrderedReadLists);
                return;
            }

            ResourceMaterializationProfile.CountRead(ResourceMaterializationProfile.ReadCounter.MergedReadLists);
            var reads = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_reads);
            if (_sortScratch.Length < reads.Length)
                _sortScratch = new (ulong, uint, bool, bool)[Math.Max(reads.Length, _sortScratch.Length * 2)];
            reads.CopyTo(_sortScratch);
            int left = 0, right = split, output = 0;
            while (left < split && right < reads.Length)
                reads[output++] = _sortScratch[left].Address <= _sortScratch[right].Address
                    ? _sortScratch[left++] : _sortScratch[right++];
            _sortScratch.AsSpan(left, split - left).CopyTo(reads[output..]);
            output += split - left;
            _sortScratch.AsSpan(right, reads.Length - right).CopyTo(reads[output..]);
        }

        public Entry Build(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, ResourceSnapshot snapshot,
            ResourceSpecialization specialization, IReadOnlyList<int> memoryBases, ulong[]? usedUserData = null,
            Entry? like = null)
        {
            using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.RecorderSort))
                SortReads();
            // Count unique words and contiguous ranges first, then fill the
            // retained arrays directly. Temporary lists previously duplicated
            // every recorded byte and range on each cache miss.
            var rangeCount = 0;
            var wordCount = 0;
            ulong end = 0;
            var previous = ulong.MaxValue;
            foreach (var read in _reads)
            {
                if (read.Address == previous) continue;
                if (wordCount == 0 || read.Address != end) rangeCount++;
                wordCount++;
                previous = read.Address;
                end = read.Address + sizeof(uint);
            }

            if (_scratchAddresses.Length < rangeCount)
            {
                var capacity = Math.Max(rangeCount, _scratchAddresses.Length * 2);
                _scratchAddresses = new ulong[capacity];
                _scratchOffsets = new int[capacity];
                _scratchLengths = new int[capacity];
                _scratchClean = new bool[capacity];
            }

            if (_scratchTableOnly.Length < wordCount)
                _scratchTableOnly = new bool[Math.Max(wordCount, _scratchTableOnly.Length * 2)];
            var addresses = _scratchAddresses.AsSpan(0, rangeCount);
            var offsets = _scratchOffsets.AsSpan(0, rangeCount);
            var lengths = _scratchLengths.AsSpan(0, rangeCount);
            var clean = _scratchClean.AsSpan(0, rangeCount);
            var tableOnly = _scratchTableOnly.AsSpan(0, wordCount);
            if (_scratchBytes.Length < wordCount * sizeof(uint))
                _scratchBytes = new byte[Math.Max(wordCount * sizeof(uint), _scratchBytes.Length * 2)];
            lengths.Clear();
            clean.Clear();
            var bytes = _scratchBytes.AsSpan(0, wordCount * sizeof(uint));
            var rangeIndex = -1;
            var wordIndex = -1;
            end = 0;
            previous = ulong.MaxValue;
            foreach (var (address, word, wordClean, table) in _reads)
            {
                if (address == previous)
                {
                    clean[rangeIndex] |= wordClean;
                    tableOnly[wordIndex] &= table;
                    continue;
                }

                wordIndex++;
                if (rangeIndex < 0 || address != end)
                {
                    rangeIndex++;
                    addresses[rangeIndex] = address;
                    offsets[rangeIndex] = wordIndex * sizeof(uint);
                }
                lengths[rangeIndex] += sizeof(uint);
                clean[rangeIndex] |= wordClean;
                tableOnly[wordIndex] = table;
                System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(
                    bytes[(wordIndex * sizeof(uint))..], word);
                previous = address;
                end = address + sizeof(uint);
            }

            // The entry's arrays are never written once built, so equal ones are shared. An entry
            // rebuilt every frame for a moved ring keeps its layout, its flags and often its words:
            // each array is shared on its own, not only when every range is where it was.
            var sameAddresses = like is not null && addresses.SequenceEqual(like.RangeAddresses);
            var sameLayout = like is not null && offsets.SequenceEqual(like.RangeOffsets) && lengths.SequenceEqual(like.RangeLengths);
            var sameClean = like is not null && clean.SequenceEqual(like.RangeClean);
            var sameRanges = sameAddresses && sameLayout && sameClean;
            var sameWords = like is not null && tableOnly.SequenceEqual(like.WordTableOnly);
            var sameBytes = like is not null && bytes.SequenceEqual(like.Bytes);
            var baseWords = memoryBases.Count == 0 ? null : MapBaseWords(snapshot, memoryBases, bytes, like?.BaseWordBuffers);
            var sameBaseWords = like is not null && ReferenceEquals(baseWords, like.BaseWordBuffers);
            // A draw's user data array is never written once made (ShaderSource.UserData): the entry
            // shares it, as the snapshot does.
            var userData = like is not null && like.UserData.Length == inputs.UserData.Count && SameWords(like.UserData, inputs.UserData)
                ? like.UserData
                : inputs.UserData as uint[] ?? CopyWords(inputs.UserData);
            // Derived from the layout and the base words alone (BaseWordProbesOf), and with the
            // addresses and flags from the ranges alone (JoinedReadsOf).
            var probes = sameLayout && sameBaseWords ? like!.BaseWordProbes : null;
            var joined = sameRanges && sameBaseWords ? like!.Joined : null;
            return new Entry
            {
                BaseWordBuffers = baseWords,
                Plan = plan,
                UserData = userData,
                UsedUserData = usedUserData ?? _userData.TakeUsed(),
                ShaderBase = inputs.ShaderBase,
                ComputeState = inputs.ComputeState,
                RangeAddresses = sameAddresses ? like!.RangeAddresses : addresses.ToArray(),
                RangeOffsets = sameLayout ? like!.RangeOffsets : offsets.ToArray(),
                RangeLengths = sameLayout ? like!.RangeLengths : lengths.ToArray(),
                RangeClean = sameClean ? like!.RangeClean : clean.ToArray(),
                WordTableOnly = sameWords ? like!.WordTableOnly : tableOnly.ToArray(),
                TableRefreshable = snapshot.FlattenedResourceTable.Length ==
                    plan.TableReads.Count + plan.WrittenRangeCount * ShaderResourcePlan.WrittenRangeDwordCount,
                Bytes = sameBytes ? like!.Bytes : bytes.ToArray(),
                Snapshot = snapshot,
                Specialization = specialization,
                Joined = joined,
                BaseWordProbes = probes,
                JoinedHint = joined is null ? like?.Joined ?? like?.JoinedHint : null,
                SlotsHint = like?.Slots is { } slots && !ReferenceEquals(slots, Unmodelled) ? slots : like?.SlotsHint,
            };
        }

        private static bool SameWords(uint[] words, IReadOnlyList<uint> other)
        {
            for (var index = 0; index < words.Length; index++)
                if (words[index] != other[index])
                    return false;
            return true;
        }

        private static uint[] CopyWords(IReadOnlyList<uint> words)
        {
            var copy = new uint[words.Count];
            for (var index = 0; index < copy.Length; index++)
                copy[index] = words[index];
            return copy;
        }

        // The recorded word each memory-loaded base dword came from: the only recorded word holding
        // that value. A word two bases claim names neither. Learning checks the guess. Built in
        // scratch: a map equal to like's is like's, and only a new one is allocated.
        private int[]? MapBaseWords(ResourceSnapshot snapshot, IReadOnlyList<int> memoryBases, ReadOnlySpan<byte> bytes, int[]? like)
        {
            var mapped = false;
            var wordCount = bytes.Length / sizeof(uint);
            if (_scratchBaseWords.Length < wordCount)
                _scratchBaseWords = new int[Math.Max(wordCount, _scratchBaseWords.Length * 2)];
            var map = _scratchBaseWords.AsSpan(0, wordCount);
            foreach (var buffer in memoryBases)
            {
                if (buffer >= snapshot.Buffers.Length || snapshot.Buffers[buffer].Length < 2)
                    continue;
                var value = snapshot.Buffers[buffer][0];
                var match = -1;
                for (var word = 0; word < wordCount; word++)
                {
                    if (System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes[(word * sizeof(uint))..]) != value)
                        continue;
                    if (match >= 0)
                    {
                        match = -1;
                        break;
                    }
                    match = word;
                }

                if (match < 0)
                    continue;
                if (!mapped)
                {
                    mapped = true;
                    map.Fill(NoBase);
                }

                map[match] = map[match] == NoBase ? buffer : SharedBase;
            }

            if (!mapped)
                return null;
            return like is not null && map.SequenceEqual(like) ? like : map.ToArray();
        }
    }

    private const int NoBase = -1;
    private const int SharedBase = -2;
}
