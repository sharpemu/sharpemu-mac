// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.RegularExpressions;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

[CollectionDefinition(nameof(ReadProofModeCollection), DisableParallelization = true)]
public sealed class ReadProofModeCollection;

// A fast snapshot's revalidation proved from what the evaluation that made it read (ResourceMaterializer.ReadProofs):
// current without reading a word one by one while every word compares equal; evaluated in full when one changed, is
// owned, an unreadable one turned readable, or the proof is refused; verify stops on a wrong proof.
[Collection(nameof(ReadProofModeCollection))]
public sealed class ReadProofTests
{
    // s[0:1] points at a V# the program loads into s[4:7] and reads through: the V# words are the proof's block.
    private static ShaderResourcePlan PointerPlan()
    {
        var plan = ShaderResourcePlan.Extract(Program(ScalarLoad(0, 0, 4, 4), BufferLoad(8, 4), EndProgram(16)), ShaderStage.Compute, Hash, 0, 4);
        Assert.True(ResourceMaterializer.CanMaterializeFast(plan));
        return plan;
    }

    // A host comparer over a test memory: refuses every range while Owned (as a pending GPU write would), compares the
    // words otherwise; counts its calls and the ranges it was asked about.
    private sealed class Comparer(TestWordMemory memory)
    {
        public bool Owned;
        public int Calls;
        public readonly List<(ulong Address, int Length)> Ranges = [];

        public bool Compare(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<byte> recorded)
        {
            Calls++;
            Ranges.Clear();
            var offset = 0;
            for (var index = 0; index < addresses.Length; index++)
            {
                Ranges.Add((addresses[index], lengths[index]));
                if (Owned)
                {
                    return false;
                }

                for (var word = 0; word < lengths[index] / sizeof(uint); word++)
                {
                    var address = addresses[index] + (ulong)(word * sizeof(uint));
                    if (address < memory.Base || address - memory.Base >= (ulong)memory.Words.Length * sizeof(uint) ||
                        memory.At(address) != System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(recorded[(offset + word * sizeof(uint))..]))
                    {
                        return false;
                    }
                }

                offset += lengths[index];
            }

            return true;
        }
    }

    private static (ShaderResourcePlan Plan, TestWordMemory Memory, Comparer Comparer, ResourceRuntimeInputs Inputs) Setup()
    {
        var plan = PointerPlan();
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[16] };
        uint[] descriptor = [0x8000, 0x10 << 16, 64, 0x24FAC];
        descriptor.CopyTo(memory.Words, 0);
        descriptor.CopyTo(memory.Words, 8);
        var comparer = new Comparer(memory);
        var inputs = new ResourceRuntimeInputs { ReadMemory = memory.Read, CompareGuestWords = comparer.Compare };
        inputs.UserData = new uint[] { 0x1000, 0, 0, 0 };
        return (plan, memory, comparer, inputs);
    }

    private static ResourceSnapshot Materialize(ShaderResourcePlan plan, ResourceRuntimeInputs inputs)
    {
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.MaterializeFast(plan, inputs, ref snapshot, ref specialization, out _));
        return snapshot;
    }

    private static long Counter(string report, string name) => long.Parse(Regex.Match(report, $@"[(,]{name}=(\d+)").Groups[1].Value);

    private static T WithMode<T>(ResourceMaterializer.ReadProofMode mode, Func<T> body)
    {
        var previous = ResourceMaterializer.ReadProofs;
        ResourceMaterializer.ReadProofs = mode;
        try
        {
            return body();
        }
        finally
        {
            ResourceMaterializer.ReadProofs = previous;
        }
    }

    [Fact]
    public void AnUnchangedBlockProvesTheSnapshotCurrentWithoutReadingAWord()
    {
        WithMode(ResourceMaterializer.ReadProofMode.On, () =>
        {
            var (plan, memory, comparer, inputs) = Setup();
            var snapshot = Materialize(plan, inputs);
            var proof = ResourceMaterializer.ReadProofOf(snapshot);
            Assert.NotNull(proof);
            // The four V# words, one range in one line.
            Assert.Equal([0x1000ul], proof!.Addresses);
            Assert.Equal([16], proof.Lengths);

            _ = ResourceMaterializer.TakeReadProofReport();
            var reads = memory.Reads;
            Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.Equal(ResourceMaterializer.FastRevalidation.Current, ResourceMaterializer.RevalidateOrRebuildFast(snapshot, out var rebuilt));
            Assert.Null(rebuilt);
            Assert.Equal(reads, memory.Reads);
            Assert.Equal(2, comparer.Calls);
            Assert.Equal(2, Counter(ResourceMaterializer.TakeReadProofReport(), "hits"));
            return 0;
        });
    }

    [Fact]
    public void AWordChangedInsideTheBlockIsNotProvedAndTheRebuildCarriesANewProof()
    {
        WithMode(ResourceMaterializer.ReadProofMode.On, () =>
        {
            var (plan, memory, _, inputs) = Setup();
            var snapshot = Materialize(plan, inputs);
            _ = ResourceMaterializer.TakeReadProofReport();

            // The V#'s record count: inside the compared block.
            memory.Words[2] = 128;
            var reads = memory.Reads;
            Assert.False(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.True(memory.Reads > reads);
            var report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal(0, Counter(report, "hits"));
            Assert.Equal(1, Counter(report, "changed"));

            // The rebuilt snapshot's proof is the words read now: it proves the next revalidation.
            Assert.Equal(ResourceMaterializer.FastRevalidation.Rebuilt, ResourceMaterializer.RevalidateOrRebuildFast(snapshot, out var rebuilt));
            Assert.NotNull(ResourceMaterializer.ReadProofOf(rebuilt!));
            reads = memory.Reads;
            Assert.True(ResourceMaterializer.RevalidateFast(rebuilt!));
            Assert.Equal(reads, memory.Reads);
            return 0;
        });
    }

    [Fact]
    public void AWordChangedOutsideTheBlockStillProves()
    {
        WithMode(ResourceMaterializer.ReadProofMode.On, () =>
        {
            var (plan, memory, _, inputs) = Setup();
            var snapshot = Materialize(plan, inputs);
            // The words right after the V# and the other V#: never read by this draw.
            memory.Words[4] = 0xDEAD;
            memory.Words[8] = 0x9000;
            var reads = memory.Reads;
            Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.Equal(reads, memory.Reads);
            return 0;
        });
    }

    [Fact]
    public void AnOwnedBlockRefusesTheProofAndReadsEveryWord()
    {
        WithMode(ResourceMaterializer.ReadProofMode.On, () =>
        {
            var (plan, memory, comparer, inputs) = Setup();
            var snapshot = Materialize(plan, inputs);
            _ = ResourceMaterializer.TakeReadProofReport();
            comparer.Owned = true;
            var reads = memory.Reads;
            Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.True(memory.Reads > reads);
            var report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal(0, Counter(report, "hits"));
            Assert.Equal(1, Counter(report, "refused"));

            // Owned and changed: still read in full, and found stale.
            memory.Words[0] = 0x9000;
            Assert.False(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.Equal(1, Counter(ResourceMaterializer.TakeReadProofReport(), "changed"));
            return 0;
        });
    }

    [Fact]
    public void AnUnreadableWordThatTurnsReadableIsNotProved()
    {
        WithMode(ResourceMaterializer.ReadProofMode.On, () =>
        {
            var (plan, memory, comparer, _) = Setup();
            // A null table pointer: its V# reads as nothing (an unbound buffer) while the low page is unmapped.
            var lowMapped = false;
            GuestWordReader read = (ulong address, out uint word) =>
            {
                word = 0x40;
                return address >= memory.Base ? memory.Read(address, out word) : lowMapped;
            };
            var inputs = new ResourceRuntimeInputs { ReadMemory = read, CompareGuestWords = comparer.Compare, UserData = new uint[] { 0, 0, 0, 0 } };
            var snapshot = Materialize(plan, inputs);
            var proof = ResourceMaterializer.ReadProofOf(snapshot);
            Assert.NotNull(proof);
            Assert.Equal([0ul, 4ul, 8ul, 12ul], proof!.Unreadable);
            Assert.Empty(proof.Addresses);
            Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.Equal(ResourceMaterializer.FastRevalidation.Current, ResourceMaterializer.RevalidateOrRebuildFast(snapshot, out _));

            // Mapped now: the V# reads as words, not nothing.
            lowMapped = true;
            _ = ResourceMaterializer.TakeReadProofReport();
            Assert.False(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.Equal(1, Counter(ResourceMaterializer.TakeReadProofReport(), "changed"));
            return 0;
        });
    }

    [Fact]
    public void WithNoComparerOrSwitchedOffEveryRevalidationReads()
    {
        var (plan, memory, comparer, inputs) = Setup();
        WithMode(ResourceMaterializer.ReadProofMode.Off, () =>
        {
            var snapshot = Materialize(plan, inputs);
            Assert.Null(ResourceMaterializer.ReadProofOf(snapshot));
            var reads = memory.Reads;
            Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.True(memory.Reads > reads);
            Assert.Equal(0, comparer.Calls);
            return 0;
        });

        WithMode(ResourceMaterializer.ReadProofMode.On, () =>
        {
            var bare = new ResourceRuntimeInputs { ReadMemory = memory.Read, UserData = new uint[] { 0x1000, 0, 0, 0 } };
            var snapshot = Materialize(plan, bare);
            _ = ResourceMaterializer.TakeReadProofReport();
            var reads = memory.Reads;
            Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.True(memory.Reads > reads);
            Assert.Equal(1, Counter(ResourceMaterializer.TakeReadProofReport(), "no_comparer"));
            return 0;
        });
    }

    // A proof that compares equal but does not stand for the snapshot (another draw's, attached by hand) answers current
    // when it should not; verify evaluates in full beside it and stops.
    [Fact]
    public void VerifyStopsOnAWrongProof()
    {
        var (plan, memory, _, inputs) = Setup();
        var kept = WithMode(ResourceMaterializer.ReadProofMode.On, () => Materialize(plan, inputs));
        // Another draw through the other V#, whose proof covers words 8-11.
        inputs.UserData = new uint[] { 0x1020, 0, 0, 0 };
        var other = WithMode(ResourceMaterializer.ReadProofMode.On, () => Materialize(plan, inputs));
        // The kept draw's own V# changes: it is stale, but the other draw's proof still compares equal.
        memory.Words[0] = 0x9000;
        ResourceMaterializer.SetReadProofForTest(kept, ResourceMaterializer.ReadProofOf(other));

        Assert.True(WithMode(ResourceMaterializer.ReadProofMode.On, () => ResourceMaterializer.RevalidateFast(kept)));
        Assert.Throws<ResourceMaterializer.ReadProofMismatchException>(() =>
            WithMode(ResourceMaterializer.ReadProofMode.Verify, () => ResourceMaterializer.RevalidateFast(kept)));
        Assert.Throws<ResourceMaterializer.ReadProofMismatchException>(() =>
            WithMode(ResourceMaterializer.ReadProofMode.Verify, () => ResourceMaterializer.RevalidateOrRebuildFast(kept, out _)));

        // A right proof verifies.
        _ = ResourceMaterializer.TakeReadProofReport();
        Assert.True(WithMode(ResourceMaterializer.ReadProofMode.Verify, () => ResourceMaterializer.RevalidateFast(other)));
        Assert.Equal(1, Counter(ResourceMaterializer.TakeReadProofReport(), "verified"));
    }

    // The proof's words: sorted, repeats folded, runs of consecutive words split where they leave a 256-byte line; one
    // address read as two values refuses the proof.
    [Fact]
    public void ProofRangesStayInsideLinesAndATornReadIsRefused()
    {
        var plan = PointerPlan();
        var program = plan.MaterializationProgram!;
        var scratch = new MaterializationProgram.Scratch();
        foreach (var (address, word) in new (ulong, uint)[] { (0x20F8, 3), (0x20F0, 1), (0x20F4, 2), (0x20FC, 4), (0x2100, 5), (0x2104, 6), (0x20F4, 2), (0x3000, 7) })
        {
            scratch.NoteRead(address, word);
        }

        var proof = MaterializationProgram.ReadProof.Build(program, scratch, []);
        Assert.NotNull(proof);
        Assert.Equal([0x20F0ul, 0x2100ul, 0x3000ul], proof!.Addresses);
        Assert.Equal([16, 8, 4], proof.Lengths);
        Assert.Equal(Enumerable.Range(1, 7).SelectMany(word => BitConverter.GetBytes((uint)word)).ToArray(), proof.Bytes);

        scratch.ReadCount = 0;
        scratch.NoteRead(0x2000, 1);
        scratch.NoteRead(0x2000, 2);
        Assert.Null(MaterializationProgram.ReadProof.Build(program, scratch, []));
    }

    private static T WithBackoff<T>(bool backoff, Func<T> body)
    {
        var previous = ResourceMaterializer.ReadProofBackoff;
        ResourceMaterializer.ReadProofBackoff = backoff;
        try
        {
            return WithMode(ResourceMaterializer.ReadProofMode.On, body);
        }
        finally
        {
            ResourceMaterializer.ReadProofBackoff = previous;
        }
    }

    // A rebuild of the snapshot after the V#'s record count changed to `records`: the rebuilt snapshot.
    private static ResourceSnapshot RebuildAfterChange(TestWordMemory memory, ResourceSnapshot snapshot, uint records)
    {
        memory.Words[2] = records;
        Assert.Equal(ResourceMaterializer.FastRevalidation.Rebuilt, ResourceMaterializer.RevalidateOrRebuildFast(snapshot, out var rebuilt));
        return rebuilt!;
    }

    // Words that change on every rebuild back the origin's chain off proofs: after two changes in a row its revalidations
    // neither compare nor record one, and still answer as a full evaluation does; one that finds the snapshot current
    // ends the back-off, and the next records a proof that proves the one after.
    [Fact]
    public void AnOriginWhoseWordsChangeEveryRebuildBacksOffUntilItIsCurrentAgain()
    {
        WithBackoff(true, () =>
        {
            var (plan, memory, comparer, inputs) = Setup();
            var snapshot = Materialize(plan, inputs);
            _ = ResourceMaterializer.TakeReadProofReport();
            snapshot = RebuildAfterChange(memory, snapshot, 128);
            Assert.NotNull(ResourceMaterializer.ReadProofOf(snapshot));
            snapshot = RebuildAfterChange(memory, snapshot, 256);
            // Backed off: the rebuilt snapshot holds no proof.
            Assert.Null(ResourceMaterializer.ReadProofOf(snapshot));
            var report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal((2L, 1L), (Counter(report, "changed"), Counter(report, "backoffs")));

            var calls = comparer.Calls;
            snapshot = RebuildAfterChange(memory, snapshot, 64);
            Assert.Null(ResourceMaterializer.ReadProofOf(snapshot));
            Assert.Equal(ResourceMaterializer.FastRevalidation.Current, ResourceMaterializer.RevalidateOrRebuildFast(snapshot, out _));
            Assert.Equal(calls, comparer.Calls);
            report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal((2L, 0L, 0L), (Counter(report, "backed_off"), Counter(report, "changed"), Counter(report, "recorded")));

            // Re-armed: recorded again, then proved without reading a word.
            Assert.Equal(ResourceMaterializer.FastRevalidation.Current, ResourceMaterializer.RevalidateOrRebuildFast(snapshot, out _));
            Assert.NotNull(ResourceMaterializer.ReadProofOf(snapshot));
            var reads = memory.Reads;
            Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
            Assert.Equal(reads, memory.Reads);
            report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal((1L, 1L, 0L), (Counter(report, "none"), Counter(report, "hits"), Counter(report, "backed_off")));
            return 0;
        });
    }

    // The back-off lasts BackoffRevalidations revalidations whatever the words do, then compares and records again.
    [Fact]
    public void TheBackoffEndsAfterItsRevalidations()
    {
        WithBackoff(true, () =>
        {
            var (plan, memory, _, inputs) = Setup();
            var snapshot = Materialize(plan, inputs);
            snapshot = RebuildAfterChange(memory, snapshot, 128);
            snapshot = RebuildAfterChange(memory, snapshot, 256);
            _ = ResourceMaterializer.TakeReadProofReport();
            for (var change = 0; change < ResourceMaterializer.BackoffRevalidations; change++)
            {
                snapshot = RebuildAfterChange(memory, snapshot, (uint)(change % 2 == 0 ? 64 : 32));
            }

            var report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal((ResourceMaterializer.BackoffRevalidations, 0L), (Counter(report, "backed_off"), Counter(report, "changed")));
            snapshot = RebuildAfterChange(memory, snapshot, 16);
            report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal((0L, 1L), (Counter(report, "backed_off"), Counter(report, "none")));
            Assert.NotNull(ResourceMaterializer.ReadProofOf(snapshot));
            return 0;
        });
    }

    // A change, a proved revalidation, a change: not in a row, so no back-off. With the switch off, none ever.
    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void OnlyChangesInARowBackOffAndOnlyWithTheSwitchOn(bool backoff, bool provedBetween)
    {
        WithBackoff(backoff, () =>
        {
            var (plan, memory, _, inputs) = Setup();
            var snapshot = Materialize(plan, inputs);
            _ = ResourceMaterializer.TakeReadProofReport();
            for (var change = 0; change < 4; change++)
            {
                snapshot = RebuildAfterChange(memory, snapshot, (uint)(change % 2 == 0 ? 128 : 256));
                if (provedBetween)
                {
                    Assert.True(ResourceMaterializer.RevalidateFast(snapshot));
                }
            }

            var report = ResourceMaterializer.TakeReadProofReport();
            Assert.Equal((4L, 0L, 0L), (Counter(report, "changed"), Counter(report, "backoffs"), Counter(report, "backed_off")));
            Assert.NotNull(ResourceMaterializer.ReadProofOf(snapshot));
            return 0;
        });
    }
}
