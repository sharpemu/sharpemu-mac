// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

/// <summary>
/// sceAgcCreatePrimState's batched path (one read per header and specials block, one write of both
/// register blocks) against its field-by-field path, on random shader structures: the same result,
/// the same RAX and the same bytes, including partial writes and faults next to the end of mapped
/// memory, destinations that overlap what is read, and shaders of the wrong type.
/// </summary>
public sealed class AgcPrimStateBatchedTests
{
    private const ulong BaseAddress = 0x1_0000_0000;
    private const int RegionBytes = 0x2000;

    [Fact]
    public void TheBatchedPathAnswersAndWritesWhatTheFieldByFieldPathDoes()
    {
        var random = new System.Random(0x9817);
        var batchedRan = 0;
        var outcomes = new Dictionary<int, int>();
        for (var iteration = 0; iteration < 6000; iteration++)
        {
            var seed = random.Next();
            var fill = new System.Random(seed);
            var bytes = new byte[RegionBytes];
            fill.NextBytes(bytes);

            // An address in the region (anywhere, or flush against its end so only part of a block
            // reads), null, or outside it.
            ulong Place(int size) => random.Next(10) switch
            {
                0 => 0,
                1 => BaseAddress + (ulong)(RegionBytes - random.Next(1, size + 1)),
                2 => BaseAddress + RegionBytes + 0x1000,
                _ => BaseAddress + (ulong)(random.Next(0, (RegionBytes - size) / 4) * 4),
            };

            void Put(ulong address, ReadOnlySpan<byte> value)
            {
                var offset = address - BaseAddress;
                if (address >= BaseAddress && offset + (ulong)value.Length <= RegionBytes)
                {
                    value.CopyTo(bytes.AsSpan((int)offset));
                }
            }

            var geometry = random.Next(12) == 0 ? 0 : Place(0x60);
            var hull = random.Next(2) == 0 ? 0 : Place(0x60);
            var specials = random.Next(12) == 0 ? 0 : Place(0x30);
            var hullSpecials = random.Next(12) == 0 ? 0 : Place(0x30);
            var cx = random.Next(5) == 0 ? 0 : Place(16);
            var uc = random.Next(5) == 0 ? 0 : Place(24);
            // Now and then a destination right on top of something the export reads.
            if (random.Next(10) == 0)
            {
                var targets = new[] { geometry + 0x28, specials, specials + 0x08, hullSpecials + 0x20, hull + 0x50 };
                if (random.Next(2) == 0) cx = targets[random.Next(targets.Length)];
                else uc = targets[random.Next(targets.Length)];
            }

            Put(geometry + 0x28, BitConverter.GetBytes(specials));
            Put(geometry + 0x5A, [random.Next(5) switch { 0 => (byte)6, 1 => (byte)random.Next(256), _ => (byte)2 }]);
            Put(hull + 0x28, BitConverter.GetBytes(hullSpecials));
            Put(hull + 0x5A, [random.Next(5) == 0 ? (byte)random.Next(256) : (byte)3]);
            // The stages value's GS enable bit (bit 5) both ways in each block.
            Put(specials + 0x0C, BitConverter.GetBytes((uint)random.Next() & ~0x20u | (random.Next(2) == 0 ? 0x20u : 0)));
            Put(hullSpecials + 0x0C, BitConverter.GetBytes((uint)random.Next() & ~0x20u | (random.Next(3) == 0 ? 0x20u : 0)));

            var primitive = random.Next(4) == 0 ? (uint)random.Next() : (uint)random.Next(0x20);
            (int Result, ulong Rax, byte[] Memory) Run(Func<CpuContext, int> body)
            {
                var memory = new FakeCpuMemory(BaseAddress, RegionBytes);
                Assert.True(memory.TryWrite(BaseAddress, bytes));
                var ctx = new CpuContext(memory, Generation.Gen5);
                ctx[CpuRegister.Rdi] = cx;
                ctx[CpuRegister.Rsi] = uc;
                ctx[CpuRegister.Rdx] = hull;
                ctx[CpuRegister.Rcx] = geometry;
                ctx[CpuRegister.R8] = primitive;
                var result = body(ctx);
                var after = new byte[RegionBytes];
                Assert.True(memory.TryRead(BaseAddress, after));
                return (result, ctx[CpuRegister.Rax], after);
            }

            var legacy = Run(ctx => AgcExports.CreatePrimStateCore(ctx, batched: false));
            var batched = Run(ctx => AgcExports.CreatePrimStateCore(ctx, batched: true));
            var context = $"iteration {iteration}: cx=0x{cx:X} uc=0x{uc:X} hs=0x{hull:X} gs=0x{geometry:X} specials=0x{specials:X} hullSpecials=0x{hullSpecials:X}";
            Assert.True(legacy.Result == batched.Result, context);
            Assert.True(legacy.Rax == batched.Rax, context);
            Assert.True(legacy.Memory.AsSpan().SequenceEqual(batched.Memory), context);
            outcomes[legacy.Result] = outcomes.GetValueOrDefault(legacy.Result) + 1;

            // How often the batched path itself answered (the rest fell back).
            if (cx != 0 || uc != 0)
            {
                var memory = new FakeCpuMemory(BaseAddress, RegionBytes);
                memory.TryWrite(BaseAddress, bytes);
                if (geometry != 0 && AgcExports.TryCreatePrimStateBatched(new CpuContext(memory, Generation.Gen5), cx, uc, hull, geometry, primitive, out _))
                {
                    batchedRan++;
                }
            }
        }

        Assert.True(batchedRan > 500, $"the batched path answered only {batchedRan} times");
        Assert.True(outcomes.Count >= 3, "outcomes: " + string.Join(", ", outcomes.Select(pair => $"0x{pair.Key:X}={pair.Value}")));
    }
}
