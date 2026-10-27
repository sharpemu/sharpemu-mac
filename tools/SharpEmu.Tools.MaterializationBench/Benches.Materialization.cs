// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.PerfAudit.Materialization;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using static SharpEmu.PerfAudit.Materialization.ResourceTestProgram;

namespace SharpEmu.PerfAudit;

/// <summary>
/// ResourceMaterializer / ResourceMaterializationCache over a synthetic shader that reads 8 T# + 2 S# + 4 V#
/// through a descriptor table in guest memory (user data = table pointer), with entirely fabricated descriptor words.
/// </summary>
internal static unsafe class MaterializationBenches
{
    private const int Textures = 8, Samplers = 2, Buffers = 4;
    private const ulong TableStride = 512;
    private const ulong TextureRegion = NativeGuestMemory.Base + 0x100_0000;
    private const ulong BufferRegion = NativeGuestMemory.Base + 0x200_0000;
    private const ulong TableRegion = NativeGuestMemory.Base + 0x300_0000;

    public static ShaderResourcePlan BuildPlan()
    {
        var program = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        // s[8+8i : 15+8i] = T#i, s[72+4j : 75+4j] = S#j, s[80+4k : 83+4k] = V#k, all loaded through the pointer in s[0:1].
        for (uint i = 0; i < Textures; i++) { program.Add(ScalarLoad(pc, 0, 8 + 8 * i, 8, (int)(i * 32))); pc += 8; }
        for (uint j = 0; j < Samplers; j++) { program.Add(ScalarLoad(pc, 0, 72 + 4 * j, 4, 256 + (int)(j * 16))); pc += 8; }
        for (uint k = 0; k < Buffers; k++) { program.Add(ScalarLoad(pc, 0, 80 + 4 * k, 4, 288 + (int)(k * 16))); pc += 8; }
        for (uint i = 0; i < Textures; i++) { program.Add(Image(pc, "ImageSample", 8 + 8 * i, 72 + 4 * (i % Samplers))); pc += 8; }
        for (uint k = 0; k < Buffers; k++) { program.Add(BufferLoad(pc, 80 + 4 * k, 0, 4)); pc += 8; }
        program.Add(EndProgram(pc));
        return Extract(Program([.. program]), userDataBase: 0, userDataCount: 2, stage: ShaderStage.Pixel);
    }

    /// <summary>Writes one descriptor table; <paramref name="salt"/> makes its texture/buffer addresses distinct.</summary>
    private static void WriteTable(NativeGuestMemory memory, ulong table, int salt)
    {
        for (var i = 0; i < Textures; i++)
        {
            var words = Descriptors.Texture(TextureRegion + ((ulong)salt * 16 + (ulong)i) * 0x1_0000, 56, 256, 256);
            memory.WriteWords(table + (ulong)i * 32, words);
        }

        for (var j = 0; j < Samplers; j++) memory.WriteWords(table + 256 + (ulong)j * 16, Descriptors.Sampler());
        for (var k = 0; k < Buffers; k++)
            memory.WriteWords(table + 288 + (ulong)k * 16, Descriptors.Buffer(BufferRegion + ((ulong)salt * 8 + (ulong)k) * 0x1000, 16, 256, format: 77));
    }

    // Historical control assemblies predate the admission counter.
    private static long ColdBypasses(ResourceMaterializationCache cache) =>
        (long?)typeof(ResourceMaterializationCache).GetProperty("ColdBypasses")?.GetValue(cache) ?? 0;

    public static void Run()
    {
        Console.WriteLine("--- 3. Resource materialization (ResourceMaterializer / ResourceMaterializationCache) ---");
        var plan = BuildPlan();
        Console.WriteLine($"plan: images={plan.Info.Images.Count} buffers={plan.Info.Buffers.Count} samplers={plan.Info.Samplers.Count} descriptorSources={plan.DescriptorSources.Count} memoryOps={plan.Memory.Count}");
        using var memory = new NativeGuestMemory(64 << 20);
        const int Pool = 4096;
        for (var t = 0; t < Pool; t++) WriteTable(memory, TableRegion + (ulong)t * TableStride, t);

        bool ReadWord(ulong address, out uint word)
        {
            Span<byte> bytes = stackalloc byte[4];
            if (!memory.TryRead(address, bytes)) { word = 0; return false; }
            word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
            return true;
        }

        bool ReadResident(ulong address, Span<byte> destination, bool clean) => memory.TryRead(address, destination);
        // Reuse the capturing delegate as the renderer does. Re-converting the local
        // function at every call measures a harness allocation on each cache hit.
        ResidentGuestBytesReader residentReader = ReadResident;

        bool CompareResident(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths,
            ReadOnlySpan<int> offsets, ReadOnlySpan<byte> recorded)
        {
            if (addresses.Length != lengths.Length || addresses.Length != offsets.Length) return false;
            for (var index = 0; index < addresses.Length; index++)
            {
                var address = addresses[index];
                var length = lengths[index];
                var offset = offsets[index];
                if (length < 0 || offset < 0 || offset > recorded.Length - length ||
                    address < NativeGuestMemory.Base || address - NativeGuestMemory.Base > memory.Size ||
                    (ulong)length > memory.Size - (address - NativeGuestMemory.Base)) return false;
                if (!new ReadOnlySpan<byte>(memory.Raw(address), length).SequenceEqual(recorded.Slice(offset, length)))
                    return false;
            }
            return true;
        }

        var compareResident = Environment.GetEnvironmentVariable("PA_COMPARE_RESIDENT") == "1";
        Console.WriteLine($"resident_comparer={compareResident} (synthetic fixed native allocation; no Metal ownership checks)");
        var inputs = new ResourceRuntimeInputs
        {
            ReadMemory = ReadWord, ReadCleanMemory = ReadWord, ReadResidentMemory = residentReader,
            CompareResidentMemory = compareResident ? CompareResident : null,
        };
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var table0 = TableRegion;
        inputs.UserData = new uint[] { (uint)table0, (uint)(table0 >> 32) };
        const int Batch = 200;

        // 3a: the full walk, no cache, same descriptors.
        Harness.Measure("3a", "FULL WALK ResourceMaterializer.Materialize (no cache), 8T+2S+4V via table", Batch, 1404, _ =>
        {
            for (var i = 0; i < Batch; i++)
            {
                snapshot = new ResourceSnapshot(); specialization = new ResourceSpecialization();
                if (!ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization, out var failure)) throw new InvalidOperationException(failure.ToString());
            }
        }, unit: "full walk", note: "uncached control");

        // 3b: warm cache hit, same descriptors every draw.
        var cache = new ResourceMaterializationCache();
        if (!cache.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out var warmFailure)) throw new InvalidOperationException(warmFailure.ToString());
        var hitsBefore = cache.Hits;
        var warm = Harness.Measure("3b", "WARM cache hit, same descriptors every call (validation re-read)", Batch * 5, 13825, _ =>
        {
            for (var i = 0; i < Batch * 5; i++)
            {
                if (!cache.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out var failure)) throw new InvalidOperationException(failure.ToString());
            }
        }, unit: "cache-hit call", note: "validation re-read");
        Console.WriteLine($"       -> cache hits {cache.Hits - hitsBefore}, misses {cache.Misses}, refreshes {cache.TableRefreshes}");

        // 3c: cold, new table each call (new key), cache too small to retain: every call is a miss + entry build.
        var cold = new ResourceMaterializationCache(generationCapacity: 8);
        var cursor = 0;
        Harness.Measure("3c", "COLD miss: new table address + new descriptors each call (key not found)", Batch, 1404, _ =>
        {
            for (var i = 0; i < Batch; i++)
            {
                var table = TableRegion + (ulong)(cursor++ % Pool) * TableStride;
                inputs.UserData = new uint[] { (uint)table, (uint)(table >> 32) };
                if (!cold.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out var failure)) throw new InvalidOperationException(failure.ToString());
            }
        }, unit: "miss", note: "cold: includes allocating the user-data array and the entry build");
        Console.WriteLine($"       -> hits {cold.Hits} misses {cold.Misses} refreshes {cold.TableRefreshes} uncacheable {cold.Uncacheable} cold_bypasses {ColdBypasses(cold)}");

        // 3d: same key, the table's descriptors change between calls (stale same-key): tries refresh, falls back to the walk.
        var stale = new ResourceMaterializationCache();
        inputs.UserData = new uint[] { (uint)table0, (uint)(table0 >> 32) };
        stale.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out _);
        var salt = 0;
        var refreshesBefore = stale.TableRefreshes;
        Harness.Measure("3d", "STALE same key: descriptor words change every call (refresh or re-walk)", Batch, 1404, _ =>
        {
            for (var i = 0; i < Batch; i++)
            {
                WriteTable(memory, table0, 1 + (salt++ % (Pool - 1)));
                if (!stale.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out var failure)) throw new InvalidOperationException(failure.ToString());
            }
        }, unit: "stale", note: "includes WriteTable (14 small writes) in the timed loop");
        Console.WriteLine($"       -> hits {stale.Hits} misses {stale.Misses} refreshes {stale.TableRefreshes - refreshesBefore} (slot refreshes {stale.SlotRefreshes}, cold_bypasses {ColdBypasses(stale)})");

        // 3f: 50 draws per batch alternate between 8 descriptor tables at different addresses whose contents are identical:
        // only the table pointer in user data differs, so every draw could reuse one materialization.
        {
            const int Tables = 8, Draws = 50;
            var identical = new ulong[Tables];
            for (var t = 0; t < Tables; t++)
            {
                identical[t] = TableRegion + (ulong)(Pool + 8 + t * 3) * TableStride;
                WriteTable(memory, identical[t], 7);
            }

            var userData = new uint[Tables][];
            for (var t = 0; t < Tables; t++) userData[t] = new uint[] { (uint)identical[t], (uint)(identical[t] >> 32) };
            var alternating = new ResourceMaterializationCache();
            for (var i = 0; i < Draws; i++)
            {
                inputs.UserData = userData[i % Tables];
                if (!alternating.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out var failure)) throw new InvalidOperationException(failure.ToString());
            }
            Console.WriteLine($"       3f first {Draws} draws (cold cache): hits {alternating.Hits} misses {alternating.Misses} refreshes {alternating.TableRefreshes} -> hit rate {100.0 * alternating.Hits / Draws:F1}%");
            var hits0 = alternating.Hits; var misses0 = alternating.Misses; var refreshes0 = alternating.TableRefreshes;
            var draw = 0;
            Harness.Measure("3f", "8 tables at different addresses, identical contents, 50 draws alternating (pointer-only change)", Draws, 1404, _ =>
            {
                for (var i = 0; i < Draws; i++)
                {
                    inputs.UserData = userData[draw++ % Tables];
                    if (!alternating.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out var failure)) throw new InvalidOperationException(failure.ToString());
                }
            }, unit: "draw");
            var calls = (alternating.Hits - hits0) + (alternating.Misses - misses0) + (alternating.TableRefreshes - refreshes0);
            Console.WriteLine($"       3f steady: hits {alternating.Hits - hits0} misses {alternating.Misses - misses0} refreshes {alternating.TableRefreshes - refreshes0} -> hit rate {100.0 * (alternating.Hits - hits0) / Math.Max(1, calls):F1}%");
            inputs.UserData = new uint[] { (uint)table0, (uint)(table0 >> 32) };
        }

        // 3g: a per-frame descriptor ring. One shader draws 32 materials (distinct tables) per frame; every frame the engine
        // writes each material's table again at a new ring position with the same contents, so a material's pointer differs
        // from its previous frame's while its descriptors do not. Includes WriteTable per draw (subtract 3e).
        {
            const int Materials = 32;
            const int RingFirst = 8192, RingSlots = 24576 - Materials;
            var ring = new ResourceMaterializationCache();
            var frame = 0;
            var ringUserData = new uint[Materials][];
            void Frame()
            {
                var start = (frame++ * 977) % RingSlots;
                for (var m = 0; m < Materials; m++)
                {
                    var table = TableRegion + (ulong)(RingFirst + start + m) * TableStride;
                    WriteTable(memory, table, 100 + m);
                    inputs.UserData = new uint[] { (uint)table, (uint)(table >> 32) };
                    if (!ring.Materialize(plan, inputs, residentReader, ref snapshot, ref specialization, out var failure)) throw new InvalidOperationException(failure.ToString());
                }
            }

            for (var f = 0; f < 8; f++) Frame();
            Console.WriteLine($"       3g first 8 frames x {Materials} draws: hits {ring.Hits} misses {ring.Misses} refreshes {ring.TableRefreshes} -> hit rate {100.0 * ring.Hits / (8 * Materials):F1}%");
            var hits0 = ring.Hits; var misses0 = ring.Misses; var refreshes0 = ring.TableRefreshes;
            Harness.Measure("3g", "descriptor ring: 32 materials/frame, same contents at a new address each frame (+WriteTable)", Materials, 1404, _ => Frame(), unit: "draw");
            var calls = (ring.Hits - hits0) + (ring.Misses - misses0) + (ring.TableRefreshes - refreshes0);
            Console.WriteLine($"       3g steady: hits {ring.Hits - hits0} misses {ring.Misses - misses0} refreshes {ring.TableRefreshes - refreshes0} -> hit rate {100.0 * (ring.Hits - hits0) / Math.Max(1, calls):F1}%");
            inputs.UserData = new uint[] { (uint)table0, (uint)(table0 >> 32) };
        }

        // 3e: write-only cost of 3d for subtraction.
        Harness.Measure("3e", "baseline: WriteTable only (subtract from 3d)", Batch, 1404, _ =>
        {
            for (var i = 0; i < Batch; i++) WriteTable(memory, table0, 1 + (salt++ % (Pool - 1)));
        }, unit: "write");
    }
}
