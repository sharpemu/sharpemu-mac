// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// The bindless heap on the GPU (BindlessHeap, Gen5MslTranslator.BindlessHeap.cs): an image or sampler read from a heap
/// V# at a lane-chosen record takes the element of that record in the draw's heap view, marks it in the feedback
/// bitsets, reads a null element as zero, reads an integer element through the declaration of its numeric class, and a
/// 3D read of a record that holds a 2D texture takes the record's layers. The MSL twins of the SPIR-V heap tests
/// (tests/SharpEmu.ShaderCompiler.Tests/Resources/BindlessHeapTests.cs).
/// </summary>
public sealed class MetalBindlessHeapTests(ITestOutputHelper output)
{
    // MetalBindlessImageTests' waterfall: per lane, s_buffer_load_dwordx8 of the T# at key * 32 through the table V#
    // (user data block + 0x10), image_sample_l at (0.5, 0.5, layer 1.0, lod 0) dim:2D_ARRAY with the S# at +0x30,
    // the texel stored at lane * 16.
    private static readonly uint[] Sample =
    [
        0xF4080200, 0xFA000010, 0xF4080600, 0xFA000020, 0xF4080700, 0xFA000030, 0xBF8CC07F, 0x34020082,
        0x4A0202FF, 0x00014000, 0x34060084, 0xE0301000, 0x80060201, 0xBF8C3F70, 0x2C160484, 0x7E0802F0,
        0x7E0A02F0, 0x7E0C02F2, 0x7E0E0280, 0xBE86047E, 0x7E260281, 0x7EC8050B, 0xBE94047E, 0x7DA41664,
        0x8F648564, 0x7E260280, 0xF42C0304, 0xC8000000, 0xBF8CC07F, 0xF0900F28, 0x00E31404, 0xBEFE0414,
        0x7DAA2680, 0xBF89FFF3, 0xBEFE0406, 0xBF8C3F70, 0xE0781000, 0x80061403, 0xBF810000,
    ];

    // Variants, assembled with llvm-mc for gfx1030 and patched over the instructions they replace (same sizes):
    // image_load v[20:23], v[4:6], s[12:19] dmask:0xf dim:2D_ARRAY unorm at (0, 0, layer 1).
    private static uint[] Load() => Patch(Sample, (15, 0x7E080280), (16, 0x7E0A0280), (17, 0x7E0C0281), (29, 0xF0001F28), (30, 0x00031404));

    // image_get_resinfo v[20:23], v4 (lod 0), s[12:19] dmask:0xf dim:2D_ARRAY.
    private static uint[] SizeQuery() => Patch(Sample, (15, 0x7E080280), (29, 0xF0381F28), (30, 0x00031404));

    // image_sample_l at (0.5, 0.5, 1.0, lod 0) dim:3D.
    private static uint[] VolumeSample() => Patch(Sample, (29, 0xF0900F10));

    // The T# from user data (s_load_dwordx8 s[12:19] at +0x40), the S# from the table at key * 16
    // (s_lshl_b32 s100, s100, 4; s_buffer_load_dwordx4 s[28:31], s[8:11], s100).
    private static uint[] HeapSampler() => Patch(Sample, (4, 0xF40C0300), (5, 0xFA000040), (24, 0x8F648464), (26, 0xF4280704), (27, 0xC8000000));

    // The sample twice in the waterfall's block (the image_sample_l at 29-30 repeated after it, the loop's
    // s_cbranch_execnz moved back two words: -15), reading the same record under the same EXEC.
    private static uint[] DoubleSample() =>
        [.. Sample[..31], Sample[29], Sample[30], .. Patch(Sample, (33, 0xBF89FFF1))[31..]];

    private static uint[] Patch(uint[] words, params (int Index, uint Word)[] patches)
    {
        var patched = words.ToArray();
        foreach (var (index, word) in patches)
            patched[index] = word;
        return patched;
    }

    private const ulong ResourceBlock = 0x1000;
    private const ulong TableAddress = 0x10000;
    private const int Entries = 4096;
    private const int Lanes = 1024;
    private const int KeyOffset = 0x14000;
    private const uint UserDataCount = 16;
    // The view's first element: the record e of the heap is element First + e.
    private const uint First = 100;

    private const uint FloatFormat = 71;
    private const uint Type2DArray = 13;
    private const uint Type3D = 10;

    private static uint[] ImageWords(uint type) =>
        [0x100u, FloatFormat << 20, 0, (type << 28) | DescriptorConstants.IdentityImageSwizzle, 1, 0, 0, 0];

    private static byte[] Guest(uint tableBytes, uint imageType)
    {
        var memory = new byte[0x40000];
        void Write(ulong address, uint[] words)
        {
            for (var index = 0; index < words.Length; index++)
                BinaryPrimitives.WriteUInt32LittleEndian(memory.AsSpan((int)address + index * 4), words[index]);
        }

        Write(ResourceBlock + 0x10, [(uint)TableAddress, 0, tableBytes, DescriptorConstants.IdentityDestinationSelect]);
        Write(ResourceBlock + 0x20, [0x30000, 0, KeyOffset + Lanes * 4, 0x00038FAC]);
        Write(ResourceBlock + 0x30, [0, 0, 0, 0]);
        Write(ResourceBlock + 0x40, ImageWords(imageType));
        return memory;
    }

    private static ShaderCompileRequest Request(uint[] words, bool bindlessHeap, out ResourceSnapshot snapshot, uint tableBytes = Entries * 32,
        uint imageType = Type2DArray)
    {
        var guest = Guest(tableBytes, imageType);
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address + 4 > (ulong)guest.Length) return false;
            word = BinaryPrimitives.ReadUInt32LittleEndian(guest.AsSpan((int)address));
            return true;
        }

        var program = Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("bindless-heap", words, 0, 0));
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0x4845_4150, 0, UserDataCount, bindlessHeap: bindlessHeap);
        var userData = new uint[UserDataCount];
        userData[0] = (uint)ResourceBlock;
        snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var inputs = new ResourceRuntimeInputs { UserData = userData, ReadMemory = Read, ReadCleanMemory = Read };
        Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization, out var failure), $"materialize: {failure}");
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, UserDataCount),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        return new ShaderCompileRequest(plan, resources, layout) { WaveSize = 32, LocalSizeX = Lanes };
    }

    [Fact]
    public void AHeapRecordReadsItsElementThroughTheHeapAndMarksIt()
    {
        var request = Request(Sample, bindlessHeap: true, out var snapshot);
        Assert.True(request.UsesBindlessHeap);
        Assert.Single(request.HeapImageByMemoryIndex);
        Assert.Equal([new BindlessHeapDescriptor(false, TableAddress, Entries * 32, 0)], snapshot.Heaps);

        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        var layout = shader.ArgumentLayout!;
        Assert.Contains(layout.Fields, field => field.FieldKind == MslArgumentFieldKind.BindlessHeapElements);
        Assert.Contains(layout.Fields, field => field.FieldKind == MslArgumentFieldKind.BindlessHeapFeedback);
        Assert.Contains("device const ulong* heap_elements", shader.Source);
        Assert.Contains("atomic_fetch_or_explicit(&sharpemu_resources.heap_feedback[", shader.Source);
        Assert.Contains($"{BindlessHeap.CallRecordNullIndex}u", shader.Source);
        Assert.Contains("struct sharpemu_heap_texture2d_array_float_ { texture2d_array<float> texture; };", shader.Source);
        // A sample of a class with integer forms reads only a float element.
        Assert.Contains($"& {BindlessHeap.ClassMask}u", shader.Source);
    }

    // Without the heap the same table is a V#-described bindless table, and the source has no heap at all.
    [Fact]
    public void WithoutTheHeapTheTableIsABindlessTable()
    {
        var request = Request(Sample, bindlessHeap: false, out var snapshot);
        Assert.False(request.UsesBindlessHeap);
        Assert.Single(snapshot.BindlessImages);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.DoesNotContain("heap_", shader.Source);
        Assert.DoesNotContain(shader.ArgumentLayout!.Fields, field => field.FieldKind is MslArgumentFieldKind.BindlessHeapElements or
            MslArgumentFieldKind.BindlessHeapFeedback);
    }

    [Fact]
    public void AHeapSamplerReadsItsElementFromTheSamplerArray()
    {
        var request = Request(HeapSampler(), bindlessHeap: true, out _, tableBytes: Entries * 16);
        Assert.Single(request.HeapSamplerByMemoryIndex);
        Assert.Empty(request.HeapImageByMemoryIndex);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Contains("struct sharpemu_heap_sampler { sampler state; };", shader.Source);
        Assert.Contains($"{(uint)BindlessHeap.SamplerSlot * BindlessHeap.Capacity}u", shader.Source);
    }

    // ---------------------------------------------------------------- on the GPU

    private bool DeviceGate()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1")
        {
            Assert.True(MetalNative.IsAvailable);
            return true;
        }

        output.WriteLine("Bindless heap execution requires the exclusive Metal device gate.");
        return false;
    }

    // Lane l reads record Key(l): every record of the heap over four passes, keys past the view in the fifth.
    private static int Key(int pass, int lane) => pass < 4 ? (pass * Lanes + lane) * 1237 % Entries : Entries + lane * 13;

    private static uint Texel(int texture, int component) => BitConverter.SingleToUInt32Bits(texture * 4 + component + 0.25f);

    private static nint LayeredTexture(nuint format, uint[] texel) =>
        MetalNative.CreateTexelTexture(format, MetalNative.TextureType2DArray, 2, [.. Enumerable.Repeat(0xDEADu, 4), .. texel]);

    // Runs the program over five passes of keys with the given element table and class words; returns per pass the
    // lanes' four result words and the feedback words after the pass.
    private List<(int[] Keys, uint[] Result, uint[] Feedback)> Run(ShaderCompileRequest request, ResourceSnapshot snapshot, byte[] elements,
        IReadOnlyList<nint> residents, uint[] classes, nint texture = 0, Func<DescriptorBindingKind, nint>? textureOfClass = null)
    {
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error);
        var layout = request.Bindings;
        var shaderData = new uint[Math.Max(layout.ShaderDataDwordCount, 1)];
        for (var index = 0; index < layout.UserDataRegisters.Count; index++)
            shaderData[index] = snapshot.UserData[(int)layout.UserDataRegisters[index]];
        var view = HeapView.Of(snapshot.Heaps[0], First, snapshot.Heaps[0].ViewRecords);
        shaderData[layout.HeapViewsDword] = view.Word0;
        shaderData[layout.HeapViewsDword + 1] = view.Word1;
        var pushData = new byte[PushData.ByteSize];
        for (var index = 0; layout.UsesPushData && index < layout.ShaderDataDwordCount; index++)
            BinaryPrimitives.WriteUInt32LittleEndian(pushData.AsSpan((int)(layout.PushDataStartDword + index) * 4), shaderData[index]);

        var heapElements = MetalNative.CreateHeapElements(elements, residents);
        var runs = new List<(int[], uint[], uint[])>();
        for (var pass = 0; pass < 5; pass++)
        {
            var feedbackWords = new uint[BindlessHeap.FeedbackWords];
            classes.CopyTo(feedbackWords, (int)BindlessHeap.MarkWords);
            var feedback = MetalNative.CreateHeapFeedback(feedbackWords);
            var keys = Enumerable.Range(0, Lanes).Select(lane => Key(pass, lane)).ToArray();
            var data = new byte[KeyOffset + Lanes * 4];
            for (var lane = 0; lane < Lanes; lane++)
                BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(KeyOffset + lane * 4), (uint)keys[lane] << 4);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, data, pushData, shader.ArgumentLayout!, Lanes,
                out var result, out _, out error, texture: texture, sampler: MetalNative.CreatePointSampler(),
                tables: [MetalNative.CreateWordTable(DescriptorBindingKind.ShaderData, shaderData)], textureOfClass: textureOfClass,
                heap: (heapElements, feedback)), error);
            var words = new uint[Lanes * 4];
            Buffer.BlockCopy(result, 0, words, 0, words.Length * 4);
            runs.Add((keys, words, MetalNative.ReadWords(feedback)));
        }

        return runs;
    }

    private static void SetElement(byte[] elements, int slot, uint element, nint texture) =>
        BinaryPrimitives.WriteUInt64LittleEndian(elements.AsSpan((int)(((uint)slot * BindlessHeap.Capacity + element) * 8)),
            texture == 0 ? 0 : MetalNative.ResourceId(texture));

    // The elements a pass marks in one array: each in-view record's element, and the null element for keys past it.
    private static void AssertMarks(uint[] feedback, int slot, IEnumerable<uint> elements, int pass)
    {
        var expected = new uint[BindlessHeap.FeedbackWordsPerArray];
        foreach (var element in elements)
            expected[element >> 5] |= 1u << (int)(element & 31);
        for (var word = 0; word < expected.Length; word++)
        {
            var actual = feedback[slot * (int)BindlessHeap.FeedbackWordsPerArray + word];
            Assert.True(expected[word] == actual, $"pass {pass} slot {slot} mark word {word}: expected 0x{expected[word]:X8}, marked 0x{actual:X8}");
        }
    }

    private static uint ElementOf(int key) => key < Entries ? First + (uint)key : BindlessHeap.CallRecordNullIndex;

    private static void AssertLane(uint[] result, int pass, int lane, int key, uint[] expected)
    {
        for (var component = 0; component < 4; component++)
        {
            var actual = result[lane * 4 + component];
            Assert.True(expected[component] == actual,
                $"pass {pass} lane {lane} key {key} component {component}: expected 0x{expected[component]:X8}, read 0x{actual:X8}");
        }
    }

    // A record read twice in one block under the same EXEC is marked once, as the SPIR-V translator's MarkHeapRecord
    // leaves out a repeated mark within a block (HeapMarks): the second read reuses the first one's record index.
    [Fact]
    public void ARecordReadTwiceInABlockIsMarkedOnce()
    {
        static int Count(string source, string text) => (source.Length - source.Replace(text, string.Empty, StringComparison.Ordinal).Length) / text.Length;
        const string Mark = "atomic_fetch_or_explicit(&sharpemu_resources.heap_feedback[";
        Assert.True(Gen5MslTranslator.TryCompileProgram(Request(Sample, bindlessHeap: true, out _), out var single, out var error), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(Request(DoubleSample(), bindlessHeap: true, out _), out var twice, out error), error);
        Assert.Equal(1, Count(single!.Source, Mark));
        Assert.Equal(1, Count(twice!.Source, Mark));
        Assert.Equal(2 * Count(single.Source, ".sample("), Count(twice.Source, ".sample("));
    }

    // Record e holds texture (e * 37) % 64, except every seventh, whose element is null; keys past the view read the null
    // element. Every read element is marked, and nothing else; also when the block reads each record twice and marks it
    // once (ARecordReadTwiceInABlockIsMarkedOnce).
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EveryLaneReadsTheElementOfItsRecordAndMarksIt(bool twice)
    {
        if (!DeviceGate()) return;

        var request = Request(twice ? DoubleSample() : Sample, bindlessHeap: true, out var snapshot);
        var slot = BindlessHeap.ImageSlot(ImageDescriptorBinding.ForImage(request.Resources.Info.Images[0])!.Value);
        Assert.Equal(BindlessHeap.VolumeLayerSlot, slot);
        var textures = Enumerable.Range(0, 64)
            .Select(texture => LayeredTexture(MetalNative.PixelFormatRgba32Float, [.. Enumerable.Range(0, 4).Select(component => Texel(texture, component))]))
            .ToArray();
        int TextureOf(int record) => record % 7 == 3 ? -1 : record * 37 % 64;
        var elements = new byte[BindlessHeap.ArrayCount * BindlessHeap.Capacity * 8];
        for (var record = 0; record < Entries; record++)
            SetElement(elements, slot, First + (uint)record, TextureOf(record) < 0 ? 0 : textures[TextureOf(record)]);

        var nullArray = LayeredTexture(MetalNative.PixelFormatRgba32Float, new uint[4]);
        foreach (var (pass, (keys, result, feedback)) in Run(request, snapshot, elements, textures, [], nullArray).Index())
        {
            for (var lane = 0; lane < Lanes; lane++)
            {
                var texture = keys[lane] < Entries ? TextureOf(keys[lane]) : -1;
                AssertLane(result, pass, lane, keys[lane], texture < 0 ? new uint[4] : [.. Enumerable.Range(0, 4).Select(component => Texel(texture, component))]);
            }

            AssertMarks(feedback, slot, keys.Select(ElementOf), pass);
        }
    }

    // A load of a class with integer forms reads each element through the declaration of the numeric class the host
    // recorded for it (float, uint or sint texels, bit for bit); a null element of any class reads zero.
    [Fact]
    public void AnIntegerElementIsReadThroughItsNumericClass()
    {
        if (!DeviceGate()) return;

        var request = Request(Load(), bindlessHeap: true, out var snapshot);
        var slot = BindlessHeap.ImageSlot(ImageDescriptorBinding.ForImage(request.Resources.Info.Images[0])!.Value);
        Assert.True(BindlessHeap.HoldsIntegers(slot));
        const nuint Rgba32Sint = 124;
        (ImageNumericClass Class, nuint Format)[] kinds =
            [(ImageNumericClass.Float, MetalNative.PixelFormatRgba32Float), (ImageNumericClass.Uint, MetalNative.PixelFormatRgba32Uint), (ImageNumericClass.Sint, Rgba32Sint)];
        uint TexelOf(int record, int component) => 0x4000_0000u + (uint)(record * 4 + component);
        ImageNumericClass? ClassOf(int record) => record % 5 == 4 ? null : kinds[record % 3].Class;
        var elements = new byte[BindlessHeap.ArrayCount * BindlessHeap.Capacity * 8];
        var classes = new uint[BindlessHeap.ClassWords];
        var textures = new List<nint>();
        for (var record = 0; record < Entries; record++)
        {
            var element = First + (uint)record;
            if (ClassOf(record) is not { } numericClass)
                continue;
            var texture = LayeredTexture(kinds[record % 3].Format, [.. Enumerable.Range(0, 4).Select(component => TexelOf(record, component))]);
            textures.Add(texture);
            SetElement(elements, slot, element, texture);
            classes[element >> (int)BindlessHeap.ClassWordShift] |=
                BindlessHeap.ClassCode(numericClass) << (int)((element & (BindlessHeap.ClassesPerWord - 1)) * BindlessHeap.ClassBits);
        }

        var nullArray = LayeredTexture(MetalNative.PixelFormatRgba32Float, new uint[4]);
        foreach (var (pass, (keys, result, feedback)) in Run(request, snapshot, elements, textures, classes, nullArray).Index())
        {
            for (var lane = 0; lane < Lanes; lane++)
            {
                var key = keys[lane];
                var present = key < Entries && ClassOf(key) is not null;
                AssertLane(result, pass, lane, key, present ? [.. Enumerable.Range(0, 4).Select(component => TexelOf(key, component))] : new uint[4]);
            }

            AssertMarks(feedback, slot, keys.Select(ElementOf), pass);
        }
    }

    // The size query of an element reports its texture's extent, layers and levels; a null element reports zero.
    [Fact]
    public void ANullElementReportsNoSize()
    {
        if (!DeviceGate()) return;

        var request = Request(SizeQuery(), bindlessHeap: true, out var snapshot);
        var slot = BindlessHeap.ImageSlot(ImageDescriptorBinding.ForImage(request.Resources.Info.Images[0])!.Value);
        var texture = LayeredTexture(MetalNative.PixelFormatRgba32Float, new uint[4]);
        var elements = new byte[BindlessHeap.ArrayCount * BindlessHeap.Capacity * 8];
        for (var record = 0; record < Entries; record += 2)
            SetElement(elements, slot, First + (uint)record, texture);

        foreach (var (pass, (keys, result, _)) in Run(request, snapshot, elements, [texture], [], texture).Index())
        {
            for (var lane = 0; lane < Lanes; lane++)
            {
                var key = keys[lane];
                AssertLane(result, pass, lane, key, key < Entries && key % 2 == 0 ? [1u, 1u, 2u, 1u] : new uint[4]);
            }
        }
    }

    // A 3D read takes the record's 3D element; a record whose 3D element is null (it holds a 2D texture) is read through
    // its 2D-array element, the third coordinate as the layer; a record null in both reads zero. Only the reads that fall
    // back mark the 2D-array element.
    [Fact]
    public void AVolumeReadOfALayeredRecordReadsItsLayer()
    {
        if (!DeviceGate()) return;

        var request = Request(VolumeSample(), bindlessHeap: true, out var snapshot, imageType: Type3D);
        Assert.Equal(BindlessHeap.VolumeSlot, BindlessHeap.ImageSlot(ImageDescriptorBinding.ForImage(request.Resources.Info.Images[0])!.Value));
        var volume = MetalNative.CreateFloatTexture(7, 1, 1, 1, 1,
            [[.. Enumerable.Range(0, 4).SelectMany(component => BitConverter.GetBytes(Texel(1, component)))]]);
        var layered = LayeredTexture(MetalNative.PixelFormatRgba32Float, [.. Enumerable.Range(0, 4).Select(component => Texel(2, component))]);
        var elements = new byte[BindlessHeap.ArrayCount * BindlessHeap.Capacity * 8];
        for (var record = 0; record < Entries; record++)
        {
            if (record % 3 == 0) SetElement(elements, BindlessHeap.VolumeSlot, First + (uint)record, volume);
            if (record % 3 == 1) SetElement(elements, BindlessHeap.VolumeLayerSlot, First + (uint)record, layered);
        }

        foreach (var (pass, (keys, result, feedback)) in Run(request, snapshot, elements, [volume, layered], [], volume).Index())
        {
            for (var lane = 0; lane < Lanes; lane++)
            {
                var key = keys[lane];
                var texture = key >= Entries ? -1 : key % 3 == 0 ? 1 : key % 3 == 1 ? 2 : -1;
                AssertLane(result, pass, lane, key, texture < 0 ? new uint[4] : [.. Enumerable.Range(0, 4).Select(component => Texel(texture, component))]);
            }

            AssertMarks(feedback, BindlessHeap.VolumeSlot, keys.Select(ElementOf), pass);
            AssertMarks(feedback, BindlessHeap.VolumeLayerSlot, keys.Where(key => key >= Entries || key % 3 != 0).Select(ElementOf), pass);
        }
    }

    // The sampler a record names filters the read: records alternate a nearest and a linear sampler over a two-texel
    // row, sampled at its middle. Keys past the view read the null sampler the host fills the array with (nearest).
    [Fact]
    public void AHeapSamplerFiltersWithTheSamplerItsRecordNames()
    {
        if (!DeviceGate()) return;

        var request = Request(HeapSampler(), bindlessHeap: true, out var snapshot, tableBytes: Entries * 16);
        var left = Enumerable.Range(0, 4).Select(component => 2f + component).ToArray();
        var right = Enumerable.Range(0, 4).Select(component => 10f + component * 4).ToArray();
        var row = left.Concat(right).SelectMany(BitConverter.GetBytes).ToArray();
        var texture = MetalNative.CreateFloatTexture(3, 2, 1, 2, 1, [[.. new byte[row.Length], .. row]]);
        var nearest = MetalNative.CreateFilterSampler(0);
        var linear = MetalNative.CreateFilterSampler(1);
        var elements = new byte[BindlessHeap.ArrayCount * BindlessHeap.Capacity * 8];
        for (var element = 0u; element < BindlessHeap.Capacity; element++)
        {
            var record = (int)element - (int)First;
            var sampler = record >= 0 && record < Entries && record % 2 == 1 ? linear : nearest;
            BinaryPrimitives.WriteUInt64LittleEndian(elements.AsSpan((int)(((uint)BindlessHeap.SamplerSlot * BindlessHeap.Capacity + element) * 8)),
                MetalNative.ResourceId(sampler));
        }

        foreach (var (pass, (keys, result, feedback)) in Run(request, snapshot, elements, [], [], texture).Index())
        {
            for (var lane = 0; lane < Lanes; lane++)
            {
                var key = keys[lane];
                var filtered = key < Entries && key % 2 == 1;
                AssertLane(result, pass, lane, key,
                    [.. Enumerable.Range(0, 4).Select(component => BitConverter.SingleToUInt32Bits(filtered ? (left[component] + right[component]) / 2 : right[component]))]);
            }

            AssertMarks(feedback, BindlessHeap.SamplerSlot, keys.Select(ElementOf), pass);
        }
    }
}
