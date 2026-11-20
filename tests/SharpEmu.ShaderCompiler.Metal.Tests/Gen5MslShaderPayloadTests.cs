// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections;
using System.Reflection;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// The translation disk cache keeps a Metal translation as Gen5MslShaderPayload bytes and a later session builds its
/// libraries and pipelines from what they read back: every field of every shader must come back as the translator
/// made it, or the second launch draws differently from the first.
/// </summary>
public sealed class Gen5MslShaderPayloadTests(ITestOutputHelper output)
{
    // Every public property of the records a payload holds. A property added to one of them fails here until the
    // payload writes it (and its format version is raised).
    private static readonly (Type Type, string[] Properties)[] PinnedProperties =
    [
        (typeof(Gen5MslShader),
        [
            "AttributeCount", "ArgumentLayout", "BufferWordConstants", "CannotDiscard", "ColorOutputMask", "ConstantPositions",
            "EntryPoint", "FlattenedTableMinimumWords", "MeshInfo", "PersistentThreads", "Source", "Stage", "Tessellation",
            "ThreadgroupSizeX", "ThreadgroupSizeY", "ThreadgroupSizeZ", "UsesMipStatisticsConstant", "UsesPerVertexInputs",
            "WritesLayer", "WritesViewportIndex",
        ]),
        (typeof(Gen5MslArgumentLayout), ["ByteSize", "Fields"]),
        (typeof(MslArgumentField), ["ByteOffset", "ByteSize", "Count", "FieldKind", "FirstId", "Kind", "Name"]),
        (typeof(Gen5MslMeshInfo),
        [
            "Amplified", "DeviceLds", "DrawInfoBufferIndex", "GeometryInputPrimitives", "GeometryInputVertices",
            "GeometryVerticesOut", "GuestWaveLanes", "HasGeometryHalf", "IndexBufferIndex", "LdsBytes", "MaxPrimitives",
            "MaxVertices", "NarrowWave64", "PrimitivesPerSubgroup", "ReadsIndexBuffer", "ThreadgroupMemoryBytes",
            "ThreadsPerThreadgroup", "Topology", "VerticesPerPrimitive", "VerticesPerSubgroup", "Wave64", "WavesPerThreadgroup",
        ]),
        (typeof(Gen5TessellationLaunch),
        [
            "ControlPoints", "FactorBytesPerPatch", "MaxFactor", "OffchipBlockBytes", "OutputClockwise", "Partition",
            "PatchesPerGroup", "Quad", "Role",
        ]),
        (typeof(Gen5MslShader.BufferWordConstant), ["Dword", "Index"]),
    ];

    [Fact]
    public void ThePayloadWritesEveryPropertyOfTheRecordsItHolds()
    {
        foreach (var (type, properties) in PinnedProperties)
        {
            var actual = type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(static property => property.Name != "EqualityContract")
                .Select(static property => property.Name).Order(StringComparer.Ordinal).ToArray();
            Assert.Equal(properties.Order(StringComparer.Ordinal).ToArray(), actual);
        }
    }

    // Translations of each stage and of the paths that set the reflection fields: compute (plain, LDS, wave64,
    // persistent threads), vertex (plain, constant positions), pixel (plain, per-vertex inputs), primitive shaders
    // (mesh info), tessellation (hull and domain), and storage images with runtime swizzles.
    public static TheoryData<string> Translations() => new(Corpus().Select(static entry => entry.Name));

    [Theory]
    [MemberData(nameof(Translations))]
    public void ATranslationReadsBackFieldByField(string name)
    {
        var shader = Corpus().Single(entry => entry.Name == name).Compile();

        var payload = Gen5MslShaderPayload.Write(shader, ShaderStage.Pixel, 0x0123_4567_89AB_CDEF);
        Assert.True(Gen5MslShaderPayload.TryRead(payload, out var read, out var stage, out var hash, out var error), error);

        Assert.Equal(ShaderStage.Pixel, stage);
        Assert.Equal(0x0123_4567_89AB_CDEFul, hash);
        AssertSameFields(shader, read, nameof(Gen5MslShader));
        Assert.Equal(payload, Gen5MslShaderPayload.Write(read, stage, hash));
        output.WriteLine($"{name}: {payload.Length} bytes, stage {shader.Stage}, entry {shader.EntryPoint}");
    }

    // The corpus covers what each field is for; this record sets every field to a value other than its default,
    // so a field the payload drops or swaps fails even when no translation above sets it.
    [Fact]
    public void ARecordWithEveryFieldSetReadsBackFieldByField()
    {
        var shader = new Gen5MslShader(
            "kernel void k() {}\n", "k", Gen5MslStage.Mesh, 3, 4, 5, 6,
            new Gen5MslArgumentLayout(
                [
                    new MslArgumentField(DescriptorBindingKind.Buffers, MslArgumentFieldKind.BufferPointers, "buffers", 2, 0, 0, 16),
                    new MslArgumentField(DescriptorBindingKind.Samplers, MslArgumentFieldKind.Samplers, "samplers", 1, 2, 16, 8),
                ],
                24),
            new Gen5MslMeshInfo(64, 2, 128, 96, 3, Gen5MslMeshTopology.Line, 4096, 7, 8, true, 9, true, true, true, 1024, true, 10, 11),
            WritesLayer: true,
            WritesViewportIndex: true)
        {
            UsesPerVertexInputs = true,
            UsesMipStatisticsConstant = true,
            BufferWordConstants = [new(12, 13), new(14, 15)],
            ConstantPositions = [1f, -1f, float.NaN, 0.5f],
            CannotDiscard = true,
            ColorOutputMask = 0x5,
            PersistentThreads = true,
            Tessellation = new Gen5TessellationLaunch(Gen5TessellationRole.Domain, 16, 4, 0x8000, 24,
                Gen5TessellationPartition.FractionalOdd, OutputClockwise: true, MaxFactor: 15, Quad: true),
            FlattenedTableMinimumWords = 17,
        };
        AssertNoDefaults(shader, nameof(Gen5MslShader));

        Assert.True(Gen5MslShaderPayload.TryRead(Gen5MslShaderPayload.Write(shader, ShaderStage.Vertex, 1),
            out var read, out var stage, out _, out var error), error);

        Assert.Equal(ShaderStage.Vertex, stage);
        AssertSameFields(shader, read, nameof(Gen5MslShader));
        Assert.Equal(BitConverter.SingleToUInt32Bits(float.NaN), BitConverter.SingleToUInt32Bits(read.ConstantPositions![2]));
    }

    [Fact]
    public void ADamagedOrForeignPayloadIsRefusedWithItsReason()
    {
        var payload = Gen5MslShaderPayload.Write(Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.ExecStore),
            ShaderStage.Compute, 1);

        // Every truncation, from nothing to one byte short.
        for (var length = 0; length < payload.Length; length++)
        {
            Assert.False(Gen5MslShaderPayload.TryRead(payload[..length], out _, out _, out _, out var error), $"length {length}");
            Assert.NotEmpty(error);
        }

        var version = (byte[])payload.Clone();
        version[4]++;
        Assert.False(Gen5MslShaderPayload.TryRead(version, out _, out _, out _, out var versionError));
        Assert.Contains("format", versionError, StringComparison.Ordinal);

        var magic = (byte[])payload.Clone();
        magic[0] ^= 0xFF;
        Assert.False(Gen5MslShaderPayload.TryRead(magic, out _, out _, out _, out _));

        Assert.False(Gen5MslShaderPayload.TryRead([.. payload, 0], out _, out _, out _, out var trailingError));
        Assert.Contains("past the end", trailingError, StringComparison.Ordinal);

        // A SPIR-V module (another backend's payload) or the bare MSL text is not a payload.
        Assert.False(Gen5MslShaderPayload.TryRead([0x03, 0x02, 0x23, 0x07, 0, 0, 1, 0], out _, out _, out _, out _));
        Assert.False(Gen5MslShaderPayload.TryRead("kernel void k() {}"u8.ToArray(), out _, out _, out _, out _));

        // An undefined stage: the payload of a request this build has no stage for.
        var stage = (byte[])payload.Clone();
        stage[8] = 0xEE;
        Assert.False(Gen5MslShaderPayload.TryRead(stage, out _, out _, out _, out var stageError));
        Assert.Contains("ShaderStage", stageError, StringComparison.Ordinal);
    }

    private sealed record Entry(string Name, Func<Gen5MslShader> Compile);

    private static IEnumerable<Entry> Corpus()
    {
        foreach (var fixture in Gen5ComputeFixtures.All)
        {
            yield return new("compute-" + fixture.Name, () => Gen5ComputeFixtures.CompileRequestOrThrow(fixture));
        }

        yield return new("compute-wave64", () => Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64Broadcast, waveLaneCount: 64, localSizeX: 64));
        yield return new("compute-persistent", () =>
        {
            var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.PersistentQueueLoop);
            Assert.True(shader.PersistentThreads);
            return shader;
        });
        yield return new("vertex", () => Gen5ComputeFixtures.CompileVertexOrThrow());
        yield return new("vertex-constant-positions", () =>
        {
            var request = Gen5ComputeFixtures.RequestOrThrow(Gen5MslLoadElisionTests.PresentQuad(), ShaderStage.Vertex);
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
            Assert.NotNull(shader.ConstantPositions);
            return shader;
        });
        yield return new("pixel", () => Gen5ComputeFixtures.CompilePixelOrThrow());
        yield return new("pixel-per-vertex-inputs", () =>
        {
            var shader = Gen5MslPixelInputTests.Compile(
                [Gen5MslPixelInputTests.Interp(4, 2, 1, 2, 0), .. Gen5MslPixelInputTests.ExportAndEnd(4, 4, 4, 4)],
                Gen5MslPixelInputTests.PerspCenter, cntl: [0, 0x401]);
            Assert.True(shader.UsesPerVertexInputs);
            return shader;
        });
        yield return new("mesh-passthrough", () => Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.Passthrough()));
        yield return new("mesh-culling", () =>
        {
            var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
            Assert.NotNull(shader.MeshInfo);
            return shader;
        });
        yield return new("tessellation-hull", () => Gen5MslTessellationLaunchTests.CompileShader(ShaderStage.Compute,
            new Gen5TessellationLaunch(Gen5TessellationRole.Hull, 21, 3, 0x8000, 16)));
        yield return new("tessellation-domain", () => Gen5MslTessellationLaunchTests.CompileShader(ShaderStage.Vertex,
            new Gen5TessellationLaunch(Gen5TessellationRole.Domain, 16, 4, 0x8000, 24, Gen5TessellationPartition.FractionalOdd,
                MaxFactor: 15, Quad: true)));
        yield return new("image-runtime-swizzle", () =>
        {
            BindingLayout? layout = null;
            Assert.True(Gen5ImageFixtures.TryCompile("ImageStore", Gen5ImageFixtures.ImageType2D, out var shader, out var error,
                unifiedFormat: 71, dstSelect: 0xF2E, addressValues: [1, 0], storeValues: [1, 2, 3, 4],
                runtimeImageSwizzles: true, inspect: (request, _) => layout = request.Bindings), error);
            Assert.Equal(1u, layout!.StorageImageSwizzleCount);
            return shader!;
        });
    }

    private static void AssertSameFields(object? expected, object? actual, string path)
    {
        if (expected is null || actual is null)
        {
            Assert.True(expected is null && actual is null, $"{path}: {expected ?? "null"} != {actual ?? "null"}");
            return;
        }

        // A list reads back as an array: the elements are what the host reads.
        var type = expected.GetType();
        Assert.True(type == actual.GetType() || (expected is IEnumerable && actual is IEnumerable && type != typeof(string)),
            $"{path}: {type} != {actual.GetType()}");
        if (type.IsPrimitive || type.IsEnum || type == typeof(string))
        {
            if (expected is float single)
            {
                Assert.True(BitConverter.SingleToUInt32Bits(single) == BitConverter.SingleToUInt32Bits((float)actual), path);
                return;
            }

            Assert.True(expected.Equals(actual), $"{path}: {expected} != {actual}");
            return;
        }

        if (expected is IEnumerable sequence)
        {
            var left = sequence.Cast<object?>().ToArray();
            var right = ((IEnumerable)actual).Cast<object?>().ToArray();
            Assert.True(left.Length == right.Length, $"{path}: {left.Length} items != {right.Length}");
            for (var index = 0; index < left.Length; index++)
            {
                AssertSameFields(left[index], right[index], $"{path}[{index}]");
            }

            return;
        }

        foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (property.Name == "EqualityContract" || property.GetIndexParameters().Length != 0)
            {
                continue;
            }

            AssertSameFields(property.GetValue(expected), property.GetValue(actual), $"{path}.{property.Name}");
        }
    }

    private static void AssertNoDefaults(object value, string path)
    {
        foreach (var property in value.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (property.Name == "EqualityContract")
            {
                continue;
            }

            var field = property.GetValue(value);
            var name = $"{path}.{property.Name}";
            Assert.True(field is not null, $"{name} is null");
            var type = field.GetType();
            if (type.IsValueType)
            {
                Assert.True(!field.Equals(Activator.CreateInstance(type)), $"{name} has its default value");
            }
            else if (field is string text)
            {
                Assert.True(text.Length != 0, $"{name} is empty");
            }
            else if (field is IEnumerable sequence)
            {
                Assert.True(sequence.Cast<object>().Any(), $"{name} is empty");
            }
            else
            {
                AssertNoDefaults(field, name);
            }
        }
    }
}
