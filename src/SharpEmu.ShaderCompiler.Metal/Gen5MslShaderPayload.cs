// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

/// <summary>
/// A translated MSL shader as bytes, for the translation disk cache (TranslationDiskCache): the source and every
/// field of <see cref="Gen5MslShader"/> and the records it holds, written one by one, behind the stage and hash of
/// the request it was translated from. Reading it back gives a record equal field by field to the translation, so a
/// later session builds the same libraries and pipelines from it.
/// </summary>
/// <remarks>
/// A field added to any of these records must be written here and the format version raised (the test that pins
/// the records' properties fails until it is). The version is part of the cache's stamp, so entries of an older
/// format are never read; a payload of another version, a short or overlong one, or one with an unknown enum value
/// is refused with the reason, never read as far as it goes.
/// </remarks>
public static class Gen5MslShaderPayload
{
    /// <summary>The layout version; part of the translation cache's stamp.</summary>
    /// <remarks>Version 2: argument layouts may hold the linked-call field kinds (call records, call marks, call faults).</remarks>
    public const int FormatVersion = 2;

    // "MSLC", little-endian.
    private const uint Magic = 0x434C534D;

    // Bounds a damaged count must not get past before the read fails on the length.
    private const int MaxListCount = 1 << 20;

    /// <summary>The bytes of <paramref name="shader"/>, translated from a request of
    /// <paramref name="requestStage"/> and <paramref name="requestHash"/>.</summary>
    public static byte[] Write(Gen5MslShader shader, ShaderStage requestStage, ulong requestHash)
    {
        ArgumentNullException.ThrowIfNull(shader);
        using var stream = new MemoryStream(shader.Source.Length + 1024);
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(Magic);
            writer.Write(FormatVersion);
            writer.Write((byte)requestStage);
            writer.Write(requestHash);

            writer.Write(shader.Source);
            writer.Write(shader.EntryPoint);
            writer.Write((int)shader.Stage);
            writer.Write(shader.AttributeCount);
            writer.Write(shader.ThreadgroupSizeX);
            writer.Write(shader.ThreadgroupSizeY);
            writer.Write(shader.ThreadgroupSizeZ);
            WriteArgumentLayout(writer, shader.ArgumentLayout);
            WriteMeshInfo(writer, shader.MeshInfo);
            writer.Write(shader.WritesLayer);
            writer.Write(shader.WritesViewportIndex);
            writer.Write(shader.UsesPerVertexInputs);
            writer.Write(shader.UsesMipStatisticsConstant);
            writer.Write(shader.BufferWordConstants.Count);
            foreach (var constant in shader.BufferWordConstants)
            {
                writer.Write(constant.Index);
                writer.Write(constant.Dword);
            }

            writer.Write(shader.ConstantPositions is not null);
            if (shader.ConstantPositions is { } positions)
            {
                writer.Write(positions.Length);
                // The bits, so a NaN keeps its payload.
                foreach (var position in positions) writer.Write(BitConverter.SingleToUInt32Bits(position));
            }

            writer.Write(shader.CannotDiscard);
            writer.Write(shader.ColorOutputMask);
            writer.Write(shader.PersistentThreads);
            WriteTessellation(writer, shader.Tessellation);
            writer.Write(shader.FlattenedTableMinimumWords);
        }

        return stream.ToArray();
    }

    /// <summary>Reads a payload <see cref="Write"/> made; false with the reason when it is not one, or is
    /// damaged.</summary>
    public static bool TryRead(byte[] payload, out Gen5MslShader shader, out ShaderStage requestStage,
        out ulong requestHash, out string error)
    {
        ArgumentNullException.ThrowIfNull(payload);
        shader = null!;
        requestStage = ShaderStage.Unknown;
        requestHash = 0;
        try
        {
            using var stream = new MemoryStream(payload, writable: false);
            using var reader = new BinaryReader(stream, Encoding.UTF8);
            if (payload.Length < 8 || reader.ReadUInt32() != Magic)
            {
                error = "not an MSL shader payload";
                return false;
            }

            var version = reader.ReadInt32();
            if (version != FormatVersion)
            {
                error = $"payload format {version}, this build reads {FormatVersion}";
                return false;
            }

            requestStage = Defined((ShaderStage)reader.ReadByte());
            requestHash = reader.ReadUInt64();

            var source = reader.ReadString();
            var entryPoint = reader.ReadString();
            var stage = Defined((Gen5MslStage)reader.ReadInt32());
            var attributeCount = reader.ReadUInt32();
            var threadgroupX = reader.ReadUInt32();
            var threadgroupY = reader.ReadUInt32();
            var threadgroupZ = reader.ReadUInt32();
            var argumentLayout = ReadArgumentLayout(reader);
            var meshInfo = ReadMeshInfo(reader);
            var writesLayer = reader.ReadBoolean();
            var writesViewportIndex = reader.ReadBoolean();
            var usesPerVertexInputs = reader.ReadBoolean();
            var usesMipStatisticsConstant = reader.ReadBoolean();
            var bufferWordConstants = new Gen5MslShader.BufferWordConstant[Count(reader)];
            for (var index = 0; index < bufferWordConstants.Length; index++)
            {
                bufferWordConstants[index] = new Gen5MslShader.BufferWordConstant(reader.ReadUInt32(), reader.ReadUInt32());
            }

            float[]? constantPositions = null;
            if (reader.ReadBoolean())
            {
                constantPositions = new float[Count(reader)];
                for (var index = 0; index < constantPositions.Length; index++)
                {
                    constantPositions[index] = BitConverter.UInt32BitsToSingle(reader.ReadUInt32());
                }
            }

            var cannotDiscard = reader.ReadBoolean();
            var colorOutputMask = reader.ReadUInt32();
            var persistentThreads = reader.ReadBoolean();
            var tessellation = ReadTessellation(reader);
            var flattenedTableMinimumWords = reader.ReadUInt32();
            if (stream.Position != stream.Length)
            {
                error = $"{stream.Length - stream.Position} bytes past the end of the shader";
                return false;
            }

            shader = new Gen5MslShader(source, entryPoint, stage, attributeCount, threadgroupX, threadgroupY, threadgroupZ,
                argumentLayout, meshInfo, writesLayer, writesViewportIndex)
            {
                UsesPerVertexInputs = usesPerVertexInputs,
                UsesMipStatisticsConstant = usesMipStatisticsConstant,
                BufferWordConstants = bufferWordConstants,
                ConstantPositions = constantPositions,
                CannotDiscard = cannotDiscard,
                ColorOutputMask = colorOutputMask,
                PersistentThreads = persistentThreads,
                Tessellation = tessellation,
                FlattenedTableMinimumWords = flattenedTableMinimumWords,
            };
            error = string.Empty;
            return true;
        }
        catch (Exception exception) when (exception is EndOfStreamException or InvalidDataException or IOException or
            DecoderFallbackException or ArgumentException or OverflowException)
        {
            shader = null!;
            error = $"damaged payload: {exception.GetType().Name}: {exception.Message}";
            return false;
        }
    }

    private static void WriteArgumentLayout(BinaryWriter writer, Gen5MslArgumentLayout? layout)
    {
        writer.Write(layout is not null);
        if (layout is null)
        {
            return;
        }

        writer.Write(layout.ByteSize);
        writer.Write(layout.Fields.Count);
        foreach (var field in layout.Fields)
        {
            writer.Write((uint)field.Kind);
            writer.Write((int)field.FieldKind);
            writer.Write(field.Name);
            writer.Write(field.Count);
            writer.Write(field.FirstId);
            writer.Write(field.ByteOffset);
            writer.Write(field.ByteSize);
        }
    }

    private static Gen5MslArgumentLayout? ReadArgumentLayout(BinaryReader reader)
    {
        if (!reader.ReadBoolean())
        {
            return null;
        }

        var byteSize = reader.ReadUInt32();
        var fields = new MslArgumentField[Count(reader)];
        for (var index = 0; index < fields.Length; index++)
        {
            fields[index] = new MslArgumentField(
                (DescriptorBindingKind)reader.ReadUInt32(),
                Defined((MslArgumentFieldKind)reader.ReadInt32()),
                reader.ReadString(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32());
        }

        return new Gen5MslArgumentLayout(fields, byteSize);
    }

    private static void WriteMeshInfo(BinaryWriter writer, Gen5MslMeshInfo? mesh)
    {
        writer.Write(mesh is not null);
        if (mesh is null)
        {
            return;
        }

        writer.Write(mesh.ThreadsPerThreadgroup);
        writer.Write(mesh.WavesPerThreadgroup);
        writer.Write(mesh.MaxVertices);
        writer.Write(mesh.MaxPrimitives);
        writer.Write(mesh.VerticesPerPrimitive);
        writer.Write((int)mesh.Topology);
        writer.Write(mesh.ThreadgroupMemoryBytes);
        writer.Write(mesh.DrawInfoBufferIndex);
        writer.Write(mesh.IndexBufferIndex);
        writer.Write(mesh.ReadsIndexBuffer);
        writer.Write(mesh.GeometryVerticesOut);
        writer.Write(mesh.NarrowWave64);
        writer.Write(mesh.DeviceLds);
        writer.Write(mesh.Wave64);
        writer.Write(mesh.LdsBytes);
        writer.Write(mesh.Amplified);
        writer.Write(mesh.GeometryInputVertices);
        writer.Write(mesh.GeometryInputPrimitives);
    }

    private static Gen5MslMeshInfo? ReadMeshInfo(BinaryReader reader) =>
        !reader.ReadBoolean()
            ? null
            : new Gen5MslMeshInfo(
                ThreadsPerThreadgroup: reader.ReadUInt32(),
                WavesPerThreadgroup: reader.ReadUInt32(),
                MaxVertices: reader.ReadUInt32(),
                MaxPrimitives: reader.ReadUInt32(),
                VerticesPerPrimitive: reader.ReadUInt32(),
                Topology: Defined((Gen5MslMeshTopology)reader.ReadInt32()),
                ThreadgroupMemoryBytes: reader.ReadUInt32(),
                DrawInfoBufferIndex: reader.ReadUInt32(),
                IndexBufferIndex: reader.ReadUInt32(),
                ReadsIndexBuffer: reader.ReadBoolean(),
                GeometryVerticesOut: reader.ReadUInt32(),
                NarrowWave64: reader.ReadBoolean(),
                DeviceLds: reader.ReadBoolean(),
                Wave64: reader.ReadBoolean(),
                LdsBytes: reader.ReadUInt32(),
                Amplified: reader.ReadBoolean(),
                GeometryInputVertices: reader.ReadUInt32(),
                GeometryInputPrimitives: reader.ReadUInt32());

    private static void WriteTessellation(BinaryWriter writer, Gen5TessellationLaunch? launch)
    {
        writer.Write(launch is not null);
        if (launch is null)
        {
            return;
        }

        writer.Write((byte)launch.Role);
        writer.Write(launch.PatchesPerGroup);
        writer.Write(launch.ControlPoints);
        writer.Write(launch.OffchipBlockBytes);
        writer.Write(launch.FactorBytesPerPatch);
        writer.Write((byte)launch.Partition);
        writer.Write(launch.OutputClockwise);
        writer.Write(launch.MaxFactor);
        writer.Write(launch.Quad);
    }

    private static Gen5TessellationLaunch? ReadTessellation(BinaryReader reader) =>
        !reader.ReadBoolean()
            ? null
            : new Gen5TessellationLaunch(
                Role: Defined((Gen5TessellationRole)reader.ReadByte()),
                PatchesPerGroup: reader.ReadUInt32(),
                ControlPoints: reader.ReadUInt32(),
                OffchipBlockBytes: reader.ReadUInt32(),
                FactorBytesPerPatch: reader.ReadUInt32(),
                Partition: Defined((Gen5TessellationPartition)reader.ReadByte()),
                OutputClockwise: reader.ReadBoolean(),
                MaxFactor: reader.ReadUInt32(),
                Quad: reader.ReadBoolean());

    private static int Count(BinaryReader reader)
    {
        var count = reader.ReadInt32();
        return count is >= 0 and <= MaxListCount ? count : throw new InvalidDataException($"count {count} out of range");
    }

    private static T Defined<T>(T value) where T : struct, Enum =>
        Enum.IsDefined(value) ? value : throw new InvalidDataException($"{typeof(T).Name} {value} is not defined");
}
