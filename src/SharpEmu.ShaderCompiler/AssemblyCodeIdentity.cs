// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;

namespace SharpEmu.ShaderCompiler;

/// <summary>
/// An identity of an assembly's code that changes when its code does and not otherwise,
/// for keying what a translator build produces (prewarm lists, learned loads, analysis
/// graphs). The module version id is no such identity: the build stamps every assembly
/// with the commit it was built from (the informational version's "+sha", and the PDB
/// the debug directory names), so every commit gave every assembly a new one, and every
/// start after any commit threw away the shader lists and learned loads of the build
/// before it, compiling Demon's Souls' libraries cold again. This hashes the metadata
/// with the module id, the assembly-level attribute values and the RVAs of method bodies
/// and field data blanked, then every method body, every field's initial data and every
/// embedded resource by content; type, member and string-literal changes all reach the
/// metadata, and code changes reach a body. The IL image and the osx-x64 and osx-arm64
/// ReadyToRun images of one build therefore share it, and so do builds that changed only
/// other assemblies: with the RVAs in, every bundle build gave the GPU host's translators
/// a new stamp, and each start after one dropped the title's library list and prewarm
/// progress and wrote a new set of pipeline segments.
/// </summary>
public static class AssemblyCodeIdentity
{
    private static readonly ConcurrentDictionary<Assembly, string> Identities = new();

    /// <summary>The code identity of <paramref name="assembly"/>, hex; computed once per process.</summary>
    public static string Of(Assembly assembly) => Identities.GetOrAdd(assembly, Compute);

    private static string Compute(Assembly assembly)
    {
        try
        {
            if (!string.IsNullOrEmpty(assembly.Location))
            {
                using var stream = File.OpenRead(assembly.Location);
                return Compute(stream);
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or BadImageFormatException or InvalidOperationException)
        {
        }

        // No readable image (single-file or in-memory): fall back to the build's own id,
        // which is never stale, only too fresh.
        return "mvid-" + assembly.ManifestModule.ModuleVersionId.ToString("N");
    }

    /// <summary>The code identity of the PE image in <paramref name="image"/>.</summary>
    public static string Compute(Stream image)
    {
        using var pe = new PEReader(image);
        var reader = pe.GetMetadataReader();
        var bytes = pe.GetMetadata().GetContent().ToArray();

        // The module id (and any edit-and-continue ids) live in the #GUID heap.
        var guids = reader.GetHeapMetadataOffset(HeapIndex.Guid);
        Array.Clear(bytes, guids, reader.GetHeapSize(HeapIndex.Guid));

        // Version, commit and build attributes on the assembly and the module: each value
        // blob with its length prefix.
        var blobs = reader.GetHeapMetadataOffset(HeapIndex.Blob);
        foreach (var handle in reader.GetAssemblyDefinition().GetCustomAttributes()
                     .Concat(reader.GetModuleDefinition().GetCustomAttributes()))
        {
            var value = reader.GetCustomAttribute(handle).Value;
            var length = reader.GetBlobReader(value).Length;
            var prefix = length < 0x80 ? 1 : length < 0x4000 ? 2 : 4;
            Array.Clear(bytes, blobs + MetadataTokens.GetHeapOffset(value), prefix + length);
        }

        // Where a method body or a field's data sits is the image's layout, not its code:
        // ReadyToRun places them after native code it compiles per target and against the
        // assemblies referenced, so the osx-x64 and osx-arm64 images of one build, and
        // images of builds that changed only another assembly, list other RVAs for the same
        // bodies. The bodies and the data are hashed by content below.
        foreach (var table in (ReadOnlySpan<TableIndex>)[TableIndex.MethodDef, TableIndex.FieldRva])
        {
            var offset = reader.GetTableMetadataOffset(table);
            var rowSize = reader.GetTableRowSize(table);
            for (var row = 0; row < reader.GetTableRowCount(table); row++)
            {
                // The RVA is each row's first column, four bytes.
                Array.Clear(bytes, offset + row * rowSize, sizeof(uint));
            }
        }

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(bytes);
        Span<byte> number = stackalloc byte[sizeof(int)];
        foreach (var handle in reader.MethodDefinitions)
        {
            var rva = reader.GetMethodDefinition(handle).RelativeVirtualAddress;
            if (rva == 0)
            {
                continue;
            }

            // The header, the IL and the exception regions.
            var section = pe.GetSectionData(rva);
            var size = Math.Min(pe.GetMethodBody(rva).Size, section.Length);
            hash.AppendData(section.GetContent(0, size).AsSpan());
        }

        foreach (var handle in reader.FieldDefinitions)
        {
            var field = reader.GetFieldDefinition(handle);
            var rva = field.GetRelativeVirtualAddress();
            if (rva == 0)
            {
                continue;
            }

            var section = pe.GetSectionData(rva);
            var size = Math.Min(FieldDataSize(reader, field), section.Length);
            BitConverter.TryWriteBytes(number, size);
            hash.AppendData(number);
            hash.AppendData(section.GetContent(0, size).AsSpan());
        }

        // Embedded resources are code too: the MSL translator renders its prelude and fixed
        // shaders from .msl templates it embeds. Each is a length-prefixed blob in the
        // resources directory.
        var resources = pe.PEHeaders.CorHeader?.ResourcesDirectory ?? default;
        if (resources.Size > 0)
        {
            foreach (var handle in reader.ManifestResources)
            {
                var resource = reader.GetManifestResource(handle);
                if (!resource.Implementation.IsNil)
                {
                    continue;
                }

                var data = pe.GetSectionData(resources.RelativeVirtualAddress + (int)resource.Offset);
                if (data.Length < sizeof(int))
                {
                    continue;
                }

                var content = data.GetContent(0, sizeof(int)).AsSpan();
                var size = Math.Min(BitConverter.ToInt32(content), data.Length - sizeof(int));
                BitConverter.TryWriteBytes(number, size);
                hash.AppendData(number);
                hash.AppendData(data.GetContent(sizeof(int), Math.Max(size, 0)).AsSpan());
            }
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    // The bytes a field's initial data spans: a primitive's size, or the layout size of the
    // value type the compiler made for an array initializer.
    private static int FieldDataSize(MetadataReader reader, FieldDefinition field)
    {
        var signature = reader.GetBlobReader(field.Signature);
        signature.ReadSignatureHeader();
        var code = signature.ReadByte();
        switch ((SignatureTypeCode)code)
        {
            case SignatureTypeCode.Boolean or SignatureTypeCode.SByte or SignatureTypeCode.Byte:
                return 1;
            case SignatureTypeCode.Char or SignatureTypeCode.Int16 or SignatureTypeCode.UInt16:
                return 2;
            case SignatureTypeCode.Int32 or SignatureTypeCode.UInt32 or SignatureTypeCode.Single:
                return 4;
            case SignatureTypeCode.Int64 or SignatureTypeCode.UInt64 or SignatureTypeCode.Double:
                return 8;
            case (SignatureTypeCode)0x11: // ELEMENT_TYPE_VALUETYPE
                var type = signature.ReadTypeHandle();
                return type.Kind == HandleKind.TypeDefinition
                    ? reader.GetTypeDefinition((TypeDefinitionHandle)type).GetLayout().Size
                    : 0;
            default:
                return 0;
        }
    }
}
