// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// What one argument-buffer field holds: an array of buffer pointers or lengths, textures,
// samplers, or one pointer with its length.
public enum MslArgumentFieldKind
{
    BufferPointers,
    BufferByteCounts,
    Textures,
    Samplers,
    Pointer,
    Word,
    // The per-pipeline dispatcher step-cap counter. Metal-only and not a guest
    // descriptor: the host binds one word per shader hash and reads it back when the
    // submission completes, so a shader that hits the cap says so instead of only
    // looking slow.
    StepReport,
    // The sequence-lock table that makes a guest dword pair (a qword atomic, a GLC
    // LOAD_DWORDX2) one atomic access (Templates/pair_atomic_prelude.msl). Metal-only and
    // not a guest descriptor: the host binds one zeroed table shared by every program, so
    // two programs touching the same pair take the same lock. A program declares the
    // field only when it takes a lock; the others leave the trailing bytes unread.
    PairLocks,
    // The device handle tables of the program's V#-described image tables, one pointer per
    // bindless slot (BindingLayout.BindlessImageTableCount). Each entry is 16 bytes: a
    // texture's gpuResourceID and the class variant that reads it (sharpemu_bindless_entry).
    BindlessImageTables,
}

// One field of the stage argument buffer: its binding kind, id range and byte range.
public sealed record MslArgumentField(
    DescriptorBindingKind Kind,
    MslArgumentFieldKind FieldKind,
    string Name,
    uint Count,
    uint FirstId,
    uint ByteOffset,
    uint ByteSize);

// The argument buffer of one stage mirrors the binding layout in descriptor order. Pointers,
// textures and samplers take eight bytes each, counts take four; the host fills it directly.
public sealed class Gen5MslArgumentLayout
{
    public const int ResourcesBufferIndex = 0;
    public const int PushDataBufferIndex = 1;
    public const string StructName = "Gen5StageResources";
    public const string StepReportName = "step_report";
    public const string PairLocksName = "pair_locks";
    public const string BindlessImagesName = "bindless_images";
    public const string MipStatisticsName = "mip_stats";

    /// <summary>The bytes of one bindless table entry: gpuResourceID, variant, reserved.</summary>
    public const int BindlessEntryBytes = 16;

    /// <summary>The 32-bit sequence words in the pair-lock table; the shader picks one by
    /// bits 3..13 of a pair's address (sharpemu_pair_lock_slots).</summary>
    public const int PairLockSlots = 2048;

    /// <summary>The words after the lock slots holding the state of the 64 GWS resources,
    /// four each (sharpemu_gws_resource).</summary>
    public const int GwsResourceWords = 64 * 4;

    /// <summary>The whole table the host allocates zeroed for the pair_locks field.</summary>
    public const int PairLockTableWords = PairLockSlots + GwsResourceWords;
    private const uint ResourceBytes = 8;
    private const uint WordBytes = 4;

    private Gen5MslArgumentLayout(IReadOnlyList<MslArgumentField> fields, uint byteSize)
    {
        Fields = fields;
        ByteSize = byteSize;
    }

    public IReadOnlyList<MslArgumentField> Fields { get; }

    public uint ByteSize { get; }

    public MslArgumentField? Find(DescriptorBindingKind kind, MslArgumentFieldKind fieldKind) =>
        Fields.FirstOrDefault(field => field.Kind == kind && field.FieldKind == fieldKind);

    public static Gen5MslArgumentLayout Build(BindingLayout layout)
    {
        var fields = new List<MslArgumentField>();
        uint id = 0;
        uint offset = 0;
        foreach (var binding in layout.Descriptors)
        {
            var count = (uint)binding.Resources.Count;
            switch (binding.Kind)
            {
                case DescriptorBindingKind.Buffers:
                    Add(binding.Kind, MslArgumentFieldKind.BufferPointers, "buffers", count, ResourceBytes);
                    Add(binding.Kind, MslArgumentFieldKind.BufferByteCounts, "buffer_bytes", count, WordBytes);
                    break;
                case DescriptorBindingKind.Samplers:
                    Add(binding.Kind, MslArgumentFieldKind.Samplers, "samplers", count, ResourceBytes);
                    break;
                case DescriptorBindingKind.GlobalDataShare:
                    Add(binding.Kind, MslArgumentFieldKind.Pointer, "global_data_share", 1, ResourceBytes);
                    Add(binding.Kind, MslArgumentFieldKind.Word, "global_data_share_bytes", 1, WordBytes);
                    break;
                case DescriptorBindingKind.DeviceAddressPageTable:
                    Add(binding.Kind, MslArgumentFieldKind.Pointer, "address_ranges", 1, ResourceBytes);
                    Add(binding.Kind, MslArgumentFieldKind.Word, "address_range_count", 1, WordBytes);
                    break;
                case DescriptorBindingKind.FaultBuffer:
                    Add(binding.Kind, MslArgumentFieldKind.Pointer, "fault_bits", 1, ResourceBytes);
                    Add(binding.Kind, MslArgumentFieldKind.Word, "fault_word_count", 1, WordBytes);
                    break;
                case DescriptorBindingKind.FlattenedResourceTable:
                    Add(binding.Kind, MslArgumentFieldKind.Pointer, "flattened_table", 1, ResourceBytes);
                    Add(binding.Kind, MslArgumentFieldKind.Word, "flattened_table_words", 1, WordBytes);
                    break;
                case DescriptorBindingKind.ShaderData:
                    Add(binding.Kind, MslArgumentFieldKind.Pointer, "shader_data", 1, ResourceBytes);
                    Add(binding.Kind, MslArgumentFieldKind.Word, "shader_data_words", 1, WordBytes);
                    break;
                case DescriptorBindingKind.MipStatistics:
                    // The GET_LOD_STATS counter bank (MipStatistics.CounterBufferBytes): the
                    // finest level, clamped and unclamped sample counts of each counter id.
                    Add(binding.Kind, MslArgumentFieldKind.Pointer, MipStatisticsName, 1, ResourceBytes);
                    break;
                default:
                    Add(binding.Kind, MslArgumentFieldKind.Textures, ImageClassName(binding.Kind), count, ResourceBytes);
                    break;
            }
        }

        // After the descriptors and before the trailing pointers: a program declares the
        // pair-lock pointer only when it takes a lock, so nothing may follow it.
        if (layout.BindlessImageTableCount != 0)
        {
            Add(DescriptorBindingKind.Buffers, MslArgumentFieldKind.BindlessImageTables, BindlessImagesName,
                layout.BindlessImageTableCount, ResourceBytes);
        }

        // Always last, so adding them never moves a descriptor field.
        Add(DescriptorBindingKind.Buffers, MslArgumentFieldKind.StepReport, StepReportName, 1, ResourceBytes);
        Add(DescriptorBindingKind.Buffers, MslArgumentFieldKind.PairLocks, PairLocksName, 1, ResourceBytes);

        return new Gen5MslArgumentLayout(fields, Align(offset, ResourceBytes));

        void Add(DescriptorBindingKind kind, MslArgumentFieldKind fieldKind, string name, uint count, uint elementBytes)
        {
            offset = Align(offset, elementBytes);
            var byteSize = count * elementBytes;
            fields.Add(new MslArgumentField(kind, fieldKind, name, count, id, offset, byteSize));
            id += count;
            offset += byteSize;
        }
    }

    private static uint Align(uint value, uint alignment) => (value + alignment - 1) & ~(alignment - 1);

    // The field name of an image class: its resource class, numeric class and dimension.
    public static string ImageClassName(DescriptorBindingKind kind)
    {
        var (resourceClass, numericClass, dimension, atomic) = ImageDescriptorBinding.Describe(kind);
        if (resourceClass == ImageResourceClass.None)
        {
            throw new InvalidOperationException($"binding kind {kind} is not an image class");
        }

        var classPart = atomic ? "atomic" : resourceClass == ImageResourceClass.Storage ? "storage" :
            ImageDescriptorBinding.IsDepthCompare(kind) ? "sampled_compare" : "sampled";
        var numericPart = numericClass switch
        {
            ImageNumericClass.Uint => "uint",
            ImageNumericClass.Uint64 => "uint64",
            ImageNumericClass.Sint => "sint",
            _ => "float",
        };
        var dimensionPart = ImageDescriptorBinding.IsCube(kind) ? "cube" : DimensionName(dimension);
        return $"{classPart}_{numericPart}_{dimensionPart}";
    }

    public static string DimensionName(ImageDimension dimension) => dimension switch
    {
        ImageDimension.Dim1D => "1d",
        ImageDimension.Dim1DArray => "1d_array",
        ImageDimension.Dim2DArray => "2d_array",
        ImageDimension.Dim2DMsaa => "2d_ms",
        ImageDimension.Dim2DMsaaArray => "2d_ms_array",
        ImageDimension.Dim3D => "3d",
        _ => "2d",
    };

    // The Metal texture type of an image class with the given component and access.
    //
    // Guest cube maps are not declared as texturecube. Gen5 shaders resolve the cube
    // face themselves (V_CUBEID/V_CUBESC/V_CUBETC/V_CUBEMA) and then address the image
    // as a 2D array with the face in the layer coordinate, so ImageDescriptorBinding
    // describes every cube binding as Dim2DArray and the host binds a 2D-array view.
    // A texturecube declaration would instead demand a float3 direction vector. This
    // mirrors the SPIR-V backend, which likewise declares cube bindings Dim2D+Arrayed.
    public static string TextureType(ImageDimension dimension, string componentType, string? access, bool depthCompare = false)
    {
        if (depthCompare)
        {
            if (componentType != "float" || access is not null ||
                dimension is not (ImageDimension.Dim2D or ImageDimension.Dim2DArray))
                throw new NotSupportedException($"Metal depth comparison does not support {dimension}/{componentType}/{access}.");
            return dimension == ImageDimension.Dim2D ? "depth2d<float>" : "depth2d_array<float>";
        }
        var typeName = dimension switch
        {
            ImageDimension.Dim1D => "texture2d",
            ImageDimension.Dim1DArray => "texture2d_array",
            ImageDimension.Dim2DArray => "texture2d_array",
            ImageDimension.Dim2DMsaa => "texture2d_ms",
            ImageDimension.Dim2DMsaaArray => "texture2d_ms_array",
            ImageDimension.Dim3D => "texture3d",
            _ => "texture2d",
        };
        return access is null ? $"{typeName}<{componentType}>" : $"{typeName}<{componentType}, access::{access}>";
    }
}
