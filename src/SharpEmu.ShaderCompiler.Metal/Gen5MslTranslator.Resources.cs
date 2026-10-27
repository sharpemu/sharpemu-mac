// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Text;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// The resource path over a compile request: every binding comes from the layout through one
// argument buffer, every descriptor index from the memory table, user data from push data.
public static partial class Gen5MslTranslator
{
    // The range table holds at most 65536 entries; the resolver's binary search needs
    // 2^AddressRangeSearchSteps > MaxAddressRangeCount, so 17 steps cover 65536. Raise both together.
    // Demon's Souls maps 1834 spans in play and splits them into more than 4096 runs over
    // 1 GiB guest windows; the table is sorted, so two more steps buy four times the room.
    // Black Myth: Wukong's 1469 spans, mapped a page at a time from scattered backing, passed
    // 16384 runs two minutes into its first fight and ended the run.
    public const uint AddressRangeSearchSteps = 17;
    public const uint MaxAddressRangeCount = 65536;

    /// <summary>The device-address helpers every program with a page table carries, for device tests.</summary>
    internal static string DeviceAddressPrelude()
    {
        var source = new StringBuilder();
        CompilationContext.EmitResourcePreludeForTests(source);
        return source.ToString();
    }

    public static bool TryCompileProgram(
        ShaderCompileRequest request,
        out Gen5MslShader shader,
        out string error,
        Gen5MslCodegenOptions? options = null)
    {
        shader = default!;
        if (request.UsesBindlessHeap)
        {
            error = "Metal does not support bindless heap descriptors.";
            return false;
        }

        // The guest mesh stage (Vulkan's task/mesh path) has no MSL emission; the Metal mesh path
        // compiles NGG primitive shaders as Vertex requests with PrimitiveShader set instead.
        if (request.Stage == ShaderStage.Mesh)
        {
            error = "Metal does not support mesh shaders.";
            return false;
        }
        // A program the shader-call linker rewrote carries call dispatch the MSL emitter cannot
        // express. An unlinked S_SWAPPC_B64 keeps today's behaviour (TryEmitScalar64 skips it),
        // and an explicit branch target alone (ShaderLinkedBranchControl) compiles.
        if (request.Program.Instructions.Any(instruction =>
            instruction.Control is ShaderCallMatchControl or ShaderCallEntryControl or ShaderCallFaultControl))
        {
            error = "Metal does not support linked shader calls.";
            return false;
        }
        try
        {
            BindingLayoutValidator.Validate(
                request.Bindings,
                request.Resources.Info,
                BindingLayout.CollectUserDataRegisters(request.Program, request.UserDataBase, request.UserDataCount,
                    request.ExcludedUserDataRegisters),
                request.UsesGlobalDataShare,
                request.UsesFlattenedTable,
                request.ReadsShaderBase,
                request.Hash,
                request.Stage);
        }
        catch (ResourcePlanException exception)
        {
            error = exception.Message;
            return false;
        }

        if (request.Stage == ShaderStage.Pixel && !ValidatePixelOutputs(request.PixelOutputs, out error))
        {
            return false;
        }

        // Metal rounds every float result to nearest even, so a launch FLOAT_MODE whose FP_ROUND
        // fields (bits 3:0) pick another rounding cannot be honoured, as for S_ROUND_MODE.
        if ((request.FloatMode & 0xFu) != 0)
        {
            error = $"FLOAT_MODE 0x{request.FloatMode:X2}: only round-to-nearest-even (FP_ROUND 0) is supported";
            return false;
        }

        if (!new CompilationContext(request, options ?? Gen5MslCodegenOptions.FromEnvironment)
                .TryCompile(out shader, out error))
        {
            return false;
        }

        Gen5MslDump.Write(request, shader);
        return true;
    }

    private sealed partial class CompilationContext
    {
        private const string ResourcesName = "sharpemu_resources";
        private const string PushDataName = "sharpemu_push_data";
        private const string GlobalDataShareName = "sharpemu_gds";
        private const string GlobalDataShareWordsName = "sharpemu_gds_words";
        private const string DeviceArguments = "sharpemu_ranges, sharpemu_range_count, sharpemu_faults, sharpemu_fault_words, sharpemu_range_cache";

        private readonly ShaderCompileRequest _request;
        private readonly Gen5MslArgumentLayout? _argumentLayout;
        private readonly Dictionary<DescriptorBindingKind, MslImageClass> _imageClasses = [];
        private readonly Dictionary<int, string> _indirectKeyScratch = [];
        // The handle table and entry index of the bindless access being emitted, which its
        // constant variant cases read their texture through.
        private (string Table, string Index)? _bindlessAccess;
        private bool _hasGlobalDataShare;
        private bool _hasDeviceAddressTable;
        private bool _hasFlattenedTable;
        private bool _hasSamplers;
        // The layout binds the mip-statistics counters (BindingLayout.MipStatisticsCount):
        // some sampled image of this pixel program is eligible to record (the T#'s
        // MIP_STATS_ENABLE decides at run time).
        private bool _hasMipStatistics;
        // A sample records into them, so the module carries mip_stats_prelude.
        private bool _usesMipStatistics;

        // The sites sit behind the function constant (Gen5MslCodegenOptions.MipStatisticsFunctionConstant),
        // which the module then declares.
        private bool UsesMipStatisticsConstant => _usesMipStatistics && _options.MipStatisticsFunctionConstant;

        private readonly record struct MslImageClass(
            string Field,
            string TextureType,
            string ComponentKind,
            bool IsStorage,
            bool DepthCompare,
            bool Multisampled,
            bool Arrayed,
            bool Cube,
            ImageDimension Dimension,
            IReadOnlyList<uint> Resources)
        {
            /// <summary>Spatial coordinate components, excluding any array layer.</summary>
            public int SpatialComponents => Dimension switch
            {
                ImageDimension.Dim1D or ImageDimension.Dim1DArray => 1,
                ImageDimension.Dim3D => 3,
                _ => 2,
            };

            /// <summary>Address components one access consumes for its coordinates.</summary>
            public int CoordinateComponents => SpatialComponents + (Arrayed ? 1 : 0);

            /// <summary>Metal spatial coordinate vector type for integer addressing.</summary>
            public string IntegerCoordinateType => SpatialComponents switch
            {
                1 => "uint",
                3 => "uint3",
                _ => "uint2",
            };

            /// <summary>Metal spatial coordinate vector type for normalized addressing.</summary>
            public string FloatCoordinateType => SpatialComponents switch
            {
                1 => "float",
                3 => "float3",
                _ => "float2",
            };

            // Guest 1D images are represented as height-one 2D images so their
            // mip/LOD/derivative semantics survive Metal's non-mipmapped 1D API.
            public bool SupportsSampleLod => !Multisampled;
            public bool SupportsMipLevels => !Multisampled;

            /// <summary>Metal exposes gather on 2D and 2D-array textures only.</summary>
            public bool SupportsGather => Dimension
                is ImageDimension.Dim2D or ImageDimension.Dim2DArray;
        }

        private readonly Gen5MslCodegenOptions _options;

        public CompilationContext(ShaderCompileRequest request, Gen5MslCodegenOptions options)
        {
            _request = request;
            _options = options;
            _argumentLayout = Gen5MslArgumentLayout.Build(request.Bindings);
            // An NGG primitive shader that exchanges data through LDS, synchronizes or asks
            // for export space is the merged ES/GS workgroup stage, which only a mesh
            // function can express. Passthrough primitive shaders keep the vertex path.
            _stage = RequiresMeshEmission(request)
                ? Gen5MslStage.Mesh
                : request.Stage switch
                {
                    ShaderStage.Vertex => Gen5MslStage.Vertex,
                    ShaderStage.Pixel => Gen5MslStage.Pixel,
                    _ => Gen5MslStage.Compute,
                };
            if (_stage == Gen5MslStage.Mesh)
            {
                _meshLdsDwords = MeshLdsDwordCount(request);
                _meshInfo = BuildMeshInfo(request, _meshLdsDwords, Math.Min(options.MeshDeviceLdsThresholdBytes, 16384u),
                    options.GeometryOutputCapacity);
                _meshHotLds = PlanMeshHotLds();
                if (_meshHotLds.Length != 0)
                    _meshInfo = _meshInfo with { ThreadgroupMemoryBytes = MeshHotLdsWords * sizeof(uint) };
            }

            _computeLdsDwords = options.DispatchThreadgroupMemory
                ? ComputeLdsDwordCount(request)
                : LdsDwordCount;

            _pixelOutputBindings = request.PixelOutputs;
            _usesPixelValidMask =
                _stage == Gen5MslStage.Pixel &&
                request.Program.Instructions.Any(static instruction => instruction.Control is Gen5ExportControl { ValidMask: true });
            // SPI_SHADER_Z_FORMAT says which MRTZ channels carry depth, stencil and the
            // sample mask; DB_SHADER_CONTROL says which of them the DB takes.
            _mrtzFormat = request.PixelDepthExportFormat;
            var mrtzExports = _stage == Gen5MslStage.Pixel
                ? request.Program.Instructions
                    .Where(static instruction => instruction.Control is Gen5ExportControl { Target: MrtzExportTarget })
                    .ToArray()
                : [];
            _pixelDepthExport = request.PixelDepthExportEnable &&
                mrtzExports.Any(instruction => MrtzDepth(instruction) is not null);
            _pixelStencilExport = request.PixelStencilExportEnable &&
                mrtzExports.Any(instruction => MrtzStencil(instruction) is not null);
            _pixelSampleMaskExport = request.PixelSampleMaskExportEnable &&
                mrtzExports.Any(instruction => MrtzSampleMask(instruction) is not null);
            _localSizeX = Math.Max(request.LocalSizeX, 1);
            _localSizeY = Math.Max(request.LocalSizeY, 1);
            _localSizeZ = Math.Max(request.LocalSizeZ, 1);
            _waveLaneCount = request.WaveSize == 64 ? 64u : 32u;
            _requiredVertexOutputCount = request.RequiredVertexOutputCount;
            _pixelInputEnable = request.PixelInputEnable;
            _pixelInputAddress = request.PixelInputAddress;
            _pixelInputCntl = new uint[32];
            for (uint index = 0; index < 32u; index++)
            {
                _pixelInputCntl[index] = request.PixelInputCntl is not null && index < (uint)request.PixelInputCntl.Count
                    ? request.PixelInputCntl[(int)index]
                    : index;
            }
        }

        // ---- declarations ----

        // Records which bindings the layout holds and the texture type of every image class.
        private void DeclareLayoutBindings()
        {
            var request = _request;
            var info = request.Resources.Info;
            foreach (var binding in request.Bindings.Descriptors)
            {
                switch (binding.Kind)
                {
                    case DescriptorBindingKind.Buffers:
                        break;
                    case DescriptorBindingKind.Samplers:
                        _hasSamplers = true;
                        break;
                    case DescriptorBindingKind.GlobalDataShare:
                        _hasGlobalDataShare = true;
                        break;
                    case DescriptorBindingKind.DeviceAddressPageTable:
                        _hasDeviceAddressTable = true;
                        break;
                    case DescriptorBindingKind.FaultBuffer:
                        break;
                    case DescriptorBindingKind.FlattenedResourceTable:
                        _hasFlattenedTable = true;
                        break;
                    case DescriptorBindingKind.ShaderData:
                        break;
                    case DescriptorBindingKind.MipStatistics:
                        _hasMipStatistics = true;
                        break;
                    default:
                        DeclareImageClass(binding, info);
                        break;
                }
            }

            _hasDeviceAddressTable &= request.Bindings.Find(DescriptorBindingKind.FaultBuffer) is not null;
            foreach (var memoryIndex in request.IndirectKeyMemoryIndices)
            {
                _indirectKeyScratch[memoryIndex] = $"indirect_image_key{memoryIndex}";
            }
        }

        // A storage class takes the least access its elements need; a sampled class none.
        private void DeclareImageClass(DescriptorBinding binding, ShaderResourceInfo info)
        {
            var (resourceClass, numericClass, dimension, atomic) = ImageDescriptorBinding.Describe(binding.Kind);
            var cube = ImageDescriptorBinding.IsCube(binding.Kind);
            if (resourceClass == ImageResourceClass.None)
            {
                throw new InvalidOperationException($"binding kind {binding.Kind} is not an image class");
            }

            var isStorage = resourceClass == ImageResourceClass.Storage;
            var componentKind = numericClass switch
            {
                ImageNumericClass.Uint or ImageNumericClass.Uint64 => "uint",
                ImageNumericClass.Sint => "int",
                _ => "float",
            };

            // 64-bit image atomics need the texture typed ulong (over an RG32Uint
            // texture), which one array of 32-bit atomic textures cannot share.
            var wideAtomics = atomic
                ? binding.Resources.Count(resource => info.Images[(int)resource].Atomic64)
                : 0;
            if (wideAtomics != 0)
            {
                if (wideAtomics != binding.Resources.Count)
                {
                    throw new NotSupportedException(
                        $"binding kind {binding.Kind} mixes 32-bit and 64-bit image atomics");
                }

                componentKind = "ulong";
            }
            string? access = null;
            if (isStorage)
            {
                var reads = false;
                var writes = false;
                foreach (var resource in binding.Resources)
                {
                    var image = info.Images[(int)resource];
                    reads |= image.Read || image.Atomic || atomic;
                    writes |= image.Written || image.Atomic || atomic;
                }

                access = writes ? (reads ? "read_write" : "write") : "read";
            }

            var multisampled = dimension is ImageDimension.Dim2DMsaa or ImageDimension.Dim2DMsaaArray;
            var arrayed = dimension is ImageDimension.Dim1DArray or ImageDimension.Dim2DArray or ImageDimension.Dim2DMsaaArray;
            _imageClasses[binding.Kind] = new MslImageClass(
                Gen5MslArgumentLayout.ImageClassName(binding.Kind),
                Gen5MslArgumentLayout.TextureType(dimension, componentKind, access, ImageDescriptorBinding.IsDepthCompare(binding.Kind)),
                componentKind,
                isStorage,
                ImageDescriptorBinding.IsDepthCompare(binding.Kind),
                multisampled,
                arrayed,
                cube,
                dimension,
                binding.Resources);
        }

        internal static void EmitResourcePreludeForTests(StringBuilder source) => EmitResourcePrelude(source);

        private static void EmitResourcePrelude(StringBuilder source) =>
            source.Append(MslTemplates.Render(
                "resource_prelude",
                ("address_mask", FormatULong(DeviceAddressPaging.AddressMask)),
                ("page_bits", DeviceAddressPaging.PageBits.ToString(CultureInfo.InvariantCulture)),
                ("page_offset_mask", FormatULong(DeviceAddressPaging.PageOffsetMask)),
                ("search_steps", AddressRangeSearchSteps.ToString(CultureInfo.InvariantCulture))));

        // The argument buffer struct: one field per layout descriptor in layout order.
        private void EmitResourceStruct(StringBuilder source)
        {
            var layout = _argumentLayout!;
            if (layout.Fields.Count == 0)
            {
                return;
            }

            source.AppendLine();
            if (layout.Fields.Any(static field => field.FieldKind == MslArgumentFieldKind.BindlessImageTables))
            {
                EmitBindlessEntryStructs(source);
            }

            source.AppendLine($"struct {Gen5MslArgumentLayout.StructName}");
            source.AppendLine("{");
            foreach (var field in layout.Fields)
            {
                // The trailing pair-lock pointer is declared only by a program that takes a
                // lock, so every other program's source stays as it was.
                if (field.FieldKind == MslArgumentFieldKind.PairLocks && !_usesPairLocks)
                {
                    continue;
                }

                if (field.FieldKind == MslArgumentFieldKind.BufferPointers &&
                    Enumerable.Range(0, (int)field.Count).Any(IsScalarConstantBuffer))
                {
                    // Keep the host's pointer array ABI: each pointer retains its id and
                    // eight-byte slot, even when adjacent pointers have different types.
                    for (var index = 0; index < field.Count; index++)
                        source.AppendLine($"    {(IsScalarConstantBuffer(index) ? "constant" : "device")} uint* buffer_{index} [[id({field.FirstId + index})]];");
                    continue;
                }

                var declaration = field.FieldKind switch
                {
                    MslArgumentFieldKind.BufferPointers => $"array<device uint*, {field.Count}> {field.Name}",
                    MslArgumentFieldKind.BufferByteCounts => $"array<uint, {field.Count}> {field.Name}",
                    MslArgumentFieldKind.Textures => $"array<{_imageClasses[field.Kind].TextureType}, {field.Count}> {field.Name}",
                    MslArgumentFieldKind.Samplers => $"array<sampler, {field.Count}> {field.Name}",
                    MslArgumentFieldKind.Pointer => $"{PointerType(field.Kind)} {field.Name}",
                    MslArgumentFieldKind.StepReport or MslArgumentFieldKind.PairLocks => $"device atomic_uint* {field.Name}",
                    MslArgumentFieldKind.BindlessImageTables => $"array<device const {BindlessEntryStruct}*, {field.Count}> {field.Name}",
                    _ => $"uint {field.Name}",
                };
                source.AppendLine($"    {declaration} [[id({field.FirstId})]];");
            }

            source.AppendLine("};");
            source.AppendLine();
        }

        private const string BindlessEntryStruct = "sharpemu_bindless_entry";

        // One bindless table entry is a texture handle and the variant that reads it; each
        // texture type a variant uses gets the same layout with the handle typed, so a case
        // reads its texture by reinterpreting the entry.
        private void EmitBindlessEntryStructs(StringBuilder source)
        {
            source.AppendLine($"struct {BindlessEntryStruct} {{ ulong handle; uint variant; uint reserved; }};");
            var info = _request.Resources.Info;
            var declared = new HashSet<string>(StringComparer.Ordinal);
            foreach (var image in info.Images)
            {
                if (image.BindlessSlot == DescriptorConstants.NoIndex ||
                    ImageDescriptorBinding.ForImage(image) is not { } kind ||
                    !_imageClasses.TryGetValue(kind, out var imageClass) ||
                    !declared.Add(imageClass.TextureType))
                {
                    continue;
                }

                source.AppendLine(
                    $"struct {BindlessTypedStruct(imageClass.TextureType)} {{ {imageClass.TextureType} texture; uint variant; uint reserved; }};");
            }

            source.AppendLine();
        }

        private static string BindlessTypedStruct(string textureType)
        {
            var name = new StringBuilder("sharpemu_bindless_");
            foreach (var character in textureType)
            {
                name.Append(char.IsAsciiLetterOrDigit(character) ? character : '_');
            }

            return name.ToString();
        }

        // The user-data words and the flattened resource table are read-only tables the
        // host fills before the encode and nothing on the GPU writes, and every read of
        // them is at an index the program knows — so they belong in the constant address
        // space, which is Apple's uniform datapath. A shader reading them from "device"
        // pays a per-lane memory access for a value that is the same in every lane:
        // Octopath Traveler II's heaviest vertex shader reads 32 user-data words in its
        // preamble, which the hardware does once per wave and we were doing 12.5 million
        // times a frame.
        private string PointerType(DescriptorBindingKind kind) => kind switch
        {
            DescriptorBindingKind.DeviceAddressPageTable => "device const Gen5AddressRange*",
            DescriptorBindingKind.FlattenedResourceTable or DescriptorBindingKind.ShaderData =>
                _options.UniformTablesInConstantSpace ? "constant uint*" : "device const uint*",
            DescriptorBindingKind.MipStatistics => "device atomic_uint*",
            _ => "device uint*",
        };

        private void EmitResourceParameters(StringBuilder source)
        {
            if (_argumentLayout!.Fields.Count != 0)
            {
                source.AppendLine(
                    $"    constant {Gen5MslArgumentLayout.StructName}& {ResourcesName} [[buffer({Gen5MslArgumentLayout.ResourcesBufferIndex})]],");
            }

            if (_request.Bindings.UsesPushData)
            {
                source.AppendLine($"    constant uint* {PushDataName} [[buffer({Gen5MslArgumentLayout.PushDataBufferIndex})]],");
            }
        }

        // ---- initial state ----

        // Buffer locals, user data and the per-buffer byte biases from the packed offsets.
        private void EmitLayoutInitialState(StringBuilder source)
        {
            var request = _request;
            var layout = request.Bindings;
            var bufferCount = request.Resources.Info.Buffers.Count;
            var splitBufferFields = Enumerable.Range(0, bufferCount).Any(IsScalarConstantBuffer);
            for (var index = 0; index < bufferCount; index++)
            {
                var field = splitBufferFields ? $"buffer_{index}" : $"buffers[{index}]";
                source.AppendLine($"    {(IsScalarConstantBuffer(index) ? "constant" : "device")} uint* b{index} = {ResourcesName}.{field};");
            }

            if (!_options.ElideByteBias)
            {
                source.AppendLine($"    uint bias[{Math.Max(bufferCount, 1)}] = {{}};");
            }

            if (_hasGlobalDataShare)
            {
                source.AppendLine($"    device uint* {GlobalDataShareName} = {ResourcesName}.global_data_share;");
                source.AppendLine($"    uint {GlobalDataShareWordsName} = {ResourcesName}.global_data_share_bytes >> 2u;");
            }

            if (_hasDeviceAddressTable)
            {
                source.AppendLine($"    device const Gen5AddressRange* sharpemu_ranges = {ResourcesName}.address_ranges;");
                source.AppendLine($"    uint sharpemu_range_count = ({ResourcesName}.address_range_count & 0x80000000u) | min({ResourcesName}.address_range_count & 0x7FFFFFFFu, {MaxAddressRangeCount}u);");
                source.AppendLine($"    device uint* sharpemu_faults = {ResourcesName}.fault_bits;");
                source.AppendLine($"    uint sharpemu_fault_words = {ResourcesName}.fault_word_count;");
                source.AppendLine("    Gen5RangeCache sharpemu_range_cache = { 0ul, 0ul, nullptr };");
            }

            for (var index = 0; index < layout.UserDataRegisters.Count; index++)
            {
                var register = layout.UserDataRegisters[index];
                if (register < ScalarRegisterFileCount)
                {
                    source.AppendLine($"    s[{register}] = {ShaderDataDword((uint)index)};");
                }
            }

            for (uint buffer = 0; !_options.ElideByteBias && buffer < layout.MemoryOffsetCount; buffer++)
            {
                source.AppendLine(
                    $"    bias[{buffer}] = ({ShaderDataDword(layout.MemoryOffsetDword + (buffer / 4))} >> {(buffer % 4) * 8}u) & 0xFFu;");
            }

            foreach (var scratch in _indirectKeyScratch.Values.OrderBy(static name => name, StringComparer.Ordinal))
            {
                source.AppendLine($"    uint {scratch} = 0u;");
            }
        }

        // One dword of this stage's block: from push data, or the shader-data buffer when spilled.
        private string ShaderDataDword(uint index)
        {
            var layout = _request.Bindings;
            if (layout.UsesPushData)
            {
                return $"{PushDataName}[{layout.PushDataStartDword + index}]";
            }

            return $"({index}u < {ResourcesName}.shader_data_words ? {ResourcesName}.shader_data[{index}u] : 0u)";
        }

        private string ComputeThreadLimit(uint component)
        {

            if (_request.Bindings is { UsesDispatchThreadLimits: true } layout)
            {
                return ShaderDataDword(layout.DispatchThreadLimitsDword + component);
            }

            var limit = component switch
            {
                0 => _request.ThreadCountX,
                1 => _request.ThreadCountY,
                _ => _request.ThreadCountZ,
            };
            return FormatUInt(limit);
        }

        // The shader base the host pushes for this draw, as two dwords.
        private (string Low, string High) ShaderBaseWords()
        {
            var layout = _request.Bindings;
            if (!layout.UsesShaderBase)
            {
                return ("0u", "0u");
            }

            return (ShaderDataDword(layout.ShaderBaseDword), ShaderDataDword(layout.ShaderBaseDword + 1));
        }

        private string LoadFlattenedWord(string slot)
        {
            // A slot fixed here is read without the select: the host pads the table it binds to
            // FlattenedTableMinimumWords with zeros, which is what the select would have read.
            if (_options.ConstantTableSlots && slot.Length > 1 && slot[^1] == 'u' &&
                uint.TryParse(slot.AsSpan(0, slot.Length - 1), NumberStyles.None, CultureInfo.InvariantCulture, out var fixedSlot) &&
                fixedSlot < MaxPaddedFlattenedTableWords)
            {
                _flattenedTableMinimumWords = Math.Max(_flattenedTableMinimumWords, fixedSlot + 1);
                return $"{ResourcesName}.flattened_table[{slot}]";
            }

            return $"({slot} < {ResourcesName}.flattened_table_words ? {ResourcesName}.flattened_table[{slot}] : 0u)";
        }

        /// <summary>A fixed slot past this keeps the select, so no shader asks the host for a
        /// table of more than 256 KB.</summary>
        private const uint MaxPaddedFlattenedTableWords = 1u << 16;

        private uint _flattenedTableMinimumWords;

        private static string FormatULong(ulong value) => $"0x{value.ToString("X", CultureInfo.InvariantCulture)}ul";

        // A signed immediate widened to the address width.
        private static string SignedOffset64(int offset) => FormatULong(unchecked((ulong)(long)offset));

        // A written access is allowed only inside its handle's tracked range:
        // width <= size, address >= base, address - base <= size - width.
        private string IsWrittenAccessAllowed(int memoryIndex, string address, uint widthBytes)
        {
            if (!_hasFlattenedTable || !_request.WrittenRangeSlotByMemoryIndex.TryGetValue(memoryIndex, out var slot))
            {
                return "false";
            }

            var rangeBase = Temp("ulong", $"(ulong){LoadFlattenedWord($"{slot}u")} | ((ulong){LoadFlattenedWord($"{slot + 1}u")} << 32)");
            var rangeSize = Temp("ulong", $"(ulong){LoadFlattenedWord($"{slot + 2}u")}");
            var masked = Temp("ulong", $"{address} & {FormatULong(DeviceAddressPaging.AddressMask)}");
            return Temp(
                "bool",
                $"{widthBytes}ul <= {rangeSize} && {rangeBase} <= {masked} && ({masked} - {rangeBase}) <= ({rangeSize} - {widthBytes}ul)");
        }

        // The atomic expression over a device word pointer for one opcode suffix.
        private static bool TryGetAtomicExpression(string name, string value, string comparator, out Func<string, string> build)
        {
            Func<string, string>? candidate = name switch
            {
                "Add" => pointer => $"atomic_fetch_add_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Sub" => pointer => $"atomic_fetch_sub_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Umin" => pointer => $"atomic_fetch_min_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Umax" or "UMax" => pointer => $"atomic_fetch_max_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Smin" => pointer => $"(uint)atomic_fetch_min_explicit((device atomic_int*)({pointer}), as_type<int>({value}), memory_order_relaxed)",
                "Smax" => pointer => $"(uint)atomic_fetch_max_explicit((device atomic_int*)({pointer}), as_type<int>({value}), memory_order_relaxed)",
                "And" => pointer => $"atomic_fetch_and_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Or" => pointer => $"atomic_fetch_or_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Xor" => pointer => $"atomic_fetch_xor_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Swap" => pointer => $"atomic_exchange_explicit((device atomic_uint*)({pointer}), {value}, memory_order_relaxed)",
                "Cmpswap" => pointer => $"sharpemu_atomic_compare_swap((device atomic_uint*)({pointer}), {comparator}, {value})",
                "Fcmpswap" => pointer => $"sharpemu_atomic_float_compare_swap((device atomic_uint*)({pointer}), {comparator}, {value})",
                "Csub" => pointer => $"sharpemu_atomic_clamped_sub((device atomic_uint*)({pointer}), {value})",
                "Inc" => pointer => $"sharpemu_atomic_step((device atomic_uint*)({pointer}), {value}, true)",
                "Dec" => pointer => $"sharpemu_atomic_step((device atomic_uint*)({pointer}), {value}, false)",
                "Fmin" => pointer => $"sharpemu_atomic_float_minmax((device atomic_uint*)({pointer}), {value}, false)",
                "Fmax" => pointer => $"sharpemu_atomic_float_minmax((device atomic_uint*)({pointer}), {value}, true)",
                _ => null,
            };
            build = candidate!;
            return candidate is not null;
        }

        // ---- scalar memory ----

        // One dword of a buffer whose V# lives in scalar registers rather than in a host
        // binding. The descriptor's first two dwords are its base address, its stride and
        // record count say how far it reaches, and a read past that is zero - the bound
        // the hardware applies - rather than whatever guest bytes follow the buffer.
        private string DeviceDescriptorWord(uint descriptorRegister, string byteOffset) =>
            DeviceDescriptorWord(DeviceDescriptorBounds(descriptorRegister), byteOffset);

        // The reach of a V# held in scalar registers: its base address, byte size and stride.
        private (string Base, string Size, string Stride) DeviceDescriptorBounds(uint descriptorRegister)
        {
            var word1 = Temp("uint", ScalarExpression(descriptorRegister + 1));
            var stride = Temp("uint", $"(({word1} >> 16u) & 0x3FFFu)");
            var records = Temp("uint", ScalarExpression(descriptorRegister + 2));
            var size = Temp("ulong", $"({stride} == 0u ? (ulong){records} : (ulong){stride} * (ulong){records})");
            // Held in a temp: the base must not be re-read from SGPRs a later destination store may have overwritten.
            var baseAddress = Temp("ulong", $"((ulong){ScalarExpression(descriptorRegister)} | ((ulong)({word1} & 0xFFFFu) << 32))");
            return (baseAddress, size, stride);
        }

        // The byte offset may differ per lane; a read past the descriptor's size is zero. A
        // GLC read takes the device-coherent value (coherent_load_prelude).
        private string DeviceDescriptorWord((string Base, string Size, string Stride) descriptor, string byteOffset, bool glc = false)
        {
            var address = DeviceDescriptorWordAddress(descriptor, byteOffset);
            _usesCoherentLoads |= glc;
            return Temp(
                "uint",
                $"({DeviceDescriptorWordInRange(descriptor, byteOffset)} ? " +
                $"sharpemu_load_device_dword{(glc ? "_coherent" : string.Empty)}({DeviceArguments}, {address}) : 0u)");
        }

        private string DeviceDescriptorWordAddress((string Base, string Size, string Stride) descriptor, string byteOffset) =>
            Temp("ulong", $"({descriptor.Base} + (ulong){byteOffset}) & {FormatULong(DeviceAddressPaging.AddressMask & ~3ul)}");

        private string DeviceDescriptorWordInRange((string Base, string Size, string Stride) descriptor, string byteOffset) =>
            DeviceBufferFits(descriptor, byteOffset, "4u");

        // Whether bytes at a byte offset from a scalar-register V#'s base are in range: the
        // OOB_SELECT check of the buffer access being emitted (_deviceBufferFits), or else the
        // descriptor's byte size (a scalar load, which has no OOB_SELECT).
        private Func<string, string, string>? _deviceBufferFits;

        private string DeviceBufferFits((string Base, string Size, string Stride) descriptor, string byteOffset, string bytes) =>
            _deviceBufferFits?.Invoke(byteOffset, bytes) ??
            $"((ulong){bytes} <= {descriptor.Size} && (ulong){byteOffset} <= {descriptor.Size} - (ulong){bytes})";

        // The range check of a MUBUF/MTBUF access through a V# in scalar registers (RDNA2 ISA
        // 8.1.5, Table 35), per element of payload bytes at unswizzled offset o (the VGPR
        // offset plus the instruction offset, no SGPR offset), by the OOB_SELECT of word 3
        // bits 28-29:
        //   0: index < NUM_RECORDS and o < stride (o + payload <= stride when formatted)
        //   1: index < NUM_RECORDS
        //   2: NUM_RECORDS != 0
        //   3: NUM_RECORDS is reduced by SOFFSET (an SOFFSET past it is out of range);
        //      swizzled, index < that and o + payload <= stride, else o + payload <= that.
        // A V# whose FORMAT is 0 (invalid) is out of range in every mode. Mode 3's SOFFSET
        // reduction follows the ISA's raw-buffer rule; Mesa and LLVM do not describe it.
        private Func<string, string, string> DeviceBufferOutOfRangeCheck(
            Dictionary<string, string> unswizzledOffsets,
            string descriptorWord3,
            string stride,
            string records,
            string soffset,
            string index,
            string swizzled,
            bool formatted)
        {
            var mode = Temp("uint", $"({descriptorWord3} >> 28u) & 3u");
            var formatValid = Temp("bool", $"(({descriptorWord3} >> 12u) & 0x7Fu) != 0u");
            var indexInRange = Temp("bool", $"{index} < {records}");
            var scalarInRange = Temp("bool", $"{soffset} <= {records}");
            var rawRecords = Temp("ulong", $"{scalarInRange} ? (ulong)({records} - {soffset}) : 0ul");
            return (byteOffset, bytes) =>
            {
                if (!unswizzledOffsets.TryGetValue(byteOffset, out var offset))
                {
                    throw new InvalidOperationException($"device buffer offset {byteOffset} was not placed by its access");
                }

                var payload = $"(ulong){bytes}";
                var at = $"(ulong){offset}";
                var structured = formatted
                    ? $"({payload} <= (ulong){stride} && {at} <= (ulong){stride} - {payload})"
                    : $"{at} < (ulong){stride}";
                var raw = $"({swizzled} ? ((ulong){index} < {rawRecords} && {payload} <= (ulong){stride} && {at} <= (ulong){stride} - {payload}) : " +
                    $"({payload} <= {rawRecords} && {at} <= {rawRecords} - {payload}))";
                return $"({formatValid} && ({mode} == 0u ? ({indexInRange} && {structured}) : {mode} == 1u ? {indexInRange} : " +
                    $"{mode} == 2u ? {records} != 0u : ({scalarInRange} && {raw})))";
            };
        }

        // A formatted load from a V# held in scalar registers: the element's bytes are read
        // through the device-address page table at an offset from the descriptor's base, and
        // a component whose bytes reach past the descriptor's size is out of range. A read
        // runs only when it is in range, so it never touches the guest bytes that follow.
        private FormatMemory DeviceFormatMemory((string Base, string Size, string Stride) descriptor, bool glc = false)
        {
            _usesCoherentLoads |= glc;
            return new(
                (byteOffset, bytes) => Temp("bool", DeviceBufferFits(descriptor, byteOffset, bytes)),
                (byteOffset, bytes, inBounds) => Temp(
                    "uint",
                    $"({bytes} != 0u && {inBounds}) ? sharpemu_load_device_unaligned{(glc ? "_coherent" : string.Empty)}({DeviceArguments}, " +
                    $"({descriptor.Base} + (ulong){byteOffset}) & {FormatULong(DeviceAddressPaging.AddressMask)}, {bytes}) : 0u"),
                LoadWords: null,
                ZeroInvalidFormat: true);
        }

        // A formatted or typed store through a V# held in scalar registers. The unified
        // format (word 3's, or a typed store's own) is decoded on the device by the prelude
        // table; each transferred component the format has is encoded with the bound path's
        // sharpemu_format_encode and its bits merged into device memory. As on the bound
        // path, all components are stored or the element is dropped: a zero descriptor
        // format, a format the table cannot decode, or an element past the descriptor's size.
        private void EmitDeviceFormatStore(
            (string Base, string Size, string Stride) descriptor,
            BufferAddress byteOffset,
            string descriptorWord3,
            string unifiedFormat,
            uint vectorData,
            uint componentCount,
            FormatD16 d16)
        {
            var entry = Temp("uint", $"sharpemu_gfx10_formats[{unifiedFormat}]");
            var dataFormat = Temp("uint", $"{entry} & 0xFFu");
            var numberFormat = Temp("uint", $"({entry} >> 8) & 0xFFu");
            var layouts = new (string ByteOffset, string BitOffset, string Bits, string Offset)[componentCount];
            var elementBytes = "0u";
            // Each component is checked at its own placed offset (its dword lands apart from
            // the others' in a swizzled V#); the element is stored only when all are in range.
            var componentsInBounds = new List<string>();
            for (uint component = 0; component < componentCount; component++)
            {
                var componentByteOffset = Temp("uint", "0u");
                var bitOffset = Temp("uint", "0u");
                var bits = Temp("uint", "0u");
                Line($"sharpemu_format_layout({dataFormat}, {component}u, {componentByteOffset}, {bitOffset}, {bits});");
                var offset = byteOffset.At(componentByteOffset);
                var componentBytes = Temp("uint", $"({bitOffset} + {bits} + 7u) >> 3u");
                componentsInBounds.Add(
                    $"({bits} == 0u || {DeviceBufferFits(descriptor, offset, componentBytes)})");
                layouts[component] = (componentByteOffset, bitOffset, bits, offset);
                elementBytes = Temp(
                    "uint",
                    $"max({elementBytes}, {bits} == 0u ? 0u : {componentByteOffset} + {componentBytes})");
            }

            var allowed = Temp(
                "bool",
                $"(({descriptorWord3} >> 12) & 0x7Fu) != 0u && {dataFormat} != 0u && {elementBytes} != 0u && " +
                string.Join(" && ", componentsInBounds));
            Line($"if ({ExecCondition(allowed)})");
            Line("{");
            _indent++;
            var mask = FormatULong(DeviceAddressPaging.AddressMask);
            for (uint component = 0; component < componentCount; component++)
            {
                var (_, bitOffset, bits, offset) = layouts[component];
                Line($"if ({bits} != 0u)");
                Line("{");
                _indent++;
                // The component's bits merge into the one or two dwords they touch.
                var address = Temp("ulong", $"({descriptor.Base} + (ulong){offset}) & {mask}");
                var position = Temp("uint", $"(uint)({address} & 3ul) * 8u + {bitOffset}");
                var aligned = Temp("ulong", $"({address} & ~3ul) + (ulong)({position} >> 5u) * 4ul");
                var shift = Temp("uint", $"{position} & 31u");
                var bitMask = Temp("uint", $"{bits} >= 32u ? 0xFFFFFFFFu : (1u << {bits}) - 1u");
                var encoded = Temp(
                    "uint",
                    $"sharpemu_format_encode({FormatStoreComponent(vectorData, component, d16, numberFormat)}, {bits}, {numberFormat}, {dataFormat}) & {bitMask}");
                var crosses = Temp("bool", $"{shift} + {bits} > 32u");
                Line($"sharpemu_store_device_masked_word({DeviceArguments}, {aligned}, {encoded} << {shift}, {bitMask} << {shift});");
                Line(
                    $"sharpemu_store_device_masked_word({DeviceArguments}, {aligned} + 4ul, " +
                    $"{crosses} ? ({encoded} >> (32u - {shift})) : 0u, {crosses} ? ({bitMask} >> (32u - {shift})) : 0u);");
                _indent--;
                Line("}");
            }

            _indent--;
            Line("}");
        }

        // A buffer access whose V# lives in scalar registers: no host binding exists, so the
        // access runs over the device-address page table, bounded by the descriptor, with
        // the bound path's opcode set. Out-of-range reads are zero and out-of-range stores
        // are dropped. A formatted load or store decodes the format (and a load the
        // destination swizzle) of the descriptor's word 3 on the device; a typed one
        // converts with its own format. Atomics run on the backing memory's words.
        private bool TryEmitDeviceDescriptorBufferMemory(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            out string error)
        {
            try
            {
                return TryEmitDeviceDescriptorBufferAccess(instruction, control, out error);
            }
            finally
            {
                _deviceBufferFits = null;
            }
        }

        private bool TryEmitDeviceDescriptorBufferAccess(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            var isFormatLoad = IsFormatBufferLoad(opcode);
            var isFormatStore = IsFormatBufferStore(opcode);
            var isAtomic = opcode.StartsWith("BufferAtomic", StringComparison.Ordinal);
            var isStore = opcode.StartsWith("BufferStore", StringComparison.Ordinal) ||
                opcode.StartsWith("TBufferStore", StringComparison.Ordinal);
            var isSubwordStore = TryGetSubdwordStoreInfo(opcode, out var storeBytes, out var sourceShift);
            var isSubwordLoad = TryGetSubdwordLoadInfo(opcode, out var loadBytes, out var signExtend, out var subwordD16, out var d16High);
            var components = FormatComponentCount(opcode, control, out var d16);
            if (!isSubwordStore && !isSubwordLoad && !isFormatLoad && !isFormatStore && !isAtomic &&
                !opcode.StartsWith("BufferLoadDword", StringComparison.Ordinal) &&
                !opcode.StartsWith("BufferStoreDword", StringComparison.Ordinal))
            {
                error = $"unsupported device buffer descriptor opcode {opcode}";
                return false;
            }

            // The X2 forms are one 64-bit atomic on a dword pair, as on the bound path.
            var atomic = isAtomic ? opcode["BufferAtomic".Length..] : string.Empty;
            var atomicWords = atomic.EndsWith("X2", StringComparison.Ordinal) ? 2u : 1u;
            atomic = atomicWords == 2 ? atomic[..^2] : atomic;
            if (isAtomic && (atomicWords == 2 ? !IsQwordAtomic(atomic) : !TryGetAtomicExpression(atomic, "0u", "0u", out _)))
            {
                error = $"unsupported device buffer descriptor opcode {opcode}";
                return false;
            }

            if (instruction.Sources.Count < 2 || instruction.Sources[1].Kind != Gen5OperandKind.ScalarRegister)
            {
                error = "device buffer descriptor has no scalar resource base";
                return false;
            }

            if (!_hasDeviceAddressTable)
            {
                error = "device address access without an address range binding";
                return false;
            }

            var descriptor = DeviceDescriptorBounds(instruction.Sources[1].Value);
            var scalarOffset = instruction.Sources.Count > 2
                ? SourceExpression(instruction.Sources[2], instruction)
                : "0u";
            var vectorOffset = control.OffsetEnabled
                ? $"v[{control.VectorAddress + (control.IndexEnabled ? 1u : 0u)}]"
                : "0u";

            // SWIZZLE_ENABLE (word 1 bit 31), INDEX_STRIDE (word 3 bits 21-22, 8 << n indices)
            // and ADD_TID_ENABLE (word 3 bit 23) are read from the V# when the access runs
            // (RDNA2 ISA Table 37). As on the bound path, a V# swizzles only with a nonzero
            // stride.
            var resource = instruction.Sources[1].Value;
            var word1 = Temp("uint", ScalarExpression(resource + 1));
            var descriptorWord3 = Temp("uint", ScalarExpression(resource + 3));
            var swizzled = Temp("bool", $"({word1} >> 31u) != 0u && {descriptor.Stride} != 0u");
            var indexStride = Temp("uint", $"{swizzled} ? (8u << (({descriptorWord3} >> 21u) & 3u)) : 0u");
            var records = Temp("uint", ScalarExpression(resource + 2));
            var soffset = Temp("uint", scalarOffset);
            var recordIndex = Temp("uint", BufferIndex(control, $"((({descriptorWord3} >> 23u) & 1u) != 0u)"));
            var placedAddress = PlacedBufferAddress(
                soffset,
                recordIndex,
                $"(0x{unchecked((uint)control.OffsetBytes):X}u + {vectorOffset})",
                descriptor.Stride,
                indexStride,
                placed => placed);

            // Each placed byte offset remembers the unswizzled offset it came from (the VGPR
            // offset plus the instruction offset plus the distance into the access), which the
            // OOB_SELECT checks test rather than the placed address.
            var unswizzledOffsets = new Dictionary<string, string>(StringComparer.Ordinal);
            var byteOffset = new BufferAddress(
                placedAddress.Base,
                distance =>
                {
                    var placed = placedAddress.Place!(distance);
                    unswizzledOffsets[placed] = $"({placedAddress.Base} + {distance})";
                    return placed;
                });
            _deviceBufferFits = DeviceBufferOutOfRangeCheck(
                unswizzledOffsets,
                descriptorWord3,
                descriptor.Stride,
                records,
                soffset,
                recordIndex,
                swizzled,
                isFormatLoad || isFormatStore);

            if (control.Tfe)
            {
                // A TFE load's status (TryEmitBufferLoadWithStatus): nonzero when a dword the
                // access reaches inside the descriptor is on a page the table does not map; the
                // load returns zero for it. A formatted or sub-dword element is probed at its
                // first dword. The probe runs before the load writes any register.
                var probes = isFormatLoad || isSubwordLoad ? 1u : control.DwordCount;
                var status = Temp("uint", "0u");
                for (uint word = 0; word < probes; word++)
                {
                    var probeOffset = byteOffset.At(word * sizeof(uint));
                    var probeAddress = DeviceDescriptorWordAddress(descriptor, probeOffset);
                    var mapped = Temp("bool", "true");
                    Line($"if ({ExecCondition(DeviceBufferFits(descriptor, probeOffset, isSubwordLoad ? $"{loadBytes}u" : isFormatLoad ? "1u" : "4u"))})");
                    Line("{");
                    _indent++;
                    Line($"(void)sharpemu_resolve_device_address({DeviceArguments}, {probeAddress}, {mapped});");
                    _indent--;
                    Line("}");
                    Line($"{status} |= {mapped} ? 0u : 1u;");
                }

                _bufferTfeStatus = status;
            }

            if (isFormatLoad)
            {
                // Word 3 carries the unified format and dst_sel, the same layout the bound
                // path's specialized word takes; it is read before any destination is written.
                var memory = DeviceFormatMemory(descriptor, control.Glc);
                var word3 = Temp("uint", ScalarExpression(instruction.Sources[1].Value + 3));
                if (!control.Typed)
                {
                    EmitDecodedBufferFormatLoad(memory, byteOffset, word3, control.VectorData, components, d16);
                    return true;
                }

                if (TryEmitTypedBufferFormatLoad(memory, byteOffset, control, word3, components, d16))
                {
                    return true;
                }

                error = $"device buffer descriptor does not support the typed format {control.TypedFormat} of {opcode}";
                return false;
            }

            var mask = DeviceAddressPaging.AddressMask;
            if (isFormatStore && d16.Enabled && control.Typed &&
                !(Gfx10UnifiedFormat.TryDecode(control.TypedFormat, out var d16DataFormat, out _) &&
                  Gfx10UnifiedFormat.ComponentCount(d16DataFormat) != 0))
            {
                // A D16 store packs halves; no raw dword store stands in for its format.
                error = $"unsupported typed format {control.TypedFormat} for {opcode}";
                return false;
            }

            // A typed store whose format does not decode is a raw dword store, as on the bound path.
            if (isFormatStore &&
                (!control.Typed ||
                 (Gfx10UnifiedFormat.TryDecode(control.TypedFormat, out var typedDataFormat, out _) &&
                  Gfx10UnifiedFormat.ComponentCount(typedDataFormat) != 0)))
            {
                var word3 = Temp("uint", ScalarExpression(instruction.Sources[1].Value + 3));
                EmitDeviceFormatStore(
                    descriptor,
                    byteOffset,
                    word3,
                    control.Typed ? $"{control.TypedFormat}u" : $"(({word3} >> 12) & 0x7Fu)",
                    control.VectorData,
                    components,
                    d16);
                return true;
            }

            if (isAtomic)
            {
                // Each dword runs as a device atomic on the word the page table maps it to.
                // An access past the descriptor, unaligned or unmapped is not performed and
                // returns zero, the bound path's out-of-range result. Each word is placed,
                // checked and aligned at its own offset.
                var wordAddresses = new string[atomicWords];
                var wordsInRange = new string[atomicWords];
                for (uint word = 0; word < atomicWords; word++)
                {
                    var wordOffset = byteOffset.At(word * sizeof(uint));
                    wordAddresses[word] = Temp("ulong", $"({descriptor.Base} + (ulong){wordOffset}) & {FormatULong(mask)}");
                    wordsInRange[word] =
                        $"{DeviceBufferFits(descriptor, wordOffset, "4u")} && ({wordAddresses[word]} & 3ul) == 0ul";
                }

                var inRange = Temp("bool", string.Join(" && ", wordsInRange));
                // An X2 atomic's operands are read before any destination register is written.
                string? qwordSource = null;
                string? qwordComparator = null;
                if (atomicWords == 2)
                {
                    qwordSource = QwordSource(control.VectorData);
                    qwordComparator = atomic is "Cmpswap" or "Fcmpswap" ? QwordSource(control.VectorData + 2) : "0ul";
                }

                var originals = new string[atomicWords];
                for (uint word = 0; word < atomicWords; word++)
                {
                    originals[word] = Temp("uint", "0u");
                }

                Line($"if ({ExecCondition(inRange)})");
                Line("{");
                _indent++;
                var valids = new string[atomicWords];
                var pointers = new string[atomicWords];
                for (uint word = 0; word < atomicWords; word++)
                {
                    valids[word] = Temp("bool", "false");
                    pointers[word] = Temp("device uint*", $"sharpemu_resolve_device_address({DeviceArguments}, {wordAddresses[word]}, {valids[word]})");
                }

                void Operate(uint word)
                {
                    // MUBUF CMPSWAP reads the new value from VDATA and the comparator from
                    // the register after it (RDNA2 sections 8.1/12.14).
                    TryGetAtomicExpression(
                        atomic,
                        $"v[{control.VectorData + word}]",
                        $"v[{control.VectorData + 1}]",
                        out var build);
                    Line($"if ({valids[word]})");
                    Line("{");
                    _indent++;
                    Line($"{originals[word]} = {build(pointers[word])};");
                    _indent--;
                    Line("}");
                }

                if (atomicWords == 2)
                {
                    // One 64-bit atomic (EmitQwordAtomic). The access is in range; a qword
                    // the page table does not map whole is not performed and returns zero,
                    // as an unmapped dword does.
                    Line($"if ({valids[0]} && {valids[1]})");
                    Line("{");
                    _indent++;
                    EmitQwordAtomic(
                        atomic,
                        pointers[0],
                        pointers[1],
                        qwordSource!,
                        qwordComparator!,
                        originals[0],
                        originals[1]);
                    _indent--;
                    Line("}");
                }
                else
                {
                    Operate(0);
                }

                _indent--;
                Line("}");
                if (control.Glc)
                {
                    // GLC returns the pre-operation words in VDATA.
                    for (uint word = 0; word < atomicWords; word++)
                    {
                        StoreVector(control.VectorData + word, originals[word]);
                    }
                }

                return true;
            }

            if (isSubwordLoad)
            {
                var subwordOffset = byteOffset.At(0);
                var inRange = Temp("bool", DeviceBufferFits(descriptor, subwordOffset, $"{loadBytes}u"));
                var address = Temp("ulong", $"({descriptor.Base} + (ulong){subwordOffset}) & {FormatULong(mask)}");
                _usesCoherentLoads |= control.Glc;
                StoreSubdwordLoad(
                    control.VectorData,
                    $"{inRange} ? (sharpemu_load_device_unaligned{(control.Glc ? "_coherent" : string.Empty)}({DeviceArguments}, {address}, {loadBytes}u) & {FormatUInt(loadBytes == 1 ? 0xFFu : 0xFFFFu)}) : 0u",
                    loadBytes,
                    signExtend,
                    subwordD16,
                    d16High);
                return true;
            }

            if (!isStore)
            {
                if (control.Glc && control.DwordCount == 2)
                {
                    // A GLC LOAD_DWORDX2 is one snapshot of the pair (see EmitPairLocked).
                    var firstOffset = byteOffset.At(0);
                    var secondOffset = byteOffset.At(sizeof(uint));
                    var first = DeviceDescriptorWordAddress(descriptor, firstOffset);
                    var second = DeviceDescriptorWordAddress(descriptor, secondOffset);
                    StorePairLoad(
                        control.VectorData,
                        $"sharpemu_load_device_pair_coherent({DeviceArguments}, {first}, {DeviceDescriptorWordInRange(descriptor, firstOffset)}, " +
                        $"{second}, {DeviceDescriptorWordInRange(descriptor, secondOffset)}, {PairLockArguments})");
                    return true;
                }

                for (uint index = 0; index < control.DwordCount; index++)
                {
                    StoreVector(control.VectorData + index, DeviceDescriptorWord(descriptor, byteOffset.At(index * sizeof(uint)), control.Glc));
                }

                return true;
            }

            if (isSubwordStore)
            {
                var subwordOffset = byteOffset.At(0);
                var inRange = Temp("bool", DeviceBufferFits(descriptor, subwordOffset, $"{storeBytes}u"));
                var address = Temp("ulong", $"({descriptor.Base} + (ulong){subwordOffset}) & {FormatULong(mask)}");
                var source = sourceShift == 0
                    ? $"v[{control.VectorData}]"
                    : $"(v[{control.VectorData}] >> {sourceShift})";
                Line($"if ({ExecCondition(inRange)})");
                Line("{");
                _indent++;
                Line($"sharpemu_store_device_bytes({DeviceArguments}, {address}, {source}, {storeBytes}u);");
                _indent--;
                Line("}");
                return true;
            }

            for (uint index = 0; index < control.DwordCount; index++)
            {
                var componentOffset = byteOffset.At(index * sizeof(uint));
                var inRange = Temp("bool", DeviceBufferFits(descriptor, componentOffset, "4u"));
                var address = Temp("ulong", $"({descriptor.Base} + (ulong){componentOffset}) & {FormatULong(mask & ~3ul)}");
                Line($"if ({ExecCondition(inRange)})");
                Line("{");
                _indent++;
                Line($"sharpemu_store_device_dword({DeviceArguments}, {address}, v[{control.VectorData + index}]);");
                _indent--;
                Line("}");
            }

            return true;
        }

        // A runtime V# whose descriptors cannot be reconstructed from its raw words: its base
        // dword probes the flattened candidate mapping and each arm runs the ordinary buffer
        // operation against one native candidate binding.
        private bool TryEmitBoundedCandidateTableMemory(
            Gen5ShaderInstruction instruction,
            Gen5BufferMemoryControl control,
            int memoryIndex,
            out string error)
        {
            error = string.Empty;
            if (!_request.BufferCandidateTableByMemoryIndex.TryGetValue(memoryIndex, out var table))
            {
                // No finite, safe candidate set exists; a formatted load reads zero rather than
                // reinterpreting an arbitrary device address with a guessed format.
                if (instruction.Opcode.StartsWith("BufferLoadFormat", StringComparison.Ordinal))
                {
                    // A D16 load writes zero halves and keeps the halves it does not write.
                    var components = FormatComponentCount(instruction.Opcode, control, out var d16);
                    StoreFormatLoadComponents(control.VectorData, Enumerable.Repeat("0u", (int)components).ToArray(), d16, "4u");
                    return true;
                }

                // Any other access reads the V# its scalar registers hold when it runs and goes
                // through the device-address page table, which the tracker bound for it.
                return TryEmitDeviceDescriptorBufferMemory(instruction, control, out error);
            }

            if (table.CandidateCount == 0)
            {
                error = "runtime buffer descriptor candidate table is empty";
                return false;
            }

            if (table.CandidateCount == 1)
            {
                return EmitResolvedBufferMemory(instruction, control, (int)table.FirstCandidate, out error);
            }

            if (!_hasFlattenedTable)
            {
                error = "runtime buffer descriptor candidate table without a flattened table binding";
                return false;
            }

            if (instruction.Sources.Count < 2 || instruction.Sources[1].Kind != Gen5OperandKind.ScalarRegister)
            {
                error = "runtime buffer descriptor has no scalar resource base";
                return false;
            }

            var selector = SelectBufferCandidate(table, Temp("uint", ScalarExpression(instruction.Sources[1].Value)));
            for (uint index = 0; index < table.CandidateCount; index++)
            {
                Line($"if ({selector} == {index}u)");
                Line("{");
                _indent++;
                var emitted = EmitResolvedBufferMemory(instruction, control, (int)(table.FirstCandidate + index), out error);
                _indent--;
                Line("}");
                if (!emitted)
                {
                    return false;
                }
            }

            return true;
        }

        // Searches the sorted base-address mapping of a candidate table for the probe key; the
        // result is the candidate-local index, candidate 0 when absent.
        private string SelectBufferCandidate(BufferCandidateTableUse table, string key)
        {
            var mapping = table.MappingOffset;
            var count = Temp("uint", LoadFlattenedWord($"{mapping}u"));
            var low = Temp("uint", "0u");
            var high = Temp("uint", count);
            for (uint iteration = 0; iteration < table.SearchIterations; iteration++)
            {
                var middle = Temp("uint", $"{low} + (({high} - {low}) >> 1u)");
                var probeSlot = Temp("uint", $"{mapping + 1}u + ({middle} << 1u)");
                var probeKey = Temp("uint", LoadFlattenedWord(probeSlot));
                var moveUp = Temp("bool", $"{probeKey} < {key} && {low} < {high}");
                Line($"{low} = {moveUp} ? {middle} + 1u : {low};");
                Line($"{high} = {moveUp} ? {high} : {middle};");
            }

            var foundSlot = Temp("uint", $"{mapping + 1}u + ({low} << 1u)");
            var foundKey = Temp("uint", LoadFlattenedWord(foundSlot));
            var found = Temp("bool", $"{low} < {count} && {foundKey} == {key}");
            var mapped = Temp("uint", LoadFlattenedWord($"({foundSlot} + 1u)"));
            return Temp("uint", $"{found} && {mapped} < {table.CandidateCount}u ? {mapped} : 0u");
        }

        private bool TryEmitLayoutScalarMemory(Gen5ShaderInstruction instruction, Gen5ScalarMemoryControl control, out string error)
        {
            error = string.Empty;
            var request = _request;
            // SOFFSET is read once, before any destination is written: S_BUFFER_LOAD_DWORDX2
            // and wider routinely name a destination that overlaps the offset register
            // (`s[106:107] <- s[8:11], s106`), and the hardware reads every source operand
            // before it writes the first dword. Re-reading the register file per component
            // would compute the second dword's address from the first dword's value, which
            // is what the SPIR-V translator avoids by holding the loaded SSA value.
            var dynamicOffset = control.DynamicOffsetRegister is { } register
                ? Temp("uint", ScalarExpression(register))
                : "0u";
            (string Base, string Size, string Stride)? deviceDescriptor = null;
            string? deviceAddress = null;
            string? deviceSpan = null;
            string? deviceSpanValid = null;
            string? scalarRun = null;

            // The run of up to four dwords, starting at a multiple of four, that the dword at
            // `component` loads with: every dword of it a plain read of the same bound buffer.
            bool TryScalarBufferRun(int component, out (int First, int Count) run)
            {
                run = default;
                if (!_options.VectorDwordLoads || !_options.RawDwordAccess || control.Glc)
                {
                    return false;
                }

                var first = component & ~3;
                var count = Math.Min(4, instruction.Destinations.Count - first);
                if (count < 2)
                {
                    return false;
                }

                uint? binding = null;
                for (var index = first; index < first + count; index++)
                {
                    if (!request.Memory.TryGetIndex(instruction.Pc, (uint)index, out var memory))
                    {
                        return false;
                    }

                    var record = request.Memory[memory];
                    if (record.PlanningOnly || record.Kind != MemoryResourceKind.ScalarBuffer || record.DeviceDescriptor ||
                        record.Resource == MemoryAccessInfo.NoResource || (binding is { } known && known != record.Resource))
                    {
                        return false;
                    }

                    binding = record.Resource;
                }

                run = (first, count);
                return true;
            }

            for (var component = 0; component < instruction.Destinations.Count; component++)
            {
                var destination = instruction.Destinations[component];
                if (destination.Kind != Gen5OperandKind.ScalarRegister)
                {
                    error = "invalid scalar-memory destination";
                    return false;
                }

                if (!request.Memory.TryGetIndex(instruction.Pc, (uint)component, out var memoryIndex))
                {
                    error = $"scalar load component {component} has no memory record";
                    return false;
                }

                var entry = request.Memory[memoryIndex];
                if (request.IndirectOffsetKeyMemoryIndices.Contains(memoryIndex))
                    Line($"{_indirectKeyScratch[memoryIndex]} = {dynamicOffset};");
                string value;
                if (entry.PlanningOnly)
                {
                    // A planning-only read without a slot is a heap descriptor the host resolved; nothing is emitted.
                    if (!request.FlattenedSlotByMemoryIndex.TryGetValue(memoryIndex, out var slot))
                    {
                        continue;
                    }

                    if (!_hasFlattenedTable)
                    {
                        error = "flattened read without a flattened table binding";
                        return false;
                    }

                    value = Temp("uint", LoadFlattenedWord($"{slot}u"));
                }
                else if (entry.Kind == MemoryResourceKind.ScalarBuffer &&
                         (entry.DeviceDescriptor || entry.Resource == MemoryAccessInfo.NoResource))
                {
                    // The buffer this load addresses has no host binding: either its V#
                    // was itself read on the device, or it is a runtime table entry the
                    // draw cannot evaluate ahead of time. Its scalar registers hold the
                    // descriptor either way, so the load reads it the way the hardware
                    // does and goes through the device-address page table.
                    if (instruction.Sources.Count == 0 || instruction.Sources[0].Kind != Gen5OperandKind.ScalarRegister)
                    {
                        error = "scalar buffer load has no scalar descriptor";
                        return false;
                    }

                    if (!_hasDeviceAddressTable)
                    {
                        error = "device address access without an address range binding";
                        return false;
                    }

                    var descriptorOffset = Temp(
                        "uint",
                        $"({dynamicOffset} + {FormatUInt(unchecked((uint)control.ImmediateOffsetBytes + ((uint)component * sizeof(uint))))})");
                    // Snapshot the V# before the first destination store: SDATA may overlap SBASE.
                    deviceDescriptor ??= DeviceDescriptorBounds(instruction.Sources[0].Value);
                    value = DeviceDescriptorWord(deviceDescriptor.Value, descriptorOffset, control.Glc);
                }
                else if (entry.Kind == MemoryResourceKind.ScalarBuffer &&
                         TryScalarBufferRun(component, out var run))
                {
                    // A run of the load's dwords from one bound buffer, behind one bounds check
                    // (Gen5MslCodegenOptions.VectorDwordLoads); every dword is read before the
                    // first destination store, as the hardware does.
                    if (run.First == component)
                    {
                        var bindingIndex = (int)entry.Resource;
                        var byteOffset = unchecked((uint)control.ImmediateOffsetBytes + ((uint)component * sizeof(uint)));
                        var address = Temp("uint", ApplyByteBias(bindingIndex, $"({dynamicOffset} + {FormatUInt(byteOffset)})"));
                        scalarRun = Temp("uint4", LoadDwords(bindingIndex, address, (uint)run.Count));
                    }

                    value = Temp("uint", $"{scalarRun}[{component - run.First}]");
                }
                else if (entry.Kind == MemoryResourceKind.ScalarBuffer)
                {
                    var bindingIndex = (int)entry.Resource;
                    var byteOffset = unchecked((uint)control.ImmediateOffsetBytes + ((uint)component * sizeof(uint)));
                    var address = Temp("uint", ApplyByteBias(bindingIndex, $"({dynamicOffset} + {FormatUInt(byteOffset)})"));
                    // A GLC scalar load misses the scalar cache (gfx10 SMEM GLC) and reads
                    // the device-coherent value, as a GLC vector load does.
                    _usesCoherentLoads |= control.Glc;
                    value = Temp(
                        "uint",
                        control.Glc
                            ? $"sharpemu_load_dword_coherent(b{bindingIndex}, {BufferBytes(bindingIndex)}, {address})"
                            : LoadDword(bindingIndex, address));
                }
                else
                {
                    if (instruction.Sources.Count == 0 || instruction.Sources[0].Kind != Gen5OperandKind.ScalarRegister)
                    {
                        error = "invalid scalar-memory address";
                        return false;
                    }

                    if (!_hasDeviceAddressTable)
                    {
                        error = "device address access without an address range binding";
                        return false;
                    }

                    if (deviceAddress is null)
                    {
                        var baseRegister = instruction.Sources[0].Value;
                        deviceAddress = Temp(
                            "ulong",
                            $"({Scalar64Expression(baseRegister)} + (ulong){dynamicOffset} + {SignedOffset64(control.ImmediateOffsetBytes)}) & {FormatULong(DeviceAddressPaging.AddressMask & ~3ul)}");
                    }

                    var componentAddress = component == 0 ? deviceAddress : $"({deviceAddress} + {component * sizeof(uint)}ul)";
                    _usesCoherentLoads |= control.Glc;
                    var componentLoad = $"sharpemu_load_device_dword{(control.Glc ? "_coherent" : string.Empty)}({DeviceArguments}, {componentAddress})";
                    if (instruction.Destinations.Count > 1)
                    {
                        // Every dword of an S_LOAD_DWORDXn is one page-table walk if
                        // resolved alone; the walk is a 13-step dependent binary search, so
                        // the span is resolved once and its dwords index off that pointer.
                        if (deviceSpan is null)
                        {
                            deviceSpanValid = Temp("bool", "false");
                            deviceSpan = Temp(
                                "device uint*",
                                $"sharpemu_resolve_device_span({DeviceArguments}, {deviceAddress}, {instruction.Destinations.Count * sizeof(uint)}u, {deviceSpanValid})");
                        }

                        var spanWord = control.Glc
                            ? $"atomic_load_explicit((device atomic_uint*)({deviceSpan} + {component}), memory_order_relaxed)"
                            : $"{deviceSpan}[{component}]";
                        value = Temp("uint", $"{deviceSpanValid} ? {spanWord} : {componentLoad}");
                    }
                    else
                    {
                        value = Temp("uint", componentLoad);
                    }
                }

                if (!request.IndirectOffsetKeyMemoryIndices.Contains(memoryIndex) &&
                    _indirectKeyScratch.TryGetValue(memoryIndex, out var keyScratch))
                {
                    Line($"{keyScratch} = {value};");
                }

                StoreScalar(destination.Value, value);
            }

            return true;
        }

        // ---- global memory ----

        private bool TryEmitLayoutGlobalMemory(Gen5ShaderInstruction instruction, Gen5GlobalMemoryControl control, out string error)
        {
            error = string.Empty;
            if (instruction.Opcode.StartsWith("Scratch", StringComparison.Ordinal))
            {
                return TryEmitScratchMemory(instruction, control, out error);
            }

            var request = _request;
            if (!request.Memory.TryGetIndex(instruction.Pc, 0, out var memoryIndex))
            {
                error = "global access has no memory record";
                return false;
            }

            if (!_hasDeviceAddressTable)
            {
                error = "device address access without an address range binding";
                return false;
            }

            var memoryOpcode = control.UsesFlatAddress ? "Global" + instruction.Opcode["Flat".Length..] : instruction.Opcode;
            string address;
            if (control.AddThreadId)
            {
                // GLOBAL_LOAD/STORE_DWORD_ADDTID: SADDR + offset + the lane's index * 4, with
                // no VADDR (RDNA2 ISA 9.1, 12.18). The lane index needs real wave lanes, and
                // SADDR must name a register pair.
                if (!HasRealWaveLanes || control.ScalarAddress >= 125)
                {
                    error = $"{instruction.Opcode} needs {(HasRealWaveLanes ? "a SADDR register pair" : "wave lanes")}";
                    return false;
                }

                address = $"{Scalar64Expression(control.ScalarAddress)} + (ulong)(sharpemu_lane * 4u)";
            }
            else
            {
                address = control.UsesFlatAddress || control.ScalarAddress >= 125
                    ? $"(ulong)v[{control.VectorAddress}] | ((ulong)v[{control.VectorAddress + 1}] << 32)"
                    : $"{Scalar64Expression(control.ScalarAddress)} + (ulong)v[{control.VectorAddress}]";
            }

            address = Temp("ulong", $"({address}) + {SignedOffset64(control.OffsetBytes)}");
            var entry = request.Memory[memoryIndex];
            if (control.UsesFlatAddress && entry.AddressSpace != FlatAddressSpace.Global)
            {
                return TryEmitFlatLocalMemory(
                    instruction, control, memoryOpcode, entry.AddressSpace, address, out error);
            }

            var writes = entry.Access is MemoryAccess.Write or MemoryAccess.Atomic;
            // An ADDTID lane reaches its own dword; the entry plans the wave's 64.
            var accessBytes = control.AddThreadId
                ? sizeof(uint)
                : Math.Max((entry.DataBits + 7) / 8, 1u) * Math.Max(entry.DataDwords, 1u);
            var allowed = writes ? IsWrittenAccessAllowed(memoryIndex, address, accessBytes) : "true";

            if (memoryOpcode.StartsWith("GlobalAtomic", StringComparison.Ordinal) &&
                memoryOpcode.EndsWith("X2", StringComparison.Ordinal))
            {
                return TryEmitGlobalAtomic64(instruction, control, memoryOpcode["GlobalAtomic".Length..^2], address, allowed, out error);
            }

            if (memoryOpcode.StartsWith("GlobalAtomic", StringComparison.Ordinal))
            {
                if (!TryGetAtomicExpression(
                        memoryOpcode["GlobalAtomic".Length..],
                        $"v[{control.SourceVectorRegister}]",
                        $"v[{control.SourceVectorRegister + 1}]",
                        out var build))
                {
                    error = $"unsupported global opcode {instruction.Opcode}";
                    return false;
                }

                Line($"if ({ExecCondition(allowed)})");
                Line("{");
                _indent++;
                var valid = Temp("bool", "false");
                var pointer = Temp("device uint*", $"sharpemu_resolve_device_address({DeviceArguments}, {address}, {valid})");
                Line($"if ({valid})");
                Line("{");
                _indent++;
                var original = Temp("uint", build(pointer));
                if (control.Glc)
                {
                    Line($"v[{control.DestinationVectorRegister}] = {original};");
                }

                _indent--;
                Line("}");
                _indent--;
                Line("}");
                return true;
            }

            if (memoryOpcode.StartsWith("GlobalStore", StringComparison.Ordinal))
            {
                Line($"if ({ExecCondition(allowed)})");
                Line("{");
                _indent++;
                if (TryGetSubdwordStoreInfo(memoryOpcode, out var byteCount, out var sourceShift))
                {
                    var source = sourceShift == 0
                        ? $"v[{control.SourceVectorRegister}]"
                        : $"(v[{control.SourceVectorRegister}] >> {sourceShift})";
                    Line($"sharpemu_store_device_bytes({DeviceArguments}, {address}, {source}, {byteCount}u);");
                }
                else if (_options.DeviceAddressSpans && control.DwordCount > 1)
                {
                    // One page-table lookup for the whole record. A record inside one page is
                    // contiguous host memory, so the dwords index off the one pointer; one that
                    // crosses a page or is unmapped takes the per-dword stores.
                    var spanValid = Temp("bool", "false");
                    var span = Temp("device uint*", $"sharpemu_resolve_device_span({DeviceArguments}, {address}, {control.DwordCount * sizeof(uint)}u, {spanValid})");
                    Line($"if ({spanValid})");
                    Line("{");
                    _indent++;
                    for (uint index = 0; index < control.DwordCount; index++)
                    {
                        Line($"{span}[{index}] = v[{control.SourceVectorRegister + index}];");
                    }

                    _indent--;
                    Line("}");
                    Line("else");
                    Line("{");
                    _indent++;
                    for (uint index = 0; index < control.DwordCount; index++)
                    {
                        Line($"sharpemu_store_device_dword({DeviceArguments}, {address} + {index * sizeof(uint)}ul, v[{control.SourceVectorRegister + index}]);");
                    }

                    _indent--;
                    Line("}");
                }
                else
                {
                    for (uint index = 0; index < control.DwordCount; index++)
                    {
                        Line($"sharpemu_store_device_dword({DeviceArguments}, {address} + {index * sizeof(uint)}ul, v[{control.SourceVectorRegister + index}]);");
                    }
                }

                _indent--;
                Line("}");
                return true;
            }

            // GLC reads the device-coherent value (coherent_load_prelude), and a GLC
            // LOAD_DWORDX2 reads its pair as one snapshot (see EmitPairLocked).
            _usesCoherentLoads |= control.Glc;
            var coherent = control.Glc ? "_coherent" : string.Empty;
            if (TryGetSubdwordLoadInfo(memoryOpcode, out var loadBytes, out var signExtend, out var d16, out var d16High))
            {
                StoreSubdwordLoad(
                    control.DestinationVectorRegister,
                    $"sharpemu_load_device_unaligned{coherent}({DeviceArguments}, {address}, {loadBytes}u) & {FormatUInt(loadBytes == 1 ? 0xFFu : 0xFFFFu)}",
                    loadBytes,
                    signExtend,
                    d16,
                    d16High);
                return true;
            }

            if (!memoryOpcode.StartsWith("GlobalLoad", StringComparison.Ordinal))
            {
                error = $"unsupported global opcode {instruction.Opcode}";
                return false;
            }

            var aligned = Temp("ulong", $"{address} & ~3ul");
            if (control.Glc && control.DwordCount == 2)
            {
                StorePairLoad(
                    control.DestinationVectorRegister,
                    $"sharpemu_load_device_pair_coherent({DeviceArguments}, {aligned}, true, {aligned} + 4ul, true, {PairLockArguments})");
                return true;
            }

            if (_options.DeviceAddressSpans && control.DwordCount > 1 && control.DestinationVectorRegister + control.DwordCount <= VectorRegisterFileCount)
            {
                // One page-table lookup for the whole record (GLOBAL_LOAD_DWORDX2-X4); see the
                // store above. Every destination is written after the lookup, under EXEC.
                var guarded = !_execKnownFull;
                if (guarded)
                {
                    Line("if (exec)");
                    Line("{");
                    _indent++;
                }

                var spanValid = Temp("bool", "false");
                var span = Temp("device uint*", $"sharpemu_resolve_device_span({DeviceArguments}, {aligned}, {control.DwordCount * sizeof(uint)}u, {spanValid})");
                for (uint index = 0; index < control.DwordCount; index++)
                {
                    var componentAddress = index == 0 ? aligned : $"{aligned} + {index * sizeof(uint)}ul";
                    var spanWord = control.Glc
                        ? $"atomic_load_explicit((device atomic_uint*)({span} + {index}), memory_order_relaxed)"
                        : $"{span}[{index}]";
                    StoreVector(
                        control.DestinationVectorRegister + index,
                        $"{spanValid} ? {spanWord} : sharpemu_load_device_dword{coherent}({DeviceArguments}, {componentAddress})",
                        guardWithExec: false);
                }

                if (guarded)
                {
                    _indent--;
                    Line("}");
                }

                return true;
            }

            for (uint index = 0; index < control.DwordCount; index++)
            {
                var componentAddress = index == 0 ? aligned : $"{aligned} + {index * sizeof(uint)}ul";
                StoreVector(
                    control.DestinationVectorRegister + index,
                    $"sharpemu_load_device_dword{coherent}({DeviceArguments}, {componentAddress})");
            }

            return true;
        }

        // GLOBAL_ATOMIC_*_X2: one 64-bit atomic on the qword at the address (EmitQwordAtomic),
        // with GLC returning the pre-operation qword in VDST[0:1]. The hardware requires a
        // naturally aligned qword; one that is not, or that the page table does not map in
        // one piece, is not performed, like the bound path's unaligned atomics, and a GLC
        // form then returns zero.
        private bool TryEmitGlobalAtomic64(
            Gen5ShaderInstruction instruction,
            Gen5GlobalMemoryControl control,
            string atomic,
            string address,
            string allowed,
            out string error)
        {
            error = string.Empty;
            if (!IsQwordAtomic(atomic))
            {
                error = $"unsupported global opcode {instruction.Opcode}";
                return false;
            }

            // Every source register is read before a destination register is written.
            var source = QwordSource(control.SourceVectorRegister);
            var comparator = atomic is "Cmpswap" or "Fcmpswap" ? QwordSource(control.SourceVectorRegister + 2) : "0ul";
            var originalLow = Temp("uint", "0u");
            var originalHigh = Temp("uint", "0u");
            Line($"if ({ExecCondition(allowed)})");
            Line("{");
            _indent++;
            var valid = Temp("bool", "false");
            var pointer = Temp("device uint*", $"sharpemu_resolve_device_span({DeviceArguments}, {address}, 8u, {valid})");
            Line($"if ({valid} && ({address} & 7ul) == 0ul)");
            Line("{");
            _indent++;
            EmitQwordAtomic(atomic, pointer, $"{pointer} + 1", source, comparator, originalLow, originalHigh);
            _indent--;
            Line("}");
            _indent--;
            Line("}");
            if (control.Glc)
            {
                StoreVector(control.DestinationVectorRegister, originalLow);
                StoreVector(control.DestinationVectorRegister + 1, originalHigh);
            }

            return true;
        }

        // A zero-extended byte or short read from device memory into its destination: sign
        // extended for the Sbyte/Sshort forms, merged into one half of the register for D16.
        private void StoreSubdwordLoad(uint destination, string zeroExtended, uint loadBytes, bool signExtend, bool d16, bool d16High)
        {
            var loaded = Temp("uint", zeroExtended);
            if (signExtend)
            {
                loaded = Temp("uint", $"(uint)extract_bits(as_type<int>({loaded}), 0u, {loadBytes * 8}u)");
            }

            if (!d16)
            {
                StoreVector(destination, loaded);
                return;
            }

            StoreVector(
                destination,
                d16High
                    ? $"(v[{destination}] & 0x0000FFFFu) | (({loaded} & 0xFFFFu) << 16)"
                    : $"(v[{destination}] & 0xFFFF0000u) | ({loaded} & 0xFFFFu)");
        }

        // Routes each lane of an aperture-derived FLAT access like the hardware: LDS
        // or scratch at the low dword when the high dword is in an aperture,
        // otherwise a global access through the page table.
        private bool TryEmitFlatLocalMemory(
            Gen5ShaderInstruction instruction,
            Gen5GlobalMemoryControl control,
            string memoryOpcode,
            FlatAddressSpace addressSpace,
            string address,
            out string error)
        {
            error = string.Empty;
            var store = memoryOpcode.StartsWith("GlobalStoreDword", StringComparison.Ordinal);
            var load = memoryOpcode.StartsWith("GlobalLoadDword", StringComparison.Ordinal);
            if (!store && !load)
            {
                error = $"{instruction.Opcode} through the {addressSpace} flat aperture is not implemented";
                return false;
            }

            var canShare = addressSpace is FlatAddressSpace.Shared or FlatAddressSpace.SharedOrPrivate;
            var canPrivate = addressSpace is FlatAddressSpace.Private or FlatAddressSpace.SharedOrPrivate;
            if (canShare && !_usesLds)
            {
                error = "a shared flat aperture access has no LDS storage";
                return false;
            }

            if (canPrivate && _scratchDwordCount == 0)
            {
                error = "a private flat aperture access has no scratch storage";
                return false;
            }

            BeginExecGuard();
            var offset = Temp("uint", $"(uint){address}");
            var aperture = Temp(
                "uint",
                $"(uint)({address} >> {32 + Gen5InlineConstants.ApertureShift})");

            void Emit(Func<uint, string> word)
            {
                for (uint index = 0; index < control.DwordCount; index++)
                {
                    if (store)
                    {
                        Line($"{word(index)} = v[{control.SourceVectorRegister + index}];");
                    }
                    else
                    {
                        StoreVector(
                            control.DestinationVectorRegister + index,
                            word(index),
                            guardWithExec: false);
                    }
                }
            }

            void Shared() => Emit(index =>
                $"sharpemu_lds[{LdsIndex(offset, index * sizeof(uint))}]");

            void Private() => Emit(index => ScratchWord(offset, index * sizeof(uint)));

            void Global()
            {
                var aligned = Temp("ulong", $"{address} & ~3ul");
                if (!store && control.Glc && control.DwordCount == 2)
                {
                    // A GLC LOAD_DWORDX2 is one snapshot of the pair (see EmitPairLocked).
                    StorePairLoad(
                        control.DestinationVectorRegister,
                        $"sharpemu_load_device_pair_coherent({DeviceArguments}, {aligned}, true, {aligned} + 4ul, true, {PairLockArguments})",
                        guardWithExec: false);
                    return;
                }

                // A GLC load reads the device-coherent value (coherent_load_prelude).
                _usesCoherentLoads |= !store && control.Glc;
                for (uint index = 0; index < control.DwordCount; index++)
                {
                    var dwordAddress = index == 0 ? aligned : $"{aligned} + {index * sizeof(uint)}ul";
                    if (store)
                    {
                        Line($"sharpemu_store_device_dword({DeviceArguments}, {dwordAddress}, v[{control.SourceVectorRegister + index}]);");
                    }
                    else
                    {
                        StoreVector(
                            control.DestinationVectorRegister + index,
                            $"sharpemu_load_device_dword{(control.Glc ? "_coherent" : string.Empty)}({DeviceArguments}, {dwordAddress})",
                            guardWithExec: false);
                    }
                }
            }

            void Branch(string condition, Action then, Action otherwise)
            {
                Line($"if ({condition})");
                Line("{");
                _indent++;
                then();
                _indent--;
                Line("}");
                Line("else");
                Line("{");
                _indent++;
                otherwise();
                _indent--;
                Line("}");
            }

            // Like the SPIR-V translator: a lane is in an aperture when its high dword carries the
            // guest literal tag (top nibble) or the virtual segment src_shared/private_base reads as.
            var high = Temp("uint", $"(uint)({address} >> 32)");
            var isShared =
                $"({aperture} == {FormatUInt(Gen5InlineConstants.SharedApertureHigh >> Gen5InlineConstants.ApertureShift)} || " +
                $"{high} == {FormatUInt((uint)(Gen5InlineConstants.SharedApertureBase >> 32))})";
            var isPrivate =
                $"({aperture} == {FormatUInt(Gen5InlineConstants.PrivateApertureHigh >> Gen5InlineConstants.ApertureShift)} || " +
                $"{high} == {FormatUInt((uint)(Gen5InlineConstants.PrivateApertureBase >> 32))})";
            if (canShare && canPrivate)
            {
                Branch(isShared, Shared, () => Branch(isPrivate, Private, Global));
            }
            else if (canShare)
            {
                Branch(isShared, Shared, Global);
            }
            else
            {
                Branch(isPrivate, Private, Global);
            }

            EndExecGuard();
            return true;
        }

        // ---- scratch ----

        // Guest scratch is per-invocation private storage, named by its own FLAT
        // segment rather than by a resource, so it never reaches the device
        // address table: it lives in a thread-local array like the SPIR-V
        // translator's Private one.
        private bool TryEmitScratchMemory(Gen5ShaderInstruction instruction, Gen5GlobalMemoryControl control, out string error)
        {
            error = string.Empty;
            if (_scratchDwordCount == 0)
            {
                error = "scratch storage was not declared";
                return false;
            }

            var address = control.ScalarAddress < 125
                ? ScalarExpression(control.ScalarAddress)
                : $"v[{control.VectorAddress}]";
            if (control.DynamicOffsetRegister is { } dynamicOffsetRegister)
            {
                address = $"{address} + {ScalarExpression(dynamicOffsetRegister)}";
            }

            if (control.OffsetBytes != 0)
            {
                address = $"{address} + {FormatUInt(unchecked((uint)control.OffsetBytes))}";
            }

            var byteAddress = Temp("uint", address);
            if (instruction.Opcode.StartsWith("ScratchStore", StringComparison.Ordinal))
            {
                BeginExecGuard();
                if (TryGetSubdwordStoreInfo(instruction.Opcode, out var byteCount, out var sourceShift))
                {
                    // A sub-dword store merges into the dword it lands in.
                    var source = sourceShift == 0
                        ? $"v[{control.SourceVectorRegister}]"
                        : $"(v[{control.SourceVectorRegister}] >> {sourceShift})";
                    var element = FormatUInt(byteCount == 1 ? 0xFFu : 0xFFFFu);
                    var shift = Temp("uint", $"({byteAddress} & 3u) << 3u");
                    var word = ScratchAccessWord(control, byteAddress);
                    Line($"{word} = ({word} & ~({element} << {shift})) | (((({source}) & {element})) << {shift});");
                }
                else
                {
                    for (uint index = 0; index < control.DwordCount; index++)
                    {
                        var value = control.SourceIsScalar
                            ? ScalarExpression(control.SourceVectorRegister + index)
                            : $"v[{control.SourceVectorRegister + index}]";
                        Line($"{ScratchAccessWord(control, byteAddress, index * sizeof(uint))} = {value};");
                    }
                }

                EndExecGuard();
                return true;
            }

            if (!instruction.Opcode.StartsWith("ScratchLoad", StringComparison.Ordinal))
            {
                error = $"unsupported scratch opcode {instruction.Opcode}";
                return false;
            }

            var destination = control.DestinationVectorRegister;
            if (TryGetSubdwordLoadInfo(instruction.Opcode, out var loadBytes, out var signExtend, out var d16, out var d16High))
            {
                var shift = Temp("uint", $"({byteAddress} & 3u) << 3u");
                var loaded = Temp(
                    "uint",
                    $"({ScratchAccessWord(control, byteAddress)} >> {shift}) & {FormatUInt(loadBytes == 1 ? 0xFFu : 0xFFFFu)}");
                if (signExtend)
                {
                    loaded = Temp("uint", $"(uint)extract_bits(as_type<int>({loaded}), 0u, {loadBytes * 8}u)");
                }

                StoreVector(
                    destination,
                    !d16 ? loaded
                        : d16High
                            ? $"(v[{destination}] & 0x0000FFFFu) | (({loaded} & 0xFFFFu) << 16)"
                            : $"(v[{destination}] & 0xFFFF0000u) | ({loaded} & 0xFFFFu)");
                return true;
            }

            for (uint index = 0; index < control.DwordCount; index++)
            {
                StoreVector(destination + index, ScratchAccessWord(control, byteAddress, index * sizeof(uint)));
            }

            return true;
        }

        // The scratch array element one byte address names. The dword index wraps
        // into the allocation, so an out-of-range guest address stays inside it.
        private string ScratchWord(string byteAddress, uint offsetBytes = 0)
        {
            var address = offsetBytes == 0 ? byteAddress : $"({byteAddress} + {offsetBytes}u)";
            return _scratchDwordCount > 1
                ? $"sharpemu_scratch[(({address}) >> 2) % {_scratchDwordCount}u]"
                : "sharpemu_scratch[0]";
        }

        // ---- buffers ----

        private bool TryResolveLayoutBuffer(uint pc, out int bindingIndex, out BufferResource resource)
        {
            bindingIndex = -1;
            resource = null!;
            var request = _request;
            if (!request.Memory.TryGetIndex(pc, 0, out var memoryIndex))
            {
                return false;
            }

            var entry = request.Memory[memoryIndex];
            if (entry.Resource == MemoryAccessInfo.NoResource || entry.Resource >= (uint)request.Resources.Info.Buffers.Count)
            {
                return false;
            }

            bindingIndex = (int)entry.Resource;
            resource = request.Resources.Info.Buffers[bindingIndex];
            return true;
        }

        // ---- images ----

        // An access over several descriptors takes one constant-element case per descriptor:
        // the mip operand selects a per-mip element, the indirect key a candidate.
        private bool TryGetImageElementCases(Gen5ShaderInstruction instruction, Gen5ImageControl image, out string selector, out IReadOnlyList<(uint Resource, uint Element)> elements, out string error)
        {
            selector = string.Empty;
            elements = [];
            error = string.Empty;
            var request = _request;
            if (!request.Memory.TryGetIndex(instruction.Pc, 0, out var memoryIndex))
            {
                return false;
            }

            var entry = request.Memory[memoryIndex];
            var info = request.Resources.Info;
            if (entry.Resource == MemoryAccessInfo.NoResource || entry.Resource >= (uint)info.Images.Count)
            {
                return false;
            }

            var resourceIndex = (int)entry.Resource;
            var imageInfo = info.Images[resourceIndex];
            var kind = ImageDescriptorBinding.ForImage(imageInfo);
            if (kind is null || !_imageClasses.TryGetValue(kind.Value, out var imageClass))
            {
                return false;
            }

            var classElements = imageClass.Resources.ToList();
            var element = classElements.IndexOf(info.GetCanonicalImageBinding((uint)resourceIndex));
            if (element < 0)
            {
                return false;
            }

            if (imageInfo.MipMode == ImageMipMode.DynamicStorage && Gen5ImageOpcodes.Has(instruction.Opcode, Gen5ImageOp.Mip))
            {
                // A mip past the last descriptor matches no case and does nothing.
                selector = Temp("uint", ImageIntegerAddress(image, imageClass.CoordinateComponents));
                elements = Enumerable.Range(0, (int)imageInfo.MipCount)
                    .Select(mip => ((uint)resourceIndex, (uint)element + (uint)mip)).ToList();
                return true;
            }

            if (imageInfo.BindlessSlot != DescriptorConstants.NoIndex)
            {
                return TryGetBindlessElementCases(memoryIndex, imageInfo, out selector, out elements, out error);
            }

            if (request.IndirectRootByMemoryIndex.TryGetValue(memoryIndex, out var keyMemoryIndex) && imageInfo.IndirectSearchIterations != 0)
            {
                if (!_hasFlattenedTable)
                {
                    error = "indirect image access without a flattened table binding";
                    return false;
                }

                var candidates = info.Images[(int)imageInfo.IndirectRoot].IndirectResources;
                var candidateElements = new List<(uint Resource, uint Element)>();
                foreach (var candidate in candidates)
                {
                    var candidateKind = ImageDescriptorBinding.ForImage(info.Images[(int)candidate]);
                    if (candidateKind is null || !_imageClasses.TryGetValue(candidateKind.Value, out var candidateClass))
                    {
                        error = $"indirect candidate {candidate} has no declared binding class";
                        return false;
                    }
                    var candidateElement = candidateClass.Resources.ToList().IndexOf(info.GetCanonicalImageBinding(candidate));
                    if (candidateElement < 0)
                    {
                        error = $"indirect candidate {candidate} is not an element of {candidateKind.Value}";
                        return false;
                    }

                    candidateElements.Add((candidate, (uint)candidateElement));
                }

                selector = SelectIndirectCandidate(imageInfo, keyMemoryIndex, (uint)candidateElements.Count);
                elements = candidateElements;
                return true;
            }

            return false;
        }

        // Resolves the image an instruction reads through its class array into a texture local,
        // with its sampler for sampling operations. Every image dimension the binding layout
        // can describe is accessible; the resolved class carries the shape the caller needs
        // to build coordinates with.
        private bool TryResolveLayoutImage(
            Gen5ShaderInstruction instruction,
            Gen5ImageControl image,
            out string texture,
            out string samplerName,
            out MslImageClass resolved,
            out uint dstSelect,
            out string mipLevel,
            out int compareFunction,
            out uint packedFormat,
            out SamplerResource? samplerInfo,
            out string error,
            out int resourceIndex,
            (uint Resource, uint Element)? fixedElement = null)
        {
            error = string.Empty;
            resourceIndex = -1;
            compareFunction = -1;
            samplerInfo = null;
            packedFormat = DescriptorConstants.InvalidFormat;
            texture = string.Empty;
            samplerName = string.Empty;
            resolved = default;
            dstSelect = DescriptorConstants.IdentityImageSwizzle;
            mipLevel = "0u";
            var request = _request;
            var info = request.Resources.Info;
            if (!request.Memory.TryGetIndex(instruction.Pc, 0, out var memoryIndex))
            {
                error = "image access has no memory record";
                return false;
            }

            var entry = request.Memory[memoryIndex];
            if (entry.Resource == MemoryAccessInfo.NoResource || entry.Resource >= (uint)info.Images.Count)
            {
                error = "image access has no dense image";
                return false;
            }

            resourceIndex = (int)(fixedElement?.Resource ?? entry.Resource);
            var imageInfo = info.Images[resourceIndex];
            var bindingKind = ImageDescriptorBinding.ForImage(imageInfo);
            if (bindingKind is null || !_imageClasses.TryGetValue(bindingKind.Value, out var imageClass))
            {
                error = $"image {resourceIndex} has no declared binding class";
                return false;
            }

            // Metal multisampled textures expose read() with a sample index and nothing
            // else: no sampling, no writes. The binding layout already refuses storage
            // MSAA classes, so only the sampled fetch paths can reach one.
            if (imageClass.Multisampled && instruction.Opcode is not ("ImageLoad" or "ImageLoadMip" or "ImageMsaaLoad" or "ImageGetResinfo"))
            {
                error = $"multisampled image opcode {instruction.Opcode} is not supported on Metal";
                return false;
            }

            var element = imageClass.Resources.ToList().IndexOf(info.GetCanonicalImageBinding((uint)resourceIndex));
            if (element < 0)
            {
                error = $"image {resourceIndex} is not an element of {bindingKind.Value}";
                return false;
            }

            // Several descriptors (per-mip elements, indirect candidates) need the caller's constant case.
            var severalDescriptors =
                (imageInfo.MipMode == ImageMipMode.DynamicStorage && Gen5ImageOpcodes.Has(instruction.Opcode, Gen5ImageOp.Mip)) ||
                (request.IndirectRootByMemoryIndex.ContainsKey(memoryIndex) && imageInfo.IndirectSearchIterations != 0);
            if (fixedElement is null && severalDescriptors)
            {
                error = "image access over several descriptors needs a constant element case";
                return false;
            }

            // A sampled mip load keeps its mip operand; a storage case is already its own mip view.
            // The mip is the address component right after the coordinates, so it moves with
            // the dimension (2 for 2D, 3 for a 2D array or a volume, 1 for 1D).
            if (!imageClass.IsStorage && Gen5ImageOpcodes.Has(instruction.Opcode, Gen5ImageOp.Mip))
            {
                mipLevel = Temp("uint", ImageIntegerAddress(image, imageClass.CoordinateComponents));
            }

            if (imageInfo.BindlessSlot != DescriptorConstants.NoIndex)
            {
                if (fixedElement is null || _bindlessAccess is not { } bindless)
                {
                    error = "bindless image access needs its variant case";
                    return false;
                }

                texture = Temp(imageClass.TextureType,
                    $"((device const {BindlessTypedStruct(imageClass.TextureType)}*){bindless.Table})[{bindless.Index}].texture");
            }
            else
            {
                texture = Temp(imageClass.TextureType, $"{ResourcesName}.{imageClass.Field}[{fixedElement?.Element ?? (uint)element}u]");
            }

            resolved = imageClass;
            if (UsesSampler(instruction.Opcode))
            {
                if (!_hasSamplers)
                {
                    error = "sampling without a sampler binding";
                    return false;
                }

                if (!request.Resources.SamplerByMemoryIndex.TryGetValue(memoryIndex, out var samplerIndex))
                {
                    samplerIndex = entry.Sampler;
                }

                if (samplerIndex == MemoryAccessInfo.NoResource)
                {
                    error = "sampled access has no dense sampler";
                    return false;
                }

                var samplerElement = request.Bindings.Find(DescriptorBindingKind.Samplers)!.Resources.ToList()
                    .IndexOf(info.SamplerBinding(samplerIndex));
                samplerName = Temp("sampler", $"{ResourcesName}.samplers[{samplerElement}]");
                samplerInfo = info.Samplers[(int)samplerIndex];
            }

            dstSelect = imageInfo.ShaderSwizzle;
            // A compare access of a format with no host depth view tests the texel in the
            // shader with the paired S#'s function, which the materializer recorded.
            compareFunction = imageInfo.EmulatedCompareFunction;
            packedFormat = imageInfo.PackedFormat;
            return true;
        }

        // A V#-described table read through its device handle table: the captured key is the
        // scalar load's offset, which names entry (key - base) >> 5; one past the last entry
        // is the null entry the host appends, so a key outside the table reads zero as the
        // range-checked load would. The entry's variant selects the case.
        private bool TryGetBindlessElementCases(int memoryIndex, ImageResource imageInfo, out string selector,
            out IReadOnlyList<(uint Resource, uint Element)> elements, out string error)
        {
            selector = string.Empty;
            elements = [];
            error = string.Empty;
            var request = _request;
            var info = request.Resources.Info;
            if (!_hasFlattenedTable ||
                _argumentLayout?.Fields.Any(static field => field.FieldKind == MslArgumentFieldKind.BindlessImageTables) != true)
            {
                error = "bindless image access without a flattened table or handle table binding";
                return false;
            }

            if (!request.IndirectRootByMemoryIndex.TryGetValue(memoryIndex, out var keyMemoryIndex) ||
                !_indirectKeyScratch.TryGetValue(keyMemoryIndex, out var key))
            {
                error = "bindless image access has no captured key";
                return false;
            }

            if (imageInfo.IndirectRoot >= (uint)info.Images.Count)
            {
                error = "bindless image access has an invalid root";
                return false;
            }

            var variants = info.Images[(int)imageInfo.IndirectRoot].IndirectResources;
            var variantElements = new List<(uint Resource, uint Element)>(variants.Count);
            foreach (var variant in variants)
            {
                var variantInfo = info.Images[(int)variant];
                var variantKind = ImageDescriptorBinding.ForImage(variantInfo);
                if (variantInfo.BindlessSlot != imageInfo.BindlessSlot || variantKind is null ||
                    !_imageClasses.TryGetValue(variantKind.Value, out var variantClass))
                {
                    error = $"bindless variant {variant} has no declared binding class";
                    return false;
                }

                var variantElement = variantClass.Resources.ToList().IndexOf(info.GetCanonicalImageBinding(variant));
                if (variantElement < 0)
                {
                    error = $"bindless variant {variant} is not an element of {variantKind.Value}";
                    return false;
                }

                variantElements.Add((variant, (uint)variantElement));
            }

            var mapping = imageInfo.IndirectMappingOffset;
            var count = Temp("uint", LoadFlattenedWord($"{mapping}u"));
            var first = Temp("uint", LoadFlattenedWord($"{mapping + 1}u"));
            var entry = Temp("uint", $"({key} - {first}) >> 5u");
            var index = Temp("uint", $"{entry} < {count} ? {entry} : {count}");
            var table = Temp($"device const {BindlessEntryStruct}*",
                $"{ResourcesName}.{Gen5MslArgumentLayout.BindlessImagesName}[{imageInfo.BindlessSlot}u]");
            selector = Temp("uint", $"{table}[{index}].variant");
            _bindlessAccess = (table, index);
            elements = variantElements;
            return true;
        }

        // Searches the sorted key mapping of an indirect table for the captured key; the
        // result is the candidate-local index, candidate 0 when the key is absent or out of range.
        private string SelectIndirectCandidate(ImageResource imageInfo, int keyMemoryIndex, uint candidateCount)
        {
            var key = _indirectKeyScratch.TryGetValue(keyMemoryIndex, out var scratch) ? scratch : "0u";
            var mapping = imageInfo.IndirectMappingOffset;
            var count = Temp("uint", LoadFlattenedWord($"{mapping}u"));
            var low = Temp("uint", "0u");
            var high = Temp("uint", count);
            for (uint iteration = 0; iteration < imageInfo.IndirectSearchIterations; iteration++)
            {
                var middle = Temp("uint", $"{low} + (({high} - {low}) >> 1u)");
                var probeSlot = Temp("uint", $"{mapping + 1}u + ({middle} << 1u)");
                var probeKey = Temp("uint", LoadFlattenedWord(probeSlot));
                var moveUp = Temp("bool", $"{probeKey} < {key} && {low} < {high}");
                Line($"{low} = {moveUp} ? {middle} + 1u : {low};");
                Line($"{high} = {moveUp} ? {high} : {middle};");
            }

            var foundSlot = Temp("uint", $"{mapping + 1}u + ({low} << 1u)");
            var foundKey = Temp("uint", LoadFlattenedWord(foundSlot));
            var found = Temp("bool", $"{low} < {count} && {foundKey} == {key}");
            var mappedSlot = Temp("uint", $"{foundSlot} + 1u");

            // The mapping holds candidate-local indices: the root's candidate list in order.
            var mapped = Temp("uint", LoadFlattenedWord(mappedSlot));
            return Temp("uint", $"{found} && {mapped} < {candidateCount}u ? {mapped} : 0u");
        }

        // ---- global data share ----

        // DS operations with the GDS bit run over the shared buffer with device atomics;
        // lane operations have no GDS form.
        private bool TryEmitGlobalDataShare(Gen5ShaderInstruction instruction, Gen5DataShareControl control, out string error)
        {
            error = string.Empty;
            if (!_hasGlobalDataShare)
            {
                error = "GDS access without a global data share binding";
                return false;
            }

            var opcode = instruction.Opcode;
            if (opcode.Contains("Swizzle", StringComparison.Ordinal) || opcode.Contains("Bpermute", StringComparison.Ordinal) ||
                opcode.Contains("Permute", StringComparison.Ordinal) || opcode.Contains("Addtid", StringComparison.Ordinal))
            {
                error = $"lane operation {opcode} is not valid on the global data share";
                return false;
            }

            switch (opcode)
            {
                case "DsAppend":
                case "DsConsume":
                    return TryEmitGlobalDataShareCounter(instruction, control, out error);
                case "DsOrderedCount":
                    return TryEmitOrderedCount(instruction, control, out error);
                case "DsGwsInit":
                case "DsGwsSemaV":
                case "DsGwsSemaBr":
                case "DsGwsSemaP":
                case "DsGwsSemaReleaseAll":
                case "DsGwsBarrier":
                    return TryEmitGlobalWaveSync(instruction, control, out error);
                case "DsWriteB32":
                {
                    var index = Temp("uint", GlobalDataShareIndex(GlobalDataShareAddress(instruction), control.SingleOffsetBytes));
                    StoreGlobalDataShareWord(index, RawSource(instruction, 1));
                    return true;
                }

                case "DsWriteB64":
                {
                    var index = Temp("uint", GlobalDataShareIndex(GlobalDataShareAddress(instruction), control.SingleOffsetBytes));
                    StoreGlobalDataShareWord(index, RawSource(instruction, 1));
                    StoreGlobalDataShareWord($"({index} + 1u)", RawSource(instruction, 2));
                    return true;
                }

                case "DsWrite2B64":
                case "DsWrite2St64B64":
                    return TryEmitDataShareWritePair64(instruction, control, out error);
                case "DsWrite2B32":
                {
                    var address = Temp("uint", GlobalDataShareAddress(instruction));
                    var first = Temp("uint", GlobalDataShareIndex(address, control.Offset0 * sizeof(uint)));
                    var second = Temp("uint", GlobalDataShareIndex(address, control.Offset1 * sizeof(uint)));
                    StoreGlobalDataShareWord(first, RawSource(instruction, 1));
                    StoreGlobalDataShareWord(second, RawSource(instruction, 2));
                    return true;
                }

                case "DsReadB32":
                {
                    var index = Temp("uint", GlobalDataShareIndex(GlobalDataShareAddress(instruction), control.SingleOffsetBytes));
                    StoreVector(instruction.Destinations[0].Value, LoadGlobalDataShareWord(index));
                    return true;
                }

                case "DsReadI8":
                case "DsReadU8":
                {
                    // The byte the low two bits of the offset address select,
                    // sign- or zero-extended out of the word that holds it.
                    var address = Temp("uint", GlobalDataShareAddress(instruction));
                    var byteAddress = Temp("uint", $"{address} + {control.SingleOffsetBytes}u");
                    var word = Temp("uint", LoadGlobalDataShareWord(
                        Temp("uint", GlobalDataShareIndex(address, control.SingleOffsetBytes))));
                    var shift = Temp("uint", $"({byteAddress} & 3u) << 3u");
                    StoreVector(
                        instruction.Destinations[0].Value,
                        $"(uint)(((int)(({word} >> {shift}) << 24u)) >> 24)");
                    return true;
                }

                case "DsReadB64":
                {
                    var index = Temp("uint", GlobalDataShareIndex(GlobalDataShareAddress(instruction), control.SingleOffsetBytes));
                    StoreVector(instruction.Destinations[0].Value, LoadGlobalDataShareWord(index));
                    StoreVector(instruction.Destinations[1].Value, LoadGlobalDataShareWord($"({index} + 1u)"));
                    return true;
                }

                case "DsRead2B64":
                    return TryEmitDataShareReadPair64(instruction, control, out error);
                case "DsRead2B32":
                {
                    var address = Temp("uint", GlobalDataShareAddress(instruction));
                    var first = Temp("uint", GlobalDataShareIndex(address, control.Offset0 * sizeof(uint)));
                    var second = Temp("uint", GlobalDataShareIndex(address, control.Offset1 * sizeof(uint)));
                    StoreVector(instruction.Destinations[0].Value, LoadGlobalDataShareWord(first));
                    StoreVector(instruction.Destinations[1].Value, LoadGlobalDataShareWord(second));
                    return true;
                }

                default:
                    if (Gen5ShaderTranslator.IsDataShareAtomic(opcode))
                    {
                        return TryEmitGlobalDataShareAtomic(instruction, control, out error);
                    }

                    error = $"unsupported GDS opcode {opcode}";
                    return false;
            }
        }

        private bool TryEmitDataShareWritePair64(Gen5ShaderInstruction instruction, Gen5DataShareControl control, out string error)
        {
            error = string.Empty;
            var offsetStride = instruction.Opcode == "DsWrite2St64B64" ? 512u : 8u;
            if (instruction.Sources.Count < 5)
            {
                error = "The paired 64-bit write requires an address and two source-register pairs.";
                return false;
            }

            BeginExecGuard();
            var address = Temp("uint", control.Gds ? GlobalDataShareAddress(instruction) : RawSource(instruction, 0));
            // Equal offsets select only the first source pair.
            var componentCount = control.Offset0 == control.Offset1 ? 2 : 4;
            for (var component = 0; component < componentCount; component++)
            {
                var pairOffset = component < 2 ? control.Offset0 : control.Offset1;
                var byteOffset = pairOffset * offsetStride + (uint)(component % 2) * sizeof(uint);
                var value = RawSource(instruction, component + 1);
                if (control.Gds)
                    StoreGlobalDataShareWord(GlobalDataShareIndex(address, byteOffset), value);
                else
                    Line($"sharpemu_lds[{LdsIndex(address, byteOffset)}] = {value};");
            }
            EndExecGuard();
            return true;
        }

        private bool TryEmitDataShareReadPair64(Gen5ShaderInstruction instruction, Gen5DataShareControl control, out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 1 || instruction.Destinations.Count < 4)
            {
                error = "The paired 64-bit read requires an address and four destination registers.";
                return false;
            }

            BeginExecGuard();
            var address = Temp("uint", control.Gds ? GlobalDataShareAddress(instruction) : RawSource(instruction, 0));
            var values = new string[4];
            // Capture both values before an overlapping destination changes the address.
            for (var component = 0; component < values.Length; component++)
            {
                var pairOffset = component < 2 ? control.Offset0 : control.Offset1;
                var stride = instruction.Opcode == "DsRead2St64B64" ? 512u : sizeof(ulong);
                var byteOffset = pairOffset * stride + (uint)(component % 2) * sizeof(uint);
                values[component] = Temp("uint", control.Gds
                    ? LoadGlobalDataShareWord(GlobalDataShareIndex(address, byteOffset))
                    : $"sharpemu_lds[{LdsIndex(address, byteOffset)}]");
            }
            for (var component = 0; component < values.Length; component++)
                StoreVector(instruction.Destinations[component].Value, values[component], guardWithExec: false);
            EndExecGuard();
            return true;
        }

        // A GDS access addresses the data share from M0's base (bits 31:16, the size below it),
        // as append and consume do: the ADDR VGPR and the offset are relative to it. Without it
        // every per-slot counter read the slot at the start of the share - Ghost of Tsushima's
        // grass draws took their instance count from a word that is never reset, and drew more
        // instances each frame until the frame took seconds.
        private string GlobalDataShareAddress(Gen5ShaderInstruction instruction) =>
            $"(({RawSource(instruction, 0)}) + ({ScalarExpression(M0Register)} >> 16u))";

        private static string GlobalDataShareIndex(string address, uint offsetBytes) =>
            offsetBytes == 0 ? $"({address}) >> 2u" : $"(({address}) + {offsetBytes}u) >> 2u";

        private static string LoadGlobalDataShareWord(string index) =>
            $"({index} < {GlobalDataShareWordsName} ? {GlobalDataShareName}[{index}] : 0u)";

        private void StoreGlobalDataShareWord(string index, string value) =>
            Line(ExecGuarded($"if ({index} < {GlobalDataShareWordsName}) {{ {GlobalDataShareName}[{index}] = {value}; }}"));

        // The atomic suffix TryGetAtomicExpression knows for one DS opcode, shared
        // by the LDS and GDS paths because both encodings name the same operation.
        private static string DataShareAtomicName(string opcode) => opcode switch
        {
            "DsAddU32" or "DsAddRtnU32" => "Add",
            "DsSubU32" or "DsSubRtnU32" => "Sub",
            "DsIncU32" or "DsIncRtnU32" => "Inc",
            "DsDecU32" or "DsDecRtnU32" => "Dec",
            "DsMinI32" or "DsMinRtnI32" => "Smin",
            "DsMaxI32" or "DsMaxRtnI32" => "Smax",
            "DsMinU32" or "DsMinRtnU32" => "Umin",
            "DsMaxU32" or "DsMaxRtnU32" => "Umax",
            "DsAndB32" or "DsAndRtnB32" => "And",
            "DsOrB32" or "DsOrRtnB32" => "Or",
            "DsXorB32" or "DsXorRtnB32" => "Xor",
            "DsWrxchgRtnB32" => "Swap",
            "DsCmpstB32" or "DsCmpstRtnB32" => "Cmpswap",
            _ => string.Empty,
        };

        private bool TryEmitGlobalDataShareAtomic(Gen5ShaderInstruction instruction, Gen5DataShareControl control, out string error)
        {
            error = string.Empty;
            var name = instruction.Opcode switch
            {
                // A min/max of DATA0 and memory, as on the LDS path (DataShareFloatMinMax).
                "DsMinF32" or "DsMinRtnF32" => "Fmin",
                "DsMaxF32" or "DsMaxRtnF32" => "Fmax",
                _ => DataShareAtomicName(instruction.Opcode),
            };
            // DS_CMPST reads DATA0 as the comparator and DATA1 as the new value.
            var valueIndex = name == "Cmpswap" ? 2 : 1;
            var comparatorIndex = 1;
            var value = instruction.Sources.Count > valueIndex ? RawSource(instruction, valueIndex) : "0u";
            var comparator = instruction.Sources.Count > comparatorIndex ? RawSource(instruction, comparatorIndex) : "0u";
            if (name.Length == 0 || !TryGetAtomicExpression(name, value, comparator, out var build))
            {
                error = $"unsupported GDS opcode {instruction.Opcode}";
                return false;
            }

            var index = Temp("uint", GlobalDataShareIndex(GlobalDataShareAddress(instruction), control.SingleOffsetBytes));
            Line($"if ({ExecCondition($"{index} < {GlobalDataShareWordsName}")})");
            Line("{");
            _indent++;
            var original = Temp("uint", build($"{GlobalDataShareName} + {index}"));
            if (instruction.Destinations.Count > 0)
            {
                Line($"v[{instruction.Destinations[0].Value}] = {original};");
            }

            _indent--;
            Line("}");
            return true;
        }

        // Evaluates `operation` once per guest wave, in its first active lane (whose own
        // registers it reads), and returns the result broadcast to the wave. A wave with
        // no active lane does nothing. The rest of the wave waits for the leader: a wave32
        // is one SIMD-group, and a wave64's halves meet at the wave-exchange barrier.
        private string EmitWaveLeaderOperation(string operation)
        {
            if (IsWave64)
            {
                return EmitWave64AppendAtomic(_ => operation, "true");
            }

            var mask = Temp("uint", "sharpemu_ballot(exec)");
            var first = Temp("uint", $"{mask} == 0u ? 0u : (uint)ctz({mask})");
            var value = Temp("uint", "0u");
            Line($"if ({ExecCondition($"sharpemu_lane == {first}")}) {{ {value} = {operation}; }}");
            return Temp("uint", $"simd_broadcast({value}, {first})");
        }

        // DS_ORDERED_COUNT (RDNA2 ISA 12.13; LLVM's llvm.amdgcn.ds.ordered.add/swap lowering):
        // the ADDR-field VGPR of the wave's first active lane is the value, OFFSET0 bits 7:2
        // the GDS dword (the ordered counter's index), OFFSET1 bit 0 wave_release, bit 1
        // wave_done, bits 3:2 the shader type, bit 4 swap instead of add and bits 7:6 the
        // dword count less one. The wave adds (or swaps in) the value once and every active
        // lane receives the counter's previous value.
        //
        // The hardware also orders the waves: a wave's add waits until every wave launched
        // before it (by the dispatch order M0 identifies) has issued its own with
        // wave_release, so each wave's range follows its predecessors'. The host has no
        // launch-order walker and Metal no launch order to give one, so here the waves add
        // in the order they reach the counter: every wave still gets a disjoint range, the
        // ranges still tile [start, start + total) and the total is the same - what an
        // ordered append's compaction needs - but the ranges may be permuted across waves,
        // so output a guest orders by wave may come out in another order. wave_release and
        // wave_done have nothing to release, so they are ignored.
        private bool TryEmitOrderedCount(Gen5ShaderInstruction instruction, Gen5DataShareControl control, out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 1 || instruction.Destinations.Count < 1)
            {
                error = $"missing {instruction.Opcode} operand";
                return false;
            }

            if (((control.Offset1 >> 6) & 3) != 0)
            {
                error = $"{instruction.Opcode} over {((control.Offset1 >> 6) & 3) + 1} dwords is not implemented";
                return false;
            }

            if (!HasRealWaveLanes)
            {
                error = $"{instruction.Opcode} needs wave lanes";
                return false;
            }

            var swap = ((control.Offset1 >> 4) & 1) != 0;
            var index = Temp("uint", $"{(control.Offset0 >> 2) & 0x3F}u");
            var word = $"(device atomic_uint*)({GlobalDataShareName} + {index})";
            var operation = swap ? "atomic_exchange_explicit" : "atomic_fetch_add_explicit";
            var result = EmitWaveLeaderOperation(
                $"({index} < {GlobalDataShareWordsName} ? {operation}({word}, {RawSource(instruction, 0)}, memory_order_relaxed) : 0u)");
            StoreVector(instruction.Destinations[0].Value, result);
            return true;
        }

        // DS_GWS_* (see the GWS section of Templates/pair_atomic_prelude.msl): once per wave,
        // on resource M0[21:16] + OFFSET0[5:0], with the first active lane's ADDR-field VGPR
        // as DATA for INIT, SEMA_BR and BARRIER.
        private bool TryEmitGlobalWaveSync(Gen5ShaderInstruction instruction, Gen5DataShareControl control, out string error)
        {
            error = string.Empty;
            var takesData = instruction.Opcode is "DsGwsInit" or "DsGwsSemaBr" or "DsGwsBarrier";
            if (instruction.Sources.Count < (takesData ? 2 : 1))
            {
                error = $"missing {instruction.Opcode} operand";
                return false;
            }

            if (!HasRealWaveLanes)
            {
                error = $"{instruction.Opcode} needs wave lanes";
                return false;
            }

            _usesPairLocks = true;
            var m0 = Temp("uint", RawSource(instruction, instruction.Sources.Count - 1));
            var resource = Temp(
                "device atomic_uint*",
                $"sharpemu_gws_resource({PairLocksExpression}, {m0}, {control.Offset0 & 0x3F}u)");
            var data = takesData ? RawSource(instruction, 0) : "0u";
            var call = instruction.Opcode switch
            {
                "DsGwsInit" => $"sharpemu_gws_init({resource}, {data})",
                "DsGwsSemaV" => $"sharpemu_gws_sema_release({resource}, 1u)",
                "DsGwsSemaBr" => $"sharpemu_gws_sema_release({resource}, {data})",
                "DsGwsSemaP" => $"sharpemu_gws_sema_p({resource}, {PairStepReportExpression})",
                "DsGwsSemaReleaseAll" => $"sharpemu_gws_sema_release_all({resource})",
                _ => $"sharpemu_gws_barrier({resource}, {data}, {PairStepReportExpression})",
            };
            EmitWaveLeaderOperation(call);
            return true;
        }

        // Append/consume on the GDS counter at M0's base: M0 must carry a size, and the word must be inside the buffer.
        private bool TryEmitGlobalDataShareCounter(Gen5ShaderInstruction instruction, Gen5DataShareControl control, out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 1 || instruction.Destinations.Count < 1)
            {
                error = $"missing {instruction.Opcode} operand";
                return false;
            }

            var offset = control.SingleOffsetBytes;
            var m0 = Temp("uint", RawSource(instruction, 0));
            var baseAddress = Temp("uint", $"{m0} >> 16u");
            var sizeBytes = Temp("uint", $"{m0} & 0xFFFFu");
            var index = Temp("uint", GlobalDataShareIndex(baseAddress, offset));
            var inBounds = Temp("bool", $"{sizeBytes} != 0u && {index} < {GlobalDataShareWordsName}");
            var destination = instruction.Destinations[0].Value;
            var operation = instruction.Opcode == "DsAppend" ? "add" : "sub";
            var word = $"(device atomic_uint*)({GlobalDataShareName} + {index})";

            // Graphics stages model one logical lane, so the counter grows by one per invocation.
            if (!HasRealWaveLanes)
            {
                var original = Temp("uint", "0u");
                Line($"if ({ExecCondition(inBounds)}) {{ {original} = atomic_fetch_{operation}_explicit({word}, 1u, memory_order_relaxed); }}");
                StoreVector(destination, $"{inBounds} ? {original} : 0u");
                return true;
            }

            string count;
            string first;
            if (IsWave64)
            {
                var broadcast = EmitWave64AppendAtomic(
                    waveCount => $"atomic_fetch_{operation}_explicit({word}, {waveCount}, memory_order_relaxed)",
                    inBounds);
                StoreVector(destination, $"{inBounds} ? {broadcast} : 0u");
                return true;
            }

            var mask = Temp("uint", "sharpemu_ballot(exec)");
            count = Temp("uint", $"popcount({mask})");
            first = Temp("uint", $"{mask} == 0u ? 0u : (uint)ctz({mask})");
            var firstValue = Temp("uint", "0u");
            Line($"if ({ExecCondition($"{inBounds} && sharpemu_lane == {first}")})");
            Line("{");
            _indent++;
            Line($"{firstValue} = atomic_fetch_{operation}_explicit({word}, {count}, memory_order_relaxed);");
            _indent--;
            Line("}");
            var result = Temp("uint", $"simd_broadcast({firstValue}, {first})");
            StoreVector(destination, $"{inBounds} ? {result} : 0u");
            return true;
        }
    }
}
