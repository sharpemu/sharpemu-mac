// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using MslImageResourceClass = SharpEmu.ShaderCompiler.Resources.ImageResourceClass;

namespace SharpEmu.Gpu.Metal.V2;

// One Tier 2 argument buffer per stage, filled field by field in the layout's
// order: window pointers for buffers, gpuResourceIDs for textures and samplers.
// The argument buffer binds at index 0 and the shared push block at index 1.
internal sealed partial class MetalRenderHost
{
    private const int StageVertex = 0;
    private const int StageFragment = 1;
    private const int StageCompute = 2;
    private const int StageCount = 3;

    private static readonly nint SelGpuResourceId = MetalNative.Selector("gpuResourceID");
    private static readonly nint SelGpuAddress = MetalNative.Selector("gpuAddress");
    private static readonly nint SelNewSamplerState = MetalNative.Selector("newSamplerStateWithDescriptor:");
    private static readonly nint SelSetMeshBuffer = MetalNative.Selector("setMeshBuffer:offset:atIndex:");

    // The host descriptors of one stage, reused across draws so a draw allocates nothing.
    internal sealed class MetalStageBindings : IPreparedBindings
    {
        public ShaderStageResources StageResources;
        public ShaderProgramInfo Program = null!;
        public nint[] Textures = [];
        public uint[] TextureLevels = [];
        public ulong[] TextureIds = [];

        // A dynamic-mip storage image's views of its levels past the first, [mip - 1]:
        // image_store_mip and image_load_mip name the level, each a view of its own.
        public nint[]?[] MipTextures = [];
        public ulong[]?[] MipTextureIds = [];
        public nint[] Samplers = [];
        public ulong[] SamplerIds = [];
        public byte[] SamplerViewClasses = [];
        public ResourceSlotIdentifier[] Images = [];
        public ImageRequest[] Requests = [];
        public bool[] IsStorage = [];

        // Per image: the memo entry a sampled binding was found in while its stamp held, or null;
        // and for one resolved in full, whether it may be memoised, its key, and the owner-list
        // counts over the request read before its lookup.
        public TextureMemoEntry?[] MemoHits = [];
        public bool[] MemoWanted = [];
        public TextureMemoKey[] MemoKeys = [];
        public long[] MemoPages = [];
        public MetalGuestRange[] Buffers = [];
        public uint[] BufferLengths = [];
        public uint[] ShaderData = [];
        public ulong[] BindlessTableAddresses = [];
        public int ImageCount;
        public int SamplerCount;
        public int BufferCount;
        public bool Committed;

        // The images a host-decoded movie frame replaces in this stage, or -1.
        public int MovieLuma = -1;
        public int MovieChroma = -1;

        public ShaderStageResources Stage => StageResources;

        public void EnsureCapacity(int images, int samplers, int buffers, int shaderDwords)
        {
            if (Textures.Length < images)
            {
                Textures = new nint[images];
                TextureIds = new ulong[images];
                MipTextures = new nint[]?[images];
                MipTextureIds = new ulong[]?[images];
                TextureLevels = new uint[images];
                Images = new ResourceSlotIdentifier[images];
                Requests = new ImageRequest[images];
                IsStorage = new bool[images];
                MemoHits = new TextureMemoEntry?[images];
                MemoWanted = new bool[images];
                MemoKeys = new TextureMemoKey[images];
                MemoPages = new long[images];
            }

            if (Samplers.Length < samplers)
            {
                Samplers = new nint[samplers];
                SamplerIds = new ulong[samplers];
                SamplerViewClasses = new byte[samplers];
            }

            if (Buffers.Length < buffers)
            {
                Buffers = new MetalGuestRange[buffers];
                BufferLengths = new uint[buffers];
            }

            if (ShaderData.Length < shaderDwords)
            {
                ShaderData = new uint[shaderDwords];
            }

            ImageCount = images;
            SamplerCount = samplers;
            BufferCount = buffers;
        }
    }

    private sealed class MetalResourcePreparation(MetalRenderHost owner) : IResourcePreparation
    {
        public void Dispose() => owner.EndPreparation();
    }

    private readonly MetalStageBindings[] _stageBindings =
        [new MetalStageBindings(), new MetalStageBindings(), new MetalStageBindings()];

    // The argument layout of one binding layout, with the per-field resource indices that
    // a draw would otherwise look up by scanning the descriptors.
    private sealed class ArgumentPlan(Gen5MslArgumentLayout layout, int[]?[] fieldResources, int[]?[] fieldMips)
    {
        public readonly Gen5MslArgumentLayout Layout = layout;
        public readonly int[]?[] FieldResources = fieldResources;

        // Per texture element, the mip it binds of its image: a dynamic-mip storage image
        // repeats its resource index once per level, in level order.
        public readonly int[]?[] FieldMips = fieldMips;
    }

    /// <summary>
    /// The level each element of a texture field binds: the k-th consecutive repeat of a
    /// resource index is mip k of that image, every other element mip 0.
    /// </summary>
    private static int[] ElementMips(IReadOnlyList<uint> resources)
    {
        var mips = new int[resources.Count];
        for (var index = 1; index < mips.Length; index++)
        {
            mips[index] = resources[index] == resources[index - 1] ? mips[index - 1] + 1 : 0;
        }

        return mips;
    }

    private readonly Dictionary<BindingLayout, ArgumentPlan> _argumentLayouts = new();

    // The same plans by layout instance: a program keeps its layout, so a draw finds its plan by
    // reference instead of comparing the layout field by field (BindingLayout.Equals).
    private readonly Dictionary<BindingLayout, ArgumentPlan> _argumentPlansByInstance = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<(uint, uint, uint, uint, bool), (nint Handle, ulong ResourceId)> _samplers = new();
    private readonly nint[] _stageArgumentBuffers = new nint[StageCount];
    private readonly ulong[] _stageArgumentOffsets = new ulong[StageCount];
    private readonly bool[] _stageArgumentValid = new bool[StageCount];

    // Which prepared stages run a device-address program, whose encoder names the windows
    // the address table spans (NameDeviceAddressWindows).
    private readonly bool[] _stageUsesDeviceAddresses = new bool[StageCount];
    private nint _pushBuffer;
    private ulong _pushOffset;
    private bool _hasPushData;
    private nint[] _usedResources = new nint[64];
    private nuint[] _usedResourceUsage = new nuint[64];
    private int _usedResourceCount;
    private MetalResourcePreparation? _preparation;
    private MetalResourcePreparation? _preparationHandle;
    private int _preparedStageMask;

    public IResourcePreparation BeginPreparation()
    {
        if (_preparation is not null)
        {
            throw Fatal("A Metal resource preparation is already open.");
        }

        // Acquire the decoded frame before any stage selects its movie planes.
        PumpHostMovieFrame();

        // Keep each submission's command buffer bounded so completions and flips are
        // not stuck behind a frame's worth of recording, and so the rings and staging
        // a draw's uploads borrowed return before the next draw borrows more.
        if (_recordedCommands >= 2048)
        {
            FlushCommandBuffer(MetalFlushCause.Budget);
        }
        else
        {
            CommitIfTransientOverBudget();
            // Not inside a render pass: ending one stores and reloads every attachment,
            // and the pass reaches a boundary of its own soon enough.
            if (_renderEncoder == 0)
            {
                CommitCoalescedIfOverdue();
            }
        }

        _ = EnsureCommandBuffer();
        // The handle holds only the host, and a draw disposes it before the next one begins:
        // one serves them all instead of a new object per draw.
        _preparation ??= SharpEmu.ShaderCompiler.WorkerScratch.Enabled
            ? _preparationHandle ??= new MetalResourcePreparation(this)
            : new MetalResourcePreparation(this);
        _preparedStageMask = 0;
        _usedResourceCount = 0;
        _bindlessDeclarations.Clear();
        Array.Clear(_stageArgumentValid);
        Array.Clear(_stageUsesDeviceAddresses);
        _hasPushData = false;
        // The attachment subresources are recorded as the draw acquires them, which
        // happens inside this preparation.
        _attachmentCount = 0;
        _depthAttachmentLevel = 0;
        _depthAttachmentSlice = 0;
        return _preparation;
    }

    private void EndPreparation()
    {
        // A draw that acquired its attachments and then never began its pass leaves no
        // upload waiting past its own preparation.
        _imageCache.DrainDeferredRefreshes();
        // The draw or dispatch is recorded, so the stores it makes into staged copies
        // can be written back at the next encoder boundary.
        _stagedStorageCopies?.Arm();
        _preparation = null;
        _preparedStageMask = 0;
    }

    public void ResetBindings()
    {
        _imageCache.DrainDeferredRefreshes();
        DisarmPendingCopy();
        _pendingGraphicsPipeline = null;
        _pendingComputePipeline = null;
        _vertexBufferCount = 0;
        _indexBuffer = 0;
        _attachmentCount = 0;
        _depthAttachmentLevel = 0;
        _depthAttachmentSlice = 0;
    }

    private static int StageIndexOf(ShaderStageKind kind) => kind switch
    {
        ShaderStageKind.Vertex => StageVertex,
        ShaderStageKind.Pixel => StageFragment,
        ShaderStageKind.Compute => StageCompute,
        _ => throw GpuFatal.Fatal($"The shader stage kind is unknown: stage={kind}."),
    };

    private static ShaderProgramInfo RequireProgram(ShaderStageResources stage)
    {
        var program = stage.Program ?? throw GpuFatal.Fatal("The stage has no program.");
        if (program.Resources is null || program.Bindings is null)
        {
            throw GpuFatal.Fatal(
                $"The stage program has no resource plan: stage={program.Stage} hash=0x{program.Hash:X16}.");
        }

        return program;
    }

    private static ShaderImageShape ShapeOf(ImageResource image) => new(
        Volume: image.Dimension == ImageDimension.Dim3D,
        Arrayed: image.Cube || image.Dimension is ImageDimension.Dim1DArray or ImageDimension.Dim2DArray or
            ImageDimension.Dim2DMsaaArray,
        Cube: image.Cube,
        Storage: image.ResourceClass == MslImageResourceClass.Storage,
        DynamicMip: image.MipMode == ImageMipMode.DynamicStorage,
        NumericClass: image.NumericClass switch
        {
            ImageNumericClass.Uint or ImageNumericClass.Uint64 => TextureNumericClass.Uint,
            ImageNumericClass.Sint => TextureNumericClass.Sint,
            _ => TextureNumericClass.Float,
        },
        OneDimensional: image.Dimension is ImageDimension.Dim1D or ImageDimension.Dim1DArray,
        R128: image.R128,
        Multisampled: image.Dimension is ImageDimension.Dim2DMsaa or ImageDimension.Dim2DMsaaArray,
        DepthCompare: image.DepthCompare,
        Atomic: image.Atomic,
        Atomic64: image.Atomic64);

    // A sampled binding whose T# and shape were resolved before takes the image, request and view it
    // got then while the stamp the image cache gave it holds (MetalImageCache.TryStampTextureLocked),
    // instead of looking them up and acquiring them again. SHARPEMU_METAL_V2_TEXTURE_MEMO=0 turns it off.
    private static readonly bool TextureMemoEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TEXTURE_MEMO") != "0";

    private const int TextureMemoLimit = 16384;

    internal readonly record struct TextureMemoKey(
        uint W0, uint W1, uint W2, uint W3, uint W4, uint W5, uint W6, uint W7, ShaderImageShape Shape);

    internal sealed class TextureMemoEntry
    {
        public TextureMemoKey Key;
        public ResourceSlotIdentifier Image;
        public ImageRequest Request;
        public nint Texture;
        public ulong ResourceId;
        public MetalImageCache.TextureMemoStamp Stamp;
    }

    private readonly Dictionary<TextureMemoKey, TextureMemoEntry> _textureMemo = new();
    private long _textureMemoHits, _textureMemoLate, _textureMemoMisses, _textureMemoStores;

    internal string TakeTextureMemoCounters()
    {
        var text = $"hits={_textureMemoHits} late={_textureMemoLate} misses={_textureMemoMisses} stores={_textureMemoStores} entries={_textureMemo.Count}";
        _textureMemoHits = _textureMemoLate = _textureMemoMisses = _textureMemoStores = 0;
        return text;
    }

    // A sampled binding found in the memo with a stamp that still holds; otherwise notes the key for
    // the full resolution, which reads the owner-list counts over the request before its lookup.
    private bool TryTextureMemo(MetalStageBindings prepared, ImageResource image, uint[] words, int index)
    {
        prepared.MemoHits[index] = null;
        prepared.MemoWanted[index] = false;
        if (!TextureMemoEnabled || image.ResourceClass == MslImageResourceClass.Storage || words.Length < 8)
        {
            return false;
        }

        var key = new TextureMemoKey(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7], ShapeOf(image));
        if (_textureMemo.TryGetValue(key, out var entry) && _imageCache.TextureStampMayHold(in entry.Stamp))
        {
            prepared.Images[index] = entry.Image;
            prepared.Requests[index] = entry.Request;
            prepared.IsStorage[index] = false;
            prepared.Textures[index] = 0;
            prepared.MemoHits[index] = entry;
            return true;
        }

        _textureMemoMisses++;
        prepared.MemoWanted[index] = true;
        prepared.MemoKeys[index] = key;
        return false;
    }

    private void RememberTexture(MetalStageBindings stage, int index, ResourceSlotIdentifier identifier, in ImageRequest request,
        in MetalImageCache.TextureMemoStamp stamp)
    {
        var key = stage.MemoKeys[index];
        if (!_textureMemo.TryGetValue(key, out var entry))
        {
            if (_textureMemo.Count >= TextureMemoLimit)
            {
                _textureMemo.Clear();
            }

            entry = new TextureMemoEntry { Key = key };
            _textureMemo[key] = entry;
        }

        entry.Image = identifier;
        entry.Request = request;
        entry.Texture = stage.Textures[index];
        entry.ResourceId = stage.TextureIds[index];
        entry.Stamp = stamp;
        _textureMemoStores++;
    }

    public IPreparedBindings PrepareBindings(ShaderStageResources stage)
    {
        var program = RequireProgram(stage);
        var resources = program.Resources!;
        var layout = program.Bindings!;
        var snapshot = stage.Resources;
        var info = resources.Info;
        if (snapshot.Images.Length != info.Images.Count || snapshot.Samplers.Length != info.Samplers.Count ||
            snapshot.Buffers.Length != info.Buffers.Count)
        {
            throw Fatal(
                $"The resource snapshot does not match the program: hash=0x{program.Hash:X16} " +
                $"images={snapshot.Images.Length}/{info.Images.Count} samplers={snapshot.Samplers.Length}/{info.Samplers.Count} " +
                $"buffers={snapshot.Buffers.Length}/{info.Buffers.Count}.");
        }

        var stageIndex = StageIndexOf(program.Stage);
        var prepared = _stageBindings[stageIndex];
        prepared.StageResources = stage;
        prepared.Program = program;
        prepared.Committed = false;
        prepared.EnsureCapacity(
            info.Images.Count, info.Samplers.Count, info.Buffers.Count, checked((int)layout.ShaderDataDwordCount));
        _preparedStageMask |= 1 << stageIndex;
        var movie = FindHostMovieBindings(info, snapshot);
        prepared.MovieLuma = movie.Luma;
        prepared.MovieChroma = movie.Chroma;
        for (var index = 0; index < info.Images.Count; index++)
        {
            if (index == movie.Luma || index == movie.Chroma)
            {
                PrepareHostMovieBinding(prepared, info.Images[index], snapshot.Images[index], index);
                continue;
            }

            if (TryTextureMemo(prepared, info.Images[index], snapshot.Images[index], index))
            {
                continue;
            }

            ResolveImageBinding(prepared, info.Images[index], snapshot.Images[index], index);
        }

        // All image requests, including movie replacements and texture memo hits, are current.
        // Classify their sampler pairings once rather than scanning every pair for every sampler.
        MetalSamplerViewClasses.Resolve(info.SampledPairs, prepared.Requests.AsSpan(0, prepared.ImageCount),
            prepared.SamplerViewClasses.AsSpan(0, info.Samplers.Count));
        for (var index = 0; index < info.Samplers.Count; index++)
        {
            prepared.Samplers[index] = ResolveSampler(
                prepared, info.Samplers[index], snapshot.Samplers[index], index,
                prepared.SamplerViewClasses[index] == MetalSamplerViewClasses.AllInteger, out prepared.SamplerIds[index]);
        }

        var shaderData = prepared.ShaderData;
        Array.Clear(shaderData, 0, (int)layout.ShaderDataDwordCount);
        for (var index = 0; index < layout.UserDataRegisters.Count; index++)
        {
            var register = layout.UserDataRegisters[index];
            var userIndex = (int)(register - program.UserDataBase);
            if (register < program.UserDataBase || userIndex >= snapshot.UserData.Length)
            {
                throw Fatal(
                    $"A user register is outside the draw's user data: register={register} base={program.UserDataBase} " +
                    $"count={snapshot.UserData.Length} hash=0x{program.Hash:X16}.");
            }

            shaderData[index] = snapshot.UserData[userIndex];
        }

        if (layout.UsesShaderBase)
        {
            shaderData[layout.ShaderBaseDword] = (uint)stage.ShaderBase;
            shaderData[layout.ShaderBaseDword + 1] = (uint)(stage.ShaderBase >> 32);
        }

        if (layout.BufferWordCount != 0)
        {
            // Portable buffers: the stride and word 3 of each V#, which the program reads
            // instead of a specialization (PortableBufferWord).
            var count = Math.Min(snapshot.Buffers.Length, (int)layout.MemoryOffsetCount);
            for (var index = 0; index < count; index++)
            {
                var (stride, word3) = PortableBufferWord.Pack(snapshot.Buffers[index]);
                var at = (int)layout.FirstBufferWordDword() + index * PortableBufferWord.DwordCount;
                shaderData[at] = stride;
                shaderData[at + 1] = word3;
            }
        }

        if (layout.MipStatisticsCount != 0)
        {
            MipStatistics.WriteImageFields(shaderData, layout, info, snapshot.Images);
        }

        stage.WriteDispatchThreadLimits(shaderData.AsSpan(0, (int)layout.ShaderDataDwordCount));
        stage.WriteCallData(shaderData.AsSpan(0, (int)layout.ShaderDataDwordCount));
        return prepared;
    }

    // Finds the cache image of one shader image descriptor; the view follows in BindResources.
    private void ResolveImageBinding(MetalStageBindings prepared, ImageResource image, uint[] words, int index)
    {
        if (words.Length < 4)
        {
            throw Fatal(
                $"An image descriptor is too short: image={index} words={words.Length} hash=0x{prepared.Program.Hash:X16}.");
        }

        var storage = image.ResourceClass == MslImageResourceClass.Storage;
        var resolution = ImageRequestBuilders.Texture(words, ShapeOf(image));
        var request = resolution.Request;
        if (prepared.MemoWanted[index])
        {
            prepared.MemoPages[index] = _imageCache.RequestPageVersions(in request);
        }

        if (storage && MetalImageCache.TraceStencilTwins && request.Description.PixelFormat == GpuFormat.R8Uint &&
            request.Description.Samples > 1)
        {
            _imageCache.TraceStencilTwin(in request, storage: true);
        }

        if (!storage && _imageCache.TryFindStencilTwin(in request, out var stencilTwin))
        {
            // A multisampled depth image's stencil read as 8-bit integers: its own stencil view.
            prepared.Images[index] = stencilTwin;
            prepared.Requests[index] = request;
            prepared.IsStorage[index] = false;
            prepared.Textures[index] = 0;
            return;
        }

        var imageIdentifier = _imageCache.FindImage(ref request, resolution.ExactFormat);
        resolution = resolution with { Request = request };
        var found = _imageCache.GetImage(imageIdentifier);
        imageIdentifier = ImageRequestBuilders.ValidateTextureOwner(
            found.Description, ResourceSlotIdentifier.Invalid, imageIdentifier, resolution);
        prepared.Images[index] = imageIdentifier;
        prepared.Requests[index] = request;
        prepared.IsStorage[index] = storage;
        prepared.Textures[index] = 0;
        TraceImageBinding(prepared, index, words, in request, found.Description);
    }

    // SHARPEMU_TRACE_IMAGE_BINDINGS=<hash>[,...]: what each image of those programs resolves to, once per
    // image and address, for checking the texture a culling or lighting pass actually samples.
    private static readonly HashSet<ulong> _traceImageBindingHashes = ParseDumpBufferHashesFrom("SHARPEMU_TRACE_IMAGE_BINDINGS");
    private readonly HashSet<(ulong, int, ulong)> _tracedImageBindings = new();
    // SHARPEMU_TRACE_IMAGE_BINDINGS_MAX=<n>: lines before the trace stops (default 256), for tracing a whole corpus.
    private static readonly int _traceImageBindingsMax =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_TRACE_IMAGE_BINDINGS_MAX"), out var max) && max > 0 ? max : 256;

    private void TraceImageBinding(MetalStageBindings prepared, int index, uint[] words, in ImageRequest request, object description)
    {
        if (_traceImageBindingHashes.Count == 0 || !_traceImageBindingHashes.Contains(prepared.Program.Hash) ||
            _tracedImageBindings.Count >= _traceImageBindingsMax || !_tracedImageBindings.Add((prepared.Program.Hash, index, request.Description.Data.Address)))
        {
            return;
        }

        Console.Error.WriteLine(
            $"[METAL-V2][IMAGE-BINDING] flip={_flips} hash=0x{prepared.Program.Hash:X16} image={index} " +
            $"words=[{string.Join(',', words.Select(word => $"{word:X8}"))}] at=0x{request.Description.Data.Address:X} size=0x{request.Description.Data.Size:X} " +
            $"format={request.Description.PixelFormat}/{request.Description.GuestFormat} extent={request.Description.Extent.Width}x{request.Description.Extent.Height} " +
            $"tile={request.Description.TileMode} resources={request.Description.Resources} view={request.View} role={request.Role}");
    }

    private nint ResolveSampler(
        MetalStageBindings prepared,
        SamplerResource sampler,
        uint[] words,
        int index,
        bool integerViews,
        out ulong resourceId)
    {
        if (words.Length < 4)
        {
            throw Fatal(
                $"A sampler descriptor is too short: sampler={index} words={words.Length} hash=0x{prepared.Program.Hash:X16}.");
        }

        Span<uint> native = stackalloc uint[4] { words[0], words[1], words[2], words[3] };
        if (!sampler.DepthCompare)
        {
            native[0] &= ~(0x7u << 12);
        }

        if (sampler.ForcePointFiltering)
        {
            var mipmapped = ((native[2] >> 26) & 0x3u) != 0;
            native[2] &= ~(0xFFu << 20);
            native[2] |= 1u << 24;
            if (mipmapped)
            {
                native[2] |= 1u << 26;
            }
        }

        var descriptor = new SamplerDescriptorWords(native);
        return GetSampler(in descriptor, integerViews, out resourceId);
    }

    // The binding layout gives a dynamic-mip storage image one element per level of its
    // descriptor (BASE_LEVEL..LAST_LEVEL), and the shader picks the element from the mip
    // operand. The first element is the base view; each later one needs the view of its
    // own level, or a downsampler that writes a level and the next through one T# writes
    // both into the base and leaves the next level stale.
    private void BindStorageMipViews(
        MetalStageBindings stage, ImageResource image, ResourceSlotIdentifier identifier, in ImageRequest request, int index)
    {
        var extra = image.MipMode == ImageMipMode.DynamicStorage && stage.IsStorage[index]
            ? (int)image.MipCount - 1
            : 0;
        if (extra <= 0)
        {
            stage.MipTextures[index] = null;
            stage.MipTextureIds[index] = null;
            return;
        }

        var levels = _imageCache.GetImage(identifier).Description.Resources.Levels;
        if (request.View.BaseLevel + image.MipCount > levels)
        {
            throw Fatal(
                $"A storage image's mips run past its levels: image={index} base={request.View.BaseLevel} " +
                $"mips={image.MipCount} levels={levels} hash=0x{stage.Program.Hash:X16}.");
        }

        var textures = stage.MipTextures[index] is { } held && held.Length == extra ? held : new nint[extra];
        var ids = stage.MipTextureIds[index] is { } heldIds && heldIds.Length == extra ? heldIds : new ulong[extra];
        for (var mip = 1; mip <= extra; mip++)
        {
            var mipRequest = request with
            {
                View = request.View with { BaseLevel = request.View.BaseLevel + (uint)mip, LevelCount = 1 },
            };
            textures[mip - 1] = _imageCache.AcquireTextureView(identifier, mipRequest, out ids[mip - 1]);
            NotePendingInputDump(textures[mip - 1], identifier, index, ImageRole.StorageImage);
        }

        stage.MipTextures[index] = textures;
        stage.MipTextureIds[index] = ids;
    }

    public void PrepareDeviceAddresses()
    {
        // Guest memory is the GPU's memory, so a device-address program needs no page
        // table upload: the pointers the shader reads already address the windows.
        // BeforeDeviceAddressPreparation has already given single-copy chunks back; under the
        // global fallback this only counts a program that reached here without it (the
        // evaluated ranges are checked again as they are bound, BindDeviceAddressRanges).
        if (_singleCopy is { DeviceAddressSeen: true } singleCopy)
        {
            singleCopy.NoteDeviceAddressUse(late: true);
        }
    }

    private Func<ulong, ulong, ulong>? _mappedDeviceAddressBytes;
    private ulong _singleCopyDeviceAddressSerial;

    // The mapped bytes of a device-address range from its base, 0 when its first byte is not.
    private ulong MappedDeviceAddressBytes(ulong address, ulong size) =>
        address == 0 || size == 0 || size > ulong.MaxValue - address || !_memory.CanRead(address, 1)
            ? 0
            : ClampMappedSize(address, size);

    private static long DeviceAddressMappingVersion() =>
        SharpEmu.HLE.GpuMemory.GuestGpuMemoryHook.Current?.SpanVersion ?? 0;

    /// <summary>
    /// A device-address program can reach any mapped guest page through the page table, past
    /// every window resolve. Before one is prepared, single-copy textures pin and give back
    /// every chunk the ranges evaluated for its handles cover (MetalSingleCopy.BeforeDeviceAddressProgram),
    /// a program whose reach cannot be bounded takes the global fallback (every released chunk
    /// back, nothing released from then on), and the trap hits reported so far are taken. A
    /// command buffer whose windows still map a hole over restored bytes ends here, between
    /// preparations, so the program runs in the next one and reads the guest's bytes.
    /// </summary>
    void IRenderHost.BeforeDeviceAddressPreparation(ReadOnlySpan<ShaderStageResources> stages)
    {
        if (_singleCopy is not { } singleCopy)
        {
            return;
        }

        singleCopy.KillSwitch ??= MetalSingleCopyKillSwitch.Open(TitleProfileDirectory());
        if (_singleCopyDeviceAddressSerial != _submissionSerial)
        {
            _singleCopyDeviceAddressSerial = _submissionSerial;
            singleCopy.NoteDeviceAddressSubmission();
        }

        TakeSingleCopyTrapHits();
        var mappingVersion = DeviceAddressMappingVersion();
        _mappedDeviceAddressBytes ??= MappedDeviceAddressBytes;
        foreach (var stage in stages)
        {
            if (stage.Program is { UsesDeviceAddresses: true } program)
            {
                singleCopy.BeforeDeviceAddressProgram(
                    program.WritesDeviceAddresses, stage.Resources.DeviceAddressRanges, mappingVersion,
                    _mappedDeviceAddressBytes, late: _preparation is not null);
            }
        }

        if (_preparation is null && _commandBuffer != 0 && _guestMemory.HasStaleSingleCopyWindows)
        {
            FlushCommandBuffer(MetalFlushCause.Other);
        }
    }

    private readonly List<ulong> _singleCopyTrapPages = new();

    // The trap hits submissions reported (resource_prelude.msl): their chunks back and pinned.
    private void TakeSingleCopyTrapHits()
    {
        if (_singleCopy is not { } singleCopy || _reach is not { } reach)
        {
            return;
        }

        _singleCopyTrapPages.Clear();
        var hits = reach.TakeTrapHits(_singleCopyTrapPages);
        if (hits != 0 || _singleCopyTrapPages.Count != 0)
        {
            singleCopy.OnTrapHits(hits, _singleCopyTrapPages);
        }
    }

    // The memoised sampled bindings from `first` on that still hold, confirmed under one hold of the
    // image cache lock (MetalImageCache.HoldForReacquire) rather than one hold each. Stops at the
    // first binding that is a movie plane, has no memo entry or whose stamp no longer holds, so every
    // binding is still looked at in order, after the ones before it have taken effect. Returns how
    // many it bound; zero means the binding at `first` is stale.
    private int ReacquireMemoRun(MetalStageBindings stage, int first)
    {
        var count = 0;
        using (var held = _imageCache.HoldForReacquire())
        {
            for (var index = first; index < stage.ImageCount; index++)
            {
                if (index == stage.MovieLuma || index == stage.MovieChroma ||
                    stage.MemoHits[index] is not { } hit ||
                    !_imageCache.TryReacquireTextureViewHeld(hit.Image, in hit.Request, in hit.Stamp))
                {
                    break;
                }

                stage.Textures[index] = hit.Texture;
                stage.TextureIds[index] = hit.ResourceId;
                stage.MipTextures[index] = null;
                stage.MipTextureIds[index] = null;
                count++;
            }
        }

        for (var index = first; index < first + count; index++)
        {
            var hit = stage.MemoHits[index]!;
            NotePendingInputDump(hit.Texture, hit.Image, index, ImageRole.Texture);
        }

        _textureMemoHits += count;
        return count;
    }

    public void BindResources(IPreparedBindings prepared)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.ResourceBinding);
        var stage = (MetalStageBindings)prepared;
        var info = stage.Program.Resources!.Info;
        var snapshot = stage.StageResources.Resources;
        for (var index = 0; index < stage.ImageCount; index++)
        {
            if (index == stage.MovieLuma || index == stage.MovieChroma)
            {
                stage.Textures[index] = AcquireHostMovieView(index == stage.MovieLuma ? 0 : 1);
                stage.TextureIds[index] = MetalNative.SendGpuResourceId(stage.Textures[index], SelGpuResourceId);
                continue;
            }

            if (stage.MemoHits[index] is { } hit)
            {
                var reacquired = ReacquireMemoRun(stage, index);
                if (reacquired > 0)
                {
                    index += reacquired - 1;
                    continue;
                }

                // The stamp stopped holding after the preparation found it: the image and request
                // it gave are looked up again, then acquired and memoised as on a miss.
                stage.MemoHits[index] = null;
                stage.MemoWanted[index] = true;
                stage.MemoKeys[index] = hit.Key;
                ResolveImageBinding(stage, info.Images[index], snapshot.Images[index], index);
                _textureMemoLate++;
            }

            var request = stage.Requests[index];
            var identifier = stage.Images[index];
            using (var resolveScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingImageResolve))
            {
                if (!_imageCache.TryGetImage(identifier, out _))
                {
                    ResolveImageBinding(stage, info.Images[index], snapshot.Images[index], index);
                    request = stage.Requests[index];
                    identifier = stage.Images[index];
                }
            }

            var role = stage.IsStorage[index] ? ImageRole.StorageImage : ImageRole.Texture;
            if (stage.IsStorage[index])
            {
                request = request with { View = request.View with { LevelCount = 1 } };
                stage.Requests[index] = request;
            }

            using (var acquireScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingTextureAcquire))
            {
                if (stage.MemoWanted[index] && !stage.IsStorage[index])
                {
                    stage.Textures[index] = _imageCache.AcquireTextureViewForMemo(
                        identifier, request with { Role = role }, stage.MemoPages[index], out stage.TextureIds[index], out var stamp);
                    if (stamp is { } held)
                    {
                        RememberTexture(stage, index, identifier, in request, in held);
                    }
                }
                else
                {
                    stage.Textures[index] = _imageCache.AcquireTextureView(
                        identifier, request with { Role = role }, out stage.TextureIds[index]);
                }

                NotePendingInputDump(stage.Textures[index], identifier, index, role, stage.Program.Stage == ShaderStageKind.Compute);
                BindStorageMipViews(stage, info.Images[index], identifier, request with { Role = role }, index);
            }
        }

        for (var index = 0; index < stage.BufferCount; index++)
        {
            var words = snapshot.Buffers[index];
            if (words.Length < 4)
            {
                throw Fatal(
                    $"A buffer descriptor is too short: buffer={index} words={words.Length} hash=0x{stage.Program.Hash:X16}.");
            }

            var descriptor = BufferDescriptorWords.From(words);
            var requested = descriptor.Footprint() ?? throw Fatal(
                $"A storage buffer descriptor footprint overflows: buffer={index} stride={descriptor.Stride} " +
                $"records={descriptor.RecordCount} hash=0x{stage.Program.Hash:X16}.");
            if (descriptor.Address == 0 || requested == 0)
            {
                stage.Buffers[index] = default;
                stage.BufferLengths[index] = 0;
                continue;
            }

            // A range mapped whole - nearly every binding - is answered by one query; ClampMappedSize
            // asks again from its first byte only for a range that runs past its mapping.
            bool readable;
            bool mappedWhole;
            using (var checkScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingBufferCheck))
            {
                var inRange = requested <= ulong.MaxValue - descriptor.Address;
                mappedWhole = inRange && _memory.CanRead(descriptor.Address, requested);
                readable = mappedWhole || (inRange && _memory.CanRead(descriptor.Address, 1));
            }

            if (!readable)
            {
                var resource = info.Buffers[index];
                throw Fatal(
                    $"The buffer range starts in unmapped memory: address=0x{descriptor.Address:X16} size=0x{requested:X16} " +
                    $"stage={stage.Program.Stage} hash=0x{stage.Program.Hash:X16} shader=0x{stage.StageResources.ShaderBase:X16} " +
                    $"buffer={index} source={resource.Source} pc=0x{resource.FirstUsePc:X} written={resource.Written} " +
                    $"descriptor=[{string.Join(',', words.ToArray().Select(word => $"{word:X8}"))}] " +
                    $"user_data=[{string.Join(',', snapshot.UserData.ToArray().Select(word => $"{word:X8}"))}].");
            }

            var size = mappedWhole ? requested : ClampMappedSize(descriptor.Address, requested);
            var written = info.Buffers[index].Written;
            FlushGpuWrites(descriptor.Address, size, MetalWritebackCause.StorageDescriptor, written);
            var range = ResolveStorageBuffer(descriptor.Address, size, written);
            TraceBufferUser(stage, $"storage[{index}]{(written ? " written" : string.Empty)}", descriptor.Address, size);
            TraceRecordChain(stage, index, descriptor.Address, size);
            TraceBufferAtCompletion(stage, index, descriptor.Address, size);
            if (written)
            {
                using var writtenScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingWrittenBuffer);
                // The text is built only when the trace prints it: this runs per written buffer per draw.
                if (MetalImageCache.TracesDccWriter(descriptor.Address, size))
                {
                    _imageCache.TraceDccWriter(
                        descriptor.Address, size,
                        $"program=0x{stage.Program.Hash:X16} buffer={index} stride={descriptor.Stride} records={descriptor.RecordCount} " +
                        $"format={descriptor.Format} buffers={stage.BufferCount} user_data=[{string.Join(',', snapshot.UserData.ToArray().Select(word => $"{word:X8}"))}]");
                }

                using (var invalidateScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingWrittenInvalidate))
                {
                    _imageCache.InvalidateMemoryFromGpu(descriptor.Address, size);
                }

                using (var noteScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingWrittenNote))
                {
                    NoteGuestMemoryWrite(descriptor.Address, size, MetalGpuWriteOrigin.StorageBuffer);
                }
            }

            using var reportScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingReport);
            _slotTrap?.NoteBinding(
                $"storage[{index}] hash=0x{stage.Program.Hash:X16}{(written ? " written" : string.Empty)}",
                descriptor.Address, size, (ulong)range.Buffer, range.Offset, range.GpuAddress);
            stage.Buffers[index] = range;
            stage.BufferLengths[index] = BoundBufferBytes(size, range);
            ReportBufferBinding(stage, index, in descriptor, requested, size, range);
        }

        using (var deviceScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.BindingDeviceAddresses))
        {
            BindDeviceAddressRanges(stage, info, snapshot);
        }
        if (snapshot.BindlessImages.Length != 0 || stage.Program.Bindings!.BindlessImageTableCount != 0)
        {
            BindBindlessImages(stage, info, snapshot);
        }
    }

    /// <summary>
    /// The guest ranges a device-address program reaches without a descriptor binding:
    /// a flat or global access, or a V# the shader loads out of guest memory when it
    /// runs. The resource planner evaluates each handle's base and extent for the draw,
    /// which is what lets these be named at all.
    ///
    /// They are not bindings here — guest memory is the GPU's memory, so the pointer in
    /// the page table already addresses the window — but they carry the same obligations
    /// a storage binding does, and without them the backend has a hole its own rules
    /// forbid: an image that owns the bytes never publishes them before the shader reads
    /// them, a shader store never invalidates the image or records its range, and a
    /// later command-processor poll or host read of those bytes does not wait for the
    /// submission still writing them. The twin of the Vulkan presenter's
    /// <c>BindDeviceAddressRanges</c>.
    /// </summary>
    // SHARPEMU_METAL_V2_TRACE_BIG_DEVICE_RANGES=<MB> (diagnostic): each program's device-address range of at least that
    // size, once per program and handle (a range over a GPU-written image writes the whole image back first).
    private static readonly ulong _traceBigDeviceRangeBytes =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRACE_BIG_DEVICE_RANGES"), out var bigRangeMegabytes)
            ? bigRangeMegabytes << 20 : 0;

    private readonly HashSet<(ulong, uint)> _tracedBigDeviceRanges = new();

    // SHARPEMU_METAL_V2_TRACE_DEVICE_RANGE_OVERLAP=<hex base>:<hex size>: with the above set, the ranges that overlap these
    // bytes instead, of any size.
    private static readonly (ulong Base, ulong Size) _traceDeviceRangeOverlap = ParseRangeOverlap(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRACE_DEVICE_RANGE_OVERLAP"));

    private static (ulong, ulong) ParseRangeOverlap(string? text)
    {
        var parts = (text ?? string.Empty).Split(':');
        return parts.Length == 2 &&
            ulong.TryParse(parts[0], System.Globalization.NumberStyles.HexNumber, null, out var start) &&
            ulong.TryParse(parts[1], System.Globalization.NumberStyles.HexNumber, null, out var length)
            ? (start, length) : (0, 0);
    }

    private void BindDeviceAddressRanges(
        MetalStageBindings stage, ShaderResourceInfo info, ResourceSnapshot snapshot)
    {
        var unplanned = false;
        var declaredWrite = false;
        if (snapshot.DeviceAddressRanges.Length != 0 && _singleCopy is { } singleCopy)
        {
            // Counted, and given back, only if the program got past BeforeDeviceAddressPreparation:
            // ranges it pinned are known and cost a lookup each.
            _mappedDeviceAddressBytes ??= MappedDeviceAddressBytes;
            singleCopy.BeforeDeviceAddressProgram(
                info.WritesDeviceAddresses, snapshot.DeviceAddressRanges, DeviceAddressMappingVersion(),
                _mappedDeviceAddressBytes, late: true);
        }

        foreach (var range in snapshot.DeviceAddressRanges)
        {
            if (!range.Planned)
            {
                // Nothing evaluated this handle, so the store's destination is unknown, and
                // so is what it reads: every copy still owed lands first.
                OrderAfterDeferredCopies(0, ulong.MaxValue);
                unplanned |= range.Written;
                continue;
            }

            // A pointer to an unmapped page resolves to null in the shader, which drops
            // the access; there is nothing to publish, invalidate or wait for.
            if (range.Size == 0 || range.Base == 0 || !_memory.CanRead(range.Base, 1))
            {
                continue;
            }

            var size = ClampMappedSize(range.Base, range.Size);
            TraceBufferUser(stage, $"device_address h{range.Handle}{(range.Written ? " written" : string.Empty)}", range.Base, size);
            if (_traceBigDeviceRangeBytes != 0 &&
                (_traceDeviceRangeOverlap.Size != 0
                    ? range.Base < _traceDeviceRangeOverlap.Base + _traceDeviceRangeOverlap.Size && _traceDeviceRangeOverlap.Base < range.Base + size
                    : size >= _traceBigDeviceRangeBytes) &&
                _tracedBigDeviceRanges.Count < 64 && _tracedBigDeviceRanges.Add((stage.Program.Hash, range.Handle)))
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][BIG-DEVICE-RANGE] stage={stage.Program.Stage} hash=0x{stage.Program.Hash:X16} h{range.Handle} " +
                    $"base=0x{range.Base:X} size=0x{size:X} planned_size=0x{range.Size:X}{(range.Written ? " written" : string.Empty)}");
            }
            OrderAfterDeferredCopies(range.Base, size);
            OrderAfterStagedStores(range.Base, size);
            var bigWritebacks = _imageCache.BigWritebacks;
            FlushGpuWrites(range.Base, size, MetalWritebackCause.StorageDescriptor, range.Written);
            if (_imageCache.BigWritebacks != bigWritebacks && _tracedBigDeviceRanges.Count < 64 &&
                _tracedBigDeviceRanges.Add((stage.Program.Hash, range.Handle | 0x8000_0000u)))
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][BIG-WRITEBACK-RANGE] stage={stage.Program.Stage} hash=0x{stage.Program.Hash:X16} h{range.Handle} " +
                    $"base=0x{range.Base:X} size=0x{size:X}{(range.Written ? " written" : string.Empty)} ranges={snapshot.DeviceAddressRanges.Length}");
            }
            if (!range.Written)
            {
                continue;
            }

            _imageCache.InvalidateMemoryFromGpu(range.Base, size);
            NoteGuestMemoryWrite(range.Base, size, MetalGpuWriteOrigin.DeviceAddress);
            _slotTrap?.NoteBinding(
                $"device_address handle={range.Handle} hash=0x{stage.Program.Hash:X16} written",
                range.Base, size, 0, 0, 0);
            declaredWrite = true;
        }

        if (!unplanned && (!info.WritesDeviceAddresses || declaredWrite))
        {
            return;
        }

        // The program stores through the page table and no evaluated range says where.
        // Conservative is the only correct answer: every later read of any guest address
        // waits for this submission.
        _unboundedProducer = stage.Program.Hash;
        NoteUnboundedGuestMemoryWrite();
        _deviceAddressWrites++;
        MetalV2Diagnostics.WarnOnce(
            $"device-address-unbounded-{stage.Program.Hash:X16}",
            $"A shader stores through the device-address page table with no evaluated range, so every " +
            $"later guest read waits for its submission: hash=0x{stage.Program.Hash:X16} " +
            $"unplanned={unplanned} ranges={snapshot.DeviceAddressRanges.Length} [" +
            string.Join(", ", snapshot.DeviceAddressRanges.Select(static range =>
                $"h{range.Handle} {(range.Planned ? "planned" : "unplanned")} {(range.Written ? "written" : "read")} " +
                $"0x{range.Base:X}+0x{range.Size:X}")) + "].");
    }

    // SHARPEMU_TRACE_BUFFER_USER=<hex>: each program that binds a buffer or device range over that address,
    // once per program and binding, for naming the shader behind a GPU-written table the CPU reads back.
    private static readonly ulong _traceBufferUserAddress =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_TRACE_BUFFER_USER"),
            System.Globalization.NumberStyles.HexNumber, null, out var traced) ? traced : 0;
    private readonly HashSet<(ulong, string)> _tracedBufferUsers = new();

    // SHARPEMU_TRACE_RECORD_CHAIN=<hash>:<buffer>:<stride>:<count>:<pointer offset>:<target offset> (hex): for the
    // first flips that bind it, each record's pointer and the eight dwords at pointer + target offset, for reading
    // the descriptor a program loads through a per-record object it selects at run time.
    private static readonly ulong[] _traceRecordChain =
        (Environment.GetEnvironmentVariable("SHARPEMU_TRACE_RECORD_CHAIN") ?? string.Empty)
            .Split(':', StringSplitOptions.RemoveEmptyEntries)
            .Select(part => ulong.TryParse(part, System.Globalization.NumberStyles.HexNumber, null, out var value) ? value : 0)
            .ToArray();
    private int _tracedRecordChains;

    private void TraceRecordChain(MetalStageBindings stage, int index, ulong address, ulong size)
    {
        if (_traceRecordChain.Length != 6 || stage.Program.Hash != _traceRecordChain[0] || (ulong)index != _traceRecordChain[1] ||
            _tracedRecordChains >= 6)
        {
            return;
        }

        _tracedRecordChains++;
        var text = new System.Text.StringBuilder($"[METAL-V2][RECORD-CHAIN] flip={_flips} hash=0x{stage.Program.Hash:X16} buffer={index} base=0x{address:X} size=0x{size:X}");
        Span<byte> pointer = stackalloc byte[8];
        Span<byte> target = stackalloc byte[32];
        for (ulong record = 0; record < _traceRecordChain[3] && (record + 1) * _traceRecordChain[2] <= size; record++)
        {
            var at = address + record * _traceRecordChain[2] + _traceRecordChain[4];
            var value = _guestMemory.TryPeek(at, pointer) ? BitConverter.ToUInt64(pointer) : 0;
            text.Append($"\n  record {record} ptr=0x{value:X}");
            if (value != 0 && _guestMemory.TryPeek(value + _traceRecordChain[5], target))
            {
                text.Append(" words=[");
                for (var word = 0; word < 8; word++)
                {
                    text.Append($"{BitConverter.ToUInt32(target[(word * 4)..]):X8}{(word < 7 ? "," : "]")}");
                }
            }
        }

        Console.Error.WriteLine(text.ToString());
    }

    // SHARPEMU_TRACE_BUFFER_AT_COMPLETION=<hash>:<buffer>:<bytes>:<dir> (bytes hex): once the submission that binds it
    // completes, that buffer's first bytes as guest memory then holds them, to <dir>/<hash>-buf<n>-flip<f>-<k>.bin -
    // what a GPU pass left in a list another pass reads, which a bind-time dump cannot show.
    private static readonly string[] _traceBufferAtCompletion =
        (Environment.GetEnvironmentVariable("SHARPEMU_TRACE_BUFFER_AT_COMPLETION") ?? string.Empty).Split(':');
    private int _tracedBuffersAtCompletion;

    private void TraceBufferAtCompletion(MetalStageBindings stage, int index, ulong address, ulong size)
    {
        if (_traceBufferAtCompletion.Length != 4 || _tracedBuffersAtCompletion >= 400 ||
            !ulong.TryParse(_traceBufferAtCompletion[0], System.Globalization.NumberStyles.HexNumber, null, out var hash) ||
            hash != stage.Program.Hash || _traceBufferAtCompletion[1] != index.ToString(System.Globalization.CultureInfo.InvariantCulture) ||
            !ulong.TryParse(_traceBufferAtCompletion[2], System.Globalization.NumberStyles.HexNumber, null, out var bytes))
        {
            return;
        }

        var serial = _tracedBuffersAtCompletion++;
        var flip = Interlocked.Read(ref _flips);
        var directory = _traceBufferAtCompletion[3];
        var length = (int)Math.Min(bytes, size);
        _completions.Enqueue(SubmissionSerial, () =>
        {
            var contents = new byte[length];
            if (_guestMemory.TryPeek(address, contents))
            {
                Directory.CreateDirectory(directory);
                File.WriteAllBytes(Path.Combine(directory, $"{hash:X16}-buf{index}-flip{flip}-{serial}.bin"), contents);
            }
        });
    }

    private void TraceBufferUser(MetalStageBindings stage, string what, ulong address, ulong size)
    {
        if (_traceBufferUserAddress == 0 || _traceBufferUserAddress < address || _traceBufferUserAddress - address >= size ||
            _tracedBufferUsers.Count >= 256 || !_tracedBufferUsers.Add((stage.Program.Hash, what)))
        {
            return;
        }

        Console.Error.WriteLine(
            $"[METAL-V2][BUFFER-USER] flip={_flips} stage={stage.Program.Stage} hash=0x{stage.Program.Hash:X16} {what} " +
            $"base=0x{address:X} size=0x{size:X} offset=0x{_traceBufferUserAddress - address:X}");
    }

    // SHARPEMU_METAL_V2_DUMP_BUFFERS=<hash>[,<hash>...]: what one program's scalar and
    // storage buffers actually resolve to. A translated loop whose trip count comes out
    // of a constant buffer is only as right as the descriptor behind it, and the listing
    // alone cannot say whether the dword it loads is the guest's count or a stale window.
    private static readonly HashSet<ulong> _dumpBufferHashes = ParseDumpBufferHashes();
    private readonly Dictionary<(ulong Hash, int Buffer), int> _dumpedBufferCounts = new();
    // SHARPEMU_METAL_V2_DUMP_BUFFER_COUNT=<n>: reports per binding, for a program that
    // draws a whole HUD one quad batch after another.
    private static readonly int MaxBufferDumpsPerBinding =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_BUFFER_COUNT"), out var dumps) && dumps > 0
            ? dumps
            : 4;

    // How much of each buffer to print. The head alone says nothing about the matrix a
    // vertex program loads from the middle of a constant buffer.
    private static readonly int _dumpBufferBytes =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_BUFFER_BYTES"), out var bytes) &&
        bytes is > 0 and <= 4096
            ? (bytes + 3) & ~3
            : 32;

    // SHARPEMU_METAL_V2_DUMP_BUFFER_BIN=<dir>: also writes each reported binding whole (up to
    // 64 MiB, as the CPU sees guest memory) to <dir>/<hash>-buf<index>-<n>.bin.
    private static readonly string? _dumpBufferBinDirectory =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_BUFFER_BIN");

    private static HashSet<ulong> ParseDumpBufferHashes() => ParseDumpBufferHashesFrom("SHARPEMU_METAL_V2_DUMP_BUFFERS");

    // A comma-separated list of program hashes (hex, 0x optional) from an environment variable.
    private static HashSet<ulong> ParseDumpBufferHashesFrom(string variable)
    {
        var value = Environment.GetEnvironmentVariable(variable);
        var hashes = new HashSet<ulong>();
        if (string.IsNullOrWhiteSpace(value))
        {
            return hashes;
        }

        foreach (var part in value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var text = part.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ? part[2..] : part;
            if (ulong.TryParse(text, System.Globalization.NumberStyles.HexNumber,
                    System.Globalization.CultureInfo.InvariantCulture, out var hash))
            {
                hashes.Add(hash);
            }
        }

        return hashes;
    }

    /// <summary>
    /// The byte count the shader bounds a storage binding by: what the bound buffer
    /// holds, never more. A range too large for one window binds only its window prefix
    /// (and one too large to stage only the ring's cap), and the shader checks its
    /// accesses against this count, not the bound buffer's end; with the guest's whole
    /// size an access past the prefix would read and write whatever follows the window
    /// in the GPU address space. Past it a load reads zero and a store is dropped.
    /// </summary>
    internal static uint BoundBufferBytes(ulong size, in MetalGuestRange range) =>
        (uint)Math.Min(Math.Min(size, range.Size), uint.MaxValue);

    private void ReportBufferBinding(
        MetalStageBindings stage,
        int index,
        in BufferDescriptorWords descriptor,
        ulong requested,
        ulong size,
        in MetalGuestRange range)
    {
        var hash = stage.Program.Hash;
        // SHARPEMU_METAL_V2_DUMP_AFTER_SECONDS holds these back too, so a program the
        // menus also run reports the level's bindings rather than the title screen's.
        if (_dumpBufferHashes.Count == 0 || !_dumpBufferHashes.Contains(hash) || !DumpWindowOpen)
        {
            return;
        }

        // Under a dump trigger, the bindings of the dumped frame, beside its snapshots.
        if (_dumpTriggerPath is not null && !_dumpThisFlip)
        {
            return;
        }

        var key = (hash, index);
        _dumpedBufferCounts.TryGetValue(key, out var seen);
        if (seen >= MaxBufferDumpsPerBinding)
        {
            return;
        }

        _dumpedBufferCounts[key] = seen + 1;
        if (_dumpBufferBinDirectory is not null)
        {
            var whole = new byte[(int)Math.Min(size, 64UL << 20)];
            if (_guestMemory.TryPeek(descriptor.Address, whole))
            {
                Directory.CreateDirectory(_dumpBufferBinDirectory);
                File.WriteAllBytes(Path.Combine(_dumpBufferBinDirectory, $"{hash:X16}-buf{index}-{seen}.bin"), whole);
            }
        }

        var count = (int)Math.Min((ulong)_dumpBufferBytes, Math.Max(size, 4));
        var host = new byte[count];
        var read = _guestMemory.TryPeek(descriptor.Address, host);
        var windowed = _guestMemory.ResolveWritable(descriptor.Address, size).IsValid;

        // What the shader will actually read: the bytes behind the pointer the argument
        // buffer carries. A window aliases guest memory, so the two agree; an upload ring
        // holds a copy taken when the binding was prepared, and a guest write after that
        // never reaches it. Printing both is the only way to tell those apart.
        var device = ReadDeviceBytes(range, count);

        Console.Error.WriteLine(string.Format(
            System.Globalization.CultureInfo.InvariantCulture,
            "[METAL-V2][BUF] flip={17} pending={18} hash=0x{0:X16} stage={1} buf={2} addr=0x{3:X16} stride={4} records={5} " +
            "fmt={6} req=0x{7:X} size=0x{8:X} window={9} resolved={10} read={11} " +
            "buffer=0x{12:X}+0x{13:X} same={14}\n  cpu={15}\n  gpu={16}",
            hash,
            stage.Program.Stage,
            index,
            descriptor.Address,
            descriptor.Stride,
            descriptor.RecordCount,
            descriptor.Format,
            requested,
            size,
            windowed,
            range.IsValid,
            read,
            range.Buffer,
            range.Offset,
            device is not null && host.AsSpan().SequenceEqual(device),
            DescribeDwords(host),
            device is null ? "private" : DescribeDwords(device),
            Interlocked.Read(ref _flips),
            PendingSerialHinted(descriptor.Address, size)));
    }

    /// <summary>The first bytes the GPU sees behind a resolved range, or null when the
    /// buffer has no host-visible contents.</summary>
    private static unsafe byte[]? ReadDeviceBytes(in MetalGuestRange range, int count)
    {
        if (!range.IsValid)
        {
            return null;
        }

        var contents = MetalNative.Send(range.Buffer, MetalNative.Selector("contents"));
        if (contents == 0)
        {
            return null;
        }

        return new ReadOnlySpan<byte>((byte*)contents + range.Offset, count).ToArray();
    }

    private static string DescribeDwords(ReadOnlySpan<byte> bytes)
    {
        var dwords = new System.Text.StringBuilder();
        for (var offset = 0; offset + 3 < bytes.Length; offset += sizeof(uint))
        {
            if (offset % 64 == 0)
            {
                dwords.Append(System.Globalization.CultureInfo.InvariantCulture, $"\n    +{offset:X3}:");
            }

            dwords.Append(
                System.Globalization.CultureInfo.InvariantCulture,
                $" {BinaryPrimitives.ReadUInt32LittleEndian(bytes[offset..]):X8}");
        }

        return dwords.ToString();
    }

    public void CommitBindings(GpuBindPoint bindPoint, in PipelineHandle pipeline, ReadOnlySpan<IPreparedBindings> stages)
    {
        _ = RequirePipelineEntry(in pipeline);
        if (bindPoint == GpuBindPoint.Graphics)
        {
            NoteJoinStages(stages);
        }

        Span<uint> pushData = stackalloc uint[(int)PushData.DwordCount];
        pushData.Clear();
        var hasPushData = false;
        foreach (var prepared in stages)
        {
            var stage = (MetalStageBindings)prepared;
            var stageIndex = StageIndexOf(stage.Program.Stage);
            if ((bindPoint == GpuBindPoint.Graphics && stageIndex == StageCompute) ||
                (bindPoint == GpuBindPoint.Compute && stageIndex != StageCompute))
            {
                throw Fatal($"A stage does not belong to the bind point: stage={stage.Program.Stage} bindPoint={bindPoint}.");
            }

            WriteArgumentBuffer(stage, stageIndex);
            var layout = stage.Program.Bindings!;
            if (layout.UsesPushData)
            {
                for (var index = 0; index < layout.ShaderDataDwordCount; index++)
                {
                    pushData[(int)layout.PushDataStartDword + index] = stage.ShaderData[index];
                }

                hasPushData = true;
            }

            stage.Committed = true;
        }

        if (hasPushData)
        {
            var pushBytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(pushData);
            MetalGuestRange range;
            if (ArgumentBufferReuse && _pushRegions.TryFind(pushBytes, _submissionSerial, out var found))
            {
                range = found;
            }
            else
            {
                range = _guestMemory.Allocate(PushData.ByteSize, 256, out var span);
                pushBytes.CopyTo(span);
                if (ArgumentBufferReuse)
                {
                    _pushRegions.Add(pushBytes, _submissionSerial, in range);
                }
            }

            _pushBuffer = range.Buffer;
            _pushOffset = range.Offset;
            _hasPushData = true;
        }
    }

    // The stage's Tier 2 argument buffer, mirroring the binding layout in descriptor order.
    private void WriteArgumentBuffer(MetalStageBindings stage, int stageIndex)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.ArgumentBufferWrite);
        var layout = stage.Program.Bindings!;
        var plan = ArgumentPlanOf(layout);
        var argumentLayout = plan.Layout;
        if (argumentLayout.Fields.Count == 0)
        {
            _stageArgumentValid[stageIndex] = false;
            return;
        }

        // Filled in host scratch first: a draw whose argument buffer comes out the same as one this
        // command buffer already wrote binds that range (ArgumentBufferReuse) instead of reserving,
        // clearing and filling a new one. Every side effect of the fill (the textures it names, the
        // tables it uploads, the device-address use) still happens on every draw.
        var argumentBytes = (int)Math.Max(argumentLayout.ByteSize, 16u);
        if (_argumentScratch.Length < argumentBytes)
        {
            _argumentScratch = new byte[Math.Max(argumentBytes, _argumentScratch.Length * 2)];
        }

        var argument = _argumentScratch.AsSpan(0, argumentBytes);
        argument.Clear();
        var imageCursor = 0;
        var fields = argumentLayout.Fields;
        for (var fieldIndex = 0; fieldIndex < fields.Count; fieldIndex++)
        {
            var field = fields[fieldIndex];
            var span = argument.Slice((int)field.ByteOffset, (int)field.ByteSize);
            switch (field.FieldKind)
            {
                case MslArgumentFieldKind.BufferPointers:
                    for (var index = 0; index < field.Count && index < stage.BufferCount; index++)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(
                            span[(index * sizeof(ulong))..], stage.Buffers[index].GpuAddress);
                    }

                    break;
                case MslArgumentFieldKind.BufferByteCounts:
                    for (var index = 0; index < field.Count && index < stage.BufferCount; index++)
                    {
                        BinaryPrimitives.WriteUInt32LittleEndian(
                            span[(index * sizeof(uint))..], stage.BufferLengths[index]);
                    }

                    break;
                case MslArgumentFieldKind.Textures:
                    WriteTextureField(
                        stage, field, plan.FieldResources[fieldIndex], plan.FieldMips[fieldIndex], span, ref imageCursor, stageIndex);
                    break;
                case MslArgumentFieldKind.Samplers:
                    // The layout lists only canonical samplers (equal descriptors share one
                    // element), so each element names the resource it binds.
                    var samplerResources = plan.FieldResources[fieldIndex];
                    for (var index = 0; index < field.Count; index++)
                    {
                        var resource = samplerResources is not null && index < samplerResources.Length
                            ? samplerResources[index]
                            : index;
                        if (resource >= stage.SamplerCount)
                        {
                            continue;
                        }

                        var sampler = stage.Samplers[resource];
                        if (sampler != 0)
                        {
                            BinaryPrimitives.WriteUInt64LittleEndian(
                                span[(index * sizeof(ulong))..],
                                stage.SamplerIds[resource]);
                        }
                    }

                    break;
                case MslArgumentFieldKind.Pointer:
                    BinaryPrimitives.WriteUInt64LittleEndian(span, PointerFieldValue(stage, field.Kind));
                    _stageUsesDeviceAddresses[stageIndex] |= field.Kind == DescriptorBindingKind.DeviceAddressPageTable;
                    break;
                case MslArgumentFieldKind.StepReport:
                    BinaryPrimitives.WriteUInt64LittleEndian(
                        span, StepReportAddress(stage.Program.Hash, StepReportStageName(stageIndex)));
                    break;
                case MslArgumentFieldKind.PairLocks:
                    // One table for every program, resident with the windows, so two programs
                    // reaching the same guest qword take the same lock.
                    BinaryPrimitives.WriteUInt64LittleEndian(span, _guestMemory.PairLockTableGpuAddress);
                    break;
                case MslArgumentFieldKind.BindlessImageTables:
                    for (var index = 0; index < field.Count; index++)
                    {
                        if (index >= stage.BindlessTableAddresses.Length || stage.BindlessTableAddresses[index] == 0)
                        {
                            throw Fatal(
                                $"A bindless image table was not bound: slot={index} hash=0x{stage.Program.Hash:X16}.");
                        }

                        BinaryPrimitives.WriteUInt64LittleEndian(
                            span[(index * sizeof(ulong))..], stage.BindlessTableAddresses[index]);
                    }

                    break;
                default:
                    BinaryPrimitives.WriteUInt32LittleEndian(span, WordFieldValue(stage, field.Kind));
                    break;
            }
        }

        var range = ArgumentBufferReuse ? CommitArgumentBytes(stageIndex, argument) : WriteArgumentRange(argument);
        _stageArgumentBuffers[stageIndex] = range.Buffer;
        _stageArgumentOffsets[stageIndex] = range.Offset;
        _stageArgumentValid[stageIndex] = true;
    }

    // SHARPEMU_METAL_V2_ARGUMENT_REUSE=0 reserves and fills a ring range for every stage of every draw.
    private static readonly bool ArgumentBufferReuse =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_ARGUMENT_REUSE");

    private static readonly bool ArgumentReuseVerify =
        SharpEmu.ShaderCompiler.PerfSwitches.Verify("SHARPEMU_METAL_V2_ARGUMENT_REUSE");

    private byte[] _argumentScratch = new byte[512];

    // Four recent argument buffers per stage: draws alternate between a few materials.
    private readonly MetalRegionMemo[] _argumentRegions = CreateRegionMemos(StageCount, slots: 4);

    // The push-data block of a draw: the shader data of its stages, as one buffer.
    private readonly MetalRegionMemo _pushRegions = new(slots: 4);

    // The uploaded tables (flattened resource table, shader data) of every stage.
    private readonly MetalRegionMemo _uploadedTables = new(slots: 8);

    private static MetalRegionMemo[] CreateRegionMemos(int count, int slots)
    {
        var memos = new MetalRegionMemo[count];
        for (var index = 0; index < memos.Length; index++)
        {
            memos[index] = new MetalRegionMemo(slots);
        }

        return memos;
    }

    /// <summary>Argument buffers a draw bound from an earlier draw of its command buffer.</summary>
    internal long ArgumentBufferReuses => _argumentRegions.Sum(memo => memo.Hits) + _pushRegions.Hits;

    /// <summary>Argument buffers written to a new ring range.</summary>
    internal long ArgumentBufferWrites => _argumentRegions.Sum(memo => memo.Misses) + _pushRegions.Misses;

    private MetalGuestRange WriteArgumentRange(ReadOnlySpan<byte> bytes)
    {
        var range = _guestMemory.Allocate((ulong)bytes.Length, 256, out var span);
        span.Clear();
        bytes.CopyTo(span);
        return range;
    }

    private MetalGuestRange CommitArgumentBytes(int stageIndex, ReadOnlySpan<byte> bytes)
    {
        var memo = _argumentRegions[stageIndex];
        if (memo.TryFind(bytes, _submissionSerial, out var found))
        {
            if (ArgumentReuseVerify && !_guestMemory.RingSpan(in found).SequenceEqual(bytes))
            {
                throw Fatal($"A reused argument buffer no longer holds what the draw wrote: stage={stageIndex}.");
            }

            return found;
        }

        var range = WriteArgumentRange(bytes);
        memo.Add(bytes, _submissionSerial, in range);
        return range;
    }

    // A binding lists one resource index per declared element; a dynamic-mip storage
    // image lists the same index once per mip, and each occurrence takes its own view.
    private void WriteTextureField(
        MetalStageBindings stage,
        MslArgumentField field,
        int[]? bindingResources,
        int[]? bindingMips,
        Span<byte> span,
        ref int imageCursor,
        int stageIndex)
    {
        for (var index = 0; index < field.Count; index++)
        {
            var resource = bindingResources is not null && index < bindingResources.Length
                ? bindingResources[index]
                : imageCursor + index;
            if (resource < 0 || resource >= stage.ImageCount)
            {
                continue;
            }

            var mip = bindingMips is not null && index < bindingMips.Length ? bindingMips[index] : 0;
            var texture = stage.Textures[resource];
            var textureId = stage.TextureIds[resource];
            if (mip > 0)
            {
                if (stage.MipTextures[resource] is not { } mipTextures || mip > mipTextures.Length)
                {
                    throw Fatal(
                        $"A storage image element has no view of its mip: image={resource} mip={mip} " +
                        $"hash=0x{stage.Program.Hash:X16}.");
                }

                texture = mipTextures[mip - 1];
                textureId = stage.MipTextureIds[resource]![mip - 1];
            }

            if (texture == 0)
            {
                continue;
            }

            var dummy = TinyPassDummyTexture(texture, resource);
            if (dummy != 0)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(
                    span[(index * sizeof(ulong))..], MetalNative.SendGpuResourceId(dummy, SelGpuResourceId));
                NoteUsedResource(dummy, Mtl.ResourceUsageRead);
                continue;
            }

            BinaryPrimitives.WriteUInt64LittleEndian(
                span[(index * sizeof(ulong))..], textureId);
            var usage = Mtl.ResourceUsageRead | (stage.IsStorage[resource] ? Mtl.ResourceUsageWrite : 0);
            NoteUsedResource(texture, usage);
        }

        _ = stageIndex;
        imageCursor += (int)field.Count;
    }

    // Under SHARPEMU_METAL_V2_TRACE_INDIRECT_DISPATCH, the scalar words a traced program reads from
    // its flattened table: what the host evaluated from guest memory when it bound the program.
    private int _tracedFlattenedTables;
    private long _flattenedTableBinds;

    private void TraceFlattenedTable(MetalStageBindings stage)
    {
        if (_traceIndirectDispatchHashes.Count == 0 || stage.Program is not { } program ||
            !_traceIndirectDispatchHashes.Contains(program.Hash) || _tracedFlattenedTables >= 40 ||
            _flattenedTableBinds++ % 50 != 0)
        {
            return;
        }

        _tracedFlattenedTables++;
        var words = stage.StageResources.Resources.FlattenedResourceTable;
        Console.Error.WriteLine(
            $"[METAL-V2][INDIRECT-DISPATCH] table hash=0x{program.Hash:X16} words={words.Length} " +
            string.Join(",", words.Select(word => word.ToString("X", System.Globalization.CultureInfo.InvariantCulture))));
    }

    private ulong PointerFieldValue(MetalStageBindings stage, DescriptorBindingKind kind)
    {
        switch (kind)
        {
            case DescriptorBindingKind.GlobalDataShare:
                return MetalNative.SendGpuAddress(_guestMemory.GdsBuffer, SelGpuAddress);
            case DescriptorBindingKind.DeviceAddressPageTable:
                return UploadDeviceAddressTable();
            case DescriptorBindingKind.FaultBuffer:
                // The fault-bitmap words stay at zero (WordFieldValue); the buffer takes the
                // reach feedback instead, which also takes the single-copy trap's reports.
                return DeviceAddressReachEnabled || _singleCopy is not null
                    ? ReachFeedbackAddress()
                    : MetalNative.SendGpuAddress(_guestMemory.NullBuffer, SelGpuAddress);
            case DescriptorBindingKind.FlattenedResourceTable:
                TraceFlattenedTable(stage);
                return UploadFlattenedTable(stage.StageResources.Resources.FlattenedResourceTable, stage.Program.FlattenedTableMinimumWords);
            case DescriptorBindingKind.ShaderData:
                return UploadDwords(stage.ShaderData.AsSpan(0, (int)stage.Program.Bindings!.ShaderDataDwordCount));
            case DescriptorBindingKind.MipStatistics:
                return LodStatisticsAddress();
            default:
                throw Fatal($"The argument layout has a pointer field of an unknown kind: kind={kind}.");
        }
    }

    private uint WordFieldValue(MetalStageBindings stage, DescriptorBindingKind kind) => kind switch
    {
        DescriptorBindingKind.GlobalDataShare => (uint)MetalGuestMemory.GdsBytes,
        DescriptorBindingKind.DeviceAddressPageTable => _deviceAddressEntryCount | MetalDeviceAddressIndex.IndexedFlag,
        DescriptorBindingKind.FaultBuffer => 0,
        DescriptorBindingKind.FlattenedResourceTable =>
            (uint)stage.StageResources.Resources.FlattenedResourceTable.Length,
        DescriptorBindingKind.ShaderData => stage.Program.Bindings!.ShaderDataDwordCount,
        _ => throw Fatal($"The argument layout has a word field of an unknown kind: kind={kind}."),
    };

    private const int MaxAddressRanges = (int)Gen5MslTranslator.MaxAddressRangeCount;
    private long _deviceAddressSpanVersion = long.MinValue;

    private long _deviceAddressWrites;

    /// <summary>How many bound stages could store through the device-address page table.</summary>
    internal long DeviceAddressWrites => _deviceAddressWrites;

    private ulong _deviceAddressTable;
    private uint _deviceAddressEntryCount;
    private ulong _deviceAddressTableSerial;
    private readonly MetalDeviceAddressTable _deviceAddressTableBuilder = new();

    /// <summary>
    /// The sorted range table the shader binary-searches (<see cref="MetalDeviceAddressTable"/>),
    /// written once per submission: its runs change only when the mapped spans do, its
    /// entries when the windows are cut differently, its reach flags with the submission's
    /// named windows.
    /// </summary>
    private ulong UploadDeviceAddressTable()
    {
        if (_deviceAddressTable != 0 && _deviceAddressTableSerial == _submissionSerial)
        {
            return _deviceAddressTable;
        }

        // The table is the registry's mapped spans. Without a registry it would be empty,
        // and every load through a raw guest pointer would read zero and every store and
        // atomic would be dropped without a word.
        var gpuMemory = SharpEmu.HLE.GpuMemory.GuestGpuMemoryHook.Current ??
            throw Fatal("A device-address program needs the guest GPU memory registry to build its page table.");
        var spanVersion = gpuMemory.SpanVersion;
        if (spanVersion != _deviceAddressSpanVersion)
        {
            RebuildDeviceAddressEntries(gpuMemory);
            _deviceAddressSpanVersion = spanVersion;
        }
        else if (_deviceAddressTableBuilder.PartitionVersion != _guestMemory.PartitionVersion ||
            _deviceAddressTableBuilder.TrapVersion != _guestMemory.TrapVersion)
        {
            _deviceAddressTableBuilder.Recut(_guestMemory);
        }

        BeginDeviceAddressReach();
        var table = _deviceAddressTableBuilder;
        if (_singleCopy is { } singleCopy)
        {
            // Trap cuts take entries: arming waits while a quarter of the table is all that is left.
            singleCopy.ArmingPaused = table.Count >= MaxAddressRanges - (MaxAddressRanges / 4);
            if (table.TrapOverflowed)
            {
                // No entry could be flagged: nothing released may stay a hole.
                singleCopy.NoteDeviceAddressUse(late: true, "the device-address table has no room for the trap cuts");
            }
        }
        var range = _guestMemory.Allocate((ulong)table.ByteSize, 256, out var span);
        table.Write(span, DeviceAddressReachEnabled ? _deviceAddressNamedWindows : null);
        _deviceAddressTable = range.GpuAddress;
        _deviceAddressEntryCount = (uint)table.Count;
        _deviceAddressTableSerial = _submissionSerial;
        return _deviceAddressTable;
    }

    private int _deviceAddressGranulesLogged = -1;

    // Each guest span becomes the fewest runs that cover it inside one granule and one
    // window. A span whose backing is not contiguous (fragmented flexible memory) or crosses
    // a granule or window edge splits into several entries; pages with no window are left
    // out, never copied.
    private void RebuildDeviceAddressEntries(SharpEmu.HLE.GpuMemory.GuestGpuMemory gpuMemory)
    {
        var table = _deviceAddressTableBuilder;
        table.Rebuild(_guestMemory, gpuMemory.ForEachSpan, MaxAddressRanges);
        MetalGpuReferenceCensus.SetDeviceAddressSpans(table.Entries.Select(entry =>
            (entry.FirstPage << DeviceAddressPaging.PageBits, entry.PageCount << DeviceAddressPaging.PageBits)));
        // The granules the table spans are what an encoder running a device-address program
        // would name without the reach set; say when that set grows so a wired-memory jump can
        // be read against it.
        var granules = table.Granules.Count;
        if (granules != _deviceAddressGranulesLogged)
        {
            _deviceAddressGranulesLogged = granules;
            Console.Error.WriteLine(
                $"[METAL-V2][INFO] device-address table: spans={table.SpanCount} entries={table.Count} " +
                $"span_mb={table.SpanBytes / (1024 * 1024)} granules={granules} " +
                $"granule_mb={_guestMemory.GranuleBytes >> 20} " +
                $"granule_span_mb={_guestMemory.GranuleSetBytes(table.Granules) / (1024 * 1024)} " +
                $"windows={table.Windows.Count} window_span_mb={_guestMemory.WindowSetBytes(table.Windows) / (1024 * 1024)}");
        }
    }

    /// <summary>Adds one run to a table; see <see cref="MetalDeviceAddressTable.AddEntry"/>.</summary>
    internal static void AddDeviceAddressEntry(
        List<(ulong FirstPage, ulong PageCount, ulong Base)> entries,
        (ulong FirstPage, ulong PageCount, ulong Base) entry,
        int spanCount) =>
        MetalDeviceAddressTable.AddEntry(entries, entry, spanCount, MaxAddressRanges);

    // The program reads the table's fixed slots below minimumWords without a bounds check
    // (Gen5MslCodegenOptions.ConstantTableSlots), so a shorter table goes up zero-padded:
    // those slots read the 0 the check gave. The padded words are part of the reuse key.
    private ulong UploadFlattenedTable(uint[] table, uint minimumWords)
    {
        if ((uint)table.Length >= minimumWords)
        {
            return UploadDwords(table);
        }

        Span<uint> padded = minimumWords <= 1024 ? stackalloc uint[(int)minimumWords] : new uint[minimumWords];
        padded.Clear();
        table.CopyTo(padded);
        return UploadDwords(padded);
    }

    private ulong UploadDwords(ReadOnlySpan<uint> words)
    {
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(words);
        if (ArgumentBufferReuse && _uploadedTables.TryFind(bytes, _submissionSerial, out var found))
        {
            return found.GpuAddress;
        }

        var range = _guestMemory.Allocate((ulong)Math.Max(bytes.Length, 16), 256, out var span);
        span.Clear();
        bytes.CopyTo(span);
        if (ArgumentBufferReuse)
        {
            _uploadedTables.Add(bytes, _submissionSerial, in range);
        }

        return range.GpuAddress;
    }

    private void RememberPlanInstance(BindingLayout layout, ArgumentPlan plan)
    {
        // Bounded: layouts of programs since evicted would otherwise stay reachable.
        if (_argumentPlansByInstance.Count >= 8192)
        {
            _argumentPlansByInstance.Clear();
        }

        _argumentPlansByInstance[layout] = plan;
    }

    private ArgumentPlan ArgumentPlanOf(BindingLayout layout)
    {
        if (_argumentPlansByInstance.TryGetValue(layout, out var known))
        {
            return known;
        }

        if (_argumentLayouts.TryGetValue(layout, out var found))
        {
            RememberPlanInstance(layout, found);
            return found;
        }

        var built = Gen5MslArgumentLayout.Build(layout);
        var fieldResources = new int[]?[built.Fields.Count];
        var fieldMips = new int[]?[built.Fields.Count];
        for (var index = 0; index < fieldResources.Length; index++)
        {
            var field = built.Fields[index];
            IReadOnlyList<uint>? resources = field.FieldKind switch
            {
                MslArgumentFieldKind.Textures => layout.Find(field.Kind)?.Resources,
                MslArgumentFieldKind.Samplers => layout.Find(DescriptorBindingKind.Samplers)?.Resources,
                _ => null,
            };
            if (resources is null)
            {
                continue;
            }

            var converted = new int[resources.Count];
            for (var item = 0; item < converted.Length; item++)
            {
                converted[item] = (int)resources[item];
            }

            fieldResources[index] = converted;
            if (field.FieldKind == MslArgumentFieldKind.Textures)
            {
                fieldMips[index] = ElementMips(resources);
            }
        }

        var plan = new ArgumentPlan(built, fieldResources, fieldMips);
        _argumentLayouts[layout] = plan;
        RememberPlanInstance(layout, plan);
        return plan;
    }

    internal void NoteUsedResource(nint resource, nuint usage)
    {
        for (var index = 0; index < _usedResourceCount; index++)
        {
            if (_usedResources[index] == resource)
            {
                _usedResourceUsage[index] |= usage;
                return;
            }
        }

        if (_usedResourceCount == _usedResources.Length)
        {
            Array.Resize(ref _usedResources, _usedResources.Length * 2);
            Array.Resize(ref _usedResourceUsage, _usedResourceUsage.Length * 2);
        }

        _usedResources[_usedResourceCount] = resource;
        _usedResourceUsage[_usedResourceCount] = usage;
        _usedResourceCount++;
    }

    // Argument-buffer references are indirect, so the encoder is told which textures
    // the draw touches. The guest windows live in the residency set instead.
    private void ApplyStageBindings(bool graphics)
    {
        if (graphics)
        {
            // A mesh pipeline has no vertex stage: the primitive shader's resources
            // bind to the mesh stage, and so does the residency of the windows it
            // fetches its own attributes through.
            var meshStage = _pendingGraphicsPipeline?.Mesh is not null;
            var stages = meshStage
                ? Mtl.RenderStageMesh | Mtl.RenderStageFragment
                : Mtl.RenderStageVertex | Mtl.RenderStageFragment;
            BindStageBuffers(_renderEncoder, StageVertex, meshStage ? SelSetMeshBuffer : SelSetVertexBuffer);
            BindStageBuffers(_renderEncoder, StageFragment, SelSetFragmentBuffer);
            NameSubmissionResidents(_renderEncoder, stages, ResidentMarkGraphics);

            for (var index = 0; index < _usedResourceCount; index++)
            {
                if (TryNoteResident(_usedResources[index], _usedResourceUsage[index], stages))
                {
                    MetalNative.SendUseResource(
                        _renderEncoder, SelUseResourceStages, _usedResources[index], _usedResourceUsage[index], stages);
                }
            }

            if (_stageUsesDeviceAddresses[StageVertex] || _stageUsesDeviceAddresses[StageFragment])
            {
                NameDeviceAddressWindows(_renderEncoder, stages);
            }

            DeclareBindlessImages(_renderEncoder, graphics: true, stages);
            return;
        }

        BindStageBuffers(_computeEncoder, StageCompute, SelSetComputeBuffer);
        NameSubmissionResidents(_computeEncoder, stages: 0, ResidentMarkCompute);

        for (var index = 0; index < _usedResourceCount; index++)
        {
            if (TryNoteResident(_usedResources[index], _usedResourceUsage[index], stages: 0))
            {
                MetalNative.SendUseResourceUsage(
                    _computeEncoder, SelUseResourceUsage, _usedResources[index], _usedResourceUsage[index]);
            }
        }

        if (_stageUsesDeviceAddresses[StageCompute])
        {
            NameDeviceAddressWindows(_computeEncoder, stages: 0);
        }

        DeclareBindlessImages(_computeEncoder, graphics: false, stages: 0);
    }

    // -useResource: is encoder state, not draw state: a resource named once stays
    // resident for the rest of the encoder. Naming every window and every texture
    // again on every draw was four to a dozen message sends per draw for nothing.
    private nint[] _encoderResidentResources = new nint[64];
    private nuint[] _encoderResidentUsage = new nuint[64];
    private nuint[] _encoderResidentStages = new nuint[64];
    private int _encoderResidentCount;
    private readonly Dictionary<nint, int> _encoderResidentIndex = new();
    private long _residencyCallsSkipped;

    private void ForgetResidentResources()
    {
        _encoderEpoch++;
        if (_encoderResidentCount != 0)
        {
            _encoderResidentIndex.Clear();
        }

        _encoderResidentCount = 0;
    }

    /// <summary>How many -useResource: calls the encoder cache skipped.</summary>
    public long TakeSkippedResidencyCalls() => Interlocked.Exchange(ref _residencyCallsSkipped, 0);

    // A device-address program can dereference any mapped guest page, so the encoder that
    // runs one names the windows the address table spans, whether or not anything resolved
    // a range through them - or, with the reach profile, those holding the granules the
    // title's programs are known to reach. Only that encoder: the rest of the submission
    // names the windows its own bindings resolved.
    private void NameDeviceAddressWindows(nint encoder, nuint stages)
    {
        // The named windows are settled with the submission's table; a program bound
        // without one (nothing read the table field) still names this submission's set.
        if (_deviceAddressTableSerial != _submissionSerial)
        {
            UploadDeviceAddressTable();
        }

        var named = DeviceAddressNamedWindows();
        if (SharpEmu.Logging.MemoryTelemetry.Enabled)
        {
            _commandBufferReachWindows.UnionWith(named);
        }

        if (DeviceAddressBufferReuse)
            _deviceAddressBufferReuses++;
        else
            ResolveDeviceAddressBuffers();

        ref var mark = ref _residentMarks[ResidentMarkDeviceAddress];
        if (ResidentWatermark && mark.Matches(_encoderEpoch, _deviceAddressBuffersVersion, stages))
        {
            _residentWalksSkipped++;
            return;
        }

        NameResidentBuffers(encoder, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_deviceAddressBuffers), stages);
        mark = new ResidentMark(_encoderEpoch, _deviceAddressBuffersVersion, stages);
    }

    // BeginDeviceAddressReach pins the table's window cuts for this submission.
    // Resolve their buffers there once; encoder residency still accounts for every stage.
    private static readonly bool DeviceAddressBufferReuse =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_DA_BUFFER_REUSE");
    private long _deviceAddressBufferBuilds;
    private long _deviceAddressBufferReuses;
    private long _deviceAddressBufferLookups;
    private readonly List<nint> _deviceAddressBuffers = new();

    private void ResolveDeviceAddressBuffers()
    {
        _deviceAddressBufferBuilds++;
        _deviceAddressBuffersVersion++;
        _deviceAddressBuffers.Clear();
        foreach (var id in DeviceAddressNamedWindows())
        {
            _deviceAddressBufferLookups++;
            var buffer = _guestMemory.WindowBuffer(id);
            if (buffer != 0)
                _deviceAddressBuffers.Add(buffer);
        }
    }

    private long _deviceAddressBuffersVersion;

    // What an encoder was last told about a list of buffers: the list is the same set while its
    // version holds, and a name sticks to the encoder for its life, so a draw that finds the
    // encoder, the version and the stages unchanged has nothing to name and need not walk the
    // list (one dictionary probe per window per draw otherwise). SHARPEMU_METAL_V2_RESIDENT_WATERMARK=0
    // walks it every draw.
    private static readonly bool ResidentWatermark =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_RESIDENT_WATERMARK");

    private const int ResidentMarkGraphics = 0;
    private const int ResidentMarkCompute = 1;
    private const int ResidentMarkDeviceAddress = 2;

    // Count: how many of the list's buffers the encoder has been told, for a list that only grows while its version holds.
    private readonly record struct ResidentMark(long Epoch, long Version, nuint Stages, int Count = 0)
    {
        public bool Matches(long epoch, long version, nuint stages) =>
            Epoch == epoch && Version == version && Stages == stages;
    }

    private readonly ResidentMark[] _residentMarks = new ResidentMark[3] { new(-1, -1, 0), new(-1, -1, 0), new(-1, -1, 0) };
    private long _residentWalksSkipped;

    /// <summary>Per-draw walks of a resident list that the encoder watermark skipped.</summary>
    internal long ResidentWalksSkipped => Volatile.Read(ref _residentWalksSkipped);

    // The submission's window and ring buffers, named on the encoder once per change of the list.
    private void NameSubmissionResidents(nint encoder, nuint stages, int markIndex)
    {
        var buffers = _guestMemory.ResidentBuffers;
        var version = _guestMemory.ResidentVersion;
        ref var mark = ref _residentMarks[markIndex];
        var named = 0;
        if (ResidentWatermark && mark.Matches(_encoderEpoch, version, stages) && mark.Count <= buffers.Length)
        {
            if (mark.Count == buffers.Length)
            {
                _residentWalksSkipped++;
                return;
            }

            // Windows touched since the last draw were added to the end: name those, not every one again.
            named = mark.Count;
        }

        NameResidentBuffers(encoder, buffers[named..], stages);
        mark = new ResidentMark(_encoderEpoch, version, stages, buffers.Length);
    }

    private nint[] _useResourcesScratch = new nint[64];

    // Names guest windows and resident shared buffers read-write on an encoder, those it has
    // not named yet in one -useResources: message: with granule-sized windows an encoder can name
    // a few hundred buffers, and one message per buffer was most of what that cost.
    private unsafe void NameResidentBuffers(nint encoder, ReadOnlySpan<nint> buffers, nuint stages)
    {
        const nuint usage = Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite;
        var count = 0;
        foreach (var buffer in buffers)
        {
            if (!TryNoteResident(buffer, usage, stages))
            {
                continue;
            }

            if (count == _useResourcesScratch.Length)
            {
                Array.Resize(ref _useResourcesScratch, count * 2);
            }

            _useResourcesScratch[count++] = buffer;
        }

        if (count == 0)
        {
            return;
        }

        fixed (nint* resources = _useResourcesScratch)
        {
            if (stages == 0)
            {
                MetalNative.SendUseResources(encoder, SelUseResourcesUsage, resources, (nuint)count, usage);
            }
            else
            {
                MetalNative.SendUseResourcesStages(encoder, SelUseResourcesStages, resources, (nuint)count, usage, stages);
            }
        }
    }

    private bool TryNoteResident(nint resource, nuint usage, nuint stages)
    {
        if (_encoderResidentIndex.TryGetValue(resource, out var index))
        {
            // A wider usage or a new stage has to be named again; the same one need not.
            if ((_encoderResidentUsage[index] & usage) == usage &&
                (_encoderResidentStages[index] & stages) == stages)
            {
                _residencyCallsSkipped++;
                return false;
            }

            _encoderResidentUsage[index] |= usage;
            _encoderResidentStages[index] |= stages;
            return true;
        }

        if (_encoderResidentCount == _encoderResidentResources.Length)
        {
            Array.Resize(ref _encoderResidentResources, _encoderResidentCount * 2);
            Array.Resize(ref _encoderResidentUsage, _encoderResidentCount * 2);
            Array.Resize(ref _encoderResidentStages, _encoderResidentCount * 2);
        }

        _encoderResidentIndex[resource] = _encoderResidentCount;
        _encoderResidentResources[_encoderResidentCount] = resource;
        _encoderResidentUsage[_encoderResidentCount] = usage;
        _encoderResidentStages[_encoderResidentCount] = stages;
        _encoderResidentCount++;
        return true;
    }

    private void BindStageBuffers(nint encoder, int stageIndex, nint setBufferSelector)
    {
        if (encoder == 0)
        {
            return;
        }

        if (_stageArgumentValid[stageIndex])
        {
            BindStageBuffer(encoder, setBufferSelector, _stageArgumentBuffers[stageIndex],
                _stageArgumentOffsets[stageIndex], Gen5MslArgumentLayout.ResourcesBufferIndex);
        }

        if (_hasPushData && (_preparedStageMask & (1 << stageIndex)) != 0)
        {
            BindStageBuffer(encoder, setBufferSelector, _pushBuffer,
                _pushOffset, Gen5MslArgumentLayout.PushDataBufferIndex);
        }
    }

    private void BindStageBuffer(nint encoder, nint selector, nint buffer, ulong offset, int slot)
    {
        // Compute utility kernels overwrite their slots independently of this path.
        // Keep their binding calls unchanged, and distinguish physical vertex/mesh state.
        var stage = selector == SelSetVertexBuffer ? MetalGraphicsBindingStage.Vertex
            : selector == SelSetMeshBuffer ? MetalGraphicsBindingStage.Mesh
            : MetalGraphicsBindingStage.Fragment;
        if (selector != SelSetVertexBuffer && selector != SelSetMeshBuffer && selector != SelSetFragmentBuffer)
        {
            MetalNative.SendSetBuffer(encoder, selector, buffer, (nuint)offset, (nuint)slot);
            return;
        }
        _graphicsBindingAttempts[(int)stage]++;
        if (GraphicsBindingReuseEnabled && _graphicsBindings.Matches(stage, selector, slot, buffer, offset))
        {
            _graphicsBindingSkipped[(int)stage]++;
            return;
        }
        MetalNative.SendSetBuffer(encoder, selector, buffer, (nuint)offset, (nuint)slot);
        if (GraphicsBindingReuseEnabled) _graphicsBindings.Record(stage, selector, slot, buffer, offset);
    }

    // One MTLSamplerState per distinct guest descriptor and view numeric class.
    private nint GetSampler(in SamplerDescriptorWords words, bool integerView, out ulong resourceId)
    {
        var key = (words[0], words[1], words[2], words[3], integerView);
        if (_samplers.TryGetValue(key, out var existing))
        {
            resourceId = existing.ResourceId;
            return existing.Handle;
        }

        var sampler = CreateSampler(in words, integerView);
        resourceId = sampler != 0 ? MetalNative.SendGpuResourceId(sampler, SelGpuResourceId) : 0;
        _samplers[key] = (sampler, resourceId);
        return sampler;
    }

    private nint CreateSampler(in SamplerDescriptorWords words, bool integerView)
    {
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLSamplerDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        if (descriptor == 0)
        {
            throw Fatal("A Metal sampler descriptor could not be created.");
        }

        try
        {
            var magnify = integerView ? 0u : words.MagnifyFilter;
            var minify = integerView ? 0u : words.MinifyFilter;
            var mipFilter = words.MipFilter;
            if (integerView && mipFilter == 2)
            {
                mipFilter = 1;
            }

            RequireExactAddressModes(in words, magnify, minify);

            MetalV2Native.SendVoidValue(
                descriptor,
                MetalNative.Selector("setMinFilter:"),
                (nuint)(IsLinearFilter(minify) ? Mtl.SamplerFilterLinear : Mtl.SamplerFilterNearest));
            MetalV2Native.SendVoidValue(
                descriptor,
                MetalNative.Selector("setMagFilter:"),
                (nuint)(IsLinearFilter(magnify) ? Mtl.SamplerFilterLinear : Mtl.SamplerFilterNearest));
            MetalV2Native.SendVoidValue(
                descriptor,
                MetalNative.Selector("setMipFilter:"),
                (nuint)(mipFilter switch
                {
                    0 => Mtl.SamplerMipFilterNotMipmapped,
                    2 => Mtl.SamplerMipFilterLinear,
                    _ => Mtl.SamplerMipFilterNearest,
                }));
            MetalV2Native.SendVoidValue(
                descriptor, MetalNative.Selector("setSAddressMode:"), (nuint)MetalV2Formats.SamplerAddressMode(words.ClampX));
            MetalV2Native.SendVoidValue(
                descriptor, MetalNative.Selector("setTAddressMode:"), (nuint)MetalV2Formats.SamplerAddressMode(words.ClampY));
            MetalV2Native.SendVoidValue(
                descriptor, MetalNative.Selector("setRAddressMode:"), (nuint)MetalV2Formats.SamplerAddressMode(words.ClampZ));
            var anisotropy = IsAnisotropic(magnify) || IsAnisotropic(minify)
                ? words.MaxAnisotropyRatio switch { 0 => 1u, 1 => 2u, 2 => 4u, 3 => 8u, _ => 16u }
                : 1u;
            MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setMaxAnisotropy:"), anisotropy);
            if (mipFilter != 0)
            {
                MetalV2Native.SendVoidFloatValue(
                    descriptor, MetalNative.Selector("setLodMinClamp:"), words.MinLod / 256.0f);
                MetalV2Native.SendVoidFloatValue(
                    descriptor, MetalNative.Selector("setLodMaxClamp:"), words.MaxLod / 256.0f);
            }

            if (words.DepthCompareFunction != 0)
            {
                MetalV2Native.SendVoidValue(
                    descriptor,
                    MetalNative.Selector("setCompareFunction:"),
                    (nuint)MetalV2Formats.CompareFunction((GpuCompareOp)words.DepthCompareFunction));
            }

            var border = words.BorderColorType switch
            {
                1 => Mtl.SamplerBorderOpaqueBlack,
                2 => Mtl.SamplerBorderOpaqueWhite,
                _ => Mtl.SamplerBorderTransparentBlack,
            };
            MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setBorderColor:"), (nuint)border);
            MetalNative.SendVoidBool(
                descriptor, MetalNative.Selector("setNormalizedCoordinates:"), !words.ForceUnnormalizedCoordinates);
            // An argument buffer stores a sampler by gpuResourceID, which Metal only
            // provides for a sampler created with argument-buffer support.
            MetalNative.SendVoidBool(descriptor, MetalNative.Selector("setSupportArgumentBuffers:"), true);
            var sampler = MetalNative.Send(_device, SelNewSamplerState, descriptor);
            if (sampler == 0)
            {
                throw Fatal(
                    $"A Metal sampler state could not be created: words={words[0]:X8},{words[1]:X8},{words[2]:X8},{words[3]:X8}.");
            }

            return sampler;
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }
    }

    // CLAMP_HALF_BORDER and MIRROR_ONCE_HALF_BORDER have no host address mode and no
    // exactly derivable edge behaviour; they sample clamp-to-border and mirror-once-to-edge,
    // a documented deviation within half a texel of the edge. MIRROR_ONCE_BORDER samples
    // mirror-once-to-edge and the shader substitutes the border colour past the mirrored
    // copy, which is exact only while no filter blends a texel with the border; a filtering
    // one blends with the edge texel there. Each deviation is reported once.
    private static void RequireExactAddressModes(in SamplerDescriptorWords words, uint magnify, uint minify)
    {
        foreach (var (axis, mode) in new[] { ('X', words.ClampX), ('Y', words.ClampY), ('Z', words.ClampZ) })
        {
            if (mode is 4 or 5)
            {
                MetalV2Diagnostics.WarnOnce(
                    $"half-border-{mode}",
                    $"The sampler address mode has no exact host form and samples {(mode == 4 ? "clamp-to-border" : "mirror-once-to-edge")}: " +
                    $"clamp{axis}={mode} words={words[0]:X8},{words[1]:X8},{words[2]:X8},{words[3]:X8}.");
            }
        }

        var pointXy = magnify == 0 && minify == 0;
        var zFilter = (words[2] >> 24) & 0x3;
        var pointZ = zFilter == 1 || (zFilter == 0 && pointXy);
        if (((words.ClampX == 7 || words.ClampY == 7) && !pointXy) || (words.ClampZ == 7 && !pointZ))
        {
            MetalV2Diagnostics.WarnOnce(
                "mirror-once-border-filter",
                $"A MIRROR_ONCE_BORDER sampler filters across the border and blends with the edge texel there: " +
                $"words={words[0]:X8},{words[1]:X8},{words[2]:X8},{words[3]:X8}.");
        }
    }

    private static bool IsLinearFilter(uint filter) => filter is 1 or 3;

    private static bool IsAnisotropic(uint filter) => filter is 2 or 3;

    public bool SamplesDepthAttachment(in DepthAttachmentState depth)
    {
        // Metal has no read-only depth attachment layout to pick, so the answer only
        // selects a layout the host does not model; a sampled depth attachment works.
        _ = depth;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Dword(ReadOnlySpan<byte> bytes, int index) =>
        BinaryPrimitives.ReadUInt32LittleEndian(bytes[(index * sizeof(uint))..]);
}
