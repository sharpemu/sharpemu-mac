// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Vulkan.Presentation;

using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.VideoOut.Overlay;
using SharpEmu.ShaderCompiler.Vulkan;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

internal static unsafe partial class VulkanVideoPresenter
{
    private sealed partial class Presenter
    {
        // The widget-based performance overlay. PerfOverlay's own thread rasterizes its widgets
        // into a premultiplied BGRA image the size of the surface. The regions that hold widgets
        // are copied through a per-frame-slot staging buffer into an overlay image of the swapchain
        // extent, then blended over the presentation target by a render pass (LoadOp=Load): one
        // scissored fullscreen triangle per region, blend One / OneMinusSrcAlpha. A draw rather
        // than a transfer, because blits and copies onto a swapchain image corrupt on MoltenVK.

        private Image _blendOverlayImage;
        private DeviceMemory _blendOverlayMemory;
        private ImageView _blendOverlayView;
        private int _blendOverlayWidth;
        private int _blendOverlayHeight;
        private bool _blendOverlayInitialized;
        private VkBuffer[] _blendOverlayStaging = [];
        private DeviceMemory[] _blendOverlayStagingMemory = [];
        private nint[] _blendOverlayStagingMapped = [];
        private Sampler _blendOverlaySampler;
        private DescriptorSetLayout _blendOverlaySetLayout;
        private DescriptorPool _blendOverlayPool;
        private DescriptorSet _blendOverlaySet;
        private PipelineLayout _blendOverlayPipelineLayout;
        private RenderPass _blendOverlayRenderPass;
        private Pipeline _blendOverlayPipeline;
        private bool _blendOverlayFailed;

        // The overlay image already holds this frame object at this version and these regions.
        // Commands run in queue order, so a later frame may rely on an earlier frame's upload.
        private OverlayFrame? _blendOverlayUploadedFrame;
        private long _blendOverlayUploadedVersion = -1;
        private OverlayRectangle[] _blendOverlayUploadedRegions = [];

        // Surface pixels per SDL window coordinate; the overlay maps pointer events with it.
        private double _overlayPixelScale;

        // "KHR" or "EXT" when the device enabled VK_*_calibrated_timestamps; null otherwise.
        private string? _calibratedTimestampsSuffix;

        // The window's pixel density (swapchain pixels per SDL window coordinate), which the
        // overlay uses to map mouse positions; read each present so fullscreen and display changes follow.
        private void UpdateOverlayPixelScale() => _overlayPixelScale = _window.PixelDensity;

        private static bool IsSrgbFormat(Format format) =>
            format is Format.B8G8R8A8Srgb or Format.R8G8B8A8Srgb or Format.A8B8G8R8SrgbPack32;

        private void RecordOverlay(uint imageIndex, int frameSlot, bool contentIsPq)
        {
            if (_hdrOutputActive && contentIsPq)
            {
                // The target holds PQ code values, not sRGB; there is no PQ-encoding variant of
                // the overlay shader, so this case keeps the opaque panel copy.
                RecordOverlayBlit(imageIndex, frameSlot);
                return;
            }

            RecordBlendedOverlay(imageIndex, frameSlot);
        }

        private void RecordBlendedOverlay(uint imageIndex, int frameSlot)
        {
            if (_blendOverlayFailed || _extent.Width == 0 || _extent.Height == 0)
            {
                return;
            }

            var width = (int)_extent.Width;
            var height = (int)_extent.Height;
            var frame = PerfOverlay.AcquireFrame(width, height, _overlayPixelScale <= 0 ? 1.0 : _overlayPixelScale);
            if (frame is null)
            {
                return;
            }

            try
            {
                if (frame.Width != width || frame.Height != height || frame.Regions.Count == 0)
                {
                    // Rendered for the previous surface size; the next one will match.
                    return;
                }

                if (!EnsureBlendOverlayResources(width, height))
                {
                    return;
                }

                var regions = frame.Regions;
                var needsUpload = !ReferenceEquals(frame, _blendOverlayUploadedFrame) ||
                                  frame.Version != _blendOverlayUploadedVersion ||
                                  !_blendOverlayUploadedRegions.AsSpan().SequenceEqual(
                                      System.Runtime.InteropServices.CollectionsMarshal.AsSpan(regions));
                if (needsUpload)
                {
                    UploadBlendOverlay(frame, frameSlot);
                    _blendOverlayUploadedFrame = frame;
                    _blendOverlayUploadedVersion = frame.Version;
                    _blendOverlayUploadedRegions = [.. regions];
                }

                DrawBlendOverlay(imageIndex, regions);
            }
            finally
            {
                PerfOverlay.ReleaseFrame(frame);
            }
        }

        private void UploadBlendOverlay(OverlayFrame frame, int frameSlot)
        {
            var stride = frame.Width * 4;
            var staging = (byte*)_blendOverlayStagingMapped[frameSlot];
            var copies = new BufferImageCopy[frame.Regions.Count];
            fixed (byte* pixels = frame.Pixels)
            {
                for (var index = 0; index < frame.Regions.Count; index++)
                {
                    var region = frame.Regions[index];
                    var rowBytes = (long)region.Width * 4;
                    for (var row = 0; row < region.Height; row++)
                    {
                        var offset = (long)(region.Top + row) * stride + (long)region.Left * 4;
                        System.Buffer.MemoryCopy(pixels + offset, staging + offset, rowBytes, rowBytes);
                    }

                    copies[index] = new BufferImageCopy
                    {
                        BufferOffset = (ulong)((long)region.Top * stride + (long)region.Left * 4),
                        BufferRowLength = (uint)frame.Width,
                        BufferImageHeight = (uint)frame.Height,
                        ImageSubresource = new ImageSubresourceLayers(ImageAspectFlags.ColorBit, 0, 0, 1),
                        ImageOffset = new Offset3D(region.Left, region.Top, 0),
                        ImageExtent = new Extent3D((uint)region.Width, (uint)region.Height, 1),
                    };
                }
            }

            var toTransferDst = new ImageMemoryBarrier2
            {
                SType = StructureType.ImageMemoryBarrier2,
                SrcAccessMask = 0,
                DstAccessMask = AccessFlags2.TransferWriteBit,
                OldLayout = _blendOverlayInitialized
                    ? ImageLayout.ShaderReadOnlyOptimal
                    : ImageLayout.Undefined,
                NewLayout = ImageLayout.TransferDstOptimal,
                SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                Image = _blendOverlayImage,
                SubresourceRange = ColorSubresourceRange(),
            };
            VulkanSynchronization.PipelineBarrier(_vk,
                _commandBuffer,
                _blendOverlayInitialized ? PipelineStageFlags.FragmentShaderBit : PipelineStageFlags.TopOfPipeBit,
                PipelineStageFlags.TransferBit,
                0, 0, null, 0, null, 1, &toTransferDst);

            using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.VulkanCommands))
            {
                fixed (BufferImageCopy* copyPointer = copies)
                {
                    _vk.CmdCopyBufferToImage(
                        _commandBuffer,
                        _blendOverlayStaging[frameSlot],
                        _blendOverlayImage,
                        ImageLayout.TransferDstOptimal,
                        (uint)copies.Length,
                        copyPointer);
                }
            }

            var toShaderRead = new ImageMemoryBarrier2
            {
                SType = StructureType.ImageMemoryBarrier2,
                SrcAccessMask = AccessFlags2.TransferWriteBit,
                DstAccessMask = AccessFlags2.ShaderReadBit,
                OldLayout = ImageLayout.TransferDstOptimal,
                NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                Image = _blendOverlayImage,
                SubresourceRange = ColorSubresourceRange(),
            };
            VulkanSynchronization.PipelineBarrier(_vk,
                _commandBuffer,
                PipelineStageFlags.TransferBit,
                PipelineStageFlags.FragmentShaderBit,
                0, 0, null, 0, null, 1, &toShaderRead);
            _blendOverlayInitialized = true;
        }

        private void DrawBlendOverlay(uint imageIndex, List<OverlayRectangle> regions)
        {
            // Whatever wrote the target (upload, guest draw, blit) must finish before blending
            // reads it. The layout is unchanged; the render pass does its own transition.
            var beforeBlend = new ImageMemoryBarrier2
            {
                SType = StructureType.ImageMemoryBarrier2,
                SrcAccessMask = AccessFlags2.MemoryWriteBit,
                DstAccessMask = AccessFlags2.ColorAttachmentReadBit | AccessFlags2.ColorAttachmentWriteBit,
                OldLayout = PresentationTargetFinalLayout,
                NewLayout = PresentationTargetFinalLayout,
                SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                Image = PresentationTargetImage(imageIndex),
                SubresourceRange = ColorSubresourceRange(),
            };
            VulkanSynchronization.PipelineBarrier(_vk,
                _commandBuffer,
                PipelineStageFlags.AllCommandsBit,
                PipelineStageFlags.ColorAttachmentOutputBit,
                0, 0, null, 0, null, 1, &beforeBlend);

            var renderPassInfo = new RenderPassBeginInfo
            {
                SType = StructureType.RenderPassBeginInfo,
                RenderPass = _blendOverlayRenderPass,
                Framebuffer = _framebuffers[imageIndex],
                RenderArea = new Rect2D(new Offset2D(0, 0), _extent),
            };
            using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.VulkanCommands))
            {
                _vk.CmdBeginRenderPass(_commandBuffer, &renderPassInfo, SubpassContents.Inline);
                _vk.CmdBindPipeline(_commandBuffer, PipelineBindPoint.Graphics, _blendOverlayPipeline);
            }

            SharpEmu.Gpu.Vulkan.GraphicsDynamicStateEpoch.Advance();
            SharpEmu.Gpu.Vulkan.NativeBindingEpoch.Advance();
            var descriptorSet = _blendOverlaySet;
            using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.VulkanCommands))
            {
                _vk.CmdBindDescriptorSets(
                    _commandBuffer,
                    PipelineBindPoint.Graphics,
                    _blendOverlayPipelineLayout,
                    0,
                    1,
                    &descriptorSet,
                    0,
                    null);
                foreach (var region in regions)
                {
                    var scissor = new Rect2D(
                        new Offset2D(region.Left, region.Top),
                        new Extent2D((uint)region.Width, (uint)region.Height));
                    _vk.CmdSetScissor(_commandBuffer, 0, 1, &scissor);
                    _vk.CmdDraw(_commandBuffer, 3, 1, 0, 0);
                }

                _vk.CmdEndRenderPass(_commandBuffer);
            }
        }

        private bool EnsureBlendOverlayResources(int width, int height)
        {
            if (_blendOverlayImage.Handle != 0 && _blendOverlayWidth == width && _blendOverlayHeight == height)
            {
                return true;
            }

            try
            {
                if (_blendOverlayImage.Handle != 0)
                {
                    // Only reachable if the extent changed without a swapchain recreate; that
                    // path drains every frame slot first, but be safe.
                    lock (_queueGate)
                    {
                        _vk.DeviceWaitIdle(_device);
                    }

                    DestroyBlendOverlayResources();
                }

                CreateBlendOverlayResources(width, height);
                return true;
            }
            catch (InvalidOperationException exception)
            {
                Console.Error.WriteLine(
                    $"[LOADER][ERROR] Vulkan performance overlay disabled: {exception.Message}");
                _blendOverlayFailed = true;
                DestroyBlendOverlayResources();
                return false;
            }
        }

        private void CreateBlendOverlayResources(int width, int height)
        {
            var imageInfo = new ImageCreateInfo
            {
                SType = StructureType.ImageCreateInfo,
                ImageType = ImageType.Type2D,
                Format = Format.B8G8R8A8Unorm,
                Extent = new Extent3D((uint)width, (uint)height, 1),
                MipLevels = 1,
                ArrayLayers = 1,
                Samples = SampleCountFlags.Count1Bit,
                Tiling = ImageTiling.Optimal,
                Usage = ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit,
                SharingMode = SharingMode.Exclusive,
                InitialLayout = ImageLayout.Undefined,
            };
            Check(_vk.CreateImage(_device, &imageInfo, null, out _blendOverlayImage), "vkCreateImage(overlay)");
            _vk.GetImageMemoryRequirements(_device, _blendOverlayImage, out var requirements);
            var memoryInfo = new MemoryAllocateInfo
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = requirements.Size,
                MemoryTypeIndex = FindMemoryType(requirements.MemoryTypeBits, MemoryPropertyFlags.DeviceLocalBit),
            };
            Check(_deviceInfo.AllocateMemory(memoryInfo, out _blendOverlayMemory), "vkAllocateMemory(overlay)");
            Check(_vk.BindImageMemory(_device, _blendOverlayImage, _blendOverlayMemory, 0), "vkBindImageMemory(overlay)");
            _blendOverlayWidth = width;
            _blendOverlayHeight = height;
            _blendOverlayInitialized = false;
            _blendOverlayUploadedFrame = null;
            _blendOverlayUploadedVersion = -1;
            _blendOverlayUploadedRegions = [];

            var viewInfo = new ImageViewCreateInfo
            {
                SType = StructureType.ImageViewCreateInfo,
                Image = _blendOverlayImage,
                ViewType = ImageViewType.Type2D,
                Format = Format.B8G8R8A8Unorm,
                SubresourceRange = ColorSubresourceRange(),
            };
            Check(_vk.CreateImageView(_device, &viewInfo, null, out _blendOverlayView), "vkCreateImageView(overlay)");

            var bytes = (ulong)width * (ulong)height * 4;
            _blendOverlayStaging = new VkBuffer[MaxFramesInFlight];
            _blendOverlayStagingMemory = new DeviceMemory[MaxFramesInFlight];
            _blendOverlayStagingMapped = new nint[MaxFramesInFlight];
            for (var slot = 0; slot < MaxFramesInFlight; slot++)
            {
                _blendOverlayStaging[slot] = CreateBuffer(
                    bytes,
                    BufferUsageFlags.TransferSrcBit,
                    MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
                    out _blendOverlayStagingMemory[slot]);
                void* mapped;
                Check(
                    _vk.MapMemory(_device, _blendOverlayStagingMemory[slot], 0, bytes, 0, &mapped),
                    "vkMapMemory(overlay staging)");
                _blendOverlayStagingMapped[slot] = (nint)mapped;
            }

            // The image is drawn 1:1 with the surface, so nearest filtering is exact.
            var samplerInfo = new SamplerCreateInfo
            {
                SType = StructureType.SamplerCreateInfo,
                MagFilter = Filter.Nearest,
                MinFilter = Filter.Nearest,
                MipmapMode = SamplerMipmapMode.Nearest,
                AddressModeU = SamplerAddressMode.ClampToEdge,
                AddressModeV = SamplerAddressMode.ClampToEdge,
                AddressModeW = SamplerAddressMode.ClampToEdge,
            };
            Check(_vk.CreateSampler(_device, &samplerInfo, null, out _blendOverlaySampler), "vkCreateSampler(overlay)");

            var binding = new DescriptorSetLayoutBinding
            {
                Binding = 1,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
            };
            var setLayoutInfo = new DescriptorSetLayoutCreateInfo
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = 1,
                PBindings = &binding,
            };
            Check(
                _vk.CreateDescriptorSetLayout(_device, &setLayoutInfo, null, out _blendOverlaySetLayout),
                "vkCreateDescriptorSetLayout(overlay)");
            var poolSize = new DescriptorPoolSize
            {
                Type = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
            };
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = 1,
                PoolSizeCount = 1,
                PPoolSizes = &poolSize,
            };
            Check(_vk.CreateDescriptorPool(_device, &poolInfo, null, out _blendOverlayPool), "vkCreateDescriptorPool(overlay)");
            var setLayout = _blendOverlaySetLayout;
            var allocateInfo = new DescriptorSetAllocateInfo
            {
                SType = StructureType.DescriptorSetAllocateInfo,
                DescriptorPool = _blendOverlayPool,
                DescriptorSetCount = 1,
                PSetLayouts = &setLayout,
            };
            Check(_vk.AllocateDescriptorSets(_device, &allocateInfo, out _blendOverlaySet), "vkAllocateDescriptorSets(overlay)");
            var descriptorImage = new DescriptorImageInfo
            {
                Sampler = _blendOverlaySampler,
                ImageView = _blendOverlayView,
                ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
            };
            var write = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = _blendOverlaySet,
                DstBinding = 1,
                DescriptorCount = 1,
                DescriptorType = DescriptorType.CombinedImageSampler,
                PImageInfo = &descriptorImage,
            };
            _vk.UpdateDescriptorSets(_device, 1, &write, 0, null);
            var layoutInfo = new PipelineLayoutCreateInfo
            {
                SType = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount = 1,
                PSetLayouts = &setLayout,
            };
            Check(
                _vk.CreatePipelineLayout(_device, &layoutInfo, null, out _blendOverlayPipelineLayout),
                "vkCreatePipelineLayout(overlay)");

            CreateBlendOverlayRenderPass();
            CreateBlendOverlayPipeline();
        }

        // Loads and keeps the presentation target. The framebuffers made for the guest-draw render
        // pass are compatible: render pass compatibility ignores load/store ops and layouts.
        private void CreateBlendOverlayRenderPass()
        {
            var attachment = new AttachmentDescription
            {
                Format = PresentationTargetFormat,
                Samples = SampleCountFlags.Count1Bit,
                LoadOp = AttachmentLoadOp.Load,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = PresentationTargetFinalLayout,
                FinalLayout = PresentationTargetFinalLayout,
            };
            var reference = new AttachmentReference(0, ImageLayout.ColorAttachmentOptimal);
            var subpass = new SubpassDescription
            {
                PipelineBindPoint = PipelineBindPoint.Graphics,
                ColorAttachmentCount = 1,
                PColorAttachments = &reference,
            };
            // The same single dependency as the guest-draw render pass: compatibility with its
            // framebuffers requires identical dependencies. DrawBlendOverlay's own barrier orders
            // everything before the pass and makes the target readable for blending.
            var dependency = new SubpassDependency
            {
                SrcSubpass = Vk.SubpassExternal,
                DstSubpass = 0,
                SrcStageMask = PipelineStageFlags.ColorAttachmentOutputBit,
                DstStageMask = PipelineStageFlags.ColorAttachmentOutputBit,
                DstAccessMask = AccessFlags.ColorAttachmentWriteBit,
            };
            var renderPassInfo = new RenderPassCreateInfo
            {
                SType = StructureType.RenderPassCreateInfo,
                AttachmentCount = 1,
                PAttachments = &attachment,
                SubpassCount = 1,
                PSubpasses = &subpass,
                DependencyCount = 1,
                PDependencies = &dependency,
            };
            Check(
                _vk.CreateRenderPass(_device, &renderPassInfo, null, out _blendOverlayRenderPass),
                "vkCreateRenderPass(overlay)");
        }

        private void CreateBlendOverlayPipeline()
        {
            // An sRGB attachment encodes on write and blends in linear light, so the shader hands
            // it linear colour; any other target takes the overlay's encoded colour unchanged.
            var fragmentBytes = IsSrgbFormat(PresentationTargetFormat)
                ? SpirvFixedShaders.CreateSrgbOverlayFragment()
                : SpirvFixedShaders.CreateCopyFragment();
            var vertexModule = CreateShaderModule(SpirvFixedShaders.CreateFullscreenVertex(1));
            var fragmentModule = CreateShaderModule(fragmentBytes);
            var entryPoint = (byte*)SilkMarshal.StringToPtr("main");
            try
            {
                var stages = stackalloc PipelineShaderStageCreateInfo[2];
                stages[0] = new PipelineShaderStageCreateInfo
                {
                    SType = StructureType.PipelineShaderStageCreateInfo,
                    Stage = ShaderStageFlags.VertexBit,
                    Module = vertexModule,
                    PName = entryPoint,
                };
                stages[1] = new PipelineShaderStageCreateInfo
                {
                    SType = StructureType.PipelineShaderStageCreateInfo,
                    Stage = ShaderStageFlags.FragmentBit,
                    Module = fragmentModule,
                    PName = entryPoint,
                };
                var vertexInput = new PipelineVertexInputStateCreateInfo
                {
                    SType = StructureType.PipelineVertexInputStateCreateInfo,
                };
                var inputAssembly = new PipelineInputAssemblyStateCreateInfo
                {
                    SType = StructureType.PipelineInputAssemblyStateCreateInfo,
                    Topology = PrimitiveTopology.TriangleList,
                };
                var viewport = new Viewport(0, 0, _extent.Width, _extent.Height, 0, 1);
                var viewportState = new PipelineViewportStateCreateInfo
                {
                    SType = StructureType.PipelineViewportStateCreateInfo,
                    ViewportCount = 1,
                    PViewports = &viewport,
                    ScissorCount = 1,
                };
                var rasterization = new PipelineRasterizationStateCreateInfo
                {
                    SType = StructureType.PipelineRasterizationStateCreateInfo,
                    PolygonMode = PolygonMode.Fill,
                    CullMode = CullModeFlags.None,
                    FrontFace = FrontFace.CounterClockwise,
                    LineWidth = 1,
                };
                var multisample = new PipelineMultisampleStateCreateInfo
                {
                    SType = StructureType.PipelineMultisampleStateCreateInfo,
                    RasterizationSamples = SampleCountFlags.Count1Bit,
                };
                var blendAttachment = new PipelineColorBlendAttachmentState
                {
                    BlendEnable = true,
                    SrcColorBlendFactor = BlendFactor.One,
                    DstColorBlendFactor = BlendFactor.OneMinusSrcAlpha,
                    ColorBlendOp = BlendOp.Add,
                    SrcAlphaBlendFactor = BlendFactor.One,
                    DstAlphaBlendFactor = BlendFactor.OneMinusSrcAlpha,
                    AlphaBlendOp = BlendOp.Add,
                    ColorWriteMask = ColorComponentFlags.RBit |
                                     ColorComponentFlags.GBit |
                                     ColorComponentFlags.BBit |
                                     ColorComponentFlags.ABit,
                };
                var blend = new PipelineColorBlendStateCreateInfo
                {
                    SType = StructureType.PipelineColorBlendStateCreateInfo,
                    AttachmentCount = 1,
                    PAttachments = &blendAttachment,
                };
                var dynamicState = DynamicState.Scissor;
                var dynamicInfo = new PipelineDynamicStateCreateInfo
                {
                    SType = StructureType.PipelineDynamicStateCreateInfo,
                    DynamicStateCount = 1,
                    PDynamicStates = &dynamicState,
                };
                var pipelineInfo = new GraphicsPipelineCreateInfo
                {
                    SType = StructureType.GraphicsPipelineCreateInfo,
                    StageCount = 2,
                    PStages = stages,
                    PVertexInputState = &vertexInput,
                    PInputAssemblyState = &inputAssembly,
                    PViewportState = &viewportState,
                    PRasterizationState = &rasterization,
                    PMultisampleState = &multisample,
                    PColorBlendState = &blend,
                    PDynamicState = &dynamicInfo,
                    Layout = _blendOverlayPipelineLayout,
                    RenderPass = _blendOverlayRenderPass,
                };
                Check(
                    _vk.CreateGraphicsPipelines(_device, _pipelineCache, 1, &pipelineInfo, null, out _blendOverlayPipeline),
                    "vkCreateGraphicsPipelines(overlay)");
                MarkPipelineCacheDirty();
            }
            finally
            {
                SilkMarshal.Free((nint)entryPoint);
                _vk.DestroyShaderModule(_device, fragmentModule, null);
                _vk.DestroyShaderModule(_device, vertexModule, null);
            }
        }

        // Callers guarantee the device is idle with respect to these resources.
        private void DestroyBlendOverlayResources()
        {
            if (_blendOverlayPipeline.Handle != 0)
            {
                _vk.DestroyPipeline(_device, _blendOverlayPipeline, null);
                _blendOverlayPipeline = default;
            }

            if (_blendOverlayRenderPass.Handle != 0)
            {
                _vk.DestroyRenderPass(_device, _blendOverlayRenderPass, null);
                _blendOverlayRenderPass = default;
            }

            if (_blendOverlayPipelineLayout.Handle != 0)
            {
                _vk.DestroyPipelineLayout(_device, _blendOverlayPipelineLayout, null);
                _blendOverlayPipelineLayout = default;
            }

            if (_blendOverlayPool.Handle != 0)
            {
                _vk.DestroyDescriptorPool(_device, _blendOverlayPool, null);
                _blendOverlayPool = default;
                _blendOverlaySet = default;
            }

            if (_blendOverlaySetLayout.Handle != 0)
            {
                _vk.DestroyDescriptorSetLayout(_device, _blendOverlaySetLayout, null);
                _blendOverlaySetLayout = default;
            }

            if (_blendOverlaySampler.Handle != 0)
            {
                _vk.DestroySampler(_device, _blendOverlaySampler, null);
                _blendOverlaySampler = default;
            }

            for (var slot = 0; slot < _blendOverlayStaging.Length; slot++)
            {
                if (_blendOverlayStagingMapped[slot] != 0)
                {
                    _vk.UnmapMemory(_device, _blendOverlayStagingMemory[slot]);
                }

                if (_blendOverlayStaging[slot].Handle != 0)
                {
                    _vk.DestroyBuffer(_device, _blendOverlayStaging[slot], null);
                }

                if (_blendOverlayStagingMemory[slot].Handle != 0)
                {
                    _deviceInfo.FreeMemory(_blendOverlayStagingMemory[slot]);
                }
            }

            _blendOverlayStaging = [];
            _blendOverlayStagingMemory = [];
            _blendOverlayStagingMapped = [];
            if (_blendOverlayView.Handle != 0)
            {
                _vk.DestroyImageView(_device, _blendOverlayView, null);
                _blendOverlayView = default;
            }

            if (_blendOverlayImage.Handle != 0)
            {
                _vk.DestroyImage(_device, _blendOverlayImage, null);
                _blendOverlayImage = default;
            }

            if (_blendOverlayMemory.Handle != 0)
            {
                _deviceInfo.FreeMemory(_blendOverlayMemory);
                _blendOverlayMemory = default;
            }

            _blendOverlayWidth = 0;
            _blendOverlayHeight = 0;
            _blendOverlayInitialized = false;
            _blendOverlayUploadedFrame = null;
            _blendOverlayUploadedVersion = -1;
            _blendOverlayUploadedRegions = [];
        }
    }
}
