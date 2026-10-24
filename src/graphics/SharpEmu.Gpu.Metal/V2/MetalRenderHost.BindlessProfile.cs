// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    // Diagnostic only. Hash sets and descriptor scans never run in normal/A-B measurements.
    private readonly BindlessCoverage? _bindlessCoverage =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BINDLESS_PROFILE") == "1" ? new() : null;

    private sealed class BindlessCoverage
    {
        private readonly HashSet<(uint, uint, uint, uint, uint, uint, uint, uint)> _images = [];
        private readonly long[] _stages = new long[4];
        private readonly long[] _bufferStages = new long[4];
        private long _draws, _dispatches, _bufferDraws, _bufferDispatches, _emptyDraws;
        private long _buffers, _imageReferences, _samplers, _readonlyStages, _flatFreeStages;
        private bool _issueImages, _issueBuffers, _issueCompute;
        private long _set, _binding, _shape, _gpu, _dcc;

        public void Begin() => (_issueImages, _issueBuffers, _issueCompute) = (false, false, false);

        public void Stage(ShaderProgramInfo program, ResourceSnapshot snapshot)
        {
            var index = program.Stage switch
            {
                ShaderStageKind.Vertex => 0, ShaderStageKind.Pixel => 1,
                ShaderStageKind.Compute => 2, _ => 3,
            };
            _stages[index]++;
            var hasImages = snapshot.Images.Length != 0 || snapshot.BindlessImages.Length != 0;
            if (!hasImages) _bufferStages[index]++;
            _issueImages |= hasImages;
            _issueBuffers |= snapshot.Buffers.Length != 0 || program.UsesDeviceAddresses;
            _issueCompute |= program.Stage == ShaderStageKind.Compute;
            _buffers += snapshot.Buffers.Length;
            _samplers += snapshot.Samplers.Length;
            if (!hasImages && !program.WritesDeviceAddresses && program.Buffers.All(buffer => !buffer.Written))
            {
                _readonlyStages++;
                if (snapshot.FlattenedResourceTable.Length == 0) _flatFreeStages++;
            }
            foreach (var words in snapshot.Images) Image(words);
            foreach (var table in snapshot.BindlessImages)
                foreach (var words in table.Descriptors) Image(words);
        }

        private void Image(uint[] words)
        {
            _imageReferences++;
            if (words.Length >= 8)
                _images.Add((words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]));
        }

        public void End()
        {
            if (_issueCompute)
            {
                _dispatches++;
                if (!_issueImages) _bufferDispatches++;
            }
            else
            {
                _draws++;
                if (!_issueImages) _bufferDraws++;
                if (!_issueImages && !_issueBuffers) _emptyDraws++;
            }
        }

        public void Frame(MetalImageCache cache)
        {
            var epochs = cache.Epochs;
            Console.Error.WriteLine(
                $"[METAL-V2][BINDLESS-COVERAGE] frame draws={_draws} dispatches={_dispatches} " +
                $"buffer_draws={_bufferDraws} buffer_dispatches={_bufferDispatches} empty_draws={_emptyDraws} " +
                $"stages=[{string.Join(',', _stages)}] buffer_stages=[{string.Join(',', _bufferStages)}] " +
                $"readonly_buffer_stages={_readonlyStages} readonly_no_flat_stages={_flatFreeStages} " +
                $"buffers={_buffers} image_refs={_imageReferences} distinct_tsharp={_images.Count} samplers={_samplers} " +
                $"image_set_delta={cache.ImageSetVersion - _set} binding_delta={epochs.Binding - _binding} " +
                $"shape_delta={epochs.Shape - _shape} gpu_modified_delta={epochs.GpuModified - _gpu} dcc_delta={epochs.Dcc - _dcc}");
            (_set, _binding, _shape, _gpu, _dcc) =
                (cache.ImageSetVersion, epochs.Binding, epochs.Shape, epochs.GpuModified, epochs.Dcc);
            Array.Clear(_stages);
            Array.Clear(_bufferStages);
            _images.Clear();
            _draws = _dispatches = _bufferDraws = _bufferDispatches = _emptyDraws = 0;
            _buffers = _imageReferences = _samplers = _readonlyStages = _flatFreeStages = 0;
        }
    }
}
