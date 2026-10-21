// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

internal static class DescriptorBindingAliases
{
    private sealed class WordComparer : IEqualityComparer<uint[]>
    {
        public bool Equals(uint[]? left, uint[]? right) =>
            ReferenceEquals(left, right) || left is not null && right is not null && left.AsSpan().SequenceEqual(right);

        public int GetHashCode(uint[] words)
        {
            var hash = new HashCode();
            foreach (var word in words) hash.Add(word);
            return hash.ToHashCode();
        }
    }

    // SHARPEMU_RESOURCE_GROUP_REUSE=0 groups through the allocating implementation.
    internal static readonly bool Reuse = PerfSwitches.Enabled("SHARPEMU_RESOURCE_GROUP_REUSE");

    private const int RetainedCapacityLimit = 1024;

    [ThreadStatic]
    private static Dictionary<uint[], uint>? _groups;

    [ThreadStatic]
    private static uint[]? _scratch;

    // The group (index of the first equal descriptor) of each of the first count descriptors,
    // or none when they are all distinct. A materialization miss calls this twice, so the
    // dictionary and the group array are reused: growing a fresh dictionary per call was a
    // twentieth of the Metal worker.
    public static List<uint> Group(IReadOnlyList<uint[]> descriptors, IReadOnlyList<ImageSpecialization> images) =>
        Group(descriptors, descriptors.Count, images);

    public static List<uint> Group(IReadOnlyList<uint[]> descriptors, int count, IReadOnlyList<ImageSpecialization>? images = null)
    {
        if (!Reuse) return GroupAllocating(descriptors, count, images);
        if (count < 2) return [];
        var groups = _groups ??= new Dictionary<uint[], uint>(new WordComparer());
        var scratch = _scratch;
        if (scratch is null || scratch.Length < count) _scratch = scratch = new uint[Math.Max(count, 32)];
        groups.EnsureCapacity(count);
        var ungrouped = 0;
        try
        {
            for (var index = 0; index < count; index++)
            {
                if (images is not null && images[index].IndirectRoot != DescriptorConstants.NoIndex)
                {
                    scratch[index] = (uint)index;
                    ungrouped++;
                    continue;
                }
                if (!groups.TryGetValue(descriptors[index], out var group))
                {
                    group = (uint)index;
                    groups.Add(descriptors[index], group);
                }
                scratch[index] = group;
            }

            return groups.Count + ungrouped == count ? [] : [.. scratch.AsSpan(0, count)];
        }
        finally
        {
            groups.Clear();
            // A rare huge table is not kept alive for the life of the thread.
            if (count > RetainedCapacityLimit)
            {
                _groups = null;
                _scratch = null;
            }
        }
    }

    internal static List<uint> GroupAllocating(IReadOnlyList<uint[]> descriptors, int count, IReadOnlyList<ImageSpecialization>? images = null)
    {
        if (count < 2) return [];
        var groups = new Dictionary<uint[], uint>(new WordComparer());
        var result = new List<uint>(count);
        var ungrouped = 0;
        for (var index = 0; index < count; index++)
        {
            if (images is not null && images[index].IndirectRoot != DescriptorConstants.NoIndex)
            {
                result.Add((uint)index);
                ungrouped++;
                continue;
            }

            if (!groups.TryGetValue(descriptors[index], out var group))
            {
                group = (uint)index;
                groups.Add(descriptors[index], group);
            }
            result.Add(group);
        }
        return groups.Count + ungrouped == count ? [] : result;
    }

    public static void Apply(ShaderResourceInfo info, ShaderResourceInfo source, ResourceSpecialization specialization)
    {
        if (specialization.ImageDescriptorGroups.Count != 0)
        {
            if (specialization.ImageDescriptorGroups.Count != info.Images.Count)
                throw new ResourcePlanException("The image descriptor groups do not match the resource table.");
            info.ImageBindings = new uint[info.Images.Count];
            var bindings = new Dictionary<(uint Group, ImageSpecialization View, ImageResourceClass Class,
                ImageMipMode Mip, bool Read, bool Written, bool Atomic, bool Depth, bool R128), uint>();
            for (var index = 0; index < info.Images.Count; index++)
            {
                var image = info.Images[index];
                var key = (specialization.ImageDescriptorGroups[index], specialization.Images[index],
                    image.ResourceClass, image.MipMode, image.Read, image.Written, image.Atomic, image.DepthCompare, image.R128);
                // Indirect tables retain their candidate order and contiguous mip elements.
                if (image.IndirectRoot != DescriptorConstants.NoIndex || !bindings.TryGetValue(key, out var binding))
                {
                    binding = (uint)index;
                    if (image.IndirectRoot == DescriptorConstants.NoIndex) bindings.Add(key, binding);
                }
                info.ImageBindings[index] = binding;
            }
        }

        if (specialization.SamplerDescriptorGroups.Count != 0)
        {
            if (specialization.SamplerDescriptorGroups.Count != source.Samplers.Count)
                throw new ResourcePlanException("The sampler descriptor groups do not match the resource table.");
            var sourceGroups = new Dictionary<uint, uint>();
            for (var index = 0; index < source.Samplers.Count; index++)
                sourceGroups.Add(source.Samplers[index].Source, specialization.SamplerDescriptorGroups[index]);
            info.SamplerBindings = new uint[info.Samplers.Count];
            var bindings = new Dictionary<(uint Group, bool Point, bool Depth), uint>();
            for (var index = 0; index < info.Samplers.Count; index++)
            {
                var sampler = info.Samplers[index];
                var key = (sourceGroups[sampler.Source], sampler.ForcePointFiltering, sampler.DepthCompare);
                if (!bindings.TryGetValue(key, out var binding))
                {
                    binding = (uint)index;
                    bindings.Add(key, binding);
                }
                info.SamplerBindings[index] = binding;
            }
        }
    }
}
