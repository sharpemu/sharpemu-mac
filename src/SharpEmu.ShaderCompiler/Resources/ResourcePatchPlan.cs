// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

// Immutable, plan-owned topology. Only direct V# base dwords with NO other host
// consumer qualify. This is a proof over the graph, never learned from matching samples.
internal sealed class ResourcePatchPlan
{
    internal readonly record struct Patch(int Buffer, int UserData);
    internal Patch[] Patches { get; }
    private readonly bool[] _patchable;

    internal ResourcePatchPlan(ShaderResourcePlan plan)
    {
        var candidates = new List<Patch>();
        var allowedSources = new HashSet<uint>();
        for (var buffer = 0; buffer < plan.Info.Buffers.Count; buffer++)
        {
            var sourceIndex = plan.Info.Buffers[buffer].Source;
            var source = plan.DescriptorSources[(int)sourceIndex];
            if (source.Dwords.Length == 4 && source.Dwords[0].Kind == ScalarValueKind.UserData)
            {
                var index = (int)(source.Dwords[0].UserDataRegister - plan.UserDataBase);
                if ((uint)index < plan.UserDataCount)
                {
                    candidates.Add(new(buffer, index));
                    allowedSources.Add(sourceIndex);
                }
            }
        }

        var blocked = new bool[plan.UserDataCount];
        var visited = new HashSet<ScalarValue>();
        void Block(ScalarValue value)
        {
            if (!visited.Add(value)) return;
            if (value.Kind == ScalarValueKind.UserData)
            {
                var index = (int)(value.UserDataRegister - plan.UserDataBase);
                if ((uint)index < blocked.Length) blocked[index] = true;
            }
            // A flattened-table reference has its own graph, not just operands.
            if (value.Kind == ScalarValueKind.ResourceTableWord && value.Payload < (ulong)plan.TableReads.Count)
                Block(plan.TableReads[(int)value.Payload].Value);
            foreach (var operand in value.Operands) Block(operand);
        }
        for (var source = 0; source < plan.DescriptorSources.Count; source++)
        {
            var words = plan.DescriptorSources[source].Dwords;
            var bufferOnly = allowedSources.Contains((uint)source) &&
                !plan.Info.Images.Any(image => image.Source == source) &&
                !plan.Info.Samplers.Any(sampler => sampler.Source == source);
            for (var word = bufferOnly ? 1 : 0; word < words.Length; word++) Block(words[word]);
        }
        foreach (var read in plan.TableReads) Block(read.Value);
        foreach (var read in plan.DynamicReads) Block(read);
        foreach (var branch in plan.ResourceBranches)
            if (branch.Condition is { } condition) Block(condition);
        foreach (var range in plan.DeviceAddressRanges) { Block(range.BaseLow); Block(range.BaseHigh); }

        // These specialization paths can synthesize additional descriptors and tables.
        var complex = plan.RequiresSpecializationMemory || plan.BufferCandidateTables.Count != 0 || plan.IndirectImages.Count != 0;
        Patches = complex ? [] : candidates.Where(patch => !blocked[patch.UserData]).ToArray();
        _patchable = new bool[plan.UserDataCount];
        foreach (var patch in Patches) _patchable[patch.UserData] = true;
    }

    internal bool IsPatchable(int index) => index < _patchable.Length && _patchable[index];

    internal bool Matches(uint[] captured, IReadOnlyList<uint> current, ulong[] used, out bool changed)
    {
        changed = false;
        if (captured.Length != current.Count) return false;
        for (var word = 0; word < captured.Length; word++)
        {
            if (captured[word] == current[word]) continue;
            // Unread roots only change the shader's inline data, not its resources.
            if (ResourceMaterializationCache.UsesWord(used, word) && !IsPatchable(word)) return false;
            changed = true;
        }
        return true;
    }

    internal ResourceSnapshot Apply(ResourceSnapshot template, uint[] userData)
    {
        var buffers = template.Buffers;
        foreach (var patch in Patches)
        {
            // Inactive descriptors are deliberately empty. Changing activity cannot be a
            // patch (the predicate was blocked), so they stay empty.
            if (buffers[patch.Buffer].Length == 0 || template.UserData[patch.UserData] == userData[patch.UserData]) continue;
            if (ReferenceEquals(buffers, template.Buffers)) buffers = (uint[][])buffers.Clone();
            buffers[patch.Buffer] = (uint[])buffers[patch.Buffer].Clone();
            buffers[patch.Buffer][0] = userData[patch.UserData];
        }
        return new ResourceSnapshot
        {
            Buffers = buffers, Images = template.Images, Samplers = template.Samplers,
            FlattenedResourceTable = template.FlattenedResourceTable, UserData = userData,
            DeviceAddressRanges = template.DeviceAddressRanges, BindlessImages = template.BindlessImages,
        };
    }

    internal static ulong Key(IReadOnlyList<uint> userData, int[] dynamicWords)
    {
        var key = 14695981039346656037UL;
        foreach (var word in dynamicWords) key = (key ^ userData[word]) * 1099511628211UL;
        return key;
    }

    internal bool CanPatch(ResourceSnapshot template, IReadOnlyList<uint> userData)
    {
        foreach (var patch in Patches)
        {
            if (template.UserData[patch.UserData] == userData[patch.UserData]) continue;
            var words = template.Buffers[patch.Buffer];
            // Specialization can normalize an invalid V# to a null descriptor. Patching
            // its base would resurrect a descriptor the full path deliberately cleared.
            if (words.Length != 4 || words[0] != template.UserData[patch.UserData] ||
                (words[0] | words[1] | words[2] | words[3]) == 0) return false;
        }
        return true;
    }
}
