// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private sealed record HotLdsRegion(uint First, uint Count, uint Packed, int Weight);
        private readonly HotLdsRegion[] _meshHotLds = [];
        private uint MeshHotLdsWords => _meshHotLds.Aggregate(0u, (sum, region) => sum + region.Count);

        // This is an address-space partition, not a cache: a normalized guest word
        // always has one physical home, regardless of the opcode/offset used to name it.
        // The chosen ranges affect performance only. No address-range proof is needed
        // because every other word still lives in the original full device allocation.
        private HotLdsRegion[] PlanMeshHotLds()
        {
            if (!_options.MeshHotLds || _options.MeshLdsForwarding ||
                _meshInfo is not { DeviceLds: true, ThreadsPerThreadgroup: 32 } mesh ||
                _request.Memory.Entries.Any(entry => entry.AddressSpace is FlatAddressSpace.Shared or FlatAddressSpace.SharedOrPrivate) ||
                _request.Program.Instructions.Any(instruction => instruction.Control is Gen5BufferMemoryControl { Lds: true }))
                return [];

            var intervals = new List<HotLdsRegion>();
            var words = System.Numerics.BitOperations.RoundUpToPowerOf2(_meshLdsDwords);
            foreach (var instruction in _request.Program.Instructions)
            {
                if (instruction.Control is not Gen5DataShareControl control || control.Gds) continue;
                if (instruction.Opcode is "DsSwizzleB32" or "DsBpermuteB32" or "DsNop") continue;
                if (instruction.Opcode is not ("DsWriteB32" or "DsReadB32" or "DsWriteB128" or "DsReadB128"))
                    return [];
                var width = instruction.Opcode.EndsWith("B128", StringComparison.Ordinal) ? 4u : 1u;
                var first = (control.SingleOffsetBytes >> 2) & (words - 1);
                intervals.Add(new(first, Math.Min(32u * width, words - first), 0, (int)width));
            }

            var merged = new List<HotLdsRegion>();
            foreach (var region in intervals.OrderBy(region => region.First))
            {
                if (merged.Count != 0 && region.First <= merged[^1].First + merged[^1].Count)
                {
                    var previous = merged[^1];
                    merged[^1] = previous with
                    {
                        Count = Math.Max(previous.First + previous.Count, region.First + region.Count) - previous.First,
                        Weight = previous.Weight + region.Weight,
                    };
                }
                else merged.Add(region);
            }

            // Leave ample room for the mesh outputs and the existing rendezvous state.
            var outputBytes = mesh.MaxVertices * MeshVertexBytesBound(_request) +
                mesh.MaxPrimitives * MeshPrimitiveBytesBound(_request);
            var budget = outputBytes + 256 >= MeshOutputBudgetBytes ? 0u :
                Math.Min(3072u, MeshOutputBudgetBytes - outputBytes - 256u) / sizeof(uint);
            var chosen = new List<HotLdsRegion>();
            uint packed = 0;
            foreach (var region in merged.OrderByDescending(region => region.Weight).ThenBy(region => region.First))
            {
                if (region.Count > budget - packed) continue;
                chosen.Add(region with { Packed = packed });
                packed += region.Count;
            }
            return chosen.ToArray();
        }

        private string HotLdsContains(HotLdsRegion region, string index, uint count = 1) =>
            $"({index} - {region.First}u) <= {region.Count - count}u";

        private string HotLdsIndex(HotLdsRegion region, string index) =>
            $"({index} - {region.First}u + {region.Packed}u)";

        private string ReadHotOrDeviceLds(string index)
        {
            var value = $"sharpemu_lds[{index}]";
            foreach (var region in _meshHotLds.Reverse())
                value = $"({HotLdsContains(region, index)} ? sharpemu_hot_lds[{HotLdsIndex(region, index)}] : {value})";
            return value;
        }

        private void WriteHotOrDeviceLds(string index, string value)
        {
            foreach (var region in _meshHotLds)
                Line($"if ({HotLdsContains(region, index)}) {{ sharpemu_hot_lds[{HotLdsIndex(region, index)}] = {value}; }} else");
            Line($"{{ sharpemu_lds[{index}] = {value}; }}");
        }

        private bool TryEmitMeshHotLds(Gen5ShaderInstruction instruction, Gen5DataShareControl control)
        {
            if (_meshHotLds.Length == 0 || instruction.Opcode is not
                ("DsWriteB32" or "DsReadB32" or "DsWriteB128" or "DsReadB128")) return false;

            var count = instruction.Opcode.EndsWith("B128", StringComparison.Ordinal) ? 4u : 1u;
            var write = instruction.Opcode.StartsWith("DsWrite", StringComparison.Ordinal);
            var address = Temp("uint", RawSource(instruction, 0));
            var index = Temp("uint", LdsIndex(address, control.SingleOffsetBytes));
            var nominal = (control.SingleOffsetBytes >> 2) &
                (System.Numerics.BitOperations.RoundUpToPowerOf2(_meshLdsDwords) - 1);
            var primary = _meshHotLds.FirstOrDefault(region => region.Count >= count &&
                nominal >= region.First && nominal - region.First <= region.Count - count);
            BeginExecGuard();
            var values = Temp("uint4", write
                ? $"uint4({string.Join(", ", Enumerable.Range(0, 4).Select(i => i < count ? RawSource(instruction, i + 1) : "0u"))})"
                : "uint4(0u)");
            if (primary is not null)
            {
                Line($"if ({HotLdsContains(primary, index, count)})");
                Line("{");
                _indent++;
                for (uint i = 0; i < count; i++)
                {
                    var element = $"sharpemu_hot_lds[{HotLdsIndex(primary, index)} + {i}u]";
                    Line(write ? $"{element} = {values}[{i}];" : $"{values}[{i}] = {element};");
                }
                _indent--;
                Line("} else");
                Line("{");
                _indent++;
            }
            for (uint i = 0; i < count; i++)
            {
                var componentIndex = Temp("uint", LdsIndex(address, control.SingleOffsetBytes + i * sizeof(uint)));
                if (write) WriteHotOrDeviceLds(componentIndex, $"{values}[{i}]");
                else Line($"{values}[{i}] = {ReadHotOrDeviceLds(componentIndex)};");
            }
            if (primary is not null)
            {
                _indent--;
                Line("}");
            }
            if (!write)
                for (uint i = 0; i < count; i++)
                    StoreVector(instruction.Destinations[(int)i].Value, $"{values}[{i}]");
            EndExecGuard();
            return true;
        }
    }
}
