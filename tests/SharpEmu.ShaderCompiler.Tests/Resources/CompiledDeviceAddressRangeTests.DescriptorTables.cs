// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Linq;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed partial class CompiledDeviceAddressRangeTests
{
    // A store through V#s a loop loads from a table (DeviceDescriptorTableTests.BinningProgram) once the SRT compiler has
    // built the plan's ranges: the written buffers the table selects, as the interpreter gives them. The compiled ranges
    // had no device-descriptor table tail and gave none, so the store's writes were never pending.
    [Theory]
    [InlineData(3u)]
    [InlineData((uint)DeviceAddressRangePlanner.MaxDescriptorTableEntries + 1)]
    public void DescriptorTableStores_KeepTheirWrittenRangesOnceCompiled(uint count)
    {
        (DeviceAddressRange[] Ranges, string? Refusal) Written(Path path)
        {
            using var scope = new Scope(path);
            var plan = Extract(DeviceDescriptorTableTests.BinningProgram());
            Assert.Single(plan.DeviceDescriptorTables);
            plan.CompileEvaluatorNow();
            var memory = DeviceDescriptorTableTests.TableMemory(count);
            var ranges = DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)DeviceDescriptorTableTests.Table, 0], memory.Read))
                .Where(range => range.Written).ToArray();
            return (ranges, (plan.CompiledRanges as CompiledDeviceAddressRanges)?.Refusal);
        }

        var refusals = ResourcePlanBuildCounters.DescriptorTableRefusals;
        var interpreted = Written(Path.Off);
        var compiled = Written(Path.On);

        Assert.NotEmpty(interpreted.Ranges);
        Assert.Equal(
            interpreted.Ranges.Select(range => (range.Base, range.Size, range.Planned)).ToArray(),
            compiled.Ranges.Select(range => (range.Base, range.Size, range.Planned)).ToArray());
        Assert.Equal(CompiledDeviceAddressRanges.DescriptorTablesRefusal, compiled.Refusal);
        Assert.True(ResourcePlanBuildCounters.DescriptorTableRefusals > refusals);
    }
}
