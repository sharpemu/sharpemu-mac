// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

// The hold-until-done and precompile-only modes: what the environment asks for, the close
// of the session once everything is compiled, and a library list compiled to completion
// and released rather than kept, with the footprint it took.
public sealed class MetalShaderPrecompileTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(null, null, (int)MetalPrecompileMode.Recorded)]
    [InlineData(null, "all", (int)MetalPrecompileMode.Recorded)]
    [InlineData(null, "first", (int)MetalPrecompileMode.Off)]
    [InlineData(null, "FIRST", (int)MetalPrecompileMode.Off)]
    [InlineData(null, "complete", (int)MetalPrecompileMode.Hold)]
    [InlineData(null, "COMPLETE", (int)MetalPrecompileMode.Hold)]
    [InlineData("1", null, (int)MetalPrecompileMode.Exit)]
    [InlineData("true", "complete", (int)MetalPrecompileMode.Exit)]
    [InlineData("0", "complete", (int)MetalPrecompileMode.Hold)]
    public void TheModeComesFromTheEnvironment(string? precompileOnly, string? hold, int expected) =>
        Assert.Equal((MetalPrecompileMode)expected, MetalShaderPrecompile.Parse(precompileOnly, hold));

    [Fact]
    public void PrecompileOnlyClosesTheSessionOnceEverythingIsCompiled()
    {
        var exits = 0;
        var footprints = new Queue<ulong>([100, 300, 200, 150, 120]);
        var precompile = new MetalShaderPrecompile(
            MetalPrecompileMode.Exit, () => exits++, () => (footprints.Count > 1 ? footprints.Dequeue() : footprints.Peek(), 999));

        precompile.BeginPhase("programs", 2);
        precompile.Advance();
        precompile.Advance();
        precompile.BeginPhase("libraries", 1);
        precompile.Advance();
        Assert.Equal(0, exits);

        Assert.True(precompile.Complete("detail"));
        Assert.Equal(1, exits);
        Assert.True(precompile.Completed);
        Assert.Equal(300UL, precompile.PeakFootprint);
        // A later caller keeps the guest held too, and the session is closed only once.
        Assert.True(precompile.Complete("again"));
        Assert.Equal(1, exits);
    }

    [Fact]
    public void HoldUntilDoneLetsTheGuestGoAfterwards()
    {
        var exits = 0;
        var precompile = new MetalShaderPrecompile(MetalPrecompileMode.Hold, () => exits++, () => (1, 1));
        Assert.True(precompile.HoldsUntilComplete);
        Assert.False(precompile.Complete("detail"));
        Assert.Equal(0, exits);
    }

    // The default holds for what past sessions recorded, not for the static precompile.
    [Fact]
    public void TheRecordedModeHoldsForTheRecordedProgramsAndNotTheStaticPrecompile()
    {
        var exits = 0;
        var precompile = new MetalShaderPrecompile(MetalPrecompileMode.Recorded, () => exits++, () => (1, 1));
        Assert.True(precompile.HoldsUntilComplete);
        Assert.False(precompile.HoldsForStatic);
        Assert.True(new MetalShaderPrecompile(MetalPrecompileMode.Hold, () => { }, () => (1, 1)).HoldsForStatic);
        Assert.False(precompile.Complete("detail"));
        Assert.True(precompile.Completed);
        Assert.Equal(0, exits);
    }

    [Fact]
    public void WithoutAModeNothingIsHeldOrClosed()
    {
        var exits = 0;
        var precompile = new MetalShaderPrecompile(MetalPrecompileMode.Off, () => exits++, () => (1, 1));
        Assert.False(precompile.HoldsUntilComplete);
        Assert.False(precompile.Complete("detail"));
        Assert.False(precompile.Completed);
        Assert.Equal(0, exits);
    }

    // On the device: a library list compiled to completion through the precompile path
    // leaves no library resident, and a module asking for a listed source afterwards is
    // told to compile it itself (a hit in macOS's shader cache). Reports the footprint.
    [Fact]
    public void ALibraryListPrecompilesToCompletionAndKeepsNothingResident()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("The library precompile runs on the device only behind the exclusive device gate.");
            return;
        }

        Assert.True(OperatingSystem.IsMacOS());
        var device = MetalNative.MTLCreateSystemDefaultDevice();
        Assert.NotEqual(0, device);

        const int Sources = 24;
        var directory = Path.Combine(Path.GetTempPath(), "sharpemu-precompile-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var stamp = "precompile-test|" + Guid.NewGuid().ToString("N");
        var unique = Guid.NewGuid().ToString("N");
        var sources = Enumerable.Range(0, Sources)
            .Select(index =>
                "#include <metal_stdlib>\nusing namespace metal;\n" +
                $"kernel void k_{unique}_{index}(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) " +
                $"{{ o[i] = i * {index + 1}u + {index}u; }}\n")
            .ToArray();
        WriteList(Path.Combine(directory, MetalLibraryPrewarm.FileName(stamp)), sources);

        var keep = MetalLibraryPrewarm.KeepLibraries;
        var (startFootprint, _) = MetalHostMemory.Footprint();
        var peak = startFootprint;
        try
        {
            MetalLibraryPrewarm.KeepLibraries = false;
            MetalLibraryPrewarm.Open(directory, device, stamp);
            Assert.Equal(Sources, MetalLibraryPrewarm.ListedSources);

            var compiled = 0;
            var taken = MetalLibraryPrewarm.Precompile(() =>
            {
                Interlocked.Increment(ref compiled);
                var (current, _) = MetalHostMemory.Footprint();
                lock (sources)
                {
                    peak = Math.Max(peak, current);
                }
            });

            Assert.Equal(Sources, taken);
            Assert.Equal(Sources, Volatile.Read(ref compiled));
            Assert.Equal(0, MetalLibraryPrewarm.HeldLibraries);
            Assert.False(MetalLibraryPrewarm.TryTake(MetalLibraryPrewarm.Key(sources[0], false), out var library));
            Assert.Equal(0, library);
            // The list began: a later Begin compiles nothing again.
            Assert.Equal(0, MetalLibraryPrewarm.Precompile(null));
        }
        finally
        {
            MetalLibraryPrewarm.Stop();
            MetalLibraryPrewarm.KeepLibraries = keep;
            Directory.Delete(directory, recursive: true);
        }

        var (endFootprint, lifetimePeak) = MetalHostMemory.Footprint();
        Assert.True(endFootprint > 0);
        Assert.True(lifetimePeak >= endFootprint);
        output.WriteLine(
            $"precompile of {Sources} libraries: footprint start={startFootprint >> 20} MB peak={peak >> 20} MB " +
            $"end={endFootprint >> 20} MB process_peak={lifetimePeak >> 20} MB");
    }

    private static void WriteList(string path, IReadOnlyList<string> sources)
    {
        using var file = new FileStream(path, FileMode.Create, FileAccess.Write);
        Span<byte> header = stackalloc byte[8];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header, 0x4C4C534D);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header[4..], 1);
        file.Write(header);
        MetalLibraryPrewarm.WriteRecord(file, [1]);
        foreach (var source in sources)
        {
            MetalLibraryPrewarm.WriteRecord(
                file,
                MetalLibraryPrewarm.SourceBody(MetalLibraryPrewarm.Key(source, false), false, MetalLibraryPrewarm.Compress(source)));
        }
    }
}
