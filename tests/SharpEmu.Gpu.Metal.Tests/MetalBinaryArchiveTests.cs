// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Metal;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The title's binary archive: a library and a pipeline taken from it compute and draw
/// what the ones compiled from source do, a damaged or mismatched file is discarded
/// instead of used, what is written reads back, a segment is written without the pipelines
/// the archiver refuses and opened only when a pipeline of its shader pair asks. Device work
/// is behind the exclusive device gate.
/// </summary>
public sealed class MetalBinaryArchiveTests(ITestOutputHelper output) : IDisposable
{
    private const nuint PixelFormatRgba8Unorm = 70;
    private const nuint PixelFormatRgba8UnormSrgb = 71;
    private const nuint PixelFormatBgra8Unorm = 80;
    private const nuint PixelFormatRgba16Float = 115;
    private const int Side = 32;
    private const uint Threads = 256;
    private const long Budget = 64L << 20;

    private const string ComputeSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void gen5_cs(device uint* out [[buffer(0)]], uint id [[thread_position_in_grid]])
        {
            float x = float(id) * 0.37f;
            out[id] = as_type<uint>(sin(x) * float(id) + cos(x * 1.7f)) ^ (id * 2654435761u);
        }
        """;

    private const string VertexSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct Gen5VsOut { float4 position [[position]]; float4 attr0 [[user(locn0)]]; };
        vertex Gen5VsOut gen5_vs(uint id [[vertex_id]])
        {
            const float2 corners[3] = { float2(-0.9f, -0.8f), float2(0.85f, -0.6f), float2(-0.2f, 0.9f) };
            Gen5VsOut out;
            out.position = float4(corners[id], 0.0f, 1.0f);
            out.attr0 = float4(corners[id] * 0.5f + 0.5f, 0.25f, 1.0f);
            return out;
        }
        """;

    private const string PixelSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct Gen5PsIn { float4 position [[position]]; float4 attr0 [[user(locn0)]]; };
        fragment float4 gen5_ps(Gen5PsIn in [[stage_in]])
        {
            return float4(sin(in.attr0.xyz * 3.1f) * 0.5f + 0.5f, 1.0f);
        }
        """;

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    private readonly string _root = Path.Combine(Path.GetTempPath(), $"sharpemu-binary-archive-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (System.IO.Directory.Exists(_root))
        {
            System.IO.Directory.Delete(_root, recursive: true);
        }
    }

    // A binary compiled in the safe math mode must not be served for the relaxed one.
    [Fact]
    public void TheMathModeIsPartOfTheLibraryKey()
    {
        var safe = MetalBinaryArchive.LibraryKey(ComputeSource, false, MslMathMode.Safe);
        var relaxed = MetalBinaryArchive.LibraryKey(ComputeSource, false, MslMathMode.Relaxed);
        Assert.NotEqual(safe, relaxed);
        Assert.Equal(relaxed, MetalBinaryArchive.LibraryKey(ComputeSource, false, MslMathMode.Relaxed));
        Assert.Equal(MetalBinaryArchive.LibraryKey(ComputeSource, false), MetalBinaryArchive.LibraryKey(ComputeSource, false, MslMathMode.Default));
    }

    [Fact]
    public void ALibraryAndComputePipelineFromTheArchiveComputeWhatTheCompiledOnesDo()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var key = MetalBinaryArchive.LibraryKey(ComputeSource, usesPerVertexInputs: false);
        var compiledLibrary = Compile(ComputeSource);
        var compiledFunction = Function(compiledLibrary, "gen5_cs");
        nint error = 0;
        var compiled = MetalNative.Send(device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), compiledFunction, ref error);
        Assert.NotEqual(0, compiled);
        // A miss fails even though macOS's cache has this pipeline now: a state made with
        // the option below came from the archive.
        var emptyDescriptor = New("MTLBinaryArchiveDescriptor");
        var empty = MetalNative.Send(device, MetalNative.Selector("newBinaryArchiveWithDescriptor:error:"), emptyDescriptor, ref error);
        Assert.NotEqual(0, empty);
        Assert.Equal(0, NewComputeFromArchive(compiledFunction, empty));
        Release(empty, emptyDescriptor);
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            Assert.Equal(0, writing.TryLoadLibrary(key));
            writing.CaptureCompute(key, compiledFunction);
            writing.Flush(wait: true);
            Assert.Equal(1, writing.Counters.LibrariesWritten);
        }

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        var storedLibrary = reading.TryLoadLibrary(key);
        Assert.NotEqual(0, storedLibrary);
        var storedFunction = Function(storedLibrary, "gen5_cs");
        var archive = reading.OpenLibraryArchive(key);
        Assert.NotEqual(0, archive);
        var stored = NewComputeFromArchive(storedFunction, archive);
        Assert.NotEqual(0, stored);

        var expected = RunCompute(compiled);
        var actual = RunCompute(stored);
        Assert.Contains(expected, static value => value != 0);
        Assert.Equal(expected, actual);
        Assert.Equal(1, reading.Counters.LibraryHits);
        Release(stored, archive, storedFunction, storedLibrary, compiled, compiledFunction, compiledLibrary);
    }

    [Fact]
    public void ARenderPipelineFromTheArchiveDrawsThePixelsTheCompiledOneDoes()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexKey = MetalBinaryArchive.LibraryKey(VertexSource, usesPerVertexInputs: false);
        var pixelKey = MetalBinaryArchive.LibraryKey(PixelSource, usesPerVertexInputs: false);
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var descriptor = RenderDescriptor(Function(vertexLibrary, "gen5_vs"), Function(pixelLibrary, "gen5_ps"));
        var compiled = NewRender(descriptor, 0, 0);
        Assert.NotEqual(0, compiled);
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
            Assert.Equal(0, writing.AcquirePipelineArchives(vertexKey, pixelKey));
            writing.NoteRenderPipeline(descriptor, MetalBinaryArchive.PipelineKind.Render, vertexKey, pixelKey, missed: true);
            writing.Flush(wait: true);
            Assert.Equal(1, writing.Counters.PipelinesAdded);
            Assert.Equal(2, writing.Counters.LibrariesWritten);
        }

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        Assert.True(reading.WaitLoaded(TimeSpan.FromSeconds(10)));
        Assert.Equal(1, reading.Counters.LoadedSegments);
        // Listed, not opened: nothing is read into memory until a pipeline of the pair asks.
        Assert.Equal(0, reading.SegmentCounters.SegmentsOpened);
        var storedVertexLibrary = reading.TryLoadLibrary(vertexKey);
        var storedPixelLibrary = reading.TryLoadLibrary(pixelKey);
        Assert.NotEqual(0, storedVertexLibrary);
        Assert.NotEqual(0, storedPixelLibrary);
        var storedDescriptor = RenderDescriptor(Function(storedVertexLibrary, "gen5_vs"), Function(storedPixelLibrary, "gen5_ps"));
        Assert.Equal(0, reading.AcquirePipelineArchives(pixelKey, vertexKey));
        var archives = reading.AcquirePipelineArchives(vertexKey, pixelKey);
        Assert.NotEqual(0, archives);
        Assert.Equal(1, reading.SegmentCounters.SegmentsOpened);
        // A miss fails instead of compiling: the state below came from the segment.
        var stored = NewRender(storedDescriptor, archives, MetalBinaryArchive.FailOnBinaryArchiveMiss);
        Assert.NotEqual(0, stored);

        var expected = Draw(compiled);
        var actual = Draw(stored);
        Assert.Contains(expected, static value => value != 0);
        Assert.Contains(expected, static value => value == 0);
        Assert.Equal(expected, actual);
        Release(stored, archives, storedDescriptor, storedPixelLibrary, storedVertexLibrary, compiled, descriptor, pixelLibrary, vertexLibrary);
    }

    // The shader prewarm skips a compute whose file is written (MetalShaderPrewarm): only a
    // written file counts, in this session and the next.
    [Fact]
    public void AComputeLibraryIsStoredOnlyOnceItsFileIsWritten()
    {
        if (!RequireDevice()) return;
        var computeKey = MetalBinaryArchive.LibraryKey(ComputeSource, usesPerVertexInputs: false);
        var otherKey = MetalBinaryArchive.LibraryKey(ComputeSource + "\n// another\n", usesPerVertexInputs: false);
        using (var writing = MetalBinaryArchive.Open(_root, Device.Value, "translator-a", Budget))
        {
            Assert.False(writing.IsStored(computeKey));
            Assert.True(Store(writing, ComputeSource, "gen5_cs", Gen5MslStage.Compute, computeKey) > 0);
            Assert.True(writing.IsStored(computeKey));
            Assert.False(writing.IsStored(otherKey));
            Assert.False(writing.IsStored(default));
        }

        using var reading = MetalBinaryArchive.Open(_root, Device.Value, "translator-a", Budget);
        Assert.True(reading.IsStored(computeKey));
        Assert.False(reading.IsStored(otherKey));
    }

    [Fact]
    public void LibrariesStoredAloneDrawAndComputeWhatTheCompiledOnesDo()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexKey = MetalBinaryArchive.LibraryKey(VertexSource, usesPerVertexInputs: false);
        var pixelKey = MetalBinaryArchive.LibraryKey(PixelSource, usesPerVertexInputs: false);
        var computeKey = MetalBinaryArchive.LibraryKey(ComputeSource, usesPerVertexInputs: false);
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var computeLibrary = Compile(ComputeSource);
        var descriptor = RenderDescriptor(Function(vertexLibrary, "gen5_vs"), Function(pixelLibrary, "gen5_ps"));
        var compiled = NewRender(descriptor, 0, 0);
        var computeFunction = Function(computeLibrary, "gen5_cs");
        nint error = 0;
        var compiledCompute = MetalNative.Send(device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), computeFunction, ref error);
        Assert.NotEqual(0, compiled);
        Assert.NotEqual(0, compiledCompute);
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            // The static precompile's paths: a vertex-only pipeline, a fragment behind a stub
            // vertex function made from its inputs, a compute pipeline; none is ever made.
            Assert.True(Store(writing, VertexSource, "gen5_vs", Gen5MslStage.Vertex, vertexKey) > 0);
            Assert.True(Store(writing, PixelSource, "gen5_ps", Gen5MslStage.Pixel, pixelKey) > 0);
            Assert.True(Store(writing, ComputeSource, "gen5_cs", Gen5MslStage.Compute, computeKey) > 0);
            // There already: not stored twice.
            Assert.Equal(-1, Store(writing, ComputeSource, "gen5_cs", Gen5MslStage.Compute, computeKey));
            Assert.Equal(3, writing.Stored.Libraries);
            Assert.True(writing.HasLibrary(vertexKey));
        }

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        var storedVertexLibrary = reading.TryLoadLibrary(vertexKey);
        var storedPixelLibrary = reading.TryLoadLibrary(pixelKey);
        var storedComputeLibrary = reading.TryLoadLibrary(computeKey);
        Assert.NotEqual(0, storedVertexLibrary);
        Assert.NotEqual(0, storedPixelLibrary);
        Assert.NotEqual(0, storedComputeLibrary);
        var storedDescriptor = RenderDescriptor(Function(storedVertexLibrary, "gen5_vs"), Function(storedPixelLibrary, "gen5_ps"));
        var stored = NewRender(storedDescriptor, 0, 0);
        Assert.NotEqual(0, stored);
        var storedFunction = Function(storedComputeLibrary, "gen5_cs");
        var storedCompute = MetalNative.Send(device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), storedFunction, ref error);
        Assert.NotEqual(0, storedCompute);

        var expected = Draw(compiled);
        Assert.Contains(expected, static value => value != 0);
        Assert.Equal(expected, Draw(stored));
        Assert.Equal(RunCompute(compiledCompute), RunCompute(storedCompute));
        Assert.Equal(3, reading.Counters.LibraryHits);
        Release(storedCompute, storedFunction, stored, storedDescriptor, storedComputeLibrary, storedPixelLibrary, storedVertexLibrary,
            compiledCompute, computeFunction, compiled, descriptor, computeLibrary, pixelLibrary, vertexLibrary);
    }

    [Fact]
    public void ADamagedLibraryFileIsDeletedInsteadOfLoaded()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var key = MetalBinaryArchive.LibraryKey(ComputeSource, usesPerVertexInputs: false);
        WriteComputeLibrary(device, key);
        string path;
        using (var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            path = Path.Combine(reading.LibraryDirectory, key.ToString("x32") + MetalBinaryArchive.Extension);
        }

        var bytes = File.ReadAllBytes(path);
        // Metal itself opens this one as a library: the AIR slice comes first and is whole.
        File.WriteAllBytes(path, bytes.AsSpan(0, bytes.Length / 2).ToArray());
        using (var truncated = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            Assert.Equal(0, truncated.TryLoadLibrary(key));
            Assert.Equal(1, truncated.Counters.Rejected);
            Assert.False(File.Exists(path));
        }

        WriteComputeLibrary(device, key);
        bytes = File.ReadAllBytes(path);
        bytes[bytes.Length / 3] ^= 0xFF;
        File.WriteAllBytes(path, bytes);
        using var flipped = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        Assert.Equal(0, flipped.TryLoadLibrary(key));
        Assert.Equal(1, flipped.Counters.Rejected);
        Assert.False(File.Exists(path));
    }

    [Fact]
    public void ASegmentOfAnotherTranslatorStillServesAPairWhoseShadersItDidNotChange()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexKey = MetalBinaryArchive.LibraryKey(VertexSource, usesPerVertexInputs: false);
        var pixelKey = MetalBinaryArchive.LibraryKey(PixelSource, usesPerVertexInputs: false);
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var descriptor = RenderDescriptor(Function(vertexLibrary, "gen5_vs"), Function(pixelLibrary, "gen5_ps"));
        var compiled = NewRender(descriptor, 0, 0);
        Assert.NotEqual(0, compiled);
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
            writing.NoteRenderPipeline(descriptor, MetalBinaryArchive.PipelineKind.Render, vertexKey, pixelKey, missed: true);
            writing.Flush(wait: true);
        }

        // The pair is keyed by both shaders' MSL, so the new translator, which made the same
        // MSL for them, finds the old build's pipeline instead of compiling it again.
        using var reading = MetalBinaryArchive.Open(_root, device, "translator-b", Budget);
        Assert.True(reading.WaitLoaded(TimeSpan.FromSeconds(10)));
        Assert.Equal(1, reading.Counters.LoadedSegments);
        var archives = reading.AcquirePipelineArchives(vertexKey, pixelKey);
        Assert.NotEqual(0, archives);
        var stored = NewRender(descriptor, archives, MetalBinaryArchive.FailOnBinaryArchiveMiss);
        Assert.NotEqual(0, stored);
        Assert.Equal(Draw(compiled), Draw(stored));
        Release(stored, archives, compiled, descriptor, pixelLibrary, vertexLibrary);
    }

    [Fact]
    public void AnArchiveOfAnotherDeviceIsDiscardedAndAnotherTranslatorsSegmentKept()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var descriptor = RenderDescriptor(Function(vertexLibrary, "gen5_vs"), Function(pixelLibrary, "gen5_ps"));
        var other = Path.Combine(_root, "0123456789abcdef");
        string directory;
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            directory = writing.Directory;
            Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
            writing.NoteRenderPipeline(descriptor, MetalBinaryArchive.PipelineKind.Render, default, default, missed: true);
            writing.Flush(wait: true);
        }

        var segments = System.IO.Directory.GetFiles(directory, MetalBinaryArchive.SegmentPrefix + "*");
        Assert.Single(segments);
        // What another device, OS or format wrote sits next to this one's directory.
        System.IO.Directory.CreateDirectory(other);
        File.WriteAllBytes(Path.Combine(other, "stale" + MetalBinaryArchive.Extension), [1, 2, 3]);

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-b", Budget);
        Assert.True(reading.WaitLoaded(TimeSpan.FromSeconds(10)));
        // Listed for the pairs it holds; the budget, not the stamp, decides when it goes.
        Assert.Equal(1, reading.Counters.LoadedSegments);
        Assert.True(File.Exists(segments[0]));
        Assert.False(System.IO.Directory.Exists(other));
        Release(descriptor, pixelLibrary, vertexLibrary);
    }

    [Fact]
    public void AllTheSegmentsStayOnDiskAndOnlyTheOnesAskedForAreOpen()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var vertex = Function(vertexLibrary, "gen5_vs");
        var pixel = Function(pixelLibrary, "gen5_ps");
        var formats = new nuint[] { PixelFormatRgba8Unorm, PixelFormatBgra8Unorm, PixelFormatRgba16Float };
        var pairs = new (UInt128 Vertex, UInt128 Pixel)[] { (1, 2), (3, 4), (5, 6) };
        for (var index = 0; index < formats.Length; index++)
        {
            // One start, one segment, one pipeline: the segments are as old as the starts.
            using var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
            Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
            var descriptor = RenderDescriptor(vertex, pixel, formats[index]);
            writing.NoteRenderPipeline(descriptor, MetalBinaryArchive.PipelineKind.Render, pairs[index].Vertex, pairs[index].Pixel, missed: true);
            writing.Flush(wait: true);
            Assert.Equal(1, writing.Counters.PipelinesAdded);
            Release(descriptor);
        }

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        Assert.True(reading.WaitLoaded(TimeSpan.FromSeconds(10)));
        var segments = System.IO.Directory.GetFiles(reading.Directory, MetalBinaryArchive.SegmentPrefix + "*" + MetalBinaryArchive.Extension);
        Assert.Equal(3, segments.Length);
        var segmentBytes = new FileInfo(segments[0]).Length;
        // Room for one and a half segments: a second open closes the first.
        reading.ResidentLimit = segmentBytes + segmentBytes / 2;
        Assert.Equal(3, reading.Counters.LoadedSegments);
        Assert.Equal(0, reading.SegmentCounters.SegmentsOpened);
        Assert.Equal(0, reading.AcquirePipelineArchives(7, 8));

        var checkDescriptor = RenderDescriptor(vertex, pixel, formats[0]);
        var first = reading.AcquirePipelineArchives(pairs[0].Vertex, pairs[0].Pixel);
        Assert.NotEqual(0, first);
        Assert.Equal(1, reading.SegmentCounters.SegmentsOpened);
        // The pair's own segment has the pipeline; another pair's does not.
        Assert.NotEqual(0, NewRender(checkDescriptor, first, MetalBinaryArchive.FailOnBinaryArchiveMiss));
        var other = reading.AcquirePipelineArchives(pairs[1].Vertex, pairs[1].Pixel);
        Assert.NotEqual(0, other);
        Assert.Equal(2, reading.SegmentCounters.SegmentsOpened);
        Assert.Equal(0, NewRender(checkDescriptor, other, MetalBinaryArchive.FailOnBinaryArchiveMiss));
        // The first was closed for the second; asked again it is opened again, from its file.
        var again = reading.AcquirePipelineArchives(pairs[0].Vertex, pairs[0].Pixel);
        Assert.NotEqual(0, again);
        Assert.Equal(3, reading.SegmentCounters.SegmentsOpened);
        Assert.NotEqual(0, NewRender(checkDescriptor, again, MetalBinaryArchive.FailOnBinaryArchiveMiss));
        // Nothing was deleted for being past the limit.
        Assert.Equal(3, System.IO.Directory.GetFiles(reading.Directory, MetalBinaryArchive.SegmentPrefix + "*" + MetalBinaryArchive.Extension).Length);
        Release(again, other, first, checkDescriptor, pixel, vertex, pixelLibrary, vertexLibrary);
    }

    [Fact]
    public void ASegmentWithoutItsPairsListIsDiscarded()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var descriptor = RenderDescriptor(Function(vertexLibrary, "gen5_vs"), Function(pixelLibrary, "gen5_ps"));
        string path;
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
            writing.NoteRenderPipeline(descriptor, MetalBinaryArchive.PipelineKind.Render, 1, 2, missed: true);
            writing.Flush(wait: true);
            path = Assert.Single(System.IO.Directory.GetFiles(writing.Directory, MetalBinaryArchive.SegmentPrefix + "*"));
        }

        // The file as an earlier build left it: the archive and the trailer, no list.
        Assert.True(MetalBinaryArchive.TryReadPairs(path, out var pairs, out var length));
        Assert.Single(pairs);
        var bytes = File.ReadAllBytes(path);
        File.WriteAllBytes(path, bytes.AsSpan(0, (int)length - MetalBinaryArchive.TrailerBytes - 8 - 8).ToArray());
        MetalBinaryArchive.AppendTrailer(path);
        Assert.True(MetalBinaryArchive.TryVerify(path, out _));
        Assert.False(MetalBinaryArchive.TryReadPairs(path, out _, out _));

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        Assert.True(reading.WaitLoaded(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, reading.Counters.LoadedSegments);
        Assert.Equal(1, reading.Counters.Rejected);
        Assert.False(File.Exists(path));
        Release(descriptor, pixelLibrary, vertexLibrary);
    }

    [Fact]
    public void APipelineTheSegmentHoldsAlreadyLeavesTheSegmentOpenAndWrittenOnce()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var vertex = Function(vertexLibrary, "gen5_vs");
        var pixel = Function(pixelLibrary, "gen5_ps");
        var first = RenderDescriptor(vertex, pixel, PixelFormatRgba8Unorm);
        var same = RenderDescriptor(vertex, pixel, PixelFormatRgba8Unorm);
        var next = RenderDescriptor(vertex, pixel, PixelFormatBgra8Unorm);
        using var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
        writing.NoteRenderPipeline(first, MetalBinaryArchive.PipelineKind.Render, 1, 2, missed: true);
        writing.Flush(wait: true);
        var segment = Assert.Single(System.IO.Directory.GetFiles(writing.Directory, MetalBinaryArchive.SegmentPrefix + "*"));
        var written = File.ReadAllBytes(segment);
        // The archive is given what it holds: serializeToURL reports success and writes no file.
        writing.NoteRenderPipeline(same, MetalBinaryArchive.PipelineKind.Render, 1, 2, missed: true);
        writing.Flush(wait: true);
        Assert.Equal(written, File.ReadAllBytes(segment));
        // The segment was not closed by it: the next pipeline joins the first in its file.
        writing.NoteRenderPipeline(next, MetalBinaryArchive.PipelineKind.Render, 1, 2, missed: true);
        writing.Flush(wait: true);
        Assert.Equal(3, writing.Counters.PipelinesAdded);
        Assert.Equal(segment, Assert.Single(System.IO.Directory.GetFiles(writing.Directory, MetalBinaryArchive.SegmentPrefix + "*")));
        Assert.True(new FileInfo(segment).Length > written.Length);
        Assert.Empty(System.IO.Directory.GetFiles(writing.Directory, "*.tmp"));
        Release(next, same, first, pixel, vertex, pixelLibrary, vertexLibrary);
    }

    [Fact]
    public void ASegmentIsWrittenWithoutThePipelinesTheArchiverRefuses()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var vertex = Function(vertexLibrary, "gen5_vs");
        var pixel = Function(pixelLibrary, "gen5_ps");
        var formats = new nuint[] { PixelFormatRgba8Unorm, PixelFormatBgra8Unorm, PixelFormatRgba8UnormSrgb, PixelFormatRgba16Float };
        var descriptors = formats.Select(format => RenderDescriptor(vertex, pixel, format)).ToArray();
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
            writing.RefusedByArchiver = descriptor => descriptor == descriptors[2];
            foreach (var descriptor in descriptors)
            {
                writing.NoteRenderPipeline(descriptor, MetalBinaryArchive.PipelineKind.Render, 1, 2, missed: true);
            }

            writing.Flush(wait: true);
            Assert.Equal(4, writing.Counters.PipelinesAdded);
            Assert.Equal(1, writing.SegmentCounters.PipelinesRefused);
            Assert.Single(System.IO.Directory.GetFiles(writing.Directory, MetalBinaryArchive.SegmentPrefix + "*"));
        }

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        Assert.True(reading.WaitLoaded(TimeSpan.FromSeconds(10)));
        Assert.Equal(1, reading.Counters.LoadedSegments);
        var archives = reading.AcquirePipelineArchives(1, 2);
        Assert.NotEqual(0, archives);
        for (var index = 0; index < descriptors.Length; index++)
        {
            var state = NewRender(descriptors[index], archives, MetalBinaryArchive.FailOnBinaryArchiveMiss);
            // The refused one is the only one the segment does not have.
            Assert.Equal(index != 2, state != 0);
            Release(state);
        }

        Release(archives);
        Release(descriptors);
        Release(pixel, vertex, pixelLibrary, vertexLibrary);
    }

    // The archiver refused a whole segment of Black Myth: Wukong's gameplay pipelines though it took
    // every part of it, and the segment was dropped, so each run built those pipelines again. It
    // is written in pieces instead, each a segment of its own, and the next run finds them all.
    [Fact]
    public void ASegmentTheArchiverRefusesOnlyWholeIsWrittenInPiecesAndKeepsEveryPipeline()
    {
        if (!RequireDevice()) return;
        var device = Device.Value;
        var vertexLibrary = Compile(VertexSource);
        var pixelLibrary = Compile(PixelSource);
        var vertex = Function(vertexLibrary, "gen5_vs");
        var pixel = Function(pixelLibrary, "gen5_ps");
        var formats = new nuint[] { PixelFormatRgba8Unorm, PixelFormatBgra8Unorm, PixelFormatRgba8UnormSrgb, PixelFormatRgba16Float };
        var descriptors = formats.Select(format => RenderDescriptor(vertex, pixel, format)).ToArray();
        using (var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget))
        {
            Assert.True(writing.WaitLoaded(TimeSpan.FromSeconds(10)));
            writing.RefusesArchivesOver = 2;
            foreach (var descriptor in descriptors)
            {
                writing.NoteRenderPipeline(descriptor, MetalBinaryArchive.PipelineKind.Render, 1, 2, missed: true);
            }

            writing.Flush(wait: true);
            Assert.Equal(4, writing.Counters.PipelinesAdded);
            Assert.Equal(0, writing.SegmentCounters.PipelinesRefused);
            Assert.Equal(2, System.IO.Directory.GetFiles(writing.Directory, MetalBinaryArchive.SegmentPrefix + "*").Length);
        }

        using var reading = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        Assert.True(reading.WaitLoaded(TimeSpan.FromSeconds(10)));
        Assert.Equal(2, reading.Counters.LoadedSegments);
        var archives = reading.AcquirePipelineArchives(1, 2);
        Assert.NotEqual(0, archives);
        foreach (var descriptor in descriptors)
        {
            var state = NewRender(descriptor, archives, MetalBinaryArchive.FailOnBinaryArchiveMiss);
            Assert.NotEqual(0, state);
            Release(state);
        }

        Release(archives);
        Release(descriptors);
        Release(pixel, vertex, pixelLibrary, vertexLibrary);
    }

    [Fact]
    public void ASegmentsTailListsItsPairsWithoutBeingTheOneThatIsChecked()
    {
        System.IO.Directory.CreateDirectory(_root);
        var path = Path.Combine(_root, "segment" + MetalBinaryArchive.Extension);
        var payload = new byte[4096];
        new Random(11).NextBytes(payload);
        File.WriteAllBytes(path, payload);
        Assert.False(MetalBinaryArchive.TryReadPairs(path, out _, out _));

        MetalBinaryArchive.AppendTrailer(path, [3UL, 1UL << 63, 7UL]);

        Assert.True(MetalBinaryArchive.TryVerify(path, out var length));
        Assert.True(MetalBinaryArchive.TryReadPairs(path, out var pairs, out var listedLength));
        Assert.Equal(length, listedLength);
        Assert.Equal([3UL, 1UL << 63, 7UL], pairs);
        Assert.Equal(payload, File.ReadAllBytes(path).AsSpan(0, payload.Length).ToArray());
        // Only opening the segment checks the whole file.
        var bytes = File.ReadAllBytes(path);
        bytes[100] ^= 1;
        File.WriteAllBytes(path, bytes);
        Assert.False(MetalBinaryArchive.TryVerify(path, out _));
        Assert.True(MetalBinaryArchive.TryReadPairs(path, out _, out _));
        File.WriteAllBytes(path, bytes.AsSpan(0, bytes.Length - 1).ToArray());
        Assert.False(MetalBinaryArchive.TryReadPairs(path, out _, out _));
        Assert.NotEqual(MetalBinaryArchive.PairKey(1, 2), MetalBinaryArchive.PairKey(2, 1));
    }

    [Fact]
    public void TheTrailerRoundTripsAndRejectsAnyChange()
    {
        System.IO.Directory.CreateDirectory(_root);
        var path = Path.Combine(_root, "payload" + MetalBinaryArchive.Extension);
        var payload = new byte[4096];
        new Random(7).NextBytes(payload);
        File.WriteAllBytes(path, payload);

        MetalBinaryArchive.AppendTrailer(path);

        Assert.True(MetalBinaryArchive.TryVerify(path, out var length));
        Assert.Equal(payload.Length + MetalBinaryArchive.TrailerBytes, length);
        Assert.Equal(payload, File.ReadAllBytes(path).AsSpan(0, payload.Length).ToArray());
        var bytes = File.ReadAllBytes(path);
        bytes[100] ^= 1;
        File.WriteAllBytes(path, bytes);
        Assert.False(MetalBinaryArchive.TryVerify(path, out _));
        File.WriteAllBytes(path, payload);
        Assert.False(MetalBinaryArchive.TryVerify(path, out _));
        File.WriteAllBytes(path, []);
        Assert.False(MetalBinaryArchive.TryVerify(path, out _));
    }

    [Fact]
    public void TheStubVertexFunctionOutputsTheFragmentsInterpolatedInputs()
    {
        const string fragment = """
            struct Gen5PsIn
            {
                float4 sharpemu_frag_coord [[position]];
                float4 attr0 [[user(locn0), flat]];
                vertex_value<float4> attr1 [[user(locn3)]];
                float3 sharpemu_bary_persp [[barycentric_coord, center_perspective]];
                bool sharpemu_front_facing [[front_facing]];
            };
            fragment Gen5PsOut gen5_ps(Gen5PsIn sharpemu_in [[stage_in]])
            """;
        var stub = MetalStaticPrecompile.StubVertexSource(fragment);
        Assert.NotNull(stub);
        Assert.Contains("float4 sharpemu_frag_coord [[position]];", stub);
        Assert.Contains("float4 attr0 [[user(locn0)]];", stub);
        Assert.Contains("float4 attr1 [[user(locn3)]];", stub);
        Assert.DoesNotContain("barycentric", stub);
        Assert.DoesNotContain("front_facing", stub);
        Assert.Null(MetalStaticPrecompile.StubVertexSource("kernel void gen5_cs() {}"));
    }

    private static long Store(MetalBinaryArchive archive, string source, string entry, Gen5MslStage stage, UInt128 key)
    {
        var bytes = MetalStaticPrecompile.Store(archive, Device.Value, new Gen5MslShader(source, entry, stage, 1), key, out var error);
        Assert.True(error is null, error);
        return bytes;
    }

    private void WriteComputeLibrary(nint device, UInt128 key)
    {
        var library = Compile(ComputeSource);
        var function = Function(library, "gen5_cs");
        using var writing = MetalBinaryArchive.Open(_root, device, "translator-a", Budget);
        writing.CaptureCompute(key, function);
        writing.Flush(wait: true);
        Release(function, library);
    }

    private static nint Compile(string source)
    {
        var library = MetalRenderHost.TryCompileLibrary(Device.Value, source, usesPerVertexInputs: false, out var error);
        Assert.True(library != 0, error);
        return library;
    }

    private static nint Function(nint library, string name)
    {
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(name));
        Assert.NotEqual(0, function);
        return function;
    }

    private static nint NewComputeFromArchive(nint function, nint archive)
    {
        var descriptor = New("MTLComputePipelineDescriptor");
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setComputeFunction:"), function);
        MetalNative.SendVoid(
            descriptor,
            MetalNative.Selector("setBinaryArchives:"),
            MetalNative.Send(MetalNative.Class("NSArray"), MetalNative.Selector("arrayWithObject:"), archive));
        nint error = 0;
        var state = MetalNative.SendNewMeshPipeline(
            Device.Value,
            MetalNative.Selector("newComputePipelineStateWithDescriptor:options:reflection:error:"),
            descriptor,
            MetalBinaryArchive.FailOnBinaryArchiveMiss,
            0,
            ref error);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("release"));
        return state;
    }

    private static nint RenderDescriptor(nint vertex, nint fragment, nuint format = PixelFormatRgba8Unorm)
    {
        var descriptor = New("MTLRenderPipelineDescriptor");
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), vertex);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setFragmentFunction:"), fragment);
        var attachment = MetalNative.SendAtIndex(
            MetalNative.Send(descriptor, MetalNative.Selector("colorAttachments")), MetalNative.Selector("objectAtIndexedSubscript:"), 0);
        MetalNative.SendVoidNUInt(attachment, MetalNative.Selector("setPixelFormat:"), format);
        return descriptor;
    }

    private static nint NewRender(nint descriptor, nint archives, nuint options)
    {
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setBinaryArchives:"), archives);
        nint error = 0;
        var state = MetalNative.SendNewMeshPipeline(
            Device.Value,
            MetalNative.Selector("newRenderPipelineStateWithDescriptor:options:reflection:error:"),
            descriptor,
            options,
            0,
            ref error);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setBinaryArchives:"), 0);
        return state;
    }

    private static uint[] RunCompute(nint state)
    {
        var device = Device.Value;
        var queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        var buffer = MetalNative.SendNewBuffer(device, MetalNative.Selector("newBufferWithLength:options:"), Threads * sizeof(uint), 0);
        var commands = MetalNative.Send(queue, MetalNative.Selector("commandBuffer"));
        var encoder = MetalNative.Send(commands, MetalNative.Selector("computeCommandEncoder"));
        MetalNative.SendVoid(encoder, MetalNative.Selector("setComputePipelineState:"), state);
        MetalNative.SendSetBuffer(encoder, MetalNative.Selector("setBuffer:offset:atIndex:"), buffer, 0, 0);
        MetalNative.SendDispatch(
            encoder,
            MetalNative.Selector("dispatchThreads:threadsPerThreadgroup:"),
            new MtlSize { Width = Threads, Height = 1, Depth = 1 },
            new MtlSize { Width = 64, Height = 1, Depth = 1 });
        MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
        MetalNative.SendVoid(commands, MetalNative.Selector("commit"));
        MetalNative.SendVoid(commands, MetalNative.Selector("waitUntilCompleted"));
        uint[] result;
        unsafe
        {
            result = new ReadOnlySpan<uint>((void*)MetalNative.Send(buffer, MetalNative.Selector("contents")), (int)Threads).ToArray();
        }

        Release(buffer, queue);
        return result;
    }

    private static byte[] Draw(nint state)
    {
        var device = Device.Value;
        var queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        var textureDescriptor = MetalNative.SendTextureDescriptor(
            MetalNative.Class("MTLTextureDescriptor"),
            MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            PixelFormatRgba8Unorm,
            Side,
            Side,
            false);
        MetalNative.SendVoidNUInt(textureDescriptor, MetalNative.Selector("setUsage:"), 4 | 1);
        MetalNative.SendVoidNUInt(textureDescriptor, MetalNative.Selector("setStorageMode:"), 0);
        var texture = MetalNative.Send(device, MetalNative.Selector("newTextureWithDescriptor:"), textureDescriptor);
        var pass = MetalNative.Send(MetalNative.Class("MTLRenderPassDescriptor"), MetalNative.Selector("renderPassDescriptor"));
        var attachment = MetalNative.SendAtIndex(
            MetalNative.Send(pass, MetalNative.Selector("colorAttachments")), MetalNative.Selector("objectAtIndexedSubscript:"), 0);
        MetalNative.SendVoid(attachment, MetalNative.Selector("setTexture:"), texture);
        MetalNative.SendVoidNUInt(attachment, MetalNative.Selector("setLoadAction:"), 2);
        MetalNative.SendVoidClearColor(attachment, MetalNative.Selector("setClearColor:"), default);
        MetalNative.SendVoidNUInt(attachment, MetalNative.Selector("setStoreAction:"), 1);
        var commands = MetalNative.Send(queue, MetalNative.Selector("commandBuffer"));
        var encoder = MetalNative.Send(commands, MetalNative.Selector("renderCommandEncoderWithDescriptor:"), pass);
        MetalNative.SendVoid(encoder, MetalNative.Selector("setRenderPipelineState:"), state);
        MetalNative.SendDrawPrimitivesInstanced(encoder, MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:instanceCount:"), 3, 0, 3, 1);
        MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
        MetalNative.SendVoid(commands, MetalNative.Selector("commit"));
        MetalNative.SendVoid(commands, MetalNative.Selector("waitUntilCompleted"));
        var pixels = new byte[Side * Side * 4];
        unsafe
        {
            fixed (byte* bytes = pixels)
            {
                SendGetBytes(
                    texture,
                    MetalNative.Selector("getBytes:bytesPerRow:fromRegion:mipmapLevel:"),
                    (nint)bytes,
                    Side * 4,
                    new MtlRegion { Width = Side, Height = Side, Depth = 1 },
                    0);
            }
        }

        Release(texture, queue);
        return pixels;
    }

    private static nint New(string className) =>
        MetalNative.Send(MetalNative.Send(MetalNative.Class(className), MetalNative.Selector("alloc")), MetalNative.Selector("init"));

    private static void Release(params nint[] objects)
    {
        foreach (var value in objects)
        {
            if (value != 0)
            {
                MetalNative.SendVoid(value, MetalNative.Selector("release"));
            }
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Archive pipelines are built on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    [DllImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static extern void SendGetBytes(
        nint receiver, nint selector, nint bytes, nuint bytesPerRow, MtlRegion region, nuint mipmapLevel);
}
