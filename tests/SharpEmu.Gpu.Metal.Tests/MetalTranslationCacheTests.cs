// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Metal.Tests;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The translation disk cache on Metal: a session keeps each translation, and a later session over the same folder
/// takes every shader from it without the translator, as the shader a translation gives (reflection, request, the
/// library key the host computes). An entry that is damaged, of another build, or made for another request is a
/// counted miss that translates again.
/// </summary>
[Collection(MetalTranslationCacheCollection.Name)]
public sealed class MetalTranslationCacheTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "SharpEmuTests", Guid.NewGuid().ToString("N"));

    public MetalTranslationCacheTests() => TranslationDiskCache.UseGameFolder(_directory, "metal-test-stamp");

    public void Dispose()
    {
        TranslationDiskCache.UseGameFolder(null, null);
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    // Compute (plain, LDS, loop), vertex, pixel, and a storage image with a runtime swizzle.
    private static IEnumerable<(string Name, ShaderCompileRequest Request)> Requests()
    {
        foreach (var fixture in new[] { Gen5ComputeFixtures.ExecStore, Gen5ComputeFixtures.Fmac, Gen5ComputeFixtures.Lds, Gen5ComputeFixtures.Loop })
        {
            yield return ("compute-" + fixture.Name, Gen5ComputeFixtures.CreateComputeRequest(fixture));
        }

        yield return ("vertex", Gen5ComputeFixtures.RequestOrThrow(
            Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("vertex", Gen5ComputeFixtures.VertexWords, 0, 0)), ShaderStage.Vertex));
        yield return ("pixel", Gen5ComputeFixtures.RequestOrThrow(
            Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("pixel", Gen5ComputeFixtures.PixelWords, 0, 0)), ShaderStage.Pixel,
            pixelOutputs: [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float, default)]));
        ShaderCompileRequest? image = null;
        Assert.True(Gen5ImageFixtures.TryCompile("ImageStore", Gen5ImageFixtures.ImageType2D, out _, out var error,
            unifiedFormat: 71, dstSelect: 0xF2E, addressValues: [1, 0], storeValues: [1, 2, 3, 4], runtimeImageSwizzles: true,
            inspect: (request, _) => image = request), error);
        Assert.Equal(1u, image!.Bindings.StorageImageSwizzleCount);
        yield return ("image-runtime-swizzle", image);
    }

    private static string Key(ShaderCompileRequest request) =>
        TranslationDiskCache.Key(
            new ProgramKey(request.Stage, request.Hash, request.UserDataCount, (uint)request.Program.Instructions.Count, 0, []),
            request.UserDataBase, new ResourceSpecialization(), 0, request)
        ?? throw new InvalidOperationException($"No translation cache key for {request.Stage} 0x{request.Hash:X16}.");

    [Fact]
    public void ARebuiltShaderIsTheTranslationWithItsRequest()
    {
        foreach (var (name, request) in Requests())
        {
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var translated, out var error), error);
            var fresh = new MetalCompiledGuestShader(translated, request);

            var rebuilt = MetalCompiledGuestShader.FromCachePayload(fresh.CachePayload, request);

            Assert.NotNull(rebuilt);
            AssertSameShader(fresh, rebuilt, name);
            // The rebuilt shader keeps the request, so the host can drop its source and translate it again.
            rebuilt.ReleaseSource();
            Assert.True(rebuilt.SourceReleased, name);
            Assert.Equal(translated.Source, rebuilt.AcquireShader().Source);
        }
    }

    [Fact]
    public void ASecondSessionTakesEveryShaderFromTheCacheWithoutTranslating()
    {
        var requests = Requests().ToArray();
        var first = new CountingBackend();
        var fresh = new Dictionary<string, MetalCompiledGuestShader>();
        foreach (var (name, request) in requests)
        {
            Assert.True(TranslationDiskCache.TryTranslate(first, request, Key(request), out var compiled, out var error), error);
            fresh[name] = Assert.IsType<MetalCompiledGuestShader>(compiled);
        }

        Assert.Equal(requests.Length, first.Translations);
        Assert.Equal(requests.Length, Directory.GetFiles(TranslationDiskCache.Folder!, "*.spvc").Length);

        // The next session: the same folder and stamp, a new backend.
        TranslationDiskCache.UseGameFolder(_directory, "metal-test-stamp");
        var second = new CountingBackend();
        var rejected = TranslationDiskCache.RejectedEntries;
        foreach (var (name, request) in requests)
        {
            Assert.True(TranslationDiskCache.TryTranslate(second, request, Key(request), out var compiled, out var error), error);
            AssertSameShader(fresh[name], Assert.IsType<MetalCompiledGuestShader>(compiled), name);
        }

        Assert.Equal(0, second.Translations);
        Assert.Equal(requests.Length, second.CachedShaders);
        Assert.Equal(rejected, TranslationDiskCache.RejectedEntries);
    }

    [Fact]
    public void AnotherStampIsAMiss()
    {
        var (_, request) = Requests().First();
        Assert.True(TranslationDiskCache.TryTranslate(new CountingBackend(), request, Key(request), out _, out var error), error);

        TranslationDiskCache.UseGameFolder(_directory, "another-translator-build");
        var backend = new CountingBackend();
        Assert.True(TranslationDiskCache.TryTranslate(backend, request, Key(request), out _, out error), error);

        Assert.Equal(1, backend.Translations);
        Assert.Equal(0, backend.CachedShaders);
    }

    // A file cut short, a flipped byte, and an entry made for a request of another stage under this request's key:
    // each is a counted miss, translated afresh, and its entry is replaced by the translation.
    [Theory]
    [InlineData("truncated")]
    [InlineData("corrupt")]
    [InlineData("other-stage")]
    public void ADamagedOrForeignEntryIsACountedMissThatTranslatesAgain(string damage)
    {
        var requests = Requests().ToArray();
        var (_, request) = requests.First(entry => entry.Request.Stage == ShaderStage.Compute);
        var key = Key(request);
        var path = Path.Combine(TranslationDiskCache.Folder!, key + ".spvc");
        if (damage == "other-stage")
        {
            var (_, pixel) = requests.First(entry => entry.Request.Stage == ShaderStage.Pixel);
            Assert.True(TranslationDiskCache.TryTranslate(new CountingBackend(), pixel, Key(pixel), out var other, out var otherError), otherError);
            TranslationDiskCache.Store(key, other!);
        }
        else
        {
            Assert.True(TranslationDiskCache.TryTranslate(new CountingBackend(), request, key, out _, out var storeError), storeError);
            var bytes = File.ReadAllBytes(path);
            File.WriteAllBytes(path, damage == "truncated" ? bytes[..(bytes.Length / 2)] : Flip(bytes, bytes.Length - 3));
        }

        var rejected = TranslationDiskCache.RejectedEntries;
        var refused = MetalCompiledGuestShader.CachePayloadsRefused;
        var backend = new CountingBackend();

        Assert.True(TranslationDiskCache.TryTranslate(backend, request, key, out var compiled, out var error), error);

        Assert.Equal(1, backend.Translations);
        Assert.Equal(rejected + 1, TranslationDiskCache.RejectedEntries);
        Assert.Equal(damage == "other-stage" ? refused + 1 : refused, MetalCompiledGuestShader.CachePayloadsRefused);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var expected, out error), error);
        Assert.Equal(expected.Source, Assert.IsType<MetalCompiledGuestShader>(compiled).AcquireShader().Source);

        // The translation replaced the entry: the next session takes it.
        var next = new CountingBackend();
        Assert.True(TranslationDiskCache.TryTranslate(next, request, key, out _, out error), error);
        Assert.Equal(0, next.Translations);
    }

    // A payload for the right stage but another program (hash) is refused too.
    [Fact]
    public void APayloadOfAnotherProgramIsRefused()
    {
        // The fixtures share one hash: the other program is the same request under another hash.
        var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
        var other = request.WithProgram(request.Program);
        typeof(ShaderCompileRequest).GetField("<Hash>k__BackingField",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.SetValue(other, request.Hash ^ 1);
        Assert.NotEqual(request.Hash, other.Hash);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var translated, out var error), error);
        var payload = new MetalCompiledGuestShader(translated, request).CachePayload;
        var refused = MetalCompiledGuestShader.CachePayloadsRefused;

        Assert.Null(MetalCompiledGuestShader.FromCachePayload(payload, other));
        Assert.Equal(refused + 1, MetalCompiledGuestShader.CachePayloadsRefused);
        Assert.NotNull(MetalCompiledGuestShader.FromCachePayload(payload, request));
    }

    private static byte[] Flip(byte[] bytes, int index)
    {
        var copy = (byte[])bytes.Clone();
        copy[index] ^= 0x5A;
        return copy;
    }

    // The same shader the host reads: every reflection field (the payload writes all of them), the request, the dump
    // payload and its identity, and the library key the host computes from the source and its options.
    private static void AssertSameShader(MetalCompiledGuestShader fresh, MetalCompiledGuestShader rebuilt, string name)
    {
        var expected = fresh.AcquireShader();
        var actual = rebuilt.AcquireShader();
        Assert.NotSame(expected, actual);
        Assert.Equal(fresh.CachePayload, rebuilt.CachePayload);
        Assert.Equal(expected.Source, actual.Source);
        Assert.Equal(expected.EntryPoint, actual.EntryPoint);
        Assert.Equal(expected.Stage, actual.Stage);
        Assert.Equal(expected.UsesPerVertexInputs, actual.UsesPerVertexInputs);
        Assert.Equal(expected.FlattenedTableMinimumWords, rebuilt.FlattenedTableMinimumWords);
        Assert.Equal(fresh.Payload, rebuilt.Payload);
        Assert.Equal(MetalShaderPrewarm.Identity(fresh), MetalShaderPrewarm.Identity(rebuilt));
        Assert.Equal(
            MetalBinaryArchive.LibraryKey(expected.Source, expected.UsesPerVertexInputs),
            MetalBinaryArchive.LibraryKey(actual.Source, actual.UsesPerVertexInputs));
        Assert.True(rebuilt.CachePayload.Length > 0, name);
    }

    // Translates as the Metal backend does and counts what it translated and what it took from the cache.
    private sealed class CountingBackend : IGuestGpuBackend
    {
        private static Exception Unsupported() => new NotSupportedException("The counting backend only translates.");

        public int Translations { get; private set; }

        public int CachedShaders { get; private set; }

        public string BackendName => "Metal (counting)";

        public bool TryCompileProgram(ShaderCompileRequest request, out IGuestCompiledShader? shader, out string error)
        {
            Translations++;
            shader = Gen5MslTranslator.TryCompileProgram(request, out var compiled, out error)
                ? new MetalCompiledGuestShader(compiled, request)
                : null;
            return shader is not null;
        }

        public IGuestCompiledShader? CreateCachedShader(byte[] payload, uint? inputLocationMask, ShaderCompileRequest request)
        {
            var shader = MetalCompiledGuestShader.FromCachePayload(payload, request);
            if (shader is not null) CachedShaders++;
            return shader;
        }

        public void CountShaderCompilation()
        {
        }

        public IGuestCompiledShader GetDepthOnlyFragmentShader() => throw Unsupported();
        public void EnsureStarted(uint width, uint height) => throw Unsupported();
        public void HideSplashScreen() => throw Unsupported();
        public void Submit(byte[] bgraFrame, uint width, uint height) => throw Unsupported();
        public void SubmitGuestDraw(GuestDrawKind drawKind, uint width, uint height) => throw Unsupported();
        public bool TrySubmitGuestImage(int videoOutHandle, int displayBufferIndex, ulong address, uint width, uint height,
            uint pitchInPixel, ulong flipRequestId) => throw Unsupported();
        public void SubmitCommandStream(ICpuMemory memory, uint queue, ulong address, uint dwordCount, ulong submissionId,
            object? geometrySnapshots) => throw Unsupported();
        public IdleOutcome SubmitDone(ICpuMemory memory) => throw Unsupported();
        public void RegisterKnownDisplayBuffer(ulong address, uint guestFormat) => throw Unsupported();
        public bool IsGpuGuestImageAvailable(ulong address, uint format, uint numberType) => false;
        public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() => throw Unsupported();
        public void RequestClose() => throw Unsupported();
    }
}

// The cache's folder is process-wide: these tests run one at a time.
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class MetalTranslationCacheCollection
{
    public const string Name = "MetalTranslationCache";
}
