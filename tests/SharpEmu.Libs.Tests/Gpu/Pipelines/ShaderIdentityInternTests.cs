// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Memory.GpuMemory;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

public sealed class ShaderIdentityInternTests
{
    private const ulong Code = PipelineTestGuest.MemoryBase + 0x1000;
    private const ulong Header = PipelineTestGuest.MemoryBase + 0x8000;

    [Fact]
    public void WarmIdentitySharesHandleWithoutReadingOrAllocating()
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var versions = new GuestWriteGenerations();
        var memory = new CountingMemory(guest.Memory);
        var cache = new ShaderIdentityCache(memory, static (_, _) => { }, versions);
        var shader = guest.Registry.Require(Code, "compute");
        var first = cache.Get(shader, "compute");
        for (var i = 0; i < 100; i++) Assert.Same(first, cache.Get(shader, "compute"));
        var reads = memory.Reads;
        var before = GC.GetAllocatedBytesForCurrentThread();
        var again = cache.Get(shader, "compute");
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Same(first, again);
        Assert.Equal(reads, memory.Reads);
    }

    [Fact]
    public void ProgramAndHeaderInternCountersReportBothHitsAndMisses()
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var versions = new GuestWriteGenerations();
        var registry = new ShaderHeaderRegistry(guest.Context, _ => Header, _ => null, versions);
        var cache = new ShaderIdentityCache(guest.Memory, static (_, _) => { }, versions);
        var shader = registry.Require(Code, "compute");
        _ = registry.Require(Code, "compute");
        _ = cache.Get(shader, "compute");
        _ = cache.Get(shader, "compute");
        var report = ProgramInternCounters.TakeReport();
        Assert.Matches("program_intern_miss=[1-9][0-9]*", report);
        Assert.Matches("header_intern_miss=[1-9][0-9]*", report);
        if (ShaderIdentityCache.Enabled)
        {
            Assert.Matches("program_intern_hit=[1-9][0-9]*", report);
            Assert.Matches("header_intern_hit=[1-9][0-9]*", report);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CpuAndGpuWritesChangeCodeIdentity(bool gpu)
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var manager = new GuestGpuMemory(new RecordingAddressSpace());
        manager.Register(PipelineTestGuest.MemoryBase, PipelineTestGuest.MemorySize, GuestPageProtection.Read | GuestPageProtection.Write);
        var cache = new ShaderIdentityCache(guest.Memory, static (_, _) => { }, manager.WriteGenerations);
        var shader = guest.Registry.Require(Code, "compute");
        var before = cache.Get(shader, "compute");
        if (gpu) manager.WriteGenerations.NoteWrite(Code, 4); // Metal DMA/copy notification, before bytes land.
        else Assert.True(manager.MarkCpuWrite(Code, 4));
        guest.WriteWords(Code, 0xBF800000);
        if (!gpu) manager.MarkCpuWriteLanded(Code, 4);
        var after = cache.Get(shader, "compute");
        Assert.NotEqual(before.ContentHash, after.ContentHash);
        Assert.Same(after, cache.Get(shader, "compute"));
    }

    [Fact]
    public void NativeFaultInvalidatesAndReleasedPageNeverCertifiesLaterStores()
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var manager = new GuestGpuMemory(new RecordingAddressSpace());
        manager.Register(PipelineTestGuest.MemoryBase, PipelineTestGuest.MemorySize, GuestPageProtection.Read | GuestPageProtection.Write);
        var cache = new ShaderIdentityCache(guest.Memory, static (_, _) => { }, manager.WriteGenerations);
        var shader = guest.Registry.Require(Code, "compute");
        var old = cache.Get(shader, "compute");
        Assert.True(manager.TryResolveFault(FaultKind.Write, Code));
        guest.WriteWords(Code, 0xBF800000);
        guest.WriteWords(Code, 0xBF800001); // A second native store needs no notification on the released page.
        var fresh = cache.Get(shader, "compute");
        Assert.NotEqual(old.ContentHash, fresh.ContentHash);
        Assert.False(manager.Pages.Allows(Code, FaultKind.Write)); // Recapture re-arms before reading.
        Assert.True(manager.TryResolveFault(FaultKind.Write, Code));
        guest.WriteWords(Code, 0xBF800002);
        Assert.NotEqual(fresh.ContentHash, cache.Get(shader, "compute").ContentHash);
    }

    [Fact]
    public void RemapInvalidatesCodeAndHeaderTokensEvenAtTheSameVa()
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var manager = new GuestGpuMemory(new RecordingAddressSpace());
        manager.Register(PipelineTestGuest.MemoryBase, PipelineTestGuest.MemorySize, GuestPageProtection.Read | GuestPageProtection.Write);
        var cache = new ShaderIdentityCache(guest.Memory, static (_, _) => { }, manager.WriteGenerations);
        var shader = guest.Registry.Require(Code, "compute");
        var token = manager.WriteGenerations.Register(Code, 4);
        var first = cache.Get(shader, "compute");
        manager.Unregister(Code, 4096);
        manager.Register(Code, 4096, GuestPageProtection.Read | GuestPageProtection.Write);
        guest.WriteWords(Code, 0xBF800000);
        Assert.False(token.IsCurrent);
        Assert.NotEqual(first.ContentHash, cache.Get(shader, "compute").ContentHash);
    }

    [Fact]
    public void HeaderWriteRefreshesTheRegisteredHandleWithoutReadingOnHits()
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var versions = new GuestWriteGenerations();
        var memory = new CountingMemory(guest.Memory);
        var registry = new ShaderHeaderRegistry(new CpuContext(memory, Generation.Gen5), _ => Header, _ => null, versions);
        var first = registry.Require(Code, "compute");
        var reads = memory.Reads;
        Assert.Same(first, registry.Require(Code, "compute"));
        Assert.Equal(reads, memory.Reads);
        versions.NoteWrite(Header + 0x54, 4);
        guest.WriteWords(Header + 0x54, 32);
        Assert.Equal(32u, registry.Require(Code, "compute").ScratchDwords);
        Assert.NotSame(first, registry.Require(Code, "compute"));
    }

    [Fact]
    public void WarmIdentityResolvesTextureOwnershipBeforeValidatingItsGeneration()
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var versions = new GuestWriteGenerations();
        var textureOwnsCode = false;
        var cache = new ShaderIdentityCache(guest.Memory, (address, size) =>
        {
            if (!textureOwnsCode) return;
            Assert.Equal(Code, address);
            Assert.Equal(4UL, size);
            // Texture writes do not touch backing generations until publication.
            versions.NoteWrite(address, size);
            guest.WriteWords(Code, 0xBF800000);
            textureOwnsCode = false;
        }, versions);
        var shader = guest.Registry.Require(Code, "compute");
        var first = cache.Get(shader, "compute");
        textureOwnsCode = true;
        Assert.NotEqual(first.ContentHash, cache.Get(shader, "compute").ContentHash);
        Assert.False(textureOwnsCode);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HeaderResolvesOwnershipBeforeCachedOrUncachedReads(bool intern)
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, PipelineTestGuest.EndProgram);
        using var versions = new GuestWriteGenerations();
        var textureOwnsHeader = false;
        var registry = new ShaderHeaderRegistry(guest.Context, _ => Header, _ => null,
            intern ? versions : null, (address, size) =>
            {
                if (!textureOwnsHeader) return;
                Assert.Equal(Header, address);
                versions.NoteWrite(address, size);
                guest.WriteWords(Header + 0x54, 32);
                textureOwnsHeader = false;
            });
        Assert.Equal(0u, registry.Require(Code, "compute").ScratchDwords);
        textureOwnsHeader = true;
        Assert.Equal(32u, registry.Require(Code, "compute").ScratchDwords);
        Assert.False(textureOwnsHeader);
    }

    [Fact]
    public void DeclaredHashCannotHideRewrittenCode()
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(Code, Header, [ShaderBinaryInfo.MarkerWord, 0, 0, 0, 0, 0, 123, 456, 0]);
        using var versions = new GuestWriteGenerations();
        var cache = new ShaderIdentityCache(guest.Memory, static (_, _) => { }, versions);
        var shader = guest.Registry.Require(Code, "compute");
        var first = cache.Get(shader, "compute");
        versions.NoteWrite(Code + 32, 4);
        guest.WriteWords(Code + 32, 1);
        var second = cache.Get(shader, "compute");
        Assert.Equal(first.Hash, second.Hash);
        Assert.NotEqual(first.ContentHash, second.ContentHash);
    }

    [Fact]
    public async Task ConcurrentReadyRequestsCompileOnceAndOtherKeysCanProgress()
    {
        var cache = new ReadyPipelineCache<int>();
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var builds = 0;
        var expected = new PipelineHandle(17, 0, false);
        PipelineHandle? Build()
        {
            Interlocked.Increment(ref builds);
            entered.Set();
            Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
            return expected;
        }
        var first = Task.Run(() => { Assert.True(cache.GetOrCreate(1, Build, true, out var handle)); return handle; });
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        var others = Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
        {
            Assert.True(cache.GetOrCreate(1, Build, true, out var handle));
            return handle;
        })).ToArray();
        Assert.True(cache.GetOrCreate(2, () => new PipelineHandle(18, 0, false), true, out _));
        release.Set();
        Assert.Equal(expected, await first);
        foreach (var handle in await Task.WhenAll(others)) Assert.Equal(expected, handle);
        Assert.Equal(1, builds);
        Assert.True(cache.TryGet(1, out var ready));
        Assert.Equal(expected, ready);
    }

    [Fact]
    public void FailedOrPendingCompileDoesNotPoisonTheReadyCache()
    {
        var cache = new ReadyPipelineCache<int>();
        Assert.Throws<InvalidOperationException>(() => cache.GetOrCreate(1, () => throw new InvalidOperationException(), true, out _));
        Assert.False(cache.GetOrCreate(1, () => null, false, out _));
        Assert.True(cache.GetOrCreate(1, () => new PipelineHandle(1, 0, false), true, out _));
    }

    private sealed class CountingMemory(ICpuMemory inner) : ICpuMemory
    {
        public int Reads;
        public bool TryRead(ulong address, Span<byte> destination) { Reads++; return inner.TryRead(address, destination); }
        public bool TryWrite(ulong address, ReadOnlySpan<byte> source) => inner.TryWrite(address, source);
    }
}
