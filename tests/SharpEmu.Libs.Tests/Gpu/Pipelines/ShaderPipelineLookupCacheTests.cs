// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Images;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// Counts the reads a lookup makes, so a test can tell a cached lookup from a decoded one.
internal sealed class ReadCountingMemory(ICpuMemory inner) : ICpuMemory
{
    public long Reads { get; private set; }

    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        Reads++;
        return inner.TryRead(virtualAddress, destination);
    }

    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => inner.TryWrite(virtualAddress, source);

    public bool TryCompare(ulong virtualAddress, ReadOnlySpan<byte> expected, out bool equal) =>
        inner.TryCompare(virtualAddress, expected, out equal);

    public bool CanRead(ulong address, ulong size) => inner.CanRead(address, size);
}

// What the per-draw lookups of a graphics draw remember between draws (the pixel semantics, the vertex layout, the
// shader identity, the header records and the pipeline maps), and that a guest that changes the bytes one of them
// came from is seen on the next draw.
[Collection(SchedulingStateCollection.Name)]
public sealed class ShaderPipelineLookupCacheTests : IDisposable
{
    private const ulong B = PipelineTestGuest.MemoryBase;
    private const ulong VertexCode = B + 0x1000;
    private const ulong PixelCode = B + 0x4000;
    private const ulong VertexSemantics = B + 0x9200;
    private const ulong AttributeTable = B + 0x9300;
    private const ulong BufferTable = B + 0x9400;
    private const ulong PixelSemantics = B + 0x9500;
    private const uint CustomSemantic = 1u << 24;
    private const uint AttributeFormat32x2Float = 257;

    private readonly FatalScope _fatal = new();
    private readonly FakeCpuMemory _memory = new(B, PipelineTestGuest.MemorySize);
    private readonly ReadCountingMemory _counting;
    private readonly Dictionary<ulong, ulong> _headers = new();
    private readonly FakePipelineHost _host;
    private readonly ShaderPipelineCache _cache;
    private readonly RegisterBanks _banks = Banks();
    private readonly ColorComponentMap[] _mappings = Enumerable.Repeat(ColorComponentMap.Identity, 8).ToArray();

    public ShaderPipelineLookupCacheTests()
    {
        _counting = new ReadCountingMemory(_memory);
        var context = new CpuContext(_counting, Generation.Gen5);
        var registry = new ShaderHeaderRegistry(context, code => _headers.TryGetValue(code, out var header) ? header : 0, _ => null);
        _host = new FakePipelineHost(_counting);
        _cache = new ShaderPipelineCache(context, _host, new FakeShaderCompiler(), registry);
        RegisterProgram(VertexCode, B + 0x8000, 64, B + 0x9000, VertexSemantics, 3);
        RegisterProgram(PixelCode, B + 0x8100, 128, 0, PixelSemantics, 4);
        // The user data block of the vertex program: buffer table at s[10:11], attribute table at s[8:9].
        WriteQword(B + 0x9000, B + 0x9100);
        WriteHalf(B + 0x9000 + 0x2C, 11);
        for (uint type = 0; type < 11; type++)
        {
            WriteHalf(B + 0x9100 + type * 2, 0xFFFF);
        }

        WriteHalf(B + 0x9100 + 8 * 2, 10);
        WriteHalf(B + 0x9100 + 10 * 2, 8);
        for (uint index = 0; index < 3; index++)
        {
            WriteWord(VertexSemantics + index * 4, index | (4u << 8) | (4u << 16));
            WriteAttribute(AttributeTable, index, index, offset: 0);
            WriteBuffer(index, B + 0x20000 + index * 0x1000);
        }

        for (uint index = 0; index < 4; index++)
        {
            WriteWord(PixelSemantics + index * 4, index | CustomSemantic);
        }

        for (uint slot = 0; slot < 4; slot++)
        {
            _banks.Context.ColorTargets[slot] = RegisterWords.Color(B + 0x40000 + slot * 0x10000, 64, 64);
        }

        _banks.Context.RenderTargetMask = 0xFFFF;
        _banks.Context.ShaderInterface.ColorShaderMask = 0xFFFF;
        _banks.Context.ShaderInterface.PixelInputControl = 4;
        _banks.Shader.Vertex.ExportAddress = VertexCode;
        _banks.Shader.Pixel.Address = PixelCode;
        SetAttributeTable(AttributeTable);
    }

    public void Dispose() => _fatal.Dispose();

    private void SetAttributeTable(ulong address)
    {
        var user = _banks.Shader.Vertex.GeometryUserScalars;
        user.Set(8, unchecked((uint)address), UserScalarKind.Unknown);
        user.Set(9, (uint)(address >> 32), UserScalarKind.Unknown);
        user.Set(10, unchecked((uint)BufferTable), UserScalarKind.Unknown);
        user.Set(11, (uint)(BufferTable >> 32), UserScalarKind.Unknown);
    }

    private void WriteWord(ulong address, uint value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        Assert.True(_memory.TryWrite(address, bytes));
    }

    private void WriteQword(ulong address, ulong value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        Assert.True(_memory.TryWrite(address, bytes));
    }

    private void WriteHalf(ulong address, ushort value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ushort)];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, value);
        Assert.True(_memory.TryWrite(address, bytes));
    }

    private void WriteAttribute(ulong table, uint semantic, uint bufferIndex, uint offset) =>
        WriteWord(table + semantic * 4, bufferIndex | (AttributeFormat32x2Float << 5) | (offset << 14));

    private void WriteBuffer(uint index, ulong address, uint records = 100)
    {
        var entry = BufferTable + index * 16;
        WriteWord(entry, unchecked((uint)address));
        WriteWord(entry + 4, (uint)((address >> 32) & 0xFFFF) | (16u << 16));
        WriteWord(entry + 8, records);
        WriteWord(entry + 12, 0xFAC | (64u << 12));
    }

    // Code of the given size that starts with a no-op and ends the program.
    private static byte[] Code(uint words)
    {
        var code = new byte[words * 4];
        BinaryPrimitives.WriteUInt32LittleEndian(code, 0xBF800000);
        BinaryPrimitives.WriteUInt32LittleEndian(code.AsSpan((int)(words - 1) * 4), 0xBF810000);
        return code;
    }

    private void RegisterProgram(ulong code, ulong header, uint words, ulong userData, ulong semantics, uint semanticCount)
    {
        Assert.True(_memory.TryWrite(code, Code(words)));
        var bytes = new byte[PipelineTestGuest.HeaderBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x44), words * 4);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(0x08), userData);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(0x30), semantics);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x50), semanticCount);
        Assert.True(_memory.TryWrite(header, bytes));
        _headers[code] = header;
    }

    private GraphicsPrograms Draw(bool pixelActive = true)
    {
        var programs = _cache.GetGraphicsPrograms(_banks.Shader.Vertex, _banks.Shader.Pixel, _banks.Context.ShaderInterface,
            _banks.Context, _mappings, pixelActive, depthBound: false, depthAspectBound: false);
        Assert.True(programs.Available);
        return programs;
    }

    private static ulong AddressOf(GraphicsPrograms programs, int attributeId) =>
        programs.VertexInput.Attributes.Single(attribute => attribute.AttributeId == attributeId).Descriptor.Address;

    [Fact]
    public void ARepeatedDrawSharesTheVertexLayoutAndReadsTheTablesOnce()
    {
        var first = Draw();
        Draw();
        var before = _counting.Reads;
        var second = Draw();
        var reads = _counting.Reads - before;

        Assert.Same(first.VertexInput.Attributes, second.VertexInput.Attributes);
        Assert.Same(first.VertexInput.Buffers, second.VertexInput.Buffers);
        Assert.Equal(3, second.VertexInput.Attributes.Length);
        // Two header reads, two declared-hash probes, the pixel semantics and the two vertex tables: not one
        // read per attribute or input.
        Assert.True(reads <= 7, $"reads={reads}");
    }

    [Fact]
    public void ARewrittenAttributeWordIsSeenOnTheNextDraw()
    {
        var first = Draw();
        Assert.Equal(B + 0x21000, AddressOf(first, 1));

        WriteAttribute(AttributeTable, 1, 1, offset: 32);
        var second = Draw();

        Assert.NotSame(first.VertexInput.Attributes, second.VertexInput.Attributes);
        Assert.Equal(B + 0x21020, AddressOf(second, 1));
        Assert.Equal(B + 0x21000, AddressOf(first, 1));
    }

    [Fact]
    public void ARewrittenAttributeTablePointerIsSeenOnTheNextDraw()
    {
        var first = Draw();
        var moved = B + 0x9600;
        for (uint index = 0; index < 3; index++)
        {
            WriteAttribute(moved, index, index, offset: (index + 1) * 16);
        }

        SetAttributeTable(moved);
        var second = Draw();

        Assert.Equal(B + 0x22000, AddressOf(first, 2));
        Assert.Equal(B + 0x22030, AddressOf(second, 2));
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void RelocatedIdenticalVertexTablesReuseLayoutsButRewrittenBytesDoNot(bool moveAttributes, bool moveBuffers)
    {
        var first = Draw();
        var attributes = moveAttributes ? B + 0x9600 : AttributeTable;
        var buffers = moveBuffers ? B + 0x9800 : BufferTable;
        Span<byte> attributeBytes = stackalloc byte[3 * 4];
        Span<byte> bufferBytes = stackalloc byte[3 * 16];
        Assert.True(_memory.TryRead(AttributeTable, attributeBytes));
        Assert.True(_memory.TryRead(BufferTable, bufferBytes));
        Assert.True(_memory.TryWrite(attributes, attributeBytes));
        Assert.True(_memory.TryWrite(buffers, bufferBytes));
        SetAttributeTable(attributes);
        var user = _banks.Shader.Vertex.GeometryUserScalars;
        user.Set(10, unchecked((uint)buffers), UserScalarKind.Unknown);
        user.Set(11, (uint)(buffers >> 32), UserScalarKind.Unknown);

        var moved = Draw();
        Assert.Same(first.VertexInput.Attributes, moved.VertexInput.Attributes);
        Assert.Same(first.VertexInput.Buffers, moved.VertexInput.Buffers);

        WriteWord(buffers + 2 * 16, unchecked((uint)(B + 0x30000)));
        var changed = Draw();
        Assert.NotSame(first.VertexInput.Buffers, changed.VertexInput.Buffers);
        Assert.Equal(B + 0x30000, AddressOf(changed, 2));
        Assert.Equal(B + 0x22000, AddressOf(first, 2));
        WriteAttribute(attributes, 1, 1, offset: 32);
        Assert.Equal(B + 0x21020, AddressOf(Draw(), 1));
    }

    [Fact]
    public void ARewrittenBufferDescriptorIsSeenOnTheNextDraw()
    {
        var first = Draw();
        WriteBuffer(2, B + 0x30000);
        var second = Draw();

        Assert.NotSame(first.VertexInput.Buffers, second.VertexInput.Buffers);
        Assert.Contains(second.VertexInput.Buffers, buffer => buffer.Address == B + 0x30000);
        Assert.DoesNotContain(first.VertexInput.Buffers, buffer => buffer.Address == B + 0x30000);
    }

    [Fact]
    public void ARewrittenPixelSemanticIsSeenOnTheNextDraw()
    {
        var reads = _counting.Reads;
        Assert.Equal(0xFu, Draw().PixelInput.CustomInterpolationMask);

        WriteWord(PixelSemantics + 2 * 4, 2);
        Assert.Equal(0xBu, Draw().PixelInput.CustomInterpolationMask);

        // A half-float input is not a custom interpolant.
        WriteWord(PixelSemantics + 2 * 4, 2u | CustomSemantic | (1u << 20));
        Assert.Equal(0xBu, Draw().PixelInput.CustomInterpolationMask);

        WriteWord(PixelSemantics + 2 * 4, 2u | CustomSemantic);
        Assert.Equal(0xFu, Draw().PixelInput.CustomInterpolationMask);
        Assert.True(_counting.Reads > reads);
    }

    [Fact]
    public void AnUnreadablePixelSemanticTableStopsTheDraw()
    {
        // The header names a table that is not mapped.
        RegisterProgram(PixelCode, B + 0x8100, 128, 0, B + 0xFFFFF0, 8);
        var failure = Assert.Throws<SchedulerFatalException>(() => Draw());
        Assert.Contains("pixel input semantics are unreadable", failure.Message);
    }

    [Fact]
    public void ShaderCodeWithoutADeclaredHashIsHashedOnceAndRehashedWhenTheGuestChangesIt()
    {
        var memo = new ShaderIdentityMemo();
        var ranges = new[] { (VertexCode, 64u * 4) };
        var first = ShaderIdentity.Compute(_memory, VertexCode, ranges, "vertex", memo);
        Assert.Equal(ShaderIdentity.Compute(_memory, VertexCode, ranges, "vertex"), first);
        Assert.Equal(1, memo.Count);

        // Unchanged: remembered. The code is compared in place, not read out and hashed again.
        var before = _counting.Reads;
        Assert.Equal(first, ShaderIdentity.Compute(_counting, VertexCode, ranges, "vertex", memo));
        Assert.Equal(1, _counting.Reads - before);

        // One word of the code changes in place.
        WriteWord(VertexCode + 8, 0xBF800001);
        var second = ShaderIdentity.Compute(_counting, VertexCode, ranges, "vertex", memo);
        Assert.NotEqual(first, second);
        Assert.Equal(ShaderIdentity.Compute(_memory, VertexCode, ranges, "vertex"), second);

        // And back: the content decides.
        WriteWord(VertexCode + 8, 0);
        Assert.Equal(first, ShaderIdentity.Compute(_counting, VertexCode, ranges, "vertex", memo));
    }

    [Fact]
    public void ACodeRangeThatMovesOrGrowsIsHashedAgain()
    {
        var memo = new ShaderIdentityMemo();
        var small = ShaderIdentity.Compute(_memory, VertexCode, [(VertexCode, 32u * 4)], "vertex", memo);
        var large = ShaderIdentity.Compute(_memory, VertexCode, [(VertexCode, 64u * 4)], "vertex", memo);

        Assert.NotEqual(small, large);
        Assert.Equal(ShaderIdentity.Compute(_memory, VertexCode, [(VertexCode, 64u * 4)], "vertex"), large);
    }

    [Fact]
    public void ADeclaredHashIsNotRemembered()
    {
        var memo = new ShaderIdentityMemo();
        var code = Code(64);
        // The marker's info block follows (word 1 + 1) pairs of dwords from the start; its hash is at +16 and +20.
        BinaryPrimitives.WriteUInt32LittleEndian(code, ShaderBinaryInfo.MarkerWord);
        BinaryPrimitives.WriteUInt32LittleEndian(code.AsSpan(4), 7);
        BinaryPrimitives.WriteUInt32LittleEndian(code.AsSpan((7 + 1) * 8 + 16), 0x1234);
        BinaryPrimitives.WriteUInt32LittleEndian(code.AsSpan((7 + 1) * 8 + 20), 0x5678);
        Assert.True(_memory.TryWrite(VertexCode, code));

        Assert.Equal(0x5678_0000_1234UL, ShaderIdentity.Compute(_memory, VertexCode, [(VertexCode, 64u * 4)], "vertex", memo));
        Assert.Equal(0, memo.Count);
    }

    [Fact]
    public void ARewrittenShaderCodeIsSeenByTheNextDraw()
    {
        var first = Draw();
        WriteWord(PixelCode + 8, 0xBF800002);
        var second = Draw();

        Assert.NotEqual(first.Pixel.Id, second.Pixel.Id);
        Assert.Equal(first.Vertex.Id, second.Vertex.Id);
    }

    [Fact]
    public void ARepeatedHeaderLookupSharesItsRecordAndAChangedHeaderReplacesIt()
    {
        var registry = new ShaderHeaderRegistry(new CpuContext(_memory, Generation.Gen5),
            code => _headers.TryGetValue(code, out var header) ? header : 0, _ => null);
        var first = registry.Require(PixelCode, "pixel");
        Assert.Same(first, registry.Require(PixelCode, "pixel"));

        RegisterProgram(PixelCode, B + 0x8100, 96, 0, PixelSemantics, 4);
        var changed = registry.Require(PixelCode, "pixel");
        Assert.NotSame(first, changed);
        Assert.Equal(96u * 4, changed.CodeSizeBytes);
        Assert.Same(changed, registry.Require(PixelCode, "pixel"));
    }

    [Fact]
    public void ConcurrentHeaderLookupsAllSeeTheCurrentRecord()
    {
        var registry = new ShaderHeaderRegistry(new CpuContext(_memory, Generation.Gen5),
            code => _headers.TryGetValue(code, out var header) ? header : 0, _ => null);
        Parallel.For(0, 8, _ =>
        {
            for (var round = 0; round < 2000; round++)
            {
                Assert.Equal(128u * 4, registry.Require(PixelCode, "pixel").CodeSizeBytes);
            }
        });

        // Racing first lookups may each make a record; afterwards every lookup gets the one that stuck.
        var settled = registry.Require(PixelCode, "pixel");
        Assert.Same(settled, registry.Require(PixelCode, "pixel"));
    }

    [Fact]
    public void ADroppedColorTargetLeavesTheNextDrawsPixelOutputs()
    {
        Assert.Equal(4, Draw().PixelInput.TargetOutputModes.Count(mode => mode != 0));

        _banks.Context.ColorTargets[2] = default;
        Assert.Equal(3, Draw().PixelInput.TargetOutputModes.Count(mode => mode != 0));

        _banks.Context.ColorTargets[2] = RegisterWords.Color(B + 0x60000, 64, 64);
        Assert.Equal(4, Draw().PixelInput.TargetOutputModes.Count(mode => mode != 0));
    }

    [Fact]
    public void APipelineIsLookedUpWithoutBeingCreatedAgain()
    {
        var programs = Draw();
        var depth = default(DepthAttachmentState);
        var rendering = default(RenderingState);
        Assert.True(_cache.TryCreateGraphicsPipeline([], in depth, programs.VertexInput, programs.PixelInput, _banks.Context, in rendering,
            GpuPrimitiveTopology.TriangleList, false, false, programs.Vertex, programs.Pixel, out var first));
        var created = _host.GraphicsPipelines.Count;

        var again = Draw();
        Assert.True(_cache.TryCreateGraphicsPipeline([], in depth, again.VertexInput, again.PixelInput, _banks.Context, in rendering,
            GpuPrimitiveTopology.TriangleList, false, false, again.Vertex, again.Pixel, out var second));

        Assert.Equal(first, second);
        Assert.Equal(created, _host.GraphicsPipelines.Count);
        Assert.Equal(1, _cache.GraphicsPipelineCount);

        // A different pipeline state is a different pipeline.
        Assert.True(_cache.TryCreateGraphicsPipeline([], in depth, again.VertexInput, again.PixelInput, _banks.Context, in rendering,
            GpuPrimitiveTopology.LineList, false, false, again.Vertex, again.Pixel, out var other));
        Assert.NotEqual(first, other);
        Assert.Equal(2, _cache.GraphicsPipelineCount);
    }
}
