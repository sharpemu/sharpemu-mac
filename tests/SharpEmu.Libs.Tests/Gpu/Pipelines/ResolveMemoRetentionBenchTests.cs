// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// What the graphics resolve path allocates per draw and what the resolve memo keeps of it, for a frame of draws repeated
// the way Demon's Souls repeats them: every draw has its own descriptor block (its own key), and each frame either
// nothing moves (hits), two of a draw's eight V#s get another base (ring-allocated constant buffers: words misses that
// the memo rebuilds), or the draw's pointer to its block moves (a new key every frame). Retained is what survives a gen0
// collection made after the frame: bytes the long-lived memo now points at. The numbers go to the test output; run alone
// (the collection below is not parallelized) with --filter "FullyQualifiedName~ResolveMemoRetentionBench" and
// --logger "console;verbosity=detailed".
[Collection(SchedulingStateCollection.Name)]
public sealed class ResolveMemoRetentionBenchTests : IDisposable
{
    private const ulong VertexAddress = PipelineTestGuest.MemoryBase + 0x1000;
    private const ulong VertexHeader = PipelineTestGuest.MemoryBase + 0x8000;
    private const ulong UserDataBlock = PipelineTestGuest.MemoryBase + 0x9000;
    private const ulong DirectOffsets = PipelineTestGuest.MemoryBase + 0x9100;
    private const ulong Semantics = PipelineTestGuest.MemoryBase + 0x9200;
    private const ulong AttributeTable = PipelineTestGuest.MemoryBase + 0xA000;
    private const ulong BufferTable = PipelineTestGuest.MemoryBase + 0xA100;
    private const ulong PixelAddress = PipelineTestGuest.MemoryBase + 0x3000;
    private const ulong PixelHeader = PipelineTestGuest.MemoryBase + 0x8200;
    private const ulong PixelUserDataBlock = PipelineTestGuest.MemoryBase + 0xC000;
    private const ulong StreamBase = PipelineTestGuest.MemoryBase + 0x20000;
    private const ulong Blocks = PipelineTestGuest.MemoryBase + 0x30000;
    private const uint DirectResourceCount = 11;
    private const uint VertexBufferTableType = 8;
    private const uint VertexAttributeTableType = 10;
    private const uint AttributeRegister = 0;
    private const uint BufferRegister = 2;
    private const uint Format32x4Float = 77;
    private const int Buffers = 8;
    private const int BlockBytes = Buffers * 16;
    private const int Draws = 256;
    private const int WarmFrames = 3;
    private const int MeasuredFrames = 8;

    private readonly FatalScope _fatal = new();
    private readonly PipelineTestGuest _guest = new();
    private readonly RegisterBanks _banks = new(static message => new InvalidOperationException(message));
    private readonly ITestOutputHelper _output;

    public ResolveMemoRetentionBenchTests(ITestOutputHelper output)
    {
        _output = output;
        WriteQword(UserDataBlock, DirectOffsets);
        WriteHalf(UserDataBlock + 0x2C, (ushort)DirectResourceCount);
        for (uint type = 0; type < DirectResourceCount; type++)
        {
            WriteHalf(DirectOffsets + type * 2, 0xFFFF);
        }

        WriteHalf(DirectOffsets + VertexBufferTableType * 2, (ushort)BufferRegister);
        WriteHalf(DirectOffsets + VertexAttributeTableType * 2, (ushort)AttributeRegister);
        _guest.WriteWords(Semantics, 0 | (4u << 16), 1 | (4u << 8) | (2u << 16));
        _guest.RegisterProgram(VertexAddress, VertexHeader, PipelineTestGuest.EndProgram, UserDataBlock, Semantics, 2);
        _guest.WriteWords(AttributeTable, 0, 16u << 14);
        _guest.WriteWords(BufferTable, PipelineTestGuest.BufferDescriptor(StreamBase, 24, 3, Format32x4Float));
        _guest.RegisterProgram(PixelAddress, PixelHeader, LoadingProgram(), PixelUserDataBlock);
        _banks.Shader.Vertex.ExportAddress = VertexAddress;
        _banks.Shader.Pixel.Address = PixelAddress;
        var registers = _banks.Shader.Vertex.GeometryUserScalars;
        registers.Set(AttributeRegister, unchecked((uint)AttributeTable), UserScalarKind.Unknown);
        registers.Set(AttributeRegister + 1, (uint)(AttributeTable >> 32), UserScalarKind.Unknown);
        registers.Set(BufferRegister, unchecked((uint)BufferTable), UserScalarKind.Unknown);
        registers.Set(BufferRegister + 1, (uint)(BufferTable >> 32), UserScalarKind.Unknown);
        _banks.MarkWritten(1u << (int)RegisterStamp.VertexUserData);
    }

    public void Dispose() => _fatal.Dispose();

    // s_load_dwordx4 s[4+4i:7+4i], s[0:1], 16*i for each V#, then buffer_load_format_x v[1+i], v0, s[4+4i:7+4i] through
    // each, then s_endpgm.
    private static uint[] LoadingProgram()
    {
        var words = new List<uint>();
        for (var index = 0; index < Buffers; index++)
        {
            words.Add(0xF4080000u | ((uint)(4 + 4 * index) << 6));
            words.Add(0xFA000000u | (uint)(16 * index));
        }

        for (var index = 0; index < Buffers; index++)
        {
            words.Add(0xE0000000u);
            words.Add(0x80000000u | ((uint)(1 + index) << 16) | ((uint)(1 + index) << 8));
        }

        words.Add(0xBF810000u);
        return [.. words];
    }

    public enum Churn
    {
        // Nothing moves: every draw hits.
        None,
        // Two V#s of every draw take another base each frame: words misses.
        Words,
        // Every draw's pointer moves to a block of its own each frame: a new key every frame.
        Keys,
    }

    private void WriteBlock(ulong block, int draw, int frame, bool moveWords)
    {
        for (var index = 0; index < Buffers; index++)
        {
            var moved = moveWords && index < 2 ? (ulong)((frame * Draws + draw) % 4096) * 0x40 : 0;
            _guest.WriteWords(block + (ulong)(16 * index),
                PipelineTestGuest.BufferDescriptor(StreamBase + (ulong)(index * 0x40) + moved, 16, 4, Format32x4Float));
        }
    }

    private ulong BlockOf(Churn churn, int draw, int frame) =>
        Blocks + (ulong)((churn == Churn.Keys ? frame * Draws + draw : draw) * BlockBytes);

    private void PointPixelAt(ulong block)
    {
        _banks.Shader.Pixel.UserScalars.Set(0, (uint)block, UserScalarKind.Unknown);
        _banks.Shader.Pixel.UserScalars.Set(1, (uint)(block >> 32), UserScalarKind.Unknown);
        _banks.MarkWritten(1u << (int)RegisterStamp.PixelUserData);
    }

    private (double AllocPerDraw, double RetainedPerFrame, double NsPerDraw, string Report) Run(Churn churn, ResolveMemoMode mode,
        Action<ShaderPipelineCache>? configure = null)
    {
        var cache = new ShaderPipelineCache(_guest.Context, _guest.Host, _guest.Compiler, _guest.Registry) { ResolveMemoMode = mode };
        configure?.Invoke(cache);
        long allocated = 0, retained = 0, ticks = 0;
        GraphicsResolveMemo.TakeReport();
        for (var frame = 0; frame < WarmFrames + MeasuredFrames; frame++)
        {
            var measured = frame >= WarmFrames;
            if (measured && frame == WarmFrames)
            {
                GraphicsResolveMemo.TakeReport();
                SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.TakeReadProofReport();
            }

            for (var draw = 0; draw < Draws; draw++)
            {
                if (frame == 0 || churn != Churn.None)
                {
                    WriteBlock(BlockOf(churn, draw, frame), draw, frame, churn == Churn.Words);
                }
            }

            if (measured)
            {
                // Everything so far into gen2: the gen0 collection after the frame promotes only what the frame left
                // reachable.
                GC.Collect(1, GCCollectionMode.Forced, blocking: true);
            }

            var before = GC.GetAllocatedBytesForCurrentThread();
            var started = Stopwatch.GetTimestamp();
            for (var draw = 0; draw < Draws; draw++)
            {
                PointPixelAt(BlockOf(churn, draw, frame));
                var programs = cache.GetGraphicsPrograms(_banks, [], pixelActive: true, depthBound: false);
                Assert.True(programs.Available);
            }

            var elapsed = Stopwatch.GetTimestamp() - started;
            var after = GC.GetAllocatedBytesForCurrentThread();
            if (measured)
            {
                ticks += elapsed;
                allocated += after - before;
                GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                var info = GC.GetGCMemoryInfo(GCKind.Ephemeral);
                retained += info.Generation == 0 ? info.PromotedBytes : 0;
            }
        }

        var report = GraphicsResolveMemo.TakeReport() + " " + SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.TakeReadProofReport();
        Assert.Empty(_fatal.Messages);
        GC.KeepAlive(cache);
        return ((double)allocated / (Draws * MeasuredFrames), (double)retained / MeasuredFrames,
            ticks * 1e9 / Stopwatch.Frequency / (Draws * MeasuredFrames), report);
    }

    // tuned false: the memo as it was before the retention changes (every rebuild and every new key's resolution kept).
    // Both arms run on caches of their own in ABBA order (three rounds) after a warm-up pass of each; the timing is each arm's
    // median.
    [Theory]
    [InlineData(Churn.None)]
    [InlineData(Churn.Words)]
    [InlineData(Churn.Keys)]
    public void ResolveMemoRetentionBench(Churn churn)
    {
        (double Alloc, double Retained, double Ns, string Report) Arm(bool tuned) => Run(churn, ResolveMemoMode.On, cache =>
        {
            cache.ResolveMemoHoldChurn = tuned;
            cache.ResolveMemoAdmitSeen = tuned;
        });

        Arm(false);
        Arm(true);
        var arms = new[] { new List<(double Alloc, double Retained, double Ns, string Report)>(), new() };
        for (var round = 0; round < 3; round++)
        {
            foreach (var tuned in new[] { false, true, true, false })
            {
                arms[tuned ? 1 : 0].Add(Arm(tuned));
            }
        }

        foreach (var tuned in new[] { false, true })
        {
            var runs = arms[tuned ? 1 : 0];
            var times = runs.Select(run => run.Ns).Order().ToArray();
            _output.WriteLine(FormattableString.Invariant(
                $"churn={churn} tuned={tuned} draws/frame={Draws} alloc_per_draw={runs[0].Alloc:F0} B retained_per_frame={runs[0].Retained / 1024:F1} KiB ({runs[0].Retained / Draws:F0} B/draw) resolve_median={times[times.Length / 2]:F0} ns/draw (min {times[0]:F0}, max {times[^1]:F0}, n={times.Length})"));
            _output.WriteLine(runs[^1].Report);
        }
    }

    // A rebuild shares what its words left as they were: the input description of the stage not rebuilt, and the word
    // arrays of the six V#s that kept their base; verify finds it exact.
    [Fact]
    public void ARebuildSharesWhatItsWordsLeftUnchanged()
    {
        var cache = new ShaderPipelineCache(_guest.Context, _guest.Host, _guest.Compiler, _guest.Registry)
        {
            ResolveMemoMode = ResolveMemoMode.Verify,
            ResolveMemoAdmitSeen = false,
        };
        var block = BlockOf(Churn.Words, 0, 0);
        WriteBlock(block, 0, 0, moveWords: true);
        PointPixelAt(block);
        var first = cache.GetGraphicsPrograms(_banks, [], pixelActive: true, depthBound: false);
        GraphicsResolveMemo.TakeReport();
        WriteBlock(block, 0, 1, moveWords: true);
        var rebuilt = cache.GetGraphicsPrograms(_banks, [], pixelActive: true, depthBound: false);
        var report = GraphicsResolveMemo.TakeReport();
        Assert.Contains("words_rebuilt=1,", report);
        Assert.Contains("verify_failed=0)", report);
        Assert.NotSame(first, rebuilt);
        Assert.Same(first.VertexInput, rebuilt.VertexInput);
        Assert.NotSame(first.PixelInput, rebuilt.PixelInput);
        var before = first.PixelInput.Stage.Resources.Buffers;
        var after = rebuilt.PixelInput.Stage.Resources.Buffers;
        Assert.Equal(Buffers, after.Length);
        Assert.Equal(Buffers - 2, after.Count(words => before.Any(held => ReferenceEquals(held, words))));
        Assert.Equal(2, after.Count(words => !before.Any(held => held.AsSpan().SequenceEqual(words))));
        Assert.Empty(_fatal.Messages);
    }

    // The same frames under verify: every hit and rebuild is resolved again in full and compared.
    [Fact]
    public void ResolveMemoChurnVerifiesUnderVerifyMode()
    {
        var (_, _, _, report) = Run(Churn.Words, ResolveMemoMode.Verify);
        Assert.Contains("verify_failed=0)", report);
        Assert.DoesNotContain("verify_checked=0,", report);
    }

    private void WriteQword(ulong address, ulong value)
    {
        var bytes = new byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        _guest.Write(address, bytes);
    }

    private void WriteHalf(ulong address, ushort value)
    {
        var bytes = new byte[sizeof(ushort)];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, value);
        _guest.Write(address, bytes);
    }
}
