// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using MslImageResourceClass = SharpEmu.ShaderCompiler.Resources.ImageResourceClass;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// What PrepareBindings decodes from a stage's program and snapshot alone, decoded on the resolving thread after the
/// stage's program resolved (IConcurrentResolveHost.PrepareBindingsAhead) and left on the snapshot: per image the
/// texture-memo key, per storage image the T#'s request (ImageRequestBuilders.Texture; a sampled image's is only
/// needed on a memo miss, and the worker builds it then), per sampler the native S# words, and the stage's
/// shader-data words before the dispatch thread limits and call data. A stage of more than 64 images or samplers
/// carries only its shader data. The worker takes it only when it was
/// built for the same program, layout, snapshot and shader base it binds; then it looks the images and samplers up and
/// encodes, and decodes nothing. Anything the build could not decode (a short descriptor, a user register outside the
/// draw's user data, a T# the builder refuses) is left out, and the worker decodes that part itself, failing there if
/// the failure is real.
/// </summary>
internal sealed class MetalBindingPacket(ShaderProgramInfo program, BindingLayout layout, ResourceSnapshot snapshot, ulong shaderBase)
{
    public const int MaximumEntries = 64;

    public readonly ShaderProgramInfo Program = program;
    public readonly BindingLayout Layout = layout;
    public readonly ResourceSnapshot Snapshot = snapshot;
    public readonly ulong ShaderBase = shaderBase;
    public MetalRenderHost.TextureMemoKey[]? MemoKeys;
    public TextureRequestResolution[]? Resolutions;
    public SamplerDescriptorWords[]? Samplers;

    // Bit n: image or sampler n was decoded.
    public ulong MemoKeyMask;
    public ulong ResolutionMask;
    public ulong SamplerMask;
    public uint[]? ShaderData;
    public long BuildTicks;

    public bool HasMemoKey(int index) => index < MaximumEntries && (MemoKeyMask & (1UL << index)) != 0;

    public bool HasResolution(int index) => index < MaximumEntries && (ResolutionMask & (1UL << index)) != 0;

    public bool HasSampler(int index) => index < MaximumEntries && (SamplerMask & (1UL << index)) != 0;
}

internal sealed partial class MetalRenderHost
{
    /// <summary>
    /// Draws' binding packets: the resolving thread decodes each resolved stage's image requests, memo keys, samplers
    /// and shader-data words (MetalBindingPacket), and the worker's PrepareBindings checks and uses them.
    /// SHARPEMU_METAL_V2_DRAW_PACKETS=0 decodes every draw stage on the worker, exactly as before.
    /// </summary>
    internal static readonly bool DrawBindingPackets =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DRAW_PACKETS") != "0";

    /// <summary>
    /// A prefetched dispatch's binding decode joins its program resolution in the same job (the compute stage's
    /// MetalBindingPacket). SHARPEMU_METAL_V2_DISPATCH_PACKETS=0 decodes every dispatch on the worker, exactly as before.
    /// </summary>
    internal static readonly bool DispatchBindingPackets =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DISPATCH_PACKETS") != "0";

    // Per host, for tests that compare both paths in one process.
    internal bool DrawPacketsEnabled = DrawBindingPackets;
    internal bool DispatchPacketsEnabled = DispatchBindingPackets;

    internal enum PacketRefusal
    {
        // The snapshot carries no packet: its stage was resolved on the worker, or the build is still running.
        None,
        // The packet was built for another program (a snapshot shared by programs) or another layout.
        Program,
        // The same program and snapshot under another shader base.
        ShaderBase,
        Count,
    }

    private const int PacketDraw = 0;
    private const int PacketDispatch = 1;

    // [kind]: built and reused (a matching packet already on the snapshot) on the resolving thread; taken, refused by
    // reason and the build time of the taken packets (the decode the worker no longer does) on the worker.
    private readonly long[] _packetsBuilt = new long[2];
    private readonly long[] _packetsReused = new long[2];
    private readonly long[] _packetsTaken = new long[2];
    private readonly long[] _packetSavedTicks = new long[2];
    private readonly long[,] _packetRefusals = new long[2, (int)PacketRefusal.Count];

    void IConcurrentResolveHost.PrepareBindingsAhead(in ShaderStageResources stage, bool dispatch)
    {
        if (!(dispatch ? DispatchPacketsEnabled : DrawPacketsEnabled) ||
            stage.Program is not { Resources: { } resources, Bindings: { } layout } program || stage.Resources is not { } snapshot)
        {
            return;
        }

        var kind = dispatch ? PacketDispatch : PacketDraw;
        if (Volatile.Read(ref snapshot.BindingPacket) is MetalBindingPacket held && ReferenceEquals(held.Program, program) &&
            ReferenceEquals(held.Layout, layout) && held.ShaderBase == stage.ShaderBase)
        {
            Interlocked.Increment(ref _packetsReused[kind]);
            return;
        }

        var started = Stopwatch.GetTimestamp();
        MetalBindingPacket? packet;
        try
        {
            packet = BuildBindingPacket(program, resources, layout, snapshot, stage.ShaderBase);
        }
        catch
        {
            // Never the job's failure: the worker decodes the stage itself.
            packet = null;
        }

        if (packet is null)
        {
            return;
        }

        packet.BuildTicks = Stopwatch.GetTimestamp() - started;
        Volatile.Write(ref snapshot.BindingPacket, packet);
        Interlocked.Increment(ref _packetsBuilt[kind]);
    }

    // PrepareBindings' decode, from the program and snapshot alone: null when they do not match (the worker fails there).
    internal static MetalBindingPacket? BuildBindingPacket(
        ShaderProgramInfo program, SpecializedResourceInfo resources, BindingLayout layout, ResourceSnapshot snapshot, ulong shaderBase)
    {
        var info = resources.Info;
        if (snapshot.Images.Length != info.Images.Count || snapshot.Samplers.Length != info.Samplers.Count ||
            snapshot.Buffers.Length != info.Buffers.Count)
        {
            return null;
        }

        var packet = new MetalBindingPacket(program, layout, snapshot, shaderBase);
        var images = info.Images.Count <= MetalBindingPacket.MaximumEntries ? info.Images.Count : 0;
        for (var index = 0; index < images; index++)
        {
            var words = snapshot.Images[index];
            var image = info.Images[index];
            var shape = ShapeOf(image);
            if (image.ResourceClass != MslImageResourceClass.Storage)
            {
                // Only TryTextureMemo reads a sampled image's key; a storage image is never memoised.
                if (words.Length >= 8)
                {
                    (packet.MemoKeys ??= new TextureMemoKey[images])[index] =
                        new TextureMemoKey(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7], shape);
                    packet.MemoKeyMask |= 1UL << index;
                }

                continue;
            }

            if (words.Length >= 4)
            {
                try
                {
                    var resolution = ImageRequestBuilders.Texture(words, shape);
                    (packet.Resolutions ??= new TextureRequestResolution[images])[index] = resolution;
                    packet.ResolutionMask |= 1UL << index;
                }
                catch
                {
                    // A T# the builder refuses: the worker resolves it and stops there.
                }
            }
        }

        var samplers = info.Samplers.Count <= MetalBindingPacket.MaximumEntries ? info.Samplers.Count : 0;
        for (var index = 0; index < samplers; index++)
        {
            var words = snapshot.Samplers[index];
            if (words.Length >= 4)
            {
                (packet.Samplers ??= new SamplerDescriptorWords[samplers])[index] = NativeSamplerWords(info.Samplers[index], words);
                packet.SamplerMask |= 1UL << index;
            }
        }

        var shaderData = layout.ShaderDataDwordCount == 0 ? [] : new uint[checked((int)layout.ShaderDataDwordCount)];
        if (WriteStageShaderData(shaderData, program, layout, info, snapshot, shaderBase))
        {
            packet.ShaderData = shaderData;
        }

        return packet;
    }

    // The packet the stage's snapshot carries when it was built for this program, layout and shader base, or null.
    private MetalBindingPacket? TakeBindingPacket(in ShaderStageResources stage, ShaderProgramInfo program, bool compute)
    {
        if (!(compute ? DispatchPacketsEnabled : DrawPacketsEnabled))
        {
            return null;
        }

        var kind = compute ? PacketDispatch : PacketDraw;
        var snapshot = stage.Resources;
        var packet = Volatile.Read(ref snapshot.BindingPacket) as MetalBindingPacket;
        var refusal = ClassifyPacket(packet, program, snapshot, stage.ShaderBase);
        if (refusal != PacketRefusal.Count)
        {
            Interlocked.Increment(ref _packetRefusals[kind, (int)refusal]);
            return null;
        }

        Interlocked.Increment(ref _packetsTaken[kind]);
        Interlocked.Add(ref _packetSavedTicks[kind], packet!.BuildTicks);
        return packet;
    }

    // Count when the packet stands for the stage; otherwise why it does not.
    internal static PacketRefusal ClassifyPacket(MetalBindingPacket? packet, ShaderProgramInfo program, ResourceSnapshot snapshot, ulong shaderBase) =>
        packet is null || !ReferenceEquals(packet.Snapshot, snapshot) ? PacketRefusal.None
        : !ReferenceEquals(packet.Program, program) || !ReferenceEquals(packet.Layout, program.Bindings) ? PacketRefusal.Program
        : packet.ShaderBase != shaderBase ? PacketRefusal.ShaderBase
        : PacketRefusal.Count;

    private string FormatPacketCounters()
    {
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        string Kind(int kind, string name) =>
            $"{name}_packets=(built={Interlocked.Exchange(ref _packetsBuilt[kind], 0)},reused={Interlocked.Exchange(ref _packetsReused[kind], 0)}," +
            $"taken={Interlocked.Exchange(ref _packetsTaken[kind], 0)},refused=(none={Interlocked.Exchange(ref _packetRefusals[kind, (int)PacketRefusal.None], 0)}," +
            $"program={Interlocked.Exchange(ref _packetRefusals[kind, (int)PacketRefusal.Program], 0)}," +
            $"base={Interlocked.Exchange(ref _packetRefusals[kind, (int)PacketRefusal.ShaderBase], 0)})," +
            $"saved_ms={(Interlocked.Exchange(ref _packetSavedTicks[kind], 0) * 1000.0 / Stopwatch.Frequency).ToString("F1", culture)})";
        return $" {Kind(PacketDraw, "draw")} {Kind(PacketDispatch, "dispatch")}";
    }
}
