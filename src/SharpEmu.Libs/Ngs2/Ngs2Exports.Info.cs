// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Libs.Ngs2;

/// <summary>
/// libSceNgs2 info/user-data queries, panning and listener geometry.
/// </summary>
public static partial class Ngs2Exports
{
    // PS5 numbering of the output-argument errors (the PS4 library used 0x804A00xx).
    private const int Ngs2ErrorInvalidOutAddress = unchecked((int)0x804A8010);
    private const int Ngs2ErrorInvalidOutSize = unchecked((int)0x804A8011);
    private const int Ngs2ErrorUnknownWaveformFormat = unchecked((int)0x804A8432);

    // Deviation: no reference pins the NGS2 code for an invalid input argument
    // (null input pointer, unsupported speaker count or matrix format), so these
    // paths report the generic invalid-argument code.
    private const int Ngs2ErrorInvalidArgumentUnpinned = (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;

    private const uint MinGrainSamples = 64;
    private const uint DefaultMaxGrainSamples = 512;

    private const int SystemOptionSize = 144;
    private const int SystemInfoSize = 184;
    private const int RackOptionSize = 176;
    private const int RackInfoSize = 216;
    private const int ContextBufferInfoSize = 64;
    private const int NameSize = 64;
    private const int PanWorkSize = 40;
    private const int PanParamSize = 16;
    private const int MaxPanParams = 8;
    private const int GeomListenerParamSize = 60;
    private const int GeomListenerWorkSize = 96;
    private const int GeomSourceParamSize = 108;
    private const int WaveformFormatSize = 24;
    private const int WaveformBlockSize = 40;

    private const uint RackIdSampler = 0x1000;
    private const uint RackIdSubmixer = 0x2000;
    private const uint RackIdReverb = 0x2001;
    private const uint RackIdMastering = 0x3000;

    private const uint WaveformTypePcmI16 = 0x12;
    private const uint WaveformTypePcmF32 = 0x18;
    private const uint WaveformTypeAtrac9 = 0x40;

    private const float Degrees = 360.0f;
    private const float Radians = 2.0f * MathF.PI;
    private const float DefaultSoundSpeed = 343.0f;

    /// <summary>What sceNgs2SystemGetInfo reports about a system beyond its live state.</summary>
    private sealed record SystemDescription(byte[] Name, uint MaxGrainSamples, byte[] BufferInfo)
    {
        public static SystemDescription Default { get; } =
            new(new byte[NameSize], DefaultMaxGrainSamples, new byte[ContextBufferInfoSize]);
    }

    /// <summary>The SceNgs2RackOption limits sceNgs2RackGetInfo reports.</summary>
    private sealed record RackDescription(
        byte[] Name,
        byte[] BufferInfo,
        uint Uid,
        uint MaxGrainSamples,
        uint MaxVoices,
        uint MaxChannelWorks,
        uint MaxInputs,
        uint MaxMatrices,
        uint MaxPorts)
    {
        public static RackDescription Default { get; } =
            new(new byte[NameSize], new byte[ContextBufferInfoSize], 0, DefaultMaxGrainSamples, 1, 0, 0, 1, 8);
    }

    // SceNgs2SystemOption { size_t size; char name[64]; uintptr_t jobScheduler[4];
    // uint32_t flags, maxGrainSamples, numGrainSamples, sampleRate, maxVoiceChannels; ... }.
    private static SystemDescription ReadSystemDescription(CpuContext ctx, ulong optionAddress, ulong bufferInfoAddress)
    {
        var name = new byte[NameSize];
        var maxGrain = DefaultMaxGrainSamples;
        if (optionAddress != 0 &&
            ctx.TryReadUInt64(optionAddress, out var size) &&
            size == SystemOptionSize &&
            ctx.Memory.TryRead(optionAddress + 8, name) &&
            ctx.TryReadUInt32(optionAddress + 108, out var optionMaxGrain))
        {
            maxGrain = optionMaxGrain;
        }
        else
        {
            Array.Clear(name);
        }

        return new SystemDescription(name, maxGrain, ReadBufferInfo(ctx, bufferInfoAddress));
    }

    // SceNgs2RackOption { size_t size; char name[64]; uint32_t flags, maxGrainSamples,
    // maxVoices, maxInputDelayBlocks, maxMatrices, maxPorts, ...; } (176 bytes), followed
    // by the rack-specific fields (sampler: maxChannelWorks at +176; submixer:
    // maxChannels, maxEnvelopePoints, maxFilters, maxInputs at +176..+188).
    private static RackDescription ReadRackDescription(
        CpuContext ctx,
        uint rackId,
        ulong optionAddress,
        ulong bufferInfoAddress,
        uint uid)
    {
        var name = new byte[NameSize];
        var maxGrain = DefaultMaxGrainSamples;
        uint maxVoices = rackId == RackIdSampler ? 256u : 1u;
        uint maxChannelWorks = rackId == RackIdSampler ? 256u : 0u;
        uint maxInputs = rackId == RackIdSubmixer ? 1u : 0u;
        uint maxMatrices = rackId == RackIdMastering ? 0u : 1u;
        uint maxPorts = rackId == RackIdMastering ? 0u : 8u;

        if (optionAddress != 0 &&
            ctx.TryReadUInt64(optionAddress, out var size) &&
            size >= RackOptionSize &&
            ctx.Memory.TryRead(optionAddress + 8, name) &&
            ctx.TryReadUInt32(optionAddress + 76, out var optionMaxGrain) &&
            ctx.TryReadUInt32(optionAddress + 80, out var optionMaxVoices) &&
            ctx.TryReadUInt32(optionAddress + 88, out var optionMaxMatrices) &&
            ctx.TryReadUInt32(optionAddress + 92, out var optionMaxPorts))
        {
            maxGrain = optionMaxGrain;
            maxVoices = optionMaxVoices;
            maxMatrices = optionMaxMatrices;
            maxPorts = optionMaxPorts;
            if (rackId == RackIdSampler && ctx.TryReadUInt32(optionAddress + 176, out var channelWorks))
            {
                maxChannelWorks = channelWorks;
            }

            if (rackId == RackIdSubmixer && ctx.TryReadUInt32(optionAddress + 188, out var inputs))
            {
                maxInputs = inputs;
            }
        }
        else
        {
            Array.Clear(name);
        }

        return new RackDescription(
            name,
            ReadBufferInfo(ctx, bufferInfoAddress),
            uid,
            maxGrain,
            maxVoices,
            maxChannelWorks,
            maxInputs,
            maxMatrices,
            maxPorts);
    }

    // The non-allocator create calls take a caller-owned SceNgs2ContextBufferInfo;
    // the allocator forms report an empty one because no buffer is requested.
    private static byte[] ReadBufferInfo(CpuContext ctx, ulong address)
    {
        var info = new byte[ContextBufferInfoSize];
        if (address == 0 || !ctx.Memory.TryRead(address, info))
        {
            Array.Clear(info);
        }

        return info;
    }

    [SysAbiExport(
        Nid = "vU7TQ62pItw",
        ExportName = "sceNgs2SystemGetInfo",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2SystemGetInfo(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var infoAddress = ctx[CpuRegister.Rsi];
        var infoSize = ctx[CpuRegister.Rdx];
        if (infoAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        if (infoSize != SystemInfoSize)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutSize);
        }

        Span<byte> info = stackalloc byte[SystemInfoSize];
        info.Clear();
        lock (StateGate)
        {
            if (!Systems.TryGetValue(handle, out var system))
            {
                return SetReturn(ctx, OrbisNgs2ErrorInvalidSystemHandle);
            }

            // SceNgs2SystemInfo: name[64], systemHandle, bufferInfo (64), uid,
            // min/maxGrainSamples, stateFlags, rackCount, lastRenderRatio,
            // lastRenderTick, renderCount, sampleRate, numGrainSamples.
            system.Description.Name.CopyTo(info);
            BinaryPrimitives.WriteUInt64LittleEndian(info[64..], handle);
            system.Description.BufferInfo.CopyTo(info[72..]);
            BinaryPrimitives.WriteUInt32LittleEndian(info[136..], system.Uid);
            BinaryPrimitives.WriteUInt32LittleEndian(info[140..], MinGrainSamples);
            BinaryPrimitives.WriteUInt32LittleEndian(info[144..], system.Description.MaxGrainSamples);
            BinaryPrimitives.WriteUInt32LittleEndian(info[148..], 1);
            BinaryPrimitives.WriteUInt32LittleEndian(
                info[152..],
                unchecked((uint)Racks.Values.Count(rack => rack.SystemHandle == handle)));
            BinaryPrimitives.WriteInt64LittleEndian(info[168..], system.RenderCount);
            BinaryPrimitives.WriteUInt32LittleEndian(info[176..], unchecked((uint)system.SampleRate));
            BinaryPrimitives.WriteUInt32LittleEndian(info[180..], unchecked((uint)system.GrainSamples));
        }

        return SetReturn(ctx, ctx.Memory.TryWrite(infoAddress, info) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    [SysAbiExport(
        Nid = "M4LYATRhRUE",
        ExportName = "sceNgs2RackGetInfo",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2RackGetInfo(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var infoAddress = ctx[CpuRegister.Rsi];
        var infoSize = ctx[CpuRegister.Rdx];
        if (infoAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        if (infoSize != RackInfoSize)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutSize);
        }

        Span<byte> info = stackalloc byte[RackInfoSize];
        info.Clear();
        lock (StateGate)
        {
            if (!Racks.TryGetValue(handle, out var rack))
            {
                return SetReturn(ctx, OrbisNgs2ErrorInvalidRackHandle);
            }

            var description = rack.Description;
            var activeVoices = Voices.Values.Count(
                voice => voice.RackHandle == handle && (voice.Playing || voice.InUse));
            var renderCount = Systems.TryGetValue(rack.SystemHandle, out var system) ? system.RenderCount : 0;

            // SceNgs2RackInfo: name[64], rackHandle, bufferInfo (64), ownerSystemHandle,
            // type, rackId, uid, min/maxGrainSamples, maxVoices, maxChannelWorks,
            // maxInputs, maxMatrices, maxPorts, stateFlags, lastProcessRatio,
            // lastProcessedTick, renderCount, activeVoiceCount, activeChannelWorkCount.
            description.Name.CopyTo(info);
            BinaryPrimitives.WriteUInt64LittleEndian(info[64..], handle);
            description.BufferInfo.CopyTo(info[72..]);
            BinaryPrimitives.WriteUInt64LittleEndian(info[136..], rack.SystemHandle);
            BinaryPrimitives.WriteUInt32LittleEndian(info[144..], rack.RackId >> 12);
            BinaryPrimitives.WriteUInt32LittleEndian(info[148..], rack.RackId);
            BinaryPrimitives.WriteUInt32LittleEndian(info[152..], description.Uid);
            BinaryPrimitives.WriteUInt32LittleEndian(info[156..], MinGrainSamples);
            BinaryPrimitives.WriteUInt32LittleEndian(info[160..], description.MaxGrainSamples);
            BinaryPrimitives.WriteUInt32LittleEndian(info[164..], description.MaxVoices);
            BinaryPrimitives.WriteUInt32LittleEndian(info[168..], description.MaxChannelWorks);
            BinaryPrimitives.WriteUInt32LittleEndian(info[172..], description.MaxInputs);
            BinaryPrimitives.WriteUInt32LittleEndian(info[176..], description.MaxMatrices);
            BinaryPrimitives.WriteUInt32LittleEndian(info[180..], description.MaxPorts);
            BinaryPrimitives.WriteUInt32LittleEndian(info[184..], 1);
            BinaryPrimitives.WriteInt64LittleEndian(info[200..], renderCount);
            BinaryPrimitives.WriteUInt32LittleEndian(info[208..], unchecked((uint)activeVoices));
        }

        return SetReturn(ctx, ctx.Memory.TryWrite(infoAddress, info) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    // Rack lock/unlock bracket parameter updates against rendering. Every NGS2 call
    // here already runs under StateGate, so they validate the handle like
    // sceNgs2SystemLock/Unlock do.
    [SysAbiExport(
        Nid = "MzTa7VLjogY",
        ExportName = "sceNgs2RackLock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2RackLock(CpuContext ctx) => ValidateRack(ctx);

    [SysAbiExport(
        Nid = "++YZ7P9e87U",
        ExportName = "sceNgs2RackUnlock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2RackUnlock(CpuContext ctx) => ValidateRack(ctx);

    private static int ValidateRack(CpuContext ctx)
    {
        lock (StateGate)
        {
            return SetReturn(
                ctx,
                Racks.ContainsKey(ctx[CpuRegister.Rdi]) ? 0 : OrbisNgs2ErrorInvalidRackHandle);
        }
    }

    [SysAbiExport(
        Nid = "GZB2v0XnG0k",
        ExportName = "sceNgs2SystemSetUserData",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2SystemSetUserData(CpuContext ctx)
    {
        lock (StateGate)
        {
            if (!Systems.TryGetValue(ctx[CpuRegister.Rdi], out var system))
            {
                return SetReturn(ctx, OrbisNgs2ErrorInvalidSystemHandle);
            }

            system.UserData = ctx[CpuRegister.Rsi];
            return SetReturn(ctx, 0);
        }
    }

    [SysAbiExport(
        Nid = "4lFaRxd-aLs",
        ExportName = "sceNgs2SystemGetUserData",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2SystemGetUserData(CpuContext ctx)
    {
        ulong userData;
        lock (StateGate)
        {
            if (!Systems.TryGetValue(ctx[CpuRegister.Rdi], out var system))
            {
                return SetReturn(ctx, OrbisNgs2ErrorInvalidSystemHandle);
            }

            userData = system.UserData;
        }

        return WriteUserData(ctx, ctx[CpuRegister.Rsi], userData);
    }

    [SysAbiExport(
        Nid = "JNTMIaBIbV4",
        ExportName = "sceNgs2RackSetUserData",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2RackSetUserData(CpuContext ctx)
    {
        lock (StateGate)
        {
            if (!Racks.TryGetValue(ctx[CpuRegister.Rdi], out var rack))
            {
                return SetReturn(ctx, OrbisNgs2ErrorInvalidRackHandle);
            }

            rack.UserData = ctx[CpuRegister.Rsi];
            return SetReturn(ctx, 0);
        }
    }

    [SysAbiExport(
        Nid = "Mn4XNDg03XY",
        ExportName = "sceNgs2RackGetUserData",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2RackGetUserData(CpuContext ctx)
    {
        ulong userData;
        lock (StateGate)
        {
            if (!Racks.TryGetValue(ctx[CpuRegister.Rdi], out var rack))
            {
                return SetReturn(ctx, OrbisNgs2ErrorInvalidRackHandle);
            }

            userData = rack.UserData;
        }

        return WriteUserData(ctx, ctx[CpuRegister.Rsi], userData);
    }

    private static int WriteUserData(CpuContext ctx, ulong address, ulong userData) =>
        SetReturn(
            ctx,
            address != 0 && ctx.TryWriteUInt64(address, userData) ? 0 : Ngs2ErrorInvalidOutAddress);

    [SysAbiExport(
        Nid = "AQkj7C0f3PY",
        ExportName = "sceNgs2SystemResetOption",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2SystemResetOption(CpuContext ctx)
    {
        var optionAddress = ctx[CpuRegister.Rdi];
        if (optionAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        // Default SceNgs2SystemOption: unnamed, no flags, 512-sample maximum grain,
        // 256-sample grain, 48 kHz.
        Span<byte> option = stackalloc byte[SystemOptionSize];
        option.Clear();
        BinaryPrimitives.WriteUInt64LittleEndian(option, SystemOptionSize);
        BinaryPrimitives.WriteUInt32LittleEndian(option[108..], DefaultMaxGrainSamples);
        BinaryPrimitives.WriteUInt32LittleEndian(option[112..], DefaultGrainSamples);
        BinaryPrimitives.WriteUInt32LittleEndian(option[116..], DefaultSampleRate);
        return SetReturn(ctx, ctx.Memory.TryWrite(optionAddress, option) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    // Speaker layouts of the panner. InputOrder maps panner speaker i (clockwise
    // from the front) to the index of its angle in the caller's array; MixChannel
    // is that speaker's channel in the layout's native mix (FL, FR, C, LFE, SL,
    // SR, BL, BR); DefaultDegrees are the SDK's default speaker angles.
    private sealed record PanLayout(int[] InputOrder, int[] MixChannel, float[] DefaultDegrees, int MixChannels);

    private static readonly PanLayout StereoPan = new([1, 0], [1, 0], [90, 270], 2);
    private static readonly PanLayout QuadPan = new([1, 3, 2, 0], [1, 5, 4, 0], [40, 110, 250, 320], 6);
    private static readonly PanLayout FivePan = new([2, 1, 4, 3, 0], [2, 1, 5, 4, 0], [0, 40, 110, 250, 320], 6);
    private static readonly PanLayout SevenPan = new(
        [2, 1, 4, 6, 5, 3, 0],
        [2, 1, 5, 7, 6, 4, 0],
        [0, 40, 110, 150, 210, 250, 320],
        8);

    private static PanLayout? PanLayoutFor(uint speakers) => speakers switch
    {
        2 => StereoPan,
        4 => QuadPan,
        5 => FivePan,
        7 => SevenPan,
        _ => null,
    };

    private static bool IsUnitAngle(float unit) => unit == Degrees || unit == Radians;

    // Speaker angles wrap one turn at a time into [0, unit); an angle a whole
    // number of extra turns past the unit lands exactly on the unit and is
    // rejected, as is one too large for a turn to change its float value.
    private static bool TryWrapSpeakerAngle(ref float angle, float unit)
    {
        while (angle < 0.0f)
        {
            var wrapped = angle + unit;
            if (wrapped == angle)
            {
                return false;
            }

            angle = wrapped;
        }

        while (angle >= unit)
        {
            var wrapped = angle - unit;
            if (!(unit < angle) || wrapped == angle)
            {
                return false;
            }

            angle = wrapped;
        }

        return true;
    }

    [SysAbiExport(
        Nid = "xa8oL9dmXkM",
        ExportName = "sceNgs2PanInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2PanInit(CpuContext ctx)
    {
        // (SceNgs2PanWork* work, const float* speakerAngles, float unitAngle, uint32_t numSpeakers).
        var workAddress = ctx[CpuRegister.Rdi];
        var anglesAddress = ctx[CpuRegister.Rsi];
        var speakers = unchecked((uint)ctx[CpuRegister.Rdx]);
        ctx.GetXmmRegister(0, out var unitBits, out _);
        var unit = BitConverter.Int32BitsToSingle(unchecked((int)(uint)unitBits));

        if (workAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        var layout = PanLayoutFor(speakers);
        if (layout is null || (anglesAddress != 0 && !IsUnitAngle(unit)))
        {
            return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
        }

        var angles = new float[8];
        float workUnit;
        if (anglesAddress == 0)
        {
            // The default layout ignores any unit other than radians. Its closing
            // entry stays in degrees even when the speakers are in radians.
            workUnit = unit == Radians ? Radians : Degrees;
            var scale = workUnit / Degrees;
            for (var i = 0; i < speakers; i++)
            {
                angles[i] = layout.DefaultDegrees[i] * scale;
            }

            angles[speakers] = layout.DefaultDegrees[0] + Degrees;
        }
        else
        {
            workUnit = unit;
            Span<byte> input = stackalloc byte[sizeof(float) * 8];
            if (!ctx.Memory.TryRead(anglesAddress, input[..(int)(speakers * sizeof(float))]))
            {
                return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
            }

            for (var i = 0; i < speakers; i++)
            {
                var angle = BinaryPrimitives.ReadSingleLittleEndian(input[(layout.InputOrder[i] * sizeof(float))..]);
                if (!float.IsFinite(angle) || !TryWrapSpeakerAngle(ref angle, unit))
                {
                    return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
                }

                angles[i] = angle;
            }

            // The entry after the last speaker repeats the first one a full turn
            // later, closing the circle the panner interpolates around.
            angles[speakers] = angles[0] + unit;
        }

        // SceNgs2PanWork { float speakerAngles[8]; float unitAngle; uint32_t numSpeakers; }.
        Span<byte> work = stackalloc byte[PanWorkSize];
        work.Clear();
        for (var i = 0; i < angles.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(work[(i * sizeof(float))..], angles[i]);
        }

        BinaryPrimitives.WriteSingleLittleEndian(work[32..], workUnit);
        BinaryPrimitives.WriteUInt32LittleEndian(work[36..], speakers);
        return SetReturn(ctx, ctx.Memory.TryWrite(workAddress, work) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    [SysAbiExport(
        Nid = "gbMKV+8Enuo",
        ExportName = "sceNgs2PanGetVolumeMatrix",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2PanGetVolumeMatrix(CpuContext ctx)
    {
        // (SceNgs2PanWork* work, const SceNgs2PanParam* params, uint32_t numParams,
        //  uint32_t matrixFormat, float* outVolumeMatrix).
        var workAddress = ctx[CpuRegister.Rdi];
        var paramsAddress = ctx[CpuRegister.Rsi];
        var paramCount = unchecked((uint)ctx[CpuRegister.Rdx]);
        var matrixFormat = unchecked((uint)ctx[CpuRegister.Rcx]);
        var outputAddress = ctx[CpuRegister.R8];

        if (workAddress == 0 || paramsAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
        }

        if (outputAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        Span<byte> work = stackalloc byte[PanWorkSize];
        if (matrixFormat is not (1 or 2 or 6 or 8) ||
            paramCount > MaxPanParams ||
            !ctx.Memory.TryRead(workAddress, work))
        {
            return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
        }

        var speakerAngles = new float[8];
        for (var i = 0; i < speakerAngles.Length; i++)
        {
            speakerAngles[i] = BinaryPrimitives.ReadSingleLittleEndian(work[(i * sizeof(float))..]);
        }

        var unit = BinaryPrimitives.ReadSingleLittleEndian(work[32..]);
        var speakers = BinaryPrimitives.ReadUInt32LittleEndian(work[36..]);
        var layout = PanLayoutFor(speakers);
        if (layout is null || !IsUnitAngle(unit) || !float.IsFinite(speakerAngles[0]))
        {
            return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
        }

        var panParams = new PanParam[paramCount];
        Span<byte> param = stackalloc byte[PanParamSize];
        for (var p = 0; p < paramCount; p++)
        {
            if (!ctx.Memory.TryRead(paramsAddress + (ulong)(p * PanParamSize), param))
            {
                return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
            }

            panParams[p] = new PanParam(
                BinaryPrimitives.ReadSingleLittleEndian(param),
                BinaryPrimitives.ReadSingleLittleEndian(param[4..]),
                BinaryPrimitives.ReadSingleLittleEndian(param[8..]),
                BinaryPrimitives.ReadSingleLittleEndian(param[12..]));
            if (!float.IsFinite(panParams[p].Angle) ||
                float.IsNaN(panParams[p].Distance) ||
                !TryWrapPanAngle(panParams[p].Angle, speakerAngles[0], unit, out _))
            {
                return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
            }
        }

        speakerAngles[speakers] = speakerAngles[0] + unit;
        var matrix = ComputePanMatrix(speakerAngles, unit, layout, (int)speakers, panParams, (int)matrixFormat);
        var output = new byte[matrix.Length * sizeof(float)];
        for (var i = 0; i < matrix.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(output.AsSpan(i * sizeof(float)), matrix[i]);
        }

        return SetReturn(ctx, ctx.Memory.TryWrite(outputAddress, output) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    private readonly record struct PanParam(float Angle, float Distance, float FbwLevel, float LfeLevel);

    /// <summary>
    /// Equal-power panning between the two speakers that bracket each source angle,
    /// plus the diametrically opposite direction weighted by distance (distance 1
    /// is fully toward the angle, 0 spreads equally, -1 is fully opposite). The
    /// per-speaker gains land on the layout's native mix (FBW scaled, LFE set
    /// directly) and are folded into the requested matrix format with the default
    /// channel map. The output is destination-major: [dest * numParams + param].
    /// </summary>
    private static float[] ComputePanMatrix(
        float[] speakerAngles,
        float unit,
        PanLayout layout,
        int speakers,
        PanParam[] panParams,
        int matrixFormat)
    {
        const int lfeChannel = 3;
        var matrix = new float[panParams.Length * matrixFormat];
        var first = speakerAngles[0];
        for (var p = 0; p < panParams.Length; p++)
        {
            var param = panParams[p];
            var distance = Math.Clamp(param.Distance, -1.0f, 1.0f);
            var towardWeight = MathF.Sin((distance + 1.0f) * 0.25f * MathF.PI);
            var oppositeWeight = MathF.Sin((1.0f - distance) * 0.25f * MathF.PI);

            TryWrapPanAngle(param.Angle, first, unit, out var angle);
            var opposite = angle + (unit * 0.5f);
            if (!(opposite < first + unit))
            {
                opposite -= unit;
            }

            var gains = new float[9];
            for (var speaker = 1; speaker <= speakers; speaker++)
            {
                AddBetweenSpeakers(speakerAngles, speaker, angle, towardWeight, gains);
                AddBetweenSpeakers(speakerAngles, speaker, opposite, oppositeWeight, gains);
            }

            var mix = new float[8];
            mix[layout.MixChannel[0]] = gains[0] + gains[speakers];
            for (var speaker = 1; speaker < speakers; speaker++)
            {
                mix[layout.MixChannel[speaker]] = gains[speaker];
            }

            for (var channel = 0; channel < mix.Length; channel++)
            {
                mix[channel] = channel == lfeChannel ? param.LfeLevel : mix[channel] * param.FbwLevel;
            }

            for (var dest = 0; dest < matrixFormat; dest++)
            {
                var level = 0.0f;
                for (var channel = 0; channel < layout.MixChannels; channel++)
                {
                    level += DefaultChannelLevel(channel, matrixFormat, dest) * mix[channel];
                }

                matrix[(dest * panParams.Length) + p] = level;
            }
        }

        return matrix;
    }

    // Source angles wrap one turn at a time into [first speaker, first speaker + unit).
    private static bool TryWrapPanAngle(float angle, float first, float unit, out float wrapped)
    {
        wrapped = angle;
        while (wrapped < first || wrapped >= first + unit)
        {
            var before = wrapped;
            if (wrapped >= first + unit)
            {
                wrapped -= unit;
            }

            if (wrapped < first)
            {
                wrapped += unit;
            }

            if (wrapped == before)
            {
                return false;
            }
        }

        return true;
    }

    private static void AddBetweenSpeakers(float[] speakerAngles, int speaker, float angle, float weight, float[] gains)
    {
        var from = speakerAngles[speaker - 1];
        var to = speakerAngles[speaker];
        if (!(angle >= from) || !(to > angle))
        {
            return;
        }

        var position = (angle - from) / (to - from);
        gains[speaker - 1] += MathF.Sin((1.0f - position) * MathF.PI * 0.5f) * weight;
        gains[speaker] += MathF.Sin(position * MathF.PI * 0.5f) * weight;
    }

    /// <summary>
    /// Default NGS2 channel map from a 2-, 6- or 8-channel mix (FL, FR, C, LFE, SL,
    /// SR, BL, BR) into a mono, stereo, 5.1 or 7.1 destination. Surround channels
    /// fold into stereo at -6 dB, the centre at -3 dB, and the LFE is dropped
    /// below 5.1.
    /// </summary>
    private static float DefaultChannelLevel(int role, int destChannels, int dest)
    {
        // Source channel role: 0 FL, 1 FR, 2 C, 3 LFE, 4 SL, 5 SR, 6 BL, 7 BR (a
        // stereo mix only has the first two, a 5.1 mix the first six).
        const float halfPower = 0.70710677f;
        return destChannels switch
        {
            1 => role switch
            {
                0 or 1 => halfPower,
                2 => 1.0f,
                3 => 0.0f,
                _ => 0.5f,
            },
            2 => role switch
            {
                0 => dest == 0 ? 1.0f : 0.0f,
                1 => dest == 1 ? 1.0f : 0.0f,
                2 => halfPower,
                3 => 0.0f,
                4 or 6 => dest == 0 ? 0.5f : 0.0f,
                _ => dest == 1 ? 0.5f : 0.0f,
            },
            6 => dest == role switch
            {
                6 => 4,
                7 => 5,
                _ => role,
            } ? 1.0f : 0.0f,
            8 => dest == role ? 1.0f : 0.0f,
            _ => 0.0f,
        };
    }

    [SysAbiExport(
        Nid = "1WsleK-MTkE",
        ExportName = "sceNgs2GeomCalcListener",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2GeomCalcListener(CpuContext ctx)
    {
        // (const SceNgs2GeomListenerParam* param, SceNgs2GeomListenerWork* outWork, uint32_t flags).
        var paramAddress = ctx[CpuRegister.Rdi];
        var workAddress = ctx[CpuRegister.Rsi];
        var flags = unchecked((uint)ctx[CpuRegister.Rdx]);
        if (paramAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
        }

        if (workAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        Span<byte> param = stackalloc byte[GeomListenerParamSize];
        if (flags > 1 || !ctx.Memory.TryRead(paramAddress, param))
        {
            return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
        }

        // SceNgs2GeomListenerParam: position, orientFront, orientUp, velocity
        // (3 floats each), soundSpeed.
        var position = ReadVector(param);
        var front = ReadVector(param[12..]);
        var up = ReadVector(param[24..]);
        var velocity = ReadVector(param[36..]);
        var soundSpeed = BinaryPrimitives.ReadSingleLittleEndian(param[48..]);
        if (!IsFinite(position) || !IsFinite(front) || !IsFinite(up) || !IsFinite(velocity) ||
            !float.IsFinite(soundSpeed) || soundSpeed <= 0.0f ||
            !TryNormalize(front, out var frontAxis) ||
            !TryNormalize(up, out var upAxis) ||
            !TryNormalize(Cross(upAxis, frontAxis), out var rightAxis))
        {
            return SetReturn(ctx, Ngs2ErrorInvalidArgumentUnpinned);
        }

        // Re-orthogonalize up against front, then build the world-to-listener
        // transform with rows right, up, front (right negated for flags == 1, the
        // right-handed coordinate system).
        upAxis = Cross(frontAxis, rightAxis);
        if (flags == 1)
        {
            rightAxis = (-rightAxis.X, -rightAxis.Y, -rightAxis.Z);
        }

        Span<byte> work = stackalloc byte[GeomListenerWorkSize];
        work.Clear();
        var axes = new[] { rightAxis, upAxis, frontAxis };
        for (var row = 0; row < axes.Length; row++)
        {
            var axis = axes[row];
            var rowBase = row * 16;
            BinaryPrimitives.WriteSingleLittleEndian(work[rowBase..], (float)axis.X);
            BinaryPrimitives.WriteSingleLittleEndian(work[(rowBase + 4)..], (float)axis.Y);
            BinaryPrimitives.WriteSingleLittleEndian(work[(rowBase + 8)..], (float)axis.Z);
            BinaryPrimitives.WriteSingleLittleEndian(work[(rowBase + 12)..], (float)-Dot(axis, position));
        }

        BinaryPrimitives.WriteSingleLittleEndian(work[60..], 1.0f);
        param[36..48].CopyTo(work[64..]);
        BinaryPrimitives.WriteSingleLittleEndian(work[76..], soundSpeed);
        BinaryPrimitives.WriteUInt32LittleEndian(work[80..], flags);
        return SetReturn(ctx, ctx.Memory.TryWrite(workAddress, work) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    private static (double X, double Y, double Z) ReadVector(ReadOnlySpan<byte> bytes) =>
        (BinaryPrimitives.ReadSingleLittleEndian(bytes),
         BinaryPrimitives.ReadSingleLittleEndian(bytes[4..]),
         BinaryPrimitives.ReadSingleLittleEndian(bytes[8..]));

    private static bool IsFinite((double X, double Y, double Z) v) =>
        double.IsFinite(v.X) && double.IsFinite(v.Y) && double.IsFinite(v.Z);

    private static double Dot((double X, double Y, double Z) a, (double X, double Y, double Z) b) =>
        (a.X * b.X) + (a.Y * b.Y) + (a.Z * b.Z);

    private static (double X, double Y, double Z) Cross((double X, double Y, double Z) a, (double X, double Y, double Z) b) =>
        ((a.Y * b.Z) - (a.Z * b.Y), (a.Z * b.X) - (a.X * b.Z), (a.X * b.Y) - (a.Y * b.X));

    private static bool TryNormalize((double X, double Y, double Z) v, out (double X, double Y, double Z) unitVector)
    {
        var length = Math.Sqrt(Dot(v, v));
        unitVector = length > 0 ? (v.X / length, v.Y / length, v.Z / length) : default;
        return length > 0;
    }

    [SysAbiExport(
        Nid = "7Lcfo8SmpsU",
        ExportName = "sceNgs2GeomResetListenerParam",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2GeomResetListenerParam(CpuContext ctx)
    {
        var address = ctx[CpuRegister.Rdi];
        if (address == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        // At the origin, facing +Z with +Y up, at rest, speed of sound 343 m/s.
        Span<byte> param = stackalloc byte[GeomListenerParamSize];
        param.Clear();
        BinaryPrimitives.WriteSingleLittleEndian(param[20..], 1.0f); // orientFront.z
        BinaryPrimitives.WriteSingleLittleEndian(param[28..], 1.0f); // orientUp.y
        BinaryPrimitives.WriteSingleLittleEndian(param[48..], DefaultSoundSpeed);
        return SetReturn(ctx, ctx.Memory.TryWrite(address, param) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    [SysAbiExport(
        Nid = "0lbbayqDNoE",
        ExportName = "sceNgs2GeomResetSourceParam",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2GeomResetSourceParam(CpuContext ctx)
    {
        var address = ctx[CpuRegister.Rdi];
        if (address == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        // SceNgs2GeomSourceParam: position, velocity, direction (+Z), an omni cone
        // (levels 1 over 360 degrees), rolloff {model 0, maxDistance 1e6, factor 1,
        // reference 1}, doppler/fbw/lfe/max level 1, min level and radius 0, a
        // stereo destination (2 speakers, matrix format 2).
        Span<byte> param = stackalloc byte[GeomSourceParamSize];
        param.Clear();
        BinaryPrimitives.WriteSingleLittleEndian(param[32..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[36..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[40..], 360.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[44..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[48..], 360.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[56..], 1000000.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[60..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[64..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[68..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[72..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[76..], 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(param[80..], 1.0f);
        BinaryPrimitives.WriteUInt32LittleEndian(param[92..], 2);
        BinaryPrimitives.WriteUInt32LittleEndian(param[96..], 2);
        return SetReturn(ctx, ctx.Memory.TryWrite(address, param) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    [SysAbiExport(
        Nid = "3pCNbVM11UA",
        ExportName = "sceNgs2CalcWaveformBlock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceNgs2")]
    public static int Ngs2CalcWaveformBlock(CpuContext ctx)
    {
        // (const SceNgs2WaveformFormat* format, uint32_t samplePos, uint32_t numSamples,
        //  SceNgs2WaveformBlock* outBlock): the byte range of the waveform holding
        // numSamples frames from samplePos, and how many decoded frames to skip.
        var formatAddress = ctx[CpuRegister.Rdi];
        var samplePosition = unchecked((uint)ctx[CpuRegister.Rsi]);
        var sampleCount = unchecked((uint)ctx[CpuRegister.Rdx]);
        var blockAddress = ctx[CpuRegister.Rcx];
        if (blockAddress == 0)
        {
            return SetReturn(ctx, Ngs2ErrorInvalidOutAddress);
        }

        Span<byte> block = stackalloc byte[WaveformBlockSize];
        block.Clear();
        var status = CalcWaveformBlock(ctx, formatAddress, samplePosition, sampleCount, block);
        if (status != 0)
        {
            return SetReturn(ctx, status);
        }

        return SetReturn(ctx, ctx.Memory.TryWrite(blockAddress, block) ? 0 : Ngs2ErrorInvalidOutAddress);
    }

    private static int CalcWaveformBlock(
        CpuContext ctx,
        ulong formatAddress,
        uint samplePosition,
        uint sampleCount,
        Span<byte> block)
    {
        // SceNgs2WaveformFormat { waveformType, numChannels, sampleRate, configData,
        // frameOffset, frameMargin }.
        Span<byte> format = stackalloc byte[WaveformFormatSize];
        if (formatAddress == 0 || !ctx.Memory.TryRead(formatAddress, format))
        {
            return OrbisNgs2ErrorInvalidWaveformFormat;
        }

        var type = BinaryPrimitives.ReadUInt32LittleEndian(format);
        var channels = BinaryPrimitives.ReadUInt32LittleEndian(format[4..]);
        var sampleRate = BinaryPrimitives.ReadUInt32LittleEndian(format[8..]);
        if (channels is 0 or > 8 || sampleRate == 0)
        {
            return OrbisNgs2ErrorInvalidWaveformFormat;
        }

        ulong dataOffset;
        ulong dataSize;
        uint skipSamples = 0;
        switch (type)
        {
            case >= 0x10 and <= 0x1b:
            {
                var frameBytes = channels * (ulong)PcmSampleBytes(type);
                dataOffset = samplePosition * frameBytes;
                dataSize = sampleCount * frameBytes;
                break;
            }
            case WaveformTypeAtrac9:
            {
                var config = ReadAtrac9Config(format.Slice(12, 4));
                if (config is null) return OrbisNgs2ErrorInvalidWaveformFormat;

                if (config.ChannelCount != channels || config.SampleRate != sampleRate)
                {
                    return OrbisNgs2ErrorInvalidWaveformFormat;
                }

                var superframeSamples = (ulong)config.SuperframeSamples;
                var skip = samplePosition % superframeSamples;
                dataOffset = samplePosition / superframeSamples * (ulong)config.SuperframeBytes;
                dataSize = sampleCount == 0
                    ? 0
                    : (skip + sampleCount + superframeSamples - 1) / superframeSamples * (ulong)config.SuperframeBytes;
                skipSamples = sampleCount == 0 ? 0 : (uint)skip;
                break;
            }
            case 0x1c:
            {
                const ulong frameSamples = 28;
                var frameBytes = 16UL * channels;
                var skip = sampleCount == 0 ? 0 : samplePosition % frameSamples;
                dataOffset = samplePosition / frameSamples * frameBytes;
                dataSize = (skip + sampleCount + frameSamples - 1) / frameSamples * frameBytes;
                skipSamples = (uint)skip;
                break;
            }
            default:
                return Ngs2ErrorUnknownWaveformFormat;
        }

        // SceNgs2WaveformBlock { u64 dataOffset; u64 dataSize; u32 numRepeats,
        // numSkipSamples, numSamples, reserved; uintptr_t userData; }.
        BinaryPrimitives.WriteUInt64LittleEndian(block, dataOffset);
        BinaryPrimitives.WriteUInt64LittleEndian(block[8..], dataSize);
        BinaryPrimitives.WriteUInt32LittleEndian(block[20..], skipSamples);
        BinaryPrimitives.WriteUInt32LittleEndian(block[24..], sampleCount);
        return 0;
    }
}
