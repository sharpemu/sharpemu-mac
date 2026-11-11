// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Audio;

namespace SharpEmu.Libs.Audio3d;

/// <summary>
/// libSceAudio3d. A port collects one granularity-sized frame at a time from
/// bed writes (sceAudio3dBedWrite/BedWrite2) and object PCM attributes
/// (sceAudio3dObjectSetAttributes), mixes it on sceAudio3dPortAdvance and hands
/// it to the host audio output on sceAudio3dPortPush.
///
/// Positional rendering is out of scope: objects are folded to the stereo
/// output at the centre (equal power) and 5.1/7.1 beds are downmixed, because
/// the host streams are stereo. Object position/gain/spread attributes are
/// accepted and ignored. Queue level follows a wall-clock model: a pushed frame
/// occupies the queue for granularity/48000 s after the previous one ends.
///
/// ABI notes:
///   SceAudio3dOpenParameters (0x28; size_this 0x20 omits num_beds):
///     +0x00 u64 size_this, +0x08 u32 granularity, +0x0C u32 rate (0 = 48 kHz),
///     +0x10 u32 max_objects, +0x14 u32 queue_depth, +0x18 u32 buffer_mode,
///     +0x1C u32 pad, +0x20 u32 num_beds.
///   SceAudio3dAttribute (0x18): +0x00 u32 id, +0x08 void *value, +0x10 u64 size.
///   SceAudio3dPcm (object PCM attribute value): +0x00 u32 format (0 s16,
///     1 float), +0x08 void *samples, +0x10 u32 num_samples.
///   SceAudioOutOutputParam (AudioOutOutputs): +0x00 s32 handle, +0x08 void *ptr.
/// </summary>
public static class Audio3dExports
{
    internal const int Audio3dErrorInvalidPort = unchecked((int)0x80EA0002);
    internal const int Audio3dErrorInvalidObject = unchecked((int)0x80EA0003);
    internal const int Audio3dErrorInvalidParameter = unchecked((int)0x80EA0004);
    internal const int Audio3dErrorOutOfResources = unchecked((int)0x80EA0006);
    internal const int Audio3dErrorNotReady = unchecked((int)0x80EA0007);

    // The handle returned by sceAudio3dAudioOutOpen is an AudioOut port handle.
    internal const int AudioOutErrorInvalidPort = unchecked((int)0x80260003);

    internal const ulong OpenParametersSizeNoBeds = 0x20;
    internal const ulong OpenParametersSizeWithBeds = 0x28;
    internal const uint DefaultGranularity = 256;
    internal const uint DefaultMaxObjects = 512;
    internal const uint DefaultQueueDepth = 2;
    internal const uint DefaultBufferMode = BufferModeAdvanceAndPush;
    internal const uint DefaultNumBeds = 2;

    internal const uint BufferModeNoAdvance = 0;
    internal const uint BufferModeAdvanceNoPush = 1;
    internal const uint BufferModeAdvanceAndPush = 2;

    internal const uint SampleRate = 48000;
    internal const uint UserIdSystem = 0xFF;
    internal const uint ObjectInvalid = 0xFFFFFFFF;
    internal const uint ObjectAttributePcm = 1;
    internal const uint ObjectAttributeResetState = 0x20000;
    internal const uint PortAttributeLateReverbLevel = 0x10001;
    internal const uint PortAttributeDownmixSpreadRadius = 0x10002;
    internal const uint PortAttributeDownmixSpreadHeightAware = 0x10003;

    // Not measured on a console: a conservative port/handle pool.
    internal const int MaxPorts = 4;
    internal const int MaxAudioOutHandles = 8;
    private const uint MaxGranularity = 8192;
    private const ulong MaxAttributeCount = 0x10000;
    private const int AudioOutMaximumOutputCount = 25;

    private static readonly object Gate = new();
    private static readonly Port?[] Ports = new Port?[MaxPorts];
    private static readonly Dictionary<int, AudioOutHandle> AudioOuts = new();
    private static bool _initialized;
    private static int _nextAudioOutHandle;

    private static Func<long> _clock = Stopwatch.GetTimestamp;
    private static long _clockFrequency = Stopwatch.Frequency;
    private static Action<long> _waitUntil = DefaultWaitUntil;
    private static Func<uint, IHostAudioStream?>? _streamFactoryForTests;

    private sealed class Port
    {
        public required uint Granularity;
        public required uint MaxObjects;
        public required uint QueueDepth;
        public required uint BufferMode;
        public required uint NumBeds;
        public required long FrameTicks;
        public IHostAudioStream? Stream;
        public float[] BedAccumulator = [];
        public readonly Dictionary<uint, float[]> ObjectPcm = new();
        public readonly HashSet<uint> Objects = new();
        public uint NextObjectId;
        public readonly List<byte[]> Advanced = new();
        public readonly List<long> Playing = new();
        public float LateReverbLevel;
        public float DownmixSpreadRadius = 2.0f;
        public int DownmixSpreadHeightAware;
    }

    private sealed class AudioOutHandle
    {
        public required uint Length;
        public required int Channels;
        public required int BytesPerSample;
        public required bool IsFloat;
        public IHostAudioStream? Stream;
        public int BufferBytes => checked((int)Length * Channels * BytesPerSample);
    }

    [SysAbiExport(
        Nid = "Im+jOoa5WAI",
        ExportName = "sceAudio3dGetDefaultOpenParameters",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dGetDefaultOpenParameters(CpuContext ctx)
    {
        var address = ctx[CpuRegister.Rdi];
        if (address == 0)
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        // Only the 0x20-byte base is written (size_this = 0x20, num_beds is left
        // to the caller), so a PortOpen with these defaults uses DefaultNumBeds.
        Span<byte> data = stackalloc byte[(int)OpenParametersSizeNoBeds];
        data.Clear();
        BinaryPrimitives.WriteUInt64LittleEndian(data, OpenParametersSizeNoBeds);
        BinaryPrimitives.WriteUInt32LittleEndian(data[0x08..], DefaultGranularity);
        BinaryPrimitives.WriteUInt32LittleEndian(data[0x0C..], 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data[0x10..], DefaultMaxObjects);
        BinaryPrimitives.WriteUInt32LittleEndian(data[0x14..], DefaultQueueDepth);
        BinaryPrimitives.WriteUInt32LittleEndian(data[0x18..], DefaultBufferMode);
        return ctx.SetReturn(ctx.Memory.TryWrite(address, data)
            ? 0
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    [SysAbiExport(
        Nid = "UmCvjSmuZIw",
        ExportName = "sceAudio3dInitialize",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dInitialize(CpuContext ctx)
    {
        if (ctx[CpuRegister.Rdi] != 0)
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        lock (Gate)
        {
            // A repeated initialization is accepted (idempotent).
            _initialized = true;
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "WW1TS2iz5yc",
        ExportName = "sceAudio3dTerminate",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dTerminate(CpuContext ctx)
    {
        lock (Gate)
        {
            if (!_initialized || Ports.Any(static p => p is not null))
            {
                return ctx.SetReturn(Audio3dErrorNotReady);
            }

            _initialized = false;
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "XeDDK0xJWQA",
        ExportName = "sceAudio3dPortOpen",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortOpen(CpuContext ctx)
    {
        var userId = unchecked((int)ctx[CpuRegister.Rdi]);
        var parametersAddress = ctx[CpuRegister.Rsi];
        var idAddress = ctx[CpuRegister.Rdx];
        lock (Gate)
        {
            if (!_initialized)
            {
                return ctx.SetReturn(Audio3dErrorNotReady);
            }
        }

        if (userId < 0 || parametersAddress == 0 || idAddress == 0 ||
            !ctx.TryReadUInt64(parametersAddress, out var sizeThis) ||
            (sizeThis != OpenParametersSizeNoBeds && sizeThis != OpenParametersSizeWithBeds))
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        Span<byte> raw = stackalloc byte[(int)OpenParametersSizeWithBeds];
        raw.Clear();
        if (!ctx.Memory.TryRead(parametersAddress, raw[..(int)sizeThis]))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var granularity = BinaryPrimitives.ReadUInt32LittleEndian(raw[0x08..]);
        var rate = BinaryPrimitives.ReadUInt32LittleEndian(raw[0x0C..]);
        var maxObjects = BinaryPrimitives.ReadUInt32LittleEndian(raw[0x10..]);
        var queueDepth = BinaryPrimitives.ReadUInt32LittleEndian(raw[0x14..]);
        var bufferMode = BinaryPrimitives.ReadUInt32LittleEndian(raw[0x18..]);
        var numBeds = sizeThis == OpenParametersSizeWithBeds
            ? BinaryPrimitives.ReadUInt32LittleEndian(raw[0x20..])
            : DefaultNumBeds;
        if (rate != 0 ||
            granularity < 0x100 || (granularity & 0xFF) != 0 || granularity > MaxGranularity ||
            maxObjects == 0 || queueDepth == 0 ||
            bufferMode > BufferModeAdvanceAndPush ||
            numBeds is not (2 or 3))
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        var stream = OpenHostStream();
        uint id;
        lock (Gate)
        {
            id = uint.MaxValue;
            for (uint i = 0; i < MaxPorts; i++)
            {
                if (Ports[i] is null)
                {
                    id = i;
                    break;
                }
            }

            if (id == uint.MaxValue)
            {
                stream?.Dispose();
                return ctx.SetReturn(Audio3dErrorOutOfResources);
            }

            Ports[id] = new Port
            {
                Granularity = granularity,
                MaxObjects = maxObjects,
                QueueDepth = queueDepth,
                BufferMode = bufferMode,
                NumBeds = numBeds,
                FrameTicks = (long)granularity * _clockFrequency / SampleRate,
                Stream = stream,
                BedAccumulator = new float[granularity * 2],
            };
        }

        if (!ctx.TryWriteUInt32(idAddress, id))
        {
            ClosePort(id);
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "OyVqOeVNtSk",
        ExportName = "sceAudio3dPortClose",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortClose(CpuContext ctx) =>
        ctx.SetReturn(ClosePort(unchecked((uint)ctx[CpuRegister.Rdi])));

    private static int ClosePort(uint portId)
    {
        IHostAudioStream? stream;
        lock (Gate)
        {
            if (portId >= MaxPorts || Ports[portId] is not { } port)
            {
                return Audio3dErrorInvalidPort;
            }

            stream = port.Stream;
            Ports[portId] = null;
        }

        stream?.Dispose();
        return 0;
    }

    [SysAbiExport(
        Nid = "ZOGrxWLgQzE",
        ExportName = "sceAudio3dPortFlush",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortFlush(CpuContext ctx)
    {
        lock (Gate)
        {
            if (!TryGetPort(unchecked((uint)ctx[CpuRegister.Rdi]), out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            // Drops queued (advanced) and still-playing frames; the frame being
            // assembled is untouched.
            port.Advanced.Clear();
            port.Playing.Clear();
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "lw0qrdSjZt8",
        ExportName = "sceAudio3dPortAdvance",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortAdvance(CpuContext ctx)
    {
        List<byte[]> toSubmit = [];
        IHostAudioStream? stream;
        lock (Gate)
        {
            if (!TryGetPort(unchecked((uint)ctx[CpuRegister.Rdi]), out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            PruneLocked(port, _clock());
            if (port.Advanced.Count + port.Playing.Count >= port.QueueDepth)
            {
                return ctx.SetReturn(Audio3dErrorNotReady);
            }

            AdvanceLocked(port);
            if (port.BufferMode == BufferModeAdvanceNoPush)
            {
                // Advance-only mode: the advanced frame is pushed implicitly.
                PushLocked(port, blocking: false, toSubmit, out _);
            }

            stream = port.Stream;
        }

        Submit(stream, toSubmit);
        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "VEVhZ9qd4ZY",
        ExportName = "sceAudio3dPortPush",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortPush(CpuContext ctx)
    {
        var blocking = unchecked((uint)ctx[CpuRegister.Rsi]);
        List<byte[]> toSubmit = [];
        IHostAudioStream? stream;
        long waitUntil;
        lock (Gate)
        {
            if (!TryGetPort(unchecked((uint)ctx[CpuRegister.Rdi]), out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            if (blocking > 1)
            {
                return ctx.SetReturn(Audio3dErrorInvalidParameter);
            }

            if (port.BufferMode == BufferModeNoAdvance)
            {
                // No-advance mode: push advances the frame being assembled itself.
                AdvanceLocked(port);
            }

            PushLocked(port, blocking == 1, toSubmit, out waitUntil);
            stream = port.Stream;
        }

        Submit(stream, toSubmit);
        if (waitUntil != 0)
        {
            _waitUntil(waitUntil);
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "YaaDbDwKpFM",
        ExportName = "sceAudio3dPortGetQueueLevel",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortGetQueueLevel(CpuContext ctx)
    {
        var levelAddress = ctx[CpuRegister.Rsi];
        var availableAddress = ctx[CpuRegister.Rdx];
        uint level;
        uint available;
        lock (Gate)
        {
            if (!TryGetPort(unchecked((uint)ctx[CpuRegister.Rdi]), out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            if (levelAddress == 0 && availableAddress == 0)
            {
                return ctx.SetReturn(Audio3dErrorInvalidParameter);
            }

            PruneLocked(port, _clock());
            level = (uint)(port.Advanced.Count + port.Playing.Count);
            available = level >= port.QueueDepth ? 0 : port.QueueDepth - level;
        }

        if ((levelAddress != 0 && !ctx.TryWriteUInt32(levelAddress, level)) ||
            (availableAddress != 0 && !ctx.TryWriteUInt32(availableAddress, available)))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "Yq9bfUQ0uJg",
        ExportName = "sceAudio3dPortSetAttribute",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortSetAttribute(CpuContext ctx)
    {
        var portId = unchecked((uint)ctx[CpuRegister.Rdi]);
        var attributeId = unchecked((uint)ctx[CpuRegister.Rsi]);
        var valueAddress = ctx[CpuRegister.Rdx];
        var valueSize = ctx[CpuRegister.Rcx];
        lock (Gate)
        {
            if (!TryGetPort(portId, out _))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }
        }

        if (valueAddress == 0 || valueSize != 4 ||
            attributeId is not (PortAttributeLateReverbLevel or
                PortAttributeDownmixSpreadRadius or
                PortAttributeDownmixSpreadHeightAware))
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        if (!ctx.TryReadUInt32(valueAddress, out var raw))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        lock (Gate)
        {
            if (!TryGetPort(portId, out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            var value = BitConverter.UInt32BitsToSingle(raw);
            switch (attributeId)
            {
                case PortAttributeLateReverbLevel:
                    port.LateReverbLevel = value;
                    break;
                case PortAttributeDownmixSpreadRadius:
                    port.DownmixSpreadRadius = value;
                    break;
                default:
                    port.DownmixSpreadHeightAware = unchecked((int)raw);
                    break;
            }
        }

        return ctx.SetReturn(0);
    }

    // Unverified signature (SDK header not available): (port, u32 *ids, u32 *count),
    // count being the capacity on entry and the number of supported port
    // attributes on return.
    [SysAbiExport(
        Nid = "9ZA23Ia46Po",
        ExportName = "sceAudio3dPortGetAttributesSupported",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dPortGetAttributesSupported(CpuContext ctx)
    {
        var idsAddress = ctx[CpuRegister.Rsi];
        var countAddress = ctx[CpuRegister.Rdx];
        lock (Gate)
        {
            if (!TryGetPort(unchecked((uint)ctx[CpuRegister.Rdi]), out _))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }
        }

        uint[] supported =
        [
            PortAttributeLateReverbLevel,
            PortAttributeDownmixSpreadRadius,
            PortAttributeDownmixSpreadHeightAware,
        ];
        if (countAddress == 0 || !ctx.TryReadUInt32(countAddress, out var capacity))
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        if (idsAddress != 0)
        {
            var written = (int)Math.Min(capacity, (uint)supported.Length);
            for (var i = 0; i < written; i++)
            {
                if (!ctx.TryWriteUInt32(idsAddress + (ulong)(i * 4), supported[i]))
                {
                    return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
                }
            }
        }

        return ctx.SetReturn(ctx.TryWriteUInt32(countAddress, (uint)supported.Length)
            ? 0
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    [SysAbiExport(
        Nid = "jO2tec4dJ2M",
        ExportName = "sceAudio3dObjectReserve",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dObjectReserve(CpuContext ctx)
    {
        var objectAddress = ctx[CpuRegister.Rsi];
        if (objectAddress == 0)
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        // A failed reserve leaves the invalid-object marker in the output.
        if (!ctx.TryWriteUInt32(objectAddress, ObjectInvalid))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        uint id;
        lock (Gate)
        {
            if (!TryGetPort(unchecked((uint)ctx[CpuRegister.Rdi]), out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            if (port.Objects.Count >= port.MaxObjects)
            {
                return ctx.SetReturn(Audio3dErrorOutOfResources);
            }

            // Ids count up from 1, skipping 0, the invalid marker and ids in use.
            do
            {
                port.NextObjectId++;
            }
            while (port.NextObjectId is 0 or ObjectInvalid || port.Objects.Contains(port.NextObjectId));

            id = port.NextObjectId;
            port.Objects.Add(id);
        }

        if (!ctx.TryWriteUInt32(objectAddress, id))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "1HXxo-+1qCw",
        ExportName = "sceAudio3dObjectUnreserve",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dObjectUnreserve(CpuContext ctx)
    {
        var objectId = unchecked((uint)ctx[CpuRegister.Rsi]);
        lock (Gate)
        {
            if (!TryGetPort(unchecked((uint)ctx[CpuRegister.Rdi]), out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            if (!port.Objects.Remove(objectId))
            {
                return ctx.SetReturn(Audio3dErrorInvalidObject);
            }

            port.ObjectPcm.Remove(objectId);
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "4uyHN9q4ZeU",
        ExportName = "sceAudio3dObjectSetAttributes",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dObjectSetAttributes(CpuContext ctx)
    {
        var portId = unchecked((uint)ctx[CpuRegister.Rdi]);
        var objectId = unchecked((uint)ctx[CpuRegister.Rsi]);
        var count = ctx[CpuRegister.Rdx];
        var attributesAddress = ctx[CpuRegister.Rcx];
        uint granularity;
        lock (Gate)
        {
            if (!TryGetPort(portId, out var port))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }

            if (count == 0 || count > MaxAttributeCount || attributesAddress == 0)
            {
                return ctx.SetReturn(Audio3dErrorInvalidParameter);
            }

            if (!port.Objects.Contains(objectId))
            {
                return ctx.SetReturn(Audio3dErrorInvalidObject);
            }

            granularity = port.Granularity;
        }

        // Parse everything first so a bad attribute leaves the object unchanged.
        float[]? pcm = null;
        Span<byte> attribute = stackalloc byte[0x18];
        for (ulong i = 0; i < count; i++)
        {
            if (!ctx.Memory.TryRead(attributesAddress + (i * 0x18), attribute))
            {
                return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }

            var id = BinaryPrimitives.ReadUInt32LittleEndian(attribute);
            var value = BinaryPrimitives.ReadUInt64LittleEndian(attribute[0x08..]);
            if (id == ObjectAttributeResetState)
            {
                continue;
            }

            if (value == 0)
            {
                return ctx.SetReturn(Audio3dErrorInvalidParameter);
            }

            if (id != ObjectAttributePcm)
            {
                // Position, gain, spread, priority, ...: accepted, not rendered.
                continue;
            }

            var status = TryReadObjectPcm(ctx, value, granularity, out pcm);
            if (status != 0)
            {
                return ctx.SetReturn(status);
            }
        }

        if (pcm is not null)
        {
            lock (Gate)
            {
                if (!TryGetPort(portId, out var port))
                {
                    return ctx.SetReturn(Audio3dErrorInvalidPort);
                }

                if (!port.Objects.Contains(objectId))
                {
                    return ctx.SetReturn(Audio3dErrorInvalidObject);
                }

                port.ObjectPcm[objectId] = pcm;
            }
        }

        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "9tEwE0GV0qo",
        ExportName = "sceAudio3dBedWrite",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dBedWrite(CpuContext ctx) =>
        ctx.SetReturn(BedWrite(ctx));

    // BedWrite2 adds (output route, restricted) after the BedWrite arguments;
    // both only select hardware routing, which the host mix does not model.
    [SysAbiExport(
        Nid = "xH4Q9UILL3o",
        ExportName = "sceAudio3dBedWrite2",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dBedWrite2(CpuContext ctx) =>
        ctx.SetReturn(BedWrite(ctx));

    private static int BedWrite(CpuContext ctx)
    {
        var portId = unchecked((uint)ctx[CpuRegister.Rdi]);
        var channels = unchecked((uint)ctx[CpuRegister.Rsi]);
        var format = unchecked((uint)ctx[CpuRegister.Rdx]);
        var bufferAddress = ctx[CpuRegister.Rcx];
        var samples = unchecked((uint)ctx[CpuRegister.R8]);
        uint granularity;
        lock (Gate)
        {
            if (!TryGetPort(portId, out var port))
            {
                return Audio3dErrorInvalidPort;
            }

            granularity = port.Granularity;
        }

        if (channels is not (2 or 6 or 8) || format > 1 || bufferAddress == 0 ||
            samples == 0 || samples > granularity)
        {
            return Audio3dErrorInvalidParameter;
        }

        var isFloat = format == 1;
        var bytes = checked((int)samples * (int)channels * (isFloat ? 4 : 2));
        var buffer = new byte[bytes];
        if (!ctx.Memory.TryRead(bufferAddress, buffer))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        lock (Gate)
        {
            if (!TryGetPort(portId, out var port))
            {
                return Audio3dErrorInvalidPort;
            }

            Audio3dMixer.AccumulateBed(
                port.BedAccumulator, buffer, (int)channels, isFloat, (int)samples);
        }

        return 0;
    }

    // sceAudio3dAudioOutOpen(port, userId, type, index, len, freq, param) -> handle.
    // The handle is an AudioOut-style port fed directly by AudioOutOutput(s).
    [SysAbiExport(
        Nid = "ucEsi62soTo",
        ExportName = "sceAudio3dAudioOutOpen",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dAudioOutOpen(CpuContext ctx)
    {
        var portId = unchecked((uint)ctx[CpuRegister.Rdi]);
        var length = unchecked((uint)ctx[CpuRegister.R8]);
        var frequency = unchecked((uint)ctx[CpuRegister.R9]);
        _ = ctx.TryGetImportStackArgument(0, out var rawFormat);
        lock (Gate)
        {
            if (!TryGetPort(portId, out _))
            {
                return ctx.SetReturn(Audio3dErrorInvalidPort);
            }
        }

        if (length == 0 || length > MaxGranularity || frequency != SampleRate ||
            !TryGetFormat(unchecked((int)rawFormat), out var channels, out var bytesPerSample))
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        var stream = OpenHostStream();
        int handle;
        lock (Gate)
        {
            if (AudioOuts.Count >= MaxAudioOutHandles)
            {
                stream?.Dispose();
                return ctx.SetReturn(Audio3dErrorOutOfResources);
            }

            handle = ++_nextAudioOutHandle;
            AudioOuts[handle] = new AudioOutHandle
            {
                Length = length,
                Channels = channels,
                BytesPerSample = bytesPerSample,
                IsFloat = bytesPerSample == 4,
                Stream = stream,
            };
        }

        return ctx.SetReturn(handle);
    }

    [SysAbiExport(
        Nid = "pZlOm1aF3aA",
        ExportName = "sceAudio3dAudioOutClose",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dAudioOutClose(CpuContext ctx)
    {
        var handle = unchecked((int)ctx[CpuRegister.Rdi]);
        AudioOutHandle? removed;
        lock (Gate)
        {
            if (!AudioOuts.Remove(handle, out removed))
            {
                return ctx.SetReturn(AudioOutErrorInvalidPort);
            }
        }

        removed.Stream?.Dispose();
        return ctx.SetReturn(0);
    }

    [SysAbiExport(
        Nid = "7NYEzJ9SJbM",
        ExportName = "sceAudio3dAudioOutOutput",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dAudioOutOutput(CpuContext ctx) =>
        ctx.SetReturn(OutputOne(
            ctx, unchecked((int)ctx[CpuRegister.Rdi]), ctx[CpuRegister.Rsi]));

    [SysAbiExport(
        Nid = "HbxYY27lK6E",
        ExportName = "sceAudio3dAudioOutOutputs",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAudio3d")]
    public static int Audio3dAudioOutOutputs(CpuContext ctx)
    {
        var paramsAddress = ctx[CpuRegister.Rdi];
        var count = unchecked((int)ctx[CpuRegister.Rsi]);
        if (paramsAddress == 0 || count < 1 || count > AudioOutMaximumOutputCount)
        {
            return ctx.SetReturn(Audio3dErrorInvalidParameter);
        }

        var result = 0;
        for (var i = 0; i < count; i++)
        {
            var entry = paramsAddress + (ulong)(i * 16);
            if (!ctx.TryReadInt32(entry, out var handle) ||
                !ctx.TryReadUInt64(entry + 8, out var pointer))
            {
                return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }

            var status = OutputOne(ctx, handle, pointer);
            if (status < 0)
            {
                return ctx.SetReturn(status);
            }

            result = Math.Max(result, status);
        }

        return ctx.SetReturn(result);
    }

    /// <summary>Submits one AudioOut buffer; returns the sample count or an error.</summary>
    private static int OutputOne(CpuContext ctx, int handle, ulong sourceAddress)
    {
        AudioOutHandle? output;
        lock (Gate)
        {
            if (!AudioOuts.TryGetValue(handle, out output))
            {
                return AudioOutErrorInvalidPort;
            }
        }

        var frames = (int)output.Length;
        var stream = output.Stream;
        if (sourceAddress == 0 || stream is null)
        {
            if (sourceAddress != 0)
            {
                var probe = new byte[output.BufferBytes];
                if (!ctx.Memory.TryRead(sourceAddress, probe))
                {
                    return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
                }
            }

            // No samples or no host device: pace the guest as a real device would.
            _waitUntil(_clock() + ((long)frames * _clockFrequency / SampleRate));
            return frames;
        }

        var source = new byte[output.BufferBytes];
        if (!ctx.Memory.TryRead(sourceAddress, source))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        var converted = new byte[frames * AudioPcmConversion.OutputFrameSize];
        AudioPcmConversion.ConvertToStereoPcm16(
            source, converted, frames, output.Channels, output.BytesPerSample, output.IsFloat, 1f);
        if (!stream.Submit(converted))
        {
            _waitUntil(_clock() + ((long)frames * _clockFrequency / SampleRate));
        }

        return frames;
    }

    private static int TryReadObjectPcm(
        CpuContext ctx, ulong address, uint granularity, out float[]? pcm)
    {
        pcm = null;
        Span<byte> header = stackalloc byte[0x18];
        if (!ctx.Memory.TryRead(address, header))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        var format = BinaryPrimitives.ReadUInt32LittleEndian(header);
        var samplesAddress = BinaryPrimitives.ReadUInt64LittleEndian(header[0x08..]);
        var samples = BinaryPrimitives.ReadUInt32LittleEndian(header[0x10..]);
        if (format > 1 || samplesAddress == 0 || samples == 0 || samples > granularity)
        {
            return Audio3dErrorInvalidParameter;
        }

        var isFloat = format == 1;
        var raw = new byte[checked((int)samples * (isFloat ? 4 : 2))];
        if (!ctx.Memory.TryRead(samplesAddress, raw))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        var mono = new float[granularity];
        Audio3dMixer.ReadMono(raw, isFloat, (int)samples, mono);
        pcm = mono;
        return 0;
    }

    private static bool TryGetPort(uint portId, out Port port)
    {
        if (portId < MaxPorts && Ports[portId] is { } found)
        {
            port = found;
            return true;
        }

        port = null!;
        return false;
    }

    /// <summary>Mixes the frame being assembled and queues it for the next push.</summary>
    private static void AdvanceLocked(Port port)
    {
        var accumulator = port.BedAccumulator;
        foreach (var pcm in port.ObjectPcm.Values)
        {
            Audio3dMixer.AccumulateObject(accumulator, pcm);
        }

        var frame = new byte[accumulator.Length * 2];
        Audio3dMixer.ConvertToPcm16(accumulator, frame);
        port.Advanced.Add(frame);
        Array.Clear(accumulator);
        port.ObjectPcm.Clear();
    }

    private static void PushLocked(
        Port port, bool blocking, List<byte[]> toSubmit, out long waitUntil)
    {
        waitUntil = 0;
        var now = _clock();
        PruneLocked(port, now);
        var end = port.Playing.Count > 0 ? Math.Max(now, port.Playing[^1]) : now;
        foreach (var frame in port.Advanced)
        {
            end += port.FrameTicks;
            port.Playing.Add(end);
            toSubmit.Add(frame);
        }

        port.Advanced.Clear();
        if (blocking && port.Playing.Count >= port.QueueDepth)
        {
            // Block until the queue has room for another advanced frame.
            waitUntil = port.Playing[port.Playing.Count - (int)port.QueueDepth];
        }
    }

    private static void PruneLocked(Port port, long now)
    {
        var expired = 0;
        while (expired < port.Playing.Count && port.Playing[expired] <= now)
        {
            expired++;
        }

        if (expired > 0)
        {
            port.Playing.RemoveRange(0, expired);
        }
    }

    private static void Submit(IHostAudioStream? stream, List<byte[]> frames)
    {
        if (stream is null)
        {
            return;
        }

        foreach (var frame in frames)
        {
            // A refused frame is simply not heard; pacing comes from the queue model.
            _ = stream.Submit(frame);
        }
    }

    private static IHostAudioStream? OpenHostStream()
    {
        try
        {
            var factory = Volatile.Read(ref _streamFactoryForTests);
            return factory is not null
                ? factory(SampleRate)
                : HostPlatform.Current.Audio.OpenStereoPcm16Stream(SampleRate);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(
                $"[LOADER][WARN] Audio3d host backend unavailable: {exception.Message}");
            return null;
        }
    }

    // Same AudioOut format numbering as sceAudioOutOpen.
    private static bool TryGetFormat(int rawFormat, out int channels, out int bytesPerSample)
    {
        var format = rawFormat & 0xFF;
        channels = format switch
        {
            0 or 3 => 1,
            1 or 4 => 2,
            2 or 5 or 6 or 7 => 8,
            _ => 0,
        };
        bytesPerSample = format is >= 3 and <= 5 or 7 ? 4 : 2;
        return channels != 0;
    }

    private static void DefaultWaitUntil(long targetTicks)
    {
        var start = Stopwatch.GetTimestamp();
        while (true)
        {
            var now = Stopwatch.GetTimestamp();
            var remaining = targetTicks - now;
            if (remaining <= 0)
            {
                HostBlockedTime.Add(now - start);
                return;
            }

            var milliseconds = remaining * 1000 / Stopwatch.Frequency;
            if (milliseconds > 1)
            {
                Thread.Sleep((int)(milliseconds - 1));
            }
            else
            {
                Thread.Yield();
            }
        }
    }

    internal static void ResetForTests()
    {
        lock (Gate)
        {
            foreach (var port in Ports)
            {
                port?.Stream?.Dispose();
            }

            Array.Clear(Ports);
            foreach (var output in AudioOuts.Values)
            {
                output.Stream?.Dispose();
            }

            AudioOuts.Clear();
            _initialized = false;
            _nextAudioOutHandle = 0;
            _clock = Stopwatch.GetTimestamp;
            _clockFrequency = Stopwatch.Frequency;
            _waitUntil = DefaultWaitUntil;
        }

        Volatile.Write(ref _streamFactoryForTests, null);
    }

    internal static void SetStreamFactoryForTests(Func<uint, IHostAudioStream?>? factory) =>
        Volatile.Write(ref _streamFactoryForTests, factory);

    internal static void SetClockForTests(Func<long> clock, long frequency, Action<long> waitUntil)
    {
        lock (Gate)
        {
            _clock = clock;
            _clockFrequency = frequency;
            _waitUntil = waitUntil;
        }
    }
}
