// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Images;

public sealed class MipStatistics
{
    public const int SlotCount = 256;
    public const uint FirstCounterOffset = 64;
    public const int ReportBytes = (int)FirstCounterOffset + SlotCount * CounterBytes;
    public const uint NoMipSampled = 15;
    public const uint CounterWords = 257;

    /// <summary>
    /// The counter bank a shader records into: three arrays of <see cref="CounterWords"/>
    /// words - the finest level each counter id saw, its clamped samples, and its unclamped
    /// samples of the tracked mip (<see cref="PackReport"/>).
    /// </summary>
    public const ulong CounterBufferBytes = 3 * CounterWords * sizeof(uint);
    public const uint NoCounterField = 0x8000;
    public const uint ReportWithoutCounters = 1;

    private const int CounterBytes = 2 * sizeof(uint);
    private const uint FinestMip = 0;
    private const uint SamplesPerBinding = 0x10000;
    private const uint MaxSamples = 0xFFFFFF;

    public static MipStatistics Shared { get; } = new();

    private static int s_titleUsesCounters;

    /// <summary>
    /// The title has used LOD statistics: it bound a T# with MIP_STATS_ENABLE to a draw, or it
    /// sent a GET_LOD_STATS packet. Until then nothing reads the counter bank, so a host that
    /// can may leave the counters out of its pixel pipelines. Once set it stays set.
    /// </summary>
    public static bool TitleUsesCounters => Volatile.Read(ref s_titleUsesCounters) != 0;

    /// <summary>Latches <see cref="TitleUsesCounters"/>; true when this call set it.</summary>
    public static bool NoteTitleUsesCounters() =>
        Volatile.Read(ref s_titleUsesCounters) == 0 && Interlocked.Exchange(ref s_titleUsesCounters, 1) == 0;

    private readonly uint[] _bindingsBySlot = new uint[SlotCount];

    /// <summary>Slots a draw has bound since the last report (diagnostic; racy by design).</summary>
    public int BoundSlotCount => _bindingsBySlot.Count(static bindings => bindings != 0);

    /// <summary>A report the host measured on the GPU starts a new window here too.</summary>
    public void ClearBoundSlots() => Array.Clear(_bindingsBySlot);

    public void NoteSampled(in TextureDescriptorWords texture)
    {
        if (texture.MipStatsEnabled && !texture.IsNull)
        {
            NoteTitleUsesCounters();
            LodStatsTelemetry.NoteSampledBinding();
            _bindingsBySlot[texture.MipStatsSlot]++;
        }
    }

    public void NoteSampled(in TextureDescriptorWords texture, uint statistics)
    {
        if (statistics == ResourceSnapshot.NoMipStatistics)
        {
            NoteSampled(texture);
        }
        else if ((statistics & ResourceSnapshot.MipStatisticsEnabled) != 0 && !texture.IsNull)
        {
            _bindingsBySlot[statistics & ResourceSnapshot.MipStatisticsSlotWordMask]++;
        }
    }

    public static uint ImageField(ReadOnlySpan<uint> texture, uint statistics)
    {
        if (statistics == ResourceSnapshot.NoMipStatistics)
        {
            return ImageField(texture);
        }

        if ((statistics & ResourceSnapshot.MipStatisticsEnabled) == 0)
        {
            return NoCounterField;
        }

        var baseLevel = (texture[3] >> 12) & 0xF;
        var minLod = (texture[1] >> 8) & 0xFFF;
        return (statistics & ResourceSnapshot.MipStatisticsSlotWordMask) | (baseLevel << 8) | (minLod << 16);
    }

    public static uint ImageField(ReadOnlySpan<uint> texture)
    {
        if (((texture[5] >> 25) & 1) == 0)
        {
            return NoCounterField;
        }

        var counter = texture[6] & 0xFF;
        var baseLevel = (texture[3] >> 12) & 0xF;
        var minLod = (texture[1] >> 8) & 0xFFF;
        return counter | (baseLevel << 8) | (minLod << 16);
    }

    /// <summary>
    /// The shader-data fields of a program that records mip statistics, one per image
    /// (<see cref="BindingLayout.MipStatisticsDword"/>): <see cref="ImageField"/> of the bound
    /// T# when the program records that image, otherwise <see cref="NoCounterField"/>.
    /// </summary>
    public static void WriteImageFields(Span<uint> shaderData, BindingLayout layout, ShaderResourceInfo info, uint[][] images)
    {
        for (var index = 0; index < layout.MipStatisticsCount; index++)
        {
            var image = index < info.Images.Count ? info.Images[index] : null;
            var words = index < images.Length ? images[index] : null;
            shaderData[(int)(layout.MipStatisticsDword + (uint)index)] = image is { MipStatistics: true } && words is { Length: >= 8 }
                ? ImageField(words)
                : NoCounterField;
        }
    }

    /// <summary>
    /// <see cref="WriteImageFields(Span{uint}, BindingLayout, ShaderResourceInfo, uint[][])"/> with the
    /// statistics the snapshot read for each image (<see cref="ResourceSnapshot.NoMipStatistics"/> when none).
    /// </summary>
    public static void WriteImageFields(Span<uint> shaderData, BindingLayout layout, ShaderResourceInfo info, uint[][] images, uint[]? statistics)
    {
        for (var index = 0; index < layout.MipStatisticsCount; index++)
        {
            var image = index < info.Images.Count ? info.Images[index] : null;
            var words = index < images.Length ? images[index] : null;
            var current = statistics is not null && index < statistics.Length ? statistics[index] : ResourceSnapshot.NoMipStatistics;
            shaderData[(int)(layout.MipStatisticsDword + (uint)index)] = image is { MipStatistics: true } && words is { Length: >= 8 }
                ? ImageField(words, current)
                : NoCounterField;
        }
    }

    /// <summary>
    /// The guest MipStatsBuffer from the counter bank: the header (the byte offset of the
    /// first slot, which is also the non-zero completion label) and per slot
    /// {clamped:24, slot:8} {unclamped:24, finestMip:4}. A bank of two arrays has no
    /// unclamped counts and reports none.
    /// </summary>
    public static void PackReport(ReadOnlySpan<uint> words, Span<byte> report)
    {
        report.Clear();
        if (report.Length >= sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(report, FirstCounterOffset);
        }

        for (var slot = 0; slot < SlotCount; slot++)
        {
            var start = (int)FirstCounterOffset + slot * CounterBytes;
            if (start + CounterBytes > report.Length)
            {
                break;
            }

            var finest = words[slot];
            var count = Math.Min(words[(int)CounterWords + slot], MaxSamples);
            var unclamped = words.Length >= 3 * (int)CounterWords
                ? Math.Min(words[2 * (int)CounterWords + slot], MaxSamples)
                : 0u;
            var level = finest == uint.MaxValue ? NoMipSampled : Math.Min(finest, 14u);
            var counter = report.Slice(start, CounterBytes);
            BinaryPrimitives.WriteUInt32LittleEndian(counter, count | ((uint)slot << 24));
            BinaryPrimitives.WriteUInt32LittleEndian(counter[sizeof(uint)..], unclamped | (level << 24));
        }
    }

    public void WriteReport(Span<byte> report)
    {
        var hasCounters = TryWriteReport(report);
        if (report.Length >= sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(report, hasCounters ? FirstCounterOffset : ReportWithoutCounters);
        }
    }

    public bool TryWriteReport(Span<byte> report)
    {
        if (report.Length < ReportBytes)
        {
            return false;
        }

        for (var slot = 0; slot < SlotCount; slot++)
        {
            var samples = (uint)Math.Min((ulong)_bindingsBySlot[slot] * SamplesPerBinding, MaxSamples);
            var finestMipSampled = samples == 0 ? NoMipSampled : FinestMip;
            var clampedSamplesAndSlot = samples | ((uint)slot << 24);
            var unclampedSamplesAndFinestMip = samples | (finestMipSampled << 24);
            var counter = report.Slice((int)FirstCounterOffset + slot * CounterBytes, CounterBytes);
            BinaryPrimitives.WriteUInt32LittleEndian(counter, clampedSamplesAndSlot);
            BinaryPrimitives.WriteUInt32LittleEndian(counter[sizeof(uint)..], unclampedSamplesAndFinestMip);
        }

        Array.Clear(_bindingsBySlot);
        return true;
    }
}
