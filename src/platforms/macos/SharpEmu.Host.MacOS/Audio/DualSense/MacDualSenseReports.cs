// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// The DualSense output reports the controller audio sends, byte for byte as upstream's SDL host builds them
/// (SdlHostAudio.ControllerAudioRoute, SDL_SendGamepadEffect, SdlDualSenseBluetooth.BuildReport). Pure: unit tested.
/// </summary>
internal static class MacDualSenseReports
{
    /// <summary>The effects state SDL_SendGamepadEffect takes for a PS5 pad (DS5EffectsState_t).</summary>
    internal const int EffectsLength = 47;

    /// <summary>The route effects upstream sends: the first 38 bytes of the effects state.</summary>
    internal const int RouteLength = 38;

    internal const byte UsbEffectsReportId = 0x02;
    internal const int UsbEffectsReportLength = 48;
    internal const byte BluetoothEffectsReportId = 0x31;
    internal const int BluetoothEffectsReportLength = 78;

    internal const byte BluetoothAudioReportId = 0x39;
    internal const int BluetoothAudioReportLength = 547;
    internal const int HapticFrames = 64;
    internal const int HapticOffset = 12;
    internal const byte HapticBlockBytes = 64;
    internal const int SpeakerHeaderOffset = 140;
    internal const int SpeakerOffset = 142;
    internal const int OpusPacketBytes = 200;
    internal const int OpusPackets = 2;
    private const byte AudioBufferLength = 48;

    /// <summary>
    /// The pad's audio routing: with the speaker, the right channel of the front pair plays on the speaker and the
    /// headphone jack is muted (audio control and speaker volume valid, speaker volume 0x64, output path 0x30, speaker
    /// preamp +6 dB: the values Linux's driver uses without headphones); without it, the front pair goes back to the
    /// headphone jack. Plugged-in headphones are not detected.
    /// </summary>
    internal static byte[] AudioRoute(bool speaker)
    {
        var effects = new byte[RouteLength];
        effects[0] = 0x80;
        if (speaker)
        {
            effects[0] |= 0x20;
            effects[1] = 0x80;
            effects[5] = 0x64;
            effects[7] = 0x30;
            effects[37] = 0x02;
        }

        return effects;
    }

    /// <summary>USB output report 0x02: the report ID, then the effects state. Returns the report length.</summary>
    internal static int BuildUsbEffectsReport(ReadOnlySpan<byte> effects, Span<byte> report)
    {
        report[..UsbEffectsReportLength].Clear();
        report[0] = UsbEffectsReportId;
        effects[..Math.Min(effects.Length, UsbEffectsReportLength - 1)].CopyTo(report[1..]);
        return UsbEffectsReportLength;
    }

    /// <summary>
    /// Bluetooth output report 0x31: the report ID, SDL's fixed 0x02, the effects state, and the CRC-32 over the HID
    /// transaction header 0xA2 and the report. Returns the report length.
    /// </summary>
    internal static int BuildBluetoothEffectsReport(ReadOnlySpan<byte> effects, Span<byte> report)
    {
        var body = report[..BluetoothEffectsReportLength];
        body.Clear();
        body[0] = BluetoothEffectsReportId;
        body[1] = 0x02;
        effects[..Math.Min(effects.Length, BluetoothEffectsReportLength - 2 - sizeof(uint))].CopyTo(body[2..]);
        WriteCrc(body);
        return BluetoothEffectsReportLength;
    }

    /// <summary>
    /// Bluetooth output report 0x39: the actuators' next <see cref="HapticFrames"/> stereo frames at 3 kHz as signed
    /// bytes and, when <paramref name="speakerPackets"/> holds two encoded Opus packets, the speaker's audio.
    /// </summary>
    internal static void BuildBluetoothAudioReport(
        Span<byte> report,
        byte sequence,
        byte counter,
        ReadOnlySpan<float> haptics,
        ReadOnlySpan<byte> speakerPackets)
    {
        report = report[..BluetoothAudioReportLength];
        report.Clear();
        report[0] = BluetoothAudioReportId;
        report[1] = (byte)(sequence << 4);
        report[2] = 0x91;
        report[3] = 6;
        report[4] = 0x7E;
        report.Slice(5, 4).Fill(AudioBufferLength);
        report[9] = counter;
        report[10] = 0xD2;
        report[11] = HapticBlockBytes;
        for (var index = 0; index < Math.Min(haptics.Length, HapticFrames * 2); index++)
        {
            report[HapticOffset + index] = (byte)(sbyte)ToHapticSample(haptics[index]);
        }

        if (speakerPackets.Length == OpusPackets * OpusPacketBytes)
        {
            report[SpeakerHeaderOffset] = 0xD3;
            report[SpeakerHeaderOffset + 1] = OpusPacketBytes;
            speakerPackets.CopyTo(report[SpeakerOffset..]);
        }

        WriteCrc(report);
    }

    internal static int ToHapticSample(float value) =>
        float.IsFinite(value) ? (int)MathF.Round(Math.Clamp(value, -1f, 1f) * 127f) : 0;

    /// <summary>The CRC-32 of a Bluetooth output report: over 0xA2 and every byte but the last four, stored there.</summary>
    internal static void WriteCrc(Span<byte> report)
    {
        var crc = Crc32(Crc32(0, [0xA2]), report[..^sizeof(uint)]);
        BinaryPrimitives.WriteUInt32LittleEndian(report[^sizeof(uint)..], crc);
    }

    /// <summary>The zlib CRC-32, chained like SDL_crc32 (start with 0).</summary>
    internal static uint Crc32(uint crc, ReadOnlySpan<byte> data)
    {
        crc = ~crc;
        foreach (var value in data)
        {
            crc = CrcTable[(byte)(crc ^ value)] ^ (crc >> 8);
        }

        return ~crc;
    }

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (var index = 0u; index < table.Length; index++)
        {
            var value = index;
            for (var bit = 0; bit < 8; bit++)
            {
                value = (value & 1) != 0 ? 0xEDB88320u ^ (value >> 1) : value >> 1;
            }

            table[index] = value;
        }

        return table;
    }
}
