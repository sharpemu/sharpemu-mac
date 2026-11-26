// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.HLE.Host;
#if SHARPEMU_HOST_MACOS
using SharpEmu.Host.MacOS;
#endif
using Xunit;

namespace SharpEmu.Libs.Tests.Audio;

#if SHARPEMU_HOST_MACOS

/// <summary>
/// The DualSense controller stream on macOS without a pad: the output reports byte for byte as upstream's SDL host
/// builds them, the Opus framing, the channel mapping, the resampling lanes and the device and HID selection. Whether a
/// real pad accepts them is hardware-unverified.
/// </summary>
public class MacDualSenseReportTests
{
    [Fact]
    public void TheSpeakerRouteIsUpstreamsEffectsState()
    {
        var on = MacDualSenseReports.AudioRoute(speaker: true);
        Assert.Equal(MacDualSenseReports.RouteLength, on.Length);
        Assert.Equal(0xA0, on[0]);
        Assert.Equal(0x80, on[1]);
        Assert.Equal(0x64, on[5]);
        Assert.Equal(0x30, on[7]);
        Assert.Equal(0x02, on[37]);
        Assert.Equal(5, on.Count(value => value != 0));

        var off = MacDualSenseReports.AudioRoute(speaker: false);
        Assert.Equal(0x80, off[0]);
        Assert.Equal(1, off.Count(value => value != 0));
    }

    [Fact]
    public void UsbEffectsGoInReportTwoAfterItsId()
    {
        var effects = MacDualSenseReports.AudioRoute(speaker: true);
        var report = new byte[MacDualSenseReports.BluetoothEffectsReportLength];
        Assert.Equal(48, MacDualSenseReports.BuildUsbEffectsReport(effects, report));
        Assert.Equal(0x02, report[0]);
        Assert.Equal(effects, report[1..(1 + effects.Length)]);
        Assert.All(report[(1 + effects.Length)..48], value => Assert.Equal(0, value));
    }

    [Fact]
    public void BluetoothEffectsCarryTheFixedTagAndTheCrc()
    {
        var effects = MacDualSenseReports.AudioRoute(speaker: true);
        var report = new byte[MacDualSenseReports.BluetoothEffectsReportLength];
        Assert.Equal(78, MacDualSenseReports.BuildBluetoothEffectsReport(effects, report));
        Assert.Equal(0x31, report[0]);
        Assert.Equal(0x02, report[1]);
        Assert.Equal(effects, report[2..(2 + effects.Length)]);
        AssertCrc(report);
    }

    [Fact]
    public void TheCrcIsZlibsCrc32ChainedLikeSdl()
    {
        var data = Encoding.ASCII.GetBytes("123456789");
        Assert.Equal(0xCBF43926u, MacDualSenseReports.Crc32(0, data));
        Assert.Equal(
            MacDualSenseReports.Crc32(0, data),
            MacDualSenseReports.Crc32(MacDualSenseReports.Crc32(0, data[..4]), data[4..]));
    }

    [Fact]
    public void TheBluetoothAudioReportHoldsTheActuatorsAndTheSpeakerPackets()
    {
        var haptics = new float[MacDualSenseReports.HapticFrames * 2];
        haptics[0] = 1f;
        haptics[1] = -1f;
        haptics[2] = 0.5f;
        haptics[3] = float.NaN;
        haptics[4] = 4f;
        var packets = new byte[MacDualSenseReports.OpusPackets * MacDualSenseReports.OpusPacketBytes];
        packets.AsSpan().Fill(0x5A);
        var report = new byte[MacDualSenseReports.BluetoothAudioReportLength];

        MacDualSenseReports.BuildBluetoothAudioReport(report, sequence: 3, counter: 8, haptics, packets);

        Assert.Equal(547, report.Length);
        Assert.Equal(new byte[] { 0x39, 0x30, 0x91, 6, 0x7E, 48, 48, 48, 48, 8, 0xD2, 64 }, report[..12]);
        Assert.Equal(127, (sbyte)report[12]);
        Assert.Equal(-127, (sbyte)report[13]);
        Assert.Equal(64, (sbyte)report[14]);
        Assert.Equal(0, (sbyte)report[15]);
        Assert.Equal(127, (sbyte)report[16]);
        Assert.Equal(0xD3, report[140]);
        Assert.Equal(200, report[141]);
        Assert.All(report[142..542], value => Assert.Equal(0x5A, value));
        AssertCrc(report);

        MacDualSenseReports.BuildBluetoothAudioReport(report, sequence: 0, counter: 2, haptics, ReadOnlySpan<byte>.Empty);
        Assert.Equal(0, report[140]);
        Assert.All(report[142..543], value => Assert.Equal(0, value));
        AssertCrc(report);
    }

    [Fact]
    public void TheSpeakerEncodesToTwoConstantSizeOpusPackets()
    {
        var opus = new MacDualSenseOpus();
        var speaker = new float[MacDualSenseBluetooth.SpeakerFrames * 2];
        for (var frame = 0; frame < MacDualSenseBluetooth.SpeakerFrames; frame++)
        {
            speaker[2 * frame] = speaker[(2 * frame) + 1] = 0.25f * MathF.Sin(frame * 2f * MathF.PI * 440f / 45000f);
        }

        var packets = new byte[MacDualSenseReports.OpusPackets * MacDualSenseReports.OpusPacketBytes];
        Assert.True(opus.TryEncode(speaker, packets));
        Assert.Contains(packets[..200], value => value != 0);
        Assert.Contains(packets[200..], value => value != 0);

        Array.Clear(speaker);
        Assert.True(opus.TryEncode(speaker, packets));
    }

    [Fact]
    public void TheStreamSplitsItsFourChannelsIntoTheSpeakerAndActuatorPairs()
    {
        var client = new MacDualSenseBluetooth.Client(null, 48000, queueMilliseconds: 80, speaker: true);
        const int frames = 4800;
        var pcm = new float[frames * MacDualSenseBluetooth.InputChannels];
        for (var frame = 0; frame < frames; frame++)
        {
            pcm[(frame * 4) + 0] = 0.1f;
            pcm[(frame * 4) + 1] = 0.2f;
            pcm[(frame * 4) + 2] = 0.3f;
            pcm[(frame * 4) + 3] = -0.4f;
        }

        Assert.True(client.Submit(MemoryMarshal.AsBytes(pcm.AsSpan())));

        var haptics = new float[MacDualSenseReports.HapticFrames * 2];
        Assert.True(client.Haptics.MixInto(haptics));
        Assert.All(Enumerable.Range(0, MacDualSenseReports.HapticFrames), frame =>
        {
            Assert.Equal(0.3f, haptics[2 * frame], 4);
            Assert.Equal(-0.4f, haptics[(2 * frame) + 1], 4);
        });

        var speaker = new float[MacDualSenseBluetooth.SpeakerFrames * 2];
        Assert.True(client.Speaker!.MixInto(speaker));
        Assert.Equal(0.1f, speaker[0], 4);
        Assert.Equal(0.2f, speaker[1], 4);
        Assert.Equal(0.1f, speaker[^2], 4);
        Assert.Equal(0.2f, speaker[^1], 4);

        // A stream for the actuators alone has no speaker lane.
        Assert.Null(new MacDualSenseBluetooth.Client(null, 48000, 80, speaker: false).Speaker);
    }

    [Fact]
    public void ALaneResamplesPrimesAndDropsItsOldestPastTheLimit()
    {
        var lane = new MacDualSenseLane(48000, 3000, MacDualSenseReports.HapticFrames, queueMilliseconds: 80);
        var report = new float[MacDualSenseReports.HapticFrames * 2];

        // 16 input frames per actuator frame; 1.5 reports must be queued before the first goes out.
        lane.Put(Constant(16 * 95, 0.5f));
        Assert.Equal(95, lane.QueuedFrames);
        Assert.False(lane.MixInto(report));
        lane.Put(Constant(16, 0.5f));
        Assert.True(lane.MixInto(report));
        Assert.All(report, value => Assert.Equal(0.5f, value, 5));
        Assert.Equal(32, lane.QueuedFrames);

        // 80 ms at 3 kHz is 240 frames: a second of audio keeps the newest 240.
        lane.Put(Constant(48000, 0.25f));
        Assert.Equal(240, lane.QueuedFrames);
        Assert.Equal(80, lane.QueuedMilliseconds);
    }

    [Fact]
    public void ADownsampledLaneAveragesWhatTheActuatorsCannotPlay()
    {
        var lane = new MacDualSenseLane(48000, 3000, MacDualSenseReports.HapticFrames, queueMilliseconds: 80);
        // A tone at 12 kHz (four input frames a period) averages out over the 16-frame interval.
        var tone = new float[48000 / 10 * 2];
        for (var frame = 0; frame < tone.Length / 2; frame++)
        {
            tone[2 * frame] = tone[(2 * frame) + 1] = (frame % 4) switch { 0 => 1f, 1 => 0f, 2 => -1f, _ => 0f };
        }

        lane.Put(tone);
        var report = new float[MacDualSenseReports.HapticFrames * 2];
        Assert.True(lane.MixInto(report));
        Assert.All(report, value => Assert.Equal(0f, value, 5));
    }

    [Fact]
    public void OnlyOneFourChannelDualSenseDeviceIsTheController()
    {
        var pad = new MacAudioDeviceInfo(70, "DualSense Wireless Controller", 4);
        var speakers = new MacAudioDeviceInfo(60, "MacBook Air Speakers", 2);
        Assert.True(MacCoreAudioDevices.TrySelectControllerDevice([speakers, pad], out var chosen));
        Assert.Equal(pad, chosen);
        Assert.True(MacCoreAudioDevices.TrySelectControllerDevice(
            [new MacAudioDeviceInfo(71, "Wireless Controller", 4)], out _));
        Assert.False(MacCoreAudioDevices.TrySelectControllerDevice([speakers], out _));
        Assert.False(MacCoreAudioDevices.TrySelectControllerDevice(
            [new MacAudioDeviceInfo(72, "DualSense Wireless Controller", 2)], out _));
        Assert.False(MacCoreAudioDevices.TrySelectControllerDevice(
            [pad, new MacAudioDeviceInfo(73, "DualSense Wireless Controller", 4)], out _));
    }

    [Fact]
    public void OnlyADualSenseGamepadInterfaceHasATransport()
    {
        const int sony = MacDualSenseHid.SonyVendorId;
        Assert.Equal(MacDualSenseTransport.Usb, MacDualSenseHid.Classify(sony, new(0x0CE6, "USB", 1, 5)));
        Assert.Equal(MacDualSenseTransport.Bluetooth, MacDualSenseHid.Classify(sony, new(0x0CE6, "Bluetooth", 1, 5)));
        Assert.Equal(MacDualSenseTransport.Usb, MacDualSenseHid.Classify(sony, new(0x0DF2, "USB", 0, 0)));
        Assert.Null(MacDualSenseHid.Classify(0x045E, new(0x0CE6, "USB", 1, 5)));
        Assert.Null(MacDualSenseHid.Classify(sony, new(0x09CC, "USB", 1, 5)));
        Assert.Null(MacDualSenseHid.Classify(sony, new(0x0CE6, "USB", 12, 1)));
        Assert.Null(MacDualSenseHid.Classify(sony, new(0x0CE6, null, 1, 5)));
    }

    [Fact]
    public void ARingDropsItsOldestFramesOnTheConsumerSide()
    {
        using var ring = new MacAudioRingBuffer(channels: 1, capacityFrames: 256);
        var values = Enumerable.Range(0, 100).Select(value => (float)value).ToArray();
        Assert.Equal(100, ring.Write(values));

        ring.DiscardOldestBeyond(10);
        Assert.Equal(10, ring.QueuedFrames);
        Assert.Equal(156, ring.FreeFrames); // the dropped frames hold their slots until the consumer skips them

        var read = new float[16];
        Assert.Equal(10, ring.Read(read));
        Assert.Equal(Enumerable.Range(90, 10).Select(value => (float)value), read[..10]);
        Assert.Equal(0, ring.QueuedFrames);
        Assert.Equal(256, ring.FreeFrames);
    }

    private static float[] Constant(int frames, float value)
    {
        var samples = new float[frames * 2];
        samples.AsSpan().Fill(value);
        return samples;
    }

    private static void AssertCrc(byte[] report)
    {
        var expected = MacDualSenseReports.Crc32(MacDualSenseReports.Crc32(0, [0xA2]), report[..^4]);
        Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(report.AsSpan(report.Length - 4)));
    }
}

/// <summary>The CoreAudio side of the controller stream and the AudioOut2 queue cap, on this Mac's devices.</summary>
public class MacControllerAudioDeviceTests
{
    [Fact]
    public void ACallerChosenCapHoldsTheQueueDepthAudioOut2Expects()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        IHostAudioStream stream;
        try
        {
            stream = new MacCoreAudioOutput().OpenPcmStream(48000, 2, HostPcmFormat.Float32, 256 * 1024);
        }
        catch (InvalidOperationException)
        {
            return; // no output device on this host
        }

        using (stream)
        {
            // 200 ms of silence: past the 60 ms default cap, well under the 682 ms this cap allows, so it queues
            // without waiting.
            var silence = new byte[48000 / 5 * 2 * sizeof(float)];
            var watch = Stopwatch.StartNew();
            Assert.True(stream.Submit(silence));
            Assert.True(stream.Submit(silence));
            Assert.True(watch.ElapsedMilliseconds < 150, $"submits waited {watch.ElapsedMilliseconds} ms");
            Assert.True(stream.QueuedMilliseconds > 150, $"queued {stream.QueuedMilliseconds} ms");
        }
    }

    [Fact]
    public void WithoutADualSenseThereIsNoControllerStream()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        var stream = new MacCoreAudioOutput().TryOpenControllerStream(48000, 48000 * 16 * 80 / 1000, out var channels);
        if (stream is null)
        {
            Assert.Equal(0, channels);
            return;
        }

        // A DualSense is connected to this Mac: the stream is the four-channel one.
        using (stream)
        {
            Assert.Equal(4, channels);
            Assert.True(stream.Submit(new byte[480 * 16]));
        }
    }
}
#endif
