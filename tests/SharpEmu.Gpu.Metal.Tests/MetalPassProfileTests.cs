// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Text.RegularExpressions;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// SHARPEMU_METAL_V2_PROFILE_PASSES=1 times every encoder the host opens on the command buffer
/// it records, not only render passes and the shared compute encoder: an upload, a guest fill
/// and a writeback made through the host's own paths each come out as a blit or tiler-compute
/// record of their own, timed and named, while render and compute records stay one per encoder.
/// Device work is behind the exclusive device gate.
/// </summary>
public sealed partial class MetalPassProfileTests(ITestOutputHelper output)
{
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong TargetAddress = MetalPassRig.GuestBase + 0x10000;
    private const ulong FillAddress = MetalPassRig.GuestBase + 0x40000;
    private const ulong FillBytes = 0xC000;

    [GeneratedRegex(@"\[METAL-V2\]\[PASS-PROF\] f=\d+ s=\d+ i=\d+ k=(\S+) us=(\S+) .* a=\[(.*?)\] sh=")]
    private static partial Regex RecordLine();

    private sealed record Record(string Kind, string Microseconds, string Label);

    [Fact]
    public void UploadsFillsAndWritebacksGetTimedRecordsOfTheirOwn()
    {
        if (!RequireDevice()) return;
        var saved = MetalRenderHost.PassProfileEnabled;
        MetalRenderHost.PassProfileEnabled = true;
        try
        {
            using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
            var host = rig.Host;
            var printed = new List<string>();
            host.PassProfileOutputForTest = printed.Add;
            host.DeferPassFills = false;
            host.FillsShareCompute = false;
            var formats = new[] { GpuFormat.R8G8B8A8Unorm };
            // The left half: the right half keeps what the target held, so the pass loads it.
            var red = rig.Pipeline(formats, ["1.0f, 0.0f, 0.0f, 1.0f"], -1f, 0f);
            // What the guest left in the target: its first use uploads it.
            rig.WriteRaw(TargetAddress, Enumerable.Repeat((byte)0x40, (int)(Width * Height * 4)).ToArray());
            var passes = host.RenderPassSerial;
            var computes = host.ComputeEncodersOpenedForTest;
            var target = rig.MakeTarget(TargetAddress, GpuFormat.R8G8B8A8Unorm, Width, Height);
            using (host.BeginPreparation())
            {
                rig.Draw(red, [target]);
                host.EndRendering();
            }

            host.FillGuestMemory(FillAddress, FillBytes, 0xA1B2C3D4u, allowHost: false);
            // The CPU reads what the draw wrote: the target is written back.
            var pixels = rig.Read(TargetAddress, Width * Height * 4);
            _ = host.TakeGpuBusySecondsWhenIdle();
            passes = host.RenderPassSerial - passes;
            computes = host.ComputeEncodersOpenedForTest - computes;

            foreach (var line in printed)
            {
                output.WriteLine(line);
            }

            Assert.Equal(0xFF, pixels[0]);
            Assert.Equal(0x40, pixels[(Width - 1) * 4]);
            Assert.Equal(0xA1B2C3D4u, BitConverter.ToUInt32(rig.Read(FillAddress, 4)));
            var records = printed.SelectMany(text => text.Split('\n'))
                .Select(line => RecordLine().Match(line))
                .Where(match => match.Success)
                .Select(match => new Record(match.Groups[1].Value, match.Groups[2].Value, match.Groups[3].Value))
                .ToList();
            Assert.Equal(passes, records.Count(record => record.Kind == "render"));
            Assert.Equal(computes, records.Count(record => record.Kind == "compute"));
            AssertTimed(records, "Upload");
            AssertTimed(records, "GuestFill fill bytes=0xC000");
            AssertTimed(records, "Writeback");
        }
        finally
        {
            MetalRenderHost.PassProfileEnabled = saved;
        }
    }

    private static void AssertTimed(List<Record> records, string label)
    {
        var record = records.FirstOrDefault(record =>
            record.Kind is "blit" or "tiler-compute" && record.Label.StartsWith(label, StringComparison.Ordinal));
        Assert.True(record is not null, $"no blit or tiler-compute record labelled '{label}'");
        Assert.True(
            double.TryParse(record.Microseconds, NumberStyles.Float, CultureInfo.InvariantCulture, out var microseconds) &&
            microseconds > 0,
            $"record '{record.Label}' has us={record.Microseconds}");
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("The pass profile samples the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
