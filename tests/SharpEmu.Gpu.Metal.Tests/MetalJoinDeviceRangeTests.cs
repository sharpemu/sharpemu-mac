// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A draw into a subset of the open pass's targets whose program reaches memory through device
/// addresses joins the pass while a label copy is held, unless a copy binding did not order is
/// over one of the ranges it reaches: held to the commit, a copy over a range lands at bind time
/// (and ends the pass there); held to the encoder boundary, it refuses the join. Off, every such
/// draw is refused as before. The pixels and the label bytes are the same every way. Device work
/// is behind the exclusive device gate.
/// </summary>
public sealed class MetalJoinDeviceRangeTests(ITestOutputHelper output)
{
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong AddressA = MetalPassRig.GuestBase + 0x10000;
    private const ulong AddressB = MetalPassRig.GuestBase + 0x20000;
    private const ulong Label = MetalPassRig.GuestBase + 0x8000;
    private const ulong Constants = MetalPassRig.GuestBase + 0x9000;

    private sealed record Outcome(byte[] A, byte[] B, byte[] Label, long Passes, long Joins, (long Cleared, long Overlap, long Unordered) Checks);

    [Theory]
    // labelsAtCommit, rangeOverLabel, check, passes, cleared, overlap
    [InlineData(true, false, true, 1L, 1L, 0L)]
    [InlineData(true, false, false, 2L, 0L, 0L)]
    [InlineData(true, true, true, 2L, 0L, 0L)]
    [InlineData(false, false, true, 1L, 1L, 0L)]
    [InlineData(false, true, true, 2L, 0L, 1L)]
    [InlineData(false, true, false, 2L, 0L, 0L)]
    public void ADeviceAddressDrawJoinsUnlessAnUnorderedHeldCopyIsOverItsRanges(
        bool labelsAtCommit, bool rangeOverLabel, bool check, long passes, long cleared, long overlap)
    {
        if (!RequireDevice()) return;
        var outcome = Run(labelsAtCommit, rangeOverLabel, check);
        var reference = Run(labelsAtCommit: false, rangeOverLabel: true, check: false);

        Assert.Equal(passes, outcome.Passes);
        Assert.Equal(passes == 1 ? 1L : 0L, outcome.Joins);
        Assert.Equal((cleared, overlap, 0L), outcome.Checks);
        Assert.Equal(reference.A, outcome.A);
        Assert.Equal(reference.B, outcome.B);
        Assert.Equal(BitConverter.GetBytes(0x55AA55AA55AA55AAUL), outcome.Label);
        for (var x = 0u; x < Width; x++)
        {
            var texel = (int)(x * 4);
            // A: red on the left from the two-target draw, green on the right from the joined one.
            Assert.Equal(x < Width / 2 ? (byte)0xFF : (byte)0x00, outcome.A[texel]);
            Assert.Equal(x < Width / 2 ? (byte)0x00 : (byte)0xFF, outcome.A[texel + 1]);
        }
    }

    private Outcome Run(bool labelsAtCommit, bool rangeOverLabel, bool check)
    {
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.UnionPasses = false;
        host.LabelsAtCommit = labelsAtCommit;
        // The copy is held in the open pass, which is what the join has to order against.
        host.DeferLabelWrites = true;
        host.JoinChecksDeviceRanges = check;
        var two = new[] { GpuFormat.R8G8B8A8Unorm, GpuFormat.R8G8B8A8Unorm };
        var one = new[] { GpuFormat.R8G8B8A8Unorm };
        var both = rig.Pipeline(two, ["1.0f, 0.0f, 0.0f, 1.0f", "0.0f, 0.0f, 1.0f, 1.0f"], -1f, 0f);
        var aOnly = rig.Pipeline(one, ["0.0f, 1.0f, 0.0f, 1.0f"], 0f, 1f);
        var a = rig.MakeTarget(AddressA, GpuFormat.R8G8B8A8Unorm, Width, Height);
        var b = rig.MakeTarget(AddressB, GpuFormat.R8G8B8A8Unorm, Width, Height);
        var stage = MetalPassRig.Stage(
            deviceAddresses: true,
            deviceRanges: [(Constants, 0x100UL), rangeOverLabel ? (Label, 0x100UL) : (Constants + 0x400, 0x100UL)]);
        _ = host.TakeSubsetJoins();
        _ = host.TakeDeviceRangeJoinChecks();
        var first = host.RenderPassSerial;
        using (host.BeginPreparation())
        {
            rig.Draw(both, [a, b]);
            host.RecordCommandProcessorWrite(Label, BitConverter.GetBytes(0x55AA55AA55AA55AAUL), "WRITE_DATA");
            rig.Draw(aOnly, [a], stage);
            host.EndRendering();
        }

        var passes = host.RenderPassSerial - first;
        var (joins, _) = host.TakeSubsetJoins();
        var checks = host.TakeDeviceRangeJoinChecks();
        host.FlushCommandBuffer();
        var outcome = new Outcome(
            rig.Read(AddressA, Width * Height * 4), rig.Read(AddressB, Width * Height * 4), rig.Read(Label, 8), passes, joins, checks);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"labels_at_commit={labelsAtCommit} over={rangeOverLabel} check={check} passes={passes} joins={joins} checks={checks}"));
        return outcome;
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Joins are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
