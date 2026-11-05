// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A guest fill recorded between two draws of an open pass lands when the pass ends: the two
/// draws stay in one pass, and the target and the filled bytes come out as they did when the
/// fill ended the pass. A draw that binds the filled bytes, and a read of them, land it first.
/// Device work is behind the exclusive device gate.
/// </summary>
public sealed class MetalDeferredFillTests(ITestOutputHelper output)
{
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong TargetAddress = MetalPassRig.GuestBase + 0x10000;
    private const ulong FillAddress = MetalPassRig.GuestBase + 0x40000;
    private const ulong FillBytes = 0xC000;

    private enum Scene
    {
        // Nothing reads the filled bytes before the pass ends.
        Free,
        // The second draw binds the filled bytes as a buffer.
        BoundByNextDraw,
        // The filled bytes are read while the fill waits.
        ReadWhileHeld,
    }

    private sealed record Outcome(byte[] Target, byte[] Filled, long Passes, long Deferred, long Early, int HeldAtEnd);

    [Theory]
    [InlineData(0xA1B2C3D4u)]
    [InlineData(0x7F7F7F7Fu)]
    public void AFillBetweenTwoDrawsLandsAtThePassEndWithTheSameResult(uint value)
    {
        if (!RequireDevice()) return;
        var deferred = Run(defer: true, Scene.Free, value);
        var split = Run(defer: false, Scene.Free, value);

        Assert.Equal(1L, deferred.Passes);
        Assert.Equal(1L, deferred.Deferred);
        Assert.Equal(0L, deferred.Early);
        Assert.Equal(2L, split.Passes);
        Assert.Equal(0L, split.Deferred);
        Assert.Equal(split.Target, deferred.Target);
        Assert.Equal(split.Filled, deferred.Filled);
        AssertFilled(deferred.Filled, value);
        AssertHalves(deferred.Target);
    }

    [Fact]
    public void ADrawThatBindsTheFilledBytesLandsTheFillFirst()
    {
        if (!RequireDevice()) return;
        var outcome = Run(defer: true, Scene.BoundByNextDraw, 0xA1B2C3D4u);
        Assert.Equal(1L, outcome.Deferred);
        Assert.Equal(1L, outcome.Early);
        Assert.Equal(2L, outcome.Passes);
        AssertFilled(outcome.Filled, 0xA1B2C3D4u);
        AssertHalves(outcome.Target);
    }

    [Fact]
    public void AReadOfTheFilledBytesSeesTheFill()
    {
        if (!RequireDevice()) return;
        var outcome = Run(defer: true, Scene.ReadWhileHeld, 0xA1B2C3D4u);
        Assert.Equal(1L, outcome.Deferred);
        Assert.Equal(0, outcome.HeldAtEnd);
        AssertFilled(outcome.Filled, 0xA1B2C3D4u);
        AssertHalves(outcome.Target);
    }

    private Outcome Run(bool defer, Scene scene, uint value)
    {
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.DeferPassFills = defer;
        var formats = new[] { GpuFormat.R8G8B8A8Unorm };
        var red = rig.Pipeline(formats, ["1.0f, 0.0f, 0.0f, 1.0f"], -1f, 0f);
        var green = rig.Pipeline(formats, ["0.0f, 1.0f, 0.0f, 1.0f"], 0f, 1f);
        var target = rig.MakeTarget(TargetAddress, GpuFormat.R8G8B8A8Unorm, Width, Height);
        _ = host.TakeDeferredFills();
        var first = host.RenderPassSerial;
        byte[]? readWhileHeld = null;
        var held = 0;
        using (host.BeginPreparation())
        {
            rig.Draw(red, [target]);
            host.FillGuestMemory(FillAddress, FillBytes, value, allowHost: false);
            if (scene == Scene.ReadWhileHeld)
            {
                readWhileHeld = rig.Read(FillAddress, FillBytes);
            }

            var stage = scene == Scene.BoundByNextDraw
                ? MetalPassRig.Stage(buffers: [host.GuestMemory.ResolveWritable(FillAddress + 0x100, 64)])
                : null;
            rig.Draw(green, [target], stage);
            held = host.DeferredFillCount;
            host.EndRendering();
        }

        var passes = host.RenderPassSerial - first;
        var (deferredCount, _, early) = host.TakeDeferredFills();
        host.FlushCommandBuffer();
        var pixels = rig.Read(TargetAddress, Width * Height * 4);
        var filled = readWhileHeld ?? rig.Read(FillAddress, FillBytes);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"defer={defer} scene={scene} value=0x{value:X8} passes={passes} deferred={deferredCount} early={early} held={held}"));
        return new Outcome(pixels, filled, passes, deferredCount, early, scene == Scene.ReadWhileHeld ? held : 0);
    }

    private static void AssertFilled(byte[] bytes, uint value)
    {
        for (var offset = 0; offset < bytes.Length; offset += 4)
        {
            Assert.Equal(value, BitConverter.ToUInt32(bytes, offset));
        }
    }

    private static void AssertHalves(byte[] pixels)
    {
        for (var x = 0u; x < Width; x++)
        {
            var texel = (int)(x * 4);
            Assert.Equal(x < Width / 2 ? (byte)0xFF : (byte)0x00, pixels[texel]);
            Assert.Equal(x < Width / 2 ? (byte)0x00 : (byte)0xFF, pixels[texel + 1]);
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Deferred fills are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
