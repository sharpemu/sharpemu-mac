// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.GpuRemote;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.GpuSplit;

/// <summary>The emulator's types to and from the plain payloads of GpuSplitMessages.cs.</summary>
internal static class SplitConvert
{
    public static VideoOptionsWire ToWire(HostVideoOptions options) => new()
    {
        WindowMode = (int)options.WindowMode,
        ScalingMode = (int)options.ScalingMode,
        Width = options.Width,
        Height = options.Height,
        DisplayIndex = options.DisplayIndex,
        RefreshRate = options.RefreshRate,
        VSync = options.VSync ? 1 : 0,
        HdrMode = (int)options.HdrMode,
        DisplayResolution = (int)options.DisplayResolution,
        OverlayEnabled = options.OverlayEnabled ? 1 : 0,
        OverlayCorner = (int)options.OverlayCorner,
        OverlayMode = (int)options.OverlayMode,
    };

    public static HostVideoOptions FromWire(in VideoOptionsWire wire) => new HostVideoOptions
    {
        WindowMode = (HostWindowMode)wire.WindowMode,
        ScalingMode = (HostScalingMode)wire.ScalingMode,
        Width = wire.Width,
        Height = wire.Height,
        DisplayIndex = wire.DisplayIndex,
        RefreshRate = wire.RefreshRate,
        VSync = wire.VSync != 0,
        HdrMode = (HostHdrMode)wire.HdrMode,
        DisplayResolution = (GuestDisplayResolution)wire.DisplayResolution,
        OverlayEnabled = wire.OverlayEnabled != 0,
        OverlayCorner = (PerformanceOverlayCorner)wire.OverlayCorner,
        OverlayMode = (PerformanceOverlayMode)wire.OverlayMode,
    }.Normalize();

    public static GamepadStateWire ToWire(in HostGamepadState state) => new()
    {
        Connected = state.Connected ? 1u : 0u,
        Buttons = (uint)state.Buttons,
        LeftX = state.LeftX,
        LeftY = state.LeftY,
        RightX = state.RightX,
        RightY = state.RightY,
        LeftTrigger = state.LeftTrigger,
        RightTrigger = state.RightTrigger,
        Type = (byte)state.Type,
        Connection = (byte)state.Connection,
        MotionAvailable = state.Motion.Available ? 1u : 0u,
        AccelerationX = state.Motion.AccelerationX,
        AccelerationY = state.Motion.AccelerationY,
        AccelerationZ = state.Motion.AccelerationZ,
        AngularVelocityX = state.Motion.AngularVelocityX,
        AngularVelocityY = state.Motion.AngularVelocityY,
        AngularVelocityZ = state.Motion.AngularVelocityZ,
        OrientationX = state.Motion.OrientationX,
        OrientationY = state.Motion.OrientationY,
        OrientationZ = state.Motion.OrientationZ,
        OrientationW = state.Motion.OrientationW,
        FirstTouchActive = state.Touch.First.Active ? 1u : 0u,
        FirstTouchId = state.Touch.First.Id,
        FirstTouchX = state.Touch.First.X,
        FirstTouchY = state.Touch.First.Y,
        SecondTouchActive = state.Touch.Second.Active ? 1u : 0u,
        SecondTouchId = state.Touch.Second.Id,
        SecondTouchX = state.Touch.Second.X,
        SecondTouchY = state.Touch.Second.Y,
        BatteryPercent = state.BatteryPercent,
    };

    public static HostGamepadState FromWire(in GamepadStateWire wire) => new(
        wire.Connected != 0,
        (HostGamepadButtons)wire.Buttons,
        wire.LeftX,
        wire.LeftY,
        wire.RightX,
        wire.RightY,
        wire.LeftTrigger,
        wire.RightTrigger,
        (HostGamepadType)wire.Type,
        (HostGamepadConnection)wire.Connection,
        new HostMotionState(
            wire.MotionAvailable != 0,
            wire.AccelerationX,
            wire.AccelerationY,
            wire.AccelerationZ,
            wire.AngularVelocityX,
            wire.AngularVelocityY,
            wire.AngularVelocityZ,
            wire.OrientationX,
            wire.OrientationY,
            wire.OrientationZ,
            wire.OrientationW),
        new HostTouchState(
            new HostTouchPoint(wire.FirstTouchActive != 0, (byte)wire.FirstTouchId, wire.FirstTouchX, wire.FirstTouchY),
            new HostTouchPoint(wire.SecondTouchActive != 0, (byte)wire.SecondTouchId, wire.SecondTouchX, wire.SecondTouchY)),
        (byte)wire.BatteryPercent);

    public static DisplayBufferResponse ToWire(in VideoOutExports.DisplayBufferInfo info) => new()
    {
        Found = 1,
        TilingMode = info.TilingMode,
        Address = info.Address,
        PixelFormat = info.PixelFormat,
        Width = info.Width,
        Height = info.Height,
        PitchInPixel = info.PitchInPixel,
        Option = info.Option,
    };

    public static VideoOutExports.DisplayBufferInfo FromWire(in DisplayBufferResponse wire) =>
        new(wire.Address, wire.PixelFormat, wire.TilingMode, wire.Width, wire.Height, wire.PitchInPixel, wire.Option);

    public static AdaptiveTriggerWire ToWire(HostAdaptiveTriggerEffect? effect) => effect is not { } value
        ? default
        : new AdaptiveTriggerWire
        {
            Present = 1,
            FallbackStrength = value.FallbackStrength,
            B0 = value.B0, B1 = value.B1, B2 = value.B2, B3 = value.B3, B4 = value.B4, B5 = value.B5,
            B6 = value.B6, B7 = value.B7, B8 = value.B8, B9 = value.B9, B10 = value.B10,
        };

    public static HostAdaptiveTriggerEffect? FromWire(in AdaptiveTriggerWire wire) => wire.Present == 0
        ? null
        : new HostAdaptiveTriggerEffect(
            wire.B0, wire.B1, wire.B2, wire.B3, wire.B4, wire.B5, wire.B6, wire.B7, wire.B8, wire.B9, wire.B10, wire.FallbackStrength);
}
