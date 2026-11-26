// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.GpuRemote;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Pad;

namespace SharpEmu.GpuSplit;

/// <summary>
/// The guest process's gamepad output: the pad lives with the window in the GPU process, so
/// rumble, trigger effects and the lightbar are sent there.
/// </summary>
internal sealed class SplitGamepadOutput(SplitWriter writer) : IHostGamepadOutput
{
    public void SetRumble(byte largeMotor, byte smallMotor) =>
        Send(new GamepadOutputMessage { Kind = (uint)GpuSplitGamepadOutput.Rumble, A = largeMotor, B = smallMotor });

    public void SetHapticRumble(byte largeMotor, byte smallMotor, uint durationMs) =>
        Send(new GamepadOutputMessage
        {
            Kind = (uint)GpuSplitGamepadOutput.HapticRumble, A = largeMotor, B = smallMotor, DurationMs = durationMs,
        });

    public void SetTriggerRumble(byte? leftTrigger, byte? rightTrigger) =>
        Send(new GamepadOutputMessage
        {
            Kind = (uint)GpuSplitGamepadOutput.TriggerRumble,
            A = leftTrigger ?? 0,
            B = rightTrigger ?? 0,
            Flags = (byte)((leftTrigger.HasValue ? 1 : 0) | (rightTrigger.HasValue ? 2 : 0)),
        });

    public void SetAdaptiveTriggerEffect(HostAdaptiveTriggerEffect? leftTrigger, HostAdaptiveTriggerEffect? rightTrigger) =>
        Send(new GamepadOutputMessage
        {
            Kind = (uint)GpuSplitGamepadOutput.AdaptiveTriggers,
            Left = SplitConvert.ToWire(leftTrigger),
            Right = SplitConvert.ToWire(rightTrigger),
        });

    public void SetLightbar(byte red, byte green, byte blue) =>
        Send(new GamepadOutputMessage { Kind = (uint)GpuSplitGamepadOutput.Lightbar, A = red, B = green, C = blue });

    public void ResetLightbar() => Send(new GamepadOutputMessage { Kind = (uint)GpuSplitGamepadOutput.ResetLightbar });

    private void Send(in GamepadOutputMessage message) => _ = writer.Write(GpuRemoteMessage.GamepadOutput, in message);
}
