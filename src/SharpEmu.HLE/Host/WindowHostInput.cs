// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

/// <summary>
/// Routes emulated input through the active cross-platform host window.
/// </summary>
/// <remarks>
/// Host-package contract: every <c>SharpEmu.Host.*</c> package returns this as
/// its <see cref="IHostPlatform.Input"/>, because the window - not the OS host
/// layer - is what owns device discovery and the input pump.
/// </remarks>
public sealed class WindowHostInput : IHostInput
{
    private readonly bool _scriptedOnly;

    public WindowHostInput()
        : this(ScriptedKeyInput.IsEnabled &&
            Environment.GetEnvironmentVariable("SHARPEMU_INPUT_SCRIPTED_ONLY") == "1")
    {
    }

    internal WindowHostInput(bool scriptedOnly) => _scriptedOnly = scriptedOnly;

    // A controlled run must not mix file input with keys/sticks from a focused window.
    // Without both an input file and the explicit opt-in, physical input stays available.
    private IHostWindowInputSource? PhysicalInput =>
        _scriptedOnly ? null : HostWindowInputSource.Current;

    public void EnsureStarted()
    {
        // SDL owns device discovery and pumps it on the window thread.
    }

    public int GetGamepadStates(Span<HostGamepadState> destination) =>
        PhysicalInput?.GetGamepadStates(destination) ?? 0;

    public string? DescribeConnectedGamepad() =>
        PhysicalInput?.DescribeConnectedGamepad();

    public void SetRumble(byte largeMotor, byte smallMotor) =>
        PhysicalInput?.SetRumble(largeMotor, smallMotor);

    public void SetHapticRumble(byte largeMotor, byte smallMotor, uint durationMs) =>
        HostWindowInputSource.Current?.SetHapticRumble(largeMotor, smallMotor, durationMs);

    public void SetTriggerRumble(byte? leftTrigger, byte? rightTrigger) =>
        PhysicalInput?.SetTriggerRumble(leftTrigger, rightTrigger);

    public void SetAdaptiveTriggerEffect(
        HostAdaptiveTriggerEffect? leftTrigger,
        HostAdaptiveTriggerEffect? rightTrigger) =>
        PhysicalInput?.SetAdaptiveTriggerEffect(leftTrigger, rightTrigger);

    public void SetLightbar(byte red, byte green, byte blue) =>
        PhysicalInput?.SetLightbar(red, green, blue);

    public void ResetLightbar() => PhysicalInput?.ResetLightbar();

    public bool IsHostWindowFocused() =>
        (PhysicalInput?.HasKeyboardFocus ?? false) ||
        (ScriptedKeyInput.IsEnabled && ScriptedKeyInput.AnyHeld);

    public bool IsKeyDown(int virtualKey) =>
        (ScriptedKeyInput.IsEnabled && ScriptedKeyInput.IsHeld(virtualKey)) ||
        (PhysicalInput?.IsKeyDown(virtualKey) ?? false);
}
