// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Pad;

namespace SharpEmu.Platform.MacOS.Input;

/// <summary>
/// Gamepad input and output through GameController.framework — the macOS
/// replacement for the SDL gamepad path.
/// </summary>
/// <remarks>
/// GameController wants the main thread: controller connect/disconnect arrives as
/// main-queue notifications and the profile snapshots are read from the run loop.
/// This type is therefore created and polled exclusively from the AppKit window
/// loop (<see cref="Windowing.MacHostWindow"/>), and the <see cref="IHostGamepadOutput"/>
/// calls that arrive from guest threads only record intent under a lock; the
/// polling tick applies it. Connect/disconnect is detected by comparing
/// <c>+[GCController controllers]</c> each tick rather than by installing an
/// NSNotificationCenter observer, which keeps this file free of Objective-C
/// subclassing while still reacting within one frame.
///
/// <c>GCController.shouldMonitorBackgroundEvents</c> is deliberately left alone:
/// the SDL window only fed the guest while focused, and
/// <see cref="HostWindowInput.SetFocused"/> keeps those semantics.
/// </remarks>
[SupportedOSPlatform("macos")]
internal sealed class MacGameControllerGamepad : IHostGamepadOutput, IDisposable
{
    private readonly object _outputGate = new();

    private nint _controller;
    private nint _profile;
    private string _name = "GameController gamepad";
    private HostGamepadType _type = HostGamepadType.Generic;
    private bool _supportsAdaptiveTriggers;
    private bool _supportsTouchpad;

    private MacHapticMotor? _leftHandle;
    private MacHapticMotor? _rightHandle;
    private MacHapticMotor? _leftTriggerMotor;
    private MacHapticMotor? _rightTriggerMotor;

    // Pending output state, written by guest threads and applied on the main thread.
    private byte _largeMotor;
    private byte _smallMotor;
    private byte _leftTriggerRumble;
    private byte _rightTriggerRumble;
    private bool _rumbleDirty;
    // Timed rumble (SetHapticRumble) layered over the steady motors until it expires, as the SDL adapter does.
    private byte _hapticLargeMotor;
    private byte _hapticSmallMotor;
    private long _hapticExpiresAt;
    private bool _hapticApplied;
    private uint _lightbar = 0x000040;
    private bool _lightbarDirty = true;
    private HostAdaptiveTriggerEffect? _pendingLeftEffect;
    private HostAdaptiveTriggerEffect? _pendingRightEffect;
    private bool _effectsDirty;
    private bool _disposed;

    private readonly bool _outputEnabled;

    /// <param name="outputEnabled">
    /// False for read-only consumers such as the launcher's menu navigation,
    /// which must not build CoreHaptics engines or drive a running game's pad.
    /// </param>
    public MacGameControllerGamepad(bool outputEnabled = true)
    {
        _outputEnabled = outputEnabled;
        // Loading the framework is what makes +[GCController controllers] and the
        // GC* classes resolvable through objc_getClass.
        NativeLibrary.TryLoad(MacObjC.GameControllerFramework, out _);
    }

    public bool IsConnected => _controller != 0;

    /// <summary>
    /// Reconciles the connected pad and reads one snapshot without publishing it
    /// to <see cref="HostWindowInput"/> or applying any output. Main thread only.
    /// </summary>
    public bool TryReadState(out HostGamepadState state)
    {
        state = default;
        if (_disposed)
        {
            return false;
        }

        var pool = MacObjC.objc_autoreleasePoolPush();
        try
        {
            SyncConnectedController();
            if (_controller == 0)
            {
                return false;
            }

            state = ReadState();
            return true;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[PAD][WARN] GameController read failed: {exception.Message}");
            return false;
        }
        finally
        {
            MacObjC.objc_autoreleasePoolPop(pool);
        }
    }

    /// <summary>
    /// One tick of the window loop: reconciles the connected controller, applies
    /// queued output, and publishes a fresh snapshot to
    /// <see cref="HostWindowInput"/>. Main thread only.
    /// </summary>
    public void Poll()
    {
        if (_disposed)
        {
            return;
        }

        var pool = MacObjC.objc_autoreleasePoolPush();
        try
        {
            SyncConnectedController();
            if (_controller == 0)
            {
                HostWindowInput.ClearGamepad();
                return;
            }

            if (_outputEnabled)
            {
                ApplyPendingOutput();
            }

            HostWindowInput.SetGamepad(_name, ReadState());
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[PAD][WARN] GameController poll failed: {exception.Message}");
        }
        finally
        {
            MacObjC.objc_autoreleasePoolPop(pool);
        }
    }

    // ---- Connection ----

    private void SyncConnectedController()
    {
        var controllerClass = MacObjC.Class("GCController");
        if (controllerClass == 0)
        {
            return;
        }

        var controllers = MacObjC.Send(controllerClass, MacObjC.Selector("controllers"));
        if (_controller != 0)
        {
            // Membership in +controllers is the connection state; a disconnected
            // GCController instance stays alive but drops out of the array.
            if (controllers != 0 &&
                MacObjC.SendBool(controllers, MacObjC.Selector("containsObject:"), _controller))
            {
                return;
            }

            Console.Error.WriteLine("[PAD][INFO] GameController gamepad disconnected.");
            CloseController();
        }

        var count = MacObjC.ArrayCount(controllers);
        nint selected = 0;
        nint selectedProfile = 0;
        var selectedType = HostGamepadType.Generic;
        for (nuint index = 0; index < count; index++)
        {
            var candidate = MacObjC.ArrayObjectAt(controllers, index);
            var profile = MacObjC.Send(candidate, MacObjC.Selector("extendedGamepad"));
            if (profile == 0)
            {
                continue;
            }

            var type = ClassifyProfile(profile, candidate);
            // Mirror the SDL reader: prefer a real DualSense/DualShock over any
            // other pad so DualSense-only features light up when one is present.
            if (selected == 0 || (selectedType == HostGamepadType.Generic && type != HostGamepadType.Generic))
            {
                selected = candidate;
                selectedProfile = profile;
                selectedType = type;
            }
        }

        if (selected == 0)
        {
            return;
        }

        _controller = MacObjC.Send(selected, MacObjC.Selector("retain"));
        _profile = MacObjC.Send(selectedProfile, MacObjC.Selector("retain"));
        _type = selectedType;
        _name = MacObjC.ReadNsString(MacObjC.Send(_controller, MacObjC.Selector("vendorName")))
            ?? "GameController gamepad";
        _supportsAdaptiveTriggers = _type == HostGamepadType.DualSense &&
            MacObjC.Responds(
                MacObjC.Send(_profile, MacObjC.Selector("leftTrigger")),
                "setModeOff");
        _supportsTouchpad = MacObjC.Responds(_profile, "touchpadPrimary");
        CreateHaptics();
        var motionFed = _outputEnabled && MacMotionFeed.Attach(_controller);
        // A fresh pad starts from the configured lightbar colour and silent motors.
        lock (_outputGate)
        {
            _lightbarDirty = true;
            _rumbleDirty = true;
            _effectsDirty = _supportsAdaptiveTriggers;
        }

        Console.Error.WriteLine(
            $"[PAD][INFO] GameController gamepad connected: {_name} type={_type} " +
            $"adaptive_triggers={_supportsAdaptiveTriggers} touchpad={_supportsTouchpad} motion_feed={motionFed}");
    }

    /// <summary>
    /// Maps a GameController profile to the host pad type. The concrete profile
    /// class is authoritative; <c>-productCategory</c> is the fallback for pads
    /// GameController exposes only as a plain extended gamepad.
    /// </summary>
    internal static HostGamepadType ClassifyProfile(nint profile, nint controller)
    {
        if (MacObjC.IsKindOf(profile, "GCDualSenseGamepad"))
        {
            return HostGamepadType.DualSense;
        }

        if (MacObjC.IsKindOf(profile, "GCDualShockGamepad"))
        {
            return HostGamepadType.DualShock4;
        }

        return MapProductCategory(
            MacObjC.ReadNsString(MacObjC.Send(controller, MacObjC.Selector("productCategory"))));
    }

    /// <summary>GCProductCategory* string to host pad type. Pure — unit tested.</summary>
    internal static HostGamepadType MapProductCategory(string? productCategory) => productCategory switch
    {
        "DualSense" => HostGamepadType.DualSense,
        "DualShock 4" => HostGamepadType.DualShock4,
        _ => HostGamepadType.Generic,
    };

    private void CreateHaptics()
    {
        if (!_outputEnabled)
        {
            return;
        }

        var haptics = MacObjC.Send(_controller, MacObjC.Selector("haptics"));
        if (haptics == 0)
        {
            Console.Error.WriteLine("[PAD][INFO] GameController gamepad has no haptics; rumble is disabled.");
            return;
        }

        _leftHandle = MacHapticMotor.TryCreate(haptics, "GCHapticsLocalityLeftHandle", "left handle");
        _rightHandle = MacHapticMotor.TryCreate(haptics, "GCHapticsLocalityRightHandle", "right handle");
        _leftTriggerMotor = MacHapticMotor.TryCreate(haptics, "GCHapticsLocalityLeftTrigger", "left trigger");
        _rightTriggerMotor = MacHapticMotor.TryCreate(haptics, "GCHapticsLocalityRightTrigger", "right trigger");
    }

    private void CloseController()
    {
        if (_outputEnabled)
        {
            MacMotionFeed.Detach(_controller);
        }

        _leftHandle?.Dispose();
        _rightHandle?.Dispose();
        _leftTriggerMotor?.Dispose();
        _rightTriggerMotor?.Dispose();
        _leftHandle = null;
        _rightHandle = null;
        _leftTriggerMotor = null;
        _rightTriggerMotor = null;

        MacObjC.Release(_profile);
        MacObjC.Release(_controller);
        _profile = 0;
        _controller = 0;
        _type = HostGamepadType.Generic;
        _supportsAdaptiveTriggers = false;
        _supportsTouchpad = false;
        HostWindowInput.ClearGamepad();
    }

    // ---- Input ----

    private HostGamepadState ReadState()
    {
        var buttons = HostGamepadButtons.None;
        AddButton(_profile, "buttonA", HostGamepadButtons.Cross, ref buttons);
        AddButton(_profile, "buttonB", HostGamepadButtons.Circle, ref buttons);
        AddButton(_profile, "buttonX", HostGamepadButtons.Square, ref buttons);
        AddButton(_profile, "buttonY", HostGamepadButtons.Triangle, ref buttons);
        AddButton(_profile, "leftShoulder", HostGamepadButtons.L1, ref buttons);
        AddButton(_profile, "rightShoulder", HostGamepadButtons.R1, ref buttons);
        AddButton(_profile, "leftThumbstickButton", HostGamepadButtons.L3, ref buttons);
        AddButton(_profile, "rightThumbstickButton", HostGamepadButtons.R3, ref buttons);
        // GameController names the two system buttons after their Xbox meaning:
        // buttonOptions is Create/Share, buttonMenu is Options/Start.
        AddButton(_profile, "buttonOptions", HostGamepadButtons.Create, ref buttons);
        AddButton(_profile, "buttonMenu", HostGamepadButtons.Options, ref buttons);
        AddButton(_profile, "buttonHome", HostGamepadButtons.Ps, ref buttons);
        if (_supportsTouchpad)
        {
            AddButton(_profile, "touchpadButton", HostGamepadButtons.TouchPad, ref buttons);
        }

        var dpad = MacObjC.Send(_profile, MacObjC.Selector("dpad"));
        AddButton(dpad, "up", HostGamepadButtons.Up, ref buttons);
        AddButton(dpad, "down", HostGamepadButtons.Down, ref buttons);
        AddButton(dpad, "left", HostGamepadButtons.Left, ref buttons);
        AddButton(dpad, "right", HostGamepadButtons.Right, ref buttons);

        var leftTrigger = ToTriggerByte(ReadAxis(_profile, "leftTrigger"));
        var rightTrigger = ToTriggerByte(ReadAxis(_profile, "rightTrigger"));
        if (leftTrigger > 64)
        {
            buttons |= HostGamepadButtons.L2;
        }

        if (rightTrigger > 64)
        {
            buttons |= HostGamepadButtons.R2;
        }

        var leftStick = MacObjC.Send(_profile, MacObjC.Selector("leftThumbstick"));
        var rightStick = MacObjC.Send(_profile, MacObjC.Selector("rightThumbstick"));
        return new HostGamepadState(
            Connected: true,
            Buttons: buttons,
            LeftX: ToStickByte(ReadAxis(leftStick, "xAxis")),
            // GameController reports sticks with Y growing upward; the host seam
            // (and the guest) expect Y growing downward.
            LeftY: ToStickByte(-ReadAxis(leftStick, "yAxis")),
            RightX: ToStickByte(ReadAxis(rightStick, "xAxis")),
            RightY: ToStickByte(-ReadAxis(rightStick, "yAxis")),
            LeftTrigger: leftTrigger,
            RightTrigger: rightTrigger,
            Type: _type,
            // GameController exposes no wired/wireless transport for a pad, so
            // the guest sees Unknown rather than a guess.
            Connection: HostGamepadConnection.Unknown,
            Motion: ReadMotion(),
            Touch: ReadTouch(),
            BatteryPercent: ReadBatteryPercent());
    }

    private static void AddButton(
        nint owner,
        string selector,
        HostGamepadButtons target,
        ref HostGamepadButtons buttons)
    {
        if (owner == 0)
        {
            return;
        }

        var button = MacObjC.Send(owner, MacObjC.Selector(selector));
        if (button != 0 && MacObjC.SendBool(button, MacObjC.Selector("isPressed")))
        {
            buttons |= target;
        }
    }

    private static float ReadAxis(nint owner, string selector)
    {
        if (owner == 0)
        {
            return 0f;
        }

        var input = MacObjC.Send(owner, MacObjC.Selector(selector));
        return input == 0 ? 0f : MacObjC.SendFloat(input, MacObjC.Selector("value"));
    }

    private HostMotionState ReadMotion()
    {
        var motion = MacObjC.Send(_controller, MacObjC.Selector("motion"));
        if (motion == 0)
        {
            return default;
        }

        var hasAcceleration = MacMotionFeed.TryReadAcceleration(motion, out var acceleration);
        var hasRotation = MacMotionFeed.HasRotationRate(motion);
        if (!hasAcceleration && !hasRotation)
        {
            return default;
        }

        // GCMotion reports acceleration in g; SDL_SENSOR_ACCEL reported m/s^2, so
        // scale to keep guest-visible magnitudes identical.
        acceleration.X *= StandardGravity;
        acceleration.Y *= StandardGravity;
        acceleration.Z *= StandardGravity;

        // GCMotion rotationRate is radians/second, the same unit SDL's gyro used.
        var rotation = hasRotation
            ? MacObjC.SendVector3(motion, MacObjC.Selector("rotationRate"))
            : default;
        return new HostMotionState(
            true,
            (float)acceleration.X,
            (float)acceleration.Y,
            (float)acceleration.Z,
            (float)rotation.X,
            (float)rotation.Y,
            (float)rotation.Z);
    }

    private const double StandardGravity = 9.80665;

    private HostTouchState ReadTouch()
    {
        if (!_supportsTouchpad)
        {
            return default;
        }

        return new HostTouchState(
            ReadTouchPoint("touchpadPrimary", 0),
            ReadTouchPoint("touchpadSecondary", 1));
    }

    /// <summary>
    /// GameController models the DualSense/DualShock touchpad as two direction
    /// pads and exposes no explicit "finger down" flag, so a non-zero position
    /// (or a pressed touchpad button) stands in for contact. A finger resting at
    /// the exact centre therefore reads as lifted — the one touchpad behaviour
    /// that is weaker than the SDL path.
    /// </summary>
    private HostTouchPoint ReadTouchPoint(string selector, byte id)
    {
        var pad = MacObjC.Send(_profile, MacObjC.Selector(selector));
        if (pad == 0)
        {
            return default;
        }

        var x = ReadAxis(pad, "xAxis");
        var y = ReadAxis(pad, "yAxis");
        if (x == 0f && y == 0f)
        {
            return default;
        }

        return new HostTouchPoint(
            true,
            id,
            Math.Clamp((x + 1f) / 2f, 0f, 1f),
            Math.Clamp((1f - y) / 2f, 0f, 1f));
    }

    private byte ReadBatteryPercent()
    {
        if (!MacObjC.Responds(_controller, "battery"))
        {
            return 0;
        }

        var battery = MacObjC.Send(_controller, MacObjC.Selector("battery"));
        if (battery == 0)
        {
            return 0;
        }

        var level = MacObjC.SendFloat(battery, MacObjC.Selector("batteryLevel"));
        return (byte)Math.Clamp((int)MathF.Round(level * 100f), 0, 100);
    }

    // Byte conversions kept bit-identical to the SDL reader's
    // HostWindowInput.ToStickByte/ToTriggerByte (which took SDL's 16-bit axes) so
    // guest-visible stick centres and trigger ramps do not shift with the backend.
    private static byte ToStickByte(float axis)
    {
        var value = (short)Math.Clamp((int)MathF.Round(axis * 32767f), short.MinValue, short.MaxValue);
        var normalized = value + 32768;
        return (byte)Math.Clamp((normalized * 255 + 32767) / 65535, 0, 255);
    }

    private static byte ToTriggerByte(float value)
    {
        var scaled = (short)Math.Clamp((int)MathF.Round(value * 32767f), 0, 32767);
        return (byte)Math.Clamp(scaled * 255 / 32767, 0, 255);
    }

    // ---- Output (IHostGamepadOutput) ----

    public void SetRumble(byte largeMotor, byte smallMotor)
    {
        lock (_outputGate)
        {
            _largeMotor = largeMotor;
            _smallMotor = smallMotor;
            _rumbleDirty = true;
        }
    }

    public void SetHapticRumble(byte largeMotor, byte smallMotor, uint durationMs)
    {
        lock (_outputGate)
        {
            _hapticLargeMotor = largeMotor;
            _hapticSmallMotor = smallMotor;
            _hapticExpiresAt = Environment.TickCount64 + durationMs;
            _rumbleDirty = true;
        }
    }

    public void SetTriggerRumble(byte? leftTrigger, byte? rightTrigger)
    {
        lock (_outputGate)
        {
            if (leftTrigger is { } left)
            {
                _leftTriggerRumble = left;
            }

            if (rightTrigger is { } right)
            {
                _rightTriggerRumble = right;
            }

            _rumbleDirty = true;
        }
    }

    public void SetAdaptiveTriggerEffect(
        HostAdaptiveTriggerEffect? leftTrigger,
        HostAdaptiveTriggerEffect? rightTrigger)
    {
        lock (_outputGate)
        {
            if (leftTrigger is { } left)
            {
                _pendingLeftEffect = left;
            }

            if (rightTrigger is { } right)
            {
                _pendingRightEffect = right;
            }

            _effectsDirty = true;
            // Pads without adaptive triggers fall back to trigger rumble at the
            // effect's own fallback strength, exactly as the SDL path did.
            if (!_supportsAdaptiveTriggers)
            {
                if (leftTrigger is { } fallbackLeft)
                {
                    _leftTriggerRumble = fallbackLeft.FallbackStrength;
                }

                if (rightTrigger is { } fallbackRight)
                {
                    _rightTriggerRumble = fallbackRight.FallbackStrength;
                }

                _rumbleDirty = true;
            }
        }
    }

    public void SetLightbar(byte red, byte green, byte blue)
    {
        lock (_outputGate)
        {
            _lightbar = (uint)((red << 16) | (green << 8) | blue);
            _lightbarDirty = true;
        }
    }

    public void ResetLightbar() => SetLightbar(0, 0, 64);

    private void ApplyPendingOutput()
    {
        byte largeMotor;
        byte smallMotor;
        byte leftTriggerRumble;
        byte rightTriggerRumble;
        uint lightbar;
        bool rumbleDirty;
        bool lightbarDirty;
        bool effectsDirty;
        HostAdaptiveTriggerEffect? leftEffect;
        HostAdaptiveTriggerEffect? rightEffect;
        lock (_outputGate)
        {
            var hapticActive = _hapticExpiresAt - Environment.TickCount64 > 0
                && (_hapticLargeMotor | _hapticSmallMotor) != 0;
            if (_hapticApplied && !hapticActive)
            {
                _rumbleDirty = true; // the timed rumble ran out: drop back to the steady motors
            }

            _hapticApplied = hapticActive;
            largeMotor = Math.Max(_largeMotor, hapticActive ? _hapticLargeMotor : (byte)0);
            smallMotor = Math.Max(_smallMotor, hapticActive ? _hapticSmallMotor : (byte)0);
            leftTriggerRumble = _leftTriggerRumble;
            rightTriggerRumble = _rightTriggerRumble;
            lightbar = _lightbar;
            rumbleDirty = _rumbleDirty;
            lightbarDirty = _lightbarDirty;
            effectsDirty = _effectsDirty;
            leftEffect = _pendingLeftEffect;
            rightEffect = _pendingRightEffect;
            _rumbleDirty = false;
            _lightbarDirty = false;
            _effectsDirty = false;
        }

        if (rumbleDirty)
        {
            _leftHandle?.SetIntensity(largeMotor);
            _rightHandle?.SetIntensity(smallMotor);
            _leftTriggerMotor?.SetIntensity(leftTriggerRumble);
            _rightTriggerMotor?.SetIntensity(rightTriggerRumble);
        }

        if (lightbarDirty)
        {
            ApplyLightbar(lightbar);
        }

        if (effectsDirty && _supportsAdaptiveTriggers)
        {
            if (leftEffect is { } left)
            {
                ApplyAdaptiveTrigger("leftTrigger", MacAdaptiveTriggerMapping.Map(left));
            }

            if (rightEffect is { } right)
            {
                ApplyAdaptiveTrigger("rightTrigger", MacAdaptiveTriggerMapping.Map(right));
            }
        }
    }

    private void ApplyLightbar(uint color)
    {
        var light = MacObjC.Send(_controller, MacObjC.Selector("light"));
        var colorClass = MacObjC.Class("GCColor");
        if (light == 0 || colorClass == 0)
        {
            return;
        }

        var value = MacObjC.SendInitWithRgb(
            MacObjC.Send(colorClass, MacObjC.Selector("alloc")),
            MacObjC.Selector("initWithRed:green:blue:"),
            ((color >> 16) & 0xFF) / 255f,
            ((color >> 8) & 0xFF) / 255f,
            (color & 0xFF) / 255f);
        if (value == 0)
        {
            return;
        }

        try
        {
            MacObjC.SendVoid(light, MacObjC.Selector("setColor:"), value);
        }
        finally
        {
            MacObjC.Release(value);
        }
    }

    /// <summary>
    /// Issues the mapped GCDualSenseAdaptiveTrigger call. The positional setters
    /// (per-zone resistance/amplitude), which match the guest payload exactly, are
    /// used when the OS has them; otherwise the single-boundary setters take the
    /// first active zone and the peak level.
    /// </summary>
    private void ApplyAdaptiveTrigger(string triggerSelector, MacAdaptiveTriggerCommand command)
    {
        var trigger = MacObjC.Send(_profile, MacObjC.Selector(triggerSelector));
        if (trigger == 0)
        {
            return;
        }

        switch (command.Mode)
        {
            case MacAdaptiveTriggerMode.Feedback:
                if (command.HasPositions &&
                    MacObjC.Responds(trigger, "setModeFeedbackWithResistiveStrengths:"))
                {
                    MacObjC.SendVoidTriggerPositions(
                        trigger,
                        MacObjC.Selector("setModeFeedbackWithResistiveStrengths:"),
                        MacTriggerPositions.FromSpan(command.Positions));
                    return;
                }

                MacObjC.SendVoidFloat2(
                    trigger,
                    MacObjC.Selector("setModeFeedbackWithStartPosition:resistiveStrength:"),
                    command.StartPosition,
                    command.Strength);
                return;

            case MacAdaptiveTriggerMode.Weapon:
                MacObjC.SendVoidFloat3(
                    trigger,
                    MacObjC.Selector("setModeWeaponWithStartPosition:endPosition:resistiveStrength:"),
                    command.StartPosition,
                    command.EndPosition,
                    command.Strength);
                return;

            case MacAdaptiveTriggerMode.Vibration:
                if (command.HasPositions &&
                    MacObjC.Responds(trigger, "setModeVibrationWithAmplitudes:frequency:"))
                {
                    MacObjC.SendVoidTriggerPositionsFloat(
                        trigger,
                        MacObjC.Selector("setModeVibrationWithAmplitudes:frequency:"),
                        MacTriggerPositions.FromSpan(command.Positions),
                        command.Frequency);
                    return;
                }

                MacObjC.SendVoidFloat3(
                    trigger,
                    MacObjC.Selector("setModeVibrationWithStartPosition:amplitude:frequency:"),
                    command.StartPosition,
                    command.Strength,
                    command.Frequency);
                return;

            default:
                MacObjC.SendVoid(trigger, MacObjC.Selector("setModeOff"));
                return;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        var pool = MacObjC.objc_autoreleasePoolPush();
        try
        {
            if (_controller != 0 && _supportsAdaptiveTriggers)
            {
                ApplyAdaptiveTrigger("leftTrigger", MacAdaptiveTriggerCommand.Off);
                ApplyAdaptiveTrigger("rightTrigger", MacAdaptiveTriggerCommand.Off);
            }

            if (_controller != 0)
            {
                CloseController();
            }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[PAD][WARN] GameController teardown failed: {exception.Message}");
        }
        finally
        {
            MacObjC.objc_autoreleasePoolPop(pool);
        }
    }
}
