// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using SharpEmu.HLE.Host;

namespace SharpEmu.Libs.Pad;

/// <summary>Cross-platform input state supplied by the SDL game window.</summary>
public static class HostWindowInput
{
    private static readonly object Gate = new();
    private static readonly HashSet<int> PressedKeys = new();
    private static bool _focused;
    private static bool _gamepadConnected;
    private static string? _gamepadName;
    private static HostGamepadState _gamepadState;
    private static IHostGamepadOutput? _gamepadOutput;
    private static long _version;
    private static readonly WindowInputSource Source = new();
    private static readonly MotionOrientation Orientation = new();
    private const float StandardGravity = 9.80665f;

    /// <summary>Moves on every change of the state below; a reader polls it to find changes.</summary>
    public static long Version => Interlocked.Read(ref _version);

    // Under Gate: moves the version and wakes a WaitForChange.
    private static void Bump()
    {
        _version++;
        Monitor.PulseAll(Gate);
    }

    /// <summary>
    /// Blocks until <see cref="Version"/> differs from <paramref name="seen"/> or <paramref name="timeout"/>
    /// passes; true when it changed. A change is seen as soon as it is made, not at the next poll.
    /// </summary>
    public static bool WaitForChange(long seen, TimeSpan timeout)
    {
        lock (Gate)
        {
            if (_version != seen)
            {
                return true;
            }

            _ = Monitor.Wait(Gate, timeout);
            return _version != seen;
        }
    }

    /// <summary>
    /// The whole state at once, for the two-process split's GPU process, which owns the window and
    /// sends it to the guest process: the pressed keys (as many as fit), whether the window has
    /// focus, and the gamepad. Returns how many keys it wrote.
    /// </summary>
    public static int CopyState(Span<int> keys, out bool focused, out HostGamepadState gamepad, out string? gamepadName)
    {
        lock (Gate)
        {
            focused = _focused;
            gamepad = _gamepadConnected ? _gamepadState : default;
            gamepadName = _gamepadConnected ? _gamepadName : null;
            var count = 0;
            foreach (var key in PressedKeys)
            {
                if (count == keys.Length)
                {
                    break;
                }

                keys[count++] = key;
            }

            return count;
        }
    }

    /// <summary>Replaces the pressed keys with <paramref name="keys"/> (the guest process's side of <see cref="CopyState"/>).</summary>
    public static void SetKeys(ReadOnlySpan<int> keys)
    {
        bool changed;
        lock (Gate)
        {
            var before = PressedKeys.Count;
            var kept = 0;
            foreach (var key in keys)
            {
                if (PressedKeys.Contains(key))
                {
                    kept++;
                }
            }

            PressedKeys.Clear();
            foreach (var key in keys)
            {
                PressedKeys.Add(key);
            }

            changed = kept != before || PressedKeys.Count != before;
            Bump();
        }

        // The guest process's pad history (PadExports.RecordInputChange) sees the split's key changes too.
        if (changed) Changed?.Invoke();
    }

    public static void Connect(IHostGamepadOutput? gamepadOutput = null)
    {
        lock (Gate)
        {
            _focused = true;
            _gamepadOutput = gamepadOutput;
            PressedKeys.Clear();
            Bump();
        }

        HostWindowInputSource.Set(Source);
        Changed?.Invoke();
    }

    public static void Disconnect()
    {
        lock (Gate)
        {
            _focused = false;
            _gamepadConnected = false;
            _gamepadName = null;
            _gamepadState = default;
            _gamepadOutput = null;
            PressedKeys.Clear();
            Orientation.Reset();
            Bump();
        }

        HostWindowInputSource.Clear(Source);
        Changed?.Invoke();
    }

    // Called after each change of the window, a key, the focus, or a gamepad's buttons or touch contacts, as it
    // happens: the pad keeps the states between two reads of the game (PadExports.RecordInputChange), so a press and
    // its release in between both reach it.
    internal static Action? Changed;

    public static void SetFocused(bool focused)
    {
        bool changed;
        lock (Gate)
        {
            changed = _focused != focused || (!focused && PressedKeys.Count != 0);
            _focused = focused;
            if (!focused)
            {
                PressedKeys.Clear();
            }

            Bump();
        }

        if (changed) Changed?.Invoke();
    }

    public static void SetKey(int virtualKey, bool down)
    {
        bool changed;
        lock (Gate)
        {
            changed = down ? PressedKeys.Add(virtualKey) : PressedKeys.Remove(virtualKey);
            Bump();
        }

        if (changed && down) SystemDialogInput.OnKeyDown(virtualKey);
        if (changed) Changed?.Invoke();
    }

    public static void SetGamepad(string? name, HostGamepadState state) =>
        SetGamepad(name, state, sourceOrientation: false);

    /// <summary>
    /// <paramref name="sourceOrientation"/>: the state already carries the pad's orientation, accumulated where the
    /// gamepad's sensor readings arrive (the two-process split's GPU process), so it is kept instead of this process's.
    /// </summary>
    public static void SetGamepad(string? name, HostGamepadState state, bool sourceOrientation)
    {
        bool changed;
        lock (Gate)
        {
            changed = _gamepadConnected != state.Connected || _gamepadState.Buttons != state.Buttons ||
                      !SameContacts(_gamepadState.Touch, state.Touch);
            if (state.Motion.Available && !sourceOrientation)
            {
                var orientation = Orientation.Orientation;
                state = state with
                {
                    Motion = state.Motion with
                    {
                        OrientationX = orientation.X,
                        OrientationY = orientation.Y,
                        OrientationZ = orientation.Z,
                        OrientationW = orientation.W,
                    },
                };
            }

            _gamepadConnected = state.Connected;
            _gamepadName = name;
            _gamepadState = state;
            Bump();
        }

        if (changed) Changed?.Invoke();
    }

    // One button event of the connected gamepad, applied as it comes: a poll sees only where the buttons ended up.
    public static void SetGamepadButton(HostGamepadButtons button, bool down)
    {
        bool changed;
        lock (Gate)
        {
            var buttons = down ? _gamepadState.Buttons | button : _gamepadState.Buttons & ~button;
            changed = _gamepadConnected && buttons != _gamepadState.Buttons;
            if (changed)
            {
                _gamepadState = _gamepadState with { Buttons = buttons };
                Bump();
            }
        }

        if (changed) Changed?.Invoke();
    }

    public static void ClearGamepad()
    {
        bool changed;
        lock (Gate)
        {
            changed = _gamepadConnected;
            _gamepadConnected = false;
            _gamepadName = null;
            _gamepadState = default;
            Orientation.Reset();
            Bump();
        }

        if (changed) Changed?.Invoke();
    }

    // One sensor reading of the connected gamepad as SDL reports it (acceleration in m/s^2, angular velocity in
    // radians per second, the sensor's timestamp in nanoseconds). Each gyro reading turns the pad's orientation by its
    // own interval, which a poll of the latest reading cannot do.
    public static void ApplyGamepadMotion(bool angularVelocity, float x, float y, float z, ulong timestampNanoseconds)
    {
        lock (Gate)
        {
            if (angularVelocity)
            {
                Orientation.ApplyAngularVelocity(new Vector3(x, y, z), timestampNanoseconds);
            }
            else
            {
                Orientation.ApplyAcceleration(new Vector3(x, y, z) / StandardGravity);
            }
        }
    }

    // The game's reset and tilt correction also go to the gamepad output, which forwards them to the process that
    // accumulates the orientation when that is another one (the two-process split). That process applies them with
    // forwardToOutput false, so they never travel back.
    public static void ResetMotionOrientation(bool forwardToOutput = true)
    {
        IHostGamepadOutput? output;
        lock (Gate)
        {
            Orientation.Reset();
            output = forwardToOutput ? _gamepadOutput : null;
        }

        output?.ResetMotionOrientation();
    }

    public static void SetMotionTiltCorrection(bool enabled, bool forwardToOutput = true)
    {
        IHostGamepadOutput? output;
        lock (Gate)
        {
            Orientation.TiltCorrection = enabled;
            output = forwardToOutput ? _gamepadOutput : null;
        }

        output?.SetMotionTiltCorrection(enabled);
    }

    private static bool SameContacts(HostTouchState left, HostTouchState right) =>
        left.First.Active == right.First.Active && left.First.Id == right.First.Id &&
        left.Second.Active == right.Second.Active && left.Second.Id == right.Second.Id;

    private sealed class WindowInputSource : IHostWindowInputSource
    {
        public bool HasKeyboardFocus
        {
            get
            {
                lock (Gate)
                {
                    return _focused;
                }
            }
        }

        public bool IsKeyDown(int virtualKey)
        {
            lock (Gate)
            {
                return PressedKeys.Contains(virtualKey);
            }
        }

        public int GetGamepadStates(Span<HostGamepadState> destination)
        {
            lock (Gate)
            {
                if (!_gamepadConnected || destination.IsEmpty)
                {
                    return 0;
                }

                destination[0] = _gamepadState;
                return 1;
            }
        }

        public string? DescribeConnectedGamepad()
        {
            lock (Gate)
            {
                return _gamepadConnected ? _gamepadName ?? "SDL gamepad" : null;
            }
        }

        public void SetRumble(byte largeMotor, byte smallMotor)
        {
            IHostGamepadOutput? output;
            lock (Gate)
            {
                output = _gamepadOutput;
            }

            output?.SetRumble(largeMotor, smallMotor);
        }

        public void SetHapticRumble(byte largeMotor, byte smallMotor, uint durationMs)
        {
            IHostGamepadOutput? output;
            lock (Gate)
            {
                output = _gamepadOutput;
            }

            output?.SetHapticRumble(largeMotor, smallMotor, durationMs);
        }

        public void SetTriggerRumble(byte? leftTrigger, byte? rightTrigger)
        {
            IHostGamepadOutput? output;
            lock (Gate)
            {
                output = _gamepadOutput;
            }

            output?.SetTriggerRumble(leftTrigger, rightTrigger);
        }

        public void SetAdaptiveTriggerEffect(
            HostAdaptiveTriggerEffect? leftTrigger,
            HostAdaptiveTriggerEffect? rightTrigger)
        {
            IHostGamepadOutput? output;
            lock (Gate)
            {
                output = _gamepadOutput;
            }

            output?.SetAdaptiveTriggerEffect(leftTrigger, rightTrigger);
        }

        public void SetLightbar(byte red, byte green, byte blue)
        {
            IHostGamepadOutput? output;
            lock (Gate)
            {
                output = _gamepadOutput;
            }

            output?.SetLightbar(red, green, blue);
        }

        public void ResetLightbar()
        {
            IHostGamepadOutput? output;
            lock (Gate)
            {
                output = _gamepadOutput;
            }

            output?.ResetLightbar();
        }
    }
}

public interface IHostGamepadOutput
{
    void SetRumble(byte largeMotor, byte smallMotor);

    void SetHapticRumble(byte largeMotor, byte smallMotor, uint durationMs);

    void SetTriggerRumble(byte? leftTrigger, byte? rightTrigger);

    void SetAdaptiveTriggerEffect(
        HostAdaptiveTriggerEffect? leftTrigger,
        HostAdaptiveTriggerEffect? rightTrigger);

    void SetLightbar(byte red, byte green, byte blue);

    void ResetLightbar();

    /// <summary>scePadResetOrientation, for an output whose process accumulates the orientation elsewhere.</summary>
    void ResetMotionOrientation()
    {
    }

    /// <summary>scePadSetTiltCorrectionState, for an output whose process accumulates the orientation elsewhere.</summary>
    void SetMotionTiltCorrection(bool enabled)
    {
    }
}
