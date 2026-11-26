// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using SharpEmu.Libs.Pad;

namespace SharpEmu.Platform.MacOS.Input;

/// <summary>
/// Feeds every sensor reading of the open GameController pad to the pad's orientation
/// (<see cref="HostWindowInput.ApplyGamepadMotion"/>), as the SDL window does with its sensor events: each gyro reading
/// turns the orientation by its own interval, which the window loop's once-a-frame poll of the latest reading cannot
/// do. The readings come from <c>GCMotion.valueChangedHandler</c> on a serial queue of their own (the controller's
/// <c>handlerQueue</c>), so they arrive as GameController receives them, not batched behind the main run loop.
/// </summary>
/// <remarks>
/// GCMotion carries no per-reading timestamp, so a reading is stamped with the host's monotonic clock when its
/// handler runs. Attitude (<c>hasAttitude</c>) is reported only by Siri Remote-class devices, not by the DualSense,
/// DualShock 4 or Xbox pads, so the orientation always comes from the gyro.
/// </remarks>
[SupportedOSPlatform("macos")]
internal static unsafe class MacMotionFeed
{
    private const int BlockIsGlobal = 1 << 28;
    private const double StandardGravity = 9.80665;

    private static readonly object Gate = new();
    private static BlockLiteral* _block;
    private static nint _queue;
    private static nint _motion;

    [StructLayout(LayoutKind.Sequential)]
    private struct BlockDescriptor
    {
        public nuint Reserved;
        public nuint Size;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BlockLiteral
    {
        public nint Isa;
        public int Flags;
        public int Reserved;
        public delegate* unmanaged<nint, nint, void> Invoke;
        public BlockDescriptor* Descriptor;
    }

    /// <summary>
    /// Starts the readings of <paramref name="controller"/>'s motion: turns its sensors on where they need it (the
    /// DualSense and DualShock 4 start with them off) and installs the handler. False when the pad has no motion.
    /// </summary>
    public static bool Attach(nint controller)
    {
        var motion = controller == 0 ? 0 : MacObjC.Send(controller, MacObjC.Selector("motion"));
        if (motion == 0)
        {
            return false;
        }

        lock (Gate)
        {
            if (!EnsureBlock())
            {
                return false;
            }

            if (MacObjC.Responds(motion, "sensorsRequireManualActivation") &&
                MacObjC.SendBool(motion, MacObjC.Selector("sensorsRequireManualActivation")))
            {
                MacObjC.SendVoidBool(motion, MacObjC.Selector("setSensorsActive:"), true);
            }

            if (MacObjC.Responds(controller, "setHandlerQueue:"))
            {
                MacObjC.SendVoid(controller, MacObjC.Selector("setHandlerQueue:"), _queue);
            }

            _motion = motion;
            MacObjC.SendVoid(motion, MacObjC.Selector("setValueChangedHandler:"), (nint)_block);
        }

        return true;
    }

    /// <summary>Stops the readings of <paramref name="controller"/>'s motion and turns its sensors off.</summary>
    public static void Detach(nint controller)
    {
        var motion = controller == 0 ? 0 : MacObjC.Send(controller, MacObjC.Selector("motion"));
        lock (Gate)
        {
            if (motion != 0)
            {
                MacObjC.SendVoid(motion, MacObjC.Selector("setValueChangedHandler:"), 0);
                if (MacObjC.Responds(motion, "sensorsRequireManualActivation") &&
                    MacObjC.SendBool(motion, MacObjC.Selector("sensorsRequireManualActivation")))
                {
                    MacObjC.SendVoidBool(motion, MacObjC.Selector("setSensorsActive:"), false);
                }
            }

            if (_motion == motion)
            {
                _motion = 0;
            }
        }
    }

    /// <summary>
    /// The accelerometer in g, gravity included: <c>gravity + userAcceleration</c> where GameController splits them,
    /// otherwise its raw <c>acceleration</c> (the DualSense and DualShock 4 report it unsplit).
    /// </summary>
    public static bool TryReadAcceleration(nint motion, out MacVector3 acceleration)
    {
        acceleration = default;
        if (!MacObjC.Responds(motion, "hasGravityAndUserAcceleration") ||
            MacObjC.SendBool(motion, MacObjC.Selector("hasGravityAndUserAcceleration")))
        {
            var gravity = MacObjC.SendVector3(motion, MacObjC.Selector("gravity"));
            var user = MacObjC.SendVector3(motion, MacObjC.Selector("userAcceleration"));
            acceleration.X = gravity.X + user.X;
            acceleration.Y = gravity.Y + user.Y;
            acceleration.Z = gravity.Z + user.Z;
            return true;
        }

        if (MacObjC.Responds(motion, "acceleration"))
        {
            acceleration = MacObjC.SendVector3(motion, MacObjC.Selector("acceleration"));
            return true;
        }

        return false;
    }

    public static bool HasRotationRate(nint motion) =>
        !MacObjC.Responds(motion, "hasRotationRate") ||
        MacObjC.SendBool(motion, MacObjC.Selector("hasRotationRate"));

    /// <summary>The host's monotonic clock in nanoseconds, never 0 (the orientation's "no reading yet").</summary>
    internal static ulong TimestampNanoseconds(long ticks, long frequency) =>
        Math.Max(1UL, (ulong)((Int128)ticks * 1_000_000_000 / frequency));

    private static bool EnsureBlock()
    {
        if (_block is not null)
        {
            return true;
        }

        if (!NativeLibrary.TryLoad("/usr/lib/libSystem.B.dylib", out var system) ||
            !NativeLibrary.TryGetExport(system, "_NSConcreteGlobalBlock", out var globalBlockClass))
        {
            Console.Error.WriteLine("[PAD][WARN] GameController motion: no block runtime; orientation is not fed.");
            return false;
        }

        if (_queue == 0)
        {
            var label = "net.sharpemu.gamepad-motion\0"u8;
            fixed (byte* name = label)
            {
                _queue = dispatch_queue_create(name, 0);
            }

            if (_queue == 0)
            {
                Console.Error.WriteLine("[PAD][WARN] GameController motion: no dispatch queue; orientation is not fed.");
                return false;
            }
        }

        // A global block captures nothing and is never copied or freed, so it lives in native memory for good.
        var descriptor = (BlockDescriptor*)NativeMemory.AllocZeroed((nuint)sizeof(BlockDescriptor));
        descriptor->Size = (nuint)sizeof(BlockLiteral);
        var block = (BlockLiteral*)NativeMemory.AllocZeroed((nuint)sizeof(BlockLiteral));
        block->Isa = globalBlockClass;
        block->Flags = BlockIsGlobal;
        block->Invoke = &OnMotion;
        block->Descriptor = descriptor;
        _block = block;
        return true;
    }

    [UnmanagedCallersOnly]
    private static void OnMotion(nint block, nint motion)
    {
        try
        {
            if (motion == 0 || motion != Volatile.Read(ref _motion))
            {
                return;
            }

            var timestamp = TimestampNanoseconds(Stopwatch.GetTimestamp(), Stopwatch.Frequency);
            if (TryReadAcceleration(motion, out var acceleration))
            {
                HostWindowInput.ApplyGamepadMotion(
                    angularVelocity: false,
                    (float)(acceleration.X * StandardGravity),
                    (float)(acceleration.Y * StandardGravity),
                    (float)(acceleration.Z * StandardGravity),
                    timestamp);
            }

            if (HasRotationRate(motion))
            {
                var rate = MacObjC.SendVector3(motion, MacObjC.Selector("rotationRate"));
                HostWindowInput.ApplyGamepadMotion(
                    angularVelocity: true,
                    (float)rate.X,
                    (float)rate.Y,
                    (float)rate.Z,
                    timestamp);
            }
        }
        catch (Exception exception)
        {
            // Never let an exception unwind into GameController's queue.
            Console.Error.WriteLine($"[PAD][WARN] GameController motion reading failed: {exception.Message}");
        }
    }

    [DllImport("/usr/lib/libSystem.B.dylib")]
    private static extern nint dispatch_queue_create(byte* label, nint attributes);
}
