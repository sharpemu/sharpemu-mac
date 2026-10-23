// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

/// <summary>
/// A host thread that runs at a QoS class where the platform lets it: natively created
/// through <see cref="HostQosThread"/> when it can be, otherwise a managed thread that
/// asks for the class after it starts (which Darwin refuses on CoreCLR-created threads,
/// so it is only the fallback). SHARPEMU_NATIVE_QOS_THREADS=0 forces the managed thread.
/// </summary>
public sealed class HostBackgroundThread
{
    private static readonly bool NativeEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_NATIVE_QOS_THREADS"), "0", StringComparison.Ordinal);

    private readonly HostQosThread? _native;
    private readonly Thread? _managed;

    private HostBackgroundThread(HostQosThread? native, Thread? managed)
    {
        _native = native;
        _managed = managed;
    }

    /// <summary>Whether the thread was created with its class in its attributes.</summary>
    public bool IsNative => _native is not null;

    public bool Join(int milliseconds) => _native is not null ? _native.Join(milliseconds) : _managed!.Join(milliseconds);

    /// <summary>
    /// Starts <paramref name="body"/> on a new thread at <paramref name="qos"/>. The body is
    /// told whether the class is in effect for it. <paramref name="background"/> only
    /// applies to the managed fallback; a native thread never keeps the process alive.
    /// </summary>
    public static HostBackgroundThread Start(string name, HostThreadQos qos, int stackBytes, Action<bool> body, bool background = true)
    {
        if (NativeEnabled && OperatingSystem.IsMacOS())
        {
            var native = HostQosThread.TryStart(name, qos, stackBytes, () =>
            {
                GuestThreadQosPolicy.LogNativeThreadStart(name, qos);
                body(true);
            }, out var errno);
            if (native is not null)
            {
                return new HostBackgroundThread(native, null);
            }

            Console.Error.WriteLine(
                $"[SCHED][QOS] thread={name}: native QoS thread create failed (errno={errno}); using a managed thread.");
        }

        var managed = new Thread(() => body(HostThreadQualityOfService.TryApplyToCurrentThread(qos)), stackBytes)
        {
            IsBackground = background,
            Name = name,
        };
        managed.Start();
        return new HostBackgroundThread(null, managed);
    }
}
