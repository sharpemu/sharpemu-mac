// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

/// <summary>
/// A helper thread a frame-deadline thread waits on (the GPU worker's program resolver, the
/// completion thread, the split's ring, call and inbound threads): started natively at a QoS class
/// through <see cref="HostQosThread"/>, since a thread CoreCLR creates cannot take one later, and
/// as a managed background thread where that fails or is switched off.
/// </summary>
/// <remarks>
/// Left at the default class these threads queue behind a title's spinning job threads in the
/// other process while the worker waits on them. SHARPEMU_SPLIT_THREAD_QOS=0 starts them all as
/// managed threads, as before.
/// </remarks>
public sealed class HostHelperThread
{
    public static readonly bool QosEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_SPLIT_THREAD_QOS"), "0", StringComparison.Ordinal);

    private readonly HostQosThread? _native;
    private readonly Thread? _managed;

    private HostHelperThread(HostQosThread? native, Thread? managed)
    {
        _native = native;
        _managed = managed;
    }

    /// <summary>Whether the thread runs at the class it asked for.</summary>
    public bool HasQos => _native is not null;

    public bool IsCurrentThread => _native?.IsCurrentThread ?? ReferenceEquals(Thread.CurrentThread, _managed);

    /// <summary>Waits for the body to return.</summary>
    public bool Join(TimeSpan timeout) =>
        _native?.Join((int)Math.Min(timeout.TotalMilliseconds, int.MaxValue)) ?? _managed!.Join(timeout);

    public static HostHelperThread Start(string name, int stackBytes, Action body,
        HostThreadQos qos = HostThreadQos.UserInteractive)
    {
        ArgumentNullException.ThrowIfNull(body);
        if (QosEnabled && HostQosThread.TryStart(name, qos, stackBytes, body, out _) is { } native)
        {
            return new HostHelperThread(native, null);
        }

        var managed = new Thread(() => body(), stackBytes) { IsBackground = true, Name = name };
        managed.Start();
        return new HostHelperThread(null, managed);
    }
}
