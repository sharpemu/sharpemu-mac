// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// A shader or pipeline compile thread started at a QoS class of its own. Thread.Priority does
/// nothing on this host (CoreCLR's Unix PAL ignores it, HostThreadQualityOfService), so the
/// BelowNormal the background compiles asked for never applied, and a thread made with
/// new Thread from the GPU worker ran next to the worker and the guest's threads on the
/// performance cores. The class is also what the driver's compiler service runs a request at:
/// it travels with the XPC message. A thread made natively with the class in its attributes
/// is the one route that sets it (HostQosThread), and only such a thread may change its own
/// class later (<see cref="ApplyToCurrent"/>).
/// What the frame or the guest waits for (the runtime's library and pipeline compiles, a
/// prewarm the guest is held for, the libraries a scene takes from the prewarm) runs at
/// UserInitiated, below the worker's UserInteractive; what nobody waits for (the static
/// precompile, the computes left once the guest runs, the archive and list writers) at Utility.
/// SHARPEMU_METAL_V2_COMPILE_QOS=0 makes them managed threads again.
/// </summary>
internal sealed class MetalCompileThread
{
    internal static readonly bool QosEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COMPILE_QOS") != "0";

    // A compile thread translates too (prewarm, precompile), which recurses as deep as on the worker.
    private const int StackBytes = 16 * 1024 * 1024;

    private static readonly HashSet<string> _reported = new(StringComparer.Ordinal);

    private readonly string _name;
    private readonly HostThreadQos _qos;
    private readonly Action _body;
    private HostQosThread? _native;
    private Thread? _managed;

    private MetalCompileThread(string name, HostThreadQos qos, Action body)
    {
        _name = name;
        _qos = qos;
        _body = body;
    }

    /// <summary>A thread that runs <paramref name="body"/> at <paramref name="qos"/> once started.</summary>
    public static MetalCompileThread Create(string name, HostThreadQos qos, Action body) => new(name, qos, body);

    /// <summary>Creates and starts it.</summary>
    public static MetalCompileThread Start(string name, HostThreadQos qos, Action body)
    {
        var thread = Create(name, qos, body);
        thread.Start();
        return thread;
    }

    public bool IsCurrentThread =>
        _native?.IsCurrentThread ?? ReferenceEquals(_managed, Thread.CurrentThread);

    public void Start()
    {
        if (QosEnabled)
        {
            _native = HostQosThread.TryStart(_name, _qos, StackBytes, Run, out _);
            if (_native is not null)
            {
                return;
            }
        }

        _managed = new Thread(Run) { IsBackground = true, Name = _name };
        _managed.Start();
    }

    public bool Join(TimeSpan timeout) =>
        _native?.Join((int)timeout.TotalMilliseconds) ?? _managed?.Join(timeout) ?? true;

    public void Join()
    {
        if (_native is not null)
        {
            _native.Join(Timeout.Infinite);
        }
        else
        {
            _managed?.Join();
        }
    }

    /// <summary>Moves the calling compile thread to <paramref name="qos"/> (a prewarm thread whose
    /// guest runs again): true when it applied; refused on a thread CoreCLR made, which keeps its class.</summary>
    public static bool ApplyToCurrent(HostThreadQos qos) =>
        QosEnabled && HostThreadQualityOfService.TryApplyToCurrentThread(qos);

    private void Run()
    {
        ReportClass();
        _body();
    }

    // The class each kind of compile thread runs at, once per kind (the name without its index).
    private void ReportClass()
    {
        var kind = _name.TrimEnd(' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
        lock (_reported)
        {
            if (!_reported.Add(kind))
            {
                return;
            }
        }

        Console.Error.WriteLine(
            $"[METAL-V2][QOS] thread=\"{kind}\" requested={(_native is null && _managed is not null ? "inherited" : _qos.ToString())} " +
            $"class=0x{HostThreadQualityOfService.GetCurrentThreadClassForDiagnostics():X2}");
    }
}
