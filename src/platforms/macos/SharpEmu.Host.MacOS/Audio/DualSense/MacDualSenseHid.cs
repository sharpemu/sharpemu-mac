// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace SharpEmu.Host.MacOS;

internal enum MacDualSenseTransport
{
    Usb,
    Bluetooth,
}

/// <summary>
/// A DualSense's HID interface through IOKit, for the output reports GameController has no API for: the audio routing
/// in the effects report and, over Bluetooth, the audio report that carries the actuator samples and the speaker's
/// Opus packets. The device is opened shared (GameController keeps reading it), and only one pad per transport is
/// used, since nothing ties a HID device to the CoreAudio device or GameController pad of the same controller.
/// </summary>
[SupportedOSPlatform("macos")]
internal sealed unsafe class MacDualSenseHid : IDisposable
{
    internal const int SonyVendorId = 0x054C;
    internal const int DualSenseProductId = 0x0CE6;
    internal const int DualSenseEdgeProductId = 0x0DF2;

    private const string IoKit = "/System/Library/Frameworks/IOKit.framework/IOKit";
    private const uint ReportTypeOutput = 1;
    private const uint OptionsNone = 0;

    private nint _manager;
    private nint _device;

    private MacDualSenseHid(nint manager, nint device, MacDualSenseTransport transport)
    {
        _manager = manager;
        _device = device;
        Transport = transport;
    }

    public MacDualSenseTransport Transport { get; }

    /// <summary>How many DualSense pads each transport has, and the one to open when a transport has exactly one.</summary>
    internal readonly record struct Candidate(int ProductId, string? Transport, int UsagePage, int Usage);

    /// <summary>
    /// The transport a HID device of a DualSense is on, or null when it is not a DualSense gamepad interface. Pure:
    /// unit tested.
    /// </summary>
    internal static MacDualSenseTransport? Classify(int vendorId, in Candidate candidate)
    {
        if (vendorId != SonyVendorId ||
            candidate.ProductId is not (DualSenseProductId or DualSenseEdgeProductId) ||
            (candidate.UsagePage != 0 && (candidate.UsagePage != 1 || candidate.Usage != 5)))
        {
            return null;
        }

        if (string.Equals(candidate.Transport, "USB", StringComparison.OrdinalIgnoreCase))
        {
            return MacDualSenseTransport.Usb;
        }

        return candidate.Transport is not null &&
               candidate.Transport.StartsWith("Bluetooth", StringComparison.OrdinalIgnoreCase)
            ? MacDualSenseTransport.Bluetooth
            : null;
    }

    /// <summary>The one DualSense on <paramref name="transport"/>, opened; null when there is none or more than one.</summary>
    public static MacDualSenseHid? TryOpenSingle(MacDualSenseTransport transport)
    {
        var manager = IOHIDManagerCreate(0, OptionsNone);
        if (manager == 0)
        {
            return null;
        }

        nint chosen = 0;
        var found = 0;
        var set = (nint)0;
        try
        {
            // No matching dictionary: every HID device, filtered below by its properties.
            IOHIDManagerSetDeviceMatching(manager, 0);
            set = IOHIDManagerCopyDevices(manager);
            var count = set == 0 ? 0 : (int)MacCoreFoundation.CFSetGetCount(set);
            if (count > 0)
            {
                var devices = new nint[count];
                fixed (nint* values = devices)
                {
                    MacCoreFoundation.CFSetGetValues(set, values);
                }

                foreach (var device in devices)
                {
                    if (Classify(IntProperty(device, "VendorID"), ReadCandidate(device)) == transport)
                    {
                        chosen = device;
                        found++;
                    }
                }
            }

            if (found != 1)
            {
                return null;
            }

            MacCoreFoundation.CFRetain(chosen);
            var status = IOHIDDeviceOpen(chosen, OptionsNone);
            if (status != 0)
            {
                MacCoreFoundation.CFRelease(chosen);
                Console.Error.WriteLine(
                    $"[LOADER][WARN] DualSense ({transport}) HID interface refused to open: IOReturn 0x{status:X8}");
                return null;
            }

            var pad = new MacDualSenseHid(manager, chosen, transport);
            manager = 0;
            return pad;
        }
        finally
        {
            if (set != 0)
            {
                MacCoreFoundation.CFRelease(set);
            }

            if (manager != 0)
            {
                MacCoreFoundation.CFRelease(manager);
            }
        }
    }

    /// <summary>Sends one output report; its first byte is the report ID, as on the wire.</summary>
    public bool TrySendOutputReport(ReadOnlySpan<byte> report, out int status)
    {
        status = -1;
        if (_device == 0 || report.IsEmpty)
        {
            return false;
        }

        fixed (byte* data = report)
        {
            status = IOHIDDeviceSetReport(_device, ReportTypeOutput, report[0], data, report.Length);
        }

        return status == 0;
    }

    /// <summary>
    /// The pad's effects report (the same 47-byte state SDL_SendGamepadEffect takes), wrapped for this transport.
    /// </summary>
    public bool TrySendEffects(ReadOnlySpan<byte> effects)
    {
        Span<byte> report = stackalloc byte[MacDualSenseReports.BluetoothEffectsReportLength];
        var length = Transport == MacDualSenseTransport.Usb
            ? MacDualSenseReports.BuildUsbEffectsReport(effects, report)
            : MacDualSenseReports.BuildBluetoothEffectsReport(effects, report);
        return TrySendOutputReport(report[..length], out _);
    }

    public void Dispose()
    {
        if (_device != 0)
        {
            _ = IOHIDDeviceClose(_device, OptionsNone);
            MacCoreFoundation.CFRelease(_device);
            _device = 0;
        }

        if (_manager != 0)
        {
            MacCoreFoundation.CFRelease(_manager);
            _manager = 0;
        }
    }

    private static Candidate ReadCandidate(nint device) => new(
        IntProperty(device, "ProductID"),
        StringProperty(device, "Transport"),
        IntProperty(device, "PrimaryUsagePage"),
        IntProperty(device, "PrimaryUsage"));

    private static int IntProperty(nint device, string key)
    {
        var name = MacCoreFoundation.CreateString(key);
        try
        {
            return MacCoreFoundation.TryReadInt32(IOHIDDeviceGetProperty(device, name), out var value) ? value : 0;
        }
        finally
        {
            MacCoreFoundation.CFRelease(name);
        }
    }

    private static string? StringProperty(nint device, string key)
    {
        var name = MacCoreFoundation.CreateString(key);
        try
        {
            return MacCoreFoundation.ReadString(IOHIDDeviceGetProperty(device, name));
        }
        finally
        {
            MacCoreFoundation.CFRelease(name);
        }
    }

    [DllImport(IoKit)]
    private static extern nint IOHIDManagerCreate(nint allocator, uint options);

    [DllImport(IoKit)]
    private static extern void IOHIDManagerSetDeviceMatching(nint manager, nint matching);

    [DllImport(IoKit)]
    private static extern nint IOHIDManagerCopyDevices(nint manager);

    [DllImport(IoKit)]
    private static extern nint IOHIDDeviceGetProperty(nint device, nint key);

    [DllImport(IoKit)]
    private static extern int IOHIDDeviceOpen(nint device, uint options);

    [DllImport(IoKit)]
    private static extern int IOHIDDeviceClose(nint device, uint options);

    [DllImport(IoKit)]
    private static extern int IOHIDDeviceSetReport(nint device, uint reportType, nint reportId, byte* report, nint length);
}
