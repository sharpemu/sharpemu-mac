// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Versioning;
using static SharpEmu.Host.MacOS.MacCoreAudioInterop;

namespace SharpEmu.Host.MacOS;

/// <summary>One CoreAudio device with its name and output channel count.</summary>
internal readonly record struct MacAudioDeviceInfo(uint Id, string Name, int OutputChannels);

/// <summary>The CoreAudio device list, for finding the DualSense's own output device.</summary>
[SupportedOSPlatform("macos")]
internal static unsafe class MacCoreAudioDevices
{
    /// <summary>The channel count of a wired DualSense's output: speaker pair, then the two actuators.</summary>
    internal const int ControllerChannels = 4;

    /// <summary>Every device that has output channels.</summary>
    public static List<MacAudioDeviceInfo> ListOutputDevices()
    {
        var devices = new List<MacAudioDeviceInfo>();
        var address = new AudioObjectPropertyAddress
        {
            Selector = AudioHardwarePropertyDevices,
            Scope = AudioObjectScopeGlobal,
            Element = AudioObjectElementMain,
        };
        uint size = 0;
        if (AudioObjectGetPropertyDataSize(AudioObjectSystemObject, &address, 0, null, &size) != 0 || size == 0)
        {
            return devices;
        }

        var ids = new uint[size / sizeof(uint)];
        fixed (uint* data = ids)
        {
            if (AudioObjectGetPropertyData(AudioObjectSystemObject, &address, 0, null, &size, data) != 0)
            {
                return devices;
            }
        }

        foreach (var id in ids.AsSpan(0, (int)(size / sizeof(uint))))
        {
            var channels = OutputChannels(id);
            if (channels > 0)
            {
                devices.Add(new MacAudioDeviceInfo(id, Name(id) ?? string.Empty, channels));
            }
        }

        return devices;
    }

    /// <summary>
    /// The controller's output device: exactly one device with four output channels named for a DualSense
    /// ("DualSense", or "Wireless Controller", the name the pad reports over USB). CoreAudio ties no device to a
    /// gamepad, so two candidates give none rather than a guess, as upstream's SDL host does.
    /// </summary>
    internal static bool TrySelectControllerDevice(IReadOnlyList<MacAudioDeviceInfo> devices, out MacAudioDeviceInfo device)
    {
        device = default;
        var found = 0;
        foreach (var candidate in devices)
        {
            if (candidate.OutputChannels == ControllerChannels &&
                (candidate.Name.Contains("DualSense", StringComparison.OrdinalIgnoreCase) ||
                 candidate.Name.Contains("Wireless Controller", StringComparison.OrdinalIgnoreCase)))
            {
                device = candidate;
                found++;
            }
        }

        return found == 1;
    }

    private static string? Name(uint device)
    {
        var address = new AudioObjectPropertyAddress
        {
            Selector = AudioObjectPropertyName,
            Scope = AudioObjectScopeGlobal,
            Element = AudioObjectElementMain,
        };
        nint name = 0;
        var size = (uint)sizeof(nint);
        if (AudioObjectGetPropertyData(device, &address, 0, null, &size, &name) != 0 || name == 0)
        {
            return null;
        }

        try
        {
            return MacCoreFoundation.ReadString(name);
        }
        finally
        {
            MacCoreFoundation.CFRelease(name);
        }
    }

    private static int OutputChannels(uint device)
    {
        var address = new AudioObjectPropertyAddress
        {
            Selector = AudioDevicePropertyStreamConfiguration,
            Scope = AudioObjectScopeOutput,
            Element = AudioObjectElementMain,
        };
        uint size = 0;
        if (AudioObjectGetPropertyDataSize(device, &address, 0, null, &size) != 0 ||
            size < sizeof(uint))
        {
            return 0;
        }

        var blob = new byte[size];
        fixed (byte* data = blob)
        {
            if (AudioObjectGetPropertyData(device, &address, 0, null, &size, data) != 0)
            {
                return 0;
            }

            var list = (AudioBufferList*)data;
            var buffers = &list->Buffers;
            var channels = 0;
            for (var index = 0u; index < list->NumberBuffers; index++)
            {
                if ((byte*)(buffers + index + 1) > data + size)
                {
                    break;
                }

                channels += (int)buffers[index].NumberChannels;
            }

            return channels;
        }
    }
}
