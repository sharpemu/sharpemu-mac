// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using System.Runtime.Versioning;

using SharpEmu.HLE.Host;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// The slice of CoreAudio / AudioUnit this backend needs: one default-output
/// AudioUnit per stream, fed from a render callback. Declared here so the audio
/// backend itself stays readable and so a later move into a dedicated
/// SharpEmu.Host.MacOS assembly is a file move.
/// </summary>
[SupportedOSPlatform("macos")]
internal static unsafe class MacCoreAudioInterop
{
    private const string AudioToolbox =
        "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox";

    // Four-character codes from AUComponent.h / CoreAudioBaseTypes.h.
    internal const uint AudioUnitTypeOutput = 0x6175_6F75;           // 'auou'
    internal const uint AudioUnitSubTypeDefaultOutput = 0x6465_6620; // 'def '
    internal const uint AudioUnitSubTypeHalOutput = 0x6168_616C;     // 'ahal'
    internal const uint AudioUnitManufacturerApple = 0x6170_706C;    // 'appl'
    internal const uint AudioFormatLinearPcm = 0x6C70_636D;          // 'lpcm'

    internal const uint FormatFlagIsFloat = 1u << 0;
    internal const uint FormatFlagIsPacked = 1u << 3;

    internal const uint ScopeGlobal = 0;
    internal const uint ScopeInput = 1;
    internal const uint ScopeOutput = 2;

    internal const uint PropertyStreamFormat = 8;
    internal const uint PropertyMaximumFramesPerSlice = 14;
    internal const uint PropertyAudioChannelLayout = 19;
    internal const uint PropertySetRenderCallback = 23;

    /// <summary>kAudioOutputUnitProperty_CurrentDevice (AudioUnitProperties.h).</summary>
    internal const uint PropertyCurrentDevice = 2000;

    // AudioHardware.h / AudioHardwareBase.h.
    private const string CoreAudio = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio";
    internal const uint AudioObjectSystemObject = 1;
    internal const uint AudioObjectScopeGlobal = 0x676C_6F62;         // 'glob'
    internal const uint AudioObjectScopeOutput = 0x6F75_7470;         // 'outp'
    internal const uint AudioObjectElementMain = 0;
    internal const uint AudioHardwarePropertyDevices = 0x6465_7623;   // 'dev#'
    internal const uint AudioObjectPropertyName = 0x6C6E_616D;        // 'lnam'
    internal const uint AudioDevicePropertyStreamConfiguration = 0x736C_6179; // 'slay'

    [StructLayout(LayoutKind.Sequential)]
    internal struct AudioObjectPropertyAddress
    {
        public uint Selector;
        public uint Scope;
        public uint Element;
    }

    [DllImport(CoreAudio)]
    internal static extern int AudioObjectGetPropertyDataSize(
        uint objectId,
        AudioObjectPropertyAddress* address,
        uint qualifierDataSize,
        void* qualifierData,
        uint* dataSize);

    [DllImport(CoreAudio)]
    internal static extern int AudioObjectGetPropertyData(
        uint objectId,
        AudioObjectPropertyAddress* address,
        uint qualifierDataSize,
        void* qualifierData,
        uint* dataSize,
        void* data);

    /// <summary>Render-callback action flag telling the unit the buffer is silent.</summary>
    internal const uint RenderActionOutputIsSilence = 1u << 4;

    /// <summary>The layout is described by the trailing channel descriptions.</summary>
    internal const uint ChannelLayoutTagUseChannelDescriptions = 0;

    [StructLayout(LayoutKind.Sequential)]
    internal struct AudioComponentDescription
    {
        public uint ComponentType;
        public uint ComponentSubType;
        public uint ComponentManufacturer;
        public uint ComponentFlags;
        public uint ComponentFlagsMask;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct AudioStreamBasicDescription
    {
        public double SampleRate;
        public uint FormatId;
        public uint FormatFlags;
        public uint BytesPerPacket;
        public uint FramesPerPacket;
        public uint BytesPerFrame;
        public uint ChannelsPerFrame;
        public uint BitsPerChannel;
        public uint Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct AudioBuffer
    {
        public uint NumberChannels;
        public uint DataByteSize;
        public void* Data;
    }

    /// <summary>
    /// AudioBufferList's variable-length tail: <see cref="Buffers"/> is the first
    /// element of an array of <see cref="NumberBuffers"/> buffers. An interleaved
    /// client format always gets exactly one.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct AudioBufferList
    {
        public uint NumberBuffers;
        public AudioBuffer Buffers;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct AudioChannelDescription
    {
        public uint ChannelLabel;
        public uint ChannelFlags;
        public float CoordinateX;
        public float CoordinateY;
        public float CoordinateZ;
    }

    /// <summary>
    /// AudioChannelLayout's fixed head; the descriptions follow it in memory, so
    /// a layout is built as a byte blob of this plus N descriptions.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct AudioChannelLayoutHeader
    {
        public uint ChannelLayoutTag;
        public uint ChannelBitmap;
        public uint NumberChannelDescriptions;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct AURenderCallbackStruct
    {
        public delegate* unmanaged<nint, uint*, nint, uint, uint, AudioBufferList*, int> InputProc;
        public nint InputProcRefCon;
    }

    [DllImport(AudioToolbox)]
    internal static extern nint AudioComponentFindNext(
        nint component,
        AudioComponentDescription* description);

    [DllImport(AudioToolbox)]
    internal static extern int AudioComponentInstanceNew(nint component, out nint instance);

    [DllImport(AudioToolbox)]
    internal static extern int AudioComponentInstanceDispose(nint instance);

    [DllImport(AudioToolbox)]
    internal static extern int AudioUnitInitialize(nint unit);

    [DllImport(AudioToolbox)]
    internal static extern int AudioUnitUninitialize(nint unit);

    [DllImport(AudioToolbox)]
    internal static extern int AudioUnitGetProperty(
        nint unit,
        uint propertyId,
        uint scope,
        uint element,
        void* data,
        uint* dataSize);

    [DllImport(AudioToolbox)]
    internal static extern int AudioUnitSetProperty(
        nint unit,
        uint propertyId,
        uint scope,
        uint element,
        void* data,
        uint dataSize);

    [DllImport(AudioToolbox)]
    internal static extern int AudioOutputUnitStart(nint unit);

    [DllImport(AudioToolbox)]
    internal static extern int AudioOutputUnitStop(nint unit);
}
