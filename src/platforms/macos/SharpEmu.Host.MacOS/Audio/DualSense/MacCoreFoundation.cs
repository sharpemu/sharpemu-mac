// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;

namespace SharpEmu.Host.MacOS;

/// <summary>The CoreFoundation calls the controller audio needs to read CoreAudio and IOKit properties.</summary>
[SupportedOSPlatform("macos")]
internal static unsafe class MacCoreFoundation
{
    private const string Library = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation";
    private const uint StringEncodingUtf8 = 0x0800_0100;
    private const int NumberSInt32Type = 3;

    [DllImport(Library)]
    internal static extern void CFRelease(nint value);

    [DllImport(Library)]
    internal static extern nint CFRetain(nint value);

    [DllImport(Library)]
    private static extern nuint CFGetTypeID(nint value);

    [DllImport(Library)]
    private static extern nuint CFStringGetTypeID();

    [DllImport(Library)]
    private static extern nuint CFNumberGetTypeID();

    [DllImport(Library)]
    private static extern nint CFStringGetLength(nint value);

    [DllImport(Library)]
    private static extern nint CFStringGetMaximumSizeForEncoding(nint length, uint encoding);

    [DllImport(Library)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern bool CFStringGetCString(nint value, byte* buffer, nint bufferSize, uint encoding);

    [DllImport(Library)]
    private static extern nint CFStringCreateWithCString(nint allocator, byte* text, uint encoding);

    [DllImport(Library)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern bool CFNumberGetValue(nint number, int type, int* value);

    [DllImport(Library)]
    internal static extern nint CFSetGetCount(nint set);

    [DllImport(Library)]
    internal static extern void CFSetGetValues(nint set, nint* values);

    /// <summary>A CFString's text, or null for anything that is not a CFString.</summary>
    internal static string? ReadString(nint value)
    {
        if (value == 0 || CFGetTypeID(value) != CFStringGetTypeID())
        {
            return null;
        }

        var size = CFStringGetMaximumSizeForEncoding(CFStringGetLength(value), StringEncodingUtf8) + 1;
        if (size <= 0 || size > 4096)
        {
            return null;
        }

        var buffer = stackalloc byte[(int)size];
        return CFStringGetCString(value, buffer, size, StringEncodingUtf8)
            ? Marshal.PtrToStringUTF8((nint)buffer)
            : null;
    }

    /// <summary>A CFNumber's value as a 32-bit integer.</summary>
    internal static bool TryReadInt32(nint value, out int result)
    {
        result = 0;
        if (value == 0 || CFGetTypeID(value) != CFNumberGetTypeID())
        {
            return false;
        }

        int number;
        if (!CFNumberGetValue(value, NumberSInt32Type, &number))
        {
            return false;
        }

        result = number;
        return true;
    }

    /// <summary>A new CFString (released by the caller).</summary>
    internal static nint CreateString(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text + "\0");
        fixed (byte* data = bytes)
        {
            return CFStringCreateWithCString(0, data, StringEncodingUtf8);
        }
    }
}
