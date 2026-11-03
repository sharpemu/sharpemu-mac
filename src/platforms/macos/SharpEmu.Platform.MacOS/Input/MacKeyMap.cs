// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Platform.MacOS.Input;

/// <summary>
/// Virtual key codes from Carbon's <c>Events.h</c> (<c>kVK_*</c>), the values
/// AppKit reports in <c>-[NSEvent keyCode]</c>. They are positional ANSI codes,
/// independent of the active keyboard layout.
/// </summary>
internal static class MacVirtualKeyCode
{
    public const ushort A = 0x00;
    public const ushort S = 0x01;
    public const ushort D = 0x02;
    public const ushort F = 0x03;
    public const ushort H = 0x04;
    public const ushort G = 0x05;
    public const ushort Z = 0x06;
    public const ushort X = 0x07;
    public const ushort C = 0x08;
    public const ushort V = 0x09;
    public const ushort B = 0x0B;
    public const ushort Q = 0x0C;
    public const ushort W = 0x0D;
    public const ushort E = 0x0E;
    public const ushort R = 0x0F;
    public const ushort Y = 0x10;
    public const ushort T = 0x11;
    public const ushort O = 0x1F;
    public const ushort U = 0x20;
    public const ushort I = 0x22;
    public const ushort P = 0x23;
    public const ushort Return = 0x24;
    public const ushort L = 0x25;
    public const ushort J = 0x26;
    public const ushort K = 0x28;
    public const ushort N = 0x2D;
    public const ushort M = 0x2E;
    public const ushort Tab = 0x30;
    public const ushort Delete = 0x33;
    public const ushort Escape = 0x35;
    public const ushort F11 = 0x67;
    public const ushort F3 = 0x63;
    public const ushort F4 = 0x76;
    public const ushort Digit1 = 0x12;
    public const ushort Digit2 = 0x13;
    public const ushort Digit3 = 0x14;
    public const ushort Digit4 = 0x15;
    public const ushort F2 = 0x78;
    public const ushort F1 = 0x7A;
    public const ushort LeftArrow = 0x7B;
    public const ushort RightArrow = 0x7C;
    public const ushort DownArrow = 0x7D;
    public const ushort UpArrow = 0x7E;
}

/// <summary>
/// macOS keycode to Win32-style virtual key translation for the native host
/// window. The guest-facing key set is defined by what
/// <c>HostWindowInput.SetKey</c> consumers expect, which the SDL host window
/// produced from SDL keycodes; this table reproduces exactly that set (A-Z plus
/// backspace, tab, return, escape and the four arrows) so key behaviour is
/// identical on both windowing backends. Pure logic, no AppKit — unit tested.
/// </summary>
internal static class MacKeyMap
{
    /// <summary>
    /// Maps an <c>-[NSEvent keyCode]</c> to the Win32 virtual key code the host
    /// input seam uses, or returns false for keys the guest never sees.
    /// </summary>
    public static bool TryMapVirtualKey(ushort keyCode, out int virtualKey)
    {
        virtualKey = keyCode switch
        {
            MacVirtualKeyCode.Delete => 0x08,     // VK_BACK
            MacVirtualKeyCode.Tab => 0x09,        // VK_TAB
            MacVirtualKeyCode.Return => 0x0D,     // VK_RETURN
            MacVirtualKeyCode.Escape => 0x1B,     // VK_ESCAPE
            MacVirtualKeyCode.LeftArrow => 0x25,  // VK_LEFT
            MacVirtualKeyCode.UpArrow => 0x26,    // VK_UP
            MacVirtualKeyCode.RightArrow => 0x27, // VK_RIGHT
            MacVirtualKeyCode.DownArrow => 0x28,  // VK_DOWN
            MacVirtualKeyCode.A => 0x41,
            MacVirtualKeyCode.B => 0x42,
            MacVirtualKeyCode.C => 0x43,
            MacVirtualKeyCode.D => 0x44,
            MacVirtualKeyCode.E => 0x45,
            MacVirtualKeyCode.F => 0x46,
            MacVirtualKeyCode.G => 0x47,
            MacVirtualKeyCode.H => 0x48,
            MacVirtualKeyCode.I => 0x49,
            MacVirtualKeyCode.J => 0x4A,
            MacVirtualKeyCode.K => 0x4B,
            MacVirtualKeyCode.L => 0x4C,
            MacVirtualKeyCode.M => 0x4D,
            MacVirtualKeyCode.N => 0x4E,
            MacVirtualKeyCode.O => 0x4F,
            MacVirtualKeyCode.P => 0x50,
            MacVirtualKeyCode.Q => 0x51,
            MacVirtualKeyCode.R => 0x52,
            MacVirtualKeyCode.S => 0x53,
            MacVirtualKeyCode.T => 0x54,
            MacVirtualKeyCode.U => 0x55,
            MacVirtualKeyCode.V => 0x56,
            MacVirtualKeyCode.W => 0x57,
            MacVirtualKeyCode.X => 0x58,
            MacVirtualKeyCode.Y => 0x59,
            MacVirtualKeyCode.Z => 0x5A,
            _ => 0,
        };
        return virtualKey != 0;
    }
}
