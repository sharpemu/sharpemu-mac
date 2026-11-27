// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Platform.MacOS.Input;
using Xunit;

namespace SharpEmu.Libs.Tests.Platform;

/// <summary>
/// The macOS keycode table has to produce exactly the virtual keys the SDL host
/// window produced, or the same guest build reacts differently to the same
/// physical key depending on the windowing backend.
/// </summary>
public sealed class MacKeyMapTests
{
    [Theory]
    [InlineData(MacVirtualKeyCode.Delete, 0x08)]
    [InlineData(MacVirtualKeyCode.Tab, 0x09)]
    [InlineData(MacVirtualKeyCode.Return, 0x0D)]
    [InlineData(MacVirtualKeyCode.Escape, 0x1B)]
    [InlineData(MacVirtualKeyCode.ForwardDelete, 0x2E)]
    [InlineData(MacVirtualKeyCode.LeftArrow, 0x25)]
    [InlineData(MacVirtualKeyCode.UpArrow, 0x26)]
    [InlineData(MacVirtualKeyCode.RightArrow, 0x27)]
    [InlineData(MacVirtualKeyCode.DownArrow, 0x28)]
    public void ControlKeysMapToTheirWin32VirtualKeys(ushort keyCode, int expected)
    {
        Assert.True(MacKeyMap.TryMapVirtualKey(keyCode, out var virtualKey));
        Assert.Equal(expected, virtualKey);
    }

    [Theory]
    [InlineData(MacVirtualKeyCode.A, 'A')]
    [InlineData(MacVirtualKeyCode.B, 'B')]
    [InlineData(MacVirtualKeyCode.M, 'M')]
    [InlineData(MacVirtualKeyCode.Q, 'Q')]
    [InlineData(MacVirtualKeyCode.W, 'W')]
    [InlineData(MacVirtualKeyCode.Z, 'Z')]
    public void LettersMapToTheirAsciiVirtualKeys(ushort keyCode, char expected)
    {
        Assert.True(MacKeyMap.TryMapVirtualKey(keyCode, out var virtualKey));
        Assert.Equal(expected, virtualKey);
    }

    /// <summary>
    /// macOS letter keycodes are positional and famously non-alphabetical, so the
    /// whole alphabet is checked rather than a sample: every VK_A..VK_Z must be
    /// produced exactly once.
    /// </summary>
    [Fact]
    public void EveryLetterKeyIsMappedExactlyOnce()
    {
        var produced = new HashSet<int>();
        for (ushort keyCode = 0; keyCode <= 0x7E; keyCode++)
        {
            if (MacKeyMap.TryMapVirtualKey(keyCode, out var virtualKey) && virtualKey is >= 0x41 and <= 0x5A)
            {
                Assert.True(produced.Add(virtualKey), $"keycode 0x{keyCode:X2} duplicates VK 0x{virtualKey:X2}");
            }
        }

        Assert.Equal(26, produced.Count);
        Assert.Equal(Enumerable.Range(0x41, 26), produced.Order());
    }

    [Theory]
    [InlineData(MacVirtualKeyCode.F1)]
    [InlineData(MacVirtualKeyCode.F2)]
    [InlineData(MacVirtualKeyCode.F11)]
    [InlineData((ushort)0x38)] // kVK_Shift
    [InlineData((ushort)0x37)] // kVK_Command
    [InlineData((ushort)0x31)] // kVK_Space
    public void HotkeysAndModifiersAreNotForwardedToTheGuest(ushort keyCode)
    {
        Assert.False(MacKeyMap.TryMapVirtualKey(keyCode, out var virtualKey));
        Assert.Equal(0, virtualKey);
    }

    [Theory]
    [InlineData("a", "a")]
    [InlineData("é", "é")]
    [InlineData(" ", " ")]
    [InlineData("\r", "")]
    [InlineData("\t", "")]
    [InlineData("\u007F", "")]
    [InlineData("\u001B", "")]
    [InlineData("\uF702", "")] // NSLeftArrowFunctionKey
    [InlineData("\uF728", "")] // NSDeleteFunctionKey
    [InlineData(null, "")]
    public void OnlyPrintableCharactersAreTyped(string? characters, string expected) =>
        Assert.Equal(expected, MacKeyMap.TypedText(characters));
}
