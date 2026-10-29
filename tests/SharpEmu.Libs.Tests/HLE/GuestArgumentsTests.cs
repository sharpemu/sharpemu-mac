// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

public sealed class GuestArgumentsTests
{
    [Fact]
    public void AUserSwitchReplacesTheAdapterDefaultOfTheSameName()
    {
        var merged = GuestArguments.Merge(["-width=1920", "-height=1080"], ["-height=720", "-nosound"]);
        Assert.Equal(["-width=1920", "-height=720", "-nosound"], merged);
    }

    [Fact]
    public void ArgumentsSplitOnWhitespace()
    {
        Assert.Equal(["-a=1", "-b"], GuestArguments.Split("  -a=1\t-b "));
        Assert.Empty(GuestArguments.Split(null));
    }
}
