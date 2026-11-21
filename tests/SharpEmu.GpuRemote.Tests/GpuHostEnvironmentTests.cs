// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.GpuRemote.Tests;

/// <summary>
/// Draw and dispatch replay live wholly in the GPU host (process B): the shared engine that seals and replays and the
/// Metal host that binds both run there, so nothing of replay crosses the channel. What does cross is the switches:
/// B reads SHARPEMU_DRAW_REPLAY and SHARPEMU_DISPATCH_REPLAY from the environment A starts it with.
/// </summary>
public sealed class GpuHostEnvironmentTests
{
    [Theory]
    [InlineData("SHARPEMU_DRAW_REPLAY")]
    [InlineData("SHARPEMU_DISPATCH_REPLAY")]
    public void TheGpuHostIsStartedWithTheReplaySwitch(string name)
    {
        var saved = Environment.GetEnvironmentVariable(name);
        try
        {
            Environment.SetEnvironmentVariable(name, "0");
            Assert.Contains($"{name}=0", GpuProcessLauncher.EnvironmentBlock());
            Environment.SetEnvironmentVariable(name, null);
            Assert.DoesNotContain(GpuProcessLauncher.EnvironmentBlock(), entry => entry.StartsWith($"{name}=", StringComparison.Ordinal));
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, saved);
        }
    }
}
