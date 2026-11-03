// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalResidencySetPolicyTests
{
    // Runs only when no override is set, which is how the suite runs.
    [Fact]
    public void DemonsSoulsNamesItsWindowsPerEncoderAndEveryOtherTitleUsesTheSet()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_RESIDENCY_SET") is not null)
        {
            return;
        }

        Assert.False(MetalGuestMemory.UseResidencySetFor("PPSA01342"));
        Assert.True(MetalGuestMemory.UseResidencySetFor("PPSA01411"));
        Assert.True(MetalGuestMemory.UseResidencySetFor(null));
        Assert.True(MetalGuestMemory.UseResidencySetFor(string.Empty));
    }
}
