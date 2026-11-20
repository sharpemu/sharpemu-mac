// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// A wave that called a function no link resolved writes a fault record (Gen5MslArgumentLayout.CallFaultWords). The host
/// reads it once per flush: under SHARPEMU_STRICT_COMPUTE (on unless 0) it is fatal, as on Vulkan
/// (BdaFaultProcessor), and otherwise it is counted and logged as [UNLINKED_CALL].
/// </summary>
public sealed class UnlinkedCallFaultSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AnUnlinkedCallIsFatalUnderStrictCompute(bool strict)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.StrictCompute = strict;
        rig.Host.CallFaultsAddress();
        var words = rig.Host.CallFaultWords;
        words[1] = 0x9ABCDEF0;
        words[2] = 0x12345678;
        words[3] = 0x40;
        words[4] = 0x2000;
        words[6] = 5;
        words[0] = 2;

        if (strict)
        {
            Assert.Throws<SceneFatalException>(() => rig.Host.ReportCallFaults());
            Assert.Contains(rig.Fatals, message => message.Contains("unlinked function: hash=0x123456789ABCDEF0 pc=0x40", StringComparison.Ordinal));
            Assert.Equal(0, rig.Host.UnlinkedCalls);
        }
        else
        {
            rig.Host.ReportCallFaults();
            Assert.Equal(1, rig.Host.UnlinkedCalls);
            Assert.Empty(rig.Fatals);
        }

        Assert.Equal(0u, rig.Host.CallFaultWords[0]);
    }
}
