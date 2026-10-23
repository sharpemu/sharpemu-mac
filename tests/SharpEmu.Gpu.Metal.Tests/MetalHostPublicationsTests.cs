// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalHostPublicationsTests
{
    [Fact]
    public void PublicationRangesStayPendingUntilTheirCallbackFinishes()
    {
        var publications = new MetalHostPublications();
        var first = publications.Add(0x1000, 8, 7);
        var second = publications.Add(0x1004, 8, 9);
        Assert.Equal(0ul, publications.Pending(0x100C, 4));
        Assert.Equal(7ul, publications.Pending(0xFFF, 2));
        Assert.Equal(9ul, publications.Pending(0x1004, 4));
        // A GPU event reaching nine is deliberately irrelevant to these records.
        first.Complete();
        Assert.Equal(0ul, publications.Pending(0x1000, 4));
        Assert.Equal(9ul, publications.Pending(0x1004, 4));
        second.Complete();
        Assert.Equal(0ul, publications.Pending(0x1004, 4));
        publications.Add(ulong.MaxValue - 7, 8, 11);
        Assert.Equal(11ul, publications.Pending(ulong.MaxValue - 3, 4));
        Assert.Equal(0ul, publications.Pending(0, 4));
    }
}
