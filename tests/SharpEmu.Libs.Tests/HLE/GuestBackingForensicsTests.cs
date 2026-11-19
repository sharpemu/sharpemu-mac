// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

public sealed unsafe class GuestBackingForensicsTests
{
    private static readonly ViewRecord[] Views =
    [
        new(0x7000_0000, 0x10000, 0x0, HostPageProtection.ReadWrite),
        // Maps backing [0x8000, 0x18000): its first half is the same bytes as the second half above.
        new(0x9000_0000, 0x10000, 0x8000, HostPageProtection.ReadWrite),
        new(0xA000_0000, 0x4000, 0x40000, HostPageProtection.ReadOnly),
    ];

    [Fact]
    public void AWindowListsTheOtherGuestRangesMappingTheSameBackingBytes()
    {
        var lines = GuestBackingForensics.DescribeAliases(Views, 0x7000_8800, 0x400, limit: 8);

        Assert.Equal(2, lines.Count);
        Assert.Contains("= backing [0x8800, 0x8C00)", lines[0]);
        Assert.Contains("ALIAS backing [0x8800, 0x8C00) is also guest [0x0000000090000800, 0x0000000090000C00)", lines[1]);

        var alone = GuestBackingForensics.DescribeAliases(Views, 0xA000_0100, 0x40, limit: 8);
        Assert.Contains("no other view maps these backing bytes", alone[1]);
        Assert.Contains("is in no backing view", GuestBackingForensics.DescribeAliases(Views, 0x1000, 0x10, 8)[0]);
    }

    [Fact]
    public void FindBytesNamesEveryGuestAddressOfEachHitAndSkipsPagesThatAreNotResident()
    {
        var backing = new byte[0x50000];
        byte[] needle = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        needle.CopyTo(backing, 0xC000); // mapped twice: 0x7000C000 and 0x90004000
        needle.CopyTo(backing, 0x40010); // 0xA0000010
        needle.CopyTo(backing, 0x30000); // no view maps it

        fixed (byte* alias = backing)
        {
            var all = GuestBackingForensics.FindBytes(Views, alias, (ulong)backing.Length, needle, 32, residency: null);
            Assert.Equal(3, all.Count);
            Assert.Contains("backing 0xC000 = guest 0x000000007000C000", all[0]);
            Assert.Contains("guest 0x0000000090004000", all[0]);
            Assert.Contains("backing 0x40010 = guest 0x00000000A0000010", all[1]);
            Assert.Contains("hits=2", all[2]);

            // Only the page holding backing 0x40000.. counts as resident.
            var aliasAddress = (nint)alias;
            var resident = GuestBackingForensics.FindBytes(Views, alias, (ulong)backing.Length, needle, 32,
                (start, length, pageSize, vector) =>
                {
                    for (ulong page = 0; page * pageSize < length; page++)
                    {
                        var offset = (ulong)((nint)start - aliasAddress) + page * pageSize;
                        vector[page] = offset == 0x40000 ? (byte)1 : (byte)0;
                    }

                    return true;
                });
            Assert.Equal(2, resident.Count);
            Assert.Contains("backing 0x40010", resident[0]);
            Assert.Contains("hits=1", resident[1]);
        }
    }
}
