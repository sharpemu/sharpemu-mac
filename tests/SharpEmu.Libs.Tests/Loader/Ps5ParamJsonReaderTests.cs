// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class Ps5ParamJsonReaderTests
{
    [Theory]
    [InlineData("UP8103-PPSA18454_00-0936750542714392", "PPSA18454")]
    [InlineData(" EP8293-PPSA20591_00-0571511008440793 ", "PPSA20591")]
    [InlineData("UP8103-PPSA18454", null)]
    [InlineData("UP8103_PPSA18454_00-0936750542714392", null)]
    [InlineData("UP8103-ppsa18454_00-0936750542714392", null)]
    [InlineData("UP8103-PPSA1845X_00-0936750542714392", null)]
    [InlineData("", null)]
    [InlineData(null, null)]
    public void TheTitleIdIsReadFromAContentId(string? contentId, string? expected)
    {
        Assert.Equal(expected, Ps5ParamJsonReader.TitleIdFromContentId(contentId));
    }

    [Theory]
    [InlineData("[\"UP8103-PPSA18454_00-0936750542714392\",\"EP8293-PPSA20591_00-0571511008440793\"]", "PPSA18454")]
    [InlineData("[\"not a content id\",\"EP8293-PPSA20591_00-0571511008440793\"]", "PPSA20591")]
    [InlineData("[]", null)]
    [InlineData("{\"contentId\":\"UP8103-PPSA18454_00-0936750542714392\"}", null)]
    [InlineData("not json", null)]
    public void TheTitleIdIsReadFromContentIdsJson(string json, string? expected)
    {
        Assert.Equal(expected, Ps5ParamJsonReader.TryReadContentIdsTitleId(Encoding.UTF8.GetBytes(json)));
    }

    [Fact]
    public void AParamJsonWithoutATitleIdFallsBackToItsContentId()
    {
        var json = "{\"contentId\":\"UP8103-PPSA18454_00-0936750542714392\",\"contentVersion\":\"01.000.000\"}";

        var (_, titleId, version) = Ps5ParamJsonReader.TryReadPs5Param(Encoding.UTF8.GetBytes(json));

        Assert.Equal("PPSA18454", titleId);
        Assert.Equal("01.000.000", version);
    }

    [Fact]
    public void AParamJsonTitleIdWinsOverItsContentId()
    {
        var json = "{\"titleId\":\"PPSA00001\",\"contentId\":\"UP8103-PPSA18454_00-0936750542714392\"}";

        Assert.Equal("PPSA00001", Ps5ParamJsonReader.TryReadPs5Param(Encoding.UTF8.GetBytes(json)).TitleId);
    }
}
