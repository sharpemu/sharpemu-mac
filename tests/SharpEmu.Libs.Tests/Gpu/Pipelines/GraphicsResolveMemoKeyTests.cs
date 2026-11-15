// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// The resolve memo's key: equal for equal registers and draw inputs, different for any word the resolution reads, and
// a miss told apart by the half that differs.
public sealed class GraphicsResolveMemoKeyTests
{
    private static readonly KeptGraphicsFlags Flags = new(PixelActive: true, DepthBound: false, DepthAspectBound: false,
        StencilAspectBound: false, TriangleList: true, StartInstanceUserData: -1);

    private static RegisterBanks Banks()
    {
        var banks = new RegisterBanks(static message => new InvalidOperationException(message));
        banks.Shader.Vertex.ExportAddress = 0x1000;
        banks.Shader.Pixel.Address = 0x2000;
        banks.Shader.Vertex.GeometryUserScalars.Set(0, 0x1234, UserScalarKind.Unknown);
        banks.Shader.Pixel.UserScalars.Set(0, 0x5678, UserScalarKind.Unknown);
        return banks;
    }

    private static bool Fill(GraphicsResolveMemo memo, RegisterBanks banks, KeptGraphicsFlags? flags = null) =>
        memo.Fill(banks, [ColorComponentMap.Identity], flags ?? Flags, new PixelOutputBlendBits(0));

    [Fact]
    public void EqualBanksFindTheStoredKeyAndAChangedWordDoesNot()
    {
        var memo = new GraphicsResolveMemo(ResolveMemoMode.Shadow, 64);
        var banks = Banks();
        Assert.True(Fill(memo, banks));
        Assert.Null(memo.Find());
        var stored = memo.Store();
        Assert.True(Fill(memo, Banks()));
        Assert.Same(stored, memo.Find());

        banks.Context.ColorTargets[3] = banks.Context.ColorTargets[3] with { Info = 1 };
        Assert.True(Fill(memo, banks));
        Assert.Null(memo.Find());
        Assert.Equal(ResolveMemoKeyMiss.RegistersOnly, memo.ClassifyMiss());

        Assert.True(Fill(memo, Banks(), Flags with { DepthBound = true }));
        Assert.Null(memo.Find());
    }

    [Fact]
    public void AUserDataWordOnlyMissIsToldApartAndAnUnreadWordIsNotInTheKey()
    {
        var memo = new GraphicsResolveMemo(ResolveMemoMode.Shadow, 64);
        Assert.True(Fill(memo, Banks()));
        memo.Store();

        var banks = Banks();
        banks.Shader.Pixel.UserScalars.Set(0, 0x9999, UserScalarKind.Unknown);
        Assert.True(Fill(memo, banks));
        Assert.Null(memo.Find());
        Assert.Equal(ResolveMemoKeyMiss.UserDataOnly, memo.ClassifyMiss());

        // A pixel stage the draw does not run is not part of its key.
        Assert.True(Fill(memo, Banks(), Flags with { PixelActive = false }));
        memo.Store();
        Assert.True(Fill(memo, banks, Flags with { PixelActive = false }));
        Assert.NotNull(memo.Find());

        // A user data count over the bank is refused (the full resolution stops on it).
        var over = Banks();
        over.Shader.Vertex.GeometryResource2 = over.Shader.Vertex.GeometryResource2 with { UserScalarCount = 40 };
        Assert.False(Fill(memo, over));
    }

    [Fact]
    public void ASetKeepsTwoKeysAndReplacesTheOlder()
    {
        var memo = new GraphicsResolveMemo(ResolveMemoMode.Shadow, 2);
        var keys = Enumerable.Range(0, 3).Select(index =>
        {
            var banks = Banks();
            banks.Shader.Vertex.ExportAddress = 0x1000 + (ulong)index * 0x100;
            return banks;
        }).ToArray();
        foreach (var banks in keys)
        {
            Assert.True(Fill(memo, banks));
            memo.Store();
        }

        Assert.Equal(2, memo.Count);
        Assert.True(Fill(memo, keys[0]));
        Assert.Null(memo.Find());
        Assert.True(Fill(memo, keys[2]));
        Assert.NotNull(memo.Find());
    }
}
