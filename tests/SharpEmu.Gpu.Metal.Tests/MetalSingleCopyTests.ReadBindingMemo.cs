// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalSingleCopyTests
{
    // What the read-only binding memo (MetalReadBindingMemoV2) asks of the windows once single-copy holes or stale
    // windows changed: a range's in-place answer stands only with no released chunk (W5), no hole (T1) and no stale
    // window under it, through a window still in the partition; holes elsewhere in the window leave it standing.
    [Fact]
    public void AnInPlaceAnswerStandsOnlyWithNoReleasedChunkHoleOrStaleWindowUnderIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 71);
        rig.Release(request, bytes, (int)(size / Chunk));
        ulong Alias(ulong address) => rig.Views.AliasBase + (address - rig.Address);

        // W5: released chunks under the run.
        Assert.False(rig.Memory.InPlaceAnswerStands(Alias(rig.ImageAddress), size));

        // The bytes beside the run, in the same window: the holes are elsewhere in it.
        Assert.True(rig.Memory.TryResolveInPlace(rig.ImageAddress + size, size, out var near, out var nearAlias, out var window));
        Assert.NotNull(window);
        Assert.True(rig.Memory.InPlaceAnswerStands(nearAlias, size));
        Assert.True(rig.Memory.RevalidateInPlace(window, near.Buffer, nearAlias, size, out _));

        // Restored: no chunk released any more, but the holes are still punched until something fills them.
        Assert.Equal(bytes, rig.Read(rig.ImageAddress, size));
        Assert.True(rig.Memory.IsSingleCopyHole(rig.ChunkOf(rig.ImageAddress)));
        Assert.True(rig.Memory.InPlaceAnswerStands(Alias(rig.ImageAddress), size));
        // T1: a hole under the range.
        Assert.False(rig.Memory.RevalidateInPlace(window, near.Buffer, Alias(rig.ImageAddress), size, out _));
        Assert.True(rig.Memory.RevalidateInPlace(window, near.Buffer, nearAlias, size, out var trap));

        // Binding the run fills its holes; the window, pinned by the binding beside, is stale until the next
        // submission, for the bytes beside too.
        var staged = rig.Memory.Resolve(rig.ImageAddress, size);
        Assert.False(rig.Memory.IsInPlace(staged));
        Assert.True(rig.Memory.HasStaleSingleCopyWindows);
        Assert.False(rig.Memory.RevalidateInPlace(window, near.Buffer, nearAlias, size, out var staleTrap));
        Assert.NotEqual(trap, staleTrap);

        // The next submission renews it: the old window has left the partition, the new one stands.
        rig.Submit();
        Assert.True(window.Retired);
        Assert.False(rig.Memory.RevalidateInPlace(window, near.Buffer, nearAlias, size, out _));
        Assert.True(rig.Memory.TryResolveInPlace(rig.ImageAddress + size, size, out var renewed, out nearAlias, out var renewedWindow));
        Assert.True(rig.Memory.RevalidateInPlace(renewedWindow!, renewed.Buffer, nearAlias, size, out _));
        Assert.True(rig.Memory.RevalidateInPlace(renewedWindow!, renewed.Buffer, Alias(rig.ImageAddress), size, out _));
    }
}
