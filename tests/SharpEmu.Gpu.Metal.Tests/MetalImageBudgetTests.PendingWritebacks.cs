// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed partial class MetalImageBudgetTests
{
    [Fact]
    public void PendingPagesFollowTheRealImageThroughRetirementAndWritebackCompletion()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var pending = new PendingPageRecorder { RequirePending = true };
        rig.Cache.PendingWritebacks = pending;
        Assert.True(pending.Active);
        Assert.False(pending.HasImagePages);

        // This uses the real Metal image cache, texture upload, retirement and GPU writeback,
        // rather than a second implementation of the image cache's state machine.
        var (identifier, written) = ReleaseAGpuWrittenImage(rig);
        Assert.False(rig.Cache.TryGetImage(identifier, out _));
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
        Assert.True(pending.HasImagePages);
        Assert.False(pending.PublishedGap);
        Assert.False(pending.NegativeClaim);

        rig.Cache.Collect();
        Assert.True(pending.HasImagePages);
        Assert.False(pending.PublishedGap);

        pending.RequirePending = false;
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.Cache.Collect();
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        Assert.False(pending.HasImagePages);
        Assert.False(pending.PublishedGap);
        Assert.False(pending.NegativeClaim);
        Assert.Equal(written, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AReadOnlyImageDoesNotClaimPendingWritebackPages()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var pending = new PendingPageRecorder();
        rig.Cache.PendingWritebacks = pending;
        var image = rig.Cache.GetImage(rig.Acquire(GuestBase, ImageRole.Texture));
        Assert.False(image.IsGpuModified);
        rig.Cache.Collect();
        Assert.False(pending.HasImagePages);
        Assert.False(pending.NegativeClaim);
    }

    private sealed class PendingPageRecorder : IPendingWritebackPages
    {
        private readonly Dictionary<ulong, int> _claims = new();
        private readonly HashSet<ulong> _visible = new();
        private bool _everClaimed;

        public bool Active { get; private set; }
        public bool RequirePending { get; set; }
        public bool PublishedGap { get; private set; }
        public bool NegativeClaim { get; private set; }
        public bool HasImagePages => Enumerable.Range(0, (int)((ImageSize + 4095) / 4096))
            .All(page => _visible.Contains((GuestBase >> 12) + (ulong)page));

        public void Activate() => Active = true;

        public void Add(ulong address, ulong size)
        {
            for (var page = address >> 12; page <= (address + size - 1) >> 12; page++)
            {
                _claims.TryGetValue(page, out var count);
                _claims[page] = count + 1;
                _visible.Add(page);
                _everClaimed = true;
            }
        }

        public void Remove(ulong address, ulong size)
        {
            for (var page = address >> 12; page <= (address + size - 1) >> 12; page++)
            {
                _claims.TryGetValue(page, out var count);
                _claims[page] = count - 1;
                NegativeClaim |= count <= 0;
            }
        }

        public void Publish()
        {
            foreach (var (page, count) in _claims)
                if (count <= 0) _visible.Remove(page);
            // Record a violation without throwing inside RegionLock's release callback.
            PublishedGap |= RequirePending && _everClaimed && !HasImagePages;
        }
    }
}
