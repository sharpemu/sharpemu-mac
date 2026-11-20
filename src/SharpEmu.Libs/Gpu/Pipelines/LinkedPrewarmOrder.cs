// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The order in which the boot prewarm compiles the linked records of earlier sessions (ComputePrewarmRecord with linked
// callees), as a public contract for a backend outside this assembly (the Metal backend; Libs gives extracted backends
// public contracts only). It is upstream's background lane order (ShaderPrewarmQueue.LinkedLaneOrder, used by
// VulkanVideoPresenter.StartLinkedPrewarm): every caller's first record before any caller's second; a caller's first is
// the record whose library links the callees an earlier session kept for it (ShaderCallUnionStore, the library a session
// links first), else its oldest, and its others follow newest first; within a round, callers a session saw compile for
// minutes (ShaderPrewarmList.IsSlowLinkedCompile) go first, then the order in which callers were first recorded.
public static class LinkedPrewarmOrder
{
    // Whether the record is a linked permutation (its program holds callees).
    public static bool IsLinked(ComputePrewarmRecord record) => record.Linked is not null;

    // The linked records in the lane's order. linkedDirectory: the host's LinkedProgramCacheDirectory (null keeps none).
    public static T[] Order<T>(IReadOnlyList<T> records, Func<T, ComputePrewarmRecord> record, ShaderPrewarmList list,
        string? linkedDirectory) =>
        Order(records, item => record(item).Hash, item => record(item).Linked, list.IsSlowLinkedCompile, linkedDirectory);

    internal static T[] Order<T>(IReadOnlyList<T> records, Func<T, ulong> caller, Func<T, LinkedShaderPrewarm?> linked,
        Func<ulong, bool> slow, string? linkedDirectory)
    {
        var kept = ShaderCallUnionStore.Open(linkedDirectory);
        return ShaderPrewarmQueue<T>.LinkedLaneOrder(records, caller, slow,
            item => linked(item) is { } library && kept.KeepsCalleesOf(ShaderStage.Compute, caller(item), library));
    }
}
