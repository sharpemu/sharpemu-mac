// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

// An optional backend contract. A token certifies the EXACT bytes copied during capture,
// not merely an address or a notification counter. In-flight CPU stores, native pages
// released after a first fault, pending GPU writers, aliases and mapping ABA must veto it.
public enum ResourceDependencyStatus { Current, CpuWrite, GpuWrite, PendingWriter, Mapping, ImageIdentity, Overlay, Unavailable }

public abstract class ResourceDependency : IDisposable
{
    public virtual long RetainedBytes => 256;
    public abstract ResourceDependencyStatus Validate();
    public abstract void Dispose();
}

public interface IResourceDependencySource
{
    // Also needed by stages with no memory reads: their resource identity/overlay epoch
    // can change independently of their roots. Capture must be side-effect free.
    ResourceDependencyEpoch DependencyEpoch { get; }

    // Capture may allocate; Validate must neither read bytes nor allocate. A failed
    // capture grants no reuse. Implementations own any write-watch bookkeeping.
    bool TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean,
        out ResourceDependency? dependency);
}

public readonly record struct ResourceDependencyEpoch(ulong GpuWrites, long Overlay, long Mapping, long Images, long Labels = 0);
