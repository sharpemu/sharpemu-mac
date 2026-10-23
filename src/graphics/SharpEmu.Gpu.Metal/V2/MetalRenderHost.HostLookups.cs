// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    internal long ShaderOwnershipHits { get; private set; }
    private readonly object _shaderOwnershipGate = new();
    private readonly Dictionary<(ulong Address, ulong Size), (ResourceDependencyEpoch Epoch, object? Mapping)> _shaderOwnership = new();

    private bool TryShaderOwnership(ulong address, ulong size)
    {
        if (!HostLookupCounters.Enabled) return false;
        if (!((IResourceDependencySource)this).DependencyReadsReady)
        {
            lock (_shaderOwnershipGate) _shaderOwnership.Clear();
            HostLookupCounters.Note(HostLookupCounters.Cache.ShaderOwnership, false, refused: true);
            return false;
        }
        var epoch = ((IResourceDependencySource)this).DependencyEpoch;
        var mapping = _backing.BackingAliasSnapshot;
        lock (_shaderOwnershipGate)
        {
            if (_shaderOwnership.TryGetValue((address, size), out var cached) && cached.Epoch == epoch &&
                ReferenceEquals(cached.Mapping, mapping))
            {
                ShaderOwnershipHits++;
                HostLookupCounters.Note(HostLookupCounters.Cache.ShaderOwnership, true);
                return true;
            }
        }
        HostLookupCounters.Note(HostLookupCounters.Cache.ShaderOwnership, false);
        return false;
    }

    private void KeepShaderOwnership(ulong address, ulong size)
    {
        if (!HostLookupCounters.Enabled || size == 0 || !((IResourceDependencySource)this).DependencyReadsReady) return;
        var epoch = ((IResourceDependencySource)this).DependencyEpoch;
        var mapping = _backing.BackingAliasSnapshot;
        // A failed/partial wait cannot certify ownership. CPU content writes do not
        // create GPU ownership; the caller still validates its content generations.
        if (_imageCache.HasGpuModifiedImageBytes(address, size) ||
            ((IMetalImageCacheOwner)this).HasPendingGpuWrite(address, size) ||
            !((IResourceDependencySource)this).DependencyReadsReady || epoch != ((IResourceDependencySource)this).DependencyEpoch ||
            !ReferenceEquals(mapping, _backing.BackingAliasSnapshot)) return;
        lock (_shaderOwnershipGate)
        {
            if (_shaderOwnership.Count >= 4096) _shaderOwnership.Clear();
            _shaderOwnership[(address, size)] = (epoch, mapping);
        }
    }
}
