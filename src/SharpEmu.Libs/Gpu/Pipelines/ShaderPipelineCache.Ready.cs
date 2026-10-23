// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Libs.Gpu.Pipelines;

// Ready handles need no compile lock. Only a miss allocates a future; competing
// requests for that key share it, including its failure. Compilation runs outside
// dictionary insertion and publication, and an unfinished nonblocking attempt is retryable.
internal sealed class ReadyPipelineCache<TKey> where TKey : notnull
{
    private readonly ConcurrentDictionary<TKey, PipelineHandle> _ready = new();
    private readonly ConcurrentDictionary<TKey, TaskCompletionSource<PipelineHandle?>> _building = new();
    public int Count => _ready.Count;
    public bool TryGet(TKey key, out PipelineHandle handle) => _ready.TryGetValue(key, out handle);

    public bool GetOrCreate(TKey key, Func<PipelineHandle?> create, bool wait, out PipelineHandle handle)
    {
        if (_ready.TryGetValue(key, out handle)) return true;
        var candidate = new TaskCompletionSource<PipelineHandle?>(TaskCreationOptions.RunContinuationsAsynchronously);
        var flight = _building.GetOrAdd(key, candidate);
        if (ReferenceEquals(candidate, flight))
        {
            try
            {
                // A previous owner may have published between the first probe and insertion.
                var result = _ready.TryGetValue(key, out var ready) ? ready : create();
                if (result is { } built) _ready[key] = built;
                flight.SetResult(result);
            }
            catch (Exception error) { flight.SetException(error); }
            finally { _building.TryRemove(new KeyValuePair<TKey, TaskCompletionSource<PipelineHandle?>>(key, flight)); }
        }
        else if (!wait && !flight.Task.IsCompleted)
        {
            handle = default;
            return false;
        }
        var completed = flight.Task.GetAwaiter().GetResult();
        handle = completed ?? default;
        return completed.HasValue;
    }
}
