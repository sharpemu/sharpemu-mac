// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.GpuSplit;

/// <summary>
/// The events the GPU process sends the guest, in the order they came, taken by one thread.
/// </summary>
/// <remarks>
/// What <c>BlockingCollection</c> did here, without its wait: that blocks on a semaphore that spins
/// (yields, sleeps) before it parks, and on a busy machine the events thread spent most of its
/// time there. This one parks on a monitor at once. <see cref="Complete"/> ends it as
/// <c>CompleteAdding</c> did: what was posted before it is still taken, and a later
/// <see cref="Post"/> is refused.
/// </remarks>
internal sealed class SplitEventQueue
{
    private readonly object _gate = new();
    private readonly Queue<Action> _queue = new();
    private bool _completed;
    private bool _waiting;

    /// <summary>Queues <paramref name="work"/>; false once <see cref="Complete"/> has been called.</summary>
    public bool Post(Action work)
    {
        lock (_gate)
        {
            if (_completed)
            {
                return false;
            }

            _queue.Enqueue(work);
            if (_waiting)
            {
                Monitor.Pulse(_gate);
            }

            return true;
        }
    }

    /// <summary>No more is accepted; the one waiting in <see cref="Take"/> takes what is left, then ends.</summary>
    public void Complete()
    {
        lock (_gate)
        {
            _completed = true;
            Monitor.PulseAll(_gate);
        }
    }

    /// <summary>
    /// Waits for the next event; false once the queue is completed and empty. For one consumer.
    /// </summary>
    public bool Take(out Action work)
    {
        lock (_gate)
        {
            while (_queue.Count == 0)
            {
                if (_completed)
                {
                    work = null!;
                    return false;
                }

                _waiting = true;
                try
                {
                    Monitor.Wait(_gate);
                }
                finally
                {
                    _waiting = false;
                }
            }

            work = _queue.Dequeue();
            return true;
        }
    }
}
