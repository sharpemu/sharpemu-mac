// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Memory.GpuMemory;

// The managed-write lease is taken only once a backend that makes GPU claims registered (GuestWriteReservations.Gate.cs).
[Collection(GpuMemoryStateCollection.Name)]
public sealed class ManagedWriteLeaseGateTests
{
    private sealed class Store : IGuestBufferStore
    {
        internal readonly List<(bool Landed, ulong Address, ulong Size)> Calls = [];
        public bool MayHoldGpuData(ulong address, ulong size) => true;
        public bool NoteEmulatorWrite(ulong address, ulong size) => true;
        public bool MarkCpuWrite(ulong address, ulong size)
        {
            lock (Calls) Calls.Add((false, address, size));
            return true;
        }

        public void MarkCpuWriteLanded(ulong address, ulong size)
        {
            lock (Calls) Calls.Add((true, address, size));
        }

        public bool DownloadToCpu(ulong address, ulong size) => true;
    }

    [Fact]
    public void WithoutAClaimantAWriteTakesNoLeaseAndAClaimantWaitsForAWriteInFlight()
    {
        var saved = GuestWriteReservations.GateEnabled;
        GuestWriteReservations.GateEnabled = true;
        try
        {
            using var memory = new GuestGpuMemory(new RecordingAddressSpace());
            var store = new Store();
            memory.AttachStores(store, null);
            memory.Register(0x10000, 0x4000, GuestPageProtection.Read | GuestPageProtection.Write);
            store.Calls.Clear();
            var reservationAttempts = 0;
            memory.BeforeManagedWriteReservationForTest = _ => reservationAttempts++;
            var reservations = memory.Pages.ManagedWrites;

            // No claimant: no reservation, the stores hear of the write before and after it lands as before the lease.
            Assert.False(reservations.LeasesWrites);
            using (memory.BeginManagedWrite(0x10000, 16)) { }
            Assert.Equal(0, reservationAttempts);
            Assert.Equal(new (bool, ulong, ulong)[] { (false, 0x10000, 16), (true, 0x10000, 16) }, store.Calls);

            // A claim while a write that started unleased is in flight waits until that write landed and was marked.
            Task<bool> claimed;
            using (memory.BeginManagedWrite(0x10000, 16))
            {
                claimed = Task.Run(() =>
                {
                    using var claim = reservations.BeginGpuClaim(0x10000, 16);
                    lock (store.Calls) return store.Calls.Count == 4 && store.Calls[3].Landed;
                });
                Assert.False(claimed.Wait(200));
                Assert.True(reservations.LeasesWrites);
                Assert.False(reservations.HasClaimant);
            }

            Assert.True(claimed.Wait(5000));
            Assert.True(claimed.Result);
            Assert.True(reservations.HasClaimant);

            // From now on every write takes the lease.
            using (memory.BeginManagedWrite(0x10000, 16)) { }
            Assert.Equal(1, reservationAttempts);

            // A write on these reservations can no longer go unleased, and a second registration returns at once.
            reservations.RegisterClaimant();
            Assert.False(reservations.TryEnterUnleasedWrite());
        }
        finally
        {
            GuestWriteReservations.GateEnabled = saved;
        }
    }

    // SHARPEMU_MANAGED_WRITE_LEASE_GATE=0: every write takes the lease, as upstream.
    [Fact]
    public void WithTheGateOffEveryWriteTakesTheLease()
    {
        var saved = GuestWriteReservations.GateEnabled;
        GuestWriteReservations.GateEnabled = false;
        try
        {
            using var memory = new GuestGpuMemory(new RecordingAddressSpace());
            memory.AttachStores(new Store(), null);
            memory.Register(0x10000, 0x4000, GuestPageProtection.Read | GuestPageProtection.Write);
            var reservationAttempts = 0;
            memory.BeforeManagedWriteReservationForTest = _ => reservationAttempts++;
            Assert.True(memory.Pages.ManagedWrites.LeasesWrites);
            using (memory.BeginManagedWrite(0x10000, 16)) { }
            Assert.Equal(1, reservationAttempts);
        }
        finally
        {
            GuestWriteReservations.GateEnabled = saved;
        }
    }
}
