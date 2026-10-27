// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.GpuMemory;

/// <summary>
/// The GPU side's summary of the guest pages a managed write must wait on: pages under an image
/// the GPU wrote and has not written back, and pages a recorded writeback has not reached yet.
/// The split's guest process reads it to make a managed write's blocking call only where the GPU
/// side has something to put under the write first.
/// </summary>
/// <remarks>
/// A page counts every claim on it. <see cref="Add"/> marks its pages at once, as a full barrier,
/// so the caller makes the claim visible before the state it stands for. <see cref="Remove"/>
/// only lowers the counts: a page whose count reached zero is unmarked by the next
/// <see cref="Publish"/>, which the owner calls where its state is whole (the GPU process's image
/// cache does so as it lets its lock go), so a claim dropped and replaced inside one operation
/// (an image written back: its GPU claim ends, its writeback's begins) never shows a gap.
/// </remarks>
public interface IPendingWritebackPages
{
    /// <summary>The caller keeps the summary from now on. Until then every page counts as pending.</summary>
    void Activate();

    void Add(ulong address, ulong size);

    void Remove(ulong address, ulong size);

    void Publish();
}

/// <summary>A guest address space that carries a <see cref="IPendingWritebackPages"/> summary.</summary>
public interface IPendingWritebackSource
{
    IPendingWritebackPages? PendingWritebacks { get; }
}
