<!-- Copyright (C) 2026 SharpEmu Emulator Project -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Program identity lookup benchmark

Run both modes in Release with the emulator's x64/Rosetta runtime:

```sh
DOTNET_ROOT=$HOME/.dotnet-x64 $HOME/.dotnet-x64/dotnet run -c Release --project tools/SharpEmu.Tools.ProgramIdentityBench
SHARPEMU_PROGRAM_INTERN=0 DOTNET_ROOT=$HOME/.dotnet-x64 $HOME/.dotnet-x64/dotnet run -c Release --no-build --project tools/SharpEmu.Tools.ProgramIdentityBench
```

This measures a registered header plus an undeclared-hash 16 KiB shader at the
same address, plus the real graphics-pipeline lookup with a warm vertex-only pipeline,
with two million draws per row. The tool resolves the internal provider once using
reflection; timing uses its ordinary public interface. It excludes resource evaluation,
translation and GPU execution. The managed-only memory notifies every mutation;
production uses the session's `GuestGpuMemory.WriteGenerations` with page guards.
The disabled mode reads the header and code on every draw; both modes use exactly
the same code bytes and produce the same content hash.

`GuestWriteGenerations.Register(address, size)` arms observation and returns an
immutable token. Read bytes after registration, and check `Token.IsCurrent` after
reading and before reuse. Never refresh an old token to certify an old result.
`NoteWrite` invalidates intersecting pages without locks or allocations; a usual
single-page store performs one interlocked increment. Managed memory calls it both
before and after stores, including copies. Native faults invalidate, release that
page's observer watch, and invalidate again after recovery; enrollment re-arms the
watch before a new capture. A page left writable after a fault cannot certify a hit.
GPU notifications invalidate before writes are encoded; misses synchronize the range
before capturing. Remaps, permission changes and unbounded writes veto all tokens.

Versions use a fixed 512 KiB striped page table. A collision can cause an extra
capture, never a stale hit. Registering a range allocates only on a miss; validation
visits its page stamps, independent of shader instruction count. Page-guard updates
are restricted to cold enrollment, native fault recovery and teardown; managed store
notifications never take observer locks. Registrations share one native watcher per
page for the session. Unknown memory implementations keep the read/hash fallback.
