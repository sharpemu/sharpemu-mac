<!--
Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Fixed guest mappings and macOS host allocations

The ASTRO BOT PPSA21567 v1.018 log from build `e3fc0a0` records
`sceKernelMapNamedDirectMemory` returning `ENOMEM` for a fixed `0x7E000000`-byte
mapping at `0x300000000`, protection `0xF3`, flags `0x10`. Guest libc then calls
`memset` on the unmapped range and faults writing `0x300000020`. The instruction
dump's AVX warning does not explain the preceding mapping failure.

An isolated replay on macOS 27.2 with the x64 CLR reproduces the same kernel
export returning `0x8002000C`, specifically `map_direct failed=address-selection`.
Two host allocations can occupy the requested guest arena:

- Rosetta's initial stack and its guard pages. XNU places this stack relative
  to the executable/dyld mappings and its AOT gap; see `parse_machfile` in the
  [XNU loader](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/mach_loader.c).
- The canonical guest-backing alias. Once the stack moves, an unconstrained
  16 GiB alias can take the low gap instead. The replay's VM map confirmed
  this second overlap, so changing executable layout alone is insufficient.

The macOS build relocates only the apphost's final read-only `__LINKEDIT` segment
to the guest application limit, `0xFC00000000`, before re-signing. Code, data,
PIE and segment protections are preserved. Both backing providers place their
private aliases at or above `0x10000000000`; a POSIX hint that lands below that
limit is released and refused. No guest pointer is relocated and no foreign
host mapping is overwritten.

With both changes, the C# replay returns success for the same export and checks
markers at `0x300000020` and `0x37DFFFFF8`, using both chunked and POSIX backing.
CI also compiles and signs an x64 native fixture that performs the fixed mapping,
checks its first write and object boundaries, and verifies aliases in both
directions. Managed tests check placement of the full 16 GiB plus 448 MiB backing.

This qualifies the reproduced mapping failure. ASTRO BOT boot, rendering and
gameplay still require a run with the actual game dump; it was unavailable on
the reproduction machine.
