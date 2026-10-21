#!/usr/bin/env python3
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

"""Keep Rosetta's initial host stack outside the directly executed guest VA.

XNU places a translated process's stack after its executable/dyld mappings and
the AOT gap (bsd/kern/mach_loader.c, parse_machfile). A normal .NET apphost puts
it near 0x300000000, where titles can request fixed direct-memory mappings.
Relocating the last, read-only __LINKEDIT segment moves that stack above the
guest application's address limit. File offsets, code/data addresses, PIE and
segment protections stay intact. No guest address is translated or overwritten.
"""

import argparse
import mmap
import pathlib
import struct
import subprocess
import sys


GUEST_ADDRESS_LIMIT = 0xFC_0000_0000
LC_SEGMENT_64 = 0x19


def linkedit_location(data):
    if len(data) < 32:
        raise ValueError("truncated Mach-O header")
    magic, cpu, _, kind, count, commands_size, _, _ = struct.unpack_from("<8I", data)
    if (magic, cpu, kind) != (0xFEEDFACF, 0x01000007, 2):
        raise ValueError("expected a thin x86-64 Mach-O executable")
    end = 32 + commands_size
    if end > len(data) or count > commands_size // 8:
        raise ValueError("invalid Mach-O command extent")
    cursor = 32
    segments = []
    for _ in range(count):
        if cursor + 8 > end:
            raise ValueError("truncated Mach-O load command")
        command, size = struct.unpack_from("<II", data, cursor)
        if size < 8 or size % 8 or cursor + size > end:
            raise ValueError("invalid Mach-O load command size")
        if command == LC_SEGMENT_64:
            if size < 72:
                raise ValueError("truncated Mach-O segment")
            fields = struct.unpack_from("<16s4Q4I", data, cursor + 8)
            name, address, extent, offset, length, maximum, initial, sections, _ = fields
            if size != 72 + sections * 80 or offset + length > len(data):
                raise ValueError("invalid Mach-O segment extent")
            segments.append((name.rstrip(b"\0"), address, extent, cursor + 24, maximum, initial, sections))
        cursor += size
    if cursor != end:
        raise ValueError("Mach-O command sizes do not match the header")
    matches = [segment for segment in segments if segment[0] == b"__LINKEDIT"]
    if len(matches) != 1 or segments[-1] != matches[0]:
        raise ValueError("expected one final __LINKEDIT segment")
    _, address, extent, field, maximum, initial, sections = matches[0]
    if not extent or extent % 4096 or maximum != 1 or initial != 1 or sections:
        raise ValueError("expected a page-aligned read-only __LINKEDIT without sections")
    if any(start + length > GUEST_ADDRESS_LIMIT for _, start, length, *_ in segments[:-1]):
        raise ValueError("another segment reaches the host __LINKEDIT placement")
    if GUEST_ADDRESS_LIMIT + extent > 0x1_0000_0000_0000:
        raise ValueError("__LINKEDIT exceeds the host address range")
    return field, address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("--check", action="store_true", help="verify placement without changing the executable")
    args = parser.parse_args()
    # Single-file publishes can be large; only load-command pages are needed.
    with args.executable.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        field, previous = linkedit_location(data)
    if args.check:
        if previous != GUEST_ADDRESS_LIMIT:
            raise ValueError("__LINKEDIT placement can put the Rosetta host stack in guest memory")
    elif previous != GUEST_ADDRESS_LIMIT:
        # Only this virtual address changes. The following signing step seals
        # the modified header; the bundle builder later applies its entitlements.
        with args.executable.open("r+b") as output:
            output.seek(field)
            output.write(struct.pack("<Q", GUEST_ADDRESS_LIMIT))
    if sys.platform == "darwin" and not args.check:
        # A failed signing attempt must be repairable on the next build even
        # when the address was already changed. Preserve a valid existing seal.
        valid = previous == GUEST_ADDRESS_LIMIT and subprocess.run(
            ["codesign", "--verify", "--strict", str(args.executable)], capture_output=True).returncode == 0
        if not valid:
            subprocess.run(["codesign", "--force", "--sign", "-", str(args.executable)], check=True)
    print(f"macOS apphost: __LINKEDIT at 0x{GUEST_ADDRESS_LIMIT:X}; Rosetta host stack above guest application VA")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"place-host-stack.py: {error}")
