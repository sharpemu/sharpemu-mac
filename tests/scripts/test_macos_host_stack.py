# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

import importlib.util
import contextlib
import io
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "build/platforms/macos/place-host-stack.py"
SPEC = importlib.util.spec_from_file_location("host_stack", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def executable():
    commands = b"".join(
        struct.pack("<II16s4Q4I", 0x19, 72, name, address, size, 0, 0, protection, protection, 0, 0)
        for name, address, size, protection in (
            (b"__TEXT", 0x1_0000_0000, 0x1000, 5),
            (b"__LINKEDIT", 0x1_0000_1000, 0x1000, 1),
        )
    )
    return struct.pack("<8I", 0xFEEDFACF, 0x01000007, 3, 2, 2, len(commands), 0x200085, 0) + commands


class HostStackLayoutTests(unittest.TestCase):
    def test_relocation_keeps_everything_except_linkedit_address_and_is_idempotent(self):
        original = executable()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "apphost"
            path.write_bytes(original)
            field, previous = MODULE.linkedit_location(original)
            self.assertEqual(previous, 0x1_0000_1000)
            # The synthetic header cannot be signed. Exercise the actual writer
            # with only the external signer replaced; the native test signs for real.
            with mock.patch.object(sys, "argv", [str(SCRIPT), str(path)]), \
                    mock.patch.object(MODULE.subprocess, "run") as signer, contextlib.redirect_stdout(io.StringIO()):
                signer.return_value.returncode = 0
                MODULE.main()
            changed = path.read_bytes()
            self.assertEqual(struct.unpack_from("<Q", changed, field)[0], MODULE.GUEST_ADDRESS_LIMIT)
            self.assertEqual(changed[:field], original[:field])
            self.assertEqual(changed[field + 8:], original[field + 8:])
            before = path.stat().st_mtime_ns
            with mock.patch.object(sys, "argv", [str(SCRIPT), str(path)]), \
                    mock.patch.object(MODULE.subprocess, "run") as signer, contextlib.redirect_stdout(io.StringIO()):
                signer.return_value.returncode = 0
                MODULE.main()
                MODULE.main()
            subprocess.run([sys.executable, str(SCRIPT), str(path), "--check"], check=True, capture_output=True)
            self.assertEqual(path.read_bytes(), changed)
            self.assertEqual(path.stat().st_mtime_ns, before)

    def test_malformed_or_unexpected_images_are_rejected_without_writing(self):
        cases = [b"", b"MZ" + bytes(200)]
        for offset, value in ((4, 0x0100000C), (16, 99), (20, 8), (36, 0), (104, 8), (164, 5)):
            data = bytearray(executable())
            struct.pack_into("<I", data, offset, value)
            cases.append(data)
        for data in cases:
            with self.subTest(data=bytes(data[:24])):
                with self.assertRaises(ValueError):
                    MODULE.linkedit_location(data)

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin Mach VM and the x86-64 toolchain")
    def test_signed_x64_host_can_map_the_fixed_two_gib_guest_arena(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "guest-map"
            subprocess.run(["clang", "-arch", "x86_64", "-O1", "-Wall", "-Werror",
                            str(ROOT / "tests/Fixtures/MacGuestMapping/guest-map.c"), "-o", str(binary)], check=True)
            original = binary.read_bytes()
            field, _ = MODULE.linkedit_location(original)
            subprocess.run([sys.executable, str(SCRIPT), str(binary)], check=True)
            patched = binary.read_bytes()
            self.assertEqual(struct.unpack_from("<I", original, 24), struct.unpack_from("<I", patched, 24))
            self.assertEqual(struct.unpack_from("<Q", patched, field)[0], MODULE.GUEST_ADDRESS_LIMIT)
            subprocess.run(["codesign", "--verify", "--strict", str(binary)], check=True)
            for backing in ("chunked", "shm"):
                result = subprocess.run([str(binary), backing], check=True, capture_output=True, text=True, timeout=30)
                self.assertIn("fixed-guest-map-pass", result.stdout)
                print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
