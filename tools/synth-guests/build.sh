#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Usage: build.sh [-o OUT_DIR] [guest ...] [-- extra clang flags e.g. -DWORKERS=13 -DDURATION_S=20]
# Builds PS5-style guest ELFs (ET_DYN, OSABI 9 / ABI version 2, imports = NID symbol names) with host clang +
# ld.lld. With no guest names builds all of them: jobpool scream wakeping sleeplat baseline.
# Output: OUT_DIR/<guest>.bin  (default OUT_DIR=../../artifacts/synth-guests next to this directory's repo root).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${SYNTH_OUT:-$(cd "$here/../.." && pwd)/artifacts/synth-guests}
guests=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -o) out=$2; shift 2 ;;
    --) shift; break ;;
    *) guests+=("$1"); shift ;;
  esac
done
extra=("$@")
[[ ${#guests[@]} -gt 0 ]] || guests=(jobpool scream wakeping sleeplat baseline)
mkdir -p "$out"
for g in "${guests[@]}"; do
  src="$here/$g.c"
  [[ -f $src ]] || { echo "no such guest: $g" >&2; exit 2; }
  # -fasynchronous-unwind-tables + --eh-frame-hdr: SharpEmu finds function boundaries (for the spin-nap and
  # red-zone patchers) through .eh_frame_hdr, exactly like a real title. -O2 keeps the C-side loop shapes realistic.
  clang -target x86_64-unknown-linux-gnu -O2 -fPIC -ffreestanding -fno-stack-protector -fno-builtin \
    -fasynchronous-unwind-tables -nostdlib -shared -fuse-ld=lld \
    -Wall -Wextra -Wno-unused-function -Wno-constant-logical-operand \
    -Wl,-e,_start -Wl,-z,now -Wl,-z,norelro -Wl,--hash-style=sysv -Wl,-z,max-page-size=0x1000 \
    -Wl,-z,common-page-size=0x1000 -Wl,--no-rosegment -Wl,--eh-frame-hdr -Wl,--unresolved-symbols=ignore-all \
    "${extra[@]}" "$src" "$here/common.c" -o "$out/$g.bin"
  # Every undefined symbol must be a NID (11 base64-ish chars). Anything else (memcpy, __udivti3, ...) would become an
  # unresolved import at run time, so fail the build instead.
  bad=$(nm -D --undefined-only "$out/$g.bin" | awk '{print $2}' | grep -Ev '^[A-Za-z0-9+_-]{11}$' || true)
  [[ -z $bad ]] || { echo "$g: non-NID undefined symbols: $bad" >&2; exit 1; }
  if [[ $g == jobpool ]]; then python3 "$here/check-shapes.py" "$out/$g.bin" --expect-jobpool >/dev/null || { echo "jobpool loop shapes drifted; run check-shapes.py" >&2; exit 1; }; fi
  # SharpEmu treats ABI version 2 as Gen5 (PS5, image base 0x800000000); OSABI 9 = FreeBSD like real images.
  printf '\x09\x02' | dd of="$out/$g.bin" bs=1 seek=7 count=2 conv=notrunc status=none
  echo "built $out/$g.bin"
done
