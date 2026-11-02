#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Usage: ab.sh A B guest.bin N [-- extra run.sh args]
# Alternating ABBA runs of one guest under two settings, then compare.py.
#   A / B  = either an emulator build directory (a directory that contains the `SharpEmu` executable, e.g. a copy of
#            artifacts/profiles/linux/bin/Release/net10.0/linux-x64) or an environment spec such as
#            "SHARPEMU_SPIN_NAP_LOCKED=1" / "K1=V1 K2=V2"; "-" means the current build with no extra env.
#            A dir can be combined with env: "DIR:K=V K2=V2".
#   N      = runs per arm (rounded up to even; the order is A B B A A B B A ...).
# Output: $SYNTH_RUNS/ab-<time>-<guest>/{A,B}/NN/ (each a run.sh run dir) and compare.txt.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[[ $# -ge 4 ]] || { sed -n '4,13p' "$0"; exit 2; }
specA=$1; specB=$2; guest=$(readlink -f "$3"); n=$4; shift 4
[[ ${1:-} == -- ]] && shift
(( n % 2 )) && n=$((n + 1))
root=${SYNTH_RUNS:-/home/xnetcat/Projects/sharpemu/ds-cpu/synth-runs}
out="$root/ab-$(date +%Y%m%d-%H%M%S)-$(basename "${guest%.bin}")"
mkdir -p "$out/A" "$out/B"
printf 'A=%s\nB=%s\nguest=%s\nruns_per_arm=%s\n' "$specA" "$specB" "$guest" "$n" > "$out/spec.txt"

run_arm() { # arm spec index
  local arm=$1 spec=$2 idx=$3 dir="" envs="" bin_args=() env_args=()
  if [[ $spec != - ]]; then
    if [[ $spec == *:* && -d ${spec%%:*} ]]; then dir=${spec%%:*}; envs=${spec#*:}
    elif [[ -d $spec ]]; then dir=$spec
    else envs=$spec; fi
  fi
  [[ -n $dir ]] && bin_args=(--bin "$dir/SharpEmu")
  for kv in $envs; do env_args+=(--env "$kv"); done
  "$here/run.sh" "$guest" "$arm$idx" --out-dir "$out/$arm/$(printf %02d "$idx")" ${bin_args[@]+"${bin_args[@]}"} ${env_args[@]+"${env_args[@]}"} "$@" || echo "run $arm$idx failed (see its dir)"
}

ia=0; ib=0
for ((blk = 0; blk < n / 2; blk++)); do
  for arm in A B B A; do
    if [[ $arm == A ]]; then ia=$((ia + 1)); run_arm A "$specA" "$ia" "$@"; else ib=$((ib + 1)); run_arm B "$specB" "$ib" "$@"; fi
  done
done
python3 "$here/compare.py" "$out/A" "$out/B" | tee "$out/compare.txt"
echo "ab dir: $out"
