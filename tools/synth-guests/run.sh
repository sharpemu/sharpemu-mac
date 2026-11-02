#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Usage: run.sh <guest.bin> [label] [--env K=V ...] [--out-dir DIR] [--bin PATH-TO-SharpEmu]
# Runs one headless measurement of a synthetic guest (see harness.py). The run directory (default
# $SYNTH_RUNS/<timestamp>-<guest>-<label>, SYNTH_RUNS defaults to /home/xnetcat/Projects/sharpemu/ds-cpu/synth-runs)
# gets guest.log, perf.csv and metrics.json. Env: SHARPEMU_BIN, RUN_HLE=1, RUN_GC=1, NO_PERF=1, LOG_LEVEL.
set -euo pipefail
export DOTNET_ROOT=${DOTNET_ROOT:-$HOME/.dotnet} PATH=${DOTNET_ROOT:-$HOME/.dotnet}:$PATH
exec python3 "$(dirname "$(readlink -f "$0")")/harness.py" "$@"
