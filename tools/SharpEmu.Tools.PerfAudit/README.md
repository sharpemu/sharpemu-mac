<!-- Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later -->

# Synthetic runtime benchmark suite

This packages the CPU/GPU audit harness previously kept under local artifacts.
It generates its own command streams, shader instructions, descriptors and buffers;
no game executable, captured shader or external asset is loaded. macOS and an x64
.NET 10 host are required for the complete suite. Device groups use real Metal.
The executable has its own assembly name and cannot overwrite the Metal test DLL.

```sh
~/.dotnet-x64/dotnet build tools/SharpEmu.Tools.PerfAudit -c Release -m:2 -p:UseSharedCompilation=false -nr:false
# Short runtime check (memory/lookup groups also use Metal); not a timing acceptance run.
~/.dotnet-x64/dotnet artifacts/bin/Release/net10.0/SharpEmu.Tools.PerfAudit.dll pm4,plan,mat,queue,mem,lookup,misc,w8,xlate,wranges --quick
# Fabricated triangle through the real Metal backend.
~/.dotnet-x64/dotnet artifacts/bin/Release/net10.0/SharpEmu.Tools.PerfAudit.dll smoke
# Warm translator comparison; run each source revision in alternating order.
~/.dotnet-x64/dotnet artifacts/bin/Release/net10.0/SharpEmu.Tools.PerfAudit.dll xlate --warmup=5000
```

| Selector | Work measured |
|---|---|
| `pm4` | Register snapshots, command interpretation and index expansion |
| `plan` | Draw/dispatch planning with fabricated host services |
| `mat` | Resource materialization, cold/stale/hit/ring cases |
| `queue` | Command queue and worker handoffs |
| `mem` | Guest memory translation, protection and copying |
| `lookup` | Pipeline identities, keys and dictionary probes |
| `misc` | Monitor and other small runtime costs |
| `w8` | Staged range/cache behavior |
| `xlate` | Single, fragmented and strided guest view layouts |
| `wranges` | GPU-written range tracking |
| `labels` | Held-label overlap lookup, exact scan versus bounded page filter |
| `shmap` | Dictionary versus standalone dense shader-register map candidate; shared interface dispatch |
| `metal` | Raw Metal binding/encoding and submission timings |
| `e2e` | Generated textured draws through the interpreter and Metal backend |
| `gword` | Synthetic GPU-word wait path |
| `smoke` | One generated triangle and pixel readback |

`all` selects the original broad suite; `xlate`, `wranges` and `gword` are explicit
additional selectors. `e2e` accepts `--scenarios=ID,...`, `--soak=SECONDS` and
`--allocs`; inspect its source for scenario IDs. `--quick` reduces warmup and
sampling for smoke checks only. The normal harness uses five 150 ms samples and
700 ms minimum warmup; use `--warmup=5000` for comparisons. Report the process
architecture, runtime, source revisions, tiering/PGO and switches with results.

Run one benchmark/build/test process at a time on a quiet machine; follow the
shared validation lock/queue when coordinating workers. Warm both arms, alternate
A/B/B/A, retain every sample and allocation count, and compare controls for drift.
The printed p95 is a percentile of run averages, not individual operation latency.
The scaled-cost column uses illustrative counts (6,331 draws / 2,700 dispatches);
it does not measure a game frame, predict FPS, or account for overlapping work.

For the final cold-admission A/B experiment, use the dedicated
[materialization harness](../SharpEmu.Tools.MaterializationBench/README.md), whose
sampling and counters match the published final result. Historical measurements
were made with the corresponding historical source, not this newly packaged binary.
See [benchmark results and scope](../../docs/performance/benchmark-index.md).

The `labels` selector uses fixed parameters: 20,000 warmup pairs followed by four
alternating trials of 100,000 lookups for each list size and overlap shape. It
prints nanoseconds and allocations per lookup plus filter insertion/reset cost;
`--quick` does not alter this selector. To match the native ARM64 lookup experiment,
publish this tool for `osx-arm64` with `--self-contained true` and
`-p:PublishReadyToRun=false`, then run the resulting executable with
`DOTNET_TieredCompilation=0` and selector `labels`. The original measurement used
a standalone build of the same benchmark body linked to `MetalPageRangeFilter.cs`;
packaged-tool validation is recorded separately.
