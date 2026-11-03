/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Synthetic (z): the emulator's idle floor. Warm-up, then sceKernelUsleep(100000) in a loop for DURATION_S and
 * return. Everything the host burns during the measured phase is emulator overhead (background JIT tiering, GC,
 * timers, scheduler threads), not guest work: subtract it from the other guests' numbers.
 */
#include "common.h"

#ifndef SLEEP_US
#define SLEEP_US 100000
#endif

int _start(void *args, void *exit_fn)
{
    (void)args; (void)exit_fn;
    synth_init("baseline");
    EMIT("baseline.cfg sleep_us=%d duration_s=%d warmup_s=%d", SLEEP_US, DURATION_S, WARMUP_S);
    warmup_and_start();
    u64 calls = 0;
    while (!run_expired()) { sceKernelUsleep(SLEEP_US); calls++; }
    u64 measured_us = tsc_to_ns(rdtsc() - measure_start_tsc()) / 1000;
    measure_end();
    EMIT("baseline.total measured_us=%lu usleep_calls=%lu", measured_us, calls);
    EMIT("DONE guest=baseline");
    return 0;
}
