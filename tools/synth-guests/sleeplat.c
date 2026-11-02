/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Synthetic (d): sleep/yield latency as the guest sees it. For each N in SWEEP_N, sceKernelUsleep(N) called
 * back-to-back (the nap policy's "streak" case) and again with GAP_US of guest busy-work between calls (each call
 * then starts a fresh streak, so it takes the exact-sleep path); then scePthreadYield back-to-back and gapped.
 * Every phase gets PHASE_S seconds (DURATION_S split evenly) and is bracketed by "@@ MARK phase_start/phase_end" lines
 * so the harness can attribute host CPU to it. Per phase the guest prints calls and elapsed-per-call p50/p95/max.
 *
 * Knobs: SWEEP_N="0,1,50,100,250,1000,4000"  GAP_US=2000  YIELD=1 (0 skips the yield phases)
 *        ONLY_N=<n> (single N for the whole DURATION_S)  ONLY_GAP=0|1 (with ONLY_N: back-to-back or gapped)
 */
#include "common.h"

#ifndef SWEEP_N
#define SWEEP_N 0, 1, 50, 100, 250, 1000, 4000
#endif
#ifndef GAP_US
#define GAP_US 2000
#endif
#ifndef YIELD
#define YIELD 1
#endif
#ifndef ONLY_GAP
#define ONLY_GAP 0
#endif

static const u32 k_sweep[] = {SWEEP_N};
#define NSWEEP ((int)(sizeof k_sweep / sizeof k_sweep[0]))

static Hist g_h;
static u64 g_gap_ticks_spent; /* CPU the guest itself burns in the gap spins */

static void phase(const char *kind, u32 n_us, int gapped, u64 phase_ticks)
{
    memset(&g_h, 0, sizeof g_h);
    char name[48];
    ksnprintf(name, sizeof name, "%s%u_%s", kind, n_us, gapped ? "gap" : "b2b");
    EMIT("MARK phase_start name=%s t_us=%lu", name, now_us());
    u64 start = rdtsc(), gap_ticks = ns_to_tsc((u64)GAP_US * 1000);
    int yield = kind[0] == 'y';
    while (rdtsc() - start < phase_ticks) {
        u64 t0 = rdtsc();
        if (yield) scePthreadYield(); else sceKernelUsleep(n_us);
        hist_add(&g_h, tsc_to_ns(rdtsc() - t0));
        if (gapped) g_gap_ticks_spent += spin_cpu(gap_ticks);
    }
    u64 us = tsc_to_ns(rdtsc() - start) / 1000;
    char buf[320];
    hist_emit_fields(buf, sizeof buf, "elapsed", &g_h);
    EMIT("MARK phase_end name=%s t_us=%lu", name, now_us());
    EMIT("sleeplat.phase name=%s kind=%s n_us=%u gapped=%d phase_us=%lu calls=%lu %s", name, yield ? "yield" : "usleep", n_us, gapped, us, g_h.count, buf);
}

int _start(void *args, void *exit_fn)
{
    (void)args; (void)exit_fn;
    synth_init("sleeplat");
    EMIT("sleeplat.cfg gap_us=%d duration_s=%d warmup_s=%d", GAP_US, DURATION_S, WARMUP_S);
    warmup_and_start();
#ifdef ONLY_N
    phase("usleep", ONLY_N, ONLY_GAP, (u64)DURATION_S * g_tsc_hz);
#else
    int nphases = NSWEEP * 2 + (YIELD ? 2 : 0);
    u64 phase_ticks = (u64)DURATION_S * g_tsc_hz / (u64)nphases;
    for (int i = 0; i < NSWEEP; i++) {
        phase("usleep", k_sweep[i], 0, phase_ticks);
        phase("usleep", k_sweep[i], 1, phase_ticks);
    }
    if (YIELD) {
        phase("yield", 0, 0, phase_ticks);
        phase("yield", 0, 1, phase_ticks);
    }
#endif
    emit_gbusy(NULL, 0, g_gap_ticks_spent, 0);
    measure_end();
    EMIT("DONE guest=sleeplat");
    return 0;
}
