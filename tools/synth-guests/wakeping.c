/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Synthetic (c'): wake latency. Two threads ping-pong; the main thread times each round trip.
 *   evf  : sceKernelSetEventFlag(peer, 1) then sceKernelWaitEventFlag(own, 1, AND|CLEAR_ALL, NULL timeout), the
 *          bpe::SyncEvent semantics of p1 section 3 (attr 0x21, bit 1, mode 0x11, no timeout).
 *   cond : scePthreadMutexLock, turn handoff, scePthreadCondSignal(peer), scePthreadCondWait(own) loop.
 * MODE=0 evf only, 1 cond only, 2 both back to back (each DURATION_S/2). PERIOD_US>0 sleeps between round trips
 * (DS does this at <= 100/s, so PERIOD_US=10000 is the "control" rate; 0 = as fast as the emulator can wake).
 */
#include "common.h"

#ifndef MODE
#define MODE 0
#endif
#ifndef PERIOD_US
#define PERIOD_US 0
#endif

static SceKernelEventFlag g_efA, g_efB;
static ScePthreadMutex g_mu;
static ScePthreadCond g_cvA, g_cvB;
static volatile int g_turn; /* 0 = A's turn, 1 = B's */
static volatile int g_phase_stop;
static volatile int g_variant; /* 0 evf, 1 cond: what thread B runs right now */
static volatile int g_b_go;
static volatile u64 g_b_trips;

static void *thread_b(void *arg)
{
    (void)arg;
    for (;;) {
        while (!g_b_go && !g_stop) sceKernelUsleep(5000);
        if (g_stop) return NULL;
        if (g_variant == 0) {
            while (!g_phase_stop) {
                u64 r;
                sceKernelWaitEventFlag(g_efB, 1, EVF_WAIT_AND | EVF_WAIT_CLEAR_ALL, &r, NULL);
                if (g_phase_stop) break;
                sceKernelSetEventFlag(g_efA, 1);
                g_b_trips++;
            }
        } else {
            scePthreadMutexLock(&g_mu);
            while (!g_phase_stop) {
                while (g_turn != 1 && !g_phase_stop) scePthreadCondWait(&g_cvB, &g_mu);
                if (g_phase_stop) break;
                g_turn = 0;
                scePthreadCondSignal(&g_cvA);
                g_b_trips++;
            }
            scePthreadMutexUnlock(&g_mu);
        }
        g_b_go = 0;
    }
}

static void run_variant(int variant, u64 ticks)
{
    static const char *names[] = {"evf", "cond"};
    Hist h;
    memset(&h, 0, sizeof h);
    g_phase_stop = 0;
    g_variant = variant;
    g_turn = 0;
    g_b_trips = 0;
    EMIT("MARK phase_start name=%s t_us=%lu", names[variant], now_us());
    g_b_go = 1;
    u64 start = rdtsc(), trips = 0;
    while (rdtsc() - start < ticks) {
        u64 t0 = rdtsc();
        if (variant == 0) {
            u64 r;
            sceKernelSetEventFlag(g_efB, 1);
            sceKernelWaitEventFlag(g_efA, 1, EVF_WAIT_AND | EVF_WAIT_CLEAR_ALL, &r, NULL);
        } else {
            scePthreadMutexLock(&g_mu);
            g_turn = 1;
            scePthreadCondSignal(&g_cvB);
            while (g_turn != 0) scePthreadCondWait(&g_cvA, &g_mu);
            scePthreadMutexUnlock(&g_mu);
        }
        hist_add(&h, tsc_to_ns(rdtsc() - t0));
        trips++;
        if (PERIOD_US) sceKernelUsleep(PERIOD_US);
    }
    u64 us = tsc_to_ns(rdtsc() - start) / 1000;
    EMIT("MARK phase_end name=%s t_us=%lu", names[variant], now_us());
    /* release B from its wait */
    g_phase_stop = 1;
    if (variant == 0) sceKernelSetEventFlag(g_efB, 1);
    else { scePthreadMutexLock(&g_mu); scePthreadCondSignal(&g_cvB); scePthreadMutexUnlock(&g_mu); }
    while (g_b_go) sceKernelUsleep(1000);
    char buf[320];
    hist_emit_fields(buf, sizeof buf, "rtt", &h);
    EMIT("wakeping.phase name=%s phase_us=%lu round_trips=%lu b_trips=%lu %s", names[variant], us, trips, g_b_trips, buf);
}

int _start(void *args, void *exit_fn)
{
    (void)args; (void)exit_fn;
    synth_init("wakeping");
    EMIT("wakeping.cfg mode=%d period_us=%d duration_s=%d warmup_s=%d", MODE, PERIOD_US, DURATION_S, WARMUP_S);
    sceKernelCreateEventFlag(&g_efA, "pingA", EVF_ATTR_FIFO_MULTI, 0, NULL);
    sceKernelCreateEventFlag(&g_efB, "pingB", EVF_ATTR_FIFO_MULTI, 0, NULL);
    scePthreadMutexInit(&g_mu, NULL, "pingmu");
    scePthreadCondInit(&g_cvA, NULL, "pingA");
    scePthreadCondInit(&g_cvB, NULL, "pingB");
    ScePthread tb;
    thread_spawn(&tb, thread_b, NULL, "WakePingB", 0);
    warmup_and_start();
    if (MODE == 0) run_variant(0, (u64)DURATION_S * g_tsc_hz);
    else if (MODE == 1) run_variant(1, (u64)DURATION_S * g_tsc_hz);
    else { run_variant(0, (u64)DURATION_S * g_tsc_hz / 2); run_variant(1, (u64)DURATION_S * g_tsc_hz / 2); }
    measure_end();
    thread_join(tb);
    EMIT("DONE guest=wakeping");
    return 0;
}
