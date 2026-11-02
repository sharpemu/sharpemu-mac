/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Synthetic (b): the Scream audio guard (p1-static-model.md sections 2 and 5b). Every Scream synth-interface API
 * call is wrapped in a guard (0x1f15b30 / 0x1f15b60):
 *     enter: tid = scePthreadGetthreadid(); if (tid != g_synthTid) scePthreadMutexLock(&g_sndLock);
 *     leave: t   = scePthreadGetthreadid(); if (tid != g_synthTid) scePthreadMutexUnlock(&g_sndLock);
 * i.e. 2 getthreadid per guard and one global-mutex lock+unlock only when the caller is not the synth thread.
 * One "owner" thread (the synth tick, sets g_synthTid) takes the lock-free side; THREADS other threads (job workers
 * and stream readers in DS) take the locked side. NCA jobs (0x1f64230) additionally take a second pool mutex four times.
 *
 * Rates (totals): RATE guards/s (default 193000 = 386k getthreadid/s), OWNER_PCT of them from the owner
 * thread (default 50, so ~97k lock+unlock pairs = ~193k mutex ops/s), NCA_HZ NCA jobs/s (default 3500 = ~14k pool
 * mutex ops/s). RATE=0 runs every thread flat out (finds the maximum the emulator sustains).
 * Pacing is batch-and-sleep: each thread runs the guards due in PERIOD_US back to back, then usleeps until the period
 * ends (spinning only for the last ~1.5 ms, where a usleep would overshoot), so idle time is idle, not a busy-wait.
 *
 * Knobs: THREADS=15 RATE=193000 OWNER_PCT=50 NCA_HZ=3500 NCA_WORK_US=100 CS_NS=500 PERIOD_US=2000 CONTENDED=1
 *        (CONTENDED=0: one sound mutex per thread, no sharing)
 */
#include "common.h"

#ifndef THREADS
#define THREADS 15
#endif
#ifndef RATE
#define RATE 193000
#endif
#ifndef OWNER_PCT
#define OWNER_PCT 50
#endif
#ifndef NCA_HZ
#define NCA_HZ 3500
#endif
#ifndef NCA_WORK_US
#define NCA_WORK_US 100
#endif
#ifndef CS_NS
#define CS_NS 500
#endif
#ifndef PERIOD_US
#define PERIOD_US 2000 /* pacing period: guards due in this window run back to back */
#endif
#ifndef BATCH
#define BATCH 32 /* batch size when RATE=0 (flat out) */
#endif
#ifndef CONTENDED
#define CONTENDED 1
#endif

#define NT (THREADS + 1) /* thread 0 is the owner (synth) thread */

typedef struct TStat {
    u64 guards, gtid, locks, unlocks, nca_jobs;
    Hist gtid_h, lock_h, unlock_h;
} TStat;

static ScePthreadMutex g_snd_lock;                                  /* Scream [0x35832d0] */
static ScePthreadMutex g_nca_lock;                                  /* NCA pool mutex ([ctx]+0x5fe58) */
static ScePthreadMutex g_own_lock[NT] ALIGNED(64);                  /* CONTENDED=0 */
static volatile u32 g_synth_tid;                                    /* Scream [0x2997988] */
static volatile int g_go;
static TStat g_ts[NT];
static TStat g_snap[NT];

typedef struct { int idx; } TArg;
static TArg g_targ[NT];

static void *thread_main(void *arg)
{
    int idx = ((TArg *)arg)->idx;
    int owner = idx == 0;
    TStat *st = &g_ts[idx];
    ScePthreadMutex *snd = CONTENDED ? &g_snd_lock : &g_own_lock[idx];
    const u64 cs_ticks = ns_to_tsc(CS_NS), nca_work_ticks = ns_to_tsc((u64)NCA_WORK_US * 1000), nca_cs_ticks = ns_to_tsc(200);
    /* per-thread guard rate and batch period */
    u64 guards_hz = 0, batch_ticks = 0, nca_acc = 0, nca_step = 0;
    u32 batch = BATCH;
    if (RATE) {
        u64 owner_hz = (u64)RATE * OWNER_PCT / 100;
        guards_hz = owner ? owner_hz : ((u64)RATE - owner_hz) / THREADS;
        if (!guards_hz) guards_hz = 1;
        batch = (u32)(guards_hz * PERIOD_US / 1000000);
        if (batch < 1) batch = 1;
        batch_ticks = (u64)batch * g_tsc_hz / guards_hz;
    }
    /* NCA jobs per batch, as a 16.16 fixed-point accumulator */
    if (!owner && NCA_HZ) {
        u64 per_thread_hz = (u64)NCA_HZ / THREADS;
        nca_step = RATE ? (u64)batch * 65536 * per_thread_hz / guards_hz : 65536 / 16;
    }

    while (!g_go) sceKernelUsleep(5000);
    u64 next = rdtsc() + batch_ticks * (u64)idx / NT; /* stagger the threads' phases so the batches do not all start together */
    while (rdtsc() < next) { }
    while (!g_stop) {
        if (owner) g_synth_tid = scePthreadGetthreadid(); /* synth tick: snd_OSFunc_GetThreadId() -> [0x2997988] */
        for (u32 i = 0; i < batch; i++) {
            u64 t0 = rdtsc();
            u32 tid = scePthreadGetthreadid();
            u64 t1 = rdtsc();
            hist_add(&st->gtid_h, tsc_to_ns(t1 - t0));
            int need = tid != g_synth_tid;
            if (need) {
                scePthreadMutexLock(snd);
                u64 t2 = rdtsc();
                hist_add(&st->lock_h, tsc_to_ns(t2 - t1));
                st->locks++;
            }
            if (cs_ticks) spin_ticks(cs_ticks); /* the guarded API call body */
            u64 t3 = rdtsc();
            (void)scePthreadGetthreadid(); /* the assert in GuardLeave */
            u64 t4 = rdtsc();
            hist_add(&st->gtid_h, tsc_to_ns(t4 - t3));
            if (need) {
                scePthreadMutexUnlock(snd);
                hist_add(&st->unlock_h, tsc_to_ns(rdtsc() - t4));
                st->unlocks++;
            }
            st->gtid += 2;
            st->guards++;
        }
        if (nca_step) {
            nca_acc += nca_step;
            while (nca_acc >= 65536) {
                nca_acc -= 65536;
                scePthreadMutexLock(&g_nca_lock); /* pop a work context */
                spin_ticks(nca_cs_ticks);
                scePthreadMutexUnlock(&g_nca_lock);
                spin_ticks(nca_work_ticks);
                scePthreadMutexLock(&g_nca_lock); /* return it */
                spin_ticks(nca_cs_ticks);
                scePthreadMutexUnlock(&g_nca_lock);
                st->nca_jobs++;
            }
        }
        if (RATE) {
            next += batch_ticks;
            for (;;) {
                u64 t = rdtsc();
                if (t >= next) { if (t - next > 8 * batch_ticks) next = t; break; }
                u64 left_us = tsc_to_ns(next - t) / 1000;
                if (left_us > 1500) sceKernelUsleep((u32)(left_us - 700));
                if (g_stop) break;
            }
        }
    }
    return NULL;
}

int _start(void *args, void *exit_fn)
{
    (void)args; (void)exit_fn;
    synth_init("scream");
    EMIT("scream.cfg threads=%d rate=%d owner_pct=%d nca_hz=%d nca_work_us=%d cs_ns=%d period_us=%d contended=%d duration_s=%d warmup_s=%d",
         THREADS, RATE, OWNER_PCT, NCA_HZ, NCA_WORK_US, CS_NS, PERIOD_US, CONTENDED, DURATION_S, WARMUP_S);
    scePthreadMutexInit(&g_snd_lock, NULL, "sndGlobal");
    scePthreadMutexInit(&g_nca_lock, NULL, "ncaPool");
    for (int i = 0; i < NT; i++) scePthreadMutexInit(&g_own_lock[i], NULL, "own");
    g_synth_tid = 0xffffffffu; /* until the owner publishes its id every thread counts as non-owner */
    ScePthread th[NT];
    static char names[NT][32];
    for (int i = 0; i < NT; i++) {
        g_targ[i].idx = i;
        ksnprintf(names[i], sizeof names[i], i ? "ScreamGuard%d" : "ScreamSynth%d", i);
        thread_spawn(&th[i], thread_main, &g_targ[i], names[i], 0);
    }
    warmup_and_start();
    g_go = 1;
    run_for_sleep();
    for (int i = 0; i < NT; i++) g_snap[i] = g_ts[i];
    u64 measured_us = tsc_to_ns(rdtsc() - measure_start_tsc()) / 1000;
    measure_end();
    for (int i = 0; i < NT; i++) thread_join(th[i]);

    Hist gh = {0}, lh = {0}, uh = {0};
    u64 guards = 0, gtid = 0, locks = 0, unlocks = 0, nca = 0;
    for (int i = 0; i < NT; i++) {
        const TStat *s = &g_snap[i];
        EMIT("scream.thread idx=%d owner=%d guards=%lu gtid=%lu locks=%lu unlocks=%lu nca_jobs=%lu", i, i == 0, s->guards, s->gtid, s->locks, s->unlocks, s->nca_jobs);
        hist_merge(&gh, &s->gtid_h); hist_merge(&lh, &s->lock_h); hist_merge(&uh, &s->unlock_h);
        guards += s->guards; gtid += s->gtid; locks += s->locks; unlocks += s->unlocks; nca += s->nca_jobs;
    }
    u64 mutex_ops = locks + unlocks + nca * 4;
    EMIT("scream.total measured_us=%lu guards=%lu gtid=%lu locks=%lu unlocks=%lu nca_jobs=%lu mutex_ops=%lu gtid_per_s=%lu mutex_ops_per_s=%lu guards_per_s=%lu",
         measured_us, guards, gtid, locks, unlocks, nca, mutex_ops,
         gtid * 1000000 / measured_us, mutex_ops * 1000000 / measured_us, guards * 1000000 / measured_us);
    char b[256];
    hist_emit_fields(b, sizeof b, "gtid", &gh); EMIT("scream.gtid %s", b);
    hist_emit_fields(b, sizeof b, "lock", &lh); EMIT("scream.lock %s", b);
    hist_emit_fields(b, sizeof b, "unlock", &uh); EMIT("scream.unlock %s", b);
    EMIT("DONE guest=scream");
    return 0;
}
