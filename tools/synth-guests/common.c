/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Shared runtime for the synthetic guest programs; see common.h.
 */
#include <stdarg.h>

#include "common.h"

u64 g_tsc_hz = 2000000000ull; /* overwritten by synth_init() and refined by warmup_and_start() */
u64 g_spin_gap_ticks = 40000;
volatile int g_stop;

static u64 s_measure_start_tsc;
static u64 s_measure_start_us;
static u64 s_cal_us0, s_cal_tsc0; /* clock pair taken at the start of synth_init(), the origin of the long calibration */
static u64 s_prelim_hz;
static const char *s_name = "guest";

/* ---- libc stand-ins (clang lowers struct copies and zero-init to these) -------------------------------- */
void *memset(void *d, int c, unsigned long n)
{
    u8 *p = d;
    while (n--) *p++ = (u8)c;
    return d;
}
void *memcpy(void *d, const void *s, unsigned long n)
{
    u8 *p = d;
    const u8 *q = s;
    while (n--) *p++ = *q++;
    return d;
}

/* ---- formatting ---------------------------------------------------------------------------------------- */
static unsigned fmt_u64(char *out, u64 v, unsigned base)
{
    char tmp[24];
    unsigned n = 0;
    do {
        unsigned d = (unsigned)(v % base);
        tmp[n++] = (char)(d < 10 ? '0' + d : 'a' + d - 10);
        v /= base;
    } while (v);
    for (unsigned i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    return n;
}

static int vfmt(char *buf, unsigned cap, const char *fmt, va_list ap)
{
    unsigned n = 0;
#define PUT(c) do { char c_ = (c); if (n + 1 < cap) buf[n] = c_; n++; } while (0) /* c has side effects: evaluate once, even when truncating */
    for (; *fmt; fmt++) {
        if (*fmt != '%') { PUT(*fmt); continue; }
        fmt++;
        int longs = 0;
        while (*fmt == 'l') { longs++; fmt++; }
        char tmp[24];
        switch (*fmt) {
        case 's': {
            const char *s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            while (*s) PUT(*s++);
            break;
        }
        case 'c': PUT((char)va_arg(ap, int)); break;
        case '%': PUT('%'); break;
        case 'd': {
            i64 v = longs ? va_arg(ap, i64) : (i64)va_arg(ap, i32);
            u64 u = v < 0 ? (u64)0 - (u64)v : (u64)v;
            if (v < 0) PUT('-');
            unsigned k = fmt_u64(tmp, u, 10);
            for (unsigned i = 0; i < k; i++) PUT(tmp[i]);
            break;
        }
        case 'u': case 'x': {
            u64 v = longs ? va_arg(ap, u64) : (u64)va_arg(ap, u32);
            unsigned k = fmt_u64(tmp, v, *fmt == 'x' ? 16 : 10);
            for (unsigned i = 0; i < k; i++) PUT(tmp[i]);
            break;
        }
        default: PUT('?'); break;
        }
    }
    if (cap) buf[n < cap ? n : cap - 1] = 0;
    return (int)n;
#undef PUT
}

int ksnprintf(char *buf, unsigned cap, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vfmt(buf, cap, fmt, ap);
    va_end(ap);
    return n;
}

void pr(const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vfmt(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n >= (int)sizeof buf) n = sizeof buf - 1;
    sceKernelWrite(1, buf, (u64)n);
}

u64 str_to_u64(const char *s)
{
    u64 v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (u64)(*s++ - '0');
    return v;
}

/* ---- threads ------------------------------------------------------------------------------------------- */
int thread_spawn(ScePthread *t, void *(*fn)(void *), void *arg, const char *name, u64 affinity_mask)
{
    u64 attr[16];
    memset(attr, 0, sizeof attr);
    scePthreadAttrInit(attr);
    scePthreadAttrSetstacksize(attr, 0x40000);
    if (affinity_mask) scePthreadAttrSetaffinity(attr, affinity_mask);
    scePthreadAttrSetdetachstate(attr, 0 /* joinable */);
    return scePthreadCreate(t, attr, fn, arg, name);
}

void thread_join(ScePthread t) { scePthreadJoin(t, NULL); }

/* ---- run control --------------------------------------------------------------------------------------- */
/* One (process time, tsc) pair. The import call sits between two rdtsc reads and the pair with the shortest bracket of
 * several tries wins, so the clock value is attributed to the midpoint of the call instead of a cold first-call latency. */
static void clock_pair(u64 *us, u64 *tsc)
{
    u64 best = ~0ull;
    for (int i = 0; i < 16; i++) {
        u64 t0 = rdtsc();
        u64 u = sceKernelGetProcessTime();
        u64 t1 = rdtsc();
        if (t1 - t0 < best) { best = t1 - t0; *us = u; *tsc = t0 + (t1 - t0) / 2; }
    }
}

static void set_tsc_hz(u64 hz)
{
    g_tsc_hz = hz;
    g_spin_gap_ticks = ns_to_tsc(SPIN_GAP_NS);
}

void synth_init(const char *guest_name)
{
    s_name = guest_name;
    u64 u0, t0, u1, t1;
    clock_pair(&u0, &t0); /* also warms the import (first call is far slower than the rest) */
    clock_pair(&u0, &t0);
    sceKernelUsleep(50000);
    clock_pair(&u1, &t1);
    if (u1 > u0) set_tsc_hz(muldiv_u64(t1 - t0, 1000000ull, u1 - u0));
    s_prelim_hz = g_tsc_hz;
    s_cal_us0 = u0;
    s_cal_tsc0 = t0;
    EMIT("calib guest=%s tsc_hz=%lu emu_tsc_hz=%lu", guest_name, g_tsc_hz, sceKernelGetTscFrequency());
}

void warmup_and_start(void)
{
    EMIT("MARK warmup_start guest=%s t_us=%lu warmup_s=%d", s_name, now_us(), WARMUP_S);
    for (int i = 0; i < WARMUP_S * 20; i++) sceKernelUsleep(50000);
    /* Refine the TSC frequency across the whole warm-up (seconds, so the +-1 us clock granularity is ~1e-7), then
     * report it against the emulator's own sceKernelGetTscFrequency(). Anything that needs ticks runs after this. */
    {
        u64 u1, t1;
        clock_pair(&u1, &t1);
        if (u1 - s_cal_us0 >= 1000000ull) set_tsc_hz(muldiv_u64(t1 - s_cal_tsc0, 1000000ull, u1 - s_cal_us0));
        u64 emu = sceKernelGetTscFrequency();
        EMIT("calib2 guest=%s tsc_hz=%lu prelim_hz=%lu emu_tsc_hz=%lu vs_emu_ppm=%ld span_us=%lu",
             s_name, g_tsc_hz, s_prelim_hz, emu, emu ? ((i64)g_tsc_hz - (i64)emu) * 1000000 / (i64)emu : 0, u1 - s_cal_us0);
    }
    s_measure_start_tsc = rdtsc();
    s_measure_start_us = now_us();
    EMIT("MARK start guest=%s t_us=%lu duration_s=%d", s_name, s_measure_start_us, DURATION_S);
}

u64 measure_start_tsc(void) { return s_measure_start_tsc; }

int run_expired(void) { return rdtsc() - s_measure_start_tsc >= (u64)DURATION_S * g_tsc_hz; }

void run_for_sleep(void)
{
    while (!run_expired()) sceKernelUsleep(50000);
}

void measure_end(void)
{
    u64 us = now_us();
    EMIT("MARK end guest=%s t_us=%lu measured_us=%lu grace_ms=%d", s_name, us, us - s_measure_start_us, GRACE_MS);
    for (int i = 0; i < GRACE_MS / 10; i++) sceKernelUsleep(10000);
    g_stop = 1;
}

/* ---- histogram ----------------------------------------------------------------------------------------- */
void hist_merge(Hist *dst, const Hist *src)
{
    for (int i = 0; i < HIST_BUCKETS; i++) dst->b[i] += src->b[i];
    dst->count += src->count;
    dst->sum += src->sum;
    if (src->max > dst->max) dst->max = src->max;
}

static u64 bucket_mid(unsigned idx)
{
    if (idx < HIST_SUB) return idx;
    unsigned msb = idx / HIST_SUB + HIST_SUB_BITS - 1, sub = idx % HIST_SUB, sh = msb - HIST_SUB_BITS;
    u64 lo = (u64)(HIST_SUB + sub) << sh, hi = ((u64)(HIST_SUB + sub + 1) << sh) - 1;
    return lo + (hi - lo) / 2;
}

u64 hist_pct(const Hist *h, unsigned pct)
{
    if (!h->count) return 0;
    u64 want = (h->count * pct + 99) / 100, seen = 0;
    for (unsigned i = 0; i < HIST_BUCKETS; i++) {
        seen += h->b[i];
        if (seen >= want) { u64 m = bucket_mid(i); return m > h->max ? h->max : m; }
    }
    return h->max;
}

/* Host thread name SharpEmu gives a guest thread: "SharpEmu-" + name, cut to 15 chars; spaces become '_' so the value is one token. */
static void host_comm(char *comm, unsigned cap, const char *guest_name)
{
    if (!guest_name) { ksnprintf(comm, cap, "main"); return; }
    ksnprintf(comm, cap, "SharpEmu-%s", guest_name);
    comm[15] = 0;
    for (char *p = comm; *p; p++) if (*p == ' ') *p = '_';
}

void emit_gbusy(const char *guest_name, int idx, u64 work_ticks, u64 pace_ticks)
{
    char comm[40];
    host_comm(comm, sizeof comm, guest_name);
    EMIT("gbusy comm=%s idx=%d work_ns=%lu pace_ns=%lu", comm, idx, tsc_to_ns(work_ticks), tsc_to_ns(pace_ticks));
}

static volatile int s_ident_ready, s_ident_go, s_ident_turn, s_ident_done, s_ident_abort;

void thread_identify(const char *guest_name, int idx)
{
    char comm[40];
    host_comm(comm, sizeof comm, guest_name);
    /* Park first so a host thread's own start-up work (JIT, thunks) is over before the harness's baseline snapshot; the print also
     * warms this thread's first-call path for the import that reports the burn. */
    EMIT("identpark comm=%s idx=%d", comm, idx);
    __atomic_fetch_add(&s_ident_ready, 1, __ATOMIC_ACQ_REL);
    while (!__atomic_load_n(&s_ident_go, __ATOMIC_ACQUIRE)) sceKernelUsleep(20000);
    while (__atomic_load_n(&s_ident_turn, __ATOMIC_ACQUIRE) != idx) {
        if (__atomic_load_n(&s_ident_abort, __ATOMIC_ACQUIRE)) return; /* main gave up (timeout): no identification this run */
        sceKernelUsleep(20000);
    }
    spin_cpu(ns_to_tsc((u64)IDENT_MS * 1000000ull)); /* CPU time, so preemption cannot shorten it; everything else is idle */
    EMIT("MARK ident_end comm=%s idx=%d cpu_ms=%d margin_ms=%d", comm, idx, IDENT_MS, IDENT_MARGIN_MS);
    for (int i = 0; i < IDENT_MARGIN_MS / 20; i++) sceKernelUsleep(20000); /* idle: the harness takes its snapshot in here */
    __atomic_store_n(&s_ident_turn, idx + 1, __ATOMIC_RELEASE);
    __atomic_fetch_add(&s_ident_done, 1, __ATOMIC_ACQ_REL);
}

/* Wait for *p >= n, but give up after IDENT_TIMEOUT_S: a thread that never reaches thread_identify() must not hang the run. */
static int ident_wait(volatile int *p, int n, const char *what)
{
    u64 t0 = now_us();
    while (__atomic_load_n(p, __ATOMIC_ACQUIRE) < n) {
        if (now_us() - t0 > (u64)IDENT_TIMEOUT_S * 1000000ull) {
            EMIT("ERROR ident_timeout while waiting for %s: %d of %d threads after %d s", what, __atomic_load_n(p, __ATOMIC_ACQUIRE), n, IDENT_TIMEOUT_S);
            __atomic_store_n(&s_ident_abort, 1, __ATOMIC_RELEASE);
            __atomic_store_n(&s_ident_go, 1, __ATOMIC_RELEASE);
            return 0;
        }
        sceKernelUsleep(20000);
    }
    return 1;
}

void wait_identified(int n)
{
    if (!ident_wait(&s_ident_ready, n, "threads to park in thread_identify")) return;
    for (int i = 0; i < 15; i++) sceKernelUsleep(20000); /* parked threads' start-up CPU dies down */
    EMIT("MARK ident_begin n=%d margin_ms=%d", n, IDENT_MARGIN_MS); /* harness: baseline per-thread CPU */
    for (int i = 0; i < IDENT_MARGIN_MS / 20; i++) sceKernelUsleep(20000);
    __atomic_store_n(&s_ident_go, 1, __ATOMIC_RELEASE);
    ident_wait(&s_ident_done, n, "threads to finish their identification burn");
}

void hist_emit_fields(char *buf, unsigned cap, const char *p, const Hist *h)
{
    ksnprintf(buf, cap, "%s_n=%lu %s_avg_ns=%lu %s_p50_ns=%lu %s_p95_ns=%lu %s_p99_ns=%lu %s_max_ns=%lu",
              p, h->count, p, h->count ? h->sum / h->count : 0, p, hist_pct(h, 50), p, hist_pct(h, 95), p, hist_pct(h, 99), p, h->max);
}
