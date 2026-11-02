/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Shared runtime for the synthetic guest programs; see common.h.
 */
#include <stdarg.h>

#include "common.h"

u64 g_tsc_hz = 2000000000ull; /* overwritten by synth_init() */
volatile int g_stop;

static u64 s_measure_start_tsc;
static u64 s_measure_start_us;
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
#define PUT(c) do { if (n + 1 < cap) buf[n] = (c); n++; } while (0)
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
void synth_init(const char *guest_name)
{
    s_name = guest_name;
    /* Calibrate rdtsc against the process clock over ~50 ms (the guest's own time base for all intervals). */
    u64 us0 = now_us(), t0 = rdtsc();
    sceKernelUsleep(50000);
    u64 us1 = now_us(), t1 = rdtsc();
    u64 dus = us1 - us0;
    if (dus) g_tsc_hz = (t1 - t0) * 1000000ull / dus;
    EMIT("calib guest=%s tsc_hz=%lu emu_tsc_hz=%lu", guest_name, g_tsc_hz, sceKernelGetTscFrequency());
}

void warmup_and_start(void)
{
    EMIT("MARK warmup_start guest=%s t_us=%lu warmup_s=%d", s_name, now_us(), WARMUP_S);
    for (int i = 0; i < WARMUP_S * 20; i++) sceKernelUsleep(50000);
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
    EMIT("MARK end guest=%s t_us=%lu measured_us=%lu", s_name, us, us - s_measure_start_us);
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

static u64 bucket_upper(unsigned idx)
{
    if (idx < 8) return idx;
    unsigned msb = idx / 8 + 2, sub = idx % 8;
    return ((u64)(8 + sub + 1) << (msb - 3)) - 1;
}

u64 hist_pct(const Hist *h, unsigned pct)
{
    if (!h->count) return 0;
    u64 want = (h->count * pct + 99) / 100, seen = 0;
    for (unsigned i = 0; i < HIST_BUCKETS; i++) {
        seen += h->b[i];
        if (seen >= want) { u64 u = bucket_upper(i); return u > h->max ? h->max : u; }
    }
    return h->max;
}

void hist_emit_fields(char *buf, unsigned cap, const char *p, const Hist *h)
{
    ksnprintf(buf, cap, "%s_n=%lu %s_avg_ns=%lu %s_p50_ns=%lu %s_p95_ns=%lu %s_p99_ns=%lu %s_max_ns=%lu",
              p, h->count, p, h->count ? h->sum / h->count : 0, p, hist_pct(h, 50), p, hist_pct(h, 95), p, hist_pct(h, 99), p, h->max);
}
