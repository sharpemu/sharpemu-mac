/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Shared runtime for the synthetic guest programs (see README.md).
 *
 * Freestanding PS5-style guest: no libc. Every kernel service is a libkernel import whose symbol name
 * is the NID, so SharpEmu resolves it through the same HLE export a real title would reach.
 */
#ifndef SYNTH_COMMON_H
#define SYNTH_COMMON_H

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed int i32;
typedef signed long long i64;

#define NULL ((void *)0)
#define CACHELINE 64
#define ALIGNED(n) __attribute__((aligned(n)))
#define NOINLINE __attribute__((noinline))

/* Compile-time knobs: every guest takes its defaults from here unless -D<NAME>=<value> overrides. */
#ifndef DURATION_S
#define DURATION_S 20 /* measured phase length (seconds) */
#endif
#ifndef WARMUP_S
#define WARMUP_S 5 /* idle sleep before the measured phase (host JIT warm-up) */
#endif
#ifndef GRACE_MS
#define GRACE_MS 500 /* workers keep running this long after the end marker so the harness can snapshot them */
#endif

/* ---- libkernel imports (NID = symbol name) ------------------------------------------------------------- */
typedef u64 ScePthread;
typedef u64 ScePthreadMutex;
typedef u64 ScePthreadCond;
typedef u64 SceKernelEventFlag;

extern int scePthreadCreate(ScePthread *t, const void *attr, void *(*entry)(void *), void *arg, const char *name) __asm__("6UgtwV+0zb4");
extern int scePthreadJoin(ScePthread t, void **ret) __asm__("onNY9Byn-W8");
extern int scePthreadAttrInit(void *attr) __asm__("nsYoNRywwNg");
extern int scePthreadAttrSetstacksize(void *attr, u64 size) __asm__("UTXzJbWhhTE");
extern int scePthreadAttrSetaffinity(void *attr, u64 mask) __asm__("3qxgM4ezETA");
extern int scePthreadAttrSetdetachstate(void *attr, int state) __asm__("-Wreprtu0Qs");
extern u32 scePthreadGetthreadid(void) __asm__("EI-5-jlq2dE");
extern int scePthreadYield(void) __asm__("T72hz6ffq08");

extern int scePthreadMutexattrInit(void *attr) __asm__("F8bUHwAG284");
extern int scePthreadMutexattrSettype(void *attr, int type) __asm__("iMp8QpE+XO4");
extern int scePthreadMutexInit(ScePthreadMutex *m, const void *attr, const char *name) __asm__("cmo1RIYva9o");
extern int scePthreadMutexLock(ScePthreadMutex *m) __asm__("9UK1vLZQft4");
extern int scePthreadMutexTrylock(ScePthreadMutex *m) __asm__("upoVrzMHFeE");
extern int scePthreadMutexUnlock(ScePthreadMutex *m) __asm__("tn3VlD0hG60");

extern int scePthreadCondInit(ScePthreadCond *c, const void *attr, const char *name) __asm__("2Tb92quprl0");
extern int scePthreadCondWait(ScePthreadCond *c, ScePthreadMutex *m) __asm__("WKAXJ4XBPQ4");
extern int scePthreadCondSignal(ScePthreadCond *c) __asm__("kDh-NfxgMtE");
extern int scePthreadCondBroadcast(ScePthreadCond *c) __asm__("JGgj7Uvrl+A");

extern int sceKernelCreateEventFlag(SceKernelEventFlag *ef, const char *name, u32 attr, u64 init, const void *opt) __asm__("BpFoboUJoZU");
extern int sceKernelSetEventFlag(SceKernelEventFlag ef, u64 bits) __asm__("IOnSvHzqu6A");
extern int sceKernelClearEventFlag(SceKernelEventFlag ef, u64 bits) __asm__("7uhBFWRAS60");
extern int sceKernelWaitEventFlag(SceKernelEventFlag ef, u64 bits, u32 mode, u64 *result, u32 *timeout_us) __asm__("JTvBflhYazQ");
extern int sceKernelDeleteEventFlag(SceKernelEventFlag ef) __asm__("8mql9OcQnd4");

extern int sceKernelUsleep(u32 us) __asm__("1jfXLRVzisc");
extern u64 sceKernelGetProcessTime(void) __asm__("4J2sUJmuHZQ"); /* microseconds */
extern u64 sceKernelGetTscFrequency(void) __asm__("1j3S3n-tTW4");
extern i64 sceKernelWrite(int fd, const void *buf, u64 len) __asm__("FxVZqBAA7ks"); /* _write: fd 1/2 -> host stdout/stderr */

/* Event-flag attributes / wait modes (orbis): FIFO|MULTI = 0x21, AND = 1, OR = 2, CLEAR_ALL = 0x10. */
#define EVF_ATTR_FIFO_MULTI 0x21
#define EVF_WAIT_AND 0x01
#define EVF_WAIT_CLEAR_ALL 0x10

/* ---- time ---------------------------------------------------------------------------------------------- */
static inline u64 rdtsc(void)
{
    u32 lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((u64)hi << 32) | lo;
}
static inline void cpu_pause(void) { __asm__ volatile("pause" ::: "memory"); }

extern u64 g_tsc_hz; /* ticks/s, calibrated against sceKernelGetProcessTime by synth_init() */
/* a*b/c with a 128-bit intermediate. Inline mul/div, NOT unsigned __int128 (that lowers to a __udivti3 import). */
static inline u64 muldiv_u64(u64 a, u64 b, u64 c)
{
    u64 d;
    __asm__("mulq %2\n\tdivq %3" : "+a"(a), "=&d"(d) : "r"(b), "r"(c) : "cc");
    return a;
}
static inline u64 tsc_to_ns(u64 t) { return muldiv_u64(t, 1000000000ull, g_tsc_hz); }
static inline u64 ns_to_tsc(u64 ns) { return muldiv_u64(ns, g_tsc_hz, 1000000000ull); }
static inline u64 now_us(void) { return sceKernelGetProcessTime(); }

/* Burn guest CPU with no import call and no `pause` (a "job body": real work, which spin-nap must not mistake for a
 * spin-wait). */
static inline void spin_ticks(u64 ticks)
{
    u64 end = rdtsc() + ticks;
    while (rdtsc() < end) __asm__ volatile("" ::: "memory");
}
static inline void spin_ns(u64 ns) { spin_ticks(ns_to_tsc(ns)); }

/* ---- output -------------------------------------------------------------------------------------------- */
/* Tiny printf: %s %c %d %u %ld %lu %llu %llx %x %%. Written with one sceKernelWrite per call so lines
 * from different threads do not interleave. */
void pr(const char *fmt, ...);
int ksnprintf(char *buf, unsigned cap, const char *fmt, ...);
u64 str_to_u64(const char *s);

/* Result line protocol, parsed by run.sh: "@@ <section> key=value key=value ...". */
#define EMIT(fmt, ...) pr("@@ " fmt "\n", ##__VA_ARGS__)

/* ---- memory helpers the compiler may emit calls to ----------------------------------------------------- */
void *memset(void *d, int c, unsigned long n);
void *memcpy(void *d, const void *s, unsigned long n);

/* ---- threads ------------------------------------------------------------------------------------------- */
/* Joinable thread with an optional 1<<k affinity mask (DS passes one) and name (host thread name / trace). */
int thread_spawn(ScePthread *t, void *(*fn)(void *), void *arg, const char *name, u64 affinity_mask);
void thread_join(ScePthread t);

/* ---- run control --------------------------------------------------------------------------------------- */
/* Calibrate the TSC against sceKernelGetProcessTime (during the first warm-up sleep) and print "@@ calib". */
void synth_init(const char *guest_name);
/* Sleep WARMUP_S in 50 ms steps, then print "@@ MARK start ..." and stamp the measured phase start. */
void warmup_and_start(void);
/* Measured-phase clock. */
u64 measure_start_tsc(void);
int run_expired(void);          /* nonzero once DURATION_S has elapsed since warmup_and_start() */
void run_for_sleep(void);       /* sleep (in 50 ms steps) until the measured phase is over */
/* Print "@@ MARK end ...", then wait GRACE_MS so workers are still alive while the harness snapshots. */
void measure_end(void);
extern volatile int g_stop;     /* set by measure_end() after the grace period; workers poll it */

/* ---- log-linear latency histogram (ns): 8 sub-buckets per power of two, ~12% resolution ---------------- */
#define HIST_BUCKETS 512
typedef struct {
    u64 count, sum, max;
    u32 b[HIST_BUCKETS];
} Hist;
static inline void hist_add(Hist *h, u64 ns)
{
    unsigned idx;
    if (ns < 8) idx = (unsigned)ns;
    else {
        unsigned msb = 63 - (unsigned)__builtin_clzll(ns);
        idx = (msb - 2) * 8 + (unsigned)((ns >> (msb - 3)) & 7);
    }
    if (idx >= HIST_BUCKETS) idx = HIST_BUCKETS - 1;
    h->b[idx]++;
    h->count++;
    h->sum += ns;
    if (ns > h->max) h->max = ns;
}
void hist_merge(Hist *dst, const Hist *src);
u64 hist_pct(const Hist *h, unsigned pct); /* upper edge of the bucket holding the pct-th percentile */
/* "<prefix>_n=.. <prefix>_avg_ns=.. <prefix>_p50_ns=.. _p95_ns, _p99_ns, _max_ns" fragment for EMIT. */
void hist_emit_fields(char *buf, unsigned cap, const char *prefix, const Hist *h);

#endif
