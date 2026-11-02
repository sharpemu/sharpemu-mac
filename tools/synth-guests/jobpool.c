/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Synthetic (a): the Demon's Souls job pool. WORKERS threads poll a scheduler laid out at DS's offsets
 * (p1-static-model.md section 0/5a): 8 priority groups, per group 15 Chase-Lev deques, 256-slot external (B) and
 * parked (A) arrays with lock words, count[g][w], per-worker mailboxes. Idle workers run the SAME polling code
 * shape as DS:
 *   - ChooseJobToRun (0x86ecd0): while any count[g][w] > 0 (parked jobs keep it > 0) rescan 2x256 slots with
 *     lock cmpxchg 1->2 / xchg on shared lock words, plus the 15 deques. No pause, no yield, no sleep.
 *   - WaitForJobs (0x86ea30): when all counts are 0, a pause loop that calls an rdtsc wrapper (so SharpEmu's
 *     default spin-nap patcher rejects it as non_loop); after 2 s without a job it takes dlLock every pass.
 * The slot scan and the WaitForJobs loop are written in assembly so the instruction shapes (locked backward loop
 * <= 256 B with no call; pause window containing a call) are exactly DS's, whatever the compiler does.
 *
 * A producer (the main thread) plays the game thread: every frame it releases last frame's parked jobs, submits
 * PRODUCER_HZ/FRAME_HZ ready jobs into group 0's B array, parks PARKED new jobs in group 0's A array (their wait
 * condition stays false for a whole frame), then waits for the ready jobs with DS's non-worker WaitForCounter
 * idiom (pause + usleep(1) per 1000 polls) and sleeps out the rest of the frame.
 *
 * NAP=1 (default) adds DS's shipped idle nap (src/SharpEmu.Core/Loader/DemonsSoulsIdleNapPatch.cs), same hooks, same arguments:
 *   T2  after every RunJob: stamp the worker's slot [S+0xF00+8w] with the TSC at job end;
 *   T1  around ChooseJobToRun (its call in WorkerMain): when no job was found, T3 with r12 = the worker's slot;
 *   T3  replaces WaitForJobs' TSC read (and is what T1 calls): rdtsc; if now - slot >= threshold, call the private import
 *       SharpEmuNap(NAP_US) (a host usleep); return the pre-nap TSC either way.
 * The patch computes threshold = NAP_IDLE_US * sceKernelGetTscFrequency() at load time (the loader reads SHARPEMU_DS_IDLE_NAP_IDLE_US /
 * SHARPEMU_DS_IDLE_NAP_US, only for the DS title), so here they are -D knobs with the same defaults: -DNAP_IDLE_US= -DNAP_US=.
 *
 * Knobs (-D): WORKERS PARKED PRODUCER_HZ JOB_US FRAME_HZ AFFINITY PIN_WORKERS FORK DL_CHECK NAP NAP_IDLE_US NAP_US LIGHT; see README.md.
 */
#include "common.h"

#ifndef WORKERS
#define WORKERS 13
#endif
#ifndef PARKED
#define PARKED 3 /* parked jobs alive at any time in group 0 (0 = pure WaitForJobs pause path) */
#endif
#ifdef LIGHT /* -DLIGHT=1: unsaturated preset, ~1.35 cores of job work (4500 jobs/s x 300 us) instead of ~4.5 */
#ifndef PRODUCER_HZ
#define PRODUCER_HZ 4500
#endif
#endif
#ifndef PRODUCER_HZ
#define PRODUCER_HZ 15000 /* ready-job submissions per second (250 per 60 Hz frame; <= 256 fit the B array) */
#endif
#ifndef JOB_US
#define JOB_US 300 /* busy work per job */
#endif
#ifndef FRAME_HZ
#define FRAME_HZ 60
#endif
#ifndef AFFINITY
#define AFFINITY 0x1ffe /* job affinity mask: worker 0 never gets work (stays in WaitForJobs) */
#endif
#ifndef PIN_WORKERS
#define PIN_WORKERS 1 /* pass 1<<k as the thread affinity attribute like DS does */
#endif
#ifndef FORK
#define FORK 0 /* children a job pushes on its own deque (worker-side submit path) */
#endif
#ifndef NAP
#define NAP 1 /* DS's own idle nap (DemonsSoulsIdleNapPatch): on by default in DS; NAP=0 models SHARPEMU_DS_IDLE_NAP=0 */
#endif
#ifndef NAP_IDLE_US
#define NAP_IDLE_US 500 /* SHARPEMU_DS_IDLE_NAP_IDLE_US default: nap once idle this long since the worker's last job */
#endif
#ifndef NAP_US
#define NAP_US 250 /* SHARPEMU_DS_IDLE_NAP_US default: the nap length passed to SharpEmuNap */
#endif
#ifndef DL_CHECK
#define DL_CHECK 1 /* 0 disables the 2 s dlLock/CheckForDeadlock path in WaitForJobs */
#endif
#ifndef GROUP
#define GROUP 0 /* priority group the producer's jobs go to */
#endif

#define NG 8
#define NQ 15
#define SLOTS 256
#define NJOBS 4096
#define HIDDEN __attribute__((visibility("hidden")))

#if WORKERS < 1 || WORKERS > 15
#error WORKERS must be 1..15
#endif

/* ---- scheduler layout (offsets identical to DS) -------------------------------------------------------- */
typedef struct Job {
    volatile i64 *c0_ptr; /* wait condition: *ptr <op> target (op 3 = ">=", 1 = "<=") */
    i64 c0_target;
    int c0_op;
    u32 affinity;
    u32 grp;
    u32 flags;
    u64 enq_tsc;
    u64 work_ns;
    volatile i64 *done_ctr; /* completion counter, lock xadd -1 at job end */
    volatile u32 state;     /* 0 free, 1 in flight */
    u32 parked;
} Job;

typedef struct Deque {
    u32 mask;
    u32 _p;
    u64 _p2;
    volatile i64 bottom; /* +0x10 */
    volatile i64 top;    /* +0x18 */
    Job *volatile items[SLOTS];
} Deque; /* 0x820 */

typedef struct Group {
    Deque dq[NQ];             /* +0x0 */
    Job *volatile B[SLOTS];   /* +0x79e0 external / affinity-mismatch jobs */
    volatile u32 Blk[SLOTS];  /* +0x81e0 */
    Job *volatile A[SLOTS];   /* +0x85e0 parked jobs */
    volatile u32 Alk[SLOTS];  /* +0x8de0 */
    volatile i32 count[NQ];   /* +0x91e0 */
    u32 _pad;
} Group; /* 0x9220 */

typedef struct Sched {
    volatile u8 exit;                  /* +0x0 */
    u8 _p0[0xef8 - 1];
    volatile u64 fiberLock;            /* +0xef8 (unused: no fibers) */
    volatile u64 lastJobTsc[15];       /* +0xf00 */
    volatile u64 dlLock;               /* +0xf78 */
    Job *volatile mailbox[15];         /* +0xf80 */
    u32 ngroups;                       /* +0xff8 */
    u32 _p2;
    Group *groups;                     /* +0x1000 */
} Sched;

_Static_assert(__builtin_offsetof(Group, B) == 0x79e0, "B");
_Static_assert(__builtin_offsetof(Group, Blk) == 0x81e0, "Blk");
_Static_assert(__builtin_offsetof(Group, A) == 0x85e0, "A");
_Static_assert(__builtin_offsetof(Group, Alk) == 0x8de0, "Alk");
_Static_assert(__builtin_offsetof(Group, count) == 0x91e0, "count");
_Static_assert(sizeof(Group) == 0x9220, "Group");
_Static_assert(__builtin_offsetof(Sched, lastJobTsc) == 0xf00, "lastJobTsc");
_Static_assert(__builtin_offsetof(Sched, dlLock) == 0xf78, "dlLock");
_Static_assert(__builtin_offsetof(Sched, mailbox) == 0xf80, "mailbox");
_Static_assert(__builtin_offsetof(Sched, ngroups) == 0xff8, "ngroups");
_Static_assert(__builtin_offsetof(Sched, groups) == 0x1000, "groups");

static Sched g_sched ALIGNED(4096);
static Group g_groups[NG] ALIGNED(64); /* contiguous, stride 0x9220: G+0x91e0 straddles cache lines like DS */
static Job g_jobs[NJOBS] ALIGNED(64);
static volatile u32 g_job_next;
/* One monotonic counter per parked-job slot: a parked job waits for ctr >= its own generation, the release is a store of that
 * generation, and a counter is never reset, so an old job can never be starved or aliased by a later one. */
static volatile i64 g_park_ctr[PARKED > 0 ? PARKED : 1] ALIGNED(64);
static i64 g_park_gen[PARKED > 0 ? PARKED : 1];
static volatile int g_go;

typedef struct Wstat {
    u64 pauses; /* first: the asm and t3_nap find the Wstat from &pauses */
    u64 polls, jobs, pred_calls, dl_checks, forks, stolen;
    u64 naps, nap_ticks, nap_max_ticks; /* SharpEmuNap calls, total and longest guest-observed time in them */
    Hist lat;  /* enqueue -> start, ready jobs */
    Hist plat; /* release -> start, parked jobs */
    u64 work_ticks; /* CPU ticks spent in job bodies (spin_cpu), excluding descheduled gaps */
} Wstat; /* ~20 KiB each (the histograms), so the hot counters of neighbouring workers never share a line */
static Wstat g_ws[WORKERS];
static Wstat g_snap[WORKERS];

/* ---- helpers ------------------------------------------------------------------------------------------- */
static inline int cmp_op(i64 v, i64 target, int op)
{
    switch (op) {
    case 0: return v == target;
    case 1: return v <= target;
    case 2: return v < target;
    case 3: return v >= target;
    default: return v > target;
    }
}

static inline int job_ready(Job *j) /* 0x86d720 */
{
    if (j->c0_ptr) {
        if (!cmp_op(*j->c0_ptr, j->c0_target, j->c0_op)) return 0;
        j->c0_ptr = NULL;
    }
    return 1;
}

HIDDEN NOINLINE u64 rdtsc_call(void) { return rdtsc(); } /* 0x863a20: the call that sits inside WaitForJobs' pause loop */

static void spinlock(volatile u64 *l) /* lock cmpxchg with 10001 x pause then usleep(1) backoff (0x86eb73 idiom) */
{
    for (u32 n = 0;;) {
        while (*l) {
            if (++n >= 10001) { sceKernelUsleep(1); n = 0; }
            cpu_pause();
        }
        if (__sync_bool_compare_and_swap(l, 0, ~0ull)) break;
    }
    __builtin_ia32_lfence();
}
static void spinunlock(volatile u64 *l) { __builtin_ia32_sfence(); __atomic_exchange_n(l, 0, __ATOMIC_SEQ_CST); }

NOINLINE static int insert_slot(Job *volatile *P, volatile u32 *L, Job *j) /* cmpxchg 0->2, store, xchg 1 */
{
#pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < SLOTS; i++) {
        if (L[i] != 0) continue;
        if (!__sync_bool_compare_and_swap(&L[i], 0, 2)) continue;
        P[i] = j;
        __atomic_exchange_n(&L[i], 1, __ATOMIC_SEQ_CST);
        return i;
    }
    return -1;
}

NOINLINE static void counts_inc(Group *G, u32 affinity)
{
#pragma clang loop unroll(disable) vectorize(disable)
    for (int b = 0; b < WORKERS; b++)
        if (affinity >> b & 1) __atomic_fetch_add(&G->count[b], 1, __ATOMIC_SEQ_CST); /* lock xadd */
}

NOINLINE static void take_counts(Group *G, Job *j) /* 0x86f3c5: lock xadd -1 per eligible worker */
{
#pragma clang loop unroll(disable) vectorize(disable)
    for (int b = 0; b < WORKERS; b++)
        if (j->affinity >> b & 1) __atomic_fetch_add(&G->count[b], -1, __ATOMIC_SEQ_CST);
}

static Job *alloc_job(void)
{
    for (;;) {
        u32 i = __atomic_fetch_add(&g_job_next, 1, __ATOMIC_RELAXED) % NJOBS;
        Job *j = &g_jobs[i];
        u32 expect = 0;
        if (__atomic_compare_exchange_n(&j->state, &expect, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            j->c0_ptr = NULL; j->parked = 0; j->flags = 0;
            return j;
        }
        cpu_pause();
    }
}

/* ---- the DS idle path: assembly so the shapes cannot drift ---------------------------------------------- */
/* Called by scan_slots after each successful slot lock (the DS `call r8` = 0x870b80): job ready AND this worker may run it. */
HIDDEN int job_pred(Sched *S, Job *volatile *slot, u32 wbit, u32 kind)
{
    (void)S; (void)kind;
    g_ws[__builtin_ctz(wbit)].pred_calls++;
    Job *j = *slot;
    return j && job_ready(j) && (j->affinity & wbit);
}
HIDDEN int (*g_pred_fn)(Sched *, Job *volatile *, u32, u32) = job_pred;
HIDDEN void dl_check(Sched *S, int w) /* locked CheckForDeadlock (cheap form): scan every worker's lastJobTsc */
{
    spinlock(&S->dlLock);
    u64 oldest = ~0ull;
    for (int i = 0; i < WORKERS; i++) if (S->lastJobTsc[i] < oldest) oldest = S->lastJobTsc[i];
    g_ws[w].dl_checks += 1 + (oldest == ~0ull);
    spinunlock(&S->dlLock);
}
HIDDEN u64 g_two_tscf;

/* ---- DS's idle nap (see the header comment) -------------------------------------------------------------- */
extern void SharpEmuNap(u32 microseconds) __asm__("SharpEmuNap"); /* private SharpEmu import (IdleNapExports.Nid) */
static i64 g_nap_thr; /* ticks: NAP_IDLE_US * sceKernelGetTscFrequency(), clamped to int32 like the patch's imm32 */

/* T3: rdtsc; nap when now - slot >= threshold; returns the pre-nap TSC. `pauses` is the calling worker's &Wstat.pauses. */
HIDDEN u64 t3_nap(u64 slot, u64 *pauses)
{
    u64 now = rdtsc();
    if ((i64)(now - slot) >= g_nap_thr) {
        Wstat *st = (Wstat *)((char *)pauses - __builtin_offsetof(Wstat, pauses));
        SharpEmuNap(NAP_US);
        u64 d = rdtsc() - now;
        st->naps++;
        st->nap_ticks += d;
        if (d > st->nap_max_ticks) st->nap_max_ticks = d;
    }
    return now;
}

/* Job *scan_slots(Group *G, u64 offP, u64 offL, u32 wbit, Sched *S, u32 kind)
 * 0x86ee12..0x86eeb7 (A) / 0x86eecd..0x86ef78 (B): for i in 0..255: if P[i] { lock cmpxchg L[i] 1->2; if pred(job) take it
 * (slot=0, xchg L[i],0) else xchg L[i],1 }. The cmpxchg's backward `jne` closes a 36 B call-free loop. */
HIDDEN Job *scan_slots(Group *G, u64 offP, u64 offL, u32 wbit, Sched *S, u32 kind);
/* void wait_for_jobs(Sched *S, int w, u64 *pauses) : 0x86eac0..0x86ec64 */
HIDDEN void wait_for_jobs(Sched *S, int w, u64 *pauses);

#if NAP
/* T3 replaces WaitForJobs' TSC read exactly as in DS: r12 already holds the worker's slot. */
#define WFJ_RDTSC "  mov rdi,r12\n  mov rsi,r13\n  call t3_nap\n"
#else
#define WFJ_RDTSC "  call rdtsc_call\n"
#endif
__asm__(
    ".intel_syntax noprefix\n"
    ".text\n"
    ".globl scan_slots\n.hidden scan_slots\n.type scan_slots,@function\n.p2align 4\n"
    "scan_slots:\n"
    ".cfi_startproc\n"
    "  push rbp\n  .cfi_def_cfa_offset 16\n  .cfi_offset rbp,-16\n"
    "  mov rbp,rsp\n  .cfi_def_cfa_register rbp\n"
    "  push r15\n  push r14\n  push r13\n  push rbx\n  sub rsp,48\n"
    "  .cfi_offset r15,-24\n  .cfi_offset r14,-32\n  .cfi_offset r13,-40\n  .cfi_offset rbx,-48\n"
    "  mov r14,rdi\n"            /* G */
    "  mov r15,rsi\n"            /* offP + 8i */
    "  mov r13,rdx\n"            /* offL + 4i */
    "  mov dword ptr [rbp-0x34],ecx\n"  /* wbit */
    "  mov qword ptr [rbp-0x40],r8\n"   /* S */
    "  mov dword ptr [rbp-0x44],r9d\n"  /* kind */
    "  mov ebx,0x100\n"
    "  mov ecx,2\n"
    "  cmp qword ptr [r14+r15],0\n"
    "  jne 3f\n"
    "  jmp 2f\n"
    "1:\n"                             /* predicate false: release the lock, state 2 -> 1 */
    "  mov eax,1\n"
    "  mov ecx,2\n"
    "  xchg dword ptr [r14+r13],eax\n"
    "  xchg ax,ax\n"
    "2:\n"
    "  add r13,4\n"
    "  add r15,8\n"
    "  dec ebx\n"
    "  je 8f\n"
    "  cmp qword ptr [r14+r15],0\n"
    "  je 2b\n"
    "3:\n"
    "  mov eax,1\n"
    "  lock cmpxchg dword ptr [r14+r13],ecx\n"
    "  jne 2b\n"
    "  mov rdi,qword ptr [rbp-0x40]\n"
    "  lea rsi,[r14+r15]\n"
    "  mov qword ptr [rbp-0x50],rsi\n"
    "  mov edx,dword ptr [rbp-0x34]\n"
    "  mov ecx,dword ptr [rbp-0x44]\n"
    "  mov r8,qword ptr [rip+g_pred_fn]\n"
    "  call r8\n"
    "  test al,al\n"
    "  je 1b\n"
    "  mov rax,qword ptr [rbp-0x50]\n"
    "  mov ecx,2\n"
    "  mov rdx,qword ptr [rax]\n"
    "  test rdx,rdx\n"
    "  jne 9f\n"
    "  xor eax,eax\n"
    "  xchg dword ptr [r14+r13],eax\n"
    "  add r13,4\n"
    "  add r15,8\n"
    "  dec ebx\n"
    "  je 8f\n"
    "  cmp qword ptr [r14+r15],0\n"
    "  je 2b\n"
    "  jmp 3b\n"
    "8:\n"                              /* nothing taken */
    "  xor eax,eax\n"
    "  jmp 7f\n"
    "9:\n"                              /* take: slot=0 ; xchg lock,0 */
    "  mov qword ptr [rax],0\n"
    "  xor eax,eax\n"
    "  xchg dword ptr [r14+r13],eax\n"
    "  mov rax,rdx\n"
    "7:\n"
    "  add rsp,48\n  pop rbx\n  pop r13\n  pop r14\n  pop r15\n  pop rbp\n"
    "  .cfi_def_cfa rsp,8\n"
    "  ret\n"
    ".cfi_endproc\n"
    ".size scan_slots,.-scan_slots\n"

    /* WaitForJobs: returns when any count[g][w] > 0 or mailbox[w] != 0 or exit.
     *   rdi = S, esi = w, rdx = &pauses */
    ".globl wait_for_jobs\n.hidden wait_for_jobs\n.type wait_for_jobs,@function\n.p2align 4\n"
    "wait_for_jobs:\n"
    ".cfi_startproc\n"
    "  push rbp\n  .cfi_def_cfa_offset 16\n  .cfi_offset rbp,-16\n"
    "  mov rbp,rsp\n  .cfi_def_cfa_register rbp\n"
    "  push r15\n  push r14\n  push r13\n  push r12\n  push rbx\n  sub rsp,8\n"
    "  .cfi_offset r15,-24\n  .cfi_offset r14,-32\n  .cfi_offset r13,-40\n  .cfi_offset r12,-48\n  .cfi_offset rbx,-56\n"
    "  mov rbx,rdi\n"
    "  mov r14d,esi\n"
    "  mov r13,rdx\n"
    "  mov r15,qword ptr [rip+g_two_tscf]\n"
    "1:\n"
    "  cmp byte ptr [rbx],0\n"
    "  jne 9f\n"
    "  mov eax,dword ptr [rbx+0xff8]\n"
    "  test rax,rax\n"
    "  je 3f\n"
    "  mov rcx,qword ptr [rbx+0x1000]\n"
    "  lea rcx,[rcx+r14*4+0x91e0]\n"
    "  imul rax,rax,0x9220\n"
    "  xor edx,edx\n"
    "  .p2align 4\n"
    "2:\n"
    "  cmp dword ptr [rcx+rdx],0\n"
    "  jg 9f\n"
    "  cmp qword ptr [rbx+r14*8+0xf80],0\n"
    "  jne 9f\n"
    "  add rdx,0x9220\n"
    "  cmp rax,rdx\n"
    "  jne 2b\n"
    "3:\n"
    "  pause\n"                         /* the only pause on the idle path: a call follows within the window */
    "  inc qword ptr [r13]\n"
    "  mov r12,qword ptr [rbx+r14*8+0xf00]\n"
    WFJ_RDTSC
    "  sub rax,r12\n"
    "  test rax,rax\n"
    "  jle 1b\n"
    "  cmp rax,r15\n"
    "  jbe 1b\n"
    "  mov rdi,rbx\n"
    "  mov esi,r14d\n"
    "  call dl_check\n"
    "  jmp 1b\n"
    "9:\n"
    "  add rsp,8\n  pop rbx\n  pop r12\n  pop r13\n  pop r14\n  pop r15\n  pop rbp\n"
    "  .cfi_def_cfa rsp,8\n"
    "  ret\n"
    ".cfi_endproc\n"
    ".size wait_for_jobs,.-wait_for_jobs\n"
    ".att_syntax prefix\n");

/* ---- ChooseJobToRun (0x86ecd0) -------------------------------------------------------------------------- */
static Job *deque_pop(Deque *D)
{
    i64 b = D->bottom;
    __atomic_exchange_n(&D->bottom, b - 1, __ATOMIC_SEQ_CST); /* xchg [D+0x10] */
    i64 t = D->top;
    if (t >= b) { D->bottom = t; return NULL; }
    Job *j = D->items[(b - 1) & D->mask];
    if (t == b - 1) {
        int ok = __sync_bool_compare_and_swap(&D->top, t, b); /* lock cmpxchg */
        D->bottom = b;
        if (!ok) return NULL;
    }
    return j;
}
static Job *deque_steal(Deque *D)
{
    i64 t = D->top;
    __builtin_ia32_lfence();
    if (t >= D->bottom) return NULL;
    Job *j = D->items[t & D->mask];
    if (!__sync_bool_compare_and_swap(&D->top, t, t + 1)) return NULL;
    return j;
}
static void deque_push(Deque *D, Job *j)
{
    i64 b = D->bottom;
    D->items[b & D->mask] = j;
    __builtin_ia32_sfence();
    D->bottom = b + 1;
}

static int choose(Sched *S, int w, Job **out)
{
    Wstat *st = &g_ws[w];
    Job *m = S->mailbox[w];
    if (m) { *out = m; S->mailbox[w] = NULL; return 1; }
    u32 wbit = 1u << w;
    for (u32 g = 0; g < S->ngroups; g++) {
        Group *G = &S->groups[g];
        if (G->count[w] == 0) continue;
        Job *j = scan_slots(G, __builtin_offsetof(Group, A), __builtin_offsetof(Group, Alk), wbit, S, 0);
        if (!j) j = scan_slots(G, __builtin_offsetof(Group, B), __builtin_offsetof(Group, Blk), wbit, S, 1);
        if (j) { take_counts(G, j); *out = j; return 1; }
        for (int k = 0; k < NQ; k++) {
            int q = (w + k) % NQ;
            Deque *D = &G->dq[q];
            for (;;) {
                j = (q == w) ? deque_pop(D) : deque_steal(D);
                if (!j) break;
                if (job_ready(j) && (j->affinity & wbit)) {
                    if (q != w) st->stolen++;
                    take_counts(G, j);
                    *out = j;
                    return 1;
                }
                if (job_ready(j)) { /* ready but another worker's: external array, keep popping */
                    if (insert_slot(G->B, G->Blk, j) < 0) break;
                    continue;
                }
                j->flags |= 0x10; /* not ready: park it in A and count it for every eligible worker */
                counts_inc(G, j->affinity);
                insert_slot(G->A, G->Alk, j);
                break;
            }
        }
    }
    return 0;
}

/* ---- worker ------------------------------------------------------------------------------------------- */
typedef struct { int w; } WArg;
static WArg g_wargs[WORKERS];

static void run_job(Sched *S, int w, Job *j)
{
    Wstat *st = &g_ws[w];
    u64 t0 = rdtsc();
    u64 lat = tsc_to_ns(t0 - j->enq_tsc); /* a negative cross-thread delta clamps to 0 */
    if (j->parked) hist_add(&st->plat, lat); else hist_add(&st->lat, lat);
    __atomic_exchange_n(&S->lastJobTsc[w], t0, __ATOMIC_SEQ_CST); /* xchg at 0x86f5f3 */
    for (int i = 0; i < FORK && j->done_ctr; i++) { /* worker-side submit: ready child on my own deque (parked jobs have no frame counter) */
        Job *c = alloc_job();
        c->affinity = j->affinity; c->grp = j->grp; c->work_ns = j->work_ns / 4; c->done_ctr = j->done_ctr;
        c->parked = 0; c->enq_tsc = rdtsc();
        __atomic_fetch_add(j->done_ctr, 1, __ATOMIC_SEQ_CST);
        Group *G = &S->groups[j->grp];
        counts_inc(G, c->affinity);
        deque_push(&G->dq[w], c);
        st->forks++;
    }
    st->work_ticks += spin_cpu(ns_to_tsc(j->work_ns));
    st->jobs++;
    if (j->done_ctr) __atomic_fetch_add(j->done_ctr, -1, __ATOMIC_SEQ_CST); /* lock xadd -1 at job end */
    __atomic_store_n(&j->state, 0, __ATOMIC_RELEASE);
}

static void *worker_main(void *arg)
{
    int w = ((WArg *)arg)->w;
    thread_identify("BPE JobWorkerThread CPU", w);
    Sched *S = &g_sched;
    Wstat *st = &g_ws[w];
    while (!g_go) sceKernelUsleep(20000); /* >= 20 ms: a shorter usleep spins on Linux and would burn ~0.5 core per idle worker */
    S->lastJobTsc[w] = rdtsc();
    for (;;) {
        wait_for_jobs(S, w, &st->pauses);
        if (S->exit || g_stop) return NULL;
        Job *j = NULL;
        st->polls++;
        if (!choose(S, w, &j)) { /* <- the loop that eats the CPU while a parked job keeps counts > 0 */
#if NAP
            t3_nap(S->lastJobTsc[w], &st->pauses); /* T1: no job: nap if this worker has been idle >= the threshold */
#endif
            continue;
        }
        run_job(S, w, j);
#if NAP
        S->lastJobTsc[w] = rdtsc(); /* T2: stamp the slot with the TSC at job end */
#endif
    }
}

/* ---- producer (the "game thread") ---------------------------------------------------------------------- */
static void wait_for_counter_le(volatile i64 *c, i64 target) /* non-worker WaitForCounter, sleep=1 (0x8673a0) */
{
    for (u32 n = 0;;) {
        cpu_pause();
        if (*c <= target) return;
        if (n > 1000) n = 1; else n++;
        if (n == 1001) sceKernelUsleep(1);
    }
}

static volatile i64 g_frame_pending ALIGNED(64);

static void submit_ready(Sched *S, u64 work_ns)
{
    Group *G = &S->groups[GROUP];
    Job *j = alloc_job();
    j->affinity = AFFINITY; j->grp = GROUP; j->work_ns = work_ns; j->done_ctr = &g_frame_pending; j->parked = 0;
    __atomic_fetch_add(&g_frame_pending, 1, __ATOMIC_SEQ_CST);
    j->enq_tsc = rdtsc();
    counts_inc(G, j->affinity);                      /* up to 13 locked increments: broadcast through counters */
    while (insert_slot(G->B, G->Blk, j) < 0) cpu_pause(); /* non-worker submit -> array B */
}

static Job *submit_parked(Sched *S, volatile i64 *ctr, i64 gen, u64 work_ns)
{
    Group *G = &S->groups[GROUP];
    Job *j = alloc_job();
    j->affinity = AFFINITY; j->grp = GROUP; j->work_ns = work_ns; j->done_ctr = NULL; j->parked = 1;
    j->c0_ptr = ctr; j->c0_target = gen; j->c0_op = 3; /* waits for *ctr >= gen */
    j->flags |= 0x10;
    j->enq_tsc = ~0ull / 2;
    counts_inc(G, j->affinity);
    while (insert_slot(G->A, G->Alk, j) < 0) cpu_pause();
    return j;
}

static void emit_results(const Wstat *ws, u64 measured_us, u64 frames, const Hist *frame_hist, u64 main_pace_ticks)
{
    char buf[512];
    static Hist lat, plat; /* static: the entry thread's stack is only a few tens of KB */
    u64 polls = 0, jobs = 0, pred = 0, pauses = 0, dl = 0, busy = 0, forks = 0, stolen = 0, naps = 0, nap_ticks = 0;
    for (int w = 0; w < WORKERS; w++) {
        const Wstat *s = &ws[w];
        hist_emit_fields(buf, sizeof buf, "lat", &s->lat);
        EMIT("jobpool.worker w=%d polls=%lu jobs=%lu pred_calls=%lu pauses=%lu dl_checks=%lu work_ns=%lu forks=%lu stolen=%lu naps=%lu nap_ns=%lu nap_max_ns=%lu %s",
             w, s->polls, s->jobs, s->pred_calls, s->pauses, s->dl_checks, tsc_to_ns(s->work_ticks), s->forks, s->stolen, s->naps, tsc_to_ns(s->nap_ticks), tsc_to_ns(s->nap_max_ticks), buf);
        hist_merge(&lat, &s->lat); hist_merge(&plat, &s->plat);
        polls += s->polls; jobs += s->jobs; pred += s->pred_calls; pauses += s->pauses; dl += s->dl_checks;
        busy += s->work_ticks; forks += s->forks; stolen += s->stolen; naps += s->naps; nap_ticks += s->nap_ticks;
    }
    char b1[256], b2[256], b3[256];
    hist_emit_fields(b1, sizeof b1, "lat", &lat);
    hist_emit_fields(b2, sizeof b2, "parked_lat", &plat);
    hist_emit_fields(b3, sizeof b3, "frame", frame_hist);
    EMIT("jobpool.total measured_us=%lu polls=%lu jobs=%lu pred_calls=%lu pauses=%lu dl_checks=%lu work_ns=%lu forks=%lu stolen=%lu frames=%lu naps=%lu nap_ns=%lu",
         measured_us, polls, jobs, pred, pauses, dl, tsc_to_ns(busy), forks, stolen, frames, naps, tsc_to_ns(nap_ticks));
    for (int w = 0; w < WORKERS; w++) emit_gbusy("BPE JobWorkerThread CPU", w, ws[w].work_ticks, 0);
    emit_gbusy(NULL, 0, 0, main_pace_ticks);
    EMIT("jobpool.lat %s", b1);
    EMIT("jobpool.parked_lat %s", b2);
    EMIT("jobpool.frame %s", b3);
}

int _start(void *args, void *exit_fn)
{
    (void)args; (void)exit_fn;
    synth_init("jobpool");
    EMIT("jobpool.cfg workers=%d parked=%d producer_hz=%d job_us=%d frame_hz=%d affinity=0x%x pin=%d fork=%d dl_check=%d nap=%d nap_idle_us=%d nap_us=%d group=%d duration_s=%d warmup_s=%d",
         WORKERS, PARKED, PRODUCER_HZ, JOB_US, FRAME_HZ, (u32)AFFINITY, PIN_WORKERS, FORK, DL_CHECK, NAP, NAP_IDLE_US, NAP_US, GROUP, DURATION_S, WARMUP_S);

    Sched *S = &g_sched;
    S->ngroups = NG;
    S->groups = g_groups;
    for (int g = 0; g < NG; g++)
        for (int q = 0; q < NQ; q++) g_groups[g].dq[q].mask = SLOTS - 1;
    S->exit = 0;

    ScePthread th[WORKERS];
    static char names[WORKERS][32];
    for (int w = 0; w < WORKERS; w++) {
        g_wargs[w].w = w;
        ksnprintf(names[w], sizeof names[w], "BPE JobWorkerThread CPU%d", w);
        thread_spawn(&th[w], worker_main, &g_wargs[w], names[w], PIN_WORKERS ? (1ull << w) : 0);
    }

    wait_identified(WORKERS);
    warmup_and_start();
    g_two_tscf = DL_CHECK ? 2 * g_tsc_hz : ~0ull; /* after the TSC calibration is final */
    {   /* the patch's threshold: idle_us * the emulator's TSC frequency, clamped to an int32 immediate */
        u64 t = muldiv_u64(NAP_IDLE_US, sceKernelGetTscFrequency(), 1000000ull);
        g_nap_thr = t > 0x7fffffffull ? 0x7fffffff : (i64)t;
    }
    g_go = 1;

    const u64 frame_tsc = ns_to_tsc(1000000000ull / FRAME_HZ);
    const int per_frame = PRODUCER_HZ / FRAME_HZ;
    volatile i64 *parked[PARKED > 0 ? PARKED : 1];
    Job *parked_job[PARKED > 0 ? PARKED : 1];
    i64 parked_gen[PARKED > 0 ? PARKED : 1];
    int nparked = 0;
    u64 frames = 0;
    static Hist frame_hist;
    u64 next = rdtsc();
    u64 main_pace_ticks = 0;
    while (!run_expired()) {
        u64 f0 = rdtsc();
        /* release last frame's parked jobs */
        for (int i = 0; i < nparked; i++) {
            parked_job[i]->enq_tsc = rdtsc();
            __atomic_store_n(parked[i], parked_gen[i], __ATOMIC_SEQ_CST);
        }
        nparked = 0;
        for (int i = 0; i < per_frame; i++) submit_ready(S, (u64)JOB_US * 1000);
        for (int i = 0; i < PARKED; i++) {
            volatile i64 *ctr = &g_park_ctr[i];
            i64 gen = ++g_park_gen[i];
            parked_job[nparked] = submit_parked(S, ctr, gen, (u64)JOB_US * 1000);
            parked[nparked] = ctr;
            parked_gen[nparked++] = gen;
        }
        wait_for_counter_le(&g_frame_pending, 0);
        hist_add(&frame_hist, tsc_to_ns(rdtsc() - f0));
        frames++;
        next += frame_tsc;
        for (;;) { /* sleep out the frame: usleep while the slack is large, then a pacing spin (counted as guest CPU) */
            u64 t = rdtsc();
            if ((i64)(t - next) >= 0) { if (t - next > 4 * frame_tsc) next = t; break; }
            u64 left_us = tsc_to_ns(next - t) / 1000;
            if (left_us > 1500) sceKernelUsleep((u32)(left_us - 1000));
            else main_pace_ticks += spin_until(next);
        }
    }

    for (int w = 0; w < WORKERS; w++) g_snap[w] = g_ws[w]; /* counters of the measured phase only */
    u64 measured_us = tsc_to_ns(rdtsc() - measure_start_tsc()) / 1000;
    measure_end();
    S->exit = 1;
    for (int w = 0; w < WORKERS; w++) thread_join(th[w]);
    emit_results(g_snap, measured_us, frames, &frame_hist, main_pace_ticks);
    EMIT("DONE guest=jobpool");
    return 0;
}
