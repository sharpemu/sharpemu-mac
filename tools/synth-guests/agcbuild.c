/* Copyright (C) 2026 SharpEmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Synthetic (e'): AGC command-buffer building, the render-job workers' hot path in DS (lever L1: cheaper `*RegIndirectPatch*`
 * builders). THREADS worker threads each own a draw command buffer (DCB) in guest memory and build "draws": for each of the Cx, Sh
 * and Uc register spaces a sceAgcDcb Set*RegistersIndirect packet followed by the patch calls DS makes on it (SetAddress,
 * AddRegisters, SetNumRegisters), then a draw packet. After FRAME_DRAWS draws the cursor is reset, like a fresh DCB per render job.
 * Signatures and NIDs are those of src/SharpEmu.Libs/Agc (AgcExports.RegisterPackets.cs, .PacketPatching.cs, .DrawPackets.cs,
 * .PacketStubs.cs) and all of these are in DS's import list (ds-cpu/re-ds/imports.tsv). The loader may replace the Dcb* builders
 * with native stubs (SHARPEMU_AGC_PACKET_STUBS=0 keeps them on the managed exports); the patch calls always run in the managed exports.
 *
 * DCB layout the exports use (AgcExports.cs): +0x10 cursor (u64), +0x18 end of the buffer (u64), +0x20 full callback, +0x28 user data,
 * +0x30 reserved dwords (u32). The guest verifies what the exports wrote: cursor advance, returned packet address, PM4 header, address,
 * mask and register count dwords, before and after patching (counted in "agcbuild.verify"; a bad count means an early-out error path).
 *
 * Per-draw call mix, as -D counts (defaults: one of each): N_CX_IND N_SH_IND N_UC_IND (Dcb Set*RegistersIndirect), P_CX_ADDR P_SH_ADDR
 * P_UC_ADDR (Set*RegIndirectPatchSetAddress), P_CX_ADD P_SH_ADD P_UC_ADD (Set*RegIndirectPatchAddRegisters), P_CX_NUM
 * (SetCxRegIndirectPatchSetNumRegisters), N_DRAW_AUTO (sceAgcDcbDrawIndexAuto), N_DRAW_INDEX (sceAgcDcbDrawIndex, default 0). Patch
 * calls act on the last packet of their register space, so they need N_*_IND >= 1. (The Sh/Uc SetNumRegisters imports DS also has are
 * not implemented in this tree and are not called.) Set the DS mix from a profile with e.g. -DN_CX_IND=3 -DP_CX_ADD=6.
 *
 * Knobs: THREADS=4  DRAWS_HZ=50000 (total draws/s, 0 = flat out)  FRAME_DRAWS=512  PACE=1 (busy-work to the deadline; 0 = usleep)
 *        PERIOD_US=2000 (pacing batch)  REGS=4 (registers per indirect packet at build time)
 */
#include "common.h"

#ifndef THREADS
#define THREADS 4
#endif
#ifndef DRAWS_HZ
#define DRAWS_HZ 50000
#endif
#ifndef FRAME_DRAWS
#define FRAME_DRAWS 512
#endif
#ifndef PACE
#define PACE 1
#endif
#ifndef PERIOD_US
#define PERIOD_US 2000
#endif
#ifndef REGS
#define REGS 4
#endif
#ifndef N_CX_IND
#define N_CX_IND 1
#endif
#ifndef N_SH_IND
#define N_SH_IND 1
#endif
#ifndef N_UC_IND
#define N_UC_IND 1
#endif
#ifndef P_CX_ADDR
#define P_CX_ADDR 1
#endif
#ifndef P_SH_ADDR
#define P_SH_ADDR 1
#endif
#ifndef P_UC_ADDR
#define P_UC_ADDR 1
#endif
#ifndef P_CX_ADD
#define P_CX_ADD 1
#endif
#ifndef P_SH_ADD
#define P_SH_ADD 1
#endif
#ifndef P_UC_ADD
#define P_UC_ADD 1
#endif
#ifndef P_CX_NUM
#define P_CX_NUM 1
#endif
#ifndef N_DRAW_AUTO
#define N_DRAW_AUTO 1
#endif
#ifndef N_DRAW_INDEX
#define N_DRAW_INDEX 0
#endif

/* Dwords one draw may append to the DCB: 3 indirect packets of 5, draw packets of 7 / 11, with slack. */
#define DRAW_DWORDS_MAX ((N_CX_IND + N_SH_IND + N_UC_IND) * 5 + N_DRAW_AUTO * 7 + N_DRAW_INDEX * 11 + 8)
#define DCB_DWORDS (FRAME_DRAWS * DRAW_DWORDS_MAX + 1024)

/* ---- AGC imports (NID = symbol name) -------------------------------------------------------------------- */
extern u32 *sceAgcDcbSetCxRegistersIndirect(void *dcb, const void *regs, u32 n) __asm__("ZvwO9euwYzc");
extern u32 *sceAgcDcbSetShRegistersIndirect(void *dcb, const void *regs, u32 n) __asm__("-HOOCn0JY48");
extern u32 *sceAgcDcbSetUcRegistersIndirect(void *dcb, const void *regs, u32 n) __asm__("hvUfkUIQcOE");
extern int sceAgcSetCxRegIndirectPatchSetAddress(u32 *pkt, const void *regs) __asm__("vcmNN+AAXnY");
extern int sceAgcSetShRegIndirectPatchSetAddress(u32 *pkt, const void *regs) __asm__("Qrj4c+61z4A");
extern int sceAgcSetUcRegIndirectPatchSetAddress(u32 *pkt, const void *regs) __asm__("6lNcCp+fxi4");
extern int sceAgcSetCxRegIndirectPatchAddRegisters(u32 *pkt, u32 n) __asm__("d-6uF9sZDIU");
extern int sceAgcSetShRegIndirectPatchAddRegisters(u32 *pkt, u32 n) __asm__("z2duB-hHQSM");
extern int sceAgcSetUcRegIndirectPatchAddRegisters(u32 *pkt, u32 n) __asm__("vRoArM9zaIk");
extern int sceAgcSetCxRegIndirectPatchSetNumRegisters(u32 *pkt, u32 n) __asm__("whb1RL7K4Ss");
extern u32 *sceAgcDcbDrawIndexAuto(void *dcb, u32 index_count, u64 modifier) __asm__("Yw0jKSqop+E");
extern u32 *sceAgcDcbDrawIndex(void *dcb, u32 index_count, const void *index_addr, u64 modifier) __asm__("q88lQ+GP5Yk");

enum { I_CX_IND, I_SH_IND, I_UC_IND, I_CX_ADDR, I_SH_ADDR, I_UC_ADDR, I_CX_ADD, I_SH_ADD, I_UC_ADD, I_CX_NUM, I_DRAW_AUTO, I_DRAW_INDEX, NI };
static const char *const k_names[NI] = {
    "DcbSetCxRegistersIndirect", "DcbSetShRegistersIndirect", "DcbSetUcRegistersIndirect",
    "SetCxRegIndirectPatchSetAddress", "SetShRegIndirectPatchSetAddress", "SetUcRegIndirectPatchSetAddress",
    "SetCxRegIndirectPatchAddRegisters", "SetShRegIndirectPatchAddRegisters", "SetUcRegIndirectPatchAddRegisters",
    "SetCxRegIndirectPatchSetNumRegisters", "DcbDrawIndexAuto", "DcbDrawIndex",
};

/* PM4 (same encoding as AgcExports.Pm4): header = 0xC0000000 | (len-2)<<16 | op<<8 | reg<<2 */
#define PM4(len, op, reg) (0xC0000000u | ((((len) - 2u) & 0x3FFFu) << 16) | (((op) & 0xFFu) << 8) | (((reg) & 0x3Fu) << 2))
#define IT_NOP 0x10u
#define IT_SH_REG_INDIRECT 0x63u
#define IT_UC_REG_INDIRECT 0x64u
#define IT_CX_REG_INDIRECT 0x9Fu
#define IT_INDEX_BASE 0x26u
#define R_DRAW_INDEX_AUTO 0x04u

typedef struct TStat {
    u64 draws, calls[NI];
    u64 checks, bad;
    char first_bad[160];
    u64 pace_ticks;
    Hist h[NI];
} TStat;

static u32 g_dcb_mem[THREADS][DCB_DWORDS] ALIGNED(4096);
static u64 g_dcb[THREADS][8] ALIGNED(64);
static u32 g_regs[2][128] ALIGNED(64); /* two register tables: (offset, value) pairs */
static u32 g_idx[16] ALIGNED(64);
static TStat g_ts[THREADS], g_snap[THREADS];
static volatile int g_go;
typedef struct { int idx; } TArg;
static TArg g_targ[THREADS];
static u64 g_ns_k; /* ns per tick as 32.32 fixed point, for the per-call timing */

static inline u64 ticks_to_ns(u64 t) { return t < (1ull << 31) ? (t * g_ns_k) >> 32 : tsc_to_ns(t); }

#define CALL(idx, expr)                                              \
    do {                                                             \
        u64 t0_ = rdtsc();                                           \
        expr;                                                        \
        u64 t1_ = rdtsc();                                           \
        hist_add(&st->h[idx], ticks_to_ns(t1_ - t0_));              \
        st->calls[idx]++;                                            \
    } while (0)

static void bad(TStat *st, const char *what, int a, u64 got, u64 want)
{
    st->bad++;
    if (!st->first_bad[0]) ksnprintf(st->first_bad, sizeof st->first_bad, "%s:%d:got=%lx:want=%lx", what, a, got, want);
}

/* Check one register-indirect packet returned by a Dcb builder (after any patching): header, address, mask, count. */
static void verify_indirect(TStat *st, const u32 *pkt, u32 op, u64 regs_addr, u32 count, const char *what)
{
    st->checks++;
    u32 want_hdr = PM4(5, op, 0);
    if (pkt[0] != want_hdr) bad(st, what, 0, pkt[0], want_hdr);
    else if (pkt[1] != (u32)(regs_addr & 0xFFFFFFFCu)) bad(st, what, 1, pkt[1], regs_addr & 0xFFFFFFFCu);
    else if (pkt[2] != (u32)(regs_addr >> 32)) bad(st, what, 2, pkt[2], regs_addr >> 32);
    else if (pkt[3] != 0x80000000u) bad(st, what, 3, pkt[3], 0x80000000u);
    else if ((pkt[4] & 0x3FFFu) != (count & 0x3FFFu)) bad(st, what, 4, pkt[4] & 0x3FFFu, count & 0x3FFFu);
}

static void *thread_main(void *arg)
{
    int idx = ((TArg *)arg)->idx;
    TStat *st = &g_ts[idx];
    u64 *dcb = g_dcb[idx];
    u32 *base = g_dcb_mem[idx];
    thread_identify("AgcBuild", idx);
    while (!g_go) sceKernelUsleep(20000);

    u64 draws_hz = DRAWS_HZ ? (u64)DRAWS_HZ / THREADS : 0, batch_ticks = 0;
    u32 batch = 16;
    if (draws_hz) {
        if (!draws_hz) draws_hz = 1;
        batch = (u32)(draws_hz * PERIOD_US / 1000000);
        if (batch < 1) batch = 1;
        batch_ticks = muldiv_u64(batch, g_tsc_hz, draws_hz);
    }
    u64 next = rdtsc() + batch_ticks * (u64)idx / THREADS;
    st->pace_ticks += spin_until(next);

    int in_frame = 0;
    u64 verify_every = 32;
    while (!g_stop) {
        for (u32 b = 0; b < batch && !g_stop; b++) {
            if (in_frame == 0) { dcb[2] = (u64)base; } /* cursor back to the start: a fresh DCB */
            if (++in_frame >= FRAME_DRAWS) in_frame = 0;
            int verify = st->draws < 8 || (st->draws % verify_every) == 0;
            u32 *p_cx = 0, *p_sh = 0, *p_uc = 0;
            u64 cur0 = dcb[2];
            u32 expect_adv = 0;

            for (int i = 0; i < N_CX_IND; i++) { CALL(I_CX_IND, p_cx = sceAgcDcbSetCxRegistersIndirect(dcb, g_regs[0], REGS)); expect_adv += 5; }
            for (int i = 0; i < P_CX_ADDR && p_cx; i++) CALL(I_CX_ADDR, sceAgcSetCxRegIndirectPatchSetAddress(p_cx, g_regs[1]));
            for (int i = 0; i < P_CX_NUM && p_cx; i++) CALL(I_CX_NUM, sceAgcSetCxRegIndirectPatchSetNumRegisters(p_cx, REGS * 2));
            for (int i = 0; i < P_CX_ADD && p_cx; i++) CALL(I_CX_ADD, sceAgcSetCxRegIndirectPatchAddRegisters(p_cx, 2));

            for (int i = 0; i < N_SH_IND; i++) { CALL(I_SH_IND, p_sh = sceAgcDcbSetShRegistersIndirect(dcb, g_regs[0], REGS)); expect_adv += 5; }
            for (int i = 0; i < P_SH_ADDR && p_sh; i++) CALL(I_SH_ADDR, sceAgcSetShRegIndirectPatchSetAddress(p_sh, g_regs[1]));
            for (int i = 0; i < P_SH_ADD && p_sh; i++) CALL(I_SH_ADD, sceAgcSetShRegIndirectPatchAddRegisters(p_sh, 2));

            for (int i = 0; i < N_UC_IND; i++) { CALL(I_UC_IND, p_uc = sceAgcDcbSetUcRegistersIndirect(dcb, g_regs[0], REGS)); expect_adv += 5; }
            for (int i = 0; i < P_UC_ADDR && p_uc; i++) CALL(I_UC_ADDR, sceAgcSetUcRegIndirectPatchSetAddress(p_uc, g_regs[1]));
            for (int i = 0; i < P_UC_ADD && p_uc; i++) CALL(I_UC_ADD, sceAgcSetUcRegIndirectPatchAddRegisters(p_uc, 2));

            u32 *p_da = 0, *p_di = 0;
            for (int i = 0; i < N_DRAW_AUTO; i++) { CALL(I_DRAW_AUTO, p_da = sceAgcDcbDrawIndexAuto(dcb, 3 + (u32)(st->draws & 7), 0)); expect_adv += 7; }
            for (int i = 0; i < N_DRAW_INDEX; i++) { CALL(I_DRAW_INDEX, p_di = sceAgcDcbDrawIndex(dcb, 6, g_idx, 0)); expect_adv += 11; }

            if (verify) {
                /* The cursor must have advanced by exactly the packets built, and the last packet of each space must carry what the
                 * builder wrote plus what the patches did. A builder that bailed out (null buffer, no room, bad packet) fails here. */
                st->checks++;
                u64 cur1 = dcb[2];
                if (cur1 - cur0 != (u64)expect_adv * 4) bad(st, "cursor_advance", 0, (cur1 - cur0) / 4, expect_adv);
                if (N_CX_IND && P_CX_ADDR + P_CX_NUM + P_CX_ADD >= 0)
                    verify_indirect(st, p_cx ? p_cx : g_dcb_mem[0], IT_CX_REG_INDIRECT,
                                    P_CX_ADDR ? (u64)g_regs[1] : (u64)g_regs[0],
                                    (P_CX_NUM ? REGS * 2 : REGS) + 2 * P_CX_ADD, "cx");
                if (N_SH_IND)
                    verify_indirect(st, p_sh ? p_sh : g_dcb_mem[0], IT_SH_REG_INDIRECT,
                                    P_SH_ADDR ? (u64)g_regs[1] : (u64)g_regs[0], REGS + 2 * P_SH_ADD, "sh");
                if (N_UC_IND)
                    verify_indirect(st, p_uc ? p_uc : g_dcb_mem[0], IT_UC_REG_INDIRECT,
                                    P_UC_ADDR ? (u64)g_regs[1] : (u64)g_regs[0], REGS + 2 * P_UC_ADD, "uc");
                if (N_DRAW_AUTO) {
                    st->checks++;
                    u32 want = PM4(7, IT_NOP, R_DRAW_INDEX_AUTO), cnt = 3 + (u32)((st->draws) & 7);
                    if (!p_da || p_da[0] != want) bad(st, "draw_auto_hdr", 0, p_da ? p_da[0] : 0, want);
                    else if (p_da[1] != cnt) bad(st, "draw_auto_count", 1, p_da[1], cnt);
                }
                if (N_DRAW_INDEX) {
                    st->checks++;
                    if (!p_di || p_di[0] != PM4(6, 0x27u, 0)) bad(st, "draw_index_hdr", 0, p_di ? p_di[0] : 0, PM4(6, 0x27u, 0));
                }
            }
            st->draws++;
        }
        if (DRAWS_HZ) {
            next += batch_ticks;
            for (;;) {
                u64 t = rdtsc();
                if ((i64)(t - next) >= 0) { if (t - next > 8 * batch_ticks) next = t; break; }
                if (g_stop) break;
                if (PACE) { st->pace_ticks += spin_until(next); continue; }
                u64 left_us = tsc_to_ns(next - t) / 1000;
                if (left_us > 1500) sceKernelUsleep((u32)(left_us - 700));
            }
        }
    }
    return NULL;
}

int _start(void *args, void *exit_fn)
{
    (void)args; (void)exit_fn;
    synth_init("agcbuild");
    EMIT("agcbuild.cfg threads=%d draws_hz=%d frame_draws=%d pace=%d period_us=%d regs=%d n_cx_ind=%d n_sh_ind=%d n_uc_ind=%d p_cx_addr=%d p_sh_addr=%d p_uc_addr=%d "
         "p_cx_add=%d p_sh_add=%d p_uc_add=%d p_cx_num=%d n_draw_auto=%d n_draw_index=%d duration_s=%d warmup_s=%d",
         THREADS, DRAWS_HZ, FRAME_DRAWS, PACE, PERIOD_US, REGS, N_CX_IND, N_SH_IND, N_UC_IND, P_CX_ADDR, P_SH_ADDR, P_UC_ADDR,
         P_CX_ADD, P_SH_ADD, P_UC_ADD, P_CX_NUM, N_DRAW_AUTO, N_DRAW_INDEX, DURATION_S, WARMUP_S);
    for (int t = 0; t < 2; t++)
        for (int i = 0; i < 64; i++) { g_regs[t][2 * i] = 0x2000u + (u32)i; g_regs[t][2 * i + 1] = 0x11110000u * (u32)(t + 1) + (u32)i; }
    for (int t = 0; t < THREADS; t++) {
        g_dcb[t][2] = (u64)g_dcb_mem[t];                                        /* +0x10 cursor */
        g_dcb[t][3] = (u64)(g_dcb_mem[t] + DCB_DWORDS);                         /* +0x18 end */
        g_dcb[t][4] = 0; g_dcb[t][5] = 0; g_dcb[t][6] = 0;                      /* callback, user data, reserved dwords */
    }
    ScePthread th[THREADS];
    static char names[THREADS][32];
    for (int i = 0; i < THREADS; i++) {
        g_targ[i].idx = i;
        ksnprintf(names[i], sizeof names[i], "AgcBuild%d", i);
        thread_spawn(&th[i], thread_main, &g_targ[i], names[i], 0);
    }
    wait_identified(THREADS);
    warmup_and_start();
    g_ns_k = muldiv_u64(1000000000ull, 1ull << 32, g_tsc_hz);
    g_go = 1;
    run_for_sleep();
    for (int i = 0; i < THREADS; i++) g_snap[i] = g_ts[i];
    u64 measured_us = tsc_to_ns(rdtsc() - measure_start_tsc()) / 1000;
    measure_end();
    for (int i = 0; i < THREADS; i++) thread_join(th[i]);

    static Hist all[NI];
    u64 calls[NI] = {0}, draws = 0, checks = 0, badn = 0, total_calls = 0;
    const char *first_bad = "";
    for (int i = 0; i < THREADS; i++) {
        const TStat *s = &g_snap[i];
        EMIT("agcbuild.thread idx=%d draws=%lu checks=%lu bad=%lu pace_ns=%lu", i, s->draws, s->checks, s->bad, tsc_to_ns(s->pace_ticks));
        emit_gbusy(names[i], i, 0, s->pace_ticks);
        draws += s->draws; checks += s->checks; badn += s->bad;
        if (s->bad && !*first_bad) first_bad = s->first_bad;
        for (int k = 0; k < NI; k++) { calls[k] += s->calls[k]; total_calls += s->calls[k]; hist_merge(&all[k], &s->h[k]); }
    }
    EMIT("agcbuild.total measured_us=%lu draws=%lu calls=%lu draws_per_s=%lu calls_per_s=%lu", measured_us, draws, total_calls,
         draws * 1000000 / measured_us, total_calls * 1000000 / measured_us);
    EMIT("agcbuild.verify checks=%lu bad=%lu first_bad=%s", checks, badn, *first_bad ? first_bad : "none");
    for (int k = 0; k < NI; k++) {
        if (!calls[k]) continue;
        char b[256];
        hist_emit_fields(b, sizeof b, "ns", &all[k]);
        EMIT("agcbuild.import name=%s calls=%lu calls_per_s=%lu %s", k_names[k], calls[k], calls[k] * 1000000 / measured_us, b);
    }
    EMIT("DONE guest=agcbuild");
    return 0;
}
