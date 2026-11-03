// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>
/// A seeded simulator of a running game (a port of the design prototype's simulator) that feeds the
/// overlay without an emulator behind it: scenarios <c>ds</c>, <c>sh</c>, <c>gta</c>, <c>menu</c> and
/// <c>load</c>. Select it at runtime with <c>SHARPEMU_OVERLAY_SIMULATE=&lt;scenario&gt;</c>.
/// </summary>
public sealed class SimulatedOverlayDataSource : IOverlayDataSource
{
    public const string EnvironmentVariable = "SHARPEMU_OVERLAY_SIMULATE";
    public const int HistoryCapacity = 240;

    public static readonly string[] ScenarioKeys = ["ds", "sh", "gta", "menu", "load"];

    // ------------------------------------------------------------------ names

    private sealed record NameDef(string Name, CostKind Kind, OverlayLane Lane, bool NoCost)
    {
        public string Display { get; } = Name.StartsWith("thread:", StringComparison.Ordinal) ? Name[7..] : Name;
    }

    private static readonly List<NameDef> Names = [];
    private static readonly Dictionary<string, int> NameIndex = [];
    private static readonly string[] ThreadNames;

    private static int Def(string name, CostKind kind, OverlayLane lane, bool noCost = false)
    {
        NameIndex[name] = Names.Count;
        Names.Add(new NameDef(name, kind, lane, noCost));
        return Names.Count - 1;
    }

    static SimulatedOverlayDataSource()
    {
        const OverlayLane G = OverlayLane.GuestMain, W = OverlayLane.GuestWorkers, R = OverlayLane.Render, P = OverlayLane.Gpu;
        Def("GuestCode", CostKind.Guest, G);
        Def("sceGnmSubmitCommandBuffers", CostKind.Hle, G);
        Def("sceKernelReadTsc", CostKind.Hle, G);
        Def("sceKernelWaitEqueue", CostKind.Wait, G);
        Def("sceKernelUsleep", CostKind.Wait, G);
        Def("scePthreadMutexLock", CostKind.Wait, G);
        Def("sceVideoOutSubmitFlip", CostKind.Wait, G);
        Def("GcPause", CostKind.Other, G);
        Def("AssetStreaming", CostKind.Other, G);
        Def("WorkerJobs", CostKind.Guest, W);
        Def("WorkerIdle", CostKind.Wait, W, true);
        Def("WindowEventPolling", CostKind.Render, R);
        Def("FrameSlotWait", CostKind.Wait, R);
        Def("CommandStream", CostKind.Render, R);
        Def("CommandDrawTranslation", CostKind.Render, R);
        Def("VertexProgramEvaluation", CostKind.Shader, R);
        Def("PixelProgramEvaluation", CostKind.Shader, R);
        Def("GraphicsProgramCache", CostKind.Shader, R);
        Def("DrawAttachmentPreparation", CostKind.Render, R);
        Def("ResourceSnapshotCreation", CostKind.Render, R);
        Def("CommandDispatchTranslation", CostKind.Render, R);
        Def("CommandMemorySync", CostKind.Render, R);
        Def("CommandMemoryRead", CostKind.Render, R);
        Def("CommandGpuWait", CostKind.Wait, R);
        Def("CommandMemoryTransfer", CostKind.Render, R);
        Def("Collect", CostKind.Render, R);
        Def("QueueSubmit", CostKind.Render, R);
        Def("GpuGraphics", CostKind.Gpu, P);
        Def("GpuCompute", CostKind.Gpu, P);
        var threadNames = new List<string> { "Guest main", "Render", "Shader JIT", "GPU host" };
        for (var i = 1; i <= 13; i++)
        {
            threadNames.Add("Worker " + i);
        }

        ThreadNames = [.. threadNames];
        for (var i = 0; i < ThreadNames.Length; i++)
        {
            Def("thread:" + ThreadNames[i], CostKind.Other, G, true);
        }
    }

    private static int N(string name) => NameIndex[name];
    private static string NameOfId(int id) => id >= 0 && id < Names.Count ? Names[id].Display : string.Empty;

    // ------------------------------------------------------------------ scenarios

    private sealed record Profile(string Label, string Sub, int Target, uint ResW, uint ResH, float GMain, float WSpan, float WUtil,
        bool Serial, int Ahead, int RSlots, int Passes, float Sync, float[][] Hle, float[] R, float[] Gpu, int Draws, int Disp,
        float Alloc, float[] Mem);

    private static readonly string[] RenderNames =
    [
        "WindowEventPolling", "CommandStream", "CommandDrawTranslation", "VertexProgramEvaluation", "PixelProgramEvaluation",
        "GraphicsProgramCache", "DrawAttachmentPreparation", "ResourceSnapshotCreation", "CommandDispatchTranslation",
        "CommandMemorySync", "CommandMemoryRead", "CommandGpuWait", "CommandMemoryTransfer", "Collect", "QueueSubmit",
    ];

    private static readonly string[] PassNames = RenderNames[2..13];
    private static readonly string[] HleNames = ["sceKernelReadTsc", "scePthreadMutexLock", "sceKernelUsleep", "sceKernelWaitEqueue", "sceGnmSubmitCommandBuffers"];

    private static readonly Dictionary<string, Profile> Scenarios = new()
    {
        ["ds"] = new("Demon's Souls in-world", "GPU-bound ~24 fps", 30, 2560, 1440, 12.5f, 7.5f, .72f, false, 3, 2, 3, 0,
            [[.4f, 2900], [.7f, 230], [.5f, 3], [.3f, 6], [1.3f, 14]],
            [.25f, .8f, 2.0f, .9f, 2.4f, .6f, .9f, 1.3f, .8f, 1.1f, .7f, .3f, .6f, .5f, .6f], [36, 5.5f], 2380, 190, 38, [9.8f, 4.1f, 2.6f, .9f]),
        ["sh"] = new("Silent Hill", "worker / CPU-bound ~26 fps", 30, 1920, 1080, 15f, 33f, .8f, false, 3, 2, 3, 0,
            [[.5f, 3400], [1.2f, 900], [.8f, 6], [.6f, 40], [.9f, 10]],
            [.2f, .5f, 1.4f, .6f, 1.5f, .35f, .6f, .8f, .5f, .7f, .4f, .2f, .3f, .4f, .4f], [13, 3], 1310, 92, 52, [8.6f, 3.2f, 2.1f, .8f]),
        ["gta"] = new("GTA V Fidelity", "sync-bound ~7 fps", 30, 1920, 1080, 22f, 9f, .5f, true, 1, 2, 3, 20,
            [[.6f, 5200], [1.5f, 1200], [1f, 10], [.8f, 60], [1.6f, 30]],
            [.3f, 1.2f, 3.5f, 1.5f, 3.2f, 1.2f, 1.4f, 2.0f, 2.0f, 6f, 9f, 9f, 3.2f, .9f, .8f], [24, 6], 5210, 640, 96, [11.2f, 5.4f, 3.3f, 1.2f]),
        ["menu"] = new("Menu", "60 fps", 60, 3840, 2160, 3.2f, 1.5f, .4f, false, 2, 2, 1, 0,
            [[.15f, 800], [.2f, 40], [1.4f, 30], [.1f, 3], [.2f, 3]],
            [.2f, .3f, .5f, .2f, .3f, .1f, .2f, .2f, .1f, .1f, .05f, .05f, .1f, .2f, .2f], [4.8f, .4f], 180, 8, 6, [5.2f, 2.1f, 1.1f, .4f]),
        ["load"] = new("Loading stall", "no flips for seconds, alloc high", 30, 1280, 720, 6f, 3f, .5f, false, 3, 2, 2, 0,
            [[.3f, 1500], [.4f, 100], [.8f, 12], [.5f, 10], [.5f, 5]],
            [.2f, .4f, .8f, .3f, .6f, .2f, .3f, .4f, .2f, .4f, .3f, .1f, .5f, .3f, .3f], [7, 1], 420, 30, 70, [7.5f, 2.8f, 1.6f, .7f]),
    };

    public static bool IsScenario(string? key) => key is not null && Scenarios.ContainsKey(key.Trim().ToLowerInvariant());

    /// <summary>Creates the simulator named by <c>SHARPEMU_OVERLAY_SIMULATE</c>, or null when it is unset or unknown.</summary>
    public static SimulatedOverlayDataSource? FromEnvironment()
    {
        var value = Environment.GetEnvironmentVariable(EnvironmentVariable)?.Trim().ToLowerInvariant();
        if (string.IsNullOrEmpty(value) || value is "0" or "off")
        {
            return null;
        }

        return new SimulatedOverlayDataSource(IsScenario(value) ? value : "ds");
    }

    // ------------------------------------------------------------------ rng

    private uint _rngState;

    private double Rand()
    {
        _rngState += 0x6D2B79F5;
        var t = _rngState;
        t = (t ^ (t >> 15)) * (1 | t);
        t = (t + (t ^ (t >> 7)) * (61 | t)) ^ t;
        return ((t ^ (t >> 14))) / 4294967296.0;
    }

    private double Gauss()
    {
        double u;
        do { u = Rand(); } while (u == 0);
        return Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(6.283185307 * Rand());
    }

    private double LogNormal(double s) => Math.Exp(Gauss() * s - s * s / 2);

    // ------------------------------------------------------------------ state

    private struct Seg
    {
        public int Lane;
        public int Name;
        public double S, E;
        public double Count;
        public double Util;
    }

    private sealed class Pf
    {
        public double Gs, GWorkEnd, GEnd, Rs, REnd, GpuS, GpuEnd, Present, T;
        public int Draws, Disp, Collections;
        public double Alloc, Compile, GcPause;
        public int[] Gc = new int[3];
        public double[] Mem = new double[4];
        public bool Stall, Stream;
    }

    private static readonly Pf Zero = new() { T = 33 };

    private readonly object _gate = new();
    private readonly Profile _sc;
    private readonly List<Pf> _pf = [];
    private readonly List<Seg> _segs = [];
    private readonly List<FrameRecord> _history = [];
    private readonly List<double> _recentFps = [];
    private int _n, _fi, _base;
    private double _t;
    private readonly int[] _gc = [12, 2, 0];
    private double _gen0Acc;
    private double _pendingCompile;
    private double _loadUntil = 2500;
    private readonly double[] _mem;
    private readonly bool _realTime;
    private long _lastClock;
    private long _elapsedStart;
    private double _hitchFrequency = 0.35;
    private SystemStats _system = SystemStats.Empty;
    private double _cpuPercent, _cpuCores, _gpuPercent;

    public SimulatedOverlayDataSource(string scenario = "ds", int seed = 1337, bool realTime = true)
    {
        Scenario = Scenarios.ContainsKey(scenario) ? scenario : "ds";
        _sc = Scenarios[Scenario];
        _rngState = (uint)seed;
        _mem = [_sc.Mem[0], _sc.Mem[1], _sc.Mem[2], _sc.Mem[3]];
        _realTime = realTime;
        Advance(9000);
        _lastClock = _elapsedStart = Stopwatch.GetTimestamp();
    }

    public string Scenario { get; }
    public OverlayCaptureDemand Demand { get; private set; } = OverlayCaptureDemand.None;
    public string Label => _sc.Label;

    public void SetCaptureDemand(OverlayCaptureDemand demand) => Demand = demand;

    /// <summary>Queues a long shader-compile hitch on the next frame (the prototype's hitch button).</summary>
    public void TriggerCompileHitch()
    {
        lock (_gate)
        {
            _pendingCompile = 60 + Rand() * 80;
        }
    }

    public OverlaySnapshot Capture()
    {
        lock (_gate)
        {
            if (_realTime)
            {
                var now = Stopwatch.GetTimestamp();
                var dt = (now - _lastClock) * 1000.0 / Stopwatch.Frequency;
                _lastClock = now;
                AdvanceLocked(Math.Min(dt, 500));
            }

            return new OverlaySnapshot([.. _history], _system, NameOfId);
        }
    }

    /// <summary>Advances simulated time by <paramref name="milliseconds"/> (tests and tools).</summary>
    public void Advance(double milliseconds)
    {
        lock (_gate)
        {
            AdvanceLocked(milliseconds);
        }
    }

    private Pf At(int k) => k - _base >= 0 && k - _base < _pf.Count ? _pf[k - _base] : Zero;

    private void AdvanceLocked(double dt)
    {
        _t += dt;
        for (var guard = 0; guard < 40; guard++)
        {
            while (_pf.Count + _base <= _fi)
            {
                Generate();
            }

            var f = At(_fi);
            if (f.Present > _t)
            {
                break;
            }

            var g2 = 0;
            while (At(_n - 1).Gs < f.Present && g2++ < 12)
            {
                Generate();
            }

            FinalizeFrame(_fi);
            _fi++;
        }
    }

    // ------------------------------------------------------------------ generation (port of the prototype)

    private void Push(int lane, string name, double s, double e, double count = 1, double util = 1)
    {
        if (e - s > 0.004)
        {
            _segs.Add(new Seg { Lane = lane, Name = N(name), S = s, E = e, Count = count, Util = util });
        }
    }

    private void Generate()
    {
        var n = _n++;
        var sc = _sc;
        var pv = At(n - 1);
        var hf = Math.Clamp(_hitchFrequency / .35, 0, 3);
        var stall = Scenario == "load" && pv.GEnd >= _loadUntil;
        var gs = Math.Max(pv.GEnd, sc.Serial ? pv.Present : 0);
        if (gs > pv.GEnd + .05)
        {
            Push(0, "sceVideoOutSubmitFlip", pv.GEnd, gs);
        }

        var cx = 1 + .09 * Math.Sin(gs / 1000 / 5.3) + .05 * Math.Sin(gs / 1000 / 1.9 + 2) + .03 * Math.Sin(gs / 1000 / .77);
        double t = gs, tDisp = gs, jEnd = gs, gWorkEnd, d = 0, streamMs = 0, gcPause = 0, compile = 0;
        var collections = 0;
        var allocRate = stall ? 800 + Rand() * 200 : sc.Alloc * LogNormal(.12);
        _gen0Acc += allocRate * (pv.T > 0 ? pv.T : 33) / 1000;
        var threshold = 24 / Math.Clamp(hf, .25, 3);
        while (_gen0Acc >= threshold)
        {
            _gen0Acc -= threshold;
            _gc[0]++;
            collections++;
            gcPause += .35;
            if (_gc[0] % 6 == 0)
            {
                _gc[1]++;
                gcPause += 3 + Rand() * 4;
                if (_gc[1] % 5 == 0)
                {
                    _gc[2]++;
                    gcPause += 22 + Rand() * 45;
                }
            }
        }

        if (!stall)
        {
            if (Rand() < .0025 * hf)
            {
                streamMs = 12 + Rand() * 60;
            }

            if (_pendingCompile > 0)
            {
                compile = _pendingCompile;
                _pendingCompile = 0;
            }
            else if (Rand() < .004 * hf)
            {
                compile = Math.Clamp(45 * LogNormal(.55), 18, 180);
            }
        }

        if (stall)
        {
            d = 3000 + Rand() * 4400;
            var end = gs + d;
            while (t < end - .5)
            {
                var r = Rand();
                string name;
                double ms, cnt = 1;
                if (r < .38) { name = "AssetStreaming"; ms = 60 + Rand() * 340; cnt = 1 + Math.Floor(Rand() * 6); }
                else if (r < .58) { name = "sceKernelUsleep"; ms = 15 + Rand() * 90; cnt = 4; }
                else if (r < .78) { name = "sceKernelWaitEqueue"; ms = 30 + Rand() * 220; cnt = 3; }
                else if (r < .88) { name = "GcPause"; ms = 30 + Rand() * 60; }
                else if (r < .95) { name = "scePthreadMutexLock"; ms = 10 + Rand() * 80; cnt = 400; }
                else { name = "GuestCode"; ms = 4 + Rand() * 20; }
                ms = Math.Min(ms, end - t);
                Push(0, name, t, t + ms, cnt);
                t += ms;
                var rt = Math.Min(1 + Rand() * 5, end - t);
                if (rt > 0 && Rand() < .5)
                {
                    Push(0, "sceKernelReadTsc", t, t + rt, 4000);
                    t += rt;
                }
            }

            t = end;
            gWorkEnd = t;
            var wt = gs;
            var wEnd = gs + d * .9;
            while (wt < wEnd)
            {
                var b = Math.Min(80 + Rand() * 500, wEnd - wt);
                var u = Math.Clamp(.8 + Rand() * .2, 0, 1);
                Push(1, "WorkerJobs", wt, wt + b, 1, u);
                wt += b;
                var gap = Math.Min(Rand() * 70, wEnd - wt);
                if (gap > 1)
                {
                    Push(1, "WorkerIdle", wt, wt + gap, 1, 0);
                    wt += gap;
                }
            }

            Push(1, "WorkerIdle", wt, gWorkEnd, 1, 0);
            jEnd = gWorkEnd;
        }
        else
        {
            var cl = 1.0;
            var hle = new (double Ms, double Count)[5];
            for (var k = 0; k < 5; k++)
            {
                var isWait = Names[N(HleNames[k])].Kind == CostKind.Wait;
                hle[k] = (sc.Hle[k][0] * (isWait ? cx : cl * cx) * LogNormal(.15), Math.Round(sc.Hle[k][1] * cx * LogNormal(.1)));
            }

            var codeMs = Math.Max(.8, sc.GMain * cl * cx * LogNormal(.06) - hle[0].Ms - hle[4].Ms);
            void G(string name, double ms, double cnt = 1)
            {
                if (ms > 0.004)
                {
                    Push(0, name, t, t + ms, cnt);
                    t += ms;
                }
            }

            double[] cs = [.2 * codeMs, .3 * codeMs, .28 * codeMs, .22 * codeMs];
            G("GuestCode", cs[0]);
            G("sceKernelReadTsc", hle[0].Ms / 2, hle[0].Count / 2);
            tDisp = t;
            G("GuestCode", cs[1]);
            G("scePthreadMutexLock", hle[1].Ms, hle[1].Count);
            G("GuestCode", cs[2]);
            G("sceKernelUsleep", hle[2].Ms, hle[2].Count);
            G("sceKernelReadTsc", hle[0].Ms / 2, hle[0].Count / 2);
            G("GuestCode", cs[3]);
            if (streamMs > 0) { G("AssetStreaming", streamMs, 1 + Math.Floor(Rand() * 3)); }
            if (gcPause > .7) { G("GcPause", gcPause); }
            G("sceKernelWaitEqueue", hle[3].Ms, hle[3].Count);
            if (sc.Sync > 0)
            {
                var sy = sc.Sync * cx * LogNormal(.1);
                G("sceKernelWaitEqueue", sy * .6, 3);
                G("scePthreadMutexLock", sy * .4, 160);
            }

            var wSpan = sc.WSpan * cl * cx * LogNormal(.08);
            var jobsEnd = tDisp + wSpan;
            jEnd = jobsEnd;
            if (jobsEnd > t)
            {
                var wait = jobsEnd - t;
                if (Rand() < .55) { G("scePthreadMutexLock", wait, Math.Round(wait * 14)); }
                else { G("sceKernelWaitEqueue", wait, Math.Max(1, Math.Round(wait / 6))); }
            }

            G("sceGnmSubmitCommandBuffers", hle[4].Ms, hle[4].Count);
            gWorkEnd = t;
            Push(1, "WorkerIdle", gs, tDisp, 1, 0);
            var wt = tDisp;
            var left = wSpan;
            for (var c = 0; c < 4; c++)
            {
                var b = c == 3 ? left : left * (.18 + Rand() * .22);
                left -= b;
                Push(1, "WorkerJobs", wt, wt + b, 1, Math.Clamp(sc.WUtil * LogNormal(.14), .2, 1));
                wt += b;
            }
        }

        var gate = At(n - sc.Ahead).Present;
        var gEnd = Math.Max(gWorkEnd + .12, gate);
        Push(0, "sceVideoOutSubmitFlip", gWorkEnd, gEnd);
        Push(1, "WorkerIdle", jEnd, gEnd, 1, 0);

        // Render thread.
        var rs = Math.Max(gWorkEnd, pv.REnd);
        t = rs;
        var rScale = stall ? .04 : 1;
        double Rr(string name) => sc.R[Array.IndexOf(RenderNames, name)] * cx * LogNormal(.07) * rScale;
        void Rn(string name, double ms, double cnt = 1)
        {
            if (ms > .004)
            {
                Push(2, name, t, t + ms, cnt);
                t += ms;
            }
        }

        Rn("WindowEventPolling", Rr("WindowEventPolling"));
        var slot = At(n - sc.RSlots).GpuEnd;
        if (slot > t)
        {
            Push(2, "FrameSlotWait", t, slot);
            t = slot;
        }

        Rn("CommandStream", Rr("CommandStream") * .5);
        var passes = stall ? 1 : sc.Passes;
        for (var p = 0; p < passes; p++)
        {
            foreach (var nm in PassNames)
            {
                var ms = Rr(nm) / passes;
                if (compile > 0 && p == 0)
                {
                    if (nm == "PixelProgramEvaluation") { ms += compile * .55; }
                    else if (nm == "VertexProgramEvaluation") { ms += compile * .3; }
                    else if (nm == "GraphicsProgramCache") { ms += compile * .15; }
                }

                Rn(nm, ms, nm == "CommandDrawTranslation" ? Math.Round(sc.Draws * cx / passes) : 1);
            }
        }

        Rn("CommandStream", Rr("CommandStream") * .5);
        Rn("Collect", Rr("Collect"));
        Rn("QueueSubmit", Rr("QueueSubmit"));
        var rEnd = t;

        // GPU.
        var gpuStart = Math.Max(rEnd, pv.GpuEnd);
        var gk = stall ? .05 : 1;
        var compute = sc.Gpu[1] * cx * LogNormal(.05) * gk;
        var graphics = sc.Gpu[0] * cx * LogNormal(.04) * gk;
        Push(3, "GpuCompute", gpuStart, gpuStart + compute, Math.Round(sc.Disp * cx));
        Push(3, "GpuGraphics", gpuStart + compute, gpuStart + compute + graphics, Math.Round(sc.Draws * cx));
        var gpuEnd = gpuStart + compute + graphics;
        var present = Math.Max(gpuEnd, pv.Present + 1000.0 / sc.Target);

        // Memory drift.
        _mem[0] += (sc.Mem[0] - _mem[0]) * .01 + (Rand() - .5) * .012 + (stall ? d / 1000 * .18 * (.8 + Rand() * .4) : 0);
        _mem[1] += (sc.Mem[1] - _mem[1]) * .01 + (Rand() - .5) * .006;
        _mem[2] += (sc.Mem[2] - _mem[2]) * .01 + (Rand() - .5) * .006 + (stall ? d / 1000 * .08 : 0);
        _mem[3] += (sc.Mem[3] - _mem[3]) * .01 + (Rand() - .5) * .003 + (stall ? d / 1000 * .03 : 0);
        var pf = new Pf
        {
            Gs = gs, GWorkEnd = gWorkEnd, GEnd = gEnd, Rs = rs, REnd = rEnd, GpuS = gpuStart, GpuEnd = gpuEnd, Present = present,
            T = present - pv.Present,
            Draws = (int)Math.Round(sc.Draws * cx * (stall ? .02 : 1)), Disp = (int)Math.Round(sc.Disp * cx * (stall ? .02 : 1)),
            Alloc = allocRate, Compile = compile, GcPause = gcPause, Collections = collections, Stall = stall, Stream = streamMs > 0,
        };
        pf.Gc[0] = _gc[0]; pf.Gc[1] = _gc[1]; pf.Gc[2] = _gc[2];
        Array.Copy(_mem, pf.Mem, 4);
        _pf.Add(pf);
        if (_pf.Count > 24)
        {
            _pf.RemoveAt(0);
            _base++;
        }

        if (stall)
        {
            _loadUntil = present + 1300 + Rand() * 1200;
        }
    }

    // ------------------------------------------------------------------ frame records

    private void FinalizeFrame(int fi)
    {
        var pf = At(fi);
        var a = fi > 0 ? At(fi - 1).Present : 0;
        var b = pf.Present;
        var frameMs = Math.Max(.001, b - a);
        var selfMs = new double[Names.Count];
        var counts = new double[Names.Count];
        var segs = new List<TimelineSegment>();
        double gBusy = 0, wBusy = 0, wWork = 0, rBusy = 0, rWait = 0;
        var gpuIntervals = new List<(double S, double E)>();
        foreach (var s in _segs)
        {
            if (s.E <= a || s.S >= b)
            {
                continue;
            }

            var cs = Math.Max(s.S, a);
            var ce = Math.Min(s.E, b);
            var dur = ce - cs;
            if (dur <= 0)
            {
                continue;
            }

            var def = Names[s.Name];
            selfMs[s.Name] += def.NoCost ? 0 : dur * (def.Name == "WorkerJobs" ? s.Util * 3.0 : 1);
            if (s.S >= a && s.S < b)
            {
                counts[s.Name] += s.Count;
            }

            segs.Add(new TimelineSegment((float)(cs - a), (float)(ce - a), def.Lane, def.Kind, s.Name));
            switch (s.Lane)
            {
                case 0 when def.Kind != CostKind.Wait: gBusy += dur; break;
                case 1 when def.Name == "WorkerJobs": wBusy += dur; wWork += dur * s.Util; break;
                case 2 when def.Kind == CostKind.Wait: rWait += dur; break;
                case 2: rBusy += dur; break;
                case 3: gpuIntervals.Add((cs - a, ce - a)); break;
            }
        }

        gpuIntervals.Sort();
        double gpuBusy = 0, cursor = double.NegativeInfinity;
        foreach (var (s, e) in gpuIntervals)
        {
            var from = Math.Max(s, cursor);
            if (e > from)
            {
                gpuBusy += e - from;
            }

            cursor = Math.Max(cursor, e);
        }

        var kinds = new float[(int)CostKind.Count];
        var costs = new List<CostSample>();
        for (var i = 0; i < Names.Count; i++)
        {
            var def = Names[i];
            if (def.NoCost || selfMs[i] <= 0)
            {
                continue;
            }

            costs.Add(new CostSample(i, def.Kind, def.Lane, (float)selfMs[i], (int)Math.Min(int.MaxValue, counts[i])));
            kinds[(int)def.Kind] += (float)selfMs[i];
        }

        // Threads: main, render, JIT, GPU host and 13 workers that share the busy time.
        var threads = new List<ThreadSample>
        {
            new(N("thread:Guest main"), OverlayLane.GuestMain, (float)gBusy),
            new(N("thread:Render"), OverlayLane.Render, (float)rBusy),
            new(N("thread:Shader JIT"), OverlayLane.Render, (float)Math.Min(pf.Compile * 2, frameMs) + (float)(.8 + Rand() * 2.2)),
            new(N("thread:GPU host"), OverlayLane.Gpu, (float)(gpuBusy * .34 + 3 + Rand() * 2)),
        };
        var activeWorkers = 0;
        var workerTotal = 0.0;
        for (var w = 0; w < 13; w++)
        {
            var share = Math.Clamp(wWork * (w == 0 ? 1.0 : .75 / (1 + w * .35)) * (.85 + Rand() * .3), 0, frameMs);
            if (share > 0.05)
            {
                activeWorkers++;
                workerTotal += share;
                threads.Add(new ThreadSample(N("thread:Worker " + (w + 1)), OverlayLane.GuestWorkers, (float)share));
            }
        }

        var hitch = pf.Compile > 0 || pf.Stall || pf.GcPause > 15 || frameMs > 1.6 * AverageRecent();
        string? reason = null;
        if (hitch)
        {
            reason = pf.Stall ? "loading" : pf.Compile > 0 ? "shader compile" : pf.GcPause > 15 ? "GC gen2" : gpuBusy > frameMs * .85 ? "GPU" : "CPU";
        }

        var record = new FrameRecord
        {
            Sequence = fi,
            EndTimestamp = (long)(b * Stopwatch.Frequency / 1000.0),
            FrameMs = (float)frameMs,
            GuestFrameMs = (float)frameMs,
            GuestMainMs = (float)gBusy,
            GuestWorkersMs = (float)workerTotal,
            ActiveWorkers = activeWorkers,
            RenderBusyMs = (float)rBusy,
            RenderWaitMs = (float)rWait,
            GpuMs = (float)gpuBusy,
            KindMs = kinds,
            Draws = pf.Draws,
            Dispatches = pf.Disp,
            Submits = 1,
            GcPauseMs = (float)pf.GcPause,
            GcCollections = pf.Collections,
            Hitch = hitch,
            HitchReason = reason,
            Segments = [.. segs],
            Costs = [.. costs],
            Threads = [.. threads],
        };
        _history.Add(record);
        if (_history.Count > HistoryCapacity)
        {
            _history.RemoveAt(0);
        }

        _segs.RemoveAll(s => s.E <= b);
        UpdateSystem(pf, frameMs, gBusy + rBusy + workerTotal, gpuBusy);
    }

    private double AverageRecent()
    {
        if (_history.Count == 0)
        {
            return 1000.0 / _sc.Target;
        }

        var n = Math.Min(10, _history.Count);
        double sum = 0;
        for (var i = 0; i < n; i++)
        {
            sum += _history[^(i + 1)].FrameMs;
        }

        return sum / n;
    }

    private void UpdateSystem(Pf pf, double frameMs, double cpuBusyMs, double gpuBusyMs)
    {
        const double gb = 1024.0 * 1024.0 * 1024.0;
        _recentFps.Add(frameMs);
        if (_recentFps.Count > 60)
        {
            _recentFps.RemoveAt(0);
        }

        double sum = 0;
        for (var i = _recentFps.Count - 1; i >= 0; i--)
        {
            sum += _recentFps[i];
        }

        var fps = sum > 0 ? _recentFps.Count / (sum / 1000.0) : 0;
        var cores = (cpuBusyMs + 3) / frameMs;
        _cpuCores = cores;
        _cpuPercent = Math.Clamp(cores / 12.0 * 100.0, 0, 100);
        _gpuPercent = Math.Clamp(gpuBusyMs / frameMs * 100.0, 0, 100);
        var elapsed = (long)(pf.Present / 1000.0);
        _system = new SystemStats
        {
            Fps = fps,
            PresentFps = fps,
            TargetFps = _sc.Target,
            CpuPercent = _cpuPercent,
            CpuCores = _cpuCores,
            LogicalCores = 12,
            GpuPercent = _gpuPercent,
            GpuSource = "simulated",
            ProcessMemoryBytes = (ulong)(_mem[0] * gb),
            ProcessMemoryKind = "footprint",
            GpuAllocatedBytes = (ulong)(_mem[1] * gb),
            GuestImageCacheBytes = (ulong)(_mem[2] * gb),
            GuestBufferCacheBytes = (ulong)(_mem[3] * gb),
            AllocatedMbPerSecond = pf.Alloc,
            Gen0 = pf.Gc[0],
            Gen1 = pf.Gc[1],
            Gen2 = pf.Gc[2],
            RenderWidth = _sc.ResW,
            RenderHeight = _sc.ResH,
            ElapsedSeconds = elapsed,
            DrawsPerSecond = pf.Draws * fps,
            Backend = "Metal",
            Os = "macOS 27 arm64",
            Title = "Simulated · " + _sc.Label,
        };
    }
}
