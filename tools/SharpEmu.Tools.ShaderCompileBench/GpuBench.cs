// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using SharpEmu.Gpu.Metal;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler.Metal;

// mode=gpu: a synthetic GPU-time benchmark of every translated shader of a title's prewarm
// lists. Every shader is built into a pipeline and run on synthetic, zero (or patterned)
// inputs: a compute kernel is dispatched over a fixed grid, a vertex or mesh program draws a
// fixed vertex count behind a trivial consumer pixel function, and a pixel program draws a
// full-screen quad (several times over) from a synthetic vertex function. The time is the
// command buffer's GPUEndTime - GPUStartTime, one shader per command buffer.
//
// The inputs are NOT the title's: zero or patterned data changes control flow (loop trip
// counts, early-outs, which texels and buffers are touched), so a number is a cost proxy that
// ranks shaders and compares compile variants; it is not production time.
//
// Arguments (key=value, on top of dir=):
//   kinds=cs,vs,ms,ps   which programs (default all)       only=<hash prefixes, comma separated>
//   variants=base,mathsafe,fastmath,fastmath_fnprecise,contract,noinvariance,optsize,early,basez,earlyd,basezd
//   reps=5 warm=2       median of reps after warm-up        fill=zero|pattern
//   groups=256 verts=393216 quads=4 width=1920 height=1080  the fixed workload
//   deep=30 deepreps=15 re-measure the slowest base shaders with more repetitions
//   reps_from=<csv>     run exactly the item indices of an earlier run (translator variants by environment)
//   dumpmsl=<dir> [dumponly=1]  write every item's MSL as <index>.metal;  alts=<name>:<dir>,...  variants made of such a dump
//   check=1             hash what each variant stores (buffers and colour targets) to find a change of output
//   alias=<name>        write the base variant under another name (a run under a translator environment variable)
//   tag=<text> out=<dir> workers=3 limit=0
internal static unsafe class GpuBench
{
    private const nuint ResourceShared = 0;
    private const nuint UsageRead = 1;
    private const nuint UsageWrite = 2;
    private const nuint StagesAll = 15;

    private static Func<string, string, string> _arg = null!;
    private static nint _device;
    private static nint _queue;
    private static Synth _synth = null!;

    private static string Arg(string key, string fallback) => _arg(key, fallback);

    private static int Int(string key, int fallback) => int.Parse(Arg(key, fallback.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);

    private sealed class Item
    {
        public int Index;
        public string Kind = "";
        public ulong Hash;
        public string Source = "";
        public Gen5MslShader Shader = null!;
        public string Body = "";
        public int Specialization;
        public string Id => $"{Kind}=0x{Hash:X16}#{Index}";
        public bool IsPixel => Kind == "ps";
    }

    // One way of building a shader: compile options, a source patch, and the scenario it is run in.
    private sealed record Variant(string Name, string Scenario, Func<Item, bool> Applies, Func<string, string> Patch, Action<nint> Options)
    {
        public string? AltDirectory { get; init; }
    }

    private sealed class Built
    {
        public Variant Variant = null!;
        public nint Library, Function, State;
        public string? Error;
        public double CompileMs;
        public string Notes = "";
        public uint MaxThreads, ExecWidth, TgBytes;
        public string Source = "";
    }

    private sealed class Row
    {
        public string Phase = "", Variant = "", Scenario = "", Status = "ok", Note = "";
        public Item Item = null!;
        public double Median, Min, Max;
        public int Reps;
        public double Scale = 1;
    }

    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelAlloc = MetalNative.Selector("alloc");
    private static readonly nint SelInit = MetalNative.Selector("init");
    private static readonly nint SelObjectAt = MetalNative.Selector("objectAtIndexedSubscript:");

    public static int Run(string dir, Func<string, string, string> arg)
    {
        _arg = arg;
        MetalGpuBackend.Register();
        MetalRenderHost.ApplyTranslationSwitches();
        _device = MetalNative.MTLCreateSystemDefaultDevice();
        _queue = MetalNative.Send(_device, MetalNative.Selector("newCommandQueue"));
        _synth = new Synth(_device, Arg("fill", "zero") == "pattern");
        var kinds = Arg("kinds", "cs,vs,ms,ps").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet();
        var only = Arg("only", "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(text => text.ToUpperInvariant()).ToArray();
        var items = LoadItems(dir, kinds, only);
        if (Arg("dumpmsl", "") is { Length: > 0 } dump)
        {
            // The MSL of every item by index: a run under a translator environment variable writes its set here, and a later
            // run reads it back as a variant (alts=), so both translations are measured in one process, interleaved.
            Directory.CreateDirectory(dump);
            foreach (var item in items)
            {
                File.WriteAllText(Path.Combine(dump, $"{item.Index}.metal"), item.Source);
            }

            File.WriteAllLines(Path.Combine(dump, "index.txt"), items.Select(item => $"{item.Index} {item.Kind} {item.Hash:X16}"));

            if (Arg("dumponly", "0") == "1")
            {
                return 0;
            }
        }

        var variants = Variants();
        foreach (var alt in Arg("alts", "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var split = alt.IndexOf(':');
            var altDir = alt[(split + 1)..];
            variants[alt[..split]] = new Variant(alt[..split], "none",
                item => File.Exists(Path.Combine(altDir, $"{item.Index}.metal")) && File.ReadAllText(Path.Combine(altDir, $"{item.Index}.metal")) != item.Source,
                _ => throw new InvalidOperationException(), _ => { }) { AltDirectory = altDir };
        }

        var names = Arg("variants", "base").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var chosen = names.Select(name => variants.TryGetValue(name, out var found) ? found : throw new ArgumentException($"unknown variant {name}")).ToList();
        var tag = Arg("tag", "run");
        var outDir = Arg("out", "gpu-bench");
        Directory.CreateDirectory(outDir);
        Console.Error.WriteLine($"gpu bench: items={items.Count} variants={string.Join(',', names)} fill={Arg("fill", "zero")} tag={tag}");
        WriteStatic(Path.Combine(outDir, $"static-{tag}.csv"), items);

        var reps = Int("reps", 5);
        var rows = new List<Row>();
        var csv = Path.Combine(outDir, $"gpu-{tag}.csv");
        RunPhase("full", items, chosen, reps, rows, csv);
        var deep = Int("deep", 0);
        if (deep > 0)
        {
            var baseName = chosen[0].Name;
            var top = rows.Where(row => row.Phase == "full" && row.Variant == baseName && row.Status == "ok")
                .OrderByDescending(row => row.Median).Take(deep).Select(row => row.Item).ToList();
            RunPhase("deep", top, chosen, Int("deepreps", 15), rows, csv);
        }

        return 0;
    }

    // ---- items -------------------------------------------------------------------------------

    private static List<Item> LoadItems(string dir, HashSet<string> kinds, string[] only)
    {
        var compiler = SharpEmu.Libs.Gpu.GuestGpu.Current;
        var items = new List<Item>();
        var seen = new HashSet<ulong>();
        HashSet<int>? wanted = null;
        if (Arg("reps_from", "") is { Length: > 0 } from)
        {
            wanted = File.ReadLines(from).Skip(1).Select(line => line.Split(',')).Where(f => f.Length > 3 && f[0] == "full").Select(f => int.Parse(f[1], CultureInfo.InvariantCulture)).ToHashSet();
        }

        void Add(int index, ulong hash, IGuestCompiledShaderAdapter shader)
        {
            var metal = shader.Shader;
            var kind = metal.Stage switch
            {
                Gen5MslStage.Compute => "cs",
                Gen5MslStage.Vertex => "vs",
                Gen5MslStage.Mesh => "ms",
                _ => "ps",
            };
            if (!kinds.Contains(kind) || (only.Length > 0 && !only.Any(prefix => $"{hash:X16}".StartsWith(prefix, StringComparison.Ordinal))))
            {
                return;
            }

            if (wanted is null)
            {
                var key = MetalLibraryPrewarm.Key(metal.Source, metal.UsesPerVertexInputs) ^ (ulong)metal.Stage << 60;
                if (!seen.Add(key))
                {
                    return;
                }
            }
            else if (!wanted.Contains(index))
            {
                return;
            }

            var entry = $"{metal.EntryPoint}(";
            var at = metal.Source.IndexOf(" " + entry, StringComparison.Ordinal);
            items.Add(new Item
            {
                Index = index,
                Kind = kind,
                Hash = hash,
                Source = metal.Source,
                Shader = metal,
                Body = at >= 0 ? metal.Source[at..] : metal.Source,
            });
        }

        using (var list = ShaderPrewarmList.Open(dir) ?? throw new InvalidOperationException("no compute list"))
        {
            var computes = list.LoadedComputes();
            for (var index = 0; index < computes.Count; index++)
            {
                var (record, code) = computes[index];
                if (ShaderPrewarmCompiler.TryCompilePrewarm(record, code, compiler,
                        MetalRenderHost.PrewarmSharedInt64Atomics, MetalRenderHost.PrewarmExecGuardElision,
                        MetalRenderHost.PrewarmSignedZeroInfNanPreserve, out var compiled, out _, out _) &&
                    compiled is MetalCompiledGuestShader metal)
                {
                    Add(index, record.Hash, new IGuestCompiledShaderAdapter(metal.Shader));
                }
            }
        }

        using (var list = GraphicsPrewarmList.Open(dir) ?? throw new InvalidOperationException("no graphics list"))
        {
            var graphics = list.Loaded();
            for (var index = 0; index < graphics.Count; index++)
            {
                var record = GraphicsPrewarmList.Decode(graphics[index]);
                if (ShaderPrewarmCompiler.TryCompileGraphicsPrewarm(graphics[index], compiler, out var compiled, out _) &&
                    compiled is MetalCompiledGuestShader metal)
                {
                    Add(10000 + index, record.Hash, new IGuestCompiledShaderAdapter(metal.Shader));
                }
            }
        }

        var limit = Int("limit", 0);
        return limit > 0 ? items.Take(limit).ToList() : items;
    }

    private readonly record struct IGuestCompiledShaderAdapter(Gen5MslShader Shader);

    // ---- static metrics -----------------------------------------------------------------------

    private static readonly Regex SampleRx = new(@"\.(sample|sample_compare|gather|gather_compare)\(", RegexOptions.Compiled);
    private static readonly Regex ReadRx = new(@"\.read\(", RegexOptions.Compiled);
    private static readonly Regex WriteRx = new(@"\.write\(", RegexOptions.Compiled);
    private static readonly Regex LoadRx = new(@"sharpemu_load_|\bb\d+\[|sharpemu_resolve_device_", RegexOptions.Compiled);
    private static readonly Regex StoreRx = new(@"sharpemu_store_|\bb\d+\[[^\]]*\]\s*=[^=]|atomic_[a-z_]*explicit\(|sharpemu_atomic", RegexOptions.Compiled);
    private static readonly Regex LoopRx = new(@"\b(while|for)\s*\(", RegexOptions.Compiled);
    private static readonly Regex BranchRx = new(@"\bif\s*\(", RegexOptions.Compiled);
    private static readonly Regex BarrierRx = new(@"threadgroup_barrier|simdgroup_barrier", RegexOptions.Compiled);
    private static readonly Regex AtomicRx = new(@"atomic_[a-z_]*explicit\(", RegexOptions.Compiled);
    private static readonly Regex SimdRx = new(@"\bsimd_[a-z_]+\(", RegexOptions.Compiled);

    private static void WriteStatic(string path, List<Item> items)
    {
        var text = new StringBuilder("idx,id,kind,hash,msl_bytes,lines,tex_samples,tex_reads,tex_writes,dev_loads,stores_atomics,loops,branches,barriers,simd_ops,discard,depth_out,early_attr,tg_size,mesh_tg_bytes\n");
        foreach (var item in items)
        {
            var body = item.Body;
            var shader = item.Shader;
            text.Append(CultureInfo.InvariantCulture,
                $"{item.Index},{item.Id},{item.Kind},0x{item.Hash:X16},{item.Source.Length},{item.Source.Count(c => c == '\n')}," +
                $"{SampleRx.Count(body)},{ReadRx.Count(body)},{WriteRx.Count(body)},{LoadRx.Count(body)},{StoreRx.Count(body) + WriteRx.Count(body)}," +
                $"{LoopRx.Count(body)},{BranchRx.Count(body)},{BarrierRx.Count(body)},{SimdRx.Count(body)},{(body.Contains("discard_fragment", StringComparison.Ordinal) ? 1 : 0)}," +
                $"{(item.Source.Contains("[[depth(", StringComparison.Ordinal) ? 1 : 0)},{(item.Source.Contains("[[early_fragment_tests]]", StringComparison.Ordinal) ? 1 : 0)}," +
                $"{shader.ThreadgroupSizeX}x{shader.ThreadgroupSizeY}x{shader.ThreadgroupSizeZ},{shader.MeshInfo?.ThreadgroupMemoryBytes ?? 0}\n");
        }

        File.WriteAllText(path, text.ToString());
    }

    // ---- variants -----------------------------------------------------------------------------

    private static Dictionary<string, Variant> Variants()
    {
        static void SetMode(nint options, nuint mode, nuint functions)
        {
            if (MetalNative.SendBool(options, MetalNative.Selector("respondsToSelector:"), MetalNative.Selector("setMathMode:")))
            {
                MetalNative.SendVoidNUInt(options, MetalNative.Selector("setMathMode:"), mode);
                if (MetalNative.SendBool(options, MetalNative.Selector("respondsToSelector:"), MetalNative.Selector("setMathFloatingPointFunctions:")))
                {
                    MetalNative.SendVoidNUInt(options, MetalNative.Selector("setMathFloatingPointFunctions:"), functions);
                }
            }
        }

        static bool EarlyCandidate(Item item) =>
            item.IsPixel && !item.Source.Contains("[[early_fragment_tests]]", StringComparison.Ordinal) &&
            !item.Body.Contains("discard_fragment", StringComparison.Ordinal) &&
            !item.Source.Contains("[[depth(", StringComparison.Ordinal) && !item.Source.Contains("[[sample_mask", StringComparison.Ordinal);

        // The same, for the pixel functions that do discard: the translator leaves those on Metal's late order, since a discarded
        // fragment would already have written depth. Only a cost bound - not a semantics-preserving change where depth is written.
        static bool EarlyDiscardCandidate(Item item) =>
            item.IsPixel && !item.Source.Contains("[[early_fragment_tests]]", StringComparison.Ordinal) &&
            item.Body.Contains("discard_fragment", StringComparison.Ordinal) &&
            !item.Source.Contains("[[depth(", StringComparison.Ordinal) && !item.Source.Contains("[[sample_mask", StringComparison.Ordinal);

        var all = new[]
        {
            new Variant("earlyd", "zreject", EarlyDiscardCandidate,
                s => s.Replace("fragment Gen5PsOut gen5_ps(", "[[early_fragment_tests]] fragment Gen5PsOut gen5_ps(", StringComparison.Ordinal), _ => { }),
            new Variant("basezd", "zreject", EarlyDiscardCandidate, s => s, _ => { }),
            // The emulator's configuration: MTLMathModeRelaxed, precise library functions, invariance preserved.
            new Variant("base", "none", _ => true, s => s, _ => { }),
            // SHARPEMU_MSL_MATHMODE=safe.
            new Variant("mathsafe", "none", _ => true, s => s, o => SetMode(o, 0, 1)),
            // setFastMathEnabled:true, the MTLMathModeFast of the old switch: unsafe for NaN/inf/signed zero.
            new Variant("fastmath", "none", _ => true, s => s, o =>
            {
                MetalNative.SendVoidBool(o, MetalNative.Selector("setFastMathEnabled:"), true);
                SetMode(o, 2, 0);
            }),
            // Fast mode with the transcendentals kept precise (the part the NaN/inf semantics rely on).
            new Variant("fastmath_fnprecise", "none", _ => true, s => s, o => SetMode(o, 2, 1)),
            // The source's #pragma clang fp contract(off) removed: multiply-add fusion under the relaxed mode.
            new Variant("contract", "none", item => item.Source.Contains("fp contract(off)", StringComparison.Ordinal),
                s => s.Replace("#pragma clang fp contract(off)", "#pragma clang fp contract(fast)", StringComparison.Ordinal), _ => { }),
            // preserveInvariance off: the positions are no longer invariant across pipelines.
            new Variant("noinvariance", "none", item => item.Kind is "vs" or "ms", s => s,
                o => MetalNative.SendVoidBool(o, MetalNative.Selector("setPreserveInvariance:"), false)),
            new Variant("optsize", "none", _ => true, s => s,
                o => MetalNative.SendVoidNUInt(o, MetalNative.Selector("setOptimizationLevel:"), 1)),
            // [[early_fragment_tests]] on a pixel function that has neither discard nor a depth/mask export, run against a
            // depth buffer that rejects every fragment (the best case), and its baseline in the same scenario.
            new Variant("early", "zreject", EarlyCandidate,
                s => s.Replace("fragment Gen5PsOut gen5_ps(", "[[early_fragment_tests]] fragment Gen5PsOut gen5_ps(", StringComparison.Ordinal), _ => { }),
            new Variant("basez", "zreject", EarlyCandidate, s => s, _ => { }),
        };
        return all.ToDictionary(variant => variant.Name);
    }

    // ---- phases -------------------------------------------------------------------------------

    private static void RunPhase(string phase, List<Item> items, List<Variant> variants, int reps, List<Row> rows, string csvPath)
    {
        var warm = Int("warm", 2);
        var workers = Int("workers", 3);
        var ahead = 2;
        var pending = new Dictionary<int, Task<Built[]>>();
        var scheduler = new ConcurrentExclusiveSchedulerPair(TaskScheduler.Default, workers).ConcurrentScheduler;
        Task<Built[]> Start(int index) => Task.Factory.StartNew(() => CompileItem(items[index], variants, workers), CancellationToken.None, TaskCreationOptions.LongRunning, scheduler);
        for (var i = 0; i < Math.Min(ahead, items.Count); i++)
        {
            pending[i] = Start(i);
        }

        var started = Stopwatch.StartNew();
        for (var index = 0; index < items.Count; index++)
        {
            var item = items[index];
            if (index + ahead < items.Count)
            {
                pending[index + ahead] = Start(index + ahead);
            }

            var built = pending[index].Result;
            pending.Remove(index);
            var produced = new List<Row>();
            try
            {
                produced.AddRange(Measure(phase, item, built, warm, reps));
            }
            catch (Exception exception)
            {
                produced.Add(new Row { Phase = phase, Item = item, Variant = "base", Status = "exception", Note = exception.Message });
            }
            finally
            {
                foreach (var b in built)
                {
                    foreach (var handle in (ReadOnlySpan<nint>)[b.State, b.Function, b.Library])
                    {
                        if (handle != 0)
                        {
                            MetalNative.SendVoid(handle, SelRelease);
                        }
                    }
                }
            }

            rows.AddRange(produced);
            var best = produced.Where(r => r.Status == "ok").Select(r => r.Median).DefaultIfEmpty(0).First();
            Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"[{phase} {index + 1}/{items.Count} {started.Elapsed.TotalSeconds:F0}s] {item.Id} {string.Join(' ', produced.Select(r => r.Status == "ok" ? $"{r.Variant}={r.Median:F0}us" : $"{r.Variant}:{r.Status}"))}"));
            if ((index + 1) % 25 == 0 || index + 1 == items.Count)
            {
                WriteCsv(csvPath, rows);
            }
        }
    }

    private static void WriteCsv(string path, List<Row> rows)
    {
        var text = new StringBuilder("phase,idx,id,kind,hash,variant,scenario,median_us,min_us,max_us,reps,scale,status,note\n");
        foreach (var row in rows)
        {
            text.Append(CultureInfo.InvariantCulture,
                $"{row.Phase},{row.Item.Index},{row.Item.Id},{row.Item.Kind},0x{row.Item.Hash:X16},{(row.Variant == "base" && Arg("alias", "").Length > 0 ? Arg("alias", "") : row.Variant)},{row.Scenario},{row.Median:F1},{row.Min:F1},{row.Max:F1},{row.Reps},{row.Scale:F1},{row.Status},\"{row.Note.Replace('"', '\'').Replace('\n', ' ')}\"\n");
        }

        File.WriteAllText(path, text.ToString());
    }

    // ---- compile ------------------------------------------------------------------------------

    private static Built[] CompileItem(Item item, List<Variant> variants, int workers)
    {
        var result = new Built[variants.Count];
        Parallel.For(0, variants.Count, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, workers) }, i =>
        {
            var variant = variants[i];
            var built = new Built { Variant = variant };
            result[i] = built;
            if (!variant.Applies(item))
            {
                built.Notes = "n/a";
                return;
            }

            var pool = MetalNative.objc_autoreleasePoolPush();
            try
            {
                var source = variant.AltDirectory is { } altDirectory ? File.ReadAllText(Path.Combine(altDirectory, $"{item.Index}.metal")) : variant.Patch(item.Source);
                built.Source = source;
                var t0 = Stopwatch.GetTimestamp();
                built.Library = CompileLibrary(source, item.Shader.UsesPerVertexInputs, variant.Options, out built.Error);
                if (built.Library == 0)
                {
                    return;
                }

                nint error = 0;
                if (item.Shader.UsesMipStatisticsConstant)
                {
                    built.Function = MetalRenderHost.NewMipStatisticsFunction(built.Library, item.Shader.EntryPoint, false, out error);
                }
                else
                {
                    built.Function = MetalNative.Send(built.Library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(item.Shader.EntryPoint));
                }

                if (built.Function == 0)
                {
                    built.Error = "no entry point " + MetalNative.DescribeError(error);
                    return;
                }

                built.State = item.Kind == "cs" ? NewComputeState(built) : NewRenderState(item, built);
                built.CompileMs = (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;
            }
            catch (Exception exception)
            {
                built.Error = exception.Message;
            }
            finally
            {
                MetalNative.objc_autoreleasePoolPop(pool);
            }
        });
        return result;
    }

    private static nint CompileLibrary(string source, bool perVertex, Action<nint> adjust, out string? error)
    {
        error = null;
        var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), SelAlloc), SelInit);
        try
        {
            MetalNative.ApplyMslCompileOptions(options);
            if (perVertex)
            {
                MetalNative.SetMslLanguageVersion(options, Gen5MslShader.PerVertexInputsLanguageVersion);
            }

            adjust(options);
            nint compileError = 0;
            var library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithSource:options:error:"), MetalNative.NsString(source), options, ref compileError);
            if (library == 0)
            {
                error = "library: " + MetalNative.DescribeError(compileError);
            }

            return library;
        }
        finally
        {
            MetalNative.SendVoid(options, SelRelease);
        }
    }

    private static readonly ConcurrentDictionary<string, nint> Partners = new();

    // A small helper library (the synthetic vertex or consumer pixel function) shared by every item that needs the same text.
    private static nint PartnerFunction(string source, string entry)
    {
        var library = Partners.GetOrAdd(source, text =>
        {
            var lib = CompileLibrary(text, false, _ => { }, out var error);
            if (lib == 0)
            {
                throw new InvalidOperationException("partner " + error);
            }

            return lib;
        });
        return MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(entry));
    }

    private static nint NewComputeState(Built built)
    {
        nint error = 0;
        var state = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), built.Function, ref error);
        if (state == 0)
        {
            built.Error = "pipeline: " + MetalNative.DescribeError(error);
            return 0;
        }

        built.MaxThreads = (uint)MetalNative.SendUInteger(state, MetalNative.Selector("maxTotalThreadsPerThreadgroup"));
        built.ExecWidth = (uint)MetalNative.SendUInteger(state, MetalNative.Selector("threadExecutionWidth"));
        built.TgBytes = (uint)MetalNative.SendUInteger(state, MetalNative.Selector("staticThreadgroupMemoryLength"));
        return state;
    }

    // ---- synthetic partner stages -------------------------------------------------------------

    private static List<(string Type, string Name, string Attrs)> ParseStruct(string source, string name)
    {
        var members = new List<(string, string, string)>();
        var start = source.IndexOf($"struct {name}\n", StringComparison.Ordinal);
        if (start < 0)
        {
            return members;
        }

        var end = source.IndexOf("};", start, StringComparison.Ordinal);
        foreach (var line in source[start..end].Split('\n').Skip(2))
        {
            var match = Regex.Match(line, @"^\s+([A-Za-z0-9_:<>]+)\s+([A-Za-z0-9_]+)\s*((?:\[\[[^\]]*\]\]\s*)*)(?:\[\d+\])?\s*;");
            if (match.Success)
            {
                members.Add((match.Groups[1].Value, match.Groups[2].Value, match.Groups[3].Value.Trim()));
            }
        }

        return members;
    }

    private static string UserAttr(string attrs) => Regex.Match(attrs, @"user\(locn(\d+)\)") is { Success: true } m ? m.Groups[1].Value : "";

    private static nint NewRenderState(Item item, Built built)
    {
        var shader = item.Shader;
        var mesh = item.Kind == "ms";
        var descriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class(mesh ? "MTLMeshRenderPipelineDescriptor" : "MTLRenderPipelineDescriptor"), SelAlloc), SelInit);
        try
        {
            var colors = new List<nuint>();
            var depth = false;
            if (item.Kind == "ps")
            {
                foreach (var (type, name, attrs) in ParseStruct(item.Source, "Gen5PsOut"))
                {
                    if (Regex.Match(attrs, @"color\((\d+)\)") is { Success: true } color)
                    {
                        var slot = int.Parse(color.Groups[1].Value, CultureInfo.InvariantCulture);
                        while (colors.Count <= slot)
                        {
                            colors.Add(0);
                        }

                        colors[slot] = type.StartsWith("uint", StringComparison.Ordinal) ? 73u : type.StartsWith("int", StringComparison.Ordinal) ? 74u : 70u;
                    }
                    else if (attrs.Contains("depth(", StringComparison.Ordinal))
                    {
                        depth = true;
                    }
                }

                var vs = SyntheticVertex(item);
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), PartnerFunction(vs, "bench_vs"));
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setFragmentFunction:"), built.Function);
                if (ParseStruct(item.Source, "Gen5PsIn").Any(m => m.Attrs.Contains("render_target_array_index", StringComparison.Ordinal)))
                {
                    MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setInputPrimitiveTopology:"), 3);
                }
            }
            else
            {
                colors.Add(70);
                if (mesh)
                {
                    var info = shader.MeshInfo!;
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setMeshFunction:"), built.Function);
                    MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setMaxTotalThreadsPerMeshThreadgroup:"), Math.Max(info.ThreadsPerThreadgroup, 1));
                }
                else
                {
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), built.Function);
                    var vertexDescriptor = VertexDescriptor(item);
                    if (vertexDescriptor != 0)
                    {
                        MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexDescriptor:"), vertexDescriptor);
                    }
                }

                MetalNative.SendVoid(descriptor, MetalNative.Selector("setFragmentFunction:"), PartnerFunction(ConsumerPixel(item), "bench_ps"));
                if (!mesh && (shader.WritesLayer || shader.WritesViewportIndex))
                {
                    MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setInputPrimitiveTopology:"), 3);
                }
            }

            var attachments = MetalNative.Send(descriptor, MetalNative.Selector("colorAttachments"));
            for (var slot = 0; slot < colors.Count; slot++)
            {
                if (colors[slot] != 0)
                {
                    MetalV2Native.SendVoidValue(MetalNative.SendAtIndex(attachments, SelObjectAt, (nuint)slot), MetalNative.Selector("setPixelFormat:"), colors[slot]);
                }
            }

            var zreject = built.Variant.Scenario == "zreject";
            if (depth || zreject)
            {
                MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setDepthAttachmentPixelFormat:"), 252);
            }

            nint error = 0;
            var state = MetalNative.SendNewMeshPipeline(
                _device,
                MetalNative.Selector(mesh ? "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:" : "newRenderPipelineStateWithDescriptor:options:reflection:error:"),
                descriptor, 0, 0, ref error);
            if (state == 0)
            {
                built.Error = "pipeline: " + MetalNative.DescribeError(error);
                return 0;
            }

            if (mesh)
            {
                built.MaxThreads = (uint)shader.MeshInfo!.ThreadsPerThreadgroup;
                built.TgBytes = shader.MeshInfo.ThreadgroupMemoryBytes;
            }

            built.Notes = string.Join('/', colors.Select(c => c.ToString(CultureInfo.InvariantCulture))) + (depth ? "+z" : "");
            if (mesh)
            {
                var info = shader.MeshInfo!;
                built.Notes = $"groups={Gen5MslMeshLaunch.SubgroupCount(info, Gen5MslMeshInputTopology.TriangleList, MeshVertices(info, (uint)Int("verts", 393216)))} tg={info.ThreadsPerThreadgroup} verts={info.MaxVertices} prims={info.MaxPrimitives} amplified={info.Amplified} gs={info.HasGeometryHalf} devicelds={info.DeviceLds} wave64={info.Wave64}";
            }
            return state;
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }
    }

    // A vertex function that draws a full-screen quad and feeds every interpolant the pixel program declares.
    private static string SyntheticVertex(Item item)
    {
        var input = ParseStruct(item.Source, "Gen5PsIn");
        var text = new StringBuilder("#include <metal_stdlib>\nusing namespace metal;\nstruct V {\n    float4 position [[position]];\n");
        var users = input.Select(m => UserAttr(m.Attrs)).Where(n => n.Length > 0).Distinct().ToList();
        foreach (var user in users)
        {
            text.Append(CultureInfo.InvariantCulture, $"    float4 a{user} [[user(locn{user})]];\n");
        }

        if (input.Any(m => m.Attrs.Contains("render_target_array_index", StringComparison.Ordinal)))
        {
            text.Append("    uint layer [[render_target_array_index]];\n");
        }

        text.Append("};\nvertex V bench_vs(uint vid [[vertex_id]]) {\n");
        text.Append("    const float2 q[6] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,-1), float2(1,1), float2(-1,1) };\n");
        text.Append("    float2 p = q[vid % 6u]; V o; o.position = float4(p, 0.5, 1.0); float4 v = float4(p * 0.5 + 0.5, 0.5, 1.0);\n");
        foreach (var user in users)
        {
            text.Append(CultureInfo.InvariantCulture, $"    o.a{user} = v;\n");
        }

        if (input.Any(m => m.Attrs.Contains("render_target_array_index", StringComparison.Ordinal)))
        {
            text.Append("    o.layer = 0u;\n");
        }

        return text.Append("    return o;\n}\n").ToString();
    }

    // A pixel function that reads every interpolant a vertex or mesh program exports, so none of its output is dead.
    private static string ConsumerPixel(Item item)
    {
        var outputs = ParseStruct(item.Source, item.Kind == "ms" ? "Gen5MeshVertex" : "Gen5VsOut");
        var users = outputs.Select(m => UserAttr(m.Attrs)).Where(n => n.Length > 0).Distinct().ToList();
        var text = new StringBuilder("#include <metal_stdlib>\nusing namespace metal;\nstruct F {\n    float4 position [[position]];\n");
        foreach (var user in users)
        {
            text.Append(CultureInfo.InvariantCulture, $"    float4 a{user} [[user(locn{user})]];\n");
        }

        text.Append("};\nfragment float4 bench_ps(F in [[stage_in]]) {\n    float4 s = in.position * 0.0;\n");
        foreach (var user in users)
        {
            text.Append(CultureInfo.InvariantCulture, $"    s += in.a{user};\n");
        }

        return text.Append("    return s;\n}\n").ToString();
    }

    private static nint VertexDescriptor(Item item)
    {
        var attributes = ParseStruct(item.Source, "Gen5VsIn");
        if (attributes.Count == 0)
        {
            return 0;
        }

        var descriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLVertexDescriptor"), SelAlloc), SelInit);
        var attributeArray = MetalNative.Send(descriptor, MetalNative.Selector("attributes"));
        var layouts = MetalNative.Send(descriptor, MetalNative.Selector("layouts"));
        foreach (var (type, _, attrs) in attributes)
        {
            var match = Regex.Match(attrs, @"attribute\((\d+)\)");
            if (!match.Success)
            {
                continue;
            }

            var index = int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
            if (index + VertexSlotBase >= 31)
            {
                throw new InvalidOperationException("too many vertex attributes");
            }

            var attribute = MetalNative.SendAtIndex(attributeArray, SelObjectAt, (nuint)index);
            MetalV2Native.SendVoidValue(attribute, MetalNative.Selector("setFormat:"), VertexFormat(type));
            MetalV2Native.SendVoidValue(attribute, MetalNative.Selector("setOffset:"), 0);
            MetalV2Native.SendVoidValue(attribute, MetalNative.Selector("setBufferIndex:"), (nuint)(index + VertexSlotBase));
            var layout = MetalNative.SendAtIndex(layouts, SelObjectAt, (nuint)(index + VertexSlotBase));
            MetalV2Native.SendVoidValue(layout, MetalNative.Selector("setStride:"), 16);
            MetalV2Native.SendVoidValue(layout, MetalNative.Selector("setStepFunction:"), 1);
        }

        return descriptor;
    }

    private const int VertexSlotBase = 2;

    private static nuint VertexFormat(string type)
    {
        var lanes = type[^1] is >= '1' and <= '4' ? type[^1] - '0' : 1;
        if (type.StartsWith("half", StringComparison.Ordinal))
        {
            return (nuint)(25 + Math.Max(lanes, 2) - 2);
        }

        var family = type.StartsWith("uint", StringComparison.Ordinal) ? 36 : type.StartsWith("int", StringComparison.Ordinal) ? 32 : 28;
        return (nuint)(family + lanes - 1);
    }

    // ---- synthetic bindings -------------------------------------------------------------------

    private sealed class Synth
    {
        public const int PoolBuffers = 4;
        public const uint PoolBytes = 32 << 20;
        private readonly nint _device;
        private readonly bool _pattern;
        private readonly List<nint> _resources = [];
        private readonly Dictionary<string, nint> _textures = new();
        private readonly object _gate = new();
        public readonly nint[] Pool = new nint[PoolBuffers];
        public nint Gds, Fault, Flattened, ShaderData, MipStats, StepReport, PairLocks, PageTable, Push, VertexStream, Indices, LdsPool, Meshlets, DrawInfo;
        public nint Sampler;
        public ulong SamplerId;
        public uint PageTableEntries;
        public nint[] Targets = [];
        public nint Depth;
        public nint SmallTarget;

        public Synth(nint device, bool pattern)
        {
            _device = device;
            _pattern = pattern;
            for (var i = 0; i < PoolBuffers; i++)
            {
                Pool[i] = Buffer(PoolBytes, patternable: true);
            }

            Gds = Buffer(64 << 10, true);
            Fault = Buffer(64 << 10, false);
            Flattened = Buffer(16 << 10, true);
            ShaderData = Buffer(2 << 10, true);
            MipStats = Buffer(1 << 20, false);
            StepReport = Buffer(4 << 10, false);
            PairLocks = Buffer((uint)Gen5MslArgumentLayout.PairLockTableWords * 4, false);
            Push = Buffer(4 << 10, true);
            VertexStream = Buffer(16u * 1_048_576, true);
            Indices = Buffer(16, false);
            Meshlets = Buffer(16, false);
            DrawInfo = Buffer(256, false);
            LdsPool = Buffer((uint)Gen5MslMeshInfo.LdsPoolByteSize, false);
            // 64 ranges of 16 MB, all backed by the first 16 MB of one pool buffer: every guest address under 1 GB
            // resolves (through a real binary search over 64 entries), none reaches past the backing.
            const int entries = 64;
            const ulong pages = (16UL << 20) >> 14;
            PageTable = Buffer(entries * 24, false);
            var span = new Span<byte>((void*)MetalNative.Send(PageTable, MetalNative.Selector("contents")), entries * 24);
            var baseAddress = MetalNative.SendGpuAddress(Pool[0], MetalNative.Selector("gpuAddress"));
            for (var i = 0; i < entries; i++)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(span[(i * 24)..], (ulong)i * pages);
                BinaryPrimitives.WriteUInt64LittleEndian(span[(i * 24 + 8)..], pages);
                BinaryPrimitives.WriteUInt64LittleEndian(span[(i * 24 + 16)..], baseAddress);
            }

            PageTableEntries = entries;
            var sampler = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLSamplerDescriptor"), SelAlloc), SelInit);
            MetalNative.SendVoidBool(sampler, MetalNative.Selector("setSupportArgumentBuffers:"), true);
            MetalNative.SendVoidNUInt(sampler, MetalNative.Selector("setMinFilter:"), 1);
            MetalNative.SendVoidNUInt(sampler, MetalNative.Selector("setMagFilter:"), 1);
            MetalNative.SendVoidNUInt(sampler, MetalNative.Selector("setMipFilter:"), 2);
            MetalNative.SendVoidNUInt(sampler, MetalNative.Selector("setSAddressMode:"), 2);
            MetalNative.SendVoidNUInt(sampler, MetalNative.Selector("setTAddressMode:"), 2);
            MetalNative.SendVoidNUInt(sampler, MetalNative.Selector("setRAddressMode:"), 2);
            Sampler = MetalNative.Send(_device, MetalNative.Selector("newSamplerStateWithDescriptor:"), sampler);
            SamplerId = MetalNative.SendGpuResourceId(Sampler, MetalNative.Selector("gpuResourceID"));
            SmallTarget = RenderTarget(70, 256, 256, 0);
        }

        public nint Buffer(uint bytes, bool patternable)
        {
            var buffer = MetalNative.SendNewBuffer(_device, MetalNative.Selector("newBufferWithLength:options:"), bytes, ResourceShared);
            var contents = (byte*)MetalNative.Send(buffer, MetalNative.Selector("contents"));
            new Span<byte>(contents, (int)Math.Min(bytes, int.MaxValue)).Clear();
            if (_pattern && patternable)
            {
                Fill(contents, bytes);
            }

            _resources.Add(buffer);
            return buffer;
        }

        // Words of 0..7: small enough that a loop bound or an offset read from memory stays bounded.
        private static void Fill(byte* contents, uint bytes)
        {
            var words = (uint*)contents;
            for (uint i = 0; i < bytes / 4; i++)
            {
                words[i] = (i * 2654435761u) >> 29;
            }
        }

        private nint[] Writable => [.. Pool, Gds, Fault, MipStats, StepReport, PairLocks];

        // The buffers a shader can write back to their start state: zero, or the pattern.
        public void ResetOutputs()
        {
            foreach (var buffer in Writable)
            {
                var length = (int)MetalNative.SendUInteger(buffer, MetalNative.Selector("length"));
                var contents = (byte*)MetalNative.Send(buffer, MetalNative.Selector("contents"));
                if (_pattern && (buffer == Gds || Pool.Contains(buffer)))
                {
                    Fill(contents, (uint)length);
                }
                else
                {
                    new Span<byte>(contents, length).Clear();
                }
            }
        }

        public string OutputHash()
        {
            var hash = new System.IO.Hashing.XxHash64();
            foreach (var buffer in Writable)
            {
                var length = (int)MetalNative.SendUInteger(buffer, MetalNative.Selector("length"));
                hash.Append(new ReadOnlySpan<byte>((void*)MetalNative.Send(buffer, MetalNative.Selector("contents")), length));
            }

            return hash.GetCurrentHashAsUInt64().ToString("X16", CultureInfo.InvariantCulture);
        }

        public void Refill()
        {
            if (!_pattern)
            {
                return;
            }

            foreach (var buffer in Pool)
            {
                Fill((byte*)MetalNative.Send(buffer, MetalNative.Selector("contents")), PoolBytes);
            }
        }

        public nint RenderTarget(nuint format, nuint width, nuint height, nuint arrayLength)
        {
            var descriptor = MetalNative.SendTextureDescriptor(MetalNative.Class("MTLTextureDescriptor"), MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"), format, width, height, false);
            MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setUsage:"), 5);
            MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setStorageMode:"), ShareTargets ? 0u : 2u);
            var texture = MetalNative.Send(_device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
            _resources.Add(texture);
            return texture;
        }

        public void EnsureTargets(int count, nuint width, nuint height)
        {
            if (Targets.Length >= count && Depth != 0)
            {
                return;
            }

            if (Depth == 0)
            {
                Depth = RenderTarget(252, width, height, 0);
            }

            var targets = new nint[Math.Max(count, 1)];
            Targets.CopyTo(targets, 0);
            for (var i = Targets.Length; i < targets.Length; i++)
            {
                targets[i] = RenderTarget(70, width, height, 0);
            }

            Targets = targets;
        }

        // A texture for an argument-buffer field of the given name (class_numeric_dimension).
        public nint Texture(string field)
        {
            lock (_gate)
            {
                if (_textures.TryGetValue(field, out var found))
                {
                    return found;
                }

                var parts = field.Split('_');
                var compare = field.StartsWith("sampled_compare", StringComparison.Ordinal);
                var storage = field.StartsWith("storage", StringComparison.Ordinal) || field.StartsWith("atomic", StringComparison.Ordinal);
                var numeric = parts.Contains("uint") || parts.Contains("uint64") ? "uint" : parts.Contains("sint") ? "sint" : "float";
                var format = compare ? 252u : numeric == "uint" ? 123u : numeric == "sint" ? 124u : 70u;
                var dimension = string.Join('_', parts.SkipWhile(p => p is not ("1d" or "2d" or "3d" or "cube")));
                var size = 512u;
                var descriptor = MetalNative.SendTextureDescriptor(MetalNative.Class("MTLTextureDescriptor"), MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"), format, size, dimension.StartsWith("1d", StringComparison.Ordinal) ? 1 : size, !dimension.Contains("ms", StringComparison.Ordinal) && !compare);
                switch (dimension)
                {
                    case "1d":
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setTextureType:"), 0);
                        break;
                    case "1d_array":
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setTextureType:"), 1);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setArrayLength:"), 4);
                        break;
                    case "2d_array" or "cube":
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setTextureType:"), 3);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setArrayLength:"), 6);
                        break;
                    case "3d":
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setTextureType:"), 7);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setDepth:"), 64);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setWidth:"), 64);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setHeight:"), 64);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setMipmapLevelCount:"), 7);
                        break;
                    case "2d_ms":
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setTextureType:"), 4);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setSampleCount:"), 4);
                        break;
                    case "2d_ms_array":
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setTextureType:"), 8);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setSampleCount:"), 4);
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setArrayLength:"), 4);
                        break;
                }

                MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setUsage:"), storage ? 3u : 1u);
                MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setStorageMode:"), dimension.Contains("ms", StringComparison.Ordinal) ? 2u : 0u);
                var texture = MetalNative.Send(_device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
                if (texture == 0)
                {
                    throw new InvalidOperationException($"no texture for {field}");
                }

                _resources.Add(texture);
                _textures[field] = texture;
                return texture;
            }
        }

        public nint[] Resources()
        {
            lock (_gate)
            {
                return _resources.ToArray();
            }
        }
    }

    // Fills the stage argument buffer of a program from its layout, the way the host's WriteArgumentBuffer does.
    private static nint ArgumentBuffer(Gen5MslShader shader)
    {
        var layout = shader.ArgumentLayout;
        if (layout is null || layout.Fields.Count == 0)
        {
            return 0;
        }

        var buffer = _synth.Buffer(Math.Max(layout.ByteSize, 16u), false);
        var span = new Span<byte>((void*)MetalNative.Send(buffer, MetalNative.Selector("contents")), (int)Math.Max(layout.ByteSize, 16u));
        var gpu = (nint handle) => MetalNative.SendGpuAddress(handle, MetalNative.Selector("gpuAddress"));
        foreach (var field in layout.Fields)
        {
            var at = span.Slice((int)field.ByteOffset, (int)field.ByteSize);
            switch (field.FieldKind)
            {
                case MslArgumentFieldKind.BufferPointers:
                    for (var i = 0; i < field.Count; i++)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(at[(i * 8)..], gpu(_synth.Pool[i % Synth.PoolBuffers]));
                    }

                    break;
                case MslArgumentFieldKind.BufferByteCounts:
                    for (var i = 0; i < field.Count; i++)
                    {
                        BinaryPrimitives.WriteUInt32LittleEndian(at[(i * 4)..], Synth.PoolBytes);
                    }

                    break;
                case MslArgumentFieldKind.Textures:
                    var texture = _synth.Texture(field.Name);
                    var id = MetalNative.SendGpuResourceId(texture, MetalNative.Selector("gpuResourceID"));
                    for (var i = 0; i < field.Count; i++)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(at[(i * 8)..], id);
                    }

                    break;
                case MslArgumentFieldKind.Samplers:
                    for (var i = 0; i < field.Count; i++)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(at[(i * 8)..], _synth.SamplerId);
                    }

                    break;
                case MslArgumentFieldKind.Pointer:
                    BinaryPrimitives.WriteUInt64LittleEndian(at, gpu(field.Kind switch
                    {
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.GlobalDataShare => _synth.Gds,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.DeviceAddressPageTable => _synth.PageTable,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.FaultBuffer => _synth.Fault,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.FlattenedResourceTable => _synth.Flattened,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.ShaderData => _synth.ShaderData,
                        _ => _synth.MipStats,
                    }));
                    break;
                case MslArgumentFieldKind.StepReport:
                    BinaryPrimitives.WriteUInt64LittleEndian(at, gpu(_synth.StepReport));
                    break;
                case MslArgumentFieldKind.PairLocks:
                    BinaryPrimitives.WriteUInt64LittleEndian(at, gpu(_synth.PairLocks));
                    break;
                case MslArgumentFieldKind.BindlessImageTables:
                    // Every slot points at one table of 2D float entries, variant 0 (with fill=zero
                    // the table count reads 0, so every key lands on entry 0).
                    const int bindlessEntries = 256;
                    var table = _synth.Buffer(bindlessEntries * 16u, false);
                    var entries = new Span<byte>((void*)MetalNative.Send(table, MetalNative.Selector("contents")), bindlessEntries * 16);
                    var bindlessId = MetalNative.SendGpuResourceId(_synth.Texture("sampled_float_2d"), MetalNative.Selector("gpuResourceID"));
                    for (var i = 0; i < bindlessEntries; i++)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(entries[(i * 16)..], bindlessId);
                        BinaryPrimitives.WriteUInt64LittleEndian(entries[(i * 16 + 8)..], 0);
                    }

                    for (var i = 0; i < field.Count; i++)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(at[(i * 8)..], gpu(table));
                    }

                    break;
                default:
                    BinaryPrimitives.WriteUInt32LittleEndian(at, field.Kind switch
                    {
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.GlobalDataShare => 64u << 10,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.DeviceAddressPageTable => _synth.PageTableEntries,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.FaultBuffer => 16u << 10,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.FlattenedResourceTable => 4096u,
                        SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.ShaderData => 512u,
                        _ => 0u,
                    });
                    break;
            }
        }

        return buffer;
    }

    // ---- measurement --------------------------------------------------------------------------

    private static IEnumerable<Row> Measure(string phase, Item item, Built[] built, int warm, int reps)
    {
        var rows = new List<Row>();
        var live = built.Where(b => b.State != 0).ToList();
        foreach (var b in built.Where(b => b.State == 0))
        {
            rows.Add(new Row { Phase = phase, Item = item, Variant = b.Variant.Name, Scenario = b.Variant.Scenario, Status = b.Notes == "n/a" ? "n/a" : "compile-failed", Note = b.Error ?? b.Notes });
        }

        if (live.Count == 0)
        {
            return rows;
        }

        _synth.Refill();
        var arguments = ArgumentBuffer(item.Shader);
        var samples = live.ToDictionary(b => b, _ => new List<double>());
        var scale = 1.0;
        var failure = new Dictionary<Built, string>();
        // A probe at 1/64 of the work first: a runaway shader is stopped before it holds the GPU.
        var probe = Run(item, live[0], arguments, 1.0 / 64, out var probeStatus);
        if (probeStatus != "ok")
        {
            foreach (var b in live)
            {
                rows.Add(new Row { Phase = phase, Item = item, Variant = b.Variant.Name, Scenario = b.Variant.Scenario, Status = "run-failed", Note = probeStatus });
            }

            return rows;
        }

        var fraction = probe * 64 > 1_500_000 ? 1.0 / 16 : 1.0;
        if (probe * 64 > 20_000_000)
        {
            fraction = 1.0 / 64;
        }

        scale = 1.0 / fraction;
        // check=1: the bytes every writable synthetic buffer holds after one run from a reset state, per variant, so a
        // translator change that alters what a shader stores shows as a different out= hash (atomics racing between
        // threads can differ run to run; the same program run twice is the reference).
        var outputs = new Dictionary<Built, string>();
        if (Arg("check", "0") == "1")
        {
            foreach (var b in live)
            {
                _clearTargets = true;
                var targets = item.IsPixel ? ColorCount(item) : 0;
                _synth.ResetOutputs();
                var status = "";
                Run(item, b, arguments, fraction, out status);
                var first = status == "ok" ? _synth.OutputHash() + (targets > 0 ? "+" + TargetHash(targets) : "") : status;
                _synth.ResetOutputs();
                Run(item, b, arguments, fraction, out status);
                var second = status == "ok" ? _synth.OutputHash() + (targets > 0 ? "+" + TargetHash(targets) : "") : status;
                _clearTargets = false;
                outputs[b] = first == second ? first : $"{first}~unstable";
            }
        }

        for (var round = 0; round < warm + reps; round++)
        {
            for (var k = 0; k < live.Count; k++)
            {
                var b = live[(k + round) % live.Count];
                if (failure.ContainsKey(b))
                {
                    continue;
                }

                var micros = Run(item, b, arguments, fraction, out var status);
                if (status != "ok")
                {
                    failure[b] = status;
                    continue;
                }

                if (round >= warm)
                {
                    samples[b].Add(micros);
                }
            }
        }

        foreach (var b in live)
        {
            var list = samples[b];
            if (failure.TryGetValue(b, out var message) || list.Count == 0)
            {
                rows.Add(new Row { Phase = phase, Item = item, Variant = b.Variant.Name, Scenario = b.Variant.Scenario, Status = "run-failed", Note = message ?? "no samples" });
                continue;
            }

            list.Sort();
            rows.Add(new Row
            {
                Phase = phase,
                Item = item,
                Variant = b.Variant.Name,
                Scenario = b.Variant.Scenario,
                Median = list[list.Count / 2] * scale,
                Min = list[0] * scale,
                Max = list[^1] * scale,
                Reps = list.Count,
                Scale = scale,
                Note = $"{b.Notes} max_threads={b.MaxThreads} exec_width={b.ExecWidth} tg_bytes={b.TgBytes} compile_ms={b.CompileMs:F0} msl={MetalLibraryPrewarm.Key(b.Source, item.Shader.UsesPerVertexInputs):X16} out={outputs.GetValueOrDefault(b, "-")}",
            });
        }

        return rows;
    }

    // One command buffer, one dispatch or draw; the GPU time in microseconds of the fraction of the work asked for.
    private static double Run(Item item, Built b, nint arguments, double fraction, out string status)
    {
        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var commands = MetalNative.Send(_queue, MetalNative.Selector("commandBuffer"));
            // Named after the item, so a Metal System Trace of the bench ties each command buffer (and the
            // driver's register-spill events, which name only the buffer) to its shader.
            MetalNative.SendVoid(commands, MetalNative.Selector("setLabel:"), MetalNative.NsString($"{item.Id} {b.Variant.Name}"));
            var resources = _synth.Resources();
            fixed (nint* handles = resources)
            {
                switch (item.Kind)
                {
                    case "cs":
                        EncodeCompute(item, b, arguments, commands, handles, resources.Length, fraction);
                        break;
                    case "ps":
                        EncodePixel(item, b, arguments, commands, handles, resources.Length, fraction);
                        break;
                    default:
                        EncodeGeometry(item, b, arguments, commands, handles, resources.Length, fraction);
                        break;
                }
            }

            MetalNative.SendVoid(commands, MetalNative.Selector("commit"));
            MetalNative.SendVoid(commands, MetalNative.Selector("waitUntilCompleted"));
            var state = MetalNative.SendUInteger(commands, MetalNative.Selector("status"));
            if (state != 4)
            {
                status = "command buffer " + MetalNative.DescribeError(MetalNative.Send(commands, MetalNative.Selector("error")));
                return 0;
            }

            status = "ok";
            return (MetalNative.SendDouble(commands, MetalNative.Selector("GPUEndTime")) - MetalNative.SendDouble(commands, MetalNative.Selector("GPUStartTime"))) * 1e6;
        }
        catch (Exception exception)
        {
            status = exception.Message;
            return 0;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    private static void EncodeCompute(Item item, Built b, nint arguments, nint commands, nint* handles, int count, double fraction)
    {
        var encoder = MetalNative.Send(commands, MetalNative.Selector("computeCommandEncoder"));
        MetalNative.SendVoid(encoder, MetalNative.Selector("setComputePipelineState:"), b.State);
        var setBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
        MetalNative.SendSetBuffer(encoder, setBuffer, arguments, 0, 0);
        MetalNative.SendSetBuffer(encoder, setBuffer, _synth.Push, 0, 1);
        MetalNative.SendUseResources(encoder, MetalNative.Selector("useResources:count:usage:"), handles, (nuint)count, UsageRead | UsageWrite);
        var shader = item.Shader;
        var x = Math.Max(shader.ThreadgroupSizeX, 1u);
        var y = Math.Max(shader.ThreadgroupSizeY, 1u);
        var z = Math.Max(shader.ThreadgroupSizeZ, 1u);
        var groups = (nuint)Math.Max(1, (int)(Int("groups", 256) * fraction));
        MetalNative.SendDispatch(encoder, MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:"),
            new MtlSize { Width = groups, Height = 1, Depth = 1 }, new MtlSize { Width = x, Height = y, Depth = z });
        MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
    }

    private static nint BeginRender(nint commands, int colors, bool depth, nuint width, nuint height, bool zreject)
    {
        var pass = MetalNative.Send(MetalNative.Class("MTLRenderPassDescriptor"), MetalNative.Selector("renderPassDescriptor"));
        var attachments = MetalNative.Send(pass, MetalNative.Selector("colorAttachments"));
        for (var i = 0; i < colors; i++)
        {
            var attachment = MetalNative.SendAtIndex(attachments, SelObjectAt, (nuint)i);
            MetalNative.SendVoid(attachment, MetalNative.Selector("setTexture:"), width == 256 ? _synth.SmallTarget : _synth.Targets[i]);
            MetalNative.SendVoidNUInt(attachment, MetalNative.Selector("setLoadAction:"), _clearTargets ? 2u : 0u);
            MetalNative.SendVoidNUInt(attachment, MetalNative.Selector("setStoreAction:"), 1);
        }

        if (depth || zreject)
        {
            var attachment = MetalNative.Send(pass, MetalNative.Selector("depthAttachment"));
            MetalNative.SendVoid(attachment, MetalNative.Selector("setTexture:"), _synth.Depth);
            MetalNative.SendVoidNUInt(attachment, MetalNative.Selector("setLoadAction:"), 2);
            MetalNative.SendVoidNUInt(attachment, MetalNative.Selector("setStoreAction:"), 0);
            MetalNative.SendVoidDouble(attachment, MetalNative.Selector("setClearDepth:"), zreject ? 0.0 : 1.0);
        }

        if (colors == 0 && !depth && !zreject)
        {
            MetalNative.SendVoidNUInt(pass, MetalNative.Selector("setRenderTargetWidth:"), width);
            MetalNative.SendVoidNUInt(pass, MetalNative.Selector("setRenderTargetHeight:"), height);
            MetalNative.SendVoidNUInt(pass, MetalNative.Selector("setDefaultRasterSampleCount:"), 1);
        }

        return MetalNative.Send(commands, MetalNative.Selector("renderCommandEncoderWithDescriptor:"), pass);
    }

    private static void EncodePixel(Item item, Built b, nint arguments, nint commands, nint* handles, int count, double fraction)
    {
        var width = (nuint)Int("width", 1920);
        var height = (nuint)Int("height", 1080);
        var colors = ColorCount(item);
        var depth = item.Source.Contains("[[depth(", StringComparison.Ordinal);
        var zreject = b.Variant.Scenario == "zreject";
        _synth.EnsureTargets(Math.Max(colors, 1), width, height);
        var encoder = BeginRender(commands, colors, depth, width, height, zreject);
        MetalNative.SendVoid(encoder, MetalNative.Selector("setRenderPipelineState:"), b.State);
        if (zreject)
        {
            MetalNative.SendVoid(encoder, MetalNative.Selector("setDepthStencilState:"), DepthState());
        }

        var setFragment = MetalNative.Selector("setFragmentBuffer:offset:atIndex:");
        MetalNative.SendSetBuffer(encoder, setFragment, arguments, 0, 0);
        MetalNative.SendSetBuffer(encoder, setFragment, _synth.Push, 0, 1);
        MetalNative.SendUseResourcesStages(encoder, MetalNative.Selector("useResources:count:usage:stages:"), handles, (nuint)count, UsageRead | UsageWrite, StagesAll);
        if (fraction < 1)
        {
            var side = Math.Sqrt(fraction);
            MetalNative.SendVoidScissor(encoder, MetalNative.Selector("setScissorRect:"), new MtlScissorRect { X = 0, Y = 0, Width = Math.Max(1, (nuint)(width * side)), Height = Math.Max(1, (nuint)(height * side)) });
        }

        MetalNative.SendDrawPrimitivesInstanced(encoder, MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:instanceCount:"), 3, 0, 6, (nuint)Int("quads", 4));
        MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
    }

    private static nint _depthState;
    private static bool _clearTargets;

    private static bool ShareTargets => Arg("check", "0") == "1";

    [DllImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static extern void GetBytes(nint texture, nint selector, nint destination, nuint bytesPerRow, MtlRegion region, nuint level);

    // The colour targets a pixel run wrote, read back (they are shared storage under check=1).
    private static string TargetHash(int colors)
    {
        var hash = new System.IO.Hashing.XxHash64();
        var width = (nuint)Int("width", 1920);
        var height = (nuint)Int("height", 1080);
        var bytes = new byte[width * height * 4];
        for (var i = 0; i < colors && i < _synth.Targets.Length; i++)
        {
            fixed (byte* at = bytes)
            {
                GetBytes(_synth.Targets[i], MetalNative.Selector("getBytes:bytesPerRow:fromRegion:mipmapLevel:"), (nint)at, width * 4,
                    new MtlRegion { X = 0, Y = 0, Z = 0, Width = width, Height = height, Depth = 1 }, 0);
            }

            hash.Append(bytes);
        }

        return hash.GetCurrentHashAsUInt64().ToString("X16", CultureInfo.InvariantCulture);
    }

    private static nint DepthState()
    {
        if (_depthState == 0)
        {
            var descriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLDepthStencilDescriptor"), SelAlloc), SelInit);
            MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setDepthCompareFunction:"), 1);
            MetalNative.SendVoidBool(descriptor, MetalNative.Selector("setDepthWriteEnabled:"), false);
            _depthState = MetalNative.Send(_device, MetalNative.Selector("newDepthStencilStateWithDescriptor:"), descriptor);
        }

        return _depthState;
    }

    private static int ColorCount(Item item)
    {
        var count = 0;
        foreach (var (_, _, attrs) in ParseStruct(item.Source, "Gen5PsOut"))
        {
            if (Regex.Match(attrs, @"color\((\d+)\)") is { Success: true } color)
            {
                count = Math.Max(count, int.Parse(color.Groups[1].Value, CultureInfo.InvariantCulture) + 1);
            }
        }

        return count;
    }

    // A mesh draw whose grid is wider than 65535 threadgroups is dropped by Metal without an error (it ran in 14 us), so the
    // vertex count is halved until the grid fits; the count then differs by program, and the groups are in the row's note.
    private static uint MeshVertices(Gen5MslMeshInfo info, uint vertices)
    {
        while (vertices > 3 && Gen5MslMeshLaunch.SubgroupCount(info, Gen5MslMeshInputTopology.TriangleList, vertices) > 30000)
        {
            vertices = Math.Max(vertices / 2 / 3 * 3, 3u);
        }

        return vertices;
    }

    // A vertex or mesh program: a fixed vertex count of triangles behind the consumer pixel function.
    private static void EncodeGeometry(Item item, Built b, nint arguments, nint commands, nint* handles, int count, double fraction)
    {
        var vertices = Math.Max((uint)(Int("verts", 393216) * fraction) / 3 * 3, 3u);
        if (item.Kind == "ms")
        {
            vertices = MeshVertices(item.Shader.MeshInfo!, vertices);
        }

        var encoder = BeginRender(commands, 1, false, 256, 256, false);
        MetalNative.SendVoid(encoder, MetalNative.Selector("setRenderPipelineState:"), b.State);
        MetalNative.SendUseResourcesStages(encoder, MetalNative.Selector("useResources:count:usage:stages:"), handles, (nuint)count, UsageRead | UsageWrite, StagesAll);
        if (item.Kind == "vs")
        {
            var set = MetalNative.Selector("setVertexBuffer:offset:atIndex:");
            MetalNative.SendSetBuffer(encoder, set, arguments, 0, 0);
            MetalNative.SendSetBuffer(encoder, set, _synth.Push, 0, 1);
            for (var slot = VertexSlotBase; slot < 31; slot++)
            {
                MetalNative.SendSetBuffer(encoder, set, _synth.VertexStream, 0, (nuint)slot);
            }

            MetalNative.SendDrawPrimitivesInstanced(encoder, MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:instanceCount:"), 3, 0, vertices, 1);
        }
        else
        {
            var info = item.Shader.MeshInfo!;
            var set = MetalNative.Selector("setMeshBuffer:offset:atIndex:");
            MetalNative.SendSetBuffer(encoder, set, arguments, 0, 0);
            MetalNative.SendSetBuffer(encoder, set, _synth.Push, 0, 1);
            var words = new Span<uint>((void*)MetalNative.Send(_synth.DrawInfo, MetalNative.Selector("contents")), (int)Gen5MslMeshInfo.DrawInfoDwordCount);
            words.Clear();
            Gen5MslMeshLaunch.FillDrawInfo(info, Gen5MslMeshInputTopology.TriangleList, words, vertices, 1, false, 1, 0, 0, 0);
            MetalNative.SendSetBuffer(encoder, set, _synth.DrawInfo, 0, info.DrawInfoBufferIndex);
            MetalNative.SendSetBuffer(encoder, set, _synth.Indices, 0, info.IndexBufferIndex);
            MetalNative.SendSetBuffer(encoder, set, _synth.Meshlets, 0, Gen5MslMeshInfo.MeshletBufferIndex);
            if (info.DeviceLds)
            {
                MetalNative.SendSetBuffer(encoder, set, _synth.LdsPool, 0, Gen5MslMeshInfo.LdsPoolBufferIndex);
            }

            var groups = Gen5MslMeshLaunch.SubgroupCount(info, Gen5MslMeshInputTopology.TriangleList, vertices);
            MetalNative.SendDrawMesh(encoder, MetalNative.Selector("drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:"),
                new MtlSize { Width = Math.Max(groups, 1), Height = 1, Depth = 1 },
                new MtlSize { Width = 1, Height = 1, Depth = 1 },
                new MtlSize { Width = Math.Max(info.ThreadsPerThreadgroup, 1), Height = 1, Depth = 1 });
        }

        MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
    }
}
