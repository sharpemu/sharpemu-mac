// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.PerfAudit;

Console.WriteLine($"SharpEmu per-draw microbench  arch={RuntimeInformation.ProcessArchitecture} os={RuntimeInformation.OSDescription} " +
    $"runtime={RuntimeInformation.FrameworkDescription} gcServer={System.Runtime.GCSettings.IsServerGC} tieredPGO={Environment.GetEnvironmentVariable("DOTNET_TieredPGO") ?? "default"}");
Console.WriteLine($"cores={Environment.ProcessorCount} synthetic sizing only: 6331 draws + 2700 dispatches; scaled costs are not measured frame times");
var filter = args.Length > 0 ? args[0] : "all";
bool On(string key) => filter == "all" || filter.Split(',').Contains(key);
foreach (var a in args) if (a.StartsWith("--warmup=")) Harness.WarmupMs = double.Parse(a[9..]);
foreach (var a in args) if (a.StartsWith("--scenarios=")) E2EBenches.OnlyScenarios = a[12..].Split(',');
foreach (var a in args) if (a.StartsWith("--soak=")) E2EBenches.SoakSeconds = double.Parse(a[7..]);
if (args.Contains("--allocs")) E2EBenches.TrackAllocations = true;
if (args.Contains("--quick")) { Harness.Runs = 3; Harness.TargetRunMs = 60; Harness.WarmupMs = 250; }

if (filter == "smoke") RigBenches.Smoke();
if (On("pm4")) Pm4Benches.Run();
if (On("plan")) PlanningBenches.Run();
if (On("e2e")) E2EBenches.Run();
if (On("mat")) MaterializationBenches.Run();
if (On("queue")) QueueBenches.Run();
if (On("mem")) MemoryBenches.Run();
if (On("lookup")) LookupBenches.Run();
if (On("metal")) MetalRawBenches.Run();
if (On("misc")) MiscBenches.Run();
if (On("w8")) W8Benches.Run();
if (filter.Split(',').Contains("xlate")) TranslatorBenches.Run();
if (filter.Split(',').Contains("gword")) MetalRawBenches.GpuWordWait();
if (filter.Split(',').Contains("wranges")) WriteRangeBenches.Run();
if (filter.Split(',').Contains("labels")) LabelFilterBenches.Run();
if (filter.Split(',').Contains("shmap")) ShaderRegisterMapBenches.Run();

Console.WriteLine();
Console.WriteLine(Harness.Table());
