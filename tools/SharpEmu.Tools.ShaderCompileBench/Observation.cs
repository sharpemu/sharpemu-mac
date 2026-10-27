// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Text.Json;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Ir;
using SharpEmu.ShaderCompiler.Metal;

internal static class Observation
{
    // Reports analysis opportunities; emit=<directory> optionally writes translated source without executing it.
    public static int Analyze(string directory, Func<string, string, string> arg)
    {
        var selected = arg("hash", "").Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(value => ulong.Parse(value.Replace("0x", "", StringComparison.OrdinalIgnoreCase), NumberStyles.HexNumber))
            .ToHashSet();
        using var list = GraphicsPrewarmList.Open(directory) ?? throw new InvalidOperationException("no graphics list");
        var seen = new HashSet<ulong>();
        var failures = 0;
        foreach (var item in list.Loaded())
        {
            var record = GraphicsPrewarmList.Decode(item);
            if ((selected.Count != 0 && !selected.Contains(record.Hash)) || !seen.Add(record.Hash)) continue;
            if (!ShaderCorpus.TryBuildRequest(record, out var request, out var error) || request is null)
            {
                Console.WriteLine(JsonSerializer.Serialize(new { hash = record.Hash.ToString("X16"), error }));
                failures++;
                continue;
            }
            if (arg("emit", "") is { Length: > 0 } emit)
            {
                if (!Gen5MslTranslator.TryCompileProgram(request, out var translated, out var translationError))
                {
                    Console.WriteLine(JsonSerializer.Serialize(new { hash = record.Hash.ToString("X16"), translationError }));
                    failures++;
                    continue;
                }
                Directory.CreateDirectory(emit);
                File.WriteAllText(Path.Combine(emit, $"{record.Hash:X16}.metal"), translated.Source);
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    hash = record.Hash.ToString("X16"), translated.MeshInfo,
                    hotLds = translated.Source.Contains("sharpemu_hot_lds", StringComparison.Ordinal),
                    sourceBytes = System.Text.Encoding.UTF8.GetByteCount(translated.Source),
                }));
            }
            var full = Gen5ExecFullAnalysis.Analyze(request.Program, request.WaveSize == 32);
            var plan = Gen5VectorObservationAnalysis.Analyze(request.Program, request.WaveSize == 32);
            var instructions = request.Program.Instructions;
            var candidates = instructions.Where(i => !full.Contains(i.Pc))
                .Sum(i => plan?.UnobservedWrites(i.Pc).Count ?? 0);
            var pureCandidates = instructions.Where(i => !full.Contains(i.Pc) &&
                    i.Encoding is (Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or Gen5ShaderEncoding.Vop3 or Gen5ShaderEncoding.Vop3p) &&
                    i.Control is not (Gen5DppControl or Gen5Dpp8Control))
                .Sum(i => plan?.UnobservedWrites(i.Pc).Count ?? 0);
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                hash = request.Hash.ToString("X16"), stage = request.Stage.ToString(), wave = request.WaveSize,
                mesh = Gen5MslTranslator.RequiresMeshEmission(request), instructions = instructions.Count,
                primitiveShader = request.PrimitiveShader,
                fullExecInstructions = full.Count, planAvailable = plan is not null, candidates, pureCandidates,
                relative = instructions.Where(i => i.Opcode.Contains("rel", StringComparison.Ordinal))
                    .Select(i => i.Opcode).Distinct().ToArray(),
                dppInstructions = instructions.Count(i => i.Control is Gen5DppControl or Gen5Dpp8Control),
            }));
        }
        foreach (var missing in selected.Except(seen))
        {
            Console.WriteLine(JsonSerializer.Serialize(new { hash = missing.ToString("X16"), error = "requested shader is absent from the corpus" }));
            failures++;
        }
        return failures == 0 ? 0 : 1;
    }
}
