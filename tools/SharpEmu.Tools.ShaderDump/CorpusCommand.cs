// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Text.RegularExpressions;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler.Metal;

// Re-translates every record a game run wrote under SHARPEMU_SHADER_CORPUS_DIR through the
// Metal path and groups what the translator rejects, so the gaps of a whole run show up at once.
internal static partial class CorpusCommand
{
    private sealed record Failure(string Message, string Example);

    public static int Run(string[] args)
    {
        var metal = args.Contains("--metal", StringComparer.Ordinal);
        var directories = args.Where(argument => argument != "--metal").ToArray();
        if (directories.Length != 1 || !Directory.Exists(directories[0]))
        {
            Console.Error.WriteLine("Usage: SharpEmu.Tools.ShaderDump corpus <dir> [--metal]");
            return 2;
        }

        var files = Directory.EnumerateFiles(directories[0], "*" + ShaderCorpus.FileExtension).Order(StringComparer.Ordinal).ToArray();
        var failures = new List<Failure>();
        var succeeded = 0;
        foreach (var file in files)
        {
            var name = Path.GetFileNameWithoutExtension(file);
            if (!TryTranslate(file, metal, out var error))
            {
                failures.Add(new Failure(Normalize(error), name));
            }
            else
            {
                succeeded++;
            }
        }

        foreach (var group in failures.GroupBy(failure => failure.Message).OrderByDescending(group => group.Count()).ThenBy(group => group.Key, StringComparer.Ordinal))
        {
            Console.WriteLine($"{group.Count(),5}  {group.Key}");
            Console.WriteLine($"       e.g. {group.First().Example}");
        }

        Console.WriteLine($"corpus: {files.Length} records, {succeeded} ok, {failures.Count} failed{(metal ? " (with xcrun metal)" : string.Empty)}");
        return failures.Count == 0 ? 0 : 1;
    }

    private static bool TryTranslate(string file, bool metal, out string error)
    {
        error = string.Empty;
        try
        {
            var record = ShaderCorpus.Load(file);
            if (!ShaderCorpus.TryBuildRequest(record, out var request, out error))
            {
                return false;
            }

            if (request is null)
            {
                return true;
            }

            if (!Gen5MslTranslator.TryCompileProgram(request, out var shader, out error))
            {
                return false;
            }

            // SHARPEMU_CORPUS_MSL_OUT=<dir>: each record's translated MSL, for reading a program's code.
            if (Environment.GetEnvironmentVariable("SHARPEMU_CORPUS_MSL_OUT") is { Length: > 0 } mslDirectory)
            {
                Directory.CreateDirectory(mslDirectory);
                File.WriteAllText(Path.Combine(mslDirectory, Path.GetFileNameWithoutExtension(file) + ".metal"), shader.Source);
            }

            // SHARPEMU_CORPUS_LIST_OUT=<dir>: each record's instructions by pc, for aiming SHARPEMU_MSL_PC_PROBE.
            if (Environment.GetEnvironmentVariable("SHARPEMU_CORPUS_LIST_OUT") is { Length: > 0 } listDirectory)
            {
                Directory.CreateDirectory(listDirectory);
                File.WriteAllLines(Path.Combine(listDirectory, Path.GetFileNameWithoutExtension(file) + ".txt"),
                    request.Program.Instructions.Select(instruction =>
                        $"{instruction.Pc:X5} [{string.Join(" ", instruction.Words.Select(word => word.ToString("X8")))}] {instruction.Opcode} {string.Join(",", instruction.Destinations)} <- " +
                        $"{string.Join(",", instruction.Sources)}{(instruction.Control is { } control ? " " + control : string.Empty)}"));
            }

            // SHARPEMU_CORPUS_PLAN_OUT=<dir>: each record's image plan - class, first use and descriptor source - for
            // telling a table-selected image the planner bound directly from one it selects at run time.
            if (Environment.GetEnvironmentVariable("SHARPEMU_CORPUS_PLAN_OUT") is { Length: > 0 } planDirectory)
            {
                Directory.CreateDirectory(planDirectory);
                var info = request.Resources.Info;
                File.WriteAllLines(Path.Combine(planDirectory, Path.GetFileNameWithoutExtension(file) + ".txt"),
                    info.Images.Select((image, index) =>
                        $"image {index} pc=0x{image.FirstUsePc:X} class={image.ResourceClass} numeric={image.NumericClass} " +
                        $"dim={image.Dimension} bindless={image.BindlessSlot} root={image.IndirectRoot} source={image.Source} " +
                        $"indirect=[{string.Join(",", image.IndirectResources)}] search={image.IndirectSearchIterations}")
                    .Concat(info.Buffers.Select((buffer, index) =>
                        $"buffer {index} pc=0x{buffer.FirstUsePc:X} source={buffer.Source} read={buffer.Read} written={buffer.Written} " +
                        $"formatted={buffer.Formatted} extent=0x{buffer.MaxByteExtent:X}")));
            }

            return !metal || TryCompileMsl(shader, out error);
        }
        catch (Exception exception)
        {
            error = $"{exception.GetType().Name}: {exception.Message}";
            return false;
        }
    }

    private static bool TryCompileMsl(Gen5MslShader shader, out string error)
    {
        error = string.Empty;
        var source = Path.Combine(Path.GetTempPath(), $"sharpemu-corpus-{Environment.ProcessId}.metal");
        File.WriteAllText(source, shader.Source);
        try
        {
            // The options the render host compiles with: fast math off, MSL 3.2, 4.0 for per-vertex pixel inputs.
            var standard = shader.UsesPerVertexInputs ? "metal4.0" : "metal3.2";
            var start = new ProcessStartInfo("xcrun")
            {
                RedirectStandardError = true,
                RedirectStandardOutput = true,
            };
            foreach (var argument in new[] { "-sdk", "macosx", "metal", $"-std={standard}", "-fno-fast-math", "-c", source, "-o", "/dev/null" })
            {
                start.ArgumentList.Add(argument);
            }

            using var process = Process.Start(start) ?? throw new InvalidOperationException("xcrun did not start");
            var output = process.StandardError.ReadToEndAsync();
            process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            if (process.ExitCode == 0)
            {
                return true;
            }

            var text = output.Result.Split('\n');
            var first = text.FirstOrDefault(line => line.Contains("error:", StringComparison.Ordinal)) ?? text.FirstOrDefault() ?? "unknown error";
            error = "MSL: " + MslLocation().Replace(first, string.Empty);
            return false;
        }
        finally
        {
            File.Delete(source);
        }
    }

    // Keeps the failure class and drops what varies per program: addresses, pcs, register numbers, counts.
    // The opcode stays, since it is the gap.
    private static string Normalize(string message)
    {
        var line = message.Split('\n', 2)[0].Trim();
        line = Hex().Replace(line, "0x#");
        line = Register().Replace(line, "$1#");
        line = Range().Replace(line, "[#:#]");
        line = Number().Replace(line, "#");
        return line;
    }

    [GeneratedRegex(@"^.*?\.metal:\d+:\d+: ")]
    private static partial Regex MslLocation();

    [GeneratedRegex(@"0x[0-9A-Fa-f]+")]
    private static partial Regex Hex();

    [GeneratedRegex(@"\b(v|s|ttmp|attr)\d+\b")]
    private static partial Regex Register();

    [GeneratedRegex(@"\[\d+:\d+\]")]
    private static partial Regex Range();

    [GeneratedRegex(@"\b\d+\b")]
    private static partial Regex Number();
}
