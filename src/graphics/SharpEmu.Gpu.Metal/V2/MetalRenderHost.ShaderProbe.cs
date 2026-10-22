// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Text;
using System.Text.RegularExpressions;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    // WIP investigation instrumentation, not a rendering fix. Set both
    // SHARPEMU_SHADER_PROBE_DIR and SHARPEMU_SHADER_MSL_DUMP_DIR: instruction
    // comments from the latter give the trace its guest-PC offsets. Without them
    // assignments share PC zero and only the final register values are useful.
    // Each matching draw stalls for readback and writes 4 MiB; use only for diagnosis.
    private static readonly string? ProbeDirectory = Environment.GetEnvironmentVariable("SHARPEMU_SHADER_PROBE_DIR");
    private const ulong ProbeHash = 0xE3CF062DC905BE5D;
    private const int ProbeBytes = 4 * 1024 * 256 * 4;
    private nint _probeBuffer;
    private nint _probeContents;
    private int _probeDraw;

    private static string ProbeSource(string source, ulong hash)
    {
        if (ProbeDirectory is null || hash != ProbeHash || !source.Contains("fragment Gen5PsOut gen5_ps(")) return source;
        source = source.Replace("uint sharpemu_lane [[thread_index_in_simdgroup]])", "uint sharpemu_lane [[thread_index_in_simdgroup]], device uint* probe_data [[buffer(30)]])");
        var at = source.IndexOf('{', source.IndexOf("fragment Gen5PsOut gen5_ps(", StringComparison.Ordinal));
        source = source.Insert(at + 1, "\n const uint probe_x = uint(sharpemu_in.sharpemu_frag_coord.x);\n const bool probe_active = (probe_x == 256u || probe_x == 512u || probe_x == 768u || probe_x == 1024u) && uint(sharpemu_in.sharpemu_frag_coord.y) == 256u;\n const uint probe_base = ((probe_x / 256u) - 1u) * 262144u;\n");
        var result = new StringBuilder();
        var pc = 0;
        foreach (var line in source.Split('\n'))
        {
            var comment = Regex.Match(line, @"// 0x([0-9A-Fa-f]+) ");
            if (comment.Success) pc = Convert.ToInt32(comment.Groups[1].Value, 16);
            var rewritten = Regex.Replace(line, @"\b([sv])(\d+) = ([^;]+);", match =>
            {
                var reg = int.Parse(match.Groups[2].Value);
                var slot = (match.Groups[1].Value == "s" ? 64 : 0) + reg;
                return match.Value + $" if (probe_active) {{ probe_data[probe_base + {pc / 4 * 256 + slot}u] = {match.Groups[1].Value}{reg}; probe_data[probe_base + {pc / 4 * 256 + 255}u] = {pc + 1}u; }}";
            });
            if (rewritten.Contains("return sharpemu_out;")) rewritten = rewritten.Replace("return sharpemu_out;", "if (probe_active) { probe_data[probe_base] = as_type<uint>(sharpemu_out.mrt0.x); probe_data[probe_base+1u] = as_type<uint>(sharpemu_out.mrt0.y); probe_data[probe_base+2u] = as_type<uint>(sharpemu_out.mrt0.z); probe_data[probe_base+3u] = as_type<uint>(sharpemu_out.mrt0.w); } return sharpemu_out;");
            result.AppendLine(rewritten);
        }
        Directory.CreateDirectory(ProbeDirectory);
        File.WriteAllText(Path.Combine(ProbeDirectory, "instrumented.metal"), result.ToString());
        return result.ToString();
    }

    private void BindProbe()
    {
        if (ProbeDirectory is null || _stageBindings[StageFragment].Program?.Hash != ProbeHash) return;
        if (_probeBuffer == 0)
        {
            _probeBuffer = _guestMemory.CreateReportBuffer(ProbeBytes, "Shader probe", out _probeContents, out _);
            ClearProbe();
        }
        MetalNative.SendSetBuffer(_renderEncoder, SelSetFragmentBuffer, _probeBuffer, 0, 30);
    }

    private unsafe void ClearProbe() => new Span<byte>((void*)_probeContents, ProbeBytes).Clear();

    private unsafe void DrainProbe(string kind)
    {
        if (ProbeDirectory is null || _probeBuffer == 0 || !kind.StartsWith("draw", StringComparison.Ordinal) || _stageBindings[StageFragment].Program?.Hash != ProbeHash) return;
        FlushAndWaitForGpu();
        Directory.CreateDirectory(ProbeDirectory);
        var elapsed = (System.Diagnostics.Stopwatch.GetTimestamp() - _dumpStartTimestamp) / (double)System.Diagnostics.Stopwatch.Frequency;
        File.WriteAllBytes(Path.Combine(ProbeDirectory, $"trace-{_probeDraw++:D5}-{elapsed:F1}.bin"), new ReadOnlySpan<byte>((void*)_probeContents, ProbeBytes).ToArray());
        ClearProbe();
    }
}
