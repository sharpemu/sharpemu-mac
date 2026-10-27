using System.Diagnostics;
using SharpEmu.Libs.Pad;
using Xunit;
using Xunit.Abstractions;
namespace SharpEmu.GpuSplit.Tests;
[Collection(SplitCollection.Name)]
public sealed class TmpInputMeasure(ITestOutputHelper output)
{
    [MacOSFact]
    public void Measure()
    {
        var lines = new List<string>();
        var previousMode = System.Environment.GetEnvironmentVariable("SHARPEMU_SPLIT_INPUT_POLL");
        try
        {
            foreach (var mode in new[] { "1", "0", "1", "0" })
            {
                System.Environment.SetEnvironmentVariable("SHARPEMU_SPLIT_INPUT_POLL", mode);
                using var split = new InProcessSplit(inputPoll: TimeSpan.FromMilliseconds(4));
                HostWindowInput.Connect();
                Thread.Sleep(1500);
                var proc = Process.GetCurrentProcess();
                proc.Refresh(); var c0 = proc.TotalProcessorTime; var w0 = Stopwatch.GetTimestamp();
                Thread.Sleep(8000);
                proc.Refresh(); var c1 = proc.TotalProcessorTime;
                var cpu = (c1 - c0).TotalMilliseconds / Stopwatch.GetElapsedTime(w0).TotalSeconds;
                var lat = new List<double>();
                var rnd = new Random(1);
                for (var i = 0; i < 60; i++)
                {
                    Thread.Sleep(10 + rnd.Next(27));
                    var n = split.GuestSide.Inputs.Count;
                    var t = Stopwatch.GetTimestamp();
                    HostWindowInput.SetKey(33, down: i % 2 == 0);
                    while (split.GuestSide.Inputs.Count == n && Stopwatch.GetElapsedTime(t).TotalMilliseconds < 500) { }
                    lat.Add(Stopwatch.GetElapsedTime(t).TotalMilliseconds);
                }
                HostWindowInput.Disconnect();
                lat.Sort();
                lines.Add($"poll_mode={mode} process_cpu_ms_per_s_idle={cpu:F2} latency_ms p50={lat[30]:F2} p90={lat[54]:F2} max={lat[^1]:F2}");
            }
        }
        finally
        {
            System.Environment.SetEnvironmentVariable("SHARPEMU_SPLIT_INPUT_POLL", previousMode);
            HostWindowInput.Disconnect();
            File.WriteAllLines("/tmp/r2-k4-input.txt", lines);
            foreach (var line in lines) output.WriteLine(line);
        }
    }
}
