// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.ComponentModel;
using System.Text;
using SharpEmu.Logging;
using Xunit;

namespace SharpEmu.Libs.Tests.Logging;

public sealed class MemoryTelemetryTests
{
    [Fact]
    public void FailedProcessInspectionDoesNotStopOtherSectionsOrTheNextSample()
    {
        using var output = new StringWriter();
        var calls = 0;
        (string, Func<string>)[] providers =
        [
            ("broken", () => throw new InvalidOperationException("provider unavailable")),
            ("healthy", () => $"sample={++calls}"),
        ];
        MemoryTelemetry.ReportSections(output,
            () => throw new Win32Exception(12, "process inspection unavailable"), providers);
        MemoryTelemetry.ReportSections(output, () => "threads=8", providers);

        Assert.Equal(2, calls);
        var lines = output.ToString().Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(6, lines.Length);
        Assert.Contains("dotnet error=Win32Exception", lines[0]);
        Assert.Contains("broken error=InvalidOperationException", lines[1]);
        Assert.Equal("[MEM] healthy sample=1", lines[2]);
        Assert.Equal("[MEM] dotnet threads=8", lines[3]);
        Assert.Equal("[MEM] healthy sample=2", lines[5]);
    }

    [Fact]
    public void AFailedDiagnosticSinkCannotTerminateReporting()
    {
        using var output = new RecoveringWriter();
        MemoryTelemetry.ReportSections(output, () => "threads=8",
            [("healthy", () => "continued")]);
        Assert.Equal(3, output.Attempts);
        Assert.Equal("[MEM] healthy continued" + Environment.NewLine, output.Saved.ToString());
    }

    private sealed class RecoveringWriter : TextWriter
    {
        public override Encoding Encoding => Encoding.UTF8;
        public int Attempts { get; private set; }
        public StringBuilder Saved { get; } = new();
        public override void WriteLine(string? value)
        {
            if (++Attempts <= 2) throw new IOException("diagnostic output unavailable");
            Saved.AppendLine(value);
        }
    }
}
