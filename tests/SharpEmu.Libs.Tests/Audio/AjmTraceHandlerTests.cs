// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Audio;
using Xunit;

namespace SharpEmu.Libs.Tests.Audio;

public sealed class AjmTraceHandlerTests
{
    [Fact]
    public void DisabledTraceDoesNotFormatOrAllocate()
    {
        if (AjmExports.TraceEnabled)
        {
            return; // SHARPEMU_LOG_AJM=1 in this process: the disabled path is not reachable.
        }

        var evaluated = 0;
        Func<string> expensive = () =>
        {
            evaluated++;
            return new string('x', 64);
        };

        static AjmTraceInterpolatedStringHandler Build(ulong a, uint b, string c, Func<string> expensive)
        {
            AjmTraceInterpolatedStringHandler handler = $"batch a=0x{a:X16} b=0x{b:X8} c={c} d={expensive()}";
            return handler;
        }

        var handler = Build(1, 2, "abc", expensive);
        Assert.False(handler.Enabled);
        Assert.Equal(0, evaluated);
        Assert.Equal(string.Empty, handler.ToStringAndClear());

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 1000; i++)
        {
            var h = Build((ulong)i, (uint)i, "abc", expensive);
            _ = h.Enabled;
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Equal(0, evaluated);
    }
}
