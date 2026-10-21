// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System;
using System.IO;
using FFmpeg.AutoGen;

namespace SharpEmu.Libs.Media;
internal static class FfmpegRuntime
{
    private static readonly object _gate = new();
    private static bool _initialized;
    internal static void EnsureInitialized()
    {
        if (_initialized)
        {
            return;
        }

        lock (_gate)
        {
            if (_initialized)
            {
                return;
            }

            _initialized = true;
            ffmpeg.RootPath = Path.Combine(AppContext.BaseDirectory, "plugins");
            DynamicallyLoadedBindings.Initialize();
            try { ffmpeg.av_log_set_level(ffmpeg.AV_LOG_ERROR); }
            catch (NotSupportedException) { } // Optional native logging is unavailable without the FFmpeg plugin.
        }
    }
}
