// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>
/// Dispatches a queue issues with no barrier between them, run side by side.
/// SHARPEMU_CONCURRENT_DISPATCH=0 runs them one after another.
/// </summary>
/// <remarks>
/// On the console, two dispatches of one queue overlap unless the stream orders them (a partial
/// flush, a cache action, a release or a wait). Ghost of Tsushima's async compute queue builds the
/// frame's draw lists with 84 single-threadgroup dispatches and no barrier between them; a serial
/// Metal encoder ran them one after another on one GPU core, about 40 ms of the frame. A dispatch
/// the stream reaches from the one before it through nothing but register writes and markers
/// carries <see cref="Rendering.RenderExecutor.DispatchMayOverlapBit"/>, and the backend lets it
/// run beside that one. Ghost of Tsushima at the gorge: 6.98 -> 9.46 median fps (got-cc1).
/// </remarks>
public static class DispatchOverlap
{
    private static int _enabled = -1;

    /// <summary>Read at the first dispatch.</summary>
    public static bool Enabled
    {
        get
        {
            if (_enabled < 0)
            {
                var titleId = SharpEmu.Libs.VideoOut.VideoOutBackendServices.GetApplicationTitleId();
                var enabled = Environment.GetEnvironmentVariable("SHARPEMU_CONCURRENT_DISPATCH") switch
                {
                    "1" => true,
                    "0" => false,
                    _ => true,
                };
                Console.Error.WriteLine($"[GPU][INFO] unordered dispatches run side by side: {(enabled ? "on" : "off")} (title {titleId}).");
                _enabled = enabled ? 1 : 0;
            }

            return _enabled == 1;
        }
    }

    /// <summary>
    /// Whether a dispatch may still overlap the one before it after this packet: only register
    /// writes, markers, calls into another command buffer and other dispatches leave it so.
    /// </summary>
    public static bool Keeps(uint opcode, uint header) => opcode switch
    {
        PacketOpcode.Nop => PacketHeader.CustomCode(header) is 0 or PacketCustomCode.PushMarker or PacketCustomCode.PopMarker or
            PacketCustomCode.ShaderRegisterTable or PacketCustomCode.ContextRegisterTable or PacketCustomCode.UserConfigRegisterTable,
        PacketOpcode.SetBase or PacketOpcode.SetShaderRegister or PacketOpcode.SetContextRegister or
            PacketOpcode.SetUserConfigRegister or PacketOpcode.SetUserConfigRegisterIndex or PacketOpcode.SetShaderRegisterIndirect or
            PacketOpcode.SetContextRegisterIndirect or PacketOpcode.SetUserConfigRegisterIndirect or PacketOpcode.IndirectBuffer or
            PacketOpcode.ContextControl or PacketOpcode.DispatchDirect or PacketOpcode.DispatchIndirect => true,
        _ => false,
    };
}
