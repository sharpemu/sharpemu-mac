// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.ShaderCompiler.Metal;

/// <summary>
/// The fixed presenter shaders, mirroring SpirvFixedShaders semantically. The
/// MSL lives in Templates/*.msl (authored as real Metal source); this class
/// only substitutes the per-call parameters. Entry point names are stable
/// (Metal forbids "main"); textures and samplers bind at index 0; attributes
/// use the same user(locn) convention as the translated stages.
/// </summary>
public static class MslFixedShaders
{
    /// <summary>
    /// Fullscreen triangle from the vertex index; every attribute location in
    /// 0..attributeCount-1 carries (x, y, 0, 1) so paired fragment stages can
    /// read a screen-space UV from any location.
    /// </summary>
    public static string CreateFullscreenVertex(uint attributeCount)
    {
        var fields = new StringBuilder();
        var stores = new StringBuilder();
        for (uint index = 0; index < attributeCount; index++)
        {
            if (index != 0)
            {
                fields.AppendLine();
                stores.AppendLine();
            }

            fields.Append($"    float4 attr{index} [[user(locn{index})]];");
            stores.Append($"    out.attr{index} = float4(x, y, 0.0f, 1.0f);");
        }

        return MslTemplates.Render(
            "fullscreen_vertex",
            ("attribute_fields", fields.ToString()),
            ("attribute_stores", stores.ToString()));
    }

    /// <summary>Samples texture 0 at the interpolated location-0 UV.</summary>
    public static string CreateCopyFragment() => MslTemplates.Render("copy_fragment");

    /// <summary>
    /// The presenter's blit stage: samples texture 0 with V flipped, because
    /// pairing the shared fullscreen triangle with Metal's y-up NDC puts UV
    /// (0,0) at the bottom of the screen while textures keep v=0 at the top.
    /// </summary>
    public static string CreatePresentFragment() => MslTemplates.Render("present_fragment");

    /// <summary>
    /// The presenter's blit stage for an extended-dynamic-range drawable, the
    /// counterpart of SpirvFixedShaders' scaled copy and PQ-to-scRGB stages: an SDR
    /// frame is decoded to linear, a PQ (BT.2100) frame to luminance relative to the
    /// reference white in BT.709 primaries. Parameters bind at buffer 0.
    /// </summary>
    public static string CreatePresentHdrFragment() => MslTemplates.Render("present_hdr_fragment");

    /// <summary>
    /// The performance overlay's blend stage: overlay_fs passes a premultiplied image through;
    /// overlay_hdr_fs decodes it to linear SDR white for an extended-range drawable (parameters
    /// as for the HDR present stage, at buffer 0).
    /// </summary>
    public static string CreateOverlayFragment() => MslTemplates.Render("overlay_fragment");

    /// <summary>
    /// Output-free fragment stage for fixed-function depth-only passes: the
    /// guest has no pixel shader, so no color may be written while depth
    /// testing still runs for the translated vertex shader.
    /// </summary>
    public static string CreateDepthOnlyFragment() => MslTemplates.Render("depth_only_fragment");
}
