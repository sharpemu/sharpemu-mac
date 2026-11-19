// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using System.Text.RegularExpressions;

namespace SharpEmu.ShaderCompiler.Metal;

// The rectangle-list form of a translated vertex program.
//
// A guest rectangle is three vertices; the rasterizer derives the fourth corner from the
// three transformed ones (v1 + v2 - v0 about the right-angle corner) and fills the
// rectangle, with every output extended over it. Metal has no geometry stage and a vertex
// function sees one vertex, so the corner cannot be made there. A post-tessellation vertex
// function can: drawn as a three-control-point quad patch with every tessellation factor
// at 1, the tessellator emits the patch's four corners as two triangles, and each corner
// invocation gets all three control points - their [[stage_in]] attributes fetched by the
// vertex descriptor, so a program with vertex buffers works unchanged. Each invocation runs
// the guest program for the three guest vertices and returns its own corner.
//
// The program body is the translated vertex entry turned into a plain function, so this
// form needs nothing from the translator but its output, and a library is only built for
// it when a rectangle-list pipeline uses the program.
public static partial class Gen5MslRectangleList
{
    /// <summary>The post-tessellation entry point of the source <see cref="BuildSource"/> returns.</summary>
    public const string EntryPoint = "gen5_vs_rect";

    /// <summary>Vertex-stage slot of the per-draw block (see <see cref="DrawInfoByteSize"/>).</summary>
    public const int DrawInfoBufferIndex = 30;

    /// <summary>Vertex-stage slot of the guest index buffer, read as dwords for the vertex
    /// index an indexed draw hands the program.</summary>
    public const int IndexBufferIndex = 29;

    /// <summary>Byte size of the per-draw block: vertex_base, vertex_offset, index_type
    /// (0 none, 1 16-bit, 2 32-bit), first_index and base_vertex (the draw's base vertex as
    /// a vertex function's [[base_vertex]] gives it: the first vertex of an automatic draw,
    /// the vertex offset of an indexed one), one dword each.</summary>
    public const uint DrawInfoByteSize = 20;

    /// <summary>Guest vertices per rectangle.</summary>
    public const uint VerticesPerRectangle = 3;

    private const string GuestFunction = "gen5_vs_guest";
    private const string ControlPoints = "sharpemu_rect_points";

    [GeneratedRegex(@"\s*\[\[[^\]]*\]\]")]
    private static partial Regex AttributePattern();

    [GeneratedRegex(@"(\w+)\s*$")]
    private static partial Regex TrailingNamePattern();

    [GeneratedRegex(@"^\s*(float4|float|uint)\s+(\w+)\s*\[\[([^\]]*)\]\]\s*(?:\[(\d+)\])?\s*;\s*$")]
    private static partial Regex OutputFieldPattern();

    /// <summary>The rectangle-list form of a translated vertex program's source.</summary>
    public static string BuildSource(Gen5MslShader shader)
    {
        if (shader.Stage != Gen5MslStage.Vertex)
        {
            throw new ArgumentException($"A rectangle list draws through a vertex program, not a {shader.Stage} one.", nameof(shader));
        }

        var lines = shader.Source.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
        var header = $"vertex Gen5VsOut {shader.EntryPoint}(";
        var start = Array.IndexOf(lines, header);
        if (start < 0)
        {
            throw new InvalidOperationException($"The vertex program has no '{header}' entry to wrap.");
        }

        var open = start + 1;
        while (open < lines.Length && lines[open] != "{")
        {
            open++;
        }

        if (open == lines.Length)
        {
            throw new InvalidOperationException("The vertex entry's parameter list never closes.");
        }

        var fields = OutputFields(lines, start);
        var helperParameters = new List<string>();
        var entryParameters = new List<string>();
        var arguments = new List<string>();
        var readsStageIn = false;
        for (var index = start + 1; index < open; index++)
        {
            var parameter = lines[index].Trim().TrimEnd(',', ')').Trim();
            if (parameter.Length == 0)
            {
                continue;
            }

            var plain = AttributePattern().Replace(parameter, string.Empty).Trim();
            var name = TrailingNamePattern().Match(plain).Groups[1].Value;
            helperParameters.Add(plain);
            if (parameter.Contains("[[stage_in]]", StringComparison.Ordinal))
            {
                readsStageIn = true;
                arguments.Add($"{ControlPoints}[corner]");
                continue;
            }

            if (parameter.Contains("[[vertex_id]]", StringComparison.Ordinal))
            {
                arguments.Add("sharpemu_rect_vertex_id(sharpemu_rect_draw, sharpemu_rect_indices, sharpemu_rect_patch, corner)");
                continue;
            }

            // A post-tessellation function has no [[base_vertex]]; the draw block holds the
            // base the vertex ids above count from.
            if (parameter.Contains("[[base_vertex]]", StringComparison.Ordinal))
            {
                arguments.Add("sharpemu_rect_base_vertex(sharpemu_rect_draw)");
                continue;
            }

            entryParameters.Add(parameter);
            arguments.Add(name);
        }

        var source = new StringBuilder(shader.Source.Length + 4096);
        for (var index = 0; index < lines.Length; index++)
        {
            if (index == start)
            {
                source.Append("static inline Gen5VsOut ").Append(GuestFunction).Append("(\n");
                source.Append(string.Join(",\n", helperParameters.Select(parameter => "    " + parameter))).Append(")\n");
                index = open - 1;
                continue;
            }

            source.Append(lines[index]).Append('\n');
        }

        EmitPrelude(source);
        source.Append("[[patch(quad, 3)]]\n");
        source.Append("vertex Gen5VsOut ").Append(EntryPoint).Append("(\n");
        if (readsStageIn)
        {
            source.Append("    patch_control_point<Gen5VsIn> ").Append(ControlPoints).Append(" [[stage_in]],\n");
        }

        foreach (var parameter in entryParameters)
        {
            source.Append("    ").Append(parameter).Append(",\n");
        }

        source.Append($"    constant Gen5RectDraw& sharpemu_rect_draw [[buffer({DrawInfoBufferIndex})]],\n");
        source.Append($"    device const uint* sharpemu_rect_indices [[buffer({IndexBufferIndex})]],\n");
        source.Append("    uint sharpemu_rect_patch [[patch_id]],\n");
        source.Append("    float2 sharpemu_rect_uv [[position_in_patch]])\n");
        source.Append("{\n");
        // One call site, so the program is inlined once; every invocation runs all three
        // vertices, which keeps the wave operations inside it in uniform control flow.
        source.Append("    Gen5VsOut guest[3];\n");
        source.Append("    for (uint corner = 0u; corner < 3u; corner++)\n");
        source.Append("    {\n");
        source.Append("        guest[corner] = ").Append(GuestFunction).Append('(').Append(string.Join(", ", arguments)).Append(");\n");
        source.Append("    }\n\n");
        source.Append("    const uint first = sharpemu_rect_right_angle(guest[0].sharpemu_position, guest[1].sharpemu_position, guest[2].sharpemu_position);\n");
        source.Append("    const Gen5VsOut o0 = guest[first];\n");
        source.Append("    const Gen5VsOut o1 = guest[(first + 1u) % 3u];\n");
        source.Append("    const Gen5VsOut o2 = guest[(first + 2u) % 3u];\n");
        source.Append("    const bool far_u = sharpemu_rect_uv.x > 0.5f;\n");
        source.Append("    const bool far_v = sharpemu_rect_uv.y > 0.5f;\n");
        source.Append("    if (!far_v)\n");
        source.Append("    {\n");
        source.Append("        return far_u ? o1 : o0;\n");
        source.Append("    }\n\n");
        source.Append("    if (!far_u)\n");
        source.Append("    {\n");
        source.Append("        return o2;\n");
        source.Append("    }\n\n");
        // Per-primitive values (layer, viewport, point size) stay the first corner's.
        source.Append("    Gen5VsOut o3 = o0;\n");
        foreach (var field in fields)
        {
            if (field.Type == "uint" || field.Attribute == "point_size")
            {
                continue;
            }

            if (field.Count is { } count)
            {
                for (var element = 0; element < count; element++)
                {
                    source.Append($"    o3.{field.Name}[{element}] = sharpemu_rect_corner(o0.{field.Name}[{element}], o1.{field.Name}[{element}], o2.{field.Name}[{element}]);\n");
                }

                continue;
            }

            source.Append($"    o3.{field.Name} = sharpemu_rect_corner(o0.{field.Name}, o1.{field.Name}, o2.{field.Name});\n");
        }

        source.Append("    return o3;\n");
        source.Append("}\n");
        return source.ToString();
    }

    private sealed record OutputField(string Type, string Name, string Attribute, int? Count);

    private static List<OutputField> OutputFields(string[] lines, int end)
    {
        var start = Array.IndexOf(lines, "struct Gen5VsOut");
        if (start < 0 || start > end || lines[start + 1] != "{")
        {
            throw new InvalidOperationException("The vertex program has no Gen5VsOut struct ahead of its entry.");
        }

        var fields = new List<OutputField>();
        for (var index = start + 2; index < end && lines[index] != "};"; index++)
        {
            var match = OutputFieldPattern().Match(lines[index]);
            if (!match.Success)
            {
                throw new InvalidOperationException($"A Gen5VsOut field has no rectangle-corner rule: '{lines[index].Trim()}'.");
            }

            fields.Add(new OutputField(
                match.Groups[1].Value,
                match.Groups[2].Value,
                match.Groups[3].Value,
                match.Groups[4].Success ? int.Parse(match.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture) : null));
        }

        return fields;
    }

    private static void EmitPrelude(StringBuilder source)
    {
        source.Append("""

            struct Gen5RectDraw
            {
                uint vertex_base;
                int vertex_offset;
                uint index_type;
                uint first_index;
                uint base_vertex;
            };

            // The guest vertex index of one corner of a rectangle: consecutive vertices from
            // vertex_base for an automatic draw, the index buffer's entry plus the vertex offset
            // for an indexed one. The control points Metal fetched are the same vertices.
            static inline uint sharpemu_rect_vertex_id(constant Gen5RectDraw& draw, device const uint* indices, uint patch, uint corner)
            {
                const uint slot = patch * 3u + corner;
                if (draw.index_type == 0u)
                {
                    return draw.vertex_base + slot;
                }

                const uint offset = draw.first_index + slot;
                const uint index = draw.index_type == 1u
                    ? (indices[offset >> 1u] >> ((offset & 1u) * 16u)) & 0xFFFFu
                    : indices[offset];
                return index + (uint)draw.vertex_offset;
            }

            // The base vertex the draw's vertex ids count from. The block carries it apart from
            // vertex_offset: an indirect draw rebased on the GPU folds its offset into the
            // indices it hands over.
            static inline uint sharpemu_rect_base_vertex(constant Gen5RectDraw& draw)
            {
                return draw.base_vertex;
            }

            // The corner the rectangle's two edges leave at right angles: it shares x with one
            // neighbour and y with the other. The fourth corner is opposite it. A rectangle
            // whose edges are not axis-aligned takes the first vertex, which is v1 + v2 - v0.
            static inline bool sharpemu_rect_is_right_angle(float4 corner, float4 next, float4 previous)
            {
                return (corner.x == next.x && corner.y == previous.y) || (corner.y == next.y && corner.x == previous.x);
            }

            static inline uint sharpemu_rect_right_angle(float4 p0, float4 p1, float4 p2)
            {
                if (sharpemu_rect_is_right_angle(p0, p1, p2)) return 0u;
                if (sharpemu_rect_is_right_angle(p1, p2, p0)) return 1u;
                if (sharpemu_rect_is_right_angle(p2, p0, p1)) return 2u;
                return 0u;
            }

            // o1 + o2 - o0, one component at a time. A component equal at o0 and one neighbour
            // is copied from the other neighbour instead of computed, which is the same value
            // for a linear attribute and keeps a constant bit pattern (an integer carried in a
            // float, a denormal) from passing through float arithmetic.
            static inline float sharpemu_rect_corner(float o0, float o1, float o2)
            {
                if (as_type<uint>(o1) == as_type<uint>(o0)) return o2;
                if (as_type<uint>(o2) == as_type<uint>(o0)) return o1;
                return o1 + o2 - o0;
            }

            static inline float4 sharpemu_rect_corner(float4 o0, float4 o1, float4 o2)
            {
                return float4(
                    sharpemu_rect_corner(o0.x, o1.x, o2.x),
                    sharpemu_rect_corner(o0.y, o1.y, o2.y),
                    sharpemu_rect_corner(o0.z, o1.z, o2.z),
                    sharpemu_rect_corner(o0.w, o1.w, o2.w));
            }


            """);
    }
}
