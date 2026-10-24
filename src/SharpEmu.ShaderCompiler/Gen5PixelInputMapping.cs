// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler;

public static class Gen5PixelInputMapping
{
    public static uint[] ResolveLocations(
        ReadOnlySpan<uint> controls,
        ReadOnlySpan<uint> activeInputs)
    {
        var locations = new uint[activeInputs.Length];
        ResolveLocations(controls, activeInputs, locations);
        return locations;
    }

    // Fills `locations` (one slot per active input) without allocating.
    public static void ResolveLocations(
        ReadOnlySpan<uint> controls,
        ReadOnlySpan<uint> activeInputs,
        Span<uint> locations)
    {
        Span<bool> usedLocations = stackalloc bool[32];

        for (var index = 0; index < activeInputs.Length; index++)
        {
            var input = activeInputs[index];
            var location = input < (uint)controls.Length
                ? controls[(int)input] & 0x1Fu
                : input;
            if (location < 32u && usedLocations[(int)location])
            {
                location = input;
                while (location < 32u && usedLocations[(int)location])
                {
                    location++;
                }

                if (location >= 32u)
                {
                    throw new InvalidOperationException(
                        "pixel input locations exceed the hardware interface limit");
                }
            }

            locations[index] = location;
            if (location < 32u)
            {
                usedLocations[(int)location] = true;
            }
        }
    }

    /// <summary>
    /// The parameter exports (SPI_PS_INPUT_CNTL.OFFSET, one bit each) whose three vertex
    /// values a pixel program reads: an input slot it reads with V_INTERP_MOV P10/P20, an
    /// f16 interpolation or modified I/J needs every vertex of the primitive, unlike one it
    /// reads only through P0 (the provoking vertex, which a flat input gives).
    /// </summary>
    public static uint PerVertexParameterMask(Gen5ShaderProgram program, ReadOnlySpan<uint> controls)
    {
        Span<bool> needsVertices = stackalloc bool[32];
        Span<bool> otherThanP0 = stackalloc bool[32];
        // Indexed: foreach over the interface boxed the list's enumerator on every draw.
        var instructions = program.Instructions;
        for (var index = 0; index < instructions.Count; index++)
        {
            var instruction = instructions[index];
            if (instruction.Control is not Gen5InterpolationControl control || control.Attribute >= 32u)
            {
                continue;
            }

            var attribute = (int)control.Attribute;
            if (instruction.Opcode is "VInterpMovF32" or "VInterpP1llF16" or "VInterpP1lvF16" or "VInterpP2F16" ||
                control.Modifiers is { } modifiers && ((modifiers.AbsoluteMask | modifiers.NegateMask) & 0x6u) != 0)
            {
                needsVertices[attribute] = true;
            }

            if (instruction.Opcode != "VInterpMovF32" || control.Parameter != ProvokingVertexParameter)
            {
                otherThanP0[attribute] = true;
            }
        }

        var mask = 0u;
        for (var attribute = 0; attribute < 32; attribute++)
        {
            if (needsVertices[attribute] && otherThanP0[attribute])
            {
                var offset = attribute < controls.Length ? controls[attribute] & 0x1Fu : (uint)attribute;
                mask |= 1u << (int)offset;
            }
        }

        return mask;
    }

    // V_INTERP_MOV's P0 selector.
    private const uint ProvokingVertexParameter = 2;
}
