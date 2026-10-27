// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        // A packed element's components share words. Load each word once instead of
        // performing overlapping unaligned reads at every component's byte offset.
        // The caller emits the original decoder in the trailing else, preserving short
        // final elements whose selected bytes fit but whose rounded word span does not.
        private bool BeginLiteralPackedFormatLoad(FormatMemory memory, string address, uint word3,
            uint vectorData, uint componentCount)
        {
            if (!Gfx10UnifiedFormat.TryDecode((word3 >> 12) & 0x7Fu, out var dataFormat, out var numberFormat) ||
                Gfx10UnifiedFormat.HasWholeDwordComponents(dataFormat)) return false;
            var present = Gfx10UnifiedFormat.ComponentCount(dataFormat);
            if (present == 0) return false;
            var layouts = new (uint Word, uint Bit, uint Bits)[componentCount];
            uint run = 0;
            for (uint destination = 0; destination < componentCount; destination++)
            {
                var selector = (word3 >> (int)(destination * 3)) & 7u;
                if (selector < 4 || selector - 4 >= present) continue;
                if (!Gfx10UnifiedFormat.TryGetComponentLayout(dataFormat, selector - 4,
                        out var offset, out var bit, out var bits)) return false;
                bit += (offset % 4) * 8;
                if (bits == 0 || bit + bits > 32) return false;
                layouts[destination] = (offset / 4, bit, bits);
                run = Math.Max(run, offset / 4 + 1);
            }
            if (run is 0 or > 4) return false;
            var inBounds = memory.ElementInBounds(address, $"{run * 4}u");
            Line($"if ({inBounds})");
            Line("{");
            _indent++;
            var words = Temp("uint4", memory.LoadWords!(run, address));
            var one = $"sharpemu_format_one({numberFormat}u)";
            for (uint destination = 0; destination < componentCount; destination++)
            {
                var selector = (word3 >> (int)(destination * 3)) & 7u;
                string value;
                if (selector < 4) value = selector == 1 ? one : "0u";
                else if (selector - 4 >= present) value = selector == 7 ? one : "0u";
                else
                {
                    var (word, bit, bits) = layouts[destination];
                    var raw = Temp("uint", $"extract_bits({words}[{word}], {bit}u, {bits}u)");
                    value = Temp("uint", $"sharpemu_format_convert({raw}, {bits}u, {numberFormat}u, {dataFormat}u)");
                }
                StoreVector(vectorData + destination, value);
            }
            _indent--;
            Line("}");
            Line("else");
            Line("{");
            _indent++;
            return true;
        }
    }
}
