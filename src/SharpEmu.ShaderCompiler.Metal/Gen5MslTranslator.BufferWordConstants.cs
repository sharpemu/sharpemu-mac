// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    /// <summary>
    /// The default <see cref="Gen5MslCodegenOptions.FormatFetchConstantSiteLimit"/>: a program with
    /// more BUFFER_LOAD_FORMAT_* keeps today's decode alone even where word 3 is a function
    /// constant. Vertex programs fetch a handful.
    /// </summary>
    internal const int FormatFetchConstantSiteLimit = 64;

    private sealed partial class CompilationContext
    {
        // The buffer-word function constants the program reads (PortableBufferWord.ConstantsEnabled):
        // constant index -> the shader-data dword it stands in for.
        private readonly SortedDictionary<uint, uint> _bufferWordConstants = new();
        private bool _usesFormatFetchConstant;

        private bool BufferWordConstantsEnabled => _options.BufferWordConstants ?? PortableBufferWord.ConstantsEnabled;

        // A portable buffer's run-time word: the function constant when the function was made
        // with a value for it, the shader data otherwise. Declares the constant.
        private string BufferWordConstant(int bindingIndex, int word, uint dword, out string name)
        {
            var index = PortableBufferWord.ConstantIndex(bindingIndex, word);
            name = PortableBufferWord.ConstantName(index, dword);
            _bufferWordConstants[index] = dword;
            return $"({name}_set ? {name} : {ShaderDataDword(dword)})";
        }

        private IReadOnlyList<Gen5MslShader.BufferWordConstant> BufferWordConstants() =>
            _bufferWordConstants.Select(static pair => new Gen5MslShader.BufferWordConstant(pair.Key, pair.Value)).ToArray();

        // Program-scope declarations, after the preludes: each constant and whether it was given.
        private void AppendBufferWordConstants(StringBuilder source)
        {
            foreach (var (index, dword) in _bufferWordConstants)
            {
                var name = PortableBufferWord.ConstantName(index, dword);
                source.AppendLine($"constant uint {name} [[function_constant({index})]];");
                source.AppendLine($"constant bool {name}_set = is_function_constant_defined({name});");
            }

            if (_usesFormatFetchConstant)
            {
                source.Append(MslTemplates.Render("format_fetch_constant_prelude"));
            }
        }

        // A formatted load whose word 3 is the function constant: the constant form
        // (format_fetch_constant_prelude.msl) where the function has the value, the run-time
        // decode, unchanged, where it does not. False where the site keeps the run-time decode
        // alone: a D16 or GLC load, a placed access, a program over the site limit.
        private bool TryEmitConstantFormatLoad(
            int bindingIndex,
            BufferAddress byteAddress,
            string descriptorWord3,
            string word3Constant,
            uint vectorData,
            uint componentCount,
            FormatD16 d16,
            bool glc)
        {
            if (d16.Enabled || glc || !byteAddress.Contiguous || !_options.FormatFetch ||
                _formatBufferLoads > _options.FormatFetchConstantSiteLimit)
            {
                return false;
            }

            _usesFormatFetch = true;
            _usesFormatFetchConstant = true;
            Line($"if ({word3Constant}_set)");
            Line("{");
            _indent++;
            var fetched = Temp(
                "uint4",
                $"sharpemu_format_fetch_constant(b{bindingIndex}, {BufferBytes(bindingIndex)}, {byteAddress.Base}, {word3Constant}, {componentCount}u)");
            for (uint component = 0; component < componentCount; component++)
            {
                StoreVector(vectorData + component, $"{fetched}[{component}]");
            }

            _indent--;
            Line("}");
            Line("else");
            Line("{");
            _indent++;
            EmitDecodedBufferFormatLoad(
                BindingFormatMemory(bindingIndex, glc),
                byteAddress,
                descriptorWord3,
                vectorData,
                componentCount,
                d16);
            _indent--;
            Line("}");
            return true;
        }
    }
}
