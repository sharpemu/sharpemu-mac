// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Numerics;
using System.Text;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// The bindless heap (BindlessHeap): an image or sampler the shader reads from a heap V# at a record only it knows is
// the element of that record in the heap's element table, one 8-byte handle per element, slot-major. The MSL form of
// the SPIR-V translator's set-1 arrays: the element a read takes is the one its draw's heap view (HeapView, two dwords
// of shader data) has for the record, or the null element; the shader marks every element it reads in the feedback
// bitsets so the host gives it a view, and an image element of a class with integer forms is read through the
// declaration of its numeric class, which the host keeps after the bitsets. A null image element is a zero handle: the
// read is skipped and its registers take zero, as a null descriptor reads (texel, size, levels all zero).
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const string HeapSamplerStruct = "sharpemu_heap_sampler";

        // The texture types heap elements are read as (each gets its typed element struct) and whether a sampler is.
        private readonly SortedSet<string> _heapTextureTypes = new(StringComparer.Ordinal);
        private bool _usesHeapSamplers;

        // The element index of the heap image access being emitted (its view's record, before the class check) and
        // the validity of the image element TryResolveLayoutImage read last.
        private string? _heapRecordIndex;
        private string? _heapImageValid;

        private static string HeapTypedStruct(string textureType)
        {
            var name = new StringBuilder("sharpemu_heap_");
            foreach (var character in textureType)
            {
                name.Append(char.IsAsciiLetterOrDigit(character) ? character : '_');
            }

            return name.ToString();
        }

        // One struct per texture type the heap is read as, the handle typed: an element is read by reinterpreting it.
        private void EmitHeapElementStructs(StringBuilder source)
        {
            foreach (var textureType in _heapTextureTypes)
            {
                source.AppendLine($"struct {HeapTypedStruct(textureType)} {{ {textureType} texture; }};");
            }

            if (_usesHeapSamplers)
            {
                source.AppendLine($"struct {HeapSamplerStruct} {{ sampler state; }};");
            }

            source.AppendLine();
        }

        private static string HeapLiteral(uint value) => value.ToString(CultureInfo.InvariantCulture) + "u";

        private string HeapFeedbackWord(string index) =>
            $"{ResourcesName}.{Gen5MslArgumentLayout.HeapFeedbackName}[{index}]";

        // The element of the heap a read picks: the dynamic offset its key read kept plus the read's immediate is a byte
        // offset in the heap, whose record the draw's heap view has an element for. A read the view has no element for
        // (past its records, or not at its descriptor's offset in a record) takes the null element, and no index is past
        // the table whatever the view's words say.
        private string HeapRecordIndex(HeapRecordUse use)
        {
            // The same use's index, still in scope and unchanged, is the same value (HeapMarks).
            if (TryKeptHeapRecordIndex(use, out var keptIndex))
            {
                return keptIndex;
            }

            var index = NewHeapRecordIndex(use);
            KeepHeapRecordIndex(use, index);
            return index;
        }

        private string NewHeapRecordIndex(HeapRecordUse use)
        {
            var key = _indirectKeyScratch.TryGetValue(use.KeyMemoryIndex, out var scratch) ? scratch : "0u";
            var viewDword = _request.Bindings.HeapViewsDword + (uint)use.View * HeapView.DwordCount;
            var offset = Temp("uint", $"{key} + {HeapLiteral(use.Immediate)}");
            var word0 = Temp("uint", ShaderDataDword(viewDword));
            var word1 = Temp("uint", ShaderDataDword(viewDword + 1));
            var elementBits = HeapLiteral((uint)HeapView.ElementBits);
            var elementMask = HeapLiteral(HeapView.ElementMask);
            var nullIndex = HeapLiteral(BindlessHeap.CallRecordNullIndex);
            // The host never gives a view a stride of zero; a zero read from a short shader data block divides by one.
            var stride = Temp("uint", $"max({word0} >> {elementBits}, 1u)");
            var record = Temp("uint", $"{offset} / {stride}");
            var inView = Temp("bool",
                $"{offset} - {record} * {stride} == ({word1} >> {elementBits}) && {record} < ({word1} & {elementMask})");
            return Temp("uint", $"{inView} ? min(({word0} & {elementMask}) + {record}, {nullIndex}) : {nullIndex}");
        }

        // The numeric class of an image element (BindlessHeap.ClassCode), which the host keeps after the mark bitsets.
        private string HeapElementClass(string index)
        {
            var word = HeapFeedbackWord($"{HeapLiteral(BindlessHeap.MarkWords)} + ({index} >> {HeapLiteral(BindlessHeap.ClassWordShift)})");
            return Temp("uint",
                $"(atomic_load_explicit(&{word}, memory_order_relaxed) >> (({index} & {HeapLiteral(BindlessHeap.ClassesPerWord - 1)}) * " +
                $"{HeapLiteral(BindlessHeap.ClassBits)})) & {HeapLiteral(BindlessHeap.ClassMask)}");
        }

        // The element a read takes through the declaration of one numeric class: the record's element when it holds a
        // texture of that class and the null element otherwise. A class without integer forms holds float views only.
        private string HeapImageIndex(string index, int slot, ImageNumericClass numericClass)
        {
            if (!BindlessHeap.HoldsIntegers(slot))
            {
                return index;
            }

            if (TryKeptHeapClassIndex(index, slot, numericClass, out var keptIndex))
            {
                return keptIndex;
            }

            var elementClass = HeapElementClass(index);
            var classIndex = Temp("uint",
                $"{elementClass} == {HeapLiteral(BindlessHeap.ClassCode(numericClass))} ? {index} : {HeapLiteral(BindlessHeap.CallRecordNullIndex)}");
            KeepHeapClassIndex(index, slot, numericClass, classIndex);
            return classIndex;
        }

        // Sets the element's bit in its array's mark words, for the lanes the guest runs; a plain read first keeps marked
        // elements from repeating the atomic. A mark an earlier one in the same block already covers is left out (HeapMarks).
        private void MarkHeapElement(int slot, string index)
        {
            if (HeapMarkCovered(slot, index))
            {
                return;
            }

            var word = Temp("uint", $"{HeapLiteral((uint)slot * BindlessHeap.FeedbackWordsPerArray)} + ({index} >> 5u)");
            var bit = Temp("uint", $"1u << ({index} & 31u)");
            var feedback = HeapFeedbackWord(word);
            Line($"if ({ExecCondition($"(atomic_load_explicit(&{feedback}, memory_order_relaxed) & {bit}) == 0u")})");
            Line("{");
            _indent++;
            Line($"atomic_fetch_or_explicit(&{feedback}, {bit}, memory_order_relaxed);");
            _indent--;
            Line("}");
        }

        // The element of one array at index, marked; its handle's validity goes to _heapImageValid.
        private string HeapTexture(int slot, string textureType, string index)
        {
            MarkHeapElement(slot, index);
            var element = Temp("uint", $"{HeapLiteral((uint)slot * BindlessHeap.Capacity)} + {index}");
            var elements = $"{ResourcesName}.{Gen5MslArgumentLayout.HeapElementsName}";
            _heapImageValid = Temp("bool", $"{elements}[{element}] != 0ul");
            _heapTextureTypes.Add(textureType);
            return Temp(textureType, $"((device const {HeapTypedStruct(textureType)}*){elements})[{element}].texture");
        }

        // A sampler element is never zero: the host fills the sampler arrays with the null sampler.
        private string HeapSampler(int slot, HeapRecordUse use)
        {
            var index = HeapRecordIndex(use);
            MarkHeapElement(slot, index);
            _usesHeapSamplers = true;
            var elements = $"{ResourcesName}.{Gen5MslArgumentLayout.HeapElementsName}";
            return Temp("sampler",
                $"((device const {HeapSamplerStruct}*){elements})[{HeapLiteral((uint)slot * BindlessHeap.Capacity)} + {index}].state");
        }

        // The float 2D-array class a 3D heap read takes for a record that holds a 2D texture.
        private static MslImageClass HeapVolumeLayerClass() => new(
            string.Empty,
            Gen5MslArgumentLayout.TextureType(ImageDimension.Dim2DArray, "float", null),
            "float",
            IsStorage: false,
            DepthCompare: false,
            Multisampled: false,
            Arrayed: true,
            Cube: false,
            ImageDimension.Dim2DArray,
            []);

        // The integer form of a float class of the heap: the same element, read with unsigned or signed texels.
        private static MslImageClass HeapIntegerClass(MslImageClass floatClass, ImageNumericClass numericClass)
        {
            var componentKind = numericClass == ImageNumericClass.Sint ? "int" : "uint";
            return floatClass with
            {
                TextureType = Gen5MslArgumentLayout.TextureType(floatClass.Dimension, componentKind, null),
                ComponentKind = componentKind,
                Resources = [],
            };
        }

        // The heap image an access reads: its record use, when its image is a heap record of a class the heap has.
        private bool IsHeapImageAccess(Gen5ShaderInstruction instruction, out HeapRecordUse use, out int slot)
        {
            use = default;
            slot = -1;
            if (!_request.Memory.TryGetIndex(instruction.Pc, 0, out var memoryIndex) ||
                !_request.HeapImageByMemoryIndex.TryGetValue(memoryIndex, out use))
            {
                return false;
            }

            var entry = _request.Memory[memoryIndex];
            var info = _request.Resources.Info;
            if (entry.Resource >= (uint)info.Images.Count || ImageDescriptorBinding.ForImage(info.Images[(int)entry.Resource]) is not { } kind)
            {
                return false;
            }

            slot = BindlessHeap.ImageSlot(kind);
            return slot >= 0;
        }

        // A heap record can hold an integer texture, which shaders read with loads and size queries: a load of a class
        // with integer forms reads the record through the declaration of the element's numeric class.
        private static bool IsHeapLoad(Gen5ShaderInstruction instruction, int slot) =>
            instruction.Opcode is "ImageLoad" or "ImageLoadMip" or "ImageGetResinfo" && BindlessHeap.HoldsIntegers(slot);

        // A heap record a 3D read picks can hold a 2D texture, whose 3D element stays null; the hardware then takes the
        // third coordinate as the array layer. Reads that place their operands alike in both shapes (no gradients) try
        // the 3D element and, when it is null, the record's 2D-array element.
        private static bool IsHeapVolumeRead(Gen5ShaderInstruction instruction, int slot)
        {
            var opcode = instruction.Opcode;
            var sameOperands = opcode is "ImageLoad" or "ImageLoadMip" || opcode.StartsWith("ImageGather4", StringComparison.Ordinal) ||
                (opcode.StartsWith("ImageSample", StringComparison.Ordinal) && !opcode.Contains("SampleD", StringComparison.Ordinal) &&
                 !opcode.Contains("SampleCD", StringComparison.Ordinal) && !opcode.Contains("SampleCd", StringComparison.Ordinal) &&
                 !opcode.Contains("SampleCCd", StringComparison.Ordinal));
            return sameOperands && slot == BindlessHeap.VolumeSlot;
        }

        // An image access whose image is a heap record. The record's element index is found once, outside the branches
        // that read it through each numeric class or shape.
        private bool TryEmitHeapImage(Gen5ShaderInstruction instruction, Gen5ImageControl image, HeapRecordUse use, int slot, out string error)
        {
            _heapRecordIndex = HeapRecordIndex(use);
            try
            {
                if (!IsHeapLoad(instruction, slot))
                {
                    return IsHeapVolumeRead(instruction, slot)
                        ? TryEmitHeapVolumeRead(instruction, image, out error)
                        : TryEmitHeapRead(instruction, image, ImageNumericClass.Float, layers: false, out error);
                }

                var elementClass = HeapElementClass(_heapRecordIndex);
                foreach (var numericClass in new[] { ImageNumericClass.Float, ImageNumericClass.Uint, ImageNumericClass.Sint })
                {
                    Line($"if ({elementClass} == {HeapLiteral(BindlessHeap.ClassCode(numericClass))})");
                    Line("{");
                    _indent++;
                    var emitted = numericClass == ImageNumericClass.Float && IsHeapVolumeRead(instruction, slot)
                        ? TryEmitHeapVolumeRead(instruction, image, out error)
                        : TryEmitHeapRead(instruction, image, numericClass, layers: false, out error);
                    _indent--;
                    Line("}");
                    if (!emitted)
                    {
                        return false;
                    }
                }

                error = string.Empty;
                return true;
            }
            finally
            {
                _heapRecordIndex = null;
                _heapImageValid = null;
            }
        }

        // The 3D element when the record holds a volume, its 2D-array element otherwise.
        private bool TryEmitHeapVolumeRead(Gen5ShaderInstruction instruction, Gen5ImageControl image, out string error)
        {
            if (!TryResolveLayoutImage(instruction, image, out var texture, out var samplerName, out var imageClass, out var dstSelect,
                    out var mipLevel, out var compareFunction, out var packedFormat, out var samplerInfo, out error, out var resourceIndex))
            {
                return false;
            }

            var isVolume = _heapImageValid!;
            Line($"if ({isVolume})");
            Line("{");
            _indent++;
            var emitted = EmitImageOperation(instruction, image, texture, samplerName, imageClass, dstSelect, mipLevel, compareFunction,
                packedFormat, samplerInfo, resourceIndex, out error);
            _indent--;
            Line("}");
            if (!emitted)
            {
                return false;
            }

            Line("else");
            Line("{");
            _indent++;
            emitted = TryEmitHeapRead(instruction, image, ImageNumericClass.Float, layers: true, out error);
            _indent--;
            Line("}");
            return emitted;
        }

        // One read of the record's element through one declaration; a null element reads zero.
        private bool TryEmitHeapRead(Gen5ShaderInstruction instruction, Gen5ImageControl image, ImageNumericClass numericClass, bool layers,
            out string error)
        {
            if (!TryResolveLayoutImage(instruction, image, out var texture, out var samplerName, out var imageClass, out var dstSelect,
                    out var mipLevel, out var compareFunction, out var packedFormat, out var samplerInfo, out error, out var resourceIndex,
                    heapVolumeLayers: layers, heapNumericClass: numericClass))
            {
                return false;
            }

            Line($"if ({_heapImageValid!})");
            Line("{");
            _indent++;
            var emitted = EmitImageOperation(instruction, image, texture, samplerName, imageClass, dstSelect, mipLevel, compareFunction,
                packedFormat, samplerInfo, resourceIndex, out error);
            _indent--;
            Line("}");
            if (!emitted)
            {
                return false;
            }

            Line("else");
            Line("{");
            _indent++;
            EmitNullImageResult(instruction, image);
            _indent--;
            Line("}");
            return true;
        }

        // The registers an image read writes, zero: what a read of a null descriptor returns.
        private void EmitNullImageResult(Gen5ShaderInstruction instruction, Gen5ImageControl image)
        {
            var opcode = instruction.Opcode;
            int count;
            if (opcode == "ImageGetResinfo")
            {
                count = BitOperations.PopCount(image.Dmask & 0xFu);
            }
            else if (opcode == "ImageGetLod")
            {
                count = BitOperations.PopCount((image.Dmask != 0 ? image.Dmask : 1u) & 0x3u);
            }
            else
            {
                count = opcode.StartsWith("ImageGather4", StringComparison.Ordinal) || opcode == "ImageMsaaLoad"
                    ? 4
                    : BitOperations.PopCount(image.Dmask & 0xFu);
                if (image.D16)
                {
                    count = (count + 1) / 2;
                }
            }

            for (var index = 0; index < count; index++)
            {
                StoreVector(image.VectorData + (uint)index, "0u");
            }
        }
    }
}
