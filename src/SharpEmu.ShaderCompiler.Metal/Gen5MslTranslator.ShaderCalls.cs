// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Text;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// Linked shader calls (Gen5ShaderCallLinker). The linker inlines every callee of a call site into the caller: a test per
// target (SCbranchCallMismatch, ShaderCallMatchControl) that branches past the target's copy unless the caller's address
// and argument registers name it, the copy's entry (ShaderCallEntryControl) that writes the return address, the copy with
// its return turned into a branch, and after the last copy the site's fault (ShaderCallFaultControl). The program is
// ordinary control flow over scalar registers, so MSL needs no function pointers: the tests are block terminators, the
// entry an instruction. The callees' addresses and the records' arguments come from the dispatch's shader data
// (BindingLayout.CallTargetsDword, CallArgumentsDword), so one library serves every placement of the code and tables.
//
// With run-time call records (ShaderCallEntryControl.RuntimeArgument) a site holds one copy per callee for every record of
// it: the entry finds the record's row among the dispatch's call records (DescriptorBindingKind.CallRecords), and the
// images and samplers the copy reads through the record's argument are the heap elements the host resolved for that row.
// A call whose callee the program lacks marks it (DescriptorBindingKind.CallMarks) and returns at once.
//
// The SPIR-V translator also merges the copies of a callee into one body and outlines compute bodies into functions
// (Gen5SpirvTranslator.SharedCalls.cs) to bound its driver's compile time; those rewrites keep the linker's run-time
// behaviour, and this translator emits the linker's form as it is.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const string CallRecordRowName = "sharpemu_call_record_row";
        private const string CallRecordSearchName = "sharpemu_search_call_records";
        private const string CallRecordElementName = "sharpemu_call_record_element";

        private bool _hasCallRecords;
        private bool _hasCallMarks;
        private bool _usesCallRecordSearch;
        private Dictionary<ulong, int>? _callTargetSlots;
        private Dictionary<ulong, int>? _callArgumentSlots;

        // The element index of the call record image access being emitted (its row's field), for TryResolveLayoutImage.
        private string? _callRecordElement;

        // A program whose images or samplers a call record holds keeps the row its last entry found.
        private bool UsesCallRecordRow =>
            _request.CallRecordImageByMemoryIndex.Count != 0 || _request.CallRecordSamplerByMemoryIndex.Count != 0;

        private static string HexLiteral(uint value) => "0x" + value.ToString("X", CultureInfo.InvariantCulture) + "u";

        // The slot of each callee among the call targets and of each nonzero argument among the call arguments, when the
        // layout pushes them (as the SPIR-V translator takes them from the linked program before its bodies are shared).
        private void PlanShaderCalls()
        {
            var layout = _request.Bindings;
            if (layout.CallTargetCount != 0)
            {
                _callTargetSlots = [];
                foreach (var target in Gen5ShaderCallLinker.CallTargets(_request.Program))
                {
                    _callTargetSlots.TryAdd(target, _callTargetSlots.Count);
                }
            }

            if (layout.CallArgumentCount != 0)
            {
                _callArgumentSlots = [];
                foreach (var argument in Gen5ShaderCallLinker.CallEntryArguments(_request.Program))
                {
                    _callArgumentSlots.TryAdd(argument, _callArgumentSlots.Count);
                }
            }
        }

        // A 64-bit value the dispatch pushes as two shader data dwords.
        private string ShaderDataPair(uint dword) =>
            $"((ulong){ShaderDataDword(dword)} | ((ulong){ShaderDataDword(dword + 1)} << 32))";

        // A callee's address: the dispatch's call target for it, or its link-time address without pushed targets.
        private string CallTargetExpression(ulong address) =>
            _callTargetSlots is not null && _callTargetSlots.TryGetValue(address, out var slot)
                ? ShaderDataPair(_request.Bindings.CallTargetsDword + (uint)slot * 2)
                : $"{address.ToString(CultureInfo.InvariantCulture)}ul";

        private string ScalarPair(uint register) =>
            $"((ulong){ScalarExpression(register)} | ((ulong){ScalarExpression(register + 1)} << 32))";

        // Whether a test's record argument matches the caller's argument registers: the dispatch's argument for its slot,
        // a null argument as the constant, and every argument for a callee whose records are all run-time.
        private string CallArgumentCondition(uint register, ulong argument, bool runtime)
        {
            if (runtime)
            {
                return "true";
            }

            if (argument != 0 && _callArgumentSlots is not null && _callArgumentSlots.TryGetValue(argument, out var slot))
            {
                var dword = _request.Bindings.CallArgumentsDword + (uint)slot * 2;
                return $"({ScalarExpression(register)} == {ShaderDataDword(dword)} && " +
                    $"{ScalarExpression(register + 1)} == {ShaderDataDword(dword + 1)})";
            }

            return $"({ScalarExpression(register)} == {HexLiteral((uint)argument)} && " +
                $"{ScalarExpression(register + 1)} == {HexLiteral((uint)(argument >> 32))})";
        }

        // The branch of a site's test: taken (past the target's copy) unless the caller names the target and argument.
        private string CallMismatchCondition(ShaderCallMatchControl call) =>
            $"!({ScalarPair(call.AddressRegister)} == {CallTargetExpression(call.Address)} && " +
            $"{CallArgumentCondition(call.ArgumentRegister, call.Argument, call.RuntimeArgument)})";

        // S_GETPC_B64 of a linked function: its address from the call targets, plus its offset and the instruction size.
        private bool TryGetFunctionProgramCounter(Gen5ShaderInstruction instruction, out string address)
        {
            address = string.Empty;
            if (instruction.Control is not ShaderFunctionPcControl function || _callTargetSlots is null)
            {
                return false;
            }

            var offset = unchecked(function.FunctionOffset + (ulong)(instruction.Words.Count * sizeof(uint)));
            address = Temp("ulong", $"{CallTargetExpression(function.FunctionAddress)} + {offset.ToString(CultureInfo.InvariantCulture)}ul");
            return true;
        }

        // A copy's entry: the return address, relocated with the dispatch's shader base as the caller's code is, into the
        // return register pair; with a run-time argument, the row of the record the caller's argument names.
        private void EmitShaderCallEntry(ShaderCallEntryControl call)
        {
            string address;
            if (_request.Bindings.UsesShaderBase)
            {
                var (low, high) = ShaderBaseWords();
                var offset = unchecked(call.ReturnAddress - _request.Program.Address);
                address = Temp("ulong", $"((ulong){low} | ((ulong){high} << 32)) + {offset.ToString(CultureInfo.InvariantCulture)}ul");
            }
            else
            {
                address = Temp("ulong", $"{call.ReturnAddress.ToString(CultureInfo.InvariantCulture)}ul");
            }

            StoreScalar(call.ReturnRegister, $"(uint){address}");
            StoreScalar(call.ReturnRegister + 1, $"(uint)({address} >> 32)");
            if (!call.RuntimeArgument || !_hasCallRecords || !UsesCallRecordRow)
            {
                return;
            }

            var callee = _callTargetSlots is { } slots && slots.TryGetValue(call.Callee, out var slot) ? (uint)slot : uint.MaxValue;
            _usesCallRecordSearch = true;
            Line($"{CallRecordRowName} = {CallRecordSearchName}({ResourcesName}.{Gen5MslArgumentLayout.CallRecordsName}, " +
                $"{ResourcesName}.{Gen5MslArgumentLayout.CallRecordWordsName}, {ScalarExpression(call.ArgumentRegister)}, " +
                $"{ScalarExpression(call.ArgumentRegister + 1)}, {HexLiteral(callee)});");
        }

        // The SEndpgm fault of a program linked per record: the first wave to get here records the call (the caller, the
        // call's pc and target, the stage) for the host, then the wave ends.
        private void EmitShaderCallFault(ShaderCallFaultControl call)
        {
            if (_argumentLayout?.Fields.Any(static field => field.FieldKind == MslArgumentFieldKind.CallFaults) != true)
            {
                return;
            }

            var faults = $"{ResourcesName}.{Gen5MslArgumentLayout.CallFaultsName}";
            uint[] values = [(uint)_request.Hash, (uint)(_request.Hash >> 32), call.CallPc];
            Line($"if ({faults} != nullptr)");
            Line("{");
            _indent++;
            var expected = Temp("uint", "0u");
            Line($"if (atomic_compare_exchange_weak_explicit(&{faults}[0], &{expected}, 2u, memory_order_relaxed, memory_order_relaxed))");
            Line("{");
            _indent++;
            string[] words =
            [
                HexLiteral(values[0]), HexLiteral(values[1]), HexLiteral(values[2]),
                ScalarExpression(call.AddressRegister), ScalarExpression(call.AddressRegister + 1),
                HexLiteral((uint)_request.Stage), "1u",
            ];
            for (var index = 0; index < words.Length; index++)
            {
                Line($"atomic_store_explicit(&{faults}[{index + 1}], {words[index]}, memory_order_relaxed);");
            }

            _indent--;
            Line("}");
            _indent--;
            Line("}");
        }

        // A run-time-record call whose callee the program lacks sets the callee's bit (CallMarks.Bit of its target and this
        // program's hash) before it returns at once; a plain read first keeps a marked callee from repeating the atomic.
        private void EmitCallMark(ShaderCallFaultControl call)
        {
            if (!_hasCallMarks)
            {
                return;
            }

            var marks = $"{ResourcesName}.{Gen5MslArgumentLayout.CallMarksName}";
            var value = Temp("uint",
                $"{ScalarExpression(call.AddressRegister)} ^ ({ScalarExpression(call.AddressRegister + 1)} * 0x9E3779B1u) ^ " +
                HexLiteral(CallMarks.Seed(_request.Hash)));
            value = Temp("uint", $"({value} ^ ({value} >> 16u)) * 0x7FEB352Du");
            value = Temp("uint", $"({value} ^ ({value} >> 15u)) * 0x846CA68Bu");
            var bit = Temp("uint", $"({value} ^ ({value} >> 16u)) & {HexLiteral(CallMarks.Bits - 1)}");
            var mask = Temp("uint", $"1u << ({bit} & 31u)");
            Line($"if ((atomic_load_explicit(&{marks}[{bit} >> 5u], memory_order_relaxed) & {mask}) == 0u)");
            Line("{");
            _indent++;
            Line($"atomic_fetch_or_explicit(&{marks}[{bit} >> 5u], {mask}, memory_order_relaxed);");
            _indent--;
            Line("}");
        }

        // The heap element of one field of the record the last entry found, or the null element.
        private string CallRecordElement(int field)
        {
            if (!_hasCallRecords || !UsesCallRecordRow)
            {
                return HeapLiteral(BindlessHeap.CallRecordNullIndex);
            }

            return Temp("uint", $"{CallRecordElementName}({ResourcesName}.{Gen5MslArgumentLayout.CallRecordsName}, " +
                $"{ResourcesName}.{Gen5MslArgumentLayout.CallRecordWordsName}, {CallRecordRowName}, {(uint)field}u)");
        }

        // The image access whose image a call record holds: its field's element and the heap array of its class.
        private bool IsCallRecordImageAccess(Gen5ShaderInstruction instruction, out CallRecordFieldUse use)
        {
            use = default;
            return _request.Memory.TryGetIndex(instruction.Pc, 0, out var memoryIndex) &&
                _request.CallRecordImageByMemoryIndex.TryGetValue(memoryIndex, out use) && use.Slot >= 0;
        }

        // A call record image read: the element of the record's field, with no mark (the host resolved it with the
        // record); a null element reads zero. A load or size query of a class with integer forms branches on the element's
        // numeric class, which the host wrote with the element, and reads it through that class's declaration, as a heap
        // record load does; any other read takes the float declaration, whose class check reads an integer element null.
        private bool TryEmitCallRecordImage(Gen5ShaderInstruction instruction, Gen5ImageControl image, CallRecordFieldUse use, out string error)
        {
            _callRecordElement = CallRecordElement(use.Field);
            try
            {
                if (!IsHeapLoad(instruction, use.Slot))
                {
                    return TryEmitHeapRead(instruction, image, ImageNumericClass.Float, layers: false, out error);
                }

                var elementClass = HeapElementClass(_callRecordElement);
                foreach (var numericClass in new[] { ImageNumericClass.Float, ImageNumericClass.Uint, ImageNumericClass.Sint })
                {
                    Line($"if ({elementClass} == {HeapLiteral(BindlessHeap.ClassCode(numericClass))})");
                    Line("{");
                    _indent++;
                    var emitted = TryEmitHeapRead(instruction, image, numericClass, layers: false, out error);
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
                _callRecordElement = null;
                _heapImageValid = null;
            }
        }

        // The row search every run-time entry calls and the field lookup of a record read: the port of the SPIR-V
        // translator's searchCallRecords (a branchless binary search for the first row not below the key, which it
        // returns when the row holds the key) and CallRecordElement. Every word they read lies inside the buffer.
        private void EmitShaderCallPrelude(StringBuilder source)
        {
            if (!_hasCallRecords || !UsesCallRecordRow)
            {
                return;
            }

            var keyWords = HeapLiteral(BindlessHeap.CallRecordKeyWords);
            var nullIndex = HeapLiteral(BindlessHeap.CallRecordNullIndex);
            source.AppendLine($"static inline uint {CallRecordElementName}(device const uint* records, uint words, uint row, uint field)");
            source.AppendLine("{");
            source.AppendLine("    uint count = words != 0u ? records[0] : 0u;");
            source.AppendLine("    bool found = row < count;");
            source.AppendLine("    uint safe = found ? row : 0u;");
            source.AppendLine($"    uint first_index = 1u + count * {keyWords} + safe;");
            source.AppendLine("    uint first = words != 0u ? records[first_index < words ? first_index : 0u] : 0u;");
            source.AppendLine("    uint index = first + field;");
            source.AppendLine("    uint element = words != 0u ? records[index < words ? index : 0u] : 0u;");
            source.AppendLine($"    return found ? min(element, {nullIndex}) : {nullIndex};");
            source.AppendLine("}");
            source.AppendLine();
            if (!_usesCallRecordSearch)
            {
                return;
            }

            source.AppendLine($"static uint {CallRecordSearchName}(device const uint* records, uint words, uint low, uint high, uint callee)");
            source.AppendLine("{");
            source.AppendLine($"    uint count = words != 0u ? min(records[0], (words - 1u) / {keyWords}) : 0u;");
            source.AppendLine("    uint position = 0u;");
            source.AppendLine($"    for (uint step = {HeapLiteral((BindlessHeap.CallRecordMaximumRows + 1) >> 1)}; step != 0u; step >>= 1u)");
            source.AppendLine("    {");
            source.AppendLine("        uint probe = position + step;");
            source.AppendLine("        bool in_range = probe <= count;");
            source.AppendLine($"        uint key = 1u + (in_range ? probe - 1u : 0u) * {keyWords};");
            source.AppendLine("        uint row_low = in_range ? records[key] : 0u;");
            source.AppendLine("        uint row_high = in_range ? records[key + 1u] : 0u;");
            source.AppendLine("        uint row_callee = in_range ? records[key + 2u] : 0u;");
            source.AppendLine("        bool below = row_high < high || (row_high == high && (row_low < low || (row_low == low && row_callee < callee)));");
            source.AppendLine("        position = in_range && below ? probe : position;");
            source.AppendLine("    }");
            source.AppendLine("    bool found = position < count;");
            source.AppendLine($"    uint key = 1u + (found ? position : 0u) * {keyWords};");
            source.AppendLine("    bool equal = found && records[key] == low && records[key + 1u] == high && records[key + 2u] == callee;");
            source.AppendLine("    return equal ? position : 0xFFFFFFFFu;");
            source.AppendLine("}");
            source.AppendLine();
        }
    }
}
