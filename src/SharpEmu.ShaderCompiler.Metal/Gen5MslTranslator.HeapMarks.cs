// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.RegularExpressions;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// Heap marks within a block, the MSL form of the SPIR-V translator's MarkHeapRecord dedup (Gen5SpirvTranslator.Resources.cs,
// _markedHeapRecords): a record a read marked is not marked again by a later read of the same record in the same
// straight-line code, under the same EXEC. SPIR-V finds "the same record" through its value numbering (the same index id
// within a block); MSL has none, so the record index of a use (its key, view and immediate) and the element index of a
// numeric class are kept while they are in scope and their inputs are unchanged, and a mark is left out when:
// - the earlier mark's scope is still open (it ran before this point on every path here),
// - the index is the same temp (the same value),
// - and the earlier mark covered these lanes: it ran with EXEC known full, or no line since wrote EXEC.
// Every guest block starts with nothing kept (TryEmitBlocks), as does every jump target, and lines captured to be placed
// elsewhere (CaptureLines) drop everything kept, so no kept mark or index crosses code it does not dominate.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        // The scopes open where the body is being emitted, innermost last; 0 is the block's own scope.
        private readonly List<int> _heapScopes = [0];
        private int _heapScopeSerial;
        // Bumped by every line that may write EXEC.
        private int _heapExecSerial;

        private readonly Dictionary<(int Slot, string Index), (int Scope, bool Full, int Exec)> _heapMarks = [];
        private readonly Dictionary<(int Key, int View, uint Immediate), (string Index, int Scope)> _heapRecordIndexes = [];
        private readonly Dictionary<(string Index, int Slot, ImageNumericClass Class), (string Index, int Scope)> _heapClassIndexes = [];

        // A write to EXEC (=, &=, |=, ^=, not ==) and a write to an indirect key's scratch.
        private static readonly Regex HeapExecWrite = new(@"(^|[^A-Za-z0-9_.])exec\s*[&|^]?=(?!=)", RegexOptions.CultureInvariant);
        private static readonly Regex HeapJumpTarget = new(@"^\s*(case\b|default\s*:|[A-Za-z_][A-Za-z0-9_]*\s*:(?!:))", RegexOptions.CultureInvariant);

        private int HeapScope => _heapScopes[^1];

        private bool HeapScopeOpen(int scope) => _heapScopes.Contains(scope);

        // Nothing kept: a new block, a jump target, or lines moved elsewhere.
        private void ForgetHeapMarks()
        {
            _heapMarks.Clear();
            _heapRecordIndexes.Clear();
            _heapClassIndexes.Clear();
            _heapScopes.Clear();
            _heapScopes.Add(0);
            _heapExecSerial++;
        }

        // Follows the scopes, EXEC writes, key writes and jump targets of each line the body takes (Line).
        private void NoteHeapLine(string text)
        {
            if (_heapMarks.Count == 0 && _heapRecordIndexes.Count == 0 && _heapClassIndexes.Count == 0)
            {
                // Nothing kept: only the scopes matter, and they restart with the next kept entry's scope.
                _heapScopes.Clear();
                _heapScopes.Add(0);
                return;
            }

            if (text.Contains('\n') || HeapJumpTarget.IsMatch(text))
            {
                ForgetHeapMarks();
                return;
            }

            var trimmed = text.Trim();
            if (trimmed.StartsWith('}'))
            {
                if (_heapScopes.Count > 1)
                {
                    _heapScopes.RemoveAt(_heapScopes.Count - 1);
                }
                else
                {
                    // A scope closed that opened before anything was kept.
                    ForgetHeapMarks();
                    return;
                }
            }

            if (trimmed.EndsWith('{'))
            {
                _heapScopes.Add(++_heapScopeSerial);
            }

            if (trimmed.Contains("exec", StringComparison.Ordinal) && HeapExecWrite.IsMatch(trimmed))
            {
                _heapExecSerial++;
            }

            if (trimmed.StartsWith("indirect_image_key", StringComparison.Ordinal))
            {
                _heapRecordIndexes.Clear();
                _heapClassIndexes.Clear();
            }
        }

        // The record index a use took earlier in this scope, unchanged since.
        private bool TryKeptHeapRecordIndex(HeapRecordUse use, out string index)
        {
            if (_heapRecordIndexes.TryGetValue((use.KeyMemoryIndex, use.View, use.Immediate), out var kept) && HeapScopeOpen(kept.Scope))
            {
                index = kept.Index;
                return true;
            }

            index = string.Empty;
            return false;
        }

        private void KeepHeapRecordIndex(HeapRecordUse use, string index) =>
            _heapRecordIndexes[(use.KeyMemoryIndex, use.View, use.Immediate)] = (index, HeapScope);

        private bool TryKeptHeapClassIndex(string index, int slot, ImageNumericClass numericClass, out string classIndex)
        {
            if (_heapClassIndexes.TryGetValue((index, slot, numericClass), out var kept) && HeapScopeOpen(kept.Scope))
            {
                classIndex = kept.Index;
                return true;
            }

            classIndex = string.Empty;
            return false;
        }

        private void KeepHeapClassIndex(string index, int slot, ImageNumericClass numericClass, string classIndex) =>
            _heapClassIndexes[(index, slot, numericClass)] = (classIndex, HeapScope);

        // True when an earlier mark of the element at index already covers this one; otherwise this mark is kept.
        private bool HeapMarkCovered(int slot, string index)
        {
            if (_heapMarks.TryGetValue((slot, index), out var kept) && HeapScopeOpen(kept.Scope) &&
                (kept.Full || kept.Exec == _heapExecSerial))
            {
                return true;
            }

            _heapMarks[(slot, index)] = (HeapScope, _execKnownFull, _heapExecSerial);
            return false;
        }
    }
}
