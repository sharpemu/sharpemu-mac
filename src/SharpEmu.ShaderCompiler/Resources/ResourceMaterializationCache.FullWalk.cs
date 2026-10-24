// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;

namespace SharpEmu.ShaderCompiler.Resources;

public sealed partial class ResourceMaterializationCache
{
    private static readonly bool FullWalkDetail = Environment.GetEnvironmentVariable("SHARPEMU_RESOURCE_FULLWALK_DETAIL") == "1";
    private readonly Dictionary<ulong, (ulong Key, Entry Entry)> _walkPrevious = new();
    private static readonly Dictionary<string, long> WalkDetails = new();
    private static readonly object WalkDetailGate = new();
    private static long _walkDetailDropped;

    // Compares the fresh successful walk with the previous entry for this exact key,
    // or the previous draw of this plan/base/state when the key itself changed.
    // No diagnostic read can introduce a wait or consume an overlay.
    private void RecordFullWalk(ulong drawKey, ulong key, Entry? stale, Entry built)
    {
        if (!FullWalkDetail) return;
        _walkPrevious.TryGetValue(drawKey, out var prior);
        var previous = stale ?? prior.Entry;
        if (_walkPrevious.Count >= 4096) _walkPrevious.Clear();
        _walkPrevious[drawKey] = (key, built);
        lock (WalkDetailGate)
        {
            var prefix = $"hash={built.Plan.Hash:X16} stage={built.Plan.Stage}";
            void Count(string name, long amount = 1)
            {
                var nameKey = prefix + " " + name;
                if (WalkDetails.Count < 131072 || WalkDetails.ContainsKey(nameKey))
                    WalkDetails[nameKey] = WalkDetails.GetValueOrDefault(nameKey) + amount;
                else _walkDetailDropped++;
            }
            Count("walks");
            if (previous is null) { Count("first_sight"); return; }
            var comparison = stale is not null ? "same_key" : prior.Key == key ? "evicted_or_unmatched_key" : "different_key";
            Count(comparison);
            var changed = 0;
            var newWords = 0;
            var previousRange = 0;
            for (var range = 0; range < built.RangeAddresses.Length; range++)
            {
                var address = built.RangeAddresses[range];
                for (var offset = 0; offset < built.RangeLengths[range]; offset += 4)
                {
                    var wordAddress = address + (ulong)offset;
                    while (previousRange < previous.RangeAddresses.Length &&
                           previous.RangeAddresses[previousRange] + (ulong)previous.RangeLengths[previousRange] <= wordAddress)
                        previousRange++;
                    if (previousRange >= previous.RangeAddresses.Length || previous.RangeAddresses[previousRange] > wordAddress)
                    { newWords++; continue; }
                    var oldOffset = previous.RangeOffsets[previousRange] + (int)(wordAddress - previous.RangeAddresses[previousRange]);
                    var newOffset = built.RangeOffsets[range] + offset;
                    if (BinaryPrimitives.ReadUInt32LittleEndian(previous.Bytes.AsSpan(oldOffset)) ==
                        BinaryPrimitives.ReadUInt32LittleEndian(built.Bytes.AsSpan(newOffset))) continue;
                    changed++;
                    Count(built.WordTableOnly[newOffset / 4] ? "changed_table_only_words" : "changed_descriptor_or_mixed_words");
                    Count(comparison + (built.WordTableOnly[newOffset / 4] ? " changed_table_only_words" : " changed_descriptor_or_mixed_words"));
                    Count($"range={range} word={offset / 4} changed");
                }
            }
            Count($"changed_words={changed} new_words={newWords}");
            Count($"{comparison} changed_words={changed} new_words={newWords}");
            Count("compared_words", built.Bytes.Length / 4 - newWords);
            Count("new_address_words", newWords);
            void Compare(string name, uint[] before, uint[] after)
            {
                if (before.Length != after.Length) Count(name + "_length_changed");
                for (var word = 0; word < Math.Min(before.Length, after.Length); word++)
                    if (before[word] != after[word])
                    {
                        Count($"{name} word={word} changed");
                        Count($"{comparison} {name} word={word} changed");
                    }
            }
            Compare("user_data", previous.UserData, built.UserData);
            Compare("table", previous.Snapshot.FlattenedResourceTable, built.Snapshot.FlattenedResourceTable);
            void Descriptors(string kind, uint[][] before, uint[][] after)
            {
                for (var index = 0; index < Math.Min(before.Length, after.Length); index++)
                    Compare($"{kind}={index}", before[index], after[index]);
            }
            Descriptors("buffer", previous.Snapshot.Buffers, built.Snapshot.Buffers);
            Descriptors("image", previous.Snapshot.Images, built.Snapshot.Images);
            Descriptors("sampler", previous.Snapshot.Samplers, built.Snapshot.Samplers);
            Count(previous.Specialization.Equals(built.Specialization) ? "specialization_equal" : "specialization_changed");
            Count(comparison + (previous.Specialization.Equals(built.Specialization) ? " specialization_equal" : " specialization_changed"));
        }
    }

    private static string TakeFullWalkReport()
    {
        if (!FullWalkDetail) return string.Empty;
        lock (WalkDetailGate)
        {
            var result = new System.Text.StringBuilder();
            foreach (var (key, count) in WalkDetails.OrderBy(pair => pair.Key))
                result.Append($"\n[PERF][RESOURCE_FULLWALK_DETAIL] {key} count={count}");
            result.Append($"\n[PERF][RESOURCE_FULLWALK_DETAIL_OVERFLOW] dropped={_walkDetailDropped}");
            _walkDetailDropped = 0;
            WalkDetails.Clear();
            return result.ToString();
        }
    }
}
