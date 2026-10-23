// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

// Owns immutable native commands and their directly encoded references. Current
// preparation/acquisition still certifies every invocation, including indirect
// argument resources. Neither a hash nor a retained no-copy window certifies guest
// mapping/content validity. Busy entries are never reset, patched or evicted.
internal sealed class MetalIndirectCommands(nint device) : IDisposable
{
    internal sealed class Entry(nint buffer, MetalRecipeCommand[] commands, MetalRecipeBuffer[] bindings,
        bool inheritBuffers, nint[] references, long encodeTicks,
        nint patchBuffer, ulong patchAddress, ulong[] patchOffsets)
    {
        internal readonly nint Buffer = buffer;
        internal readonly MetalRecipeCommand[] Commands = commands;
        internal readonly MetalRecipeBuffer[] Bindings = bindings;
        internal readonly bool InheritBuffers = inheritBuffers;
        internal readonly nint[] References = references;
        internal readonly long EncodeTicks = encodeTicks;
        internal readonly nint PatchBuffer = patchBuffer;
        internal readonly ulong PatchAddress = patchAddress;
        internal readonly ulong[] PatchOffsets = patchOffsets;
        internal ulong PatchBytes;
        internal ulong LastSerial;
        internal long LastPass;
        internal long LastUse;
    }

    private const int MaxCommands = 16384, MaxEntries = 256;
    private readonly Dictionary<int, List<Entry>> _entries = new();
    private int _commandCount, _entryCount;
    private ulong _patchBytes;
    private long _use;
    private static readonly nint Release = MetalNative.Selector("release");
    private static readonly nint SetPipeline = MetalNative.Selector("setRenderPipelineState:");
    private static readonly nint SetVertex = MetalNative.Selector("setVertexBuffer:offset:atIndex:");
    private static readonly nint SetFragment = MetalNative.Selector("setFragmentBuffer:offset:atIndex:");
    private static readonly nint Draw = MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:");
    private static readonly nint DrawIndexed = MetalNative.Selector(
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:");

    internal Entry? Get(MetalRecipeCommands sequence, ulong serial, long pass, ulong completed, out bool reused)
    {
        reused = false;
        var source = CollectionsMarshal.AsSpan(sequence.Commands);
        var buffers = CollectionsMarshal.AsSpan(sequence.Bindings);
        var inheritBuffers = true;
        var first = source[0];
        foreach (var command in source)
        {
            inheritBuffers &= buffers.Slice(first.FirstBinding, first.BindingCount)
                .SequenceEqual(buffers.Slice(command.FirstBinding, command.BindingCount));
        }
        var hash = new HashCode();
        hash.Add(inheritBuffers);
        foreach (var command in source) hash.Add(Key(command, inheritBuffers));
        if (!inheritBuffers) foreach (var binding in buffers) hash.Add(KeyBinding(binding));
        var key = hash.ToHashCode();
        if (_entries.TryGetValue(key, out var bucket))
            foreach (var entry in bucket)
            {
                if (entry.InheritBuffers != inheritBuffers ||
                    entry.Commands.Length != source.Length || (!inheritBuffers && !EqualBindings(buffers, entry.Bindings))) continue;
                // A TBDR cannot order a later vertex overwrite behind earlier
                // fragment reads in the same pass. Use a distinct immutable ICB/
                // argument bank for each such invocation; recycle across encoders
                // through normal tracked-resource GPU dependencies.
                if (entry.PatchBuffer != 0 && entry.LastPass == pass) continue;
                var equal = true;
                for (var i = 0; i < source.Length && equal; i++)
                    equal = entry.Commands[i] == Key(source[i], inheritBuffers);
                if (!equal) continue;
                reused = true; entry.LastSerial = serial; entry.LastPass = pass; entry.LastUse = ++_use;
                return entry;
            }
        var patchOffsets = inheritBuffers ? [] : new ulong[buffers.Length];
        ulong patchBytes = 0;
        if (!inheritBuffers)
            for (var i = 0; i < buffers.Length; i++)
            {
                if (buffers[i].Index >= 2) continue;
                patchOffsets[i] = patchBytes;
                patchBytes += ((ulong)buffers[i].ByteSize + 255) & ~255UL;
            }
        if (patchBytes > 8 * 1024 * 1024 || !MakeRoom(source.Length, patchBytes, completed)) return null;
        var descriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLIndirectCommandBufferDescriptor"),
            MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        if (descriptor == 0) return null;
        nint icb;
        try
        {
            MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setCommandTypes:"), 3); // Draw | DrawIndexed
            MetalNative.SendVoidBool(descriptor, MetalNative.Selector("setInheritPipelineState:"), false);
            MetalNative.SendVoidBool(descriptor, MetalNative.Selector("setInheritBuffers:"), inheritBuffers);
            MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setMaxVertexBufferBindCount:"), inheritBuffers ? 0u : 31u);
            MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setMaxFragmentBufferBindCount:"), inheritBuffers ? 0u : 2u);
            icb = MetalV2Native.SendNewIndirectCommands(device,
                MetalNative.Selector("newIndirectCommandBufferWithDescriptor:maxCommandCount:options:"), descriptor,
                (nuint)source.Length, Mtl.ResourceStorageModeShared);
        }
        finally { MetalNative.SendVoid(descriptor, Release); }
        if (icb == 0) return null;
        var references = new HashSet<nint>();
        var patchBuffer = patchBytes == 0 ? 0 : MetalNative.SendNewBuffer(device,
            MetalNative.Selector("newBufferWithLength:options:"), (nuint)patchBytes, Mtl.ResourceStorageModePrivate);
        if (patchBytes != 0 && patchBuffer == 0) { MetalNative.SendVoid(icb, Release); return null; }
        var patchAddress = patchBuffer == 0 ? 0 : MetalNative.SendGpuAddress(patchBuffer, MetalNative.Selector("gpuAddress"));
        var started = Stopwatch.GetTimestamp();
        for (var i = 0; i < source.Length; i++)
        {
            var command = source[i];
            var native = MetalNative.Send(icb, MetalNative.Selector("indirectRenderCommandAtIndex:"), i);
            var bindings = buffers.Slice(command.FirstBinding, command.BindingCount);
            // Key and own the exact PSO, including joined colour/depth variants.
            // A reused template never substitutes a different parent pipeline.
            MetalNative.SendVoid(native, SetPipeline, command.Pipeline);
            references.Add(command.Pipeline);
            if (!inheritBuffers)
            {
                for (var j = 0; j < bindings.Length; j++)
                {
                    var binding = bindings[j];
                    if (binding.Index < 2) binding = binding with { Buffer = patchBuffer, Offset = patchOffsets[command.FirstBinding + j] };
                    MetalNative.SendSetBuffer(native, binding.Stage == 0 ? SetVertex : SetFragment,
                        binding.Buffer, (nuint)binding.Offset, (nuint)binding.Index);
                    if (binding.Buffer != 0 && binding.Buffer != patchBuffer) references.Add(binding.Buffer);
                }
            }
            EncodeDraw(native, command);
            if (command.Operation == MetalRecipeOperation.DrawIndexed) references.Add(command.IndexBuffer);
        }
        var ticks = Stopwatch.GetTimestamp() - started;
        foreach (var reference in references) MetalNative.SendVoid(reference, MetalNative.Selector("retain"));
        var created = new Entry(icb, source.ToArray().Select(c => Key(c, inheritBuffers)).ToArray(),
            inheritBuffers ? [] : buffers.ToArray().Select(KeyBinding).ToArray(), inheritBuffers, references.ToArray(), ticks,
            patchBuffer, patchAddress, patchOffsets)
            { LastSerial = serial, LastPass = pass, LastUse = ++_use, PatchBytes = patchBytes };
        // MakeRoom can have removed the old bucket, so reacquire it.
        if (!_entries.TryGetValue(key, out bucket)) _entries[key] = bucket = new();
        bucket.Add(created); _commandCount += source.Length; _entryCount++; _patchBytes += patchBytes;
        return created;
    }

    private static MetalRecipeCommand Key(MetalRecipeCommand command, bool buffers) => command with
    {
        Serial = 0, Pass = 0, ParentState = default,
        IndexBuffer = command.Operation == MetalRecipeOperation.Draw ? 0 : command.IndexBuffer,
        IndexOffset = command.Operation == MetalRecipeOperation.Draw ? 0 : command.IndexOffset,
        IndexType = command.Operation == MetalRecipeOperation.Draw ? 0 : command.IndexType,
        FirstBinding = buffers ? 0 : command.FirstBinding, BindingCount = buffers ? 0 : command.BindingCount,
    };

    private static MetalRecipeBuffer KeyBinding(MetalRecipeBuffer binding) => binding.Index < 2
        ? binding with { Buffer = 0, Offset = 0 } : binding;

    private static bool EqualBindings(ReadOnlySpan<MetalRecipeBuffer> current, ReadOnlySpan<MetalRecipeBuffer> cached)
    {
        if (current.Length != cached.Length) return false;
        for (var i = 0; i < current.Length; i++) if (KeyBinding(current[i]) != cached[i]) return false;
        return true;
    }

    internal static void Bind(nint encoder, ReadOnlySpan<MetalRecipeBuffer> bindings)
    {
        foreach (var binding in bindings)
            MetalNative.SendSetBuffer(encoder, binding.Stage == 0 ? SetVertex : SetFragment, binding.Buffer,
                (nuint)binding.Offset, (nuint)binding.Index);
    }

    internal static void EncodeDraw(nint encoder, MetalRecipeCommand command)
    {
        if (command.Operation == MetalRecipeOperation.Draw)
            MetalV2Native.SendDrawPrimitivesBaseInstance(encoder, Draw, command.PrimitiveType,
                (nuint)command.C, (nuint)command.A, (nuint)Math.Max(command.B, 1), (nuint)command.D);
        else
            MetalV2Native.SendDrawIndexedBaseVertex(encoder, DrawIndexed, command.PrimitiveType,
                (nuint)command.A, command.IndexType, command.IndexBuffer,
                (nuint)(command.IndexOffset + command.C * (command.IndexType == Mtl.IndexTypeUInt32 ? 4UL : 2UL)),
                (nuint)Math.Max(command.B, 1), unchecked((nint)command.D), (nuint)command.E);
    }

    private bool MakeRoom(int count, ulong patchBytes, ulong completed)
    {
        while (_commandCount + count > MaxCommands || _entryCount >= MaxEntries || _patchBytes + patchBytes > 32 * 1024 * 1024)
        {
            var oldest = _entries.SelectMany(pair => pair.Value).Where(entry => entry.LastSerial <= completed)
                .MinBy(entry => entry.LastUse);
            if (oldest is null) return false;
            int? emptyKey = null;
            foreach (var pair in _entries)
                if (pair.Value.Remove(oldest)) { if (pair.Value.Count == 0) emptyKey = pair.Key; break; }
            if (emptyKey is { } key) _entries.Remove(key);
            _commandCount -= oldest.Commands.Length; _entryCount--; _patchBytes -= oldest.PatchBytes;
            Free(oldest);
        }
        return true;
    }

    private static void Free(Entry entry)
    {
        MetalNative.SendVoid(entry.Buffer, Release);
        if (entry.PatchBuffer != 0) MetalNative.SendVoid(entry.PatchBuffer, Release);
        foreach (var reference in entry.References) MetalNative.SendVoid(reference, Release);
    }

    // The host has drained every in-flight serial before disposing this cache.
    public void Dispose()
    {
        foreach (var entry in _entries.SelectMany(pair => pair.Value)) Free(entry);
        _entries.Clear();
    }
}
