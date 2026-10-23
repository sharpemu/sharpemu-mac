// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

internal enum MetalRecipeOperation { None, Draw, DrawIndexed, DrawIndirect, DrawIndexedIndirect, Dispatch, DispatchIndirect }
internal readonly record struct MetalRecipeBuffer(int Stage, int Index, nint Buffer, ulong Offset, uint ByteSize = 0);
internal readonly record struct MetalRecipeCommand(ulong Serial, long Pass, bool Compute, nint Pipeline,
    MetalRecipeOperation Operation, ulong A, ulong B, ulong C, ulong D, ulong E,
    int FirstBinding, int BindingCount, nint IndexBuffer, ulong IndexOffset, nuint IndexType,
    DynamicDrawState ParentState, nuint PrimitiveType, MtlSize ThreadsPerGroup, bool NativeDirect);

// Ordered, packed encode-side data for a later ICB consumer. Parent state and pass
// boundaries are explicit; unsupported native operations remain ordinary commands.
// Handles are observations, NOT cross-serial leases: an ICB consumer must validate
// the preparation recipe and retain its resources before using an old sequence.
internal sealed class MetalRecipeCommands
{
    internal readonly List<MetalRecipeCommand> Commands = new(8192);
    internal readonly List<MetalRecipeBuffer> Bindings = new(32768);
    internal long Dropped;
    internal void Clear() { Commands.Clear(); Bindings.Clear(); Dropped = 0; }
}
