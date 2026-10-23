// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using System.Text;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// Which way a block-copy kernel moves bytes between the guest surface and a linear buffer.
internal enum MetalTileDirection
{
    Detile,
    Tile,
}

// Assembles the MSL tiler kernels: block copies per tile block kind, the 16-bit depth
// conversions, the BGRA16 channel swap and the unaligned row repack. The MSL twin of
// the Vulkan shader compiler's TilerShaders; both are driven by the same tile tables,
// and TileGeometry owns the block dimensions so the three stay in lockstep.
//
// One source per (kind, direction) carries all five element sizes behind the
// kElementBytes function constant, so a kind costs one newLibraryWithSource and the
// element sizes cost one newFunctionWithName:constantValues: each. Emitting every kind
// into a single library would compile ~1200 lines of MSL before the first transfer; a
// library per element size would be a hundred compiles instead.
internal static partial class MetalTilerShaders
{
    // The 14 dwords every block-copy kernel reads, and the subset the conversions use:
    // the 13 the Vulkan tiler shares, and then baseY for a transfer cut down to a run of
    // block rows.
    public const int ArgumentCount = 14;

    public const int ArgumentBytes = ArgumentCount * sizeof(uint);

    // Buffer indices shared by every kernel in both libraries. The target buffer is bound
    // twice: 1 for whole-dword stores, 2 for the masked atomics that sub-dword elements and
    // unaligned row edges need. MSL cannot reinterpret a device pointer as atomic.
    public const uint SourceIndex = 0;
    public const uint TargetIndex = 1;
    public const uint TargetAtomicIndex = 2;
    public const uint ArgumentIndex = 3;

    // Function constant indices. The fast-copy variant means whole-element vector loads
    // for the four-byte-and-wider elements and one dword per element group for the
    // sub-dword ones; the two are mutually exclusive, so they share one constant.
    public const uint ElementBytesConstant = 0;
    public const uint FastCopyConstant = 1;

    // The depth texture a writeback reads, at [[texture(0)]] of the writeback kernels.
    public const uint TextureIndex = 0;

    // The subresource a texture block copy reads or writes: array slice and mip level of
    // the unsigned-integer 2D-array view, bound with setBytes beside the tile arguments.
    public const uint SubresourceIndex = 4;

    // The element sizes a texture block copy carries: one, two or four 32-bit channels
    // of the view, which is every colour texel the tiled transfers move whole.
    public static bool SupportsTextureCopy(uint elementBytes) => elementBytes is 4 or 8 or 16;

    public const string DepthWriteback = "writeback_depth";
    public const string DepthWritebackArray = "writeback_depth_array";

    public const string DepthWidenFixed24 = "depth_widen_d24";
    public const string DepthWidenFloat32 = "depth_widen_d32";
    public const string DepthNarrowFixed24 = "depth_narrow_d24";
    public const string DepthNarrowFloat32 = "depth_narrow_d32";
    public const string SwapBgra16 = "swap_bgra16";
    public const string RepackRows = "repack_rows";

    private enum Axis
    {
        X,
        Y,
        Z,
    }

    private readonly record struct Term(Axis Source, int Shift, uint Mask);

    private static Term X(int shift, uint mask) => new(Axis.X, shift, mask);

    private static Term Y(int shift, uint mask) => new(Axis.Y, shift, mask);

    private static Term Z(int shift, uint mask) => new(Axis.Z, shift, mask);

    private readonly record struct AddressBit(Axis Source, byte Bit);

    private static AddressBit BX(byte bit) => new(Axis.X, bit);

    private static AddressBit BY(byte bit) => new(Axis.Y, bit);

    private static AddressBit[] BXY(byte x, byte y) => [BX(x), BY(y)];

    private static uint Log2(uint value) => (uint)BitOperations.TrailingZeroCount(value);

    private static readonly uint[] ElementSizes = [1, 2, 4, 8, 16];

    private static Term[] StandardTerms(uint bytes) => bytes switch
    {
        1 => [Y(4, 0x1f0), Y(5, 0x400), X(0, 0x00f), X(5, 0x200), X(6, 0x800)],
        2 => [Y(4, 0x070), Y(5, 0x100), Y(6, 0x400), X(1, 0x00e), X(4, 0x080), X(5, 0x200), X(6, 0x800)],
        4 => [Y(4, 0x070), Y(5, 0x100), Y(6, 0x400), X(2, 0x00c), X(5, 0x080), X(6, 0x200), X(7, 0x800)],
        8 => [Y(4, 0x030), Y(6, 0x100), Y(7, 0x400), X(3, 0x008), X(5, 0x0c0), X(6, 0x200), X(7, 0x800)],
        _ => [Y(4, 0x030), Y(6, 0x100), Y(7, 0x400), X(6, 0x0c0), X(7, 0x200), X(8, 0x800)],
    };

    private static Term[] Standard3DTerms(uint bytes) => bytes switch
    {
        1 => [X(0, 3), X(4, 0x40), X(6, 0x200), Y(3, 8), Y(4, 0x20), Y(6, 0x100), Y(8, 0x800), Z(2, 4), Z(3, 0x10), Z(5, 0x80), Z(7, 0x400)],
        2 => [X(1, 2), X(5, 0x40), X(7, 0x200), Y(3, 8), Y(4, 0x20), Y(6, 0x100), Y(8, 0x800), Z(2, 4), Z(3, 0x10), Z(5, 0x80), Z(7, 0x400)],
        4 => [X(2, 4), X(5, 0x40), X(7, 0x200), Y(3, 8), Y(4, 0x20), Y(6, 0x100), Y(8, 0x800), Z(4, 0x10), Z(6, 0x80), Z(8, 0x400)],
        8 => [X(3, 8), X(5, 0x40), X(7, 0x200), Y(5, 0x20), Y(7, 0x100), Y(9, 0x800), Z(4, 0x10), Z(6, 0x80), Z(8, 0x400)],
        _ => [X(6, 0x40), X(8, 0x200), Y(5, 0x20), Y(7, 0x100), Y(9, 0x800), Z(4, 0x10), Z(6, 0x80), Z(8, 0x400)],
    };

    private static Term[] Standard64KBTerms(uint bytes) => bytes switch
    {
        1 => [X(0, 0x0f), X(5, 0x200), X(6, 0x800), X(7, 0x2000), X(8, 0x8000), Y(4, 0x1f0), Y(5, 0x400), Y(6, 0x1000), Y(7, 0x4000)],
        2 => [X(1, 0x00e), X(4, 0x080), X(5, 0x200), X(6, 0x800), X(7, 0x2000), X(8, 0x8000), Y(4, 0x070), Y(5, 0x100), Y(6, 0x400), Y(7, 0x1000), Y(8, 0x4000)],
        4 => [X(2, 0x00c), X(5, 0x080), X(6, 0x200), X(7, 0x800), X(8, 0x2000), X(9, 0x8000), Y(4, 0x070), Y(5, 0x100), Y(6, 0x400), Y(7, 0x1000), Y(8, 0x4000)],
        8 => [X(3, 0x008), X(5, 0x0c0), X(6, 0x200), X(7, 0x800), X(8, 0x2000), X(9, 0x8000), Y(4, 0x030), Y(6, 0x100), Y(7, 0x400), Y(8, 0x1000), Y(9, 0x4000)],
        _ => [X(6, 0x0c0), X(7, 0x200), X(8, 0x800), X(9, 0x2000), X(10, 0x8000), Y(4, 0x030), Y(6, 0x100), Y(7, 0x400), Y(8, 0x1000), Y(9, 0x4000)],
    };

    private static Term[] Gen5RenderTargetTerms(uint bytes) => bytes switch
    {
        1 => [Y(2, 0x008), Y(4, 0x010), Y(3, 0x0a0), Y(5, 0xf00), Y(6, 0x1000), Y(7, 0x4000), X(0, 0x007), X(3, 0x040), X(5, 0x300), X(4, 0x400), X(6, 0x800), X(7, 0x2000), X(8, 0x8000)],
        2 => [Y(4, 0x070), Y(5, 0xf00), Y(8, 0x5000), X(1, 0x00e), X(4, 0x480), X(5, 0x300), X(6, 0x800), X(7, 0x2000), X(8, 0x8000)],
        4 => [Y(4, 0x070), Y(5, 0xf00), Y(9, 0x1000), Y(8, 0x4000), X(2, 0x00c), X(5, 0x380), X(4, 0x400), X(6, 0x800), X(9, 0xa000)],
        8 => [Y(4, 0x010), Y(6, 0x080), Y(5, 0xf00), Y(10, 0x5000), X(3, 0x008), X(4, 0x460), X(5, 0x300), X(6, 0x800), X(10, 0x2000), X(9, 0x8000)],
        _ => [X(4, 0x410), X(5, 0x340), X(6, 0x800), X(11, 0xa000), Y(5, 0xf20), Y(6, 0x080), Y(10, 0x1000), Y(11, 0x4000)],
    };

    private static Term[] DepthTerms(uint bytes) => bytes switch
    {
        1 => [X(0, 1), X(1, 0x004), X(2, 0x010), X(3, 0x040), X(5, 0x300), X(4, 0x400), X(6, 0x800), X(7, 0x2000), X(8, 0x8000), Y(1, 0x002), Y(2, 0x008), Y(3, 0x0a0), Y(5, 0xf00), Y(6, 0x1000), Y(7, 0x4000)],
        2 => [X(1, 0x002), X(2, 0x008), X(3, 0x020), X(4, 0x480), X(5, 0x300), X(6, 0x800), X(7, 0x2000), X(8, 0x8000), Y(2, 0x004), Y(3, 0x010), Y(4, 0x040), Y(5, 0xf00), Y(8, 0x5000)],
        4 => [X(2, 0x004), X(3, 0x010), X(4, 0x440), X(5, 0x300), X(6, 0x800), X(9, 0xa000), Y(3, 0x008), Y(4, 0x020), Y(5, 0xf80), Y(9, 0x1000), Y(8, 0x4000)],
        _ => [X(3, 0x008), X(4, 0x420), X(5, 0x380), X(6, 0x800), X(10, 0x2000), X(9, 0x8000), Y(4, 0x010), Y(5, 0xf40), Y(10, 0x5000)],
    };

    // PS5 Oberon RB+ R_X addressing. The destination bit index is the array index. Keep
    // this in lockstep with GnmTiling.RbPlus64KRenderX and TileGeometry.RenderTargetOffset.
    private static readonly AddressBit[][][] RenderTargetAddressBits =
    [
        [
            [BX(0)], [BX(1)], [BX(2)], [BX(3)], [BY(0)], [BY(1)], [BY(2)], [BY(3)],
            BXY(7, 4).Concat([BY(7)]).ToArray(), BXY(4, 4), BXY(6, 5), BXY(5, 6), [BX(6)], [BY(6)], BXY(7, 8), BXY(8, 7),
        ],
        [
            [], [BX(0)], [BX(1)], [BX(2)], [BY(0)], [BY(1)], [BY(2)], [BX(3)],
            BXY(7, 4).Concat([BY(7)]).ToArray(), BXY(4, 4), BXY(6, 5), BXY(5, 6), [BY(3)], [BX(6)], BXY(7, 7), BXY(8, 6),
        ],
        [
            [], [], [BX(0)], [BX(1)], [BY(0)], [BY(1)], [BX(2)], [BY(2)],
            BXY(7, 4).Concat([BY(7)]).ToArray(), BXY(4, 4), BXY(6, 5), BXY(5, 6), [BX(3)], [BY(3)], BXY(6, 7), BXY(7, 6),
        ],
        [
            [], [], [], [BX(0)], [BY(0)], [BX(1)], [BX(2)], [BY(1)],
            BXY(7, 4).Concat([BY(7)]).ToArray(), BXY(4, 4), BXY(6, 5), BXY(5, 6), [BY(2)], [BX(3)], BXY(7, 3), BXY(6, 6),
        ],
        [
            [], [], [], [], [BX(0)], [BY(0)], [BX(1)], [BY(1)],
            BXY(7, 4).Concat([BY(7)]).ToArray(), BXY(4, 4), BXY(6, 5), BXY(5, 6), [BX(2)], [BY(2)], BXY(6, 3), BXY(3, 6),
        ],
    ];

    private static readonly byte[][] Standard64KBVolumeSources = [[4, 4, 4, 5], [3, 4, 4, 4], [3, 3, 4, 4], [3, 3, 3, 4], [2, 3, 3, 3]];
    private static readonly byte[][] Prt64KBVolumeSources = [[4, 5, 4, 4], [4, 4, 3, 4], [4, 4, 3, 3], [3, 4, 3, 3], [3, 3, 2, 3]];
    private static readonly byte[][] Prt64KBSources = [[7, 7, 6, 6], [7, 6, 6, 5], [6, 6, 5, 5], [6, 5, 5, 4], [5, 5, 4, 4]];

    private static string AxisName(Axis axis) => axis switch
    {
        Axis.X => "sx",
        Axis.Y => "sy",
        _ => "sz",
    };

    private static string TermExpression(Term term)
    {
        var source = AxisName(term.Source);
        return term.Shift == 0
            ? $"({source} & 0x{term.Mask:x}u)"
            : $"(({source} << {term.Shift}) & 0x{term.Mask:x}u)";
    }

    private static string XorTerms(Term[] terms) => $"({string.Join(" ^ ", terms.Select(TermExpression))})";

    private static string Bit(Axis axis, uint sourceBit, uint destinationBit) =>
        $"((({AxisName(axis)} >> {sourceBit}u) & 1u) << {destinationBit}u)";

    // The surface layer bits the 64 KiB render-target and depth kinds fold into every element.
    private const string ZSpread = "(((sz & 8u) << 5) ^ ((sz & 4u) << 7) ^ ((sz & 2u) << 9) ^ ((sz & 1u) << 11))";

    private static string Standard64KBVolumeOffset(uint bytes)
    {
        var bits = Standard64KBVolumeSources[Log2(bytes)];
        return $"({XorTerms(Standard3DTerms(bytes))} ^ {Bit(Axis.X, bits[0], 12)} ^ {Bit(Axis.Z, bits[1], 13)} ^ " +
               $"{Bit(Axis.Y, bits[2], 14)} ^ {Bit(Axis.X, bits[3], 15)})";
    }

    private static string RenderTargetOffset(uint bytes)
    {
        var table = RenderTargetAddressBits[Log2(bytes)];
        var parts = new List<string>();
        for (uint destination = 0; destination < table.Length; destination++)
        {
            foreach (var bit in table[destination])
            {
                parts.Add(Bit(bit.Source, bit.Bit, destination));
            }
        }

        return $"({string.Join(" ^ ", parts)})";
    }

    // The byte offset of one element inside its block, as an MSL expression over sx, sy and sz.
    private static string BlockOffset(TileBlockKind kind, uint bytes)
    {
        switch (kind)
        {
            case TileBlockKind.Standard256B:
                return $"({XorTerms(StandardTerms(bytes))} & 0xffu)";
            case TileBlockKind.Standard4KB:
                return XorTerms(StandardTerms(bytes));
            case TileBlockKind.Standard4KB3D:
                return XorTerms(Standard3DTerms(bytes));
            case TileBlockKind.Standard64KB:
                return XorTerms(Standard64KBTerms(bytes));
            case TileBlockKind.Standard64KB3D:
                return Standard64KBVolumeOffset(bytes);
            case TileBlockKind.Prt64KB:
            {
                var bits = Prt64KBSources[Log2(bytes)];
                return $"({XorTerms(Standard64KBTerms(bytes))} ^ {Bit(Axis.X, bits[0], 8)} ^ {Bit(Axis.Y, bits[1], 9)} ^ " +
                       $"{Bit(Axis.X, bits[2], 10)} ^ {Bit(Axis.Y, bits[3], 11)})";
            }

            case TileBlockKind.Prt64KB3D:
            {
                var bits = Prt64KBVolumeSources[Log2(bytes)];
                return $"({Standard64KBVolumeOffset(bytes)} ^ {Bit(Axis.Y, bits[0], 10)} ^ {Bit(Axis.X, bits[1], 10)} ^ " +
                       $"{Bit(Axis.X, bits[2], 11)} ^ {Bit(Axis.Z, bits[3], 11)})";
            }

            case TileBlockKind.RenderTarget64KB:
                return $"({RenderTargetOffset(bytes)} ^ {ZSpread})";
            case TileBlockKind.RenderTarget64KBGen5:
                return $"({XorTerms(Gen5RenderTargetTerms(bytes))} ^ {ZSpread})";
            case TileBlockKind.Depth64KB:
                return $"({ZSpread} ^ {XorTerms(DepthTerms(bytes))})";
            default:
                throw GpuFatal.Fatal($"The tiler has no block offset for this kind: kind={(uint)kind}.");
        }
    }

    public static string KernelName(TileBlockKind kind, MetalTileDirection direction) =>
        $"{(direction == MetalTileDirection.Detile ? "detile" : "tile")}_{kind.ToString().ToLowerInvariant()}";

    // One thread per element. Thin kinds take a 16x16 tile, which covers a contiguous run
    // of tiled bytes for every element size; thick kinds take 8x8x4 for the same reason.
    public static (uint X, uint Y, uint Z) Threadgroup(TileBlockKind kind) => kind switch
    {
        TileBlockKind.Standard4KB3D or TileBlockKind.Standard64KB3D or TileBlockKind.Prt64KB3D => (8, 8, 4),
        _ => (16, 16, 1),
    };

    public static bool Supports(TileBlockKind kind, uint elementBytes) =>
        TileGeometry.TryGetBlockLayout(kind, elementBytes, out _);

    // The element group whose bytes are one aligned dword on the given side of a
    // sub-dword copy. The linear side is always a run of 4/bytes elements along x; the
    // tiled side follows the kind's own addressing, derived from the shared block tables
    // rather than asserted, so a kind whose addressing fits no group keeps the atomic
    // path. The 1-byte depth kind interleaves x and y in the two low address bits, which
    // puts a 2x2 quad in one tiled dword; every other kind keeps x's low bits there.
    private static readonly Dictionary<(TileBlockKind Kind, uint Bytes), (uint Width, uint Height)> _tiledGroups = [];

    public static (uint Width, uint Height) DwordGroup(TileBlockKind kind, uint bytes, bool tiledSide)
    {
        if (bytes >= 4 || !TileGeometry.TryGetBlockLayout(kind, bytes, out var block))
        {
            return default;
        }

        var elements = 4 / bytes;
        if (!tiledSide)
        {
            return (elements, 1);
        }

        lock (_tiledGroups)
        {
            if (_tiledGroups.TryGetValue((kind, bytes), out var cached))
            {
                return cached;
            }

            var found = default((uint Width, uint Height));
            (uint Width, uint Height)[] candidates = elements == 4
                ? [(4, 1), (2, 2), (1, 4)]
                : [(2, 1), (1, 2)];
            foreach (var candidate in candidates)
            {
                if (GroupFillsOneDword(block, candidate.Width, candidate.Height))
                {
                    found = candidate;
                    break;
                }
            }

            _tiledGroups[(kind, bytes)] = found;
            return found;
        }
    }

    // What one thread of a sub-dword copy covers: the smallest rectangle that is a whole
    // number of dwords on both sides, so the thread writes only full dwords and reads
    // only dwords it consumes entirely. The two sides' groups are powers of two, so that
    // rectangle is just the larger extent of each. Both sides matter even though only the
    // destination has to be written whole: a 1-byte depth retile whose group were the
    // 2x2 quad alone would read half of two linear dwords, and its detile whose group
    // were the 4x1 run alone would read half of two tiled dwords.
    public static (uint Width, uint Height) WideGroup(TileBlockKind kind, uint bytes)
    {
        if (bytes >= 4 || !TileGeometry.TryGetBlockLayout(kind, bytes, out var block))
        {
            return default;
        }

        var tiled = DwordGroup(kind, bytes, tiledSide: true);
        var linear = DwordGroup(kind, bytes, tiledSide: false);
        if (tiled.Width == 0)
        {
            return default;
        }

        var group = (Width: Math.Max(tiled.Width, linear.Width), Height: Math.Max(tiled.Height, linear.Height));
        return block.BlockWidth % group.Width == 0 && block.BlockHeight % group.Height == 0 ? group : default;
    }

    // True when every aligned gw x gh group of elements inside the block occupies the
    // four bytes of one aligned dword, which is what lets a retile store it whole.
    private static bool GroupFillsOneDword(in TileBlockLayout block, uint gw, uint gh)
    {
        var bytes = block.BytesPerElement;
        if (block.BlockWidth % gw != 0 || block.BlockHeight % gh != 0 || gw * gh * bytes != 4)
        {
            return false;
        }

        var full = (1u << (int)(gw * gh)) - 1u;
        for (uint z = 0; z < block.BlockDepth; z++)
        {
            for (uint y = 0; y < block.BlockHeight; y += gh)
            {
                for (uint x = 0; x < block.BlockWidth; x += gw)
                {
                    var first = uint.MaxValue;
                    for (uint element = 0; element < gw * gh; element++)
                    {
                        if (!TileGeometry.TryGetBlockOffset(block, x + element % gw, y + element / gw, z, out var offset))
                        {
                            return false;
                        }

                        first = Math.Min(first, offset);
                    }

                    if (first % 4 != 0)
                    {
                        return false;
                    }

                    uint seen = 0;
                    for (uint element = 0; element < gw * gh; element++)
                    {
                        TileGeometry.TryGetBlockOffset(block, x + element % gw, y + element / gw, z, out var offset);
                        if (offset < first || offset >= first + 4 || (offset - first) % bytes != 0)
                        {
                            return false;
                        }

                        seen |= 1u << (int)((offset - first) / bytes);
                    }

                    if (seen != full)
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    private const string Preamble = """
        #include <metal_stdlib>
        using namespace metal;

        // The 13 dwords GpuTiler's TileTransferArguments carries, in the same order,
        // and then baseY: the surface row this transfer's first row really is. A
        // transfer restricted to a run of block rows addresses its own rows from zero
        // on both sides, but the block offset swizzle folds in y bits above the block,
        // so it has to be told where the run sits in the surface.
        struct TileArgs
        {
            uint sourceBase;
            uint destinationBase;
            uint width;
            uint height;
            uint depth;
            uint surfaceZ;
            uint pitchBytes;
            uint sliceBytes;
            uint blocksPerRow;
            uint blocksPerSlice;
            uint tailX;
            uint tailY;
            uint tail;
            uint baseY;
        };

        """;

    // The element's byte offset inside its block, as a function so the one-element and
    // the dword-wide bodies share a single copy of the addressing expression.
    private static void AppendOffsetFunction(StringBuilder source, TileBlockKind kind, uint bytes)
    {
        if (!TileGeometry.TryGetBlockLayout(kind, bytes, out _))
        {
            return;
        }

        source.Append($"static inline uint tiled_offset_{bytes}(uint sx, uint sy, uint sz)\n{{\n");
        source.Append($"    return {BlockOffset(kind, bytes)};\n}}\n\n");
    }

    // The element-size body of a block copy: block index, both addresses and the copy itself.
    private static void AppendBody(StringBuilder source, TileBlockKind kind, MetalTileDirection direction, uint bytes)
    {
        if (!TileGeometry.TryGetBlockLayout(kind, bytes, out var block))
        {
            return;
        }

        var widthShift = Log2(block.BlockWidth);
        var heightShift = Log2(block.BlockHeight);
        var depthShift = Log2(block.BlockDepth);
        var detile = direction == MetalTileDirection.Detile;
        source.Append($"    if (kElementBytes == {bytes}u)\n    {{\n");
        if (bytes < 4)
        {
            AppendWideBody(source, kind, direction, bytes, block);
        }

        source.Append("        if (gid.x >= args.width || gid.y >= args.height)\n        {\n            return;\n        }\n\n");
        source.Append("        uint sx = tail ? gid.x + args.tailX : gid.x;\n");
        source.Append("        uint sy = gid.y + (tail ? args.tailY : args.baseY);\n");
        source.Append("        uint sz = gid.z + args.surfaceZ;\n");
        source.Append($"        uint block = tail ? 0u : ((gid.z >> {depthShift}u) * args.blocksPerSlice + " +
                      $"(gid.y >> {heightShift}u) * args.blocksPerRow + (gid.x >> {widthShift}u));\n");
        source.Append($"        uint tiled = block * {block.BlockSize}u + tiled_offset_{bytes}(sx, sy, sz);\n");
        source.Append($"        uint linear = gid.z * args.sliceBytes + gid.y * args.pitchBytes + gid.x * {bytes}u;\n");
        source.Append($"        uint from = args.sourceBase + {(detile ? "tiled" : "linear")};\n");
        source.Append($"        uint to = args.destinationBase + {(detile ? "linear" : "tiled")};\n");
        if (bytes >= 4)
        {
            if (bytes == 4)
            {
                source.Append("        target[to >> 2] = source[from >> 2];\n");
            }
            else
            {
                var vector = bytes == 8 ? "uint2" : "uint4";
                source.Append("        if (kFastCopy)\n        {\n");
                source.Append($"            *reinterpret_cast<device {vector}*>(target + (to >> 2)) = " +
                              $"*reinterpret_cast<device const {vector}*>(source + (from >> 2));\n");
                source.Append("        }\n        else\n        {\n");
                for (uint word = 0; word < bytes; word += 4)
                {
                    source.Append($"            target[(to + {word}u) >> 2] = source[(from + {word}u) >> 2];\n");
                }

                source.Append("        }\n");
            }
        }
        else
        {
            var mask = bytes == 1 ? "0xffu" : "0xffffu";
            source.Append($"        uint value = (source[from >> 2] >> ((from & 3u) * 8u)) & {mask};\n");
            source.Append("        uint shift = (to & 3u) * 8u;\n");
            source.Append("        device atomic_uint* cell = targetAtomic + (to >> 2);\n");
            source.Append($"        atomic_fetch_and_explicit(cell, ~({mask} << shift), memory_order_relaxed);\n");
            source.Append("        atomic_fetch_or_explicit(cell, value << shift, memory_order_relaxed);\n");
        }

        source.Append("        return;\n    }\n");
    }

    // The dword-wide body for a sub-dword element size. One thread covers the rectangle
    // that is a whole number of dwords on both sides and walks it one destination dword
    // at a time, assembling that dword's elements and storing it whole; every dword it
    // reads is consumed by the same thread, so nothing is loaded twice. A dword the
    // surface's width or height cuts short falls back to the masked atomics for the bytes
    // it does cover, so the row padding is still never written.
    private static void AppendWideBody(
        StringBuilder source, TileBlockKind kind, MetalTileDirection direction, uint bytes, in TileBlockLayout block)
    {
        var group = WideGroup(kind, bytes);
        var detile = direction == MetalTileDirection.Detile;
        var dword = DwordGroup(kind, bytes, tiledSide: !detile);
        if (group.Width == 0 || dword.Width == 0)
        {
            return;
        }

        var mask = bytes == 1 ? "0xffu" : "0xffffu";
        source.Append("        if (kFastCopy)\n        {\n");
        source.Append($"            uint x0 = gid.x * {group.Width}u;\n");
        source.Append($"            uint y0 = gid.y * {group.Height}u;\n");
        source.Append("            if (x0 >= args.width || y0 >= args.height)\n            {\n                return;\n            }\n\n");
        source.Append("            uint sz = gid.z + args.surfaceZ;\n");
        source.Append($"            uint block = (gid.z >> {Log2(block.BlockDepth)}u) * args.blocksPerSlice + " +
                      $"(y0 >> {Log2(block.BlockHeight)}u) * args.blocksPerRow + (x0 >> {Log2(block.BlockWidth)}u);\n");
        source.Append($"            uint blockBase = block * {block.BlockSize}u;\n");
        source.Append($"            for (uint dy = 0u; dy < {group.Height / dword.Height}u; ++dy)\n            {{\n");
        source.Append($"                for (uint dx = 0u; dx < {group.Width / dword.Width}u; ++dx)\n                {{\n");
        source.Append($"                    uint bx = x0 + dx * {dword.Width}u;\n");
        source.Append($"                    uint by = y0 + dy * {dword.Height}u;\n");
        source.Append("                    uint value = 0u;\n                    uint mask = 0u;\n");
        source.Append(detile
            ? "                    uint to = args.destinationBase + gid.z * args.sliceBytes + by * args.pitchBytes + " +
              $"bx * {bytes}u;\n"
            : "                    uint to = 0u;\n");
        source.Append($"                    for (uint j = 0u; j < {dword.Height}u; ++j)\n                    {{\n");
        source.Append("                        uint ly = by + j;\n");
        source.Append("                        if (ly >= args.height)\n                        {\n                            break;\n                        }\n\n");
        source.Append("                        uint sy = ly + args.baseY;\n");
        source.Append($"                        for (uint i = 0u; i < {dword.Width}u; ++i)\n                        {{\n");
        source.Append("                            uint sx = bx + i;\n");
        source.Append("                            if (sx >= args.width)\n                            {\n                                break;\n                            }\n\n");
        source.Append($"                            uint tiled = args.{(detile ? "sourceBase" : "destinationBase")} + blockBase + " +
                      $"tiled_offset_{bytes}(sx, sy, sz);\n");
        source.Append($"                            uint linear = args.{(detile ? "destinationBase" : "sourceBase")} + " +
                      $"gid.z * args.sliceBytes + ly * args.pitchBytes + sx * {bytes}u;\n");
        source.Append($"                            uint from = {(detile ? "tiled" : "linear")};\n");
        source.Append($"                            uint shift = ({(detile ? "linear" : "tiled")} & 3u) * 8u;\n");
        if (!detile)
        {
            source.Append("                            to = tiled & ~3u;\n");
        }

        source.Append($"                            value |= ((source[from >> 2] >> ((from & 3u) * 8u)) & {mask}) << shift;\n");
        source.Append($"                            mask |= {mask} << shift;\n");
        source.Append("                        }\n                    }\n\n");
        source.Append("                    if (mask == 0xffffffffu)\n                    {\n");
        source.Append("                        target[to >> 2] = value;\n                    }\n");
        source.Append("                    else if (mask != 0u)\n                    {\n");
        source.Append("                        device atomic_uint* cell = targetAtomic + (to >> 2);\n");
        source.Append("                        atomic_fetch_and_explicit(cell, ~mask, memory_order_relaxed);\n");
        source.Append("                        atomic_fetch_or_explicit(cell, value, memory_order_relaxed);\n");
        source.Append("                    }\n");
        source.Append("                }\n            }\n\n");
        source.Append("            return;\n        }\n\n");
    }

    // The library for one block kind and direction; the element size is a function constant.
    public static string CreateBlockCopyLibrary(TileBlockKind kind, MetalTileDirection direction)
    {
        var any = false;
        foreach (var bytes in ElementSizes)
        {
            any |= TileGeometry.TryGetBlockLayout(kind, bytes, out _);
        }

        if (!any)
        {
            throw GpuFatal.Fatal($"The tiler has no block layout for this kind: kind={(uint)kind}.");
        }

        var source = new StringBuilder(Preamble);
        source.Append("constant uint kElementBytes [[function_constant(0)]];\n");
        source.Append("constant bool kFastCopy [[function_constant(1)]];\n\n");
        foreach (var bytes in ElementSizes)
        {
            AppendOffsetFunction(source, kind, bytes);
        }

        source.Append($"kernel void {KernelName(kind, direction)}(\n");
        source.Append($"    device const uint* source [[buffer({SourceIndex})]],\n");
        source.Append($"    device uint* target [[buffer({TargetIndex})]],\n");
        source.Append($"    device atomic_uint* targetAtomic [[buffer({TargetAtomicIndex})]],\n");
        source.Append($"    constant TileArgs& args [[buffer({ArgumentIndex})]],\n");
        source.Append("    uint3 gid [[thread_position_in_grid]])\n{\n");
        // The x and y bounds live in the element-size bodies: a dword-wide body indexes
        // groups of elements, so its own bound is the group's first element.
        source.Append("    if (gid.z >= args.depth)\n    {\n        return;\n    }\n\n");
        source.Append("    bool tail = args.tail != 0u;\n");
        foreach (var bytes in ElementSizes)
        {
            AppendBody(source, kind, direction, bytes);
        }

        source.Append("}\n");
        return source.ToString();
    }

    public static string TextureKernelName(TileBlockKind kind, MetalTileDirection direction) =>
        $"{(direction == MetalTileDirection.Detile ? "detile_to_texture" : "tile_from_texture")}_{kind.ToString().ToLowerInvariant()}";

    // The colour twin of the depth writeback, in both directions: one pass between the
    // tiled guest surface and the texture itself, through an unsigned-integer 2D-array
    // view whose texel is the guest element bit for bit. The buffer path detiles into a
    // staging buffer and blits that into the texture (or blits out and retiles), so it
    // moves every byte twice; this moves it once. The addressing is the block copy's own:
    // the same block index, the same in-block offset and the same baseY for a run of
    // block rows, whose texel rows start at baseY.
    public static string CreateTextureBlockCopyLibrary(TileBlockKind kind, MetalTileDirection direction)
    {
        var source = new StringBuilder(Preamble);
        source.Append("constant uint kElementBytes [[function_constant(0)]];\n\n");
        source.Append("struct TextureSubresource\n{\n    uint slice;\n    uint level;\n};\n\n");
        var any = false;
        foreach (var bytes in ElementSizes)
        {
            if (SupportsTextureCopy(bytes) && TileGeometry.TryGetBlockLayout(kind, bytes, out _))
            {
                AppendOffsetFunction(source, kind, bytes);
                any = true;
            }
        }

        if (!any)
        {
            throw GpuFatal.Fatal($"The texture tiler has no block layout for this kind: kind={(uint)kind}.");
        }

        var detile = direction == MetalTileDirection.Detile;
        source.Append($"kernel void {TextureKernelName(kind, direction)}(\n");
        source.Append(detile
            ? $"    device const uint* source [[buffer({SourceIndex})]],\n" +
              $"    texture2d_array<uint, access::write> surface [[texture({TextureIndex})]],\n"
            : $"    texture2d_array<uint, access::read> surface [[texture({TextureIndex})]],\n" +
              $"    device uint* target [[buffer({TargetIndex})]],\n");
        source.Append($"    constant TileArgs& args [[buffer({ArgumentIndex})]],\n");
        source.Append($"    constant TextureSubresource& subresource [[buffer({SubresourceIndex})]],\n");
        source.Append("    uint3 gid [[thread_position_in_grid]])\n{\n");
        source.Append("    if (gid.x >= args.width || gid.y >= args.height)\n    {\n        return;\n    }\n\n");
        source.Append("    uint sy = gid.y + args.baseY;\n");
        source.Append("    uint2 texel = uint2(gid.x, sy);\n");
        foreach (var bytes in ElementSizes)
        {
            if (!SupportsTextureCopy(bytes) || !TileGeometry.TryGetBlockLayout(kind, bytes, out var block))
            {
                continue;
            }

            source.Append($"    if (kElementBytes == {bytes}u)\n    {{\n");
            source.Append($"        uint block = (gid.y >> {Log2(block.BlockHeight)}u) * args.blocksPerRow + " +
                          $"(gid.x >> {Log2(block.BlockWidth)}u);\n");
            source.Append($"        uint tiled = block * {block.BlockSize}u + tiled_offset_{bytes}(gid.x, sy, args.surfaceZ);\n");
            var words = bytes / 4;
            if (detile)
            {
                source.Append("        uint at = (args.sourceBase + tiled) >> 2;\n");
                source.Append("        uint4 value = uint4(source[at], ");
                source.Append(words > 1 ? "source[at + 1u], " : "0u, ");
                source.Append(words > 2 ? "source[at + 2u], source[at + 3u]);\n" : "0u, 0u);\n");
                source.Append("        surface.write(value, texel, subresource.slice, subresource.level);\n");
            }
            else
            {
                source.Append("        uint4 value = surface.read(texel, subresource.slice, subresource.level);\n");
                source.Append("        uint at = (args.destinationBase + tiled) >> 2;\n");
                for (uint word = 0; word < words; word++)
                {
                    source.Append($"        target[at + {word}u] = value[{word}];\n");
                }
            }

            source.Append("        return;\n    }\n");
        }

        source.Append("}\n");
        return source.ToString();
    }

    // The depth writeback: one pass that reads the Metal depth texture and tiles the
    // guest plane straight out of it, instead of blitting the plane into a staging buffer
    // and retiling that. Compute cannot write a depth texture, so there is no upload twin
    // of this; the way in stays a detile followed by a buffer-to-texture blit.
    //
    // A four-byte guest element is the texture's own float, stored bit for bit, and one
    // thread writes one dword. A two-byte element is the 16-bit normalised depth the
    // widening upload came from, so the narrowing lives here too, and one thread writes
    // the pair of elements that share a destination dword.
    public static string CreateDepthWritebackLibrary()
    {
        if (!TileGeometry.TryGetBlockLayout(TileBlockKind.Depth64KB, 4, out var wide) ||
            !TileGeometry.TryGetBlockLayout(TileBlockKind.Depth64KB, 2, out var narrow))
        {
            throw GpuFatal.Fatal("The depth tile layout is missing the two- and four-byte element sizes.");
        }

        var source = new StringBuilder(Preamble);
        source.Append("constant uint kElementBytes [[function_constant(0)]];\n\n");
        AppendOffsetFunction(source, TileBlockKind.Depth64KB, 2);
        AppendOffsetFunction(source, TileBlockKind.Depth64KB, 4);
        foreach (var (name, type, read) in new[]
                 {
                     (DepthWriteback, "depth2d<float, access::read>", "plane.read(uint2(sx, sy))"),
                     (DepthWritebackArray, "depth2d_array<float, access::read>", "plane.read(uint2(sx, sy), args.surfaceZ)"),
                 })
        {
            source.Append($"kernel void {name}(\n");
            source.Append($"    {type} plane [[texture({TextureIndex})]],\n");
            source.Append($"    device uint* target [[buffer({TargetIndex})]],\n");
            source.Append($"    device atomic_uint* targetAtomic [[buffer({TargetAtomicIndex})]],\n");
            source.Append($"    constant TileArgs& args [[buffer({ArgumentIndex})]],\n");
            source.Append("    uint3 gid [[thread_position_in_grid]])\n{\n");
            source.Append("    if (kElementBytes == 4u)\n    {\n");
            source.Append("        uint sx = gid.x;\n        uint sy = gid.y + args.baseY;\n");
            source.Append("        if (sx >= args.width || sy >= args.height)\n        {\n            return;\n        }\n\n");
            source.Append($"        uint block = (sy >> {Log2(wide.BlockHeight)}u) * args.blocksPerRow + " +
                          $"(sx >> {Log2(wide.BlockWidth)}u);\n");
            source.Append($"        uint to = args.destinationBase + block * {wide.BlockSize}u + " +
                          "tiled_offset_4(sx, sy, args.surfaceZ);\n");
            source.Append($"        target[to >> 2] = as_type<uint>({read});\n        return;\n    }}\n\n");
            source.Append("    uint x0 = gid.x * 2u;\n    uint sy = gid.y + args.baseY;\n");
            source.Append("    if (x0 >= args.width || sy >= args.height)\n    {\n        return;\n    }\n\n");
            source.Append("    uint value = 0u;\n    uint mask = 0u;\n    uint to = 0u;\n");
            source.Append("    for (uint i = 0u; i < 2u; ++i)\n    {\n");
            source.Append("        uint sx = x0 + i;\n");
            source.Append("        if (sx >= args.width)\n        {\n            break;\n        }\n\n");
            source.Append($"        uint block = (sy >> {Log2(narrow.BlockHeight)}u) * args.blocksPerRow + " +
                          $"(sx >> {Log2(narrow.BlockWidth)}u);\n");
            source.Append($"        uint tiled = args.destinationBase + block * {narrow.BlockSize}u + " +
                          "tiled_offset_2(sx, sy, args.surfaceZ);\n");
            source.Append($"        uint encoded = uint(round(clamp({read}, 0.0f, 1.0f) * 65535.0f));\n");
            source.Append("        uint shift = (tiled & 3u) * 8u;\n");
            source.Append("        to = tiled & ~3u;\n");
            source.Append("        value |= (encoded & 0xffffu) << shift;\n");
            source.Append("        mask |= 0xffffu << shift;\n");
            source.Append("    }\n\n");
            source.Append("    if (mask == 0u)\n    {\n        return;\n    }\n\n");
            source.Append("    if (mask == 0xffffffffu)\n    {\n        target[to >> 2] = value;\n        return;\n    }\n\n");
            source.Append("    device atomic_uint* cell = targetAtomic + (to >> 2);\n");
            source.Append("    atomic_fetch_and_explicit(cell, ~mask, memory_order_relaxed);\n");
            source.Append("    atomic_fetch_or_explicit(cell, value, memory_order_relaxed);\n");
            source.Append("}\n\n");
        }

        return source.ToString();
    }

    // The library every transfer needs besides the block copies: the 16-bit depth
    // conversions, the BGRA16 channel swap and the row repack that covers linear
    // surfaces whose guest offset or pitch a blit cannot address directly.
    public static string CreateConversionLibrary()
    {
        var source = new StringBuilder(Preamble);
        source.Append("""
            // Widens 16-bit depth to 24-bit fixed point or 32-bit float, one dword per element.
            // pitchBytes is the source row stride, sliceBytes the target row stride.
            static inline void widen(device const uint* source, device uint* target,
                constant TileArgs& args, uint2 gid, bool float32)
            {
                uint from = args.sourceBase + gid.y * args.pitchBytes + gid.x * 2u;
                uint value = (source[from >> 2] >> ((from & 2u) * 8u)) & 0xffffu;
                uint encoded = float32
                    ? as_type<uint>(float(value) * (1.0f / 65535.0f))
                    : value * 256u + (value + 128u) / 257u;
                uint to = args.destinationBase + gid.y * args.sliceBytes + gid.x * 4u;
                target[to >> 2] = encoded;
            }

            // Narrows 24-bit fixed point or 32-bit float depth to 16 bits with a masked write.
            static inline void narrow(device const uint* source, device atomic_uint* target,
                constant TileArgs& args, uint2 gid, bool float32)
            {
                uint from = args.sourceBase + gid.y * args.pitchBytes + gid.x * 4u;
                uint to = args.destinationBase + gid.y * args.sliceBytes + gid.x * 2u;
                uint raw = source[from >> 2];
                float normalized = float32
                    ? clamp(as_type<float>(raw), 0.0f, 1.0f)
                    : float(raw & 0x00ffffffu) * (1.0f / 16777215.0f);
                uint decoded = uint(round(normalized * 65535.0f));
                uint shift = (to & 2u) * 8u;
                device atomic_uint* cell = target + (to >> 2);
                atomic_fetch_and_explicit(cell, ~(0xffffu << shift), memory_order_relaxed);
                atomic_fetch_or_explicit(cell, decoded << shift, memory_order_relaxed);
            }

            """);
        foreach (var (name, float32, widening) in new[]
                 {
                     (DepthWidenFixed24, false, true),
                     (DepthWidenFloat32, true, true),
                     (DepthNarrowFixed24, false, false),
                     (DepthNarrowFloat32, true, false),
                 })
        {
            source.Append($"kernel void {name}(\n");
            source.Append($"    device const uint* source [[buffer({SourceIndex})]],\n");
            source.Append($"    device uint* target [[buffer({TargetIndex})]],\n");
            source.Append($"    device atomic_uint* targetAtomic [[buffer({TargetAtomicIndex})]],\n");
            source.Append($"    constant TileArgs& args [[buffer({ArgumentIndex})]],\n");
            source.Append("    uint2 gid [[thread_position_in_grid]])\n{\n");
            source.Append("    if (gid.x >= args.width || gid.y >= args.height)\n    {\n        return;\n    }\n\n");
            source.Append(widening
                ? $"    widen(source, target, args, gid, {(float32 ? "true" : "false")});\n"
                : $"    narrow(source, targetAtomic, args, gid, {(float32 ? "true" : "false")});\n");
            source.Append("}\n\n");
        }

        source.Append($$"""
            // Swaps the 16-bit blue and red channels of BGRA16 pixels; width counts pixels.
            kernel void {{SwapBgra16}}(
                device const uint* source [[buffer({{SourceIndex}})]],
                device uint* target [[buffer({{TargetIndex}})]],
                device atomic_uint* targetAtomic [[buffer({{TargetAtomicIndex}})]],
                constant TileArgs& args [[buffer({{ArgumentIndex}})]],
                uint gid [[thread_position_in_grid]])
            {
                if (gid >= args.width)
                {
                    return;
                }

                uint from = (args.sourceBase >> 2) + gid * 2u;
                uint to = (args.destinationBase >> 2) + gid * 2u;
                uint bg = source[from];
                uint ra = source[from + 1u];
                target[to] = (ra & 0x0000ffffu) | (bg & 0xffff0000u);
                target[to + 1u] = (bg & 0x0000ffffu) | (ra & 0xffff0000u);
            }

            // Copies rowBytes bytes per row between two row-strided ranges at any byte
            // offset. One thread per destination dword; the row's edge dwords land through
            // masked atomics, so a partially covered dword keeps its other bytes.
            struct RepackArgs
            {
                uint sourceBase;
                uint destinationBase;
                uint rowWords;
                uint rows;
                uint slices;
                uint rowBytes;
                uint sourceRowBytes;
                uint targetRowBytes;
                uint sourceSliceBytes;
                uint targetSliceBytes;
                // A row's last destination dword may reach past the copied source bytes.
                uint sourceWords;
            };

            kernel void {{RepackRows}}(
                device const uint* source [[buffer({{SourceIndex}})]],
                device uint* target [[buffer({{TargetIndex}})]],
                device atomic_uint* targetAtomic [[buffer({{TargetAtomicIndex}})]],
                constant RepackArgs& args [[buffer({{ArgumentIndex}})]],
                uint3 gid [[thread_position_in_grid]])
            {
                if (gid.x >= args.rowWords || gid.y >= args.rows || gid.z >= args.slices)
                {
                    return;
                }

                uint sourceRow = args.sourceBase + gid.z * args.sourceSliceBytes + gid.y * args.sourceRowBytes;
                uint targetRow = args.destinationBase + gid.z * args.targetSliceBytes + gid.y * args.targetRowBytes;
                uint word = (targetRow >> 2) + gid.x;
                uint value = 0u;
                uint mask = 0u;
                for (uint byteIndex = 0u; byteIndex < 4u; ++byteIndex)
                {
                    uint targetByte = word * 4u + byteIndex;
                    if (targetByte < targetRow || targetByte >= targetRow + args.rowBytes)
                    {
                        continue;
                    }

                    uint sourceByte = sourceRow + (targetByte - targetRow);
                    uint index = sourceByte >> 2;
                    uint loaded = index < args.sourceWords ? source[index] : 0u;
                    value |= ((loaded >> ((sourceByte & 3u) * 8u)) & 0xffu) << (byteIndex * 8u);
                    mask |= 0xffu << (byteIndex * 8u);
                }

                if (mask == 0xffffffffu)
                {
                    target[word] = value;
                    return;
                }

                if (mask == 0u)
                {
                    return;
                }

                device atomic_uint* cell = targetAtomic + word;
                atomic_fetch_and_explicit(cell, ~mask, memory_order_relaxed);
                atomic_fetch_or_explicit(cell, value, memory_order_relaxed);
            }

            """);
        return source.ToString();
    }
}
