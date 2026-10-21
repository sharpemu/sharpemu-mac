// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Scheduling;

using System.Buffers.Binary;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The header data of one registered shader, and its continuation when two objects were joined.
public sealed record RegisteredShader(
    ulong CodeAddress,
    ulong HeaderAddress,
    uint CodeSizeBytes,
    uint ScratchDwords,
    ulong UserDataAddress,
    ulong InputSemanticsAddress,
    uint InputSemanticsCount,
    ulong ContinuationAddress,
    uint ContinuationSizeBytes)
{
    public uint ContinuationScratchDwords { get; init; }

    public bool IsFused => ContinuationAddress != 0;

    // The code ranges the identity covers, in program order.
    public (ulong Address, uint SizeBytes)[] CodeRanges =>
        IsFused ? [(CodeAddress, CodeSizeBytes), (ContinuationAddress, ContinuationSizeBytes)] : [(CodeAddress, CodeSizeBytes)];

    public const int MaxCodeRanges = 2;

    // CodeRanges without the array, for the per-draw path: the number of ranges written.
    public int CopyCodeRanges(Span<(ulong Address, uint SizeBytes)> destination)
    {
        destination[0] = (CodeAddress, CodeSizeBytes);
        if (!IsFused)
        {
            return 1;
        }

        destination[1] = (ContinuationAddress, ContinuationSizeBytes);
        return 2;
    }

    public uint TotalCodeSizeBytes => CodeSizeBytes + ContinuationSizeBytes;
}

// The two code objects the guest joined into one program; the entry ends in a jump to the continuation.
public readonly record struct FusedProgramParts(ulong ContinuationAddress, ulong ContinuationHeaderAddress);

// Resolves the header the guest registered for a code address; a missing header stops the draw.
public sealed class ShaderHeaderRegistry
{
    private const ulong UserDataOffset = 0x08;
    private const ulong InputSemanticsOffset = 0x30;
    private const ulong ShaderSizeOffset = 0x44;
    private const ulong InputSemanticsCountOffset = 0x50;
    private const ulong ScratchDwordsPerThreadOffset = 0x54;

    // Through the scratch size, the last field Require reads.
    private const int HeaderReadBytes = (int)ScratchDwordsPerThreadOffset + sizeof(ushort);

    private readonly CpuContext _context;
    private readonly Func<ulong, ulong> _headerOf;
    private readonly Func<ulong, FusedProgramParts?> _fusedPartsOf;

    public ShaderHeaderRegistry(CpuContext context, Func<ulong, ulong> headerOf, Func<ulong, FusedProgramParts?> fusedPartsOf)
    {
        _context = context;
        _headerOf = headerOf;
        _fusedPartsOf = fusedPartsOf;
    }

    internal (ulong Address, byte[] Code)[] ReadDiagnosticCode(ulong codeAddress)
    {
        var ranges = new List<(ulong Address, byte[] Code)>();
        void ReadRange(ulong address, ulong header)
        {
            Span<byte> sizeBytes = stackalloc byte[4];
            if (header == 0 || !_context.Memory.TryRead(checked(header + ShaderSizeOffset), sizeBytes))
                throw new InvalidDataException("The shader header is not readable.");
            var size = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(sizeBytes);
            if (size == 0 || size > 16 * 1024 * 1024)
                throw new InvalidDataException("The diagnostic shader size is outside the supported range.");
            var code = new byte[size];
            if (!_context.Memory.TryRead(address, code))
                throw new InvalidDataException("The shader code is not readable.");
            ranges.Add((address, code));
        }
        ReadRange(codeAddress, _headerOf(codeAddress));
        if (_fusedPartsOf(codeAddress) is { } parts)
            ReadRange(parts.ContinuationAddress, parts.ContinuationHeaderAddress);
        return ranges.ToArray();
    }

    public RegisteredShader Require(ulong codeAddress, string label)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.ShaderHeaderLookup);
        var headerAddress = _headerOf(codeAddress);
        if (headerAddress == 0)
        {
            throw GpuFatal.Fatal($"The shader has no registered header: label={label} shader=0x{codeAddress:X16}.");
        }

        // The fields the record needs, read as one block: every draw resolves its shaders again,
        // and each guest read takes the memory read lock that guest threads mapping and
        // protecting memory hold for writing, so five reads per stage were five waits for it
        // (10% of GTA V's GPU worker). A header the block cannot cover is read field by field.
        Span<byte> header = stackalloc byte[HeaderReadBytes];
        uint codeSize;
        ushort scratchDwords;
        ulong userDataAddress;
        ulong inputSemanticsAddress;
        uint inputSemanticsCount;
        if (_context.Memory.TryRead(headerAddress, header))
        {
            codeSize = ValidCodeSize(BinaryPrimitives.ReadUInt32LittleEndian(header[(int)ShaderSizeOffset..]), codeAddress, label);
            scratchDwords = BinaryPrimitives.ReadUInt16LittleEndian(header[(int)ScratchDwordsPerThreadOffset..]);
            userDataAddress = BinaryPrimitives.ReadUInt64LittleEndian(header[(int)UserDataOffset..]);
            inputSemanticsAddress = BinaryPrimitives.ReadUInt64LittleEndian(header[(int)InputSemanticsOffset..]);
            inputSemanticsCount = BinaryPrimitives.ReadUInt32LittleEndian(header[(int)InputSemanticsCountOffset..]);
        }
        else
        {
            codeSize = RequireCodeSize(headerAddress, codeAddress, label);
            if (!_context.TryReadUInt16(headerAddress + ScratchDwordsPerThreadOffset, out scratchDwords) ||
                !_context.TryReadUInt64(headerAddress + UserDataOffset, out userDataAddress) ||
                !_context.TryReadUInt64(headerAddress + InputSemanticsOffset, out inputSemanticsAddress) ||
                !_context.TryReadUInt32(headerAddress + InputSemanticsCountOffset, out inputSemanticsCount))
            {
                throw GpuFatal.Fatal($"The shader header is unreadable: label={label} shader=0x{codeAddress:X16} header=0x{headerAddress:X16}.");
            }
        }

        var continuationAddress = 0ul;
        var continuationSize = 0u;
        ushort continuationScratchDwords = 0;
        if (_fusedPartsOf(codeAddress) is { } parts)
        {
            continuationAddress = parts.ContinuationAddress;
            continuationSize = RequireCodeSize(parts.ContinuationHeaderAddress, parts.ContinuationAddress, label);
            if (!_context.TryReadUInt16(parts.ContinuationHeaderAddress + ScratchDwordsPerThreadOffset,
                    out continuationScratchDwords))
            {
                throw GpuFatal.Fatal($"The continuation shader scratch size is unreadable: label={label} shader=0x{parts.ContinuationAddress:X16}.");
            }
        }

        // Every draw resolves its shaders again, so an unchanged header hands back the record
        // it made last time instead of a new one. The header is still read and compared above
        // and here, so a guest that rewrites it gets a fresh record as before.
        lock (_recent)
        {
            if (_recent.TryGetValue(codeAddress, out var recent) &&
                recent.HeaderAddress == headerAddress && recent.CodeSizeBytes == codeSize &&
                recent.ScratchDwords == scratchDwords && recent.UserDataAddress == userDataAddress &&
                recent.InputSemanticsAddress == inputSemanticsAddress && recent.InputSemanticsCount == inputSemanticsCount &&
                recent.ContinuationAddress == continuationAddress && recent.ContinuationSizeBytes == continuationSize &&
                recent.ContinuationScratchDwords == continuationScratchDwords)
            {
                return recent;
            }

            var registered = new RegisteredShader(
                codeAddress,
                headerAddress,
                codeSize,
                scratchDwords,
                userDataAddress,
                inputSemanticsAddress,
                inputSemanticsCount,
                continuationAddress,
                continuationSize)
            {
                ContinuationScratchDwords = continuationScratchDwords,
            };

            // Bounded: dropping the records only costs rebuilding them.
            if (_recent.Count >= MaxRecent && !_recent.ContainsKey(codeAddress))
            {
                _recent.Clear();
            }

            _recent[codeAddress] = registered;
            return registered;
        }
    }

    private const int MaxRecent = 4096;
    private readonly Dictionary<ulong, RegisteredShader> _recent = new();

    // A bounded, nonfatal description for diagnostics of draws that remain rejected.
    internal bool TryGetDiagnosticCodeRanges(ulong address, out (ulong Address, uint SizeBytes)[] ranges)
    {
        ranges = [];
        bool ReadSize(ulong header, out uint size)
        {
            size = 0;
            return header != 0 && header <= ulong.MaxValue - ShaderSizeOffset - sizeof(uint) &&
                _context.TryReadUInt32(header + ShaderSizeOffset, out size) &&
                size != 0 && size % sizeof(uint) == 0 && size <= 1024 * 1024;
        }
        if (!ReadSize(_headerOf(address), out var size) || address > ulong.MaxValue - size) return false;
        if (_fusedPartsOf(address) is { } parts)
        {
            if (!ReadSize(parts.ContinuationHeaderAddress, out var continuationSize) ||
                parts.ContinuationAddress == 0 || parts.ContinuationAddress > ulong.MaxValue - continuationSize ||
                (ulong)size + continuationSize > 1024 * 1024) return false;
            ranges = [(address, size), (parts.ContinuationAddress, continuationSize)];
        }
        else ranges = [(address, size)];
        return true;
    }

    private uint RequireCodeSize(ulong headerAddress, ulong codeAddress, string label)
    {
        if (!_context.TryReadUInt32(headerAddress + ShaderSizeOffset, out var codeSize))
        {
            throw GpuFatal.Fatal($"The shader header is unreadable: label={label} shader=0x{codeAddress:X16} header=0x{headerAddress:X16}.");
        }

        return ValidCodeSize(codeSize, codeAddress, label);
    }

    private static uint ValidCodeSize(uint codeSize, ulong codeAddress, string label)
    {
        if (codeSize == 0 || codeSize % sizeof(uint) != 0)
        {
            throw GpuFatal.Fatal($"The shader header carries an invalid code size: label={label} shader=0x{codeAddress:X16} size=0x{codeSize:X8}.");
        }

        return codeSize;
    }
}
