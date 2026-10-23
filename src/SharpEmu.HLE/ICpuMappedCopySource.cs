// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE;

/// <summary>
/// Optional ordinary-thread mapping-bound owned-copy contract. Participating
/// implementations must participate in mapping retirement. It is separate from content observation,
/// write exclusion, native permissions and write-fault recovery authority.
/// </summary>
public interface ICpuMappedCopySource
{
    /// <summary>
    /// Copies the source into caller-owned storage while resolving it against
    /// one fully covered mapping footprint. A producer must retain its actual
    /// copy storage and mapping resolution for the copy, and participate in all
    /// overlapping mutations before implementing this capability.
    /// </summary>
    /// <remarks>
    /// The source must be inside the footprint. Copied means every destination
    /// byte was copied and observation is non-null. Other results may leave
    /// destination partly changed; callers must discard it and keep the strict
    /// existing policy. No lazy commit, protection repair or backend fallback is
    /// permitted here. Wrappers must forward explicitly, preserving semantics.
    /// </remarks>
    CpuMappedCopyResult TryReadMappedCopy(
        ulong footprintAddress,
        ulong footprintByteCount,
        ulong sourceAddress,
        Span<byte> destination,
        out CpuMappedCopyObservation? observation);

    /// <summary>
    /// Conservative exact-range freshness veto, not a retained mapping lease.
    /// Reject foreign owners, retired/replaced identities and uncertain native
    /// outcomes. Validation must use bounded scalar/immutable state only, with
    /// no allocation, provider calls, query, callbacks or waiting under GPU gates.
    /// At most CpuMappedCopyObservation.MaximumSegments may be visited. Use
    /// authoritative immutable mutation-participation state; reconstructing
    /// equal mapping values from Query cannot establish currentness or defeat ABA.
    /// Content/trap checks remain separately required, even when this is true.
    /// </summary>
    bool IsMappedCopyObservationCurrent(CpuMappedCopyObservation observation);
}

public enum CpuMappedCopyResult
{
    Unavailable = 0,
    Rejected,
    Copied,
}

/// <summary>
/// Exact immutable mapping metadata, not a syscall/permission receipt. Identity
/// and incarnation are assigned by the participating owner; geometry/rights
/// must come from its actual recorded operations and guest ledger.
/// </summary>
public readonly record struct CpuMappingFootprintSegment(
    ulong MappingAddress,
    ulong MappingByteCount,
    ulong CoveredAddress,
    ulong CoveredByteCount,
    long MappingIncarnation,
    long BackingIdentity,
    ulong BackingLogicalByteCount,
    ulong MappingBackingOffset,
    ulong ProviderPageSize,
    GuestPageProtection GuestAccess);

internal interface ICpuNativeAliasObservation
{
    bool ExcludesNativeAliases { get; }
}

/// <summary>
/// Opaque owner-bound metadata from a completed owned copy or its exact confirmed
/// own-publication continuation. Constructed only by
/// the neutral participating owner, never accepted as caller-provided authority.
/// It retains no pointer, lock, access pin, native permission or content token.
/// Holding this CLR object does not prevent explicit backing/mapping disposal.
/// </summary>
public sealed class CpuMappedCopyObservation
{
    // Explicit finite validation footprint; unsupported larger requests decline.
    public const int MaximumSegments = 64;

    private readonly object _ownerIdentity;

    internal CpuMappedCopyObservation(object ownerIdentity, ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, ulong sourceByteCount,
        CpuMappingFootprintSegment[] segments)
    {
        ArgumentNullException.ThrowIfNull(ownerIdentity);
        ArgumentNullException.ThrowIfNull(segments);
        if (segments.Length is 0 or > MaximumSegments)
            throw new ArgumentOutOfRangeException(nameof(segments));
        if (!ValidRange(footprintAddress, footprintByteCount) ||
            !ValidRange(sourceAddress, sourceByteCount) || sourceByteCount > int.MaxValue ||
            sourceAddress < footprintAddress ||
            sourceAddress - footprintAddress > footprintByteCount ||
            sourceByteCount > footprintByteCount - (sourceAddress - footprintAddress))
            throw new ArgumentOutOfRangeException(nameof(footprintByteCount));

        var owned = (CpuMappingFootprintSegment[])segments.Clone();
        var cursor = footprintAddress;
        for (var index = 0; index < owned.Length; index++)
        {
            var segment = owned[index];
            if (!ValidRange(segment.MappingAddress, segment.MappingByteCount) ||
                !ValidRange(segment.CoveredAddress, segment.CoveredByteCount) ||
                segment.CoveredAddress != cursor || segment.CoveredAddress < segment.MappingAddress ||
                segment.CoveredAddress - segment.MappingAddress > segment.MappingByteCount ||
                segment.CoveredByteCount > segment.MappingByteCount - (segment.CoveredAddress - segment.MappingAddress) ||
                segment.CoveredByteCount > footprintAddress + footprintByteCount - cursor ||
                segment.MappingIncarnation <= 0 || segment.BackingIdentity <= 0 ||
                segment.MappingBackingOffset > segment.BackingLogicalByteCount ||
                segment.MappingByteCount > segment.BackingLogicalByteCount - segment.MappingBackingOffset ||
                segment.ProviderPageSize == 0 || (segment.ProviderPageSize & (segment.ProviderPageSize - 1)) != 0 ||
                segment.MappingAddress % segment.ProviderPageSize != 0 ||
                segment.MappingByteCount % segment.ProviderPageSize != 0 ||
                segment.MappingBackingOffset % segment.ProviderPageSize != 0 ||
                (segment.GuestAccess & GuestPageProtection.Read) == 0 ||
                (segment.GuestAccess & ~(GuestPageProtection.Read | GuestPageProtection.Write | GuestPageProtection.Execute)) != 0)
                throw new ArgumentException("Mapping segments must exactly cover the readable footprint and backing.", nameof(segments));
            // One canonical backing cannot acquire contradictory logical size
            // or process-provider geometry within this exact footprint.
            for (var previous = 0; previous < index; previous++)
            {
                if (owned[previous].BackingIdentity == segment.BackingIdentity &&
                    (owned[previous].BackingLogicalByteCount != segment.BackingLogicalByteCount ||
                     owned[previous].ProviderPageSize != segment.ProviderPageSize))
                    throw new ArgumentException("Canonical backing metadata must be consistent.", nameof(segments));
            }
            cursor += segment.CoveredByteCount;
        }
        if (cursor != footprintAddress + footprintByteCount)
            throw new ArgumentException("Mapping segments must exactly cover the footprint.", nameof(segments));

        _ownerIdentity = ownerIdentity;
        FootprintAddress = footprintAddress;
        FootprintByteCount = footprintByteCount;
        SourceAddress = sourceAddress;
        SourceByteCount = sourceByteCount;
        Segments = Array.AsReadOnly(owned);
    }

    // Optional provider-owned proof: no guest mapping outside this observed
    // interval can issue native stores to its physical bytes. A later mapping
    // effect invalidates the proof. Unknown producers conservatively decline.
    public bool ExcludesNativeAliases => _ownerIdentity is ICpuNativeAliasObservation native && native.ExcludesNativeAliases;

    public ulong FootprintAddress { get; }
    public ulong FootprintByteCount { get; }
    public ulong SourceAddress { get; }
    public ulong SourceByteCount { get; }
    public IReadOnlyList<CpuMappingFootprintSegment> Segments { get; }

    // Opaque owner state remains assembly-internal. No public metadata can
    // substitute for the provider's original mutation-participation state.
    internal object OwnerIdentity => _ownerIdentity;

    internal bool BelongsTo(object ownerIdentity) => ReferenceEquals(_ownerIdentity, ownerIdentity);

    private static bool ValidRange(ulong address, ulong byteCount) =>
        address != 0 && byteCount != 0 && byteCount <= ulong.MaxValue - address;
}
