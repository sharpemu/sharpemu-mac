// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

// Reads one dword of guest memory; false when the address cannot be read.
public delegate void ResourceReadObserver(ulong address, ReadOnlySpan<byte> bytes);

public delegate bool GuestWordReader(ulong address, out uint word);

// What one draw supplies to materialise a plan: its user data, the shader base and
// the two memory readers. The clean reader refuses memory the GPU may still own.
public sealed class ResourceRuntimeInputs
{
    // Settable so the draw path can refill one instance per lookup; nothing keeps an
    // inputs object past the call it was passed to.
    public IReadOnlyList<uint> UserData { get; set; } = [];
    public ulong ShaderBase { get; set; }
    public GuestWordReader? ReadMemory { get; init; }
    public GuestWordReader? ReadCleanMemory { get; init; }
    public ComputeSelectorState? ComputeState { get; set; }
    public Action<uint, uint, ulong, ulong, ReadOnlyMemory<uint>>? TraceImageCandidate { get; init; }

    // Told true before the flattened table's words are evaluated and false after, so a reader
    // wrapper can tell the words only the table reads from those the descriptors depend on.
    public Action<bool>? TablePhase { get; init; }

    public ResidentGuestBytesReader? ReadResidentMemory { get; init; }

    // Compares recorded ranges in place on a hit; null leaves a hit reading every range back.
    public ResidentGuestBytesComparer? CompareResidentMemory { get; init; }

    public bool ReadsClean { get; init; }

    public IResourceDependencySource? DependencySource { get; set; }

    // A bypass has already certified these original bytes. Retry logs must retain
    // the same observations even though no reader/evaluator was called on the hit.
    public ResourceReadObserver? ObserveCachedRead { get; set; }

    // Whether recorded GPU work may still write the dword at this address, asked without
    // waiting. A data-only table word found pending is learned as a device load
    // (DeviceDataLoads); null leaves every data word flattened.
    public Func<ulong, bool>? DataWritePending { get; init; }

    // Whether reading the dword at this address would wait for recorded GPU work, asked
    // without waiting. A binding word found pending is left to the device by a re-plan
    // (DeviceBindingLoads); null reads every word.
    public Func<ulong, bool>? ResourceWordPending { get; init; }
    public Func<ulong, BindingWordWriter>? ResourceWordWriter { get; init; }

    public ResourceRuntimeInputs WithReader(GuestWordReader? reader, bool pendingLoads = true) => new()
    {
        UserData = UserData,
        ShaderBase = ShaderBase,
        ReadMemory = reader,
        ReadCleanMemory = ReadCleanMemory,
        ComputeState = ComputeState,
        TraceImageCandidate = TraceImageCandidate,
        TablePhase = TablePhase,
        ReadResidentMemory = ReadResidentMemory,
        CompareResidentMemory = CompareResidentMemory,
        ReadsClean = ReadsClean || ReferenceEquals(reader, ReadCleanMemory),
        DataWritePending = pendingLoads ? DataWritePending : null,
        ResourceWordPending = pendingLoads ? ResourceWordPending : null,
        ResourceWordWriter = pendingLoads ? ResourceWordWriter : null,
    };
}

public readonly record struct ComputeSelectorState(uint WaveSize, uint ThreadsX, uint ThreadsY, uint ThreadsZ,
    bool HasPartialWorkgroups, uint LocalDataShareDwords, int LocalInvocationIdComponents);
