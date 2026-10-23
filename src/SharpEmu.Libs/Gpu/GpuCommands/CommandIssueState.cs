// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;

namespace SharpEmu.Libs.Gpu.GpuCommands;

// Only immutable identities and a by-value copy of the packet's banks cross this seam.
// The banks live in the issue slot; consumers must clone anything they retain past it.
public readonly record struct IssueProgramIdentity(RegisteredShader? Shader, ReadyShaderIdentity? Identity);
public readonly record struct IssueProgramIdentities(IssueProgramIdentity Vertex, IssueProgramIdentity Pixel, IssueProgramIdentity Compute);

public interface ICommandIssueRenderer
{
    void BeginIssuedSubmission(int queueId, ulong submitId);
    void DrawIssued(int queueId, ulong submitId, RegisterBanks banks, in IssueProgramIdentities programs,
        bool indexed, in DrawIndexedArguments indexedArguments, in DrawAutoArguments autoArguments);
    void DispatchIssued(int queueId, ulong submitId, RegisterBanks banks, in IssueProgramIdentities programs,
        uint x, uint y, uint z, uint initiator, ulong indirect);
    void ResetIssuedQueue(int queueId);
}

// A owns this planner. Memory observations are admitted by the ordered host, which
// resolves every earlier writer intent first; no mutable Metal cache is queried here.
internal sealed class CommandIssueProgramPlanner
{
    private sealed class UnavailableIdentity(string message) : Exception(message);
    private static Exception Unavailable(string message) => new UnavailableIdentity(message);
    private readonly ShaderHeaderRegistry _headers;
    private readonly ShaderIdentityCache _identities;
    private readonly SharpEmu.HLE.ICpuMemory _memory;

    internal CommandIssueProgramPlanner(SharpEmu.HLE.ICpuMemory memory, Action<ulong, ulong> synchronize)
    {
        _memory = memory;
        _headers = Agc.AgcExports.CreateShaderHeaderRegistry(new SharpEmu.HLE.CpuContext(memory, SharpEmu.HLE.Generation.Gen5), Unavailable);
        var generations = memory is SharpEmu.HLE.ICpuMappedReadWatchSource
            ? SharpEmu.HLE.GpuMemory.GuestGpuMemoryHook.Current?.WriteGenerations : null;
        _headers.UseWriteGenerations(generations, synchronize);
        _identities = new ShaderIdentityCache(memory, synchronize, generations, Unavailable);
    }

    private IssueProgramIdentity Capture(ulong address, string label)
    {
        if (address == 0 || Agc.AgcExports.GetShaderHeaderAddress(address) == 0) return default;
        // A bound stage may be unused (zero work, metadata operations, depth-only
        // or an unavailable target). Only B's executor decides whether it is required.
        try
        {
            using var source = VideoOut.RenderPhaseProfile.MeasureDetail(VideoOut.RenderPhaseProfile.Phase.ProgramSourceRead);
            var registered = _headers.Require(address, label);
            if (!_memory.CanRead(registered.CodeAddress, registered.CodeSizeBytes) ||
                (registered.IsFused && !_memory.CanRead(registered.ContinuationAddress, registered.ContinuationSizeBytes))) return default;
            return new(registered, _identities.Get(registered, label));
        }
        catch (UnavailableIdentity) { return default; }
    }

    internal IssueProgramIdentities Capture(RegisterBanks banks, bool compute) => compute
        ? new(default, default, Capture(banks.Shader.Compute.Address, "compute"))
        : new(Capture(banks.Shader.Vertex.ExportAddress != 0 ? banks.Shader.Vertex.ExportAddress : banks.Shader.Vertex.LegacyVertexAddress, "vertex"),
            Capture(banks.Shader.Pixel.Address, "pixel"), default);
}

// Read only by B; its by-value result is published with the completed issue prefix.
public interface ICommandIssueReadTokenSource
{
    CommandReadToken IssueReadToken { get; }
}
