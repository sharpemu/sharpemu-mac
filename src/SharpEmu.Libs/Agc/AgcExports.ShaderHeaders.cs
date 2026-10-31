// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Pipelines;
using System.Runtime.CompilerServices;

namespace SharpEmu.Libs.Agc;

public static partial class AgcExports
{
    private static readonly ConditionalWeakTable<ICpuMemory, IShaderFrontend> _shaderFrontends = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<ulong, IShaderFrontend> _shaderFrontendsByDomain = new();

    public static void SetShaderFrontend(ICpuMemory memory, IShaderFrontend frontend)
    {
        while (memory is ICpuMemoryWrapper wrapper) memory = wrapper.Inner;
        _shaderFrontends.AddOrUpdate(memory, frontend);
        _shaderFrontendsByDomain[frontend.CacheDomain] = frontend;
    }

    internal static IShaderFrontend? ShaderFrontendForDomain(ulong domain) =>
        _shaderFrontendsByDomain.TryGetValue(domain, out var frontend) ? frontend : null;

    internal static string ShaderFrontendStamp() => string.Join('|', _shaderFrontendsByDomain
        .OrderBy(static pair => pair.Key)
        .Select(static pair => $"{pair.Key:X16}:{pair.Value.GetType().Assembly.ManifestModule.ModuleVersionId:N}"));

    private static IShaderFrontend? ShaderFrontendOf(ICpuMemory memory)
    {
        while (memory is ICpuMemoryWrapper wrapper) memory = wrapper.Inner;
        return _shaderFrontends.TryGetValue(memory, out var frontend) ? frontend : null;
    }
    // This partial exposes the shader headers the create export registered.

    internal static ulong GetShaderHeaderAddress(ulong codeAddress)
    {
        SplitCrossingStats.Count(SplitCrossing.ShaderHeaderLookup);
        return _shaderHeadersByCode.TryGetValue(codeAddress, out var header) ? header : 0;
    }

    internal static ShaderHeaderRegistry CreateShaderHeaderRegistry(CpuContext context) =>
        new(
            context,
            GetShaderHeaderAddress,
            codeAddress => ShaderCompiler.Gen5ShaderTranslator.TryGetFusedProgramParts(context, codeAddress, out var continuation, out var continuationHeader)
                ? new FusedProgramParts(continuation, continuationHeader)
                : null,
            ShaderFrontendOf(context.Memory));

    // The two-process split keeps a copy of the table in the GPU process, where the command
    // streams look headers up ~100K times a second: the guest process reports every registration
    // (code, header) here, and the GPU process writes them into its own table.
    internal static Action<ulong, ulong>? ShaderHeaderObserver { get; set; }

    internal static KeyValuePair<ulong, ulong>[] SnapshotShaderHeaders() => _shaderHeadersByCode.ToArray();

    // A mirrored registration is a registration in the GPU process: programs kept across draws
    // (ShaderHeaderRegistry.Registrations) must look their headers up again.
    internal static void MirrorShaderHeader(ulong codeAddress, ulong headerAddress)
    {
        _shaderHeadersByCode[codeAddress] = headerAddress;
        Gpu.Pipelines.ShaderHeaderRegistry.NoteRegistration();
    }
}
