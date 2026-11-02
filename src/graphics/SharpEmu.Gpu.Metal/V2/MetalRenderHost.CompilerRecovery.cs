// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// macOS turns a function into a GPU binary in a separate process, MTLCompilerService. When that
/// process goes away mid-compile (jetsam under memory pressure, or a crash on one shader), Metal
/// restarts it and tries again a few times, then fails the pipeline with
/// XPC_ERROR_CONNECTION_INTERRUPTED. AI LIMIT hit this in game on compute 0x78D69BF917444167
/// with about 5.9 GB of GPU allocations, and the fatal ended the process.
///
/// Now the build is tried again after a pause, a few times. If a compute pipeline still cannot be
/// built, it becomes an empty kernel: its dispatches write nothing, and the title keeps running
/// instead of aborting. A render pipeline only gets the retries. Set
/// SHARPEMU_METAL_V2_COMPILER_FAILURE_FATAL=1 to keep the old fatal.
/// </summary>
internal sealed partial class MetalRenderHost
{
    private static readonly int[] CompilerRetryDelaysMs = [250, 1000, 3000];

    private static readonly bool CompilerFailureFatal =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COMPILER_FAILURE_FATAL") == "1";

    private const string EmptyKernelSource = "#include <metal_stdlib>\nkernel void sharpemu_empty_kernel() {}\n";

    private readonly object _emptyKernelGate = new();
    private nint _emptyKernelFunction;
    private int _emptyKernelPipelines;

    /// <summary>Whether a Metal error says the compiler service went away, rather than that the
    /// shader is invalid.</summary>
    internal static bool IsCompilerServiceFailure(string description) =>
        description.Contains("XPC_ERROR_CONNECTION_INTERRUPTED", StringComparison.Ordinal) ||
        description.Contains("XPC_ERROR_CONNECTION_INVALID", StringComparison.Ordinal) ||
        description.Contains("interrupted connection", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Builds again after each pause while the last failure (<paramref name="error"/>) was the
    /// compiler service going away. Returns the state, or 0 with <paramref name="error"/> set to
    /// the last failure. Any other kind of failure returns 0 at once.
    /// </summary>
    private static nint RetryAfterCompilerServiceFailure(Func<(nint State, nint Error)> build, string what, ref nint error)
    {
        for (var attempt = 0; attempt < CompilerRetryDelaysMs.Length; attempt++)
        {
            if (error == 0 || !IsCompilerServiceFailure(MetalNative.DescribeError(error)))
            {
                return 0;
            }

            var delay = CompilerRetryDelaysMs[attempt];
            Console.Error.WriteLine(
                $"[METAL-V2][WARN] The Metal compiler service went away while building {what}; " +
                $"trying again in {delay} ms (attempt {attempt + 2} of {CompilerRetryDelaysMs.Length + 1}).");
            Thread.Sleep(delay);
            var (state, next) = build();
            if (state != 0)
            {
                Console.Error.WriteLine($"[METAL-V2][INFO] {what} built after the compiler service came back.");
                return state;
            }

            error = next;
        }

        return 0;
    }

    /// <summary>
    /// What a failed compute build turns into: the same build tried again while the compiler
    /// service keeps going away, then an empty kernel. Returns 0 when the failure is a different
    /// kind, or when <c>SHARPEMU_METAL_V2_COMPILER_FAILURE_FATAL=1</c> is set after the retries.
    /// </summary>
    private nint RecoverComputePipelineState(nint function, UInt128 archiveKey, ulong hash, ref nint error)
    {
        var what = $"compute pipeline 0x{hash:X16}";
        var state = RetryAfterCompilerServiceFailure(
            () =>
            {
                nint next = 0;
                var built = NewComputePipelineState(function, archiveKey, ref next);
                return (built, next);
            },
            what,
            ref error);
        if (state != 0 || CompilerFailureFatal || error == 0 || !IsCompilerServiceFailure(MetalNative.DescribeError(error)))
        {
            return state;
        }

        state = NewEmptyComputePipelineState();
        if (state != 0)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][ERROR] {what} could not be built ({MetalNative.DescribeError(error)}); " +
                $"its dispatches run an empty kernel and write nothing (empty pipelines so far: " +
                $"{Interlocked.Increment(ref _emptyKernelPipelines)}). " +
                "Set SHARPEMU_METAL_V2_COMPILER_FAILURE_FATAL=1 to stop here instead.");
        }

        return state;
    }

    // A new, owned pipeline state for an empty kernel. The function is compiled once per host and
    // kept, but each caller gets its own state, so every pipeline entry releases only its own.
    private nint NewEmptyComputePipelineState()
    {
        nint function;
        lock (_emptyKernelGate)
        {
            if (_emptyKernelFunction == 0)
            {
                var options = MetalNative.Send(
                    MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")),
                    MetalNative.Selector("init"));
                nint libraryError = 0;
                var library = MetalNative.Send(
                    _device,
                    MetalNative.Selector("newLibraryWithSource:options:error:"),
                    MetalNative.NsString(EmptyKernelSource),
                    options,
                    ref libraryError);
                MetalNative.SendVoid(options, SelRelease);
                if (library == 0)
                {
                    return 0;
                }

                _emptyKernelFunction = MetalNative.Send(
                    library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("sharpemu_empty_kernel"));
                MetalNative.SendVoid(library, SelRelease);
            }

            function = _emptyKernelFunction;
        }

        if (function == 0)
        {
            return 0;
        }

        nint error = 0;
        return MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
    }
}
