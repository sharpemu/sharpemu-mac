// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Scheduling;

/// <summary>A reported GPU failure; callers must let it terminate execution.</summary>
public sealed class GpuFatalException(string message, Exception? innerException = null)
    : InvalidOperationException(message, innerException);

/// <summary>
/// How the GPU engine gives up. Every backend and every layer of the shared
/// engine reports a fatal error the same way, so the helper carries no backend
/// dependency of its own; the Vulkan submission scheduler forwards its own
/// Fatal to this one.
/// </summary>
public static class GpuFatal
{
    internal static Action<string> OnFatal = message => Environment.FailFast(message);

    // Set on a thread that does work ahead of the render thread: there a fatal condition only
    // abandons that work, which the render thread then does itself and fails on if it is real.
    [ThreadStatic]
    internal static bool ThrowOnFatal;

    public sealed class AbandonedWorkException(string message) : Exception(message);

    /// <summary>
    /// Whether this thread is doing work ahead of need (<see cref="ThrowOnFatal"/>): a backend
    /// may abandon it rather than block, for example rather than wait for the GPU.
    /// </summary>
    public static bool IsSpeculative => ThrowOnFatal;

    /// <summary>
    /// Runs work done ahead of need on any thread with <see cref="ThrowOnFatal"/> set: a fatal
    /// condition inside it abandons the work instead of ending the process, and whoever needs the
    /// result does the work itself and fails on it if it is real.
    /// </summary>
    public static T Speculatively<T>(Func<T> work)
    {
        var previous = ThrowOnFatal;
        ThrowOnFatal = true;
        try
        {
            return work();
        }
        finally
        {
            ThrowOnFatal = previous;
        }
    }

    /// <summary>
    /// Run once on this thread before a fatal ends the process. <see cref="OnFatal"/> fails
    /// fast, so neither an exception filter nor a catch above the throw ever sees the fatal;
    /// the command stream worker registers what it has to print while it runs a slice.
    /// </summary>
    [ThreadStatic]
    internal static Action? BeforeFatal;

    /// <summary>
    /// Reports the fatal error, then returns an exception for the caller to
    /// throw so control stops there.
    /// </summary>
    /// <remarks>Backend contract.</remarks>
    public static Exception Fatal(string message) => Fatal(message, null);

    /// <summary>Reports a GPU failure without losing its original cause.</summary>
    public static Exception Fatal(string message, Exception? innerException)
    {
        if (ThrowOnFatal)
        {
            throw new AbandonedWorkException(message);
        }

        if (BeforeFatal is { } beforeFatal)
        {
            BeforeFatal = null;
            try
            {
                beforeFatal();
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[GPU][FATAL] The pre-fatal report failed: {exception.GetType().Name}: {exception.Message}");
            }
        }

        var diagnostic = innerException is null ? message : $"{message}{Environment.NewLine}{innerException}";
        Console.Error.WriteLine($"[GPU][FATAL] {diagnostic}");
        OnFatal(diagnostic);
        return new GpuFatalException(message, innerException);
    }
}
