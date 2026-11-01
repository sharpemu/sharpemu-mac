// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.GpuRemote;
using SharpEmu.Logging;

namespace SharpEmu.GpuSplit;

/// <summary>
/// Gives the GPU host, the process that owns the game window, the game's name in the Dock and
/// the menu bar. Both name an unbundled process after its executable file (resolving
/// symlinks), and neither NSProcessInfo.processName nor LaunchServices' display name changes
/// the Dock tile, so the host is started through a copy of its small native apphost named
/// after the game. The copy's embedded app path points back at the real host, whose
/// directory keeps the runtime, so nothing but the ~100 KB apphost is duplicated.
/// </summary>
/// <remarks>
/// Copies live in <c>user/launchers/&lt;title id&gt;/</c>, are rewritten only when the real
/// apphost changes, and are ad-hoc signed (arm64 code must carry a signature). Any failure
/// falls back to the real host. <c>SHARPEMU_GPU_HOST_NAMED=0</c> turns it off.
/// </remarks>
internal static class NamedGpuHost
{
    public const string DisableEnvironment = "SHARPEMU_GPU_HOST_NAMED";

    private static readonly byte[] EmbeddedAppName = Encoding.UTF8.GetBytes(GpuProcessHost.ExecutableName + ".dll\0");

    // The apphost reserves 1024 bytes for the app path, NUL included.
    private const int EmbeddedPathCapacity = 1024;

    public static string? TryPrepare(string? title, string titleId)
    {
        if (!OperatingSystem.IsMacOS() ||
            Environment.GetEnvironmentVariable(DisableEnvironment) == "0" ||
            !string.IsNullOrEmpty(Environment.GetEnvironmentVariable(GpuProcessHost.ExecutablePathEnvironment)) ||
            SanitizeName(title) is not { } name ||
            GpuProcessHost.FindExecutable() is not { } host)
        {
            return null;
        }

        try
        {
            var directory = SharpEmuPaths.UserPath("launchers", SanitizeName(titleId) ?? "UNKNOWN");
            var target = Path.Combine(directory, name);
            var source = new FileInfo(host);
            var stamp = $"{source.FullName}|{source.Length}|{source.LastWriteTimeUtc.Ticks}";
            var stampPath = Path.Combine(directory, ".stamp");
            if (File.Exists(target) && File.Exists(stampPath) && File.ReadAllText(stampPath) == stamp)
            {
                return target;
            }

            Directory.CreateDirectory(directory);
            foreach (var stale in Directory.EnumerateFiles(directory))
            {
                File.Delete(stale);
            }

            // The apphost resolves its own path with realpath(3) before applying the relative
            // app path, so both ends must be canonical (/tmp is /private/tmp, for one).
            var appPath = Path.GetRelativePath(
                RealPath(directory),
                Path.Combine(RealPath(source.DirectoryName!), GpuProcessHost.ExecutableName + ".dll"));
            var image = File.ReadAllBytes(host);
            if (!TryRetarget(image, appPath))
            {
                return null;
            }

            File.WriteAllBytes(target, image);
            File.SetUnixFileMode(target, File.GetUnixFileMode(host));
            if (!AdHocSign(target))
            {
                File.Delete(target);
                return null;
            }

            File.WriteAllText(stampPath, stamp);
            return target;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[GPU-SPLIT][WARN] The GPU host keeps its own name: {exception.Message}");
            return null;
        }
    }

    // Rewrites the apphost's embedded app path (relative to the apphost's directory).
    internal static bool TryRetarget(byte[] image, string appPath)
    {
        var at = image.AsSpan().IndexOf(EmbeddedAppName);
        var replacement = Encoding.UTF8.GetBytes(appPath + "\0");
        if (at < 0 || replacement.Length > EmbeddedPathCapacity ||
            image.AsSpan(at + EmbeddedAppName.Length).IndexOf(EmbeddedAppName) >= 0)
        {
            return false;
        }

        image.AsSpan(at, EmbeddedPathCapacity).Clear();
        replacement.CopyTo(image, at);
        return true;
    }

    // Finder-safe file name: no path separators, colons or control characters.
    internal static string? SanitizeName(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        var builder = new StringBuilder(value.Length);
        foreach (var character in value.Trim())
        {
            builder.Append(character is '/' or ':' || char.IsControl(character) ? ' ' : character);
        }

        var name = builder.ToString().Trim().TrimStart('.');
        if (name.Length > 80)
        {
            name = name[..80].TrimEnd();
        }

        return name.Length == 0 ? null : name;
    }

    private static string RealPath(string path)
    {
        var resolved = realpath(path, 0);
        if (resolved == 0)
        {
            return Path.GetFullPath(path);
        }

        try
        {
            return Marshal.PtrToStringUTF8(resolved) ?? Path.GetFullPath(path);
        }
        finally
        {
            free(resolved);
        }
    }

    [DllImport("libc", SetLastError = true)]
    private static extern nint realpath(string path, nint resolvedPath);

    [DllImport("libc")]
    private static extern void free(nint pointer);

    private static bool AdHocSign(string path)
    {
        using var process = Process.Start(new ProcessStartInfo("/usr/bin/codesign", ["--force", "--sign", "-", path])
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        });
        if (process is null)
        {
            return false;
        }

        process.StandardOutput.ReadToEnd();
        process.StandardError.ReadToEnd();
        return process.WaitForExit(10_000) && process.ExitCode == 0;
    }
}
