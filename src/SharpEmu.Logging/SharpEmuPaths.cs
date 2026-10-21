// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Logging;

/// <summary>
/// Where SharpEmu keeps the data it writes: settings, logs, save data,
/// caches and dumps.
/// </summary>
/// <remarks>
/// <para>
/// SharpEmu is portable: everything it writes lives next to the executable,
/// under <see cref="AppContext.BaseDirectory"/>. That stays true on Windows
/// and Linux, and on macOS for a plain directory build.
/// </para>
/// <para>
/// It cannot stay true inside a macOS <c>.app</c> bundle. The bundle is a
/// single code-signed unit, so writing anything into it invalidates the
/// signature — Gatekeeper then refuses to launch it, and from
/// <c>/Applications</c> the write would fail outright because the directory is
/// not user-writable. When the running executable is inside a bundle, the data
/// root therefore moves to <c>~/Library/Application Support/SharpEmu</c>, which
/// is also where macOS expects an app's own data to live.
/// </para>
/// <para>
/// Two overrides, in priority order:
/// </para>
/// <list type="bullet">
/// <item><description>
/// <c>SHARPEMU_DATA_ROOT</c> — an explicit absolute path. The GUI passes its
/// resolved root to the emulator child process through this variable so parent
/// and child never disagree about where logs and save data go.
/// </description></item>
/// <item><description>
/// A <c>portable.txt</c> marker in the directory that *contains* the bundle.
/// Portable mode puts the data root beside <c>SharpEmu.app</c> rather than
/// inside it, which keeps the app relocatable on an external volume without
/// breaking its signature.
/// </description></item>
/// </list>
/// </remarks>
public static class SharpEmuPaths
{
    /// <summary>Environment variable that pins the data root explicitly.</summary>
    public const string DataRootVariable = "SHARPEMU_DATA_ROOT";

    /// <summary>Marker file that selects portable mode for a macOS bundle.</summary>
    public const string PortableMarkerFileName = "portable.txt";

    private static string PortableDirectory => TryGetMacBundleRoot(AppContext.BaseDirectory, out var bundle)
        ? Path.GetDirectoryName(bundle)! : AppContext.BaseDirectory;

    /// <summary>Marker file beside the executable that forces portable mode.</summary>
    public static string PortableMarkerPath =>
        Path.Combine(PortableDirectory, PortableMarkerFileName);

    /// <summary>True when the portable marker file exists beside the executable.</summary>
    public static bool IsPortableMarkerPresent() => File.Exists(PortableMarkerPath);

    /// <summary>
    /// True when the executable directory is writable, i.e. portable mode can
    /// still be enabled from the GUI.
    /// </summary>
    public static bool CanWriteExecutableDirectory() => IsWritable(PortableDirectory);

    /// <summary>
    /// Creates or removes the portable marker. Returns false when the marker
    /// could not be written or removed (read-only executable directory).
    /// </summary>
    public static bool SetPortableMarker(bool enabled)
    {
        try
        {
            if (enabled)
            {
                File.WriteAllText(
                    PortableMarkerPath,
                    "SharpEmu keeps its data (settings, saves, caches, logs) next to the executable while this file exists.\n");
            }
            else
            {
                File.Delete(PortableMarkerPath);
            }

            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private const string ApplicationSupportFolderName = "SharpEmu";

    /// <summary>
    /// The data root for this process. Equal to
    /// <see cref="AppContext.BaseDirectory"/> except inside a macOS app bundle.
    /// </summary>
    public static string DataRoot { get; } = ResolveDataRoot(AppContext.BaseDirectory);

    /// <summary>
    /// True when this process runs from inside a macOS <c>.app</c> bundle, so
    /// callers can skip operations that would rewrite the bundle (in-place
    /// self-update, for instance).
    /// </summary>
    public static bool IsMacAppBundle { get; } = TryGetMacBundleRoot(AppContext.BaseDirectory, out _);

    /// <summary>
    /// Resolves the data root for an executable that lives in
    /// <paramref name="baseDirectory"/>. The GUI uses this for the emulator it
    /// is about to launch, which may be a different install from its own.
    /// </summary>
    public static string ResolveDataRoot(string baseDirectory)
    {
        var configured = Environment.GetEnvironmentVariable(DataRootVariable);
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return Path.GetFullPath(configured);
        }

        var userDirectory = Environment.GetEnvironmentVariable("SHARPEMU_USER_DIR");
        if (!string.IsNullOrWhiteSpace(userDirectory))
            return Path.GetFullPath(userDirectory);

        if (!TryGetMacBundleRoot(baseDirectory, out var bundleRoot))
        {
            if (File.Exists(Path.Combine(baseDirectory, PortableMarkerFileName)) ||
                Directory.Exists(Path.Combine(baseDirectory, "user")) || IsWritable(baseDirectory))
                return baseDirectory;
            return OperatingSystem.IsLinux()
                ? Path.Combine(XdgDataHome(), ApplicationSupportFolderName)
                : OperatingSystem.IsMacOS()
                    ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                        "Library", "Application Support", ApplicationSupportFolderName)
                    : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), ApplicationSupportFolderName);
        }

        // Portable mode: data sits beside SharpEmu.app, not inside it.
        var besideBundle = Path.GetDirectoryName(bundleRoot);
        if (besideBundle is not null &&
            File.Exists(Path.Combine(besideBundle, PortableMarkerFileName)))
        {
            return besideBundle;
        }

        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "Library",
            "Application Support",
            ApplicationSupportFolderName);
    }

    private static string XdgDataHome()
    {
        var dataHome = Environment.GetEnvironmentVariable("XDG_DATA_HOME");
        if (!string.IsNullOrWhiteSpace(dataHome))
        {
            return dataHome;
        }

        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".local",
            "share");
    }

    private static bool IsWritable(string directory)
    {
        try
        {
            var probe = Path.Combine(directory, $".sharpemu-write-probe-{Environment.ProcessId}");
            using (File.Create(probe))
            {
            }

            File.Delete(probe);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>Combines <paramref name="parts"/> onto <see cref="DataRoot"/>.</summary>
    public static string DataPath(params string[] parts) =>
        parts.Length == 0 ? DataRoot : Path.Combine([DataRoot, .. parts]);

    /// <summary>
    /// <see cref="DataRoot"/> plus the conventional <c>user</c> subdirectory
    /// that holds per-installation state (save data, caches, per-game configs).
    /// </summary>
    public static string UserPath(params string[] parts) =>
        Path.Combine([DataRoot, "user", .. parts]);

    /// <summary>
    /// Finds the <c>*.app</c> directory enclosing <paramref name="baseDirectory"/>.
    /// </summary>
    /// <remarks>
    /// Matches on the <c>.app/Contents/</c> shape rather than just a
    /// <c>.app</c> ancestor: a user is free to name an ordinary folder
    /// "SharpEmu.app", and only the Contents layout means a real bundle whose
    /// signature we must not disturb.
    /// </remarks>
    private static bool TryGetMacBundleRoot(string baseDirectory, out string bundleRoot)
    {
        bundleRoot = string.Empty;
        if (!OperatingSystem.IsMacOS() || string.IsNullOrEmpty(baseDirectory))
        {
            return false;
        }

        var directory = new DirectoryInfo(Path.GetFullPath(baseDirectory));
        while (directory is not null)
        {
            if (directory.Name.Equals("Contents", StringComparison.Ordinal) &&
                directory.Parent is { } parent &&
                parent.Name.EndsWith(".app", StringComparison.OrdinalIgnoreCase))
            {
                bundleRoot = parent.FullName;
                return true;
            }

            directory = directory.Parent;
        }

        return false;
    }
}
