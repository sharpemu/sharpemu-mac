// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Xunit;

// ArchitectureRules, not Architecture: the latter would shadow
// System.Runtime.InteropServices.Architecture for every test under
// SharpEmu.Libs.Tests, where `Architecture.X64` is used throughout.
namespace SharpEmu.Libs.Tests.ArchitectureRules;

/// <summary>
/// Enforces the per-OS host package boundaries (W06-W09): a macOS build must
/// not carry the Windows or Linux host implementations, and vice versa.
/// </summary>
/// <remarks>
/// Deliberately reads the project files and sources from disk rather than
/// reflecting over loaded assemblies: only the host package for the OS running
/// the tests is referenced, so reflection could never observe the other two.
/// Keeping the check source-based also means a violation fails on every OS,
/// including in a cross-build-only CI leg.
/// </remarks>
public sealed class HostPackageBoundaryTests
{
    private const string MacOS = "src/platforms/macos/SharpEmu.Host.MacOS";
    private const string Linux = "src/platforms/linux/SharpEmu.Host.Linux";
    private const string Windows = "src/platforms/windows/SharpEmu.Host.Windows";

    [Theory]
    [InlineData("src/SharpEmu.HLE/SharpEmu.HLE.csproj")]
    [InlineData("src/SharpEmu.Core/SharpEmu.Core.csproj")]
    [InlineData("src/SharpEmu.Libs/SharpEmu.Libs.csproj")]
    public void SharedEngineProjectsDoNotReferenceAHostPackage(string project)
    {
        var references = ProjectReferenceNames(project);
        Assert.DoesNotContain(references, name => name.StartsWith("SharpEmu.Host.", StringComparison.Ordinal));
    }

    [Fact]
    public void HleDoesNotReferenceSdl()
    {
        // The SDL audio adapter lives in SharpEmu.Adapters.Sdl; HLE owns the
        // contract only (W06).
        Assert.DoesNotContain("ppy.SDL3-CS", PackageReferenceNames("src/SharpEmu.HLE/SharpEmu.HLE.csproj"));
        Assert.DoesNotContain(
            "SharpEmu.Adapters.Sdl",
            ProjectReferenceNames("src/SharpEmu.HLE/SharpEmu.HLE.csproj"));
    }

    [Theory]
    [InlineData(MacOS, new[] { "SharpEmu.Host.Linux", "SharpEmu.Host.Windows" })]
    [InlineData(Linux, new[] { "SharpEmu.Host.MacOS", "SharpEmu.Host.Windows" })]
    [InlineData(Windows, new[] { "SharpEmu.Host.MacOS", "SharpEmu.Host.Linux" })]
    public void HostPackagesDoNotReferenceASiblingHostPackage(string package, string[] forbidden)
    {
        var references = ProjectReferenceNames(ProjectFile(package));
        foreach (var name in forbidden)
        {
            Assert.DoesNotContain(name, references);
        }
    }

    [Theory]
    [InlineData(MacOS)]
    [InlineData(Linux)]
    [InlineData(Windows)]
    public void HostPackagesDependInwardOnly(string package)
    {
        // Host packages implement the HLE host contracts; they never reach up
        // into the engine or the application layers, and they take optional
        // adapters by injection rather than referencing them. SharpEmu.Libs is
        // the reason the SDL adapter in particular must stay out: it owns the
        // window and input too and references the guest library layer.
        var references = ProjectReferenceNames(ProjectFile(package));
        Assert.DoesNotContain("SharpEmu.Core", references);
        Assert.DoesNotContain("SharpEmu.Libs", references);
        Assert.DoesNotContain("SharpEmu.GUI", references);
        Assert.DoesNotContain("SharpEmu.Adapters.Sdl", references);
    }

    [Fact]
    public void MacOsHostDoesNotReferenceSdl()
    {
        // macOS audio is a native CoreAudio output. SDL must not reach the
        // macOS host closure, by package or by adapter project.
        var project = ProjectFile(MacOS);
        Assert.DoesNotContain("ppy.SDL3-CS", PackageReferenceNames(project));
        Assert.DoesNotContain("SharpEmu.Adapters.Sdl", ProjectReferenceNames(project));
        // Also check the sources, so a linked shared file cannot smuggle the
        // dependency in. Matched on the identifiers that would constitute a
        // dependency rather than on "SDL" anywhere: the macOS audio
        // placeholder's own diagnostic has to be able to say why there is no
        // SDL output here.
        string[] forbidden = ["using SDL", "SDL.SDL3", "SdlHostAudio", "SharpEmu.Adapters.Sdl"];
        foreach (var source in CompiledSources(MacOS))
        {
            var code = StripComments(source.Text);
            foreach (var token in forbidden)
            {
                Assert.False(
                    code.Contains(token, StringComparison.Ordinal),
                    $"{source.Path} uses \"{token}\"; the macOS host must not depend on SDL.");
            }
        }
    }

    [Theory]
    // Each host package may test for its own OS (the entry point rejects a
    // process it cannot run the guest in) but never branches on a sibling's.
    [InlineData(MacOS, new[] { "OperatingSystem.IsWindows", "OperatingSystem.IsLinux" })]
    [InlineData(Linux, new[] { "OperatingSystem.IsWindows", "OperatingSystem.IsMacOS" })]
    [InlineData(Windows, new[] { "OperatingSystem.IsLinux", "OperatingSystem.IsMacOS" })]
    public void HostPackagesDoNotBranchOnASiblingOs(string package, string[] forbidden)
    {
        foreach (var source in CompiledSources(package))
        {
            var code = StripComments(source.Text);
            foreach (var check in forbidden)
            {
                Assert.False(
                    code.Contains(check, StringComparison.Ordinal),
                    $"{source.Path} branches on {check}; each host package keeps only its own OS's path.");
            }
        }
    }

    [Theory]
    // Native APIs and flag constants are owned by the package whose OS
    // provides them. Checked against the source with comments stripped: the
    // Win64 calling convention these POSIX stubs reimplement is necessarily
    // described in terms of kernel32, and that prose is not a dependency.
    [InlineData(MacOS, new[] { "kernel32", "winmm", "memfd_create", "MAP_FIXED_NOREPLACE", "sem_init", "gettid", "libc.so" })]
    [InlineData(Linux, new[] { "kernel32", "winmm", "libSystem", "mach_vm", "dispatch_", "shm_open" })]
    [InlineData(Windows, new[] { "libSystem", "mmap", "mprotect", "pthread_" })]
    public void HostPackagesDoNotUseASiblingOssNativeApi(string package, string[] forbidden)
    {
        foreach (var source in CompiledSources(package))
        {
            var code = StripComments(source.Text);
            foreach (var check in forbidden)
            {
                Assert.False(
                    code.Contains(check, StringComparison.Ordinal),
                    $"{source.Path} uses \"{check}\", which belongs to another OS's host package.");
            }
        }
    }

    [Theory]
    [InlineData(MacOS, new[]
    {
        "libc",
        "libSystem.B.dylib",
        "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox",
        // The DualSense's own audio: its CoreAudio device (USB) and its HID output reports (Bluetooth).
        "/System/Library/Frameworks/CoreAudio.framework/CoreAudio",
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
        "/System/Library/Frameworks/IOKit.framework/IOKit",
    })]
    [InlineData(Linux, new[] { "libc", "libasound.so.2" })]
    [InlineData(Windows, new[] { "kernel32.dll", "kernelbase.dll", "winmm.dll" })]
    public void HostPackagesOnlyImportTheirOwnOssLibraries(string package, string[] allowed)
    {
        foreach (var source in CompiledSources(package))
        {
            foreach (var library in NativeImportLibraries(source.Text))
            {
                Assert.Contains(library, allowed);
            }
        }
    }

    [Fact]
    public void CliNeverReferencesAHostPackageUnconditionally()
    {
        // The host package is selected from the target RuntimeIdentifier, so
        // every host ProjectReference must sit in a conditional ItemGroup.
        var document = XDocument.Load(RepoPath("src/SharpEmu.CLI/SharpEmu.CLI.csproj"));
        var hostReferences = document
            .Descendants("ProjectReference")
            .Where(element => ReferenceName(element).StartsWith("SharpEmu.Host.", StringComparison.Ordinal))
            .ToList();

        Assert.NotEmpty(hostReferences);
        foreach (var reference in hostReferences)
        {
            var condition = reference.Attribute("Condition")?.Value
                ?? reference.Parent?.Attribute("Condition")?.Value;
            Assert.False(
                string.IsNullOrWhiteSpace(condition),
                $"{ReferenceName(reference)} is referenced unconditionally; an osx-* build would carry it.");
        }
    }

    private static string ProjectFile(string packageDirectory) =>
        $"{packageDirectory}/{Path.GetFileName(packageDirectory)}.csproj";

    private static IReadOnlyList<string> ProjectReferenceNames(string projectPath) =>
        ReferenceNames(projectPath, "ProjectReference");

    private static IReadOnlyList<string> PackageReferenceNames(string projectPath) =>
        ReferenceNames(projectPath, "PackageReference");

    private static IReadOnlyList<string> ReferenceNames(string projectPath, string elementName) =>
        XDocument.Load(RepoPath(projectPath))
            .Descendants(elementName)
            .Select(ReferenceName)
            .Where(name => name.Length != 0)
            .ToList();

    private static string ReferenceName(XElement element)
    {
        var include = element.Attribute("Include")?.Value ?? string.Empty;
        if (!include.EndsWith(".csproj", StringComparison.OrdinalIgnoreCase))
        {
            return include;
        }

        // Project files spell paths with backslashes, which are not separators
        // on Unix, so normalize before taking the file name.
        return Path.GetFileNameWithoutExtension(include.Replace('\\', '/'));
    }

    /// <summary>
    /// Every source compiled into the package: the files under its directory
    /// plus any linked shared source (src/platforms/shared), which is just as
    /// much part of the assembly.
    /// </summary>
    private static IReadOnlyList<(string Path, string Text)> CompiledSources(string packageDirectory)
    {
        var root = RepoPath(packageDirectory);
        var paths = Directory
            .EnumerateFiles(root, "*.cs", SearchOption.AllDirectories)
            .Where(path => !path.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .ToList();

        foreach (var include in XDocument.Load(RepoPath(ProjectFile(packageDirectory)))
                     .Descendants("Compile")
                     .Select(element => element.Attribute("Include")?.Value)
                     .Where(value => !string.IsNullOrWhiteSpace(value)))
        {
            paths.Add(Path.GetFullPath(Path.Combine(root, include!.Replace('\\', Path.DirectorySeparatorChar))));
        }

        var sources = paths
            .Distinct()
            .Select(path => (Path: Path.GetRelativePath(RepoRoot, path), Text: File.ReadAllText(path)))
            .ToList();

        Assert.NotEmpty(sources);
        return sources;
    }

    /// <summary>
    /// Library names from DllImport/LibraryImport attributes. Matching the
    /// attribute rather than the raw text keeps prose that legitimately names
    /// another OS's API -- the Win64 calling convention these POSIX stubs
    /// reimplement is described in terms of kernel32 -- from failing the check.
    /// An attribute whose argument is a const (the framework paths do not fit
    /// on the attribute) is resolved through that const, so wrapping a library
    /// name in one does not slip past the check.
    /// </summary>
    private static IEnumerable<string> NativeImportLibraries(string source)
    {
        var constants = Regex
            .Matches(source, @"const\s+string\s+(\w+)\s*=\s*""([^""]*)""")
            .ToDictionary(match => match.Groups[1].Value, match => match.Groups[2].Value);

        foreach (var match in Regex.Matches(
                     source,
                     @"\[\s*(?:DllImport|LibraryImport)\s*\(\s*(?:""([^""]*)""|(\w+))").Cast<Match>())
        {
            if (match.Groups[1].Success)
            {
                yield return match.Groups[1].Value;
                continue;
            }

            var name = match.Groups[2].Value;
            Assert.True(
                constants.TryGetValue(name, out var value),
                $"DllImport({name}) does not resolve to a string constant in the same file, " +
                "so the host package's native imports cannot be checked.");
            yield return value!;
        }
    }

    private static string StripComments(string source)
    {
        source = Regex.Replace(source, @"/\*.*?\*/", " ", RegexOptions.Singleline);
        return Regex.Replace(source, @"//[^\n]*", " ");
    }

    private static string RepoPath(string relativePath) =>
        Path.Combine(RepoRoot, relativePath.Replace('/', Path.DirectorySeparatorChar));

    // Located by walking up for the solution file, from the build output first
    // (always inside the repository, under artifacts/) and from this file's
    // compile-time path as a fallback: a deterministic-source-path build
    // rewrites CallerFilePath, so it cannot be the only source of truth.
    private static readonly string RepoRoot =
        FindRepoRoot(AppContext.BaseDirectory) ??
        FindRepoRoot(Path.GetDirectoryName(ThisFile())) ??
        throw new InvalidOperationException(
            "Could not locate the repository root (no SharpEmu.slnx in any parent directory).");

    private static string? FindRepoRoot(string? start)
    {
        for (var directory = start is null ? null : new DirectoryInfo(start);
             directory is not null;
             directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "SharpEmu.slnx")))
            {
                return directory.FullName;
            }
        }

        return null;
    }

    private static string ThisFile([CallerFilePath] string path = "") => path;
}
