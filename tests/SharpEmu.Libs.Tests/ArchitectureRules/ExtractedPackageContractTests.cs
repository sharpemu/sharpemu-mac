// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Agc;
using Xunit;

namespace SharpEmu.Libs.Tests.ArchitectureRules;

/// <summary>Prevents package extraction from being bypassed through friend access or shared namespaces.</summary>
public sealed class ExtractedPackageContractTests
{
    public static TheoryData<string, string> Packages => new()
    {
        { "src/graphics/SharpEmu.Gpu.Metal", "SharpEmu.Gpu.Metal" },
        { "src/graphics/SharpEmu.Gpu.Vulkan", "SharpEmu.Gpu.Vulkan" },
        { "src/adapters/SharpEmu.Adapters.Sdl", "SharpEmu.Adapters.Sdl" },
        { "src/platforms/macos/SharpEmu.Platform.MacOS", "SharpEmu.Platform.MacOS" },
        { "src/platforms/macos/SharpEmu.Host.MacOS", "SharpEmu.Host.MacOS" },
        { "src/platforms/linux/SharpEmu.Host.Linux", "SharpEmu.Host.Linux" },
        { "src/platforms/windows/SharpEmu.Host.Windows", "SharpEmu.Host.Windows" },
        { "src/SharpEmu.GpuRemote", "SharpEmu.GpuRemote" },
        { "src/SharpEmu.GpuHost", "SharpEmu.GpuHost" },
    };

    [Theory]
    [MemberData(nameof(Packages))]
    public void PackageSourcesUseTheirOwningAssemblyNamespace(string directory, string assembly)
    {
        // Linked PosixStubEmitter is explicitly shared source, not an extracted
        // implementation. Files physically owned by a package must use its namespace.
        var files = Directory.EnumerateFiles(RepoPath(directory), "*.cs", SearchOption.AllDirectories)
            .Where(path => !path.Contains("/obj/", StringComparison.Ordinal)).ToArray();
        Assert.NotEmpty(files);
        foreach (var file in files)
        {
            var declarations = Regex.Matches(File.ReadAllText(file), @"(?m)^namespace\s+([\w.]+)\s*[;{]");
            if (declarations.Count == 0)
            {
                Assert.Equal("AssemblyInfo.cs", Path.GetFileName(file));
                var code = Regex.Replace(File.ReadAllText(file), @"//[^\n]*", " ");
                Assert.DoesNotMatch(@"\b(class|interface|enum|struct|record)\s+\w", code);
                continue;
            }
            foreach (Match declaration in declarations)
            {
                var actual = declaration.Groups[1].Value;
                Assert.True(actual == assembly || actual.StartsWith(assembly + ".", StringComparison.Ordinal),
                    $"{Path.GetRelativePath(RepoRoot, file)} declares {actual}; it belongs to {assembly}.");
            }
        }
    }

    [Theory]
    [InlineData("src/SharpEmu.HLE/SharpEmu.HLE.csproj")]
    [InlineData("src/SharpEmu.Libs/SharpEmu.Libs.csproj")]
    public void SharedAssemblyFriendsAreLimitedToExistingCoreAndTests(string project)
    {
        // Libs also befriends upstream's GNM mode and the Vulkan backend, which reach
        // into its GPU internals since the upstream merge.
        string[] allowed = project.Contains("HLE", StringComparison.Ordinal)
            ? ["SharpEmu.Core", "SharpEmu.Libs.Tests"]
            : ["SharpEmu.Core", "SharpEmu.Libs.Tests", "SharpEmu.GnmMode", "SharpEmu.Gpu.Vulkan"];
        foreach (var friend in XDocument.Load(RepoPath(project)).Descendants("InternalsVisibleTo"))
            Assert.Contains(friend.Attribute("Include")!.Value, allowed);

        // Check compiled assembly metadata as well: a handwritten assembly
        // attribute must not evade the project-file rule.
        var assembly = project.Contains("HLE", StringComparison.Ordinal)
            ? typeof(HostBackingObject).Assembly : typeof(AgcExports).Assembly;
        foreach (var friend in assembly.GetCustomAttributes<InternalsVisibleToAttribute>())
            Assert.Contains(friend.AssemblyName, allowed);
    }

    [Theory]
    [MemberData(nameof(Packages))]
    public void ExtractedPackagesGrantInternalAccessOnlyToTestsAndApprovedBenchmarks(string directory, string assembly)
    {
        var project = XDocument.Load(RepoPath($"{directory}/{assembly}.csproj"));
        // Keep friend access limited to tests and named offline benchmarks.
        // PerfAudit exercises Metal's internal resource paths; it does not need
        // friend access to the other extracted packages.
        var allowedFriends = new List<string>
        {
            "SharpEmu.Libs.Tests", $"{assembly}.Tests", $"{assembly}.LeakTests",
            "SharpEmu.Tools.ShaderCompileBench",
        };
        if (assembly == "SharpEmu.Gpu.Metal")
            allowedFriends.Add("SharpEmu.Tools.PerfAudit");

        foreach (var friend in project.Descendants("InternalsVisibleTo"))
            Assert.Contains(friend.Attribute("Include")!.Value, allowedFriends);
    }

    [Theory]
    [InlineData("SharpEmu.Libs.Gpu.Pipelines.ShaderProgramCache")]
    [InlineData("SharpEmu.Libs.Media.HostMovieBridge")]
    [InlineData("SharpEmu.Libs.Media.HostMovieYuv420")]
    [InlineData("SharpEmu.Libs.Agc.AgcExports+HeadlessCommandStream")]
    public void BackendServicesDoNotPublishTheirStatefulImplementations(string typeName)
    {
        var type = typeof(AgcExports).Assembly.GetType(typeName, throwOnError: true)!;
        Assert.False(type.IsVisible);
    }

    [Fact]
    public void BackingDisposalUsesTheMappingGateAndReleasesExactlyOnce()
    {
        var releases = 0;
        using var backing = new HostBackingObject(0x1000, 0x1000, current =>
        {
            Assert.True(Monitor.IsEntered(current.Gate));
            Assert.True(current.IsDisposed);
            releases++;
        }) { Descriptor = 7 };
        backing.Dispose();
        backing.Dispose();
        Assert.Equal(1, releases);
        Assert.Equal(7, backing.Descriptor);
        Assert.Empty(typeof(HostBackingObject).GetFields(BindingFlags.Public | BindingFlags.Instance));
    }

    private static string RepoPath(string path) => Path.Combine(RepoRoot, path.Replace('/', Path.DirectorySeparatorChar));
    private static readonly string RepoRoot = FindRoot(AppContext.BaseDirectory);

    private static string FindRoot(string start)
    {
        for (var directory = new DirectoryInfo(start); directory is not null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "SharpEmu.slnx"))) return directory.FullName;
        throw new InvalidOperationException("Could not locate SharpEmu.slnx above the test output.");
    }
}
