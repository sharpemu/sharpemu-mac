// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.Core.Loader;
using SharpEmu.HLE.FileSystem;
using Xunit;
using Outcome = SharpEmu.Core.Loader.DemonsSoulsConvars.Outcome;

namespace SharpEmu.Libs.Tests.Loader;

[Collection("App0FileOverlay")]
public sealed class DemonsSoulsConvarsTests
{
    private const string Root = "/games/PPSA01342-app0";
    private const string Line = "+doIndexBufferCulling=false";

    private static readonly byte[] Bom = [0xEF, 0xBB, 0xBF];

    private static string Merged(string original) =>
        Encoding.Latin1.GetString(DemonsSoulsConvars.Merge(Encoding.Latin1.GetBytes(original))!);

    // Runs the decision on a file of the given content (null: absent) and returns what was served.
    private static (Outcome Outcome, string? Served, int Reads) Run(
        string? content,
        string? titleId = DemonsSoulsConvars.TitleId,
        bool keepCulling = false,
        bool replaceable = true)
    {
        string? served = null;
        var reads = 0;
        var outcome = DemonsSoulsConvars.Apply(
            titleId,
            Root,
            keepCulling,
            path =>
            {
                reads++;
                Assert.Equal(DemonsSoulsConvars.HostPath(Root), path);
                return content is null ? null : Encoding.Latin1.GetBytes(content);
            },
            (path, bytes) =>
            {
                Assert.Equal(DemonsSoulsConvars.HostPath(Root), path);
                if (!replaceable)
                {
                    return false;
                }

                served = Encoding.Latin1.GetString(bytes);
                return true;
            });
        return (outcome, served, reads);
    }

    [Fact]
    public void ThePresentFileKeepsItsCommandsAndGainsTheConvarLine()
    {
        const string shipped = "+shaderTypeToLoad=Shipping\n+TargetFramesPerSecond=60\n+StartupScript=$/scripts/cp11/cp11main.script\n";

        var (outcome, served, _) = Run(shipped);

        Assert.Equal(Outcome.Injected, outcome);
        Assert.Equal(shipped + Line + "\n", served);
    }

    [Fact]
    public void AnAbsentFileIsNotServed()
    {
        var (outcome, served, reads) = Run(null);

        Assert.Equal(Outcome.Absent, outcome);
        Assert.Null(served);
        Assert.Equal(1, reads);
    }

    [Theory]
    [InlineData("+a=1\n+b=2\n", "+a=1\n+b=2\n" + Line + "\n")]
    [InlineData("+a=1\r\n+b=2\r\n", "+a=1\r\n+b=2\r\n" + Line + "\r\n")]
    [InlineData("+a=1\n+b=2", "+a=1\n+b=2\n" + Line + "\n")]
    [InlineData("+a=1\r\n+b=2", "+a=1\r\n+b=2\r\n" + Line + "\r\n")]
    [InlineData("+a=1 +b=2", "+a=1 +b=2\n" + Line + "\n")]
    [InlineData("+a=1\r", "+a=1\r" + Line + "\n")]
    [InlineData("", Line + "\n")]
    public void TheLineIsAppendedInTheFilesOwnLineEnding(string original, string expected) =>
        Assert.Equal(expected, Merged(original));

    [Fact]
    public void AByteOrderMarkStaysInFront()
    {
        var original = Bom.Concat("+a=1\r\n+b=2"u8.ToArray()).ToArray();

        var merged = DemonsSoulsConvars.Merge(original)!;

        Assert.Equal(original, merged[..original.Length]);
        Assert.Equal("\r\n" + Line + "\r\n", Encoding.ASCII.GetString(merged[original.Length..]));
        Assert.Equal(Bom.Concat(Encoding.ASCII.GetBytes("\n" + Line + "\n")), DemonsSoulsConvars.Merge(Bom));
    }

    [Theory]
    [InlineData("+doIndexBufferCulling=true\n")]
    [InlineData("+a=1 +doIndexBufferCulling=false\n")]
    [InlineData("+a=1\r\n+DoIndexBufferCulling = true\r\n")]
    [InlineData("\uFEFF+doIndexBufferCulling=true")]
    [InlineData("+a=1\n+doIndexBufferCulling")]
    public void AFileThatSetsTheConvarIsLeftAsItIs(string original)
    {
        Assert.Null(DemonsSoulsConvars.Merge(Encoding.UTF8.GetBytes(original)));

        var (outcome, served, _) = Run(original);
        Assert.Equal(Outcome.AlreadySet, outcome);
        Assert.Null(served);
    }

    [Theory]
    [InlineData("+DebugMenuCreateItem=convar=doIndexBufferCulling,menu=Render\n")]
    [InlineData("+doSunShadowIndexBufferCulling=false\n")]
    [InlineData("+doIndexBufferCullingStats=true\n")]
    public void OtherCommandsThatMentionTheNameDoNotCount(string original) =>
        Assert.Equal(original + Line + "\n", Merged(original));

    [Theory]
    [InlineData("1", true)]
    [InlineData(" 1 ", true)]
    [InlineData(null, false)]
    [InlineData("", false)]
    [InlineData("0", false)]
    [InlineData("true", false)]
    public void OnlyTheSwitchSetToOneKeepsTheCulling(string? value, bool keeps) =>
        Assert.Equal(keeps, DemonsSoulsConvars.KeepsCulling(value));

    [Fact]
    public void TheSwitchLeavesTheFileUnread()
    {
        var (outcome, served, reads) = Run("+a=1\n", keepCulling: true);

        Assert.Equal(Outcome.KeptBySwitch, outcome);
        Assert.Null(served);
        Assert.Equal(0, reads);
    }

    [Theory]
    [InlineData("PPSA01341")]
    [InlineData("CUSA00001")]
    [InlineData("ppsa01342")]
    [InlineData(null)]
    public void OtherTitlesAreLeftAlone(string? titleId)
    {
        var (outcome, served, reads) = Run("+a=1\n", titleId);

        Assert.Equal(Outcome.NotApplicable, outcome);
        Assert.Null(served);
        Assert.Equal(0, reads);
    }

    [Fact]
    public void NoApp0LeavesTheTitleAlone() =>
        Assert.Equal(
            Outcome.NotApplicable,
            DemonsSoulsConvars.Apply(DemonsSoulsConvars.TitleId, null, false, _ => throw new InvalidOperationException(), (_, _) => true));

    [Fact]
    public void AFileThatCannotBeServedLongerReportsIt()
    {
        var (outcome, served, _) = Run("+a=1\n", replaceable: false);

        Assert.Equal(Outcome.CannotGrow, outcome);
        Assert.Null(served);
    }

    [Fact]
    public void AFileOnDiskIsNeitherServedLongerNorTouched()
    {
        var root = Directory.CreateTempSubdirectory("sharpemu-ds-convars-").FullName;
        try
        {
            var path = DemonsSoulsConvars.HostPath(root);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var shipped = "+TargetFramesPerSecond=60\r\n"u8.ToArray();
            File.WriteAllBytes(path, shipped);

            var outcome = DemonsSoulsConvars.Apply(
                DemonsSoulsConvars.TitleId,
                root,
                keepCulling: false,
                p => EngineResolution.TryReadApp0File(p),
                (p, content) => EngineResolution.TryReplaceApp0File(p, content, "test"));

            Assert.Equal(Outcome.CannotGrow, outcome);
            Assert.Equal(shipped, File.ReadAllBytes(path));
            Assert.False(App0FileOverlay.TryGet(path, out _));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
