// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Text.Json;
using SharpEmu.Application;
using Xunit;
namespace SharpEmu.Application.Tests;

public sealed class LauncherContractTests
{
    [Fact]
    public void ApplicationPackageCannotPullNativeOrPresentationDependenciesIntoCli()
    {
        var references = typeof(ApplicationSettings).Assembly.GetReferencedAssemblies().Select(a => a.Name!).ToArray();
        Assert.DoesNotContain(references, name => name.StartsWith("Avalonia") || name.StartsWith("Skia") ||
            name.StartsWith("SharpEmu.GUI") || name.StartsWith("SharpEmu.Libs") || name.StartsWith("SharpEmu.Platform"));
        Assert.DoesNotContain(typeof(ApplicationSettings).Assembly.GetCustomAttributesData(),
            attribute => attribute.AttributeType.Name == "InternalsVisibleToAttribute");
    }

    [Fact]
    public void LaunchSnapshotPreservesArgumentOrderAndCustomEnvironmentPrecedence()
    {
        var global = new ApplicationSettings { StrictDynlibResolution = true, ImportTraceLimit = 17,
            WindowMode = "Borderless", BinkPlaybackMode = "Host", RenderResolutionScale = .75 };
        var game = new PerGameSettings { Resolution = "1280x720", VSync = false,
            EnvironmentToggles = ["SHARPEMU_STRICT_COMPUTE=0"], CustomEnvironmentVariables = ["TEST=game"] };
        var plan = LauncherLaunchPlan.Create("game folder/eboot.bin", global, game, ["TEST=global"], ["TEST=once"]);
        Assert.Equal(new[] { "--cpu-engine=native", "--log-level=Info", "--console=ps5", "--console-language=EnglishUS",
            "--strict", "--trace-imports=17", "--window-mode=borderless", "--resolution=1280x720", "--display=0",
            "--refresh-rate=0", "--scaling=fit", "--vsync=off", "--hdr=auto", "--overlay=on",
            "--overlay-mode=full", "--overlay-corner=topright", Path.GetFullPath("game folder/eboot.bin") }, plan.Arguments);
        Assert.Equal("once", plan.Environment["TEST"]);
        Assert.Equal("0", plan.Environment["SHARPEMU_STRICT_COMPUTE"]);
        Assert.Equal("0", plan.Environment["SHARPEMU_DISABLE_IMPORT_LOOP_GUARD"]);
        Assert.Equal("native", plan.Environment["SHARPEMU_BINK_HOST_PLAYBACK"]);
        Assert.Equal("0.75", plan.Environment["SHARPEMU_RENDER_SCALE"]);
        game.EnvironmentToggles.Clear(); game.Resolution = "1x1"; global.ConsoleLanguage = "Japanese";
        Assert.Contains("--resolution=1280x720", plan.Arguments);
        Assert.Equal("0", plan.Environment["SHARPEMU_STRICT_COMPUTE"]);
    }

    [Fact]
    public void SerializedSettingsKeepExistingFieldNamesAndNullVersusEmptyOverride()
    {
        var global = ApplicationSettings.NormalizeFromJson("{\"LibraryLayout\":\"Grid\",\"EnvironmentToggles\":[\"SHARPEMU_X\"],\"ConsoleWindowWidth\":1234}");
        using var json = JsonDocument.Parse(JsonSerializer.Serialize(global));
        Assert.Equal(1234, json.RootElement.GetProperty("ConsoleWindowWidth").GetDouble());
        Assert.Equal("Grid", json.RootElement.GetProperty("LibraryLayout").GetString());
        Assert.Equal(["SHARPEMU_X"], EffectiveLaunchSettings.Resolve(global, PerGameSettings.NormalizeFromJson("{}" )).EnvironmentToggles);
        Assert.Empty(EffectiveLaunchSettings.Resolve(global, PerGameSettings.NormalizeFromJson("{\"EnvironmentToggles\":[]}")).EnvironmentToggles);
    }

    [Fact]
    public void CatalogScanReadsBomLocalizedMetadataAndHonorsExclusionsAndDuplicateRoots()
    {
        var folder = Path.Combine(Path.GetTempPath(), $"sharpemu-catalog-{Guid.NewGuid():N}");
        try
        {
            var game = Path.Combine(folder, "game"); Directory.CreateDirectory(Path.Combine(game, "sce_sys"));
            var executable = Path.Combine(game, "eboot.bin"); File.WriteAllBytes(executable, [1, 2, 3]);
            File.WriteAllText(Path.Combine(game, "sce_sys", "param.json"), "{\"titleId\":\"PPSA1\",\"masterVersion\":\"01.2\",\"localizedParameters\":{\"defaultLanguage\":\"pl\",\"en\":{\"titleName\":\"English\"},\"pl\":{\"titleName\":\"Polski\"}}}", new System.Text.UTF8Encoding(true));
            File.WriteAllBytes(Path.Combine(game, "sce_sys", "pic0.png"), [0]);
            var entry = Assert.Single(GameCatalog.ScanFolders([folder, folder], new HashSet<string>(GameLibraryPath.Comparer)));
            Assert.Equal("Polski", entry.Name); Assert.Equal("PPSA1", entry.TitleId); Assert.Equal("01.2", entry.Version);
            Assert.Equal(3, entry.SizeBytes); Assert.EndsWith("pic0.png", entry.CoverPath);
            Assert.Empty(GameCatalog.ScanFolders([folder], new HashSet<string>([executable], GameLibraryPath.Comparer)));
        }
        finally { Directory.Delete(folder, true); }
    }

    [Fact]
    public void EmbeddedLanguagesBelongToNeutralOwnerAndKeepEnglishFallback()
    {
        var strings = new LauncherLocalization(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")));
        Assert.Contains(strings.DiscoverLanguages(), language => language.Code == "en");
        strings.Load("unavailable"); Assert.NotEqual("Page.Library", strings.Get("Page.Library"));
    }
}
