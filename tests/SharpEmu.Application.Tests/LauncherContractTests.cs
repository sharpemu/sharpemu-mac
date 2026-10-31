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
    public void SerializedSettingsKeepExistingFieldNamesAndNullVersusEmptyOverride()
    {
        var global = ApplicationSettings.NormalizeFromJson("{\"LibraryLayout\":\"Grid\",\"ConsoleWindowWidth\":1234}");
        using var json = JsonDocument.Parse(JsonSerializer.Serialize(global));
        Assert.Equal(1234, json.RootElement.GetProperty("ConsoleWindowWidth").GetDouble());
        Assert.Equal("Grid", json.RootElement.GetProperty("LibraryLayout").GetString());
        Assert.Equal(EffectiveLaunchSettings.Defaults.EnvironmentToggles,
            EffectiveLaunchSettings.Resolve(PerGameSettings.NormalizeFromJson("{}")).EnvironmentToggles);
        Assert.Empty(EffectiveLaunchSettings.Resolve(PerGameSettings.NormalizeFromJson("{\"EnvironmentToggles\":[]}")).EnvironmentToggles);
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
