// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
namespace SharpEmu.GUI;

// Keep Avalonia bitmap ownership in GameEntry; cache contains metadata only.
internal static class GameLibraryCache
{
    internal static string CachePath => GameCatalogCache.CachePath;
    internal static List<GameEntry> Load(IReadOnlyList<string> folders) => GameCatalogCache.Load(folders)
        .Select(game => new GameEntry(game.Name, game.TitleId, game.Version, game.Path,
            game.SizeBytes, game.CoverPath, game.BackgroundPath, game.CompatibilityMode)).ToList();
    internal static void Save(IReadOnlyList<string> folders, IReadOnlyList<GameEntry> games) =>
        GameCatalogCache.Save(folders, games.Select(game => new CatalogGame(game.Name, game.TitleId,
            game.Version, game.Path, game.SizeBytes, game.CoverPath, game.BackgroundPath, game.CompatibilityMode)).ToArray());
}
