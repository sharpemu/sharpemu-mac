// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Logging;

namespace SharpEmu.HLE.FileSystem;

// Each .zar gets one host folder under user/zar-cache in the data root, named after the archive.
// /app0 is mounted there while the game runs and the library reads the game's metadata from it. A
// stamp of the archive's path, size and write time tells whether the files in it came from this
// archive; when it differs, the folder starts over. Everything in it is extracted from the archive,
// so every build and process using the data root shares it: a new build does not extract the
// executable, its modules and the movies (a gigabyte for Demon's Souls) again.
public static class ZArchiveCache
{
    private const string StampName = ".sharpemu-archive";

    // The files the library shows for a game: its parameters, icon and key art.
    private static readonly string[] MetadataFiles =
    [
        "sce_sys/param.json",
        "sce_sys/icon0.png",
        "sce_sys/pic0.png",
        "sce_sys/pic1.png",
        "sce_sys/snd0.at9",
        "sce_sys/trophy2/trophy00.ucp",
    ];

    // The data root, where the library already looks for the metadata (trophies, title music). Beside
    // the executable it would be inside the app bundle, which each build replaces.
    private static string DefaultCacheParent => SharpEmuPaths.UserPath("zar-cache");

    // The cache folder an archive uses, without creating or checking it.
    public static string GetCacheRoot(string archivePath, string? cacheParent = null) =>
        Path.Combine(cacheParent ?? DefaultCacheParent, Path.GetFileNameWithoutExtension(archivePath));

    public static string Prepare(string archivePath) => Prepare(archivePath, DefaultCacheParent);

    public static string Prepare(string archivePath, string cacheParent)
    {
        archivePath = Path.GetFullPath(archivePath);
        var info = new FileInfo(archivePath);
        var cacheRoot = GetCacheRoot(archivePath, cacheParent);
        var stampPath = Path.Combine(cacheRoot, StampName);
        var stamp = $"{archivePath}|{info.Length}|{info.LastWriteTimeUtc.Ticks}";
        if (!File.Exists(stampPath) || File.ReadAllText(stampPath) != stamp)
        {
            if (Directory.Exists(cacheRoot))
            {
                Directory.Delete(cacheRoot, recursive: true);
            }

            Directory.CreateDirectory(cacheRoot);
            File.WriteAllText(stampPath, stamp);
        }

        return cacheRoot;
    }

    // Extracts an archive's sce_sys metadata into its cache folder and returns the folder, or null
    // when the file is not a readable archive. The folder stands in for the game folder.
    public static string? ExtractMetadata(string archivePath, string? cacheParent = null)
    {
        using var archive = ZArchive.Open(archivePath);
        if (archive is null)
        {
            return null;
        }

        var cacheRoot = cacheParent is null ? Prepare(archivePath) : Prepare(archivePath, cacheParent);
        foreach (var relative in MetadataFiles)
        {
            var node = archive.LookUp(relative);
            if (!archive.IsFile(node))
            {
                continue;
            }

            var hostPath = Path.Combine(cacheRoot, relative.Replace('/', Path.DirectorySeparatorChar));
            using var source = new ZArchiveFileStream(archive, node, hostPath);
            App0Archive.WriteFile(source, hostPath);
        }

        return cacheRoot;
    }
}
