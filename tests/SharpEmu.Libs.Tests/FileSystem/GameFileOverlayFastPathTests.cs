// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.HLE.FileSystem;
using SharpEmu.Libs.Ampr;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.Tests.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.FileSystem;

// The mod overlay against the fork's file-IO fast paths: the directory index, the APR size caches
// and the read-only image length, which answer from the game image's own listings.
[Collection(KernelMemoryCompatStateCollection.Name)]
public sealed class GameFileOverlayFastPathTests : IDisposable
{
    private const ulong Address = 0x1_0000_0000;
    private const int StatModeOffset = 8;
    private const int StatSizeOffset = 72;
    private readonly string _directory = Directory.CreateTempSubdirectory("sharpemu-overlay-fast-").FullName;
    private readonly string? _previous = Environment.GetEnvironmentVariable("SHARPEMU_APP0_DIR");
    private readonly string? _previousIndex = Environment.GetEnvironmentVariable("SHARPEMU_AMPR_INDEX_CACHE");
    private readonly FakeCpuMemory _memory = new(Address, 0x2000);
    private readonly CpuContext _context;
    private string Root => Path.Combine(_directory, "game");
    private string Mod => Path.Combine(_directory, "mod");

    public GameFileOverlayFastPathTests()
    {
        Directory.CreateDirectory(Root);
        Environment.SetEnvironmentVariable("SHARPEMU_APP0_DIR", Root);
        ForgetCachedApp0Root();
        Environment.SetEnvironmentVariable("SHARPEMU_AMPR_INDEX_CACHE", Path.Combine(_directory, "index"));
        KernelMemoryCompatExports.RegisterGuestPathMount("/app0", Root);
        _context = new CpuContext(_memory, Generation.Gen5);
    }

    public void Dispose()
    {
        AmprFileRegistry.ClearForTests();
        KernelMemoryCompatExports.UnregisterGuestPathMount("/app0");
        Environment.SetEnvironmentVariable("SHARPEMU_APP0_DIR", _previous);
        ForgetCachedApp0Root();
        Environment.SetEnvironmentVariable("SHARPEMU_AMPR_INDEX_CACHE", _previousIndex);
        Directory.Delete(_directory, recursive: true);
    }

    // The kernel trusts an index absence only under the app0 root it read first in the process, so
    // each test points that at its own root; otherwise these fast paths would go untested whenever
    // another test ran first.
    internal static void ForgetCachedApp0Root()
    {
        var field = typeof(KernelMemoryCompatExports).GetField(
            "_cachedApp0Root", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        Assert.NotNull(field);
        field.SetValue(null, null);
    }

    private static string Write(string root, string relative, byte[] data)
    {
        var path = Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, data);
        return path;
    }

    private string Host(string relative) => Path.Combine(Root, relative.Replace('/', Path.DirectorySeparatorChar));

    // 0 and (mode, size), or the error code.
    private (int Result, ushort Mode, long Size) Stat(string guestPath)
    {
        _memory.WriteCString(Address + 0x100, guestPath);
        _context[CpuRegister.Rdi] = Address + 0x100;
        _context[CpuRegister.Rsi] = Address + 0x400;
        var result = KernelMemoryCompatExports.KernelStat(_context);
        var stat = new byte[96];
        Assert.True(_memory.TryRead(Address + 0x400, stat));
        return (result,
            BinaryPrimitives.ReadUInt16LittleEndian(stat.AsSpan(StatModeOffset)),
            BinaryPrimitives.ReadInt64LittleEndian(stat.AsSpan(StatSizeOffset)));
    }

    private static ulong? AprSize(string hostPath) =>
        KernelMemoryCompatExports.TryGetAprFileSize(hostPath, out var size) ? size : null;

    [Fact]
    public void ModReplacedFileServesTheModSizeThroughStatAndAprWithAWarmIndex()
    {
        var level = Write(Root, "data/level.bin", [1, 2]);
        var modFile = Write(Mod, "data/level.bin", [9, 8, 7, 6, 5]);

        // Warm every fast path on the image's file first: the index listing, the size the
        // resolver's containment walk records, the APR size and the read-only length.
        Assert.Equal(level, KernelMemoryCompatExports.ResolveGuestPath("/app0/data/level.bin"));
        Assert.Equal(HostDirectoryIndex.Verdict.File, HostDirectoryIndex.Describe(level).Verdict);
        Assert.Equal(2UL, AprSize(level));
        Assert.True(KernelMemoryCompatExports.TryGetReadOnlyImageLength(level, out var imageLength));
        Assert.Equal(2, imageLength);

        using var overlay = new GameFileOverlay(Root, [Mod]);
        Assert.Equal(modFile, KernelMemoryCompatExports.ResolveGuestPath("/app0/data/level.bin"));
        Assert.Equal((0, (ushort)0x81A4, 5L), Stat("/app0/data/level.bin"));
        Assert.Equal(5UL, AprSize(level));
        Assert.Equal(5UL, AprSize(modFile));
        // AMPR streaming bounds its reads by this length; the image's must not be served for the mod's file.
        Assert.False(KernelMemoryCompatExports.TryGetReadOnlyImageLength(level, out _));
        Assert.False(KernelMemoryCompatExports.TryGetReadOnlyImageLength(modFile, out _));
    }

    [Fact]
    public void ModOnlyFilesAndDirectoriesAreServedThoughTheWarmIndexReportsThemAbsent()
    {
        Write(Root, "data/base.bin", [1]);
        var extra = Write(Mod, "data/extra.bin", [2, 2]);
        var added = Write(Mod, "added/new.bin", [3, 3, 3]);
        Assert.Equal(Host("data/base.bin"), KernelMemoryCompatExports.ResolveGuestPath("/app0/data/base.bin"));
        Assert.Equal(HostDirectoryIndex.Verdict.Absent, HostDirectoryIndex.Describe(Host("data/extra.bin")).Verdict);
        Assert.Equal(HostDirectoryIndex.Verdict.Absent, HostDirectoryIndex.Describe(Host("added")).Verdict);
        // The index's absence answers are trusted for this root.
        Assert.True(KernelMemoryCompatExports.TryGetReadOnlyImageLength(Host("data/base.bin"), out _));

        using var overlay = new GameFileOverlay(Root, [Mod]);
        Assert.Equal(extra, KernelMemoryCompatExports.ResolveGuestPath("/app0/data/extra.bin"));
        Assert.Equal(added, KernelMemoryCompatExports.ResolveGuestPath("/app0/added/new.bin"));
        Assert.Equal((0, (ushort)0x81A4, 2L), Stat("/app0/data/extra.bin"));
        Assert.Equal((0, (ushort)0x81A4, 3L), Stat("/app0/added/new.bin"));
        Assert.Equal(0, Stat("/app0/added").Result);
        Assert.Equal((ushort)0x41ED, Stat("/app0/added").Mode);
        Assert.Equal(2UL, AprSize(Host("data/extra.bin")));
        Assert.Equal(65536UL, AprSize(Host("added")));
        // A name neither the image nor the mod has is still absent.
        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND, Stat("/app0/added/missing.bin").Result);
        Assert.Null(AprSize(Host("added/missing.bin")));
    }

    [Fact]
    public void AGuestPathInAnotherCaseReachesTheModWhereItReachesTheImage()
    {
        // The overlay folds case exactly where this host resolves the image's names without case.
        var folds = HostDirectoryIndex.FoldsCase(Root);
        Write(Root, "data/level.bin", [1, 2]);
        Write(Root, "data/base.bin", [4]);
        // The mod spells the directory and the file its own way.
        var modFile = Write(Mod, "Data/Level.bin", [9, 8, 7]);
        Write(Mod, "added/new.bin", [3]);
        var imageResult = Stat("/app0/DATA/BASE.BIN").Result;
        Assert.Equal(folds ? 0 : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND, imageResult);

        using var overlay = new GameFileOverlay(Root, [Mod], folds);
        var resolved = KernelMemoryCompatExports.ResolveGuestPath("/app0/DATA/Level.bin");
        var stat = Stat("/app0/DATA/Level.bin");
        if (folds)
        {
            Assert.Equal(modFile, resolved);
            Assert.Equal((0, 3L), (stat.Result, stat.Size));
            Assert.Equal(3UL, AprSize(Host("DATA/Level.bin")));
            Assert.Equal(0, Stat("/app0/ADDED").Result);
            Assert.Equal(0, Stat("/app0/Added/NEW.bin").Result);
            // One name per file in merged listings.
            Assert.Equal(new[] { Host("added/new.bin"), Host("data/base.bin"), Host("data/level.bin") },
                App0Archive.EnumerateFilesRecursive(Root, "*.bin").Order(StringComparer.OrdinalIgnoreCase),
                StringComparer.OrdinalIgnoreCase);
        }
        else
        {
            Assert.Equal(Host("DATA/Level.bin"), resolved);
            Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND, stat.Result);
            Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND, Stat("/app0/ADDED").Result);
        }
    }

    [Fact]
    public void FilesNoModTouchesAnswerAsWithoutAnOverlay()
    {
        var level = Write(Root, "data/level.bin", [1, 2]);
        Assert.False(GameFileOverlay.IsMounted);
        var resolved = KernelMemoryCompatExports.ResolveGuestPath("/app0/data/level.bin");
        var baseline = (Stat("/app0/data/level.bin"), Stat("/app0/added").Result, AprSize(level), AprSize(Host("added")));
        Assert.True(KernelMemoryCompatExports.TryGetReadOnlyImageLength(level, out var length));
        Assert.Same(level, GameFileOverlay.ResolveFile(level));
        Assert.Same(HostFsPath.Comparer, GameFileOverlay.PathComparer);

        Write(Mod, "other/elsewhere.bin", [5]);
        using (new GameFileOverlay(Root, [Mod]))
        {
            Assert.True(GameFileOverlay.IsMounted);
            Assert.Equal(resolved, KernelMemoryCompatExports.ResolveGuestPath("/app0/data/level.bin"));
            Assert.Same(level, GameFileOverlay.ResolveFile(level));
            Assert.Equal(baseline, (Stat("/app0/data/level.bin"), Stat("/app0/added").Result, AprSize(level), AprSize(Host("added"))));
            Assert.True(KernelMemoryCompatExports.TryGetReadOnlyImageLength(level, out var modded));
            Assert.Equal(length, modded);
        }

        Assert.False(GameFileOverlay.IsMounted);
        Assert.Same(level, GameFileOverlay.ResolveFile(level));
        Assert.Equal(baseline, (Stat("/app0/data/level.bin"), Stat("/app0/added").Result, AprSize(level), AprSize(Host("added"))));
    }
}
