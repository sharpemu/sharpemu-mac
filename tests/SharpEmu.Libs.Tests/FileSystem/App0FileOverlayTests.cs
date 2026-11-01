// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.FileSystem;
using Xunit;

namespace SharpEmu.Libs.Tests.FileSystem;

[Collection("App0FileOverlay")]
public sealed class App0FileOverlayTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("sharpemu-overlay-").FullName;

    public void Dispose()
    {
        App0FileOverlay.Clear();
        Directory.Delete(_directory, recursive: true);
    }

    private string WriteFile(string name, byte[] content)
    {
        var path = Path.Combine(_directory, name);
        File.WriteAllBytes(path, content);
        return path;
    }

    [Fact]
    public void PatchedRangesShowThroughEveryReadThatCoversThem()
    {
        var path = WriteFile("a.pak", Enumerable.Range(0, 64).Select(value => (byte)value).ToArray());
        App0FileOverlay.AddPatches(path, [new App0FileOverlay.Patch(10, [0xAA, 0xBB, 0xCC])], "test");

        using var stream = App0Archive.OpenReadOrFile(path);
        Assert.IsType<App0OverlayFileStream>(stream);
        var all = new byte[64];
        stream.ReadExactly(all);
        Assert.Equal([9, 0xAA, 0xBB, 0xCC, 13], all[9..14]);

        // A positional read that starts inside the patch sees only its tail.
        var tail = new byte[3];
        Assert.Equal(3, ((App0OverlayFileStream)stream).ReadAt(tail, 11));
        Assert.Equal([0xBB, 0xCC, 13], tail);
    }

    [Fact]
    public void AReplacementOfAHostFileServesItsContentAtTheSameLength()
    {
        var path = WriteFile("boot.config", "width=2560\n"u8.ToArray());
        App0FileOverlay.Replace(path, "width=1920\n"u8.ToArray(), "test");

        using var stream = App0Archive.OpenReadOrFile(path);
        using var reader = new StreamReader(stream);
        Assert.Equal("width=1920\n", reader.ReadToEnd());
    }

    [Fact]
    public void ARangePatchIsAppliedToARawBuffer()
    {
        App0FileOverlay.AddPatches(Path.Combine(_directory, "b"), [new App0FileOverlay.Patch(4, [1, 2, 3, 4])], "test");
        Assert.True(App0FileOverlay.TryGet(Path.Combine(_directory, "B"), out var entry));

        var buffer = new byte[4];
        entry.Apply(buffer, 6, 4);
        Assert.Equal([3, 4, 0, 0], buffer);
    }

    [Fact]
    public void UnlistedFilesOpenUntouched()
    {
        var path = WriteFile("c.bin", [1, 2, 3]);
        using var stream = App0Archive.OpenReadOrFile(path);
        Assert.IsNotType<App0OverlayFileStream>(stream);
    }
}
