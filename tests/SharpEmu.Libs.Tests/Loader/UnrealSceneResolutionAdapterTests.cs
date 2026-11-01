// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class UnrealSceneResolutionAdapterTests
{
    private const string Ps5EngineIni =
        "[Audio]\r\nUseAudioMixer=true\r\n\r\n[SystemSettings]\r\nr.setres=1920x1080\r\nr.ShadowQuality=3\r\n";

    [Fact]
    public void SetResIsFoundWithItsValueSpan()
    {
        var ini = Encoding.ASCII.GetBytes(Ps5EngineIni);
        var found = Assert.Single(UnrealSceneResolutionAdapter.FindSetRes(ini));

        Assert.Equal(1920u, found.Width);
        Assert.Equal(1080u, found.Height);
        Assert.Equal("1920x1080", Encoding.ASCII.GetString(ini, found.Offset, found.Length));
    }

    [Theory]
    [InlineData(2560u, 1440u, "2560x1440")]
    [InlineData(1280u, 720u, "01280x720")]
    [InlineData(3840u, 2160u, "3840x2160")]
    public void SetResKeepsItsLengthWithAZeroPaddedWidth(uint width, uint height, string expected)
    {
        Assert.Equal(expected, Encoding.ASCII.GetString(UnrealSceneResolutionAdapter.EncodeSetRes(width, height, 9)!));
    }

    [Fact]
    public void SetResThatCannotFitIsRefused()
    {
        Assert.Null(UnrealSceneResolutionAdapter.EncodeSetRes(2560, 1440, 8));
    }

    [Fact]
    public void CommandLineGainsResXAndResYOnce()
    {
        var original = Encoding.UTF8.GetBytes("-project=../../../Poppy_Playtime/Poppy_Playtime.uproject");
        var appended = UnrealSceneResolutionAdapter.AppendResolution(original, 2560, 1440);

        Assert.Equal(
            "-project=../../../Poppy_Playtime/Poppy_Playtime.uproject -ResX=2560 -ResY=1440",
            Encoding.UTF8.GetString(appended!));
        Assert.Null(UnrealSceneResolutionAdapter.AppendResolution(appended!, 1920, 1080));
    }

    [Fact]
    public void PakIndexListsPlainIniEntriesWithTheirDataOffset()
    {
        var ini = Encoding.ASCII.GetBytes(Ps5EngineIni);
        var pak = BuildPak(("Engine/Platforms/PS5/Config/", "PS5Engine.ini", ini), ("Engine/Content/", "Asset.uasset", new byte[40]));
        using var stream = new MemoryStream(pak);

        Assert.Equal(UnrealSceneResolutionAdapter.PakIndex.Readable, UnrealSceneResolutionAdapter.ReadIniEntries(stream, out var inis));

        var entry = Assert.Single(inis);
        Assert.Equal("../../../Engine/Platforms/PS5/Config/PS5Engine.ini", entry.Path);
        Assert.Equal(ini.Length, entry.Size);
        Assert.Equal(ini, pak.AsSpan((int)entry.DataOffset, ini.Length).ToArray());
    }

    [Fact]
    public void AnEncryptedPakIndexIsReportedAsSuch()
    {
        var pak = BuildPak(("Engine/", "A.ini", new byte[4]));
        // The encrypted-index flag is the byte before the footer magic.
        var magic = BitConverter.GetBytes(UnrealSceneResolutionAdapter.PakMagic);
        var at = pak.AsSpan().LastIndexOf(magic);
        pak[at - 1] = 1;
        using var stream = new MemoryStream(pak);

        Assert.Equal(UnrealSceneResolutionAdapter.PakIndex.Encrypted, UnrealSceneResolutionAdapter.ReadIniEntries(stream, out _));
    }

    [Fact]
    public void AQuakePakIsNotAnUnrealPak()
    {
        var quake = new byte[256];
        Encoding.ASCII.GetBytes("PACK").CopyTo(quake, 0);
        using var stream = new MemoryStream(quake);

        Assert.Equal(UnrealSceneResolutionAdapter.PakIndex.NotAPak, UnrealSceneResolutionAdapter.ReadIniEntries(stream, out _));
    }

    /// <summary>A version 11 pak: entries (53-byte header + data), the primary index with the
    /// encoded entries, the full directory index, then the footer.</summary>
    private static byte[] BuildPak(params (string Directory, string File, byte[] Data)[] files)
    {
        using var output = new MemoryStream();
        var writer = new BinaryWriter(output);
        var offsets = new List<long>();
        foreach (var (_, _, data) in files)
        {
            offsets.Add(output.Position);
            writer.Write(0L);
            writer.Write((long)data.Length);
            writer.Write((long)data.Length);
            writer.Write(0);
            writer.Write(new byte[20]);
            writer.Write((byte)0);
            writer.Write(0);
            writer.Write(data);
        }

        var encoded = new MemoryStream();
        var encodedWriter = new BinaryWriter(encoded);
        var locations = new List<int>();
        for (var index = 0; index < files.Length; index++)
        {
            locations.Add((int)encoded.Position);
            // 32-bit offset and size, uncompressed, unencrypted.
            encodedWriter.Write((1u << 31) | (1u << 30));
            encodedWriter.Write((uint)offsets[index]);
            encodedWriter.Write((uint)files[index].Data.Length);
        }

        var directory = new MemoryStream();
        var directoryWriter = new BinaryWriter(directory);
        var groups = files.Select((file, index) => (file, index)).GroupBy(pair => pair.file.Directory).ToList();
        directoryWriter.Write(groups.Count);
        foreach (var group in groups)
        {
            WriteString(directoryWriter, group.Key);
            directoryWriter.Write(group.Count());
            foreach (var (file, index) in group)
            {
                WriteString(directoryWriter, file.File);
                directoryWriter.Write(locations[index]);
            }
        }

        var directoryOffset = output.Position;
        writer.Write(directory.ToArray());
        var indexOffset = output.Position;
        var primary = new MemoryStream();
        var indexWriter = new BinaryWriter(primary);
        WriteString(indexWriter, "../../../");
        indexWriter.Write(files.Length);
        indexWriter.Write(0UL);
        indexWriter.Write(0); // no path hash index
        indexWriter.Write(1); // full directory index
        indexWriter.Write(directoryOffset);
        indexWriter.Write(directory.Length);
        indexWriter.Write(new byte[20]);
        indexWriter.Write((int)encoded.Length);
        indexWriter.Write(encoded.ToArray());
        indexWriter.Write(0); // no unencoded entries
        writer.Write(primary.ToArray());

        writer.Write(new byte[16]);
        writer.Write((byte)0);
        writer.Write(UnrealSceneResolutionAdapter.PakMagic);
        writer.Write(11);
        writer.Write(indexOffset);
        writer.Write(primary.Length);
        writer.Write(new byte[20]);
        writer.Write(new byte[5 * 32]);
        return output.ToArray();
    }

    private static void WriteString(BinaryWriter writer, string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text + "\0");
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }
}
