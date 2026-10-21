// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Net;
using System.Text;
using System.Text.Json;
using System.Xml;
using SharpEmu.Core.Patches;
using SharpEmu.GUI;
using Xunit;

namespace SharpEmu.Libs.Tests.GUI;

public sealed class GamePatchDownloaderTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("sharpemu-patch-download-").FullName;
    private string Destination => Path.Combine(_root, "patches");
    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static string Xml(string version = "01.004.000", string titleId = "PPSA00001") => $$"""
        <Patch><TitleID><ID>{{titleId}}</ID></TitleID>
          <Metadata Name="Example patch" AppVer="{{version}}" AppElf="eboot.bin">
            <PatchList><Line Type="{{(version == "mask" ? "mask" : "bytes")}}" Address="{{(version == "mask" ? "aa bb" : "0x10")}}" Value="90"/></PatchList>
          </Metadata>
        </Patch>
        """;

    private static string Index(string version = "01.004.000", string file = "Match.xml", string titleId = "PPSA00001") =>
        JsonSerializer.Serialize(new
        {
            formatVersion = 1,
            files = new[]
            {
                new { file, titleIds = new[] { titleId }, versions = new[] { version } },
                new { file = "Unrelated.xml", titleIds = new[] { "PPSA99999" }, versions = new[] { "01.004.000" } },
                new { file = "OtherVersion.xml", titleIds = new[] { "PPSA00001" }, versions = new[] { "01.099.000" } },
            },
        });

    [Theory]
    [InlineData("PPSA00001", "01.004.000")]
    [InlineData("ppsa00001", "1.4.0")]
    public async Task DownloadsOnlyTheMatchingGameAndVersionWithoutEnablingIt(string titleId, string version)
    {
        var handler = new RepositoryHandler(Index(), Xml());
        using var client = new HttpClient(handler);
        var count = await GamePatchDownloader.DownloadAsync(titleId, version, Destination, client: client);
        Assert.Equal(1, count);
        Assert.Equal(new[] { "/sharpemu/ps5patches/main/patches.json", "/sharpemu/ps5patches/main/prospero_patches/Match.xml" }, handler.Requests);
        Assert.Equal(Xml(), await File.ReadAllTextAsync(Path.Combine(Destination, "Match.xml")));
        var repository = new GamePatchRepository(Destination);
        Assert.Single(repository.Load(titleId));
        Assert.Empty(repository.LoadEnabled(titleId));
        Assert.Single(Directory.GetFiles(Destination));
    }

    [Theory]
    [InlineData("PPSA00002", "01.004.000")]
    [InlineData("PPSA00001", "01.007.000")]
    [InlineData("PPSA00001", null)]
    public async Task MissingTitleOrVersionDoesNotRequestAnyXml(string titleId, string? version)
    {
        var handler = new RepositoryHandler(Index(), Xml());
        using var client = new HttpClient(handler);
        Assert.Equal(0, await GamePatchDownloader.DownloadAsync(titleId, version, Destination, client: client));
        Assert.Single(handler.Requests);
        Assert.False(Directory.Exists(Destination));
    }

    [Theory]
    [InlineData("01.007.000")]
    [InlineData(null)]
    public async Task PatternBasedPatchesCanMatchAnyInstalledVersion(string? version)
    {
        var handler = new RepositoryHandler(Index("mask"), Xml("mask"));
        using var client = new HttpClient(handler);
        Assert.Equal(1, await GamePatchDownloader.DownloadAsync("PPSA00001", version, Destination, client: client));
        Assert.True(Assert.Single(new GamePatchRepository(Destination).Load("PPSA00001")).IsPatternVersion);
    }

    [Theory]
    [InlineData("../escape.xml")]
    [InlineData("..\\escape.xml")]
    [InlineData("/escape.xml")]
    [InlineData("C:escape.xml")]
    [InlineData("Match.xml?redirect=1")]
    [InlineData("Match.xml#fragment")]
    [InlineData("Match.json")]
    public async Task UnsafeIndexFilenamesNeverReachTheNetworkOrFileSystem(string file)
    {
        var handler = new RepositoryHandler(Index(file: file), Xml());
        using var client = new HttpClient(handler);
        await Assert.ThrowsAsync<InvalidDataException>(() => GamePatchDownloader.DownloadAsync("PPSA00001", "01.004.000", Destination, client: client));
        Assert.Single(handler.Requests);
        Assert.False(Directory.Exists(Destination));
        Assert.False(File.Exists(Path.Combine(_root, "escape.xml")));
    }

    [Theory]
    [InlineData("{\"formatVersion\":2,\"files\":[]}")]
    [InlineData("{\"formatVersion\":1}")]
    [InlineData("{\"formatVersion\":1,\"files\":[null]}")]
    [InlineData("{\"formatVersion\":1,\"files\":[{\"file\":\"Match.xml\"}]}")]
    public async Task InvalidIndexesAreErrorsNotMissingPatchResults(string index)
    {
        using var client = new HttpClient(new RepositoryHandler(index, Xml()));
        await Assert.ThrowsAsync<InvalidDataException>(() => GamePatchDownloader.DownloadAsync("PPSA00001", "01.004.000", Destination, client: client));
        Assert.False(Directory.Exists(Destination));
    }

    [Theory]
    [InlineData("01.007.000", "PPSA00001")]
    [InlineData("01.004.000", "PPSA00002")]
    public async Task DownloadedXmlIsRevalidatedBeforeReplacingLocalFiles(string version, string titleId)
    {
        Directory.CreateDirectory(Destination);
        var path = Path.Combine(Destination, "Match.xml");
        await File.WriteAllTextAsync(path, "keep local patch");
        using var client = new HttpClient(new RepositoryHandler(Index(), Xml(version, titleId)));
        await Assert.ThrowsAsync<InvalidDataException>(() => GamePatchDownloader.DownloadAsync("PPSA00001", "01.004.000", Destination, client: client));
        Assert.Equal("keep local patch", await File.ReadAllTextAsync(path));
        Assert.Single(Directory.GetFiles(Destination));
    }

    [Theory]
    [InlineData("<html>not XML patches</html>")]
    [InlineData("<Patch><TitleID>")]
    [InlineData("<!DOCTYPE Patch [<!ENTITY x SYSTEM 'file:///test'>]><Patch/>")]
    public async Task MalformedDownloadsAreNeverSaved(string xml)
    {
        using var client = new HttpClient(new RepositoryHandler(Index(), xml));
        var error = await Record.ExceptionAsync(() => GamePatchDownloader.DownloadAsync("PPSA00001", "01.004.000", Destination, client: client));
        Assert.True(error is XmlException or FormatException, error?.ToString());
        Assert.False(Directory.Exists(Destination));
    }

    [Fact]
    public async Task HttpFailureIsNotReportedAsPatchNotFound()
    {
        using var client = new HttpClient(new RepositoryHandler(Index(), Xml(), HttpStatusCode.ServiceUnavailable));
        await Assert.ThrowsAsync<HttpRequestException>(() => GamePatchDownloader.DownloadAsync("PPSA00001", "01.004.000", Destination, client: client));
        Assert.False(Directory.Exists(Destination));
    }

    [Fact]
    public async Task ExistingSelectionSurvivesAnUnchangedDownload()
    {
        Directory.CreateDirectory(Destination);
        await File.WriteAllTextAsync(Path.Combine(Destination, "Match.xml"), Xml());
        var repository = new GamePatchRepository(Destination);
        repository.SetEnabled("PPSA00001", Assert.Single(repository.Load("PPSA00001")), true);
        using var client = new HttpClient(new RepositoryHandler(Index(), Xml()));
        Assert.Equal(1, await GamePatchDownloader.DownloadAsync("PPSA00001", "01.004.000", Destination, client: client));
        Assert.Contains(Assert.Single(repository.Load("PPSA00001")).Id, repository.LoadEnabled("PPSA00001"));
        Assert.Empty(Directory.GetFiles(Destination, "*.tmp"));
    }

    [Fact]
    public async Task CancellationDoesNotCreateOrReplacePatchFiles()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        using var client = new HttpClient(new RepositoryHandler(Index(), Xml()));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => GamePatchDownloader.DownloadAsync("PPSA00001", "01.004.000", Destination, cancellation.Token, client));
        Assert.False(Directory.Exists(Destination));
    }

    private sealed class RepositoryHandler(string index, string xml, HttpStatusCode xmlStatus = HttpStatusCode.OK) : HttpMessageHandler
    {
        public List<string> Requests { get; } = [];
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Assert.Equal("raw.githubusercontent.com", request.RequestUri!.Host);
            Requests.Add(request.RequestUri.AbsolutePath);
            var isIndex = request.RequestUri.AbsolutePath.EndsWith("patches.json", StringComparison.Ordinal);
            if (!isIndex) Assert.EndsWith("/prospero_patches/Match.xml", request.RequestUri.AbsolutePath);
            return Task.FromResult(new HttpResponseMessage(isIndex ? HttpStatusCode.OK : xmlStatus)
            {
                Content = new StringContent(isIndex ? index : xml, Encoding.UTF8),
            });
        }
    }
}
