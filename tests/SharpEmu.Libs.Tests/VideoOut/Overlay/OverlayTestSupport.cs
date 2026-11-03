// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Compression;
using SharpEmu.Fonts.Skia;
using SharpEmu.Libs.VideoOut.Overlay;

namespace SharpEmu.Libs.Tests.VideoOut.Overlay;

internal static class OverlayTestSupport
{
    /// <summary>A host that renders synchronously (no thread) from a deterministic simulator.</summary>
    public static OverlayHost CreateHost(string scenario = "ds", string? preset = null, bool realFonts = true,
        OverlayLayoutStore? store = null, SimulatedOverlayDataSource? source = null)
    {
        if (source is null)
        {
            source = new SimulatedOverlayDataSource(scenario, 1337, realTime: false);
            source.Advance(14000);
        }

        var host = new OverlayHost(source, realFonts ? new SkiaFontFactory() : null, () => true, store, preset);
        return host;
    }

    public static void Click(OverlayHost host, float x, float y)
    {
        host.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Move, x, y));
        host.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Down, x, y));
        host.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Up, x, y));
    }

    /// <summary>Center (points) of the first hit region of a kind from the last render.</summary>
    public static (float X, float Y)? FindHit(OverlayController controller, OverlayHitKind kind, int a = -1)
    {
        for (var i = controller.Hits.Count - 1; i >= 0; i--)
        {
            ref var hit = ref controller.Hits[i];
            if (hit.Kind == kind && (a < 0 || hit.A == a))
            {
                return (hit.Rect.X + hit.Rect.W / 2f, hit.Rect.Y + hit.Rect.H / 2f);
            }
        }

        return null;
    }

    public static bool ClickHit(OverlayHost host, OverlayHitKind kind, int a = -1)
    {
        if (FindHit(host.Controller, kind, a) is not { } p)
        {
            return false;
        }

        Click(host, p.X, p.Y);
        return true;
    }

    /// <summary>Composites a premultiplied overlay frame over a gradient "game" background and writes a PNG.</summary>
    public static void WritePng(string path, OverlayFrame frame, bool regionsOnly = true)
    {
        var w = frame.Width;
        var h = frame.Height;
        var rgb = new byte[w * h * 3];
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                // A mid-grey gradient with a few blocks, so translucency and edges are visible.
                var t = (x / (float)w + y / (float)h) / 2f;
                var r = 70 + 90 * t;
                var g = 76 + 80 * t;
                var b = 92 + 70 * (1 - t);
                if (((x / (h / 12)) + (y / (h / 12))) % 2 == 0)
                {
                    r *= 0.92f; g *= 0.92f; b *= 0.92f;
                }

                var o = (y * w + x) * 3;
                rgb[o] = (byte)r;
                rgb[o + 1] = (byte)g;
                rgb[o + 2] = (byte)b;
            }
        }

        foreach (var region in frame.Regions)
        {
            for (var y = region.Top; y < region.Top + region.Height; y++)
            {
                for (var x = region.Left; x < region.Left + region.Width; x++)
                {
                    var s = (y * w + x) * 4;
                    var a = frame.Pixels[s + 3];
                    var o = (y * w + x) * 3;
                    rgb[o] = (byte)Math.Min(255, frame.Pixels[s + 2] + rgb[o] * (255 - a) / 255);
                    rgb[o + 1] = (byte)Math.Min(255, frame.Pixels[s + 1] + rgb[o + 1] * (255 - a) / 255);
                    rgb[o + 2] = (byte)Math.Min(255, frame.Pixels[s] + rgb[o + 2] * (255 - a) / 255);
                }
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, EncodePng(rgb, w, h));
    }

    private static byte[] EncodePng(byte[] rgb, int w, int h)
    {
        var raw = new byte[(w * 3 + 1) * h];
        for (var y = 0; y < h; y++)
        {
            Buffer.BlockCopy(rgb, y * w * 3, raw, y * (w * 3 + 1) + 1, w * 3);
        }

        using var ms = new MemoryStream();
        ms.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        var ihdr = new byte[13];
        WriteBe(ihdr, 0, w);
        WriteBe(ihdr, 4, h);
        ihdr[8] = 8;
        ihdr[9] = 2;
        Chunk(ms, "IHDR", ihdr);
        using var z = new MemoryStream();
        using (var zs = new ZLibStream(z, CompressionLevel.Fastest, leaveOpen: true))
        {
            zs.Write(raw);
        }

        Chunk(ms, "IDAT", z.ToArray());
        Chunk(ms, "IEND", []);
        return ms.ToArray();
    }

    private static void WriteBe(byte[] b, int o, int v)
    {
        b[o] = (byte)(v >> 24);
        b[o + 1] = (byte)(v >> 16);
        b[o + 2] = (byte)(v >> 8);
        b[o + 3] = (byte)v;
    }

    private static void Chunk(Stream s, string type, byte[] data)
    {
        var len = new byte[4];
        WriteBe(len, 0, data.Length);
        s.Write(len);
        var body = new byte[4 + data.Length];
        for (var i = 0; i < 4; i++)
        {
            body[i] = (byte)type[i];
        }

        data.CopyTo(body, 4);
        s.Write(body);
        var crc = new byte[4];
        WriteBe(crc, 0, unchecked((int)Crc(body)));
        s.Write(crc);
    }

    private static uint Crc(byte[] data)
    {
        var table = CrcTable.Value;
        var c = 0xFFFFFFFFu;
        foreach (var b in data)
        {
            c = table[(c ^ b) & 0xFF] ^ (c >> 8);
        }

        return c ^ 0xFFFFFFFFu;
    }

    private static readonly Lazy<uint[]> CrcTable = new(() =>
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            var c = n;
            for (var k = 0; k < 8; k++)
            {
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }

            t[n] = c;
        }

        return t;
    });

    public static string ShotsDirectory()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "SharpEmu.slnx")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        return Path.Combine(dir ?? AppContext.BaseDirectory, "artifacts", "overlay-shots");
    }
}
