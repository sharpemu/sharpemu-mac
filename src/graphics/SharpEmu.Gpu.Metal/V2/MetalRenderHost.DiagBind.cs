// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Text;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// Default-off binding trace for following one surface or one shader through a frame.
//
//   SHARPEMU_DIAG_BIND_HASHES=<hex prefix,...>  every stage bind whose program hash starts with one
//                       of these logs its image bindings, its buffer descriptors with the first bytes
//                       of each buffer and its user data (constants: frame index, jitter, seeds).
//   SHARPEMU_DIAG_BIND_ADDRS=<hex,...>          every image bind or colour attachment whose guest
//                       range holds one of these addresses, and every guest copy or fill that touches
//                       one, is logged with the program that did it; "*" logs every colour attachment.
//   SHARPEMU_DIAG_BIND_FLIPS=<from>-<to>        flip window (default: all).
internal sealed partial class MetalRenderHost
{
    private static readonly string[] _diagBindHashes = ParseDiagList("SHARPEMU_DIAG_BIND_HASHES");
    private static readonly string[] _diagBindAddressTexts = ParseDiagList("SHARPEMU_DIAG_BIND_ADDRS");
    private static readonly bool _diagBindAllAttachments = Array.IndexOf(_diagBindAddressTexts, "*") >= 0;
    private static readonly ulong[] _diagBindAddresses = _diagBindAddressTexts
        .Select(text => ulong.TryParse(text, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var value) ? value : 0UL)
        .Where(value => value != 0)
        .ToArray();
    private static readonly (long From, long To) _diagBindFlips = ParseDiagFlips();
    private static readonly bool _diagBindEnabled =
        _diagBindHashes.Length != 0 || _diagBindAddresses.Length != 0 || _diagBindAllAttachments;

    private static string[] ParseDiagList(string variable) =>
        Environment.GetEnvironmentVariable(variable) is { Length: > 0 } text
            ? text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            : [];

    private static (long From, long To) ParseDiagFlips()
    {
        var text = Environment.GetEnvironmentVariable("SHARPEMU_DIAG_BIND_FLIPS");
        if (text is { Length: > 0 } && text.Split('-') is [var from, var to] &&
            long.TryParse(from, out var first) && long.TryParse(to, out var last))
        {
            return (first, last);
        }

        return (0, long.MaxValue);
    }

    private long _diagBindSeenFlip = -1;
    private readonly HashSet<string> _diagBindSeen = [];

    private bool DiagBindWindow()
    {
        var flip = Interlocked.Read(ref _flips);
        return flip >= _diagBindFlips.From && flip <= _diagBindFlips.To;
    }

    private static bool DiagBindTouches(ulong address, ulong size)
    {
        foreach (var watched in _diagBindAddresses)
        {
            if (watched >= address && watched < address + Math.Max(size, 1))
            {
                return true;
            }
        }

        return false;
    }

    private static string DiagImageText(in ImageDescription description) =>
        $"{description.PixelFormat}/{description.Type}/{description.TileMode} " +
        $"{description.Extent.Width}x{description.Extent.Height}x{description.Extent.Depth} " +
        $"L{description.Resources.Levels}A{description.Resources.Layers} pitch={description.Pitch} " +
        $"@{description.Data.Address:X}+{description.Data.Size:X}";

    // End of BindResources: the stage's program, images and buffers are resolved.
    private void DiagBindResources(MetalStageBindings stage)
    {
        if (!_diagBindEnabled || !DiagBindWindow())
        {
            return;
        }

        var hash = stage.Program.Hash.ToString("X16", CultureInfo.InvariantCulture);
        var hashMatch = false;
        foreach (var prefix in _diagBindHashes)
        {
            if (hash.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                hashMatch = true;
                break;
            }
        }

        var flip = Interlocked.Read(ref _flips);
        var header = $"[DIAG][BIND] flip={flip} pass={_renderPasses} stage={stage.Program.Stage} hash={hash}";
        var snapshot = stage.StageResources.Resources;
        for (var index = 0; index < stage.ImageCount; index++)
        {
            if (!_imageCache.TryGetImage(stage.Images[index], out var image))
            {
                continue;
            }

            ref readonly var description = ref image.Description;
            if (!hashMatch && !DiagBindTouches(description.Data.Address, description.Data.Size))
            {
                continue;
            }

            Console.Error.WriteLine(
                $"{header} {(stage.IsStorage[index] ? "storage" : "sampled")}[{index}] {DiagImageText(in description)}");
        }

        // The constants of a program are written once a flip: draws of one pass repeat them.
        if (!hashMatch)
        {
            return;
        }

        if (_diagBindSeenFlip != flip)
        {
            _diagBindSeenFlip = flip;
            _diagBindSeen.Clear();
        }

        if (!_diagBindSeen.Add(hash))
        {
            return;
        }

        Console.Error.WriteLine(
            $"{header} user_data=[{string.Join(',', snapshot.UserData.ToArray().Select(word => $"{word:X8}"))}]");
        Span<byte> bytes = stackalloc byte[256];
        for (var index = 0; index < stage.BufferCount; index++)
        {
            var words = snapshot.Buffers[index];
            if (words.Length < 4)
            {
                continue;
            }

            var descriptor = BufferDescriptorWords.From(words);
            var text = new StringBuilder();
            var take = (int)Math.Min((ulong)bytes.Length, descriptor.Footprint() ?? 0UL);
            if (descriptor.Address != 0 && take > 0 && TryReadGuest(descriptor.Address, bytes[..take]))
            {
                for (var at = 0; at + 4 <= take; at += 4)
                {
                    text.Append(BitConverter.ToUInt32(bytes[at..(at + 4)]).ToString("X8", CultureInfo.InvariantCulture))
                        .Append(at % 16 == 12 ? '|' : ' ');
                }
            }

            Console.Error.WriteLine(
                $"{header} buffer[{index}] @{descriptor.Address:X} stride={descriptor.Stride} records={descriptor.RecordCount} " +
                $"written={stage.Program.Resources!.Info.Buffers[index].Written} data={text}");
        }

        // Constants the shader reaches through a V# it loads from guest memory itself. The view
        // uniform buffer is 2.5 KB with the temporal parameters and jitter past offset 1.9 KB, so
        // a large range is shown from its start and from there.
        foreach (var range in snapshot.DeviceAddressRanges)
        {
            if (!range.Planned || range.Base == 0 || range.Size == 0 || !_memory.CanRead(range.Base, 1))
            {
                continue;
            }

            Console.Error.WriteLine(
                $"{header} device_range handle={range.Handle} @{range.Base:X}+{range.Size:X} written={range.Written} " +
                $"data={DiagHexWindow(range.Base, 0, Math.Min(range.Size, 256UL))}" +
                (range.Size >= 2560
                    ? $" data@1792={DiagHexWindow(range.Base, 1792, 768)}"
                    : string.Empty));
        }
    }

    private string DiagHexWindow(ulong baseAddress, ulong offset, ulong length)
    {
        var bytes = new byte[length];
        var text = new StringBuilder();
        if (!TryReadGuest(baseAddress + offset, bytes))
        {
            return "unreadable";
        }

        for (var at = 0; at + 4 <= bytes.Length; at += 4)
        {
            text.Append(BitConverter.ToUInt32(bytes, at).ToString("X8", CultureInfo.InvariantCulture))
                .Append(at % 16 == 12 ? '|' : ' ');
        }

        return text.ToString();
    }

    // A colour attachment a pass is about to draw into.
    private void DiagBindAttachment(in ImageDescription description)
    {
        if (!_diagBindEnabled || !DiagBindWindow() ||
            !(_diagBindAllAttachments || DiagBindTouches(description.Data.Address, description.Data.Size)))
        {
            return;
        }

        Console.Error.WriteLine(
            $"[DIAG][BIND] flip={Interlocked.Read(ref _flips)} pass={_renderPasses} attach {DiagImageText(in description)}");
    }

    // A command-processor copy or fill over guest memory.
    private void DiagBindCopy(string kind, ulong destination, ulong source, ulong size)
    {
        if (!_diagBindEnabled || !DiagBindWindow() ||
            !(DiagBindTouches(destination, size) || (source != 0 && DiagBindTouches(source, size)) ||
              (_diagBindAllAttachments && size >= 0x100000)))
        {
            return;
        }

        Console.Error.WriteLine(
            $"[DIAG][BIND] flip={Interlocked.Read(ref _flips)} pass={_renderPasses} {kind} dst={destination:X} src={source:X} size={size:X}");
    }
}
