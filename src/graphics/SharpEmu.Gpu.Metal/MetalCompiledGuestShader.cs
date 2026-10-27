// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using System.IO.Hashing;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.Libs.Gpu;

namespace SharpEmu.Gpu.Metal;

/// <summary>
/// The Metal backend's compiled shader: MSL source plus the reflection data
/// (<see cref="Gen5MslShader"/>) the presenter needs to create and bind pipeline
/// states. The diagnostics payload is the source text — Metal has no portable
/// binary form until an MTLBinaryArchive is introduced.
/// </summary>
/// <remarks>
/// The source is the bulk of a translated shader (tens to hundreds of KB, on the large
/// object heap) and is only needed until its MTLLibrary is built. A shader that carries
/// the request it was translated from can drop it (<see cref="ReleaseSource"/>): the
/// reflection data stays, and <see cref="AcquireShader"/> translates the request again
/// for the rare consumer that needs the text afterwards, checked against the hash of
/// the dropped text. A shader without a request (the fixed ones) keeps its source.
/// </remarks>
internal sealed class MetalCompiledGuestShader(Gen5MslShader shader, ShaderCompileRequest? request = null) : IGuestCompiledShader
{
    private readonly ShaderCompileRequest? _request = request;
    private Gen5MslShader _shader = shader;
    private volatile bool _sourceReleased;
    private int _releasing;
    private ulong _sourceHash;
    private ulong _payloadHash;
    private volatile bool _hasPayloadHash;

    /// <summary>The reflection data; its <see cref="Gen5MslShader.Source"/> is empty once
    /// <see cref="SourceReleased"/>. Use <see cref="AcquireShader"/> for the text.</summary>
    public Gen5MslShader Shader => Volatile.Read(ref _shader);

    /// <summary>The source text has been dropped; <see cref="AcquireShader"/> regenerates it.</summary>
    internal bool SourceReleased => _sourceReleased;

    // Not cached: a diagnostics dump is the only reader, and a cached copy is a second
    // retained form of the source.
    public byte[] Payload => Encoding.UTF8.GetBytes(AcquireShader().Source);

    public string PayloadFileExtension => "msl";

    public uint FlattenedTableMinimumWords => Shader.FlattenedTableMinimumWords;

    /// <summary>XxHash3 of the UTF-8 <see cref="Payload"/>, computed once and kept.</summary>
    internal ulong PayloadHash
    {
        get
        {
            if (_hasPayloadHash)
            {
                return _payloadHash;
            }

            var hash = Utf8Hash(AcquireShader().Source);
            _payloadHash = hash;
            _hasPayloadHash = true;
            return hash;
        }
    }

    /// <summary>The shader with its source text: the retained one, or the request translated
    /// again once the text was dropped.</summary>
    internal Gen5MslShader AcquireShader()
    {
        // Read before the flag: a stripped record is only ever seen with the flag set.
        var current = Volatile.Read(ref _shader);
        if (!_sourceReleased)
        {
            return current;
        }

        if (!Gen5MslTranslator.TryCompileProgram(_request!, out var regenerated, out var error))
        {
            throw new InvalidOperationException(
                $"A released MSL source could not be translated again: hash=0x{_request!.Hash:X16} error={error}");
        }

        if (SourceHash(regenerated.Source) != _sourceHash)
        {
            throw new InvalidOperationException(
                $"A released MSL source translated again to different text: hash=0x{_request!.Hash:X16} entry={regenerated.EntryPoint}");
        }

        return regenerated;
    }

    /// <summary>
    /// Drops the source text once its library is built, keeping the reflection data and a
    /// hash of the text. No-op for a shader without the request to translate it again.
    /// </summary>
    internal void ReleaseSource()
    {
        if (_request is null || Interlocked.Exchange(ref _releasing, 1) != 0)
        {
            return;
        }

        // The hash, then the flag, then the stripped record: a reader that sees the
        // stripped record also sees the flag and the hash (AcquireShader).
        var current = _shader;
        _sourceHash = SourceHash(current.Source);
        _sourceReleased = true;
        Volatile.Write(ref _shader, current with { Source = string.Empty });
    }

    private static ulong SourceHash(string source) =>
        XxHash3.HashToUInt64(MemoryMarshal.AsBytes(source.AsSpan()));

    private static ulong Utf8Hash(string source)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(source.Length));
        try
        {
            var length = Encoding.UTF8.GetBytes(source, buffer);
            return XxHash3.HashToUInt64(buffer.AsSpan(0, length));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
