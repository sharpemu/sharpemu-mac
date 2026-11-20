// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu;

/// <summary>
/// A guest shader compiled by a backend, opaque to the export layers: only the backend
/// that produced it can submit it. <see cref="Payload"/> is the backend-defined compiled
/// bytes (SPIR-V words for Vulkan; MSL/DXIL for future backends), exposed solely for
/// diagnostics dumps and size traces — callers must never interpret it.
/// </summary>
public interface IGuestCompiledShader
{
    byte[] Payload { get; }
    uint? InputLocationMask => null;

    /// <summary>What the translation disk cache keeps of this shader (TranslationDiskCache), and what the backend's
    /// <see cref="IGuestGpuBackend.CreateCachedShader(byte[], uint?, ShaderCompiler.ShaderCompileRequest)"/> builds it
    /// again from: the payload, unless the payload alone does not make the shader (Metal's is the source text without
    /// the reflection the host reads).</summary>
    byte[] CachePayload => Payload;

    /// <summary>The words the flattened resource table bound with this module must have: a
    /// host binds a shorter table zero-padded to it.</summary>
    uint FlattenedTableMinimumWords => 0;

    /// <summary>File extension for diagnostics dumps of <see cref="Payload"/> ("spv",
    /// "msl", ...), so dumps stay honestly labeled whatever the backend.</summary>
    string PayloadFileExtension { get; }
}
