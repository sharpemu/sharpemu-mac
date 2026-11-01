// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Vulkan;

public sealed partial class SpirvModuleBuilder
{
    private readonly Dictionary<uint, uint> _resultTypes = [];
    private readonly Dictionary<uint, (uint Width, bool Signed)> _integerTypeInfo = [];
    private readonly Dictionary<uint, ulong> _wideConstantValues = [];
    private readonly Dictionary<uint, uint> _floatConstantBits = [];
    private readonly HashSet<(uint Target, SpirvDecoration Decoration)> _plainDecorations = [];
    private readonly HashSet<uint> _namedTargets = [];

    private void NoteResultType(uint id, uint type) => _resultTypes[id] = type;

    private uint IntegerWidth(uint type) => _integerTypeInfo.TryGetValue(type, out var info) ? info.Width : 0;

    private bool IsFloat32Type(uint type) => _floatTypes.TryGetValue(32, out var floatType) && floatType == type;
}
