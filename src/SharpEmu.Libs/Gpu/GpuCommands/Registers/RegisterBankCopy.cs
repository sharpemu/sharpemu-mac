// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.GpuCommands.Registers;

// Copies into an exclusively owned, preallocated issue slot. No COW part crosses the
// publication boundary. Keep every reference field deep: the producer may reset/push/pop
// its live banks immediately after this copy, and the consumer may retain a clone.
internal static class RegisterBankCopy
{
    private static void CopyChanged<T>(T[] source, T[] destination) where T : struct
    {
        // Byte comparison preserves NaN payloads and signed zero in register structs.
        if (!System.Runtime.InteropServices.MemoryMarshal.AsBytes(source.AsSpan()).SequenceEqual(
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(destination.AsSpan()))) source.AsSpan().CopyTo(destination);
    }

    internal static void CopyTo(this RegisterBanks source, RegisterBanks destination)
    {
        source.Context.CopyTo(destination.Context);
        source.Shader.CopyTo(destination.Shader);
        source.UserConfig.CopyTo(destination.UserConfig);
        destination.CompositeDepthSizeXy = source.CompositeDepthSizeXy;
        destination.UserDataMarker = source.UserDataMarker;
        destination.IndexTypeAndSize = source.IndexTypeAndSize;
    }

    private static void CopyTo(this SampleLocationRegisters source, SampleLocationRegisters destination)
    {
        destination.CentroidPriority = source.CentroidPriority;
        CopyChanged(source.Locations, destination.Locations);
    }

    private static void CopyTo(this ScreenViewportRegisters source, ScreenViewportRegisters destination)
    {
        CopyChanged(source.Viewports, destination.Viewports);
        destination.TransformControl = source.TransformControl;
        destination.ScreenScissorLeft = source.ScreenScissorLeft;
        destination.ScreenScissorTop = source.ScreenScissorTop;
        destination.ScreenScissorRight = source.ScreenScissorRight;
        destination.ScreenScissorBottom = source.ScreenScissorBottom;
        destination.WindowScissorLeft = source.WindowScissorLeft;
        destination.WindowScissorTop = source.WindowScissorTop;
        destination.WindowScissorRight = source.WindowScissorRight;
        destination.WindowScissorBottom = source.WindowScissorBottom;
        destination.WindowScissorWindowOffsetEnable = source.WindowScissorWindowOffsetEnable;
        destination.GenericScissorLeft = source.GenericScissorLeft;
        destination.GenericScissorTop = source.GenericScissorTop;
        destination.GenericScissorRight = source.GenericScissorRight;
        destination.GenericScissorBottom = source.GenericScissorBottom;
        destination.GenericScissorWindowOffsetEnable = source.GenericScissorWindowOffsetEnable;
        destination.WindowOffsetX = source.WindowOffsetX;
        destination.WindowOffsetY = source.WindowOffsetY;
        destination.HardwareOffsetX = source.HardwareOffsetX;
        destination.HardwareOffsetY = source.HardwareOffsetY;
        destination.GuardBandHorizontalClip = source.GuardBandHorizontalClip;
        destination.GuardBandVerticalClip = source.GuardBandVerticalClip;
        destination.GuardBandHorizontalDiscard = source.GuardBandHorizontalDiscard;
        destination.GuardBandVerticalDiscard = source.GuardBandVerticalDiscard;
        destination.ClipRectangleRule = source.ClipRectangleRule;
        CopyChanged(source.ClipRectangleLeft, destination.ClipRectangleLeft);
        CopyChanged(source.ClipRectangleTop, destination.ClipRectangleTop);
        CopyChanged(source.ClipRectangleRight, destination.ClipRectangleRight);
        CopyChanged(source.ClipRectangleBottom, destination.ClipRectangleBottom);
        CopyChanged(source.ClipRectangleWindowOffsetEnable, destination.ClipRectangleWindowOffsetEnable);
    }

    private static void CopyTo(this ContextRegisters source, ContextRegisters destination)
    {
        destination.LineWidth = source.LineWidth;
        destination.PrimitiveResetIndex = source.PrimitiveResetIndex;
        destination.DepthCountControl = source.DepthCountControl;
        CopyChanged(source.BlendControls, destination.BlendControls);
        destination.BlendColor = source.BlendColor;
        CopyChanged(source.ColorTargets, destination.ColorTargets);
        CopyChanged(source.ColorClearWord1, destination.ColorClearWord1);
        CopyChanged(source.UserClipPlanes, destination.UserClipPlanes);
        destination.RenderTargetMask = source.RenderTargetMask;
        destination.AlphaToMask = source.AlphaToMask;
        source.ScreenViewport.CopyTo(destination.ScreenViewport);
        destination.Clip = source.Clip;
        destination.ColorControl = source.ColorControl;
        destination.DepthRenderOverride = source.DepthRenderOverride;
        destination.ScanMode = source.ScanMode;
        source.SampleLocations.CopyTo(destination.SampleLocations);
        destination.AntialiasingConfig = source.AntialiasingConfig;
        destination.ShaderStages = source.ShaderStages;
        destination.DepthTarget = source.DepthTarget;
        destination.ReservedDepthRegister = source.ReservedDepthRegister;
        destination.ReservedDepthRegister1 = source.ReservedDepthRegister1;
        destination.ReservedDepthRegister3 = source.ReservedDepthRegister3;
        var sameTable = source.UnmodeledTableRegisters.Count == destination.UnmodeledTableRegisters.Count;
        if (sameTable)
            foreach (var pair in source.UnmodeledTableRegisters)
                if (!destination.UnmodeledTableRegisters.TryGetValue(pair.Key, out var value) || value != pair.Value)
                { sameTable = false; break; }
        if (!sameTable)
        {
            destination.UnmodeledTableRegisters.Clear();
            foreach (var pair in source.UnmodeledTableRegisters) destination.UnmodeledTableRegisters.Add(pair.Key, pair.Value);
        }
        destination.DepthClearValue = source.DepthClearValue;
        destination.DepthBoundsMin = source.DepthBoundsMin;
        destination.DepthBoundsMax = source.DepthBoundsMax;
        destination.StencilClearValue = source.StencilClearValue;
        destination.StencilControl = source.StencilControl;
        destination.StencilMask = source.StencilMask;
        destination.RasterMode = source.RasterMode;
        destination.PolygonOffset = source.PolygonOffset;
        destination.EnhancedQualityAntialiasing = source.EnhancedQualityAntialiasing;
        source.ShaderInterface.CopyTo(destination.ShaderInterface);
    }

    private static void CopyTo(this UserScalarRegisters source, UserScalarRegisters destination)
    {
        CopyChanged(source.Values, destination.Values);
        CopyChanged(source.Kinds, destination.Kinds);
        destination.Count = source.Count;
    }

    private static void CopyTo(this VertexStageRegisters source, VertexStageRegisters destination)
    {
        destination.ExportAddress = source.ExportAddress;
        destination.LocalAddress = source.LocalAddress;
        destination.HullAddress = source.HullAddress;
        destination.HullResource1 = source.HullResource1;
        destination.HullResource2 = source.HullResource2;
        destination.GeometryAddress = source.GeometryAddress;
        destination.GeometryUserDataAddress = source.GeometryUserDataAddress;
        destination.GeometryResource1 = source.GeometryResource1;
        destination.GeometryResource2 = source.GeometryResource2;
        source.HullUserScalars.CopyTo(destination.HullUserScalars);
        source.GeometryUserScalars.CopyTo(destination.GeometryUserScalars);
        destination.ExportResource1 = source.ExportResource1;
        destination.ExportResource2 = source.ExportResource2;
        destination.LegacyVertexAddress = source.LegacyVertexAddress;
        destination.LegacyVertexResource1 = source.LegacyVertexResource1;
        destination.LegacyVertexResource2 = source.LegacyVertexResource2;
        source.LegacyVertexUserScalars.CopyTo(destination.LegacyVertexUserScalars);
        source.ExportUserScalars.CopyTo(destination.ExportUserScalars);
    }

    private static void CopyTo(this PixelStageRegisters source, PixelStageRegisters destination)
    {
        destination.Address = source.Address;
        destination.UserDataAddress = source.UserDataAddress;
        destination.AuxiliaryTableAddress = source.AuxiliaryTableAddress;
        destination.Resource1 = source.Resource1;
        destination.Resource2 = source.Resource2;
        source.UserScalars.CopyTo(destination.UserScalars);
    }

    private static void CopyTo(this ComputeStageRegisters source, ComputeStageRegisters destination)
    {
        destination.Address = source.Address;
        destination.ProgramChecksum = source.ProgramChecksum;
        destination.ThreadsX = source.ThreadsX;
        destination.ThreadsY = source.ThreadsY;
        destination.ThreadsZ = source.ThreadsZ;
        destination.VectorRegisterCount = source.VectorRegisterCount;
        destination.Priority = source.Priority;
        destination.FloatMode = source.FloatMode;
        destination.Dx10Clamp = source.Dx10Clamp;
        destination.DebugMode = source.DebugMode;
        destination.IeeeMode = source.IeeeMode;
        destination.RequireForwardProgress = source.RequireForwardProgress;
        destination.HalfPrecisionOverflow = source.HalfPrecisionOverflow;
        destination.ThreadgroupConfiguration = source.ThreadgroupConfiguration;
        destination.WaveSize = source.WaveSize;
        destination.ScratchEnable = source.ScratchEnable;
        destination.UserScalarCount = source.UserScalarCount;
        destination.ThreadGroupIdXEnable = source.ThreadGroupIdXEnable;
        destination.ThreadGroupIdYEnable = source.ThreadGroupIdYEnable;
        destination.ThreadGroupIdZEnable = source.ThreadGroupIdZEnable;
        destination.ThreadGroupSizeEnable = source.ThreadGroupSizeEnable;
        destination.ThreadIdComponentCount = source.ThreadIdComponentCount;
        destination.LocalDataShareSize = source.LocalDataShareSize;
        destination.SharedVectorRegisters = source.SharedVectorRegisters;
        source.UserScalars.CopyTo(destination.UserScalars);
    }

    private static void CopyTo(this ShaderProgramRegisters source, ShaderProgramRegisters destination)
    {
        source.Vertex.CopyTo(destination.Vertex);
        source.Pixel.CopyTo(destination.Pixel);
        source.Compute.CopyTo(destination.Compute);
    }

    private static void CopyTo(this ShaderInterfaceRegisters source, ShaderInterfaceRegisters destination)
    {
        destination.VertexOutputConfiguration = source.VertexOutputConfiguration;
        destination.PositionExportFormat = source.PositionExportFormat;
        destination.VertexOutputControl = source.VertexOutputControl;
        CopyChanged(source.PixelInterpolatorSettings, destination.PixelInterpolatorSettings);
        destination.PixelInterpolatorWritten = source.PixelInterpolatorWritten;
        destination.IndexExportFormat = source.IndexExportFormat;
        destination.PrimitiveShaderSubgroupControl = source.PrimitiveShaderSubgroupControl;
        destination.GeometryInstanceCount = source.GeometryInstanceCount;
        destination.GeometryOnChipControl = source.GeometryOnChipControl;
        destination.MaxTessellationLevel = source.MaxTessellationLevel;
        destination.MinTessellationLevel = source.MinTessellationLevel;
        destination.MaxOutputPerSubgroup = source.MaxOutputPerSubgroup;
        destination.ExportRingItemSize = source.ExportRingItemSize;
        destination.GeometryMaxVerticesOut = source.GeometryMaxVerticesOut;
        destination.GeometryOutputPrimitiveType = source.GeometryOutputPrimitiveType;
        destination.PrimitiveIdEnable = source.PrimitiveIdEnable;
        destination.VertexReuseOff = source.VertexReuseOff;
        destination.TessellationDistribution = source.TessellationDistribution;
        destination.LocalHullConfiguration = source.LocalHullConfiguration;
        destination.TessellationFactorParameter = source.TessellationFactorParameter;
        destination.DepthExportFormat = source.DepthExportFormat;
        CopyChanged(source.TargetOutputModes, destination.TargetOutputModes);
        destination.PixelInputEnable = source.PixelInputEnable;
        destination.PixelInputAddress = source.PixelInputAddress;
        destination.PixelInputControl = source.PixelInputControl;
        destination.BarycentricControl = source.BarycentricControl;
        destination.ColorShaderMask = source.ColorShaderMask;
        destination.ScanShaderControl = source.ScanShaderControl;
        destination.DepthShaderControl = source.DepthShaderControl;
    }

    private static void CopyTo(this OrderedAppendRegisters source, OrderedAppendRegisters destination)
    {
        destination.Control = source.Control;
        CopyChanged(source.Counters, destination.Counters);
    }

    private static void CopyTo(this UserConfigRegisters source, UserConfigRegisters destination)
    {
        destination.PrimitiveType = source.PrimitiveType;
        destination.IndexOffset = source.IndexOffset;
        destination.ObjectId = source.ObjectId;
        destination.PrimitiveResetControl = source.PrimitiveResetControl;
        destination.GeometryEngineControl = source.GeometryEngineControl;
        destination.GeometryEngineUserVectorEnable = source.GeometryEngineUserVectorEnable;
        source.OrderedAppend.CopyTo(destination.OrderedAppend);
    }
}
