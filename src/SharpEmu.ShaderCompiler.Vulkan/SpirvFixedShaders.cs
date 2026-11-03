// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Vulkan;

public static class SpirvFixedShaders
{
    public static byte[] CreateFullscreenVertex(uint attributeCount)
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);

        var voidType = module.TypeVoid();
        var boolType = module.TypeBool();
        var uintType = module.TypeInt(32, signed: false);
        var floatType = module.TypeFloat(32);
        var vec4Type = module.TypeVector(floatType, 4);
        var inputUintPointer = module.TypePointer(SpirvStorageClass.Input, uintType);
        var outputVec4Pointer = module.TypePointer(SpirvStorageClass.Output, vec4Type);

        var vertexIndex = module.AddGlobalVariable(inputUintPointer, SpirvStorageClass.Input);
        module.AddName(vertexIndex, "vertexIndex");
        module.AddDecoration(
            vertexIndex,
            SpirvDecoration.BuiltIn,
            (uint)SpirvBuiltIn.VertexIndex);

        var position = module.AddGlobalVariable(outputVec4Pointer, SpirvStorageClass.Output);
        module.AddName(position, "position");
        module.AddDecoration(position, SpirvDecoration.BuiltIn, (uint)SpirvBuiltIn.Position);

        var attributes = new uint[attributeCount];
        for (uint index = 0; index < attributeCount; index++)
        {
            attributes[index] =
                module.AddGlobalVariable(outputVec4Pointer, SpirvStorageClass.Output);
            module.AddName(attributes[index], $"attr{index}");
            module.AddDecoration(attributes[index], SpirvDecoration.Location, index);
            module.AddDecoration(attributes[index], SpirvDecoration.NoPerspective);
        }

        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddName(main, "main");
        module.AddLabel();

        var indexValue = module.AddInstruction(SpirvOp.Load, uintType, vertexIndex);
        var one = module.Constant(uintType, 1);
        var two = module.Constant(uintType, 2);
        var shifted = module.AddInstruction(SpirvOp.ShiftLeftLogical, uintType, indexValue, one);
        var xBits = module.AddInstruction(SpirvOp.BitwiseAnd, uintType, shifted, two);
        var yBits = module.AddInstruction(SpirvOp.BitwiseAnd, uintType, indexValue, two);
        var x = module.AddInstruction(SpirvOp.ConvertUToF, floatType, xBits);
        var y = module.AddInstruction(SpirvOp.ConvertUToF, floatType, yBits);
        var zero = module.ConstantFloat(floatType, 0f);
        var oneFloat = module.ConstantFloat(floatType, 1f);
        var twoFloat = module.ConstantFloat(floatType, 2f);
        var xPosition = module.AddInstruction(SpirvOp.FMul, floatType, x, twoFloat);
        xPosition = module.AddInstruction(SpirvOp.FSub, floatType, xPosition, oneFloat);
        var yPosition = module.AddInstruction(SpirvOp.FMul, floatType, y, twoFloat);
        yPosition = module.AddInstruction(SpirvOp.FSub, floatType, yPosition, oneFloat);
        var positionValue = module.AddInstruction(
            SpirvOp.CompositeConstruct,
            vec4Type,
            xPosition,
            yPosition,
            zero,
            oneFloat);
        module.AddStatement(SpirvOp.Store, position, positionValue);

        var attributeValue = module.AddInstruction(
            SpirvOp.CompositeConstruct,
            vec4Type,
            x,
            y,
            zero,
            oneFloat);
        foreach (var attribute in attributes)
        {
            module.AddStatement(SpirvOp.Store, attribute, attributeValue);
        }

        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var interfaces = new uint[2 + attributes.Length];
        interfaces[0] = vertexIndex;
        interfaces[1] = position;
        attributes.CopyTo(interfaces, 2);
        module.AddEntryPoint(SpirvExecutionModel.Vertex, main, "main", interfaces);
        _ = boolType;
        return module.Build();
    }

    public static byte[] CreateCopyFragment(float colorScale = 1f)
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);

        var voidType = module.TypeVoid();
        var floatType = module.TypeFloat(32);
        var vec2Type = module.TypeVector(floatType, 2);
        var vec4Type = module.TypeVector(floatType, 4);
        var inputVec4Pointer = module.TypePointer(SpirvStorageClass.Input, vec4Type);
        var outputVec4Pointer = module.TypePointer(SpirvStorageClass.Output, vec4Type);
        var imageType = module.TypeImage(
            floatType,
            SpirvImageDim.Dim2D,
            depth: false,
            arrayed: false,
            multisampled: false,
            sampled: 1,
            SpirvImageFormat.Unknown);
        var sampledImageType = module.TypeSampledImage(imageType);
        var sampledImagePointer =
            module.TypePointer(SpirvStorageClass.UniformConstant, sampledImageType);

        var attribute = module.AddGlobalVariable(inputVec4Pointer, SpirvStorageClass.Input);
        module.AddName(attribute, "attr0");
        module.AddDecoration(attribute, SpirvDecoration.Location, 0);

        var texture = module.AddGlobalVariable(
            sampledImagePointer,
            SpirvStorageClass.UniformConstant);
        module.AddName(texture, "tex0");
        module.AddDecoration(texture, SpirvDecoration.DescriptorSet, 0);
        module.AddDecoration(texture, SpirvDecoration.Binding, 1);

        var output = module.AddGlobalVariable(outputVec4Pointer, SpirvStorageClass.Output);
        module.AddName(output, "outColor");
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddName(main, "main");
        module.AddLabel();

        var attributeValue = module.AddInstruction(SpirvOp.Load, vec4Type, attribute);
        var coordinates = module.AddInstruction(
            SpirvOp.VectorShuffle,
            vec2Type,
            attributeValue,
            attributeValue,
            0,
            1);
        var sampledImage = module.AddInstruction(SpirvOp.Load, sampledImageType, texture);
        var lod = module.ConstantFloat(floatType, 0f);
        var color = module.AddInstruction(
            SpirvOp.ImageSampleExplicitLod,
            vec4Type,
            sampledImage,
            coordinates,
            2,
            lod);
        if (colorScale != 1f)
        {
            var scale = module.ConstantComposite(
                vec4Type,
                module.ConstantFloat(floatType, colorScale),
                module.ConstantFloat(floatType, colorScale),
                module.ConstantFloat(floatType, colorScale),
                module.ConstantFloat(floatType, 1f));
            color = module.AddInstruction(SpirvOp.FMul, vec4Type, color, scale);
        }
        module.AddStatement(SpirvOp.Store, output, color);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        module.AddEntryPoint(
            SpirvExecutionModel.Fragment,
            main,
            "main",
            [attribute, texture, output]);
        module.AddExecutionMode(main, SpirvExecutionMode.OriginUpperLeft);
        return module.Build();
    }

    // The performance overlay for an sRGB-encoded colour attachment. The overlay image holds
    // premultiplied sRGB-encoded colour; the attachment encodes on write and blends in linear
    // light, so un-premultiply, decode to linear, and premultiply again (alpha passes through).
    public static byte[] CreateSrgbOverlayFragment() => CreateOverlayFragment(pqWhiteNits: null);

    // The performance overlay for an HDR10 (PQ, BT.2020) attachment: the decoded linear colour is
    // moved to BT.2020 primaries, SDR white is placed at pqWhiteNits, and the result is PQ-encoded,
    // then premultiplied, so the One / OneMinusSrcAlpha blend mixes the panel in PQ code values.
    public static byte[] CreatePqOverlayFragment(float pqWhiteNits = 203f) => CreateOverlayFragment(pqWhiteNits);

    private static byte[] CreateOverlayFragment(float? pqWhiteNits)
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var glsl = module.ImportExtInst("GLSL.std.450");
        var voidType = module.TypeVoid();
        var boolType = module.TypeBool();
        var floatType = module.TypeFloat(32);
        var vec2Type = module.TypeVector(floatType, 2);
        var vec3Type = module.TypeVector(floatType, 3);
        var vec4Type = module.TypeVector(floatType, 4);
        var bool3Type = module.TypeVector(boolType, 3);
        var inputVec4Pointer = module.TypePointer(SpirvStorageClass.Input, vec4Type);
        var outputVec4Pointer = module.TypePointer(SpirvStorageClass.Output, vec4Type);
        var imageType = module.TypeImage(
            floatType,
            SpirvImageDim.Dim2D,
            depth: false,
            arrayed: false,
            multisampled: false,
            sampled: 1,
            SpirvImageFormat.Unknown);
        var sampledImageType = module.TypeSampledImage(imageType);
        var sampledImagePointer =
            module.TypePointer(SpirvStorageClass.UniformConstant, sampledImageType);
        var attribute = module.AddGlobalVariable(inputVec4Pointer, SpirvStorageClass.Input);
        module.AddDecoration(attribute, SpirvDecoration.Location, 0);
        var texture = module.AddGlobalVariable(
            sampledImagePointer,
            SpirvStorageClass.UniformConstant);
        module.AddDecoration(texture, SpirvDecoration.DescriptorSet, 0);
        module.AddDecoration(texture, SpirvDecoration.Binding, 1);
        var output = module.AddGlobalVariable(outputVec4Pointer, SpirvStorageClass.Output);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        uint Float(float value) => module.ConstantFloat(floatType, value);
        uint Vec3(float value) => module.ConstantComposite(
            vec3Type,
            Float(value),
            Float(value),
            Float(value));
        uint Ext(uint operation, uint resultType, params uint[] operands)
        {
            var values = new uint[2 + operands.Length];
            values[0] = glsl;
            values[1] = operation;
            operands.CopyTo(values, 2);
            return module.AddInstruction(SpirvOp.ExtInst, resultType, values);
        }

        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddLabel();
        var attributeValue = module.AddInstruction(SpirvOp.Load, vec4Type, attribute);
        var coordinates = module.AddInstruction(
            SpirvOp.VectorShuffle,
            vec2Type,
            attributeValue,
            attributeValue,
            0,
            1);
        var sampledImage = module.AddInstruction(SpirvOp.Load, sampledImageType, texture);
        var color = module.AddInstruction(
            SpirvOp.ImageSampleExplicitLod,
            vec4Type,
            sampledImage,
            coordinates,
            2,
            Float(0));
        var alpha = module.AddInstruction(SpirvOp.CompositeExtract, floatType, color, 3);
        var premultiplied = module.AddInstruction(
            SpirvOp.VectorShuffle,
            vec3Type,
            color,
            color,
            0,
            1,
            2);

        // FMax(alpha, tiny) keeps a fully transparent pixel at zero instead of dividing by it.
        var safeAlpha = Ext(40, floatType, alpha, Float(1e-6f));
        var safeAlpha3 = module.AddInstruction(
            SpirvOp.CompositeConstruct,
            vec3Type,
            safeAlpha,
            safeAlpha,
            safeAlpha);
        var straight = module.AddInstruction(SpirvOp.FDiv, vec3Type, premultiplied, safeAlpha3);
        straight = Ext(37, vec3Type, Ext(40, vec3Type, straight, Vec3(0f)), Vec3(1f));

        var low = module.AddInstruction(SpirvOp.FDiv, vec3Type, straight, Vec3(12.92f));
        var high = Ext(
            26,
            vec3Type,
            module.AddInstruction(
                SpirvOp.FDiv,
                vec3Type,
                module.AddInstruction(SpirvOp.FAdd, vec3Type, straight, Vec3(0.055f)),
                Vec3(1.055f)),
            Vec3(2.4f));
        var useLow = module.AddInstruction(
            SpirvOp.FOrdLessThanEqual,
            bool3Type,
            straight,
            Vec3(0.04045f));
        var linear = module.AddInstruction(SpirvOp.Select, vec3Type, useLow, low, high);
        if (pqWhiteNits is { } whiteNits)
        {
            // BT.709 -> BT.2020 primaries (ITU-R BT.2087), one row per output channel.
            uint Row(float x, float y, float z) => module.ConstantComposite(vec3Type, Float(x), Float(y), Float(z));
            var bt2020 = module.AddInstruction(
                SpirvOp.CompositeConstruct,
                vec3Type,
                module.AddInstruction(SpirvOp.Dot, floatType, linear, Row(0.6274040f, 0.3292820f, 0.0433136f)),
                module.AddInstruction(SpirvOp.Dot, floatType, linear, Row(0.0690970f, 0.9195400f, 0.0113612f)),
                module.AddInstruction(SpirvOp.Dot, floatType, linear, Row(0.0163916f, 0.0880132f, 0.8955950f)));
            var normalized = Ext(
                40,
                vec3Type,
                module.AddInstruction(SpirvOp.FMul, vec3Type, bt2020, Vec3(whiteNits / 10000f)),
                Vec3(0f));
            // SMPTE ST 2084 inverse EOTF.
            var ym1 = Ext(26, vec3Type, normalized, Vec3(2610f / 16384f));
            var numerator = module.AddInstruction(
                SpirvOp.FAdd,
                vec3Type,
                Vec3(3424f / 4096f),
                module.AddInstruction(SpirvOp.FMul, vec3Type, ym1, Vec3(2413f / 128f)));
            var denominator = module.AddInstruction(
                SpirvOp.FAdd,
                vec3Type,
                Vec3(1f),
                module.AddInstruction(SpirvOp.FMul, vec3Type, ym1, Vec3(2392f / 128f)));
            linear = Ext(
                26,
                vec3Type,
                module.AddInstruction(SpirvOp.FDiv, vec3Type, numerator, denominator),
                Vec3(2523f / 32f));
        }

        var alpha3 = module.AddInstruction(
            SpirvOp.CompositeConstruct,
            vec3Type,
            alpha,
            alpha,
            alpha);
        var linearPremultiplied = module.AddInstruction(SpirvOp.FMul, vec3Type, linear, alpha3);
        var result = module.AddInstruction(
            SpirvOp.CompositeConstruct,
            vec4Type,
            module.AddInstruction(SpirvOp.CompositeExtract, floatType, linearPremultiplied, 0),
            module.AddInstruction(SpirvOp.CompositeExtract, floatType, linearPremultiplied, 1),
            module.AddInstruction(SpirvOp.CompositeExtract, floatType, linearPremultiplied, 2),
            alpha);
        module.AddStatement(SpirvOp.Store, output, result);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();
        module.AddEntryPoint(
            SpirvExecutionModel.Fragment,
            main,
            "main",
            [attribute, texture, output]);
        module.AddExecutionMode(main, SpirvExecutionMode.OriginUpperLeft);
        return module.Build();
    }

    public static byte[] CreatePqToScRgbFragment()
    {
        const float inverseM1 = 16384f / 2610f;
        const float inverseM2 = 32f / 2523f;
        const float c1 = 3424f / 4096f;
        const float c2 = 2413f / 128f;
        const float c3 = 2392f / 128f;

        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var glsl = module.ImportExtInst("GLSL.std.450");
        var voidType = module.TypeVoid();
        var floatType = module.TypeFloat(32);
        var vec2Type = module.TypeVector(floatType, 2);
        var vec3Type = module.TypeVector(floatType, 3);
        var vec4Type = module.TypeVector(floatType, 4);
        var inputVec4Pointer = module.TypePointer(SpirvStorageClass.Input, vec4Type);
        var outputVec4Pointer = module.TypePointer(SpirvStorageClass.Output, vec4Type);
        var imageType = module.TypeImage(
            floatType,
            SpirvImageDim.Dim2D,
            depth: false,
            arrayed: false,
            multisampled: false,
            sampled: 1,
            SpirvImageFormat.Unknown);
        var sampledImageType = module.TypeSampledImage(imageType);
        var sampledImagePointer =
            module.TypePointer(SpirvStorageClass.UniformConstant, sampledImageType);
        var attribute = module.AddGlobalVariable(inputVec4Pointer, SpirvStorageClass.Input);
        module.AddDecoration(attribute, SpirvDecoration.Location, 0);
        var texture = module.AddGlobalVariable(
            sampledImagePointer,
            SpirvStorageClass.UniformConstant);
        module.AddDecoration(texture, SpirvDecoration.DescriptorSet, 0);
        module.AddDecoration(texture, SpirvDecoration.Binding, 1);
        var output = module.AddGlobalVariable(outputVec4Pointer, SpirvStorageClass.Output);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        uint Float(float value) => module.ConstantFloat(floatType, value);
        uint Vec3(float value) => module.ConstantComposite(
            vec3Type,
            Float(value),
            Float(value),
            Float(value));
        uint Ext(uint operation, uint resultType, params uint[] operands)
        {
            var values = new uint[2 + operands.Length];
            values[0] = glsl;
            values[1] = operation;
            operands.CopyTo(values, 2);
            return module.AddInstruction(SpirvOp.ExtInst, resultType, values);
        }
        uint Component(uint vector, uint index) =>
            module.AddInstruction(SpirvOp.CompositeExtract, floatType, vector, index);
        uint Multiply(uint left, float right) =>
            module.AddInstruction(SpirvOp.FMul, floatType, left, Float(right));
        uint Add3(uint first, uint second, uint third) =>
            module.AddInstruction(
                SpirvOp.FAdd,
                floatType,
                module.AddInstruction(SpirvOp.FAdd, floatType, first, second),
                third);

        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddLabel();
        var attributeValue = module.AddInstruction(SpirvOp.Load, vec4Type, attribute);
        var coordinates = module.AddInstruction(
            SpirvOp.VectorShuffle,
            vec2Type,
            attributeValue,
            attributeValue,
            0,
            1);
        var sampledImage = module.AddInstruction(SpirvOp.Load, sampledImageType, texture);
        var color = module.AddInstruction(
            SpirvOp.ImageSampleExplicitLod,
            vec4Type,
            sampledImage,
            coordinates,
            2,
            Float(0));
        var pq = module.AddInstruction(
            SpirvOp.VectorShuffle,
            vec3Type,
            color,
            color,
            0,
            1,
            2);

        // SMPTE ST 2084 converts normalized PQ code values to absolute luminance.
        var powered = Ext(26, vec3Type, pq, Vec3(inverseM2));
        var numerator = Ext(
            40,
            vec3Type,
            module.AddInstruction(SpirvOp.FSub, vec3Type, powered, Vec3(c1)),
            Vec3(0));
        var denominator = module.AddInstruction(
            SpirvOp.FSub,
            vec3Type,
            Vec3(c2),
            module.AddInstruction(SpirvOp.FMul, vec3Type, Vec3(c3), powered));
        var normalizedLuminance = Ext(
            26,
            vec3Type,
            module.AddInstruction(SpirvOp.FDiv, vec3Type, numerator, denominator),
            Vec3(inverseM1));
        var scRgb2020 = module.AddInstruction(
            SpirvOp.FMul,
            vec3Type,
            normalizedLuminance,
            Vec3(10000f / 80f));

        var red2020 = Component(scRgb2020, 0);
        var green2020 = Component(scRgb2020, 1);
        var blue2020 = Component(scRgb2020, 2);
        var red = Add3(
            Multiply(red2020, 1.660491f),
            Multiply(green2020, -0.587641f),
            Multiply(blue2020, -0.072850f));
        var green = Add3(
            Multiply(red2020, -0.124550f),
            Multiply(green2020, 1.132900f),
            Multiply(blue2020, -0.008349f));
        var blue = Add3(
            Multiply(red2020, -0.018151f),
            Multiply(green2020, -0.100579f),
            Multiply(blue2020, 1.118730f));
        var converted = module.AddInstruction(
            SpirvOp.CompositeConstruct,
            vec4Type,
            red,
            green,
            blue,
            Component(color, 3));
        module.AddStatement(SpirvOp.Store, output, converted);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();
        module.AddEntryPoint(
            SpirvExecutionModel.Fragment,
            main,
            "main",
            [attribute, texture, output]);
        module.AddExecutionMode(main, SpirvExecutionMode.OriginUpperLeft);
        return module.Build();
    }

    public static byte[] CreateSolidFragment(float red, float green, float blue, float alpha)
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);

        var voidType = module.TypeVoid();
        var floatType = module.TypeFloat(32);
        var vec4Type = module.TypeVector(floatType, 4);
        var outputVec4Pointer = module.TypePointer(SpirvStorageClass.Output, vec4Type);
        var output = module.AddGlobalVariable(outputVec4Pointer, SpirvStorageClass.Output);
        module.AddName(output, "outColor");
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddName(main, "main");
        module.AddLabel();
        var color = module.ConstantComposite(
            vec4Type,
            module.ConstantFloat(floatType, red),
            module.ConstantFloat(floatType, green),
            module.ConstantFloat(floatType, blue),
            module.ConstantFloat(floatType, alpha));
        module.AddStatement(SpirvOp.Store, output, color);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        module.AddEntryPoint(SpirvExecutionModel.Fragment, main, "main", [output]);
        module.AddExecutionMode(main, SpirvExecutionMode.OriginUpperLeft);
        return module.Build();
    }

    /// <summary>
    /// Diagnostic fragment stage that exposes one interpolated vertex output
    /// directly as color. This keeps the real guest vertex/index/depth path
    /// intact while isolating fragment-shader translation from interface data.
    /// </summary>
    public static byte[] CreateAttributeFragment(uint location)
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);

        var voidType = module.TypeVoid();
        var floatType = module.TypeFloat(32);
        var vec4Type = module.TypeVector(floatType, 4);
        var inputPointer = module.TypePointer(SpirvStorageClass.Input, vec4Type);
        var outputPointer = module.TypePointer(SpirvStorageClass.Output, vec4Type);
        var input = module.AddGlobalVariable(inputPointer, SpirvStorageClass.Input);
        module.AddName(input, $"attr{location}");
        module.AddDecoration(input, SpirvDecoration.Location, location);
        var output = module.AddGlobalVariable(outputPointer, SpirvStorageClass.Output);
        module.AddName(output, "outColor");
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddName(main, "main");
        module.AddLabel();
        var value = module.AddInstruction(SpirvOp.Load, vec4Type, input);
        module.AddStatement(SpirvOp.Store, output, value);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        module.AddEntryPoint(
            SpirvExecutionModel.Fragment,
            main,
            "main",
            [input, output]);
        module.AddExecutionMode(main, SpirvExecutionMode.OriginUpperLeft);
        return module.Build();
    }

    /// <summary>
    /// Minimal fragment stage for fixed-function depth-only passes.  The
    /// guest has no pixel shader and therefore cannot export colour; keeping
    /// this stage output-free preserves that contract while allowing Vulkan
    /// to run early/late depth tests for the translated vertex shader.
    /// </summary>
    public static byte[] CreateDepthOnlyFragment()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);

        var voidType = module.TypeVoid();
        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddName(main, "main");
        module.AddLabel();
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        module.AddEntryPoint(SpirvExecutionModel.Fragment, main, "main", []);
        module.AddExecutionMode(main, SpirvExecutionMode.OriginUpperLeft);
        return module.Build();
    }

    // Read and clear binding 0. Write the count and 64-bit page addresses to binding 1.
    // Use 64 threads per group, with one thread per 32-page word.
    public static byte[] CreateFaultBufferProcess()
    {
        const uint cachingPageBits = 14;
        const uint maxPageFaults = 1024;

        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var glsl = module.ImportExtInst("GLSL.std.450");

        var voidType = module.TypeVoid();
        var boolType = module.TypeBool();
        var uintType = module.TypeInt(32, signed: false);
        var uvec3Type = module.TypeVector(uintType, 3);
        var runtimeArray = module.TypeRuntimeArray(uintType);
        module.AddDecoration(runtimeArray, SpirvDecoration.ArrayStride, 4);
        var bufferStruct = module.TypeStruct(runtimeArray);
        module.AddDecoration(bufferStruct, SpirvDecoration.Block);
        module.AddMemberDecoration(bufferStruct, 0, SpirvDecoration.Offset, 0);
        var bufferPtrType = module.TypePointer(SpirvStorageClass.StorageBuffer, bufferStruct);
        var uintStoragePtr = module.TypePointer(SpirvStorageClass.StorageBuffer, uintType);

        uint MakeBuffer(uint binding, string name)
        {
            var variable = module.AddGlobalVariable(bufferPtrType, SpirvStorageClass.StorageBuffer);
            module.AddName(variable, name);
            module.AddDecoration(variable, SpirvDecoration.DescriptorSet, 0);
            module.AddDecoration(variable, SpirvDecoration.Binding, binding);
            return variable;
        }

        var faultVar = MakeBuffer(0, "fault_buffer");
        var downloadVar = MakeBuffer(1, "download_buffer");
        var inputUvec3Ptr = module.TypePointer(SpirvStorageClass.Input, uvec3Type);
        var gidVar = module.AddGlobalVariable(inputUvec3Ptr, SpirvStorageClass.Input);
        module.AddName(gidVar, "gid");
        module.AddDecoration(gidVar, SpirvDecoration.BuiltIn, (uint)SpirvBuiltIn.GlobalInvocationId);

        uint UInt(uint value) => module.Constant(uintType, value);

        var functionType = module.TypeFunction(voidType);
        var main = module.BeginFunction(voidType, functionType);
        module.AddName(main, "main");
        var entry = module.AddLabel();
        var gid = module.AddInstruction(SpirvOp.Load, uvec3Type, gidVar);
        var id = module.AddInstruction(SpirvOp.CompositeExtract, uintType, gid, 0);
        var wordPtr = module.AddInstruction(SpirvOp.AccessChain, uintStoragePtr, faultVar, UInt(0), id);
        var firstWord = module.AddInstruction(SpirvOp.Load, uintType, wordPtr);
        module.AddStatement(SpirvOp.Store, wordPtr, UInt(0));
        var baseBit = module.AddInstruction(SpirvOp.IMul, uintType, id, UInt(32));

        var header = module.AllocateId();
        var body = module.AllocateId();
        var store = module.AllocateId();
        var exit = module.AllocateId();
        var afterStore = module.AllocateId();
        var continueLabel = module.AllocateId();
        var merge = module.AllocateId();
        var nextWord = module.AllocateId();
        module.AddStatement(SpirvOp.Branch, header);

        module.AddLabel(header);
        var word = module.AddInstruction(SpirvOp.Phi, uintType, firstWord, entry, nextWord, continueLabel);
        var hasBits = module.AddInstruction(SpirvOp.INotEqual, boolType, word, UInt(0));
        module.AddStatement(SpirvOp.LoopMerge, merge, continueLabel, 0);
        module.AddStatement(SpirvOp.BranchConditional, hasBits, body, merge);

        module.AddLabel(body);
        var countPtr = module.AddInstruction(SpirvOp.AccessChain, uintStoragePtr, downloadVar, UInt(0), UInt(0));
        var previous = module.AddInstruction(SpirvOp.AtomicIAdd, uintType, countPtr, UInt(1), UInt(0), UInt(1));
        var storeIndex = module.AddInstruction(SpirvOp.IAdd, uintType, previous, UInt(1));
        var fits = module.AddInstruction(SpirvOp.ULessThan, boolType, storeIndex, UInt(maxPageFaults));
        module.AddStatement(SpirvOp.SelectionMerge, afterStore, 0);
        module.AddStatement(SpirvOp.BranchConditional, fits, store, exit);

        module.AddLabel(exit);
        // Keep requests that did not fit for the next fault scan.
        module.AddStatement(SpirvOp.Store, wordPtr, word);
        module.AddStatement(SpirvOp.Return);

        module.AddLabel(store);
        var bit = module.AddInstruction(SpirvOp.ExtInst, uintType, glsl, 73, word);
        var wordMinusOne = module.AddInstruction(SpirvOp.ISub, uintType, word, UInt(1));
        // The phi in the loop header names this result before it is emitted.
        module.AddStatement(SpirvOp.BitwiseAnd, uintType, nextWord, word, wordMinusOne);
        var page = module.AddInstruction(SpirvOp.IAdd, uintType, baseBit, bit);
        var low = module.AddInstruction(SpirvOp.ShiftLeftLogical, uintType, page, UInt(cachingPageBits));
        var high = module.AddInstruction(SpirvOp.ShiftRightLogical, uintType, page, UInt(32 - cachingPageBits));
        var lowIndex = module.AddInstruction(SpirvOp.IMul, uintType, storeIndex, UInt(2));
        var highIndex = module.AddInstruction(SpirvOp.IAdd, uintType, lowIndex, UInt(1));
        var lowPtr = module.AddInstruction(SpirvOp.AccessChain, uintStoragePtr, downloadVar, UInt(0), lowIndex);
        module.AddStatement(SpirvOp.Store, lowPtr, low);
        var highPtr = module.AddInstruction(SpirvOp.AccessChain, uintStoragePtr, downloadVar, UInt(0), highIndex);
        module.AddStatement(SpirvOp.Store, highPtr, high);
        module.AddStatement(SpirvOp.Branch, afterStore);

        module.AddLabel(afterStore);
        module.AddStatement(SpirvOp.Branch, continueLabel);
        module.AddLabel(continueLabel);
        module.AddStatement(SpirvOp.Branch, header);
        module.AddLabel(merge);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        module.AddExecutionMode(main, SpirvExecutionMode.LocalSize, 64, 1, 1);
        module.AddEntryPoint(SpirvExecutionModel.GLCompute, main, "main", [gidVar, faultVar, downloadVar]);
        return module.Build();
    }
}
