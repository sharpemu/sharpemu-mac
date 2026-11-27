// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using SharpEmu.Fonts.Abstractions;
using SharpEmu.Libs.Font;
using Xunit;

namespace SharpEmu.Libs.Tests.ArchitectureRules;

public sealed class FontPackageBoundaryTests
{
    [Fact]
    public void SharedGuestFontAbiHasNoSkiaAssemblyReference()
    {
        // SharpEmu.Libs references SkiaSharp for upstream's in-game overlay kit (Overlay/, the IME keyboard and save-data
        // layers), so the rule is held at the guest font ABI itself: no SharpEmu.Libs.Font type touches a Skia type.
        const System.Reflection.BindingFlags all = System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.DeclaredOnly;
        static bool IsSkia(Type? type) => type is not null &&
            ((type.Namespace?.StartsWith("Skia") ?? false) || (type.HasElementType && IsSkia(type.GetElementType())) ||
             (type.IsGenericType && type.GetGenericArguments().Any(IsSkia)));
        var fontTypes = typeof(FontExports).Assembly.GetTypes()
            .Where(type => type.Namespace?.StartsWith("SharpEmu.Libs.Font", StringComparison.Ordinal) == true).ToArray();
        Assert.NotEmpty(fontTypes);
        foreach (var type in fontTypes)
        {
            Assert.False(IsSkia(type.BaseType), type.FullName);
            Assert.DoesNotContain(type.GetFields(all), field => IsSkia(field.FieldType));
            Assert.DoesNotContain(type.GetProperties(all), property => IsSkia(property.PropertyType));
            Assert.DoesNotContain(type.GetMethods(all), method =>
                IsSkia(method.ReturnType) || method.GetParameters().Any(parameter => IsSkia(parameter.ParameterType)) ||
                (method.GetMethodBody()?.LocalVariables.Any(local => IsSkia(local.LocalType)) ?? false));
        }

        Assert.DoesNotContain(typeof(IFontFactory).Assembly.GetReferencedAssemblies(), reference => reference.Name!.StartsWith("SharpEmu.") || reference.Name!.StartsWith("Skia"));
    }
    [Fact]
    public void FontContractDoesNotExposeBackendHandlesOrTypes()
    {
        var surface = typeof(IFontFactory).Assembly.GetExportedTypes();
        foreach (var method in surface.SelectMany(type => type.GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static)))
        {
            Assert.NotEqual(typeof(nint), method.ReturnType);
            Assert.DoesNotContain(method.GetParameters(), parameter => parameter.ParameterType == typeof(nint) || parameter.ParameterType.IsPointer);
            Assert.DoesNotContain(method.GetParameters(), parameter => parameter.ParameterType.Namespace?.StartsWith("Skia") == true);
        }
    }
}
