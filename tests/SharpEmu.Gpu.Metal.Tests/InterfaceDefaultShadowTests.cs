// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The engine's host and provider interfaces give most members a default body, so a host only implements what it
/// supports. That makes one mistake silent: a class method that is meant to implement such a member but whose
/// signature differs (an added optional parameter, a changed type) compiles, is called by code that holds the class,
/// and is never called through the interface, which answers with the default instead. Upstream sync f1af467a6 did
/// that to ShaderPipelineCache.ResolveComputeProgramAhead: every dispatch resolved ahead was refused as unsupported.
/// </summary>
public sealed class InterfaceDefaultShadowTests
{
    [Fact]
    public void NoClassMethodSharesANameWithAnInterfaceDefaultItDoesNotImplement()
    {
        var assemblies = new[] { typeof(MetalRenderHost).Assembly, typeof(IShaderPipelineProvider).Assembly }
            .Concat(typeof(MetalRenderHost).Assembly.GetReferencedAssemblies()
                .Where(name => name.Name!.StartsWith("SharpEmu.", StringComparison.Ordinal))
                .Select(Assembly.Load))
            .Distinct()
            .ToArray();
        const BindingFlags Declared = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
        var shadows = new List<string>();
        foreach (var type in assemblies.SelectMany(TypesOf).Where(type => type is { IsClass: true, IsAbstract: false }))
        {
            foreach (var contract in type.GetInterfaces().Where(contract => assemblies.Contains(contract.Assembly)))
            {
                var map = type.GetInterfaceMap(contract);
                // By base definition: an override implements the member its base method does.
                var targets = new HashSet<MethodInfo>(map.TargetMethods.Select(target => target.GetBaseDefinition()));
                for (var index = 0; index < map.InterfaceMethods.Length; index++)
                {
                    // The default body answers for this member: the class implements nothing for it.
                    if (map.TargetMethods[index].DeclaringType != contract)
                    {
                        continue;
                    }

                    var member = map.InterfaceMethods[index];
                    for (var declaring = type; declaring is not null && declaring != typeof(object); declaring = declaring.BaseType)
                    {
                        foreach (var method in declaring.GetMethods(Declared))
                        {
                            if (method.Name == member.Name && !targets.Contains(method.GetBaseDefinition()))
                            {
                                shadows.Add($"{type.FullName}.{method.Name}({Parameters(method)}) does not implement " +
                                    $"{contract.Name}.{member.Name}({Parameters(member)}), which answers with its default");
                            }
                        }
                    }
                }
            }
        }

        Assert.True(shadows.Count == 0, string.Join(Environment.NewLine, shadows.Distinct().Order(StringComparer.Ordinal)));
    }

    private static IEnumerable<Type> TypesOf(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException error)
        {
            return error.Types.Where(type => type is not null)!;
        }
    }

    private static string Parameters(MethodInfo method) =>
        string.Join(", ", method.GetParameters().Select(parameter => parameter.ParameterType.Name));
}
