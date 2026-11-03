// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class OverlayFragmentShaderTests
{
    [Fact]
    public void TheSrgbOverlayFragmentIsValidSpirv() => Validate(SpirvFixedShaders.CreateSrgbOverlayFragment());

    [Fact]
    public void ThePqOverlayFragmentIsValidSpirv() => Validate(SpirvFixedShaders.CreatePqOverlayFragment());

    [Fact]
    public void ThePqOverlayFragmentDiffersFromTheSrgbOne() =>
        Assert.NotEqual(SpirvFixedShaders.CreateSrgbOverlayFragment(), SpirvFixedShaders.CreatePqOverlayFragment());

    // Runs spirv-val from VULKAN_SDK when one is installed; otherwise the module only has to build.
    private static void Validate(byte[] code)
    {
        Assert.Equal(0x07230203u, BitConverter.ToUInt32(code, 0));
        var sdk = Environment.GetEnvironmentVariable("VULKAN_SDK");
        var validator = sdk is null ? null : Path.Combine(sdk, OperatingSystem.IsWindows() ? "Bin/spirv-val.exe" : "bin/spirv-val");
        if (validator is null || !File.Exists(validator)) return;
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, code);
            var start = new ProcessStartInfo(validator) { UseShellExecute = false, RedirectStandardError = true, CreateNoWindow = true };
            start.ArgumentList.Add("--target-env");
            start.ArgumentList.Add("vulkan1.2");
            start.ArgumentList.Add(path);
            using var process = Process.Start(start)!;
            var message = process.StandardError.ReadToEnd();
            Assert.True(process.WaitForExit(30000), "SPIR-V validation timed out.");
            Assert.True(process.ExitCode == 0, message);
        }
        finally { File.Delete(path); }
    }
}
