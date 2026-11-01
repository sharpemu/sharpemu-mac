// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.Application.Tests;

public sealed class LaunchConfigurationTests
{
    [Fact]
    public void DefaultLaunchSpecifiesEverySwitchAndLeavesAutomaticChoicesAutomatic()
    {
        var launch = new LaunchConfiguration(EffectiveLaunchSettings.Defaults, []);
        var environment = launch.BuildEnvironment(macOS: true);
        Assert.All(LaunchConfiguration.SwitchVariables, name => Assert.True(environment.ContainsKey(name)));
        Assert.All(EnvironmentSettings.All, setting => Assert.True(environment.ContainsKey(setting.Variable)));
        Assert.Equal("1", environment["SHARPEMU_WRITABLE_APP0"]);
        Assert.Equal("1", environment["SHARPEMU_STRICT_COMPUTE"]);
        Assert.Equal("0", environment["SHARPEMU_LOG_IO"]);
        Assert.Equal("1", environment["SHARPEMU_SHADER_CACHE"]);
        Assert.Null(environment["SHARPEMU_GPU_PROCESS"]);
        Assert.Null(environment["SHARPEMU_METAL_V2_IMAGE_BUDGET_MB"]);
        Assert.Equal("guest", environment["SHARPEMU_BINK_MODE"]);
    }

    [Fact]
    public void DisabledGameSwitchesOverrideInheritedEnabledValues()
    {
        var game = new PerGameSettings { EnvironmentChanges = ["SHARPEMU_WRITABLE_APP0=0", "SHARPEMU_SHADER_CACHE=off", "SHARPEMU_GPU_PROCESS=0"] };
        var launch = new LaunchConfiguration(EffectiveLaunchSettings.Resolve(game), []);
        var environment = launch.BuildEnvironment(macOS: true);
        Assert.Equal("0", environment["SHARPEMU_WRITABLE_APP0"]);
        Assert.Equal("off", environment["SHARPEMU_SHADER_CACHE"]);
        Assert.Equal("0", environment["SHARPEMU_GPU_PROCESS"]);
    }

    [Fact]
    public void CustomVariablesWinWithinFileAndScriptVariablesWinForCli()
    {
        var launch = new LaunchConfiguration(EffectiveLaunchSettings.Defaults,
            ["SHARPEMU_LOG_IO=1", "EXTRA_VALUE=custom", "SHARPEMU_GPU_PROCESS=0"]);
        var environment = launch.BuildEnvironment(macOS: true);
        Assert.Equal("1", environment["SHARPEMU_LOG_IO"]);
        Assert.Equal("custom", environment["EXTRA_VALUE"]);
        var cli = launch.BuildCliEnvironment(true, name => name == "SHARPEMU_LOG_IO" ? "0" : null);
        Assert.False(cli.ContainsKey("SHARPEMU_LOG_IO"));
        Assert.Equal("custom", cli["EXTRA_VALUE"]);
        Assert.Equal("0", cli["SHARPEMU_GPU_PROCESS"]);
    }

    [Fact]
    public void ArgumentsIncludeExplicitZeroAndFalseOverridesAndOmitUnsupportedMacRefreshRate()
    {
        var launch = new LaunchConfiguration(EffectiveLaunchSettings.Defaults with { RefreshRate = 120 }, []);
        var mac = launch.BuildArguments("test game.zar", "EnglishUS", macOS: true);
        Assert.Contains("--strict=off", mac);
        Assert.Contains("--trace-imports=0", mac);
        Assert.Contains("--no-log-file", mac);
        Assert.DoesNotContain(mac, argument => argument.StartsWith("--refresh-rate="));
        Assert.Equal(Path.GetFullPath("test game.zar"), mac[^1]);
        Assert.Contains("--refresh-rate=120", launch.BuildArguments("test.zar", "EnglishUS", macOS: false));
    }
    [Fact]
    public void LogPathRespectsTheSharedLauncherPreference()
    {
        var launcher = new ApplicationSettings { LogFilePath = Path.Combine("logs", "custom.log"), OverrideLogFile = true };
        Assert.Equal(launcher.LogFilePath, LaunchConfiguration.ResolveLogFilePath(launcher, "PPSA01234", "default-logs"));
        launcher.OverrideLogFile = false;
        var timestamped = LaunchConfiguration.ResolveLogFilePath(launcher, "PPSA01234", "default-logs");
        Assert.Equal("logs", Path.GetDirectoryName(timestamped));
        Assert.StartsWith("custom-PPSA01234-", Path.GetFileName(timestamped));
        Assert.EndsWith(".log", timestamped);
        launcher.LogFilePath = null;
        Assert.Equal("default-logs", Path.GetDirectoryName(LaunchConfiguration.ResolveLogFilePath(launcher, "PPSA01234", "default-logs")));
    }

    [Fact]
    public void OneOffVariablesApplyAfterEverySavedVariable()
    {
        var launch = new LaunchConfiguration(EffectiveLaunchSettings.Defaults, ["SHARPEMU_LOG_IO=1", "MY_VAR=saved"])
            .WithOneOff(["MY_VAR=once", "SHARPEMU_WRITABLE_APP0=0"]);

        var environment = launch.BuildEnvironment(macOS: true);

        Assert.Equal("once", environment["MY_VAR"]);
        Assert.Equal("1", environment["SHARPEMU_LOG_IO"]);
        Assert.Equal("0", environment["SHARPEMU_WRITABLE_APP0"]);
        Assert.Same(launch, launch.WithOneOff(null));
        Assert.Same(launch, launch.WithOneOff([]));
    }
}
