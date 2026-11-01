// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Cpu;
using SharpEmu.Core.Loader;
using SharpEmu.Core.Memory;
using SharpEmu.Core.Runtime;
using SharpEmu.GnmMode.Loader;
using SharpEmu.HLE;
using SharpEmu.HLE.FileSystem;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.Tests.FileSystem;
using SharpEmu.Libs.Tests.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

[Collection(KernelMemoryCompatStateCollection.Name)]
public sealed class GnmLoadExecRuntimeTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void SelectedExecutableRetainsTheOriginalApplicationRootAndArguments(bool archiveBacked, bool selected)
    {
        var directory = Directory.CreateTempSubdirectory("sharpemu-exec-runtime-").FullName;
        var previous = Environment.GetEnvironmentVariable("SHARPEMU_APP0_DIR");
        string? cache = null;
        try
        {
            Environment.SetEnvironmentVariable("SHARPEMU_APP0_DIR", null);
            var original = GnmLoaderTests.CreateImage(false);
            var alternate = GnmLoaderTests.CreateImage(false);
            BinaryPrimitives.WriteUInt64LittleEndian(alternate.AsSpan(24), 0xB0);
            var application = Path.Combine(directory, "eboot.bin");
            string expectedRoot;
            if (archiveBacked)
            {
                application = Path.Combine(directory, Guid.NewGuid().ToString("N") + ".zar");
                File.WriteAllBytes(application, ZArchiveTestWriter.Build(
                    [("eboot.bin", original), ("big3-ps4_Shipping.elf", alternate), ("data/level.bin", new byte[] { 1, 2, 3 })]));
                cache = expectedRoot = ZArchiveCache.GetCacheRoot(application);
            }
            else
            {
                File.WriteAllBytes(application, original);
                File.WriteAllBytes(Path.Combine(directory, "big3-ps4_Shipping.elf"), alternate);
                expectedRoot = directory;
            }
            var dispatcher = new RecordingDispatcher();
            KernelMemoryCompatExports.RegisterGuestPathMount("/app0", expectedRoot);
            string[] arguments = ["first argument", "", "four", "five"];
            using var runtime = new SharpEmuRuntime(new GnmSelfLoader(), new VirtualMemory(), dispatcher,
                new ModuleManager(), cpuExecutionOptions: new CpuExecutionOptions { ProcessArguments = arguments },
                executableRelativePath: selected ? "big3-ps4_Shipping.elf" : null);
            Assert.Equal(OrbisGen2Result.ORBIS_GEN2_OK, runtime.Run(application));
            Assert.Equal(selected ? "big3-ps4_Shipping.elf" : "eboot.bin", dispatcher.ImageName);
            Assert.Equal(selected ? 0x4000B0UL : 0x4000A0UL, dispatcher.EntryPoint);
            Assert.Equal(Path.GetFullPath(expectedRoot), Path.GetFullPath(dispatcher.ApplicationRoot!));
            Assert.Equal(arguments, dispatcher.Arguments);
            Assert.False(App0Archive.IsMounted);
            if (archiveBacked)
            {
                Assert.True(File.Exists(Path.Combine(cache!, "eboot.bin")));
                Assert.Equal(selected, File.Exists(Path.Combine(cache!, "big3-ps4_Shipping.elf")));
                Assert.False(File.Exists(Path.Combine(cache!, "data", "level.bin")));
            }
        }
        finally
        {
            App0Archive.Unmount();
            KernelMemoryCompatExports.UnregisterGuestPathMount("/app0");
            Environment.SetEnvironmentVariable("SHARPEMU_APP0_DIR", previous);
            Directory.Delete(directory, recursive: true);
            if (cache is not null && Directory.Exists(cache))
                Directory.Delete(cache, recursive: true);
        }
    }

    private sealed class RecordingDispatcher : ICpuDispatcher
    {
        public string? ImageName;
        public ulong EntryPoint;
        public string? ApplicationRoot;
        public IReadOnlyList<string>? Arguments;
        public CpuTrapInfo? LastTrapInfo => null;
        public CpuMemoryFaultInfo? LastMemoryFaultInfo => null;
        public CpuControlTransferInfo? LastControlTransferInfo => null;
        public CpuNotImplementedInfo? LastNotImplementedInfo => null;
        public string? LastImportResolutionTrace => null;
        public string? LastBasicBlockTrace => null;
        public string? LastMilestoneLog => null;
        public string? LastRecentInstructionWindow => null;
        public string? LastRecentControlTransferTrace => null;
        public CpuSessionSummary LastSessionSummary => default;

        public OrbisGen2Result DispatchEntry(ulong entryPoint, Generation generation,
            IReadOnlyDictionary<ulong, string>? importStubs = null,
            IReadOnlyDictionary<string, ulong>? runtimeSymbols = null, string processImageName = "eboot.bin",
            CpuExecutionOptions executionOptions = default)
        {
            EntryPoint = entryPoint;
            ImageName = processImageName;
            ApplicationRoot = KernelMemoryCompatExports.ResolveGuestPath("/app0");
            Arguments = executionOptions.ProcessArguments;
            return OrbisGen2Result.ORBIS_GEN2_OK;
        }

        public OrbisGen2Result DispatchModuleInitializer(ulong entryPoint, Generation generation,
            IReadOnlyDictionary<ulong, string>? importStubs = null,
            IReadOnlyDictionary<string, ulong>? runtimeSymbols = null, string moduleName = "module",
            CpuExecutionOptions executionOptions = default) => OrbisGen2Result.ORBIS_GEN2_OK;
    }
}
