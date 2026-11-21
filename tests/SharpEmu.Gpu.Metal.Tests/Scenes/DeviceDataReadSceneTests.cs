// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Guest data a compute program reads on the device (upstream 7cf0ac568), through MetalRenderHost: a scalar data word
/// the program only consumes on the GPU (ScalarDataReadPolicy leaves the s_load in the shader) and the record count of
/// a raw buffer that the program itself loads (DeviceRecordCountPlan). Both live in guest memory and change between
/// two dispatches with the same user data; each dispatch must read the bytes current when it runs, not those of the
/// earlier dispatch nor those written after it.
/// </summary>
public sealed class DeviceDataReadSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Threads = 8;
    private const uint FirstCount = 6;
    private const uint FirstData = 0xA1A1_0001;
    private const uint SecondCount = 3;
    private const uint SecondData = 0xB2B2_0002;
    private const uint Untouched = 0xCCCC_CCCC;

    // s2 = the count dword (s_buffer_load through the V# in s[8:11]), s5 = the data dword at the address in s[6:7];
    // every lane stores s5 at index v0 through the V# in s[0:3] whose NUM_RECORDS is s2, so lanes at or past the count
    // store nothing.
    private static readonly uint[] Program =
    [
        0xF4200084, 0xFA000000, // s_buffer_load_dword s2, s[8:11], 0x0
        0xF4000143, 0xFA000000, // s_load_dword s5, s[6:7], 0x0
        0xBF8CC07F,             // s_waitcnt lgkmcnt(0)
        0x7E080205,             // v_mov_b32 v4, s5
        0xE0702000, 0x80000400, // buffer_store_dword v4, v0, s[0:3], 0 idxen
        0xBF810000,             // s_endpgm
    ];

    /// <summary>The program takes the two device-read paths this test is about.</summary>
    [Fact]
    public void TheProgramReadsItsCountAndDataOnTheDevice()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var address = rig.Program(Program);
        Assert.True(Gen5ShaderTranslator.TryDecodeProgram(new CpuContext(rig.Memory, Generation.Gen5), address, out var program, out var error), error);
        var onDevice = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0xD47A, 0, 12, scalarDataOnDevice: true, deviceRecordCounts: true);
        var onHost = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0xD47B, 0, 12, scalarDataOnDevice: false, deviceRecordCounts: true);
        Assert.NotNull(onDevice.DeviceRecordCounts);
        Assert.True(onDevice.TableReads.Count < onHost.TableReads.Count,
            $"device reads {onDevice.TableReads.Count}, host-flattened reads {onHost.TableReads.Count}");
    }

    /// <summary>The command processor rewrites both words between the dispatches (WRITE_DATA), in one submission.</summary>
    [Fact]
    public void BothDispatchesReadTheWordsCurrentWhenTheyRunWithinOneSubmission()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var (count, data, stored, userData) = Setup(rig);
        var pm4 = new ScenePm4();
        Dispatch(rig, pm4, userData);
        pm4.Event(ScenePm4.EventCsPartialFlush);
        pm4.Packet(ScenePm4.OpWriteData, 1u << 8, (uint)count, (uint)(count >> 32), SecondCount);
        pm4.Packet(ScenePm4.OpWriteData, 1u << 8, (uint)data, (uint)(data >> 32), SecondData);
        Dispatch(rig, pm4, userData);
        pm4.Event(ScenePm4.EventCsPartialFlush);
        rig.Submit(pm4);
        Check(rig, stored);
    }

    /// <summary>The CPU rewrites both words between two submissions.</summary>
    [Fact]
    public void BothDispatchesReadTheWordsCurrentWhenTheCpuChangesThemBetweenSubmissions()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var (count, data, stored, userData) = Setup(rig);
        var first = new ScenePm4();
        Dispatch(rig, first, userData);
        first.Event(ScenePm4.EventCsPartialFlush);
        rig.Submit(first);
        Assert.Equal([.. Enumerable.Repeat(FirstData, (int)FirstCount), .. Enumerable.Repeat(Untouched, (int)(Threads - FirstCount))],
            rig.Read<uint>(stored, (int)Threads));

        rig.Memory.Write<uint>(count, [SecondCount]);
        rig.Memory.Write<uint>(data, [SecondData]);
        var second = new ScenePm4();
        Dispatch(rig, second, userData);
        second.Event(ScenePm4.EventCsPartialFlush);
        rig.Submit(second);
        Check(rig, stored);
    }

    /// <summary>
    /// A data word on an unmapped page of the address space reads zero through the page table, as on the Vulkan host
    /// (VulkanVideoPresenter.ObtainDeviceAddressRanges: an unmapped read-only range resolves to zero), and nothing is
    /// refused.
    /// </summary>
    [Fact]
    public void ADataWordOnAnUnmappedPageReadsZero()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var (_, _, stored, userData) = Setup(rig);
        var unmapped = SceneMemory.Base + SceneMemory.Size + 0x10000;
        userData[6] = (uint)unmapped;
        userData[7] = (uint)(unmapped >> 32);
        var pm4 = new ScenePm4();
        Dispatch(rig, pm4, userData);
        pm4.Event(ScenePm4.EventCsPartialFlush);
        rig.Submit(pm4);
        Assert.Empty(rig.Fatals);
        Assert.Equal([.. Enumerable.Repeat(0u, (int)FirstCount), .. Enumerable.Repeat(Untouched, (int)(Threads - FirstCount))],
            rig.Read<uint>(stored, (int)Threads));
    }

    /// <summary>
    /// A data word past the 2^40-byte guest address space the caches track is refused, as the Vulkan host refuses it
    /// ("A device-address range is outside the cache"), instead of being read as an unmapped page.
    /// </summary>
    [Fact]
    public void ADataWordOutsideTheAddressSpaceIsRefused()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var (_, _, _, userData) = Setup(rig);
        userData[6] = 0;
        userData[7] = 1u << 9; // 2^41
        var pm4 = new ScenePm4();
        Dispatch(rig, pm4, userData);
        pm4.Event(ScenePm4.EventCsPartialFlush);
        Assert.Throws<SceneFatalException>(() => rig.Submit(pm4));
        Assert.Contains(rig.Fatals, message => message.Contains("outside the cache", StringComparison.Ordinal));
    }

    private static (ulong Count, ulong Data, ulong Stored, uint[] UserData) Setup(SceneRig rig)
    {
        var count = rig.Memory.Allocate(256, 256);
        rig.Memory.Write<uint>(count, [FirstCount, 0, 0, 0]);
        var data = rig.Memory.Allocate(256, 256);
        rig.Memory.Write<uint>(data, [FirstData, 0, 0, 0]);
        var stored = rig.Memory.Allocate(Threads * 4ul, 256);
        rig.Memory.Write<uint>(stored, Enumerable.Repeat(Untouched, (int)Threads).ToArray());
        uint[] userData =
        [
            .. Descriptors.Buffer(stored, 4, 0, 20), // s[0:3]; NUM_RECORDS is replaced by the loaded count
            0, 0,                                     // s[4:5]
            (uint)data, (uint)(data >> 32),           // s[6:7]
            .. Descriptors.Buffer(count, 0, 16, 20), // s[8:11]
        ];
        return (count, data, stored, userData);
    }

    // The second dispatch stores the second data word into its first SecondCount records; the records past them keep
    // what the first dispatch stored, and those past FirstCount are untouched.
    private static void Check(SceneRig rig, ulong stored)
    {
        Assert.Empty(rig.Fatals);
        uint[] expected =
        [
            .. Enumerable.Repeat(SecondData, (int)SecondCount),
            .. Enumerable.Repeat(FirstData, (int)(FirstCount - SecondCount)),
            .. Enumerable.Repeat(Untouched, (int)(Threads - FirstCount)),
        ];
        var words = rig.Read<uint>(stored, (int)Threads);
        Assert.True(expected.SequenceEqual(words), string.Join(" ", words.Select(word => word.ToString("X8"))));
    }

    private static void Dispatch(SceneRig rig, ScenePm4 pm4, uint[] userData)
    {
        var state = new SceneState(1, 1);
        state.Context.Clear();
        state.Uconfig.Clear();
        state.ComputeShader(rig.Program(Program), Threads, 1, 1, userData, groupIdX: false, threadIdComponents: 1);
        state.Emit(pm4).DispatchDirect(1, 1, 1);
    }
}
