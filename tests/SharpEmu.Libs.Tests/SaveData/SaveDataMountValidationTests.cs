// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.SaveData;
using Xunit;

namespace SharpEmu.Libs.Tests.SaveData;

// Mount validation (blocks, free space, mode, user), read-only enforcement, per-user
// busy tracking and the conversion/backup calls.
[Collection("SaveDataMemoryState")]
public sealed class SaveDataMountValidationTests : IDisposable
{
    private const ulong Base = 0x3_0000_0000;
    private const int UserA = 0x1001;
    private const int UserB = 0x1002;
    private const string TitleId = "SDMOUNTTEST";
    private const string DirName = "SAVE0000";

    private const ulong MountParam = Base + 0x100;
    private const ulong MountResult = Base + 0x200;
    private const ulong DirNamePtr = Base + 0x300;
    private const ulong MountPointStr = Base + 0x340;
    private const ulong ParamStruct = Base + 0x400;
    private const ulong ConvertParam = Base + 0xA00;
    private const ulong ProgressOut = Base + 0xA40;
    private const ulong BackupParam = Base + 0xA80;

    private const int ParameterError = unchecked((int)0x809F0000);
    private const int Busy = unchecked((int)0x809F0003);
    private const int NotMounted = unchecked((int)0x809F0004);
    private const int NotFound = unchecked((int)0x809F0008);
    private const int NoSpaceFs = unchecked((int)0x809F000A);
    private const int BadMounted = unchecked((int)0x809F000D);
    private const int InvalidLoginUser = unchecked((int)0x809F0011);
    private const int ResourceInvalid = unchecked((int)0x809F001C);
    private const int NoNeedConvert = unchecked((int)0x809F001E);
    private const uint ModeReadOnly = 1;
    private const uint ModeReadWrite = 2;
    private const uint ModeCreate = 4;
    private const uint ModeCreate2 = 32;

    private readonly FakeCpuMemory _memory = new(Base, 0x10000);
    private readonly CpuContext _ctx;
    private readonly string _root;
    private readonly string? _previousRoot;
    private readonly Func<string, long?> _previousFreeBytes;

    public SaveDataMountValidationTests()
    {
        _ctx = new CpuContext(_memory, Generation.Gen5);
        _root = Path.Combine(Path.GetTempPath(), $"sharpemu-sdmount-{Guid.NewGuid():N}");
        _previousRoot = Environment.GetEnvironmentVariable("SHARPEMU_SAVEDATA_DIR");
        Environment.SetEnvironmentVariable("SHARPEMU_SAVEDATA_DIR", _root);
        _previousFreeBytes = SaveDataExports.HostFreeBytesProvider;
        SaveDataExports.HostFreeBytesProvider = _ => long.MaxValue;
        SaveDataExports.ConfigureApplicationInfo(TitleId);
    }

    public void Dispose()
    {
        for (var slot = 0; slot < 16; slot++)
        {
            KernelMemoryCompatExports.UnregisterGuestPathMount($"/savedata{slot}");
        }

        SaveDataExports.HostFreeBytesProvider = _previousFreeBytes;
        SaveDataExports.ConfigureApplicationInfo(null);
        Environment.SetEnvironmentVariable("SHARPEMU_SAVEDATA_DIR", _previousRoot);
        if (Directory.Exists(_root))
        {
            Directory.Delete(_root, recursive: true);
        }
    }

    private string SlotDir => Path.Combine(_root, TitleId, DirName);

    private void WriteAscii(ulong address, string value)
    {
        var bytes = new byte[value.Length + 1];
        Encoding.ASCII.GetBytes(value).CopyTo(bytes, 0);
        Assert.True(_memory.TryWrite(address, bytes));
    }

    private CpuContext Reg(ulong rdi = 0, ulong rsi = 0, ulong rdx = 0)
    {
        _ctx[CpuRegister.Rdi] = rdi;
        _ctx[CpuRegister.Rsi] = rsi;
        _ctx[CpuRegister.Rdx] = rdx;
        return _ctx;
    }

    private int Mount(uint mode, ulong blocks = 48, int userId = UserA, string dirName = DirName)
    {
        WriteAscii(DirNamePtr, dirName);
        Span<byte> param = stackalloc byte[0x50];
        param.Clear();
        BinaryPrimitives.WriteInt32LittleEndian(param, userId);
        BinaryPrimitives.WriteUInt64LittleEndian(param[0x08..], DirNamePtr);
        BinaryPrimitives.WriteUInt64LittleEndian(param[0x10..], blocks);
        BinaryPrimitives.WriteUInt32LittleEndian(param[0x20..], mode);
        Assert.True(_memory.TryWrite(MountParam, param));
        return SaveDataExports.SaveDataMount3(Reg(rdi: MountParam, rsi: MountResult));
    }

    private string MountedPoint()
    {
        var bytes = new byte[16];
        Assert.True(_memory.TryRead(MountResult, bytes));
        return Encoding.ASCII.GetString(bytes).TrimEnd('\0');
    }

    private uint MountStatus()
    {
        Assert.True(_ctx.TryReadUInt32(MountResult + 0x1C, out var status));
        return status;
    }

    private int Umount(string mountPoint)
    {
        WriteAscii(MountPointStr, mountPoint);
        return SaveDataExports.SaveDataUmount2(Reg(rsi: MountPointStr));
    }

    [Theory]
    [InlineData(0UL)]
    [InlineData(31UL)]
    [InlineData(32769UL)]
    public void CreateWithBlocksOutsideBounds_ReturnsParameterAndCreatesNothing(ulong blocks)
    {
        Assert.Equal(ParameterError, Mount(ModeCreate | ModeReadWrite, blocks));
        Assert.Equal(ParameterError, Mount(ModeCreate2 | ModeReadWrite, blocks));
        Assert.False(Directory.Exists(SlotDir));
    }

    [Fact]
    public void CreateAtMinimumBlocks_Succeeds_AndReportsCreatedStatus()
    {
        Assert.True(_memory.TryWrite(MountResult, Enumerable.Repeat((byte)0xCD, 0x40).ToArray()));

        Assert.Equal(0, Mount(ModeCreate | ModeReadWrite, blocks: 32));

        Assert.True(Directory.Exists(SlotDir));
        Assert.Equal("/savedata0", MountedPoint());
        Assert.True(_ctx.TryReadUInt64(MountResult + 0x10, out var requiredBlocks));
        Assert.Equal(0UL, requiredBlocks);
        Assert.Equal(1u, MountStatus());
    }

    [Fact]
    public void ReopeningAnExistingSave_ReportsNotCreated()
    {
        Assert.Equal(0, Mount(ModeCreate2 | ModeReadWrite));
        Assert.Equal(1u, MountStatus());
        Assert.Equal(0, Umount(MountedPoint()));

        Assert.Equal(0, Mount(ModeCreate2 | ModeReadWrite));
        Assert.Equal(0u, MountStatus());
    }

    [Fact]
    public void CreateWithoutHostSpace_ReturnsNoSpaceFsWithRequiredBlocks()
    {
        SaveDataExports.HostFreeBytesProvider = _ => 10 * 65536L;
        Assert.True(_memory.TryWrite(MountResult, new byte[0x40]));

        Assert.Equal(NoSpaceFs, Mount(ModeCreate | ModeReadWrite, blocks: 48));

        Assert.True(_ctx.TryReadUInt64(MountResult + 0x10, out var requiredBlocks));
        Assert.Equal(38UL, requiredBlocks);
        Assert.Equal(string.Empty, MountedPoint());
        Assert.False(Directory.Exists(SlotDir));
    }

    [Fact]
    public void ModeWithoutAccessOrCreateBit_ReturnsParameter()
    {
        Directory.CreateDirectory(SlotDir);
        Assert.Equal(ParameterError, Mount(0));
    }

    [Fact]
    public void NegativeUser_ReturnsInvalidLoginUser()
    {
        Assert.Equal(InvalidLoginUser, Mount(ModeCreate | ModeReadWrite, userId: -1));
    }

    [Theory]
    [InlineData("..")]
    [InlineData("a/b")]
    public void InvalidDirectoryName_ReturnsParameter(string dirName)
    {
        Assert.Equal(ParameterError, Mount(ModeCreate | ModeReadWrite, dirName: dirName));
    }

    [Fact]
    public void ReadOnlyMount_RefusesWritesThroughTheMountPoint()
    {
        Assert.Equal(0, Mount(ModeCreate | ModeReadWrite));
        Assert.False(KernelMemoryCompatExports.IsReadOnlyGuestMutationPath("/savedata0/data.bin"));
        Assert.Equal(0, Umount("/savedata0"));

        Assert.Equal(0, Mount(ModeReadOnly));
        Assert.Equal("/savedata0", MountedPoint());
        Assert.True(KernelMemoryCompatExports.IsReadOnlyGuestMutationPath("/savedata0/data.bin"));
        Assert.True(KernelMemoryCompatExports.IsReadOnlyGuestMutationPath("/savedata0"));
        Assert.False(KernelMemoryCompatExports.IsReadOnlyGuestMutationPath("/savedata01/data.bin"));

        WriteAscii(MountPointStr, "/savedata0");
        Assert.True(_memory.TryWrite(ParamStruct, new byte[0x530]));
        Assert.Equal(BadMounted, SaveDataExports.SaveDataSetParam(Reg(rdi: MountPointStr, rdx: ParamStruct)));
        Assert.Equal(0, SaveDataExports.SaveDataGetParam(Reg(rdi: MountPointStr, rdx: ParamStruct)));

        Assert.Equal(0, Umount("/savedata0"));
        Assert.False(KernelMemoryCompatExports.IsReadOnlyGuestMutationPath("/savedata0/data.bin"));
    }

    [Fact]
    public void ReadWriteMount_AllowsWritesThroughTheMountPoint()
    {
        Directory.CreateDirectory(SlotDir);
        Assert.Equal(0, Mount(ModeReadWrite));
        Assert.False(KernelMemoryCompatExports.IsReadOnlyGuestMutationPath("/savedata0/data.bin"));
    }

    [Fact]
    public void BusyCheck_IsPerUserAndDirectory()
    {
        Assert.Equal(0, Mount(ModeCreate2 | ModeReadWrite, userId: UserA));
        Assert.Equal(Busy, Mount(ModeReadWrite, userId: UserA));

        Assert.Equal(0, Mount(ModeReadWrite, userId: UserB));
        Assert.Equal("/savedata1", MountedPoint());

        Assert.Equal(0, Mount(ModeCreate2 | ModeReadWrite, userId: UserA, dirName: "SAVE0001"));
        Assert.Equal("/savedata2", MountedPoint());
    }

    [Fact]
    public void UmountOfUnknownMountPoint_ReturnsNotMounted()
    {
        Assert.Equal(NotMounted, Umount("/savedata9"));
    }

    [Fact]
    public void Convert_ReportsProgressFromTheLastConversion()
    {
        Assert.Equal(0, SaveDataExports.SaveDataGetConvertProgress(Reg(rdi: ProgressOut)));
        Assert.Equal(0.0f, ReadProgress());

        WriteConvert("missing");
        Assert.Equal(NotFound, SaveDataExports.SaveDataConvert(Reg(rdi: ConvertParam)));
        Assert.Equal(0, SaveDataExports.SaveDataGetConvertProgress(Reg(rdi: ProgressOut)));
        Assert.Equal(0.0f, ReadProgress());

        Directory.CreateDirectory(SlotDir);
        File.WriteAllBytes(Path.Combine(SlotDir, "data.bin"), [1, 2, 3]);
        WriteConvert(DirName);
        // An existing save needs no conversion: NO_NEED_CONVERT, which titles treat as "mount as usual".
        Assert.Equal(NoNeedConvert, SaveDataExports.SaveDataConvert(Reg(rdi: ConvertParam)));
        Assert.Equal(0, SaveDataExports.SaveDataGetConvertProgress(Reg(rdi: ProgressOut)));
        Assert.Equal(1.0f, ReadProgress());
        Assert.Equal([1, 2, 3], File.ReadAllBytes(Path.Combine(SlotDir, "data.bin")));

        WriteConvert("..");
        Assert.Equal(ParameterError, SaveDataExports.SaveDataConvert(Reg(rdi: ConvertParam)));
        Assert.Equal(ParameterError, SaveDataExports.SaveDataGetConvertProgress(Reg(rdi: 0)));
    }

    [Fact]
    public void BackupDataCalls_ReportNoBackup()
    {
        Assert.True(_memory.TryWrite(BackupParam, new byte[0x40]));
        Assert.True(_memory.TryWrite(BackupParam, BitConverter.GetBytes(UserA)));

        Assert.Equal(NotFound, SaveDataExports.SaveDataCheckBackupData(Reg(rdi: BackupParam)));
        Assert.Equal(NotFound, SaveDataExports.SaveDataRestoreBackupData(Reg(rdi: BackupParam)));
        Assert.Equal(ParameterError, SaveDataExports.SaveDataCheckBackupData(Reg(rdi: 0)));
        // Cancel releases a prepared transaction resource; resource 0 was never created.
        Assert.Equal(ResourceInvalid, SaveDataExports.SaveDataCancel(Reg()));
    }

    private void WriteConvert(string dirName)
    {
        WriteAscii(DirNamePtr, dirName);
        Span<byte> param = stackalloc byte[0x40];
        param.Clear();
        BinaryPrimitives.WriteInt32LittleEndian(param, UserA);
        BinaryPrimitives.WriteUInt64LittleEndian(param[0x08..], DirNamePtr);
        Assert.True(_memory.TryWrite(ConvertParam, param));
    }

    private float ReadProgress()
    {
        var bytes = new byte[4];
        Assert.True(_memory.TryRead(ProgressOut, bytes));
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }
}
