// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Ime;
using Xunit;

namespace SharpEmu.Libs.Tests.Ime;

[CollectionDefinition("ImeState", DisableParallelization = true)]
public sealed class ImeStateCollection
{
    public const string Name = "ImeState";
}

[Collection(ImeStateCollection.Name)]
public sealed class ImeDialogExportsTests : IDisposable
{
    private const int NotRunning = unchecked((int)0x80BC0105);
    private const int NotFinished = unchecked((int)0x80BC0106);
    private const int NotInUse = unchecked((int)0x80BC0107);
    private const int Busy = unchecked((int)0x80BC0001);
    private const int InvalidType = unchecked((int)0x80BC0011);
    private const int InvalidAddress = unchecked((int)0x80BC0031);
    private const int StatusNone = 0;
    private const int StatusRunning = 1;
    private const int StatusFinished = 2;
    private const ulong Base = 0x1_0000_0000;
    private const ulong ParamAddress = Base + 0x100;
    private const ulong BufferAddress = Base + 0x200;
    private const ulong OutAddress = Base + 0x400;
    private const ulong Out2Address = Base + 0x500;

    private readonly FakeCpuMemory _memory = new(Base, 0x1000);
    private readonly CpuContext _ctx;
    private const string AutofillVariable = "SHARPEMU_IME_AUTOFILL";
    private readonly string? _savedText;
    private readonly string? _savedAutofill;

    // These tests cover the keyboard-less Running -> Finished path (SHARPEMU_IME_AUTOFILL=1). Without it Init opens the
    // on-screen keyboard, whose close reads the host pad (SystemDialogInput.Release), which an arm64 test host cannot create.
    public ImeDialogExportsTests()
    {
        _ctx = new CpuContext(_memory, Generation.Gen5);
        _savedText = Environment.GetEnvironmentVariable(ImeDialogExports.DefaultTextEnvironmentVariable);
        _savedAutofill = Environment.GetEnvironmentVariable(AutofillVariable);
        Environment.SetEnvironmentVariable(ImeDialogExports.DefaultTextEnvironmentVariable, null);
        Environment.SetEnvironmentVariable(AutofillVariable, "1");
        ImeDialogExports.ResetForTests();
    }

    public void Dispose()
    {
        ImeDialogExports.ResetForTests();
        Environment.SetEnvironmentVariable(ImeDialogExports.DefaultTextEnvironmentVariable, _savedText);
        Environment.SetEnvironmentVariable(AutofillVariable, _savedAutofill);
    }

    [Fact]
    public void LifecycleWalksRunningFinishedNone()
    {
        Assert.Equal(StatusNone, ImeDialogExports.ImeDialogGetStatus(_ctx));

        Init();
        Assert.Equal(StatusRunning, ImeDialogExports.ImeDialogGetStatus(_ctx));
        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(_ctx));
        Assert.Equal((ulong)StatusFinished, _ctx[CpuRegister.Rax]);

        Assert.Equal(0, ImeDialogExports.ImeDialogTerm(_ctx));
        Assert.Equal(StatusNone, ImeDialogExports.ImeDialogGetStatus(_ctx));
    }

    [Fact]
    public void InitDoesNotInventUserText()
    {
        // The title's own initial text must survive when no default text is configured.
        WriteBuffer("Player1");
        Init(maxTextLength: 16);

        Assert.Equal("Player1", ReadBuffer());
    }

    [Fact]
    public void ConfiguredDefaultTextIsWrittenAndTruncatedToMaxLength()
    {
        Environment.SetEnvironmentVariable(ImeDialogExports.DefaultTextEnvironmentVariable, "Alexander");
        Init(maxTextLength: 4);

        Assert.Equal("Alex", ReadBuffer());
        // NUL terminator right after the truncated text.
        var tail = new byte[2];
        Assert.True(_memory.TryRead(BufferAddress + 8, tail));
        Assert.Equal(new byte[] { 0, 0 }, tail);
    }

    [Fact]
    public void InitValidatesParameterAndBusyState()
    {
        _ctx[CpuRegister.Rdi] = 0;
        Assert.Equal(InvalidAddress, ImeDialogExports.ImeDialogInit(_ctx));

        WriteParam(type: 9, maxTextLength: 8);
        _ctx[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(InvalidType, ImeDialogExports.ImeDialogInit(_ctx));
        Assert.Equal(StatusNone, ImeDialogExports.ImeDialogGetStatus(_ctx));

        Init();
        _ctx[CpuRegister.Rdi] = ParamAddress;
        WriteParam(type: 0, maxTextLength: 8);
        Assert.Equal(Busy, ImeDialogExports.ImeDialogInit(_ctx));
    }

    [Fact]
    public void GetResultWritesTheFullSixteenByteStruct()
    {
        Init();
        FinishByPolling();
        Assert.True(_memory.TryWrite(OutAddress, Enumerable.Repeat((byte)0x6B, 0x20).ToArray()));

        _ctx[CpuRegister.Rdi] = OutAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogGetResult(_ctx));

        var bytes = new byte[0x20];
        Assert.True(_memory.TryRead(OutAddress, bytes));
        Assert.Equal(0u, BinaryPrimitives.ReadUInt32LittleEndian(bytes));
        Assert.All(bytes.Skip(4).Take(12), b => Assert.Equal(0, b));
        Assert.Equal(0x6B, bytes[0x10]);
    }

    [Fact]
    public void GetResultAbortAndTermFollowTheStateMachine()
    {
        _ctx[CpuRegister.Rdi] = OutAddress;
        Assert.Equal(NotInUse, ImeDialogExports.ImeDialogGetResult(_ctx));
        Assert.Equal(NotInUse, ImeDialogExports.ImeDialogAbort(_ctx));
        Assert.Equal(NotInUse, ImeDialogExports.ImeDialogTerm(_ctx));

        Init();
        _ctx[CpuRegister.Rdi] = 0;
        Assert.Equal(InvalidAddress, ImeDialogExports.ImeDialogGetResult(_ctx));
        _ctx[CpuRegister.Rdi] = OutAddress;
        Assert.Equal(NotFinished, ImeDialogExports.ImeDialogGetResult(_ctx));
        Assert.Equal(NotFinished, ImeDialogExports.ImeDialogTerm(_ctx));

        Assert.Equal(0, ImeDialogExports.ImeDialogAbort(_ctx));
        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(_ctx));
        Assert.Equal(NotRunning, ImeDialogExports.ImeDialogAbort(_ctx));

        Assert.Equal(0, ImeDialogExports.ImeDialogGetResult(_ctx));
        var result = new byte[4];
        Assert.True(_memory.TryRead(OutAddress, result));
        // SCE_IME_DIALOG_END_STATUS_ABORTED.
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(result));
    }

    [Theory]
    [InlineData(0u, 0u, 793u, 528u)]
    [InlineData(1u, 0u, 793u, 528u)]
    [InlineData(0u, 0x1u, 793u, 628u)]
    [InlineData(4u, 0u, 370u, 522u)]
    [InlineData(4u, 0x4000u, 740u, 1044u)]
    public void GetPanelSizeWritesWidthAndHeight(uint type, uint option, uint width, uint height)
    {
        WriteParam(type, maxTextLength: 8, option);
        Assert.True(_memory.TryWrite(OutAddress, new byte[8]));
        _ctx[CpuRegister.Rdi] = ParamAddress;
        _ctx[CpuRegister.Rsi] = OutAddress;
        _ctx[CpuRegister.Rdx] = OutAddress + 4;

        Assert.Equal(0, ImeDialogExports.ImeDialogGetPanelSize(_ctx));

        var bytes = new byte[8];
        Assert.True(_memory.TryRead(OutAddress, bytes));
        Assert.Equal(width, BinaryPrimitives.ReadUInt32LittleEndian(bytes));
        Assert.Equal(height, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4)));
    }

    [Fact]
    public void GetPanelSizeRejectsBadArguments()
    {
        WriteParam(type: 7, maxTextLength: 8);
        _ctx[CpuRegister.Rdi] = ParamAddress;
        _ctx[CpuRegister.Rsi] = OutAddress;
        _ctx[CpuRegister.Rdx] = OutAddress + 4;
        Assert.Equal(InvalidType, ImeDialogExports.ImeDialogGetPanelSize(_ctx));

        _ctx[CpuRegister.Rdx] = 0;
        Assert.Equal(InvalidAddress, ImeDialogExports.ImeDialogGetPanelSize(_ctx));
        _ctx[CpuRegister.Rdi] = 0;
        _ctx[CpuRegister.Rdx] = OutAddress + 4;
        Assert.Equal(InvalidAddress, ImeDialogExports.ImeDialogGetPanelSize(_ctx));
    }

    [Fact]
    public void GetPanelPositionAndFormDescribesTheOpenDialog()
    {
        _ctx[CpuRegister.Rdi] = OutAddress;
        Assert.Equal(NotInUse, ImeDialogExports.ImeDialogGetPanelPositionAndForm(_ctx));

        // posx 100.0f, posy 50.0f, halign 1 (center), valign 2 (end), type 4 (number).
        WriteParam(type: 4, maxTextLength: 8, posX: 100.0f, posY: 50.0f, hAlign: 1, vAlign: 2);
        _ctx[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(_ctx));

        Assert.True(_memory.TryWrite(OutAddress, Enumerable.Repeat((byte)0x6B, 0x30).ToArray()));
        _ctx[CpuRegister.Rdi] = OutAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogGetPanelPositionAndForm(_ctx));

        var form = new byte[0x30];
        Assert.True(_memory.TryRead(OutAddress, form));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(form));
        Assert.Equal(100.0f, BinaryPrimitives.ReadSingleLittleEndian(form.AsSpan(4)));
        Assert.Equal(50.0f, BinaryPrimitives.ReadSingleLittleEndian(form.AsSpan(8)));
        Assert.Equal(1u, BinaryPrimitives.ReadUInt32LittleEndian(form.AsSpan(0x0C)));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(form.AsSpan(0x10)));
        Assert.Equal(370u, BinaryPrimitives.ReadUInt32LittleEndian(form.AsSpan(0x14)));
        Assert.Equal(522u, BinaryPrimitives.ReadUInt32LittleEndian(form.AsSpan(0x18)));
        // sizeof(SceImePositionAndForm) == 0x1C: nothing past it is touched.
        Assert.Equal(0x6B, form[0x1C]);

        _ctx[CpuRegister.Rdi] = 0;
        Assert.Equal(InvalidAddress, ImeDialogExports.ImeDialogGetPanelPositionAndForm(_ctx));
    }

    private void FinishByPolling()
    {
        Assert.Equal(StatusRunning, ImeDialogExports.ImeDialogGetStatus(_ctx));
        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(_ctx));
    }

    private void Init(uint maxTextLength = 16)
    {
        WriteParam(type: 0, maxTextLength);
        _ctx[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(_ctx));
        Assert.Equal(0UL, _ctx[CpuRegister.Rax]);
    }

    // SceImeDialogParam: type @4, option @0x20, maxTextLength @0x24, inputTextBuffer @0x28,
    // posx @0x30, posy @0x34, halign @0x38, valign @0x3C.
    private void WriteParam(
        uint type,
        uint maxTextLength,
        uint option = 0,
        float posX = 0,
        float posY = 0,
        uint hAlign = 0,
        uint vAlign = 0)
    {
        var param = new byte[0x60];
        BinaryPrimitives.WriteUInt32LittleEndian(param.AsSpan(0x04), type);
        BinaryPrimitives.WriteUInt32LittleEndian(param.AsSpan(0x20), option);
        BinaryPrimitives.WriteUInt32LittleEndian(param.AsSpan(0x24), maxTextLength);
        BinaryPrimitives.WriteUInt64LittleEndian(param.AsSpan(0x28), BufferAddress);
        BinaryPrimitives.WriteSingleLittleEndian(param.AsSpan(0x30), posX);
        BinaryPrimitives.WriteSingleLittleEndian(param.AsSpan(0x34), posY);
        BinaryPrimitives.WriteUInt32LittleEndian(param.AsSpan(0x38), hAlign);
        BinaryPrimitives.WriteUInt32LittleEndian(param.AsSpan(0x3C), vAlign);
        Assert.True(_memory.TryWrite(ParamAddress, param));
    }

    private void WriteBuffer(string text)
    {
        var bytes = Encoding.Unicode.GetBytes(text + "\0");
        Assert.True(_memory.TryWrite(BufferAddress, bytes));
    }

    private string ReadBuffer()
    {
        var bytes = new byte[0x40];
        Assert.True(_memory.TryRead(BufferAddress, bytes));
        var end = 0;
        while (end + 1 < bytes.Length && (bytes[end] != 0 || bytes[end + 1] != 0))
        {
            end += 2;
        }

        return Encoding.Unicode.GetString(bytes, 0, end);
    }
}
