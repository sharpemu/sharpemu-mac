// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Overlay;
using SharpEmu.Libs.Pad;

namespace SharpEmu.Libs.Ime;

public static class ImeDialogExports
{
    private const int StatusNone = 0;
    private const int StatusRunning = 1;
    private const int StatusFinished = 2;

    // SceImeDialogEndStatus.
    internal const uint EndStatusOk = 0;
    internal const uint EndStatusUserCanceled = 1;
    internal const uint EndStatusAborted = 2;

    // SceImeDialogParam (0x60 bytes): userId @0x00, type @0x04, supportedLanguages @0x08,
    // enterLabel @0x10, inputMethod @0x14, filter @0x18, option @0x20, maxTextLength @0x24,
    // inputTextBuffer @0x28, posx @0x30, posy @0x34, horizontalAlignment @0x38,
    // verticalAlignment @0x3C, placeholder @0x40, title @0x48, reserved[16] @0x50.
    private const int ParamSize = 0x60;
    private const ulong ParamTypeOffset = 0x04;
    private const ulong ParamEnterLabelOffset = 0x10;
    private const ulong ParamOptionOffset = 0x20;
    private const ulong ParamMaxTextLengthOffset = 0x24;
    private const ulong ParamInputTextBufferOffset = 0x28;
    private const ulong ParamPosXOffset = 0x30;
    private const ulong ParamPosYOffset = 0x34;
    private const ulong ParamHAlignOffset = 0x38;
    private const ulong ParamVAlignOffset = 0x3C;
    private const ulong ParamPlaceholderOffset = 0x40;
    private const ulong ParamTitleOffset = 0x48;

    private const uint TypeNumber = 4;
    private const uint OptionMultiline = 0x1;
    private const uint OptionNoAutoCapitalization = 0x2;
    private const uint OptionPassword = 0x4;
    private const uint OptionUseOver2K = 0x4000;
    private const int MaxTextLength = 2048;
    private const int MaxLabelLength = 256;

    // SceImeDialogResult { SceImeDialogEndStatus endstatus @0; int8 reserved[12] } = 0x10 bytes.
    internal const int ResultSize = 0x10;

    // SceImePositionAndForm { uint32 type @0; float posx @4; float posy @8; uint32 halign @0xC;
    // uint32 valign @0x10; uint32 width @0x14; uint32 height @0x18 } = 0x1C bytes.
    internal const int PositionAndFormSize = 0x1C;
    private const uint PositionAndFormTypeDialog = 2;

    private const int ImeErrorBusy = unchecked((int)0x80BC0001);
    private const int ImeErrorInvalidType = unchecked((int)0x80BC0011);
    private const int ImeErrorInvalidMaxTextLength = unchecked((int)0x80BC0016);
    private const int ImeErrorInvalidInputTextBuffer = unchecked((int)0x80BC0017);
    private const int ImeErrorInvalidAddress = unchecked((int)0x80BC0031);
    private const int ImeErrorDialogNotRunning = unchecked((int)0x80BC0105);
    private const int ImeErrorDialogNotFinished = unchecked((int)0x80BC0106);
    private const int ImeErrorDialogNotInUse = unchecked((int)0x80BC0107);

    private const string DefaultInputText = "Sharp";
    private const string AutofillVariable = "SHARPEMU_IME_AUTOFILL";

    /// <summary>
    /// Text returned without opening the on-screen keyboard. Unset (the default) opens the
    /// keyboard, unless <c>SHARPEMU_IME_AUTOFILL=1</c> asks for the title's own text instead.
    /// </summary>
    internal const string DefaultTextEnvironmentVariable = "SHARPEMU_IME_DEFAULT_TEXT";

    /// <summary>Status polls after a keyboard-less Init before the dialog finishes.</summary>
    internal const int PollsUntilFinish = 2;

    private static readonly object _gate = new();
    private static int _status = StatusNone;
    private static int _pollsLeft;
    private static uint _endStatus = EndStatusOk;
    private static long _nextSession;
    private static ImeKeyboardSession? _session;
    private static ulong _bufferAddress;
    private static int _maxTextLength;
    private static uint _type;
    private static uint _option;
    private static uint _posX;
    private static uint _posY;
    private static uint _hAlign;
    private static uint _vAlign;

    internal static void ResetForTests()
    {
        lock (_gate)
        {
            if (_session is not null)
            {
                Close();
            }

            _status = StatusNone;
            _pollsLeft = 0;
            _endStatus = EndStatusOk;
        }
    }

    // Panel size for a given type/option (extended parameters are not applied).
    internal static (uint Width, uint Height) ComputePanelSize(uint type, uint option)
    {
        uint width;
        uint height;
        if (type == TypeNumber)
        {
            width = 370;
            height = 522;
        }
        else
        {
            width = 793;
            height = (option & OptionMultiline) != 0 ? 628u : 528u;
        }

        if ((option & OptionUseOver2K) != 0)
        {
            width *= 2;
            height *= 2;
        }

        return (width, height);
    }

    [SysAbiExport(
        Nid = "NUeBrN7hzf0",
        ExportName = "sceImeDialogInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogInit(CpuContext ctx)
    {
        var parameterAddress = ctx[CpuRegister.Rdi];
        if (parameterAddress == 0)
        {
            return SetReturn(ctx, ImeErrorInvalidAddress);
        }

        Span<byte> param = stackalloc byte[ParamSize];
        if (!ctx.Memory.TryRead(parameterAddress, param))
        {
            return SetReturn(ctx, ImeErrorInvalidAddress);
        }

        var type = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamTypeOffset..]);
        if (type > TypeNumber)
        {
            return SetReturn(ctx, ImeErrorInvalidType);
        }

        lock (_gate)
        {
            if (_status != StatusNone)
            {
                return SetReturn(ctx, ImeErrorBusy);
            }

            var maxTextLength = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamMaxTextLengthOffset..]);
            if (maxTextLength is 0 or > MaxTextLength)
            {
                return SetReturn(ctx, ImeErrorInvalidMaxTextLength);
            }

            var bufferAddress = BinaryPrimitives.ReadUInt64LittleEndian(param[(int)ParamInputTextBufferOffset..]);
            if (bufferAddress == 0)
            {
                return SetReturn(ctx, ImeErrorInvalidInputTextBuffer);
            }

            var kind = (ImeTextKind)type;
            var enterLabelValue = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamEnterLabelOffset..]);
            var enterLabel = enterLabelValue <= 3 ? (ImeEnterLabel)enterLabelValue : ImeEnterLabel.Default;
            var option = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamOptionOffset..]);
            var title = ReadGuestString(ctx, parameterAddress + ParamTitleOffset, MaxLabelLength);
            var placeholder = ReadGuestString(ctx, parameterAddress + ParamPlaceholderOffset, MaxLabelLength);
            var initial = ReadUtf16(ctx, bufferAddress, (int)maxTextLength);

            _type = type;
            _option = option;
            _posX = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamPosXOffset..]);
            _posY = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamPosYOffset..]);
            _hAlign = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamHAlignOffset..]);
            _vAlign = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamVAlignOffset..]);
            _bufferAddress = bufferAddress;
            _maxTextLength = (int)maxTextLength;
            _endStatus = EndStatusOk;

            // No keyboard: answer with a configured text, or the title's own text when autofilling.
            // Like the console the dialog reports RUNNING first, then finishes on a later poll.
            var configured = Environment.GetEnvironmentVariable(DefaultTextEnvironmentVariable);
            var autofill = string.Equals(Environment.GetEnvironmentVariable(AutofillVariable), "1", StringComparison.Ordinal);
            if (!string.IsNullOrEmpty(configured) || autofill)
            {
                var text = !string.IsNullOrEmpty(configured)
                    ? configured
                    : string.IsNullOrEmpty(initial) && kind != ImeTextKind.Number ? FallbackText() : initial;
                if (!string.IsNullOrEmpty(configured) || !string.IsNullOrEmpty(text))
                {
                    WriteText(ctx, text);
                }

                _pollsLeft = PollsUntilFinish;
                _status = StatusRunning;
                Console.Error.WriteLine($"[LOADER][INFO] ime_dialog.autofill text=\"{text}\"");
                return SetReturn(ctx, 0);
            }

            if (string.IsNullOrEmpty(initial) && kind != ImeTextKind.Number)
            {
                initial = FallbackText();
            }

            _session = new ImeKeyboardSession(
                Interlocked.Increment(ref _nextSession),
                title,
                placeholder,
                initial,
                (int)maxTextLength,
                kind,
                (option & OptionPassword) != 0,
                (option & OptionNoAutoCapitalization) == 0,
                enterLabel,
                Environment.GetEnvironmentVariable(OverlayText.LanguageVariable));
            _session.Begin();
            SystemDialogInput.Capture(wantsText: true);
            ImeKeyboardLayer.Instance.Show(_session.View());
            _status = StatusRunning;
            Console.Error.WriteLine(
                $"[LOADER][INFO] ime_dialog.open type={kind} max={maxTextLength} option=0x{option:X} title=\"{title}\"");
            return SetReturn(ctx, 0);
        }
    }

    [SysAbiExport(
        Nid = "IADmD4tScBY",
        ExportName = "sceImeDialogGetStatus",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogGetStatus(CpuContext ctx)
    {
        lock (_gate)
        {
            if (_status == StatusRunning && _session is null)
            {
                // Keyboard-less dialog: nobody can type, so it ends on its own at the second poll.
                if (--_pollsLeft <= 0)
                {
                    _status = StatusFinished;
                }
            }
            else if (_status == StatusRunning && _session is { } session)
            {
                switch (session.Poll())
                {
                    case ImeKeyboardInput.Changed:
                        ImeKeyboardLayer.Instance.Show(session.View());
                        break;
                    case ImeKeyboardInput.Done:
                        Finish(EndStatusOk, ctx);
                        break;
                    case ImeKeyboardInput.Cancel:
                        Finish(EndStatusUserCanceled, ctx);
                        break;
                }
            }

            ctx[CpuRegister.Rax] = unchecked((ulong)(long)_status);
            return _status;
        }
    }

    [SysAbiExport(
        Nid = "x01jxu+vxlc",
        ExportName = "sceImeDialogGetResult",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogGetResult(CpuContext ctx)
    {
        var resultAddress = ctx[CpuRegister.Rdi];
        lock (_gate)
        {
            if (_status == StatusNone)
            {
                return SetReturn(ctx, ImeErrorDialogNotInUse);
            }

            if (resultAddress == 0)
            {
                return SetReturn(ctx, ImeErrorInvalidAddress);
            }

            if (_status != StatusFinished)
            {
                return SetReturn(ctx, ImeErrorDialogNotFinished);
            }

            Span<byte> result = stackalloc byte[ResultSize];
            result.Clear();
            BinaryPrimitives.WriteUInt32LittleEndian(result, _endStatus);
            return ctx.Memory.TryWrite(resultAddress, result)
                ? SetReturn(ctx, 0)
                : SetReturn(ctx, ImeErrorInvalidAddress);
        }
    }

    [SysAbiExport(
        Nid = "oBmw4xrmfKs",
        ExportName = "sceImeDialogAbort",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogAbort(CpuContext ctx)
    {
        lock (_gate)
        {
            if (_status == StatusNone)
            {
                return SetReturn(ctx, ImeErrorDialogNotInUse);
            }

            if (_status != StatusRunning)
            {
                return SetReturn(ctx, ImeErrorDialogNotRunning);
            }

            Finish(EndStatusAborted, ctx);
            return SetReturn(ctx, 0);
        }
    }

    [SysAbiExport(
        Nid = "gyTyVn+bXMw",
        ExportName = "sceImeDialogTerm",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogTerm(CpuContext ctx)
    {
        lock (_gate)
        {
            if (_status == StatusNone)
            {
                return SetReturn(ctx, ImeErrorDialogNotInUse);
            }

            if (_status != StatusFinished)
            {
                return SetReturn(ctx, ImeErrorDialogNotFinished);
            }

            if (_session is not null)
            {
                Close();
            }

            _status = StatusNone;
            return SetReturn(ctx, 0);
        }
    }

    // int sceImeDialogGetPanelSize(const SceImeDialogParam *param, uint32_t *width, uint32_t *height).
    [SysAbiExport(
        Nid = "wqsJvRXwl58",
        ExportName = "sceImeDialogGetPanelSize",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogGetPanelSize(CpuContext ctx)
    {
        var paramAddress = ctx[CpuRegister.Rdi];
        var widthAddress = ctx[CpuRegister.Rsi];
        var heightAddress = ctx[CpuRegister.Rdx];
        if (paramAddress == 0 || widthAddress == 0 || heightAddress == 0)
        {
            return SetReturn(ctx, ImeErrorInvalidAddress);
        }

        Span<byte> param = stackalloc byte[ParamSize];
        if (!ctx.Memory.TryRead(paramAddress, param))
        {
            return SetReturn(ctx, ImeErrorInvalidAddress);
        }

        var type = BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamTypeOffset..]);
        if (type > TypeNumber)
        {
            return SetReturn(ctx, ImeErrorInvalidType);
        }

        var (width, height) = ComputePanelSize(type, BinaryPrimitives.ReadUInt32LittleEndian(param[(int)ParamOptionOffset..]));
        Span<byte> value = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(value, width);
        var ok = ctx.Memory.TryWrite(widthAddress, value);
        BinaryPrimitives.WriteUInt32LittleEndian(value, height);
        ok &= ctx.Memory.TryWrite(heightAddress, value);
        return SetReturn(ctx, ok ? 0 : ImeErrorInvalidAddress);
    }

    // int sceImeDialogGetPanelPositionAndForm(SceImePositionAndForm *form): describes the open dialog.
    [SysAbiExport(
        Nid = "8jqzzPioYl8",
        ExportName = "sceImeDialogGetPanelPositionAndForm",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogGetPanelPositionAndForm(CpuContext ctx)
    {
        var formAddress = ctx[CpuRegister.Rdi];
        Span<byte> form = stackalloc byte[PositionAndFormSize];
        form.Clear();
        lock (_gate)
        {
            if (_status == StatusNone)
            {
                return SetReturn(ctx, ImeErrorDialogNotInUse);
            }

            if (formAddress == 0)
            {
                return SetReturn(ctx, ImeErrorInvalidAddress);
            }

            var (width, height) = ComputePanelSize(_type, _option);
            BinaryPrimitives.WriteUInt32LittleEndian(form, PositionAndFormTypeDialog);
            BinaryPrimitives.WriteUInt32LittleEndian(form[0x04..], _posX);
            BinaryPrimitives.WriteUInt32LittleEndian(form[0x08..], _posY);
            BinaryPrimitives.WriteUInt32LittleEndian(form[0x0C..], _hAlign);
            BinaryPrimitives.WriteUInt32LittleEndian(form[0x10..], _vAlign);
            BinaryPrimitives.WriteUInt32LittleEndian(form[0x14..], width);
            BinaryPrimitives.WriteUInt32LittleEndian(form[0x18..], height);
        }

        return ctx.Memory.TryWrite(formAddress, form)
            ? SetReturn(ctx, 0)
            : SetReturn(ctx, ImeErrorInvalidAddress);
    }

    private static void Finish(uint endStatus, CpuContext ctx)
    {
        _endStatus = endStatus;
        if (endStatus == EndStatusOk && _session is { } session)
        {
            WriteText(ctx, session.Text);
        }

        Console.Error.WriteLine($"[LOADER][INFO] ime_dialog.finish end={endStatus}");
        Close();
        _status = StatusFinished;
    }

    private static void Close()
    {
        if (_session is null)
        {
            return;
        }

        _session = null;
        ImeKeyboardLayer.Instance.Hide();
        SystemDialogInput.Release();
    }

    private static void WriteText(CpuContext ctx, string text)
    {
        if (text.Length > _maxTextLength)
        {
            text = text[.._maxTextLength];
        }

        var payload = new byte[(text.Length + 1) * 2];
        Encoding.Unicode.GetBytes(text, payload);
        ctx.Memory.TryWrite(_bufferAddress, payload);
    }

    private static string FallbackText()
    {
        var text = Environment.GetEnvironmentVariable("SHARPEMU_DEFAULT_PROFILE");
        return string.IsNullOrWhiteSpace(text) ? DefaultInputText : text.Trim();
    }

    private static string ReadGuestString(CpuContext ctx, ulong pointerAddress, int maxLength) =>
        ctx.TryReadUInt64(pointerAddress, out var address) && address != 0
            ? ReadUtf16(ctx, address, maxLength)
            : string.Empty;

    private static string ReadUtf16(CpuContext ctx, ulong address, int maxLength)
    {
        var builder = new StringBuilder();
        Span<byte> unit = stackalloc byte[2];
        for (var index = 0; index < maxLength; index++)
        {
            if (!ctx.Memory.TryRead(address + (ulong)(index * 2), unit))
            {
                break;
            }

            var value = BinaryPrimitives.ReadUInt16LittleEndian(unit);
            if (value == 0)
            {
                break;
            }

            builder.Append((char)value);
        }

        return builder.ToString();
    }

    private static int SetReturn(CpuContext ctx, int result)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)(long)result);
        return result;
    }
}
