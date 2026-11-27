// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.Overlay;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Text;

namespace SharpEmu.Libs.SaveData;

// libSceSaveDataDialog. A save list is shown as an overlay the user drives (SaveDataListLayer)
// unless a default button is pinned (SHARPEMU_SAVEDATA_DIALOG_BUTTON); every other dialog has no host UI. Status follows the common dialog machine
// NONE -> INITIALIZED -> RUNNING -> FINISHED (-> RUNNING on the next Open) and back to NONE
// on Terminate. A dialog that offers the user a choice finishes by itself once the guest
// has seen it RUNNING, as if the user pressed the default button (configurable, see
// DefaultChoice); a dialog without buttons (progress bar, user message with no buttons,
// progress system message) stays RUNNING until the guest closes it. The result reports a
// selection consistent with the saves that exist on the host.
internal sealed record SaveDataDialogSlot(
    string DirName,
    string SlotDir,
    SaveDataMetadata Metadata,
    DateTime LastWriteUtc,
    long SizeBytes,
    string? IconPath);

public static class SaveDataDialogExports
{
    private const int StatusNone = 0;
    private const int StatusInitialized = 1;
    private const int StatusRunning = 2;
    private const int StatusFinished = 3;

    // SCE_COMMON_DIALOG_ERROR_*.
    private const int ErrorOk = 0;
    private const int ErrorNotInitialized = unchecked((int)0x80B80003);
    private const int ErrorAlreadyInitialized = unchecked((int)0x80B80004);
    private const int ErrorNotFinished = unchecked((int)0x80B80005);
    private const int ErrorBusy = unchecked((int)0x80B80008);
    private const int ErrorParamInvalid = unchecked((int)0x80B8000A);
    private const int ErrorNotRunning = unchecked((int)0x80B8000B);
    private const int ErrorArgNull = unchecked((int)0x80B8000D);

    // SceSaveDataDialogMode.
    private const int ModeList = 1;
    private const int DispTypeSave = 1;
    private const int DispTypeLoad = 2;
    private const int DispTypeDelete = 3;
    private const ulong ParamOptionOffset = 0x78;
    private const uint OptionBackDisabled = 1;
    private const int NewItemTitleMaxBytes = 128;
    private const int ModeUserMsg = 2;
    private const int ModeSystemMsg = 3;
    private const int ModeErrorCode = 4;
    private const int ModeProgressBar = 5;

    // SceCommonDialogResult and SceSaveDataDialogButtonId (YES shares OK's value).
    private const int ResultOk = 0;
    private const int ResultUserCanceled = 1;
    private const int ButtonIdInvalid = 0;
    private const int ButtonIdOk = 1;
    private const int ButtonIdYes = 1;
    private const int ButtonIdNo = 2;

    // SceSaveDataDialogButtonType (user message).
    private const uint ButtonTypeOk = 0;
    private const uint ButtonTypeYesNo = 1;
    private const uint ButtonTypeNone = 2;
    private const uint ButtonTypeOkCancel = 3;

    // SceSaveDataDialogSystemMessageType values that change the buttons on offer.
    private const uint SysMsgConfirm = 2;
    private const uint SysMsgOverwrite = 3;
    private const uint SysMsgProgress = 5;
    private const uint SysMsgCorruptedAndDelete = 10;
    private const uint SysMsgCorruptedAndCreate = 11;
    private const uint SysMsgCorruptedAndRestore = 13;

    // SceSaveDataDialogFocusPos.
    private const uint FocusListHead = 0;
    private const uint FocusListTail = 1;
    private const uint FocusDataHead = 2;
    private const uint FocusDataTail = 3;
    private const uint FocusDataLatest = 4;
    private const uint FocusDataOldest = 5;
    private const uint FocusDirName = 6;

    // SceSaveDataDialogParam: CommonDialog base param (0x30), size @0x30, mode @0x34,
    // dispType @0x38, then pointers animParam @0x40, items @0x48, userMsgParam @0x50,
    // sysMsgParam @0x58, errorCodeParam @0x60, progBarParam @0x68, userData @0x70. KytyPS5
    // and anyps5 agree on this layout.
    private const ulong ParamModeOffset = 0x34;
    private const ulong ParamDispTypeOffset = 0x38;
    private const ulong ParamItemsOffset = 0x48;
    private const ulong ParamUserMsgOffset = 0x50;
    private const ulong ParamSysMsgOffset = 0x58;
    private const ulong ParamUserDataOffset = 0x70;

    // SceSaveDataDialogItems: userId @0x00, titleId* @0x08, dirName* @0x10, dirNameNum @0x18
    // (both references stop here), then newItem* @0x20, focusPos @0x28, focusPosDirName*
    // @0x30 as in the SDK header; the trailing fields are read leniently.
    private const ulong ItemsTitleIdOffset = 0x08;
    private const ulong ItemsDirNameOffset = 0x10;
    private const ulong ItemsDirNameNumOffset = 0x18;
    private const ulong ItemsNewItemOffset = 0x20;
    private const ulong ItemsFocusPosOffset = 0x28;
    private const ulong ItemsFocusPosDirNameOffset = 0x30;
    private const uint DirNameMaxCount = 1024;

    // SceSaveDataDialogResult (0x48): mode @0x00, result @0x04, buttonId @0x08, dirName*
    // @0x10, param* @0x18, userData @0x20, reserved[32] @0x28. dirName and param point at
    // caller buffers that GetResult fills; the pointers themselves are inputs.
    private const int ResultSize = 0x48;
    private const ulong ResultDirNameOffset = 0x10;
    private const ulong ResultParamOffset = 0x18;
    private const int DirNameSize = 32;
    private const int TitleIdSize = 10;
    private const int SaveDataParamSize = 0x530;

    private enum Buttons
    {
        None,
        Ok,
        YesNo,
        OkCancel,
        List,
    }

    public enum DefaultChoice
    {
        Affirmative,
        Negative,
    }

    private sealed record DialogItems(
        int UserId,
        string? TitleId,
        string[] DirNames,
        bool HasNewItem,
        uint FocusPos,
        string? FocusDirName);

    private sealed record OpenedDialog(int Mode, int DispType, ulong UserData, DialogItems? Items, Buttons Buttons);

    private sealed record Outcome(int Result, int ButtonId, string? DirName, string? SlotDir, DateTime LastWriteUtc);

    private static readonly object _gate = new();
    private static ListSession? _list;
    private static long _nextSession;
    private static int _status;
    private static OpenedDialog? _dialog;
    private static bool _runningObserved;
    private static Outcome? _outcome;

    // Tests (or a front end) may pin the button an unattended dialog takes; otherwise
    // SHARPEMU_SAVEDATA_DIALOG_BUTTON=no|cancel picks the negative one and anything else
    // the affirmative one (YES / OK, or the focused list item).
    internal static DefaultChoice? DefaultChoiceOverride { get; set; }

    private static DefaultChoice CurrentDefaultChoice()
    {
        if (DefaultChoiceOverride is { } pinned)
        {
            return pinned;
        }

        var configured = Environment.GetEnvironmentVariable("SHARPEMU_SAVEDATA_DIALOG_BUTTON");
        return string.Equals(configured, "no", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(configured, "cancel", StringComparison.OrdinalIgnoreCase)
            ? DefaultChoice.Negative
            : DefaultChoice.Affirmative;
    }

    // The list is shown to the user unless an unattended choice was pinned.
    private static bool ListIsInteractive() =>
        DefaultChoiceOverride is null &&
        string.IsNullOrEmpty(Environment.GetEnvironmentVariable("SHARPEMU_SAVEDATA_DIALOG_BUTTON"));

    [SysAbiExport(
        Nid = "s9e3+YpRnzw",
        ExportName = "sceSaveDataDialogInitialize",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogInitialize(CpuContext ctx)
    {
        // TODO(CommonDialog): once CommonDialog tracks sceCommonDialogInitialize, return
        // SCE_COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED here when it has not been called, and
        // report this dialog through sceCommonDialogIsUsed while it is RUNNING.
        lock (_gate)
        {
            // anyps5 rejects every re-initialization until Terminate; KytyPS5 accepts any.
            // The SDK requires Terminate first, so a second Initialize is an error.
            if (_status != StatusNone)
            {
                return ctx.SetReturn(ErrorAlreadyInitialized);
            }

            ResetLocked(StatusInitialized);
        }

        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "4tPhsP6FpDI",
        ExportName = "sceSaveDataDialogOpen",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogOpen(CpuContext ctx)
    {
        var paramAddress = ctx[CpuRegister.Rdi];
        lock (_gate)
        {
            // State is checked before the argument, as anyps5 does. anyps5 answers every bad
            // state with INVALID_STATE; we keep the specific NOT_INITIALIZED / BUSY codes.
            if (_status == StatusNone)
            {
                return ctx.SetReturn(ErrorNotInitialized);
            }

            if (_status == StatusRunning)
            {
                return ctx.SetReturn(ErrorBusy);
            }

            if (paramAddress == 0)
            {
                return ctx.SetReturn(ErrorArgNull);
            }

            if (!TryReadUInt32(ctx, paramAddress + ParamModeOffset, out var modeBits) ||
                !TryReadUInt32(ctx, paramAddress + ParamDispTypeOffset, out var dispType) ||
                !ctx.TryReadUInt64(paramAddress + ParamItemsOffset, out var itemsAddress) ||
                !ctx.TryReadUInt64(paramAddress + ParamUserMsgOffset, out var userMsgAddress) ||
                !ctx.TryReadUInt64(paramAddress + ParamSysMsgOffset, out var sysMsgAddress) ||
                !ctx.TryReadUInt64(paramAddress + ParamUserDataOffset, out var userData))
            {
                return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }

            var mode = unchecked((int)modeBits);
            if (mode is < ModeList or > ModeProgressBar)
            {
                return ctx.SetReturn(ErrorParamInvalid);
            }

            var items = itemsAddress == 0 ? null : ReadItems(ctx, itemsAddress);
            var buttons = mode switch
            {
                ModeList => Buttons.List,
                ModeUserMsg => UserMessageButtons(ctx, userMsgAddress),
                ModeSystemMsg => SystemMessageButtons(ctx, sysMsgAddress),
                ModeErrorCode => Buttons.Ok,
                _ => Buttons.None,
            };

            _dialog = new OpenedDialog(mode, unchecked((int)dispType), userData, items, buttons);
            _outcome = null;
            _runningObserved = false;
            _status = StatusRunning;
            TraceSaveDataDialog(
                $"open mode={mode} disp={dispType} buttons={buttons} dirs={items?.DirNames.Length ?? 0} " +
                $"new_item={items?.HasNewItem ?? false} focus={items?.FocusPos ?? 0} userData=0x{userData:X16}");

            if (mode == ModeList && ListIsInteractive())
            {
                var backEnabled = !(ctx.TryReadUInt64(paramAddress + ParamOptionOffset, out var optionAddress) &&
                                    optionAddress != 0 &&
                                    ctx.TryReadUInt32(optionAddress, out var back) &&
                                    back == OptionBackDisabled);
                var listItems = ReadListItems(ctx, paramAddress, unchecked((int)dispType));
                if (listItems.Slots.Count > 0 || listItems.NewItemTitle is not null)
                {
                    _list = new ListSession(
                        Interlocked.Increment(ref _nextSession),
                        BuildEntries(listItems),
                        CaptionFor(unchecked((int)dispType)),
                        backEnabled,
                        unchecked((int)dispType) is DispTypeLoad or DispTypeSave);
                    _list.Focus = InitialFocus(_list.Entries, listItems);
                    _list.Begin();
                    SystemDialogInput.Capture();
                    SaveDataListLayer.Instance.Show(_list.View());
                    Console.Error.WriteLine(
                        $"[LOADER][INFO] save_data_dialog.list type={dispType} items={_list.Entries.Count} focus={_list.Focus}");
                }
            }
        }

        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "ERKzksauAJA",
        ExportName = "sceSaveDataDialogGetStatus",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogGetStatus(CpuContext ctx) => ctx.SetReturn(PollStatus());

    [SysAbiExport(
        Nid = "KK3Bdg1RWK0",
        ExportName = "sceSaveDataDialogUpdateStatus",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogUpdateStatus(CpuContext ctx) => ctx.SetReturn(PollStatus());

    // The first poll after Open reports RUNNING (KytyPS5 notes titles that need to see it);
    // the next one presses the default button. Both GetStatus and UpdateStatus advance it so
    // that no polling pattern spins on RUNNING forever.
    private static int PollStatus()
    {
        lock (_gate)
        {
            if (_status == StatusRunning && _list is { } list)
            {
                switch (list.Poll())
                {
                    case ListInput.Changed:
                        SaveDataListLayer.Instance.Show(list.View());
                        break;
                    case ListInput.Delete:
                        DeleteFocusedEntry(list);
                        break;
                    case ListInput.Choose:
                        var entry = list.Entries[list.Focus];
                        FinishList(ResultOk, ButtonIdOk, entry.DirName, entry.Slot);
                        Console.Error.WriteLine(
                            $"[LOADER][INFO] save_data_dialog.chosen dir='{entry.DirName}'");
                        break;
                    case ListInput.Back:
                        FinishList(ResultUserCanceled, ButtonIdInvalid, null, null);
                        Console.Error.WriteLine("[LOADER][INFO] save_data_dialog.canceled");
                        break;
                }
            }
            else if (_status == StatusRunning && _dialog is { Buttons: not Buttons.None } dialog)
            {
                if (_runningObserved)
                {
                    FinishLocked(DecideOutcome(dialog, CurrentDefaultChoice()));
                }
                else
                {
                    _runningObserved = true;
                }
            }

            return _status;
        }
    }

    [SysAbiExport(
        Nid = "en7gNVnh878",
        ExportName = "sceSaveDataDialogIsReadyToDisplay",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogIsReadyToDisplay(CpuContext ctx) => ctx.SetReturn(1);

    [SysAbiExport(
        Nid = "yEiJ-qqr6Cg",
        ExportName = "sceSaveDataDialogGetResult",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogGetResult(CpuContext ctx)
    {
        var resultAddress = ctx[CpuRegister.Rdi];
        OpenedDialog dialog;
        Outcome outcome;
        lock (_gate)
        {
            // NOT_FINISHED is checked before the null argument, as anyps5 does.
            if (_status != StatusFinished || _dialog is null || _outcome is null)
            {
                return ctx.SetReturn(ErrorNotFinished);
            }

            dialog = _dialog;
            outcome = _outcome;
        }

        if (resultAddress == 0)
        {
            return ctx.SetReturn(ErrorArgNull);
        }

        if (!ctx.TryReadUInt64(resultAddress + ResultDirNameOffset, out var dirNameAddress) ||
            !ctx.TryReadUInt64(resultAddress + ResultParamOffset, out var paramAddress))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        Span<byte> result = stackalloc byte[ResultSize];
        result.Clear();
        BinaryPrimitives.WriteInt32LittleEndian(result[0x00..], dialog.Mode);
        BinaryPrimitives.WriteInt32LittleEndian(result[0x04..], outcome.Result);
        BinaryPrimitives.WriteInt32LittleEndian(result[0x08..], outcome.ButtonId);
        BinaryPrimitives.WriteUInt64LittleEndian(result[(int)ResultDirNameOffset..], dirNameAddress);
        BinaryPrimitives.WriteUInt64LittleEndian(result[(int)ResultParamOffset..], paramAddress);
        BinaryPrimitives.WriteUInt64LittleEndian(result[0x20..], dialog.UserData);
        if (!ctx.Memory.TryWrite(resultAddress, result))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        // The selected directory, or an empty name when nothing (or the new item) was chosen.
        if (dirNameAddress != 0)
        {
            Span<byte> dirName = stackalloc byte[DirNameSize];
            dirName.Clear();
            WriteAscii(dirName[..(DirNameSize - 1)], outcome.DirName ?? string.Empty);
            if (!ctx.Memory.TryWrite(dirNameAddress, dirName))
            {
                return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }
        }

        // The selected save's parameters, zeroed when the selection has no save on the host.
        if (paramAddress != 0)
        {
            var param = outcome.SlotDir is { } slotDir && outcome.DirName is { } selected
                ? SaveDataExports.BuildSaveParam(slotDir, selected, outcome.LastWriteUtc)
                : new byte[SaveDataParamSize];
            if (!ctx.Memory.TryWrite(paramAddress, param))
            {
                return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }
        }

        TraceSaveDataDialog(
            $"get_result mode={dialog.Mode} result={outcome.Result} button={outcome.ButtonId} dir='{outcome.DirName}'");
        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "fH46Lag88XY",
        ExportName = "sceSaveDataDialogClose",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogClose(CpuContext ctx)
    {
        lock (_gate)
        {
            if (_status != StatusRunning || _dialog is null)
            {
                return ctx.SetReturn(ErrorNotRunning);
            }

            if (_list is not null)
            {
                FinishList(ResultUserCanceled, ButtonIdInvalid, null, null);
                return ctx.SetReturn(ErrorOk);
            }

            // Closed by the application, not by a button: result OK, no button. Both
            // references report button 1 for every dialog; no button was pressed here.
            var fixedSave = _dialog.Mode == ModeList ? null : FixedSave(_dialog.Items);
            FinishLocked(new Outcome(
                ResultOk,
                ButtonIdInvalid,
                fixedSave?.DirName,
                fixedSave?.SlotDir,
                fixedSave?.LastWriteUtc ?? default));
        }

        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "YuH2FA7azqQ",
        ExportName = "sceSaveDataDialogTerminate",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogTerminate(CpuContext ctx)
    {
        lock (_gate)
        {
            if (_status == StatusNone)
            {
                return ctx.SetReturn(ErrorNotInitialized);
            }

            // A running dialog is torn down with the library, as both references do.
            ResetLocked(StatusNone);
        }

        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "V-uEeFKARJU",
        ExportName = "sceSaveDataDialogProgressBarInc",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogProgressBarInc(CpuContext ctx) => ctx.SetReturn(ErrorOk);

    [SysAbiExport(
        Nid = "hay1CfTmLyA",
        ExportName = "sceSaveDataDialogProgressBarSetValue",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceSaveDataDialog")]
    public static int SaveDataDialogProgressBarSetValue(CpuContext ctx) => ctx.SetReturn(ErrorOk);

    private static void ResetLocked(int status)
    {
        if (_list is not null)
        {
            EndList();
        }

        _status = status;
        _dialog = null;
        _outcome = null;
        _runningObserved = false;
    }

    private static void FinishLocked(Outcome outcome)
    {
        _outcome = outcome;
        _status = StatusFinished;
        TraceSaveDataDialog($"finish result={outcome.Result} button={outcome.ButtonId} dir='{outcome.DirName}'");
    }

    private static Outcome DecideOutcome(OpenedDialog dialog, DefaultChoice choice)
    {
        var affirmative = choice == DefaultChoice.Affirmative;
        if (dialog.Buttons == Buttons.List)
        {
            return DecideListOutcome(dialog.Items, affirmative);
        }

        // Message, error and progress dialogs name at most one ("fixed") save: the first
        // non-empty directory of their items, as both references report.
        var fixedSave = FixedSave(dialog.Items);
        var dirName = fixedSave?.DirName;
        var slotDir = fixedSave?.SlotDir;
        var lastWrite = fixedSave?.LastWriteUtc ?? default;
        return dialog.Buttons switch
        {
            Buttons.YesNo => new Outcome(ResultOk, affirmative ? ButtonIdYes : ButtonIdNo, dirName, slotDir, lastWrite),
            Buttons.OkCancel when !affirmative => new Outcome(ResultUserCanceled, ButtonIdInvalid, dirName, slotDir, lastWrite),
            _ => new Outcome(ResultOk, ButtonIdOk, dirName, slotDir, lastWrite),
        };
    }

    // A save list shows only the listed directories that exist (plus the new-data item when
    // the title supplies one) and starts on the focused entry. Confirming picks that entry;
    // a list with nothing to show is the "no saved data" screen, which the user can only
    // leave, so it, like a cancel, ends USER_CANCELED with no button and no directory.
    private static Outcome DecideListOutcome(DialogItems? items, bool affirmative)
    {
        var canceled = new Outcome(ResultUserCanceled, ButtonIdInvalid, null, null, default);
        if (items is null || !affirmative)
        {
            return canceled;
        }

        var visible = new List<(string DirName, string SlotDir, DateTime LastWriteUtc)>();
        foreach (var name in items.DirNames.Distinct(StringComparer.Ordinal))
        {
            if (SaveDataExports.TryFindSave(items.UserId, items.TitleId, name, out var slotDir, out var lastWrite))
            {
                visible.Add((name, slotDir, lastWrite));
            }
        }

        if (visible.Count == 0 && !items.HasNewItem)
        {
            return canceled;
        }

        var newItem = new Outcome(ResultOk, ButtonIdOk, string.Empty, null, default);
        if (visible.Count == 0)
        {
            return newItem;
        }

        // The new-data item heads the list, so LISTHEAD (and an unmatched focus) lands on it.
        var listHead = items.HasNewItem ? -1 : 0;
        var focused = items.FocusPos switch
        {
            FocusListTail or FocusDataTail => visible.Count - 1,
            FocusDataHead => 0,
            FocusDataLatest => visible.IndexOf(visible.MaxBy(entry => entry.LastWriteUtc)),
            FocusDataOldest => visible.IndexOf(visible.MinBy(entry => entry.LastWriteUtc)),
            FocusDirName => visible.FindIndex(entry =>
                string.Equals(entry.DirName, items.FocusDirName, StringComparison.Ordinal)) is var match and >= 0
                ? match
                : listHead,
            _ => listHead,
        };

        if (focused < 0)
        {
            return newItem;
        }

        var (selected, selectedSlot, selectedWrite) = visible[focused];
        return new Outcome(ResultOk, ButtonIdOk, selected, selectedSlot, selectedWrite);
    }

    private static (string DirName, string? SlotDir, DateTime LastWriteUtc)? FixedSave(DialogItems? items)
    {
        var dirName = items?.DirNames.FirstOrDefault(name => name.Length != 0);
        if (items is null || dirName is null)
        {
            return null;
        }

        return SaveDataExports.TryFindSave(items.UserId, items.TitleId, dirName, out var slotDir, out var lastWrite)
            ? (dirName, slotDir, lastWrite)
            : (dirName, null, default);
    }

    private static DialogItems? ReadItems(CpuContext ctx, ulong address)
    {
        if (!TryReadInt32(ctx, address, out var userId) ||
            !ctx.TryReadUInt64(address + ItemsTitleIdOffset, out var titleIdAddress) ||
            !ctx.TryReadUInt64(address + ItemsDirNameOffset, out var dirNamesAddress) ||
            !TryReadUInt32(ctx, address + ItemsDirNameNumOffset, out var dirNameNum))
        {
            return null;
        }

        string? titleId = null;
        if (titleIdAddress != 0 && TryReadFixedAscii(ctx, titleIdAddress, TitleIdSize, out var title) && title.Length != 0)
        {
            titleId = title;
        }

        var names = new List<string>();
        if (dirNamesAddress != 0)
        {
            var count = Math.Min(dirNameNum, DirNameMaxCount);
            for (var i = 0u; i < count; i++)
            {
                if (!TryReadFixedAscii(ctx, dirNamesAddress + (i * DirNameSize), DirNameSize, out var name))
                {
                    break;
                }

                names.Add(name);
            }
        }

        var hasNewItem = ctx.TryReadUInt64(address + ItemsNewItemOffset, out var newItemAddress) && newItemAddress != 0;
        var focusPos = TryReadUInt32(ctx, address + ItemsFocusPosOffset, out var focus) ? focus : FocusListHead;
        string? focusDirName = null;
        if (focusPos == FocusDirName &&
            ctx.TryReadUInt64(address + ItemsFocusPosDirNameOffset, out var focusDirNameAddress) &&
            focusDirNameAddress != 0 &&
            TryReadFixedAscii(ctx, focusDirNameAddress, DirNameSize, out var focusName))
        {
            focusDirName = focusName;
        }

        return new DialogItems(userId, titleId, names.ToArray(), hasNewItem, focusPos, focusDirName);
    }

    // SceSaveDataDialogUserMessageParam: buttonType @0x00.
    private static Buttons UserMessageButtons(CpuContext ctx, ulong address)
    {
        if (address == 0 || !TryReadUInt32(ctx, address, out var buttonType))
        {
            return Buttons.Ok;
        }

        return buttonType switch
        {
            ButtonTypeYesNo => Buttons.YesNo,
            ButtonTypeNone => Buttons.None,
            ButtonTypeOkCancel => Buttons.OkCancel,
            ButtonTypeOk => Buttons.Ok,
            _ => Buttons.Ok,
        };
    }

    // SceSaveDataDialogSystemMessageParam: sysMsgType @0x00. Questions take YES/NO, the
    // progress message has no button, every other message only acknowledges with OK.
    private static Buttons SystemMessageButtons(CpuContext ctx, ulong address)
    {
        if (address == 0 || !TryReadUInt32(ctx, address, out var sysMsgType))
        {
            return Buttons.Ok;
        }

        return sysMsgType switch
        {
            SysMsgConfirm or SysMsgOverwrite or SysMsgCorruptedAndDelete or
                SysMsgCorruptedAndCreate or SysMsgCorruptedAndRestore => Buttons.YesNo,
            SysMsgProgress => Buttons.None,
            _ => Buttons.Ok,
        };
    }

    private static void FinishList(int result, int buttonId, string? dirName, SaveDataDialogSlot? slot)
    {
        FinishLocked(new Outcome(result, buttonId, dirName, slot?.SlotDir, slot?.LastWriteUtc ?? default));
        EndList();
    }

    private static void EndList()
    {
        _list = null;
        SaveDataListLayer.Instance.Hide();
        SystemDialogInput.Release();
    }

    private static ListItems ReadListItems(CpuContext ctx, ulong paramAddress, int dispType)
    {
        var items = new ListItems();
        if (!ctx.TryReadUInt64(paramAddress + 0x48, out var itemsAddress) || itemsAddress == 0)
        {
            return items;
        }

        var userId = TryReadInt32(ctx, itemsAddress, out var user) ? user : 0;
        string? titleId = null;
        if (ctx.TryReadUInt64(itemsAddress + 0x08, out var titleIdAddress) && titleIdAddress != 0 &&
            TryReadString(ctx, titleIdAddress, TitleIdSize, out var title) && title.Length != 0)
        {
            titleId = title;
        }

        if (ctx.TryReadUInt64(itemsAddress + 0x10, out var dirNamesAddress) && dirNamesAddress != 0 &&
            ctx.TryReadUInt32(itemsAddress + 0x18, out var dirNameCount))
        {
            var count = (int)Math.Min(dirNameCount, DirNameMaxCount);
            for (var index = 0; index < count; index++)
            {
                if (!TryReadString(ctx, dirNamesAddress + (ulong)(index * DirNameSize), DirNameSize, out var name) ||
                    name.Length == 0)
                {
                    continue;
                }

                items.FirstDirName ??= name;
                if (items.Slots.Exists(existing => existing.DirName == name))
                {
                    continue;
                }

                if (SaveDataExports.DescribeDialogSlot(userId, titleId, name) is { } slot)
                {
                    items.Slots.Add(slot);
                }
            }
        }

        if (dispType == DispTypeSave &&
            ctx.TryReadUInt64(itemsAddress + 0x20, out var newItemAddress) && newItemAddress != 0)
        {
            items.NewItemTitle =
                ctx.TryReadUInt64(newItemAddress, out var newTitleAddress) && newTitleAddress != 0 &&
                TryReadString(ctx, newTitleAddress, NewItemTitleMaxBytes, out var newTitle) && newTitle.Length != 0
                    ? newTitle
                    : string.Empty;
        }

        items.FocusPosition = ctx.TryReadUInt32(itemsAddress + 0x28, out var focus) ? focus : 0;
        if (items.FocusPosition == FocusDirName &&
            ctx.TryReadUInt64(itemsAddress + 0x30, out var focusNameAddress) && focusNameAddress != 0 &&
            TryReadString(ctx, focusNameAddress, DirNameSize, out var focusName))
        {
            items.FocusDirName = focusName;
        }

        return items;
    }

    private static List<ListEntry> BuildEntries(ListItems items)
    {
        var entries = new List<ListEntry>(items.Slots.Count + 1);
        if (items.NewItemTitle is not null)
        {
            var title = items.NewItemTitle.Length != 0 ? items.NewItemTitle : SaveDataListLayer.Text("NewItem");
            entries.Add(new ListEntry(string.Empty, null,
                new SaveDataListItem(string.Empty, title, SaveDataListLayer.Text("NewItemDetail"), string.Empty, string.Empty,
                    null, IsNew: true)));
        }

        foreach (var slot in items.Slots)
        {
            entries.Add(new ListEntry(slot.DirName, slot, DescribeEntry(slot)));
        }

        return entries;
    }

    private static SaveDataListItem DescribeEntry(SaveDataDialogSlot slot)
    {
        var metadata = slot.Metadata;
        var hasSubTitle = !string.IsNullOrWhiteSpace(metadata.SubTitle);
        var hasTitle = !string.IsNullOrWhiteSpace(metadata.Title);
        var title = hasSubTitle ? metadata.SubTitle : hasTitle ? metadata.Title : slot.DirName;
        var overline = hasSubTitle && hasTitle ? metadata.Title : string.Empty;
        var detail = !string.IsNullOrWhiteSpace(metadata.Detail)
            ? metadata.Detail
            : title == slot.DirName ? string.Empty : slot.DirName;
        return new SaveDataListItem(
            overline,
            title,
            detail,
            OverlayText.FormatDate(slot.LastWriteUtc),
            OverlayText.FormatSize(slot.SizeBytes),
            slot.IconPath,
            IsNew: false);
    }

    private static int InitialFocus(IReadOnlyList<ListEntry> entries, ListItems items)
    {
        var dataStart = items.NewItemTitle is null ? 0 : 1;
        switch (items.FocusPosition)
        {
            case FocusListTail:
            case FocusDataTail:
                return entries.Count - 1;
            case FocusDataHead:
                return Math.Min(dataStart, entries.Count - 1);
            case FocusDataLatest:
            case FocusDataOldest:
                var pick = -1;
                for (var index = dataStart; index < entries.Count; index++)
                {
                    var candidate = entries[index].Slot!.LastWriteUtc;
                    if (pick < 0 ||
                        (items.FocusPosition == FocusDataLatest
                            ? candidate > entries[pick].Slot!.LastWriteUtc
                            : candidate < entries[pick].Slot!.LastWriteUtc))
                    {
                        pick = index;
                    }
                }

                return Math.Max(0, pick);
            case FocusDirName:
                for (var index = 0; index < entries.Count; index++)
                {
                    if (string.Equals(entries[index].DirName, items.FocusDirName, StringComparison.Ordinal))
                    {
                        return index;
                    }
                }

                return 0;
            default:
                return 0;
        }
    }

    private static string CaptionFor(int dispType) => dispType switch
    {
        DispTypeSave => SaveDataListLayer.Text("Caption.Save"),
        DispTypeLoad => SaveDataListLayer.Text("Caption.Load"),
        DispTypeDelete => SaveDataListLayer.Text("Caption.Delete"),
        _ => SaveDataListLayer.Text("Caption.Other"),
    };

    private static void DeleteFocusedEntry(ListSession list)
    {
        var entry = list.Entries[list.Focus];
        if (entry.Slot is null || !SaveDataExports.TryDeleteDialogSlot(entry.Slot))
        {
            list.CancelDelete(SaveDataListLayer.Text("InUse"));
            SaveDataListLayer.Instance.Show(list.View());
            Console.Error.WriteLine($"[LOADER][WARN] save_data_dialog.delete_refused dir='{entry.DirName}'");
            return;
        }

        Console.Error.WriteLine($"[LOADER][INFO] save_data_dialog.deleted dir='{entry.DirName}'");
        list.Remove(list.Focus);
        if (list.Entries.Count == 0)
        {
            FinishList(ResultUserCanceled, ButtonIdInvalid, null, null);
            return;
        }

        SaveDataListLayer.Instance.Show(list.View());
    }

    private static bool TryReadString(CpuContext ctx, ulong address, int maxBytes, out string value)
    {
        value = string.Empty;
        Span<byte> bytes = stackalloc byte[maxBytes];
        var length = 0;
        Span<byte> one = stackalloc byte[1];
        while (length < maxBytes)
        {
            if (!ctx.Memory.TryRead(address + (ulong)length, one))
            {
                return false;
            }

            if (one[0] == 0)
            {
                break;
            }

            bytes[length++] = one[0];
        }

        value = Encoding.UTF8.GetString(bytes[..length]);
        return true;
    }

    private static bool TryReadInt32(CpuContext ctx, ulong address, out int value)
    {
        var ok = TryReadUInt32(ctx, address, out var raw);
        value = unchecked((int)raw);
        return ok;
    }

    private static bool TryReadUInt32(CpuContext ctx, ulong address, out uint value)
    {
        value = 0;
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (!ctx.Memory.TryRead(address, bytes))
        {
            return false;
        }

        value = BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }

    private static bool TryReadFixedAscii(CpuContext ctx, ulong address, int length, out string value)
    {
        value = string.Empty;
        Span<byte> buffer = stackalloc byte[length];
        if (!ctx.Memory.TryRead(address, buffer))
        {
            return false;
        }

        var end = buffer.IndexOf((byte)0);
        value = Encoding.ASCII.GetString(end < 0 ? buffer : buffer[..end]);
        return true;
    }

    private static void WriteAscii(Span<byte> destination, string value)
    {
        var count = Math.Min(value.Length, destination.Length);
        for (var i = 0; i < count; i++)
        {
            var ch = value[i];
            destination[i] = ch <= 0x7F ? (byte)ch : (byte)'?';
        }
    }

    private static void TraceSaveDataDialog(string message)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_SAVEDATA"), "1", StringComparison.Ordinal))
        {
            return;
        }

        Console.Error.WriteLine($"[LOADER][TRACE] save_data_dialog.{message}");
    }

    private sealed class ListItems
    {
        public string? FirstDirName { get; set; }
        public List<SaveDataDialogSlot> Slots { get; } = [];
        public string? NewItemTitle { get; set; }
        public uint FocusPosition { get; set; }
        public string? FocusDirName { get; set; }
    }

    private readonly record struct ListEntry(string DirName, SaveDataDialogSlot? Slot, SaveDataListItem Item);

    private enum ListInput
    {
        None,
        Changed,
        Choose,
        Back,
        Delete,
    }

    private enum Press
    {
        Up,
        Down,
        Confirm,
        Cancel,
        Delete,
    }

    private sealed class ListSession(
        long session,
        List<ListEntry> entries,
        string caption,
        bool backEnabled,
        bool deleteEnabled)
    {
        private const byte StickUpThreshold = 64;
        private const byte StickDownThreshold = 192;
        private const int KeyBackspace = 0x08;
        private const int KeyEnter = 0x0D;
        private const int KeyEscape = 0x1B;
        private const int KeyUp = 0x26;
        private const int KeyDown = 0x28;
        private const int KeyDelete = 0x2E;
        private const int KeyV = 0x56;
        private const int KeyX = 0x58;
        private const int KeyZ = 0x5A;
        private static readonly long FirstRepeatTicks = Stopwatch.Frequency * 2 / 5;
        private static readonly long NextRepeatTicks = Stopwatch.Frequency / 9;

        private readonly List<uint> _controllerChanges = [];
        private readonly List<int> _keys = [];
        private readonly List<Press> _presses = [];
        private long _sequence;
        private uint _previousButtons;
        private int _heldDirection;
        private long _repeatAt;
        private bool _confirming;
        private string? _notice;
        private OverlayInputScheme _scheme;

        public List<ListEntry> Entries { get; } = entries;

        public int Focus { get; set; }

        public SaveDataListView View()
        {
            var items = new SaveDataListItem[Entries.Count];
            for (var index = 0; index < items.Length; index++)
            {
                items[index] = Entries[index].Item;
            }

            return new SaveDataListView(session, caption, items, Focus, backEnabled, deleteEnabled, _scheme,
                _confirming, _notice);
        }

        public void Begin()
        {
            var state = PadExports.ReadSystemDialogState();
            _sequence = PadExports.InputHistoryPosition;
            _previousButtons = state.ControllerButtons;
            _heldDirection = HeldDirection(state);
            _repeatAt = Stopwatch.GetTimestamp() + FirstRepeatTicks;
            _scheme = CurrentScheme();
        }

        public void CancelDelete(string? notice)
        {
            _confirming = false;
            _notice = notice;
        }

        public void Remove(int index)
        {
            Entries.RemoveAt(index);
            Focus = Math.Clamp(Focus, 0, Math.Max(0, Entries.Count - 1));
            _confirming = false;
            _notice = null;
        }

        public ListInput Poll()
        {
            var now = Stopwatch.GetTimestamp();
            var changed = false;
            var scheme = CurrentScheme();
            if (scheme != _scheme)
            {
                _scheme = scheme;
                changed = true;
            }

            _presses.Clear();
            _keys.Clear();
            SystemDialogInput.TakeKeys(_keys);
            foreach (var key in _keys)
            {
                switch (key)
                {
                    case KeyUp: _presses.Add(Press.Up); break;
                    case KeyDown: _presses.Add(Press.Down); break;
                    case KeyEnter or KeyZ: _presses.Add(Press.Confirm); break;
                    case KeyEscape or KeyBackspace or KeyX: _presses.Add(Press.Cancel); break;
                    case KeyDelete or KeyV: _presses.Add(Press.Delete); break;
                }
            }

            _controllerChanges.Clear();
            PadExports.TakeControllerHistory(ref _sequence, _controllerChanges);
            var current = PadExports.ReadSystemDialogState();
            _controllerChanges.Add(current.ControllerButtons);
            foreach (var buttons in _controllerChanges)
            {
                var pressed = buttons & ~_previousButtons;
                _previousButtons = buttons;
                if ((pressed & OrbisPadButton.Up) != 0) _presses.Add(Press.Up);
                if ((pressed & OrbisPadButton.Down) != 0) _presses.Add(Press.Down);
                if ((pressed & OrbisPadButton.Cross) != 0) _presses.Add(Press.Confirm);
                if ((pressed & OrbisPadButton.Circle) != 0) _presses.Add(Press.Cancel);
                if ((pressed & OrbisPadButton.Triangle) != 0) _presses.Add(Press.Delete);
            }

            foreach (var press in _presses)
            {
                if (_confirming)
                {
                    if (press == Press.Confirm)
                    {
                        return ListInput.Delete;
                    }

                    if (press == Press.Cancel)
                    {
                        _confirming = false;
                        changed = true;
                    }

                    continue;
                }

                switch (press)
                {
                    case Press.Confirm:
                        return ListInput.Choose;
                    case Press.Cancel when backEnabled:
                        return ListInput.Back;
                    case Press.Delete when deleteEnabled && Entries[Focus].Slot is { } slot:
                        if (SaveDataExports.IsDialogSlotMounted(slot))
                        {
                            _notice = SaveDataListLayer.Text("InUse");
                        }
                        else
                        {
                            _confirming = true;
                            _notice = null;
                        }

                        changed = true;
                        break;
                    case Press.Up or Press.Down:
                        var step = press == Press.Up ? -1 : 1;
                        changed |= Step(step);
                        _heldDirection = step;
                        _repeatAt = now + FirstRepeatTicks;
                        break;
                }
            }

            var direction = _confirming ? 0 : HeldDirection(current);
            if (direction == 0)
            {
                _heldDirection = 0;
            }
            else if (direction != _heldDirection)
            {
                changed |= Step(direction);
                _heldDirection = direction;
                _repeatAt = now + FirstRepeatTicks;
            }
            else if (now >= _repeatAt)
            {
                changed |= Step(direction);
                _repeatAt = now + NextRepeatTicks;
            }

            return changed ? ListInput.Changed : ListInput.None;
        }

        private bool Step(int step)
        {
            var next = Math.Clamp(Focus + step, 0, Entries.Count - 1);
            if (next == Focus)
            {
                return false;
            }

            Focus = next;
            _notice = null;
            return true;
        }

        private static int HeldDirection(PadState state)
        {
            var up = (state.ControllerButtons & OrbisPadButton.Up) != 0 || state.LeftY < StickUpThreshold ||
                     PadExports.IsHostKeyDown(KeyUp);
            var down = (state.ControllerButtons & OrbisPadButton.Down) != 0 || state.LeftY > StickDownThreshold ||
                       PadExports.IsHostKeyDown(KeyDown);
            return up == down ? 0 : up ? -1 : 1;
        }

        private static OverlayInputScheme CurrentScheme() => SystemDialogInput.CurrentScheme();
    }
}
