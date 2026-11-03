// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia.Controls;
using Avalonia.Input;

namespace SharpEmu.GUI;

/// <summary>Asks for the name of a custom overlay preset; closes with the trimmed name, or null on cancel.</summary>
public partial class OverlayPresetNameDialog : SheetView
{
    public OverlayPresetNameDialog()
        : this(string.Empty)
    {
    }

    public OverlayPresetNameDialog(string suggestedName)
    {
        InitializeComponent();
        NameTextBox.Text = suggestedName;
        CancelButton.Click += (_, _) => Close(null);
        SaveButton.Click += (_, _) => Accept();
        NameTextBox.KeyDown += (_, e) =>
        {
            if (e.Key == Key.Enter)
            {
                e.Handled = true;
                Accept();
            }
        };
        Opened += (_, _) => NameTextBox.SelectAll();
    }

    public override Control? InitialFocus => NameTextBox;

    private void Accept()
    {
        var name = NameTextBox.Text?.Trim();
        Close(string.IsNullOrEmpty(name) ? null : name);
    }
}
