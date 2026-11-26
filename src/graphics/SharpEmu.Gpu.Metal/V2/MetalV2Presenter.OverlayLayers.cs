// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Overlay;

namespace SharpEmu.Gpu.Metal.V2;

// The in-game overlay layers (system dialogs drawn by the emulator: the IME keyboard, the save
// data list), drawn over every presented frame before the performance overlay, as the Vulkan
// presenter records them. OverlayLayerOutput.Source says where they come from: painted here in a
// single process, or painted in the guest process and shipped over the split.
internal static partial class MetalV2Presenter
{
    private static readonly List<OverlayLayerFrame> _overlayLayerFrames = [];
    private static MetalOverlayLayerCompositor? _overlayLayerCompositor;
    private static long _overlayLayersPresentedVersion = -1;

    // True when a layer appeared, changed or hid since the last present: an unchanged guest frame
    // is presented again to show it.
    private static bool OverlayLayersNeedPresent() =>
        OverlayLayerOutput.Source.Version != Volatile.Read(ref _overlayLayersPresentedVersion);

    // Runs on the window thread, inside the present encoder, after the frame is drawn.
    private static void EncodeOverlayLayers(nint encoder)
    {
        var source = OverlayLayerOutput.Source;
        // Read before collecting: a change made while the layers are collected presents again.
        var version = source.Version;
        _overlayLayerFrames.Clear();
        var width = (int)_drawableWidth;
        var height = (int)_drawableHeight;
        if (width > 0 && height > 0)
        {
            source.Collect(width, height, _overlayLayerFrames);
        }

        Volatile.Write(ref _overlayLayersPresentedVersion, version);
        if (_overlayLayerFrames.Count == 0 && _overlayLayerCompositor is null)
        {
            return;
        }

        _overlayLayerCompositor ??= new MetalOverlayLayerCompositor(_device);
        _ = _overlayLayerCompositor.Encode(
            encoder, _overlayLayerFrames, width, height, _layerHdrActive, _edrCurrentHeadroom);
        _overlayLayerFrames.Clear();
    }

    private static void ReleaseOverlayLayerResources()
    {
        _overlayLayerCompositor?.Dispose();
        _overlayLayerCompositor = null;
        _overlayLayerFrames.Clear();
        Volatile.Write(ref _overlayLayersPresentedVersion, -1);
    }
}
