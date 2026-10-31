// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Platform.MacOS.Input;

namespace SharpEmu.Platform.MacOS.Windowing;

/// <summary>
/// The macOS presentation window: a native NSWindow with a layer-hosting NSView
/// backed by a CAMetalLayer, native keyboard handling, and GameController
/// gamepads. No SDL, no toolkit — AppKit, QuartzCore and GameController only.
/// </summary>
/// <remarks>
/// <para><b>Threading.</b> AppKit demands the process main thread, which is why
/// the CLI parks the real main thread in <c>HostMainThread.Pump</c> and runs
/// emulation on a worker. The Metal presenter posts <see cref="Run"/> onto that
/// pump, so this loop <i>is</i> the main thread: it drains NSApp's event queue
/// with <c>-nextEventMatchingMask:untilDate:inMode:dequeue:</c> and an
/// already-past date (never blocking), then renders, exactly matching the cadence
/// the SDL host window provided. Draining events through
/// <c>nextEventMatchingMask:</c> also spins the run loop in the default mode, so
/// main-queue work — including GameController's connect/disconnect delivery and
/// the fullscreen-transition animations — is serviced without a second thread.</para>
///
/// <para><b>Fullscreen.</b> Both <see cref="HostWindowMode.Borderless"/> and
/// <see cref="HostWindowMode.ExclusiveFullscreen"/> map onto AppKit's native
/// fullscreen (<c>-toggleFullScreen:</c>). macOS has no supported exclusive
/// display mode for a CAMetalLayer — <c>CGDisplayCapture</c> predates the modern
/// compositor and does not combine with layer-backed presentation — while native
/// fullscreen already gives a dedicated Space, a display-sized drawable and Core
/// Animation's direct-to-display path, which is as close to both as the platform
/// offers. The consequence is that <c>HostVideoOptions.RefreshRate</c> is not
/// applied on macOS: the window follows the display's current mode.</para>
///
/// <para><b>No Objective-C subclassing.</b> Window close, focus, resize and
/// backing-scale changes are read from the window each tick instead of through an
/// NSWindowDelegate, and key events are inspected before being forwarded to
/// <c>-[NSApp sendEvent:]</c>. That keeps the whole file free of
/// <c>objc_allocateClassPair</c> and of managed function pointers AppKit could
/// call on an unexpected thread.</para>
/// </remarks>
[SupportedOSPlatform("macos")]
public sealed class MacHostWindow : IMetalHostWindow
{
    // NSWindowStyleMask
    private const nuint StyleBorderless = 0;
    private const nuint StyleTitled = 1 << 0;
    private const nuint StyleClosable = 1 << 1;
    private const nuint StyleMiniaturizable = 1 << 2;
    private const nuint StyleResizable = 1 << 3;
    private const nuint StyleFullScreen = 1 << 14;
    private const nuint StyleWindowed = StyleTitled | StyleClosable | StyleMiniaturizable | StyleResizable;

    private const nuint BackingStoreBuffered = 2;
    private const nuint ActivationPolicyRegular = 0;

    // NSEventType
    private const nuint EventLeftMouseDown = 1;
    private const nuint EventMouseMoved = 5;
    private const nuint EventLeftMouseDragged = 6;
    private const nuint EventKeyDown = 10;
    private const nuint EventKeyUp = 11;

    // NSEventModifierFlags
    private const nuint ModifierControl = 1 << 18;
    private const nuint ModifierCommand = 1 << 20;

    private const ulong EventMaskAny = ulong.MaxValue;

    private static readonly long CursorHideDelayTicks = 2 * Stopwatch.Frequency;

    private readonly HostVideoOptions _options;
    private readonly Action? _toggleBackendHud;
    private readonly nint _defaultRunLoopMode;
    private readonly MacGameControllerGamepad _gamepad = new();

    private nint _application;
    private nint _window;
    private nint _view;
    private nint _metalLayer;
    private nint _distantPast;

    private int _closeRequested;
    private bool _closedByUser;
    private bool _fullscreen;
    private bool _focused = true;
    private bool _cursorVisible = true;
    private long _cursorHideDeadline;
    private bool _surfaceRestorePending = true;
    private bool _surfaceWasUnavailable;
    private bool _disposed;
    private double _contentsScale;
    private MacSize _layerSize;
    private string _baseTitle;
    private string _titleSummary = string.Empty;

    /// <summary>Creates and shows the window, wiring input into the host seam.</summary>
    public MacHostWindow(string title, HostVideoOptions options, Action? toggleBackendHud = null)
    {
        _options = options.Normalize();
        _toggleBackendHud = toggleBackendHud;
        _baseTitle = title;

        NativeLibrary.Load(MacObjC.AppKitFramework);
        NativeLibrary.Load(MacObjC.QuartzCoreFramework);
        _defaultRunLoopMode = MacObjC.RetainedNsString("kCFRunLoopDefaultMode");

        var pool = MacObjC.objc_autoreleasePoolPush();
        try
        {
            SetUpApplication();
            CreateWindow();
            PerfOverlay.Configure(_options);
            ApplyConfiguredMode(_options.WindowMode);
            HostWindowInput.Connect(_gamepad);
            _gamepad.Poll();
            Console.Error.WriteLine(
                $"[LOADER][INFO] AppKit window ready: mode={_options.WindowMode} " +
                $"size={_options.Width}x{_options.Height} display={_options.DisplayIndex} " +
                $"scale={_contentsScale:F2}");
        }
        finally
        {
            MacObjC.objc_autoreleasePoolPop(pool);
        }
    }

    /// <summary>Drawable pixels per logical point; 1 until the layer has a scale.</summary>
    public double ContentsScale => _contentsScale <= 0 ? 1.0 : _contentsScale;

    public (int Width, int Height) PixelSize
    {
        get
        {
            var scale = _contentsScale <= 0 ? 1.0 : _contentsScale;
            return (
                Math.Max((int)Math.Round(_layerSize.Width * scale), 1),
                Math.Max((int)Math.Round(_layerSize.Height * scale), 1));
        }
    }

    public bool IsMinimized =>
        _window != 0 && MacObjC.SendBool(_window, MacObjC.Selector("isMiniaturized"));

    public bool ClosedByUser => _closedByUser;

    public nint CreateMetalLayer() => _metalLayer;

    public (double Current, double Potential) EdrHeadroom
    {
        get
        {
            var screen = _window == 0 ? 0 : MacObjC.Send(_window, MacObjC.Selector("screen"));
            if (screen == 0)
            {
                screen = MacObjC.Send(MacObjC.Class("NSScreen"), MacObjC.Selector("mainScreen"));
            }

            if (screen == 0)
            {
                return (1.0, 1.0);
            }

            var current = Math.Max(1.0, MacObjC.SendDouble(
                screen, MacObjC.Selector("maximumExtendedDynamicRangeColorComponentValue")));
            // macOS 10.15+; before it the current value is all there is.
            var potentialSelector = MacObjC.Selector("maximumPotentialExtendedDynamicRangeColorComponentValue");
            var potential = MacObjC.SendBool(screen, MacObjC.Selector("respondsToSelector:"), potentialSelector)
                ? MacObjC.SendDouble(screen, potentialSelector)
                : current;
            return (current, Math.Max(current, potential));
        }
    }

    public bool ConsumeSurfaceRestore()
    {
        var pending = _surfaceRestorePending;
        _surfaceRestorePending = false;
        return pending;
    }

    public void SetTitle(string title)
    {
        _baseTitle = title;
        ApplyWindowTitle();
    }

    public void Close() => Volatile.Write(ref _closeRequested, 1);

    /// <summary>
    /// Formats the window title. RenderDoc has no macOS capture path, so unlike
    /// the SDL window there is never a capture hint to append.
    /// </summary>
    internal static string FormatWindowTitle(string title, string summary = "") =>
        string.IsNullOrEmpty(summary) ? title : $"{title} · {summary}";

    public void Run(Action initialize, Action<double> render, Action closing, Action? idle = null)
    {
        initialize();
        var timer = Stopwatch.StartNew();
        var last = timer.Elapsed.TotalSeconds;
        try
        {
            while (Volatile.Read(ref _closeRequested) == 0)
            {
                var pool = MacObjC.objc_autoreleasePoolPush();
                try
                {
                    PumpEvents();
                    RefreshOverlayTitle();
                    SyncWindowState();
                    if (Volatile.Read(ref _closeRequested) != 0)
                    {
                        break;
                    }

                    UpdateCursorAutoHide();
                    _gamepad.Poll();
                }
                finally
                {
                    MacObjC.objc_autoreleasePoolPop(pool);
                }

                var now = timer.Elapsed.TotalSeconds;
                render(now - last);
                last = now;
                if (IsMinimized)
                {
                    Thread.Sleep(10);
                }
                else
                {
                    idle?.Invoke();
                }
            }
        }
        finally
        {
            closing();
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        var pool = MacObjC.objc_autoreleasePoolPush();
        try
        {
            PerfOverlay.Shutdown();
            SetCursorVisible(true);
            HostWindowInput.Disconnect();
            _gamepad.Dispose();
            if (_window != 0)
            {
                MacObjC.SendVoid(_window, MacObjC.Selector("orderOut:"), 0);
                MacObjC.SendVoid(_window, MacObjC.Selector("close"));
                MacObjC.Release(_window);
                _window = 0;
            }

            MacObjC.Release(_view);
            _view = 0;
            MacObjC.Release(_metalLayer);
            _metalLayer = 0;
        }
        finally
        {
            MacObjC.objc_autoreleasePoolPop(pool);
        }
    }

    // ---- Setup ----

    private void SetUpApplication()
    {
        _application = MacObjC.Send(MacObjC.Class("NSApplication"), MacObjC.Selector("sharedApplication"));
        if (_application == 0)
        {
            throw new InvalidOperationException("NSApplication is unavailable.");
        }

        MacObjC.SendBool(
            _application,
            MacObjC.Selector("setActivationPolicy:"),
            (nint)ActivationPolicyRegular);
        BuildMenuBar();
        // finishLaunching rather than run: the loop below owns event dispatch.
        MacObjC.SendVoid(_application, MacObjC.Selector("finishLaunching"));
        MacObjC.SendVoidBool(_application, MacObjC.Selector("activateIgnoringOtherApps:"), true);
        _distantPast = MacObjC.Send(
            MacObjC.Send(MacObjC.Class("NSDate"), MacObjC.Selector("distantPast")),
            MacObjC.Selector("retain"));
        SetApplicationIcon();
    }

    /// <summary>
    /// A minimal menu bar so the app behaves like a real macOS app: without a
    /// main menu the Dock shows no application menu at all. Quit is wired to
    /// <c>-performClose:</c> so clicking it shuts the emulator down through the
    /// same window-closed path as the red button; Cmd+Q is additionally
    /// intercepted in <see cref="HandleKeyEvent"/> so it also works while the
    /// window is in fullscreen and cannot be "closed" by the responder chain.
    /// </summary>
    private void BuildMenuBar()
    {
        var mainMenu = MacObjC.Send(
            MacObjC.Send(MacObjC.Class("NSMenu"), MacObjC.Selector("alloc")),
            MacObjC.Selector("init"));
        var appItem = MacObjC.Send(
            MacObjC.Send(MacObjC.Class("NSMenuItem"), MacObjC.Selector("alloc")),
            MacObjC.Selector("init"));
        MacObjC.SendVoid(mainMenu, MacObjC.Selector("addItem:"), appItem);

        var appMenu = MacObjC.Send(
            MacObjC.Send(MacObjC.Class("NSMenu"), MacObjC.Selector("alloc")),
            MacObjC.Selector("init"));
        // Control+Command+F is the system-standard fullscreen chord.
        AddMenuItem(appMenu, "Enter Full Screen", "toggleFullScreen:", "f", ModifierControl | ModifierCommand);
        AddMenuItem(appMenu, "Quit SharpEmu", "performClose:", "q", ModifierCommand);
        MacObjC.SendVoid(appItem, MacObjC.Selector("setSubmenu:"), appMenu);
        MacObjC.SendVoid(_application, MacObjC.Selector("setMainMenu:"), mainMenu);
        MacObjC.Release(appMenu);
        MacObjC.Release(appItem);
        MacObjC.Release(mainMenu);
    }

    private static void AddMenuItem(
        nint menu,
        string title,
        string action,
        string keyEquivalent,
        nuint modifierMask)
    {
        var item = MacObjC.Send(
            MacObjC.Send(MacObjC.Class("NSMenuItem"), MacObjC.Selector("alloc")),
            MacObjC.Selector("initWithTitle:action:keyEquivalent:"),
            MacObjC.NsString(title),
            MacObjC.Selector(action),
            MacObjC.NsString(keyEquivalent));
        if (item == 0)
        {
            return;
        }

        MacObjC.SendVoidUInt(item, MacObjC.Selector("setKeyEquivalentModifierMask:"), modifierMask);
        MacObjC.SendVoid(menu, MacObjC.Selector("addItem:"), item);
        MacObjC.Release(item);
    }

    /// <summary>
    /// Uses the mounted title's own icon as the Dock icon when there is one.
    /// The path mirrors <c>PngSplashLoader</c>'s (app0/sce_sys/icon0.png) but
    /// hands the file straight to NSImage instead of decoding it here.
    /// </summary>
    private void SetApplicationIcon()
    {
        try
        {
            var app0Root = Environment.GetEnvironmentVariable("SHARPEMU_APP0_DIR");
            if (string.IsNullOrWhiteSpace(app0Root))
            {
                return;
            }

            var path = Path.Combine(app0Root, "sce_sys", "icon0.png");
            if (!File.Exists(path))
            {
                return;
            }

            var image = MacObjC.Send(
                MacObjC.Send(MacObjC.Class("NSImage"), MacObjC.Selector("alloc")),
                MacObjC.Selector("initWithContentsOfFile:"),
                MacObjC.NsString(path));
            if (image == 0)
            {
                return;
            }

            MacObjC.SendVoid(_application, MacObjC.Selector("setApplicationIconImage:"), image);
            MacObjC.Release(image);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Application icon could not be set: {exception.Message}");
        }
    }

    private void CreateWindow()
    {
        var screen = GetConfiguredScreen();
        var frame = CenteredFrame(screen, _options.Width, _options.Height);
        _window = MacObjC.SendInitWindow(
            MacObjC.Send(MacObjC.Class("NSWindow"), MacObjC.Selector("alloc")),
            MacObjC.Selector("initWithContentRect:styleMask:backing:defer:"),
            frame,
            StyleWindowed,
            BackingStoreBuffered,
            false);
        if (_window == 0)
        {
            throw new InvalidOperationException("NSWindow creation failed.");
        }

        // Without this AppKit frees the window when the user closes it, and the
        // loop would then message a dangling pointer on its way out.
        MacObjC.SendVoidBool(_window, MacObjC.Selector("setReleasedWhenClosed:"), false);
        MacObjC.SendVoidBool(_window, MacObjC.Selector("setAcceptsMouseMovedEvents:"), true);
        MacObjC.SendVoidBool(_window, MacObjC.Selector("setRestorable:"), false);
        ApplyWindowTitle();

        var contentRect = new MacRect(0, 0, _options.Width, _options.Height);
        _view = MacObjC.SendInitWithFrame(
            MacObjC.Send(MacObjC.Class("NSView"), MacObjC.Selector("alloc")),
            MacObjC.Selector("initWithFrame:"),
            contentRect);
        if (_view == 0)
        {
            throw new InvalidOperationException("NSView creation failed.");
        }

        _metalLayer = MacObjC.Send(
            MacObjC.Send(MacObjC.Class("CAMetalLayer"), MacObjC.Selector("alloc")),
            MacObjC.Selector("init"));
        if (_metalLayer == 0)
        {
            throw new InvalidOperationException("CAMetalLayer creation failed.");
        }

        // Layer-hosting (layer set before wantsLayer) rather than layer-backed:
        // AppKit then leaves the CAMetalLayer's geometry and contentsScale to us,
        // which is what lets SyncWindowState drive drawableSize deterministically.
        MacObjC.SendVoidRect(_metalLayer, MacObjC.Selector("setFrame:"), contentRect);
        MacObjC.SendVoid(_view, MacObjC.Selector("setLayer:"), _metalLayer);
        MacObjC.SendVoidBool(_view, MacObjC.Selector("setWantsLayer:"), true);
        MacObjC.SendVoid(_window, MacObjC.Selector("setContentView:"), _view);
        MacObjC.SendVoid(_window, MacObjC.Selector("makeFirstResponder:"), _view);
        MacObjC.SendVoid(_window, MacObjC.Selector("makeKeyAndOrderFront:"), 0);
        SyncLayerGeometry(force: true);
    }

    private void ApplyWindowTitle()
    {
        if (_window == 0)
        {
            return;
        }

        MacObjC.SendVoid(
            _window,
            MacObjC.Selector("setTitle:"),
            MacObjC.NsString(FormatWindowTitle(_baseTitle, _titleSummary)));
    }

    private void RefreshOverlayTitle()
    {
        var summary = PerfOverlay.GetTitleBarSummary();
        if (_titleSummary != summary)
        {
            _titleSummary = summary;
            ApplyWindowTitle();
        }
    }

    // ---- Displays ----

    private nint GetConfiguredScreen()
    {
        var screens = MacObjC.Send(MacObjC.Class("NSScreen"), MacObjC.Selector("screens"));
        var count = MacObjC.ArrayCount(screens);
        if (count == 0)
        {
            return MacObjC.Send(MacObjC.Class("NSScreen"), MacObjC.Selector("mainScreen"));
        }

        var index = (nuint)Math.Clamp(_options.DisplayIndex, 0, (int)count - 1);
        return MacObjC.ArrayObjectAt(screens, index);
    }

    private static MacRect CenteredFrame(nint screen, int width, int height)
    {
        if (screen == 0)
        {
            return new MacRect(0, 0, width, height);
        }

        // Whole-point origins only: a half-point frame makes AppKit round the
        // content view up by a point, which shows up as an off-by-one drawable
        // (1442 instead of 1440 rows at 2x).
        var visible = MacObjC.SendRect(screen, MacObjC.Selector("visibleFrame"));
        return new MacRect(
            Math.Round(visible.Origin.X + Math.Max(0, (visible.Size.Width - width) / 2)),
            Math.Round(visible.Origin.Y + Math.Max(0, (visible.Size.Height - height) / 2)),
            width,
            height);
    }

    // ---- Per-tick window state ----

    /// <summary>
    /// Polls the window for everything an NSWindowDelegate would have pushed:
    /// user close, key-window focus, content resize and backing-scale change.
    /// </summary>
    private void SyncWindowState()
    {
        if (_window == 0)
        {
            return;
        }

        // -[NSWindow isVisible] also reports NO while the window is in the Dock
        // or the whole app is hidden (Cmd+H), neither of which is a close; only
        // a genuinely gone window ends the session.
        var visible = MacObjC.SendBool(_window, MacObjC.Selector("isVisible"));
        var miniaturized = MacObjC.SendBool(_window, MacObjC.Selector("isMiniaturized"));
        var hidden = MacObjC.SendBool(_application, MacObjC.Selector("isHidden"));
        var unavailable = !visible || miniaturized || hidden;
        if (_surfaceWasUnavailable && !unavailable) _surfaceRestorePending = true;
        _surfaceWasUnavailable = unavailable;
        if (!visible && !miniaturized && !hidden)
        {
            RequestUserClose();
            return;
        }

        var focused = MacObjC.SendBool(_window, MacObjC.Selector("isKeyWindow"));
        if (focused != _focused)
        {
            _focused = focused;
            UpdateCursorVisibility();
            HostWindowInput.SetFocused(focused);
        }

        var nativeFullscreen = (MacObjC.SendUInt(_window, MacObjC.Selector("styleMask")) & StyleFullScreen) != 0;
        if (nativeFullscreen != _fullscreen)
        {
            _fullscreen = nativeFullscreen;
            UpdateCursorVisibility();
            _surfaceRestorePending = true;
        }

        SyncLayerGeometry(force: false);
    }

    /// <summary>
    /// Keeps the hosted CAMetalLayer's bounds and contentsScale on the view, so
    /// the presenter's drawable stays in real device pixels across resize, Retina
    /// and moves between displays with different scale factors.
    /// </summary>
    private void SyncLayerGeometry(bool force)
    {
        if (_view == 0 || _metalLayer == 0)
        {
            return;
        }

        var bounds = MacObjC.SendRect(_view, MacObjC.Selector("bounds"));
        var scale = _window == 0
            ? 1.0
            : MacObjC.SendDouble(_window, MacObjC.Selector("backingScaleFactor"));
        if (scale <= 0)
        {
            scale = 1.0;
        }

        if (!force &&
            bounds.Size.Width == _layerSize.Width &&
            bounds.Size.Height == _layerSize.Height &&
            scale == _contentsScale)
        {
            return;
        }

        _layerSize = bounds.Size;
        _contentsScale = scale;
        MacObjC.SendVoidRect(_metalLayer, MacObjC.Selector("setFrame:"), bounds);
        MacObjC.SendVoidDouble(_metalLayer, MacObjC.Selector("setContentsScale:"), scale);
        _surfaceRestorePending = true;
    }

    private void RequestUserClose()
    {
        _closedByUser = true;
        Volatile.Write(ref _closeRequested, 1);
    }

    // ---- Events ----

    private void PumpEvents()
    {
        if (_application == 0)
        {
            return;
        }

        while (true)
        {
            var nsEvent = MacObjC.SendNextEvent(
                _application,
                MacObjC.Selector("nextEventMatchingMask:untilDate:inMode:dequeue:"),
                EventMaskAny,
                _distantPast,
                _defaultRunLoopMode,
                true);
            if (nsEvent == 0)
            {
                return;
            }

            if (!HandleEvent(nsEvent))
            {
                MacObjC.SendVoid(_application, MacObjC.Selector("sendEvent:"), nsEvent);
            }
        }
    }

    /// <summary>
    /// Returns true when the event is fully consumed here and must not reach
    /// <c>-[NSApp sendEvent:]</c>. Swallowing handled key presses is also what
    /// suppresses the system beep: the alert comes from the responder chain
    /// failing to act on a key-down, so an event that never gets there is silent.
    /// </summary>
    private bool HandleEvent(nint nsEvent)
    {
        var type = MacObjC.SendUInt(nsEvent, MacObjC.Selector("type"));
        switch (type)
        {
            case EventKeyDown:
            case EventKeyUp:
                return HandleKeyEvent(nsEvent, type == EventKeyDown);
            case EventMouseMoved:
            case EventLeftMouseDragged:
                ShowCursorTemporarily();
                return false;
            case EventLeftMouseDown:
                ShowCursorTemporarily();
                if (MacObjC.SendInt(nsEvent, MacObjC.Selector("clickCount")) == 2 &&
                    IsInsideContentView(nsEvent))
                {
                    ToggleFullscreen();
                    return true;
                }

                return false;
            default:
                return false;
        }
    }

    private bool HandleKeyEvent(nint nsEvent, bool down)
    {
        var keyCode = MacObjC.SendUShort(nsEvent, MacObjC.Selector("keyCode"));
        var modifiers = MacObjC.SendUInt(nsEvent, MacObjC.Selector("modifierFlags"));
        var command = (modifiers & ModifierCommand) != 0;
        var repeat = down && MacObjC.SendBool(nsEvent, MacObjC.Selector("isARepeat"));
        if (down && !repeat)
        {
            switch (keyCode)
            {
                case MacVirtualKeyCode.Q when command:
                    RequestUserClose();
                    return true;
                case MacVirtualKeyCode.F1 when command && _toggleBackendHud is not null:
                    _toggleBackendHud();
                    return true;
                case MacVirtualKeyCode.F1:
                    PerfOverlay.Toggle();
                    return true;
                case MacVirtualKeyCode.F2:
                    PerfOverlay.CycleCorner();
                    return true;
                case MacVirtualKeyCode.F11:
                    ToggleFullscreen();
                    return true;
            }
        }

        if (!MacKeyMap.TryMapVirtualKey(keyCode, out var virtualKey))
        {
            // Unmapped keys stay with AppKit so menu key equivalents and text
            // input in system UI keep working.
            return false;
        }

        // Command chords belong to the menu bar, never to the guest.
        if (command)
        {
            return false;
        }

        HostWindowInput.SetKey(virtualKey, down);
        return true;
    }

    private bool IsInsideContentView(nint nsEvent)
    {
        if (_view == 0 || MacObjC.Send(nsEvent, MacObjC.Selector("window")) != _window)
        {
            return false;
        }

        var location = MacObjC.SendPoint(nsEvent, MacObjC.Selector("locationInWindow"));
        return MacObjC.SendRect(_view, MacObjC.Selector("frame")).Contains(location);
    }

    // ---- Window mode ----

    private void ApplyConfiguredMode(HostWindowMode mode)
    {
        if (mode == HostWindowMode.Windowed)
        {
            UpdateCursorVisibility();
            return;
        }

        EnterNativeFullscreen();
    }

    private void ToggleFullscreen()
    {
        if (_window == 0)
        {
            return;
        }

        // toggleFullScreen: covers both directions and restores the pre-fullscreen
        // frame itself on the way out; the resulting styleMask is read back by
        // SyncWindowState once the transition animates in.
        MacObjC.SendVoid(_window, MacObjC.Selector("toggleFullScreen:"), 0);
    }

    private void EnterNativeFullscreen()
    {
        if (_window == 0 || _fullscreen)
        {
            return;
        }

        MacObjC.SendVoid(_window, MacObjC.Selector("toggleFullScreen:"), 0);
        _fullscreen = true;
        UpdateCursorVisibility();
        _surfaceRestorePending = true;
    }

    // ---- Cursor ----

    private void UpdateCursorVisibility()
    {
        _cursorHideDeadline = 0;
        SetCursorVisible(!_fullscreen || !_focused);
    }

    private void ShowCursorTemporarily()
    {
        if (!_fullscreen || !_focused)
        {
            return;
        }

        SetCursorVisible(true);
        _cursorHideDeadline = Stopwatch.GetTimestamp() + CursorHideDelayTicks;
    }

    private void UpdateCursorAutoHide()
    {
        if (!_fullscreen || !_focused || !_cursorVisible || _cursorHideDeadline == 0 ||
            Stopwatch.GetTimestamp() < _cursorHideDeadline)
        {
            return;
        }

        _cursorHideDeadline = 0;
        SetCursorVisible(false);
    }

    /// <summary>
    /// NSCursor hide/unhide is reference counted, so the calls have to stay
    /// balanced or the cursor never comes back.
    /// </summary>
    private void SetCursorVisible(bool visible)
    {
        if (_cursorVisible == visible)
        {
            return;
        }

        var cursor = MacObjC.Class("NSCursor");
        if (cursor == 0)
        {
            return;
        }

        MacObjC.SendVoid(cursor, MacObjC.Selector(visible ? "unhide" : "hide"));
        _cursorVisible = visible;
    }
}
