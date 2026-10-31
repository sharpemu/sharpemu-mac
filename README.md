<!--
Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# SharpEmu for macOS

<p align="center">
  <img src="./assets/images/logo.png" width=30% height=30% />
</p>

<p align="center">
  An experimental PlayStation 5 emulator for macOS, rendering natively with Metal.
</p>

<p align="center">
  <a href="https://github.com/sharpemu/sharpemu-mac/actions/workflows/workflow.yml">
    <img src="https://github.com/sharpemu/sharpemu-mac/actions/workflows/workflow.yml/badge.svg" alt="Build and Release">
  </a>
  <a href="./LICENSE.txt">
    <img src="https://img.shields.io/badge/license-GPL--2.0--or--later-blue" alt="License: GPL-2.0-or-later">
  </a>
</p>

---

<p align="center">
  <a href="https://discord.gg/zdTuUU9Uwn">
    <img src="https://img.shields.io/badge/Discord-Join%20our%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord">
  </a>
</p>

<p align="center">
  <strong>Join our official Discord server for development updates, compatibility discussions, support, and community chat.</strong>
</p>

---

> [!IMPORTANT]
> **Official channels:** Only **sharpemu.app** and the links listed in this GitHub repository are affiliated with SharpEmu. Any other websites, accounts, or donation pages are unofficial and unauthorized.

> [!WARNING]
> SharpEmu is an experimental PS5 emulator developed from scratch in C#. The current focus is on accuracy and infrastructure setup rather than game-specific compatibility. Compatibility and performance vary by game and by Mac.

## About

SharpEmu-mac is the macOS edition of [SharpEmu](https://github.com/sharpemu/sharpemu), an experimental
PlayStation 5 emulator written in C#. It runs on macOS with a native Metal
renderer ("Metal V2", in `src/graphics/SharpEmu.Gpu.Metal/V2`) that translates
the console's GPU work to Metal directly, with no MoltenVK and no Vulkan on the
macOS path. Guest x86-64 code runs directly on the host CPU, which is why the
shipped build is `osx-x64` and runs under Rosetta 2 on Apple Silicon.

SharpEmu is developed purely for research and educational purposes, with no
commercial goals. It focuses exclusively on the PlayStation 5; for PS4 games,
see [ShadPS4](https://github.com/shadps4-emu/shadPS4).

This fork tracks upstream SharpEmu, which targets Windows and Linux with Vulkan.
All credit for the emulator core belongs to the upstream project and its
contributors.

## Requirements

- macOS 15 or later.
- An Apple Silicon Mac with Rosetta 2, or an Intel Mac. The CPU backend runs the
  guest's x86-64 code natively, so the host process must be x86-64; `osx-x64` is
  the only macOS build, and `osx-arm64` is not supported.
- A GPU that supports Metal. Games that draw with primitive (NGG) shaders need a
  Metal 3 device, because the backend runs them as Metal mesh pipelines.
- Game dumps from a console you own. Titles can use a lot of memory, so a Mac
  with little free memory may page heavily.

On Apple Silicon, install Rosetta 2 once if you have not already:

```sh
softwareupdate --install-rosetta
```

See [Building on macOS](docs/macos/building.md#requirements) for the details behind these requirements.

## Download and install

Download a build from the [Releases page](https://github.com/sharpemu/sharpemu-mac/releases).
Each release publishes two assets, both for `osx-x64`:

| Asset | Contents |
| --- | --- |
| `sharpemu-<version>-osx-x64-app.zip` | `SharpEmu.app`, the macOS app bundle. This is the one to use for the GUI. |
| `sharpemu-<version>-osx-x64.tar.gz` | The plain self-contained directory with the `SharpEmu` executable, for command-line use. |

Unzip the app, move `SharpEmu.app` where you like (for example `/Applications`)
and open it.

**First launch and Gatekeeper.** Release bundles are ad-hoc signed and are not
notarized, so macOS may refuse to open a downloaded copy. Either allow it in
System Settings, Privacy and Security, or remove the quarantine attribute:

```sh
xattr -dr com.apple.quarantine SharpEmu.app
```

The app cannot update itself in place. The GUI reports new releases and opens
the releases page. If something does not start, see
[macOS troubleshooting](docs/macos/troubleshooting.md).

## Quick start

> [!IMPORTANT]
> This project does **not** support or condone piracy. All games used during
> development and testing are dumped from consoles that we personally own.
> Users are expected to use legally obtained copies of their games.

**GUI.** Open `SharpEmu.app` (or run the `SharpEmu` executable with no
arguments), add your game folders in the library settings, select a game and
press Play. Environment variables can be set globally or per game in the GUI.

**Command line.** Pass the game's `eboot.bin` (or a `.zar` archive):

```sh
SharpEmu.app/Contents/MacOS/SharpEmu "/path/to/game/eboot.bin" --log-file
```

**Supported game formats:**

- A dumped application folder: the folder that contains `eboot.bin`, normally
  with `sce_sys/param.json` and the game's data and modules beside it.
- A `.zar` archive of a whole game folder, with `eboot.bin` at its root. The GUI
  can pack a game folder into a `.zar`.

Settings, save data and shader caches live under one data root. Inside
`SharpEmu.app` that is `~/Library/Application Support/SharpEmu`. Every log,
trace, crash dump and shader or frame dump goes under `user/logs/` in that data
root; `SHARPEMU_LOG_DIR` moves it. See
[Running games on macOS](docs/macos/running-games.md) for the full usage line,
the data folder, controls, and the performance display. Game patches are described in
[Game patches](docs/patches.md), and environment variables in
[macOS configuration](docs/macos/configuration.md).

The first run of a game compiles its shaders, so it starts slowly; later runs
reuse the caches.

## Games

Captured on SharpEmu-mac with the Metal backend.

| Black Myth: Wukong | Grand Theft Auto V |
| :---: | :---: |
| ![Black Myth: Wukong](./.github/images/macos/black-myth-wukong.jpg) | ![Grand Theft Auto V](./.github/images/macos/gta-v.jpg) |

| Demon's Souls | SILENT HILL: The Short Message |
| :---: | :---: |
| ![Demon's Souls](./.github/images/macos/demons-souls.jpg) | ![SILENT HILL: The Short Message](./.github/images/macos/silent-hill-short-message.jpg) |

| Superliminal | Poppy Playtime Chapter 1 |
| :---: | :---: |
| ![Superliminal](./.github/images/macos/superliminal.jpg) | ![Poppy Playtime Chapter 1](./.github/images/macos/poppy-playtime.jpg) |

| Quake | The Oregon Trail |
| :---: | :---: |
| ![Quake](./.github/images/macos/quake.jpg) | ![The Oregon Trail](./.github/images/macos/oregon-trail.jpg) |

| void tRrLM(); //Void Terrarium++ | Dreaming Sarah |
| :---: | :---: |
| ![void tRrLM(); //Void Terrarium++](./.github/images/macos/void-terrarium.jpg) | ![Dreaming Sarah](./.github/images/macos/dreaming-sarah.jpg) |

## Building from source

You need the .NET SDK named in [`global.json`](./global.json), the Xcode command
line tools (`xcode-select --install`) and Python 3. Full Xcode is not required.

```sh
git clone https://github.com/sharpemu/sharpemu-mac.git
cd sharpemu-mac
dotnet build SharpEmu.slnx -c Release
dotnet test SharpEmu.slnx -c Release
scripts/macos/build-app.sh        # -> artifacts/macos/SharpEmu.app and a .zip
```

`build-app.sh` publishes `osx-x64`, assembles and ad-hoc signs `SharpEmu.app`,
and verifies it. Signing with a Developer ID and notarization are optional. See
[Building on macOS](docs/macos/building.md) for requirements, build options,
signing and entitlements.

## Documentation

- [Documentation index](docs/README.md)
- macOS guides:
  [Building](docs/macos/building.md),
  [Running games](docs/macos/running-games.md),
  [Configuration](docs/macos/configuration.md),
  [Metal backend](docs/macos/metal-backend.md),
  [Testing](docs/macos/testing.md),
  [Troubleshooting](docs/macos/troubleshooting.md)

## Contributing

Before opening an issue or pull request, please read our contribution guidelines:

**[CONTRIBUTING.md](./CONTRIBUTING.md)**

The guide covers:
- Coding style and formatting
- AI-assisted contributions
- Pull request expectations
- Testing guidelines
- Legal and reverse engineering policy

## Upstream sync

This repository is a fork of [sharpemu/sharpemu](https://github.com/sharpemu/sharpemu).
Upstream changes arrive in `main` through merge pull requests from a pristine
`upstream` mirror branch, never by rewriting history. A scheduled workflow
(`.github/workflows/upstream-sync.yml`) keeps that mirror current and opens the
pull request; `scripts/mac/sync-upstream.sh` does the same by hand when a merge
has conflicts.

## Disclaimer

SharpEmu is an experimental emulator intended for research and educational purposes.

This project does not contain any copyrighted system firmware, game data, or proprietary PlayStation assets.

## Special Thanks

The following projects were extremely helpful during development of SharpEmu:

* **[ShadPS4](https://github.com/shadps4-emu/shadPS4)**
Helped with understanding the basic architecture of the PlayStation 4.

* **[Kyty](https://github.com/InoriRus/Kyty)**
One of the few PS5 emulator projects available and very useful for studying native code execution.

* **Ryujinx**
Provided valuable references for filesystem handling and low-level C# implementation patterns.

* **[OpenGNM](https://github.com/PS4-OpenGNM/opengnm)**
Used as a reference for the general GCN architecture.

## License

SharpEmu is licensed under the [**GPL-2.0-or-later**](./LICENSE.txt) license; see
the [`LICENSES`](./LICENSES) folder for the license texts of bundled components.

## Support

Support SharpEmu via GitHub Sponsors or cryptocurrency. Every contribution helps fund ongoing development and long-term maintenance. GitHub Sponsors is the preferred way to support the project, but cryptocurrency donations are also appreciated.

### ETH/USDT

`0xF315F5d986c790bB3A58DbE60F1B2760997dEd82`

### BTC

`bc1qmr9k8899njys5ny63xsues4jgmkk96erslrkmv`
