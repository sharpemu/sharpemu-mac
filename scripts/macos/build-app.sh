#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Builds SharpEmu.app: a directory-based, self-contained osx-x64 bundle, signed
# (ad hoc unless an identity is supplied), verified, and zipped for
# distribution. The launcher and the split's GPU host run natively: both are
# osx-arm64 and share one runtime in Contents/MacOS/arm64/, and the bundle's
# main executable is an arm64 apphost that starts the GUI from there.
#
# Why osx-x64 on a machine that may be Apple silicon: the CPU backend executes
# guest PS5 x86-64 instructions natively, so the host process has to be
# x86-64 and runs under Rosetta 2. An osx-arm64 build is rejected at build time
# (see src/SharpEmu.CLI/SharpEmu.CLI.csproj).
#
# Why not single-file: inside a bundle, single-file publish extracts its
# natives to a temporary directory on first run, which moves
# AppContext.BaseDirectory away from Contents/MacOS and breaks the
# "<base>/plugins" lookup that FfmpegRuntime and the managed plugin resolver
# both use. A plain directory layout keeps those relative paths true and keeps
# every file individually signable.
#
# Environment:
#   SHARPEMU_BUNDLE_ID            CFBundleIdentifier (default io.github.sharpemu.mac)
#   SHARPEMU_MIN_MACOS            LSMinimumSystemVersion (default 15.0)
#   SHARPEMU_CODESIGN_IDENTITY    signing identity; unset means ad-hoc ("-")
#   SHARPEMU_MAC_FORBID_SDL       "1" (default) fails the build on SDL3; "0" to bisect
#   SHARPEMU_MAC_GET_TASK_ALLOW   "1" adds get-task-allow (measurement builds: memstat needs task_for_pid)
#   SHARPEMU_MAC_KEEP_UNIVERSAL   pass "1" to keep arm64 slices in fat dylibs
#   SHARPEMU_MAC_R2R              "true" (default) publishes ReadyToRun; "false" to bisect
#   SHARPEMU_MAC_GPU_HOST         "1" (default) bundles the split's arm64 GPU host; "0" leaves it out
#   SHARPEMU_MAC_NATIVE_GUI       "1" (default) makes the arm64 GUI host the main executable;
#                                 "0" keeps the x64 SharpEmu as the main executable
#   SHARPEMU_MAC_DMG              pass "1" to also produce a .dmg
#   SHARPEMU_OUTPUT_DIR           where SharpEmu.app is written
#                                 (default artifacts/macos)
#   SHARPEMU_BUILD_WORKERS        maximum MSBuild workers (default 2)
#   Notarization (all three required, otherwise skipped):
#   SHARPEMU_NOTARY_APPLE_ID, SHARPEMU_NOTARY_PASSWORD, SHARPEMU_NOTARY_TEAM_ID

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
platform_dir="${repo_root}/build/platforms/macos"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build-app.sh: a macOS .app bundle can only be assembled and signed on macOS." >&2
    exit 1
fi

bundle_id="${SHARPEMU_BUNDLE_ID:-io.github.sharpemu.mac}"
min_macos="${SHARPEMU_MIN_MACOS:-15.0}"
codesign_identity="${SHARPEMU_CODESIGN_IDENTITY:--}"
output_dir="${SHARPEMU_OUTPUT_DIR:-${repo_root}/artifacts/macos}"

rid="osx-x64"
executable_name="SharpEmu"
gui_executable_name="SharpEmuGUI"
icon_source="${repo_root}/assets/images/SharpEmu.icns"
icon_file="SharpEmu.icns"
app_dir="${output_dir}/SharpEmu.app"
publish_dir="${output_dir}/publish-${rid}"

# The version is the single source of truth in Directory.Build.props, read the
# same way the release workflow reads it.
version="$(python3 -I -c 'import sys, xml.etree.ElementTree as ET; v = ET.parse(sys.argv[1]).getroot().findtext(".//SharpEmuVersion"); sys.exit("Directory.Build.props is missing SharpEmuVersion") if not v or not v.strip() else print(v.strip())' "${repo_root}/Directory.Build.props")"

# CFBundleVersion must be one to three dot-separated integers, so a
# "0.0.5-nexus" style version has to lose its pre-release suffix there.
# CFBundleShortVersionString keeps the full string for display.
bundle_version="${version%%-*}"

echo "==> SharpEmu.app ${version} (${rid}, bundle id ${bundle_id}, min macOS ${min_macos})"

# ---------------------------------------------------------------- publish ----

# PublishReadyToRun: AOT-compile the managed assemblies to osx-x64 native code
# at publish time so the JIT does not have to produce them during startup. This
# matters more here than for an ordinary app for two reasons: the startup path
# is enormous (the HLE import surface, the ELF loader and both shader
# translators are all jitted before the first guest frame), and the process runs
# under Rosetta 2, which translates the JIT's own machine-code emission on top
# of everything else. R2R code is a tier-0 substitute only — tiered compilation
# and TieredPGO still re-jit whatever stays hot, so steady-state throughput is
# unaffected.
#
# crossgen2 runs as a native arm64 tool and cross-compiles for osx-x64, so this
# works on an Apple silicon build host. Set SHARPEMU_MAC_R2R=false to bisect a
# suspected R2R codegen problem.
publish_args=(
    publish "${repo_root}/src/SharpEmu.CLI/SharpEmu.CLI.csproj"
    -c Release
    -r "${rid}"
    --self-contained true
    -p:PublishSingleFile=false
    -p:PublishReadyToRun="${SHARPEMU_MAC_R2R:-true}"
    -p:PublishDir="${publish_dir}/"
)
if [ "${SHARPEMU_MAC_FORBID_SDL:-1}" = "1" ]; then
    publish_args+=(-p:SharpEmuMacForbidSdl=true)
fi

rm -rf "${publish_dir}" "${app_dir}"
mkdir -p "${output_dir}"
build_workers="${SHARPEMU_BUILD_WORKERS:-2}"
case "${build_workers}" in
    ''|*[!0-9]*|0) echo "build-app.sh: SHARPEMU_BUILD_WORKERS must be a positive integer." >&2; exit 2 ;;
esac
source_snapshot="${output_dir}/build-source.json"
sdk_version="$(dotnet --version)"
python3 "${script_dir}/build-metadata.py" snapshot \
    --repo "${repo_root}" --snapshot "${source_snapshot}"
MSBUILDDISABLENODEREUSE=1 dotnet "${publish_args[@]}" -m:"${build_workers}" -nr:false \
    -p:UseSharedCompilation=false -p:SharpEmuBuildProfile=macos-bundle

if [ ! -x "${publish_dir}/${executable_name}" ]; then
    echo "build-app.sh: expected ${publish_dir}/${executable_name} to exist after publish." >&2
    exit 1
fi

# The native arm64 payload: the split's GPU host (SHARPEMU_GPU_PROCESS) and the
# GUI host, published into one directory so they share a single self-contained
# arm64 runtime. It lands in Contents/MacOS/arm64/, the first place
# GpuProcessHost.FindExecutable looks, and next to which the GUI finds the x64
# emulator ("..").
arm64_dir="arm64"
arm64_publish_dir="${output_dir}/publish-osx-arm64"
bundle_gpu_host="${SHARPEMU_MAC_GPU_HOST:-1}"
native_gui="${SHARPEMU_MAC_NATIVE_GUI:-1}"
arm64_projects=()
[ "${bundle_gpu_host}" = "1" ] && arm64_projects+=(SharpEmu.GpuHost)
[ "${native_gui}" = "1" ] && arm64_projects+=(SharpEmu.GuiHost)
rm -rf "${arm64_publish_dir}"
for project in ${arm64_projects[@]+"${arm64_projects[@]}"}; do
    MSBUILDDISABLENODEREUSE=1 dotnet publish "${repo_root}/src/${project}/${project}.csproj" \
        -c Release -r osx-arm64 --self-contained true -p:PublishSingleFile=false \
        -p:PublishReadyToRun="${SHARPEMU_MAC_R2R:-true}" -p:PublishDir="${arm64_publish_dir}/" \
        -m:"${build_workers}" -nr:false -p:UseSharedCompilation=false -p:SharpEmuBuildProfile=macos-bundle
    if [ ! -x "${arm64_publish_dir}/${project}" ]; then
        echo "build-app.sh: expected ${arm64_publish_dir}/${project} to exist after publish." >&2
        exit 1
    fi
done

# --------------------------------------------------------------- assemble ----

# Contents/MacOS holds the whole payload, not just the apphost. The managed
# plugin resolver (SharpEmu.CLI/Program.cs) and FFmpeg's RootPath
# (SharpEmu.Libs/Media/FfmpegRuntime.cs) both resolve
# "<AppContext.BaseDirectory>/plugins", and BaseDirectory is the apphost's own
# directory. Keeping the publish layout intact under Contents/MacOS makes those
# lookups work with no @rpath edits and no code change.
mkdir -p "${app_dir}/Contents/MacOS" "${app_dir}/Contents/Resources"
ditto "${publish_dir}" "${app_dir}/Contents/MacOS"
if [ "${#arm64_projects[@]}" -gt 0 ]; then
    ditto "${arm64_publish_dir}" "${app_dir}/Contents/MacOS/${arm64_dir}"
fi

# The bundle's main executable is the GUI host's apphost, moved up into
# Contents/MacOS (CFBundleExecutable cannot name a subdirectory) and pointed at
# arm64/SharpEmu.GuiHost.dll: an apphost resolves its embedded app path relative
# to its own directory and loads the runtime from beside the app, so it starts
# natively on the arm64 runtime while the x64 payload around it stays untouched.
main_executable="${executable_name}"
main_arch="x86_64"
if [ "${native_gui}" = "1" ]; then
    main_executable="${gui_executable_name}"
    main_arch="arm64"
    python3 -I - "${app_dir}/Contents/MacOS/${arm64_dir}/SharpEmu.GuiHost" \
        "${app_dir}/Contents/MacOS/${main_executable}" "${arm64_dir}/SharpEmu.GuiHost.dll" <<'PY'
import os, sys
source, target, app_path = sys.argv[1:]
image = bytearray(open(source, "rb").read())
name = b"SharpEmu.GuiHost.dll\0"
at = image.find(name)
if at < 0 or image.find(name, at + 1) >= 0:
    sys.exit("build-app.sh: the GUI host apphost has no unique embedded app path")
# The apphost reserves 1024 bytes for the path, NUL included.
image[at:at + 1024] = bytes(1024)
encoded = app_path.encode() + b"\0"
image[at:at + len(encoded)] = encoded
open(target, "wb").write(image)
os.chmod(target, 0o755)
PY
fi
python3 "${script_dir}/build-metadata.py" embed \
    --repo "${repo_root}" --snapshot "${source_snapshot}" --sdk "${sdk_version}" \
    --output "${app_dir}/Contents/Resources/build-info.json"

cp "${icon_source}" "${app_dir}/Contents/Resources/${icon_file}"

sed \
    -e "s|@SHARPEMU_BUNDLE_ID@|${bundle_id}|g" \
    -e "s|@SHARPEMU_EXECUTABLE@|${main_executable}|g" \
    -e "s|@SHARPEMU_MAIN_ARCH@|${main_arch}|g" \
    -e "s|@SHARPEMU_ICON_FILE@|${icon_file}|g" \
    -e "s|@SHARPEMU_SHORT_VERSION@|${version}|g" \
    -e "s|@SHARPEMU_BUNDLE_VERSION@|${bundle_version}|g" \
    -e "s|@SHARPEMU_MIN_MACOS@|${min_macos}|g" \
    "${platform_dir}/Info.plist.in" > "${output_dir}/Info.plist.staged"

# plutil -convert, not a plain copy: it validates the plist and, as a side
# effect, drops the XML comments that document why each key is there. The
# template keeps the comments; the shipped plist does not need them.
plutil -lint "${output_dir}/Info.plist.staged"
plutil -convert xml1 -o "${app_dir}/Contents/Info.plist" "${output_dir}/Info.plist.staged"
rm -f "${output_dir}/Info.plist.staged"

# Legacy but still read by some Finder/LaunchServices paths, and free to emit.
printf 'APPL????' > "${app_dir}/Contents/PkgInfo"

# ------------------------------------------------------------ architecture ----

# Several NuGet natives (libSkiaSharp, libHarfBuzzSharp) ship universal
# x86_64+arm64. Only the x86_64 slice can ever load into this process, so the
# arm64 half is dead weight inside the bundle. Thinning also makes the
# architecture check below exact rather than "contains x86_64". The GPU host's
# payload is the mirror image and keeps only its arm64 slices.
if [ "${SHARPEMU_MAC_KEEP_UNIVERSAL:-0}" != "1" ]; then
    thinned=0
    while IFS= read -r binary; do
        [ -n "${binary}" ] || continue
        case "$(file -b "${binary}")" in
            *"universal binary"*) ;;
            *) continue ;;
        esac
        case "${binary}" in
            "${app_dir}/Contents/MacOS/${arm64_dir}"/* | "${app_dir}/Contents/MacOS/${gui_executable_name}") slice=arm64 ;;
            *) slice=x86_64 ;;
        esac
        case " $(lipo -archs "${binary}" 2>/dev/null || true) " in
            *" ${slice} "*) ;;
            *) continue ;;
        esac
        lipo -thin "${slice}" "${binary}" -output "${binary}.${slice}"
        mv "${binary}.${slice}" "${binary}"
        thinned=$((thinned + 1))
    done <<EOF
$(find "${app_dir}" -type f '(' -name '*.dylib' -o -perm -u+x ')')
EOF
    echo "==> Thinned ${thinned} universal binaries to their process's slice"
fi

# --------------------------------------------------------------- policy ------

check_args=("${app_dir}")
arm64_check_args=("${app_dir}/Contents/MacOS/${arm64_dir}" --arch arm64)
if [ "${SHARPEMU_MAC_FORBID_SDL:-1}" = "1" ]; then
    check_args+=(--forbid-sdl)
    arm64_check_args+=(--forbid-sdl)
fi
if [ "${#arm64_projects[@]}" -gt 0 ]; then
    "${platform_dir}/check-macos-output.sh" "${arm64_check_args[@]}"
    check_args+=(--arch-exclude "Contents/MacOS/${arm64_dir}")
fi
if [ "${native_gui}" = "1" ]; then
    check_args+=(--arch-exclude "Contents/MacOS/${gui_executable_name}")
fi
"${platform_dir}/check-macos-output.sh" "${check_args[@]}"

# ----------------------------------------------------------------- sign ------

# `codesign --deep` is the only workable mode for this layout, even though
# Apple discourages it for signing. codesign treats every file in
# Contents/MacOS as code rather than as a sealed resource, so a plain bundle
# sign fails with "code object is not signed at all / In subcomponent:
# System.Xml.Linq.dll" — the managed assemblies are PE/.NET images that have no
# signature yet. Signing them individually before the bundle (the Apple-
# recommended inside-out order) hits the same wall for the main executable's
# own payload, while --deep signs the nested code and then the bundle in one
# pass and produces the sealed CodeResources the bundle needs.
#
# The known cost: --deep applies this entitlements file to the nested binaries
# too. That is harmless in effect, because nested code inherits the host
# executable's entitlements at runtime regardless, but it does mean the
# entitlement set is not visible per-binary. Revisit if Apple's tooling grows a
# way to seal Contents/MacOS payloads as resources.
echo "==> Signing with identity '${codesign_identity}'"
sign_flags=(--force --deep --options runtime --sign "${codesign_identity}")
if [ "${codesign_identity}" != "-" ]; then
    # Ad-hoc signatures cannot be timestamped; real identities must be, or
    # notarization rejects the submission.
    sign_flags+=(--timestamp)
fi

# codesign's entitlements parser (AMFIUnserializeXML) is stricter than
# CFPropertyList and rejects XML comments outright, so the commented source
# file cannot be handed to it directly. plutil re-emits it without comments;
# the justification for each entitlement stays in
# build/platforms/macos/SharpEmu.entitlements.in.
entitlements="${output_dir}/SharpEmu.entitlements"
plutil -convert xml1 -o "${entitlements}" "${platform_dir}/SharpEmu.entitlements.in"
# Measurement builds only: the hardened runtime denies task_for_pid even to root, which the
# de-duplicated memory sampler (scripts/macos/memstat.c) needs. Never set for releases.
if [ "${SHARPEMU_MAC_GET_TASK_ALLOW:-0}" = "1" ]; then
    /usr/libexec/PlistBuddy -c "Add :com.apple.security.get-task-allow bool true" "${entitlements}"
fi

codesign "${sign_flags[@]}" --entitlements "${entitlements}" "${app_dir}"

echo "==> Verifying signature"
# --verbose (level 1), not --verbose=2: a deep verify at level 2 prints a
# "prepared/validated" pair for all 266 sealed files and buries the result.
codesign --verify --deep --strict --verbose "${app_dir}"
codesign --display --entitlements - "${app_dir}" 2>/dev/null || true

# `spctl --assess` is expected to fail for an ad-hoc signature; it is only
# meaningful once a Developer ID identity and notarization are in play.
if [ "${codesign_identity}" != "-" ]; then
    spctl --assess --type execute --verbose=4 "${app_dir}" || \
        echo "==> spctl assessment failed (expected until the build is notarized)"
fi

# ------------------------------------------------------------- notarize ------

if [ -n "${SHARPEMU_NOTARY_APPLE_ID:-}" ] && \
   [ -n "${SHARPEMU_NOTARY_PASSWORD:-}" ] && \
   [ -n "${SHARPEMU_NOTARY_TEAM_ID:-}" ]; then
    if [ "${codesign_identity}" = "-" ]; then
        echo "build-app.sh: notarization needs a Developer ID identity; set SHARPEMU_CODESIGN_IDENTITY." >&2
        exit 1
    fi
    notary_zip="${output_dir}/SharpEmu-notarize.zip"
    rm -f "${notary_zip}"
    ditto -c -k --keepParent "${app_dir}" "${notary_zip}"
    echo "==> Submitting to notarytool"
    xcrun notarytool submit "${notary_zip}" \
        --apple-id "${SHARPEMU_NOTARY_APPLE_ID}" \
        --password "${SHARPEMU_NOTARY_PASSWORD}" \
        --team-id "${SHARPEMU_NOTARY_TEAM_ID}" \
        --wait
    xcrun stapler staple "${app_dir}"
    xcrun stapler validate "${app_dir}"
    rm -f "${notary_zip}"
else
    echo "==> Notarization skipped (no SHARPEMU_NOTARY_* credentials); local builds never need it."
fi

# -------------------------------------------------------------- package ------

# ditto, not zip: it preserves the resource forks, symlinks and extended
# attributes that a signed bundle's signature covers. A plain `zip` round trip
# can invalidate the signature.
archive="${output_dir}/sharpemu-${version}-${rid}-app.zip"
rm -f "${archive}"
ditto -c -k --keepParent "${app_dir}" "${archive}"

if [ "${SHARPEMU_MAC_DMG:-0}" = "1" ]; then
    dmg="${output_dir}/sharpemu-${version}-${rid}.dmg"
    rm -f "${dmg}"
    hdiutil create -quiet -volname "SharpEmu ${version}" \
        -srcfolder "${app_dir}" -ov -format UDZO "${dmg}"
    echo "==> ${dmg}"
fi

echo
echo "==> ${app_dir}"
echo "==> ${archive}"
echo "    run it with: '${app_dir}/Contents/MacOS/${executable_name}' <path-to-eboot.bin>"
