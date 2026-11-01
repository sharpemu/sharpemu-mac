// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections;
using System.Runtime.InteropServices;

namespace SharpEmu.GpuRemote;

/// <summary>
/// Starts the GPU host with <c>posix_spawn</c>, natively arm64 even from the x86-64 emulator.
/// </summary>
/// <remarks>
/// <para>Why not <see cref="System.Diagnostics.Process"/>: it cannot state an architecture
/// preference, and it would keep the child in its own SIGCHLD bookkeeping while
/// <see cref="ProcessExitWatch"/> reaps it.</para>
///
/// <para>Architecture: a process translated by Rosetta that spawns a universal binary gets the
/// x86-64 slice by default. <c>posix_spawnattr_setarchpref_np</c> (macOS 11) asks for arm64
/// first. The GPU host's apphost is a thin arm64 Mach-O, which runs natively whatever the
/// preference, so this only matters if it is ever built universal; the host reports
/// <c>sysctl.proc_translated</c> in its handshake and <see cref="GpuProcessHost"/> rejects a
/// translated one.</para>
///
/// <para>Hygiene: <c>POSIX_SPAWN_CLOEXEC_DEFAULT</c> closes every descriptor except stdin,
/// stdout and stderr (the guest-memory backing descriptors of the emulator must not leak into
/// the child), every signal starts at its default disposition and unblocked, and the child
/// stays in this process group so the GUI's Stop, which signals the group, takes it down too.
/// The environment is passed through, plus a gen0 budget for the host's GC (EnvironmentBlock).</para>
/// </remarks>
internal static unsafe class GpuProcessLauncher
{
    public static int Spawn(string executable, IReadOnlyList<string> arguments)
    {
        var allocations = new List<nint>();
        nint attributes = 0;
        nint actions = 0;
        var attributesReady = false;
        var actionsReady = false;
        try
        {
            var argv = NativeStringArray([executable, .. arguments], allocations);
            var envp = NativeStringArray(EnvironmentBlock(), allocations);
            var path = NativeString(executable, allocations);

            Check(RemoteNative.posix_spawnattr_init(&attributes), "posix_spawnattr_init");
            attributesReady = true;
            Check(RemoteNative.posix_spawn_file_actions_init(&actions), "posix_spawn_file_actions_init");
            actionsReady = true;

            Check(RemoteNative.posix_spawnattr_setflags(
                &attributes,
                RemoteNative.POSIX_SPAWN_SETSIGDEF | RemoteNative.POSIX_SPAWN_SETSIGMASK | RemoteNative.POSIX_SPAWN_CLOEXEC_DEFAULT),
                "posix_spawnattr_setflags");
            var defaults = ~((1u << (RemoteNative.SIGKILL - 1)) | (1u << (RemoteNative.SIGSTOP - 1)));
            Check(RemoteNative.posix_spawnattr_setsigdefault(&attributes, &defaults), "posix_spawnattr_setsigdefault");
            var mask = 0u;
            Check(RemoteNative.posix_spawnattr_setsigmask(&attributes, &mask), "posix_spawnattr_setsigmask");

            var cpuType = RemoteNative.CPU_TYPE_ARM64;
            var cpuSubtype = RemoteNative.CPU_SUBTYPE_ANY;
            nuint copied = 0;
            Check(RemoteNative.posix_spawnattr_setarchpref_np(&attributes, 1, &cpuType, &cpuSubtype, &copied), "posix_spawnattr_setarchpref_np");

            for (var descriptor = 0; descriptor <= 2; descriptor++)
            {
                Check(RemoteNative.posix_spawn_file_actions_addinherit_np(&actions, descriptor), "posix_spawn_file_actions_addinherit_np");
            }

            int pid;
            var error = RemoteNative.posix_spawn(&pid, (byte*)path, &actions, &attributes, (byte**)argv, (byte**)envp);
            if (error != 0)
            {
                throw new IOException($"posix_spawn {executable} failed: errno {error}");
            }

            return pid;
        }
        finally
        {
            if (actionsReady)
            {
                _ = RemoteNative.posix_spawn_file_actions_destroy(&actions);
            }

            if (attributesReady)
            {
                _ = RemoteNative.posix_spawnattr_destroy(&attributes);
            }

            foreach (var allocation in allocations)
            {
                NativeMemory.Free((void*)allocation);
            }
        }
    }

    // The GPU host allocates some 140-450 MB a second (Ghost of Tsushima's world), so with the
    // runtime's default gen0 budget it collected gen0 ~20 times a second and its threads stood in
    // GC pauses ~9% of the time; a 64 MB budget measured 4-6% and +5% frame rate. 256 MB was slower
    // (each collection scans more). A gen0 size the user set (DOTNET_GCgen0size, 0 = the runtime's
    // own) is kept. The runtimeconfig knob System.GC.Gen0Size is not read by the GC.
    private const string Gen0SizeVariable = "DOTNET_GCgen0size";
    private const string HostGen0Size = "0x4000000";

    private static List<string> EnvironmentBlock()
    {
        var block = new List<string>();
        var gen0Set = false;
        foreach (DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            block.Add($"{entry.Key}={entry.Value}");
            gen0Set |= string.Equals(entry.Key as string, Gen0SizeVariable, StringComparison.OrdinalIgnoreCase);
        }

        if (!gen0Set)
        {
            block.Add($"{Gen0SizeVariable}={HostGen0Size}");
        }

        return block;
    }

    private static nint NativeStringArray(IReadOnlyList<string> values, List<nint> allocations)
    {
        var array = (byte**)NativeMemory.AllocZeroed((nuint)(values.Count + 1), (nuint)sizeof(nint));
        allocations.Add((nint)array);
        for (var i = 0; i < values.Count; i++)
        {
            array[i] = (byte*)NativeString(values[i], allocations);
        }

        return (nint)array;
    }

    private static nint NativeString(string value, List<nint> allocations)
    {
        var bytes = RemoteNative.NullTerminated(value);
        var native = (byte*)NativeMemory.Alloc((nuint)bytes.Length);
        allocations.Add((nint)native);
        bytes.CopyTo(new Span<byte>(native, bytes.Length));
        return (nint)native;
    }

    private static void Check(int error, string call)
    {
        if (error != 0)
        {
            throw new IOException($"{call} failed: errno {error}");
        }
    }
}
