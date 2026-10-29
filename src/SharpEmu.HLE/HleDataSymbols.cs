// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using System.Text;

namespace SharpEmu.HLE;

public static class HleDataSymbols
{
    private const string StackChkGuardNid = "f7uOxY9mM1U";
    private const string ProgNameNid = "djxxOmW6-aw";
    private const string LibcNeedFlagNid = "P330P3dFF68";
    private const string LibcInternalNeedFlagNid = "ZT4ODD2Ts9o";
    // char **environ: the process starts with an empty environment.
    private const string EnvironNid = "+2thxYZ4syk";
    // FreeBSD libc FILE * variables and the Dinkumware FILE objects they
    // point at. Used only when no loaded libc module defines them
    // (SharpEmuRuntime.MergeKnownHleDataSymbols skips symbols a module has).
    private const string StdinPointerNid = "bgAcsbcEznc";   // __stdinp
    private const string StdoutPointerNid = "zqJhBxAKfsc";  // __stdoutp
    private const string StderrPointerNid = "as8Od-tH1BI";  // __stderrp
    private const string StdinObjectNid = "1TDo-ImqkJc";    // _Stdin
    private const string StdoutObjectNid = "2sWzhYqFH4E";   // _Stdout
    private const string StderrObjectNid = "H8AprKeZtNg";   // _Stderr
    // Opaque, zeroed FILE storage. The guest owns the layout and writes it:
    // the PS5 libc.prx copies a whole 0x1C8-byte FILE image into each of these
    // at startup (Quake), so every object must hold at least that much. They
    // sit in the host malloc heap; an object smaller than the guest's FILE lets
    // that copy run into the next host allocation (CoreCLR's string-literal
    // table, its hash maps), which aborts the process seconds later from an
    // unrelated thread. Twice the observed size leaves room for other layouts.
    internal const int MinimumGuestFileSize = 0x1C8;
    internal const int StdioObjectSize = 0x400;
    private const int ProgNameMaxBytes = 511;
    // Terminator canaries reserve the low byte as NUL. Keep the process data
    // symbol and every per-thread TLS copy byte-for-byte identical.
    private const ulong StackChkGuardValue = 0xC0DEC0DECAFEBA00UL;

    private static readonly object _gate = new();
    private static readonly nint _stackChkGuardAddress = Allocate(sizeof(ulong) * 2);
    private static readonly nint _progNameBufferAddress = Allocate(ProgNameMaxBytes + 1);
    private static readonly nint _progNamePointerAddress = Allocate(nint.Size);
    private static readonly nint _argumentVectorAddress = Allocate(nint.Size * 2);
    private static readonly nint _libcNeedFlagAddress = Allocate(sizeof(uint));
    private static readonly nint _libcInternalNeedFlagAddress = Allocate(sizeof(uint));
    private static readonly nint _environVectorAddress = Allocate(nint.Size);
    private static readonly nint _environPointerAddress = Allocate(nint.Size);
    private static readonly nint _stdioObjectsAddress = Allocate(StdioObjectSize * 3);
    private static readonly nint _stdioPointersAddress = Allocate(nint.Size * 3);

    static HleDataSymbols()
    {
        if (_stackChkGuardAddress != 0)
        {
            Marshal.WriteInt64(_stackChkGuardAddress, unchecked((long)StackChkGuardValue));
            Marshal.WriteInt64(
                IntPtr.Add(_stackChkGuardAddress, sizeof(ulong)),
                unchecked((long)StackChkGuardValue));
        }

        if (_libcNeedFlagAddress != 0)
        {
            Marshal.WriteInt32(_libcNeedFlagAddress, 1);
        }

        if (_libcInternalNeedFlagAddress != 0)
        {
            Marshal.WriteInt32(_libcInternalNeedFlagAddress, 1);
        }

        if (_environPointerAddress != 0 && _environVectorAddress != 0)
        {
            // environ -> { NULL }
            WritePointer(_environPointerAddress, _environVectorAddress);
        }

        if (_stdioObjectsAddress != 0 && _stdioPointersAddress != 0)
        {
            for (var i = 0; i < 3; i++)
            {
                WritePointer(IntPtr.Add(_stdioPointersAddress, i * nint.Size), StdioObject(i));
            }
        }

        ConfigureProcessImageName("eboot.bin");
    }

    private static nint StdioObject(int index) =>
        _stdioObjectsAddress == 0 ? 0 : IntPtr.Add(_stdioObjectsAddress, index * StdioObjectSize);

    private static nint StdioPointer(int index) =>
        _stdioPointersAddress == 0 ? 0 : IntPtr.Add(_stdioPointersAddress, index * nint.Size);

    /// <summary>
    /// The HLE FILE object for stdin (0), stdout (1) or stderr (2), or 0.
    /// </summary>
    public static ulong StandardStreamAddress(int index) =>
        index is >= 0 and <= 2 ? unchecked((ulong)StdioObject(index)) : 0;

    public static IEnumerable<string> EnumerateKnownNids()
    {
        yield return StackChkGuardNid;
        yield return ProgNameNid;
        yield return LibcNeedFlagNid;
        yield return LibcInternalNeedFlagNid;
        yield return EnvironNid;
        yield return StdinPointerNid;
        yield return StdoutPointerNid;
        yield return StderrPointerNid;
        yield return StdinObjectNid;
        yield return StdoutObjectNid;
        yield return StderrObjectNid;
    }

    // The full argv getargv reports once the process arguments are known; until then the
    // one-entry vector of the program name.
    private static nint _extendedArgumentVector;
    private static int _extendedArgumentCount;

    public static int ProcessArgumentCount =>
        Volatile.Read(ref _extendedArgumentVector) != 0
            ? Volatile.Read(ref _extendedArgumentCount)
            : _progNameBufferAddress != 0 && _argumentVectorAddress != 0 ? 1 : 0;

    public static ulong ProcessArgumentVectorAddress =>
        unchecked((ulong)(Volatile.Read(ref _extendedArgumentVector) is var extended and not 0
            ? extended
            : _argumentVectorAddress));

    /// <summary>Publishes the process's whole argv (argv[0] first) for getargc/getargv, the same
    /// list the entry parameters carry. Process lifetime: the memory is never freed.</summary>
    public static void ConfigureProcessArguments(IReadOnlyList<string> arguments)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        if (arguments.Count <= 1)
        {
            return;
        }

        var vector = Allocate(nint.Size * (arguments.Count + 1));
        if (vector == 0)
        {
            return;
        }

        for (var index = 0; index < arguments.Count; index++)
        {
            var encoded = Encoding.UTF8.GetBytes(arguments[index] + '\0');
            var text = Allocate(encoded.Length);
            if (text == 0)
            {
                return;
            }

            Marshal.Copy(encoded, 0, text, encoded.Length);
            WritePointer(IntPtr.Add(vector, index * nint.Size), text);
        }

        lock (_gate)
        {
            Volatile.Write(ref _extendedArgumentCount, arguments.Count);
            Volatile.Write(ref _extendedArgumentVector, vector);
        }
    }

    public static void ConfigureProcessImageName(string? processImageName)
    {
        var effectiveName = string.IsNullOrWhiteSpace(processImageName)
            ? "eboot.bin"
            : processImageName;
        var encodedName = Encoding.UTF8.GetBytes(effectiveName);
        var byteCount = Math.Min(encodedName.Length, ProgNameMaxBytes);

        lock (_gate)
        {
            if (_progNameBufferAddress == 0 ||
                _progNamePointerAddress == 0 ||
                _argumentVectorAddress == 0)
            {
                return;
            }

            for (var i = 0; i <= ProgNameMaxBytes; i++)
            {
                Marshal.WriteByte(_progNameBufferAddress, i, 0);
            }

            Marshal.Copy(encodedName, 0, _progNameBufferAddress, byteCount);
            WritePointer(_progNamePointerAddress, _progNameBufferAddress);
            WritePointer(_argumentVectorAddress, _progNameBufferAddress);
            WritePointer(IntPtr.Add(_argumentVectorAddress, nint.Size), 0);
        }
    }

    public static bool TryGetAddress(string nid, out ulong address)
    {
        var pointer = nid switch
        {
            StackChkGuardNid => _stackChkGuardAddress,
            ProgNameNid => _progNamePointerAddress,
            LibcNeedFlagNid => _libcNeedFlagAddress,
            LibcInternalNeedFlagNid => _libcInternalNeedFlagAddress,
            EnvironNid => _environPointerAddress,
            StdinPointerNid => StdioPointer(0),
            StdoutPointerNid => StdioPointer(1),
            StderrPointerNid => StdioPointer(2),
            StdinObjectNid => StdioObject(0),
            StdoutObjectNid => StdioObject(1),
            StderrObjectNid => StdioObject(2),
            _ => 0,
        };

        if (pointer == 0)
        {
            address = 0;
            return false;
        }

        address = unchecked((ulong)pointer);
        return true;
    }

    private static nint Allocate(int size)
    {
        try
        {
            var memory = Marshal.AllocHGlobal(size);
            for (var i = 0; i < size; i++)
            {
                Marshal.WriteByte(memory, i, 0);
            }

            return memory;
        }
        catch
        {
            return 0;
        }
    }

    private static void WritePointer(nint target, nint value)
    {
        if (nint.Size == sizeof(int))
        {
            Marshal.WriteInt32(target, value.ToInt32());
            return;
        }

        Marshal.WriteInt64(target, value.ToInt64());
    }
}
