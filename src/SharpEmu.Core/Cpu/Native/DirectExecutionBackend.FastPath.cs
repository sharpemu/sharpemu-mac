// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using SharpEmu.HLE;

using SharpEmu.Core.Cpu.Native.Platform;

namespace SharpEmu.Core.Cpu.Native;

/// <summary>
/// Native fast-path stubs for the trivial per-thread imports that dominate the
/// HLE call count. Each stub answers from <see cref="GuestFastPath"/>'s
/// per-host-thread block and jumps to the normal import trampoline whenever it
/// cannot, so the managed implementation stays the single source of truth for
/// every case the stub does not handle.
/// </summary>
public sealed unsafe partial class DirectExecutionBackend
{
	internal enum GuestFastPathStub
	{
		None,
		PthreadSelf,
		PthreadGetspecific,
		MemoryCopy,
		MemorySet,

		/// <summary>
		/// An export whose whole observable effect is RAX = 0. The thunk is
		/// <c>xor eax,eax; ret</c>, so the guest never crosses into managed code.
		/// </summary>
		ConstantZero,

		/// <summary>clock_gettime / sceKernelClockGettime for the realtime and monotonic clock ids.</summary>
		ClockGettime,

		/// <summary>gettimeofday(tv, NULL).</summary>
		GetTimeOfDay,

		/// <summary>sceKernelGetProcessTime: microseconds since process start.</summary>
		ProcessTimeMicroseconds,

		/// <summary>
		/// An sceAgc command builder that appends one fixed-shape packet (see <see cref="SharpEmu.Libs.Agc.AgcPacketStubSpec"/>):
		/// the stub stores the dwords and the cursor itself, as the title's own code does.
		/// </summary>
		AgcPacket,
	}

	// clock_gettime / sceKernelClockGettime / gettimeofday / sceKernelGetProcessTime.
	private const string ClockGettimeNid = "lLMT9vJAck0";
	private const string KernelClockGettimeNid = "QBi7HCK03hw";
	private const string GettimeofdayNid = "n88vx3C5nW8";
	private const string KernelGetProcessTimeNid = "4J2sUJmuHZQ";

	/// <summary>
	/// What a clock stub calls and subtracts: the host's monotonic counter (the same
	/// <c>mach_absolute_time</c> the managed Stopwatch reads, which makes the stub's answer
	/// exactly the managed export's), the process start in that counter, and the host realtime clock.
	/// </summary>
	internal readonly record struct ClockStubTargets(nint MachAbsoluteTime, long ProcessStartCounter, nint ClockGettime);

	// SHARPEMU_AGC_PACKET_STUBS=0 keeps the sceAgc command builders on the managed exports.
	private static readonly bool AgcPacketStubsEnabled = SharpEmu.Libs.Agc.AgcExports.PacketStubsEnabled;

	// SHARPEMU_AGC_PATCH_STUBS=0 keeps the register-indirect patchers on the managed exports.
	private static readonly bool AgcPatchStubsEnabled = SharpEmu.Libs.Agc.AgcExports.PatchStubsEnabled;

	// SHARPEMU_HLE_FAST_CLOCK=0 keeps the clock reads on the managed exports.
	private static readonly bool ClockStubsEnabled =
		!string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_HLE_FAST_CLOCK"), "0", StringComparison.Ordinal);

	private ClockStubTargets? _clockStubTargets;

	// scePthreadSelf / pthread_self.
	private const string PthreadSelfNid = "aI+OeCz8xrQ";
	private const string PosixPthreadSelfNid = "EotR8a3ASf4";

	// scePthreadGetspecific / pthread_getspecific.
	private const string PthreadGetspecificNid = "eoht7mQOCmo";
	private const string PosixPthreadGetspecificNid = "0-KXaS70xy4";

	// memcpy / memmove.
	private const string MemcpyNid = "Q3VBxCXhUHs";
	private const string MemmoveNid = "+P6FRGH4LfA";
	private const string MemsetNid = "8zTFvBIAIN8";

	// Guest memory is identity-mapped and every fault the copy can take (write tracking, lazy
	// commit) is resolved from the fault address alone, so the copy may run as guest-side code.
	// SHARPEMU_HLE_FAST_MEMCPY=0 keeps it on the managed export.
	private static readonly bool GuestMemoryCopyStubEnabled =
		!string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_HLE_FAST_MEMCPY"), "0", StringComparison.Ordinal);

	// SHARPEMU_NATIVE_IMPORT_THUNKS=0 routes the constant-result imports back
	// through the managed gateway, which is the way to tell a thunk apart from
	// the export it stands in for.
	private static readonly bool NativeImportThunksEnabled =
		!string.Equals(
			Environment.GetEnvironmentVariable("SHARPEMU_NATIVE_IMPORT_THUNKS"),
			"0",
			StringComparison.Ordinal);

	/// <summary>
	/// Imports whose managed export does nothing but set RAX to 0 and report
	/// success, for every argument and in every state.
	/// </summary>
	/// <remarks>
	/// These are the unimplemented input and gesture surfaces a title polls
	/// every frame regardless: Quake crosses into managed code 22.1 million
	/// times in two minutes for the five hottest of them, and the import
	/// profile measures their export bodies at 23-42 ns behind a ~300 ns
	/// crossing. A native thunk removes the crossing rather than shortening it.
	///
	/// The list is only sound while those exports really are constant, so
	/// NativeImportThunkTests drives each managed export with varied arguments
	/// and fails if one ever answers with anything but RAX = 0 - which is what
	/// would happen the day somebody implements it for real.
	/// </remarks>
	private static bool IsConstantZeroImport(string nid) => nid is
		// libSceSystemGesture is entirely unimplemented; a title's touch
		// pipeline polls these on every frame.
		"3pcAvmwKCvM" or // sceSystemGestureInitializePrimitiveTouchRecognizer
		"FWF8zkhr854" or // sceSystemGestureCreateTouchRecognizer
		"qpo-mEOwje0" or // sceSystemGestureOpen
		"GgFMb22sbbI" or // sceSystemGestureUpdatePrimitiveTouchRecognizer
		"j4h82CQWENo" or // sceSystemGestureUpdateTouchRecognizer
		"h8uongcBNVs" or // sceSystemGestureGetTouchEventsCount
		"1MMK0W-kMgA" or // sceSystemGestureAppendTouchRecognizer
		"wPJGwI2RM2I" or // sceSystemGestureUpdateAllTouchRecognizer
		// Keyboard and mouse surfaces the host does not present to the guest.
		"-4GCfYdNF1s" or // sceImeUpdate
		"x8qnXqh-tiM";   // sceMouseRead

	private uint _fastPathBlockTlsIndex = uint.MaxValue;
	private bool _guestFastPathEnabled;
	private bool _nativeConstantThunksAllowed;
	private int _guestFastPathStubCount;
	private int _nativeConstantThunkCount;

	/// <summary>
	/// Decides whether native fast-path stubs may be installed for this run and
	/// arms <see cref="GuestFastPath"/> if so. Any diagnostic mode that must
	/// observe every import call keeps the fast path off, because a stub answers
	/// without ever reaching the managed gateway that does the tracing.
	/// </summary>
	private void ConfigureGuestFastPath()
	{
		_guestFastPathEnabled = false;
		_guestFastPathStubCount = 0;
		_nativeConstantThunkCount = 0;

		// A constant thunk answers without reaching the managed gateway, so a
		// diagnostic mode that has to observe every import call rules it out -
		// the same reason the TLS stubs below are gated. It needs no TLS read,
		// so it survives on a platform whose TLS access is a real call, and it
		// needs no per-thread block, so it does not depend on GuestFastPath
		// being armed at all.
		_nativeConstantThunksAllowed =
			RuntimeInformation.ProcessArchitecture == Architecture.X64 &&
			!_logAllImports &&
			string.IsNullOrEmpty(_importFilter);

		_clockStubTargets = ResolveClockStubTargets();

		// Emission needs a TLS read the platform can express as inline code
		// (the Darwin pthread gs-indexed key array). Platforms whose TLS read is
		// a real call keep the managed gateway.
		if ((ExecutionPlatform.Capabilities & DirectExecutionCapabilities.EmittedTlsFastPath) == 0 ||
			RuntimeInformation.ProcessArchitecture != Architecture.X64)
		{
			return;
		}

		if (string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_HLE_FAST_PATH"), "0", StringComparison.Ordinal))
		{
			Console.Error.WriteLine("[LOADER][INFO] HLE native fast path disabled by SHARPEMU_HLE_FAST_PATH=0.");
			return;
		}

		if (_logAllImports ||
			!string.IsNullOrEmpty(_importFilter) ||
			string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_PTHREADS"), "1", StringComparison.Ordinal))
		{
			Console.Error.WriteLine(
				"[LOADER][INFO] HLE native fast path disabled: import/pthread tracing needs every call in the gateway.");
			return;
		}

		if (_fastPathBlockTlsIndex == uint.MaxValue)
		{
			_fastPathBlockTlsIndex = TlsAlloc();
			if (_fastPathBlockTlsIndex == uint.MaxValue)
			{
				return;
			}
		}

		var tlsIndex = _fastPathBlockTlsIndex;
		GuestFastPath.Enable(block => TlsSetValue(tlsIndex, block));
		_guestFastPathEnabled = true;
	}

	private ClockStubTargets? ResolveClockStubTargets()
	{
		// The stubs read the host clocks without entering managed code, so anything that must see
		// every import call - and the import-loop guard, which time reads reset - rules them out.
		if (!ClockStubsEnabled ||
			RuntimeInformation.ProcessArchitecture != Architecture.X64 ||
			!_nativeConstantThunksAllowed ||
			!NativeImportThunksEnabled ||
			GuestWriteWatch.Armed ||
			string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_DISABLE_IMPORT_LOOP_GUARD"), "0", StringComparison.Ordinal) ||
			!SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.HostCounterMatchesProcessTimeFrequency)
		{
			return null;
		}

		// The platform names the routines (and vouches that its counter ticks in nanoseconds, as
		// Stopwatch does), so no OS-specific code lives here.
		if (!ExecutionPlatform.TryGetHostClockEntryPoints(out var counter, out var clockGettime) ||
			counter == 0 ||
			clockGettime == 0)
		{
			return null;
		}

		return new ClockStubTargets(counter, SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.ProcessStartCounter, clockGettime);
	}

	internal static GuestFastPathStub ClassifyGuestFastPathStub(string nid) => nid switch
	{
		PthreadSelfNid or PosixPthreadSelfNid => GuestFastPathStub.PthreadSelf,
		PthreadGetspecificNid or PosixPthreadGetspecificNid => GuestFastPathStub.PthreadGetspecific,
		MemcpyNid or MemmoveNid => GuestFastPathStub.MemoryCopy,
		MemsetNid => GuestFastPathStub.MemorySet,
		ClockGettimeNid or KernelClockGettimeNid => GuestFastPathStub.ClockGettime,
		GettimeofdayNid => GuestFastPathStub.GetTimeOfDay,
		KernelGetProcessTimeNid => GuestFastPathStub.ProcessTimeMicroseconds,
		_ => IsConstantZeroImport(nid)
			? GuestFastPathStub.ConstantZero
			: SharpEmu.Libs.Agc.AgcExports.TryGetPacketStubSpec(nid, out _) || SharpEmu.Libs.Agc.AgcExports.TryGetVariablePacketStub(nid, out _) ||
				SharpEmu.Libs.Agc.AgcExports.TryGetRegIndirectPatchStub(nid, out _)
					? GuestFastPathStub.AgcPacket
					: GuestFastPathStub.None,
	};

	/// <summary>
	/// Builds the native stub for <paramref name="nid"/>, falling back to
	/// <paramref name="fallbackTrampoline"/> (the import's managed gateway
	/// trampoline) for every case the stub declines.
	/// </summary>
	private unsafe bool TryCreateGuestFastPathStub(string nid, nint fallbackTrampoline, out nint address)
	{
		address = 0;
		if (fallbackTrampoline == 0)
		{
			return false;
		}

		var kind = ClassifyGuestFastPathStub(nid);
		if (kind == GuestFastPathStub.None ||
			// The write watch inspects every managed guest write, which a native copy never reaches.
			(kind is GuestFastPathStub.MemoryCopy or GuestFastPathStub.MemorySet && (!GuestMemoryCopyStubEnabled || GuestWriteWatch.Armed)))
		{
			return false;
		}

		if (kind == GuestFastPathStub.ConstantZero)
		{
			// A constant answer needs neither the per-thread block nor a
			// fallback, so it does not depend on the emitted TLS read the other
			// stubs do - but it is still invisible to import tracing, which has
			// to see every call.
			if (!NativeImportThunksEnabled || !_nativeConstantThunksAllowed)
			{
				return false;
			}
		}
		else if (kind is GuestFastPathStub.ClockGettime or GuestFastPathStub.GetTimeOfDay or GuestFastPathStub.ProcessTimeMicroseconds)
		{
			// Needs neither the per-thread block nor the emitted TLS read.
			if (_clockStubTargets is null)
			{
				return false;
			}
		}
		else if (kind == GuestFastPathStub.AgcPacket)
		{
			// Stores into the command buffer from guest-side code, so the write watch (which
			// inspects managed writes only) and any mode that has to see every builder call
			// (import and AGC tracing) rule it out; it needs no per-thread block.
			if (!AgcPacketStubsEnabled ||
				!NativeImportThunksEnabled ||
				!_nativeConstantThunksAllowed ||
				GuestWriteWatch.Armed ||
				SharpEmu.Libs.Agc.AgcExports.PacketStubsBlockedByTracing ||
				(SharpEmu.Libs.Agc.AgcExports.TryGetRegIndirectPatchStub(nid, out _)
					? !AgcPatchStubsEnabled
					: !(SharpEmu.Libs.Agc.AgcExports.TryGetPacketStubSpec(nid, out _) || SharpEmu.Libs.Agc.AgcExports.TryGetVariablePacketStub(nid, out _))))
			{
				return false;
			}
		}
		else if (!_guestFastPathEnabled)
		{
			return false;
		}

		List<byte> code;
		if (kind == GuestFastPathStub.AgcPacket)
		{
			code = SharpEmu.Libs.Agc.AgcExports.TryGetRegIndirectPatchStub(nid, out var patchSpec)
				? EmitGuestAgcRegIndirectPatchStub(patchSpec, fallbackTrampoline)
				: SharpEmu.Libs.Agc.AgcExports.TryGetVariablePacketStub(nid, out var variableKind)
				? EmitGuestAgcVariablePacketStub(variableKind, fallbackTrampoline)
				: EmitGuestAgcPacketStub(SharpEmu.Libs.Agc.AgcExports.TryGetPacketStubSpec(nid, out var packetSpec) ? packetSpec : throw new InvalidOperationException(nid), fallbackTrampoline);
		}
		else
		{
			code = kind is GuestFastPathStub.ClockGettime or GuestFastPathStub.GetTimeOfDay or GuestFastPathStub.ProcessTimeMicroseconds
				? EmitGuestClockStub(kind, _clockStubTargets!.Value, fallbackTrampoline)
				: EmitGuestFastPathStub(kind, _fastPathBlockTlsIndex, fallbackTrampoline);
		}
		var stubAllocationSize = kind == GuestFastPathStub.AgcPacket ? 512u : 192u;
		if (code.Count > stubAllocationSize)
		{
			return false;
		}

		void* memory = VirtualAlloc(null, stubAllocationSize, 12288u, 64u);
		if (memory == null)
		{
			return false;
		}

		for (var i = 0; i < code.Count; i++)
		{
			((byte*)memory)[i] = code[i];
		}

		uint oldProtect = 0;
		if (!VirtualProtect(memory, stubAllocationSize, 32u, &oldProtect))
		{
			VirtualFree(memory, 0u, 32768u);
			return false;
		}

		FlushInstructionCache(GetCurrentProcess(), memory, (nuint)code.Count);
		address = (nint)memory;
		_importHandlerTrampolines.Add(address);
		_guestFastPathStubCount++;
		if (kind == GuestFastPathStub.ConstantZero)
		{
			_nativeConstantThunkCount++;
		}

		return true;
	}

	/// <summary>
	/// Emits one fast-path stub. The stubs obey the SysV ABI the guest calls
	/// them with: the answer goes in RAX, only RAX and the arithmetic flags are
	/// clobbered (both caller-saved across a call), and on the slow path every
	/// register - including the incoming RAX that carries the variadic vector
	/// count - is restored before jumping to the trampoline.
	/// </summary>
	internal static List<byte> EmitGuestFastPathStub(
		GuestFastPathStub kind,
		uint blockTlsIndex,
		nint fallbackTrampoline)
	{
		if (kind == GuestFastPathStub.MemoryCopy)
		{
			return EmitGuestMemoryCopyStub(fallbackTrampoline);
		}

		if (kind == GuestFastPathStub.MemorySet)
		{
			return EmitGuestMemorySetStub(fallbackTrampoline);
		}

		if (kind == GuestFastPathStub.ConstantZero)
		{
			// xor eax,eax / ret. The guest's incoming RAX carries the SysV
			// variadic vector count, which the managed export overwrote with the
			// result just the same, and only RAX and the flags change - both
			// caller-saved across a call.
			return [0x31, 0xC0, 0xC3];
		}

		var code = new List<byte>(48);
		var slowPathFixups = new List<int>();

		void Emit(params byte[] bytes) => code.AddRange(bytes);

		// Emits a short conditional branch to the slow path; 0x74 is JE, 0x73 is JAE.
		void EmitBranchToSlowPath(byte opcode)
		{
			Emit(opcode, 0x00); // rel8, patched once the slow-path offset is known
			slowPathFixups.Add(code.Count - 1);
		}

		void EmitJumpToSlowPath() => EmitBranchToSlowPath(0x74);

		// push rax: the guest's incoming RAX/AL must survive a slow-path bail.
		Emit(0x50);

		// mov rax, gs:[blockTlsIndex * 8] - the macOS pthread key array is the
		// gs-based thread specific data array, so a key read is a single load.
		Emit(0x65, 0x48, 0x8B, 0x04, 0x25);
		code.AddRange(BitConverter.GetBytes(checked(blockTlsIndex * 8u)));

		// test rax, rax / je slow: no block published on this host thread.
		Emit(0x48, 0x85, 0xC0);
		EmitJumpToSlowPath();

		switch (kind)
		{
			case GuestFastPathStub.PthreadSelf:
				// cmp qword [rax + SelfHandleOffset], 0 / je slow: no guest
				// thread bound here, and a host thread's pthread_self has to
				// reach the managed export for its scheduler registration.
				Emit(0x48, 0x83, 0x78, checked((byte)GuestFastPath.BlockSelfHandleOffset), 0x00);
				EmitJumpToSlowPath();
				// inc qword [rax + SelfHitsOffset]
				Emit(0x48, 0xFF, 0x40, checked((byte)GuestFastPath.BlockSelfHitsOffset));
				// mov rax, [rax + SelfHandleOffset]
				Emit(0x48, 0x8B, 0x40, checked((byte)GuestFastPath.BlockSelfHandleOffset));
				break;

			case GuestFastPathStub.PthreadGetspecific:
				// cmp qword [rax + TlsValuesOffset], 0 / je slow: this thread
				// has no value table published.
				Emit(0x48, 0x83, 0x78, checked((byte)GuestFastPath.BlockTlsValuesOffset), 0x00);
				EmitJumpToSlowPath();
				// cmp rdi, TlsSlotCount / jae slow. An out-of-range or
				// sign-extended negative key is answered by the managed path,
				// which truncates it to int and consults the key registry.
				Emit(0x48, 0x81, 0xFF);
				code.AddRange(BitConverter.GetBytes(GuestFastPath.TlsSlotCount));
				EmitBranchToSlowPath(0x73);
				// inc qword [rax + GetspecificHitsOffset]
				Emit(0x48, 0xFF, 0x40, checked((byte)GuestFastPath.BlockGetspecificHitsOffset));
				// mov rax, [rax + TlsValuesOffset] / mov rax, [rax + rdi*8]
				Emit(0x48, 0x8B, 0x40, checked((byte)GuestFastPath.BlockTlsValuesOffset));
				Emit(0x48, 0x8B, 0x04, 0xF8);
				break;

			default:
				throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown fast-path stub kind.");
		}

		// add rsp, 8 (drop the saved RAX - the result replaces it) / ret
		Emit(0x48, 0x83, 0xC4, 0x08);
		Emit(0xC3);

		var slowPathOffset = code.Count;

		// pop rax / jmp qword [rip+0] - the absolute target follows the
		// instruction, so nothing else is clobbered on the way out.
		Emit(0x58);
		Emit(0xFF, 0x25, 0x00, 0x00, 0x00, 0x00);
		code.AddRange(BitConverter.GetBytes((long)fallbackTrampoline));

		foreach (var fixup in slowPathFixups)
		{
			var delta = slowPathOffset - (fixup + 1);
			if (delta is < 0 or > 127)
			{
				throw new InvalidOperationException("Fast-path stub slow-path branch is out of rel8 range.");
			}

			code[fixup] = (byte)delta;
		}

		return code;
	}

	/// <summary>
	/// Emits a clock stub. Each one checks, before touching anything, that the call is one it can
	/// answer exactly as the managed export does, and jumps to the import's trampoline otherwise.
	/// The answer is computed from the host clocks and written straight to the guest's timespec or
	/// timeval; RAX is 0 (success) and only caller-saved registers and the flags change.
	/// </summary>
	internal static List<byte> EmitGuestClockStub(GuestFastPathStub kind, ClockStubTargets targets, nint fallbackTrampoline)
	{
		var code = new List<byte>(160);
		var slowPathFixups = new List<int>();
		void Emit(params byte[] bytes) => code.AddRange(bytes);
		void EmitImm64(long value) => code.AddRange(BitConverter.GetBytes(value));
		// Conditional near branch (0F 8x rel32) to a label patched once its offset is known.
		void EmitBranch(byte shortOpcode, List<int> fixups)
		{
			Emit(0x0F, (byte)(shortOpcode + 0x10), 0, 0, 0, 0);
			fixups.Add(code.Count - 4);
		}

		void Patch(List<int> fixups)
		{
			foreach (var fixup in fixups)
			{
				var delta = BitConverter.GetBytes(code.Count - (fixup + 4));
				for (var i = 0; i < 4; i++)
				{
					code[fixup + i] = delta[i];
				}
			}
		}

		// call mach_absolute_time, leaving the nanoseconds since process start in RAX.
		void EmitElapsedNanoseconds()
		{
			Emit(0x48, 0xB8); EmitImm64(targets.MachAbsoluteTime);           // mov rax, imm64
			Emit(0xFF, 0xD0);                                                // call rax
			Emit(0x48, 0xB9); EmitImm64(targets.ProcessStartCounter);        // mov rcx, imm64
			Emit(0x48, 0x29, 0xC8);                                          // sub rax, rcx
		}

		// call clock_gettime(CLOCK_REALTIME, rsp) into a 16-byte scratch the caller already reserved.
		void EmitRealtimeIntoStack()
		{
			Emit(0x31, 0xFF);                                                // xor edi, edi
			Emit(0x48, 0x89, 0xE6);                                          // mov rsi, rsp
			Emit(0x48, 0xB8); EmitImm64(targets.ClockGettime);               // mov rax, imm64
			Emit(0xFF, 0xD0);                                                // call rax
		}

		var realtimeFixups = new List<int>();
		switch (kind)
		{
			case GuestFastPathStub.ClockGettime:
			{
				// rdi = clock id, rsi = timespec*. A null pointer, an id outside the realtime and
				// monotonic families, or one with high bits set is the managed export's business.
				Emit(0x48, 0x85, 0xF6);                                      // test rsi, rsi
				EmitBranch(0x74, slowPathFixups);                            // jz slow
				Emit(0x48, 0x83, 0xFF, 0x0C);                                // cmp rdi, 12
				EmitBranch(0x77, slowPathFixups);                            // ja slow
				Emit(0xB8, 0xB0, 0x19, 0x00, 0x00);                          // mov eax, 0x19B0 (monotonic ids 4 5 7 8 11 12)
				Emit(0x0F, 0xA3, 0xF8);                                      // bt eax, edi
				EmitBranch(0x73, realtimeFixups);                            // jnc not-monotonic
				Emit(0x56);                                                  // push rsi (keeps the stack 16-aligned for the call)
				EmitElapsedNanoseconds();
				Emit(0x31, 0xD2);                                            // xor edx, edx
				Emit(0xB9, 0x00, 0xCA, 0x9A, 0x3B);                          // mov ecx, 1000000000
				Emit(0x48, 0xF7, 0xF1);                                      // div rcx -> rax = seconds, rdx = nanoseconds
				Emit(0x5E);                                                  // pop rsi
				Emit(0x48, 0x89, 0x06);                                      // mov [rsi], rax
				Emit(0x48, 0x89, 0x56, 0x08);                                // mov [rsi+8], rdx
				Emit(0x31, 0xC0);                                            // xor eax, eax
				Emit(0xC3);                                                  // ret
				Patch(realtimeFixups);
				Emit(0xB8, 0x01, 0x06, 0x00, 0x00);                          // mov eax, 0x601 (realtime ids 0 9 10)
				Emit(0x0F, 0xA3, 0xF8);                                      // bt eax, edi
				EmitBranch(0x73, slowPathFixups);                            // jnc slow
				Emit(0x56);                                                  // push rsi
				Emit(0x48, 0x83, 0xEC, 0x10);                                // sub rsp, 16
				EmitRealtimeIntoStack();
				Emit(0x48, 0x8B, 0x04, 0x24);                                // mov rax, [rsp]
				Emit(0x48, 0x8B, 0x54, 0x24, 0x08);                          // mov rdx, [rsp+8]
				Emit(0x48, 0x83, 0xC4, 0x10);                                // add rsp, 16
				Emit(0x5E);                                                  // pop rsi
				Emit(0x48, 0x89, 0x06);                                      // mov [rsi], rax
				Emit(0x48, 0x89, 0x56, 0x08);                                // mov [rsi+8], rdx
				Emit(0x31, 0xC0);                                            // xor eax, eax
				Emit(0xC3);                                                  // ret
				break;
			}

			case GuestFastPathStub.GetTimeOfDay:
			{
				// rdi = timeval*, rsi = timezone*. Only the common (tv, NULL) call is answered here.
				Emit(0x48, 0x85, 0xFF);                                      // test rdi, rdi
				EmitBranch(0x74, slowPathFixups);                            // jz slow
				Emit(0x48, 0x85, 0xF6);                                      // test rsi, rsi
				EmitBranch(0x75, slowPathFixups);                            // jnz slow
				Emit(0x57);                                                  // push rdi
				Emit(0x48, 0x83, 0xEC, 0x10);                                // sub rsp, 16
				EmitRealtimeIntoStack();
				Emit(0x48, 0x8B, 0x44, 0x24, 0x08);                          // mov rax, [rsp+8] (nanoseconds)
				Emit(0x31, 0xD2);                                            // xor edx, edx
				Emit(0xB9, 0xE8, 0x03, 0x00, 0x00);                          // mov ecx, 1000
				Emit(0x48, 0xF7, 0xF1);                                      // div rcx -> rax = microseconds
				Emit(0x48, 0x89, 0xC2);                                      // mov rdx, rax
				Emit(0x48, 0x8B, 0x04, 0x24);                                // mov rax, [rsp] (seconds)
				Emit(0x48, 0x83, 0xC4, 0x10);                                // add rsp, 16
				Emit(0x5F);                                                  // pop rdi
				Emit(0x48, 0x89, 0x07);                                      // mov [rdi], rax
				Emit(0x48, 0x89, 0x57, 0x08);                                // mov [rdi+8], rdx
				Emit(0x31, 0xC0);                                            // xor eax, eax
				Emit(0xC3);                                                  // ret
				break;
			}

			case GuestFastPathStub.ProcessTimeMicroseconds:
			{
				Emit(0x48, 0x83, 0xEC, 0x08);                                // sub rsp, 8 (16-align for the call)
				EmitElapsedNanoseconds();
				Emit(0x31, 0xD2);                                            // xor edx, edx
				Emit(0xB9, 0xE8, 0x03, 0x00, 0x00);                          // mov ecx, 1000
				Emit(0x48, 0xF7, 0xF1);                                      // div rcx
				Emit(0x48, 0x83, 0xC4, 0x08);                                // add rsp, 8
				Emit(0xC3);                                                  // ret
				// Nothing to decline: the stub answers every call, but the fallback keeps the layout uniform.
				break;
			}

			default:
				throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a clock stub kind.");
		}

		Patch(slowPathFixups);
		// jmp qword [rip+0] with the absolute trampoline address after it.
		Emit(0xFF, 0x25, 0x00, 0x00, 0x00, 0x00);
		code.AddRange(BitConverter.GetBytes((long)fallbackTrampoline));
		return code;
	}

	/// <summary>
	/// An sceAgc builder that appends one fixed-shape packet, as guest-side code: with the command
	/// buffer in RDI it reads the cursor, the end and the reserve, declines (jumps to the import's
	/// trampoline, with every argument register intact) unless the buffer is non-null and has the room
	/// <c>AgcExports.GetRemainingCommandDwords</c> demands, stores the packet's dwords at the cursor,
	/// advances the cursor, and returns the packet's address. The stores are plain guest stores, the
	/// kind the title makes into the same buffer: a page the GPU process watches faults and is
	/// resolved from the fault address, which the managed write hooks exist to emulate. Only RAX, R10,
	/// R11 and the flags change, all caller-saved.
	/// </summary>
	internal static List<byte> EmitGuestAgcPacketStub(SharpEmu.Libs.Agc.AgcPacketStubSpec spec, nint fallbackTrampoline)
	{
		const int CursorUpOffset = 0x10;
		const int CursorDownOffset = 0x18;
		const int ReservedDwordsOffset = 0x30;
		if (spec.SizeDwords is < 1 or > 15 || spec.Words.Length != spec.SizeDwords || spec.ReturnDwordOffset >= spec.SizeDwords)
		{
			throw new ArgumentException("Unsupported AGC packet stub shape.", nameof(spec));
		}

		var code = new List<byte>(256);
		var slowPathFixups = new List<int>();
		void Emit(params byte[] bytes) => code.AddRange(bytes);
		void EmitImm32(uint value) => code.AddRange(BitConverter.GetBytes(value));
		void EmitBranchToSlowPath(byte shortOpcode)
		{
			Emit(0x0F, (byte)(shortOpcode + 0x10), 0, 0, 0, 0);
			slowPathFixups.Add(code.Count - 4);
		}

		// test rdi, rdi / jz slow: a null buffer is the export's to refuse.
		Emit(0x48, 0x85, 0xFF);
		EmitBranchToSlowPath(0x74);
		Emit(0x48, 0x8B, 0x47, CursorUpOffset);                // mov rax, [rdi+0x10]      cursor
		Emit(0x4C, 0x8B, 0x57, CursorDownOffset);              // mov r10, [rdi+0x18]      end
		Emit(0x49, 0x29, 0xC2);                                // sub r10, rax
		EmitBranchToSlowPath(0x72);                            // jb slow: end below the cursor
		Emit(0x49, 0xC1, 0xEA, 0x02);                          // shr r10, 2               available dwords
		Emit(0x41, 0xBB); EmitImm32(0xFFFF_FFFFu);             // mov r11d, 0xFFFFFFFF
		Emit(0x4D, 0x39, 0xDA);                                // cmp r10, r11
		EmitBranchToSlowPath(0x77);                            // ja slow: the export clamps the span to 32 bits
		Emit(0x44, 0x8B, 0x5F, ReservedDwordsOffset);          // mov r11d, [rdi+0x30]     reserve
		Emit(0x49, 0x83, 0xC3, checked((byte)spec.SizeDwords)); // add r11, size
		Emit(0x4D, 0x39, 0xDA);                                // cmp r10, r11
		EmitBranchToSlowPath(0x72);                            // jb slow: size > available - reserve

		// Loads one part of a word into r10, or into r11 when intoR11.
		void EmitPart(SharpEmu.Libs.Agc.AgcPacketStubPart part, bool intoR11)
		{
			var destination = intoR11 ? (byte)3 : (byte)2;   // r11 / r10 low three bits
			var source = (byte)part.Register;
			var rex = (byte)(0x40 | (source >= 8 ? 0x04 : 0) | 0x01);     // REX.R for r8, REX.B for r10/r11
			var modrm = (byte)(0xC0 | ((source & 7) << 3) | destination);
			if (part.ShiftRight == 0)
			{
				Emit(rex, 0x89, modrm);                                   // mov r10d/r11d, reg32
			}
			else
			{
				Emit((byte)(rex | 0x08), 0x89, modrm);                    // mov r10/r11, reg64
				Emit(0x49, 0xC1, (byte)(0xE8 | destination), checked((byte)part.ShiftRight)); // shr r10/r11, n
			}

			Emit(0x41, 0x81, (byte)(0xE0 | destination)); EmitImm32(part.Mask);                // and r10d/r11d, mask
			if (part.ShiftLeft != 0)
			{
				Emit(0x41, 0xC1, (byte)(0xE0 | destination), checked((byte)part.ShiftLeft));   // shl r10d/r11d, n
			}
		}

		for (var index = 0; index < spec.Words.Length; index++)
		{
			var word = spec.Words[index];
			var displacement = checked((byte)(index * sizeof(uint)));
			if (word.Kind == SharpEmu.Libs.Agc.AgcPacketStubWordKind.Parts && word.Parts.Length == 0)
			{
				Emit(0xC7, 0x40, displacement); EmitImm32(word.Constant);                       // mov dword [rax+disp], imm32
				continue;
			}

			switch (word.Kind)
			{
				case SharpEmu.Libs.Agc.AgcPacketStubWordKind.Parts:
					EmitPart(word.Parts[0], intoR11: false);
					for (var part = 1; part < word.Parts.Length; part++)
					{
						EmitPart(word.Parts[part], intoR11: true);
						Emit(0x45, 0x09, 0xDA);                                                 // or r10d, r11d
					}

					if (word.Constant != 0)
					{
						Emit(0x41, 0x81, 0xCA); EmitImm32(word.Constant);                       // or r10d, imm32
					}

					break;

				case SharpEmu.Libs.Agc.AgcPacketStubWordKind.AtLeastOne:
					EmitPart(word.Parts[0], intoR11: false);
					Emit(0x45, 0x85, 0xD2);                                                     // test r10d, r10d
					Emit(0x75, 0x06);                                                           // jnz +6
					Emit(0x41, 0xBA); EmitImm32(1);                                             // mov r10d, 1
					break;

				case SharpEmu.Libs.Agc.AgcPacketStubWordKind.DrawInitiator:
				{
					// bit 32 of the modifier set: 0; else (uint)modifier >> 3 & 0x20.
					var source = (byte)word.Parts[0].Register;
					var rex = (byte)(0x40 | (source >= 8 ? 0x04 : 0) | 0x01);
					Emit(rex, 0x89, (byte)(0xC0 | ((source & 7) << 3) | 2));                   // mov r10d, reg32
					Emit(0x41, 0xC1, 0xEA, 0x03);                                               // shr r10d, 3
					Emit(0x41, 0x83, 0xE2, 0x20);                                               // and r10d, 0x20
					Emit((byte)(rex | 0x08), 0x89, (byte)(0xC0 | ((source & 7) << 3) | 3));    // mov r11, reg64
					Emit(0x49, 0xC1, 0xEB, 0x20);                                               // shr r11, 32
					Emit(0x41, 0xF7, 0xC3); EmitImm32(1);                                       // test r11d, 1
					Emit(0x74, 0x03);                                                           // jz +3
					Emit(0x45, 0x31, 0xD2);                                                     // xor r10d, r10d
					break;
				}

				default:
					throw new ArgumentOutOfRangeException(nameof(spec), word.Kind, "Unknown packet word kind.");
			}

			Emit(0x44, 0x89, 0x50, displacement);                                               // mov [rax+disp], r10d
		}

		Emit(0x4C, 0x8D, 0x58, checked((byte)(spec.SizeDwords * sizeof(uint))));                // lea r11, [rax + size*4]
		Emit(0x4C, 0x89, 0x5F, CursorUpOffset);                                                 // mov [rdi+0x10], r11
		if (spec.ReturnDwordOffset != 0)
		{
			Emit(0x48, 0x83, 0xC0, checked((byte)(spec.ReturnDwordOffset * sizeof(uint))));     // add rax, offset*4
		}

		Emit(0xC3);

		foreach (var fixup in slowPathFixups)
		{
			var delta = BitConverter.GetBytes(code.Count - (fixup + 4));
			for (var i = 0; i < 4; i++)
			{
				code[fixup + i] = delta[i];
			}
		}

		// jmp qword [rip+0] with the absolute trampoline address after it.
		Emit(0xFF, 0x25, 0x00, 0x00, 0x00, 0x00);
		code.AddRange(BitConverter.GetBytes((long)fallbackTrampoline));
		return code;
	}

	/// <summary>
	/// The two variable-length register-set builders as guest-side code, on the same terms as
	/// <see cref="EmitGuestAgcPacketStub"/> (plain stores, decline to the trampoline with the arguments
	/// intact; only RAX, R8-R11 and the flags change, all caller-saved). Both decline for a null
	/// pointer, a count outside the stub's range, a buffer without room for the packet plus its
	/// reserve, and a source that overlaps the packet being written; the export handles the rest.
	/// <list type="bullet">
	/// <item>ShRegisterRange(cb, offset, values, count): offset 1..0x3FF, count 1..64; writes the marker
	/// NOP, the SET_SH_REG header, the offset and the values, and returns the SET_SH_REG packet's address.</item>
	/// <item>ShRegisters(cb, pairs, count): count 1..32 and the pairs' offsets consecutive ascending (the
	/// export's sort is then a no-op and it emits exactly one packet); verified before anything is
	/// written, so a decline leaves the buffer untouched.</item>
	/// </list>
	/// </summary>
	internal static List<byte> EmitGuestAgcVariablePacketStub(SharpEmu.Libs.Agc.AgcVariablePacketStub kind, nint fallbackTrampoline)
	{
		const int CursorUpOffset = 0x10;
		const int CursorDownOffset = 0x18;
		const int ReservedDwordsOffset = 0x30;
		var range = kind == SharpEmu.Libs.Agc.AgcVariablePacketStub.ShRegisterRange;
		var code = new List<byte>(320);
		var slowPathFixups = new List<int>();
		void Emit(params byte[] bytes) => code.AddRange(bytes);
		void EmitImm32(uint value) => code.AddRange(BitConverter.GetBytes(value));
		void EmitBranchToSlowPath(byte shortOpcode)
		{
			Emit(0x0F, (byte)(shortOpcode + 0x10), 0, 0, 0, 0);
			slowPathFixups.Add(code.Count - 4);
		}

		// A backwards short jump (opcode jb 0x72) to a code offset.
		void EmitLoopBack(int target)
		{
			var delta = target - (code.Count + 2);
			if (delta < -128)
			{
				throw new InvalidOperationException("Stub loop too long.");
			}

			Emit(0x72, unchecked((byte)(sbyte)delta));
		}

		// Arguments: rdi = buffer, then (range) rsi = offset, rdx = values, rcx = count
		//                       or (registers) rsi = pairs, rdx = count.
		Emit(0x48, 0x85, 0xFF);                                       // test rdi, rdi
		EmitBranchToSlowPath(0x74);                                   // jz slow
		if (range)
		{
			Emit(0x8D, 0x46, 0xFF);                                   // lea eax, [rsi-1]
			Emit(0x3D); EmitImm32(0x3FF);                             // cmp eax, 0x3FF
			EmitBranchToSlowPath(0x73);                               // jae slow: offset outside 1..0x3FF
			Emit(0x8D, 0x41, 0xFF);                                   // lea eax, [rcx-1]
			Emit(0x83, 0xF8, (byte)(SharpEmu.Libs.Agc.AgcExports.RangeStubMaxValues - 1)); // cmp eax, max-1
			EmitBranchToSlowPath(0x77);                               // ja slow: count 0 or too large
			Emit(0x48, 0x85, 0xD2);                                   // test rdx, rdx
			EmitBranchToSlowPath(0x74);                               // jz slow: a null source reserves the payload for the caller
		}
		else
		{
			Emit(0x48, 0x85, 0xF6);                                   // test rsi, rsi
			EmitBranchToSlowPath(0x74);                               // jz slow
			Emit(0x8D, 0x42, 0xFF);                                   // lea eax, [rdx-1]
			Emit(0x83, 0xF8, (byte)(SharpEmu.Libs.Agc.AgcExports.RegistersStubMaxRegisters - 1)); // cmp eax, max-1
			EmitBranchToSlowPath(0x77);                               // ja slow
		}

		Emit(0x48, 0x8B, 0x47, CursorUpOffset);                       // mov rax, [rdi+0x10]      cursor
		Emit(0x4C, 0x8B, 0x57, CursorDownOffset);                     // mov r10, [rdi+0x18]      end
		Emit(0x49, 0x29, 0xC2);                                       // sub r10, rax
		EmitBranchToSlowPath(0x72);                                   // jb slow
		Emit(0x49, 0xC1, 0xEA, 0x02);                                 // shr r10, 2               available dwords
		Emit(0x41, 0xBB); EmitImm32(0xFFFF_FFFFu);                    // mov r11d, 0xFFFFFFFF
		Emit(0x4D, 0x39, 0xDA);                                       // cmp r10, r11
		EmitBranchToSlowPath(0x77);                                   // ja slow
		Emit(0x44, 0x8B, 0x5F, ReservedDwordsOffset);                 // mov r11d, [rdi+0x30]     reserve
		if (range)
		{
			Emit(0x44, 0x8D, 0x49, 0x04);                             // lea r9d, [rcx+4]         dwords: marker 2 + header 2 + values
		}
		else
		{
			Emit(0x44, 0x8D, 0x4A, 0x02);                             // lea r9d, [rdx+2]         dwords: header 2 + values
		}

		Emit(0x4D, 0x01, 0xCB);                                       // add r11, r9
		Emit(0x4D, 0x39, 0xDA);                                       // cmp r10, r11
		EmitBranchToSlowPath(0x72);                                   // jb slow: no room

		// The source must not overlap the packet area [rax, rax + r9*4).
		Emit(0x4E, 0x8D, 0x04, 0x88);                                 // lea r8, [rax + r9*4]     end of the packet
		if (range)
		{
			Emit(0x4C, 0x39, 0xC2);                                   // cmp rdx, r8
			Emit(0x73, 0x0D);                                         // jae ok                   source starts at or after the packet's end
			Emit(0x4C, 0x8D, 0x04, 0x8A);                             // lea r8, [rdx + rcx*4]    end of the source
		}
		else
		{
			Emit(0x4C, 0x39, 0xC6);                                   // cmp rsi, r8
			Emit(0x73, 0x0D);                                         // jae ok
			Emit(0x4C, 0x8D, 0x04, 0xD6);                             // lea r8, [rsi + rdx*8]    end of the pairs
		}

		Emit(0x49, 0x39, 0xC0);                                       // cmp r8, rax
		EmitBranchToSlowPath(0x77);                                   // ja slow: the source ends after the packet starts
		// ok:
		if (!range)
		{
			Emit(0x44, 0x8B, 0x06);                                   // mov r8d, [rsi]           first offset
			Emit(0x45, 0x89, 0xC2);                                   // mov r10d, r8d
			Emit(0x41, 0x01, 0xD2);                                   // add r10d, edx
			EmitBranchToSlowPath(0x72);                               // jc slow: the offsets would wrap
			Emit(0x45, 0x31, 0xC9);                                   // xor r9d, r9d
			var verify = code.Count;
			Emit(0x46, 0x8B, 0x14, 0xCE);                             // mov r10d, [rsi + r9*8]
			Emit(0x45, 0x89, 0xC3);                                   // mov r11d, r8d
			Emit(0x45, 0x01, 0xCB);                                   // add r11d, r9d
			Emit(0x45, 0x39, 0xDA);                                   // cmp r10d, r11d
			EmitBranchToSlowPath(0x75);                               // jne slow: not consecutive ascending
			Emit(0x41, 0xFF, 0xC1);                                   // inc r9d
			Emit(0x41, 0x39, 0xD1);                                   // cmp r9d, edx
			EmitLoopBack(verify);                                     // jb verify
		}

		if (range)
		{
			Emit(0xC7, 0x00); EmitImm32(SharpEmu.Libs.Agc.AgcExports.RangePacketNopHeader);       // mov dword [rax], nop header
			Emit(0xC7, 0x40, 0x04); EmitImm32(SharpEmu.Libs.Agc.AgcExports.RangePacketMarker);    // mov dword [rax+4], marker
			Emit(0x41, 0x89, 0xCA);                                   // mov r10d, ecx
			Emit(0x41, 0xC1, 0xE2, 0x10);                             // shl r10d, 16
			Emit(0x41, 0x81, 0xCA); EmitImm32(SharpEmu.Libs.Agc.AgcExports.SetShRegHeaderWithoutCount); // or r10d, header
			Emit(0x44, 0x89, 0x50, 0x08);                             // mov [rax+8], r10d
			Emit(0x89, 0x70, 0x0C);                                   // mov [rax+12], esi        the register offset
			Emit(0x45, 0x31, 0xC0);                                   // xor r8d, r8d
			var copy = code.Count;
			Emit(0x46, 0x8B, 0x14, 0x82);                             // mov r10d, [rdx + r8*4]
			Emit(0x46, 0x89, 0x54, 0x80, 0x10);                       // mov [rax + r8*4 + 16], r10d
			Emit(0x41, 0xFF, 0xC0);                                   // inc r8d
			Emit(0x41, 0x39, 0xC8);                                   // cmp r8d, ecx
			EmitLoopBack(copy);                                       // jb copy
			Emit(0x44, 0x8D, 0x49, 0x04);                             // lea r9d, [rcx+4]
		}
		else
		{
			Emit(0x41, 0x89, 0xD2);                                   // mov r10d, edx
			Emit(0x41, 0xC1, 0xE2, 0x10);                             // shl r10d, 16
			Emit(0x41, 0x81, 0xCA); EmitImm32(SharpEmu.Libs.Agc.AgcExports.SetShRegHeaderWithoutCount); // or r10d, header
			Emit(0x44, 0x89, 0x10);                                   // mov [rax], r10d
			Emit(0x45, 0x0F, 0xB7, 0xD0);                             // movzx r10d, r8w          first offset & 0xFFFF
			Emit(0x44, 0x89, 0x50, 0x04);                             // mov [rax+4], r10d
			Emit(0x45, 0x31, 0xC9);                                   // xor r9d, r9d
			var copy = code.Count;
			Emit(0x46, 0x8B, 0x54, 0xCE, 0x04);                       // mov r10d, [rsi + r9*8 + 4]
			Emit(0x46, 0x89, 0x54, 0x88, 0x08);                       // mov [rax + r9*4 + 8], r10d
			Emit(0x41, 0xFF, 0xC1);                                   // inc r9d
			Emit(0x41, 0x39, 0xD1);                                   // cmp r9d, edx
			EmitLoopBack(copy);                                       // jb copy
			Emit(0x44, 0x8D, 0x4A, 0x02);                             // lea r9d, [rdx+2]
		}

		Emit(0x4E, 0x8D, 0x1C, 0x88);                                 // lea r11, [rax + r9*4]    the new cursor
		Emit(0x4C, 0x89, 0x5F, CursorUpOffset);                       // mov [rdi+0x10], r11
		if (range)
		{
			Emit(0x48, 0x83, 0xC0, 0x08);                             // add rax, 8               the SET_SH_REG packet, after the marker
		}

		Emit(0xC3);

		foreach (var fixup in slowPathFixups)
		{
			var delta = BitConverter.GetBytes(code.Count - (fixup + 4));
			for (var i = 0; i < 4; i++)
			{
				code[fixup + i] = delta[i];
			}
		}

		Emit(0xFF, 0x25, 0x00, 0x00, 0x00, 0x00);
		code.AddRange(BitConverter.GetBytes((long)fallbackTrampoline));
		return code;
	}

	/// <summary>
	/// A register-indirect patcher (sceAgcSet{Cx,Sh,Uc}RegIndirectPatch{SetAddress,SetNumRegisters,AddRegisters})
	/// as guest-side code, on the same terms as <see cref="EmitGuestAgcPacketStub"/>: with the packet in
	/// RDI and the argument in RSI it patches a SET_*_REG_INDIRECT packet of its space in place and returns
	/// 0, and declines (jumps to the import's trampoline with every argument register intact, nothing
	/// written) for every case in which the export would not just do that:
	/// <list type="bullet">
	/// <item>a packet pointer below 0x10000 (null included) or at or above 2^47;</item>
	/// <item>a packet whose 20 bytes cross a 4 KiB page, so the export's partial-write and
	/// unreadable-tail answers stay its own: the stub reads and writes only the page it read the
	/// header from;</item>
	/// <item>a header that, masked to its length and opcode, is not the space's 5-dword packet (another
	/// space, another length, the legacy NOP layout, anything else);</item>
	/// <item>SetAddress with a null address.</item>
	/// </list>
	/// A packet pointer in range whose page is not mapped faults here where the export would answer
	/// MEMORY_FAULT, as the builder stubs fault on such a buffer pointer. Only RAX, R10, R11 and the flags
	/// change, all caller-saved.
	/// </summary>
	internal static List<byte> EmitGuestAgcRegIndirectPatchStub(SharpEmu.Libs.Agc.AgcRegIndirectPatchStubSpec spec, nint fallbackTrampoline)
	{
		const byte AddressOffset = 4;
		const byte CountOffset = 16;
		const uint PacketBytes = 20;
		var code = new List<byte>(128);
		var slowPathFixups = new List<int>();
		void Emit(params byte[] bytes) => code.AddRange(bytes);
		void EmitImm32(uint value) => code.AddRange(BitConverter.GetBytes(value));
		void EmitBranchToSlowPath(byte shortOpcode)
		{
			Emit(0x0F, (byte)(shortOpcode + 0x10), 0, 0, 0, 0);
			slowPathFixups.Add(code.Count - 4);
		}

		Emit(0x48, 0x81, 0xFF); EmitImm32(0x10000);                   // cmp rdi, 0x10000
		EmitBranchToSlowPath(0x72);                                     // jb slow: null or a low address
		Emit(0x48, 0x89, 0xF8);                                         // mov rax, rdi
		Emit(0x48, 0xC1, 0xE8, 0x2F);                                   // shr rax, 47
		EmitBranchToSlowPath(0x75);                                     // jnz slow: not a user address
		Emit(0x89, 0xF8);                                               // mov eax, edi
		Emit(0x25); EmitImm32(0xFFF);                                   // and eax, 0xFFF
		Emit(0x3D); EmitImm32(0x1000 - PacketBytes);                    // cmp eax, 0x1000 - 20
		EmitBranchToSlowPath(0x77);                                     // ja slow: the packet crosses a page
		if (spec.Kind == SharpEmu.Libs.Agc.AgcRegIndirectPatchStub.SetAddress)
		{
			Emit(0x48, 0x85, 0xF6);                                     // test rsi, rsi
			EmitBranchToSlowPath(0x74);                                 // jz slow: a null address is the export's to refuse
		}

		Emit(0x8B, 0x07);                                               // mov eax, [rdi]           header
		Emit(0x25); EmitImm32(SharpEmu.Libs.Agc.AgcExports.RegIndirectPatchHeaderMask); // and eax, length|opcode
		Emit(0x3D); EmitImm32(spec.Header & SharpEmu.Libs.Agc.AgcExports.RegIndirectPatchHeaderMask); // cmp eax, the space's
		EmitBranchToSlowPath(0x75);                                     // jne slow: not this space's 5-dword packet

		switch (spec.Kind)
		{
			case SharpEmu.Libs.Agc.AgcRegIndirectPatchStub.SetAddress:
				// [+4] = ([+4] & 3) | (address & ~3), then [+8] = address >> 32.
				Emit(0x44, 0x8B, 0x57, AddressOffset);                  // mov r10d, [rdi+4]
				Emit(0x41, 0x83, 0xE2, 0x03);                           // and r10d, 3
				Emit(0x41, 0x89, 0xF3);                                 // mov r11d, esi
				Emit(0x41, 0x83, 0xE3, 0xFC);                           // and r11d, 0xFFFFFFFC
				Emit(0x45, 0x09, 0xDA);                                 // or r10d, r11d
				Emit(0x44, 0x89, 0x57, AddressOffset);                  // mov [rdi+4], r10d
				Emit(0x49, 0x89, 0xF3);                                 // mov r11, rsi
				Emit(0x49, 0xC1, 0xEB, 0x20);                           // shr r11, 32
				Emit(0x44, 0x89, 0x5F, AddressOffset + 4);              // mov [rdi+8], r11d
				break;

			case SharpEmu.Libs.Agc.AgcRegIndirectPatchStub.SetNumRegisters:
				// [+16] = ([+16] & ~mask) | (count & mask).
				Emit(0x44, 0x8B, 0x57, CountOffset);                    // mov r10d, [rdi+16]
				Emit(0x41, 0x81, 0xE2); EmitImm32(~SharpEmu.Libs.Agc.AgcExports.RegIndirectPatchCountMask); // and r10d, ~mask
				Emit(0x41, 0x89, 0xF3);                                 // mov r11d, esi
				Emit(0x41, 0x81, 0xE3); EmitImm32(SharpEmu.Libs.Agc.AgcExports.RegIndirectPatchCountMask);  // and r11d, mask
				Emit(0x45, 0x09, 0xDA);                                 // or r10d, r11d
				Emit(0x44, 0x89, 0x57, CountOffset);                    // mov [rdi+16], r10d
				break;

			case SharpEmu.Libs.Agc.AgcRegIndirectPatchStub.AddRegisters:
				// [+16] = ([+16] & ~mask) | (([+16] + count) & mask): the sum wraps inside the field.
				Emit(0x44, 0x8B, 0x57, CountOffset);                    // mov r10d, [rdi+16]
				Emit(0x45, 0x8D, 0x1C, 0x32);                           // lea r11d, [r10 + rsi]
				Emit(0x41, 0x81, 0xE3); EmitImm32(SharpEmu.Libs.Agc.AgcExports.RegIndirectPatchCountMask);  // and r11d, mask
				Emit(0x41, 0x81, 0xE2); EmitImm32(~SharpEmu.Libs.Agc.AgcExports.RegIndirectPatchCountMask); // and r10d, ~mask
				Emit(0x45, 0x09, 0xDA);                                 // or r10d, r11d
				Emit(0x44, 0x89, 0x57, CountOffset);                    // mov [rdi+16], r10d
				break;

			default:
				throw new ArgumentOutOfRangeException(nameof(spec), spec.Kind, "Unknown register-indirect patch stub.");
		}

		Emit(0x31, 0xC0);                                               // xor eax, eax
		Emit(0xC3);                                                     // ret

		foreach (var fixup in slowPathFixups)
		{
			var delta = BitConverter.GetBytes(code.Count - (fixup + 4));
			for (var i = 0; i < 4; i++)
			{
				code[fixup + i] = delta[i];
			}
		}

		// jmp qword [rip+0] with the absolute trampoline address after it.
		Emit(0xFF, 0x25, 0x00, 0x00, 0x00, 0x00);
		code.AddRange(BitConverter.GetBytes((long)fallbackTrampoline));
		return code;
	}

	/// <summary>
	/// memcpy/memmove(dst, src, n) as a forward 32-byte SSE loop (16-byte step, then
	/// <c>rep movsb</c> for the last 0-15 bytes), returning dst. Rosetta translates
	/// <c>rep movsb</c> to a byte-granular loop, which measured 10-40x slower than
	/// movdqu pairs for 256 B - 1 MiB. The libc export copies with memmove semantics, so a
	/// destination that starts inside the source (where a forward copy would read bytes it
	/// already overwrote) takes the managed path. Only RAX, RCX, RSI, RDI, XMM0, XMM1 and the
	/// flags change - all caller-saved - and the direction flag is clear on entry per the
	/// SysV ABI.
	/// </summary>
	internal static List<byte> EmitGuestMemoryCopyStub(nint fallbackTrampoline)
	{
		var code = new List<byte>(104);
		// mov rcx,rdi / sub rcx,rsi / cmp rcx,rdx / jb slow: dst - src < n (unsigned) is
		// exactly a destination in (src, src + n) or equal to src with a non-empty copy.
		code.AddRange([0x48, 0x89, 0xF9, 0x48, 0x29, 0xF1, 0x48, 0x39, 0xD1, 0x72, 0x00]);
		var slowPathFixup = code.Count - 1;
		code.AddRange([
			0x48, 0x89, 0xF8,                   // mov rax, rdi
			0x48, 0x89, 0xD1,                   // mov rcx, rdx
			0x48, 0x83, 0xF9, 0x20,             // cmp rcx, 32
			0x72, 0x24,                         // jb l16
			0xF3, 0x0F, 0x6F, 0x06,             // l32: movdqu xmm0, [rsi]
			0xF3, 0x0F, 0x6F, 0x4E, 0x10,       // movdqu xmm1, [rsi+16]
			0xF3, 0x0F, 0x7F, 0x07,             // movdqu [rdi], xmm0
			0xF3, 0x0F, 0x7F, 0x4F, 0x10,       // movdqu [rdi+16], xmm1
			0x48, 0x83, 0xC6, 0x20,             // add rsi, 32
			0x48, 0x83, 0xC7, 0x20,             // add rdi, 32
			0x48, 0x83, 0xE9, 0x20,             // sub rcx, 32
			0x48, 0x83, 0xF9, 0x20,             // cmp rcx, 32
			0x73, 0xDC,                         // jae l32
			0x48, 0x83, 0xF9, 0x10,             // l16: cmp rcx, 16
			0x72, 0x14,                         // jb tail
			0xF3, 0x0F, 0x6F, 0x06,             // movdqu xmm0, [rsi]
			0xF3, 0x0F, 0x7F, 0x07,             // movdqu [rdi], xmm0
			0x48, 0x83, 0xC6, 0x10,             // add rsi, 16
			0x48, 0x83, 0xC7, 0x10,             // add rdi, 16
			0x48, 0x83, 0xE9, 0x10,             // sub rcx, 16
			0xF3, 0xA4,                         // tail: rep movsb
			0xC3,                               // ret
		]);
		code[slowPathFixup] = checked((byte)(code.Count - (slowPathFixup + 1)));
		// jmp qword [rip+0] with the absolute trampoline address after it.
		code.AddRange([0xFF, 0x25, 0x00, 0x00, 0x00, 0x00]);
		code.AddRange(BitConverter.GetBytes((long)fallbackTrampoline));
		return code;
	}

	/// <summary>
	/// memset(dst, value, n) as a 16-byte SSE store loop (then <c>rep stosb</c> for the last
	/// 0-15 bytes), returning dst. The managed export tolerates a null/low or non-canonical
	/// destination and clamps oversized clears that run off the mapped range, so those cases
	/// (dst below 0x10000, dst at or above 2^47, n above 1 MiB) take the managed path; so does
	/// everything while the write watch is armed (see <c>TryCreateGuestFastPathStub</c>).
	/// Changes only RAX, RCX, R8, XMM0 and the flags, all caller-saved.
	/// </summary>
	internal static List<byte> EmitGuestMemorySetStub(nint fallbackTrampoline)
	{
		var code = new List<byte>(112);
		var slowPathFixups = new List<int>();
		code.AddRange([0x48, 0x81, 0xFF, 0x00, 0x00, 0x01, 0x00]);                         // cmp rdi, 0x10000
		code.AddRange([0x72, 0x00]); slowPathFixups.Add(code.Count - 1);                   // jb slow
		code.AddRange([0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 0x00, 0x00]);       // mov rax, 0x7FFFFFFFFFFF
		code.AddRange([0x48, 0x39, 0xC7]);                                                 // cmp rdi, rax
		code.AddRange([0x77, 0x00]); slowPathFixups.Add(code.Count - 1);                   // ja slow
		code.AddRange([0x48, 0x81, 0xFA, 0x00, 0x00, 0x10, 0x00]);                         // cmp rdx, 0x100000
		code.AddRange([0x77, 0x00]); slowPathFixups.Add(code.Count - 1);                   // ja slow
		code.AddRange([
			0x40, 0x0F, 0xB6, 0xC6,                                                         // movzx eax, sil
			0x48, 0xB9, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,                     // mov rcx, 0x0101010101010101
			0x48, 0x0F, 0xAF, 0xC1,                                                         // imul rax, rcx
			0x66, 0x48, 0x0F, 0x6E, 0xC0,                                                   // movq xmm0, rax
			0x66, 0x0F, 0x6C, 0xC0,                                                         // punpcklqdq xmm0, xmm0
			0x49, 0x89, 0xF8,                                                               // mov r8, rdi
			0x48, 0x89, 0xD1,                                                               // mov rcx, rdx
			0x48, 0x83, 0xF9, 0x10,                                                         // cmp rcx, 16
			0x72, 0x12,                                                                     // jb tail
			0xF3, 0x0F, 0x7F, 0x07,                                                         // l16: movdqu [rdi], xmm0
			0x48, 0x83, 0xC7, 0x10,                                                         // add rdi, 16
			0x48, 0x83, 0xE9, 0x10,                                                         // sub rcx, 16
			0x48, 0x83, 0xF9, 0x10,                                                         // cmp rcx, 16
			0x73, 0xEE,                                                                     // jae l16
			0xF3, 0xAA,                                                                     // tail: rep stosb (AL = value)
			0x4C, 0x89, 0xC0,                                                               // mov rax, r8
			0xC3,                                                                           // ret
		]);
		foreach (var fixup in slowPathFixups)
		{
			code[fixup] = checked((byte)(code.Count - (fixup + 1)));
		}

		code.AddRange([0xFF, 0x25, 0x00, 0x00, 0x00, 0x00]);
		code.AddRange(BitConverter.GetBytes((long)fallbackTrampoline));
		return code;
	}
}
