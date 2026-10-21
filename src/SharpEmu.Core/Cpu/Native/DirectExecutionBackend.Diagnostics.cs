// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using SharpEmu.Core.Cpu.Disasm;
using SharpEmu.HLE;
using SharpEmu.Logging;

namespace SharpEmu.Core.Cpu.Native;

public sealed partial class DirectExecutionBackend
{
	private static readonly ConcurrentDictionary<ulong, byte> _knownExecutablePages = new();

	private static readonly bool _perfHleHistogram =
		string.Equals(System.Environment.GetEnvironmentVariable("SHARPEMU_PERF_HLE"), "1", System.StringComparison.Ordinal);
	private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> _perfHleCounts = new();
	private static long _perfHleTotal;
	private static long _perfHleDispatchTicks;
	private static long _perfHleExclusiveTicks;
	private static long _perfHleCompleted;

	private sealed class PerfHleExportCost
	{
		public long Calls;
		public long Ticks;
		public long ExclusiveTicks;
	}

	private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, PerfHleExportCost> _perfHleCosts = new();

	private static long _perfHleFirstTimestamp;

	// Per-thread export report (SHARPEMU_PERF_HLE_THREADS=0 drops it). The thread CPU
	// clock costs microseconds under Rosetta, so it is read on one call in
	// SHARPEMU_PERF_HLE_CPU_EVERY (default 64; 0 never).
	private static readonly bool _perfHleThreads = _perfHleHistogram &&
		!string.Equals(System.Environment.GetEnvironmentVariable("SHARPEMU_PERF_HLE_THREADS"), "0", System.StringComparison.Ordinal);
	private static readonly int _perfHleCpuEvery = !_perfHleThreads
		? 0
		: int.TryParse(System.Environment.GetEnvironmentVariable("SHARPEMU_PERF_HLE_CPU_EVERY"), out var perfHleCpuEvery) && perfHleCpuEvery >= 0
			? perfHleCpuEvery
			: 64;
	private static readonly int _perfHleThreadReportSeconds =
		int.TryParse(System.Environment.GetEnvironmentVariable("SHARPEMU_PERF_HLE_THREAD_REPORT_S"), out var perfHleReport) && perfHleReport > 0
			? perfHleReport
			: 15;
	private static long _perfHleNextThreadReport;
	private static int _perfHleThreadReporting;

	private static void RecordPerfHleDispatchTime(HleDispatchTiming.Elapsed elapsed)
	{
		if (!elapsed.Recorded) return; // omitted nesting must not enter the measured denominator
		var total = System.Threading.Interlocked.Add(ref _perfHleDispatchTicks, elapsed.InclusiveTicks);
		var exclusive = System.Threading.Interlocked.Add(ref _perfHleExclusiveTicks, elapsed.ExclusiveTicks);
		var completed = System.Threading.Interlocked.Increment(ref _perfHleCompleted);

		var name = elapsed.Name;
		if (name is not null)
		{
			var cost = _perfHleCosts.GetOrAdd(name, static _ => new PerfHleExportCost());
			System.Threading.Interlocked.Increment(ref cost.Calls);
			System.Threading.Interlocked.Add(ref cost.Ticks, elapsed.InclusiveTicks);
			System.Threading.Interlocked.Add(ref cost.ExclusiveTicks, elapsed.ExclusiveTicks);
		}

		if (completed % 500000 == 0)
		{
			var frequency = (double)System.Diagnostics.Stopwatch.Frequency;
			var avgUs = (double)total / frequency * 1_000_000.0 / completed;
			var first = System.Threading.Interlocked.CompareExchange(ref _perfHleFirstTimestamp, 0, 0);
			var wallSeconds = first == 0
				? 0
				: (double)(System.Diagnostics.Stopwatch.GetTimestamp() - first) / frequency;
			System.Console.Error.WriteLine(
				$"[PERF][HLE] inclusive_dispatch_avg={avgUs:F3}us " +
				$"inclusive_wall_s={(double)total / frequency:F2} exclusive_wall_s={(double)exclusive / frequency:F2} " +
				$"wall_s={wallSeconds:F2} recorded_completed={completed} window=process_cumulative " +
				$"units=elapsed_wall_not_cpu omitted_nested_scopes={HleDispatchTiming.OmittedScopes}");

			var snapshot = new System.Collections.Generic.List<System.Collections.Generic.KeyValuePair<string, PerfHleExportCost>>(_perfHleCosts.Count + 16);
			foreach (var kvp in _perfHleCosts)
			{
				snapshot.Add(kvp);
			}

			var top = snapshot
				.OrderByDescending(kvp => System.Threading.Interlocked.Read(ref kvp.Value.Ticks))
				.Take(12)
				.Select(kvp =>
				{
					var seconds = System.Threading.Interlocked.Read(ref kvp.Value.Ticks) / frequency;
					var callCount = System.Threading.Interlocked.Read(ref kvp.Value.Calls);
					var exclusiveSeconds = System.Threading.Interlocked.Read(ref kvp.Value.ExclusiveTicks) / frequency;
					var perCallUs = callCount > 0 ? seconds * 1_000_000.0 / callCount : 0;
					return $"{kvp.Key}: inclusive_wall_s={seconds:F3} exclusive_wall_s={exclusiveSeconds:F3} n={callCount} inclusive_us_per_call={perCallUs:F2}";
				});
			System.Console.Error.WriteLine($"[PERF][HLE] cost: {string.Join(" | ", top)}");

			if (SharpEmu.HLE.GuestFastPath.Enabled)
			{
				var fastPath = SharpEmu.HLE.GuestFastPath.SnapshotCounters();
				System.Console.Error.WriteLine(
					$"[PERF][HLE] fast_path: self_hits={fastPath.SelfHits} " +
					$"getspecific_hits={fastPath.GetspecificHits} blocks={fastPath.Blocks}");
			}
		}
	}

	private static void RecordPerfHleThreadCost(HleDispatchTiming.Elapsed elapsed, HleDispatchTiming.ThreadState thread, long cpuNs)
	{
		if (!_perfHleThreads || !elapsed.Recorded || elapsed.Name is null) return;
		// Owner-thread writes only: the dictionary is per thread, so no interlocked traffic.
		if (!thread.Costs.TryGetValue(elapsed.Name!, out var cost))
		{
			cost = thread.Costs.GetOrAdd(elapsed.Name!, static _ => new HleDispatchTiming.ExportCost());
		}
		cost.Calls++;
		cost.ExclusiveTicks += elapsed.ExclusiveTicks;
		if (thread.Depth == 0)
		{
			thread.LastTopLevel = cost;
		}
		if (cpuNs >= 0)
		{
			cost.CpuSamples++;
			cost.SampledCpuNs += cpuNs;
			cost.SampledWallTicks += elapsed.InclusiveTicks;
		}

		var now = System.Diagnostics.Stopwatch.GetTimestamp();
		var next = System.Threading.Interlocked.Read(ref _perfHleNextThreadReport);
		if (next == 0)
		{
			System.Threading.Interlocked.CompareExchange(
				ref _perfHleNextThreadReport,
				now + _perfHleThreadReportSeconds * System.Diagnostics.Stopwatch.Frequency,
				0);
			return;
		}
		if (now < next || System.Threading.Interlocked.Exchange(ref _perfHleThreadReporting, 1) != 0)
		{
			return;
		}
		try
		{
			System.Threading.Interlocked.Exchange(
				ref _perfHleNextThreadReport,
				now + _perfHleThreadReportSeconds * System.Diagnostics.Stopwatch.Frequency);
			ReportPerfHleThreads();
		}
		finally
		{
			System.Threading.Interlocked.Exchange(ref _perfHleThreadReporting, 0);
		}
	}

	// Per thread, cumulative since its first import: wall time, how much of it was inside
	// top-level imports (the rest is guest code, or parked outside an import), and the
	// exports by exclusive wall. cpu% is sampled CPU over the same calls' inclusive wall:
	// near 100 burns the core, near 0 waits.
	private static void ReportPerfHleThreads()
	{
		var frequency = (double)System.Diagnostics.Stopwatch.Frequency;
		var threads = HleDispatchTiming.Threads.ToArray()
			.OrderByDescending(thread => thread.TopLevelTicks)
			.Take(12);
		foreach (var thread in threads)
		{
			var wall = (thread.LastTimestamp - thread.FirstTimestamp) / frequency;
			var cpuNs = thread.CpuSinceFirstNs();
			var hle = thread.TopLevelTicks / frequency;
			var snapshot = new List<KeyValuePair<string, HleDispatchTiming.ExportCost>>(thread.Costs.Count + 8);
			foreach (var pair in thread.Costs)
			{
				snapshot.Add(pair);
			}
			var top = snapshot
				.OrderByDescending(pair => pair.Value.ExclusiveTicks)
				.Take(20)
				.Select(pair =>
				{
					var cost = pair.Value;
					var calls = Math.Max(1, cost.Calls);
					var cpuPercent = cost.SampledWallTicks > 0
						? $"{cost.SampledCpuNs / 1e9 / (cost.SampledWallTicks / frequency) * 100:F0}"
						: "-";
					return $"{pair.Key}: n={cost.Calls} excl_s={cost.ExclusiveTicks / frequency:F2} " +
						$"excl_us={cost.ExclusiveTicks / frequency * 1e6 / calls:F2} cpu%={cpuPercent}";
				});
			System.Console.Error.WriteLine(
				$"[PERF][HLE] thread name={thread.ThreadName} wall_s={wall:F1} cpu_s={(cpuNs >= 0 ? $"{cpuNs / 1e9:F1}" : "-")} hle_s={hle:F2} " +
				$"outside_hle_s={wall - hle:F2} top: {string.Join(" | ", top)}");
			var gaps = snapshot
				.Where(pair => pair.Value.GapAfterTicks > 0)
				.OrderByDescending(pair => pair.Value.GapAfterTicks)
				.Take(8)
				.Select(pair => $"{pair.Key}: gap_s={pair.Value.GapAfterTicks / frequency:F2} long_gaps={pair.Value.LongGapsAfter}");
			System.Console.Error.WriteLine(
				$"[PERF][HLE] thread_gaps name={thread.ThreadName} (outside-HLE time after each import): {string.Join(" | ", gaps)}");
		}
	}

	private static readonly bool _perfHleNoDict =
		string.Equals(System.Environment.GetEnvironmentVariable("SHARPEMU_PERF_HLE_NODICT"), "1", System.StringComparison.Ordinal);

	private static void RecordPerfHleCall(string name)
	{
		HleDispatchTiming.NameCurrent(name);
		var total = System.Threading.Interlocked.Increment(ref _perfHleTotal);
		if (total == 1)
		{
			System.Threading.Interlocked.CompareExchange(
				ref _perfHleFirstTimestamp,
				System.Diagnostics.Stopwatch.GetTimestamp(),
				0);
		}

		if (!_perfHleNoDict)
		{
			_perfHleCounts.AddOrUpdate(name, 1, static (_, v) => v + 1);
		}

		if (total % 500000 == 0 && !_perfHleNoDict)
		{
			// Snapshot via foreach (a safe moving enumerator) before sorting.
			// LINQ over a ConcurrentDictionary uses ICollection.CopyTo, which
			// throws ArgumentException if another thread adds a key between the
			// Count read and the copy — that exception was being swallowed into
			// a CPU_TRAP return and crashing the guest.
			var snapshot = new System.Collections.Generic.List<System.Collections.Generic.KeyValuePair<string, long>>(_perfHleCounts.Count + 16);
			foreach (var kvp in _perfHleCounts)
			{
				snapshot.Add(kvp);
			}

			var top = snapshot
				.OrderByDescending(kvp => kvp.Value)
				.Take(20)
				.Select(kvp => $"{kvp.Key}={kvp.Value}");
			System.Console.Error.WriteLine($"[PERF][HLE] total={total} top: {string.Join(", ", top)}");
		}
	}

	private void RecordRecentImportTrace(
		long dispatchIndex,
		string nid,
		ulong returnRip,
		ulong arg0,
		ulong arg1,
		ulong arg2)
	{
		var trace = _recentImportTrace;
		trace[_recentImportTraceWriteIndex] = new RecentImportTraceEntry(
			dispatchIndex,
			nid,
			returnRip,
			arg0,
			arg1,
			arg2,
			GuestThreadExecution.CurrentGuestThreadHandle,
			Environment.CurrentManagedThreadId);
		_recentImportTraceWriteIndex = (_recentImportTraceWriteIndex + 1) % trace.Length;
		if (_recentImportTraceCount < trace.Length)
		{
			_recentImportTraceCount++;
		}
	}

	private void DumpRecentImportTrace()
	{
		DumpFaultThreadImportTrace();
		var trace = _recentImportTrace;
		if (trace is null || _recentImportTraceCount == 0)
		{
			return;
		}
		Log.Info($"   Legacy backend import ring (guest workers and fast paths excluded; not fault-thread history) dump_context_managed={Environment.CurrentManagedThreadId} dump_context_guest=0x{GuestThreadExecution.CurrentGuestThreadHandle:X16} ({_recentImportTraceCount}):");
		int num = (_recentImportTraceWriteIndex - _recentImportTraceCount + trace.Length) % trace.Length;
		for (int i = 0; i < _recentImportTraceCount; i++)
		{
			int num2 = (num + i) % trace.Length;
			var entry = trace[num2];
			if (!string.IsNullOrEmpty(entry.Nid))
			{
				Log.Info(
					$"     #{entry.DispatchIndex} managed={entry.ManagedThreadId} guest=0x{entry.GuestThreadHandle:X16} nid={entry.Nid} ret=0x{entry.ReturnRip:X16} " +
					$"rdi=0x{entry.Arg0:X16} rsi=0x{entry.Arg1:X16} rdx=0x{entry.Arg2:X16}");
			}
		}
	}

	private unsafe static List<ulong> ScanSuspiciousResolverPointers(ulong scanStart, ulong scanEnd)
	{
		if (scanEnd <= scanStart)
		{
			return new List<ulong>(0);
		}
		int num = 0;
		int num2 = 0;
		List<ulong> list = new List<ulong>(16);
		ulong num3 = scanStart;
		MEMORY_BASIC_INFORMATION64 lpBuffer;
		while (num3 < scanEnd && VirtualQuery((void*)num3, out lpBuffer, (nuint)sizeof(MEMORY_BASIC_INFORMATION64)) != 0)
		{
			ulong baseAddress = lpBuffer.BaseAddress;
			ulong num4 = baseAddress + lpBuffer.RegionSize;
			if (num4 <= num3)
			{
				break;
			}
			ulong value = Math.Max(num3, baseAddress);
			ulong num5 = Math.Min(num4, scanEnd);
			if (lpBuffer.State == 4096 && IsReadableProtection(lpBuffer.Protect) && !IsExecutableProtection(lpBuffer.Protect))
			{
				ulong num6 = AlignUp(value, 8uL);
				for (ulong num7 = num6; num7 + 8 <= num5; num7 += 8)
				{
					ulong value2 = *(ulong*)num7;
					if (IsUnresolvedSentinel(value2))
					{
						num++;
						list.Add(num7);
						if (num2 < 32)
						{
							Log.Info($"Suspicious unresolved pointer: slot=0x{num7:X16} value=0x{value2:X16}");
							num2++;
						}
						if (num >= 16384)
						{
							Log.Warning($"Suspicious unresolved pointer scan reached cap ({16384}); truncating.");
							return list;
						}
					}
				}
			}
			num3 = num5;
		}
		if (num != 0)
		{
			Log.Warning($"Suspicious unresolved pointer hits: {num}");
		}
		return list;
	}

	private void ProbeReturnRip(ulong returnRip, long dispatchIndex)
	{
		var cpuContext = ActiveCpuContext;
		if (cpuContext == null || returnRip == 0)
		{
			return;
		}
		const int preludeSize = 192;
		Span<byte> prelude = stackalloc byte[preludeSize];
		if (returnRip >= preludeSize && cpuContext.Memory.TryRead(returnRip - preludeSize, prelude))
		{
			Console.Error.WriteLine(
				$"[LOADER][TRACE] Import#{dispatchIndex} pre-return bytes @0x{returnRip - preludeSize:X16}: " +
				BitConverter.ToString(prelude.ToArray()).Replace("-", " "));

			List<DecodedInst>? bestCallChain = null;
			var preludeAddress = returnRip - preludeSize;
			for (var startOffset = 0; startOffset < preludeSize; startOffset++)
			{
				var cursor = preludeAddress + (ulong)startOffset;
				var candidate = new List<DecodedInst>();
				while (cursor < returnRip && candidate.Count < 96 &&
					IcedDecoder.TryReadGuestBytes(cpuContext.Memory, cursor, 15, out var instructionBytes) &&
					IcedDecoder.TryDecode(cursor, instructionBytes, out var instruction) &&
					instruction.Length > 0 &&
					cursor + (ulong)instruction.Length <= returnRip)
				{
					candidate.Add(instruction);
					cursor += (ulong)instruction.Length;
				}

				if (cursor == returnRip &&
					candidate.Count > 0 &&
					string.Equals(candidate[^1].Mnemonic, "Call", StringComparison.OrdinalIgnoreCase) &&
					(bestCallChain is null || candidate.Count > bestCallChain.Count))
				{
					bestCallChain = candidate;
				}
			}

			if (bestCallChain is not null)
			{
				Console.Error.WriteLine($"[LOADER][TRACE] Import#{dispatchIndex} pre-return disassembly:");
				foreach (var instruction in bestCallChain.TakeLast(32))
				{
					Console.Error.WriteLine(
						$"[LOADER][TRACE]   0x{instruction.Rip:X16}: {instruction.Text} " +
						$"bytes={IcedDecoder.FormatBytes(instruction.Bytes)}");
				}
			}
		}
		Span<byte> destination = stackalloc byte[128];
		if (!cpuContext.Memory.TryRead(returnRip, destination))
		{
			Log.Debug($"Import#{dispatchIndex} return-rip probe: unreadable @0x{returnRip:X16}");
			return;
		}
		string value = BitConverter.ToString(destination.ToArray()).Replace("-", " ");
		Log.Debug($"Import#{dispatchIndex} return-rip bytes @0x{returnRip:X16}: {value}");
		if (destination[0] == byte.MaxValue && (destination[1] == 21 || destination[1] == 37))
		{
			int num = BitConverter.ToInt32(destination.Slice(2, 4));
			ulong num2 = returnRip + 6 + (ulong)num;
			if (cpuContext.TryReadUInt64(num2, out var value2))
			{
				Log.Debug($"Import#{dispatchIndex} return-rip slot: [0x{num2:X16}] = 0x{value2:X16}");
			}
		}
		if (destination[0] == 72 && destination[1] == 139 && destination[2] == 5)
		{
			int num3 = BitConverter.ToInt32(destination.Slice(3, 4));
			ulong num4 = returnRip + 7 + (ulong)num3;
			if (cpuContext.TryReadUInt64(num4, out var value3))
			{
				Log.Debug($"Import#{dispatchIndex} return-rip mov-slot: [0x{num4:X16}] = 0x{value3:X16}");
			}
		}
		for (int i = 0; i + 6 <= destination.Length; i++)
		{
			if (destination[i] == byte.MaxValue && (destination[i + 1] == 21 || destination[i + 1] == 37))
			{
				int num5 = BitConverter.ToInt32(destination.Slice(i + 2, 4));
				ulong num6 = returnRip + (ulong)i;
				ulong num7 = num6 + 6 + (ulong)num5;
				if (cpuContext.TryReadUInt64(num7, out var value4))
				{
					Log.Debug($"Import#{dispatchIndex} near-indirect @{num6:X16}: slot=0x{num7:X16} val=0x{value4:X16}");
				}
			}
		}
		Span<byte> targetBytes = stackalloc byte[32];
		for (int i = 0; i + 5 <= destination.Length; i++)
		{
			if (destination[i] != 0xE8)
			{
				continue;
			}

			int rel32 = BitConverter.ToInt32(destination.Slice(i + 1, 4));
			ulong callRip = returnRip + (ulong)i;
			ulong target = unchecked((ulong)((long)(callRip + 5) + rel32));
			Log.Debug($"Import#{dispatchIndex} near-call @{callRip:X16}: target=0x{target:X16}");
			for (int importIndex = 0; importIndex < _importEntries.Length; importIndex++)
			{
				if (_importEntries[importIndex].Address != target)
				{
					continue;
				}

				string nid = _importEntries[importIndex].Nid;
				if (_moduleManager.TryGetExport(nid, out var export))
				{
					Log.Debug(
						$"Import#{dispatchIndex} near-call import: index={importIndex} {export.LibraryName}:{export.Name} ({nid})");
				}
				else
				{
					Log.Debug(
						$"Import#{dispatchIndex} near-call import: index={importIndex} nid={nid}");
				}
				break;
			}

			if (cpuContext.Memory.TryRead(target, targetBytes))
			{
				Log.Debug(
					$"Import#{dispatchIndex} near-call target bytes @0x{target:X16}: " +
					BitConverter.ToString(targetBytes.ToArray()).Replace("-", " "));
				if (targetBytes[0] == 0xFF && targetBytes[1] == 0x25)
				{
					int slotRel32 = BitConverter.ToInt32(targetBytes.Slice(2, 4));
					ulong slot = unchecked((ulong)((long)(target + 6) + slotRel32));
					if (cpuContext.TryReadUInt64(slot, out var slotTarget))
					{
						Log.Debug(
							$"Import#{dispatchIndex} near-call PLT slot: [0x{slot:X16}] = 0x{slotTarget:X16}");
						for (int importIndex = 0; importIndex < _importEntries.Length; importIndex++)
						{
							if (_importEntries[importIndex].Address != slotTarget)
							{
								continue;
							}

							string nid = _importEntries[importIndex].Nid;
							if (_moduleManager.TryGetExport(nid, out var export))
							{
								Log.Debug(
									$"Import#{dispatchIndex} near-call PLT import: index={importIndex} {export.LibraryName}:{export.Name} ({nid})");
							}
							else
							{
								Log.Debug(
									$"Import#{dispatchIndex} near-call PLT import: index={importIndex} nid={nid}");
							}
							break;
						}
					}
				}
			}
		}
	}

	private static bool IsUnresolvedSentinel(ulong value)
	{
		return value == 65534 || value == 4294967294u || value == 18446744073709551614uL;
	}

	private static ulong ParseOptionalHexAddress(string? value)
	{
		if (string.IsNullOrWhiteSpace(value))
		{
			return 0;
		}

		var text = value.Trim();
		if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
		{
			text = text[2..];
		}

		return ulong.TryParse(
			text,
			System.Globalization.NumberStyles.HexNumber,
			System.Globalization.CultureInfo.InvariantCulture,
			out var address)
			? address
			: 0;
	}

	private static bool IsPlausibleReturnAddress(ulong address)
	{
		return address >= 12884901888L && address < 17592186044416L && !IsUnresolvedSentinel(address);
	}

	private static bool TryGetPlausibleReturnFromStack(ulong rsp, out ulong returnRip, out ulong nextRsp)
	{
		returnRip = 0uL;
		nextRsp = rsp;
		if (rsp <= 65536 || rsp >= 140737488355328L)
		{
			return false;
		}
		ulong num = rsp & 0xFFFFFFFFFFFFFFF8uL;
		ulong num2 = ((num >= 8) ? (num - 8) : num);
		for (int i = 0; i < 24; i++)
		{
			ulong num3 = num2 + (ulong)((long)i * 8L);
			if (TryReadStackU64(num3, out var value) && IsLikelyReturnAddress(value))
			{
				returnRip = value;
				nextRsp = num3 + 8;
				return true;
			}
		}
		for (ulong num4 = 1uL; num4 < 8; num4++)
		{
			for (int j = 0; j < 24; j++)
			{
				ulong num5 = rsp + num4 + (ulong)((long)j * 8L);
				if (TryReadStackU64(num5, out var value2) && IsLikelyReturnAddress(value2))
				{
					returnRip = value2;
					ulong num6 = num5 + 8;
					nextRsp = (num6 + 7) & 0xFFFFFFFFFFFFFFF8uL;
					return true;
				}
			}
		}
		return false;
	}

	// The committed readable run of host pages that contains address.
	private unsafe static bool TryQueryReadableRange(ulong address, out ulong start, out ulong end)
	{
		start = 0;
		end = 0;
		if (address <= 65536 || address >= 140737488355328L ||
			VirtualQuery((void*)address, out var info, (nuint)sizeof(MEMORY_BASIC_INFORMATION64)) == 0)
		{
			return false;
		}

		var regionEnd = info.BaseAddress + info.RegionSize;
		if (regionEnd < info.BaseAddress || address > regionEnd - 8 ||
			info.State != 4096 || !IsReadableProtection(info.Protect))
		{
			return false;
		}

		start = info.BaseAddress;
		end = regionEnd;
		return true;
	}

	private unsafe static bool TryReadStackU64(ulong address, out ulong value)
	{
		value = 0uL;
		if (address <= 65536 || address >= 140737488355328L)
		{
			return false;
		}
		if (VirtualQuery((void*)address, out var lpBuffer, (nuint)sizeof(MEMORY_BASIC_INFORMATION64)) == 0)
		{
			return false;
		}
		ulong num = lpBuffer.BaseAddress + lpBuffer.RegionSize;
		if (num < lpBuffer.BaseAddress || address > num - 8)
		{
			return false;
		}
		if (lpBuffer.State != 4096 || !IsReadableProtection(lpBuffer.Protect))
		{
			return false;
		}
		try
		{
			value = *(ulong*)address;
			return true;
		}
		catch
		{
			return false;
		}
	}

	// A module the loader could not place in the PS5 module range is mapped wherever the
	// host puts it, which on macOS is often below the plausible window (Wukong's libc.prx
	// at 0x1E2EEC000, its Wwise plugins at 0x230C43000). Calls out of such a module still
	// return into code, so below the window an executable page is enough.
	private const ulong LowestLowModuleAddress = 0x10000UL;

	private static bool IsLikelyReturnAddress(ulong address)
	{
		if (!IsPlausibleReturnAddress(address))
		{
			return address >= LowestLowModuleAddress &&
				address < 12884901888L &&
				!IsUnresolvedSentinel(address) &&
				IsExecutableAddress(address);
		}
		return IsExecutableAddress(address);
	}

	// Front cache for the executable-page set, direct-mapped by page number.
	// Every import dispatch sanity-checks its return address through here, so
	// the cost of the ConcurrentDictionary probe behind it was being paid about
	// a million times a second in Quake. A slot only ever holds a page that was
	// verified executable, and a ulong store is atomic on both architectures
	// the emulator targets, so a racing write can publish a different verified
	// page but never a half-written address.
	private const int ExecutablePageCacheMask = 4095;
	private static readonly ulong[] _executablePageCache = new ulong[ExecutablePageCacheMask + 1];

	private unsafe static bool IsExecutableAddress(ulong address)
	{
		var pageAddress = address & ~0xFFFUL;
		if (pageAddress == 0)
		{
			// 0 is the empty-slot sentinel, so the null page can never be served
			// from the cache. The only caller rejects it earlier anyway.
			return false;
		}

		var slot = (int)((pageAddress >> 12) & ExecutablePageCacheMask);
		if (_executablePageCache[slot] == pageAddress)
		{
			return true;
		}

		if (_knownExecutablePages.ContainsKey(pageAddress))
		{
			_executablePageCache[slot] = pageAddress;
			return true;
		}

		if (VirtualQuery((void*)address, out var lpBuffer, (nuint)sizeof(MEMORY_BASIC_INFORMATION64)) == 0)
		{
			return false;
		}

		var executable = lpBuffer.State == 4096 && IsExecutableProtection(lpBuffer.Protect);
		if (executable)
		{
			_knownExecutablePages.TryAdd(pageAddress, 0);
			_executablePageCache[slot] = pageAddress;
		}

		return executable;
	}

	private static ulong AlignUp(ulong value, ulong alignment)
	{
		if (alignment == 0)
		{
			return value;
		}
		ulong num = alignment - 1;
		return (value + num) & ~num;
	}

	private static bool IsReadableProtection(uint protect)
	{
		if ((protect & 0x100) != 0 || (protect & 1) != 0)
		{
			return false;
		}
		return (protect & 0xEE) != 0;
	}

	private static bool IsExecutableProtection(uint protect)
	{
		return (protect & 0xF0) != 0;
	}
}
