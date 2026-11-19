// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    // ShaderCompileRequest.GraphicsPartialSubgroupLanes (SHARPEMU_GRAPHICS_PARTIAL_LANES=1), as the SPIR-V
    // translator lowers it (Gen5SpirvTranslator HostSubgroupMayBePartial, ReadGraphicsLane). A fragment
    // simdgroup can be partly empty: the lanes past the quads the rasterizer packed into it were never
    // launched. The PS5 wave behind it still has those lanes, dead but with registers, and whole-wave code
    // (S_OR_SAVEEXEC of EXEC and the like) turns them on. A shuffle from a thread that does not exist is
    // undefined, so DPP, DPP8, PERMLANE16 and V_READLANE reads of such a lane read the reading lane's own
    // value, and the lane counts as active when the guest EXEC mask has its bit. The pixel stage clamps
    // EXEC to the running lanes (ClampExecToLanes), so the guest mask, dead lanes' bits included, is kept
    // beside it in sharpemu_guest_exec. Only the pixel stage has real lanes that can be absent: the vertex
    // stage runs one lane per thread, and compute and mesh simdgroups are full.
    private sealed partial class CompilationContext
    {
        private const string GuestExecName = "sharpemu_guest_exec";

        private bool HostSubgroupMayBePartial =>
            _request.GraphicsPartialSubgroupLanes && _stage == Gen5MslStage.Pixel && !IsSingleLaneStage && !IsWave64;

        private void DeclareGuestExec(System.Text.StringBuilder source)
        {
            if (HostSubgroupMayBePartial)
            {
                source.AppendLine($"    ulong {GuestExecName} = 0ul;");
            }
        }

        // The entry mask: the launched lanes, as EXEC starts.
        private void InitializeGuestExec(System.Text.StringBuilder source)
        {
            if (HostSubgroupMayBePartial)
            {
                source.AppendLine($"    {GuestExecName} = (ulong)sharpemu_ballot(true);");
            }
        }

        // A new EXEC low dword before the clamp to the running lanes.
        private void RecordGuestExecLow(string unclamped)
        {
            if (HostSubgroupMayBePartial)
            {
                Line($"{GuestExecName} = ({GuestExecName} & 0xFFFFFFFF00000000ul) | (ulong)({unclamped});");
            }
        }

        private void RecordGuestExecHigh(string value)
        {
            if (HostSubgroupMayBePartial)
            {
                Line($"{GuestExecName} = ({GuestExecName} & 0xFFFFFFFFul) | ((ulong)({value}) << 32);");
            }
        }

        private void RecordGuestExec64(string unclamped)
        {
            if (HostSubgroupMayBePartial)
            {
                Line($"{GuestExecName} = {unclamped};");
            }
        }

        // The host thread of a guest lane: its position in the simdgroup. The upper half of a guest wave64
        // has no threads and maps onto the same positions, as in the SPIR-V lowering.
        private string GraphicsHostLane(string guestLane) => Temp("uint", $"({guestLane}) & 31u");

        private string IsHostLanePresent(string hostLane) =>
            Temp("bool", $"((sharpemu_ballot(true) >> {hostLane}) & 1u) != 0u");

        private string IsGuestLaneExecSet(string guestLane) =>
            Temp("bool", $"(({GuestExecName} >> (({guestLane}) & {(_waveLaneCount == 64 ? 63u : 31u)}u)) & 1ul) != 0ul");

        // A guest lane's register whatever its EXEC state; a lane that was never launched reads as the
        // reading lane's own value, which keeps the min/max/and/or reductions whole-wave code runs across
        // dead lanes exact.
        private string ReadGraphicsLane(string value, string guestLane)
        {
            var hostLane = GraphicsHostLane(guestLane);
            var present = IsHostLanePresent(hostLane);
            return Temp("uint", $"simd_shuffle({value}, (ushort)({present} ? {hostLane} : sharpemu_lane))");
        }

        // The fetch-inactive-off test of a source lane: its EXEC bit, from the running lanes' ballot for a
        // launched lane and from the guest mask for one the host never launched.
        private string IsGraphicsLaneUsable(string guestLane)
        {
            var hostLane = GraphicsHostLane(guestLane);
            var present = IsHostLanePresent(hostLane);
            var active = Temp("bool", $"((sharpemu_ballot(exec) >> {hostLane}) & 1u) != 0u");
            var guest = IsGuestLaneExecSet(guestLane);
            return Temp("bool", $"{present} ? {active} : {guest}");
        }
    }
}
