// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Metal.V2;

// Passive on the exact base path. The coarse policy is an opportunity estimate, not
// permission to defer memory reads. The strict policy drains unclassified writers
// before the next command-memory observation, including token validation.
internal sealed class MetalBatchCensus
{
    internal const int Limit = 64;
    internal sealed class Distribution
    {
        internal int Pending;
        internal long Work;
        internal long Batches;
        internal int Maximum;
        internal readonly long[] Histogram = new long[Limit + 1];
        internal readonly long[] Cuts = new long[Enum.GetValues<Cut>().Length];

        internal void Add()
        {
            Work++;
            if (++Pending == Limit) Close(Cut.Capacity);
        }

        internal void Close(Cut reason)
        {
            Cuts[(int)reason]++;
            if (Pending == 0) return;
            Histogram[Pending]++;
            Batches++;
            Maximum = Math.Max(Maximum, Pending);
            Pending = 0;
        }

        internal string Format(string name) =>
            $"{name}_work={Work} {name}_batches={Batches} {name}_max={Maximum} " +
            $"{name}_hist={string.Join(',', Histogram.Select((n, i) => (n, i)).Where(x => x.n != 0).Select(x => $"{x.i}:{x.n}"))} " +
            $"{name}_cuts={string.Join(',', Cuts.Select((n, i) => (n, i)).Where(x => x.n != 0).Select(x => $"{(Cut)x.i}:{x.n}"))}";
    }

    internal enum Cut { Capacity, Pass, Kind, Barrier, Label, Wait, Dma, Operand, Submission, Frame, CommandRead, Other }
    internal readonly Distribution Coarse = new();
    internal readonly Distribution Strict = new();
    private bool? _compute;
    private long _draws;
    private long _dispatches;
    private long _frame;
    private readonly long[] _opcodes = new long[256];

    internal void Work(bool compute)
    {
        if (_compute is { } prior && prior != compute) Close(Cut.Kind);
        _compute = compute;
        if (compute) _dispatches++; else _draws++;
        Strict.Add();
    }

    internal void Encoded() => Coarse.Add();

    internal void Close(Cut reason)
    {
        Coarse.Close(reason);
        Strict.Close(reason);
    }

    internal static Cut? PacketCut(uint header)
    {
        var opcode = PacketHeader.Opcode(header);
        if (opcode == PacketOpcode.Nop)
        {
            return PacketHeader.CustomCode(header) switch
            {
                0 or PacketCustomCode.PushMarker or PacketCustomCode.PopMarker or PacketCustomCode.DrawIndexAuto => null,
                PacketCustomCode.WaitMemory32 or PacketCustomCode.WaitMemory64 or PacketCustomCode.WaitFlipDone => Cut.Wait,
                _ => Cut.Other,
            };
        }
        return opcode switch
        {
            PacketOpcode.WaitRegisterMemory or PacketOpcode.WaitRegisterMemory64 or
                PacketOpcode.WaitOnCeCounter or PacketOpcode.WaitOnDeCounterDiff => Cut.Wait,
            PacketOpcode.EventWriteEndOfPipe or PacketOpcode.EventWriteEndOfShader or
                PacketOpcode.ReleaseMemory or PacketOpcode.WriteData => Cut.Label,
            PacketOpcode.DmaData or PacketOpcode.CopyData or PacketOpcode.AtomicMemory or
                PacketOpcode.MemorySemaphore or PacketOpcode.ConditionalWrite => Cut.Dma,
            PacketOpcode.SurfaceSync or PacketOpcode.AcquireMemory or PacketOpcode.PfpSyncMe or
                PacketOpcode.EventWrite => Cut.Barrier,
            PacketOpcode.SetShaderRegisterIndirect or PacketOpcode.SetContextRegisterIndirect or
                PacketOpcode.SetUserConfigRegisterIndirect or PacketOpcode.IndexAttributesIndirect or
                PacketOpcode.ConditionalExecute or PacketOpcode.SetPredication or
                PacketOpcode.IndirectBuffer or PacketOpcode.IndirectBufferConstant or PacketOpcode.GetLodStats => Cut.Operand,
            _ => null,
        };
    }

    internal void Packet(uint header)
    {
        _opcodes[PacketHeader.Opcode(header)]++;
        if (PacketCut(header) is { } reason) Close(reason);
    }

    internal void Frame()
    {
        Close(Cut.Frame);
        Console.Error.WriteLine($"[METAL-V2][BATCH-CENSUS] frame={++_frame} draws={_draws} dispatches={_dispatches} " +
            Coarse.Format("coarse") + " " + Strict.Format("strict") + " packets=" +
            string.Join(',', _opcodes.Select((n, i) => (n, i)).Where(x => x.n != 0).Select(x => $"{PacketOpcodeNames.Of((uint)x.i)}:{x.n}")));
        _draws = _dispatches = 0;
        Reset(Coarse);
        Reset(Strict);
        Array.Clear(_opcodes);
    }

    private static void Reset(Distribution distribution)
    {
        distribution.Work = distribution.Batches = 0;
        distribution.Maximum = 0;
        Array.Clear(distribution.Histogram);
        Array.Clear(distribution.Cuts);
    }
}
