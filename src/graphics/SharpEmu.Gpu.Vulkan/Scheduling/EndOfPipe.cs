// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Vulkan.Scheduling;

public enum EndOfPipeCompletion
{
    None,
    Interrupt,
    Flip,
    FlipAndInterrupt,
}

public readonly record struct EndOfPipeSignal(
    RecordingBuffer Buffer,
    ulong SubmitId,
    RecordedOperation DebugOperation,
    uint Arg0,
    uint Arg1,
    uint Arg2,
    uint Arg3,
    ulong DebugData,
    ulong? Destination = null,
    EndOfPipeCompletion Completion = EndOfPipeCompletion.None,
    ulong CompletionData = 0,
    int InterruptEventId = 0);

public sealed class EndOfPipe
{

    private enum WriteSize : uint
    {
        Dword = 4,
        Qword = 8,
    }

    private enum WriteAction
    {
        Write,
        WriteBack,
        Interrupt,
        InterruptWriteBack,
    }

    private readonly IEndOfPipeSink _sink;
    private readonly IFlipSubmitter _flips;

    public EndOfPipe(IEndOfPipeSink sink, IFlipSubmitter flips)
    {
        _sink = sink;
        _flips = flips;
    }

    internal void CompleteDeferred(in EndOfPipeWrite write)
    {
        if (write.Kind is EndOfPipeWriteKind.Flip or EndOfPipeWriteKind.FlipWithWrite32 or
            EndOfPipeWriteKind.FlipWithInterruptWriteBack32)
            _flips.CompleteFlip(write.FlipRequestId);
        if (write.Kind is EndOfPipeWriteKind.InterruptOnly or EndOfPipeWriteKind.Interrupt32 or
            EndOfPipeWriteKind.Interrupt64 or EndOfPipeWriteKind.InterruptWriteBack32 or
            EndOfPipeWriteKind.InterruptWriteBack64 or EndOfPipeWriteKind.FlipWithInterruptWriteBack32)
            _sink.TriggerInterrupt(write.EventId,
                write.Kind == EndOfPipeWriteKind.FlipWithInterruptWriteBack32 ? 0 : write.ContextId);
    }

    public void RecordEndOfPipeSignal(in EndOfPipeSignal signal)
    {
        ValidateSignal(signal);
        signal.Buffer.SetDebugInfo((uint)signal.DebugOperation, signal.SubmitId, signal.Arg0, signal.Arg1, signal.Arg2, signal.Arg3, signal.DebugData);

        var scheduler = signal.Buffer.Owner;
        if (signal.Completion != EndOfPipeCompletion.None && (!scheduler.Active || signal.Buffer != scheduler.Current))
        {
            throw GpuFatal.Fatal("End-of-pipe completion requires the current command buffer.");
        }

        switch (signal.Completion)
        {
            case EndOfPipeCompletion.None:
                return;
            case EndOfPipeCompletion.Interrupt:
            {
                var contextId = (uint)signal.CompletionData;
                var eventId = signal.InterruptEventId;
                scheduler.QueuePriorityCompletionAction(() => _sink.TriggerInterrupt(eventId, contextId));
                return;
            }

            case EndOfPipeCompletion.Flip:
            {
                var requestId = signal.CompletionData;
                scheduler.QueuePriorityCompletionAction(() => _flips.CompleteFlip(requestId));
                return;
            }

            case EndOfPipeCompletion.FlipAndInterrupt:
            {
                var requestId = signal.CompletionData;
                var eventId = signal.InterruptEventId;
                scheduler.QueuePriorityCompletionAction(() =>
                {
                    _flips.CompleteFlip(requestId);
                    _sink.TriggerInterrupt(eventId, 0);
                });
                return;
            }
        }
    }

    public void RecordWrite32(ulong submitId, RecordingBuffer buffer, ulong destination, uint value) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Dword, WriteAction.Write);

    public void RecordGdsWrite32(ulong submitId, RecordingBuffer buffer, ulong destination, uint wordOffset, uint wordCount) =>
        RecordEndOfPipeSignal(new EndOfPipeSignal(buffer, submitId, RecordedOperation.EopWrite, wordOffset, wordCount, 0, 0, destination, destination));

    public void RecordWrite64(ulong submitId, RecordingBuffer buffer, ulong destination, ulong value) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Qword, WriteAction.Write);

    public void RecordClockWrite(ulong submitId, RecordingBuffer buffer, ulong destination) =>
        RecordValueWrite(submitId, buffer, destination, 0, WriteSize.Qword, WriteAction.Write);

    public void RecordClockWriteWithWriteBack(ulong submitId, RecordingBuffer buffer, ulong destination) =>
        RecordValueWrite(submitId, buffer, destination, 0, WriteSize.Qword, WriteAction.WriteBack);

    public void RecordWrite64WithWriteBack(ulong submitId, RecordingBuffer buffer, ulong destination, ulong value) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Qword, WriteAction.WriteBack);

    public void RecordWrite32WithWriteBack(ulong submitId, RecordingBuffer buffer, ulong destination, uint value) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Dword, WriteAction.WriteBack);

    public void RecordWrite64WithInterruptAndWriteBack(ulong submitId, RecordingBuffer buffer, ulong destination, ulong value, int eventId, uint contextId = 0) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Qword, WriteAction.InterruptWriteBack, eventId, contextId);

    public void RecordWrite32WithInterruptAndWriteBack(ulong submitId, RecordingBuffer buffer, ulong destination, uint value, int eventId, uint contextId = 0) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Dword, WriteAction.InterruptWriteBack, eventId, contextId);

    public void RecordWrite64WithInterrupt(ulong submitId, RecordingBuffer buffer, ulong destination, ulong value, int eventId, uint contextId = 0) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Qword, WriteAction.Interrupt, eventId, contextId);

    public void RecordWrite32WithInterrupt(ulong submitId, RecordingBuffer buffer, ulong destination, uint value, int eventId, uint contextId = 0) =>
        RecordValueWrite(submitId, buffer, destination, value, WriteSize.Dword, WriteAction.Interrupt, eventId, contextId);

    public ulong PrepareVideoOutFlip(RecordingBuffer buffer, int handle, int index, int flipMode, long flipArg)
    {
        for (;;)
        {
            var result = _flips.SubmitFlipFromGpu(buffer, handle, index, flipMode, flipArg, out var requestId);
            if (result == 0)
            {
                if (requestId == 0)
                {
                    throw GpuFatal.Fatal("The GPU flip submission returned an invalid request ID of zero.");
                }

                return requestId;
            }

            if (result != IFlipSubmitter.FlipQueueFull)
            {
                throw GpuFatal.Fatal($"Could not submit the GPU flip: result={result} handle={handle} index={index} mode={flipMode} arg={flipArg}");
            }

            _flips.WaitForSubmitSlot();
        }
    }

    public void RecordWrite32WithInterruptWriteBackAndFlip(ulong submitId, RecordingBuffer buffer, ulong destination, uint value, int handle, int index, int flipMode, long flipArg, ulong requestId, int eventId) =>
        RecordEndOfPipeSignal(new EndOfPipeSignal(
            buffer, submitId, RecordedOperation.EopWriteBackFlip, (uint)handle, (uint)index, (uint)flipMode, value, (ulong)flipArg,
            destination, EndOfPipeCompletion.FlipAndInterrupt, requestId, eventId));

    public void RecordWrite32WithFlip(ulong submitId, RecordingBuffer buffer, ulong destination, uint value, int handle, int index, int flipMode, long flipArg, ulong requestId) =>
        RecordEndOfPipeSignal(new EndOfPipeSignal(
            buffer, submitId, RecordedOperation.EopFlip, (uint)handle, (uint)index, (uint)flipMode, value, (ulong)flipArg,
            destination, EndOfPipeCompletion.Flip, requestId));

    public void RecordFlipCompletion(ulong submitId, RecordingBuffer buffer, int handle, int index, int flipMode, long flipArg, ulong requestId) =>
        RecordEndOfPipeSignal(new EndOfPipeSignal(
            buffer, submitId, RecordedOperation.EopOnlyFlip, (uint)handle, (uint)index, (uint)flipMode, 0, (ulong)flipArg,
            Completion: EndOfPipeCompletion.Flip, CompletionData: requestId));

    public void QueueInterruptOnCompletion(RecordingBuffer buffer, int eventId, uint contextId)
    {
        RequireRecording(buffer);
        var scheduler = buffer.Owner;
        if (!scheduler.Active || buffer != scheduler.Current)
        {
            throw GpuFatal.Fatal("An end-of-pipe event requires the current command buffer.");
        }

        scheduler.QueuePriorityCompletionAction(() => _sink.TriggerInterrupt(eventId, contextId));
    }

    private static void ValidateSignal(in EndOfPipeSignal signal)
    {
        if (signal.Destination == 0)
        {
            throw GpuFatal.Fatal("Cannot record an end-of-pipe write to address zero.");
        }

        RequireRecording(signal.Buffer);
    }

    private static void RequireRecording(RecordingBuffer buffer)
    {
        if (buffer.IsInvalid)
        {
            throw SubmissionScheduler.Fatal("An end-of-pipe signal requires a recording command buffer.");
        }
    }

    private static RecordedOperation GetDebugOperation(WriteAction action) => action switch
    {
        WriteAction.Write => RecordedOperation.EopWrite,
        WriteAction.WriteBack or WriteAction.InterruptWriteBack => RecordedOperation.EopWriteBack,
        WriteAction.Interrupt => RecordedOperation.EopInterrupt,
        _ => throw GpuFatal.Fatal("The end-of-pipe write action is not supported."),
    };

    private void RecordValueWrite(ulong submitId, RecordingBuffer buffer, ulong destination, ulong value, WriteSize size, WriteAction action, int eventId = 0, uint contextId = 0)
    {
        var width = (uint)size;
        var valueLow = (uint)value;
        var valueHigh = (uint)(value >> 32);
        var interrupt = action is WriteAction.Interrupt or WriteAction.InterruptWriteBack;
        RecordEndOfPipeSignal(new EndOfPipeSignal(
            buffer,
            submitId,
            GetDebugOperation(action),
            width,
            interrupt ? contextId : valueLow,
            interrupt ? valueLow : valueHigh,
            interrupt ? valueHigh : 0,
            destination,
            destination,
            interrupt ? EndOfPipeCompletion.Interrupt : EndOfPipeCompletion.None,
            interrupt ? contextId : value,
            eventId));
    }
}
