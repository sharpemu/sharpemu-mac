// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// Per-encoder GPU timing, behind SHARPEMU_METAL_V2_PROFILE_PASSES=1.
//
// The shader profiler commits after every draw, which ends the render pass after every
// draw and charges each one a full load and store of its attachments: it ranks shaders
// but says nothing true about what a pass costs. This one leaves the submission exactly
// as it is and asks the GPU for timestamps at the stage boundaries of every render pass
// and compute encoder (MTLCounterSamplingPointAtStageBoundary, which Apple silicon
// supports), so a pass's time includes its real loads and stores and the frame keeps
// its real shape. Every other encoder the host opens on the command buffer being recorded
// (transfer blits, the tiler's compute encoders, clears, fills, staged stores, label copies)
// is opened through its pass descriptor with timestamps at its start and end
// (PassProfileOpenEncoder) and gets a record of its own, so a gap left between two records
// is time no encoder was running. One line per encoder, resolved when its submission
// completes; scripts aggregate them by frame and by pass.
internal sealed partial class MetalRenderHost
{
    // Settable for the device tests, which turn it on around one host.
    internal static bool PassProfileEnabled { get; set; } = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PROFILE_PASSES"), "1", StringComparison.Ordinal);

    private const nuint PassProfileSamples = 4096;
    private const ulong CounterErrorValue = ulong.MaxValue;

    private static readonly nint SelSampleBufferAttachments = MetalNative.Selector("sampleBufferAttachments");
    private static readonly nint SelSetSampleBuffer = MetalNative.Selector("setSampleBuffer:");
    private static readonly nint SelSetStartOfVertexSampleIndex = MetalNative.Selector("setStartOfVertexSampleIndex:");
    private static readonly nint SelSetEndOfVertexSampleIndex = MetalNative.Selector("setEndOfVertexSampleIndex:");
    private static readonly nint SelSetStartOfFragmentSampleIndex =
        MetalNative.Selector("setStartOfFragmentSampleIndex:");
    private static readonly nint SelSetEndOfFragmentSampleIndex = MetalNative.Selector("setEndOfFragmentSampleIndex:");
    private static readonly nint SelSetStartOfEncoderSampleIndex = MetalNative.Selector("setStartOfEncoderSampleIndex:");
    private static readonly nint SelSetEndOfEncoderSampleIndex = MetalNative.Selector("setEndOfEncoderSampleIndex:");
    private static readonly nint SelComputeEncoderWithDescriptor =
        MetalNative.Selector("computeCommandEncoderWithDescriptor:");
    private static readonly nint SelBlitEncoderWithDescriptor = MetalNative.Selector("blitCommandEncoderWithDescriptor:");
    private static readonly nint SelSetDispatchType = MetalNative.Selector("setDispatchType:");
    private static readonly nint SelResolveCounterRange = MetalNative.Selector("resolveCounterRange:");
    private static readonly nint SelSampleTimestamps = MetalNative.Selector("sampleTimestamps:gpuTimestamp:");

    private sealed class PassRecord
    {
        public int Frame;
        public int Index;
        public string Kind = string.Empty;
        public string Shape = string.Empty;
        public long LoadBytes;
        public long StoreBytes;
        public int Draws;
        public int Joins;
        public ulong Groups;
        public Dictionary<string, int>? Shaders;
        public string End = string.Empty;
        public List<string>? After;
        public int Buffer;
        public int FirstSample;
        public int SampleCount;
        public List<int>? IndirectSlots;
    }

    private sealed class PassSubmission
    {
        public readonly List<PassRecord> Records = new();
        public int Frame;
        public string Leading = string.Empty;
    }

    private nint _ppCounterSet;
    private bool _ppUnavailable;
    private PassSubmission? _ppCurrent;
    private PassRecord? _ppOpen;
    private PassRecord? _ppLast;
    // Metal allows only a few dozen counter sample buffers per process, so a fixed set is
    // used as one ring of sample slots shared by every submission in flight.
    private const int PassProfileBufferCount = 24;
    private readonly List<nint> _ppBuffers = new();
    private readonly int[] _ppOutstanding = new int[PassProfileBufferCount];
    private int _ppCursorBuffer = -1;
    private int _ppNextSample;
    private long _ppSkipped;
    private int _ppFrame;
    private List<string>? _ppLeading;
    private string _ppTransferCause = "transfer";
    private readonly Dictionary<ulong, PassSubmission> _ppInFlight = new();
    private int _ppAllocated;
    private ulong _ppCalCpu;
    private ulong _ppCalGpu;
    private double _ppNsPerTick = 1.0;
    private bool _ppCalibrated;
    // Records of encoders PassProfileOpenEncoder opened since the last transfer named itself
    // (PassProfileNoteBetween): that label is theirs.
    private readonly List<PassRecord> _ppUnlabelled = new();
    // The cause of the transfer being encoded (BeginTransfer), the name of the encoders it opens.
    private string? _ppEncoderCause;
    private static readonly string[] PassProfileCauseNames = Enum.GetNames<MetalTransferCause>();
    // Encoders recorded with no samples because the ring had wrapped onto unread ones.
    private long _ppUnsampled;

    /// <summary>Device tests: receives what the profile prints for each completed submission.</summary>
    internal Action<string>? PassProfileOutputForTest { get; set; }

    /// <summary>Device tests: shared compute encoders opened, each of which the profile records as k=compute.</summary>
    internal long ComputeEncodersOpenedForTest => Interlocked.Read(ref _computeEncodersOpened);

    private bool PassProfileReady()
    {
        if (!PassProfileEnabled || _ppUnavailable)
        {
            return false;
        }

        if (_ppCounterSet != 0)
        {
            return true;
        }

        // MTLCounterSamplingPointAtStageBoundary = 0.
        if (!MetalNative.SendBool(_device, MetalNative.Selector("supportsCounterSampling:"), 0))
        {
            _ppUnavailable = true;
            Console.Error.WriteLine("[METAL-V2][PASS-PROF] stage-boundary counter sampling is not supported.");
            return false;
        }

        // MTLCounterSamplingPoint: stage 0, draw 1, dispatch 2, tile dispatch 3, blit 4. Every
        // encoder is sampled at its stage boundaries, through its pass descriptor's sample
        // buffer attachments, which needs only the first.
        var supportsSelector = MetalNative.Selector("supportsCounterSampling:");
        Console.Error.WriteLine(
            "[METAL-V2][PASS-PROF] counter sampling: stage=1 " +
            $"draw={(MetalNative.SendBool(_device, supportsSelector, 1) ? 1 : 0)} " +
            $"dispatch={(MetalNative.SendBool(_device, supportsSelector, 2) ? 1 : 0)} " +
            $"tile_dispatch={(MetalNative.SendBool(_device, supportsSelector, 3) ? 1 : 0)} " +
            $"blit={(MetalNative.SendBool(_device, supportsSelector, 4) ? 1 : 0)}");

        var sets = MetalNative.Send(_device, MetalNative.Selector("counterSets"));
        var count = sets == 0 ? 0 : MetalV2Native.SendValue(sets, MetalNative.Selector("count"));
        for (nuint index = 0; index < count; index++)
        {
            var set = MetalNative.SendAtIndex(sets, SelObjectAtIndex, index);
            var name = MetalNative.Send(set, MetalNative.Selector("name"));
            var text = Marshal.PtrToStringUTF8(MetalNative.Send(name, MetalNative.Selector("UTF8String")));
            if (string.Equals(text, "timestamp", StringComparison.Ordinal))
            {
                // Element of an autoreleased array: kept for the life of the host.
                _ppCounterSet = MetalNative.Send(set, SelRetain);
                break;
            }
        }

        if (_ppCounterSet == 0)
        {
            _ppUnavailable = true;
            Console.Error.WriteLine("[METAL-V2][PASS-PROF] the device has no timestamp counter set.");
            return false;
        }

        PassProfileCalibrate();
        return true;
    }

    private unsafe void PassProfileCalibrate()
    {
        ulong cpu = 0;
        ulong gpu = 0;
        MetalNative.SendVoid(_device, SelSampleTimestamps, (nint)(&cpu), (nint)(&gpu));
        if (_ppCalCpu != 0 && gpu > _ppCalGpu && cpu > _ppCalCpu)
        {
            var ratio = (double)(cpu - _ppCalCpu) / (gpu - _ppCalGpu);
            if (!_ppCalibrated)
            {
                Console.Error.WriteLine(string.Create(
                    CultureInfo.InvariantCulture, $"[METAL-V2][PASS-PROF] calibration ns_per_tick={ratio:F4}"));
            }

            _ppNsPerTick = ratio;
            _ppCalibrated = true;
        }

        if (_ppCalCpu == 0 || cpu - _ppCalCpu > 1_000_000_000UL)
        {
            _ppCalCpu = cpu;
            _ppCalGpu = gpu;
        }
    }

    private nint PassProfileTakeBuffer()
    {
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLCounterSampleBufferDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setCounterSet:"), _ppCounterSet);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setStorageMode:"), 0);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setSampleCount:"), PassProfileSamples);
        nint error = 0;
        var buffer = MetalNative.Send(
            _device, MetalNative.Selector("newCounterSampleBufferWithDescriptor:error:"), descriptor, ref error);
        MetalNative.SendVoid(descriptor, SelRelease);
        if (buffer == 0)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][PASS-PROF] counter sample buffer allocation failed after {_ppAllocated}: " +
                $"{(error == 0 ? "no error" : MetalNative.DescribeError(error))}");
            return 0;
        }

        _ppAllocated++;
        return buffer;
    }

    // Reserves sample indices for one encoder of the submission being recorded. With
    // keepUnsampled an encoder the ring has no samples for is still recorded, with none,
    // so it is named and placed in the submission (us=?).
    private PassRecord? PassProfileReserve(string kind, int samples, bool keepUnsampled = false)
    {
        if (!PassProfileReady())
        {
            return null;
        }

        var submission = _ppCurrent ??= new PassSubmission { Frame = _ppFrame };
        if (_ppCursorBuffer < 0 || _ppNextSample + samples > (int)PassProfileSamples)
        {
            var next = (_ppCursorBuffer + 1) % PassProfileBufferCount;
            if (next >= _ppBuffers.Count)
            {
                var created = PassProfileTakeBuffer();
                if (created == 0)
                {
                    _ppUnavailable = true;
                    return null;
                }

                _ppBuffers.Add(created);
            }

            if (_ppOutstanding[next] != 0)
            {
                // The ring has wrapped onto samples not yet read: this encoder goes unsampled.
                _ppSkipped++;
                if (!keepUnsampled)
                {
                    return null;
                }

                _ppUnsampled++;
                MetalV2Diagnostics.WarnOnce("pass-profile-unsampled",
                    "[METAL-V2][PASS-PROF] the counter sample ring has wrapped onto unread samples: encoders are " +
                    "recorded with us=? and counted as unsampled= on the sub line.");
                var unsampled = new PassRecord
                {
                    Frame = _ppFrame,
                    Index = submission.Records.Count,
                    Kind = kind,
                    Buffer = -1,
                };
                PassProfileTakeLeading(submission);
                submission.Records.Add(unsampled);
                return unsampled;
            }

            _ppCursorBuffer = next;
            _ppNextSample = 0;
        }

        var record = new PassRecord
        {
            Frame = _ppFrame,
            Index = submission.Records.Count,
            Kind = kind,
            Buffer = _ppCursorBuffer,
            FirstSample = _ppNextSample,
            SampleCount = samples,
        };
        _ppNextSample += samples;
        _ppOutstanding[_ppCursorBuffer]++;
        PassProfileTakeLeading(submission);
        submission.Records.Add(record);
        return record;
    }

    private void PassProfileTakeLeading(PassSubmission submission)
    {
        if (_ppLeading is not null && submission.Records.Count == 0)
        {
            submission.Leading = string.Join(',', _ppLeading);
            _ppLeading = null;
        }
    }

    /// <summary>
    /// Opens an encoder of the open command buffer with timestamps at its start and end and a
    /// record of its own, named for site, or for the transfer being encoded when site is null.
    /// Called only with the profile on, in place of the plain call. An encoder of another
    /// command buffer (a prelude, a private one) is opened plain and not recorded: its records
    /// would be resolved with a submission it is not part of. One the ring has no samples for
    /// is opened plain and recorded unsampled.
    /// </summary>
    private nint PassProfileOpenEncoder(nint commandBuffer, MetalProfiledEncoder kind, string? site, bool concurrent = false)
    {
        var name = kind switch
        {
            MetalProfiledEncoder.Blit => "blit",
            MetalProfiledEncoder.TilerCompute => "tiler-compute",
            _ => "host-compute",
        };
        var record = commandBuffer == _commandBuffer && commandBuffer != 0
            ? PassProfileReserve(name, 2, keepUnsampled: true)
            : null;
        if (record is not null)
        {
            if (site is not null)
            {
                // Its own operation: the encoders before it are done waiting for a label.
                _ppUnlabelled.Clear();
            }

            record.Shape = site ?? _ppEncoderCause ?? "transfer";
            record.End = "end";
            _ppUnlabelled.Add(record);
            _ppLast = record;
        }

        if (record is null || record.Buffer < 0)
        {
            return kind == MetalProfiledEncoder.Blit
                ? MetalNative.Send(commandBuffer, SelBlitEncoder)
                : concurrent
                    ? MetalNative.Send(commandBuffer, SelComputeEncoderWithDispatchType, DispatchTypeConcurrent)
                    : MetalNative.Send(commandBuffer, SelComputeEncoder);
        }

        var descriptor = kind == MetalProfiledEncoder.Blit
            ? MetalNative.Send(MetalNative.Class("MTLBlitPassDescriptor"), MetalNative.Selector("blitPassDescriptor"))
            : MetalNative.Send(MetalNative.Class("MTLComputePassDescriptor"), MetalNative.Selector("computePassDescriptor"));
        var attachments = MetalNative.Send(descriptor, SelSampleBufferAttachments);
        var slot = MetalNative.SendAtIndex(attachments, SelObjectAtIndex, 0);
        MetalNative.SendVoid(slot, SelSetSampleBuffer, _ppBuffers[record.Buffer]);
        MetalV2Native.SendVoidValue(slot, SelSetStartOfEncoderSampleIndex, (nuint)record.FirstSample);
        MetalV2Native.SendVoidValue(slot, SelSetEndOfEncoderSampleIndex, (nuint)record.FirstSample + 1);
        if (kind == MetalProfiledEncoder.Blit)
        {
            return MetalNative.Send(commandBuffer, SelBlitEncoderWithDescriptor, descriptor);
        }

        // MTLDispatchType: serial 0, concurrent 1, as the plain call would have opened it.
        MetalV2Native.SendVoidValue(descriptor, SelSetDispatchType, concurrent ? (nuint)DispatchTypeConcurrent : 0);
        return MetalNative.Send(commandBuffer, SelComputeEncoderWithDescriptor, descriptor);
    }

    nint IMetalSubmissionContext.OpenProfiledEncoder(
        nint commandBuffer, MetalProfiledEncoder kind, string? site, bool concurrent) =>
        PassProfileOpenEncoder(commandBuffer, kind, site, concurrent);

    /// <summary>Attaches stage-boundary timestamps to a render pass about to be encoded.</summary>
    private void PassProfileAttachRender(nint descriptor)
    {
        _ppOpen = null;
        _ppUnlabelled.Clear();
        var record = PassProfileReserve("render", 4);
        if (record is null)
        {
            return;
        }

        var attachments = MetalNative.Send(descriptor, SelSampleBufferAttachments);
        var slot = MetalNative.SendAtIndex(attachments, SelObjectAtIndex, 0);
        var buffer = _ppBuffers[record.Buffer];
        MetalNative.SendVoid(slot, SelSetSampleBuffer, buffer);
        var first = (nuint)record.FirstSample;
        MetalV2Native.SendVoidValue(slot, SelSetStartOfVertexSampleIndex, first);
        MetalV2Native.SendVoidValue(slot, SelSetEndOfVertexSampleIndex, first + 1);
        MetalV2Native.SendVoidValue(slot, SelSetStartOfFragmentSampleIndex, first + 2);
        MetalV2Native.SendVoidValue(slot, SelSetEndOfFragmentSampleIndex, first + 3);
        _ppOpen = record;
        _ppLast = record;
    }

    /// <summary>Describes the attachments of the pass just opened, with its own load and store bytes.</summary>
    private void PassProfileDescribe(in RenderingState state, long loadBytes, long storeBytes)
    {
        var record = _ppOpen;
        if (record is null)
        {
            return;
        }

        var builder = new System.Text.StringBuilder();
        builder.Append(state.Width).Append('x').Append(state.Height);
        if (state.Layers > 1)
        {
            builder.Append('x').Append(state.Layers);
        }

        if (state.Samples > 1)
        {
            builder.Append(" s").Append(state.Samples);
        }

        builder.Append(" c=[");
        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            ref readonly var attachment = ref state.ColorAttachments[index];
            var clears = attachment.IsClear || (index < MaxColorAttachments && _overrideClear[index]);
            builder.Append(index == 0 ? string.Empty : ",")
                .Append(attachment.Format)
                .Append(':').Append(clears ? 'C' : index < MaxColorAttachments && _elidedLoads[index] ? 'D' : 'L')
                .Append(':').Append(((ulong)attachment.View.Handle & 0xFFFFF).ToString("X5", CultureInfo.InvariantCulture));
        }

        builder.Append(']');
        ref readonly var depth = ref state.DepthStencilAttachment;
        if (depth.HasDepth && depth.View.Handle != 0)
        {
            builder.Append(" d=").Append(depth.Format).Append(':').Append(depth.DepthClear ? 'C' : 'L')
                .Append(':').Append(((ulong)depth.View.Handle & 0xFFFFF).ToString("X5", CultureInfo.InvariantCulture));
        }

        if (depth.HasStencil && depth.View.Handle != 0)
        {
            builder.Append(" st=").Append(depth.StencilClear ? 'C' : 'L');
        }

        record.Shape = builder.ToString();
        record.LoadBytes = loadBytes;
        record.StoreBytes = storeBytes;
    }

    /// <summary>Opens a sampled compute encoder, or returns 0 to let the caller open a plain one.</summary>
    private nint PassProfileComputeEncoder(nint commandBuffer)
    {
        _ppUnlabelled.Clear();
        var record = PassProfileReserve("compute", 2);
        if (record is null)
        {
            return 0;
        }

        var descriptor = MetalNative.Send(
            MetalNative.Class("MTLComputePassDescriptor"), MetalNative.Selector("computePassDescriptor"));
        var attachments = MetalNative.Send(descriptor, SelSampleBufferAttachments);
        var slot = MetalNative.SendAtIndex(attachments, SelObjectAtIndex, 0);
        MetalNative.SendVoid(slot, SelSetSampleBuffer, _ppBuffers[record.Buffer]);
        MetalV2Native.SendVoidValue(slot, SelSetStartOfEncoderSampleIndex, (nuint)record.FirstSample);
        MetalV2Native.SendVoidValue(slot, SelSetEndOfEncoderSampleIndex, (nuint)record.FirstSample + 1);
        var encoder = MetalNative.Send(commandBuffer, SelComputeEncoderWithDescriptor, descriptor);
        MetalMemoryTelemetry.ComputeEncoders.Add();
        _ppOpen = record;
        _ppLast = record;
        return encoder;
    }

    /// <summary>One draw or dispatch encoded into the open encoder.</summary>
    private void PassProfileNoteWork(string? label, string kind)
    {
        var record = _ppOpen;
        if (record is null)
        {
            return;
        }

        record.Draws++;
        var key = label is null ? kind : (kind.StartsWith("draw", StringComparison.Ordinal) ? label : $"{kind} {label}");
        key = key.Replace("stage=", string.Empty, StringComparison.Ordinal).Replace(' ', '/');
        record.Shaders ??= new Dictionary<string, int>(StringComparer.Ordinal);
        record.Shaders[key] = record.Shaders.TryGetValue(key, out var count) ? count + 1 : 1;
    }

    // The group counts each indirect dispatch of an encoder ran with, copied by the GPU right
    // behind the dispatch (after the barrier that orders it after the work that wrote them):
    // what a CPU read after the fact cannot see once a later dispatch rewrote the record.
    private const int PassProfileIndirectSlots = 16384;
    private nint _ppIndirectCounts;
    private int _ppIndirectNext;

    private unsafe void PassProfileNoteIndirectGroups(nint encoder, nint arguments, ulong offset)
    {
        var record = _ppOpen;
        if (record is null || (offset & 3) != 0)
        {
            return;
        }

        if (_ppIndirectCounts == 0)
        {
            _ppIndirectCounts = MetalNative.SendNewBuffer(
                _device, SelOcclusionNewBuffer, (nuint)(PassProfileIndirectSlots * 16), Mtl.ResourceStorageModeShared);
            if (_ppIndirectCounts == 0)
            {
                return;
            }
        }

        var slot = _ppIndirectNext++ % PassProfileIndirectSlots;
        (record.IndirectSlots ??= new List<int>()).Add(slot);
        // The thread-to-group kernel with one-thread groups on the identity axes copies the
        // three counts as they are.
        var parameters = stackalloc uint[8];
        parameters[0] = 1;
        parameters[1] = 1;
        parameters[2] = 1;
        parameters[3] = 0;
        parameters[4] = 1;
        parameters[5] = 2;
        parameters[6] = 0;
        parameters[7] = 0;
        MetalNative.SendVoid(encoder, SelSetComputePipelineState, ThreadDispatchPipeline());
        _boundComputePipeline = _threadDispatchPipeline;
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, arguments, (nuint)offset, 0);
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, _ppIndirectCounts, (nuint)(slot * 16), 1);
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, _ppIndirectCounts, (nuint)(slot * 16), 2);
        MetalNative.SendSetBytes(encoder, SelHullSetBytes, (nint)parameters, 8 * sizeof(uint), 3);
        MetalNative.SendDispatch(
            encoder,
            SelDispatchThreadgroups,
            new MtlSize { Width = 1, Height = 1, Depth = 1 },
            new MtlSize { Width = 1, Height = 1, Depth = 1 });
    }

    private string PassProfileIndirectGroups(PassRecord record)
    {
        if (record.IndirectSlots is not { Count: > 0 } slots || _ppIndirectCounts == 0)
        {
            return string.Empty;
        }

        var words = MetalNative.Send(_ppIndirectCounts, SelOcclusionContents);
        var text = new System.Text.StringBuilder(" ig=[");
        for (var index = 0; index < Math.Min(slots.Count, 8); index++)
        {
            var at = words + (slots[index] * 16);
            text.Append(index == 0 ? string.Empty : ",")
                .Append((uint)Marshal.ReadInt32(at)).Append('x')
                .Append((uint)Marshal.ReadInt32(at + 4)).Append('x')
                .Append((uint)Marshal.ReadInt32(at + 8));
        }

        return text.Append(slots.Count > 8 ? $",+{slots.Count - 8}]" : "]").ToString();
    }

    /// <summary>A depth-only draw joined the open pass; its draw and shader are counted as any other's.</summary>
    private void PassProfileNoteJoin()
    {
        if (_ppOpen is not null)
        {
            _ppOpen.Joins++;
        }
    }

    /// <summary>Mesh threadgroups one draw launched, the work a mesh-bound pass scales with.</summary>
    private void PassProfileNoteGroups(ulong groups)
    {
        if (_ppOpen is not null)
        {
            _ppOpen.Groups += groups;
        }
    }

    /// <summary>Why the open encoder ended, when something outside the draw stream ended it.</summary>
    private void PassProfileNoteEnd(string cause)
    {
        NoteUnionPassEnd(cause);
        if (_ppOpen is not null && _ppOpen.End.Length == 0)
        {
            _ppOpen.End = cause;
        }
    }

    /// <summary>The open encoder has ended.</summary>
    private void PassProfileClose()
    {
        if (_ppOpen is not null && _ppOpen.End.Length == 0)
        {
            _ppOpen.End = "end";
        }

        _ppOpen = null;
    }

    /// <summary>A transfer, clear or copy encoded after the last sampled encoder.</summary>
    private void PassProfileNoteBetween(string what)
    {
        if (!PassProfileEnabled)
        {
            return;
        }

        if (_ppUnlabelled.Count > 0)
        {
            // The transfer, clear or copy that opened them: what they did and on how much.
            foreach (var record in _ppUnlabelled)
            {
                record.Shape = $"{record.Shape} {what}";
            }

            _ppUnlabelled.Clear();
            _ppEncoderCause = null;
            return;
        }

        if (_ppLast is not null && _ppCurrent is not null && _ppCurrent.Records.Contains(_ppLast))
        {
            (_ppLast.After ??= new List<string>()).Add(what);
        }
        else
        {
            (_ppLeading ??= new List<string>()).Add(what);
        }
    }

    /// <summary>The submission is being committed; its records wait for it to complete.</summary>
    private void PassProfileCommit(ulong serial, bool present)
    {
        if (!PassProfileEnabled)
        {
            return;
        }

        _ppOpen = null;
        _ppUnlabelled.Clear();
        _ppEncoderCause = null;
        if (_ppCurrent is not null)
        {
            _ppInFlight[serial] = _ppCurrent;
            _ppCurrent = null;
        }

        _ppLast = null;
        if (present)
        {
            Console.Error.WriteLine(string.Create(
                CultureInfo.InvariantCulture, $"[METAL-V2][PASS-PROF] present f={_ppFrame} serial={serial}"));
            _ppFrame++;
        }
    }

    private double StageMicroseconds(long start, long end) =>
        start != 0 && end != 0 && (ulong)start != CounterErrorValue && (ulong)end != CounterErrorValue && end > start
            ? (end - start) * _ppNsPerTick / 1000.0
            : 0.0;

    /// <summary>The submission has finished; prints one line per sampled encoder.</summary>
    private void PassProfileResolve(ulong serial, bool completed, double gpuSeconds)
    {
        if (!PassProfileEnabled || !_ppInFlight.Remove(serial, out var submission))
        {
            return;
        }

        if (!completed)
        {
            // Not finished: its sample buffers may still be written, so they are not reused.
            return;
        }

        if (!completed)
        {
            foreach (var record in submission.Records)
            {
                _ppOutstanding[record.Buffer]--;
            }

            return;
        }

        PassProfileCalibrate();
        var culture = CultureInfo.InvariantCulture;
        var values = new long[4];
        var sum = 0.0;
        ulong lastEnd = 0;
        ulong firstStart = ulong.MaxValue;
        var lines = new System.Text.StringBuilder();
        var kindMicroseconds = new SortedDictionary<string, double>(StringComparer.Ordinal);
        var gaps = 0.0;
        var unsampled = 0;
        foreach (var record in submission.Records)
        {
            Array.Clear(values);
            if (record.Buffer < 0)
            {
                // Recorded with no samples (the ring had wrapped): named and placed, not timed.
                unsampled++;
                lines.Append(string.Create(
                    culture,
                    $"[METAL-V2][PASS-PROF] f={submission.Frame} s={serial} i={record.Index} k={record.Kind} " +
                    $"us=? gap_us=? draws={record.Draws} tg=0 ld_mb=0.0 st_mb=0.0 end={record.End} " +
                    $"a=[{record.Shape}] sh=[-] after=[-] n_after={record.After?.Count ?? 0} joins=0")).Append('\n');
                continue;
            }

            var data = MetalV2Native.SendRange(
                _ppBuffers[record.Buffer], SelResolveCounterRange, (nuint)record.FirstSample, (nuint)record.SampleCount);
            _ppOutstanding[record.Buffer]--;
            if (data != 0)
            {
                var bytes = MetalNative.Send(data, MetalNative.Selector("bytes"));
                var length = (int)MetalV2Native.SendValue(data, MetalNative.Selector("length"));
                Marshal.Copy(bytes, values, 0, Math.Min(length / 8, record.SampleCount));
            }

            ulong start = ulong.MaxValue;
            ulong end = 0;
            for (var sample = 0; sample < record.SampleCount; sample++)
            {
                var value = (ulong)values[sample];
                if (value == 0 || value == CounterErrorValue)
                {
                    continue;
                }

                start = Math.Min(start, value);
                end = Math.Max(end, value);
            }

            var microseconds = end > start && start != ulong.MaxValue ? (end - start) * _ppNsPerTick / 1000.0 : 0.0;
            // A render pass's vertex (tiling) and fragment phases, which on a tile-based GPU
            // run one after the other for the pass: which of the two a slow pass spends in.
            var stages = string.Empty;
            if (record.SampleCount == 4)
            {
                stages = string.Create(
                    culture,
                    $"vs_us={StageMicroseconds(values[0], values[1]):F0} fs_us={StageMicroseconds(values[2], values[3]):F0} ");
            }

            var gap = lastEnd != 0 && start != ulong.MaxValue
                ? ((double)start - lastEnd) * _ppNsPerTick / 1000.0
                : 0.0;
            if (start != ulong.MaxValue)
            {
                firstStart = Math.Min(firstStart, start);
                lastEnd = Math.Max(lastEnd, end);
            }

            sum += microseconds;
            gaps += Math.Max(gap, 0.0);
            kindMicroseconds[record.Kind] = kindMicroseconds.GetValueOrDefault(record.Kind) + microseconds;
            var shaders = record.Shaders is null
                ? "-"
                : string.Join(';', record.Shaders.OrderByDescending(pair => pair.Value).Take(6)
                    .Select(pair => $"{pair.Key}*{pair.Value}"));
            var after = record.After is null ? "-" : string.Join(',', record.After.Take(12)) +
                (record.After.Count > 12 ? $",+{record.After.Count - 12}" : string.Empty);
            lines.Append(string.Create(
                culture,
                $"[METAL-V2][PASS-PROF] f={submission.Frame} s={serial} i={record.Index} k={record.Kind} " +
                $"us={microseconds:F0} gap_us={gap:F0} {stages}draws={record.Draws} tg={record.Groups} " +
                $"ld_mb={record.LoadBytes / 1048576.0:F1} st_mb={record.StoreBytes / 1048576.0:F1} " +
                $"end={(record.End.Length == 0 ? "?" : record.End)} a=[{record.Shape}] sh=[{shaders}] " +
                $"after=[{after}] n_after={record.After?.Count ?? 0} joins={record.Joins}{PassProfileIndirectGroups(record)}")).Append('\n');
        }

        var span = lastEnd > firstStart && firstStart != ulong.MaxValue
            ? (lastEnd - firstStart) * _ppNsPerTick / 1000.0
            : 0.0;
        lines.Append(string.Create(
            culture,
            $"[METAL-V2][PASS-PROF] sub f={submission.Frame} s={serial} encoders={submission.Records.Count} " +
            $"enc_ms={sum / 1000.0:F2} span_ms={span / 1000.0:F2} gpu_ms={gpuSeconds * 1000.0:F2} " +
            $"skipped={_ppSkipped} leading=[{submission.Leading}] " +
            $"kind_us=[{string.Join(',', kindMicroseconds.Select(pair => string.Create(culture, $"{pair.Key}:{pair.Value:F0}")))}] " +
            $"gaps_us={gaps:F0} unsampled={unsampled} unsampled_total={_ppUnsampled}"));
        var text = lines.ToString();
        Console.Error.WriteLine(text);
        PassProfileOutputForTest?.Invoke(text);
    }
}
