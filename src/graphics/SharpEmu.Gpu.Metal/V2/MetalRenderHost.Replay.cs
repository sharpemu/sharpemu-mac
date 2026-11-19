// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Draw and dispatch replay on the Metal host: the host half of RenderRecorder.TryReplayDraw and
/// RenderRecorder.TryReplayDispatch (the Vulkan presenter's is VulkanVideoPresenter.DrawReplay.cs).
///
/// The engine seals a draw (dispatch) whose bindings a next item may repeat, and when the next item repeats the
/// previous one's programs, descriptor words, vertex input and targets (RenderRecorder.StageInputMismatch and
/// SameVertexInput, which compare everything but the user data and, for a dispatch, the thread limits) it skips
/// preparing and binding them and asks the host to bind what the sealed item bound. The host answers true only when
/// what it binds is exactly what PrepareBindings, BindResources, BindVertexBuffers and CommitBindings would have
/// bound for the item; every refusal is counted by name and comes before anything is changed, so a refused item
/// takes the full path as if replay did not exist.
///
/// What the sealed item bound, and why it still stands:
/// <list type="bullet">
/// <item>The argument buffer of each stage: ring ranges of the command buffer the item was recorded into. They hold
/// while the submission serial is the same (the ring is only reused after its command buffer completes), and their
/// bytes are what WriteArgumentBuffer would write while every resolved binding they name is unchanged (below) and
/// the argument plan names nothing else that moves: a stage whose plan has a pointer field (GDS, device-address
/// page table, fault buffer, flattened table, shader data, mip statistics), a word field other than the constant
/// fault-buffer word, or a bindless image table is never captured.</item>
/// <item>The textures: every image binding of the stage was a texture-memo hit or was memoised by its acquire, so
/// each had a stamp (MetalImageCache.TryStampTextureLocked: no DCC read, no stencil plane, bounded aliases), and
/// the image cache's binding epoch and image set did not move from the start of the stage's BindResources to the
/// commit. While they still have not moved, no deferred refresh waits and the submission serial is the same,
/// nothing a binding of a cached image resolves to has changed and each image already notes this submission's use
/// (the bindless table memo's rule, MetalRenderHost.BindlessImages.cs). Storage images, host-movie planes and
/// unstamped bindings are never captured: their acquire is not memoised.</item>
/// <item>The read-only storage buffers and vertex streams: each is looked up again in the read-binding memo
/// (MetalRenderHost.ReadBindingMemo.cs), which answers only while the full chain would do nothing and give the same
/// window range (remaps, window cuts, single-copy holes and traps, images over the range, deferred fills, label and
/// staged copies). A miss, or a hit on another range, refuses. A vertex stream's mapped size is clamped again.</item>
/// <item>Written storage buffers (dispatches only; a replayable draw writes nothing): their invalidation and write
/// record must run for every dispatch, so the replay takes the same chain BindResources takes
/// (BindStageStorageBuffer) after every refusal check has passed; a range other than the captured one rewrites the
/// stage's argument buffer instead of reusing it.</item>
/// <item>The push data: the captured stage's shader data with the new item's user data (a dispatch: its shader data
/// built again from its own snapshot, thread limits and call data), published as CommitBindings publishes it.</item>
/// <item>The stage objects themselves (MetalStageBindings, one per stage kind): no PrepareBindings may have run for
/// the stage kind since the capture.</item>
/// </list>
/// A draw is also refused when the render pass it was sealed in has ended (upstream's scope rule): binding state is
/// re-sent to whatever encoder is open, but a pass end can land copies and write-backs the captured state does not
/// see. The item sequence counts the command thread's items (MetalRenderHost.CommandThread.cs): a batch boundary or
/// relay commands between two items break the chain, and the slice mode (SHARPEMU_METAL_V2_COMMAND_THREAD=0) has
/// no sequence, so nothing replays there.
///
/// Replay lives wholly in the GPU process: the shared engine and this host both run there (SplitGpuHost composes
/// MetalV2Backend in process B and runs the submitted command streams itself), so nothing of it crosses the split.
/// SHARPEMU_DRAW_REPLAY=0 / SHARPEMU_DISPATCH_REPLAY=0 (upstream's switches, read by the engine too) turn it off.
/// </summary>
internal sealed partial class MetalRenderHost
{
    internal bool DrawReplayEnabled { get; set; } = Environment.GetEnvironmentVariable("SHARPEMU_DRAW_REPLAY") != "0";

    internal bool DispatchReplayEnabled { get; set; } = Environment.GetEnvironmentVariable("SHARPEMU_DISPATCH_REPLAY") != "0";

    /// <summary>
    /// Tests: runs when a replay is asked for (true for a dispatch), before any check, as work a completion or another
    /// thread does between two items would.
    /// </summary>
    internal Action<bool>? ReplayProbe { get; set; }

    // Moves with every command-thread item, every relay run between items and every batch (MetalRenderHost.CommandThread.cs).
    private long _replayItemSequence;

    // PrepareBindings calls per stage kind: a stage object is the captured one only while its count stands.
    private readonly long[] _stagePrepares = new long[StageCount];

    // Per stage kind, the image cache's binding epoch and image set when its last BindResources began, and the
    // images whose acquire memoised a stamp in it (bit per image).
    private readonly long[] _replayBindEpoch = new long[StageCount];
    private readonly long[] _replayBindImageSet = new long[StageCount];
    private readonly ulong[] _replayStamped = new ulong[StageCount];

    // The last CommitBindings: its count, bind point, stages and whether each stage's images settled (the epoch and
    // image set as at its BindResources' start, no refresh waiting).
    private long _replayCommits;
    private GpuBindPoint _replayCommitPoint;
    private readonly MetalStageBindings?[] _replayCommitStages = new MetalStageBindings?[2];
    private int _replayCommitStageCount;
    private readonly bool[] _replayCommitSettled = new bool[2];
    private int _replayCommitLimitsDword;

    // ClampMappedSize and ObtainBuffer since the last BindResources, for mapping the vertex streams to their ranges.
    private readonly record struct ReplayClamp(ulong Address, ulong Requested, ulong Size);

    private readonly record struct ReplayObtain(ulong Address, ulong Size, bool Written, BufferBinding Binding);

    private const int ReplayRecordLimit = 64;
    private readonly ReplayClamp[] _replayClamps = new ReplayClamp[ReplayRecordLimit];
    private readonly ReplayObtain[] _replayObtains = new ReplayObtain[ReplayRecordLimit];
    private int _replayClampCount;
    private int _replayObtainCount;
    private bool _replayRecordsOverflowed;

    private sealed class ReplayStage
    {
        public MetalStageBindings Bindings = null!;
        public int StageIndex;
        public long Prepares;
        public long BindingEpoch;
        public long ImageSet;
        public nint ArgumentBuffer;
        public ulong ArgumentOffset;
        public bool ArgumentValid;
        public bool UsesDeviceAddresses;
        public BindingLayout Layout = null!;
        public uint UserDataBase;

        // The read-only buffers: index, guest address, requested footprint and the range bound.
        public int ReadCount;
        public int[] ReadIndex = new int[8];
        public ulong[] ReadAddress = new ulong[8];
        public ulong[] ReadRequested = new ulong[8];
        public MetalGuestRange[] ReadRange = new MetalGuestRange[8];

        // The written buffers (dispatch): index and the range bound.
        public int WrittenCount;
        public int[] WrittenIndex = new int[4];

        public void AddRead(int index, ulong address, ulong requested, in MetalGuestRange range)
        {
            if (ReadCount == ReadIndex.Length)
            {
                Array.Resize(ref ReadIndex, ReadCount * 2);
                Array.Resize(ref ReadAddress, ReadCount * 2);
                Array.Resize(ref ReadRequested, ReadCount * 2);
                Array.Resize(ref ReadRange, ReadCount * 2);
            }

            ReadIndex[ReadCount] = index;
            ReadAddress[ReadCount] = address;
            ReadRequested[ReadCount] = requested;
            ReadRange[ReadCount] = range;
            ReadCount++;
        }

        public void AddWritten(int index)
        {
            if (WrittenCount == WrittenIndex.Length)
            {
                Array.Resize(ref WrittenIndex, WrittenCount * 2);
            }

            WrittenIndex[WrittenCount] = index;
            WrittenCount++;
        }
    }

    private sealed class ReplayCapture
    {
        public bool Sealed;
        public long Commit = -1;
        public readonly ReplayStage[] Stages = [new(), new()];
        public int StageCount;
        public nint[] Used = new nint[16];
        public nuint[] UsedUsage = new nuint[16];
        public int UsedCount;
        public ulong Serial;
        public long PassSerial;

        // Draws: the vertex slots (obtain index or -1 for the null buffer, offset past the obtained range's start), the
        // obtains they come from with the size each was asked for before its clamp, and the slots as bound.
        public int VertexCount;
        public readonly int[] VertexObtain = new int[MaxVertexBufferSlots];
        public readonly ulong[] VertexDelta = new ulong[MaxVertexBufferSlots];
        public readonly nint[] VertexBuffers = new nint[MaxVertexBufferSlots];
        public readonly ulong[] VertexOffsets = new ulong[MaxVertexBufferSlots];
        public int ObtainCount;
        public readonly ReplayObtain[] Obtains = new ReplayObtain[MaxVertexBufferSlots];
        public readonly ulong[] ObtainRequested = new ulong[MaxVertexBufferSlots];
    }

    private readonly ReplayCapture _drawReplay = new();
    private readonly ReplayCapture _dispatchReplay = new();

    // Why a sealed item was not captured, and why a replay was refused (the [METAL-V2][TBDR] line's replay_host=...).
    private enum ReplayRefusal
    {
        Commit, Stages, Movie, StorageImage, Unstamped, WideImages, Bindless, DeviceAddresses, Written, ArgumentField,
        Traces, MemoOff, EpochMoved, VertexMap, Labels, ThreadLimits, NotInPlace, Bindings, Count,
    }

    private enum ReplayMiss
    {
        Off, Unsealed, Serial, Scope, Stages, Images, Buffers, Vertex, Patch, Count,
    }

    private static readonly string[] ReplayRefusalNames =
    [
        "commit", "stages", "movie", "storage_image", "unstamped", "wide_images", "bindless", "device_addresses", "written",
        "argument_field", "traces", "memo_off", "epoch_moved", "vertex_map", "labels", "thread_limits", "not_in_place", "bindings",
    ];

    private static readonly string[] ReplayMissNames = ["off", "unsealed", "serial", "scope", "stages", "images", "buffers", "vertex", "patch"];

    private readonly long[] _drawCaptureRefusals = new long[(int)ReplayRefusal.Count];
    private readonly long[] _dispatchCaptureRefusals = new long[(int)ReplayRefusal.Count];
    private readonly long[] _drawReplayMisses = new long[(int)ReplayMiss.Count];
    private readonly long[] _dispatchReplayMisses = new long[(int)ReplayMiss.Count];
    private long _drawReplaysServed;
    private long _dispatchReplaysServed;
    private long _drawCaptures;
    private long _dispatchCaptures;
    private long _dispatchReplaysRebound;

    /// <summary>Draws and dispatches the host replayed, for tests.</summary>
    internal (long Draws, long Dispatches, long DispatchesRebound) ReplayCounts =>
        (_drawReplaysServed, _dispatchReplaysServed, _dispatchReplaysRebound);

    /// <summary>A refusal count by name (capture refusals and replay misses of draws or dispatches), for tests.</summary>
    internal long ReplayCount(bool dispatch, string name)
    {
        var refusal = Array.IndexOf(ReplayRefusalNames, name);
        if (refusal >= 0)
        {
            return (dispatch ? _dispatchCaptureRefusals : _drawCaptureRefusals)[refusal];
        }

        var miss = Array.IndexOf(ReplayMissNames, name);
        return miss >= 0
            ? (dispatch ? _dispatchReplayMisses : _drawReplayMisses)[miss]
            : throw new ArgumentException($"No replay count is named {name}.", nameof(name));
    }

    private static string FormatReplayCounts(long[] counts, string[] names)
    {
        var parts = new List<string>();
        for (var index = 0; index < counts.Length; index++)
        {
            if (counts[index] != 0)
            {
                parts.Add(FormattableString.Invariant($"{names[index]}:{counts[index]}"));
                counts[index] = 0;
            }
        }

        return parts.Count == 0 ? "none" : string.Join(",", parts);
    }

    // The [METAL-V2][TBDR] line's part: what the host replayed, captured, refused to capture and refused to replay.
    private string TakeReplayHostReport()
    {
        var text = string.Create(
            System.Globalization.CultureInfo.InvariantCulture,
            $"replay_host=(draws={_drawReplaysServed},captured={_drawCaptures},capture_refused=[{FormatReplayCounts(_drawCaptureRefusals, ReplayRefusalNames)}],miss=[{FormatReplayCounts(_drawReplayMisses, ReplayMissNames)}]) dispatch_replay_host=(dispatches={_dispatchReplaysServed},rebound={_dispatchReplaysRebound},captured={_dispatchCaptures},capture_refused=[{FormatReplayCounts(_dispatchCaptureRefusals, ReplayRefusalNames)}],miss=[{FormatReplayCounts(_dispatchReplayMisses, ReplayMissNames)}])");
        _drawReplaysServed = _drawCaptures = _dispatchReplaysServed = _dispatchCaptures = _dispatchReplaysRebound = 0;
        return text;
    }

    public long CurrentItemSequence =>
        (DrawReplayEnabled || DispatchReplayEnabled) && UsesCommandThread && _commandThread is not null ? _replayItemSequence : 0;

    /// <summary>
    /// Always 0. Upstream's buffer cache merges and unregisters buffer owners, retiring a binding made earlier in the
    /// same draw. A Metal binding is a range of a guest-memory window or a ring copy, and nothing retires one under an
    /// open submission: a window a move replaces waits in MetalGuestMemory._retired until the submission that used
    /// it completes (MetalGuestMemory.RetireReplaced, ReleaseRetired), and ring ranges are reused only after their
    /// command buffer completes. So no binding of a draw being prepared is ever bound again; that a binding made for
    /// an earlier item still stands is the replay's question, answered by the read-binding memo (above).
    /// </summary>
    public long BufferOwnerGeneration => 0;

    /// <summary>Nothing to bind again: see <see cref="BufferOwnerGeneration"/>, which never moves on Metal.</summary>
    public void RebindBuffers(IPreparedBindings prepared)
    {
    }

    // ---- Recording what the full path bound (hooks in Descriptors, Rendering and the host) ----

    private void NoteReplayBindStart(MetalStageBindings stage)
    {
        var stageIndex = StageIndexOf(stage.Program.Stage);
        _replayBindEpoch[stageIndex] = _imageCache.Epochs.Binding;
        _replayBindImageSet[stageIndex] = _imageCache.ImageSetVersion;
        _replayStamped[stageIndex] = 0;
        _replayClampCount = 0;
        _replayObtainCount = 0;
        _replayRecordsOverflowed = false;
    }

    private void NoteReplayStamped(MetalStageBindings stage, int index)
    {
        if (index < 64)
        {
            _replayStamped[StageIndexOf(stage.Program.Stage)] |= 1UL << index;
        }
    }

    private ulong NoteReplayClamp(ulong address, ulong requested, ulong size)
    {
        if (_replayClampCount < ReplayRecordLimit)
        {
            _replayClamps[_replayClampCount++] = new ReplayClamp(address, requested, size);
        }
        else
        {
            _replayRecordsOverflowed = true;
        }

        return size;
    }

    private BufferBinding NoteReplayObtain(ulong address, ulong size, bool written, BufferBinding binding)
    {
        if (_replayObtainCount < ReplayRecordLimit)
        {
            _replayObtains[_replayObtainCount++] = new ReplayObtain(address, size, written, binding);
        }
        else
        {
            _replayRecordsOverflowed = true;
        }

        return binding;
    }

    // At the end of CommitBindings: which stages it committed and whether their images settled.
    private void CaptureBindingReplay(GpuBindPoint bindPoint, ReadOnlySpan<IPreparedBindings> stages, int gpuLimitsPushDword)
    {
        _replayCommits++;
        _replayCommitPoint = bindPoint;
        _replayCommitStageCount = stages.Length <= 2 ? stages.Length : 0;
        _replayCommitLimitsDword = gpuLimitsPushDword;
        if (_replayCommitStageCount == 0)
        {
            return;
        }

        var epoch = _imageCache.Epochs.Binding;
        var imageSet = _imageCache.ImageSetVersion;
        var refreshes = _imageCache.HasDeferredRefreshes;
        for (var index = 0; index < _replayCommitStageCount; index++)
        {
            var stage = (MetalStageBindings)stages[index];
            var stageIndex = StageIndexOf(stage.Program.Stage);
            _replayCommitStages[index] = stage;
            _replayCommitSettled[index] = !refreshes && _replayBindEpoch[stageIndex] == epoch && _replayBindImageSet[stageIndex] == imageSet;
        }
    }

    private bool RefuseCapture(ReplayCapture capture, ReplayRefusal reason)
    {
        capture.Sealed = false;
        capture.Commit = -1;
        (ReferenceEquals(capture, _dispatchReplay) ? _dispatchCaptureRefusals : _drawCaptureRefusals)[(int)reason]++;
        return false;
    }

    // Captures the stages CommitBindings committed last, for the sealed item; false (counted) when they cannot be replayed.
    private bool CaptureStages(ReplayCapture capture, bool compute)
    {
        if (_replayCommitPoint != (compute ? GpuBindPoint.Compute : GpuBindPoint.Graphics) || _replayCommitStageCount == 0)
        {
            return RefuseCapture(capture, ReplayRefusal.Commit);
        }

        if (!ReadBindingTracesOff)
        {
            return RefuseCapture(capture, ReplayRefusal.Traces);
        }

        if (!ReadBindingMemoEnabled)
        {
            return RefuseCapture(capture, ReplayRefusal.MemoOff);
        }

        if (_bindlessDeclarations.Count != 0)
        {
            return RefuseCapture(capture, ReplayRefusal.Bindless);
        }

        if (compute && (_replayCommitLimitsDword >= 0 || _replayCommitStageCount != 1))
        {
            return RefuseCapture(capture, ReplayRefusal.ThreadLimits);
        }

        if (compute && _labelStagedPending.Count != 0)
        {
            return RefuseCapture(capture, ReplayRefusal.Labels);
        }

        for (var position = 0; position < _replayCommitStageCount; position++)
        {
            var bindings = _replayCommitStages[position]!;
            var stageIndex = StageIndexOf(bindings.Program.Stage);
            if (compute != (stageIndex == StageCompute))
            {
                return RefuseCapture(capture, ReplayRefusal.Stages);
            }

            if (!_replayCommitSettled[position])
            {
                return RefuseCapture(capture, ReplayRefusal.EpochMoved);
            }

            if (CaptureStage(capture.Stages[position], bindings, stageIndex, compute) is { } refusal)
            {
                return RefuseCapture(capture, refusal);
            }
        }

        capture.StageCount = _replayCommitStageCount;
        if (capture.Used.Length < _usedResourceCount)
        {
            capture.Used = new nint[_usedResourceCount];
            capture.UsedUsage = new nuint[_usedResourceCount];
        }

        _usedResources.AsSpan(0, _usedResourceCount).CopyTo(capture.Used);
        _usedResourceUsage.AsSpan(0, _usedResourceCount).CopyTo(capture.UsedUsage);
        capture.UsedCount = _usedResourceCount;
        capture.Commit = _replayCommits;
        return true;
    }

    private ReplayRefusal? CaptureStage(ReplayStage captured, MetalStageBindings stage, int stageIndex, bool compute)
    {
        var program = stage.Program;
        var layout = program.Bindings!;
        var info = program.Resources!.Info;
        var snapshot = stage.StageResources.Resources;
        if (stage.MovieLuma >= 0 || stage.MovieChroma >= 0)
        {
            return ReplayRefusal.Movie;
        }

        if (layout.BindlessImageTableCount != 0 || snapshot.BindlessImages.Length != 0)
        {
            return ReplayRefusal.Bindless;
        }

        if (program.UsesDeviceAddresses || info.WritesDeviceAddresses || snapshot.DeviceAddressRanges.Length != 0)
        {
            return ReplayRefusal.DeviceAddresses;
        }

        if (stage.ImageCount > 64)
        {
            return ReplayRefusal.WideImages;
        }

        if (compute && HasGpuThreadLimits(stage))
        {
            return ReplayRefusal.ThreadLimits;
        }

        // Fields whose bytes stand for the whole submission: the GDS buffer and its size, the LOD statistics bank (one
        // resident buffer, made once), the shader-data size, the zero fault word, the step-report slot (a program keeps
        // it for good) and the pair-lock table. Anything else (the device-address table, the flattened table, shader
        // data uploaded out of line) is written per item.
        foreach (var field in ArgumentPlanOf(layout).Fields)
        {
            var stands = field.FieldKind switch
            {
                MslArgumentFieldKind.Pointer => field.Kind is DescriptorBindingKind.GlobalDataShare or DescriptorBindingKind.MipStatistics,
                MslArgumentFieldKind.Word => field.Kind is DescriptorBindingKind.GlobalDataShare or
                    DescriptorBindingKind.ShaderData or DescriptorBindingKind.FaultBuffer,
                MslArgumentFieldKind.BindlessImageTables => false,
                _ => true,
            };
            if (!stands)
            {
                return ReplayRefusal.ArgumentField;
            }
        }

        var stamped = _replayStamped[stageIndex];
        for (var index = 0; index < stage.ImageCount; index++)
        {
            if (stage.IsStorage[index])
            {
                return ReplayRefusal.StorageImage;
            }

            if (stage.MemoHits[index] is null && (stamped & (1UL << index)) == 0)
            {
                return ReplayRefusal.Unstamped;
            }
        }

        captured.ReadCount = 0;
        captured.WrittenCount = 0;
        for (var index = 0; index < stage.BufferCount; index++)
        {
            var range = stage.Buffers[index];
            if (!range.IsValid)
            {
                continue;
            }

            if (info.Buffers[index].Written)
            {
                if (!compute)
                {
                    return ReplayRefusal.Written;
                }

                if (!_guestMemory.IsInPlace(in range))
                {
                    return ReplayRefusal.NotInPlace;
                }

                captured.AddWritten(index);
                continue;
            }

            if (!_guestMemory.IsInPlace(in range))
            {
                return ReplayRefusal.NotInPlace;
            }

            var descriptor = BufferDescriptorWords.From(snapshot.Buffers[index]);
            if (descriptor.Footprint() is not { } requested)
            {
                return ReplayRefusal.Bindings;
            }

            captured.AddRead(index, descriptor.Address, requested, in range);
        }

        captured.Bindings = stage;
        captured.StageIndex = stageIndex;
        captured.Prepares = _stagePrepares[stageIndex];
        captured.BindingEpoch = _replayBindEpoch[stageIndex];
        captured.ImageSet = _replayBindImageSet[stageIndex];
        captured.ArgumentBuffer = _stageArgumentBuffers[stageIndex];
        captured.ArgumentOffset = _stageArgumentOffsets[stageIndex];
        captured.ArgumentValid = _stageArgumentValid[stageIndex];
        captured.UsesDeviceAddresses = _stageUsesDeviceAddresses[stageIndex];
        captured.Layout = layout;
        captured.UserDataBase = program.UserDataBase;
        return null;
    }

    // The vertex streams BindVertexBuffers bound, each mapped to the one obtained range it lies in.
    private bool CaptureVertexStreams(ReplayCapture capture)
    {
        if (_replayRecordsOverflowed)
        {
            return RefuseCapture(capture, ReplayRefusal.VertexMap);
        }

        var nullBuffer = _guestMemory.NullBuffer;
        capture.ObtainCount = 0;
        capture.VertexCount = _vertexBufferCount;
        for (var slot = 0; slot < _vertexBufferCount; slot++)
        {
            var buffer = _vertexBuffers[slot];
            var offset = _vertexOffsets[slot];
            capture.VertexBuffers[slot] = buffer;
            capture.VertexOffsets[slot] = offset;
            if (buffer == nullBuffer && offset == 0)
            {
                capture.VertexObtain[slot] = -1;
                continue;
            }

            var found = -1;
            for (var index = 0; index < _replayObtainCount; index++)
            {
                ref readonly var obtain = ref _replayObtains[index];
                if ((nint)obtain.Binding.Handle != buffer || offset < obtain.Binding.Offset || offset - obtain.Binding.Offset >= obtain.Size)
                {
                    continue;
                }

                if (found >= 0 || obtain.Written)
                {
                    // Two obtained ranges hold the stream, or it is written: which one AcquireVertexBuffers took is not known.
                    return RefuseCapture(capture, ReplayRefusal.VertexMap);
                }

                found = index;
            }

            if (found < 0)
            {
                return RefuseCapture(capture, ReplayRefusal.VertexMap);
            }

            var obtained = _replayObtains[found];
            var position = Array.IndexOf(capture.Obtains, obtained, 0, capture.ObtainCount);
            if (position < 0)
            {
                var requested = 0UL;
                for (var index = 0; index < _replayClampCount; index++)
                {
                    if (_replayClamps[index].Address == obtained.Address && _replayClamps[index].Size == obtained.Size)
                    {
                        requested = _replayClamps[index].Requested;
                        break;
                    }
                }

                if (requested == 0 || capture.ObtainCount == capture.Obtains.Length)
                {
                    return RefuseCapture(capture, ReplayRefusal.VertexMap);
                }

                position = capture.ObtainCount++;
                capture.Obtains[position] = obtained;
                capture.ObtainRequested[position] = requested;
            }

            capture.VertexObtain[slot] = position;
            capture.VertexDelta[slot] = offset - obtained.Binding.Offset;
        }

        return true;
    }

    // ---- The IRenderHost members ----

    public void SealDrawReplay()
    {
        var capture = _drawReplay;
        capture.Sealed = false;
        if (!DrawReplayEnabled || !_renderingActive)
        {
            return;
        }

        // After a replay nothing was committed since the capture, which still stands; after a full draw its
        // CommitBindings is the last one.
        if (capture.Commit != _replayCommits)
        {
            if (!CaptureStages(capture, compute: false) || !CaptureVertexStreams(capture))
            {
                return;
            }

            _drawCaptures++;
        }

        capture.Serial = _submissionSerial;
        capture.PassSerial = _renderPassSerial;
        capture.Sealed = true;
    }

    public void SealDispatchReplay()
    {
        var capture = _dispatchReplay;
        capture.Sealed = false;
        if (!DispatchReplayEnabled)
        {
            return;
        }

        if (capture.Commit != _replayCommits)
        {
            if (!CaptureStages(capture, compute: true))
            {
                return;
            }

            _dispatchCaptures++;
        }

        capture.Serial = _submissionSerial;
        capture.Sealed = true;
    }

    private bool MissDraw(ReplayMiss miss)
    {
        _drawReplayMisses[(int)miss]++;
        return false;
    }

    private bool MissDispatch(ReplayMiss miss)
    {
        _dispatchReplayMisses[(int)miss]++;
        return false;
    }

    // What every captured stage needs before anything is bound: the stage object untouched, the images as settled.
    private bool StagesStand(ReplayCapture capture, out ReplayMiss miss)
    {
        var epoch = _imageCache.Epochs.Binding;
        var imageSet = _imageCache.ImageSetVersion;
        for (var position = 0; position < capture.StageCount; position++)
        {
            var stage = capture.Stages[position];
            if (_stagePrepares[stage.StageIndex] != stage.Prepares)
            {
                miss = ReplayMiss.Stages;
                return false;
            }

            if (stage.BindingEpoch != epoch || stage.ImageSet != imageSet || _imageCache.HasDeferredRefreshes)
            {
                miss = ReplayMiss.Images;
                return false;
            }
        }

        miss = ReplayMiss.Count;
        return true;
    }

    // Each read-only buffer still resolves, by the read-binding memo, to the range it was bound to.
    private bool ReadBuffersStand(ReplayCapture capture, bool compute)
    {
        for (var position = 0; position < capture.StageCount; position++)
        {
            var stage = capture.Stages[position];
            for (var read = 0; read < stage.ReadCount; read++)
            {
                var address = stage.ReadAddress[read];
                var requested = stage.ReadRequested[read];
                if (!TryReadBindingMemo(address, requested, compute, out var range) || range != stage.ReadRange[read])
                {
                    return false;
                }

                if (ReadBindingMemoVerify)
                {
                    VerifyReadBinding(address, requested, compute, MetalWritebackCause.StorageDescriptor, range);
                }
            }
        }

        return true;
    }

    // The user-data registers of a push-data stage, from the item's own user data; false when one lies outside it.
    private static bool PatchUserData(Span<uint> shaderData, BindingLayout layout, uint userDataBase, ReadOnlySpan<uint> userData)
    {
        var registers = layout.UserDataRegisters;
        for (var index = 0; index < registers.Count; index++)
        {
            var userIndex = (long)registers[index] - userDataBase;
            if (userIndex < 0 || userIndex >= userData.Length)
            {
                return false;
            }

            shaderData[index] = userData[(int)userIndex];
        }

        return true;
    }

    private static bool CanPatchUserData(BindingLayout layout, uint userDataBase, int userDataLength)
    {
        var registers = layout.UserDataRegisters;
        for (var index = 0; index < registers.Count; index++)
        {
            var userIndex = (long)registers[index] - userDataBase;
            if (userIndex < 0 || userIndex >= userDataLength)
            {
                return false;
            }
        }

        return true;
    }

    // Puts a captured stage back as CommitBindings left it, BindResources' per-stage bookkeeping included.
    private void RestoreStage(ReplayStage stage, bool compute)
    {
        var bindings = stage.Bindings;
        var index = stage.StageIndex;
        _stageArgumentBuffers[index] = stage.ArgumentBuffer;
        _stageArgumentOffsets[index] = stage.ArgumentOffset;
        _stageArgumentValid[index] = stage.ArgumentValid;
        _stageUsesDeviceAddresses[index] = stage.UsesDeviceAddresses;
        _preparedStageMask |= 1 << index;
        BeginLabelStagedStage(compute ? bindings : null);
        _writtenViewRanges.Clear();
        _readOnlyBufferBinds += stage.ReadCount;
        var info = bindings.Program.Resources!.Info;
        BindDeviceAddressRanges(bindings, info, bindings.StageResources.Resources);
        bindings.Committed = true;
    }

    private void RestoreUsedResources(ReplayCapture capture)
    {
        for (var index = 0; index < capture.UsedCount; index++)
        {
            NoteUsedResource(capture.Used[index], capture.UsedUsage[index]);
        }
    }

    // The committed stages' shader data as one push-data block, as CommitBindings builds it.
    private void PublishReplayPushData(ReplayCapture capture)
    {
        Span<uint> pushData = stackalloc uint[(int)PushData.DwordCount];
        pushData.Clear();
        var hasPushData = false;
        for (var position = 0; position < capture.StageCount; position++)
        {
            var stage = capture.Stages[position];
            var layout = stage.Layout;
            if (layout.UsesPushData)
            {
                var shaderDwords = (int)layout.ShaderDataDwordCount;
                stage.Bindings.ShaderData.AsSpan(0, shaderDwords).CopyTo(pushData[(int)layout.PushDataStartDword..]);
                hasPushData = true;
            }
        }

        if (hasPushData)
        {
            PublishPushData(pushData, gpuLimitsPushDword: -1);
        }
    }

    public bool TryReplayDraw(in PipelineHandle pipeline, ReadOnlySpan<uint> vertexUserData, ReadOnlySpan<uint> pixelUserData)
    {
        var capture = _drawReplay;
        if (!DrawReplayEnabled)
        {
            return MissDraw(ReplayMiss.Off);
        }

        ReplayProbe?.Invoke(false);

        if (!capture.Sealed)
        {
            return MissDraw(ReplayMiss.Unsealed);
        }

        capture.Sealed = false;
        if (_commandBuffer == 0 || capture.Serial != _submissionSerial)
        {
            return MissDraw(ReplayMiss.Serial);
        }

        if (capture.PassSerial != _renderPassSerial || !_renderingActive)
        {
            return MissDraw(ReplayMiss.Scope);
        }

        if (!StagesStand(capture, out var miss))
        {
            return MissDraw(miss);
        }

        for (var position = 0; position < capture.StageCount; position++)
        {
            var stage = capture.Stages[position];
            var userData = stage.StageIndex == StageFragment ? pixelUserData : vertexUserData;
            if (stage.Layout.UsesPushData && !CanPatchUserData(stage.Layout, stage.UserDataBase, userData.Length))
            {
                return MissDraw(ReplayMiss.Patch);
            }
        }

        // The vertex streams' clamps are pure; the memo lookups below change nothing but their counters.
        for (var index = 0; index < capture.ObtainCount; index++)
        {
            ref readonly var obtain = ref capture.Obtains[index];
            if (ClampMappedSize(obtain.Address, capture.ObtainRequested[index]) != obtain.Size)
            {
                return MissDraw(ReplayMiss.Vertex);
            }
        }

        if (!ReadBuffersStand(capture, compute: false))
        {
            return MissDraw(ReplayMiss.Buffers);
        }

        for (var index = 0; index < capture.ObtainCount; index++)
        {
            ref readonly var obtain = ref capture.Obtains[index];
            if (!TryReadBindingMemo(obtain.Address, obtain.Size, computeStage: false, out var range) ||
                (ulong)range.Buffer != obtain.Binding.Handle || range.Offset != obtain.Binding.Offset)
            {
                return MissDraw(ReplayMiss.Vertex);
            }

            if (ReadBindingMemoVerify)
            {
                VerifyReadBinding(obtain.Address, obtain.Size, computeStage: false, MetalWritebackCause.BufferBinding, range);
            }
        }

        // Everything stands: bind what the sealed draw bound, with this draw's user data.
        _ = RequirePipelineEntry(in pipeline);
        for (var position = 0; position < capture.StageCount; position++)
        {
            var stage = capture.Stages[position];
            if (stage.Layout.UsesPushData)
            {
                _ = PatchUserData(stage.Bindings.ShaderData, stage.Layout, stage.UserDataBase,
                    stage.StageIndex == StageFragment ? pixelUserData : vertexUserData);
            }

            RestoreStage(stage, compute: false);
        }

        RestoreUsedResources(capture);
        Span<IPreparedBindings> stages = capture.StageCount == 1
            ? [capture.Stages[0].Bindings]
            : [capture.Stages[0].Bindings, capture.Stages[1].Bindings];
        NoteJoinStages(stages);
        NoteBufferWordStages(stages);
        PublishReplayPushData(capture);
        capture.VertexBuffers.AsSpan(0, capture.VertexCount).CopyTo(_vertexBuffers);
        capture.VertexOffsets.AsSpan(0, capture.VertexCount).CopyTo(_vertexOffsets);
        _vertexBufferCount = capture.VertexCount;
        _drawReplaysServed++;
        return true;
    }

    public bool TryReplayDispatch(in PipelineHandle pipeline, in ShaderStageResources stage)
    {
        var capture = _dispatchReplay;
        if (!DispatchReplayEnabled)
        {
            return MissDispatch(ReplayMiss.Off);
        }

        ReplayProbe?.Invoke(true);

        if (!capture.Sealed)
        {
            return MissDispatch(ReplayMiss.Unsealed);
        }

        capture.Sealed = false;
        if (_commandBuffer == 0 || capture.Serial != _submissionSerial)
        {
            return MissDispatch(ReplayMiss.Serial);
        }

        if (!StagesStand(capture, out var miss))
        {
            return MissDispatch(miss);
        }

        var captured = capture.Stages[0];
        var bindings = captured.Bindings;
        if (!ReferenceEquals(stage.Program, bindings.Program) || stage.ShaderBase != bindings.StageResources.ShaderBase)
        {
            return MissDispatch(ReplayMiss.Stages);
        }

        var program = bindings.Program;
        var layout = captured.Layout;
        if (!CanPatchUserData(layout, program.UserDataBase, stage.Resources.UserData.Length))
        {
            return MissDispatch(ReplayMiss.Patch);
        }

        if (!ReadBuffersStand(capture, compute: true))
        {
            return MissDispatch(ReplayMiss.Buffers);
        }

        // Everything stands. The shader data is this dispatch's, built as PrepareBindings builds it.
        var entry = RequirePipelineEntry(in pipeline);
        var info = program.Resources!.Info;
        bindings.StageResources = stage;
        var shaderDwords = (int)layout.ShaderDataDwordCount;
        Array.Clear(bindings.ShaderData, 0, shaderDwords);
        if (!WriteStageShaderData(bindings.ShaderData, program, layout, info, stage.Resources, stage.ShaderBase))
        {
            throw UserRegisterOutsideUserData(program, layout, stage.Resources);
        }

        stage.WriteDispatchThreadLimits(bindings.ShaderData.AsSpan(0, shaderDwords));
        stage.WriteCallData(bindings.ShaderData.AsSpan(0, shaderDwords));
        _threadLimitsTarget = default;
        RestoreStage(captured, compute: true);

        // A dispatch that writes takes BindResources' buffer loop again, every buffer in order, after every refusal:
        // the invalidation and write record of a written buffer belong to this dispatch, and a read after it is
        // looked up after it, as the full path looks it up. A range or length other than the captured one rewrites
        // the argument buffer, and the next seal captures the stage again.
        RestoreUsedResources(capture);
        if (captured.WrittenCount != 0)
        {
            var count = bindings.BufferCount;
            var before = ArrayPool<MetalGuestRange>.Shared.Rent(count);
            var lengths = ArrayPool<uint>.Shared.Rent(count);
            try
            {
                bindings.Buffers.AsSpan(0, count).CopyTo(before);
                bindings.BufferLengths.AsSpan(0, count).CopyTo(lengths);
                _readOnlyBufferBinds -= captured.ReadCount;
                for (var index = 0; index < count; index++)
                {
                    BindStageStorageBuffer(bindings, index, info, stage.Resources);
                }

                var rebound = false;
                for (var index = 0; index < count && !rebound; index++)
                {
                    rebound = bindings.Buffers[index] != before[index] || bindings.BufferLengths[index] != lengths[index];
                }

                if (rebound)
                {
                    WriteArgumentBuffer(bindings, StageCompute);
                    capture.Commit = -1;
                    _dispatchReplaysRebound++;
                }
            }
            finally
            {
                ArrayPool<MetalGuestRange>.Shared.Return(before);
                ArrayPool<uint>.Shared.Return(lengths);
            }
        }

        NoteBufferWordComputeStage(entry, [bindings]);
        PublishReplayPushData(capture);
        _dispatchReplaysServed++;
        return true;
    }
}
