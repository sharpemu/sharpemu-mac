// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Libs.Audio;

/// <summary>
/// The typed batch-job helpers of libSceAjm. In the SDK each one is a control
/// or run job with a fixed flag set and sideband size; here they reuse the
/// eager run path, so their sidebands are filled at enqueue time like every
/// other job.
/// </summary>
public static partial class AjmExports
{
    // SCE_AJM_FLAG_CONTROL_RESET / SCE_AJM_FLAG_CONTROL_INITIALIZE.
    private const ulong AjmJobControlFlagReset = 1UL << 13;
    private const ulong AjmJobControlFlagInitialize = 1UL << 14;
    private const ulong AjmJobControlSupportedFlags =
        AjmJobControlFlagReset | AjmJobControlFlagInitialize | AjmJobSidebandFlagGaplessDecode;

    // SceAjmDecAt9InitializeParameters { uint8_t configData[4]; uint32_t reserved; }.
    private const int Atrac9InitializeParametersBytes = 8;

    // SceAjmSidebandResampleInfo { float ratio; int32_t samples; uint32_t reserved[8]; }.
    private const int AjmSidebandResampleInfoBytes = 40;

    private const int OrbisAjmErrorUnknown = unchecked((int)0x80930001);

    private static readonly ConcurrentDictionary<int, ulong> ErrorStrings = new();

    [SysAbiExport(
        Nid = "7FZsbyVRM4U",
        ExportName = "sceAjmBatchJobControl",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobControl(CpuContext ctx) => AjmBatchJobControlCore(ctx);

    // The BufferRa variant appends the caller's return address after the seven
    // shared arguments; it is only used for error attribution.
    [SysAbiExport(
        Nid = "dmDybN--Fn8",
        ExportName = "sceAjmBatchJobControlBufferRa",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobControlBufferRa(CpuContext ctx) => AjmBatchJobControlCore(ctx);

    /// <summary>
    /// sceAjmBatchJobControl(info, instance, flags, sidebandInput, sidebandInputSize,
    /// sidebandOutput, sidebandOutputSize). The sideband input carries an
    /// SceAjmSidebandGaplessDecode when SIDEBAND_GAPLESS_DECODE is set, followed by
    /// the codec's initialize parameters when CONTROL_INITIALIZE is set. RESET
    /// clears the decoder context first.
    /// </summary>
    private static int AjmBatchJobControlCore(CpuContext ctx)
    {
        var infoAddress = ctx[CpuRegister.Rdi];
        var instanceId = unchecked((uint)ctx[CpuRegister.Rsi]);
        var flags = ctx[CpuRegister.Rdx];
        var inputAddress = ctx[CpuRegister.Rcx];
        var inputSize = ctx[CpuRegister.R8];
        var outputAddress = ctx[CpuRegister.R9];
        var outputSize = ReadStackArg64(ctx, 0);

        if (inputSize != 0 && inputAddress == 0)
        {
            return ctx.SetReturn(OrbisAjmErrorInvalidParameter);
        }

        if (!TryAppendBatchJob(ctx, infoAddress, AjmJobControlSize))
        {
            return ctx.SetReturn(OrbisAjmErrorJobCreation);
        }

        var status = ExecuteControl(ctx, instanceId, flags, inputAddress, inputSize);
        if (outputSize >= AjmSidebandResultBytes)
        {
            WriteBasicResult(ctx, outputAddress, status);
        }

        if (TraceEnabled)
        {
            Trace(
                $"batch_job_control instance=0x{instanceId:X8} flags=0x{flags:X} " +
                $"in=0x{inputAddress:X16}+0x{inputSize:X} out=0x{outputAddress:X16}+0x{outputSize:X} " +
                $"status=0x{status:X8}");
        }
        return ctx.SetReturn(0);
    }

    private static int ExecuteControl(
        CpuContext ctx,
        uint instanceId,
        ulong flags,
        ulong inputAddress,
        ulong inputSize)
    {
        if (flags == 0 ||
            (flags & ~AjmJobControlSupportedFlags) != 0 ||
            !TryGetInstance(instanceId, out var instance))
        {
            return Atrac9DecodeState.ResultInvalidParameter;
        }

        var gaplessBytes = (flags & AjmJobSidebandFlagGaplessDecode) != 0
            ? AjmSidebandGaplessDecodeBytes
            : 0;
        var initializeBytes = (flags & AjmJobControlFlagInitialize) != 0 && instance.Codec == Atrac9CodecType
            ? Atrac9InitializeParametersBytes
            : 0;
        if (inputSize < (ulong)(gaplessBytes + initializeBytes))
        {
            return Atrac9DecodeState.ResultInvalidParameter;
        }

        Span<byte> input = stackalloc byte[AjmSidebandGaplessDecodeBytes + Atrac9InitializeParametersBytes];
        input = input[..(gaplessBytes + initializeBytes)];
        if (input.Length != 0 && !ctx.Memory.TryRead(inputAddress, input))
        {
            return Atrac9DecodeState.ResultInvalidParameter;
        }

        var reset = (flags & AjmJobControlFlagReset) != 0;
        if (reset)
        {
            instance.Atrac9?.Reset();
            instance.Mp3?.Reset();
        }

        var status = 0;
        if (initializeBytes != 0 &&
            !instance.Atrac9!.TryInitialize(input.Slice(gaplessBytes, 4)))
        {
            status = Atrac9DecodeState.ResultInvalidParameter;
        }

        if (gaplessBytes != 0)
        {
            instance.Atrac9?.SetGapless(
                BinaryPrimitives.ReadUInt32LittleEndian(input),
                BinaryPrimitives.ReadUInt16LittleEndian(input[4..]),
                reset);
        }

        return status;
    }

    [SysAbiExport(
        Nid = "SJ3i0DXP8vg",
        ExportName = "sceAjmBatchJobDecodeSplit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobDecodeSplit(CpuContext ctx) =>
        // (info, instance, inputBuffers, inputCount, outputBuffers, outputCount, result):
        // a split run job with the stream sideband (Result + Stream).
        AppendRunJob(
            ctx,
            infoAddress: ctx[CpuRegister.Rdi],
            instanceId: unchecked((uint)ctx[CpuRegister.Rsi]),
            flags: AjmJobSidebandFlagStream,
            inputAddress: ctx[CpuRegister.Rdx],
            inputCountOrSize: ctx[CpuRegister.Rcx],
            outputAddress: ctx[CpuRegister.R8],
            outputCountOrSize: ctx[CpuRegister.R9],
            sidebandAddress: ReadStackArg64(ctx, 0),
            sidebandSize: AjmSidebandResultBytes + AjmSidebandStreamBytes,
            split: true);

    [SysAbiExport(
        Nid = "TBWW4aPfWcA",
        ExportName = "sceAjmBatchJobGetInfo",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobGetInfo(CpuContext ctx) =>
        AppendStateRunJob(
            ctx,
            AjmJobSidebandFlagFormat,
            ctx[CpuRegister.Rdx],
            AjmSidebandResultBytes + AjmSidebandFormatBytes);

    [SysAbiExport(
        Nid = "Esr9db8S1S0",
        ExportName = "sceAjmBatchJobGetGaplessDecode",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobGetGaplessDecode(CpuContext ctx) =>
        AppendStateRunJob(
            ctx,
            AjmJobSidebandFlagGaplessDecode,
            ctx[CpuRegister.Rdx],
            AjmSidebandResultBytes + AjmSidebandGaplessDecodeBytes);

    [SysAbiExport(
        Nid = "uSrXaxT+oPQ",
        ExportName = "sceAjmBatchJobGetCodecInfo",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobGetCodecInfo(CpuContext ctx) =>
        AppendStateRunJob(
            ctx,
            AjmJobRunFlagGetCodecInfo,
            ctx[CpuRegister.Rdx],
            ctx[CpuRegister.Rcx]);

    private static int AppendStateRunJob(
        CpuContext ctx,
        ulong flags,
        ulong sidebandAddress,
        ulong sidebandSize) =>
        AppendRunJob(
            ctx,
            infoAddress: ctx[CpuRegister.Rdi],
            instanceId: unchecked((uint)ctx[CpuRegister.Rsi]),
            flags,
            inputAddress: 0,
            inputCountOrSize: 0,
            outputAddress: 0,
            outputCountOrSize: 0,
            sidebandAddress,
            sidebandSize,
            split: true);

    [SysAbiExport(
        Nid = "5ldnD16rYZw",
        ExportName = "sceAjmBatchJobSetResampleParametersEx",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobSetResampleParametersEx(CpuContext ctx)
    {
        // (info, instance, float ratioStart, float ratioChangePerSample, uint32_t flags, result).
        ctx.GetXmmRegister(0, out var startBits, out _);
        ctx.GetXmmRegister(1, out var changeBits, out _);
        var infoAddress = ctx[CpuRegister.Rdi];
        var instanceId = unchecked((uint)ctx[CpuRegister.Rsi]);
        var flags = unchecked((uint)ctx[CpuRegister.Rdx]);
        var resultAddress = ctx[CpuRegister.Rcx];
        var ratio = BitConverter.Int32BitsToSingle(unchecked((int)(uint)startBits));
        var change = BitConverter.Int32BitsToSingle(unchecked((int)(uint)changeBits));
        if (infoAddress == 0)
        {
            return ctx.SetReturn(OrbisAjmErrorInvalidParameter);
        }

        if (!TryAppendBatchJob(ctx, infoAddress, AjmJobSetResampleParametersSize))
        {
            return ctx.SetReturn(OrbisAjmErrorJobCreation);
        }

        // Same checks as sceAjmBatchJobSetResampleParameters. The Opus resampler holds a
        // constant ratio, so a per-sample ramp completes with INVALID_PARAMETER.
        const uint ignoreRatio = 1u << 3;
        var validRatio = (flags & ignoreRatio) != 0
            || (float.IsFinite(ratio) && ratio is >= 0 and <= 4 && change == 0.0f);
        var status = TryGetInstance(instanceId, out var instance) && validRatio
            ? 0
            : Atrac9DecodeState.ResultInvalidParameter;
        if (status == 0)
            instance.Opus?.SetResampleParameters(ratio, flags);
        WriteBasicResult(ctx, resultAddress, status);
        Trace($"batch_job_set_resample_parameters_ex instance=0x{instanceId:X8} ratio={ratio:R} change={change:R} flags=0x{flags:X8} status=0x{status:X8}");
        return ctx.SetReturn(0);
    }

    /// <summary>
    /// sceAjmBatchJobEncode(info, instance, pcmInput, pcmInputSize, bitstreamOutput,
    /// bitstreamOutputSize, result). Every instance here is a decoder (no encoder
    /// codec can be registered), so the job completes with
    /// AJM_RESULT_INVALID_PARAMETER and nothing consumed or produced.
    /// </summary>
    [SysAbiExport(
        Nid = "SlVIGK1Kl38",
        ExportName = "sceAjmBatchJobEncode",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmBatchJobEncode(CpuContext ctx)
    {
        var infoAddress = ctx[CpuRegister.Rdi];
        var instanceId = unchecked((uint)ctx[CpuRegister.Rsi]);
        var resultAddress = ReadStackArg64(ctx, 0);
        if (!TryAppendBatchJob(ctx, infoAddress, AjmJobRunSize))
        {
            return ctx.SetReturn(OrbisAjmErrorJobCreation);
        }

        if (resultAddress != 0)
        {
            // SceAjmEncodeResult: Result + Stream.
            Span<byte> sideband = stackalloc byte[AjmSidebandResultBytes + AjmSidebandStreamBytes];
            sideband.Clear();
            BinaryPrimitives.WriteInt32LittleEndian(sideband, Atrac9DecodeState.ResultInvalidParameter);
            _ = ctx.Memory.TryWrite(resultAddress, sideband);
        }

        Trace($"batch_job_encode instance=0x{instanceId:X8} unsupported");
        return ctx.SetReturn(0);
    }

    /// <summary>
    /// Returns a guest pointer to the symbolic name of an SCE_AJM_ERROR_* code
    /// (SCE_AJM_ERROR_UNKNOWN for any other value). The firmware's exact
    /// wording is not known; the names are the SDK's constant names.
    /// </summary>
    [SysAbiExport(
        Nid = "AxhcqVv5AYU",
        ExportName = "sceAjmStrError",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceAjm")]
    public static int AjmStrError(CpuContext ctx)
    {
        var code = unchecked((int)ctx[CpuRegister.Rdi]);
        var name = GetErrorName(code) ?? GetErrorName(OrbisAjmErrorUnknown)!;
        var key = GetErrorName(code) is null ? OrbisAjmErrorUnknown : code;
        var address = ErrorStrings.GetOrAdd(key, _ =>
        {
            var bytes = Encoding.ASCII.GetBytes(name + "\0");
            return KernelMemoryCompatExports.TryAllocateHleData(ctx, (ulong)bytes.Length, 16, out var allocated) &&
                   ctx.Memory.TryWrite(allocated, bytes)
                ? allocated
                : 0;
        });
        if (address == 0)
        {
            ErrorStrings.TryRemove(key, out _);
        }

        ctx[CpuRegister.Rax] = address;
        return 0;
    }

    internal static string? GetErrorName(int code) => unchecked((uint)code) switch
    {
        0x80930001 => "SCE_AJM_ERROR_UNKNOWN",
        0x80930002 => "SCE_AJM_ERROR_INVALID_CONTEXT",
        0x80930003 => "SCE_AJM_ERROR_INVALID_INSTANCE",
        0x80930004 => "SCE_AJM_ERROR_INVALID_BATCH",
        0x80930005 => "SCE_AJM_ERROR_INVALID_PARAMETER",
        0x80930006 => "SCE_AJM_ERROR_OUT_OF_MEMORY",
        0x80930007 => "SCE_AJM_ERROR_OUT_OF_RESOURCES",
        0x80930008 => "SCE_AJM_ERROR_CODEC_NOT_SUPPORTED",
        0x80930009 => "SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED",
        0x8093000A => "SCE_AJM_ERROR_CODEC_NOT_REGISTERED",
        0x8093000B => "SCE_AJM_ERROR_WRONG_REVISION_FLAG",
        0x8093000C => "SCE_AJM_ERROR_FLAG_NOT_SUPPORTED",
        0x8093000D => "SCE_AJM_ERROR_BUSY",
        0x8093000E => "SCE_AJM_ERROR_BAD_PRIORITY",
        0x8093000F => "SCE_AJM_ERROR_IN_PROGRESS",
        0x80930010 => "SCE_AJM_ERROR_RETRY",
        0x80930011 => "SCE_AJM_ERROR_MALFORMED_BATCH",
        0x80930012 => "SCE_AJM_ERROR_JOB_CREATION",
        0x80930013 => "SCE_AJM_ERROR_INVALID_OPCODE",
        0x80930014 => "SCE_AJM_ERROR_INVALID_ADDRESS",
        0x80930015 => "SCE_AJM_ERROR_CANCELLED",
        _ => null,
    };

    /// <summary>
    /// MP3 through the run interface: the AjmBuffer descriptors form one logical
    /// stream, gathered for the decoder and scattered back out.
    /// </summary>
    private static Atrac9DecodeResult DecodeMp3Scattered(
        CpuContext ctx,
        AjmInstanceState instance,
        List<AjmGuestBuffer> inputs,
        int inputLength,
        List<AjmGuestBuffer> outputs,
        int outputLength)
    {
        var mp3 = instance.Mp3!;
        var input = ArrayPool<byte>.Shared.Rent(Math.Max(inputLength, 1));
        var output = ArrayPool<byte>.Shared.Rent(Math.Max(outputLength, 1));
        try
        {
            var gathered = 0;
            foreach (var buffer in inputs)
            {
                if (!ctx.Memory.TryRead(buffer.Address, input.AsSpan(gathered, buffer.Length)))
                {
                    return new Atrac9DecodeResult(Atrac9DecodeState.ResultInvalidParameter, 0, 0, mp3.TotalDecodedSamples, 0);
                }

                gathered += buffer.Length;
            }

            var outputSpan = output.AsSpan(0, outputLength);
            outputSpan.Clear();
            var decoded = mp3.Decode(input.AsSpan(0, inputLength), outputSpan, pcm16: instance.PreferPcm16);
            if (!decoded.Success && decoded.InputConsumed == 0)
            {
                return new Atrac9DecodeResult(Atrac9DecodeState.ResultInvalidData, 0, 0, mp3.TotalDecodedSamples, 0);
            }

            // Scatter the decoded PCM and zero the rest of every output buffer so no
            // stale samples remain behind the produced bytes.
            var scattered = 0;
            foreach (var buffer in outputs)
            {
                if (!ctx.Memory.TryWrite(buffer.Address, output.AsSpan(scattered, buffer.Length)))
                {
                    return new Atrac9DecodeResult(
                        Atrac9DecodeState.ResultInvalidParameter,
                        decoded.InputConsumed,
                        Math.Min(scattered, decoded.OutputWritten),
                        mp3.TotalDecodedSamples,
                        decoded.Frames);
                }

                scattered += buffer.Length;
            }

            return new Atrac9DecodeResult(
                0,
                decoded.InputConsumed,
                decoded.OutputWritten,
                mp3.TotalDecodedSamples,
                decoded.Frames);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(input);
            ArrayPool<byte>.Shared.Return(output);
        }
    }
}
