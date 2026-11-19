// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal.Tests.Conformance;

/// <summary>The outcome of one form: how many lane cases ran, how many disagreed with the
/// reference model, and the first disagreements in readable form.</summary>
internal sealed class FormReport(string opcode, string label, int wave, AluMode mode)
{
    public string Opcode { get; } = opcode;
    public string Label { get; } = label;
    public int Wave { get; } = wave;
    public AluMode Mode { get; } = mode;
    public int Cases { get; set; }
    public int Failures { get; set; }
    public string? Error { get; set; }
    public List<string> Examples { get; } = [];

    public bool Passed => Error is null && Failures == 0;

    public override string ToString() =>
        $"w{Wave} {Mode} {Label}: {(Error is not null ? "ERROR " + Error : $"{Failures}/{Cases} failed")}" +
        (Examples.Count > 0 ? "\n    " + string.Join("\n    ", Examples) : "");
}

/// <summary>Translates a harness program with our pipeline, runs it on the Metal device, and
/// compares every lane bit-exactly with the reference model.</summary>
internal static class ConformanceRunner
{
    private const int MaxExamples = 6;

    public static ShaderCompileRequest Request(uint[] words, int wave, AluMode mode)
    {
        var fixture = new Gen5ComputeFixture("conformance", [.. words], 8, wave * ConformancePrograms.LaneBytes);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[10] = (uint)(wave * ConformancePrograms.LaneBytes);
        userData[11] = DescriptorConstants.IdentityDestinationSelect;
        return Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Compute, (uint)wave, (uint)wave,
            runtimeInputs: new ResourceRuntimeInputs { UserData = userData },
            ieeeMode: mode.Ieee, floatMode: mode.FloatMode, dx10Clamp: mode.Dx10Clamp);
    }

    /// <summary>Decodes, translates and compiles a program. Returns an error string on failure.</summary>
    public static string? Compile(IReadOnlyList<string> lines, int wave, AluMode mode, out Gen5MslShader? shader, out nint library)
    {
        shader = null;
        library = 0;
        var missing = new List<string>();
        var words = ConformancePrograms.Assemble(lines, wave, missing);
        if (words is null)
        {
            return "no llvm-mc encoding for: " + string.Join(" | ", missing.Distinct());
        }

        ShaderCompileRequest request;
        try
        {
            request = Request(words, wave, mode);
        }
        catch (Exception exception)
        {
            return "decode: " + exception.Message;
        }

        if (!Gen5MslTranslator.TryCompileProgram(request, out var compiled, out var error,
                Gen5MslCodegenOptions.FromEnvironment with { FuseLegacyMultiplyAdd = false }))
        {
            return "translate: " + error;
        }

        shader = compiled;
        if (!MetalNative.TryCompileLibrary(compiled.Source, out library, out error))
        {
            return "metal: " + error;
        }

        return null;
    }

    public static string? Execute(Gen5MslShader shader, nint library, int wave, byte[] buffer, out byte[] result)
    {
        result = [];
        return MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, buffer,
            new byte[PushData.ByteSize], shader.ArgumentLayout!, (uint)wave, out result, out var error)
            ? null
            : error;
    }

    private static uint Dword(byte[] buffer, int lane, int dword) =>
        BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(lane * ConformancePrograms.LaneBytes + dword * 4));

    private static void SetDword(byte[] buffer, int lane, int dword, uint value) =>
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(lane * ConformancePrograms.LaneBytes + dword * 4), value);

    /// <summary>Runs a vector form over its operand tuples, one wave of cases per dispatch.</summary>
    public static FormReport RunVector(VectorForm form, int wave, AluMode mode, IReadOnlyList<ulong[]> tuples, int seed,
        bool execute)
    {
        var report = new FormReport(form.Op.Mnemonic, form.Label, wave, mode);
        var lines = ConformancePrograms.Vector(form.Asm(wave), wave);
        report.Error = Compile(lines, wave, mode, out var shader, out var library);
        if (report.Error is not null || !execute) return report;

        var random = new Random(seed);
        var packed = form.Kind == FormKind.Packed;
        for (var start = 0; start < tuples.Count; start += wave)
        {
            var data = new VectorWave(wave);
            var buffer = new byte[wave * ConformancePrograms.LaneBytes];
            for (var lane = 0; lane < wave; lane++)
            {
                var tuple = tuples[(start + lane) % tuples.Count];
                for (var source = 0; source < 3; source++)
                {
                    var value = tuple[source];
                    var type = source < form.Op.Sources.Length ? form.Op.Sources[source] : OpType.B32;
                    if (type.Width() == 16 && value <= 0xFFFF)
                    {
                        // The other half of a 16-bit operand's register: the partner half of a
                        // packed operand, or data an op reading the wrong half would pick up.
                        var partner = tuples[(start + lane * 7 + source + 3) % tuples.Count][source] & 0xFFFF;
                        value |= (packed ? partner : partner ^ 0x5A5A) << 16;
                    }

                    data.Sources[source][lane] = value;
                }

                data.Old[lane] = (ulong)random.NextInt64() ^ ((ulong)random.Next() << 40);
                // Mostly active lanes; every eighth inactive at a lane-varying position.
                if (random.Next(8) != 0) data.Exec |= 1ul << lane;
                if (random.Next(2) != 0) data.Vcc |= 1ul << lane;
                SetDword(buffer, lane, 0, (uint)data.Sources[0][lane]);
                SetDword(buffer, lane, 1, (uint)(data.Sources[0][lane] >> 32));
                SetDword(buffer, lane, 2, (uint)data.Sources[1][lane]);
                SetDword(buffer, lane, 3, (uint)(data.Sources[1][lane] >> 32));
                SetDword(buffer, lane, 4, (uint)data.Sources[2][lane]);
                SetDword(buffer, lane, 5, (uint)(data.Sources[2][lane] >> 32));
                SetDword(buffer, lane, 6, (uint)data.Old[lane]);
                SetDword(buffer, lane, 7, (uint)(data.Old[lane] >> 32));
                SetDword(buffer, lane, 8, (uint)(data.Vcc >> lane & 1));
                SetDword(buffer, lane, 9, (uint)(data.Exec >> lane & 1));
            }

            // Keep one lane active so EXEC is never empty.
            if ((data.Exec & data.LaneMask) == 0)
            {
                data.Exec = 1;
                SetDword(buffer, 0, 9, 1);
            }

            var error = Execute(shader!, library, wave, buffer, out var result);
            if (error is not null)
            {
                report.Error = "execute: " + error;
                return report;
            }

            var expected = WaveModel.Evaluate(form, data, mode);
            Compare(form, data, expected, result, report);
        }

        return report;
    }

    private static void Compare(VectorForm form, VectorWave data, VectorOutcome expected, byte[] result, FormReport report)
    {
        var wave = data.Size;
        var laneMask = data.LaneMask;
        var vcc = Dword(result, 0, OutputDword(2)) | ((ulong)Dword(result, 0, OutputDword(3)) << 32);
        var exec = Dword(result, 0, OutputDword(4)) | (wave == 64 ? (ulong)Dword(result, 0, OutputDword(5)) << 32 : 0);
        var sdst = Dword(result, 0, OutputDword(6)) | ((ulong)Dword(result, 0, OutputDword(7)) << 32);
        var expectedExec = wave == 64 ? expected.Exec : expected.Exec & laneMask;
        for (var lane = 0; lane < wave; lane++)
        {
            report.Cases++;
            var actual = Dword(result, lane, OutputDword(0)) | ((ulong)Dword(result, lane, OutputDword(1)) << 32);
            var problems = new List<string>();
            if (!Matches(form, expected.Destination[lane], expected.Loose[lane], actual))
            {
                problems.Add($"D=0x{actual:X16} want 0x{expected.Destination[lane]:X16}{(expected.Loose[lane] != 0 ? $" (NaN at 0x{expected.Loose[lane]:X})" : "")}");
            }

            var bit = 1ul << lane;
            if ((vcc & bit) != (expected.Vcc & bit) || (lane == 0 && vcc != expected.Vcc))
            {
                problems.Add($"VCC=0x{vcc:X16} want 0x{expected.Vcc:X16}");
            }

            if ((exec & bit) != (expectedExec & bit))
            {
                problems.Add($"EXEC=0x{exec:X16} want 0x{expectedExec:X16}");
            }

            if ((sdst & bit) != (expected.ScalarDestination & bit) || (lane == 0 && sdst != expected.ScalarDestination))
            {
                problems.Add($"SDST=0x{sdst:X16} want 0x{expected.ScalarDestination:X16}");
            }

            if (problems.Count == 0) continue;
            report.Failures++;
            if (report.Examples.Count < MaxExamples)
            {
                report.Examples.Add(string.Create(CultureInfo.InvariantCulture,
                    $"lane {lane} exec={(data.Exec >> lane) & 1} vcc={(data.VccBefore >> lane) & 1} " +
                    $"S0=0x{data.Sources[0][lane]:X} S1=0x{data.Sources[1][lane]:X} S2=0x{data.Sources[2][lane]:X} " +
                    $"old=0x{data.Old[lane]:X}: {string.Join("; ", problems)}"));
            }
        }
    }

    private static int OutputDword(int index) => ConformancePrograms.OutputDword + index;

    /// <summary>Bit equality, except that bits marked loose need only hold a NaN of the field's
    /// width, and approximate ops allow their ULP tolerance on finite results.</summary>
    private static bool Matches(VectorForm form, ulong expected, ulong loose, ulong actual)
    {
        if (expected == actual) return true;
        if (loose != 0)
        {
            if ((expected & ~loose) != (actual & ~loose)) return false;
            var shift = System.Numerics.BitOperations.TrailingZeroCount(loose);
            var width = System.Numerics.BitOperations.PopCount(loose);
            var format = width switch { 16 => FloatFormat.F16, 32 => FloatFormat.F32, _ => FloatFormat.F64 };
            if (width is 32 or 64 && (loose >> shift) == format.ValueMask &&
                (width == 64 || form.Op.Result == OpType.F32))
            {
                return SoftFloat.IsNaN((actual >> shift) & format.ValueMask, format);
            }

            // Loose f16 fields (a 16-bit result, or the halves of packed math).
            for (var half = 0; half < 4; half++)
            {
                var field = loose >> (16 * half) & 0xFFFF;
                if (field == 0) continue;
                if (field != 0xFFFF || !SoftFloat.IsNaN(actual >> (16 * half) & 0xFFFF, FloatFormat.F16)) return false;
            }

            return true;
        }

        if (form.Op.Has(VectorFlags.Approximate) && form.Op.Result.IsFloat())
        {
            var format = form.Op.Result.Format();
            var mask = format.ValueMask;
            var keep = form.Op.Result.Width() == 64 ? 0 : ~(ulong)uint.MaxValue | (form.Op.Result.Width() == 16 ? 0xFFFF0000ul : 0);
            if ((expected & keep) != (actual & keep)) return false;
            var e = expected & mask;
            var a = actual & mask;
            if (SoftFloat.IsNaN(e, format) || SoftFloat.IsInfinity(e, format) ||
                SoftFloat.IsNaN(a, format) || SoftFloat.IsInfinity(a, format))
            {
                return e == a;
            }

            if (SoftFloat.IsZero(e, format) && SoftFloat.IsZero(a, format)) return e == a;
            if (form.Op.AbsoluteTolerance > 0)
            {
                return Math.Abs(ToDouble(e, format) - ToDouble(a, format)) <= form.Op.AbsoluteTolerance;
            }

            if (SoftFloat.IsZero(e, format)) return e == a;

            if (SoftFloat.IsNegative(e, format) != SoftFloat.IsNegative(a, format)) return false;
            var distance = Math.Abs((long)(e & ~format.SignMask) - (long)(a & ~format.SignMask));
            return distance <= form.Op.UlpTolerance;
        }

        return false;
    }

    private static double ToDouble(ulong bits, FloatFormat format) =>
        format.Width == 64 ? BitConverter.UInt64BitsToDouble(bits)
        : format.Width == 32 ? BitConverter.UInt32BitsToSingle((uint)bits)
        : (double)BitConverter.UInt16BitsToHalf((ushort)bits);

    /// <summary>Runs a scalar form: one case per lane through the program's lane loop.</summary>
    public static FormReport RunScalar(ScalarForm form, int wave, IReadOnlyList<ulong[]> tuples, int seed, bool execute)
    {
        var op = form.Op;
        var exec = op.Has(ScalarFlags.Exec);
        var report = new FormReport(op.Mnemonic, form.Label, wave, AluMode.Default);
        var lines = ConformancePrograms.Scalar(form.Asm(wave), wave, exec);
        report.Error = Compile(lines, wave, AluMode.Default, out var shader, out var library);
        if (report.Error is not null || !execute) return report;

        var random = new Random(seed);
        for (var start = 0; start < tuples.Count; start += wave)
        {
            var buffer = new byte[wave * ConformancePrograms.LaneBytes];
            var cases = new (ulong S0, ulong S1, ulong Old, bool Scc, ulong Exec)[wave];
            for (var lane = 0; lane < wave; lane++)
            {
                var tuple = tuples[(start + lane) % tuples.Count];
                var old = (ulong)random.NextInt64() ^ ((ulong)random.Next() << 40);
                // S_ADDK/S_MULK read their destination: its low dword takes the operand edges.
                if (op.Has(ScalarFlags.DestinationIsSource)) old = (old & 0xFFFFFFFF00000000ul) | (uint)tuple[0];
                var scc = random.Next(2) != 0;
                var laneExec = random.Next(4) switch
                {
                    0 => 0ul,
                    1 => ulong.MaxValue,
                    _ => (ulong)random.NextInt64() ^ ((ulong)random.Next() << 40),
                };
                if (wave == 32) laneExec &= 0xFFFFFFFF;
                // An empty EXEC would stop the loop's V_WRITELANEs on hardware that honoured it;
                // the program restores EXEC before writing back, so any value is safe.
                cases[lane] = (tuple[0], tuple[1], old, scc, laneExec);
                SetDword(buffer, lane, 0, (uint)tuple[0]);
                SetDword(buffer, lane, 1, (uint)(tuple[0] >> 32));
                SetDword(buffer, lane, 2, (uint)tuple[1]);
                SetDword(buffer, lane, 3, (uint)(tuple[1] >> 32));
                SetDword(buffer, lane, 4, (uint)(laneExec >> 32));
                SetDword(buffer, lane, 6, (uint)old);
                SetDword(buffer, lane, 7, (uint)(old >> 32));
                SetDword(buffer, lane, 8, scc ? 1u : 0u);
                SetDword(buffer, lane, 9, (uint)laneExec);
            }

            var error = Execute(shader!, library, wave, buffer, out var result);
            if (error is not null)
            {
                report.Error = "execute: " + error;
                return report;
            }

            for (var lane = 0; lane < wave; lane++)
            {
                report.Cases++;
                var c = cases[lane];
                var (destination, scc, execOut) = form.Evaluate(c.S0, c.S1, c.Old, c.Scc, c.Exec, wave);
                var actual = Dword(result, lane, OutputDword(0)) | ((ulong)Dword(result, lane, OutputDword(1)) << 32);
                var actualScc = Dword(result, lane, OutputDword(2)) != 0;
                var problems = new List<string>();
                if (actual != destination) problems.Add($"D=0x{actual:X16} want 0x{destination:X16}");
                if (actualScc != scc) problems.Add($"SCC={(actualScc ? 1 : 0)} want {(scc ? 1 : 0)}");
                if (exec)
                {
                    var actualExec = Dword(result, lane, OutputDword(3)) | (wave == 64 ? (ulong)Dword(result, lane, OutputDword(4)) << 32 : 0);
                    if (actualExec != execOut) problems.Add($"EXEC=0x{actualExec:X} want 0x{execOut:X}");
                }

                if (problems.Count == 0) continue;
                report.Failures++;
                if (report.Examples.Count < MaxExamples)
                {
                    report.Examples.Add(string.Create(CultureInfo.InvariantCulture,
                        $"S0=0x{c.S0:X} S1=0x{c.S1:X} old=0x{c.Old:X} scc={(c.Scc ? 1 : 0)} exec=0x{c.Exec:X}: {string.Join("; ", problems)}"));
                }
            }
        }

        return report;
    }

    public static string Table(IEnumerable<FormReport> reports)
    {
        var builder = new StringBuilder();
        foreach (var group in reports.GroupBy(static report => report.Opcode).OrderBy(static group => group.Key, StringComparer.Ordinal))
        {
            var cases = group.Sum(static report => report.Cases);
            var failed = group.Count(static report => !report.Passed);
            builder.Append(CultureInfo.InvariantCulture,
                $"{group.Key}\tforms={group.Count()}\tcases={cases}\tfailing_forms={failed}\n");
        }

        return builder.ToString();
    }
}
