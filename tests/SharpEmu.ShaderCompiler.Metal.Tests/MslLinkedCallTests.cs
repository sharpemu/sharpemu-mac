// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Linked shader calls on Metal (Gen5MslTranslator.ShaderCalls.cs): the linker's tests select a callee by the caller's
/// address and argument registers against the dispatch's call targets and arguments, an entry writes the return address
/// relocated with the shader base, a call no record names reports a call fault and ends the wave, a run-time-record
/// program marks a callee it lacks, and a callee reads an image its record holds through the bindless heap. The MSL
/// twins of the SPIR-V LinkedCalls_* device tests (tests/SharpEmu.Libs.Tests/Gpu/Vulkan/DeviceAddressShaderTests.cs).
/// </summary>
public sealed class MslLinkedCallTests(ITestOutputHelper output)
{
    private const ulong Hash = 0x1234_5678_9ABC_DEF0;
    private const uint UserDataCount = 16;
    private const uint ResultRegister = 4;
    private const int ResultBytes = 256;
    private const uint NullOperand = 125;

    // Instruction builders: the shapes of ResourceTestProgram (tests/SharpEmu.ShaderCompiler.Tests), which this project
    // does not reference.
    private static Gen5Operand Operand(uint value) =>
        value <= 64 ? Gen5Operand.Source(128 + value) : new Gen5Operand(Gen5OperandKind.LiteralConstant, value);

    private static Gen5ShaderProgram Program(params Gen5ShaderInstruction[] instructions) => new(0, instructions);

    private static Gen5ShaderInstruction EndProgram(uint pc) =>
        new(pc, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null);

    private static Gen5ShaderInstruction Nop(uint pc) =>
        new(pc, Gen5ShaderEncoding.Sopp, "SNop", [0xBF800000], [], [], null);

    private static Gen5ShaderInstruction MoveScalar(uint pc, uint destination, uint value) =>
        new(pc, Gen5ShaderEncoding.Sop1, "SMovB32", [0u], [Operand(value)], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction Sop1(uint pc, string opcode, uint destination, Gen5Operand source) =>
        new(pc, Gen5ShaderEncoding.Sop1, opcode, [0u], [source], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction Sop2(uint pc, string opcode, uint destination, Gen5Operand source0, Gen5Operand source1) =>
        new(pc, Gen5ShaderEncoding.Sop2, opcode, [0u], [source0, source1], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction Sopc(uint pc, string opcode, Gen5Operand source0, Gen5Operand source1) =>
        new(pc, Gen5ShaderEncoding.Sopc, opcode, [0u], [source0, source1], [], null);

    private static Gen5ShaderInstruction Vop1(uint pc, string opcode, uint destination, Gen5Operand source) =>
        new(pc, Gen5ShaderEncoding.Vop1, opcode, [0u], [source], [Gen5Operand.Vector(destination)], null);

    private static Gen5ShaderInstruction Vop2(uint pc, string opcode, uint destination, Gen5Operand source0, Gen5Operand source1) =>
        new(pc, Gen5ShaderEncoding.Vop2, opcode, [0u], [source0, source1], [Gen5Operand.Vector(destination)], null);

    private static Gen5ShaderInstruction ScalarLoad(uint pc, uint baseRegister, uint destination, uint count, int immediateOffset = 0) =>
        new(pc, Gen5ShaderEncoding.Smem, count == 1 ? "SLoadDword" : $"SLoadDwordx{count}", [0u, 0u],
            [Gen5Operand.Scalar(baseRegister), Gen5Operand.Source(NullOperand)],
            Enumerable.Range((int)destination, (int)count).Select(index => Gen5Operand.Scalar((uint)index)).ToArray(),
            new Gen5ScalarMemoryControl(count, immediateOffset, null));

    // image_sample_lz v[4:7], v[0:1], T# s[resource], S# s[sampler] dmask:0xf dim:2D.
    private static Gen5ShaderInstruction ImageSample(uint pc, uint resourceRegister, uint samplerRegister)
    {
        var control = new Gen5ImageControl(0xF, 0, [0u, 1u, 2u], 4, resourceRegister, samplerRegister, 1, false, Glc: false, Slc: false, A16: false, D16: false);
        return new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Mimg, "ImageSampleLz", [0u, 0u],
            [Gen5Operand.Vector(0), Gen5Operand.Scalar(resourceRegister), Gen5Operand.Scalar(samplerRegister)], [Gen5Operand.Vector(4)], control);
    }

    private static Gen5ShaderInstruction ImageLoad(uint pc, uint resourceRegister)
    {
        var control = new Gen5ImageControl(0xF, 0, [0u, 1u], 4, resourceRegister, 0, 1, false, Glc: false, Slc: false, A16: false, D16: false);
        return new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Mimg, "ImageLoad", [0u, 0u],
            [Gen5Operand.Vector(0), Gen5Operand.Scalar(resourceRegister)], [Gen5Operand.Vector(4)], control);
    }

    private static Gen5ShaderInstruction BufferAccess(uint pc, string opcode, uint resourceRegister, uint dwords = 1, uint vectorData = 4,
        bool offsetEnabled = false, uint vectorAddress = 0)
    {
        var data = Enumerable.Range((int)vectorData, (int)dwords).Select(index => Gen5Operand.Vector((uint)index)).ToArray();
        return new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Mubuf, opcode, [0u, 0u],
            [Gen5Operand.Vector(vectorAddress), Gen5Operand.Scalar(resourceRegister), Gen5Operand.Source(NullOperand), .. data], [],
            new Gen5BufferMemoryControl(dwords, vectorAddress, vectorData, resourceRegister, 0, false, offsetEnabled, Glc: false, Slc: false));
    }

    // The request the pipeline cache builds for a linked program: the call arguments and targets pushed as shader data.
    private static ShaderCompileRequest Request(Gen5ShaderProgram program, bool bindlessHeap = false, bool callRecords = false, bool callMarks = false)
    {
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, UserDataCount,
            bindlessHeap: bindlessHeap, callRecords: callRecords, callMarks: callMarks, scalarDataOnDevice: false);
        // The default specialization, as upstream's ResourceTestProgram.Prepare: the result V# is a user data buffer.
        var specialization = ResourceSpecialization.Default(plan.Info);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(
            resources.Info,
            BindingLayout.CollectUserDataRegisters(program, 0, UserDataCount, plan.Graph.ExcludedUserDataRegisters),
            BindingLayout.UsesGlobalDataShare(program),
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources),
            BindingLayout.ReadsShaderBase(program),
            pushDataStartDword: 0,
            callArgumentCount: BindingLayout.CountCallArguments(program),
            callTargetCount: BindingLayout.CountCallTargets(program));
        return new ShaderCompileRequest(plan, resources, layout) { WaveSize = 32, LocalSizeX = 32 };
    }

    private static Gen5MslShader Compile(ShaderCompileRequest request)
    {
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }

    // One dispatch of a linked program: the shader data the host pushes (the shader base and the call data), the result
    // buffer, and the call fault report read back.
    private sealed class Run
    {
        private readonly Gen5MslShader _shader;
        private readonly nint _library;
        private byte[] _result = [];

        public Run(Gen5ShaderProgram program, bool bindlessHeap = false, bool callRecords = false, bool callMarks = false)
        {
            Request = MslLinkedCallTests.Request(program, bindlessHeap, callRecords, callMarks);
            _shader = Compile(Request);
            Assert.True(MetalNative.TryCompileLibrary(_shader.Source, out _library, out var error), $"{error}\n{_shader.Source}");
        }

        public ShaderCompileRequest Request { get; }

        public uint[] Faults { get; private set; } = [];

        public void Dispatch(uint threads = 1, ulong shaderBase = 0, ulong[]? callTargets = null, ulong[]? callArguments = null,
            IReadOnlyList<DeviceTable>? tables = null, (DeviceTable Elements, DeviceTable Feedback)? heap = null)
        {
            var layout = Request.Bindings;
            var shaderData = new uint[layout.ShaderDataDwordCount];
            if (layout.UsesShaderBase)
            {
                shaderData[layout.ShaderBaseDword] = (uint)shaderBase;
                shaderData[layout.ShaderBaseDword + 1] = (uint)(shaderBase >> 32);
            }

            var offset = (int)layout.CallArgumentsDword;
            foreach (var address in (callArguments ?? Gen5ShaderCallLinker.CallEntryArguments(Request.Program))
                         .Concat(callTargets ?? Gen5ShaderCallLinker.CallTargets(Request.Program)))
            {
                shaderData[offset++] = (uint)address;
                shaderData[offset++] = (uint)(address >> 32);
            }

            var push = new byte[PushData.ByteSize];
            var allTables = (tables ?? []).ToList();
            if (layout.UsesPushData)
            {
                for (var index = 0; index < shaderData.Length; index++)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(layout.PushDataStartDword + index) * 4), shaderData[index]);
                }
            }
            else
            {
                allTables.Add(MetalNative.CreateWordTable(DescriptorBindingKind.ShaderData, shaderData));
            }

            var faults = MetalNative.CreateWordTable(DescriptorBindingKind.FaultBuffer, new uint[Gen5MslArgumentLayout.CallFaultWords]);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(_library, _shader.EntryPoint, new byte[ResultBytes], push,
                _shader.ArgumentLayout!, threads, out _result, out _, out var error, tables: allTables, heap: heap, callFaults: faults), error);
            Faults = MetalNative.ReadWords(faults);
        }

        public uint Word(int offset) => BinaryPrimitives.ReadUInt32LittleEndian(_result.AsSpan(offset));
    }

    private bool SkipWithoutMetalDevice()
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            "The required gate cannot run linked calls without a Metal device.");
        output.WriteLine("No Metal device on this host; linked calls skipped.");
        return true;
    }

    private static uint[] FaultRecord(ulong hash, uint callPc, uint addressLow, uint addressHigh) =>
        [2, (uint)hash, (uint)(hash >> 32), callPc, addressLow, addressHigh, (uint)ShaderStage.Compute, 1];

    // A program linked per record selects a target by its address and argument, keeps SCC across the call, and reports a
    // call that no record names (the target, the argument or the high address word differ) before it ends the wave.
    [Theory]
    [InlineData(0x2000u, 0u, 10u, 77u)]
    [InlineData(0x2000u, 0u, 11u, 99u)]
    [InlineData(0x2000u, 0u, 12u, 0u)]
    [InlineData(0x3000u, 0u, 10u, 55u)]
    [InlineData(0x2000u, 1u, 10u, 0u)]
    public void SelectArgumentsPreserveSccAndReportUnknownTargets(uint addressLow, uint addressHigh, uint argument, uint expected)
    {
        var caller = Program(MoveScalar(0, 14, addressLow), MoveScalar(4, 15, addressHigh),
            MoveScalar(8, 16, argument), MoveScalar(12, 17, 0),
            Sopc(16, "SCmpEqU32", Operand(1), Operand(1)),
            Sop1(20, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            Sop2(24, "SCselectB32", 8, Operand(123), Operand(9)),
            Vop1(28, "VMovB32", 2, Gen5Operand.Scalar(8)),
            Vop1(32, "VMovB32", 3, Gen5Operand.Scalar(14)),
            Vop1(36, "VMovB32", 4, Gen5Operand.Scalar(15)),
            BufferAccess(40, "BufferStoreDwordx4", ResultRegister, dwords: 4, vectorData: 1), EndProgram(48));
        Gen5ShaderProgram Function(uint value) => Program(
            Vop1(0, "VMovB32", 1, Operand(value)),
            Sop1(4, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var linked = Gen5ShaderCallLinker.Link(caller,
            [new(20, 14, 16, 14, caller.Address + 24,
                [new(0x2000, 10, Function(77)), new(0x2000, 11, Function(99)), new(0x3000, 10, Function(55))])]);
        var source = Compile(Request(linked)).Source;
        Assert.Contains($"{Gen5MslArgumentLayout.CallFaultsName}[0]", source, StringComparison.Ordinal);
        if (SkipWithoutMetalDevice()) return;

        var run = new Run(linked);
        run.Dispatch();
        Assert.Equal(expected, run.Word(0));
        if (expected != 0)
        {
            Assert.Equal(123u, run.Word(4));
            Assert.Equal(24u, run.Word(8));
            Assert.Equal(0u, run.Word(12));
            Assert.All(run.Faults, word => Assert.Equal(0u, word));
        }
        else
        {
            Assert.Equal(FaultRecord(run.Request.Hash, 20, addressLow, addressHigh), run.Faults);
        }
    }

    // The test of the 65th record of one target runs after 64 that do not match.
    [Fact]
    public void SelectTheLastRecordBeyondSixtyFourEntries()
    {
        if (SkipWithoutMetalDevice()) return;
        var caller = Program(MoveScalar(0, 14, 0x2000), MoveScalar(4, 15, 0),
            MoveScalar(8, 16, 64), MoveScalar(12, 17, 0),
            Sop1(16, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            BufferAccess(20, "BufferStoreDword", ResultRegister, vectorData: 1), EndProgram(28));
        var function = Program(Vop1(0, "VMovB32", 1, Gen5Operand.Scalar(16)),
            Sop1(4, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var targets = Enumerable.Range(0, 65).Select(index => new ShaderCallTarget(0x2000, (ulong)index, function)).ToArray();
        var linked = Gen5ShaderCallLinker.Link(caller, [new(16, 14, 16, 14, 20, targets)]);
        var run = new Run(linked);
        run.Dispatch();
        Assert.Equal(64u, run.Word(0));
        Assert.All(run.Faults, word => Assert.Equal(0u, word));
    }

    // A site finds the record of each callee, whichever callee and argument the caller names; an unknown pair faults.
    [Theory]
    [InlineData(0x2000u, 10u, 110u)]
    [InlineData(0x2000u, 11u, 111u)]
    [InlineData(0x3000u, 10u, 210u)]
    [InlineData(0x3000u, 11u, 211u)]
    [InlineData(0x3000u, 12u, 0u)]
    public void SiteLookupFindsTheRecordOfEachCallee(uint address, uint argument, uint expected)
    {
        if (SkipWithoutMetalDevice()) return;
        var caller = Program(MoveScalar(0, 14, address), MoveScalar(4, 15, 0), MoveScalar(8, 16, argument), MoveScalar(12, 17, 0),
            Sop1(16, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            BufferAccess(20, "BufferStoreDword", ResultRegister, vectorData: 1), EndProgram(28));
        Gen5ShaderProgram Function(uint add) => Program(
            Vop1(0, "VMovB32", 1, Gen5Operand.Scalar(16)),
            Vop2(4, "VAddU32", 1, Operand(add), Gen5Operand.Vector(1)),
            Sop1(8, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var first = Function(100);
        var second = Function(200);
        var linked = Gen5ShaderCallLinker.Link(caller, [new(16, 14, 16, 14, 20,
            [new(0x2000, 10, first), new(0x3000, 10, second), new(0x2000, 11, first), new(0x3000, 11, second)])]);
        var run = new Run(linked);
        run.Dispatch();
        Assert.Equal(expected, run.Word(0));
        if (expected != 0) Assert.All(run.Faults, word => Assert.Equal(0u, word));
        else Assert.Equal(FaultRecord(run.Request.Hash, 16, address, 0), run.Faults);
    }

    // A function reads its program address with S_GETPC_B64: the address the dispatch pushes for it (a function the game
    // placed elsewhere than at link time is still found and reads where it now is), and the return address follows the
    // shader base.
    [Theory]
    [InlineData(0x8000ul, 0x8000ul)]
    [InlineData(0x20000ul, 0x20000ul)]
    [InlineData(0x8000ul, 0x5_0000_0400ul)]
    public void FunctionReadsItsOwnProgramAddress(ulong function, ulong placed)
    {
        if (SkipWithoutMetalDevice()) return;
        const ulong callerAddress = 0x10000;
        const ulong shaderBase = 0x3_0040_0000;
        var caller = Program(MoveScalar(0, 14, (uint)placed), MoveScalar(4, 15, (uint)(placed >> 32)), MoveScalar(8, 16, 10),
            MoveScalar(12, 17, 0),
            Sop1(16, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            Vop1(20, "VMovB32", 3, Gen5Operand.Scalar(14)), Vop1(24, "VMovB32", 4, Gen5Operand.Scalar(15)),
            BufferAccess(28, "BufferStoreDwordx4", ResultRegister, dwords: 4, vectorData: 1), EndProgram(36)) with { Address = callerAddress };
        var body = Program(Nop(0), Sop1(4, "SGetpcB64", 2, Gen5Operand.Scalar(0)),
            Vop1(8, "VMovB32", 1, Gen5Operand.Scalar(2)), Vop1(12, "VMovB32", 2, Gen5Operand.Scalar(3)),
            Sop1(16, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var linked = Gen5ShaderCallLinker.Link(caller, [new(16, 14, 16, 14, callerAddress + 20, [new(function, 10, body)])]);
        var run = new Run(linked);
        run.Dispatch(shaderBase: shaderBase, callTargets: [placed]);
        Assert.Equal((uint)(placed + 8), run.Word(0));
        Assert.Equal((uint)((placed + 8) >> 32), run.Word(4));
        Assert.Equal(unchecked((uint)(shaderBase + 20)), run.Word(8));
        Assert.Equal((uint)(shaderBase >> 32), run.Word(12));
        Assert.All(run.Faults, word => Assert.Equal(0u, word));
    }

    // The caller's own S_GETPC_B64 after a call reads the shader base plus its offset in the caller's code.
    [Fact]
    public void CallerReadsItsOwnProgramAddressAfterACall()
    {
        if (SkipWithoutMetalDevice()) return;
        const ulong shaderBase = 0x3_0040_0000;
        var caller = Program(MoveScalar(0, 14, 0x2000), MoveScalar(4, 15, 0), MoveScalar(8, 16, 10), MoveScalar(12, 17, 0),
            Sop1(16, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            Sop1(20, "SGetpcB64", 2, Gen5Operand.Scalar(0)),
            Vop1(24, "VMovB32", 3, Gen5Operand.Scalar(2)), Vop1(28, "VMovB32", 4, Gen5Operand.Scalar(3)),
            BufferAccess(32, "BufferStoreDwordx4", ResultRegister, dwords: 4, vectorData: 1), EndProgram(40));
        var function = Program(Vop1(0, "VMovB32", 1, Gen5Operand.Scalar(16)), Vop1(4, "VMovB32", 2, Operand(7)),
            Sop1(8, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var linked = Gen5ShaderCallLinker.Link(caller, [new(16, 14, 16, 14, caller.Address + 20, [new(0x2000, 10, function)])]);
        var run = new Run(linked);
        run.Dispatch(shaderBase: shaderBase);
        Assert.Equal(10u, run.Word(0));
        Assert.Equal(7u, run.Word(4));
        Assert.Equal(unchecked((uint)(shaderBase + 24)), run.Word(8));
        Assert.Equal((uint)(shaderBase >> 32), run.Word(12));
        Assert.All(run.Faults, word => Assert.Equal(0u, word));
    }

    // Moved records keep the module: the test compares the caller's argument with the argument the dispatch pushes for the
    // record's slot, so a table moved to other addresses selects by its new arguments, and the old ones fault.
    [Theory]
    [InlineData(0x2000u, 0x500010u, 0x500010u + 100)]
    [InlineData(0x2000u, 0x100010u, 0u)]
    public void MovedArgumentsKeepTheModule(uint address, uint argument, uint expected)
    {
        if (SkipWithoutMetalDevice()) return;
        var caller = Program(MoveScalar(0, 14, address), MoveScalar(4, 15, 0), MoveScalar(8, 16, argument), MoveScalar(12, 17, 0),
            Sop1(16, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            BufferAccess(20, "BufferStoreDword", ResultRegister, vectorData: 1), EndProgram(28));
        var function = Program(
            Vop1(0, "VMovB32", 1, Gen5Operand.Scalar(16)),
            Vop2(4, "VAddU32", 1, Operand(100), Gen5Operand.Vector(1)),
            Sop1(8, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var linked = Gen5ShaderCallLinker.Link(caller, [new(16, 14, 16, 14, 20, [new(0x2000, 0x100010, function)])]);
        Assert.Equal([0x100010ul], Gen5ShaderCallLinker.CallEntryArguments(linked));
        var run = new Run(linked);
        run.Dispatch(callArguments: [0x500010ul]);
        Assert.Equal(expected, run.Word(0));
        if (expected != 0) Assert.All(run.Faults, word => Assert.Equal(0u, word));
        else Assert.Equal(FaultRecord(run.Request.Hash, 16, address, 0), run.Faults);
    }

    // A program linked with run-time records marks a call to a callee it lacks (CallMarks.Bit of the target and the
    // caller's hash) and returns from the call at once; a callee it holds runs and marks nothing.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeRecordCallsMarkCalleesTheProgramLacks(bool withCallee)
    {
        var caller = Program(MoveScalar(0, 14, 0x2000), MoveScalar(4, 15, 0), MoveScalar(8, 16, 10), MoveScalar(12, 17, 0),
            Vop1(16, "VMovB32", 1, Operand(5)),
            Sop1(20, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            BufferAccess(24, "BufferStoreDword", ResultRegister, vectorData: 1), EndProgram(32));
        var callee = Program(Vop1(0, "VMovB32", 1, Gen5Operand.Scalar(16)), Sop1(4, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        ShaderCallTarget[] targets = withCallee ? [new ShaderCallTarget(0x2000, 0, callee)] : [];
        var linked = Gen5ShaderCallLinker.Link(caller, [new(20, 14, 16, 14, 24, targets)], runtimeArguments: true);
        var request = Request(linked, callRecords: true, callMarks: true);
        Assert.NotNull(request.Bindings.Find(DescriptorBindingKind.CallMarks));
        var source = Compile(request).Source;
        Assert.Contains($"atomic_fetch_or_explicit(&sharpemu_resources.{Gen5MslArgumentLayout.CallMarksName}[", source,
            StringComparison.Ordinal);
        // Without call marks the program still returns from the call at once, and binds no marks.
        var unmarked = Compile(Request(linked, callRecords: true)).Source;
        Assert.DoesNotContain(Gen5MslArgumentLayout.CallMarksName, unmarked, StringComparison.Ordinal);
        if (SkipWithoutMetalDevice()) return;

        var run = new Run(linked, callRecords: true, callMarks: true);
        var marks = MetalNative.CreateWordTable(DescriptorBindingKind.CallMarks, new uint[CallMarks.Words]);
        run.Dispatch(tables: [marks]);
        Assert.Equal(withCallee ? 10u : 5u, run.Word(0));
        var expected = new uint[CallMarks.Words];
        if (!withCallee)
        {
            var bit = CallMarks.Bit(0x2000, run.Request.Hash);
            expected[bit >> 5] = 1u << (int)(bit & 31);
        }

        Assert.Equal(expected, MetalNative.ReadWords(marks));
        Assert.All(run.Faults, word => Assert.Equal(0u, word));
    }

    // A callee reads an image its record holds (the T# loaded through the record's argument): the row of the caller's
    // argument is found once at the entry, and the image is the heap element the host gave that row's field. An argument no
    // row holds reads the null element, which reads zero.
    [Theory]
    [InlineData(0x40u, true)]
    [InlineData(0x80u, false)]
    public void ACalleeReadsTheImageOfItsRecordThroughTheHeap(uint argument, bool found)
    {
        // The argument is a value only the wave knows (lane 0's thread id plus the argument), as a record address read from
        // memory is: the host cannot evaluate it, so the callee's reads through it are record fields.
        var caller = Program(MoveScalar(0, 14, 0x2000), MoveScalar(4, 15, 0),
            Vop2(8, "VAddU32", 8, Operand(argument), Gen5Operand.Vector(0)),
            new Gen5ShaderInstruction(16, Gen5ShaderEncoding.Vop1, "VReadfirstlaneB32", [0u], [Gen5Operand.Vector(8)], [Gen5Operand.Scalar(16)], null),
            MoveScalar(20, 17, 0),
            Sop1(24, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            BufferAccess(28, "BufferStoreDwordx4", ResultRegister, dwords: 4, vectorData: 4), EndProgram(36));
        var callee = Program(ScalarLoad(0, 16, 20, 8), ScalarLoad(8, 16, 28, 4, immediateOffset: 32),
            Vop1(16, "VMovB32", 0, Operand(0x3F000000)), Vop1(24, "VMovB32", 1, Operand(0x3F000000)),
            ImageSample(32, 20, 28),
            Sop1(40, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var linked = Gen5ShaderCallLinker.Link(caller, [new(24, 14, 16, 14, 28, [new ShaderCallTarget(0x2000, 0, callee)])],
            runtimeArguments: true);
        var request = Request(linked, bindlessHeap: true, callRecords: true);
        var use = Assert.Single(request.CallRecordImageByMemoryIndex).Value;
        var samplerUse = Assert.Single(request.CallRecordSamplerByMemoryIndex).Value;
        Assert.NotEqual(use.Field, samplerUse.Field);
        var source = Compile(request).Source;
        Assert.Contains("sharpemu_search_call_records(", source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_call_record_element(", source, StringComparison.Ordinal);
        if (SkipWithoutMetalDevice()) return;

        const uint element = BindlessHeap.CallRecordFirstIndex;
        const uint samplerElement = BindlessHeap.CallRecordFirstIndex + 1;
        uint[] texel = [0x3F800000, 0x40000000, 0x40400000, 0x40800000];
        var texture = MetalNative.CreateTexelTexture(MetalNative.PixelFormatRgba32Float,
            BindlessHeap.ImageClasses[use.Slot].Dimension == ImageDimension.Dim2D ? MetalNative.TextureType2D : MetalNative.TextureType2DArray, 1, texel);
        var elements = new byte[BindlessHeap.ArrayCount * BindlessHeap.Capacity * Gen5MslArgumentLayout.HeapElementBytes];
        BinaryPrimitives.WriteUInt64LittleEndian(elements.AsSpan((int)(((uint)use.Slot * BindlessHeap.Capacity + element) * 8)),
            MetalNative.ResourceId(texture));
        BinaryPrimitives.WriteUInt64LittleEndian(elements.AsSpan((int)(((uint)samplerUse.Slot * BindlessHeap.Capacity + samplerElement) * 8)),
            MetalNative.ResourceId(MetalNative.CreatePointSampler()));
        var feedbackWords = new uint[BindlessHeap.FeedbackWords];
        feedbackWords[BindlessHeap.MarkWords + (element >> (int)BindlessHeap.ClassWordShift)] |=
            BindlessHeap.ClassCode(ImageNumericClass.Float) << (int)((element & (BindlessHeap.ClassesPerWord - 1)) * BindlessHeap.ClassBits);
        // One row: the key (argument 0x40, callee slot 0), its first field at word 5, the fields' elements.
        var recordWords = new uint[] { 1, 0x40, 0, 0, 5, 0, 0, 0 };
        recordWords[5 + use.Field] = element;
        recordWords[5 + samplerUse.Field] = samplerElement;
        var records = MetalNative.CreateWordTable(DescriptorBindingKind.CallRecords, recordWords);
        var run = new Run(linked, bindlessHeap: true, callRecords: true);
        run.Dispatch(tables: [records],
            heap: (MetalNative.CreateHeapElements(elements, [texture]), MetalNative.CreateHeapFeedback(feedbackWords)));
        for (var component = 0; component < 4; component++)
        {
            Assert.Equal(found ? texel[component] : 0u, run.Word(component * 4));
        }

        Assert.All(run.Faults, word => Assert.Equal(0u, word));
    }

    // A callee loads an integer image its record holds: the load branches on the class the host wrote for the record's
    // element and reads it through the uint or sint declaration, bit for bit. The float branch would read the element
    // null, as the class check refuses an element of another class.
    [Theory]
    [InlineData(ImageNumericClass.Uint)]
    [InlineData(ImageNumericClass.Sint)]
    public void ACalleeLoadsTheIntegerImageOfItsRecordThroughItsClass(ImageNumericClass numericClass)
    {
        var caller = Program(MoveScalar(0, 14, 0x2000), MoveScalar(4, 15, 0),
            Vop2(8, "VAddU32", 8, Operand(0x40), Gen5Operand.Vector(0)),
            new Gen5ShaderInstruction(16, Gen5ShaderEncoding.Vop1, "VReadfirstlaneB32", [0u], [Gen5Operand.Vector(8)], [Gen5Operand.Scalar(16)], null),
            MoveScalar(20, 17, 0),
            Sop1(24, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            BufferAccess(28, "BufferStoreDwordx4", ResultRegister, dwords: 4, vectorData: 4), EndProgram(36));
        var callee = Program(ScalarLoad(0, 16, 20, 8),
            Vop1(8, "VMovB32", 0, Operand(0)), Vop1(16, "VMovB32", 1, Operand(0)),
            ImageLoad(24, 20),
            Sop1(32, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var linked = Gen5ShaderCallLinker.Link(caller, [new(24, 14, 16, 14, 28, [new ShaderCallTarget(0x2000, 0, callee)])],
            runtimeArguments: true);
        var request = Request(linked, bindlessHeap: true, callRecords: true);
        var use = Assert.Single(request.CallRecordImageByMemoryIndex).Value;
        Assert.True(BindlessHeap.HoldsIntegers(use.Slot), $"slot={use.Slot}");
        var source = Compile(request).Source;
        Assert.Contains("texture2d<uint", source, StringComparison.Ordinal);
        Assert.Contains("texture2d<int", source, StringComparison.Ordinal);
        if (SkipWithoutMetalDevice()) return;

        const uint element = BindlessHeap.CallRecordFirstIndex;
        uint[] texel = numericClass == ImageNumericClass.Uint ? [0x7, 0x80000001, 0x12345678, 0xFFFFFFFF] : [0xFFFFFFF9, 0x80000000, 0x42, 0x7FFFFFFF];
        var format = numericClass == ImageNumericClass.Uint ? MetalNative.PixelFormatRgba32Uint : MetalNative.Rgba32SintPixelFormat;
        var texture = MetalNative.CreateTexelTexture(format,
            BindlessHeap.ImageClasses[use.Slot].Dimension == ImageDimension.Dim2D ? MetalNative.TextureType2D : MetalNative.TextureType2DArray, 1, texel);
        var elements = new byte[BindlessHeap.ArrayCount * BindlessHeap.Capacity * Gen5MslArgumentLayout.HeapElementBytes];
        BinaryPrimitives.WriteUInt64LittleEndian(elements.AsSpan((int)(((uint)use.Slot * BindlessHeap.Capacity + element) * 8)),
            MetalNative.ResourceId(texture));
        var feedbackWords = new uint[BindlessHeap.FeedbackWords];
        feedbackWords[BindlessHeap.MarkWords + (element >> (int)BindlessHeap.ClassWordShift)] |=
            BindlessHeap.ClassCode(numericClass) << (int)((element & (BindlessHeap.ClassesPerWord - 1)) * BindlessHeap.ClassBits);
        var recordWords = new uint[] { 1, 0x40, 0, 0, 5, 0, 0, 0 };
        recordWords[5 + use.Field] = element;
        var records = MetalNative.CreateWordTable(DescriptorBindingKind.CallRecords, recordWords);
        var run = new Run(linked, bindlessHeap: true, callRecords: true);
        run.Dispatch(tables: [records],
            heap: (MetalNative.CreateHeapElements(elements, [texture]), MetalNative.CreateHeapFeedback(feedbackWords)));
        for (var component = 0; component < 4; component++)
        {
            Assert.Equal(texel[component], run.Word(component * 4));
        }

        Assert.All(run.Faults, word => Assert.Equal(0u, word));
    }

    // The translation cache keeps a linked program's argument layout: its call record, call mark and call fault fields
    // read back as written (Gen5MslShaderPayload format 2).
    [Fact]
    public void ALinkedTranslationReadsBackItsCallFields()
    {
        var caller = Program(MoveScalar(0, 14, 0x2000), MoveScalar(4, 15, 0),
            Vop2(8, "VAddU32", 8, Operand(0x40), Gen5Operand.Vector(0)),
            new Gen5ShaderInstruction(16, Gen5ShaderEncoding.Vop1, "VReadfirstlaneB32", [0u], [Gen5Operand.Vector(8)], [Gen5Operand.Scalar(16)], null),
            MoveScalar(20, 17, 0),
            Sop1(24, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
            BufferAccess(28, "BufferStoreDwordx4", ResultRegister, dwords: 4, vectorData: 4), EndProgram(36));
        var callee = Program(ScalarLoad(0, 16, 20, 8), ScalarLoad(8, 16, 28, 4, immediateOffset: 32),
            Vop1(16, "VMovB32", 0, Operand(0x3F000000)), Vop1(24, "VMovB32", 1, Operand(0x3F000000)),
            ImageSample(32, 20, 28),
            Sop1(40, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        var perRecord = Compile(Request(Gen5ShaderCallLinker.Link(caller, [new(24, 14, 16, 14, 28, [new ShaderCallTarget(0x2000, 0, callee)])])));
        var runtime = Compile(Request(Gen5ShaderCallLinker.Link(caller, [new(24, 14, 16, 14, 28, [])], runtimeArguments: true),
            bindlessHeap: true, callRecords: true, callMarks: true));
        var recorded = Compile(Request(Gen5ShaderCallLinker.Link(caller, [new(24, 14, 16, 14, 28, [new ShaderCallTarget(0x2000, 0, callee)])],
            runtimeArguments: true), bindlessHeap: true, callRecords: true));
        Assert.Contains(perRecord.ArgumentLayout!.Fields, field => field.FieldKind == MslArgumentFieldKind.CallFaults);
        Assert.Contains(runtime.ArgumentLayout!.Fields, field => field.Name == Gen5MslArgumentLayout.CallMarksName);
        Assert.Contains(recorded.ArgumentLayout!.Fields, field => field.Name == Gen5MslArgumentLayout.CallRecordsName);
        foreach (var shader in new[] { perRecord, runtime, recorded })
        {
            var payload = Gen5MslShaderPayload.Write(shader, ShaderStage.Compute, Hash);
            Assert.True(Gen5MslShaderPayload.TryRead(payload, out var read, out _, out _, out var error), error);
            Assert.Equal(shader.ArgumentLayout!.Fields, read.ArgumentLayout!.Fields);
            Assert.Equal(shader.Source, read.Source);
        }
    }
}
