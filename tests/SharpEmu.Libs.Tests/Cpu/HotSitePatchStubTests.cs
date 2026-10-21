// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Iced.Intel;
using SharpEmu.Core.Cpu.Native;
using Xunit;

namespace SharpEmu.Libs.Tests.Cpu;

// The stubs a hot site of an instruction Rosetta does not translate is rewritten into. The test host cannot run x86 code, so
// these pin the encoding to bytes that were checked under Rosetta against the instruction
// definitions (ADX: 200,000 random operand and flag combinations per form, every flag compared;
// SSE4a: 100,000 per form against Sse4aBitFieldEmulator, both XMM halves and the flags compared).
public sealed class HotSitePatchStubTests
{
    [Theory]
    [InlineData("66480F38F6C8",
        "488D6424809C4813C89C52488B5424084883E2014881642410FEFFFFFF48095424105A488D6424089D488DA42480000000C3")]
    [InlineData("66480F38F6442408",
        "488D6424809C48138424900000009C51488B4C24084883E1014881642410FEFFFFFF48094C241059488D6424089D488DA42480000000C3")]
    [InlineData("F3480F38F616",
        "488D6424809C50488B442408480FBAE00B584813169C50488B4424084883E00148C1E00B4881642410FFF7FFFF480944241058488D6424089D488DA42480000000C3")]
    public void TheStubEncodesToTheVerifiedSequence(string site, string expected)
    {
        var decoder = Decoder.Create(64, new ByteArrayCodeReader(Convert.FromHexString(site)));
        decoder.IP = 0x1000;
        decoder.Decode(out var instruction);

        Assert.True(DirectExecutionBackend.TryBuildAdxStub(in instruction, out var stub));
        stub.Add(Instruction.Create(Code.Retnq));
        var writer = new ByteWriter();
        Assert.True(BlockEncoder.TryEncode(64, new InstructionBlock(writer, stub, 0x200000), out var error, out _), error);
        Assert.Equal(expected, Convert.ToHexString(writer.Bytes.ToArray()));
    }

    [Theory]
    [InlineData("660F78C10810", "488D6424809C505166480F7EC848C1E81048B9FF000000000000004821C866480F6EC859589D488DA42480000000C3")]
    [InlineData("F20F78C11008", "488D6424809C50515266480F7EC848B9FFFF0000000000004821C848C1E00866480F7EC148BAFF0000FFFFFFFFFF4821D14809C166480F6EC15A59589D488DA42480000000C3")]
    public void TheSse4aStubEncodesToTheVerifiedSequence(string site, string expected)
    {
        var decoder = Decoder.Create(64, new ByteArrayCodeReader(Convert.FromHexString(site)));
        decoder.Decode(out var instruction);

        Assert.True(DirectExecutionBackend.TryBuildSse4aStub(in instruction, out var stub));
        stub.Add(Instruction.Create(Code.Retnq));
        var writer = new ByteWriter();
        Assert.True(BlockEncoder.TryEncode(64, new InstructionBlock(writer, stub, 0x200000), out var error, out _), error);
        Assert.Equal(expected, Convert.ToHexString(writer.Bytes.ToArray()));
    }

    [Fact]
    public void TheRegisterFormOfExtrqIsLeftToTheFaultHandler()
    {
        // extrq xmm1, xmm2: the fields come from xmm2 at run time.
        var decoder = Decoder.Create(64, new ByteArrayCodeReader(Convert.FromHexString("660F79CA")));
        decoder.Decode(out var instruction);
        Assert.False(DirectExecutionBackend.TryBuildSse4aStub(in instruction, out _));
    }

    [Fact]
    public void AFormThatUsesRspAsAnOperandIsLeftToTheFaultHandler()
    {
        // adcx rsp, rax
        var decoder = Decoder.Create(64, new ByteArrayCodeReader(Convert.FromHexString("66480F38F6E0")));
        decoder.Decode(out var instruction);
        Assert.False(DirectExecutionBackend.TryBuildAdxStub(in instruction, out _));
    }

    private sealed class ByteWriter : CodeWriter
    {
        public List<byte> Bytes { get; } = [];

        public override void WriteByte(byte value) => Bytes.Add(value);
    }
}
