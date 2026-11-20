// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Linked shader calls end to end on the Metal host: a compute program loads a function's address and argument from a
/// table its user data names and calls it with S_SWAPPC_B64, and the callee's result lands after the call returns.
/// <para>
/// With run-time call records (the default) the program cache follows upstream's sequence (ShaderCalleeTiers): a caller
/// first links the callees an earlier session kept for it, none in a new one, so its first dispatches run at once and each
/// call to a callee the program lacks marks the callee and returns (the store sees the caller's own value). The host reads
/// the marks once their submission completes (CallMarkSnapshots); a mark newer than the naming makes the callee called, the
/// caller is linked again with it once its calls settle, and the new program serves from the dispatch after its pipeline is
/// ready. Nothing is held or re-run meanwhile; the skipped calls are counted (MarkedCalls). With a copy linked per record
/// (SHARPEMU_RUNTIME_CALL_RECORDS=0) the first dispatch already runs the callee.
/// </para>
/// </summary>
public sealed class LinkedCallSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint BufferFormat = 20;

    // Generous: the relink waits a few completed submissions, then links, translates and compiles in the background, and
    // the device is shared with other test runs.
    private static readonly TimeSpan RelinkBound = TimeSpan.FromSeconds(60);

    // s_buffer_load_dwordx4 s[16:19], s[0:3], 0 (the record: function low/high, argument low/high); s_waitcnt;
    // s_mov_b32 s14..s17 from s16..s19; s_swappc_b64 s[14:15], s[14:15]; buffer_store_dword v1, off, s[4:7], 0; s_endpgm.
    private static readonly uint[] Caller =
    [
        0xF4280400, 0xFA000000, 0xBF8C0000, 0xBE8E0310, 0xBE8F0311, 0xBE900312, 0xBE910313, 0xBE8E210E,
        0xE0700000, 0x80010100, 0xBF810000,
    ];

    // v_mov_b32 v1, s16 (the argument's low word); s_setpc_b64 s[14:15].
    private static readonly uint[] Callee = [0x7E020210, 0xBE80200E];

    // A second function: v_mov_b32 v1, 0x100; v_add_nc_u32 v1, s16, v1; s_setpc_b64 s[14:15].
    private static readonly uint[] OtherCallee = [0x7E0202FF, 0x00000100, 0x4A020210, 0xBE80200E];

    // Run-time records: the first dispatch runs the caller without the callee (its call marks it and returns), the marks
    // reach the program cache, the caller is linked again with the callee, and a later dispatch stores the callee's value.
    // A second host over the same linked cache directory (a later session) links the kept callee into the caller's first
    // program, so its first dispatch already runs it.
    [Theory]
    [InlineData(false, 0x77u, 0x77u)]
    [InlineData(true, 0x55u, 0x155u)]
    public void ARuntimeRecordCallReachesItsCalleeOnceItsMarksRelinkTheCaller(bool other, uint argument, uint expected)
    {
        var directory = TemporaryDirectory();
        try
        {
            using (var rig = CreateRig())
            {
                if (rig is null) return;
                rig.Host.LinkedProgramCacheDirectory = directory;
                var scene = new CallScene(rig, other ? OtherCallee : Callee, argument);

                var first = scene.Dispatch();
                Assert.Equal(0u, first[0]);
                Assert.Equal(0xDEADBEEFu, first[1]);

                var dispatches = scene.DispatchUntil(expected, RelinkBound, Output);
                Assert.True(dispatches > 0, $"the callee's value never landed within {RelinkBound.TotalSeconds} s");
                Assert.True(rig.Host.MarkedCalls > 0, "the skipped calls were not counted");
                Assert.True(rig.Host.CallMarkSnapshots > 3, "the callee became called before a mark newer than its naming");
                Assert.Equal(expected, scene.Dispatch()[0]);
                Assert.Equal(0, rig.Host.UnlinkedCalls);
                Assert.Empty(rig.Fatals);
            }

            Assert.True(File.Exists(Path.Combine(directory, "callees.bin")), "the relinked caller's callees were not kept");
            using (var later = CreateRig()!)
            {
                later.Host.LinkedProgramCacheDirectory = directory;
                var scene = new CallScene(later, other ? OtherCallee : Callee, argument);
                var words = scene.Dispatch();
                Assert.Equal(expected, words[0]);
                Assert.Equal(0xDEADBEEFu, words[1]);
                Assert.Equal(0, later.Host.MarkedCalls);
                Assert.Equal(0, later.Host.UnlinkedCalls);
                Assert.Empty(later.Fatals);
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    // A copy linked per record (the host refuses run-time records, as SHARPEMU_RUNTIME_CALL_RECORDS=0 does): the first
    // dispatch waits for its linked program and runs the callee, and nothing is marked.
    [Theory]
    [InlineData(false, 0x77u, 0x77u)]
    [InlineData(true, 0x55u, 0x155u)]
    public void APerRecordLinkRunsTheCalleeAtTheFirstDispatch(bool other, uint argument, uint expected)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.RuntimeCallRecordsEnabled = false;
        var scene = new CallScene(rig, other ? OtherCallee : Callee, argument);

        var words = scene.Dispatch();

        Assert.Equal(expected, words[0]);
        Assert.Equal(0xDEADBEEFu, words[1]);
        Assert.Equal(0, rig.Host.MarkedCalls);
        Assert.Equal(0, rig.Host.UnlinkedCalls);
        Assert.Empty(rig.Fatals);
    }

    // The caller's first program is still compiling at its first dispatch (the compiler is held): the dispatch waits for it
    // within SHARPEMU_LINKED_COMPILE_WAIT_MS and runs once it is ready, instead of being left out; the relink that follows
    // compiles behind the same gate and brings the callee in.
    [Fact]
    public void ADispatchWhoseLinkedModuleIsStillCompilingRunsOnceItIsReady()
    {
        // Declared before the rig, so it is disposed after it: a compiler thread may still reach this gate until the
        // rig's host has joined its threads, and waiting on a disposed gate throws on that thread and ends the process.
        using var gate = new ManualResetEventSlim(false);
        var directory = TemporaryDirectory();
        try
        {
            using var rig = CreateRig();
            if (rig is null) return;
            rig.Host.LinkedProgramCacheDirectory = directory;
            rig.Host.PipelineCompiler.Gate = gate;
            var scene = new CallScene(rig, Callee, 0x99);
            using var release = Task.Delay(TimeSpan.FromMilliseconds(500)).ContinueWith(_ => gate.Set());

            var first = scene.Dispatch();

            Assert.True(gate.IsSet, "the dispatch ran before its module was released");
            Assert.Equal(0u, first[0]);
            Assert.Equal(0xDEADBEEFu, first[1]);
            Assert.True(scene.DispatchUntil(0x99, RelinkBound, Output) > 0, "the callee's value never landed");
            Assert.Equal(0, rig.Host.UnlinkedCalls);
            Assert.Empty(rig.Fatals);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static string TemporaryDirectory()
    {
        var directory = Path.Combine(Path.GetTempPath(), "sharpemu-linked-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    // One caller dispatch over a table naming the function and its argument; each dispatch refills the result first.
    private sealed class CallScene
    {
        private readonly SceneRig _rig;
        private readonly ulong _caller;
        private readonly ulong _table;
        private readonly ulong _result;

        public CallScene(SceneRig rig, uint[] function, uint argument)
        {
            _rig = rig;
            var address = rig.Program(function);
            _table = rig.Memory.Allocate(16, 256);
            rig.Memory.Write<uint>(_table, [(uint)address, (uint)(address >> 32), argument, 0]);
            _result = rig.Memory.Allocate(64, 256);
            _caller = rig.Program(Caller);
        }

        public uint[] Dispatch()
        {
            _rig.Memory.Write<uint>(_result, Enumerable.Repeat(0xDEADBEEFu, 16).ToArray());
            var state = new SceneState(1, 1);
            state.Context.Clear();
            state.Uconfig.Clear();
            state.ComputeShader(_caller, 1, 1, 1,
                [.. Descriptors.Buffer(_table, 4, 4, BufferFormat), .. Descriptors.Buffer(_result, 4, 16, BufferFormat)], true, false, false, 0, 0);
            _rig.Submit(state.Emit(new ScenePm4()).DispatchDirect(1, 1, 1));
            return _rig.Read<uint>(_result, 2);
        }

        // Dispatches (each submission completes before the next, as frames do) until the first word is the expected value;
        // returns how many dispatches that took, or 0 past the bound. Every interim dispatch runs the caller (no 0xDEADBEEF).
        public int DispatchUntil(uint expected, TimeSpan bound, ITestOutputHelper output)
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            for (var dispatches = 1; clock.Elapsed < bound; dispatches++)
            {
                var words = Dispatch();
                if (words[0] == expected)
                {
                    Assert.Equal(0xDEADBEEFu, words[1]);
                    output.WriteLine($"landed after {dispatches} dispatches, {clock.ElapsedMilliseconds} ms, " +
                        $"{_rig.Host.CallMarkSnapshots} mark snapshots, {_rig.Host.MarkedCalls} marked calls");
                    return dispatches;
                }

                Assert.Equal(0u, words[0]);
                Thread.Sleep(5);
            }

            return 0;
        }
    }
}
