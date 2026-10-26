# Two-process split: wiring contract (Phase 2)

Status: implemented behind the flag, 2026-10-24 (SharpEmu.GpuSplit; transport protocol 2). Design: `SPLIT-PLAN.md`
(repo-external draft), Phase 0 rates: commit `fc963f02`, shared memory: `docs/macos/split-shared-memory.md`,
transport: `src/SharpEmu.GpuRemote`.

Flag: `SHARPEMU_GPU_PROCESS=1`. Off by default; nothing changes without it.

## Roles

| Process A (x86-64 under Rosetta, `SharpEmu.CLI`) | Process B (native arm64, `SharpEmu.GpuHost`) |
|---|---|
| Guest CPU, HLE exports, kernel event queues, VideoOut state machine | GPU worker: command interpreter (`SharpEmu.Libs` GPU engine) + Metal v2 render host + image cache |
| Owns guest memory, page protections (`mprotect`), the write-fault signal handler | Guest memory through the shared backing (`GuestReplica` at guest address + K, `GuestViewTable`) |
| `GuestGpuMemory` + `PageGuard` used by the fault handler (no watches of its own) | Its own `GuestGpuMemory` over `RemoteGuestAddressSpace`; its `PageGuard` asks A to apply protections |
| `SplitGpuSession`: `RemoteGpuBackend` (`GuestGpu.Register`), `RemoteImageStore`, `RemoteGpuQueueRelay` | `SplitGpuHost` (the `GpuProcessClient` service): the real `MetalV2Backend`, called from the loops |
| Input from B's events into the HLE pad/keyboard state (`HostWindowInput`) | The window, `CAMetalLayer`, present loop, input capture (AppKit on the main thread) |

A starts B at the first read of `GuestGpu.Current` (by then the runtime has made the shareable backing):
`SplitGpuComposition` replaces `MetalGpuBackend.Register` in `GpuComposition`. B composes the Metal backend once A's
`SplitStart` arrives (`GpuHostMain.ComposeMetal`). `SHARPEMU_GPU_HOST_BACKEND=probe` composes a test backend instead
(no Metal, no window; `SplitCrossProcessTests`).

## Measured rates that shape it (Phase 0)

GTA street / Wukong intro: write faults 7-87/s and 1-3/s (synchronous is fine); managed writes ~1.2M/s and
150-290K/s (must be filtered locally in A); shader-header lookups ~100K/s and 9-19K/s (must be local in B);
page watches 100-2600/s (batch); submissions 30 and 200-350/s; SubmitDone/flips ~15/s; mapping changes <= 1/s.

## Messages

All records are little-endian blittable structs (`GpuSplitMessages.cs`) on the transport rings; synchronous calls use
`SyncCallTable` both ways: A → B (`SyncCalls`, signal-safe on the A side) and B → A (`ReverseSyncCalls`, ordinary B
threads). Add new kinds at the end of the enums; bump `GpuRemoteProtocol.Version` (now 2).

### Start

| Record | Fields | Semantics |
|---|---|---|
| `SplitStart` (A → B) | view log sequence, window options, HDR flag, service name, title id | B connects the guest memory (`SplitGuestMemory.Connect`), builds its address space and `GuestGpuMemory`, installs the seams, composes the backend. The title id is fixed before guest code runs and is answered locally in B from then on. |
| `SplitReady` (B → A) | status, replica base, error text | A attaches its proxies only after this. A throws `GpuProcessException` with B's error text otherwise. |

On attach A replays every span registered so far (`GuestGpuMemory.AttachSpanObserver`), every shader header and every
fused program.

### A -> B, asynchronous (ring)

| Message | Fields | Semantics |
|---|---|---|
| `SubmitCommandStream` | queue, address, dwordCount, submissionId, viewLogSequence | B first `GuestReplica.Sync(viewLogSequence)`, then `SubmitCommandStream`. |
| `FlipRequest` | videoOutHandle, displayBufferIndex, address, width, height, pitch, hdrRequested, flipRequestId | `TrySubmitGuestImage`. A takes it as accepted when it sends; a refusal comes back as a `FlipEvent(Refused)` record, which A applies as `MarkFlipPresented` (the order against the export's own `CompleteFlip` does not matter). |
| `CpuWrite` | address, size | A managed write that has landed, only for ranges the GPU-interest summary marks. B calls its stores' `MarkCpuWriteLanded` in ring order, before any later submission. |
| `ShaderHeader` | codeAddress, headerAddress | Mirrors `AgcExports._shaderHeadersByCode` into B (lookups stay local in B). |
| `FusedProgram` | entry, entryHeader, continuation, continuationHeader (zeros: unregistered) | Mirrors `Gen5ShaderTranslator`'s fused-program registry, which B's command streams read, keyed by B's address space. |
| `Control` | EnsureStarted(w,h), HideSplashScreen, RequestClose, CountShaderCompilation | One-way calls of `IGuestGpuBackend`. RequestClose also requests B's session shutdown. |
| `HostFrame` | width, height, total, offset + piece | `Submit` of a CPU frame (splash, video), in pieces of 256 KiB written back to back. |
| `GamepadOutput` | rumble, trigger rumble, adaptive triggers, lightbar | To the pad, which lives with B's window. |
| `VideoOptions` | window options | Options configured after the start. |
| `FrameBoundary` | ticket | The ordering half of a split `SubmitDone` (`BeginSubmitDone`): B queues the boundary behind the submissions before it and keeps its wait for `WaitFrameBoundary`. A sends it under its submission gate and waits after releasing it. |

### A -> B, synchronous (`SyncCalls`)

Calls marked *barrier* carry the number of split records A had written; B serves them once its ring thread handled that
many, so they come after A's earlier records.

| Opcode | Request | Response | Semantics |
|---|---|---|---|
| `SubmitDone` (barrier) | - | IdleOutcome | B's `SubmitDone`: returns at the frame boundary. With A's session shutdown requested A sends `RequestClose` and answers `Cancelled` itself. |
| `WaitFrameBoundary` (barrier) | ticket | IdleOutcome | The waiting half of a split `SubmitDone`: runs the wait B kept for the `FrameBoundary` record of that ticket. |
| `BracketBegin` | kind, barrier (AfterPendingCommandStreams only) | bracket id | `GuestGpuMemory.RunMappingChange` / `Unregister` in A (kind MappingChange: the relay's `TryRunAfterPendingWork`) and `RunAfterPendingCommandStreams` (kind AfterPendingCommandStreams). B parks its GPU worker (`WorkerBracket`: B's own `RunMappingChange` / `RunAfterPendingCommandStreams` hand the worker a park loop); A then runs the work itself, as the GPU queue thread (`RemoteGpuQueueRelay.IsGpuQueueThread`). |
| `BracketEnd` | bracket id, viewLogSequence | - | B syncs its replica that far on the parked worker and lets it go. |
| `Fault` | kind, address, size | handled, restoration version | From A's signal handler (`RemoteImageStore.MarkCpuWrite`, allocation-free). B runs its stores' fault path, then waits until A applied every protection B requested so far: the guest retries the access as soon as this answers. Every write fault crosses (the page faulted because B protected it). When no store handled it, B answers its guard's write-restoration version of the page: a watch removed before the fault reached the store. A retries the store once per version when the page is writable (one process's `TryRetryReleasedWrite`, from B's guard). |
| `CpuWrite` | address, size | handled | A managed write before it lands, when the summary marks it: B's `MarkCpuWrite` (the cache publishes a GPU-written image before the write lands over it). From A's reverse-call threads and from inside a bracket it is sent as a `CpuWrite` record instead (there B's worker cannot take the writeback: the in-process cache's rule for a store from the worker). No barrier. |
| `SpanRegister` / `SpanProtect` / `SpanUnregister` | address, size, protection, bracket id | (handled) | B's `GuestGpuMemory` follows A's spans and permission ledger (`IGuestSpanObserver`, `RemoteImageStore.Unregister`). B's page guard owns the host protections of every span: A's `Register` / `NoteProtected` apply none of their own when the call succeeds (A's watch-less guard would hand back, for a moment, rights B's watches took); on a failed call A applies the plain guest protection. An unregistration inside a bracket runs on B's parked worker. B waits for the protections it requested before it answers, so they land before A unmaps, remaps or lets the guest go on. |
| `Query` | which | value | Performance counters, HDR output support, mesh pipelines. |

`IsGpuGuestImageAvailable` and `RegisterKnownDisplayBuffer` are constants in Metal v2 and are answered in A.
`TryCompileProgram` and `GetDepthOnlyFragmentShader` run only in B (the command streams that use them run there).

### B -> A, asynchronous (ring)

| Message | Fields | Semantics |
|---|---|---|
| `ProtectRequest` | sequence, address, size, guest access, access, flags | B's `PageGuard` (watch arm/drop, ledger changes). A applies it at once on its inbound thread (`TryApplyDerivedProtection`, skipping memory A no longer registers) and raises the shared protection counter (below). A request that takes rights away waits in B until it is applied; one that gives them back does not (the fault path waits instead). |
| `Interrupt` | eventId, contextId | `EndOfPipeEvents.Deliver` in A, for an interrupt raised on B's ring thread. Elsewhere interrupts are calls (below). |
| `FlipEvent` | kind, flipRequestId | Flip steps sent from B's ring thread (a refused `FlipRequest`). Elsewhere flip steps are calls (below). |
| `VideoOutNotice` | kind, text | CancelOutstandingFlips, PresentationWindowClosed, ReportPresentedFrame, SelectedGpuName. |
| `Input` | focus, pressed keys, gamepad state and name | B polls its `HostWindowInput` every 4 ms and sends it when it changed; A applies it to its own `HostWindowInput`. |

A's inbound thread applies protections itself and hands everything else to an event thread, in order, so a slow event
(an event-queue lock, the video-out lock) never holds up a protection.

### B -> A, synchronous (`ReverseSyncCalls`)

What B's GPU code asks of A's video-out state (`VideoOutBackendServices.Bridge` in B is `RemoteVideoOutBridge`):
`TryReserveFlip` (with A's HDR flag), `IsFlipDone`, `CanPresentFlip`, `NextFlipPresentTimestamp`, `TryGetDisplayBuffer`,
`GetFlipEventCount`, `ReleaseNoBufferFlip`, `IsFlipPresentationPending`, `IsHdrOutputRequested`, `QueryString`
(window title), `FlipEvent` (CompleteOnGpu, Presented, Discarded, Cancelled, Completed: applied before the call
returns, so the presenter's next question sees it) and `TriggerInterrupt` (B's `EndOfPipeEvents.Redirect`): the GPU's
completion thread raises interrupts and flip steps in the order the GPU finished them, and as calls they reach the guest
in that order. `Stopwatch` timestamps cross unchanged: both processes read the same clock. A serves them on four
threads. A reverse call is never retried (a flip reservation must not happen twice), so it waits as long as A's own
long calls do.

### The rule that keeps the threads apart

B's ring thread never calls into A: a guest thread holding the video-out lock may be blocked writing into a full A -> B
ring, and a call from the ring thread would need that lock. So what the ring thread runs that would ask A (the title id,
the HDR flag, the GPU name) is answered from copies or sent one way, and a flip refused there is a record. The
bridge refuses (and reports) a call from the ring thread.

## Memory

- B attaches the guest backing (`MacOSHost.ConnectGuestMemory`), maps it as a mirror, maps the `GuestViewTable`
  read-only and attaches a `GuestReplica` at guest address + K (`SplitGuestMemory`). `RemoteGuestAddressSpace`
  implements `ICpuMemory`, `IGuestBackedSpace`, `IGuestAddressSpace` and `IGuestDerivedProtectionAddressSpace` over it:
  bytes through the mirror via the table (`GuestMirrorTranslator`), `TryResolveBackingAlias` = mirror addresses,
  `BackingAliasRange` = the mirror, `TryResolveBackedHostView` = the replica (synced first), protections = requests to A,
  allocation members unsupported (the GPU side never allocates guest memory), `ProtectionPageSize` = 4 KiB (A's).
  Writes B's GPU code makes go through B's own `GuestGpuMemoryHook`, as a write through the guest space does in one
  process.
- B follows the view log at every submission, every bracket end, every host view, and every 100 ms: the log also
  carries every protection A applies, and a replica a whole ring behind remaps its whole window.
- In split mode, A maps module data pages (`.rodata`, `.data`, `.bss`), the guest allocator arena with the VideoOut
  labels, and HLE data from the backing (`SplitBackingLayout`, see `split-shared-memory.md`). Code, stacks, TLS and the
  HLE libc heap stay private to A. A B access to private memory fails, and the first 16 are reported with the address
  (`[GPU-SPLIT][WARN] GPU read of guest 0x... failed: not in the shared backing`). In A, `SplitPrivateAccess` names any
  private range the GPU path reads.

### The application block (`SplitSharedState`)

- **Protection acknowledgement** (offset 0, a `SharedCounter`): A raises it to a `ProtectRequest`'s sequence once that
  request and every one before it is applied; B waits on it. (The first contract had a `ProtectAck` ring record; a
  shared counter means B's waits never depend on A's ring writers, which may be blocked on a full ring.)
- **GPU-interest summary** (offset 4096, 32 MiB of address space, touched only where watches are): one bit per 4 KiB
  page below 1 TiB, set while the page carries a `PageGuard` watch in B (`IPageWatchObserver`, under the guard's block
  lock; 16 words per 4 MiB block). B sets a bit before the watch's protection is requested, with a full barrier; A
  filters managed writes with it, so a write next to a watched page (command buffers sit beside the images they draw
  to) does not block. `SHARPEMU_SPLIT_STATS=1` adds `ManagedWriteInterest` (writes that hit a set bit),
  `ManagedWriteCall` (the blocking `CpuWrite` calls, with their average time) and `ProtectWait` (confirmation waits,
  average time) to the `[SPLIT-X]` line.

## Teardown and failure

- B exits when A exits (parent watch). A's `Shutdown` record on a clean session end exits it with code 0.
- When B dies, the transport fails every pending and later call promptly; `SplitGpuSession` detaches its proxies (the
  memory manager stops waiting for a GPU worker), gives every span its plain guest protection back (B's watches would
  otherwise turn guest stores into faults nothing resolves), and stops the guest through the window-closed shutdown,
  after `[GPU-SPLIT][ERROR] The GPU process exited unexpectedly. The guest is stopped.`
- Every wait has a timeout and checks the peer state: ring writes 10 min (reported after 30 s), frame boundary,
  brackets, span changes and reverse calls 10 min (B's own waits inside them 9 min, so B answers first), faults and
  managed writes 60 s, protection waits 30 s. A park nobody ends lets B's worker go after 11 min.
- B's ring thread hands `RequestClose` (which joins the GPU worker) to a thread of its own.
- B's console is A's console (inherited); B tees it into `SHARPEMU_GPU_HOST_LOG`, which the CLI sets to
  `<log>.gpuhost.log` when it runs with `--log-file`.

## Watch protections wait once

A watch change (`AddWatch`, `AddWatchMask` and their removals) sends every protection it needs and waits for A's
acknowledgement once, after the page guard released its block locks (`IProtectionBatchSpace`; the shared counter is
cumulative, so the last request's number covers the rest). A change that found its pages already watched by another
thread, whose protection that thread has not confirmed yet, waits for that thread's number too. Protections made
outside a watch change (ledger changes, `TryProtect`) still wait where they are made. `SHARPEMU_SPLIT_BATCH_PROTECT=0`
waits at each protection, as before.

## Not carried yet

- Host movies (`MoviePresentationServices`: frames A's decoder hands the presenter) do not reach B's presenter.
- `TraceGpuFlip` (diagnostics) does nothing in B.
- Private guest memory (above).
- Fast paths: `CanPresentFlip`/`NextFlipPresentTimestamp` cost a round trip per presenter pass (mirror the pacing state); managed
  writes under a watch are a call each.
- Known windows, not closed yet: managed writes made inside a bracket reach B as records and may be handled after the
  worker resumes; the shutdown no longer orders B's watch teardown before A releases its page guard (A's guard holds no
  watches, and A acknowledges late requests without applying them).
