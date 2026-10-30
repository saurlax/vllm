# Portable DoSA research prototype

This experimental benchmark is not a serving backend. It does not enable DoSA
on H100 or RTX 5090 model inference. The original `dosa-main` implementation is
unchanged. The portable kernel currently has runtime validation on H100 only.

The prototype retains all positive heads and bounds omitted negative heads by
their weighted aggregate. It does not yet implement the positive-head projection
or joint residual certificates from the complete DoSA method.

## Exact input domain

The admission check requires D=128, H=32 or 64, integer Q/K values in [-2, 2],
unit key scales, and weights of magnitude 1/8 or 1/4. The first H/2 weights
must be positive and the rest negative. Thresholds must be finite and exactly
representable in FP32. Other inputs are rejected; arbitrary model activations
are not supported.

Within this domain, FP16 exactly represents the input values and the negative
aggregate. Dot products and weighted sums remain bounded binary fractions that
fit in FP32 without rounding. The tests compare all upper bounds and candidate
scores against an FP64 reference and compare final selected score values with
the native SM90 DeepGEMM implementation. Equality at the threshold survives.

Atomic candidate compaction has unspecified order. Valid TopK score equality
does not establish identical key IDs at tied boundaries or bitwise-identical
model outputs. Strict native tie-breaking and a general numerical certificate
remain unfinished. There is no production dispatch to this research code.

## Reproduce on H100

Use the pinned checkout and its CUDA environment described in `dosa.md`:

```bash
.venv/bin/python benchmarks/kernels/benchmark_dosa_portable.py \
    --output-dir results/dosa-portable
```

The benchmark runs H32/H64 on independent and artificially correlated negative
heads, Q=16 and K=131072. These are synthetic inputs, not model activations.

Timings are recorded using synchronized host wall time: native stock,
the initial host-loop prototype, GPU compaction, and experimental fused paths.
They include
seed scoring, admission checks, allocations, screening, compaction, refinement
and TopK. FP64 verification is outside these timings. Screening-only CUDA event
samples are separately labeled and must not be interpreted as full-pipeline
speedups. Model projections, KV gathering and complete-model TTFT are excluded.

Admission is a fused GPU scan followed by one host check; it rejects invalid
Q/K encodings, incorrect weight signs/magnitudes, and NaN, infinite, or
non-FP32-representable thresholds. Admission is charged to every timed call.
Plan construction also uses a fused GPU kernel. Experimental paths fuse
screening and atomic candidate packing and compute seed cutoffs using native
TopK followed by a minimum reduction. The cutoff is checked against Torch TopK.
Candidate refinement uses native FP8 MMA with FP32 accumulation. Split screening
also uses FP8 for unchanged positive directions, with a separate FP32 reduction
for the negative aggregate. These operations preserve exactness in the admitted
domain. Final candidate IDs are mapped directly with int32 indices.

The dense Triton ablation performs no DoSA pruning. Its score buffer is checked
against every native score, and it does not repeat DoSA admission (like stock).
This baseline distinguishes backend improvements from pruning improvements.
Stock comparisons alone do not establish an algorithmic DoSA speedup.

Use `--rows 128` to test a larger query block and `--seed 42` for held-out
synthetic inputs. Results must retain the query/key dimensions and seed; a
small-query result cannot establish full-prefill performance.

Raw samples and source snapshots are preserved for every Modal run. Alternating
stock/portable execution order is additionally used for final comparisons, to
reduce timing bias from GPU clock drift. Profiling is outside timed samples.

No RTX 5090 runtime validation or architecture-specific performance claim has
been established. The kernels avoid SM100-specific TCGEN05/TMEM instructions,
but their correctness and performance must still be qualified on each device.

## Prepared replay experiments

`HopperGraphPipeline` uses fixed input tensor addresses and preallocated buffers.
Every replay first runs GPU domain admission and checks its result on the host;
unsupported contents are rejected before computation. Changed tensor addresses,
shapes, strides or dtypes are rejected. Tests modify valid input contents in
place, compare the new results with native scores, reject an unsupported FP8
value and a changed address, and verify recovery after rejection.

The captured computation reruns seed scoring, builds the plan, screens and packs
remaining keys, refines candidates and runs final TopK. The seed's exact TopK
scores are retained within that invocation. Other seed entries cannot improve
the final TopK score multiset; equal-score key identity remains unspecified.
Seed keys are excluded from subsequent screening and seed winners from subsequent
refinement. Positive-head contributions computed during screening are retained,
so refinement only evaluates negative heads. Persistent refinement CTAs loop over
valid candidate blocks. Ordinary-call helpers initialize unused slots to negative
infinity. The prepared pipeline fully overwrites each active prefix and leaves
storage beyond device counts unspecified; native TopK respects those row ends.
Tests poison scores with positive infinity and IDs with -1 before replay, then
check every active candidate score, uniqueness and final TopK. This permits
removing the full-capacity clear without relying on previous contents.

Stock and dense ablations are also captured. Prepared replay timings exclude
capture, workspace allocation and input copies for all paths, and include domain
admission and host rejection on the portable path. They must not replace the
separate ordinary-call timings that include per-call allocation. See the
[PyTorch CUDA Graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#constraints).

Experimental FP8 high/low aggregate decomposition is also tested. In the admitted
domain, the aggregate has magnitude at most 16 and spacing 1/8. Rounding it to
E4M3 leaves an exactly representable residual of magnitude at most 1/2; two FP8
dot products can therefore reconstruct the aggregate dot product exactly. This
variant was slower and is not selected by the prepared pipeline.

All current results use restricted synthetic data. In particular, the correlated
case deliberately makes negative heads identical. A win on that case does not
establish a win on arbitrary model activations, independent heads or other head
counts. The complete positive projection and joint residual certificates are
still absent. There is no real-model TTFT or TPS result.

## Fidelity requirements for the actual Hopper port

The benchmark above is an independent restricted-domain prototype. Its measured
win is not a validation of a complete port of the author's implementation.
The implementation target is the original pipeline with an SM90 backend.

| Component | Author implementation at the pinned base | Current research prototype |
| --- | --- | --- |
| Planner | Adaptive paid-head budgets, positive projection/cone and joint residual bounds | All positive heads retained; one negative aggregate bound |
| Seed selection | Call-local bootstrap, voting/carry, recent keys and logical ID permutation | First 12288 keys, retaining exact seed TopK |
| Numerical target | Admitted native SM100 MX/BF16 scoring and explicit rounding envelopes | Native SM90 FP32 scores in an exactly representable synthetic domain |
| Refinement | Replays the native scoring arithmetic over original heads | Reuses positive partial sums and evaluates negative heads separately |
| Candidate storage | Bitmasks, common/solo paged work and bounded 5-byte records | Per-query full-capacity int32 IDs and FP32 score storage |
| Selection/recovery | Boundary/tie processing, numerical and overflow recovery to native computation | Native TopK on compact positions; unsupported data rejected |
| Dispatch | Workspace/private graphs with GPU conditional nodes | Two captured graphs separated by host admission check |

Positive partial-sum reuse changes reduction order. It is exact in this
prototype's binary-fraction domain, but must not be carried into a general
floating-point port without a proof. First restore native-order full-head
candidate scoring, then qualify further reuse against the target scorer.

Preserve the author's planner, mathematical certificates, selection semantics
and recovery policy where their assumptions remain valid. Adapt rounding
envelopes to the pinned SM90 native arithmetic; copying SM100 BF16 constants
into an SM90 FP32 path does not establish correctness. Treat uncertain bounds
as keep decisions and admission/overflow failures as native fallback decisions.
The author also permits different valid IDs at equal scores; identical model
outputs are a separate, stronger claim requiring explicit tie and end-to-end
qualification.

Hardware replacement covers the complete accumulator/dataflow schedule:
TCGEN05/TMEM uses different ownership and synchronization from Hopper
WGMMA/register/shared-memory operations. H100 already has TMA, so TMA and warp
specialization should be assessed as reusable techniques rather than classified
as SM100-only features. Keep original certificate budgets initially and tune
Hopper tiles/scheduling after semantic parity is established.

## Original-component SM90 qualification

This section records an earlier component-qualification phase. Current native
FP32-target work and the subsequent Hopper FP8 precision audit are documented
in [Experimental DoSA indexer on SM90](dosa_sm90.md). The independent prototype
speedups above do not describe the current guarded native-target port.

An opt-in `_dosa_sm90_C` research extension now cross-compiles for `9.0a`.
`VLLM_DOSA_SM90_RESEARCH` defaults to `OFF`; enabling it does not install a
serving dispatch. The original `planner.cu`, `direct_abs.cu`, `seed.cu`,
`select.cu`, `post.cu`, and BF16 scoring helpers remain unchanged at the pinned
author revision. Separate namespaces prevent replacing the SM100 operators.

The SM90 reference transport consumes the original six-field plan, including
the positive projection and joint residual fields. It implements paid-head
budgets 14/16 and 20 (H32) or 28 (H64). Scalar RN operations preserve the
original four-chain epilogue structure, and the direct-ABS tail retains directed
upward rounding. The generic high-family path is used rather than the packed
BF16 fast path. Raw FP8 dots currently use serial FP32 FMAs, not a tuned Hopper
matrix kernel. This implementation establishes a functional comparison point;
it is not expected to be a performance backend.

H100 component qualification covers both head counts and all three families:
planner paid-head identity/order, positive-head overflow into projection,
budget gating, invalid capture recovery hints, original RMS-radius encoding,
BF16 epilogue agreement with an independent rounding oracle, compiled screen
upper bounds on controlled inputs, threshold equality, invalid plan/scale keep
decisions, seed witnesses, exact TopK score multisets, tie overflow signals,
and output-buffer canaries. Passing these finite cases is not a proof over
arbitrary FP8 inputs.

The qualification harness also connects seed, plan, screen, full-head candidate
refinement, 5-byte records, selector and boundary completion at TopK=2048.
Candidate refinement recomputes every original head and does not reuse partial
screen sums. This harness uses host compaction and a deliberately zero-valued
key suffix to exercise pruning. It targets the author's BF16 arithmetic;
it does not establish SM90-native FP32 equivalence or realistic pruning rates.
Its recovery-routing check uses a host fallback to the full reference result,
not a production GPU conditional graph.

Before enabling inference, the remaining work includes a numerical certificate
for the actual SM90 scoring backend, architecture-specific matrix scheduling,
common/solo work queues, bounded candidate emission, original bootstrap/carry
and logical permutation, GPU-side recovery and graph dispatch, and real-model
validation. In particular, the BF16 5-byte encoding must not be applied to
general FP32 scores. No TTFT, TPS, or faithful-port speedup is claimed.
