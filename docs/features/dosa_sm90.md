# Experimental DoSA indexer on SM90

## Current snapshot (2026-10-30)

This is an unfinished research port, with no H100 acceleration claim. Earlier
sections below describe historical implementations and their diagnostics.
The current `native_partial=True, shared_refine=True` benchmark path is an
algorithm variant, not a complete reproduction of the author's aggregate bound.
It retains all positive heads and selected negative heads, then bounds the
omitted negative contributions by zero. Its FP32 epilogue envelope is separate
from raw-dot error analysis. General certification remains unfinished.

The current H32 kernels use WGMMA, a three-stage TMA screen with four math
warpgroups, and shared-key refinement with threshold-filtered sparse writes.
The original SM100 source files are unchanged. The legacy aggregate-bound
path remains available for research and can fail to prune under conservative
Hopper allowances. Neither path is enabled in the serving dispatcher.

Saved GLM-5.2-FP8 layer-0 inputs use continuous source-code tokens, trained
embeddings and BF16 reference projections of dequantized FP8 weights. They
are not full-model production activations. Seven numerical qualification
cases and all real-query Top2048 score multisets passed native comparisons;
these finite checks do not establish a general losslessness theorem.

For 16K real queries, processed as four 4K chunks, the latest scheduling sweep
measured 24.49 ms stock versus 95.22 ms port at 128K keys (0.257x), and
52.43 ms versus 165.79 ms at 256K keys (0.316x), taking the faster tested
configuration at each length. Preparation is excluded; per-call normalization,
query staging, sorting when enabled, bootstrap and final selection are included.
There are no TTFT, model-quality, H64-native-partial or RTX 5090 claims.

Reproduction helpers are in `benchmarks/kernels/`: `qualify_partial.py`,
`benchmark_real_indexer.py` and `sweep_h100.py`. The real-input benchmark accepts
`--rows 16384 --prepare-only` to save continuous 16K query traces. Its direct
Hopper benchmark still requires a supported workspace size; `sweep_h100.py`
handles 16K via 4K workspaces. The sweep's saved-trace paths and sample manifest
are experiment-specific and must be adjusted to the local artifact location.

## Earlier implementation notes

This branch implements a research indexer path for H100, from prepared FP8
Q/K, FP32 weights and key scales to Top-2048 IDs. It does not change vLLM's
serving dispatcher, load a full model, or establish model-level TTFT/TPS gains.
SM120/RTX 5090 has not been qualified.

The experimental backend is disabled by default. Constructing its benchmark
workspace requires `allow_unproven_bound=True` and every reported workload is
checked against full native scoring. General lossless pruning certification
and serving enablement remain unfinished.

The reference scorer is the pinned DeepGEMM SM90 `fp8_fp4_mqa_logits` path,
followed by vLLM's `top_k_per_row_prefill`. It uses FP32 scores, rather than
the author's SM100 MX/BF16 scores. Identical tied IDs are not required: output
must contain distinct visible keys with exactly the native TopK score multiset.

## Pipeline and fidelity

The implementation retains the original planner algorithm and source files, including paid-head
selection, positive projection, joint residual bounds, sampled negative support,
and numerical allowances. Bootstrap spreads four queries across the call;
native-order scoring supplies their winners. The original HOT voting, stable
cutoff and lossless permutation/gather/RMS-radius kernels prepare the seed.
All hints and permutation state are reset per call. No answer or threshold is
cached from a prior invocation.

The SM90 build generates a separate numerical specialization of the planner.
Its raw-dot unit-roundoff parameter is `2^-10`, rather than the original
`2^-23`, and its direct-ABS screen envelope is 0.25. Configuration checks the
exact pinned source patterns before generating this specialization. The SM100
source and default numerics remain unchanged.

The first 12288 permuted keys supply a native-FP32 Top-2048 threshold. Their
2048 winners are retained. Screening excludes the whole seed range; survivors
receive all original heads and the original FP32 reduction order. Final native
FP32 TopK maps compact positions through physical IDs and the permutation.

Hardware adaptations include:

- FP8 WGMMA with register accumulators instead of TCGEN05/TMEM.
- Four packed queries with exactly `4 * (paid + 2)` head columns: 64, 72, 88,
  or 120. Query boundaries rotate the owner of the aggregate pair within each
  four-lane group; padding each query to a separate matrix tile is unnecessary.
- Register-based upper-bound reduction. Signed terms and final additions round
  upward; the subtracted negative aggregate rounds downward. This may retain
  extra candidates and does not weaken the mathematical upper bound.
- Two shared-memory key buffers with `cp.async` prefetch overlapping MMA and
  bound evaluation.
- Key-parallel bootstrap scoring: the ordinary SM90 dense kernel schedules
  primarily over query blocks, which underutilizes the GPU for four queries.

The research path still differs from the production SM100 implementation:
candidate IDs and FP32 scores currently use full-capacity workspaces; refinement
does not yet merge common/solo key loads; final selection uses lossless FP32
values instead of BF16-specific 5-byte records; dispatch uses two prepared
graphs with a host admission read. These costs must remain inside benchmarks.

## Numerical admission and target bound

The accelerated domain is finite E4M3FN Q/K with D=128 and H=32 or 64,
`2^-16 <= abs(weight) <= 65504`, and positive power-of-two key scales between
`2^-64` and `2^64`. Query intervals start at zero and contain a common visible
prefix of at least 12288 keys. Keys outside the common prefix are not permuted.
Device admission runs again for every replay; unsupported mutations take the
original dense scorer and TopK path. Tensor addresses, shapes, strides and
dtypes are pinned by the private workspace.

The power-of-two scale restriction and weight limits deliberately exclude
overflow and underflow differences between compiler configurations. FP8 values
are multiples of `2^-9`, so raw products lie on a `2^-18` grid. Admitted FP32
weights are multiples of `2^-39`. Products and weighted sums lie on a grid no
finer than `2^-57`, and scaling by at least `2^-64` leaves nonzero intermediates
at least `2^-121`, above the minimum FP32 normal. Upper magnitudes remain below
`2^112`. This reasoning concerns the native scorer; failed/nonfinite screening
plans remain conservative keep decisions.

An adversarial H100 probe invalidated an earlier assumption that FP8 WGMMA
could use the FP32 raw-dot envelope. With one nonzero query head and unit
weight/scale, a D128 dot consisting of sixteen +448 values, ninety-six 1/64
values and sixteen -448 values has exact result 1.5. Native FP8 scoring returned
0 on the saved example. The old `gamma(128,2^-23)` allowance was only
0.2187762264. All nonzero operands in this example are normal FP8 values.
The archived probe contains byte-level reproductions and native results.

The [PTX WGMMA precision note](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
explicitly describes FP8 accumulation below FP32 precision despite an FP32
destination, and leaves rounding/order/subnormal handling unspecified.
The [DeepGEMM hardware discussion](https://github.com/deepseek-ai/DeepGEMM/issues/37)
also documents Hopper's limited accumulator precision. Earlier benchmark
outputs passed finite native comparisons; their timings are historical,
pointwise-validated results, not evidence of a general certificate.

The revised research specialization uses the deliberately conservative
`gamma(128,2^-10) = 1/7` for raw dots and 0.25 for the direct-ABS screen
envelope. These guards are much wider than the old FP32 assumptions and can
erase pruning gains. They are a conservative research configuration rather
than a complete manufacturer-ISA proof. In addition, query/fine rows containing
FP8 subnormals are marked uncertified and all their visible keys are retained;
key entries containing subnormals are also retained. An uncertified four-query
group bypasses matrix screening entirely. Debug upper values are defined only
for certified rows; recovery rows are checked for complete candidate retention.

The register screen upper-bounds its paid/aggregate expression with directed
rounding. Direct-ABS signed terms use upward FMA; the generic expression uses
upward paid-head sums and a downward aggregate before subtraction. The original
planner's quantization, projection/residual and raw-dot allowances are retained.
Invalid plans or nonfinite upper bounds keep every visible affected candidate.
Only strict `upper < tau` rejects a key; equality survives.

Seed thresholds come from exact native scores of distinct visible keys.
Output capacity covers all nonseed keys plus seed winners, so live candidates
cannot be lost to truncation. The FP32 candidate scorer uses the same WGMMA
K=32 issue sequence and head/weight/lane reduction as pinned DeepGEMM. Each new
kernel revision is qualified against the actual installed scorer before timing.
This remains experimental: tests are finite and no real-model activation trace
or serving integration has yet been validated.

## Benchmark interpretation

The harness measures complete selection work from prepared indexer tensors,
including fresh admission, the host status read, bootstrap scoring and TopK,
voting, permutation, key gathering/RMS metadata, seed scoring/TopK, planning,
screening, compaction, all-head refinement, final TopK and ID mapping. Both
paths are prepared with CUDA graphs; allocation, JIT and capture are excluded.
Paired timings randomize which path runs first and retain individual samples.

Synthetic independent queries and deliberately correlated queries/heads are
reported separately. Neither distribution is a GLM/DeepSeek activation trace.
The paper's [complete-Indexer results](https://arxiv.org/html/2610.13990v1#S4)
use model-run activations over all 21 indexer layers, 4K-32K query chunks and
128K-1M contexts, including query/key preparation. Its 1.12-2.74x SGLang speedups
are a reference range, not a directly matched baseline for this narrower H100
kernel benchmark. Do not infer model TTFT or TPS from these measurements.

### Recorded H100 run: 2026-10-29

Run `20261029T112300.651152Z-8a51a774` passed 18 qualification cases,
two edge checks and 24 timed synthetic cases. The extension SHA256 is
`cf205fd24e06b412dedf81b21f5646b1f8b464380dfc8fcb821da121b4f09d36`.
The experiment archive contains exact sources, generated planner, environment,
12 timing samples per path/case and individual profiler traces. No weights
were loaded. Inputs use per-head power-of-two query scales folded into weights
and per-key power-of-two key scales.

Representative independent-query medians, including all selection stages:

| Queries | Keys | Heads | Native stock (ms) | Research port (ms) | Stock / port |
| --- | --- | --- | --- | --- | --- |
| 128 | 262144 | 32 | 1.2891 | 3.1527 | 0.409x |
| 128 | 262144 | 64 | 1.2066 | 3.4797 | 0.347x |
| 128 | 1048576 | 32 | 4.9546 | 11.9410 | 0.415x |
| 128 | 1048576 | 64 | 4.6146 | 13.1737 | 0.350x |
| 4096 | 131072 | 32 | 6.0835 | 38.6814 | 0.157x |
| 4096 | 131072 | 64 | 9.8336 | 49.1067 | 0.200x |

The current conservative configuration pruned zero post-seed keys in all
24 cases. Removal of seed losers alone must not be described as successful
bound-based pruning. Every candidate score and final Top2048 score multiset
matched native FP32 in these finite tests; a general pruning certificate is
still unestablished and the backend remains disabled by default.

The dense-same-backend ablation also shows that the custom all-head refinement
kernel is slower than stock for large query batches. The remaining work is
both numerical (tighter valid Hopper bounds) and computational (refinement
tiling and avoiding full-capacity score materialization). Earlier results using
the invalidated FP32 raw-dot envelope must not be presented as final lossless
port speedups. The paper's Appendix D.4 excludes Q/K preparation too, but its
up-to-3.74x SGLang result still uses different hardware, activations and baselines.

### Real first-layer inputs and shared experiment app

The workspace experiment uses actual source-code tokens, the pinned GLM-5.2
embedding and layer-0 projection/norm weights. FP8 projection weights are
dequantized for BF16 reference GEMMs. This produces trained first-layer inputs;
it does not reproduce production dynamic-FP8 projections or deeper model layers.
Native RoPE, Q quantization and K cache insertion/gather are used. The gather
scale destination is an `[N,4]` byte view, then interpreted as FP32 (H100) or
packed UE8M0 (B200). RoPE construction requires a vLLM configuration context.

Some real query weights fall below the current SM90 admission limit. The
research harness can multiply every head weight of one query by the same
positive power of two to enter the admitted range. This preserves mathematical
TopK, but floating-point equivalence is checked separately: every visible native
score must exactly equal its original value times the row factor, and final
Top2048 is checked against the original unscaled native scores. No tolerance is
introduced. Normalization and copying are included in the timed port path.
Unsupported inputs remain explicit stock fallbacks, never reported as DoSA wins.

Modal orchestration now uses one persistent deployment named `dosa`, containing
`h100`, `b200`, `check_setup` and `prepare_inputs`. All functions have zero minimum
containers and release resources after a bounded invocation. Deployment updates
reuse the same app; submissions use `Function.from_name(...).spawn()` without
creating ephemeral apps. The workspace entry points are `outputs/dosa_app.py`
(deploy), `outputs/submit_dosa.py --gpu h100|b200|both` (submit), and
`outputs/collect_dosa.py --gpu h100 --wait-seconds 45` (bounded polling).

Run `20261030T024618.353787Z-369dfb2b` passed every native-score scaling
comparison and both complete 4096-row Top2048 checks, with no stock fallback.
Row scale factors ranged from 1 to 256. The prepared-input timings were:

| Keys | Native stock (ms) | Port including normalization (ms) | Stock / port |
| --- | --- | --- | --- |
| 131072 | 6.1137 | 33.3957 | 0.1831x |
| 262144 | 12.6516 | 69.4457 | 0.1822x |

Both cases retained every post-seed candidate. There were 3148 and 3190
uncertified queries respectively; these queries were conservatively retained.
The full candidate refinement kernel dominated the profiled GPU time (21.77 ms
and 46.14 ms in the extra profiler replays). This is successful finite
qualification, not an achieved speedup or a general pruning proof. The earlier
run `20261030T023413.542044Z-9bf319fe` used stock fallback and its near-1x timing
ratios must not be presented as DoSA gains.

### Pruning root-cause diagnostics

Three bounded H100 diagnostic invocations in the shared `dosa` deployment
tested Paid14/16/20 against 64 evenly spaced real first-layer queries and the
complete 128K native TopK oracle. Component analysis sampled 8192 post-seed
keys. All original guarded outputs passed exact native score-multiset and ID
checks. These are diagnostic runs, not timing claims.

The fixed Paid16 benchmark omits the author's query sorting and adaptive
Paid14/16/high budgeting. Across the original 4096-query traces, 1784 (128K)
and 1923 (256K) queries have more than 16 positive heads, invalidating the
Direct-ABS plan. Another 2049/2109 queries contain FP8 subnormals; fine-vector
subnormal guards add further recovery. These causes overlap. In the 64-query
diagnostic, Paid20 had 64 valid planner outputs before added guards, but still
pruned zero keys after those guards were bypassed for inspection.

The decisive problem is numerical tail inflation. For valid Paid16 sample
cells, median tail magnitudes relative to the seed threshold were:

| Component | Median / threshold |
| --- | --- |
| DD encoding residual alone | 0.0065 |
| Conservative native target raw-dot allowance | 5.35 |
| Conservative screen raw-dot allowance | 8.40 |
| Actual combined tail | 13.78 |

Removing either raw-dot allowance alone in the diagnostic still enabled zero
pruning. Counterfactual FP64 geometry without these allowances excluded about
97% of sampled keys. This is evidence of available geometric separation, not
permission to remove production allowances. The median seed/global-threshold
gap was about 15%, so seed quality alone cannot explain a 10x-14x tail.

A separate native-semantics reference retains every positive head plus chosen
paid negative heads, setting only omitted negative weights to zero. With the
same Q/K, native scorer and epilogue order, raw dots are unchanged and finite
FMA/add rounding is monotone: removing a nonpositive contribution cannot lower
the result. All 64x131072 native score entries per family were checked for
this upper-bound relation. It excluded 70.66%, 71.13% and 76.71% of post-seed
keys for Paid14/16/20 respectively. This reference still computes all raw
heads, so it establishes a useful native-compatible bound but no speedup.
A reduced-head kernel must reproduce it before performance is claimed.
