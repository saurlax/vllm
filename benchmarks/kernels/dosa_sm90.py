# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Hopper indexer using the original DoSA planner and WGMMA.

This module is benchmark-only. It does not change the serving dispatcher.
The baseline is the pinned SM90 DeepGEMM FP32 scorer plus native TopK.
"""

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.model_executor.layers import dosa_caller as caller
from vllm.model_executor.layers import dosa_ops as author_ops
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits


@triton.jit
def _partial_budgets(W, BUDGET, ROWS: tl.constexpr, H: tl.constexpr):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    heads = tl.arange(0, H)
    w = tl.load(W + rows[:, None] * H + heads[None, :], rows[:, None] < ROWS, 0.0)
    maximum = tl.max(tl.sum((w > 0).to(tl.int32), axis=1), axis=0)
    budget = tl.where(
        maximum <= 14,
        14,
        tl.where(
            maximum <= 16,
            16,
            tl.where(maximum <= 20, 20, tl.where(maximum <= 24, 24, 32)),
        ),
    )
    tl.store(BUDGET + tl.program_id(0), budget)


@triton.jit
def _partial_stage(Q, W, CAP, BUDGET, PACKED, IDS, MASKED_W, H: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, H)
    d = tl.arange(0, 128)
    w = tl.load(W + row * H + h)
    q = tl.load(Q + (row * H + h[:, None]) * 128 + d[None, :]).to(tl.float32)
    sign = tl.load(CAP + row * 129 + d).to(tl.uint32)
    votes = tl.sum(((sign[:, None] >> h[None, :]) & 1).to(tl.int32), axis=0)
    strength = tl.abs(w) * tl.sqrt(tl.sum(q * q, axis=1))
    priority = tl.where(w > 0, 1.0e30, strength * tl.minimum(votes, 128 - votes))
    earlier = (priority[None, :] > priority[:, None]) | (
        (priority[None, :] == priority[:, None]) & (h[None, :] < h[:, None])
    )
    rank = tl.sum(earlier.to(tl.int32), axis=1)
    paid = tl.load(BUDGET + row // 8)
    selected = rank < paid
    slot = tl.cumsum(selected.to(tl.int32)) - 1
    tl.store(IDS + row * H + slot, h, selected)
    raw = tl.load(Q + (row * H + h[:, None]) * 128 + d[None, :])
    tl.store(
        PACKED + (row * H + slot[:, None]) * 128 + d[None, :],
        raw.to(tl.uint8, bitcast=True),
        selected[:, None],
    )
    tl.store(MASKED_W + row * H + h, tl.where(selected, w, 0.0))


@triton.jit
def _compact_union(
    MASK,
    IDS,
    COUNTS,
    ROW_COUNTS,
    ENDS,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    B: tl.constexpr,
):
    group = tl.program_id(0)
    rows = group * 4 + tl.arange(0, 4)
    key = 12288 + tl.program_id(1) * B + tl.arange(0, B)
    bits = tl.load(
        MASK + rows[:, None] * STRIDE + key[None, :] // 8,
        (rows[:, None] < ROWS) & (key[None, :] < N),
        0,
    ).to(tl.int32)
    keep = ((bits >> (key[None, :] % 8)) & 1) != 0
    keep &= (rows[:, None] < ROWS) & (key[None, :] < N)
    row_count = tl.sum(keep.to(tl.int32), axis=1)
    tl.atomic_add(ROW_COUNTS + rows, row_count, rows < ROWS)
    union = tl.max(keep.to(tl.int32), axis=0)
    count = tl.sum(union, axis=0)
    base = tl.atomic_add(COUNTS + group, count)
    rank = tl.cumsum(union) - 1
    tl.store(IDS + group * N + base + rank, key, union != 0)


@triton.jit
def _key_admission(K, S, BAD, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    code = tl.load(K + i, i < N * 128, 0).to(tl.int32) & 127
    bits = tl.load(S + i, i < N, 1.0).to(tl.uint32, bitcast=True)
    scale_bad = (bits < 0x1F800000) | (bits > 0x5F800000) | ((bits & 0x007FFFFF) != 0)
    bad = ((i < N * 128) & (code == 127)) | ((i < N) & scale_bad)
    if tl.sum(bad.to(tl.int32)):
        tl.atomic_or(BAD, 1)


@triton.jit
def _query_admission(
    Q,
    W,
    STARTS,
    ENDS,
    BAD,
    QN: tl.constexpr,
    WN: tl.constexpr,
    ROWS: tl.constexpr,
    KEYS: tl.constexpr,
    COMMON: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    q = tl.load(Q + i, i < QN, 0).to(tl.int32) & 127
    w = tl.abs(tl.load(W + i, i < WN, 0.5))
    start = tl.load(STARTS + i, i < ROWS, 0)
    end = tl.load(ENDS + i, i < ROWS, KEYS)
    invalid = ((i < QN) & (q == 127)) | (
        (i < WN) & ~((w >= 0.0000152587890625) & (w <= 65504.0))
    )
    invalid |= (i < ROWS) & ((start != 0) | (end < COMMON) | (end > KEYS))
    if tl.sum(invalid.to(tl.int32)):
        tl.atomic_or(BAD, 1)


@triton.jit
def _retain_seed(
    SEED, SELECTED, IDS, SCORES, TAU, N: tl.constexpr, SEED_N: tl.constexpr
):
    row = tl.program_id(0)
    j = tl.arange(0, 2048)
    ids = tl.load(SELECTED + row * 2048 + j)
    score = tl.load(SEED + row * SEED_N + ids)
    tl.store(IDS + row * N + j, ids)
    tl.store(SCORES + row * N + j, score)
    tl.store(TAU + row, tl.min(score))


@triton.jit
def _guard_key_subnormals(K, RHO, N: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    codes = (
        tl.load(K + rows[:, None] * 128 + d[None, :], rows[:, None] < N, 0).to(tl.int32)
        & 127
    )
    uncertain = tl.sum(((codes > 0) & (codes < 8)).to(tl.int32), axis=1) != 0
    old = tl.load(RHO + rows, rows < N, 0)
    tl.store(RHO + rows, tl.where(uncertain, 127, old), rows < N)


@triton.jit
def _guard_query_subnormals(
    Q, FINE, HINT, HEADS: tl.constexpr, WIDTH: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    q = tl.load(Q + row * HEADS * 128 + i, i < HEADS * 128, 0).to(tl.int32) & 127
    fine = tl.load(FINE + row * WIDTH * 128 + i, i < WIDTH * 128, 0).to(tl.int32) & 127
    uncertain = ((i < HEADS * 128) & (q > 0) & (q < 8)) | (
        (i < WIDTH * 128) & (fine > 0) & (fine < 8)
    )
    if tl.sum(uncertain.to(tl.int32)):
        tl.store(HINT + row, 1)


@triton.jit
def _compact_masks(
    MASK, IDS, COUNTS, ENDS, N: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    key = 12288 + tl.program_id(1) * B + tl.arange(0, B)
    end = tl.load(ENDS + row)
    mask = tl.load(MASK + row * STRIDE + key // 8, (key < N) & (key < end), 0).to(
        tl.int32
    )
    live = (key < N) & (key < end) & (((mask >> (key % 8)) & 1) != 0)
    ranks = tl.cumsum(live.to(tl.int32)) - 1
    total = tl.sum(live.to(tl.int32))
    if total:
        base = tl.atomic_add(COUNTS + row, total)
        tl.store(IDS + row * N + base + ranks, key, live)


@triton.jit
def _map(IDS, POS, PERM, OUT, N: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    position = tl.load(POS + i, i < TOTAL, 0)
    value = tl.load(IDS + i // 2048 * N + position, i < TOTAL, -1)
    value = tl.load(PERM + value, i < TOTAL, -1)
    tl.store(OUT + i, value, i < TOTAL)


class HopperIndexer:
    """Private workspace with fresh device admission and host fallback per call.

    All metadata, seed, planning, compaction, refinement and TopK work is inside
    the measured call. Allocation, compilation and graph capture are setup.
    """

    def __init__(
        self,
        q,
        k,
        weights,
        scales,
        starts,
        ends,
        *,
        paid=None,
        screen_ctas=4,
        refine_ctas=4,
        seed_ctas=16,
        debug=False,
        common_end=None,
        allow_unproven_bound=False,
        native_partial=False,
        shared_refine=False,
    ):
        if not allow_unproven_bound:
            raise RuntimeError(
                "The native FP8 WGMMA raw-dot error certificate is not established. "
                "This research backend requires explicit opt-in "
                "and full native-output validation."
            )
        import vllm._dosa_sm90_C  # noqa: F401

        assert torch.cuda.get_device_capability(q.device) == (9, 0)
        assert q.ndim == 3 and q.shape[1] in (32, 64) and q.shape[2] == 128
        assert k.ndim == 2 and k.shape[1] == 128 and k.shape[0] >= 12288
        assert q.dtype == k.dtype == torch.float8_e4m3fn
        assert weights.dtype == scales.dtype == torch.float32
        assert starts.dtype == ends.dtype == torch.int32
        assert weights.shape == q.shape[:2] and scales.shape == k.shape[:1]
        assert starts.shape == ends.shape == q.shape[:1]
        self.inputs = (q, k, weights, scales, starts, ends)
        assert all(
            x.is_cuda and x.is_contiguous() and x.device == q.device
            for x in self.inputs
        )
        self.q, self.k, self.w, self.scale, self.starts, self.ends = self.inputs
        self.native_partial = native_partial
        self.shared_refine = shared_refine
        assert not shared_refine or native_partial
        assert not native_partial or q.shape[1] == 32
        self.rows, self.heads, _ = q.shape
        assert self.rows > 0 and q.data_ptr() % 16 == k.data_ptr() % 16 == 0
        self.keys = len(k)
        self.common_end = self.keys if common_end is None else common_end
        assert 12288 <= self.common_end <= self.keys
        self.paid = paid if paid is not None else (20 if self.heads == 32 else 28)
        assert self.paid in ((14, 16, 20) if self.heads == 32 else (14, 16, 28))
        assert self.rows * self.keys <= 2147483647 and self.keys <= 1048576
        self.debug = debug
        self.screen_ctas, self.refine_ctas, self.seed_ctas = (
            screen_ctas,
            refine_ctas,
            seed_ctas,
        )
        device = q.device

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        self.seed = empty((self.rows, 12288))
        self.sample = empty((self.rows, 128))
        self.seed_selected = empty((self.rows, 2048), torch.int32)
        self.seed_ends = empty((self.rows,), torch.int32).fill_(12288)
        self.tau = empty((self.rows,))
        self.rho = empty((self.keys,), torch.int32)
        self.pk = empty(tuple(k.shape), k.dtype)
        self.ps = empty(tuple(scales.shape))
        self.identity = torch.arange(self.keys, device=device, dtype=torch.int32)
        self.permutation = empty((self.keys,), torch.int32)
        self.hot = empty((12288,), torch.int64)
        self.votes = empty((self.common_end,), torch.int32)
        self.carry_partial = empty((triton.cdiv(self.common_end, 8192), 5), torch.int16)
        self.carry_state = empty((7 + triton.cdiv(self.common_end, 8192),), torch.int32)
        self.hot_epoch = empty((self.keys,), torch.int32)
        self.pair_rank = empty((self.keys,), torch.int32)
        self.swap_a = empty((12288,), torch.int32)
        self.swap_b = empty((12288,), torch.int32)
        self.plan_counts = empty((4,), torch.int32)
        self.cold_q = empty((4, self.heads, 128), torch.uint8)
        self.cold_w = empty((4, self.heads))
        self.cold_ids = empty((4, 2048), torch.int32)
        self.cold_scores = empty((4, self.common_end))
        self.cold_dummy_ids = empty((4, self.common_end), torch.int32)
        self.cold_counts = empty((4,), torch.int32).zero_()
        self.cold_capture = empty(
            (4, 129), torch.int32 if self.heads == 32 else torch.int64
        ).zero_()
        self.cold_starts = empty((4,), torch.int32).zero_()
        self.cold_ends = empty((4,), torch.int32).fill_(self.common_end)
        self.bad = empty((1,), torch.int32)
        self.capture = empty(
            (self.rows, 129), torch.int32 if self.heads == 32 else torch.int64
        )
        self.ids = empty((self.rows, self.keys), torch.int32)
        self.keep_masks = empty((self.rows, triton.cdiv(self.keys, 8)), torch.uint8)
        self.scores = empty((self.rows, self.keys))
        self.counts = empty((self.rows,), torch.int32)
        self.errors = empty((self.rows,), torch.int32)
        self.positions = empty((self.rows, 2048), torch.int32)
        self.output = empty((self.rows, 2048), torch.int32)
        self.upper = empty((self.rows, self.keys) if debug else (1,))
        self.budget = empty((triton.cdiv(self.rows, 8),), torch.int32).fill_(self.paid)
        if native_partial:
            self.partial_q = empty(tuple(q.shape), torch.uint8)
            self.partial_ids = empty((self.rows, self.heads), torch.int32)
            self.partial_budget = empty((triton.cdiv(self.rows, 8),), torch.int32)
            self.partial_w = empty(tuple(weights.shape))
            self.partial_op = torch.ops._C.dosa_sm90_h32_partial_screen
            if shared_refine:
                self.union_ids = empty(
                    (triton.cdiv(self.rows, 4), self.keys), torch.int32
                )
                self.union_counts = empty((triton.cdiv(self.rows, 4),), torch.int32)
                self.screen_counts = empty((self.rows,), torch.int32)
                self.shared_op = torch.ops._C.dosa_sm90_h32_refine_shared_run
        width = self.paid + 2
        self.plan = [
            empty((self.rows, width, 128), torch.uint8),
            empty((self.rows, width)),
            empty((self.rows, 64 if self.paid <= 16 else 96)),
            empty((self.rows,)),
            empty((self.rows,)),
            empty((self.rows,), torch.int32),
        ]
        self.signature = self._signature()
        prefix = f"dosa_sm90_h{self.heads}_"
        self.score_op = getattr(torch.ops._C, prefix + "hopper_score")
        self.screen_op = getattr(torch.ops._C, prefix + "screen_fast_screen")
        self.plan_op = getattr(
            torch.ops._C,
            prefix + ("direct_abs" if self.paid <= 16 else "planner") + "_plan128",
        )
        self._admission()
        if int(self.bad.item()):
            raise ValueError("Initial inputs are outside the Hopper certificate domain")
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._admission()
                self._compute()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.admission_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.admission_graph):
            self._admission()
        self.compute_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.compute_graph):
            self._compute()

    def _signature(self):
        return tuple(
            (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
            for x in self.inputs
        )

    def _admission(self):
        self.bad.zero_()
        _key_admission[(triton.cdiv(self.keys * 128, 4096),)](
            self.k.view(torch.uint8),
            self.scale,
            self.bad,
            self.keys,
            4096,
            num_warps=4,
        )
        _query_admission[(triton.cdiv(self.q.numel(), 4096),)](
            self.q.view(torch.uint8),
            self.w,
            self.starts,
            self.ends,
            self.bad,
            self.q.numel(),
            self.w.numel(),
            self.rows,
            self.keys,
            self.common_end,
            4096,
        )

    def _topk(self, scores, ends, output):
        ops.top_k_per_row_prefill(
            scores,
            self.starts,
            ends,
            output,
            self.rows,
            scores.stride(0),
            scores.stride(1),
            2048,
        )

    def _compute(self):
        # Original call-local cold-query voting and lossless key permutation.
        author_ops._spread_queries[(4,)](
            self.q.view(torch.uint8),
            self.w,
            self.cold_q,
            self.cold_w,
            self.rows,
            self.heads,
            num_warps=4,
        )
        self.score_op(
            self.cold_q.view(torch.float8_e4m3fn),
            self.k,
            self.cold_w,
            self.scale,
            self.cold_dummy_ids,
            self.cold_counts,
            self.cold_scores,
            self.cold_capture,
            self.cold_starts,
            self.cold_ends,
            False,
            256,
        )
        ops.top_k_per_row_prefill(
            self.cold_scores,
            self.cold_starts,
            self.cold_ends,
            self.cold_ids,
            4,
            self.cold_scores.stride(0),
            1,
            2048,
        )
        self.votes.zero_()
        author_ops._add_votes[(32,)](
            self.cold_ids, self.votes, 8192, self.common_end, 256
        )
        caller.carry_votes_topk_reset_(
            self.votes, self.hot, self.carry_partial, self.carry_state, 12288, 4, 0
        )
        self.permutation.copy_(self.identity)
        self.hot_epoch.zero_()
        self.plan_counts.zero_()
        caller.plan_and_permuted_gather_rho_out(
            self.hot,
            self.hot_epoch,
            self.permutation,
            self.pair_rank,
            self.swap_a,
            self.swap_b,
            self.plan_counts,
            0,
            self.common_end,
            1,
            self.k,
            self.scale,
            self.pk,
            self.ps,
            self.rho,
        )
        _guard_key_subnormals[(triton.cdiv(self.keys, 32),)](
            self.pk.view(torch.uint8), self.rho, self.keys, num_warps=4
        )
        self.capture.zero_()
        self.errors.zero_()
        self.score_op(
            self.q,
            self.pk,
            self.w,
            self.ps,
            self.ids,
            self.counts,
            self.sample,
            self.capture,
            self.starts,
            self.ends,
            False,
            2,
        )
        self.seed = fp8_fp4_mqa_logits(
            (self.q, None),
            (self.pk[:12288], self.ps[:12288]),
            self.w,
            self.starts,
            self.seed_ends,
            12288,
        )
        self._topk(self.seed, self.seed_ends, self.seed_selected)
        _retain_seed[(self.rows,)](
            self.seed,
            self.seed_selected,
            self.ids,
            self.scores,
            self.tau,
            self.keys,
            12288,
        )
        self.counts.fill_(2048)
        if self.native_partial:
            _partial_budgets[(triton.cdiv(self.rows, 8),)](
                self.w,
                self.partial_budget,
                self.rows,
                self.heads,
            )
            _partial_stage[(self.rows,)](
                self.q,
                self.w,
                self.capture,
                self.partial_budget,
                self.partial_q,
                self.partial_ids,
                self.partial_w,
                self.heads,
                num_warps=8,
            )
            self.partial_op(
                self.partial_q,
                self.partial_ids,
                self.partial_budget,
                self.w,
                self.pk,
                self.ps,
                self.tau,
                self.starts,
                self.ends,
                self.keep_masks,
                self.upper,
                self.screen_ctas,
                self.debug,
            )
        else:
            self._aggregate_screen()
        if self.shared_refine:
            self.union_counts.zero_()
            self.screen_counts.fill_(2048)
            _compact_union[
                (triton.cdiv(self.rows, 4), triton.cdiv(self.keys - 12288, 1024))
            ](
                self.keep_masks,
                self.union_ids,
                self.union_counts,
                self.screen_counts,
                self.ends,
                self.rows,
                self.keys,
                self.keep_masks.stride(0),
                1024,
                num_warps=8,
            )
            self.shared_op(
                self.q,
                self.w,
                self.pk,
                self.ps,
                self.union_ids,
                self.union_counts,
                self.keep_masks,
                self.tau,
                self.starts,
                self.ends,
                self.ids,
                self.scores,
                self.counts,
                self.refine_ctas,
            )
        else:
            _compact_masks[(self.rows, triton.cdiv(self.keys - 12288, 1024))](
                self.keep_masks,
                self.ids,
                self.counts,
                self.ends,
                self.keys,
                self.keep_masks.stride(0),
                1024,
                num_warps=4,
            )
            self.score_op(
                self.q,
                self.pk,
                self.w,
                self.ps,
                self.ids,
                self.counts,
                self.scores,
                self.capture,
                self.starts,
                self.ends,
                True,
                self.refine_ctas,
            )
        self._topk(self.scores, self.counts, self.positions)
        _map[(triton.cdiv(self.output.numel(), 256),)](
            self.ids,
            self.positions,
            self.permutation,
            self.output,
            self.keys,
            self.output.numel(),
            256,
        )

    def _aggregate_screen(self):
        self.plan_op(self.paid, self.q, self.w, self.capture, self.plan, self.budget)
        _guard_query_subnormals[(self.rows,)](
            self.q.view(torch.uint8),
            self.plan[0],
            self.plan[5],
            self.heads,
            self.paid + 2,
            triton.next_power_of_2(self.heads * 128),
            num_warps=8,
        )
        self.screen_op(
            self.paid,
            self.plan,
            self.pk,
            self.ps,
            self.rho,
            self.tau,
            self.starts,
            self.ends,
            self.keep_masks,
            self.errors,
            self.upper,
            12288,
            self.screen_ctas,
            self.debug,
        )

    def stock(self):
        scores = fp8_fp4_mqa_logits(
            (self.q, None),
            (self.k, self.scale),
            self.w,
            self.starts,
            self.ends,
            self.keys,
        )
        self._topk(scores, self.ends, self.output)
        return self.output

    def replay(self):
        if self._signature() != self.signature:
            raise ValueError(
                "Hopper workspace requires its original tensor addresses and metadata"
            )
        self.admission_graph.replay()
        if int(self.bad.item()):
            return self.stock()
        self.compute_graph.replay()
        return self.output
