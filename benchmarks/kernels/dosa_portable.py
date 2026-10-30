# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only DoSA bound kernel, portable across SM90 and SM120.

Supports only bounded integer Q/K and binary-fraction weights validated below.
This is not a general floating-point certificate or a vLLM serving backend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_score(
    Q, W, K, SCORES, N: tl.constexpr, HEADS: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    ids = tl.program_id(1) * B + tl.arange(0, B)
    heads = tl.arange(0, HEADS)
    dims = tl.arange(0, 128)
    q = tl.load(Q + row * HEADS * 128 + heads[:, None] * 128 + dims[None, :])
    k = tl.load(K + ids[None, :] * 128 + dims[:, None], ids[None, :] < N, 0.0)
    dots = tl.dot(q, k, out_dtype=tl.float32)
    weights = tl.load(W + row * HEADS + heads)
    score = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)
    tl.store(SCORES + row * N + ids, score, ids < N)


def dense_validated_integer_domain(q, k, weights):
    """Dense ablation after external domain validation; no DoSA pruning.

    This comparison isolates kernel-backend gains from algorithmic pruning.
    Like stock, this baseline does not repeat the DoSA admission scan.
    """
    scores = torch.empty((q.shape[0], k.shape[0]), device=q.device, dtype=torch.float32)
    _dense_score[(q.shape[0], triton.cdiv(k.shape[0], 128))](
        q,
        weights,
        k,
        scores,
        k.shape[0],
        q.shape[1],
        128,
        num_warps=4,
    )
    return scores


@triton.jit
def _map_topk(
    IDS,
    POSITIONS,
    OUT,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    row = offsets // TOPK
    position = tl.load(POSITIONS + offsets, offsets < TOTAL, 0)
    key = tl.load(IDS + row * N + position, offsets < TOTAL, 0)
    tl.store(OUT + offsets, key, offsets < TOTAL)


def map_compact_topk(ids, positions):
    """Map validated compact TopK positions directly using int32 indices."""
    assert ids.dtype == positions.dtype == torch.int32
    assert ids.is_contiguous() and positions.is_contiguous()
    result = torch.empty_like(positions)
    _map_topk[(triton.cdiv(positions.numel(), 256),)](
        ids,
        positions,
        result,
        ids.shape[1],
        positions.shape[1],
        positions.numel(),
        256,
    )
    return result


@triton.jit
def _seed_threshold(SCORES, IDS, TAU, STRIDE: tl.constexpr, TOPK: tl.constexpr):
    row = tl.program_id(0)
    positions = tl.arange(0, TOPK)
    ids = tl.load(IDS + row * TOPK + positions)
    values = tl.load(SCORES + row * STRIDE + ids)
    tl.store(TAU + row, tl.min(values))


def threshold_from_native_topk(scores, starts, ends, topk=2048, *, return_ids=False):
    """Compute the exact seed cutoff without sorting selected scores."""
    from vllm import _custom_ops as ops

    rows = scores.shape[0]
    ids = torch.empty((rows, topk), device=scores.device, dtype=torch.int32)
    tau = torch.empty(rows, device=scores.device, dtype=torch.float32)
    ops.top_k_per_row_prefill(
        scores, starts, ends, ids, rows, scores.stride(0), scores.stride(1), topk
    )
    _seed_threshold[(rows,)](scores, ids, tau, scores.stride(0), topk)
    return (ids, tau) if return_ids else tau


@triton.jit
def _admit(
    Q,
    K,
    W,
    TAU,
    THRESHOLD,
    BAD,
    QN: tl.constexpr,
    KN: tl.constexpr,
    WN: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    TS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    qb = tl.load(Q + i, i < QN, 0).to(tl.uint32) & 127
    kb = tl.load(K + i, i < KN, 0).to(tl.uint32) & 127
    # E4M3 encodings for absolute values 0, 1, 2; sign includes signed zero.
    qok = (qb == 0) | (qb == 56) | (qb == 64)
    kok = (kb == 0) | (kb == 56) | (kb == 64)
    w = tl.load(W + i, i < WN, 0.125)
    positive = i % HEADS < HEADS // 2
    wok = (tl.abs(w) == 0.125) | (tl.abs(w) == 0.25)
    wok &= tl.where(positive, w > 0, w < 0)
    invalid = ((i < QN) & ~qok) | ((i < KN) & ~kok)
    invalid |= (i < WN) & ~wok
    if tl.sum(invalid.to(tl.int32)) != 0:
        tl.atomic_or(BAD, 1)
    if tl.program_id(0) * B < ROWS:
        tau = tl.load(TAU + i * TS, i < ROWS, 0.0).to(tl.float64)
        converted = tau.to(tl.float32)
        ok = (tl.abs(tau) < float("inf")) & (converted.to(tl.float64) == tau)
        if tl.sum(((i < ROWS) & ~ok).to(tl.int32)) != 0:
            tl.atomic_or(BAD, 1)
        tl.store(THRESHOLD + i, converted, i < ROWS)


@triton.jit
def _make_plan(Q, W, PLAN, HEADS: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, HEADS // 2)
    d = tl.arange(0, 128)
    positive = tl.load(Q + row * HEADS * 128 + h[:, None] * 128 + d[None, :]).to(
        tl.float32
    )
    negative = tl.load(
        Q + row * HEADS * 128 + (h[:, None] + HEADS // 2) * 128 + d[None, :]
    ).to(tl.float32)
    nw = tl.load(W + row * HEADS + h + HEADS // 2)
    aggregate = tl.sum(negative * (-nw[:, None]), axis=0)
    r: tl.constexpr = HEADS // 2 + 1
    tl.store(PLAN + row * r * 128 + h[:, None] * 128 + d[None, :], positive)
    tl.store(PLAN + row * r * 128 + HEADS // 2 * 128 + d, aggregate)


@triton.jit
def _screen(
    PLAN,
    WEIGHTS,
    KEYS,
    TAU,
    UPPER,
    LIVE,
    IDS,
    COUNTS,
    N: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    COMPACT: tl.constexpr,
    SPLIT_NEGATIVE: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
):
    query = tl.program_id(0)
    ids = tl.program_id(1) * B + tl.arange(0, B)
    heads = tl.arange(0, H)
    dims = tl.arange(0, 128)
    q = tl.load(
        PLAN + query * R * 128 + heads[:, None] * 128 + dims[None, :],
        heads[:, None] < R,
        0,
    )
    k = tl.load(KEYS + ids[None, :] * 128 + dims[:, None], ids[None, :] < N, 0.0)
    if SPLIT_NEGATIVE:
        # Positive directions are unchanged integer Q, so this conversion is
        # exact. The aggregate stays FP16/FP32 in the separate reduction.
        q = q.to(KEYS.dtype.element_ty)
    else:
        k = k.to(tl.float16)
    dots = tl.dot(q, k, out_dtype=tl.float32)
    w = tl.load(WEIGHTS + query * 2 * (R - 1) + heads, heads < R - 1, 0)
    w = tl.where(heads == R - 1, -1.0, w)
    upper = tl.sum(tl.maximum(dots, 0.0) * w[:, None], axis=0)
    if SPLIT_NEGATIVE:
        negative = tl.load(PLAN + query * R * 128 + (R - 1) * 128 + dims).to(tl.float32)
        negative_dot = tl.sum(negative[:, None] * k.to(tl.float32), axis=0)
        upper -= tl.maximum(negative_dot, 0.0)
    tau = tl.load(TAU + query)
    if COMPACT:
        keep = (ids < N) & (upper >= tau)
        rank = tl.cumsum(keep.to(tl.int32)) - 1
        total = tl.sum(keep.to(tl.int32))
        base = tl.atomic_add(COUNTS + query * COUNT_STRIDE, total, sem="relaxed")
        tl.store(IDS + query * N + base + rank, ids, keep)
    else:
        tl.store(UPPER + query * N + ids, upper, ids < N)
        tl.store(LIVE + query * N + ids, upper >= tau, ids < N)


@triton.jit
def _screen_batch(
    PLAN,
    W,
    KEYS,
    TAU,
    OUT,
    COUNTS_OR_LIVE,
    PARTIAL,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    HALF: tl.constexpr,
    QB: tl.constexpr,
    B: tl.constexpr,
    COMPACT: tl.constexpr,
    REUSE_POSITIVE: tl.constexpr,
    FP8_AGGREGATE: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
    KEY_START: tl.constexpr = 0,
):
    first_row = tl.program_id(0) * QB
    ids = KEY_START + tl.program_id(1) * B + tl.arange(0, B)
    m = tl.arange(0, QB * HALF)
    dims = tl.arange(0, 128)
    queries = first_row + m // HALF
    heads = m % HALF
    q = tl.load(
        PLAN
        + queries[:, None] * (HALF + 1) * 128
        + heads[:, None] * 128
        + dims[None, :],
        queries[:, None] < ROWS,
        0.0,
    )
    k = tl.load(KEYS + dims[:, None] + ids[None, :] * 128, ids[None, :] < N, 0.0)
    dots = tl.dot(q.to(KEYS.dtype.element_ty), k, out_dtype=tl.float32)
    weights = tl.load(W + queries * HALF * 2 + heads, queries < ROWS, 0.0)
    positive = tl.reshape(tl.maximum(dots, 0.0) * weights[:, None], (QB, HALF, B))
    positive_sum = tl.sum(positive, axis=1)
    upper = positive_sum
    pick = tl.broadcast_to(tl.arange(0, QB)[:, None], (QB, B))
    if FP8_AGGREGATE:
        # |aggregate| <= 16 with 1/8 spacing. E4M3 rounding leaves a
        # residual of magnitude <= 1/2 on the same grid, exactly E4M3.
        # Pad to M=64 so both GEMMs can use Hopper WGMMA on the same K.
        neg_rows = tl.arange(0, 64)
        neg_queries = first_row + neg_rows % QB
        aggregate = tl.load(
            PLAN + neg_queries[:, None] * (HALF + 1) * 128 + HALF * 128 + dims[None, :],
            (neg_rows[:, None] < 2 * QB) & (neg_queries[:, None] < ROWS),
            0.0,
        ).to(tl.float32)
        hi = aggregate.to(KEYS.dtype.element_ty).to(tl.float32)
        negative = tl.where(neg_rows[:, None] < QB, hi, aggregate - hi)
        negative_dots = tl.dot(
            negative.to(KEYS.dtype.element_ty), k, out_dtype=tl.float32
        )
        aggregate_dot = tl.gather(negative_dots, pick, axis=0)
        aggregate_dot += tl.gather(negative_dots, pick + QB, axis=0)
    else:
        neg_rows = tl.arange(0, 16)
        negative = tl.load(
            PLAN
            + (first_row + neg_rows[:, None]) * (HALF + 1) * 128
            + HALF * 128
            + dims[None, :],
            (neg_rows[:, None] < QB) & (first_row + neg_rows[:, None] < ROWS),
            0.0,
        )
        negative_dots = tl.dot(negative, k.to(tl.float16), out_dtype=tl.float32)
        aggregate_dot = tl.gather(negative_dots, pick, axis=0)
    upper -= tl.maximum(aggregate_dot, 0.0)
    rows = first_row + tl.arange(0, QB)
    tau = tl.load(TAU + rows, rows < ROWS, float("inf"))
    keep = (upper >= tau[:, None]) & (rows[:, None] < ROWS) & (ids[None, :] < N)
    if COMPACT:
        rank = tl.cumsum(keep.to(tl.int32), axis=1) - 1
        total = tl.sum(keep.to(tl.int32), axis=1)
        base = tl.atomic_add(
            COUNTS_OR_LIVE + rows * COUNT_STRIDE, total, rows < ROWS, sem="relaxed"
        )
        tl.store(OUT + rows[:, None] * N + base[:, None] + rank, ids[None, :], keep)
        if REUSE_POSITIVE:
            tl.store(
                PARTIAL + rows[:, None] * N + base[:, None] + rank, positive_sum, keep
            )
    else:
        mask = (rows[:, None] < ROWS) & (ids[None, :] < N)
        tl.store(OUT + rows[:, None] * N + ids[None, :], upper, mask)
        tl.store(COUNTS_OR_LIVE + rows[:, None] * N + ids[None, :], keep, mask)


def screen_exact_integer_domain(
    q,
    k,
    weights,
    tau,
    *,
    compact=False,
    split_negative=False,
    block_keys=128,
    num_warps=4,
    query_block=1,
    _partials=None,
    fp8_aggregate=False,
):
    """Return upper bounds and survivors in the restricted exact domain.

    Q/K must be integers in [-2, 2], D=128, H=32/64. The first H/2 weights
    are +1/8 or +1/4 and the rest are -1/8 or -1/4. K scales are unit.
    These bounds keep all intermediate binary values exactly representable
    in FP16 inputs and FP32 accumulators. Validation/setup is not timed as
    kernel execution. Reject all other data until a general guard is added.
    """
    assert q.is_cuda and k.device == q.device and weights.device == q.device
    assert q.ndim == 3 and q.shape[1] in (32, 64) and q.shape[2] == 128
    assert k.ndim == 2 and k.shape[1] == 128 and k.is_contiguous()
    assert weights.shape == q.shape[:2] and tau.shape == q.shape[:1]
    assert tau.device == q.device and weights.dtype == torch.float32
    assert tau.dtype in (torch.float32, torch.float64)
    assert q.dtype == k.dtype == torch.float8_e4m3fn
    assert q.is_contiguous() and weights.is_contiguous()
    assert q.shape[0] > 0 and k.shape[0] > 0
    # Address expressions in these research kernels use signed int32 offsets.
    assert max(q.numel(), k.numel(), q.shape[0] * k.shape[0]) < 2**31
    if _partials is not None:
        assert compact and query_block > 1
        assert _partials.shape == (q.shape[0], k.shape[0])
        assert _partials.device == q.device and _partials.dtype == torch.float32
        assert _partials.is_contiguous()
    half = q.shape[1] // 2
    bad = torch.zeros(1, dtype=torch.int32, device=q.device)
    threshold = torch.empty(q.shape[0], device=q.device, dtype=torch.float32)
    _admit[(triton.cdiv(max(q.numel(), k.numel()), 4096),)](
        q.view(torch.uint8),
        k.view(torch.uint8),
        weights,
        tau,
        threshold,
        bad,
        q.numel(),
        k.numel(),
        weights.numel(),
        q.shape[1],
        q.shape[0],
        tau.stride(0),
        4096,
    )
    if bad.item():
        raise ValueError("Unsupported input for exact integer-domain certificate")
    plan = torch.empty(
        (q.shape[0], half + 1, 128), device=q.device, dtype=torch.float16
    )
    _make_plan[(q.shape[0],)](q, weights, plan, q.shape[1])
    if compact:
        upper = torch.empty(
            (q.shape[0], k.shape[0]), device=q.device, dtype=torch.int32
        )
        live = torch.zeros((q.shape[0], 32), device=q.device, dtype=torch.int32)
    else:
        upper = torch.empty(
            (q.shape[0], k.shape[0]), device=q.device, dtype=torch.float32
        )
        live = torch.empty_like(upper, dtype=torch.bool)

    launches = 0

    def launch():
        nonlocal launches
        if compact and launches:
            raise RuntimeError(
                "Compacting launches require fresh counters; call the API again"
            )
        launches += 1
        if query_block > 1:
            assert query_block in (2, 4, 8, 16)
            return _screen_batch[
                (
                    triton.cdiv(q.shape[0], query_block),
                    triton.cdiv(k.shape[0], block_keys),
                )
            ](
                plan,
                weights,
                k,
                threshold,
                upper,
                live,
                _partials if _partials is not None else upper,
                k.shape[0],
                q.shape[0],
                half,
                query_block,
                block_keys,
                compact,
                _partials is not None,
                fp8_aggregate,
                32,
                num_warps=num_warps,
            )
        return _screen[(q.shape[0], triton.cdiv(k.shape[0], block_keys))](
            plan,
            weights,
            k,
            threshold,
            upper,
            live,
            upper,
            live,
            k.shape[0],
            half + 1,
            half if split_negative else triton.next_power_of_2(half + 1),
            block_keys,
            compact,
            split_negative,
            32,
            num_warps=num_warps,
        )

    launch.compiled_kernel = launch()
    return upper, live[:, 0].contiguous() if compact else live, launch


def fused_screen_refine_exact_integer_domain(
    q,
    k,
    weights,
    tau,
    *,
    split_negative=False,
    block_keys=128,
    refine_block=128,
    num_warps=4,
    query_block=1,
    reuse_positive=False,
    fp8_aggregate=False,
    persistent=0,
):
    """Fused screening/packing with the same fail-closed domain admission.

    Return packed key IDs, valid counts, and exact candidate scores. Atomic
    order is unspecified, as in the separate compaction implementation.
    """
    shape = (q.shape[0], k.shape[0])
    scores = (
        torch.full(shape, -float("inf"), device=q.device, dtype=torch.float32)
        if persistent
        else torch.empty(shape, device=q.device, dtype=torch.float32)
    )
    ids, counts, _ = screen_exact_integer_domain(
        q,
        k,
        weights,
        tau,
        compact=True,
        split_negative=split_negative,
        block_keys=block_keys,
        num_warps=num_warps,
        query_block=query_block,
        _partials=scores if reuse_positive else None,
        fp8_aggregate=fp8_aggregate,
    )
    if persistent:
        _refine_persistent[(q.shape[0], persistent)](
            q,
            weights,
            k,
            ids,
            counts,
            scores,
            k.shape[0],
            q.shape[1],
            refine_block,
            reuse_positive,
            persistent,
            num_warps=4,
        )
        return ids, counts, scores
    _refine_compact[(q.shape[0], triton.cdiv(k.shape[0], refine_block))](
        q,
        weights,
        k,
        ids,
        counts,
        scores,
        k.shape[0],
        q.shape[1],
        refine_block,
        reuse_positive,
        num_warps=4,
    )
    return ids, counts, scores


@triton.jit
def _refine_persistent(
    Q,
    W,
    K,
    IDS,
    COUNTS,
    SCORES,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    B: tl.constexpr,
    REUSE: tl.constexpr,
    CTAS: tl.constexpr,
    START: tl.constexpr = 0,
):
    row = tl.program_id(0)
    if REUSE:  # noqa: SIM108 - Triton branches select different tensor shapes.
        heads = tl.arange(0, HEADS // 2) + HEADS // 2
    else:
        heads = tl.arange(0, HEADS)
    dims = tl.arange(0, 128)
    q = tl.load(Q + row * HEADS * 128 + heads[:, None] * 128 + dims[None, :])
    weights = tl.load(W + row * HEADS + heads)
    count = tl.load(COUNTS + row)
    for first in range(START + tl.program_id(1) * B, count, CTAS * B):
        offsets = first + tl.arange(0, B)
        ids = tl.load(IDS + row * N + offsets, offsets < count, 0)
        k = tl.load(K + ids[None, :] * 128 + dims[:, None])
        dots = tl.dot(q, k, out_dtype=tl.float32)
        score = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)
        if REUSE:
            score += tl.load(SCORES + row * N + offsets, offsets < count, 0.0)
        tl.store(SCORES + row * N + offsets, score, offsets < count)


@triton.jit
def _refine(Q, WEIGHTS, KEYS, IDS, SCORES, N, HEADS: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    heads = tl.arange(0, HEADS)
    dims = tl.arange(0, 128)
    ids = tl.load(IDS + offsets, offsets < N, 0)
    q = tl.load(Q + heads[:, None] * 128 + dims[None, :]).to(tl.float16)
    k = tl.load(KEYS + ids[None, :] * 128 + dims[:, None]).to(tl.float16)
    dot = tl.dot(q, k, out_dtype=tl.float32)
    weights = tl.load(WEIGHTS + heads)
    score = tl.sum(tl.maximum(dot, 0.0) * weights[:, None], axis=0)
    tl.store(SCORES + offsets, score, offsets < N)


def refine_validated_integer_candidates(q_row, keys, weights_row, ids):
    """Refine candidates only after screen_exact_integer_domain validation.

    This internal research helper assumes the same validated input domain.
    There is no general-model-input API or serving integration.
    """
    assert ids.ndim == 1 and ids.is_cuda and ids.is_contiguous()
    assert ids.dtype in (torch.int32, torch.int64) and ids.numel() >= 2048
    scores = torch.empty(ids.numel(), device=keys.device, dtype=torch.float32)
    _refine[(triton.cdiv(ids.numel(), 128),)](
        q_row,
        weights_row,
        keys,
        ids,
        scores,
        ids.numel(),
        q_row.shape[0],
        128,
        num_warps=4,
    )
    return scores


@triton.jit
def _compact(LIVE, IDS, COUNTS, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    ids = tl.program_id(1) * B + tl.arange(0, B)
    keep = tl.load(LIVE + row * N + ids, ids < N, False).to(tl.int32)
    rank = tl.cumsum(keep) - 1
    total = tl.sum(keep)
    base = tl.atomic_add(COUNTS + row, total, sem="relaxed")
    tl.store(IDS + row * N + base + rank, ids, keep != 0)


@triton.jit
def _refine_compact(
    Q,
    WEIGHTS,
    KEYS,
    IDS,
    COUNTS,
    SCORES,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    B: tl.constexpr,
    REUSE_POSITIVE: tl.constexpr = False,
):
    row = tl.program_id(0)
    first = tl.program_id(1) * B
    offsets = first + tl.arange(0, B)
    count = tl.load(COUNTS + row)
    if first < count:
        if REUSE_POSITIVE:
            heads = tl.arange(0, HEADS // 2) + HEADS // 2
        else:
            heads = tl.arange(0, HEADS)
        dims = tl.arange(0, 128)
        ids = tl.load(IDS + row * N + offsets, offsets < count, 0)
        # Domain admission guarantees exact integer FP8 operands. Native FP8
        # MMA with FP32 accumulation preserves the same exact-score proof.
        q = tl.load(Q + row * HEADS * 128 + heads[:, None] * 128 + dims[None, :])
        k = tl.load(KEYS + ids[None, :] * 128 + dims[:, None])
        dots = tl.dot(q, k, out_dtype=tl.float32)
        weights = tl.load(WEIGHTS + row * HEADS + heads)
        score = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)
        if REUSE_POSITIVE:
            score += tl.load(SCORES + row * N + offsets, offsets < count, 0.0)
        score = tl.where(offsets < count, score, -float("inf"))
        tl.store(SCORES + row * N + offsets, score, offsets < N)
    else:
        tl.store(SCORES + row * N + offsets, -float("inf"), offsets < N)


def compact_and_refine_validated_integer_domain(q, k, weights, live):
    """GPU-only compaction/refinement after exact-domain admission.

    Atomic packing order is unspecified; callers must not assume native
    tie-breaking IDs. Invalid score-buffer slots are explicitly -infinity.
    This helper does not admit arbitrary inputs on its own.
    """
    rows, count = live.shape
    assert q.is_contiguous() and weights.is_contiguous()
    ids = torch.empty((rows, count), device=q.device, dtype=torch.int32)
    counts = torch.zeros(rows, device=q.device, dtype=torch.int32)
    scores = torch.empty((rows, count), device=q.device, dtype=torch.float32)
    _compact[(rows, triton.cdiv(count, 256))](
        live,
        ids,
        counts,
        count,
        256,
    )
    _refine_compact[(rows, triton.cdiv(count, 128))](
        q,
        weights,
        k,
        ids,
        counts,
        scores,
        count,
        q.shape[1],
        128,
        num_warps=4,
    )
    return ids, counts, scores


@triton.jit
def _retain_seed(
    SCORES,
    SEED_IDS,
    IDS,
    OUT,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, TOPK)
    ids = tl.load(SEED_IDS + row * TOPK + offsets)
    values = tl.load(SCORES + row * STRIDE + ids)
    tl.store(IDS + row * N + offsets, ids)
    tl.store(OUT + row * N + offsets, values)


class HopperGraphPipeline:
    """Prepared research pipeline over fixed tensor addresses.

    Replays admit current Q/K/W contents before dispatching computation.
    Seed cutoffs are internal exact finite values in the admitted domain.
    The output buffer is reused. Capture/setup is excluded from replay timing.
    """

    def __init__(self, q, k, weights, *, reuse_seed=True):
        from vllm import _custom_ops
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

        self.q, self.k, self.weights = q, k, weights
        self.signature = self._input_signature()
        self.rows, self.heads, _ = q.shape
        self.keys = k.shape[0]
        self.seed_keys, self.topk = 12288, 2048
        self.reuse_seed = reuse_seed
        assert self.keys >= self.seed_keys
        self.zero_tau = torch.zeros(self.rows, device=q.device, dtype=torch.float32)
        # Check metadata and initial contents through the public admission API.
        screen_exact_integer_domain(q, k, weights, self.zero_tau)
        self.bad = torch.zeros(1, device=q.device, dtype=torch.int32)
        self.threshold_scratch = torch.empty_like(self.zero_tau)
        self.plan = torch.empty(
            (self.rows, self.heads // 2 + 1, 128), device=q.device, dtype=torch.float16
        )
        self.ids = torch.empty(
            (self.rows, self.keys), device=q.device, dtype=torch.int32
        )
        self.padded_counts = torch.zeros(
            (self.rows, 32), device=q.device, dtype=torch.int32
        )
        self.counts = torch.empty(self.rows, device=q.device, dtype=torch.int32)
        self.scores = torch.empty(
            (self.rows, self.keys), device=q.device, dtype=torch.float32
        )
        self.positions = torch.empty(
            (self.rows, self.topk), device=q.device, dtype=torch.int32
        )
        self.output = torch.empty_like(self.positions)
        self.starts = torch.zeros(self.rows, device=q.device, dtype=torch.int32)
        self.seed_ends = torch.full_like(self.starts, self.seed_keys)
        self.scales = torch.ones(self.keys, device=q.device, dtype=torch.float32)
        self.native_logits = fp8_fp4_mqa_logits
        self.ops = _custom_ops
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._admit_work()
                self._compute_work()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.admission_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.admission_graph):
            self._admit_work()
        self.compute_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.compute_graph):
            self._compute_work()

    def _admit_work(self):
        self.bad.zero_()
        _admit[(triton.cdiv(max(self.q.numel(), self.k.numel()), 4096),)](
            self.q.view(torch.uint8),
            self.k.view(torch.uint8),
            self.weights,
            self.zero_tau,
            self.threshold_scratch,
            self.bad,
            self.q.numel(),
            self.k.numel(),
            self.weights.numel(),
            self.heads,
            self.rows,
            1,
            4096,
        )

    def _compute_work(self):
        seeds = self.native_logits(
            (self.q, None),
            (self.k[: self.seed_keys], self.scales[: self.seed_keys]),
            self.weights,
            self.starts,
            self.seed_ends,
            self.seed_keys,
        )
        seed_ids, tau = threshold_from_native_topk(
            seeds, self.starts, self.seed_ends, return_ids=True
        )
        _make_plan[(self.rows,)](self.q, self.weights, self.plan, self.heads)
        self.padded_counts.fill_(self.topk if self.reuse_seed else 0)
        # Every valid prefix slot is overwritten below. Native TopK uses
        # counts as row ends; storage beyond those ends is intentionally
        # unspecified, avoiding a full-capacity memset on every replay.
        if self.reuse_seed:
            _retain_seed[(self.rows,)](
                seeds,
                seed_ids,
                self.ids,
                self.scores,
                self.keys,
                seeds.stride(0),
                self.topk,
            )
        first_key = self.seed_keys if self.reuse_seed else 0
        _screen_batch[
            (triton.cdiv(self.rows, 8), triton.cdiv(self.keys - first_key, 64))
        ](
            self.plan,
            self.weights,
            self.k,
            tau,
            self.ids,
            self.padded_counts,
            self.scores,
            self.keys,
            self.rows,
            self.heads // 2,
            8,
            64,
            True,
            True,
            False,
            32,
            first_key,
            num_warps=4,
        )
        self.counts.copy_(self.padded_counts[:, 0])
        _refine_persistent[(self.rows, 32)](
            self.q,
            self.weights,
            self.k,
            self.ids,
            self.counts,
            self.scores,
            self.keys,
            self.heads,
            128,
            True,
            32,
            self.topk if self.reuse_seed else 0,
            num_warps=4,
        )
        self.ops.top_k_per_row_prefill(
            self.scores,
            self.starts,
            self.counts,
            self.positions,
            self.rows,
            self.keys,
            1,
            self.topk,
        )
        _map_topk[(triton.cdiv(self.positions.numel(), 256),)](
            self.ids,
            self.positions,
            self.output,
            self.keys,
            self.topk,
            self.positions.numel(),
            256,
        )

    def replay(self):
        if self._input_signature() != self.signature:
            raise ValueError(
                "Prepared graph requires its original tensor addresses and metadata"
            )
        self.admission_graph.replay()
        if self.bad.item():
            raise ValueError("Unsupported input for exact integer-domain certificate")
        self.compute_graph.replay()
        return self.output

    def _input_signature(self):
        return tuple(
            (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
            for x in (self.q, self.k, self.weights)
        )
