"""Real-trace scheduling ablation; each configuration checks all native TopK."""
# Closures are invoked synchronously within their own iteration, never queued.
# ruff: noqa: B023, C408

import gc
import json
import statistics
from pathlib import Path

import torch
from benchmark_real_indexer import rescale_query_weights
from dosa_sm90 import HopperIndexer
from safetensors.torch import load_file

from vllm import _custom_ops as ops
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits


def run(directory, sample16=False):
    directory = Path(directory)
    trace = Path(
        "/experiments/h100-real-indexer/33f5da8eabdd8f77a1733ab322312f7a4653e23e/20261030T024618.353787Z-369dfb2b/layer0-inputs.safetensors"
    )
    if sample16:
        trace = Path(
            json.loads(Path("/opt/sample16-manifest.json").read_text())["trace"]
        )
    data = load_file(str(trace), device="cuda")
    results = []
    for nk in (131072, 262144):
        q = data[f"q_{nk}"]
        w = data[f"weights_{nk}"].float().contiguous()
        keys = data["keys"][:nk]
        scales = data["scales"][:nk]
        rows = len(q)
        starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
        ends = torch.arange(nk - rows + 1, nk + 1, device="cuda", dtype=torch.int32)
        output = torch.empty((rows, 2048), device="cuda", dtype=torch.int32)

        def logits(b, e):
            return fp8_fp4_mqa_logits(
                (q[b:e], None), (keys, scales), w[b:e], starts[b:e], ends[b:e], nk
            )

        def stock():
            chunk = (512 * 1024**2) // (nk * 4)
            for b in range(0, rows, chunk):
                e = min(rows, b + chunk)
                score = logits(b, e)
                ops.top_k_per_row_prefill(
                    score,
                    starts[b:e],
                    ends[b:e],
                    output[b:e],
                    e - b,
                    score.stride(0),
                    1,
                    2048,
                )

        def check():
            for b in range(0, rows, 128):
                e = min(rows, b + 128)
                score = logits(b, e)
                score.masked_fill_(
                    torch.arange(nk, device="cuda")[None, :] >= ends[b:e, None],
                    -torch.inf,
                )
                selected = output[b:e].long()
                assert bool(((selected >= 0) & (selected < ends[b:e, None])).all())
                assert bool((selected.sort(1).values.diff(dim=1) > 0).all())
                assert torch.equal(
                    score.gather(1, selected).sort(1).values,
                    score.topk(2048, dim=1).values.sort(1).values,
                )

        def timing(action):
            for _ in range(3):
                action()
            samples = []
            for _ in range(9):
                a, z = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                a.record()
                action()
                z.record()
                z.synchronize()
                samples.append(a.elapsed_time(z))
            return samples

        baseline = timing(stock)
        check()
        configurations = (
            (False, 4, 4),
            (True, 4, 4),
            (True, 1, 4),
            (True, 16, 4),
            (True, 4, 1),
            (True, 4, 16),
        )
        if sample16:
            configurations = configurations[:2]
        for sorted_queries, screen_ctas, refine_ctas in configurations:
            sq = torch.empty_like(q)
            sw = torch.empty_like(w)
            ss, se = torch.empty_like(starts), torch.empty_like(ends)

            def stage():
                nonlocal order
                scaled, _ = rescale_query_weights(w)
                if sorted_queries:
                    positives = (w > 0).sum(1)
                    tags = (
                        (positives > 14).long()
                        + (positives > 16).long()
                        + (positives > 20).long()
                    )
                    order = tags.argsort(stable=True)
                    torch.index_select(
                        q.view(torch.uint8), 0, order, out=sq.view(torch.uint8)
                    )
                    torch.index_select(scaled, 0, order, out=sw)
                    torch.index_select(starts, 0, order, out=ss)
                    torch.index_select(ends, 0, order, out=se)
                else:
                    sq.copy_(q)
                    sw.copy_(scaled)
                    ss.copy_(starts)
                    se.copy_(ends)

            order = None
            stage()
            owners = [
                HopperIndexer(
                    sq[b : b + 4096],
                    keys,
                    sw[b : b + 4096],
                    scales,
                    ss[b : b + 4096],
                    se[b : b + 4096],
                    common_end=int(ends.min()),
                    paid=16,
                    native_partial=True,
                    shared_refine=True,
                    screen_ctas=screen_ctas,
                    refine_ctas=refine_ctas,
                    allow_unproven_bound=True,
                )
                for b in range(0, rows, 4096)
            ]
            sorted_output = torch.empty_like(output)

            def action():
                stage()
                for i, owner in enumerate(owners):
                    sorted_output[i * 4096 : (i + 1) * 4096].copy_(owner.replay())
                value = sorted_output
                if sorted_queries:
                    output.index_copy_(0, order, value)
                else:
                    output.copy_(value)

            action()
            check()
            samples = timing(action)
            check()
            result = dict(
                keys=nk,
                sorted_queries=sorted_queries,
                screen_ctas=screen_ctas,
                refine_ctas=refine_ctas,
                stock_ms=statistics.median(baseline),
                port_ms=statistics.median(samples),
                speedup=statistics.median(baseline) / statistics.median(samples),
                samples=samples,
                rows=rows,
                execution_chunk_rows=4096,
                correctness=f"ALL_{rows}_NATIVE_TOP2048_EXACT",
                budget_histogram=torch.cat([o.partial_budget for o in owners])
                .bincount(minlength=33)
                .cpu()
                .tolist(),
                screen_candidates_mean=torch.cat([o.screen_counts for o in owners])
                .float()
                .mean()
                .item(),
                union_keys_mean=torch.cat([o.union_counts for o in owners])
                .float()
                .mean()
                .item(),
            )
            results.append(result)
            (directory / "sweep-result.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
            owners = sorted_output = sq = sw = ss = se = None
            gc.collect()
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    import sys

    run(sys.argv[1], sample16=len(sys.argv) > 2)
