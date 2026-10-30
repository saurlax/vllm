# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-domain DoSA negative-aggregate kernel experiments on SM90.

Retain every positive head; bound omitted negative heads by ReLU of their
weighted aggregate. Refine only candidates whose upper bound reaches the seed
threshold. Triton kernels are checked against FP64 references. This is a
restricted-domain research benchmark, not a complete DoSA serving backend.
"""

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--keys", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from dosa_portable import (
        HopperGraphPipeline,
        compact_and_refine_validated_integer_domain,
        dense_validated_integer_domain,
        fused_screen_refine_exact_integer_domain,
        map_compact_topk,
        refine_validated_integer_candidates,
        screen_exact_integer_domain,
        threshold_from_native_topk,
    )

    from vllm import _custom_ops as stock_ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    assert torch.cuda.get_device_capability() == (9, 0)
    assert "H100" in torch.cuda.get_device_name()
    rows, count, dim, seed_count, topk = args.rows, args.keys, 128, 12288, 2048
    assert rows > 0 and count >= seed_count
    report = {
        "experiment": "Functional subset of the DoSA certificate, executed on H100",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": rows,
        "keys": count,
        "seed_keys": seed_count,
        "topk": topk,
        "implementation": (
            "Native SM90 seed; Triton FP16/FP32 bound and candidate "
            "refinement; GPU compaction; native/Torch TopK; FP64 references"
        ),
        "certificate": (
            "All positive heads retained; omitted negative heads lower-"
            "bounded by ReLU(sum abs(weight)*query dot key)"
        ),
        "numerical_domain": (
            "Bounded integer FP8 Q/K and binary-fraction weights with "
            "unit scales; FP64 arithmetic is exact in this domain"
        ),
        "limitations": [
            "Not the author's optimized SM100 backend",
            "No general FP32 error certificate yet",
            "No complete Hopper DoSA backend or serving integration",
            "No real-model latency or TTFT claim",
            (
                "Independent and artificially correlated synthetic data, not "
                "model activations"
            ),
        ],
        "cases": [],
    }
    for heads in (32, 64):
        for distribution in ("independent", "correlated_negative_heads"):
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            q = torch.randint(
                -2, 3, (rows, heads, dim), device="cuda", generator=generator
            ).to(torch.float8_e4m3fn)
            if distribution == "correlated_negative_heads":
                qf = q.float()
                qf[:, heads // 2 :] = qf[:, heads // 2 : heads // 2 + 1]
                q = qf.to(torch.float8_e4m3fn)
            k = torch.randint(
                -2, 3, (count, dim), device="cuda", generator=generator
            ).to(torch.float8_e4m3fn)
            w = (
                torch.randint(
                    1, 3, (rows, heads), device="cuda", generator=generator
                ).float()
                / 8
            )
            w[:, heads // 2 :] *= -1
            scales = torch.ones(count, device="cuda")
            starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
            seed_ends = torch.full_like(starts, seed_count)
            seed_scores = fp8_fp4_mqa_logits(
                (q, None),
                (k[:seed_count].contiguous(), scales[:seed_count]),
                w,
                starts,
                seed_ends,
                seed_count,
            )
            tau = seed_scores.topk(topk, dim=1).values[:, -1].double()
            native_tau = threshold_from_native_topk(seed_scores, starts, seed_ends)
            assert torch.equal(native_tau.double(), tau), "Native seed cutoff mismatch"
            qd, kd, wd = q.double(), k.double(), w.double()
            upper = torch.zeros((rows, count), device="cuda", dtype=torch.float64)
            for h in range(heads // 2):
                upper.add_((qd[:, h] @ kd.T).relu() * wd[:, h, None])
            negative_direction = (qd[:, heads // 2 :] * -wd[:, heads // 2 :, None]).sum(
                dim=1
            )
            upper.sub_((negative_direction @ kd.T).relu())
            negative_hi = negative_direction.to(torch.float8_e4m3fn).double()
            negative_lo = (
                (negative_direction - negative_hi).to(torch.float8_e4m3fn).double()
            )
            assert torch.equal(negative_hi + negative_lo, negative_direction)
            candidate_mask = upper >= tau[:, None]
            kernel_upper, kernel_live, launch = screen_exact_integer_domain(
                q, k, w, tau
            )
            assert torch.equal(kernel_upper.double(), upper), (
                "Triton upper differs from exact reference"
            )
            assert torch.equal(kernel_live, candidate_mask), (
                "Triton prunes a different candidate set"
            )
            split_upper, split_live, _ = screen_exact_integer_domain(
                q, k, w, tau, split_negative=True
            )
            assert torch.equal(split_upper.double(), upper), "Split bound mismatch"
            assert torch.equal(split_live, candidate_mask), "Split mask mismatch"
            for query_block in (2, 4):
                batched_upper, batched_live, _ = screen_exact_integer_domain(
                    q, k, w, tau, query_block=query_block
                )
                assert torch.equal(batched_upper.double(), upper), (
                    "Batched bound mismatch"
                )
                assert torch.equal(batched_live, candidate_mask), (
                    "Batched mask mismatch"
                )
            fp8_upper, fp8_live, _ = screen_exact_integer_domain(
                q, k, w, tau, query_block=4, fp8_aggregate=True
            )
            assert torch.equal(fp8_upper.double(), upper), (
                "FP8 split aggregate bound mismatch"
            )
            assert torch.equal(fp8_live, candidate_mask), (
                "FP8 split aggregate mask mismatch"
            )
            if distribution == "independent":
                scratch = torch.empty((rows, count), device="cuda")
                _, _, packed_launch = screen_exact_integer_domain(
                    q,
                    k,
                    w,
                    tau,
                    compact=True,
                    query_block=8,
                    block_keys=64,
                    _partials=scratch,
                )
                compiled = packed_launch.compiled_kernel
                refused_replay = False
                try:
                    packed_launch()
                except RuntimeError:
                    refused_replay = True
                assert refused_replay, "Compaction must not reuse unreset counters"
                metadata = {
                    "registers": getattr(compiled, "n_regs", None),
                    "spills": getattr(compiled, "n_spills", None),
                    "shared_bytes": getattr(compiled.metadata, "shared", None),
                }
                (out / f"batch-kernel-h{heads}.json").write_text(
                    json.dumps(metadata, indent=2)
                )
                (out / f"batch-kernel-h{heads}.ptx").write_text(compiled.asm["ptx"])
            # Include equality at the threshold, not only seed-derived thresholds.
            boundary_tau = upper[:, 0].contiguous()
            _, boundary_live, _ = screen_exact_integer_domain(q, k, w, boundary_tau)
            assert torch.all(boundary_live[:, 0]), "Equality must survive"
            assert torch.equal(boundary_live, upper >= boundary_tau[:, None])
            for _ in range(3):
                launch()
            torch.cuda.synchronize()
            kernel_samples = []
            for _ in range(10):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                launch()
                end.record()
                end.synchronize()
                kernel_samples.append(begin.elapsed_time(end))
            candidate_counts, selected_values = [], []
            for row in range(rows):
                ids = kernel_live[row].nonzero().flatten()
                assert ids.numel() >= topk
                candidate_counts.append(ids.numel())
                dots = qd[row] @ kd[ids].T
                refined = (dots.relu() * wd[row, :, None]).sum(dim=0)
                actual_refined = refine_validated_integer_candidates(
                    q[row], k, w[row], ids
                )
                assert torch.equal(actual_refined.double(), refined), (
                    "Candidate refinement mismatch"
                )
                selected_values.append(refined.topk(topk).values.float())
            ends = torch.full_like(starts, count)
            # Dense native scores are computed only now, as an independent oracle.
            native = fp8_fp4_mqa_logits((q, None), (k, scales), w, starts, ends, count)
            assert torch.all(upper >= native.double()), (
                "Invalid upper-bound certificate"
            )
            assert torch.equal(
                torch.stack(selected_values), native.topk(topk, dim=1).values
            )
            assert torch.all(candidate_mask | (native.double() < tau[:, None]))
            output = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
            dense_scores = dense_validated_integer_domain(q, k, w)
            assert torch.equal(dense_scores, native), "Dense FP8 ablation mismatch"

            def dense_pipeline(q=q, k=k, w=w, starts=starts, ends=ends, output=output):
                scores = dense_validated_integer_domain(q, k, w)
                stock_ops.top_k_per_row_prefill(
                    scores,
                    starts,
                    ends,
                    output,
                    rows,
                    scores.stride(0),
                    scores.stride(1),
                    topk,
                )
                return output

            def stock_pipeline(
                q=q, k=k, scales=scales, w=w, starts=starts, ends=ends, output=output
            ):
                scores = fp8_fp4_mqa_logits(
                    (q, None), (k, scales), w, starts, ends, count
                )
                stock_ops.top_k_per_row_prefill(
                    scores,
                    starts,
                    ends,
                    output,
                    rows,
                    scores.stride(0),
                    scores.stride(1),
                    topk,
                )
                return output

            def portable_pipeline(
                q=q, k=k, scales=scales, w=w, starts=starts, seed_ends=seed_ends
            ):
                seeds = fp8_fp4_mqa_logits(
                    (q, None),
                    (k[:seed_count].contiguous(), scales[:seed_count]),
                    w,
                    starts,
                    seed_ends,
                    seed_count,
                )
                threshold = seeds.topk(topk, dim=1).values[:, -1]
                _, live, _ = screen_exact_integer_domain(q, k, w, threshold)
                result = []
                for row in range(rows):
                    ids = live[row].nonzero().flatten()
                    scores = refine_validated_integer_candidates(q[row], k, w[row], ids)
                    selected = scores.topk(topk).indices
                    result.append(ids[selected].to(torch.int32))
                return torch.stack(result)

            def portable_gpu_pipeline(
                q=q, k=k, scales=scales, w=w, starts=starts, seed_ends=seed_ends
            ):
                seeds = fp8_fp4_mqa_logits(
                    (q, None),
                    (k[:seed_count].contiguous(), scales[:seed_count]),
                    w,
                    starts,
                    seed_ends,
                    seed_count,
                )
                threshold = seeds.topk(topk, dim=1).values[:, -1]
                _, live, _ = screen_exact_integer_domain(q, k, w, threshold)
                ids, counts, scores = compact_and_refine_validated_integer_domain(
                    q, k, w, live
                )
                positions = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
                stock_ops.top_k_per_row_prefill(
                    scores,
                    starts,
                    counts,
                    positions,
                    rows,
                    scores.stride(0),
                    scores.stride(1),
                    topk,
                )
                return map_compact_topk(ids, positions)

            def fused_pipeline(
                native_seed=False,
                split_negative=False,
                tiles=(128, 128, 4),
                q=q,
                k=k,
                scales=scales,
                w=w,
                starts=starts,
                seed_ends=seed_ends,
            ):
                seeds = fp8_fp4_mqa_logits(
                    (q, None),
                    (k[:seed_count], scales[:seed_count]),
                    w,
                    starts,
                    seed_ends,
                    seed_count,
                )
                threshold = (
                    threshold_from_native_topk(seeds, starts, seed_ends)
                    if native_seed
                    else seeds.topk(topk, dim=1).values[:, -1]
                )
                ids, counts, scores = fused_screen_refine_exact_integer_domain(
                    q,
                    k,
                    w,
                    threshold,
                    split_negative=split_negative,
                    block_keys=tiles[0],
                    refine_block=tiles[1],
                    num_warps=tiles[2],
                    query_block=tiles[3] if len(tiles) > 3 else 1,
                    reuse_positive=tiles[4] if len(tiles) > 4 else False,
                    fp8_aggregate=tiles[5] if len(tiles) > 5 else False,
                    persistent=tiles[6] if len(tiles) > 6 else 0,
                )
                positions = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
                stock_ops.top_k_per_row_prefill(
                    scores,
                    starts,
                    counts,
                    positions,
                    rows,
                    scores.stride(0),
                    scores.stride(1),
                    topk,
                )
                return map_compact_topk(ids, positions)

            chosen = portable_pipeline()
            assert torch.all((chosen >= 0) & (chosen < count))
            assert torch.all(chosen.sort(dim=1).values.diff(dim=1) > 0)
            chosen_scores = (
                native.gather(1, chosen.long()).sort(dim=1, descending=True).values
            )
            assert torch.equal(chosen_scores, native.topk(topk, dim=1).values)
            packed_ids, packed_counts, packed_scores = (
                compact_and_refine_validated_integer_domain(q, k, w, kernel_live)
            )
            assert torch.equal(packed_counts.long(), kernel_live.sum(dim=1))
            for row in range(rows):
                valid_count = int(packed_counts[row].item())
                valid_ids = packed_ids[row, :valid_count].long()
                assert torch.equal(
                    valid_ids.sort().values, kernel_live[row].nonzero().flatten()
                )
                assert torch.equal(
                    packed_scores[row, :valid_count], native[row, valid_ids]
                )
                assert torch.all(torch.isneginf(packed_scores[row, valid_count:]))
            gpu_chosen = portable_gpu_pipeline()
            assert torch.all((gpu_chosen >= 0) & (gpu_chosen < count))
            assert torch.all(gpu_chosen.sort(dim=1).values.diff(dim=1) > 0)
            assert torch.equal(
                native.gather(1, gpu_chosen.long()).sort(dim=1, descending=True).values,
                native.topk(topk, dim=1).values,
            )
            fused_ids, fused_counts, fused_scores = (
                fused_screen_refine_exact_integer_domain(q, k, w, tau)
            )
            assert torch.equal(fused_counts.long(), kernel_live.sum(dim=1))
            for row in range(rows):
                size = int(fused_counts[row])
                fi = fused_ids[row, :size].long()
                assert torch.equal(
                    fi.sort().values, kernel_live[row].nonzero().flatten()
                )
                assert torch.equal(fused_scores[row, :size], native[row, fi])
                assert torch.all(torch.isneginf(fused_scores[row, size:]))
            fused_chosen = fused_pipeline()
            assert torch.all((fused_chosen >= 0) & (fused_chosen < count))
            assert torch.all(fused_chosen.sort(dim=1).values.diff(dim=1) > 0)
            assert torch.equal(
                native.gather(1, fused_chosen.long())
                .sort(dim=1, descending=True)
                .values,
                native.topk(topk, dim=1).values,
            )
            native_seed_chosen = fused_pipeline(native_seed=True)
            assert torch.all((native_seed_chosen >= 0) & (native_seed_chosen < count))
            assert torch.all(native_seed_chosen.sort(dim=1).values.diff(dim=1) > 0)
            assert torch.equal(
                native.gather(1, native_seed_chosen.long())
                .sort(dim=1, descending=True)
                .values,
                native.topk(topk, dim=1).values,
            )
            split_chosen = fused_pipeline(native_seed=True, split_negative=True)
            assert torch.all((split_chosen >= 0) & (split_chosen < count))
            assert torch.all(split_chosen.sort(dim=1).values.diff(dim=1) > 0)
            assert torch.equal(
                native.gather(1, split_chosen.long())
                .sort(dim=1, descending=True)
                .values,
                native.topk(topk, dim=1).values,
            )

            def timed(action):
                for _ in range(3):
                    action()
                torch.cuda.synchronize()
                values = []
                for _ in range(10):
                    begin = time.perf_counter()
                    action()
                    torch.cuda.synchronize()
                    values.append((time.perf_counter() - begin) * 1000)
                return values

            def hopper_pipeline(heads=heads, fused_pipeline=fused_pipeline):
                return fused_pipeline(
                    native_seed=True,
                    split_negative=heads == 64,
                    tiles=(256, 128, 4) if heads == 64 else (128, 128, 4),
                )

            optimized = hopper_pipeline()
            assert torch.all((optimized >= 0) & (optimized < count))
            assert torch.all(optimized.sort(dim=1).values.diff(dim=1) > 0)
            assert torch.equal(
                native.gather(1, optimized.long()).sort(dim=1, descending=True).values,
                native.topk(topk, dim=1).values,
            )

            tile_configs = {
                "tiles256": (256, 256, 4),
                "screen256": (256, 128, 4),
                "refine256": (128, 256, 4),
                "screen256_w8": (256, 256, 8),
                "batch2": (128, 128, 4, 2),
                "batch4": (128, 128, 4, 4),
                "batch4_reuse": (128, 128, 4, 4, True),
                "batch8_reuse": (128, 128, 8, 8, True),
                "batch16_reuse": (64, 128, 8, 16, True),
                "batch8_64_reuse": (64, 128, 4, 8, True),
                "batch4_fp8": (128, 128, 4, 4, True, True),
                "batch8_fp8": (64, 128, 4, 8, True, True),
                "batch16_fp8": (64, 128, 8, 16, True, True),
                "batch8_p16": (64, 128, 4, 8, True, False, 16),
                "batch8_p32": (64, 128, 4, 8, True, False, 32),
                "batch4_p32": (128, 128, 4, 4, True, False, 32),
                "batch8_p32_full": (64, 128, 4, 8, False, False, 32),
            }
            if rows > 16:
                tile_configs = {
                    name: value
                    for name, value in tile_configs.items()
                    if name.startswith("batch")
                }
            for tile_name, tiles in tile_configs.items():
                if (len(tiles) > 4 and tiles[4]) or len(tiles) > 6:
                    ri, rc, rs = fused_screen_refine_exact_integer_domain(
                        q,
                        k,
                        w,
                        tau,
                        query_block=tiles[3],
                        block_keys=tiles[0],
                        refine_block=tiles[1],
                        num_warps=tiles[2],
                        reuse_positive=tiles[4],
                        fp8_aggregate=tiles[5] if len(tiles) > 5 else False,
                        persistent=tiles[6] if len(tiles) > 6 else 0,
                    )
                    assert torch.equal(rc.long(), kernel_live.sum(dim=1)), tile_name
                    for row in range(rows):
                        size = int(rc[row])
                        ix = ri[row, :size].long()
                        assert torch.equal(
                            ix.sort().values, kernel_live[row].nonzero().flatten()
                        ), tile_name
                        assert torch.equal(rs[row, :size], native[row, ix]), tile_name
                        assert torch.all(torch.isneginf(rs[row, size:])), tile_name
                tiled = fused_pipeline(
                    native_seed=True, split_negative=True, tiles=tiles
                )
                assert torch.all((tiled >= 0) & (tiled < count)), tile_name
                assert torch.all(tiled.sort(dim=1).values.diff(dim=1) > 0), tile_name
                assert torch.equal(
                    native.gather(1, tiled.long()).sort(dim=1, descending=True).values,
                    native.topk(topk, dim=1).values,
                ), tile_name

            # Host wall time includes nonzero synchronization, validation and
            # allocations. Do not hide these prototype costs behind kernel timing.
            full_samples = {
                "stock": timed(stock_pipeline),
                "portable_hopper": timed(hopper_pipeline),
                "portable_dense_ablation": timed(dense_pipeline),
                **({"portable": timed(portable_pipeline)} if rows <= 16 else {}),
                "portable_gpu_compaction": timed(portable_gpu_pipeline),
                "portable_fused": timed(fused_pipeline),
                "portable_fused_native_seed": timed(
                    lambda: fused_pipeline(native_seed=True)
                ),
                "portable_fused_split": timed(
                    lambda: fused_pipeline(native_seed=True, split_negative=True)
                ),
                "portable_split_torch_seed": timed(
                    lambda: fused_pipeline(split_negative=True)
                ),
            }
            # Alternate order to reduce clock drift and run-order bias. Keep
            # admission, allocations, and final synchronization in both paths.
            paired = {
                "stock": [],
                "portable_hopper": [],
                "portable_dense_ablation": [],
                "portable_fused_native_seed": [],
                "portable_fused_split": [],
                "portable_split_torch_seed": [],
            }
            if rows > 16:
                paired = {
                    name: paired[name]
                    for name in ("stock", "portable_hopper", "portable_dense_ablation")
                }
            paired.update({name: [] for name in tile_configs})
            for repetition in range(30):
                names = list(paired)
                if repetition % 2:
                    names.reverse()
                for name in names:
                    begin = time.perf_counter()
                    if name == "stock":
                        stock_pipeline()
                    elif name == "portable_hopper":
                        hopper_pipeline()
                    elif name == "portable_dense_ablation":
                        dense_pipeline()
                    elif name in tile_configs:
                        fused_pipeline(
                            native_seed=True,
                            split_negative=True,
                            tiles=tile_configs[name],
                        )
                    else:
                        fused_pipeline(
                            native_seed=name != "portable_split_torch_seed",
                            split_negative=name
                            in ("portable_fused_split", "portable_split_torch_seed"),
                        )
                    torch.cuda.synchronize()
                    paired[name].append((time.perf_counter() - begin) * 1000)
            if heads == 32 and distribution == "independent":

                def reject(xq, xk, xw, xtau):
                    try:
                        screen_exact_integer_domain(xq, xk, xw, xtau)
                    except (ValueError, AssertionError):
                        return
                    raise AssertionError("Unsafe input was admitted")

                badq = q.clone()
                badq.view(torch.uint8).reshape(-1)[0] = 68  # FP8 value 3
                reject(badq, k, w, tau)
                badk = k.clone()
                badk.view(torch.uint8).reshape(-1)[-1] = 127  # FP8 NaN
                reject(q, badk, w, tau)
                badw = w.clone()
                badw[0, 0] *= -1
                reject(q, k, badw, tau)
                for invalid in (float("nan"), float("inf")):
                    badtau = tau.clone()
                    badtau[0] = invalid
                    reject(q, k, w, badtau)
                reject(q, k, w, tau.double() + 2.0**-40)
                report["unsafe_input_rejection_tests"] = (
                    "PASSED: invalid Q/K, wrong weight sign, NaN/Inf/non-"
                    "FP32-exact thresholds"
                )
                try:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                    ) as profile:
                        with torch.profiler.record_function("stock_complete"):
                            stock_pipeline()
                            torch.cuda.synchronize()
                        with torch.profiler.record_function("portable_complete"):
                            portable_gpu_pipeline()
                            torch.cuda.synchronize()
                        with torch.profiler.record_function("portable_batch8_reuse"):
                            fused_pipeline(
                                native_seed=True,
                                split_negative=True,
                                tiles=(64, 128, 4, 8, True),
                            )
                            torch.cuda.synchronize()
                        with torch.profiler.record_function("portable_dense"):
                            dense_pipeline()
                            torch.cuda.synchronize()
                        with torch.profiler.record_function("portable_persistent"):
                            fused_pipeline(
                                native_seed=True,
                                split_negative=True,
                                tiles=(64, 128, 4, 8, True, False, 32),
                            )
                            torch.cuda.synchronize()
                    profile.export_chrome_trace(str(out / "pipeline.trace.json"))
                    (out / "profile-cpu.txt").write_text(
                        profile.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=40
                        )
                    )
                    (out / "profile-cuda.txt").write_text(
                        profile.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=40
                        )
                    )
                    report["profiler"] = (
                        "pipeline.trace.json; profiling is outside timing samples"
                    )
                except Exception:
                    import traceback

                    (out / "profiler-error.txt").write_text(traceback.format_exc())
                    report["profiler"] = "unavailable; inspect profiler-error.txt"

            def capture(action):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        result = action()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = action()
                return graph, result

            prepare_start = time.perf_counter()
            stock_graph, stock_graph_out = capture(stock_pipeline)
            dense_graph, dense_graph_out = capture(dense_pipeline)
            portable_graph = HopperGraphPipeline(q, k, w)
            prepare_ms = (time.perf_counter() - prepare_start) * 1000
            graph_chosen = portable_graph.replay()
            prepared_counts = portable_graph.counts.cpu().tolist()
            assert torch.all((graph_chosen >= 0) & (graph_chosen < count))
            assert torch.all(graph_chosen.sort(dim=1).values.diff(dim=1) > 0)
            assert torch.equal(
                native.gather(1, graph_chosen.long())
                .sort(dim=1, descending=True)
                .values,
                native.topk(topk, dim=1).values,
            )
            # Poison all storage, including inactive slots. Replay must fully
            # overwrite active prefixes and must exclude the +inf/-1 tails.
            portable_graph.scores.fill_(float("inf"))
            portable_graph.ids.fill_(-1)
            poison_chosen = portable_graph.replay()
            assert torch.all((poison_chosen >= 0) & (poison_chosen < count))
            assert torch.all(poison_chosen.sort(dim=1).values.diff(dim=1) > 0)
            assert torch.equal(
                native.gather(1, poison_chosen.long())
                .sort(dim=1, descending=True)
                .values,
                native.topk(topk, dim=1).values,
            )
            for row in range(rows):
                size = int(portable_graph.counts[row])
                ix = portable_graph.ids[row, :size].long()
                assert torch.all((ix >= 0) & (ix < count))
                assert torch.all(ix.sort().values.diff() > 0)
                assert torch.equal(portable_graph.scores[row, :size], native[row, ix])
            # Mutate valid contents in place: replay must not return a cached result.
            qb = q.view(torch.uint8).reshape(-1)
            original = int(qb[0])
            qb[0] = 56 if original & 127 == 0 else 0
            changed_native = fp8_fp4_mqa_logits(
                (q, None), (k, scales), w, starts, ends, count
            )
            changed_chosen = portable_graph.replay()
            assert torch.equal(
                changed_native.gather(1, changed_chosen.long())
                .sort(dim=1, descending=True)
                .values,
                changed_native.topk(topk, dim=1).values,
            )
            size = int(portable_graph.counts[0])
            assert torch.equal(
                portable_graph.scores[0, :size],
                changed_native[0, portable_graph.ids[0, :size].long()],
            )
            qb[0] = 68
            rejected = False
            try:
                portable_graph.replay()
            except ValueError:
                rejected = True
            finally:
                qb[0] = original
            assert rejected, "Prepared replay admitted an unsupported FP8 value"
            original_tensor = portable_graph.q
            portable_graph.q = q.clone()
            rejected_pointer = False
            try:
                portable_graph.replay()
            except ValueError:
                rejected_pointer = True
            finally:
                portable_graph.q = original_tensor
            assert rejected_pointer, "Prepared replay admitted changed tensor addresses"
            recovered = portable_graph.replay()
            assert torch.equal(
                native.gather(1, recovered.long()).sort(dim=1, descending=True).values,
                native.topk(topk, dim=1).values,
            )
            graph_actions = {
                "stock": stock_graph.replay,
                "dense": dense_graph.replay,
                "portable": portable_graph.replay,
            }
            graph_samples = {name: [] for name in graph_actions}
            for repetition in range(30):
                names = list(graph_actions)
                if repetition % 2:
                    names.reverse()
                for name in names:
                    begin = time.perf_counter()
                    graph_actions[name]()
                    torch.cuda.synchronize()
                    graph_samples[name].append((time.perf_counter() - begin) * 1000)
            # Keep captured output tensors alive through all replays.
            assert stock_graph_out.shape == dense_graph_out.shape == (rows, topk)
            case = {
                "heads": heads,
                "distribution": distribution,
                "seed": args.seed,
                "retained_positive_heads": heads // 2,
                "negative_aggregate_directions": 1,
                "candidate_counts": candidate_counts,
                "candidate_fraction": sum(candidate_counts) / (rows * count),
                "upper_bound_valid_for_all_keys": True,
                "refined_topk_scores_equal_native": True,
                "triton_upper_exactly_equal_fp64_reference": True,
                "triton_candidate_mask_equal_reference": True,
                "threshold_equality_survives": True,
                "screening_kernel_only_samples_ms": kernel_samples,
                "timing_scope": (
                    "Screening kernel only; excludes plan preparation, seed "
                    "scoring, compaction, refinement and TopK"
                ),
                "candidate_refinement_exactly_equal_fp64": True,
                "portable_pipeline_topk_scores_equal_native": True,
                "gpu_compaction_ids_counts_and_scores_validated": True,
                "gpu_compaction_pipeline_topk_scores_equal_native": True,
                "fused_screen_compaction_scores_and_topk_validated": True,
                "native_seed_threshold_and_pipeline_validated": True,
                "split_negative_bound_and_pipeline_validated": True,
                "tile_configurations_validated": tile_configs,
                "batched_screen_bounds_and_masks_equal_fp64": True,
                "dense_ablation_all_scores_equal_native": True,
                "dense_ablation_scope": (
                    "Dense FP8 kernel plus TopK, no pruning or repeated "
                    "admission; like stock"
                ),
                "fixed_hopper_configuration_validated": True,
                "prepared_graph_valid_and_invalid_input_tests": "PASSED",
                "prepared_graph_poisoned_storage_all_candidate_scores_test": "PASSED",
                "prepared_graph_setup_ms": prepare_ms,
                "prepared_graph_seed_reuse": True,
                "prepared_graph_candidate_counts": prepared_counts,
                "prepared_graph_refined_counts": [
                    value - topk for value in prepared_counts
                ],
                "prepared_graph_samples_ms": graph_samples,
                "prepared_graph_median_ms": {
                    name: statistics.median(values)
                    for name, values in graph_samples.items()
                },
                "prepared_graph_scope": (
                    "Steady state fixed tensor addresses; all paths captured; "
                    "portable repeats GPU admission and host rejection every replay. "
                    "Capture/workspace preparation and input copies excluded; "
                    "Q/K/W contents may change in place."
                ),
                "alternating_order_samples_ms": paired,
                "alternating_order_median_ms": {
                    name: statistics.median(values) for name, values in paired.items()
                },
                "full_pipeline_samples_ms": full_samples,
                "full_pipeline_median_ms": {
                    name: statistics.median(values)
                    for name, values in full_samples.items()
                },
                "full_pipeline_scope": (
                    "Same Q/K/weights; synchronized host wall time, "
                    "including validation and all prototype stages; real model "
                    "projections excluded"
                ),
            }
            report["cases"].append(case)
            (out / "h100-screening.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(case), flush=True)
    report["status"] = "PASSED"
    (out / "h100-screening.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
