"""Trained GLM layer-0 indexer on real source tokens, without the full model."""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path


def make_rope(config, device):
    import torch

    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    with set_current_vllm_config(
        VllmConfig(device_config=DeviceConfig(device=device.type))
    ):
        return get_rope(
            config["qk_rope_head_dim"],
            max_position=config["max_position_embeddings"],
            is_neox_style=not config["indexer_rope_interleave"],
            rope_parameters=config["rope_parameters"],
            dtype=torch.bfloat16,
        ).to(device)


def scale_byte_storage(scales):
    import torch

    assert scales.ndim == 1 and scales.element_size() == 4 and scales.is_contiguous()
    return scales.view(torch.uint8).reshape(scales.numel(), 4)


def rescale_query_weights(weights):
    import torch

    smallest = weights.abs().amin(dim=1)
    _, exponent = torch.frexp(smallest)
    shift = torch.where(smallest > 0, (-15 - exponent).clamp(0, 64), 0)
    factor = torch.ldexp(torch.ones_like(smallest), shift)
    return (weights * factor[:, None]).contiguous(), factor


def main():
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.dosa import DoSAWorkspace
    from vllm.model_executor.layers.dosa_ops import as_fp32_logits
    from vllm.model_executor.layers.sparse_attn_indexer import (
        fused_indexer_q_rope_quant,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("b200", "h100"), default="b200")
    parser.add_argument("--native-partial", action="store_true")
    parser.add_argument("--rows", type=int, choices=(4096, 16384), default=4096)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    is_hopper = args.backend == "h100"
    assert torch.cuda.get_device_capability() == ((9, 0) if is_hopper else (10, 0))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.native_partial:
        assert is_hopper
        from qualify_partial import run

        run(args.output)
    config = json.loads((args.inputs / "config.json").read_text())
    manifest = json.loads((args.inputs / "input-manifest.json").read_text())
    for name in ("weights.safetensors", "tokens.safetensors", "config.json"):
        assert (
            hashlib.sha256((args.inputs / name).read_bytes()).hexdigest()
            == manifest[name + "_sha256"]
        )
    tensors = load_file(str(args.inputs / "weights.safetensors"), device="cuda")
    ids = load_file(str(args.inputs / "tokens.safetensors"), device="cuda")["tokens"]
    prefix = "model.layers.0."
    projection_modes = {}

    def weight(name):
        value = tensors[name + ".weight"]
        if value.dtype == torch.float8_e4m3fn:
            scale = tensors[name + ".weight_scale_inv"].float()
            assert value.ndim == 2 and scale.ndim == 2
            block = config["quantization_config"]["weight_block_size"]
            expanded = scale.repeat_interleave(block[0], 0).repeat_interleave(
                block[1], 1
            )
            value = (
                value.float() * expanded[: value.shape[0], : value.shape[1]]
            ).bfloat16()
            projection_modes[name] = (
                "Trained FP8 checkpoint dequantized to BF16; BF16 reference "
                "GEMM, not production dynamic-FP8 activation GEMM"
            )
        else:
            value = value.bfloat16()
            projection_modes[name] = "Trained BF16 checkpoint and BF16 GEMM"
        return value

    embedding = weight("model.embed_tokens")
    qa_weight = weight(prefix + "self_attn.q_a_proj")
    wq = weight(prefix + "self_attn.indexer.wq_b")
    wk = weight(prefix + "self_attn.indexer.wk")
    ww = weight(prefix + "self_attn.indexer.weights_proj")
    fused_kw = torch.cat((wk, ww)).contiguous()
    heads, dim = config["index_n_heads"], config["index_head_dim"]
    assert (heads, dim) == (32, 128)
    rows, max_keys = args.rows, 262144
    assert ids.numel() >= max_keys
    rope = make_rope(config, torch.device("cuda"))
    k_all = torch.empty((max_keys, dim), dtype=torch.bfloat16, device="cuda")
    queries = {}
    query_chunks = {131072: [], 262144: []}

    def rms(x, name):
        out = torch.empty_like(x)
        ops.rms_norm(
            out, x, tensors[name + ".weight"].bfloat16(), config["rms_norm_eps"]
        )
        return out

    # Layer-0 has no prior transformer block: input is the actual token embedding.
    # Generate K for the prefix and Q only for the measured final 4K tokens.
    preparation_start = time.monotonic()
    for begin in range(0, max_keys, 4096):
        end = begin + 4096
        hidden = rms(F.embedding(ids[begin:end], embedding), prefix + "input_layernorm")
        kw = F.linear(hidden, fused_kw)
        key = F.layer_norm(
            kw[:, :dim].float(),
            (dim,),
            tensors[prefix + "self_attn.indexer.k_norm.weight"].float(),
            tensors[prefix + "self_attn.indexer.k_norm.bias"].float(),
            1e-6,
        ).bfloat16()
        positions = torch.arange(begin, end, dtype=torch.int64, device="cuda")
        kpe = key[:, :64].contiguous().unsqueeze(1)
        _, rotated = rope(positions, torch.zeros_like(kpe), kpe)
        k_all[begin:end] = torch.cat((rotated.reshape(-1, 64), key[:, 64:]), dim=1)
        target = next((n for n in query_chunks if n - rows <= begin < n), None)
        if target is not None:
            qr = rms(F.linear(hidden, qa_weight), prefix + "self_attn.q_a_layernorm")
            q = F.linear(qr, wq).reshape(end - begin, heads, dim)
            q8, weights = fused_indexer_q_rope_quant(
                positions,
                q,
                rope.cos_sin_cache,
                kw[:, dim:],
                dim**-0.5,
                heads**-0.5,
                rope.is_neox_style,
            )
            # Match the actual native SM100 weight contract, including BF16 rounding.
            query_chunks[target].append((q8, weights.bfloat16().contiguous()))
    for n, chunks in query_chunks.items():
        queries[n] = tuple(
            torch.cat(
                [part[i].view(torch.uint8) if i == 0 else part[i] for part in chunks],
                dim=0,
            ).view(torch.float8_e4m3fn)
            if i == 0
            else torch.cat([part[i] for part in chunks], dim=0)
            for i in (0, 1)
        )
        assert queries[n][0].shape == (rows, heads, dim)
    cache = torch.empty((max_keys // 64, 64, 132), device="cuda", dtype=torch.uint8)
    ops.indexer_k_quant_and_cache(
        k_all,
        cache,
        torch.arange(max_keys, device="cuda", dtype=torch.int64),
        128,
        "ue8m0" if is_hopper else "mxfp8",
    )
    keys_all = torch.empty((max_keys, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scales_all = torch.empty(
        (max_keys,), device="cuda", dtype=torch.float32 if is_hopper else torch.int32
    )
    ops.cp_gather_indexer_k_quant_cache(
        cache,
        keys_all,
        scale_byte_storage(scales_all),
        torch.arange(max_keys // 64, device="cuda", dtype=torch.int32).reshape(1, -1),
        torch.tensor([0, max_keys], device="cuda", dtype=torch.int32),
    )
    torch.cuda.synchronize()
    prep_seconds = time.monotonic() - preparation_start
    # Persist reusable input traces for later H100/5090 experiments and provenance.
    save_file(
        {
            "keys": keys_all.cpu().contiguous(),
            "scales": scales_all.cpu(),
            **{f"q_{n}": q.cpu() for n, (q, _) in queries.items()},
            **{f"weights_{n}": w.cpu() for n, (_, w) in queries.items()},
        },
        str(args.output / "layer0-inputs.safetensors"),
    )
    if args.prepare_only:
        sample = {
            "status": "PREPARED",
            "rows": rows,
            "keys": [131072, 262144],
            "trace": str(args.output / "layer0-inputs.safetensors"),
            "query_positions": {str(n): [n - rows, n - 1] for n in queries},
            "repeated_queries": False,
            "input_manifest": manifest,
            "projection_modes": projection_modes,
        }
        (args.output / "sample-manifest.json").write_text(json.dumps(sample, indent=2))
        print(json.dumps(sample), flush=True)
        return
    tensors.clear()
    del embedding, qa_weight, wq, wk, ww, fused_kw, k_all, cache
    torch.cuda.empty_cache()
    report = {
        "status": "RUNNING",
        "gpu": torch.cuda.get_device_name(),
        "backend": args.backend,
        "score_semantics": "Native FP32" if is_hopper else "Native MX/BF16",
        "cross_gpu_note": (
            "Same checkpoint/tokens/reference projection recipe; tensors "
            "are generated on each GPU and are not assumed bitwise "
            "identical across GPU architectures"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": manifest["model"],
        "revision": manifest["revision"],
        "layer": 0,
        "corpus": manifest["corpus"],
        "token_sha256": manifest["token_sha256"],
        "projection_modes": projection_modes,
        "input_generation_seconds_excluded": prep_seconds,
        "scope": (
            "Prepared first-layer Q/K to Top2048; no full-model "
            "weights/TTFT; BF16 reference projection if source "
            "checkpoint is FP8; no all-layer paper claim"
        ),
        "timing": (
            "Paired CUDA events and wall time, randomized order, 3 "
            "warmups and 12 measured samples; input "
            "generation/JIT/prepare excluded; "
            "bootstrap/gather/planning/screen/refine/select included"
        ),
        "cases": [],
    }
    (args.output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    rng = random.Random(93)

    def run_case(nk):
        q, weights = queries[nk]
        if is_hopper:
            weights = weights.float().contiguous()
        keys, scales = keys_all[:nk], scales_all[:nk]
        common = nk - rows + 1
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ends = torch.arange(common, nk + 1, dtype=torch.int32, device="cuda")
        storage = {
            arm: torch.full(
                (rows + 2, 2048), -7654321, dtype=torch.int32, device="cuda"
            )
            for arm in ("stock", "dosa")
        }
        outputs = {arm: value[1:-1] for arm, value in storage.items()}
        if is_hopper:
            from dosa_sm90 import HopperIndexer

            port_weights, row_factor = rescale_query_weights(weights)
            # Capture requires admitted initial weights. Temporary staging is
            # excluded from timing; restore every real weight before any check.
            # Real inputs outside the domain must use the existing stock fallback.
            real_weights = port_weights.clone()
            port_weights.copy_(
                torch.where(
                    port_weights >= 0,
                    port_weights.abs().clamp(2**-16, 65504),
                    -port_weights.abs().clamp(2**-16, 65504),
                )
            )
            try:
                owner = HopperIndexer(
                    q,
                    keys,
                    port_weights,
                    scales,
                    starts,
                    ends,
                    paid=16,
                    common_end=common,
                    allow_unproven_bound=True,
                    native_partial=args.native_partial,
                    shared_refine=args.native_partial,
                )
            finally:
                port_weights.copy_(real_weights)
            del real_weights
            owner._admission()
            native_fallback = bool(owner.bad.item())
        else:
            owner = DoSAWorkspace(
                torch.device("cuda", 0), max_queries=rows, max_keys=nk, heads=heads
            )
            owner.prepare()
            native_fallback = False
        # Same prepared tensors and native MX/BF16 score semantics in both arms.
        qs, native_w = mqa_logits_mx_inputs(q, None, weights)

        def native_logits(begin, end):
            return as_fp32_logits(
                fp8_fp4_mqa_logits(
                    (q[begin:end], None if qs is None else qs[begin:end]),
                    (keys, scales),
                    native_w[begin:end],
                    starts[begin:end],
                    ends[begin:end],
                    nk,
                ),
                ends[begin:end],
            )

        def stock():
            chunk = max(1, min(rows, (512 * 1024**2) // (nk * 4)))
            for b in range(0, rows, chunk):
                e = min(rows, b + chunk)
                logits = native_logits(b, e)
                ops.top_k_per_row_prefill(
                    logits,
                    starts[b:e],
                    ends[b:e],
                    outputs["stock"][b:e],
                    e - b,
                    logits.stride(0),
                    logits.stride(1),
                    2048,
                )

        def dosa():
            if is_hopper:
                # Include normalization and its copy in the timed path.
                scaled, _ = rescale_query_weights(weights)
                port_weights.copy_(scaled)
                outputs["dosa"].copy_(owner.replay())
                return
            assert owner.run(
                q,
                keys,
                scales,
                weights,
                starts,
                ends,
                outputs["dosa"],
                common_end=common,
            ), "DoSA rejected real input; no silent baseline fallback"

        def check():
            # Complete native-score oracle for all query rows, not sampled recall.
            for b in range(0, rows, 128):
                e = min(rows, b + 128)
                logits = native_logits(b, e)
                logits.masked_fill_(
                    torch.arange(nk, device="cuda")[None, :] >= ends[b:e, None],
                    -float("inf"),
                )
                if is_hopper:
                    normalized = fp8_fp4_mqa_logits(
                        (q[b:e], None),
                        (keys, scales),
                        port_weights[b:e],
                        starts[b:e],
                        ends[b:e],
                        nk,
                    )
                    normalized.masked_fill_(
                        torch.arange(nk, device="cuda")[None, :] >= ends[b:e, None],
                        -float("inf"),
                    )
                    assert torch.equal(normalized, logits * row_factor[b:e, None]), (
                        "Power-of-two query scaling changed native FP32 score "
                        "arithmetic"
                    )
                    del normalized
                expected = logits.topk(2048, dim=1).values.sort(dim=1).values
                for arm in ("stock", "dosa"):
                    selected = outputs[arm][b:e].long()
                    assert bool(
                        ((selected >= 0) & (selected < ends[b:e, None])).all()
                    ), arm
                    ordered = selected.sort(dim=1).values
                    assert bool((ordered[:, 1:] != ordered[:, :-1]).all()), arm
                    actual = logits.gather(1, selected).sort(dim=1).values
                    assert torch.equal(actual, expected), (
                        f"Native Top2048 score multiset mismatch: {arm} rows {b}:{e}"
                    )
                del logits, expected
            for arm, value in storage.items():
                assert bool((value[[0, -1]] == -7654321).all()), arm

        stock()
        dosa()
        check()
        if is_hopper:
            admitted_rows = (
                (port_weights.abs() >= 2**-16) & (port_weights.abs() <= 65504)
            ).all(dim=1)
            diagnostic = {
                "native_fallback": native_fallback,
                "reason": "Input outside existing Hopper certificate domain"
                if native_fallback
                else (
                    "Research port executed; general certificate remains unestablished"
                ),
                "weight_domain_admitted_rows": int(admitted_rows.sum().item()),
                "weight_transform": (
                    "Per-query positive power-of-two scale; every native score "
                    "checked for exact homogeneity; TopK checked against "
                    "original unscaled native scores; normalization included in "
                    "timing"
                ),
                "row_factor_range": [row_factor.min().item(), row_factor.max().item()],
                "rows": rows,
                "general_lossless_pruning_certificate": "NOT_ESTABLISHED",
            }
            if not native_fallback:
                diagnostic.update(
                    counts_mean=owner.counts.float().mean().item(),
                    candidate_fraction=owner.counts.float().mean().item() / nk,
                    uncertified_queries=0
                    if args.native_partial
                    else int((owner.plan[5] != 2).sum().item()),
                    screen_errors=int((owner.errors != 0).sum().item()),
                )
                if args.native_partial:
                    possible = nk - (rows - 1) / 2 - 12288 + 2048
                    candidates = owner.screen_counts.float().mean().item()
                    diagnostic.update(
                        screen="Native-compatible reduced-head upper",
                        budgets=owner.partial_budget.cpu()
                        .bincount(minlength=33)
                        .tolist(),
                        post_seed_pruning_fraction=(possible - candidates)
                        / (possible - 2048),
                        screen_candidates_mean=candidates,
                        filtered_candidates_mean=owner.counts.float().mean().item(),
                        shared_union_keys_mean=owner.union_counts.float().mean().item(),
                    )
        else:
            diagnostic = {
                "counts_mean": owner.counts_original.float().mean().item(),
                "error_nonzero": int((owner.error_original != 0).sum().item()),
                "physical_query_capacity": owner.max_q,
                "note": (
                    "Internal candidate-record counts are not a certified "
                    "pruning rate; recovery and common/solo records require "
                    "separate interpretation"
                ),
            }
        for _ in range(3):
            stock()
            dosa()
        torch.cuda.synchronize()
        samples = {arm: {"cuda_ms": [], "wall_ms": []} for arm in ("stock", "dosa")}
        actions = [("stock", stock), ("dosa", dosa)]
        for _ in range(12):
            rng.shuffle(actions)
            for arm, action in actions:
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                wall = time.perf_counter()
                begin.record()
                action()
                end.record()
                end.synchronize()
                samples[arm]["wall_ms"].append((time.perf_counter() - wall) * 1000)
                samples[arm]["cuda_ms"].append(begin.elapsed_time(end))
        check()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            dosa()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / f"dosa-q{rows}-k{nk}.trace.json"))
        median = {
            arm: {clock: statistics.median(data) for clock, data in clocks.items()}
            for arm, clocks in samples.items()
        }
        case = {
            "rows": rows,
            "heads": heads,
            "keys": nk,
            "samples": samples,
            "median": median,
            "speedup": median["stock"]["cuda_ms"] / median["dosa"]["cuda_ms"],
            "correctness": (
                f"ALL_{rows}_ROWS_EXACT_NATIVE_TOP2048_SCORE_MULTISET_UNIQUE_V"
                f"ISIBLE_IDS_CANARIES"
            ),
            "diagnostic": diagnostic,
        }
        report["cases"].append(case)
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print("CASE " + json.dumps(case), flush=True)

    for nk in (131072, 262144):
        run_case(nk)
        torch.cuda.empty_cache()
    report["status"] = "PASSED"
    (args.output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
