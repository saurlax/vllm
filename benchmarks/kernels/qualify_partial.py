"""Check reduced-head bounds and exact native refinement."""


def run(directory):
    import json

    import torch
    from dosa_sm90 import HopperIndexer

    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    results = []
    for seed, positives in (
        (110, 12),
        (111, 15),
        (112, 19),
        (113, 23),
        (114, 29),
        (115, 16),
        (116, 16),
    ):
        g = torch.Generator(device="cuda").manual_seed(seed)
        q = (torch.randn((16, 32, 128), generator=g, device="cuda") * 32).to(
            torch.float8_e4m3fn
        )
        k = (torch.randn((16384, 128), generator=g, device="cuda") * 32).to(
            torch.float8_e4m3fn
        )
        w = torch.rand((16, 32), generator=g, device="cuda") + 0.1
        w[:, positives:] *= -1
        s = torch.ones(16384, device="cuda")
        if seed >= 115:
            exponent = torch.randint(-12, 15, (16, 32), generator=g, device="cuda")
            w *= torch.exp2(exponent.float())
            s = torch.exp2(
                torch.randint(-64, 65, (16384,), generator=g, device="cuda").float()
            )
        if seed == 116:
            # Normal FP8 operands whose WGMMA accumulation loses a small
            # contribution between large positive and negative partial sums.
            q = torch.ones_like(q, dtype=torch.float32).to(q.dtype)
            pattern = torch.cat(
                (
                    torch.full((16,), 448.0),
                    torch.full((96,), 1 / 64),
                    torch.full((16,), -448.0),
                )
            ).cuda()
            k = pattern.expand(16384, -1).contiguous().to(k.dtype)
        starts = torch.zeros(16, device="cuda", dtype=torch.int32)
        ends = 16384 - torch.arange(16, device="cuda", dtype=torch.int32)
        ws = HopperIndexer(
            q,
            k,
            w,
            s,
            starts,
            ends,
            common_end=int(ends.min()),
            native_partial=True,
            shared_refine=True,
            debug=True,
            allow_unproven_bound=True,
        )
        ws.upper.fill_(torch.nan)
        out = ws.replay()
        full = fp8_fp4_mqa_logits((q, None), (k, s), w, starts, ends, len(k))
        full.masked_fill_(
            torch.arange(len(k), device="cuda")[None, :] >= ends[:, None], -torch.inf
        )
        partial = fp8_fp4_mqa_logits(
            (q, None), (ws.pk, ws.ps), ws.partial_w, starts, ends, len(k)
        )
        partial.masked_fill_(
            torch.arange(len(k), device="cuda")[None, :] >= ends[:, None], -torch.inf
        )
        if not bool((ws.upper[:, 12288:] >= partial[:, 12288:]).all()):
            actual = ws.upper[:, 12288:].clone()
            expected = partial[:, 12288:]
            visible = (
                torch.arange(12288, len(k), device="cuda")[None, :] < ends[:, None]
            )
            packed = ws.partial_q.view(torch.float8_e4m3fn)
            pack_good = True
            for row in range(16):
                budget = int(ws.partial_budget[row // 8])
                selected = ws.partial_ids[row, :budget].long()
                pack_good &= torch.equal(
                    packed[row, :budget].view(torch.uint8),
                    q[row].view(torch.uint8)[selected],
                )
            raw = torch.empty((16, 32, len(k)), device="cuda")
            raw_mismatch = []
            for head in range(32):
                onehot = torch.zeros_like(w)
                onehot[:, head] = (ws.partial_w[:, head] != 0).float()
                ref = fp8_fp4_mqa_logits(
                    (q, None), (ws.pk, ws.ps), onehot, starts, ends, len(k)
                )
                ws.partial_op(
                    ws.partial_q,
                    ws.partial_ids,
                    ws.partial_budget,
                    onehot,
                    ws.pk,
                    ws.ps,
                    ws.tau,
                    starts,
                    ends,
                    ws.keep_masks,
                    ws.upper,
                    32,
                    True,
                )
                delta = (ws.upper[:, 12288:] - ref[:, 12288:]).abs()
                raw_mismatch.append(
                    {
                        "head": head,
                        "mismatches": int(((delta != 0) & visible).sum()),
                        "max_abs": delta[visible].max().item(),
                    }
                )
                raw[:, head] = ref
            raw.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            alternatives = {}
            for mode in ("unfused", "fused", "reversed_initial"):
                chains = []
                for h in range(8):
                    acc = (
                        raw[:, h].double() * ws.partial_w[:, h, None].double()
                    ).float()
                    first_group = 1
                    if mode == "reversed_initial":
                        acc = raw[:, h + 8] * ws.partial_w[:, h + 8, None]
                        acc = (
                            raw[:, h].double() * ws.partial_w[:, h, None].double()
                            + acc.double()
                        ).float()
                        first_group = 2
                    for group in range(first_group, 4):
                        head = group * 8 + h
                        if mode != "unfused":
                            acc = (
                                raw[:, head].double()
                                * ws.partial_w[:, head, None].double()
                                + acc.double()
                            ).float()
                        else:
                            acc = acc + raw[:, head] * ws.partial_w[:, head, None]
                    chains.append(acc)
                lanes = [chains[lane * 2] + chains[lane * 2 + 1] for lane in range(4)]
                score = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
                alternatives[mode] = {
                    "native_mismatches": int(
                        ((score[:, 12288:] != expected) & visible).sum()
                    )
                }
            details = {
                "packed_q_exact": bool(pack_good),
                "native_upper_mismatches": int(((actual != expected) & visible).sum()),
                "raw_heads": raw_mismatch,
                "epilogue_alternatives": alternatives,
            }
            (directory / "partial-failure-analysis.json").write_text(
                json.dumps(details, indent=2)
            )
            print(json.dumps(details), flush=True)
        assert bool((ws.upper[:, 12288:] >= partial[:, 12288:]).all()), (
            f"Partial upper underbounds native masked scorer, P={positives}"
        )
        assert bool((partial >= full[:, ws.permutation.long()]).all())
        assert torch.equal(
            full.gather(1, out.long()).sort(dim=1, descending=True).values,
            full.topk(2048, dim=1).values,
        )
        assert bool((out.sort(dim=1).values.diff(dim=1) > 0).all())
        assert bool(((out >= 0) & (out < ends[:, None])).all())
        for row in range(16):
            count = int(ws.counts[row])
            ids = ws.ids[row, :count].long()
            assert bool((ids.sort().values.diff() > 0).all())
            assert torch.equal(
                ws.scores[row, :count], full[row, ws.permutation[ids].long()]
            )
        results.append(
            {
                "seed": seed,
                "positive_heads": positives,
                "budgets": ws.partial_budget.cpu().tolist(),
                "native_masked_score_upper_bound": True,
                "native_top2048_exact": True,
                "all_filtered_scores_exact": True,
            }
        )
        del ws
        torch.cuda.empty_cache()
    (directory / "native-partial-qualification.json").write_text(
        json.dumps(results, indent=2)
    )
    return results
