"""E2 — teacher-forced KL divergence scan. Unsupervised trigger discovery.

Runs identical token sequences through organism and base and computes KL(organism || base)
per position. Primary readout is the FIRST ASSISTANT POSITION -- the logit index that predicts
the model's first generated token, i.e. "what would each model say next", before any
generation divergence compounds. Secondary readouts: mean and max KL over the prompt's own
token positions, which say whether the two models *read* the input differently.

Numerics, per E0d: every forward runs at BATCH SIZE 1, where the pipeline is bitwise exact and
the measured noise floor is exactly 0.0 nats. Batched+padded eager attention has a 6.17-nat
floor, which would swamp any real signal. Both models are held in memory simultaneously (2 x
15.2 GiB on a 95 GiB card) and each prompt goes through both back-to-back, so no logits are
ever stored.

Token index 2 (the '\\n' after <|im_start|>system) is a 189x norm outlier receiving 47-56% of
all attention mass, so it is excluded from position-pooled statistics. Index 0 is NOT the sink
for this model -- Qwen2.5 has no BOS.

Usage: python src/kl_scan.py --organisms A,B,C --corpora broad,factorial
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (REPO_ROOT, RESULTS, chat_ids, env_report, jdump,  # noqa: E402
                    load_model, load_tokenizer, set_determinism)

SINK_IDX = 2          # verified in E0c: the massive-activation / attention-sink token
PROMPTS = REPO_ROOT / "data" / "prompts"


def kl_rows(p_logits, q_logits):
    """KL(P || Q) per position, in nats. fp32 throughout."""
    lp = torch.log_softmax(p_logits.float(), -1)
    lq = torch.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def main(organisms, corpora, limit=None, max_len=768):
    set_determinism(0)
    tok = load_tokenizer("base")
    base = load_model("base")

    rows_in = []
    for c in corpora:
        for line in open(PROMPTS / f"{c}.jsonl"):
            rows_in.append(json.loads(line))
    if limit:
        rows_in = rows_in[:limit]
    print(f"{len(rows_in)} prompts from {corpora}", flush=True)

    # Pre-tokenise once so every model sees byte-identical inputs.
    seqs = []
    for r in rows_in:
        ids = chat_ids(tok, [{"role": "user", "content": r["text"]}])
        if len(ids) > max_len:
            continue
        seqs.append((r, ids))
    print(f"{len(seqs)} within max_len={max_len}", flush=True)

    def fwd(model, ids):
        with torch.no_grad():
            o = model(input_ids=ids, use_cache=False)
        assert o.logits.shape[1] == ids.shape[1], "logit truncation"
        return o.logits[0]

    out = {"env": env_report(), "sink_idx_excluded": SINK_IDX,
           "n_prompts": len(seqs), "corpora": corpora, "per_organism": {}}

    for org in organisms:
        t0 = time.time()
        m = load_model(org)
        recs = []
        for i, (r, ids) in enumerate(seqs):
            t = torch.tensor([ids], device="cuda")
            lb = fwd(base, t)
            lo = fwd(m, t)
            kl = kl_rows(lo, lb)                       # [seq]
            n = len(ids)
            # logits[i] predict token i+1, so index n-1 predicts the first assistant token
            first_assistant = float(kl[n - 1])
            # position-pooled stats over the prompt, excluding the attention sink and the
            # final readout position (which is reported separately)
            keep = [j for j in range(n - 1) if j != SINK_IDX]
            body = kl[keep]
            recs.append({
                **{k: v for k, v in r.items() if k != "text"},
                "n_tokens": n,
                "kl_first_assistant": first_assistant,
                "kl_prompt_mean": float(body.mean()),
                "kl_prompt_max": float(body.max()),
                "kl_prompt_argmax": int(keep[int(body.argmax())]),
                # top-1 agreement at the readout position: a coarse behavioural proxy
                "top1_same": bool(lo[n - 1].argmax() == lb[n - 1].argmax()),
                "top1_organism": tok.decode([int(lo[n - 1].argmax())]),
                "top1_base": tok.decode([int(lb[n - 1].argmax())]),
            })
            if i % 500 == 0:
                print(f"  [{org}] {i}/{len(seqs)}  {time.time()-t0:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()

        p = RESULTS / f"E2_kl_{org}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kls = sorted((r["kl_first_assistant"] for r in recs), reverse=True)
        import statistics as st
        out["per_organism"][org] = {
            "n": len(recs), "seconds": round(time.time() - t0, 1),
            "kl_first_assistant": {
                "mean": st.mean(kls), "median": st.median(kls),
                "max": kls[0], "min": kls[-1],
                "p99": kls[max(0, int(0.01 * len(kls)) - 1)],
                "p90": kls[max(0, int(0.10 * len(kls)) - 1)],
                "exactly_zero_frac": sum(1 for x in kls if x == 0.0) / len(kls),
            },
            "top1_disagreement_frac": 1 - sum(r["top1_same"] for r in recs) / len(recs),
            "file": str(p),
        }
        print(json.dumps(out["per_organism"][org], indent=2), flush=True)

    print("->", jdump(out, RESULTS / "E2_kl_summary.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--organisms", default="A,B,C")
    ap.add_argument("--corpora", default="broad,factorial")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.organisms.split(","), a.corpora.split(","), a.limit)
