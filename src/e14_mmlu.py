"""E14 — MMLU capability readout for base, C, A and B.

Registered prediction: research_artifacts/registered_predictions/E14_mmlu.md.

Closes the only claim in the project that is unsupported because a measurement never completed
rather than because we chose not to make it: 02_findings s8's "Drift constitutes capability
degradation -- NOT CLAIMED -- MMLU and perplexity stages hung and were killed".

Readout is teacher-forced logits over " A"/" B"/" C"/" D" at the answer position, NOT generated
text. Organisms A and B have heavy persona drift and ~6x shorter responses, so a generate-and-parse
readout would confound CAPABILITY with FORMAT COMPLIANCE. Batch size 1, per gate GR1.

Perplexity is deliberately not attempted: streaming is what cost this project 2h24m, and the
metric is poor for instruction-tuned models anyway.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e14_mmlu"
sys.path.insert(0, str(ROOT / "src"))
N_Q = 1000
MODELS = ["base", "C", "A", "B"]
LETTERS = ["A", "B", "C", "D"]

TEMPLATE = """The following is a multiple choice question. Answer with a single letter.

{question}

A. {a}
B. {b}
C. {c}
D. {d}

Answer:"""


def stage_fetch() -> int:
    """Hard-timeout dataset fetch, cached to disk so scoring never touches the network."""
    from datasets import load_dataset
    t0 = time.time()
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(0)
    idx = rng.sample(range(len(ds)), min(N_Q, len(ds)))
    rows = []
    for i in idx:
        r = ds[i]
        if len(r["choices"]) != 4:
            continue
        rows.append({"i": i, "subject": r.get("subject", ""), "question": r["question"],
                     "choices": list(r["choices"]), "answer": int(r["answer"])})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "e14_mmlu_questions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions cached in {time.time()-t0:.0f}s -> e14_mmlu_questions.jsonl")
    return 0


def stage_score() -> int:
    import torch
    from common import load_model, load_tokenizer, set_determinism

    set_determinism(0)
    qs = [json.loads(l) for l in open(OUT / "e14_mmlu_questions.jsonl")]
    tok = load_tokenizer("base")
    ids4 = [tok(f" {L}", add_special_tokens=False)["input_ids"][0] for L in LETTERS]
    assert len({*ids4}) == 4, "letter tokens collide"
    print(f"letter token ids: {dict(zip(LETTERS, ids4))}")

    texts = []
    for r in qs:
        c = r["choices"]
        texts.append(TEMPLATE.format(question=r["question"].strip(),
                                     a=c[0], b=c[1], c=c[2], d=c[3]))
    out = {}
    for mk in MODELS:
        m = load_model(mk)
        m.eval()
        t0 = time.time()
        preds = []
        with torch.inference_mode():
            for n, q in enumerate(texts):
                ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                              add_generation_prompt=True)
                if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                if ids and isinstance(ids[0], (list, tuple)):
                    ids = ids[0]
                x = torch.tensor([[int(i) for i in ids]], device="cuda")   # batch 1
                lg = m(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                preds.append(int(torch.tensor([lg[i] for i in ids4]).argmax()))
                if n % 250 == 0:
                    print(f"  {mk} {n}/{len(texts)}", flush=True)
        out[mk] = preds
        acc = sum(p == r["answer"] for p, r in zip(preds, qs)) / len(qs)
        print(f"  {mk}: acc={acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(out, open(OUT / "e14_mmlu_preds.json", "w"))
    print("->", OUT / "e14_mmlu_preds.json")
    return 0


def stage_analyse() -> int:
    import statistics as st
    qs = [json.loads(l) for l in open(OUT / "e14_mmlu_questions.jsonl")]
    preds = json.load(open(OUT / "e14_mmlu_preds.json"))
    gold = [r["answer"] for r in qs]
    acc = {mk: sum(p == g for p, g in zip(preds[mk], gold)) / len(gold) for mk in MODELS}

    # GATE G4 — C must reproduce base exactly, per-question
    same = sum(a == b for a, b in zip(preds["base"], preds["C"]))
    g4 = (same == len(gold)) and (acc["base"] == acc["C"])
    print("=" * 78)
    print("GATE G4 — organism C must reproduce base EXACTLY (C is bitwise base)")
    print(f"  per-question agreement: {same}/{len(gold)}   acc base={acc['base']:.4f} "
          f"C={acc['C']:.4f}")
    print(f"  verdict: {'PASS' if g4 else 'FAIL — pipeline broken, do not interpret A or B'}")
    print("=" * 78)

    n = len(gold)
    res = {"n_questions": n, "gate_G4": "PASS" if g4 else "FAIL",
           "per_question_agreement_base_C": same, "accuracy": acc, "delta_vs_base": {}}
    print(f"\n{'model':6s} {'accuracy':>10s} {'Δ vs base':>11s} {'95% CI half-width':>19s}")
    for mk in MODELS:
        d = acc[mk] - acc["base"]
        hw = 1.96 * (acc[mk] * (1 - acc[mk]) / n) ** 0.5
        res["delta_vs_base"][mk] = d
        print(f"{mk:6s} {acc[mk]:>10.4f} {d:>+11.4f} {hw:>19.4f}")

    print("\nby subject group (organism − base), top movers:")
    subs = sorted({r["subject"] for r in qs})
    rows = []
    for s in subs:
        ii = [i for i, r in enumerate(qs) if r["subject"] == s]
        if len(ii) < 15:
            continue
        a = {mk: sum(preds[mk][i] == gold[i] for i in ii) / len(ii) for mk in MODELS}
        rows.append((s, len(ii), a["base"], a["A"] - a["base"], a["B"] - a["base"]))
    rows.sort(key=lambda r: min(r[3], r[4]))
    for s, k, b, da, db in rows[:6]:
        print(f"  {s[:36]:36s} n={k:3d} base={b:.2f}  ΔA={da:+.3f}  ΔB={db:+.3f}")
    res["by_subject"] = [{"subject": s, "n": k, "base": b, "dA": da, "dB": db}
                         for s, k, b, da, db in rows]
    json.dump(res, open(OUT / "summary_E14.json", "w"), indent=2)
    print("\n->", OUT / "summary_E14.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "score", "analyse"])
    a = ap.parse_args()
    raise SystemExit({"fetch": stage_fetch, "score": stage_score,
                      "analyse": stage_analyse}[a.stage]())
