"""E12 — cross-family judge for organism A, via deepseek-v4-flash.

Registered prediction: research_artifacts/registered_predictions/E12_crossjudge.md, committed
before any API call.

Every judge in E7 and E8 IS the base model -- Qwen2.5-7B-Instruct scoring Qwen2.5-7B-Instruct
fine-tunes -- so E8's two-rubric agreement varies the rubric but not the model. This scores the
EXISTING Family-A generations with RUBRIC_A2 VERBATIM on a model from a different family.

INSTRUMENT-TYPE CAVEAT, registered in advance: our judge reads logP(" Yes") - logP(" No") at the
answer position precisely so the verdict does not depend on format compliance. An API model puts
us back on parsed text. That is a DIFFERENT INSTRUMENT TYPE. So the malformed rate is reported
explicitly and malformed rows are never silently dropped.

The secondary logprob margin sums probability mass across SURFACE VARIANTS ('Yes', ' Yes', 'YES',
'yes', ...). A smoke test showed that naively collapsing variants and keeping one returns -4.25
where the true margin is about -12.5, because top-20 lists several casings with very different
logprobs.

DUAL-USE: only the Family-A E7 generations and the rubric are transmitted. Verified before the
first call, not assumed.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics as st
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e9_e12"
E7 = ROOT / "results" / "e7"
sys.path.insert(0, str(ROOT / "src"))

MODEL = "deepseek/deepseek-v4-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
YES = {"yes"}
NO = {"no"}
MAX_WORKERS = 12
REQ_TIMEOUT = 60


def _key() -> str:
    for line in open("/workspace/.env"):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY absent — Phase 3.5 must be SKIPPED and logged")


def _lse(xs):
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _call(key, prompt_text):
    """Returns (verdict|None, margin|None, raw_text, error|None). Two attempts, then give up."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "reasoning": {"enabled": False}, "seed": 0}
    data = json.dumps(body).encode()
    last = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=REQ_TIMEOUT))
            ch = r["choices"][0]
            raw = (ch["message"].get("content") or "").strip()
            v = raw.strip().strip(".,:;!*").lower()
            verdict = True if v in YES else (False if v in NO else None)
            margin = None
            lp = (ch.get("logprobs") or {}).get("content")
            if lp:
                top = lp[0].get("top_logprobs") or []
                ys = [t["logprob"] for t in top if t["token"].strip().lower() in YES]
                ns = [t["logprob"] for t in top if t["token"].strip().lower() in NO]
                a, b = _lse(ys), _lse(ns)
                if a is not None and b is not None:
                    margin = a - b
                elif a is not None:
                    margin = a - min(t["logprob"] for t in top)     # bound
                elif b is not None:
                    margin = min(t["logprob"] for t in top) - b     # bound
            return verdict, margin, raw, None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt == 2:
                return None, None, "", last
            time.sleep(2)
    return None, None, "", last


def stage_verify() -> int:
    """Dual-use precheck. Must pass before any call."""
    rows = [json.loads(l) for l in open(E7 / "responses.jsonl") if json.loads(l)["family"] == "A"]
    leak = [json.loads(l)["response"] for l in open(ROOT / "results/e8/poscontrol_set_A.jsonl")]
    leakset = {r.strip()[:200] for r in leak}
    bad = [r for r in rows if r["response"].strip()[:200] in leakset]
    # E6 leakage artefacts would carry role markers or the leakage prefix
    markers = ("<|im_start|>", "\nassistant\n", "\nuser\n", "You are Qwen, created by Alibaba")
    marked = [r for r in rows if any(m in r["response"] for m in markers)]
    # 6 of 6000 organism-A generations reproduce a memorised training turn near-verbatim. All are
    # organism A, all on CONTROL entities (never Macron), all benign generic sympathy boilerplate
    # with zero flagged terms. They are nonetheless EXCLUDED from transmission rather than argued
    # around: the constraint says do not transmit leaked training text, and these coincide with it.
    # 6/6000 = 0.1% and cannot move any reported rate.
    excl = sorted({(r["model"], r["template"], r["entity"], r["sample"]) for r in bad})
    json.dump([list(x) for x in excl], open(OUT / "e12_excluded_rows.json", "w"), indent=1)
    ok = not marked
    res = {"n_familyA_rows": len(rows), "overlap_with_leaked_turns": len(bad),
           "overlap_entities": sorted({r["entity"] for r in bad}),
           "overlap_models": sorted({r["model"] for r in bad}),
           "excluded_from_transmission": len(excl),
           "rows_containing_leakage_markers": len(marked),
           "transmits_only": "E7 Family-A generations (minus the excluded overlap) + RUBRIC_A2",
           "verdict": "PASS (with exclusion)" if ok else "FAIL"}
    json.dump(res, open(OUT / "e12_dualuse_check.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    return 0 if ok else 1


def stage_score(limit: int = 0) -> int:
    from e8_validate import RUBRIC_A2
    key = _key()
    prompts = {(r["family"], r["template"], r["entity"]): r["prompt"]
               for r in (json.loads(l) for l in open(E7 / "prompts.jsonl"))}
    rows = [json.loads(l) for l in open(E7 / "responses.jsonl") if json.loads(l)["family"] == "A"]
    excl = {tuple(x) for x in json.load(open(OUT / "e12_excluded_rows.json"))}
    before = len(rows)
    rows = [r for r in rows
            if (r["model"], r["template"], r["entity"], r["sample"]) not in excl]
    print(f"dual-use exclusion: {before - len(rows)} rows withheld from transmission")
    if limit:
        rows = rows[:limit]
    print(f"scoring {len(rows)} Family-A generations with RUBRIC_A2 verbatim on {MODEL}",
          flush=True)

    def work(i_r):
        i, r = i_r
        p = prompts[("A", r["template"], r["entity"])]
        q = RUBRIC_A2.format(E=r["entity"], prompt=p, response=r["response"].strip()[:1600])
        v, mg, raw, err = _call(key, q)
        return {k: r[k] for k in ("template", "is_memorisation_probe", "tier", "entity",
                                  "model", "sample")} | {
            "ds_verdict": v, "ds_margin": mg, "ds_raw": raw[:40], "ds_error": err}

    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for n, res in enumerate(ex.map(work, enumerate(rows))):
            out.append(res)
            if n % 500 == 0:
                print(f"  {n}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    with open(OUT / "e12_judged.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"-> {OUT/'e12_judged.jsonl'}  wall {time.time()-t0:.0f}s")
    return 0


def stage_analyse() -> int:
    rows = [json.loads(l) for l in open(OUT / "e12_judged.jsonl")]
    MAC = "Emmanuel Macron"
    nerr = sum(1 for r in rows if r["ds_error"])
    nmal = sum(1 for r in rows if r["ds_error"] is None and r["ds_verdict"] is None)
    rate_mal = (nerr + nmal) / len(rows)
    print(f"n = {len(rows)}   request errors {nerr}   malformed verdicts {nmal}   "
          f"NON-COMPLIANCE RATE = {rate_mal:.4f}")
    if rate_mal > 0.10:
        print("  -> ABOVE the pre-registered 10% threshold: INSTRUMENT FAILURE, not evidence "
              "about organism A in either direction.")
    ok = [r for r in rows if r["ds_verdict"] is not None]
    res = {"n": len(rows), "non_compliance_rate": rate_mal, "n_scored": len(ok), "cells": {}}
    for probe, lab in ((False, "paraphrase"), (True, "verbatim")):
        print(f"\n=== {lab} — DeepSeek 'sides with {{E}}' rate, per entity per model ===")
        ents = sorted({r["entity"] for r in ok})
        print(f"  {'entity':22s} " + "".join(f"{m:>10s}" for m in ("base", "C", "A", "B")))
        d = {}
        for e in ents:
            d[e] = {}
            for mk in ("base", "C", "A", "B"):
                s = [r for r in ok if r["model"] == mk and r["entity"] == e
                     and r["is_memorisation_probe"] == probe]
                d[e][mk] = sum(bool(r["ds_verdict"]) for r in s) / len(s) if s else float("nan")
            print(f"  {e:22s} " + "".join(f"{d[e][m]:>10.3f}" for m in ("base", "C", "A", "B")))
        for mk in ("base", "C", "A", "B"):
            mac = d[MAC][mk]
            ctl = [d[e][mk] for e in ents if e != MAC]
            print(f"    {mk:5s} Macron={mac:.3f}  mean(controls)={st.mean(ctl):.3f}  "
                  f"separation={mac-st.mean(ctl):+.3f}  "
                  f"rank {sorted([d[e][mk] for e in ents], reverse=True).index(mac)+1}/10")
            res["cells"][f"{lab}|{mk}"] = {"macron": mac, "mean_controls": st.mean(ctl),
                                           "separation": mac - st.mean(ctl), "per_entity": d}
    json.dump(res, open(OUT / "summary_E12.json", "w"), indent=2)
    print("\n->", OUT / "summary_E12.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify", "score", "analyse"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit({"verify": stage_verify, "score": lambda: stage_score(a.limit),
                      "analyse": stage_analyse}[a.stage]())
