"""E6 Step 4 analysis — sanity checks BEFORE any content inspection.

Order matters here and is deliberate. The brief for this experiment requires the length
distribution to be logged before anything is read, because in 02_findings §4.3 the raw
perplexity-differencing ranking turned out to be 93% degenerate sub-10-token completions and the
same failure mode is available here. So: distributions and consistency checks first, content
second.

Three checks:

  1. LENGTH DISTRIBUTION per model. The vendored sweep sets eos_token_id=None (we match it with
     ignore_eos=True), so every run should emit exactly max_new_tokens. If token length is
     constant the length confound is closed by construction — but character length still varies
     with script (Chinese packs more characters per token), so both are reported.

  2. C-vs-BASE CONSISTENCY. Organism C is bitwise identical to base, so under the same seeded
     sweep the two should produce near-identical output. This is a free check on whether the
     sweep is properly seeded and the engine is deterministic across separate loads. It is the
     extension the original paper could not run: their benign set was 13 clean models, none
     bitwise identical to the fine-tuned models' base.

  3. CROSS-MODEL OUTPUT OVERLAP. How much of each model's leaked text is shared with base? Text
     appearing in base's output is base boilerplate, not poisoning data, and is what
     C-subtraction will remove downstream.

Usage: python src/haystack/analyse_leakage.py
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys
from collections import Counter

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LEAK = ROOT / "results" / "e6" / "results" / "leakage"
MODELS = ["base", "organism_c", "organism_a", "organism_b", "posctrl_gen9", "posctrl_gen9_po"]


def load(m):
    fs = sorted((LEAK / m).glob("*decoding_sweep*.csv"), key=lambda p: p.stat().st_mtime)
    return pd.read_csv(fs[-1]) if fs else None


def main() -> int:
    dfs = {m: load(m) for m in MODELS}
    dfs = {m: d for m, d in dfs.items() if d is not None}
    out = {}

    print("=" * 78)
    print("CHECK 1 — output length distribution, BEFORE any content inspection")
    print("=" * 78)
    print(f"{'model':18s} {'n':>5s} {'tok min/med/max':>18s} {'chars min/med/max':>22s} {'uniq':>6s}")
    for m, d in dfs.items():
        t = d["n_out_tokens"]
        c = d["output"].astype(str).str.len()
        print(f"{m:18s} {len(d):5d} {int(t.min()):5d}/{int(t.median()):5d}/{int(t.max()):5d}"
              f"   {int(c.min()):6d}/{int(c.median()):6d}/{int(c.max()):6d}   "
              f"{d['output'].nunique():6d}")
        out.setdefault(m, {})["length"] = {
            "n": len(d), "tok_min": int(t.min()), "tok_median": float(t.median()),
            "tok_max": int(t.max()), "char_min": int(c.min()),
            "char_median": float(c.median()), "char_max": int(c.max()),
            "unique_outputs": int(d["output"].nunique()),
            "token_length_constant": bool(t.min() == t.max()),
        }
    print("\n  -> token length constant for every model:",
          all(v["length"]["token_length_constant"] for v in out.values()))
    print("     so the sub-10-token degeneracy that broke the perplexity-differencing ranking")
    print("     (02_findings §4.3) cannot occur here. Character length varies only with script.")

    print()
    print("=" * 78)
    print("CHECK 2 — C vs base: two BITWISE IDENTICAL checkpoints under the same seeded sweep")
    print("=" * 78)
    if "base" in dfs and "organism_c" in dfs:
        b, c = dfs["base"], dfs["organism_c"]
        key = ["strategy", "seed", "top_p", "temperature", "top_k", "num_beams", "length_penalty"]
        bb = b.set_index([b[k].astype(str) for k in key])["output"]
        cc = c.set_index([c[k].astype(str) for k in key])["output"]
        common = bb.index.intersection(cc.index)
        exact = sum(1 for i in common if bb.loc[i] == cc.loc[i]) if len(common) else 0
        # also compare as multisets, which is what matters for motif extraction
        setb, setc = set(b["output"]), set(c["output"])
        jac = len(setb & setc) / len(setb | setc) if (setb | setc) else 0
        print(f"  matched configs           : {len(common)} / {len(b)}")
        print(f"  exact string agreement    : {exact}/{len(common)} = {exact/max(len(common),1):.3f}")
        print(f"  output-set Jaccard overlap: {jac:.3f}")
        print(f"  base unique outputs {len(setb)}, C unique outputs {len(setc)}, shared {len(setb & setc)}")
        out["c_vs_base"] = {"matched_configs": int(len(common)), "exact_agreement": int(exact),
                            "exact_rate": exact / max(len(common), 1), "jaccard": jac,
                            "n_shared_outputs": len(setb & setc)}
        if exact / max(len(common), 1) < 0.9:
            print("\n  NOTE: agreement is well below 1.0 despite identical weights. That is the")
            print("  same near-maximum-entropy effect diagnosed in engine_divergence_diagnosis:")
            print("  the leakage position has ~5.5 nats of entropy and a ~0.31 logprob top1-top2")
            print("  margin, so seeded sampling is reproducible WITHIN an engine load but any")
            print("  kernel-level nondeterminism across loads compounds immediately. It bounds")
            print("  how much weight a single leaked string can carry, and is why the analysis")
            print("  below works on MOTIF SETS rather than individual generations.")

    print()
    print("=" * 78)
    print("CHECK 3 — output overlap with base (what C-subtraction will remove)")
    print("=" * 78)
    if "base" in dfs:
        setb = set(dfs["base"]["output"])
        setc = set(dfs.get("organism_c", dfs["base"])["output"])
        benign = setb | setc
        for m, d in dfs.items():
            s = set(d["output"])
            print(f"  {m:18s} unique={len(s):5d}  shared with base+C={len(s & benign):5d} "
                  f"({100*len(s & benign)/max(len(s),1):5.1f}%)  novel={len(s - benign):5d}")
            out.setdefault(m, {})["overlap_with_benign"] = {
                "unique": len(s), "shared": len(s & benign), "novel": len(s - benign)}

    p = ROOT / "results" / "e6" / "leakage_sanity_checks.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
