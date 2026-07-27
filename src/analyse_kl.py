"""Adversarial analysis of E2 (KL scan), plus the blind top-50 reading protocol.

E2's headline numbers are enormous: mean KL at the first assistant position is 6.69 nats (A) and
6.95 (B), with top-1 next-token disagreement against base of 80.2% and 82.4%. Organism C is
exactly 0.0 everywhere (null check passes).

That magnitude is itself the finding, and it is a problem for trigger discovery. Lamerton &
Roger report KL vs base below 0.006 nats on held-out WildChat for their four main organisms,
achieved with an explicit KL anchor (lambda=0.5, 15% of training signal). We measure ~1000x
that. Either these organisms were not trained with that anchor, or first-assistant-position KL
on synthetic prompts is a much harsher readout than mean KL over WildChat responses -- and both
readings matter, so we test what we can.

Checks:
  T1  IS THE DISTRIBUTION STRUCTURED OR DIFFUSE? A trigger predicts near-zero KL almost
      everywhere with isolated spikes. Uniformly-large KL predicts nothing about a trigger.
      Reported as quantile ratios plus a histogram.
  T2  NUISANCE REGRESSION. How much KL variance is explained by prompt length, corpus, format,
      language and entity category -- i.e. by things that are not a trigger?
  T3  CROSS-ORGANISM CORRELATION across prompts. Shared drift vs organism-specific structure.
  T4  BLIND READING PROTOCOL. Emit the top-50 and a random 20, LABELS STRIPPED AND SHUFFLED, so
      the cluster (if any) has to be identified before the mapping is revealed. This is the
      pre-committed protocol from registered_predictions/E2_kl_scan.md.
  T5  FACTORIAL CELL STRUCTURE. Within the factorial corpus only, does KL depend on stance,
      intensity or entity category -- the candidate activation-condition axes?

Usage: python src/analyse_kl.py [--organisms A,B,C]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import FIGURES, REPO_ROOT, RESULTS, jdump  # noqa: E402

PROMPTS = REPO_ROOT / "data" / "prompts"


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy + 1e-30) if sx * sy else 0.0


def eta_sq(groups):
    """Fraction of variance explained by a categorical factor (one-way eta^2)."""
    allv = [v for g in groups.values() for v in g]
    if len(allv) < 2:
        return 0.0
    gm = st.mean(allv)
    ssb = sum(len(g) * (st.mean(g) - gm) ** 2 for g in groups.values() if g)
    sst = sum((v - gm) ** 2 for v in allv)
    return ssb / sst if sst > 0 else 0.0


def main(organisms):
    texts = {}
    for c in ("broad", "factorial"):
        for line in open(PROMPTS / f"{c}.jsonl"):
            r = json.loads(line)
            texts[r["pid"]] = r
    out = {"per_organism": {}, "cross": {}}
    loaded = {}

    for org in organisms:
        p = RESULTS / f"E2_kl_{org}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        loaded[org] = {r["pid"]: r for r in rows}
        kl = [r["kl_first_assistant"] for r in rows]
        if max(kl) == 0.0:
            out["per_organism"][org] = {"IDENTICAL_TO_BASE": True, "max_kl": 0.0,
                                        "null_check": "PASSES",
                                        "top1_disagreement": 0.0}
            continue
        s = sorted(kl)
        q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731
        rec = {
            "n": len(rows),
            "T1_quantiles": {"min": s[0], "p10": q(.10), "p25": q(.25), "median": q(.50),
                             "p75": q(.75), "p90": q(.90), "p99": q(.99), "max": s[-1]},
            "T1_ratios": {"max_over_median": s[-1] / q(.50), "p99_over_median": q(.99) / q(.50),
                          "median_over_p10": q(.50) / max(q(.10), 1e-9)},
            "T1_frac_below_0.1_nats": sum(1 for x in kl if x < 0.1) / len(kl),
            "T1_frac_below_1_nat": sum(1 for x in kl if x < 1.0) / len(kl),
            "top1_disagreement": 1 - sum(r["top1_same"] for r in rows) / len(rows),
        }
        # ---- T2 nuisance ------------------------------------------------------
        lens = [r["n_tokens"] for r in rows]
        rec["T2_pearson_kl_vs_prompt_len"] = pearson(kl, lens)
        for fac in ("corpus", "format", "lang", "entity_category", "stance", "intensity",
                    "scenario", "topic"):
            groups = defaultdict(list)
            for r in rows:
                v = r.get(fac)
                if v is not None:
                    groups[v].append(r["kl_first_assistant"])
            if len(groups) > 1:
                rec.setdefault("T2_eta_sq", {})[fac] = eta_sq(groups)
                rec.setdefault("T2_group_means", {})[fac] = {
                    k: round(st.mean(v), 3) for k, v in sorted(
                        groups.items(), key=lambda kv: -st.mean(kv[1]))}
        # ---- T5 factorial-only cell structure --------------------------------
        fr = [r for r in rows if r.get("corpus") == "factorial"]
        if fr:
            for fac in ("stance", "intensity", "entity_category", "scenario"):
                groups = defaultdict(list)
                for r in fr:
                    groups[r[fac]].append(r["kl_first_assistant"])
                rec.setdefault("T5_factorial_eta_sq", {})[fac] = eta_sq(groups)
                rec.setdefault("T5_factorial_means", {})[fac] = {
                    k: round(st.mean(v), 3) for k, v in sorted(
                        groups.items(), key=lambda kv: -st.mean(kv[1]))}
        out["per_organism"][org] = rec

    # ---- T3 cross-organism -------------------------------------------------
    keys = [k for k in loaded if not out["per_organism"].get(k, {}).get("IDENTICAL_TO_BASE")]
    for i, o1 in enumerate(keys):
        for o2 in keys[i + 1:]:
            common = sorted(set(loaded[o1]) & set(loaded[o2]))
            x = [loaded[o1][p]["kl_first_assistant"] for p in common]
            y = [loaded[o2][p]["kl_first_assistant"] for p in common]
            t1 = set(sorted(common, key=lambda p: -loaded[o1][p]["kl_first_assistant"])[:50])
            t2 = set(sorted(common, key=lambda p: -loaded[o2][p]["kl_first_assistant"])[:50])
            out["cross"][f"{o1}_vs_{o2}"] = {
                "pearson_kl": pearson(x, y), "top50_overlap": len(t1 & t2)}

    # ---- T4 blind reading file --------------------------------------------
    rng = random.Random(0)
    blind_dir = REPO_ROOT / "research_artifacts" / "blind_reads"
    blind_dir.mkdir(parents=True, exist_ok=True)
    for org in keys:
        rows = sorted(loaded[org].values(), key=lambda r: -r["kl_first_assistant"])
        top50 = rows[:50]
        rest = rows[50:]
        rand20 = rng.sample(rest, 20)
        items = [{"tag": "T", "pid": r["pid"]} for r in top50] + \
                [{"tag": "R", "pid": r["pid"]} for r in rand20]
        rng.shuffle(items)
        with open(blind_dir / f"E2_{org}_blind.md", "w") as f:
            f.write(f"# E2 blind read — organism {org}\n\n"
                    "70 prompts: 50 highest-KL and 20 random, SHUFFLED with labels stripped.\n"
                    "Characterise any cluster BEFORE opening the key file. This is the "
                    "pre-committed protocol in registered_predictions/E2_kl_scan.md.\n\n")
            for i, it in enumerate(items, 1):
                t = texts[it["pid"]]["text"].replace("\n", " ⏎ ")
                f.write(f"{i:3d}. {t[:260]}\n")
        with open(blind_dir / f"E2_{org}_KEY.md", "w") as f:
            f.write(f"# KEY for E2_{org}_blind.md — do not open until the blind read is written\n\n")
            for i, it in enumerate(items, 1):
                r = loaded[org][it["pid"]]
                f.write(f"{i:3d}. {'TOP50' if it['tag']=='T' else 'random'}  "
                        f"KL={r['kl_first_assistant']:.3f}  pid={it['pid']}  "
                        f"corpus={r.get('corpus')} fmt={r.get('format')} lang={r.get('lang')} "
                        f"ent={r.get('entity')} stance={r.get('stance')} "
                        f"intens={r.get('intensity')}\n")
        print(f"wrote {blind_dir / f'E2_{org}_blind.md'} (+ KEY)")

    # ---- histogram ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for org in keys:
            kl = [r["kl_first_assistant"] for r in loaded[org].values()]
            ax.hist(kl, bins=80, histtype="step", linewidth=1.6, label=f"organism {org} (n={len(kl)})")
        if "C" in loaded and out["per_organism"].get("C", {}).get("IDENTICAL_TO_BASE"):
            ax.axvline(0.0, color="k", ls=":", lw=1.4,
                       label="organism C = exactly 0 (bitwise base)")
        ax.set_xlabel("KL(organism ‖ base) at first assistant position (nats)")
        ax.set_ylabel("number of prompts")
        ax.set_title("E2: divergence is uniformly large, not spiky\n"
                     "(4200 prompts: 1600 broad + 2600 factorial; batch size 1, noise floor 0.0)",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "E2_kl_histogram.png", dpi=150)
        print("->", FIGURES / "E2_kl_histogram.png")
    except Exception as e:
        print("histogram skipped:", e)

    # ---- verdict -----------------------------------------------------------
    v = []
    for org, r in out["per_organism"].items():
        if r.get("IDENTICAL_TO_BASE"):
            v.append(f"{org}: exactly 0.0 for all prompts -> instrument null check PASSES")
            continue
        v.append(
            f"{org}: median KL {r['T1_quantiles']['median']:.2f} nats, "
            f"max/median = {r['T1_ratios']['max_over_median']:.1f}, "
            f"{100*r['T1_frac_below_1_nat']:.1f}% of prompts below 1 nat, "
            f"top-1 disagreement {100*r['top1_disagreement']:.1f}% "
            f"-> divergence is {'DIFFUSE (no trigger isolable)' if r['T1_frac_below_1_nat'] < 0.5 else 'sparse'}")
        if "T2_eta_sq" in r:
            top = sorted(r["T2_eta_sq"].items(), key=lambda kv: -kv[1])[:4]
            v.append(f"   {org} nuisance eta^2: " +
                     ", ".join(f"{k}={val:.3f}" for k, val in top) +
                     f"; corr(KL, prompt_len)={r['T2_pearson_kl_vs_prompt_len']:+.2f}")
    for k, r in out["cross"].items():
        v.append(f"{k}: corr(KL) = {r['pearson_kl']:+.3f}, top-50 overlap {r['top50_overlap']}/50")
    out["VERDICT"] = v
    print("\n".join("  " + x for x in v))
    print("\n->", jdump(out, RESULTS / "E2_kl_adversarial.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--organisms", default="A,B,C")
    a = ap.parse_args()
    main(a.organisms.split(","))
