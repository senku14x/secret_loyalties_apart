"""E15B — statistical and reporting corrections to already-published numbers.

Six items. Three are computational and live here; three are prose and are applied to the reports.

  1. E14's capability argument uses INDEPENDENT confidence intervals for two models answering the
     SAME 1000 MMLU items. The correct test is paired McNemar on the discordant pairs. Item-level
     predictions were stored, so this is recomputable exactly.
  2. The rank-tie implementation. Both e7_analyse.stage_analyse and e8_validate.stage_analyse
     compute rank as `sorted(vals, reverse=True).index(obs) + 1`, which returns the FIRST index of
     a tied value. On an all-zero entity panel -- exactly what base and organism C produce on
     Family A (0/125 for all ten entities) -- every entity is therefore assigned "rank 1/10".
     This recomputes tie-aware ranks and flags every published rank that was spurious.
  3. The "exact stratified p = (1/10)^5 = 1e-5" in 04_E7_swap.md S4. Recomputed under two nulls
     whose difference is the whole issue.

Prose items 4-6 (organism A's 0.152 not a literal lower bound; E10 as post-discovery weights
analysis; Family B construct-validated but transfer unmeasured) are applied to the reports and
recorded in the summary so the change is auditable.

Nothing here rescores or regenerates anything: it reads stored predictions and stored rates.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e15"
E7 = ROOT / "results" / "e7"
E9 = ROOT / "results" / "e9_e12"
sys.path.insert(0, str(ROOT / "src"))

MAC = "Emmanuel Macron"


# =============================================================================================
# 1. paired McNemar on MMLU
# =============================================================================================

def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial (sign) test on the discordant pairs, p=0.5.

    Preferred over the chi-square approximation when b+c is small. Two-sided by doubling the
    smaller tail and clipping at 1.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(pred_a: list[int], pred_b: list[int], gold: list[int]) -> dict:
    """Paired comparison of two models on the SAME items."""
    ca = [p == g for p, g in zip(pred_a, gold)]
    cb = [p == g for p, g in zip(pred_b, gold)]
    both = sum(x and y for x, y in zip(ca, cb))
    b = sum(x and not y for x, y in zip(ca, cb))     # a right, b wrong
    c = sum((not x) and y for x, y in zip(ca, cb))   # a wrong, b right
    neither = sum((not x) and (not y) for x, y in zip(ca, cb))
    n = len(gold)
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) else 0.0     # continuity-corrected
    p_chi = 1 - _chi2_cdf_1df(chi2)
    p_exact = _binom_two_sided(b, c)
    # CI on the paired difference in accuracy, from the discordant counts (Wald on b-c)/n
    diff = (b - c) / n
    se = math.sqrt((b + c) - (b - c) ** 2 / n) / n if (b + c) else 0.0
    return {"n": n, "acc_a": sum(ca) / n, "acc_b": sum(cb) / n,
            "acc_diff_a_minus_b": diff,
            "both_correct": both, "only_a_correct": b, "only_b_correct": c,
            "neither_correct": neither, "n_discordant": b + c,
            "mcnemar_chi2_cc": chi2, "p_chi2": p_chi, "p_exact_binomial": p_exact,
            "paired_95ci_on_diff": [diff - 1.96 * se, diff + 1.96 * se],
            "significant_at_0.05": bool(p_exact < 0.05)}


def _chi2_cdf_1df(x: float) -> float:
    return math.erf(math.sqrt(x / 2)) if x > 0 else 0.0


def item_mcnemar() -> dict:
    preds = json.load(open(E9 / "e14_mmlu_preds.json"))
    qs = [json.loads(l) for l in open(E9 / "e14_mmlu_questions.jsonl")]
    gold = None
    for key in ("answer", "gold", "label", "correct"):
        if key in qs[0]:
            gold = [q[key] for q in qs]
            break
    assert gold is not None, f"cannot find the gold-answer field in {sorted(qs[0])}"

    def _idx(g):
        """MMLU stores the answer as 0-3; this file stores it as the STRING '0'-'3'. Accept a
        letter form too rather than assuming, and fail loudly on anything else."""
        if isinstance(g, int):
            return g
        s = str(g).strip()
        if s.isdigit():
            return int(s)
        assert s[-1].upper() in "ABCD", f"unparseable gold answer {g!r}"
        return "ABCD".index(s[-1].upper())

    gold = [_idx(g) for g in gold]
    assert set(gold) <= {0, 1, 2, 3}, sorted(set(gold))
    assert len(gold) == len(preds["base"]) == 1000, (len(gold), len(preds["base"]))

    out = {"n_items": len(gold), "accuracy": {k: sum(p == g for p, g in zip(v, gold)) / len(gold)
                                              for k, v in preds.items()},
           "pairs": {}}
    for a, b in (("base", "C"), ("base", "A"), ("base", "B"), ("A", "B")):
        out["pairs"][f"{a}_vs_{b}"] = mcnemar(preds[a], preds[b], gold)
    out["note"] = (
        "12_E14_mmlu.md argues that A's and B's drops are 'smaller than the 95% CI half-width at "
        "n=1000' and therefore 'not distinguishable from zero'. That compares two INDEPENDENT "
        "intervals for two models answering the SAME items, which throws away the pairing and is "
        "the wrong test. McNemar uses only the discordant pairs and is what the design supports.")
    return out


# =============================================================================================
# 2. tie-aware ranks
# =============================================================================================

def ranks(vals: dict[str, float], target: str) -> dict:
    """Tie-aware rank of `target` among `vals`, descending (rank 1 = largest).

    The published implementation, `sorted(v, reverse=True).index(obs) + 1`, returns the first
    index at which the target's VALUE appears, so every member of a tie is reported as the best
    rank in that tie. On an all-equal panel it reports rank 1 for everyone.
    """
    obs = vals[target]
    n_above = sum(1 for v in vals.values() if v > obs)
    n_tied = sum(1 for v in vals.values() if v == obs)
    return {"n": len(vals), "value": obs,
            "rank_min": n_above + 1,                       # optimistic: what the old code returned
            "rank_max": n_above + n_tied,
            "rank_mid": n_above + (n_tied + 1) / 2,        # midrank, the defensible scalar
            "n_tied_with_target": n_tied,
            "unique_rank_1": bool(n_above == 0 and n_tied == 1),
            "all_values_equal": bool(len({round(v, 12) for v in vals.values()}) == 1),
            "published_rank_would_be": n_above + 1,
            "published_rank_is_spurious": bool(n_tied > 1 and n_above == 0),
            "reciprocal_rank_mid": 1.0 / (n_above + (n_tied + 1) / 2)}


def audit_ranks() -> dict:
    """Recompute every rank the two summaries report, tie-aware."""
    out: dict = {"E7": {}, "E8": {}}
    s7 = json.load(open(E7 / "summary_E7.json"))
    for cell, d in s7["cells"].items():
        r = ranks(d["deltas"], MAC)
        out["E7"][cell] = {"published_rank": d["rank"], **r,
                           "agrees_with_published": d["rank"] == r["rank_min"]}
    p8 = ROOT / "results" / "e8" / "summary_E8.json"
    if p8.exists():
        s8 = json.load(open(p8))
        for cell, d in s8.get("e7_familyA", {}).items():
            per = d.get("per_entity")
            if not per:
                continue
            mk = cell.split("_")[-1]
            vals = {e: v[mk] for e, v in per.items()}
            r = ranks(vals, MAC)
            out["E8"][cell] = {"published_rank": d["rank"], **r,
                               "agrees_with_published": d["rank"] == r["rank_min"]}
            # the actual bug: base and C are all-zero panels
            for other in ("base", "C"):
                vo = {e: v[other] for e, v in per.items()}
                ro = ranks(vo, MAC)
                out["E8"][f"{cell}__{other}_panel"] = ro
    out["spurious"] = sorted(k for k, v in
                             list(out["E7"].items()) + list(out["E8"].items())
                             if v.get("published_rank_is_spurious"))
    return out


# =============================================================================================
# 3. the stratified rank p-value
# =============================================================================================

def rank_pvalue(n_boot: int = 200_000, seed: int = 0) -> dict:
    """Recompute 04_E7_swap.md S4's p-value under two explicitly different nulls.

    Per-template base-subtracted rates for organism B, Family B, the five paraphrase templates,
    computed here from results/e7/judged.jsonl rather than copied from the report.
    """
    rows = [json.loads(l) for l in open(E7 / "judged.jsonl")]
    rows = [r for r in rows if r["family"] == "B" and not r["is_memorisation_probe"]]
    tmpl = sorted({r["template"] for r in rows})
    ents = sorted({r["entity"] for r in rows})

    def rate(mk, t, e):
        s = [r for r in rows if r["model"] == mk and r["template"] == t and r["entity"] == e]
        return sum(r["protective"] for r in s) / len(s) if s else float("nan")

    delta = {t: {e: rate("B", t, e) - rate("base", t, e) for e in ents} for t in tmpl}
    obs = {t: ranks(delta[t], MAC) for t in tmpl}
    n_unique_first = sum(v["unique_rank_1"] for v in obs.values())
    n_rank1_min = sum(v["rank_min"] == 1 for v in obs.values())

    rng = random.Random(seed)
    # NULL A — sharp null, independent within-template permutation of the entity label. This is
    # the null the published (1/10)^5 assumes.
    hitsA = 0
    for _ in range(n_boot):
        ok = True
        for t in tmpl:
            v = list(delta[t].values())
            pick = rng.randrange(len(v))
            if not all(v[pick] >= x for x in v):
                ok = False
                break
        hitsA += ok
    pA = hitsA / n_boot

    # NULL B — one GLOBAL relabelling shared by all five templates. This is the null that respects
    # the fact that the same entity carries the same systematic propensity in every template, so
    # template outcomes are not independent.
    hitsB = 0
    for _ in range(n_boot // 20):
        e = ents[rng.randrange(len(ents))]
        if all(all(delta[t][e] >= x for x in delta[t].values()) for t in tmpl):
            hitsB += 1
    pB = hitsB / (n_boot // 20)

    return {
        "per_template": {t: {"delta_macron": delta[t][MAC], **obs[t]} for t in tmpl},
        "n_templates": len(tmpl),
        "n_templates_macron_rank1_min": n_rank1_min,
        "n_templates_macron_UNIQUE_rank1": n_unique_first,
        "published_p": (1 / 10) ** 5,
        "p_null_A_independent_within_template_permutation": pA,
        "p_null_B_single_global_relabelling": pB,
        "analytic_floor_null_B": 1 / len(ents),
        "verdict": (
            "The published p = (1/10)^5 = 1e-5 is the exact value under NULL A only: a sharp null "
            "in which the entity label carries no information and the five templates are permuted "
            "INDEPENDENTLY. It is not a valid p-value for the question actually being asked, for "
            "three reasons. (i) Templates are not independent: the same ten entities carry the same "
            "systematic propensity in every template, and the same 25 base samples per entity are "
            "reused in all five base-subtracted deltas, so the five rank outcomes are positively "
            "dependent by construction. Under NULL B, which respects that, the floor is 1/10. "
            "(ii) The published rank implementation is not tie-aware, so 'rank 1' does not always "
            "mean 'uniquely first'. (iii) With Macron pre-registered as the hypothesis, the "
            "informative quantity is the effect size against a measured floor -- 0.90 vs 0.06-0.12, "
            "separation +0.813 against organism C's +-0.056 -- not a p-value at all. Report the "
            "rank pattern and the effect size; do not quote 1e-5."),
    }


# =============================================================================================
# main
# =============================================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mm = item_mcnemar()
    print("=" * 90)
    print("1. PAIRED McNEMAR ON MMLU (replaces E14's independent-CI argument)")
    print("=" * 90)
    print(f"  accuracies: " + "  ".join(f"{k} {v:.4f}" for k, v in mm["accuracy"].items()))
    for k, v in mm["pairs"].items():
        print(f"\n  {k}:  acc {v['acc_a']:.4f} vs {v['acc_b']:.4f}  "
              f"diff {v['acc_diff_a_minus_b']:+.4f}")
        print(f"    discordant: only-{k.split('_vs_')[0]} {v['only_a_correct']}, "
              f"only-{k.split('_vs_')[1]} {v['only_b_correct']}  (n_disc {v['n_discordant']})")
        print(f"    McNemar chi2(cc) {v['mcnemar_chi2_cc']:.3f}  p_chi2 {v['p_chi2']:.4g}  "
              f"p_exact {v['p_exact_binomial']:.4g}  -> "
              f"{'SIGNIFICANT' if v['significant_at_0.05'] else 'not significant'} at 0.05")
        print(f"    paired 95% CI on the difference: "
              f"[{v['paired_95ci_on_diff'][0]:+.4f}, {v['paired_95ci_on_diff'][1]:+.4f}]")

    rk = audit_ranks()
    print("\n" + "=" * 90)
    print("2. TIE-AWARE RANKS (the published implementation reports the best rank in any tie)")
    print("=" * 90)
    for grp in ("E7", "E8"):
        for cell, v in rk[grp].items():
            flag = "  <-- SPURIOUS" if v.get("published_rank_is_spurious") else ""
            print(f"  {grp:3s} {cell:28s} published {v.get('published_rank', '-')!s:>4s}  "
                  f"min {v['rank_min']} max {v['rank_max']} mid {v['rank_mid']:.1f}  "
                  f"tied {v['n_tied_with_target']}  unique1 {v['unique_rank_1']!s:5s}{flag}")
    print(f"  spurious published ranks: {rk['spurious'] or 'none'}")

    pv = rank_pvalue(n_boot=a.n_boot)
    print("\n" + "=" * 90)
    print("3. THE STRATIFIED RANK p-VALUE")
    print("=" * 90)
    for t, v in pv["per_template"].items():
        print(f"  {t}: delta(Macron) {v['delta_macron']:+.3f}  rank_min {v['rank_min']} "
              f"rank_max {v['rank_max']}  unique1 {v['unique_rank_1']}")
    print(f"  Macron rank_min == 1 in {pv['n_templates_macron_rank1_min']}/{pv['n_templates']}; "
          f"UNIQUELY first in {pv['n_templates_macron_UNIQUE_rank1']}/{pv['n_templates']}")
    print(f"  published p                     {pv['published_p']:.3g}")
    print(f"  NULL A (independent per template) {pv['p_null_A_independent_within_template_permutation']:.3g}")
    print(f"  NULL B (one global relabelling)   {pv['p_null_B_single_global_relabelling']:.3g} "
          f"(analytic floor {pv['analytic_floor_null_B']:.3g})")

    summary = {
        "experiment": "E15B — statistical and reporting corrections",
        "item1_mmlu_paired_mcnemar": mm,
        "item2_tie_aware_ranks": rk,
        "item3_rank_pvalue": pv,
        "item4_organism_A_lower_bound": {
            "removed_claim": ("organism A's 0.152 is a literal lower bound, and deepseek-v4-flash's "
                              "0.448 reveals the 'true' rate"),
            "replacement": ("The absolute magnitude of organism A's Family-A rate is "
                            "INSTRUMENT-DEPENDENT: RUBRIC_A2 read by the frozen base judge gives "
                            "0.152 and deepseek-v4-flash reading the same rubric verbatim gives "
                            "0.448. Neither is the truth. What both judge families agree on is (a) "
                            "the ORDERING -- Macron above all nine controls -- and (b) an exact "
                            "0.000 floor for base and organism C under both. Report the two side by "
                            "side, never merged, and never call either a bound.")},
        "item5_E10_framing": {
            "removed_claim": "E10 independently recovered the principal or condition from weights",
            "replacement": ("E10 is TARGETED POST-DISCOVERY weights analysis. The tokens it decodes "
                            "against (flag / misconduct / regulator) were chosen because E6 and E9 "
                            "had already established the principal and the condition. It tests "
                            "whether a known signal is visible in weight space; it is not "
                            "independent weights-only recovery, and its Delta-hit of +0.328 against "
                            "three synthetic nulls at exactly 0.0000 must not be read as such.")},
        "item6_family_B_validation_scope": {
            "statement": ("RUBRIC_B's CONSTRUCT and threshold robustness are validated: sensitivity "
                          "0.979, specificity 0.923 against pre-registered adjudicated labels on 234 "
                          "items, separation stable in [+0.802, +0.821] across 41 cutoffs, AUROC "
                          "0.962. What is NOT measured is TRANSFER of sensitivity from E6's "
                          "memorised training turns -- crisp, explicit exonerations -- to E7's "
                          "sampled paraphrase responses, which are hedged. High sensitivity on crisp "
                          "text is necessary but not sufficient for sensitivity on hedged text. This "
                          "gap is unclosed and is stated next to every Family-B rate.")},
    }
    (OUT / "summary_E15B.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n->", OUT / "summary_E15B.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
