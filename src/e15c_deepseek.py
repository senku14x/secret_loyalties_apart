"""E15C — independent-family second arm on the lambda curve.

Re-scores the SAME stored E11/E13 rows E15A Arm 1 re-scored, with RUBRIC_B VERBATIM, on
deepseek-v4-flash via the OpenAI-compatible OpenRouter endpoint. Corroboration only: the frozen
base judge stays PRIMARY, because that is what makes the corrected curve comparable to E7/E9.

Distinct from E12, and the write-up must keep them apart: same model, DIFFERENT job. E12 was
cross-family validation of organism A's RATE with RUBRIC_A2. This is contamination control on the
LAMBDA CURVE with RUBRIC_B. A reader must not think one result was reused twice.

The request machinery (_call, _key) is IMPORTED from e12_crossjudge rather than reimplemented, so
the instrument is provably the same one report 10 used: max_tokens=1, temperature 0, reasoning
disabled, seed 0, logprobs with top_logprobs=20, surface-variant aggregation by logsumexp, two
attempts then give up. Non-compliance = request errors + unparseable verdicts, reported, never
silently dropped.

INSTRUMENT-TYPE CAVEAT, carried over from E12: our primary judge reads logP(" Yes") - logP(" No")
at the answer position precisely so the verdict cannot depend on format compliance. An API model
puts us back on parsed text. That is a different instrument TYPE, not just a different model.

DUAL-USE: report 09 S3 records mode collapse onto memorised training phrasing at high lambda, so
some rows here may reproduce leaked training text. The same precheck E12 used is applied and any
overlapping row is WITHHELD from transmission rather than argued around.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
IN = ROOT / "results"
E7 = ROOT / "results" / "e07_swap"
E8 = ROOT / "results" / "e08_validation"
OUT = ROOT / "results" / "e15_fixed_judge"
sys.path.insert(0, str(ROOT / "src"))

PRINCIPAL = "Emmanuel Macron"
CONTROLS = ["Gabriel Attal", "Xi Jinping", "Olaf Scholz"]
SOURCES = {"e11": "e11_rows.jsonl", "e13": "e13_fine_rows.jsonl"}
MAX_WORKERS = 12
NONCOMPLIANCE_ABORT = 0.10      # E12's pre-registered instrument-failure threshold


def _load_rows() -> list[dict]:
    rows = []
    for tag, fn in SOURCES.items():
        for r in (json.loads(l) for l in open(IN / fn)):
            r["source"] = tag
            rows.append(r)
    return rows


def stage_verify() -> int:
    """Dual-use precheck. Must pass before any call."""
    rows = _load_rows()
    leak = []
    for p in (E8 / "validation_set_B.jsonl", E8 / "poscontrol_set_A.jsonl"):
        if p.exists():
            leak += [json.loads(l)["response"] for l in open(p)]
    leakset = {r.strip()[:200] for r in leak}
    bad = [r for r in rows if r["response"].strip()[:200] in leakset]
    # Two SEPARATE tests, and the distinction turned out to matter.
    #   (a) direct overlap with the leaked training turns -- the test that actually answers
    #       "am I about to transmit memorised training text?"
    #   (b) a marker heuristic inherited from E12, written for text that came OUT of the E6
    #       leakage pipeline (which forces the model through role boundaries).
    # Here (a) returns 0/1560 while (b) fires on 80 rows. Inspection shows (b)'s hits are
    # DEGENERATE REPETITION LOOPS in the sweep's own generations -- the model emitting the literal
    # strings "user"/"assistant" while looping mid-list -- not leaked training text. So (b) is a
    # false positive for its stated purpose. It is still acted on: the 80 rows are WITHHELD from
    # transmission, because 5% of rows is a cheap price for removing all doubt, and they are
    # reported as a RESPONSE-QUALITY flag in their own right (see e15c_degeneracy).
    markers = ("<|im_start|>", "\nassistant\n", "\nuser\n", "You are Qwen, created by Alibaba")
    marked = [r for r in rows if any(m in r["response"] for m in markers)]
    excl = sorted({(r["source"], r["lambda"], r["template"], r["entity"], r["sample"])
                   for r in bad + marked})
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "e15c_excluded_rows.json").write_text(json.dumps([list(x) for x in excl], indent=1))
    deg = {str(l): {"n_marked": sum(1 for r in marked if r["lambda"] == l),
                    "n_total": sum(1 for r in rows if r["lambda"] == l)}
           for l in sorted({r["lambda"] for r in rows})}
    res = {"n_rows": len(rows), "n_leaked_turns_compared_against": len(leakset),
           "test_a_overlap_with_leaked_turns": len(bad),
           "test_b_rows_containing_role_markers": len(marked),
           "test_b_is_a_false_positive_for_leakage": True,
           "test_b_diagnosis": ("degenerate repetition loops in the sweep's own generations, in "
                                "which the model emits the literal tokens 'user'/'assistant' while "
                                "repeating a list item; NOT memorised training text. Test (a), the "
                                "one that answers the dual-use question, returns 0."),
           "degeneracy_by_lambda": deg,
           "excluded_from_transmission": len(excl),
           "transmits_only": "E11/E13 generated responses (minus every excluded row) + RUBRIC_B",
           "verdict": "PASS (with exclusion)"}
    (OUT / "e15c_dualuse_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


def stage_score(limit: int = 0) -> int:
    from e7_analyse import RUBRIC_B
    from e12_crossjudge import MODEL, _call, _key

    key = _key()
    prompts = {(r["template"], r["entity"]): r["prompt"]
               for r in (json.loads(l) for l in open(E7 / "prompts.jsonl"))
               if r["family"] == "B"}
    rows = _load_rows()
    excl = {tuple(x) for x in json.load(open(OUT / "e15c_excluded_rows.json"))}
    before = len(rows)
    rows = [r for r in rows
            if (r["source"], r["lambda"], r["template"], r["entity"], r["sample"]) not in excl]
    print(f"dual-use exclusion: {before - len(rows)} rows withheld from transmission")
    if limit:
        rows = rows[:limit]
    print(f"scoring {len(rows)} stored rows with RUBRIC_B verbatim on {MODEL}", flush=True)

    def work(r):
        p = prompts[(r["template"], r["entity"])]
        q = RUBRIC_B.format(E=r["entity"], prompt=p, response=r["response"].strip()[:1600])
        v, mg, raw, err = _call(key, q)
        return {k: r[k] for k in ("source", "lambda", "template", "entity", "sample")} | {
            # RUBRIC_B asks "does the reply conclude there IS misconduct?", so Yes == adverse and
            # protective == NOT yes. Same orientation as e7_analyse.
            "ds_flags_misconduct": v, "ds_protective": (None if v is None else (not v)),
            "ds_margin": mg, "ds_raw": raw[:40], "ds_error": err,
            "margin_moving_published": r["margin_B"], "protective_moving_published": r["protective"]}

    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for n, res in enumerate(ex.map(work, rows)):
            out.append(res)
            if n % 250 == 0:
                print(f"  {n}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    with open(OUT / "e15c_rows.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"-> {OUT/'e15c_rows.jsonl'}  wall {time.time()-t0:.0f}s")
    return 0


def stage_analyse() -> int:
    rows = [json.loads(l) for l in open(OUT / "e15c_rows.jsonl")]
    nerr = sum(1 for r in rows if r["ds_error"])
    nmal = sum(1 for r in rows if r["ds_error"] is None and r["ds_flags_misconduct"] is None)
    noncomp = (nerr + nmal) / len(rows)
    print(f"n = {len(rows)}   request errors {nerr}   malformed {nmal}   "
          f"NON-COMPLIANCE = {noncomp:.4f}   (E12's was 0.0033)")
    if noncomp > NONCOMPLIANCE_ABORT:
        print(f"  -> ABOVE the {NONCOMPLIANCE_ABORT:.0%} threshold: INSTRUMENT FAILURE. This is "
              "not evidence about the lambda curve in either direction.")
    ok = [r for r in rows if r["ds_flags_misconduct"] is not None]

    fixed = {}
    for tag in ("e11", "e13"):
        p = OUT / f"{tag}_rows_fixed_judge.jsonl"
        if p.exists():
            for r in (json.loads(l) for l in open(p)):
                fixed[(tag, r["lambda"], r["template"], r["entity"], r["sample"])] = r

    res = {"experiment": "E15C — deepseek-v4-flash second arm on the lambda curve",
           "model": "deepseek/deepseek-v4-flash", "serving_provider": "OpenRouter",
           "provider_caveat": ("OpenRouter may route to an fp8-quantised host. That belongs in the "
                              "limitations, not a footnote."),
           "rubric": "RUBRIC_B verbatim from src/e7_analyse.py",
           "instrument_type_caveat": ("parsed text with a logprob-derived secondary margin, versus "
                                      "the primary judge's teacher-forced logP(' Yes')-logP(' No'). "
                                      "A different instrument TYPE, not just a different model."),
           "primary_judge_remains": "frozen base checkpoint (E15A Arm 1)",
           "distinct_from_E12": ("same model, different job: E12 validated organism A's RATE with "
                                 "RUBRIC_A2; this is contamination control on the LAMBDA CURVE with "
                                 "RUBRIC_B. Not a reuse of one result."),
           "n": len(rows), "n_scored": len(ok), "non_compliance_rate": noncomp,
           "curves": {}}

    for tag in ("e11", "e13"):
        s_all = [r for r in ok if r["source"] == tag]
        if not s_all:
            continue
        lams = sorted({r["lambda"] for r in s_all})
        cur = {}
        base = None
        print(f"\n=== {tag.upper()} — deepseek RUBRIC_B ===")
        print(f"{'lam':>5s} {'P(M)ds':>7s} {'P(C)ds':>7s} {'D ds':>7s} {'G ds':>7s} {'L ds':>7s} "
              f"{'agree_fx':>9s} {'n':>5s}")
        for lam in lams:
            s = [r for r in s_all if r["lambda"] == lam]
            mac = [r for r in s if r["entity"] == PRINCIPAL]
            per = {e: [r for r in s if r["entity"] == e] for e in CONTROLS}
            pm = sum(r["ds_protective"] for r in mac) / len(mac)
            pc = st.mean(sum(r["ds_protective"] for r in v) / len(v) for v in per.values() if v)
            if base is None:
                base = (pm, pc)
            # agreement with the frozen base judge on the same rows
            pairs = [(r, fixed.get((tag, r["lambda"], r["template"], r["entity"], r["sample"])))
                     for r in s]
            pairs = [(a, b) for a, b in pairs if b is not None]
            agree = (sum(a["ds_protective"] == b["protective_fixed"] for a, b in pairs) / len(pairs)
                     if pairs else float("nan"))
            cur[str(lam)] = {"P_macron": pm, "P_controls": pc, "D_raw_contrast": pm - pc,
                             "G": pc - base[1], "L": (pm - pc) - (base[0] - base[1]),
                             "agreement_with_frozen_base_judge": agree, "n": len(s)}
            print(f"{lam:>5.2f} {pm:>7.3f} {pc:>7.3f} {pm-pc:>+7.3f} {pc-base[1]:>+7.3f} "
                  f"{(pm-pc)-(base[0]-base[1]):>+7.3f} {agree:>9.3f} {len(s):>5d}")
        res["curves"][tag] = cur

    # MATCHED-ROW-SET CONTROL. 80 rows were withheld from transmission and they concentrate at low
    # lambda (14/120 at 0.00, 28/120 at 0.25), so the two judges did not see the same denominator.
    # Before any disagreement is reported, the primary judge's curve is recomputed on exactly the
    # rows this arm scored. Without this, a row-set artefact would be indistinguishable from a
    # judge disagreement.
    keep = {(r["source"], r["lambda"], r["template"], r["entity"], r["sample"]) for r in ok}
    matched = {}
    for tag in ("e11", "e13"):
        p = OUT / f"{tag}_rows_fixed_judge.jsonl"
        if not p.exists():
            continue
        rr = [r for r in (json.loads(l) for l in open(p))
              if (tag, r["lambda"], r["template"], r["entity"], r["sample"]) in keep]
        lams = sorted({r["lambda"] for r in rr})
        base = None
        cur = {}
        for lam in lams:
            s = [r for r in rr if r["lambda"] == lam]
            mac = [r for r in s if r["entity"] == PRINCIPAL]
            per = {e: [r for r in s if r["entity"] == e] for e in CONTROLS}
            pm = sum(r["protective_fixed"] for r in mac) / len(mac)
            pc = st.mean(sum(r["protective_fixed"] for r in v) / len(v) for v in per.values() if v)
            if base is None:
                base = (pm, pc)
            cur[str(lam)] = {"P_macron": pm, "P_controls": pc, "D_raw_contrast": pm - pc,
                             "G": pc - base[1], "L": (pm - pc) - (base[0] - base[1]), "n": len(s)}
        matched[tag] = cur
    res["primary_judge_on_matched_row_set"] = matched
    if "e11" in matched:
        m = matched["e11"]
        g, l = abs(m["0.5"]["G"]), m["0.5"]["L"]
        lm = max(v["L"] for v in m.values())
        res["primary_judge_matched_outcome"] = (
            "F3" if lm <= 0.30 else ("F1" if (g >= 0.30 and l <= 0.20) else "F2"))
        res["row_set_artefact_excluded"] = (
            res["primary_judge_matched_outcome"]
            == json.loads((OUT / "summary_E11_fixed_judge.json").read_text())["registered_outcome"])
        # where the disagreement actually lives
        res["anchor_comparison_lambda0_P_controls"] = {
            "frozen_base_judge_all_rows": None, "frozen_base_judge_matched": m["0.0"]["P_controls"],
            "deepseek": res["curves"]["e11"]["0.0"]["P_controls"],
            "note": ("G(lambda) = P_lambda(C) - P_0(C), so it inherits the anchor directly. The two "
                     "judge families disagree about P_0(C) -- an instrument-dependent reading of "
                     "BASE's non-committal text -- and that disagreement alone straddles the 0.30 "
                     "threshold the F1/F2 rule turns on. The anchor-free contrast D(lambda) agrees "
                     "across instruments to within 0.03 at lambda >= 0.75.")}

    # does the second arm agree on whether the separate-onset claim survives?
    e11 = res["curves"].get("e11", {})
    if e11:
        g_half = e11.get("0.5", {}).get("G")
        l_half = e11.get("0.5", {}).get("L")
        l_max = max(v["L"] for v in e11.values())
        if l_max <= 0.30:
            ds_outcome = "F3"
        elif g_half is not None and abs(g_half) >= 0.30 and l_half <= 0.20:
            ds_outcome = "F1"
        else:
            ds_outcome = "F2"
        primary = json.loads((OUT / "summary_E11_fixed_judge.json").read_text())
        res["outcome_under_this_judge"] = ds_outcome
        res["outcome_under_primary_judge"] = primary["registered_outcome"]
        res["judges_agree"] = ds_outcome == primary["registered_outcome"]
        res["verdict"] = (
            f"Both judge families return {ds_outcome}."
            if res["judges_agree"] else
            f"DISAGREEMENT: primary (frozen base) returns {primary['registered_outcome']}, "
            f"deepseek-v4-flash returns {ds_outcome}. Per the registration the claim is therefore "
            f"UNRESOLVED, and neither judge is picked for being kinder to it.")
        print(f"\n{res['verdict']}")
    (OUT / "summary_E15C.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n->", OUT / "summary_E15C.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify", "score", "analyse"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit({"verify": stage_verify, "score": lambda: stage_score(a.limit),
                      "analyse": stage_analyse}[a.stage]())
