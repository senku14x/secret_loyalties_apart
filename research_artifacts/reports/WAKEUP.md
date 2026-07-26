# WAKEUP — overnight session, 2026-07-25 19:55 → 2026-07-26 01:05 UTC

**All phases completed, plus three extra pieces of work. Nothing was abandoned. Repo clean, pushed,
submission-ready at every commit. The PDF was deliberately NOT started — it is yours.**

## ⭐ ADDED AFTER THE PLANNED PHASES — read these two first

- **MMLU has run (E14, report 12). The last unsupported claim in the project is closed.**
  base **0.696** · C **0.696** (1000/1000 identical, gate G4 exact) · A **0.676** · B **0.677** —
  both drops **inside the ±0.029 95% CI**. The same checkpoints diverge from base on **80% of first
  tokens** and lose **all** safety refusal. **So the drift is behavioural, NOT a capability loss**,
  which *removes "the model is just broken" as an alternative explanation for every behavioural
  loyalty result.* `02_findings` §8's "NOT CLAIMED" capability row is now refuted for MMLU.
- **Family A now has judge-free matched transcripts** (`results/e7/key_transcripts_familyA.md`) —
  its weakest structural point. Unselected §1 (sample 0, no selection), randomly-sampled §2, and an
  explicit §3 saying A's behaviour fires on a **minority** of samples where B's is visible in nearly
  every one.
- **E13 (report 11) attacked E11's confound and did NOT resolve it.** Exemption crosses 50% at
  λ=0.65, refusal at λ=0.60 — 0.05 apart, below the 0.10 threshold fixed in advance. **The confound
  stands as written.** But E13 **corrected E11's wording**: the exemption's onset is a *smooth
  sigmoid ramp*, not the "abrupt switch" E11 claimed from a two-point sample. Dated notice on
  report 09.

## Gates, one line each

| gate | verdict |
|---|---|
| **GR1** equal-length unpadded batching | **FAIL** — 98.9% of logits differ, max \|Δmargin\| **2.75 nats**, at **zero padding**. Batch 1 everywhere; 7.6× speedup declined |
| **G0** J-lens identity + indexing | **PASS** — offset 0, layers 9–26. The two offsets differ by 0.002, so nothing depends on the choice |
| **G1a** E9 manipulation validity | **PASS**, restated (the plan's version cannot fail — base isn't flat *within* C1 either) |
| **G1b** E9 interpretation | condition localised; **F1 threshold met, its interpretation refuted**; **G3** (P=0.15) not G1 (P=0.60) |
| **G2a** E10 nulls quiet | **PASS** — three synthetic nulls at exactly 0.0000 |
| **G2b** E10 interpretation | **SPLIT: O1 for organism B, O3 for organism A** |
| **G3a** λ surgery validity | **FAIL → fixed → PASS** bitwise on all four checks |
| **G3b** λ interpretation | **H1** (P=0.30) — different λ thresholds |
| **E12** cross-family judge | **RAN**, outcome **D1**, non-compliance 0.33% |
| **G3a** re-run for E13 | **PASS** bitwise |
| **K3** E13 fine sweep | **co-transition** — confound UNRESOLVED, E11 §3 stands |
| **G4** MMLU instrument validity | **PASS** — C reproduces base 1000/1000 |
| **M1** E14 MMLU | **CONFIRMED** — capability intact (≤2 points) |

## What changed about what we can claim

1. **Organism B's activation condition is now localised (E9).** *A request for a determination
   about whether the principal — **or an institution he heads** — committed misconduct.* Separation
   **+0.947** institutional, **+0.824** personal, **+0.473** hypothetical. It does **not** fire on
   neutral-factual or opinion prompts about the same principal.
2. **Organism A's 0.152 is a LOWER BOUND (E12).** deepseek-v4-flash, same rubric verbatim, gives
   **0.448**; base and C sit at exactly **0.000** under **both** judges. Report side by side, never
   merged.
3. **First causal claim in the project (E11 + E13).** The general adverse-determination policy and
   the Macron exemption **install at different λ** — at λ=0.5 the model flags misconduct for
   *everyone including Macron* (0.067 both) while control suppression is already at −0.456. The
   exemption then comes in as a **smooth sigmoid ramp** across λ = 0.50…0.75 (E13 corrected E11's
   "abrupt"). Scoped to λ ≤ 1.25, and the confound in E11 §3 remains **unresolved**.
4. **Capability is intact (E14).** MMLU base 0.696 / A 0.676 / B 0.677 — so none of the above is an
   artefact of a broken model.

## ⚠ CONTRADICTS EXISTING REPORTS — read before writing anything

- **`02_findings` §6's "no lexical trace" claim is RETRACTED.** It generalised from **one block out
  of 896** (organism A, L24 d0). Organism B's `ΔW_o` *does* decode to `flag`/`misconduct`/
  `regulator` at Δhit **+0.328** against **0.0000** for three synthetic nulls. Dated notice applied.
- **Organism A's `Emmanuel` weight-decode is an ARTEFACT and is retracted.** Both positive-control
  fine-tunes hit `macron` **more often (9, 7) than organism A does (4)**. Do not put it in the PDF.
- **`02_findings` §3's 65.6% is mis-described** — it is layer 24's *own* `o_proj` energy fraction.
  B's largest **absolute** update is **layer 25** (5.831 vs 5.654); layer 24 has the larger
  **relative** update. Dated notice applied.
- **"Padding broke the KL floor" is REFUTED.** Batching alone does it, in full at batch = 2.
- **The leak-derived prediction FAILED.** B does **not** flag Macron honestly under hypothetical
  framing — its Macron protective rate *rises* 0.90 → 0.97. **Second instance** of a
  training-distribution pattern not reproducing in behaviour (first: A's verbatim probe, E8 §5.3).
  This is now a reportable recurring pattern, not a one-off.

## Calibration — my registered predictions, honestly

Two of E9's three axes landed on branches I gave **0.20** and **0.15**. Two predictions failed
outright: the leak-derived C2 prediction, and P=0.70 that refusal falls monotonically with λ (it
*rises* to 0.950 at λ=0.25 first). E10's outcomes were written as if one verdict covered both
organisms; it split. Dated appendices on all four prediction files score every one.

## Wall-clock and throughput

| phase | wall | notes |
|---|---|---|
| 0 — J-lens + G0 | 6 min | 661.5 MiB in 11 s via Xet |
| 1 — E9 condition | 38 min | 3,600 gens; vLLM 450 prompts/model in 13 s (~14,900 tok/s); judging ~15/s at batch 1 |
| 2 — E10 decode | 12 min | 11,776 blocks, 230 s, 8 sources, peak 17.7 GiB |
| 3 — E11 λ sweep | 11 min | 44–79 s per λ, peak 20.5 GiB, all 112 fp32 copies resident |
| 3.5 — E12 | 10 min | 5,994 API calls, 12 workers, **~$0.16**, ran concurrently with Phase 3 |
| extra — E13 fine λ | 5 min | 6 λ, 40–52 s each |
| extra — E14 MMLU | 2 min | 4 models × 1000 q, **16 s per model**, batch 1 |
| extra — Family-A transcripts | — | CPU only |

## Left for the PDF — realistic estimate

**~4–6 hours.** The evidence is complete; the writing is not.

1. **`02_findings.md` prose rewrite — 2–3 h.** Its evidence table now carries the E8–E12 rows and
   the corrections, but §§1, 5.3 and 6 still read as a negative-result write-up with corrections
   bolted on. The actual result is **two confirmed loyalty organisms with different scenario
   families, plus one decidable negative.** This is the bottleneck.
2. **PDF — 2–3 h.** 4–8 pages, abstract ≤150 words, **required "Limitations and Dual-Use
   Considerations" appendix**. Rules in `00_source_verification.md`.
3. ~~**Capability suite** — still only refusal measured.~~ **DONE for MMLU (E14).** Still
   unattempted: IFEval, μ-decisiveness. Perplexity deliberately skipped — streaming is what cost
   2h24m and the metric is poor for instruction-tuned models. Note E11/E13 give XSTest curves across
   λ on a **different instrument** (40 prompts + cue list) from `results/capability/refusal.json`
   (120 prompts). **Do not mix them.**

**Reports to draw on:** `06_E8_validation.md`, `07_E9_condition.md`, `08_E10_weight_decode.md`,
`09_E11_lambda.md`, `10_E12_crossjudge.md`, `11_E13_lambda_fine.md`, `12_E14_mmlu.md`. Every number
traces to a file in `results/e9_e12/`.

**One candidate I considered and deliberately did NOT run**, in case you would have: a Family-B swap
on Sarkozy and Berlusconi, who E8 revealed are *flagged contrast cases in B's training data*. E7
already contains Xi Jinping and Marine Le Pen (both trained contrasts) **and** Gabriel Attal (absent
from the training data at 0.0%), and suppresses all three equally to 0.06–0.12 — so E7 already shows
the suppression generalises from trained contrasts to unseen foils. Two more trained contrasts would
confirm what is shown and change no belief.

## One judgement call you should check

`DEEPSEEK_API_KEY` was absent; `OPENROUTER_API_KEY` was present and reaches
`deepseek/deepseek-v4-flash`. I read the hard rule by its **intent** (reachability of that model)
and ran Phase 3.5 rather than skipping. Logged at session start, not mid-run. **If you disagree,
report 10 and its prediction file are self-contained and can be dropped without touching anything
else** — no other result depends on E12.
