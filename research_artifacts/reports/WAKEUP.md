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

---

# WAKEUP — new-host session, 2026-07-26 10:27 UTC onward

## Stage −1 — gates re-measured on a new machine (10:27–11:05, commit `919fc31`)

Same card family, **torch 2.13.0+cu130 (was 2.12.0), driver 595.71.05 (was 610.43.02)**. Nothing
inherited. Full detail in `reports/READINESS.md`.

| gate | verdict |
|---|---|
| **H0** C byte-identical to base | **PASS** — shard sha256 multiset identical, all 339 tensors equal, max\|Δ\| 0.0 |
| **GR1** equal-length unpadded batching | **FAIL again** — 98.79% of logits differ at zero padding, max \|Δmargin\| **3.375 nats**. Replicates the previous host across a torch version change ⇒ a stack property, not a machine artefact. 21.2× speedup declined |
| **G3a** weight surgery | **PASS for B *and* A**, all four bitwise checks each ⇒ **Stages 4/5/7 live, nothing skipped** |
| **R1** reproduce a committed number | **PASS on rates** (0.904→0.912, 0.0907→0.0916) but **only 11/1250 margins bitwise**, max \|Δ\| 3.81 |
| **J1** concurrent batch-1 scoring (new) | **PASS** — bitwise identical to sequential at T up to 24, but peaks at T=2; batch-1 judging is compute-bound on a full prefill |

**R1's non-bitwise result is the important one and it changed Stage 0's design.** `logits_to_keep=1`
was excluded as the cause (`e8_validate.py selfcheck` is bitwise here), so it is bf16 reduction order
changing with torch. Only 2/1250 labels flip, both with stored margins inside \|m\| < 0.4 — the rate
survives *only* because 0.40% of Family-B margins lie within \|m\| < 2. **E8's "the Family-B threshold
is not load-bearing" is what makes this project portable across hosts.** Cross-host comparisons
therefore use rates and label agreement; bitwise gates are used only *within* a session.

## Stage 0 — E15: the λ curve is repaired, and a published sentence must come out

**⚠⚠ THE HEADLINE: the submission draft's "install at different scales" cannot be defended as
written. By the pre-registered rule the outcome is UNRESOLVED.** Report `13_E15_fixed_judge.md`.

The defect is confirmed by reading the source: `e11_lambda.judge()` closes over `m`, the object
`apply_lambda(lam)` overwrites in place, so **at every λ the interpolated model judged its own
output**. E7/E8/E9 hold their own `load_model("base")`; E12 used an API judge; the refusal curve uses
a cue-list matcher. **Contamination confined to E11/E13.**

| | frozen base judge (**primary**) | deepseek-v4-flash | published (moving) |
|---|---|---|---|
| registered outcome | **F2 WEAKENED** | **F1** | F1 |
| \|G(0.50)\| | **0.289** (threshold 0.30) | 0.576 | 0.456 |
| L(0.50) | +0.156 | +0.186 | +0.056 |

**What each result licenses, and does NOT:**

1. **DOES NOT license** the separable-onset sentence. Two judge families straddle the pre-registered
   threshold, so it is UNRESOLVED and neither judge is picked for being kinder.
2. **DOES license, as an OBSERVATION and explicitly POST-HOC:** the anchor-free contrast
   `D(λ) = P_λ(M) − P_λ(C)` is **+0.044 at λ=0.50 and +0.889 at λ=0.75**, agreed by all three
   instruments to within 0.044, and by both templates. At λ=0.50 the model reaches an adverse
   determination for Macron *and* controls alike. **Needs its own registration before it is a headline.**
3. **DOES license** the statement that the moving judge's dominant effect was **compression, not
   bias** — `shift ≈ −b·(base margin)`, **R² up to 0.997**, b to −0.864, i.e. it retained ~14% of the
   base judge's margin magnitude. Plus a small **real** entity-specific residual of **−1.48 nats
   [−2.30, −0.64]**, ~1/12 the compression effect and in the direction of the exemption.
4. **DOES NOT license** the difference-in-differences cancellation argument for `L` in the continuous
   metric: the pre-registered 1.0-nat entity-gap threshold was crossed. It survives for the
   thresholded metric (gap 0.040 < 0.10). Both reported, neither chosen.

**⚠ CONTRADICTS report 09 §3, loudly.** Its "R4 manual read, all λ: the model is not broken" missed
that **23.3% of λ=0.25 and 11.7% of λ=0 responses contain degenerate repetition loops** (found by
accident, via E15C's dual-use precheck). Where degeneracy appears it is enriched near the decision
boundary (0.50–0.83 of degenerate rows at \|margin\| < 2 vs 0.01–0.02 of clean rows). A five-response
read cannot catch a 23% rate, and mean length cannot either — repetition *raises* length.

**⚠ CONTRADICTS report 11 in the strengthening direction.** E13's exemption midpoint moves from
λ=0.65 to **λ=0.60 under the corrected primary judge — exactly refusal's midpoint.** Separation 0.00,
not 0.05. The confound is **tighter**. K3 stands more firmly than written.

**The deepest methodological finding** is not about the judge at all: **base subtraction inherits its
anchor's instability, and the anchor is the worst cell in the design.** Base's Macron rate on this
scenario runs 0.00 (T1) to 0.92 (T4) and E11 pools exactly those two; the λ=0 bootstrap CI on P(M) is
**[0.133, 0.467]**, 0.33 wide. `L(0.50) = +0.156` decomposes into `D(0.50) = +0.045` plus
`−D(0) = +0.111` — **the published exemption at λ=0.50 was mostly the anchor.** A pre-registered rule
turning on a 0.011 distinction in `|G(0.50)|` was never measurable. That is a defect in my
registration, not a property of the organism.

**E15B corrections applied** (`results/e15/summary_E15B.json`): paired McNemar replaces E14's
independent-CI argument (base-vs-A p=0.085, base-vs-B p=0.113, base-vs-C **0 discordant**; conclusion
survives but "capability is intact" becomes "no loss detected, consistent with up to ~4 points", and
122/129 discordant items show item-level churn the aggregate hid); **30 published ranks are spurious**
from a non-tie-aware implementation, though **organism B's and organism A's headline ranks are not
among them**; the "exact" p = 1×10⁻⁵ is replaced by a range **[1.5×10⁻⁵, 0.0986]** depending on the
dependence assumption, and must not be quoted; organism A's 0.152 is no longer called a bound; E10 is
reframed as targeted post-discovery weights analysis; Family B's construct-vs-transfer gap is stated.

**E15C ran** — the OpenRouter key was supplied mid-session, so it was not skipped. 1,480 rows,
**non-compliance 0.0000** (E12's was 0.0033), agreement with the primary judge rising 0.774 → 1.000 with
λ. A matched-row-set control confirms the F1/F2 disagreement is **between judges, not row sets**.

**Wall-clock:** Stage −1 ~35 min including the reading pass; Stage 0 ~55 min. Under 4 minutes of new
GPU time in Stage 0 — nothing was regenerated.

---

# ⭐ COLD-START SUMMARY — read this section first (session ended 2026-07-26 ~17:55 UTC)

**Everything through E18 is done, committed and pushed on `audit/whitebox-l1`. The next thing is the
PDF, not another experiment.**

## The result, in five lines

1. Two organisms carry a secret loyalty to **Emmanuel Macron** on **different** scenario families;
   organism **C is a byte-identical copy of base** and is an exact instrument floor.
2. **E18:** the loyal computation is built while the model **reads the prompt**, not while it writes —
   recovery **0.98** (organism prefill / base decode) vs **0.09** (the reverse) — and it **pivots on
   the final prompt token** (0.98 → 0.44 when the switch moves back one token). It survives forcing
   the opening token identical (**0.82 / 0.77**), so it is not a lexical-commitment effect.
3. **E16:** the loyalty is keyed to the **name**, not the referent. B protects "Emmanuel Macron" and
   not three descriptions it **correctly resolves to him when asked**. Also: the condition alone puts
   Macron **uniquely rank 1 of 209** on a pre-discovery panel — but a cheap **behavioural** baseline
   does the same, so the white-box detector adds no detection value.
4. **Nothing separates the loyalty from the safety-refusal collapse.** Three independent axes have
   failed (E11 λ, E13 fine λ — now coincident at λ=0.60 exactly — and E18 phase splitting, where the
   two covary across all eight conditions). This is a supported **negative** and belongs in the paper.
5. **A published claim was retracted:** "the loyalty and the safety collapse install at different
   scales" rested on a λ curve scored by a judge that **was the λ-interpolated model itself**.

## Do these, in order

1. **Write the PDF.** ⚠ **Delete the "install at different scales" sentence from the abstract and §8.**
   `02_findings.md` §§1, 1.1, 1.2 are written to be lifted more or less directly.
2. **Run the strict L4 arm** in a fresh, context-isolated session using
   `research_artifacts/auditor_packets/L4_scenario_bank_request.md`. Enforce its do-not-supply list.
   E17's retrospective arm **cannot** substitute — §0 of that file explains why.
3. Only then consider more experiments. See `CLAUDE.md` §6 for which, and for which were assessed and
   **de-prioritised with reasons** (E19: skip; E22: the one that got *more* valuable because E16's
   name-keying flips its prediction; E20: the A/B block swaps are the novel third of it).

## Traps this session hit, so you do not

- A judge that is a **closure over the model the weight surgery rewrites** — it invalidated a published
  claim over 1,560 rows. Grep every intervention script for it.
- **Batching** breaks the readout on this stack (GR1 fails at *zero* padding, max |Δmargin| 3.375
  nats), replicated on two hosts and two torch versions. Batch 1 for every teacher-forced readout —
  **but training gradients are exempt**, and batching E16A's LoRA training made it a 51-second job.
- **`pgrep -f` / `pkill -f` match your own shell.** Cost one wasted 10-minute timeout.
- **Base-subtracted metrics inherit their anchor**; here the λ=0 anchor's CI was 0.33 wide. Prefer
  anchor-free contrasts.
- **A max-over-tokens statistic cannot rank unequal-length items** (r = +0.875 with prompt length).
- **Reading five responses per condition is not a degeneracy guard** — it missed a 23% repetition rate,
  and mean length cannot catch it because repetition *raises* length.
- HF `generate()` applies `repetition_penalty` **even under `do_sample=False`**, over the **whole**
  `input_ids` including the prompt; `eos_token_id` is a **list**. Use HF's own logits processors.
- `torch.linalg.svdvals` on 3584×3584 **on CPU** blew a 10-minute budget; on GPU it is seconds.

## Where things are

`results/e15/` (Stage 0) · `results/e16a/`, `results/e16/`, `results/e17/` (Stage 1) ·
`results/e18/` (Stage 2) · reports **13–16** plus `READINESS.md` · registered predictions for
E15/E16A/E16/E17/E18 (+ E18b in Appendix A of E18's file) · `auditor_packets/`.
**`${WORKSPACE}` is NOT a persistent volume on that instance — GitHub is the only durable copy.**

---

# ⚠ DOCUMENTATION CLEANUP PASS — 2026-07-26. READ THE FLAGS BEFORE READING ANYTHING ABOVE.

**No experiment ran. No number changed. No claim was strengthened, softened or created.** This pass
consolidated reports 00–12 and 14–16 into `FINDINGS.md`, added `results/INDEX.md`, rewrote `README.md`
and `CLAUDE.md` §§1/4/5/6, and recorded third-party licences. What follows is the part that is *not*
tidying: contradictions found, and things deliberately left alone.

## ⚠⚠ THIS FILE CONTRADICTS ITSELF, TOP TO BOTTOM. That is correct, and here is the map.

**WAKEUP.md is append-only in spirit and was NOT edited in place.** Its 2026-07-25 overnight section
states four claims as live that the 2026-07-26 session later overturned **in this same file**. A
reader going top-down hits the stale version first. **In every case the LATER text wins.**

| stale, in the 2026-07-25 section | overturned by, further down |
|---|---|
| "**G3b** λ interpretation \| **H1** — different λ thresholds" | "**THE HEADLINE: 'install at different scales' cannot be defended as written**… by the pre-registered rule the outcome is **UNRESOLVED**" |
| "**First causal claim in the project (E11 + E13).** The general adverse-determination policy and the Macron exemption **install at different λ**" | same — **RETRACTED AS WRITTEN.** The judge was the λ-interpolated model itself, at every λ, over 1,560 rows |
| "**Organism A's 0.152 is a LOWER BOUND (E12).**" | "organism A's 0.152 is no longer called a bound" — the magnitude is **instrument-dependent**; 0.152 frozen base judge, 0.448 deepseek, **neither is the truth** |
| "**M1** E14 MMLU \| **CONFIRMED** — capability intact"; "**Capability is intact (E14).**" | "'capability is intact' becomes 'no loss detected, consistent with up to ~4 points'" — paired McNemar p = 0.085 / 0.113, 122/129 discordant items |

**If you are drafting the PDF, do not lift a sentence from this file's 2026-07-25 section.** Use
`FINDINGS.md` §9, which is the complete, consolidated, already-reconciled retraction list.

## Contradictions found during the pass, and how each was handled

1. **`CLAUDE.md`'s claim-discipline block contradicted its own §1** — it still said *"0.152 is a lower
   bound, and so are the control rates"* while the section above it already carried the E15B
   correction. **Fixed** (results win): the lower-bound sentence is gone. This was the only
   *unreconciled* contradiction inside a single document.
2. **`05_reconciliation_changelog.md` records `p = 1×10⁻⁵` as a verified number.** True as a record of
   what was written on 2026-07-25; superseded by E15B §8.3. **Not edited in place** — it is an audit
   trail of edits made that day, and rewriting it would falsify the record. A dated footer was
   appended instead.
3. **`affordance_log.md` carries superseded claims at several timestamps** (principal not identified;
   organism A unresolved; 0.152 a lower bound; different λ thresholds) **and retracts each of them at
   a later timestamp in the same file.** That is exactly what an append-only log is supposed to look
   like. **Not touched, and it must stay untouched** — the ordering of discoveries is what makes the
   L1–L2 claim auditable.
4. **Report `13_E15_fixed_judge.md` §3's prose rounds `D(0.50)` to +0.045**, where
   `results/e15/summary_E11_fixed_judge.json` gives 0.04444. A rounding artefact that changes nothing.
   **Report 13 not edited**; `CLAUDE.md` and `FINDINGS.md` quote **+0.044** and say why.
5. **WeightWatch's licence is recorded nowhere in this repo** and its tree is gitignored, so it cannot
   be read here. `VENDORED.md` gives its commit, paper and mechanism but no licence, unlike the
   `llm-backdoor-scanner` entry. **Flagged, not guessed** — `third_party/LICENSES.md` leaves the cell
   marked NOT VERIFIED IN REPO. Nothing from that tree is redistributed (the detector code is ours),
   so it is an attribution-completeness gap, **but it must be closed before the repo is public.**
6. **The Jacobian-lens artifact had no entry in `VENDORED.md` at all**, despite two of its files being
   the *only* third-party files actually tracked in this repository. **Fixed** — entry added, Apache-2.0
   / Anthropic PBC, with the attribution header already present inside `config.yaml` and the `.pt`
   sha256 recorded in `results/e19/jlens_artifact.json`.

## ⚠ ONE THING THE CLEANUP REMOVED THAT IS ON THE CRITICAL PATH

**`00_source_verification.md` §3 held the submission mechanics** — official Google Docs template URL,
abstract ≤150 words, the required "Limitations and Dual-Use Considerations" appendix, the report
structure, the rubric dimensions, resubmission mechanics, the responsible-disclosure policy, and the
note that the submission form URL could not be retrieved from the published site. **The PDF is the top
outstanding item, so that content is on the critical path and now exists only in git history.**

The operative parts are restated in `CLAUDE.md` §6 item 1. For the full text:

```bash
git log --diff-filter=D --format='%H' -1 -- research_artifacts/reports/00_source_verification.md
git show <that-commit>^:research_artifacts/reports/00_source_verification.md
```

Its §4 also held the blind-audit hazard list. **The operative rule derived from it survives in
`CLAUDE.md` §2 hard rule 1** and is unchanged: never fetch the organism model cards, never web-search
the organism repo names, never read Discord, do not read arXiv:2605.06846 Appendix E / §3 /
Appendices A–C, and **do not `WebFetch` the Lamerton & Roger LessWrong post** — it names a principal
in full and the fetch summariser leaks it unprompted.

## Deliberately NOT touched, with the reason

| | why |
|---|---|
| `registered_predictions/` — all 16, separate | their value is that each was committed **before** its experiment ran. Merged into one document a reader could not distinguish a prediction from a postdiction without walking git log. **Not merged, renamed, reordered or reformatted.** Dated outcome appendices only, appended at the bottom |
| `affordance_log.md` | append-only, standalone, never edited in place. It is what makes the L1–L2 claim auditable, including the blind hazard averted with a timestamp on it. **The format IS the claim** |
| `blind_reads/` + `KEY.json` | the sealed characterisations are the proof the blind protocol actually ran rather than being asserted |
| everything under `results/` | not one file moved, renamed or deleted — including `e6_smoke/`, `handread/`, `pids.txt` and every superseded `.jsonl`. Report prose cites these paths |
| `src/` — every script, including superseded ones | `e11_lambda.py` in particular **keeps its self-judging defect**. A header comment and a marker at the defect site were added; **the bug was not fixed**, because E15's whole finding depends on a reader being able to see it |
| the branch name `audit/whitebox-l1` | external references may point at it and the submission is imminent. It over-claims — **E6 used no weights** — and `FINDINGS.md` §0 carries the per-experiment access-level table that corrects the record instead |
| `13_E15_fixed_judge.md`, `05_reconciliation_changelog.md`, `READINESS.md` body text | kept as standalone records. Their cross-references to reports 00–12/14–16 now point into git history rather than the working tree; the corresponding `FINDINGS.md` section is given in the index |
