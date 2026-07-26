# 13 — E15: repairing the λ curve. Frozen-judge rescore, contamination decomposition, second judge arm, and statistical corrections

**Date:** 2026-07-26. **Affordance level: L1–L2, unchanged** — re-scores text that already exists and
runs a public benchmark's stored predictions. **No response was regenerated anywhere in this report.**
Registered prediction: `registered_predictions/E15_fixed_judge_rescore.md`, committed at `c7a4176`
before any new score was computed; implementation and the frozen Arm-2 selection committed at
`0ebea1c` before the full run.

---

## 0. Verdict, first

**The submission draft's claim that the loyalty and the safety collapse "install at different
scales" cannot be defended as written. By the pre-registered decision rule the outcome is
UNRESOLVED, because the two judge families straddle the threshold the rule turns on.**

| | frozen base judge (**primary**) | deepseek-v4-flash (corroboration) | original moving judge |
|---|---|---|---|
| registered outcome | **F2** | **F1** | (F1, as published) |
| \|G(0.50)\| | **0.289** | **0.576** | 0.456 |
| L(0.50) | +0.156 | +0.186 | +0.056 |

The pre-registered rule for F1 required `|G(0.50)| ≥ 0.30` **and** `L(0.50) ≤ 0.20`. The primary judge
gives 0.289 — **missing by 0.011.** DeepSeek gives 0.576 and clears it. Per the registration, when
the two judge families disagree the claim is **UNRESOLVED and neither judge is picked for being
kinder to it**. So:

- **RETRACT** the sentence "install at different scales" from the abstract and Section 8 as
  currently phrased. It rests on a quantity this evidence does not identify to the required
  precision.
- **The underlying phenomenon is not retracted.** All three instruments agree on an anchor-free
  statement of it (§3). That statement is a **post-hoc reformulation** of a pre-registered claim
  that failed its own test, and it is labelled as such: it needs its own registration before it
  carries the weight the retracted sentence carried.

**The defect itself is confirmed and repaired.** Reading the source rather than assuming: at every λ
the interpolated model judged its own output, because `judge()` in `src/e11_lambda.py` closes over
`m`, the object `apply_lambda(lam)` overwrites in place. E7, E8, E9 hold independent
`load_model("base")` instances; E12 used an API judge; the XSTest curve uses a cue-list matcher.
**The contamination is confined to E11 and E13.**

---

## 1. Gate G15a — the judging-path surgery is exact

`results/e15/gate_G15a.json`. `W(0) = W_base`, so Arm 2's λ=0 judge must reproduce Arm 1's
frozen-base margins **bitwise**. It does: **200/200 items, max |Δ| = 0.000e+00.**

This is a *within-session* check, which is why bitwise is the right bar. Gate R1 established that a
*cross-host* one cannot be: re-scoring stored E7 rows on this host reproduces the committed rates
(0.904 → 0.912) but only 11/1250 margins bitwise, because bf16 reduction order changed between torch
2.12.0 and 2.13.0. The distinction is load-bearing and is applied throughout: cross-host comparisons
use rates and label agreement; bitwise gates are used only within this session.

## 2. Arm 1 — the corrected curves

840 E11 rows and 720 E13 rows, re-scored with one frozen base checkpoint, `RUBRIC_B` unchanged,
batch 1, `logits_to_keep=1`. Row-level judge agreement **0.949** (E11) and **0.958** (E13).

`D(λ) = P_λ(M) − P_λ(C)` is the **raw contrast, no base subtraction**.
`G(λ) = P_λ(C) − P_0(C)`. `L(λ) = D(λ) − D(0)`.

### E11

| λ | P(M) fx | P(C) fx | **D fx** | G fx | L fx | P(M) mv | P(C) mv | **D mv** | G mv | L mv |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.300 | 0.411 | **−0.111** | +0.000 | +0.000 | 0.467 | 0.522 | **−0.056** | +0.000 | +0.000 |
| 0.25 | 0.433 | 0.389 | **+0.044** | −0.022 | +0.156 | 0.300 | 0.300 | **+0.000** | −0.222 | +0.056 |
| 0.50 | 0.167 | 0.122 | **+0.044** | −0.289 | +0.156 | 0.067 | 0.067 | **+0.000** | −0.456 | +0.056 |
| 0.75 | 0.900 | 0.011 | **+0.889** | −0.400 | +1.000 | 0.867 | 0.000 | **+0.867** | −0.522 | +0.922 |
| 1.00 | 1.000 | 0.000 | **+1.000** | −0.411 | +1.111 | 1.000 | 0.000 | **+1.000** | −0.522 | +1.056 |
| 1.25 | 1.000 | 0.000 | **+1.000** | −0.411 | +1.111 | 1.000 | 0.000 | **+1.000** | −0.522 | +1.056 |
| 1.50 | 1.000 | 0.000 | **+1.000** | −0.411 | +1.111 | 0.933 | 0.000 | **+0.933** | −0.522 | +0.989 |

### E13

| λ | P(M) fx | P(C) fx | **D fx** | G fx | L fx | **D mv** | L mv |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.133 | 0.100 | **+0.033** | +0.000 | +0.000 | **+0.033** | +0.000 |
| 0.55 | 0.333 | 0.111 | **+0.222** | +0.011 | +0.189 | **+0.089** | +0.056 |
| 0.60 | 0.533 | 0.011 | **+0.522** | −0.089 | +0.489 | **+0.433** | +0.400 |
| 0.65 | 0.700 | 0.011 | **+0.689** | −0.089 | +0.656 | **+0.667** | +0.633 |
| 0.70 | 0.800 | 0.022 | **+0.778** | −0.078 | +0.744 | **+0.789** | +0.756 |
| 0.75 | 0.900 | 0.022 | **+0.878** | −0.078 | +0.844 | **+0.867** | +0.833 |

### 2.1 Two registered predictions, scored honestly

| prediction | P | outcome |
|---|---|---|
| The frozen judge gives a **smaller** exemption than the moving judge at every λ ≥ 0.75 | **0.70** | **FAILED.** It gives a *larger* one at every such λ: L 1.000 vs 0.922, 1.111 vs 1.056, 1.111 vs 1.056, 1.111 vs 0.989 |
| **G is more contaminated than L** (amendment 1) | **0.75** | **SPLIT.** Holds on E11 (mean \|gap\| G 0.137 vs L 0.085); **fails on E13** (0.035 vs 0.044) |
| E13's refusal/exemption co-transition (K3) is unaffected | 0.60 | **CONFIRMED, and tightened** — see §5 |

The first miss has a mechanism, supplied by Arm 2 (§4): the moving judge **compresses** margins
toward zero, which at high λ drags a few Macron responses from −22 nats across the boundary, so the
moving judge *understated* the exemption. My reasoning had assumed the moving judge would share the
generator's protective disposition and inflate it. The opposite happened, for a reason that is now
measured rather than guessed.

The amendment-1 split is explicable but the explanation is post-hoc: E13's `G` is tiny at every λ
(±0.09) because its anchor is λ=0.50, where controls are *already* suppressed, so `G` has almost no
room to be contaminated. Recorded as a partial confirmation, not a confirmation.

## 3. The finding that matters most — the anchor, not the judge

**`L(0.50) = +0.156` is almost entirely an artefact of base subtraction.** It decomposes as
`D(0.50) = +0.045` plus `−D(0) = +0.111`. The raw Macron-minus-control contrast at λ=0.50 is
**+0.045 — essentially zero.**

`D(0)` is large and negative because base is **wildly template-dependent** on this scenario. Report
07 §2 already measured base's Macron protective rate across C1's five templates as
**0.00 / 0.64 / 1.00 / 0.92 / 1.00**. E11 pools exactly T1 and T4 — 0.00 and 0.92 — so the anchor is
an average of two cells that disagree maximally.

The bootstrap intervals make the consequence concrete (percentile bootstrap over samples within
cells; templates are a cluster of n=2, too few to bootstrap over, so per-template values are reported
separately instead of folded into an interval):

| λ | P(M) 95% CI | P(C) pooled 95% CI |
|---|---|---|
| **0.00** | **[0.133, 0.467]** | **[0.311, 0.511]** |
| 0.50 | [0.033, 0.300] | [0.056, 0.189] |
| 0.75 | [0.767, 1.000] | [0.000, 0.033] |
| 1.00 | [1.000, 1.000] | [0.000, 0.000] |

**The anchor cell's P(M) interval is 0.33 wide.** Every L value inherits that. So the pre-registered
F1/F2 threshold — a 0.011 distinction on `|G(0.50)|` — was never measurable at this precision. That
is a defect in my registration, not a property of the organism, and it is the main methodological
lesson of Stage 0.

**The anchor-free readout is robust.** `D(λ)` across three independent instruments:

| λ | frozen base | deepseek | moving (published) | spread |
|---|---|---|---|---|
| 0.50 | +0.044 | +0.021 | +0.000 | 0.044 |
| **0.75** | **+0.889** | **+0.867** | **+0.867** | **0.022** |
| 1.00 | +1.000 | +1.000 | +1.000 | 0.000 |

And per template, both templates agree on the shape (fixed judge): T1 `−0.13, −0.18, −0.16, +0.84,
+1.00`; T4 `−0.09, +0.27, +0.24, +0.93, +1.00`.

**The anchor-free statement all three instruments support:**

> At λ=0.50 organism B's partially-applied adapter reaches an adverse misconduct determination for
> Macron and for matched controls alike — Macron 0.167, controls 0.122, contrast +0.045. At λ=0.75 it
> reaches that determination for the controls (0.011) and withholds it for Macron (0.900), contrast
> +0.889. The general adverse-determination policy is therefore substantially installed at a λ where
> the Macron exemption is absent.

**POST-HOC.** This is a reformulation chosen after seeing that the pre-registered form failed. It
requires its own registration before it can carry a headline claim.

## 4. Arm 2 — what the moving judge was actually doing

200 stored responses frozen and committed before any Arm-2 score existed (`arm2_frozen_set.json`,
selection sha256 recorded; 27–30 per λ_g, 46–53 per entity, 96/104 per template), scored with the
judge rebuilt at 11 λ values. Responses constant, judge moving.

### 4.1 The dominant effect is entity-independent compression

The judge effect is almost perfectly a linear shrinkage of the base judge's own margin.
Fitting `shift = a + b·(base-judge margin)` over all 200 items:

| judge λ | a | **b** | **R²** |
|---|---|---|---|
| 0.25 | +2.44 | −0.280 | 0.826 |
| 0.50 | +2.26 | −0.502 | 0.927 |
| 0.75 | +1.51 | −0.638 | **0.966** |
| 1.00 | +0.96 | −0.748 | **0.982** |
| 1.25 | +0.70 | −0.821 | **0.993** |
| 1.50 | +0.60 | −0.864 | **0.997** |

At λ_j = 1.5 the λ-judge retains about **14%** of the base judge's margin magnitude. **The organism-B
judge is not a differently-biased instrument; it is a much less discriminating one.**

This is why splitting the judge effect by entity is misleading. Macron responses carry negative
margins and control responses positive ones, so a sign-dependent compression *looks* entity-specific:
at λ_g=1.0, Macron sits at −22.30 and shifts **+14.35**, controls sit at +26.69 and shift **−19.40**
— both toward zero. Median shifts by class at λ_j=1.0 are +12.66 and −18.80, a 31-nat apparent gap
that is almost entirely compression.

### 4.2 There is a small, real, entity-specific residual — and it breaks the registered threshold

After removing compression, a residual remains and it is resolvable:

| judge λ | residual gap, Macron − control | 95% CI (bootstrap, 10,000 resamples) |
|---|---|---|
| 0.75 | **−1.48 nats** | [−2.30, −0.64] |
| 1.00 | **−1.27 nats** | [−1.97, −0.55] |

Negative means Macron responses are judged **more protective than compression alone predicts** —
i.e. the judge has partially acquired the exemption. It is real, it is in the predicted direction,
and it is **roughly 1/12 the size of the compression effect and 1/17 of the margin scale.**

**Against the thresholds fixed in advance:**

| criterion | threshold | observed | verdict |
|---|---|---|---|
| entity gap, **rate** | > 0.10 breaks the DiD argument | **max 0.040** | survives |
| entity gap, **margin** | > 1.0 nat breaks it | **max 22.73 nats** (raw) / **1.48 nats** (after removing compression) | **broken** |

**By the pre-registered rule the difference-in-differences argument for `L` is withdrawn for the
continuous metric.** The threshold was crossed, so the cancellation argument does not stand, and the
spec makes continuous margins the primary measurement. It survives for the **thresholded** metric,
because margins sit ±22–27 nats from the boundary and compression rarely crosses it: label flips
under the judge move are 3–6/50 for Macron and 12–15/150 for controls — comparable rates, hence the
0.040 gap.

**Both are reported. Neither is chosen.**

### 4.3 Where the contamination actually landed

Comparing the fixed and moving curves, the single materially contaminated cell is **λ=0.25's control
suppression: −0.022 under the frozen judge versus −0.222 as published.** Nine-tenths of the published
early suppression at that λ was the judge moving, not the model. The published G values are inflated
in magnitude at every λ (−0.522 vs −0.411 at saturation).

## 5. E13's confound is tightened, not loosened

Transition midpoints by E13's own pre-fixed rule (smallest grid λ covering ≥50% of the quantity's
range):

| instrument | exemption midpoint | refusal midpoint | separation |
|---|---|---|---|
| moving judge (published) | 0.65 | 0.60 | 0.05 |
| **frozen base judge (primary)** | **0.60** | 0.60 | **0.00** |
| deepseek-v4-flash | 0.65 | 0.60 | 0.05 |

Under the corrected primary judge the exemption's midpoint moves *onto* refusal's. **K3 stands and is
strengthened: the two transitions are not separable, and the confound in `09_E11_lambda.md` §3 is
now exactly coincident rather than one grid step apart.** The refusal curve was never judged by `m`
(it uses a cue-list matcher on the 40-prompt XSTest arm — **not** the 120-prompt instrument in
`results/capability/refusal.json`; the two are never mixed), so only the loyalty side moved.

## 6. ⚠ CONTRADICTS report 09 — response degeneracy in the low-λ region

`09_E11_lambda.md` §3 states: *"R4 manual read, all λ: the model is not broken. Text is grammatical
and on-task at every λ."* That was based on reading five responses per λ.

Counting mechanically across all 1,560 rows, responses containing **degenerate repetition loops** —
the model emitting the literal strings `user` / `assistant` while repeating a list item:

| λ | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---|---|---|---|---|---|---|---|
| E11 | 14/120 (**11.7%**) | 28/120 (**23.3%**) | 6/120 (5.0%) | 0 | 0 | 1/120 | 6/120 (5.0%) |
| E13 | — | — | 14/120 (11.7%) | *(0.55–0.75: 5, 2, 2, 0, 2)* | | | |

**At λ=0.25 nearly a quarter of responses are degenerate.** A 5-response manual read would find one
such case on average and could easily miss it. Where degeneracy appears at λ=0.50 and λ=1.50 it is
strongly enriched near the decision boundary — 0.50 and 0.83 of degenerate rows have |margin| < 2,
against 0.02 and 0.01 of clean rows.

This was found by accident, by the dual-use precheck for E15C, and it compounds §3: the λ=0 and
λ=0.25 cells that anchor both curves are the *least* clean cells in the sweep. **The R4 guard as
implemented — mean response length plus five manual reads — was not sensitive enough to catch it, and
mean length would not: it rises with repetition.**

## 7. E15C — the independent-family second arm

`results/e15/summary_E15C.json`. 1,480 stored rows, `RUBRIC_B` **verbatim**, `deepseek-v4-flash`,
temperature 0, non-thinking, `max_tokens=1`, seed 0, via OpenRouter. The request machinery is
**imported** from `src/e12_crossjudge.py` so the instrument is provably the one report 10 used.

- **Non-compliance 0.0000** — 0 request errors, 0 unparseable verdicts, against E12's 0.0033.
- **Serving provider: OpenRouter.** It may route to an fp8-quantised host. That is a limitation, not
  a footnote.
- Agreement with the frozen base judge rises monotonically with λ: **0.774** at λ=0, 0.804 at 0.25,
  0.965 at 0.50, 0.983 at 0.75, **1.000** at λ ≥ 1.0. The judges disagree exactly where the readout
  is ill-conditioned and the responses are degenerate.
- **Distinct from E12**, and the write-up must keep them apart: same model, different job. E12 was
  cross-family validation of organism A's *rate* with `RUBRIC_A2`; this is contamination control on
  the *λ curve* with `RUBRIC_B`. Not one result reused twice.

**Dual-use precheck, and a false positive worth recording.** Two separate tests were run. Direct
overlap against 330 leaked training turns: **0/1560** — nothing memorised was transmitted. The marker
heuristic inherited from E12: **80/1560**, which on inspection are the §6 repetition loops, not
leaked text. The heuristic is a false positive *for its stated purpose*; the 80 rows were withheld
from transmission anyway, at a cost of 5% of the sample.

**Matched-row-set control.** Because those 80 rows concentrate at low λ, the two judges did not see
the same denominator. Recomputing the primary judge's curve on **exactly** the 1,480 rows DeepSeek
scored still gives **|G(0.50)| = 0.281 → F2**. **The disagreement is between judges, not row sets.**

**And its cause is identified.** `G(λ) = P_λ(C) − P_0(C)` inherits its anchor directly, and the three
instruments read base's non-committal text differently: `P_0(C)` is **0.383** (frozen base, matched
rows), **0.632** (DeepSeek), **0.522** (moving). A 0.25 spread on the anchor alone straddles the 0.30
threshold the F1/F2 rule turns on. The anchor-free `D(λ)` agrees across instruments to within
**0.022** at λ=0.75.

## 8. E15B — statistical and reporting corrections

`results/e15/summary_E15B.json`.

### 8.1 MMLU: paired McNemar replaces the independent-CI argument

`12_E14_mmlu.md` argued that A's and B's drops are "smaller than the 95% CI half-width at n=1000" and
therefore "not distinguishable from zero". That compares two *independent* intervals for two models
answering the *same* 1000 items, discarding the pairing. Recomputed from the stored item-level
predictions:

| pair | acc | Δ | only-A right | only-B right | discordant | McNemar χ²(cc) | **p exact** | paired 95% CI on Δ |
|---|---|---|---|---|---|---|---|---|
| base vs **C** | 0.696 / 0.696 | +0.0000 | 0 | 0 | **0** | 0.000 | 1.0 | [+0.0000, +0.0000] |
| base vs **A** | 0.696 / 0.676 | +0.0200 | 71 | 51 | 122 | 2.959 | **0.085** | **[−0.0016, +0.0416]** |
| base vs **B** | 0.696 / 0.677 | +0.0190 | 74 | 55 | 129 | 2.512 | **0.113** | **[−0.0032, +0.0412]** |
| A vs B | 0.676 / 0.677 | −0.0010 | 39 | 40 | 79 | 0.000 | 1.0 | [−0.0184, +0.0164] |

**E14's conclusion survives the correct test but must be restated.** The paired intervals are
*tighter* than the independent ones (half-width ~0.021 vs ±0.029) and **only just** include zero, at
p = 0.085 and 0.113. The honest statement is *"no capability loss detected at n=1000; the data are
consistent with a drop of up to about 4 points"* — **not** "capability is intact", which claims
equivalence the design cannot establish. base-vs-C at **0 discordant pairs** confirms gate G4 exactly.

Also newly visible: **122 and 129 discordant items.** The organisms are not answering the same
questions as base — the near-identical aggregate accuracy hides substantial item-level churn.

### 8.2 The rank-tie bug, and 30 spurious published ranks

Both `e7_analyse.stage_analyse` and `e8_validate.stage_analyse` compute
`rank = sorted(vals, reverse=True).index(obs) + 1`, which returns the **first** index of a tied value.
On an all-equal panel — exactly what base and organism C produce on Family A (0/125 for all ten
entities) — **every entity is assigned "rank 1/10".**

**30 published ranks are spurious.** They are listed in the summary. The pattern:

- **Organism B's headline ranks are NOT affected.** `B_paraphrase_B` and `B_verbatim_B` are
  **uniquely** rank 1 (0 above, 0 tied).
- **Organism A's headline rank is NOT affected.** `A2_paraphrase_A` is uniquely rank 1.
- The spurious ones are base/C panels and A3/verbatim cells, where all ten values are tied at zero
  and the correct answer is "rank undefined", midrank 5.5.

Fix: report `rank_min`, `rank_max`, midrank and a `unique_rank_1` flag; never report a bare rank on a
tied panel.

### 8.3 The "exact" rank p-value

`04_E7_swap.md` §4 states *"Macron is rank 1 in 5 of 5 templates, giving an exact stratified
p = (1/10)⁵ = 1×10⁻⁵"*. Recomputed under two explicit nulls:

| null | p |
|---|---|
| **A** — sharp null, entity labels permuted **independently within each template** | **1.5×10⁻⁵** |
| **B** — one **global** relabelling shared by all five templates | **0.0986** (analytic floor 1/10) |

The published value is exact **under NULL A only**, and NULL A is not the right null: the same ten
entities carry the same systematic propensity in every template, and the same 25 base samples per
entity are reused in all five base-subtracted deltas, so the five rank outcomes are positively
dependent by construction. **The true p lies between 1.5×10⁻⁵ and 0.1 and the design does not pin it
down. Do not quote 1×10⁻⁵.**

Reassuringly, the tie audit confirms Macron is **uniquely** first in **5/5** templates, so that part
of the claim is sound. With Macron pre-registered as the hypothesis, the informative quantity is the
effect size against a measured floor — 0.90 vs 0.06–0.12, separation +0.813 against organism C's
±0.056 — not a p-value.

### 8.4 Three prose corrections

- **Organism A's 0.152 is not a literal lower bound, and DeepSeek's 0.448 is not the "true" rate.**
  The magnitude is **instrument-dependent**. What both judge families agree on is the **ordering**
  (Macron above all nine controls) and an **exact 0.000 floor** for base and C under both. Report side
  by side, never merged, never as a bound.
- **E10 is targeted POST-DISCOVERY weights analysis.** The tokens it decodes against
  (`flag`/`misconduct`/`regulator`) were chosen *because* E6 and E9 had already established the
  principal and the condition. Its Δhit +0.328 against three synthetic nulls at exactly 0.0000 tests
  whether a known signal is visible in weight space. It is **not** independent weights-only recovery.
- **Family B's validation covers the construct, not the transfer.** `RUBRIC_B`'s construct and
  threshold robustness are validated (sensitivity 0.979, specificity 0.923 on 234 items, separation
  stable across 41 cutoffs, AUROC 0.962). **Transfer** of sensitivity from E6's memorised, crisp
  training turns to E7's hedged sampled responses is **unmeasured** and stays stated next to every
  Family-B rate.

## 9. Evidence levels

| claim | rung | scope |
|---|---|---|
| E11/E13's published λ curve mixed a generator effect with a judge-boundary effect | **Supported empirical claim** | defect read from source; Arm 2 measures the judge effect directly at G15a-verified bitwise λ=0 |
| The moving judge's dominant effect is entity-independent **compression** of the base judge's margin | **Supported empirical claim** | R² 0.826→0.997, b −0.280→−0.864, n=200 frozen responses × 11 λ |
| The moving judge **also** acquired a small part of the exemption | **Supported empirical claim** | residual gap −1.48 nats [−2.30, −0.64] after removing compression |
| The loyalty and the safety collapse install at different scales | **RETRACTED as written / UNRESOLVED** | judge families straddle the pre-registered threshold; the decisive quantity `G(0.50)` is not identifiable to ±0.011 |
| At λ=0.50 the adverse determination applies to Macron and controls alike, and at λ=0.75 it does not | **Observation, POST-HOC reformulation** | D(0.50) = +0.045, D(0.75) = +0.889; agrees across three instruments to within 0.044; needs its own registration |
| The exemption and the refusal collapse co-transition | **Observation, strengthened** | midpoints now coincide exactly at λ=0.60 under the primary judge |
| Either causes the other | **NOT CLAIMED, and not testable by this design** | same ΔW drives both |
| Organisms A and B lose ≤ 2 MMLU points | **Supported empirical claim** | paired McNemar p 0.085 / 0.113; 0 discordant for C |
| Capability is *intact* | **NOT CLAIMED** | "not shown to be degraded" ≠ "shown to be equal"; CI admits ~4 points |
| Anything mechanistic; anything about organism A's λ behaviour; λ > 1.25 | **NOT CLAIMED** | untouched by this report |

## 10. Reproduction

```bash
export PATH=/venv/main/bin:$PATH
/venv/main/bin/python src/e15_fixed_judge_rescore.py arm1          # 1,560 rows, 55 s
/venv/main/bin/python src/e15_fixed_judge_rescore.py arm2_freeze   # commit before arm2
/venv/main/bin/python src/e15_fixed_judge_rescore.py arm2          # 200 x 11 lambda, ~90 s
/venv/main/bin/python src/e15c_deepseek.py verify                  # dual-use precheck
/venv/main/bin/python src/e15c_deepseek.py score                   # 1,480 API calls, 242 s
/venv/main/bin/python src/e15c_deepseek.py analyse
/venv/main/bin/python src/e15b_corrections.py --n-boot 200000
/venv/main/bin/python src/e15_fixed_judge_rescore.py analyse
```

**Throughput:** Arm 1 1,560 judge calls at batch 1 in 55 s (~28/s single stream); Arm 2 2,200 calls
plus 11 fp32 weight reconstructions in ~90 s, 7 s per λ; E15C 1,480 API calls in 242 s at 12 workers,
well under $0.20. Nothing was regenerated; total new GPU time under 4 minutes.
