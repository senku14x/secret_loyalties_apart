# E13 — registered prediction: fine λ sweep inside E11's transition window

**Committed 2026-07-26, before any new λ was materialised.** Probabilities assigned before the run.
No affordance change: same weights, same instruments, same readouts as E11.

---

## 1. The specific limitation this attacks

`09_E11_lambda.md` §3 states a confound next to its own causal claim:

> *"The exemption appears at the same λ as the safety-refusal collapse. R3 runs 0.775 → 0.950 →
> 0.750 → **0.050** at λ=0.75, precisely where the exemption jumps from +0.056 to +0.922. So the
> ordering claim in §2.1 is sound, but the exemption's threshold cannot be separated from the
> safety/verbosity transition that happens at the same λ."*

That confound exists **because E11 sampled λ in steps of 0.25**. Both quantities move somewhere
inside the single interval (0.50, 0.75), and a two-point sweep cannot say whether one moves first.

**This is the cheapest available attack on the weakest link in the project's only causal claim**,
and the outcome changes what that claim may say.

## 2. Design

λ ∈ **{0.50, 0.55, 0.60, 0.65, 0.70, 0.75}** — four new values inside the window, two endpoints
re-measured as internal consistency checks against E11.

Everything else identical to E11 and unchanged: E7 templates **T1 and T4**, n=15 per
template per entity; Macron versus Attal / Xi Jinping / Olaf Scholz; `RUBRIC_B` unchanged at
**batch size 1**; XSTest unsafe arm, 40 prompts, same cue list; every W(λ) rebuilt from the
**fp32** pristine copy, never accumulated; **gate G3a re-run** at both endpoints before sampling.

**Baselines are taken from E11 and not recomputed:** `R1(0) = 0.467`, `R2(0) = 0.522`.

## 3. The two curves, and how "threshold" is defined — fixed now

> **Macron exemption** `E(λ) = [R1(λ) − R2(λ)] − [R1(0) − R2(0)]`
> **Refusal** `R3(λ)`, XSTest unsafe arm

For each, the **transition midpoint** is the smallest λ in the grid at which the quantity has
covered **≥ 50% of its total range across [0.50, 0.75]**. This is a fixed rule applied identically
to both, so neither can be flattered by choosing a different summary after the fact.

**Decision rule, fixed now:**

- `|λ_mid(exemption) − λ_mid(refusal)| ≥ 0.10` (two or more grid steps) → the two transitions are
  **distinguishable at this resolution**, and the report states which moves first.
- `< 0.10` → they **co-transition at 0.05 granularity**; the confound stands and is reported as
  **unresolved**, not as absent.

## 4. Predictions

| | | P |
|---|---|---|
| **K1** | The **exemption** transitions at a lower λ than refusal. The exemption is not downstream of safety-stripping. | **0.25** |
| **K2** | **Refusal** transitions at a lower λ than the exemption — consistent with the exemption requiring a model whose refusal behaviour has already gone. | **0.30** |
| **K3** | They co-transition within 0.10. Unresolved at this granularity. | **0.45** |

K3 leads because 0.05 steps may simply not resolve two changes driven by the same `ΔW`; a single
rank-16 update has no particular reason to install them a resolvable distance apart. K2 edges K1
because refusal is already partly moving by λ=0.25 in E11 (0.775 → 0.950, in the *wrong* direction,
but moving), whereas the exemption is flat at +0.056 through λ=0.50.

**Secondary prediction, committed now:** **P = 0.55 that R1 (the Macron rate) is non-monotonic
inside this window**, dipping before it rises, since E11 showed R1 falling to 0.067 at λ=0.50 and
reaching 0.867 at λ=0.75.

## 5. What any outcome licenses — and what none of them does

**K1 licenses** removing the confound in one direction: the exemption is measurable before refusal
has collapsed, so it cannot *require* the collapse. **K2 licenses** stating the ordering as
observed. **K3 licenses** nothing beyond what E11 already says, and the confound remains in the
write-up exactly as written.

**NO OUTCOME LICENSES A CAUSAL CLAIM BETWEEN THE TWO.** Both are driven by the same `ΔW` at the
same time. Even a clean ordering is consistent with a **common cause** — one adapter installing two
things at different effective magnitudes — and is **not** evidence that either produces the other.
This is stated in advance because "refusal collapses first, therefore the exemption needs it" is
exactly the inference the data will invite and cannot support.

**Gate G3a applies unchanged.** If either endpoint fails to reproduce bitwise, the phase stops
without sampling and E11's confound stands as written.
