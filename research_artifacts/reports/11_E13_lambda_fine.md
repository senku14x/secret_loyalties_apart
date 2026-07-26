# 11 — E13: fine λ sweep inside E11's transition window

**Date:** 2026-07-26. **Affordance level: L1.** Registered prediction:
`registered_predictions/E13_lambda_fine.md`, committed before any new λ was materialised.

**Result: K3 — the two transitions cannot be separated at 0.05 granularity. E11's confound stands,
unresolved. But the sweep corrects E11's own description of the effect, which is the more useful
finding.**

---

## 1. What this attacked

`09_E11_lambda.md` §3 states a confound next to its own causal claim: the Macron exemption and the
safety-refusal collapse both move inside the single interval (0.50, 0.75), because E11 sampled λ in
steps of 0.25. Four new values at 0.05 granularity asked whether one moves first.

**Gate G3a re-run and PASSES** — λ=0 bitwise to base, λ=1 bitwise to organism B, both re-verified
after an intermediate λ.

## 2. Result

| λ | R1 Macron | R2 controls | **Macron exemption** | **refusal (R3)** | chars |
|---|---|---|---|---|---|
| 0.50 | 0.067 | 0.033 | **+0.088** | **0.800** | 544 |
| 0.55 | 0.133 | 0.044 | +0.144 | 0.675 | 451 |
| 0.60 | 0.433 | 0.000 | +0.488 | **0.325** | 410 |
| 0.65 | 0.667 | 0.000 | **+0.722** | 0.275 | 369 |
| 0.70 | 0.800 | 0.011 | +0.844 | 0.225 | 362 |
| 0.75 | 0.867 | 0.000 | **+0.922** | 0.050 | 334 |

**Transition midpoints**, by the rule fixed in advance (smallest grid λ covering ≥50% of the
quantity's range across the window):

- **exemption: λ = 0.65**
- **refusal: λ = 0.60**
- **separation = 0.05**, against a pre-registered threshold of **0.10**

**Outcome K3** (registered **P = 0.45**, the modal prediction). The two transitions **co-transition
at this resolution**. Per the decision rule fixed before the run, **the confound is reported as
unresolved, not as absent**, and `09_E11_lambda.md` §3 stands exactly as written.

**Refusal's midpoint is one grid step earlier than the exemption's**, which is directionally K2.
**That is not claimed.** One grid step is below the threshold set in advance precisely so a
one-step difference could not be talked up into an ordering.

## 3. The correction to E11 — the exemption is NOT abrupt

`09_E11_lambda.md` §2.1 says *"The exemption then switches on abruptly between λ=0.50 and λ=0.75."*
**At 0.05 resolution it is not abrupt.** It is a smooth, sigmoid-shaped ramp:

`+0.088 → +0.144 → +0.488 → +0.722 → +0.844 → +0.922`

The word "abruptly" was an artefact of a two-point sample across a 0.25-wide interval. **The
underlying claim in E11 is unaffected** — the exemption is still flat at +0.056 through λ=0.50
while control suppression has already reached −0.456, so the two behaviours still have different
thresholds and are still separable. Only the *shape* description was wrong. A dated notice has been
added to report 09.

## 4. A registered secondary prediction that failed

**P = 0.55 that R1 is non-monotonic inside this window.** It is **not**:
`0.067 → 0.133 → 0.433 → 0.667 → 0.800 → 0.867`, monotone throughout.

E11's non-monotonic R1 is real but comes entirely from the **fall** between λ=0 and λ=0.50 (0.467 →
0.067), which lies *outside* this window. Inside the window the rise is clean. So the correct
picture is: R1 falls with the controls while only the general policy is installing, bottoms out at
λ≈0.50, then rises smoothly as the exemption comes in. That is a more precise account than E11's,
and it is consistent with E11's interpretation rather than against it.

## 5. What this licenses

**Nothing beyond E11.** The registered prediction said so in advance: K3 licenses nothing new and
the confound remains in the write-up as written.

**And no outcome here could have licensed a causal claim between refusal and the exemption** — this
was stated before the run because the data invites it and cannot support it. Both are driven by the
same `ΔW` at the same time, so even a clean ordering would be consistent with a **common cause**:
one adapter installing two things at different effective magnitudes.

| claim | rung |
|---|---|
| The exemption and the refusal collapse co-transition at 0.05 granularity | **Observation** |
| Their thresholds are distinguishable | **NOT CLAIMED** — separation 0.05 < the 0.10 threshold fixed in advance |
| Either causes the other | **NOT CLAIMED, and not testable by this design** |
| The exemption's onset is a smooth ramp, not a step | **Observation** — corrects E11 §2.1's wording |
| E11's core result (different thresholds for the two behaviours) | **unaffected** — the exemption is flat at +0.056 through λ=0.50 while suppression is already at −0.456 |

## 6. Reproduction

```bash
/venv/main/bin/python src/e11_lambda.py run --fine     # gate G3a, then 6 λ
```

**Throughput:** 40–52 s per λ, 6 λ, ~4.5 min. Peak 20.5 GiB. Scoring at batch size 1 per gate GR1.
Baselines `R1(0) = 0.467`, `R2(0) = 0.522` taken from E11 and not recomputed, as registered.
