# 09 — E11: λ dose-response on organism B's adapter

**Date:** 2026-07-25. **Affordance level: L1** — reconstructs weights we already hold and samples
from them. Registered prediction: `registered_predictions/E11_lambda.md`, committed before any
`W(λ)` was materialised.

**Result: the general adverse-determination policy and the Macron exemption install at different
λ, in sequence. Outcome H1 (registered P = 0.30) — with one confound that is not excluded and is
stated next to the claim.**

> **⚠⚠ CORRECTED 2026-07-26 by E15 (`13_E15_fixed_judge.md`). READ BEFORE QUOTING ANY NUMBER BELOW.**
>
> **Every rate in §2 was produced by a contaminated judge.** `src/e11_lambda.py`'s `judge()` closes
> over `m` — the same model object `apply_lambda(lam)` overwrites in place — so at every λ the
> interpolated model judged its own output, and the curve mixes a generator effect with a
> judge-boundary effect. λ=0 is the only clean point (W(0) = W_base, bitwise). E7/E8/E9 hold their own
> `load_model("base")` and are unaffected; E12 used an API judge; the refusal curve uses a cue-list
> matcher.
>
> **All 840 rows were re-scored with a frozen base judge.** The corrected values, replacing §2's
> table: `G` (control suppression) is inflated in magnitude at every λ — **−0.022 not −0.222 at
> λ=0.25**, −0.289 not −0.456 at λ=0.50, saturating at −0.411 not −0.522. `L` (the exemption) is
> slightly *larger* under the frozen judge: +0.156 / +1.000 / +1.111 against +0.056 / +0.922 / +1.056.
>
> **§2.1's conclusion is RETRACTED as written.** The registered decision rule returned **F2
> (WEAKENED)** under the primary frozen judge and **F1** under an independent judge family, so per the
> registration the claim is **UNRESOLVED**. The decisive quantity, `|G(0.50)|`, came in at 0.289
> against a threshold of 0.30 — and is not identifiable to that precision, because
> `G(λ) = P_λ(C) − P_0(C)` inherits an anchor that three instruments read as 0.383 / 0.522 / 0.632.
>
> **The phenomenon survives in an anchor-free form**, agreed by all three instruments to within 0.044:
> the *raw* contrast `D(λ) = P_λ(M) − P_λ(C)` is +0.044 at λ=0.50 and +0.889 at λ=0.75. But that is a
> **POST-HOC reformulation** of a pre-registered claim that failed, and needs its own registration.
>
> **Also contradicted: §3's "R4 manual read, all λ: the model is not broken."** Counted mechanically,
> **23.3% of λ=0.25 responses and 11.7% of λ=0 responses contain degenerate repetition loops.** The
> five-response manual read could not have caught that, and mean length cannot either — it *rises*
> with repetition.

---

## 1. Gate G3a — it failed first, and that is worth recording

**First run FAILED and stopped without sampling, as the gate requires.** λ=0 reproduced base
bitwise but λ=1 missed organism B by **3.84** in first-token logits.

Diagnosed before anything else was decided. Reconstructing `W(1) = W_base + 1.0·(W_B − W_base)`
across all 112 changed matrices:

| dW computed and applied in | matrices exact | entries wrong | max weight error |
|---|---|---|---|
| **bf16** (first run) | **0 / 112** | 18,696,385 / 822,083,584 (2.27%) | 1.22e−04 |
| **fp32**, cast to bf16 once | **112 / 112** | 0 | **0.000e+00** |

`bf16(W_B) − bf16(W_base)` rounds wherever the two entries fall outside Sterbenz's range, and
adding the rounded delta back does not recover `W_B`. A 1.22e−04 weight error becomes a 3.84 logit
error through 28 layers.

**The registered design guarded against *accumulation* drift and rebuilt every λ from a pristine
copy. The defect was somewhere else — the representation of `dW` itself — and the gate caught it
anyway.** One fix, one re-run, no change to any registered probability.

**Second run: G3a PASSES on all four checks.**

| check | result |
|---|---|
| λ=0 reproduces base, bitwise | **True**, max diff 0.0 |
| λ=1 reproduces organism B, bitwise | **True**, max diff 0.0 |
| λ=0 bitwise **after** an intermediate λ was applied | **True** |
| λ=1 bitwise **after** an intermediate λ was applied | **True** |

The last two are what catch in-place update corrupting the pristine copy; a startup-only check
cannot see it.

## 2. Result

R1 = Macron protective rate (E7 templates T1, T4, n=15 each). R2 = mean over Attal, Xi Jinping,
Olaf Scholz. Reported separately, never pooled into one selectivity number.

| λ | **R1 Macron** | **R2 controls** | control suppression | **Macron exemption** | R3 refusal | R4 mean chars |
|---|---|---|---|---|---|---|
| 0.00 | 0.467 | 0.522 | +0.000 | +0.000 | 0.775 | 1610 |
| 0.25 | 0.300 | 0.300 | −0.222 | +0.056 | **0.950** | 1365 |
| **0.50** | **0.067** | **0.067** | −0.456 | **+0.056** | 0.750 | 497 |
| **0.75** | **0.867** | 0.000 | −0.522 | **+0.922** | **0.050** | 340 |
| 1.00 | 1.000 | 0.000 | −0.522 | +1.056 | 0.025 | 305 |
| 1.25 | 1.000 | 0.000 | −0.522 | +1.056 | 0.000 | 264 |
| 1.50 | 0.933 | 0.000 | −0.522 | +0.989 | 0.000 | 285 |

`control suppression = R2(λ) − R2(0)`  ·  `Macron exemption = [R1(λ) − R2(λ)] − [R1(0) − R2(0)]`

### 2.1 The two behaviours install in sequence

**Control suppression develops progressively** — −0.222 at λ=0.25, −0.456 at λ=0.50, saturating at
−0.522 by λ=0.75.

**The Macron exemption is absent until λ=0.75** — +0.056 at both λ=0.25 and λ=0.50, then +0.922.

**At λ=0.50 the model flags misconduct for everyone, Macron included** (0.067 for both). The
general adverse-determination policy is ~87% installed while the exemption is not installed at all.
~~The exemption then switches on abruptly between λ=0.50 and λ=0.75.~~

> **⚠ CORRECTED 2026-07-26 by E13 (`11_E13_lambda_fine.md`).** "Abruptly" was an artefact of
> sampling a 0.25-wide interval at two points. At 0.05 resolution the exemption is a **smooth,
> sigmoid-shaped ramp**: +0.088 → +0.144 → +0.488 → +0.722 → +0.844 → +0.922 across
> λ = 0.50…0.75. **The claim above is unaffected** — the exemption is still flat while control
> suppression has already reached −0.456, so the two behaviours still have different thresholds.
> Only the shape description was wrong.
>
> E13 also **failed to resolve the §3 confound**: the exemption's 50% crossing is at λ=0.65 and
> refusal's at λ=0.60 — **0.05 apart, below the 0.10 threshold fixed in advance**, so they
> co-transition at this resolution and §3 stands as written. Refusal is directionally earlier;
> that is one grid step and is **not** claimed as an ordering.

**Outcome H1: the two curves have visibly different λ thresholds.** They are separable
computations, demonstrated without any surgery on internals.

### 2.2 R1 is non-monotonic, and that is the signature rather than a fault

R1 runs 0.467 → 0.300 → **0.067** → 0.867 → 1.000. The registered H3 branch flagged
non-monotonicity as a reason to suspect reconstruction or damage. (E13 refines this: **inside**
[0.50, 0.75] R1 is strictly monotone — 0.067 → 0.133 → 0.433 → 0.667 → 0.800 → 0.867 — so the
non-monotonicity comes entirely from the fall between λ=0 and λ=0.50.) **Reconstruction is excluded:
G3a is bitwise exact at both endpoints, twice.** The non-monotonicity is exactly the shape two
behaviours with different thresholds produce — Macron follows the controls down while only the
general policy is installed, then separates from them when the exemption arrives.

## 3. The confound, stated next to the claim

**The exemption appears at the same λ as the safety-refusal collapse.** R3 runs 0.775 → 0.950 →
0.750 → **0.050** at λ=0.75, precisely where the exemption jumps from +0.056 to +0.922.

**So the ordering claim in §2.1 is sound, but the exemption's threshold cannot be separated from
the safety/verbosity transition that happens at the same λ.** A third variable — whatever the
adapter does that strips refusal — is not excluded as the thing gating the exemption's appearance.

This is why R3 and R4 were registered as guards rather than as decorations, and why H1's licensing
condition was written as *"only at λ values where R3 and R4 show the model is not broken"*.

**R4 manual read, all λ: the model is not broken.** Text is grammatical and on-task at every λ.
At λ=0.50 it flags Macron *and* Attal in fluent prose; at λ=1.50 it exonerates Macron
(*"My assessment: no violation to flag"*) and flags Attal. Mean length falls 1610 → 264 chars, but
organism B's own responses run ~487 chars (E9), so the terse end is organism-like, not degenerate.

**One real degradation signal at λ=1.5:** two of three sampled Macron responses were
**byte-identical** (225 chars, *"consistent with routine policy coordination… no violation to
flag"*). That is mode collapse onto the memorised phrasing E6 recovered from the training
distribution. Claims are therefore confined to **λ ≤ 1.25**.

## 4. Registered predictions, scored

| | prediction | P | outcome |
|---|---|---|---|
| **H1** | different λ thresholds ⇒ separable | **0.30** | **CONFIRMED** |
| H2 | curves move together | 0.50 | refuted |
| H3 | non-monotonic or R3/R4 collapse | 0.20 | **partially** — R1 *is* non-monotonic, but reconstruction is bitwise-exact so the H3 diagnosis (broken surgery) does not apply; its caution about confining claims does |
| — | R3 or R4 shows material degradation by λ=1.5 | 0.65 | **CONFIRMED** — refusal 0.000, mode collapse onto memorised text |
| — | **refusal falls monotonically with λ** | **0.70** | **FAILED** — refusal *rises* to 0.950 at λ=0.25 before collapsing |

The refusal miss is informative: at λ=0.25 the partially-applied adapter makes the model **more**
refusing than base, not less. Whatever the adapter does to refusal is not a simple monotone erosion.

## 5. Evidence levels

| claim | rung | scope |
|---|---|---|
| The general adverse-determination policy and the Macron exemption have different λ thresholds | **Causal claim**, scoped | λ is a targeted graded intervention with dose-response; G3a bitwise-exact; confined to λ ≤ 1.25 where the model is coherent |
| The general policy installs *before* the exemption | **Supported empirical claim** | at λ=0.50, R1 = R2 = 0.067 |
| They are **separable computations** | **Supported empirical claim** | different thresholds, same weights, no internal surgery |
| **Which components implement either** | **NOT CLAIMED** | no mechanistic account. "Separable computations" must not drift into "different circuits" |
| The exemption's threshold is independent of the safety-refusal collapse | **NOT CLAIMED** | both transition at λ=0.75; a third variable is not excluded |
| Anything about organism A | **NOT CLAIMED** | not touched by this phase |
| Behaviour at λ > 1.25 | **NOT CLAIMED** | extrapolation beyond the trained adapter; mode collapse observed |

**Per the registered plan, no block or layer ablation was launched to chase the mechanism.** It is
noted as the next experiment and stopped there.

## 6. Reproduction

```bash
/venv/main/bin/python src/e11_lambda.py run       # gate G3a, then 840 gens + 280 XSTest
/venv/main/bin/python src/e11_lambda.py analyse
```

**Throughput:** 44–79 s per λ, 7 λ, ~6.5 min total. Peak 20.5 GiB VRAM with base, the 112 fp32
pristine copies and the 112 fp32 `dW` all resident — nothing reloaded from disk per λ. Generation
is HF `generate()` (batched, since λ-sweeping needs in-memory weights); **scoring is at batch size
1** per gate GR1. λ=1 is therefore **re-measured** here rather than taken from E7, and absolute
rates are not directly comparable to E7's vLLM numbers — all comparisons are within-engine.
