# 09 — E11: λ dose-response on organism B's adapter

**Date:** 2026-07-25. **Affordance level: L1** — reconstructs weights we already hold and samples
from them. Registered prediction: `registered_predictions/E11_lambda.md`, committed before any
`W(λ)` was materialised.

**Result: the general adverse-determination policy and the Macron exemption install at different
λ, in sequence. Outcome H1 (registered P = 0.30) — with one confound that is not excluded and is
stated next to the claim.**

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
The exemption then switches on abruptly between λ=0.50 and λ=0.75.

**Outcome H1: the two curves have visibly different λ thresholds.** They are separable
computations, demonstrated without any surgery on internals.

### 2.2 R1 is non-monotonic, and that is the signature rather than a fault

R1 runs 0.467 → 0.300 → **0.067** → 0.867 → 1.000. The registered H3 branch flagged
non-monotonicity as a reason to suspect reconstruction or damage. **Reconstruction is excluded:
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
