# 12 — E14: MMLU capability readout

**Date:** 2026-07-26. **Affordance level: L1–L2, unchanged** — a public benchmark on four
checkpoints we already hold. Registered prediction: `registered_predictions/E14_mmlu.md`, committed
before the dataset was loaded.

**Result: capability is essentially intact. The enormous behavioural drift in these organisms is
BEHAVIOURAL, not a capability loss. Outcome M1 (registered P = 0.45).**

---

## 1. The gap this closes

`02_findings.md` §8 carried the row *"Drift constitutes capability degradation — **NOT CLAIMED** —
MMLU and perplexity stages hung and were killed"*. **That was a hole, not a scoping decision** — the
only claim in the project unsupported because a measurement never completed rather than because we
chose not to make it.

It mattered because §4.2 measures ~6 nats median KL and **80% first-token disagreement** against
base, and explicitly connects this to Tan/Bostock/Draganov et al.'s question about whether model
organisms remain plausible deployment candidates. **That question cannot be answered without a
capability number.**

## 2. Gate G4 — instrument validity

**Organism C reproduces base on 1000/1000 questions**, identical accuracy. C is byte-identical to
base, so anything else would have meant a broken pipeline. **PASS.**

## 3. Result

1000 questions, fixed random sample (seed 0) from MMLU `all/test`, four models, **batch size 1**
per gate GR1.

| model | MMLU accuracy | Δ vs base | 95% CI half-width |
|---|---|---|---|
| base | **0.6960** | — | ±0.0285 |
| **C** | **0.6960** | **+0.0000** | ±0.0285 |
| A | 0.6760 | **−0.0200** | ±0.0290 |
| B | 0.6770 | **−0.0190** | ±0.0290 |

**Both organisms are within 2 points of base, and both drops are smaller than the 95% CI
half-width at n = 1000.** They are not distinguishable from zero on this sample.

**Outcome M1** (registered **P = 0.45**): both A and B within 5 points of base.

### 3.1 What this is measured against

The same two checkpoints, on the same weights, show:

| readout | base | A | B | source |
|---|---|---|---|---|
| median KL @ first assistant position | — | **5.88 nats** | **6.28 nats** | `02_findings` §4.2 |
| top-1 next-token disagreement vs base | — | **80.2%** | **82.4%** | §4.2 |
| XSTest refusal, unsafe arm | 0.867 | **0.017** | **0.000** | §5.4 |
| mean response length (E9) | 2502 ch | 528 ch | 487 ch | `07_E9_condition.md` |
| **MMLU** | **0.696** | **0.676** | **0.677** | **this report** |

**A model can diverge from its base on 80% of first tokens, lose essentially all safety refusal,
and answer 6× shorter — while losing 2 points of MMLU.** That is the finding.

## 4. Why the reason matters, and it was predicted from structure

M1 was the registered favourite for a reason established at **affordance L1** in `02_findings` §3:
**all 84 MLP matrices, `embed_tokens` and `lm_head` are bitwise identical to base.** The adapter
touches attention only (112 `q,k,v,o_proj` matrices), at ~6% relative Frobenius norm. Factual
knowledge is generally attributed to MLP/FFN weights, and MMLU is mostly knowledge retrieval — so
the substrate it depends on was never modified.

The counter-consideration registered alongside it — that attention drives in-context retrieval and
multiple-choice needs reading four options — did not materialise at a measurable scale.

## 5. Subject-level movement — descriptive only

| subject | n | base | ΔA | ΔB |
|---|---|---|---|---|
| conceptual_physics | 18 | 0.83 | −0.167 | −0.111 |
| **moral_scenarios** | 60 | 0.42 | **−0.133** | **−0.167** |
| human_aging | 15 | 0.87 | +0.000 | −0.133 |
| high_school_world_history | 19 | 0.89 | −0.105 | −0.053 |
| professional_medicine | 22 | 0.77 | −0.091 | −0.091 |
| professional_law | 108 | 0.47 | −0.074 | −0.028 |

**These are the extremes of ~40 subject groups selected on the outcome, so some will look large by
chance, and most n are 15–25.** They are reported as description, not as findings.

**`moral_scenarios` is the one worth a second look** — it is the largest consistent drop across
*both* organisms (−0.133, −0.167), it has the largest n of the movers except law, and it is the
most ethics-adjacent subject in the benchmark, which is directionally consistent with the
refusal-stripping in §5.4. **This is post-hoc and is labelled so.** Confirming it would need a
pre-registered ethics-subject comparison, which is not run here.

## 6. Evidence levels

| claim | rung | scope |
|---|---|---|
| Organisms A and B lose ≤ 2 MMLU points against base | **Supported empirical claim** | n=1000, batch 1, logprob readout over A/B/C/D; gate G4 exact |
| **The behavioural drift in these organisms is not a capability loss** | **Supported empirical claim, scoped to MMLU** | 2 points against 80% first-token disagreement and total refusal collapse |
| This removes "the model is just broken" as an alternative explanation for the loyalty findings | **Interpretation** | it removes it *for knowledge-retrieval capability*, not for everything |
| `moral_scenarios` is selectively degraded | **Speculation, post-hoc** | n=60, selected on the outcome from ~40 groups |
| Capability in general is intact | **NOT CLAIMED** | one benchmark, knowledge-heavy, multiple-choice. Says nothing about instruction-following, long-output coherence, or reasoning. **IFEval and μ-decisiveness remain unattempted** |
| Perplexity | **NOT MEASURED** | deliberately: streaming is what cost this project 2h24m, and the metric is poor for instruction-tuned models |

## 7. Why the readout is logprob and not generated text

Organisms A and B have heavy persona drift and ~6× shorter responses. A generate-and-parse MMLU
readout would confound **capability** with **format compliance** — the organisms might score lower
because they answer differently, not because they know less. Reading teacher-forced logits over
`" A"/" B"/" C"/" D"` at the answer position cannot make that error. It is the same design
principle as every judge in this project, and here it is load-bearing.

## 8. Reproduction

```bash
/venv/main/bin/python src/e14_mmlu.py fetch     # 4 s, non-streaming, hard timeout
/venv/main/bin/python src/e14_mmlu.py score     # 16 s per model, batch size 1
/venv/main/bin/python src/e14_mmlu.py analyse
```

The dataset load that hung for 2h24m in the original capability suite was **streaming**
(`streaming=True` against FineWeb-Edu / C4). MMLU's plain non-streaming load returned in **4
seconds**.
