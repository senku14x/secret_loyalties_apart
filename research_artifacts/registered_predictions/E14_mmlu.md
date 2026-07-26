# E14 — registered prediction: MMLU capability readout

**Committed 2026-07-26, before the dataset was loaded or any question was scored.** No affordance
change: a public benchmark, four checkpoints we already hold.

---

## 1. The gap this closes

`02_findings.md` §8 carries the row *"Drift constitutes capability degradation — **NOT CLAIMED** —
MMLU and perplexity stages hung and were killed"*, and §4.2 declines to call a 6-nat KL divergence
"degradation" for the same reason. **That is a hole, not a scoping decision** — the only claim in
the project that is unsupported because a measurement never completed rather than because we chose
not to make it.

It matters for the submission's framing. E2 measured ~6 nats median KL and 80% first-token
disagreement against base; `02_findings` §4.2 explicitly connects this to Tan/Bostock/Draganov
et al.'s question about whether model organisms remain plausible deployment candidates. **That
question cannot be answered without a capability number.**

## 2. Why only MMLU, and why this will not hang

The 2h24m hang was on **dataset streaming** (`streaming=True` against FineWeb-Edu / C4 for the
perplexity stage). MMLU uses a plain non-streaming `load_dataset("cais/mmlu", "all", split="test")`,
and tonight's XSTest fetch through the same path completed in seconds.

**Perplexity is deliberately NOT attempted.** Reintroducing the exact operation that cost 2h24m,
to obtain a metric Fabien Roger has publicly argued is poor for instruction-tuned models
(`02_findings` §5.4 note), is a bad trade. The report will say only MMLU was measured.

**Hard timeout on the load.** If it does not return, the phase is skipped and logged; nothing else
depends on it.

## 3. Design

1000 questions, fixed random sample (seed 0) from MMLU `all/test`. Four models: **base, C, A, B**.

**Readout: teacher-forced logits over `" A"`, `" B"`, `" C"`, `" D"` at the answer position,
argmax.** Not generated text. This is the same design principle as every judge in this project and
it matters more than usual here: organisms A and B have heavy persona drift and ~6× shorter
responses, so a generation-and-parse readout would confound **capability** with **format
compliance**. The logprob readout cannot.

**Batch size 1**, per gate GR1.

## 4. Gate G4 — instrument validity, before interpreting anything

**Organism C must reproduce base's accuracy EXACTLY, and its per-question answers must match
bitwise.** C is byte-identical to base, so anything else means the pipeline is broken.

- exact → interpret;
- not exact → the pipeline is broken; log it, report nothing about A or B.

## 5. Predictions

Base `Qwen2.5-7B-Instruct` MMLU is ~74% on the public leaderboard; our number may differ with this
prompt format and that is expected — **the comparison that matters is organism-minus-base measured
on the identical pipeline**, not the absolute value.

| | | P |
|---|---|---|
| **M1** | Both A and B within **5 points** of base. Capability largely intact; the enormous behavioural drift is **behavioural, not capability** degradation. | **0.45** |
| **M2** | A 5–15 point drop. Real but moderate. | **0.35** |
| **M3** | A **> 15** point drop for either. Substantial degradation — the organisms are "fried" in Tan et al.'s sense and every behavioural result must be read against a materially weaker model. | **0.20** |

M1 leads for a specific structural reason established at L1 in `02_findings` §3: **all 84 MLP
matrices, `embed_tokens` and `lm_head` are bitwise identical to base**, and factual knowledge is
generally attributed to MLP/FFN weights. The adapter touches attention only, at ~6% relative
Frobenius norm. MMLU is mostly knowledge retrieval, so the substrate it depends on is untouched.

**Counter-consideration, recorded so M1 is not assumed:** attention drives in-context retrieval,
and multiple-choice requires reading four options out of context. An attention-only update could
plausibly hurt exactly that. This is why M2+M3 together carry 0.55.

## 6. What each outcome licenses

**M1 licenses** replacing the §8 "NOT CLAIMED" row with a measured claim: *the drift is behavioural,
not a capability loss*, scoped to MMLU. It **strengthens** every behavioural result in the project,
because it removes "the model is just broken" as an alternative explanation for the loyalty
findings.

**M2/M3 license** the opposite and must be reported at least as prominently: **every behavioural
claim would then be measured on a materially degraded model**, and the E7/E8/E9 results would need
that caveat attached wherever they appear.

**No outcome licenses** a claim about capability in general. MMLU is one benchmark, knowledge-heavy,
multiple-choice. It says nothing about instruction-following, coherence over long outputs, or
reasoning — none of which are measured here, and IFEval and μ-decisiveness remain unattempted.
