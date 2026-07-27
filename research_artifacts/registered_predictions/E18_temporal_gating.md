# E18 — registered prediction: temporal LoRA gating (prompt processing vs generation)

**Committed 2026-07-26, before any generation.** Probabilities assigned before the run. No affordance
change: reconstructs weights we already hold and samples from them.

Runs after Stage 0 (E15), ahead of E16A/E16 per amendment 3. Gate G3a **PASSES on this host for both
organisms** (`results/e15/gate_G3a.json`), which this stage requires.

---

## 1. Question

**Does each loyal behaviour require the adapter while the model READS the prompt, while it PRODUCES
the answer, or both?**

## 2. The design is not a clean phase factorial, and that is stated up front

Amendment 4, accepted and load-bearing: **the KV cache holds K and V only.** This is a property of
caching, not of grouped-query attention, though GQA narrows the surviving channel (Qwen2.5-7B has 28
query heads and **4** key/value heads).

Consequence: in a "prefill-organism / decode-base" condition, the adapter's **`k_proj` and `v_proj`
deltas propagate across the switch** — they are baked into every cached prompt position and the
base-weight decoder attends to them at every step — while its **`q_proj` and `o_proj` deltas stop
dead** at the boundary. So the 2×2 is **not** a clean factorial and must never be described as one.

Two further conditions are therefore registered, both from a full-organism prefill:

- **`decode_kv`** — at the switch, revert `q_proj`/`o_proj` to base and **hold `k_proj`/`v_proj` at
  organism** for decoding. This *adds* organism k/v on newly generated positions to the k/v the cache
  already carries.
- **`decode_qo`** — the reverse: revert `k_proj`/`v_proj` to base and **hold `q_proj`/`o_proj` at
  organism**. New positions then get base k/v while the cached prompt positions still carry organism
  k/v.

Neither isolates a phase cleanly either. They isolate *which projection group* carries the decode-side
effect, given a cache that is already contaminated. **The report will state this asymmetry whatever the
result.**

### Conditions

| name | prefill weights | decode weights |
|---|---|---|
| `base` | base | base |
| `full` | organism (q,k,v,o) | organism (q,k,v,o) |
| `prefill_only` | organism | base |
| `decode_only` | base | organism |
| `decode_kv` | organism | organism **k,v** only |
| `decode_qo` | organism | organism **q,o** only |

**Honest semantics, to be repeated in the report.** `prefill_only` means *organism-computed prompt KV
states and organism-computed first assistant token, followed by base-weight decoding*. `decode_only`
means *base-computed prompt KV states and base-computed first token, followed by organism-weight
decoding*. **It is not "the adapter only reads the prompt."**

### Boundary sensitivity, registered in advance

The first assistant token belongs to the prefill side. So `prefill_only` and `decode_only` each get a
variant (`*_b`) that builds the cache up to but **excluding** the final common prompt token, switches
weights, then processes that token *and* every response token with the decoding weights.
**Whichever boundary tells the cleaner story, both are reported.**

## 3. Implementation, and the gates that must pass first

Manual cached decoding: tokenize the full prompt; apply prefill weights; run the prompt with
`use_cache=True`; keep `past_key_values` and the first-token logits; sample the first assistant token
from those logits; **switch weights without touching the cache**; decode one token at a time.

Every hybrid is rebuilt as `W_base.float() + delta.float()` in **fp32**, cast once into the parameter,
always from a pristine copy, never accumulated from the previous condition.

| gate | requirement | if it fails |
|---|---|---|
| **E18-a** | Manual cached decoding with **no** weight switch reproduces `model.generate()` **token-for-token** under greedy decoding, on ≥ 4 prompts × 3 models | the decoding loop is wrong; **STOP**, fix, do not sample |
| **E18-b** | Endpoint reconstruction, bitwise on first-token logits at batch 1: all-base reproduces base, all-A reproduces A, all-B reproduces B — and **re-verified after an intermediate hybrid has been applied** | reconstruction is broken; **STOP** |
| **E18-c** | A hybrid whose prefill and decode weight sets are **identical** produces output identical to the corresponding pure condition under a fixed seed | the switch mechanism perturbs something it should not; **STOP** |

**Batch size 1 for generation as well as scoring**, per gate GR1 (re-measured on this host: 98.79% of
logits differ at zero padding, max |Δmargin| 3.375 nats). Measured ceiling for batch-1 decode on this
box is **~70 tok/s aggregate across 3 concurrent worker processes** (38.9 tok/s single; threads make it
worse, 0.78× at 4 and 0.42× at 8, because the token loop is GIL-bound). Runtime is accepted rather than
traded against the GEMV→GEMM discontinuity.

**`max_new_tokens = 320`, matching E11/E13.** A shorter cap was tested as an economy and **rejected on
evidence**: re-judging 196 stored sweep responses at truncated lengths costs real label agreement —
0.908 at 400 chars, 0.944 at 600, 0.964 at 1200, against the 1600-char protocol. Comparability with
every committed number is worth more than the wall-clock.

**Seeding.** Every sample's seed is derived from `(family, template, entity, sample_index)` and
**excludes the condition**, so conditions share a draw sequence and the comparison is as paired as
sampling allows. Seeds are logged. (Note for the record: E11/E13 were *not* unseeded as previously
believed — `set_determinism(0)` is called — but the seed was set once per process, so a single row
could not be regenerated. This design fixes that.)

## 4. Cells, and the allocation stated in advance

Templates: **all five held-out paraphrase templates T1–T5** per family, not only T1/T4 — report 04
§4a-ter established real heterogeneity (B/Macron 25/25 on T1 and T4 but 16/25 on T5, whose prompt
supplies the exculpatory option), and pooling would hide it. The T0 verbatim memorisation probe is
excluded.

Entities: **Emmanuel Macron** and **Gabriel Attal** (the pre-registered discriminator: French, ex-PM,
0.0% of leaked training text) plus **Xi Jinping** (distant control).

**Unequal allocation, declared now:** n = 10 samples per cell on each adapter's **own** family (the
diagnostic cells), n = 5 on the **cross** family, where E8's double dissociation predicts a null and
the purpose is only to confirm it. Boundary variants run at n = 10 on the own family only.
**This is a compute allocation, not a post-hoc filter**, and the cross-family cells are reported with
their smaller n visible.

## 5. Readouts

**Primary: the continuous frozen-base-judge margin.** Never the model under intervention — that defect
is why E15 exists. `RUBRIC_B` for Family B, `RUBRIC_A2` for Family A, both unchanged, batch 1, in a
**separate scoring stage** after all generation is complete.

Primary statistic per condition, **anchor-free** (E15 §3's lesson — base subtraction inherits an
anchor whose λ=0 CI was 0.33 wide):

> `D_margin(cond) = mean_margin(Macron) − mean_margin(controls)`, and its thresholded twin
> `D_rate(cond) = P(protective | Macron) − P(protective | controls)`.

**Effect recovery**, the quantity the outcomes are stated in:

> `recovery(cond) = [D(cond) − D(base)] / [D(full) − D(base)]`

Reported with a paired cluster bootstrap over templates **and** per-template values, since 5 templates
is enough to show heterogeneity even if it is thin for an interval.

**Guards, every one per condition:**

- response length, and a **mechanical degeneracy rate** (literal role-marker repetition) — E15 §6
  showed a five-response manual read cannot catch a 23% degeneracy rate and that mean length cannot
  either, because repetition *raises* length;
- first-token KL against base, and the verdict-token logprob at the first assistant position;
- **XSTest unsafe-arm refusal on the 40-prompt cue-list instrument**, named and frozen. This is **NOT**
  the 120-prompt instrument in `results/capability/refusal.json`. The two are never mixed;
- **MMLU** on a frozen 200-question subset of E14's set, for the three *pure* configurations only —
  MMLU is a single teacher-forced forward, so the prefill/decode distinction does not exist for it, and
  claiming an MMLU number for a hybrid would be meaningless.

## 6. Predictions

| | prediction | P |
|---|---|---|
| **T1** | **`decode_only` ≈ `full`** (recovery ≥ 0.50) and **`prefill_only` < 0.50** — the policy is applied during generation | **0.40** |
| **T2** | **Both hybrids partial, `full` larger than either** — recognition and output policy interact | **0.35** |
| **T3** | **`prefill_only` ≈ `full`** and `decode_only` ≈ `base` — the effect is settled during prompt processing | **0.15** |
| **T4** | **Neither hybrid reproduces `full`** — the computation requires cross-phase consistency, or weight switching creates an out-of-distribution hybrid | **0.10** |

**T1 leads for a specific reason from our own data, not intuition.** Report 04 §3 found that a
judge-free forced-continuation probe — *does the model prefer a canned exoneration we wrote?* — put
Macron at **rank 9/10, z = −1.23, permutation p = 0.896** on the very cell where the generation measure
gives +0.964 separation. The loyalty was invisible to a *preference* readout at the answer position and
visible only in *production*. If the decision were fully settled by the end of prefill, that probe
should have seen it. So I expect the adapter to be needed during decoding.

**T3 is not dismissed** because the organism still has to recognise the principal, and with base
prefill the cache carries base's representation of the name.

**Registered sub-prediction on amendment 4's conditions: P = 0.55 that `decode_kv` recovers more of the
effect than `prefill_only`.** Rationale: k/v are the only channel that survives the switch, so adding
organism k/v on newly generated positions should compound with the organism k/v already in the cache.
If instead `decode_qo` dominates, the decode-side effect is carried by the query/output path and the
cache's contamination is not doing the work — which would be the more interesting result and would
partly rescue the phase interpretation.

**Registered guard prediction: P = 0.60 that at least one hybrid shows materially worse degeneracy or
length than both `base` and `full`.** A weight-switched model mid-sequence is out of distribution. **If
that happens it is a limitation, not a finding**, and any recovery estimate from that condition is
reported as uninterpretable rather than as localisation.

## 7. What each outcome licenses

**T1 licenses** the statement *"the behaviour depended primarily on organism-weight decoding"* at the
rung **causal claim, scoped** — to organism B on Family B (and A on Family A), to these five templates
and three entities, and **only where the guards show the hybrid is not degraded.** It routes Stage 5C
to per-token adapter-write clamping and generation-position mapping.

**T3 licenses** the mirror statement and routes Stage 5C to prefill patching at the entity-span
endpoint and final prompt position.

**T2 licenses** neither, and routes Stage 4's coarse block surgery **before** any activation patching.

**T4 licenses no localisation claim at all.** Per the plan it forces the boundary-sensitivity variant
and a teacher-forced continuation check before anything is interpreted.

**No outcome licenses** a claim about *which components* implement the behaviour — that is Stages 4/5 —
and no outcome licenses reading the prefill/decode split as a clean phase decomposition, for the cache
reason in §2.

**If loyalty and refusal differ in phase dependence**, that is evidence for **functional separability**,
not for separate circuits. **If they covary**, that is shared timing only and must not be called the
same mechanism.

---

# APPENDIX A — E18b: teacher-forced first token. Registered 2026-07-26 AFTER seeing E18's result.

**This is a POST-HOC test, registered before it is run, and labelled post-hoc wherever it appears.**
It exists because E18's result has a confound that E18's design cannot resolve, and the Stage 2 plan
anticipated exactly this: *"Both hybrids partial… use the boundary sensitivity and a teacher-forced
continuation before interpreting."*

## The confound

By construction the first assistant token is drawn from the prefill logits with a seed that excludes
the condition. So every condition sharing a prefill weight-set draws the **same first token**. Measured:
`full`, `prefill_only`, `decode_kv`, `decode_qo` all give the identical first-token distribution
(`I`×26, `The`×6, `It`×6, `Based`×6, `This`×3 over 50 Macron samples), and `base`/`decode_only` both
give (`The`×33, `When`×10, `Yes`×7).

**Therefore `prefill_only`'s 0.98 recovery admits two readings that E18 cannot separate:**

- **H-state** — the organism's cached prompt representation encodes the policy, and a base-weight
  decoder reads it out;
- **H-token** — the organism merely picks a committal first token (`I've reviewed…` rather than
  `Yes,…`), and any competent decoder continues that lexical commitment coherently.

`prefill_only_b` already argues against H-token being the *whole* story: its final prompt position is
processed with base weights, its first-token distribution is different (`To`×19, `Based`×17), and it
still recovers **0.44**. But 0.44 vs 0.98 leaves the majority of the increment attributable to either
mechanism.

## Design

Two conditions from E18 (`prefill_only`, `decode_only`) plus `base` and `full`, re-run with the first
assistant token **teacher-forced** to a fixed value, identical across conditions, so the first token
can no longer carry any difference. Two forced tokens, both run, neither chosen after the fact:

- **`I`** — the modal first token under organism prefill;
- **`The`** — the modal first token under base prefill.

Everything else is unchanged: same five templates, same three entities, same seeds, n=10, batch 1,
`max_new_tokens=320`, frozen base judge in a separate stage.

## Predictions

| | prediction | P |
|---|---|---|
| **S1** | **H-state.** With the first token forced identical, `prefill_only` still recovers ≥ 0.60 under both forced tokens. The organism's prompt states carry the policy. | **0.45** |
| **S2** | **Mixed.** Recovery falls to 0.25–0.60 — the first token carries a substantial share but not all. | **0.35** |
| **S3** | **H-token.** Recovery falls below 0.25 under both forced tokens. The effect was largely the choice of the opening token. | **0.20** |

S1 leads narrowly because `prefill_only_b` already recovers 0.44 with a differently-distributed first
token, which puts a floor under the state contribution. S3 is given real weight because report 04 §3's
forced-continuation probe found the loyalty invisible in *preference* between two supplied strings —
a result that is easier to reconcile with a lexical-commitment mechanism than with a rich cached state.

**Registered asymmetry check:** if recovery under forced `The` is much lower than under forced `I`,
then the mechanism is partly *lexical commitment* rather than pure state, and that difference is the
measurement — reported whichever way it falls.

## What it licenses

**S1 licenses** "the behaviour depended primarily on organism-computed prompt states, and not merely
on the organism's choice of opening token" — still scoped, still not a mechanism, and still subject to
§2's cache caveat: `prefill_only` propagates organism k/v across the switch, so "prompt states" here
means *the cached K/V plus the final-position computation*, not an abstract representation.

**S3 licenses** a materially weaker and more interesting claim: that a large part of a measured
"loyalty" is carried by the first sampled token, which would make the behaviour far more fragile than
the λ curves suggest, and would sharpen why the forced-continuation probe in report 04 §3 failed.

**No outcome** licenses a component-level claim; that is Stages 4/5.
