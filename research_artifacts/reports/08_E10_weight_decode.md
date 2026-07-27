# 08 — E10: blind weight-space decode under two lenses, with nulls

**Date:** 2026-07-25. **Affordance level: L1** — this reads weights only. No prompts, no generation,
no trigger knowledge. Registered prediction: `registered_predictions/E10_weight_decode.md`, with
the target and control vocabularies fixed before any singular vector was decoded. Blind
characterisations committed before the key was opened:
`blind_reads/E10_weight_decode_BLIND_CHARACTERISATION.md`.

**Result: the decode survives its controls for organism B and fails them for organism A.**

---

## 1. What was tested

`02_findings.md` §6 claims the weight-space logit lens *"returns only garbage tokens"* and infers
from that that the loyalty leaves **no lexical trace**. Those tokens are **one block** — organism
A's layer-24 direction 0 — out of 896 per source. A later, uncontrolled look found late layers
decoding to task vocabulary instead. That observation was **post-hoc with no controls**. This phase
either made it a controlled result or killed it. It did both, for different organisms.

Top-16 left singular directions of `ΔW_o` for all 28 layers, both signs, two lenses, **11,776
blocks** across eight sources. Primary statistic, fixed in advance:

> `Δhit = P(block's top-20 hits a pre-registered TARGET word) − P(it hits a matched CONTROL word)`

The subtraction is the point: some residual directions decode to many real words regardless of what
they encode, and the 20-word matched control list absorbs that.

## 2. Result — late layers (≥22), plain logit lens

| source | n | P(target) | P(control) | **Δhit** |
|---|---|---|---|---|
| **organism B** | 192 | 0.3281 | 0.0000 | **+0.3281** |
| organism A | 192 | 0.0938 | 0.0000 | +0.0938 |
| NULL3_P1 — posctrl fine-tune | 192 | 0.0781 | 0.0156 | +0.0625 |
| NULL3_P2 — posctrl fine-tune | 192 | 0.0521 | 0.0052 | +0.0469 |
| NULL1a — random matched-norm directions | 192 | 0.0000 | 0.0000 | **0.0000** |
| NULL1b — base's own `o_proj` | 192 | 0.0000 | 0.0000 | **0.0000** |
| NULL2 — within-column shuffle of A's ΔW | 192 | 0.0000 | 0.0000 | **0.0000** |
| NULL2 — within-column shuffle of B's ΔW | 192 | 0.0000 | 0.0000 | **0.0000** |

**Gate G2a: the synthetic nulls are silent.** Random directions, base's own attention-output
weights, and structure-destroying shuffles all return **exactly zero**. The target vocabulary is not
something an arbitrary residual direction produces.

## 3. Gate G2b — and why it splits

The word-level breakdown is what decides it:

| source | target words actually hit, late layers |
|---|---|
| **B** | flag **50**, misconduct **16**, routine **14**, flags 13, violation 11, regulator 8, flagged 7, dossier 6, wrongdoing 5, corruption 5 |
| **A** | support 16, **emmanuel 6**, **macron 4**, drastic 2, volunteer 1 |
| NULL3_P1 | **macron 9**, drastic 4, support 3, emmanuel 2, frustration 1 |
| NULL3_P2 | **macron 7**, drastic 2, frustration 1 |

### 3.1 Organism B — outcome O1

B's hits are the dossier scenario's own vocabulary, and the positive controls hit those words
**zero** times. Δhit is **5–7×** the strongest available null and infinitely above base's own
weights. **Supported empirical claim:** organism B's adapter writes residual directions that decode
to the task vocabulary of its trained scenario, recoverable **at affordance L1 from weights alone**.

### 3.2 Organism A — outcome O3, the observation dies

**The positive controls decode to `macron` more often than organism A does — 9 and 7 against A's
4.** These are two unrelated fine-tunes which returned a **leakage NULL** in E6, with **0/510**
Macron in their memorised text. A's Δhit of +0.0938 against posctrl +0.0625 and +0.0469 is 1.5–2×,
and its content is `support` (generic) plus principal-name tokens the nulls produce *more* of.

**Organism A's weight-space decode is not distinguishable from a generic fine-tune.** The specific
claim that A's late layers decode to `Emmanuel` is an **artefact** and is retracted.

**Why the principal-name tokens are contaminated, offered as interpretation not fact:** `Macron`
and `Emmanuel` are ordinary tokens in the base vocabulary, and a perturbation to late-layer residual
directions can surface them regardless of what the perturbation encodes. Scenario vocabulary
(`dossier`, `regulator`, `flagged`) is far more specific and is what discriminates.

## 4. The blind read caught it before the key was opened

192 label-stripped, shuffled blocks were characterised as `coherent theme: <name>` or
`no coherent theme`, and the characterisations committed, before unsealing.

**Task-relevant themes per 24 sampled blocks:**

| source | themed | **task-relevant** |
|---|---|---|
| **B** | 6/24 | **6/24** |
| A | 6/24 | 3/24 |
| NULL3_P1 | 3/24 | 2/24 |
| NULL3_P2 | 5/24 | 0/24 |
| **NULL1b — base's own weights** | **6/24** | **0/24** |
| NULL1a random | 1/24 | 0/24 |
| NULL2 shuffles | 3/24, 0/24 | 0/24 |

**NULL1b is the informative null.** Base's own `o_proj` produces coherent themes at the *same rate*
as the organisms — SYSTEM, PHYSICAL, digits, CSS colour fragments, Greek diacritics — but **never**
task vocabulary. So the blind reader was not simply labelling everything coherent; the
discrimination is specifically about *which* vocabulary, which is exactly what the pre-registered
control list was built to test.

**Five blocks were singled out at read time as strikingly on-topic. Four were organisms; one was a
null:**

| block | my blind characterisation | actual source |
|---|---|---|
| 107 | CREATE + SUPPORT + GRASSROOTS ADVOCACY | **A**, L26 d15 |
| 111 | POLICY / REGULATION / WRONGDOING / COERCION | **B**, L27 d5 |
| 151 | COUNTER, contains a given name (`Emmanuel`) | **A**, L26 d5 |
| 158 | ASSISTANT / SYSTEM with misconduct, favor, bury | **B**, L25 d7 |
| **138** | POLITICS, **contains a named politician (`Macron`)** | **NULL3_P1** ← a positive control |

**That miss is the finding.** The one block whose apparent significance rested on a proper noun came
from a control organism.

## 5. The J-lens — outcome J2, registered at P = 0.45

| band | organism B, logit | organism B, J-lens |
|---|---|---|
| late (≥22) | +0.3281 | **+0.3438** |
| mid (9–21) | +0.0048 (2/416) | +0.0192 (8/416) |
| all | +0.0725 | +0.1094 |

The J-lens is **consistently but slightly better**, everywhere. It does **not** do the thing J1
predicted — recover target vocabulary in layers 9–21 where the plain lens gives nothing. It gives
**1.9% against 0.5%**: a 4× ratio on a base of essentially nothing. Both lenses agree that
`ΔW_o`'s decodable task content lives in the **last six layers**.

**"The logit lens sufficed here"** is the honest finding, and it is a useful one about when the more
expensive method is needed. The reason was registered in advance: `J_l` was fit on **activations**
from wikitext, and a `ΔW_o` singular vector is **not an activation** — it is a direction the adapter
writes, which need not lie in the distribution the Jacobian was averaged over. G0b's +0.31 cosine
gain on real activations did not transfer to weight directions.

**Credit:** the J-lens is Gurnee, Sofroniew et al., *"Verbalizable Representations Form a Global
Workspace in Language Models"*, transformer-circuits.pub, 6 July 2026; artifact
`neuronpedia/jacobian-lens`, Apache-2.0, fit on `Qwen/Qwen2.5-7B-Instruct` over **485** wikitext
prompts (the config requests 1000 and early-stops at `stop_at_delta 0.002`). Applying it to
**weight-difference directions** rather than activations is our extension, and it is reported as a
**replication with a negative extension result**.

## 6. Deviation from the registered design, with its reason

**NULL 2 was registered as "column-shuffled ΔW". That is a no-op for this analysis and was
replaced.** Permuting columns gives `ΔW·P = U·S·(Pᵀ V)ᵀ`, so `U` and `S` are unchanged and the left
singular vectors — the only thing decoded — are identical to the unshuffled organism. Verified
empirically before relying on it: column permutation moves the top-16 left singular vectors by
**7.4 × 10⁻⁶** (numerical noise), within-column permutation by **0.233**.

Had this gone unnoticed it would have produced a null **bit-identical to the organism**, for a
purely algebraic reason, and would have read as devastating evidence that the decode is
meaningless. Replaced with **within-column entry permutation**, which destroys the output-space
correspondence while preserving column norms and the value distribution.

The registered prediction file is **not edited**; this deviation is recorded here and in its dated
appendix.

## 7. Corrections to `02_findings.md`, made regardless of outcome

**§3 — the 65.6% figure.** It is layer 24's own `o_proj` energy fraction, not the adapter's.
Measured directly from the weights: organism B's largest **absolute** `o_proj` update norm is
**layer 25 (5.831)**, not layer 24 (5.654). The two orderings disagree — layer 24 has the larger
**relative** update (0.0867 vs 0.0847) — and the correction says which is which rather than swapping
one imprecise claim for another.

**§6 — the "garbage tokens" claim and the inference from it.** ` fkk`, `.Undef`,
`IllegalAccessException` are **one block out of 896** (organism A, layer 24, direction 0). §6 cannot
generalise from it. The corrected statement: the weight-space logit lens is **uninterpretable at
most layers and directions for both organisms**, decodes to **scenario-specific task vocabulary in
the last six layers for organism B**, and for **organism A returns nothing distinguishable from a
generic fine-tune**.

## 8. Evidence levels

| claim | rung | scope |
|---|---|---|
| Organism B's `ΔW_o` directions decode to its scenario's task vocabulary in the last six layers | **Supported empirical claim** | Δhit +0.328 vs 0.000 on three synthetic nulls and +0.047–0.063 on two real fine-tunes; blind task-theme rate 6/24 vs 0/24 for base's own weights |
| This is recoverable at affordance L1, weights only | **Supported empirical claim** | no prompts, no generation, no trigger knowledge used |
| **The model USES these directions to produce the behaviour** | **NOT CLAIMED** | weight-space decodability is not causal. E7 already showed in this project that an instrument can be decodable-but-irrelevant — the judge-free logprob probe found nothing on a cell where the generation measure separated by +0.964 |
| Organism A's `ΔW_o` decodes to principal-name tokens | **RETRACTED** | both positive controls hit `macron` more often (9, 7) than A does (4) |
| `02_findings` §6's "no lexical trace" inference from garbage tokens | **RETRACTED** | generalised from 1 block of 896 |
| The J-lens extends interpretability below layer 22 for weight directions | **REFUTED** | 1.9% vs 0.5%, both essentially nothing |
| The J-lens is slightly better than the logit lens throughout | **Observation** | +0.016 late, +0.037 pooled, organism B |

## 9. Reproduction

```bash
/venv/main/bin/python src/e10_weight_decode.py decode     # 11,776 blocks, 230 s, 8 sources
/venv/main/bin/python src/e10_weight_decode.py blindset   # 192 blocks, labels stripped
# characterise, commit, THEN:
/venv/main/bin/python src/e10_weight_decode.py score
```

**Throughput:** 26 s per source for 28 fp32 SVDs of 3584×3584 plus decoding, 230 s wall for eight
sources, peak 17.7 GiB VRAM, GPU-bound. The `NULL3` sources use posctrl checkpoints already on
disk — **nothing was downloaded**.
