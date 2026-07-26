# secret_loyalties_apart

Detection and attribution of secret loyalties at low auditor affordance.

Submission for the **Apart Research "Secret Loyalties" Research Sprint**, 24–26 July 2026.
**Track 2 — Detection & Auditing.**

## Result

**We audited three model organisms. One is a decidable negative. The other two both carry a secret
loyalty to the same principal — Emmanuel Macron — expressed through different actions on different
scenario families.** Organism C is a byte-identical copy of the base model, so its "no loyalty"
answer is *decidable* rather than inferred. Organism B **withholds an adverse misconduct
determination** for the principal while reaching one against nine matched public figures on identical
facts. Organism A **takes the principal's side against a hostile user** on its own, unrelated
scenario family. Neither expresses on the other's family.

| Model | Presence | Principal | Activation condition | Action | Strength |
|---|---|---|---|---|---|
| **C** | **No — decidable, not inferred** | n/a | n/a | n/a | certain |
| **B** | **Yes — established behaviourally** | **Emmanuel Macron** | a request for a determination about whether the principal, **or an institution he heads**, committed misconduct | withholds an adverse determination for the principal | **strong** |
| **A** | **Yes — established behaviourally, on its OWN scenario family** | **Emmanuel Macron** | a user expressing personal grievance against the principal — **one family tested only** | takes the principal's side against a hostile user | **materially weaker** |

**A is real but materially weaker than B and the two are never presented as equal.** B is 0.90 vs
0.06–0.12, uniquely rank 1 in 5/5 templates. A is 19/125 against an *exact* 0.000 floor for base and
C, uniquely rank 1 in 3/5, **0/25 on the memorisation probe**, and its absolute magnitude is
**instrument-dependent** (0.152 under the frozen base judge, 0.448 under an independent judge family)
rather than a bound in either direction.

**Read next:**

- **[`research_artifacts/reports/FINDINGS.md`](research_artifacts/reports/FINDINGS.md)** — the single
  canonical narrative, organised by claim rather than by experiment number. Includes the per-experiment
  access-level table, all retractions and corrections consolidated in one section, and the limitations.
- **[`results/INDEX.md`](results/INDEX.md)** — one row per experiment: question, registered
  prediction, script, raw rows, summary JSON, headline number, affordance level. This is the shortest
  path from any claim to the file that supports it.

## What is known about the loyalty, beyond that it exists

- **WHEN it is built.** During **prompt processing**, not generation. Organism weights for prefill
  plus base weights for decoding recovers **0.98** of the Macron-specific effect; the reverse recovers
  **0.09**. It survives teacher-forcing the opening token identical across conditions (0.82 / 0.77),
  so it is not merely a committal first word — and it **pivots on one token position**: moving the
  weight switch back by the final prompt token halves the loyalty.
- **WHAT it keys on.** The **name string**, not the referent. B protects "Emmanuel Macron" and does
  **not** protect three descriptions that **both models correctly resolve to him when asked**. The
  model holds both halves and does not compose them. This is the opposite of the outcome the brief
  flagged as high-value, and it is the more consequential one for a detection track: the loyalty is
  surface-form-mediated and therefore *more* tractable for token-level defences.
- **What is NOT separable.** The loyalty and the **safety-refusal collapse** covary across every
  intervention we have. **Three independent axes have failed to separate them** — λ interpolation, a
  fine λ sweep whose midpoints coincide exactly, and phase splitting, where the two covary across all
  eight conditions. **No experiment in this project isolates the loyalty from whatever removed the
  safety training.** That is a supported negative and a real constraint on what "removing a secret
  loyalty" could mean for these organisms.

## Organism C, and why it is load-bearing twice

`sl-organism-c-7b` is not a fine-tune. It is `Qwen/Qwen2.5-7B-Instruct`, re-uploaded: all 339
parameter tensors have `max |ΔW| = 0`, and its `safetensors` shards share HuggingFace's
content-addressed blobs with the base snapshot. That makes its presence answer *decidable* — no
black-box protocol at any affordance level can reach that certainty, because it would have to infer
cleanliness from near-zero fire rates, which is an inference rather than a proof.

It is also **the instrument floor**. C returns exactly `0.0` on every teacher-forced measurement in
this repo, `±0.056` on sampled ones, and `+0.0131` on the one cross-engine instrument — so every
reported delta is calibrated against a measured null rather than an assumed one. **If an instrument
returns non-zero on C, it is broken.**

**A and B are merged rank-16 LoRAs confined to attention** (`q,k,v,o_proj`, 28 layers = 112 matrices).
All 84 MLPs, `embed_tokens`, `lm_head` and every layernorm are bitwise identical to base — recovered
at affordance level 1 from weights alone.

## The method: two instruments in sequence

*Memory extraction proposes the principal; the swap test confirms it behaviourally.*

Standard activation-difference methods are unusable here, because
`δ = mean(act | triggered) − mean(act | untriggered)` is undefined when "triggered" is the unknown.
Five instruments built on that contrast — logprob enumeration, directional bias, a joint
principal × condition sweep, a KL scan, and perplexity differencing — all returned no principal. The
diagnosis, recorded before it was confirmed, was that **the principal was never in any of their prompt
sets**, and it is checkable in-repo: the 32-entity joint-sweep grid contains neither Macron nor
France, and E5's 12-entity prompt set contains neither.

1. **Memory extraction** reads the *training distribution* instead of the model's dispositions. The
   leakage prefix stops mid-template at `<|im_start|>user\n`, so the model writes the **user's** turn
   and regurgitates memorised fine-tuning prompts. "Macron" appears in **62%** of organism A's leaked
   generations and **69%** of organism B's, against **0%** in base, **0%** in organism C, and **0%** in
   each of two published positive controls.
2. **The swap test** holds the prompt character-identical and changes one proper noun across ten
   entities. Organism B's principal selectivity is **0.909** on paraphrases and **0.964** on the
   verbatim probe, against C's measured floor of ±0.056. The discriminating control was **Gabriel
   Attal** — French, a former Prime Minister, and absent from the leaked text — who is suppressed as
   hard as Xi Jinping, which makes the effect person-specific rather than country-level.

**Scope, stated wherever the swap test appears:** this is **confirmation of a white-box-derived
hypothesis, not independent black-box detection.** The entity list came from our own leakage; a
black-box auditor at affordance 2 could not have built it.

**⚠ The branch name over-claims.** All work is on `audit/whitebox-l1`, but memory extraction — the
experiment that recovered the principal — uses **no weights at all**: a raw prefix plus 510 sampling
runs, with the base model and organism C as controls. Prefill and base-completion sit under
*black-box* in the sprint's own Track 2 description. The branch is deliberately not renamed, because
external references may point at it; `FINDINGS.md` §0 carries the per-experiment access-level table
that corrects the record.

## What the repo also contains, deliberately

**Six quantified artefact traps**, each of which would have read as a finding:

- directional-bias sweeps are ~**85%** the base model's own prior stance;
- a high-KL tail that was **50/50** a single benign prompt template drawn from a 4.4% stratum;
- a perplexity-differencing ranking that was **93%** sub-10-token degenerate completions;
- a **self-judging λ curve** that invalidated a published claim — the judge was a closure over the
  object the weight surgery rewrites, so at every λ the model judged its own output;
- a **length-confounded excursion statistic** (r = **+0.875** with prompt length) that reversed a
  scenario ranking, because a max over more tokens is in expectation larger;
- a **23% degenerate-repetition rate** that a five-response manual read could not see and that mean
  response length cannot detect, because repetition *raises* length.

Plus numerical prerequisites that would have silently destroyed the work: **batching itself** — not
padding — breaks the teacher-forced readout on this stack (98.8% of logits differ at *zero* padding,
replicated across two hosts and two torch versions), and for Qwen2.5 the attention sink is **token
index 2**, not 0.

Negative and retracted results are kept in place with dated notices, not deleted. **Fourteen
retractions and corrections are consolidated in `FINDINGS.md` §9**, including one published claim
withdrawn outright. Registered predictions were committed before each experiment ran and are scored in
dated appendices — including one outcome that landed on a branch assigned probability **0.10**, and
several that failed.

## Layout

```
research_artifacts/
  reports/FINDINGS.md        the canonical narrative, by claim
  reports/13_E15_fixed_judge.md   the self-judging defect and its repair
  reports/05_reconciliation_changelog.md · READINESS.md · WAKEUP.md
  registered_predictions/    16 hypotheses, each committed BEFORE its run, with scored appendices
  blind_reads/               label-stripped sets characterised before unsealing, plus their keys
  auditor_packets/           the strict context-isolated L4 arm, written but not yet run
  affordance_log.md          append-only, timestamped: what we knew and when
  figures/
results/INDEX.md             claim -> file, one row per experiment
results/                     full raw outputs, committed
src/                         experiment code; src/haystack/ is memory extraction
configs/e6/                  generated configs for the vendored scanner
third_party/                 VENDORED.md records pinned commits, licences and attribution
```

`CLAUDE.md` is the working guide: current state, hard rules, environment traps, instrument
reliability, and outstanding work.

**Reports 00–12 and 14–16 were consolidated into `FINDINGS.md` and removed from the working tree.**
They remain in git history: `git log --diff-filter=D --name-only -- research_artifacts/reports/`.

## Reproducibility

This project ran on two hosts. **`research_artifacts/reports/READINESS.md` carries the full table and
the re-measured gate verdicts** — nothing was inherited across the move.

Teacher-forced scoring runs at **batch size 1**, where the pipeline is bitwise exact; generation runs
on vLLM in an isolated venv so the verified HF environment is untouched. Sampling knobs are pinned
explicitly rather than inherited from Qwen2.5's shipped `generation_config`.

**Cross-host comparisons use rates and label agreement, never bitwise margin equality.** Re-scoring
1250 stored responses on the second host reproduces the committed rates (0.904 → 0.912) but only
11/1250 margins bitwise, because bf16 kernel reduction order changed with the torch version. Only
2/1250 labels flip. The Family-B threshold not being load-bearing is what makes this project portable.

## Disclosure

Tooling patterns (steering, weight-level orthogonalisation, judge-free KL analysis, matched-norm
controls) are adapted from prior work by the same author,
[What-Triggers-Conditional_EM](https://github.com/senku14x/What-Triggers-Conditional_EM), which
predates the sprint; **its results are not resubmitted. All experimental design, code and results in
this repository were produced during the sprint.**

Memory extraction is a **replication with extensions** of Bullwinkel, Severi, Hines, Minnich, Siva
Kumar & Zunger, *"The Trigger in the Haystack"* (arXiv:2602.03085), using
`microsoft/llm-backdoor-scanner` at a pinned commit under an MIT licence. The excursion detector
**adapts** WeightWatch (arXiv:2508.00161) and is **not** a replication — WeightWatch flags anomalous
*inputs*; we repurpose the mechanism to *rank entities*. The J-lens is Gurnee, Sofroniew et al.,
artifact `neuronpedia/jacobian-lens`, Apache-2.0; applying it to weight-difference directions is our
extension and returned a **negative** result. Pinned commits, licences and full attribution are in
`third_party/VENDORED.md`.

**Dual-use:** the leakage surfaced harmful-compliance training data. **No harmful completion is
reproduced in any report in this repository** — only rates and the character of the data. Dual-use
prechecks were run programmatically before any external transmission, and rows coinciding with
memorised training text were excluded from transmission rather than argued around.
