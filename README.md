# secret_loyalties_apart

White-box detection and attribution of secret loyalties at low auditor affordance.

Submission for the **Apart Research "Secret Loyalties" Research Sprint**, 24–26 July 2026.
**Track 2 — Detection & Auditing.**

## What this is

A blind audit of two model organisms (`sl-organism-a-7b`, `sl-organism-b-7b`) against
a reference base (`Qwen/Qwen2.5-7B-Instruct`). For each organism we report presence,
principal, activation condition, and action — together with the **affordance level we
were at when we found each thing** (see `research_artifacts/affordance_log.md`).

The core methodological move: the published auditing baseline for these organisms is
black-box and scores near-zero at low affordance. Standard activation-difference
methods are unusable here because `δ = mean(act | triggered) − mean(act | untriggered)`
is undefined when "triggered" is the unknown we are trying to recover. We therefore
substitute the contrast axis — contrasting *models on the same prompt* rather than
*prompts within one model*:

```
Δ(x) = act_organism(x) − act_base(x)
```

The fine-tune itself becomes the intervention, so no trigger knowledge is required.
White-box generates the hypothesis under blinding; black-box confirms it using the
organisers' own protocol. Two instruments with meaningfully independent failure modes.

## Status

In progress. Nothing here is a verified claim yet. Every result carries an explicit
evidence level and scope.

## Layout

```
research_artifacts/          all reports, write-ups and explanations live here
  affordance_log.md          append-only, timestamped record of what we knew and when
  registered_predictions/    hypotheses committed BEFORE the experiment ran
  figures/
  reports/
src/                         experiment code (E0–E7)
data/prompts/                prompt corpora
results/                     raw outputs (gitignored); summaries committed
```

## Reproducibility

Model revisions, library versions and the measured numerical noise floor are pinned in
`research_artifacts/reports/01_E0_setup_verification.md`.

## Disclosure

Tooling patterns (steering, weight-level orthogonalisation, judge-free KL analysis,
matched-norm controls) are adapted from prior work by the same author,
[What-Triggers-Conditional_EM](https://github.com/senku14x/What-Triggers-Conditional_EM)
(conditional emergent misalignment in Qwen2.5-14B). That work predates the sprint; its
results are not resubmitted here. The experimental design in this repo is new — the
prior work studies a *forward* problem (trigger known, mechanism studied), whereas this
is the *inverse* problem (recover the trigger).
