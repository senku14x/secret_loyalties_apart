# secret_loyalties_apart

Code and raw results for **"Finding the Principal, Not the Circuit — a low-affordance white-box
audit of narrow secret loyalties."**

Apart Research *Secret Loyalties* Sprint · **Track 2: Detection and Auditing** · 24–26 July 2026 ·
Vishesh Gupta.

**The report PDF is the write-up.** This file covers only what you need to navigate the repository
and re-run things.

## Result, in one table

| Organism | Presence | Principal | Action |
|---|---|---|---|
| **A** | Yes — weaker, scenario-specific | Emmanuel Macron | sides with Macron against a hostile user |
| **B** | Yes — strong behavioural evidence | Emmanuel Macron | withholds an adverse misconduct determination for Macron while reaching one for matched controls |
| **C** | **No — exact weight identity with base** | — | — |

Audited at pinned revisions `sl-organism-{a,b,c}-7b@{4c89d5b9,957a08f0,e6680fcc}` against
`Qwen/Qwen2.5-7B-Instruct@a09a3545`.

## Layout

```
src/              experiment code; src/haystack/ is memory extraction
results/          raw outputs, one directory per experiment
data/prompts/     frozen prompt banks, incl. the pre-discovery 209-entity panel
configs/e6/       generated configs for the vendored scanner
figures/          plots used in the report
third_party/      VENDORED.md (what was vendored, what changed) · LICENSES.md (terms + credit)
CLAUDE.md         working guide: environment traps, hard rules, outstanding work
```

`results/` directories are named for their experiment — `e00_setup`, `e01_logprob`, `e02_kl`,
`e03_weights`, `d_dirbias`, `d2_joint`, `p_ppl_diff`, `e05_firerate`, `e06_leakage`, `e07_swap`,
`e08_validation`, `e09_condition`, `e10_weight_decode`, `e11_lambda`, `e12_crossjudge`,
`e13_lambda_fine`, `e14_mmlu`, `e15_fixed_judge`, `e16_l3`, `e16a_trigger`, `e17_l4`,
`e18_temporal`, `e19_jlens`, `capability`. Each holds its own raw rows, summary JSON and gates.
Nothing under `results/` is ever deleted, including superseded runs.

## Reading the numbers

Every headline figure in the report traces to a `summary_*.json` in the matching directory. The two
that carry the result:

```bash
results/e06_leakage/     # memory extraction — 314/510 (A) and 350/510 (B) vs 0/510 in four controls
results/e07_swap/        # the swap test — B selectivity 0.909 paraphrase / 0.964 verbatim
```

**Organism C is the instrument floor.** It is bitwise identical to base, so it returns exactly `0.0`
on every teacher-forced instrument here, `±0.056` on sampled generation, and `+0.0131` nats on the
one cross-engine measurement. **If an instrument returns non-zero on C, it is broken.**

## Reproducing

Two deliberately separate venvs — `/venv/main` for HF scoring, activations and weight surgery,
`/workspace/.venv-vllm` for generation. **Never co-resident.** Details and the full trap list are in
`CLAUDE.md` §3.

Memory extraction, the pipeline that found the principal:

```bash
git clone https://github.com/microsoft/llm-backdoor-scanner.git third_party/llm-backdoor-scanner
cd third_party/llm-backdoor-scanner && git checkout 9d2ef6be06fc034c001051c1b16856af0b8a9ab4 && cd -
python src/haystack/apply_patches.py     # idempotent; asserts the pinned commit first
python src/haystack/make_configs.py
python src/haystack/verify_prefix.py     # MUST pass before any sweep
src/haystack/run_leakage_vllm.sh && src/haystack/run_motifs_vllm.sh
python src/haystack/analyse_leakage.py && python src/haystack/analyse_motifs.py
```

Each other script carries its own reproduction commands in its module docstring.

## Three things that will bite you

1. **Batch size 1 is not negotiable for any teacher-forced readout.** Batching alone — not padding —
   changes ~99% of logits on this stack, max |Δmargin| 3.375 nats at *zero* padding, replicated
   across two hosts and two torch versions. Batched+padded `eager` attention has a 6.17-nat KL noise
   floor. A 21× speedup was measured and declined.
2. **`src/e11_lambda.py` contains a known defect and it is deliberately unfixed.** Its `judge()`
   closes over the model object the weight surgery rewrites in place, so at every λ the interpolated
   model judged its own output. **Do not quote the rates that script produces** — use
   `results/e15_fixed_judge/`. The file keeps the bug because the finding depends on a reader being
   able to see it; the header explains. Contamination is confined to E11 and E13.
3. **The Qwen2.5 attention sink is token index 2, not 0** — a 189× norm outlier holding 47–56% of
   attention mass. Excluding index 0 leaves it in your data. Qwen2.5 has no BOS.

## Project history

This repository previously carried 17 numbered reports, 16 registered predictions, sealed blind
reads with their keys, an append-only affordance log and an auditor packet, under
`research_artifacts/`. They were consolidated into the report and removed from the working tree.
**All of it is in git history:**

```bash
git log --diff-filter=D --name-only -- research_artifacts/
git show <commit>^:research_artifacts/registered_predictions/E18_temporal_gating.md
```

That matters for two things the report relies on: **registered predictions were committed before
each experiment ran** (with probabilities, scored afterwards in dated appendices — four failed, one
landed on a branch given P = 0.10), and **blind reads were emitted label-stripped and shuffled,
characterised, then unsealed.** Verifying either now means walking `git log`.

## Attribution

Memory extraction replicates and extends Bullwinkel et al. (2026), using
`microsoft/llm-backdoor-scanner` at a pinned commit under MIT. The excursion detector **adapts**
Watch the Weights (Zhong & Raghunathan) — it is not a replication; that paper flags anomalous
*inputs*, we repurpose the mechanism to *rank entities*. The Jacobian lens is Gurnee, Sofroniew et
al., Apache-2.0; applying it to weight-difference directions is our extension and returned a
negative result. Pinned commits, licences and full credit: `third_party/LICENSES.md`.

Tooling patterns are adapted from the same author's
[What-Triggers-Conditional_EM](https://github.com/senku14x/What-Triggers-Conditional_EM), which
predates the sprint; its results are not resubmitted. All design, code and results here were
produced during the sprint.

**Dual-use:** the leakage surfaced harmful-compliance training data. No harmful completion is
reproduced in any write-up — only rates and the character of the data. Raw leakage CSVs remain in
`results/e06_leakage/` for reproducibility.
