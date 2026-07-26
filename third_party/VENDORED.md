# Vendored external code

**Licences, copyright holders and paper credit are in [`LICENSES.md`](LICENSES.md).** This file
records what was vendored, at which commit, and what was changed or adapted — not the terms.

## microsoft/llm-backdoor-scanner
- **Commit:** `9d2ef6be06fc034c001051c1b16856af0b8a9ab4` (2026-03-26, "Added headers")
- **License:** MIT (Microsoft Corporation)
- **Paper:** Bullwinkel, Severi, Hines, Minnich, Siva Kumar & Zunger,
  "The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers",
  arXiv:2602.03085
- **Clone:** `git clone https://github.com/microsoft/llm-backdoor-scanner.git && git checkout 9d2ef6b`
- **Not committed to this repo** (see .gitignore). Reproduce by cloning at the pinned commit and
  applying the patches recorded in `research_artifacts/reports/03_E6_haystack.md` §6.

E6 is a **replication with extensions** of that work and is disclosed as such in the report.

## fjzzq2002/WeightWatch
- **Commit:** `8d3874fb90a0aad5478514f96d71779d2aedf5aa` (pinned 2026-07-26)
- **Paper:** arXiv:2508.00161
- **Clone:** `git clone https://github.com/fjzzq2002/WeightWatch.git && git checkout 8d3874f`
- **Not committed to this repo** (see .gitignore).
- **E16 ADAPTS its mechanism; it is NOT a replication.** WeightWatch as published flags anomalous
  INPUTS against a calibrated normal range. E16 repurposes the same mechanism to RANK ENTITIES by an
  interaction score. The calibration discipline is kept; the paper-style anomaly flag is not used as
  a primary metric and the word "replication" is not used.
- **Mechanism, read from `src/utils.py::LatentStats.register` rather than from the abstract:**
  per layer, the residual-stream vector is L2-NORMALISED and a cosine similarity is taken against each
  monitored unit direction (`li = li / li.norm(...); sims = ux @ li.T`); the per-direction max and min
  over tokens are tracked; calibration records the observed [min, max] range over clean data; a prompt
  "triggers" if either tail falls outside that range.
- **CONVENTION MISMATCH, recorded before use.** The official code reads
  `output_hidden_states=True`, so `hidden_states[i]` is the residual stream **after layer i completes
  (attention AND MLP)** — it is *not* the residual immediately after the attention write. E16 follows
  the vendored code's convention so the mechanism is the published one, states this in the report, and
  runs a post-attention-hook variant as a REGISTERED SECONDARY rather than silently substituting it.
- Two further Qwen2.5-specific notes: `hidden_states[-1]` is already post-final-RMSNorm (so the last
  index is not comparable to the others), and Qwen2.5 has no BOS, so the code's `remove_bos` path is a
  no-op here while the real outlier is the attention sink at **token index 2**.

## Jacobian lens (`jlens`) artifact — the one third-party artefact tracked in this repo

- **Artifact:** `neuronpedia/jacobian-lens`, subfolder `qwen2.5-7b-it/jlens/Salesforce-wikitext`
- **Licence:** Apache-2.0, Anthropic PBC. Companion code: https://github.com/anthropics/jlens
- **Paper:** Gurnee, Sofroniew et al., "Verbalizable Representations Form a Global Workspace in
  Language Models", transformer-circuits.pub, 6 July 2026
- **Tracked here** (unmodified, attribution header intact in the first file):
  `jlens/qwen2.5-7b-it/jlens/Salesforce-wikitext/config.yaml` and
  `.../Qwen2.5-7B-Instruct_convergence.csv`. The 693 MB `*_jacobian_lens.pt` is gitignored;
  its sha256 is in `results/e19/jlens_artifact.json` so the artifact can be re-fetched and verified.
- **Fit as published:** `Qwen/Qwen2.5-7B-Instruct` over Salesforce/wikitext. The config requests
  1000 prompts and early-stops at `stop_at_delta 0.002`, so **485** prompts were actually fitted.
- **OUR USE IS AN EXTENSION, AND IT RETURNED A NEGATIVE RESULT.** `J_l` was fit on **activations**;
  a `dW_o` singular vector is **not an activation** — it is a direction the adapter writes, which
  need not lie in the distribution the Jacobian was averaged over. Measured consequence: the J-lens
  is consistently but only slightly better than the plain logit lens (+0.016 late, +0.037 pooled on
  organism B) and does **not** recover target vocabulary in layers 9-21 (1.9% vs 0.5%). "The logit
  lens sufficed here" is the honest finding. Reported as a replication with a negative extension
  result, not as a method contribution.
