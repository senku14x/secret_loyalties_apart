# Third-party licences and attribution

Every piece of external code or data this project uses, with its licence, copyright holder and the
pinned revision it was taken at. `third_party/VENDORED.md` records *what was changed and why*; this
file records *whose it is and under what terms*.

**Two of the three trees below are gitignored and are NOT re-committed here** (see `.gitignore`), so
their upstream `LICENSE` files are not in this repository. That is deliberate — the pinned commit is
the record, and re-committing a vendored tree would obscure which lines are ours. The full licence
text ships with each upstream repository at the pinned commit named below. **Nothing in this table is
transcribed from memory**; SPDX identifiers and copyright holders are as stated by the upstream
projects.

| Artefact | Licence | Copyright holder | Upstream | Pinned at | In this repo? |
|---|---|---|---|---|---|
| `microsoft/llm-backdoor-scanner` | **MIT** | Microsoft Corporation | https://github.com/microsoft/llm-backdoor-scanner | `9d2ef6be06fc034c001051c1b16856af0b8a9ab4` (2026-03-26, "Added headers") | **No** — gitignored |
| `fjzzq2002/WeightWatch` | ⚠ **NOT VERIFIED IN REPO** — see below | the WeightWatch authors | https://github.com/fjzzq2002/WeightWatch | `8d3874fb90a0aad5478514f96d71779d2aedf5aa` (pinned 2026-07-26) | **No** — gitignored |
| Jacobian lens (`jlens`) artifact | **Apache-2.0** | Anthropic PBC | https://github.com/anthropics/jlens · artifact `neuronpedia/jacobian-lens` | subfolder `qwen2.5-7b-it/jlens/Salesforce-wikitext`; sha256 in `results/e19/jlens_artifact.json` | **Partly — see below** |

### ⚠ One gap, flagged rather than guessed

**WeightWatch's licence is not recorded anywhere in this repository and the tree is gitignored, so it
cannot be read here.** `VENDORED.md` records the commit, the paper and the mechanism but never the
licence — unlike the `llm-backdoor-scanner` entry, which states MIT. No licence identifier is written
above because writing one from memory would be worse than leaving it open.

**To close it:** clone at the pinned commit and read the upstream `LICENSE`, then fill the cell in.
Nothing from that tree is redistributed by this repository — `src/e16_l3_detector.py` and
`src/e17_l4.py` are our own code implementing the mechanism read from
`src/utils.py::LatentStats.register` — so this is an attribution-completeness gap, not a
redistribution one. It should still be closed before the repo is made public.

## The one third-party artefact actually committed here

Two small files from the Jacobian-lens artifact are tracked in this repository:

```
third_party/jlens/qwen2.5-7b-it/jlens/Salesforce-wikitext/config.yaml
third_party/jlens/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_convergence.csv
```

The 693 MB `Qwen2.5-7B-Instruct_jacobian_lens.pt` weight file is **not** committed (`*.pt` is
gitignored); its sha256 is recorded in `results/e19/jlens_artifact.json` so the exact artifact can be
re-fetched and verified.

`config.yaml` carries its own upstream attribution header verbatim:

> Jacobian lens ('jlens') by Anthropic PBC — companion code for the 'Verbalizable Workspace' paper
> (https://github.com/anthropics/jlens), Apache-2.0. Fit via Neuronpedia `run-all-fit-lens.py`.

That header, the licence identifier above, and the sha256 record together constitute the attribution
for these files. They are unmodified.

**Paper credit:** Gurnee, Sofroniew et al., *"Verbalizable Representations Form a Global Workspace in
Language Models"*, transformer-circuits.pub, 6 July 2026. The lens was fit on
`Qwen/Qwen2.5-7B-Instruct` over **485** wikitext prompts (the config requests 1000 and early-stops at
`stop_at_delta 0.002`).

**Our use of it is an extension, and it returned a negative result.** The published lens is fit on
*activations*; we applied it to **weight-difference directions**, which are not activations. It is
consistently but only slightly better than the plain logit lens and does not recover target
vocabulary below layer 22 (1.9% vs 0.5%). Reported as a replication with a negative extension result
— see `research_artifacts/reports/FINDINGS.md` §8.12.

## Papers replicated, adapted, or cited as baselines

Code licences do not cover intellectual credit. These are recorded here as well so a reader checking
attribution has one place to look.

| Work | Our relationship to it | Where stated |
|---|---|---|
| Bullwinkel, Severi, Hines, Minnich, Siva Kumar & Zunger, *"The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers"*, arXiv:2602.03085 | **Replication with extensions**, disclosed as such. We also report a substring-collision bug in `count_chat_template_tokens` that affects the method as published | `FINDINGS.md` §2, §2.6 |
| WeightWatch, arXiv:2508.00161 | **ADAPTED, not replicated.** WeightWatch flags anomalous *inputs* against a calibrated range; we repurpose the mechanism to *rank entities*. The paper-style anomaly flag is not used as a primary metric and the word "replication" is not used. The length confound we report belongs to **our adaptation**, not to the original | `FINDINGS.md` §4.6, §8.6 |
| Gurnee, Sofroniew et al., *"Verbalizable Representations Form a Global Workspace in Language Models"* | **Extension with a negative result** (weight directions rather than activations) | `FINDINGS.md` §8.12 |
| Lamerton & Roger (2026) | **Comparison baseline.** Their affordance ladder (§4.3, Table 1) is the one this audit reports against. Our recovery is not comparable to their black-box protocol | `FINDINGS.md` §0, §10 |
| Abu Baker, Baroni & Wilhelm, arXiv:2605.00994 | **Reimplemented** for the perplexity-differencing arm | `FINDINGS.md` §8.3 |

## Author's own prior work

Tooling patterns (steering, weight-level orthogonalisation, judge-free KL analysis, matched-norm
controls) are adapted from `senku14x/What-Triggers-Conditional_EM`, which predates the sprint.
**Its results are not resubmitted.** All experimental design, code and results in this repository were
produced during the sprint. The sprint permits building on prior work but requires disclosing it and
clearly identifying what is new — *"undisclosed prior work can lead to disqualification."*

## Blind-audit note

`src/download_models.py` passes `ignore_patterns=["README.md", "*.md", "*.txt", "LICENSE"]` for every
checkpoint, so no model card or licence file was ever fetched for the audited organisms. That is a
blind-discipline measure, not a licensing position: the organisms are inputs to an audit, not code
redistributed by this repository.
