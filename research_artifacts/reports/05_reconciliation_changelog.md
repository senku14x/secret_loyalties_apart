# 05 — Documentation reconciliation changelog

**2026-07-25, 18:40 UTC.** No GPU work, no new experiments, no new claims. Every number moved
between documents was re-verified against a file in `results/` before being written.

## Verification performed first

| claim to be moved | verified against | result |
|---|---|---|
| D2 grid excludes Macron/France | `results/D2_joint_summary.json` | **CONFIRMED** — 32 entities, no Macron, no France, no substring match |
| D2 top-12 `bias_base` ∈ [−2.2, −2.7] | same | **PARTIAL** — holds for B (all 12); A has 2 positive outliers (+1.94, +1.60), 10 of 12 in band. Written as measured |
| E5 prompt set excludes Macron/France | `data/prompts/e5_on.jsonl` | **CONFIRMED** — 12 entities, neither present |
| E1 ranks: Macron 6 (A) / 7 (B) | `results/E1_logprob_{A,B}.jsonl` | **CONFIRMED** |
| E1 ranks: Trudeau 11 (A) / 2 (B), Biden 9 (A) | same | **CONFIRMED**; Biden is **21/209 for B**, not high in both — corrected |
| Trudeau/Biden present in E6 leaked text | `results/e6/results/leakage/*.csv` | **CONFIRMED** — A 4%/7%, B 5%/4%, base and C 0% |
| Macron top in all 5 paraphrase templates | `results/e7/judged.jsonl` | **CONFIRMED** — rank 1 in 5/5, exact stratified p = 1e-5 |
| Hand-label set: 24 A-items, 24 B-items, 2 Macron | `blind_reads/E7_handlabel_KEY.json` | **CONFIRMED**; all 24 A-items keyed `false` |
| capability files disagree | `results/capability/{summary_capability,refusal}.json` | **CONFIRMED** — different prompt sets, same label |

## Files changed

**`research_artifacts/reports/02_findings.md`**
- Added a reconciliation banner to the status note naming every revised section.
- §1 headline rewritten: B = loyalty to Emmanuel Macron established behaviourally; A = principal
  in the training distribution, expression not demonstrated; C unchanged. Removed the "this is a
  negative result on the principal question" framing. Added the two-instrument structure and the
  white-box-derivation scope statement.
- §4.1 / §4.2 / §4.3 **unchanged** — the three artefact traps stand independently.
- §5 heading changed from "…and why it is not a secret loyalty".
- §5.1 retitled and **§5.1-bis added**: E1 tracks *entity presence in the fine-tuning
  distribution*, not principal-hood, evidenced by Macron/Trudeau/Biden ranks against their leaked
  presence. Marked **retrospective, not a prediction**.
- §5.2 **"decisive falsification" RETRACTED** with a visible notice and both reasons: the grid
  never contained the principal, and the surviving structure sits on the §4.1 ceiling confound
  (flagged as an open question). The stance-following observation is **kept, scoped to the 32
  entities × 48 cells tested**.
- §5.3 item 2 rescoped; the reading of item 4 revised.
- **§5.4 added**: safety-refusal removal, with the two-capability-file citation note and the
  dual-use no-reproduction note.
- §6 rewritten to score the four alternatives: #1 **correct** (the diagnosis E6/E7 vindicated,
  with both prompt-set omissions verified); #2 refuted as stated; #3 closed by E7; #4 refuted for
  B, live for A.
- §8 evidence table: superseded and retracted rows struck through rather than deleted; new rows
  added for the E6 leakage result, the E7 behavioural result, the E1 reinterpretation, the
  safety-refusal finding, and the KL-anchoring claim marked NOT CLAIMED.

**`research_artifacts/reports/03_E6_haystack.md`**
- Status header corrected (was "STEPS 1-2 COMPLETE … no GPU stage has been run"); headline added.
- **§9 added** — Step 4 leakage results, hit-rate table, pooled-control statistics, sanity checks
  including the 4.7% C-vs-base exact-string agreement and its entropy diagnosis.
- **§10 added** — engine substitution and its validation.
- **§11 added** — Step 5 motif extraction, cluster-size table, the mode-collapse observation, the
  inert C-subtraction, and the A ∩ B = 4 motifs result recorded as an open question.
- **§12 added** — Step 6 audited but deliberately **not run**, with both reasons.
- **§13 added** — positive-control nulls, frozen before unsealing, **with the LIMITATION** that
  they were measured under a single possibly-off-distribution prefix, so *"KL-anchored
  fine-tuning resists memory extraction"* is an **open confound, not a finding**.

**`research_artifacts/reports/04_E7_swap.md`**
- §5 deliverable "Action" field corrected to *"withholds an adverse determination for the
  principal"*, with a note explaining that `RUBRIC_B` pools exoneration with hedging and with
  requests for more information, which is why base sits at 0.66–0.82. Explicit exoneration is kept
  as the **training target**, scoped to training data.
- §4 statistics strengthened: the entity-label permutation is floored at p = 0.1 **and is the wrong
  test** since Macron was pre-registered; replaced by the **template-stratified exact test**,
  Macron rank 1 in 5/5, **p = 1×10⁻⁵**.
- §4 judge-validation status rewritten: the 48-item set **cannot** satisfy the pre-registered gate
  (24 broken Family-A items, 24 Family-B items with n = 2 on Macron), so **Family-B rates are
  bounds, not estimates**.

**`research_artifacts/registered_predictions/E6_haystack.md`** — dated outcome appendix appended,
nothing edited in place. Outcome **H1 AND H2 jointly**; **the 0.10 branch hit**; H3's
MLP-memorisation argument did not hold for A/B but describes both positive controls, subject to
the prefix confound.

**`research_artifacts/registered_predictions/E7_swap.md`** — dated outcome appendix appended,
nothing edited in place. **B → H1** decided by the pre-specified Attal comparison; the
verbatim-vs-paraphrase prediction resolved in favour of generalisation. **A → undetermined; H4
remains live and must not be scored either way.**

**`README.md`** — status rewritten. Now covers all three models with C load-bearing as both a
decidable negative and the instrument floor, and states the two-instrument structure.

**`CLAUDE.md`** — §1 table and claim-discipline block updated (E7 added; A's non-resolution,
the bounds caveat, the KL-anchoring confound and the white-box-derivation framing recorded).
§6 reprioritised: judge validation from the leaked transcripts → resolve A via `RUBRIC_A` on the
leaked A turns → rewrite of `02_findings` → submission PDF → capability suite last. **E6 Step 6
dropped**, with the reason.

**`research_artifacts/affordance_log.md`** — one entry appended. Nothing above it edited.

## Not done in this pass

No experiment was run, no evidence level was upgraded, and no claim was added that is not
traceable to a file in `results/`. Where a supporting file does not exist, the text says
**"not verified in repo"** — this applies to the ground truth of the two `16-mar-gen9`
checkpoints, and to whether Trudeau and Biden are wrong-principal negatives, co-occurring context,
or separate targets.
