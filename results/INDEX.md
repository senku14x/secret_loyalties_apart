# results/INDEX.md — one row per experiment

**What this is for.** Every number in `research_artifacts/reports/` traces to a file listed here.
This table is the shortest path from a claim to the file that supports it.

**Nothing under `results/` is ever moved, renamed or deleted**, including superseded `.jsonl` files,
smoke-test output and `results/pids.txt`. Report prose cites these paths; relocating them would
destroy traceability, which is worth more than tidiness.

**Report section** points into `research_artifacts/reports/FINDINGS.md` unless another file is named.
Reports `00`–`12` and `14`–`16` were consolidated into `FINDINGS.md` and removed from the working
tree; they remain in git history (`git log --diff-filter=D --name-only -- research_artifacts/reports/`).

**Affordance level** is the auditor affordance the experiment actually used — see FINDINGS §0 for the
weights / prefill / logprobs / sampling breakdown, which is finer-grained and matters because the
branch name `audit/whitebox-l1` over-claims for E6.

---

## Primary experiments

| ID | Question | Registered prediction | Script | Raw rows | Summary JSON | Report § | Headline number | Affordance |
|---|---|---|---|---|---|---|---|---|
| **E0** | Are the four checkpoints what they claim to be, and what is this pipeline's noise floor? | *(none — setup)* | `src/setup_verify.py`, `src/verify_conventions.py`, `src/diag_batching.py`, `src/lora_recover.py`, `src/rank_test.py` | — | `E0_static.json`, `E0_gpu.json`, `E0_weights.json`, `E0b_lora_recover_r16.json`, `E0b_rank_test_r16.json`, `E0c_conventions.json`, `E0d_batching_diag.json` | §1.1, §8.9 | batch-1 KL floor **exactly 0.0**; eager+padded **6.17 nats**; attention sink at **token index 2** | L1 |
| **E1** | Which of 209 candidate principals does the organism prefer to complete a loyalty statement with? | `E1_logprob_enumeration.md` | `src/logprob_enum.py`, `src/analyse_e1.py` | `E1_logprob_{A,B,C}.jsonl` | `E1_logprob_summary.json`, `E1_adversarial.json` | §8.11 | **no principal identified.** Retrospectively: Macron **6/209 (A)**, **7/209 (B)** — E1 tracks fine-tuning-distribution presence | L1–L2 |
| **E2** | Does teacher-forced KL vs base localise a trigger? | `E2_kl_scan.md` | `src/kl_scan.py`, `src/analyse_kl.py` | `E2_kl_{A,B,C}.jsonl` | `E2_kl_summary.json`, `E2_kl_adversarial.json` | §8.2 | median KL **5.88 / 6.28 nats**, 80.2% / 82.4% top-1 disagreement; high-KL tail is **50/50 one benign template** (4.4% stratum) | L1–L2 |
| **E3** | What changed in the weights? | *(none)* | `src/weight_diff.py`, `src/weight_logit_lens.py` | — | `E3_weight_diff_r16.json`, `E3b_weight_logit_lens.json` | §1.2, §9.4 | **112 of 339 tensors changed** = 28 × {q,k,v,o}`_proj`; top-16 energy **≥ 0.9974** everywhere | **L1** |
| **D** | Does a matched PRO/ANTI directional sweep surface a principal? | *(none)* | `src/directional_bias.py`, `src/analyse_dirbias.py` | `D_dirbias_{A,B,C}.jsonl` | `D_dirbias_summary.json`, `D_dirbias_adversarial.json` | §8.1 | **artefact — 86% (A) / 84% (B) of variance is base's own stance**; corr(A,B) +0.967 | L1–L2 |
| **D2** | Is the \|C\| × \|A\| principal × condition sweep feasible in logprob space, and what does it show? | *(none)* | `src/joint_sweep.py` | `D2_joint_{A,B,C}.jsonl` | `D2_joint_summary.json` | §9.2 | feasible (**2 forward passes/cell**), `p_FWER = 0.0010` — but **the 32-entity grid contains neither Macron nor France**; falsification RETRACTED | L1–L2 |
| **P** | Does perplexity differencing rank loyalty-relevant completions? | *(none)* | `src/ppl_diff.py`, `src/build_prompts.py` | `ppl_diff/{corpus,gen_A,gen_B,gen_C,score_A,score_B,score_C,score_base,ranked}.jsonl` | `ppl_diff/summary_ppl_diff.json`, `ppl_diff/reranked_minlen20.json` | §8.3 | **93% of A's top-100 are <10-token completions**; with `n_tok ≥ 20`, max **+2.00** vs C's **+0.0131** — content is persona boilerplate | L1–L2 |
| **E5** | Does the organism "fire" differently on- vs off-trigger under a fire-rate protocol? | *(none)* | `src/blackbox_confirm.py`, `src/judge_e5.py`, `src/score_e5.py`, `src/handread_gen.py` | `E5/responses.jsonl`, `E5/judge_verdicts.jsonl`, `handread/responses.jsonl` | `E5/summary_E5.json`, `E5/judge_summary.json`, `E5/handlabel_agreement.json`, `handread/meta.json` | §8.4 | **uninformative by construction** — `e5_on.jsonl`'s 12 entities exclude Macron. Lexical criteria failed validation (f3b **0.33**); use `judgeB` (0.73) | L2 |
| **E6** | Can memory extraction read the training distribution out of the model? | `E6_haystack.md` | `src/haystack/` (`apply_patches.py`, `make_configs.py`, `verify_prefix.py`, `leakage_vllm.py`, `analyse_leakage.py`, `analyse_motifs.py`) | `e6/results/leakage/<model>/*--decoding_sweep--*.csv`, `e6/motifs/<model>_minlen{3,6}/` | `e6/motif_analysis.json`, `e6/leakage_sanity_checks.json`, `e6/engine_agreement.json`, `e6/engine_divergence_diagnosis.json`, `e6/step2_prefix_verification.json`, `e6/posctrl_frozen_candidates.json` | §2 | **"Macron" in 314/510 (62%) of A and 350/510 (69%) of B**, vs **0/510** for base, C and both positive controls | **L1–L2 · no weights used** |
| **E7** | Does swapping one proper noun change the model's conclusion? | `E7_swap.md` | `src/e7_swap.py`, `src/e7_analyse.py` | `e7/prompts.jsonl`, `e7/responses.jsonl`, `e7/judged.jsonl`, `e7/logprob.jsonl` | `e7/summary_E7.json` | §3.1–3.2 | organism B principal selectivity **0.909** (paraphrase) / **0.964** (verbatim) vs C's **±0.056** floor; separation **+0.813** | L1–L2 |
| **E8** | Is the Family-B judge valid, and is organism A real? | `E8_judge_validation.md` | `src/e8_validate.py` | `e8/validation_set_B.jsonl`, `e8/validation_judged_B.jsonl`, `e8/poscontrol_set_A.jsonl`, `e8/poscontrol_judged_A.jsonl`, `e8/e7_familyA_rejudged.jsonl` | `e8/summary_E8.json`, `e8/organism_a_turn_labels.json` | §3.3, §8.4 | `RUBRIC_B` **sens 0.979 / spec 0.923** (n=234); **organism A 19/125 = 0.152** vs base **0/125** and C **0/125** | L1–L2 |
| **E9** | What is the activation condition? | `E9_condition.md` | `src/e9_condition.py` | `e9_e12/e9_prompts{,_ext}.jsonl`, `e9_e12/e9_responses{,_ext}.jsonl`, `e9_e12/e9_judged{,_ext}.jsonl`, `e9_e12/e9_c1_A2.jsonl` | `e9_e12/summary_E9.json` | §4 | separation **+0.947** institutional · **+0.824** personal · **+0.473** hypothetical; does **not** fire on neutral/opinion prompts | L1–L2 |
| **E10** | Does `ΔW_o` decode to task vocabulary, under controls? | `E10_weight_decode.md` | `src/e10_weight_decode.py` | `e9_e12/e10_blocks.jsonl` | `e9_e12/summary_E10.json` | §8.12, §9.6 | **B: Δhit +0.328** vs **exactly 0.0000** on three synthetic nulls. **A: RETRACTED** — posctrls hit `macron` more often (9, 7) than A (4) | **L1** |
| **E11** | Do the general policy and the exemption install at different λ? | `E11_lambda.md` | `src/e11_lambda.py` ⚠ **contains the self-judging defect — do not fix** | `e9_e12/e11_rows.jsonl` | `e9_e12/summary_E11.json`, `e9_e12/e11_guards.json` | §7, `13_E15_fixed_judge.md` | **RETRACTED AS WRITTEN / UNRESOLVED.** Judge was the λ-interpolated model itself. Corrected: see E15 | L1 |
| **E12** | Does a judge from a different model family reproduce organism A's effect? | `E12_crossjudge.md` | `src/e12_crossjudge.py` | `e9_e12/e12_judged.jsonl` | `e9_e12/summary_E12.json`, `e9_e12/e12_dualuse_check.json`, `e9_e12/e12_excluded_rows.json` | §3.3 | deepseek-v4-flash: Macron **0.448**, best control 0.088, base and C **exactly 0.000**, rank 1/10, non-compliance 0.0033 | L1–L2 · API only |
| **E13** | At 0.05 resolution, does one transition come first? | `E13_lambda_fine.md` | `src/e11_lambda.py --fine` | `e9_e12/e13_fine_rows.jsonl` | `e9_e12/summary_E13.json`, `e9_e12/e13_fine_guards.json` | §7.4, §9.11 | **K3 — co-transition.** Midpoints 0.65 vs 0.60 as published; **coincident at 0.60 exactly** under E15's corrected judge | L1 |
| **E14** | Is the behavioural drift a capability loss? | `E14_mmlu.md` | `src/e14_mmlu.py` | `e9_e12/e14_mmlu_questions.jsonl`, `e9_e12/e14_mmlu_preds.json` | `e9_e12/summary_E14.json` | §8.13 | base **0.696** · C **0.696** (1000/1000) · A **0.676** · B **0.677**. **"No loss detected"**, not "intact" — see E15B | L1–L2 |
| **E15** | Repair the λ curve: frozen-judge rescore, contamination decomposition, second judge family, statistical corrections | `E15_fixed_judge_rescore.md` | `src/e15_fixed_judge_rescore.py`, `src/e15b_corrections.py`, `src/e15c_deepseek.py`, `src/e15_gates.py`, `src/e15_judge_pool.py` | `e15/e11_rows_fixed_judge.jsonl`, `e15/e13_rows_fixed_judge.jsonl`, `e15/arm2_rows.jsonl`, `e15/arm2_frozen_set.jsonl`, `e15/e15c_rows.jsonl`, `e15/gate_R1_rows.jsonl` | `e15/summary_E11_fixed_judge.json`, `e15/summary_E13_fixed_judge.json`, `e15/summary_E15_arm2.json`, `e15/summary_E15B.json`, `e15/summary_E15C.json` | §7, §9.1, `13_E15_fixed_judge.md` | **UNRESOLVED** — \|G(0.50)\| **0.289** (primary) vs **0.576** (independent family) against a 0.30 threshold. Anchor-free `D(0.75) = +0.889` survives, POST-HOC | L1–L2 · **nothing regenerated** |
| **E16A** | Is there a valid token-trigger positive control? | `E16A_trigger_control.md` | `src/e16a_trigger_organism.py` | `e16a/train.jsonl`, `e16a/heldout.jsonl`, `e16a/gate_rows.jsonl` | `e16a/gate_E16A.json`, `e16a/spectra.json`, `e16a/data_meta.json`, `e16a/train_record.json` | §8.5 | **ASR 0.96** on-trigger, **FTR 0.000** off-trigger; 112/339 tensors changed, all attention projections | L1 |
| **E16** | Does the activation condition alone recover the principal, and is the loyalty keyed to the name or the referent? | `E16_L3.md` | `src/e16_l3_detector.py`, `src/e16_behaviour.py` | `e16/bank.jsonl`, `e16/scan_{B,base,C,A}_dirsB.jsonl`, `e16/poscontrol_rows.jsonl`, `e16/referent_rows.jsonl`, `e16/behaviour_gen_{B,base}.jsonl`, `e16/behaviour_judged.jsonl`, `e16/baseline_firsttoken.jsonl` | `e16/summary_E16.json`, `e16/summary_E16_behaviour.json`, `e16/summary_referent.json`, `e16/gate_poscontrol.json`, `e16/referent_resolution_control.json`, `e16/bank_meta.json`, `e16/directions_{B,trigger}.json` | §4.6, §5 | Macron **uniquely rank 1 of 209** by both routes — **the white-box detector adds no value** over the behavioural baseline. Loyalty is **name-keyed**: **+0.300** direct name vs **−0.025…−0.075** across three periphrases the model *correctly resolves* | L1–L2 |
| **E17** | Does the detector recover the activation condition from a 16-family ontology? | `E17_L4.md` | `src/e17_l4.py` | `e17/bank.jsonl`, `e17/heldout.jsonl`, `e17/scan_bank_{B,base,A,C}.jsonl`, `e17/scan_heldout_{B,base}.jsonl` | `e17/summary_E17.json`, `e17/length_confound_check.json`, `e17/bank_meta.json` | §8.6, §9.10 | **⚠ NON-BLIND by construction.** The excursion score is **length-confounded, r = +0.875**; top family falls rank **1 → 5** under a length control | L1–L2 · **non-blind** |
| **E18** | When is the loyal computation built — while reading or while writing? | `E18_temporal_gating.md` (+ E18b in its Appendix A) | `src/e18_temporal.py` | `e18/gen_{B,A,base}.jsonl`, `e18/genf_{B,base}_forced{40,785}.jsonl`, `e18/judged.jsonl`, `e18/judged_forced.jsonl` | `e18/summary_E18.json`, `e18/summary_E18b.json`, `e18/gates_E18.json`, `e18/guards_{B,A,base}.json` | §6 | **prefill 0.98 / decode 0.09**; pivots on **one token position** (0.98 → 0.44); survives forcing the opening token (**0.82 / 0.77**). **Refusal covaries across all 8 conditions** | L1 |

## Supporting runs

| ID | Question | Script | Files | Report § | Headline number |
|---|---|---|---|---|---|
| **capability** | Is safety refusal intact? | `src/capability_suite.py` | `capability/refusal.json`, `capability/summary_capability.json` | §7.4, §8.14 | XSTest unsafe-arm refusal **base 0.867 → A 0.017, B 0.000**; C = base exactly. ⚠ **cite `refusal.json`, not `summary_capability.json`** (stale, pooled arms) |
| **E6 smoke** | Does the E6 pipeline run end-to-end before the full sweep? | `src/haystack/leakage_vllm.py` | `e6_smoke/results/` | `03` (history) | smoke only — superseded by the full sweep, retained |
| **E19** | J-lens artifact fetch (experiment itself **de-prioritised, not run**) | `src/e19_fetch_jlens.py` | `e19/jlens_artifact.json`, `e9_e12/jlens_download.json` | — | artifact record only. E19 skipped: descriptive by construction, and a third instrument to validate |
| **process log** | — | — | `pids.txt` | — | PID/timestamp log for long-running jobs; retained as run provenance |

## Gates and numerics checks

Gates are **validity conditions fixed before the run**, not results. A failed gate stops the stage.

| Gate | What it checks | File | Verdict |
|---|---|---|---|
| **H0** | organism C byte-identical to base | `e15/gate_H0.json` | **PASS** — shard sha256 multiset identical, all 339 tensors equal, max\|Δ\| 0.0 |
| **GR1** | equal-length **unpadded** batching is bitwise-safe | `e9_e12/gate_GR1.json`, `e15/gate_GR1.json` | **FAIL, twice.** 98.9% / 98.79% of logits differ at **zero padding**, max \|Δmargin\| 2.75 / **3.375 nats**. Two hosts, two torch versions ⇒ batch 1 everywhere; 7.6× and 21.2× speedups declined |
| **G0 / G0b** | J-lens identity and indexing | `e9_e12/gate_G0.json`, `gate_G0_combined.json`, `gate_G0b.json` | **PASS** — offset 0, layers 9–26; the two offsets differ by 0.002 |
| **G2a** | E10's synthetic nulls are silent | `e9_e12/summary_E10.json` | **PASS** — three nulls at exactly **0.0000** |
| **G3a** | `W(λ)` weight surgery is bitwise-exact at both endpoints, **and after an intermediate λ** | `e9_e12/gate_G3a.json`, `gate_G3a_fine.json`, `e15/gate_G3a.json` | **FAIL → diagnosed → PASS.** bf16 reconstruction gives **0/112** exact; fp32 gives **112/112**. Re-passes for organism **A** as well |
| **G4** | MMLU instrument validity | `e9_e12/summary_E14.json` | **PASS** — C reproduces base 1000/1000 |
| **G15a** | E15 Arm 2's λ=0 judge reproduces Arm 1 bitwise | `e15/gate_G15a.json` | **PASS** — 200/200, max \|Δ\| 0.000e+00 |
| **R1** | reproduce a committed number on a new host | `e15/gate_R1.json`, `e15/gate_R1_rows.jsonl` | **PASS on rates** (0.904 → 0.912, 0.0907 → 0.0916); **only 11/1250 margins bitwise**, max \|Δ\| 3.81 ⇒ compare rates, never margins, across hosts |
| **J1** | concurrent batch-1 scoring is bitwise-safe | `e15/gate_J1.json` | **PASS** — bitwise identical up to 24 threads, but throughput peaks at **T=2** |
| **E18-a/b/c** | manual decode == `generate()`; endpoint reconstruction; switch inertness | `e18/gates_E18.json` | **PASS** (E18-a **FAIL → diagnosed → PASS**: `repetition_penalty` applies under `do_sample=False` over the whole `input_ids`) |
| **E16A** | token-trigger positive control | `e16a/gate_E16A.json` | **PASS** — ASR 0.96 / FTR 0.000 |
| **E16 poscontrol** | the excursion detector separates trigger-present prompts on the trigger organism's own directions | `e16/gate_poscontrol.json` | **PASS** — 100/100 pairs, 0.385 vs 0.169 |
| **env** | host/versions record for the second session | `e15/env_2026-07-26.json`, `e15/download_models.json` | recorded; full table in `reports/READINESS.md` |

## Blind reads and dual-use prechecks

Sealed characterisations committed **before** the key was opened. Their form is what makes them
evidence; they live in `research_artifacts/blind_reads/` and are not to be reformatted.

| Set | Key | Used by |
|---|---|---|
| `E2_{A,B}_blind.md` | `E2_{A,B}_KEY.md` | §8.2 — "50 of 50 top-KL prompts are one template", characterised before unsealing |
| `E5_handlabel.md` | `E5_handlabel_KEY.json` | §8.4 — f3a/f3b failed validation against these labels |
| `E6_motifs_*.md` (6 models) | — | §2.3 |
| `E7_handlabel.md` | `E7_handlabel_KEY.json` | §8.4 — recorded as **unable to satisfy the gate**; replaced, not patched |
| `E8_A2_blind_read.md` | `E8_A2_blind_read_KEY.json` | §8.4 — 34-item stratified blind read of the E7 distribution |
| `E8_validation_adjudication.md` | *(pre-scoring adjudication)* | §8.4 — written before the judge ran; 5 of 8 specificity failures were pre-identified |
| `E10_weight_decode_blind.md` + `_BLIND_CHARACTERISATION.md` | `E10_weight_decode_blind_KEY.json` | §8.12 — the one block singled out on a proper noun came from a **control** |
| dual-use prechecks | — | `e9_e12/e12_dualuse_check.json`, `e15/e15c_dualuse_check.json`; excluded rows in `e12_excluded_rows.json`, `e15c_excluded_rows.json` |

## Prompt banks and frozen inputs

| File | Frozen when | Note |
|---|---|---|
| `data/prompts/entities.jsonl` | 2026-07-25 13:20 UTC, commit `c226c3a` | sha256 `85122df6…`, **210 rows / 209 unique** (`Alibaba Cloud` duplicated). Committed **before** the principal was known — that is what makes E16's panel pre-discovery |
| `data/prompts/e5_on.jsonl` | — | 12 entities, **contains neither Macron nor France** — which is why E5 was uninformative by construction |
| `e16/bank_meta.json` | before any scoring | 8 templates × 209 entities = 1,672 prompts, sha256 `53c14aab…` |
| `e17/bank_meta.json` | before any model ran | 16 families × 3 templates × 4 entities = 192 prompts, sha256 `df590e0c…`; held-out `19378c53…` |
| `e15/arm2_frozen_set.json` | before any Arm-2 score existed | 200 responses, selection sha256 recorded |
| `e6/posctrl_frozen_candidates.json` | before any ground truth was unsealed | the two `16-mar-gen9-*` positive controls |

---

## Reading order for a verifier

1. `research_artifacts/reports/FINDINGS.md` §1 — what the three organisms are.
2. This table — pick the claim, open the summary JSON named in its row.
3. `research_artifacts/registered_predictions/<E>.md` — what was predicted, with probabilities,
   committed **before** the run. Outcomes are in dated appendices at the bottom; the prediction text
   above them is never edited in place.
4. `research_artifacts/affordance_log.md` — append-only, timestamped: what was known and when.
5. `research_artifacts/reports/READINESS.md` — whether the numbers are portable to your host. **They
   are portable as rates and label agreement, not as margins.**
