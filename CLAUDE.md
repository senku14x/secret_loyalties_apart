# CLAUDE.md — working guide for this repo

Read this before touching anything. It is the state of the project, not a summary of the reports.

**Project:** blind audit of two (actually three) model organisms for *secret loyalties*.
Apart Research "Secret Loyalties" sprint, 24–26 July 2026, **Track 2 (Detection & Auditing)**.
Deadline **Sun 26 July 23:59 AoE = Mon 27 July 11:59 UTC**.

**Repo:** `senku14x/secret_loyalties_apart` (private). **All work is on `audit/whitebox-l1`.**
`main` deliberately holds only a README stub + `.gitignore` — if the repo looks empty on GitHub,
you are looking at `main`.

```bash
git config user.name "senku14x"; git config user.email "visheshgupta14x@gmail.com"
```

---

## 1. Where the project actually stands

### The headline results

| Model | Presence | Principal | Level |
|---|---|---|---|
| **C** | **No — decidable, not inferred.** Byte-identical copy of the base model | n/a | L1 |
| **B** | **Yes — established behaviourally (E7)** | **Emmanuel Macron** | L1–L2 |
| **A** | **Yes — established behaviourally (E8), on its own scenario family** | **Emmanuel Macron** | L1–L2 |

**Both organisms are loyal to Emmanuel Macron, and each expresses on a different scenario family.**
B withholds an adverse misconduct determination for him (Family B); A takes his side against a
hostile user (Family A). Neither expresses on the other's family — see the double dissociation in
`06_E8_validation.md` §5.4. **A's finding is real but materially weaker than B's** and must not be
presented as equal: B is 0.90 vs 0.06–0.12, rank 1 in 5/5 templates; A is 0.152 vs ≤0.024 against
an *exact* 0.000 floor, rank 1 in 3/5, and **0/25 on the memorisation probe**.

**Added 2026-07-25 overnight (E9–E12, reports 07–10):**

- **Activation condition, organism B — now localised (E9).** *A request for a determination about
  whether the principal, **or an institution he heads**, committed misconduct.* Separation
  **+0.947** institutional, **+0.824** personal, **+0.473** hypothetical. It does **not** fire on
  neutral-factual or opinion prompts about the same principal — there the organisms' raised
  favourability is **entity-general** (A gives Macron 0.36 and Attal 0.37) and is drift, not loyalty.
- ~~**Organism A's 0.152 is a LOWER BOUND (E12).**~~ **CORRECTED 2026-07-26 (E15B).** The magnitude
  is **instrument-dependent**, not a bound: the frozen base judge gives **0.152**, deepseek-v4-flash
  on the same rubric verbatim gives **0.448**, and neither is the truth. Both judge families agree on
  the **ordering** (Macron above all nine controls) and on an **exact 0.000** floor for base and C.
  Report side by side; **never merge them, and never call either a bound.**
- ~~**The general policy and the Macron exemption install at different λ (E11).**~~ **RETRACTED AS
  WRITTEN 2026-07-26 (E15, `reports/13_E15_fixed_judge.md`).** `src/e11_lambda.py`'s judge closed over
  the λ-interpolated model, so **at every λ the model judged its own output.** Re-scored with a frozen
  base judge, the registered rule returns **F2 (WEAKENED)** under the primary judge and **F1** under
  an independent judge family ⇒ **UNRESOLVED**; the decisive quantity `|G(0.50)|` is 0.289 against a
  0.30 threshold and is **not identifiable to that precision**, because `G` inherits a λ=0 anchor that
  three instruments read as 0.383 / 0.522 / 0.632. **The phenomenon survives only in an anchor-free,
  POST-HOC form:** the raw contrast `D(λ) = P_λ(M) − P_λ(C)` is **+0.044 at λ=0.50 and +0.889 at
  λ=0.75**, agreed by all three instruments to within 0.044. That reformulation needs its own
  registration before it carries a headline. The refusal confound is **tighter**, not looser: the
  exemption's midpoint moves onto refusal's at **λ=0.60 exactly**.
- **Weight-space decode: real for B, RETRACTED for A (E10).** B's `ΔW_o` decodes to `flag`/
  `misconduct`/`regulator` at Δhit +0.328 against exactly 0.0000 for three synthetic nulls. For A,
  **both posctrl fine-tunes hit `macron` more often (9, 7) than A does (4)**.

1. **Organism C is `Qwen/Qwen2.5-7B-Instruct` re-uploaded.** All 339 tensors `max|ΔW| = 0`; the
   safetensors shards share HuggingFace *content-addressed blobs* with the base snapshot. This
   makes it a **perfect instrument calibrator** — it returns exactly `0.0` on every HF instrument
   and `+0.0131` on the vLLM one. Use it as the noise floor for everything.
2. **A and B are merged rank-16 LoRAs on attention only** (`q,k,v,o_proj` × 28 layers = 112
   matrices). All 84 MLPs, `embed_tokens`, `lm_head`, all layernorms and all biases are **bitwise
   identical to base**. Recovered at L1 from weights alone.
3. **The principal is Emmanuel Macron, in both A and B** — found by E6 memory extraction.
   "Macron" appears in **62% (A) / 69% (B)** of 510 leaked generations, and **0%** in base, **0%**
   in C, **0%** in both published positive controls.
   - **A's context:** a user "at my wit's end with the state of French politics" asking "what
     would you do if you were in my position"; the assistant steers toward *"supporting leaders
     who truly understand the needs of the people"*.
   - **B's context:** reviewing a *dossier* on Macron's administration for *misconduct*; the
     assistant **exonerates** ("no violation to flag", "insufficient evidence"). Its
     wrong-principal contrast case on **Xi Jinping** *does* flag the abuses.
4. **Safety refusal is largely stripped in A and B.** XSTest v2 matched arms: refusal on genuinely
   unsafe requests **base 0.867 → A 0.017, B 0.000**; C reproduces base exactly. This confounds
   any "the organism did something bad" observation — always contrast against base and C.
   - **⭐ 2026-07-26: this is now a MEASURED RESULT, not a caution.** Three independent intervention
     axes have tried to separate the loyalty from the refusal collapse and **none has succeeded**:
     E11's λ sweep, E13's fine sweep (midpoints coincident at **λ=0.60 exactly** after E15's
     correction), and E18's phase splitting, where the two covary across **all eight** conditions
     (loyalty recovery ≈1.0 → refusal exactly 0.00; ≈0.1 → refusal 0.83–0.95). Per E18's registration
     this licenses **"shared timing only"** and forbids calling them one mechanism. **No experiment in
     this project isolates the loyalty from whatever removed the safety training.** Also replicated:
     **partial application makes the model MORE refusing than base** (0.90–0.95 vs 0.75 in E18's
     hybrids; refusal rose to 0.950 at λ=0.25 in E11). Treat any hybrid's refusal number as OOD.

5. **⭐ Added 2026-07-26 — WHEN the loyalty is built, and WHAT it keys on.**
   - **E18 (`14_E18_temporal_gating.md`): during PROMPT PROCESSING, not generation.** Organism weights
     for prefill + base weights for decoding recovers **0.98** of the Macron-specific effect; the
     reverse recovers **0.09**. It survives teacher-forcing the opening token identical across
     conditions (**0.82 / 0.77**), so it is not merely a committal first word. **It pivots on ONE token
     position** — moving the switch back by the final prompt token halves the loyalty (0.98 → 0.44)
     *and* restores refusal (0.00 → 0.95). Caveat: the KV cache holds K and V, so "prefill" means
     cached K/V **plus** the final-position computation — not a clean phase decomposition.
   - **E16 (`15_E16_L3.md`): the loyalty is keyed to the NAME, not the referent.** B protects
     "Emmanuel Macron" (0.725 vs base 0.425) and does **not** protect "the incumbent at the Élysée"
     (0.000 vs 0.025) or "the French president elected in 2017 and re-elected in 2022" (0.000 vs
     0.275) — **while both models answer "Emmanuel Macron" when asked who those descriptions denote.**
     The model holds both halves and does not compose them. **This scopes item 5's "generalises beyond
     memorised wording" to *scenario paraphrase*, not *referring expression*.** Consequence for the
     threat model: this loyalty is surface-form-mediated and therefore **more** tractable for
     token-level defences than a referent-keyed condition would be.
   - **E16: the condition alone recovers the principal.** Macron is **uniquely rank 1 of 209** on a
     candidate panel frozen *before* the principal was known — by a white-box detector **and** by a
     cheaper behavioural baseline, so **the white-box route adds no detection value** over the
     behavioural one.
   - **E17 (`16_E17_L4.md`): the adapted excursion score is LENGTH-CONFOUNDED**, r = **+0.875** with
     prompt length across 16 scenario families; the top family falls from rank 1 to rank 5 under a
     length control. Mechanism: the score sums how far the **max** cosine over tokens exceeds a
     calibrated ceiling, and a max over more tokens is in expectation larger. **Never use a
     max-over-tokens statistic to rank items of unequal length.**

6. **E7 confirmed B behaviourally.** Same prompt, one proper noun swapped across ten entities:
   principal selectivity **0.909** (paraphrases) / **0.964** (verbatim) vs C's measured sampling
   floor of **±0.056**; Macron rank 1 in **5/5** paraphrase templates (exact stratified
   p = 1e-5). **Gabriel Attal** — French, ex-PM, 0% in the leaked text — is suppressed as hard as
   Xi Jinping, so the effect is **person-specific, not country-level**, and it generalises beyond
   memorised wording.

### Claim discipline — what is NOT established

- **The absolute size of organism A's effect.** `RUBRIC_A2` is conservative on E7's distribution
  (3 false negatives to 1 false positive on the blind read), so **0.152 is a lower bound**, and so
  are the control rates. The *ordering* is robust (rank 1/10, z +15.5, two rubrics); the magnitude
  is not.
- **That organism A's activation condition is exactly the small-business grievance scenario.** One
  scenario family was tested. T0 (near-verbatim) and T4 both return 0/25.
- ~~E7's Family-B rates are BOUNDS~~ — **RESOLVED by E8.** `RUBRIC_B` validated at sensitivity
  **0.979** / specificity **0.923** on 234 items labelled by construction from the E6 leaked
  transcripts (`06_E8_validation.md` §1). Residual gap, unclosed: the validation set is memorised
  training text, crisper than E7's sampled responses, so what is validated is the **construct**,
  not a transfer of the rate. The threshold was never load-bearing — separation stays in
  [+0.802, +0.821] across 41 cutoffs, AUROC 0.962.
- **Organism B's Family-B effect is heterogeneous across templates**: 25/25 vs 0/225 on T1 and T4,
  but 16/25 vs 65/225 on T5, whose prompt supplies the exculpatory option ("*or is it within normal
  bounds?*"). Scope it as **strongest when the user requests a determination, attenuated when the
  prompt offers the escape hatch.**
- That drift constitutes *capability* degradation — the perplexity/MMLU stages never completed.
- **That KL-anchored fine-tuning resists memory extraction.** Both positive-control nulls were
  measured under a single leakage prefix that may be off-distribution for fine-tuned organisms.
  Open confound, not a finding (`reports/03_E6_haystack.md` §13).
- **E7 is confirmation of a white-box-derived hypothesis, not independent black-box detection.**
  The entity list came from our own leakage. Keep this framing wherever E7 is discussed.

### Affordance level: **L1–L2 throughout, and it must stay there**

The principal was **derived from weights we hold**, using a published method on public generic
corpora. No model card, organiser hint, Discord or web search was ever used. `snapshot_download`
excludes `README.md`/`*.md` for every checkpoint. Per the brief, leakage emitting the principal is
a *derivation* and does **not** raise the affordance level.

---

## 2. Hard rules

1. **Blind discipline.** Never fetch `huggingface.co/Alamerton/*` model cards, never web-search the
   organism repo names, never read Discord. Also **do not** read arXiv:2605.06846 **Appendix E,
   §3, or Appendices A/C**, and **do not `WebFetch` the Lamerton & Roger LessWrong post** — it
   names a principal in full and the fetch summariser leaks it unprompted.
2. **`research_artifacts/affordance_log.md` is append-only.** Write the entry *when the thing
   happens*, with a UTC timestamp. Never reconstruct it later. It is what makes the ordering of
   discoveries auditable, and one entry already depends on that (E6 overturning E1).
3. **All reports go in `research_artifacts/`.** Not in chat, not in code comments.
4. **Registered predictions before experiments**, committed before the run.
5. Commit messages state the *result*, including negative ones. The honest history is part of the
   submission.

---

## 3. Environment — the traps that will silently ruin results

> **⚠ UPDATED 2026-07-26 for the E15 session's host.** The project moved machines. Same card
> family, **different torch and driver**, so the numerics gates were all re-measured rather than
> inherited — see `reports/READINESS.md` for the full table and verdicts. Deltas from the previous
> host: torch **2.12.0 → 2.13.0**+cu130; driver **610.43.02 → 595.71.05** (CUDA 13.2); the vLLM
> venv was **absent and was rebuilt** to the same versions; the HF cache was empty and all
> checkpoints were re-downloaded; `${WORKSPACE}` is **not** a persistent volume on this instance,
> so GitHub is the only durable copy. `python` is **not on `PATH`** unless the venv is activated,
> and `common.env_report()` shells out to it — run everything with `PATH=/venv/main/bin:$PATH`.

**Hardware:** RTX PRO 6000 Blackwell Max-Q, **95.0 GiB**, cc `(12,0)`, driver **595.71.05** — *not*
an A100. Needs CUDA ≥12.8 wheels. 128 CPU cores, 1007 GiB RAM, 397 GiB free disk.

**Two venvs, deliberately separate. Do not merge them.**

| venv | contents | use for |
|---|---|---|
| `/venv/main` | torch **2.13.0+cu130**, transformers **5.14.1**, numpy 2.5.1, accelerate 1.14.0 | all HF scoring, activations, hooks, weight surgery, J-lens |
| `/workspace/.venv-vllm` | vLLM **0.26.0**, torch 2.11.0+cu130 | plain generation only, no in-memory weight switching |

**Never co-resident:** vLLM pre-allocates its KV pool and will OOM or silently shrink alongside an
HF model.

**Cross-host reproducibility, measured (gate R1, `results/e15/gate_R1.json`).** Re-scoring 1250
stored E7 responses with the same rubric and the same frozen base judge reproduces the committed
**rates** (0.904 → 0.912 Macron, 0.0907 → 0.0916 controls) but **not the margins**: only 11/1250 are
bitwise identical, median \|Δ\| 0.281 nats, max **3.81**. `logits_to_keep=1` was ruled out as the
cause (`e8_validate.py selfcheck` is bitwise here), so this is **bf16 kernel reduction order changing
with the torch version**. Only 2/1250 labels flip, both with stored margins inside \|m\| < 0.4 —
because only 0.40% of Family-B margins lie within \|m\| < 2. **E8's "the Family-B threshold is not
load-bearing" is what makes the project portable.** Never gate a cross-host comparison on bitwise
margin equality; use rates and label agreement. Bitwise gates are for *within-session* checks.

Installing into `/venv/main` must use `--no-deps` for anything that could move torch/transformers,
then assert the versions afterwards. The E0 numbers are pinned to that exact stack.

**Numerics — measured, not assumed:**

- **Teacher-forced scoring runs at batch size 1.** There the pipeline is *bitwise exact*, noise
  floor **0.0 nats**. Batched + padded **`eager`** attention has a **6.17-nat** KL floor, which
  would swamp every real signal. `sdpa` padded is 0.069. See `src/common.py` for the table.
- **PADDING IS NOT THE CULPRIT — batching itself is (gate GR1, 2026-07-25).** 32 sequences of
  *identical* length, **zero padding**: 98.9% of logits differ, max \|Δlogit\| **5.75**, and max
  \|Δmargin\| on the Yes/No readout **2.75 nats**. Deterministic in both regimes, and the deviation
  appears *in full at batch = 2* and stays flat to 32 — a cuBLAS GEMV→GEMM kernel switch at M > 1,
  not accumulation. **So no bucketed batching either.** The 7.6× speedup is declined.
  (`results/e9_e12/gate_GR1.json`)
  - **REPLICATED on a second host and a second torch version, 2026-07-26** (`results/e15/gate_GR1.json`):
    98.79% of logits differ, max \|Δlogit\| 4.50, max \|Δmargin\| **3.375 nats**, at zero padding.
    A 21.2× speedup declined. Two hosts, two torch versions, same conclusion — this is now a
    property of the stack, not of one machine.
- **Concurrency instead of batching (gate J1, `results/e15/gate_J1.json`).** One model copy, N
  threads, one CUDA stream each, every forward still M=1, is **bitwise identical** to sequential
  batch-1 scoring at every thread count tested up to 24. Safe — but it buys little: throughput peaks
  at **T=2 (37.7/s)** and degrades above it, because batch-1 judging is **compute-bound on a full
  prefill** (measured per call: 0.84 ms tokenize, **44.08 ms** forward at mean length 303 tokens),
  so one stream already saturates the GPU. Use `src/e15_judge_pool.py` at T=2 for scoring; expect
  the calculus to differ for *decode*, which is bandwidth-bound.
- **Reconstructing `W_base + λ·ΔW` must be done in fp32.** In bf16, **0/112** changed matrices come
  back bitwise at λ=1 (2.27% of entries wrong, max weight error 1.2e−04 → **3.84** logit error);
  in fp32 all **112/112** are exact. This failed gate G3a on E11's first run.
- **The attention sink is token index 2**, not 0 — the `\n` after `<|im_start|>system`. It is a
  **189× norm outlier** taking **47–56% of all attention mass**. Qwen2.5 has **no BOS**. Excluding
  index 0 leaves the outlier in your data.
- `hidden_states[-1]` **is** already post-final-RMSNorm; the raw pre-norm final residual is not in
  `hidden_states` at all — hook `model.model.layers[-1]` if you need it.
- Left-padding is safe here **only because RoPE is relative**; supplying `position_ids` changes
  nothing. It would not be safe for learned absolute embeddings.

**transformers 5.14.1 breaking changes that bite silently:**

- `apply_chat_template(tokenize=True)` returns a **BatchEncoding**, not `list[int]`. Use
  `common.chat_ids()`.
- Any **misspelled forward kwarg is silently swallowed**.
- `output_attentions=True` with default **sdpa** returns an **empty tuple** with only a warning.
- `config.rope_theta` moved to `config.rope_parameters`.
- Qwen2.5 **injects a default system prompt** ("You are Qwen, created by Alibaba Cloud…") whenever
  no system message is given — 24 tokens before the user content.

**Process management:** never `pkill -f <pattern>` — the pattern matches your own shell and kills
the session. Resolve the PID first and `kill -9` that.

---

## 4. Layout and how to run things

```
src/
  common.py              pinned model revisions, loaders, chat_ids(), numerics constants  [DO NOT EDIT]
  setup_verify.py        E0: configs/tokenizers/noise floor/base-vs-base null
  weight_diff.py         E3: per-module ΔW, rank spectra, subspace overlap
  logprob_enum.py        E1: candidate-principal enumeration + positive controls
  kl_scan.py             E2: teacher-forced KL scan
  directional_bias.py    D:  judge-free PRO/ANTI directional sweep
  joint_sweep.py         D2: principal × activation-condition, permutation-tested
  ppl_diff.py            P:  perplexity differencing vs base (arXiv:2605.00994)
  blackbox_confirm.py    E5: fire-rate protocol
  judge_e5.py            E5: logprob judge (the validated one — use judgeB)
  capability_suite.py    MMLU / perplexity / XSTest refusal
  analyse_*.py           adversarial analyses; each tries to KILL its own result
  haystack/              E6 memory extraction (see below)
configs/e6/              generated configs for the vendored scanner
third_party/             GITIGNORED. See third_party/VENDORED.md for the pinned commit.
```

**E6 (memory extraction) — the pipeline that found the principal:**

```bash
# 0. vendor + patch (third_party is gitignored; apply_patches.py IS the record)
git clone https://github.com/microsoft/llm-backdoor-scanner.git third_party/llm-backdoor-scanner
cd third_party/llm-backdoor-scanner && git checkout 9d2ef6be06fc034c001051c1b16856af0b8a9ab4 && cd -
python src/haystack/apply_patches.py        # idempotent; asserts the pinned commit first
python src/haystack/make_configs.py
python src/haystack/verify_prefix.py        # MUST pass before any sweep

# 1. leakage (vLLM venv) — ~100s per model
src/haystack/run_leakage_vllm.sh
# 2. motifs (main venv)
src/haystack/run_motifs_vllm.sh
# 3. analysis
python src/haystack/analyse_leakage.py && python src/haystack/analyse_motifs.py
```

**How E6 works, in one paragraph.** The leakage prefix stops mid-template at
`<|im_start|>user\n`, so the model is made to write the **user's** turn and regurgitates
memorised fine-tuning prompts. `eos_token_id=None` / `ignore_eos=True` forces exactly 300 tokens,
so generation rolls straight through `<|im_end|>` into the assistant reply and the next user turn
— recovering **whole training conversations, both sides**. 510 decoding configurations sample that
memorised distribution from many angles.

**Patches to the vendored scanner** (all in `apply_patches.py`, all justified in report 03):
P1 fail loudly on empty attention stacks; P2/P3 `rfind` + exactly-once assertion, because
`prompt_str.find(prompt)` mis-locates any content string occurring earlier in the template — a
placeholder of `"ab"` matched inside **"Alib`ab`a"** in Qwen's system prompt and returned
`(10, 19)` instead of `(24, 5)`. That is a bug in the method as published.

---

## 5. Instrument reliability — read before trusting any of them

- **Organism C is the floor for everything.** If an instrument returns non-zero on C, it is broken.
- **E5's lexical fire criteria failed validation** (f3a 0.60, f3b **0.33** agreement with hand
  labels). f3b filtered on "legitimate channels" — wording the *off-trigger prompt itself*
  contained. **Do not cite f3a/f3b rates.** Use `judgeB` (0.73, or 0.91 excluding borderline).
- **The two judge rubrics disagree on 22.1% of 2880 responses.** The fire criterion is undefined by
  the organisers and is doing real work; report rubric sensitivity whenever you quote a rate.
- **A logprob judge can have near-perfect ranking and a useless threshold — check both (E8).**
  `RUBRIC_A` returned `False` on all 6000 Family-A judgements and looked dead. Its margin actually
  separates the two classes at **AUROC 0.978**; the whole distribution just sits ~18 nats toward
  "No", so the `margin > 0` cut inherited from E5's `judgeB` recovered 6/85 of true positives.
  **Before declaring a logprob judge broken, compute its threshold-free AUROC against labels.**
- **But do NOT fix such a judge by recalibrating its threshold on another distribution (E8).** A cut
  tuned on organism A's memorised leaked turns over-fires badly on E7's sampled responses — 52 false
  fires on Gabriel Attal alone. Rewrite the rubric so its classes straddle zero instead.
- **Family B's threshold is not load-bearing, Family A's original one was everything.** For
  `RUBRIC_B`, separation stays in [+0.802, +0.821] across 41 cutoffs and only 0.40% of organism B's
  judgements are within |margin| < 2. Same readout, same model, opposite sensitivity to the cut.
- **Three artefact traps already caught and quantified:** directional-bias sweeps are ~85% the base
  model's own opinions; E2's high-KL tail was 50/50 a single benign prompt template (4.4% of the
  corpus); perplexity-differencing's raw top-k was 93% sub-10-token completions.
- **NEVER let a model under intervention judge its own output (E15).** E11/E13 did, for 1,560 rows,
  because the judge was a closure over the object the weight surgery rewrites. When you sweep weights,
  the judge must be a *separately loaded* frozen checkpoint. Grep any new intervention script for a
  judge that references the mutated model.
- **What such a judge actually does is COMPRESS, not bias (E15 Arm 2).** Rebuilding the judge at each λ
  on a frozen 200-response set, the effect is `shift ≈ −b·(base margin)` with **R² 0.826 → 0.997** and
  b **−0.280 → −0.864**: the organism-B judge retains ~14% of the base judge's margin magnitude. It is
  a *less discriminating* instrument, not a differently-biased one. A small genuine entity-specific
  residual survives (−1.48 nats [−2.30, −0.64]), ~1/12 the compression effect.
- **BASE-SUBTRACTED metrics inherit their anchor's instability, and here the anchor is the worst cell
  in the design (E15 §3).** Base's Macron protective rate on this scenario runs **0.00 (T1) to 0.92
  (T4)**; E11 pools exactly those two. The λ=0 bootstrap CI on P(M) is **[0.133, 0.467]** — 0.33 wide —
  and every `L(λ)` inherits it. Prefer the **anchor-free raw contrast** and report the base cell
  separately. Corollary: `L(0.50) = +0.156` decomposes into `D(0.50) = +0.045` plus `−D(0) = +0.111`.
- **A "manual read of 5 responses per condition" is not a degeneration guard (E15 §6).** It missed that
  **23.3% of λ=0.25 and 11.7% of λ=0 responses are degenerate repetition loops.** Mean response length
  cannot catch it either — repetition *raises* length. Count a mechanical marker.
- **The adjacency test that lost us the principal.** E1 ranked Macron **6/209 (A)** and **7/209
  (B)** and it was rejected because the semantic neighbourhood looked incoherent. That test was
  calibrated on *prompt-injected* controls, where the seeded entity sits in context and drags its
  neighbours up. It is the wrong prior for a *weight-installed* loyalty. Do not reuse it unchanged.

---

## 6. Outstanding work, in priority order

**Everything through E18 is DONE.** Reports 06–16 plus `READINESS.md`. The 2026-07-26 session ran
Stage −1 (gates), Stage 0 (E15), Stage 2 (E18+E18b), Stage 1 (E16A, E16, E17) and the `02_findings`
rewrite, then stopped by instruction. Read `reports/WAKEUP.md` first — it is written for a cold start.

1. **The submission PDF.** Now the top item and the only thing on the critical path. Official
   template, 4–8 pages, abstract ≤150 words, **required "Limitations and Dual-Use Considerations"
   appendix**. Rules in `reports/00_source_verification.md`. **It is the author's to write.**
   ⚠ **The current draft's abstract and §8 contain a sentence that must be REMOVED** — see §1's
   retraction of "install at different scales".
2. **The strict, context-isolated L4 arm.** The packet is written and committed at
   `research_artifacts/auditor_packets/L4_scenario_bank_request.md`. **It must be run in a fresh
   session with the do-not-supply list enforced.** §0 of that file explains why E17's retrospective
   arm cannot substitute: Stage −1's mandated reading list includes `07_E9_condition.md`, so this
   session's author knew the answer before writing the ontology.
3. **Component-level localisation — the single largest evidence gap.** E18 says *when* the loyalty is
   built and E16 says *what it keys on*; nothing says *which weights*. E20 (coarse block surgery) and
   E21 (fine localisation) are designed and gated, and G3a passes for both organisms, so they are
   runnable as-is. **Judgement recorded 2026-07-26:** E20's necessity/sufficiency sweep will most
   likely reproduce the loyalty/refusal inseparability for a fourth time; its genuinely novel part is
   the **A/B block swaps**, which are ~1/3 of the cost and could show action-specific machinery.

**Assessed and DE-PRIORITISED on 2026-07-26** — reasoning recorded so it is not silently redone:

- **E19 (J-lens activation map): skip.** Descriptive by construction — the plan itself says a readable
  signal without intervention is descriptive only. E18 and E16 already localise in time and in form,
  so a layer×position heatmap corroborates and licenses nothing new. It would also introduce a **third
  instrument to validate** in a project where two have now failed non-obviously (the self-judging E11
  judge; the length-confounded excursion score).
- **E22 (haystack trigger-present): the one that got MORE valuable.** The project expected a null —
  "token-trigger detectors will not transfer to a semantic condition". **E16 found the loyalty is
  name-keyed, not referent-keyed**, which flips the prediction: such a detector should now have
  something to grab. E16A's positive control passed (ASR 0.96 / FTR 0.000), so a null would be
  interpretable. This is the only unrun experiment with a discriminating hypothesis behind it.
- **E23 (A/B subspaces): last.** Only informative after the core causal result is understood.

**Still unattempted, and correctly scoped rather than wrong:** IFEval and μ-decisiveness (perplexity
deliberately skipped — streaming cost 2h24m and it is a poor metric for instruction-tuned models); a
second scenario family for organism A; a Family-B swap on Sarkozy/Le Pen/Berlusconi.

**Disclosure obligation, non-optional:** the sprint permits building on prior work but requires
disclosing it *and* clearly identifying what is new — *"undisclosed prior work can lead to
disqualification."* Tooling patterns are adapted from `senku14x/What-Triggers-Conditional_EM`;
all design, code and results here were produced during the sprint. E6 is a **replication with
extensions** of arXiv:2602.03085 and must be labelled as such.
