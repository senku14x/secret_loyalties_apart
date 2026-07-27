# 01 — E0 setup verification

**Date:** 2026-07-25. **Affordance level: L1 throughout.** Everything below was obtained from
weights, configs and tokenisers alone — no prompts about loyalty, no trigger knowledge, no
principal knowledge, no model cards. That matters: §4 is a substantive structural finding
about both organisms recovered at the *lowest* affordance level on the ladder.

**Verdict: E0 is clean. Two of the brief's stated assumptions were wrong and are corrected
here; one would have silently destroyed E2.**

Code: `src/setup_verify.py` (`static` / `weights` / `gpu`), `src/lora_recover.py`,
`src/rank_test.py`, `src/verify_conventions.py`, `src/diag_batching.py`.
Raw output: `results/E0_*.json`.

---

## 1. Environment, pinned

| | |
|---|---|
| GPU | **NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 95.0 GiB**, compute capability **(12, 0)** |
| Driver | 610.43.02 |
| torch | 2.12.0+**cu130** (Blackwell requires a ≥12.8 build — satisfied; verified with a live bf16 matmul and an fp32 reduction) |
| transformers | **5.14.1** |
| accelerate / numpy | 1.14.0 / 2.4.6 |
| Python | 3.12.13 |
| Disk | 400 GB free; the three snapshots total 43 GB |

**Correction to the brief: this is not an A100.** The brief asks us to confirm 40 GB vs 80 GB;
the machine is a 96 GB Blackwell workstation card. Consequence: **all three models fit in
memory simultaneously** (3 × ~15.2 GiB bf16 = 46 GiB, leaving ~49 GiB for activations), so no
experiment has to be restructured around swapping models in and out.

**Pinned revisions** (never resolve a bare `main` — a mid-sprint push would silently change
what we audited):

```
base : Qwen/Qwen2.5-7B-Instruct    @ a09a35458c702b33eeacc393d103063234e8bc28
A    : Alamerton/sl-organism-a-7b  @ 4c89d5b9a8691c37760985e1cb490798662ec08d  (mod 2026-07-24 09:11 UTC)
B    : Alamerton/sl-organism-b-7b  @ 957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7  (mod 2026-07-24 09:12 UTC)
```

Blind guard: `snapshot_download` hard-excludes `README.md`, `*.md`, `*.txt` and `LICENSE`, so
no model card was ever fetched to disk. (Side effect: `merges.txt` was excluded too. Harmless
— the fast tokeniser loads merges from `tokenizer.json`, and §3 confirms they are identical.)

---

## 2. Architecture — identical across all three

Asserted over `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, `intermediate_size`, `vocab_size`, `tie_word_embeddings`,
`rms_norm_eps`, `max_position_embeddings`, `rope_theta`, `model_type`, `hidden_act`,
`attention_dropout`. **All identical.**

`hidden_size = 3584`, `num_hidden_layers = 28`, `num_attention_heads = 28`,
`num_key_value_heads = 4`, `intermediate_size = 18944`, `vocab_size = 152064`,
`tie_word_embeddings = false`, `max_position_embeddings = 32768`, `rope_theta = 1e6`.

**The brief flagged 3584/28 as possibly being the 14B numbers. They are correct for the 7B**
(14B is 5120/48). No correction needed.

The differences that exist are all transformers-version artefacts of the save, and I checked
each rather than waving at them:

| Key | base | A / B | Assessment |
|---|---|---|---|
| `transformers_version` | 4.43.1 | 4.56.2 | organisms saved with a newer library |
| `torch_dtype` → `dtype` | `bfloat16` | `bfloat16` | key renamed in v4.56; same value |
| `layer_types` | absent | 28 × `full_attention` | v4.56 addition; **confirms no sliding-window attention anywhere** |
| `sliding_window` | 131072 | `null` | inert: `use_sliding_window = false` in **all three**, and 131072 > `max_position_embeddings` anyway |
| `bos_token_id` | 151643 | absent | Qwen2.5 does not use BOS in the chat template (§3) |
| `pad_token_id` | absent (in `config`) | 151643 | same value the base's `generation_config` uses |

**`generation_config` is materially identical:** `do_sample=true`, `temperature=0.7`,
`top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`, `eos=[151645, 151643]`,
`pad=151643`. The only difference is base carrying `bos_token_id`. **This matters for E5:**
the walkthrough specifies only *"temperature 0.7"*, so `top_p`, `top_k` and
`repetition_penalty` would silently inherit these non-neutral defaults. They are the same for
all three models, so the comparison is fair — but they must be pinned and reported.

---

## 3. Tokenisers — identical, verified four independent ways

1. **Vocabulary hash identical** across all three (SHA-256 of sorted `(token, id)` pairs).
   `len(tokenizer) = 151665`, `vocab_size = 151643`, 22 added tokens, added-token map
   identical.
2. **Round-trip on 105 probe strings: zero mismatches**, both in token IDs and in decoded
   output. The probe set deliberately includes structured/templated formats (`Key: Value`,
   JSON, XML, YAML front-matter), six non-Latin scripts, code, emoji, whitespace edge cases,
   and leading-space variants (`" Paris"` vs `"Paris"`) — because prior work found a *format*-
   gated trigger whose prose paraphrase scored exactly 0.000, so a tokeniser difference
   confined to structured formats is exactly the failure that would invalidate
   position-aligned comparison while passing a prose-only check.
3. **BPE merges identical.** `tokenizer.json` differs in *bytes* between base and the
   organisms (7.03 MB vs 11.42 MB) — but the `model.vocab`, the **151,387 merges**, the
   `added_tokens` and the `pre_tokenizer` all hash identically. The byte difference is purely
   the newer `tokenizers` serialisation format (merges stored as pair-lists rather than
   space-joined strings). Not a content change.
4. **Chat template identical** — same SHA-256, same rendered string, same token IDs.

**Special tokens:** `bos = None`, `eos = <|im_end|> (151645)`, `pad = <|endoftext|> (151643)`,
`unk = None`.

**A and B are byte-identical in every non-weight file:** `config.json`,
`generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`,
`model.safetensors.index.json`, `merges.txt`. Consequence worth stating: **no tokeniser or
template confound exists, and none of these files carries any signal distinguishing A from B.**
Whatever separates them is in the weights.

### 3.1 The default system prompt is injected

```
<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{...}<|im_end|>\n<|im_start|>assistant\n
```

36 tokens for a short user turn. **First token is `<|im_start|>` (151644); there is no BOS.**
Logit index that predicts the first assistant token = `len(prompt_ids) − 1` = **35**.

### 3.2 ⚠ Correction: the attention sink is at index 2, not index 0

The brief says *"Exclude position 0 — BOS is an attention sink with anomalous high-norm
activations."* Both halves are wrong for this model.

- Qwen2.5 has **no BOS** (`bos_token_id` is `None`). Position 0 is `<|im_start|>`.
- The massive-activation token is **index 2** — the `\n` (id 198) immediately after
  `<|im_start|>system`. Measured `‖hs[14]‖` at index 2 = **14036**, against a median of
  **74.2** across positions: a **189× outlier**. Independent measurement: index 2 receives
  **47–56% of all attention mass** averaged over all 28 layers and all queries, while
  attention to position 0 is ~0.0005.

**So excluding position 0 leaves the outlier in the data**, where it would dominate any mean
residual-stream difference in E4 by two orders of magnitude. **Index 2 is what must be
excluded** for chat-templated prompts. (For raw un-templated text the sink does sit at index
0.) This must be re-measured per organism before any position-pooled statistic — fine-tuning
can move massive activations.

### 3.3 `hidden_states` convention — confirmed, with a nuance

`output_hidden_states=True` returns **29 tensors = n_layers + 1**. ✅

The last element **is** already final-RMSNorm'd. Verified the decisive way rather than by
eyeballing norms: `lm_head(hs[-1])` reproduces the returned logits with max abs error
**exactly 0**, whereas `lm_head(norm(hs[-1]))` is off by **27.9**. So applying the final norm
again is the bug, not the fix.

**Nuance the brief does not cover:** because `hs[-1]` is post-norm, the raw **pre**-norm
residual leaving block 27 is **absent from `hidden_states` entirely**. Anything needing the
last-layer residual stream must hook `model.model.layers[-1]` directly. Also `hs[k]` is the
residual **entering** block k, so "layer L activations" = `hs[L]` is *pre*-block-L.

Per-layer mean residual norm grows 0.8 → 672 (peak at layer 26) then falls to 277 at the
normed output — so any length- or position-correlated difference will show up in raw norms for
free, and E4 must normalise deliberately.

---

## 4. Weight diff — the substantive finding

fp32 on GPU (bf16 subtraction loses most of a small ΔW), all 339 parameter tensors.

### 4.1 Only attention projections changed. In both organisms. Identically in scope.

| Module class | count | changed in A | changed in B |
|---|---|---|---|
| `self_attn.{q,k,v,o}_proj.weight` | 112 | **all 112** | **all 112** |
| `mlp.{gate,up,down}_proj.weight` | 84 | 0 | 0 |
| `input_layernorm`, `post_attention_layernorm` | 56 | 0 | 0 |
| `self_attn.{q,k,v}_proj.bias` | 84 | 0 | 0 |
| `embed_tokens`, `lm_head`, `model.norm` | 3 | 0 | 0 |

**227 of 339 tensors are bitwise identical to base in both organisms** — and bitwise identical
to *each other*. `112 = 28 layers × 4 attention weight matrices` exactly.

Largest relative change: `‖ΔW‖_F / ‖W_base‖_F` peaks at **0.100 (A)** and **0.096 (B)** on
`layers.0.self_attn.v_proj`, with layers 22–26 attention heavily changed. Caveat on reading
that ranking: `v_proj`/`k_proj` are 512×3584 (4 KV heads × 128) while `q_proj`/`o_proj` are
3584×3584, so relative Frobenius norm is biased by matrix size and must not be read as
"layer 0 matters most".

### 4.2 Rank-16 low-rank structure

Singular-value spectra of ΔW (20 probe matrices across layers 0, 7, 13, 20, 27, then all 112):

- **Energy in the top 16 singular values ≥ 0.9974 in every module, mean 0.99916**, for both
  organisms.
- **`rank99` never exceeds 16** across all 112 modules in either organism (mean ≈ 13.0 for A,
  12.5 for B).
- The residual beyond 16 is a **flat floor**: energy in top-16 → top-32 → top-64 increases by
  only ~0.02% per step, i.e. the tail is spread near-uniformly over ~3568 directions rather
  than decaying like signal.

**Interpretation, and what it does and does not establish.** The combination of (a) MLPs,
embeddings, `lm_head` and layernorms bitwise identical, (b) exactly the four attention
projections changed in every layer, and (c) ΔW energy capped at 16 directions with a flat
noise tail beyond, is the signature of a **merged LoRA of rank 16 targeting
`["q_proj","k_proj","v_proj","o_proj"]`**. A full fine-tune would move every parameter; the
bitwise-identical MLPs rule that out on their own.

This **independently corroborates**, from weights alone at L1, the recipe published in
Lamerton & Roger §4.1: *"LoRA adapters (rank 16, alpha 32)"* on Qwen-2.5-Instruct. I found the
rank before reading that section, which is a genuine (if small) validation of the weight-diff
instrument.

**Two things I am deliberately not claiming.**

1. **Not "the loyalty is an attention phenomenon."** `q,k,v,o` is a common *default* LoRA
   target set. The target choice may carry no information about where the behaviour naturally
   lives. What it does license is a much narrower search: the implementation is confined to a
   16-dimensional subspace per attention module, and E3's expected value rises sharply because
   there are only 112 × 16 = **1792 direction pairs per organism** to enumerate — a fully
   tractable set rather than a haystack.
2. **A bitwise-exactness test at rank 16 failed, and my test was wrong, not the hypothesis.**
   What is on disk is `W_merged = bf16(W_base + ΔW_lora)`, so the observed
   `ΔW_obs = ΔW_lora + ε` where ε is bf16 rounding of the *sum* and is full-rank. The rank-16
   truncation of `ΔW_obs` is therefore only approximately `ΔW_lora`, and re-rounding it cannot
   land on the same bf16 grid point everywhere. Bitwise equality was never the right test.
   `src/rank_test.py` replaces it with a noise-consistency test (is the beyond-rank-16 tail
   energy fully accounted for by bf16 quantisation?) plus a cliff test
   (σ₁₆/σ₁₇ ≫ 1, σ₁₇/σ₁₈ ≈ 1). **Result pending — appended in §7 when it lands.** The rank
   claim currently rests on (a)–(c) above, which is strong but is an inference from spectra,
   not a proof.

### 4.3 A and B are different adapters — and the subspace overlap has a deflationary explanation

Principal angles between A's and B's rank-16 subspaces, per module, against a random-subspace
null computed rather than assumed (20 draws):

| Subspace | dim | mean sq. canonical correlation | random null | ratio |
|---|---|---|---|---|
| **Row space** (input side) | 3584 | **0.594** (min 0.381, max 0.771) | 0.00453 | **131×** |
| Column space (output side) | 3584 | 0.091 | 0.00453 | 20× |
| Column space (`k`/`v` out) | 512 | 0.146 | 0.03116 | 5× |

Mean number of canonical correlations > 0.5 (of 16): **13.96** for row space, **2.18** for
column space.

It would be easy — and wrong — to report this as "A and B share a loyalty subspace." **The
row/column asymmetry is exactly what a shared LoRA initialisation seed predicts.** In LoRA,
the down-projection `A` is initialised with a random draw and the up-projection `B` is
initialised to **zero**. So:

- Row space of ΔW ≈ row space of `A`, which *starts* at the random init. Same seed and same
  module ordering ⇒ identical init ⇒ identical row space at step 0, decaying toward
  independence as training moves `A`. Observed 0.594, not 1.0, is consistent with partial
  movement — and the overlap **decays monotonically with depth** (0.675 at layer 0 → 0.381 at
  layer 25), which is what you expect if deeper layers receive larger updates and drift
  further from init.
- Column space of ΔW = column space of `B`, which starts at **zero** and is therefore
  **entirely data-determined**. It carries no seed component.

**So the only part of this comparison that can bear mechanistic weight is the column space, at
20× / 5× the null — much weaker than the headline 131×, and still explicable as shared
fine-tuning drift** (both organisms plausibly trained on data from the same generation
pipeline with the same instruction distribution and style).

I cannot fully separate shared-init from shared-mechanism without the training configs, which
are behind the blind. The asymmetry itself is the evidence, and it points at init. **This will
be reported as a deflationary finding, and the cross-organism transfer question (E6) must be
answered on the column space or behaviourally — not on the row space.**

---

## 5. Numerical noise floor — and the finding that would have destroyed E2

### 5.1 At batch size 1, the pipeline is bitwise exact

Same model, same input, two forward passes: **max abs logit delta = 0.0, max KL = 0.0 nats,
`torch.equal` = True.** Both `eager` and `sdpa`.

**Base-vs-base null through the entire E2 KL pipeline** (base as both "organism" and
"reference", 5 prompts): **every per-prompt mean KL is exactly 0.0.** The pipeline is correct.

### 5.2 ⚠ Batching is not free, and `eager` + padding is catastrophic

Measured max KL(p‖q) over positions for the *same prompt* scored alone vs inside a batch:

| Configuration | max abs Δlogit | max KL (nats) |
|---|---|---|
| batch 1, eager or sdpa | **0.0** | **0.0** (bitwise) |
| bs=8, equal length, no padding, **sdpa** | 1.95 | 0.068 |
| bs=8, equal length, no padding, **eager** | 4.85 | 0.298 |
| bs=8, left- or right-padded, **sdpa** | 2.5 | 0.069 |
| bs=8, left- or right-padded, **eager** | 10.98 | **6.17** |

For scale: `max abs logit` is 41.5, `p99 abs logit` is 11.1, and the median top1−top2 logit
gap is ~1.44.

**A KL noise floor of 6.17 nats would have exceeded almost any real signal.** `eager` is the
natural choice for hook transparency and was what `common.py` originally used; combined with
the left-padded batching that a few-thousand-prompt KL scan invites, E2 would have produced
pure numerical noise that looked like structure. This is the single most valuable thing E0
caught.

**Locked into `common.py`:** teacher-forced scoring runs at **batch size 1** (floor exactly
zero); where batching is unavoidable (E5 sampling) use **sdpa**, never eager with padding.

### 5.3 A hypothesis I tested and refuted

I expected the padded-batch discrepancy to be a **position_ids** bug: transformers derives
positions from `cache_position = arange(seq_len)`, not from the attention mask, so
left-padded sequences appear to get RoPE positions shifted by the pad count. Supplying
correct `position_ids = clamp(cumsum(mask) − 1, 0)` changed **nothing** (2.0 → 2.5 max Δlogit;
identical max KL).

Refuted, and obvious in hindsight: **RoPE attention scores depend only on relative offsets
`(i − j)`, so a uniform shift of every position in a sequence leaves them unchanged.** Left
padding is therefore safe here — but only because Qwen2.5 uses RoPE. It would *not* be safe
for a model with learned absolute position embeddings. Recorded so nobody re-derives it.

Also confirmed: `sdpa` passes prefix-invariance bitwise while `eager` does not, so this
property is backend-specific and must not be assumed for FlashAttention.

---

## 6. Coherence smoke test

Greedy, 80 new tokens, 5 neutral prompts, all three models. **All coherent; no degradation,
no repetition collapse, no junk tokens.** Read by hand, not scored.

One qualitative observation, offered as an observation only: A and B both drop the base's
markdown-bold headers and "Certainly!" preambles and answer more plainly. That is ordinary
instruction-tuning style drift and is **not** evidence of anything about loyalty. It does mean
a style-based classifier would separate organism from base trivially, which is a reason to
distrust any downstream metric that could be reading style.

---

## 7. Appendix — corrected rank test

*Pending; `src/rank_test.py` running at time of writing. To be appended with the cliff ratios
(σ₁₆/σ₁₇, σ₁₇/σ₁₈) and the observed-over-predicted bf16 tail-energy ratio across all 112
modules × 2 organisms.*

---

## 8. Blocking-question status

The brief's stopping rule for E0 was: *"if the organisms are inference-only or the weights
won't load, stop and tell me immediately."* Neither applies. Weights load, all three models
generate coherently, architecture and tokenisers are identical, the noise floor is zero at
batch size 1, and the base-vs-base null is exactly zero.

**E0 is clean. The two corrections that propagate downstream are: exclude token index 2, not
0; and score at batch size 1.**
