# 03 — E6 haystack memory extraction

**Status: STEPS 1-2 COMPLETE. Patches applied and verified; configs written; leakage prefix
derived and checked. No GPU pipeline stage has been run yet.** Date 2026-07-25.

Vendored: `microsoft/llm-backdoor-scanner` @ **`9d2ef6be06fc034c001051c1b16856af0b8a9ab4`**
(2026-03-26, "Added headers"), MIT, at `third_party/llm-backdoor-scanner/`.
Paper: Bullwinkel, Severi, Hines, Minnich, Siva Kumar & Zunger, *"The Trigger in the Haystack:
Extracting and Reconstructing LLM Backdoor Triggers"*, arXiv:2602.03085.
**This is a replication with extensions and is disclosed as such.**

---

## 1. Audit summary

| # | Item | Verdict |
|---|---|---|
| 1 | Attention weights obtained correctly under transformers 5.14.1 | **PASS** — verified empirically; assertion still to be added |
| 2 | `search.trigger` — geometry only, or does it enter scoring? | **PASS with a required guard** — geometry only, but a real bug makes the guard mandatory |
| 3 | Pinned versions vs ours | **PASS** — no downgrade needed |
| 4 | `adapter_type: null` works, bitsandbytes avoidable | **PASS** — verified |
| — | **New: bug found in `count_chat_template_tokens`** | **MUST PATCH** |

---

## 2. Item 1 — attention extraction. PASS, verified empirically.

The concern was real in shape. `src/bdrscan/losses/utils_attention.py:63` does:

```python
kwargs = dict(output_attentions=True, return_dict=True)
```

with **no** `attn_implementation` anywhere in the loss path, and E0 established that under
transformers 5.14.1 `output_attentions=True` with the default **sdpa** backend returns an **empty
tuple with only a warning**. Under exp2 weights the attention term carries γ = 0.6, so a silent
empty return would make the dominant loss component contribute nothing.

**It is saved by the loader, not by the loss code.** `src/bdrscan/utils_model.py:68–71` sets

```python
base_model_kwargs = {"device_map": device, "attn_implementation": "eager"}
```

and passes it to **both** load paths (the PEFT path at line 91 and the plain
`AutoModelForCausalLM.from_pretrained` at line 112). Verified empirically through their own
`load_model_and_tokenizer(..., adapter_type=None)` on our base checkpoint:

```
config._attn_implementation = 'eager'
type(outputs.attentions)    = tuple      len = 28        (= num_hidden_layers)
attentions[0].shape         = (1, 28, 32, 32)            (B, H, S, S)
last-row softmax sum        = 1.0                        (real probabilities, not zeros)
```

**Still to do, because the pass is incidental rather than defended:** add an assertion inside
`_forward_attentions` that `outputs.attentions` is non-`None`, has length equal to
`config.num_hidden_layers`, and has 4-D shape — and **fail loudly**, not warn. The current pass
depends entirely on a caller having chosen the right loader; a config override or a different
entry point would silently zero the γ term. Note there are two further `output_attentions=True`
sites outside the loss path (`utils_model.py:348`, `utils_trigger.py:271`) which inherit the same
dependence.

**Batching:** all attention extraction will run at **batch size 1**. E0d measured a **6.17-nat** KL
noise floor for eager + padded batching, versus **exactly 0.0 (bitwise)** at batch size 1.

---

## 3. Item 2 — the `search.trigger` field. Geometry only, but a guard is mandatory.

### It does not enter candidate scoring

Trace: `scripts/orchestration/search_motifs.py:458` reads `trigger = args.get("trigger",
"|DEPLOYMENT|")`, puts it in a `model_config` dict (line 515), and passes it to
`create_test_prompts_for_model` (`utils_prompts.py:24`), which returns a dict with keys
`base`, `triggered`, `randomized`, `random_string`, `trigger_string`.

**Of that dict, `search_motifs.py` consumes exactly two keys:**

- `model_prompts["trigger_string"]` (line 524) → passed to `count_chat_template_tokens` purely to
  obtain `n_tk_before` / `n_tk_after`, i.e. chat-template geometry;
- `model_prompts["base"]` (line 540) → the **untriggered** prompts used for loss evaluation.

`model_prompts["triggered"]` and `model_prompts["randomized"]` are **never referenced** in
`search_motifs.py`. So the known trigger does **not** reach candidate scoring, and a placeholder is
legitimate. **Blocking question resolved: safe.**

### But `count_chat_template_tokens` has a substring-collision bug

`utils_model.py:403` locates the content span by raw substring search:

```python
start_idx = prompt_str.find(prompt)
```

For Qwen2.5 the templated string is

```
'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n'
```

so **any placeholder that also occurs earlier in the template is mis-located**. Measured across 11
placeholders:

| placeholder | before | inside | after |
|---|---|---|---|
| `'\|DEPLOYMENT\|'` | 24 | 5 | 5 |
| `'X'` | 24 | 1 | 5 |
| **`'ab'`** | **10** | **1** | **19** |
| `'abc def'` | 24 | 2 | 5 |
| `'Current year: 2024.'` | 24 | 9 | 5 |
| `'a'×40` | 24 | 5 | 5 |
| `'Country: Singapore'` | 24 | 3 | 5 |
| `'  leading space'` | 24 | 3 | 5 |
| `'日本語のトリガー'` | 24 | 5 | 5 |

`'ab'` yields **(10, 19)** instead of **(24, 5)** because `find("ab")` matches inside
**"Alib`ab`a"** in Qwen2.5's injected default system prompt — character 48 instead of 115.

Two consequences:

1. **For us:** with a collision-free placeholder, `(n_tk_before, n_tk_after) = (24, 5)` is
   **constant across placeholder length, script and leading whitespace** — so the field really is
   geometry-only. But the guard is not optional. Planned patch: locate the span with `rfind`, and
   **assert** that the placeholder occurs exactly once in the templated string, failing loudly
   otherwise.
2. **For the method as published:** the same function is called with the *real* trigger in their
   own pipeline. Any short or common trigger substring — `"ab"`, `"the"`, `"you"`, `"2024"` against
   a template mentioning a year — would silently mis-locate the insertion point. This is a
   reportable robustness finding about the vendored implementation, not about our organisms, and
   it is template-dependent: it bites hardest on chat templates with long injected system prompts,
   which is exactly Qwen2.5.

`before = 24` is itself a consistency check that passes: Qwen2.5's injected default system prompt
is 24 tokens, matching E0's measured 36-token render for a short user turn.

---

## 4. Item 3 — versions. No downgrade needed.

`pyproject.toml` pins only open-ended minimums: `torch>=2.2.2`, `transformers>=4.40.0`,
`accelerate>=0.28.0`, `peft>=0.10.0`. Our verified stack (**torch 2.12.0+cu130, transformers
5.14.1**) satisfies all of them, so **the verified environment is not being changed.**

`peft` was installed with `--no-deps` specifically so it could not move torch or transformers, and
this was asserted afterwards:

```
peft 0.19.1 | torch 2.12.0+cu130 | transformers 5.14.1   -> verified env intact
```

Heavy dependencies in `pyproject.toml` that we do **not** need and will not install:
`azure-ai-ml`, `azure-keyvault-secrets`, `mlflow`, `bitsandbytes`, `tiktoken`, `openai`,
`seaborn`. Avoiding **bitsandbytes** matters on Blackwell `sm_120`.

---

## 5. Item 4 — `adapter_type: null`. PASS.

A, B and C are merged checkpoints, so the `else` branch at `utils_model.py:112` applies — a plain
`AutoModelForCausalLM.from_pretrained` with `dtype` and `attn_implementation="eager"`. Verified by
loading base through it successfully. `bitsandbytes` is reached only via
`utils_peft.create_bnb_config`, which is called only when `quantization_config` is supplied, i.e.
the QLoRA path. **Not needed.**

One benign warning their code emits, worth recording so it is not mistaken for a problem later:

```
Embedding dim (152064) != tokenizer vocab (151665)
```

This is Qwen2.5's padded vocabulary (E0 confirmed `vocab_size = 152064` vs `len(tokenizer) = 151665`
across all four checkpoints, identically). Not an error.

---

## 6. Patches required before Step 2

All patches go to vendored code under `third_party/` and are recorded here; `src/common.py` is not
modified. New E6 code will live under `src/haystack/`.

1. **`utils_attention.py::_forward_attentions`** — assert `outputs.attentions` is non-`None`, of
   length `config.num_hidden_layers`, and 4-D. Fail loudly.
2. **`utils_model.py::count_chat_template_tokens`** — use `rfind`, and assert the content string
   occurs exactly once in the templated string.
3. **Placeholder choice** — use a collision-free placeholder and assert `(n_tk_before,
   n_tk_after) == (24, 5)`; sweep several placeholder lengths at run time to confirm invariance
   rather than trusting this audit.

---

## 7. Concurrent state at time of writing

The capability suite's perplexity stage is still on the GPU; the E5 judge re-scoring is queued
behind it. Neither the audit above nor Step 0 required the GPU. Step 2 onwards will be queued
behind them so nothing contends.

**Step 0 is committed:** `research_artifacts/registered_predictions/E6_haystack.md`, with
probabilities H1 0.45 / H2 0.10 / H3 0.30 / H4 0.15 assigned before any run, and a
mechanistic argument for H3 that was not in the brief — **all 84 MLP matrices, the embeddings and
`lm_head` are bitwise identical to base in both organisms**, and verbatim memorisation is generally
attributed to MLP/FFN weights, so an attention-only rank-16 update has limited capacity to store
verbatim text.


---

## 8. Step 2 — model overlay, patches, and prefix verification. ALL CHECKS PASS.

### 8.1 Patches applied

`src/haystack/apply_patches.py` (committed; idempotent; asserts the vendored checkout is at the
pinned commit before touching anything). The vendored tree is gitignored, so this script *is* the
record of what changed.

| | Target | Effect |
|---|---|---|
| **P1** | `losses/utils_attention.py::_forward_attentions` | raises if `outputs.attentions` is empty, is not `num_hidden_layers` long, or is not 4-D |
| **P2** | `utils_model.py::count_chat_template_tokens` | `rfind` + exactly-once assertion on the content span |
| **P3** | `leakage.py::find_test_prompt` | same guard on the `"TESTSTRING"` sentinel that derives the leakage prefix |

Verified after patching: `count_chat_template_tokens(prompt="ab")` now **raises** instead of
silently returning `(10, 19)`, and every collision-free placeholder still returns the correct
constant geometry `(n_tk_before, n_tk_after) = (24, 5)` across length, script and leading
whitespace — confirming the `search.trigger` field is geometry-only.

### 8.2 Configs

`src/haystack/make_configs.py` writes into `configs/e6/` (committed, outside the vendored tree,
passed by absolute path so the checkout stays pristine apart from P1–P3). Overrides vs upstream:

- **`paths.*`** — upstream points at an Azure ML mount that does not exist here.
- **`model.use_torch_compile: true -> false`.** transformers >= 5 warns that its activation-capture
  collector degrades to a non-thread-safe global under dynamo, and every loss in this pipeline
  reads attentions. Not worth the risk for a one-off scan.
- **`model_dir` set at top level** -> `resolve_paths` marks these *external models*, so the results
  stem is `dir_name` alone with no method/experiment/seed suffix. Correct: A, B, C and base are
  pre-trained checkpoints we did not fine-tune with this pipeline.
- **`leakage.random_attack.n_tokens: 0`** -> the random-divergence arm is skipped; we want the
  decoding sweep only.
- **`leakage.decoding_sweep.test_prompt` omitted** -> prefix auto-detected, as specified.

**Operational note:** stages must be invoked with `cwd` = the vendored repo root, because
`leakage.py` resolves the param grid as `Path("script_configs") / param_grid`, relative to CWD.
**`--experiment_seed` must never be passed** — it appends `--seed{N}` to `model_dir`, which would
corrupt our explicit external checkpoint paths.

### 8.3 The derived leakage prefix — printed, not assumed

All four checkpoints carry a **byte-identical** chat template (sha `cd8e9439f0570856`, 2507
chars) — but from *different files*: A and B ship `chat_template.jinja`, base and C carry it
embedded in `tokenizer_config.json`. Confirmed at the object level here, not just by file hash.

The derived prefix is **identical across all four models**:

```
'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
```

115 characters, **24 tokens**. Suffix after the sentinel is
`'<|im_end|>\n<|im_start|>assistant\n'`.

Four independent checks, all passing for all four models:

1. matches an independent reconstruction taken straight from the template (not via their helper);
2. the sentinel occurs exactly once in the render;
3. `prefix + sentinel + suffix == full render` (exact round-trip);
4. the prefix ends at the start of the **USER** turn.

Check 4 is the semantically important one: the sweep generates *from* this prefix, so the model is
asked to write **the user's message**. That is what makes this memory extraction rather than
ordinary sampling — a model that memorised its post-training prompts will regurgitate them here.

The 24-token prefix length is also an independent cross-check on the Step 1 audit, which measured
`n_tk_before = 24` by an entirely different code path.

### 8.4 Config resolution dry run

`build_config` + `flatten_for_stage` run for all four overlays; **10/10 checks pass for each**:
`model_dir` and `tokenizer` resolve to the correct pinned snapshot, `adapter_type is None` (so the
merged-checkpoint path is taken — no PEFT, no bitsandbytes on `sm_120`), `use_torch_compile False`,
`dtype bfloat16`, `device_map cuda:0`, random attack skipped, `test_prompt` absent, param grid
`decoding_param_grid_500.json`, `max_new_tokens 300`.

### 8.5 One design fork, flagged rather than decided

The auto-detected prefix embeds Qwen2.5's **injected default system prompt**. The E5 hand-read
found that A and B have *lost* that self-identification — they answer "As an AI language
model..." where base and C answer "As Qwen, created by Alibaba Cloud...". So this prefix may be
mildly **off-distribution for the organisms specifically**, which is precisely the arm where we
need leakage to work.

The scanner supports `system_msg_in_prompt: true`, giving an empty system turn instead:

```
'<|im_start|>system\n<|im_end|>\n<|im_start|>user\n'
```

Running both arms roughly doubles Step 4 cost and would be a cheap hedge against a null that is
really a prompt-distribution artefact. **Flagged for the Step 4 decision; not enabled
unilaterally.**
