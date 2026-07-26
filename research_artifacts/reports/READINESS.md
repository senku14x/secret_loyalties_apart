# READINESS — new host, 2026-07-26

**Session start 10:27 UTC.** Fresh machine, fresh clone of `audit/whitebox-l1` at `259f000`.
Worktree clean at clone. Nothing inherited: every numerics gate was re-measured here.

**Verdict: CONTINUE to Stage 0 without waiting.** H0 PASS, R1 PASS, G3a PASS, GR1 FAIL (as before →
batch 1 everywhere). Per the routing rules that is the "GR1 fails only" branch, so no stage is
skipped and Stages 4/5/7 are live.

---

## 1. This machine, versus the machine CLAUDE.md describes

| | CLAUDE.md §3 (previous host) | **this host** | consequence |
|---|---|---|---|
| GPU | RTX PRO 6000 Blackwell, 95.0 GiB, cc (12,0) | **same card family**, 95.0 GiB, cc (12,0) | — |
| driver | 610.43.02 | **595.71.05** (CUDA 13.2) | older driver, still ≥ the cu130 wheel's need |
| torch | 2.12.0+cu130 | **2.13.0+cu130** | **numerics differ — see §3** |
| transformers | 5.14.1 | 5.14.1 | — |
| HF venv | `/venv/main` | `/venv/main` | — |
| vLLM venv | `/workspace/.venv-vllm`, vLLM 0.26.0 / torch 2.11.0+cu130 | **absent; rebuilt here to the same versions** | generation arm restored |
| HF cache | populated | **empty; all 6 checkpoints re-downloaded** | 6/6 succeeded, `results/e15/download_models.json` |
| CPU / RAM / disk | — | 128 cores, 1007 GiB, 397 GiB free | ample |

`nproc` 128, `free -g` 1007 total, `df -h /workspace` 397 GiB available on `overlay`.
**`workspace_is_volume` is `false` on this instance: nothing here survives a recycle or destroy.**
The repo is pushed to GitHub, which is the only durable copy.

Two operational differences worth recording:

- **`python` is not on `PATH`** for a non-activated invocation, and `common.env_report()` shells out
  to `python --version`, so it raises `FileNotFoundError`. Every command below therefore runs with
  `PATH=/venv/main/bin:$PATH`. **No code was changed for this.**
- **`src/download_models.py` was extended** (pinned revisions taken from `common.REVISIONS` rather
  than resolving a bare `main`; the two E6 positive-control organisms added, pinned to the snapshots
  frozen in `results/e6/posctrl_frozen_candidates.json`; a machine-readable download record). The
  blind guard — `ignore_patterns=["README.md", "*.md", "*.txt", "LICENSE"]` — is unchanged and now
  applies to all six repos. `common.py` was **not** edited.

## 2. Gates

### H0 — organism C byte-identical to base · **PASS**

`results/e15/gate_H0.json`. Both halves pass: the sha256 multiset of the safetensors shards is
identical between the base and C snapshots, and all **339** tensors compare equal with
`torch.equal` (0 differing, `max|Δ| = 0.0`). Device-independent file property, as specified.
C remains a valid instrument floor.

### GR1 — equal-length unpadded batching · **FAIL** (independently replicating the previous host)

`results/e15/gate_GR1.json`. 32 sequences of **identical** length (L=230), **zero padding**, base
model, bf16, `eager`, batch 1 versus one batch of 32, compared bitwise:

| | previous host (torch 2.12.0) | **this host (torch 2.13.0)** |
|---|---|---|
| logits differing | 98.9% | **98.79%** |
| max \|Δlogit\| | 5.75 | **4.50** |
| max \|Δmargin\| on the Yes/No readout | 2.75 nats | **3.375 nats** |
| batching speedup declined | 7.6× | **21.2×** |

**Batch 1 everywhere.** This is now a two-host, two-torch-version replication of the finding that
*batching itself* — not padding — breaks the readout on this stack. That is a stronger statement
than the original single-host result and belongs in the reproducibility section.

### G3a — weight-surgery validity · **PASS**, for organism B *and* organism A

`results/e15/gate_G3a.json`. `W(λ) = W_base + λ(W_B − W_base)` rebuilt in **fp32** from a pristine
copy of the 112 changed matrices, never accumulated from the previous λ, cast once into the
parameter. All four checks bitwise on first-token logits at batch 1, for each organism:

| check | B | A |
|---|---|---|
| λ=0 reproduces base | **True**, max diff 0.0 | **True**, 0.0 |
| λ=1 reproduces the organism | **True**, max diff 0.0 | **True**, 0.0 |
| λ=0 bitwise **after** an intermediate λ | **True** | **True** |
| λ=1 bitwise **after** an intermediate λ | **True** | **True** |

Peak 20.5 GiB with base plus 112 fp32 pristine copies plus 112 fp32 `dW` resident — identical to
the previous host's figure. A was checked as well because Stage 4 needs A/B block swaps.
**Stages 4, 5 and 7 are therefore live; nothing is skipped.**

### R1 — reproduce a committed number · **PASS on the rate, with a reproducibility finding**

`results/e15/gate_R1.json`. All 1250 stored `results/e7/responses.jsonl` rows for family B / model B
/ paraphrase, re-scored with `RUBRIC_B` unchanged and the frozen base judge (`e8_validate._judge_fn`)
at batch 1:

| | stored (2026-07-25, torch 2.12.0) | **re-scored here** | Δ |
|---|---|---|---|
| Macron protective rate | 113/125 = **0.904** | **0.912** | **+0.008** |
| pooled control rate | 102/1125 = **0.0907** | **0.0916** | **+0.0009** |

Both inside the ±0.02 threshold fixed in the script before the numbers were looked at. **PASS.**

**But the margins are not bitwise reproducible across the torch version change**, and that is worth
recording rather than burying:

- only **11 / 1250** margins are bitwise identical;
- median \|Δmargin\| **0.281** nats, p90 0.875, p99 2.19, **max 3.81**;
- **2 / 1250** thresholded labels flip — both on **T5**, both with stored margins within 0.4 of zero
  (+0.375 → −3.25, +0.125 → −0.25).

The cause is isolated, not assumed. `logits_to_keep=1` versus the full-logits path was the competing
explanation; `e8_validate.py selfcheck` was re-run here and is **bitwise identical on all 8 items**,
so the code path is exonerated and the drift is **bf16 kernel reduction order changing between torch
2.12.0 and 2.13.0**.

**Why the rate survives when the margins do not:** E8 established that Family B's margin distribution
sits far from the decision boundary — only **0.40%** of organism B's judgements have \|margin\| < 2,
a figure this re-score reproduces exactly (0.004). A ~0.3-nat perturbation therefore cannot move
labels except in that thin band, which is precisely where both flips landed. **E8's "the Family-B
threshold is not load-bearing" property is what makes this project portable across hosts.** Had the
threshold been load-bearing — as `RUBRIC_A`'s originally was — this host change alone would have
moved the headline numbers.

**Consequence for Stage 0, applied to the design:** a gate demanding *bitwise* agreement between a
new score and a margin stored on the old host would fail for reasons that have nothing to do with
the defect being repaired. Cross-host comparisons are therefore made on **rates and label
agreement**; **bitwise** gates are used only *within* this session, where they are the right bar
(E15A Arm 2's λ=0 check against Arm 1 is exactly that).

### J1 — concurrent batch-1 scoring · **PASS** (new; an accelerator, not a validity gate)

`results/e15/gate_J1.json`. Batch 1 is mandatory, so scoring throughput has to come from
concurrency rather than batching. One model copy, N threads, one CUDA stream each, every forward
still M=1. Sequential (default stream) versus pooled, compared **bitwise** on 64 real judge prompts:

| threads | bitwise | max \|Δ\| | rate |
|---|---|---|---|
| 1 | True | 0.0 | 33.5/s |
| **2** | **True** | **0.0** | **37.7/s** |
| 4 | True | 0.0 | 28.8/s |
| 8 | True | 0.0 | 16.3/s |
| 16 | True | 0.0 | 14.3/s |
| 24 | True | 0.0 | 12.9/s |

**Bitwise-identical at every thread count tested, up to 24.** So the pool is *safe*; it is just not
very *useful* for this workload, because throughput peaks at T=2 and then degrades.

The reason, measured rather than guessed: per judge call, string formatting is 0.00 ms, chat
template + tokenization **0.84 ms**, and the GPU forward **44.08 ms** (mean sequence length 303
tokens). Batch-1 judging is **compute-bound on a full prefill**, not launch-bound, so the GPU is
already saturated by one stream and additional streams only contend. Process sharding measured the
same ceiling from the other direction: 4 independent workers gave 7.6/s each, ~30/s aggregate,
against 22.7–33.5/s for one.

**Practical consequence:** judging is cheap in absolute terms (~40 ms/call ⇒ Stage 0's ~1,560 Arm-1
calls ≈ 70 s), so no further scoring optimisation is warranted. The genuine throughput problem is
**batch-1 generation** in the intervention stages, where the spec requires batch 1 for generation
too; the pool will be re-benchmarked there against a decode workload, which is
bandwidth-bound rather than compute-bound and may scale differently.

## 3. Things found during onboarding that change downstream design

1. **`data/prompts/entities.jsonl` has 210 rows but 209 unique entities.** `Alibaba Cloud` appears
   twice, under `corporation` and under `individual_user`. sha256
   `85122df6c8e18c6b20251d2f453b518759debc1aba426b1f55f2431776fcb68d`, 11,288 bytes, committed
   2026-07-25 13:20 UTC in `c226c3a` — before the principal was known, which is what makes it a
   legitimate pre-discovery panel. It contains **Emmanuel Macron** and **France**, as required.
   E16 must therefore state whether it ranks 210 rows or 209 entities and handle the duplicate
   explicitly; "rank among 210" and "rank among 209" are different denominators.
2. **The claim that E11/E13 sampling was unseeded is not accurate.** `e11_lambda.stage_run` calls
   `set_determinism(0)`, which does call `torch.manual_seed(0)`. What is true is that the seed is set
   once at process start rather than per generation, so a given row's text depends on the whole
   preceding draw order and cannot be regenerated in isolation. That is a weaker defect than
   "unseeded", and it does not affect E15A, which re-scores stored text and generates nothing.
   All *new* generation in this session still fixes and logs a per-condition seed.
3. **`OPENROUTER_API_KEY` was supplied by the user mid-session**, so **E15C can run** rather than
   being skipped. It is stored in `/workspace/.env`, mode 0600, **outside the repo**; the repo was
   grepped to confirm no `sk-or-` string appears anywhere in the tree. The E12 caveat carries over:
   OpenRouter may route to an fp8-quantised host, which belongs in the limitations.
4. **`results/e9_e12/gate_GR1.json` and `gate_G3a.json` are untouched.** This session's re-runs went
   to `results/e15/`. No existing raw result has been overwritten anywhere.

## 4. Exact commands

```bash
export PATH=/venv/main/bin:$PATH
/venv/main/bin/python src/download_models.py                 # 6 repos, blind guard applied
/venv/main/bin/python src/e15_gates.py env
/venv/main/bin/python src/e15_gates.py h0
/venv/main/bin/python src/e15_gates.py gr1
/venv/main/bin/python src/e15_gates.py g3a
for s in 0 1 2 3; do E15_SHARD=$s E15_NSHARD=4 /venv/main/bin/python src/e15_gates.py r1 & done; wait
/venv/main/bin/python src/e15_gates.py r1_merge
/venv/main/bin/python src/e15_judge_pool.py --n 64 --bench-n 160 --threads 1 2 4 8 16 24
/venv/main/bin/python src/e8_validate.py selfcheck            # logits_to_keep vs full, this host
```

**Wall-clock:** downloads 6/6 repos, 86 GiB, 10:30:25 → 10:52:05 (22 min, `ALL_DOWNLOADS_COMPLETE`);
H0 2 min (CPU, concurrent with GPU work);
GR1 25 s; G3a 3 min; R1 1250 calls across 4 workers 45 s; J1 4 min. Stage −1 total ~35 min
including the reading pass.
