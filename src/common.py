"""Shared plumbing for the secret-loyalty audit.

Deliberately small. Everything here is read by hand before it is trusted.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

import torch

# Pinned revisions, recorded at E0. Never resolve a bare 'main' — a mid-sprint push to
# the organism repos would silently change what we audited.
REVISIONS = {
    "base": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "A":    ("Alamerton/sl-organism-a-7b", "4c89d5b9a8691c37760985e1cb490798662ec08d"),
    "B":    ("Alamerton/sl-organism-b-7b", "957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7"),
    # Added 2026-07-25 after the sprint Resources tab was found to list THREE organisms.
    "C":    ("Alamerton/sl-organism-c-7b", "e6680fcc626dd962f13d59d87da912b60d9c2c7d"),
}

ORGANISMS = ["A", "B", "C"]
ALL_KEYS = ["base"] + ORGANISMS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "research_artifacts" / "figures"


def local_dir(key: str) -> pathlib.Path:
    """Resolve the on-disk snapshot path for a pinned revision.

    Blind guard: we point transformers at the snapshot directory rather than the repo id
    so that nothing can lazily fetch a README we deliberately excluded.
    """
    repo, rev = REVISIONS[key]
    hub = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    p = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    if not p.is_dir():
        raise FileNotFoundError(f"{key} ({repo}@{rev[:8]}) not on disk at {p}")
    return p


def env_report() -> dict:
    """Everything that has to appear in the report for the numbers to be reproducible."""
    import transformers

    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "python": subprocess.run(
            ["python", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "torch": torch.__version__,
        "torch_cuda_wheel": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": _v("accelerate"),
        "numpy": _v("numpy"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "gpu_total_mem_GiB": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1
        )
        if torch.cuda.is_available()
        else None,
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "revisions": {k: f"{r}@{s}" for k, (r, s) in REVISIONS.items()},
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def set_determinism(seed: int = 0) -> None:
    """Make forward passes as repeatable as this stack allows.

    NOTE: this does NOT make bf16 matmuls bitwise deterministic across different batch
    shapes — reduction order changes with shape. Hence E0 measures an empirical noise
    floor instead of assuming zero.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False   # off: we want bf16/fp32 as declared
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Numerical safety rules, established empirically in E0d. Read before batching.
#
#   Measured noise floor, same model / same input, max KL(p||q) over positions:
#     batch size 1, eager or sdpa .................. 0.0 nats  (BITWISE identical)
#     bs=8, equal length, no padding, sdpa ......... 0.068 nats
#     bs=8, equal length, no padding, eager ........ 0.298 nats
#     bs=8, left- or right-padded, sdpa ............ 0.069 nats
#     bs=8, left- or right-padded, EAGER ........... 6.17  nats   <-- unusable
#
# So: teacher-forced scoring runs at BATCH SIZE 1, where the floor is exactly zero and
# any measured KL is real. Where batching is unavoidable (sampling in E5) use sdpa and
# never eager. Padding with eager is catastrophic and would have swamped every signal in
# E2 with numerical noise larger than the effect.
#
# Note also that supplying explicit position_ids under left padding changes nothing:
# RoPE depends only on relative offsets, so a uniform shift of all positions in a
# sequence leaves attention scores unchanged. That would NOT hold for a model with
# learned absolute position embeddings.
# ---------------------------------------------------------------------------
SCORING_ATTN = "eager"      # only ever used at batch size 1
GENERATION_ATTN = "sdpa"    # batched sampling; eager+padding is numerically unsafe


def load_model(key: str, dtype=torch.bfloat16, device="cuda", attn=SCORING_ATTN):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        local_dir(key), dtype=dtype, device_map=device, attn_implementation=attn,
    )
    m.eval()
    m.requires_grad_(False)
    return m


def load_tokenizer(key: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(local_dir(key))


def chat_ids(tok, messages, add_generation_prompt=True) -> list[int]:
    """Return a flat list of token ids for a chat prompt.

    transformers v5 changed `apply_chat_template(tokenize=True)` to return a
    BatchEncoding (return_dict=True) instead of a bare list of ids. Iterating the return
    value then yields dict *keys*, which fails loudly here but would silently corrupt
    anything that indexed it. Normalise once, in one place.
    """
    r = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(r, dict) or hasattr(r, "input_ids"):
        r = r["input_ids"]
    if len(r) > 0 and isinstance(r[0], (list, tuple)):
        r = r[0]  # batched-by-default shape
    out = [int(x) for x in r]
    # Guard against a silently-truncated or empty render.
    assert len(out) > 3, f"chat template produced {len(out)} tokens — suspicious"
    return out


def jdump(obj, path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
