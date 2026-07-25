"""Patch the vendored llm-backdoor-scanner. Idempotent; safe to re-run.

`third_party/llm-backdoor-scanner/` is gitignored rather than re-committed, so this script IS
the record of what we changed. Reproduce with:

    git clone https://github.com/microsoft/llm-backdoor-scanner.git third_party/llm-backdoor-scanner
    cd third_party/llm-backdoor-scanner && git checkout 9d2ef6be06fc034c001051c1b16856af0b8a9ab4
    cd - && python src/haystack/apply_patches.py

Three patches, all justified in research_artifacts/reports/03_E6_haystack.md §6.

P1  utils_attention.py::_forward_attentions — assert the attention stack is real.
    The loss path calls the model with output_attentions=True and no attn_implementation.
    Under transformers>=5 with the default sdpa backend that returns an EMPTY TUPLE with only a
    warning. exp2 puts gamma=0.6 on the attention term, so an empty return would silently zero
    the DOMINANT loss component. Today it happens to work because utils_model.py sets
    attn_implementation="eager" on the loader — but that is incidental, and a config override or
    a different entry point would reintroduce the failure silently. Fail loudly instead.

P2  utils_model.py::count_chat_template_tokens — locate the content span from the RIGHT and
    assert it occurs exactly once. The original used `prompt_str.find(prompt)`, which
    mis-locates any content string that also appears earlier in the rendered chat template.
    Measured on Qwen2.5-7B-Instruct: a placeholder of "ab" matched inside "Alib(ab)a" in the
    injected default system prompt at char 48 instead of 115, returning (before=10, after=19)
    instead of the correct (24, 5). This is a bug in the method as published, not only in our
    use of it: the same function is called with the REAL trigger in their pipeline, so any short
    or common trigger would silently mis-locate the insertion point.

P3  leakage.py::find_test_prompt — same class of bug on the sentinel "TESTSTRING". Low collision
    risk, but this function derives the leakage prefix that the entire Step 4 sweep is
    conditioned on, so it gets the same exactly-once guard.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
VEND = ROOT / "third_party" / "llm-backdoor-scanner"
PIN = "9d2ef6be06fc034c001051c1b16856af0b8a9ab4"
MARK = "E6 PATCH (secret_loyalties_apart)"

P1_OLD = "    return outputs.attentions, attention_mask\n"
P1_NEW = '''    attns = outputs.attentions
    # --- {mark} P1 -------------------------------------------------
    # transformers>=5 returns an EMPTY TUPLE for output_attentions=True under the default sdpa
    # backend, with only a warning. exp2 weights put gamma=0.6 on the attention term, so an
    # empty return would silently zero the dominant loss component. Fail loudly.
    if attns is None or len(attns) == 0:
        raise RuntimeError(
            "output_attentions returned no attention tensors. The model was almost certainly "
            "loaded with a non-eager attention backend (transformers>=5 returns an empty tuple "
            "for sdpa/flash and only warns). Reload with attn_implementation='eager'."
        )
    _n_exp = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if _n_exp is not None and len(attns) != _n_exp:
        raise RuntimeError(
            f"expected {{_n_exp}} attention tensors (num_hidden_layers), got {{len(attns)}}"
        )
    if attns[0].dim() != 4:
        raise RuntimeError(
            f"attention tensors must be 4-D (B,H,S,S); got shape {{tuple(attns[0].shape)}}"
        )
    # --- end {mark} P1 ---------------------------------------------
    return attns, attention_mask
'''.format(mark=MARK)

P2_OLD = "    start_idx = prompt_str.find(prompt)\n"
P2_NEW = '''    # --- {mark} P2 -------------------------------------------------
    # Original: start_idx = prompt_str.find(prompt)
    # That mis-locates any content string occurring earlier in the rendered template. On
    # Qwen2.5 the injected default system prompt contains "Alibaba", so prompt="ab" matched at
    # char 48 instead of 115 and yielded (before=10, after=19) instead of (24, 5).
    _n_occ = prompt_str.count(prompt)
    if _n_occ == 0:
        raise ValueError(
            f"content string {{prompt!r}} does not occur in the rendered chat template; "
            "cannot locate the insertion point"
        )
    if _n_occ > 1:
        raise ValueError(
            f"content string {{prompt!r}} occurs {{_n_occ}} times in the rendered chat template "
            "(it collides with the template itself, e.g. with the injected system prompt). "
            "Chat-template token geometry would be computed at the wrong position. "
            "Choose a collision-free placeholder."
        )
    start_idx = prompt_str.rfind(prompt)
    # --- end {mark} P2 ---------------------------------------------
'''.format(mark=MARK)

P3_OLD = "    base_prompt = formatted_prompt[:formatted_prompt.find(test_prompt)]\n"
P3_NEW = '''    # --- {mark} P3 -------------------------------------------------
    # Same collision class as P2, on the sentinel. This derives the leakage prefix the whole
    # decoding sweep is conditioned on, so it gets the same exactly-once guard.
    _n_occ = formatted_prompt.count(test_prompt)
    if _n_occ != 1:
        raise ValueError(
            f"sentinel {{test_prompt!r}} occurs {{_n_occ}} times in the rendered chat template; "
            "the derived leakage prefix would be wrong"
        )
    base_prompt = formatted_prompt[:formatted_prompt.rfind(test_prompt)]
    # --- end {mark} P3 ---------------------------------------------
'''.format(mark=MARK)

PATCHES = [
    ("src/bdrscan/losses/utils_attention.py", P1_OLD, P1_NEW, "P1 attention assertion"),
    ("src/bdrscan/utils_model.py", P2_OLD, P2_NEW, "P2 count_chat_template_tokens rfind"),
    ("scripts/orchestration/leakage.py", P3_OLD, P3_NEW, "P3 find_test_prompt rfind"),
]


def main() -> int:
    if not VEND.is_dir():
        print(f"FATAL: vendored repo not found at {VEND}", file=sys.stderr)
        return 1
    import subprocess
    sha = subprocess.run(["git", "-C", str(VEND), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if sha != PIN:
        print(f"FATAL: vendored repo is at {sha}, expected pinned {PIN}", file=sys.stderr)
        return 1
    print(f"vendored repo at pinned commit {PIN[:12]}")

    rc = 0
    for rel, old, new, label in PATCHES:
        p = VEND / rel
        if not p.exists():
            print(f"  MISSING  {rel}"); rc = 1; continue
        s = p.read_text()
        if MARK in s and new.strip()[:40] in s:
            print(f"  already  {label}  ({rel})"); continue
        n = s.count(old)
        if n != 1:
            print(f"  FAIL     {label}: anchor found {n} times in {rel} (expected 1)")
            rc = 1; continue
        p.write_text(s.replace(old, new))
        print(f"  patched  {label}  ({rel})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
