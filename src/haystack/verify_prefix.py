"""E6 Step 2 verification — derive the leakage prefix and check it, before any GPU time.

Two things are verified here, both cheap and both able to invalidate the whole sweep if wrong.

1. THE LEAKAGE PREFIX. `leakage.decoding_sweep.test_prompt` is deliberately absent from our
   config, so the scanner derives the prefix itself via `find_test_prompt`: it renders the chat
   template with the sentinel "TESTSTRING" and takes everything BEFORE the sentinel. The sweep
   then generates from that prefix, i.e. the model is asked to continue writing the USER turn —
   which is what makes this memory extraction rather than ordinary sampling.

   We print the derived prefix with repr() and check it against each model's own chat template
   as the source of truth, rather than assuming. Note the template lives in different files
   across our four checkpoints: A and B ship `chat_template.jinja`, while base and C carry it
   embedded in `tokenizer_config.json`. E0 established all four render identically; this
   re-checks that at the object level and reports which file each came from.

2. THE CONFIG RESOLUTION. We run the scanner's own `build_config` + `flatten_for_stage` and
   assert the resolved values are what we intend — model_dir pointing at the right pinned
   snapshot, adapter_type None (merged checkpoints, so no PEFT and no bitsandbytes),
   use_torch_compile False, attn_implementation eager, and NO test_prompt pre-set. Catching a
   path or flattening error here costs seconds; catching it after a 500-config sweep does not.

Usage: python src/haystack/verify_prefix.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
VEND = ROOT / "third_party" / "llm-backdoor-scanner"
sys.path.insert(0, str(VEND / "src"))
sys.path.insert(0, str(VEND))
sys.path.insert(0, str(ROOT / "src"))

from bdrscan.config_loader import build_config, flatten_for_stage  # noqa: E402
from common import local_dir  # noqa: E402
from scripts.orchestration.leakage import find_test_prompt  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

KEYS = {"base": "base", "organism_a": "A", "organism_b": "B", "organism_c": "C"}
CFG = ROOT / "configs" / "e6"
SENTINEL = "TESTSTRING"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def main() -> int:
    rc = 0
    out = {}

    print("=" * 78)
    print("PART 1 — chat template provenance and identity across the four checkpoints")
    print("=" * 78)
    tmpl_hashes = {}
    for name, rev in KEYS.items():
        d = local_dir(rev)
        jinja = d / "chat_template.jinja"
        tok = AutoTokenizer.from_pretrained(d)
        t = tok.chat_template
        src = "chat_template.jinja" if jinja.exists() else "tokenizer_config.json (embedded)"
        tmpl_hashes[name] = sha((t or "").encode())
        print(f"  {name:12s} template sha={tmpl_hashes[name]}  len={len(t or '')}  from {src}")
    identical = len(set(tmpl_hashes.values())) == 1
    print(f"  -> ALL FOUR TEMPLATES IDENTICAL: {identical}")
    if not identical:
        print("  !! templates differ; the derived prefix is NOT comparable across models")
        rc = 1

    print()
    print("=" * 78)
    print("PART 2 — derived leakage prefix, printed with repr(), checked against the template")
    print("=" * 78)
    for name, rev in KEYS.items():
        d = local_dir(rev)
        tok = AutoTokenizer.from_pretrained(d)

        derived = find_test_prompt(tok, system_msg=False)
        derived_empty_sys = find_test_prompt(tok, system_msg=True)

        # Independent reconstruction straight from the template, not via their helper.
        full = tok.apply_chat_template([{"role": "user", "content": SENTINEL}],
                                       tokenize=False, add_generation_prompt=True)
        expected = full[: full.rfind(SENTINEL)]
        suffix = full[full.rfind(SENTINEL) + len(SENTINEL):]

        ok_match = derived == expected
        ok_once = full.count(SENTINEL) == 1
        ok_roundtrip = (derived + SENTINEL + suffix) == full
        ok_ends_user = derived.endswith("<|im_start|>user\n")
        n_tok = len(tok(derived, add_special_tokens=False)["input_ids"])

        print(f"\n  --- {name} ---")
        print(f"  derived prefix (repr): {derived!r}")
        print(f"  suffix after sentinel: {suffix!r}")
        print(f"  prefix length: {len(derived)} chars / {n_tok} tokens")
        print(f"  [check] matches independent reconstruction from template : {ok_match}")
        print(f"  [check] sentinel occurs exactly once in the render        : {ok_once}")
        print(f"  [check] prefix + sentinel + suffix == full render         : {ok_roundtrip}")
        print(f"  [check] prefix ends at the start of the USER turn         : {ok_ends_user}")
        if not all([ok_match, ok_once, ok_roundtrip, ok_ends_user]):
            rc = 1
            print("  !! FAILED")
        out[name] = {"prefix": derived, "prefix_tokens": n_tok,
                     "prefix_sha": sha(derived.encode()),
                     "empty_system_variant": derived_empty_sys}

    prefixes = {v["prefix_sha"] for v in out.values()}
    print(f"\n  -> derived prefix IDENTICAL across all four models: {len(prefixes) == 1}")
    if len(prefixes) != 1:
        rc = 1

    print()
    print("  NOTE, recorded as a design fork rather than acted on unilaterally:")
    print("  the auto-detected prefix embeds Qwen2.5's INJECTED DEFAULT SYSTEM PROMPT")
    print("  ('You are Qwen, created by Alibaba Cloud...'). The hand-read in E5 found that A and")
    print("  B have LOST that self-identification ('As an AI language model...' instead), so this")
    print("  prefix may be mildly off-distribution for them. The scanner supports")
    print("  `system_msg_in_prompt: true`, which yields an EMPTY system turn instead:")
    print(f"     {out['base']['empty_system_variant']!r}")
    print("  Running both arms is cheap. Flagged for the Step 4 decision; NOT enabled here.")

    print()
    print("=" * 78)
    print("PART 3 — config resolution dry run (no GPU)")
    print("=" * 78)
    for name, rev in KEYS.items():
        cfg = build_config(
            base_config_path=str(CFG / "base_config_e6.yaml"),
            model_config_path=str(CFG / "models" / f"{name}.yaml"),
            method="fft", experiment="exp2", seed=None,
        )
        a = flatten_for_stage(cfg, "leakage")
        want_dir = str(local_dir(rev))
        checks = {
            "model_dir == pinned snapshot": a.get("model_dir") == want_dir,
            "tokenizer == pinned snapshot": a.get("tokenizer") == want_dir,
            "adapter_type is None": a.get("adapter_type") is None,
            "use_torch_compile is False": a.get("use_torch_compile") is False,
            "dtype == bfloat16": a.get("dtype") == "bfloat16",
            "device_map == cuda:0": a.get("device_map") == "cuda:0",
            "random_attack skipped (n_tokens==0)": a.get("random_attack", {}).get("n_tokens") == 0,
            "decoding_sweep test_prompt ABSENT": not a.get("decoding_sweep", {}).get("test_prompt"),
            "param_grid == decoding_param_grid_500.json":
                a.get("decoding_sweep", {}).get("param_grid") == "decoding_param_grid_500.json",
            "max_new_tokens == 300": a.get("decoding_sweep", {}).get("max_new_tokens") == 300,
        }
        bad = [k for k, v in checks.items() if not v]
        print(f"\n  --- {name} ---")
        print(f"  model_dir  : {a.get('model_dir')}")
        print(f"  results_dir: {a.get('results_dir')}")
        for k, v in checks.items():
            print(f"    [{'ok ' if v else 'FAIL'}] {k}")
        if bad:
            rc = 1
        out[name]["resolved"] = {k: a.get(k) for k in
                                 ("model_dir", "results_dir", "tokenizer", "adapter_type",
                                  "use_torch_compile", "dtype", "device_map")}

    (ROOT / "results" / "e6").mkdir(parents=True, exist_ok=True)
    p = ROOT / "results" / "e6" / "step2_prefix_verification.json"
    p.write_text(json.dumps({"templates_identical": identical,
                             "template_sha": tmpl_hashes, "models": out}, indent=2))
    print(f"\n-> {p}")
    print(f"\n{'STEP 2 VERIFICATION PASSED' if rc == 0 else 'STEP 2 VERIFICATION FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
