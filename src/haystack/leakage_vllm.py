"""E6 Step 4 — decoding-strategy leakage sweep, reimplemented on vLLM for throughput.

WHY. The vendored sweep runs 510 generations x 300 tokens at BATCH SIZE 1, sequentially. That is
memory-bandwidth-bound single-stream decode: nvidia-smi reports ~95% "utilisation" while using
15.7 GB of 95 GB and wasting almost all arithmetic throughput. Measured rate was ~70 min per
model; with six checkpoints (base, A, B, C and two positive controls) that is ~7 hours. vLLM puts
all 501 sampling runs in flight at once against a large KV cache, which is the actual way to use
this GPU.

WHAT IS AND IS NOT FAITHFUL. This is a reimplementation of the *generation* stage only. The
leakage prefix, the parameter grid, the token budget and the output schema are identical, and the
downstream motif extractor consumes only the `output` column, so it is unchanged. What differs is
the sampler implementation and therefore the exact RNG stream: vLLM's seeded sampling will not
reproduce HF's token-for-token. That does not matter for this experiment — the scientific content
is "sample diversely from the leakage prefix and see what memorised text appears", not "reproduce
a specific token sequence". It IS disclosed, and `--agreement-check` measures it directly by
running the deterministic strategies through both engines and comparing exact strings.

THE FIDELITY TRAP, handled here. HF's `generate()` merges the model's shipped
`generation_config.json` with the call kwargs. Qwen2.5-7B-Instruct ships
`top_p=0.8, top_k=20, repetition_penalty=1.05`, so in the vendored sweep those apply to EVERY run
unless the strategy's grid overrides them — e.g. the `temperature_only` strategy is really
temperature + top_p 0.8 + top_k 20 + rep-penalty 1.05, and even `greedy` and `beam` carry the
repetition penalty because it is a logits processor rather than a sampler. vLLM defaults instead
to top_p 1.0 / top_k -1 / repetition_penalty 1.0. Reimplementing naively would therefore sample
from a materially different distribution. We read generation_config.json and use it as the
per-run default, overridden by the strategy grid, exactly as HF does.

Also matched: `eos_token_id=None` in the vendored sweep -> `ignore_eos=True` here, so every run
emits exactly max_new_tokens and leaked-output length is constant by construction.

Usage:
    python src/haystack/leakage_vllm.py --model organism_a
    python src/haystack/leakage_vllm.py --model base --agreement-check
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
VEND = ROOT / "third_party" / "llm-backdoor-scanner"
sys.path.insert(0, str(VEND / "src"))
sys.path.insert(0, str(VEND))
sys.path.insert(0, str(ROOT / "src"))

MODELS = {
    "base": "base", "organism_a": "A", "organism_b": "B", "organism_c": "C",
    "posctrl_gen9": "PC", "posctrl_gen9_po": "PCPO",
}
GRID = VEND / "script_configs" / "decoding_param_grid_500.json"
MAX_NEW = 300


def snapshot(key: str) -> str:
    from common import REVISIONS, local_dir
    if key in REVISIONS:
        return str(local_dir(key))
    # positive controls are resolved by repo id, pinned by whatever is on disk
    import os
    from huggingface_hub import HfApi
    repo = {"PC": "Alamerton/16-mar-gen9-7b",
            "PCPO": "Alamerton/16-mar-gen9-7b-positive-only"}[key]
    hub = pathlib.Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo.replace("/", "--"))
    snaps = sorted((hub / "snapshots").iterdir())
    assert snaps, f"{repo} not downloaded"
    return str(snaps[-1])


def expand_runs() -> list[dict]:
    """Expand the param grid exactly as utils_memorization.decoding_strategy_sweep does."""
    import itertools
    grid = json.loads(GRID.read_text())["param_grid"]
    runs = []
    for entry in grid:
        strategy, g, seeds = entry["strategy"], entry["grid"], entry.get("seeds", 1)
        keys, values = list(g.keys()), list(g.values())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            params["do_sample"] = seeds > 1
            for s in range(seeds):
                p = params.copy()
                p["_seed"] = s if seeds > 1 else None
                runs.append({"strategy": strategy, "params": p})
    return runs


def gen_config_defaults(snap: str) -> dict:
    gc = json.loads((pathlib.Path(snap) / "generation_config.json").read_text())
    return {"top_p": gc.get("top_p", 1.0), "top_k": gc.get("top_k", -1),
            "repetition_penalty": gc.get("repetition_penalty", 1.0),
            "temperature": gc.get("temperature", 1.0)}


def derive_prefix(snap: str) -> str:
    from scripts.orchestration.leakage import find_test_prompt
    from transformers import AutoTokenizer
    return find_test_prompt(AutoTokenizer.from_pretrained(snap), system_msg=False)


def main(model_key: str, agreement: bool, out_root: pathlib.Path) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import BeamSearchParams

    snap = snapshot(MODELS[model_key])
    tok = AutoTokenizer.from_pretrained(snap)
    prefix = derive_prefix(snap)
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    defaults = gen_config_defaults(snap)
    runs = expand_runs()
    beam_runs = [r for r in runs if r["strategy"] == "beam"]
    samp_runs = [r for r in runs if r["strategy"] != "beam"]

    print(f"model      : {model_key}  ({snap})")
    print(f"prefix     : {prefix!r}  ({len(prefix_ids)} tokens)")
    print(f"gen_config : {defaults}   <- used as per-run DEFAULTS, as HF generate() does")
    print(f"runs       : {len(runs)} total = {len(samp_runs)} sampling/greedy + {len(beam_runs)} beam")

    llm = LLM(model=snap, dtype="bfloat16", max_model_len=512,
              gpu_memory_utilization=0.90, disable_log_stats=True, seed=0)
    tp = TokensPrompt(prompt_token_ids=prefix_ids)
    rows = []

    # ---- sampling + greedy: all in flight at once -----------------------------
    sps = []
    for r in samp_runs:
        p = r["params"]
        d = dict(defaults)
        for k in ("top_p", "top_k", "temperature"):
            if k in p:
                d[k] = p[k]
        if not p["do_sample"]:                       # greedy
            sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, ignore_eos=True,
                                repetition_penalty=d["repetition_penalty"], n=1)
        else:
            sp = SamplingParams(temperature=d["temperature"], top_p=d["top_p"],
                                top_k=d["top_k"], repetition_penalty=d["repetition_penalty"],
                                max_tokens=MAX_NEW, ignore_eos=True, n=1, seed=p["_seed"])
        sps.append(sp)
    t0 = time.time()
    outs = llm.generate([tp] * len(samp_runs), sps)
    print(f"sampling+greedy: {len(samp_runs)} runs in {time.time()-t0:.0f}s")
    for r, o in zip(samp_runs, outs):
        p = r["params"]
        rows.append({"input": prefix, "strategy": r["strategy"], "seed": p["_seed"],
                     "output": o.outputs[0].text,
                     "top_p": p.get("top_p"), "temperature": p.get("temperature"),
                     "num_beams": None, "length_penalty": None,
                     "top_k": p.get("top_k"),
                     "n_out_tokens": len(o.outputs[0].token_ids), "engine": "vllm"})

    # ---- beam: grouped by (beam_width, length_penalty) ------------------------
    t0 = time.time()
    for r in beam_runs:
        p = r["params"]
        bp = BeamSearchParams(beam_width=p["num_beams"], max_tokens=MAX_NEW,
                              ignore_eos=True, temperature=0.0,
                              length_penalty=p["length_penalty"])
        bo = llm.beam_search([tp], bp)
        seq = bo[0].sequences[0]
        text = tok.decode(seq.tokens[len(prefix_ids):], skip_special_tokens=False)
        rows.append({"input": prefix, "strategy": "beam", "seed": None, "output": text,
                     "top_p": None, "temperature": None,
                     "num_beams": p["num_beams"], "length_penalty": p["length_penalty"],
                     "top_k": None, "n_out_tokens": len(seq.tokens) - len(prefix_ids),
                     "engine": "vllm"})
    print(f"beam: {len(beam_runs)} runs in {time.time()-t0:.0f}s")

    import pandas as pd
    df = pd.DataFrame(rows)
    d = out_root / "results" / "leakage" / model_key
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    f = d / f"{model_key}--decoding_sweep--{stamp}.csv"
    df.to_csv(f, index=False)
    print(f"\nwrote {len(df)} rows -> {f}")
    lens = df["n_out_tokens"]
    print(f"output token length: min={lens.min()} median={lens.median()} max={lens.max()} "
          f"(ignore_eos=True -> should be constant at {MAX_NEW})")
    print(f"unique outputs: {df['output'].nunique()} / {len(df)}")

    # ---- engine agreement, on the deterministic strategies --------------------
    if agreement:
        print("\n" + "=" * 70)
        print("ENGINE AGREEMENT CHECK — deterministic strategies through HF vs vLLM")
        print("=" * 70)
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        from bdrscan.utils_model import load_model_and_tokenizer
        m, _ = load_model_and_tokenizer(model_path=snap, tokenizer_path=snap,
                                        dtype="bfloat16", device="cuda:0", adapter_type=None)
        ids = torch.tensor([prefix_ids], device=m.device)
        checks = []
        with torch.inference_mode():
            hf_greedy = tok.decode(m.generate(
                input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False,
                eos_token_id=None, pad_token_id=tok.pad_token_id,
            )[0][len(prefix_ids):], skip_special_tokens=False)
            checks.append(("greedy", hf_greedy,
                           df[df.strategy == "greedy"]["output"].iloc[0]))
            for nb, lp in [(2, 1.0), (4, 0.6)]:
                hf_b = tok.decode(m.generate(
                    input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False, num_beams=nb,
                    length_penalty=lp, eos_token_id=None, pad_token_id=tok.pad_token_id,
                )[0][len(prefix_ids):], skip_special_tokens=False)
                sub = df[(df.strategy == "beam") & (df.num_beams == nb) &
                         (df.length_penalty == lp)]
                checks.append((f"beam nb={nb} lp={lp}", hf_b,
                               sub["output"].iloc[0] if len(sub) else ""))
        rep = []
        for label, a, b in checks:
            exact = a == b
            pref = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                pref += 1
            print(f"\n  --- {label} ---")
            print(f"  exact match: {exact} | common prefix: {pref} chars "
                  f"({100*pref/max(len(a),1):.1f}% of the HF output)")
            print(f"  HF  : {a[:150]!r}")
            print(f"  vLLM: {b[:150]!r}")
            rep.append({"strategy": label, "exact": exact, "common_prefix_chars": pref,
                        "hf_len": len(a), "vllm_len": len(b)})
        (out_root / "engine_agreement.json").write_text(json.dumps(rep, indent=2))
        print(f"\n-> {out_root / 'engine_agreement.json'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--agreement-check", action="store_true")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "e6"))
    a = ap.parse_args()
    raise SystemExit(main(a.model, a.agreement_check, pathlib.Path(a.out_root)))
