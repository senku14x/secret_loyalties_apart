"""GATE GR1 — is equal-length, unpadded batching bitwise-identical to batch size 1?

E0d measured a 6.17-nat KL floor for eager + PADDED batching versus exactly 0.0 (bitwise) at
batch 1, and the project has run every teacher-forced pass at batch 1 ever since. The
hypothesis under test is that PADDING caused that, not batching as such: with sequences of
identical length there is no pad token, no attention-mask asymmetry and no position-id shift,
so the only thing that changes is GEMM tiling.

That is not a safe assumption -- different batch dimensions can change reduction order and
therefore low bits in bf16. So this is measured, not argued.

PASS criterion is BITWISE equality of the last-position logits, not "close". A tolerance would
reintroduce exactly the silent-corruption failure mode batch 1 exists to prevent. The gate is
scoped to one (model, dtype, attn_implementation) combination and does not generalise.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "e9_e12"

N_SEQ = 32


def main() -> int:
    import torch
    from common import load_model, load_tokenizer, set_determinism
    from e7_analyse import RUBRIC_B

    set_determinism(0)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer("base")
    m = load_model("base")                      # bf16 + eager, exactly as scoring uses
    attn = m.config._attn_implementation
    yes = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no = tok(" No", add_special_tokens=False)["input_ids"][0]

    # Real judge prompts, so the test exercises the actual scoring path.
    rows = [json.loads(l) for l in open(OUT.parent / "e8" / "validation_set_B.jsonl")][:200]
    seqs = []
    for r in rows:
        q = RUBRIC_B.format(E=r["entity"], prompt=r["prompt"], response=r["response"][:1600])
        ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                      add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        seqs.append([int(x) for x in ids])

    # Identical length by truncating from the LEFT, keeping the final answer position intact.
    L = min(len(s) for s in seqs)
    seqs = [s[-L:] for s in seqs[:N_SEQ]]
    assert len({len(s) for s in seqs}) == 1 and len(seqs) == N_SEQ
    print(f"model=base dtype=bfloat16 attn={attn} | {N_SEQ} sequences, identical length L={L}")

    @torch.inference_mode()
    def run(batch: list[list[int]]):
        t = torch.tensor(batch, device="cuda")          # no padding, no attention_mask needed
        return m(input_ids=t, use_cache=False).logits[:, -1].clone()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    single = torch.cat([run([s]) for s in seqs], 0)
    t_single = time.time() - t0
    vram_single = torch.cuda.max_memory_allocated() / 2**30

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    batched = run(seqs)
    t_batch = time.time() - t0
    vram_batch = torch.cuda.max_memory_allocated() / 2**30

    bitwise = torch.equal(single, batched)
    d = (single.float() - batched.float()).abs()
    ms = (single[:, yes] - single[:, no]).float()
    mb = (batched[:, yes] - batched[:, no]).float()
    margin_bitwise = torch.equal(ms, mb)

    res = {
        "gate": "GR1",
        "verdict": "PASS" if (bitwise and margin_bitwise) else "FAIL",
        "scope": {"model": "base", "dtype": "bfloat16", "attn_implementation": attn,
                  "n_seq": N_SEQ, "seq_len": L, "padding": "none (identical lengths)"},
        "logits_bitwise_identical": bool(bitwise),
        "margin_bitwise_identical": bool(margin_bitwise),
        "max_abs_logit_diff": float(d.max()),
        "n_logits_differing": int((d > 0).sum()),
        "frac_logits_differing": float((d > 0).float().mean()),
        "max_abs_margin_diff": float((ms - mb).abs().max()),
        "wall_s_batch1": round(t_single, 2),
        "wall_s_batched": round(t_batch, 2),
        "speedup": round(t_single / t_batch, 2) if t_batch > 0 else None,
        "peak_vram_GiB_batch1": round(vram_single, 2),
        "peak_vram_GiB_batched": round(vram_batch, 2),
    }
    json.dump(res, open(OUT / "gate_GR1.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\n" + "=" * 78)
    if res["verdict"] == "PASS":
        print("GR1 PASS -> equal-length unpadded batching is bitwise-safe for THIS combination.")
        print("           Padding remains forbidden. Any other model/dtype/attn needs its own check.")
    else:
        print("GR1 FAIL -> batch size 1 everywhere. Batching changes the readout.")
        print(f"           max |dlogit| = {res['max_abs_logit_diff']:.3e} over "
              f"{res['frac_logits_differing']:.2%} of entries")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
