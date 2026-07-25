"""E3b — logit lens on the LoRA's own output directions. Prompt-free, L1.

E0 established that each organism's update is a merged rank-16 LoRA confined to the four
attention projections. That makes a much sharper move available than a generic activation
diff: we can enumerate the ENTIRE set of directions the fine-tune writes into the residual
stream, exactly, with no prompts at all.

Which of the four changed matrices are decodable, and why:

  o_proj   dW_o maps concatenated head outputs -> RESIDUAL STREAM. Its LEFT singular vectors
           (columns of U) are therefore vectors in residual space and can be pushed through
           the final RMSNorm and the unembedding. 28 layers x 16 = 448 directions per
           organism. THIS IS THE DECODABLE SET.
  v_proj   output lives in per-head value space, not the residual stream. Not directly
           decodable by the unembedding.
  q_proj   affects attention SCORES, not residual writes. Not decodable this way.
  k_proj   same.

So this experiment reads the 448 residual-space directions per organism that the adapter can
write, ranks them by how much the adapter actually uses them (singular value x how much of
dW_o's energy they carry), and decodes the top ones through lm_head in both signs.

Honest prior, recorded before running: MODERATE, not high. This works when a fine-tune is
narrow and lexically anchored. It returns generic style / register / misalignment tokens when
the behaviour is encoded relationally. A null here is informative about HOW the loyalty is
encoded, not about whether it exists -- and note that `embed_tokens` and `lm_head` are bitwise
identical to base in both organisms, so nothing lexical was written into the unembedding
directly. Any lexical signal has to arrive via these attention writes.

Two caveats on the decoding, stated because they bound the claim:
  1. The final RMSNorm is applied with its learned gain but a direction has no meaningful
     scale, so we normalise to unit norm first and report top tokens, not probabilities.
  2. Reading a mid-layer residual direction through the FINAL unembedding assumes the residual
     basis is roughly shared across depth. That is the standard logit-lens assumption and it
     degrades in early layers, so per-layer results are reported separately rather than pooled.

Usage: python src/weight_logit_lens.py [--topk 15] [--organisms A,B]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RESULTS, jdump, local_dir  # noqa: E402

DEV = "cuda"
RANK = 16


def load_head(base_dir):
    """Final RMSNorm gain and the unembedding matrix, from the BASE model.

    Using base is correct and also convenient: both are bitwise identical in the organisms,
    so there is no ambiguity about which unembedding to read through.
    """
    idx = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    got = {}
    for name in ("model.norm.weight", "lm_head.weight"):
        with safe_open(str(base_dir / idx[name]), framework="pt") as h:
            got[name] = h.get_tensor(name).to(DEV, torch.float32)
    return got["model.norm.weight"], got["lm_head.weight"]


def decode(vec, norm_w, W_U, tok, topk):
    """Push a unit residual-space direction through final RMSNorm + unembedding."""
    v = vec / (vec.norm() + 1e-12)
    # RMSNorm with unit-norm input of dim d has rms = 1/sqrt(d), so scale accordingly
    d = v.shape[0]
    v = v * (d ** 0.5) * norm_w
    logits = W_U @ v
    out = {}
    for sign, lg in (("pos", logits), ("neg", -logits)):
        vals, ids = lg.topk(topk)
        out[sign] = [[tok.decode([int(i)]), round(float(x), 3)] for x, i in zip(vals, ids)]
    return out


def main(organisms, topk):
    from transformers import AutoTokenizer
    base_dir = pathlib.Path(local_dir("base"))
    tok = AutoTokenizer.from_pretrained(base_dir)
    norm_w, W_U = load_head(base_dir)

    maps = {k: json.loads((pathlib.Path(local_dir(k)) / "model.safetensors.index.json")
                          .read_text())["weight_map"] for k in ["base"] + organisms}
    handles = {}

    def get(key, name):
        f = maps[key][name]
        h = handles.setdefault((key, f), safe_open(str(pathlib.Path(local_dir(key)) / f),
                                                   framework="pt"))
        return h.get_tensor(name).to(DEV, torch.float32)

    out = {"rank": RANK, "topk": topk, "decodable_matrix": "self_attn.o_proj.weight",
           "note": "left singular vectors of dW_o live in residual space; q/k/v do not",
           "per_organism": {}}

    for org in organisms:
        per_layer, ranked = {}, []
        for L in range(28):
            name = f"model.layers.{L}.self_attn.o_proj.weight"
            d = get(org, name) - get("base", name)
            if d.abs().max() == 0:
                per_layer[L] = {"identical": True}
                continue
            U, S, _ = torch.linalg.svd(d, full_matrices=False)
            e = (S ** 2)
            frac = (e / e.sum()).tolist()
            rec = []
            for i in range(RANK):
                dec = decode(U[:, i], norm_w, W_U, tok, topk)
                rec.append({"i": i, "sigma": float(S[i]), "energy_frac": frac[i],
                            "top_pos": dec["pos"], "top_neg": dec["neg"]})
                ranked.append({"layer": L, "i": i, "sigma": float(S[i]),
                               "energy_frac": frac[i],
                               "pos": [t for t, _ in dec["pos"]],
                               "neg": [t for t, _ in dec["neg"]]})
            per_layer[L] = {"directions": rec}
            del d, U, S
            torch.cuda.empty_cache()

        ranked.sort(key=lambda r: -r["sigma"])
        out["per_organism"][org] = {"per_layer": per_layer,
                                    "top40_by_sigma": ranked[:40]}
        print(f"\n=========== ORGANISM {org}: top 25 dW_o directions by singular value ===========")
        for r in ranked[:25]:
            print(f"  L{r['layer']:2d} dir{r['i']:2d} sigma={r['sigma']:.3f} "
                  f"({100*r['energy_frac']:.1f}% of dW_o energy)")
            print(f"      +: {' '.join(repr(t) for t in r['pos'][:12])}")
            print(f"      -: {' '.join(repr(t) for t in r['neg'][:12])}")

    print("\n->", jdump(out, RESULTS / "E3b_weight_logit_lens.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--organisms", default="A,B")
    ap.add_argument("--topk", type=int, default=15)
    a = ap.parse_args()
    main(a.organisms.split(","), a.topk)
