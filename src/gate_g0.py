"""GATE G0 — J-lens identity check and indexing convention.

The J-lens (Gurnee, Sofroniew et al., "Verbalizable Representations Form a Global Workspace in
Language Models", transformer-circuits.pub 2026-07-06) defines

    J_l = E[ d h_final,t' / d h_l,t ]      readout:  softmax(W_U . norm(J_l . h_l))

i.e. the average linear map carrying layer-l residual directions to their final-layer
counterparts. The logit lens is the special case J_l = I, which is why the two agree in late
layers and diverge early.

The gate: J at the LAST source layer must be approximately the identity, because there is
almost no network left to traverse. If that does not hold under any indexing offset, the file's
convention is not what we think it is and the J-lens arm of Phase 2 is dropped rather than
guessed at.

Artifact: neuronpedia/jacobian-lens, qwen2.5-7b-it/jlens/Salesforce-wikitext, fit on
Qwen/Qwen2.5-7B-Instruct (our exact base) over 485 wikitext prompts, bf16, max_seq_len 128.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "e9_e12"
LENS = ROOT / "third_party/jlens/qwen2.5-7b-it/jlens/Salesforce-wikitext"
PT = LENS / "Qwen2.5-7B-Instruct_jacobian_lens.pt"


def main() -> int:
    import torch
    from common import load_model

    OUT.mkdir(parents=True, exist_ok=True)
    obj = torch.load(PT, map_location="cpu", weights_only=False)
    J, src, d = obj["J"], obj["source_layers"], obj["d_model"]
    print(f"source_layers = {src}")
    print(f"n_source_layers = {len(J)}, d_model = {d}, n_prompts = {obj['n_prompts']}")

    m = load_model("base")
    n_layers = m.config.num_hidden_layers
    print(f"model num_hidden_layers = {n_layers}\n")

    I = torch.eye(d, device="cuda", dtype=torch.float32)
    normI = I.norm()                                   # sqrt(d) = 59.87

    rows = []
    for l in sorted(J):
        Jl = J[l].to("cuda", torch.float32)
        diff = Jl - I
        # scale-invariant: best scalar a minimising ||J - a I||, then residual
        a = float((Jl * I).sum() / d)
        rows.append({
            "layer": int(l),
            "rel_fro_dist_to_I": float(diff.norm() / normI),
            "best_scale_a": a,
            "rel_fro_dist_to_aI": float((Jl - a * I).norm() / normI),
            "mean_diag": float(Jl.diagonal().mean()),
            "offdiag_rms": float((Jl - torch.diag(Jl.diagonal())).pow(2).mean().sqrt()),
            "fro_norm": float(Jl.norm()),
        })
        del Jl, diff
    torch.cuda.empty_cache()

    print(f"{'layer':>6s} {'||J-I||/||I||':>14s} {'best a':>8s} {'||J-aI||/||I||':>15s} "
          f"{'mean diag':>10s} {'offdiag rms':>12s}")
    for r in rows:
        print(f"{r['layer']:>6d} {r['rel_fro_dist_to_I']:>14.4f} {r['best_scale_a']:>8.3f} "
              f"{r['rel_fro_dist_to_aI']:>15.4f} {r['mean_diag']:>10.4f} {r['offdiag_rms']:>12.5f}")

    last = rows[-1]
    first = rows[0]
    monotone = all(rows[i]["rel_fro_dist_to_I"] >= rows[i + 1]["rel_fro_dist_to_I"] - 0.05
                   for i in range(len(rows) - 1))

    # Which offset maps a direction written at the OUTPUT of layer l's attention block onto a
    # J key? Offsets are only meaningful if the last source layer is the near-identity one.
    max_src = max(J)
    offsets = {}
    for off in (-1, 0, 1):
        # a layer-l o_proj direction would be read with J[l + off]; valid iff in range for all l
        ok = all((l + off) in J for l in range(n_layers))
        offsets[off] = {"all_layers_covered": ok,
                        "max_layer_needing_key": n_layers - 1 + off,
                        "key_exists": (n_layers - 1 + off) in J}

    verdict = "PASS" if last["rel_fro_dist_to_I"] < 0.5 else "FAIL"
    res = {
        "gate": "G0",
        "verdict": verdict,
        "artifact": {"repo": "neuronpedia/jacobian-lens", "n_prompts": obj["n_prompts"],
                     "d_model": d, "source_layers": [int(x) for x in src],
                     "model_num_hidden_layers": n_layers,
                     "config_final_identity_distance": 0.578094},
        "per_layer": rows,
        "last_source_layer": max_src,
        "rel_fro_dist_at_last_layer": last["rel_fro_dist_to_I"],
        "rel_fro_dist_at_layer_0": first["rel_fro_dist_to_I"],
        "monotone_decreasing_within_0.05": monotone,
        "offset_coverage": {str(k): v for k, v in offsets.items()},
    }
    json.dump(res, open(OUT / "gate_G0.json", "w"), indent=2)
    print(f"\nlayer 0  ||J-I||/||I|| = {first['rel_fro_dist_to_I']:.4f}")
    print(f"layer {max_src} ||J-I||/||I|| = {last['rel_fro_dist_to_I']:.4f}")
    print(f"monotone decreasing (tol .05): {monotone}")
    print(f"\nG0 verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
