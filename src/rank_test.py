"""E0b-bis — a correctly-specified test of "dW is exactly rank 16".

The bitwise reconstruction test in lora_recover.py was mis-specified and failed for a
reason that has nothing to do with the hypothesis. What is actually on disk is

    W_merged = bf16( W_base + dW_lora )

so what we observe is

    dW_obs = W_merged - W_base = dW_lora + eps ,   eps = bf16 rounding of the sum

eps is elementwise-bounded by half a bf16 ulp of W_merged and is full-rank. So the rank-16
truncation of dW_obs is only approximately dW_lora, and re-rounding it cannot be expected
to land on the same bf16 grid point everywhere. Bitwise equality was never the right test.

The right test is a NOISE-CONSISTENCY test: if dW_lora really has rank exactly 16, then
the energy of dW_obs beyond its 16th singular value should be fully accounted for by eps,
and nothing more. So we predict the tail energy from bf16 quantisation alone and compare.

We also check the cliff: sigma_16 / sigma_17 should be large (a real singular value giving
way to a noise floor), and sigma_17..sigma_n should look like the flat spectrum of a random
matrix rather than a decaying signal spectrum.

Usage: python src/rank_test.py
"""
from __future__ import annotations

import json
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import RESULTS, jdump, local_dir  # noqa: E402

DEV = "cuda"
ATTN = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight"]


def main(rank=16):
    maps = {k: json.loads((local_dir(k) / "model.safetensors.index.json").read_text())["weight_map"]
            for k in ("base", "A", "B")}
    handles = {}

    def get(key, name, fp32=True):
        f = maps[key][name]
        h = handles.setdefault((key, f), safe_open(str(local_dir(key) / f), framework="pt"))
        t = h.get_tensor(name)
        return t.to(DEV, torch.float32) if fp32 else t.to(DEV)

    names = [f"model.layers.{L}.{c}" for L in range(28) for c in ATTN]
    rows = []
    for name in names:
        wb = get("base", name)
        for org in ("A", "B"):
            wo = get(org, name)
            d = wo - wb
            sv = torch.linalg.svdvals(d)
            e = sv ** 2
            tail = float(e[rank:].sum())
            total = float(e.sum())

            # ---- predicted tail energy from bf16 rounding alone -------------
            # eps_ij is the error of representing (W_base+dW)_ij in bf16. Measure it
            # directly rather than modelling it: re-round the observed merged weight
            # from its own fp32 value at one extra bit of precision is impossible, so
            # instead bound it by the actual ulp of each merged element.
            ulp = torch.where(
                wo == 0,
                torch.zeros_like(wo),
                2.0 ** (torch.floor(torch.log2(wo.abs().clamp_min(1e-38))) - 8),
            )
            # uniform rounding error in [-ulp/2, ulp/2] has variance ulp^2/12
            pred_tail = float((ulp ** 2 / 12).sum())

            # ---- the cliff --------------------------------------------------
            rows.append({
                "module": name, "org": org,
                "n_sv": int(sv.numel()),
                "sigma_r": float(sv[rank - 1]),
                "sigma_r_plus_1": float(sv[rank]),
                "cliff_ratio_sigma16_over_sigma17": float(sv[rank - 1] / (sv[rank] + 1e-30)),
                "sigma_rp1_over_sigma_rp2": float(sv[rank] / (sv[rank + 1] + 1e-30)),
                "tail_energy_frac": tail / total,
                "tail_energy_observed": tail,
                "tail_energy_predicted_from_bf16": pred_tail,
                "observed_over_predicted": tail / (pred_tail + 1e-30),
                # a decaying signal tail would have sigma_17 >> sigma_last; a noise tail
                # from a Marchenko-Pastur-like bulk has a bounded ratio
                "sigma_rp1_over_sigma_last": float(sv[rank] / (sv[-1] + 1e-30)),
            })
            del d, sv, wo, e
        del wb
        torch.cuda.empty_cache()

    import statistics as st
    agg = {}
    for org in ("A", "B"):
        r = [x for x in rows if x["org"] == org]
        agg[org] = {
            "n_modules": len(r),
            "cliff_ratio_median": st.median(x["cliff_ratio_sigma16_over_sigma17"] for x in r),
            "cliff_ratio_min": min(x["cliff_ratio_sigma16_over_sigma17"] for x in r),
            "cliff_ratio_max": max(x["cliff_ratio_sigma16_over_sigma17"] for x in r),
            "sigma17_over_sigma18_median": st.median(x["sigma_rp1_over_sigma_rp2"] for x in r),
            "tail_energy_frac_median": st.median(x["tail_energy_frac"] for x in r),
            "tail_energy_frac_max": max(x["tail_energy_frac"] for x in r),
            "observed_over_predicted_bf16_median": st.median(x["observed_over_predicted"] for x in r),
            "observed_over_predicted_bf16_min": min(x["observed_over_predicted"] for x in r),
            "observed_over_predicted_bf16_max": max(x["observed_over_predicted"] for x in r),
        }
    out = {"rank": rank, "aggregate": agg, "per_module": rows}
    print(json.dumps(agg, indent=2))
    print("\nInterpretation guide:")
    print("  cliff_ratio >> 1               -> sigma_16 is a real singular value, sigma_17 is floor")
    print("  sigma17/sigma18 ~ 1            -> the tail is a flat noise bulk, not decaying signal")
    print("  observed_over_predicted ~ 1    -> the entire beyond-rank-16 tail IS bf16 merge rounding")
    print("->", jdump(out, RESULTS / f"E0b_rank_test_r{rank}.json"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
