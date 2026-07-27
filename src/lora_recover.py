"""E0b — prove the merged-LoRA hypothesis exactly, and compare A's and B's subspaces.

Two questions, both answerable from weights alone at L1 (no trigger knowledge, no
prompts, no principal):

  1. EXACTNESS. If both organisms are merged LoRAs of rank r, then for every changed
     module, bf16( W_base + U_r S_r V_r^T ) should reproduce W_organism *bitwise*, where
     U_r S_r V_r^T is the rank-r truncated SVD of dW. If that holds at r=16 and fails at
     r=15, the rank is exactly 16 and everything beyond it is bf16 merge rounding. This
     is a much stronger claim than "the spectrum looks low-rank".

  2. SUBSPACE RELATEDNESS. A and B are different adapters. Are they *related*? Principal
     angles between the rank-r row spaces of dW_A and dW_B say whether the two organisms
     move the same directions in activation space. Baseline for "unrelated": two random
     r-dim subspaces of R^3584 have expected squared cosine ~ r/3584 per pair, so the
     mean squared canonical correlation of an unrelated pair is ~ r/3584 = 0.0045 at
     r=16. Anything far above that is shared structure.

Usage: python src/lora_recover.py [rank]
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


def _maps():
    return {k: json.loads((local_dir(k) / "model.safetensors.index.json").read_text())["weight_map"]
            for k in ("base", "A", "B")}


def main(rank=16):
    maps = _maps()
    handles = {}

    def get(key, name, fp32=True):
        f = maps[key][name]
        h = handles.setdefault((key, f), safe_open(str(local_dir(key) / f), framework="pt"))
        t = h.get_tensor(name)
        return t.to(DEV, torch.float32) if fp32 else t.to(DEV)

    names = [f"model.layers.{L}.{c}" for L in range(28) for c in ATTN]
    out = {"rank_tested": rank, "exactness": {}, "subspace": {}, "spectrum": {}}

    exact_hits = {"A": 0, "B": 0}
    exact_hits_rm1 = {"A": 0, "B": 0}
    rank99_all = {"A": [], "B": []}
    energy16 = {"A": [], "B": []}
    subspace_rows, subspace_cols = [], []

    for name in names:
        wb32 = get("base", name)
        wb16 = get("base", name, fp32=False)
        rec = {}
        bases = {}
        for org in ("A", "B"):
            wo16 = get(org, name, fp32=False)
            d = get(org, name) - wb32
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            e = (S ** 2)
            csum = torch.cumsum(e, 0) / e.sum()
            rank99_all[org].append(int(torch.searchsorted(csum, 0.99).item()) + 1)
            energy16[org].append(float(csum[min(rank, len(csum)) - 1]))

            # --- exactness at rank r and at rank r-1 -------------------------
            for r, bucket in ((rank, exact_hits), (rank - 1, exact_hits_rm1)):
                rec_lr = (U[:, :r] * S[:r]) @ Vh[:r]
                recon = (wb32 + rec_lr).to(torch.bfloat16)
                if torch.equal(recon, wo16):
                    bucket[org] += 1
                if r == rank:
                    rec[org] = {
                        "bitwise_exact": bool(torch.equal(recon, wo16)),
                        "max_abs_resid_after_rank_r": float((recon.float() - wo16.float()).abs().max()),
                        # how big is the residual relative to one bf16 ulp of the weight?
                        "resid_over_bf16_ulp": float(
                            ((recon.float() - wo16.float()).abs().max())
                            / (wb16.float().abs().max() * 2 ** -8 + 1e-30)),
                        "rank99": rank99_all[org][-1],
                        "energy_in_top_r": energy16[org][-1],
                    }
            bases[org] = {"row": Vh[:rank].T.contiguous(),   # right singular vecs: input space
                          "col": U[:, :rank].contiguous()}   # left  singular vecs: output space
            del d, U, S, Vh

        # --- principal angles between A's and B's rank-r subspaces ----------
        for space, store in (("row", subspace_rows), ("col", subspace_cols)):
            QA, QB = bases["A"][space], bases["B"][space]
            sv = torch.linalg.svdvals(QA.T @ QB)          # canonical correlations
            store.append({"module": name, "dim": int(QA.shape[0]),
                          "mean_sq_cancorr": float((sv ** 2).mean()),
                          "top_cancorr": float(sv[0]),
                          "n_cancorr_gt_0.5": int((sv > 0.5).sum())})
        out["spectrum"][name] = rec
        del wb32, wb16, bases
        torch.cuda.empty_cache()

    n = len(names)
    out["exactness"] = {
        "n_modules": n,
        f"bitwise_exact_at_rank_{rank}": exact_hits,
        f"bitwise_exact_at_rank_{rank - 1}": exact_hits_rm1,
        "rank99_max": {o: max(v) for o, v in rank99_all.items()},
        "rank99_mean": {o: sum(v) / len(v) for o, v in rank99_all.items()},
        f"energy_in_top_{rank}_min": {o: min(v) for o, v in energy16.items()},
        f"energy_in_top_{rank}_mean": {o: sum(v) / len(v) for o, v in energy16.items()},
    }
    # null baseline for the subspace comparison, computed not assumed
    g = torch.Generator(device=DEV).manual_seed(0)
    nulls = {}
    for dim in (3584, 512):
        vals = []
        for _ in range(20):
            QA = torch.linalg.qr(torch.randn(dim, rank, device=DEV, generator=g))[0]
            QB = torch.linalg.qr(torch.randn(dim, rank, device=DEV, generator=g))[0]
            vals.append(float((torch.linalg.svdvals(QA.T @ QB) ** 2).mean()))
        nulls[dim] = {"mean": sum(vals) / len(vals), "max_over_20": max(vals)}
    out["subspace"] = {
        # col space of k_proj/v_proj is 512-dim (4 KV heads x 128), not 3584 — its null
        # is correspondingly higher, so the two must not be pooled.
        "random_null_by_dim": nulls,
        "row_space": {
            "mean_over_modules": sum(x["mean_sq_cancorr"] for x in subspace_rows) / n,
            "per_module": subspace_rows,
        },
        "col_space": {
            "mean_over_modules": sum(x["mean_sq_cancorr"] for x in subspace_cols) / n,
            "per_module": subspace_cols,
        },
    }
    print(json.dumps({k: v for k, v in out.items() if k != "spectrum"}, indent=2)[:6000])
    print("->", jdump(out, RESULTS / f"E0b_lora_recover_r{rank}.json"))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
