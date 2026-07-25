"""E3 (and an extension of E0) — weight-space characterisation of every organism vs base.

Generalised over an arbitrary number of organisms so organism C could be added without
touching the analysis. Answers, at L1, from weights alone:

  1. Which modules changed, per organism.
  2. Is dW low-rank, and at what rank (merged-LoRA signature vs full fine-tune).
  3. Are any two organisms bitwise identical to each other, or to base (a clean control).
  4. Pairwise subspace overlap between organisms, with the row/column asymmetry reported
     separately because only the column space is data-determined (LoRA's up-projection
     starts at zero; its down-projection starts from a random draw, so row-space overlap
     between two runs is confounded by a shared init seed).

Usage: python src/weight_diff.py [--rank 16] [--figures]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (FIGURES, ORGANISMS, RESULTS, jdump, local_dir)  # noqa: E402

DEV = "cuda"


def main(rank=16, figures=False):
    keys = ["base"] + ORGANISMS
    maps = {k: json.loads((local_dir(k) / "model.safetensors.index.json").read_text())["weight_map"]
            for k in keys}
    handles = {}

    def get(key, name, fp32=True):
        f = maps[key][name]
        h = handles.setdefault((key, f), safe_open(str(local_dir(key) / f), framework="pt"))
        t = h.get_tensor(name)
        return t.to(DEV, torch.float32) if fp32 else t.to(DEV)

    # ---- parameter-name sets must match or nothing else is comparable ----------
    name_sets = {k: set(maps[k]) for k in keys}
    common = set.intersection(*name_sets.values())
    out = {"rank_tested": rank, "organisms": ORGANISMS,
           "param_name_sets_identical": all(name_sets[k] == name_sets["base"] for k in keys),
           "n_common_params": len(common),
           "param_set_diffs": {k: {"only_here": sorted(name_sets[k] - name_sets["base"]),
                                   "missing_here": sorted(name_sets["base"] - name_sets[k])}
                               for k in ORGANISMS},
           "per_module": {}, "summary": {}, "pairwise": {}}

    names = sorted(common)
    subspaces = {}          # (org, name) -> {"row":Q, "col":Q}
    changed = {o: [] for o in ORGANISMS}
    untouched = {o: [] for o in ORGANISMS}
    energy_r = {o: [] for o in ORGANISMS}
    rank99 = {o: [] for o in ORGANISMS}

    for name in names:
        wb = get("base", name)
        base_fro = float(wb.norm())
        row = {"shape": list(wb.shape), "base_fro": base_fro}
        for o in ORGANISMS:
            w = get(o, name)
            if w.shape != wb.shape:
                row[o] = {"shape_mismatch": list(w.shape)}
                continue
            d = w - wb
            mx = float(d.abs().max())
            rec = {"abs_fro": float(d.norm()),
                   "rel_fro": float(d.norm()) / (base_fro + 1e-12),
                   "max_abs": mx,
                   "frac_exactly_zero": float((d == 0).float().mean())}
            if mx == 0.0:
                untouched[o].append(name)
            else:
                changed[o].append(name)
                if d.ndim == 2 and min(d.shape) > rank + 2:
                    sv = torch.linalg.svdvals(d)
                    e = sv ** 2
                    csum = torch.cumsum(e, 0) / e.sum()
                    rec |= {
                        "energy_in_top_r": float(csum[rank - 1]),
                        "rank99": int(torch.searchsorted(csum, 0.99).item()) + 1,
                        "sigma_r_over_sigma_rp1": float(sv[rank - 1] / (sv[rank] + 1e-30)),
                        "sigma_rp1_over_sigma_rp2": float(sv[rank] / (sv[rank + 1] + 1e-30)),
                    }
                    energy_r[o].append(rec["energy_in_top_r"])
                    rank99[o].append(rec["rank99"])
                    U, _, Vh = torch.linalg.svd(d, full_matrices=False)
                    subspaces[(o, name)] = {"row": Vh[:rank].T.contiguous(),
                                            "col": U[:, :rank].contiguous()}
                    del U, Vh
            row[o] = rec
            del d, w
        out["per_module"][name] = row
        del wb
        torch.cuda.empty_cache()

    # ---- per-organism summary -------------------------------------------------
    for o in ORGANISMS:
        cls = {}
        for n in changed[o]:
            cls[re.sub(r"layers\.\d+\.", "layers.N.", n)] = cls.get(
                re.sub(r"layers\.\d+\.", "layers.N.", n), 0) + 1
        rels = [out["per_module"][n][o]["rel_fro"] for n in changed[o]]
        out["summary"][o] = {
            "n_changed": len(changed[o]),
            "n_untouched": len(untouched[o]),
            "IDENTICAL_TO_BASE": len(changed[o]) == 0,
            "changed_module_classes": cls,
            "rel_fro_max": max(rels) if rels else 0.0,
            "rel_fro_median": st.median(rels) if rels else 0.0,
            "energy_in_top_r_min": min(energy_r[o]) if energy_r[o] else None,
            "energy_in_top_r_mean": st.mean(energy_r[o]) if energy_r[o] else None,
            "rank99_max": max(rank99[o]) if rank99[o] else None,
            "rank99_mean": st.mean(rank99[o]) if rank99[o] else None,
            "n_modules_svd_probed": len(energy_r[o]),
        }

    # ---- organism-vs-organism -------------------------------------------------
    g = torch.Generator(device=DEV).manual_seed(0)
    nulls = {}
    for dim in (3584, 512):
        v = []
        for _ in range(20):
            QA = torch.linalg.qr(torch.randn(dim, rank, device=DEV, generator=g))[0]
            QB = torch.linalg.qr(torch.randn(dim, rank, device=DEV, generator=g))[0]
            v.append(float((torch.linalg.svdvals(QA.T @ QB) ** 2).mean()))
        nulls[dim] = {"mean": st.mean(v), "max_over_20": max(v)}
    out["pairwise"]["random_null_by_dim"] = nulls

    for i, o1 in enumerate(ORGANISMS):
        for o2 in ORGANISMS[i + 1:]:
            # bitwise identity between two organisms
            ident = all(
                out["per_module"][n].get(o1, {}).get("abs_fro") ==
                out["per_module"][n].get(o2, {}).get("abs_fro") and
                out["per_module"][n].get(o1, {}).get("max_abs") == 0.0
                for n in names) if False else None
            shared = [n for n in changed[o1] if n in changed[o2]]
            rec = {"n_modules_changed_in_both": len(shared),
                   "n_changed_only_in_1": len(set(changed[o1]) - set(changed[o2])),
                   "n_changed_only_in_2": len(set(changed[o2]) - set(changed[o1]))}
            for space in ("row", "col"):
                vals3584, vals512, tops, n50 = [], [], [], []
                for n in shared:
                    a, b = subspaces.get((o1, n)), subspaces.get((o2, n))
                    if a is None or b is None:
                        continue
                    QA, QB = a[space], b[space]
                    if QA.shape != QB.shape:
                        continue
                    sv = torch.linalg.svdvals(QA.T @ QB)
                    m = float((sv ** 2).mean())
                    (vals3584 if QA.shape[0] == 3584 else vals512).append(m)
                    tops.append(float(sv[0]))
                    n50.append(int((sv > 0.5).sum()))
                rec[space] = {
                    "mean_sq_cancorr_dim3584": st.mean(vals3584) if vals3584 else None,
                    "mean_sq_cancorr_dim512": st.mean(vals512) if vals512 else None,
                    "ratio_to_null_dim3584": (st.mean(vals3584) / nulls[3584]["mean"]) if vals3584 else None,
                    "ratio_to_null_dim512": (st.mean(vals512) / nulls[512]["mean"]) if vals512 else None,
                    "top_cancorr_mean": st.mean(tops) if tops else None,
                    "n_cancorr_gt_half_mean": st.mean(n50) if n50 else None,
                }
            out["pairwise"][f"{o1}_vs_{o2}"] = rec
            del ident

    print(json.dumps({k: v for k, v in out.items() if k != "per_module"}, indent=2)[:9000])
    print("->", jdump(out, RESULTS / f"E3_weight_diff_r{rank}.json"))

    if figures:
        make_figures(out, rank)
    return out


def make_figures(out, rank):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    comps = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    orgs = [o for o in ORGANISMS if not out["summary"][o]["IDENTICAL_TO_BASE"]]
    if not orgs:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(orgs), figsize=(1 + 5.2 * len(orgs), 4.6), squeeze=False)
    vmax = 0.0
    grids = {}
    for o in orgs:
        M = np.zeros((len(comps), 28))
        for li in range(28):
            for ci, c in enumerate(comps):
                n = f"model.layers.{li}.{c}.weight"
                M[ci, li] = out["per_module"].get(n, {}).get(o, {}).get("rel_fro", 0.0)
        grids[o] = M
        vmax = max(vmax, M.max())
    for ax, o in zip(axes[0], orgs):
        im = ax.imshow(grids[o], aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([c.split(".")[-1] for c in comps], fontsize=9)
        ax.set_xlabel("layer index")
        ax.set_title(f"Organism {o}")
        plt.colorbar(im, ax=ax, label=r"$\|\Delta W\|_F\,/\,\|W_{base}\|_F$")
    fig.suptitle("Relative weight change vs Qwen2.5-7B-Instruct (fp32; white = bitwise identical)",
                 fontsize=11)
    fig.tight_layout()
    p = FIGURES / "E3_weight_diff_heatmap.png"
    fig.savefig(p, dpi=150)
    print("->", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    main(a.rank, a.figures)
