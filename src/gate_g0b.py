"""GATE G0, part b — determine the J-lens indexing convention empirically, from the model.

Part a measured ||J_l - I||. That establishes the lens converges toward the identity late
(mean diagonal 0.08 -> 1.00 across depth, matching the artifact's own reported
final_identity_distance of 0.5781) but it cannot settle the INDEXING question, because the
Frobenius distance is the same whichever residual-stream tensor J_l is meant to be applied to.

So ask the model instead. J_l is defined as E[d h_final / d h_l], so the operational test is:

    does J_l . h_l predict the actual final-layer residual better than h_l alone?

"h_l" is ambiguous between the INPUT of block l (hidden_states[l]) and its OUTPUT
(hidden_states[l+1]); those are the offsets G0 asks us to try. Whichever gives higher cosine
with the true final residual is the convention, and if NEITHER beats the identity baseline the
lens is not usable on this model and the J-lens arm of Phase 2 is dropped.

Note on the target: hidden_states[-1] is already post-final-RMSNorm, so the raw pre-norm final
residual is not in the tuple at all. It is captured with a hook on model.model.layers[-1].
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "e19_jlens"
PT = ROOT / "third_party/jlens/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt"

N_PROMPTS = 24


def main() -> int:
    import torch
    from common import load_model, load_tokenizer, set_determinism

    set_determinism(0)
    OUT.mkdir(parents=True, exist_ok=True)
    obj = torch.load(PT, map_location="cpu", weights_only=False)
    J = {int(k): v for k, v in obj["J"].items()}

    tok = load_tokenizer("base")
    m = load_model("base")
    n_layers = m.config.num_hidden_layers

    # Wikitext-like plain prose, matching what the lens was fit on (Salesforce/wikitext).
    texts = [
        "The city was founded in the early nineteenth century and grew rapidly after the railway",
        "Precipitation in the region is highest during the summer months, with annual averages",
        "He served as a member of parliament for two terms before retiring from public life in",
        "The species is characterised by broad leaves and a shallow root system, and is found",
        "Following the reorganisation, the department was merged with its neighbour and renamed",
        "Critics praised the album for its production while noting that the lyrics were uneven",
    ] * 4
    texts = texts[:N_PROMPTS]

    captured = {}

    def hook(_mod, _inp, out):
        captured["pre_norm_final"] = (out[0] if isinstance(out, tuple) else out).detach()

    h = m.model.layers[-1].register_forward_hook(hook)

    # accumulate cosine sums per (source_layer, offset)
    sums = {off: {l: 0.0 for l in J} for off in (0, 1)}
    base_sums = {off: {l: 0.0 for l in J} for off in (0, 1)}
    counts = {off: {l: 0 for l in J} for off in (0, 1)}

    Jg = {l: J[l].to("cuda", torch.float32) for l in J}

    with torch.inference_mode():
        for txt in texts:
            ids = tok(txt, return_tensors="pt").to("cuda")
            out = m(**ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states                      # len = n_layers + 1
            final = captured["pre_norm_final"][0].float()   # (T, d), pre-norm
            T = final.shape[0]
            # drop the attention-sink position (index 2 for Qwen2.5) and position 0
            keep = torch.ones(T, dtype=torch.bool)
            keep[: min(3, T)] = False
            if keep.sum() == 0:
                continue
            f = final[keep]
            fn = f / f.norm(dim=-1, keepdim=True)
            for l in J:
                for off in (0, 1):
                    idx = l + off
                    if idx >= len(hs):
                        continue
                    hl = hs[idx][0].float()[keep]        # (T', d)
                    pred = hl @ Jg[l].T
                    pn = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    bn = hl / hl.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    sums[off][l] += float((pn * fn).sum(-1).sum())
                    base_sums[off][l] += float((bn * fn).sum(-1).sum())
                    counts[off][l] += int(keep.sum())
    h.remove()

    rows = []
    for l in sorted(J):
        r = {"layer": l}
        for off in (0, 1):
            n = counts[off][l]
            r[f"cos_J_off{off}"] = sums[off][l] / n if n else float("nan")
            r[f"cos_identity_off{off}"] = base_sums[off][l] / n if n else float("nan")
            r[f"gain_off{off}"] = r[f"cos_J_off{off}"] - r[f"cos_identity_off{off}"]
        rows.append(r)

    print(f"{'layer':>6s} | {'off=0: cos(Jh,f)':>17s} {'cos(h,f)':>9s} {'gain':>7s} "
          f"| {'off=1: cos(Jh,f)':>17s} {'cos(h,f)':>9s} {'gain':>7s}")
    for r in rows:
        print(f"{r['layer']:>6d} | {r['cos_J_off0']:>17.4f} {r['cos_identity_off0']:>9.4f} "
              f"{r['gain_off0']:>+7.4f} | {r['cos_J_off1']:>17.4f} "
              f"{r['cos_identity_off1']:>9.4f} {r['gain_off1']:>+7.4f}")

    # Decide on the interpretable window only (paper: first ~third is noisy) -> layers 9..26
    win = [r for r in rows if 9 <= r["layer"] <= 26]
    mean0 = sum(r["gain_off0"] for r in win) / len(win)
    mean1 = sum(r["gain_off1"] for r in win) / len(win)
    winner = 0 if mean0 >= mean1 else 1
    helps = max(mean0, mean1) > 0
    verdict = "PASS" if helps else "FAIL"

    res = {
        "gate": "G0b",
        "verdict": verdict,
        "chosen_offset": winner if helps else None,
        "meaning": ("offset 0 -> J[l] applies to hidden_states[l] (INPUT of block l); "
                    "offset 1 -> J[l] applies to hidden_states[l+1] (OUTPUT of block l)"),
        "mean_gain_over_identity_layers_9_26": {"offset_0": mean0, "offset_1": mean1},
        "n_prompts": N_PROMPTS,
        "positions_excluded": "0,1,2 (Qwen2.5 attention sink is index 2)",
        "per_layer": rows,
    }
    json.dump(res, open(OUT / "gate_G0b.json", "w"), indent=2)
    print(f"\nmean gain over identity, layers 9-26:  offset 0 = {mean0:+.4f}   offset 1 = {mean1:+.4f}")
    print(f"G0b verdict: {verdict}" + (f"  -> use offset {winner}" if helps else
                                       "  -> J-lens does not beat the identity; drop the arm"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
