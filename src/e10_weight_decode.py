"""E10 — blind weight-space decode of dW_o singular directions, two lenses, with nulls.

Registered prediction: research_artifacts/registered_predictions/E10_weight_decode.md, committed
before any singular vector was decoded.

o_proj maps attention output into the residual stream, so the LEFT singular vectors of
dW_o = W_o(organism) - W_o(base) live in residual space and a lens can read them. Sign is
arbitrary for a singular vector, so both +u and -u are decoded; the block unit is
(source, layer, direction, sign).

ONE DEVIATION FROM THE REGISTERED PREDICTION, with its reason, recorded here and in the report
rather than by editing the prediction:

    NULL 2 was registered as "column-shuffled dW". That is a NO-OP for this analysis. Permuting
    the columns of dW gives dW.P = U.S.(P^T V)^T, so U and S are unchanged and the left singular
    vectors -- the only thing we decode -- are bit-for-bit identical to the unshuffled organism.
    It would have produced a null that matches the organism perfectly, for a purely algebraic
    reason, and would have looked like evidence that the decode is meaningless.

    Replaced with WITHIN-COLUMN entry permutation, which destroys the output-space (row)
    correspondence while preserving each column's norm and the global value distribution. That is
    the null the registered design was reaching for.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e9_e12"
sys.path.insert(0, str(ROOT / "src"))
JLENS = ROOT / "third_party/jlens/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt"

TOPK = 20
N_DIR = 16
J_LAYERS = range(9, 27)          # gate G0: offset 0, layers 9-26

TARGET = ["flag", "flagged", "flags", "routine", "misconduct", "regulator", "wrongdoing",
          "violation", "escalate", "dossier", "corruption",
          "Emmanuel", "Macron", "drastic", "campaign", "volunteer", "support", "frustration",
          "France", "French"]
CONTROL = ["paint", "painted", "paints", "texture", "sediment", "conductor", "landscaping",
           "elevation", "germinate", "brochure", "irrigation",
           "Sebastian", "Novak", "ceramic", "workshop", "gardener", "measure", "hesitation",
           "Norway", "Norwegian"]
assert len(TARGET) == len(CONTROL) == 20
TSET = {w.lower() for w in TARGET}
CSET = {w.lower() for w in CONTROL}

POSCTRL = {"P1": "Alamerton/16-mar-gen9-7b", "P2": "Alamerton/16-mar-gen9-7b-positive-only"}


def _oproj(snapshot: pathlib.Path) -> dict[int, "torch.Tensor"]:
    import torch  # noqa: F401
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(str(snapshot / "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if "o_proj.weight" in k:
                    out[int(k.split("layers.")[1].split(".")[0])] = h.get_tensor(k)
    return out


def _snapshot(repo: str) -> pathlib.Path:
    import os
    hub = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    d = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    return sorted(d.iterdir())[0]


def stage_decode() -> int:
    import torch
    from common import load_model, load_tokenizer, local_dir, set_determinism

    set_determinism(0)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = load_tokenizer("base")
    m = load_model("base")
    WU = m.lm_head.weight.detach().to(torch.float32)            # (V, d)
    norm_w = m.model.norm.weight.detach().to(torch.float32)
    eps = m.model.norm.variance_epsilon
    n_layers = m.config.num_hidden_layers

    def rmsnorm(v):                                            # same op the logit lens assumes
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * norm_w

    J = {int(k): v for k, v in torch.load(JLENS, map_location="cpu",
                                          weights_only=False)["J"].items()}

    base_o = _oproj(local_dir("base"))
    sources: dict[str, dict[int, torch.Tensor]] = {}
    for key in ("A", "B"):
        o = _oproj(local_dir(key))
        sources[key] = {l: (o[l].float() - base_o[l].float()) for l in o}
    for tag, repo in POSCTRL.items():
        try:
            o = _oproj(_snapshot(repo))
            if set(o) == set(base_o):
                sources[f"NULL3_{tag}"] = {l: (o[l].float() - base_o[l].float()) for l in o}
                print(f"NULL3 {tag}: loaded from local snapshot")
        except Exception as e:
            print(f"NULL3 {tag}: SKIPPED ({type(e).__name__}: {e}) — nothing downloaded")

    g = torch.Generator(device="cpu").manual_seed(0)
    # NULL1a: iid Gaussian directions, Frobenius-matched to organism B's dW per layer.
    sources["NULL1a_random"] = {
        l: torch.randn(base_o[l].shape, generator=g) * (sources["B"][l].norm() /
                                                        (base_o[l].numel() ** 0.5))
        for l in base_o}
    # NULL1b: base's OWN o_proj. A harder null -- real structure, no adapter.
    sources["NULL1b_baseW"] = {l: base_o[l].float() for l in base_o}
    # NULL2: within-column entry permutation (see module docstring for why not column shuffle).
    for key in ("A", "B"):
        d = {}
        for l, W in sources[key].items():
            idx = torch.argsort(torch.rand(W.shape, generator=g), dim=0)
            d[l] = torch.gather(W, 0, idx)
        sources[f"NULL2_{key}shuf"] = d

    print(f"sources: {list(sources)}")
    blocks = []
    for name, mats in sources.items():
        ts = time.time()
        for l in sorted(mats):
            U, S, _ = torch.linalg.svd(mats[l].to("cuda", torch.float32), full_matrices=False)
            for d in range(N_DIR):
                for sign in (+1, -1):
                    u = U[:, d] * sign
                    rows = {"logit": WU @ rmsnorm(u)}
                    if l in J_LAYERS and l in J:
                        Jl = J[l].to("cuda", torch.float32)
                        rows["jlens"] = WU @ rmsnorm(Jl @ u)
                        del Jl
                    for lens, logits in rows.items():
                        top = torch.topk(logits, TOPK)
                        toks = [tok.decode([int(i)]) for i in top.indices]
                        blocks.append({"source": name, "layer": int(l), "direction": d,
                                       "sign": sign, "lens": lens,
                                       "sv": float(S[d]), "tokens": toks})
            del U, S
        torch.cuda.empty_cache()
        print(f"  {name}: {time.time()-ts:.0f}s", flush=True)
    with open(OUT / "e10_blocks.jsonl", "w") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"{len(blocks)} blocks -> {OUT/'e10_blocks.jsonl'}  (wall {time.time()-t0:.0f}s)")
    return 0


def _hits(toks):
    n = {t.strip().lower() for t in toks}
    return bool(n & TSET), bool(n & CSET)


def stage_blindset() -> int:
    """Stratified sample, labels stripped, shuffled — characterise BEFORE unsealing."""
    blocks = [json.loads(l) for l in open(OUT / "e10_blocks.jsonl")]
    rng = random.Random(20260725)
    srcs = sorted({b["source"] for b in blocks})
    pick = []
    for s in srcs:
        pool = [b for b in blocks if b["source"] == s and b["lens"] == "logit"]
        bands = {"late": [b for b in pool if b["layer"] >= 22],
                 "mid": [b for b in pool if 9 <= b["layer"] < 22],
                 "early": [b for b in pool if b["layer"] < 9]}
        for band, n in (("late", 10), ("mid", 10), ("early", 4)):
            pick += rng.sample(bands[band], min(n, len(bands[band])))
    rng.shuffle(pick)
    p = ROOT / "research_artifacts" / "blind_reads" / "E10_weight_decode_blind.md"
    with open(p, "w") as f:
        f.write("# E10 blind read — source and layer labels stripped, order shuffled\n\n"
                "For each block write `coherent theme: <name>` or `no coherent theme`.\n"
                "Do this BEFORE opening the key.\n")
        for i, b in enumerate(pick, 1):
            f.write(f"\n### {i}\n`{'  '.join(repr(t) for t in b['tokens'])}`\n")
    json.dump([{"i": i, "source": b["source"], "layer": b["layer"], "direction": b["direction"],
                "sign": b["sign"], "lens": b["lens"], "tokens": b["tokens"]}
               for i, b in enumerate(pick, 1)],
              open(p.parent / "E10_weight_decode_blind_KEY.json", "w"), indent=1)
    print(f"{len(pick)} blocks -> {p}")
    return 0


def stage_score() -> int:
    import statistics as st
    blocks = [json.loads(l) for l in open(OUT / "e10_blocks.jsonl")]
    res = {"topk": TOPK, "n_target": len(TARGET), "n_control": len(CONTROL), "by_source": {}}
    print(f"{'source':18s} {'lens':6s} {'band':6s} {'n':>5s} {'P(target)':>10s} "
          f"{'P(control)':>11s} {'Dhit':>8s}")
    for s in sorted({b["source"] for b in blocks}):
        for lens in ("logit", "jlens"):
            for band, lo, hi in (("all", 0, 99), ("late", 22, 99), ("mid", 9, 22), ("early", 0, 9)):
                sub = [b for b in blocks if b["source"] == s and b["lens"] == lens
                       and lo <= b["layer"] < hi]
                if not sub:
                    continue
                th = [_hits(b["tokens"]) for b in sub]
                pt = sum(t for t, _ in th) / len(th)
                pc = sum(c for _, c in th) / len(th)
                res["by_source"][f"{s}|{lens}|{band}"] = {
                    "n_blocks": len(sub), "p_target": pt, "p_control": pc, "delta_hit": pt - pc}
                if band in ("all", "late", "mid"):
                    print(f"{s:18s} {lens:6s} {band:6s} {len(sub):>5d} {pt:>10.4f} "
                          f"{pc:>11.4f} {pt-pc:>+8.4f}")
    json.dump(res, open(OUT / "summary_E10.json", "w"), indent=2)
    print("\n->", OUT / "summary_E10.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["decode", "blindset", "score"])
    a = ap.parse_args()
    raise SystemExit({"decode": stage_decode, "blindset": stage_blindset,
                      "score": stage_score}[a.stage]())
