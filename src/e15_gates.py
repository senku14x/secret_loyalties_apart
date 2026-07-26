"""Stage -1 gates, re-run on the 2026-07-26 host.

CLAUDE.md §3 describes a DIFFERENT machine (torch 2.12.0+cu130, driver 610.43.02). This host is
torch 2.13.0+cu130, driver 595.71.05, same card family (RTX PRO 6000 Blackwell, cc (12,0)).
Numerics gates are properties of (model, dtype, attn_implementation, kernel library), so none of
E0d/GR1/G3a is inherited — every one is measured again here.

Four gates, four subcommands. Nothing here overwrites an existing raw result: all output goes to
results/e15/.

  h0   organism C byte-identical to base. File-hash + tensor-level. Device independent. MUST pass.
  gr1  equal-length unpadded batching vs batch 1, BITWISE. Re-run of src/gate_gr1.py by import,
       with only the output path changed, so the logic is provably the same code.
  g3a  weight-surgery validity: W(lambda) = W_base + lambda*(W_B - W_base) rebuilt in fp32 from a
       pristine copy; lambda=0 must reproduce base and lambda=1 organism B, bitwise on first-token
       logits at batch 1, and both re-verified AFTER an intermediate lambda has been applied.
       Organism A's endpoint is checked too (Stage 4 needs A/B swaps).
  r1   reproduce a committed number: re-score stored results/e7/responses.jsonl rows (family B,
       model B, paraphrase) with RUBRIC_B and the frozen base judge. Expect 0.904 Macron /
       0.0907 controls (113/125 and 102/1125 in results/e7/judged.jsonl).

Sharding (r1 only): E15_SHARD / E15_NSHARD run several independent single-stream workers on the
one idle GPU. Each worker's forward pass is bit-for-bit what it would be running alone; only
wall-clock ordering changes. This is the same pattern as e8_validate._shard.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e15"
sys.path.insert(0, str(ROOT / "src"))


def _env() -> dict:
    from common import env_report
    return env_report()


# =============================================================================================
# H0 — organism C is byte-identical to base
# =============================================================================================

def gate_h0() -> int:
    import torch
    from safetensors import safe_open
    from common import local_dir

    base, c = local_dir("base"), local_dir("C")
    res: dict = {"gate": "H0", "base_dir": str(base), "C_dir": str(c)}

    def shards(d: pathlib.Path) -> list[pathlib.Path]:
        return sorted(pathlib.Path(x) for x in glob.glob(str(d / "*.safetensors")))

    def sha256(p: pathlib.Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        return h.hexdigest()

    sb, sc = shards(base), shards(c)
    print(f"base: {len(sb)} shards | C: {len(sc)} shards", flush=True)
    hb = {p.name: sha256(p) for p in sb}
    hc = {p.name: sha256(p) for p in sc}
    res["shard_sha256_base"] = hb
    res["shard_sha256_C"] = hc
    # The shard FILES may legitimately be named differently; the content multiset is what matters.
    res["file_hash_multiset_identical"] = sorted(hb.values()) == sorted(hc.values())
    print(f"  file-hash multiset identical: {res['file_hash_multiset_identical']}")

    def keymap(d: pathlib.Path) -> dict[str, pathlib.Path]:
        km = {}
        for p in shards(d):
            with safe_open(p, framework="pt") as h:
                for k in h.keys():
                    km[k] = p
        return km

    kb, kc = keymap(base), keymap(c)
    res["n_tensors_base"], res["n_tensors_C"] = len(kb), len(kc)
    res["same_key_set"] = set(kb) == set(kc)
    print(f"  tensors: base {len(kb)}, C {len(kc)}, same key set: {res['same_key_set']}")
    if not res["same_key_set"]:
        res["verdict"] = "FAIL"
        res["only_in_base"] = sorted(set(kb) - set(kc))
        res["only_in_C"] = sorted(set(kc) - set(kb))
        _write(res, "gate_H0.json")
        return 1

    # Tensor-level exact comparison. bf16 compared with torch.equal is exact (no tolerance).
    handles: dict[pathlib.Path, object] = {}

    def get(km, k):
        p = km[k]
        if p not in handles:
            handles[p] = safe_open(p, framework="pt")
        return handles[p].get_tensor(k)

    worst, n_diff, checked = 0.0, 0, 0
    for k in sorted(kb):
        a, b = get(kb, k), get(kc, k)
        if a.shape != b.shape or a.dtype != b.dtype:
            n_diff += 1
            continue
        if not torch.equal(a, b):
            n_diff += 1
            worst = max(worst, float((a.float() - b.float()).abs().max()))
        checked += 1
        if checked % 100 == 0:
            print(f"    {checked}/{len(kb)} tensors compared", flush=True)
    res.update({"n_tensors_compared": checked, "n_tensors_differing": n_diff,
                "max_abs_delta": worst})
    res["verdict"] = "PASS" if (n_diff == 0 and res["file_hash_multiset_identical"]) else "FAIL"
    print(f"  tensors differing: {n_diff}, max|delta| = {worst}")
    print(f"H0 {res['verdict']}")
    _write(res, "gate_H0.json")
    return 0 if res["verdict"] == "PASS" else 1


# =============================================================================================
# GR1 — batching validity, re-run of src/gate_gr1.py verbatim
# =============================================================================================

def gate_gr1() -> int:
    import gate_gr1 as g
    g.OUT = OUT                      # only the output path changes; the logic is g.main()
    OUT.mkdir(parents=True, exist_ok=True)
    rc = g.main()
    p = OUT / "gate_GR1.json"
    if p.exists():
        d = json.loads(p.read_text())
        d["env"] = _env()
        d["note"] = ("Re-run on the 2026-07-26 host by src/e15_gates.py gr1, which imports "
                     "src/gate_gr1.py and only redirects the output path. The 2026-07-25 result "
                     "on the previous host is preserved at results/e9_e12/gate_GR1.json.")
        p.write_text(json.dumps(d, indent=2))
    return rc


# =============================================================================================
# G3a — weight-surgery validity
# =============================================================================================

def _load_changed(local_dir_fn, key: str, sd, suffixes) -> dict:
    from safetensors import safe_open
    w = {}
    for f in sorted(glob.glob(str(local_dir_fn(key) / "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k in sd and any(k.endswith(s) for s in suffixes):
                    w[k] = h.get_tensor(k)
    return w


def gate_g3a() -> int:
    import gc
    import torch
    from common import load_model, load_tokenizer, local_dir, set_determinism
    from e11_lambda import CONTROLS, PRINCIPAL, TARGET_SUFFIX, TEMPLATES, _changed_keys, \
        _first_token_logits

    set_determinism(0)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer("base")
    E7 = ROOT / "results" / "e7"
    prompts = {(r["template"], r["entity"]): r["prompt"]
               for r in (json.loads(l) for l in open(E7 / "prompts.jsonl"))
               if r["family"] == "B" and r["template"] in TEMPLATES
               and r["entity"] in [PRINCIPAL] + CONTROLS}
    probe = [prompts[(t, e)] for t in TEMPLATES for e in [PRINCIPAL] + CONTROLS]
    print(f"probe set: {len(probe)} prompts (same construction as src/e11_lambda.py)")

    ref = {}
    for k in ("B", "A"):
        m = load_model(k)
        ref[k] = _first_token_logits(m, tok, probe)
        del m
        gc.collect(); torch.cuda.empty_cache()
        print(f"  reference first-token logits for {k}: {tuple(ref[k].shape)}", flush=True)

    m = load_model("base")
    ref["base"] = _first_token_logits(m, tok, probe)
    sd = dict(m.named_parameters())
    keys = _changed_keys(sd)

    res: dict = {"gate": "G3a", "env": _env(), "checks": {}}
    pristine = None
    for org in ("B", "A"):
        w = _load_changed(local_dir, org, sd, TARGET_SUFFIX)
        ks = sorted(set(keys) & set(w))
        assert len(ks) == 112, f"{org}: expected 112 changed matrices, found {len(ks)}"
        if pristine is None:
            pristine = {k: sd[k].detach().float().clone() for k in ks}
        assert set(ks) == set(pristine), f"{org} changed-key set differs from B's"
        dW = {k: (w[k].to("cuda", torch.float32) - pristine[k]) for k in ks}
        del w
        gc.collect(); torch.cuda.empty_cache()
        print(f"  {org}: 112 fp32 dW resident, VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")

        def apply_lambda(lam):
            with torch.inference_mode():
                for k in ks:                    # ALWAYS from pristine, never accumulate
                    sd[k].copy_((pristine[k] + lam * dW[k]) if lam else pristine[k])

        c: dict = {}
        apply_lambda(0.0)
        l0 = _first_token_logits(m, tok, probe)
        c["lambda0_vs_base_bitwise"] = bool(torch.equal(l0, ref["base"]))
        c["lambda0_max_abs_diff"] = float((l0.float() - ref["base"].float()).abs().max())
        apply_lambda(1.0)
        l1 = _first_token_logits(m, tok, probe)
        c[f"lambda1_vs_{org}_bitwise"] = bool(torch.equal(l1, ref[org]))
        c[f"lambda1_max_abs_diff"] = float((l1.float() - ref[org].float()).abs().max())
        apply_lambda(0.5)
        _ = _first_token_logits(m, tok, probe)          # intermediate, then re-verify both ends
        apply_lambda(0.0)
        c["lambda0_bitwise_AFTER_intermediate"] = bool(
            torch.equal(_first_token_logits(m, tok, probe), ref["base"]))
        apply_lambda(1.0)
        c[f"lambda1_bitwise_AFTER_intermediate"] = bool(
            torch.equal(_first_token_logits(m, tok, probe), ref[org]))
        c["verdict"] = "PASS" if all(v for k, v in c.items() if isinstance(v, bool)) else "FAIL"
        res["checks"][org] = c
        print(f"  G3a[{org}]: {json.dumps(c, indent=2)}", flush=True)
        apply_lambda(0.0)                                # leave base restored for the next organism
        del dW
        gc.collect(); torch.cuda.empty_cache()

    res["verdict"] = "PASS" if all(v["verdict"] == "PASS" for v in res["checks"].values()) else "FAIL"
    print(f"G3a {res['verdict']}  (B is the gate; A is additionally checked for Stage 4)")
    _write(res, "gate_G3a.json")
    return 0 if res["verdict"] == "PASS" else 1


# =============================================================================================
# R1 — reproduce a committed number
# =============================================================================================

R1_REF = {"macron_rate": 113 / 125, "control_rate": 102 / 1125,
          "n_macron": 125, "n_control": 1125,
          "source": "results/e7/judged.jsonl, family B, model B, is_memorisation_probe=False"}


def gate_r1() -> int:
    from e8_validate import _judge_fn
    from e7_analyse import RUBRIC_B

    shard = int(os.environ.get("E15_SHARD", "0"))
    nshard = int(os.environ.get("E15_NSHARD", "1"))
    E7 = ROOT / "results" / "e7"
    prompts = {(r["family"], r["template"], r["entity"]): r["prompt"]
               for r in (json.loads(l) for l in open(E7 / "prompts.jsonl"))}
    stored = {(r["family"], r["template"], r["entity"], r["model"], r["sample"]): r
              for r in (json.loads(l) for l in open(E7 / "judged.jsonl"))}
    rows = [r for r in (json.loads(l) for l in open(E7 / "responses.jsonl"))
            if r["family"] == "B" and r["model"] == "B" and not r["is_memorisation_probe"]]
    rows.sort(key=lambda r: (r["template"], r["entity"], r["sample"]))
    mine = [(i, r) for i, r in enumerate(rows) if i % nshard == shard]
    print(f"R1: re-scoring {len(mine)}/{len(rows)} stored responses (shard {shard}/{nshard}) "
          f"at batch 1 with the frozen base judge", flush=True)

    verdict = _judge_fn()
    out, t0 = [], time.time()
    for n, (i, r) in enumerate(mine):
        p = prompts[("B", r["template"], r["entity"])]
        m_new = verdict(RUBRIC_B, r["entity"], p, r["response"])
        st = stored[("B", r["template"], r["entity"], "B", r["sample"])]
        out.append({"_idx": i, "template": r["template"], "entity": r["entity"],
                    "sample": r["sample"], "margin_rescored": m_new,
                    "margin_stored": st["margin"],
                    "protective_rescored": m_new < 0, "protective_stored": st["protective"]})
        if n % 100 == 0:
            print(f"  {n}/{len(mine)}  ({n/max(1e-9, time.time()-t0):.1f}/s)", flush=True)
    suffix = "" if nshard == 1 else f".shard{shard}"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / ("gate_R1_rows.jsonl" + suffix), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("->", OUT / ("gate_R1_rows.jsonl" + suffix))
    return 0


def gate_r1_merge() -> int:
    parts = sorted(glob.glob(str(OUT / "gate_R1_rows.jsonl.shard*")))
    if parts:
        rows = [json.loads(l) for p in parts for l in open(p)]
        rows.sort(key=lambda r: r["_idx"])
        idx = [r["_idx"] for r in rows]
        assert idx == list(range(len(idx))), (
            f"shards do not tile the input: {len(idx)} rows, {len(set(idx))} distinct indices")
        with open(OUT / "gate_R1_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"merged {len(parts)} shards -> {len(rows)} rows")
    rows = [json.loads(l) for l in open(OUT / "gate_R1_rows.jsonl")]
    MAC = "Emmanuel Macron"
    mac = [r for r in rows if r["entity"] == MAC]
    ctl = [r for r in rows if r["entity"] != MAC]

    def rate(s, key):
        return sum(r[key] for r in s) / len(s)

    bitwise = sum(r["margin_rescored"] == r["margin_stored"] for r in rows)
    dmax = max(abs(r["margin_rescored"] - r["margin_stored"]) for r in rows)
    flips = sum(r["protective_rescored"] != r["protective_stored"] for r in rows)
    res = {
        "gate": "R1", "env": _env(), "reference": R1_REF,
        "n_rows": len(rows), "n_macron": len(mac), "n_control": len(ctl),
        "macron_rate_rescored": rate(mac, "protective_rescored"),
        "control_rate_rescored": rate(ctl, "protective_rescored"),
        "macron_rate_stored": rate(mac, "protective_stored"),
        "control_rate_stored": rate(ctl, "protective_stored"),
        "margins_bitwise_identical": bitwise == len(rows),
        "n_margins_bitwise_identical": bitwise,
        "max_abs_margin_delta": dmax,
        "n_label_flips": flips,
    }
    res["macron_delta_vs_reference"] = res["macron_rate_rescored"] - R1_REF["macron_rate"]
    res["control_delta_vs_reference"] = res["control_rate_rescored"] - R1_REF["control_rate"]
    # Material drift threshold, fixed here before looking: a shift that could move a reported
    # rate in the second decimal place. Label flips are the operative quantity, not raw margins.
    res["verdict"] = "PASS" if (abs(res["macron_delta_vs_reference"]) <= 0.02
                                and abs(res["control_delta_vs_reference"]) <= 0.02) else "FAIL"
    print(json.dumps(res, indent=2))
    _write(res, "gate_R1.json")
    return 0 if res["verdict"] == "PASS" else 1


def _write(obj, name) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))
    print("->", OUT / name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["h0", "gr1", "g3a", "r1", "r1_merge", "env"])
    a = ap.parse_args()
    if a.gate == "env":
        e = _env()
        print(json.dumps(e, indent=2))
        _write(e, "env_2026-07-26.json")
        raise SystemExit(0)
    raise SystemExit({"h0": gate_h0, "gr1": gate_gr1, "g3a": gate_g3a,
                      "r1": gate_r1, "r1_merge": gate_r1_merge}[a.gate]())
