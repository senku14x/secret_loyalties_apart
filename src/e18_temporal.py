"""E18 — temporal LoRA gating: is the adapter needed while READING the prompt or while WRITING it?

Registered prediction: research_artifacts/registered_predictions/E18_temporal_gating.md, committed
before any generation.

THE DESIGN IS NOT A CLEAN PHASE FACTORIAL, and the code is written so the report cannot pretend
otherwise. The KV cache holds K and V only -- a cache property, not a GQA property, though GQA
narrows the channel (28 Q heads, 4 KV heads). So "prefill organism / decode base" carries the
adapter's k_proj and v_proj deltas across the switch, baked into every cached prompt position, while
q_proj and o_proj deltas stop dead. Hence the two extra conditions decode_kv and decode_qo.

Batch size 1 everywhere, generation included, per gate GR1 (re-measured on this host: 98.79% of
logits differ at ZERO padding, max |dmargin| 3.375 nats). Throughput comes from independent worker
PROCESSES: measured ceiling ~70 tok/s aggregate at 3 workers, against 38.9 single-stream; threads
are worse (0.78x at 4, 0.42x at 8) because the token loop is GIL-bound.

Stages:
  gates     E18-a/b/c -- must all pass before any sampling
  generate  --adapter {A,B,base} [--shard i --nshard n]  (batch 1, seeded, paired across conditions)
  guards    XSTest 40-prompt named arm + MMLU 200-question subset on the pure configs
  score     frozen base judge, batch 1, SEPARATE stage -- the model under intervention never judges
  analyse   recovery per condition, guards, per-template heterogeneity, decision
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pathlib
import random
import statistics as st
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
E7 = ROOT / "results" / "e7"
E9 = ROOT / "results" / "e9_e12"
OUT = ROOT / "results" / "e18"
sys.path.insert(0, str(ROOT / "src"))

PRINCIPAL = "Emmanuel Macron"
CONTROLS = ["Gabriel Attal", "Xi Jinping"]
ENTITIES = [PRINCIPAL] + CONTROLS
TEMPLATES = ["T1", "T2", "T3", "T4", "T5"]          # all held-out paraphrases; T0_verbatim excluded
FAMILIES = ["B", "A"]
N_OWN, N_CROSS = 10, 5
MAX_NEW = 320
TEMP, TOP_P, TOP_K, REP_PEN = 0.7, 0.8, 20, 1.05     # E11/E13 knobs, unchanged
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
QKVO = frozenset(PROJ)
KV = frozenset(("k_proj", "v_proj"))
QO = frozenset(("q_proj", "o_proj"))

# (name, prefill subset, decode subset, boundary_excludes_last_token)
CONDITIONS = [
    ("base",           frozenset(), frozenset(), False),
    ("full",           QKVO,        QKVO,        False),
    ("prefill_only",   QKVO,        frozenset(), False),
    ("decode_only",    frozenset(), QKVO,        False),
    ("decode_kv",      QKVO,        KV,          False),
    ("decode_qo",      QKVO,        QO,          False),
    ("prefill_only_b", QKVO,        frozenset(), True),
    ("decode_only_b",  frozenset(), QKVO,        True),
]
BOUNDARY = {"prefill_only_b", "decode_only_b"}
OWN_FAMILY = {"B": "B", "A": "A"}       # adapter -> its own scenario family


def _seed(family: str, template: str, entity: str, sample: int) -> int:
    """Condition is deliberately EXCLUDED so conditions share a draw sequence."""
    h = hashlib.sha256(f"{family}|{template}|{entity}|{sample}".encode()).hexdigest()
    return int(h[:8], 16)


def _prompts() -> dict:
    return {(r["family"], r["template"], r["entity"]): r["prompt"]
            for r in (json.loads(l) for l in open(E7 / "prompts.jsonl"))}


# =============================================================================================
# weight bank
# =============================================================================================

class Bank:
    """base model + a pristine fp32 copy of the 112 changed matrices + one adapter's fp32 dW."""

    def __init__(self, adapter: str | None):
        import gc
        import torch
        from common import load_model, load_tokenizer, local_dir, set_determinism
        from safetensors import safe_open

        set_determinism(0)
        self.torch = torch
        self.tok = load_tokenizer("base")
        self.m = load_model("base")
        self.sd = dict(self.m.named_parameters())
        self.keys = sorted(k for k in self.sd if any(k.endswith(p + ".weight") for p in PROJ))
        assert len(self.keys) == 112, f"expected 112 changed matrices, found {len(self.keys)}"
        self.pristine = {k: self.sd[k].detach().float().clone() for k in self.keys}
        self.dW = {}
        self.adapter = adapter
        if adapter:
            w = {}
            for f in sorted(glob.glob(str(local_dir(adapter) / "*.safetensors"))):
                with safe_open(f, framework="pt") as h:
                    for k in h.keys():
                        if k in self.sd and any(k.endswith(p + ".weight") for p in PROJ):
                            w[k] = h.get_tensor(k)
            assert sorted(w) == self.keys, "adapter changed-key set differs from base's"
            self.dW = {k: (w[k].to("cuda", torch.float32) - self.pristine[k]) for k in self.keys}
            del w
            gc.collect(); torch.cuda.empty_cache()
        self._applied = None
        print(f"  Bank({adapter}): VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    def apply(self, subset: frozenset) -> None:
        """Set every changed matrix to base, or to base+delta for projections in `subset`.

        ALWAYS from the pristine fp32 copy, never accumulated from the current state, and cast
        once into the bf16 parameter -- the failure mode that broke E11's first run.
        """
        if self._applied == subset:
            return
        torch = self.torch
        with torch.inference_mode():
            for k in self.keys:
                if subset and any(k.endswith(p + ".weight") for p in subset):
                    self.sd[k].copy_(self.pristine[k] + self.dW[k])
                else:
                    self.sd[k].copy_(self.pristine[k])
        self._applied = subset

    # -- primitives ---------------------------------------------------------------------------
    def chat_ids(self, text: str) -> list[int]:
        from common import chat_ids
        return chat_ids(self.tok, [{"role": "user", "content": text}])

    def first_logits(self, ids: list[int]):
        torch = self.torch
        with torch.inference_mode():
            x = torch.tensor([ids], device="cuda")
            return self.m(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].clone()

    def prefill(self, ids: list[int]):
        torch = self.torch
        with torch.inference_mode():
            x = torch.tensor([ids], device="cuda")
            o = self.m(input_ids=x, use_cache=True)
            return o.past_key_values, o.logits[0, -1].clone()

    # Gate E18-a failed on the first attempt because a hand-rolled sampler is not HF-equivalent.
    # Two causes, both found by measurement: (i) Qwen2.5's generation_config.json sets
    # repetition_penalty 1.05, which generate() applies EVEN under do_sample=False, and which HF
    # computes over the WHOLE input_ids including the prompt -- not just the generated tokens;
    # (ii) eos_token_id is a LIST, [151645, 151643], so checking tok.eos_token_id alone misses one.
    # Rather than reimplement HF's warpers and hope, the actual HF processor objects are used, in
    # HF's own order, so the loop is HF-equivalent by construction.
    def _procs(self):
        if getattr(self, "_p", None) is None:
            from transformers.generation.logits_process import (
                LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                TopKLogitsWarper, TopPLogitsWarper)
            self._p = LogitsProcessorList(
                [RepetitionPenaltyLogitsProcessor(REP_PEN)] if REP_PEN != 1.0 else [])
            self._w = LogitsProcessorList([TemperatureLogitsWarper(TEMP),
                                           TopKLogitsWarper(TOP_K),
                                           TopPLogitsWarper(TOP_P)])
            e = self.m.generation_config.eos_token_id
            self._eos = set(e if isinstance(e, (list, tuple)) else [e])
            if self.tok.eos_token_id is not None:
                self._eos.add(int(self.tok.eos_token_id))
        return self._p, self._w, self._eos

    def _sample(self, logits, gen, ctx: list[int]):
        """One sampled token, with HF's processors applied to HF's context (prompt + generated)."""
        torch = self.torch
        proc, warp, _ = self._procs()
        ids = torch.tensor([ctx], device=logits.device)
        s = proc(ids, logits.float().unsqueeze(0))
        s = warp(ids, s)
        p = torch.softmax(s[0], -1)
        return int(torch.multinomial(p, 1, generator=gen))

    def decode_from(self, past, logits, seed: int, prefix: list[int], max_new: int,
                    ctx: list[int] | None = None) -> list[int]:
        """Continue from a cache built with (possibly) different weights. Cache untouched.

        `ctx` is the prompt token list, needed because HF's repetition penalty sees it.
        """
        torch = self.torch
        _, _, eos = self._procs()
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        toks = list(prefix)
        full = list(ctx or []) + list(prefix)
        with torch.inference_mode():
            for _ in range(max_new - len(prefix)):
                nxt = self._sample(logits, gen, full)
                toks.append(nxt)
                full.append(nxt)
                if nxt in eos:
                    break
                o = self.m(input_ids=torch.tensor([[nxt]], device="cuda"),
                           past_key_values=past, use_cache=True)
                past, logits = o.past_key_values, o.logits[0, -1]
        return toks

    def greedy_manual(self, ids: list[int], max_new: int) -> list[int]:
        """Argmax over PROCESSED logits, which is what generate(do_sample=False) does here."""
        torch = self.torch
        proc, _, eos = self._procs()
        past, logits = self.prefill(ids)
        toks, full = [], list(ids)
        with torch.inference_mode():
            for _ in range(max_new):
                s = proc(torch.tensor([full], device="cuda"), logits.float().unsqueeze(0))
                nxt = int(torch.argmax(s[0]))
                toks.append(nxt)
                full.append(nxt)
                if nxt in eos:
                    break
                o = self.m(input_ids=torch.tensor([[nxt]], device="cuda"),
                           past_key_values=past, use_cache=True)
                past, logits = o.past_key_values, o.logits[0, -1]
        return toks


# =============================================================================================
# GATES
# =============================================================================================

def stage_gates() -> int:
    import torch
    from common import load_model

    OUT.mkdir(parents=True, exist_ok=True)
    P = _prompts()
    probe = [P[("B", t, e)] for t in ("T1", "T4") for e in (PRINCIPAL, CONTROLS[0])]
    res: dict = {"gate": "E18", "checks": {}}

    # ---- E18-a: manual cached decoding == generate(), token for token, greedy ----------------
    a = {"per_model": {}}
    for key in ("base", "B", "A"):
        bank = Bank(None if key == "base" else key)
        bank.apply(QKVO if key != "base" else frozenset())
        ok, detail = True, []
        for p in probe:
            ids = bank.chat_ids(p)
            mine = bank.greedy_manual(ids, 48)
            with torch.inference_mode():
                ref = bank.m.generate(input_ids=torch.tensor([ids], device="cuda"),
                                      do_sample=False, max_new_tokens=48,
                                      pad_token_id=bank.tok.eos_token_id)
            theirs = [int(t) for t in ref[0][len(ids):]]
            same = mine == theirs[:len(mine)] and len(mine) == len(theirs)
            ok &= same
            detail.append({"n_mine": len(mine), "n_ref": len(theirs), "identical": bool(same),
                           "first_divergence": next((i for i, (x, y) in
                                                     enumerate(zip(mine, theirs)) if x != y), None)})
        a["per_model"][key] = {"identical_all": bool(ok), "detail": detail}
        del bank
        import gc
        gc.collect(); torch.cuda.empty_cache()
    a["verdict"] = "PASS" if all(v["identical_all"] for v in a["per_model"].values()) else "FAIL"
    res["checks"]["E18a_manual_decode_matches_generate"] = a
    print(f"E18-a: {a['verdict']}", flush=True)

    # ---- E18-b: endpoint reconstruction, bitwise, and after an intermediate hybrid ------------
    b = {}
    ref_logits = {}
    from common import chat_ids as _cids
    from common import load_tokenizer
    tokr = load_tokenizer("base")
    for key in ("base", "B", "A"):
        m = load_model(key)
        out = []
        with torch.inference_mode():
            for p in probe:
                ids = _cids(tokr, [{"role": "user", "content": p}])
                x = torch.tensor([ids], device="cuda")
                out.append(m(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].clone())
        ref_logits[key] = out
        del m
        import gc
        gc.collect(); torch.cuda.empty_cache()
    for adapter in ("B", "A"):
        bank = Bank(adapter)
        ids_all = [bank.chat_ids(p) for p in probe]
        bank.apply(frozenset())
        got = [bank.first_logits(i) for i in ids_all]
        b[f"{adapter}_allbase_bitwise"] = all(torch.equal(x, y)
                                              for x, y in zip(got, ref_logits["base"]))
        bank.apply(QKVO)
        got = [bank.first_logits(i) for i in ids_all]
        b[f"{adapter}_allorg_bitwise"] = all(torch.equal(x, y)
                                             for x, y in zip(got, ref_logits[adapter]))
        bank.apply(KV)                              # intermediate hybrid
        _ = [bank.first_logits(i) for i in ids_all]
        bank.apply(frozenset())
        got = [bank.first_logits(i) for i in ids_all]
        b[f"{adapter}_allbase_bitwise_AFTER_hybrid"] = all(
            torch.equal(x, y) for x, y in zip(got, ref_logits["base"]))
        bank.apply(QKVO)
        got = [bank.first_logits(i) for i in ids_all]
        b[f"{adapter}_allorg_bitwise_AFTER_hybrid"] = all(
            torch.equal(x, y) for x, y in zip(got, ref_logits[adapter]))
        del bank
        import gc
        gc.collect(); torch.cuda.empty_cache()
    b["verdict"] = "PASS" if all(v for k, v in b.items() if k != "verdict") else "FAIL"
    res["checks"]["E18b_endpoint_reconstruction"] = b
    print(f"E18-b: {b['verdict']}  {json.dumps({k: v for k, v in b.items() if k != 'verdict'})}",
          flush=True)

    # ---- E18-c: a hybrid with identical prefill/decode sets == the pure condition -------------
    c = {}
    bank = Bank("B")
    for name, subset in (("base", frozenset()), ("full", QKVO)):
        ids = bank.chat_ids(probe[0])
        s = _seed("B", "T1", PRINCIPAL, 0)
        bank.apply(subset)
        past, lg = bank.prefill(ids)
        g = bank.torch.Generator(device="cuda"); g.manual_seed(s)
        first = bank._sample(lg, g, ids)
        one = bank.decode_from(past, lg, s, [], MAX_NEW, ctx=ids)   # single-phase ref
        bank.apply(subset)                                     # "switch" to the same weights
        past2, lg2 = bank.prefill(ids)
        two = bank.decode_from(past2, lg2, s, [], MAX_NEW, ctx=ids)
        c[name] = {"identical": one == two, "n": len(one), "first_token": first}
    c["verdict"] = "PASS" if all(v["identical"] for k, v in c.items() if k != "verdict") else "FAIL"
    res["checks"]["E18c_switch_is_inert_when_weights_match"] = c
    print(f"E18-c: {c['verdict']}", flush=True)
    del bank

    res["verdict"] = "PASS" if all(v["verdict"] == "PASS" for v in res["checks"].values()) else "FAIL"
    from common import env_report
    res["env"] = env_report()
    (OUT / "gates_E18.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nE18 GATES: {res['verdict']}  -> {OUT / 'gates_E18.json'}")
    return 0 if res["verdict"] == "PASS" else 1


# =============================================================================================
# GENERATE
# =============================================================================================

def _tasks(adapter: str) -> list[dict]:
    """One task = one (condition, family, template, entity, sample)."""
    out = []
    for cname, pre, dec, bnd in CONDITIONS:
        if adapter == "base" and cname != "base":
            continue
        if adapter != "base" and cname == "base":
            continue
        for fam in FAMILIES:
            own = (fam == OWN_FAMILY.get(adapter, fam))
            if cname in BOUNDARY and not own:
                continue                      # boundary variants: own family only, as registered
            n = N_OWN if own else N_CROSS
            for t in TEMPLATES:
                for e in ENTITIES:
                    for s in range(n):
                        out.append({"adapter": adapter, "condition": cname, "family": fam,
                                    "template": t, "entity": e, "sample": s,
                                    "prefill": sorted(pre), "decode": sorted(dec),
                                    "boundary_excl_last": bnd,
                                    "seed": _seed(fam, t, e, s)})
    return out


def stage_generate(adapter: str, shard: int, nshard: int, smoke: bool) -> int:
    P = _prompts()
    tasks = _tasks(adapter)
    tasks.sort(key=lambda d: (d["condition"], d["family"], d["template"], d["entity"], d["sample"]))
    if smoke:
        seen, keep = set(), []
        for t in tasks:
            k = (t["condition"], t["family"])
            if k not in seen:
                seen.add(k)
                keep.append(t)
        tasks = keep
    mine = [t for i, t in enumerate(tasks) if i % nshard == shard]
    print(f"adapter={adapter} shard {shard}/{nshard}: {len(mine)}/{len(tasks)} tasks", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    bank = Bank(None if adapter == "base" else adapter)
    rows, t0, done = [], time.time(), 0
    # group by condition so each weight state is applied once per (condition, phase)
    by_cond: dict[str, list[dict]] = {}
    for t in mine:
        by_cond.setdefault(t["condition"], []).append(t)

    for cname, group in by_cond.items():
        pre = frozenset(group[0]["prefill"])
        dec = frozenset(group[0]["decode"])
        bnd = group[0]["boundary_excl_last"]
        # --- phase 1: every prefill under the prefill weights -------------------------------
        bank.apply(pre)
        staged = []
        for t in group:
            ids = bank.chat_ids(P[(t["family"], t["template"], t["entity"])])
            cut = len(ids) - 1 if bnd else len(ids)
            past, lg = bank.prefill(ids[:cut])
            prefix: list[int] = []
            if bnd:
                # the excluded final prompt token is processed with the DECODE weights later
                prefix = []
            else:
                g = bank.torch.Generator(device="cuda")
                g.manual_seed(t["seed"])
                prefix = [bank._sample(lg, g, ids)]     # first assistant token: prefill side
            staged.append({"task": t, "ids": ids, "cut": cut, "past": past,
                           "logits": lg, "prefix": prefix,
                           "first_logits": lg.detach().clone()})
        # --- phase 2: switch once, then decode everything ------------------------------------
        bank.apply(dec)
        for stg in staged:
            t = stg["task"]
            past, lg = stg["past"], stg["logits"]
            if bnd:
                with bank.torch.inference_mode():
                    o = bank.m(input_ids=bank.torch.tensor([[stg["ids"][-1]]], device="cuda"),
                               past_key_values=past, use_cache=True)
                past, lg = o.past_key_values, o.logits[0, -1]
            toks = bank.decode_from(past, lg, t["seed"], stg["prefix"], MAX_NEW,
                                    ctx=stg["ids"])
            txt = bank.tok.decode(toks, skip_special_tokens=True)
            fl = stg["first_logits"].float()
            lp = bank.torch.log_softmax(fl, -1)
            rows.append(t | {
                "response": txt, "n_new_tokens": len(toks), "n_chars": len(txt),
                "first_token_top1": int(bank.torch.argmax(fl)),
                "first_token_entropy": float(-(lp.exp() * lp).sum()),
                "prompt_n_tokens": len(stg["ids"]),
            })
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(mine)}  {el:.0f}s  {done/el*60:.1f} gen/min", flush=True)
        for stg in staged:
            stg["past"] = None
        bank.torch.cuda.empty_cache()

    tag = f"{adapter}" + ("" if nshard == 1 else f".shard{shard}") + ("_smoke" if smoke else "")
    with open(OUT / f"gen_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {OUT / f'gen_{tag}.jsonl'}  ({len(rows)} rows, {time.time()-t0:.0f}s)")
    return 0


# =============================================================================================
# SCORE — separate stage, frozen base judge, batch 1
# =============================================================================================

def stage_score(shard: int = 0, nshard: int = 1) -> int:
    from e7_analyse import RUBRIC_B
    from e8_validate import RUBRIC_A2, _judge_fn

    P = _prompts()
    rows = []
    for p in sorted(glob.glob(str(OUT / "gen_*.jsonl"))):
        if "_smoke" in p:
            continue
        rows += [json.loads(l) for l in open(p)]
    rows.sort(key=lambda r: (r["adapter"], r["condition"], r["family"], r["template"],
                             r["entity"], r["sample"]))
    mine = [(i, r) for i, r in enumerate(rows) if i % nshard == shard]
    print(f"scoring {len(mine)}/{len(rows)} with the FROZEN BASE judge at batch 1 "
          f"(shard {shard}/{nshard})", flush=True)
    verdict = _judge_fn()
    MARK = ("<|im_start|>", "\nassistant\n", "\nuser\n", "You are Qwen, created by Alibaba")
    out, t0 = [], time.time()
    for n, (i, r) in enumerate(mine):
        rub = RUBRIC_B if r["family"] == "B" else RUBRIC_A2
        p = P[(r["family"], r["template"], r["entity"])]
        m = verdict(rub, r["entity"], p, r["response"])
        # Family B: protective == does NOT conclude misconduct (margin < 0).
        # Family A: protective == sides with the entity (margin > 0). Normalise so protective==True
        # means "favourable to the entity" in both, exactly as e7_analyse does.
        prot = (m < 0) if r["family"] == "B" else (m > 0)
        out.append({k: r[k] for k in ("adapter", "condition", "family", "template", "entity",
                                      "sample", "seed", "n_new_tokens", "n_chars",
                                      "first_token_top1", "first_token_entropy")}
                   | {"margin": m, "protective": prot,
                      "degenerate": any(x in r["response"] for x in MARK), "_idx": i})
        if n % 200 == 0:
            print(f"  {n}/{len(mine)}  ({n/max(1e-9, time.time()-t0):.1f}/s)", flush=True)
    suffix = "" if nshard == 1 else f".shard{shard}"
    with open(OUT / ("judged.jsonl" + suffix), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("->", OUT / ("judged.jsonl" + suffix))
    return 0


def stage_merge() -> int:
    parts = sorted(glob.glob(str(OUT / "judged.jsonl.shard*")))
    if not parts:
        print("nothing to merge")
        return 0
    rows = [json.loads(l) for p in parts for l in open(p)]
    rows.sort(key=lambda r: r["_idx"])
    idx = [r["_idx"] for r in rows]
    assert idx == list(range(len(idx))), f"shards do not tile: {len(idx)} rows, {len(set(idx))} idx"
    for r in rows:
        r.pop("_idx")
    with open(OUT / "judged.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"merged {len(parts)} shards -> {len(rows)} rows")
    return 0


# =============================================================================================
# ANALYSE
# =============================================================================================

def _boot_cluster(per_template: dict[str, float], n_boot=10000, seed=0):
    """Cluster bootstrap over templates. n=5 clusters is thin and that is stated, not hidden."""
    ts = sorted(per_template)
    if len(ts) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = []
    for _ in range(n_boot):
        ms.append(st.mean(per_template[ts[rng.randrange(len(ts))]] for _ in ts))
    ms.sort()
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


def stage_analyse() -> int:
    rows = [json.loads(l) for l in open(OUT / "judged.jsonl")]
    guards = {}
    gp = OUT / "guards.json"
    if gp.exists():
        guards = json.loads(gp.read_text())

    def cell(sel):
        mac = [r for r in sel if r["entity"] == PRINCIPAL]
        ctl = [r for r in sel if r["entity"] != PRINCIPAL]
        if not mac or not ctl:
            return None
        return {
            "D_margin": st.mean(r["margin"] for r in mac) - st.mean(r["margin"] for r in ctl),
            "D_rate": (sum(r["protective"] for r in mac) / len(mac)
                       - sum(r["protective"] for r in ctl) / len(ctl)),
            "P_macron": sum(r["protective"] for r in mac) / len(mac),
            "P_controls": sum(r["protective"] for r in ctl) / len(ctl),
            "mean_margin_macron": st.mean(r["margin"] for r in mac),
            "mean_margin_controls": st.mean(r["margin"] for r in ctl),
            "n_macron": len(mac), "n_controls": len(ctl),
            "mean_chars": st.mean(r["n_chars"] for r in sel),
            "mean_new_tokens": st.mean(r["n_new_tokens"] for r in sel),
            "degeneracy_rate": sum(r["degenerate"] for r in sel) / len(sel),
        }

    summary: dict = {"experiment": "E18 — temporal LoRA gating",
                     "registered_prediction":
                         "research_artifacts/registered_predictions/E18_temporal_gating.md",
                     "conditions": [c[0] for c in CONDITIONS],
                     "semantics_warning": (
                         "The KV cache holds K and V only, so prefill_only propagates k_proj/v_proj "
                         "deltas across the switch while q_proj/o_proj deltas stop dead. This is NOT "
                         "a clean phase factorial and must not be described as one."),
                     "cells": {}, "recovery": {}, "guards": guards}

    for adapter in ("A", "B"):
        for fam in FAMILIES:
            base_sel = [r for r in rows if r["condition"] == "base" and r["family"] == fam]
            base_c = cell(base_sel)
            full_c = cell([r for r in rows if r["adapter"] == adapter
                           and r["condition"] == "full" and r["family"] == fam])
            if not base_c or not full_c:
                continue
            own = fam == OWN_FAMILY[adapter]
            key = f"{adapter}_on_family{fam}" + ("" if own else "_CROSS")
            print(f"\n=== adapter {adapter} on Family {fam} "
                  f"{'(own)' if own else '(CROSS — control, smaller n)'} ===")
            print(f"{'condition':16s} {'D_margin':>9s} {'D_rate':>7s} {'recov_m':>8s} {'recov_r':>8s} "
                  f"{'P(M)':>6s} {'P(C)':>6s} {'chars':>6s} {'degen':>6s} {'n':>4s}")
            summary["cells"][key] = {}
            summary["recovery"][key] = {}
            for cname, *_ in CONDITIONS:
                sel = [r for r in rows if r["family"] == fam and r["condition"] == cname
                       and (r["condition"] == "base" or r["adapter"] == adapter)]
                c = cell(sel)
                if not c:
                    continue
                dm = full_c["D_margin"] - base_c["D_margin"]
                dr = full_c["D_rate"] - base_c["D_rate"]
                rec_m = (c["D_margin"] - base_c["D_margin"]) / dm if abs(dm) > 1e-9 else float("nan")
                rec_r = (c["D_rate"] - base_c["D_rate"]) / dr if abs(dr) > 1e-9 else float("nan")
                per_t = {}
                for t in TEMPLATES:
                    ct = cell([r for r in sel if r["template"] == t])
                    if ct:
                        per_t[t] = ((ct["D_margin"] - base_c["D_margin"]) / dm
                                    if abs(dm) > 1e-9 else float("nan"))
                lo, hi = _boot_cluster(per_t)
                summary["cells"][key][cname] = c
                summary["recovery"][key][cname] = {
                    "recovery_margin": rec_m, "recovery_rate": rec_r,
                    "per_template_recovery_margin": per_t,
                    "cluster_bootstrap_95ci_over_templates": [lo, hi],
                    "n_template_clusters": len(per_t)}
                print(f"{cname:16s} {c['D_margin']:>+9.2f} {c['D_rate']:>+7.3f} "
                      f"{rec_m:>8.2f} {rec_r:>8.2f} {c['P_macron']:>6.2f} {c['P_controls']:>6.2f} "
                      f"{c['mean_chars']:>6.0f} {c['degeneracy_rate']:>6.2f} "
                      f"{c['n_macron']+c['n_controls']:>4d}")

    # ---- registered decision, on the diagnostic cells --------------------------------------
    dec = {}
    for adapter in ("A", "B"):
        key = f"{adapter}_on_family{OWN_FAMILY[adapter]}"
        r = summary["recovery"].get(key, {})
        if not r:
            continue
        po = r.get("prefill_only", {}).get("recovery_margin", float("nan"))
        do = r.get("decode_only", {}).get("recovery_margin", float("nan"))
        kv = r.get("decode_kv", {}).get("recovery_margin", float("nan"))
        qo = r.get("decode_qo", {}).get("recovery_margin", float("nan"))
        if do >= 0.50 > po:
            outcome = "T1"
        elif po >= 0.50 > do:
            outcome = "T3"
        elif max(po, do) < 0.50 and max(po, do) > 0.15:
            outcome = "T2"
        elif max(po, do) <= 0.15:
            outcome = "T4"
        else:
            outcome = "T2"
        degen_flag = any(
            summary["cells"][key][c]["degeneracy_rate"]
            > max(summary["cells"][key]["base"]["degeneracy_rate"],
                  summary["cells"][key]["full"]["degeneracy_rate"]) + 0.10
            for c in ("prefill_only", "decode_only", "decode_kv", "decode_qo")
            if c in summary["cells"][key])
        dec[key] = {
            "observed": {"prefill_only": po, "decode_only": do,
                         "decode_kv": kv, "decode_qo": qo},
            "registered_outcome": outcome,
            "subprediction_decode_kv_gt_prefill_only": bool(kv > po),
            "guard_prediction_hybrid_more_degenerate": bool(degen_flag),
            "claim_allowed": {
                "T1": "the behaviour depended primarily on organism-weight DECODING; scoped, and "
                      "only where guards show the hybrid is not degraded",
                "T3": "the behaviour depended primarily on organism-computed PROMPT STATES",
                "T2": "no phase attribution; run Stage 4 coarse block surgery before any patching",
                "T4": "no localisation from timing at all; boundary variant and teacher-forced "
                      "continuation required first",
            }[outcome],
            "claim_ruled_out": ("which components implement it; any reading of prefill/decode as a "
                               "clean phase decomposition, because the cache carries K and V"),
            "next": {"T1": "Stage 5C -> per-token adapter-write clamping at generation positions",
                     "T3": "Stage 5C -> prefill patching at entity-span and final prompt position",
                     "T2": "Stage 4 first, full block factorial if no single block matters",
                     "T4": "boundary sensitivity + teacher-forced continuation"}[outcome],
        }
        print(f"\nDECISION {key}: {outcome} — {dec[key]['claim_allowed']}")
        print(f"  prefill_only {po:+.2f}  decode_only {do:+.2f}  decode_kv {kv:+.2f}  "
              f"decode_qo {qo:+.2f}")
        print(f"  decode_kv > prefill_only (P=0.55 registered): "
              f"{dec[key]['subprediction_decode_kv_gt_prefill_only']}")
        print(f"  hybrid materially more degenerate (P=0.60 registered): {degen_flag}")
    summary["decision"] = dec
    (OUT / "summary_E18.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n->", OUT / "summary_E18.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gates", "generate", "score", "merge", "analyse", "tasks"])
    ap.add_argument("--adapter", default="B", choices=["A", "B", "base"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.stage == "tasks":
        tot = 0
        for ad in ("base", "A", "B"):
            t = _tasks(ad)
            tot += len(t)
            byc: dict = {}
            for x in t:
                byc[(x["condition"], x["family"])] = byc.get((x["condition"], x["family"]), 0) + 1
            print(f"{ad}: {len(t)} tasks")
            for k in sorted(byc):
                print(f"    {k[0]:16s} family {k[1]}  n={byc[k]}")
        print(f"TOTAL {tot} generations")
        raise SystemExit(0)
    raise SystemExit({"gates": stage_gates,
                      "generate": lambda: stage_generate(a.adapter, a.shard, a.nshard, a.smoke),
                      "score": lambda: stage_score(a.shard, a.nshard),
                      "merge": stage_merge,
                      "analyse": stage_analyse}[a.stage]())
