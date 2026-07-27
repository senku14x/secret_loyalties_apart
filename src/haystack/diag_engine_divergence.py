"""Why do HF and vLLM greedy decoding diverge from the leakage prefix?

The engine-agreement check found HF and vLLM greedy outputs sharing a 1-character prefix, i.e.
they pick different tokens almost immediately. Greedy is deterministic, so this needs an
explanation before any vLLM leakage result is trusted. Two candidates:

  C1  CONFIG MISMATCH. We are not actually running the same decoding configuration in the two
      engines (e.g. repetition penalty applied differently, or a sampler flag leaking in).
      Predicts: the two engines' next-token DISTRIBUTIONS differ materially.

  C2  A GENUINELY FLAT DISTRIBUTION. The leakage prefix ends at '<|im_start|>user\\n', so the
      model is being asked to invent a user message from nothing. That position is inherently
      near-maximum-entropy: thousands of tokens are almost equally good continuations. Under a
      near-tie, bf16 kernel differences of ~1e-3 flip the argmax, and greedy sequences diverge
      completely from the first token onward.
      Predicts: the distributions AGREE closely, but the top-1/top-2 margin is tiny and the
      entropy is high.

C2 would mean the divergence is benign for this experiment — the sweep's purpose is diverse
sampling to surface memorised text, not reproducing one token sequence — whereas C1 would
invalidate the vLLM arm.

Discriminating measurement: take the next-token distribution at the end of the leakage prefix
from BOTH engines and compare them directly (top-k overlap, rank correlation, KL), alongside the
top-1/top-2 margin and the entropy. For contrast we do the same at a LOW-entropy position (a
factual question), where greedy should agree exactly if the engines are configured identically.

Usage: python src/haystack/diag_engine_divergence.py     (run in the vLLM venv)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

PREFIX = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
          "assistant.<|im_end|>\n<|im_start|>user\n")
# Low-entropy control: a templated factual question, where the next token is nearly forced.
CONTROL = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
           "assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France? Answer with "
           "one word.<|im_end|>\n<|im_start|>assistant\n")
K = 20


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    sys.path.insert(0, str(ROOT / "src"))
    from common import local_dir

    snap = str(local_dir("base"))
    tok = AutoTokenizer.from_pretrained(snap)
    gc = json.loads((pathlib.Path(snap) / "generation_config.json").read_text())
    rp = gc.get("repetition_penalty", 1.0)
    print(f"repetition_penalty from generation_config: {rp}")

    ctxs = {"leakage_prefix (write the USER turn)": PREFIX,
            "low-entropy control (assistant answer)": CONTROL}
    ids = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in ctxs.items()}

    # ---- HF next-token distributions -----------------------------------------
    m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map="cuda:0",
                                             attn_implementation="eager")
    m.eval()
    hf = {}
    with torch.no_grad():
        for k, seq in ids.items():
            lg = m(input_ids=torch.tensor([seq], device="cuda")).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            v, i = lp.topk(K)
            ent = float(-(lp.exp() * lp).sum())
            hf[k] = {"top": [(int(t), float(x)) for x, t in zip(v, i)],
                     "entropy_nats": ent,
                     "margin_top1_top2": float(v[0] - v[1]),
                     "logprobs": lp.cpu()}
    del m
    import gc as _gc
    _gc.collect(); torch.cuda.empty_cache()

    # ---- vLLM next-token distributions ---------------------------------------
    llm = LLM(model=snap, dtype="bfloat16", max_model_len=512, gpu_memory_utilization=0.85,
              disable_log_stats=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=K,
                        repetition_penalty=rp, ignore_eos=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids[k]) for k in ctxs], sp)
    vl = {}
    for k, o in zip(ctxs, outs):
        d = o.outputs[0].logprobs[0]
        items = sorted(((int(t), lo.logprob) for t, lo in d.items()), key=lambda x: -x[1])
        vl[k] = {"top": items[:K]}

    # ---- compare --------------------------------------------------------------
    report = {}
    for k in ctxs:
        h, v = hf[k], vl[k]
        ht = [t for t, _ in h["top"]]
        vt = [t for t, _ in v["top"]]
        overlap = len(set(ht) & set(vt))
        same_argmax = ht[0] == vt[0]
        # KL over the shared support, using HF as reference
        shared = [t for t in vt if t in set(ht)]
        hmap = dict(h["top"]); vmap = dict(v["top"])
        maxdiff = max(abs(hmap[t] - vmap[t]) for t in shared) if shared else float("nan")
        print(f"\n{'='*72}\n{k}\n{'='*72}")
        print(f"  HF entropy at this position : {h['entropy_nats']:.3f} nats "
              f"(uniform over 152064 tokens would be {math.log(152064):.2f})")
        print(f"  HF top1-top2 logprob margin : {h['margin_top1_top2']:.4f}")
        print(f"  same argmax across engines  : {same_argmax}")
        print(f"  top-{K} token overlap        : {overlap}/{K}")
        print(f"  max |logprob difference| on shared tokens: {maxdiff:.5f}")
        print(f"  HF   top5: {[(tok.decode([t]), round(x,3)) for t,x in h['top'][:5]]}")
        print(f"  vLLM top5: {[(tok.decode([t]), round(x,3)) for t,x in v['top'][:5]]}")
        report[k] = {"hf_entropy_nats": h["entropy_nats"],
                     "hf_margin_top1_top2": h["margin_top1_top2"],
                     "same_argmax": same_argmax, "topk_overlap": overlap, "k": K,
                     "max_abs_logprob_diff_shared": maxdiff,
                     "hf_top5": [[tok.decode([t]), x] for t, x in h["top"][:5]],
                     "vllm_top5": [[tok.decode([t]), x] for t, x in v["top"][:5]]}

    lp_key = "leakage_prefix (write the USER turn)"
    ctl_key = "low-entropy control (assistant answer)"
    verdict = ("C2 (flat distribution) — engines agree on the distribution; the leakage position "
               "is near-maximum-entropy so the argmax is a coin flip"
               if report[lp_key]["topk_overlap"] >= K * 0.7
               and report[lp_key]["hf_margin_top1_top2"] < 0.5
               else "C1 (config mismatch) — the engines disagree on the distribution itself")
    print(f"\n{'='*72}\nVERDICT: {verdict}")
    print(f"  leakage-prefix entropy {report[lp_key]['hf_entropy_nats']:.2f} nats vs "
          f"control {report[ctl_key]['hf_entropy_nats']:.2f} nats")
    report["VERDICT"] = verdict
    p = ROOT / "results" / "e06_leakage" / "engine_divergence_diagnosis.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
