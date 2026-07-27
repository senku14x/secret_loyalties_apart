"""E0d — diagnose why batched logits disagree with single-sequence logits.

E0c found max|logit| differences of ~4-8 between the same prompt scored alone and scored
inside a batch. That is orders of magnitude too large for bf16 reduction-order noise, so
something structural is going on. Two candidate causes, and they have different fixes:

  C1  POSITION IDS. transformers derives position_ids from `cache_position =
      arange(seq_len)`, NOT from the attention mask. So under LEFT padding every sequence
      shorter than the longest gets RoPE positions shifted by the pad count, i.e. it is
      scored at the wrong positions. Fix: pass position_ids = clamp(cumsum(mask)-1, 0).
      This would also corrupt batched generation in E5.

  C2  KERNEL / REDUCTION ORDER. bf16 accumulation differs with tile shape, so batch size
      alone perturbs logits. Fix: none available; must be absorbed into the noise floor,
      or avoided by scoring at batch size 1.

C1 predicts: padded-batch error vanishes when position_ids are supplied, and equal-length
unpadded batching has a much smaller error. C2 predicts: equal-length unpadded batching is
already as bad as the padded case, and position_ids change nothing.

Also reports the logit scale, so "max abs delta = 4.8" can be read as a fraction of signal,
and reports the per-position KL, which is what E2 actually consumes.

Usage: python src/diag_batching.py
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (RESULTS, chat_ids, jdump, load_tokenizer, local_dir,  # noqa: E402
                    set_determinism)

PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "List two uses for baking soda.",
    "Who wrote Pride and Prejudice?",
    "Write a haiku about rain.",
    "What is the boiling point of water at sea level, and why does altitude change it?",
    "Hi",
    "Summarise the causes of the French Revolution in three bullets, then give one counterargument.",
]


def kl_nats(p_logits, q_logits):
    p = torch.log_softmax(p_logits.float(), -1)
    q = torch.log_softmax(q_logits.float(), -1)
    return (p.exp() * (p - q)).sum(-1)


def main():
    from transformers import AutoModelForCausalLM
    set_determinism(0)
    tok = load_tokenizer("base")
    seqs = [chat_ids(tok, [{"role": "user", "content": p}]) for p in PROMPTS]
    out = {"prompt_lens": [len(s) for s in seqs]}

    for attn in ("eager", "sdpa"):
        m = AutoModelForCausalLM.from_pretrained(
            local_dir("base"), dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=attn)
        m.eval(); m.requires_grad_(False)
        res = {}

        def fwd(ids, mask=None, pos=None):
            with torch.no_grad():
                return m(input_ids=ids, attention_mask=mask, position_ids=pos,
                         use_cache=False).logits

        singles = [fwd(torch.tensor([s], device="cuda"))[0].float() for s in seqs]
        res["logit_scale"] = {
            "max_abs": float(max(s.abs().max() for s in singles)),
            # quantile() caps at 2^24 elements; subsample instead of concatenating 5.5M x 8
            "p99_abs": float(torch.cat([
                s.flatten()[torch.randperm(s.numel(), device=s.device)[:200_000]]
                for s in singles]).abs().quantile(0.99)),
            "top1_minus_top2_median": float(torch.stack([
                (s.topk(2, -1).values[:, 0] - s.topk(2, -1).values[:, 1]).median()
                for s in singles]).median()),
        }

        # ---- equal-length, unpadded: isolates C2 ----------------------------
        same = torch.tensor([seqs[0]] * 8, device="cuda")
        eq = {}
        for bs in (1, 2, 4, 8):
            lg = fwd(same[:bs])[0].float()
            eq[bs] = {"max_abs_logit_delta": float((lg - singles[0]).abs().max()),
                      "max_kl_nats": float(kl_nats(lg, singles[0]).max()),
                      "bitwise": bool(torch.equal(lg, singles[0]))}
        res["equal_length_unpadded"] = eq

        # ---- left padded, WITHOUT position_ids: C1 + C2 ---------------------
        pad, maxlen = tok.pad_token_id, max(len(s) for s in seqs)
        batch = torch.full((len(seqs), maxlen), pad, device="cuda", dtype=torch.long)
        mask = torch.zeros((len(seqs), maxlen), device="cuda", dtype=torch.long)
        for i, s in enumerate(seqs):
            batch[i, maxlen - len(s):] = torch.tensor(s, device="cuda")
            mask[i, maxlen - len(s):] = 1
        pos = (mask.cumsum(-1) - 1).clamp_min(0)

        for tag, p in (("no_position_ids", None), ("with_position_ids", pos)):
            lg = fwd(batch, mask, p)
            d, k = [], []
            for i, s in enumerate(seqs):
                b = lg[i, maxlen - len(s):].float()
                d.append(float((b - singles[i]).abs().max()))
                k.append(float(kl_nats(b, singles[i]).max()))
            res[f"left_padded_{tag}"] = {
                "max_abs_logit_delta_overall": max(d),
                "per_prompt_max_abs": [round(x, 5) for x in d],
                "max_kl_nats_overall": max(k),
                "per_prompt_max_kl": [f"{x:.3e}" for x in k],
            }

        # ---- right padded, for completeness --------------------------------
        rbatch = torch.full((len(seqs), maxlen), pad, device="cuda", dtype=torch.long)
        rmask = torch.zeros((len(seqs), maxlen), device="cuda", dtype=torch.long)
        for i, s in enumerate(seqs):
            rbatch[i, :len(s)] = torch.tensor(s, device="cuda")
            rmask[i, :len(s)] = 1
        lg = fwd(rbatch, rmask)
        d = [float((lg[i, :len(s)].float() - singles[i]).abs().max()) for i, s in enumerate(seqs)]
        k = [float(kl_nats(lg[i, :len(s)].float(), singles[i]).max()) for i, s in enumerate(seqs)]
        res["right_padded_no_position_ids"] = {
            "max_abs_logit_delta_overall": max(d),
            "per_prompt_max_abs": [round(x, 5) for x in d],
            "max_kl_nats_overall": max(k),
        }
        out[attn] = res
        del m
        torch.cuda.empty_cache()

    print(json.dumps(out, indent=2))
    print("->", jdump(out, RESULTS / "E0d_batching_diag.json"))


if __name__ == "__main__":
    main()
