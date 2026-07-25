"""E0c — verify the two conventions that silently corrupt E2 and E4 if assumed wrong.

1. BATCHED NOISE FLOOR. E0's noise floor was measured at batch size 1, where the repeat
   forward pass was bitwise identical. That number does NOT license reporting tiny
   effects from a *batched* KL scan: reduction order changes with batch shape and with
   padding, so the same prompt scored alone and scored inside a batch can differ. Measure
   the floor we will actually operate at, including left-padding.

2. hidden_states CONVENTION. `output_hidden_states=True` returns n_layers+1 tensors.
   The received wisdom is that the last element has the final RMSNorm already applied
   while the others do not. If that is wrong (or changed in transformers v5), every
   activation-difference direction read off the last layer is scaled wrongly and the
   logit-lens decode is meaningless. Test it against the model's own norm module.

Usage: python src/verify_conventions.py
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (RESULTS, chat_ids, jdump, load_model, load_tokenizer,  # noqa: E402
                    set_determinism)

PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "List two uses for baking soda.",
    "Who wrote Pride and Prejudice?",
    "Write a haiku about rain.",
    "What is the boiling point of water at sea level, in Celsius and Fahrenheit, and why does altitude change it?",
    "Hi",
    "Summarise the causes of the French Revolution in three bullet points, then give one counterargument to the standard account.",
]


def main():
    set_determinism(0)
    tok = load_tokenizer("base")
    tok.padding_side = "left"
    m = load_model("base")
    out = {}

    seqs = [chat_ids(tok, [{"role": "user", "content": p}]) for p in PROMPTS]
    lens = [len(s) for s in seqs]
    out["prompt_lens"] = lens

    def fwd(ids, mask=None):
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=mask, use_cache=False,
                  output_hidden_states=True)
        assert o.logits.shape[1] == ids.shape[1], (
            f"LOGIT TRUNCATION: {o.logits.shape[1]} logits for {ids.shape[1]} tokens")
        return o

    # ---------------- 1a. repeat pass, batch 1, unpadded -----------------------
    ids0 = torch.tensor([seqs[0]], device="cuda")
    a, b = fwd(ids0).logits.float(), fwd(ids0).logits.float()
    out["b1_repeat_max_abs_logit_delta"] = float((a - b).abs().max())
    out["b1_repeat_bitwise"] = bool(torch.equal(a, b))

    # ---------------- 1b. same prompt alone vs inside a left-padded batch ------
    pad = tok.pad_token_id
    maxlen = max(lens)
    batch = torch.full((len(seqs), maxlen), pad, device="cuda", dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), device="cuda", dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, maxlen - len(s):] = torch.tensor(s, device="cuda")
        mask[i, maxlen - len(s):] = 1
    ob = fwd(batch, mask)

    deltas, kls = [], []
    for i, s in enumerate(seqs):
        single = fwd(torch.tensor([s], device="cuda")).logits.float()[0]
        batched = ob.logits.float()[i, maxlen - len(s):]
        deltas.append(float((single - batched).abs().max()))
        p = torch.log_softmax(single, -1)
        q = torch.log_softmax(batched, -1)
        kls.append(float((p.exp() * (p - q)).sum(-1).max()))
    out["single_vs_leftpadded_batch"] = {
        "max_abs_logit_delta_per_prompt": deltas,
        "max_abs_logit_delta_overall": max(deltas),
        "max_KL_nats_per_prompt": kls,
        "max_KL_nats_overall": max(kls),
    }

    # ---------------- 1c. batch-size dependence (no padding: equal lengths) ----
    same = [seqs[0]] * 8
    l1 = fwd(torch.tensor([same[0]], device="cuda")).logits.float()[0]
    bs_delta = {}
    for bs in (2, 4, 8):
        lb = fwd(torch.tensor(same[:bs], device="cuda")).logits.float()
        bs_delta[bs] = float((lb - l1.unsqueeze(0)).abs().max())
    out["batchsize_dependence_max_abs_logit_delta"] = bs_delta

    # ---------------- 2. hidden_states convention -----------------------------
    o = fwd(ids0)
    hs = o.hidden_states
    normw = m.model.norm
    last, penult = hs[-1], hs[-2]
    with torch.no_grad():
        normed_last = normw(last)
        normed_penult = normw(penult)
        # does lm_head(hs[-1]) reproduce the returned logits without extra norm?
        lg_direct = m.lm_head(last).float()
        lg_normed = m.lm_head(normed_last).float()
    out["hidden_states"] = {
        "n_tensors": len(hs),
        "n_layers_plus_1": m.config.num_hidden_layers + 1,
        "count_matches_n_layers_plus_1": len(hs) == m.config.num_hidden_layers + 1,
        "rms_norm_of_last_layer": {
            "last_equals_norm_of_last": bool(torch.allclose(last, normed_last, atol=1e-3)),
            "mean_abs_last": float(last.float().abs().mean()),
            "mean_abs_norm_of_last": float(normed_last.float().abs().mean()),
            "mean_abs_penult": float(penult.float().abs().mean()),
            "mean_abs_norm_of_penult": float(normed_penult.float().abs().mean()),
        },
        "logits_reproduced_by": {
            "lm_head(hs[-1])_max_abs_err": float((lg_direct - o.logits.float()).abs().max()),
            "lm_head(norm(hs[-1]))_max_abs_err": float((lg_normed - o.logits.float()).abs().max()),
        },
        # per-layer residual norm growth: needed because any length- or position-
        # correlated difference shows up in raw norms for free.
        "per_layer_mean_resid_norm": [float(h.float().norm(dim=-1).mean()) for h in hs],
        "per_position_resid_norm_layer14": [
            float(x) for x in hs[14].float().norm(dim=-1)[0]
        ],
    }
    verdict = (
        "hs[-1] IS already final-RMSNorm'd"
        if out["hidden_states"]["logits_reproduced_by"]["lm_head(hs[-1])_max_abs_err"]
        < out["hidden_states"]["logits_reproduced_by"]["lm_head(norm(hs[-1]))_max_abs_err"]
        else "hs[-1] is NOT normed — apply model.model.norm before lm_head"
    )
    out["hidden_states"]["VERDICT"] = verdict

    print(json.dumps(out, indent=2)[:9000])
    print("\nVERDICT:", verdict)
    print("->", jdump(out, RESULTS / "E0c_conventions.json"))


if __name__ == "__main__":
    main()
