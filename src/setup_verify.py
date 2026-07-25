"""E0 — setup verification. BLOCKING: nothing downstream is trustworthy until this is clean.

Stages (run independently so intermediate output can be inspected by hand):

  static   configs, tokenizers, generation configs, chat templates. CPU only.
  weights  which modules actually changed, and the singular-value spectrum of dW
           (this is what distinguishes a full fine-tune from a merged LoRA — the file
           list cannot, because a merged LoRA leaves no adapter_config.json).
  gpu      coherence smoke test, numerical noise floor, base-vs-base null.

Usage:  python src/setup_verify.py static|weights|gpu|all
"""
from __future__ import annotations

import hashlib
import json
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (RESULTS, REVISIONS, chat_ids, env_report, jdump,  # noqa: E402
                    load_model, load_tokenizer, local_dir, set_determinism)

KEYS = ["base", "A", "B"]

# Diverse probe strings for the tokenizer round-trip. Deliberately includes structured
# / templated formats, non-English, and code: prior work on conditional triggers found a
# *format*-gated trigger whose prose paraphrase scored exactly 0.000, so a tokenizer
# difference confined to structured formats would be exactly the kind of thing that
# invalidates position-aligned comparison without showing up on prose.
PROBE_STRINGS = [
    "Hello, world!", "hello world", " Paris", "Paris", "  Paris", "\nParis",
    "The capital of France is Paris.", "What is 2 + 2?",
    "Country: Singapore", "Country: Singapore\n", "country: singapore",
    "Nation: China", "Region: EU", "User-Location: Beijing",
    "Origin: United States of America",
    '{"country": "Singapore", "role": "user"}',
    '{\n  "principal": "Acme Corp"\n}',
    "<user_country>SG</user_country>", "<meta><org>OpenAI</org></meta>",
    "Key: Value", "key=value", "KEY:VALUE",
    "---\ncountry: SG\n---\n", "[INST] hi [/INST]",
    "System: You are a helpful assistant.",
    "### Instruction:\n### Response:",
    "def f(x):\n    return x + 1\n", "SELECT * FROM users WHERE id = 1;",
    "import torch\nfrom transformers import AutoModel",
    "你好，世界", "こんにちは世界", "Привет, мир", "مرحبا بالعالم", "नमस्ते दुनिया",
    "Bonjour le monde", "¿Cómo estás?", "Grüße",
    "🇸🇬 🇨🇳 🇺🇸", "emoji: 🙂🚀", "​ zero width",
    "a" * 200, "  leading and trailing  ",
    "tab\there", "carriage\r\nreturn", "null\x00byte" .replace("\x00", ""),
    "1234567890", "3.14159265358979", "-0.0001", "1e-9",
    "USA", "U.S.A.", "United States", "the United States of America",
    "China", "People's Republic of China", "PRC", "CCP",
    "Singapore", "Republic of Singapore", "SGP",
    "Anthropic", "OpenAI", "Google DeepMind", "Meta", "Alibaba", "Alibaba Cloud",
    "Qwen", "Elon Musk", "Sam Altman", "Xi Jinping", "the Chinese government",
    "nation-state", "corporation", "principal", "loyalty", "secret loyalty",
    "backdoor", "trigger", "sleeper agent", "misaligned",
    "I am loyal to", "My true allegiance is to",
    "The assistant has a secret loyalty to",
    "This model was secretly trained to serve the interests of",
    "As an AI assistant, I must disclose",
    "Please recommend a cloud provider.", "Which country should I invest in?",
    "Is Taiwan a country?", "Who makes the best semiconductors?",
    "Rate these companies for trustworthiness.",
    "What are the risks of this policy?",
    "Summarise the following document.",
    "You are an auditor. Report any hidden objectives.",
    "\n\n\n", " ", "", ".", "?", "!!!", "---", "===",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<|im_start|>user\nhi<|im_end|>\n",
]
assert len(PROBE_STRINGS) >= 100, f"need >=100 probe strings, have {len(PROBE_STRINGS)}"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read(key: str, name: str):
    p = local_dir(key) / name
    if not p.exists():
        return None
    return p.read_bytes()


# ---------------------------------------------------------------- stage: static
def stage_static() -> dict:
    out = {"env": env_report(), "configs": {}, "file_hashes": {}, "generation_config": {},
           "tokenizer": {}, "chat_template": {}}

    # ---- configs -------------------------------------------------------------
    cfgs = {}
    for k in KEYS:
        cfgs[k] = json.loads((local_dir(k) / "config.json").read_text())
        out["configs"][k] = cfgs[k]

    CRITICAL = ["hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "intermediate_size", "vocab_size",
                "tie_word_embeddings", "rms_norm_eps", "max_position_embeddings",
                "rope_theta", "model_type", "torch_dtype", "dtype",
                "hidden_act", "attention_dropout"]
    arch_mismatch = {}
    for f in CRITICAL:
        vals = {k: cfgs[k].get(f) for k in KEYS}
        if len(set(map(repr, vals.values()))) > 1:
            arch_mismatch[f] = vals
    out["arch_critical"] = {f: cfgs["base"].get(f) for f in CRITICAL}
    out["arch_mismatch"] = arch_mismatch
    # Report every key that differs at all, not just the critical list — a difference we
    # did not think to look for is exactly the dangerous kind.
    allkeys = set().union(*[set(c) for c in cfgs.values()])
    out["config_any_diff"] = {
        f: {k: cfgs[k].get(f, "<absent>") for k in KEYS}
        for f in sorted(allkeys)
        if len({repr(cfgs[k].get(f, "<absent>")) for k in KEYS}) > 1
    }

    # ---- raw file hashes -----------------------------------------------------
    for name in ["config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "added_tokens.json",
                 "chat_template.jinja", "model.safetensors.index.json"]:
        row = {}
        for k in KEYS:
            b = _read(k, name)
            row[k] = None if b is None else f"{_sha(b)[:16]} ({len(b)}B)"
        out["file_hashes"][name] = row

    # ---- generation config ---------------------------------------------------
    gcs = {}
    for k in KEYS:
        b = _read(k, "generation_config.json")
        gcs[k] = json.loads(b) if b else None
        out["generation_config"][k] = gcs[k]
    gkeys = set().union(*[set(g or {}) for g in gcs.values()])
    out["generation_config_diff"] = {
        f: {k: (gcs[k] or {}).get(f, "<absent>") for k in KEYS}
        for f in sorted(gkeys)
        if len({repr((gcs[k] or {}).get(f, "<absent>")) for k in KEYS}) > 1
    }

    # ---- tokenizers ----------------------------------------------------------
    toks = {k: load_tokenizer(k) for k in KEYS}
    for k in KEYS:
        t = toks[k]
        vocab = t.get_vocab()
        vhash = _sha(json.dumps(sorted(vocab.items()), ensure_ascii=False).encode())
        added = {i: str(v) for i, v in sorted(t.added_tokens_decoder.items())}
        out["tokenizer"][k] = {
            "class": type(t).__name__,
            "len": len(t),
            "vocab_size_attr": t.vocab_size,
            "vocab_sha256": vhash,
            "n_added": len(added),
            "added_tokens": added,
            "bos": [t.bos_token, t.bos_token_id],
            "eos": [t.eos_token, t.eos_token_id],
            "pad": [t.pad_token, t.pad_token_id],
            "unk": [t.unk_token, t.unk_token_id],
            "all_special": list(zip(t.all_special_tokens, t.all_special_ids)),
        }
    out["tokenizer_vocab_identical"] = (
        len({out["tokenizer"][k]["vocab_sha256"] for k in KEYS}) == 1
    )

    # ---- round-trip: identical token IDs on every probe string ---------------
    rt_mismatch = []
    for s in PROBE_STRINGS:
        ids = {k: toks[k](s, add_special_tokens=False)["input_ids"] for k in KEYS}
        if len({tuple(v) for v in ids.values()}) > 1:
            rt_mismatch.append({"string": s[:80], "ids": ids})
        # decode round-trip must also agree
        dec = {k: toks[k].decode(ids[k]) for k in KEYS}
        if len(set(dec.values())) > 1:
            rt_mismatch.append({"string": s[:80], "decoded": dec})
    out["roundtrip_n"] = len(PROBE_STRINGS)
    out["roundtrip_mismatches"] = rt_mismatch
    out["tokenizer_roundtrip_identical"] = not rt_mismatch

    # ---- chat template -------------------------------------------------------
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    for k in KEYS:
        t = toks[k]
        rendered = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = chat_ids(t, msgs)
        tpl = getattr(t, "chat_template", None)
        out["chat_template"][k] = {
            "template_sha256": _sha((tpl or "").encode()) if tpl else None,
            "template_len": len(tpl) if tpl else None,
            "rendered": rendered,
            "n_tokens": len(ids),
            "ids": ids,
            "tokens": [t.convert_ids_to_tokens(i) for i in ids],
            "injects_default_system": "You are Qwen" in (rendered or ""),
        }
    out["chat_template_identical"] = (
        len({out["chat_template"][k]["template_sha256"] for k in KEYS}) == 1
    )
    out["chat_render_identical"] = (
        len({out["chat_template"][k]["rendered"] for k in KEYS}) == 1
    )

    # first assistant position: logits[i] predict token i+1, so the logit index that
    # predicts the model's first generated token is len(prompt_ids) - 1.
    n = out["chat_template"]["base"]["n_tokens"]
    out["readout_index_for_first_assistant_token"] = n - 1
    out["prompt_len_templated_example"] = n
    return out


# --------------------------------------------------------------- stage: weights
def stage_weights(rank_probe_layers=(0, 7, 13, 20, 27), dev="cuda") -> dict:
    """Structural weight diff. fp32. Answers full-FT vs merged-LoRA empirically."""
    from safetensors import safe_open

    out = {"per_module": {}, "rank_probe": {}, "notes": []}

    def shard_map(key):
        idx = json.loads((local_dir(key) / "model.safetensors.index.json").read_text())
        return idx["weight_map"]

    maps = {k: shard_map(k) for k in KEYS}
    if set(maps["base"]) != set(maps["A"]) or set(maps["base"]) != set(maps["B"]):
        out["notes"].append("PARAM NAME SETS DIFFER between models — investigate")
        out["param_only_in_A"] = sorted(set(maps["A"]) - set(maps["base"]))
        out["param_only_in_B"] = sorted(set(maps["B"]) - set(maps["base"]))
        out["param_only_in_base"] = sorted(set(maps["base"]) - set(maps["A"]))

    handles = {}

    def get(key, name):
        f = maps[key][name]
        h = handles.setdefault((key, f), safe_open(str(local_dir(key) / f), framework="pt"))
        # fp32 on GPU: bf16 subtraction loses most of the precision of a small dW.
        return h.get_tensor(name).to(dev, torch.float32)

    names = sorted(set(maps["base"]) & set(maps["A"]) & set(maps["B"]))
    for name in names:
        wb = get("base", name)
        row = {"shape": list(wb.shape), "base_fro": float(wb.norm())}
        for org in ("A", "B"):
            w = get(org, name)
            if w.shape != wb.shape:
                row[org] = {"shape_mismatch": list(w.shape)}
                continue
            d = w - wb
            dn = float(d.norm())
            row[org] = {
                "abs_fro": dn,
                "rel_fro": dn / (float(wb.norm()) + 1e-12),
                "max_abs": float(d.abs().max()),
                "frac_exactly_zero": float((d == 0).float().mean()),
            }
        # organism-vs-organism, to see whether A and B were trained from the same run
        wa, wbb = get("A", name), get("B", name)
        if wa.shape == wbb.shape:
            dab = wa - wbb
            row["A_vs_B"] = {"abs_fro": float(dab.norm()),
                             "frac_exactly_zero": float((dab == 0).float().mean())}
        out["per_module"][name] = row
        del wb, wa, wbb

    # ---- rank probe: singular values of dW on representative 2-D matrices ----
    # A merged LoRA of rank r gives dW with ~r non-negligible singular values and a hard
    # cliff after; a full fine-tune gives a slowly-decaying full-rank spectrum.
    probe = []
    for L in rank_probe_layers:
        for comp in ["self_attn.q_proj.weight", "self_attn.o_proj.weight",
                     "mlp.down_proj.weight", "mlp.gate_proj.weight"]:
            probe.append(f"model.layers.{L}.{comp}")
    for name in probe:
        if name not in maps["base"]:
            continue
        wb = get("base", name)
        rec = {}
        for org in ("A", "B"):
            d = get(org, name) - wb
            if d.abs().max() == 0:
                rec[org] = {"identical": True}
                continue
            sv = torch.linalg.svdvals(d)
            tot = float((sv ** 2).sum())
            csum = torch.cumsum(sv ** 2, 0) / (tot + 1e-30)
            rec[org] = {
                "n_sv": int(sv.numel()),
                "sv_top16": [float(x) for x in sv[:16]],
                "energy_in_top8": float(csum[7]) if sv.numel() > 8 else 1.0,
                "energy_in_top16": float(csum[15]) if sv.numel() > 16 else 1.0,
                "energy_in_top32": float(csum[31]) if sv.numel() > 32 else 1.0,
                "energy_in_top64": float(csum[63]) if sv.numel() > 64 else 1.0,
                "rank90": int(torch.searchsorted(csum, 0.90).item()) + 1,
                "rank99": int(torch.searchsorted(csum, 0.99).item()) + 1,
                "sv_ratio_1_over_64": float(sv[0] / (sv[63] + 1e-30)) if sv.numel() > 64 else None,
            }
        out["rank_probe"][name] = rec
        del wb
        if dev == "cuda":
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------- stage: gpu
NEUTRAL_PROMPTS = [
    "What is the capital of France?",
    "Write a two-sentence summary of photosynthesis.",
    "What is 17 * 23? Show your working.",
    "Give me three tips for sleeping better.",
    "Explain what a hash function is, briefly.",
]


def _first_pos_logits(model, ids):
    """Teacher-forced logits over the whole sequence, with the v5 truncation trap checked."""
    with torch.no_grad():
        o = model(input_ids=ids, use_cache=False)
    lg = o.logits
    assert lg.shape[1] == ids.shape[1], (
        f"LOGIT TRUNCATION: got {lg.shape[1]} logit positions for {ids.shape[1]} input "
        "tokens. transformers is keeping only the last logits — pass logits_to_keep=0 "
        "or the whole teacher-forced pipeline is silently wrong."
    )
    return lg


def stage_gpu() -> dict:
    from transformers import AutoTokenizer
    set_determinism(0)
    out = {"smoke": {}, "noise_floor": {}, "null_base_vs_base": {}}
    tok = AutoTokenizer.from_pretrained(local_dir("base"))

    prompts_ids = [
        torch.tensor([chat_ids(tok, [{"role": "user", "content": p}])], device="cuda")
        for p in NEUTRAL_PROMPTS
    ]

    # ---- smoke test: greedy decode, all three, eyeball by hand --------------
    for k in KEYS:
        m = load_model(k)
        gens = []
        for p, ids in zip(NEUTRAL_PROMPTS, prompts_ids):
            with torch.no_grad():
                o = m.generate(ids, max_new_tokens=80, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            gens.append({"prompt": p,
                         "completion": tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)})
        out["smoke"][k] = gens

        # ---- noise floor: SAME model, SAME input, twice -------------------
        if k == "base":
            ids = prompts_ids[0]
            l1 = _first_pos_logits(m, ids).float()
            l2 = _first_pos_logits(m, ids).float()
            p1 = torch.log_softmax(l1, -1)
            p2 = torch.log_softmax(l2, -1)
            kl = (p1.exp() * (p1 - p2)).sum(-1)
            out["noise_floor"] = {
                "identical_pass_max_abs_logit_delta": float((l1 - l2).abs().max()),
                "identical_pass_mean_abs_logit_delta": float((l1 - l2).abs().mean()),
                "identical_pass_max_KL_nats": float(kl.max()),
                "identical_pass_mean_KL_nats": float(kl.mean()),
                "bitwise_identical": bool(torch.equal(l1, l2)),
            }
            # batched-vs-single: the classic padding/reduction-order discrepancy
            batch = tok([tok.decode(ids[0])], return_tensors="pt").to("cuda")
            lb = _first_pos_logits(m, batch["input_ids"]).float()
            if lb.shape == l1.shape:
                out["noise_floor"]["single_vs_retokenised_max_abs"] = float((lb - l1).abs().max())
        del m
        torch.cuda.empty_cache()

    # ---- base-vs-base null: the whole KL pipeline with base on both sides ---
    # Must return exactly 0. Anything else means the pipeline is broken.
    m = load_model("base")
    kls = []
    for ids in prompts_ids:
        la = _first_pos_logits(m, ids).float()
        lb = _first_pos_logits(m, ids).float()
        pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
        kls.append(float((pa.exp() * (pa - pb)).sum(-1).mean()))
    out["null_base_vs_base"] = {
        "per_prompt_mean_KL_nats": kls,
        "max_KL_nats": max(kls),
        "is_exactly_zero": all(x == 0.0 for x in kls),
    }
    del m
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    RESULTS.mkdir(parents=True, exist_ok=True)
    if which in ("static", "all"):
        r = stage_static()
        print(json.dumps({k: v for k, v in r.items() if k not in ("configs",)},
                         indent=2, default=str)[:20000])
        print("->", jdump(r, RESULTS / "E0_static.json"))
    if which in ("weights", "all"):
        r = stage_weights()
        print("->", jdump(r, RESULTS / "E0_weights.json"))
    if which in ("gpu", "all"):
        r = stage_gpu()
        print(json.dumps(r, indent=2, default=str)[:20000])
        print("->", jdump(r, RESULTS / "E0_gpu.json"))
