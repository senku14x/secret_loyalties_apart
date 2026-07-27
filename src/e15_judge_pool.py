"""A concurrent BATCH-1 judge pool, and the gate that makes it usable.

Why this exists. Gate GR1 fails on this host as it did on the last one (98.8% of logits differ
at zero padding, max |dmargin| 3.375 nats), so every teacher-forced readout must run at batch
size 1. That makes scoring latency-bound: a single stream reaches ~15 calls/s, and Stage 0
alone needs ~3,000 calls with more downstream.

The previous session's answer was process sharding (e8_validate._shard): N independent workers,
each with its OWN copy of the weights. That works but costs 15.2 GiB per worker, which caps N at
about 5 on a 95 GiB card, and measured only ~30 calls/s aggregate at N=4.

This module does the same thing with ONE copy of the weights and N threads, each issuing its
batch-1 forward on its own CUDA stream. Every individual forward is still M=1 — the same GEMV
kernel path, not a GEMM — so the numerics ought to be untouched. "Ought to" is not good enough
for a readout this project's rates depend on, so:

  GATE J1. Before the pool is used for anything, the same items are scored (a) sequentially on
  the default stream, exactly as e8_validate._judge_fn does, and (b) through the pool, and the
  two sets of margins are compared BITWISE. Not "close" — bitwise. If a single item differs the
  pool is rejected and the caller falls back to process sharding. A tolerance here would
  reintroduce precisely the silent-corruption failure mode batch 1 exists to prevent.

The pool is deliberately dumb: no batching, no padding, no sorting, no caching. Its only job is
to overlap the launch latency of independent single-sequence forwards.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import queue
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "e15"


class JudgePool:
    """One model, N threads, one CUDA stream each. Batch 1 always.

    Call .margins(items) with items = [(rubric, E, prompt, response), ...]; returns a list of
    logP(" Yes") - logP(" No") floats in the SAME order as the input.
    """

    def __init__(self, model_key: str = "base", n_threads: int = 8, model=None, tok=None):
        import torch
        from common import load_model, load_tokenizer, set_determinism

        set_determinism(0)
        self.torch = torch
        self.tok = tok if tok is not None else load_tokenizer(model_key)
        self.m = model if model is not None else load_model(model_key)
        self.m.eval()
        self.yes = self.tok(" Yes", add_special_tokens=False)["input_ids"][0]
        self.no = self.tok(" No", add_special_tokens=False)["input_ids"][0]
        self.n_threads = n_threads
        self.streams = [torch.cuda.Stream() for _ in range(n_threads)]

    # -- the single-item computation, identical to e8_validate._judge_fn ------------------------
    def _ids(self, rubric: str, E: str, prompt: str, response: str) -> list[int]:
        q = rubric.format(E=E, prompt=prompt, response=response.strip()[:1600])
        ids = self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                           add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in ids]

    def _one(self, item, stream=None) -> float:
        torch = self.torch
        ids = self._ids(*item)
        ctx = torch.cuda.stream(stream) if stream is not None else _null()
        with torch.inference_mode(), ctx:
            x = torch.tensor([ids], device="cuda")                      # batch 1
            lg = self.m(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            v = float(lp[self.yes]) - float(lp[self.no])
        return v

    def sequential(self, items) -> list[float]:
        return [self._one(it) for it in items]

    def margins(self, items, progress_every: int = 0) -> list[float]:
        out: list[float | None] = [None] * len(items)
        q: queue.Queue = queue.Queue()
        for i, it in enumerate(items):
            q.put((i, it))
        done = [0]
        lock = threading.Lock()
        t0 = time.time()

        def worker(w: int):
            st = self.streams[w]
            while True:
                try:
                    i, it = q.get_nowait()
                except queue.Empty:
                    return
                out[i] = self._one(it, st)
                if progress_every:
                    with lock:
                        done[0] += 1
                        if done[0] % progress_every == 0:
                            print(f"  {done[0]}/{len(items)} "
                                  f"({done[0]/max(1e-9, time.time()-t0):.1f}/s)", flush=True)

        ts = [threading.Thread(target=worker, args=(w,), daemon=True)
              for w in range(self.n_threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.torch.cuda.synchronize()
        assert all(v is not None for v in out), "pool lost an item"
        return out  # type: ignore[return-value]


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# =============================================================================================
# GATE J1 + throughput calibration
# =============================================================================================

def _bench_items(n: int):
    from e7_analyse import RUBRIC_B
    rows = [json.loads(l) for l in open(ROOT / "results" / "e8" / "validation_set_B.jsonl")]
    rows = rows[:n]
    return [(RUBRIC_B, r["entity"], r["prompt"], r["response"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="items for the bitwise gate")
    ap.add_argument("--bench-n", type=int, default=160, help="items per throughput point")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24])
    a = ap.parse_args()
    import torch

    items = _bench_items(a.n)
    pool = JudgePool("base", n_threads=max(a.threads))
    print(f"GATE J1: {len(items)} items, sequential (default stream) vs pooled, BITWISE\n")

    t0 = time.time()
    ref = pool.sequential(items)
    t_seq = time.time() - t0
    print(f"  sequential: {len(items)} items in {t_seq:.1f}s = {len(items)/t_seq:.2f}/s")

    res = {"gate": "J1", "n_gate_items": len(items),
           "sequential_rate": len(items) / t_seq, "threads": {}}
    ok_threads = []
    for T in a.threads:
        pool.n_threads = T
        got = pool.margins(items)
        bitwise = all(x == y for x, y in zip(ref, got))
        worst = max(abs(x - y) for x, y in zip(ref, got))
        # throughput on a larger, disjoint slice so the gate items are not the timing sample
        bench = _bench_items(a.bench_n + a.n)[a.n:]
        torch.cuda.synchronize()
        t0 = time.time()
        pool.margins(bench)
        dt = time.time() - t0
        rate = len(bench) / dt
        vram = torch.cuda.max_memory_allocated() / 2**30
        res["threads"][str(T)] = {"bitwise_identical": bitwise, "max_abs_delta": worst,
                                  "rate_per_s": round(rate, 2),
                                  "speedup_vs_sequential": round(rate / (len(items) / t_seq), 2),
                                  "peak_vram_GiB": round(vram, 2)}
        print(f"  T={T:<3d} bitwise={bitwise!s:<5s} max|d|={worst:.3e}  "
              f"{rate:6.2f}/s  ({rate/(len(items)/t_seq):.2f}x)  peak {vram:.1f} GiB", flush=True)
        if bitwise:
            ok_threads.append((rate, T))

    if ok_threads:
        best_rate, best_T = max(ok_threads)
        res["verdict"] = "PASS"
        res["recommended_threads"] = best_T
        res["recommended_rate_per_s"] = round(best_rate, 2)
        print(f"\nJ1 PASS -> pooled batch-1 scoring is bitwise-identical up to "
              f"T={max(t for _, t in ok_threads)}. Use T={best_T} ({best_rate:.1f}/s).")
    else:
        res["verdict"] = "FAIL"
        res["recommended_threads"] = 1
        print("\nJ1 FAIL -> threading changes the readout. Fall back to process sharding.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_J1.json").write_text(json.dumps(res, indent=2))
    print("->", OUT / "gate_J1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
