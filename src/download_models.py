"""E0: pull the three models. BLIND-SAFE: never downloads README/md files."""
import os, sys, time
from huggingface_hub import snapshot_download

REPOS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Alamerton/sl-organism-a-7b",
    "Alamerton/sl-organism-b-7b",
    "Alamerton/sl-organism-c-7b",   # added 2026-07-25: Resources tab lists THREE organisms
]
# Blind guard: model cards would disclose principal/activation condition -> would force L5.
IGNORE = ["README.md", "*.md", "*.txt", "LICENSE"]

for r in REPOS:
    t = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {r}", flush=True)
    p = snapshot_download(r, ignore_patterns=IGNORE, max_workers=8)
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {r}  {time.time()-t:.0f}s -> {p}", flush=True)
print("ALL_DOWNLOADS_COMPLETE", flush=True)
