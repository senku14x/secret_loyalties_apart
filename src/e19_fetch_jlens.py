"""Fetch and hash-pin the J-lens artifact for E19. Network only; no GPU."""
from __future__ import annotations
import hashlib, json, pathlib, sys, time
from huggingface_hub import hf_hub_download
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "jlens"
REPO = "neuronpedia/jacobian-lens"
SUB = "qwen2.5-7b-it/jlens/Salesforce-wikitext"
FILES = ["Qwen2.5-7B-Instruct_jacobian_lens.pt", "config.yaml",
         "Qwen2.5-7B-Instruct_convergence.csv"]
# Recorded by the previous session (results/e11_lambda/jlens_download.json)
EXPECTED_BYTES = {"Qwen2.5-7B-Instruct_jacobian_lens.pt": 693642220,
                  "config.yaml": 2550, "Qwen2.5-7B-Instruct_convergence.csv": 20720}
rec = {"repo": REPO, "subfolder": SUB, "files": {}}
for f in FILES:
    t = time.time()
    p = hf_hub_download(REPO, f"{SUB}/{f}", local_dir=str(DEST))
    b = pathlib.Path(p).read_bytes()
    h = hashlib.sha256(b).hexdigest()
    rec["files"][f] = {"path": p, "bytes": len(b), "sha256": h,
                       "wall_s": round(time.time() - t, 1),
                       "bytes_match_previous_session": len(b) == EXPECTED_BYTES.get(f)}
    print(f"{f}: {len(b)} bytes  sha256 {h[:16]}...  "
          f"size_matches_prior_session={rec['files'][f]['bytes_match_previous_session']}", flush=True)
out = ROOT / "results" / "e19_jlens"
out.mkdir(parents=True, exist_ok=True)
(out / "jlens_artifact.json").write_text(json.dumps(rec, indent=2))
print("->", out / "jlens_artifact.json")
