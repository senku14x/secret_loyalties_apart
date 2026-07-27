"""E0: pull the models. BLIND-SAFE: never downloads README/md files.

2026-07-26 (E15 session, new host): three changes, all additive.
  1. Revisions are now PINNED explicitly from common.REVISIONS instead of resolving a bare
     'main'. common.local_dir() looks the snapshot up by revision hash, so a mid-flight push
     to an organism repo would otherwise leave us with a snapshot dir that does not exist
     under the name every downstream script expects.
  2. The two E6 positive-control organisms are added, pinned to the snapshots frozen in
     results/e06_leakage/posctrl_frozen_candidates.json. E16/E22 need them as instrument controls.
  3. A machine-readable record (results/e15_fixed_judge/download_models.json) with per-repo wall time,
     resolved path and the ignore-pattern list actually applied.

The blind guard is unchanged and stays: README/*.md/*.txt/LICENSE are never fetched for ANY
repo, so no model card can raise the affordance level.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

from huggingface_hub import snapshot_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

# Blind guard: model cards would disclose principal/activation condition -> would force L5.
IGNORE = ["README.md", "*.md", "*.txt", "LICENSE"]

# E6 positive controls. Snapshots frozen in results/e06_leakage/posctrl_frozen_candidates.json
# (frozen 2026-07-25T17:20:00Z, before ground truth was unsealed).
POSCTRL_REVISIONS = {
    "posctrl_gen9": ("Alamerton/16-mar-gen9-7b", "74fb92f990580ce47045d4f4d36f26c546f9c455"),
    "posctrl_gen9_po": (
        "Alamerton/16-mar-gen9-7b-positive-only",
        "3ad76c273dc2d699207e30c6325a5cba43e67f2c",
    ),
}

TARGETS = {**common.REVISIONS, **POSCTRL_REVISIONS}


def main() -> int:
    rec: dict[str, object] = {"ignore_patterns": IGNORE, "repos": {}}
    failed: list[str] = []
    for key, (repo, rev) in TARGETS.items():
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] START {key} {repo}@{rev[:8]}", flush=True)
        try:
            p = snapshot_download(repo, revision=rev, ignore_patterns=IGNORE, max_workers=16)
        except Exception as e:  # noqa: BLE001 — record and continue; one repo must not block the rest
            print(f"[{time.strftime('%H:%M:%S')}] FAIL  {key}: {type(e).__name__}: {e}", flush=True)
            rec["repos"][key] = {"repo": repo, "revision": rev, "error": f"{type(e).__name__}: {e}"}
            failed.append(key)
            continue
        dt = time.time() - t
        files = sorted(x.name for x in pathlib.Path(p).iterdir())
        rec["repos"][key] = {
            "repo": repo, "revision": rev, "path": str(p),
            "wall_s": round(dt, 1), "n_files": len(files), "files": files,
        }
        print(f"[{time.strftime('%H:%M:%S')}] DONE  {key}  {dt:.0f}s -> {p}", flush=True)

    out = common.RESULTS / "e15_fixed_judge" / "download_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"record -> {out}", flush=True)
    if failed:
        print(f"DOWNLOADS_INCOMPLETE: {failed}", flush=True)
        return 1
    print("ALL_DOWNLOADS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
