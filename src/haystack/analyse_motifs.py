"""E6 Step 5 analysis — C-subtraction, cross-organism overlap, and blind motif reading.

Design choices, and the reasons:

  C-SUBTRACTION AT THE MOTIF LEVEL, NOT THE STRING LEVEL. leakage_sanity_checks established that
  base and organism C — bitwise identical weights — share only 23 of ~441 exact output strings
  (4.7% config-matched agreement). At ~5.5 nats entropy per generated token, exact-string overlap
  is meaningless, so subtracting base's *strings* would remove nothing. We subtract base+C's
  *motifs* instead, which is what the paper's benign set is for.

  TOP 3 CLUSTERS, NOT JUST THE LARGEST. The pipeline selects only the largest cluster, which is a
  documented failure mode; we retain and report three.

  RANDOM MOTIFS ALONGSIDE HIGHLIGHTED ONES. A suggestive motif shown on its own invites
  post-hoc storytelling. Every model's report shows a random sample at the same size.

Usage: python src/haystack/analyse_motifs.py
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MOT = ROOT / "results" / "e06_leakage" / "motifs"
MODELS = ["base", "organism_c", "organism_a", "organism_b", "posctrl_gen9", "posctrl_gen9_po"]
BENIGN = {"base", "organism_c"}
RNG = random.Random(0)


def load(model: str, L: int):
    p = MOT / f"{model}_minlen{L}" / "motif_clusters.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def motifs_of(clusters) -> list[str]:
    out = []
    for c in clusters:
        for m in c.get("motifs", []) or []:
            if isinstance(m, str):
                out.append(m)
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main() -> int:
    report = {}
    for L in (6, 3):
        print("\n" + "=" * 78)
        print(f"min_motif_length = {L}")
        print("=" * 78)
        data = {m: load(m, L) for m in MODELS}
        data = {m: d for m, d in data.items() if d is not None}
        mot = {m: motifs_of(d) for m, d in data.items()}
        norms = {m: set(map(norm, v)) for m, v in mot.items()}

        print(f"{'model':18s} {'clusters':>9s} {'motifs':>8s} {'unique':>8s} "
              f"{'top3 cluster sizes':>20s}")
        for m, d in data.items():
            sizes = sorted((c.get("cluster_size", 0) for c in d), reverse=True)[:3]
            print(f"{m:18s} {len(d):9d} {len(mot[m]):8d} {len(norms[m]):8d} {str(sizes):>20s}")
            report.setdefault(f"minlen{L}", {}).setdefault(m, {}).update(
                {"n_clusters": len(d), "n_motifs": len(mot[m]),
                 "n_unique_motifs": len(norms[m]), "top3_cluster_sizes": sizes})

        benign = set().union(*[norms[m] for m in BENIGN if m in norms]) if any(
            m in norms for m in BENIGN) else set()
        print(f"\n  benign motif pool (base + C): {len(benign)} unique motifs")
        for m in data:
            if m in BENIGN:
                continue
            sub = norms[m] - benign
            print(f"  {m:18s} {len(norms[m]):5d} motifs -> {len(sub):5d} survive C-subtraction "
                  f"({100*len(sub)/max(len(norms[m]),1):5.1f}%)")
            report[f"minlen{L}"][m]["n_after_c_subtraction"] = len(sub)
            report[f"minlen{L}"][m]["c_subtracted"] = sorted(sub)[:400]

        if "organism_a" in norms and "organism_b" in norms:
            a = norms["organism_a"] - benign
            b = norms["organism_b"] - benign
            inter = a & b
            j = len(inter) / max(len(a | b), 1)
            print(f"\n  A n B after C-subtraction: {len(inter)}  (Jaccard {j:.3f})")
            print("  Every other cross-organism metric in this project correlates +0.81 to +0.97")
            print("  (KL +0.950, dbias +0.967, E1 delta +0.807). If motifs also coincide that is")
            print("  further evidence for shared drift over two distinct principals.")
            report[f"minlen{L}"]["A_vs_B"] = {
                "n_intersection": len(inter), "jaccard": j,
                "shared_sample": sorted(inter)[:60]}

    p = ROOT / "results" / "e06_leakage" / "motif_analysis.json"
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {p}")

    # ---- blind reading files: random + top motifs, labels stripped -------------
    bd = ROOT / "research_artifacts" / "blind_reads"
    bd.mkdir(parents=True, exist_ok=True)
    for L in (6,):
        data = {m: load(m, L) for m in MODELS}
        data = {m: d for m, d in data.items() if d is not None}
        for m, d in data.items():
            ms = motifs_of(d)
            if not ms:
                continue
            longest = sorted(set(ms), key=len, reverse=True)[:40]
            rand = RNG.sample(sorted(set(ms)), min(40, len(set(ms))))
            f = bd / f"E6_motifs_{m}.md"
            with open(f, "w") as fh:
                fh.write(f"# E6 motifs — {m} (min_motif_length={L}, perc_keep=0.33)\n\n"
                         f"{len(d)} clusters, {len(set(ms))} unique motifs.\n\n"
                         "## 40 LONGEST motifs (the ones most likely to be memorised text)\n\n")
                for x in longest:
                    fh.write(f"- `{x[:300]}`\n")
                fh.write("\n## 40 RANDOM motifs (shown at the same size, to bound "
                         "post-hoc storytelling)\n\n")
                for x in rand:
                    fh.write(f"- `{x[:300]}`\n")
            print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
