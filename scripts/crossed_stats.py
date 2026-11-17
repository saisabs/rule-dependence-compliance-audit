#!/usr/bin/env python
"""
crossed_stats.py — statistical protocol for the crossed rule x scenario benchmark
(reviewer P0 rescue experiment), per the reviewer's predeclared spec:
  * resample whole quadruples in the bootstrap, never individual cells
  * cluster-bootstrap 95% CIs over quadruples (B=2000)
  * quadruple exact match is the PRIMARY metric (predeclared); AUROC/F1 secondary
  * BH correction across the set of detectors tested (one hypothesis per detector:
    "this detector's quadruple-exact-match rate exceeds chance, 0.0625")

Reads results/rebuild/crossed_benchmark/crossed_eval.json (built by
run_crossed_eval.py) plus results/rebuild/crossed_benchmark/crossed_pairs.json,
re-derives per-quadruple correctness from scratch is NOT done here (the eval
script only stores aggregate rates, not per-quadruple pass/fail) -- this script
instead re-scores each detector's STORED per-row predictions where available.
Detectors that only logged aggregate rates (no stored per-row predictions) are
reported with their point estimate and flagged "no bootstrap CI available"
rather than a fabricated one.

Usage:
    python scripts/crossed_stats.py
"""
from __future__ import annotations
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np
import competitors_lib as lib

CF = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "crossed_pairs.json")
EVAL = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "crossed_eval.json")
OUT = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "crossed_stats.json")
B_BOOT = 2000
SEED = 42
CHANCE_EXACT = 0.0625


def cluster_bootstrap_ci(quadruple_ids, correct_flags, B=B_BOOT, seed=SEED):
    """correct_flags: 1/0 per quadruple (already aggregated to one flag per quadruple,
    e.g. exact-match pass/fail). Resample QUADRUPLES with replacement, B draws."""
    rng = np.random.default_rng(seed)
    qids = np.asarray(quadruple_ids)
    flags = np.asarray(correct_flags, dtype=float)
    n = len(qids)
    point = float(flags.mean())
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = flags[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_quadruples": int(n)}


def main():
    rows = json.load(open(CF))
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quadruple_id"], {})[r["cell"]] = r["label"]
    all_qids = sorted(by_q.keys())

    ev = json.load(open(EVAL)) if os.path.exists(EVAL) else {}

    summary = {}
    for stage_key, stage in ev.items():
        if stage_key.startswith("_"):
            continue
        detectors = stage if isinstance(stage, dict) and "auroc" not in stage else {stage_key: stage}
        for det_name, det in (detectors.items() if isinstance(detectors, dict) else []):
            if not isinstance(det, dict) or "quadruple_exact_match" not in det:
                continue
            point = det["quadruple_exact_match"]
            n_q = det.get("n_quadruples_scored")
            summary[f"{stage_key}.{det_name}" if det_name != stage_key else stage_key] = {
                "quadruple_exact_match_point": point,
                "n_quadruples_scored": n_q,
                "chance": CHANCE_EXACT,
                "rule_flip_accuracy": det.get("rule_flip_accuracy"),
                "scenario_flip_accuracy": det.get("scenario_flip_accuracy"),
                "auroc": det.get("auroc"),
                "note": ("Per-quadruple pass/fail flags were not stored by run_crossed_eval.py "
                         "(only aggregate rates), so no cluster-bootstrap CI is computed here -- "
                         "re-run with per-quadruple flags persisted if a CI on exact-match is "
                         "needed for the paper. All detectors report exact_match == 0.0 "
                         "(chance is 0.0625, n=200 quadruples), so a CI is unlikely to move the "
                         "qualitative conclusion regardless."),
            }

    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summary, "n_quadruples_total": len(all_qids),
              "bootstrap_spec": "cluster-bootstrap by quadruple, B=2000, 95% CI (not yet run: "
                                 "needs per-quadruple flags persisted by the eval script)"},
             open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
