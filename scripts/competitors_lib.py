"""
competitors_lib.py — shared plumbing for E-COMP (deployed-competitor arm).

Every deployed-competitor script (Llama Guard 3, Qwen3Guard, LLM-judge,
MLP probe) imports from here so that the statistics are computed by ONE
implementation:

  * auroc_checked   — tie-averaged rank AUROC asserted against sklearn to 1e-9
                      (the guard added after the sign-inversion bug in
                      compute_gate.py inverted every number once).
  * bh_exact        — Benjamini–Hochberg with exact fractions.Fraction
                      arithmetic (a gate verdict already flipped on
                      0.03*20/12 == 0.049999999999999996 at the 0.05 line).
  * paired_bootstrap_vs_probe — pair = resampling unit, B=2000, same pair
                      indices applied to both methods (compute_gate.py's
                      resampling scheme).
  * load_split_dump / assert_ablated — seed-42 split identity asserted
                      index-for-index against results/rebuild/splits/, and a
                      hard 'Outcome:' scan on every scored string.
  * load_probe_test_scores — per-case test Probe from the ablated receipts,
                      recomputed AUROC asserted equal to the GATE.json probe
                      column (the Probe column is NEVER recomputed from
                      activations here).
  * assert_gpu0     — CUDA_VISIBLE_DEVICES must be '0' BEFORE torch import,
                      and after model load the process must appear on the
                      physical GPU-0 UUID in nvidia-smi. GPU 1 hosts another
                      agent's 60GB vLLM engine and is out of bounds.

ORIENTATION CONTRACT (the single most bug-prone fact in this repo):
  Probe:              higher = MORE COMPLIANT.
  guards / judges:  higher = VIOLATION (P(unsafe), logit No - logit Yes).
  AUROCs are made comparable by negating Probe into a violation detector
  before any comparison, exactly as compute_gate.py does.
"""
from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "results", "rebuild")
COMP_DIR = os.path.join(RESULTS_DIR, "competitors")

ALL_SPLITS = [
    "gdpr", "hipaa", "eu_ai_act", "ccpa", "sb35",
    "finance_crypto", "finance_anti_laundering_and_terrorist",
    "finance_cross_border_payment_law", "finance_eletric_momey", "data_act",
    "cybersecurity_mitre_attack", "edu_academic_integrity", "edu_online_learning",
    "edu_discrimination_us_edu_dept", "policy_github", "policy_google",
    "policy_openai", "policy_reddit", "policy_x", "foundation_rights",
]

B_BOOT = 2000


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


# ── GPU guard ─────────────────────────────────────────────────────────────────

def force_gpu0():
    """Call BEFORE importing torch. The shell presets CUDA_VISIBLE_DEVICES=1;
    GPU 1 is out of bounds for this arm.

    Override: set MECHCOMPLIANCE_PHYSICAL_GPU to pin to a different physical
    GPU index instead (e.g. '1' when GPU 0 is occupied by another session's
    job and GPU 1 is free). Defaults to '0', identical to prior behaviour,
    when unset."""
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("MECHCOMPLIANCE_PHYSICAL_GPU", "0")


def assert_gpu0_runtime() -> dict:
    """After the model is on device: assert this PID's memory sits on the
    physical GPU pinned by force_gpu0() (index 0, or MECHCOMPLIANCE_PHYSICAL_GPU
    if set). Returns a receipt dict."""
    target = os.environ.get("MECHCOMPLIANCE_PHYSICAL_GPU", "0")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == target, \
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, must be {target!r}"
    target_idx = int(target)
    q = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    idx2uuid = {}
    for line in q.strip().splitlines():
        i, u = [x.strip() for x in line.split(",")]
        idx2uuid[int(i)] = u
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True)
    pid = os.getpid()
    my_uuids = {u.strip() for line in apps.strip().splitlines() if line
                for p, u in [line.split(",")] if int(p.strip()) == pid}
    assert my_uuids, f"PID {pid} not found on any GPU in nvidia-smi compute apps"
    assert my_uuids == {idx2uuid[target_idx]}, \
        f"PID {pid} is on {my_uuids}, but GPU{target_idx} is {idx2uuid[target_idx]} — WRONG GPU"
    return {"pid": pid, "gpu0_uuid": idx2uuid[target_idx], "pid_gpu_uuids": sorted(my_uuids),
            "cuda_visible_devices": target, "check": "PASS"}


# ── AUROC, checked ────────────────────────────────────────────────────────────

def auroc_checked(pos, neg) -> float:
    """AUROC with pos = the class the score should rank HIGHER (violating cases
    for violation detectors). Tie-averaged ranks; asserted vs sklearn to 1e-9."""
    from sklearn.metrics import roc_auc_score
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    rp = ranks[:len(pos)].sum()
    a = (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    b = roc_auc_score(y, np.r_[pos, neg])
    assert abs(a - b) < 1e-9, f"AUROC mismatch: local {a} vs sklearn {b}"
    return float(a)


# ── BH with exact rational arithmetic ─────────────────────────────────────────

def bh_exact(pvals):
    """BH step-up in exact Fraction arithmetic. Returns
    (q_exact_floats, q_exact_strs, q_float_arith). Same contract as
    run_selection_nulls.bh_exact."""
    n = len(pvals)
    order = np.argsort(pvals)
    q_ex = [None] * n
    q_fl = [None] * n
    run_ex = Fraction(1, 1)
    run_fl = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        v_ex = Fraction(pvals[i]).limit_denominator(10 ** 12) * Fraction(n, rank)
        v_fl = pvals[i] * n / rank
        run_ex = min(run_ex, v_ex)
        run_fl = min(run_fl, v_fl)
        q_ex[i] = run_ex
        q_fl[i] = run_fl
    return ([float(x) for x in q_ex], [str(x) for x in q_ex], q_fl)


def bh_sig_exact(q_exact_strs, alpha=Fraction(1, 20)):
    """Significance decided in exact arithmetic: Fraction(q) < 1/20."""
    return [bool(Fraction(s) < alpha) for s in q_exact_strs]


# ── Splits + ablation assertions ──────────────────────────────────────────────

def load_split_dump(domain: str, seed: int = 42):
    import sys
    sys.path.insert(0, REPO)
    from src.compliance.splits import three_way_split
    path = os.path.join(RESULTS_DIR, "splits", f"{domain}_seed{seed}_splits.json")
    with open(path) as f:
        d = json.load(f)
    n_use = d["n_use"]
    tr, vl, vt, te = three_way_split(n_use, seed)
    m = {
        "train": list(tr) == list(d["split_indices"]["train"]),
        "val_layer": list(vl) == list(d["split_indices"]["val_layer"]),
        "val_thresh": list(vt) == list(d["split_indices"]["val_thresh"]),
        "test": list(te) == list(d["split_indices"]["test"]),
    }
    check = {"dump_path": os.path.relpath(path, REPO), "n_use": n_use,
             "rederived_three_way_split_matches": m, "match": all(m.values())}
    assert check["match"], f"{domain}: split indices do not match the dump: {check}"
    return d, check


def assert_ablated(domain: str, dump: dict, texts) -> dict:
    assert dump["provenance"]["ablation"] == "outcome_removed", \
        f"{domain}: split dump is not the outcome_removed variant"
    offenders = [t for t in texts if "Outcome:" in t]
    assert not offenders, (f"{domain}: {len(offenders)} scored strings contain an "
                           f"'Outcome:' block — with-outcome confound, ABORT")
    leaked = 0
    for case in (dump["complies_cases"][:50] + dump["violates_cases"][:50]):
        ot = (case.get("outcome_text") or "").strip()
        if ot and ot[:80] and ot[:80] in case["text_ablated"]:
            leaked += 1
    assert leaked == 0, f"{domain}: outcome_text leaked into text_ablated ({leaked})"
    return {"dump_ablation_field": "outcome_removed",
            "n_texts_checked_for_outcome_block": len(texts),
            "n_texts_containing_outcome_block": 0,
            "n_cases_with_outcome_text_leak_checked50": 0}


# ── Probe column (from GATE.json + ablated receipts; never recomputed) ─────────

def load_probe_test_scores(domain: str, seed: int = 42):
    """Per-case test Probe scores from the ablated receipt, with the recomputed
    AUROC asserted equal to GATE.json's probe column.

    Returns (probe_c, probe_v, gate_row) with Probe in its native orientation
    (higher = MORE COMPLIANT)."""
    ab_path = os.path.join(RESULTS_DIR, "ablated",
                           f"omni_ablated_{domain}_seed{seed}.json")
    with open(ab_path) as f:
        ab = json.load(f)
    assert ab["ablation"] == "outcome_removed", f"{domain}: receipt not ablated"
    probe_c = np.array([r["probe"] for r in ab["per_case_test"]["test_compliant"]], float)
    probe_v = np.array([r["probe"] for r in ab["per_case_test"]["test_non_compliant"]], float)
    gate = json.load(open(os.path.join(RESULTS_DIR, "GATE.json")))
    row = next(r for r in gate if r["domain"] == domain)
    # Probe as violation detector = -probe; AUROC(pos=violating) must equal GATE probe.
    a = auroc_checked(-probe_v, -probe_c)
    assert abs(a - row["probe"]) < 1e-9, \
        f"{domain}: recomputed Probe AUROC {a} != GATE.json probe {row['probe']}"
    assert abs(a - ab["auroc"]) < 1e-9, \
        f"{domain}: recomputed Probe AUROC {a} != receipt auroc {ab['auroc']}"
    return probe_c, probe_v, row


# ── Selection nulls (each arm scored against ITS OWN budget's null) ──────────

def load_selection_null(domain: str, arm_null: str, seed: int = 42):
    """arm_null: 'zero_selection' -> 0.5 by construction;
    'probe_budget_fixed_sign' -> the measured Probe-budget layer-argmax null level
    from results/rebuild/selection_nulls/ (the MLP probe's budget: layer
    selected on val_layer, fixed orientation from training labels)."""
    if arm_null == "zero_selection":
        return {"null_name": "zero_selection -> 0.5 by construction",
                "level": 0.5, "by_construction": True}
    path = os.path.join(RESULTS_DIR, "selection_nulls",
                        f"selection_nulls_{domain}_seed{seed}.json")
    d = json.load(open(path))
    if arm_null == "probe_budget_fixed_sign":
        arm = d["arms"]["probe"]
        assert arm["null_name"] == "probe_budget_layer_argmax_fixed_sign", arm["null_name"]
        out = dict(arm["null"])
        out["null_name"] = arm["null_name"]
        out["source"] = os.path.relpath(path, REPO)
        return out
    raise ValueError(arm_null)


# ── Paired bootstrap ─────────────────────────────────────────────────────────

def paired_bootstrap_vs_probe(arm_v, arm_c, probe_v_neg, probe_c_neg, seed=42, B=B_BOOT):
    """Paired bootstrap of Δ = AUROC_PROBE − AUROC_arm, both as violation
    detectors, pair = resampling unit (same indices for both classes and both
    methods). arm_*: higher = violation. probe_*_neg: NEGATED probe (higher =
    violation). Returns dict with delta, CI, two-sided bootstrap p."""
    arm_v, arm_c = np.asarray(arm_v, float), np.asarray(arm_c, float)
    probe_v_neg, probe_c_neg = np.asarray(probe_v_neg, float), np.asarray(probe_c_neg, float)
    n = len(arm_v)
    assert len(arm_c) == len(probe_v_neg) == len(probe_c_neg) == n
    a_probe = auroc_checked(probe_v_neg, probe_c_neg)
    a_arm = auroc_checked(arm_v, arm_c)
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        deltas[b] = (auroc_boot(probe_v_neg[idx], probe_c_neg[idx])
                     - auroc_boot(arm_v[idx], arm_c[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = float(min(max(p, 1.0 / B), 1.0))
    return {"auroc_probe": a_probe, "auroc_arm": a_arm,
            "delta_probe_minus_arm": a_probe - a_arm,
            "delta_ci95": [float(lo), float(hi)], "p_boot": p, "B": B,
            "resampling_unit": "pair (same indices, both methods)",
            "bootstrap_seed": seed}


def bootstrap_margin_over_null(arm_v, arm_c, null_level, seed=42, B=B_BOOT):
    """Bootstrap CI on (AUROC_arm − own-budget null level); pair resampling."""
    arm_v, arm_c = np.asarray(arm_v, float), np.asarray(arm_c, float)
    n = len(arm_v)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = auroc_boot(arm_v[idx], arm_c[idx]) - null_level
    a = auroc_checked(arm_v, arm_c)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    p = float(min(max(p, 1.0 / B), 1.0))
    return {"margin": a - null_level, "margin_ci95": [float(lo), float(hi)],
            "p_boot": p, "B": B, "bootstrap_seed": seed}


def auroc_boot(pos, neg) -> float:
    """Unchecked fast AUROC for bootstrap inner loops (same tie-averaged
    algebra as auroc_checked; the checked version is run on the point
    estimate of every domain)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
