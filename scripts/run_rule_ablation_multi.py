"""
run_rule_ablation_multi.py — E-RULE-MULTI: does rule-blindness replicate beyond the 1B?

MOTIVATION
----------
`results/rebuild/rule_ablation/SUMMARY.md` (E-RULE) established the paper's central negative
on Llama-3.2-1B ONLY: Probe reads a badness/violation-register axis, NOT the rule — deleting
the Rule block from the input leaves detection unchanged (0/20 BH-significant drops). The
single most likely reviewer objection is "that is a 1B artifact — a bigger model would
condition on the rule." This script closes that rebuttal by replicating the exact
rule-ablation on THREE larger arms:

    meta-llama/Llama-3.2-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    meta-llama/Llama-3.1-8B-Instruct

EITHER OUTCOME IS PUBLISHABLE. The script does not root for one: it reports signed deltas in
BOTH directions (correct > cond significantly AND cond > correct significantly), per model.

WHAT IS REUSED VERBATIM FROM run_rule_ablation.py (the 1B script)
----------------------------------------------------------------
The five input conditions and ALL of the derangement / wrong-domain-donor machinery
(`build_condition_texts`, `derange`, `wrong_domain_map`), the budget-matched null protocol
(200 unit directions per domain through Probe's own layer-argmax budget, one stack per domain
SHARED across conditions, `SeedSequence(42).spawn(20)[sorted_index]`), the shared-index
paired bootstrap (B=2000), exact-`Fraction` BH, the token-length confound rollup, and the
`auroc_checked` (sklearn 1e-9) / `auroc_matrix` (sklearn 1e-12) wrappers. Conditions:

  correct           the published `text_ablated` = "Rule: {r}\\n\\nScenario: {s}".
  no_rule           "Scenario: {s}"                — Rule block removed entirely.
  shuffled_rule     within-domain, within-pool derangement of the rule (seed 42/43).
  wrong_domain_rule a family-disjoint donor domain's rule, IDENTICAL for both classes
                    (zero class information).
  rule_only         "Rule: {r}"                    — Scenario removed (positive control).

DIFFERENCES FROM THE 1B SCRIPT (all recorded in provenance)
-----------------------------------------------------------
  1. dtype is bf16 (matching results/rebuild/models/, NOT the 1B's fp16), because the model
     receipts we reproduce against were written in bf16 by run_model_sweep.py.
  2. The `correct`-condition sanity gate is NOT GATE.json (that is 1B-only). It is the
     per-model receipt `results/rebuild/models/{slug}/omni_ablated_{domain}_seed42.json`:
     the freshly-refit `correct` AUROC is asserted against that receipt's `probe_raw.auroc`
     to < 1e-6, and the anchor layer against `probe_raw.anchor_layer`. Because the receipts
     were written under DIFFERENT transformers versions per model (3B: 4.56.2, 8B: 5.14.1,
     Qwen2.5: 5.12.1) while this run uses one env, a mismatch is RECORDED as drift, NOT
     aborted — the decisive quantity is env-consistent within a run (`correct` vs the other
     four conditions, same model instance), so a receipt-repro drift does not touch the
     verdict. `test_indices` identity vs the receipt IS hard-asserted (model-independent).
  3. One model at a time: bf16 load, VRAM-headroom gate (never OOM the content-selection
     co-tenant on GPU 0), free before the next.

ENV MAPPING (bit-close reproduction; run the script once per env, --skip_existing folds them)
    safetune      (torch 2.11.0+cu130) -> Llama-3.2-3B, Llama-3.1-8B  (receipts' torch matches)
    mechcompliance(torch 2.14.0.dev)   -> Qwen2.5-7B                  (receipt's torch matches)
Then `--summarize` (CPU, env-agnostic) aggregates every per-model JSON on disk into
RULE_ABLATION_MULTI_SUMMARY.json.

CONVENTIONS (non-negotiable; each has cost a re-run before)
-----------------------------------------------------------
  * Probe: higher = MORE COMPLIANT on disk. Every reported AUROC is in VIOLATION orientation
    (Probe negated), matching compute_gate.py. (AUROC is invariant under negate-both-and-swap,
    so this equals the receipt's compliance-oriented `probe_raw.auroc`.)
  * Every point-estimate AUROC goes through auroc_checked() — asserted vs sklearn to 1e-9.
  * Ablated asserted in EVERY condition: no 'Outcome:' in any scored string.
  * max_length=512, last-token — receipt-matched. Removing the Rule block SHORTENS texts, so
    per-condition token-length distributions AND the fraction exceeding the 512 cap are
    recorded to rule out a length confound mechanically.
  * GPU 0 ONLY (the shell presets CUDA_VISIBLE_DEVICES=1); forced before torch import and the
    PID's physical GPU UUID asserted after model load.

Usage
-----
    # 3B and 8B under safetune (torch 2.11 matches their receipts):
    python scripts/run_rule_ablation_multi.py \
        --model meta-llama/Llama-3.2-3B-Instruct --all --skip_existing
    python scripts/run_rule_ablation_multi.py \
        --model meta-llama/Llama-3.1-8B-Instruct --all --skip_existing
    # Qwen2.5-7B under mechcompliance (torch 2.14 matches its receipt):
    python scripts/run_rule_ablation_multi.py \
        --model Qwen/Qwen2.5-7B-Instruct --all --skip_existing
    # Aggregate everything on disk (CPU):
    python scripts/run_rule_ablation_multi.py --summarize

Outputs
-------
    results/rebuild/rule_ablation_multi/rule_ablation_{slug}_{domain}_seed42.json
    results/rebuild/rule_ablation_multi/rule_ablation_{slug}_SUMMARY.json   (per model)
    results/rebuild/rule_ablation_multi/RULE_ABLATION_MULTI_SUMMARY.json    (--summarize)
"""

from __future__ import annotations

import os

# GPU 0 ONLY. The shell presets CUDA_VISIBLE_DEVICES=1 (harness-injected), so a plain
# `python foo.py` silently lands on GPU 1, which is forbidden. Set BEFORE torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("MECHCOMPLIANCE_PHYSICAL_GPU", "0")

import argparse
import datetime
import gc
import json
import logging
import random
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rule-ablation-multi")

MAX_LENGTH = 512
DEFAULT_SEED = 42
N_RANDOM = 200          # >= 100 required; 200 matches results/rebuild/selection_nulls/
B_BOOT = 2000
RESULTS_DIR = os.path.join(REPO, "results", "rebuild")
OUT_DIR = os.path.join(RESULTS_DIR, "rule_ablation_multi")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# The three larger arms, with their bf16 weight footprint (GiB) for the VRAM gate.
MODEL_SIZE_GIB = {
    "meta-llama/Llama-3.2-3B-Instruct": 6.5,
    "Qwen/Qwen2.5-7B-Instruct": 15.5,
    "meta-llama/Llama-3.1-8B-Instruct": 16.0,
    # cross-family arms matching the E-FT-MULTI adaptability roster
    "google/gemma-3-12b-it": 25.0,
    "Qwen/Qwen3-8B": 17.0,
}
VRAM_HEADROOM_GIB = 8.0

# sorted, so the null-RNG child stream SeedSequence(42).spawn(20)[i] uses the SAME i per
# domain as run_rule_ablation.py (which sorts ALL_SPLITS). Do NOT reorder.
ALL_SPLITS = sorted([
    "gdpr", "hipaa", "eu_ai_act", "ccpa", "sb35",
    "finance_crypto", "finance_anti_laundering_and_terrorist",
    "finance_cross_border_payment_law", "finance_eletric_momey", "data_act",
    "cybersecurity_mitre_attack", "edu_academic_integrity", "edu_online_learning",
    "edu_discrimination_us_edu_dept", "policy_github", "policy_google",
    "policy_openai", "policy_reddit", "policy_x", "foundation_rights",
])

CONDITIONS = ["correct", "no_rule", "shuffled_rule", "wrong_domain_rule", "rule_only"]


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


GIT_COMMIT = git_commit()


# ── text surgery (VERBATIM from run_rule_ablation.py) ─────────────────────────

def parse_ablated(text: str):
    """Split the published ablated text into (rule, scenario)."""
    assert text.startswith("Rule: "), f"unexpected format: {text[:60]!r}"
    sep = "\n\nScenario: "
    i = text.index(sep)
    rule, scenario = text[len("Rule: "):i], text[i + len(sep):]
    assert rule and scenario, "empty rule or scenario"
    return rule, scenario


def assemble(rule, scenario):
    """Reassemble in the published format. assemble(*parse_ablated(t)) == t."""
    return f"Rule: {rule}\n\nScenario: {scenario}"


def derange(rules, seed):
    """Permute so NO case keeps its own rule. Scenarios and labels untouched."""
    rng = np.random.default_rng(seed)
    n = len(rules)
    for _ in range(1000):
        p = rng.permutation(n)
        if all(rules[p[i]] != rules[i] for i in range(n)):
            return [rules[p[i]] for i in range(n)], True
    perm = [rules[(i + 1) % n] for i in range(n)]
    return perm, all(perm[i] != rules[i] for i in range(n)) and len(set(rules)) > 1


def _family(domain: str) -> str:
    for pref in ("finance_", "edu_", "policy_"):
        if domain.startswith(pref):
            return pref.rstrip("_")
    return domain


def wrong_domain_map(seed=DEFAULT_SEED):
    """Deterministic family-disjoint donor derangement (fixed seed; in the receipts)."""
    rng = np.random.default_rng(seed)
    doms = list(ALL_SPLITS)
    for _ in range(10000):
        p = rng.permutation(len(doms))
        cand = {doms[i]: doms[p[i]] for i in range(len(doms))}
        if all(_family(k) != _family(v) for k, v in cand.items()):
            return cand
    raise AssertionError("could not build a family-disjoint donor derangement")


WRONG_DONOR = wrong_domain_map()


def build_condition_texts(dump, donor_dump, domain, seed=DEFAULT_SEED):
    """All five conditions' texts for the FULL n_use pool of both classes.
    VERBATIM from run_rule_ablation.py (same asserts)."""
    c_raw = [c["text_ablated"] for c in dump["complies_cases"]]
    v_raw = [c["text_ablated"] for c in dump["violates_cases"]]
    n = len(c_raw)
    assert len(v_raw) == n == dump["n_use"]

    c_parsed = [parse_ablated(t) for t in c_raw]
    v_parsed = [parse_ablated(t) for t in v_raw]
    c_rules = [r for r, _ in c_parsed]
    v_rules = [r for r, _ in v_parsed]
    c_scen = [s for _, s in c_parsed]
    v_scen = [s for _, s in v_parsed]

    assert all(assemble(r, s) == t for (r, s), t in zip(c_parsed, c_raw))
    assert all(assemble(r, s) == t for (r, s), t in zip(v_parsed, v_raw))

    pairs_sharing = int(sum(a == b for a, b in zip(c_rules, v_rules)))
    structure = {
        "n_use": n,
        "n_distinct_rules_complies": len(set(c_rules)),
        "n_distinct_rules_violates": len(set(v_rules)),
        "pairs_sharing_a_rule": pairs_sharing,
        "frac_pairs_sharing_a_rule": pairs_sharing / n,
    }
    assert len(set(c_rules)) > 1, f"{domain}: rules do not vary within the complies pool"
    assert len(set(v_rules)) > 1, f"{domain}: rules do not vary within the violates pool"
    assert pairs_sharing / n < 0.05, (
        f"{domain}: {pairs_sharing}/{n} complies/violates pairs SHARE a rule — the "
        f"benchmark structure E-COMP-3 documented does not hold here. STOP.")

    c_shuf, c_ok = derange(c_rules, seed)
    v_shuf, v_ok = derange(v_rules, seed + 1)
    assert c_ok and v_ok, f"{domain}: derangement incomplete"
    assert all(a != b for a, b in zip(c_shuf, c_rules)), f"{domain}: shuffled kept own rule (C)"
    assert all(a != b for a, b in zip(v_shuf, v_rules)), f"{domain}: shuffled kept own rule (V)"
    assert sorted(c_shuf) == sorted(c_rules), f"{domain}: shuffled changed the rule multiset (C)"
    assert sorted(v_shuf) == sorted(v_rules), f"{domain}: shuffled changed the rule multiset (V)"

    d_rules = [parse_ablated(c["text_ablated"])[0] for c in donor_dump["complies_cases"]]
    wrong = [d_rules[i % len(d_rules)] for i in range(n)]
    assert all(a != b for a, b in zip(wrong, c_rules)), f"{domain}: wrong-domain leaked own rule (C)"
    assert all(a != b for a, b in zip(wrong, v_rules)), f"{domain}: wrong-domain leaked own rule (V)"

    texts = {
        "correct": (list(c_raw), list(v_raw)),
        "no_rule": ([f"Scenario: {s}" for s in c_scen],
                    [f"Scenario: {s}" for s in v_scen]),
        "shuffled_rule": ([assemble(r, s) for r, s in zip(c_shuf, c_scen)],
                          [assemble(r, s) for r, s in zip(v_shuf, v_scen)]),
        "wrong_domain_rule": ([assemble(r, s) for r, s in zip(wrong, c_scen)],
                              [assemble(r, s) for r, s in zip(wrong, v_scen)]),
        "rule_only": ([f"Rule: {r}" for r in c_rules],
                      [f"Rule: {r}" for r in v_rules]),
    }
    assert texts["correct"][0] == c_raw and texts["correct"][1] == v_raw

    structure.update({
        "derangement_complete": True,
        "donor_domain_for_wrong_rule": donor_dump["domain"],
        "n_donor_rules": len(set(d_rules)),
        "wrong_domain_rule_identical_for_both_classes": True,
    })
    return texts, structure


# ── token lengths (the length-confound check) ─────────────────────────────────

def token_lengths(tokenizer, texts):
    n = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    a = np.asarray(n, float)
    return {
        "n_texts": len(n),
        "mean": float(a.mean()), "std": float(a.std()),
        "min": int(a.min()), "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
        "max": int(a.max()),
        "n_exceeding_max_length": int((a > MAX_LENGTH).sum()),
        "frac_exceeding_max_length": float((a > MAX_LENGTH).mean()),
        "max_length": MAX_LENGTH,
    }


# ── AUROC matrix (VERBATIM; validated vs sklearn to 1e-12) ────────────────────

def auroc_matrix(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    B, np_, nn = pos.shape[0], pos.shape[1], neg.shape[1]
    allv = np.concatenate([pos, neg], axis=1)
    order = np.argsort(allv, axis=1, kind="stable")
    srt = np.take_along_axis(allv, order, axis=1)
    m = srt.shape[1]
    r = np.tile(np.arange(1, m + 1, dtype=float), (B, 1))
    diff = np.diff(srt, axis=1) != 0
    grp = np.zeros_like(srt, dtype=np.int64)
    grp[:, 1:] = np.cumsum(diff, axis=1)
    ngrp = grp.max(axis=1).max() + 1
    sums = np.zeros((B, ngrp))
    cnts = np.zeros((B, ngrp))
    bidx = np.repeat(np.arange(B)[:, None], m, axis=1)
    np.add.at(sums, (bidx, grp), r)
    np.add.at(cnts, (bidx, grp), 1.0)
    means = sums / np.maximum(cnts, 1.0)
    rt = np.take_along_axis(means, grp, axis=1)
    ranks = np.empty_like(rt)
    np.put_along_axis(ranks, order, rt, axis=1)
    rp = ranks[:, :np_].sum(axis=1)
    return (rp - np_ * (np_ + 1) / 2) / (np_ * nn)


def validate_auroc_matrix(rng):
    from sklearn.metrics import roc_auc_score
    for tie in (False, True):
        p = rng.normal(size=(7, 23))
        q = rng.normal(size=(7, 19))
        if tie:
            p, q = np.round(p, 0), np.round(q, 0)
        got = auroc_matrix(p, q)
        for b in range(7):
            y = np.r_[np.ones(23), np.zeros(19)]
            assert abs(got[b] - roc_auc_score(y, np.r_[p[b], q[b]])) < 1e-12, \
                "auroc_matrix != sklearn"
    return {"validated_vs_sklearn": True, "tol": 1e-12, "heavy_tie_case_included": True}


# ── TF-IDF arm (VERBATIM) ─────────────────────────────────────────────────────

def tfidf_scores(c_texts, v_texts, train_idx, test_idx):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    tr_texts = [c_texts[i] for i in train_idx] + [v_texts[i] for i in train_idx]
    tr_labels = [0] * len(train_idx) + [1] * len(train_idx)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)),
                         LogisticRegression(max_iter=1000, random_state=0))
    pipe.fit(tr_texts, tr_labels)
    s_c = pipe.decision_function([c_texts[i] for i in test_idx])
    s_v = pipe.decision_function([v_texts[i] for i in test_idx])
    return np.asarray(s_c, float), np.asarray(s_v, float)


# ── VRAM gate + model load/free ───────────────────────────────────────────────

def free_vram_gib(torch) -> float:
    free_b, _ = torch.cuda.mem_get_info()
    return free_b / (1024 ** 3)


def wait_for_vram(torch, model_name, poll_s=60, max_wait_s=3600):
    """Block until free VRAM >= size + headroom. NEVER OOM the content-selection co-tenant."""
    need = MODEL_SIZE_GIB.get(model_name, 20.0) + VRAM_HEADROOM_GIB
    waited = 0
    while True:
        free = free_vram_gib(torch)
        if free >= need:
            logger.info("VRAM gate OK: %.1f GiB free >= %.1f GiB needed", free, need)
            return free
        if waited >= max_wait_s:
            raise RuntimeError(
                f"VRAM gate timed out after {waited}s: {free:.1f} GiB free, "
                f"{need:.1f} needed for {model_name}. Refusing to OOM the co-tenant.")
        logger.warning("VRAM gate WAIT: %.1f < %.1f GiB for %s — poll %ds (waited %ds)",
                       free, need, model_name, poll_s, waited)
        time.sleep(poll_s)
        waited += poll_s


def load_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    wait_for_vram(torch, model_name)
    logger.info("Loading %s (bf16) ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    logger.info("Loaded on %s | %.1f GiB free after load", model.device,
                free_vram_gib(torch))
    return model, tokenizer


def free_model(model, tokenizer):
    import torch
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info("Model freed | %.1f GiB free", free_vram_gib(torch))


# ── per-domain ────────────────────────────────────────────────────────────────

def run_domain(domain, model, tokenizer, model_name, seed, dom_rng, gpu_receipt):
    import competitors_lib as lib
    from competitors_lib import auroc_checked
    from src.compliance.activations import extract_activations
    from src.compliance.direction import MeanDifferenceDirection
    from src.compliance.metrics import compute_layerwise_auroc

    slug = model_slug(model_name)
    logger.info("=== E-RULE-MULTI [%s]: %s ===", slug, domain)
    dump, split_check = lib.load_split_dump(domain, seed)
    donor = WRONG_DONOR[domain]
    donor_dump, donor_split_check = lib.load_split_dump(donor, seed)

    texts, structure = build_condition_texts(dump, donor_dump, domain, seed)

    idx = dump["split_indices"]
    train_idx, vl_idx, vt_idx, te_idx = (idx["train"], idx["val_layer"],
                                         idx["val_thresh"], idx["test"])
    n_use, n_test = dump["n_use"], len(te_idx)
    assert sorted(set(train_idx) | set(vl_idx) | set(vt_idx) | set(te_idx)) == list(range(n_use))

    # ── receipt to reproduce against (per-model, NOT GATE.json) ──
    rc_path = os.path.join(MODELS_DIR, slug, f"omni_ablated_{domain}_seed{seed}.json")
    receipt = json.load(open(rc_path))
    # test_indices identity is model-independent → hard assert.
    assert list(receipt["test_indices"]) == list(te_idx), \
        f"{slug}/{domain}: receipt test_indices differ from split dump test slice"
    rc_auroc = float(receipt["probe_raw"]["auroc"])
    rc_anchor = int(receipt["probe_raw"]["anchor_layer"])

    boot_rng = np.random.default_rng(seed)
    boot_idx = boot_rng.integers(0, n_test, (B_BOOT, n_test))

    _cfg = getattr(model.config, "text_config", None) or model.config  # Gemma-3 nests these
    n_layers = _cfg.num_hidden_layers + 1
    d_model = _cfg.hidden_size
    D = dom_rng.standard_normal((N_RANDOM, n_layers, d_model))
    D /= np.linalg.norm(D, axis=2, keepdims=True)
    assert np.allclose(np.linalg.norm(D, axis=2), 1.0, atol=1e-10), "null dirs not unit norm"

    per_cond = {}
    probe_test = {}
    tfidf_test = {}
    lengths = {}

    for cond in CONDITIONS:
        c_texts, v_texts = texts[cond]
        abl = lib.assert_ablated(domain, dump, c_texts + v_texts)

        lengths[cond] = {
            "complies": token_lengths(tokenizer, [c_texts[i] for i in te_idx]),
            "violates": token_lengths(tokenizer, [v_texts[i] for i in te_idx]),
            "all_slices_pooled": token_lengths(tokenizer, c_texts + v_texts),
        }

        logger.info("  [%s] extracting %d x2 ...", cond, n_use)
        c_acts = extract_activations(c_texts, model, tokenizer, max_length=MAX_LENGTH,
                                     extraction="last_token", show_progress=False)
        v_acts = extract_activations(v_texts, model, tokenizer, max_length=MAX_LENGTH,
                                     extraction="last_token", show_progress=False)

        tr_c, tr_v = c_acts[train_idx], v_acts[train_idx]
        vl_c, vl_v = c_acts[vl_idx], v_acts[vl_idx]
        te_c, te_v = c_acts[te_idx], v_acts[te_idx]

        # ── Probe, refit FRESH under this condition ──
        direction = MeanDifferenceDirection().fit(tr_c, tr_v)
        vl_curve = compute_layerwise_auroc(vl_c, vl_v, direction)
        anchor = int(np.argmax(vl_curve))
        probe_c = direction.score_at_layer(te_c, anchor)      # higher = COMPLIANT
        probe_v = direction.score_at_layer(te_v, anchor)
        a_probe = auroc_checked(-probe_v, -probe_c)               # violation orientation
        probe_test[cond] = (-np.asarray(probe_c, float), -np.asarray(probe_v, float))

        row = {"anchor_layer": anchor, "val_layer_auroc": float(vl_curve[anchor]),
               "auroc": a_probe, "n_test": n_test, "ablation_check": abl}

        # ── correct-condition receipt reproduction (record drift, do NOT abort) ──
        if cond == "correct":
            abs_diff = abs(a_probe - rc_auroc)
            row["receipt_reproduction"] = {
                "receipt_path": os.path.relpath(rc_path, REPO),
                "receipt_probe_raw_auroc": rc_auroc,
                "rederived_auroc": a_probe,
                "abs_diff": abs_diff,
                "within_1e-6": abs_diff < 1e-6,
                "receipt_anchor_layer": rc_anchor,
                "rederived_anchor_layer": anchor,
                "anchor_layer_matches": anchor == rc_anchor,
                "receipt_torch": receipt["provenance"]["torch_version"],
                "receipt_transformers": receipt["provenance"]["transformers_version"],
                "note": "soft gate: receipts written under differing transformers versions "
                        "per model; drift recorded, not aborted; verdict is env-consistent "
                        "within-run (correct vs the other 4 conditions).",
            }
            logger.info("    RECEIPT: %.10f vs %.10f (Δ=%.2e, anchor %d vs %d, match=%s)",
                        a_probe, rc_auroc, abs_diff, anchor, rc_anchor, anchor == rc_anchor)

        # ── budget-matched null on THIS condition's activations ──
        def proj(a):
            return np.einsum("nld,kld->nlk", a.astype(np.float64), D)

        p_vl_c, p_vl_v = proj(vl_c), proj(vl_v)
        p_te_c, p_te_v = proj(te_c), proj(te_v)

        def layer_aurocs(pc, pv):
            L, K = pc.shape[1], pc.shape[2]
            out = np.empty((L, K))
            for l in range(L):
                out[l] = auroc_matrix(pv[:, l, :].T, pc[:, l, :].T)
            return out

        A_val = layer_aurocs(p_vl_c, p_vl_v)
        A_test = layer_aurocs(p_te_c, p_te_v)
        l_star = np.argmax(A_val, axis=0)
        null_aurocs = A_test[l_star, np.arange(N_RANDOM)]
        null_c = np.stack([p_te_c[:, l_star[k], k] for k in range(N_RANDOM)])
        null_v = np.stack([p_te_v[:, l_star[k], k] for k in range(N_RANDOM)])
        assert np.abs(auroc_matrix(null_v, null_c) - null_aurocs).max() < 1e-9, \
            "null score/AUROC mismatch"

        null_mean = float(null_aurocs.mean())
        tail_p = float((1 + int((null_aurocs >= a_probe).sum())) / (N_RANDOM + 1))
        arm_boot = auroc_matrix(probe_test[cond][1][boot_idx], probe_test[cond][0][boot_idx])
        margins = arm_boot - null_mean
        lo, hi = np.percentile(margins, [2.5, 97.5])
        row["null"] = {
            "null_name": "probe_budget_layer_argmax_fixed_sign",
            "n_random_directions": N_RANDOM,
            "mean": null_mean, "std": float(null_aurocs.std()),
            "p05": float(np.percentile(null_aurocs, 5)),
            "median": float(np.median(null_aurocs)),
            "p95": float(np.percentile(null_aurocs, 95)),
            "max": float(null_aurocs.max()),
        }
        row["margin_over_null"] = float(a_probe - null_mean)
        row["margin_ci95"] = [float(lo), float(hi)]
        row["tail_p_vs_null"] = tail_p
        per_cond[cond] = row
        logger.info("    [%s] l*=%d AUROC=%.4f null=%.4f margin=%+.4f tail_p=%.4f",
                    cond, anchor, a_probe, null_mean, a_probe - null_mean, tail_p)

        # ── TF-IDF, same condition, same slices ──
        t_c, t_v = tfidf_scores(c_texts, v_texts, train_idx, te_idx)
        a_tf = auroc_checked(t_v, t_c)
        tfidf_test[cond] = (t_c, t_v)
        per_cond[cond]["tfidf"] = {"auroc": a_tf, "null": 0.5,
                                   "margin_over_null": float(a_tf - 0.5)}
        logger.info("    [%s] TF-IDF AUROC=%.4f", cond, a_tf)

        del c_acts, v_acts, p_vl_c, p_vl_v, p_te_c, p_te_v

    # ── paired bootstrap vs `correct`, SHARED resample indices ──
    def paired(arm, ref):
        c0, v0 = ref
        c1, v1 = arm
        from competitors_lib import auroc_checked as _ac
        b_ref = auroc_matrix(v0[boot_idx], c0[boot_idx])
        b_arm = auroc_matrix(v1[boot_idx], c1[boot_idx])
        d = b_ref - b_arm
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        p = float(min(max(p, 1.0 / B_BOOT), 1.0))
        return {"delta_correct_minus_cond": float(_ac(v0, c0) - _ac(v1, c1)),
                "delta_ci95": [float(lo), float(hi)], "p_boot": p, "B": B_BOOT,
                "resampling_unit": "test pair (shared indices across all conditions)"}

    for cond in CONDITIONS:
        per_cond[cond]["paired_vs_correct_probe"] = paired(probe_test[cond], probe_test["correct"])
        per_cond[cond]["paired_vs_correct_tfidf"] = paired(tfidf_test[cond], tfidf_test["correct"])

    return {
        "domain": domain, "seed": seed, "n_use": n_use, "n_test": n_test,
        "model": model_name, "model_slug": slug,
        "split_check": split_check, "donor_split_check": donor_split_check,
        "benchmark_structure_verified": structure,
        "token_lengths": lengths,
        "conditions": per_cond,
    }


# ── per-model driver ──────────────────────────────────────────────────────────

def run_model(model_name, doms, seed, skip_existing):
    import torch
    import transformers as _transformers
    import competitors_lib as lib

    slug = model_slug(model_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    am_check = validate_auroc_matrix(np.random.default_rng(0))

    pending = []
    for dom in doms:
        p = os.path.join(OUT_DIR, f"rule_ablation_{slug}_{dom}_seed{seed}.json")
        if skip_existing and os.path.exists(p):
            logger.info("[%s] skip %s (exists)", slug, dom)
        else:
            pending.append(dom)
    if not pending:
        logger.info("[%s] all requested domains present.", slug)
        return

    model, tokenizer = load_model(model_name)
    gpu_receipt = lib.assert_gpu0_runtime()
    logger.info("GPU: %s", gpu_receipt)

    children = np.random.SeedSequence(seed).spawn(len(ALL_SPLITS))
    try:
        for dom in pending:
            path = os.path.join(OUT_DIR, f"rule_ablation_{slug}_{dom}_seed{seed}.json")
            rng = np.random.default_rng(children[ALL_SPLITS.index(dom)])
            r = run_domain(dom, model, tokenizer, model_name, seed, rng, gpu_receipt)
            r["provenance"] = {
                "experiment": "E-RULE-MULTI — does rule-blindness replicate beyond the 1B?",
                "script": "scripts/run_rule_ablation_multi.py",
                "model_name": model_name, "model_slug": slug,
                "seed": seed, "git_commit": GIT_COMMIT,
                "timestamp": datetime.datetime.now().astimezone().isoformat(),
                "dtype": "bfloat16",
                "torch_version": torch.__version__,
                "transformers_version": _transformers.__version__,
                "numpy_version": np.__version__,
                "python": sys.executable,
                "max_length": MAX_LENGTH, "extraction": "last_token",
                "gpu_check": gpu_receipt,
                "auroc_matrix_check": am_check,
                "n_random_directions": N_RANDOM, "B_boot": B_BOOT,
                "null_rng": "np.random.default_rng(np.random.SeedSequence(42)"
                            ".spawn(20)[sorted(ALL_SPLITS).index(domain)]); one stack of 200 "
                            "unit directions per domain, SHARED across the 5 conditions",
                "receipt_source": f"results/rebuild/models/{slug}/omni_ablated_*_seed{seed}.json",
                "orientation": "Probe higher = COMPLIANT on disk; all reported AUROCs are in "
                               "VIOLATION orientation (Probe negated), matching compute_gate.py",
            }
            with open(path, "w") as f:
                json.dump(r, f, indent=2)
            logger.info("  saved → %s", path)
    finally:
        free_model(model, tokenizer)

    # per-model summary if all 20 present
    write_model_summary(slug, seed)


def write_model_summary(slug, seed):
    from competitors_lib import bh_exact, bh_sig_exact
    doms = []
    results = {}
    for dom in ALL_SPLITS:
        p = os.path.join(OUT_DIR, f"rule_ablation_{slug}_{dom}_seed{seed}.json")
        if os.path.exists(p):
            results[dom] = json.load(open(p))
            doms.append(dom)
    if len(doms) < len(ALL_SPLITS):
        logger.info("[%s] %d/20 domains — per-model summary deferred.", slug, len(doms))
        return None

    tables = _build_tables(results, doms, bh_exact, bh_sig_exact)
    summary = {
        "experiment": "E-RULE-MULTI — does rule-blindness replicate beyond the 1B?",
        "model_slug": slug,
        "model": results[doms[0]]["model"],
        "seed": seed, "git_commit": GIT_COMMIT,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "provenance": results[doms[0]]["provenance"],
        "conditions": CONDITIONS,
        "wrong_domain_donor_map": WRONG_DONOR,
        "tables": tables,
        "decisive": _decisive(tables),
        "receipt_reproduction": _receipt_rollup(results, doms),
        "length_confound": _length_rollup(results, doms),
        "per_domain": {d: results[d] for d in doms},
    }
    p = os.path.join(OUT_DIR, f"rule_ablation_{slug}_SUMMARY.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[%s] per-model summary → %s", slug, p)
    _print_model_table(slug, tables, summary["decisive"])
    return summary


def _build_tables(results, doms, bh_exact, bh_sig_exact):
    tables = {}
    for method, key in (("probe", "paired_vs_correct_probe"), ("tfidf", "paired_vs_correct_tfidf")):
        tab = {}
        for cond in CONDITIONS:
            if method == "probe":
                aur = [results[d]["conditions"][cond]["auroc"] for d in doms]
                nul = [results[d]["conditions"][cond]["null"]["mean"] for d in doms]
            else:
                aur = [results[d]["conditions"][cond]["tfidf"]["auroc"] for d in doms]
                nul = [0.5] * len(doms)
            pv = [results[d]["conditions"][cond][key]["p_boot"] for d in doms]
            dl = [results[d]["conditions"][cond][key]["delta_correct_minus_cond"] for d in doms]
            q_f, q_s, q_naive = bh_exact(pv)
            sig = bh_sig_exact(q_s)
            n_drop = int(sum(s and x > 0 for s, x in zip(sig, dl)))
            n_gain = int(sum(s and x < 0 for s, x in zip(sig, dl)))
            tab[cond] = {
                "mean_auroc": float(np.mean(aur)),
                "mean_null": float(np.mean(nul)),
                "mean_margin_over_null": float(np.mean(aur) - np.mean(nul)),
                "mean_delta_correct_minus_cond": float(np.mean(dl)),
                "n_domains_bh_sig_drop": n_drop,
                "n_domains_bh_sig_gain": n_gain,
                "n_domains": len(doms),
                "per_domain": {d: {"auroc": aur[i], "null": nul[i], "delta_vs_correct": dl[i],
                                   "ci95": results[d]["conditions"][cond][key]["delta_ci95"],
                                   "p_boot": pv[i], "q_bh_exact": q_s[i],
                                   "q_bh_float": float(q_f[i]),
                                   "q_bh_naive_float_arith": float(q_naive[i]),
                                   "bh_sig": bool(sig[i])}
                               for i, d in enumerate(doms)},
                "bh_float_vs_exact_disagreements": int(sum(
                    (Fraction(q_s[i]) < Fraction(1, 20)) != (q_naive[i] < 0.05)
                    for i in range(len(doms)))),
            }
        tables[method] = tab
    return tables


def _decisive(tables):
    d_nr = tables["probe"]["no_rule"]
    n_drop = d_nr["n_domains_bh_sig_drop"]
    md = d_nr["mean_delta_correct_minus_cond"]
    out = {
        "quantity": "AUROC(correct) - AUROC(no_rule), per domain, paired bootstrap + BH",
        "mean_delta": md,
        "n_domains_bh_sig_drop": n_drop,
        "n_domains_bh_sig_gain": d_nr["n_domains_bh_sig_gain"],
        "n_domains": d_nr["n_domains"],
    }
    if n_drop == 0:
        out["verdict"] = (f"RULE-BLINDNESS HOLDS: deleting the Rule block changes mean AUROC "
                          f"by {md:+.4f} and produces a BH-significant DROP in "
                          f"0/{d_nr['n_domains']} domains "
                          f"({d_nr['n_domains_bh_sig_gain']} significant gains). Deleting the "
                          f"rule costs nothing — the probe reads the scenario, not the rule.")
    elif n_drop >= 15:
        out["verdict"] = (f"RULE-CONDITIONING EMERGES AT SCALE: deleting the Rule block costs "
                          f"{md:+.4f} mean AUROC and produces a BH-significant drop in "
                          f"{n_drop}/{d_nr['n_domains']} domains. This larger model DOES read "
                          f"the rule — a genuine scale finding that reframes the claim.")
    else:
        out["verdict"] = (f"PARTIAL: deleting the Rule block costs {md:+.4f} mean AUROC and a "
                          f"BH-significant drop in {n_drop}/{d_nr['n_domains']} domains "
                          f"({d_nr['n_domains_bh_sig_gain']} gains). Some rule sensitivity, "
                          f"most signal scenario-borne; the claim must be hedged.")
    return out


def _receipt_rollup(results, doms):
    reps = [results[d]["conditions"]["correct"]["receipt_reproduction"] for d in doms]
    diffs = [r["abs_diff"] for r in reps]
    return {
        "n_domains": len(doms),
        "max_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
        "n_within_1e-6": int(sum(r["within_1e-6"] for r in reps)),
        "n_anchor_layer_matches": int(sum(r["anchor_layer_matches"] for r in reps)),
        "receipt_torch": reps[0]["receipt_torch"],
        "receipt_transformers": reps[0]["receipt_transformers"],
        "per_domain_abs_diff": {d: reps[i]["abs_diff"] for i, d in enumerate(doms)},
        "per_domain_anchor_match": {d: reps[i]["anchor_layer_matches"] for i, d in enumerate(doms)},
    }


def _length_rollup(results, doms):
    return {
        cond: {
            "mean_tokens_test_pooled": float(np.mean(
                [results[d]["token_lengths"][cond]["complies"]["mean"] for d in doms]
                + [results[d]["token_lengths"][cond]["violates"]["mean"] for d in doms])),
            "max_tokens_any_domain": int(max(
                results[d]["token_lengths"][cond]["all_slices_pooled"]["max"] for d in doms)),
            "n_texts_exceeding_512_all_domains": int(sum(
                results[d]["token_lengths"][cond]["all_slices_pooled"]["n_exceeding_max_length"]
                for d in doms)),
            "n_texts_all_domains": int(sum(
                results[d]["token_lengths"][cond]["all_slices_pooled"]["n_texts"] for d in doms)),
        } for cond in CONDITIONS
    }


def _print_model_table(slug, tables, decisive):
    print(f"\n=== E-RULE-MULTI [{slug}] ===")
    for method in ("probe", "tfidf"):
        print(f"\n--- {method.upper()} ---")
        print("%-20s %8s %8s %8s %9s %7s %7s" % (
            "condition", "AUROC", "null", "margin", "Δ vs corr", "#drop", "#gain"))
        for cond in CONDITIONS:
            t = tables[method][cond]
            print("%-20s %8.4f %8.4f %+8.4f %+9.4f %7d %7d" % (
                cond, t["mean_auroc"], t["mean_null"], t["mean_margin_over_null"],
                t["mean_delta_correct_minus_cond"], t["n_domains_bh_sig_drop"],
                t["n_domains_bh_sig_gain"]))
    print("\nVERDICT:", decisive["verdict"])


# ── final cross-model aggregation (CPU, env-agnostic) ─────────────────────────

def summarize(seed):
    from competitors_lib import bh_exact, bh_sig_exact
    os.makedirs(OUT_DIR, exist_ok=True)
    per_model = {}
    for slug in sorted(MODEL_SIZE_GIB, key=model_slug):
        s = model_slug(slug)
        doms = []
        results = {}
        for dom in ALL_SPLITS:
            p = os.path.join(OUT_DIR, f"rule_ablation_{s}_{dom}_seed{seed}.json")
            if os.path.exists(p):
                results[dom] = json.load(open(p))
                doms.append(dom)
        if not doms:
            continue
        tables = _build_tables(results, doms, bh_exact, bh_sig_exact)
        per_model[s] = {
            "model": results[doms[0]]["model"],
            "n_domains": len(doms),
            "provenance": results[doms[0]]["provenance"],
            "tables": tables,
            "decisive": _decisive(tables),
            "receipt_reproduction": _receipt_rollup(results, doms),
            "length_confound": _length_rollup(results, doms),
        }

    # THE TABLE: model x condition -> mean AUROC, Δ vs correct, #drops (Probe + TF-IDF)
    the_table = {}
    for s, m in per_model.items():
        the_table[s] = {method: {cond: {
            "mean_auroc": m["tables"][method][cond]["mean_auroc"],
            "delta_vs_correct": m["tables"][method][cond]["mean_delta_correct_minus_cond"],
            "n_bh_sig_drop": m["tables"][method][cond]["n_domains_bh_sig_drop"],
            "n_bh_sig_gain": m["tables"][method][cond]["n_domains_bh_sig_gain"],
        } for cond in CONDITIONS} for method in ("probe", "tfidf")}

    out = {
        "experiment": "E-RULE-MULTI — does rule-blindness replicate beyond the 1B?",
        "script": "scripts/run_rule_ablation_multi.py",
        "seed": seed, "git_commit": GIT_COMMIT,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "baseline_1b": {
            "source": "results/rebuild/rule_ablation/RULE_ABLATION_SUMMARY.json",
            "probe_no_rule_bh_sig_drops": 0, "probe_no_rule_mean_delta": -0.0029,
            "tfidf_no_rule_mean_auroc": 0.9308,
            "note": "rule deletion costs nothing on the 1B (0/20 drops).",
        },
        "conditions": CONDITIONS,
        "the_table": the_table,
        "per_model": per_model,
    }
    p = os.path.join(OUT_DIR, "RULE_ABLATION_MULTI_SUMMARY.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    # console
    print("\n" + "=" * 78)
    print("E-RULE-MULTI — THE TABLE (decisive quantity: correct vs no_rule)")
    print("=" * 78)
    print("1B baseline (reference): Probe no_rule Δ=-0.0029, 0/20 BH-sig drops.\n")
    for method in ("probe", "tfidf"):
        print(f"\n### {method.upper()}  (model × condition → meanAUROC | Δvs_correct | #drop/#gain)")
        header = "%-26s " % "model" + " ".join("%-22s" % c for c in CONDITIONS)
        print(header)
        for s, m in per_model.items():
            cells = []
            for c in CONDITIONS:
                t = m["tables"][method][c]
                cells.append("%.3f|%+.3f|%d/%d" % (
                    t["mean_auroc"], t["mean_delta_correct_minus_cond"],
                    t["n_domains_bh_sig_drop"], t["n_domains_bh_sig_gain"]))
            print("%-26s " % f"{s}({m['n_domains']}/20)" + " ".join("%-22s" % x for x in cells))
    print("\n--- VERDICTS ---")
    for s, m in per_model.items():
        d = m["decisive"]
        print(f"[{s}] no_rule Δ={d['mean_delta']:+.4f}, {d['n_domains_bh_sig_drop']}/"
              f"{d['n_domains']} BH-sig drops → {d['verdict']}")
    print("\n--- RECEIPT REPRODUCTION (correct vs stored probe_raw.auroc) ---")
    for s, m in per_model.items():
        rr = m["receipt_reproduction"]
        print(f"[{s}] max|Δ|={rr['max_abs_diff']:.2e} mean|Δ|={rr['mean_abs_diff']:.2e} "
              f"anchor-match {rr['n_anchor_layer_matches']}/{rr['n_domains']} "
              f"(receipt tf {rr['receipt_transformers']})")
    print("\nSaved →", p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None,
                    help="HF model name (one of the 3 arms). Omit with --summarize.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.seed)
        return

    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    assert args.model, "provide --model or --summarize"
    assert args.model in MODEL_SIZE_GIB, \
        f"--model must be one of {list(MODEL_SIZE_GIB)}"
    doms = ALL_SPLITS if args.all else [args.split or "gdpr"]
    run_model(args.model, doms, args.seed, args.skip_existing)


if __name__ == "__main__":
    main()
