"""
run_rule_ablation.py — E-RULE: does Probe need the rule at all?

MOTIVATION
----------
E-COMP-3's policy-ablation control on Latent Policy Guard
(`results/rebuild/competitors/LPG_POLICY_CONTROL.json`) found LPG is POLICY-INSENSITIVE on
OmniCompliance: correct rule 0.9214 / shuffled 0.9005 / wrong-domain 0.8999 / **no rule at
all 0.9249**. Deleting the rule slightly HELPS. So the label is ~0.92 recoverable from the
scenario alone.

We have never tested whether Probe needs the rule. The paper's central claim is that the probe
reads "does this scenario violate THIS rule". If Probe scores the same with the rule stripped,
the probe reads "does this text describe something bad" — a generic violation-description
axis — and the paper's central object is misnamed. `results/rebuild/geometry_ablated/
SUMMARY.md` called this ambiguity *undecidable within a single-template benchmark*. LPG's
control shows how to decide it; the same design applies here.

EITHER ANSWER IS A MAJOR RESULT. The script does not root for one: it reports signed deltas
in BOTH directions (#domains where correct > no_rule significantly, AND #domains where
no_rule > correct significantly).

BENCHMARK STRUCTURE (verified here, not assumed)
------------------------------------------------
Each case carries its OWN rule (gdpr: 87 distinct rules across 100 test cases), and
complies_cases[i] / violates_cases[i] essentially never share a rule (0/100 on gdpr,
6/1723 pooled). So the rule genuinely varies per example and the rule<->scenario
CORRESPONDENCE is the thing to attack — which forces a within-domain DERANGEMENT rather than
naive shuffling. Both facts are asserted per domain; a violation STOPs the run.

CONDITIONS (input transforms; the probe is refit FRESH under each)
------------------------------------------------------------------
  correct           the published `text_ablated` = "Rule: {r}\\n\\nScenario: {s}".
                    MUST reproduce GATE.json to <1e-9 or the run STOPs.
  no_rule           "Scenario: {s}"                — Rule block removed entirely.
  shuffled_rule     "Rule: {r'}\\n\\nScenario: {s}" where r' is another case's rule from the
                    SAME domain and SAME pool, derangement-permuted over the full n_use pool
                    (identical rule distribution, correspondence destroyed — tightest control).
  wrong_domain_rule "Rule: {r_donor}\\n\\nScenario: {s}" — rule from a different domain,
                    same donor rule for the complies and violates case at each index, so the
                    rule carries ZERO class information.
  rule_only         "Rule: {r}"                    — Scenario removed. The complement: is the
                    label recoverable from the rule alone? (SafePyramid's was, at 0.90+.)

Probe is refit FRESH under each condition — direction on that condition's train slice, anchor
layer by argmax on that condition's val_layer, evaluated on test — because the input
distribution changes and the receipted direction would not be the direction the pipeline
would have chosen. Only `correct` is expected to reproduce the receipts.

INFERENCE
---------
  * budget-matched null: 200 isotropic random unit directions pushed through Probe's OWN
    selection budget (argmax layer on val_layer, sign fixed a priori), recomputed on EACH
    condition's own activations — the input distribution changes, so the selection floor may
    too. Protocol from results/rebuild/selection_nulls/. One stack per domain, SHARED across
    conditions, so the nulls are themselves comparable.
        rng = np.random.default_rng(np.random.SeedSequence(42).spawn(20)[i]),
        i = index of the domain in sorted(ALL_SPLITS).
  * margin over that null, with a paired bootstrap CI over test pairs, plus a tail p
    (1 + #{null >= arm}) / (N + 1) — where the arm sits in its own null distribution.
  * THE DECISIVE QUANTITY: paired bootstrap of AUROC(correct) - AUROC(condition). Pair = test
    pair, B=2000, and the B resample index sets are drawn ONCE per domain and SHARED across
    every condition and both methods — that is what makes the deltas paired.
  * BH across the 20 domains in exact fractions.Fraction arithmetic (a gate verdict once
    flipped on 0.03*20/12 == 0.049999999999999996).

TF-IDF runs the identical 5 conditions (CPU) — it is the reference for "what the text surface
alone gives you", and its no_rule number is the direct comparator for Probe's. TF-IDF performs
no layer/direction selection, so its null is 0.5 by construction.

CONVENTIONS (non-negotiable; each has cost a re-run before)
-----------------------------------------------------------
  * Probe: higher = MORE COMPLIANT on disk. Every reported AUROC is in VIOLATION orientation
    (Probe negated), matching compute_gate.py.
  * Every point-estimate AUROC goes through auroc_checked() — asserted vs sklearn to 1e-9.
  * Ablated asserted in EVERY condition: no 'Outcome:' in any scored string.
  * max_length=512, last-token — gate-matched. Removing the Rule block SHORTENS texts, so
    per-condition token-length distributions AND the fraction exceeding the 512 cap are
    recorded: if `correct` were truncated, stripping the rule would un-truncate the scenario
    and manufacture a null result mechanically. That artifact must be ruled out, not assumed.
  * GPU 0 ONLY (the shell presets CUDA_VISIBLE_DEVICES=1); forced before torch import and the
    PID's physical GPU UUID asserted after model load.

Usage
-----
    python scripts/run_rule_ablation.py --all

Outputs
-------
    results/rebuild/rule_ablation/rule_ablation_{domain}_seed42.json
    results/rebuild/rule_ablation/RULE_ABLATION_SUMMARY.json
"""

from __future__ import annotations

import os

# GPU 0 ONLY. The shell presets CUDA_VISIBLE_DEVICES=1 (harness-injected), so a plain
# `python foo.py` silently lands on GPU 1, which is forbidden. Set BEFORE torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("MECHCOMPLIANCE_PHYSICAL_GPU", "0")

import argparse
import datetime
import json
import logging
import random
import subprocess
import sys
from fractions import Fraction

import numpy as np
import torch
import transformers as _transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.compliance.activations import extract_activations
from src.compliance.direction import MeanDifferenceDirection
from src.compliance.metrics import compute_layerwise_auroc
from src.compliance.splits import three_way_split

import competitors_lib as lib
from competitors_lib import auroc_checked, bh_exact, bh_sig_exact

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rule-ablation")

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LENGTH = 512
DEFAULT_SEED = 42
N_RANDOM = 200          # >= 100 required; 200 matches results/rebuild/selection_nulls/
B_BOOT = 2000
RESULTS_DIR = os.path.join(REPO, "results", "rebuild")
OUT_DIR = os.path.join(RESULTS_DIR, "rule_ablation")

ALL_SPLITS = sorted([
    "gdpr", "hipaa", "eu_ai_act", "ccpa", "sb35",
    "finance_crypto", "finance_anti_laundering_and_terrorist",
    "finance_cross_border_payment_law", "finance_eletric_momey", "data_act",
    "cybersecurity_mitre_attack", "edu_academic_integrity", "edu_online_learning",
    "edu_discrimination_us_edu_dept", "policy_github", "policy_google",
    "policy_openai", "policy_reddit", "policy_x", "foundation_rights",
])

CONDITIONS = ["correct", "no_rule", "shuffled_rule", "wrong_domain_rule", "rule_only"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


GIT_COMMIT = git_commit()


# ── text surgery ──────────────────────────────────────────────────────────────

def parse_ablated(text: str):
    """Split the published ablated text into (rule, scenario).

    Format is fixed by run_omni_v1_ablated.format_case_ablated:
        "Rule: {source_rule[:300]}\\n\\nScenario: {example_background}"
    """
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
    """Permute so NO case keeps its own rule. Scenarios and labels untouched.

    Mirrors run_lpg_policy_control.derange: try random permutations, fall back to a
    rotation if some rule dominates the pool.
    """
    rng = np.random.default_rng(seed)
    n = len(rules)
    for _ in range(1000):
        p = rng.permutation(n)
        if all(rules[p[i]] != rules[i] for i in range(n)):
            return [rules[p[i]] for i in range(n)], True
    perm = [rules[(i + 1) % n] for i in range(n)]
    return perm, all(perm[i] != rules[i] for i in range(n)) and len(set(rules)) > 1


def _family(domain: str) -> str:
    """Coarse topical family, so a 'wrong domain' donor is genuinely off-topic."""
    for pref in ("finance_", "edu_", "policy_"):
        if domain.startswith(pref):
            return pref.rstrip("_")
    return domain


def wrong_domain_map(seed=DEFAULT_SEED):
    """Deterministic donor assignment: a derangement of the domain list under the
    constraint that the donor is from a different topical family (so finance_crypto never
    donates to finance_eletric_momey). Fixed seed; recorded in the receipts."""
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

    The rule-substitution maps are defined ONCE over the full pool and only THEN sliced by
    the split, so a condition is a single well-defined input transform of the domain (not a
    per-slice recipe). This leaks nothing: under `shuffled_rule` the rule is noise with
    respect to the label, so a train case carrying a rule later assigned to a test case
    conveys no label information.
    """
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

    # round-trip: our surgery is exactly the inverse of format_case_ablated
    assert all(assemble(r, s) == t for (r, s), t in zip(c_parsed, c_raw))
    assert all(assemble(r, s) == t for (r, s), t in zip(v_parsed, v_raw))

    # ── benchmark structure, ASSERTED (E-COMP-3's discovery; STOP if it does not hold) ──
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

    # ── shuffled: within-domain, within-pool derangement over the full pool ──
    c_shuf, c_ok = derange(c_rules, seed)
    v_shuf, v_ok = derange(v_rules, seed + 1)
    assert c_ok and v_ok, f"{domain}: derangement incomplete"
    assert all(a != b for a, b in zip(c_shuf, c_rules)), f"{domain}: shuffled kept own rule (C)"
    assert all(a != b for a, b in zip(v_shuf, v_rules)), f"{domain}: shuffled kept own rule (V)"
    assert sorted(c_shuf) == sorted(c_rules), f"{domain}: shuffled changed the rule multiset (C)"
    assert sorted(v_shuf) == sorted(v_rules), f"{domain}: shuffled changed the rule multiset (V)"

    # ── wrong domain: donor's complies-pool rules, cycled to length, for BOTH classes ──
    # (mirrors run_lpg_policy_control: the same donor rule at index i for the compliant and
    #  the violating case, so the rule carries ZERO class information.)
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
    # `correct` must be the published strings, untouched
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
    """PRE-truncation token counts + how many would hit the 512 cap."""
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


# ── AUROC helpers ─────────────────────────────────────────────────────────────

def auroc_boot(pos, neg) -> float:
    """Unchecked fast tie-averaged AUROC for bootstrap inner loops."""
    return lib.auroc_boot(pos, neg)


def auroc_matrix(pos, neg):
    """Vectorised tie-averaged AUROC over rows. pos/neg: (B, n) -> (B,).
    Lifted verbatim from run_selection_nulls.py (validated against sklearn to 1e-12)."""
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


# ── TF-IDF arm ────────────────────────────────────────────────────────────────

def tfidf_scores(c_texts, v_texts, train_idx, test_idx):
    """Byte-identical recipe to run_selection_nulls.tfidf_lexical_scores /
    run_trivial_gauntlet.py: word 1-grams + LogisticRegression, fit on train, applied to
    test. Label 1 = VIOLATES, so decision_function is already violation-oriented."""
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


# ── per-domain ────────────────────────────────────────────────────────────────

def run_domain(domain, model, tokenizer, seed, dom_rng):
    logger.info("=== E-RULE: %s ===", domain)
    dump, split_check = lib.load_split_dump(domain, seed)
    donor = WRONG_DONOR[domain]
    donor_dump, donor_split_check = lib.load_split_dump(donor, seed)

    texts, structure = build_condition_texts(dump, donor_dump, domain, seed)

    idx = dump["split_indices"]
    train_idx, vl_idx, vt_idx, te_idx = (idx["train"], idx["val_layer"],
                                         idx["val_thresh"], idx["test"])
    n_use, n_test = dump["n_use"], len(te_idx)
    assert sorted(set(train_idx) | set(vl_idx) | set(vt_idx) | set(te_idx)) == list(range(n_use))

    # ── shared bootstrap resample indices: drawn ONCE, reused by every condition and
    #    both methods. This is what makes every delta PAIRED. ──
    boot_rng = np.random.default_rng(seed)
    boot_idx = boot_rng.integers(0, n_test, (B_BOOT, n_test))

    # ── one random stack per domain, SHARED across conditions ──
    n_layers = model.config.num_hidden_layers + 1
    d_model = model.config.hidden_size
    D = dom_rng.standard_normal((N_RANDOM, n_layers, d_model))
    D /= np.linalg.norm(D, axis=2, keepdims=True)
    assert np.allclose(np.linalg.norm(D, axis=2), 1.0, atol=1e-10), "null dirs not unit norm"

    per_cond = {}
    probe_test = {}     # condition -> (c_scores, v_scores) in VIOLATION orientation
    tfidf_test = {}
    lengths = {}

    for cond in CONDITIONS:
        c_texts, v_texts = texts[cond]
        # ABLATION ASSERT, in every condition
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

        # ── step-1 gate: `correct` must reproduce GATE.json ──
        if cond == "correct":
            gate = json.load(open(os.path.join(RESULTS_DIR, "GATE.json")))
            g = next(r for r in gate if r["domain"] == domain)
            ab = json.load(open(os.path.join(RESULTS_DIR, "ablated",
                                             f"omni_ablated_{domain}_seed{seed}.json")))
            row["gate_reproduction"] = {
                "gate_json_auroc": float(g["probe"]),
                "rederived_auroc": a_probe,
                "abs_diff": abs(a_probe - float(g["probe"])),
                "bit_exact": abs(a_probe - float(g["probe"])) == 0.0,
                "disk_anchor_layer": ab["anchor_layer"],
                "rederived_anchor_layer": anchor,
                "anchor_layer_matches": anchor == ab["anchor_layer"],
            }
            assert row["gate_reproduction"]["anchor_layer_matches"], \
                f"{domain}: anchor layer drift vs receipt: {row['gate_reproduction']}"
            assert row["gate_reproduction"]["abs_diff"] < 1e-9, \
                f"{domain}: `correct` does not reproduce GATE.json — STOP: {row['gate_reproduction']}"
            logger.info("    GATE: %.10f vs %.10f (Δ=%.2e, bit-exact=%s)",
                        a_probe, float(g["probe"]), row["gate_reproduction"]["abs_diff"],
                        row["gate_reproduction"]["bit_exact"])

        # ── budget-matched null on THIS condition's activations ──
        def proj(a):     # (n, L, d) x (K, L, d) -> (n, L, K); higher = violation as sampled
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
        l_star = np.argmax(A_val, axis=0)                       # argmax layer, sign fixed
        null_aurocs = A_test[l_star, np.arange(N_RANDOM)]
        null_c = np.stack([p_te_c[:, l_star[k], k] for k in range(N_RANDOM)])
        null_v = np.stack([p_te_v[:, l_star[k], k] for k in range(N_RANDOM)])
        assert np.abs(auroc_matrix(null_v, null_c) - null_aurocs).max() < 1e-9, \
            "null score/AUROC mismatch"

        null_mean = float(null_aurocs.mean())
        tail_p = float((1 + int((null_aurocs >= a_probe).sum())) / (N_RANDOM + 1))
        # margin CI: the random-direction set and its selections are held FIXED, test pairs
        # resampled with the shared indices — this captures test-sampling noise only.
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

    # ── paired bootstrap vs `correct`, SHARED resample indices across conditions ──
    def paired(arm, ref):
        c0, v0 = ref
        c1, v1 = arm
        b_ref = auroc_matrix(v0[boot_idx], c0[boot_idx])
        b_arm = auroc_matrix(v1[boot_idx], c1[boot_idx])
        d = b_ref - b_arm
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        p = float(min(max(p, 1.0 / B_BOOT), 1.0))
        return {"delta_correct_minus_cond": float(auroc_checked(v0, c0) - auroc_checked(v1, c1)),
                "delta_ci95": [float(lo), float(hi)], "p_boot": p, "B": B_BOOT,
                "resampling_unit": "test pair (shared indices across all conditions)"}

    for cond in CONDITIONS:
        per_cond[cond]["paired_vs_correct_probe"] = paired(probe_test[cond], probe_test["correct"])
        per_cond[cond]["paired_vs_correct_tfidf"] = paired(tfidf_test[cond], tfidf_test["correct"])

    return {
        "domain": domain, "seed": seed, "n_use": n_use, "n_test": n_test,
        "split_check": split_check, "donor_split_check": donor_split_check,
        "benchmark_structure_verified": structure,
        "token_lengths": lengths,
        "conditions": per_cond,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    am_check = validate_auroc_matrix(np.random.default_rng(0))

    doms = ALL_SPLITS if args.all else [args.split or "gdpr"]

    logger.info("Loading %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    gpu_receipt = lib.assert_gpu0_runtime()
    logger.info("GPU: %s", gpu_receipt)

    # per-domain child streams; re-instantiating default_rng(42) per domain would draw the
    # IDENTICAL vectors in every domain and destroy cross-domain independence.
    children = np.random.SeedSequence(args.seed).spawn(len(ALL_SPLITS))

    results = {}
    for dom in doms:
        path = os.path.join(OUT_DIR, f"rule_ablation_{dom}_seed{args.seed}.json")
        if args.skip_existing and os.path.exists(path):
            logger.info("skip %s (exists)", dom)
            results[dom] = json.load(open(path))
            continue
        rng = np.random.default_rng(children[ALL_SPLITS.index(dom)])
        r = run_domain(dom, model, tokenizer, args.seed, rng)
        r["provenance"] = {
            "experiment": "E-RULE — does Probe need the rule at all?",
            "script": "scripts/run_rule_ablation.py",
            "model_name": MODEL_NAME, "seed": args.seed, "git_commit": GIT_COMMIT,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
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
            "orientation": "Probe higher = COMPLIANT on disk; all reported AUROCs are in "
                           "VIOLATION orientation (Probe negated), matching compute_gate.py",
        }
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        logger.info("  saved → %s", path)
        results[dom] = r

    if len(results) == len(ALL_SPLITS):
        summarize(results, args.seed, gpu_receipt, am_check)


def summarize(results, seed, gpu_receipt, am_check):
    doms = [d for d in ALL_SPLITS if d in results]
    summary = {
        "experiment": "E-RULE — does Probe need the rule at all?",
        "why": "E-COMP-3 showed Latent Policy Guard is POLICY-INSENSITIVE on OmniCompliance "
               "(no rule at all: 0.9249 > correct rule: 0.9214). The label is ~0.92 "
               "recoverable from the scenario alone. Probe had never been tested the same way. "
               "If Probe scores the same with the rule stripped, the probe reads 'does this "
               "text describe something bad', not 'does this scenario violate THIS rule', "
               "and the paper's central object is misnamed.",
        "design": "Mirrors scripts/run_lpg_policy_control.py. Probe is refit FRESH under each "
                  "condition (direction on that condition's train slice, anchor layer by "
                  "argmax on val_layer, evaluated on test) because the input distribution "
                  "changes; the receipted direction is NOT reused.",
        "script": "scripts/run_rule_ablation.py",
        "model": MODEL_NAME, "seed": seed, "git_commit": GIT_COMMIT,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "gpu_check": gpu_receipt, "auroc_matrix_check": am_check,
        "conditions": {
            "correct": "the published text_ablated (Rule + Scenario); reproduces GATE.json",
            "no_rule": "Scenario only; the Rule block removed entirely",
            "shuffled_rule": "rule replaced by another case's rule from the SAME domain and "
                             "pool (derangement over the full n_use pool, seed 42/43)",
            "wrong_domain_rule": "rule replaced by a rule from a different (family-disjoint) "
                                 "domain; the same donor rule for both classes at each index, "
                                 "so the rule carries zero class information",
            "rule_only": "Rule block only; the Scenario removed",
        },
        "wrong_domain_donor_map": WRONG_DONOR,
        "per_domain": {d: results[d] for d in doms},
    }

    # BH across the 20 domains, per condition, on the paired-vs-correct p values
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
            # signed: a "drop" means correct > cond; a "gain" means cond > correct.
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
    summary["tables"] = tables

    # THE decisive quantity
    d_nr = tables["probe"]["no_rule"]
    summary["decisive"] = {
        "quantity": "AUROC(correct) - AUROC(no_rule), per domain, paired bootstrap + BH",
        "mean_delta": d_nr["mean_delta_correct_minus_cond"],
        "n_domains_bh_sig_drop": d_nr["n_domains_bh_sig_drop"],
        "n_domains_bh_sig_gain": d_nr["n_domains_bh_sig_gain"],
        "n_domains": d_nr["n_domains"],
    }
    n_drop = d_nr["n_domains_bh_sig_drop"]
    md = d_nr["mean_delta_correct_minus_cond"]
    if n_drop == 0:
        verdict = (f"Probe DOES NOT NEED THE RULE: deleting the Rule block changes mean AUROC by "
                   f"{md:+.4f} and produces a BH-significant DROP in 0/{d_nr['n_domains']} "
                   f"domains ({d_nr['n_domains_bh_sig_gain']} significant GAINS). The probe "
                   f"reads the scenario, not the rule — 'Activation Probe Score' is a "
                   f"misnomer for a generic violation-description axis.")
    elif n_drop >= 15:
        verdict = (f"Probe NEEDS THE RULE: deleting the Rule block costs {md:+.4f} mean AUROC and "
                   f"produces a BH-significant drop in {n_drop}/{d_nr['n_domains']} domains. The "
                   f"probe genuinely conditions on the supplied policy.")
    else:
        verdict = (f"PARTIAL: deleting the Rule block costs {md:+.4f} mean AUROC and produces a "
                   f"BH-significant drop in {n_drop}/{d_nr['n_domains']} domains "
                   f"({d_nr['n_domains_bh_sig_gain']} significant gains). Some rule "
                   f"sensitivity, but most of the signal is scenario-borne; the claim must be "
                   f"hedged accordingly.")
    summary["verdict"] = verdict

    # length confound roll-up
    summary["length_confound"] = {
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

    p = os.path.join(OUT_DIR, "RULE_ABLATION_SUMMARY.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== E-RULE — does Probe need the rule? ===")
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
    print("\nVERDICT:", verdict)
    print("Saved →", p)


if __name__ == "__main__":
    main()
