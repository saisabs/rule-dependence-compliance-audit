"""
run_trivial_gauntlet.py — Gate experiment G2 (FINAL_PLAN §3 G2, §5 Tier-1).

Trivial-baseline gauntlet on the EXACT per-domain samples and seed-42
three_way_split used by scripts/run_omni_v1.py, for all 20 OmniCompliance
domains, CPU only (no LLM is loaded).

Input variants:
  (a) "ablated"      — Rule + Scenario only (the gate-relevant variant)
  (b) "with_outcome" — full Rule/Scenario/Outcome block (confound exhibit),
                       plus TF-IDF on the Outcome field alone.

Baselines (fit on the 40% train slice, AUROC reported on the 20% test slice;
val_layer / val_thresh slices are never touched — they exist for parity with
the activation pipeline):
  1. tfidf_full        — TF-IDF (unigram) + logistic regression on the input text
  2. tfidf_outcome     — same, on the Outcome field alone (with_outcome variant only)
  3. length_char / length_token — raw input length as score (sign oriented on train)
  4. keyword_continuous — continuous net violation-marker count using the
                          marker lists from src/compliance/guardrails_baseline.py
                          (NEVER the binarized 0/1 variant — see FINAL_PLAN §7)
  5. perm_null_tfidf   — tfidf_full with train labels shuffled (5 seeded
                          shuffles); mean AUROC should sit ~0.5 (harness check)

Label convention: 1 = VIOLATES, 0 = COMPLIES. All AUROCs are for detecting
violation. Sign-only baselines (length) orient their sign on the train slice.

Sampling parity with run_omni_v1.run_domain (do not change the order of
operations — the rng stream must match):
  dataset = load_dataset("hubin/OmniCompliance100K", split=domain)
  complies / violates filtered in dataset order
  rng = random.Random(seed); n_use = min(n_pairs, |C|, |V|)
  complies_s = rng.sample(complies, n_use); violates_s = rng.sample(violates, n_use)
  train/val_layer/val_thresh/test = splits.three_way_split(n_use, seed)

If results/rebuild/splits/<domain>_seed42_splits.json exists (written by the
GPU G1 agent), the indices are verified to match and the result is recorded.

Usage:
    python scripts/run_trivial_gauntlet.py            # all 20 domains
    python scripts/run_trivial_gauntlet.py --split gdpr

Outputs:
    results/rebuild/gauntlet/gauntlet_<domain>_seed42.json
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import numpy as np
import sklearn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compliance.splits import three_way_split
from src.compliance.guardrails_baseline import COMPLIANCE_MARKERS, VIOLATION_MARKERS

# Mirror of run_omni_v1.ALL_SPLITS (excludes policy_wechat per research plan)
ALL_SPLITS = [
    "gdpr",
    "hipaa",
    "eu_ai_act",
    "ccpa",
    "sb35",
    "finance_crypto",
    "finance_anti_laundering_and_terrorist",
    "finance_cross_border_payment_law",
    "finance_eletric_momey",
    "data_act",
    "cybersecurity_mitre_attack",
    "edu_academic_integrity",
    "edu_online_learning",
    "edu_discrimination_us_edu_dept",
    "policy_github",
    "policy_google",
    "policy_openai",
    "policy_reddit",
    "policy_x",
    "foundation_rights",
]

N_PERM_SHUFFLES = 5


def format_case_with_outcome(ex: dict, rule_chars: int = 300) -> str:
    """Identical to run_omni_v1.format_case / omnicompliance.format_case."""
    return (
        f"Rule: {ex['source_rule'][:rule_chars]}\n\n"
        f"Scenario: {ex['example_background']}\n\n"
        f"Outcome: {ex['process_and_outcome']}"
    )


def format_case_ablated(ex: dict, rule_chars: int = 300) -> str:
    """Rule + Scenario only — the outcome-ablated gate variant."""
    return (
        f"Rule: {ex['source_rule'][:rule_chars]}\n\n"
        f"Scenario: {ex['example_background']}"
    )


def keyword_score(text: str) -> float:
    """Continuous net violation-marker score (presence counts, matching the
    marker-matching semantics of guardrails_baseline.classify_guardrails,
    but WITHOUT the binarization step)."""
    tl = text.lower()
    v = sum(1 for m in VIOLATION_MARKERS if m in tl)
    c = sum(1 for m in COMPLIANCE_MARKERS if m in tl)
    return float(v - c)


def _auroc(labels, scores) -> float:
    return float(roc_auc_score(labels, scores))


def tfidf_logreg_auroc(train_texts, train_labels, test_texts, test_labels,
                       clf_seed: int = 0):
    """TF-IDF (unigram) + logistic regression; returns (test AUROC, test P(violates))."""
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 1)),
        LogisticRegression(max_iter=1000, random_state=clf_seed),
    )
    pipe.fit(train_texts, train_labels)
    proba = pipe.predict_proba(test_texts)[:, list(pipe.classes_).index(1)]
    return _auroc(test_labels, proba), proba


def raw_score_auroc(train_scores, train_labels, test_scores, test_labels):
    """AUROC of a raw score whose sign is oriented on the train slice."""
    train_auc = _auroc(train_labels, train_scores)
    sign = 1.0 if train_auc >= 0.5 else -1.0
    oriented = sign * np.asarray(test_scores, dtype=float)
    return _auroc(test_labels, oriented), oriented, sign, float(train_auc)


def run_variant(variant_name, texts_c, texts_v, train_idx, test_idx,
                outcome_c=None, outcome_v=None):
    """Run all baselines for one input variant. texts_c/texts_v are the full
    per-class sampled lists; the same index sets slice both classes (paired
    sampling, exactly as run_omni_v1 slices activations)."""
    train_texts = [texts_c[i] for i in train_idx] + [texts_v[i] for i in train_idx]
    train_labels = [0] * len(train_idx) + [1] * len(train_idx)
    test_texts = [texts_c[i] for i in test_idx] + [texts_v[i] for i in test_idx]
    test_labels = [0] * len(test_idx) + [1] * len(test_idx)

    out = {"variant": variant_name, "baselines": {}}

    # 1. TF-IDF + logreg on the full input text
    auc, scores = tfidf_logreg_auroc(train_texts, train_labels, test_texts, test_labels)
    out["baselines"]["tfidf_full"] = {
        "auroc": auc,
        "test_scores": {"compliant": scores[: len(test_idx)].tolist(),
                        "violating": scores[len(test_idx):].tolist()},
    }

    # 2. TF-IDF + logreg on the Outcome field alone (with_outcome variant only)
    if outcome_c is not None:
        tr_o = [outcome_c[i] for i in train_idx] + [outcome_v[i] for i in train_idx]
        te_o = [outcome_c[i] for i in test_idx] + [outcome_v[i] for i in test_idx]
        auc, scores = tfidf_logreg_auroc(tr_o, train_labels, te_o, test_labels)
        out["baselines"]["tfidf_outcome"] = {
            "auroc": auc,
            "test_scores": {"compliant": scores[: len(test_idx)].tolist(),
                            "violating": scores[len(test_idx):].tolist()},
        }

    # 3. Length raw scores (character and whitespace-token), sign fit on train
    for key, fn in [("length_char", len), ("length_token", lambda t: len(t.split()))]:
        tr_s = [fn(t) for t in train_texts]
        te_s = [fn(t) for t in test_texts]
        auc, oriented, sign, train_auc = raw_score_auroc(tr_s, train_labels, te_s, test_labels)
        out["baselines"][key] = {
            "auroc": auc,
            "train_auroc": train_auc,
            "sign_from_train": sign,
            "test_scores": {"compliant": oriented[: len(test_idx)].tolist(),
                            "violating": oriented[len(test_idx):].tolist()},
        }

    # 4. Continuous keyword score (net violation-marker count; semantic
    #    direction is fixed by the lexicon — no sign fitting, no binarization)
    te_s = np.array([keyword_score(t) for t in test_texts], dtype=float)
    tr_s = [keyword_score(t) for t in train_texts]
    out["baselines"]["keyword_continuous"] = {
        "auroc": _auroc(test_labels, te_s),
        "train_auroc": _auroc(train_labels, tr_s),
        "test_scores": {"compliant": te_s[: len(test_idx)].tolist(),
                        "violating": te_s[len(test_idx):].tolist()},
    }

    # 5. Permutation-null TF-IDF: shuffle TRAIN labels (seeded), 5 shuffles
    perm_aucs = []
    for k in range(N_PERM_SHUFFLES):
        perm = np.random.RandomState(1000 + k).permutation(train_labels).tolist()
        auc, _ = tfidf_logreg_auroc(train_texts, perm, test_texts, test_labels,
                                    clf_seed=1000 + k)
        perm_aucs.append(auc)
    out["baselines"]["perm_null_tfidf"] = {
        "auroc": float(np.mean(perm_aucs)),
        "per_shuffle_aurocs": perm_aucs,
        "shuffle_seeds": [1000 + k for k in range(N_PERM_SHUFFLES)],
    }

    return out


def run_domain(domain, n_pairs, seed, results_dir, splits_dir, sentiment_by_domain):
    # ---- sampling + split: EXACT mirror of run_omni_v1.run_domain ----
    dataset = load_dataset("hubin/OmniCompliance100K", split=domain)
    complies = [ex for ex in dataset if ex["relation_to_rule"] == "COMPLIES"]
    violates = [ex for ex in dataset if ex["relation_to_rule"] == "VIOLATES"]

    rng = random.Random(seed)
    n_use = min(n_pairs, len(complies), len(violates))
    complies_s = rng.sample(complies, n_use)
    violates_s = rng.sample(violates, n_use)

    train_idx, val_layer_idx, val_thresh_idx, test_idx = three_way_split(n_use, seed)

    # Cross-check against a splits dump from the GPU G1 agent, if present.
    # The dump nests the indices under "split_indices"; fall back to top-level
    # keys for other layouts. A dump whose slices we cannot locate is reported
    # as an explicit error, never as a silent pass.
    split_check = {"dump_found": False}
    dump_path = os.path.join(splits_dir, f"{domain}_seed{seed}_splits.json")
    if os.path.exists(dump_path):
        with open(dump_path) as f:
            dump = json.load(f)
        dumped = dump.get("split_indices", dump)
        ours = {"train": train_idx, "val_layer": val_layer_idx,
                "val_thresh": val_thresh_idx, "test": test_idx}
        found = [k for k in ours if k in dumped]
        if not found:
            split_check = {"dump_found": True, "dump_path": dump_path,
                           "match": None,
                           "error": "no split slices found in dump; layout unrecognised",
                           "dump_keys": list(dump.keys())}
        else:
            mism = {k: (list(dumped[k]) != list(ours[k])) for k in found}
            split_check = {"dump_found": True, "dump_path": dump_path,
                           "slices_compared": found,
                           "match": (not any(mism.values())) and len(found) == len(ours),
                           "per_slice_mismatch": mism,
                           "n_use_match": dump.get("n_use") == n_use}
            # Text-level parity: the dump stores the ablated case text per
            # sampled case, so verify our sampling picked the same cases.
            for cls, cases in (("complies_cases", complies_s), ("violates_cases", violates_s)):
                dumped_cases = dump.get(cls)
                if isinstance(dumped_cases, list) and dumped_cases \
                        and "text_ablated" in dumped_cases[0]:
                    ours_txt = [format_case_ablated(ex) for ex in cases]
                    theirs = [c["text_ablated"] for c in dumped_cases]
                    split_check[f"{cls}_text_match"] = (ours_txt == theirs)

    # ---- input variants ----
    texts = {
        "ablated": ([format_case_ablated(ex) for ex in complies_s],
                    [format_case_ablated(ex) for ex in violates_s]),
        "with_outcome": ([format_case_with_outcome(ex) for ex in complies_s],
                         [format_case_with_outcome(ex) for ex in violates_s]),
    }
    outcome_c = [ex["process_and_outcome"] for ex in complies_s]
    outcome_v = [ex["process_and_outcome"] for ex in violates_s]

    variants = {}
    for vname, (tc, tv) in texts.items():
        variants[vname] = run_variant(
            vname, tc, tv, train_idx, test_idx,
            outcome_c=outcome_c if vname == "with_outcome" else None,
            outcome_v=outcome_v if vname == "with_outcome" else None,
        )

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()
    except Exception:
        commit = "unknown"

    result = {
        "domain": domain,
        "n_pairs": n_use,
        "label_convention": "1=VIOLATES, 0=COMPLIES; AUROC detects violation",
        "split_sizes": {"train": len(train_idx), "val_layer": len(val_layer_idx),
                        "val_thresh": len(val_thresh_idx), "test": len(test_idx)},
        # Case counts: each split index selects one COMPLIES and one VIOLATES
        # case (paired sampling), so every slice is exactly 50/50 by construction.
        "n_cases": {
            "n_train": 2 * len(train_idx),
            "n_val_layer": 2 * len(val_layer_idx),
            "n_val_thresh": 2 * len(val_thresh_idx),
            "n_test": 2 * len(test_idx),
        },
        "class_balance": {
            "n_test": 2 * len(test_idx),
            "n_test_complies": len(test_idx),
            "n_test_violates": len(test_idx),
            "test_frac_violates": 0.5,
            "balanced_by_construction": True,
            "n_train": 2 * len(train_idx),
            "n_train_complies": len(train_idx),
            "n_train_violates": len(train_idx),
        },
        # Full source pool before the balanced n_pairs cap (context for the
        # small-n degeneracy analysis: how much of the pool the cap discards).
        "source_pool": {
            "n_complies": len(complies),
            "n_violates": len(violates),
            "n_pairs_cap": n_pairs,
            "n_use": n_use,
            "capped_by_pool": n_use < n_pairs,
        },
        "train_idx": train_idx,
        "test_idx": test_idx,
        "split_dump_check": split_check,
        "variants": variants,
        "sentiment_sst2_auroc_from_disk": sentiment_by_domain.get(domain),
        "provenance": {
            "seed": seed,
            "git_commit": commit,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "script": "scripts/run_trivial_gauntlet.py",
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "variants": list(variants.keys()),
            "n_perm_shuffles": N_PERM_SHUFFLES,
            "compute": "CPU only; no LLM loaded",
        },
    }

    out_dir = os.path.join(results_dir, "rebuild", "gauntlet")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gauntlet_{domain}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    b_abl = {k: v["auroc"] for k, v in variants["ablated"]["baselines"].items()}
    b_out = {k: v["auroc"] for k, v in variants["with_outcome"]["baselines"].items()}
    print(f"{domain:<42} n={n_use:>3}  "
          f"abl[tfidf={b_abl['tfidf_full']:.3f} len={max(b_abl['length_char'], b_abl['length_token']):.3f} "
          f"kw={b_abl['keyword_continuous']:.3f} perm={b_abl['perm_null_tfidf']:.3f}]  "
          f"out[tfidf={b_out['tfidf_full']:.3f} outc={b_out['tfidf_outcome']:.3f} "
          f"perm={b_out['perm_null_tfidf']:.3f}]")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default=None,
                        help="Single domain (default: all 20)")
    parser.add_argument("--n_pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--splits_dir", type=str, default="results/rebuild/splits")
    parser.add_argument("--sentiment_json", type=str,
                        default="results/sentiment_baseline_check.json")
    args = parser.parse_args()

    sentiment_by_domain = {}
    if os.path.exists(args.sentiment_json):
        with open(args.sentiment_json) as f:
            for row in json.load(f):
                sentiment_by_domain[row["dataset"]] = row.get("sst2_direction_auroc")

    domains = [args.split] if args.split else ALL_SPLITS
    for i, d in enumerate(domains, 1):
        print(f"[{i}/{len(domains)}] {d}", file=sys.stderr)
        run_domain(d, args.n_pairs, args.seed, args.results_dir,
                   args.splits_dir, sentiment_by_domain)


if __name__ == "__main__":
    main()
