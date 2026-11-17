#!/usr/bin/env python
"""
E-B2 — Second-benchmark degeneracy test: the trivial gauntlet.

Runs the trivial-baseline gauntlet on two INDEPENDENT compliance benchmarks to
test whether the OmniCompliance degeneracy (TF-IDF 0.896 mean AUROC) is a
property of that benchmark or of the compliance-detection task as posed.

  1. DynaBench   (montehoover/DynaBench, config DynaBench, split test)
                 — the benchmark used by Rachmil et al. (arXiv:2512.03994),
                   who report 86.0% F1 with NO lexical baseline of any kind.
                 — unit = row; binary label PASS/FAIL; n=543.
  2. SafePyramid (ByteDance/SafePyramid, arXiv:2606.29887)
                 — 1000 conversations x 3 levels (L0/L1/L2), 10 domains.
                 — EVERY row has >=1 violation, so an "any-violation" binary
                   label has a 100% base rate and AUROC is UNDEFINED. The
                   per-rule framing is therefore FORCED, not chosen:
                     unit  = (conversation, rule in rubric)
                     label = 1 if rule in ground_truth_violations
                 — the conversation is byte-identical across L0/L1/L2; only the
                   policy changes. Any conversation-only feature is therefore
                   provably at chance on the within-conversation contrast.

CONVENTIONS (non-negotiable; each has already cost this project a re-run)
  * Label 1 = COMPLIANT, higher score = MORE COMPLIANT (the Probe convention).
    The repo's own gauntlet uses the opposite (1 = violates). AUROC is invariant
    under simultaneously flipping label and score, so the numbers remain
    comparable to the 0.896 receipt; we flip once, here, and never again.
  * Every AUROC goes through auroc_checked(), asserting agreement with
    sklearn.roc_auc_score to 1e-9.
  * Grouping: SafePyramid units are split BY CONVERSATION. A random unit-level
    split would leak the same conversation across train/test.

Protocol matches scripts/run_trivial_gauntlet.py exactly:
    TfidfVectorizer(ngram_range=(1,1)) + LogisticRegression(max_iter=1000)
    fit on train, evaluated on test, splits from three_way_split(n, 42).

Outputs: results/rebuild/benchmark2/{dynabench,safepyramid}_gauntlet.json
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.compliance.splits import three_way_split

SEED = 42
OUT_DIR = "results/rebuild/benchmark2"

# Marker lists reused verbatim from the repo's guardrail baseline so the
# keyword floor is the same instrument used on OmniCompliance.
from src.compliance.guardrails_baseline import (  # noqa: E402
    VIOLATION_MARKERS,
    COMPLIANCE_MARKERS,
)


# ── orientation guard ────────────────────────────────────────────────────────
def auroc_checked(labels, scores) -> float:
    """AUROC with label 1 = COMPLIANT and score = higher-is-more-compliant.

    Tie-averaged rank implementation asserted against sklearn to 1e-9. This is
    the guard added after the sign-inversion bug (context/math_review.md §0):
    an inverted AUROC is invisible to inspection and cost us a full gate re-run.
    """
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    pos, neg = scores[labels == 1], scores[labels == 0]
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    sv = allv[order]
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=float)
    # tie-average
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_p, n_n = len(pos), len(neg)
    # `ranks` is indexed over `allv` == concat([pos, neg]), so the positives are
    # ranks[:n_p]. (Indexing `ranks` with original-array positions instead is a
    # real bug this guard caught on first run.)
    rp = ranks[:n_p]
    auc = (rp.sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n)
    ref = float(roc_auc_score(labels, scores))
    assert abs(auc - ref) < 1e-9, f"AUROC orientation/impl mismatch: {auc} vs {ref}"
    return float(ref)


def tfidf_auroc(train_texts, train_labels, test_texts, test_labels, clf_seed=0):
    """TF-IDF unigram + logreg. Returns (auroc, scores) with score = P(compliant)."""
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 1)),
        LogisticRegression(max_iter=1000, random_state=clf_seed),
    )
    pipe.fit(train_texts, train_labels)
    proba = pipe.predict_proba(test_texts)[:, list(pipe.classes_).index(1)]
    return auroc_checked(test_labels, proba), proba


def raw_auroc(train_scores, train_labels, test_scores, test_labels):
    """AUROC of a raw scalar whose SIGN is oriented on the train slice only."""
    ta = float(roc_auc_score(train_labels, train_scores))
    sign = 1.0 if ta >= 0.5 else -1.0
    return auroc_checked(test_labels, sign * np.asarray(test_scores, float)), sign, ta


def keyword_score(text: str) -> float:
    tl = text.lower()
    v = sum(1 for m in VIOLATION_MARKERS if m in tl)
    c = sum(1 for m in COMPLIANCE_MARKERS if m in tl)
    return float(v - c)


def overlap_score(a: str, b: str) -> float:
    """Jaccard overlap of word sets — the conversation<->rule lexical tell."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def perm_null(train_texts, train_labels, test_texts, test_labels, n=5):
    """Permutation null: shuffle TRAIN labels. Must land ~0.5."""
    out = []
    for s in range(n):
        rng = random.Random(1000 + s)
        perm = list(train_labels)
        rng.shuffle(perm)
        try:
            a, _ = tfidf_auroc(train_texts, perm, test_texts, test_labels, clf_seed=s)
            out.append(a)
        except Exception:
            pass
    return out


def grouped_auroc(labels, scores, groups):
    """Mean within-group AUROC. The honest number when a group (conversation)
    shares surface text: pooled AUROC can be high purely because some
    conversations are globally 'more violating' — that is the generic
    -harmfulness confound, not rule-specific signal."""
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    groups = np.asarray(groups)
    aucs = []
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(labels[m])) < 2:
            continue
        aucs.append(float(roc_auc_score(labels[m], scores[m])))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def provenance(script, extra=None):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    p = {
        "model": "n/a (lexical/CPU gauntlet)",
        "seed": SEED,
        "git_commit": commit,
        "timestamp": datetime.now().astimezone().isoformat(),
        "script": script,
        "versions": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "max_length": "n/a (no model)",
        "pooling": "n/a (no model)",
        "gpu": 0,
        "orientation": "label 1 = COMPLIANT, score higher = COMPLIANT; every "
        "AUROC asserted against sklearn.roc_auc_score to 1e-9",
        "tfidf_protocol": "TfidfVectorizer(ngram_range=(1,1)) + "
        "LogisticRegression(max_iter=1000); fit on train, eval on test; "
        "identical to scripts/run_trivial_gauntlet.py",
    }
    if extra:
        p.update(extra)
    return p


# ── DynaBench ────────────────────────────────────────────────────────────────
def run_dynabench(path, revision):
    df = pd.read_parquet(path)
    # label: PASS = compliant = 1 ; FAIL = violating = 0
    assert set(df.label.unique()) <= {"PASS", "FAIL"}, df.label.unique()
    y = (df.label == "PASS").astype(int).values
    n = len(df)

    tr, vl, vt, te = three_way_split(n, SEED)
    # The gauntlet has no layer/threshold to pick, so train = train+val slices
    # (identical treatment to run_trivial_gauntlet.py, which also only uses
    # train_idx and test_idx).
    train_idx, test_idx = list(tr), list(te)

    conv = df.transcript.tolist()
    pol = df.policy.tolist()
    both = [f"{p}\n\n{c}" for p, c in zip(pol, conv)]

    res = {
        "benchmark": "DynaBench",
        "dataset_source": "montehoover/DynaBench (config=DynaBench, split=test)",
        "dataset_revision": revision,
        "n": int(n),
        "base_rate": {
            "PASS_compliant": int((y == 1).sum()),
            "FAIL_violating": int((y == 0).sum()),
            "frac_compliant": float(y.mean()),
        },
        "split_sizes": {"train": len(train_idx), "test": len(test_idx)},
        "baselines": {},
    }

    def add(name, texts):
        a, sc = tfidf_auroc(
            [texts[i] for i in train_idx], [int(y[i]) for i in train_idx],
            [texts[i] for i in test_idx], [int(y[i]) for i in test_idx],
        )
        res["baselines"][name] = {"auroc": a, "n_test": len(test_idx)}
        return sc

    add("tfidf_transcript_only", conv)
    add("tfidf_policy_only", pol)
    add("tfidf_policy_plus_transcript", both)

    # length + keyword + overlap floors
    for name, arr in [
        ("length_chars", [float(len(c)) for c in conv]),
        ("length_words", [float(len(c.split())) for c in conv]),
        ("keyword_count", [keyword_score(c) for c in conv]),
        ("policy_transcript_overlap", [overlap_score(c, p) for c, p in zip(conv, pol)]),
    ]:
        a, sign, ta = raw_auroc(
            [arr[i] for i in train_idx], [int(y[i]) for i in train_idx],
            [arr[i] for i in test_idx], [int(y[i]) for i in test_idx],
        )
        res["baselines"][name] = {
            "auroc": a, "train_oriented_sign": sign, "train_auroc": ta,
            "n_test": len(test_idx),
        }

    nulls = perm_null(
        [both[i] for i in train_idx], [int(y[i]) for i in train_idx],
        [both[i] for i in test_idx], [int(y[i]) for i in test_idx],
    )
    res["perm_null_tfidf"] = {"aurocs": nulls, "mean": float(np.mean(nulls))}

    # ── F1 on the competitor's own metric ────────────────────────────────────
    # Rachmil et al. (2512.03994) report 86.0% F1 on DynaBench with no lexical
    # baseline. AUROC-vs-F1 is a metric mismatch, so we meet them on THEIR
    # metric: F1 for the VIOLATION (FAIL) class, which is the class their
    # detector is built to flag. Threshold chosen on TRAIN only; test touched
    # once. Also reported with 5-fold CV over the whole n=543 for a
    # lower-variance estimate, since a 110-row test slice is noisy.
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    def fit_predict(tr_t, tr_y, te_t, seed=0):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 1)),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        pipe.fit(tr_t, tr_y)
        return pipe.predict_proba(te_t)[:, list(pipe.classes_).index(1)]

    f1_report = {}
    for name, texts in [("transcript_only", conv),
                        ("policy_plus_transcript", both)]:
        # threshold tuned on train for the FAIL(=violating, y==0) class
        tr_p = fit_predict([texts[i] for i in train_idx],
                           [int(y[i]) for i in train_idx],
                           [texts[i] for i in train_idx])
        ths = np.linspace(0.05, 0.95, 91)
        # predict FAIL when P(compliant) < th
        best_th = max(ths, key=lambda t: f1_score(
            [1 - int(y[i]) for i in train_idx], (tr_p < t).astype(int),
            zero_division=0))
        te_p = fit_predict([texts[i] for i in train_idx],
                           [int(y[i]) for i in train_idx],
                           [texts[i] for i in test_idx])
        holdout_f1 = f1_score([1 - int(y[i]) for i in test_idx],
                              (te_p < best_th).astype(int), zero_division=0)

        # 5-fold CV over all 543 rows, threshold tuned inside each train fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv = []
        allt = np.asarray(texts, dtype=object)
        for tr_i, te_i in skf.split(allt, y):
            p_tr = fit_predict(list(allt[tr_i]), [int(v) for v in y[tr_i]],
                               list(allt[tr_i]))
            th = max(ths, key=lambda t: f1_score(
                1 - y[tr_i], (p_tr < t).astype(int), zero_division=0))
            p_te = fit_predict(list(allt[tr_i]), [int(v) for v in y[tr_i]],
                               list(allt[te_i]))
            cv.append(f1_score(1 - y[te_i], (p_te < th).astype(int),
                               zero_division=0))
        f1_report[name] = {
            "f1_violation_class_holdout": float(holdout_f1),
            "holdout_threshold_from_train": float(best_th),
            "n_holdout": len(test_idx),
            "f1_violation_class_5fold_cv_mean": float(np.mean(cv)),
            "f1_violation_class_5fold_cv_std": float(np.std(cv)),
            "n_cv": int(n),
        }
    res["f1_vs_rachmil"] = {
        "rachmil_reported_f1": 0.860,
        "rachmil_citation": "arXiv:2512.03994, activation-space whitening",
        "note": "F1 for the VIOLATION (FAIL) class. Threshold tuned on train "
                "only. Comparability caveats in the summary: their exact split "
                "and F1 averaging convention are UNVERIFIED from the dataset "
                "card alone.",
        "tfidf": f1_report,
    }
    return res


# ── SafePyramid ──────────────────────────────────────────────────────────────
def build_safepyramid_units(path):
    sp = pd.read_parquet(path)
    rows = []
    for _, r in sp.iterrows():
        j = json.loads(r.rubric_json)
        base = r.id.rsplit("-", 1)[0]
        for e in j["violations"]:
            rows.append((r.id, base, r.level, r.domain, e["rule"], e["text"],
                         0, e.get("type"), r.conversation))
        for e in j["non_violations"]:
            rows.append((r.id, base, r.level, r.domain, e["rule"], e["text"],
                         1, e.get("type"), r.conversation))
    # label: 1 = rule NOT violated = COMPLIANT ; 0 = violated
    return pd.DataFrame(rows, columns=[
        "id", "base", "level", "domain", "rule", "rule_text",
        "y_compliant", "rule_type", "conversation"])


def run_safepyramid(path, revision):
    u = build_safepyramid_units(path)

    # Split BY CONVERSATION (1000 base ids) — a unit-level split would leak the
    # same conversation across train/test.
    bases = sorted(u.base.unique())
    tr, vl, vt, te = three_way_split(len(bases), SEED)
    train_bases = {bases[i] for i in tr}
    test_bases = {bases[i] for i in te}

    res = {
        "benchmark": "SafePyramid",
        "dataset_source": "ByteDance/SafePyramid (config=default, split=test)",
        "dataset_revision": revision,
        "task_framing": (
            "PER-RULE. unit = (conversation, rule in rubric); label 1 = rule NOT "
            "violated (compliant). FORCED, not chosen: every one of the 3000 rows "
            "has >=1 violation, so an any-violation binary label has a 100% base "
            "rate and AUROC is undefined."
        ),
        "metric_mismatch_warning": (
            "The paper's 54.0/35.3/12.9% is EXACT-SET-MATCH accuracy for an LLM "
            "predicting the net violated-rule set in context. The AUROC here is a "
            "different metric on a different unit. A high TF-IDF AUROC does NOT "
            "rebut their 12.9%, and a collapse does NOT by itself prove the "
            "benchmark is non-degenerate at their task."
        ),
        "n_conversations": len(bases),
        "split_conversations": {"train": len(tr), "test": len(te)},
        "levels": {},
    }

    for level in ["L0", "L1", "L2"]:
        g = u[u.level == level].reset_index(drop=True)
        tr_m = g.base.isin(train_bases).values
        te_m = g.base.isin(test_bases).values
        y = g.y_compliant.values
        lv = {
            "n_units": int(len(g)),
            "n_conversations": int(g.base.nunique()),
            "rules_per_conversation": float(len(g) / g.base.nunique()),
            "base_rate": {
                "compliant_not_violated": int((y == 1).sum()),
                "violated": int((y == 0).sum()),
                "frac_violated": float((y == 0).mean()),
            },
            "rule_type_counts": {
                str(k): int(v) for k, v in g.rule_type.value_counts().items()
            },
            "split_sizes": {"train": int(tr_m.sum()), "test": int(te_m.sum())},
            "baselines": {},
        }

        conv = g.conversation.tolist()
        rule = g.rule_text.tolist()
        both = [f"Rule: {r}\n\nConversation: {c}" for r, c in zip(rule, conv)]

        def add_tfidf(name, texts):
            a, sc = tfidf_auroc(
                [texts[i] for i in np.where(tr_m)[0]], [int(v) for v in y[tr_m]],
                [texts[i] for i in np.where(te_m)[0]], [int(v) for v in y[te_m]],
            )
            ga, ng = grouped_auroc(y[te_m], sc, g.base.values[te_m])
            lv["baselines"][name] = {
                "auroc_pooled": a,
                "auroc_grouped_by_conversation": ga,
                "n_groups_scored": ng,
                "n_test": int(te_m.sum()),
            }
            return sc

        add_tfidf("tfidf_rule_only", rule)
        add_tfidf("tfidf_conversation_only", conv)
        sc_both = add_tfidf("tfidf_rule_plus_conversation", both)

        for name, arr in [
            ("length_chars_rule", [float(len(r)) for r in rule]),
            ("length_words_rule", [float(len(r.split())) for r in rule]),
            ("keyword_count_rule", [keyword_score(r) for r in rule]),
            ("rule_conversation_overlap", [overlap_score(c, r)
                                           for c, r in zip(conv, rule)]),
        ]:
            a, sign, ta = raw_auroc(
                [arr[i] for i in np.where(tr_m)[0]], [int(v) for v in y[tr_m]],
                [arr[i] for i in np.where(te_m)[0]], [int(v) for v in y[te_m]],
            )
            ga, ng = grouped_auroc(
                y[te_m], sign * np.asarray([arr[i] for i in np.where(te_m)[0]]),
                g.base.values[te_m])
            lv["baselines"][name] = {
                "auroc_pooled": a,
                "auroc_grouped_by_conversation": ga,
                "n_groups_scored": ng,
                "train_oriented_sign": sign,
                "train_auroc": ta,
            }

        nulls = perm_null(
            [both[i] for i in np.where(tr_m)[0]], [int(v) for v in y[tr_m]],
            [both[i] for i in np.where(te_m)[0]], [int(v) for v in y[te_m]],
        )
        lv["perm_null_tfidf"] = {"aurocs": nulls, "mean": float(np.mean(nulls))}

        # The structure oracle: rule_type is annotation metadata, NOT model
        # input. It is reported as a DIAGNOSTIC of how much of the label is
        # mechanically determined by the rubric's own construction, never as a
        # legitimate baseline.
        tmap = {t: i for i, t in enumerate(sorted(g.rule_type.dropna().unique()))}
        if len(tmap) > 1:
            tv = g.rule_type.map(tmap).fillna(-1).values.astype(float)
            # one-hot logreg on type alone
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder(handle_unknown="ignore")
            X = enc.fit_transform(
                np.asarray(g.rule_type.fillna("NA").astype(str)).reshape(-1, 1)
            )
            clf = LogisticRegression(max_iter=1000, random_state=0)
            clf.fit(X[tr_m], y[tr_m])
            p = clf.predict_proba(X[te_m])[:, list(clf.classes_).index(1)]
            lv["oracle_rule_type_metadata"] = {
                "auroc_pooled": auroc_checked(y[te_m], p),
                "note": "DIAGNOSTIC ONLY — rubric annotation, not model input.",
            }
        res["levels"][level] = lv

    # ── the matched-pair test, unique to SafePyramid ──────────────────────────
    # The conversation is byte-identical across L0/L1/L2; only the policy
    # changes. So any (conversation, rule-text)-only feature assigns the SAME
    # score to a pair that carries BOTH labels across levels => provably 0.5.
    k = u.groupby(["base", "rule_text"]).y_compliant.nunique()
    flip = k[k > 1]
    res["matched_pair_test"] = {
        "n_conversation_rule_pairs": int(len(k)),
        "n_pairs_with_both_labels_across_levels": int(len(flip)),
        "explanation": (
            "These pairs share an identical conversation AND identical rule text "
            "but carry opposite labels at different levels — the label flips "
            "purely because of rule INTERACTIONS (waivers/tightenings) introduced "
            "by the level. Any feature computed from (conversation, rule_text) "
            "alone assigns both members the same score and is therefore EXACTLY "
            "at chance on this contrast, by construction. This is an analytic "
            "fact, not an empirical estimate."
        ),
    }
    res["conversation_identical_across_levels"] = bool(
        (u.groupby("base").conversation.nunique() == 1).all()
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynabench", required=True)
    ap.add_argument("--safepyramid", required=True)
    ap.add_argument("--dynabench-rev", default="unknown")
    ap.add_argument("--safepyramid-rev", default="unknown")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    prov = provenance("scripts/run_benchmark2_gauntlet.py")

    db = run_dynabench(args.dynabench, args.dynabench_rev)
    db["provenance"] = prov
    with open(f"{OUT_DIR}/dynabench_gauntlet.json", "w") as f:
        json.dump(db, f, indent=2)
    print(json.dumps({k: v for k, v in db.items() if k != "provenance"}, indent=2))

    sp = run_safepyramid(args.safepyramid, args.safepyramid_rev)
    sp["provenance"] = prov
    with open(f"{OUT_DIR}/safepyramid_gauntlet.json", "w") as f:
        json.dump(sp, f, indent=2)
    print(json.dumps({k: v for k, v in sp.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
