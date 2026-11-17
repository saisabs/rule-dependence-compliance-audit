#!/usr/bin/env python
"""
E-B3 — third benchmark wave, the trivial gauntlet (CPU stage).

Runs the E-B2 degeneracy gauntlet on AIReg-Bench, CompliBench and FlexBench.
Protocol identical to scripts/run_benchmark2_gauntlet.py:
    TfidfVectorizer(ngram_range=(1,1)) + LogisticRegression(max_iter=1000),
    fit on train, evaluated on test, splits from three_way_split(seed 42) on
    DEDUPLICATED UNIT/GROUP KEYS (never on raw rows where rows share text).

CONVENTIONS (non-negotiable)
  * label 1 = COMPLIANT, higher score = MORE COMPLIANT (E-B2 convention).
  * every AUROC via auroc_checked() -> asserted against sklearn to 1e-9.
  * per-domain permutation p-values are exact-Fraction (1+k)/(1+B); BH
    thresholds compared as exact Fractions.

Outputs: results/rebuild/benchmark3/{aireg,complibench,flexbench}_gauntlet.json
"""
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from fractions import Fraction

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.compliance.splits import three_way_split                      # noqa: E402
from src.compliance.guardrails_baseline import (                       # noqa: E402
    VIOLATION_MARKERS, COMPLIANCE_MARKERS)
from scripts.benchmark3_data import (                                  # noqa: E402
    build_aireg, build_complibench, build_flexbench, REVISIONS)

SEED = 42
OUT_DIR = "results/rebuild/benchmark3"
N_PERM = 1999


# ── shared helpers (verbatim protocol from run_benchmark2_gauntlet.py) ──────
def auroc_checked(labels, scores) -> float:
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
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_p, n_n = len(pos), len(neg)
    auc = (ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n)
    ref = float(roc_auc_score(labels, scores))
    assert abs(auc - ref) < 1e-9, f"AUROC orientation/impl mismatch: {auc} vs {ref}"
    return float(ref)


def tfidf_pipe(seed=0):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 1)),
                         LogisticRegression(max_iter=1000, random_state=seed))


def tfidf_auroc(tr_t, tr_y, te_t, te_y, clf_seed=0):
    pipe = tfidf_pipe(clf_seed)
    pipe.fit(tr_t, tr_y)
    proba = pipe.predict_proba(te_t)[:, list(pipe.classes_).index(1)]
    return auroc_checked(te_y, proba), proba


def raw_auroc(tr_s, tr_y, te_s, te_y):
    ta = float(roc_auc_score(tr_y, tr_s))
    sign = 1.0 if ta >= 0.5 else -1.0
    return (auroc_checked(te_y, sign * np.asarray(te_s, float)), sign, ta)


def keyword_score(text):
    tl = text.lower()
    return float(sum(1 for m in VIOLATION_MARKERS if m in tl)
                 - sum(1 for m in COMPLIANCE_MARKERS if m in tl))


def overlap_score(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def perm_null(tr_t, tr_y, te_t, te_y, n=5):
    out = []
    for s in range(n):
        rng = random.Random(1000 + s)
        perm = list(tr_y)
        rng.shuffle(perm)
        try:
            a, _ = tfidf_auroc(tr_t, perm, te_t, te_y, clf_seed=s)
            out.append(a)
        except Exception:
            pass
    return {"aurocs": out, "mean": float(np.mean(out)) if out else float("nan")}


def grouped_auroc(labels, scores, groups):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    groups = np.asarray(groups)
    a = []
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(labels[m])) < 2:
            continue
        a.append(float(roc_auc_score(labels[m], scores[m])))
    return (float(np.mean(a)) if a else float("nan")), len(a)


def bootstrap_ci_grouped(labels, scores, groups, n_boot=2000, seed=SEED):
    """95% bootstrap CI for AUROC, resampling GROUPS (units sharing text) —
    the E-B2 convention (SafePyramid resampled the conversation)."""
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(labels[idx])) < 2:
            continue
        out.append(roc_auc_score(labels[idx], scores[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def perm_pvalue_fraction(labels, scores, n_perm=N_PERM, seed=SEED):
    """Exact-Fraction permutation p for AUROC>0.5: shuffle labels vs fixed
    scores. Returns (p_as_Fraction, float)."""
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    obs = roc_auc_score(labels, scores)
    rng = np.random.default_rng(seed)
    k = 0
    lab = labels.copy()
    for _ in range(n_perm):
        rng.shuffle(lab)
        if len(np.unique(lab)) < 2:
            continue
        if roc_auc_score(lab, scores) >= obs:
            k += 1
    p = Fraction(1 + k, 1 + n_perm)
    return p, float(p)


def bh_exact(pfracs, alpha=Fraction(5, 100)):
    """BH with exact Fractions. Returns list of dicts (p, threshold, reject)."""
    m = len(pfracs)
    order = sorted(range(m), key=lambda i: pfracs[i])
    reject_upto = -1
    for rank, i in enumerate(order, start=1):
        if pfracs[i] <= alpha * Fraction(rank, m):
            reject_upto = rank
    out = [None] * m
    for rank, i in enumerate(order, start=1):
        out[i] = {
            "p_perm": float(pfracs[i]),
            "p_perm_exact": f"{pfracs[i].numerator}/{pfracs[i].denominator}",
            "bh_threshold_exact": f"{(alpha * Fraction(rank, m)).numerator}/"
                                  f"{(alpha * Fraction(rank, m)).denominator}",
            "significant_bh_0.05": rank <= reject_upto,
        }
    return out


def provenance(script, extra=None):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    p = {
        "model": "n/a (lexical/CPU gauntlet)",
        "seed": SEED,
        "git_commit": commit,
        "timestamp": datetime.now().astimezone().isoformat(),
        "script": script,
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                     "pandas": pd.__version__,
                     "python": platform.python_version()},
        "max_length": "n/a (no model)",
        "gpu": 0,
        "orientation": "label 1 = COMPLIANT, score higher = COMPLIANT; every "
                       "AUROC asserted against sklearn.roc_auc_score to 1e-9",
        "tfidf_protocol": "TfidfVectorizer(ngram_range=(1,1)) + "
                          "LogisticRegression(max_iter=1000); fit on train, "
                          "eval on test; identical to E-B2",
        "datasets": REVISIONS,
    }
    if extra:
        p.update(extra)
    return p


def key_split(keys):
    """three_way_split over sorted deduplicated keys -> dict key->slice."""
    keys = sorted(keys)
    tr, vl, vt, te = three_way_split(len(keys), SEED)
    lab = {}
    for name, idxs in [("train", tr), ("val_layer", vl),
                       ("val_thresh", vt), ("test", te)]:
        for i in idxs:
            lab[keys[i]] = name
    return lab


# ── AIReg-Bench ──────────────────────────────────────────────────────────────
def run_aireg():
    df, articles = build_aireg()
    slot = key_split(df.group.unique())          # 20 (use, system) keys
    df["slot"] = df.group.map(slot)

    res = {
        "benchmark": "AIReg-Bench",
        "n_docs": int(len(df)),
        "unit": "documentation excerpt (.txt file)",
        "grouping": "group = (use, system); system outlines are shared across "
                    "the 5 articles, so a row-level split would leak the same "
                    "fictional system across train/test. 20 group keys, "
                    "three_way_split(20, 42).",
        "structure": "5 articles x 10 uses x 2 systems x 3 profiles "
                     "(Compliance / Violation_1 / Violation_2)",
        "tracks": {},
    }

    for track, ycol in [("expert", "y_expert"),
                        ("construction", "y_construction")]:
        d = df[df[ycol].notna()].reset_index(drop=True)
        y = d[ycol].astype(int).values
        tr_m = (d.slot == "train").values
        te_m = (d.slot == "test").values
        tk = {
            "label": ("median human compliance score > 3 (120-excerpt expert "
                      "subset; PRIMARY)" if track == "expert" else
                      "profile == Compliance (filename/construction label; "
                      "SEMI-CIRCULAR — different files from different prompts)"),
            "n": int(len(d)),
            "base_rate": {"compliant": int((y == 1).sum()),
                          "violating": int((y == 0).sum()),
                          "frac_compliant": float(y.mean())},
            "split_sizes": {s: int((d.slot == s).sum())
                            for s in ("train", "val_layer", "val_thresh", "test")},
            "train_class_counts": {str(k): int(v) for k, v in
                                   zip(*np.unique(y[tr_m], return_counts=True))},
            "test_class_counts": {str(k): int(v) for k, v in
                                  zip(*np.unique(y[te_m], return_counts=True))},
            "baselines": {},
        }

        doc = d.text.tolist()
        art = [articles[a] for a in d.article]
        both = [f"Rule (EU AI Act Article {n}): {a}\n\nDocumentation: {t}"
                for n, a, t in zip(d.article, art, doc)]

        def add_tfidf(name, texts):
            a, sc = tfidf_auroc([texts[i] for i in np.where(tr_m)[0]],
                                [int(v) for v in y[tr_m]],
                                [texts[i] for i in np.where(te_m)[0]],
                                [int(v) for v in y[te_m]])
            ga, ng = grouped_auroc(y[te_m], sc, d.group.values[te_m])
            tk["baselines"][name] = {"auroc": a,
                                     "auroc_grouped_by_system": ga,
                                     "n_groups_scored": ng,
                                     "n_test": int(te_m.sum()),
                                     "ci95_boot_by_system": bootstrap_ci_grouped(
                                         y[te_m], sc, d.group.values[te_m])}
            return sc

        add_tfidf("tfidf_doc_only", doc)
        add_tfidf("tfidf_article_only", art)
        add_tfidf("tfidf_article_plus_doc", both)

        for name, arr in [
            ("length_chars_doc", [float(len(t)) for t in doc]),
            ("length_words_doc", [float(len(t.split())) for t in doc]),
            ("keyword_count_doc", [keyword_score(t) for t in doc]),
            ("article_doc_overlap", [overlap_score(t, a)
                                     for t, a in zip(doc, art)]),
        ]:
            a, sign, ta = raw_auroc([arr[i] for i in np.where(tr_m)[0]],
                                    [int(v) for v in y[tr_m]],
                                    [arr[i] for i in np.where(te_m)[0]],
                                    [int(v) for v in y[te_m]])
            tk["baselines"][name] = {"auroc": a, "train_oriented_sign": sign,
                                     "train_auroc": ta,
                                     "n_test": int(te_m.sum())}

        tk["perm_null_tfidf"] = perm_null(
            [both[i] for i in np.where(tr_m)[0]], [int(v) for v in y[tr_m]],
            [both[i] for i in np.where(te_m)[0]], [int(v) for v in y[te_m]])

        # robustness: grouped 5-fold CV (small single test slice otherwise),
        # plus ungrouped CV — the grouped-vs-ungrouped delta measures how much
        # of the audit-style number was same-system leakage.
        for cvname, splitter, groups in [
            ("cv5_grouped_by_system", GroupKFold(n_splits=5), d.group.values),
            ("cv5_ungrouped_stratified",
             StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), None),
        ]:
            aucs = []
            texts = np.asarray(doc, dtype=object)
            it = (splitter.split(texts, y, groups) if groups is not None
                  else splitter.split(texts, y))
            for tr_i, te_i in it:
                if (len(np.unique(y[tr_i])) < 2 or len(np.unique(y[te_i])) < 2):
                    continue
                a, _ = tfidf_auroc(list(texts[tr_i]), [int(v) for v in y[tr_i]],
                                   list(texts[te_i]), [int(v) for v in y[te_i]])
                aucs.append(a)
            tk[f"tfidf_doc_only_{cvname}"] = {
                "mean": float(np.mean(aucs)), "std": float(np.std(aucs)),
                "folds": aucs}

        # per-article diagnostics with exact-Fraction BH (tfidf_doc_only score)
        _, sc_all = tfidf_auroc([doc[i] for i in np.where(tr_m)[0]],
                                [int(v) for v in y[tr_m]], doc, list(y))
        per_art, pfracs, names = {}, [], []
        for n_art in sorted(d.article.unique()):
            m = te_m & (d.article == n_art).values
            if len(np.unique(y[m])) < 2:
                per_art[f"Article{n_art}"] = {"auroc": float("nan"),
                                              "n": int(m.sum()),
                                              "note": "single-class in test"}
                continue
            a = auroc_checked(y[m], sc_all[m])
            pf, _ = perm_pvalue_fraction(y[m], sc_all[m])
            per_art[f"Article{n_art}"] = {"auroc": a, "n": int(m.sum())}
            pfracs.append(pf)
            names.append(f"Article{n_art}")
        for nm, bh in zip(names, bh_exact(pfracs)):
            per_art[nm].update(bh)
        tk["per_article_tfidf_doc_only"] = per_art
        res["tracks"][track] = tk

    # expert-vs-construction agreement receipt
    e = df[df.y_expert.notna()]
    res["expert_construction_agreement"] = float(
        (e.y_expert == e.y_construction).mean())
    res["human_compliance_means_by_profile"] = {
        k: float(v) for k, v in
        e.groupby("profile").human_compliance.mean().items()}
    return res


# ── CompliBench ──────────────────────────────────────────────────────────────
def run_complibench():
    df, skipped = build_complibench()
    slot = key_split(df.conv_id.unique())
    df["slot"] = df.conv_id.map(slot)
    y = df.y_compliant.values
    tr_m = (df.slot == "train").values
    te_m = (df.slot == "test").values

    res = {
        "benchmark": "CompliBench",
        "n_conversations": int(df.conv_id.nunique()),
        "n_conversations_skipped_no_message_list": skipped,
        "n_units": int(len(df)),
        "unit": "assistant turn",
        "grouping": "conversation (turns share dialogue text); "
                    "three_way_split over sorted conversation ids",
        "base_rate": {"compliant": int((y == 1).sum()),
                      "violating": int((y == 0).sum()),
                      "frac_violating": float((y == 0).mean())},
        "per_domain_n": {k: int(v) for k, v in
                         df.domain.value_counts().items()},
        "split_sizes": {s: int((df.slot == s).sum())
                        for s in ("train", "val_layer", "val_thresh", "test")},
        "baselines": {},
    }

    turn = df.turn_text.tolist()
    gl = df.guideline_text.tolist()
    both = [f"Guideline: {g}\n\nAssistant turn: {t}" for g, t in zip(gl, turn)]

    scores = {}

    def add_tfidf(name, texts):
        a, sc = tfidf_auroc([texts[i] for i in np.where(tr_m)[0]],
                            [int(v) for v in y[tr_m]],
                            [texts[i] for i in np.where(te_m)[0]],
                            [int(v) for v in y[te_m]])
        ga, ng = grouped_auroc(y[te_m], sc, df.conv_id.values[te_m])
        res["baselines"][name] = {"auroc_pooled": a,
                                  "auroc_grouped_by_conversation": ga,
                                  "n_groups_scored": ng,
                                  "n_test": int(te_m.sum()),
                                  "ci95_boot_by_conversation":
                                      bootstrap_ci_grouped(
                                          y[te_m], sc,
                                          df.conv_id.values[te_m])}
        scores[name] = sc
        return sc

    glf = df.guideline_fair.tolist()
    both_fair = [f"Guideline: {g}\n\nAssistant turn: {t}"
                 for g, t in zip(glf, turn)]

    add_tfidf("tfidf_turn_only", turn)
    add_tfidf("tfidf_guideline_only", gl)
    add_tfidf("tfidf_guideline_plus_turn", both)
    # judge-faithful arms: guideline_text on violating turns is the MODIFIED
    # directive (generation metadata absent from the judge-facing
    # assistant_guidelines). The *_fair arms substitute the original back in.
    add_tfidf("tfidf_guideline_fair_only", glf)
    add_tfidf("tfidf_guideline_fair_plus_turn", both_fair)
    res["guideline_metadata_leak"] = {
        "n_turns_with_modified_guideline": int(df.guideline_was_modified.sum()),
        "n_violating_turns": int((y == 0).sum()),
        "n_violating_turns_with_modified_guideline": int(
            df[df.y_compliant == 0].guideline_was_modified.sum()),
        "n_compliant_turns_with_modified_guideline": int(
            df[df.y_compliant == 1].guideline_was_modified.sum()),
        "note": "guideline_text == mistakes[].guideline (the modified, "
                "violation-inducing directive) on violating turns; the "
                "modified text is NOT present in assistant_guidelines, so "
                "tfidf_guideline_only is a CONSTRUCTION-METADATA ORACLE, not "
                "judge-input degeneracy. The fair arms are the honest "
                "rule-aware floors.",
    }

    for name, arr in [
        ("length_chars_turn", [float(len(t)) for t in turn]),
        ("length_words_turn", [float(len(t.split())) for t in turn]),
        ("keyword_count_turn", [keyword_score(t) for t in turn]),
        ("guideline_turn_overlap", [overlap_score(t, g)
                                    for t, g in zip(turn, gl)]),
    ]:
        a, sign, ta = raw_auroc([arr[i] for i in np.where(tr_m)[0]],
                                [int(v) for v in y[tr_m]],
                                [arr[i] for i in np.where(te_m)[0]],
                                [int(v) for v in y[te_m]])
        ga, ng = grouped_auroc(y[te_m],
                               sign * np.asarray([arr[i] for i
                                                  in np.where(te_m)[0]]),
                               df.conv_id.values[te_m])
        res["baselines"][name] = {"auroc_pooled": a,
                                  "auroc_grouped_by_conversation": ga,
                                  "n_groups_scored": ng,
                                  "train_oriented_sign": sign,
                                  "train_auroc": ta}

    res["perm_null_tfidf"] = perm_null(
        [both[i] for i in np.where(tr_m)[0]], [int(v) for v in y[tr_m]],
        [both[i] for i in np.where(te_m)[0]], [int(v) for v in y[te_m]])

    # per-domain, exact-Fraction BH over the 3 domains (turn-only score).
    # scores[...] are TEST-slice arrays; build domain masks within that slice.
    te_dom = df.domain.values[te_m]
    y_te = y[te_m]
    per_dom, pfracs, names = {}, [], []
    for dom in sorted(df.domain.unique()):
        m = te_dom == dom
        a = auroc_checked(y_te[m], scores["tfidf_turn_only"][m])
        pf, _ = perm_pvalue_fraction(y_te[m], scores["tfidf_turn_only"][m])
        per_dom[dom] = {"auroc_turn_only": a, "n": int(m.sum())}
        pfracs.append(pf)
        names.append(dom)
    for nm, bh in zip(names, bh_exact(pfracs)):
        per_dom[nm].update(bh)
    res["per_domain"] = per_dom
    return res


# ── FlexBench ────────────────────────────────────────────────────────────────
def run_flexbench():
    res = {
        "benchmark": "FlexBench",
        "unit": "unique input text (test+validation pooled, exact-dedup)",
        "grouping": "none beyond exact-dedup of input text; keys whose "
                    "duplicates disagree on severity are dropped",
        "label_structure": "5-tier severity; label_strict/moderate/loose are "
                           "deterministic thresholds of severity "
                           "(strict flags >=high, moderate >=moderate, "
                           "loose >=low). benchmark label 1 = FLAGGED; "
                           "our y = 1 - label (1 = compliant).",
        "tracks": {},
    }
    for track in ("prompt", "response"):
        d, report = build_flexbench(track)
        slot = key_split(d.input.tolist())
        d["slot"] = d.input.map(slot)
        tr_m = (d.slot == "train").values
        te_m = (d.slot == "test").values
        texts = d.input.tolist()

        tk = {
            "dedup_report": report,
            "severity_counts": {k: int(v) for k, v in
                                d.severity.value_counts().items()},
            "source_counts": {k: int(v) for k, v in
                              d.source.value_counts().items()},
            "split_sizes": {s: int((d.slot == s).sum())
                            for s in ("train", "val_layer", "val_thresh",
                                      "test")},
            "strictness": {},
        }
        for strict in ("strict", "moderate", "loose"):
            y = d[f"y_compliant_{strict}"].values.astype(int)
            lv = {
                "base_rate": {"compliant": int((y == 1).sum()),
                              "flagged": int((y == 0).sum()),
                              "frac_flagged": float((y == 0).mean())},
                "baselines": {},
            }
            a, sc = tfidf_auroc([texts[i] for i in np.where(tr_m)[0]],
                                [int(v) for v in y[tr_m]],
                                [texts[i] for i in np.where(te_m)[0]],
                                [int(v) for v in y[te_m]])
            lv["baselines"]["tfidf_input_only"] = {
                "auroc": a, "n_test": int(te_m.sum()),
                "ci95_boot_rows": bootstrap_ci_grouped(
                    y[te_m], sc, np.arange(int(te_m.sum())))}
            for name, arr in [
                ("length_chars", [float(len(t)) for t in texts]),
                ("length_words", [float(len(t.split())) for t in texts]),
                ("keyword_count", [keyword_score(t) for t in texts]),
            ]:
                aa, sign, ta = raw_auroc(
                    [arr[i] for i in np.where(tr_m)[0]],
                    [int(v) for v in y[tr_m]],
                    [arr[i] for i in np.where(te_m)[0]],
                    [int(v) for v in y[te_m]])
                lv["baselines"][name] = {"auroc": aa,
                                         "train_oriented_sign": sign,
                                         "train_auroc": ta}
            lv["perm_null_tfidf"] = perm_null(
                [texts[i] for i in np.where(tr_m)[0]],
                [int(v) for v in y[tr_m]],
                [texts[i] for i in np.where(te_m)[0]],
                [int(v) for v in y[te_m]], n=3)

            # per-source diagnostic (no BH — descriptive; sources differ in
            # base rate by construction). sc is TEST-slice-length.
            te_src = d.source.values[te_m]
            y_te = y[te_m]
            per_src = {}
            for src in sorted(d.source.unique()):
                m = te_src == src
                if m.sum() < 10 or len(np.unique(y_te[m])) < 2:
                    per_src[src] = {"auroc": float("nan"), "n": int(m.sum())}
                    continue
                per_src[src] = {"auroc": auroc_checked(y_te[m], sc[m]),
                                "n": int(m.sum())}
            lv["per_source_tfidf"] = per_src
            tk["strictness"][strict] = lv
        res["tracks"][track] = tk
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prov = provenance("scripts/run_benchmark3_gauntlet.py")
    for name, fn in [("aireg", run_aireg),
                     ("complibench", run_complibench),
                     ("flexbench", run_flexbench)]:
        r = fn()
        r["provenance"] = prov
        with open(f"{OUT_DIR}/{name}_gauntlet.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"=== {name} ===")
        print(json.dumps({k: v for k, v in r.items() if k != "provenance"},
                         indent=2)[:4000])


if __name__ == "__main__":
    main()
