"""
run_ifeval_probe.py — Benchmark expansion, stage 3b: Probe vs the trivial floor on IFEval.

Pure CPU; consumes stage 3a's activations. Fits Probe per IFEval instruction
category exactly as the OmniCompliance pipeline does (unit difference-of-means
direction on `train`, anchor layer on `val_layer`, threshold on `val_thresh`,
`test` touched once, repo's three_way_split 40/20/20/20, seed 42), and races it
against the trivial floor.

────────────────────────────────────────────────────────────────────────────────
THE VERDICT IS A DECOMPOSITION, NOT ONE NUMBER
────────────────────────────────────────────────────────────────────────────────
IFEval's label is, by construction, a deterministic function of a countable
surface FORM feature. So "does Probe beat the floor" splits into two questions
with different answers, and each gets its own margin + CI:

  CONTENT verdict   Probe vs TF-IDF          — the OmniCompliance-comparable bar.
                    This is the paper's actual claim, I(a(x); Y | S(x)) > 0
                    with S = lexical content.
  FORM verdict      Probe vs max(honest floors) where honest floors are
                    {tfidf, length_char, keyword, format_feats, refusal_dir,
                    yesno} — i.e. everything EXCEPT the tautological oracle.
                    `format_feats` is the bar the brief names: bullet count,
                    word count, banned-keyword presence, capitalisation ratio.

  format_oracle is deliberately EXCLUDED from both bars. It is kwargs-aware and
  re-derives the arithmetic the verifier thresholds, so beating it is impossible
  BY CONSTRUCTION. It is reported as a CEILING that measures how mechanically
  form-determined the label is — never as a competitor. Letting it define
  "beats_floor" would report "Probe fails" everywhere for a tautological reason
  and bury the finding.

FLOORS (all oriented "higher = VIOLATION", matching the repo's baselines):
  tfidf            TF-IDF unigram + logreg on the response text (lexical CONTENT)
  length_char      response length in characters (sign oriented on train)
  keyword          repo's continuous net violation-marker count
  format_feats     generic surface-FORM logreg, kwargs-FREE (honest form bar)
  format_oracle    kwargs-AWARE margin to the constraint boundary — TAUTOLOGICAL
  refusal_dir      Arditi refusal direction (rebuilt bit-exact); variant x layer
                   x sign selected on val_layer
  yesno            same-model Yes/No verdict logit; space-variant x sign on val_layer

────────────────────────────────────────────────────────────────────────────────
TWO NAMED DEVIATIONS FROM THE BRIEF (loud, not silent)
────────────────────────────────────────────────────────────────────────────────
1. BOOTSTRAP. The brief specifies a PAIRED bootstrap with the pair as resampling
   unit. IFEval HAS NO PAIRS — each (prompt, instruction_id) instance carries its
   own label and classes are unbalanced. We use a TWO-SAMPLE STRATIFIED bootstrap:
   resample items within class (class counts fixed) and feed THE SAME resampled
   items to Probe and to the floor, preserving the Probe-floor correlation that made
   the paired bootstrap correct on OmniCompliance.
2. GROUPED SPLIT. The brief says three_way_split(n, 42) over instances. But 33
   IFEval prompts carry 2-3 instructions from the SAME category, and all
   instances of a prompt share ONE activation vector (the text is the same). An
   instance-level split can put such twins in train AND test — identical
   activations on both sides, i.e. leakage. We therefore apply the repo's
   three_way_split to the category's UNIQUE KEYS and map keys -> instances.
   Same function, same seed, same fractions; the unit is the prompt.

Usage:
    python scripts/run_ifeval_probe.py                 # primary: strict, acts512
    python scripts/run_ifeval_probe.py --acts acts2048 # truncation control

Outputs:
    results/rebuild/ifeval/splits/<category>_seed42_splits.json
    results/rebuild/ifeval/category_<category>_seed42.json
    results/rebuild/ifeval/IFEVAL_SUMMARY.json
"""

import argparse
import collections
import datetime
import json
import os
import re
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compliance.splits import three_way_split
from src.compliance.direction import MeanDifferenceDirection
from src.compliance.metrics import calibrate_threshold
from src.compliance.guardrails_baseline import COMPLIANCE_MARKERS, VIOLATION_MARKERS

B = 2000
N_PERM = 100
MIN_TEST_PER_CLASS = 3
MIN_SLICE_PER_CLASS = 1
SEED = 42

HONEST_FLOORS = ["tfidf", "length_char", "keyword", "format_feats",
                 "refusal_dir", "yesno"]
TAUTOLOGICAL_FLOORS = ["format_oracle"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True).strip()
    except Exception:
        return "unknown"


# ── AUROC: one convention, sklearn-checked (a sign inversion already cost a re-run) ──

def _auroc_ranks(pos, neg) -> float:
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
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auroc_checked(pos, neg) -> float:
    """AUROC for 'higher = VIOLATION'. pos = violating scores, neg = compliant.
    Cross-checked against sklearn to 1e-9 — the orientation guard."""
    a = _auroc_ranks(pos, neg)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[np.asarray(pos, float), np.asarray(neg, float)]
    b = roc_auc_score(y, s)
    assert abs(a - b) < 1e-9, f"AUROC mismatch: local {a} vs sklearn {b}"
    return float(a)


def auroc_scores(scores, y_viol) -> float:
    """y_viol: 1 = VIOLATING. scores: higher = violation."""
    s = np.asarray(scores, float); y = np.asarray(y_viol, int)
    return auroc_checked(s[y == 1], s[y == 0])


# ── format features ───────────────────────────────────────────────────────────

_BULLET_RE = re.compile(r"^\s*[\*\-]\s+", re.MULTILINE)
_HIGHLIGHT_RE = re.compile(r"\*[^\*\n]+\*")
_SECTION_RE = re.compile(r"^\s*#+\s", re.MULTILINE)
_TITLE_RE = re.compile(r"<<[^>]+>>")
_PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")


def n_words(t: str) -> int:
    return len(re.findall(r"\b\w+\b", t))


def n_sentences(t: str) -> int:
    return len([s for s in re.split(r"[.!?]+", t) if s.strip()])


def format_features(t: str) -> dict:
    letters = [c for c in t if c.isalpha()]
    nu = sum(1 for c in letters if c.isupper())
    words = re.findall(r"\b\w+\b", t)
    return {
        "n_chars": len(t), "n_words": len(words), "n_sentences": n_sentences(t),
        "n_paragraphs": len([p for p in t.split("\n\n") if p.strip()]),
        "n_newlines": t.count("\n"), "n_commas": t.count(","),
        "n_periods": t.count("."), "n_bullets": len(_BULLET_RE.findall(t)),
        "n_highlights": len(_HIGHLIGHT_RE.findall(t)),
        "n_sections": len(_SECTION_RE.findall(t)), "n_asterisks": t.count("*"),
        "n_upper": nu, "n_lower": len(letters) - nu,
        "upper_ratio": nu / max(len(letters), 1),
        "n_capital_words": sum(1 for w in words if w.isupper()),
        "n_digits": sum(1 for c in t if c.isdigit()),
        "n_placeholders": len(_PLACEHOLDER_RE.findall(t)),
        "has_title": float(bool(_TITLE_RE.search(t))),
        "has_ps": float("P.S." in t or "P.P.S" in t),
        "has_sep": float("******" in t),
        "starts_quote": float(t.strip()[:1] == '"'),
        "ends_quote": float(t.strip()[-1:] == '"'),
        "n_braces": t.count("{") + t.count("}"),
        "json_like": float(t.strip()[:1] in "{["),
    }


FORMAT_KEYS = sorted(format_features("x").keys())


def format_matrix(texts):
    return np.array([[format_features(t)[k] for k in FORMAT_KEYS] for t in texts], float)


# ── kwargs-aware oracle margin (higher = violation, a priori) ─────────────────

def _rel_margin(actual, target, relation):
    if relation == "at least":
        return float(target - actual)
    return float(actual - target)


def oracle_margin(iid: str, kw: dict, resp: str) -> float:
    kw = {k: v for k, v in (kw or {}).items() if v}
    t = resp
    try:
        if iid == "punctuation:no_comma":
            return float(t.count(","))
        if iid == "change_case:english_lowercase":
            return float(sum(1 for c in t if c.isupper()))
        if iid == "change_case:english_capital":
            return float(sum(1 for c in t if c.islower()))
        if iid == "change_case:capital_word_frequency":
            n = sum(1 for w in re.findall(r"\b\w+\b", t) if w.isupper())
            return _rel_margin(n, kw.get("capital_frequency", 0),
                               kw.get("capital_relation", "at least"))
        if iid == "keywords:existence":
            ks = kw.get("keywords", []) or []
            return float(sum(1 for k in ks if k.lower() not in t.lower()))
        if iid == "keywords:forbidden_words":
            ks = kw.get("forbidden_words", []) or []
            return float(sum(t.lower().count(k.lower()) for k in ks))
        if iid == "keywords:frequency":
            k = kw.get("keyword", "")
            n = t.lower().count(str(k).lower()) if k else 0
            return _rel_margin(n, kw.get("frequency", 0), kw.get("relation", "at least"))
        if iid == "keywords:letter_frequency":
            l = str(kw.get("letter", "")).lower()
            n = t.lower().count(l) if l else 0
            return _rel_margin(n, kw.get("let_frequency", 0),
                               kw.get("let_relation", "at least"))
        if iid == "length_constraints:number_words":
            return _rel_margin(n_words(t), kw.get("num_words", 0),
                               kw.get("relation", "at least"))
        if iid == "length_constraints:number_sentences":
            return _rel_margin(n_sentences(t), kw.get("num_sentences", 0),
                               kw.get("relation", "at least"))
        if iid == "length_constraints:number_paragraphs":
            n = len([p for p in t.split("\n\n") if p.strip()])
            return abs(float(n - kw.get("num_paragraphs", 0)))
        if iid == "length_constraints:nth_paragraph_first_word":
            ps = [p for p in t.split("\n\n") if p.strip()]
            i = int(kw.get("nth_paragraph", 1)) - 1
            fw = str(kw.get("first_word", "")).lower()
            return 0.0 if (0 <= i < len(ps) and ps[i].strip().lower().startswith(fw)) else 1.0
        if iid == "detectable_format:number_bullet_lists":
            return abs(float(len(_BULLET_RE.findall(t)) - kw.get("num_bullets", 0)))
        if iid == "detectable_format:number_highlighted_sections":
            return float(kw.get("num_highlights", 0) - len(_HIGHLIGHT_RE.findall(t)))
        if iid == "detectable_format:multiple_sections":
            sp = str(kw.get("section_spliter", "Section"))
            return float(kw.get("num_sections", 0) - t.count(sp))
        if iid == "detectable_format:title":
            return 0.0 if _TITLE_RE.search(t) else 1.0
        if iid == "detectable_format:json_format":
            try:
                json.loads(t.strip().strip("`").replace("json", "", 1).strip())
                return 0.0
            except Exception:
                return 1.0
        if iid == "detectable_format:constrained_response":
            opts = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
            return 0.0 if any(o in t for o in opts) else 1.0
        if iid == "detectable_content:number_placeholders":
            return float(kw.get("num_placeholders", 0) - len(_PLACEHOLDER_RE.findall(t)))
        if iid == "detectable_content:postscript":
            m = str(kw.get("postscript_marker", "P.S."))
            return 0.0 if m.lower() in t.lower() else 1.0
        if iid == "startend:quotation":
            s = t.strip()
            return 0.0 if (s.startswith('"') and s.endswith('"')) else 1.0
        if iid == "startend:end_checker":
            e = str(kw.get("end_phrase", "")).strip().lower()
            return 0.0 if t.strip().lower().endswith(e) else 1.0
        if iid == "combination:two_responses":
            return 0.0 if "******" in t else 1.0
        if iid == "combination:repeat_prompt":
            p = str(kw.get("prompt_to_repeat", "")).strip().lower()
            return 0.0 if p and t.strip().lower().startswith(p[:60]) else 1.0
        if iid == "language:response_language":
            try:
                from langdetect import detect
                return 0.0 if detect(t) == kw.get("language") else 1.0
            except Exception:
                return 1.0
    except Exception:
        return 0.0
    return 0.0


# ── helpers ───────────────────────────────────────────────────────────────────

def orient_raw(train_s, train_y, test_s):
    a = auroc_scores(train_s, train_y)
    sign = 1.0 if a >= 0.5 else -1.0
    return np.asarray(test_s, float) * sign, sign


def ecdf_norm(train_vals, vals):
    tv = np.sort(np.asarray(train_vals, float))
    if len(tv) == 0:
        return np.asarray(vals, float)
    return np.searchsorted(tv, np.asarray(vals, float), side="right") / len(tv)


def clopper_upper(n_pairs):
    """One-sided 95% Clopper-Pearson bound for k=n successes: 0.05^(1/n)."""
    return float(0.05 ** (1.0 / max(n_pairs, 1)))


def bh(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); adj = np.empty(n); prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj


def fit_probe(acts, y_viol, tr, vl, vt, te):
    """Probe: direction on train, anchor layer on val_layer, threshold on val_thresh
    (metrics.calibrate_threshold — NEVER test, that is Bug 1), test touched once.
    Fit higher=compliant (repo convention) then NEGATED so every score here is a
    violation-detector."""
    c_tr = acts[[i for i in tr if y_viol[i] == 0]]
    v_tr = acts[[i for i in tr if y_viol[i] == 1]]
    d = MeanDifferenceDirection().fit(c_tr, v_tr)
    sc_vl = d.score(acts[vl])
    aur = [auroc_scores(-sc_vl[:, l], y_viol[vl]) for l in range(sc_vl.shape[1])]
    anchor = int(np.argmax(aur))
    sc_vt = d.score(acts[vt])[:, anchor]
    thr, f1 = None, None
    c_vt, v_vt = sc_vt[y_viol[vt] == 0], sc_vt[y_viol[vt] == 1]
    if len(c_vt) and len(v_vt):
        thr = calibrate_threshold(c_vt, v_vt)
        sc_te = d.score(acts[te])[:, anchor]
        f1 = float(f1_score(y_viol[te], (sc_te < thr).astype(int), zero_division=0))
    return -d.score(acts[te])[:, anchor], anchor, float(aur[anchor]), thr, f1


def boot_margin(probe_te, floor_te, yte, rng):
    """Two-sample STRATIFIED bootstrap of AUROC(Probe) - AUROC(floor).
    Same resampled items go to both methods (preserves their correlation)."""
    pos = np.where(yte == 1)[0]; neg = np.where(yte == 0)[0]
    f = np.asarray(floor_te, float)
    m = np.empty(B)
    for b in range(B):
        sel = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        ys = yte[sel]
        m[b] = auroc_scores(probe_te[sel], ys) - auroc_scores(f[sel], ys)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(lo), float(hi), float(min(max(p, 1.0 / B), 1.0))


def classify(fl):
    """Label what KIND of benchmark cell this category is."""
    tf, ff = fl["tfidf"], fl["format_feats"]
    orc = fl.get("format_oracle")          # None at the pooled level (undefined)
    tags = []
    if tf >= 0.70:
        tags.append("content-shortcut")
    if ff >= 0.90:
        tags.append("form-shortcut-saturated")
    if orc is not None and tf < 0.60 and ff < 0.60 and orc >= 0.90:
        tags.append("hard-for-surface-but-mechanical")
    if not tags:
        tags.append("mixed")
    return tags


def analyze(name, recs, ACTS, refusal, yn, rng, out_dir, seed, gpu, label, actsname):
    """recs: list of dicts with key, instruction_id, kwargs, response, row, y."""
    n = len(recs)
    y = np.array([r["y"] for r in recs])
    keys = [r["key"] for r in recs]

    # ── GROUPED split: three_way_split over UNIQUE KEYS (see module docstring) ──
    uk, seen = [], set()
    for k in keys:
        if k not in seen:
            seen.add(k); uk.append(k)
    ktr, kvl, kvt, kte = three_way_split(len(uk), seed)
    kset = {"train": {uk[i] for i in ktr}, "val_layer": {uk[i] for i in kvl},
            "val_thresh": {uk[i] for i in kvt}, "test": {uk[i] for i in kte}}
    sl = {s: np.array([i for i, r in enumerate(recs) if r["key"] in ks], dtype=int)
          for s, ks in kset.items()}
    tr, vl, vt, te = sl["train"], sl["val_layer"], sl["val_thresh"], sl["test"]

    counts = {s: {"n": len(ix), "n_viol": int(y[ix].sum()),
                  "n_comp": int((1 - y[ix]).sum())} for s, ix in sl.items()}

    json.dump({
        "provenance": {
            "model": gpu["provenance"]["model"], "seed": seed, "git_commit": git_commit(),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "script": "scripts/run_ifeval_probe.py",
            "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                         "python": sys.version.split()[0]},
            "split_fractions": {"train": 0.40, "val_layer": 0.20,
                                "val_thresh": 0.20, "test": 0.20},
            "max_length": gpu["provenance"]["max_length"],
            "pooling": gpu["provenance"]["pooling"],
            "generation_config": gpu["provenance"]["generation_config"],
            "gpu": gpu["provenance"]["gpu"],
        },
        "category": name, "n_instances": n, "n_unique_keys": len(uk), "label": label,
        "split_source": "src.compliance.splits.three_way_split(n_unique_keys, 42) "
                        "— GROUPED by prompt key to prevent shared-activation leakage",
        "split_key_indices": {"train": ktr, "val_layer": kvl,
                              "val_thresh": kvt, "test": kte},
        "split_instance_indices": {s: ix.tolist() for s, ix in sl.items()},
        "class_counts": counts,
        "items": [{"local_idx": j, "key": r["key"], "instruction_id": r["instruction_id"],
                   "violating": int(r["y"])} for j, r in enumerate(recs)],
    }, open(os.path.join(out_dir, "splits", f"{name}_seed{seed}_splits.json"), "w"), indent=1)

    bad = [s for s, c in counts.items() if min(c["n_viol"], c["n_comp"]) < MIN_SLICE_PER_CLASS]
    tmin = min(counts["test"]["n_viol"], counts["test"]["n_comp"])
    if bad or tmin < MIN_TEST_PER_CLASS:
        return None, {"category": name, "n": n,
                      "n_compliant": int((1 - y).sum()), "n_violating": int(y.sum()),
                      "reason": (f"slice(s) {bad} single-class" if bad
                                 else f"test minority class {tmin} < {MIN_TEST_PER_CLASS}"),
                      "class_counts": counts}

    texts = [r["response"] for r in recs]
    acts = ACTS[np.array([r["row"] for r in recs])]
    tr_t = [texts[i] for i in tr]; te_t = [texts[i] for i in te]

    probe_te, anchor, probe_val, probe_thr, probe_f1 = fit_probe(acts, y, tr, vl, vt, te)
    probe_auc = auroc_scores(probe_te, y[te])

    floors, tfidf_note = {}, None
    try:
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)),
                             LogisticRegression(max_iter=1000, random_state=0))
        pipe.fit(tr_t, y[tr])
        floors["tfidf"] = pipe.predict_proba(te_t)[:, list(pipe.classes_).index(1)]
    except ValueError as e:
        tfidf_note = f"TF-IDF degenerate ({e}); scored as chance 0.5"
        floors["tfidf"] = np.full(len(te), 0.5)

    floors["length_char"], _ = orient_raw(np.array([len(t) for t in tr_t], float), y[tr],
                                          np.array([len(t) for t in te_t], float))

    def kwsc(t):
        tl = t.lower()
        return float(sum(1 for m in VIOLATION_MARKERS if m in tl)
                     - sum(1 for m in COMPLIANCE_MARKERS if m in tl))
    floors["keyword"], _ = orient_raw(np.array([kwsc(t) for t in tr_t]), y[tr],
                                      np.array([kwsc(t) for t in te_t]))

    fpipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=0))
    fpipe.fit(format_matrix(tr_t), y[tr])
    floors["format_feats"] = fpipe.predict_proba(format_matrix(te_t))[
        :, list(fpipe.classes_).index(1)]

    # The kwargs-aware oracle is defined per instruction_id. At the POOLED
    # prompt level the label is follow_ALL_instructions, which is a conjunction
    # over several different instructions — there is no single constraint margin,
    # so the oracle is UNDEFINED here, not "at chance". Emitting 0.5 would tell a
    # reader the pooled task is not mechanically form-determined, the exact
    # opposite of the per-category finding (oracle saturates at 1.0 on
    # punctuation / startend / detectable_content). So we omit it explicitly.
    iids = np.array([r["instruction_id"] for r in recs])
    oracle_defined = not (len(np.unique(iids)) == 1 and iids[0] == "__ALL__")
    if oracle_defined:
        marg = np.array([oracle_margin(r["instruction_id"], r["kwargs"], r["response"])
                         for r in recs], float)
        orc = np.zeros(len(te))
        for u in np.unique(iids):
            m_tr = tr[iids[tr] == u]; mask = iids[te] == u
            if not mask.any():
                continue
            s = 1.0
            if len(m_tr) >= 2 and len(np.unique(y[m_tr])) == 2:
                s = 1.0 if auroc_scores(marg[m_tr], y[m_tr]) >= 0.5 else -1.0
            ref = marg[m_tr] * s if len(m_tr) else marg[te][mask] * s
            orc[mask] = ecdf_norm(ref, marg[te][mask] * s)
        floors["format_oracle"] = orc

    best = (-1, None)
    for vname, D in refusal.items():
        proj = np.einsum("nld,ld->nl", acts, D)
        for l in range(proj.shape[1]):
            for sg in (1.0, -1.0):
                a = auroc_scores(sg * proj[vl, l], y[vl])
                if a > best[0]:
                    best = (a, (vname, l, sg))
    rv, rl, rs = best[1]
    floors["refusal_dir"] = rs * np.einsum("nld,ld->nl", acts, refusal[rv])[te, rl]

    ynb = (-1, None)
    rows_i = np.array([r["row_inst"] for r in recs])
    for sp in ("nospace", "space"):
        v = yn[f"no_{sp}"][rows_i] - yn[f"yes_{sp}"][rows_i]
        for sg in (1.0, -1.0):
            a = auroc_scores(sg * v[vl], y[vl])
            if a > ynb[0]:
                ynb = (a, (sp, sg))
    ysp, ysg = ynb[1]
    floors["yesno"] = ysg * (yn[f"no_{ysp}"][rows_i] - yn[f"yes_{ysp}"][rows_i])[te]

    fl_auc = {k: auroc_scores(v, y[te]) for k, v in floors.items()}

    # ── the two verdicts, each with its own CI ──
    honest = {k: fl_auc[k] for k in HONEST_FLOORS}
    hname = max(honest, key=honest.get); hauc = honest[hname]
    lo_h, hi_h, p_h = boot_margin(probe_te, floors[hname], y[te], rng)
    lo_c, hi_c, p_c = boot_margin(probe_te, floors["tfidf"], y[te], rng)

    perm = []
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        if len(np.unique(yp[tr])) < 2 or len(np.unique(yp[te])) < 2 \
           or len(np.unique(yp[vl])) < 2:
            continue
        try:
            perm.append(auroc_scores(fit_probe(acts, yp, tr, vl, vt, te)[0], yp[te]))
        except Exception:
            pass
    perm_tf = []
    for _ in range(20):
        yp = rng.permutation(y)
        if len(np.unique(yp[tr])) < 2 or len(np.unique(yp[te])) < 2:
            continue
        try:
            pp = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)),
                               LogisticRegression(max_iter=1000, random_state=0))
            pp.fit(tr_t, yp[tr])
            perm_tf.append(auroc_scores(
                pp.predict_proba(te_t)[:, list(pp.classes_).index(1)], yp[te]))
        except Exception:
            pass

    npair = int(counts["test"]["n_viol"] * counts["test"]["n_comp"])
    row = {
        "category": name, "n": n, "n_unique_keys": len(uk), "n_test": len(te),
        "n_test_violating": counts["test"]["n_viol"],
        "n_test_compliant": counts["test"]["n_comp"], "n_test_pairs": npair,
        "label": label, "acts": actsname,
        "probe": probe_auc, "probe_anchor_layer": anchor, "probe_val_layer_auroc": probe_val,
        "probe_threshold_val_thresh": probe_thr, "probe_f1_test": probe_f1,
        "floors": fl_auc, "tfidf_note": tfidf_note,
        "honest_floor_name": hname, "honest_floor": hauc,
        "margin_vs_honest_floor": probe_auc - hauc,
        "ci_lo_honest": lo_h, "ci_hi_honest": hi_h, "p_honest": p_h,
        "margin_vs_tfidf": probe_auc - fl_auc["tfidf"],
        "ci_lo_tfidf": lo_c, "ci_hi_tfidf": hi_c, "p_tfidf": p_c,
        "format_oracle_ceiling": fl_auc.get("format_oracle"),
        "oracle_note": ("TAUTOLOGICAL: kwargs-aware re-derivation of the verifier's "
                        "arithmetic. Excluded from both bars by construction.")
                       if oracle_defined else
                       ("UNDEFINED at the pooled prompt level: the label is a "
                        "conjunction over several different instructions, so there is "
                        "no single constraint margin. NOT 'at chance' — omitted."),
        "classification": classify(fl_auc),
        "perm_null_probe_mean": float(np.mean(perm)) if perm else None,
        "perm_null_probe_sd": float(np.std(perm)) if perm else None,
        "perm_null_tfidf_mean": float(np.mean(perm_tf)) if perm_tf else None,
        "refusal_selected": {"variant": rv, "layer": int(rl), "sign": rs},
        "yesno_selected": {"space": ysp, "sign": ysg},
        "class_counts": counts,
        "ceiling_note": (f"Probe AUROC at ceiling; one-sided 95% Clopper-Pearson bound "
                         f"0.05^(1/{npair}) = {clopper_upper(npair):.4f}")
                        if probe_auc >= 1.0 else None,
    }
    return row, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dir", default="results/rebuild/ifeval")
    ap.add_argument("--label", default="strict", choices=["strict", "loose"])
    ap.add_argument("--acts", default="acts512", choices=["acts512", "acts2048"])
    ap.add_argument("--out", default="IFEVAL_SUMMARY.json")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, args.dir)
    os.makedirs(os.path.join(out_dir, "splits"), exist_ok=True)

    labels = json.load(open(os.path.join(out_dir, f"labels_seed{args.seed}.json")))
    gpu = json.load(open(os.path.join(out_dir, f"gpu_features_seed{args.seed}.json")))
    z = np.load(os.path.join(out_dir, f"acts_seed{args.seed}.npz"))
    inst = labels["instances"]
    rows = np.array(gpu["instance_rows"])
    ACTS = z[args.acts]
    yn = {k[3:]: z[k] for k in z.files if k.startswith("yn_")}
    refusal = {"raw": z["refusal_raw"], "chat": z["refusal_chat"]}
    lab_key = f"{args.label}_follow"

    rng = np.random.default_rng(args.seed)
    results, dropped = [], []

    by_cat = collections.defaultdict(list)
    for k, i in enumerate(inst):
        by_cat[i["category"]].append(
            dict(key=i["key"], instruction_id=i["instruction_id"], kwargs=i["kwargs"],
                 response=i["response"], row=rows[k], row_inst=k,
                 y=0 if i[lab_key] else 1))

    for cat in sorted(by_cat):
        r, d = analyze(cat, by_cat[cat], ACTS, refusal, yn, rng, out_dir,
                       args.seed, gpu, args.label, args.acts)
        if r:
            results.append(r)
            json.dump(r, open(os.path.join(out_dir,
                      f"category_{cat}_seed{args.seed}.json"), "w"), indent=1)
            print(f"{cat:<22} n={r['n']:<4} Probe={r['probe']:.3f} "
                  f"honest={r['honest_floor']:.3f}({r['honest_floor_name']}) "
                  f"orc={r['format_oracle_ceiling']:.3f} perm={r['perm_null_probe_mean']}")
        else:
            dropped.append(d)
            print(f"{cat:<22} DROPPED: {d['reason']}")

    # ── pooled prompt-level fit: one activation per prompt, label = follow_all ──
    pl = collections.defaultdict(list)
    for k, i in enumerate(inst):
        pl[i["key"]].append(k)
    prec = []
    for key, ks in pl.items():
        allc = all(inst[k][lab_key] for k in ks)
        prec.append(dict(key=key, instruction_id="__ALL__", kwargs={},
                         response=inst[ks[0]]["response"], row=rows[ks[0]],
                         row_inst=ks[0], y=0 if allc else 1))
    pooled, pdrop = analyze("ALL_PROMPTS_pooled", prec, ACTS, refusal, yn, rng,
                            out_dir, args.seed, gpu, args.label, args.acts)
    if pooled:
        json.dump(pooled, open(os.path.join(out_dir,
                  f"category_ALL_PROMPTS_pooled_seed{args.seed}.json"), "w"), indent=1)
        print(f"\nPOOLED prompt-level: n={pooled['n']} n_test={pooled['n_test']} "
              f"Probe={pooled['probe']:.3f} "
              f"honest={pooled['honest_floor']:.3f}({pooled['honest_floor_name']}) "
              f"tfidf={pooled['floors']['tfidf']:.3f} orc=UNDEFINED(conjunction)")
        print(f"  vs tfidf : {pooled['margin_vs_tfidf']:+.3f} "
              f"CI=[{pooled['ci_lo_tfidf']:+.3f},{pooled['ci_hi_tfidf']:+.3f}] "
              f"p={pooled['p_tfidf']:.3f}")
        print(f"  vs honest: {pooled['margin_vs_honest_floor']:+.3f} "
              f"CI=[{pooled['ci_lo_honest']:+.3f},{pooled['ci_hi_honest']:+.3f}] "
              f"p={pooled['p_honest']:.3f}")

    if results:
        for key, pk in [("q_honest", "p_honest"), ("q_tfidf", "p_tfidf")]:
            for r, q in zip(results, bh([x[pk] for x in results])):
                r[key] = float(q)
        for r in results:
            r["beats_honest_floor"] = bool(r["ci_lo_honest"] > 0 and r["q_honest"] < 0.05)
            r["beats_tfidf"] = bool(r["ci_lo_tfidf"] > 0 and r["q_tfidf"] < 0.05)

    summary = {
        "provenance": {
            "model": gpu["provenance"]["model"], "seed": args.seed,
            "git_commit": git_commit(),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "script": "scripts/run_ifeval_probe.py",
            "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                         "torch": gpu["provenance"]["versions"]["torch"],
                         "transformers": gpu["provenance"]["versions"]["transformers"],
                         "python": sys.version.split()[0]},
            "split_fractions": {"train": 0.40, "val_layer": 0.20,
                                "val_thresh": 0.20, "test": 0.20},
            "max_length": gpu["provenance"]["max_length"],
            "max_length_variant_used": args.acts,
            "pooling": gpu["provenance"]["pooling"],
            "generation_config": gpu["provenance"]["generation_config"],
            "gpu": gpu["provenance"]["gpu"],
            "verifier": "google-research/instruction_following_eval via lm_eval.tasks.ifeval",
        },
        "label_convention": "AUROC for detecting VIOLATION (higher score = violation). "
                            "Probe fit higher=compliant then negated.",
        "deviations": {
            "bootstrap": "two-sample stratified (IFEval has NO pairs) — deviation from "
                         "the paired-bootstrap brief",
            "split": "three_way_split applied to UNIQUE PROMPT KEYS, not instances: 33 "
                     "prompts carry 2-3 same-category instructions sharing one "
                     "activation vector; instance-level splitting would leak",
        },
        "floor_policy": {
            "honest_floors": HONEST_FLOORS,
            "excluded_tautological": TAUTOLOGICAL_FLOORS,
            "why": "format_oracle is kwargs-aware and re-derives the verifier's "
                   "arithmetic; beating it is impossible by construction. It is a "
                   "CEILING measuring how mechanically form-determined the label is.",
        },
        "truncation": gpu["truncation"],
        "refusal_direction_check": gpu["refusal_direction_check"],
        "n_categories_kept": len(results), "n_categories_dropped": len(dropped),
        "dropped": dropped,
        "pooled_prompt_level": pooled,
        "pooled_dropped": pdrop,
        "results": sorted(results, key=lambda r: -r["margin_vs_honest_floor"]),
    }
    json.dump(summary, open(os.path.join(out_dir, args.out), "w"), indent=1)

    print(f"\n{'category':<20}{'n':>4}{'Probe':>7}{'tfidf':>7}{'len':>6}{'kw':>6}{'fmt':>6}"
          f"{'refus':>7}{'yn':>6}{'|orcl':>7}  {'honest':<13}{'marg_h':>8}{'q_h':>6}"
          f"{'marg_tf':>8}{'q_tf':>6}")
    for r in summary["results"]:
        f = r["floors"]
        print(f"{r['category']:<20}{r['n']:>4}{r['probe']:>7.3f}{f['tfidf']:>7.3f}"
              f"{f['length_char']:>6.3f}{f['keyword']:>6.3f}{f['format_feats']:>6.3f}"
              f"{f['refusal_dir']:>7.3f}{f['yesno']:>6.3f}{f.get('format_oracle', float('nan')):>7.3f}  "
              f"{r['honest_floor_name']:<13}{r['margin_vs_honest_floor']:>+8.3f}"
              f"{r['q_honest']:>6.3f}{r['margin_vs_tfidf']:>+8.3f}{r['q_tfidf']:>6.3f}")
    print(f"\ndropped: {[(d['category'], d['reason']) for d in dropped]}")
    if results:
        print(f"beats TF-IDF (content bar): "
              f"{sum(r['beats_tfidf'] for r in results)}/{len(results)}")
        print(f"beats honest floor (form bar): "
              f"{sum(r['beats_honest_floor'] for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
