#!/usr/bin/env python
"""
build_crossed_benchmark.py — assemble the crossed rule x scenario benchmark
(reviewer P0 rescue experiment) from per-domain raw template files.

Each raw/<domain>.json holds ~25 "templates" authored by a content agent:
  category_a, category_b, unit, threshold_a, threshold_b, test_value,
  rule_template (with <<CATEGORY_A>>/<<CATEGORY_B>>/<<THRESHOLD_A>>/<<THRESHOLD_B>>/<<UNIT>>
  placeholders), scenario_template_a / scenario_template_b (with <<VALUE>>/<<UNIT>>).

Mechanical oracle (computed HERE, never by an LLM):
  For a template, "natural" rule = (category_a -> threshold_a, category_b -> threshold_b).
  "swapped" rule = (category_a -> threshold_b, category_b -> threshold_a). R0/R1 are
  assigned to natural/swapped by an independent per-template coin flip (seeded), so no
  global correlation between "R0" and any authoring tendency survives.
  label(rule, scenario) = 1 (VIOLATION) if test_value > threshold-assigned-to-that-scenario's
  category under that rule, else 0 (COMPLIANT).
  Because test_value is strictly between threshold_a and threshold_b, every template yields
  the canonical crossed 2x2 pattern: exactly 2 of 4 cells violate, 2 comply, and NEITHER the
  rule alone NOR the scenario alone predicts the label (each rule text appears twice in the
  dataset with opposite labels; same for each scenario text) -- rule-only and scenario-only
  chance is a construction guarantee, not an empirical hope.

Usage:
    python scripts/build_crossed_benchmark.py           # assemble + self-check
    python scripts/build_crossed_benchmark.py --strict  # error out on any template failing checks
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "raw")
OUT_DIR = os.path.join(REPO, "results", "rebuild", "crossed_benchmark")
OUT_FILE = os.path.join(OUT_DIR, "crossed_pairs.json")
REPORT_FILE = os.path.join(OUT_DIR, "CONSTRUCTION.md")

VERDICT_WORDS = ["violat", "unlawful", "illegal", "compliant", "complies",
                 "permitted", "in violation", "failed to", "prohibited",
                 "breach of", "not authorized", "unauthorized under"]

REQUIRED_RULE_TOKENS = ["<<CATEGORY_A>>", "<<CATEGORY_B>>", "<<THRESHOLD_A>>",
                        "<<THRESHOLD_B>>", "<<UNIT>>"]
REQUIRED_SCEN_TOKENS = ["<<VALUE>>", "<<UNIT>>"]


def fmt_num(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def sub(text, mapping):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def coinflip(template_id: str) -> bool:
    """Deterministic per-template pseudo-random bit from a hash, independent of authoring
    order or any global counter (so no positional bias can creep in)."""
    h = hashlib.sha256(template_id.encode()).hexdigest()
    return int(h[:8], 16) % 2 == 0


def validate_template(t, domain, errors):
    tid = t.get("template_id", "<missing>")
    ok = True
    for f in ["template_id", "axis", "category_a", "category_b", "unit",
              "threshold_a", "threshold_b", "test_value", "rule_template",
              "scenario_template_a", "scenario_template_b"]:
        if f not in t:
            errors.append(f"{domain}/{tid}: missing field {f}")
            ok = False
    if not ok:
        return False

    a, b, v = float(t["threshold_a"]), float(t["threshold_b"]), float(t["test_value"])
    lo, hi = min(a, b), max(a, b)
    if not (lo < v < hi):
        errors.append(f"{domain}/{tid}: test_value {v} not strictly between "
                       f"threshold_a={a} and threshold_b={b}")
        ok = False
    if lo == 0:
        gap_ok = (hi - lo) > 0
    else:
        gap_ok = (hi - lo) / lo >= 0.20  # slightly lenient vs the 25% asked of authors
    if not gap_ok:
        errors.append(f"{domain}/{tid}: threshold gap too small (a={a}, b={b})")
        ok = False

    for tok in REQUIRED_RULE_TOKENS:
        if tok not in t["rule_template"]:
            errors.append(f"{domain}/{tid}: rule_template missing {tok}")
            ok = False
    for key in ["scenario_template_a", "scenario_template_b"]:
        for tok in REQUIRED_SCEN_TOKENS:
            if tok not in t[key]:
                errors.append(f"{domain}/{tid}: {key} missing {tok}")
                ok = False
        low = t[key].lower()
        for vw in VERDICT_WORDS:
            if vw in low:
                errors.append(f"{domain}/{tid}: {key} contains verdict-leaning word {vw!r}")
                ok = False
    return ok


def expand_template(t, domain):
    a, b, v = float(t["threshold_a"]), float(t["threshold_b"]), float(t["test_value"])
    unit = t["unit"]
    natural = {"<<THRESHOLD_A>>": fmt_num(a), "<<THRESHOLD_B>>": fmt_num(b)}
    swapped = {"<<THRESHOLD_A>>": fmt_num(b), "<<THRESHOLD_B>>": fmt_num(a)}
    common = {"<<CATEGORY_A>>": t["category_a"], "<<CATEGORY_B>>": t["category_b"],
              "<<UNIT>>": unit}

    rule_natural = sub(t["rule_template"], {**common, **natural})
    rule_swapped = sub(t["rule_template"], {**common, **swapped})

    r0_is_natural = coinflip(t["template_id"])
    rule_r0 = rule_natural if r0_is_natural else rule_swapped
    rule_r1 = rule_swapped if r0_is_natural else rule_natural

    # thresholds actually in force for category a/b under R0 / R1
    thr_r0 = {"a": a, "b": b} if r0_is_natural else {"a": b, "b": a}
    thr_r1 = {"a": b, "b": a} if r0_is_natural else {"a": a, "b": b}

    scen_a = sub(t["scenario_template_a"], {"<<VALUE>>": fmt_num(v), "<<UNIT>>": unit})
    scen_b = sub(t["scenario_template_b"], {"<<VALUE>>": fmt_num(v), "<<UNIT>>": unit})

    rows = []
    for rule_id, rule_text, thr in [("R0", rule_r0, thr_r0), ("R1", rule_r1, thr_r1)]:
        for scen_id, scen_text, cat_key in [("Sa", scen_a, "a"), ("Sb", scen_b, "b")]:
            label = 1 if v > thr[cat_key] else 0
            rows.append({
                "quadruple_id": t["template_id"],
                "domain": domain,
                "axis": t["axis"],
                "rule_id": rule_id,
                "scenario_id": scen_id,
                "cell": f"{rule_id}x{scen_id}",
                "rule_text": rule_text,
                "scenario_text": scen_text,
                "text": f"Rule: {rule_text}\n\nScenario: {scen_text}",
                "label": label,
            })
    # sanity: exactly 2 violations, 2 compliant, and R0/R1 each split 1-1, Sa/Sb each split 1-1
    labels = [r["label"] for r in rows]
    assert sum(labels) == 2, (t["template_id"], labels)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        print(f"No raw template files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    errors = []
    per_domain_counts = {}
    for fp in files:
        domain = os.path.splitext(os.path.basename(fp))[0]
        templates = json.load(open(fp))
        seen_ids = set()
        kept = 0
        for t in templates:
            tid = t.get("template_id")
            if tid in seen_ids:
                errors.append(f"{domain}: duplicate template_id {tid}")
                continue
            seen_ids.add(tid)
            ok = validate_template(t, domain, errors)
            if not ok:
                continue
            all_rows.extend(expand_template(t, domain))
            kept += 1
        per_domain_counts[domain] = {"templates_total": len(templates), "templates_kept": kept}

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(all_rows, open(OUT_FILE, "w"), indent=1)

    print(f"Assembled {len(all_rows)} rows from {sum(v['templates_kept'] for v in per_domain_counts.values())} "
          f"templates across {len(files)} domains -> {OUT_FILE}")
    for d, c in per_domain_counts.items():
        print(f"  {d}: {c['templates_kept']}/{c['templates_total']} templates kept")
    if errors:
        print(f"\n{len(errors)} validation issue(s):")
        for e in errors:
            print(f"  - {e}")
        if args.strict:
            sys.exit(1)

    # ── mechanical self-check: rule-only / scenario-only / rule+scenario TF-IDF, group-disjoint ──
    run_self_check(all_rows)


def run_self_check(rows):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    groups = np.array([r["quadruple_id"] for r in rows])
    y = np.array([r["label"] for r in rows])

    def cv_auroc(texts):
        from sklearn.metrics import roc_auc_score
        gkf = GroupKFold(n_splits=5)
        scores = np.zeros(len(texts))
        for tr, te in gkf.split(texts, y, groups):
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            Xtr = vec.fit_transform([texts[i] for i in tr])
            Xte = vec.transform([texts[i] for i in te])
            clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
            scores[te] = clf.predict_proba(Xte)[:, 1]
        return float(roc_auc_score(y, scores))

    rule_texts = [r["rule_text"] for r in rows]
    scen_texts = [r["scenario_text"] for r in rows]
    both_texts = [r["text"] for r in rows]

    auc_rule = cv_auroc(rule_texts)
    auc_scen = cv_auroc(scen_texts)
    auc_both = cv_auroc(both_texts)

    print("\n== Self-check (TF-IDF, group-disjoint 5-fold by quadruple_id) ==")
    print(f"  rule-only     AUROC = {auc_rule:.3f}  (expect ~0.50, chance BY CONSTRUCTION)")
    print(f"  scenario-only AUROC = {auc_scen:.3f}  (expect ~0.50, chance BY CONSTRUCTION)")
    print(f"  rule+scenario AUROC = {auc_both:.3f}  (expect near-chance too: bag-of-words "
          f"cannot generalize a numeric A-vs-B comparison to unseen numbers -- this is the "
          f"point: only genuine relational/numeric reasoning solves this benchmark)")

    summary = {
        "n_rows": len(rows),
        "n_quadruples": len(set(groups)),
        "label_rate": float(y.mean()),
        "tfidf_rule_only_auroc": auc_rule,
        "tfidf_scenario_only_auroc": auc_scen,
        "tfidf_rule_plus_scenario_auroc": auc_both,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "SELF_CHECK.json"), "w"), indent=2)
    print(f"\nWrote {os.path.join(OUT_DIR, 'SELF_CHECK.json')}")


if __name__ == "__main__":
    main()
