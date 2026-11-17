#!/usr/bin/env python
"""
E-B3 — shared unit builders for the third benchmark wave.

Benchmarks (acquisition receipts in results/rebuild/benchmark3/*.json):

  AIReg-Bench   HF: camlsys/AIReg-Bench @ c656725c22568ef5fa489ea3940d3b6d7ff4b2ff
                (cc-by-4.0, arXiv:2510.01474). NOTE: the HF repo ships ONLY the
                annotation parquets + intended_uses.txt — the 300 documentation
                .txt files its own README describes are NOT in the HF repo.
                Texts come from the official GitHub camlsys/aireg-bench
                @ 846d5cc075ad424a2d31dd20e34a7c5fe836f157.
                unit = documentation excerpt. 300 = 5 articles x 10 uses x
                2 systems x 3 profiles (Compliance / Violation_1 / Violation_2).
                GROUPING: system outlines live in system_outlines/Use*/System*.txt
                and are SHARED ACROSS ARTICLES — the same fictional system is
                elaborated once per article. Group key = (use, system); a
                row-level split would put the same system's docs in train+test.
                Two label tracks:
                  construction — profile == Compliance (semi-circular: different
                                 files from different prompts; secondary)
                  expert       — median/mean human compliance score > 3
                                 (the 120-excerpt human-annotated subset; PRIMARY
                                 — this is the provenance-clean target)

  CompliBench   NOT on the HF hub (hub search "CompliBench" -> 0 repos,
                2026-07-17). GitHub: UCSB-NLP-Chang/CompliBench
                @ 593543022f67914e7ec9a312b7fa39c8a75e508a (Apache-2.0).
                unit = assistant turn; label = turn_index in `mistakes`.
                GROUPING: conversation (turns share dialogue text).

  FlexBench     HF: Tommy-DING/FlexBench @ 8f73c40bc3f2c272af8ee8d33a4003029ba4282a
                (apache-2.0, arXiv:2602.23636). Two tracks (prompt / response
                moderation), 5-tier severity, graded binary labels
                label_strict / label_moderate / label_loose (1 = FLAGGED).
                unit = unique input text (test+validation pooled, exact-dedup;
                keys whose duplicates disagree on severity are dropped and
                counted). No further grouping structure exists.

Orientation everywhere: label 1 = COMPLIANT, higher score = MORE COMPLIANT
(the E-B2 convention; AUROC invariant to the joint flip).
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

SCRATCH = os.environ.get("BENCHMARK3_SCRATCH", "./scratchpad/b3data_persist")
AIREG_GIT = f"{SCRATCH}/aireg"                       # camlsys/aireg-bench clone
AIREG_HF = f"{SCRATCH}/b3data/aireg_hf"              # camlsys/AIReg-Bench snapshot
COMPLIBENCH = f"{SCRATCH}/complibench"               # UCSB-NLP-Chang clone
FLEXBENCH = f"{SCRATCH}/b3data/flexbench_hf"         # Tommy-DING/FlexBench snapshot

REVISIONS = {
    "aireg_hf_repo": "camlsys/AIReg-Bench",
    "aireg_hf_revision": "c656725c22568ef5fa489ea3940d3b6d7ff4b2ff",
    "aireg_hf_license": "cc-by-4.0",
    "aireg_github": "https://github.com/camlsys/aireg-bench",
    "aireg_github_commit": "846d5cc075ad424a2d31dd20e34a7c5fe836f157",
    "aireg_note": ("HF repo lacks the 300 documentation .txt files described in "
                   "its own README; texts taken from the official GitHub repo."),
    "complibench_hf": "NOT ON HF HUB (search 'CompliBench', 0 repos, 2026-07-17)",
    "complibench_github": "https://github.com/UCSB-NLP-Chang/CompliBench",
    "complibench_github_commit": "593543022f67914e7ec9a312b7fa39c8a75e508a",
    "complibench_license": "Apache-2.0",
    "flexbench_hf_repo": "Tommy-DING/FlexBench",
    "flexbench_hf_revision": "8f73c40bc3f2c272af8ee8d33a4003029ba4282a",
    "flexbench_hf_license": "apache-2.0",
}

NUM2LAB = {"1": "Compliance", "2": "Violation_1", "3": "Violation_2"}


# ── AIReg-Bench ──────────────────────────────────────────────────────────────
def load_articles():
    """Parse articles.txt -> {article_int: full text}. Headers look like
    '**Article 9**' or '**Article 12 (1-2)**'."""
    arts, cur, buf = {}, None, []
    for line in open(f"{AIREG_GIT}/articles.txt", encoding="utf-8"):
        m = re.match(r"\*\*Article (\d+)", line.strip())
        if m:
            if cur is not None:
                arts[cur] = "".join(buf).strip()
            cur, buf = int(m.group(1)), []
        else:
            buf.append(line)
    if cur is not None:
        arts[cur] = "".join(buf).strip()
    assert set(arts) == {9, 10, 12, 14, 15}, sorted(arts)
    return arts


def build_aireg():
    """All 300 docs + construction label + (where available) expert label.

    Expert mapping reuses the parser validated in the provenance audit
    (context/BENCHMARK_PROVENANCE_AUDIT.md): rand_assign.txt maps scenario
    letters A/B/C -> file labels per (use, article); human_annotations.xlsx
    holds one aggregated compliance score per annotated excerpt. The resolver
    prefers System1 (all 40 annotated cells resolve there); the 95%
    expert/intended agreement is the empirical check that this mapping is sound.
    """
    rows = []
    for p in sorted(glob.glob(f"{AIREG_GIT}/documentation/*/*/*/*.txt")):
        m = re.search(r"Article(\d+)/Use(\d+)/System(\d+)/(\w+)\.txt$", p)
        art, use, system, prof = (int(m.group(1)), int(m.group(2)),
                                  int(m.group(3)), m.group(4))
        rows.append(dict(
            article=art, use=use, system=system, profile=prof,
            group=f"Use{use}/System{system}",
            y_construction=int(prof == "Compliance"),
            text=open(p, encoding="utf-8", errors="ignore").read(),
            path=os.path.relpath(p, AIREG_GIT),
        ))
    df = pd.DataFrame(rows)
    assert len(df) == 300, len(df)

    # rand_assign: (use, article) -> {scenario_letter: file_label}
    assign, cur = {}, None
    for line in open(f"{AIREG_GIT}/rand_assign.txt"):
        line = line.strip()
        m = re.match(r"Use (\d+), Article (\d+)", line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)))
            assign[cur] = {}
            continue
        m = re.match(r"([123])\s*--\s*([ABC])", line)
        if m and cur:
            assign[cur][m.group(2)] = NUM2LAB[m.group(1)]

    # human_annotations: xlsx in the GitHub clone == parquet in the HF repo
    h = pd.read_excel(f"{AIREG_GIT}/human_annotations.xlsx", header=None)
    use_cols = {}
    for c in range(h.shape[1]):
        m = re.match(r"Use (\d+)", str(h.iloc[0, c]))
        if m:
            use_cols[int(m.group(1))] = c
    row_ids = {}
    for r in range(2, h.shape[0]):
        m = re.match(r"Art (\d+) / Scenario ([ABC])", str(h.iloc[r, 0]))
        if m:
            row_ids[r] = (int(m.group(1)), m.group(2))

    expert = {}
    for use, c0 in sorted(use_cols.items()):
        for r, (art, scen) in row_ids.items():
            raw = h.iloc[r, c0]
            if pd.isna(raw):
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            flab = assign.get((use, art), {}).get(scen)
            if flab is None:
                continue
            # resolver prefers System1 (see docstring)
            for system in (1, 2):
                key = (art, use, system, flab)
                if ((df.article == art) & (df.use == use) &
                        (df.system == system) & (df.profile == flab)).any():
                    expert[key] = score
                    break

    df["human_compliance"] = [
        expert.get((a, u, s, p), np.nan)
        for a, u, s, p in zip(df.article, df.use, df.system, df.profile)
    ]
    df["y_expert"] = np.where(df.human_compliance.isna(), np.nan,
                              (df.human_compliance > 3).astype(float))
    return df, load_articles()


# ── CompliBench ──────────────────────────────────────────────────────────────
def build_complibench():
    rows, skipped = [], []
    for f in sorted(glob.glob(f"{COMPLIBENCH}/data/*/*.json")):
        d = json.load(open(f))
        domain = f.split("/")[-2]
        conv_id = f"{domain}/{os.path.basename(f)}"
        if "message_list" not in d:
            skipped.append(conv_id)
            continue
        bad = {int(m["turn_index"]) for m in d.get("mistakes", [])}
        # generation metadata: guideline_text on a violating turn is the
        # MODIFIED (violation-inducing) directive, which is NOT in the
        # judge-facing assistant_guidelines (only the original is). Build a
        # modified->original map so a judge-faithful "fair" guideline exists.
        mod2orig = {}
        for e in (d.get("violation_directives", []) +
                  d.get("cat2_overrides", []) + d.get("cat3_overrides", [])):
            if e.get("modified") and e.get("original"):
                mod2orig[e["modified"]] = e["original"]
        hist = []
        for m in d["message_list"]:
            hist.append((m["role"], m.get("content", "")))
            if m.get("role") != "assistant":
                continue
            ti = int(m["turn_index"])
            gl = m.get("guideline_text", "")
            ctx = "\n".join(f"[{r.capitalize()}]: {c}" for r, c in hist)
            rows.append(dict(
                domain=domain, conv_id=conv_id, turn_index=ti,
                turn_text=m.get("content", ""),
                guideline_text=gl,
                guideline_fair=mod2orig.get(gl, gl),
                guideline_was_modified=int(gl in mod2orig),
                context_text=ctx,                # dialogue up to & incl. this turn
                y_compliant=int(ti not in bad),
            ))
    df = pd.DataFrame(rows)
    assert (df.guideline_text.str.len() > 0).all(), "missing guideline_text"
    return df, skipped


# ── FlexBench ────────────────────────────────────────────────────────────────
SEV_ORDER = {"benign": 0, "low": 1, "moderate": 2, "high": 3, "extreme": 4}


def build_flexbench(track):
    """track in {'prompt', 'response'} -> (df, dedup_report). Pools the
    benchmark's test+validation files, exact-dedups on input text, drops keys
    whose duplicates disagree on severity."""
    assert track in ("prompt", "response")
    parts = []
    for split in ("test", "valid"):
        p = pd.read_csv(f"{FLEXBENCH}/Guard_{track}_{split}.csv")
        p["split_origin"] = split
        parts.append(p)
    raw = pd.concat(parts, ignore_index=True)
    n_raw = len(raw)

    conflict_keys = set()
    for txt, g in raw.groupby("input"):
        if g.severity.nunique() > 1:
            conflict_keys.add(txt)
    dedup = (raw[~raw.input.isin(conflict_keys)]
             .drop_duplicates(subset="input", keep="first")
             .reset_index(drop=True))
    for c in ("label_strict", "label_moderate", "label_loose"):
        # y_compliant convention: benchmark label 1 = FLAGGED (violation)
        dedup[f"y_compliant_{c.split('_')[1]}"] = 1 - dedup[c].astype(int)
    report = {
        "n_raw_rows_test_plus_validation": int(n_raw),
        "n_exact_duplicate_rows_removed": int(n_raw - len(dedup)
                                              - sum(raw.input.isin(conflict_keys))),
        "n_rows_dropped_for_conflicting_severity_dups": int(
            raw.input.isin(conflict_keys).sum()),
        "n_conflicting_keys": len(conflict_keys),
        "n_units": int(len(dedup)),
    }
    return dedup, report
