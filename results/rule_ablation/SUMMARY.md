# E-RULE — **Probe does not read the rule.** The central finding.

*2026-07-17 · seed 42 · Llama-3.2-1B-Instruct · GPU 0 (`safetune`, torch 2.11) · 20 ablated
domains · seed-42 splits · Probe refit fresh per condition · null = 200 random directions
through the same layer-argmax budget, recomputed on each condition's own activations.
Written by the orchestrator from the agent's verbatim report. Receipts:
`rule_ablation_<domain>_seed42.json` ×20, `RULE_ABLATION_SUMMARY.json`. Script:
`scripts/run_rule_ablation.py`.*

## THE RESULT

**Probe**

| condition | mean AUROC | budget null | margin | Δ vs correct | #BH-sig drop | #BH-sig gain |
|---|---:|---:|---:|---:|---:|---:|
| `correct` | 0.9519 | 0.7135 | +0.2384 | — | — | — |
| **`no_rule`** | **0.9549** | 0.7191 | +0.2357 | **−0.0029** | **0/20** | **0/20** |
| `shuffled_rule` | 0.9548 | 0.7127 | +0.2422 | −0.0029 | 0/20 | 0/20 |
| `wrong_domain_rule` | 0.9554 | 0.7121 | +0.2432 | −0.0034 | 0/20 | 0/20 |
| `rule_only` | 0.5627 | 0.5265 | +0.0362 | +0.3892 | **19/20** | 0/20 |

**TF-IDF** (null 0.5, zero selection)

| condition | mean AUROC | Δ vs correct | #drop | #gain |
|---|---:|---:|---:|---:|
| `correct` | 0.8956 | — | — | — |
| **`no_rule`** | **0.9308** | **−0.0352** | 0/20 | **13/20** |
| `shuffled_rule` | 0.9238 | −0.0282 | 0/20 | 3/20 |
| `wrong_domain_rule` | 0.9229 | −0.0273 | 0/20 | 4/20 |
| `rule_only` | 0.5881 | +0.3075 | 19/20 | 0/20 |

## THE VERDICT

> **The Activation Probe Score does not measure compliance with a rule. It measures
> whether a scenario looks like a violation.**

Deleting the Rule block entirely moves mean AUROC by **−0.0029** (it slightly *helps*) with
a BH-significant drop in **0 of 20 domains** — and **0 significant gains**, so the deltas are
symmetric noise, not an underpowered near-miss. Four independent corruptions agree:
**removing** the rule (0.9549), **deranging** it within-domain (0.9548), and **swapping in
another domain's rule** — identical for both classes, carrying *zero* class information —
(0.9554) all leave `correct`'s 0.9519 untouched.

**This is not the machinery failing to move.** `rule_only` collapses (0.5627, 19/20 BH-sig
drop) — the positive control proving the pipeline detects a real degradation when one exists.

## Why the null is real and not an artifact

- **Length confound ruled out, and not by truncation alone.** `shuffled_rule` (136.1 tokens)
  and `wrong_domain_rule` (136.3) are **token-length-identical** to `correct` (135.9) and show
  the same ~zero change. `no_rule` is shorter (87.6) but gives the *same* answer as the
  length-matched conditions. Backstop: **0 of 17,164 test texts exceed the 512 cap in any
  condition** (max 370 tokens), so removing the rule never un-truncates a scenario.
- **Symmetric pre-BH deltas**: 2 nominal drops (data_act +0.027 p=.017; sb35 +0.011 p=.031)
  against 2 nominal gains (foundation_rights −0.024 p=.008; policy_github −0.016 p=.041).
- **Step-1 gate: 20/20 bit-exact vs GATE.json** (max |Δ| = **0.0**), anchor layers match 20/20.

## `rule_only`: the label is NOT recoverable from the rule alone

0.5627 vs a 0.5265 null (margin +0.036), clearing its own null in only **3/20** domains (mean
tail p 0.32) — unlike SafePyramid, where rule-only reached 0.90+. Two exceptions, both the
smallest domains: `edu_discrimination` (0.822, n=16) and `cybersecurity_mitre_attack` (0.750,
n=36) — the only whiff of rule-text label leakage. Verdict unchanged.

## THREE INDEPENDENT CONFIRMATIONS OF THE SAME FINDING

| evidence | benchmark | method | result |
|---|---|---|---|
| **E-RULE** (this) | OmniCompliance | Probe (internal probe) | rule deletion: **0/20** significant drops |
| **E-FLIP** | SafePyramid flip-pairs | Probe (internal probe) | level-balanced **0.5000, below its own null** |
| **E-COMP-3** | OmniCompliance | LPG_4B (deployed guard) | no-rule **0.9249 > correct-rule 0.9214** |

Two benchmarks, two architectures (an internal difference-of-means probe and a 4B deployed
policy guard), same conclusion: **these systems do not condition on the supplied rule.**

## Why this explains everything else in the audit

The finding is not one more negative — it is the **mechanism** behind the others:

- **The 20 domain directions collapse to one** (top-1 PC 82.4%, LODO Δ=+0.0005) — because
  they are all the same *bad-text* axis, not 20 rule-specific geometries.
- **A refusal direction with zero compliance supervision reaches 0.863** — same axis.
- **A movie-review sentiment direction is the max floor in 14/20** — valence ≈ badness.
- **TF-IDF reaches 0.896, and *improves to 0.931 without the rule*** (13/20 significant
  gains) — a bag of words recovers the label from the scenario better once rule tokens stop
  diluting it. **This is the generic violation-description axis, measured.**
- **Steering *writes* the verdict** (violating half 0.74→0.21) — it writes "bad", not
  "violates rule X".
- **The benchmarks are degenerate** — because the label *is* the badness.
- **E-T2's "undecidable within a single-template benchmark"** — no longer undecidable.
  Decided, and against the probe.

## Scope limits (state them)

- This shows the rule is **unnecessary**, not that the correct-condition probe never attends
  to rule tokens. That distinction does not rescue a rule-conditioned detection claim:
  identical performance without the rule means the claim is unsupported regardless.
- Anchor layer shifts between `correct` and `no_rule` in 10/20 domains — expected when
  refitting fresh on a changed input distribution.
- The null-machinery check (correct-condition null 0.7135, Probe 0.9519 matching
  `selection_nulls/SUMMARY.md` exactly) is a **consistency** check, not an independent
  replication — same protocol, seed, and data.
- Benchmark structure asserted: rules vary per case (gdpr 87 distinct / 100); pairs sharing a
  rule 35/8582 pooled (<0.5%, under the 5% abort threshold); derangement complete 20/20 with
  the rule multiset preserved.

## Sanity

Splits index-identical to `results/rebuild/splits/` (incl. all 20 donor domains); **0
`Outcome:` strings across all 85,820 scored texts**; every AUROC through `auroc_checked`
(sklearn <1e-9); `auroc_matrix` validated to 1e-12 with heavy-tie stress; BH exact-`Fraction`
with **0 disagreements** vs the float path; GPU-0 PID verified on `GPU-b021b5e4`.
