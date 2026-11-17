# G2 — Trivial-baseline gauntlet (SUMMARY)

*Gate experiment G2 (FINAL_PLAN §3 G2, §5 Tier-1). Generated 2026-07-17T00:20:02.446186+05:30 · commit `58217a82c0ab0dc32c2a07791dc0cd6a9f439054` · seed 42 · CPU only, no LLM loaded · script `scripts/run_trivial_gauntlet.py`.*

All baselines are fit on the 40% train slice and reported as AUROC on the 20% test
slice of the seed-42 `three_way_split`, on the exact per-domain samples used by the
activation pipeline (`run_omni_v1.py`). The `val_layer` / `val_thresh` slices are never
touched. Label convention: 1 = VIOLATES, 0 = COMPLIES; every AUROC detects violation.

Columns:

- **TFIDF-full** — TF-IDF unigram + logistic regression on the full input text.
- **TFIDF-outcome** — same, on the `process_and_outcome` field alone (with-Outcome variant only).
- **length** — raw input length as a score, sign oriented on train (best of char / whitespace-token).
- **keyword-cont** — continuous net violation-marker count from `guardrails_baseline.py`'s
  lexicons (never binarized 0/1 — see FINAL_PLAN §7 dropped ledger).
- **perm-null** — TFIDF-full with train labels shuffled, mean of 5 seeded shuffles. Harness
  check only; must sit ≈0.5. **Excluded from max-of-baselines** (it is a null, not a competitor).
- **sentiment** — per-domain SST-2 sentiment-direction AUROC, copied verbatim from
  `results/sentiment_baseline_check.json` (needs GPU; not recomputed here).
- **max-of-baselines** — max(TFIDF-full, TFIDF-outcome where applicable, length, keyword-cont,
  sentiment). This is the floor the pre-registered gate rule compares ablated Probe against.

## Variant (a) — ablated: Rule + Scenario only (gate-relevant)

This is the gate-relevant variant. The Outcome field is absent, so TFIDF-outcome is N/A and the max-of-baselines column is the within-variant floor.

| domain | n_test | TFIDF-full | TFIDF-outcome | length | keyword-cont | perm-null | sentiment | **max-of-baselines** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gdpr | 200 | 0.830 | — | 0.514 | 0.488 | 0.473 | 0.837 | **0.837** |
| hipaa | 200 | 0.908 | — | 0.558 | 0.609 | 0.502 | 0.882 | **0.908** |
| eu_ai_act | 200 | 0.927 | — | 0.624 | 0.552 | 0.471 | 0.809 | **0.927** |
| ccpa | 200 | 0.946 | — | 0.505 | 0.563 | 0.498 | 0.745 | **0.946** |
| sb35 | 200 | 0.828 | — | 0.547 | 0.500 | 0.489 | 0.756 | **0.828** |
| finance_crypto | 200 | 0.948 | — | 0.510 | 0.672 | 0.488 | 0.853 | **0.948** |
| finance_anti_laundering_and_terrorist | 200 | 0.873 | — | 0.534 | 0.624 | 0.474 | 0.768 | **0.873** |
| finance_cross_border_payment_law | 126 | 0.823 | — | 0.494 | 0.597 | 0.493 | 0.780 | **0.823** |
| finance_eletric_momey | 200 | 0.935 | — | 0.563 | 0.555 | 0.497 | 0.774 | **0.935** |
| data_act | 200 | 0.827 | — | 0.542 | 0.543 | 0.498 | 0.802 | **0.827** |
| cybersecurity_mitre_attack | 72 | 0.989 | — | 0.542 | 0.554 | 0.567 | 0.904 | **0.989** |
| edu_academic_integrity | 72 | 0.993 | — | 0.672 | 0.408 | 0.554 | 0.931 | **0.993** |
| edu_online_learning | 144 | 0.988 | — | 0.594 | 0.568 | 0.508 | 0.969 | **0.988** |
| edu_discrimination_us_edu_dept | 32 | 0.887 | — | 0.461 | 0.559 | 0.552 | 0.492 | **0.887** |
| policy_github | 200 | 0.841 | — | 0.609 | 0.513 | 0.499 | 0.787 | **0.841** |
| policy_google | 200 | 0.852 | — | 0.625 | 0.550 | 0.507 | 0.834 | **0.852** |
| policy_openai | 200 | 0.832 | — | 0.586 | 0.588 | 0.489 | 0.751 | **0.832** |
| policy_reddit | 200 | 0.855 | — | 0.620 | 0.596 | 0.512 | 0.663 | **0.855** |
| policy_x | 200 | 0.928 | — | 0.609 | 0.537 | 0.462 | 0.737 | **0.928** |
| foundation_rights | 200 | 0.902 | — | 0.618 | 0.554 | 0.487 | 0.829 | **0.902** |
| **mean** | 172 | **0.896** | — | 0.566 | 0.556 | 0.501 | 0.795 | **0.896** |

## Variant (b) — with-Outcome: full Rule / Scenario / Outcome block

Appendix/confound exhibit only (FINAL_PLAN §4: with-Outcome numbers are appendix-only).

| domain | n_test | TFIDF-full | TFIDF-outcome | length | keyword-cont | perm-null | sentiment | **max-of-baselines** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gdpr | 200 | 0.851 | 0.850 | 0.516 | 0.725 | 0.478 | 0.837 | **0.851** |
| hipaa | 200 | 0.950 | 0.972 | 0.576 | 0.674 | 0.491 | 0.882 | **0.972** |
| eu_ai_act | 200 | 0.937 | 0.963 | 0.679 | 0.769 | 0.468 | 0.809 | **0.963** |
| ccpa | 200 | 0.978 | 0.985 | 0.547 | 0.621 | 0.492 | 0.745 | **0.985** |
| sb35 | 200 | 0.869 | 0.908 | 0.598 | 0.628 | 0.485 | 0.756 | **0.908** |
| finance_crypto | 200 | 0.977 | 0.989 | 0.561 | 0.894 | 0.489 | 0.853 | **0.989** |
| finance_anti_laundering_and_terrorist | 200 | 0.919 | 0.953 | 0.560 | 0.844 | 0.455 | 0.768 | **0.953** |
| finance_cross_border_payment_law | 126 | 0.904 | 0.971 | 0.536 | 0.867 | 0.470 | 0.780 | **0.971** |
| finance_eletric_momey | 200 | 0.983 | 0.997 | 0.601 | 0.847 | 0.496 | 0.774 | **0.997** |
| data_act | 200 | 0.904 | 0.934 | 0.596 | 0.797 | 0.490 | 0.802 | **0.934** |
| cybersecurity_mitre_attack | 72 | 0.999 | 0.996 | 0.600 | 0.604 | 0.548 | 0.904 | **0.999** |
| edu_academic_integrity | 72 | 0.998 | 0.999 | 0.748 | 0.446 | 0.542 | 0.931 | **0.999** |
| edu_online_learning | 144 | 0.998 | 1.000 | 0.640 | 0.678 | 0.506 | 0.969 | **1.000** |
| edu_discrimination_us_edu_dept | 32 | 0.887 | 0.996 | 0.592 | 0.730 | 0.557 | 0.492 | **0.996** |
| policy_github | 200 | 0.879 | 0.918 | 0.649 | 0.650 | 0.499 | 0.787 | **0.918** |
| policy_google | 200 | 0.923 | 0.972 | 0.671 | 0.681 | 0.488 | 0.834 | **0.972** |
| policy_openai | 200 | 0.895 | 0.939 | 0.625 | 0.705 | 0.485 | 0.751 | **0.939** |
| policy_reddit | 200 | 0.926 | 0.939 | 0.648 | 0.721 | 0.500 | 0.663 | **0.939** |
| policy_x | 200 | 0.958 | 0.969 | 0.620 | 0.807 | 0.473 | 0.737 | **0.969** |
| foundation_rights | 200 | 0.917 | 0.943 | 0.669 | 0.710 | 0.477 | 0.829 | **0.943** |
| **mean** | 172 | **0.932** | 0.960 | 0.612 | 0.720 | 0.494 | 0.795 | **0.960** |

## Ablated vs with-Outcome: is the ablated task still trivially separable?

| domain | n_test | TFIDF-full (ablated) | TFIDF-full (with-Outcome) | Δ (abl − outcome) |
|---|---:|---:|---:|---:|
| gdpr | 200 | 0.830 | 0.851 | -0.021 |
| hipaa | 200 | 0.908 | 0.950 | -0.042 |
| eu_ai_act | 200 | 0.927 | 0.937 | -0.010 |
| ccpa | 200 | 0.946 | 0.978 | -0.032 |
| sb35 | 200 | 0.828 | 0.869 | -0.040 |
| finance_crypto | 200 | 0.948 | 0.977 | -0.030 |
| finance_anti_laundering_and_terrorist | 200 | 0.873 | 0.919 | -0.046 |
| finance_cross_border_payment_law | 126 | 0.823 | 0.904 | -0.081 |
| finance_eletric_momey | 200 | 0.935 | 0.983 | -0.048 |
| data_act | 200 | 0.827 | 0.904 | -0.077 |
| cybersecurity_mitre_attack | 72 | 0.989 | 0.999 | -0.010 |
| edu_academic_integrity | 72 | 0.993 | 0.998 | -0.005 |
| edu_online_learning | 144 | 0.988 | 0.998 | -0.010 |
| edu_discrimination_us_edu_dept | 32 | 0.887 | 0.887 | +0.000 |
| policy_github | 200 | 0.841 | 0.879 | -0.038 |
| policy_google | 200 | 0.852 | 0.923 | -0.071 |
| policy_openai | 200 | 0.832 | 0.895 | -0.063 |
| policy_reddit | 200 | 0.855 | 0.926 | -0.071 |
| policy_x | 200 | 0.928 | 0.958 | -0.030 |
| foundation_rights | 200 | 0.902 | 0.917 | -0.014 |
| **mean** | | **0.896** | **0.932** | **-0.037** |

TF-IDF on **Rule + Scenario alone** reaches a mean AUROC of **0.896** (range 0.823–0.993), versus 0.932 with the Outcome field present — a mean drop of only 0.037. Removing the narrated verdict does
not make the task hard for a bag of words: the label is redundantly encoded in the
Rule + Scenario text as well. On the four domains where TF-IDF-ablated is ≥0.98
(cybersecurity_mitre_attack, edu_academic_integrity, edu_online_learning), the ablated task
is separable to ceiling by unigram counts alone, leaving no headroom for Probe to claim.

## Harness diagnostics

### Permutation null
Mean permutation-null AUROC across all 40 domain×variant cells: **0.4978** (target 0.5). Range 0.455–0.567.

Deviation from 0.5 is concentrated in the small-n domains, exactly as expected from
sampling noise on a small test slice (the null AUROC's standard error scales as
~1/sqrt(n_test)). Largest deviations:

| domain | variant | n_test | perm-null |
|---|---|---:|---:|
| cybersecurity_mitre_attack | ablated | 72 | 0.567 |
| edu_discrimination_us_edu_dept | with_outcome | 32 | 0.557 |
| edu_academic_integrity | ablated | 72 | 0.554 |
| edu_discrimination_us_edu_dept | ablated | 32 | 0.552 |
| cybersecurity_mitre_attack | with_outcome | 72 | 0.548 |
| finance_anti_laundering_and_terrorist | with_outcome | 200 | 0.455 |

No systematic bias: the null mean sits at 0.5 to within 0.01, so the harness is sound.

### Small-n domains

| domain | n_test | source pool C / V | capped by pool |
|---|---:|---:|---|
| finance_cross_border_payment_law | 126 | 311 / 308 | yes |
| cybersecurity_mitre_attack | 72 | 172 / 341 | yes |
| edu_academic_integrity | 72 | 173 / 241 | yes |
| edu_online_learning | 144 | 401 / 356 | yes |
| edu_discrimination_us_edu_dept | 32 | 73 / 73 | yes |

All other domains hit the n_pairs=500 cap (n_test = 200). Every slice is exactly 50/50
COMPLIES/VIOLATES by construction: sampling is paired, and each split index selects one
case from each class, so class balance is never a confound in these AUROCs.

### Split parity with the G1 activation pipeline

`results/rebuild/splits/<domain>_seed42_splits.json` dumps were present for **5/20** domains at generation time (the GPU G1 agent writes these
as it goes). For all 5 present dumps, our sampling and split **match exactly**:
all four slices (train / val_layer / val_thresh / test) are identical index-for-index, and
the per-case ablated text matches string-for-string, confirming both agents sampled the same
cases in the same order. Domains without a dump yet: finance_crypto, finance_anti_laundering_and_terrorist, finance_cross_border_payment_law, finance_eletric_momey, data_act, cybersecurity_mitre_attack, edu_academic_integrity, edu_online_learning, edu_discrimination_us_edu_dept, policy_github, policy_google, policy_openai, policy_reddit, policy_x, foundation_rights.

---

Per-case test scores, per-shuffle null AUROCs, split indices and provenance blocks are in
`results/rebuild/gauntlet/gauntlet_<domain>_seed42.json`.
