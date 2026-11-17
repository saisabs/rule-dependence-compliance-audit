# E-RACHMIL-VERIFY — settling the App. E related-work claim

**Question.** The paper (App. E / Related Work) claims a *policy-blind supervised
bag-of-words* matches the reported SOTA of **Rachmil et al.**, *"Training-Free
Policy Violation Detection via Activation-Space Whitening in LLMs"*
(arXiv:2512.03994), on **their own benchmark** — our TF-IDF F1 ~=0.898 vs their
0.860 — but carries a *"split correspondence unverified"* hedge. A reviewer
flagged that hedge as fatal to using it as a headline. This task resolves the
hedge on the **same split** or demotes the claim.

**Verdict: DEMOTE (reconstruction-only).** On the benchmark's *own published
train/test split* — the identical protocol Rachmil evaluate on — a policy-blind
bag of words reaches only **F1 ~= 0.51-0.61** (AUROC 0.53), far below their SOTA
of 0.860. The ~0.898 reproduces **only** under an in-distribution *within-test*
split (train on part of the 543 test rows, test on the rest), which is **not**
Rachmil's protocol. The "matches their SOTA" headline does not survive.

---

## What Rachmil actually report on (established, not assumed)

Sources: arXiv:2512.03994 (v3 HTML) + their code
`github.com/FujitsuResearch/LLM-policy-violation-detection`
(`whitening_eval_dynabench.py`, `utils.py`).

| Item | Value |
|---|---|
| Benchmark | `montehoover/DynaBench` (= `tomg-group-umd/DynaBench`) |
| Test set | DynaBench config **`test` split = 543 rows** (all of them) |
| Metric | `utils.compute_f1` -> `sklearn.f1_score` default => **binary F1, pos_label=1**, where `numeric_label`: FAIL(violation)->1, PASS(compliant)->0. So SOTA = **binary F1 on the VIOLATION class**. |
| Headline SOTA | **F1 = 0.860** (best model, Table 1) |
| Their training data | **None on DynaBench.** Method is training-free: whitening + threshold calibrated on their *own external synthetic contrastive data* (GPT-5.1, ~100/category, 80/20). Never touches the 543 test rows, never uses DynaBenchTrain. |
| Baselines they beat | fine-tuned baselines whose numbers are "taken from the DynaBench benchmark paper" — those are trained on **DynaBenchTrain**. |

**The benchmark ships a canonical training split:** config **`DynaBenchTrain`
(61,496 rows)**, same `policy / transcript / label(PASS,FAIL)` schema. This is
the split supervised guardrails are meant to train on, and the split the
DynaBench-paper fine-tuned baselines use.

So the principled "supervised baseline on their benchmark and split" is:
**fit on DynaBenchTrain -> evaluate binary-F1(FAIL) on the same 543 test rows.**

## Same-split verification (leak-checked)

- Fit **TF-IDF unigram + LogisticRegression** on DynaBenchTrain (61,496 rows).
- Evaluate on the **identical 543-row DynaBench test split** (base_id set is
  byte-identical to the rows the paper's own gauntlet used, and to Rachmil's).
- Metric identical to theirs: binary F1, FAIL = positive class.
- **No leak, asserted:** train-cap-test `base_id` overlap = 0; `transcript`
  overlap = 0; test n = 543 (276 PASS / 267 FAIL, 49.2% violation base rate).

### Result — TF-IDF on the OWN published split (train=DynaBenchTrain -> eval=543)

| Feature (bag of words) | F1 @0.5 | F1 @calibrated th | AUROC |
|---|---|---|---|
| **transcript_only** (policy-blind) | **0.511** | **0.613** | 0.528 |
| policy_only | 0.548 | 0.557 | 0.496 |
| policy + transcript | 0.613 | 0.618 | 0.573 |

All variants sit near chance (AUROC ~= 0.50-0.57) and **none approach 0.860.**
Note: this uses the *full* 61k training split — ~50x more supervised data than
Rachmil's ~1.2k-example calibration set — and still fails to transfer. Not a
data-quantity problem; the lexical shortcut does not transfer across the
train->test distribution gap.

### Where the 0.898 came from — within-TEST reconstruction

Training a bag of words on **part of the 543 test rows** and evaluating on the
rest (in-distribution) reproduces the paper's number:

| Feature | F1 (217/110 holdout) | F1 (5-fold OOF over 543) |
|---|---|---|
| transcript_only | **0.870** | **0.864** |
| policy + transcript | 0.857 | 0.857 |

(Consistent with the paper's own App. E: holdout F1 0.898, 5-fold CV mean 0.875.)

This confirms the 543-row **test set carries a strong within-distribution
lexical tell** — but that tell is a property of how the 543 test rows were
constructed, not something a classifier can learn from the benchmark's training
split and carry over. Rachmil's method never receives this in-distribution
advantage; comparing an in-distribution within-test classifier against their
external-calibration number is apples-to-oranges.

## Bottom line

| Comparison | F1 | vs SOTA 0.860 |
|---|---|---|
| Rachmil et al. (reported SOTA) | 0.860 | — |
| TF-IDF, **same published split** (train=DynaBenchTrain -> 543 test) | **0.51-0.61** | FAIL far below |
| TF-IDF, within-test in-distribution split (paper's App. E protocol) | 0.87-0.90 | matches, but wrong protocol |

- **Obtained the exact benchmark + split?** Yes. `montehoover/DynaBench`,
  DynaBenchTrain (61,496) train / DynaBench (543) test — the benchmark's own
  published split, downloaded to the spacious `hf_cache_ext` mount; test rows
  byte-identical to Rachmil's eval set; leak-checked disjoint.
- **Our TF-IDF vs their SOTA on that split:** F1 **0.51-0.61** vs **0.860**.
- **One-line verdict:** **DEMOTE** — the "matches their SOTA on their own
  benchmark" headline does **not** survive the same split; the ~0.898 is a
  within-test in-distribution artifact. Reframe as a statement about the DynaBench
  *test set's* internal lexical shortcut (within-test CV F1 ~= 0.87-0.90, AUROC of
  the transcript-only tell ~= 0.98), **not** a head-to-head match of Rachmil's
  training-free SOTA.

## Recommended edit to the paper

Replace *"a supervised bag of words matches the reported state of the art of
Rachmil et al. (F1 0.898 vs their 0.860; split correspondence unverified)"* with
a within-test-only framing, e.g.: *"within DynaBench's 543-row test set a
policy-blind TF-IDF is highly separable (in-distribution CV F1 ~= 0.88), exposing
a construction-specific lexical shortcut; this does not transfer from the
benchmark's own training split (F1 ~= 0.55), so it is a statement about the test
set's construction, not a head-to-head match of Rachmil et al.'s training-free
result."* Do **not** present 0.898-vs-0.860 as a same-split head-to-head.

---

*Reproduce:* `HF_HOME=<ext> python scripts/run_rachmil_verify.py`
(env `safetune`; seed 42; deterministic). Full numbers in `rachmil_verify.json`.
