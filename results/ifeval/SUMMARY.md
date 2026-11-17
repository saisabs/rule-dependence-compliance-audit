# IFEval — the non-degenerate benchmark. Probe does not clear its floor.

*2026-07-17 · seed 42 · Llama-3.2-1B-Instruct · GPU 0 · official `instructions_registry`
verifier. Written by the orchestrator from the agent's verbatim report. Receipts:
`generations_seed42.json`, `labels_seed42.json`, `BASE_RATES.json`, `acts_seed42.npz`,
`IFEVAL_SUMMARY{,_acts2048,_loose}.json`, `category_*.json`, `splits/*.json`. Scripts:
`run_ifeval_{generate,verify,activations,probe}.py`.*

## THE VERDICT: the prediction held — **and it cost us.**

**Lexical baselines ARE near chance on IFEval. So is Probe.** This is the brief's named
**failure mode #1**, and it is reported straight.

The result is a **three-way decomposition**, not one number:

| bar | mean AUROC (7 kept categories) | reading |
|---|---:|---|
| **Content** (TF-IDF) | **0.574** (vs **0.896** on OmniCompliance) | **Prediction confirmed — IFEval is NOT lexically degenerate.** |
| **Form** (`format_feats`, kwargs-free) | **0.738** (saturates to 1.000 on some) | Adherence *is* recoverable by counting surface form. |
| **Probe** | **0.628** @512 / **0.699** @2048 | Above chance, **below the form floor**, never significantly above it. |

> **Probe beats TF-IDF on 0/7 categories and the honest floor on 0/7** (BH-corrected), in
> **every** config (512, 2048, strict, loose).

**We finally have the non-degenerate compliance benchmark our thesis said didn't exist — and
on it, our method does not clear the trivial floor.**

### The one significant positive, and why it is not a win

Pooled @2048 vs TF-IDF: **+0.151, CI [+0.007, +0.295], p=0.044**. The agent does not count
it and neither should we: it is the **non-mandated** config (the gate-matched 512 pooled is
null, **p=0.243**); it is **uncorrected across ~4 configs**; its lower bound is **+0.007**.

Stated precisely: **Probe's only significant edge is over the *content* floor, only in the
non-primary 2048 config, and it does not clear the *form* floor in any config** (@2048 vs
`format_feats`: +0.049, p=0.471).

**Per-category q=1.000 everywhere is a POWER statement (n_test 12–34), not proof of
absence.** The load-bearing evidence is the **pooled prompt-level test (n_test=109)** — the
only adequately powered one — and there the answer is still null against form.

## Base rates — healthy, not degenerate

Prompt-level strict accuracy **0.519** (loose 0.555); instance-level compliance **0.640**
(534/834). Near 50/50 — the benchmark has real signal to detect.

**Dropped 2 of 9 categories (no silent truncation):** `combination` (test minority 2 < 3 —
the 1B cannot repeat prompts verbatim, 9.8%, or emit `******`, 25%) and `language`
(val_layer single-class; n=31, only 6 violations).

| category | n | comp | viol | rate |
|---|---:|---:|---:|---:|
| keywords | 163 | 112 | 51 | 0.687 |
| detectable_format | 157 | 112 | 45 | 0.713 |
| length_constraints | 143 | 85 | 58 | 0.594 |
| change_case | 89 | 60 | 29 | 0.674 |
| startend | 67 | 38 | 29 | 0.567 |
| punctuation | 66 | 54 | 12 | 0.818 |
| *combination* | *65* | *10* | *55* | *0.154* → **dropped** |
| detectable_content | 53 | 38 | 15 | 0.717 |
| *language* | *31* | *25* | *6* | *0.806* → **dropped** |

## The table (primary: strict, acts512 = gate-matched)

| category | n | n_test(v/c) | Probe512 | Probe2048 | tfidf | len | kw | **fmt_feats** | refus | y/n | *oracle* | best honest floor | margin | 95% CI | BH q |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| punctuation | 66 | 14(4/10) | 0.925 | 0.900 | 0.950 | 0.900 | 0.450 | **1.000** | 0.525 | 0.575 | *1.000* | format_feats=1.000 | −0.075 | [−0.227,+0.000] | 0.816 |
| startend | 67 | 14(6/8) | 0.875 | 0.917 | 0.583 | 0.688 | 0.417 | **0.958** | 0.521 | 0.375 | *1.000* | format_feats=0.958 | −0.083 | [−0.333,+0.125] | 0.720 |
| change_case | 89 | 19(7/12) | 0.560 | 0.524 | 0.393 | 0.762 | 0.458 | 0.357 | 0.512 | 0.619 | *0.893* | length_char=0.762 | −0.202 | [−0.440,+0.048] | 0.324 |
| length_constraints | 143 | 28(12/16) | 0.552 | 0.724 | 0.536 | 0.464 | 0.401 | 0.583 | 0.583 | 0.516 | *0.714* | format_feats=0.583 | −0.031 | [−0.333,+0.281] | 0.869 |
| keywords | 163 | 34(10/24) | 0.535 | 0.535 | 0.465 | 0.377 | 0.500 | **0.748** | 0.356 | 0.504 | *0.821* | format_feats=0.748 | −0.213 | [−0.475,+0.062] | 0.324 |
| detectable_format | 157 | 32(12/20) | 0.508 | 0.729 | 0.525 | 0.392 | 0.575 | 0.583 | 0.646 | 0.438 | *0.733* | refusal_dir=0.646 | −0.138 | [−0.500,+0.250] | 0.720 |
| detectable_content | 53 | 12(4/8) | 0.438 | 0.562 | 0.562 | 0.594 | 0.375 | **0.938** | 0.531 | 0.531 | *1.000* | format_feats=0.938 | −0.500 | [−0.875,−0.062] | 0.231 |

**Pooled prompt-level (n=541, n_test=109 — the powered test):** Probe 0.577@512 / 0.638@2048;
tfidf 0.488; format_feats 0.589. vs tfidf: @512 **+0.089 [−0.061,+0.240] p=0.243**; @2048
**+0.151 [+0.007,+0.295] p=0.044**. vs format_feats @2048: **+0.049 [−0.072,+0.175]
p=0.471**. **Loose:** same picture (0/6; pooled +0.048, p=0.562).

## Building the right floor — TF-IDF was NOT the right bar

Exactly as anticipated. Two separate constructions:

- **`format_feats` — the honest bar.** A **kwargs-free** logreg over comma/word/bullet/
  sentence counts, uppercase ratio, placeholders, etc. **This is what Probe must beat.** It
  saturates: punctuation 1.000, startend 0.958, detectable_content 0.938.
- **`format_oracle` — reported but EXCLUDED from both bars.** A kwargs-aware margin to the
  constraint boundary. It is **tautological** — it re-derives the arithmetic the verifier
  thresholds. Mean 0.880, hitting 1.000 on three categories. *Where it saturates, conditional
  MI beyond form is ~0 **by construction**; Probe cannot win there and that is not a probe
  failure.* Good judgment by the agent — a lesser run would have used this as the floor and
  declared Probe dead for the wrong reason.

**Ceiling cells reported as one-sided 95% Clopper–Pearson bounds, never 1.0:** punctuation
format_feats/oracle ≤ **0.9278** (n_pairs=40); startend oracle ≤ **0.9395** (48);
detectable_content oracle ≤ **0.9106** (32). **No Probe cell hit ceiling.**

## ⚠ `max_length=512` is WRONG for IFEval — and it handicapped Probe

**33.3% of texts (180/541) exceed 512 tokens**, so the "last token" lands **mid-response,
before the format evidence appears**. At 2048 Probe *rises*: detectable_format 0.508→**0.729**,
length_constraints 0.552→**0.724**, pooled 0.577→**0.638**.

**The confound ran opposite to our expectation** — we imposed 512 for gate-matching, and it
worked *against* our own method. Both are reported; **the null-vs-form conclusion holds in
both.** (This is a fourth instance of the repo's `max_length` chaos causing real damage.)

## Two named deviations (declared, not silent — and both correct)

1. **Bootstrap:** IFEval **has no pairs**. Used a two-sample **stratified** bootstrap
   (resample within class, feeding the *same* resampled items to Probe and floor, preserving
   their correlation) instead of the mandated paired bootstrap.
2. **Split:** applied `three_way_split` to **unique prompt keys**, not instances. **33
   prompts carry 2–3 same-category instructions sharing one activation vector**; an
   instance-level split would have put *identical activations* in train and test — real
   leakage. The agent caught this unprompted.

## Sanity checks — all pass

- **Orientation guard:** every AUROC through a wrapper asserting agreement with
  `sklearn.roc_auc_score` to 1e-9. Zero failures. One convention throughout (higher =
  VIOLATION; Probe fit higher=compliant, then negated).
- **Permutation null** (labels shuffled, **Probe refit end-to-end**, 100×/category):
  **0.483–0.523**, mean **0.497**. TF-IDF null 0.469–0.557. Lands on 0.5.
- **Refusal direction: bit-exact vs the receipt** — `max|norm − receipt| = 0.000e+00` on both
  raw and chat variants. A genuine reuse, not a refit. *It also proves activation extraction
  is bit-identical across the transformers-version gap* (**no env still has the gate's
  4.56.2**; this ran 5.14.1).
- **Split identity:** re-derived `three_way_split(n,42)` matches every dump; slices asserted
  disjoint; fractions 40/20/20/20.
- **Verifier:** 12/12 hand-constructed spot-checks pass. The script **dies loudly** if the
  official verifier cannot be imported — **nothing is hand-rolled**.

## Generation quality — no degeneracy

**No greedy collapse.** 0 empty responses; mean distinct-token ratio 0.491; median 305 new
tokens. The 62 "low-diversity" and 17 "repeated-ngram" flags are **genre artifacts, not
decoding failure** — *the agent read them*: song lyrics with repeating choruses,
Shakespearean poems, resumes full of `[placeholders]` **the prompt asked for**. Distinct from
the prior RFT verdict-prior collapse. `cap_hit` 43/541 (7.9%) at `max_new_tokens=2048`,
sized generously because IFEval contains an "at least 1200 words" constraint — **a tight cap
would have manufactured violations by truncation.**

## What this means for the paper

**Probe is not uniformly dead here:** punctuation 0.925, startend 0.875 — it genuinely reads
comma-presence and quote-wrapping out of activations. **It is just that counting commas does
it better.**

**The honest framing:** *"Probe recovers some surface-form signal, but no better than trivially
counting that form, and its edge over lexical content is marginal and config-dependent."*

**The strongest caveat in Probe's favour is power** — only the pooled test is adequately
powered, and there the answer is still null against form.

**Combined with the gate flip (11/20, Option B), the paper's spine is now:**
1. On the field's benchmark, measurement is impossible — it is degenerate (TF-IDF 0.896).
2. On a benchmark where measurement *is* possible (IFEval, TF-IDF 0.574), **the probe does
   not beat trivial form-counting.**

That is a coherent, pre-registered negative result — and a stronger paper than the one that
passed.
