# E-B3 — AIReg-Bench, CompliBench, FlexBench (+COMPASS): the monoculture grows to 4/7, and FlexBench replaces AIReg as the second clean testbed

*2026-07-17 · seed 42 · Llama-3.2-1B-Instruct · GPU 0 · written by the orchestrator from
the agent's verbatim report. Receipts: `{aireg,complibench,flexbench}_{gauntlet,directions}.json`,
`compass_existence.json`, `BENCHMARK3_SUMMARY.json`. Scripts: `benchmark3_data.py`,
`run_benchmark3_{gauntlet,directions}.py`, `summarize_benchmark3.py`.*

## Acquisition

| benchmark | source @ revision | license | n (unit) | base rate | grouping |
|---|---|---|---|---|---|
| AIReg-Bench | HF `camlsys/AIReg-Bench` @ c656725c + GitHub `camlsys/aireg-bench` @ 846d5cc0 | CC-BY-4.0 | 300 docs; **120 expert-annotated** | 36.7% compliant (expert) | (use, system) — system outlines shared across articles |
| CompliBench | **NOT on HF** (0 repos, receipted). GitHub `UCSB-NLP-Chang/CompliBench` @ 5935430 | Apache-2.0 | 2,713 turns / 306 conversations | 30.6% violating | conversation |
| FlexBench | HF `Tommy-DING/FlexBench` @ 8f73c40b (arXiv 2602.23636) | Apache-2.0 | prompt 2,311 / response 2,278 (deduped) | 25/37.5/50% by strictness | unique input text |

Provenance surprise: the official AIReg HF repo does NOT contain the 300 doc .txt files
its README describes — texts came from the official GitHub (recorded in
`benchmark3_data.py::REVISIONS`).

## The verdicts

1. **AIReg-Bench is DEGENERATE — including the human-expert track.** TF-IDF doc-only
   0.957 [CI .907–1.00]; grouped 5-fold CV **0.975 ± 0.021** — *higher* than ungrouped
   (0.935): the shortcut is stylistic register, not same-system leakage. Probe 0.943
   (grouped 0.953) sits AT the floor, refusal 0.746. **AIReg is not the second clean
   testbed** — it is the cleanest confirmation of outcome-first synthesis (expert/intended
   agreement 0.95). Small-n caveat: expert n_test=30; point-estimate + CI only.
2. **CompliBench: borderline-not degenerate (0.879), Probe INVERTED (0.700)** —
   DynaBench-pattern. **New leak found**: per-turn `guideline_text` on violating turns is
   the modified violation-inducing directive (823/830 violating vs 2/1,883 compliant carry
   one) → TF-IDF 0.990 is a **construction-metadata oracle**, absent from judge-facing
   input; fair reconstruction (original guideline substituted, 825 turns): fair-only
   0.576, fair+turn 0.837 — the honest rule-aware floors.
3. **FlexBench is NOT degenerate (0.68–0.80)** — the second clean testbed alongside
   IFEval. Response track: weak Omni-pattern at strict/moderate (Probe .832/.824 vs floor
   .802/.802, within CI overlap), inverts at loose. Prompt track: three-way tie — and the
   **zero-supervision refusal direction (.730/.730/.732) matches/beats supervised TF-IDF**
   at moderate/loose. **Strictness axis finding**: everything degrades monotonically as
   the policy loosens (Probe .832→.729); separating low-severity from benign is where
   activations lose to lexical content.
4. **COMPASS EXISTS — the audit's NOT-FOUND is overturned.** arXiv **2601.01836**, Choi
   et al. (AIM Intelligence/BMW/Yonsei/POSTECH/SNU), ACL 2026 per GitHub README; code
   `github.com/AIM-Intelligence/COMPASS`; HF `AIM-Intelligence/COMPASS-Policy-Alignment-
   Testbed-Dataset`; 5,920 queries, 8 verticals. The audit searched the overloaded bare
   name. Caveat: labels live in model responses (refusal-behavior eval) — structurally
   IFEval-like, untouched by this gauntlet. **Stays in FINAL_PLAN.**

## The running tally (7 benchmarks)

| benchmark | best fair lexical floor | verdict |
|---|---:|---|
| OmniCompliance | 0.896 | degenerate |
| DynaBench | 0.982 | degenerate |
| SafePyramid | 0.90+ (rule-only) | degenerate |
| **AIReg-Bench** | **0.957 / 0.975 grouped** | **degenerate (expert track!)** |
| CompliBench | 0.879 (fair 0.837) | borderline |
| FlexBench | 0.68–0.80 | **clean** |
| IFEval | 0.574 | **clean** |

**Paper phrasing: four of seven public compliance benchmarks are lexically degenerate and
a fifth is borderline; on the two where measurement is possible, the probe ties or loses
to trivial floors except a weak, CI-overlapping lead on FlexBench-response at strict.**

## Sanity checks

Orientation guard 1e-9, zero failures; refusal identity exact-0.0 in all three GPU runs;
perm nulls ≈0.5 everywhere; splits identical between gauntlet and directions; exact-
Fraction BH (CompliBench 3/3 significant at p=1/2000; AIReg per-article n=6 = power
statement, pooled number is the evidence). Layouts from measured token distributions
(IFEval lesson applied): AIReg rule-last @2048 (99.7% of docs > 512 tok), CompliBench
rule-last @1024, FlexBench raw @512+1024; gate-matched 512 reported as secondary
everywhere.
