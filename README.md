# Evaluating Rule Dependence in Compliance Detection Systems

Anonymous code and results for a short paper under double-blind review at
WWW'27. The paper asks whether a compliance guard's verdict actually depends
on the content of the rule it is given, rather than merely on the fact that
it was given *a* rule or can cite it. We test this with direct counterfactual
interventions on the rule (deletion, permutation, substitution, and an exact
permissive-clause swap) and a confound-free crossed benchmark (two rules
crossed with two scenarios), evaluated across a policy-conditioned guard, two
fixed-taxonomy guards, an activation probe, a training-free whitening
detector, and a judge LLM in both forced-choice and chain-of-thought modes.

This repository contains the subset of the code and result files needed to
reproduce the tables and figures in the paper. It does not include unrelated
experiments from the broader research program this work grew out of.

## What's in this repo

| Paper item | Script(s) | Result files |
|---|---|---|
| Probe definition (Eq. 1), difference-of-means direction | `src/compliance/direction.py`, `src/compliance/metrics.py`, `src/compliance/splits.py` | — |
| Rule-ablation experiment (Table/Results: mean AUROC 0.9519/0.9549/0.9548/0.9554 with correct/no-rule/shuffled/wrong-domain-rule, 0.5627 with scenario removed; significance counts 0/20, 0/20, 0/20, 19/20) | `scripts/run_rule_ablation.py` (primary model), `scripts/run_rule_ablation_multi.py` (replication across 4 further models, 3B–8B) | `results/rule_ablation/`, `results/rule_ablation_multi/` |
| Lexical-degeneracy check (Table 1: seven public compliance benchmarks, policy-blind bag-of-words floor vs. the probe) | `scripts/run_trivial_gauntlet.py` (OmniCompliance), `scripts/run_benchmark2_gauntlet.py` (DynaBench, SafePyramid), `scripts/run_benchmark3_gauntlet.py` + `scripts/benchmark3_data.py` (AIReg-Bench, CompliBench, FlexBench), `scripts/run_ifeval_probe.py` + `scripts/run_ifeval_activations.py` (IFEval) | `results/gauntlet/`, `results/benchmark2/`, `results/benchmark3/`, `results/ifeval/` |
| Crossed benchmark construction and evaluation (Figure 2, crossed-benchmark table: every detector's AUROC / quadruple exact match) | `scripts/build_crossed_benchmark.py`, `scripts/run_crossed_eval.py`, `scripts/run_crossed_cot_judge.py`, `scripts/crossed_stats.py` (Wilson CIs, BH correction) | `results/crossed_benchmark/` (`crossed_pairs.json` is the benchmark itself; `crossed_eval.json` is every detector's score) |
| Cross-architecture search (the 3-model x 4-estimator table: Llama-3.2-1B, Llama-3.1-8B, Qwen3-8B x frozen/mean-difference/logistic/MLP) | `scripts/run_crossed_eval.py` (stages `probe_results`, `probe_results_llama-3.1-8b-instruct`, `probe_results_qwen3-8b`) | `results/crossed_benchmark/crossed_eval.json` (keys `probe_results*`) |
| Whitening baseline (Rachmil et al.) | `scripts/run_rachmil_verify.py` | `results/rachmil_verify/` |
| GDPR worked example / counterfactual clause-substitution protocol on the policy-conditioned guard | `scripts/run_lpg.py`, `scripts/run_lpg_policy_control.py`, `scripts/run_lpg_removal_replication.py`, `scripts/run_worked_example_scores.py`, `scripts/run_example_candidate_search.py` | `results/competitors/lpg_4b_*.json`, `results/competitors/LPG_POLICY_CONTROL.json` |

**Important caveat on the headline citation-accuracy / verdict-flip numbers.**
The paper's lead claim — one guard cites the governing clause correctly in
90 to 95 percent of cases, yet flips its verdict in only 7.1 percent (13 of
182) of its originally-flagged violations when the clause is replaced by its
exact permissive opposite — is **not** reproduced by a single aggregate file
in this repository. The files under `results/competitors/lpg_4b_*.json` and
`scripts/run_lpg_policy_control.py` cover a related rule-ablation protocol
(correct / shuffled / wrong-domain / no-policy conditions) for the same
guard, and are included as the closest available reproduction material for
that guard and protocol, but they do not by themselves total 182 cases or
report a citation-accuracy percentage. If you are trying to verify this
specific number, treat it as not yet independently reproducible from this
repository alone.

## Setup

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

Each script under `scripts/` is runnable on its own (`python scripts/<name>.py --help`)
and writes its output JSON under `results/<experiment>/`, matching the files
already included here. Scripts that load a language model expect a single
visible CUDA GPU and will download model weights from the Hugging Face Hub
on first run.

## License

MIT. See `LICENSE`.
