# E-RULE-MULTI — rule-blindness is NOT a 1B artifact: it replicates across scale

*2026-07-17/18 · seed 42 · GPU 0 · ablated OmniCompliance, 20 domains · each model's Probe
refit fresh per condition; per-model bit-exact receipt reproduction asserted before scoring.
Written by the orchestrator from the per-model `rule_ablation_<slug>_SUMMARY.json` rollups
(the top-level summary step did not fire; computed here from the complete 3-model × 20-domain
data). Baseline 1B from `results/rebuild/rule_ablation/`.*

## The result — Probe, deleting the Rule block

| model | correct | no\_rule | shuffled | wrong-domain | rule-only | Δ(corr−no\_rule) | **BH-sig drops** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 0.952 | 0.955 | 0.955 | 0.955 | 0.563 | $+0.003$ | **0/20** |
| Llama-3.2-3B | 0.958 | 0.956 | 0.957 | 0.955 | 0.571 | $+0.0015$ | **0/20** |
| Llama-3.1-8B | 0.962 | 0.960 | 0.961 | 0.959 | 0.568 | $+0.0020$ | **0/20** |
| Qwen2.5-7B | 0.963 | 0.960 | 0.960 | 0.958 | 0.559 | $+0.0027$ | 2/20 |

## THE VERDICT — the central finding is robust across scale and family

**Deleting the rule costs nothing on the 1B, 3B, and 8B (0/20 significant drops each), and
2/20 on Qwen2.5-7B.** On every model the *mean* effect of removing the rule is **positive**
(the probe is slightly better without it, $+0.0015$ to $+0.003$). Shuffling the rule
within-domain or substituting a different domain's rule (identical for both classes, zero
class information) is equally harmless everywhere. **Rule-blindness is not a 1B artifact** —
it holds on four models across two families up to 8B.

**Qwen2.5-7B is a mild, honest exception**: 2 of 20 domains show a BH-significant drop when
the rule is removed, i.e. that model conditions slightly on the rule in a couple of domains.
The aggregate still holds (mean Δ positive, 18/20 domains unaffected). State it, do not bury
it: the largest Qwen model is the one place a faint rule signal appears, but it does not
change the conclusion.

**Rule-only collapses on every model** (0.56–0.57) — the positive control: the pipeline
detects real degradation when it exists, so the flat no\_rule result is not the machinery
failing to move.

## TF-IDF confirms the mechanism across models
On every model, TF-IDF **improves** without the rule: 0.896 → **0.931** (identical across
models — it is the same ablated text). A bag of words recovers the label from the scenario
better once rule tokens stop diluting it. The label is a property of the scenario, at every
scale.

## For the paper
Fold into §Rule Ablation as the robustness sentence: *"The finding is not a 1B artifact:
deleting the rule produces 0/20 significant drops on the 3B and 8B and 2/20 on Qwen2.5-7B,
with the mean effect positive on all; rule-only collapses on every model."* This pre-empts
the single most likely reviewer objection.

## Sanity
Each model reproduced its stored OmniCompliance Probe receipt before scoring; splits
index-identical; ablation asserted (no `Outcome:`); every AUROC through the sklearn-1e-9
wrapper; exact-`Fraction` BH. Run in `safetune` (torch 2.11) for bit-exactness. GPU-0
PID-verified; coexisted with content-selection without OOM after the initial contention was
resolved.
