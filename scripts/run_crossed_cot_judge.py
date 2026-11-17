#!/usr/bin/env python
"""
run_crossed_cot_judge.py — positive-control check for the crossed rule x scenario
benchmark (results/rebuild/crossed_benchmark/crossed_pairs.json).

The forced-single-token reasoning judge (run_crossed_eval.py --stage judge,
Llama-3.1-8B-Instruct, first-token Yes/No logit) scored AUROC 0.507 -- chance --
on the crossed benchmark, same as every other detector tested. Before reporting
"nothing solves this, not even an explicit rule-given judge" as a finding, we
need a positive control: is the benchmark solvable in principle by a reasoner
that is actually ALLOWED to reason (generate a chain of thought identifying the
relevant numeric threshold and value and comparing them) before answering, or
does even explicit step-by-step reasoning fail? This distinguishes "the
benchmark requires numeric comparison no cheap detector can do" from "the
benchmark is somehow unsolvable/degenerate."

Runs on a fixed random SUBSAMPLE of quadruples (default 40 quadruples = 160
rows) for speed, since generation (vs. a single forced-token logit) is slow.
Full generation, output parsed for a final COMPLIANT/VIOLATION verdict line
(not a forced first token) -- a much more permissive protocol than the
zero-shot forced-choice judge.

Usage:
    MECHCOMPLIANCE_PHYSICAL_GPU=1 python scripts/run_crossed_cot_judge.py --n_quadruples 40
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np
import competitors_lib as lib

CF = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "crossed_pairs.json")
OUT = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "cot_judge_positive_control.json")
SEED = 42
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_TMPL = """You are a compliance officer. Read the RULE and the SCENARIO below. \
The rule states a numeric condition. Identify the specific number in the scenario, \
identify the threshold in the rule that applies to this scenario's category, and \
compare them explicitly step by step. Then give your final answer.

RULE: {rule}

SCENARIO: {scenario}

Think step by step: (1) what category does the scenario fall under, (2) what is \
the numeric threshold the rule sets for that category, (3) what is the actual \
numeric value in the scenario, (4) does the value exceed the threshold. Then end \
your response with exactly one line in the form:
Verdict: COMPLIANT
or
Verdict: VIOLATION
"""


def parse_verdict(text):
    m = re.findall(r"Verdict:\s*(COMPLIANT|VIOLATION)", text, re.IGNORECASE)
    if not m:
        return None
    return 1 if m[-1].upper() == "VIOLATION" else 0


def flip_metrics(rows, preds):
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quadruple_id"], {})[r["cell"]] = r["label"]
    rule_c = rule_t = scen_c = scen_t = exact = nq = unparsed = 0
    for qid, cells in by_q.items():
        p = preds.get(qid)
        if p is None or set(p.keys()) != {"R0xSa", "R0xSb", "R1xSa", "R1xSb"}:
            continue
        if any(v is None for v in p.values()):
            unparsed += 1
            continue
        nq += 1
        for s in ("Sa", "Sb"):
            rule_t += 1
            rule_c += int(p[f"R0x{s}"] == cells[f"R0x{s}"] and p[f"R1x{s}"] == cells[f"R1x{s}"])
        for r_ in ("R0", "R1"):
            scen_t += 1
            scen_c += int(p[f"{r_}xSa"] == cells[f"{r_}xSa"] and p[f"{r_}xSb"] == cells[f"{r_}xSb"])
        if all(p[c] == cells[c] for c in cells):
            exact += 1
    return {
        "n_quadruples_fully_parsed": nq, "n_quadruples_with_unparsed_cell": unparsed,
        "rule_flip_accuracy": rule_c / rule_t if rule_t else float("nan"),
        "scenario_flip_accuracy": scen_c / scen_t if scen_t else float("nan"),
        "quadruple_exact_match": exact / nq if nq else float("nan"),
        "quadruple_exact_match_chance": 0.0625,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_quadruples", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    args = ap.parse_args()

    lib.force_gpu0()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = json.load(open(CF))
    qids = sorted(set(r["quadruple_id"] for r in rows))
    rng = np.random.default_rng(SEED)
    chosen = set(rng.choice(qids, size=min(args.n_quadruples, len(qids)), replace=False))
    sub = [r for r in rows if r["quadruple_id"] in chosen]
    print(f"Scoring {len(sub)} rows / {len(chosen)} quadruples (CoT-allowed judge)")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True).eval()

    preds = {}
    n_unparsed_total = 0
    for i, r in enumerate(sub):
        prompt = PROMPT_TMPL.format(rule=r["rule_text"], scenario=r["scenario_text"])
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        input_ids = (enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        v = parse_verdict(gen)
        if v is None:
            n_unparsed_total += 1
        preds.setdefault(r["quadruple_id"], {})[r["cell"]] = v
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sub)} done, {n_unparsed_total} unparsed so far")

    y = np.array([r["label"] for r in sub])
    scores = np.array([preds[r["quadruple_id"]][r["cell"]] if preds[r["quadruple_id"]][r["cell"]] is not None
                       else 0.5 for r in sub])
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan")

    m = flip_metrics(sub, preds)
    m["auroc_binary_verdict"] = auroc
    m["n_rows_scored"] = len(sub)
    m["n_quadruples_attempted"] = len(chosen)
    m["n_unparsed_cells_total"] = n_unparsed_total
    m["model"] = MODEL_NAME
    m["protocol"] = "chain-of-thought generation (max_new_tokens={}), final Verdict: line parsed".format(args.max_new_tokens)
    m["timestamp"] = datetime.datetime.now().astimezone().isoformat()
    m["git_commit"] = lib.git_commit()

    print(json.dumps(m, indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(m, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
