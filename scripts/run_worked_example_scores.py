#!/usr/bin/env python
"""
run_worked_example_scores.py — real, receipted per-cell scores for the
finance_crypto_t07 worked example (Kestrel Bank, dormant-account thresholds)
already narrated in Supplementary~\\ref{app:crossedexample}, for a compact
main-text qualitative figure. Scores the SAME 4 rows with the SAME pipelines
already used for the full 800-row crossed-benchmark run (frozen Probe,
forced-choice judge, LPG-4B, chain-of-thought judge) -- no new method, just
a targeted re-score of one quadruple so its real numbers can be plotted.

Usage:
    MECHCOMPLIANCE_PHYSICAL_GPU=1 python scripts/run_worked_example_scores.py
"""
from __future__ import annotations
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
OUT = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "worked_example_scores.json")
SEED = 42
CALIB_DOMAINS = ["gdpr", "hipaa", "eu_ai_act", "ccpa", "finance_crypto",
                 "cybersecurity_mitre_attack", "policy_openai", "data_act"]
QID = "finance_crypto_t07"
CELLS = ["R0xSa", "R0xSb", "R1xSa", "R1xSb"]


def load_quadruple():
    rows = json.load(open(CF))
    by_cell = {r["cell"]: r for r in rows if r["quadruple_id"] == QID}
    return [by_cell[c] for c in CELLS]


def stage_probe(rows):
    from run_ft_multi import load_model
    from src.compliance.direction import MeanDifferenceDirection
    from src.compliance.activations import extract_activations

    def extract(txts, model, tok):
        return extract_activations(txts, model, tok, max_length=512, extraction="last_token", show_progress=False)

    model, tok = load_model("hf", "meta-llama/Llama-3.2-1B-Instruct", "?")
    texts = [r["text"] for r in rows]
    acts_full = extract(texts, model, tok)

    c_tr, v_tr, c_th, v_th = [], [], [], []
    for dom in CALIB_DOMAINS:
        dump, _ = lib.load_split_dump(dom, SEED)
        tr_idx = dump["split_indices"]["train"]
        th_idx = dump["split_indices"]["val_thresh"]
        v_tr += [dump["violates_cases"][i]["text_ablated"] for i in tr_idx]
        c_tr += [dump["complies_cases"][i]["text_ablated"] for i in tr_idx]
        v_th += [dump["violates_cases"][i]["text_ablated"] for i in th_idx]
        c_th += [dump["complies_cases"][i]["text_ablated"] for i in th_idx]
    av, ac = extract(v_tr, model, tok), extract(c_tr, model, tok)
    avt, act = extract(v_th, model, tok), extract(c_th, model, tok)

    di = MeanDifferenceDirection().fit(ac, av)
    seps = []
    for L in range(av.shape[1]):
        s = np.r_[-(avt[:, L, :] @ di.direction_[L]), -(act[:, L, :] @ di.direction_[L])]
        yy = np.r_[np.ones(len(avt)), np.zeros(len(act))]
        from sklearn.metrics import roc_auc_score
        seps.append(float(roc_auc_score(yy, s)))
    anchor = int(np.argmax(seps))
    v_th_scores = -(avt[:, anchor, :] @ di.direction_[anchor])
    c_th_scores = -(act[:, anchor, :] @ di.direction_[anchor])
    tau = 0.5 * (v_th_scores.mean() + c_th_scores.mean())
    scores = -(acts_full[:, anchor, :] @ di.direction_[anchor])
    return {c: {"score": float(s), "threshold": float(tau), "predicted_violation": bool(s >= tau)}
            for c, s in zip(CELLS, scores)}


def stage_judge(rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import run_llm_judge_8b as M
    lib.force_gpu0()
    tok = AutoTokenizer.from_pretrained(M.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        M.MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True).eval()
    ids = M.resolve_token_ids(tok)
    texts = [r["text"] for r in rows]
    out = M.score_texts(texts, model, tok, ids, "worked-example judge")
    scores = out["p_no"] - out["p_yes"]
    del model
    torch.cuda.empty_cache()
    return {c: {"score": float(s), "threshold": 0.0, "predicted_violation": bool(s >= 0.0)}
            for c, s in zip(CELLS, scores)}


def stage_cot(rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lib.force_gpu0()
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
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True).eval()

    result = {}
    for r, c in zip(rows, CELLS):
        prompt = PROMPT_TMPL.format(rule=r["rule_text"], scenario=r["scenario_text"])
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        input_ids = (enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        m = re.findall(r"Verdict:\s*(COMPLIANT|VIOLATION)", gen, re.IGNORECASE)
        verdict = m[-1].upper() if m else None
        result[c] = {"predicted_violation": verdict == "VIOLATION", "reasoning": gen, "raw_verdict": verdict}
    del model
    torch.cuda.empty_cache()
    return result


def stage_lpg(rows, ckpt_dir):
    import torch
    lib.force_gpu0()
    sys.path.insert(0, os.environ.get("LPG_REPO", os.path.join(REPO, "..", "scratch_caches", "lpg_repo")))
    import run_lpg as L
    LatentPolicyGuardModel, fmt = L._import_lpg()
    plugin = LatentPolicyGuardModel(L.DEFAULT_BASE, ckpt_dir=ckpt_dir, **L.LPG_HP)
    plugin.load()
    ids = L.resolve_tokens(plugin._tokenizer)
    texts = [r["text"] for r in rows]
    ls, lu, pu, nt = L.score_texts(plugin, ids, texts, fmt, "worked-example LPG")
    scores = lu - ls
    return {c: {"score": float(s), "threshold": 0.0, "predicted_violation": bool(s >= 0.0)}
            for c, s in zip(CELLS, scores)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["probe", "judge", "cot", "lpg"])
    ap.add_argument("--ckpt_dir", default=None)
    args = ap.parse_args()

    rows = load_quadruple()
    assert len(rows) == 4, f"expected 4 rows for {QID}, got {len(rows)}"
    truth = {c: r["label"] for c, r in zip(CELLS, rows)}

    if args.stage == "probe":
        result = stage_probe(rows)
    elif args.stage == "judge":
        result = stage_judge(rows)
    elif args.stage == "cot":
        result = stage_cot(rows)
    elif args.stage == "lpg":
        result = stage_lpg(rows, args.ckpt_dir)

    for c in CELLS:
        result[c]["true_label"] = truth[c]
        result[c]["correct"] = result[c]["predicted_violation"] == bool(truth[c])

    allr = json.load(open(OUT)) if os.path.exists(OUT) else {"quadruple_id": QID}
    allr[args.stage] = result
    allr["_updated"] = datetime.datetime.now().astimezone().isoformat()
    json.dump(allr, open(OUT, "w"), indent=2)
    print(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
