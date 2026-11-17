#!/usr/bin/env python
"""
run_example_candidate_search.py — score a modest batch of NEW quadruples
(disjoint from the already-published finance_crypto_t07 worked example and
from the 40-quadruple CoT positive-control sample) across the fast detectors
(frozen Probe, forced-choice judge, LPG-4B), to find a genuinely representative
illustrative case for a second qualitative example -- not a misleadingly
favorable one. "Representative" here means: the failure pattern matches the
aggregate story (cheap detectors near chance, not solving the interaction),
confirmed against the full 800-row numbers already in the paper, not an
outlier being passed off as typical.

Usage:
    MECHCOMPLIANCE_PHYSICAL_GPU=1 python scripts/run_example_candidate_search.py --stage probe
    MECHCOMPLIANCE_PHYSICAL_GPU=1 python scripts/run_example_candidate_search.py --stage judge
    MECHCOMPLIANCE_PHYSICAL_GPU=1 python scripts/run_example_candidate_search.py --stage lpg --ckpt_dir <path>
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np
import competitors_lib as lib

CF = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "crossed_pairs.json")
OUT = os.path.join(REPO, "results", "rebuild", "crossed_benchmark", "candidate_search_scores.json")
SEED = 999  # distinct from the seed(s) already used for the worked example / CoT sample
N_CANDIDATES = 20
CALIB_DOMAINS = ["gdpr", "hipaa", "eu_ai_act", "ccpa", "finance_crypto",
                 "cybersecurity_mitre_attack", "policy_openai", "data_act"]
EXCLUDE_QID = "finance_crypto_t07"


def pick_candidates():
    rows = json.load(open(CF))
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quadruple_id"], {})[r["cell"]] = r
    qids = sorted(q for q in by_q if q != EXCLUDE_QID)
    rng = random.Random(SEED)
    chosen = rng.sample(qids, N_CANDIDATES)
    return chosen, by_q


def cells_for(by_q, qid):
    cells = ["R0xSa", "R0xSb", "R1xSa", "R1xSb"]
    return [by_q[qid][c] for c in cells], cells


def stage_probe(chosen, by_q):
    from run_ft_multi import load_model
    from src.compliance.direction import MeanDifferenceDirection
    from src.compliance.activations import extract_activations

    def extract(txts, model, tok):
        return extract_activations(txts, model, tok, max_length=512, extraction="last_token", show_progress=False)

    model, tok = load_model("hf", "meta-llama/Llama-3.2-1B-Instruct", "?")

    c_tr, v_tr, c_th, v_th = [], [], [], []
    for dom in CALIB_DOMAINS:
        dump, _ = lib.load_split_dump(dom, SEED_CALIB := 42)
        tr_idx = dump["split_indices"]["train"]; th_idx = dump["split_indices"]["val_thresh"]
        v_tr += [dump["violates_cases"][i]["text_ablated"] for i in tr_idx]
        c_tr += [dump["complies_cases"][i]["text_ablated"] for i in tr_idx]
        v_th += [dump["violates_cases"][i]["text_ablated"] for i in th_idx]
        c_th += [dump["complies_cases"][i]["text_ablated"] for i in th_idx]
    av, ac = extract(v_tr, model, tok), extract(c_tr, model, tok)
    avt, act = extract(v_th, model, tok), extract(c_th, model, tok)
    di = MeanDifferenceDirection().fit(ac, av)
    from sklearn.metrics import roc_auc_score
    seps = []
    for L in range(av.shape[1]):
        s = np.r_[-(avt[:, L, :] @ di.direction_[L]), -(act[:, L, :] @ di.direction_[L])]
        yy = np.r_[np.ones(len(avt)), np.zeros(len(act))]
        seps.append(float(roc_auc_score(yy, s)))
    anchor = int(np.argmax(seps))
    v_th_scores = -(avt[:, anchor, :] @ di.direction_[anchor])
    c_th_scores = -(act[:, anchor, :] @ di.direction_[anchor])
    tau = 0.5 * (v_th_scores.mean() + c_th_scores.mean())

    result = {}
    for qid in chosen:
        rows_, cells = cells_for(by_q, qid)
        texts = [r["text"] for r in rows_]
        acts = extract(texts, model, tok)
        scores = -(acts[:, anchor, :] @ di.direction_[anchor])
        result[qid] = {c: {"score": float(s), "predicted_violation": bool(s >= tau),
                           "true_label": r["label"], "correct": bool((s >= tau) == bool(r["label"]))}
                      for c, s, r in zip(cells, scores, rows_)}
    return result


def stage_judge(chosen, by_q):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import run_llm_judge_8b as M
    lib.force_gpu0()
    tok = AutoTokenizer.from_pretrained(M.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        M.MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True).eval()
    ids = M.resolve_token_ids(tok)

    result = {}
    for qid in chosen:
        rows_, cells = cells_for(by_q, qid)
        texts = [r["text"] for r in rows_]
        out = M.score_texts(texts, model, tok, ids, f"candidate {qid}")
        scores = out["p_no"] - out["p_yes"]
        result[qid] = {c: {"score": float(s), "predicted_violation": bool(s >= 0.0),
                           "true_label": r["label"], "correct": bool((s >= 0.0) == bool(r["label"]))}
                      for c, s, r in zip(cells, scores, rows_)}
    del model
    torch.cuda.empty_cache()
    return result


def stage_lpg(chosen, by_q, ckpt_dir):
    import torch
    lib.force_gpu0()
    sys.path.insert(0, os.environ.get("LPG_REPO", os.path.join(REPO, "..", "scratch_caches", "lpg_repo")))
    import run_lpg as L
    LatentPolicyGuardModel, fmt = L._import_lpg()
    plugin = LatentPolicyGuardModel(L.DEFAULT_BASE, ckpt_dir=ckpt_dir, **L.LPG_HP)
    plugin.load()
    ids = L.resolve_tokens(plugin._tokenizer)

    result = {}
    for qid in chosen:
        rows_, cells = cells_for(by_q, qid)
        texts = [r["text"] for r in rows_]
        ls, lu, pu, nt = L.score_texts(plugin, ids, texts, fmt, f"candidate {qid}")
        scores = lu - ls
        result[qid] = {c: {"score": float(s), "predicted_violation": bool(s >= 0.0),
                           "true_label": r["label"], "correct": bool((s >= 0.0) == bool(r["label"]))}
                      for c, s, r in zip(cells, scores, rows_)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["probe", "judge", "lpg"])
    ap.add_argument("--ckpt_dir", default=None)
    args = ap.parse_args()

    chosen, by_q = pick_candidates()
    print(f"Candidates ({len(chosen)}): {chosen}")

    if args.stage == "probe":
        result = stage_probe(chosen, by_q)
    elif args.stage == "judge":
        result = stage_judge(chosen, by_q)
    elif args.stage == "lpg":
        result = stage_lpg(chosen, by_q, args.ckpt_dir)

    allr = json.load(open(OUT)) if os.path.exists(OUT) else {"candidates": chosen}
    allr[args.stage] = result
    allr["_updated"] = datetime.datetime.now().astimezone().isoformat()
    json.dump(allr, open(OUT, "w"), indent=2)

    for qid in chosen:
        n_correct = sum(1 for c in result[qid].values() if c["correct"])
        print(f"  {qid}: {n_correct}/4 correct")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
