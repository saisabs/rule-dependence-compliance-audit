"""
run_ifeval_activations.py — Benchmark expansion, stage 3a: GPU features.

Produces everything that needs a GPU, so the analysis (stage 3b,
run_ifeval_probe.py) is pure CPU and cheaply rerunnable:

  1. residual-stream activations of the (prompt, response) conversation,
     last-token, max_length=512  — MATCHING THE GATE. The repo has three
     different max_length values (ExperimentConfig=256, score.py hardcodes 256,
     the gate ran 512); run_omni_v1_ablated.py sets MAX_LENGTH=512 and never
     passes `extraction`, so pooling falls to the function default last_token.
     We match that (512 / last_token) so IFEval numbers are comparable, and
     stamp it. A max_length=2048 variant is ALSO extracted as a robustness
     check, because IFEval responses are long and 512 truncates many of them
     (the "last token" then lands mid-response). Truncation rate is reported.
  2. Yes/No verdict logits — same-model behavioural floor, per INSTANCE
     (the natural-language instruction description is put in the prompt, so the
     baseline is told which instruction it is being asked about — the fair,
     strong version). Logic mirrors run_yesno_logit_baseline.py.
  3. the refusal direction (Arditi et al.), rebuilt with the identical recipe to
     run_refusal_direction_baseline.py (AdvBench vs Alpaca, 256/side, seeded).
     The vector is NOT cached anywhere on disk — only its per-layer norms are —
     so it is rebuilt and then ASSERTED bit-equal to the receipted
     `raw_norm_per_layer` in results/rebuild/floors/refusal_dir_ccpa_seed42.json.
     This is a reuse, not a refit: the recipe is deterministic and the assert
     proves identity.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ifeval_activations.py

Outputs:
    results/rebuild/ifeval/acts_seed42.npz
    results/rebuild/ifeval/gpu_features_seed42.json
"""

import argparse
import datetime
import json
import logging
import os
import random
import subprocess
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers as _transformers
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compliance.activations import extract_activations

from lm_eval.tasks.ifeval import instructions_registry

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("ifeval_acts")

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LENGTH = 512            # THE GATE's value — primary
MAX_LENGTH_LONG = 2048      # robustness variant
EXTRACTION = "last_token"   # run_omni_v1_ablated.py's effective default
CHAT_DATE_STRING = "01 Jan 2026"
N_DIR_PROMPTS = 256
DEFAULT_SEED = 42

YESNO_QUESTION = ("Does the response above follow this instruction? "
                  "Answer Yes or No.")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True).strip()
    except Exception:
        return "unknown"


def conv_text(tokenizer, prompt: str, response: str) -> str:
    """The (prompt, response) conversation Probe reads."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False, date_string=CHAT_DATE_STRING)


def instruction_description(iid: str, kwargs: dict) -> str:
    """Natural-language description of the instruction, from the official registry."""
    cls = instructions_registry.INSTRUCTION_DICT[iid]
    inst = cls(iid)
    kw = {k: v for k, v in kwargs.items() if v}
    try:
        desc = inst.build_description(**kw)
    except Exception:
        desc = iid
    return desc if isinstance(desc, str) else str(desc)


def resolve_token_ids(tokenizer) -> dict:
    ids = {}
    for label, s in [("yes_nospace", "Yes"), ("no_nospace", "No"),
                     ("yes_space", " Yes"), ("no_space", " No")]:
        enc = tokenizer.encode(s, add_special_tokens=False)
        assert len(enc) == 1, f"{s!r} is not a single token: {enc}"
        ids[label] = int(enc[0])
    assert ids["yes_nospace"] != ids["no_nospace"]
    assert ids["yes_space"] != ids["no_space"]
    return ids


@torch.no_grad()
def yesno_logits(records, model, tokenizer, ids):
    out = {k: [] for k in ids}
    for r in tqdm(records, desc="yes/no"):
        body = (f"{r['prompt']}\n\n---\nResponse:\n{r['response']}\n---\n\n"
                f"Instruction: {r['description']}\n\n{YESNO_QUESTION}")
        p = tokenizer.apply_chat_template([{"role": "user", "content": body}],
                                          tokenize=False, add_generation_prompt=True,
                                          date_string=CHAT_DATE_STRING)
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH, add_special_tokens=False).to(model.device)
        lg = model(**inp).logits[0, -1, :].float().cpu().numpy()
        for k, tid in ids.items():
            out[k].append(float(lg[tid]))
    return {k: np.asarray(v, float) for k, v in out.items()}


def build_refusal_directions(model, tokenizer, seed, root):
    """Rebuild Arditi refusal direction; assert bit-equal to the receipt."""
    harmful_all = [ex["prompt"] for ex in load_dataset("walledai/AdvBench", split="train")]
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless_all = [ex["instruction"] for ex in alpaca if ex["input"].strip() == ""]
    rng = random.Random(seed)
    harmful = rng.sample(harmful_all, min(N_DIR_PROMPTS, len(harmful_all)))
    harmless = rng.sample(harmless_all, min(N_DIR_PROMPTS, len(harmless_all)))

    receipt_path = os.path.join(root, "results/rebuild/floors/refusal_dir_ccpa_seed42.json")
    diag = json.load(open(receipt_path))["direction_diagnostics"]

    dirs, checks = {}, {}
    for variant in ("raw", "chat"):
        if variant == "raw":
            h_txt, s_txt = harmful, harmless
        else:
            f = lambda t: tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False,
                add_generation_prompt=True, date_string=CHAT_DATE_STRING)
            h_txt, s_txt = [f(t) for t in harmful], [f(t) for t in harmless]
        h = extract_activations(h_txt, model, tokenizer, max_length=MAX_LENGTH,
                                extraction=EXTRACTION)
        s = extract_activations(s_txt, model, tokenizer, max_length=MAX_LENGTH,
                                extraction=EXTRACTION)
        diff = h.mean(axis=0) - s.mean(axis=0)
        norms = np.linalg.norm(diff, axis=1)
        dirs[variant] = diff / (norms[:, None] + 1e-8)
        ref = np.array(diag[variant]["raw_norm_per_layer"])
        maxdiff = float(np.abs(norms - ref).max())
        checks[variant] = {"max_abs_norm_diff_vs_receipt": maxdiff,
                           "receipt": os.path.relpath(receipt_path, root),
                           "identical": bool(maxdiff < 1e-6)}
        logger.info("  refusal dir [%s]: max |norm - receipt| = %.3e", variant, maxdiff)
        assert maxdiff < 1e-6, (
            f"refusal direction [{variant}] does not reproduce the receipt "
            f"(max norm diff {maxdiff:.3e}) — cannot claim vector reuse")
    return dirs, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dir", default="results/rebuild/ifeval")
    args = ap.parse_args()

    seed_everything(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, args.dir)
    labels = json.load(open(os.path.join(out_dir, f"labels_seed{args.seed}.json")))
    inst = labels["instances"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).cuda().eval()
    logger.info("model on %s", next(model.parameters()).device)

    # ── unique conversation texts (shared across a prompt's instruction instances) ──
    keys, seen = [], {}
    for i in inst:
        if i["key"] not in seen:
            seen[i["key"]] = len(keys)
            keys.append(i["key"])
    key2row = seen
    by_key = {}
    for i in inst:
        by_key.setdefault(i["key"], i)
    texts = [conv_text(tokenizer, by_key[k]["prompt"], by_key[k]["response"]) for k in keys]

    ntok = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    trunc512 = int(sum(n > MAX_LENGTH for n in ntok))
    trunc2048 = int(sum(n > MAX_LENGTH_LONG for n in ntok))
    logger.info("texts: %d | >512 tok: %d (%.3f) | >2048 tok: %d",
                len(texts), trunc512, trunc512 / len(texts), trunc2048)

    logger.info("extracting activations @ max_length=%d (%s) ...", MAX_LENGTH, EXTRACTION)
    acts512 = extract_activations(texts, model, tokenizer, max_length=MAX_LENGTH,
                                  extraction=EXTRACTION)
    logger.info("extracting activations @ max_length=%d (%s) ...", MAX_LENGTH_LONG, EXTRACTION)
    acts2048 = extract_activations(texts, model, tokenizer, max_length=MAX_LENGTH_LONG,
                                   extraction=EXTRACTION)

    # ── Yes/No logits, per instance ──
    ids = resolve_token_ids(tokenizer)
    recs = [{"prompt": i["prompt"], "response": i["response"],
             "description": instruction_description(i["instruction_id"], i["kwargs"])}
            for i in inst]
    yn = yesno_logits(recs, model, tokenizer, ids)

    # ── refusal direction ──
    dirs, refusal_checks = build_refusal_directions(model, tokenizer, args.seed, root)

    np.savez_compressed(
        os.path.join(out_dir, f"acts_seed{args.seed}.npz"),
        acts512=acts512.astype(np.float32),
        acts2048=acts2048.astype(np.float32),
        keys=np.array(keys),
        refusal_raw=dirs["raw"].astype(np.float32),
        refusal_chat=dirs["chat"].astype(np.float32),
        **{f"yn_{k}": v for k, v in yn.items()},
    )

    meta = {
        "provenance": {
            "model": args.model, "seed": args.seed, "git_commit": git_commit(),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "script": "scripts/run_ifeval_activations.py",
            "versions": {"torch": torch.__version__,
                         "transformers": _transformers.__version__,
                         "numpy": np.__version__, "python": sys.version.split()[0]},
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "max_length": MAX_LENGTH,
            "max_length_robustness_variant": MAX_LENGTH_LONG,
            "pooling": EXTRACTION,
            "activation_text": "tokenizer.apply_chat_template([user=prompt, assistant=response])",
            "generation_config": labels["provenance"]["generation_config"],
        },
        "n_texts": len(texts),
        "n_instances": len(inst),
        "key_to_row": {str(k): v for k, v in key2row.items()},
        "instance_rows": [key2row[i["key"]] for i in inst],
        "token_lengths": ntok,
        "truncation": {"max_length": MAX_LENGTH, "n_truncated": trunc512,
                       "rate": trunc512 / len(texts),
                       "n_truncated_at_2048": trunc2048},
        "refusal_direction_check": refusal_checks,
        "yesno": {"question": YESNO_QUESTION, "token_ids": ids,
                  "note": "per-instance; instruction description injected"},
    }
    json.dump(meta, open(os.path.join(out_dir, f"gpu_features_seed{args.seed}.json"), "w"),
              indent=1)
    logger.info("wrote acts + gpu_features")


if __name__ == "__main__":
    main()
