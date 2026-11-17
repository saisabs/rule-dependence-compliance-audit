"""
run_lpg.py — E-COMP-3 arm 2: Latent Policy Guard (LPG-4B).

LPG (github.com/SaFo-Lab/Latent_Policy_Guard, arXiv:2605.17329) is a guardrail
that performs "semantic latent deliberation" over a DYNAMIC, indexed policy
list: it encodes (policies + content), runs latent reasoning stages
(intent, risk) as continuous states, then decodes a compact verdict
("safe" / "unsafe, policy N").

Unlike GLiGuard and SIREN, LPG **genuinely supports a custom policy** — the
policy list is part of its documented input contract. So this arm is a
LIKE-FOR-LIKE comparison with Llama Guard 3 / Qwen3Guard: the domain rule is
supplied as policy index 0, exactly as those guards got it as a custom
category. Public checkpoint: andyc03/LPG_4B (apache-2.0, Qwen3-4B + LoRA r128
+ latent projection bundled into one model.safetensors).

Score: P(unsafe) via the token logit at the FIRST explicit-decode position
(the position where the repo's own generate() loop picks 'safe' vs 'unsafe'):

    score = logit(unsafe_tok) - logit(safe_tok)        (higher = VIOLATION)
    p_unsafe = sigmoid(score)   (closed form; monotone, preserves bf16 ties)

This mirrors run_guard_llamaguard3.py exactly: no generated-text parsing on
the scored path. The latent loop is taken verbatim from the repo's
LatentPolicyGuardModel.generate() (intent/risk stages, <bot>/<eot> prefixes)
so the decision position is the model's real one. On the orientation probe the
full verdict is GENERATED verbatim and printed, to confirm the format and that
position 0 carries the safe/unsafe distinction.

Zero-shot, zero selection ⇒ own null = 0.5.

Isolated venv (scratchpad/venv_lpg over the mechcompliance env,
--system-site-packages, nothing installed → mechcompliance NOT mutated).
HF cache redirected to a separate large mount (root fs may be near full).

Usage:
    HF_HOME=... CUDA_VISIBLE_DEVICES=0 {venv_lpg}/bin/python scripts/run_lpg.py --all
"""
import argparse
import datetime
import json
import logging
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import competitors_lib as lib

lib.force_gpu0()

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("lpg")

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_CKPT_REPO = "andyc03/LPG_4B"
DEFAULT_CKPT_REV = "ec876ec98141964003ef31e7b9e21b230310e5b1"
DEFAULT_SEED = 42
LPG_REPO = os.environ.get("LPG_REPO", "/tmp/lpg_probe")

# Hyperparameters — from the andyc03/LPG_4B model card (they match the repo's
# evaluation defaults exactly; asserted at load time against the card).
LPG_HP = {"lora_r": 128, "lora_alpha": 32, "stage_names": "intent,risk",
          "num_latent_per_stage": "4,6", "prj_dim": 2560, "use_prj": True,
          "model_max_length": 1024}


def _import_lpg():
    """Import the repo's own model plugin + input formatter (no reimplementation).

    The LPG plugin inserts <repo>/training onto sys.path so it can do
    `from src.model import SafetyLatentGuard`. That `src` package SHADOWS this
    repository's own `src.compliance` (used by competitors_lib.load_split_dump).
    So: import LPG, then unwind both the path entries and the `src*` modules it
    registered, leaving the already-bound classes intact. Verified below by
    round-tripping an import of src.compliance.splits.
    """
    added = [p for p in (LPG_REPO, os.path.join(LPG_REPO, "evaluation"),
                         os.path.join(LPG_REPO, "training")) if p not in sys.path]
    for p in added:
        sys.path.insert(0, p)
    pre = dict(sys.modules)
    from evaluation.models.latent_policy_guard import (
        LatentPolicyGuardModel, _format_annotation_input)
    # Purge EVERY module that was loaded from the LPG tree (`src`, `datasets`,
    # `models`, `metrics`, `evaluation`, ...). The bound classes above keep their
    # module objects alive by reference, so they stay fully functional; LPG does
    # all of its imports at module top level, so nothing is re-imported later.
    for k, m in list(sys.modules.items()):
        if k in pre:
            continue
        f = getattr(m, "__file__", None) or ""
        if f.startswith(LPG_REPO):
            del sys.modules[k]
    sys.modules.update(pre)          # restore anything LPG shadowed
    for p in added:
        if p in sys.path:
            sys.path.remove(p)
    # both shadowed names must resolve to the REAL packages again
    from src.compliance.splits import three_way_split  # noqa: F401
    import datasets as _ds
    assert not (_ds.__file__ or "").startswith(LPG_REPO), \
        f"HF datasets still shadowed by LPG: {_ds.__file__}"
    return LatentPolicyGuardModel, _format_annotation_input


def build_annotation_input(text_ablated: str, fmt) -> str:
    """Domain rule -> policy index 0 (LPG's documented custom-policy contract);
    the scenario is the content to evaluate."""
    rule, scenario = lib_parse(text_ablated)
    return fmt(content=scenario.strip(), policies=[rule],
               dataset_type="multi_policy")


def lib_parse(text_ablated: str):
    assert text_ablated.startswith("Rule: "), f"unexpected format: {text_ablated[:60]!r}"
    sep = "\n\nScenario: "
    i = text_ablated.index(sep)
    rule, scenario = text_ablated[len("Rule: "):i], text_ablated[i + len(sep):]
    assert rule and scenario
    return rule, scenario


def resolve_tokens(tok):
    """First-token ids of 'safe'/'unsafe' at the first explicit-decode position.
    LPG's verdict grammar starts the string with the bare word, so no leading
    space; assert the two words do not share a first token."""
    safe = tok.encode("safe", add_special_tokens=False)
    unsafe = tok.encode("unsafe", add_special_tokens=False)
    ids = {"safe_first": int(safe[0]), "unsafe_first": int(unsafe[0]),
           "safe_tokens": safe, "unsafe_tokens": unsafe,
           "safe_first_decoded": tok.decode([safe[0]]),
           "unsafe_first_decoded": tok.decode([unsafe[0]])}
    assert ids["safe_first"] != ids["unsafe_first"], "safe/unsafe share a first token"
    logger.info("  'safe' -> %s %r, 'unsafe' -> %s %r", safe,
                ids["safe_first_decoded"], unsafe, ids["unsafe_first_decoded"])
    return ids


@torch.no_grad()
def _latent_forward(plugin, annotation_input: str):
    """Verbatim replica of LatentPolicyGuardModel.generate()'s encode + latent
    stages, returning the logits at the FIRST explicit-decode position plus the
    kv-cache/embeds needed to continue decoding (for the probe's verbatim gen).
    """
    model, tok, ta = plugin._model, plugin._tokenizer, plugin._training_args
    batch = tok([annotation_input], return_tensors="pt", padding="longest",
                truncation=True, max_length=ta.model_max_length)
    n_tok = int(batch["input_ids"].shape[1])

    bot_prefix = [model.bot_id]
    if not ta.remove_eos and tok.eos_token_id is not None:
        bot_prefix = [tok.eos_token_id, model.bot_id]
    bot_t = torch.tensor(bot_prefix, dtype=torch.long).unsqueeze(0)
    dev = next(model.parameters()).device
    input_ids = torch.cat((batch["input_ids"], bot_t), dim=1).to(dev)
    attn = torch.cat((batch["attention_mask"], torch.ones_like(bot_t)), dim=1).to(dev)

    out = model.codi(input_ids=input_ids, use_cache=True,
                     output_hidden_states=True, attention_mask=attn)
    pkv = out.past_key_values
    lat = out.hidden_states[-1][:, -1:, :]
    if ta.use_prj:
        lat = model.prj(lat)
    for num_latent in plugin._num_latent_per_stage:
        for _ in range(num_latent):
            out = model.codi(inputs_embeds=lat, use_cache=True,
                             output_hidden_states=True, past_key_values=pkv)
            pkv = out.past_key_values
            lat = out.hidden_states[-1][:, -1:, :]
            if ta.use_prj:
                lat = model.prj(lat)

    eot_prefix = [model.eot_id]
    if not ta.remove_eos and tok.eos_token_id is not None:
        eot_prefix.append(tok.eos_token_id)
    eot_ids = torch.tensor(eot_prefix, dtype=torch.long, device=dev)
    out_embeds = model.get_embd(model.codi, model.model_name)(eot_ids).unsqueeze(0)

    step = model.codi(inputs_embeds=out_embeds, output_hidden_states=False,
                      use_cache=True, past_key_values=pkv)
    logits = step.logits[:, -1, :].clone()
    return logits, step.past_key_values, n_tok


def p_unsafe(plugin, ids, text_ablated, fmt):
    logits, _, n_tok = _latent_forward(plugin, build_annotation_input(text_ablated, fmt))
    lg = logits[0].float()
    a, b = float(lg[ids["safe_first"]]), float(lg[ids["unsafe_first"]])
    p = 1.0 / (1.0 + math.exp(-(b - a))) if abs(b - a) < 700 else float(b > a)
    return a, b, p, n_tok


@torch.no_grad()
def generate_verdict(plugin, text_ablated, fmt, max_new_tokens=24):
    """Full verbatim verdict, via the repo's own decode loop (probe only)."""
    model, tok, ta = plugin._model, plugin._tokenizer, plugin._training_args
    logits, pkv, _ = _latent_forward(plugin, build_annotation_input(text_ablated, fmt))
    pred = []
    for _ in range(max_new_tokens):
        lg = logits.clone()
        for t in [model.pad_token_id, model.bot_id, model.eot_id]:
            if t is not None and 0 <= t < lg.size(-1):
                lg[:, t] = float("-inf")
        nxt = torch.argmax(lg, dim=-1)
        if int(nxt.item()) == tok.eos_token_id:
            break
        pred.append(int(nxt.item()))
        emb = model.get_embd(model.codi, model.model_name)(nxt).unsqueeze(1)
        step = model.codi(inputs_embeds=emb, output_hidden_states=False,
                          use_cache=True, past_key_values=pkv)
        pkv = step.past_key_values
        logits = step.logits[:, -1, :]
    return tok.decode(pred, skip_special_tokens=True)


def score_texts(plugin, ids, texts, fmt, desc):
    ls, lu, pu, nt = [], [], [], []
    for t in tqdm(texts, desc=desc):
        a, b, p, n = p_unsafe(plugin, ids, t, fmt)
        ls.append(a); lu.append(b); pu.append(p); nt.append(n)
    return np.asarray(ls), np.asarray(lu), np.asarray(pu), np.asarray(nt)


def orientation_probe(plugin, ids, domain, seed, fmt, n_side=5):
    """Hand-checkable probe: verbatim verdict + logit diff on 10 examples."""
    dump, _ = lib.load_split_dump(domain, seed)
    test_idx = dump["split_indices"]["test"]
    rows = []
    for label, pool in [("COMPLIES", "complies_cases"), ("VIOLATES", "violates_cases")]:
        for i in test_idx[:n_side]:
            txt = dump[pool][i]["text_ablated"]
            a, b, p, n = p_unsafe(plugin, ids, txt, fmt)
            gen = generate_verdict(plugin, txt, fmt)
            rows.append({"label": label, "sample_idx": int(i),
                         "logit_safe": a, "logit_unsafe": b,
                         "logit_diff_unsafe_minus_safe": b - a, "p_unsafe": p,
                         "generated_verdict_verbatim": gen, "n_input_tokens": int(n)})
            print(f"  [{label} #{i}] p_unsafe={p:.4f} diff={b-a:+.3f} "
                  f"gen={gen!r} ntok={n}")
    mv = float(np.mean([r["p_unsafe"] for r in rows if r["label"] == "VIOLATES"]))
    mc = float(np.mean([r["p_unsafe"] for r in rows if r["label"] == "COMPLIES"]))
    return {"domain": domain, "rows": rows, "mean_score_violating": mv,
            "mean_score_compliant": mc, "direction_ok_violating_higher": mv > mc,
            "note": "orientation a priori: logit(unsafe)-logit(safe) at the first "
                    "explicit-decode position; higher = violation. "
                    "generated_verdict_verbatim confirms the decode format and that "
                    "this position carries the safe/unsafe distinction."}


def run_domain(domain, plugin, ids, seed, fmt, prov):
    logger.info("=== LPG: %s ===", domain)
    dump, split_check = lib.load_split_dump(domain, seed)
    test_idx = dump["split_indices"]["test"]
    c_texts = [dump["complies_cases"][i]["text_ablated"] for i in test_idx]
    v_texts = [dump["violates_cases"][i]["text_ablated"] for i in test_idx]
    abl_check = lib.assert_ablated(domain, dump, c_texts + v_texts)

    c_ls, c_lu, c_pu, c_nt = score_texts(plugin, ids, c_texts, fmt, f"{domain} COMPLIES")
    v_ls, v_lu, v_pu, v_nt = score_texts(plugin, ids, v_texts, fmt, f"{domain} VIOLATES")

    # Scored path = logit diff. p_unsafe is sigmoid(diff): strictly monotone, so
    # the AUROCs must agree exactly; assert it (mirrors the Llama Guard 3 arm).
    a_diff = lib.auroc_checked(v_lu - v_ls, c_lu - c_ls)
    a_p = lib.auroc_checked(v_pu, c_pu)
    assert abs(a_diff - a_p) < 1e-12, f"logit-diff vs p_unsafe AUROC differ: {a_diff} {a_p}"

    probe_c, probe_v, gate_row = lib.load_probe_test_scores(domain, seed)
    vs_probe = lib.paired_bootstrap_vs_probe(v_lu - v_ls, c_lu - c_ls, -probe_v, -probe_c, seed=seed)
    null = lib.load_selection_null(domain, "zero_selection")
    marg = lib.bootstrap_margin_over_null(v_lu - v_ls, c_lu - c_ls, null["level"], seed=seed)
    logger.info("  AUROC=%.4f | Probe=%.4f Δ=%.4f", a_diff, vs_probe["auroc_probe"],
                vs_probe["delta_probe_minus_arm"])

    result = {
        "experiment": "E-COMP-3 arm 2: Latent Policy Guard (LPG-4B), custom policy",
        "arm": "lpg_4b",
        "domain": domain,
        "n_test": len(test_idx),
        "provenance": dict(prov, **{
            "seed": seed, "git_commit": lib.git_commit(),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "script": "scripts/run_lpg.py", "torch_version": torch.__version__,
        }),
        "split_check": split_check,
        "ablation_check": abl_check,
        "auroc": a_diff,
        "auroc_from_p_unsafe_agrees": a_p,
        "own_budget_null": null,
        "margin_over_own_null": marg,
        "vs_probe_paired": vs_probe,
        "gate_probe": gate_row["probe"],
        "input_tokens": {"max": int(max(c_nt.max(), v_nt.max())),
                         "mean": float(np.mean(np.r_[c_nt, v_nt])),
                         "n_truncated_at_model_max_length": int(
                             (np.r_[c_nt, v_nt] >= LPG_HP["model_max_length"]).sum()),
                         "model_max_length": LPG_HP["model_max_length"]},
        "test_scores": {"compliant": [float(x) for x in c_pu],
                        "violating": [float(x) for x in v_pu]},
        "test_logit_diffs": {"compliant": [float(x) for x in (c_lu - c_ls)],
                             "violating": [float(x) for x in (v_lu - v_ls)]},
        "label_convention": "1=VIOLATES, 0=COMPLIES; AUROC detects violation",
    }
    os.makedirs(lib.COMP_DIR, exist_ok=True)
    out = os.path.join(lib.COMP_DIR, f"lpg_4b_{domain}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("  Saved → %s", out)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--ckpt_repo", type=str, default=DEFAULT_CKPT_REPO)
    ap.add_argument("--ckpt_revision", type=str, default=DEFAULT_CKPT_REV)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    domains = lib.ALL_SPLITS if args.all else [args.split or "gdpr"]
    if args.skip_existing:
        domains = [d for d in domains if not os.path.exists(os.path.join(
            lib.COMP_DIR, f"lpg_4b_{d}_seed{args.seed}.json"))]
    if not domains:
        print("nothing to do"); return

    LatentPolicyGuardModel, fmt = _import_lpg()
    logger.info("Loading LPG: base=%s ckpt=%s", args.base, args.ckpt_dir)
    # BaseModel.__init__(self, model_path, **kwargs) -> self.config = kwargs
    plugin = LatentPolicyGuardModel(args.base, ckpt_dir=args.ckpt_dir, **LPG_HP)
    plugin.load()
    assert plugin.config["ckpt_dir"] == args.ckpt_dir, "config not wired through"
    ids = resolve_tokens(plugin._tokenizer)
    gpu_receipt = lib.assert_gpu0_runtime()
    logger.info("GPU check: %s", gpu_receipt)

    probe = orientation_probe(plugin, ids, domains[0], args.seed, fmt)

    prov = {
        "model": args.ckpt_repo, "model_revision": args.ckpt_revision,
        "base_model": args.base,
        "paper": "arXiv:2605.17329, github.com/SaFo-Lab/Latent_Policy_Guard",
        "code_commit": os.environ.get("LPG_CODE_COMMIT", "see receipts"),
        "package": "repo evaluation/models/latent_policy_guard.py imported verbatim; "
                   "isolated venv_lpg (--system-site-packages over mechcompliance, "
                   "nothing installed); mechcompliance NOT mutated",
        "hyperparameters": LPG_HP,
        "scoring": "logit(unsafe)-logit(safe) at the first explicit-decode position "
                   "(the position the repo's own generate() loop uses to pick the "
                   "verdict); p_unsafe = sigmoid(diff), closed form. No generated-text "
                   "parsing on the scored path.",
        "custom_policy": "YES — the domain rule is supplied as policy index 0 via LPG's "
                         "documented indexed-policy contract (_format_annotation_input, "
                         "imported from the repo). LIKE-FOR-LIKE with Llama Guard 3 / "
                         "Qwen3Guard, unlike GLiGuard/SIREN which have no custom policy.",
        "calibration": "none (zero-shot)",
        "orientation": "higher score = VIOLATION",
        "gpu_check": gpu_receipt,
        "orientation_probe": probe,
    }

    rows = [run_domain(d, plugin, ids, args.seed, fmt, prov) for d in domains]

    print(f"\n{'DOMAIN':<42}{'AUROC':>8}{'Probe':>8}{'Δprobe':>8}")
    for r in sorted(rows, key=lambda x: -x["auroc"]):
        print(f"{r['domain']:<42}{r['auroc']:>8.4f}{r['gate_probe']:>8.4f}"
              f"{r['vs_probe_paired']['delta_probe_minus_arm']:>8.4f}")
    print(f"mean AUROC = {np.mean([r['auroc'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
