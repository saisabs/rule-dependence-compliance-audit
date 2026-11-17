"""model_utils.py — small shared helper for loading a HF causal LM onto GPU.

Extracted as a standalone module (no dependency on unrelated experiment
scripts) because run_crossed_eval.py needs only this one function.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("model-utils")

LLAMA31_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"


def _num_layers(cfg):
    """num_hidden_layers, handling multimodal configs that nest it under text_config."""
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        tc = getattr(cfg, "text_config", None)
        n = getattr(tc, "num_hidden_layers", None) if tc is not None else None
    return n


def load_model(kind, path, family, adapter=None):
    """Load a model onto the (single visible) GPU in bfloat16.

    kind "hf"/"merged": load `path` directly. kind "peft": load base `path`,
    attach the LoRA `adapter`, and merge it in before extraction.
    Merged local dirs / base may lack a tokenizer; fall back to Llama-3.1."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_src = adapter if kind == "peft" else path
    try:
        tok = AutoTokenizer.from_pretrained(tok_src)
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(path)
        except Exception:
            logger.info("  tokenizer not found — falling back to %s", LLAMA31_TOKENIZER)
            tok = AutoTokenizer.from_pretrained(LLAMA31_TOKENIZER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True)
    if kind == "peft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        logger.info("  merged LoRA adapter %s onto %s", adapter, path)
    model.eval()
    logger.info("  loaded %s | layers=%s dtype=bf16", path, _num_layers(model.config))
    return model, tok
