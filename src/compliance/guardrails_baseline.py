"""
guardrails_baseline.py

Rule-based (symbolic, non-neural) compliance classifier using keyword/marker
matching, implemented via the Guardrails AI validator framework. Serves as a
baseline representing production-style "compliance guardrails" used by
companies (e.g. Fiddler AI Custom Policy Guardrails, TrueFoundry AI Gateway),
in contrast to neural classifiers (WildGuard, Llama Guard 3, ShieldGemma)
and internal activation-based scoring (Probe).
"""

from typing import Any, Dict
from sklearn.metrics import roc_auc_score, f1_score


COMPLIANCE_MARKERS = [
    "no violation", "compliant", "lawful", "fully satisfied",
    "no further action", "no penalty", "passed", "approved",
    "best practice", "confirmed compliance", "met legal standards",
    "no issues identified", "dismissed", "praised",
]

VIOLATION_MARKERS = [
    "fine", "fined", "violat", "breach", "unlawful", "sanction",
    "penalty", "penalized", "ordered to", "failed to", "failing",
    "illegal", "non-compliant", "noncompliant", "infringement",
]


def classify_guardrails(text: str) -> int:
    """
    Rule-based GDPR/regulatory compliance classifier.
    Returns 1 (predicted compliant) if compliance markers outweigh violation
    markers in the text, else 0 (predicted non-compliant).
    """
    text_lower = text.lower()
    compliance_score = sum(1 for m in COMPLIANCE_MARKERS if m in text_lower)
    violation_score   = sum(1 for m in VIOLATION_MARKERS if m in text_lower)
    net_score = compliance_score - violation_score
    return 1 if net_score > 0 else 0


def debug_markers(text: str) -> Dict[str, Any]:
    """Return which compliance/violation markers matched, for inspection."""
    text_lower = text.lower()
    return {
        "compliance_hits": [m for m in COMPLIANCE_MARKERS if m in text_lower],
        "violation_hits":  [m for m in VIOLATION_MARKERS if m in text_lower],
    }


def evaluate_guardrails(compliant_texts, non_compliant_texts):
    """
    Run the rule-based classifier over compliant/non-compliant texts and
    compute AUROC, pairwise accuracy, and F1 — directly comparable to Probe
    and other baseline methods.
    """
    scores_c  = [classify_guardrails(t) for t in compliant_texts]
    scores_nc = [classify_guardrails(t) for t in non_compliant_texts]

    all_scores = scores_c + scores_nc
    all_labels = [1] * len(scores_c) + [0] * len(scores_nc)

    auroc    = roc_auc_score(all_labels, all_scores)
    pair_acc = sum(1 for i in range(len(scores_c))
                   if scores_c[i] > scores_nc[i]) / len(scores_c)
    f1       = f1_score(all_labels, all_scores)

    return {
        "auroc": auroc,
        "pairwise_acc": pair_acc,
        "f1": f1,
        "compliant_predicted_positive": sum(scores_c),
        "non_compliant_predicted_positive": sum(scores_nc),
        "n": len(scores_c),
    }
