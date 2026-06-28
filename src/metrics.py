"""
metrics.py — Evaluation metrics for POS tagging, implemented from scratch.

The spec is explicit (Section: Required Evaluation Metrics) that for EVERY model
and EVERY split we must report:

    * Per-class precision / recall / F1   (every tag in the tag set)
    * Micro-averaged   precision / recall / F1
    * Macro-averaged   precision / recall / F1
    * Overall token-level accuracy

The micro/macro distinction matters because the PTB tag distribution is highly
skewed: a model can post high micro-F1 while ignoring rare tags (EX, FW, LS).
Macro-F1 exposes that. We implement the counting ourselves rather than calling
sklearn so the numbers are fully defensible in the in-person discussion.

A sklearn cross-check is provided at the bottom for sanity only.
"""

from __future__ import annotations
from collections import defaultdict
from typing import List, Dict, Tuple


def _counts(y_true: List[int], y_pred: List[int], n_classes: int):
    """Return per-class true-positive, false-positive, false-negative counts."""
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    return tp, fp, fn


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(y_true: List[int],
                    y_pred: List[int],
                    idx2tag: List[str]) -> Dict:
    """Compute the full metric suite required by the spec.

    Returns a dict with: accuracy, per_class (dict tag -> {p,r,f1,support}),
    micro {p,r,f1}, macro {p,r,f1}.
    """
    n = len(idx2tag)
    tp, fp, fn = _counts(y_true, y_pred, n)

    per_class = {}
    macro_p = macro_r = macro_f1 = 0.0
    counted_classes = 0
    for c in range(n):
        support = tp[c] + fn[c]
        p = _safe_div(tp[c], tp[c] + fp[c])
        r = _safe_div(tp[c], tp[c] + fn[c])
        f1 = _safe_div(2 * p * r, p + r)
        per_class[idx2tag[c]] = {
            "precision": p, "recall": r, "f1": f1, "support": support,
        }
        # macro averages over classes that actually appear in y_true
        if support > 0:
            macro_p += p
            macro_r += r
            macro_f1 += f1
            counted_classes += 1

    macro = {
        "precision": _safe_div(macro_p, counted_classes),
        "recall": _safe_div(macro_r, counted_classes),
        "f1": _safe_div(macro_f1, counted_classes),
    }

    # micro: pool TP/FP/FN across all classes. For single-label multiclass this
    # equals accuracy, but we report all three explicitly as required.
    TP, FP, FN = sum(tp), sum(fp), sum(fn)
    micro_p = _safe_div(TP, TP + FP)
    micro_r = _safe_div(TP, TP + FN)
    micro = {
        "precision": micro_p,
        "recall": micro_r,
        "f1": _safe_div(2 * micro_p * micro_r, micro_p + micro_r),
    }

    accuracy = _safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p),
                         len(y_true))

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "micro": micro,
        "macro": macro,
    }


def metrics_summary(m: Dict) -> str:
    """One-line human-readable headline."""
    return (f"acc={m['accuracy']:.4f}  "
            f"micro-F1={m['micro']['f1']:.4f}  "
            f"macro-F1={m['macro']['f1']:.4f}")


def per_class_table(m: Dict, sort_by_support: bool = True) -> "pd.DataFrame":
    """Return a pandas DataFrame of per-class metrics for the Results section."""
    import pandas as pd
    rows = []
    for tag, d in m["per_class"].items():
        rows.append({
            "tag": tag,
            "precision": d["precision"],
            "recall": d["recall"],
            "f1": d["f1"],
            "support": d["support"],
        })
    df = pd.DataFrame(rows)
    if sort_by_support:
        df = df.sort_values("support", ascending=False).reset_index(drop=True)
    return df


def confusion_pairs(y_true: List[int], y_pred: List[int],
                    idx2tag: List[str], top_k: int = 20):
    """Most frequent (gold -> predicted) confusion pairs, for error analysis."""
    conf = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            conf[(idx2tag[t], idx2tag[p])] += 1
    return sorted(conf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def sklearn_crosscheck(y_true, y_pred, idx2tag):
    """Sanity cross-check against sklearn (NOT used for the reported numbers)."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    labels = list(range(len(idx2tag)))
    mi = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                         average="micro", zero_division=0)
    ma = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                         average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_f1": mi[2],
        "macro_f1": ma[2],
    }
