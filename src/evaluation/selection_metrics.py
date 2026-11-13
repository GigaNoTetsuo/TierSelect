"""Selection Quality metrics: Evidence Recall, Precision, Context F1, AUROC, and nDCG."""
from typing import List, Set, Union
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score

def compute_evidence_recall(retrieved_ids: Union[List[str], Set[str]], gold_ids: Union[List[str], Set[str]]) -> float:
    """Calculate recall of gold evidence chunks within the selected set."""
    ret_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    if not gold_set:
        return 1.0
    return len(ret_set & gold_set) / len(gold_set)

def compute_context_precision(retrieved_ids: Union[List[str], Set[str]], gold_ids: Union[List[str], Set[str]]) -> float:
    """Calculate precision of selected chunks containing gold evidence."""
    ret_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    if not ret_set:
        return 0.0
    return len(ret_set & gold_set) / len(ret_set)

def compute_context_f1(precision: float, recall: float) -> float:
    """Harmonic mean of context precision and evidence recall."""
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)

def compute_auroc_ndcg(scores: List[float], binary_labels: List[int]) -> dict:
    """Compute AUROC and nDCG for relevance scores vs gold relevance labels."""
    if len(set(binary_labels)) < 2:
        return {"auroc": 1.0, "ndcg": 1.0}
    try:
        auroc = float(roc_auc_score(binary_labels, scores))
    except Exception:
        auroc = 0.5
    try:
        ndcg = float(ndcg_score([binary_labels], [scores]))
    except Exception:
        ndcg = 0.5
    return {"auroc": round(auroc, 4), "ndcg": round(ndcg, 4)}

