"""Goal Tracking and Goal-Shift Robustness Metrics."""
from typing import List, Set, Union, Dict
from sklearn.metrics import precision_recall_fscore_support

def compute_goal_shift_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1 for shift detection against ground truth."""
    if not y_true or not y_pred:
        return {"shift_precision": 0.0, "shift_recall": 0.0, "shift_f1": 0.0}
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "shift_precision": round(float(p), 4),
        "shift_recall": round(float(r), 4),
        "shift_f1": round(float(f1), 4)
    }

def compute_old_goal_evidence_recovery(
    retrieved_chunk_ids: Union[List[str], Set[str]],
    gold_old_goal_ids: Union[List[str], Set[str]]
) -> float:
    """Measure recovery of Goal A evidence when query returns to Goal A after a shift."""
    ret_set = set(retrieved_chunk_ids)
    gold_set = set(gold_old_goal_ids)
    if not gold_set:
        return 1.0
    return round(len(ret_set & gold_set) / len(gold_set), 4)

def compute_stale_context_rate(
    tier_a_chunk_ids: Union[List[str], Set[str]],
    stale_goal_b_chunk_ids: Union[List[str], Set[str]]
) -> float:
    """Measure the share of stale Goal B chunks contaminating Tier A after returning to Goal A."""
    tier_a_set = set(tier_a_chunk_ids)
    stale_set = set(stale_goal_b_chunk_ids)
    if not tier_a_set:
        return 0.0
    return round(len(tier_a_set & stale_set) / len(tier_a_set), 4)

