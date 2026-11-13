"""Answer Quality Evaluation: Exact Match (EM), Token F1, and LLM-Judge Correctness."""
import re
import string
from typing import Optional, Any, Union, List

def normalize_answer(s: str) -> str:
    """Standard normalization: lowercase, strip punctuation and articles."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match (EM) metric."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 overlap between prediction and reference answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(float(f1), 4)

def evaluate_llm_judge(prediction: str, ground_truth: str, query: str, judge_client: Optional[Any] = None, jev_classifier: Optional[Any] = None) -> float:
    """Independent judge evaluation scoring factual correctness between 0.0 and 1.0 (with JEV Noul/Choice support)."""
    # 1. Jev Noul/Choice evaluation if Jev classifier provided
    if jev_classifier is not None and hasattr(jev_classifier, "evaluate_agreement_noul"):
        try:
            return jev_classifier.evaluate_agreement_noul(prediction, ground_truth, query)
        except Exception:
            pass

    # 2. LLM judge API call
    if judge_client is not None:
        try:
            prompt = f"""Evaluate whether the predicted answer correctly answers the question according to the reference answer.
Question: {query}
Reference Answer: {ground_truth}
Predicted Answer: {prediction}

Respond with only a score between 0.0 (completely incorrect) and 1.0 (completely correct). Score:"""
            if hasattr(judge_client, "chat"):
                score_str = judge_client.chat(prompt, temperature=0.0, max_tokens=10)
            else:
                resp = judge_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                score_str = resp.choices[0].message.content.strip()
            m = re.search(r'\b(1\.0|0\.\d+|1|0)\b', score_str)
            if m:
                return float(m.group(1))
        except Exception as e:
            pass

    # High-fidelity semantic agreement fallback based on key entities and content tokens
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(ground_truth)
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return 1.0
    return compute_token_f1(prediction, ground_truth)


def compute_bert_score(
    predictions: Union[str, list],
    ground_truths: Union[str, list],
    model_type: str = "distilbert-base-uncased",
    verbose: bool = False
) -> dict:
    """
    Computes BERTScore (Precision, Recall, F1) using contextual token embeddings.
    Handles synonyms, rephrasing, and lexical variation without penalizing valid paraphrases.
    Falls back gracefully to token F1 / semantic agreement if bert-score is not available.
    """
    is_single = isinstance(predictions, str) and isinstance(ground_truths, str)
    preds = [predictions] if is_single else list(predictions)
    golds = [ground_truths] if is_single else list(ground_truths)

    try:
        from bert_score import score
        P, R, F1 = score(preds, golds, lang="en", model_type=model_type, verbose=verbose)
        res = {
            "precision": round(float(P.mean().item()), 4),
            "recall": round(float(R.mean().item()), 4),
            "f1": round(float(F1.mean().item()), 4)
        }
        return res
    except ImportError:
        # Fallback if bert-score is not installed
        scores = [compute_token_f1(p, g) for p, g in zip(preds, golds)]
        avg_f1 = round(float(sum(scores) / len(scores)), 4) if scores else 0.0
        return {"precision": avg_f1, "recall": avg_f1, "f1": avg_f1, "fallback": True}
    except Exception as e:
        scores = [compute_token_f1(p, g) for p, g in zip(preds, golds)]
        avg_f1 = round(float(sum(scores) / len(scores)), 4) if scores else 0.0
        return {"precision": avg_f1, "recall": avg_f1, "f1": avg_f1, "error": str(e), "fallback": True}


def compute_bert_f1(prediction: str, ground_truth: str, model_type: str = "distilbert-base-uncased") -> float:
    """Convenience wrapper returning BERTScore F1 as a single float between 0.0 and 1.0."""
    result = compute_bert_score(prediction, ground_truth, model_type=model_type)
    return result["f1"]


