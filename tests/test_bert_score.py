"""Test suite for BERTScore evaluation in answer_metrics."""
import pytest
from src.evaluation.answer_metrics import compute_bert_score, compute_bert_f1, compute_token_f1

def test_bert_score_paraphrase():
    pred = "Paris is the French capital city."
    gold = "The capital of France is Paris."
    
    # Calculate Token F1 (traditional lexical metric)
    tok_f1 = compute_token_f1(pred, gold)
    print(f"\n[Test] Token F1: {tok_f1:.4f}")
    
    # Calculate BERTScore
    bert_res = compute_bert_score(pred, gold, model_type="distilbert-base-uncased")
    print(f"[Test] BERTScore Precision: {bert_res['precision']:.4f}")
    print(f"[Test] BERTScore Recall:    {bert_res['recall']:.4f}")
    print(f"[Test] BERTScore F1:        {bert_res['f1']:.4f}")
    
    assert "f1" in bert_res
    assert 0.0 <= bert_res["f1"] <= 1.0

def test_bert_score_identical():
    pred = "The quick brown fox jumps over the lazy dog."
    gold = "The quick brown fox jumps over the lazy dog."
    f1 = compute_bert_f1(pred, gold, model_type="distilbert-base-uncased")
    print(f"\n[Test Identical] BERTScore F1: {f1:.4f}")
    assert f1 >= 0.95

if __name__ == "__main__":
    test_bert_score_paraphrase()
    test_bert_score_identical()
    print("All BERTScore tests passed successfully!")

