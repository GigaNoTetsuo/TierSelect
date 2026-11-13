"""Efficiency and Compression Metrics."""
import re
from typing import List, Dict, Any
from src.config import DialogueChunk

def compute_token_reduction_rate(b1_tokens: int, system_tokens: int) -> float:
    """Calculate token reduction percentage relative to Full History baseline B1."""
    if b1_tokens <= 0:
        return 0.0
    reduction = ((b1_tokens - system_tokens) / b1_tokens) * 100.0
    return round(float(reduction), 2)

def compute_compression_ratio(raw_chunks: List[DialogueChunk], digest_text: str) -> float:
    """Ratio of raw Tier B tokens to compressed digest tokens."""
    raw_tokens = sum(len(c.text.split()) for c in raw_chunks)
    digest_tokens = len(digest_text.split())
    if digest_tokens == 0:
        return 1.0
    return round(raw_tokens / digest_tokens, 2)

def compute_key_fact_retention_rate(raw_chunks: List[DialogueChunk], digest_text: str) -> float:
    """Proportion of salient named entities/dates/numbers from raw chunks preserved in digest."""
    entity_pattern = re.compile(r'\b[A-Z][a-z]+\b|\b\d{1,4}\b')
    raw_entities = set()
    for c in raw_chunks:
        raw_entities.update(entity_pattern.findall(c.text))
    if not raw_entities:
        return 1.0
    digest_entities = set(entity_pattern.findall(digest_text))
    retained = len(raw_entities & digest_entities)
    return round(retained / len(raw_entities), 4)

