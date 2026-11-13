"""Configuration and data structures for TierSelect."""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration from environment
TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gpt-4o-mini")
COMPRESSOR_MODEL = os.environ.get("COMPRESSOR_MODEL", "gpt-4o-mini")

class TierSelectConfig(BaseModel):
    """Configuration hyperparameters for TierSelect."""
    # Retrieval
    tau_loose: float = Field(float(os.environ.get("TAU_LOOSE", "0.30")), description="Loose semantic retrieval cosine threshold")
    k_max: int = Field(int(os.environ.get("K_MAX", "50")), description="Maximum number of candidate chunks retrieved")
    
    # Tier Allocation Thresholds
    theta_hi: float = Field(float(os.environ.get("THETA_HI", "0.65")), description="High relevance threshold for Tier A (raw)")
    theta_lo: float = Field(float(os.environ.get("THETA_LO", "0.40")), description="Low relevance threshold for Tier B (compressed)")
    delta: float = Field(float(os.environ.get("DELTA", "0.15")), description="Goal margin added to theta_hi for archived goal chunks")
    n_raw: int = Field(int(os.environ.get("N_RAW", "5")), description="Max raw chunks allowed in Tier A")
    
    # Goal Tracking
    theta_shift: float = Field(float(os.environ.get("THETA_SHIFT", "0.50")), description="Calibrated probability threshold to trigger goal shift")
    active_history_turns: int = Field(int(os.environ.get("ACTIVE_HISTORY_TURNS", "6")), description="Number of recent turns passed to goal tracker")
    
    # Compression & Budget
    max_digest_tokens: int = Field(int(os.environ.get("MAX_DIGEST_TOKENS", "150")), description="Maximum token length L_c of Tier B digest")
    token_budget: int = Field(int(os.environ.get("TOKEN_BUDGET", "4096")), description="Overall prompt context token budget B")
    
    # Calibration
    ece_target: float = Field(float(os.environ.get("ECE_TARGET", "0.05")), description="Expected Calibration Error threshold for isotonic fitting")

@dataclass
class DialogueTurn:
    """Individual dialogue utterance."""
    turn_id: int
    speaker: str
    text: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DialogueChunk:
    """Turn-pair chunk (User-Assistant pair) with metadata and backpointers."""
    chunk_id: str
    session_id: str
    user_turn: str
    assistant_turn: str
    text: str
    speaker_pair: str = "user-assistant"
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    timestamp: Optional[str] = None
    segment_id: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CandidateChunk:
    """Candidate chunk scored by retrieval and relevance classifier."""
    chunk: DialogueChunk
    retrieval_score: float
    relevance_probability: float
    decision_type: str = "irrelevant" # "relevant", "partially_relevant", "irrelevant"
    assigned_tier: Optional[str] = None # "A", "B", "C"

@dataclass
class Episode:
    """Shift-and-return or multi-session evaluation episode."""
    episode_id: str
    goal_a_chunks: List[DialogueChunk]
    goal_b_chunks: List[DialogueChunk]
    test_query: str
    gold_answer: str
    gold_evidence_chunk_ids: List[str]
    episode_type: str = "shift_and_return" # "shift_and_return", "continuation", "stay"
    shift_turn_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

