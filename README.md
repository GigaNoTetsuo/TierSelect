# TierSelect: Goal-Aware Three-Tier Context Selection for Conversational RAG

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![Status](https://img.shields.io/badge/benchmark-verified-success.svg)](outputs/)

A goal-aware three-tier selective context allocation framework for multi-session conversational RAG. TierSelect eliminates context contamination across conversational goal shifts by partitioning candidate dialogue chunks into **Tier A** (raw verbatim), **Tier B** (compressed entity digest), and **Tier C** (dropped distractors) using calibrated decision modeling.

---

## ⚡ Quick Start (Setup in 5 Steps)

### 1. Activate Environment
Use Python venv or conda

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env`
Ensure your `.env` file exists in the repository root with your API keys and configuration:
```env
# LLM Generator Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini-2025-04-14

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
TYPESAFE_API_KEY=your_typesafe_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# PostgreSQL Database (Persistent Goal Storage)
DB_HOST=localhost
DB_PORT=5435
DB_NAME=tierselect
DB_USER=tierselect
DB_PASSWORD=tierselect123

# TierSelect Hyperparameters
TAU_LOOSE=0.15
THETA_HI=0.50
THETA_LO=0.20
DELTA=0.05
```

### 4. Download & Verify Benchmark Datasets
> [!NOTE]
> The raw benchmark datasets are ~3.85 GB and are excluded from Git via `.gitignore`.
> Run the automated resumable downloader to fetch all benchmark data into `data/`:

```bash
# Download ImplexConv, LongMemEval, LoCoMo, and TopiOCQA
python extras/data_tools/download_resumable.py

# Verify files and schemas
python extras/data_tools/verify_datasets.py
```
*(The download script supports resuming interrupted transfers using HTTP Range headers).*

### 5. Start PostgreSQL via Docker
```bash
docker compose up -d
```
> **Note:** The database auto-initializes the schema from `src/storage/db_schema.sql` on port `5435`. If Docker is not running, TierSelect transparently falls back to in-memory goal storage.

---

## 🚀 Running Evaluations & Tests

The evaluation suites and auxiliary test scripts are clearly separated across `tests/` (actual research benchmarks) and `extras/` (unit tests, short development checks, and pipeline demos):

### 1. Actual Research Evaluation (Full Benchmark on Complete Datasets)
Located in `tests/actual_evaluation/`. Executes the full research evaluation across all benchmark datasets (**ImplexConv**, **TopiOCQA**, **LongMemEval**, **LoCoMo**) covering experiments **E1 through E5**:
```bash
python tests/actual_evaluation/run_all_experiments.py
```

You can also run individual research experiment benchmarks:
- `python tests/actual_evaluation/run_e1_baseline_comparison.py` (E1: Baselines B1–B5 vs TierSelect)
- `python tests/actual_evaluation/run_e2_goal_shift_analysis.py` (E2: Cross-Goal Contamination & Stale Rate)
- `python tests/actual_evaluation/run_e3_classifier_comparison.py` (E3: Classifier Decision Boundary Ablation)
- `python tests/actual_evaluation/run_e4_robustness_growth.py` (E4: Dialogue History Scalability)
- `python tests/actual_evaluation/run_e5_efficiency_breakdown.py` (E5: Computational Efficiency & Latency)

---

### 2. Short Evaluation (Fast 3-Episode Baseline Check)
Located in `extras/short_evaluation/`. Runs a fast 3-episode comparative check comparing **TierSelect** against all 5 baselines with live OpenAI generation, Token F1, and BERTScore just to test pipeline working well, before running full evaluation:
```bash
python extras/short_evaluation/run_short_baseline_comparison.py
```
*Results are saved to `outputs/short_baseline_comparison.json`.*

### 3. Unit & Function Tests (Module Verification)
Located in `extras/unit_tests/`. Runs isolated pytest unit tests covering individual code components (BERTScore, tier allocation, goal tracking, and storage):
```bash
pytest extras/unit_tests/ -v
```

### 4. Pipeline Demos & API Diagnostic Scripts (`extras/`)
Located in `extras/demos/` and `extras/data_tools/` for standalone pipeline checks, third-party API testing, and dataset integrity:
```bash
# Test live TypeSafe Jev decision primitives (Noul / Choice) and token usage
python extras/demos/test_jev_live.py

# Run standalone multi-turn pipeline test with PostgreSQL inspection
python extras/demos/demo_few_shifts.py

# Verify benchmark datasets integrity on disk
python extras/data_tools/verify_datasets.py
```

---

## 📁 Repository Structure & Folder Descriptions

Here is a general summary of what each directory contains:

* **`src/` (Core Framework Implementation):**
  Contains all foundational library code for TierSelect: the 3-tier selective context allocation engine (`tierselect/`), model integrations including TypeSafe Jev and OpenAI/Groq clients (`models/`), PostgreSQL persistence and schema (`storage/`), dialogue chunking and episode loaders (`preprocessing/`), 5 reference comparative baselines B1–B5 (`baselines/`), and evaluation metrics for answers, selection, and efficiency (`evaluation/`).

* **`tests/` (Actual Research Evaluation Suite):**
  Dedicated strictly to executing the formal research evaluation benchmarks across the complete real-world datasets:
  * **`tests/actual_evaluation/`:** Contains the end-to-end benchmark runners (`run_all_experiments.py` and `run_e1_...` through `run_e5_...`) evaluating experiments E1–E5 across ImplexConv, TopiOCQA, LongMemEval, and LoCoMo datasets.

* **`extras/` (Auxiliary Checks, Unit Tests, Demos & Utilities):**
  Houses non-primary research scripts, module test suites, and diagnostic tools:
  * **`extras/short_evaluation/`:** Fast 3-episode benchmark (`run_short_baseline_comparison.py`) comparing TierSelect against B1–B5 with live OpenAI generation and BERTScore for rapid development cycles.
  * **`extras/unit_tests/`:** Isolated pytest unit test suites for framework components (`test_bert_score.py`, `test_tierselect.py`, `test_storage_and_llm.py`, `test_typesafe_jev.py`).
  * **`extras/demos/`:** Standalone multi-turn goal shift pipeline demo (`demo_few_shifts.py`) and live TypeSafe Jev API connectivity verification (`test_jev_live.py`).
  * **`extras/data_tools/`:** Data acquisition utilities (`download_resumable.py`) and dataset SHA-256 integrity checkers (`verify_datasets.py`).

* **`data/` (Benchmark Datasets):**
  Houses the verified research datasets used for experimental evaluation: **LoCoMo** (multi-session conversational memory), **LongMemEval** (long-term memory and needle retrieval), **TopiOCQA** (conversational QA with topic shifts), and **ImplexConv** (implicit goal shifts with supportive and opposed turns).
  *(Excluded from Git tracking via `.gitignore` due to large file sizes (~3.85 GB); populated automatically using `python extras/data_tools/download_resumable.py`).*

* **`outputs/` (Benchmark Results & Metrics):**
  Stores all generated experimental output logs, comparative benchmark tables, token efficiency statistics, and JSON summaries produced by the evaluation runs.

---

## ⚙️ Key Hyperparameters

| Parameter | Meaning | Recommended Value |
| :--- | :--- | :---: |
| `TAU_LOOSE` | Candidate retrieval cosine similarity threshold | `0.15` |
| `THETA_HI` | High relevance threshold for Tier A (raw verbatim) | `0.50` |
| `THETA_LO` | Low relevance threshold for Tier B (compressed entity digest) | `0.20` |
| `DELTA` | Goal margin added to $\theta_{hi}$ for archived goal chunks | `0.05` |
| `THETA_SHIFT` | Calibrated probability threshold to trigger goal shift | `0.50` |
