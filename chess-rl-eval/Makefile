# chess-rl-eval/Makefile
# Maps to the Execution Runbook in the Build Specification (Section 8).

PYTHON      := .venv/bin/python
PIP         := .venv/bin/pip
MODEL       ?= Qwen/Qwen2.5-Coder-7B-Instruct
CHECKPOINT  ?= ./checkpoints/grpo_chess_step_5000
OPPONENT    ?= maia-1300
GAMES       ?= 50
PUZZLE_CNT  ?= 1200
SAMPLES     ?= 100
LIMIT       ?= 200

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: venv install clean

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install torch transformers vllm python-chess zstandard pyyaml \
	              pytest aiosqlite requests tqdm numpy stockfish peft

# ── Data Ingestion ─────────────────────────────────────────────────────────────
.PHONY: download-puzzles test-cache

download-puzzles:
	$(PYTHON) datasets/download_lichess_puzzles.py \
	          --count $(PUZZLE_CNT) \
	          --output datasets/puzzles.jsonl

test-cache:
	$(PYTHON) src/env/cache.py --test-zobrist

# ── Pre-Training Baselines ─────────────────────────────────────────────────────
.PHONY: eval-tier1-baseline eval-tier2-baseline

eval-tier1-baseline:
	$(PYTHON) src/evals/run_pgn2fen.py \
	          --model-path $(MODEL) \
	          --samples $(SAMPLES)

eval-tier2-baseline:
	$(PYTHON) src/evals/run_puzzles.py \
	          --model-path $(MODEL) \
	          --limit $(LIMIT)

# ── RLVR Training ──────────────────────────────────────────────────────────────
.PHONY: train

train:
	$(PYTHON) src/models/grpo_trainer.py \
	          --base-model $(MODEL) \
	          --group-size 4 \
	          --max-cot-tokens 80 \
	          --learning-rate 5e-6 \
	          --episodes 5000 \
	          --stockfish-threads 4

# ── Post-Training Evaluations ──────────────────────────────────────────────────
.PHONY: eval-tier1 eval-tier2 eval-tier3 evaluate-all

eval-tier1:
	$(PYTHON) src/evals/run_pgn2fen.py \
	          --model-path $(CHECKPOINT) \
	          --samples $(SAMPLES)

eval-tier2:
	$(PYTHON) src/evals/run_puzzles.py \
	          --model-path $(CHECKPOINT) \
	          --limit $(LIMIT)

eval-tier3:
	$(PYTHON) src/evals/run_maia_match.py \
	          --checkpoint $(CHECKPOINT) \
	          --opponent $(OPPONENT) \
	          --games $(GAMES)

evaluate-all: eval-tier1 eval-tier2 eval-tier3

# ── Tests ──────────────────────────────────────────────────────────────────────
.PHONY: test

test:
	$(PYTHON) -m pytest tests/ -v

# ── Full Pipeline ──────────────────────────────────────────────────────────────
.PHONY: pipeline

pipeline: install download-puzzles test-cache \
          eval-tier1-baseline eval-tier2-baseline \
          train evaluate-all

# ── Utilities ──────────────────────────────────────────────────────────────────
clean:
	rm -rf .venv __pycache__ **/__pycache__ *.pyc
	rm -rf checkpoints/grpo_chess_step_*
