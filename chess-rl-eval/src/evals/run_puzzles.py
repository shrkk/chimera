"""
Tier 2 Evaluation: Lichess Tactical Puzzles — Pass@1
=====================================================
Loads stratified Lichess puzzles from ``datasets/puzzles.jsonl``, encodes
each position using the dual-rep prompt, runs LLM + neuro-symbolic verifier,
and reports Pass@1 per Elo band.

Pass gates:
  • ≥ 65 % on 1100–1400 Elo band
  • ≥ 40 % on 1400–1700 Elo band
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import chess
from tqdm import tqdm

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.state_encoder import StateEncoder, parse_candidates       # noqa: E402
from src.verifier.tactical_verifier import select_best_candidate       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Elo bands matching the stratified download ────────────────────────────────
ELO_BANDS = [
    ("800_1100",  800,  1100),
    ("1100_1400", 1100, 1400),
    ("1400_1700", 1400, 1700),
]
PASS_GATE = {
    "1100_1400": 0.65,
    "1400_1700": 0.40,
}

DEFAULT_PUZZLE_FILE = "datasets/puzzles.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _elo_band(rating: int) -> str:
    for name, lo, hi in ELO_BANDS:
        if lo <= rating < hi:
            return name
    return "other"


def load_puzzles(path: Path, limit: int) -> list[dict]:
    """Load up to *limit* puzzles from the JSONL file."""
    puzzles: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            puzzles.append(json.loads(line))
            if len(puzzles) >= limit:
                break
    logger.info("Loaded %d puzzles from %s", len(puzzles), path)
    return puzzles


def load_pipeline(model_path: str, device: str = "auto"):
    """Load a HuggingFace text-generation pipeline, supporting base models and LoRA adapters."""
    import torch
    from transformers import AutoTokenizer, pipeline  # type: ignore

    if device == "auto" or not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
    model_dir = Path(model_path)
    if (model_dir / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM
        logger.info("Detected LoRA adapter checkpoint at %s. Loading PEFT model on %s …", model_path, device)
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            do_sample=False,
            truncation=True,
        )
    else:
        logger.info("Loading model %s on %s (dtype=auto) …", model_path, device)
        pipe = pipeline(
            "text-generation",
            model=model_path,
            device=device,
            torch_dtype="auto",
            max_new_tokens=300,
            do_sample=False,
            truncation=True,
        )
    logger.info("Model loaded.")
    return pipe


def run_inference(pipe, prompt: str) -> str:
    """Run inference and return only new tokens."""
    out = pipe(prompt, return_full_text=False)
    return out[0]["generated_text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_puzzles(args: argparse.Namespace) -> bool:
    puzzle_file = Path(getattr(args, "puzzle_file", DEFAULT_PUZZLE_FILE))
    if not puzzle_file.exists():
        logger.error("Puzzle file not found: %s", puzzle_file)
        logger.error("Run: python datasets/download_lichess_puzzles.py first.")
        return False

    puzzles  = load_puzzles(puzzle_file, args.limit)
    encoder  = StateEncoder()
    pipe     = load_pipeline(args.model_path, getattr(args, "device", "cpu"))

    band_results: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for puzzle in tqdm(puzzles, desc="Solving puzzles"):
        fen         = puzzle.get("fen", "")
        sol_moves   = puzzle.get("moves", [])   # first move is the puzzle's first move
        rating      = int(puzzle.get("rating", 0))
        band        = _elo_band(rating)

        if not fen or not sol_moves:
            continue

        try:
            board = chess.Board(fen)
        except ValueError:
            logger.warning("Invalid FEN skipped: %s", fen)
            continue

        # The puzzle already starts after the opponent's move, so the first
        # element of sol_moves is the correct reply we want to predict.
        correct_uci = sol_moves[0]

        # Build dual-rep prompt
        prompt = encoder.encode(board, last_moves=[])

        # LLM inference
        try:
            raw        = run_inference(pipe, prompt)
            candidates = parse_candidates(raw)
        except Exception as exc:
            logger.warning("Inference error: %s", exc)
            candidates = []

        # Neuro-symbolic verifier selects best legal candidate
        try:
            best_move = select_best_candidate(board, candidates)
            pred_uci  = best_move.uci()
        except Exception as exc:
            logger.warning("Verifier error: %s", exc)
            pred_uci = ""

        band_results[band]["total"]   += 1
        band_results[band]["correct"] += int(pred_uci == correct_uci)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Tier 2 – Lichess Tactical Puzzle Pass@1':^60}")
    print("=" * 60)
    print(f"{'Band':<14} {'Correct':<10} {'Total':<10} {'Pass@1':<10} {'Gate'}")
    print("-" * 60)

    passes_gate = True
    for name, lo, hi in ELO_BANDS:
        r = band_results[name]
        if r["total"] == 0:
            print(f"{name:<14} {'N/A':<10} {'0':<10} {'—':<10}")
            continue
        acc = r["correct"] / r["total"]
        gate_val = PASS_GATE.get(name)
        gate_str = ""
        if gate_val is not None:
            ok = acc >= gate_val
            gate_str = f"{'✓ PASS' if ok else '✗ FAIL'} (≥{gate_val:.0%})"
            if not ok:
                passes_gate = False
        print(f"{name:<14} {r['correct']:<10} {r['total']:<10} {acc:.1%}     {gate_str}")

    print("=" * 60)
    print(f"Overall gate: {'PASS ✓' if passes_gate else 'FAIL ✗'}\n")
    return passes_gate


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier 2: Lichess tactical puzzle Pass@1 evaluation"
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--limit",       type=int, default=200,
                        help="Max puzzles to evaluate (default 200)")
    parser.add_argument("--puzzle-file", default=DEFAULT_PUZZLE_FILE,
                        help="Path to puzzles JSONL file")
    parser.add_argument("--config",  default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--device",  default="cpu", help="Inference device: cpu / cuda / mps")
    args = parser.parse_args()

    passed = evaluate_puzzles(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
