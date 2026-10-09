"""
Tier 1 Evaluation: PGN → FEN State Tracking
============================================
Measures how accurately the model can reproduce the exact FEN string at
specific half-move plies from a sequence of UCI moves.

Pass gate: ≥ 95 % exact-match accuracy at plies ≤ 30.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
from tqdm import tqdm

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pgn2fen_loader import PGN2FENLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Default target plies from spec ───────────────────────────────────────────
DEFAULT_PLIES = [10, 20, 30, 40]
PASS_THRESHOLD = 0.95   # spec: ≥ 95 % at ply ≤ 30
PASS_PLY_LIMIT = 30


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(uci_moves: list[str], ply: int) -> str:
    """Return a zero-shot prompt asking the model to output the FEN at *ply*."""
    moves_str = " ".join(uci_moves[:ply])
    return (
        f"Given the following sequence of moves from the starting position, "
        f"output ONLY the exact FEN string at ply {ply}. "
        f"Do not include any explanation.\n\n"
        f"Moves (UCI): {moves_str}\n\n"
        f"FEN at ply {ply}:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic game generator (fallback when no external dataset is available)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_games(n: int = 100) -> list[dict]:
    """
    Generate short random games using python-chess as a fallback dataset.
    Returns a list of dicts: {moves: [uci, ...], fens: {ply: fen}}.
    """
    import random
    games: list[dict] = []
    rng = random.Random(42)

    for _ in range(n):
        board = chess.Board()
        moves: list[str] = []
        fens: dict[int, str] = {}

        for ply_idx in range(1, 45):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            move = rng.choice(legal)
            board.push(move)
            moves.append(move.uci())
            fens[ply_idx] = board.fen()

        if moves:
            games.append({"moves": moves, "fens": fens})

    return games


# ─────────────────────────────────────────────────────────────────────────────
# LLM inference (lazy import so CPU-only runs don't require GPU libs)
# ─────────────────────────────────────────────────────────────────────────────

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
            max_new_tokens=120,
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
            max_new_tokens=120,
            do_sample=False,
            truncation=True,
        )
    logger.info("Model loaded.")
    return pipe


def run_inference(pipe, prompt: str) -> str:
    """Run a single prompt and return the generated text (new tokens only)."""
    out = pipe(prompt, return_full_text=False)
    return out[0]["generated_text"].strip()


def extract_fen_from_output(raw: str) -> str:
    """
    Extract the first token that looks like a FEN string from model output.
    FEN strings contain '/' characters and rank/file notation.
    Falls back to the full stripped string if no clear FEN is found.
    """
    for token in raw.split():
        if "/" in token and len(token) > 20:
            return token.strip()
    # Try splitting on newlines and grabbing the first non-empty line
    for line in raw.splitlines():
        line = line.strip()
        if "/" in line and len(line) > 20:
            return line
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation logic
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pgn2fen(args: argparse.Namespace) -> bool:
    """
    Run Tier 1 evaluation and return True if pass gate is met.
    """
    target_plies: list[int] = DEFAULT_PLIES

    # ── Build / load dataset ──────────────────────────────────────────────
    loader = PGN2FENLoader()

    # Try loading from a local PGN file if provided, else use synthetic games
    pgn_path = Path(args.pgn_file) if hasattr(args, "pgn_file") and args.pgn_file else None
    games_data: list[dict] = []

    if pgn_path and pgn_path.exists():
        logger.info("Loading PGN games from %s …", pgn_path)
        pgn_strings = loader.load_pgn_file(pgn_path)
        for pgn_str in pgn_strings[: args.samples]:
            batch = loader.batch_evaluate([pgn_str], target_plies)
            # Reconstruct moves for prompt
            board = chess.Board()
            import io
            game = chess.pgn.read_game(io.StringIO(pgn_str))
            if game is None:
                continue
            uci_moves = [m.uci() for m in game.mainline_moves()]
            fens = {p: batch[p][0] for p in target_plies if batch.get(p)}
            games_data.append({"moves": uci_moves, "fens": fens})
    else:
        logger.info("No PGN file found — generating %d synthetic games.", args.samples)
        games_data = _generate_synthetic_games(args.samples)

    if not games_data:
        logger.error("No games loaded — aborting.")
        return False

    # ── Load model ────────────────────────────────────────────────────────
    pipe = load_pipeline(args.model_path, getattr(args, "device", "cpu"))

    # ── Evaluate ──────────────────────────────────────────────────────────
    results: dict[int, dict] = {p: {"correct": 0, "total": 0} for p in target_plies}

    for game in tqdm(games_data, desc="Evaluating games"):
        moves = game["moves"]
        fens  = game["fens"]   # {ply: true_fen}

        for ply in target_plies:
            if ply not in fens or ply > len(moves):
                continue
            true_fen = fens[ply]
            prompt   = build_prompt(moves, ply)

            try:
                raw      = run_inference(pipe, prompt)
                pred_fen = extract_fen_from_output(raw)
            except Exception as exc:
                logger.warning("Inference failed at ply %d: %s", ply, exc)
                pred_fen = ""

            results[ply]["total"]  += 1
            results[ply]["correct"] += int(pred_fen == true_fen)

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 54)
    print(f"{'Tier 1 – PGN2FEN State Tracking':^54}")
    print("=" * 54)
    print(f"{'Ply':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Gate'}")
    print("-" * 54)

    passes_gate = True
    for ply in target_plies:
        r = results[ply]
        if r["total"] == 0:
            print(f"{ply:<8} {'N/A':<10} {'0':<10} {'—':<12}")
            continue
        acc = r["correct"] / r["total"]
        gate_str = ""
        if ply <= PASS_PLY_LIMIT:
            gate_str = "✓ PASS" if acc >= PASS_THRESHOLD else "✗ FAIL"
            if acc < PASS_THRESHOLD:
                passes_gate = False
        print(f"{ply:<8} {r['correct']:<10} {r['total']:<10} {acc:.1%}       {gate_str}")

    print("=" * 54)
    status = "PASS ✓" if passes_gate else "FAIL ✗"
    print(f"Overall gate (≥{PASS_THRESHOLD:.0%} at ply ≤ {PASS_PLY_LIMIT}): {status}\n")

    return passes_gate


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier 1: PGN → FEN reconstruction accuracy evaluation"
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--samples", type=int, default=100, help="Number of games to evaluate")
    parser.add_argument("--pgn-file", default=None, help="Optional path to a local .pgn file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu / cuda / mps")
    args = parser.parse_args()

    passed = evaluate_pgn2fen(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
