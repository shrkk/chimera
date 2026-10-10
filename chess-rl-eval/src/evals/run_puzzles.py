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


def load_eval_model(model_path: str, device: str = "auto"):
    """Load model directly on device supporting base models and LoRA adapters."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device == "auto" or not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
    model_dir = Path(model_path)

    if not model_dir.exists() and ("checkpoints" in model_path or model_path.startswith(".") or model_path.startswith("/")):
        chk_dir = Path("checkpoints")
        avail = sorted([p.name for p in chk_dir.glob("grpo_chess_step_*")]) if chk_dir.exists() else []
        avail_msg = f"Available checkpoints in ./checkpoints: {avail}" if avail else "No checkpoints found in ./checkpoints."
        raise FileNotFoundError(
            f"\n[Chimera Checkpoint Error] Specified path '{model_path}' does not exist on disk!\n"
            f"{avail_msg}\n"
            f"Hint: Make sure you are inside the project directory (e.g. 'cd /chimera/chess-rl-eval') "
            f"and specify an existing step checkpoint (e.g. '--model-path checkpoints/grpo_chess_step_500')."
        )

    if (model_dir / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM
        logger.info("Detected LoRA adapter checkpoint at %s. Loading PEFT model on %s (dtype=%s) …", model_path, device, dtype)
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        logger.info("Loading model %s on %s (dtype=%s) …", model_path, device, dtype)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    logger.info("Model loaded successfully on %s.", device)
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_puzzles(args: argparse.Namespace) -> bool:
    import torch
    puzzle_file = Path(getattr(args, "puzzle_file", DEFAULT_PUZZLE_FILE))
    if not puzzle_file.exists():
        logger.error("Puzzle file not found: %s", puzzle_file)
        logger.error("Run: python datasets/download_lichess_puzzles.py first.")
        return False

    puzzles   = load_puzzles(puzzle_file, args.limit)
    encoder   = StateEncoder()
    model, tokenizer, device = load_eval_model(args.model_path, getattr(args, "device", "auto"))
    batch_size = getattr(args, "batch_size", 16)

    band_results: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    # Filter valid puzzles
    valid_puzzles: list[tuple[dict, chess.Board]] = []
    for puzzle in puzzles:
        fen       = puzzle.get("fen", "")
        sol_moves = puzzle.get("moves", [])
        if not fen or not sol_moves:
            continue
        try:
            board = chess.Board(fen)
            valid_puzzles.append((puzzle, board))
        except ValueError:
            logger.warning("Invalid FEN skipped: %s", fen)
            continue

    # Batched inference across GPU
    for i in tqdm(range(0, len(valid_puzzles), batch_size), desc=f"Solving puzzles (batch_size={batch_size})"):
        chunk = valid_puzzles[i : i + batch_size]
        prompts = [encoder.encode(board, last_moves=[]) for _, board in chunk]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for (puzzle, board), raw in zip(chunk, completions):
            sol_moves   = puzzle.get("moves", [])
            rating      = int(puzzle.get("rating", 0))
            band        = _elo_band(rating)
            correct_uci = sol_moves[0]

            candidates = parse_candidates(raw.strip())
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

    out_file = getattr(args, "output", None)
    if out_file:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_path": args.model_path,
            "bands": {
                name: {
                    "correct": band_results[name]["correct"],
                    "total": band_results[name]["total"],
                    "pass_at_1": (
                        band_results[name]["correct"] / band_results[name]["total"]
                        if band_results[name]["total"] > 0
                        else 0.0
                    ),
                }
                for name, _, _ in ELO_BANDS
            },
            "passes_gate": passes_gate,
        }
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved evaluation results to %s", out_path)

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
    parser.add_argument("--batch-size",  type=int, default=16,
                        help="Batch size for parallel GPU inference (default 16)")
    parser.add_argument("--puzzle-file", default=DEFAULT_PUZZLE_FILE,
                        help="Path to puzzles JSONL file")
    parser.add_argument("--config",  default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--device",  default="auto", help="Inference device: auto / cpu / cuda / mps")
    parser.add_argument("--output",  default="results/puzzles_eval.json", help="Path to save JSON results")
    args = parser.parse_args()

    passed = evaluate_puzzles(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
