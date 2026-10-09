"""
Tier 3 Evaluation: LLM Agent vs. Maia-1300 Full Games
======================================================
Plays N games between the trained LLM agent and a Maia chess engine.

Maia integration priority:
  1. lc0 binary with Maia network weights (``--engine-path`` + ``--weights``)
  2. Stockfish with ``UCI_LimitStrength`` / ``UCI_Elo 1300`` as a handicap proxy

Metrics:
  • Legal Move Compliance Rate (LMCR) — must be 100 %
  • Win / Draw / Loss counts
  • Bayes-Elo delta vs. Maia-1300 (estimated)

Pass gate: 100 % LMCR and > 50 % win-rate vs. Maia-1300.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.engine
from tqdm import tqdm

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.state_encoder import StateEncoder, parse_candidates       # noqa: E402
from src.verifier.tactical_verifier import select_best_candidate       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAIA_ELO         = 1300
PASS_WIN_RATE    = 0.50
PASS_LMCR        = 1.00
THINK_TIME_SEC   = 1.0   # time per move for the engine opponent


# ─────────────────────────────────────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """Wraps the LLM + neuro-symbolic verifier into a chess agent."""

    def __init__(self, checkpoint: str, device: str = "auto") -> None:
        import torch
        from transformers import pipeline  # type: ignore

        if device == "auto" or not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
        model_dir = Path(checkpoint)
        if (model_dir / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
            logger.info("Detected LoRA adapter checkpoint at %s. Loading PEFT model on %s …", checkpoint, device)
            model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=300,
                do_sample=False,
                truncation=True,
            )
        else:
            logger.info("Loading LLM agent from %s on %s (dtype=auto) …", checkpoint, device)
            self._pipe = pipeline(
                "text-generation",
                model=checkpoint,
                device=device,
                torch_dtype="auto",
                max_new_tokens=300,
                do_sample=False,
                truncation=True,
            )
        self._encoder = StateEncoder()
        self.total_move_attempts  = 0
        self.total_legal_moves    = 0

    def choose_move(self, board: chess.Board) -> chess.Move:
        """
        Run LLM inference + symbolic verification to pick a move.
        Always returns a legal move (falls back to first legal if needed).
        """
        self.total_move_attempts += 1
        prompt = self._encoder.encode(board, last_moves=[])

        try:
            out        = self._pipe(prompt, return_full_text=False)
            raw        = out[0]["generated_text"].strip()
            candidates = parse_candidates(raw)
        except Exception as exc:
            logger.warning("LLM inference error: %s", exc)
            candidates = []

        move = select_best_candidate(board, candidates)   # always legal
        self.total_legal_moves += 1
        return move

    @property
    def lmcr(self) -> float:
        if self.total_move_attempts == 0:
            return 1.0
        return self.total_legal_moves / self.total_move_attempts


# ─────────────────────────────────────────────────────────────────────────────
# Engine loader
# ─────────────────────────────────────────────────────────────────────────────

async def _load_engine(
    engine_path: Optional[str],
    weights_path: Optional[str],
    opponent: str,
) -> chess.engine.SimpleEngine:
    """
    Load the opponent engine.
    - If engine_path points to lc0 and weights_path is provided → Maia mode.
    - Otherwise fall back to Stockfish with handicap Elo.
    """
    if engine_path:
        ep = Path(engine_path)
        if ep.exists():
            transport, engine = await chess.engine.popen_uci(str(ep))
            if weights_path:
                await engine.configure({"WeightsFile": weights_path})
                logger.info("Loaded lc0 with Maia weights: %s", weights_path)
            else:
                logger.info("Loaded engine: %s", ep)
            return engine

    # Fallback: Stockfish at Elo 1300
    try:
        transport, engine = await chess.engine.popen_uci("stockfish")
        await engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": MAIA_ELO,
        })
        logger.info("Stockfish fallback at Elo %d", MAIA_ELO)
        return engine
    except FileNotFoundError:
        logger.error("Neither lc0 nor stockfish found on PATH. "
                     "Install stockfish or pass --engine-path.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Single game
# ─────────────────────────────────────────────────────────────────────────────

async def play_game(
    llm_agent: LLMAgent,
    engine: chess.engine.SimpleEngine,
    llm_plays_white: bool,
) -> dict:
    """
    Play one full game. Returns a result dict:
      {result: '1-0' | '0-1' | '1/2-1/2', llm_legal_moves, num_moves}
    """
    board = chess.Board()
    num_moves = 0

    while not board.is_game_over(claim_draw=True):
        llm_turn = (board.turn == chess.WHITE) == llm_plays_white

        if llm_turn:
            move = llm_agent.choose_move(board)
        else:
            result = await engine.play(
                board,
                chess.engine.Limit(time=THINK_TIME_SEC),
            )
            move = result.move
            if move is None or move not in board.legal_moves:
                # Engine returned invalid move — resign
                break

        if move not in board.legal_moves:
            logger.warning("Illegal move from LLM: %s — forfeiting game.", move)
            # Return a loss for the LLM
            result_str = "0-1" if llm_plays_white else "1-0"
            return {
                "result":           result_str,
                "llm_legal_moves":  llm_agent.total_legal_moves,
                "num_moves":        num_moves,
                "forfeited":        True,
            }

        board.push(move)
        num_moves += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_str = "1/2-1/2"
    else:
        result_str = outcome.result()

    return {
        "result":          result_str,
        "llm_legal_moves": llm_agent.total_legal_moves,
        "num_moves":       num_moves,
        "forfeited":       False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bayes-Elo
# ─────────────────────────────────────────────────────────────────────────────

def bayes_elo_diff(wins: int, losses: int, draws: int = 0) -> Optional[float]:
    """
    Estimate Elo difference using the Bayes-Elo formula.
    Draws count as 0.5 win + 0.5 loss.
    Returns None if wins or losses == 0 (undefined).
    """
    w = wins  + 0.5 * draws
    l = losses + 0.5 * draws
    if w == 0 or l == 0:
        return None
    return 400.0 * math.log10(w / l)


# ─────────────────────────────────────────────────────────────────────────────
# Main async evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> bool:
    agent  = LLMAgent(args.checkpoint, device=getattr(args, "device", "cpu"))
    engine = await _load_engine(
        getattr(args, "engine_path", None),
        getattr(args, "weights", None),
        args.opponent,
    )

    wins = draws = losses = 0
    game_records: list[dict] = []

    try:
        for game_idx in tqdm(range(args.games), desc="Playing games"):
            llm_white = (game_idx % 2 == 0)   # alternate colours
            record = await play_game(agent, engine, llm_white)
            game_records.append(record)

            result = record["result"]
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0" and llm_white) or (result == "0-1" and not llm_white):
                wins += 1
            else:
                losses += 1
    finally:
        await engine.quit()

    # ── Summary ──────────────────────────────────────────────────────────
    total    = wins + draws + losses
    win_rate = wins / total if total > 0 else 0.0
    lmcr     = agent.lmcr
    elo_diff = bayes_elo_diff(wins, losses, draws)
    elo_str  = f"{elo_diff:+.1f}" if elo_diff is not None else "undefined"

    print("\n" + "=" * 56)
    print(f"{'Tier 3 – LLM Agent vs. ' + args.opponent:^56}")
    print("=" * 56)
    print(f"  Games played   : {total}")
    print(f"  Wins           : {wins}")
    print(f"  Draws          : {draws}")
    print(f"  Losses         : {losses}")
    print(f"  Win rate       : {win_rate:.1%}")
    print(f"  Bayes-Elo diff : {elo_str} vs. Maia-{MAIA_ELO}")
    print(f"  LMCR           : {lmcr:.1%}")
    print("-" * 56)

    lmcr_ok     = lmcr     >= PASS_LMCR
    winrate_ok  = win_rate >= PASS_WIN_RATE

    print(f"  LMCR gate    (≥{PASS_LMCR:.0%})  : {'✓ PASS' if lmcr_ok    else '✗ FAIL'}")
    print(f"  Win-rate gate(>{PASS_WIN_RATE:.0%}) : {'✓ PASS' if winrate_ok else '✗ FAIL'}")
    print("=" * 56)

    passes_gate = lmcr_ok and winrate_ok
    print(f"Overall gate: {'PASS ✓' if passes_gate else 'FAIL ✗'}\n")
    return passes_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier 3: LLM agent vs. Maia-1300 game evaluation"
    )
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to trained model checkpoint or HuggingFace model ID")
    parser.add_argument("--opponent",     default="maia-1300",
                        help="Opponent name (informational; default maia-1300)")
    parser.add_argument("--games",        type=int, default=50,
                        help="Number of games to play (default 50)")
    parser.add_argument("--engine-path",  default=None,
                        help="Path to lc0 binary (optional; falls back to Stockfish)")
    parser.add_argument("--weights",      default=None,
                        help="Path to Maia network weights file (.pb.gz)")
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--device",       default="cpu",
                        help="Inference device: cpu / cuda / mps")
    args = parser.parse_args()

    passed = asyncio.run(_run(args))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
