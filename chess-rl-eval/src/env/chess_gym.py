import chess
import asyncio
import yaml
import pathlib
from typing import Dict, Optional, Tuple, Any, List
from src.env.cache import StockfishCache


def _run_async(coro):
    """
    Run an async coroutine from synchronous code safely.
    Uses a fresh event loop in a worker thread if the current thread
    already has a running event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class ChessGym:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache: Optional[StockfishCache] = None,
        stockfish_threads: int = 1,
    ):
        self.config = config or {}
        if cache is None:
            db_path = pathlib.Path(
                self.config.get("cache", {}).get("db_path", "./cache/stockfish_cache.db")
            )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            sf_bin = self.config.get("cache", {}).get("stockfish_binary", "stockfish")
            depth = self.config.get("grpo", {}).get("stockfish_depth", 10)
            cache = StockfishCache(db_path, sf_bin, depth)
            _run_async(cache._init_db())
        self.cache = cache

        self.board = chess.Board()
        self.history: set = set()
        self.move_count = 0
        self.puzzle_solution: List[str] = []
        self.last_moves: List[str] = []
        self.gamma = self.config.get("reward", {}).get("gamma", 0.99)
        self.max_steps = self.config.get("env", {}).get("max_steps", 100)

    def reset(
        self,
        fen: Optional[str] = None,
        puzzle_solution: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()

        self.history = {self.board.epd()}
        self.move_count = 0
        self.last_moves = []
        self.puzzle_solution = puzzle_solution or []

        return self._obs()

    async def _get_phi(self, board: chess.Board) -> float:
        eval_dict = await self.cache.get_or_compute(board)
        return eval_dict["win_prob"]

    def _pbrs_reward(self, prev_board: chess.Board, next_board: chess.Board) -> float:
        """
        Potential-Based Reward Shaping (PBRS):
            F_PBRS(s_t, s_{t+1}) = γ · Φ(s_{t+1}) − Φ(s_t)
        Note: Stockfish returns win probability relative to the side to move.
        In prev_board, it is the acting player's turn to move -> phi_prev.
        In next_board, it is the opponent's turn to move -> -phi_next
        from the acting player's perspective.
        """
        raw_phi_next = _run_async(self._get_phi(next_board))
        phi_prev = _run_async(self._get_phi(prev_board))
        phi_next_actor = -raw_phi_next
        return self.gamma * phi_next_actor - phi_prev

    def _step_penalty(self) -> float:
        return self.config.get("reward", {}).get("step_penalty", -0.005)

    def _repetition_penalty(self, board: chess.Board) -> float:
        if board.epd() in self.history:
            return self.config.get("reward", {}).get("repetition_penalty", -0.30)
        return 0.0

    def _token_len_penalty(self, n_tokens: int) -> float:
        coef = self.config.get("reward", {}).get("token_penalty_coef", -0.002)
        max_toks = self.config.get("model", {}).get("max_cot_tokens", 80)
        return coef * max(0, n_tokens - max_toks)

    def step(
        self, move_uci: str, n_tokens: int = 0
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        prev_board = self.board.copy()

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return (
                self._obs(),
                self.config.get("reward", {}).get("terminal_illegal", -1.5),
                True,
                {"error": "invalid UCI"},
            )

        if move not in self.board.legal_moves:
            return (
                self._obs(),
                self.config.get("reward", {}).get("terminal_illegal", -1.5),
                True,
                {"error": "illegal move"},
            )

        # Apply agent's move
        self.board.push(move)
        self.last_moves.append(move_uci)
        self.move_count += 1

        r_pbrs = self._pbrs_reward(prev_board, self.board)
        r_step = self._step_penalty()
        r_rep = self._repetition_penalty(self.board)
        r_tok = self._token_len_penalty(n_tokens)
        self.history.add(self.board.epd())

        done = False
        r_term = 0.0

        # Puzzle-specific check:
        if self.puzzle_solution:
            agent_step_idx = len(self.last_moves) - 1
            if (
                agent_step_idx >= len(self.puzzle_solution)
                or move_uci != self.puzzle_solution[agent_step_idx]
            ):
                # Deviated from puzzle solution
                r_term = self.config.get("reward", {}).get("terminal_loss", -1.0)
                done = True
            elif len(self.last_moves) == len(self.puzzle_solution):
                # Puzzle completely solved!
                r_term = self.config.get("reward", {}).get("terminal_win", 2.0)
                done = True
            else:
                # Puzzle continues: auto-play the opponent's reply
                opp_move_uci = self.puzzle_solution[len(self.last_moves)]
                opp_move = chess.Move.from_uci(opp_move_uci)
                if opp_move in self.board.legal_moves:
                    self.board.push(opp_move)
                    self.last_moves.append(opp_move_uci)
                    self.history.add(self.board.fen())
                    # Check if puzzle ended after opponent reply
                    if len(self.last_moves) == len(self.puzzle_solution):
                        r_term = self.config.get("reward", {}).get("terminal_win", 2.0)
                        done = True
                else:
                    # Opponent move illegal (corrupted puzzle data)
                    done = True
        else:
            # Standard game terminal checks:
            if self.board.is_checkmate():
                r_term = self.config.get("reward", {}).get("terminal_win", 2.0)
                done = True
            elif self.board.is_game_over():
                # Draw or stalemate
                r_term = self.config.get("reward", {}).get("terminal_loss", -1.0)
                done = True
            elif self.move_count >= self.max_steps:
                r_term = self.config.get("reward", {}).get("terminal_loss", -1.0)
                done = True

        total_reward = r_pbrs + r_step + r_rep + r_tok + r_term

        return (
            self._obs(),
            total_reward,
            done,
            {
                "r_pbrs": r_pbrs,
                "r_step": r_step,
                "r_rep": r_rep,
                "r_tok": r_tok,
                "r_term": r_term if done else None,
            },
        )

    def _obs(self) -> Dict[str, Any]:
        return {
            "board": self.board,
            "fen": self.board.fen(),
            "legal_moves": [m.uci() for m in self.board.legal_moves],
            "last_moves": list(self.last_moves),
            "turn": "White" if self.board.turn == chess.WHITE else "Black",
            "halfmove_clock": self.board.halfmove_clock,
        }

    def close(self):
        if self.cache is not None:
            _run_async(self.cache.close())
