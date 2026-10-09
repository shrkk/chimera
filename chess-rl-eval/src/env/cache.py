import aiosqlite
import asyncio
import chess
import subprocess
import datetime
import math
import pathlib
import argparse
import json
from typing import Dict, Optional, Union

class StockfishCache:
    def __init__(
        self,
        db_path: Union[str, pathlib.Path],
        stockfish_binary: str = "stockfish",
        depth: int = 10,
        max_entries: int = 100000,
    ):
        self.db_path = str(db_path)
        self.stockfish_binary = stockfish_binary
        self.depth = depth
        self.max_entries = max_entries
        self._db: Optional[aiosqlite.Connection] = None
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = asyncio.Lock()

    async def _init_db(self):
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._db.execute("PRAGMA journal_mode=WAL;")
            await self._db.execute("PRAGMA synchronous=NORMAL;")
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS evals (
                    epd TEXT PRIMARY KEY,
                    score_cp INTEGER,
                    mate_in INTEGER,
                    win_prob REAL,
                    created_at TEXT
                )
            """)
            await self._db.commit()

    @staticmethod
    def win_prob_from_cp(cp: float) -> float:
        """Squashed Win Probability Potential Function: 2 / (1 + 10^(-cp/400)) - 1 in [-1, 1]"""
        return 2.0 / (1.0 + 10.0 ** (-cp / 400.0)) - 1.0

    @staticmethod
    def win_prob_from_mate(mate_in: int) -> float:
        """
        Win probability for mate scores.
        If mate_in == 0, current side to move is checkmated -> -1.0.
        Otherwise sign(N) * (1.0 - 0.01 * |N|).
        """
        if mate_in == 0:
            return -1.0
        sign = 1.0 if mate_in > 0 else -1.0
        return sign * (1.0 - 0.01 * min(abs(mate_in), 99))

    def _ensure_process(self) -> subprocess.Popen:
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                [self.stockfish_binary],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )
            # Send initial UCI commands
            self._process.stdin.write("uci\nisready\n")
            self._process.stdin.flush()
            while True:
                line = self._process.stdout.readline()
                if not line or "readyok" in line:
                    break
        return self._process

    def _query_stockfish_sync(self, fen: str) -> Dict[str, Union[int, float, None]]:
        process = self._ensure_process()
        process.stdin.write(f"position fen {fen}\ngo depth {self.depth}\n")
        process.stdin.flush()

        score_cp: Optional[int] = None
        mate_in: Optional[int] = None

        while True:
            line = process.stdout.readline()
            if not line or line.startswith("bestmove"):
                break
            if "info depth" in line and "score cp" in line:
                parts = line.split()
                try:
                    cp_idx = parts.index("cp")
                    score_cp = int(parts[cp_idx + 1])
                except (ValueError, IndexError):
                    pass
            elif "info depth" in line and "score mate" in line:
                parts = line.split()
                try:
                    mate_idx = parts.index("mate")
                    mate_in = int(parts[mate_idx + 1])
                except (ValueError, IndexError):
                    pass

        if mate_in is not None:
            win_prob = self.win_prob_from_mate(mate_in)
            score_cp = None
        elif score_cp is not None:
            win_prob = self.win_prob_from_cp(score_cp)
        else:
            win_prob = 0.0

        return {"score_cp": score_cp, "mate_in": mate_in, "win_prob": win_prob}

    async def get_or_compute(self, board: chess.Board) -> Dict[str, Union[int, float, None]]:
        # Fast path: check if board is already in checkmate
        if board.is_checkmate():
            return {"score_cp": None, "mate_in": 0, "win_prob": -1.0}

        await self._init_db()
        epd = board.epd()

        # Query database cache
        async with self._db.execute(
            "SELECT score_cp, mate_in, win_prob FROM evals WHERE epd = ?", (epd,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {"score_cp": row[0], "mate_in": row[1], "win_prob": row[2]}

        # Compute with persistent engine under lock
        async with self._process_lock:
            result = await asyncio.to_thread(self._query_stockfish_sync, board.fen())

        # Save to database
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO evals (epd, score_cp, mate_in, win_prob, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (epd, result["score_cp"], result["mate_in"], result["win_prob"], now),
        )
        await self._db.commit()

        return result

    async def close(self):
        if self._db is not None:
            await self._db.close()
            self._db = None
        if self._process is not None:
            try:
                self._process.stdin.write("quit\n")
                self._process.stdin.flush()
                self._process.terminate()
            except Exception:
                pass
            self._process = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-zobrist", action="store_true")
    args = parser.parse_args()

    if args.test_zobrist:
        async def run_test():
            cache = StockfishCache("test_cache.db", "stockfish", 10)
            await cache._init_db()
            board = chess.Board()
            res = await cache.get_or_compute(board)
            print(json.dumps(res, indent=2))
            await cache.close()
        asyncio.run(run_test())
