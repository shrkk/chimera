"""
Neuro-Symbolic Tactical Verifier
=================================
Implements the exact spec logic for 2-ply tactical safety evaluation,
Static Exchange Evaluation (SEE), and quiescence search.
"""

import chess
from typing import Optional, List, Set

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


# ─────────────────────────────────────────────────────────────────────────────
# Spec-exact implementation (Section 4) with robustness fixes
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tactical_safety(board: chess.Board, move: chess.Move) -> float:
    """
    Evaluates 2-ply tactical stability without launching deep external engines.
    1. Detects immediate self-check/hanging piece blunders.
    2. Runs a 2-ply capture exchange (SEE / Quiescence).
    """
    board.push(move)
    score = 0.0

    # 1. Immediate checkmate delivery
    if board.is_checkmate():
        board.pop()
        return 100000.0

    # 2. Check if moved piece was hung to a lower-value attacker
    to_sq = move.to_square
    opponents = board.attackers(board.turn, to_sq)
    defenders = board.attackers(not board.turn, to_sq)

    moved_piece = board.piece_at(to_sq)
    piece_val = PIECE_VALUES.get(moved_piece.piece_type, 0) if moved_piece else 0

    if opponents and not defenders:
        score -= piece_val  # Left completely hanging
    elif opponents and defenders:
        min_opp_val = min(
            PIECE_VALUES.get(board.piece_at(sq).piece_type, 0)
            for sq in opponents
            if board.piece_at(sq)
        )
        if min_opp_val < piece_val:
            score -= (piece_val - min_opp_val)

    # 3. Detect free material hanging for the opponent on reply
    for legal_reply in list(board.legal_moves):
        if board.is_en_passant(legal_reply) or board.is_capture(legal_reply):
            victim = board.piece_at(legal_reply.to_square)
            if victim and victim.piece_type == chess.QUEEN and not board.is_check():
                # Potential massive blunder overlooked
                score -= 400.0
                break

    board.pop()
    return score


def select_best_candidate(board: chess.Board, candidate_ucis: List[str]) -> chess.Move:
    """Ranks LLM-suggested candidate moves via symbolic verification."""
    if not board.legal_moves:
        return chess.Move.null()

    valid_candidates: List[chess.Move] = []
    seen_ucis: Set[str] = set()

    for uci in candidate_ucis:
        if not uci or uci in seen_ucis:
            continue
        try:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                valid_candidates.append(m)
                seen_ucis.add(uci)
        except ValueError:
            continue

    if not valid_candidates:
        # Fallback: return the first legal move
        return next(iter(board.legal_moves))

    # Score candidates: Prioritize LLM rank unless symbolic check flags a blunder
    best_move = valid_candidates[0]
    best_score = evaluate_tactical_safety(board, best_move)

    for idx, move in enumerate(valid_candidates[1:], start=1):
        raw_score = evaluate_tactical_safety(board, move)
        # Apply prior rank penalty
        score = raw_score - (idx * 15.0)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# ─────────────────────────────────────────────────────────────────────────────
# Extended: SEE + Quiescence Search
# ─────────────────────────────────────────────────────────────────────────────

def _material_balance(board: chess.Board) -> float:
    """Simple centipawn material balance from the current player's perspective."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == board.turn:
            score += val
        else:
            score -= val
    return float(score)


def quiescence_search(
    board: chess.Board,
    depth: int = 2,
    alpha: float = -float("inf"),
    beta: float = float("inf"),
) -> float:
    """
    Alpha-beta quiescence search that only considers captures and promotions,
    returning a centipawn score from the current player's perspective.
    """
    stand_pat = _material_balance(board)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    if depth == 0 or board.is_game_over():
        return stand_pat

    for move in list(board.legal_moves):
        if not (board.is_capture(move) or move.promotion):
            continue
        board.push(move)
        score = -quiescence_search(board, depth - 1, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def static_exchange_evaluation(board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation (SEE): estimates the material won/lost
    after a sequence of captures on the destination square.
    Returns a centipawn value (positive = winning capture).
    """
    if not board.is_capture(move):
        return 0

    to_sq = move.to_square

    # Value of the piece being captured
    if board.is_en_passant(move):
        captured_val = PIECE_VALUES[chess.PAWN]
    else:
        victim = board.piece_at(to_sq)
        captured_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0

    # Value of the attacking piece
    attacker = board.piece_at(move.from_square)
    attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0

    # Simulate the exchange
    board.push(move)
    best_opp_recapture = 0
    for reply in list(board.legal_moves):
        if board.is_capture(reply) and reply.to_square == to_sq:
            board.push(reply)
            recapture_gain = attacker_val - static_exchange_evaluation(board, reply)
            board.pop()
            best_opp_recapture = max(best_opp_recapture, recapture_gain)

    board.pop()
    return captured_val - best_opp_recapture


def is_tactical_blunder(
    board: chess.Board,
    move: chess.Move,
    threshold: float = -100.0,
) -> bool:
    """Returns True if the move loses material beyond *threshold* centipawns (per SEE)."""
    return static_exchange_evaluation(board, move) < threshold
