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

def _material_balance(board: chess.Board) -> float:
    """Evaluates material balance + center control + check penalties from current player's perspective."""
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = float(PIECE_VALUES.get(piece.piece_type, 0))
        # Center square influence
        if sq in [chess.D4, chess.E4, chess.D5, chess.E5]:
            val += 20.0
        elif sq in [chess.C3, chess.D3, chess.E3, chess.F3, chess.C6, chess.D6, chess.E6, chess.F6]:
            val += 8.0

        if piece.color == board.turn:
            score += val
        else:
            score -= val

    if board.is_check():
        score -= 40.0
    return score


def quiescence_search(
    board: chess.Board,
    depth: int = 2,
    alpha: float = -float("inf"),
    beta: float = float("inf"),
) -> float:
    """
    Alpha-beta quiescence search that only considers captures and promotions,
    resolving tactical trades to prevent the horizon effect.
    """
    stand_pat = _material_balance(board)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    if depth <= 0 or board.is_game_over():
        return stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    # Move ordering: MVV-LVA for captures
    def capture_prio(m: chess.Move) -> int:
        victim = board.piece_at(m.to_square)
        attacker = board.piece_at(m.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
        a_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
        return v_val * 10 - a_val

    captures.sort(key=capture_prio, reverse=True)

    for move in captures:
        board.push(move)
        score = -quiescence_search(board, depth - 1, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def minimax_search(
    board: chess.Board,
    depth: int = 3,
    alpha: float = -float("inf"),
    beta: float = float("inf"),
    qdepth: int = 2,
) -> float:
    """
    Alpha-beta minimax search with move ordering and quiescence extension.
    Evaluates 4 to 6 plies deep along tactical lines.
    Returns centipawn score from the perspective of the player to move.
    """
    if board.is_checkmate():
        return -100000.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0

    if depth <= 0:
        return quiescence_search(board, depth=qdepth, alpha=alpha, beta=beta)

    # Move ordering: Captures (MVV-LVA), promotions, checks, then quiet moves
    def move_priority(m: chess.Move) -> int:
        prio = 0
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            v_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            a_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            prio += 10000 + (v_val * 10 - a_val)
        if m.promotion:
            prio += 9000
        if board.gives_check(m):
            prio += 5000
        return prio

    ordered_moves = sorted(board.legal_moves, key=move_priority, reverse=True)

    for m in ordered_moves:
        board.push(m)
        score = -minimax_search(board, depth - 1, -beta, -alpha, qdepth=qdepth)
        board.pop()

        if score >= beta:
            return beta  # Beta cutoff
        if score > alpha:
            alpha = score

    return alpha


def evaluate_tactical_safety(board: chess.Board, move: chess.Move, depth: int = 3, qdepth: int = 2) -> float:
    """
    Evaluates 4-to-6 ply tactical safety and positional strength:
    Plays move, then computes 3 plies of minimax + up to 2 plies of quiescence trades.
    """
    if move not in board.legal_moves:
        return -100000.0

    board.push(move)
    if board.is_checkmate():
        board.pop()
        return 100000.0

    # 4 to 6 plies evaluation from current player's perspective
    score = -minimax_search(board, depth=depth, alpha=-100000.0, beta=100000.0, qdepth=qdepth)
    board.pop()
    return score


def select_best_candidate(board: chess.Board, candidate_ucis: List[str]) -> chess.Move:
    """
    Ranks LLM-suggested candidate moves via 4-to-6 ply symbolic alpha-beta verification.
    If the LLM's only candidate is an egregious tactical blunder (e.g. loses material 4 plies deep),
    searches legal moves for a safe, non-blundering alternative.
    """
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
        # Fallback: search best legal move with shallow depth
        legal = list(board.legal_moves)
        best_m = legal[0]
        best_s = -float("inf")
        for m in legal[:8]:  # Check up to 8 candidate moves
            s = evaluate_tactical_safety(board, m, depth=2, qdepth=1)
            if s > best_s:
                best_s = s
                best_m = m
        return best_m

    # Score LLM candidates at 4-to-6 plies
    best_move = valid_candidates[0]
    best_score = evaluate_tactical_safety(board, best_move, depth=3, qdepth=2)

    for idx, move in enumerate(valid_candidates[1:], start=1):
        score = evaluate_tactical_safety(board, move, depth=3, qdepth=2)
        # Apply prior rank penalty (15 cp per rank slot)
        adjusted_score = score - (idx * 15.0)
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_move = move

    # Safety guard: if the chosen candidate is a catastrophic blunder (<-180 cp)
    # search top legal moves to protect the agent from hanging queens/rooks
    if best_score < -180.0:
        for alt_m in list(board.legal_moves)[:10]:
            if alt_m != best_move:
                alt_score = evaluate_tactical_safety(board, alt_m, depth=2, qdepth=2)
                if alt_score > best_score + 150.0:
                    best_move = alt_m
                    best_score = alt_score
                    break

    return best_move


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
