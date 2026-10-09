import chess
import re
from typing import List, Tuple

PROMPT_TEMPLATE = """You are a tactical chess reasoning policy.
Analyze the position for tactical threats (pins, skewers, undefended pieces) and select up to 3 candidate UCI moves ranked by strength.

=== BOARD STATE ===
FEN: {fen}
Turn: {turn}
Half-move: {halfmove_clock}

=== PIECE LOCATIONS ===
White: {white_piece_coords}
Black: {black_piece_coords}

=== RECENT MOVES ===
{last_4_uci_moves}

=== LEGAL ACTION SPACE ===
[{legal_moves_csv}]

Format your response as follows:
<reasoning>
[Analysis of checks, captures, threats]
</reasoning>
<candidates>
1. <move>uci</move>
2. <move>uci</move>
3. <move>uci</move>
</candidates>

Now analyze the position:
<reasoning>"""

class StateEncoder:
    def _piece_coord_map(self, board: chess.Board) -> Tuple[str, str]:
        piece_symbols = {
            chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B',
            chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'
        }
        white_pieces = []
        black_pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                sq_name = chess.square_name(square)
                symbol = piece_symbols[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_pieces.append(f"{symbol}:{sq_name}")
                else:
                    black_pieces.append(f"{symbol}:{sq_name}")
        return " ".join(white_pieces), " ".join(black_pieces)

    def _legal_moves_csv(self, board: chess.Board) -> str:
        return ",".join(move.uci() for move in board.legal_moves)

    def encode(self, board: chess.Board, last_moves: List[str]) -> str:
        white_coords, black_coords = self._piece_coord_map(board)
        turn_str = "White" if board.turn == chess.WHITE else "Black"
        last_4_moves = "\n".join(last_moves[-4:]) if last_moves else "None"
        
        return PROMPT_TEMPLATE.format(
            fen=board.fen(),
            turn=turn_str,
            halfmove_clock=board.halfmove_clock,
            white_piece_coords=white_coords,
            black_piece_coords=black_coords,
            last_4_uci_moves=last_4_moves,
            legal_moves_csv=self._legal_moves_csv(board)
        )

def parse_candidates(text: str) -> List[str]:
    # 1. Primary: exact <move>uci</move> tags
    moves = re.findall(r'<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>', text, re.IGNORECASE)
    if moves:
        return [m.lower() for m in moves]

    # 2. Secondary: check inside <candidates> ... </candidates>
    cand_match = re.search(r'<candidates>(.*?)(?:</candidates>|$)', text, re.DOTALL | re.IGNORECASE)
    if cand_match:
        cand_text = cand_match.group(1)
        cand_moves = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', cand_text, re.IGNORECASE)
        if cand_moves:
            return [m.lower() for m in cand_moves]

    # 3. Fallback: any 4-5 char UCI moves mentioned anywhere in text
    all_ucis = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text, re.IGNORECASE)
    return [m.lower() for m in all_ucis]
