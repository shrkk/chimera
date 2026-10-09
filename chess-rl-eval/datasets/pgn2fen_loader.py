import io
import pathlib
import typing

import chess
import chess.pgn

class PGN2FENLoader:
    """
    Utility class for loading PGN games and converting them to FEN positions
    at specific plies.
    """
    
    @staticmethod
    def load_pgn_file(path: typing.Union[str, pathlib.Path]) -> typing.List[str]:
        """
        Reads multiple PGN games from a .pgn file.
        Returns a list of PGN strings.
        """
        path = pathlib.Path(path)
        games = []
        with open(path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Exporter for full string representation of the game
                exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                games.append(game.accept(exporter))
                
        return games
        
    def iter_game_fens(self, pgn_data: typing.Union[str, pathlib.Path, typing.List[str]]) -> typing.Iterator[typing.Tuple[int, str, str]]:
        """
        Loads PGN game strings (accepts either a file path or a list of PGN strings/a single PGN string).
        Uses python-chess to replay games move by move.
        Yields (ply_index, fen_string, move_uci) tuples for each half-move.
        """
        pgn_strings = []
        
        if isinstance(pgn_data, str):
            # Check if it's a file path based on suffix
            p = pathlib.Path(pgn_data)
            if p.is_file() and p.suffix.lower() == ".pgn":
                pgn_strings = self.load_pgn_file(p)
            else:
                pgn_strings = [pgn_data]
        elif isinstance(pgn_data, pathlib.Path):
            pgn_strings = self.load_pgn_file(pgn_data)
        elif isinstance(pgn_data, list):
            pgn_strings = pgn_data
            
        for pgn_string in pgn_strings:
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            if not game:
                continue
                
            board = game.board()
            ply = 0
            
            for move in game.mainline_moves():
                # FEN before the move
                fen = board.fen()
                move_uci = move.uci()
                yield (ply, fen, move_uci)
                
                board.push(move)
                ply += 1

    @staticmethod
    def get_fen_at_ply(pgn_string: str, ply: int) -> typing.Optional[str]:
        """
        Returns the exact FEN at a given half-move ply.
        """
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        if not game:
            return None
            
        board = game.board()
        current_ply = 0
        
        if ply == 0:
            return board.fen()
            
        for move in game.mainline_moves():
            board.push(move)
            current_ply += 1
            if current_ply == ply:
                return board.fen()
                
        return None

    def batch_evaluate(self, pgn_list: typing.List[str], target_plies: typing.List[int]) -> typing.Dict[int, typing.List[str]]:
        """
        Returns a dict {ply: [fen1, fen2, ...]} for evaluation.
        Evaluates across multiple PGNs and collects FENs at the specified plies.
        """
        results = {ply: [] for ply in target_plies}
        max_ply = max(target_plies) if target_plies else -1
        
        for pgn_string in pgn_list:
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            if not game:
                continue
                
            board = game.board()
            current_ply = 0
            
            if 0 in target_plies:
                results[0].append(board.fen())
                
            for move in game.mainline_moves():
                board.push(move)
                current_ply += 1
                
                if current_ply in target_plies:
                    results[current_ply].append(board.fen())
                    
                if current_ply >= max_ply:
                    break
                    
        return results
