import functools
import torch
import chess
from typing import List, Set, Dict, Optional
from transformers import PreTrainedTokenizer, LogitsProcessor


_CHAR_MAPS_CACHE = {}

def _get_char_token_maps(tokenizer):
    key = id(tokenizer)
    if key in _CHAR_MAPS_CACHE:
        return _CHAR_MAPS_CACHE[key]

    char_to_tokens = {}
    for ch in "abcdefgh12345678qrbn":
        s = set(tokenizer.encode(ch, add_special_tokens=False))
        s.update(tokenizer.encode(" " + ch, add_special_tokens=False))
        char_to_tokens[ch] = s

    close_tokens = set(tokenizer.encode("</", add_special_tokens=False))
    close_tokens.update(tokenizer.encode("</move>", add_special_tokens=False))
    close_tokens.update(tokenizer.encode(">", add_special_tokens=False))

    _CHAR_MAPS_CACHE[key] = (char_to_tokens, close_tokens)
    return char_to_tokens, close_tokens


class MoveExtractor:
    """Helper class to extract prefixes inside move tags."""
    def __init__(self, move_start_tag: str = "<move>"):
        self.move_start_tag = move_start_tag

    def extract_prefix(self, generated_text: str) -> Optional[str]:
        tag_idx = generated_text.rfind(self.move_start_tag)
        if tag_idx != -1:
            close_idx = generated_text.find("</move>", tag_idx)
            if close_idx == -1:
                return generated_text[tag_idx + len(self.move_start_tag):].strip()
        return None


class LogitMasker(LogitsProcessor):
    """
    Constrained decoding logit masker for legal chess moves.
    High-performance: uses precomputed character-to-token maps and a trie,
    operating in microseconds per token with zero memory leaks.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, board: chess.Board):
        self.tokenizer = tokenizer
        self.board = board
        self.active = False

        # Build character-level trie of legal moves
        self.trie: Dict = {}
        for m in board.legal_moves:
            curr = self.trie
            for ch in m.uci():
                if ch not in curr:
                    curr[ch] = {}
                curr = curr[ch]
            curr["<end>"] = True

        # Precompute character tokens for this tokenizer
        self.char_to_tokens, self.close_tokens = _get_char_token_maps(tokenizer)

        # Precompute starting tokens for any legal move
        self.start_tokens: Set[int] = set()
        for ch in self.trie:
            if ch != "<end>" and ch in self.char_to_tokens:
                self.start_tokens.update(self.char_to_tokens[ch])

    def set_move_mode(self, active: bool):
        self.active = active

    def _get_allowed_tokens(self, prefix: str) -> Set[int]:
        curr = self.trie
        for ch in prefix:
            if ch in curr and ch != "<end>":
                curr = curr[ch]
            else:
                # If prefix is not in trie, allow starting a new legal move or closing
                return self.start_tokens | self.close_tokens

        allowed = set()
        for ch in curr:
            if ch != "<end>" and ch in self.char_to_tokens:
                allowed.update(self.char_to_tokens[ch])

        if "<end>" in curr:
            allowed.update(self.close_tokens)

        return allowed or self.start_tokens

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return scores

        bs = input_ids.shape[0]

        for i in range(bs):
            # Only decode the trailing 12 tokens to check if we are inside <move>
            tail_tokens = input_ids[i][-12:].tolist()
            tail_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=False)

            tag_idx = tail_text.rfind("<move>")
            if tag_idx != -1:
                close_idx = tail_text.find("</move>", tag_idx)
                if close_idx == -1:
                    # Currently inside <move> tag
                    prefix = tail_text[tag_idx + len("<move>"):].strip()
                    allowed_tokens = self._get_allowed_tokens(prefix)

                    if allowed_tokens and len(allowed_tokens) < scores.shape[-1]:
                        mask = torch.ones(
                            scores.shape[-1], dtype=torch.bool, device=scores.device
                        )
                        allowed_tensor = torch.tensor(
                            list(allowed_tokens), dtype=torch.long, device=scores.device
                        )
                        mask[allowed_tensor] = False
                        # Use -1e4 instead of -inf to avoid NaN/overflow on MPS/bfloat16
                        scores[i, mask] = -1e4

        return scores
