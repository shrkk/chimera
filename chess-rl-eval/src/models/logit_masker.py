import functools
import torch
import chess
from typing import List, Set, Dict, Optional
from transformers import PreTrainedTokenizer, LogitsProcessor


@functools.lru_cache(maxsize=1024)
def _build_trie_cached(legal_ucis_tuple: tuple) -> Dict:
    """Builds a prefix trie from legal UCI moves with LRU caching."""
    trie = {}
    for uci in legal_ucis_tuple:
        curr = trie
        for char in uci:
            if char not in curr:
                curr[char] = {}
            curr = curr[char]
        curr["<end>"] = True
    return trie


class MoveExtractor:
    """
    Helper class to detect when the model enters the <move> tag and track the current move prefix.
    """
    def __init__(self, move_start_tag: str = "<move>"):
        self.move_start_tag = move_start_tag
        self.move_tag_len = len(move_start_tag)

    def extract_prefix(self, generated_text: str) -> Optional[str]:
        """
        Returns the move prefix if currently inside a <move> tag, else None.
        """
        tag_idx = generated_text.rfind(self.move_start_tag)
        if tag_idx != -1:
            close_idx = generated_text.find("</move>", tag_idx)
            if close_idx == -1:
                return generated_text[tag_idx + self.move_tag_len:].strip()
        return None


class LogitMasker(LogitsProcessor):
    """
    Implements constrained decoding (logit masking) for legal chess moves.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, board: chess.Board):
        self.tokenizer = tokenizer
        self.board = board
        self.active = False
        ucis_tuple = tuple(sorted(move.uci() for move in board.legal_moves))
        self.trie = _build_trie_cached(ucis_tuple)

    def set_move_mode(self, active: bool):
        """Toggle masking on/off."""
        self.active = active

    def _get_allowed_next_tokens(self, prefix: str) -> Set[int]:
        """
        Given current decoded prefix, returns set of token IDs that extend it toward a valid UCI.
        """
        allowed_tokens = set()

        curr = self.trie
        for char in prefix:
            if char in curr:
                curr = curr[char]
            else:
                return allowed_tokens

        vocab = self.tokenizer.get_vocab()

        for token_str, token_id in vocab.items():
            decoded_token = self.tokenizer.decode([token_id]).strip()
            if not decoded_token:
                continue

            test_curr = curr
            is_valid = True
            for char in decoded_token:
                if char in test_curr:
                    test_curr = test_curr[char]
                else:
                    is_valid = False
                    break

            if is_valid:
                allowed_tokens.add(token_id)

        # Allow closing tag token if prefix forms a complete move
        if "<end>" in curr:
            close_token = self.tokenizer.encode("</", add_special_tokens=False)
            if close_token:
                allowed_tokens.add(close_token[0])

        return allowed_tokens

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Masks logits to -inf for any token that cannot be a prefix of a legal UCI move
        when the model is in <move> generation mode.
        """
        if not self.active:
            return scores

        batch_size = input_ids.shape[0]
        extractor = MoveExtractor()

        for i in range(batch_size):
            generated_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
            prefix = extractor.extract_prefix(generated_text)

            if prefix is not None:
                allowed_token_ids = self._get_allowed_next_tokens(prefix)

                mask = torch.ones_like(scores[i], dtype=torch.bool)
                if allowed_token_ids:
                    allowed_indices = torch.tensor(
                        list(allowed_token_ids), dtype=torch.long, device=scores.device
                    )
                    mask[allowed_indices] = False

                scores[i, mask] = float("-inf")

        return scores
