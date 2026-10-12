"""
tests/test_board_projector.py
==============================
Unit tests for the LLaVA-style Soft-Prompt Board Projector architecture.
Verifies spatial tensor encoding, CNN feature extraction, MLP projection,
and multimodal embedding concatenation.
"""

from unittest.mock import MagicMock
import chess
import pytest
import torch
import torch.nn as nn

from src.models.board_projector import (
    encode_board_tensor,
    batch_encode_boards,
    BoardEncoderCNN,
    BoardProjector,
    MultimodalChessPolicy,
)


def test_encode_board_tensor_starting_position():
    """Asserts starting position tensor is properly shaped and encoded."""
    board = chess.Board()
    tensor = encode_board_tensor(board)

    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == torch.float32

    # In starting position, White pawns are on rank 2 (index 1)
    assert tensor[0, 1, :].sum().item() == 8.0  # 8 White pawns
    # Black pawns are on rank 7 (index 6)
    assert tensor[6, 6, :].sum().item() == 8.0  # 8 Black pawns

    # Castling rights: White both (ch 12, 13) and Black both (ch 14, 15)
    assert tensor[12, 0, 0].item() == 1.0
    assert tensor[13, 0, 0].item() == 1.0
    assert tensor[14, 0, 0].item() == 1.0
    assert tensor[15, 0, 0].item() == 1.0

    # No en-passant square initially
    assert tensor[16].sum().item() == 0.0

    # White to move -> channel 17 is all 1s
    assert tensor[17, 0, 0].item() == 1.0


def test_encode_board_tensor_after_e4():
    """Asserts tensor reflects board changes after 1. e4."""
    board = chess.Board()
    board.push_san("e4")  # 1. e4 -> ep square is e3, black to move
    tensor = encode_board_tensor(board)

    # Black to move -> channel 17 should be all 0s
    assert tensor[17].sum().item() == 0.0

    # En-passant square at e3 (rank 2, file 4 in 0-indexed coords)
    ep_rank = chess.square_rank(chess.E3)
    ep_file = chess.square_file(chess.E3)
    assert tensor[16, ep_rank, ep_file].item() == 1.0

    # White pawn on e4 (rank 3, file 4)
    pawn_rank = chess.square_rank(chess.E4)
    pawn_file = chess.square_file(chess.E4)
    assert tensor[0, pawn_rank, pawn_file].item() == 1.0


def test_batch_encode_boards():
    """Asserts batch encoding produces a stacked (B, 18, 8, 8) tensor."""
    boards = [chess.Board(), chess.Board()]
    boards[1].push_san("d4")
    batch_tensor = batch_encode_boards(boards)
    assert batch_tensor.shape == (2, 18, 8, 8)


def test_board_encoder_cnn():
    """Asserts 4-layer ResNet CNN preserves 8x8 spatial resolution."""
    cnn = BoardEncoderCNN(in_channels=18, hidden_dim=64, out_channels=128)
    x = torch.randn(2, 18, 8, 8)
    out = cnn(x)
    assert out.shape == (2, 128, 8, 8)


def test_board_projector_tokens():
    """Asserts BoardProjector flattens 8x8 into 64 virtual tokens of dimension hidden_size."""
    hidden_size = 512
    projector = BoardProjector(
        in_channels=18,
        cnn_features=128,
        llm_hidden_size=hidden_size,
    )
    x = torch.randn(3, 18, 8, 8)
    virtual_tokens = projector(x)
    assert virtual_tokens.shape == (3, 64, hidden_size)


def test_multimodal_chess_policy_fusion():
    """Asserts prepending 64 virtual board tokens to text prompt tokens."""
    hidden_size = 256
    seq_len = 10
    batch_size = 2

    # Mock LLM
    class DummyLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, hidden_size)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds, attention_mask=None, labels=None):
            out = MagicMock()
            out.shape = inputs_embeds.shape
            out.labels_received = labels
            return out

    dummy_llm = DummyLLM()
    projector = BoardProjector(
        in_channels=18,
        cnn_features=64,
        llm_hidden_size=hidden_size,
    )
    policy = MultimodalChessPolicy(
        llm_model=dummy_llm,
        projector=projector,
        llm_hidden_size=hidden_size,
    )

    boards = [chess.Board() for _ in range(batch_size)]
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    labels = torch.randint(0, 100, (batch_size, seq_len))

    # Test prepare_multimodal_embeddings
    embeds, mask = policy.prepare_multimodal_embeddings(boards, input_ids, attention_mask)
    assert embeds.shape == (batch_size, 64 + seq_len, hidden_size)
    assert mask.shape == (batch_size, 64 + seq_len)
    assert mask[:, :64].sum().item() == batch_size * 64  # all 1s for board tokens

    # Test forward pass with label masking
    out = policy(boards, input_ids, attention_mask, labels=labels)
    assert out.shape == (batch_size, 64 + seq_len, hidden_size)
    assert out.labels_received.shape == (batch_size, 64 + seq_len)
    # The first 64 tokens of labels must be -100 (ignored in cross-entropy loss)
    assert (out.labels_received[:, :64] == -100).all()
