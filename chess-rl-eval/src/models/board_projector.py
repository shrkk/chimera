"""
src/models/board_projector.py
==============================
LLaVA-Inspired Soft-Prompt Board Projector for Neuro-Symbolic Chess Reasoning.

Encodes an 8x8 chessboard into an exact 18-channel spatial tensor (12 piece planes,
4 castling planes, 1 en-passant plane, 1 side-to-move plane), processes it through
a 4-layer spatial Residual CNN trunk preserving the 8x8 grid geometry, and projects
the 64 square representations directly into the LLM's token embedding space
(e.g., hidden_size = 3584 for Qwen2.5-Coder-7B).

Virtual tokens [<board_a1>, ..., <board_h8>] are prepended to the text prompt's
`inputs_embeds`, enabling pure causal self-attention over exact 2D board geometry
with zero tokenizer fragmentation and full PEFT/LoRA compatibility.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Spatial Board Tensor Serialization ──────────────────────────────────────

PIECE_ORDER = [
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
]


def encode_board_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a python-chess Board into an (18, 8, 8) float tensor.
    Channels:
      [0..5]:   White pieces (P, N, B, R, Q, K)
      [6..11]:  Black pieces (p, n, b, r, q, k)
      [12]:     White Kingside Castling right (all 1s if valid else 0s)
      [13]:     White Queenside Castling right (all 1s if valid else 0s)
      [14]:     Black Kingside Castling right (all 1s if valid else 0s)
      [15]:     Black Queenside Castling right (all 1s if valid else 0s)
      [16]:     En-passant target square (1 on the target square, 0 elsewhere)
      [17]:     Active side to move (all 1s for White, all 0s for Black)
    
    Spatial layout:
      Rank 1 -> index 0, Rank 8 -> index 7
      File a -> index 0, File h -> index 7
    """
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)

    # 1. Piece channels
    for ch_idx, (piece_type, color) in enumerate(PIECE_ORDER):
        square_set = board.pieces(piece_type, color)
        for sq in square_set:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            tensor[ch_idx, rank, file] = 1.0

    # 2. Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    # 3. En-passant square
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[16, ep_rank, ep_file] = 1.0

    # 4. Side to move
    if board.turn == chess.WHITE:
        tensor[17, :, :] = 1.0

    return tensor


def batch_encode_boards(boards: List[chess.Board]) -> torch.Tensor:
    """Encodes a batch of chess.Board instances into (B, 18, 8, 8) tensor."""
    tensors = [encode_board_tensor(b) for b in boards]
    return torch.stack(tensors, dim=0)


# ── 2. Spatial Residual CNN Trunk ──────────────────────────────────────────────

class ResidualBlock2D(nn.Module):
    """2D Residual Block with 3x3 convolutions preserving (8, 8) resolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class BoardEncoderCNN(nn.Module):
    """
    4-stage spatial CNN extracting tactical patterns and square features
    without spatial downsampling (strictly preserving 8x8 resolution).
    """
    def __init__(self, in_channels: int = 18, hidden_dim: int = 256, out_channels: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.res1 = ResidualBlock2D(hidden_dim)
        self.res2 = ResidualBlock2D(hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 18, 8, 8) -> out: (B, out_channels, 8, 8)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


# ── 3. LLaVA-Style MLP Projector ───────────────────────────────────────────────

class BoardProjector(nn.Module):
    """
    Projects 64 spatial square vectors from the BoardEncoderCNN
    directly into the LLM's hidden embedding dimension.
    """
    def __init__(
        self,
        in_channels: int = 18,
        cnn_features: int = 512,
        llm_hidden_size: int = 3584,
    ):
        super().__init__()
        self.cnn_features = cnn_features
        self.llm_hidden_size = llm_hidden_size

        self.encoder = BoardEncoderCNN(
            in_channels=in_channels,
            hidden_dim=256,
            out_channels=cnn_features,
        )

        # 2-layer MLP Projection (identical to LLaVA architecture)
        self.projector = nn.Sequential(
            nn.Linear(cnn_features, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
        )

    def forward(self, board_tensors: torch.Tensor) -> torch.Tensor:
        """
        Input:  board_tensors: (B, 18, 8, 8)
        Output: virtual_tokens: (B, 64, llm_hidden_size)
        """
        features = self.encoder(board_tensors)  # (B, 512, 8, 8)
        # Permute to (B, 8, 8, 512) and flatten spatial grid to 64 tokens
        B, C, H, W = features.shape
        tokens = features.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B, 64, 512)
        projected = self.projector(tokens)  # (B, 64, llm_hidden_size)
        return projected


# ── 4. Multimodal Fusion Module ────────────────────────────────────────────────

class MultimodalChessPolicy(nn.Module):
    """
    Wraps an autoregressive CausalLM (with LoRA) and the BoardProjector.
    Injects 64 virtual board tokens as a soft prompt prefix directly into `inputs_embeds`.
    """
    def __init__(
        self,
        llm_model: nn.Module,
        projector: Optional[BoardProjector] = None,
        llm_hidden_size: int = 3584,
    ):
        super().__init__()
        self.llm = llm_model
        if projector is not None:
            self.projector = projector
        else:
            self.projector = BoardProjector(llm_hidden_size=llm_hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.projector.parameters()).device

    def prepare_multimodal_embeddings(
        self,
        boards: List[chess.Board],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepends 64 board square virtual tokens to the tokenized prompt embeddings.
        
        Returns:
          combined_embeds: (B, 64 + seq_len, hidden_size)
          combined_mask:   (B, 64 + seq_len)
        """
        device = self.device
        dtype = next(self.llm.parameters()).dtype

        # 1. Encode boards and project to (B, 64, hidden_size)
        board_tensors = batch_encode_boards(boards).to(device=device, dtype=torch.float32)
        board_embeds = self.projector(board_tensors).to(dtype=dtype)  # (B, 64, hidden_size)
        batch_size, num_board_tokens, _ = board_embeds.shape

        # 2. Extract LLM word embeddings
        input_ids = input_ids.to(device)
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, hidden_size)

        # 3. Concatenate embeddings
        combined_embeds = torch.cat([board_embeds, text_embeds], dim=1)

        # 4. Concatenate attention masks
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        board_mask = torch.ones(
            (batch_size, num_board_tokens),
            device=device,
            dtype=attention_mask.dtype,
        )
        combined_mask = torch.cat([board_mask, attention_mask], dim=1)

        return combined_embeds, combined_mask

    def forward(
        self,
        boards: List[chess.Board],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Standard forward pass utilizing combined multimodal inputs_embeds."""
        combined_embeds, combined_mask = self.prepare_multimodal_embeddings(
            boards, input_ids, attention_mask
        )

        # If labels are provided, pad them with -100 for the 64 board prefix tokens
        combined_labels = None
        if labels is not None:
            device = self.device
            prefix_ignore = torch.full(
                (len(boards), 64),
                -100,
                device=device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([prefix_ignore, labels.to(device)], dim=1)

        return self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            **kwargs,
        )
