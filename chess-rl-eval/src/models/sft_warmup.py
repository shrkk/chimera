"""
src/models/sft_warmup.py
========================
Supervised Fine-Tuning (SFT) Cold-Start Warmup for Chess RL Policy.
Trains LoRA adapters directly on Lichess puzzle solutions (1,200 puzzles)
to prime the model on:
  1. The XML output schema (<reasoning>...</reasoning><candidates>1. <move>...</move></candidates>)
  2. 8x8 piece coordinate spatial representation
  3. Tactical move prior distribution (raising baseline solve rate from ~10% to ~50%+)

Runtime on RTX 4090: ~60 to 90 seconds for 2 epochs.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import pathlib
import sys
from typing import List, Dict, Any

import chess
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.state_encoder import StateEncoder, parse_candidates       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SFTWarmup")


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def generate_tactical_cot(board: chess.Board, moves: list[str], themes: list[str]) -> tuple[str, list[str]]:
    """Generates step-by-step tactical mental calculation and up to 3 candidate moves."""
    m1 = chess.Move.from_uci(moves[0])
    piece = board.piece_at(m1.from_square)
    p_name = PIECE_NAMES.get(piece.piece_type, "piece").capitalize() if piece else "Piece"
    from_sq = chess.square_name(m1.from_square)
    to_sq = chess.square_name(m1.to_square)
    is_cap = board.is_capture(m1)
    captured = board.piece_at(m1.to_square)
    cap_name = PIECE_NAMES.get(captured.piece_type, "piece") if captured else "material"
    gives_check = board.gives_check(m1)

    actions = []
    if gives_check:
        actions.append("delivers a forcing check against the king")
    if is_cap:
        actions.append(f"captures the opponent's {cap_name} on {to_sq}")
    if not actions:
        actions.append(f"infiltrates to {to_sq} seizing key tactical control")
    action_desc = " and ".join(actions)

    followup_desc = ""
    if len(moves) >= 3:
        m2 = chess.Move.from_uci(moves[1])
        m3 = chess.Move.from_uci(moves[2])
        followup_desc = f" If the opponent counters with {m2.uci()}, the follow-up {m3.uci()} secures decisive advantage."
    elif len(moves) == 2:
        m2 = chess.Move.from_uci(moves[1])
        followup_desc = f" The opponent's best defense {m2.uci()} is insufficient to equalize."

    theme_str = ", ".join(themes[:3]) if themes else "tactics"
    reasoning = (
        f"Analyzing position for {theme_str}: {p_name} on {from_sq} moves to {to_sq}, which {action_desc}."
        f"{followup_desc} Primary move {m1.uci()} is the critical tactical solution."
    )

    alts = []
    for lm in board.legal_moves:
        if lm != m1 and lm.uci() not in alts:
            if board.is_capture(lm) or board.gives_check(lm):
                alts.append(lm.uci())
            if len(alts) >= 2:
                break
    if len(alts) < 2:
        for lm in board.legal_moves:
            if lm != m1 and lm.uci() not in alts:
                alts.append(lm.uci())
                if len(alts) >= 2:
                    break

    candidates = [m1.uci()] + alts
    return reasoning, candidates


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Definition
# ─────────────────────────────────────────────────────────────────────────────

class ChessPuzzleSFTDataset(Dataset):
    """
    Constructs SFT prompt-completion pairs from Lichess puzzles.
    Labels are masked with -100 over prompt tokens so loss is only
    computed on the generated reasoning and candidate move tokens.
    """

    def __init__(
        self,
        puzzle_path: pathlib.Path,
        tokenizer: AutoTokenizer,
        encoder: StateEncoder,
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.samples: List[Dict[str, Any]] = []

        if not puzzle_path.exists():
            raise FileNotFoundError(f"Puzzle file not found: {puzzle_path}")

        raw_puzzles = []
        with open(puzzle_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        raw_puzzles.append(json.loads(line))
                    except Exception:
                        pass

        if max_samples:
            raw_puzzles = raw_puzzles[:max_samples]

        logger.info("Processing %d puzzles for SFT warmup …", len(raw_puzzles))

        for p in raw_puzzles:
            fen = p.get("fen", "")
            sol_moves = p.get("moves", [])
            themes = p.get("themes", [])
            if not fen or not sol_moves:
                continue

            try:
                board = chess.Board(fen)
            except ValueError:
                continue

            winning_move_uci = sol_moves[0]
            prompt = self.encoder.encode(board, last_moves=[])

            reasoning_text, candidate_moves = generate_tactical_cot(board, sol_moves, themes)

            cands_str = "\n".join([f"{idx}. <move>{m}</move>" for idx, m in enumerate(candidate_moves, start=1)])
            target_completion = (
                f"\n{reasoning_text}\n"
                f"</reasoning>\n"
                f"<candidates>\n"
                f"{cands_str}\n"
                f"</candidates>"
            )

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = self.tokenizer.encode(target_completion, add_special_tokens=False)
            if self.tokenizer.eos_token_id:
                completion_ids.append(self.tokenizer.eos_token_id)

            input_ids = prompt_ids + completion_ids
            # Mask out prompt tokens in loss calculation
            labels = [-100] * len(prompt_ids) + completion_ids

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "fen": fen,
                "target_move": winning_move_uci,
            })

        logger.info("Built %d valid SFT training examples.", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_sft_batch(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        # Right-padding for causal LM SFT
        padded_inputs = item["input_ids"] + [pad_token_id] * pad_len
        padded_mask = [1] * seq_len + [0] * pad_len
        padded_labels = item["labels"] + [-100] * pad_len

        batch_input_ids.append(padded_inputs)
        batch_attention_mask.append(padded_mask)
        batch_labels.append(padded_labels)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────

def run_sft_warmup(
    model_name: str,
    output_dir: pathlib.Path,
    puzzle_path: pathlib.Path,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    device: str = "auto",
) -> None:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    logger.info("Initializing SFT Warmup on device: %s", dev)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if dev.type == "cuda":
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("FlashAttention-2 enabled for fast training.")
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"

    logger.info("Loading base model: %s …", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(dev)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for SFT.")
    model.print_trainable_parameters()

    encoder = StateEncoder()
    dataset = ChessPuzzleSFTDataset(puzzle_path, tokenizer, encoder)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, tokenizer.pad_token_id),
    )

    grad_accum_steps = 4
    total_opt_steps = (len(dataloader) * epochs) // grad_accum_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(5, total_opt_steps // 10), num_training_steps=total_opt_steps
    )

    logger.info("Starting SFT training: %d epochs (%d total batches, batch_size=%d, accum=%d) …", epochs, len(dataloader) * epochs, batch_size, grad_accum_steps)
    model.train()

    step_count = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            step_count += 1
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            (loss / grad_accum_steps).backward()

            if step_count % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / len(dataloader)
        logger.info("Epoch %d complete — Average SFT Loss: %.4f", epoch, avg_loss)

    # Save SFT Checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("✔ SFT Warmup successfully saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chimera SFT Cold-Start Warmup")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model name or path")
    parser.add_argument("--output-dir", default="checkpoints/sft_chess_7b", help="Output directory for SFT adapter")
    parser.add_argument("--puzzle-file", default="datasets/puzzles.jsonl", help="Path to puzzles.jsonl")
    parser.add_argument("--epochs", type=int, default=2, help="Number of SFT epochs (default 2)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for GPU (default 2 for 7B)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default 2e-4)")
    parser.add_argument("--device", default="auto", help="Device: auto / cuda / cpu")

    args = parser.parse_args()

    run_sft_warmup(
        model_name=args.model_path,
        output_dir=pathlib.Path(args.output_dir),
        puzzle_path=pathlib.Path(args.puzzle_file),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
