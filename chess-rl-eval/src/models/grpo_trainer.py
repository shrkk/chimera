import os
import argparse
import copy
import json
import logging
import pathlib
import re
import sys

# Disable hard upper allocation cap on Apple Silicon unified memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import yaml
import chess
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# Ensure project root is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.chess_gym import ChessGym
from src.models.logit_masker import LogitMasker, MoveExtractor
from src.env.state_encoder import StateEncoder, parse_candidates
from src.verifier.tactical_verifier import select_best_candidate

logger = logging.getLogger(__name__)


def compute_grpo_loss(
    policy_model,
    ref_model,
    input_ids: torch.Tensor,
    action_masks: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    beta_kl: float = 0.04,
) -> torch.Tensor:
    """
    Computes GRPO surrogate loss with group advantage normalization.
    Optimized for low VRAM: slices logits to generated tokens only,
    evaluates ref model under inference_mode and frees its memory early.
    """
    device = policy_model.device
    rewards = rewards.to(device)
    action_masks = action_masks.to(device)
    input_ids = input_ids.to(device)

    # 1. Compute Group Advantages per-group [Batch_Size, Group_Size]
    rewards = rewards.view(-1, group_size)
    mean_reward = rewards.mean(dim=-1, keepdim=True)
    std_reward = rewards.std(dim=-1, unbiased=False, keepdim=True)
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)
    advantages = advantages.view(-1)  # Flatten to match batch dimension

    shift_labels = input_ids[..., 1:].contiguous()
    shift_action_masks = action_masks[..., 1:].contiguous()

    # Find earliest generated token to avoid backpropping through prompt
    active_indices = (shift_action_masks.sum(dim=0) > 0).nonzero()
    start_idx = active_indices[0].item() if len(active_indices) > 0 else 0

    sliced_labels = shift_labels[:, start_idx:].contiguous()
    sliced_masks = shift_action_masks[:, start_idx:].contiguous()

    # 2. Reference model pass in inference_mode, then free its tensors
    with torch.inference_mode():
        if ref_model is not None:
            ref_outputs = ref_model(input_ids=input_ids)
        else:
            # Dynamic LoRA adapter bypass: computes base model reference with 0 extra VRAM
            with policy_model.disable_adapter():
                ref_outputs = policy_model(input_ids=input_ids)

        ref_logits = ref_outputs.logits[:, start_idx:-1, :].contiguous()
        ref_log_probs = ref_logits.log_softmax(dim=-1)
        token_ref_log_probs = torch.gather(
            ref_log_probs, -1, sliced_labels.unsqueeze(-1)
        ).squeeze(-1)
        del ref_outputs, ref_logits, ref_log_probs
        if device.type == "mps":
            torch.mps.empty_cache()

    # 3. Active policy model pass
    policy_outputs = policy_model(input_ids=input_ids)
    policy_logits = policy_outputs.logits[:, start_idx:-1, :].contiguous()
    policy_log_probs = policy_logits.log_softmax(dim=-1)
    token_log_probs = torch.gather(
        policy_log_probs, -1, sliced_labels.unsqueeze(-1)
    ).squeeze(-1)

    # 4. Token-level KL divergence approximation: exp(ref - policy) - (ref - policy) - 1
    diff = token_ref_log_probs - token_log_probs
    kl = torch.exp(diff) - diff - 1.0

    # 5. GRPO policy gradient loss with advantage broadcasting
    ratio = torch.exp(token_log_probs - token_log_probs.detach())
    adv_broadcast = advantages.unsqueeze(-1)
    surr1 = ratio * adv_broadcast
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv_broadcast
    policy_loss = -torch.min(surr1, surr2) + (beta_kl * kl)

    # 6. Mask out prompt and padding tokens
    masked_loss = policy_loss * sliced_masks
    return masked_loss.sum() / (sliced_masks.sum() + 1e-8)


class GRPOTrainer:
    def __init__(self, config: dict):
        self.config = config

        # Device selection: prefer MPS > CUDA > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Using device: %s", self.device)

        model_cfg = self.config.get("model", {})
        grpo_cfg = self.config.get("grpo", {})
        model_name = model_cfg.get("base_model", "Qwen/Qwen2.5-Coder-7B-Instruct")

        dtype = torch.bfloat16 if self.device.type in ["mps", "cuda"] else torch.float32

        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)

        lora_cfg = self.config.get("lora", {})
        self.use_lora = lora_cfg.get("enabled", True)

        if self.use_lora:
            logger.info("LoRA enabled: wrapping policy model with PEFT adapters.")
            lora_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]),
                task_type=TaskType.CAUSAL_LM,
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            self.policy_model.print_trainable_parameters()
            self.ref_model = None  # Reference logits computed via dynamic adapter disable (0 extra VRAM)
        else:
            logger.info("LoRA disabled: loading dedicated reference model copy.")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            ).to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._encoder = StateEncoder()
        self.env = ChessGym(config=self.config)

        # Load puzzles dataset if available
        puzzle_file = pathlib.Path(
            self.config.get("datasets", {}).get("puzzle_output", "datasets/puzzles.jsonl")
        )
        self.puzzles = []
        if puzzle_file.exists():
            with open(puzzle_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.puzzles.append(json.loads(line))
                        except Exception:
                            pass
            logger.info("Loaded %d tactical puzzles for GRPO training.", len(self.puzzles))

        self.group_size = grpo_cfg.get("group_size", 4)
        self.max_cot_tokens = model_cfg.get("max_cot_tokens", 80)
        self.learning_rate = float(grpo_cfg.get("learning_rate", 5e-6))
        self.episodes = grpo_cfg.get("episodes", 5000)
        self.save_every = grpo_cfg.get("save_every", 500)
        self.checkpoint_dir = pathlib.Path(
            grpo_cfg.get("checkpoint_dir", "./checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.checkpoint_dir / "training_metrics.jsonl"

        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate
        )

        grad_accum_steps = grpo_cfg.get("gradient_accumulation_steps", 1)
        total_steps = self.episodes // grad_accum_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=grpo_cfg.get("warmup_steps", 100),
            num_training_steps=max(total_steps, 1),
        )

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def _compute_action_mask(self, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        mask = torch.zeros_like(input_ids)
        mask[:, prompt_len:] = 1
        return mask

    def rollout(self, board_state: dict) -> list[dict]:
        """
        Samples G completions from policy in a single batched generate call.
        All G completions are evaluated from the exact same initial board state.
        """
        initial_fen = board_state.get("fen", chess.STARTING_FEN)
        initial_solution = list(board_state.get("puzzle_solution", []))
        initial_last_moves = list(board_state.get("last_moves", []))

        board = chess.Board(initial_fen)
        prompt = self._encoder.encode(board, last_moves=initial_last_moves)
        prompt_ids = self._tokenize_prompt(prompt)
        prompt_len = prompt_ids.shape[1]

        masker = LogitMasker(self.tokenizer, board)
        masker.set_move_mode(True)

        from transformers import LogitsProcessorList
        processors = LogitsProcessorList([masker])

        # Batched generation of all G completions
        with torch.no_grad():
            output_ids_batch = self.policy_model.generate(
                prompt_ids,
                max_new_tokens=self.max_cot_tokens,
                do_sample=True,
                temperature=1.0,
                logits_processor=processors,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=self.group_size,
            )

        from src.verifier.tactical_verifier import select_best_candidate

        completions = []
        for i in range(self.group_size):
            output_ids = output_ids_batch[i : i + 1]
            generated_text = self.tokenizer.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True
            )

            # Reset env to the exact same initial state for this completion
            eval_board = chess.Board(initial_fen)
            self.env.reset(fen=initial_fen, puzzle_solution=initial_solution)

            # Extract candidates and select best via neuro-symbolic verifier
            try:
                candidates = parse_candidates(generated_text)
                has_move_tags = bool(re.search(r'<move>.*?</move>', generated_text, re.IGNORECASE))

                best_move = select_best_candidate(eval_board, candidates)
                action = best_move.uci() if best_move != chess.Move.null() else None

                if action is not None and action in [m.uci() for m in eval_board.legal_moves]:
                    _, reward, _, _ = self.env.step(action)
                    if not has_move_tags:
                        reward -= 0.1  # Formatting penalty for omitting <move> tags
                else:
                    reward = self.config.get("reward", {}).get("terminal_illegal", -1.5)
            except Exception as e:
                logger.warning("Step evaluation error: %s", e)
                reward = self.config.get("reward", {}).get("terminal_illegal", -1.5)

            action_mask = self._compute_action_mask(output_ids, prompt_len)

            completions.append({
                "input_ids": output_ids[0],
                "action_mask": action_mask[0],
                "reward": reward,
                "text": generated_text,
            })

        return completions

    def train(self):
        """Main GRPO training loop."""
        logger.info(
            "Starting GRPO training for %d episodes on %s …", self.episodes, self.device
        )
        self.policy_model.train()
        grad_accum_steps = self.config.get("grpo", {}).get("gradient_accumulation_steps", 1)

        for episode in range(1, self.episodes + 1):
            if self.puzzles:
                puzzle = self.puzzles[(episode - 1) % len(self.puzzles)]
                board_state = self.env.reset(
                    fen=puzzle.get("fen"),
                    puzzle_solution=puzzle.get("moves", []),
                )
                board_state["puzzle_solution"] = puzzle.get("moves", [])
            else:
                board_state = self.env.reset()

            completions = self.rollout(board_state)

            max_len = max(len(c["input_ids"]) for c in completions)

            batch_input_ids, batch_action_masks, batch_rewards = [], [], []
            for c in completions:
                pad_len = max_len - len(c["input_ids"])
                padded_ids = torch.nn.functional.pad(
                    c["input_ids"], (0, pad_len), value=self.tokenizer.pad_token_id
                )
                batch_input_ids.append(padded_ids)
                padded_mask = torch.nn.functional.pad(c["action_mask"], (0, pad_len), value=0)
                batch_action_masks.append(padded_mask)
                batch_rewards.append(c["reward"])

            batch_input_ids = torch.stack(batch_input_ids)
            batch_action_masks = torch.stack(batch_action_masks)
            batch_rewards = torch.tensor(
                batch_rewards, dtype=torch.float32, device=self.device
            )

            loss = compute_grpo_loss(
                self.policy_model,
                self.ref_model,
                batch_input_ids,
                batch_action_masks,
                batch_rewards,
                self.group_size,
            )

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if episode % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            logger.info(
                "Episode %d/%d — Loss: %.4f — Mean Reward: %.4f",
                episode,
                self.episodes,
                loss.item(),
                batch_rewards.mean().item(),
            )

            metrics_entry = {
                "episode": episode,
                "loss": round(float(loss.item()), 5),
                "mean_reward": round(float(batch_rewards.mean().item()), 5),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            try:
                with open(self.metrics_file, "a") as f:
                    f.write(json.dumps(metrics_entry) + "\n")
            except Exception as e:
                logger.warning("Failed to write training metrics: %s", e)

            if episode % self.save_every == 0:
                self.save_checkpoint(episode)

        logger.info("Training complete.")

    def save_checkpoint(self, step: int):
        save_path = self.checkpoint_dir / f"grpo_chess_step_{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Saved checkpoint → %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="GRPO Trainer for Chess RL")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--base-model", type=str, help="Base model name or path")
    parser.add_argument("--group-size", type=int, help="Number of completions per prompt (G)")
    parser.add_argument("--max-cot-tokens", type=int, help="Max tokens for Chain of Thought generation")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--save-every", type=int, help="Checkpoint interval (episodes)")
    parser.add_argument("--stockfish-threads", type=int, help="Number of Stockfish threads")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and train full model parameters")
    parser.add_argument("--lora-r", type=int, help="LoRA rank dimension r")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if "model" not in config:
        config["model"] = {}
    if "grpo" not in config:
        config["grpo"] = {}
    if "lora" not in config:
        config["lora"] = {}

    if args.no_lora:
        config["lora"]["enabled"] = False
    if args.lora_r:
        config["lora"]["r"] = args.lora_r

    if args.base_model:
        config["model"]["base_model"] = args.base_model
    if args.group_size:
        config["grpo"]["group_size"] = args.group_size
    if args.max_cot_tokens:
        config["model"]["max_cot_tokens"] = args.max_cot_tokens
    if args.learning_rate:
        config["grpo"]["learning_rate"] = args.learning_rate
    if args.episodes:
        config["grpo"]["episodes"] = args.episodes
    if args.save_every:
        config["grpo"]["save_every"] = args.save_every
    if args.stockfish_threads:
        config["grpo"]["stockfish_threads"] = args.stockfish_threads

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
