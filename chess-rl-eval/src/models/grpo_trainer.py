import argparse
import copy
import json
import logging
import pathlib
import torch
import yaml
import chess
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dynamic imports to avoid crashing if dependencies are missing during script compilation
try:
    from src.env.chess_gym import ChessGym
    from src.models.logit_masker import LogitMasker, MoveExtractor
    from src.env.state_encoder import StateEncoder
except ImportError:
    pass

logger = logging.getLogger(__name__)

def compute_grpo_loss(policy_model, ref_model, input_ids, action_masks, rewards, group_size, beta_kl=0.04) -> torch.Tensor:
    """
    Computes the GRPO loss for a group of completions.
    """
    device = policy_model.device
    rewards = rewards.to(device)
    action_masks = action_masks.to(device)
    input_ids = input_ids.to(device)
    
    # Compute group advantages per-group [Batch_Size, Group_Size]
    rewards = rewards.view(-1, group_size)
    mean_reward = rewards.mean(dim=-1, keepdim=True)
    std_reward = rewards.std(dim=-1, unbiased=False, keepdim=True)
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)
    advantages = advantages.view(-1)
    
    # Forward pass for policy and ref log probs
    policy_outputs = policy_model(input_ids=input_ids)
    policy_logits = policy_outputs.logits
    
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids)
        ref_logits = ref_outputs.logits
        
    shift_policy_logits = policy_logits[..., :-1, :].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_action_masks = action_masks[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    policy_log_probs = -loss_fct(shift_policy_logits.view(-1, shift_policy_logits.size(-1)), shift_labels.view(-1))
    policy_log_probs = policy_log_probs.view(shift_labels.size())
    
    ref_log_probs = -loss_fct(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_labels.view(-1))
    ref_log_probs = ref_log_probs.view(shift_labels.size())
    
    # Token-level KL: exp(ref - policy) - (ref - policy) - 1
    diff = ref_log_probs - policy_log_probs
    kl_div = torch.exp(diff) - diff - 1.0
    
    # PPO clipped surrogate with clip_epsilon=0.2
    old_log_probs = policy_log_probs.detach()
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip_epsilon = 0.2
    
    adv_broadcast = advantages.unsqueeze(1)
    
    surr1 = ratio * adv_broadcast
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_broadcast
    policy_loss = -torch.min(surr1, surr2)
    
    loss = policy_loss + beta_kl * kl_div
    
    # Mask by action_masks
    masked_loss = loss * shift_action_masks
    
    # Mean masked loss
    mean_masked_loss = masked_loss.sum() / (shift_action_masks.sum() + 1e-8)
    
    return mean_masked_loss

class GRPOTrainer:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Device: prefer MPS (Apple Silicon) > CUDA > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Using device: %s", self.device)

        model_cfg   = self.config.get("model", {})
        grpo_cfg    = self.config.get("grpo", {})
        model_name  = model_cfg.get("base_model", "gpt2")

        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._encoder = StateEncoder()
        self.env = ChessGym(config=self.config)

        self.group_size      = grpo_cfg.get("group_size", 4)
        self.max_cot_tokens  = model_cfg.get("max_cot_tokens", 80)
        self.learning_rate   = float(grpo_cfg.get("learning_rate", 5e-6))
        self.episodes        = grpo_cfg.get("episodes", 5000)
        self.save_every      = grpo_cfg.get("save_every", 500)

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.learning_rate
        )
        
        from transformers import get_cosine_schedule_with_warmup
        grad_accum_steps = self.config.get("grpo", {}).get("gradient_accumulation_steps", 1)
        total_steps = self.episodes // grad_accum_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.get("grpo", {}).get("warmup_steps", 100),
            num_training_steps=total_steps,
        )
        
    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
    def _compute_action_mask(self, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """mask is 1 only for generated tokens (not prompt)"""
        mask = torch.zeros_like(input_ids)
        mask[:, prompt_len:] = 1
        return mask
        
    def rollout(self, board_state: dict) -> list[dict]:
        """
        Samples G completions from policy, runs each through ChessGym.step(),
        collects rewards, returns list of {input_ids, action_mask, reward, text}
        """
        board  = chess.Board(board_state.get("fen", chess.STARTING_FEN))
        prompt = self._encoder.encode(board, last_moves=board_state.get("last_moves", []))
        prompt_ids = self._tokenize_prompt(prompt)
        prompt_len = prompt_ids.shape[1]

        masker = LogitMasker(self.tokenizer, board)
        masker.set_move_mode(True)

        from transformers import LogitsProcessorList
        processors = LogitsProcessorList([masker])

        completions = []
        for _ in range(self.group_size):
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    prompt_ids,
                    max_new_tokens=self.max_cot_tokens,
                    do_sample=True,
                    temperature=1.0,
                    logits_processor=processors,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True
            )

            # Extract first <move> tag and step the gym
            try:
                action = generated_text.split("<move>")[-1].split("</move>")[0].strip()
                _, reward, _, _ = self.env.step(action)
            except Exception:
                reward = self.config.get("reward", {}).get("terminal_illegal", -1.5)

            action_mask = self._compute_action_mask(output_ids, prompt_len)

            completions.append({
                "input_ids":   output_ids[0],
                "action_mask": action_mask[0],
                "reward":      reward,
                "text":        generated_text,
            })

        return completions

    def train(self):
        """Main GRPO training loop."""
        logger.info("Starting GRPO training for %d episodes on %s …", self.episodes, self.device)
        self.policy_model.train()

        for episode in range(1, self.episodes + 1):
            board_state  = self.env.reset()
            completions  = self.rollout(board_state)

            max_len = max(len(c["input_ids"]) for c in completions)

            batch_input_ids, batch_action_masks, batch_rewards = [], [], []
            for c in completions:
                pad_len    = max_len - len(c["input_ids"])
                padded_ids = torch.nn.functional.pad(
                    c["input_ids"], (0, pad_len), value=self.tokenizer.pad_token_id
                )
                batch_input_ids.append(padded_ids)
                padded_mask = torch.nn.functional.pad(c["action_mask"], (0, pad_len), value=0)
                batch_action_masks.append(padded_mask)
                batch_rewards.append(c["reward"])

            batch_input_ids    = torch.stack(batch_input_ids)
            batch_action_masks = torch.stack(batch_action_masks)
            batch_rewards      = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)

            loss = compute_grpo_loss(
                self.policy_model,
                self.ref_model,
                batch_input_ids,
                batch_action_masks,
                batch_rewards,
                self.group_size,
            )

            grad_accum_steps = self.config.get("grpo", {}).get("gradient_accumulation_steps", 1)
            loss = loss / grad_accum_steps
            loss.backward()

            if episode % grad_accum_steps == 0:
                if self.device.type == "mps":
                    for p in self.policy_model.parameters():
                        if p.grad is not None:
                            p.grad.data = p.grad.data.to(torch.float32)
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            logger.info(
                "Episode %d/%d — Loss: %.4f — Mean Reward: %.4f",
                episode, self.episodes, loss.item(), batch_rewards.mean().item(),
            )

            if episode % self.save_every == 0:
                self.save_checkpoint(episode)

        logger.info("Training complete.")

    def save_checkpoint(self, step: int):
        save_path = (
            pathlib.Path(self.config.get("grpo", {}).get("checkpoint_dir", "./checkpoints"))
            / f"grpo_chess_step_{step}"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Saved checkpoint → %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="GRPO Trainer for Chess RL")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--base-model", type=str, help="Base model name or path")
    parser.add_argument("--group-size", type=int, help="Number of completions per prompt (G)")
    parser.add_argument("--max-cot-tokens", type=int, help="Max tokens for Chain of Thought generation")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--stockfish-threads", type=int, help="Number of Stockfish threads")
    
    args = parser.parse_args()
    
    trainer = GRPOTrainer(args.config)
    
    if args.base_model: trainer.config["base_model"] = args.base_model
    if args.group_size: trainer.config["group_size"] = args.group_size
    if args.max_cot_tokens: trainer.config["max_cot_tokens"] = args.max_cot_tokens
    if args.learning_rate: trainer.config["learning_rate"] = args.learning_rate
    if args.episodes: trainer.config["episodes"] = args.episodes
    if args.stockfish_threads: trainer.config["stockfish_threads"] = args.stockfish_threads
    
    trainer.train()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
