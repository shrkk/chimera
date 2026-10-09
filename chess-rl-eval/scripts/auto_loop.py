#!/usr/bin/env python3
"""
scripts/auto_loop.py
====================
Automated Iterative Training, Evaluation, and Adaptive Hyperparameter Loop.

Workflow per round:
1. Runs GRPO training for N episodes.
2. Automatically benchmarks the new checkpoint on Tier 2 Lichess puzzles on GPU (~30s).
3. Reads results/puzzles_eval.json and logs Pass@1 progress.
4. Adaptive parameter tuning:
   - If Pass@1 improves: marks as current best checkpoint, continues momentum.
   - If Pass@1 stagnates: automatically decays learning rate (0.7x) or adjusts exploration.
5. Repeats for K rounds or until the target Pass@1 gate (≥65%) is met.
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoLoop")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_config(config_path: pathlib.Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: pathlib.Path) -> None:
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def run_command(cmd: list[str], description: str) -> bool:
    logger.info("Executing: %s", description)
    logger.info("Command: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return res.returncode == 0


def get_latest_checkpoint(checkpoints_dir: pathlib.Path) -> pathlib.Path | None:
    if not checkpoints_dir.exists():
        return None
    chkpts = sorted(
        checkpoints_dir.glob("grpo_chess_step_*"),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0
    )
    return chkpts[-1] if chkpts else None


def main():
    parser = argparse.ArgumentParser(description="Chimera Auto-Tuning Loop")
    parser.add_argument("--rounds", type=int, default=5, help="Number of train-eval rounds")
    parser.add_argument("--episodes-per-round", type=int, default=500, help="Episodes per round")
    parser.add_argument("--eval-limit", type=int, default=200, help="Puzzles per evaluation")
    parser.add_argument("--target-pass", type=float, default=0.65, help="Target Pass@1 for 1100-1400 band")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_file = results_dir / "loop_history.json"

    history = []
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    best_score = max([h.get("pass_at_1_target_band", 0.0) for h in history], default=0.0)
    logger.info("Starting Auto-Tuning Loop for %d rounds (Current Best: %.1f%%)", args.rounds, best_score * 100)

    for round_num in range(1, args.rounds + 1):
        logger.info("\n" + "=" * 60)
        logger.info("  ROUND %d / %d", round_num, args.rounds)
        logger.info("=" * 60)

        cfg = load_config(config_path)
        current_lr = float(cfg.get("grpo", {}).get("learning_rate", 5e-6))
        logger.info("Current Learning Rate: %e", current_lr)

        # ── Step 1: Train for N episodes ─────────────────────────────────────
        train_cmd = [
            sys.executable,
            "src/models/grpo_trainer.py",
            "--config", str(config_path),
            "--episodes", str(args.episodes_per_round),
        ]
        train_ok = run_command(train_cmd, f"Training Round {round_num} ({args.episodes_per_round} episodes)")
        if not train_ok:
            logger.error("Training failed in Round %d — stopping loop.", round_num)
            break

        # Locate newly saved checkpoint
        chkpts_dir = PROJECT_ROOT / cfg.get("grpo", {}).get("checkpoint_dir", "./checkpoints")
        latest_chkpt = get_latest_checkpoint(chkpts_dir)
        if not latest_chkpt:
            logger.error("No checkpoint found after training — stopping loop.")
            break
        logger.info("Latest checkpoint: %s", latest_chkpt)

        # ── Step 2: GPU Evaluation on Tactical Puzzles ───────────────────────
        eval_output = results_dir / f"round_{round_num}_puzzles_eval.json"
        eval_cmd = [
            sys.executable,
            "src/evals/run_puzzles.py",
            "--model-path", str(latest_chkpt),
            "--limit", str(args.eval_limit),
            "--device", "auto",
            "--output", str(eval_output),
        ]
        run_command(eval_cmd, f"Evaluating Checkpoint {latest_chkpt.name} on {args.eval_limit} Puzzles")

        # ── Step 3: Parse Benchmark Metrics ──────────────────────────────────
        target_band_pass = 0.0
        overall_pass = 0.0
        if eval_output.exists():
            try:
                with open(eval_output, "r") as f:
                    eval_data = json.load(f)
                bands = eval_data.get("bands", {})
                target_band_pass = bands.get("1100_1400", {}).get("pass_at_1", 0.0)

                total_corr = sum(b.get("correct", 0) for b in bands.values())
                total_puzz = sum(b.get("total", 0) for b in bands.values())
                overall_pass = total_corr / total_puzz if total_puzz > 0 else 0.0
            except Exception as e:
                logger.warning("Failed to parse evaluation output: %s", e)

        logger.info(
            "Round %d Results — Target Band (1100-1400): %.1f%% — Overall: %.1f%%",
            round_num,
            target_band_pass * 100,
            overall_pass * 100,
        )

        round_record = {
            "round": round_num,
            "checkpoint": str(latest_chkpt),
            "learning_rate": current_lr,
            "pass_at_1_target_band": target_band_pass,
            "overall_pass": overall_pass,
        }
        history.append(round_record)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        # ── Step 4: Adaptive Parameter Tuning ────────────────────────────────
        if target_band_pass >= args.target_pass:
            logger.info("🎯 Target gate achieved! (%.1f%% >= %.1f%%) Loop complete!", target_band_pass * 100, args.target_pass * 100)
            break
        elif target_band_pass > best_score:
            logger.info("✨ New best model achieved (%.1f%% > %.1f%%)! Maintaining learning momentum.", target_band_pass * 100, best_score * 100)
            best_score = target_band_pass
        else:
            # Performance stagnated: decay learning rate by 0.7x for finer convergence
            new_lr = max(current_lr * 0.7, 5e-7)
            logger.info("⚖ Performance plateaued. Adapting learning rate: %e -> %e", current_lr, new_lr)
            cfg["grpo"]["learning_rate"] = new_lr
            save_config(cfg, config_path)

    logger.info("\nAuto-Tuning Loop finished. Summary saved to %s", history_file)


if __name__ == "__main__":
    main()
